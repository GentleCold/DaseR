# SPDX-License-Identifier: Apache-2.0
"""HTTP load generation helpers for benchmark services."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import statistics
import time
from typing import Any

import httpx

from benchmarks.utils.constants import BLOCK_TOKENS
from benchmarks.utils.datasets import BenchmarkSample
from benchmarks.utils.metrics import (
    compute_metric_delta,
    extract_lmcache_status_metrics,
    extract_prometheus_counters,
    first_available_hit_ratio,
    hit_ratio_from_metrics,
)
from benchmarks.utils.prompts import build_prompt_payloads
from benchmarks.utils.servers import LMCACHE_HTTP_PORT, BenchmarkManifest

_LMCACHE_QUIESCENCE_TIMEOUT_SECONDS = 600.0
_DASER_DRAIN_TIMEOUT_SECONDS = 360.0


@dataclass
class RequestResult:
    """Result of one benchmark HTTP request."""

    sample_id: int
    dataset: str
    generated_text: str
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    cache_hits: int = 0
    cache_chunks_total: int = 0
    queue_ms: float = 0.0


@dataclass
class PhaseResult:
    """Results and metric deltas for one load phase.

    Args:
        requests: Per-request benchmark results.
        metrics: Backend and vLLM metric deltas for this phase.
        elapsed_ms: Wall-clock load phase duration.

    Thread-safety:
        Immutable by convention after construction; safe to pass between
        asyncio tasks after all request tasks have completed.
    """

    requests: list[RequestResult]
    metrics: dict[str, Any]
    elapsed_ms: float


async def run_vllm_phase(
    manifest: BenchmarkManifest,
    samples: list[BenchmarkSample],
    tokenizer: Any,
    max_inflight: int,
    gen_params: dict[str, Any],
    timeout: float,
    chunk_aligned_prompts: bool = False,
) -> PhaseResult:
    """Send full prompts to vLLM completions.

    Args:
        manifest: Service manifest with a vLLM endpoint.
        samples: Benchmark samples.
        tokenizer: Tokenizer for prompt construction.
        max_inflight: Maximum concurrent requests.
        gen_params: Generation parameters.
        timeout: Per-request timeout.
        chunk_aligned_prompts: use DaseR chunk-mode padded token prompts.

    Returns:
        Request results aligned with samples.
    """
    before_metrics = await collect_phase_metrics(manifest)
    requests, elapsed_ms = await _run_vllm_phase_requests(
        manifest,
        samples,
        tokenizer,
        max_inflight,
        gen_params,
        timeout,
        chunk_aligned_prompts=chunk_aligned_prompts,
        block_tokens=manifest.block_size,
    )
    return PhaseResult(
        requests=requests,
        metrics=await collect_phase_metrics(manifest, before_metrics),
        elapsed_ms=elapsed_ms,
    )


async def run_daser_chunk(
    manifest: BenchmarkManifest,
    samples: list[BenchmarkSample],
    max_inflight: int,
    gen_params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Run DaseR chunk cold document upload then warm inference.

    Args:
        manifest: DaseR service manifest.
        samples: Benchmark samples.
        max_inflight: Maximum concurrent requests.
        gen_params: Generation parameters.
        timeout: Per-request timeout.

    Returns:
        Dict with cold upload metadata and warm request results.
    """
    daser_url = manifest.endpoints["daser"].url
    unique_contexts: dict[str, dict[str, Any]] = {}
    for sample in samples:
        unique_contexts.setdefault(sample.context, {})
    sem = asyncio.Semaphore(max_inflight)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        t0 = time.perf_counter()
        upload_tasks = [
            _daser_upload_doc(client, daser_url, f"doc_{i}", context, sem)
            for i, context in enumerate(unique_contexts)
        ]
        upload_results = await asyncio.gather(*upload_tasks)
        upload_ms = (time.perf_counter() - t0) * 1000
        for context, result in zip(unique_contexts, upload_results, strict=False):
            unique_contexts[context] = result

        before_metrics = await collect_phase_metrics(manifest)
        infer_t0 = time.perf_counter()
        infer_tasks = [
            _daser_infer(
                client,
                daser_url,
                sample,
                [unique_contexts[sample.context]["doc_id"]],
                gen_params,
                sem,
                timeout,
            )
            for sample in samples
        ]
        infer_results = list(await asyncio.gather(*infer_tasks))
        infer_elapsed_ms = (time.perf_counter() - infer_t0) * 1000
        warm_results = PhaseResult(
            requests=infer_results,
            metrics=await collect_phase_metrics(manifest, before_metrics),
            elapsed_ms=infer_elapsed_ms,
        )
    return {
        "cold": {
            "uploaded_documents": len(upload_results),
            "upload_ms": upload_ms,
        },
        "warm": warm_results,
    }


async def run_daser_prefix(
    manifest: BenchmarkManifest,
    samples: list[BenchmarkSample],
    tokenizer: Any,
    max_inflight: int,
    gen_params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Run DaseR prefix cold and warm full-prompt phases."""
    before_cold = await collect_phase_metrics(manifest)
    cold, cold_elapsed_ms = await _run_vllm_phase_requests(
        manifest,
        samples,
        tokenizer,
        max_inflight,
        gen_params,
        timeout,
        chunk_aligned_prompts=True,
        block_tokens=manifest.block_size,
    )
    cold_phase = PhaseResult(
        requests=cold,
        metrics=await collect_phase_metrics(manifest, before_cold),
        elapsed_ms=cold_elapsed_ms,
    )
    await _wait_daser_drained(manifest)
    before_warm = await collect_phase_metrics(manifest)
    warm, warm_elapsed_ms = await _run_vllm_phase_requests(
        manifest,
        samples,
        tokenizer,
        max_inflight,
        gen_params,
        timeout,
        chunk_aligned_prompts=True,
        block_tokens=manifest.block_size,
    )
    warm_phase = PhaseResult(
        requests=warm,
        metrics=await collect_phase_metrics(manifest, before_warm),
        elapsed_ms=warm_elapsed_ms,
    )
    await _wait_daser_drained(manifest)
    return {"cold": cold_phase, "warm": warm_phase}


async def run_lmcache(
    manifest: BenchmarkManifest,
    samples: list[BenchmarkSample],
    tokenizer: Any,
    max_inflight: int,
    gen_params: dict[str, Any],
    timeout: float,
    settle_seconds: float = 0.0,
    chunk_aligned_prompts: bool = False,
) -> dict[str, Any]:
    """Run LMCache cold and warm full-prompt phases."""
    prompts = build_prompt_payloads(
        tokenizer,
        samples,
        chunk_aligned=chunk_aligned_prompts,
        block_tokens=manifest.block_size,
    )
    before_cold = await collect_phase_metrics(manifest)
    cold, cold_elapsed_ms = await _run_vllm_phase_requests(
        manifest,
        samples,
        tokenizer,
        max_inflight,
        gen_params,
        timeout,
        chunk_aligned_prompts=chunk_aligned_prompts,
        prompts=prompts,
    )
    cold_phase = PhaseResult(
        requests=cold,
        metrics=await collect_phase_metrics(manifest, before_cold),
        elapsed_ms=cold_elapsed_ms,
    )
    await _wait_lmcache_quiescent(manifest, settle_seconds)
    before_warm = await collect_phase_metrics(manifest)
    warm, warm_elapsed_ms = await _run_vllm_phase_requests(
        manifest,
        samples,
        tokenizer,
        max_inflight,
        gen_params,
        timeout,
        chunk_aligned_prompts=chunk_aligned_prompts,
        prompts=prompts,
    )
    warm_phase = PhaseResult(
        requests=warm,
        metrics=await collect_phase_metrics(manifest, before_warm),
        elapsed_ms=warm_elapsed_ms,
    )
    await _wait_daser_drained(manifest)
    return {"cold": cold_phase, "warm": warm_phase}


async def _run_vllm_phase_requests(
    manifest: BenchmarkManifest,
    samples: list[BenchmarkSample],
    tokenizer: Any,
    max_inflight: int,
    gen_params: dict[str, Any],
    timeout: float,
    chunk_aligned_prompts: bool = False,
    prompts: list[str | list[int]] | None = None,
    block_tokens: int = BLOCK_TOKENS,
) -> tuple[list[RequestResult], float]:
    if prompts is None:
        prompts = build_prompt_payloads(
            tokenizer,
            samples,
            chunk_aligned=chunk_aligned_prompts,
            block_tokens=block_tokens,
        )
    sem = asyncio.Semaphore(max_inflight)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        phase_t0 = time.perf_counter()
        tasks = [
            vllm_completion_stream(
                client,
                manifest.endpoints["vllm"].url,
                sample,
                prompt,
                gen_params,
                sem,
                timeout,
            )
            for sample, prompt in zip(samples, prompts, strict=False)
        ]
        results = list(await asyncio.gather(*tasks))
        elapsed_ms = (time.perf_counter() - phase_t0) * 1000
        return results, elapsed_ms


async def collect_phase_metrics(
    manifest: BenchmarkManifest,
    before_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect backend/vLLM metric snapshot or delta."""
    snapshot = await _collect_metric_snapshot(manifest)
    if before_metrics is None:
        return snapshot
    delta = {
        "vllm_prometheus": compute_metric_delta(
            before_metrics.get("vllm_prometheus", {}),
            snapshot.get("vllm_prometheus", {}),
        ),
        "backend_prometheus": compute_metric_delta(
            before_metrics.get("backend_prometheus", {}),
            snapshot.get("backend_prometheus", {}),
        ),
        "backend_status": compute_metric_delta(
            before_metrics.get("backend_status", {}),
            snapshot.get("backend_status", {}),
        ),
    }
    delta["hit_ratios"] = _metric_hit_ratios(delta)
    return delta


async def _collect_metric_snapshot(manifest: BenchmarkManifest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        vllm_prom = await _get_prometheus(client, manifest.endpoints["vllm"].url)
        backend_prom: dict[str, float] = {}
        backend_status: dict[str, float] = {}
        if manifest.backend == "lmcache":
            status_url = f"http://127.0.0.1:{LMCACHE_HTTP_PORT}"
            metrics_url = lmcache_metrics_url(manifest)
            backend_prom = await _get_prometheus(client, metrics_url)
            status = await _get_json(client, f"{status_url}/status")
            backend_status = extract_lmcache_status_metrics(status or {})
        elif manifest.backend == "daser" and "daser" in manifest.endpoints:
            backend_url = manifest.endpoints["daser"].url
            backend_prom = await _get_prometheus(client, backend_url)
    return {
        "vllm_prometheus": vllm_prom,
        "backend_prometheus": backend_prom,
        "backend_status": backend_status,
    }


def lmcache_metrics_url(manifest: BenchmarkManifest) -> str:
    """Return the LMCache MP HTTP endpoint that exposes metrics.

    Args:
        manifest: Benchmark service manifest.

    Returns:
        Base URL for LMCache metrics collection.

    Thread-safety:
        Pure function.
    """
    del manifest
    return f"http://127.0.0.1:{LMCACHE_HTTP_PORT}"


async def _get_prometheus(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    try:
        response = await client.get(f"{base_url}/metrics")
        response.raise_for_status()
    except Exception:
        return {}
    return extract_prometheus_counters(response.text)


async def _get_json(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _metric_hit_ratios(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    vllm = metrics.get("vllm_prometheus", {})
    backend_prom = metrics.get("backend_prometheus", {})
    backend_status = metrics.get("backend_status", {})
    return {
        "vllm_external_prefix": hit_ratio_from_metrics(
            vllm,
            hits_key="vllm:external_prefix_cache_hits_total",
            queries_key="vllm:external_prefix_cache_queries_total",
        ),
        "lmcache_prometheus_retrieve": hit_ratio_from_metrics(
            backend_prom,
            hits_key="lmcache_interval_hit_tokens_total",
            queries_key="lmcache_interval_requested_tokens_total",
        ),
        "lmcache_prometheus_lookup": first_available_hit_ratio(
            backend_prom,
            (
                (
                    "lmcache_mp_lookup_hit_tokens_total",
                    "lmcache_mp_lookup_requested_tokens_total",
                ),
                (
                    "lmcache_mp_lookup_hit_total",
                    "lmcache_mp_lookup_requested_total",
                ),
                (
                    "lmcache_num_hit_tokens_total",
                    "lmcache_num_requested_tokens_total",
                ),
            ),
        ),
        "lmcache_status_prefetch": hit_ratio_from_metrics(
            backend_status,
            hits_key="lmcache_prefetch_hit_tokens",
            queries_key="lmcache_prefetch_requested_tokens",
        ),
        "daser_prometheus_tokens": hit_ratio_from_metrics(
            backend_prom,
            hits_key="daser_cache_matched_tokens_total",
            queries_key="daser_cache_requested_tokens_total",
        ),
        "daser_external_prefix": hit_ratio_from_metrics(
            backend_prom,
            hits_key="daser_external_prefix_cache_hits_total",
            queries_key="daser_external_prefix_cache_queries_total",
        ),
        "daser_prometheus_requests": hit_ratio_from_metrics(
            backend_prom,
            hits_key='daser_cache_lookup_total{result="hit"}',
            queries_key="daser_cache_lookup_total",
        ),
    }


def backend_server_hit_rate(hit_ratios: dict[str, Any]) -> float | None:
    """Return the backend token-level cache hit ratio used for comparison.

    Args:
        hit_ratios: Named hit-ratio candidates from phase metric deltas.

    Returns:
        Token-level backend cache hit ratio when available.

    Thread-safety:
        Pure helper.
    """
    for key in (
        "daser_prometheus_tokens",
        "lmcache_prometheus_lookup",
        "lmcache_prometheus_retrieve",
        "lmcache_status_prefetch",
    ):
        ratio = hit_ratios.get(key)
        if ratio is not None:
            return float(ratio)
    return None


async def _wait_lmcache_quiescent(
    manifest: BenchmarkManifest, settle_seconds: float
) -> None:
    del manifest
    timeout_seconds = max(_LMCACHE_QUIESCENCE_TIMEOUT_SECONDS, float(settle_seconds))
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"LMCache did not become quiescent within {timeout_seconds:.1f}s"
                )
            status = await _get_json(
                client, f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/status"
            )
            if _lmcache_is_quiescent(status or {}):
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
            await asyncio.sleep(1.0)


def _lmcache_is_quiescent(status: dict[str, Any]) -> bool:
    storage = status.get("storage_manager", {})
    if not isinstance(storage, dict):
        return False
    store = storage.get("store_controller", {})
    prefetch = storage.get("prefetch_controller", {})
    if not isinstance(store, dict) or not isinstance(prefetch, dict):
        return False
    zero_fields = (
        (store, "pending_keys_count"),
        (store, "in_flight_task_count"),
        (prefetch, "submission_queue_size"),
        (prefetch, "pending_queue_size"),
        (prefetch, "in_flight_request_count"),
        (prefetch, "lookup_phase_count"),
        (prefetch, "load_phase_count"),
    )
    return all(int(mapping.get(key, 0)) == 0 for mapping, key in zero_fields)


async def _wait_daser_drained(manifest: BenchmarkManifest) -> None:
    daser = manifest.endpoints.get("daser")
    if daser is None:
        return
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_DASER_DRAIN_TIMEOUT_SECONDS)
    ) as client:
        response = await client.post(f"{daser.url}/drain")
        response.raise_for_status()


def summarise_results(results: list[RequestResult]) -> dict[str, Any]:
    """Summarise request results.

    Args:
        results: Request results.

    Returns:
        Aggregate counters and latency metrics.
    """
    ok = [result for result in results if result.error is None]
    ttfts = [result.ttft_ms for result in ok]
    latencies = [result.latency_ms for result in ok]
    queues = [result.queue_ms for result in ok]
    hits = sum(result.cache_hits for result in ok)
    total_chunks = sum(result.cache_chunks_total for result in ok)
    return {
        "num_requests": len(results),
        "num_errors": len(results) - len(ok),
        "ttft_ms_mean": statistics.mean(ttfts) if ttfts else 0.0,
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "queue_ms_mean": statistics.mean(queues) if queues else 0.0,
        "prompt_tokens_total": sum(result.prompt_tokens for result in ok),
        "completion_tokens_total": sum(result.completion_tokens for result in ok),
        "cache_hits": hits,
        "cache_chunks_total": total_chunks,
        "cache_hit_rate": hits / total_chunks if total_chunks else 0.0,
    }


async def vllm_completion_stream(
    client: httpx.AsyncClient,
    vllm_url: str,
    sample: BenchmarkSample,
    prompt: str | list[int],
    gen_params: dict[str, Any],
    sem: asyncio.Semaphore,
    timeout: float,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": "",
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(gen_params)
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    queued_at = time.perf_counter()
    first_token_at: float | None = None
    queue_ms = 0.0
    try:
        async with sem:
            t0 = time.perf_counter()
            queue_ms = (t0 - queued_at) * 1000
            async with client.stream(
                "POST",
                f"{vllm_url}/v1/completions",
                json=payload,
                timeout=httpx.Timeout(timeout),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    chunk = json.loads(data)
                    if chunk.get("usage") is not None:
                        usage = dict(chunk["usage"])
                    for choice in chunk.get("choices", []):
                        fragment = str(choice.get("text", ""))
                        if not fragment:
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(fragment)
    except Exception as exc:
        return RequestResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text="",
            ttft_ms=0.0,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
            queue_ms=queue_ms,
        )
    wall_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = ((first_token_at or time.perf_counter()) - t0) * 1000
    if not usage:
        return RequestResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text="".join(text_parts),
            ttft_ms=ttft_ms,
            latency_ms=wall_ms,
            prompt_tokens=0,
            completion_tokens=0,
            error="stream completed without usage",
            queue_ms=queue_ms,
        )
    return RequestResult(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        generated_text="".join(text_parts),
        ttft_ms=ttft_ms,
        latency_ms=wall_ms,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        queue_ms=queue_ms,
    )


async def _daser_upload_doc(
    client: httpx.AsyncClient,
    daser_url: str,
    title: str,
    text: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        response = await client.post(
            f"{daser_url}/documents",
            json={"title": title, "text": text},
        )
        response.raise_for_status()
        return dict(response.json())


async def _daser_infer(
    client: httpx.AsyncClient,
    daser_url: str,
    sample: BenchmarkSample,
    doc_ids: list[str],
    gen_params: dict[str, Any],
    sem: asyncio.Semaphore,
    timeout: float,
) -> RequestResult:
    body = {
        "doc_ids": doc_ids,
        "task": sample.question,
        "use_kv_cache": True,
        "trace_cache": True,
        "gen_params": gen_params,
    }
    queued_at = time.perf_counter()
    queue_ms = 0.0
    try:
        async with sem:
            t0 = time.perf_counter()
            queue_ms = (t0 - queued_at) * 1000
            response = await client.post(
                f"{daser_url}/infer",
                json=body,
                timeout=httpx.Timeout(timeout),
            )
            response.raise_for_status()
        wall_ms = (time.perf_counter() - t0) * 1000
        payload = response.json()
        cache_hits = payload.get("cache_hits", [])
        return RequestResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text=str(payload.get("text", "")),
            ttft_ms=float(payload.get("ttft_ms", 0.0)),
            latency_ms=float(payload.get("latency_ms", wall_ms)),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            cache_hits=sum(1 for hit in cache_hits if hit.get("chunk_key")),
            cache_chunks_total=len(cache_hits),
            queue_ms=queue_ms,
        )
    except Exception as exc:
        return RequestResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            generated_text="",
            ttft_ms=0.0,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
            queue_ms=queue_ms,
        )
