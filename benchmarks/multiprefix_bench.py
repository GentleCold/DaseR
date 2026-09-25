# SPDX-License-Identifier: Apache-2.0
"""Run a deterministic multi-prefix completion workload against vLLM.

The built-in ``vllm bench serve --random-prefix-len`` workload uses one shared
prefix.  This helper keeps the service and connector path unchanged while
constructing model-native token-ID prompts for several independent prefix
families.  A generated request manifest is reusable across raw and packed
arms, which makes cache coverage and endpoint latency comparable.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import httpx

from benchmarks.utils.datasets import BenchmarkSample
from benchmarks.utils.loadgen import (
    _DASER_DRAIN_TIMEOUT_SECONDS,
    collect_phase_metrics,
    summarise_results,
    vllm_completion_stream,
)
from benchmarks.utils.servers import BenchmarkManifest

DEFAULT_FAMILIES = 40
DEFAULT_REQUESTS_PER_FAMILY = 5
DEFAULT_PREFIX_TOKENS = 7296
DEFAULT_SUFFIX_TOKENS = 896
DEFAULT_BLOCK_TOKENS = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse multi-prefix benchmark arguments.

    Args:
        argv: Optional argument vector; ``None`` uses ``sys.argv``.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=DEFAULT_FAMILIES)
    parser.add_argument(
        "--requests-per-family",
        type=int,
        default=DEFAULT_REQUESTS_PER_FAMILY,
    )
    parser.add_argument("--prefix-tokens", type=int, default=DEFAULT_PREFIX_TOKENS)
    parser.add_argument("--suffix-tokens", type=int, default=DEFAULT_SUFFIX_TOKENS)
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _eligible_tokens(tokenizer: Any) -> tuple[int, ...]:
    special = {int(token_id) for token_id in tokenizer.all_special_ids}
    tokens = tuple(
        token_id for token_id in range(len(tokenizer)) if token_id not in special
    )
    if len(tokens) < 2:
        raise ValueError("tokenizer must expose at least two ordinary token IDs")
    return tokens


def _token_at(tokens: tuple[int, ...], state: int) -> int:
    """Return one deterministic ordinary token from a 64-bit state."""
    return tokens[state % len(tokens)]


def build_request_manifest(
    *,
    tokenizer: Any,
    families: int,
    requests_per_family: int,
    prefix_tokens: int,
    suffix_tokens: int,
    block_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Build deterministic block-aligned prompts for independent families.

    Args:
        tokenizer: Model tokenizer used only to identify valid non-special IDs.
        families: Number of independent prefix families.
        requests_per_family: Requests sharing one family prefix.
        prefix_tokens: Tokens in each family prefix.
        suffix_tokens: Tokens unique to each request.
        block_tokens: KV block size in tokens.
        seed: Seed for deterministic token generation.

    Returns:
        JSON-serializable manifest with request prompts and expected hit math.

    Raises:
        ValueError: If dimensions are invalid or not block aligned.
    """
    values = (families, requests_per_family, prefix_tokens, suffix_tokens, block_tokens)
    if any(value <= 0 for value in values):
        raise ValueError(
            "families, requests, token lengths, and block size must be positive"
        )
    if prefix_tokens % block_tokens or suffix_tokens % block_tokens:
        raise ValueError("prefix and suffix lengths must be block aligned")
    tokens = _eligible_tokens(tokenizer)
    requests: list[dict[str, Any]] = []
    for family_id in range(families):
        family_state = (seed + 1) * 0x9E3779B185EBCA87 + family_id * 0xD1B54A32D192ED03
        prefix = [
            _token_at(tokens, family_state + position * 0x9E3779B185EBCA87)
            for position in range(prefix_tokens)
        ]
        for request_in_family in range(requests_per_family):
            request_id = family_id * requests_per_family + request_in_family
            suffix_state = family_state ^ ((request_in_family + 1) * 0x94D049BB133111EB)
            suffix = [
                _token_at(tokens, suffix_state + position * 0xBF58476D1CE4E5B9)
                for position in range(suffix_tokens)
            ]
            requests.append(
                {
                    "sample_id": request_id,
                    "family_id": family_id,
                    "request_in_family": request_in_family,
                    "prompt": prefix + suffix,
                }
            )
    total_requests = families * requests_per_family
    total_prompt_tokens = total_requests * (prefix_tokens + suffix_tokens)
    expected_hit_tokens = families * max(0, requests_per_family - 1) * prefix_tokens
    return {
        "version": 1,
        "seed": seed,
        "families": families,
        "requests_per_family": requests_per_family,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "block_tokens": block_tokens,
        "total_requests": total_requests,
        "total_prompt_tokens": total_prompt_tokens,
        "expected_hit_tokens": expected_hit_tokens,
        "expected_hit_rate": expected_hit_tokens / total_prompt_tokens,
        "requests": requests,
    }


def _percentile(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile for non-empty values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((percentile / 100.0) * len(ordered))))
    return ordered[index]


def _request_samples(
    manifest: dict[str, Any],
) -> tuple[list[BenchmarkSample], list[list[int]]]:
    """Convert a saved request manifest to benchmark sample/prompt pairs."""
    samples: list[BenchmarkSample] = []
    prompts: list[list[int]] = []
    for request in manifest["requests"]:
        sample_id = int(request["sample_id"])
        family_id = int(request["family_id"])
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                dataset=f"multiprefix-family-{family_id}",
                context="",
                question=f"family-{family_id}-request-{request['request_in_family']}",
                answers=[],
            )
        )
        prompts.append([int(token_id) for token_id in request["prompt"]])
    return samples, prompts


def _summary_with_percentiles(results: list[Any]) -> dict[str, Any]:
    """Add robust TTFT percentiles to the shared request summary."""
    summary = summarise_results(results)
    ttfts = [result.ttft_ms for result in results if result.error is None]
    summary.update(
        {
            "ttft_ms_p50": _percentile(ttfts, 50.0),
            "ttft_ms_p95": _percentile(ttfts, 95.0),
            "ttft_ms_p99": _percentile(ttfts, 99.0),
        }
    )
    return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one complete multi-prefix phase and persist request diagnostics."""
    service = BenchmarkManifest.read(args.manifest)
    request_path = args.request_manifest
    request_path.parent.mkdir(parents=True, exist_ok=True)
    if request_path.exists():
        request_manifest = json.loads(request_path.read_text(encoding="utf-8"))
    else:
        model_path = args.model or Path(service.model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        request_manifest = build_request_manifest(
            tokenizer=tokenizer,
            families=args.families,
            requests_per_family=args.requests_per_family,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            block_tokens=args.block_tokens,
            seed=args.seed,
        )
        request_path.write_text(
            json.dumps(request_manifest, separators=(",", ":")), encoding="utf-8"
        )
    samples, prompts = _request_samples(request_manifest)
    if len(samples) != int(request_manifest["total_requests"]):
        raise ValueError("request manifest count does not match total_requests")

    before_metrics = await collect_phase_metrics(service)
    gen_params = {"max_tokens": args.output_len, "temperature": 0.0, "top_p": 1.0}
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        seed_pairs = [
            (sample, prompt)
            for sample, prompt in zip(samples, prompts, strict=True)
            if sample.sample_id % int(request_manifest["requests_per_family"]) == 0
        ]
        warm_pairs = [
            (sample, prompt)
            for sample, prompt in zip(samples, prompts, strict=True)
            if sample.sample_id % int(request_manifest["requests_per_family"]) != 0
        ]
        seed_results = await _send_requests(
            client,
            service,
            seed_pairs,
            gen_params,
            args.max_inflight,
            args.timeout,
        )
        await _drain_before_warm(client, service)
        warm_results = await _send_requests(
            client,
            service,
            warm_pairs,
            gen_params,
            args.max_inflight,
            args.timeout,
        )
        results = seed_results + warm_results
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics = await collect_phase_metrics(service, before_metrics)
    summary = _summary_with_percentiles(results)
    summary.update(
        {
            "wall_time_ms": elapsed_ms,
            "family_count": request_manifest["families"],
            "requests_per_family": request_manifest["requests_per_family"],
            "expected_hit_rate": request_manifest["expected_hit_rate"],
            "request_manifest": str(request_path),
        }
    )
    payload = {
        "manifest": asdict(service),
        "request_contract": {
            key: value for key, value in request_manifest.items() if key != "requests"
        },
        "summary": summary,
        "metrics": metrics,
        "requests": [
            {
                "sample_id": result.sample_id,
                "generated_text": result.generated_text,
                "error": result.error,
                "ttft_ms": result.ttft_ms,
                "latency_ms": result.latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
            for result in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if service.backend == "daser":
        daser = service.endpoints.get("daser")
        if daser is not None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_DASER_DRAIN_TIMEOUT_SECONDS)
            ) as client:
                response = await client.post(f"{daser.url}/drain")
                response.raise_for_status()
    print(json.dumps(summary, indent=2))
    return payload


async def _send_requests(
    client: httpx.AsyncClient,
    service: BenchmarkManifest,
    pairs: list[tuple[BenchmarkSample, list[int]]],
    gen_params: dict[str, Any],
    max_inflight: int,
    timeout: float,
) -> list[Any]:
    """Send one request wave with a bounded client-side concurrency."""
    semaphore = asyncio.Semaphore(max_inflight)
    tasks = [
        vllm_completion_stream(
            client,
            service.endpoints["vllm"].url,
            sample,
            prompt,
            gen_params,
            semaphore,
            timeout,
        )
        for sample, prompt in pairs
    ]
    return list(await asyncio.gather(*tasks))


async def _drain_before_warm(
    client: httpx.AsyncClient,
    service: BenchmarkManifest,
) -> None:
    """Wait for DaseR seed stores before issuing family warm requests."""
    if service.backend != "daser":
        return
    daser = service.endpoints.get("daser")
    if daser is None:
        return
    response = await client.post(f"{daser.url}/drain")
    response.raise_for_status()


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark and exit non-zero when a request fails."""
    args = parse_args(argv)
    payload = asyncio.run(run(args))
    if int(payload["summary"]["num_errors"]) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
