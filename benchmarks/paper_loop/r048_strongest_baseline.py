# SPDX-License-Identifier: Apache-2.0

"""Run the R048 R-cap strongest-baseline characterization."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

from datasets import Dataset
import httpx
from vllm.tokenizers import get_tokenizer

from benchmarks.utils.loadgen import collect_phase_metrics
from benchmarks.utils.servers import (
    BenchmarkManifest,
    ServerManager,
    stop_from_pid_file,
)
from benchmarks.utils.sizing import parse_size_bytes
from daser.connector.ipc_client import IPCClientSync

GPU_KV_BYTES = 24 * 1024**3
BLOCK_SIZE = 128
DEFAULT_MAX_TOKENS = 128


def json_default(value: Any) -> Any:
    """Convert benchmark metadata containing bytes into JSON."""
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class MatchedServer(ServerManager):
    """Apply the common prefix-cache and GPU KV settings to every arm."""

    def vllm_command(self, kv_transfer_config: dict[str, Any] | None) -> list[str]:
        """Return a vLLM command with the matched prefix-cache settings."""
        command = super().vllm_command(kv_transfer_config)
        command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
        command.extend(["--kv-cache-memory-bytes", str(GPU_KV_BYTES)])
        return command

    def _daser_server_command(self) -> list[str]:
        """Return the DaseR command with format-independent features disabled."""
        command = super()._daser_server_command()
        command.extend(
            [
                "--no-bip-enabled",
                "--no-coalesce-load-misses",
                "--l1-accounting",
                "stored",
            ]
        )
        return command


class SimpleCPUOffloadServer(MatchedServer):
    """Start vLLM with its registered SimpleCPUOffloadConnector."""

    def offload_config(self) -> dict[str, Any]:
        """Return the explicit CPU-KV capacity connector configuration."""
        return {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": self.l1_size_bytes,
            },
        }

    async def start_vllm_only(self) -> None:
        """Start the vLLM process with CPU KV offload enabled."""
        await self._start_vllm("vllm_offload.log", self.offload_config())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the reproducible R048 benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arxiv-shard", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--gpu-id", default="2")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--tokens-per-document", type=int, default=125000)
    parser.add_argument("--l1-size", type=parse_size_bytes, default="64gib")
    parser.add_argument("--l2-size", type=parse_size_bytes, default="256gib")
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args(argv)


def prompts_from_arxiv(
    model: str,
    shard_path: str,
    count: int,
    tokens_per_document: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Build disjoint real-text token prompts from the local ArXiv shard."""
    tokenizer = get_tokenizer(model, trust_remote_code=True)
    shard = Dataset.from_file(shard_path)
    prompts: list[list[int]] = []
    provenance: list[list[str]] = []
    consumed = 0
    for document in range(count):
        tokens: list[int] = []
        articles: list[str] = []
        while len(tokens) < tokens_per_document:
            row = shard[consumed]
            consumed += 1
            text = str(row["text"] or "")
            if len(text) < 2000:
                continue
            tokens.extend(tokenizer.encode("\n\n" + text, add_special_tokens=False))
            articles.append(str(row["arxivid"]))
        prompts.append(tokens[:tokens_per_document])
        provenance.append(articles)
        print(
            f"prepared document {document + 1}/{count} articles={len(articles)}",
            flush=True,
        )
    metadata = {
        "source": shard_path,
        "model": model,
        "count": count,
        "tokens_per_document": tokens_per_document,
        "total_unique_prompt_tokens": count * tokens_per_document,
        "articles": provenance,
        "prompt_sha256": [
            hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            for tokens in prompts
        ],
    }
    return prompts, metadata


async def one_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model_name: str,
    prompt: list[int],
    document: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """Send one deterministic streaming completion and record timings."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "return_token_ids": True,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    token_ids: list[int] = []
    text_parts: list[str] = []
    first_token: float | None = None
    usage: dict[str, Any] = {}
    start = time.perf_counter()
    async with client.stream(
        "POST",
        f"{endpoint}/v1/completions",
        json=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids and first_token is None:
                    first_token = time.perf_counter()
                token_ids.extend(ids)
                text_parts.append(choice.get("text") or "")
    finish = time.perf_counter()
    if first_token is None or len(token_ids) != max_tokens:
        raise RuntimeError(
            f"document {document} received {len(token_ids)} tokens; usage={usage}"
        )
    return {
        "document": document,
        "ttft_ms": (first_token - start) * 1000,
        "latency_ms": (finish - start) * 1000,
        "tpot_ms": (finish - first_token) * 1000 / max(1, len(token_ids) - 1),
        "token_ids": token_ids,
        "text": "".join(text_parts),
        "usage": usage,
    }


async def phase(
    manifest: BenchmarkManifest,
    prompts: list[list[int]],
    model_name: str,
    repeats: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """Run sequential cold or warm rotations and collect backend metrics."""
    before_metrics = await collect_phase_metrics(manifest)
    requests: list[dict[str, Any]] = []
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        for repeat in range(repeats):
            for document, prompt in enumerate(prompts):
                requests.append(
                    await one_request(
                        client,
                        manifest.endpoints["vllm"].url,
                        model_name,
                        prompt,
                        document,
                        max_tokens,
                        timeout,
                    )
                )
                if (document + 1) % 4 == 0 or document + 1 == len(prompts):
                    print(
                        f"phase rotation {repeat + 1}/{repeats}: "
                        f"{document + 1}/{len(prompts)}",
                        flush=True,
                    )
    elapsed = time.perf_counter() - started
    ttft = [result["ttft_ms"] for result in requests]
    tpot = [result["tpot_ms"] for result in requests]
    latency = [result["latency_ms"] for result in requests]
    return {
        "elapsed_seconds": elapsed,
        "ttft_p50_ms": statistics.median(ttft),
        "ttft_p90_ms": sorted(ttft)[int(0.9 * (len(ttft) - 1))],
        "tpot_p50_ms": statistics.median(tpot),
        "output_tokens_per_second": len(requests) * max_tokens / elapsed,
        "request_timing_profile": {
            "ttft_p50_ms": statistics.median(ttft),
            "decode_tail_p50_ms": statistics.median(tpot) * max(1, max_tokens - 1),
            "end_to_end_latency_p50_ms": statistics.median(latency),
        },
        "metrics": await collect_phase_metrics(manifest, before_metrics),
        "requests": requests,
    }


async def settle(arm: str, manifest: BenchmarkManifest) -> None:
    """Wait for backend stores before starting the warm rotation."""
    if arm == "daser":
        async with httpx.AsyncClient(timeout=900.0) as client:
            response = await client.post(f"{manifest.endpoints['daser'].url}/drain")
            response.raise_for_status()
    elif arm == "lmcache":
        from benchmarks.utils.loadgen import _wait_lmcache_quiescent

        await _wait_lmcache_quiescent(manifest, 0.0)
    else:
        await asyncio.sleep(5.0)


def make_manager(
    arm: str,
    model: str,
    store_dir: Path,
    gpu_id: str,
    l1_size: int,
    l2_size: int,
    max_model_len: int,
) -> MatchedServer:
    """Create one isolated manager for a benchmark arm."""
    manager_type: type[MatchedServer] = (
        SimpleCPUOffloadServer if arm == "vllm-offload" else MatchedServer
    )
    return manager_type(
        run_id=f"r048-{arm}",
        backend="lmcache"
        if arm == "lmcache"
        else ("daser" if arm in {"daser-raw", "daser-compressed"} else "vllm"),
        model=model,
        store_dir=store_dir,
        gpu_id=gpu_id,
        gpu_util=0.85,
        max_num_seqs=1,
        l1_size_bytes=l1_size,
        l2_size_bytes=l2_size,
        max_model_len=max_model_len,
        block_size=BLOCK_SIZE,
        reuse_mode="prefix",
        transfer_mode="iouring",
        trust_remote_code=True,
        daser_prefetch_max_requests=0,
        storage_format=("compressed-online" if arm == "daser-compressed" else "raw"),
        skip_l2=False,
    )


async def run_arm(
    arm: str,
    prompts: list[list[int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one isolated arm and persist its cold/warm measurements."""
    store_dir = Path(args.scratch) / arm
    manager = make_manager(
        arm,
        args.model,
        store_dir,
        args.gpu_id,
        args.l1_size,
        args.l2_size,
        args.max_model_len,
    )
    print(f"starting {arm}", flush=True)
    try:
        manifest = await manager.start()
        runtime_config: dict[str, Any] | None = None
        if arm in {"daser-raw", "daser-compressed"}:
            ipc = IPCClientSync(str(manager.socket_path))
            try:
                runtime_config = ipc.get_runtime_config()
            finally:
                ipc.close()
        cold = await phase(
            manifest,
            prompts,
            Path(args.model).name,
            1,
            args.max_tokens,
            args.timeout,
        )
        await settle("daser" if arm.startswith("daser") else arm, manifest)
        warm = await phase(
            manifest,
            prompts,
            Path(args.model).name,
            2,
            args.max_tokens,
            args.timeout,
        )
        result = {
            "arm": arm,
            "manifest": asdict(manifest),
            "vllm_command": manager.vllm_command(
                manager.daser_kv_transfer_config()
                if arm.startswith("daser")
                else (
                    manager.lmcache_kv_transfer_config()
                    if arm == "lmcache"
                    else (manager.offload_config() if arm == "vllm-offload" else None)
                )
            ),
            "runtime_config": runtime_config,
            "cold": cold,
            "warm": warm,
            "cold_warm_exact": [
                warm_result["token_ids"]
                == cold["requests"][index % len(prompts)]["token_ids"]
                for index, warm_result in enumerate(warm["requests"])
            ],
        }
        output = Path(args.artifact) / f"{arm}.json"
        output.write_text(json.dumps(result, indent=2, default=json_default))
        print(f"finished {arm} warm_ttft_p50_ms={warm['ttft_p50_ms']}", flush=True)
        return result
    finally:
        stop_from_pid_file(store_dir / "pids.json")


def first_divergence(left: list[int], right: list[int]) -> int | None:
    """Return the first differing token position, if any."""
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=True)):
        if left_token != right_token:
            return index
    return len(left) if len(left) != len(right) else None


def compare_to_native(
    arm_result: dict[str, Any], native_result: dict[str, Any]
) -> dict[str, Any]:
    """Summarize relaxed end-to-end correctness against native APC."""
    arm_requests = arm_result["warm"]["requests"]
    native_requests = native_result["warm"]["requests"]
    comparisons = []
    for left, right in zip(arm_requests, native_requests, strict=True):
        comparisons.append(
            {
                "document": left["document"],
                "token_count_equal": len(left["token_ids"]) == len(right["token_ids"]),
                "token_id_equal": left["token_ids"] == right["token_ids"],
                "first_divergence": first_divergence(
                    left["token_ids"], right["token_ids"]
                ),
                "arm_text": left["text"],
                "native_text": right["text"],
            }
        )
    return {
        "token_id_equal_count": sum(item["token_id_equal"] for item in comparisons),
        "request_count": len(comparisons),
        "first_divergence_positions": [
            item["first_divergence"]
            for item in comparisons
            if item["first_divergence"] is not None
        ],
        "samples": comparisons[:2],
    }


async def main_async(args: argparse.Namespace) -> None:
    """Run the five matched arms and write a comparison summary."""
    artifact = Path(args.artifact)
    artifact.mkdir(parents=True, exist_ok=True)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    prompts, workload = prompts_from_arxiv(
        args.model,
        args.arxiv_shard,
        args.count,
        args.tokens_per_document,
    )
    (artifact / "workload.json").write_text(json.dumps(workload, indent=2))
    results: dict[str, dict[str, Any]] = {}
    for arm in (
        "daser-compressed",
        "daser-raw",
        "native-apc",
        "vllm-offload",
        "lmcache",
    ):
        results[arm] = await run_arm(arm, prompts, args)
    native = results["native-apc"]
    comparison = {
        arm: compare_to_native(result, native)
        for arm, result in results.items()
        if arm != "native-apc"
    }
    summary = {
        "conditions": {
            "model": args.model,
            "gpu_id": args.gpu_id,
            "gpu_kv_bytes": GPU_KV_BYTES,
            "l1_size_bytes": args.l1_size,
            "l2_size_bytes": args.l2_size,
            "count": args.count,
            "tokens_per_document": args.tokens_per_document,
            "max_tokens": args.max_tokens,
            "block_size": BLOCK_SIZE,
        },
        "warm_ttft_p50_ms": {
            arm: result["warm"]["ttft_p50_ms"] for arm, result in results.items()
        },
        "warm_ttft_p90_ms": {
            arm: result["warm"]["ttft_p90_ms"] for arm, result in results.items()
        },
        "warm_tpot_p50_ms": {
            arm: result["warm"]["tpot_p50_ms"] for arm, result in results.items()
        },
        "warm_output_tokens_per_second": {
            arm: result["warm"]["output_tokens_per_second"]
            for arm, result in results.items()
        },
        "comparison_to_native_apc": comparison,
        "arm_files": {arm: f"{arm}.json" for arm in results},
    }
    (artifact / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default)
    )
    print(json.dumps(summary, default=json_default), flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run the asynchronous R048 characterization."""
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
