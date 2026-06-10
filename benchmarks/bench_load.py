# SPDX-License-Identifier: Apache-2.0
"""Send benchmark load to services described by a manifest."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils.constants import (
    BLOCK_TOKENS,
    COMPARISON_IOURING_MEM,
    SLOT_SIZE,
)
from benchmarks.utils.datasets import (
    BenchmarkSample,
    ImdbDataset,
    LongBenchDataset,
    dedup_by_context,
    interleave_samples,
)
from benchmarks.utils.loadgen import (
    PhaseResult,
    RequestResult,
    run_daser_chunk,
    run_daser_prefix,
    run_lmcache,
    run_vllm_phase,
    summarise_results,
)
from benchmarks.utils.metrics import contains_accuracy, request_text_exact_match
from benchmarks.utils.prompts import (
    build_prompt_payloads,
    count_prompt_payload_tokens,
    filter_by_token_limit,
    workload_blocks,
)
from benchmarks.utils.servers import BenchmarkManifest
from benchmarks.utils.sizing import (
    BenchmarkCapacityLimits,
    derive_benchmark_sizing,
    derive_capacity_limits,
    format_capacity,
    parse_size_bytes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument(
        "--cache-reuse-mode", choices=("chunk", "prefix"), default="chunk"
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset", choices=("imdb", "longbench"), required=True)
    parser.add_argument("--imdb")
    parser.add_argument("--longbench-dir")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--no-dedup-context", action="store_true")
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--gen-max-tokens", type=int, default=128)
    parser.add_argument("--gen-temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--evict", action="store_true")
    parser.add_argument("--max-l1-size", type=parse_size_bytes, default=None)
    parser.add_argument("--max-l2-size", type=parse_size_bytes, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> None:
    """Load samples, send benchmark traffic, and write JSON results."""
    manifest = (
        None
        if args.prepare_only
        else BenchmarkManifest.read(_required(args.manifest, "--manifest"))
    )
    model = args.model if args.prepare_only else manifest.model
    store_dir = args.store_dir if args.prepare_only else manifest.store_dir
    if model is None:
        raise ValueError("--model is required with --prepare-only")
    if store_dir is None:
        raise ValueError("--store-dir is required with --prepare-only")
    samples = _load_samples(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    reuse_mode = args.cache_reuse_mode if args.prepare_only else manifest.reuse_mode
    chunk_aligned_prompts = reuse_mode == "chunk"
    prompts = build_prompt_payloads(
        tokenizer, samples, chunk_aligned=chunk_aligned_prompts
    )
    token_counts = count_prompt_payload_tokens(tokenizer, prompts)
    samples, prompts, token_counts = filter_by_token_limit(
        samples, prompts, token_counts, args.max_context_tokens
    )
    if not args.no_dedup_context and reuse_mode != "prefix":
        samples = dedup_by_context(samples)
        prompts = build_prompt_payloads(
            tokenizer, samples, chunk_aligned=chunk_aligned_prompts
        )
        token_counts = count_prompt_payload_tokens(tokenizer, prompts)
    samples = interleave_samples(samples)
    prompts = build_prompt_payloads(
        tokenizer, samples, chunk_aligned=chunk_aligned_prompts
    )
    token_counts = count_prompt_payload_tokens(tokenizer, prompts)
    total_blocks, max_prompt_blocks = workload_blocks(token_counts, BLOCK_TOKENS)
    capacity_limits = _capacity_limits(args, store_dir)
    sizing = derive_benchmark_sizing(
        total_blocks=total_blocks,
        max_prompt_blocks=max_prompt_blocks,
        slot_size=SLOT_SIZE,
        mode=COMPARISON_IOURING_MEM,
        evict=args.evict,
        capacity_limits=capacity_limits,
    )
    gen_params = {
        "max_tokens": args.gen_max_tokens,
        "temperature": args.gen_temperature,
    }
    answers_by_id = {
        sample.sample_id: sample.answers for sample in samples if sample.answers
    }

    common_config = {
        "dataset": args.dataset,
        "num_samples": len(samples),
        "max_inflight": args.max_inflight,
        "gen_params": gen_params,
        "total_prompt_tokens": sum(token_counts),
        "total_blocks": total_blocks,
        "max_prompt_blocks": max_prompt_blocks,
        "derived_l1_size_bytes": sizing.daser_l1_bytes,
        "derived_l1_size": format_capacity(sizing.daser_l1_bytes),
        "derived_l2_size_bytes": sizing.daser_l2_bytes,
        "derived_l2_size": format_capacity(sizing.daser_l2_bytes),
        "lmcache_l1_gb": sizing.lmcache_cpu_gb,
        "lmcache_l2_gb": sizing.lmcache_disk_gb,
        "capacity_capped": sizing.capacity_capped,
        "evict": args.evict,
        "planned_skip_l2": not args.evict,
    }
    if args.prepare_only:
        output = {"config": common_config}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(json.dumps(common_config, indent=2, ensure_ascii=False))
        print(f"prepare={args.out}")
        return

    assert manifest is not None

    if manifest.backend == "vllm":
        phase = await run_vllm_phase(
            manifest,
            samples,
            tokenizer,
            args.max_inflight,
            gen_params,
            args.timeout,
            chunk_aligned_prompts=chunk_aligned_prompts,
        )
        result: dict[str, Any] = {"baseline": _serialise_phase(phase, answers_by_id)}
    elif manifest.backend == "lmcache":
        phases = await run_lmcache(
            manifest,
            samples,
            tokenizer,
            args.max_inflight,
            gen_params,
            args.timeout,
            chunk_aligned_prompts=chunk_aligned_prompts,
        )
        result = {
            name: _serialise_phase(phase, answers_by_id)
            for name, phase in phases.items()
        }
        _add_phase_comparison(result)
    elif manifest.backend == "daser" and manifest.reuse_mode == "chunk":
        phases = await run_daser_chunk(
            manifest, samples, args.max_inflight, gen_params, args.timeout
        )
        result = {
            "cold": phases["cold"],
            "warm": _serialise_phase(phases["warm"], answers_by_id),
        }
    elif manifest.backend == "daser" and manifest.reuse_mode == "prefix":
        phases = await run_daser_prefix(
            manifest, samples, tokenizer, args.max_inflight, gen_params, args.timeout
        )
        result = {
            name: _serialise_phase(phase, answers_by_id)
            for name, phase in phases.items()
        }
        _add_phase_comparison(result)
    else:
        raise ValueError(
            "unsupported backend/reuse combination: "
            f"{manifest.backend}/{manifest.reuse_mode}"
        )

    output = {
        "manifest": asdict(manifest),
        "config": {
            **common_config,
            "manifest_l1_size_bytes": manifest.l1_size_bytes,
            "manifest_l1_size": format_capacity(manifest.l1_size_bytes),
            "manifest_l2_size_bytes": manifest.l2_size_bytes,
            "manifest_l2_size": format_capacity(manifest.l2_size_bytes),
            "manifest_skip_l2": manifest.skip_l2,
            "storage_tier": "l1-only" if manifest.skip_l2 else "l1+l2",
        },
        "result": result,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["config"], indent=2, ensure_ascii=False))
    print(f"results={args.out}")


def _load_samples(args: argparse.Namespace) -> list[BenchmarkSample]:
    if args.dataset == "imdb":
        if not args.imdb:
            raise ValueError("--imdb is required for --dataset imdb")
        return ImdbDataset(args.imdb, max_samples=args.max_samples).load()
    if not args.longbench_dir:
        raise ValueError("--longbench-dir is required for --dataset longbench")
    datasets = None
    if args.datasets:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    return LongBenchDataset(
        args.longbench_dir, datasets=datasets, max_samples=args.max_samples
    ).load()


def _required(value: str | None, flag: str) -> str:
    if value is None:
        raise ValueError(f"{flag} is required")
    return value


def _add_phase_comparison(result: dict[str, Any]) -> None:
    cold = result.get("cold", {})
    warm = result.get("warm", {})
    if not isinstance(cold, dict) or not isinstance(warm, dict):
        return
    cold_requests = cold.get("requests")
    warm_requests = warm.get("requests")
    if not isinstance(cold_requests, list) or not isinstance(warm_requests, list):
        return
    result["correctness"] = {
        "cold_warm_exact_match": request_text_exact_match(
            [
                RequestResult(**request) if isinstance(request, dict) else request
                for request in cold_requests
            ],
            [
                RequestResult(**request) if isinstance(request, dict) else request
                for request in warm_requests
            ],
        )
    }


def _capacity_limits(
    args: argparse.Namespace, store_dir: str
) -> BenchmarkCapacityLimits:
    limits = derive_capacity_limits(store_dir)
    return BenchmarkCapacityLimits(
        max_l1_bytes=min(limits.max_l1_bytes, args.max_l1_size)
        if args.max_l1_size is not None
        else limits.max_l1_bytes,
        max_l2_bytes=min(limits.max_l2_bytes, args.max_l2_size)
        if args.max_l2_size is not None
        else limits.max_l2_bytes,
        memory_available_bytes=limits.memory_available_bytes,
        disk_available_bytes=limits.disk_available_bytes,
    )


def _serialise_phase(
    phase: PhaseResult | list[Any], answers_by_id: dict[int, list[str]]
) -> dict[str, Any]:
    if isinstance(phase, PhaseResult):
        requests = phase.requests
        metrics = phase.metrics
        elapsed_ms = phase.elapsed_ms
    else:
        requests = phase
        metrics = {}
        elapsed_ms = 0.0
    summary = summarise_results(requests)
    if elapsed_ms > 0:
        summary["phase_elapsed_ms"] = elapsed_ms
        summary["phase_prompt_tok_per_s"] = (
            summary["prompt_tokens_total"] / (elapsed_ms / 1000)
            if elapsed_ms > 0
            else None
        )
    summary["http_trace_cache_hit_rate"] = summary.pop("cache_hit_rate")
    summary["answer_contains_accuracy"] = contains_accuracy(requests, answers_by_id)
    hit_ratios = metrics.get("hit_ratios", {}) if isinstance(metrics, dict) else {}
    summary["vllm_external_prefix_cache_hit_rate"] = hit_ratios.get(
        "vllm_external_prefix"
    )
    summary["backend_server_cache_hit_rate"] = _backend_server_hit_rate(hit_ratios)
    return {
        "summary": summary,
        "metrics": metrics,
        "requests": [asdict(result) for result in requests],
    }


def _backend_server_hit_rate(hit_ratios: dict[str, Any]) -> float | None:
    if (
        hit_ratios.get("daser_external_prefix") is not None
        or hit_ratios.get("daser_prometheus_tokens") is not None
        or hit_ratios.get("daser_prometheus_requests") is not None
    ):
        ratio = hit_ratios.get("daser_external_prefix")
        return float(ratio) if ratio is not None else None
    for key in (
        "lmcache_prometheus_lookup",
        "lmcache_prometheus_retrieve",
        "lmcache_status_prefetch",
    ):
        ratio = hit_ratios.get(key)
        if ratio is not None:
            return float(ratio)
    return None


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
