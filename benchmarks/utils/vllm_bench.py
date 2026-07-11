# SPDX-License-Identifier: Apache-2.0
"""vLLM ``bench serve`` load generator for the benchmark runner.

This module isolates the synthetic ``--load-generator vllm-bench`` path so the
orchestrator in ``run_bench.py`` only forks on which generator to use, not on
how each one works. It drives vLLM's built-in random workload through
``vllm bench serve`` and normalizes the result into the shared summary shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from benchmarks.utils.constants import (
    COMPARISON_IOURING_MEM,
    slot_size_for_block_tokens,
)
from benchmarks.utils.loadgen import (
    _wait_lmcache_quiescent,
    backend_server_hit_rate,
    collect_phase_metrics,
)
from benchmarks.utils.servers import BenchmarkManifest
from benchmarks.utils.sizing import (
    derive_benchmark_sizing,
    derive_capacity_limits,
    format_capacity,
)

if TYPE_CHECKING:
    from benchmarks.run_bench import BackendRun, RunBenchArgs


def bench_output_len(args: RunBenchArgs) -> int:
    """Return the resolved vLLM bench output length.

    Args:
        args: Benchmark runner arguments.

    Returns:
        ``bench_output_len`` when set, else the shared ``gen_max_tokens``.
    """
    if args.bench_output_len is not None:
        return args.bench_output_len
    return args.gen_max_tokens


def bench_max_concurrency(args: RunBenchArgs) -> int:
    """Return the resolved vLLM bench max concurrency.

    Args:
        args: Benchmark runner arguments.

    Returns:
        ``bench_max_concurrency`` when set, else the shared ``max_inflight``.
    """
    if args.bench_max_concurrency is not None:
        return args.bench_max_concurrency
    return args.max_inflight


def random_prefix_len(args: RunBenchArgs) -> int:
    """Return the resolved random shared-prefix length for vLLM bench.

    Args:
        args: Benchmark runner arguments.

    Returns:
        Explicit ``bench_random_prefix_len`` when set. For the dedicated
        ``vllm-bench-prefix`` mode, derives the length from
        ``bench_input_len * bench_prefix_ratio`` where ``bench_input_len`` is
        the final total prompt length.

    Thread-safety:
        Pure calculation over immutable argument values; safe to call from any
        thread and does not perform asyncio work.
    """
    explicit = int(args.bench_random_prefix_len)
    if explicit:
        return explicit
    if args.load_generator == "vllm-bench-prefix":
        return int(args.bench_input_len * args.bench_prefix_ratio)
    return 0


def bench_suffix_input_len(args: RunBenchArgs) -> int:
    """Return the random suffix length passed to ``vllm bench --input-len``.

    Args:
        args: Benchmark runner arguments.

    Returns:
        For ``vllm-bench-prefix``, the final total prompt length minus the
        shared prefix. For the original ``vllm-bench`` mode, the configured
        input length is already the random dataset input length.

    Thread-safety:
        Pure calculation over immutable argument values; safe to call from any
        thread and does not perform asyncio work.
    """
    if args.load_generator != "vllm-bench-prefix":
        return args.bench_input_len
    return args.bench_input_len - random_prefix_len(args)


def validate_args(args: RunBenchArgs) -> None:
    """Validate vLLM bench arguments with clear preflight errors.

    Args:
        args: Benchmark runner arguments.

    Raises:
        ValueError: If any vLLM bench argument is out of range.
    """
    if args.bench_num_prompts <= 0:
        raise ValueError("bench_num_prompts must be positive")
    if args.bench_input_len <= 0:
        raise ValueError("bench_input_len must be positive")
    if bench_output_len(args) <= 0:
        raise ValueError("bench_output_len must be positive")
    if bench_max_concurrency(args) <= 0:
        raise ValueError("bench_max_concurrency must be positive")
    if args.bench_random_prefix_len < 0:
        raise ValueError("bench_random_prefix_len must be non-negative")
    if args.bench_prefix_ratio < 0.0 or args.bench_prefix_ratio > 1.0:
        raise ValueError("bench_prefix_ratio must be in [0, 1]")
    if random_prefix_len(args) > args.bench_input_len:
        raise ValueError("bench_random_prefix_len must not exceed bench_input_len")
    if args.load_generator == "vllm-bench-prefix" and random_prefix_len(args) <= 0:
        raise ValueError("vllm-bench-prefix requires a positive shared prefix")
    if args.load_generator == "vllm-bench-prefix" and bench_suffix_input_len(args) <= 0:
        raise ValueError("vllm-bench-prefix requires a positive random suffix")
    if args.bench_random_range_ratio < 0.0:
        raise ValueError("bench_random_range_ratio must be non-negative")
    if args.bench_burstiness <= 0.0:
        raise ValueError("bench_burstiness must be positive")
    _validate_request_rate(args.bench_request_rate)


def _validate_request_rate(value: str) -> None:
    """Validate the vLLM bench request-rate argument."""
    if value == "inf":
        return
    try:
        rate = float(value)
    except ValueError as exc:
        raise ValueError(
            "bench_request_rate must be 'inf' or a positive number"
        ) from exc
    if rate <= 0.0 or math.isinf(rate) or math.isnan(rate):
        raise ValueError("bench_request_rate must be 'inf' or a positive number")


def _max_prompt_tokens(args: RunBenchArgs) -> int:
    variable_tokens = math.ceil(
        bench_suffix_input_len(args) * (1.0 + args.bench_random_range_ratio)
    )
    return max(1, random_prefix_len(args) + variable_tokens)


def prepare_config(args: RunBenchArgs, run_root: Path) -> dict[str, Any]:
    """Build the prepare config for synthetic vLLM bench random workloads.

    Args:
        args: Benchmark runner arguments.
        run_root: Run root used for capacity probing.

    Returns:
        JSON-serializable prepare config.

    Thread-safety:
        Reads current disk and host memory state through sizing helpers.
    """
    prompt_tokens = _max_prompt_tokens(args)
    max_prompt_blocks = max(1, math.ceil(prompt_tokens / args.block_size))
    total_blocks = args.bench_num_prompts * max_prompt_blocks
    slot_size = slot_size_for_block_tokens(
        args.model,
        args.block_size,
        args.tensor_parallel_size,
    )
    sizing = derive_benchmark_sizing(
        total_blocks=total_blocks,
        max_prompt_blocks=max_prompt_blocks,
        slot_size=slot_size,
        mode=COMPARISON_IOURING_MEM,
        evict=args.evict,
        capacity_limits=derive_capacity_limits(run_root),
    )
    return {
        "dataset": (
            "vllm-bench-prefix-random"
            if args.load_generator == "vllm-bench-prefix"
            else "vllm-bench-random"
        ),
        "num_samples": args.bench_num_prompts,
        "max_inflight": bench_max_concurrency(args),
        "gen_params": {
            "max_tokens": bench_output_len(args),
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.bench_seed,
        },
        "total_prompt_tokens": args.bench_num_prompts * prompt_tokens,
        "total_blocks": total_blocks,
        "max_prompt_blocks": max_prompt_blocks,
        "max_prompt_tokens": prompt_tokens,
        "block_size": args.block_size,
        "slot_size_bytes": slot_size,
        "bench_num_prompts": args.bench_num_prompts,
        "bench_input_len": args.bench_input_len,
        "bench_suffix_input_len": bench_suffix_input_len(args),
        "bench_output_len": bench_output_len(args),
        "bench_request_rate": args.bench_request_rate,
        "bench_max_concurrency": bench_max_concurrency(args),
        "bench_random_prefix_len": random_prefix_len(args),
        "bench_prefix_ratio": args.bench_prefix_ratio,
        "bench_random_range_ratio": args.bench_random_range_ratio,
        "bench_seed": args.bench_seed,
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


def run_load(
    args: RunBenchArgs,
    manifest: BenchmarkManifest | None,
    backend_run: BackendRun,
    backend_dir: Path,
    result_path: Path,
    *,
    run_command: Callable[[list[str]], Any],
    print_kv: Callable[[str, Any], None],
) -> None:
    """Run vLLM bench phases and write results.json.

    Args:
        args: Benchmark runner arguments.
        manifest: Started service manifest; read from disk when None.
        backend_run: Resolved backend row.
        backend_dir: Per-backend output directory.
        result_path: results.json destination.
        run_command: Callable that runs a subprocess command (injected).
        print_kv: Callable that prints a key/value progress line (injected).
    """
    if manifest is None:
        manifest = BenchmarkManifest.read(backend_dir / "manifest.json")
    if args.load_generator == "vllm-bench-prefix":
        result = _run_single_prefix_load(
            args,
            manifest,
            backend_run,
            backend_dir,
            run_command=run_command,
        )
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if backend_run.backend == "vllm":
        raw = backend_dir / "vllm_bench_baseline.json"
        baseline_metrics, baseline_hit_rate = _run_phase(
            args, manifest, raw, run_command=run_command
        )
        baseline_summary = _normalise_result(raw)
        _apply_phase_metrics(baseline_summary, baseline_hit_rate)
        result = {
            "manifest": asdict(manifest),
            "result": {
                "baseline": {
                    "summary": baseline_summary,
                    "metrics": baseline_metrics,
                }
            },
        }
    else:
        cold_raw = backend_dir / "vllm_bench_cold.json"
        warm_raw = backend_dir / "vllm_bench_warm.json"
        cold_metrics, cold_hit_rate = _run_phase(
            args, manifest, cold_raw, run_command=run_command
        )
        if backend_run.backend == "lmcache":
            print_kv("lmcache_warm_wait", "quiescent")
            asyncio.run(_wait_lmcache_quiescent(manifest, settle_seconds=0.0))
        elif backend_run.backend == "daser":
            _drain_daser(manifest, print_kv=print_kv)
        warm_metrics, warm_hit_rate = _run_phase(
            args, manifest, warm_raw, run_command=run_command
        )
        cold_summary = _normalise_result(cold_raw)
        warm_summary = _normalise_result(warm_raw)
        _apply_phase_metrics(cold_summary, cold_hit_rate)
        _apply_phase_metrics(warm_summary, warm_hit_rate)
        result = {
            "manifest": asdict(manifest),
            "result": {
                "cold": {"summary": cold_summary, "metrics": cold_metrics},
                "warm": {"summary": warm_summary, "metrics": warm_metrics},
            },
            "correctness": _compare_outputs(cold_raw, warm_raw),
        }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _run_single_prefix_load(
    args: RunBenchArgs,
    manifest: BenchmarkManifest,
    backend_run: BackendRun,
    backend_dir: Path,
    *,
    run_command: Callable[[list[str]], Any],
) -> dict[str, Any]:
    """Run one shared-prefix benchmark phase for one backend row."""
    if backend_run.backend == "vllm":
        raw = backend_dir / "vllm_bench_baseline.json"
        baseline_metrics, baseline_hit_rate = _run_phase(
            args, manifest, raw, run_command=run_command
        )
        baseline_summary = _normalise_result(raw)
        _apply_phase_metrics(baseline_summary, baseline_hit_rate)
        return {
            "manifest": asdict(manifest),
            "raw_result": str(raw),
            "result": {
                "baseline": {
                    "summary": baseline_summary,
                    "metrics": baseline_metrics,
                }
            },
        }

    raw = backend_dir / "vllm_bench_prefix.json"
    metrics, hit_rate = _run_phase(args, manifest, raw, run_command=run_command)
    summary = _normalise_result(raw)
    _apply_phase_metrics(summary, hit_rate)
    baseline_raw = backend_dir.parent / "baseline" / "vllm_bench_baseline.json"
    return {
        "manifest": asdict(manifest),
        "raw_result": str(raw),
        "baseline_raw_result": str(baseline_raw),
        "result": {
            "prefix": {
                "summary": summary,
                "metrics": metrics,
            }
        },
        "correctness": _compare_with_baseline(baseline_raw, raw),
    }


def _run_phase(
    args: RunBenchArgs,
    manifest: BenchmarkManifest,
    raw_path: Path,
    *,
    run_command: Callable[[list[str]], Any],
) -> tuple[dict[str, Any], float | None]:
    before_metrics = asyncio.run(collect_phase_metrics(manifest))
    run_command(_bench_command(args, manifest.endpoints["vllm"], raw_path))
    return _collect_phase_metrics(manifest, before_metrics)


def _collect_phase_metrics(
    manifest: BenchmarkManifest,
    before_metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], float | None]:
    """Collect vLLM bench phase metric deltas and backend token hit rate.

    Args:
        manifest: Started benchmark service manifest.
        before_metrics: Optional pre-phase metric snapshot; absent yields an
            empty delta.

    Returns:
        Phase metric deltas and the backend token-level cache hit ratio.
    """
    if before_metrics is None:
        before_metrics = {
            "vllm_prometheus": {},
            "backend_prometheus": {},
            "backend_status": {},
        }
    metrics = asyncio.run(collect_phase_metrics(manifest, before_metrics))
    hit_ratios = metrics.get("hit_ratios", {}) if isinstance(metrics, dict) else {}
    return metrics, backend_server_hit_rate(hit_ratios)


def _apply_phase_metrics(
    summary: dict[str, Any],
    backend_hit_rate: float | None,
) -> None:
    if backend_hit_rate is not None:
        summary["backend_server_cache_hit_rate"] = backend_hit_rate


def _drain_daser(
    manifest: BenchmarkManifest,
    *,
    print_kv: Callable[[str, Any], None],
) -> None:
    endpoint = manifest.endpoints.get("daser")
    if endpoint is None:
        return
    try:
        response = httpx.post(f"{endpoint.url}/drain", timeout=30.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print_kv("daser_drain_status", f"unavailable ({exc})")


def _bench_command(
    args: RunBenchArgs,
    endpoint: Any,
    raw_path: Path,
) -> list[str]:
    """Build a ``vllm bench serve`` command for one benchmark phase."""
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        endpoint.url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        *(["--trust-remote-code"] if args.trust_remote_code else []),
        "--dataset-name",
        "random",
        "--num-prompts",
        str(args.bench_num_prompts),
        "--input-len",
        str(bench_suffix_input_len(args)),
        "--output-len",
        str(bench_output_len(args)),
        "--request-rate",
        str(args.bench_request_rate),
        "--max-concurrency",
        str(bench_max_concurrency(args)),
        "--random-prefix-len",
        str(random_prefix_len(args)),
        "--random-range-ratio",
        str(args.bench_random_range_ratio),
        "--seed",
        str(args.bench_seed),
        "--burstiness",
        str(args.bench_burstiness),
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(raw_path.parent),
        "--result-filename",
        raw_path.name,
    ]


def _normalise_result(path: Path) -> dict[str, Any]:
    """Convert a vLLM bench JSON result to the benchmark summary shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    duration_s = float(_first_number(payload, ("duration", "benchmark_duration"), 0.0))
    prompt_tokens = int(_first_number(payload, ("total_input_tokens",), 0))
    completion_tokens = int(_first_number(payload, ("total_output_tokens",), 0))
    return {
        "num_requests": int(_first_number(payload, ("completed",), 0)),
        "num_errors": int(_first_number(payload, ("failed",), 0)),
        "ttft_ms_mean": float(_first_number(payload, ("mean_ttft_ms",), 0.0)),
        "latency_ms_mean": float(
            _first_number(
                payload,
                ("mean_e2el_ms", "mean_latency_ms", "mean_ttft_ms"),
                0.0,
            )
        ),
        "phase_elapsed_ms": duration_s * 1000.0,
        "phase_prompt_tok_per_s": prompt_tokens / duration_s if duration_s > 0 else 0.0,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
    }


def _compare_outputs(cold_path: Path, warm_path: Path) -> dict[str, Any]:
    """Compare detailed vLLM bench generated text across cold and warm phases."""
    cold = json.loads(cold_path.read_text(encoding="utf-8"))
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    cold_texts = _generated_texts(cold)
    warm_texts = _generated_texts(warm)
    if cold_texts is None or warm_texts is None:
        return {
            "cold_warm_exact_match": {
                "available": False,
                "matches": 0,
                "total": 0,
                "accuracy": None,
                "reason": "vLLM bench result did not include generated text details",
            }
        }
    paired = min(len(cold_texts), len(warm_texts))
    total = max(len(cold_texts), len(warm_texts))
    matches = sum(1 for idx in range(paired) if cold_texts[idx] == warm_texts[idx])
    return {
        "cold_warm_exact_match": {
            "available": True,
            "matches": matches,
            "total": total,
            "accuracy": matches / total if total else None,
            "length_mismatch": len(cold_texts) != len(warm_texts),
        }
    }


def _compare_with_baseline(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    """Compare detailed vLLM bench generated text against baseline output.

    Args:
        baseline_path: Raw baseline vLLM bench JSON path.
        candidate_path: Raw backend vLLM bench JSON path.

    Returns:
        Correctness dict keyed by ``baseline_exact_match``.
    """
    result = _compare_generated_texts(
        baseline_path,
        candidate_path,
        unavailable_reason=("vLLM bench result did not include generated text details"),
    )
    return {"baseline_exact_match": result}


def _compare_generated_texts(
    expected_path: Path,
    actual_path: Path,
    *,
    unavailable_reason: str,
) -> dict[str, Any]:
    """Compare generated text lists from two raw vLLM bench JSON files."""
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    expected_texts = _generated_texts(expected)
    actual_texts = _generated_texts(actual)
    if expected_texts is None or actual_texts is None:
        return {
            "available": False,
            "matches": 0,
            "total": 0,
            "accuracy": None,
            "reason": unavailable_reason,
        }
    paired = min(len(expected_texts), len(actual_texts))
    total = max(len(expected_texts), len(actual_texts))
    matches = sum(
        1 for idx in range(paired) if expected_texts[idx] == actual_texts[idx]
    )
    return {
        "available": True,
        "matches": matches,
        "total": total,
        "accuracy": matches / total if total else None,
        "length_mismatch": len(expected_texts) != len(actual_texts),
    }


def _generated_texts(payload: dict[str, Any]) -> list[str] | None:
    """Extract generated texts from known vLLM bench result shapes."""
    generated_texts = payload.get("generated_texts")
    if isinstance(generated_texts, list) and all(
        isinstance(text, str) for text in generated_texts
    ):
        return generated_texts
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        return None
    texts: list[str] = []
    for output in outputs:
        if not isinstance(output, dict):
            return None
        text = output.get("generated_text")
        if not isinstance(text, str):
            return None
        texts.append(text)
    return texts


def _first_number(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
    return default
