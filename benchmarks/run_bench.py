# SPDX-License-Identifier: Apache-2.0
"""Run end-to-end service benchmarks across backend comparison rows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils.constants import (
    COMPARISON_IOURING_MEM,
    slot_size_for_block_tokens,
)
from benchmarks.utils.servers import BenchmarkManifest, stop_from_pid_file
from benchmarks.utils.sizing import (
    BenchmarkCapacityLimits,
    derive_benchmark_sizing,
    derive_capacity_limits,
    format_capacity,
)

_DASER_METRICS_SETTLE_SECONDS = 2.0
_BACKEND_ROWS = ("baseline", "lmcache", "daser-prefix", "daser-chunk")
_BACKEND_SELECTIONS = ("all", *_BACKEND_ROWS)


@dataclass(frozen=True)
class BackendRun:
    """Resolved benchmark backend row.

    Args:
        label: Directory and report label for this benchmark row.
        backend: Backend name accepted by ``bench_start_servers.py``.
        reuse_mode: DaseR cache reuse mode, or ``none`` for non-DaseR rows.

    Thread-safety:
        Immutable value object.
    """

    label: str
    backend: str
    reuse_mode: str


@dataclass(frozen=True)
class RunBenchArgs:
    """Arguments for the benchmark orchestration entrypoint.

    Args:
        backend: Requested backend row, comma-separated row list, or ``all``.
        load_generator: Load generator implementation.
        dataset: Dataset family.
        model: Model path.
        store_dir: Parent directory for the generated run root.
        imdb: Optional IMDB CSV path.
        longbench_dir: Optional LongBench data directory.
        datasets: Optional comma-separated LongBench dataset stems.
        max_samples: Maximum samples per dataset.
        gpu_id: GPU selection mode or device ID.
        gpu_util: vLLM GPU memory utilization.
        max_num_seqs: vLLM maximum sequence concurrency.
        max_num_batched_tokens: Optional vLLM scheduler token budget.
        block_size: vLLM KV block size in tokens.
        max_inflight: HTTP load generator concurrency.
        gen_max_tokens: Maximum generated tokens.
        max_context_tokens: Prompt token ceiling; 0 infers from model metadata.
        bench_num_prompts: vLLM bench random prompt count.
        bench_input_len: vLLM bench random input length.
        bench_output_len: vLLM bench random output length.
        bench_request_rate: vLLM bench request rate.
        bench_max_concurrency: vLLM bench max in-flight requests.
        bench_random_prefix_len: Fixed prefix length for vLLM random dataset.
        bench_random_range_ratio: vLLM random input/output length range ratio.
        bench_seed: vLLM bench random seed.
        bench_burstiness: vLLM bench burstiness factor.
        evict: Whether to enable L2 and eviction sizing.
        prometheus_url: Optional Prometheus base URL for scrape diagnostics.

    Thread-safety:
        Immutable value object.
    """

    backend: str = "all"
    load_generator: str = "internal"
    dataset: str = "longbench"
    model: str = ""
    store_dir: str = ""
    imdb: str | None = None
    longbench_dir: str | None = None
    datasets: str | None = None
    max_samples: int = 20
    gpu_id: str = "auto"
    gpu_util: float = 0.85
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 0
    block_size: int = 16
    max_inflight: int = 32
    gen_max_tokens: int = 128
    max_context_tokens: int = 0
    bench_num_prompts: int = 1000
    bench_input_len: int = 1024
    bench_output_len: int | None = None
    bench_request_rate: str = "inf"
    bench_max_concurrency: int | None = None
    bench_random_prefix_len: int = 0
    bench_random_range_ratio: float = 0.0
    bench_seed: int = 42
    bench_burstiness: float = 1.0
    evict: bool = False
    prometheus_url: str = "http://127.0.0.1:9090"


def _backend_selection(value: str) -> str:
    """Validate and normalize a benchmark backend selection."""
    value = value.strip()
    if value == "all":
        return value
    names = [name.strip() for name in value.split(",")]
    invalid = [name for name in names if name not in _BACKEND_ROWS]
    if not names or any(not name for name in names) or invalid:
        valid = ", ".join(_BACKEND_SELECTIONS)
        raise argparse.ArgumentTypeError(
            f"unknown backend selection: {value}; choose from {valid}"
        )
    return ",".join(names)


def parse_args(argv: list[str] | None = None) -> RunBenchArgs:
    """Parse CLI arguments.

    Args:
        argv: Optional argument vector without the executable name.

    Returns:
        Parsed benchmark orchestration arguments.

    Thread-safety:
        Pure except for argparse error handling on invalid input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="all",
        metavar="{all,baseline,lmcache,daser-prefix,daser-chunk}[,...]",
        type=_backend_selection,
        help=(
            "Benchmark rows to run. Use all, one row, or a comma-separated "
            "subset such as baseline,lmcache,daser-prefix."
        ),
    )
    parser.add_argument(
        "--load-generator",
        choices=("internal", "vllm-bench"),
        default="internal",
    )
    parser.add_argument("--dataset", choices=("imdb", "longbench"), default="longbench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--imdb")
    parser.add_argument("--longbench-dir")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--gen-max-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--bench-num-prompts", type=int, default=1000)
    parser.add_argument("--bench-input-len", type=int, default=1024)
    parser.add_argument("--bench-output-len", type=int, default=None)
    parser.add_argument("--bench-request-rate", default="inf")
    parser.add_argument("--bench-max-concurrency", type=int, default=None)
    parser.add_argument("--bench-random-prefix-len", type=int, default=0)
    parser.add_argument("--bench-random-range-ratio", type=float, default=0.0)
    parser.add_argument("--bench-seed", type=int, default=42)
    parser.add_argument("--bench-burstiness", type=float, default=1.0)
    parser.add_argument("--evict", action="store_true")
    parser.add_argument(
        "--prometheus-url",
        default="http://127.0.0.1:9090",
        help=(
            "Prometheus base URL used to print DaseR scrape diagnostics; "
            "set to empty to disable"
        ),
    )
    args = parser.parse_args(argv)
    return RunBenchArgs(
        backend=args.backend,
        load_generator=args.load_generator,
        dataset=args.dataset,
        model=args.model,
        store_dir=args.store_dir,
        imdb=args.imdb,
        longbench_dir=args.longbench_dir,
        datasets=args.datasets,
        max_samples=args.max_samples,
        gpu_id=args.gpu_id,
        gpu_util=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        max_inflight=args.max_inflight,
        gen_max_tokens=args.gen_max_tokens,
        max_context_tokens=args.max_context_tokens,
        bench_num_prompts=args.bench_num_prompts,
        bench_input_len=args.bench_input_len,
        bench_output_len=args.bench_output_len,
        bench_request_rate=args.bench_request_rate,
        bench_max_concurrency=args.bench_max_concurrency,
        bench_random_prefix_len=args.bench_random_prefix_len,
        bench_random_range_ratio=args.bench_random_range_ratio,
        bench_seed=args.bench_seed,
        bench_burstiness=args.bench_burstiness,
        evict=args.evict,
        prometheus_url=args.prometheus_url,
    )


def run_benchmark(args: RunBenchArgs) -> Path:
    """Run prepare, service startup, load, and cleanup for benchmark rows.

    Args:
        args: Parsed benchmark orchestration arguments.

    Returns:
        Run root directory containing prepare, manifests, logs, and results.

    Thread-safety:
        Not thread-safe. The function starts local subprocess services, writes
        under ``args.store_dir``, and owns cleanup for pid files below the run
        root.
    """
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.store_dir).expanduser() / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    prepare_path = run_root / "prepare.json"
    backend_runs = _expand_backend_runs(args.backend)
    _validate_backend_runs(backend_runs, load_generator=args.load_generator)

    _print_stage("prepare")
    _print_kv("load_generator", args.load_generator)
    if args.load_generator == "vllm-bench":
        _print_kv("dataset", "vllm-bench-random")
        _print_kv("bench_num_prompts", args.bench_num_prompts)
        _print_kv("bench_input_len", args.bench_input_len)
        _print_kv("bench_output_len", _bench_output_len(args))
    else:
        _print_kv("dataset", args.dataset)
        _print_kv("max_samples", args.max_samples)
    _print_kv("block_size", args.block_size)
    _print_kv("output", prepare_path)
    if args.load_generator == "vllm-bench":
        prepare = {"config": _bench_prepare_config(args, run_root)}
        prepare_path.write_text(json.dumps(prepare, indent=2), encoding="utf-8")
        print(json.dumps(prepare["config"], indent=2), flush=True)
        print(f"prepare={prepare_path}", flush=True)
    else:
        _run_command(_prepare_command(args, run_root, prepare_path))
        prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    config = prepare["config"]
    derived_l1 = int(config["derived_l1_size_bytes"])
    derived_l2 = int(config["derived_l2_size_bytes"])
    _print_kv("derived_l1_bytes", derived_l1)
    _print_kv("derived_l2_bytes", derived_l2)

    result_paths: list[tuple[BackendRun, Path]] = []
    try:
        for backend_run in backend_runs:
            result_paths.append(
                (
                    backend_run,
                    _run_backend(
                        args,
                        run_id,
                        run_root,
                        prepare_path,
                        backend_run,
                        derived_l1,
                        derived_l2,
                    ),
                )
            )
    finally:
        _cleanup(run_root)

    _print_comparison_summary(result_paths)
    _print_stage("complete")
    _print_kv("run_root", run_root)
    return run_root


def _expand_backend_runs(backend: str) -> list[BackendRun]:
    """Resolve a requested backend into concrete benchmark rows.

    Args:
        backend: User-facing backend name.

    Returns:
        Concrete backend rows in execution order.

    Thread-safety:
        Pure helper.
    """
    row_map = {
        "baseline": BackendRun("baseline", "vllm", "none"),
        "lmcache": BackendRun("lmcache", "lmcache", "none"),
        "daser-chunk": BackendRun("daser-chunk", "daser", "chunk"),
        "daser-prefix": BackendRun("daser-prefix", "daser", "prefix"),
    }
    if backend.strip() == "all":
        return [
            row_map["baseline"],
            row_map["lmcache"],
            row_map["daser-chunk"],
            row_map["daser-prefix"],
        ]
    names = _backend_selection(backend).split(",")
    return [row_map[name] for name in names]


def _validate_backend_runs(
    backend_runs: list[BackendRun],
    *,
    load_generator: str,
) -> None:
    """Validate backend rows against the selected load generator.

    Args:
        backend_runs: Resolved benchmark rows.
        load_generator: Selected load generator name.

    Raises:
        ValueError: If a backend row is incompatible.

    Thread-safety:
        Pure helper.
    """
    if load_generator != "vllm-bench":
        return
    unsupported = [run.label for run in backend_runs if run.label == "daser-chunk"]
    if unsupported:
        raise ValueError(
            "vllm-bench load generator does not support daser-chunk; "
            "select baseline,lmcache,daser-prefix or use --load-generator internal"
        )


def _run_backend(
    args: RunBenchArgs,
    run_id: str,
    run_root: Path,
    prepare_path: Path,
    backend_run: BackendRun,
    derived_l1: int,
    derived_l2: int,
) -> Path:
    """Start one backend, run load, and clean services."""
    backend_dir = run_root / backend_run.label
    backend_dir.mkdir(parents=True, exist_ok=True)
    _print_stage("start", backend_run.label)
    _print_kv("backend", backend_run.backend)
    if backend_run.backend == "daser":
        _print_kv("reuse_mode", backend_run.reuse_mode)
    _print_kv("work_dir", backend_dir)
    _run_command(
        _start_command(
            args,
            run_id,
            backend_dir,
            backend_run,
            derived_l1,
            derived_l2,
        )
    )
    manifest = None
    if _should_probe_daser_metrics(backend_run):
        manifest = BenchmarkManifest.read(backend_dir / "manifest.json")
        _probe_daser_metrics(
            manifest,
            phase="startup",
            prometheus_url=args.prometheus_url,
        )

    result_path = backend_dir / "results.json"
    _print_stage("cold/warm load", backend_run.label)
    _print_kv("output", result_path)
    if args.load_generator == "vllm-bench":
        _run_vllm_bench_load(args, manifest, backend_run, backend_dir, result_path)
    else:
        _run_command(_load_command(args, backend_dir, prepare_path, result_path))
    if manifest is not None:
        _probe_daser_metrics(
            manifest,
            phase="post-load",
            prometheus_url=args.prometheus_url,
        )
    _print_kv("results", result_path)
    _cleanup(run_root)
    return result_path


def _bench_prepare_config(args: RunBenchArgs, run_root: Path) -> dict[str, Any]:
    """Build prepare config for synthetic vLLM bench random workloads.

    Args:
        args: Benchmark runner arguments.
        run_root: Run root used for capacity probing.

    Returns:
        JSON-serializable prepare config.

    Thread-safety:
        Reads current disk and host memory state through sizing helpers.
    """
    prompt_tokens = _bench_max_prompt_tokens(args)
    max_prompt_blocks = max(1, math.ceil(prompt_tokens / args.block_size))
    total_blocks = args.bench_num_prompts * max_prompt_blocks
    slot_size = slot_size_for_block_tokens(args.block_size)
    sizing = derive_benchmark_sizing(
        total_blocks=total_blocks,
        max_prompt_blocks=max_prompt_blocks,
        slot_size=slot_size,
        mode=COMPARISON_IOURING_MEM,
        evict=args.evict,
        capacity_limits=_bench_capacity_limits(args, run_root),
    )
    return {
        "dataset": "vllm-bench-random",
        "num_samples": args.bench_num_prompts,
        "max_inflight": _bench_max_concurrency(args),
        "gen_params": {
            "max_tokens": _bench_output_len(args),
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.bench_seed,
        },
        "total_prompt_tokens": args.bench_num_prompts * prompt_tokens,
        "total_blocks": total_blocks,
        "max_prompt_blocks": max_prompt_blocks,
        "max_prompt_tokens": prompt_tokens,
        "block_size": args.block_size,
        "bench_num_prompts": args.bench_num_prompts,
        "bench_input_len": args.bench_input_len,
        "bench_output_len": _bench_output_len(args),
        "bench_request_rate": args.bench_request_rate,
        "bench_max_concurrency": _bench_max_concurrency(args),
        "bench_random_prefix_len": args.bench_random_prefix_len,
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


def _bench_capacity_limits(
    args: RunBenchArgs,
    run_root: Path,
) -> BenchmarkCapacityLimits:
    return derive_capacity_limits(run_root)


def _bench_max_prompt_tokens(args: RunBenchArgs) -> int:
    variable_tokens = math.ceil(
        args.bench_input_len * (1.0 + args.bench_random_range_ratio)
    )
    return max(1, args.bench_random_prefix_len + variable_tokens)


def _bench_output_len(args: RunBenchArgs) -> int:
    if args.bench_output_len is not None:
        return args.bench_output_len
    return args.gen_max_tokens


def _bench_max_concurrency(args: RunBenchArgs) -> int:
    if args.bench_max_concurrency is not None:
        return args.bench_max_concurrency
    return args.max_inflight


def _run_vllm_bench_load(
    args: RunBenchArgs,
    manifest: BenchmarkManifest | None,
    backend_run: BackendRun,
    backend_dir: Path,
    result_path: Path,
) -> None:
    if manifest is None:
        manifest = BenchmarkManifest.read(backend_dir / "manifest.json")
    if backend_run.backend == "vllm":
        raw = backend_dir / "vllm_bench_baseline.json"
        _run_command(_vllm_bench_command(args, manifest.endpoints["vllm"], raw))
        result = {
            "manifest": _manifest_payload(manifest),
            "result": {"baseline": {"summary": _normalise_vllm_bench_result(raw)}},
        }
    else:
        cold_raw = backend_dir / "vllm_bench_cold.json"
        warm_raw = backend_dir / "vllm_bench_warm.json"
        _run_command(_vllm_bench_command(args, manifest.endpoints["vllm"], cold_raw))
        if backend_run.backend == "lmcache":
            _wait_with_message("lmcache_warm_settle_s", 10.0)
        elif backend_run.backend == "daser":
            _drain_daser(manifest)
        _run_command(_vllm_bench_command(args, manifest.endpoints["vllm"], warm_raw))
        result = {
            "manifest": _manifest_payload(manifest),
            "result": {
                "cold": {"summary": _normalise_vllm_bench_result(cold_raw)},
                "warm": {"summary": _normalise_vllm_bench_result(warm_raw)},
            },
        }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _vllm_bench_command(
    args: RunBenchArgs,
    endpoint: Any,
    raw_path: Path,
) -> list[str]:
    """Build a vLLM bench serve command for one benchmark phase."""
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
        "--dataset-name",
        "random",
        "--num-prompts",
        str(args.bench_num_prompts),
        "--input-len",
        str(args.bench_input_len),
        "--output-len",
        str(_bench_output_len(args)),
        "--request-rate",
        str(args.bench_request_rate),
        "--max-concurrency",
        str(_bench_max_concurrency(args)),
        "--random-prefix-len",
        str(args.bench_random_prefix_len),
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
        "--result-dir",
        str(raw_path.parent),
        "--result-filename",
        raw_path.name,
    ]


def _normalise_vllm_bench_result(path: Path) -> dict[str, Any]:
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


def _manifest_payload(manifest: BenchmarkManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "backend": manifest.backend,
        "reuse_mode": manifest.reuse_mode,
        "model": manifest.model,
        "store_dir": manifest.store_dir,
        "l1_size_bytes": manifest.l1_size_bytes,
        "l2_size_bytes": manifest.l2_size_bytes,
        "skip_l2": manifest.skip_l2,
        "endpoints": {
            name: {"url": endpoint.url} for name, endpoint in manifest.endpoints.items()
        },
        "log_dir": manifest.log_dir,
        "pid_file": manifest.pid_file,
        "block_size": manifest.block_size,
    }


def _wait_with_message(label: str, seconds: float) -> None:
    _print_kv(label, seconds)
    time.sleep(seconds)


def _drain_daser(manifest: BenchmarkManifest) -> None:
    endpoint = manifest.endpoints.get("daser")
    if endpoint is None:
        return
    try:
        response = httpx.post(f"{endpoint.url}/drain", timeout=30.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _print_kv("daser_drain_status", f"unavailable ({exc})")


def _prepare_command(
    args: RunBenchArgs, run_root: Path, prepare_path: Path
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/bench_load.py",
        "--prepare-only",
        "--model",
        args.model,
        "--store-dir",
        str(run_root),
        "--dataset",
        args.dataset,
        "--max-samples",
        str(args.max_samples),
        "--max-inflight",
        str(args.max_inflight),
        "--block-size",
        str(args.block_size),
        "--gen-max-tokens",
        str(args.gen_max_tokens),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--out",
        str(prepare_path),
    ]
    _append_common_optional_args(command, args)
    return command


def _start_command(
    args: RunBenchArgs,
    run_id: str,
    backend_dir: Path,
    backend_run: BackendRun,
    derived_l1: int,
    derived_l2: int,
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/bench_start_servers.py",
        "--backend",
        backend_run.backend,
        "--model",
        args.model,
        "--store-dir",
        str(backend_dir),
        "--run-id",
        run_id,
        "--gpu-id",
        args.gpu_id,
        "--gpu-util",
        str(args.gpu_util),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--block-size",
        str(args.block_size),
        "--l1-size",
        str(derived_l1),
        "--l2-size",
        str(derived_l2),
    ]
    if backend_run.backend == "daser":
        command.extend(["--cache-reuse-mode", backend_run.reuse_mode])
    if not args.evict and backend_run.backend in ("daser", "lmcache"):
        command.append("--skip-l2")
    return command


def _load_command(
    args: RunBenchArgs,
    backend_dir: Path,
    prepare_path: Path,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/bench_load.py",
        "--manifest",
        str(backend_dir / "manifest.json"),
        "--prepared-config",
        str(prepare_path),
        "--dataset",
        args.dataset,
        "--max-samples",
        str(args.max_samples),
        "--max-inflight",
        str(args.max_inflight),
        "--block-size",
        str(args.block_size),
        "--gen-max-tokens",
        str(args.gen_max_tokens),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--out",
        str(result_path),
    ]
    _append_common_optional_args(command, args)
    return command


def _append_common_optional_args(command: list[str], args: RunBenchArgs) -> None:
    if args.evict:
        command.append("--evict")
    if args.imdb:
        command.extend(["--imdb", args.imdb])
    if args.longbench_dir:
        command.extend(["--longbench-dir", args.longbench_dir])
    if args.datasets:
        command.extend(["--datasets", args.datasets])


def _print_comparison_summary(result_paths: list[tuple[BackendRun, Path]]) -> None:
    _print_stage("comparison summary")
    if not result_paths:
        return
    for backend_run, result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"{backend_run.label}:", flush=True)
        for key, value in _comparison_fields(result).items():
            if value is not None:
                print(f"  {key}: {value}", flush=True)
        print(f"  results: {result_path}", flush=True)


def _comparison_fields(result: dict[str, Any]) -> dict[str, Any]:
    phases = result.get("result")
    if not isinstance(phases, dict):
        return {}
    fields: dict[str, Any] = {}
    baseline_summary = _phase_summary(phases, "baseline")
    if baseline_summary is not None:
        _add_summary_fields(fields, "baseline", baseline_summary)
    cold = phases.get("cold")
    if isinstance(cold, dict):
        _add_cold_fields(fields, cold)
    warm_summary = _phase_summary(phases, "warm")
    if warm_summary is not None:
        _add_summary_fields(fields, "warm", warm_summary)
    _add_correctness_fields(fields, result.get("correctness"))
    return fields


def _phase_summary(
    phases: dict[str, Any],
    phase_name: str,
) -> dict[str, Any] | None:
    phase = phases.get(phase_name)
    if not isinstance(phase, dict):
        return None
    summary = phase.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def _add_cold_fields(fields: dict[str, Any], cold: dict[str, Any]) -> None:
    cold_summary = cold.get("summary")
    if isinstance(cold_summary, dict):
        _add_summary_fields(fields, "cold", cold_summary)
    for source_key, target_key in (
        ("uploaded_documents", "cold_uploaded_documents"),
        ("upload_ms", "cold_upload_ms"),
    ):
        value = cold.get(source_key)
        if value is not None:
            fields[target_key] = value


def _add_summary_fields(
    fields: dict[str, Any],
    prefix: str,
    summary: dict[str, Any],
) -> None:
    for source_key, target_key in (
        ("ttft_ms_mean", f"{prefix}_ttft_ms_mean"),
        ("latency_ms_mean", f"{prefix}_latency_ms_mean"),
        ("phase_elapsed_ms", f"{prefix}_elapsed_ms"),
        ("phase_prompt_tok_per_s", f"{prefix}_prompt_tok_per_s"),
        ("backend_server_cache_hit_rate", f"{prefix}_backend_cache_hit_rate"),
        ("answer_contains_accuracy", f"{prefix}_answer_contains_accuracy"),
    ):
        value = summary.get(source_key)
        if value is not None:
            fields[target_key] = value


def _add_correctness_fields(
    fields: dict[str, Any],
    correctness: Any,
) -> None:
    if not isinstance(correctness, dict):
        return
    exact_match = correctness.get("cold_warm_exact_match")
    if not isinstance(exact_match, dict):
        return
    accuracy = exact_match.get("accuracy")
    if accuracy is not None:
        fields["cold_warm_exact_match_accuracy"] = accuracy


def _run_command(command: list[str]) -> None:
    """Print and run a benchmark subprocess command.

    Args:
        command: Command argv.

    Thread-safety:
        Reentrant, but subprocess effects depend on the invoked command.
    """
    print(f"[bench] run: {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _stage_title(label: str | None, phase: str | None = None) -> str:
    """Return a visual benchmark stage title.

    Args:
        label: Optional backend label, or a global phase when ``phase`` is None.
        phase: Optional backend phase name.

    Returns:
        Uppercase stage separator text.

    Thread-safety:
        Pure helper.
    """
    if phase is None:
        text = str(label or "").upper()
    else:
        text = f"{label} {phase}".upper()
    return f"== {text} =="


def _print_stage(phase: str, label: str | None = None) -> None:
    title = _stage_title(phase) if label is None else _stage_title(label, phase)
    print(f"\n{title}", flush=True)


def _print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}", flush=True)


def _should_probe_daser_metrics(backend_run: BackendRun) -> bool:
    """Return whether a benchmark row should expose DaseR metrics."""
    return backend_run.backend == "daser"


def _probe_daser_metrics(
    manifest: BenchmarkManifest,
    *,
    phase: str,
    prometheus_url: str | None = "http://127.0.0.1:9090",
    timeout_seconds: float = 5.0,
    settle_seconds: float = _DASER_METRICS_SETTLE_SECONDS,
) -> None:
    """Check DaseR metrics readiness and leave a short scrape window.

    Args:
        manifest: Started benchmark service manifest.
        phase: Human-readable probe phase printed with status lines.
        prometheus_url: Optional Prometheus base URL for scrape diagnostics.
        timeout_seconds: Maximum time to wait for DaseR ``/metrics``.
        settle_seconds: Extra delay after readiness so external Prometheus can
            scrape at least once when configured with a one-second interval.

    Thread-safety:
        Synchronous HTTP polling helper intended for the single benchmark
        orchestration process.
    """
    endpoint = manifest.endpoints.get("daser")
    if endpoint is None:
        return
    metrics_url = f"{endpoint.url}/metrics"
    _print_kv(f"daser_metrics_{phase}", metrics_url)
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(metrics_url, timeout=1.0)
            if response.status_code == 200 and "daser_up" in response.text:
                _print_kv(f"daser_metrics_{phase}_status", "ready")
                if settle_seconds > 0:
                    _print_kv("prometheus_scrape_wait_s", settle_seconds)
                    time.sleep(settle_seconds)
                _probe_prometheus_daser_scrape(prometheus_url, phase=phase)
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.25)
    _print_kv(f"daser_metrics_{phase}_status", f"unreachable ({last_error})")


def _probe_prometheus_daser_scrape(
    prometheus_url: str | None,
    *,
    phase: str,
) -> None:
    """Print Prometheus-side DaseR scrape diagnostics when available.

    Args:
        prometheus_url: Prometheus base URL, or an empty value to disable.
        phase: Benchmark phase used in printed keys.

    Thread-safety:
        Synchronous diagnostic helper intended for benchmark orchestration.
    """
    if not prometheus_url:
        return
    base_url = prometheus_url.rstrip("/")
    try:
        targets = httpx.get(f"{base_url}/api/v1/targets", timeout=2.0).json()
        target = _find_prometheus_daser_target(targets)
        if target is None:
            _print_kv(f"prometheus_daser_target_{phase}", "missing")
        else:
            health = target.get("health", "unknown")
            scrape_url = target.get("scrapeUrl", "unknown")
            last_error = target.get("lastError") or ""
            message = f"health={health} scrape_url={scrape_url}"
            if last_error:
                message = f"{message} last_error={last_error}"
            _print_kv(f"prometheus_daser_target_{phase}", message)

        up = httpx.get(
            f"{base_url}/api/v1/query",
            params={"query": 'daser_up{job="daser"}'},
            timeout=2.0,
        ).json()
        value = _first_prometheus_value(up)
        _print_kv(
            f"prometheus_daser_up_{phase}",
            "missing" if value is None else f"value={value}",
        )
    except Exception as exc:  # noqa: BLE001
        _print_kv(f"prometheus_daser_scrape_{phase}", f"unavailable ({exc})")


def _find_prometheus_daser_target(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the active Prometheus target with ``job=daser`` if present."""
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    targets = data.get("activeTargets")
    if not isinstance(targets, list):
        return None
    for target in targets:
        if not isinstance(target, dict):
            continue
        labels = target.get("labels")
        if isinstance(labels, dict) and labels.get("job") == "daser":
            return target
    return None


def _first_prometheus_value(payload: dict[str, Any]) -> str | None:
    """Return the first instant-query sample value from a Prometheus payload."""
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if not isinstance(result, list) or not result:
        return None
    first = result[0]
    if not isinstance(first, dict):
        return None
    value = first.get("value")
    if not isinstance(value, list) or len(value) < 2:
        return None
    return str(value[1])


def _cleanup(run_root: Path) -> None:
    for pid_file in run_root.glob("*/pids.json"):
        stop_from_pid_file(pid_file)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
