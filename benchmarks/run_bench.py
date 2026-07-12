# SPDX-License-Identifier: Apache-2.0
"""Run end-to-end service benchmarks across backend comparison rows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils import vllm_bench
from benchmarks.utils.constants import BLOCK_TOKENS
from benchmarks.utils.datasets import add_dataset_cli_args
from benchmarks.utils.servers import BenchmarkManifest, stop_from_pid_file

_DASER_METRICS_SETTLE_SECONDS = 2.0
_BACKEND_CLEANUP_SETTLE_SECONDS = 2.0
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
        bench_input_len: vLLM bench random input length. In
            ``vllm-bench-prefix`` mode this is the final total prompt length,
            split between the shared prefix and random suffix.
        bench_output_len: vLLM bench random output length.
        bench_request_rate: vLLM bench request rate.
        bench_max_concurrency: vLLM bench max in-flight requests.
        bench_random_prefix_len: Fixed prefix length for vLLM random dataset.
        bench_prefix_ratio: Shared-prefix length ratio for vLLM prefix bench.
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
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    block_size: int = BLOCK_TOKENS
    max_inflight: int = 32
    gen_max_tokens: int = 128
    max_context_tokens: int = 0
    bench_num_prompts: int = 1000
    bench_input_len: int = 8192
    bench_output_len: int | None = None
    bench_request_rate: str = "inf"
    bench_max_concurrency: int | None = None
    bench_random_prefix_len: int = 0
    bench_prefix_ratio: float = 0.5
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
        choices=("internal", "vllm-bench", "vllm-bench-prefix"),
        default="internal",
    )
    add_dataset_cli_args(parser, default="longbench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--block-size", type=int, default=BLOCK_TOKENS)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--gen-max-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--bench-num-prompts", type=int, default=1000)
    parser.add_argument("--bench-input-len", type=int, default=8192)
    parser.add_argument("--bench-output-len", type=int, default=None)
    parser.add_argument("--bench-request-rate", default="inf")
    parser.add_argument("--bench-max-concurrency", type=int, default=None)
    parser.add_argument("--bench-random-prefix-len", type=int, default=0)
    parser.add_argument("--bench-prefix-ratio", type=float, default=0.5)
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
    parsed = RunBenchArgs(
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
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
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
        bench_prefix_ratio=args.bench_prefix_ratio,
        bench_random_range_ratio=args.bench_random_range_ratio,
        bench_seed=args.bench_seed,
        bench_burstiness=args.bench_burstiness,
        evict=args.evict,
        prometheus_url=args.prometheus_url,
    )
    try:
        _validate_run_args(parsed)
    except ValueError as exc:
        parser.error(str(exc))
    return parsed


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
    _validate_run_args(args)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.store_dir).expanduser() / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    prepare_path = run_root / "prepare.json"
    backend_runs = _expand_backend_runs(args.backend)
    _validate_backend_runs(backend_runs, load_generator=args.load_generator)

    _print_stage("prepare")
    _print_kv("load_generator", args.load_generator)
    if args.load_generator in ("vllm-bench", "vllm-bench-prefix"):
        _print_kv("dataset", "vllm-bench-random")
        _print_kv("bench_num_prompts", args.bench_num_prompts)
        _print_kv("bench_input_len", args.bench_input_len)
        _print_kv("bench_output_len", vllm_bench.bench_output_len(args))
        _print_kv("bench_random_prefix_len", vllm_bench.random_prefix_len(args))
    else:
        _print_kv("dataset", args.dataset)
        _print_kv("max_samples", args.max_samples)
    _print_kv("block_size", args.block_size)
    _print_kv("output", prepare_path)
    if args.load_generator in ("vllm-bench", "vllm-bench-prefix"):
        prepare = {"config": vllm_bench.prepare_config(args, run_root)}
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
    if load_generator not in ("vllm-bench", "vllm-bench-prefix"):
        return
    unsupported = [run.label for run in backend_runs if run.label == "daser-chunk"]
    if unsupported:
        raise ValueError(
            "vllm-bench load generator does not support daser-chunk; "
            "select baseline,lmcache,daser-prefix or use --load-generator internal"
        )
    if load_generator == "vllm-bench-prefix" and (
        not backend_runs or backend_runs[0].label != "baseline"
    ):
        raise ValueError(
            "vllm-bench-prefix requires baseline as the first backend row so "
            "LMCache and DaseR correctness can compare against baseline"
        )


def _validate_run_args(args: RunBenchArgs) -> None:
    """Validate benchmark runner arguments with clear preflight errors."""
    positive_ints = {
        "block_size": args.block_size,
        "max_num_seqs": args.max_num_seqs,
        "max_inflight": args.max_inflight,
        "gen_max_tokens": args.gen_max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    non_negative_ints = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_context_tokens": args.max_context_tokens,
    }
    for name, value in non_negative_ints.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.max_samples <= 0 and args.load_generator == "internal":
        raise ValueError("max_samples must be positive")
    if args.gpu_util <= 0.0 or args.gpu_util > 1.0:
        raise ValueError("gpu_util must be in (0, 1]")
    if args.load_generator in ("vllm-bench", "vllm-bench-prefix"):
        vllm_bench.validate_args(args)


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
    load_stage = (
        "prefix load"
        if args.load_generator == "vllm-bench-prefix"
        else "cold/warm load"
    )
    _print_stage(load_stage, backend_run.label)
    _print_kv("output", result_path)
    if args.load_generator in ("vllm-bench", "vllm-bench-prefix"):
        vllm_bench.run_load(
            args,
            manifest,
            backend_run,
            backend_dir,
            result_path,
            run_command=_run_command,
            print_kv=_print_kv,
        )
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
    _post_backend_settle()
    return result_path


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
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
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
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--block-size",
        str(args.block_size),
        "--l1-size",
        str(derived_l1),
        "--l2-size",
        str(derived_l2),
    ]
    if backend_run.backend == "daser":
        command.extend(["--cache-reuse-mode", backend_run.reuse_mode])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
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
    prefix_summary = _phase_summary(phases, "prefix")
    if prefix_summary is not None:
        _add_summary_fields(fields, "prefix", prefix_summary)
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
    if isinstance(exact_match, dict):
        prefix = "cold_warm_exact_match"
    else:
        exact_match = correctness.get("baseline_exact_match")
        prefix = "baseline_exact_match"
    if not isinstance(exact_match, dict):
        return
    available = exact_match.get("available")
    if available is not None:
        fields[f"{prefix}_available"] = available
    accuracy = exact_match.get("accuracy")
    if accuracy is not None:
        fields[f"{prefix}_accuracy"] = accuracy
    reason = exact_match.get("reason")
    if reason is not None:
        fields[f"{prefix}_reason"] = reason


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


def _post_backend_settle(seconds: float = _BACKEND_CLEANUP_SETTLE_SECONDS) -> None:
    """Give vLLM/CUDA subprocess teardown a short stabilization window."""
    if seconds > 0:
        time.sleep(seconds)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
