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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.utils.servers import stop_from_pid_file


@dataclass(frozen=True)
class BackendRun:
    """Resolved benchmark backend row.

    Args:
        label: Directory and report label for this benchmark row.
        backend: Backend name accepted by ``bench_start_servers.py``.
        reuse_mode: Cache reuse mode passed to DaseR-compatible services.

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
        backend: Requested backend row or ``all``.
        cache_reuse_mode: Compatibility reuse mode for ``--backend daser``.
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
        max_inflight: HTTP load generator concurrency.
        gen_max_tokens: Maximum generated tokens.
        max_context_tokens: Prompt token ceiling; 0 infers from model metadata.
        evict: Whether to enable L2 and eviction sizing.

    Thread-safety:
        Immutable value object.
    """

    backend: str = "all"
    cache_reuse_mode: str = "chunk"
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
    max_inflight: int = 32
    gen_max_tokens: int = 128
    max_context_tokens: int = 0
    evict: bool = False


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
        choices=(
            "all",
            "baseline",
            "vllm",
            "lmcache",
            "daser",
            "daser-chunk",
            "daser-prefix",
        ),
        default="all",
    )
    parser.add_argument(
        "--cache-reuse-mode", choices=("chunk", "prefix"), default="chunk"
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
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--gen-max-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--evict", action="store_true")
    args = parser.parse_args(argv)
    return RunBenchArgs(
        backend=args.backend,
        cache_reuse_mode=args.cache_reuse_mode,
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
        max_inflight=args.max_inflight,
        gen_max_tokens=args.gen_max_tokens,
        max_context_tokens=args.max_context_tokens,
        evict=args.evict,
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

    print(
        f"[bench] prepare workload dataset={args.dataset} "
        f"max_samples={args.max_samples} out={prepare_path}",
        flush=True,
    )
    _run_command(_prepare_command(args, run_root, prepare_path))
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    config = prepare["config"]
    derived_l1 = int(config["derived_l1_size_bytes"])
    derived_l2 = int(config["derived_l2_size_bytes"])
    print(
        f"[bench] prepared l1_bytes={derived_l1} l2_bytes={derived_l2}",
        flush=True,
    )

    try:
        for backend_run in _expand_backend_runs(
            args.backend, default_reuse_mode=args.cache_reuse_mode
        ):
            _run_backend(
                args,
                run_id,
                run_root,
                prepare_path,
                backend_run,
                derived_l1,
                derived_l2,
            )
    finally:
        _cleanup(run_root)

    print(f"run_root={run_root}", flush=True)
    return run_root


def _expand_backend_runs(backend: str, *, default_reuse_mode: str) -> list[BackendRun]:
    """Resolve a requested backend into concrete benchmark rows.

    Args:
        backend: User-facing backend name.
        default_reuse_mode: Reuse mode for compatibility ``daser`` requests.

    Returns:
        Concrete backend rows in execution order.

    Thread-safety:
        Pure helper.
    """
    if backend == "all":
        return [
            BackendRun("baseline", "vllm", default_reuse_mode),
            BackendRun("lmcache", "lmcache", default_reuse_mode),
            BackendRun("daser-chunk", "daser", "chunk"),
            BackendRun("daser-prefix", "daser", "prefix"),
        ]
    if backend in ("baseline", "vllm"):
        return [BackendRun("baseline", "vllm", default_reuse_mode)]
    if backend == "lmcache":
        return [BackendRun("lmcache", "lmcache", default_reuse_mode)]
    if backend == "daser":
        return [BackendRun("daser", "daser", default_reuse_mode)]
    if backend == "daser-chunk":
        return [BackendRun("daser-chunk", "daser", "chunk")]
    if backend == "daser-prefix":
        return [BackendRun("daser-prefix", "daser", "prefix")]
    raise ValueError(f"unknown backend: {backend}")


def _run_backend(
    args: RunBenchArgs,
    run_id: str,
    run_root: Path,
    prepare_path: Path,
    backend_run: BackendRun,
    derived_l1: int,
    derived_l2: int,
) -> None:
    """Start one backend, run load, print a summary, and clean services."""
    backend_dir = run_root / backend_run.label
    backend_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[bench] start backend "
        f"label={backend_run.label} backend={backend_run.backend} "
        f"reuse={backend_run.reuse_mode} dir={backend_dir}",
        flush=True,
    )
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

    result_path = backend_dir / "results.json"
    print(
        f"[bench] load backend label={backend_run.label} out={result_path}",
        flush=True,
    )
    _run_command(_load_command(args, backend_dir, prepare_path, result_path))
    _print_result_summary(backend_run.label, result_path)
    _cleanup(run_root)


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
        "--cache-reuse-mode",
        args.cache_reuse_mode,
        "--dataset",
        args.dataset,
        "--max-samples",
        str(args.max_samples),
        "--max-inflight",
        str(args.max_inflight),
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
        "--l1-size",
        str(derived_l1),
        "--l2-size",
        str(derived_l2),
        "--cache-reuse-mode",
        backend_run.reuse_mode,
    ]
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


def _print_result_summary(label: str, result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = _summary_from_result(result)
    if not summary:
        print(f"[bench] summary label={label} results={result_path}", flush=True)
        return
    fields = {
        "ttft_ms_mean": summary.get("ttft_ms_mean"),
        "latency_ms_mean": summary.get("latency_ms_mean"),
        "phase_prompt_tok_per_s": summary.get("phase_prompt_tok_per_s"),
        "backend_server_cache_hit_rate": summary.get("backend_server_cache_hit_rate"),
    }
    rendered = " ".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    print(
        f"[bench] summary label={label} {rendered} results={result_path}".rstrip(),
        flush=True,
    )


def _summary_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    phases = result.get("result")
    if not isinstance(phases, dict):
        return None
    for phase_name in ("warm", "baseline", "cold"):
        phase = phases.get(phase_name)
        if isinstance(phase, dict) and isinstance(phase.get("summary"), dict):
            return phase["summary"]
    return None


def _run_command(command: list[str]) -> None:
    """Print and run a benchmark subprocess command.

    Args:
        command: Command argv.

    Thread-safety:
        Reentrant, but subprocess effects depend on the invoked command.
    """
    print(f"[bench] run: {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _cleanup(run_root: Path) -> None:
    for pid_file in run_root.glob("*/pids.json"):
        stop_from_pid_file(pid_file)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
