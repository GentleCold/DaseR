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

from benchmarks.utils.servers import BenchmarkManifest, stop_from_pid_file

_DASER_METRICS_SETTLE_SECONDS = 2.0


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
        prometheus_url: Optional Prometheus base URL for scrape diagnostics.

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
    prometheus_url: str = "http://127.0.0.1:9090"


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

    _print_stage("prepare")
    _print_kv("dataset", args.dataset)
    _print_kv("max_samples", args.max_samples)
    _print_kv("output", prepare_path)
    _run_command(_prepare_command(args, run_root, prepare_path))
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    config = prepare["config"]
    derived_l1 = int(config["derived_l1_size_bytes"])
    derived_l2 = int(config["derived_l2_size_bytes"])
    _print_kv("derived_l1_bytes", derived_l1)
    _print_kv("derived_l2_bytes", derived_l2)

    result_paths: list[tuple[BackendRun, Path]] = []
    try:
        for backend_run in _expand_backend_runs(
            args.backend, default_reuse_mode=args.cache_reuse_mode
        ):
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
            BackendRun("baseline", "vllm", "none"),
            BackendRun("lmcache", "lmcache", "none"),
            BackendRun("daser-chunk", "daser", "chunk"),
            BackendRun("daser-prefix", "daser", "prefix"),
        ]
    if backend in ("baseline", "vllm"):
        return [BackendRun("baseline", "vllm", "none")]
    if backend == "lmcache":
        return [BackendRun("lmcache", "lmcache", "none")]
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
