# SPDX-License-Identifier: Apache-2.0
"""Micro benchmark for RoPE delta application backends."""

from __future__ import annotations

# Standard
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys

# Third Party
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# First Party
from daser.ops.rope_apply import (  # noqa: E402
    RopeApplyBackend,
    apply_rope_delta_to_key_block,
    clear_rope_apply_compile_cache,
)


@dataclass(frozen=True)
class BenchCase:
    """One RoPE benchmark input shape.

    Args:
        batch_blocks: Number of cache blocks transformed by one operator call.
        block_tokens: Number of tokens in one KV block.
        heads: Number of KV heads.
        head_dim: Per-head hidden size.
        rotary_dim: Rotary dimensions inside ``head_dim``.
    """

    batch_blocks: int
    block_tokens: int
    heads: int
    head_dim: int
    rotary_dim: int


@dataclass(frozen=True)
class BenchResult:
    """Timing result for one backend and input shape.

    Args:
        backend: Backend name.
        dtype: Tensor dtype name.
        batch_blocks: Number of cache blocks transformed by one operator call.
        block_tokens: Number of tokens in one KV block.
        heads: Number of KV heads.
        head_dim: Per-head hidden size.
        rotary_dim: Rotary dimensions inside ``head_dim``.
        mean_us: Mean steady-state latency in microseconds.
        p50_us: Median steady-state latency in microseconds.
        p90_us: 90th percentile steady-state latency in microseconds.
        speedup_vs_naive: Naive latency divided by backend latency.
        max_abs_diff: Maximum absolute difference versus naive output.
    """

    backend: str
    dtype: str
    batch_blocks: int
    block_tokens: int
    heads: int
    head_dim: int
    rotary_dim: int
    mean_us: float
    p50_us: float
    p90_us: float
    speedup_vs_naive: float
    max_abs_diff: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.

    Async/thread-safety:
        Pure command-line parsing with no shared state.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--backends",
        default="naive,compile,tilelang",
        help="Comma-separated backends: naive,compile,tilelang",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the RoPE apply micro benchmark."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("bench_rope_apply.py requires a CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    backends = _parse_backends(args.backends)
    cases = _default_cases()
    results = run_benchmark(
        cases=cases,
        backends=backends,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    _print_results(results)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )


def run_benchmark(
    cases: list[BenchCase],
    backends: list[RopeApplyBackend],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    """Run benchmark cases for requested backends.

    Args:
        cases: Input shape list.
        backends: Backends to run.
        dtype: Tensor dtype for benchmark inputs.
        device: CUDA device.
        warmup: Warmup iterations per backend and shape.
        iters: Timed iterations per backend and shape.

    Returns:
        Flat list of timing results.

    Async/thread-safety:
        Uses the current process CUDA context and clears the RoPE compile cache
        at case boundaries. Do not run concurrently in the same process.
    """
    results: list[BenchResult] = []
    for case in cases:
        base = torch.randn(
            _input_shape(case),
            dtype=dtype,
            device=device,
        )
        naive_ref = _apply_once(base, "naive", case.rotary_dim)
        naive_mean_us = 0.0
        case_results: list[BenchResult] = []

        for backend in backends:
            clear_rope_apply_compile_cache()
            _reset_torch_dynamo()
            actual = _apply_once(base, backend, case.rotary_dim)
            max_abs_diff = float((actual.float() - naive_ref.float()).abs().max())
            timings = _time_backend(
                base=base,
                backend=backend,
                rotary_dim=case.rotary_dim,
                warmup=warmup,
                iters=iters,
            )
            mean_us = statistics.fmean(timings)
            if backend == "naive":
                naive_mean_us = mean_us
            speedup = naive_mean_us / mean_us if naive_mean_us else 1.0
            case_results.append(
                BenchResult(
                    backend=backend,
                    dtype=str(dtype).replace("torch.", ""),
                    batch_blocks=case.batch_blocks,
                    block_tokens=case.block_tokens,
                    heads=case.heads,
                    head_dim=case.head_dim,
                    rotary_dim=case.rotary_dim,
                    mean_us=mean_us,
                    p50_us=statistics.median(timings),
                    p90_us=_percentile(timings, 0.90),
                    speedup_vs_naive=speedup,
                    max_abs_diff=max_abs_diff,
                )
            )
        results.extend(_fill_speedups(case_results))
    return results


def _parse_backends(raw: str) -> list[RopeApplyBackend]:
    """Parse and validate comma-separated backend names."""
    backends: list[RopeApplyBackend] = []
    for item in raw.split(","):
        backend = item.strip()
        if backend not in ("naive", "compile", "tilelang"):
            raise SystemExit(f"unsupported backend for this benchmark: {backend}")
        backends.append(backend)
    if "naive" not in backends:
        backends.insert(0, "naive")
    return backends


def _reset_torch_dynamo() -> None:
    """Reset torch.compile caches between independent benchmark cases."""
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if reset is not None:
        reset()


def _default_cases() -> list[BenchCase]:
    """Return representative RoPE apply benchmark shapes."""
    cases = []
    for batch_blocks in (1, 2, 4, 8, 16):
        cases.append(
            BenchCase(
                batch_blocks=batch_blocks,
                block_tokens=16,
                heads=8,
                head_dim=128,
                rotary_dim=128,
            )
        )
    for block_tokens in (32, 64):
        cases.append(
            BenchCase(
                batch_blocks=4,
                block_tokens=block_tokens,
                heads=8,
                head_dim=128,
                rotary_dim=128,
            )
        )
    for heads in (16, 32):
        cases.append(
            BenchCase(
                batch_blocks=4,
                block_tokens=16,
                heads=heads,
                head_dim=128,
                rotary_dim=128,
            )
        )
    for head_dim in (64,):
        cases.append(
            BenchCase(
                batch_blocks=4,
                block_tokens=16,
                heads=8,
                head_dim=head_dim,
                rotary_dim=head_dim,
            )
        )
    return cases


def _input_shape(case: BenchCase) -> tuple[int, ...]:
    """Return the tensor shape for a benchmark case."""
    if case.batch_blocks == 1:
        return (case.block_tokens, case.heads, case.head_dim)
    return (case.batch_blocks, case.block_tokens, case.heads, case.head_dim)


def _apply_once(
    base: torch.Tensor,
    backend: RopeApplyBackend,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply one backend to a cloned input and return the clone."""
    tensor = base.clone()
    apply_rope_delta_to_key_block(
        tensor,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=rotary_dim,
        is_neox_style=True,
        backend=backend,
    )
    torch.cuda.synchronize(tensor.device)
    return tensor


def _time_backend(
    base: torch.Tensor,
    backend: RopeApplyBackend,
    rotary_dim: int,
    warmup: int,
    iters: int,
) -> list[float]:
    """Return per-iteration latency in microseconds for one backend."""
    for _ in range(warmup):
        _apply_once(base, backend, rotary_dim)

    timings = []
    for _ in range(iters):
        tensor = base.clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        apply_rope_delta_to_key_block(
            tensor,
            delta=128,
            rope_base=1000000.0,
            rotary_dim=rotary_dim,
            is_neox_style=True,
            backend=backend,
        )
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def _percentile(values: list[float], q: float) -> float:
    """Return nearest-rank percentile for a non-empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def _fill_speedups(results: list[BenchResult]) -> list[BenchResult]:
    """Fill speedups after naive timing is known for a case."""
    naive = next((result.mean_us for result in results if result.backend == "naive"), 0)
    if naive <= 0:
        return results
    return [
        BenchResult(
            backend=result.backend,
            dtype=result.dtype,
            batch_blocks=result.batch_blocks,
            block_tokens=result.block_tokens,
            heads=result.heads,
            head_dim=result.head_dim,
            rotary_dim=result.rotary_dim,
            mean_us=result.mean_us,
            p50_us=result.p50_us,
            p90_us=result.p90_us,
            speedup_vs_naive=naive / result.mean_us,
            max_abs_diff=result.max_abs_diff,
        )
        for result in results
    ]


def _print_results(results: list[BenchResult]) -> None:
    """Print benchmark results as a compact table."""
    header = (
        "backend dtype batches block heads head_dim rotary mean_us p50_us "
        "p90_us speedup max_diff"
    )
    print(header)
    for result in results:
        print(
            "{backend:7s} {dtype:8s} {batch_blocks:7d} {block_tokens:5d} "
            "{heads:5d} {head_dim:8d} {rotary_dim:6d} {mean_us:8.2f} "
            "{p50_us:7.2f} {p90_us:7.2f} {speedup_vs_naive:7.2f} "
            "{max_abs_diff:.3e}".format(**asdict(result))
        )


if __name__ == "__main__":
    main()
