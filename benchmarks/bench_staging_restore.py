# SPDX-License-Identifier: Apache-2.0
"""Micro benchmark for DaseR staging-to-KV restore layouts."""

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
from daser.connector.staging import copy_staging_to_kv_cache  # noqa: E402
from daser.ops.rope_apply import clear_rope_apply_compile_cache  # noqa: E402


@dataclass(frozen=True)
class RestoreResult:
    """Timing result for one staging restore layout.

    Args:
        layout: Restore layout name.
        blocks: Number of KV blocks restored in one call.
        layers: Number of model layers.
        block_tokens: Number of tokens per KV block.
        heads: Number of KV heads.
        head_dim: Per-head hidden size.
        mean_us: Mean steady-state latency in microseconds.
        p50_us: Median steady-state latency in microseconds.
        p90_us: 90th percentile steady-state latency in microseconds.
        copy_calls: Number of high-level copy operations returned by the helper.
    """

    layout: str
    blocks: int
    layers: int
    block_tokens: int
    heads: int
    head_dim: int
    mean_us: float
    p50_us: float
    p90_us: float
    copy_calls: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.

    Async/thread-safety:
        Pure command-line parsing with no shared state.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=128)
    parser.add_argument("--pos-offset", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the staging restore micro benchmark."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("bench_staging_restore.py requires a CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    results = run_benchmark(
        device=device,
        dtype=dtype,
        blocks=args.blocks,
        layers=args.layers,
        block_tokens=args.block_tokens,
        heads=args.heads,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        pos_offset=args.pos_offset,
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
    device: torch.device,
    dtype: torch.dtype,
    blocks: int,
    layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    pos_offset: int,
    warmup: int,
    iters: int,
) -> list[RestoreResult]:
    """Run restore benchmarks for per-layer and cross-layer layouts.

    Args:
        device: CUDA device.
        dtype: KV dtype.
        blocks: Number of KV blocks restored per call.
        layers: Number of model layers.
        block_tokens: Number of tokens per KV block.
        heads: Number of KV heads.
        head_dim: Per-head hidden size.
        rotary_dim: Number of rotary dimensions.
        pos_offset: Relative RoPE position offset.
        warmup: Warmup iterations per layout.
        iters: Timed iterations per layout.

    Returns:
        Timing results for supported restore layouts.

    Async/thread-safety:
        Uses the current CUDA context. Do not run concurrently in one process.
    """
    staging, slot_size = _make_staging(
        device=device,
        dtype=dtype,
        blocks=blocks,
        layers=layers,
        block_tokens=block_tokens,
        heads=heads,
        head_dim=head_dim,
    )
    layer_names = [f"layer.{idx}" for idx in range(layers)]
    block_ids = list(range(blocks))
    results = []
    for layout in ("per_layer", "cross_layer"):
        clear_rope_apply_compile_cache()
        kv_caches = _make_kv_caches(
            layout=layout,
            device=device,
            dtype=dtype,
            blocks=blocks,
            layers=layers,
            block_tokens=block_tokens,
            heads=heads,
            head_dim=head_dim,
        )
        copy_calls = _restore_once(
            staging=staging,
            kv_caches=kv_caches,
            layer_names=layer_names,
            block_ids=block_ids,
            slot_size=slot_size,
            rotary_dim=rotary_dim,
            pos_offset=pos_offset,
        )
        torch.cuda.synchronize(device)
        timings = _time_restore(
            staging=staging,
            kv_caches=kv_caches,
            layer_names=layer_names,
            block_ids=block_ids,
            slot_size=slot_size,
            rotary_dim=rotary_dim,
            pos_offset=pos_offset,
            warmup=warmup,
            iters=iters,
        )
        results.append(
            RestoreResult(
                layout=layout,
                blocks=blocks,
                layers=layers,
                block_tokens=block_tokens,
                heads=heads,
                head_dim=head_dim,
                mean_us=statistics.fmean(timings),
                p50_us=statistics.median(timings),
                p90_us=_percentile(timings, 0.90),
                copy_calls=copy_calls,
            )
        )
    return results


def _make_staging(
    device: torch.device,
    dtype: torch.dtype,
    blocks: int,
    layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, int]:
    """Create slot-major staging bytes with realistic KV geometry."""
    logical = torch.randn(
        blocks,
        layers,
        2,
        block_tokens,
        heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    staging = logical.contiguous().view(torch.uint8)
    slot_size = logical[0].nbytes
    return staging, slot_size


def _make_kv_caches(
    layout: str,
    device: torch.device,
    dtype: torch.dtype,
    blocks: int,
    layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Create destination KV cache tensors for one restore layout."""
    if layout == "cross_layer":
        return {
            "__cross_layers__": torch.empty(
                blocks,
                layers,
                2,
                block_tokens,
                heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
        }
    return {
        f"layer.{idx}": torch.empty(
            2,
            blocks,
            block_tokens,
            heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        for idx in range(layers)
    }


def _restore_once(
    staging: torch.Tensor,
    kv_caches: dict[str, torch.Tensor],
    layer_names: list[str],
    block_ids: list[int],
    slot_size: int,
    rotary_dim: int,
    pos_offset: int,
) -> int:
    """Run one restore call and return the helper copy count."""
    return copy_staging_to_kv_cache(
        staging=staging,
        kv_caches=kv_caches,
        layer_names=layer_names,
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=pos_offset,
        rope_base=1000000.0,
        rope_rotary_dim=rotary_dim,
    )


def _time_restore(
    staging: torch.Tensor,
    kv_caches: dict[str, torch.Tensor],
    layer_names: list[str],
    block_ids: list[int],
    slot_size: int,
    rotary_dim: int,
    pos_offset: int,
    warmup: int,
    iters: int,
) -> list[float]:
    """Return per-iteration staging restore latency in microseconds."""
    for _ in range(warmup):
        _restore_once(
            staging,
            kv_caches,
            layer_names,
            block_ids,
            slot_size,
            rotary_dim,
            pos_offset,
        )
    torch.cuda.synchronize(staging.device)

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _restore_once(
            staging,
            kv_caches,
            layer_names,
            block_ids,
            slot_size,
            rotary_dim,
            pos_offset,
        )
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def _percentile(values: list[float], q: float) -> float:
    """Return nearest-rank percentile for a non-empty list."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def _print_results(results: list[RestoreResult]) -> None:
    """Print benchmark results as a markdown table."""
    print(
        "| layout | blocks | layers | block | heads | head_dim | "
        "mean us | p50 us | p90 us | copy calls |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.layout} | {result.blocks} | {result.layers} | "
            f"{result.block_tokens} | {result.heads} | {result.head_dim} | "
            f"{result.mean_us:.2f} | {result.p50_us:.2f} | "
            f"{result.p90_us:.2f} | {result.copy_calls} |"
        )


if __name__ == "__main__":
    main()
