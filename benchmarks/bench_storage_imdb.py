# SPDX-License-Identifier: Apache-2.0
"""Cold/warm NVMe storage benchmark: DaseR (kvikio) vs LMCache (LocalDiskBackend).

Simulates KV-cache save/load patterns derived from IMDB review lengths.

Usage:
    source /data/zwt/vllm/bin/activate
    python benchmarks/bench_storage_imdb.py [--num-chunks 100] [--store-dir /data/zwt/daser_test]
                                            [--daser-threads 4]
"""

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from typing import Any

# Third Party
import cupy
import torch

# First Party - add project root so we can import daser without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from daser.connector.gds_transfer import GDSTransferLayer, TransferBackend


# ---------------------------------------------------------------------------
# Constants — mimic a realistic LLM KV layout
# ---------------------------------------------------------------------------
NUM_LAYERS = 32  # Llama-3-8B / similar
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_TOKENS = 16       # tokens per vLLM KV block
KV_DTYPE = torch.bfloat16

# Bytes per block per layer: 2 (K+V) * BLOCK_TOKENS * NUM_KV_HEADS * HEAD_DIM * dtype_bytes
_BYTES_PER_LAYER = 2 * BLOCK_TOKENS * NUM_KV_HEADS * HEAD_DIM * 2  # 65 536 B
SLOT_SIZE = NUM_LAYERS * _BYTES_PER_LAYER                           # 2 097 152 B = 2 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) of a sorted list."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _sample_imdb(path: str, n: int = 1000) -> list[int]:
    """Return block-counts derived from first n IMDB review token estimates."""
    counts: list[int] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(counts) >= n:
                break
            words = len(row["review"].split())
            # 1.3 tokens/word estimate; align to block_tokens
            tokens = max(BLOCK_TOKENS, round(words * 1.3 / BLOCK_TOKENS) * BLOCK_TOKENS)
            counts.append(tokens // BLOCK_TOKENS)  # number of blocks
    return counts


def _make_slot_gpu() -> cupy.ndarray:
    """Allocate one slot worth of data on GPU (random bytes)."""
    return cupy.random.randint(0, 256, SLOT_SIZE, dtype=cupy.uint8)


def _make_slot_cpu() -> torch.Tensor:
    """Allocate one slot worth of data on CPU."""
    return torch.randint(0, 256, (SLOT_SIZE,), dtype=torch.uint8)


def _alloc_store(path: str, num_slots: int) -> None:
    """Pre-allocate a DaseR store file."""
    size = num_slots * SLOT_SIZE
    with open(path, "wb") as f:
        f.truncate(size)


# ---------------------------------------------------------------------------
# DaseR benchmark
# ---------------------------------------------------------------------------

async def _daser_write_all(
    gds: GDSTransferLayer, bufs: list[cupy.ndarray]
) -> tuple[float, list[float]]:
    """Returns (total_elapsed, per_op_latency_ms_list)."""
    op_latencies: list[float] = []
    lock = asyncio.Lock()

    async def _timed_write(buf: cupy.ndarray, offset: int) -> None:
        t = time.perf_counter()
        await gds.write_async(buf, file_offset=offset)
        lat = (time.perf_counter() - t) * 1000
        async with lock:
            op_latencies.append(lat)

    t0 = time.perf_counter()
    tasks = [
        asyncio.ensure_future(_timed_write(buf, i * SLOT_SIZE))
        for i, buf in enumerate(bufs)
    ]
    await asyncio.gather(*tasks)
    return time.perf_counter() - t0, op_latencies


async def _daser_read_all(
    gds: GDSTransferLayer, num: int
) -> tuple[float, list[cupy.ndarray], list[float]]:
    """Returns (total_elapsed, dst_bufs, per_op_latency_ms_list)."""
    dst_bufs = [cupy.empty(SLOT_SIZE, dtype=cupy.uint8) for _ in range(num)]
    op_latencies: list[float] = []
    lock = asyncio.Lock()

    async def _timed_read(buf: cupy.ndarray, offset: int) -> None:
        t = time.perf_counter()
        await gds.read_into_async(buf, file_offset=offset)
        lat = (time.perf_counter() - t) * 1000
        async with lock:
            op_latencies.append(lat)

    t0 = time.perf_counter()
    tasks = [
        asyncio.ensure_future(_timed_read(buf, i * SLOT_SIZE))
        for i, buf in enumerate(dst_bufs)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    return elapsed, dst_bufs, op_latencies


def _bench_daser(store_path: str, num_chunks: int, nthreads: int = 4) -> dict[str, Any]:
    print(f"\n[DaseR] Backend: kvikio — writing {num_chunks} slots × {SLOT_SIZE/1024/1024:.1f} MB "
          f"(threads={nthreads})")

    _alloc_store(store_path, num_chunks)
    bufs = [_make_slot_gpu() for _ in range(num_chunks)]
    total_bytes = num_chunks * SLOT_SIZE

    gds = GDSTransferLayer(store_path, nthreads=nthreads)
    print(f"[DaseR] Active backend: {gds.backend.name}")

    # --- cold write ---
    write_elapsed, write_lats = asyncio.run(_daser_write_all(gds, bufs))
    write_gbps = total_bytes / write_elapsed / 1e9
    print(f"[DaseR] cold write: {write_gbps:.3f} GB/s  ({write_elapsed*1000:.1f} ms total)  "
          f"p50={_percentile(write_lats,50):.1f}ms  "
          f"p95={_percentile(write_lats,95):.1f}ms  "
          f"p99={_percentile(write_lats,99):.1f}ms")

    # --- cold read ---
    read_elapsed_cold, read_bufs_cold, read_cold_lats = asyncio.run(_daser_read_all(gds, num_chunks))
    read_gbps_cold = total_bytes / read_elapsed_cold / 1e9
    print(f"[DaseR] cold read : {read_gbps_cold:.3f} GB/s  ({read_elapsed_cold*1000:.1f} ms total)  "
          f"p50={_percentile(read_cold_lats,50):.1f}ms  "
          f"p95={_percentile(read_cold_lats,95):.1f}ms  "
          f"p99={_percentile(read_cold_lats,99):.1f}ms")

    # --- warm read (OS page cache likely warm) ---
    read_elapsed_warm, read_bufs_warm, read_warm_lats = asyncio.run(_daser_read_all(gds, num_chunks))
    read_gbps_warm = total_bytes / read_elapsed_warm / 1e9
    print(f"[DaseR] warm read : {read_gbps_warm:.3f} GB/s  ({read_elapsed_warm*1000:.1f} ms total)  "
          f"p50={_percentile(read_warm_lats,50):.1f}ms  "
          f"p95={_percentile(read_warm_lats,95):.1f}ms  "
          f"p99={_percentile(read_warm_lats,99):.1f}ms")

    # Correctness spot-check
    match = bool(cupy.array_equal(bufs[0], read_bufs_cold[0]))
    print(f"[DaseR] data integrity check: {'OK' if match else 'FAIL'}")

    gds.close()
    return {
        "system": "DaseR",
        "backend": gds.backend.name,
        "nthreads": nthreads,
        "num_chunks": num_chunks,
        "chunk_bytes": SLOT_SIZE,
        "total_gb": total_bytes / 1e9,
        "write_gbps": write_gbps,
        "read_cold_gbps": read_gbps_cold,
        "read_warm_gbps": read_gbps_warm,
        "write_ms": write_elapsed * 1000,
        "read_cold_ms": read_elapsed_cold * 1000,
        "read_warm_ms": read_elapsed_warm * 1000,
        "write_p50_ms": _percentile(write_lats, 50),
        "write_p95_ms": _percentile(write_lats, 95),
        "write_p99_ms": _percentile(write_lats, 99),
        "read_cold_p50_ms": _percentile(read_cold_lats, 50),
        "read_cold_p95_ms": _percentile(read_cold_lats, 95),
        "read_cold_p99_ms": _percentile(read_cold_lats, 99),
        "read_warm_p50_ms": _percentile(read_warm_lats, 50),
        "read_warm_p95_ms": _percentile(read_warm_lats, 95),
        "read_warm_p99_ms": _percentile(read_warm_lats, 99),
    }


# ---------------------------------------------------------------------------
# LMCache benchmark (LocalDiskBackend)
# ---------------------------------------------------------------------------

def _bench_lmcache(store_dir: str, num_chunks: int) -> dict[str, Any]:
    """Benchmark LMCache LocalDiskBackend write + read."""
    try:
        # Third Party
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import (
            AdHocMemoryAllocator,
            MemoryFormat,
            MemoryObjMetadata,
            TensorMemoryObj,
        )
        from lmcache.v1.metadata import LMCacheMetadata
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
    except ImportError as exc:
        print(f"\n[LMCache] SKIPPED: {exc}")
        return {"system": "LMCache", "skipped": True, "reason": str(exc)}

    # LMCache chunk shape: (2, BLOCK_TOKENS, NUM_KV_HEADS, HEAD_DIM) per layer
    # We bundle all layers → (NUM_LAYERS, 2, BLOCK_TOKENS, NUM_KV_HEADS, HEAD_DIM)
    # but LocalDiskBackend works per-chunk, so use a flat shape matching SLOT_SIZE
    lm_shape = torch.Size([NUM_LAYERS, 2, BLOCK_TOKENS, NUM_KV_HEADS, HEAD_DIM])
    lm_dtype = KV_DTYPE
    total_bytes = num_chunks * SLOT_SIZE

    print(f"\n[LMCache] writing {num_chunks} chunks × {SLOT_SIZE/1024/1024:.1f} MB")

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    metadata = LMCacheMetadata(
        model_name="benchmark",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=lm_dtype,
        kv_shape=(NUM_LAYERS, 2, BLOCK_TOKENS, NUM_KV_HEADS, HEAD_DIM),
    )
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=BLOCK_TOKENS,
        local_cpu=True,
        max_local_cpu_size=0.5,
        lmcache_instance_id="bench",
    )
    config.local_disk = store_dir
    config.max_local_disk_size = max(5.0, total_bytes / 1e9 * 3)

    allocator = AdHocMemoryAllocator(device="cpu")
    local_cpu = LocalCPUBackend(
        config=config, metadata=metadata,
        dst_device="cpu", memory_allocator=allocator,
    )
    backend = LocalDiskBackend(
        config=config, loop=loop,
        local_cpu_backend=local_cpu,
        dst_device="cpu", metadata=metadata,
    )

    keys = [CacheEngineKey("benchmark", 1, 0, i, lm_dtype) for i in range(num_chunks)]

    # Build CPU memory objects
    objs = []
    for _ in range(num_chunks):
        obj = allocator.allocate([lm_shape], [lm_dtype], fmt=MemoryFormat.KV_T2D)
        assert obj is not None
        obj.tensor.fill_(7)
        objs.append(obj)

    # --- cold write ---
    completed = 0
    lock = threading.Lock()
    done_evt = threading.Event()
    write_lats: list[float] = []
    write_start_times: dict[Any, float] = {}

    t_batch_start = time.perf_counter()
    for k in keys:
        write_start_times[k] = time.perf_counter()

    def on_done(key: CacheEngineKey) -> None:
        nonlocal completed
        lat = (time.perf_counter() - write_start_times.get(key, t_batch_start)) * 1000
        with lock:
            write_lats.append(lat)
            completed += 1
            if completed >= num_chunks:
                done_evt.set()

    t0 = time.perf_counter()
    backend.batched_submit_put_task(keys, objs, on_complete_callback=on_done)
    done_evt.wait(timeout=300)
    write_elapsed = time.perf_counter() - t0
    write_gbps = total_bytes / write_elapsed / 1e9
    print(f"[LMCache] cold write: {write_gbps:.3f} GB/s  ({write_elapsed*1000:.1f} ms total)  "
          f"p50={_percentile(write_lats,50):.1f}ms  "
          f"p95={_percentile(write_lats,95):.1f}ms  "
          f"p99={_percentile(write_lats,99):.1f}ms")

    # --- cold read (parallel blocking reads from disk) ---
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _read_one_timed(key: Any) -> tuple[Any, float]:
        t = time.perf_counter()
        obj = backend.get_blocking(key)
        lat = (time.perf_counter() - t) * 1000
        return obj, lat

    t0 = time.perf_counter()
    with _TPE(max_workers=8) as ex:
        cold_pairs = list(ex.map(_read_one_timed, keys))
    read_elapsed_cold = time.perf_counter() - t0
    read_cold_lats = [lat for _, lat in cold_pairs]
    for obj, _ in cold_pairs:
        if obj is not None:
            try:
                obj.ref_count_down()
            except Exception:
                pass
    read_gbps_cold = total_bytes / read_elapsed_cold / 1e9
    print(f"[LMCache] cold read : {read_gbps_cold:.3f} GB/s  ({read_elapsed_cold*1000:.1f} ms total)  "
          f"p50={_percentile(read_cold_lats,50):.1f}ms  "
          f"p95={_percentile(read_cold_lats,95):.1f}ms  "
          f"p99={_percentile(read_cold_lats,99):.1f}ms")

    # --- warm read (repeat; OS page cache likely hot) ---
    t0 = time.perf_counter()
    with _TPE(max_workers=8) as ex:
        warm_pairs = list(ex.map(_read_one_timed, keys))
    read_elapsed_warm = time.perf_counter() - t0
    read_warm_lats = [lat for _, lat in warm_pairs]
    for obj, _ in warm_pairs:
        if obj is not None:
            try:
                obj.ref_count_down()
            except Exception:
                pass
    read_gbps_warm = total_bytes / read_elapsed_warm / 1e9
    print(f"[LMCache] warm read : {read_gbps_warm:.3f} GB/s  ({read_elapsed_warm*1000:.1f} ms total)  "
          f"p50={_percentile(read_warm_lats,50):.1f}ms  "
          f"p95={_percentile(read_warm_lats,95):.1f}ms  "
          f"p99={_percentile(read_warm_lats,99):.1f}ms")

    # cleanup
    try:
        backend.disk_worker.close()
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)
    loop.close()

    return {
        "system": "LMCache",
        "backend": "LocalDiskBackend",
        "num_chunks": num_chunks,
        "chunk_bytes": SLOT_SIZE,
        "total_gb": total_bytes / 1e9,
        "write_gbps": write_gbps,
        "read_cold_gbps": read_gbps_cold,
        "read_warm_gbps": read_gbps_warm,
        "write_ms": write_elapsed * 1000,
        "read_cold_ms": read_elapsed_cold * 1000,
        "read_warm_ms": read_elapsed_warm * 1000,
        "write_p50_ms": _percentile(write_lats, 50),
        "write_p95_ms": _percentile(write_lats, 95),
        "write_p99_ms": _percentile(write_lats, 99),
        "read_cold_p50_ms": _percentile(read_cold_lats, 50),
        "read_cold_p95_ms": _percentile(read_cold_lats, 95),
        "read_cold_p99_ms": _percentile(read_cold_lats, 99),
        "read_warm_p50_ms": _percentile(read_warm_lats, 50),
        "read_warm_p95_ms": _percentile(read_warm_lats, 95),
        "read_warm_p99_ms": _percentile(read_warm_lats, 99),
    }


# ---------------------------------------------------------------------------
# IMDB profiling summary
# ---------------------------------------------------------------------------

def _imdb_summary(imdb_path: str) -> dict[str, Any]:
    blocks = _sample_imdb(imdb_path, n=1000)
    return {
        "sampled_reviews": len(blocks),
        "avg_blocks_per_review": statistics.mean(blocks),
        "median_blocks": statistics.median(blocks),
        "p90_blocks": sorted(blocks)[int(len(blocks) * 0.9)],
        "block_tokens": BLOCK_TOKENS,
        "slot_bytes": SLOT_SIZE,
        "slot_mb": SLOT_SIZE / 1024 / 1024,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DaseR vs LMCache storage benchmark")
    parser.add_argument("--num-chunks", type=int, default=100,
                        help="Number of KV slots/chunks to write and read (default: 100)")
    parser.add_argument("--store-dir", default="/data/zwt/daser_test",
                        help="Directory for benchmark store files")
    parser.add_argument("--imdb", default="/data/zwt/imdb.csv",
                        help="Path to IMDB CSV dataset")
    parser.add_argument("--daser-threads", type=int, default=4,
                        help="kvikio thread-pool size for DaseR compat mode (default: 4)")
    parser.add_argument("--skip-lmcache", action="store_true",
                        help="Skip LMCache benchmark (e.g. not installed)")
    parser.add_argument("--out", default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    os.makedirs(args.store_dir, exist_ok=True)

    print("=" * 70)
    print("DaseR vs LMCache — NVMe KV Storage Benchmark")
    print("=" * 70)

    # IMDB stats
    if os.path.exists(args.imdb):
        summary = _imdb_summary(args.imdb)
        print(f"\nIMDB dataset stats ({summary['sampled_reviews']} reviews sampled):")
        print(f"  avg blocks/review : {summary['avg_blocks_per_review']:.1f} × {BLOCK_TOKENS} tokens")
        print(f"  median            : {summary['median_blocks']:.0f} blocks")
        print(f"  P90               : {summary['p90_blocks']} blocks")
        print(f"  slot size         : {summary['slot_mb']:.1f} MB  "
              f"({NUM_LAYERS}L × {_BYTES_PER_LAYER//1024}KB/layer)")
    else:
        summary = {}
        print(f"\nIMDB not found at {args.imdb}, skipping stats")

    print(f"\nBenchmark config: {args.num_chunks} chunks × {SLOT_SIZE/1024/1024:.1f} MB "
          f"= {args.num_chunks * SLOT_SIZE / 1e9:.2f} GB total")

    # --- DaseR ---
    daser_store = os.path.join(args.store_dir, "bench_daser.store")
    daser_result = _bench_daser(daser_store, args.num_chunks, nthreads=args.daser_threads)

    # --- LMCache ---
    lmcache_dir = os.path.join(args.store_dir, "bench_lmcache")
    os.makedirs(lmcache_dir, exist_ok=True)
    if args.skip_lmcache:
        lmcache_result = {"system": "LMCache", "skipped": True, "reason": "--skip-lmcache"}
    else:
        lmcache_result = _bench_lmcache(lmcache_dir, args.num_chunks)

    # --- Report ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'DaseR':>12} {'LMCache':>12}")
    print("-" * 56)

    def _fmt(r: dict, key: str) -> str:
        if r.get("skipped"):
            return "SKIPPED"
        val = r.get(key)
        if val is None:
            return "N/A"
        if "gbps" in key:
            return f"{val:.3f} GB/s"
        if "ms" in key:
            return f"{val:.0f} ms"
        return str(val)

    for label, key in [
        ("Write throughput", "write_gbps"),
        ("Cold read throughput", "read_cold_gbps"),
        ("Warm read throughput", "read_warm_gbps"),
        ("Write latency (total)", "write_ms"),
        ("Cold read latency (total)", "read_cold_ms"),
        ("Warm read latency (total)", "read_warm_ms"),
    ]:
        print(f"  {label:<28} {_fmt(daser_result, key):>12} {_fmt(lmcache_result, key):>12}")

    if not lmcache_result.get("skipped"):
        speedup_cold = (
            daser_result.get("read_cold_gbps", 0)
            / max(lmcache_result.get("read_cold_gbps", 1e-9), 1e-9)
        )
        speedup_warm = (
            daser_result.get("read_warm_gbps", 0)
            / max(lmcache_result.get("read_warm_gbps", 1e-9), 1e-9)
        )
        print(f"\n  DaseR cold read speedup : {speedup_cold:.2f}×")
        print(f"  DaseR warm read speedup : {speedup_warm:.2f}×")

    print(f"\n  DaseR IO backend        : {daser_result.get('backend', '?')}")
    if not lmcache_result.get("skipped"):
        print(f"  LMCache IO backend      : {lmcache_result.get('backend', '?')}")
    print("=" * 70)

    # JSON output
    results = {
        "config": {
            "num_chunks": args.num_chunks,
            "daser_threads": args.daser_threads,
            "slot_bytes": SLOT_SIZE,
            "num_layers": NUM_LAYERS,
            "block_tokens": BLOCK_TOKENS,
        },
        "imdb_stats": summary,
        "daser": daser_result,
        "lmcache": lmcache_result,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}")
    else:
        print("\nJSON results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
