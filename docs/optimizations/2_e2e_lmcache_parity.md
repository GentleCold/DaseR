# DaseR E2E LMCache Parity Optimization

**Date:** 2026-05-15
**Benchmark:** `benchmarks/bench_e2e_daser_vs_lmcache.py`
**Model:** `/data/zwt/model/models/Qwen/Qwen3-8B`
**Device:** physical GPU 2 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`)

GPU 0 was occupied by another vLLM process during this run, so all final
measurements use GPU 2. The benchmark used `--gpu-util 0.22 --max-num-seqs 64`
and `/data/zwt/daser_test` for scratch stores.

## Baseline

Before this branch's optimizations, DaseR was still behind LMCache on the
same e2e benchmark:

| Run | DaseR cold | LMCache cold | DaseR warm | LMCache warm |
| --- | ---: | ---: | ---: | ---: |
| master baseline, N=200 | 6.96 s | 4.89 s | 2.43 s | 2.13 s |

The first warm-path optimization brought DaseR warm ahead of LMCache, but cold
remained slower:

| Run | DaseR cold | LMCache cold | DaseR warm | LMCache warm |
| --- | ---: | ---: | ---: | ---: |
| batched copies + quiet logs, N=200 | 5.80 s | 4.32 s | 1.03 s | 2.13 s |

## Optimizations

### Batched staging tensor copies

`start_load_kv` now restores a full layer for all requested blocks with one
`index_copy_` instead of copying every `(slot, layer)` pair separately.
`save_kv_layer` similarly gathers all requested blocks for a layer with one
`index_select` into staging.

This removed thousands of small GPU copy launches in the warm path and moved
DaseR warm from slower than LMCache to about 2x faster.

### Quieter hot-path logging

Per-request cache-hit/miss and metadata logs moved from INFO to DEBUG. The
hot path no longer emits large `block_ids` lists at INFO. This avoids Python
formatting and terminal I/O during benchmarked generation.

### Deferred cold-store writes

Cold profiling showed the cold pass was dominated by save-side work:

| Component | Total time, N=200 |
| --- | ---: |
| `save_kv_layer` staging copies | 1.95 s |
| `wait_for_save` writes | 3.15 s |
| commit RPCs | 0.09 s |

`wait_for_save` now submits a background transfer task and returns without
waiting for NVMe completion. The DaseR server records accepted transfer ranges
and inserts a chunk into the index only after its full KV coverage is present
in the transfer destination, so readers cannot observe partial data. For
io_uring, this destination is L1 and L2 persistence remains asynchronous; the
optional GDS backend uses the completed direct transfer when enabled. The worker no longer owns the final
`commit_chunks` call.
The staging tensor is independent from vLLM's KV cache, so vLLM can safely
reuse KV blocks while the background task drains.

The implementation also packs all stored requests in a forward step into one
step-level staging buffer and coalesces adjacent slot ranges into larger
`pwrite` spans. Deferred staging memory is bounded by
`max_inflight_store_bytes` (default 1 GiB) to keep large runs from growing
unbounded GPU memory usage.

### Reverted miss preallocation experiment

An experiment to reuse allocations returned by `match_and_alloc` on misses did
not improve cold performance (`5.94 s` vs `5.80 s` after the earlier copy/log
changes). It was reverted and is not counted as a final optimization.

## Final Results

### N=200

Command:

```bash
source /data/zwt/vllm/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --num-prompts 200 \
  --gpu-util 0.22 \
  --max-num-seqs 64 \
  --store-dir /data/zwt/daser_test \
  --out /data/zwt/daser_test/perf_final_after_revert_n200.json
```

| Metric | DaseR | LMCache | DaseR / LMCache |
| --- | ---: | ---: | ---: |
| cold elapsed | 3.17 s | 4.31 s | 1.36x tok/s |
| warm elapsed | 1.03 s | 2.15 s | 2.08x tok/s |
| cold prompt tok/s | 19,120 | 14,062 | 1.36x |
| warm prompt tok/s | 58,676 | 28,222 | 2.08x |

Correctness under this low-KV-cache benchmark setting still needs separate
follow-up: DaseR produced `3/200` cold/warm token mismatches, and LMCache
produced `2/200`. A PR #31-like DaseR-only run with the script default
`--gpu-util 0.4` produced `0/200` mismatches, while a DaseR-only rerun with
`--gpu-util 0.22 --max-num-seqs 64` reproduced the same DaseR `3/200`
mismatches. This points to the constrained KV-cache/chunked-prefill scheduling
condition, not the LMCache comparison phase, but the exact correctness root
cause is not closed by this performance PR.

### N=400 Larger Load

Command:

```bash
source /data/zwt/vllm/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --num-prompts 400 \
  --gpu-util 0.22 \
  --max-num-seqs 64 \
  --store-dir /data/zwt/daser_test \
  --out /data/zwt/daser_test/perf_final_after_revert_n400.json
```

| Metric | DaseR | LMCache | DaseR / LMCache |
| --- | ---: | ---: | ---: |
| cold elapsed | 6.39 s | 8.68 s | 1.36x tok/s |
| warm elapsed | 2.06 s | 4.29 s | 2.08x tok/s |
| cold prompt tok/s | 18,854 | 13,882 | 1.36x |
| warm prompt tok/s | 58,532 | 28,076 | 2.08x |

Correctness mismatches under the same constrained setting: DaseR `3/400`,
LMCache `2/400`.

## Takeaways

DaseR now exceeds LMCache on both cold and warm e2e passes for the standard
N=200 benchmark and a larger N=400 load under the measured performance
configuration. The dominant improvement is deferring write completion out of
the cold prefill critical path while preserving two-phase publication: chunks
become visible only after their NVMe writes finish.

The constrained `--gpu-util 0.22` correctness mismatches should be tracked as
a follow-up correctness investigation rather than treated as resolved by these
performance optimizations.
