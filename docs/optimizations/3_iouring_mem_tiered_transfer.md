# DaseR io_uring Memory Tier Optimization

**Date:** 2026-05-19
**Branch:** `feat/iouring-mem-tiered-transfer`
**Benchmark:** `benchmarks/bench_e2e_daser_vs_lmcache.py`
**Model:** Qwen3-8B, bfloat16
**Device:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`

This record documents the `iouring-mem` transfer backend and the follow-up
hot-path optimizations. The comparison target is LMCache SSD + local CPU mode,
with LMCache disk and CPU capacities aligned to DaseR's store and L1 sizes.

## Implementation Summary

The backend adds a three-level path:

- L0: vLLM GPU KV cache.
- L1: DaseR pinned host-memory cache with LRU eviction. Entries returned by
  lookup are pinned during load and cannot be evicted until release.
- L2: existing DaseR SSD ring-buffer store.

Writes publish to L1 first, then persist to L2:

1. Reserve a pinned L1 entry with a durable-write pin.
2. Copy the staged GPU bytes into the L1 entry.
3. Commit L1 residency over IPC so same-process warm reads can hit memory.
4. Write L1 bytes to the SSD store through native io_uring.
5. Mark the entry durable, release the durable pin, then commit L2 durability.

Reads prefer L1 and fall back to L2 only when the server says the chunk is
durable in L2. L1 eviction uses an explicit policy interface; the current
policy is LRU.

## Native io_uring Status

The production `iouring-mem` path does not use a pread/pwrite fallback. It uses
`daser/connector/transfer/iouring/native.py`, a minimal Python `ctypes` wrapper over the
Linux `io_uring_setup` and `io_uring_enter` syscalls:

- SQ/CQ rings and SQEs are mmap'd directly.
- Each operation fills one SQE with `IORING_OP_READ` or `IORING_OP_WRITE`.
- The wrapper waits for the matching CQE and returns the kernel result.
- The async transfer layer runs the blocking syscall wrapper in the event-loop
  default executor and serializes access with an asyncio lock.

`PreadPwriteTestEngine` remains only for explicit unit-test injection, so tests
can exercise L1 behavior without depending on kernel io_uring availability.

This is not a C++/liburing implementation yet. A C++ extension is still a
reasonable next optimization if we want:

- batched SQE submission/completion,
- registered buffers or fixed files,
- less Python/ctypes overhead,
- deeper queue depth without one-at-a-time locking.

The current implementation was chosen to remove the fallback dependency while
keeping the patch self-contained and easy to test.

## Profiling Method

Connector micro-profiling is guarded by `DASER_PROFILE_CONNECTOR=1`. The
instrumentation logs coarse per-step timings from the vLLM worker process:

- save pack time in `save_kv_layer`,
- save submission time in `wait_for_save`,
- background `write_chunk_async` elapsed time,
- load read time,
- direct/merge/fallback KV copy time,
- number of GPU copy calls.

Example profiling command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
DASER_PROFILE_CONNECTOR=1 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --num-prompts 20 \
  --max-num-seqs 16 \
  --gpu-util 0.35 \
  --daser-transfer-backend iouring-mem \
  --daser-l1-cache-size 16gb \
  --skip-lmcache \
  --out <scratch>/iouring_native_preindex_profile_n20.json \
  > <scratch>/iouring_native_preindex_profile_n20.log 2>&1
```

The profile showed that the cold foreground path was dominated by save-side
packing and staging work, not by waiting for SSD writes. Background
`write_chunk_async` logs often appear after the benchmark's cold elapsed line,
because persistence is deferred behind the two-phase publication semantics.

For the N=20 profiled run after block-index reuse:

| Pack block count | Layer calls | Total pack time |
| ---: | ---: | ---: |
| 172 | 36 | 15.3 ms |
| 37 | 36 | 2.1 ms |
| 24 | 36 | 1.4 ms |
| 10 | 36 | 12.2 ms |
| 2 | 36 | 1.5 ms |

Total measured save pack time was about 32 ms. This reduced the earlier large
per-layer index construction overhead, but did not fully close the cold gap
against LMCache.

## GPU Staging Rationale

Warm L1 loads originally tried a direct pinned-host-to-KV-cache copy path for
contiguous blocks. On the measured N=20 warm batch this still caused many
small layer copies: roughly 15 requests x 36 layers = 540 host-to-device copy
operations in one step.

The optimized path reads each L1 hit into a temporary GPU staging tensor, then
uses the existing merge/index-copy path:

1. Copy each chunk from pinned L1 memory into a per-request GPU staging tensor.
2. Merge compatible staged chunks by block order.
3. Restore all requested blocks for each layer with batched GPU copies.

This intentionally adds a transient GPU buffer and can look like "copy twice":

- L1 pinned CPU -> GPU staging.
- GPU staging -> vLLM KV cache.

The tradeoff was still faster in the measured workload because it replaces
hundreds of small per-request/per-layer host-to-device copies with a much
smaller number of larger GPU-side copies. In the N=20 profile, warm load copy
count dropped to 36 layer-level copies for the merged batch, and warm elapsed
fell to about 0.07-0.08 s.

The staging memory is transient and proportional to loaded chunk bytes in the
current step. It is not a reserved permanent pool. The downside is real: peak
GPU memory increases during load. For memory-tight workloads, a future option
could add a threshold that uses direct host copy for small loads or when there
is insufficient staging headroom. The current benchmark setting had enough
headroom (`--gpu-util 0.35`) and benefited from the batched copy path.

## Save-Path Optimizations

Two safe save-path changes were kept:

- Reuse a precomputed block index tensor for all layers in one save step,
  instead of rebuilding the same CUDA index tensor per layer.
- Use slice views when block IDs are contiguous, falling back to `index_select`
  only for non-contiguous blocks.
- Keep chunk-aware save staging as a torch tensor for every transfer backend.
  Connector save/load code now calls the same chunk transfer API for GDS and
  iouring-mem, so connector hot paths no longer branch on transfer type.

The attempted layer-major L1 layout was reverted because it broke e2e
correctness. The attempted batch-write API was also reverted because it made
cold slower by moving more GPU-to-host staging work into the foreground path.

## Final E2E Result

Command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --num-prompts 50 \
  --daser-transfer-backend iouring-mem \
  --daser-l1-cache-size 16gb \
  --out <scratch>/iouring_native_final_vs_lmcache_ssd_cpu_n50.json
```

Configuration:

- DaseR transfer backend: `iouring-mem`.
- LMCache mode: `local-cpu-disk`.
- DaseR L1 cache size: 16 GB.
- LMCache max local CPU size: 16 GB.
- DaseR store size and LMCache max local disk size: 3.176 GB.

| Metric | DaseR `iouring-mem` | LMCache SSD + local CPU | DaseR / LMCache |
| --- | ---: | ---: | ---: |
| correctness | 50/50 | 50/50 | - |
| cold elapsed | 1.51 s | 1.12 s | 0.74x tok/s |
| warm elapsed | 0.18 s | 0.22 s | 1.22x tok/s |
| cold prompt tok/s | 9,294 | 12,554 | 0.74x |
| warm prompt tok/s | 78,733 | 64,450 | 1.22x |

Warm performance exceeds LMCache under this aligned SSD + local CPU comparison.
Cold does not yet exceed LMCache; the measured evidence does not support
claiming cold parity.

## Eviction Pressure Benchmark Mode

The benchmark now has `--pressure-eviction` for comparisons where the workload
must exceed the memory tier:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --num-prompts 1000 \
  --daser-transfer-backend iouring-mem \
  --daser-l1-cache-size 2gb \
  --pressure-eviction \
  --out <scratch>/iouring_mem_pressure_vs_lmcache_ssd_cpu.json
```

The pressure mode keeps the benchmark's default `gpu_memory_utilization` and
`max_num_seqs`; the command does not need to override them. It enforces:

- total prompt KV bytes must exceed DaseR L1 / LMCache local CPU capacity,
- total prompt KV bytes must also exceed DaseR SSD store capacity,
- DaseR SSD store capacity is larger than L1,
- LMCache disk capacity is aligned to the DaseR store size,
- total prompt KV bytes must exceed LMCache disk capacity,
- LMCache local CPU capacity is aligned to DaseR L1 for `local-cpu-disk`,
- the JSON/report include KV/L1 and store/L1 ratios.

This mode is intended to force both L1/CPU eviction and SSD/ring-buffer
eviction, instead of measuring only the all-in-memory case.

## Remaining Work

Likely next steps for cold-path performance:

- Replace the one-at-a-time Python/ctypes io_uring wrapper with a C++/liburing
  extension that supports batched SQEs and completions.
- Investigate whether cold save publication can further overlap with vLLM
  without exposing chunks before L1 bytes are complete.
- Profile scheduler/control-plane costs under the `iouring-mem` backend, since
  the foreground cold gap is no longer explained by raw SSD write completion.
- Add adaptive warm-load staging to reduce peak GPU memory for tight workloads.
