# Server-Managed Transfer Layer Optimization

**Date:** 2026-05-20
**Benchmark:** `benchmarks/bench_e2e_daser_vs_lmcache.py`

This branch moves transfer ownership from the vLLM connector process into the
DaseR server. The connector now exposes temporary CUDA staging buffers through
CUDA IPC and sends transfer RPCs; the server owns the SSD file, transfer backend
selection, L1/L2 sizing, and replacement policy.

## Transfer Modes

The server selects one transfer mode at startup:

- `gds`: a server-owned `GDSTransferLayer` using kvikio/cuFile when direct GDS
  opens succeed; on the measured host kvikio selected its compat path because
  direct cuFile open returned an internal error.
- `iouring`: a server-owned tiered transfer layer. It publishes stores to
  L1 first, schedules L2 writes asynchronously through native io_uring syscalls,
  serves loads from L1 spans when present, and reads L2 plus promotes into L1 on
  misses. The L1 replacement policy is pluggable and currently backed by LRU.
- `iouring --skip-l2`: a volatile L1-only transfer path using
  `L1OnlyTransferLayer`. It keeps the same IPC lookup/store flow and logical
  slot offsets, but does not create `daser.store`, does not write L2, and does
  not persist `daser.index`. Loads only succeed for ranges still resident in L1.
  This mode is rejected with `gds` because GDS requires an L2 store file.

For both modes, the server performs transfer operations against CUDA IPC handles
provided by the worker. The connector no longer chooses a transfer implementation
or opens SSD files.

## Benchmark Modes

The E2E benchmark now has two comparison modes:

- `gds-vs-lmcache-local-ssd`
- `iouring-mem-vs-lmcache-local-ssd-mem`

`--evict` derives smaller DaseR L2/L1 capacities from the workload so the run
exercises eviction. Without `--evict`, DaseR sizes L2 above the workload KV
footprint; for the iouring mode it also sizes L1 above the workload KV
footprint. DaseR warm runs pass `daser_skip_save` so the warm path measures
lookup and load rather than re-saving the same KV.

LMCache local-disk and local-CPU limits are derived from the same DaseR L2/L1
sizes. In `gds-vs-lmcache-local-ssd`, LMCache uses local SSD plus a small fixed
CPU staging allowance because DaseR has no L1 tier in GDS mode. In
`iouring-mem-vs-lmcache-local-ssd-mem`, LMCache disk and CPU limits match the
DaseR L2 and L1 byte ceilings. LMCache interprets those size knobs with a
`1024**3` multiplier, so the benchmark converts DaseR byte capacities to that
unit directly rather than using decimal GB.

Representative command shape:

```bash
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --model /path/to/model \
  --store-dir /path/to/benchmark-scratch \
  --imdb /path/to/imdb.csv \
  --num-prompts 200 \
  --max-input-tokens 512 \
  --max-num-seqs 64 \
  --comparison-mode gds-vs-lmcache-local-ssd \
  --out results.json
```

Use `--comparison-mode iouring-mem-vs-lmcache-local-ssd-mem` for the tiered
comparison and add `--evict` for eviction runs.
For DaseR L1-only measurements, use the same iouring comparison mode and add
`--skip-l2`; the benchmark records `daser_skip_l2=true` and `storage_tier` as
`l1-only`.

The benchmark defaults to `--gpu-util 0.9` and `--gpu-id auto`. Auto GPU
selection queries `nvidia-smi`, exposes the GPU with the most free memory
through `CUDA_VISIBLE_DEVICES`, and sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before
CUDA libraries are imported. Use `--gpu-id current` to preserve an existing
`CUDA_VISIBLE_DEVICES` value or pass a concrete GPU index to pin the run.

Capacity sizing is workload-derived but capped by the current machine state.
DaseR L2 is capped by free space under `--store-dir` and an absolute ceiling;
DaseR L1 for iouring is capped by free host memory.
If the cap cannot fit the largest single prompt, the benchmark fails early
instead of silently running with an invalid store size. For larger workloads
that fit at least one prompt but exceed the no-evict target, the benchmark caps
L1/L2 at the derived ceilings and records `capacity_capped=true` in the JSON
config.

## Optimizations Tried

### Server-side transfer abstraction

The new `daser.transfer.TransferLayer` API gives the IPC server a single
`store_bytes()` / `load_bytes()` surface for GDS and tiered iouring transfer.
Runtime config is returned to the connector during registration so the
connector can remain backend-agnostic.

### CUDA IPC staging

Transfer RPCs carry CUDA IPC payloads for worker-owned staging tensors. The
server opens those handles and writes into or reads out of the mapped GPU memory.
A same-process pointer path is used only for local unit-test and benchmark cases
where producer and consumer PIDs match.

Worker-side staging is shared by both GDS and iouring modes through
`daser.connector.staging`. `register_kv_caches` creates a bounded
`CudaStagingPool` and preallocates one reusable buffer so the hot path does not
pay a fresh CUDA allocation for the common batch size. Store batches keep their
lease alive until the background transfer future completes; load batches release
their lease immediately after copying bytes back into vLLM KV cache.

The pool is intentionally small relative to model KV cache. The single-buffer
cap is derived from both total and currently free VRAM after vLLM has allocated
KV cache; pending staged bytes are also capped so staging does not reserve a
fixed large fraction of the device. On an 80 GB GPU with ample free memory this
now caps at a 1.5 GiB single staging buffer and 3 GiB of pending store staging.
That single staging buffer is preallocated during connector initialization so
the cold save path does not pay a first-use CUDA allocation. Load spans are
split into the same bounded batches instead of allocating one unbounded
warm-path tensor for the whole step.

Store staging records a CUDA event on the producer stream and synchronizes that
event before server transfer. Load staging is synchronized after each server RPC
returns before bytes are copied back into vLLM KV cache.

### Positioned L2 IO

The tiered transfer layer now uses positioned reads and writes for L2 spans.
This avoids shared file-offset races when multiple asynchronous L2 operations are
in flight.

The io_uring implementation calls `io_uring_setup` and `io_uring_enter`
directly and does not issue synchronous positioned file IO on the tiered L2
path. The tiered L2 file is opened with `O_DIRECT`, and all L2 offsets and byte
counts must be 4096-byte aligned. The pinned L1 pool is allocated during
`TieredIOUringTransferLayer` initialization, triggered by connector
`init_transfer` during LLM construction, so pinned allocation is part of
initialization rather than cold-pass timing.

### L1 span hits and pending-write waits

Tiered loads first check whether each requested byte span is fully resident in
L1. Hits are copied from L1 directly to the CUDA staging target. Misses wait for
any pending L2 write for that key, read from L2, and promote the loaded span into
L1. This preserves the store flow where L1 becomes readable before L2 completes.
Overlapping L1 ranges are invalidated on writes so later subrange stores cannot
leave stale wider cached ranges behind. Overlapping pending L2 writes are chained
so rewritten spans persist in order.

The tiered mode intentionally publishes data after L1 insertion, not after L2
durability. That matches the current L1-first flow, but it means unclean process
death can lose recently committed L1-only bytes before the background L2 write
drains.

### Batched connector loads

The connector batches load spans in a forward step into bounded staging batches
and issues one `transfer_load` RPC per batch. When no per-request load transform
is required, it restores loaded blocks back into vLLM KV cache tensors in a
combined copy path per batch. This keeps warm-path GPU staging memory bounded
while preserving most of the earlier batching benefit.

### Direct GDS compatibility path

The GDS layer prefers direct cuFile, but the measured environment returned a
cuFile internal error during direct open. The final GDS measurements therefore
ran through the kvikio compat path. The benchmark still exercises the
server-managed GDS transfer API, but it is not evidence for direct cuFile
throughput on this machine.

The compat path was sensitive to kvikio task granularity. The `task_size`
setting here is kvikio's compat-mode IO chunk size, not a benchmark parameter
and not an io_uring setting. The final configuration uses 64 kvikio threads,
64 MiB compat tasks for writes, and 4 MiB compat tasks for reads. Larger read
tasks improved cold writes but hurt warm-load latency.

### Removed benchmark-side ordering

An earlier evict benchmark sorted prompts by block length before running the
workload. That improved eviction locality but changed the workload shape, so it
has been removed. The benchmark now preserves IMDB input order in both no-evict
and evict modes; performance work should be in the transfer architecture rather
than prompt ordering.

## Final Results

All runs used 200 IMDB prompts, 55,362 prompt tokens, max input length 512, one
vLLM instance, TP=1, `max_num_seqs=64`, `gpu_util=0.9`, auto-selected GPU, and
seed `42`.
Cold timing includes DaseR store submission and completion; warm DaseR passes
use `daser_skip_save`.

Correctness runs as a separate untimed pass over all 200 prompts from the same
workload, using isolated scratch stores so the timed benchmark pass cannot
pre-populate the correctness cold baseline. The check is exact generated text
and generated token ID equality between cold and warm runs. The benchmark
reports exact mismatch counts only and treats DaseR as passing parity when its
mismatch count is no more than one above LMCache's count.

Each benchmark invocation creates a fresh `run_<uuid>` scratch root under the
provided `--store-dir`; LMCache timed, LMCache correctness, DaseR timed, and
DaseR correctness stores are all created below that root. This prevents repeated
runs from reusing old DaseR store files or LMCache local-disk files.

The final correctness rerun also fixed a DaseR load-path synchronization bug:
after server-owned transfer copied KV bytes into worker staging and then into
vLLM's KV cache, the connector now waits for the CUDA copy stream before vLLM
continues model execution. Before that barrier, exact mismatches appeared
sporadically under 200-request batches because warm generation could race with
the KV-cache copy becoming visible.

The rerun also fixed DaseR's full-prompt-hit scheduling for block-aligned input
lengths. LMCache keeps the input unchanged, reports a full cache hit, and then
subtracts one token from the number of externally loaded tokens when the hit
covers the entire request. For a 512-token prompt with 16-token KV blocks, that
means vLLM sees 511 external tokens and recomputes the final token. DaseR now
matches that behavior instead of dropping a whole 16-token block. The connector
still loads the complete KV block containing token 511 from the server, so the
worker never restores a partial block into vLLM's KV cache.

LMCache local-disk store is asynchronous: its `batched_put()` submits
`LocalDiskBackend` writes to a background worker and inserts keys into the
lookup index only after the file write completes. The benchmark now waits for
the LMCache local-disk files to become quiescent before warm generation, so
LMCache warm runs do not start while SSD writes from cold are still being
published.

| Mode | DaseR evict | DaseR cold | LMCache cold | Cold ratio | DaseR warm | LMCache warm | Warm ratio | DaseR exact mismatches | LMCache exact mismatches | Parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GDS vs local SSD | no | 5.00 s | 5.67 s | 1.13x | 0.94 s | 2.03 s | 2.16x | 0 | 0 | pass |
| GDS vs local SSD | yes | 4.57 s | 5.49 s | 1.20x | 1.36 s | 2.69 s | 1.97x | 0 | 0 | pass |
| iouring+mem vs local SSD+mem | no | 3.18 s | 5.40 s | 1.70x | 0.44 s | 0.57 s | 1.28x | 0 | 0 | pass |
| iouring+mem vs local SSD+mem | yes | 3.09 s | 5.43 s | 1.76x | 1.60 s | 1.93 s | 1.21x | 0 | 0 | pass |

Ratios are DaseR prompt-token throughput divided by LMCache prompt-token
throughput. Values above `1.0x` mean DaseR is faster. DaseR warm uses
`daser_skip_save` to skip duplicate warm stores, while LMCache does not expose an
equivalent benchmark-local skip-save control in this script.

The benchmark logs prompt alignment, `max_num_seqs` wave index, prompt length,
chosen cold/warm token IDs, generated text, and DaseR visible-prefix counts.
These diagnostics are tracked separately from byte-level transfer tests.

## Current Assessment

The server-managed architecture and benchmark harness now meet the target in
all four rows: DaseR and LMCache both report zero mismatches on the fixed
200-prompt workload, and DaseR cold and warm throughput are above LMCache for
both transfer systems with and without DaseR eviction pressure. The latest run
also used the benchmark's automatic GPU selection and machine-derived capacity
caps; none of the four rows hit the caps on the measured host.
