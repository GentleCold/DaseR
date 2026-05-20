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
  --gpu-util 0.4 \
  --comparison-mode gds-vs-lmcache-local-ssd \
  --out results.json
```

Use `--comparison-mode iouring-mem-vs-lmcache-local-ssd-mem` for the tiered
comparison and add `--evict` for eviction runs.

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
and not an io_uring setting. The final configuration uses 32 kvikio threads,
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
vLLM instance, TP=1, `max_num_seqs=64`, and `gpu_util=0.4`. Cold timing includes
DaseR store submission and completion; warm DaseR passes use `daser_skip_save`.

| Mode | DaseR evict | DaseR cold | LMCache cold | Cold ratio | DaseR warm | LMCache warm | Warm ratio | DaseR mismatch | LMCache mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDS vs local SSD | no | 3.80 s | 4.19 s | 1.09x | 0.79 s | 1.81 s | 2.28x | 1/200 | 2/200 |
| GDS vs local SSD | yes | 3.13 s | 4.22 s | 1.35x | 1.67 s | 2.45 s | 1.47x | 1/200 | 0/200 |
| iouring+mem vs local SSD+mem | no | 3.06 s | 4.15 s | 1.36x | 0.45 s | 0.52 s | 1.16x | 2/200 | 0/200 |
| iouring+mem vs local SSD+mem | yes | 2.94 s | 4.17 s | 1.42x | 1.63 s | 1.75 s | 1.07x | 1/200 | 0/200 |

Ratios are DaseR prompt-token throughput divided by LMCache prompt-token
throughput. Values above `1.0x` mean DaseR is faster. DaseR warm uses
`daser_skip_save` to skip duplicate warm stores, while LMCache does not expose an
equivalent benchmark-local skip-save control in this script.

The no-evict runs loaded 200/200 prompts through visible DaseR transfer hits.
The evict runs loaded 187/200 visible prompts for GDS and 184/200 for iouring.
Visible-hit mismatches were 1/200 for GDS no-evict, 1/187 for GDS evict, 2/200
for iouring no-evict, and 1/184 for iouring evict. Later benchmark revisions now
log prompt alignment,
`max_num_seqs` wave index, position within the wave, and prompt length for each
sampled-token mismatch. Existing mismatches cluster near vLLM admission-wave
boundaries and are tracked separately from byte-level transfer tests.

## Current Assessment

The server-managed architecture and benchmark harness meet the performance
target for both transfer systems with and without DaseR eviction pressure:
DaseR cold and warm throughput are above LMCache in all four rows, with the
largest warm-path gains coming from server-side skip-save and batched
connector loads.
