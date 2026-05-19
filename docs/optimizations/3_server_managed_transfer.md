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
  opens succeed, with a kvikio compat fallback when direct cuFile is unavailable.
- `iouring_pinned`: a server-owned tiered transfer layer. It publishes stores to
  L1 first, schedules L2 writes asynchronously, serves loads from L1 spans when
  present, and falls back to L2 reads plus L1 promotion on misses. The L1
  replacement policy is pluggable and currently backed by LRU.

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

Representative command shape:

```bash
python benchmarks/bench_e2e_daser_vs_lmcache.py \
  --num-prompts 20 \
  --max-input-tokens 512 \
  --max-num-seqs 8 \
  --gpu-util 0.4 \
  --comparison-mode gds-vs-lmcache-local-ssd \
  --out results.json
```

Use `--comparison-mode iouring-mem-vs-lmcache-local-ssd-mem` for the tiered
comparison and add `--evict` for eviction runs.

## Optimizations Tried

### Server-side transfer abstraction

The new `daser.transfer.TransferLayer` API gives the IPC server a single
`store()` / `load()` surface for GDS and tiered iouring transfer. Runtime config
is returned to the connector during registration so the connector can remain
backend-agnostic.

### CUDA IPC staging

Transfer RPCs carry CUDA IPC payloads for worker-owned staging tensors. The
server opens those handles and writes into or reads out of the mapped GPU memory.
A same-process pointer fallback is used only for local unit-test and benchmark
cases where producer and consumer PIDs match.

### Positioned L2 IO

The tiered transfer layer now uses positioned reads and writes for L2 spans.
This avoids shared file-offset races when multiple asynchronous L2 operations are
in flight.

### L1 span hits and pending-write waits

Tiered loads first check whether each requested byte span is fully resident in
L1. Hits are copied from L1 directly to the CUDA staging target. Misses wait for
any pending L2 write for that key, read from L2, and promote the loaded span into
L1. This preserves the store flow where L1 becomes readable before L2 completes.

### Batched connector loads

The connector batches all load spans in a forward step into one staging tensor
and one `transfer_load` RPC. When no per-request load transform is required, it
restores all loaded blocks back into vLLM KV cache tensors in a single combined
path. This is the main warm-path optimization in this branch.

### Direct GDS fallback

The GDS layer prefers direct cuFile, but the measured environment returned a
cuFile internal error during direct open. The final GDS measurements therefore
ran through the kvikio compat path. The benchmark still exercises the
server-managed GDS transfer API, but it is not evidence for direct cuFile
throughput on this machine.

## Final Results

All runs used 20 IMDB prompts, 4,061 prompt tokens, max input length 512,
`max_num_seqs=8`, and `gpu_util=0.4`.

| Mode | Evict | DaseR cold | LMCache cold | Cold ratio | DaseR warm | LMCache warm | Warm ratio | DaseR mismatch | LMCache mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDS vs local SSD | no | 1.00 s | 0.38 s | 0.38x | 0.16 s | 0.31 s | 1.89x | 2/20 | 1/20 |
| GDS vs local SSD | yes | 1.05 s | 0.38 s | 0.36x | 0.16 s | 0.23 s | 1.42x | 9/20 | 1/20 |
| iouring+mem vs local SSD+mem | no | 1.37 s | 0.37 s | 0.27x | 0.18 s | 0.14 s | 0.81x | 2/20 | 1/20 |
| iouring+mem vs local SSD+mem | yes | 1.36 s | 0.37 s | 0.27x | 0.54 s | 0.21 s | 0.39x | 7/20 | 1/20 |

Ratios are DaseR prompt-token throughput divided by LMCache prompt-token
throughput. Values above `1.0x` mean DaseR is faster.

## Current Assessment

The architecture change is implemented and the GDS warm path is faster than
LMCache in both no-evict and evict runs. The requested acceptance target is not
fully met:

- DaseR cold is still slower than LMCache in all four runs.
- The tiered iouring+mem warm path is still slower than LMCache, especially when
  eviction is enabled.
- Correctness mismatches remain above the "rare mismatch" target in the GDS
  evict and iouring evict runs.
- The measured GDS path used kvikio compat fallback because direct cuFile open
  failed in this environment.

The next performance work should focus on cold-path save overhead and tiered
load promotion under eviction. The next correctness work should isolate whether
the mismatches come from chunked-prefill scheduling, restore ordering, or stale
metadata after eviction.
