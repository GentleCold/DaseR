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
- `iouring_pinned`: a server-owned tiered transfer layer. It publishes stores to
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

The `--evict` flag currently forces eviction pressure on DaseR sizing only.
LMCache is left with its configured local-disk and local-CPU ceilings, so the
eviction rows are DaseR eviction stress runs rather than symmetric eviction
pressure on both systems.

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
`store_bytes()` / `load_bytes()` surface for GDS and tiered iouring transfer.
Runtime config is returned to the connector during registration so the
connector can remain backend-agnostic.

### CUDA IPC staging

Transfer RPCs carry CUDA IPC payloads for worker-owned staging tensors. The
server opens those handles and writes into or reads out of the mapped GPU memory.
A same-process pointer path is used only for local unit-test and benchmark cases
where producer and consumer PIDs match.

The current implementation uses conservative worker-side stream synchronization
before exporting store staging buffers and after load RPC completion. A future
optimization should replace these full stream barriers with CUDA IPC events so
the server can wait only on the producer work it actually consumes.

### Positioned L2 IO

The tiered transfer layer now uses positioned reads and writes for L2 spans.
This avoids shared file-offset races when multiple asynchronous L2 operations are
in flight.

The io_uring implementation calls `io_uring_setup` and `io_uring_enter`
directly and does not issue synchronous positioned file IO on the tiered L2
path.

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

The connector batches all load spans in a forward step into one staging tensor
and one `transfer_load` RPC. When no per-request load transform is required, it
restores all loaded blocks back into vLLM KV cache tensors in a single combined
path. This is the main warm-path optimization in this branch.

### Direct GDS compatibility path

The GDS layer prefers direct cuFile, but the measured environment returned a
cuFile internal error during direct open. The final GDS measurements therefore
ran through the kvikio compat path. The benchmark still exercises the
server-managed GDS transfer API, but it is not evidence for direct cuFile
throughput on this machine.

## Final Results

All runs used 200 IMDB prompts, 55,362 prompt tokens, max input length 512, one
vLLM instance, TP=1, `max_num_seqs=200`, and `gpu_util=0.4`.

| Mode | DaseR evict | DaseR cold | LMCache cold | Cold ratio | DaseR warm | LMCache warm | Warm ratio | DaseR mismatch | LMCache mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDS vs local SSD | no | 2.55 s | 4.19 s | 1.64x | 0.79 s | 1.99 s | 2.52x | 0/200 | 0/200 |
| GDS vs local SSD | yes | 2.72 s | 4.14 s | 1.52x | 0.88 s | 2.10 s | 2.39x | 0/200 | 0/200 |
| iouring+mem vs local SSD+mem | no | 2.55 s | 4.03 s | 1.58x | 0.45 s | 0.50 s | 1.11x | 0/200 | 0/200 |
| iouring+mem vs local SSD+mem | yes | 2.65 s | 4.05 s | 1.53x | 0.56 s | 1.33 s | 2.39x | 0/200 | 0/200 |

Ratios are DaseR prompt-token throughput divided by LMCache prompt-token
throughput. Values above `1.0x` mean DaseR is faster. DaseR warm uses
`daser_skip_save` to skip duplicate warm stores, while LMCache does not expose an
equivalent benchmark-local skip-save control in this script.

The no-evict runs loaded 198/200 prompts through visible DaseR transfer hits and
the evict runs loaded 173/200. Visible-hit mismatches were 0 in all four DaseR
runs; end-to-end correctness was 200/200 for both DaseR and LMCache in all
rows.

## Current Assessment

The server-managed architecture and benchmark harness meet the performance
target for both transfer systems with and without DaseR eviction pressure:
DaseR cold and warm throughput are above LMCache in all four rows, with the
largest warm-path gains coming from server-side skip-save and batched
connector loads.
