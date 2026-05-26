# RoPE Apply Torch Compile Optimization

**Date:** 2026-05-26
**Benchmark:** `benchmarks/bench_rope_apply.py`
**Target:** chunk-reuse load-path RoPE delta application

## Bottleneck

Chunk reuse can load cached KV blocks at a different prompt position than the
position where the block was originally stored. DaseR fixes the key cache by
applying a relative RoPE delta in `copy_staging_to_kv_cache()`.

Before this change, `apply_rope_delta_to_key_block()` built the full PyTorch
eager graph on every layer/block:

- fp32 `arange` and inverse-frequency construction;
- `cos` / `sin`;
- split-half NeoX rotation or interleaved rotation;
- dtype conversion and in-place copy back.

In chunk-reuse mode, this sits on the warm load path and scales with
`cache_hits * layers * blocks`.

## Implementation

This branch adds a custom-operator area under `daser/ops/` and moves the RoPE
delta implementation there:

- `daser.ops.rope_apply.apply_rope_delta_to_key_block()` is the backend-aware
  operator entrypoint.
- `daser.ops.rope_apply.apply_rope_delta_to_key_block_naive()` preserves the
  previous eager behavior as a correctness oracle and fallback.
- `daser.connector.staging.apply_rope_delta_to_key_block()` remains the
  connector-facing compatibility wrapper and delegates into `daser.ops`.

The default `auto` backend tries optional TileLang first, then `torch.compile`,
then naive PyTorch. TileLang and compiled functions are cached by dtype, input
shape, rotary dimension, and RoPE layout. CPU tensors and unsupported shapes
use the naive path directly.

The worker warms representative RoPE apply shapes during
`register_kv_caches()`, including single-block, common multi-block, and the
largest block count allowed by the preallocated staging buffer. That moves the
first TileLang or `torch.compile` cost out of the first chunk-reuse infer
request and into worker initialization, where vLLM is already doing model and
graph warmup work. Because TileLang and `torch.compile` cache by full input
shape, document upload also warms any actual stored chunk block counts that
startup did not predict; a worker-local warmed-shape set keeps repeated uploads
from paying this warmup again.

The load path also batches transform work for a copy run. Instead of copying
one block and then applying scale/RoPE one block at a time, it decodes the
slot-major staging bytes into a `[blocks, 2, block_tokens, heads, head_dim]`
layer batch, applies key/value scale and RoPE relocation once per layer over
all loaded blocks that share the same position delta, then copies the batch
into the destination KV cache. This removes per-block RoPE dispatch from warm
chunk hits and keeps non-contiguous destination block IDs correct through
`index_copy_`.

TileLang is used only for contiguous CUDA tensors and supported dtypes/shapes;
otherwise DaseR falls back without changing correctness. The load path makes
the K batch contiguous before RoPE apply so warm chunk hits take the TileLang
fast path when the optional package is installed.

## Benchmark Command

```bash
source <venv>/bin/activate
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_rope_apply.py \
  --device cuda:0 \
  --warmup 10 \
  --iters 50 \
  --backends naive,compile,tilelang
```

Environment:

- GPU: NVIDIA GeForce RTX 4090
- dtype: bf16
- RoPE base: 1,000,000
- delta: 128
- layout: NeoX split-half

The benchmark clears the local RoPE compile cache and resets Torch Dynamo
between independent shape/backend cases. Timing uses CUDA events and excludes
the first compile from steady-state measurements via warmup iterations.

## E2E Metric

For service-demo validation, use `ttft_ms` rather than full request wall time.
Chunk reuse accelerates prompt prefill by loading external KV blocks before
decode starts. It does not remove the autoregressive decode work after the first
token, so total latency can hide the prefill benefit when `max_tokens` is large.

The HTTP `/infer` path now calls vLLM completions in streaming mode, records
time to the first non-empty streamed text fragment, and returns `ttft_ms` next
to the existing `latency_ms`. The demo prints both metrics.

## Load-Path Root Cause

The slow chunk-reuse load observed during service validation was not caused by
the io_uring transfer backend. With iouring and an L1 warm hit, loading
`424,673,280` bytes from server-owned L1 into the CUDA IPC staging buffer took
`7.8-7.9 ms` with `l1_hits=4` and `l2_reads=0`. Worker-side restoration of that
same load, including IPC wait, KV copy, scale, and RoPE, took about `16 ms`
after warmup (`ipc_ms=8.7-9.0`, `copy_ms=7.1-7.4`, `sync_ms=0.03`).

The earlier first chunk-reuse infer instead spent about `23.8 s` in worker
`copy_ms`. A micro reproduction for the large RoPE batch shape showed the first
TileLang call paying seconds of kernel compilation while steady-state execution
was sub-millisecond. The fix is therefore to pay shape compilation before the
first reuse request:

- startup warms representative batch block counts, bounded by the staging
  buffer capacity;
- document upload warms the exact chunk block counts that will later be loaded;
- load-time timing is kept available at debug level so future regressions can
  separate server transfer time from worker restoration time.

## Results

| batched blocks | block | heads | head_dim | naive us | compile us | tilelang us | tilelang vs compile | tilelang vs naive | max diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 8 | 128 | 53.85 | 36.35 | 4.18 | 8.70x | 12.89x | 0.000e+00 |
| 2 | 16 | 8 | 128 | 55.12 | 27.30 | 5.91 | 4.62x | 9.33x | 9.766e-04 |
| 4 | 16 | 8 | 128 | 94.11 | 28.09 | 4.27 | 6.58x | 22.04x | 0.000e+00 |
| 8 | 16 | 8 | 128 | 55.35 | 28.21 | 4.24 | 6.65x | 13.04x | 0.000e+00 |
| 16 | 16 | 8 | 128 | 57.85 | 28.95 | 4.69 | 6.18x | 12.35x | 1.953e-03 |
| 4 | 32 | 8 | 128 | 56.02 | 28.22 | 18.79 | 1.50x | 2.98x | 0.000e+00 |
| 4 | 64 | 8 | 128 | 156.70 | 37.51 | 6.48 | 5.78x | 24.17x | 1.562e-02 |
| 4 | 16 | 16 | 128 | 57.83 | 27.95 | 4.27 | 6.54x | 13.54x | 4.883e-04 |
| 4 | 16 | 32 | 128 | 57.81 | 28.62 | 4.68 | 6.11x | 12.35x | 3.725e-09 |
| 4 | 16 | 8 | 64 | 55.45 | 27.70 | 4.27 | 6.49x | 12.99x | 0.000e+00 |

Correctness was checked against the naive backend for every case. The largest
reported max absolute difference was `1.5625e-2`, within bf16 tolerance for
this operation.

## Service Validation

Commands used the Qwen3-8B local model, `--gpu-memory-utilization 0.4`,
`--max-model-len 4096`, `--max-num-seqs 2`, vLLM prefix caching disabled, and
DaseR's default `iouring` transfer mode. The chunk-reuse service ran on GPU 2
and the prefix baseline service ran on GPU 3.

Short `examples/service_demo/demo.py --compare-baseline` run:

| mode | prompt tokens | completion tokens | TTFT ms | latency ms | cache hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix baseline | 346 | 80 | 58.8 | 595.8 | 0 |
| chunk reuse | 369 | 72 | 41.2 | 706.4 | 4 |

Long-document repeat-10 run:

| mode | run | prompt tokens | completion tokens | TTFT ms | latency ms | cache hits | hit tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefix baseline | 1 | 2869 | 32 | 152.0 | 376.9 | 0 | `[]` |
| prefix baseline | 2 | 2869 | 32 | 140.3 | 365.5 | 0 | `[]` |
| chunk reuse | 1 | 2905 | 32 | 53.8 | 345.5 | 4 | `[16, 1440, 16, 1408]` |
| chunk reuse | 2 | 2905 | 32 | 53.8 | 345.6 | 4 | `[16, 1440, 16, 1408]` |

The long-document first-token speedup was `152.0 / 53.8 = 2.83x` versus the
prefix baseline. The first chunk-reuse infer no longer showed the previous
seconds-long RoPE compile spike; the same run's server transfer logs reported
`424,673,280` bytes served from L1 in `7.765 ms`, and worker logs reported
`copy_ms=7.313`.

## Interpretation

`torch.compile` removes about half of the steady-state RoPE apply latency for
the common `head_dim=128` chunk-reuse shapes on this GPU. TileLang is materially
faster than both baselines: for four loaded blocks with `block=16, heads=8,
head_dim=128`, the old per-block eager path would issue four calls at roughly
`4 * 53.85 = 215.40 us` per layer, while the batched TileLang path issues one
call at `4.27 us`, a derived 50.44x reduction for the RoPE transform portion.
Against the batched `torch.compile` baseline for that same shape, TileLang is
6.58x faster.

The first TileLang call for a new shape pays kernel compilation cost. Worker
startup already warms the common KV block shape, and unsupported or failed
TileLang execution disables that backend and falls through to `torch.compile`
and then naive PyTorch.
