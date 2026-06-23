# KV Cache Copy Bandwidth Microbench

This directory contains a standalone CUDA C++ microbenchmark for comparing eight
ways to restore CPU-resident KV bytes into a vLLM-like, per-layer GPU KV cache:

1. `direct_h2d_scatter`: copy each layer/block slice directly from pinned CPU
   memory into the destination KV cache tensor.
2. `staging_h2d_then_d2d_scatter`: copy the full slot-major payload from pinned
   CPU memory into a contiguous GPU staging buffer, then scatter from staging
   into the destination KV cache with device-to-device copies.
3. `staging_h2d_then_kernel_scatter`: copy the full slot-major payload into GPU
   staging, then scatter with one CUDA kernel launch.
4. `h2d_to_staging_only`: measure only the contiguous host-to-device staging
   copy from the fastest staging path.
5. `kernel_scatter_only`: measure only the GPU staging to KV cache scatter
   kernel from the fastest staging path.
6. `pipelined_staging_kernel_scatter`: split the slot-major payload into chunks,
   copy chunks into a double-buffered GPU staging ring on one stream, and scatter
   ready chunks to the KV cache on another stream.
7. `cross_request_pipelined_staging_kernel_scatter`: repeat the pipelined path
   for multiple requests in one timed region so fill/drain costs can be amortized.
8. `mapped_host_kernel_scatter`: expose pinned CPU memory as mapped host memory
   and let one CUDA kernel scatter directly from host memory into the KV cache.

The benchmark does not import DaseR or vLLM. It only models the bandwidth shape
of DaseR's restore path when the final vLLM KV cache is distributed across
layers and optionally across non-contiguous block slots.

## Build

```bash
cd benchmarks/bandwidth
make
```

If the default CUDA architecture selection is not accepted by your `nvcc`, pass
an explicit target:

```bash
make CUDA_ARCH=sm_90
```

## Run

```bash
./kv_cache_copy_bandwidth \
  --device 0 \
  --layers 36 \
  --blocks 64 \
  --block-tokens 16 \
  --heads 8 \
  --head-dim 128 \
  --dtype-bytes 2 \
  --block-stride 2 \
  --pipeline-chunks 4 \
  --requests 8 \
  --warmup 10 \
  --iters 50
```

`--block-stride 1` writes restored blocks contiguously. Larger values leave gaps
between destination block IDs to approximate a fragmented vLLM KV cache.
`--pipeline-chunks` controls how many block chunks the pipelined staging path
uses; two GPU staging slots are reused as a double buffer.
`--requests` controls how many repeated restores the cross-request pipeline
measures in one timed region.

Bandwidth is reported as effective restored KV bytes divided by elapsed time.
The staging paths also report total traffic because they perform both host-to-
device and GPU-side copies. `mapped_host_kernel_scatter` is a diagnostic path:
it removes per-slice copy submission overhead, but GPU reads host memory across
PCIe, so it is not expected to match copy-engine H2D bandwidth.
