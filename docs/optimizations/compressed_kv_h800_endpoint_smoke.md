# H800 PCIe Compressed KV Endpoint Smoke

This preliminary smoke validates the integrated io_uring compressed-read-only
load path. It is correctness and direction evidence, not the formal endpoint
result: each condition has one measured request rather than five randomized
paired repetitions, and the request is a deterministic synthetic 642-token
prompt rather than the frozen LongBench profile.

## Configuration

- GPU: one otherwise idle NVIDIA H800 PCIe
- Model: GLM-4-9B-chat-1m, BF16, TP1
- Attention: FlashAttention 3
- KV block size: 128 tokens
- Input/output: 642 prompt tokens, one generated token
- Transfer: O_DIRECT + io_uring; each condition starts with an empty L1
- GPU memory: no explicit `--gpu-memory-utilization`; vLLM used its default 0.9
- Compression: static 15-entry high-byte codebook calibrated on slot 0

## Result

| Condition | L2/H2D bytes | L2 reads | TTFT | End-to-end |
|---|---:|---:|---:|---:|
| Raw | 52,428,800 | 1 | 320.65 ms | 321.41 ms |
| Compressed | 40,980,480 | 5 | 234.93 ms | 235.71 ms |

The five populated slots all used compressed mode. Their aligned transfer
ratio was 0.78164, reducing L2/H2D bytes by 21.84%. The one-pair TTFT reduction
was 26.73%. Raw and compressed generated the same byte-exact output text.

The compressed data file, control index, and side index retained identical
sizes and mtimes through the request and shutdown. Metrics recorded one lookup,
one successful transfer load, five L2 reads, and no store operation.

## Bugs Found By The Endpoint

The first integrated request exposed two issues absent from isolated kernel
tests:

- vLLM creates connector-owned tensors under `InferenceMode`; metadata updated
  later on the load thread must be allocated under `torch.inference_mode(False)`.
- opening an immutable store through the raw io_uring constructor performed a
  same-size truncate and changed mtime; compressed mode now opens L2 with
  `O_RDONLY | O_DIRECT` and rejects transfer-layer store APIs.

## Remaining Gate

Formal evidence still requires the frozen 97-request LongBench prefill profile,
the 200-request decode profile, five randomized paired repetitions, full aligned
hit coverage, and hierarchical bootstrap confidence intervals. This smoke must
not be cited as satisfying those gates.
