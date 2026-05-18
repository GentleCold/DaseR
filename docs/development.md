# Development Guide

## Environment Setup

```bash
source <venv>/bin/activate
pip install -e .
```

---

## Running DaseR and vLLM

Start vLLM first with only the connector identity and IPC socket path. Runtime
values such as `store_path`, `slot_size`, `block_tokens`, and `model_id` are
owned by the DaseR server and fetched by the connector over IPC.

```bash
vllm serve /path/to/model \
    --port 8001 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser.sock"}}'
```

Then start DaseR. The server reads vLLM's served model id from `/v1/models`,
derives KV slot geometry from the local model `config.json`, and creates
`<store-dir>/daser.store` when needed.

```bash
python -m daser.server \
    --vllm-base-url http://127.0.0.1:8001 \
    --store-dir /path/to/daser-state \
    --store-size 10gb \
    --socket-path /tmp/daser.sock \
    --host 0.0.0.0 \
    --port 8080
```

If vLLM is started with a non-local served model name, pass the local model
directory explicitly so DaseR can read tokenizer and geometry metadata:

```bash
python -m daser.server \
    --vllm-base-url http://127.0.0.1:8001 \
    --model-path /path/to/model \
    --store-dir /path/to/daser-state
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-base-url` | (required) | Base URL for the vLLM OpenAI-compatible server |
| `--model-path` | optional | Local HuggingFace model path; required only when vLLM `/v1/models` does not return a local model directory |
| `--store-dir` | (required) | Directory for `daser.store` and `daser.index` |
| `--store-size` | `10 GiB` | Requested store capacity; accepts bytes or `mb`/`gb`/`mib`/`gib` and is rounded down to whole KV slots |
| `--socket-path` | `/tmp/daser.sock` | IPC server Unix socket path |
| `--host` | `0.0.0.0` | HTTP server bind host |
| `--port` | `8080` | HTTP server bind port |

---

## Tests

```bash
# Full suite
pytest -xvs tests/

# Single file
pytest -xvs tests/test_chunk_manager.py

# Single test
pytest -xvs tests/test_chunk_manager.py::test_ring_wrap
```

### vLLM E2E Integration Test

Runs a cold → warm inference cycle through DaserConnector + vLLM, verifying cache-hit correctness and speedup.

**Requirements:**
- CUDA GPU with ≥ 24 GB VRAM
- Qwen3-8B weights at `models/Qwen/Qwen3-8B`
- vLLM installed in the active venv

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
pytest -xvs tests/integration/test_vllm_e2e.py -m integration \
    --log-cli-level=INFO
```

The test fixture automatically starts an in-process DaseR `IPCServer` (no external server needed) with a temporary store file, then tears it down after the module completes.

---

## Linting and Formatting

```bash
# Run all checks (recommended before committing)
pre-commit run --all-files

# Individual tools
ruff check .        # lint
ruff format .       # format (line-length 88)
mypy .              # type checking
```

---

## Storage Benchmark

Compares DaseR against LMCache `LocalDiskBackend` using IMDB review-derived KV chunk sizes (2 MB/slot, 32 layers, bfloat16).

```bash
python benchmarks/bench_storage_imdb.py \
    --num-chunks 100 \
    --store-dir /path/to/scratch-dir \
    --imdb /path/to/imdb.csv
```

**Latest results** (100 chunks × 2 MB, btrfs, kvikio compat mode):

| Metric | DaseR | LMCache | Speedup |
|--------|-------|---------|---------|
| Write | 1.83 GB/s | 1.96 GB/s | 0.93× |
| Cold read | 11.6 GB/s | 7.5 GB/s | **1.54×** |
| Warm read | 14.7 GB/s | 8.0 GB/s | **1.83×** |

> Write gap: DaseR writes from GPU memory (cupy) — compat mode requires a GPU→CPU staging step. On GDS-capable hardware (XFS + cuFile direct DMA) this gap disappears.

---

## Connecting vLLM to DaseR

Pass the DaseR connector to `vllm serve` with a minimal inline
`--kv-transfer-config`:

```bash
vllm serve /path/to/model \
    --port 8001 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser.sock"}}'
```
