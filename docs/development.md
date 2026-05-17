# Development Guide

## Environment Setup

```bash
source <venv>/bin/activate
pip install -e .
```

---

## Running the DaseR Server

```bash
python -m daser.server \
    --vllm-base-url http://127.0.0.1:8001 \
    --model /path/to/model \
    --tokenizer /path/to/model \
    --store-path /path/to/daser.store \
    --store-size 10737418240 \
    --socket-path /tmp/daser.sock \
    --index-path /tmp/daser.index \
    --host 0.0.0.0 \
    --port 8080
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-base-url` | (required) | Base URL for the vLLM OpenAI-compatible server |
| `--model` | (required) | Model identifier passed to vLLM |
| `--tokenizer` | (required) | Tokenizer name/path used by the HTTP server |
| `--store-path` | (required) | Pre-allocated NVMe store file |
| `--store-size` | `10 GB` | Total store capacity in bytes |
| `--socket-path` | `/tmp/daser.sock` | IPC server Unix socket path |
| `--index-path` | `/tmp/daser.index` | Metadata index file |
| `--slot-size` | `2097152` (2 MB) | Bytes per KV slot |
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

```python
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="DaserConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "socket_path": "/tmp/daser.sock",
        "store_path": "/path/to/daser.store",
        "slot_size": 2097152,   # must match server --slot-size
        "block_tokens": 16,
        "model_id": "my-model",
    },
)

llm = LLM(model="...", kv_transfer_config=ktc, enable_prefix_caching=False)
```
