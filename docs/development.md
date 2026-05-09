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
    --store-path /path/to/daser.store \
    --store-size 10737418240 \
    --socket-path /tmp/daser.sock \
    --index-path /tmp/daser.index
```

| Flag | Default | Description |
|------|---------|-------------|
| `--store-path` | (required) | Pre-allocated NVMe store file |
| `--store-size` | `10 GB` | Total store capacity in bytes |
| `--socket-path` | `/tmp/daser.sock` | Unix domain socket path |
| `--index-path` | `/tmp/daser.index` | Metadata index file |
| `--slot-size` | `2097152` (2 MB) | Bytes per KV slot |
| `--infer-cache-mode` | `prefix` | `prefix` keeps strict prefix lookup; `doc-rope` enables registered-document chunk planning |

`doc-rope` is limited to service requests that reference registered `doc_ids`.
The server returns resident document chunks with source and target position
offsets, and the connector can rotate RoPE key cache blocks when those chunks
are loaded at a different prompt position. It does not perform global substring
retrieval and does not recompute cross-chunk attention.

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
