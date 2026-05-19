# Development Guide

## Environment

```bash
source <venv>/bin/activate
pip install -e .[dev]
```

Use the same Python environment for DaseR and vLLM integration tests so the
connector imports resolve consistently.

---

## Running vLLM and DaseR

DaseR uses a vLLM-first startup sequence. Start `vllm serve` with only the
connector identity and the IPC socket path:

```bash
vllm serve /path/to/model \
    --port 8001 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser.sock"}}'
```

Then start DaseR:

```bash
python -m daser.server \
    --vllm-base-url http://127.0.0.1:8001 \
    --store-dir /path/to/daser-state \
    --store-size 10gb \
    --socket-path /tmp/daser.sock \
    --host 0.0.0.0 \
    --port 8080
```

`python -m daser.server` starts both servers in one process:

- HTTP server: FastAPI routes for health checks, document upload/list/get/delete,
  and `/infer`.
- IPC server: Unix socket + msgpack endpoint used only by `DaserConnector`.

The DaseR server reads vLLM's served model id from `/v1/models`, derives KV
geometry from the local model `config.json`, creates `<store-dir>/daser.store`,
and saves metadata to `<store-dir>/daser.index` on shutdown. Runtime values such
as `store_path`, `slot_size`, `block_tokens`, and `model_id` are owned by DaseR
and fetched by the connector over IPC.

If vLLM exposes a non-local served model name, pass the local model path
explicitly:

```bash
python -m daser.server \
    --vllm-base-url http://127.0.0.1:8001 \
    --model-path /path/to/model \
    --store-dir /path/to/daser-state
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-base-url` | required | Base URL for the vLLM OpenAI-compatible server |
| `--model-path` | optional | Local HuggingFace model path; required when `/v1/models` returns an alias |
| `--store-dir` | required | Directory for `daser.store` and `daser.index` |
| `--store-size` | `10 GiB` | Store capacity; accepts bytes or `mb`/`gb`/`mib`/`gib` and is rounded down to whole KV slots |
| `--socket-path` | `/tmp/daser.sock` | IPC server Unix socket path |
| `--host` | `0.0.0.0` | HTTP server bind host |
| `--port` | `8080` | HTTP server bind port |
| `--cache-reuse-mode` | `prefix` | `prefix` for exact prefix reuse, `chunk` for block-aligned document chunk reuse |
| `--transfer-backend` | `gds` | `gds` for kvikio/cuFile direct or compat IO, `iouring-mem` for pinned host L1 plus SSD IO |
| `--l1-cache-size` | `0` | Pinned host L1 capacity; required and positive when `--transfer-backend iouring-mem` |

---

## Tests

### Unit and Component Tests

Run the default non-integration suite before committing code changes:

```bash
pytest -q -m "not integration and not slow"
```

Useful focused commands:

```bash
pytest -q tests/server
pytest -q tests/connector
pytest -q tests/retrieval tests/position
pytest -q tests/server/test_main_cli.py::test_parse_size_bytes_accepts_human_readable_units
```

These tests use public module interfaces and lightweight stubs where possible.
They do not require a running vLLM server.

### Integration Tests

Integration tests require CUDA, vLLM, CuPy/kvikio, and local model weights:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
pytest -q tests/integration -m integration --log-cli-level=INFO
```

The integration fixtures start an in-process DaseR `IPCServer` with a temporary
store file. They exercise the vLLM connector path without requiring an external
`python -m daser.server` process.

### Benchmarks

The maintained benchmark is the vLLM end-to-end DaseR vs LMCache comparison:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 200 \
    --max-num-seqs 64 \
    --gpu-util 0.4 \
    --out /path/to/results.json
```

For a quick DaseR smoke run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 1 \
    --max-num-seqs 1 \
    --gpu-util 0.35 \
    --skip-lmcache
```

The benchmark starts an in-process `IPCServer` for DaseR, passes only
`socket_path` to vLLM, and verifies cold/warm output consistency.

Use `--transfer-backend gds` when comparing DaseR against LMCache's single-SSD
mode. Use `--transfer-backend iouring-mem` when comparing against LMCache
SSD + local CPU mode. The benchmark derives aligned capacities automatically:
GDS uses only L2/SSD, while `iouring-mem` uses L1 memory plus L2 SSD and maps
LMCache local CPU/local disk to the same sizes. Add `--evict` to derive
`total KV > L2 > L1`; without it, the benchmark derives `L2 > total KV > L1`
so all chunks remain durable in SSD while the memory tier still sees
replacement.

The measured DaseR warm pass uses
`kv_transfer_params={"daser_skip_save": True}` so warm reads are not mixed with
same-pass stores. Store logs around the warm section usually come from draining
cold-pass background stores before measurement; LMCache does not have an
equivalent skip-save flag in this benchmark harness.

---

## Linting and Formatting

```bash
pre-commit run --all-files
ruff check .
ruff format .
mypy .
```
