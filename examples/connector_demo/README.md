# DaseR Connector Demo

Minimal runnable demo of DaseR as a **KV cache backend** for vLLM via
`DaserConnector` — no HTTP service layer involved. For the higher-level
HTTP service demo (document upload → list → infer → delete), see
[`examples/service_demo/`](../service_demo/README.md).

## Files

| File | Role |
|------|------|
| `run_daser_server.py` | Thin CLI wrapper over `daser.server.run_server`; pre-allocates the store file |
| `vllm_cold_warm.py` | Builds a vLLM `LLM` wired to `DaserConnector`, runs cold → warm generation |

Both scripts are intentionally small so you can read them start-to-finish before plugging DaseR into your own pipeline.

## Prerequisites

- DaseR installed in the active venv (`pip install -e .` from the repo root).
- vLLM installed in the same venv.
- A CUDA GPU with enough VRAM for your model.
- A directory on your NVMe for the store file.

The defaults assume Qwen3-8B (36 layers, 8 KV heads, bf16). If you use a different model, override `--num-layers`, `--num-kv-heads`, `--head-dim`, and `--slot-size` accordingly.

## Run

**Terminal A — start the server:**

```bash
mkdir -p /tmp/daser_example
python examples/connector_demo/run_daser_server.py \
    --store-path  /tmp/daser_example/daser.store \
    --socket-path /tmp/daser_example/daser.sock \
    --index-path  /tmp/daser_example/daser.index
```

Wait for `[SERVER] DaseR server ready`.

**Terminal B — run the cold → warm demo:**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python examples/connector_demo/vllm_cold_warm.py \
    --model       /path/to/Qwen3-8B \
    --store-path  /tmp/daser_example/daser.store \
    --socket-path /tmp/daser_example/daser.sock
```

The summary line at the end looks like:

```
[EXAMPLE] summary: cold=<t1>s warm=<t2>s speedup=<x>x match=True
```

Stop the server with `Ctrl-C`; it snapshots the index to `--index-path` on shutdown.

## See also

- `docs/development.md` — full operator guide and env setup.
- `docs/design/` — architecture and data-flow docs.
- `tests/integration/test_vllm_e2e.py` — the pytest version of this demo (starts the server in-process and asserts on correctness / speedup).
