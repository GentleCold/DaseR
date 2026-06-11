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
    --l2-size 10gb \
    --socket-path /tmp/daser.sock \
    --host 0.0.0.0 \
    --port 2026
```

`python -m daser.server` starts both servers in one process:

- HTTP server: FastAPI routes for health checks, document upload/list/get/delete,
  and `/infer`.
- IPC server: Unix socket + msgpack endpoint used only by `DaserConnector`.

The DaseR server reads vLLM's served model id from `/v1/models`, derives KV
geometry from the local model `config.json`, creates `<store-dir>/daser.store`,
and saves metadata to `<store-dir>/daser.index` on SIGTERM/SIGINT shutdown.
The shutdown path is a fast consistent stop: DaseR stops accepting new HTTP and
IPC work, snapshots the currently committed chunk metadata and registered
documents, then closes transfer resources. It does not wait for connector-side
pending writes that have not reached `commit_chunk`.

Runtime values such as `store_path`, `slot_size`, `block_tokens`, and
`model_id` are owned by DaseR and fetched by the connector over IPC.
The exported L2 capacity is the slot-aligned store size, so the transfer layer
does not resize `daser.store` back to the raw `--l2-size` value. If an older
run left a larger raw-size store file, startup truncates it to the aligned
capacity; smaller existing store files are rejected.

`daser.index` includes ring-buffer state, chunk metadata, and the document
registry. Only documents whose `register_document` call completed are restored
by `/documents` after restart. Committed chunks that were not attached to a
document are retained as ordinary cache entries; if the same document content is
uploaded later, matching chunk keys can be reused. Uncommitted bytes in
`daser.store` are ignored after restart.

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
| `--l2-size` | `10 GiB` | L2 SSD capacity; accepts bytes or `mb`/`gb`/`mib`/`gib` and is rounded down to whole KV slots |
| `--l1-size` | `min(1 GiB, --l2-size)` | L1 pinned-memory capacity for `--transfer-mode iouring`; must not exceed `--l2-size` |
| `--transfer-mode` | `iouring` | `iouring` for pinned-memory L1 + SSD L2 transfer or `gds` for kvikio/cuFile GPU-to-SSD transfer |
| `--socket-path` | `/tmp/daser.sock` | IPC server Unix socket path |
| `--host` | `0.0.0.0` | HTTP server bind host |
| `--port` | `2026` | HTTP server bind port |
| `--cache-reuse-mode` | `chunk` | `chunk` for block-aligned document chunk reuse, `prefix` for rolling-prefix slot reuse |

---

## Tests

### Scratch layout and cleanup

Pytest temp files (`tmp_path`, integration `daser.store`, IPC sockets) are written
to a scratch root instead of `/tmp`:

| Priority | Scratch root |
|----------|--------------|
| 1 | `$DASER_TEST_SCRATCH_ROOT` when set |
| 2 | `/data/$USER/daser-test` when `/data` exists and is writable |
| 3 | `<repo>/.test-scratch` on CI runners and machines without `/data` |

Within the scratch root:

| Path | Purpose | Cleanup |
|------|---------|---------|
| `pytest/<pid>-<uuid8>/` | per-session pytest `--basetemp` for `daser.store`, `daser.index`, sockets | removed after that session finishes; stale dirs pruned after 6 hours |
| `results/` | optional artifacts you want to keep across runs | preserved |

Each pytest invocation uses its own basetemp subdirectory so parallel local
runs do not delete another session's active temp files. Passing an explicit
`--basetemp` on the CLI skips managed allocation and cleanup of that path.

Repo-local `.pytest_cache/` is unchanged. Benchmark JSON output from
`--out` is also preserved because it is written to a user-provided path,
not the scratch root.

Override the scratch root when needed:

```bash
export DASER_TEST_SCRATCH_ROOT=/data/<user>/daser-test
pytest -q -m "not integration and not slow"
```

Use the `test_results_dir` fixture when a test needs to persist output under
`results/`:

```python
def test_writes_report(test_results_dir: Path) -> None:
    (test_results_dir / "report.json").write_text("{}")
```

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

The maintained benchmark is the service-oriented vLLM/DaseR/LMCache comparison:

```bash
benchmarks/run_bench.sh \
    --backend all \
    --dataset longbench \
    --model /path/to/model \
    --longbench-dir /path/to/longbench_data \
    --datasets 2wikimqa,hotpotqa_e \
    --store-dir /path/to/benchmark-scratch \
    --max-samples 20 \
    --max-num-seqs 64 \
    --max-inflight 64
```

`--backend all` runs `baseline`, `lmcache`, `daser-chunk`, and `daser-prefix`
under one run directory. Use a specific backend name to run only one row of the
matrix; `--backend daser` still honors `--cache-reuse-mode` for compatibility.

By default the benchmark uses `--gpu-util 0.85` and `--gpu-id auto`, which picks
the GPU with the most free memory and sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before
CUDA libraries initialize. Each invocation creates a unique run root below
`--store-dir`, so repeated runs do not reuse old DaseR stores or LMCache disk
files.

For a quick DaseR smoke run:

```bash
benchmarks/run_bench.sh \
    --backend daser-prefix \
    --dataset longbench \
    --model /path/to/model \
    --store-dir /path/to/benchmark-scratch/smoke-run \
    --longbench-dir /path/to/longbench_data \
    --datasets triviaqa \
    --max-samples 1 \
    --max-num-seqs 1 \
    --max-inflight 1
```

The benchmark starts subprocess services, sends HTTP load, and writes per-backend
manifest and result JSON files under the run directory. Default no-evict runs
pass `--skip-l2` to DaseR and LMCache so load hits are measured from L1 only;
add `--evict` to keep L2 enabled and exercise eviction behavior.

---

## Linting and Formatting

```bash
pre-commit run --all-files
ruff check .
ruff format .
mypy .
```
