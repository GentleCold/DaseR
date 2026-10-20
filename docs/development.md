# Development Guide

## Environment

```bash
source <venv>/bin/activate
pip install -e .[dev]
```

Use the same Python environment for DaseR and vLLM integration tests so the
connector imports resolve consistently.

DaseR targets vLLM 0.30.0 and does not support earlier releases. Install vLLM
and its PyTorch build first (vLLM 0.30.0 pins `torch==2.13.0`); take
torch-family wheels from the PyTorch index that matches your CUDA version, then
install DaseR into that environment. The connector asks vLLM for the
block-outermost `BLNHC` KV cache layout through
`get_required_kvcache_layout`, so every KV block holds all attention layers
contiguously and one chunk moves as a single strided copy.

---

## Running vLLM and DaseR

DaseR uses a vLLM-first startup sequence. Start `vllm serve` with only the
connector identity and the IPC socket path:

```bash
vllm serve /path/to/model \
    --port 8001 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"socket_path":"/tmp/daser.sock"}}'
```

A lookup hit is not pinned, so a chunk can be evicted before its load runs.
`kv_load_failure_policy` `recompute` makes vLLM recompute those blocks; the
default `fail` aborts the request.

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

- HTTP server: FastAPI routes for the UI, health, metrics, documents, `/infer`,
  `/drain` and optional queue hints.
- IPC server: Unix socket + msgpack endpoint used by `DaserConnector` and tools.

The DaseR server reads vLLM's served model id from `/v1/models`, derives KV
geometry from the local model `config.json`, and creates or validates
`<store-dir>/daser.store` when L2 is enabled. Only raw + L2 saves metadata to
`<store-dir>/daser.index` on SIGTERM/SIGINT shutdown.
The shutdown path is a fast consistent stop: DaseR stops accepting new HTTP and
IPC work, drains transfer work already submitted to the server, saves published
chunks with a complete L2 copy and the document registry, then closes transfer
resources. It does not wait for connector-side writes not yet submitted.

Runtime values such as `store_path`, `slot_size`, `block_tokens`, `model_id`,
`tier_grain`, and the per-tier lane sizes are owned by DaseR and fetched by the
connector over IPC. The connector never computes L1 or L2 offsets: store spans
name staging bytes and chunk slots, load spans name `(start_slot, num_slots)`,
and the server resolves both (see
[docs/design/tier-residency.md](design/tier-residency.md)). The exported L2
capacity is the grain-aligned store size. If an older run left a larger store
file, startup truncates it to the aligned capacity; smaller existing store
files are rejected.

`daser.index` includes eligible chunk metadata with their L2 extents, the tier
geometry, and the document registry. Only documents whose `register_document` call completed are restored
by `/documents` after restart. Published chunks with a complete L2 copy that
were not attached to a document are retained as ordinary cache entries; if the same document content is
uploaded later, matching chunk keys can be reused. Uncommitted bytes in
`daser.store` are ignored after restart.

L1 is always empty after restart. `--skip-l2` neither creates a store file nor
reads/writes an index. `compressed-online` also starts cold, deletes any old
index at startup, and does not save one at shutdown. The document API requires
chunk reuse; prefix mode rejects document upload.

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
| `--l2-size` | `10 GiB` | L2 SSD capacity; accepts bytes or `mb`/`gb`/`mib`/`gib`. The store file is exactly this size rounded down to `--tier-grain` per TP lane |
| `--l1-size` | `1 GiB` | Total pinned host memory for `iouring`, independent of L2 size; includes temporary staging, with the remainder assigned to L1 |
| `--tier-grain` | `64 KiB` | Allocation grain for both tiers; must be a power of two and a multiple of 4 KiB |
| `--transfer-mode` | `iouring` | `iouring` for the current pinned-memory L1 + SSD L2 path; `gds` remains an optional kvikio/cuFile compatibility backend and is not the current validation path |
| `--socket-path` | `/tmp/daser.sock` | IPC server Unix socket path |
| `--host` | `0.0.0.0` | HTTP server bind host |
| `--port` | `2026` | HTTP server bind port |
| `--cache-reuse-mode` | `chunk` | `chunk` for block-aligned document chunk reuse, `prefix` for rolling-prefix slot reuse |
| `--block-tokens` | `16` | Must match vLLM `--block-size` |
| `--tensor-parallel-size` | `1` | Must match vLLM TP size |
| `--skip-l2` | off | Volatile L1 only; incompatible with GDS |
| `--storage-format` | `raw` | `raw` or `compressed-online`; online compression requires iouring, prefix reuse, TP=1, BF16 and block size 128 |
| `--compressed-codebooks` | unset | Optional offline model-bound calibration directory for compressed-online |
| `--l1-accounting` | `stored` | Charge stored bytes or raw-equivalent bytes (`raw`) in L1 |
| `--prefetch-lease-budget` | `0` | Cap bytes pinned by scheduler leases; 0 means no budget cap |
| `--l2-read-stripe` | `0` | Optional parallel read stripe size, a multiple of 4 KiB; 0 disables splitting |
| `--extra-kv-layers` | `0` | Additional attention layers beyond the model config, e.g. a drafter |

The server places chunks in L1 and L2 and evicts each tier with its own
leaf-first LRU, identically for raw and compressed-online (see
[docs/design/tier-residency.md](design/tier-residency.md)). The iouring backend
always merges physically adjacent L2 reads into bounded reads; this is not
configurable. A new store file is created NOCOW and fallocated; failure to set
NOCOW is logged and the server continues.

`DASER_EXCLUSIVE_TIERS=1` enables the experimental L1-first store policy when
both tiers exist. Background demotion writes dirty L1 copies to L2; promotion
keeps the L2 source. This mode does not grant prefetch leases, so layerwise
loading falls back to ordinary loads. Default write-through remains unchanged
when the variable is unset.

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
python benchmarks/run_bench.py \
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
`--cache-reuse-mode` is only passed to DaseR rows; baseline and LMCache use the
same prepared prompts and record `reuse_mode: none`.
Scheduler-side DaseR prefetch is off by default. Pass `--daser-prefetch` to
enable it with two workers, or use the compatibility override
`--daser-prefetch-max-requests N` (including `0` to force it off). The numeric
override takes precedence over the boolean flag, and the effective state is
recorded in the DaseR manifest.
The Python entry point prints stage separators for prepare, backend start,
cold/warm load, a final `== COMPARISON SUMMARY ==` with cold, warm,
correctness, elapsed-time, and throughput fields, and the final `run_root`.

By default the benchmark uses `--gpu-util 0.85` and `--gpu-id auto`, which picks
the GPU with the most free memory and sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before
CUDA libraries initialize. Each invocation creates a unique run root below
`--store-dir`, so repeated runs do not reuse old DaseR stores or LMCache disk
files.

For a quick DaseR smoke run:

```bash
python benchmarks/run_bench.py \
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
both no-evict and evict sizing cap L1 at 80% of current host `MemFree`. Add
`--evict` to keep L2 enabled and exercise eviction behavior. Evict sizing targets
L2 capacity for the workload plus headroom, subject to the machine cap, and
sizes L1 to 80% of the workload by default with a largest-prompt lower bound.
`--evict-l1-fraction` changes that share and must be in `(0, 1)` with `--evict`.
Evict runs fail fast when the requested L1 size exceeds the MemFree cap.

For online compression, select `--storage-format compressed-online` and pass
`--daser-codebooks /path/to/calibration` to the benchmark runner if calibrated
codebooks are needed; the runner forwards them as server
`--compressed-codebooks`. Otherwise it uses the generic palette.

---

## Linting and Formatting

```bash
pre-commit run --all-files
ruff check .
ruff format .
mypy .
```
