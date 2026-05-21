# DaseR Server RAG Demo

End-to-end walkthrough of the DaseR HTTP API: upload two short documents, list
them, run inference over both of them, then delete one.

The demo drives the public HTTP API only. Start `python -m daser.server` first,
after `vllm serve` is already listening with the DaseR connector enabled.

## 0. Environment

From the repository root, use the same Python environment for DaseR, vLLM, and
the demo (venv or conda). See [docs/development.md](../../docs/development.md).

```bash
source <venv>/bin/activate   # or: conda activate <env>
pip install -e .

export MODEL_PATH=/path/to/model
source examples/service_demo/env.sh
```

`env.sh` sets `CUDA_DEVICE_ORDER`, `CUDA_VISIBLE_DEVICES`, and vLLM
`kv-transfer-config` helpers (`KV_BASELINE`, `KV_CHUNK`). If `vllm serve` fails
with ``GLIBCXX_3.4.31 not found`` when importing `kvikio`, prepend your active
environment's `lib` directory (the script does this automatically when
`CONDA_PREFIX` or `VIRTUAL_ENV` is set).

Verify connector import:

```bash
python -c "import daser.connector.daser_connector; print('ok')"
```

## 2. Start vLLM

Use the **full** JSON (not `...` placeholders). Pick the socket for the DaseR
instance you are measuring:

```bash
# Baseline run (pairs with --socket-path /tmp/daser_baseline.sock)
vllm serve "$MODEL_PATH" \
  --port "$VLLM_PORT" \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_BASELINE" \
  2>&1 | tee /tmp/vllm_demo.log
```

Chunk-reuse run (restart vLLM after baseline measurements):

```bash
vllm serve "$MODEL_PATH" \
  --port "$VLLM_PORT" \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_CHUNK" \
  2>&1 | tee /tmp/vllm_demo.log
```

`--no-enable-prefix-caching` disables vLLM's in-GPU prefix cache so the demo's
warm-run cache hits come from DaseR. Without it, vLLM may satisfy part of the
prompt from its own prefix cache and hide whether DaseR loaded KV from NVMe.

Log vLLM stdout to a file so connector load lines are available for issue #35:

```bash
vllm serve <model-path> ... 2>&1 | tee /tmp/vllm_demo.log
```

## 3. Start DaseR server

### Interactive walkthrough (prefix mode, port 8080)

```bash
python -m daser.server \
  --host 0.0.0.0 --port 8080 \
  --vllm-base-url http://127.0.0.1:8001 \
  --store-dir /tmp/daser_demo_baseline \
  --store-size 10gb \
  --socket-path /tmp/daser_baseline.sock
```

### Chunk-reuse benchmark instance (port 8081)

Use a **separate store directory and IPC socket** from the baseline instance:

```bash
python -m daser.server \
  --host 0.0.0.0 --port 8081 \
  --vllm-base-url http://127.0.0.1:8001 \
  --store-dir /tmp/daser_demo_chunk \
  --store-size 10gb \
  --socket-path /tmp/daser_chunk.sock \
  --cache-reuse-mode chunk
```

Point vLLM at the socket for the server you are measuring. For A/B runs, restart
vLLM with the matching `socket_path`, or run two vLLM processes on different
ports if you need both servers up at once.

DaseR reads vLLM's served model id from `/v1/models`, derives KV geometry from
`<model-path>/config.json`, and creates `<store-dir>/daser.store`. If vLLM
serves a model alias instead of a local path, add `--model-path <model-path>` to
the DaseR command.

## 4. Run the demo

```bash
python examples/service_demo/demo.py --service-url http://127.0.0.1:8080
```

Expected output (truncated):

```
==> health
{ "status": "ok", "vllm": true }

==> upload doc A
{ "doc_id": "...", "status": "ready", "chunk_count": 1, "chunk_count_cached": 1, "prefill_ms": 180.2 }
...
==> infer over both docs
{ "text": "...", "prompt_tokens": 513, "completion_tokens": 1, "latency_ms": 612.8 }
```

The second upload of the same document is a no-op: the chunk keys hash
to the same values, so `ServerCore.register_document` just attaches a
new `doc_id` to the existing `ChunkMeta.doc_ids` list.

---

## Performance measurement (issue #35)

### What we measure

| Layer | Metric | Source |
|-------|--------|--------|
| HTTP / TTFT proxy | `server_latency_ms` | `/infer` response (`max_tokens=1`) |
| HTTP client | `client_wall_ms` | demo wall clock |
| Control plane | `cache_hits`, `cache_summary` | `trace_cache=true` on `/infer` |
| Data plane (ground truth) | GDS reads, GPU copies, RoPE | vLLM log: `[CONNECTOR] start_load_kv:` |

`cache_hits` reflects `ServerCore.lookup`, not the scheduler's final
`extra_tokens`. Treat connector logs as authoritative for load cost.

### Demo flags

| Flag | Purpose |
|------|---------|
| `--benchmark` | Repeat infer trials; print median/mean `server_latency_ms` |
| `--compare-baseline` | Single-shot baseline vs chunk-reuse with `trace_cache` on both |
| `--ttft-only` | `max_tokens=1` (default for benchmark/compare) |
| `--e2e` | `max_tokens=80` full completion latency |
| `--trials N` | Benchmark repetitions (default 5) |
| `--num-layers N` | Add `estimated_layer_index_copies` / `estimated_rope_block_ops` |
| `--block-tokens N` | Block size for summaries (default 16) |
| `--json-out PATH` | Write machine-readable benchmark payload |

### Execution plan

**Phase 0 — Environment**

1. `source examples/service_demo/env.sh` in every shell (vLLM, DaseR, demo).
2. GPU node with GDS/kvikio working; Qwen3-8B → `--num-layers 36`.
3. vLLM with `--no-enable-prefix-caching`; log file at `/tmp/vllm_demo.log`.
4. One vLLM process uses **one** `socket_path`; switch `KV_BASELINE` / `KV_CHUNK`
   and restart vLLM between A/B runs.

**Phase 1 — Functional smoke**

```bash
python examples/service_demo/demo.py --service-url http://127.0.0.1:8081
```

Confirm uploads succeed and `cache_hits` is non-empty on infer.

**Phase 2 — TTFT benchmark (primary, issue #35)**

1. Start baseline server (`--cache-reuse-mode prefix`, port 8080, store A, socket A).
2. Start chunk server (`--cache-reuse-mode chunk`, port 8081, store B, socket B).
3. Run vLLM against socket A; upload + benchmark; repeat for socket B **or** use
   two vLLM instances if both servers stay up.
4. Run:

```bash
python examples/service_demo/demo.py --benchmark \
  --baseline-url http://127.0.0.1:8080 \
  --chunk-reuse-url http://127.0.0.1:8081 \
  --trials 10 \
  --num-layers 36 \
  --json-out /tmp/daser_demo_benchmark.json
```

5. Grep connector load lines from the vLLM log for the chunk-reuse infer:

```bash
grep 'start_load_kv' /tmp/vllm_demo.log
```

Record `GDS reads` and `GPU copies` next to the JSON summary.

**Phase 3 — Optional E2E latency**

```bash
python examples/service_demo/demo.py --benchmark --e2e \
  --baseline-url http://127.0.0.1:8080 \
  --chunk-reuse-url http://127.0.0.1:8081 \
  --json-out /tmp/daser_demo_e2e.json
```

**Phase 4 — Analysis checklist**

- Compare `summary.delta_median_ms` (chunk reuse − baseline).
- If chunk reuse is slower despite hits, check:
  - `cache_summary.hit_count` and `estimated_gds_reads`
  - `estimated_rope_block_ops` when `--num-layers` is set
  - `contiguous_hit_tokens` vs `reused_token_sum` (gaps reduce real reuse)
- Correlate with connector log lines; HTTP estimates are not a substitute.

**Phase 4b — PyTorch profiler (connector load segments)**

Requires ``pip install 'daser[profile]'`` (installs ``tensorboard`` and
``torch-tb-profiler``; the latter provides the **PYTORCH PROFILER** tab).

Enable profiling in vLLM ``kv_connector_extra_config``:

```json
{
  "socket_path": "/tmp/daser_chunk.sock",
  "load_profile": true,
  "load_profile_tensorboard_dir": "examples/service_demo/out/tensorboard"
}
```

Or run the helper script (upload + warm infer, prints segment table + TensorBoard traces):

```bash
export MODEL_PATH=/path/to/model
source examples/service_demo/env.sh
bash examples/service_demo/run_profile_load.sh
grep -A 30 "load profile report" /tmp/daser_load_profile.log
```

The vLLM log report has two tables:

- **CUDA/CPU (PyTorch profiler)**: GPU time per segment; `daser::gds_read` is often
  **CPU-only** because NVMe I/O runs on the connector background thread.
- **Wall clock**: end-to-end milliseconds with `cuda.synchronize()` where applicable.

Segment labels in the report and TensorBoard trace:

| Label | Stage |
|-------|--------|
| `daser::alloc_staging` | GPU staging buffer allocation |
| `daser::gds_read` | NVMe read (compat/GDS) |
| `daser::staging_to_kv` | Per-chunk copy wrapper |
| `daser::index_copy` | Layer `index_copy_` into vLLM KV |
| `daser::kv_scale` | Optional K/V scale |
| `daser::rope_delta` | Chunk-position RoPE correction |

**Visualize in TensorBoard** (works well with remote GPU servers via SSH tunnel):

```bash
# On the server — starts TensorBoard on 127.0.0.1:6006
bash examples/service_demo/view_profile_tensorboard.sh
```

On your **local** machine (separate terminal):

```bash
ssh -L 6006:127.0.0.1:6006 user@your-gpu-server
```

Then open http://127.0.0.1:6006/#pytorch_profiler and search for ``daser::``.

If you see *"There's no dashboard by the name of pytorch_profiler"*, install the
plugin and **restart** TensorBoard:

```bash
pip install torch-tb-profiler
# stop the running tensorboard, then:
bash examples/service_demo/view_profile_tensorboard.sh
```

Traces are written under ``examples/service_demo/out/tensorboard/`` in the repo
workspace (one ``*.pt.trace.json`` per profiled load).

**Phase 5 — Report for #35**

Attach `/tmp/daser_demo_benchmark.json`, vLLM log excerpt, server CLI flags, model
name, and a one-line conclusion (regression confirmed / not reproduced / fixed).

### Acceptance criteria

- Chunk-reuse run shows `hit_count >= 1` and `server_latency_ms` statistics.
- Connector log captured for the same run.
- Baseline and chunk-reuse both used `trace_cache=true` in benchmark mode.
- TTFT mode uses `max_tokens=1` unless explicitly running `--e2e`.

---

## Troubleshooting

- **Connection refused to the socket**: make sure `daser.server` is running
  before sending demo requests, and that vLLM uses the same `socket_path`.
- **`doc N has no cached tokens for prompt rebuild`**: a doc must be
  uploaded through `/documents` before `/infer` can use it; inferring
  against a doc that was evicted requires re-uploading.
- **vLLM rejects `max_tokens=0`**: the service uses `max_tokens=1` for
  prefill and discards the single decoded token.
- **Baseline vs chunk-reuse on one vLLM**: only one `socket_path` is active per
  vLLM process; switch the socket in `kv_connector_extra_config` between runs or
  run two vLLM processes.
