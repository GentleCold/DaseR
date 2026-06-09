# Benchmarks

The maintained benchmark stack is service-oriented: one script starts vLLM,
DaseR, or LMCache services, one script sends load, and one shell entry point
runs the full comparison. The old in-process IMDB and LongBench runners were
removed so all benchmark paths use the same dataset, prompt, sizing, and HTTP
load-generation logic.

## File Structure

| Path | Purpose |
|------|---------|
| `bench_start_servers.py` | Starts one backend and writes a run manifest |
| `bench_load.py` | Loads IMDB or LongBench samples and sends cold/warm HTTP load |
| `run_bench.sh` | End-to-end orchestration for one or more backends |
| `utils/` | Shared dataset, prompt, sizing, server, loadgen, and metric helpers |
| `bench_rope_apply.py` | RoPE microbenchmark, unchanged |
| `bench_staging_restore.py` | Staging restore microbenchmark, unchanged |

## Common Setup

Use the vLLM environment and put benchmark scratch data under `/data`, not
`/tmp`:

```bash
source /data/<user>/vllm/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Full Run

```bash
benchmarks/run_bench.sh \
  --backend all \
  --cache-reuse-mode chunk \
  --dataset longbench \
  --model /data/<user>/model/models/Qwen/Qwen3-8B \
  --longbench-dir /data/<user>/dataset/longbench/data \
  --datasets 2wikimqa,hotpotqa_e,2wikimqa_e,musique,triviaqa \
  --store-dir /data/<user>/daser_bench/unified \
  --gpu-id 2 \
  --gpu-util 0.85 \
  --max-num-seqs 32 \
  --max-inflight 32 \
  --max-samples 20 \
  --max-context-tokens 40000
```

`--backend all` runs `vllm`, `lmcache`, and `daser` sequentially. Use
`--backend daser`, `--backend lmcache`, or `--backend vllm` for a single
backend.

## Dataset Modes

`--dataset imdb` expects `--imdb /path/to/imdb.csv` with a `review` column. The
review text becomes the document context and the shared task is sentiment
summarization.

`--dataset longbench` expects `--longbench-dir /path/to/data`. `--datasets` is a
comma-separated list of JSONL stems. Multiple LongBench datasets are supported
in one run because the loader normalizes them to one sample stream and
interleaves samples by dataset, reducing queue-order bias.

## Prompt Construction

All service benchmarks use the same RAG chat prompt shape:

```text
system: You are a helpful assistant answering questions using the following documents.
user: Documents:
<context>

Task: <question>
assistant:
```

When the tokenizer has `apply_chat_template`, the prompt is rendered through the
model chat template with `enable_thinking=False`; otherwise a simple
role-prefixed fallback is used.

## Cache Semantics

| Backend | Cold phase | Warm phase |
|---------|------------|------------|
| `vllm` | Full-prompt completions | Not applicable |
| `lmcache` | Full-prompt completions populate LMCache | Same prompts repeated after a settle window |
| `daser chunk` | Documents uploaded to `/documents` | `/infer` uses returned `doc_id` values |
| `daser prefix` | Full prompts sent to vLLM to store rolling-prefix slots | Same full prompts repeated in the same service lifetime |

DaseR defaults to `--transfer-mode iouring`. vLLM prefix caching is disabled for
all service modes so external KV storage is the measured reuse path.

LMCache warm traffic waits for the MP server status queues to drain before the
second pass. The runner polls `/status` until the store controller has no
pending or in-flight L2 store work and the prefetch controller has no queued or
in-flight lookup/load work, then applies the configured settle window.

## Sizing

`bench_load.py` tokenizes the workload and derives workload blocks using the
shared Qwen3-8B slot size. Without `--evict`, the end-to-end runner passes
`--skip-l2` to DaseR and LMCache so load hits are measured from L1 only:
DaseR keeps logical slots but does not create a store file, and LMCache starts
without an MP `--l2-adapter`. With `--evict`, both backends keep their L2 tiers
enabled and L1/L2 are chosen below workload size while still fitting the
largest single prompt. Machine caps come from host `MemAvailable` and free disk
under the run store directory. `--max-l1-size` and `--max-l2-size` can lower
those caps.

Reports include both derived sizes and the manifest sizes passed to the
services:

- `derived_l1_size_bytes`, `derived_l2_size_bytes`
- `derived_l1_size`, `derived_l2_size`
- `manifest_l1_size_bytes`, `manifest_l2_size_bytes`
- `manifest_l1_size`, `manifest_l2_size`
- `planned_skip_l2`, `manifest_skip_l2`, `storage_tier`
- `lmcache_l1_gb`, `lmcache_l2_gb`

Human-readable sizes use MiB below 1 GiB and GiB otherwise. DaseR receives the
manifest L1/L2 byte sizes directly. LMCache's L1 CLI accepts only integer GiB,
so the runner rounds the LMCache L1 value up for service startup and records it
as `lmcache_l1_gb`. In no-evict runs, the derived L2 size is a logical DaseR
slot bound only and `lmcache_l2_gb` is `null` because LMCache has no L2
adapter. In evict runs, LMCache's current FS L2 adapter CLI does not expose an
L2 capacity limit, so the report should treat LMCache L2 as bounded by the
filesystem free space rather than by the derived DaseR L2 size.

## Direct Script Usage

Start a backend:

```bash
python benchmarks/bench_start_servers.py \
  --backend daser \
  --model /data/<user>/model/models/Qwen/Qwen3-8B \
  --store-dir /data/<user>/daser_bench/run1/daser \
  --gpu-id 2 \
  --l1-size 256gib \
  --l2-size 300gib \
  --cache-reuse-mode chunk \
  --skip-l2
```

Send load:

```bash
python benchmarks/bench_load.py \
  --manifest /data/<user>/daser_bench/run1/daser/manifest.json \
  --dataset longbench \
  --longbench-dir /data/<user>/dataset/longbench/data \
  --datasets 2wikimqa,hotpotqa_e \
  --max-samples 20 \
  --max-inflight 32 \
  --out /data/<user>/daser_bench/run1/daser/results.json
```

## Output

Each backend writes `results.json` containing:

- manifest: backend, endpoints, store paths, and configured L1/L2 sizes
- config: dataset, sample count, token/block counts, derived sizing
- result: cold/warm summaries and per-request details

Warm summaries include TTFT mean, latency mean, prompt/completion token totals,
cache hit chunks, total trace chunks, and cache hit rates from multiple
sources:

- `http_trace_cache_hit_rate`: per-request DaseR `/infer` trace hit rate when
  available; OpenAI-compatible vLLM/LMCache responses do not expose this.
- `vllm_external_prefix_cache_hit_rate`: vLLM Prometheus external prefix cache
  hit ratio from `vllm:external_prefix_cache_*` counter deltas.
- `backend_server_cache_hit_rate`: DaseR `/metrics` cache lookup counters or
  LMCache MP Prometheus lookup counters, depending on backend. DaseR uses the
  token hit ratio when `daser_cache_matched_tokens_total` and
  `daser_cache_requested_tokens_total` are present, with request hit ratio as
  fallback.
- `metrics`: raw vLLM Prometheus, backend Prometheus, backend status counter
  deltas, and all named hit-ratio candidates.

For datasets with answer labels, each summary also includes
`answer_contains_accuracy`; datasets without labels report `null`.

## Correctness

Service-mode correctness is checked through deterministic request setup and the
per-phase `answer_contains_accuracy` field for datasets that provide answers.
Utility-level exact cold/warm token/text correctness remains available in
`benchmarks.utils.metrics` for low-level tests.

## Troubleshooting

If vLLM reports CUDA fork errors, ensure:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

If startup fails due to free memory, lower `--gpu-util`, reduce
`--max-num-seqs`, set `--max-model-len`, or wait for an H800 to become free.
The starter checks benchmark ports before launching and records
`cuda_visible_devices` in `pids.json` to make GPU placement auditable.
