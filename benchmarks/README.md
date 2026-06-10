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

## Benchmark Scenarios

Use IMDB for quick performance and correctness validation. IMDB prompts are
shorter and cheaper to run; cap generation to one output token so cold/warm
correctness focuses on whether the cache path preserves the next-token result:

```bash
benchmarks/run_bench.sh \
  --backend all \
  --cache-reuse-mode prefix \
  --dataset imdb \
  --imdb /data/<user>/datasets/imdb.csv \
  --model /data/<user>/model/models/Qwen/Qwen3-8B \
  --store-dir /data/<user>/daser_bench/imdb_prefix \
  --gpu-id 2 \
  --gpu-util 0.85 \
  --max-num-seqs 8 \
  --max-inflight 4 \
  --max-samples 20 \
  --gen-max-tokens 1
```

Use LongBench for long-context performance validation. LongBench samples stress
document-size prompts, KV transfer volume, cache sizing, and warm-phase reuse
under realistic long-text workloads. For LongBench, `--max-samples` is applied
per selected dataset; with the five datasets below, `--max-samples 20` produces
about 100 requests before context deduplication.

```bash
benchmarks/run_bench.sh \
  --backend all \
  --cache-reuse-mode chunk \
  --dataset longbench \
  --model /data/<user>/model/models/Qwen/Qwen3-8B \
  --longbench-dir /data/<user>/dataset/longbench/data \
  --datasets 2wikimqa,hotpotqa_e,2wikimqa_e,musique,triviaqa \
  --store-dir /data/<user>/daser_bench/longbench_chunk \
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
summarization. The recommended quick-validation configuration uses
`--gen-max-tokens 1` and compares cold/warm generated output for correctness.

`--dataset longbench` expects `--longbench-dir /path/to/data`. `--datasets` is a
comma-separated list of JSONL stems. Multiple LongBench datasets are supported
in one run because the loader normalizes them to one sample stream and
interleaves samples by dataset, reducing queue-order bias. `--max-samples`
limits each selected JSONL file independently. LongBench is intended for
long-text performance validation rather than fast correctness smoke tests.

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

In `chunk` mode, baseline vLLM and LMCache receive token-ID prompts constructed
with the same DaseR chunk padding semantics as `/infer`: the chat prefix and
document segment are padded to vLLM block boundaries before the task suffix is
appended. This keeps `prompt_tokens_total`, sizing, and throughput denominators
aligned with DaseR chunk runs. In `prefix` mode, all backends use the ordinary
full prompt string without chunk padding, and context deduplication is disabled
for every backend so repeated full prompts remain part of the workload.

## Cache Semantics

| Backend | Cold phase | Warm phase |
|---------|------------|------------|
| `vllm` | Full-prompt completions | Not applicable |
| `lmcache` | Full-prompt completions populate LMCache | Same prompts repeated after a settle window |
| `daser chunk` | Documents uploaded to `/documents`; this is recorded as upload metadata, not request TTFT | `/infer` uses returned `doc_id` values |
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
- result: cold/warm summaries and per-request details. Baseline vLLM has a
  single `baseline` phase. DaseR chunk mode has cold upload metadata and a
  warm inference phase; DaseR prefix and LMCache have cold and warm request
  phases.

Warm summaries include TTFT mean, latency mean, prompt/completion token totals,
cache hit chunks, total trace chunks, and cache hit rates from multiple
sources:

- `http_trace_cache_hit_rate`: per-request DaseR `/infer` trace hit rate when
  available; OpenAI-compatible vLLM/LMCache responses do not expose this.
- `vllm_external_prefix_cache_hit_rate`: vLLM Prometheus external prefix cache
  hit ratio from `vllm:external_prefix_cache_*` counter deltas.
- `backend_server_cache_hit_rate`: summary hit ratio used for backend
  comparison. DaseR reports its internal
  `daser_external_prefix_cache_*` counters, which the connector records with
  the same queried-token / accepted-token semantics as vLLM's
  `vllm:external_prefix_cache_*` counters. LMCache reports MP server token hit
  counters when available and falls back to request counters only when token
  counters are absent. DaseR control-plane lookup counters are still kept in raw
  metrics as `daser_prometheus_tokens` and `daser_prometheus_requests`.
- `metrics`: raw vLLM Prometheus, backend Prometheus, backend status counter
  deltas, and all named hit-ratio candidates.

For datasets with answer labels, each summary also includes
`answer_contains_accuracy`; datasets without labels report `null`.

## Correctness

IMDB correctness is a cold/warm exact generated-text comparison for backends
that have both request phases. The recommended IMDB setup uses
`--gen-max-tokens 1`, so this checks whether the cache path preserves the next
token under deterministic generation.

LongBench correctness is reported as `answer_contains_accuracy` per request
phase when the dataset provides answer labels. The generated text is checked
for any accepted answer string. For LMCache and DaseR prefix, the runner also
adds `cold_warm_exact_match` because both phases generate the same request set.
DaseR chunk mode uploads documents during cold and only generates during warm,
so it reports warm `answer_contains_accuracy` but has no cold/warm exact-match
comparison.

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
