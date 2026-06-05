# Benchmarks

DaseR vs LMCache end-to-end inference benchmarks running inside vLLM.

## File structure

| File | Purpose |
|------|---------|
| `bench_common.py` | Shared harnesses, runners, utilities, correctness checkers, and reporters |
| `bench_imdb.py` | IMDB short-context benchmark |
| `bench_longbench.py` | LongBench long-context benchmark |

## Common setup

Both benchmarks require the Qwen3-8B model and vLLM + LMCache installed. They automatically select the GPU with the most free memory (override with `--gpu-id`).

---

## IMDB benchmark (`bench_imdb.py`)

Tests DaseR vs LMCache on IMDB movie reviews with a fixed 2048-token context. Runs 200 prompts through a cold pass (prefill + save) then a warm pass (prefill from cache), comparing elapsed time and throughput.

```bash
python benchmarks/bench_imdb.py \
    --model /path/to/model \
    --imdb /path/to/imdb.csv \
    --store-dir /path/to/scratch 
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HF model path |
| `--store-dir` | *(required)* | Scratch directory for DaseR stores and LMCache disk caches |
| `--imdb` | *(required)* | Path to IMDB CSV (column: `review`) |
| `--num-prompts` | 200 | Number of reviews to use |
| `--max-input-tokens` | 1792 | Per-prompt token ceiling |
| `--gpu-util` | 0.9 | vLLM `gpu_memory_utilization` |
| `--max-num-seqs` | 64 | vLLM `max_num_seqs` |
| `--gpu-id` | auto | GPU ID (`auto` = most free memory, `0`, `1`, …) |
| `--comparison-mode` | gds | `gds-vs-lmcache-local-ssd` or `iouring-mem-vs-lmcache-local-ssd-mem` |
| `--cache-reuse-mode` | prefix | DaseR cache reuse strategy: `prefix` or `chunk` |
| `--evict` | *(off)* | Force L2/L1 eviction during the workload |
| `--skip-daser` | *(off)* | Run LMCache only |
| `--skip-lmcache` | *(off)* | Run DaseR only |
| `--out` | *(none)* | Path for JSON results |

### Example: quick smoke test

```bash
python benchmarks/bench_imdb.py \
    --model /path/to/model \
    --store-dir /path/to/scratch \
    --imdb /path/to/imdb.csv \
    --num-prompts 10 \
    --gpu-id 1
```

### Example: io_uring comparison with forced eviction

```bash
python benchmarks/bench_imdb.py \
    --model /path/to/model \
    --store-dir /path/to/scratch \
    --imdb /path/to/imdb.csv \
    --comparison-mode iouring-mem-vs-lmcache-local-ssd-mem \
    --evict
```

---

## LongBench benchmark (`bench_longbench.py`)

Tests DaseR vs LMCache on the [LongBench](https://github.com/THUDM/LongBench) dataset. Auto-calculates the longest context that fits in GPU VRAM (clamped to the model's `max_position_embeddings`). Supports iterating over multiple datasets in a single run and produces an aggregate comparison table.

```bash
python benchmarks/bench_longbench.py \
    --model /path/to/model \
    --longbench-dir /path/to/longbench_data \
    --skip-correctness \
    --store-dir /path/to/scratch
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HF model path |
| `--store-dir` | *(required)* | Scratch directory |
| `--longbench-dir` | *(required)* | LongBench JSONL data directory |
| `--datasets` | `multi_news` | Dataset names (comma-separated) or `all` |
| `--num-prompts` | 0 (all) | Max prompts per dataset |
| `--gpu-util` | 0.9 | vLLM `gpu_memory_utilization` |
| `--max-num-seqs` | 64 | vLLM `max_num_seqs` |
| `--gpu-id` | auto | GPU ID (`auto` = most free memory) |
| `--comparison-mode` | iouring-mem | `gds-vs-lmcache-local-ssd` or `iouring-mem-vs-lmcache-local-ssd-mem` |
| `--cache-reuse-mode` | prefix | DaseR cache reuse strategy: `prefix` or `chunk` |
| `--max-model-len` | 0 (auto) | Override auto-calculated `max_model_len` (0 = from VRAM) |
| `--max-input-tokens` | 0 (auto) | Per-prompt token ceiling (0 = `max_model_len` − 256) |
| `--evict` | *(off)* | Force L2/L1 eviction |
| `--skip-daser` | *(off)* | Run LMCache only |
| `--skip-lmcache` | *(off)* | Run DaseR only |
| `--skip-correctness` | *(off)* | Skip correctness checks (faster iteration) |
| `--out` | *(none)* | Path for JSON results |

### Example: quick smoke test

```bash
python benchmarks/bench_longbench.py \
    --model /path/to/model \
    --store-dir /path/to/scratch \
    --longbench-dir /path/to/longbench_data \
    --num-prompts 10 \
    --skip-correctness
```

### Example: DaseR-only run on a specific dataset

```bash
python benchmarks/bench_longbench.py \
    --model /path/to/model \
    --store-dir /path/to/scratch \
    --longbench-dir /path/to/longbench_data \
    --datasets narrativeqa \
    --num-prompts 50 \
    --skip-lmcache \
    --skip-correctness
```

---

## E2E Stress benchmark (`bench_e2e_stress.py`)

A standalone end-to-end stress benchmark that compares vLLM vs vLLM+LMCache vs vLLM+DaseR on [LongBench](https://github.com/THUDM/LongBench) datasets. Starts all servers via subprocess, sends concurrent completion requests, and scores generated answers against ground truth. Decoupled from the DaseR codebase so it is not affected by internal refactors.

### Modes

| `--mode` | Description |
|----------|-------------|
| `vllm` | Vanilla vLLM with no KV connector (baseline) |
| `lmcache` | vLLM + LMCache MP connector (cold pass → warm pass) |
| `daser` | vLLM + DaseR connector (chunk or prefix cache reuse) |
| `all` | Sequential runs of all three modes |

### Cache reuse strategies

DaseR mode supports two cache reuse strategies via `--cache-reuse-mode`:

| Mode | Connector | Behavior |
|------|-----------|----------|
| `chunk` (default) | `ChunkReuseStrategy` | Uploads documents via HTTP API, block-aligned content-hash lookup. Deduplication is enabled by default to avoid biasing against vLLM/LMCache. |
| `prefix` | `PrefixReuseStrategy` | Sends prompts directly to vLLM `/v1/completions`. DaseR connector caches rolling-prefix KV. Dedup is **disabled** to allow duplicate contexts to trigger cache hits. |

### Basic usage

```bash
python benchmarks/bench_e2e_stress.py \
    --mode daser \
    --model /path/to/model \
    --data-dir /path/to/longbench_data \
    --store-dir /path/to/scratch_bench \
    --socket-path /path/to/daser.sock
```

### Required flags

| Flag | Description |
|------|-------------|
| `--mode` | `vllm`, `lmcache`, `daser`, or `all` |
| `--model` | HF model path served by vLLM |
| `--data-dir` | Directory containing LongBench JSONL files |
| `--store-dir` | Scratch directory for KV stores, LMCache disk files, and logs |

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | 5 default QA datasets | Comma-separated dataset names |
| `--max-samples` | 20 | Max samples per dataset (0 = all) |
| `--max-context-tokens` | 0 (no limit) | Filter out samples whose full prompt exceeds this token count (e.g. 40000 for Qwen3-8B) |
| `--max-inflight` | 32 | Max concurrent in-flight requests |
| `--gpu-id` | auto | GPU device index |
| `--gpu-util` | 0.85 | vLLM `gpu_memory_utilization` |
| `--max-num-seqs` | 32 | vLLM `max_num_seqs` |
| `--l2-size` | 300gib | DaseR L2 / LMCache disk capacity |
| `--l1-size` | 256gib | DaseR L1 / LMCache CPU memory capacity |
| `--socket-path` | /tmp/daser.sock | Unix domain socket path for DaseR IPC (required for `--mode daser` or `all`) |
| `--cache-reuse-mode` | chunk | `chunk` or `prefix` (DaseR only) |
| `--no-dedup-context` | off | Disable context deduplication (all modes) |
| `--vllm-port` | 8001 | vLLM HTTP port |
| `--daser-port` | 2026 | DaseR HTTP port |
| `--gen-max-tokens` | 128 | Max generated tokens per request |
| `--gen-temperature` | 0.0 | Generation temperature |
| `--timeout` | 600 | Per-request HTTP timeout (seconds) |
| `--startup-timeout` | 180 | Server health-check timeout (seconds) |
| `--gpu-monitor-secs` | 15 | GPU utilisation logging interval (0 = disable) |
| `--output` | auto-generated | JSON results file path |
| `--keep-alive` | off | Keep servers running after benchmark |

### Example: DaseR prefix mode with context-length filtering

```bash
python benchmarks/bench_e2e_stress.py \
    --mode daser \
    --model /path/to/model \
    --data-dir /path/to/longbench_data \
    --store-dir /path/to/scratch_bench \
    --socket-path /path/to/daser.sock \
    --cache-reuse-mode prefix \
    --datasets qmsum \
    --max-samples 0 \
    --max-context-tokens 40000
```

### Example: compare DaseR chunk mode vs vLLM baseline

```bash
python benchmarks/bench_e2e_stress.py \
    --mode all \
    --model /path/to/model \
    --data-dir /path/to/longbench_data \
    --store-dir /path/to/scratch_bench \
    --socket-path /path/to/daser.sock \
    --datasets triviaqa,2wikimqa \
    --max-samples 50
```

### Example: DaseR prefix vs vLLM with fair dedup settings

To compare DaseR prefix mode against vLLM, run separately and disable dedup on both:

```bash
# DaseR prefix (dedup auto-disabled)
python benchmarks/bench_e2e_stress.py \
    --mode daser \
    --model /path/to/model \
    --data-dir /path/to/longbench_data \
    --store-dir /path/to/scratch_bench \
    --socket-path /path/to/daser.sock \
    --cache-reuse-mode prefix \
    --datasets qmsum \
    --max-samples 0 \
    --max-context-tokens 40000

# vLLM baseline (explicitly disable dedup for 1:1 sample comparison)
python benchmarks/bench_e2e_stress.py \
    --mode vllm \
    --model /path/to/model \
    --data-dir /path/to/longbench_data \
    --store-dir /path/to/scratch_bench \
    --socket-path /path/to/daser.sock \
    --datasets qmsum \
    --max-samples 0 \
    --max-context-tokens 40000 \
    --no-dedup-context
```

### Output

Each mode produces a per-dataset summary table:

```
================================================================================
  Mode: daser
================================================================================
Dataset                     #  Err  Contains  TTFT_mean  TTFT_p50  TTFT_p99   Lat_mean
--------------------------------------------------------------------------------------
qmsum                     200    0     0.0%   40306.7  34661.5 101114.3   57241.3
--------------------------------------------------------------------------------------
AGGREGATE                 200          0.0%
```

JSON results are saved per-mode to the `--output` path (with mode suffix). In `--mode all`, an additional combined comparison file is written.

---

## Output format

Both benchmarks print a per-run comparison table:

```
Metric                                     DaseR             LMCache
------------------------------------------------------------------------
cold elapsed                              1.14 s              2.40 s
warm elapsed                              0.13 s              0.13 s
cold tok/s (prompt)                       14,613               6,921
warm tok/s (prompt)                      125,827             131,298
exact mismatches                               1                   0
warm/cold speedup                          8.61×              18.97×
------------------------------------------------------------------------
DaseR warm tok/s / LMCache warm tok/s = 0.96×
```

The LongBench benchmark additionally prints an **aggregate table** after all datasets complete, showing per-dataset warm throughput ratios and correctness parity.

Correctness checks compare cold vs warm generated outputs (exact token IDs and text match). Parity is OK when DaseR mismatches ≤ LMCache mismatches + 1.

## GPU memory notes

- The LongBench benchmark auto-calculates `max_model_len` from free VRAM (Qwen3-8B bf16: ~16 GB weights, ~144 KB KV cache per token).
- Reduce `--gpu-util` or set `--max-model-len` explicitly if you encounter OOM.
- For GPUs with 24 GB (e.g., RTX 4090), max tokens ≈ 17K at `gpu_util=0.85`.
- For GPUs with 80 GB (e.g., H800), max tokens is clamped to the model's native limit (40,960 for Qwen3-8B).

## Troubleshooting

**`Cannot re-initialize CUDA in forked subprocess`**: The LongBench benchmark handles this automatically. If running other scripts, set `VLLM_WORKER_MULTIPROC_METHOD=spawn` or `CUDA_VISIBLE_DEVICES` before launching.

**`max_model_len is greater than the derived max_model_len`**: The benchmark auto-clamps to the model's `max_position_embeddings`. If overriding and the model uses RoPE, positions beyond this value produce NaN.
