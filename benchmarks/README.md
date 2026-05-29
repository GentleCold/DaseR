# Benchmarks

DaseR vs LMCache end-to-end inference benchmarks running inside vLLM.

## File structure

| File | Purpose |
|------|---------|
| `bench_common.py` | Shared harnesses, runners, correctness checkers, and reporters |
| `bench_e2e_daser_vs_lmcache.py` | IMDB short-context benchmark |
| `bench_longbench.py` | LongBench long-context benchmark |
| `utils.py` | Shared utilities (prompt loading, tokenisation, GPU selection, sizing) |

## Common setup

Both benchmarks require the Qwen3-8B model and vLLM + LMCache installed. They automatically select the GPU with the most free memory (override with `--gpu-id`).

---

## IMDB benchmark (`bench_e2e_daser_vs_lmcache.py`)

Tests DaseR vs LMCache on IMDB movie reviews with a fixed 2048-token context. Runs 200 prompts through a cold pass (prefill + save) then a warm pass (prefill from cache), comparing elapsed time and throughput.

```bash
python benchmarks/bench_e2e_daser_vs_lmcache.py
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `/data/zwt/model/models/Qwen/Qwen3-8B` | HF model path |
| `--store-dir` | `/data/$USER/daser_test` | Scratch directory for DaseR stores and LMCache disk caches |
| `--imdb` | `/data/zwt/imdb.csv` | Path to IMDB CSV (column: `review`) |
| `--num-prompts` | 200 | Number of reviews to use |
| `--max-input-tokens` | 1792 | Per-prompt token ceiling |
| `--gpu-util` | 0.9 | vLLM `gpu_memory_utilization` |
| `--max-num-seqs` | 64 | vLLM `max_num_seqs` |
| `--gpu-id` | auto | GPU ID (`auto` = most free memory, `0`, `1`, …) |
| `--comparison-mode` | gds | `gds-vs-lmcache-local-ssd` or `iouring-mem-vs-lmcache-local-ssd-mem` |
| `--evict` | *(off)* | Force L2/L1 eviction during the workload |
| `--skip-daser` | *(off)* | Run LMCache only |
| `--skip-lmcache` | *(off)* | Run DaseR only |
| `--out` | *(none)* | Path for JSON results |

### Example: quick smoke test

```bash
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --model /data/zwt/model/models/Qwen/Qwen3-8B \
    --store-dir /data/$USER/daser_test \
    --imdb /path/to/imdb.csv \
    --num-prompts 10 \
    --gpu-id 1
```

### Example: io_uring comparison with forced eviction

```bash
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --model /data/zwt/model/models/Qwen/Qwen3-8B \
    --store-dir /data/$USER/daser_test \
    --imdb /path/to/imdb.csv \
    --comparison-mode iouring-mem-vs-lmcache-local-ssd-mem \
    --evict
```

---

## LongBench benchmark (`bench_longbench.py`)

Tests DaseR vs LMCache on the [LongBench](https://github.com/THUDM/LongBench) dataset. Auto-calculates the longest context that fits in GPU VRAM (clamped to the model's `max_position_embeddings`). Supports iterating over multiple datasets in a single run and produces an aggregate comparison table.

```bash
python benchmarks/bench_longbench.py --skip-correctness
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `/data/zwt/model/models/Qwen/Qwen3-8B` | HF model path |
| `--store-dir` | `/data/$USER/daser_test` | Scratch directory |
| `--longbench-dir` | `/data/ld/longbench_data/data` | LongBench JSONL data directory |
| `--datasets` | `multi_news` | Dataset names (comma-separated) or `all` |
| `--num-prompts` | 0 (all) | Max prompts per dataset |
| `--gpu-util` | 0.9 | vLLM `gpu_memory_utilization` |
| `--max-num-seqs` | 64 | vLLM `max_num_seqs` |
| `--gpu-id` | auto | GPU ID (`auto` = most free memory) |
| `--comparison-mode` | iouring-mem | `gds-vs-lmcache-local-ssd` or `iouring-mem-vs-lmcache-local-ssd-mem` |
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
    --num-prompts 10 \
    --skip-correctness
```

### Example: DaseR-only run on a specific dataset

```bash
python benchmarks/bench_longbench.py \
    --datasets narrativeqa \
    --num-prompts 50 \
    --skip-lmcache \
    --skip-correctness
```

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
