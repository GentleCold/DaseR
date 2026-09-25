# Online Pack: Cold/Warm Eviction and PPT Prefix Sweeps

Date: 2026-09-09. Packed production source: `3ad1c7c` on
`perf/tilelang-ttft-20-target`. Production, vLLM, and LMCache code were not
modified for these measurements.

## Common Configuration

- Qwen3-8B BF16, H800 PCIe, TP1, block size 128.
- 200 measured requests, concurrency 8, one output token, seed 42.
- vLLM prefix caching disabled, DaseR prefetch disabled, io_uring with L2 enabled.
- Retained TileLang online pack: `DASER_ONLINE_PACK_PIPELINE_SLOTS=32`,
  `DASER_ONLINE_PACK_RAW_TAIL_FRACTION=0.25`, normal streams, no Green Context,
  automatic load depth 3 and store depth 1. The pipeline-slot and raw-tail
  experiment knobs were later removed; the merged store path packs every slot
  in each staged batch, and compressed-online now shares raw's staging
  load/store partition, so these numbers are historical rather than a default
  configuration.
- TileLang compile and representative launch happen before measured requests.
- TTFT is client-measured mean time to first token. Time is the complete timed
  request phase, excluding startup, explicit prefix warmup, and shutdown.

## Cold/Warm Eviction

This comparison uses the maintained `benchmarks/run_bench.py` vLLM-bench flow:
8192 input tokens, no shared prefix, two complete scans of the same 200 prompts,
L1 180 GiB, L2 236.25 GiB, async scheduling disabled for both backends.
There is no backend-specific priming replay. Stores drain between phases.

| Backend | Phase | Mean TTFT (ms) | Time (s) | Prompt tokens/s | Token hit rate |
|---|---|---:|---:|---:|---:|
| LMCache | Cold | 2871.254 | 73.072 | 22421.8 | 0% |
| Online pack | Cold | 2750.591 | 70.015 | 23400.5 | 0% |
| LMCache | Warm | 374.763 | 9.479 | 172842.4 | 100% |
| Online pack | Warm | 188.314 | 4.755 | 344585.6 | 100% |

Every phase completed 200/200 requests with zero HTTP errors. Packed warm mean
TTFT is 49.75% lower than LMCache in this single run; cold TTFT is 4.20% lower.

### Capacity and Tier Activity

Packed payload occupies 187,951,063,040 bytes (175.043 GiB), compared with
241,591,910,400 raw-equivalent bytes (225 GiB): **22.203% less storage**.
It fits in the configured 180 GiB L1. All 12,800 packed warm slot loads hit L1,
with zero L2 reads. LMCache fetched 6,507 chunks from L2 during warm traffic,
equivalent to 114.381 GiB of raw KV.

Both systems have 100% external token hit rate, but different tier residency.
This is a capacity-assisted result, not evidence of a 49.75% codec-only speedup.
`--evict` enables L2 and fixed capacity budgets; it does not force L2 reads when
the compressed working set fits in L1. LMCache retains its 0.8 eviction
watermark, and its filesystem L2 adapter does not enforce a strict byte cap.

### Output Correctness Limitation

Cold outputs match between backends for 200/200 requests. Exact cold/warm
generated-output comparisons are 196/200 for LMCache and 198/200 for pack.
Pack's warm differences at request indexes 99 and 106 also occur in LMCache;
LMCache additionally differs at 87 and 136. Attribution is unresolved. These
measurements must not be described as complete end-to-end output equivalence.

Four focused CUDA integration tests passed, covering byte-exact mixed packed/raw
restore, escapes and raw overflow, production geometry and partial tails, and
io_uring/H2D restore. This verifies those codec cases, not the cause of the
generated-text differences above.

## PPT Slide 15 Addendum

The protocol follows slide 15 of
`daser-kvcache-ppt_20260908_experiments_polished.pptx`: L1 270 GiB,
L2 354.375 GiB, async scheduling enabled, and one exact untimed warmup request
before each 200-request measurement. Input and prefix lengths are not shortened.

Raw DaseR, LMCache, and vLLM references are the archived 2026-09-08 PPT data.
Pack is measured on 2026-09-09. These are historical single-run comparisons,
not same-session paired repetitions or confidence-interval evidence.

Positive reduction means pack is faster. Every point completed 200/200 requests,
with zero HTTP errors and 200/200 exact matches against archived baseline output.

| Input | Prefix | Raw TTFT ms | LMCache TTFT ms | Pack TTFT ms | Pack time s | Reduction vs raw |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1024 | 343.279 | 355.430 | 349.219 | 8.843 | -1.73% |
| 4096 | 2048 | 681.732 | 727.935 | 697.858 | 17.662 | -2.37% |
| 8192 | 4096 | 1505.368 | 1554.197 | 1507.473 | 38.303 | -0.14% |
| 12288 | 6144 | 2409.458 | 2518.213 | 2440.385 | 61.964 | -1.28% |
| 12288 | 2048 | 3700.810 | 3925.207 | 3782.651 | 96.204 | -2.21% |
| 12288 | 4096 | 3092.030 | 3226.760 | 3136.619 | 79.849 | -1.44% |
| 12288 | 8192 | 1736.342 | 1763.579 | 1717.671 | 43.634 | +1.08% |
| 12288 | 12288 | 280.863 | 343.481 | 330.909 | 8.370 | -17.82% |

Measured pack token hit rates, in table order, are 50.22%, 50.25%, 50.23%,
50.24%, 17.08%, 33.66%, 66.83%, and 100%. Actual filesystem allocation matches
the packed payload byte counters for every point. Storage savings range from
22.191% to 22.205%, including the retained raw-tail portion. All packed loads
hit L1; no point performed L2 reads.

Pack is 1.75-4.13% faster than historical LMCache across these eight points,
but slower than historical raw on seven of eight. The only raw improvement is
1.08%; this matrix does not demonstrate a 20% TTFT gain over raw.

The full-prefix endpoint has **zero timed store bytes** and 19,200 L1 slot
loads. Its 17.82% TTFT regression cannot be explained solely by store
compression competing with prefill. The load/decode path and scheduling still
need separate profiling, and historical run-to-run variability is not isolated
by this experiment. Do not attribute the regression to a specific kernel
without additional evidence.

## Reproduction and Evidence

Cold/warm runner options:

```bash
python benchmarks/run_bench.py \
  --backend lmcache,daser-prefix --model "$MODEL_PATH" \
  --store-dir "$BENCH_SCRATCH" --gpu-id 2 --max-num-seqs 8 \
  --load-generator vllm-bench --evict \
  --bench-num-prompts 200 --bench-input-len 8192 \
  --bench-output-len 1 --bench-max-concurrency 8 \
  --daser-storage-format compressed-online
```

Use the project venv on `PATH`, the read-only LMCache checkout on `PYTHONPATH`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
Set the retained pack environment above. Scratch must be on the approved data
disk, not `/tmp`; derived capacities must match those reported above.

Evidence identifiers: `cold_warm_evict_pptpack_20260909/run_20260909_020918`
and `ppt_pack_20260909`. JSON results, configurations, manifests, logs, metric
snapshots, and per-point store allocation/cleanup records are retained outside
the repository. The report generator emits Markdown, CSV, JSON, and a two-panel
PPT comparison chart.

All owned services stopped and generated stores/indexes/sockets were deleted;
no benchmark GPU allocation remains. Two failed cold/warm attempts are excluded:
one failed before traffic because the venv was absent from `PATH`, and one
returned HTTP 404 due to a model alias mismatch. The first two PPT points had
cleanup wait timeouts after complete measurements; those measurements were
reused without rerunning traffic after shutdown was verified.

## Verification

Final default suite: 478 passed, 111 deselected. Focused CUDA integration:
4 passed. Changed Python files pass ruff, formatting, compileall, and diff checks.
Mypy reports one pre-existing error in untouched `benchmarks/utils/prompts.py:189`
(`list[str]` versus `list[str | list[int]]`); checking that file alone reproduces
the error. No type suppression or third-party source edits were introduced.
