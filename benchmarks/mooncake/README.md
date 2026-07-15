# Mooncake Trace Benchmark

This benchmark replays the public Mooncake FAST'25 Tool and Agent production
trace against an already running OpenAI-compatible vLLM endpoint. It preserves
the released millisecond arrival offsets, input/output lengths, and reusable
512-token prefix-block relationships.

Source:

- <https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces>
- `toolagent_trace.jsonl`: 23,608 requests sampled from one hour of Kimi Tool
  and Agent traffic.

The trace does not include raw prompts. The runner maps each released hash ID
to a deterministic synthetic 512-token block and sends integer token IDs
directly to `/v1/completions`. Synthetic content has no semantic meaning.

## Prepare

Download the trace to an approved data scratch directory outside the repo:

```bash
curl -L \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o <scratch>/toolagent_trace.jsonl
```

Start Qwen3-8B with prefix caching and prompt-token usage details enabled. The
local model configuration used by this project has a total context limit of
40,960 tokens:

```bash
vllm serve <model>/Qwen3-8B \
  --port 8001 \
  --max-model-len 40960 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details
```

## Smoke Replay

```bash
python -m benchmarks.mooncake.benchmark \
  --trace <scratch>/toolagent_trace.jsonl \
  --tokenizer <model>/Qwen3-8B \
  --server-url http://127.0.0.1:8001 \
  --served-model-name <served-model-name> \
  --output-dir <scratch>/smoke \
  --max-model-len 40960 \
  --max-requests 100 \
  --max-inflight 32 \
  --time-scale 10
```

## Production-Timing Replay

Omit `--max-requests` and use `--time-scale 1` to preserve the original
one-hour production timing:

```bash
python -m benchmarks.mooncake.benchmark \
  --trace <scratch>/toolagent_trace.jsonl \
  --tokenizer <model>/Qwen3-8B \
  --server-url http://127.0.0.1:8001 \
  --served-model-name <served-model-name> \
  --output-dir <scratch>/full \
  --max-model-len 40960 \
  --max-inflight 32 \
  --time-scale 1
```

Requests for which `input_length + output_length > 40960` are explicitly
written as `skipped_context_limit`; their counts and token mass remain in
`summary.json`. Use `--overflow error` to reject the complete run instead.
The runner never truncates or rescales token lengths.

Outputs:

- `requests.jsonl`: one completion-, failure-, or skip-record per selected
  trace row.
- `summary.json`: coverage, failures, wall time, achieved completion rate,
  server-visible TTFT/latency, trace-arrival-to-first-token/completion latency,
  client admission delay, and local/external prefix cache token hit rates from
  vLLM counter deltas. Use the trace-arrival metrics for online SLO analysis.

Large outputs must use project-approved data scratch, never `/tmp`.

## Verified Qwen3-8B Baseline

A complete production-timing replay was verified on one H800 80GB with
Qwen3-8B, vLLM prefix caching enabled, `max_model_len=40960`, and
`max_num_seqs=32`:

| Metric | Result |
| --- | ---: |
| Selected trace rows | 23,608 |
| Eligible and completed | 23,106 |
| Context-limit skips | 502 |
| Request failures | 0 |
| Wall time | 9,266.2 s |
| Completion throughput | 2.494 req/s |
| Local prefix query tokens | 455,236,820 |
| Local prefix hit tokens | 80,455,440 |
| Local prefix token hit rate | 17.67% |
| Server-visible TTFT mean / p50 / p95 / p99 | 675.8 / 213.0 / 2,738.1 / 5,990.6 ms |
| Arrival-to-first-token mean / p50 / p95 / p99 | 2,892 / 2,932 / 5,426 / 5,664 s |

All successful requests generated exactly the released `output_length`. The
original trace averages 6.67 req/s, well above the measured 2.494 req/s
completion rate on this single-GPU setup. The multi-minute arrival-to-first-
token values therefore describe sustained overload and must not be confused
with server-visible TTFT after client admission.
