# TraceLab coding-agent benchmark

This adapter converts TraceLab coding-agent sessions into the CSV schema used
by TraceLab's closed-loop runner. It preserves source round order, input and
output lengths, reusable-prefix lengths, and wall-clock tool gaps. DaseR does
not vendor the TraceLab dataset, DuckDB environment, or runner.

## Prepare a replay

Run the exporter with an environment that provides DuckDB:

```bash
PYTHONPATH="$PWD" /path/to/tracelab/python \
  -m benchmarks.tracelab.prepare_trace \
  --db /path/to/coding_trace.duckdb \
  --model /path/to/model \
  --tensor-parallel-size 2 \
  --max-sessions 12 \
  --pause-window-rounds 8 \
  --out /path/to/scratch/trace.csv
```

Use `--dense-window-rounds` with `--max-dense-gap-seconds` to select contiguous
long-prefix windows. Dense selection rejects overlapping source rounds instead
of turning them into artificial zero-time dependencies.

The exporter prints selected session/round counts and model-derived peak KV
capacity. Model, database, and output paths are local inputs and must not be
committed.

## Build a model-native token pool

Models with custom tokenizers must use token IDs produced by that model's own
tokenizer:

```bash
/path/to/python -m benchmarks.tracelab.prepare_token_pool \
  --model /path/to/model \
  --text-file /path/to/corpus.txt \
  --output /path/to/scratch/model_tokens.u32 \
  --limit 1000000 \
  --trust-remote-code
```

The output is a headerless little-endian sequence of unsigned 32-bit token
IDs. Configure the closed-loop runner to submit those IDs directly. For later
rounds, append the exact generated token IDs returned by vLLM; decoding and
re-tokenizing output can change the prefix.

Large inputs and outputs belong in project-approved scratch storage, never
`/tmp`.
