# Prefix Output-1 TTFT Optimization

**Date:** 2026-06-11
**Mode:** prefix cache reuse, L1-only, no eviction
**Workload:** LongBench `multifieldqa_en,qasper,hotpotqa,narrativeqa`
**Generation:** `max_tokens=1`, `temperature=0.0`, `top_p=1.0`, `seed=42`

This optimization targets DaseR's TTFT in prefix mode when the benchmark only
generates one token. The benchmark scripts were not changed.

## Benchmark Shape

Representative command shape:

```bash
bash benchmarks/run_bench.sh \
  --backend all \
  --cache-reuse-mode prefix \
  --dataset longbench \
  --model <user>/models/Qwen3-8B \
  --store-dir <user>/daser_bench/prefix-output1 \
  --longbench-dir <user>/longbench/data \
  --datasets multifieldqa_en,qasper,hotpotqa,narrativeqa \
  --max-samples 25 \
  --gpu-id 2 \
  --gpu-util 0.85 \
  --max-num-seqs 32 \
  --max-inflight 8 \
  --gen-max-tokens 1 \
  --max-context-tokens 40959
```

Prepared workload:

| Item | Value |
| --- | ---: |
| Requests | 97 |
| Prompt tokens | 1,164,605 |
| KV blocks | 72,743 |
| Max prompt blocks | 2,313 |
| DaseR storage tier | L1-only |
| Derived L1 size | 239 GiB |

## Root Cause

Before this change, prefix lookup and load metadata stayed at rolling-prefix
slot granularity. A warm pass over this workload produced 85,912 one-slot
`ReqLoadSpec` entries. Worker staging could still batch bytes by capacity, but
the scheduler, IPC payload, and server L1 lookup path still processed tens of
thousands of tiny logical spans.

Diagnostic DEBUG run before the fix:

| Metric | Value |
| --- | ---: |
| `meta LOAD` entries | 85,912 |
| Loaded blocks | 85,912 |
| Mean blocks per load spec | 1.0 |
| L1 hit accounting events | 85,912 |
| `transfer_load_ms` total | 270.2 ms |
| `transfer_sync_ms` total | 3,725.4 ms |
| Worker `start_load_kv` total | 4,706.2 ms |

The remaining large cost is the real data movement from server-owned pinned L1
memory into the worker CUDA staging buffer. However, the one-slot metadata shape
added avoidable Python, IPC serialization, and L1 range lookup overhead on top
of that copy.

## Changes

### Server-side prefix coalescing

`PrefixHashIndex.lookup()` now coalesces adjacent rolling-prefix slot hits into
one `RetrievalMatch` when the slots are contiguous in storage and prompt
position. The merged match keeps the last slot's rolling key so a warm request
that needs to store later missing slots can continue the rolling hash chain from
the correct prefix state.

### Scheduler-side load coalescing

`SchedulerConnectorMixin.build_connector_meta()` now converts ready load chunks
into `ReqLoadSpec` objects and merges adjacent specs for the same request when
all of the following are contiguous and compatible:

- DaseR slot range
- byte file offset
- target prompt token range
- vLLM block IDs
- RoPE position offset

The worker load path still consumes regular `ReqLoadSpec` objects. This keeps
the optimization local to metadata construction and does not change transfer
semantics.

## Validation

Focused tests cover both new coalescing layers:

- Prefix index coalesces adjacent rolling-prefix slots and preserves the last
  rolling key.
- Scheduler metadata coalesces adjacent load specs for one request.
- Existing prefix store tests verify warm hits still resume store from the
  first missing slot.

DEBUG diagnostics after both coalescing changes:

| Metric | Before | After |
| --- | ---: | ---: |
| `meta LOAD` entries | 85,912 | 450 |
| Loaded blocks | 85,912 | 85,912 |
| Mean blocks per load spec | 1.0 | 190.9 |
| L1 hit accounting events | 85,912 | 584 |
| `transfer_load_ms` total | 270.2 ms | 24.2 ms |
| Worker `start_load_kv` total | 4,706.2 ms | 4,384.3 ms |

The loaded byte volume is unchanged; the improvement comes from removing
slot-granular control-plane work.

## Final Result

Final comparison, same benchmark shape, normal logging:

| Backend | Warm mean TTFT | Warm p50 TTFT | Warm p90 TTFT | Warm p99 TTFT | Warm elapsed | External hit | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM baseline | 4,655.5 ms | 4,502.7 ms | 6,332.5 ms | 7,403.9 ms | 57.45 s | n/a | n/a |
| LMCache | 545.5 ms | 486.1 ms | 912.1 ms | 1,197.6 ms | 6.77 s | 96.86% | 96/97 |
| DaseR | 463.6 ms | 474.9 ms | 666.4 ms | 695.0 ms | 5.77 s | 99.94% | 97/97 |

Compared with the master baseline for DaseR on the same workload:

| Metric | Master | Optimized | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 570.4 ms | 463.6 ms | 23.0% faster |
| p99 TTFT | 811.3 ms | 695.0 ms | 116.3 ms lower |

Compared with LMCache in the final run:

| Metric | LMCache | DaseR | Difference |
| --- | ---: | ---: | ---: |
| Mean TTFT | 545.5 ms | 463.6 ms | DaseR 17.7% lower |
| p99 TTFT | 1,197.6 ms | 695.0 ms | DaseR 72.3% lower |

