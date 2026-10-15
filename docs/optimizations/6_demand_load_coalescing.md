# Demand Load Coalescing

**Date:** 2026-10-15
**Implementation:** version 2, based on master `5184597`
**Mode:** ordinary io_uring demand loads, pinned L1 plus NVMe L2
**Status:** v2 H800 formal, small-L1, and probe-disabled comparisons complete;
Qwen RTX4090 ABBA comparison complete

Version 2 shares in-flight L2 promotions between concurrent loads. On the tested
H800, a shared 16 MiB range at concurrency 16 reduces request p50 from 24.161 ms
to 14.347 ms (40.6% lower), with storage reads falling from 256 MiB to 16 MiB per
wave. Half-range overlap at that size and concurrency reduces p50 from
24.700 ms to 19.747 ms (20.1% lower).

These gains do not apply to every workload. At 4 MiB/concurrency 4, disjoint and
partially overlapping loads regress in all five process pairs. Warm 4 KiB loads
at concurrency 4 also show a substantial p95 regression. The evidence supports
sharing duplicate cold reads under the measured conditions, not a general
latency improvement. A separate Qwen3-0.6B ABBA experiment below measures TTFT
on a shared RTX4090 with a deliberately high-sharing synthetic workload.

## Root Cause

Master already coalesces adjacent misses within a load and batches small
io_uring reads. Its promotion bookkeeping tracks buffer completion without
identifying the byte range being promoted. Concurrent loads can resolve the
same cold range, reserve separate pinned buffers, and issue duplicate L2 reads
before any caller publishes the data to L1.

The candidate associates each promotion with its physical range and lets
followers reuse the resulting L1 bytes. Every destination still receives its
requested bytes; the optimization removes duplicate storage reads, not the
copies required by distinct GPU destinations.

## Changes

### Share pending promotions by physical range

`TieredIOUringTransferLayer` records each promotion's range, cache mutation
epoch, and completion future. Under the existing metadata lock, a load checks
current L1 contents and pending owners before reserving pages. A range that
overlaps an active promotion or pending write waits for its owner, then resolves
L1 again and reads only the remaining gaps.

A completion future means the reservation ended; it does not guarantee that
all requested bytes became resident. Re-resolution handles eviction, failed
owners, and stores that supersede older reads. Logical destination offsets,
raw-byte accounting charges, and request leases remain attached to the correct
subranges. Demand loads and prefetches share this ownership mechanism.

### Submit bounded rounds and retain native owners

A batch submits the reservations it obtained before waiting for more pool
capacity. It processes completions and releases native ownership before its
next admission round. This avoids a capacity deadlock where batches each hold
queued reservations and wait for pages held by another batch.

Pinned buffers remain owned until the native executor and destination CUDA
copies finish. Per-read callbacks can arrive before their batch executor
returns, so callback completion alone cannot release the buffer. Error and
repeated-cancellation paths drain owners; cancelling a follower does not cancel
a shared read. Mutation epochs prevent stale publication over newer stores.

### Avoid unnecessary work in version 2

A fully cold extent already has a bounded, target-aware coalesced plan. If its
physical range has no L1 overlap, v2 reuses that plan while still checking
pending promotions and writes under the lock. Only spans with resident overlap
need subrange resolution and reconstruction.

The drain helper also checks whether all supplied futures are already complete.
It consumes every outcome directly in that case, avoiding an extra `gather`
event-loop turn. Pending owners still use shielded draining, and completed
sibling failures are consumed before the first exception is propagated. This
shared helper is also used by warm loads, so the warm path is not unchanged
between master and v2.

The change preserves public transfer APIs, connector IPC, and startup backend
selection. See [components.md](../design/components.md) for transfer ownership.

## Version 2 Benchmark Shape

### Source identity and environment

The clean baseline is master
`5184597f8ec07685afb465e64ab45bef1c156603`; v2 is the same commit plus the measured
transfer patch. All ten process results match these identities. The candidate
hashes also match the completed Qwen correctness run.

| Identity | SHA256 |
| --- | --- |
| Baseline `layer.py` | `8735ee309c2484ba43b41d3eb49f30ca2975a359609af13e28453f2e8a98b475` |
| Version 2 `layer.py` | `2d389c45099ada0ccdfcb61cbb9d5cbcec11d416491fc97440a3b085b16581e3` |
| Version 2 `git diff HEAD -- daser/transfer/` | `145218b0b529df85f0dc0dd7e8b5d1bdd3df3765ab99344131aee3bd8d614381` |

| Item | Value |
| --- | --- |
| GPU | NVIDIA H800; same UUID and PCI address in all ten processes |
| Storage | NVMe file, `O_DIRECT`, native io_uring reads |
| Destination | Separate real CuPy CUDA array and stream per request |
| OS / driver | Linux 6.8 / NVIDIA 575.57.08 |
| Python | 3.10.20, isolated uv venv (not conda-managed) |
| vLLM / PyTorch | 0.30.0+cu129 / 2.13.0+cu129 |
| CuPy / KvikIO / Transformers | 14.2.0 / 26.2.0 / 5.19.0 |
| IO workers / seed | 8 / 42 |
| Microbenchmark L1 replacement | BIP explicitly disabled |
| Production server default | BIP enabled, unchanged |

This is the H800 used for the v2 comparison, not the H800 PCIe used for the v1
exploration. All performance comparisons below are master versus v2 on the
same H800. The different devices prevent a direct numerical v1-versus-v2 claim.

The benchmark imports the selected checkout's production transfer layer. It
uses real IO and H2D copies without launching vLLM; the vLLM version identifies
the evaluation environment, not a timed component. Disabling BIP isolates
demand-read sharing and leaves replacement-policy performance to model-level
validation.

### Workloads and process protocol

[bench_demand_load.py](../../benchmarks/bench_demand_load.py) tests sizes 4 KiB,
256 KiB, 4 MiB, and 16 MiB at concurrency 1, 4, and 16.

| Case | Requested ranges | Initial L1 state |
| --- | --- | --- |
| `same` | All requests use the same range | Empty |
| `overlap` | Start offsets advance by half a request | Empty |
| `disjoint` | Adjacent, non-overlapping ranges | Empty |
| `mixed` | All requests use the same range | First half resident |
| `warm` | All requests use the same range | Entire range resident |

L1 capacity is request size times concurrency. There are 54 valid cells. The
six 4 KiB overlap/mixed cells are skipped because their half-range violates
4096-byte direct-IO alignment; the requested size is not silently increased.

Five independent process pairs run sequentially in AB / BA / AB / BA / AB
order. Every process runs two untimed warmup waves and ten measured waves per
cell, giving 50 measured waves per arm per cell. All ten processes exit
successfully; no slow cells are discarded.

Each wave creates a fresh transfer layer over a fully initialized store. Store
initialization, L1 priming, CUDA allocation and initialization, and byte-exact
verification are outside timing. Requests start from a shared asyncio gate.
Request latency includes grouped load and CUDA stream completion; wave time
covers the complete concurrent group. There is no global page-cache flush.
Stores and scratch directories are removed after each subprocess exits, while
results and logs are retained.

Tables pool request latencies across all five processes per arm. p50/p95 are
request percentiles, not percentiles of process medians. Throughput is
**logical delivered GiB/s**: `requested_bytes / 2^30 / wave_elapsed_seconds`,
aggregated across waves. It includes the total logical H2D volume delivered to
each request's separate destination; it is not SSD bandwidth. Paired per-process p50 changes
are also retained. These samples do not supply a confidence interval or a
formal significance test.

### Evidence audit

All **5,400 measured waves and 37,800 requests** record byte-exact CUDA output.
An independent read-only audit verifies every process's source identity,
device, configuration, completion status, request count, and sample result.
Recomputing all 54 summary cells from the raw samples reproduces their p50,
p95, wave median, and throughput.

Completed native read bytes equal `/proc/self/io` storage-read bytes in every
wave. Both counters also match the expected physical-byte formulas below.
Native wrappers call the real public read methods and count their completed
extents and bytes; they do not replace IO. An API extent is not guaranteed to
correspond to one kernel SQE. The completed v2 probe-disabled sensitivity
check is reported separately below.

| Case | Candidate / baseline physical bytes at C=4 | At C=16 |
| --- | ---: | ---: |
| Same cold range | 1 / 4 | 1 / 16 |
| Mixed, shared cold half | 1 / 4 | 1 / 16 |
| Half-range overlap | 2.5 / 4 | 8.5 / 16 |
| Disjoint | 1 / 1 | 1 / 1 |
| Warm | Zero bytes in both arms | Zero bytes in both arms |

For example, 16 MiB requests at concurrency 16 read 256 → 16 MiB in `same`,
256 → 136 MiB in `overlap`, and 128 → 8 MiB in `mixed`. All warm waves read zero
storage bytes. All disjoint waves retain the original physical-byte volume.

The filesystem's block-device statistics path is unavailable: every
`nvme_read_bytes` field is null. Completed native reads and per-process kernel
accounting establish storage traffic here; there is no whole-device NVMe
counter measurement. A whole-device counter would also include other users' IO.

## Version 2 Results

In all tables, **negative p50 change means lower latency**; positive means a
regression. Positive throughput change means higher logical throughput.
`C` denotes concurrency. The complete 54-cell table is included below.

### Shared cold reads

All eight concurrent same-range cells and all six concurrent mixed-tier cells
have lower pooled p50. At concurrency 16, their paired p50 reductions are
positive in all five process pairs. Partial overlap improves five of its six
concurrent cells; the 4 MiB/concurrency 4 exception is discussed separately.

| Case | Size | C | p50, master → v2 (ms) | p50 change | p95, master → v2 (ms) | Throughput change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| same | 256 KiB | 16 | 4.793 → 1.970 | -58.9% | 12.500 → 2.652 | +135.7% |
| same | 16 MiB | 16 | 24.161 → 14.347 | -40.6% | 41.711 → 22.350 | +102.5% |
| overlap | 16 MiB | 16 | 24.700 → 19.747 | -20.1% | 39.870 → 25.410 | +53.4% |

The 16 MiB/concurrency 16 same-range p50 reduction is 40.6% when pooling all
requests; its five paired reductions are 39.9%, 38.6%, 44.8%, 39.2%, and 39.9%.
For overlap at the same size, the pooled reduction is 20.1%, with paired
reductions of 20.1%, 19.6%, 20.1%, 21.6%, and 18.0%.

### Regressions and control cases

The following are observed limitations of this candidate. In particular, the
two 4 MiB/concurrency 4 regressions occur in every pair and cannot be dismissed
as noise:

- **Disjoint 4 MiB/concurrency 4:** p50 rises 4.401 → 4.814 ms (+9.4%), p95
  rises 7.2%, and throughput falls 5.2%. The five paired p50 increases are
  1.1%, 42.8%, 7.0%, 0.6%, and 6.8%; the median paired increase is 6.8%.
- **Overlap 4 MiB/concurrency 4:** p50 rises 3.966 → 4.333 ms (+9.3%), p95
  rises 4.8%, and throughput falls 5.4%, despite 37.5% fewer physical bytes.
  Paired p50 increases are 0.5%, 19.4%, 14.5%, 2.4%, and 1.2%; the median
  paired increase is 2.4%. Physical-byte reduction alone does not determine
  latency; this measurement does not isolate the cause.
- **Warm 4 KiB/concurrency 4:** p95 rises 0.532 → 0.970 ms (+82.4%), while
  p50 is nearly unchanged and throughput falls 13.3%. This tail regression
  remains part of the result even though no L2 reads occur. Its cause has not
  been established by these measurements.

Other material regressions remain in the full-cell appendix: disjoint
single-request p50 rises 10.6% at 4 KiB and 11.3% at 16 MiB; disjoint
16 MiB/concurrency 4 p95 rises 11.7%. Same-range 256 KiB/concurrency 1 p95 rises
1.242 → 1.622 ms (+30.6%), and warm 16 MiB/concurrency 4 p95 rises
1.649 → 2.188 ms (+32.7%).

Disjoint p50 changes span 3.1% lower to 11.3% higher. Warm p50 changes span
5.7% lower to 2.4% higher, but that range hides the larger tail changes above.
Single-request cells cannot share a duplicate read with another request; their
mixed results do not establish a broad improvement. The all-cell appendix
retains smaller regressions and favorable tail outliers rather than selecting
only the largest gains.

## Validation

- Full non-integration/non-slow suite: **573 passed, 132 deselected** in
  36.97 seconds using the new Python 3.10 environment (`full-tests-v2.log`).
- All v2 transfer comparisons: **7,120 waves and 51,040 real GPU byte-exact
  request checks** pass: 37,800 formal, 4,000 small-L1, and 9,240 probe-disabled.
  The capacity-constrained runs also complete without deadlock.
- Qwen3-0.6B ABBA comparison: **1,344/1,344 requests** pass, including matching
  output token IDs and text across all eight arms. An earlier candidate-only
  correctness run also passed 168/168 requests.

Focused tests cover range sharing, partial overlap and destination offsets,
prefetch sharing, cancellation, owner failure, tiny-L1 progress, duplicate
grouped spans, and newer overlapping stores. The added v2 case verifies that a
fully cold coalesced extent preserves scattered targets and raw-byte accounting
charges when its existing plan is reused.

## Qwen3-0.6B End-to-End Comparison

### Workload and protocol

The completed model comparison uses two sequential ABBA cycles: **eight arms,
192 waves, and 1,344 requests**, with four baseline and four v2 processes. All
arms use the source identities above, Python 3.10.20, vLLM 0.30.0+cu129, and
PyTorch 2.13.0+cu129 on the same shared RTX4090.

The model is `Qwen/Qwen3-0.6B`, pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Its download manifest retains file
hashes and official URLs. Every arm uses the same request manifest, SHA256
`9fdfcf88dd22f4fffa2dfe9ebd0f415c8f05d3861a631060372a4d86f93e7efb`.

Each of eight synthetic families has one 2,176-token prompt: a 2,048-token
prefix plus a 128-token suffix. A concurrent wave repeats that **entire same
prompt** four or sixteen times. The suffix is fixed within the family, not
varied for each request. The eight different families rotate through a
1.859375 GiB KV working set that exceeds the 512 MiB L1. This intentionally
high-sharing workload is not a real RAG question-answer dataset.

Each arm first sends eight cold seed requests, then one eight-family warm
cycle at concurrency 4 and one at concurrency 16. Here "warm" means previously
stored data; every measured warm wave reaches SSD after capacity eviction.
Other settings are raw storage, 128-token blocks, 8 GiB L2, BIP enabled,
prefetch disabled, vLLM prefix caching disabled, `temperature=0`, `top_p=1`,
`seed=42`, and **one generated token**. The single-token output targets TTFT;
it does not measure long-decode throughput.

### TTFT results and pair variation

The E2E summary uses **nearest-rank percentiles**: sort the request values and
select rank `ceil(q * n)` for quantile `q`. This is distinct from the interpolated
NumPy percentiles used in the transfer microbenchmark. All requests are retained;
no arms or waves are filtered.

| Phase / concurrency | Requests per variant, pooled | p50, master → v2 (ms) | p50 change | p95, master → v2 (ms) | p95 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold seed / 1 | 32 | 36.606 → 37.727 | +3.06% | 80.047 → 79.316 | -0.91% |
| Stored-data cycle / 4 | 128 | 143.816 → 93.865 | -34.73% | 195.013 → 108.655 | -44.28% |
| Stored-data cycle / 16 | 512 | 388.171 → 167.210 | -56.92% | 648.736 → 236.299 | -63.58% |

Pooled p50 is not the median of arm p50s. At concurrency 4, the median of the
four arm p50s is **120.415 → 93.535 ms**; pooling the batched request
distributions gives **143.816 → 93.865 ms**. Reporting only the latter would
hide the variation across runs. The four adjacent A/B pairs give these p50
reductions, with positive values meaning lower candidate latency:

| Pair, baseline / candidate arm | Concurrency 4 p50 reduction | Concurrency 16 p50 reduction |
| --- | ---: | ---: |
| 00 / 01 | 23.99% | 55.31% |
| 03 / 02 | 17.19% | 54.04% |
| 04 / 05 | 33.80% | 56.92% |
| 07 / 06 | 26.91% | 55.23% |

Thus the observed paired p50 reductions span **17.19%–33.80%** at concurrency 4
and **54.04%–56.92%** at concurrency 16. The cold-seed p50 instead rises 3.06%;
those requests perform no external reads, so no cold-prefill benefit is claimed.

### Storage activity and output correctness

The following values hold for every eight-family cycle in every arm, not only
for the pooled average. `l2_reads` is the server's transfer read counter; it is
not a count of kernel SQEs. Storage bytes come from the DaseR server process's
kernel `read_bytes` counter.

| Concurrency | Server L2 reads per cycle, master → v2 | Storage GiB per cycle, master → v2 | Logical external load GiB per cycle, both arms |
| --- | ---: | ---: | ---: |
| 4 | 32 → 8 | 7.4375 → 1.859375 | 7.4375 |
| 16 | 128 → 8 | 29.75 → 1.859375 | 29.75 |

Master reads the shared range four or sixteen times per wave; v2 reads it
once. Logical bytes delivered through the external load/H2D path remain the
same. The storage reductions are 75% and 93.75%, respectively. All 128 warm
waves across the eight arms touch SSD, and no prefetch operations occur.

All 1,344 requests return one output token without request or validation
errors. Every repeated request's output token IDs and text match its family's
cold seed, and the seed outputs match across all eight arms. An independent
read-only audit reproduces pooled nearest-rank quantiles and all paired changes
from raw requests and checks source hashes, manifest identity, and per-wave
storage/load bytes.

### Shared-device limitation and cleanup

A pre-existing process remains on the RTX4090 throughout the experiment,
holding **6,538 MiB**. Its recorded `pmon` SM and memory-activity values are
`-`, which is unavailable telemetry, not evidence of inactivity. The run does
not establish that external interference was absent and is not an isolated-GPU
performance measurement. ABBA order, complete request retention, and the paired
results expose some variation but do not remove this limitation.

The cleanup audit confirms all **32 owned processes** are gone, all **16 service
ports** are closed, and every arm's store plus the suite runtime directory has
been deleted. The pre-existing process was left untouched.

These measurements cover this Qwen3-0.6B, repeated-prompt, single-token workload
under the recorded shared-device conditions. They do not establish performance
for distinct per-request suffixes, real RAG workloads, long decoding, or Llama
3.1 8B.

## Version 2 Small-L1 Comparison

This follow-up uses the same H800, environment, and v2 source hashes as the
formal transfer matrix. L1 holds only **two request-sized buffers** while
concurrency is 4 or 16. Five sequential process pairs run ten measured waves
per arm per cell: **400 waves and 4,000 requests** over four disjoint-load cells.
All ten processes exit successfully; all requests are byte-exact and complete
without deadlock. Every wave's native and process read bytes equal its disjoint
requested volume, with no byte reduction between arms. Whole-device counters
remain unavailable.

| Size | C | p50, master → v2 (ms) | p50 change | p95, master → v2 (ms) | Logical throughput change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 KiB | 4 | 1.564 → 1.644 | +5.1% | 2.909 → 3.257 | -2.6% |
| 256 KiB | 16 | 5.216 → 6.075 | +16.5% | 9.752 → 10.425 | -8.5% |
| 16 MiB | 4 | 11.862 → 11.761 | -0.8% | 19.124 → 19.397 | -1.1% |
| 16 MiB | 16 | 37.318 → 37.385 | +0.2% | 65.898 → 65.768 | +0.0% |

The 256 KiB/concurrency 16 p50 regresses in **all five pairs**, with increases
of 20.1%, 13.4%, 24.6%, 17.6%, and 9.0%. Small-block loads under capacity pressure
therefore retain a repeatable cost. Successful progress and correctness do not
make this a performance win. The 16 MiB pooled p50 changes are small, with
mixed paired results; no broad constrained-L1 improvement is claimed.

## Version 2 Native-Counter Sensitivity

The completed `--no-native-counts` follow-up uses the same H800, environment,
and source hashes. Two sequential process pairs test 4 KiB, 4 MiB, and 16 MiB
same/overlap/disjoint/warm loads at concurrency 1, 4, and 16. The 33 valid cells
have 20 measured waves per arm: **1,320 waves and 9,240 requests**, all byte-exact.
All four processes exit successfully. The three 4 KiB half-range overlap cells
are skipped for the same alignment constraint.

Native wrapper counters are deliberately zero and are not read-volume evidence
in this run. Every wave's independent process `read_bytes` matches its expected
physical bytes, including the same-range and overlap reductions and zero warm
reads. Whole-device counters remain unavailable. Raw samples reproduce all
pooled p50/p95 values.

The large shared-read benefit remains: at 16 MiB/concurrency 16, `same` p50 is
**24.740 → 14.636 ms (-40.8%)**, and `overlap` is **24.900 → 19.104 ms (-23.3%)**.
Important controls and regressions remain:

- Overlap 4 MiB/concurrency 4 p50 rises **3.946 → 4.245 ms (+7.6%)**, with
  paired increases of 9.8% and 6.3%. Its p95 falls 9.8%, so the metrics do not
  all move in the same direction.
- Warm 4 KiB/concurrency 4 p95 rises **0.500 → 0.882 ms (+76.5%)**, p50 rises
  4.4%, and logical throughput falls 18.1%.
- Warm 16 MiB/concurrency 16 p50 rises **2.762 → 3.160 ms (+14.4%)**; both
  pairs regress, by 8.1% and 20.8%.
- Disjoint 4 KiB/concurrency 16 p50 rises **4.163 → 4.425 ms (+6.3%)**, and
  p95 rises 16.0%. Disjoint 4 MiB/concurrency 1 also has an 18.1% p95 increase.

Disjoint 4 MiB/concurrency 4 reverses direction in this follow-up:
**5.083 → 4.599 ms (-9.5%)**, with paired p50 reductions of 17.7% and 0.2%.
The formal five-pair run's 9.4% increase was present in all five pairs, but it
does not repeat under every measurement setting. This difference neither
dismisses the formal regression as noise nor establishes that the probe has no
effect. Two pairs and a separately timed follow-up cannot isolate a precise
instrumentation cost. The evidence supports the shared-read mechanism while
retaining the measured sensitivity of control workloads.

## Version 1 Exploratory History

Version 1 was tested against master on a different **H800 PCIe**. It established
duplicate-read removal and exposed control regressions: disjoint and warm
controls included regressions, while a one-pair small-L1 check showed a 30.5%
p50 regression at 256 KiB/concurrency 16. That check had only five measured
waves per arm, so it was not a stable performance estimate. A two-pair v1
no-probe check preserved the shared-read benefit but also retained regressions.

Those findings motivated the v2 cold-plan and completed-future changes. This
history does not show how much v2 improves over v1, because the formal devices
differ. The retained v1 groups are `h800-pcie-formal`, `h800-no-probe`, and
`h800-small-l1`; the archived implementation is `variant-v1/layer.py` with
`variant-v1/transfer.patch`.

| Version 1 identity | SHA256 |
| --- | --- |
| Layer | `1d0894f237979d76a25ba97af31d74b9bb34c0966f57f3d0c274f13b5fada344` |
| Transfer patch | `7d134d0708c80d1ff146f52dac6b43182463f39c7bcf493079bd4b21894e44c2` |

The v1 full suite had 572 passes and 132 deselections; its targeted transfer
suite had 101 passes. Four existing GPU integration tests also passed in the
previous Python 3.10 / vLLM 0.19.1 environment. These historical checks are not
presented as v2 or vLLM 0.30 model-comparison results.

## Reproduction and Retained Evidence

Run the same benchmark script against separate baseline and candidate
checkouts. The example executes one arm; use unique result files and repeat in
five sequential AB / BA / AB / BA / AB pairs. Select an idle GPU by UUID and
place scratch outside the repository and `/tmp`. All paths below are
placeholders.

```bash
CUDA_VISIBLE_DEVICES="<gpu-uuid>" \
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 DASER_LOG_LEVEL=ERROR \
TMPDIR="<scratch-root>" \
"<python-environment>/bin/python" "<candidate-checkout>/benchmarks/bench_demand_load.py" \
  --repo "<arm-checkout>" \
  --variant "<pair>-<baseline-or-candidate>" \
  --expect-commit 5184597 \
  --scratch-root "<scratch-root>" \
  --output "<results-root>/<pair>-<arm>.json" \
  --device 0 \
  --sizes 4KiB,256KiB,4MiB,16MiB \
  --concurrency 1,4,16 \
  --cases same,overlap,disjoint,mixed,warm \
  --warmup 2 --repeat 10 --io-workers 8 --timeout 30
```

`h800-v2-formal` contains ten per-process JSON files/logs, `runs.json`, and
`summary.json`/`summary.csv`. Each result records source hashes and the selected
GPU. `audit-v2-formal.json` records the independent sample and summary audit;
`variant-v2/manifest.json` identifies the archived layer, patch, benchmark, and
regression test. The runner and summarizer are also retained alongside results.
The sensitivity results are retained under `h800-v2-small-l1` and
`h800-v2-no-probe`. The model A/B evidence is grouped under `demand-e2e-qwen-v2-abba-20261015`: eight arm results,
`suite.json`, `summary.json`, request manifest, script snapshot, GPU/process
monitoring, and cleanup audit. The earlier candidate-only check remains under
`demand-e2e-qwen-v2-correctness-20261015`. Artifact names identify the evidence
without embedding machine-specific private paths.

`cleanup-final.json` records no remaining test stores or task scratch
directories. The baseline worktree and unused Python 3.12 venv were removed;
the Python 3.10 uv venv and models remain available. Environment manifests are
retained under `environment/py310` and `environment/unused-py312`.

## Complete Version 2 Cell Results

All 54 valid cells follow. Values are pooled across five processes per arm;
negative p50 change means lower candidate latency. Six unaligned 4 KiB
half-range cells are excluded by the documented alignment rule, not by outcome.

| Case | Size | C | p50, master → v2 (ms) | p50 change | p95, master → v2 (ms) | Throughput change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| disjoint | 4 KiB | 1 | 0.839 → 0.928 | +10.6% | 1.425 → 1.402 | -4.3% |
| disjoint | 4 KiB | 4 | 1.447 → 1.435 | -0.8% | 2.323 → 2.428 | -1.3% |
| disjoint | 4 KiB | 16 | 4.588 → 4.445 | -3.1% | 7.183 → 6.596 | +1.2% |
| disjoint | 256 KiB | 1 | 1.012 → 0.999 | -1.4% | 1.597 → 1.617 | -0.3% |
| disjoint | 256 KiB | 4 | 1.825 → 1.872 | +2.6% | 2.736 → 2.825 | -0.1% |
| disjoint | 256 KiB | 16 | 4.595 → 4.588 | -0.2% | 6.787 → 6.796 | -0.4% |
| disjoint | 4 MiB | 1 | 3.095 → 3.201 | +3.4% | 4.254 → 4.377 | -3.2% |
| disjoint | 4 MiB | 4 | 4.401 → 4.814 | +9.4% | 5.931 → 6.360 | -5.2% |
| disjoint | 4 MiB | 16 | 7.229 → 7.231 | +0.0% | 10.835 → 11.080 | -2.8% |
| disjoint | 16 MiB | 1 | 6.294 → 7.003 | +11.3% | 7.331 → 7.301 | -0.7% |
| disjoint | 16 MiB | 4 | 12.292 → 12.538 | +2.0% | 15.178 → 16.957 | -8.2% |
| disjoint | 16 MiB | 16 | 22.604 → 23.235 | +2.8% | 39.410 → 39.710 | +0.5% |
| mixed | 256 KiB | 1 | 0.779 → 0.747 | -4.2% | 1.488 → 1.253 | +8.8% |
| mixed | 256 KiB | 4 | 1.841 → 1.074 | -41.6% | 2.836 → 2.458 | +54.1% |
| mixed | 256 KiB | 16 | 5.198 → 2.565 | -50.7% | 8.289 → 3.411 | +118.1% |
| mixed | 4 MiB | 1 | 2.681 → 2.657 | -0.9% | 3.136 → 3.188 | +1.9% |
| mixed | 4 MiB | 4 | 3.218 → 3.077 | -4.4% | 10.784 → 3.537 | +78.9% |
| mixed | 4 MiB | 16 | 6.103 → 4.506 | -26.2% | 11.398 → 6.335 | +101.7% |
| mixed | 16 MiB | 1 | 4.546 → 4.571 | +0.5% | 5.009 → 4.914 | -1.0% |
| mixed | 16 MiB | 4 | 6.089 → 5.554 | -8.8% | 10.115 → 7.292 | +37.6% |
| mixed | 16 MiB | 16 | 12.848 → 9.489 | -26.1% | 24.156 → 16.704 | +79.6% |
| overlap | 256 KiB | 1 | 1.006 → 0.947 | -5.9% | 1.378 → 1.222 | +8.7% |
| overlap | 256 KiB | 4 | 1.890 → 1.533 | -18.9% | 2.836 → 1.976 | +18.6% |
| overlap | 256 KiB | 16 | 4.518 → 3.833 | -15.2% | 6.857 → 5.961 | +20.7% |
| overlap | 4 MiB | 1 | 3.089 → 3.114 | +0.8% | 4.094 → 4.068 | -0.6% |
| overlap | 4 MiB | 4 | 3.966 → 4.333 | +9.3% | 5.562 → 5.827 | -5.4% |
| overlap | 4 MiB | 16 | 7.095 → 6.681 | -5.8% | 10.429 → 9.256 | +19.4% |
| overlap | 16 MiB | 1 | 6.070 → 6.356 | +4.7% | 7.189 → 7.678 | -5.2% |
| overlap | 16 MiB | 4 | 12.364 → 10.410 | -15.8% | 15.371 → 13.041 | +12.3% |
| overlap | 16 MiB | 16 | 24.700 → 19.747 | -20.1% | 39.870 → 25.410 | +53.4% |
| same | 4 KiB | 1 | 0.822 → 0.780 | -5.1% | 1.422 → 1.021 | +7.6% |
| same | 4 KiB | 4 | 1.578 → 0.801 | -49.2% | 2.817 → 1.596 | +71.9% |
| same | 4 KiB | 16 | 4.371 → 2.028 | -53.6% | 6.550 → 2.449 | +143.5% |
| same | 256 KiB | 1 | 0.983 → 0.951 | -3.3% | 1.242 → 1.622 | -3.4% |
| same | 256 KiB | 4 | 1.993 → 0.909 | -54.4% | 3.412 → 1.430 | +132.2% |
| same | 256 KiB | 16 | 4.793 → 1.970 | -58.9% | 12.500 → 2.652 | +135.7% |
| same | 4 MiB | 1 | 3.087 → 3.099 | +0.4% | 4.168 → 3.671 | +2.0% |
| same | 4 MiB | 4 | 4.024 → 3.667 | -8.9% | 7.421 → 4.448 | +57.2% |
| same | 4 MiB | 16 | 8.647 → 6.178 | -28.6% | 18.352 → 9.489 | +94.7% |
| same | 16 MiB | 1 | 6.121 → 6.112 | -0.1% | 28.587 → 7.560 | +39.8% |
| same | 16 MiB | 4 | 9.378 → 8.063 | -14.0% | 14.199 → 10.063 | +46.3% |
| same | 16 MiB | 16 | 24.161 → 14.347 | -40.6% | 41.711 → 22.350 | +102.5% |
| warm | 4 KiB | 1 | 0.153 → 0.156 | +1.6% | 0.225 → 0.244 | -1.2% |
| warm | 4 KiB | 4 | 0.262 → 0.260 | -0.9% | 0.532 → 0.970 | -13.3% |
| warm | 4 KiB | 16 | 0.848 → 0.833 | -1.7% | 1.471 → 1.367 | +1.5% |
| warm | 256 KiB | 1 | 0.158 → 0.161 | +2.4% | 0.283 → 0.292 | -5.0% |
| warm | 256 KiB | 4 | 0.256 → 0.261 | +1.8% | 0.939 → 0.543 | +22.9% |
| warm | 256 KiB | 16 | 0.808 → 0.816 | +1.0% | 1.289 → 1.289 | -4.5% |
| warm | 4 MiB | 1 | 1.245 → 1.246 | +0.0% | 1.839 → 1.752 | +1.8% |
| warm | 4 MiB | 4 | 0.394 → 0.371 | -5.7% | 1.299 → 1.274 | -1.0% |
| warm | 4 MiB | 16 | 0.960 → 0.960 | -0.0% | 1.609 → 1.567 | -0.3% |
| warm | 16 MiB | 1 | 1.242 → 1.245 | +0.2% | 1.619 → 1.466 | +1.6% |
| warm | 16 MiB | 4 | 1.357 → 1.364 | +0.5% | 1.649 → 2.188 | -5.0% |
| warm | 16 MiB | 16 | 3.001 → 2.989 | -0.4% | 13.724 → 5.004 | +14.3% |

These results apply to the recorded v2 patch, one H800, the specified IO and
replacement settings, and these concurrent range patterns. The repeatable
4 MiB/concurrency 4 regressions, warm-tail and small-L1 regressions, and shared
RTX4090 conditions limit the conclusion to the measured shared-read workloads.
Probe-disabled results preserve large shared-read gains while showing that
some control results change with the measurement setting.
