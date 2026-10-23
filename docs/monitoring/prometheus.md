# Prometheus Monitoring

DaseR exposes Prometheus text metrics from the HTTP server at `/metrics`.
Metrics are intentionally low-cardinality: do not add `doc_id`, `chunk_key`,
`req_id`, local filesystem paths, prompts, or exception messages as labels.

## Docker Compose Metrics Stack

The repository includes a metrics-only Docker Compose stack under
`deploy/monitoring/`. It starts Prometheus and Grafana only; start vLLM and DaseR
as external processes before starting the stack.

From the compose directory:

```bash
cd deploy/monitoring
cp .env.example .env
docker compose --env-file .env up -d
```

Default endpoints:

| Service | URL |
|---------|-----|
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |

The default Grafana login is `admin` / `admin`. Change
`GRAFANA_ADMIN_PASSWORD` in `deploy/monitoring/.env` before starting the stack
on a shared machine.

Prometheus scrapes DaseR at `host.docker.internal:2026` and vLLM at
`host.docker.internal:8001`. If either runs on another host or port, edit
`deploy/monitoring/prometheus/prometheus.yml`.

Set `DASER_MONITORING_DATA_ROOT` in `deploy/monitoring/.env` to a persistent
directory on the data disk for Prometheus and Grafana state.

The Grafana dashboard is provisioned automatically from:

```text
deploy/monitoring/grafana/dashboards/daser-overview.json
```

## Scrape Targets

```yaml
scrape_configs:
  - job_name: daser
    static_configs:
      - targets: ["host.docker.internal:2026"]

  - job_name: vllm
    static_configs:
      - targets: ["host.docker.internal:8001"]
```

The Prometheus scrape interval is `1s`, and the provisioned Grafana data source
sets `timeInterval: 1s` so Grafana does not default to a coarser query step such
as 15 seconds. If Grafana was already running with a persisted data source,
restart/reprovision it after changing the data source file.

The vLLM TTFT/TPOT panels use `$__rate_interval`, do not span null gaps, and
filter latency series to periods with recent `vllm:request_success_total`
increments. When benchmark traffic stops, those latency panels should stop
drawing new values instead of extending the last observed latency as a flat
line.

During `benchmarks/run_bench.py --backend all`, DaseR is only running during the
`daser-chunk` and `daser-prefix` rows. Prometheus will show the `daser` target
down during `baseline` and `lmcache` rows because those rows do not start a
DaseR HTTP server.

For `--backend daser`, `--backend daser-chunk`, and `--backend daser-prefix`,
the benchmark runner prints:

```text
daser_metrics_startup: http://127.0.0.1:2026/metrics
daser_metrics_startup_status: ready
prometheus_scrape_wait_s: 2.0
prometheus_daser_target_startup: health=up scrape_url=http://host.docker.internal:2026/metrics
prometheus_daser_up_startup: value=1
daser_metrics_post-load: http://127.0.0.1:2026/metrics
daser_metrics_post-load_status: ready
prometheus_scrape_wait_s: 2.0
prometheus_daser_target_post-load: health=up scrape_url=http://host.docker.internal:2026/metrics
prometheus_daser_up_post-load: value=1
```

If Prometheus still shows `connection refused` for
`host.docker.internal:2026`, then the DaseR HTTP server is not listening on port
2026 from Prometheus' point of view. Check the runner output above, the
`daser.log` file under the backend run directory, and Prometheus
`Status -> Targets`. A healthy DaseR backend row should expose `daser_up` from
the URL printed by `daser_metrics`, and should print
`prometheus_daser_target_*: health=up` plus `prometheus_daser_up_*: value=1`.
If the local DaseR probe is ready but the Prometheus target is down, the issue
is the Docker host-gateway scrape path or the configured target, not the DaseR
metric names.

Grafana panels must be viewed over a time range that overlaps the active DaseR
backend row. After the row exits, Prometheus keeps `up{job="daser"}` as `0`, but
DaseR-owned `daser_*` metric series stop receiving fresh samples.

The dashboard service-status panels use Prometheus scrape health:
`max(up{job="daser"}) or vector(0)` and
`max(up{job="vllm"}) or vector(0)`. This makes a stopped or never-scraped
service render as `DOWN` instead of an empty panel. DaseR-owned metrics such as
`daser_up` and `daser_vllm_health_up` remain useful for diagnostics only while
the DaseR process is running.

The dashboard also has four observation groups: GPU scheduling and KV
occupancy, actual FragPack allocation, external prefix reuse, and tier
management. Prometheus samples every second; the dashboard refreshes every
10 seconds. These are monitoring views, not workload/configuration controls.

| Observation | Scrape target | Interpretation |
|-------------|---------------|----------------|
| Running/waiting requests, queue time, KV pool usage | vLLM | Scheduler state and KV pool occupancy, not total GPU memory usage |
| FragPack blocks and memory components | vLLM | Actual worker allocations forwarded through connector statistics |
| External prefix query/accepted tokens | DaseR | Reuse decisions, not proof of completed GPU loads |
| L1/L2 capacity, occupancy and management | DaseR | Tier allocation and eviction/migration activity |

The L1/L2 panels use `daser_tier_bytes_used` and
`daser_tier_bytes_capacity`, selected by `tier="l1"` or `tier="l2"`.
TPOT uses vLLM 0.30's `vllm:request_time_per_output_token_seconds`.

The DaseR dashboard assumes these Prometheus job labels:

```text
job="daser"   — DaseR server
job="vllm"    — vLLM inference server
```

## Metric Groups

### Service Health

- `daser_up`: DaseR HTTP process liveness.
- `daser_vllm_health_up`: vLLM health as observed by `/health`.
- `daser_info{mode,transfer}`: static server configuration (cache reuse mode and transfer backend).

### IPC

- `daser_ipc_requests_total{op,status}`: IPC request count by connector op.
- `daser_ipc_request_duration_seconds{op}`: IPC latency histogram.

### Cache Effectiveness

- `daser_cache_lookup_total{result}`: cache lookup hit/miss count.
- `daser_cache_requested_tokens_total`: tokens checked for reuse.
- `daser_cache_matched_tokens_total`: tokens matched by cache lookup.
- `daser_cache_committed_chunks_total`: chunks committed and visible.
- `daser_cache_late_evicted_commits_total`: commits ignored after eviction.
- `daser_cache_evicted_chunks_total{reason}`: explicit and capacity evictions.
- `daser_cache_prefix_reuse_tokens`: histogram of tokens reused per cache hit.

Use token hit ratio rather than only request hit ratio when judging benefit:

```promql
rate(daser_cache_matched_tokens_total[30s])
/
clamp_min(rate(daser_cache_requested_tokens_total[30s]), 1)
```

### Transfer & Storage

- `daser_transfer_operations_total{op,status}`: transfer op count.
- `daser_transfer_bytes_total{op}`: transfer bytes.
- `daser_transfer_duration_seconds{op}`: transfer latency histogram.
- `daser_transfer_chunk_size_bytes{op}`: transfer size distribution.
- `daser_l1_hits_total`: L1 memory cache hits.
- `daser_l1_misses_total`: L1 memory cache misses.
- `daser_tier_bytes_used{tier}`: current L1/L2 extent allocation.
- `daser_tier_bytes_capacity{tier}`: L1/L2 capacity.

The IPC server also logs transfer summaries with decimal GB/s throughput:

```text
[IPC] transfer_load summary backend=iouring status=ok bytes=... elapsed_ms=... throughput_gbps=...
```

Grafana should compute throughput from bytes:

```promql
rate(daser_transfer_bytes_total[30s]) / 1e9
```

### vLLM Inference (from vLLM scrape target)

- `vllm:time_to_first_token_seconds`: TTFT histogram.
- `vllm:request_time_per_output_token_seconds`: per-request average TPOT
  histogram. Requests with at most one output token contribute zero.
- `vllm:request_success_total`: successful request count.

### GPU Scheduling and KV Occupancy

- `vllm:num_requests_running`: running requests per engine.
- `vllm:num_requests_waiting`: waiting requests per engine.
- `vllm:num_requests_waiting_by_reason{reason}`: waiting requests by reason.
- `vllm:request_queue_time_seconds`: queue-time histogram, observed when a
  request completes; the dashboard shows p50/p95/p99, not current queue age.
- `vllm:kv_cache_usage_perc`: fraction of the engine's KV block pool occupied.

Waiting requests are not necessarily waiting for GPU memory. Read the waiting
reason together with KV occupancy and actual capacity before attributing delay.

### Actual FragPack Capacity

These **DaseR-defined metrics are exposed by vLLM's `/metrics` endpoint**.
The attention backend captures host-side tensor geometry after creating its
store. The worker forwards the snapshot through vLLM's public connector
statistics hooks; vLLM's API process owns the Prometheus collectors. There is
no additional telemetry RPC or CUDA synchronization in the attention path.

Metrics retain the standard vLLM model/engine labels and add `tp_rank`:

| Metric | Meaning per worker |
|--------|--------------------|
| `daser_fragpack_blocks` | Actual allocated block count, including reserved blocks |
| `daser_fragpack_block_tokens` | Token slots per block |
| `daser_fragpack_layers` | Attention layers in the store |
| `daser_fragpack_kv_heads` | Local KV heads per layer |
| `daser_fragpack_page_bytes` | Actual bytes per block-layer |
| `daser_fragpack_raw_page_bytes` | Raw BF16 bytes for the same block-layer geometry |
| `daser_fragpack_memory_bytes{component}` | Persistent tensor allocation by component |

Components are `arena`, `staging`, `scratch`, `overflow`, `metadata`,
`codebooks`, `prefill_workspace`, `decode_workspace`, and `connector_staging`.
The arena uses actual row strides; connector staging reports allocated pool
bytes rather than its configured budget. Memory excludes model weights,
allocator reserve, CUDA graph storage and per-step temporary tensors.

The dashboard shows blocks, allocated token slots, arena/raw-equivalent bytes,
auxiliary allocations, tracked allocation totals and the layout-only ratio.
**Do not sum TP block counts**: workers hold shards of the same logical blocks.
Summing memory across components is valid per worker. The raw-equivalent value
is computed for the same block count, not measured from a raw-backend run, and
the layout ratio does not include auxiliary memory or imply an equal total-GPU
capacity gain.

Values appear only after FragPack initializes and a subsequent model execution
delivers connector statistics. An idle freshly started server or a raw/external-
only configuration may have no FragPack samples. Keep vLLM statistics enabled;
do not interpret absent samples as zero. Restart vLLM to load instrumentation
changes, and reload/reprovision Grafana after changing the dashboard JSON.

### External Prefix Reuse

- `daser_external_prefix_cache_queries_total`: prompt tokens queried through
  the connector's external prefix path.
- `daser_external_prefix_cache_hits_total`: queried tokens accepted by vLLM
  as reusable external prefix tokens.

The dashboard shows both rates and their ratio, separate from generic
`daser_cache_*` lookup statistics (which can include HTTP tracing lookups).
It leaves the ratio absent when there are no queries. Accepted tokens are not
an acknowledgement that restoration completed successfully.

### Tier Management

- `daser_tier_bytes_capacity{tier}` / `daser_tier_bytes_used{tier}`: capacity
  and extent allocation, summed across TP lanes.
- `daser_tier_free_holes{tier}`: disjoint free extents per lane; fragmentation
  information, not a percentage of free capacity.
- `daser_tier_evicted_copies_total{tier}` /
  `daser_tier_evicted_bytes_total{tier}`: evicted copies and per-lane bytes.
- `daser_tier_extra_evicted_bytes_total{tier}`: per-lane eviction beyond the
  allocation deficit that triggered it.
- `daser_tier_l1_admission_skips_total`: chunks L1 could not admit.
- `daser_tier_demotions_total{outcome}` /
  `daser_tier_demoted_bytes_total`: exclusive-tier L1-to-L2 migration outcomes
  and successful bytes across all lanes.
- `daser_tier_l1_clean_bytes` / `daser_tier_demoting_bytes`: exclusive-tier
  L1 bytes also valid in L2 / currently being written to L2, across all lanes.
- `daser_tier_demotion_subtree_drops_total`: subtree removals when exclusive
  demotion cannot preserve a dependency chain.

Counter series are created when their events occur. Exclusive-tier series
require the exclusive-tier configuration; absent series are not fabricated as
zeros. Dropping one tier copy is different from losing the chunk's last copy.

## Suggested Alerts

```yaml
groups:
  - name: daser
    rules:
      - alert: DaseRDown
        expr: daser_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: DaseR metrics endpoint is down

      - alert: DaseRVLLMDown
        expr: daser_vllm_health_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: DaseR cannot reach vLLM

      - alert: DaseRTransferErrors
        expr: rate(daser_transfer_operations_total{status="error"}[5m]) > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: DaseR transfer errors detected

      - alert: DaseRLateEvictedCommit
        expr: increase(daser_cache_late_evicted_commits_total[5m]) > 0
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: DaseR observed commits after allocation eviction

      - alert: DaseRL2CapacityHigh
        expr: daser_tier_bytes_used{tier="l2"} / daser_tier_bytes_capacity{tier="l2"} > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: DaseR L2 store is above 90 percent capacity

      - alert: DaseRL1CapacityHigh
        expr: daser_tier_bytes_used{tier="l1"} / daser_tier_bytes_capacity{tier="l1"} > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: DaseR L1 memory cache is above 95 percent capacity
```
