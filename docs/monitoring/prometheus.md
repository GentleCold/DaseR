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

Prometheus and Grafana state is stored under `/data/zwt/daser_monitoring/` by
default. Override it with `DASER_MONITORING_DATA_ROOT` in
`deploy/monitoring/.env`.

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
rate(daser_cache_matched_tokens_total[5m])
/
clamp_min(rate(daser_cache_requested_tokens_total[5m]), 1)
```

### Transfer & Storage

- `daser_transfer_operations_total{op,status}`: transfer op count.
- `daser_transfer_bytes_total{op}`: transfer bytes.
- `daser_transfer_duration_seconds{op}`: transfer latency histogram.
- `daser_transfer_chunk_size_bytes{op}`: transfer size distribution.
- `daser_l1_hits_total`: L1 memory cache hits.
- `daser_l1_misses_total`: L1 memory cache misses.
- `daser_l1_bytes_used`: current L1 memory usage.
- `daser_l1_bytes_capacity`: total L1 capacity.
- `daser_store_l2_bytes_capacity`: total L2 bytes.
- `daser_store_l2_bytes_used`: used L2 bytes.

The IPC server also logs transfer summaries with decimal GB/s throughput:

```text
[IPC] transfer_load summary backend=iouring status=ok bytes=... elapsed_ms=... throughput_gbps=...
```

Grafana should compute throughput from bytes:

```promql
rate(daser_transfer_bytes_total[5m]) / 1e9
```

### vLLM Inference (from vLLM scrape target)

- `vllm:time_to_first_token_seconds`: TTFT histogram.
- `vllm:time_per_output_token_seconds`: TPOT histogram.
- `vllm:request_success_total`: successful request count.

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
        expr: daser_store_l2_bytes_used / daser_store_l2_bytes_capacity > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: DaseR L2 store is above 90 percent capacity

      - alert: DaseRL1CapacityHigh
        expr: daser_l1_bytes_used / clamp_min(daser_l1_bytes_capacity, 1) > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: DaseR L1 memory cache is above 95 percent capacity
```
