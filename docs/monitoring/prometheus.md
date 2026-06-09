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

Prometheus scrapes DaseR at `host.docker.internal:2026`, which maps to the host
through Docker's host gateway. If DaseR runs on another host or port, edit
`deploy/monitoring/prometheus/prometheus.yml`.

Prometheus and Grafana state is stored under `/data/zwt/daser_monitoring/` by
default. Override it with `DASER_MONITORING_DATA_ROOT` in
`deploy/monitoring/.env`.

The Grafana dashboard is provisioned automatically from:

```text
deploy/monitoring/grafana/dashboards/daser-overview.json
```

## Scrape Target

```yaml
scrape_configs:
  - job_name: daser
    static_configs:
      - targets: ["127.0.0.1:2026"]
```

The DaseR dashboard assumes this Prometheus job label:

```text
job="daser"
```

## Metric Groups

### Service Health

- `daser_up`: DaseR HTTP process liveness.
- `daser_vllm_health_up`: vLLM health as observed by `/health`.
- `daser_http_requests_total{route,status}`: HTTP request count.
- `daser_http_request_duration_seconds{route}`: HTTP latency histogram.
- `daser_http_inflight_requests{route}`: in-flight HTTP requests.

### IPC

- `daser_ipc_requests_total{op,status}`: IPC request count by connector op.
- `daser_ipc_request_duration_seconds{op}`: IPC latency histogram.
- `daser_ipc_inflight_requests{op}`: in-flight IPC requests.

### Cache Effectiveness

- `daser_cache_lookup_total{result}`: cache lookup hit/miss count.
- `daser_cache_requested_tokens_total`: tokens checked for reuse.
- `daser_cache_matched_tokens_total`: tokens matched by cache lookup.
- `daser_cache_committed_chunks_total`: chunks committed and visible.
- `daser_cache_late_evicted_commits_total`: commits ignored after eviction.
- `daser_cache_evicted_chunks_total{reason}`: explicit and capacity evictions.

Use token hit ratio rather than only request hit ratio when judging benefit:

```promql
rate(daser_cache_matched_tokens_total[5m])
/
clamp_min(rate(daser_cache_requested_tokens_total[5m]), 1)
```

### Transfer

- `daser_transfer_operations_total{backend,op,status}`: transfer op count.
- `daser_transfer_bytes_total{backend,op}`: transfer bytes.
- `daser_transfer_duration_seconds{backend,op}`: transfer latency histogram.

The IPC server also logs transfer summaries with decimal GB/s throughput:

```text
[IPC] transfer_load summary backend=iouring status=ok bytes=... elapsed_ms=... throughput_gbps=...
```

Grafana should compute throughput from bytes:

```promql
rate(daser_transfer_bytes_total[5m]) / 1e9
```

### Capacity

- `daser_store_l2_slots_capacity`: total L2 KV slots.
- `daser_store_l2_slots_used`: used L2 KV slots.
- `daser_store_l2_bytes_capacity`: total L2 bytes.
- `daser_store_l2_bytes_used`: used L2 bytes.

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
```
