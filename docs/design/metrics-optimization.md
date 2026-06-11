# DaseR 可观测性指标优化设计

## 背景

当前 DaseR 有 25 个 Prometheus 指标，覆盖 HTTP、IPC、Cache、Transfer、Capacity 五个维度。
经专业评审存在以下问题：

1. **冗余指标**：HTTP 层 3 个指标对性能分析无意义（数据面走 IPC）；L2 slots 指标与 bytes 指标完全重复。
2. **低价值指标**：`daser_ipc_inflight_requests` 在单 event loop 架构下几乎总是 0-1。
3. **不合理 label**：Transfer 指标的 `backend` label 启动后固定不变，增加 cardinality 无信息增益。
4. **关键缺失**：无 TTFT/TPOT 可视化、无 L1 缓存指标、无 chunk 大小分布、无 prefix reuse 深度。

## 目标

- 删除 7 个冗余/低价值指标
- 新增 7 个高价值指标（L1 层 + cache 深度 + transfer 分布 + 静态信息）
- Prometheus 多 target scrape 集成 vLLM TTFT/TPOT
- 重新设计 Grafana dashboard 为 4 行分层布局

## 指标体系设计

### Layer 1: Health & Overview

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_up` | Gauge | 保留 | — | 服务存活 |
| `daser_vllm_health_up` | Gauge | 保留 | — | vLLM 健康状态 |
| `daser_info` | Gauge(1) | **新增** | `mode`, `transfer` | 静态配置信息暴露 |

### Layer 2: Cache Performance

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_cache_lookup_total` | Counter | 保留 | `result` | hit/miss 计数 |
| `daser_cache_requested_tokens_total` | Counter | 保留 | — | token 级 denominator |
| `daser_cache_matched_tokens_total` | Counter | 保留 | — | token 级 numerator |
| `daser_cache_committed_chunks_total` | Counter | 保留 | — | chunk 提交数 |
| `daser_cache_evicted_chunks_total` | Counter | 保留 | `reason` | 驱逐计数 |
| `daser_cache_late_evicted_commits_total` | Counter | 保留 | — | 迟到提交 |
| `daser_cache_prefix_reuse_tokens` | Histogram | **新增** | — | 每次 hit 复用 token 数分布 |

`daser_cache_prefix_reuse_tokens` buckets: `(16, 64, 128, 256, 512, 1024, 2048, 4096)`

### Layer 3: Transfer & Storage

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_transfer_operations_total` | Counter | **修改** | `op`, `status` | 去掉 `backend` |
| `daser_transfer_bytes_total` | Counter | **修改** | `op` | 去掉 `backend` |
| `daser_transfer_duration_seconds` | Histogram | **修改** | `op` | 去掉 `backend` |
| `daser_transfer_chunk_size_bytes` | Histogram | **新增** | `op` | 单次传输大小分布 |
| `daser_l1_hits_total` | Counter | **新增** | — | L1 命中次数 |
| `daser_l1_misses_total` | Counter | **新增** | — | L1 miss 次数 |
| `daser_l1_bytes_used` | Gauge | **新增** | — | L1 当前使用量 |
| `daser_l1_bytes_capacity` | Gauge | **新增** | — | L1 总容量 |
| `daser_store_l2_bytes_used` | Gauge | 保留 | — | L2 使用量 |
| `daser_store_l2_bytes_capacity` | Gauge | 保留 | — | L2 总容量 |

`daser_transfer_chunk_size_bytes` buckets: `(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456)`

### Layer 4: IPC (精简)

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_ipc_requests_total` | Counter | 保留 | `op`, `status` | IPC 请求计数 |
| `daser_ipc_request_duration_seconds` | Histogram | 保留 | `op` | IPC 延迟 |

### 删除列表

| 指标 | 原因 |
|------|------|
| `daser_http_requests_total` | HTTP 仅 /health + /metrics，无数据面价值 |
| `daser_http_request_duration_seconds` | 同上 |
| `daser_http_inflight_requests` | 同上 |
| `daser_ipc_inflight_requests` | 单 event loop 下值始终 0-1，无信号 |
| `daser_store_l2_slots_capacity` | 与 bytes_capacity 重复 |
| `daser_store_l2_slots_used` | 与 bytes_used 重复 |
| Transfer `backend` label | 运行时只有一个值，增加 cardinality 无收益 |

## vLLM 集成

Prometheus 新增 scrape target：

```yaml
- job_name: vllm
  static_configs:
    - targets:
        - host.docker.internal:8000
```

引用 vLLM 暴露的指标：
- `vllm:time_to_first_token_seconds` — TTFT 分布
- `vllm:time_per_output_token_seconds` — TPOT 分布
- `vllm:request_success_total` — 请求吞吐

## Dashboard 设计

### Row 1: Overview (stat panels, y=0)

| Panel | 查询 | 单位 |
|-------|------|------|
| DaseR Up | `daser_up` | boolean |
| vLLM Health | `daser_vllm_health_up` | boolean |
| Mode | `daser_info` | label value |
| Request Hit Rate | `rate(daser_cache_lookup_total{result="hit"}[5m]) / clamp_min(rate(daser_cache_lookup_total[5m]), 1)` | percent |
| Token Hit Rate | `rate(daser_cache_matched_tokens_total[5m]) / clamp_min(rate(daser_cache_requested_tokens_total[5m]), 1)` | percent |
| L2 Usage | `daser_store_l2_bytes_used / daser_store_l2_bytes_capacity` | percent |

### Row 2: Cache Deep Dive (timeseries, y=5)

| Panel | 查询 |
|-------|------|
| Cache Lookups | `sum by (result) (rate(daser_cache_lookup_total[5m]))` |
| Prefix Reuse Distribution | `histogram_quantile(0.5/0.95/0.99, rate(daser_cache_prefix_reuse_tokens_bucket[5m]))` |
| Evictions & Late Commits | evict rate by reason + late commit increase |

### Row 3: Transfer & Storage (timeseries, y=12)

| Panel | 查询 |
|-------|------|
| Transfer Latency | `histogram_quantile(0.5/0.95/0.99, sum by (le, op) (rate(daser_transfer_duration_seconds_bucket[5m])))` |
| Transfer Throughput | `sum by (op) (rate(daser_transfer_bytes_total[5m])) / 1e9` |
| Chunk Size Distribution | `histogram_quantile(0.5/0.95, rate(daser_transfer_chunk_size_bytes_bucket[5m]))` |
| L1 Hit Rate | `rate(daser_l1_hits_total[5m]) / clamp_min(rate(daser_l1_hits_total[5m]) + rate(daser_l1_misses_total[5m]), 1)` |
| L1 Usage | `daser_l1_bytes_used / daser_l1_bytes_capacity` |

### Row 4: Inference Impact (timeseries, y=19)

| Panel | 查询 |
|-------|------|
| TTFT | `histogram_quantile(0.5/0.95/0.99, rate(vllm:time_to_first_token_seconds_bucket{job="vllm"}[5m]))` |
| TPOT | `histogram_quantile(0.5/0.95/0.99, rate(vllm:time_per_output_token_seconds_bucket{job="vllm"}[5m]))` |
| vLLM Request Rate | `rate(vllm:request_success_total{job="vllm"}[5m])` |

## 实现计划

1. 修改 `daser/server/http/app.py` — 删除 HTTP metrics middleware，添加 `daser_info` gauge
2. 修改 `daser/server/ipc/server.py` — 删除 inflight gauge，去掉 transfer `backend` label，添加 chunk size histogram 和 L1 指标
3. 修改 `daser/server/core.py` — 删除 slots gauges，添加 prefix reuse histogram
4. 修改 `deploy/monitoring/prometheus/prometheus.yml` — 添加 vLLM scrape job
5. 重写 `deploy/monitoring/grafana/dashboards/daser-overview.json` — 4 行布局
6. 更新 `docs/monitoring/prometheus.md` — 反映新指标体系
7. 更新测试文件 — 适配指标变更

