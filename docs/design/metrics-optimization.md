# DaseR 可观测性指标优化设计

## 背景

DaseR 的 Prometheus 指标覆盖 HTTP、IPC、Cache、Transfer、Capacity 五个维度。
指标设计遵循以下原则：

1. **聚焦数据路径**：HTTP 暴露健康状态和 `/metrics`；KV 数据传输的请求指标由 IPC 采集。
2. **保留在途状态**：asyncio 可并发等待多个请求，`daser_ipc_requests_inflight` 用于定位等待和阻塞。
3. **限制 label**：Transfer backend 启动后固定，由 `daser_info` 表达；不把请求、文档或 chunk ID 放入 label。
4. **关联推理表现**：结合 vLLM TTFT/TPOT、L1/L2 驻留和 prefix reuse 深度分析缓存收益。

## 目标

- 避免重复的 HTTP 请求和 slots 容量指标
- 覆盖 L1/L2、cache 深度、transfer 分布和静态配置信息
- Prometheus 多 target scrape 集成 vLLM TTFT/TPOT
- Grafana dashboard 按 4 行分层展示

## 指标体系设计

### Layer 1: Health & Overview

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_up` | Gauge | 保留 | — | 服务存活 |
| `daser_vllm_health_up` | Gauge | 保留 | — | DaseR `/health` 探测到的 vLLM 状态 |
| `daser_info` | Gauge(1) | 保留 | `mode`, `transfer` | 静态配置信息暴露 |

### Layer 2: Cache Performance

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_cache_lookup_total` | Counter | 保留 | `result` | hit/miss 计数 |
| `daser_cache_requested_tokens_total` | Counter | 保留 | — | token 级 denominator |
| `daser_cache_matched_tokens_total` | Counter | 保留 | — | token 级 numerator |
| `daser_cache_committed_chunks_total` | Counter | 保留 | — | chunk 提交数 |
| `daser_cache_evicted_chunks_total` | Counter | 保留 | `reason` | 驱逐计数 |
| `daser_cache_late_evicted_commits_total` | Counter | 保留 | — | 迟到提交 |
| `daser_cache_prefix_reuse_tokens` | Histogram | 保留 | — | 每次 hit 复用 token 数分布 |
| `daser_external_prefix_cache_queries_total` / `daser_external_prefix_cache_hits_total` | Counter | 保留 | — | vLLM 外部前缀查询/命中 token 数 |
| `daser_cache_publish_seconds` | Histogram | 保留 | — | 写者取得归属到 chunk 发布 |
| `daser_cache_pending_overlap_seconds` | Histogram | 保留 | — | 首次需要在途 chunk 的 lookup 到发布 |

`daser_cache_prefix_reuse_tokens` buckets: `(16, 64, 128, 256, 512, 1024, 2048, 4096)`

### Layer 3: Transfer & Storage

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_transfer_operations_total` | Counter | 保留 | `op`, `status` | 成功 transfer 计数，当前采集点写入 `status="ok"` |
| `daser_transfer_bytes_total` | Counter | 保留 | `op` | 传输字节 |
| `daser_transfer_duration_seconds` | Histogram | 保留 | `op` | 传输延迟 |
| `daser_transfer_chunk_size_bytes` | Histogram | 保留 | `op` | 单次传输大小，可能包含多个 chunk |
| `daser_l1_hits_total` | Counter | 保留 | — | 普通 load 从 L1 读取的 chunk 段数 |
| `daser_l1_misses_total` | Counter | 保留 | — | 普通 load 从 L2 读取的 chunk 段数 |
| `daser_tier_bytes_used` | Gauge | 保留 | `tier` | L1/L2 已分配 extent 字节 |
| `daser_tier_bytes_capacity` | Gauge | 保留 | `tier` | L1/L2 容量，不含临时 staging |
| `daser_tier_free_holes` | Gauge | 保留 | `tier` | 每个 lane 的空闲 extent 数 |
| `daser_tier_evicted_copies_total` / `daser_tier_evicted_bytes_total` | Counter | 保留 | `tier` | 各层副本淘汰次数/字节 |
| `daser_tier_l1_admission_skips_total` | Counter | 保留 | — | L1 未接纳的 chunk 数 |
| `daser_tier_demotions_total` | Counter | 保留 | `outcome` | 独占分层降级结果 |
| `daser_prefetch_operations_total` | Counter | 保留 | `status` | lease prefetch 结果 |
| `daser_readahead_bytes_total` | Counter | 保留 | `outcome` | lookup readahead 字节 |
| `daser_load_phase_seconds_total` | Counter | 保留 | `phase` | load 的规划等待和 IO 时间 |
| `daser_l2_write_bytes_total` / `daser_l2_reads_total` | Counter | 保留 | — | 物理 L2 写字节/读操作数 |
| `daser_packed_record_ratio` | Histogram | 保留 | `mode` | stored record / raw slot 字节比 |

`daser_transfer_chunk_size_bytes` buckets: `(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456)`

容量按全部 TP lanes 合计，已用字节包含粒度取整和在途 placement。副本淘汰与
chunk 移出索引分别计数；独占分层和 queue hint 的详细指标见
[tier-residency.md](tier-residency.md)。

store 可在 L1 staged 后回复，L2 写入继续在后台进行。transfer 请求失败看
`daser_ipc_requests_total{status="error"}`；早回复后的 L2 写失败需结合 server
日志和 drain 结果检查，不能用成功请求的延迟推断 L2 写入完成时间。

### Layer 4: IPC (精简)

| 指标 | 类型 | 状态 | Labels | 说明 |
|------|------|------|--------|------|
| `daser_ipc_requests_total` | Counter | 保留 | `op`, `status` | IPC 请求计数 |
| `daser_ipc_request_duration_seconds` | Histogram | 保留 | `op` | IPC 延迟 |
| `daser_ipc_requests_inflight` | Gauge | 保留 | `op` | 正在处理的请求，包含等待 IO 的 coroutine |

### 删除列表

| 指标 | 原因 |
|------|------|
| `daser_http_requests_total` | KV 数据传输的请求指标在 IPC 层采集 |
| `daser_http_request_duration_seconds` | 同上 |
| `daser_http_inflight_requests` | 同上 |
| `daser_store_l2_slots_capacity` | 与 bytes_capacity 重复 |
| `daser_store_l2_slots_used` | 与 bytes_used 重复 |
| `daser_l1_bytes_used` / `daser_l1_bytes_capacity` | 统一为 `daser_tier_bytes_used` / `daser_tier_bytes_capacity`，`tier="l1"` |
| `daser_store_l2_bytes_used` / `daser_store_l2_bytes_capacity` | 统一为 `daser_tier_bytes_used` / `daser_tier_bytes_capacity`，`tier="l2"` |
| Transfer `backend` label | 运行时只有一个值，增加 cardinality 无收益 |

## vLLM 集成

Prometheus 单独采集 vLLM：

```yaml
- job_name: vllm
  static_configs:
    - targets:
        - host.docker.internal:8001
```

引用 vLLM 暴露的指标：

- `vllm:time_to_first_token_seconds` — TTFT 分布
- `vllm:time_per_output_token_seconds` — TPOT 分布
- `vllm:request_success_total` — 请求吞吐

## Dashboard 设计

以下分位数查询以 p95 为例；p50、p99 分别把 `0.95` 改为 `0.5`、`0.99`。
仓库模板中的 L1/L2 容量面板仍使用旧指标，部署时应替换为下表的 tier 查询。

### Row 1: Overview (stat panels, y=0)

| Panel | 查询 | 单位 |
|-------|------|------|
| DaseR Up | `max(up{job="daser"}) or vector(0)` | boolean |
| vLLM Health | `max(up{job="vllm"}) or vector(0)` | boolean |
| Mode | `daser_info` | label value |
| Request Hit Rate | `sum(rate(daser_cache_lookup_total{result="hit"}[5m])) / clamp_min(sum(rate(daser_cache_lookup_total[5m])), 1e-9)` | percent |
| Token Hit Rate | `rate(daser_cache_matched_tokens_total[5m]) / clamp_min(rate(daser_cache_requested_tokens_total[5m]), 1e-9)` | percent |
| L2 Usage | `daser_tier_bytes_used{tier="l2"} / clamp_min(daser_tier_bytes_capacity{tier="l2"}, 1)` | percent |

### Row 2: Cache Deep Dive (timeseries, y=5)

| Panel | 查询 |
|-------|------|
| Cache Lookups | `sum by (result) (rate(daser_cache_lookup_total[5m]))` |
| Prefix Reuse Distribution | `histogram_quantile(0.95, sum by (le) (rate(daser_cache_prefix_reuse_tokens_bucket[5m])))` |
| Evictions & Late Commits | evict rate by reason + late commit increase |

### Row 3: Transfer & Storage (timeseries, y=12)

| Panel | 查询 |
|-------|------|
| Transfer Latency | `histogram_quantile(0.95, sum by (le, op) (rate(daser_transfer_duration_seconds_bucket[5m])))` |
| Transfer Throughput | `sum by (op) (rate(daser_transfer_bytes_total[5m])) / 1e9` |
| Chunk Size Distribution | `histogram_quantile(0.95, sum by (le, op) (rate(daser_transfer_chunk_size_bytes_bucket[5m])))` |
| L1 Hit Rate | `rate(daser_l1_hits_total[5m]) / clamp_min(rate(daser_l1_hits_total[5m]) + rate(daser_l1_misses_total[5m]), 1e-9)` |
| L1 Usage | `daser_tier_bytes_used{tier="l1"} / clamp_min(daser_tier_bytes_capacity{tier="l1"}, 1)` |

### Row 4: Inference Impact (timeseries, y=19)

| Panel | 查询 |
|-------|------|
| TTFT | `histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{job="vllm"}[5m])))` |
| TPOT | `histogram_quantile(0.95, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket{job="vllm"}[5m])))` |
| vLLM Request Rate | `rate(vllm:request_success_total{job="vllm"}[5m])` |

## 实现位置

1. `daser/server/http/app.py` — 健康状态、静态配置和 `/metrics`
2. `daser/server/ipc/server.py` — IPC、transfer、prefetch 和 backend IO 指标
3. `daser/server/core.py` — lookup、外部前缀命中、chunk 发布和移除
4. `daser/server/tiers.py` — 分层容量、淘汰、降级和 queue hint
5. `deploy/monitoring/prometheus/prometheus.yml` — DaseR 与 vLLM scrape jobs
6. `deploy/monitoring/grafana/dashboards/daser-overview.json` — 4 行布局
7. `docs/monitoring/prometheus.md` — 部署、查询和告警示例

