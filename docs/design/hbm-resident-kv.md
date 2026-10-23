# HBM 常驻压缩 KV（FragPack attention backend）

GPU KV cache 以无损 FragPack 记录常驻 HBM，解压融合进 decode attention 内核。
同样的 KV 显存能放下 1.38 倍 block，前缀命中率随之上升；只要 attention 不比
FlashAttention 慢，多出的容量就是净收益。

代码：`daser/attention/`（vLLM backend 与插件）、`daser/ops/hbm_kv.*`
（store）、`daser/ops/fused_attn_decode.*`（decode 内核）、
`daser/compression/attn_page.py`（记录格式参考实现）、
`daser/compression/attn_tables.py`（码表标定）。

## 记录格式

一条记录是一个 (block, layer, K/V, kv head) 的 128 token × 128 维 BF16 页：

- dense 部分 22528 B：低字节 16384 B + 3-bit 高字节符号 6144 B，按内核
  线程顺序排列，解码时每线程直接读自己的片段。
- 可变部分 ≤ `VAR_CAP` = 2048 B：header、escape nibble、raw 标量，16 B 对齐。
- 可变部分超过 `VAR_CAP`（或某 warp 的 raw 标量超过 32）时整页以 32768 B
  raw 存入 overflow 池。

码表按 (layer, K/V, head) 离线标定（`python -m daser.compression.attn_tables`），
运行时由 `DASER_FRAGPACK_TABLES` 指定。`attn_page.encode_record` 是逐位参考。

## Arena 与 vLLM KV spec

`FragPackAttentionBackend.customize_spec` 把每层 page 改成
`page_bytes(H) = 2H × (22528 + VAR_SHARE)`，`VAR_SHARE` = 1280（H=4 时
190464 B，raw 262144 B），vLLM 照常按 KV 字节预算分 block，block 数 ×1.38。
block `b` 第 `l` 层的 page 就是这条 KV cache 的第 `b` 行；store 不另外分配
arena。

page 由该 block-layer 的 `2H` 条记录共享，按编码完成顺序紧密排布：encode
用 CAS 在每 (layer, block) 的 `fill` 计数上占 `22528 + var_len` 字节，放不下
就溢出（spill）为 raw 存入 overflow 池（`StoreStats.spills`）。CAS 失败不推进
`fill`，同 block 更小的记录仍可放进剩余空间。`prepare` 在 block 被 claim
时把它各层的 `fill` 清零。decode / gather 只按 `rec_addr` 读连续的
dense + 可变部分，不关心记录在 page 内的位置。

`VAR_SHARE` 的取值：标定码表下真实 KV 的可变部分平均约 550 B；cc trace 样本
上 1280 B 无 spill，1024 B 约 2% 记录 spill（每条多占 32 KiB overflow，得不
偿失）。固定 2048 B 槽位（`VAR_CAP`）是 ×4/3；理想变长堆 ×1.42。

每条记录经 `rec_addr[L, NB, 2, H]`（绝对地址）和 `rec_len`（可变部分字节，
raw 为 `RAW_VAR_LEN`）定位，可位于：

| 位置 | 时机 |
|------|------|
| 全零 null 记录 | block 从未被写过（null block、dummy run） |
| 持久 staging | block 跨 step 保持未满（decode 尾块、chunked prefill 边界） |
| 临时 scratch | block 在同一 step 内开、满（一层用一份，逐层复用） |
| arena page | block 已满、可编码且 page 放得下 |
| overflow 池 | block 已满但不可编码或 page 已满（raw） |

## 每 step 状态机

1. `prepare`（metadata build 时，图外 eager）：回收不在 live 列表中的
   block 的持久 staging；给收到 offset 0 的 block 分配 staging；列出收到
   offset 127 的 block；把被分配 block 的记录指向 staging，归还其旧
   overflow 记录，并清空其 page 的 `fill`。
2. 每层：`write` 把新 token 写入 staging（内核顺序）→ attention 经表读取
   → `encode` 把本 step 满的 block 编码进 arena page（或 overflow）。
3. 最后一层后 `commit`：释放满 block 的 staging 并清空 close 列表。

不变量：

- 每个运行中请求至多一个跨 step 未满的 block，所以持久 staging 槽数 =
  `max_num_seqs`，不会耗尽。
- live 列表只含每请求「已写一部分」的那个 block。超集不安全：未满时被释放、
  又被新请求分到写位置之后的 block 若留在 live 中，会带着旧 staging，新请求
  就同时占两个槽，并发高时耗尽（GSM8K 32 并发实测触发）。
- 被释放又重分配的 block 不在 live 中，staging 归还；新 owner 写到 offset 0
  时重新 claim。
- 前缀缓存命中的只会是已满、已编码的 block。
- 计数器（编码数、fallback、staging/overflow 丢弃）经 `poll` 异步读回，
  任何丢弃都抛 `StoreError`。

live 列表：vLLM 不把 block 释放事件交给 attention backend，因此插件包装
`GPUModelRunner.execute_model`，按 scheduler output 镜像每请求 block 列表
（`OpenBlocks`），live = 每请求 `num_computed_tokens` 不按 128 对齐时其所在的
block。

## 容量观测

store 创建或重新绑定时，attention 记录实际 block 数、page stride 和持久
GPU tensor 的字节数。connector 加上自身已分配的 load/store staging pool，
通过 vLLM 公开的 `get_kv_connector_stats` / `build_kv_connector_stats` /
`build_prom_metrics` 接口传到 API 进程，再由 vLLM `/metrics` 暴露
`daser_fragpack_*` 指标。观测只读取 host 端 shape，不增加传输 RPC 或
CUDA 同步；统计量按 engine 和 TP rank 保留，重复采样替换快照，不累加容量。

报告包含 arena、staging、scratch、overflow、metadata、码表、prefill/decode
workspace 和 connector staging；不包含模型权重、allocator reserve、CUDA
graph 或每 step 临时分配。block 数包含引擎保留块，不能跨 TP rank 相加。
首次 model execution 上报前没有样本；raw/external-only 模式不伪造 FragPack
容量。完整指标和面板说明见 [监控文档](../monitoring/prometheus.md)。

## Attention 路径

- decode：`decode_attention` 在内核内逐页解码 dense / 可变部分并做
  split-K attention，读 KV 字节量为 raw 的 0.69–0.75。
- prefill / 混合 batch：`gather` 把上下文页解码为 BF16 workspace
  `[ceil(max_model_len/128), 128, H, 128]`，再用 FA3 paged varlen；
  workspace 不够时按请求分组。

## CUDA graph

`_cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE`：decode-only batch 走
FULL graph，混合 batch 走 PIECEWISE（attention eager）。

- `prepare` 在 `build` 中 eager 执行，FULL replay 前已就位。
- 每层内核（write / decode / encode）的 launch 形状只依赖每个 batch 形状
  固定的 host 值（padded token 数、`max_close`、splits）。
- capture 时的 build 不调 `prepare`；`commit` 清空 close 列表，因此 capture
  或 warmup 的前向不会重编码上一 step 的 block；slot −1 的写入为空操作。

## 接入与限制

插件经 entry point `vllm.general_plugins` 注册 `AttentionBackendEnum.CUSTOM`，
`vllm serve ... --attention-backend CUSTOM` 启用；不修改 vLLM 源码。

要求：V2 model runner、BF16 KV、head 128、GQA 8、block 128、SM90、TP 1、
无投机解码。与 DaseR 外部层同时使用时，server 需 `--storage-format fragpack`，
记录以 block blob 原样存取，见 [FragPack 记录的外部层存储](fragpack-external-tiers.md)。

环境变量：`DASER_FRAGPACK_TABLES`（必需）、`DASER_FRAGPACK_OVERFLOW`（overflow 记录数，
默认 8192，256 MiB）。

overflow 池是一张原子位图（每槽 1 bit），encode、`prepare` 的 claim 与
外部层 import 可在不同 CUDA 流上并发分配 / 释放。

overflow 容量：真实 KV（cc trace 前 40 请求，287 万条记录）fallback 只有
1 条；单 token 重复的退化 prompt 超出标定码表，约 96% 记录 fallback，一个
2048 token 的此类 prompt 占约 5.9k 条且随前缀缓存常驻。默认值按这种 prompt
加余量取。overflow 耗尽时丢弃的记录计入 `err_overflow`，`poll` 抛
`StoreError`，不会静默读错数据；更多退化 prompt 同时常驻时需调大
`DASER_FRAGPACK_OVERFLOW`。

## 端到端结果

Qwen3-30B-A3B-FP8，H800（单卡），cc agent trace（P16 会话、客户端并发 4、
300 请求），同 KV 显存 28 GiB，APC 开启，标定码表：

| arm | GPU KV token | ΣTTFT | Σlat | span | TTFT p50 / p90 | TPOT p50 |
|-----|-------------:|------:|-----:|-----:|---------------:|---------:|
| FlashAttention | 305k | 3565 s | 10491 s | 2634 s | 8.6 / 28.1 s | 59.5 ms |
| FlashAttention，raw 显存放大到 FragPack 容量 | 398k | 2282 s | 9700 s | 2432 s | 6.4 / 15.7 s | 62.7 ms |
| FragPack（固定 24576 B 槽位） | 395k | 2264 s | 9651 s | 2419 s | 6.2 / 15.7 s | 63.2 ms |
| FragPack（共享 page，当前实现） | 408k | 2306 s | 9769 s | 2449 s | 6.4 / 16.0 s | 61.8 ms |

FragPack 对 FlashAttention ΣTTFT 1.57×、Σlat / span 1.09×，与同容量 raw 上界相差
1% 以内：融合解压的开销在端到端不可见，收益全部来自容量。负载在这一区间对
容量在 300k–400k token 区间很敏感（约 14 s ΣTTFT / 1k token）。

共享 page 多出 3.2% block，命中 token 多 9%（2.34M vs 2.15M），但 ΣTTFT 在
单次运行的噪声内（命中相同的 287 个请求 TTFT 中位比 1.012，与 raw 上界对
固定槽位的差同量级）。encode 微基准两种布局都是每层 30 µs（64 block），CAS
无可见开销。这个负载在约 400k token 后对容量已不敏感，额外容量要在更紧的
显存或更长的会话下才体现。
