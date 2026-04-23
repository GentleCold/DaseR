# DaseR Connector 热路径优化

**日期：** 2026-04-23
**优化对象：** `DaserConnector` 读/写热路径 + IPC + 哈希函数
**基准：** `benchmarks/bench_e2e_daser_vs_lmcache.py` N=200

## 结果速览

| 指标                          |   优化前 |   优化后 | 变化        |
|-------------------------------|---------:|---------:|:------------|
| DaseR cold 耗时 (s)           |   27.64  |    6.78  | **4.1× 快** |
| DaseR warm 耗时 (s)           |   13.58  |    2.11  | **6.4× 快** |
| DaseR warm prompt tok/s       |   4,462  |  28,772  | **6.4× 快** |
| warm 吞吐 DaseR / LMCache     |   0.17×  |   0.98×  | **追平**    |

参考：LMCache warm 耗时 2.07 s、吞吐 29,335 tok/s（基本同量级）。
correctness 保持不变：DaseR 仍 2/200（已知的 KV 复用精度效应），
LMCache 0/200。

## 瓶颈定位

上一轮 E2E 基准显示 DaseR warm 路径比 LMCache 慢 6×，但裸 bytes I/O
微基准显示 DaseR **比 LMCache 快** 1.5–1.8×。差距不在 GDS 数据平面。
逐行走读 `DaserConnector` 后找到五处 connector 层开销：

1. **`start_load_kv` 按 (layer × block) 粒度单独提交 `kvikio.pread`。**
   一个 300-token 的 prompt 在 Qwen3-8B 上产生 `36 层 × 19 块 ≈ 684`
   次独立 I/O，每次都伴随 `torch.empty(GPU)` + `cupy.asarray` + 闭包
   构造。单 N=200 就是 ~130 K 次 Python/FFI 往返。
2. **读完后又串行做 684 次 GPU 小 copy。** 没有合并，源 buffer 散落。
3. **IPC 短连接。** `IPCClientSync` 每次 `call()` 都 `open/connect/send/
   recv/close` 一个新 Unix socket。每请求 2 次 RPC（lookup + alloc_chunk）。
4. **scheduler 侧 SHA256 哈希前缀。** 对长度从 N 向下按 block_tokens
   递减的所有前缀分别做 SHA256，且单请求内算一次、server 侧又算一次。
5. **`save_kv_layer` 每层独立发起 I/O。** 和读路径同样的 684× 问题。

关键约束：FULL CUDA graph 模式下，vLLM 不会回调 `wait_for_layer_load`
（见 memory `project_cuda_graph_kv_load`），所以 `start_load_kv` 必须
**一次性同步完成**——不能像 LMCache 那样用 per-layer async generator
和注意力计算流水重叠。所以我们只能在"整体同步"的前提下让它做更少的事。

## 关键观察：磁盘布局本身支持单次读

Slot 编址为
`offset = (start_slot + slot_i) × SLOT_SIZE + layer_idx × LAYER_SIZE`，
其中 `SLOT_SIZE = NUM_LAYERS × LAYER_SIZE`。所以同一 chunk 的
**所有 (slot_i, layer_idx) 在磁盘上严格连续**。理论上一次
`kvikio.pread` 可以替掉 684 次。这是本轮优化最大的红利来源。

## 改动清单

### 1. 读路径：per-chunk 合并读 `daser/connector/daser_connector.py:start_load_kv`

**改动**：每个 cache-hit 请求只分配 1 个 `num_slots × SLOT_SIZE` 的
GPU 连续 staging buffer，发出 1 次 `kvikio.pread` 覆盖整 chunk。
所有请求的 read 协程 `asyncio.gather` 并发，然后按 (slot, layer)
切片从 staging 拷到 per-layer KV tensor。

**影响**：I/O 提交数 684 → 1/请求；`torch.empty` 调用同比例下降；
copy-out 次数不变，但源来自连续 GPU buffer。

### 2. 写路径：save 聚合 + 合并写 `daser/connector/daser_connector.py:save_kv_layer` / `wait_for_save`

**改动**：`save_kv_layer` 不再每层立即发起 I/O，只把该层的数据按
磁盘布局偏移 copy 进**per-请求的 GPU staging buffer**。36 层全部过完
后，`wait_for_save` 一次性为每个请求发 1 次 `kvikio.pwrite` 覆盖整
chunk，再批量 commit。

**影响**：写 I/O 提交数同样 684 → 1/请求。GPU copy 次数不变，
但路径更清晰。

### 3. IPC：长连接 + 合并 RPC `daser/connector/ipc_client.py` / `daser/server/ipc_server.py`

**改动**：
- `IPCClientSync` 改为懒连接 + 持久 socket + 单锁串行化；遇到
  `ConnectionError/BrokenPipeError` 会 reset + 重试一次。
- Server 端 `_handle_connection` 改成循环读帧，同一条连接处理多个
  请求直到客户端关闭。
- 新增 `match_and_alloc` RPC：server 先 lookup，命中则返回 chunks；
  未命中则就地 alloc 一个 slot 并返回 alloc info。
- `get_num_new_matched_tokens` 改用 `match_and_alloc` 代替
  `lookup` + `alloc_chunk` 两次调用。

**影响**：每请求 IPC 次数 2 → 1，且省掉每次 socket setup/teardown
（TCP_NODELAY 不需要，走 Unix socket）。

### 4. 哈希：SHA256 → xxh3_128 `daser/connector/daser_connector.py:hash_tokens` / `daser/retrieval/prefix.py:_hash_tokens`

**改动**：
- 两处 `hash_tokens` 都改用 `xxhash.xxh3_128`（128-bit non-crypto hash）。
- 实现方式：`bytes(array.array("i", tokens))` 一次性打包成连续
  C-int 数组后一次哈希，省掉原来 `for tok: h.update(tok.to_bytes(4, ...))`
  的 Python 循环。
- 依赖：`pyproject.toml` 加 `xxhash>=3.4`。

**影响**：短 prompt 上单次哈希 ~5–10× 加速；更关键的是 Python 循环
消失了。安全性：哈希只用于 cache-key 相等性比较，不需要抗碰撞。

### 5. 其他

- 删除了 save 路径上的 `self._store_futures` 列表（被 `_save_staging`
  取代）。
- `wait_for_layer_load` 仍然是 no-op——设计上如此，不能改（CUDA graph
  限制）。

## 验证

### 单元测试

```bash
python -m pytest tests/ --ignore=tests/integration
# 42 passed in 4.76s
```

所有既有单元测试全过。

### Correctness

- DaseR：2/200 mismatch，与优化前一致。属于 vLLM KV 复用的已知
  精度效应（bf16 attention 归约顺序变化 → argmax 在 logit 接近处
  可能翻），与本次优化无关。
- LMCache：0/200 mismatch（这一次运行的 batch 更小，刚好没落在
  logit 临界的样本上）。

### 性能（N=200，GPU CUDA 2，Qwen3-8B，`gpu_util=0.3`）

```
                              DaseR            LMCache
cold elapsed              6.78 s              5.07 s
warm elapsed              2.11 s              2.07 s
cold tok/s (prompt)        8,940              11,962
warm tok/s (prompt)       28,772              29,335
warm/cold speedup          3.22×               2.45×
DaseR warm / LMCache warm = 0.98×
```

## 复现

```bash
source /data/zwt/vllm/bin/activate
pip install xxhash  # 如未装
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 200 \
    --gpu-util 0.3 \
    --max-num-seqs 64
```

## 剩余 gap 分析

DaseR cold 仍比 LMCache 慢 ~35%（6.78 s vs 5.07 s）。可能来源：

- `start_load_kv` 的 copy-out 阶段仍是 `num_reqs × num_layers × num_blocks`
  次小 GPU copy（~137 K 次 @ N=200）。合并成 per-layer 一次 scatter copy
  理论上可继续省 ~19×，但 KV cache 里 block 不一定连续，可能需要
  先 gather 到 block-连续布局再写回。
- cold 路径的 scheduler 侧 hash 还是 O(前缀数)，每个请求会算
  `len(tokens) / block_tokens` 次 xxhash（LMCache 的 `ChunkedTokenDatabase`
  用了链式哈希，可以 O(1) 增量）。
- IPC 虽然是长连接，但仍是一次同步等待；LMCache 的 `LocalDiskBackend`
  完全在进程内，无 socket。如果继续追 cold，可能要让 scheduler 进程
  直接持有 `MetadataStore`，把 IPC 降级为 worker→server 的数据平面
  通道（会破坏进程隔离，属于架构改动，非本轮范围）。

warm 吞吐已经追平 LMCache，说明本轮优化命中了主要瓶颈。如果要
继续打磨，下一步应该对 cold 的 scheduler 路径做 profiling。
