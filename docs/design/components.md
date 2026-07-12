# 组件详解

## 组件职责

| 组件 | 进程 | 职责 |
|------|------|------|
| `DaserConnector` | vLLM | vLLM `KVConnectorBase_V1` 入口；保留在 `daser/connector/daser_connector.py` 供 `kv_connector_module_path` 加载 |
| `SchedulerConnectorMixin` | vLLM scheduler | `daser/connector/scheduler/adapter.py`；适配 vLLM scheduler hooks，不拥有请求 lifecycle 状态 |
| `RequestLifecycle` | vLLM scheduler | 集中 lookup、pending load/store/alloc/async-save、slot 分配、preemption、completion 和 connector metadata 构造 |
| `WorkerConnectorMixin` | vLLM worker | `daser/connector/worker/adapter.py`；适配 vLLM worker hooks 和启动期 kernel warmup，不拥有 pipeline 状态 |
| `WorkerRuntime` | vLLM worker | 集中 KV layout、step metadata、load/store completion 和 shutdown，组合两个独立 pipeline |
| `LoadPipeline` / `StorePipeline` | vLLM worker | 分别拥有 load/store event loop、IPC client、staging lease、future、backpressure 和 transfer plan |
| worker memory module | vLLM worker | 负责 load/store 独立 fixed pool、indexed lease，以及约 6 GiB combined budget 的单点推导 |
| staging tensor module | vLLM worker | 负责 slot-major KV copy、CUDA producer synchronization 和 RoPE restore，不拥有 request/IPC 状态 |
| `IPCClientSync` | vLLM scheduler | 阻塞式 Unix socket 客户端，用于 `get_runtime_config`、`lookup`、`alloc_chunk` |
| `IPCClientAsync` | vLLM worker | asyncio Unix socket 客户端，用于 `transfer_store`、`transfer_load`、`commit_chunks` |
| `TransferLayer` | DaseR | `daser/transfer/base.py`；server-owned KV 数据传输抽象，由 `IPCServer` 按 runtime config 初始化 |
| `GDSTransferLayer` | DaseR | `daser/transfer/gds/`；封装 kvikio cuFile / compat IO；backend 在初始化时选定，运行期不可切换 |
| `TieredIOUringTransferLayer` | DaseR | `daser/transfer/iouring/layer.py`；编排 L1 pinned-memory + L2 SSD transfer。组合 `L1Cache` + 可选 `L2IoEngine`，拷贝逻辑在 `copy_ops`；`--skip-l2` 即不构造 `L2IoEngine` |
| `L1Cache` / `L2IoEngine` | DaseR | `daser/transfer/iouring/l1_cache.py`（range-keyed pinned-host LRU 缓存）/ `l2_engine.py`（io_uring positioned I/O）；编排层组合二者，pinned-predicate 单向解耦 |
| `ReplacementPolicy` | DaseR | `daser/replacement/`；通用替换策略抽象，当前实现为 `LRUReplacementPolicy` |
| `python -m daser.server` | DaseR | CLI 入口；解析配置，构造 `ServerCore`，启动 HTTP server 和 IPC server，关机保存 index |
| `HTTP server` | DaseR | `daser/server/http/`；FastAPI routes、tokenize/chunk、vLLM HTTP 调用、文档 API 和 `/infer` |
| `IPCServer` | DaseR | `daser/server/ipc/server.py`；Unix socket lifecycle、msgpack framing、op→handler dispatch table |
| `ServerCore` | DaseR | 共享控制面核心；管理 chunk、doc、retrieval、position 状态 |
| `ChunkLifecycle` | DaseR | `daser/server/chunk_lifecycle.py`；chunk commit/写者归属/淘汰状态 + commit waiter，由 `ServerCore` 持有 |
| `ChunkManager` | DaseR | ring buffer slot 分配、淘汰、引用计数和持久化 |
| `MetadataStore` | DaseR | `chunk_index` 和 `slot_map` 内存状态 |
| `DocRegistry` | DaseR | `doc_id -> DocEntry` 文档状态 |
| `RetrievalIndex` | DaseR | cache lookup 抽象；当前实现为 `PrefixHashIndex` 和 `ChunkReuseIndex` |
| `PositionEncoder` | DaseR | position offset 抽象；当前实现为 `FixedOffsetEncoder` 和 `ChunkPositionEncoder` |

---

## 控制面状态边界

`ServerCore` 是 HTTP server 和 IPC server 共享的控制面协调器。它本身不做
KV bytes 的 load/store，而是把请求分发给更窄的组件：slot 分配交给
`ChunkManager`，metadata 查询和持久化状态交给 `MetadataStore`，文档生命周期
交给 `DocRegistry`，cache lookup 交给 `RetrievalIndex`，position offset 交给
`PositionEncoder`。chunk 的 commit/写者归属/淘汰状态和 commit waiter 由
`ChunkLifecycle` 集中维护，保证这几个并行集合的状态转换一致。

`ChunkManager` 和 `MetadataStore` 刻意保持分离。`MetadataStore` 是纯状态容器，
保存 `chunk_key -> ChunkMeta` 的 `chunk_index` 和 `slot_id -> SlotEntry` 的
`slot_map`，并负责这部分状态的 msgpack 序列化。`ChunkManager` 是 ring-buffer
allocator，维护 head/tail，处理 wrap-around skip block、自动淘汰、引用计数联动
和完整 index save/load。也就是说，`MetadataStore` 描述“现在有哪些 chunk 以及
占哪些 slot”，`ChunkManager` 决定“下一次应该写到哪里以及需要淘汰谁”。

`DocRegistry` 记录用户文档视角的状态：`doc_id`、title、原始 token 数、
chunk 列表和每个 chunk 是否仍被 cache 命中。`ChunkManager` 只关心 KV store
里的 slot 资源；当 chunk 被淘汰时，它通知 `DocRegistry` 更新文档的 cached mask。
因此一个是文档目录，一个是 KV ring buffer allocator，二者不能合并。

iouring transfer 的 L1 状态不进入 `MetadataStore`。L1 是
`TieredIOUringTransferLayer` 进程内的 pinned-memory 热缓存，按 byte range 维护
命中表和 LRU `ReplacementPolicy`；它是可丢失的加速层，重启后可以从 L2
`daser.store` 重新填充。L2/ring-buffer metadata 才需要随 `daser.index` 持久化。

`--skip-l2` 下，`TieredIOUringTransferLayer` 复用同样的 logical
slot/file_offset 控制面，但不分配 `daser.store`，不创建 io_uring rings，也不保存
或恢复 `daser.index`。因此 lookup 和 store 都只对当前进程内的 L1 bytes 有意义；
L1 淘汰或进程重启后没有 L2 可恢复。

---

## 存储布局

DaseR 在 `--store-dir` 下维护两个文件：

- `daser.store`：固定 slot 大小的 KV 数据文件。
- `daser.index`：msgpack metadata 快照，关机保存、启动恢复。

当 `--skip-l2` 启用时，这两个文件都不会由当前运行创建或更新；`--store-dir`
只用于普通配置路径和 benchmark scratch 根目录。

```mermaid
block-beta
    columns 7
    s0["s0\nchunk A"]:2
    s2["s2\nSKIP"]:1
    s3["s3\nchunk B"]:2
    s5["s5\nchunk C"]:2
```

每个 slot 保存一个 vLLM KV block 的所有层：

```mermaid
block-beta
    columns 4
    block:slot["slot_k 内部布局"]:4
        l0["layer_0 KV\n(layer_size bytes)"]
        l1["layer_1 KV\n(layer_size bytes)"]
        dots["..."]
        lN["layer_N KV\n(layer_size bytes)"]
    end
```

slot size 从模型 `config.json` 推导：

```text
slot_size = num_kv_heads * head_dim * 2 * num_layers * block_tokens * dtype_bytes
```

文件偏移：

```text
slot_offset(slot_i) = slot_i * slot_size
layer_offset(slot_i, layer_idx) = slot_i * slot_size + layer_idx * layer_size
```

`slot_map` 记录每个 slot 的状态：

- `chunk`：chunk 的首 slot，保存 chunk_key 和 num_slots。
- `cont`：chunk 的后续 slot。
- `skip`：wrap-around 时用于填充文件尾部碎片。

---

## 可插拔接口

### RetrievalIndex

```python
class RetrievalIndex(ABC):
    async def lookup(self, tokens: list[int], model_id: str) -> list[RetrievalMatch]: ...
    async def insert(self, meta: ChunkMeta) -> None: ...      # base: primary _index
    async def remove(self, chunk_key: str) -> None: ...       # base: primary _index
    def _on_insert(self, meta: ChunkMeta) -> None: ...        # hook for secondary index
    def _on_remove(self, chunk_key, meta) -> None: ...        # hook for secondary index
```

base 持有主 `_index`（按 `chunk_key`）并实现 `insert`/`remove`，子类只覆盖
`lookup` 这个真正分叉的方法，需要维护二级结构时覆盖 `_on_insert`/`_on_remove`
钩子。

实现：

- `PrefixHashIndex`：用 `h_i = H(h_{i-1}, block_tokens_i)` 计算 rolling
  prefix key，每个命中对应一个 KV slot，并返回从 prompt 开头连续命中的
  slot 列表。只用主 `_index`。
- `ChunkReuseIndex`：返回多个 block-aligned chunk 命中，用于文档 chunk
  复用；用 `_on_insert`/`_on_remove` 维护按 token-count 分桶的二级索引。

### PositionEncoder

```python
class PositionEncoder(ABC):
    def assign_offset(self, chunk_key: str, token_count: int) -> int: ...
    def get_offset(self, meta: ChunkMeta) -> int: ...
```

实现：

- `FixedOffsetEncoder`：固定 offset，适合 prefix reuse。
- `ChunkPositionEncoder`：为 chunk reuse 分配和读取 chunk position offset。

`daser/server/__main__.py` 根据 `--cache-reuse-mode` 选择 retrieval 和
position 组件。`ServerCore` 只依赖 ABC。

### TransferLayer

```python
class TransferLayer(ABC):
    coalesce_store_spans: bool = False        # backend opts into span coalescing

    @property
    def stats(self) -> TransferStats: ...      # tiering counters (zeroed by default)
    @property
    def l1_bytes_used(self) -> int: ...         # resident L1 bytes (0 by default)

    async def load_bytes(self, dst, file_offset, nbytes) -> int: ...      # abstract
    async def store_bytes(self, src, file_offset, nbytes) -> int: ...     # abstract
    async def load_bytes_grouped(self, dst, spans) -> int: ...   # default: loop
    async def store_bytes_grouped(self, src, spans) -> int: ...  # default: loop
    async def drain(self) -> None: ...          # default no-op; override for write-back
    def close(self) -> None: ...                # abstract
```

可选能力（`coalesce_store_spans`、`stats`、`l1_bytes_used`、`drain`、grouped
方法）都在 ABC 上声明并带默认实现，`IPCServer` 直接按接口调用，不做 `getattr`
探测，也不读后端私有属性。

实现：

- `GDSTransferLayer`：server 通过 CUDA IPC 打开 worker staging buffer，
  再用 kvikio/cuFile 在 GPU buffer 和 SSD file 之间直接传输。无 L1 tier，
  `stats`/`l1_bytes_used` 取 ABC 默认零值，`drain` 为 no-op。
- `TieredIOUringTransferLayer`：编排层，组合两个内聚组件——
  `L1Cache`（`l1_cache.py`，range-keyed pinned-host LRU 缓存）和可选的
  `L2IoEngine`（`l2_engine.py`，io_uring positioned I/O）；纯拷贝/marshalling
  逻辑在 `copy_ops.py`。store 把 bytes 放入 L1 pool 立即对 load 可见，再异步
  写入 L2 SSD；load 先查 L1，miss 再经 L2 读入并 promote 回 L1。L2 文件用
  `O_DIRECT` 打开，offset/byte count 要求 4096-byte 对齐。L1 与 L2 通过注入的
  pinned-predicate 单向解耦（在途 L2 写 pin 住的 L1 slice 不被淘汰 close），
  写回调度胶水留在编排层。`--skip-l2` 即「不构造 `L2IoEngine`」
  （`self._l2 is None`）：只写 L1、load miss 直接报错、不打开 SSD 文件。该模式和
  GDS 冲突。

connector 不感知具体 transfer 实现，只发送 `transfer_store` /
`transfer_load` IPC 请求和 CUDA IPC handle。

---

## IPC 协议

传输层为 Unix socket + 4 字节大端长度前缀 + msgpack body。Scheduler
使用同步客户端，worker 使用 async 客户端。

| op | 调用方 | 请求字段 | 响应字段 |
|----|--------|----------|----------|
| `get_runtime_config` | scheduler / worker init | - | `runtime_config` |
| `lookup` | scheduler | `tokens`, `model_id` | `chunks: list[dict]` |
| `match_and_alloc` | scheduler | `tokens`, `chunk_key`, `model_id` | `chunks`, `alloc` |
| `alloc_chunk` | scheduler | `chunk_key`, `token_count`, `model_id` | `start_slot`, `num_slots`, `file_offset`, `pos_offset` |
| `transfer_store` | worker | `payload`, `spans` | `ok: true`, `bytes` |
| `transfer_load` | worker | `payload`, `spans` | `ok: true`, `bytes` |
| `commit_chunk` | worker | `chunk_key` | `ok: true` |
| `evict_chunk` | scheduler | `chunk_key` | `ok: true` |

`runtime_config` 包含：

```json
{
  "socket_path": "/tmp/daser.sock",
  "store_path": "/path/to/daser-state/daser.store",
  "slot_size": 2359296,
  "block_tokens": 16,
  "model_id": "/path/to/model-or-served-id",
  "transfer_mode": "iouring",
  "l1_size_bytes": 1073741824,
  "l2_size_bytes": 10000000000,
  "skip_l2": false
}
```

`skip_l2=true` 时，`store_path` 为空字符串，`l2_size_bytes` 仍表示控制面
可分配的逻辑 slot 容量。

IPC 错误以 `{"error": "..."}` 返回，connector client 会转换为
`RuntimeError`。
