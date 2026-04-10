# LMCache 源码深度分析文档

> 版本：基于当前仓库代码（2026-04-01）
> 目标：理解 LMCache 整体运作流程，重点解析 Local Storage 的 Load/Store 过程，
> 以及 Load 与 Inference 是否能时间重叠。

---

## 目录

- [0. 核心结论速览](#0-核心结论速览)
- [1. LMCache 整体架构](#1-lmcache-整体架构)
- [2. LMCacheEngineConfig 配置项详解](#2-lmcacheengineconfig-配置项详解)
- [3. vLLM KVConnector API 解析](#3-vllm-kvconnector-api-解析)
- [4. LMCacheConnectorV1 实现分析](#4-lmcacheconnectorv1-实现分析)
- [5. Local Storage Store 流程](#5-local-storage-store-流程)
- [6. Local Storage Load 流程](#6-local-storage-load-流程)
- [7. Load 与 Inference 重叠分析（重点）](#7-load-与-inference-重叠分析重点)
- [8. FSL2Adapter（分布式 L2 适配器）](#8-fsl2adapter分布式-l2-适配器)
- [9. vLLM Async Scheduling 分析](#9-vllm-async-scheduling-分析)
- [10. vLLM OffloadingConnector 分析](#10-vllm-offloadingconnector-分析)

---

## 0. 核心结论速览

| 模式 | Load 与 Inference 是否重叠 | 说明 |
|------|--------------------------|------|
| LMCache 同步模式（默认） | **否** | `start_load_kv()` 完全阻塞：磁盘读 → CPU → GPU，全部完成后才开始 forward pass |
| LMCache 异步模式（`enable_async_loading=True`）| **SSD→CPU 与上一 batch GPU 重叠** | prefetch 提前触发，SSD 读盘在后台进行；CPU→GPU memcpy 仍在 `start_load_kv()` 内阻塞 |
| LMCache Layerwise 模式（`use_layerwise=True`）| **层级 pipeline** | 第 i 层的 CPU→GPU 传输与第 i+1 层的 attention 计算可 pipeline，但不是请求级别的并行 |
| **async scheduling（本项目默认开启）+ async_loading** | **SSD→CPU 与 GPU 重叠，CPU→GPU 不重叠** | `schedule(batch_N)` 在 `execute_model(batch_N-1)` 期间运行，prefetch 提前触发窗口更大；CPU→GPU 仍阻塞 |
| LMCache CPU Backend（无 SSD） | **完全不重叠** | `batched_get_non_blocking()` 是纯内存字典查找（微秒级），无 I/O 可重叠；CPU→GPU memcpy 同步阻塞仍在关键路径 |
| **vLLM OffloadingConnector** | **CPU→GPU 与 GPU inference 真正重叠** ✓ | `start_load_kv()` 仅提交 CUDA stream 任务（非阻塞），transfer 在独立 CUDA stream 上运行，通过 `get_finished()` 异步回调通知完成；下一 batch 调度时才感知结果 |

---

## 1. LMCache 整体架构

### 1.1 组件层次

```
vLLM 引擎（Scheduler 进程）
    │
    ├── LMCacheConnectorV1（Scheduler 侧）
    │       └── LMCacheConnectorV1Impl（Scheduler 侧）
    │               ├── LookupClient ─── ZMQ ───> LookupServer（Worker 侧）
    │               └── load_specs / request_trackers（状态管理）
    │
    └── vLLM Worker 进程
            │
            ├── LMCacheConnectorV1（Worker 侧）
            │       └── LMCacheConnectorV1Impl（Worker 侧）
            │               ├── LMCacheEngine
            │               │       ├── StorageManager
            │               │       │       ├── LocalCPUBackend（CPU 内存池）
            │               │       │       └── LocalDiskBackend（SSD 缓存）
            │               │       ├── GPUConnector（GPU ↔ CPU 数据搬运）
            │               │       └── TokenDatabase（token → 缓存 key）
            │               └── ZMQOffloadServer
            │
            └── vLLM 模型 Worker（attention layers）
```

### 1.2 两阶段工作流（Cold / Warm）

**Cold 阶段**（只写，不需要 GPU 显存共享）：
```
vllm serve 启动（只需加载 Data 前缀）
    └─► forward pass
            └─► save_kv_layer() / wait_for_save()
                    └─► engine.store()
                            └─► GPU KV ─► CPU MemoryObj ─► SSD 文件
```

**Warm 阶段**（加载+推理）：
```
新请求到来（prompt = Data + Task）
    ├─► Scheduler：get_num_new_matched_tokens()
    │       └─► lookup_client.lookup() → LookupServer 查询缓存命中
    │
    ├─► Scheduler：build_connector_meta()
    │       └─► 将 LoadSpec / SaveSpec 打包进 ConnectorMetadata
    │
    └─► Worker forward pass：
            ├─► start_load_kv()
            │       └─► engine.retrieve() → SSD ─► CPU ─► GPU
            │
            ├─► [GPU attention 计算]
            │
            └─► wait_for_save()  （本次 Task tokens 写回 SSD，可选）
```

### 1.3 进程/线程拓扑

- **主进程**：vLLM Scheduler + LMCacheConnectorV1（Scheduler 侧）
- **Worker 进程**：vLLM Worker + LMCacheConnectorV1（Worker 侧）+ LMCacheEngine
- **LMCacheEngine 内部线程**：
  - `storage-manager-event-loop` 线程：运行 StorageManager 的 asyncio 事件循环，处理异步 prefetch 任务
  - `LocalDiskWorker` 内有 `AsyncPQThreadPoolExecutor`（4 个线程）负责实际磁盘 I/O
  - `async-lookup-client-thread`：异步 Lookup 客户端的响应处理线程
  - `LMCacheAsyncLookupServer`：运行在 Worker 侧，负责接收 Scheduler 的 lookup 请求并触发预取

---

## 2. LMCacheEngineConfig 配置项详解

> 源码：`LMCache/lmcache/v1/config.py`

### 2.1 配置加载方式

优先级（从高到低）：
1. 代码显式传入的 `overrides` 字典
2. 环境变量 `LMCACHE_<大写字段名>` （如 `LMCACHE_LOCAL_DISK`）
3. `LMCACHE_CONFIG_FILE` 指定的 YAML 文件
4. 字段默认值

### 2.2 项目相关核心配置项

| 字段名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `chunk_size` | int | 256 | KV 缓存分片大小（token 数量），是 lookup/store 的基本单元 |
| `local_cpu` | bool | True | 是否启用 CPU 内存作为缓存层（项目中设为 **False**） |
| `max_local_cpu_size` | float | 5.0 | CPU 缓存最大容量（GB） |
| `local_disk` | str\|None | None | SSD 缓存路径，如 `/data/zwt/lmcache_kv/`（项目中必须设置） |
| `max_local_disk_size` | float | 0.0 | SSD 缓存最大容量（GB），0 表示不限制 |
| `enable_async_loading` | bool | False | 是否在 Scheduler 阶段提前发起 prefetch（推荐开启以减少 Load 延迟） |
| `use_layerwise` | bool | False | 是否逐层 pipeline（逐层从 CPU→GPU，可与 attention 计算流水） |
| `save_decode_cache` | bool | False | 是否保存 decode 阶段产生的 token 的 KV 缓存 |
| `save_unfull_chunk` | bool | False | 是否保存不满 chunk_size 的最后一个 chunk（Cold 阶段建议开启） |
| `cache_policy` | str | "LRU" | 磁盘缓存淘汰策略，支持 LRU / LFU / FIFO / MRU |
| `blocking_timeout_secs` | int | 10 | 阻塞操作超时秒数 |
| `min_retrieve_tokens` | int | 0 | 命中 token 数低于此值时跳过 retrieve（节省磁盘 I/O） |
| `pre_caching_hash_algorithm` | str | "builtin" | token → chunk hash 算法 |

### 2.3 项目典型配置（local_disk 场景）

```yaml
# LMCACHE_CONFIG_FILE 指向的 YAML
chunk_size: 256
local_cpu: false           # 项目不用 CPU 缓存
local_disk: /data/zwt/lmcache_kv/
max_local_disk_size: 100   # GB
enable_async_loading: true # 推荐开启
use_layerwise: false
save_unfull_chunk: false
cache_policy: LRU
```

或等价的环境变量：
```bash
export LMCACHE_LOCAL_CPU=false
export LMCACHE_LOCAL_DISK=/data/zwt/lmcache_kv/
export LMCACHE_MAX_LOCAL_DISK_SIZE=100
export LMCACHE_ENABLE_ASYNC_LOADING=true
```

---

## 3. vLLM KVConnector API 解析

> 源码：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 3.1 接口总览

`KVConnectorBase_V1` 是所有 KV Connector 的抽象基类，分为两侧的方法：

```
KVConnectorBase_V1
    │
    ├── Scheduler 侧（在 Scheduler 进程中调用）
    │   ├── get_num_new_matched_tokens(request, num_computed_tokens)
    │   │       → 查询外部 KV 缓存命中数，返回可额外加载的 token 数
    │   ├── update_state_after_alloc(request, blocks, num_external_tokens)
    │   │       → vLLM CacheManager 分配 block 后更新 Connector 状态
    │   ├── build_connector_meta(scheduler_output)
    │   │       → 构建本轮 forward pass 的元数据，传递给 Worker
    │   ├── update_connector_output(connector_output)
    │   │       → Worker 返回后更新 Scheduler 侧状态
    │   ├── request_finished(request, block_ids)
    │   │       → 请求完成时调用，返回是否异步保存（及是否延迟释放 block）
    │   └── take_events()
    │           → 获取 KV 缓存事件（用于观测/监控）
    │
    └── Worker 侧（在每个 GPU Worker 进程中调用）
        ├── register_kv_caches(kv_caches)
        │       → 注册 vLLM 的 KV cache 张量（初始化时调用一次）
        ├── start_load_kv(forward_context)
        │       → forward pass 开始前调用，触发外部 KV 缓存加载到 GPU
        ├── wait_for_layer_load(layer_name)
        │       → 每个 attention layer 前调用，确保该层 KV 已就位
        ├── save_kv_layer(layer_name, kv_layer, attn_metadata)
        │       → 每个 attention layer 后调用，发起该层 KV 的异步保存
        ├── wait_for_save()
        │       → forward pass 结束后调用，等待所有保存完成
        └── get_finished(finished_req_ids)
                → 通知已完成的请求，返回异步传输完成的请求 ID
```

### 3.2 元数据流转

```
Scheduler ──build_connector_meta()──► KVConnectorMetadata
                                              │
                                    （IPC 传递给 Worker）
                                              │
                                              ▼
                                      Worker bind_connector_metadata()
                                              │
                            start_load_kv() ──┘── save_kv_layer()
```

`KVConnectorMetadata` 对 LMCache 的具体实现是 `LMCacheConnectorMetadata`，包含：
- `requests: list[ReqMeta]`：每个请求的 token_ids、slot_mapping、LoadSpec、SaveSpec
- `lookup_requests_in_step: list[str]`：本轮 lookup 的请求 ID（用于 unpin）

---

## 4. LMCacheConnectorV1 实现分析

> 源码：
> - `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`（包装层）
> - `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py`（实现层）

### 4.1 双层结构

```
LMCacheConnectorV1（包装层）
    └── self._lmcache_engine = LMCacheConnectorV1Impl（实现层）
```

包装层只是把所有调用转发到实现层。

### 4.2 Scheduler 侧流程

```python
# 每个新请求到来时
get_num_new_matched_tokens(request, num_computed_tokens):
    1. lookup_client.lookup(token_ids, lookup_id=request_id)
       → 同步/异步查询缓存命中数
    2. 记录 LoadSpec：
       LoadSpec(
           vllm_cached_tokens=num_computed_tokens,  # vLLM 本地已有
           lmcache_cached_tokens=hit_tokens,         # LMCache 中命中
           can_load=False  # 待 update_state_after_alloc 确认
       )
    3. 返回 need_to_allocate = hit_tokens - num_computed_tokens

update_state_after_alloc(request, blocks, num_external_tokens):
    → 确认 block 分配成功，设置 load_specs[req_id].can_load = True

build_connector_meta(scheduler_output):
    → 为每个新请求和运行中请求构建 ReqMeta
    → ReqMeta 包含：token_ids、slot_mapping（物理 KV slot 位置）、
                     LoadSpec、SaveSpec
    → 返回 LMCacheConnectorMetadata
```

### 4.3 Worker 侧流程

```python
start_load_kv(forward_context):
    → 获取 LMCacheConnectorMetadata
    → 对每个有 LoadSpec 的请求：
        engine.retrieve(tokens, mask, kvcaches, slot_mapping)
    → 数据从磁盘/CPU 加载到 GPU 的 paged KV buffer

wait_for_save():
    → 调用 engine.lookup_unpin()（释放 async 预取中的 pin）
    → 对每个有 SaveSpec 的请求：
        engine.store(tokens, mask, kvcaches, slot_mapping)
    → GPU KV → CPU → 磁盘（异步）
```

### 4.4 关键数据结构

```python
@dataclass
class LoadSpec:
    vllm_cached_tokens: int    # vLLM 本地 prefix cache 命中数
    lmcache_cached_tokens: int # LMCache 总命中数（含 vLLM 本地的）
    can_load: bool             # 是否被调度器批准加载

@dataclass
class SaveSpec:
    skip_leading_tokens: int   # 已保存过的 token 数（跳过重复保存）
    can_save: bool             # 是否需要保存

@dataclass
class ReqMeta:
    req_id: str
    token_ids: list[int]
    slot_mapping: torch.Tensor  # 每个 token 对应 GPU KV buffer 中的物理 slot
    load_spec: LoadSpec | None
    save_spec: SaveSpec | None
```

---

## 5. Local Storage Store 流程

> 源码：`LMCache/lmcache/v1/storage_backend/local_disk_backend.py`

### 5.1 调用链

```
wait_for_save()
    └─► engine.store(tokens, mask, kvcaches, slot_mapping)
            ├─► token_database.process_tokens()  → 生成 (start, end, CacheEngineKey)
            ├─► storage_manager.allocate()       → 在 CPU 内存池中分配 MemoryObj
            ├─► gpu_connector.batched_from_gpu() → GPU KV chunk ─► CPU MemoryObj
            └─► storage_manager.batched_put()
                    └─► LocalDiskBackend.batched_submit_put_task(keys, memory_objs)
                            └─► submit_put_task(key, memory_obj)
                                    └─► asyncio.run_coroutine_threadsafe(
                                            disk_worker.submit_task("put", async_save_bytes_to_disk),
                                            loop
                                        )
```

### 5.2 数据路径

```
GPU paged KV buffer（vLLM 维护的分页显存）
    │
    │  gpu_connector.batched_from_gpu()
    │  （CUDA stream 异步拷贝到 pinned CPU 内存）
    ▼
CPU MemoryObj（LocalCPUBackend 中的 pinned tensor）
    │
    │  async_save_bytes_to_disk()（在 LocalDiskWorker 线程池中执行）
    │  write_file(buffer, path)
    ▼
SSD 文件：{local_disk}/{model}@{rank}@{chunk_hash}.pt
```

### 5.3 LocalDiskWorker 优先级队列

`LocalDiskWorker` 使用 `AsyncPQThreadPoolExecutor`（4 个工作线程），任务优先级：

| 优先级 | 任务类型 | 说明 |
|--------|----------|------|
| 0（最高） | prefetch | 预取（读磁盘到 CPU） |
| 1 | delete | 缓存淘汰（删除文件） |
| 2（最低） | put | 写入（GPU→CPU→SSD） |

**写盘是最低优先级**，读优先于写，确保 Warm 阶段的 load 不被 Cold 阶段的 write 阻塞。

### 5.4 防重复写入

```python
def submit_put_task(key, memory_obj):
    if self.exists_in_put_tasks(key):
        return None   # 已在写入队列，跳过

    # 检查磁盘空间，必要时 LRU 淘汰
    while current_cache_size + required_size > max_cache_size:
        evict_candidates = cache_policy.get_evict_candidates(...)
        remove(evict_key)

    # 引用计数 +1，防止 CPU 内存被提前释放
    memory_obj.ref_count_up()

    # 异步提交写任务
    asyncio.run_coroutine_threadsafe(disk_worker.submit_task("put", ...), loop)
```

### 5.5 文件命名规则

```python
# key.to_string() 示例：
# Qwen-Qwen3-8B@rank0@world_size1@<chunk_hash_hex>
path = os.path.join(local_disk_path, key.to_string().replace("/", "-") + ".pt")
```

---

## 6. Local Storage Load 流程

### 6.1 同步模式（`enable_async_loading=False`，默认）

```
start_load_kv()
    └─► engine.retrieve(tokens, mask, kvcaches, slot_mapping)
            ├─► _process_tokens_internal()
            │       ├─► token_database.process_tokens() → CacheEngineKey 列表
            │       ├─► storage_manager.get_block_mapping() → 每个 key 在哪个 backend
            │       └─► storage_manager.batched_get(keys, location="LocalDiskBackend")
            │               └─► LocalDiskBackend.get_blocking(key)
            │                       └─► load_bytes_from_disk()
            │                               └─► local_cpu_backend.allocate()
            │                                   read_file(buffer, path)  ← 阻塞读 SSD
            │
            └─► gpu_connector.batched_to_gpu()
                    └─► CPU MemoryObj → GPU paged KV buffer
                        （CUDA stream 异步拷贝，但会同步等待完成）
```

**整个 retrieve() 调用是阻塞的**，在 start_load_kv() 返回之前，所有 KV 数据已经在 GPU 上就位。

### 6.2 异步模式（`enable_async_loading=True`）

异步模式的核心是"**预取提前化**"——在 Scheduler 阶段就开始从磁盘加载到 CPU，而不是等到 forward pass 才加载。

#### 阶段一：Scheduler 阶段触发预取

```
Scheduler: get_num_new_matched_tokens()
    └─► LMCacheAsyncLookupClient.lookup(token_ids, lookup_id=req_id)
            └─► ZMQ PUSH → LMCacheAsyncLookupServer（Worker 侧后台线程）
                    └─► StorageManager.async_lookup_and_prefetch(lookup_id, keys)
                            ├─► backend.batched_async_contains() → 查哪些 key 存在
                            └─► async_serializer.run(
                                    backend.batched_get_non_blocking(lookup_id, keys)
                                )
                                → 创建 asyncio.Task，开始后台加载
                                → ZMQ PUSH hit_count 给 Scheduler（不等加载完成）
```

**注意**：Scheduler 只等待 `hit_count`（命中数），不等待数据加载完成。数据加载在后台进行。

#### 阶段二：Worker forward pass 取结果

```
Worker: start_load_kv()
    └─► engine.retrieve(tokens, mask, ..., req_id=req_id)
            └─► _async_process_tokens_internal()
                    └─► event_manager.get_event_future(EventType.LOADING, req_id)
                        future.result()  ← 阻塞直到预取完成（如已完成则立即返回）
                        → 获取 dict[CacheEngineKey → MemoryObj]（已在 CPU 内存中）
```

#### 阶段三：CPU → GPU

```
            └─► gpu_connector.batched_to_gpu()
                    CPU MemoryObj → GPU paged KV buffer
```

#### 关键点：AsyncSingleSerializer

```python
# StorageManager 中：
if not self.enable_pd and self.config.enable_async_loading:
    self.async_serializer = AsyncSingleSerializer(self.loop)

# AsyncSingleSerializer：
async def run(self, coro_fn, *args, **kwargs):
    if self.lock is None:
        self.lock = asyncio.Lock()
    async with self.lock:   # ← 同一时刻只有 1 个 batched_get 在执行
        return await coro_fn
```

**结论**：即使多个请求同时发起预取，`AsyncSingleSerializer` 保证同一时刻只有一个请求的磁盘读任务在执行，避免并发读导致 CPU 内存不足（死锁）。

### 6.3 数据路径（异步模式）

```
SSD 文件
    │  LocalDiskBackend.batched_get_non_blocking()
    │  （在 LocalDiskWorker 优先级队列中执行，4 线程）
    ▼
CPU MemoryObj（LocalCPUBackend 内存池中，已 pin）
    │  gpu_connector.batched_to_gpu()
    │  （在 forward pass 的 start_load_kv() 中）
    ▼
GPU paged KV buffer（vLLM 维护的分页显存）
```

---

## 7. Load 与 Inference 重叠分析（重点）

### 7.1 同步模式时序图

```
时间轴 →
─────────────────────────────────────────────────────────────►

vLLM Scheduler:
  Step N ──[lookup: A,B,C 命中查询]──────────────────────────►
                                   │
                                   ▼
vLLM Worker forward pass:
          [start_load_kv()]
           A: SSD→CPU→GPU ──────────────►
                                         B: SSD→CPU→GPU ──►
                                                            C: SSD→CPU→GPU ─►
                                                                              [GPU Attention forward]
                                                                              A+B+C 同时推理

特点：
  - A、B、C 的 load 完全串行（在同一个 retrieve() 调用中循环处理）
  - ALL load 完成后，forward pass 才真正开始
  - CPU 和 GPU 的利用率呈锯齿状：load 时 CPU 忙 GPU 闲，inference 时 GPU 忙 CPU 闲
```

### 7.2 异步模式时序图

```
时间轴 →
─────────────────────────────────────────────────────────────►

vLLM Scheduler (Step N):
  [lookup: A,B,C] → 触发 Worker 侧预取

Worker 侧后台线程 (storage-manager-event-loop):
  [A: SSD→CPU]──►[B: SSD→CPU]──►[C: SSD→CPU]──►   (串行，AsyncSingleSerializer)

GPU (上一批次的 inference):
  ─────────────[Batch N-1 GPU Attention]──────────────────────►

vLLM Worker forward pass (Step N):
                                  [start_load_kv()]
                                  future.result()  ← A 已在内存，立即取
                                  [A,B,C: CPU→GPU]
                                  [GPU Attention A+B+C]

图示重叠：
  Batch N-1 GPU inference   ◄─── 与 ───► A,B,C 的 SSD→CPU 预取 **重叠**！

特点：
  - SSD→CPU 的 prefetch 在 Scheduler 阶段就发起，可以与上一批次的 GPU 计算重叠
  - CPU→GPU 仍然是在 start_load_kv() 内串行完成（不与本批次 inference 重叠）
  - 如果 prefetch 在 start_load_kv() 调用时已完成，`future.result()` 立即返回
  - 如果 prefetch 未完成，start_load_kv() 会阻塞等待
```

### 7.3 场景分析：A、B、C 请求同时到达

**问题**：step1 请求 A、B、C 同时到达并触发 load，step2 A、B load 完成并开始 inference，而 C 还在 load，能否 A、B 先推理？

**答案：在当前实现中，不可能。**

#### 原因详解

vLLM 采用**批量推理（batched inference）**模型，一个 forward pass 处理同一批次中的所有请求。`start_load_kv()` 的调用路径如下：

```python
# vllm_v1_adapter.py: start_load_kv()
for idx, request in enumerate(metadata.requests):
    if request.load_spec is None:
        continue

    # 顺序处理每个请求的 retrieve
    ret_token_mask = self.lmcache_engine.retrieve(
        tokens[:lmcache_cached_tokens],
        ...
        req_id=request.req_id,
    )
    # A 处理完，才处理 B，B 处理完才处理 C
```

对于每个请求，`engine.retrieve()` 内部调用 `future.result()` **阻塞等待**该请求的预取完成。所以：

1. 先等待 A 的 SSD→CPU 完成，再将 A 的数据 CPU→GPU
2. 再等待 B 的 SSD→CPU 完成，再将 B 的数据 CPU→GPU
3. 再等待 C 的 SSD→CPU 完成，再将 C 的数据 CPU→GPU
4. **全部完成后**，`start_load_kv()` 返回，GPU attention forward pass 开始

#### 为什么不能并行？

从架构层面看，限制来自两处：

1. **`AsyncSingleSerializer` 的 asyncio.Lock**：同一时刻只允许一个 `batched_get_non_blocking` 协程运行，A 在读磁盘时 B/C 的预取只能等待

2. **`start_load_kv()` 的同步等待**：即使磁盘读有并行潜力，但 `future.result()` 是在 forward pass 的主线程中被依次调用的，没有 A、B、C 并行加载到 GPU 的机制

3. **vLLM 批量推理的基本假设**：一个 forward pass 开始时，所有请求的 KV 数据必须已经在 GPU 上

### 7.4 异步模式的真正价值

异步模式的优化发生在**跨 batch 边界**：

```
Batch N:    [Scheduler N] → [prefetch N: SSD→CPU]
                                    │
                    ┌───────────────┘ （并发）
                    ▼
Batch N-1:          [GPU Attention for Batch N-1]

Batch N 开始时：
    [CPU→GPU for Batch N] → [GPU Attention for Batch N]
```

通过提前 prefetch，原本在 forward pass 内的 SSD→CPU 时间被"隐藏"到上一个 batch 的 GPU 计算背后，理想情况下可以显著减少 forward pass 的等待时间。

**实际效果取决于**：
- SSD 读取速度（NVMe vs SATA）
- 上一个 batch 的 GPU 计算时间
- Prefetch 是否提前足够多的时间发起

### 7.5 Layerwise 模式的 Pipeline

如果设置 `use_layerwise=True`，会使用 `retrieve_layer()` 生成器，实现层级 pipeline：

```
Layer 0:  [CPU→GPU 搬运 Layer 0 KV]
Layer 1:  [Attention Layer 0]  +  [CPU→GPU 搬运 Layer 1 KV]  ← 流水线重叠
Layer 2:  [Attention Layer 1]  +  [CPU→GPU 搬运 Layer 2 KV]
...
```

但这是**单请求批次内的层级流水线**，不是请求 A/B/C 之间的并行。

---

## 8. FSL2Adapter（分布式 L2 适配器）

> 源码：`LMCache/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py`

### 8.1 与 LocalDiskBackend 的区别

| 特征 | LocalDiskBackend | FSL2Adapter |
|------|------------------|-------------|
| 使用场景 | 单节点 vLLM + LMCache 集成 | 分布式 LMCache 集群（L2 层） |
| 调用方式 | 通过 `StorageManager` | 通过 `L1Manager` / `StorageController` |
| I/O 接口 | 同步（`ThreadPoolExecutor`）| 完全异步（`aiofiles` + asyncio） |
| 元数据存储 | Python dict + `DiskCacheMetadata` | 文件名即 key（无额外元数据文件） |
| 文件命名 | `{key}.pt`（PyTorch 序列化格式） | `{model}@{kv_rank}@{chunk_hash}.data`（纯字节） |
| 通知机制 | 回调函数 `on_complete_callback` | `eventfd`（Linux 事件文件描述符） |

### 8.2 asyncio 事件循环线程模型

```python
class FSL2Adapter:
    def __init__(self, config):
        # 独立的 asyncio 事件循环，运行在 daemon 线程中
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # 3 个 eventfd，分别通知 store/lookup/load 任务完成
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
```

### 8.3 任务提交与结果获取

**提交（非阻塞）**：
```python
task_id = adapter.submit_store_task(keys, objects)
# 立即返回 task_id，后台异步执行
```

**查询结果（轮询）**：
```python
# 通过 eventfd 判断是否有完成的任务
completed = adapter.pop_completed_store_tasks()
# 返回 {task_id: success/failure}
```

**文件命名格式**：
```
{safe_model_name}@{kv_rank:#010x}@{chunk_hash_hex}.data

例：Qwen-Qwen3-8B@0x00000000@a1b2c3d4....data
```

### 8.4 O_DIRECT 支持

```python
# 通过 use_odirect=True 绕过操作系统页缓存（Page Cache）
# 适用于大块对齐写入，减少 CPU 内存占用
# 要求 buffer 大小对齐到文件系统 block size（通常 4096 字节）
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
```

---

## 9. vLLM Async Scheduling 分析

### 9.1 async_scheduling 是否默认开启？

**结论：是的，在标准单机场景下默认开启。**

配置逻辑位于 `vllm/vllm/config/vllm.py:588`：

```python
elif self.scheduler_config.async_scheduling is None:
    # Enable async scheduling unless there is an incompatible option.
    if self.parallel_config.pipeline_parallel_size > 1:
        self.scheduler_config.async_scheduling = False  # PP 不兼容
    elif speculative_config is not None and ...:
        self.scheduler_config.async_scheduling = False  # 部分 spec decode 不兼容
    elif not executor_supports_async_sched:
        self.scheduler_config.async_scheduling = False  # 非 mp/uni/external_launcher
    else:
        self.scheduler_config.async_scheduling = True   # ← 默认走这里
```

`async_scheduling` 字段默认值为 `None`，vLLM 在 `finalize_model_config()` 中自动检测：
- 无流水线并行（`pipeline_parallel_size=1`）
- 无不兼容的 spec decode
- 使用 `mp`（多进程）或 `uni`（单进程统一执行器）后端

以上条件均满足时，**自动启用 async scheduling**，并打印日志：
```
Asynchronous scheduling is enabled.
```

我们的项目（单卡、无 spec decode、默认 mp executor）**走的就是 async scheduling**。

---

### 9.2 Async Scheduling 工作原理

核心思想：让 **Scheduler Step N** 与 **GPU 执行 Step N-1** 同时运行，消除 GPU 等待调度的空隙。

**关键参数**：
```python
# vllm/v1/executor/multiproc_executor.py
max_concurrent_batches = 2  # async_scheduling=True 时

# vllm/v1/engine/core.py
batch_queue = deque(maxlen=2)
```

**执行函数切换**：
```python
# core.py __init__
self.step = self.step if self.batch_queue is None else self.step_with_batch_queue
```

**`step_with_batch_queue()` 流程**（`vllm/v1/engine/core.py:411`）：

```
时间轴：
  t0: schedule(batch_1) → execute_model(batch_1, non_block=True)
      batch_queue = [(future_1, sched_out_1)]
      queue 未满，立即 return（不阻塞）

  t1: schedule(batch_2) → execute_model(batch_2, non_block=True)
      batch_queue = [(future_2, sched_out_2), (future_1, sched_out_1)]
      queue 已满（size=2），不能继续 schedule

  t2: batch_queue.pop() → future_1.result()（阻塞等待 batch_1 GPU 完成）
      update_from_output(sched_out_1, output_1)

  t3: schedule(batch_3) → execute_model(batch_3, non_block=True)
      ...重复循环
```

**关键点**：
- `execute_model(..., non_block=True)` 提交任务但**不等待**结果
- GPU 执行 batch_1 时，CPU 已经在调度 batch_2
- `future.result()` 只在 queue 满时才阻塞取结果

---

### 9.3 与 LMCache 的交互：更大的 overlap 窗口

`AsyncScheduler` 类（`vllm/v1/core/sched/async_scheduler.py`）继承自 `Scheduler`，调度逻辑**基本相同**，只增加了 `num_output_placeholders` 机制。
因此，`get_num_new_matched_tokens()`（触发 LMCache prefetch）的调用时机由 `step_with_batch_queue()` 的流程决定。

**完整时序（async scheduling + enable_async_loading=True）**：

```
step N-1:
  [CPU] schedule(batch_N-1):
        get_num_new_matched_tokens(A,B,C)  ← 触发 ZMQ lookup+prefetch
        → LookupServer 向 LocalDiskWorker 提交读文件任务（priority=0）
  [GPU] execute_model(batch_N-2)           ← 与上面的 CPU 调度**同时进行**

step N（batch_queue 满，需要等 batch_N-1 结果）:
  [GPU] execute_model(batch_N-1) 开始运行
  [CPU] 等待 future_{N-1}.result()

  此时：
    - SSD → CPU 的 prefetch 在后台 LocalDiskWorker 线程池（4线程）中运行
    - GPU 正在执行 batch_{N-1} 的 forward pass
    ★ SSD 读取 与 GPU forward 在此时间窗口内重叠！

  [CPU] future_{N-1}.result() 返回（GPU batch_N-1 完成）
  [CPU] update_from_output(batch_N-1)      ← 此时 prefetch 可能已完成

step N+1（处理 batch_N）:
  [CPU] schedule(batch_N) 已在队列中
  [CPU] start_load_kv(batch_N):
        _async_process_tokens_internal(A,B,C)
        → event_manager.get_event_future(LOADING, req_id).result()
          （阻塞等待 prefetch 完成，如果还没完成的话）
  [GPU] execute_model(batch_N)             ← start_load_kv 完成后才开始
```

**结论对比**：

| 模式 | SSD→CPU 重叠 | CPU→GPU 重叠 |
|------|------------|------------|
| sync（默认） | 否 | 否 |
| async_loading（无 async scheduling） | **跨 batch** 部分重叠（prefetch 在上一 batch 的 GPU 执行期间触发） | 否（start_load_kv 仍阻塞） |
| async_loading + **async scheduling** | **重叠窗口更大、更确定** | 否（start_load_kv 仍然先于 execute_model） |

在 **async scheduling** 下，`get_num_new_matched_tokens()`（prefetch 触发）提前了整整一个 `schedule()` 调用的时间——此时 GPU 正在执行上一个 batch，SSD→CPU 的读取和 GPU 计算**同时进行**。

---

### 9.4 本项目（Cold/Warm 场景）的实际影响

**项目配置**：`async_scheduling=True`（默认），`enable_async_loading=True`（已配置）

**Warm 阶段时序（理想情况）**：

```
Batch N-1 调度阶段（GPU 运行 batch N-2）：
  A、B、C 的 lookup 发起 ZMQ prefetch
  → LocalDiskWorker 4线程开始读 SSD

Batch N-1 GPU 执行阶段：
  SSD → CPU 的读取在后台进行               ← overlap 窗口

Batch N-1 output 处理完毕后：
  start_load_kv() 等待 prefetch event
  若 SSD 读完 → 立即开始 CPU→GPU 搬运
  若未读完  → 阻塞等待

start_load_kv() 返回 → execute_model(batch N) 开始
```

**实际 overlap 效果**：
- SSD 读取（通常是瓶颈）可以与 GPU 推理重叠
- CPU→GPU 搬运（`batched_to_gpu()`）仍然在 `start_load_kv()` 内阻塞完成，**不与 GPU 重叠**
- 多请求（A、B、C）的 SSD 读取通过 `LocalDiskWorker` 4线程并行执行（`AsyncPQThreadPoolExecutor`），但 CPU→GPU 阶段由 `AsyncSingleSerializer` 串行化

**吞吐提升来源**：主要是 SSD IO 与 GPU 计算的批间流水线，而非单次请求的延迟降低。

---

### 9.5 AsyncScheduler 类的额外工作

`AsyncScheduler._update_after_schedule()` 相比基类额外做了：
1. 为即将生成 token 的请求增加 `num_output_placeholders`
2. 预填充 `spec_token_ids = [-1] * num_spec_tokens`（spec decode 场景）

`AsyncScheduler._update_request_with_output()` 相比基类额外做了：
1. 处理 `discard_latest_async_tokens` 标志（异步模式下 preemption 时丢弃最新 token）
2. 调用 `kv_cache_manager.cache_blocks()` 缓存新生成 token 的 KV 块

这些变化对 LMCache 的 load/store 逻辑没有影响。

---

## 10. vLLM OffloadingConnector 分析

### 10.1 架构概览

OffloadingConnector 是 vLLM 官方实现的 CPU-GPU KV offloading 方案，与 LMCacheConnector 使用同一套 `KVConnectorBase_V1` 接口，但实现思路截然不同。

```
vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py
vllm/vllm/v1/kv_offload/worker/cpu_gpu.py   ← 核心传输实现
vllm/vllm/v1/kv_offload/worker/worker.py    ← OffloadingWorker/Handler 抽象
vllm/vllm/v1/kv_offload/abstract.py         ← OffloadingManager
```

**两个子组件**：
- `OffloadingConnectorScheduler`：运行在 scheduler 侧，管理 `OffloadingManager`（LRU/ARC block 管理），查询 CPU cache 命中
- `OffloadingConnectorWorker`：运行在 worker 侧，通过 `SingleDirectionOffloadingHandler` 执行真正的传输

---

### 10.2 CPU→GPU 传输机制：独立 CUDA stream

`SingleDirectionOffloadingHandler.transfer_async()`（`cpu_gpu.py:108`）：

```python
def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
    # 从 stream pool 取一个 CUDA stream（或新建）
    stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
    event = self._event_pool.pop() if self._event_pool else torch.Event()

    if self._transfers:
        _, _, last_event = self._transfers[-1]
        stream.wait_event(last_event)     # 保证顺序：等上一个传输完成
    with torch.cuda.stream(stream):
        ops.swap_blocks(...)              # 提交 CPU->GPU 拷贝到独立 stream
        event.record(stream)              # 记录完成事件

    self._transfers.append((job_id, stream, event))
    return True                           # 立即返回，不阻塞！
```

**关键点**：
- `ops.swap_blocks()` 在独立 CUDA stream 上执行，与 default stream（inference）并发
- `transfer_async()` 提交后**立即返回 True**，不等待传输完成
- 传输完成通过 `event.query()`（非阻塞轮询）或 `event.synchronize()`（阻塞等待）感知

---

### 10.3 Load 流程：真正的非阻塞提交

**`start_load_kv()`** → `start_kv_transfers(metadata)`（`offloading_connector.py:493`）：

```python
def start_kv_transfers(self, metadata):
    # 提交上一步延迟的 store 任务
    for job_id, transfer_spec in self._unsubmitted_store_jobs:
        self.worker.transfer_async(job_id, transfer_spec)   # 非阻塞
    self._unsubmitted_store_jobs.clear()

    # 提交本步所有 load 任务
    for req_id, transfer_spec in metadata.reqs_to_load.items():
        job_id = self._generate_job_id()
        self._load_job[req_id] = job_id
        self.worker.transfer_async(job_id, transfer_spec)   # 非阻塞，立即返回
```

`start_load_kv()` 执行完毕时，**CPU→GPU 传输已在独立 CUDA stream 上运行，但尚未完成**。随后 inference forward pass 在 default stream 上启动，两者并发执行。

---

### 10.4 完成通知：通过 get_finished() 异步回调

`get_finished()` 每个 step 被调用一次（`update_from_output` 内），用 `event.query()`（非阻塞）轮询：

```python
def get_finished(self) -> list[TransferResult]:
    results = []
    while self._transfers and self._transfers[0][2].query():  # CUDA event 已完成？
        job_id, stream, event = self._transfers.popleft()
        results.append((job_id, True))
        # 回收 stream/event 到 pool
    return results
```

Scheduler 侧通过 `finished_recving_kv_req_ids` 感知哪些请求的 load 已完成，**下一个调度 step** 才将该请求真正纳入 batch 进行 prefill。

---

### 10.5 时序图：CPU→GPU 与 inference 真正重叠

```
时间轴（OffloadingConnector + async scheduling）：

Step N-1 调度阶段（GPU 跑 batch_N-2）：
  get_num_new_matched_tokens(req_A) → 命中 CPU offload cache
  update_state_after_alloc(req_A)  → 准备好 src/dst block_ids
  build_connector_meta()           → reqs_to_load = {req_A: transfer_spec}

Step N 执行阶段（GPU 跑 batch_N-1）：
  start_load_kv()
    → worker.transfer_async(job_A, spec)  ← 提交到独立 CUDA stream，立即返回
  model.forward(batch_N-1)               ← default stream
  ★ CPU→GPU 传输（独立 stream）与 inference（default stream）同时运行！

Step N update_from_output：
  get_finished() → event.query() → 若传输完成：
    finished_recving_kv_req_ids.add(req_A)

Step N+1 调度阶段：
  _update_waiting_for_remote_kv(req_A) → 已完成 → 正式调度 req_A 进 prefill
```

**与 LMCache 的核心差异**：

| 对比项 | LMCache | OffloadingConnector |
|--------|---------|---------------------|
| `start_load_kv()` 是否阻塞 | **阻塞**（等 CPU→GPU memcpy 完成） | **不阻塞**（仅提交 CUDA stream 任务） |
| CPU→GPU 传输机制 | `batched_to_gpu()`（同步 memcpy） | `ops.swap_blocks()` on 独立 CUDA stream |
| 完成通知方式 | 同步返回（调用即完成） | `get_finished()` 轮询 CUDA event |
| 请求调度方式 | load 完成后同 batch 参与 inference | load 提交后当前 batch 继续，**下一 batch** 才调度该请求 |
| CPU→GPU 与 inference overlap | **否** | **是** ✓ |

---

### 10.6 Store 流程：延迟提交设计

`wait_for_save()` → `prepare_store_kv(metadata)`（`offloading_connector.py:507`）：

```python
def prepare_store_kv(self, metadata):
    for req_id, transfer_spec in metadata.reqs_to_store.items():
        job_id = self._generate_job_id()
        # 不立即提交！先放入 _unsubmitted_store_jobs
        self._unsubmitted_store_jobs.append((job_id, transfer_spec))
```

Store 任务被**故意延迟**到下一个 step 的 `start_kv_transfers()` 开头才提交：
```python
# NOTE(orozery): defer the store to the beginning of the next engine step,
# so that offloading starts AFTER transfers related to token sampling,
# thereby avoiding delays to token generation due to offloading.
```

GPU→CPU 传输也在独立 stream 上，`stream.wait_stream(current_stream)` 保证等 inference 完成后再开始 offload。

---

### 10.7 为什么 LMCache 没有做到这一点？

LMCache 的 `batched_to_gpu()` 是在 `start_load_kv()` 内**同步执行的 CPU-initiated memcpy**，需要 CPU 线程等待传输完成后才能继续。

OffloadingConnector 使用 `ops.swap_blocks()` 这个 **CUDA kernel**，它在 GPU 上执行，提交后 CPU 立即返回。default stream 上的 inference kernel 和独立 stream 上的 swap_blocks 可以由 GPU 的 copy engine 并发处理。

这是两种根本不同的传输发起方式：
- CPU 发起 `cudaMemcpy` → CPU 阻塞或需要显式异步管理
- GPU 上运行 `swap_blocks` kernel → 天然与其他 stream 并发

---

## 附录：关键文件速查

| 文件路径 | 说明 |
|----------|------|
| `LMCache/lmcache/v1/config.py` | 所有配置项定义（`_CONFIG_DEFINITIONS`），环境变量映射 |
| `LMCache/lmcache/v1/manager.py` | `LMCacheManager`：组件生命周期管理 |
| `LMCache/lmcache/v1/cache_engine.py` | `LMCacheEngine`：store/retrieve/store_layer/retrieve_layer |
| `LMCache/lmcache/v1/storage_backend/storage_manager.py` | `StorageManager`：多后端调度、async_lookup_and_prefetch |
| `LMCache/lmcache/v1/storage_backend/local_disk_backend.py` | `LocalDiskBackend`：磁盘读写、优先级队列 |
| `LMCache/lmcache/v1/storage_backend/local_cpu_backend.py` | `LocalCPUBackend`：CPU 内存池，作为磁盘读写的中间缓冲 |
| `LMCache/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` | `FSL2Adapter`：分布式场景的 L2 文件系统适配器 |
| `LMCache/lmcache/v1/lookup_client/lmcache_async_lookup_client.py` | 异步 Lookup 客户端（Scheduler 侧）和服务端（Worker 侧） |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py` | `KVConnectorBase_V1`：抽象基类，完整 API 文档 |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | `LMCacheConnectorV1`：vLLM 侧包装层 |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` | `LMCacheConnectorV1Impl`：核心实现，包含 RequestTracker、ReqMeta、LoadSpec、SaveSpec |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | `OffloadingConnector`：vLLM 官方 CPU offloading 实现，CPU→GPU 与 inference 真正重叠 |
| `vllm/vllm/v1/kv_offload/worker/cpu_gpu.py` | `SingleDirectionOffloadingHandler`：独立 CUDA stream + `swap_blocks` kernel，非阻塞传输 |
| `vllm/vllm/v1/kv_offload/worker/worker.py` | `OffloadingWorker`/`OffloadingHandler`：传输抽象层 |
