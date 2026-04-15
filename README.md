# DaseR

RAG-native KV cache service for LLM inference.
Integrates with vLLM via `KVConnectorBase_V1`; stores KV tensors directly to NVMe using
NVIDIA cuFile (GDS) or kvikio compat-mode as a fallback.

```
┌─────────────────────────────────────────┐
│               vLLM process              │
│  ┌──────────────────────────────────┐   │
│  │  DaserConnector (KVConnectorV1)  │   │
│  │  ┌──────────────┐ ┌──────────┐  │   │
│  │  │ Scheduler    │ │ Worker   │  │   │
│  │  │ IPCClientSync│ │ GDS IO   │  │   │
│  │  └──────┬───────┘ └────┬─────┘  │   │
│  └─────────┼──────────────┼────────┘   │
└────────────┼──────────────┼────────────┘
             │ Unix socket  │ cuFile DMA
   ┌──────────▼──────────┐  │
   │    DaseR Server      │  │  ┌──────────┐
   │  IPCServer           │  └─►│  NVMe    │
   │  PrefixHashIndex     │     │ .store   │
   │  ChunkManager        │     └──────────┘
   └─────────────────────┘
```

## Quick Start

### 1. Environment

```bash
source /data/zwt/vllm/bin/activate
cd /home/zwt/daser_project/DaseR
pip install -e .
```

### 2. Run the DaseR server

```bash
python -m daser.server \
    --store-path /data/zwt/daser_test/daser.store \
    --store-size 10737418240 \
    --socket-path /tmp/daser.sock \
    --index-path /tmp/daser.index
```

| Flag | Default | Description |
|------|---------|-------------|
| `--store-path` | `/data/zwt/daser_test/daser.store` | Pre-allocated NVMe store file |
| `--store-size` | `10 GB` | Total store capacity in bytes |
| `--socket-path` | `/tmp/daser.sock` | Unix domain socket |
| `--index-path` | `/tmp/daser.index` | Metadata index file |
| `--slot-size` | `2097152` (2 MB) | Bytes per KV slot |

### 3. Run tests

```bash
pytest tests/ -q
# Expected: 42 passed
```

### 4. Lint

```bash
ruff check daser/ tests/
ruff format --check daser/ tests/
```

### 5. Storage benchmark (DaseR vs LMCache)

Compares kvikio compat-mode IO against LMCache `LocalDiskBackend`
using IMDB review-derived KV chunk sizes (2 MB/slot, 32 layers, bfloat16).

```bash
python benchmarks/bench_storage_imdb.py \
    --num-chunks 100 \
    --store-dir /data/zwt/daser_test \
    --imdb /data/zwt/imdb.csv
```

**Latest results** (100 chunks × 2 MB = 0.21 GB, btrfs filesystem, kvikio compat mode):

| Metric | DaseR | LMCache | Speedup |
|--------|-------|---------|---------|
| Write | 1.83 GB/s | 1.96 GB/s | 0.93× |
| Cold read | 11.6 GB/s | 7.5 GB/s | **1.54×** |
| Warm read | 14.7 GB/s | 8.0 GB/s | **1.83×** |

> Write gap: DaseR writes from GPU memory (cupy) — compat mode requires a GPU→CPU
> staging step. On GDS-capable hardware (XFS + cuFile direct DMA) this gap disappears.

### 6. Connect vLLM to DaseR

Add to your `vllm serve` or `LLM(...)` call:

```python
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="DaserConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "socket_path": "/tmp/daser.sock",
        "store_path": "/data/zwt/daser_test/daser.store",
        "slot_size": 2097152,       # must match server --slot-size
        "block_tokens": 16,
        "model_id": "my-model",
    },
)

llm = LLM(model="...", kv_transfer_config=ktc, enable_prefix_caching=False)
```

## Repository Layout

```
daser/
  config.py               DaserConfig dataclass
  connector/
    daser_connector.py    KVConnectorBase_V1 (Scheduler + Worker roles)
    gds_transfer.py       GDSTransferLayer — kvikio async IO wrapper
    ipc_client.py         IPCClientSync (scheduler) / IPCClientAsync (worker)
  server/
    __main__.py           Server entry point
    ipc_server.py         Unix socket IPC handler (lookup / alloc / commit)
    chunk_manager.py      Ring buffer with LRU eviction
    metadata_store.py     Slot metadata (key → SlotEntry)
  retrieval/
    prefix.py             PrefixHashIndex — SHA256 prefix cache lookup
  position/
    fixed_offset.py       FixedOffsetEncoder — slot → file byte offset
benchmarks/
  bench_storage_imdb.py   Cold/warm IO benchmark vs LMCache
tests/                    42 unit tests (pytest)
doc/
  design/                 Design docs (see CLAUDE.md for writing guidelines)
  plans/                  Implementation plans
```

## Architecture Notes

- **Control plane** (index lookup, slot allocation) runs in the DaseR server process over Unix socket IPC.
- **Data plane** (NVMe DMA) runs in the vLLM worker process, which owns the CUDA context and registered GPU buffers.
- **GDS backend** is selected once at `GDSTransferLayer` init: `CuFileBackend` when GDS is available, `COMPAT` (kvikio thread pool) otherwise.
- **Worker async safety**: `DaserConnector` in worker role uses a dedicated background `asyncio` event loop (`_bg_loop` / `_bg_thread`) so that synchronous vLLM callbacks can submit IO via `asyncio.run_coroutine_threadsafe` without re-entering vLLM's own loop.

## 整体架构分析（RAG KV Cache）

1. **分层清晰**：`vLLM` 侧仅负责 KV 读写执行（Connector + GDS），`DaseR Server` 负责检索、分配与元数据管理，控制面/数据面职责边界明确。  
2. **高性能数据路径**：优先走 cuFile/GDS 直达 NVMe↔GPU；不可用时退化到 compat/io_uring 路径，保证可用性与性能上限。  
3. **可扩展检索能力**：`RetrievalIndex` 抽象让前缀匹配之外的语义检索/混合检索可以独立演进，不耦合底层存储。  
4. **存储管理可控**：单大文件 + 固定槽位 ring buffer，配合 `ChunkManager` 做淘汰，避免“小文件风暴”和碎片化元数据开销。  
5. **多实例共享潜力**：DaseR 作为独立服务，可被多个 vLLM 实例复用，适合 RAG 场景下跨请求、跨会话复用热点 KV。  
6. **关键约束**：GDS DMA 必须在持有 CUDA 上下文的 vLLM 进程执行；服务端不直接做 GPU IO，仅通过 IPC 提供元数据决策。  

> 结论：该系统采用“**服务端控制面 + 推理侧数据面**”的架构，在保持检索扩展性的同时，最大化利用 GDS 的吞吐优势，适用于高并发、长上下文、可复用前缀较多的 RAG 推理场景。
