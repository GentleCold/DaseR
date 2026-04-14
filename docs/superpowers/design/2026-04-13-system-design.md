# DaseR System Design

**Date:** 2026-04-13
**Status:** Draft

---

## 1. Overview

DaseR is a **RAG-native KV cache service** for LLM inference. It runs as an independent server process and integrates with vLLM via the `KVConnectorBase_V1` interface.

### Positioning vs LMCache

| Dimension | LMCache | DaseR |
|-----------|---------|-------|
| Retrieval | Token prefix hash (exact) | Pluggable — prefix / semantic / hybrid |
| Storage | One file per KV chunk (safetensors) | Single large XFS file, slot-based ring buffer |
| Async model | threading + asyncio + ThreadPoolExecutor mixed | Unified asyncio + cuFile async |
| GDS | Optional plugin | Primary data path (io_uring fallback) |
| Process model | Embedded library in vLLM | Independent server; vLLM is client |
| Position encoding | Fixed (RoPE baked in) | Explicit position offset, pluggable strategy |
| Multi-instance | Each vLLM owns its cache | Multiple vLLM instances share one DaseR server |

DaseR is not based on LMCache. It can be compared against it for benchmarking but has no code dependency on it.

---

## 2. Architecture

### 2.1 Process Topology

```
╔══════════════════════════════════════════════════════════╗
║  vLLM Worker Process                                     ║
║                                                          ║
║  ┌─────────────────────────────────────────────────┐    ║
║  │  DaserConnector  (KVConnectorBase_V1 impl)       │    ║
║  │                                                  │    ║
║  │  Scheduler side    │  Worker side                │    ║
║  │  ──────────────────────────────────             │    ║
║  │  get_num_new_      │  start_load_kv()            │    ║
║  │  matched_tokens()  │  wait_for_layer_load()      │    ║
║  │  update_state_     │  save_kv_layer()            │    ║
║  │  after_alloc()     │  wait_for_save()            │    ║
║  │                                                  │    ║
║  │  ┌──────────────────────────────────────────┐   │    ║
║  │  │  GDSTransferLayer                         │   │    ║
║  │  │  cuFile async  |  io_uring fallback       │   │    ║
║  │  └──────────────────┬───────────────────────┘   │    ║
║  └─────────────────────┼──────────────────┬─────────┘    ║
║                        │ GDS DMA          │ Unix socket  ║
╚════════════════════════╪══════════════════╪══════════════╝
                         │                  │
                NVMe ◄───┘                  ▼
╔═════════════════════════════════════════════════════╗
║  DaseR Server Process                               ║
║                                                     ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ ║
║  │ Retrieval   │  │   Chunk     │  │  Position   │ ║
║  │ Index       │  │   Manager   │  │  Encoder    │ ║
║  │ (pluggable) │  │ (ring buf)  │  │ (pluggable) │ ║
║  └─────────────┘  └─────────────┘  └─────────────┘ ║
║  ┌─────────────┐  ┌─────────────┐                  ║
║  │  Metadata   │  │   Logger    │                  ║
║  │  Store      │  │  (unified)  │                  ║
║  └─────────────┘  └─────────────┘                  ║
╚═════════════════════════════════════════════════════╝
```

**Why the data plane stays in vLLM's process:**
`cuFileBufRegister` binds GPU memory to the calling process's CUDA context. Cross-process GPU memory access via CUDA IPC adds unacceptable latency for the hot path. Therefore GDS DMA (the data plane) runs inside the vLLM worker process, while index management and retrieval (the control plane) run in the separate DaseR server.

### 2.2 Module Responsibilities

| Module | Process | Responsibility |
|--------|---------|----------------|
| DaserConnector | vLLM | Implements `KVConnectorBase_V1`; dispatches GDS IO; calls DaseR Server over IPC for metadata |
| GDSTransferLayer | vLLM | cuFile async read/write; io_uring fallback; backend selected once at startup |
| RetrievalIndex | DaseR Server | Pluggable retrieval interface; first impl: prefix hash |
| ChunkManager | DaseR Server | Ring buffer slot allocation and eviction |
| MetadataStore | DaseR Server | In-memory index serialized to disk on shutdown; deserialized on startup |
| PositionEncoder | DaseR Server | Records and applies position offsets per chunk; pluggable strategy |
| Logger | Both | Unified structured logging with component tags |

---

## 3. Storage Layer

### 3.1 Files

```
/mnt/xfs/
├── daser.store     # Pure KV data — ring buffer of fixed-size slots
└── daser.index     # Serialized snapshot of in-memory index (msgpack)
```

`daser.store` is pre-allocated with `fallocate` at startup. It never contains metadata — it is a flat array of slots.

On clean shutdown, the in-memory index is serialized to `daser.index`. On startup, if `daser.index` exists it is deserialized to restore ring buffer state. If it does not exist (first run or unclean shutdown), a cold start is performed.

No WAL, no fault tolerance in the current scope.

### 3.2 Slot Layout

```
slot_size = num_kv_heads × head_dim × 2(K+V) × num_layers × block_tokens × dtype_bytes
```

`block_tokens` is aligned to vLLM's block size (default 16). `slot_size` is computed once at startup from model config and stays constant.

A document chunk of `N` tokens occupies `ceil(N / block_tokens)` contiguous slots.

```
daser.store
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ s0   │ s1   │ s2   │ s3   │ s4   │ s5   │ s6   │ s7   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
 ←── chunk A (3 slots) ──→  ←── chunk B (2 slots) ──→
                             ^tail                   ^head

file_offset(slot_id) = slot_id × slot_size          # always aligned
```

### 3.3 Ring Buffer

`head` and `tail` are slot indices. `tail` always points to the start of the oldest chunk.

**Alloc:**
1. If `head + num_slots > total_slots`: mark trailing slots as `SKIP`, wrap `head` to 0.
2. While free slots < `num_slots`: call `evict_oldest()`.
3. Return `start_slot = head`; advance `head += num_slots`.

**Evict oldest:**
1. Look up `slot_map[tail]` to get `chunk_key` and `num_slots`.
2. Remove from MetadataStore index.
3. Advance `tail += num_slots` (handles `SKIP` entries transparently).

**SKIP slots** occupy at most `max_chunk_slots - 1` slots per wrap — negligible waste.

### 3.4 MetadataStore

In-memory structures:

```python
@dataclass
class ChunkMeta:
    chunk_key:   str    # SHA256(token_ids) or document ID
    start_slot:  int
    num_slots:   int
    token_count: int
    pos_offset:  int    # position encoding start index for this chunk
    model_id:    str    # prevents cross-model reuse
    created_at:  float

# Primary index
chunk_index: dict[str, ChunkMeta]       # chunk_key → ChunkMeta

# Reverse index for eviction
slot_map: list[SlotEntry]               # slot_id → {chunk_key, num_slots, type}
```

Serialized with msgpack on shutdown to `daser.index`.

---

## 4. GDS Transfer Layer

### 4.1 Backend Selection

Detected once at startup:

```python
class GDSTransferLayer:
    _backend: CuFileBackend | IOUringBackend  # immutable after init

    def _probe_cufile(self) -> bool:
        try:
            import cufile
            return cufile.is_supported()
        except ImportError:
            return False
```

Backend type is logged at startup: `[GDS] backend=cufile` or `[GDS] backend=io_uring`.

### 4.2 IO Paths

**GDS path (cuFile):**
```
Write: cuFileWriteAsync(fd, gpu_ptr, size, offset)  →  asyncio.Future
Read:  cuFileReadAsync(fd, gpu_ptr, size, offset)   →  asyncio.Future
```
No CPU involvement. DMA goes directly between NVMe and GPU HBM.

**io_uring fallback:**
```
Write: cudaMemcpyAsync(gpu → cpu_bounce, stream)
       await stream_event (cudaStreamAddCallback → asyncio)
       io_uring_prep_write(fd, cpu_bounce, size, offset)
       await io_uring completion

Read:  io_uring_prep_read(fd, cpu_bounce, size, offset)
       await io_uring completion
       cudaMemcpyAsync(cpu_bounce → gpu, stream)
       await stream_event
```

The interface is identical to callers in both cases.

### 4.3 Load Strategy and CUDA Graph Compatibility

#### Current implementation: eager synchronous load in `start_load_kv`

`start_load_kv()` submits all GDS reads for **all layers × all blocks** concurrently
(via a single `asyncio.gather` on the background loop), blocks until every read
completes, then copies all staging buffers into the KV cache on the calling thread
before returning.  `wait_for_layer_load()` is a documented no-op.

This is required for correctness under vLLM's **FULL CUDA graph** mode.

#### Why FULL CUDA graphs break layer-wise loading

vLLM uses two CUDA graph modes:

| Mode | Used for | Python re-executes during replay? |
|------|----------|----------------------------------|
| **PIECEWISE** | prefill (mixed) steps | Attention layers run **eager** — yes |
| **FULL** | pure decode steps | Entire forward pass replayed — **no** |

The `maybe_transfer_kv_layer` decorator (which calls `wait_for_layer_load`) wraps
the attention forward function.  In FULL graph replay Python does **not** re-execute,
so the decorator is never invoked and `wait_for_layer_load` is silently skipped.

This bites when a cache-hit request has **all prompt tokens covered** by the DaseR
cache (token count is an exact multiple of `block_tokens`).  In that case there are
zero new tokens to prefill; vLLM schedules the first step as a pure **decode**, uses
the FULL graph, and `wait_for_layer_load` is never called → KV cache blocks contain
zeros → garbage output.

#### Future optimization: layerwise pipelining on PIECEWISE path

The original layer-wise design allows NVMe reads to overlap with GPU attention:

```
Timeline (PIECEWISE / prefill only):
  NVMe→GPU:  [L0 DMA]─────[L1 DMA]─────[L2 DMA]─────...
  GPU:              [attn L0]     [attn L1]     [attn L2]...
```

This optimization is **valid on the PIECEWISE path** (prefill, some new tokens) but
**invalid on the FULL path** (decode, zero new tokens).  A future implementation can
re-enable it by detecting the path:

```python
# Pseudocode — not yet implemented
if num_new_tokens > 0:          # PIECEWISE path: layer-wise async
    _submit_reads_per_layer()
    # wait_for_layer_load() per layer handles copy
else:                           # FULL graph path: eager synchronous
    _submit_all_reads_and_copy_sync()
```

---

## 5. IPC Protocol (Connector ↔ DaseR Server)

Transport: Unix socket. Serialization: msgpack.

| Message | Direction | Payload |
|---------|-----------|---------|
| `lookup` | C → S | `{tokens, model_id}` → `list[ChunkMeta]` ordered by match quality (for `PrefixHashIndex`: exact prefix match only, longest prefix first) |
| `alloc_chunk` | C → S | `{chunk_key, num_slots, token_count, model_id}` → `{start_slot, offset, pos_offset}` where `pos_offset` is assigned by `PositionEncoder.assign_offset()` |
| `commit_chunk` | C → S | `{chunk_key}` → `ok` |
| `evict_chunk` | C → S | `{chunk_key}` → `ok` |

RTT target: < 0.1 ms (same machine, Unix socket).

---

## 6. Request Lifecycle

```
vLLM Scheduler          DaserConnector            DaseR Server
      │                       │                         │
      │ get_num_new_matched()  │                         │
      │ ─────────────────────►│   lookup(tokens)        │
      │                       │ ───────────────────────►│
      │                       │ ◄───────────────────────│
      │ ◄── matched_tokens ── │   [ChunkMeta list]      │
      │                       │                         │
      │ build_connector_meta()│                         │
      │ ─────────────────────►│                         │
      │                       │                         │
 ─ ─ ─ ─ ─ ─ ─ ─ Worker side ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      │                       │                         │
      │ start_load_kv()       │                         │
      │ ─────────────────────►│  cuFileReadAsync × all  │
      │                       │ ════════════════════►   │
      │                       │   (GDS DMA: NVMe→GPU)   │
      │                       │  await all Futures      │
      │                       │  copy all → KV cache    │
      │ ◄──────────────────── │  (KV cache fully ready) │
      │  [forward pass / CUDA graph replay]             │
      │  wait_for_layer_load() is no-op (see §4.3)      │
      │                       │                         │
      │ save_kv_layer(L0, kv) │  cuFileWriteAsync       │
      │ ─────────────────────►│ ════════════════════►   │
      │       ...             │   (GDS DMA: GPU→NVMe)   │
      │                       │                         │
      │ wait_for_save()       │  await all write Futures│
      │ ─────────────────────►│ ◄════════════════════   │
      │                       │                         │
      │ request_finished()    │  commit_chunk(key)      │
      │ ─────────────────────►│ ───────────────────────►│
      │                       │   [index updated]       │
```

---

## 7. Pluggable Interfaces

### RetrievalIndex (ABC)

```python
class RetrievalIndex(ABC):
    @abstractmethod
    async def lookup(self, tokens: list[int], model_id: str) -> list[ChunkMeta]: ...

    @abstractmethod
    async def insert(self, meta: ChunkMeta) -> None: ...

    @abstractmethod
    async def remove(self, chunk_key: str) -> None: ...
```

First implementation: `PrefixHashIndex` — exact token prefix hash matching.

### PositionEncoder (ABC)

```python
class PositionEncoder(ABC):
    @abstractmethod
    def assign_offset(self, chunk_key: str, token_count: int) -> int:
        """Return the position offset to store for this chunk."""
        ...

    @abstractmethod
    def get_offset(self, meta: ChunkMeta) -> int:
        """Return the position offset to apply when loading this chunk."""
        ...
```

First implementation: `FixedOffsetEncoder` — returns the stored `pos_offset` unchanged.

---

## 8. Logging

Unified logging module at `daser/logging.py`, wrapping Python's standard `logging`.

```python
from daser.logging import init_logger
logger = init_logger(__name__)
```

Component tags: `[GDS]`, `[INDEX]`, `[CHUNK]`, `[IPC]`, `[CONNECTOR]`.

Separate `perf_logger` for latency-sensitive paths: records IO latency, cache hit rate, throughput. Structured JSON format (optional, configurable).

---

## 9. Project Layout

```
DaseR/
├── CLAUDE.md
├── pyproject.toml
├── daser/
│   ├── __init__.py
│   ├── logging.py
│   ├── config.py
│   ├── connector/
│   │   ├── daser_connector.py   # KVConnectorBase_V1 impl
│   │   └── gds_transfer.py      # GDS + io_uring
│   ├── server/
│   │   ├── __main__.py          # server entry point
│   │   ├── ipc_server.py        # Unix socket handler
│   │   ├── chunk_manager.py     # ring buffer
│   │   └── metadata_store.py   # in-memory index + serialization
│   ├── retrieval/
│   │   ├── base.py              # RetrievalIndex ABC
│   │   └── prefix.py            # PrefixHashIndex
│   └── position/
│       ├── base.py              # PositionEncoder ABC
│       └── fixed_offset.py
├── doc/
│   └── design/
│       └── 2026-04-13-system-design.md  # this file
└── tests/

project_doc/                     # private, not in DaseR repo
├── resource.md                  # GPU / NVMe / NUMA topology
├── paths.md                     # key filesystem paths
└── conventions.md               # private team conventions
```

---

## 10. Out of Scope (Current Phase)

- Fault tolerance / crash recovery beyond clean-shutdown serialization
- Multi-node deployment
- Semantic / vector retrieval (interface reserved, not implemented)
- Custom position encoding strategies beyond fixed offset
- Authentication / multi-tenant isolation
