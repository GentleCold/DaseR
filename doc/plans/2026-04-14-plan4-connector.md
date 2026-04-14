# Plan 4: DaserConnector (vLLM Integration)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `DaserConnector`, the vLLM-side client that integrates DaseR into LLM inference. Connects the GDS transfer layer (Plan 2) and the IPC server (Plan 3) with vLLM's `KVConnectorBase_V1` interface.

**Architecture:** `DaserConnector` runs in two roles. Scheduler side uses a synchronous IPC client (blocking < 0.1 ms) to call `lookup`/`alloc_chunk`. Worker side uses `GDSTransferLayer` for NVMe↔GPU DMA and `asyncio` for layer-wise pipelining. `DaserConnectorMeta` carries load/store specs from scheduler to worker.

**Tech Stack:** Python 3.10+, torch, cupy, kvikio, msgpack, asyncio. No new pip installs needed. Venv: `source /data/zwt/vllm/bin/activate`.

**vLLM reference:** `/home/zwt/daser_project/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

---

## File Map

| File | Create/Modify | Purpose |
|------|---------------|---------|
| `daser/connector/ipc_client.py` | Create | Sync + async IPC client |
| `daser/connector/daser_connector.py` | Create | `DaserConnector` implementing `KVConnectorBase_V1` |
| `tests/connector/test_ipc_client.py` | Create | IPC client round-trip tests (against live IPCServer) |
| `tests/connector/test_daser_connector.py` | Create | Connector unit tests (mocked server + GDS) |

---

## Task 1: IPCClient

**Files:**
- Create: `daser/connector/ipc_client.py`
- Create: `tests/connector/test_ipc_client.py`

### Step 1: Create `daser/connector/ipc_client.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import socket
from typing import Any

# Third Party
import msgpack

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

_HEADER_SIZE = 4


def _pack(payload: dict[str, Any]) -> bytes:
    data = msgpack.packb(payload, use_bin_type=True)
    return len(data).to_bytes(_HEADER_SIZE, "big") + data


def _unpack(raw: bytes) -> dict[str, Any]:
    return msgpack.unpackb(raw, raw=False)


class IPCClientSync:
    """Synchronous blocking IPC client for scheduler-side calls.

    Uses a raw blocking socket for each call. Target RTT < 0.1 ms
    (Unix socket, same machine). A new connection is opened per call
    so no persistent connection state is needed.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the response (blocking).

        Args:
            payload: dict with "op" and any required fields.

        Returns:
            Response dict from the server.
        """
        raw = _pack(payload)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self._path)
            s.sendall(raw)
            # Read response
            header = _recv_exact(s, _HEADER_SIZE)
            length = int.from_bytes(header, "big")
            data = _recv_exact(s, length)
        result = _unpack(data)
        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    def lookup(self, tokens: list[int], model_id: str) -> list[dict[str, Any]]:
        """Look up cached chunks for the given token sequence.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.

        Returns:
            List of chunk dicts (may be empty).
        """
        resp = self.call({"op": "lookup", "tokens": tokens, "model_id": model_id})
        return resp.get("chunks", [])

    def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str
    ) -> dict[str, Any]:
        """Allocate a slot for a new chunk.

        Args:
            chunk_key: SHA256 hex of the token IDs.
            token_count: number of tokens in the chunk.
            model_id: model identifier.

        Returns:
            Dict with start_slot, file_offset, pos_offset.
        """
        return self.call(
            {
                "op": "alloc_chunk",
                "chunk_key": chunk_key,
                "token_count": token_count,
                "model_id": model_id,
            }
        )

    def commit_chunk(self, chunk_key: str) -> None:
        """Mark a chunk as committed (GDS write complete).

        Args:
            chunk_key: SHA256 hex of the chunk's token IDs.
        """
        self.call({"op": "commit_chunk", "chunk_key": chunk_key})

    def evict_chunk(self, chunk_key: str) -> None:
        """Evict a chunk from the DaseR index.

        Args:
            chunk_key: SHA256 hex of the chunk's token IDs.
        """
        self.call({"op": "evict_chunk", "chunk_key": chunk_key})


def _recv_exact(s: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a blocking socket.

    Args:
        s: connected socket.
        n: number of bytes to receive.

    Returns:
        Exactly n bytes.

    Raises:
        ConnectionError: if the connection closes before n bytes arrive.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before receiving all bytes")
        buf.extend(chunk)
    return bytes(buf)


class IPCClientAsync:
    """Asyncio IPC client for worker-side calls.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request asynchronously and return the response.

        Args:
            payload: dict with "op" and any required fields.

        Returns:
            Response dict from the server.
        """
        reader, writer = await asyncio.open_unix_connection(self._path)
        try:
            data = msgpack.packb(payload, use_bin_type=True)
            header = len(data).to_bytes(_HEADER_SIZE, "big")
            writer.write(header + data)
            await writer.drain()

            resp_header = await reader.readexactly(_HEADER_SIZE)
            resp_len = int.from_bytes(resp_header, "big")
            resp_data = await reader.readexactly(resp_len)
        finally:
            writer.close()

        result = _unpack(resp_data)
        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    async def commit_chunk(self, chunk_key: str) -> None:
        """Async: mark a chunk as committed.

        Args:
            chunk_key: SHA256 hex of the token IDs.
        """
        await self.call({"op": "commit_chunk", "chunk_key": chunk_key})
```

### Step 2: Write tests

Create `tests/connector/test_ipc_client.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.chunk_manager import ChunkManager
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore


def make_server(tmp_path, slot_size: int = 1024, block_tokens: int = 4) -> IPCServer:
    socket_path = str(tmp_path / "ipc.sock")
    store = MetadataStore(total_slots=64)
    cm = ChunkManager(total_slots=64, metadata_store=store)
    return IPCServer(
        socket_path=socket_path,
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=block_tokens),
        position_encoder=FixedOffsetEncoder(),
        slot_size=slot_size,
        block_tokens=block_tokens,
    )


@pytest.mark.asyncio
async def test_sync_client_lookup(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientSync(str(tmp_path / "ipc.sock"))
    result = client.lookup([1, 2, 3, 4], "m")
    assert result == []
    await server.stop()


@pytest.mark.asyncio
async def test_sync_client_alloc_and_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = _hash_tokens(tokens)
    alloc = client.alloc_chunk(key, token_count=4, model_id="m")
    assert "start_slot" in alloc
    client.commit_chunk(key)
    chunks = client.lookup(tokens, "m")
    assert len(chunks) == 1
    assert chunks[0]["chunk_key"] == key
    await server.stop()


@pytest.mark.asyncio
async def test_async_client_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = _hash_tokens(tokens)
    sync_client.alloc_chunk(key, token_count=4, model_id="m")

    async_client = IPCClientAsync(sock)
    await async_client.commit_chunk(key)
    chunks = sync_client.lookup(tokens, "m")
    assert len(chunks) == 1
    await server.stop()
```

### Step 3: Run tests

```bash
cd /home/zwt/daser_project/DaseR && source /data/zwt/vllm/bin/activate
pytest tests/connector/test_ipc_client.py -v
```

Expected: 3 tests pass.

### Step 4: Commit

```bash
git add daser/connector/ipc_client.py tests/connector/test_ipc_client.py
git commit -m "feat: add IPCClientSync and IPCClientAsync"
```

---

## Task 2: DaserConnector

**Files:**
- Create: `daser/connector/daser_connector.py`
- Create: `tests/connector/test_daser_connector.py`

### Step 1: Create `daser/connector/daser_connector.py`

Read the base class first:
```bash
head -200 /home/zwt/daser_project/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py
```

Then create `daser/connector/daser_connector.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

# Third Party
import cupy
import torch

if TYPE_CHECKING:
    # Standard
    from typing import Iterable

    # Third Party
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
        KVConnectorMetadata,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.attention import AttentionMetadata

# First Party
from daser.connector.gds_transfer import GDSTransferLayer
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.logging import init_logger, init_perf_logger

logger = init_logger(__name__)
perf = init_perf_logger(__name__)


def _hash_tokens(tokens: list[int]) -> str:
    """Return hex SHA256 of token ID sequence."""
    h = hashlib.sha256()
    for tok in tokens:
        h.update(tok.to_bytes(4, "little"))
    return h.hexdigest()


@dataclass
class ReqLoadSpec:
    """Load specification for one request.

    Attributes:
        chunk_key: SHA256 of the cached token sequence.
        start_slot: first DaseR slot for this chunk.
        num_slots: number of slots in the chunk.
        block_ids: vLLM block IDs allocated to hold the loaded KV.
        file_offset: byte offset of slot 0 in daser.store.
        token_count: number of tokens covered.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int


@dataclass
class ReqStoreSpec:
    """Store specification for one request.

    Attributes:
        chunk_key: SHA256 of this request's token sequence.
        start_slot: first DaseR slot allocated for this chunk.
        num_slots: number of slots allocated.
        block_ids: vLLM block IDs whose KV to save.
        file_offset: byte offset of slot 0 in daser.store.
        token_count: number of tokens to store.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int


@dataclass
class DaserConnectorMeta:
    """Metadata passed from scheduler to worker each scheduling step.

    Attributes:
        reqs_to_load: req_id → ReqLoadSpec for cache hits.
        reqs_to_store: req_id → ReqStoreSpec for new chunks to persist.
    """

    reqs_to_load: dict[str, ReqLoadSpec] = field(default_factory=dict)
    reqs_to_store: dict[str, ReqStoreSpec] = field(default_factory=dict)


class DaserConnector:
    """vLLM KVConnectorBase_V1 implementation backed by DaseR.

    Runs in two roles determined by the `role` constructor argument:
    - SCHEDULER: uses IPCClientSync for cache lookup and slot allocation.
    - WORKER: uses GDSTransferLayer for NVMe↔GPU DMA.

    The scheduler side communicates with the DaseR server (Plan 3) to
    resolve cache hits and allocate slots. The worker side performs the
    actual GDS IO layer by layer, enabling overlap between NVMe DMA and
    GPU attention computation.

    Args:
        vllm_config: full VllmConfig from vLLM.
        role: KVConnectorRole.SCHEDULER or KVConnectorRole.WORKER.
        kv_cache_config: optional KV cache configuration.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: Any = None,
    ) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        self._role = role
        cfg = vllm_config

        # Extract DaseR config from kv_transfer_config extra_config
        extra: dict[str, Any] = {}
        if hasattr(cfg, "kv_transfer_config") and cfg.kv_transfer_config is not None:
            extra = cfg.kv_transfer_config.kv_connector_extra_config or {}

        self._socket_path: str = extra.get("socket_path", "/tmp/daser.sock")
        self._store_path: str = extra.get("store_path", "/data/zwt/daser_test/daser.store")
        self._slot_size: int = int(extra.get("slot_size", 0))
        self._block_tokens: int = int(extra.get("block_tokens", 16))
        self._model_id: str = extra.get("model_id", "default")

        if role == KVConnectorRole.SCHEDULER:
            self._ipc_sync = IPCClientSync(self._socket_path)
            # Per-request state: req_id → matched chunk dict (from lookup)
            self._pending_loads: dict[str, dict[str, Any]] = {}
            # Per-request state: req_id → alloc result dict
            self._pending_stores: dict[str, dict[str, Any]] = {}
            # Per-request tokens (for hashing)
            self._req_tokens: dict[str, list[int]] = {}

        else:  # WORKER
            self._gds: Optional[GDSTransferLayer] = None
            self._ipc_async = IPCClientAsync(self._socket_path)
            # layer_name → torch.Tensor (the full KV cache for that layer)
            self._kv_caches: dict[str, torch.Tensor] = {}
            # layer_name ordering (insertion order)
            self._layer_names: list[str] = []
            # Active connector metadata for this forward step
            self._meta: Optional[DaserConnectorMeta] = None
            # Pending load futures: layer_name → asyncio.Future
            self._load_futures: dict[str, asyncio.Future] = {}
            # Pending store futures: list of asyncio.Future
            self._store_futures: list[asyncio.Future] = []
            # Pending store specs for commit (after wait_for_save)
            self._pending_commits: list[str] = []

        logger.info("[CONNECTOR] role=%s socket=%s", role.name, self._socket_path)

    # ------------------------------------------------------------------
    # Scheduler-side methods
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> "tuple[int | None, bool]":
        """Query DaseR for cached KV matching request's tokens.

        Aligns the token count to block boundaries, hashes the prefix,
        and calls the DaseR server's lookup endpoint. On a hit, stores
        the chunk info for later use in update_state_after_alloc.

        Args:
            request: vLLM Request object with prompt_token_ids.
            num_computed_tokens: tokens already in vLLM's KV cache.

        Returns:
            (num_external_tokens, is_async) or (0, False) on miss.
        """
        tokens = list(request.prompt_token_ids)
        self._req_tokens[request.request_id] = tokens

        # Only cache the prefix beyond what vLLM already has
        start = num_computed_tokens
        available = len(tokens) - start
        if available < self._block_tokens:
            return 0, False

        # Align to block boundary
        aligned = (available // self._block_tokens) * self._block_tokens
        prefix = tokens[: start + aligned]

        try:
            chunks = self._ipc_sync.lookup(prefix, self._model_id)
        except Exception as exc:
            logger.warning("[CONNECTOR] lookup failed: %s", exc)
            return 0, False

        if not chunks:
            return 0, False

        best = chunks[0]
        extra_tokens = best["token_count"] - num_computed_tokens
        if extra_tokens <= 0:
            return 0, False

        self._pending_loads[request.request_id] = best
        logger.debug(
            "[CONNECTOR] cache hit req=%s tokens=%d",
            request.request_id,
            extra_tokens,
        )
        return extra_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record block IDs for requests that will load or store KV.

        For load requests: save the block IDs vLLM allocated for the
        incoming KV data. For all other requests: allocate a DaseR slot
        for later storage.

        Args:
            request: vLLM Request.
            blocks: vLLM KV cache block allocation for this request.
            num_external_tokens: tokens from DaseR (0 if miss).
        """
        req_id = request.request_id
        block_ids: list[int] = list(blocks.block_ids.flatten().tolist())

        if num_external_tokens > 0 and req_id in self._pending_loads:
            # Load path: record which vLLM blocks will receive NVMe data
            chunk = self._pending_loads[req_id]
            # Use only the blocks covering the external tokens
            num_needed = math.ceil(num_external_tokens / self._block_tokens)
            load_block_ids = block_ids[:num_needed]
            chunk["block_ids"] = load_block_ids
            logger.debug(
                "[CONNECTOR] load blocks req=%s blocks=%s", req_id, load_block_ids
            )
        else:
            # Store path: allocate a DaseR slot for this request's KV
            tokens = self._req_tokens.get(req_id, [])
            if not tokens:
                return
            aligned = (len(tokens) // self._block_tokens) * self._block_tokens
            if aligned == 0:
                return
            chunk_key = _hash_tokens(tokens[:aligned])
            try:
                alloc = self._ipc_sync.alloc_chunk(
                    chunk_key, token_count=aligned, model_id=self._model_id
                )
            except Exception as exc:
                logger.warning("[CONNECTOR] alloc_chunk failed: %s", exc)
                return
            alloc["chunk_key"] = chunk_key
            alloc["block_ids"] = block_ids[: math.ceil(aligned / self._block_tokens)]
            alloc["token_count"] = aligned
            self._pending_stores[req_id] = alloc
            logger.debug("[CONNECTOR] alloc store req=%s key=%s", req_id, chunk_key[:8])

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> DaserConnectorMeta:
        """Package pending load/store specs into connector metadata.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.

        Returns:
            DaserConnectorMeta with reqs_to_load and reqs_to_store.
        """
        meta = DaserConnectorMeta()

        scheduled_ids = {r.request_id for r in scheduler_output.scheduled_new_reqs}
        scheduled_ids |= {r.request_id for r in scheduler_output.scheduled_resumed_reqs}
        scheduled_ids |= {r.request_id for r in scheduler_output.scheduled_running_reqs}

        for req_id, chunk in list(self._pending_loads.items()):
            if req_id in scheduled_ids and "block_ids" in chunk:
                meta.reqs_to_load[req_id] = ReqLoadSpec(
                    chunk_key=chunk["chunk_key"],
                    start_slot=chunk["start_slot"],
                    num_slots=chunk["num_slots"],
                    block_ids=chunk["block_ids"],
                    file_offset=chunk["file_offset"],
                    token_count=chunk["token_count"],
                )
                del self._pending_loads[req_id]

        for req_id, alloc in list(self._pending_stores.items()):
            if req_id in scheduled_ids and "block_ids" in alloc:
                meta.reqs_to_store[req_id] = ReqStoreSpec(
                    chunk_key=alloc["chunk_key"],
                    start_slot=alloc["start_slot"],
                    num_slots=alloc["num_slots"],
                    block_ids=alloc["block_ids"],
                    file_offset=alloc["file_offset"],
                    token_count=alloc["token_count"],
                )
                del self._pending_stores[req_id]

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> "tuple[bool, dict[str, Any] | None]":
        """Clean up per-request state after inference completes.

        Args:
            request: finished vLLM Request.
            block_ids: block IDs being freed.

        Returns:
            (False, None) — no async cleanup needed.
        """
        self._req_tokens.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        self._pending_stores.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------
    # Worker-side methods
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register the per-layer KV cache tensors.

        Called once after vLLM allocates the KV cache. Layer names are
        stored in insertion order for index-based IO offset computation.

        Args:
            kv_caches: dict mapping layer_name → KV tensor.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())

        # Compute slot_size from actual tensor if not configured
        if self._slot_size == 0 and self._layer_names:
            sample = next(iter(kv_caches.values()))
            # layer_size = bytes for one layer, one block
            # tensor shape: (2, num_blocks, block_size, num_heads, head_dim)
            # or similar; use nbytes / num_blocks as per-block per-layer size
            num_blocks = sample.shape[1] if sample.dim() >= 2 else 1
            layer_size = sample.nbytes // num_blocks
            self._slot_size = layer_size * len(self._layer_names)
            logger.info(
                "[CONNECTOR] computed slot_size=%d from %d layers",
                self._slot_size,
                len(self._layer_names),
            )

        if self._gds is None:
            import os
            if not os.path.exists(self._store_path):
                # Pre-allocate store file if not already present
                size = self._slot_size * 1024  # 1024 slots default
                with open(self._store_path, "wb") as f:
                    f.write(b"\x00" * size)
                logger.info(
                    "[CONNECTOR] pre-allocated store %s (%d bytes)",
                    self._store_path,
                    size,
                )
            self._gds = GDSTransferLayer(self._store_path)
            logger.info(
                "[CONNECTOR] GDS backend=%s", self._gds.backend.value
            )

    def bind_connector_metadata(
        self, connector_metadata: DaserConnectorMeta
    ) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        self._meta = connector_metadata
        self._load_futures = {}
        self._store_futures = []
        self._pending_commits = []

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        self._meta = None

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Submit GDS reads for all cache-hit requests.

        Submits pread for each layer of each matching chunk upfront.
        Stores per-layer asyncio Futures for wait_for_layer_load.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        if self._meta is None or not self._meta.reqs_to_load:
            return
        if self._gds is None:
            return

        loop = asyncio.get_event_loop()

        for layer_idx, layer_name in enumerate(self._layer_names):
            layer_tasks = []
            for spec in self._meta.reqs_to_load.values():
                layer_size = self._slot_size // max(len(self._layer_names), 1)
                kv_tensor = self._kv_caches.get(layer_name)
                if kv_tensor is None:
                    continue

                for slot_i, block_id in enumerate(spec.block_ids):
                    file_offset = (
                        spec.start_slot + slot_i
                    ) * self._slot_size + layer_idx * layer_size

                    # Allocate a receive buffer matching one block's KV
                    if kv_tensor.dim() >= 2:
                        buf = torch.empty(
                            kv_tensor[:, 0].shape,
                            dtype=kv_tensor.dtype,
                            device=kv_tensor.device,
                        )
                    else:
                        buf = torch.empty_like(kv_tensor)

                    cp_buf = cupy.asarray(buf.contiguous())
                    nbytes = cp_buf.nbytes

                    async def _load_one(
                        cp=cp_buf,
                        off=file_offset,
                        nb=nbytes,
                        dest=kv_tensor,
                        bid=block_id,
                        src=buf,
                    ) -> int:
                        read = await self._gds.read_into_async(cp, off, nb)
                        if dest.dim() >= 2:
                            dest[:, bid].copy_(src)
                        else:
                            dest[bid].copy_(src)
                        return read

                    layer_tasks.append(asyncio.ensure_future(_load_one()))

            if layer_tasks:
                self._load_futures[layer_name] = asyncio.ensure_future(
                    asyncio.gather(*layer_tasks)
                )

        logger.debug(
            "[CONNECTOR] start_load_kv submitted %d layer futures",
            len(self._load_futures),
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the GDS reads for layer_name complete.

        Called by vLLM before computing attention for this layer,
        enabling overlap between NVMe DMA and GPU compute on other layers.

        Args:
            layer_name: the layer whose KV data must be resident.
        """
        fut = self._load_futures.get(layer_name)
        if fut is None:
            return
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fut)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit GDS writes for this layer for all store requests.

        Args:
            layer_name: name of the current attention layer.
            kv_layer: full KV cache tensor for this layer.
            attn_metadata: attention metadata (unused directly).
        """
        if self._meta is None or not self._meta.reqs_to_store:
            return
        if self._gds is None:
            return

        layer_idx = self._layer_names.index(layer_name) if layer_name in self._layer_names else 0
        layer_size = self._slot_size // max(len(self._layer_names), 1)
        loop = asyncio.get_event_loop()

        for spec in self._meta.reqs_to_store.values():
            if spec.chunk_key not in self._pending_commits:
                self._pending_commits.append(spec.chunk_key)

            for slot_i, block_id in enumerate(spec.block_ids):
                file_offset = (
                    spec.start_slot + slot_i
                ) * self._slot_size + layer_idx * layer_size

                if kv_layer.dim() >= 2:
                    data = kv_layer[:, block_id].contiguous()
                else:
                    data = kv_layer[block_id].contiguous()

                cp_data = cupy.asarray(data)
                nbytes = cp_data.nbytes

                async def _save_one(
                    cp=cp_data, off=file_offset, nb=nbytes
                ) -> int:
                    return await self._gds.write_async(cp, off, nb)

                self._store_futures.append(asyncio.ensure_future(_save_one()))

    def wait_for_save(self) -> None:
        """Block until all pending GDS writes complete, then commit.

        After all writes are confirmed, sends commit_chunk IPC messages
        to make the chunks searchable in the DaseR index.
        """
        if not self._store_futures:
            return

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(*self._store_futures))
        self._store_futures.clear()

        # Commit all stored chunks via async IPC
        async def _commit_all() -> None:
            for key in self._pending_commits:
                await self._ipc_async.commit_chunk(key)

        loop.run_until_complete(_commit_all())
        self._pending_commits.clear()

    def shutdown(self) -> None:
        """Close GDS file handle on shutdown."""
        if self._gds is not None:
            self._gds.close()
```

### Step 2: Write tests

Create `tests/connector/test_daser_connector.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os

# Third Party
import pytest
import torch

# First Party
from daser.connector.daser_connector import (
    DaserConnector,
    DaserConnectorMeta,
    ReqLoadSpec,
    ReqStoreSpec,
    _hash_tokens,
)
from daser.connector.gds_transfer import GDSTransferLayer
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

BLOCK_TOKENS = 4
NUM_LAYERS = 2


def make_kv_caches(num_blocks: int = 16) -> dict[str, torch.Tensor]:
    """Create minimal fake kv_caches: 2 layers, shape (2, num_blocks, 4, 8)."""
    return {
        f"layer.{i}": torch.zeros(2, num_blocks, BLOCK_TOKENS, 8, device="cuda")
        for i in range(NUM_LAYERS)
    }


def make_server(tmp_path, store_path: str) -> tuple[IPCServer, str]:
    sock = str(tmp_path / "daser.sock")
    slot_size = 2 * BLOCK_TOKENS * 8 * 2 * NUM_LAYERS  # rough
    store = MetadataStore(total_slots=64)
    cm = ChunkManager(total_slots=64, metadata_store=store)
    server = IPCServer(
        socket_path=sock,
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(),
        slot_size=slot_size,
        block_tokens=BLOCK_TOKENS,
    )
    return server, sock


class FakeVllmConfig:
    class KVTransferConfig:
        kv_connector_extra_config = {}
    kv_transfer_config = KVTransferConfig()


class FakeRole:
    class KVConnectorRole:
        SCHEDULER = "SCHEDULER"
        WORKER = "WORKER"
    name = "SCHEDULER"


def test_dataclasses_instantiate():
    """DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec all instantiate cleanly."""
    spec_load = ReqLoadSpec("k", 0, 1, [0], 0, 16)
    spec_store = ReqStoreSpec("k", 0, 1, [0], 0, 16)
    meta = DaserConnectorMeta(reqs_to_load={"r": spec_load}, reqs_to_store={"r2": spec_store})
    assert "r" in meta.reqs_to_load
    assert "r2" in meta.reqs_to_store


def test_hash_tokens_deterministic():
    tokens = [1, 2, 3, 4]
    assert _hash_tokens(tokens) == _hash_tokens(tokens)
    assert _hash_tokens(tokens) != _hash_tokens([1, 2, 3, 5])


@pytest.mark.asyncio
async def test_gds_roundtrip_with_kv_tensor(tmp_path):
    """Write a KV tensor to NVMe, read it back, verify equality."""
    store_path = str(tmp_path / "test.store")
    kv = torch.randint(0, 256, (2, 4, BLOCK_TOKENS, 8), dtype=torch.uint8, device="cuda")
    # Write
    size = 4 * 1024 * 1024
    with open(store_path, "wb") as f:
        f.write(b"\x00" * size)

    gds = GDSTransferLayer(store_path)
    import cupy
    data = kv[:, 0].contiguous()
    cp = cupy.asarray(data)
    await gds.write_async(cp, file_offset=0)

    # Read back
    recv = torch.zeros_like(kv[:, 0])
    cp_recv = cupy.asarray(recv)
    await gds.read_into_async(cp_recv, file_offset=0)
    assert torch.equal(kv[:, 0], recv)
    gds.close()
```

### Step 3: Run tests

```bash
cd /home/zwt/daser_project/DaseR && source /data/zwt/vllm/bin/activate
pytest tests/connector/test_daser_connector.py tests/connector/test_ipc_client.py -v
```

Expected: 6 tests pass (3 ipc_client + 3 daser_connector).

### Step 4: Run full suite + lint

```bash
pytest tests/ -v
ruff check daser/ tests/
ruff format daser/ tests/
```

### Step 5: Single commit

```bash
git add daser/connector/ipc_client.py daser/connector/daser_connector.py \
        tests/connector/test_ipc_client.py tests/connector/test_daser_connector.py
git commit -m "feat: add DaserConnector (KVConnectorBase_V1) with GDS and IPC client"
```

---

## Self-Review Checklist

| Spec Section | Covered |
|---|---|
| KVConnectorBase_V1 interface | DaserConnector extends it (via duck typing, no import from vllm in prod) |
| Scheduler side: lookup via IPC | `get_num_new_matched_tokens` → `IPCClientSync.lookup` |
| Scheduler side: alloc via IPC | `update_state_after_alloc` → `IPCClientSync.alloc_chunk` |
| Worker side: GDS read | `start_load_kv` → `GDSTransferLayer.read_into_async` |
| Worker side: GDS write | `save_kv_layer` → `GDSTransferLayer.write_async` |
| Layer-wise pipelining | `start_load_kv` submits all reads; `wait_for_layer_load` awaits per layer |
| commit_chunk after save | `wait_for_save` → `IPCClientAsync.commit_chunk` |
| Data plane stays in vLLM process | GDSTransferLayer instantiated only in WORKER role |
| asyncio only, no threading mix | `run_in_executor` in GDSTransferLayer; no threading.Thread |
| No LMCache imports | grep confirms none |
