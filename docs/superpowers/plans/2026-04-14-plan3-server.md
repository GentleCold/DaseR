# Plan 3: DaseR Server

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the DaseR server process: pluggable retrieval index, position encoder, Unix socket IPC. Wires together ChunkManager + MetadataStore from Plan 1 with lookup and allocation over IPC.

**Architecture:** `ipc_server.py` runs an asyncio Unix socket server. Each connection dispatches msgpack messages to handlers that call `ChunkManager`, `RetrievalIndex`, and `PositionEncoder`. `PrefixHashIndex` stores chunks by SHA256 of their token_ids. `FixedOffsetEncoder` returns the stored pos_offset unchanged.

**Tech Stack:** Python 3.10+, asyncio, msgpack, hashlib, pytest, pytest-asyncio. Venv: `source /data/zwt/vllm/bin/activate`.

---

## File Map

| File | Create/Modify | Purpose |
|------|---------------|---------|
| `daser/retrieval/__init__.py` | Create | Package marker |
| `daser/retrieval/base.py` | Create | `RetrievalIndex` ABC |
| `daser/retrieval/prefix.py` | Create | `PrefixHashIndex` |
| `daser/position/__init__.py` | Create | Package marker |
| `daser/position/base.py` | Create | `PositionEncoder` ABC |
| `daser/position/fixed_offset.py` | Create | `FixedOffsetEncoder` |
| `daser/server/ipc_server.py` | Create | `IPCServer` — asyncio Unix socket handler |
| `daser/server/__main__.py` | Create | Server entry point |
| `tests/retrieval/__init__.py` | Create | Package marker |
| `tests/retrieval/test_prefix_index.py` | Create | PrefixHashIndex tests |
| `tests/position/__init__.py` | Create | Package marker |
| `tests/position/test_fixed_offset.py` | Create | FixedOffsetEncoder tests |
| `tests/server/test_ipc_server.py` | Create | IPC round-trip tests |

---

## Task 1: RetrievalIndex ABC + PrefixHashIndex

**Files:**
- Create: `daser/retrieval/__init__.py`
- Create: `daser/retrieval/base.py`
- Create: `daser/retrieval/prefix.py`
- Create: `tests/retrieval/__init__.py`
- Create: `tests/retrieval/test_prefix_index.py`

### Step 1: Create package markers

`daser/retrieval/__init__.py`:
```python
# SPDX-License-Identifier: Apache-2.0
```

`tests/retrieval/__init__.py`: empty file.

### Step 2: Create `daser/retrieval/base.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod

# First Party
from daser.server.metadata_store import ChunkMeta


class RetrievalIndex(ABC):
    """Pluggable retrieval interface for DaseR's KV cache index.

    Implementations map token sequences to stored ChunkMeta objects.
    The first implementation is PrefixHashIndex (exact prefix hash matching).
    Future implementations may use vector similarity or hybrid strategies.
    """

    @abstractmethod
    async def lookup(self, tokens: list[int], model_id: str) -> list[ChunkMeta]:
        """Find cached chunks matching the given token sequence.

        Args:
            tokens: full token sequence for the request.
            model_id: model identifier; chunks with a different model_id
                      must not be returned.

        Returns:
            List of matching ChunkMeta, ordered by match quality
            (longest / best match first). May be empty.
        """
        ...

    @abstractmethod
    async def insert(self, meta: ChunkMeta) -> None:
        """Add a committed chunk to the retrieval index.

        Args:
            meta: ChunkMeta describing the stored chunk.
        """
        ...

    @abstractmethod
    async def remove(self, chunk_key: str) -> None:
        """Remove an evicted chunk from the retrieval index.

        Args:
            chunk_key: key of the chunk to remove.
        """
        ...
```

### Step 3: Create `daser/retrieval/prefix.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import hashlib
from typing import Optional

# First Party
from daser.logging import init_logger
from daser.retrieval.base import RetrievalIndex
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


def _hash_tokens(tokens: list[int]) -> str:
    """Return a hex SHA256 of the token ID sequence.

    Args:
        tokens: list of integer token IDs.

    Returns:
        64-character hex string.
    """
    h = hashlib.sha256()
    for tok in tokens:
        h.update(tok.to_bytes(4, "little"))
    return h.hexdigest()


class PrefixHashIndex(RetrievalIndex):
    """Exact token-prefix hash retrieval index.

    Stores chunks indexed by SHA256(token_ids). For a lookup query,
    iterates over prefix lengths (at block_token boundaries) from
    longest to shortest and returns the first (longest) match.

    Args:
        block_tokens: vLLM block size in tokens (default 16). Prefix
                      lengths are quantised to multiples of this value.
    """

    def __init__(self, block_tokens: int = 16) -> None:
        self._block_tokens = block_tokens
        # hash → ChunkMeta
        self._index: dict[str, ChunkMeta] = {}

    async def lookup(self, tokens: list[int], model_id: str) -> list[ChunkMeta]:
        """Return the longest cached prefix that matches tokens.

        Iterates prefix lengths from len(tokens) down to block_tokens
        in steps of block_tokens, hashing each prefix. Returns the
        first hit (longest match) or an empty list.

        Args:
            tokens: full token sequence to match against.
            model_id: only chunks with this model_id are returned.

        Returns:
            List with at most one ChunkMeta (the longest prefix match).
        """
        n = len(tokens)
        # Snap n down to nearest block boundary
        n = (n // self._block_tokens) * self._block_tokens
        while n >= self._block_tokens:
            key = _hash_tokens(tokens[:n])
            meta = self._index.get(key)
            if meta is not None and meta.model_id == model_id:
                logger.debug(
                    "[INDEX] prefix hit key=%s matched=%d tokens", key[:8], n
                )
                return [meta]
            n -= self._block_tokens
        return []

    async def insert(self, meta: ChunkMeta) -> None:
        """Insert a chunk into the prefix index.

        Args:
            meta: ChunkMeta with chunk_key = SHA256(token_ids).
        """
        self._index[meta.chunk_key] = meta
        logger.debug("[INDEX] insert chunk_key=%s", meta.chunk_key[:8])

    async def remove(self, chunk_key: str) -> None:
        """Remove a chunk from the prefix index.

        Args:
            chunk_key: key to remove; silently ignored if not present.
        """
        self._index.pop(chunk_key, None)
        logger.debug("[INDEX] remove chunk_key=%s", chunk_key[:8])
```

### Step 4: Write tests

Create `tests/retrieval/test_prefix_index.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import time

# Third Party
import pytest

# First Party
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.metadata_store import ChunkMeta


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_meta(tokens: list[int], start: int = 0, num: int = 1) -> ChunkMeta:
    key = _hash_tokens(tokens)
    return ChunkMeta(
        chunk_key=key,
        start_slot=start,
        num_slots=num,
        token_count=len(tokens),
        pos_offset=0,
        model_id="m",
        created_at=time.time(),
    )


def test_insert_and_exact_lookup():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4]
    meta = make_meta(tokens)
    _run(idx.insert(meta))
    result = _run(idx.lookup(tokens, "m"))
    assert len(result) == 1
    assert result[0].chunk_key == meta.chunk_key


def test_lookup_longer_query_finds_prefix():
    idx = PrefixHashIndex(block_tokens=4)
    # Store 4-token chunk; query with 8 tokens — should still find 4-token match
    tokens4 = [1, 2, 3, 4]
    meta = make_meta(tokens4)
    _run(idx.insert(meta))
    result = _run(idx.lookup([1, 2, 3, 4, 5, 6, 7, 8], "m"))
    assert len(result) == 1
    assert result[0].chunk_key == meta.chunk_key


def test_lookup_miss_returns_empty():
    idx = PrefixHashIndex(block_tokens=4)
    result = _run(idx.lookup([1, 2, 3, 4], "m"))
    assert result == []


def test_remove():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4]
    meta = make_meta(tokens)
    _run(idx.insert(meta))
    _run(idx.remove(meta.chunk_key))
    result = _run(idx.lookup(tokens, "m"))
    assert result == []


def test_model_id_isolation():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4]
    meta = make_meta(tokens)
    _run(idx.insert(meta))
    # Same tokens but different model_id — should not match
    result = _run(idx.lookup(tokens, "other-model"))
    assert result == []
```

### Step 5: Run tests

```bash
cd /home/zwt/daser_project/DaseR && source /data/zwt/vllm/bin/activate
pytest tests/retrieval/test_prefix_index.py -v
```

Expected: 5 tests pass.

### Step 6: Commit

```bash
git add daser/retrieval/ tests/retrieval/
git commit -m "feat: add RetrievalIndex ABC and PrefixHashIndex"
```

---

## Task 2: PositionEncoder ABC + FixedOffsetEncoder

**Files:**
- Create: `daser/position/__init__.py`
- Create: `daser/position/base.py`
- Create: `daser/position/fixed_offset.py`
- Create: `tests/position/__init__.py`
- Create: `tests/position/test_fixed_offset.py`

### Step 1: Create package markers

`daser/position/__init__.py`:
```python
# SPDX-License-Identifier: Apache-2.0
```

`tests/position/__init__.py`: empty file.

### Step 2: Create `daser/position/base.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod

# First Party
from daser.server.metadata_store import ChunkMeta


class PositionEncoder(ABC):
    """Pluggable position encoding strategy for DaseR KV chunks.

    Manages position offsets so that KV computed at one position can be
    reused when the chunk is loaded at a (potentially different) position.

    The first implementation is FixedOffsetEncoder, which stores the
    position offset at insert time and returns it unchanged at load time.
    Future implementations may support dynamic position remapping.
    """

    @abstractmethod
    def assign_offset(self, chunk_key: str, token_count: int) -> int:
        """Return the position offset to store for this chunk.

        Called at alloc_chunk time, before the chunk is written.

        Args:
            chunk_key: unique key for the chunk being allocated.
            token_count: number of tokens in the chunk.

        Returns:
            Position offset (first position ID) to record in ChunkMeta.
        """
        ...

    @abstractmethod
    def get_offset(self, meta: ChunkMeta) -> int:
        """Return the position offset to apply when loading this chunk.

        Called at load time, after the chunk key is resolved.

        Args:
            meta: ChunkMeta of the chunk being loaded.

        Returns:
            Position offset to use when inserting KV into the model.
        """
        ...
```

### Step 3: Create `daser/position/fixed_offset.py`

```python
# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.logging import init_logger
from daser.position.base import PositionEncoder
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


class FixedOffsetEncoder(PositionEncoder):
    """Position encoder that stores and returns a fixed position offset.

    The offset is assigned at construction time and used for every chunk.
    This is the simplest strategy: all chunks start at the same position,
    which is appropriate when RoPE is re-applied at load time or when
    chunks are always prefixed at position 0.

    Args:
        fixed_offset: position offset assigned to every chunk (default 0).
    """

    def __init__(self, fixed_offset: int = 0) -> None:
        self._offset = fixed_offset

    def assign_offset(self, chunk_key: str, token_count: int) -> int:
        """Return the fixed offset for any new chunk.

        Args:
            chunk_key: not used; present for interface compatibility.
            token_count: not used; present for interface compatibility.

        Returns:
            The fixed offset set at construction time.
        """
        logger.debug(
            "[INDEX] assign_offset chunk_key=%s offset=%d", chunk_key[:8], self._offset
        )
        return self._offset

    def get_offset(self, meta: ChunkMeta) -> int:
        """Return the stored pos_offset from ChunkMeta.

        Args:
            meta: ChunkMeta whose pos_offset was set by assign_offset.

        Returns:
            meta.pos_offset unchanged.
        """
        return meta.pos_offset
```

### Step 4: Write tests

Create `tests/position/test_fixed_offset.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import time

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.server.metadata_store import ChunkMeta


def _meta(pos_offset: int) -> ChunkMeta:
    return ChunkMeta(
        chunk_key="test",
        start_slot=0,
        num_slots=1,
        token_count=16,
        pos_offset=pos_offset,
        model_id="m",
        created_at=time.time(),
    )


def test_default_offset_is_zero():
    enc = FixedOffsetEncoder()
    assert enc.assign_offset("key", 16) == 0


def test_custom_fixed_offset():
    enc = FixedOffsetEncoder(fixed_offset=512)
    assert enc.assign_offset("key", 16) == 512


def test_get_offset_returns_meta_pos_offset():
    enc = FixedOffsetEncoder(fixed_offset=0)
    meta = _meta(pos_offset=128)
    assert enc.get_offset(meta) == 128
```

### Step 5: Run tests

```bash
cd /home/zwt/daser_project/DaseR && source /data/zwt/vllm/bin/activate
pytest tests/position/test_fixed_offset.py -v
```

Expected: 3 tests pass.

### Step 6: Commit

```bash
git add daser/position/ tests/position/
git commit -m "feat: add PositionEncoder ABC and FixedOffsetEncoder"
```

---

## Task 3: IPCServer

**Files:**
- Create: `daser/server/ipc_server.py`
- Create: `daser/server/__main__.py`
- Create: `tests/server/test_ipc_server.py`

### Step 1: Create `daser/server/ipc_server.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import math
import os
from typing import Any

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.position.base import PositionEncoder
from daser.retrieval.base import RetrievalIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)

# Wire size prefix: 4-byte big-endian uint32 before each msgpack frame
_HEADER_SIZE = 4


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one length-prefixed msgpack frame from reader.

    Args:
        reader: asyncio stream reader.

    Returns:
        Decoded dict.
    """
    header = await reader.readexactly(_HEADER_SIZE)
    length = int.from_bytes(header, "big")
    data = await reader.readexactly(length)
    return msgpack.unpackb(data, raw=False)


async def _write_frame(
    writer: asyncio.StreamWriter, payload: dict[str, Any]
) -> None:
    """Write one length-prefixed msgpack frame to writer.

    Args:
        writer: asyncio stream writer.
        payload: dict to encode and send.
    """
    data = msgpack.packb(payload, use_bin_type=True)
    header = len(data).to_bytes(_HEADER_SIZE, "big")
    writer.write(header + data)
    await writer.drain()


class IPCServer:
    """Asyncio Unix socket server for DaseR's control plane.

    Handles four message types from DaserConnector:
    - lookup:       {tokens, model_id} → {chunks: list[dict]}
    - alloc_chunk:  {chunk_key, token_count, model_id} → {start_slot, file_offset, pos_offset}
    - commit_chunk: {chunk_key} → {ok: true}
    - evict_chunk:  {chunk_key} → {ok: true}

    Wire protocol: 4-byte big-endian length prefix + msgpack body.

    Args:
        socket_path: Unix socket path.
        chunk_manager: ChunkManager for ring buffer alloc/evict.
        retrieval_index: RetrievalIndex for lookup/insert/remove.
        position_encoder: PositionEncoder for offset assignment.
        slot_size: bytes per slot (for computing file_offset).
        block_tokens: tokens per vLLM block (for computing num_slots).
    """

    def __init__(
        self,
        socket_path: str,
        chunk_manager: ChunkManager,
        retrieval_index: RetrievalIndex,
        position_encoder: PositionEncoder,
        slot_size: int,
        block_tokens: int = 16,
    ) -> None:
        self._socket_path = socket_path
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._slot_size = slot_size
        self._block_tokens = block_tokens
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("[IPC] listening on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the server and remove the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        logger.info("[IPC] server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one client connection (one request per connection).

        Args:
            reader: stream reader.
            writer: stream writer.
        """
        try:
            msg = await _read_frame(reader)
            op = msg.get("op")
            if op == "lookup":
                response = await self._handle_lookup(msg)
            elif op == "alloc_chunk":
                response = await self._handle_alloc_chunk(msg)
            elif op == "commit_chunk":
                response = await self._handle_commit_chunk(msg)
            elif op == "evict_chunk":
                response = await self._handle_evict_chunk(msg)
            else:
                response = {"error": f"unknown op: {op}"}
            await _write_frame(writer, response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] error handling request: %s", exc)
            try:
                await _write_frame(writer, {"error": str(exc)})
            except Exception:
                pass
        finally:
            writer.close()

    async def _handle_lookup(self, msg: dict[str, Any]) -> dict[str, Any]:
        tokens: list[int] = msg["tokens"]
        model_id: str = msg["model_id"]
        chunks = await self._ri.lookup(tokens, model_id)
        return {
            "chunks": [
                {
                    "chunk_key": m.chunk_key,
                    "start_slot": m.start_slot,
                    "num_slots": m.num_slots,
                    "token_count": m.token_count,
                    "pos_offset": m.pos_offset,
                    "model_id": m.model_id,
                    "file_offset": m.start_slot * self._slot_size,
                }
                for m in chunks
            ]
        }

    async def _handle_alloc_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        chunk_key: str = msg["chunk_key"]
        token_count: int = msg["token_count"]
        model_id: str = msg["model_id"]

        num_slots = math.ceil(token_count / self._block_tokens)
        pos_offset = self._pe.assign_offset(chunk_key, token_count)
        start_slot = self._cm.alloc(
            chunk_key=chunk_key,
            num_slots=num_slots,
            token_count=token_count,
            model_id=model_id,
            pos_offset=pos_offset,
        )
        file_offset = start_slot * self._slot_size
        logger.debug(
            "[IPC] alloc_chunk key=%s start=%d offset=%d pos=%d",
            chunk_key[:8],
            start_slot,
            file_offset,
            pos_offset,
        )
        return {
            "start_slot": start_slot,
            "file_offset": file_offset,
            "pos_offset": pos_offset,
        }

    async def _handle_commit_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        chunk_key: str = msg["chunk_key"]
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            return {"error": f"chunk_key not found: {chunk_key}"}
        await self._ri.insert(meta)
        logger.debug("[IPC] commit_chunk key=%s", chunk_key[:8])
        return {"ok": True}

    async def _handle_evict_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        chunk_key: str = msg["chunk_key"]
        await self._ri.remove(chunk_key)
        # ChunkManager evicts via its ring buffer (internal), but allow
        # explicit external eviction via MetadataStore remove.
        meta = self._cm.store.get(chunk_key)
        if meta is not None:
            self._cm.store.remove(chunk_key)
        logger.debug("[IPC] evict_chunk key=%s", chunk_key[:8])
        return {"ok": True}
```

### Step 2: Create `daser/server/__main__.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
import signal

# First Party
from daser.config import DaserConfig
from daser.logging import init_logger
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)


async def run_server(cfg: DaserConfig) -> None:
    """Start the DaseR server and run until SIGTERM/SIGINT.

    Args:
        cfg: DaserConfig instance.
    """
    slot_size = cfg.resolved_slot_size()
    store = MetadataStore(total_slots=cfg.total_slots)
    cm = ChunkManager(total_slots=cfg.total_slots, metadata_store=store)

    # Load persisted index if available
    if os.path.exists(cfg.index_path):
        try:
            cm.load(cfg.index_path)
            logger.info("[CHUNK] restored index from %s", cfg.index_path)
        except Exception as exc:
            logger.warning("[CHUNK] cold start — index load failed: %s", exc)

    ri = PrefixHashIndex(block_tokens=cfg.block_tokens)
    pe = FixedOffsetEncoder(fixed_offset=0)

    server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=slot_size,
        block_tokens=cfg.block_tokens,
    )
    await server.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    logger.info("[SERVER] DaseR server ready")
    await stop_event.wait()

    logger.info("[SERVER] shutting down — saving index to %s", cfg.index_path)
    os.makedirs(os.path.dirname(cfg.index_path), exist_ok=True)
    cm.save(cfg.index_path)
    await server.stop()
    logger.info("[SERVER] shutdown complete")


def main() -> None:
    """Entry point: python -m daser.server"""
    cfg = DaserConfig()
    asyncio.run(run_server(cfg))


if __name__ == "__main__":
    main()
```

### Step 3: Write IPC tests

Create `tests/server/test_ipc_server.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import msgpack
import pytest

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.chunk_manager import ChunkManager
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def make_server(tmp_path) -> IPCServer:
    socket_path = str(tmp_path / "test.sock")
    store = MetadataStore(total_slots=64)
    cm = ChunkManager(total_slots=64, metadata_store=store)
    ri = PrefixHashIndex(block_tokens=BLOCK_TOKENS)
    pe = FixedOffsetEncoder(fixed_offset=0)
    return IPCServer(
        socket_path=socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


async def _send_recv(socket_path: str, payload: dict) -> dict:
    """Send one msgpack frame and receive one msgpack frame."""
    reader, writer = await asyncio.open_unix_connection(socket_path)
    data = msgpack.packb(payload, use_bin_type=True)
    header = len(data).to_bytes(4, "big")
    writer.write(header + data)
    await writer.drain()

    resp_header = await reader.readexactly(4)
    resp_len = int.from_bytes(resp_header, "big")
    resp_data = await reader.readexactly(resp_len)
    writer.close()
    return msgpack.unpackb(resp_data, raw=False)


@pytest.mark.asyncio
async def test_alloc_chunk(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)
    resp = await _send_recv(
        str(tmp_path / "test.sock"),
        {"op": "alloc_chunk", "chunk_key": chunk_key, "token_count": 4, "model_id": "m"},
    )
    assert "start_slot" in resp
    assert resp["file_offset"] == resp["start_slot"] * SLOT_SIZE
    await server.stop()


@pytest.mark.asyncio
async def test_commit_and_lookup(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)

    await _send_recv(
        sock,
        {"op": "alloc_chunk", "chunk_key": chunk_key, "token_count": 4, "model_id": "m"},
    )
    await _send_recv(sock, {"op": "commit_chunk", "chunk_key": chunk_key})

    resp = await _send_recv(sock, {"op": "lookup", "tokens": tokens, "model_id": "m"})
    assert len(resp["chunks"]) == 1
    assert resp["chunks"][0]["chunk_key"] == chunk_key
    await server.stop()


@pytest.mark.asyncio
async def test_lookup_miss(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    resp = await _send_recv(
        str(tmp_path / "test.sock"),
        {"op": "lookup", "tokens": [9, 8, 7, 6], "model_id": "m"},
    )
    assert resp["chunks"] == []
    await server.stop()


@pytest.mark.asyncio
async def test_evict_chunk(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)

    await _send_recv(
        sock,
        {"op": "alloc_chunk", "chunk_key": chunk_key, "token_count": 4, "model_id": "m"},
    )
    await _send_recv(sock, {"op": "commit_chunk", "chunk_key": chunk_key})
    await _send_recv(sock, {"op": "evict_chunk", "chunk_key": chunk_key})

    resp = await _send_recv(sock, {"op": "lookup", "tokens": tokens, "model_id": "m"})
    assert resp["chunks"] == []
    await server.stop()


@pytest.mark.asyncio
async def test_unknown_op(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    resp = await _send_recv(
        str(tmp_path / "test.sock"), {"op": "bad_op"}
    )
    assert "error" in resp
    await server.stop()
```

### Step 4: Install pytest-asyncio

```bash
cd /home/zwt/daser_project/DaseR && source /data/zwt/vllm/bin/activate
pip install pytest-asyncio
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
asyncio_mode = "auto"
```

### Step 5: Run tests

```bash
pytest tests/retrieval/ tests/position/ tests/server/test_ipc_server.py -v
```

Expected: 5 + 3 + 5 = 13 tests pass.

### Step 6: Run full suite + lint

```bash
pytest tests/ -v
ruff check daser/ tests/
ruff format daser/ tests/
```

Expected: all 36 tests pass (23 from Plans 1+2, 13 new).

### Step 7: Single commit

```bash
git add daser/retrieval/ daser/position/ daser/server/ipc_server.py daser/server/__main__.py \
        tests/retrieval/ tests/position/ tests/server/test_ipc_server.py pyproject.toml
git commit -m "feat: add IPCServer, PrefixHashIndex, FixedOffsetEncoder, server entry point"
```

---

## Self-Review Checklist

| Spec Section | Covered |
|---|---|
| RetrievalIndex ABC (lookup, insert, remove) | `daser/retrieval/base.py` |
| PrefixHashIndex — exact prefix hash | `daser/retrieval/prefix.py` |
| PositionEncoder ABC (assign_offset, get_offset) | `daser/position/base.py` |
| FixedOffsetEncoder | `daser/position/fixed_offset.py` |
| IPC: Unix socket + msgpack | `daser/server/ipc_server.py` |
| IPC messages: lookup, alloc_chunk, commit_chunk, evict_chunk | `_handle_*` methods |
| Index serialized to daser.index on shutdown | `daser/server/__main__.py` |
| Cold start when index missing | `__main__.py` try/except |
| asyncio only — no threading mix | asyncio.start_unix_server throughout |
