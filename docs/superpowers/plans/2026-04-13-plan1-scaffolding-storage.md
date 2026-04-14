# Plan 1: Scaffolding + Storage Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the testable, GPU-free foundation of DaseR: package setup, logging, config, in-memory metadata store, and ring buffer chunk manager.

**Architecture:** `DaserConfig` drives everything. `MetadataStore` holds the in-memory index (chunk_key → ChunkMeta) plus a reverse slot_map. `ChunkManager` implements the ring buffer on top of `MetadataStore`, handling alloc, evict, and wrap-around. No I/O, no GDS, no vLLM in this plan.

**Tech Stack:** Python 3.10+, msgpack, pytest, ruff, mypy. Venv: `source /data/zwt/vllm/bin/activate`.

---

## File Map

| File | Create/Modify | Purpose |
|------|---------------|---------|
| `pyproject.toml` | Create | Package metadata, deps, tool config |
| `daser/__init__.py` | Create | Package marker |
| `daser/logging.py` | Create | `init_logger` + `init_perf_logger` |
| `daser/config.py` | Create | `DaserConfig` dataclass |
| `daser/server/__init__.py` | Create | Package marker |
| `daser/server/metadata_store.py` | Create | `ChunkMeta`, `SlotEntry`, `MetadataStore` |
| `daser/server/chunk_manager.py` | Create | `ChunkManager` (ring buffer) |
| `tests/__init__.py` | Create | Package marker |
| `tests/server/__init__.py` | Create | Package marker |
| `tests/server/test_metadata_store.py` | Create | Unit tests for MetadataStore |
| `tests/server/test_chunk_manager.py` | Create | Unit tests for ChunkManager |

---

## Task 1: Package Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `daser/__init__.py`
- Create: `daser/server/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/server/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "daser"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "msgpack>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "B", "SLF"]

[tool.ruff.lint.isort]
known-first-party = ["daser"]
force-sort-within-sections = true

[tool.mypy]
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package markers**

`daser/__init__.py`:
```python
# SPDX-License-Identifier: Apache-2.0
```

`daser/server/__init__.py`:
```python
# SPDX-License-Identifier: Apache-2.0
```

`tests/__init__.py` and `tests/server/__init__.py`: empty files.

- [ ] **Step 3: Install package**

```bash
cd /home/zwt/daser_project/DaseR
source /data/zwt/vllm/bin/activate
pip install -e ".[dev]"
```

Expected: installs successfully, `import daser` works.

- [ ] **Step 4: Verify**

```bash
python -c "import daser; print('ok')"
pytest --collect-only   # should find 0 tests, no errors
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml daser/ tests/
git commit -m "feat: add package scaffolding"
```

---

## Task 2: Logging Module

**Files:**
- Create: `daser/logging.py`

- [ ] **Step 1: Write the module**

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
from typing import Optional


def init_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a logger for the given module name.

    The log level is taken from the DASER_LOG_LEVEL environment variable
    (default INFO). Callers should use component tags in messages:
    [GDS], [INDEX], [CHUNK], [IPC], [CONNECTOR].

    Args:
        name: typically __name__ of the calling module.
        level: override log level string (e.g. "DEBUG"). If None, reads
               DASER_LOG_LEVEL env var, defaulting to "INFO".

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    resolved_level = level or os.environ.get("DASER_LOG_LEVEL", "INFO")
    logger.setLevel(resolved_level.upper())
    return logger


class PerfLogger:
    """Logger for performance metrics (latency, hit rate, throughput).

    Writes structured records to a dedicated logger at DEBUG level.
    Enable with DASER_PERF_LOG=1.

    Args:
        name: module name for the underlying logger.
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(f"{name}.perf")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s PERF %(name)s %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
        self._enabled = os.environ.get("DASER_PERF_LOG", "0") == "1"

    def record(self, metric: str, value: float, unit: str = "") -> None:
        """Record a single metric value.

        Args:
            metric: metric name, e.g. "gds_read_latency_ms".
            value: numeric value.
            unit: optional unit string for clarity.
        """
        if self._enabled:
            self._logger.debug("%s=%.4f%s", metric, value, unit)


def init_perf_logger(name: str) -> PerfLogger:
    """Return a PerfLogger for the given module name.

    Args:
        name: typically __name__ of the calling module.

    Returns:
        PerfLogger instance.
    """
    return PerfLogger(name)
```

- [ ] **Step 2: Smoke test (no pytest yet)**

```bash
python -c "
from daser.logging import init_logger, init_perf_logger
log = init_logger('test')
log.info('[CHUNK] logger works')
perf = init_perf_logger('test')
perf.record('latency_ms', 1.23, 'ms')
print('logging ok')
"
```

Expected: prints log line then `logging ok`.

- [ ] **Step 3: Commit**

```bash
git add daser/logging.py
git commit -m "feat: add unified logging module"
```

---

## Task 3: Config

**Files:**
- Create: `daser/config.py`

- [ ] **Step 1: Write the module**

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field


@dataclass
class DaserConfig:
    """Top-level configuration for DaseR.

    slot_size is computed automatically from model params when it is 0.
    All paths should be absolute.

    Attributes:
        store_path: absolute path to the KV data file (daser.store).
        index_path: absolute path to the serialized index (daser.index).
        total_slots: number of fixed-size slots in the ring buffer.
        slot_size: size of one slot in bytes. 0 means compute from model params.
        ipc_socket_path: Unix socket path for Connector ↔ Server IPC.
        log_level: log level string passed to init_logger.
        perf_log_enabled: whether to activate PerfLogger output.
        num_kv_heads: number of KV attention heads (for slot_size computation).
        head_dim: attention head dimension (for slot_size computation).
        num_layers: number of transformer layers (for slot_size computation).
        block_tokens: tokens per vLLM block (default 16).
        dtype_bytes: bytes per element, e.g. 2 for bf16.
        model_id: identifier string for the model, used to prevent
                  cross-model cache reuse.
    """

    store_path: str = "/mnt/xfs/daser.store"
    index_path: str = "/mnt/xfs/daser.index"
    total_slots: int = 1024
    slot_size: int = 0
    ipc_socket_path: str = "/tmp/daser.sock"
    log_level: str = "INFO"
    perf_log_enabled: bool = False

    # Model params used only when slot_size == 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_layers: int = 0
    block_tokens: int = 16
    dtype_bytes: int = 2
    model_id: str = "default"

    def resolved_slot_size(self) -> int:
        """Return slot_size, computing it from model params if slot_size is 0.

        Returns:
            Slot size in bytes.

        Raises:
            ValueError: if slot_size is 0 and any model param is 0.
        """
        if self.slot_size > 0:
            return self.slot_size
        for param, name in [
            (self.num_kv_heads, "num_kv_heads"),
            (self.head_dim, "head_dim"),
            (self.num_layers, "num_layers"),
        ]:
            if param == 0:
                raise ValueError(
                    f"slot_size is 0 but {name} is also 0; "
                    "provide either slot_size or all model params"
                )
        return (
            self.num_kv_heads
            * self.head_dim
            * 2  # K and V
            * self.num_layers
            * self.block_tokens
            * self.dtype_bytes
        )
```

- [ ] **Step 2: Write tests**

Create `tests/test_config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
from daser.config import DaserConfig


def test_explicit_slot_size():
    cfg = DaserConfig(slot_size=1024)
    assert cfg.resolved_slot_size() == 1024


def test_computed_slot_size():
    cfg = DaserConfig(
        num_kv_heads=8,
        head_dim=128,
        num_layers=28,
        block_tokens=16,
        dtype_bytes=2,
    )
    expected = 8 * 128 * 2 * 28 * 16 * 2
    assert cfg.resolved_slot_size() == expected


def test_missing_model_param_raises():
    cfg = DaserConfig(num_kv_heads=8, head_dim=128)  # num_layers missing
    with pytest.raises(ValueError, match="num_layers"):
        cfg.resolved_slot_size()
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add daser/config.py tests/test_config.py
git commit -m "feat: add DaserConfig with slot_size resolution"
```

---

## Task 4: MetadataStore

**Files:**
- Create: `daser/server/metadata_store.py`
- Create: `tests/server/test_metadata_store.py`

- [ ] **Step 1: Write the module**

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import time
from dataclasses import asdict, dataclass
from typing import Literal, Optional

# Third Party
import msgpack

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)


@dataclass
class ChunkMeta:
    """Metadata for one cached KV chunk.

    Attributes:
        chunk_key: SHA256(token_ids) or document ID identifying this chunk.
        start_slot: index of the first slot in daser.store.
        num_slots: number of contiguous slots occupied.
        token_count: number of tokens whose KV is stored.
        pos_offset: position encoding offset applied at load time.
        model_id: model identifier, prevents cross-model reuse.
        created_at: unix timestamp of insertion.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class SlotEntry:
    """Describes one logical unit in the ring buffer's slot_map.

    kind="chunk" marks the first slot of a real KV chunk.
    kind="skip"  marks the first slot of a SKIP block (wrap padding).
    kind="cont"  marks continuation slots (positions 1..n-1 of a chunk/skip).

    For kind="chunk" and kind="skip", num_slots is the total size of the unit.
    For kind="cont", num_slots is unused (set to 0).

    Attributes:
        kind: one of "chunk", "skip", "cont".
        chunk_key: set for kind="chunk"; None otherwise.
        num_slots: total slots in this unit (only meaningful for first slot).
    """

    kind: Literal["chunk", "skip", "cont"]
    chunk_key: Optional[str] = None
    num_slots: int = 0


class MetadataStore:
    """In-memory index for DaseR's ring buffer.

    Maintains two structures:
    - chunk_index: maps chunk_key → ChunkMeta for O(1) lookup and removal.
    - slot_map: list[SlotEntry] indexed by slot_id, for ring buffer traversal.

    Can be serialized to / deserialized from a msgpack file.

    Args:
        total_slots: total number of slots in the ring buffer (must match
                     ChunkManager's total_slots).
    """

    def __init__(self, total_slots: int) -> None:
        self._total_slots = total_slots
        self._chunk_index: dict[str, ChunkMeta] = {}
        self._slot_map: list[SlotEntry] = [
            SlotEntry(kind="cont") for _ in range(total_slots)
        ]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, meta: ChunkMeta) -> None:
        """Insert a chunk into the index and mark its slots in slot_map.

        The first slot gets kind="chunk", remaining slots get kind="cont".

        Args:
            meta: ChunkMeta to insert.

        Raises:
            ValueError: if chunk_key already exists.
        """
        if meta.chunk_key in self._chunk_index:
            raise ValueError(f"chunk_key already exists: {meta.chunk_key}")
        self._chunk_index[meta.chunk_key] = meta
        self._slot_map[meta.start_slot] = SlotEntry(
            kind="chunk", chunk_key=meta.chunk_key, num_slots=meta.num_slots
        )
        for i in range(1, meta.num_slots):
            self._slot_map[meta.start_slot + i] = SlotEntry(kind="cont")
        logger.debug(
            "[INDEX] insert chunk_key=%s start_slot=%d num_slots=%d",
            meta.chunk_key,
            meta.start_slot,
            meta.num_slots,
        )

    def insert_skip(self, start_slot: int, num_slots: int) -> None:
        """Mark a range of slots as SKIP (wrap-around padding).

        Args:
            start_slot: first slot of the SKIP block.
            num_slots: number of slots in the block.
        """
        self._slot_map[start_slot] = SlotEntry(
            kind="skip", chunk_key=None, num_slots=num_slots
        )
        for i in range(1, num_slots):
            self._slot_map[start_slot + i] = SlotEntry(kind="cont")
        logger.debug(
            "[INDEX] insert_skip start_slot=%d num_slots=%d",
            start_slot,
            num_slots,
        )

    def remove(self, chunk_key: str) -> None:
        """Remove a chunk from the index.

        Does not clear slot_map entries — ChunkManager advances tail past them.

        Args:
            chunk_key: key of the chunk to remove.

        Raises:
            KeyError: if chunk_key is not found.
        """
        if chunk_key not in self._chunk_index:
            raise KeyError(f"chunk_key not found: {chunk_key}")
        del self._chunk_index[chunk_key]
        logger.debug("[INDEX] remove chunk_key=%s", chunk_key)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, chunk_key: str) -> Optional[ChunkMeta]:
        """Look up a chunk by key.

        Args:
            chunk_key: the key to look up.

        Returns:
            ChunkMeta if found, None otherwise.
        """
        return self._chunk_index.get(chunk_key)

    def get_slot_entry(self, slot_id: int) -> SlotEntry:
        """Return the SlotEntry at the given slot index.

        Args:
            slot_id: slot index in [0, total_slots).

        Returns:
            SlotEntry for that slot.
        """
        return self._slot_map[slot_id]

    def __len__(self) -> int:
        """Return number of chunks currently stored."""
        return len(self._chunk_index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to a msgpack file.

        Writes ring buffer state (head, tail) are not stored here — they
        are passed in by ChunkManager. Call ChunkManager.save() instead
        to persist the full state.

        Args:
            path: absolute path to write the msgpack file.
        """
        payload = {
            "total_slots": self._total_slots,
            "chunk_index": {k: asdict(v) for k, v in self._chunk_index.items()},
            "slot_map": [
                {"kind": e.kind, "chunk_key": e.chunk_key, "num_slots": e.num_slots}
                for e in self._slot_map
            ],
        }
        with open(path, "wb") as f:
            f.write(msgpack.packb(payload, use_bin_type=True))
        logger.info("[INDEX] saved %d chunks to %s", len(self._chunk_index), path)

    def load(self, path: str) -> None:
        """Deserialize the index from a msgpack file, replacing current state.

        Args:
            path: absolute path to the msgpack file.
        """
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        self._total_slots = payload["total_slots"]
        self._chunk_index = {
            k: ChunkMeta(**v) for k, v in payload["chunk_index"].items()
        }
        self._slot_map = [
            SlotEntry(
                kind=e["kind"],
                chunk_key=e["chunk_key"],
                num_slots=e["num_slots"],
            )
            for e in payload["slot_map"]
        ]
        logger.info("[INDEX] loaded %d chunks from %s", len(self._chunk_index), path)
```

- [ ] **Step 2: Write tests**

Create `tests/server/test_metadata_store.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import tempfile
import time

import pytest

from daser.server.metadata_store import ChunkMeta, MetadataStore, SlotEntry


def make_meta(key: str, start: int, num: int, tokens: int = 16) -> ChunkMeta:
    return ChunkMeta(
        chunk_key=key,
        start_slot=start,
        num_slots=num,
        token_count=tokens,
        pos_offset=0,
        model_id="test-model",
        created_at=time.time(),
    )


def test_insert_and_get():
    store = MetadataStore(total_slots=8)
    meta = make_meta("abc", start=0, num=3)
    store.insert(meta)
    assert store.get("abc") == meta
    assert len(store) == 1


def test_slot_map_after_insert():
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=2, num=3))
    entry = store.get_slot_entry(2)
    assert entry.kind == "chunk"
    assert entry.chunk_key == "abc"
    assert entry.num_slots == 3
    assert store.get_slot_entry(3).kind == "cont"
    assert store.get_slot_entry(4).kind == "cont"


def test_remove():
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=2))
    store.remove("abc")
    assert store.get("abc") is None
    assert len(store) == 0


def test_remove_nonexistent_raises():
    store = MetadataStore(total_slots=8)
    with pytest.raises(KeyError):
        store.remove("nonexistent")


def test_insert_duplicate_raises():
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=2))
    with pytest.raises(ValueError):
        store.insert(make_meta("abc", start=2, num=2))


def test_insert_skip():
    store = MetadataStore(total_slots=8)
    store.insert_skip(start_slot=6, num_slots=2)
    entry = store.get_slot_entry(6)
    assert entry.kind == "skip"
    assert entry.num_slots == 2
    assert store.get_slot_entry(7).kind == "cont"


def test_save_and_load(tmp_path):
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=3))
    store.insert(make_meta("def", start=3, num=2))
    path = str(tmp_path / "daser.index")
    store.save(path)

    store2 = MetadataStore(total_slots=8)
    store2.load(path)
    assert store2.get("abc") is not None
    assert store2.get("def") is not None
    assert len(store2) == 2
    assert store2.get_slot_entry(0).kind == "chunk"
    assert store2.get_slot_entry(1).kind == "cont"
```

- [ ] **Step 3: Run tests (expect failure first)**

```bash
pytest tests/server/test_metadata_store.py -v
```

Expected: `ModuleNotFoundError` since the file doesn't exist yet. Verify the error points to `metadata_store`.

- [ ] **Step 4: Create the file and run again**

After writing the file in Step 1:

```bash
pytest tests/server/test_metadata_store.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daser/server/metadata_store.py tests/server/test_metadata_store.py
git commit -m "feat: add MetadataStore with ChunkMeta and SlotEntry"
```

---

## Task 5: ChunkManager (Ring Buffer)

**Files:**
- Create: `daser/server/chunk_manager.py`
- Create: `tests/server/test_chunk_manager.py`

- [ ] **Step 1: Write tests first (TDD)**

Create `tests/server/test_chunk_manager.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import tempfile
import time

import pytest

from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import MetadataStore


def make_manager(total_slots: int = 8) -> ChunkManager:
    store = MetadataStore(total_slots=total_slots)
    return ChunkManager(total_slots=total_slots, metadata_store=store)


def test_alloc_returns_correct_start_slot():
    mgr = make_manager(8)
    slot = mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert slot == 0


def test_alloc_advances_head():
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    slot = mgr.alloc("key2", num_slots=2, token_count=32, model_id="m", pos_offset=48)
    assert slot == 3


def test_free_slots_decreases():
    mgr = make_manager(8)
    assert mgr.free_slots == 8
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert mgr.free_slots == 5


def test_evicts_oldest_when_full():
    mgr = make_manager(6)
    # fill up: key1 (3 slots), key2 (3 slots)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    # alloc key3 (3 slots) — must evict key1
    mgr.alloc("key3", num_slots=3, token_count=48, model_id="m", pos_offset=96)
    assert mgr.store.get("key1") is None
    assert mgr.store.get("key2") is not None
    assert mgr.store.get("key3") is not None


def test_evict_advances_tail():
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert mgr.tail == 0
    mgr.evict_oldest()
    assert mgr.tail == 3
    assert mgr.store.get("key1") is None


def test_wrap_around():
    # total=8; write key1(3), key2(3) → head=6; now key3(4) wraps
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    # head=6, only 2 slots left — key3 needs 4, triggers wrap
    # evicts key1 (tail=0→3), inserts SKIP at [6,7], head wraps to 0
    # evicts key2 (tail=3→6) to make room
    # key3 alloc'd at slot 0
    slot = mgr.alloc("key3", num_slots=4, token_count=64, model_id="m", pos_offset=96)
    assert slot == 0
    assert mgr.store.get("key3") is not None


def test_wrap_skip_advances_tail():
    # After wrap, tail must skip past the SKIP block naturally
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.alloc("key3", num_slots=4, token_count=64, model_id="m", pos_offset=96)
    # key1 and key2 were evicted; key3 at [0..3]; SKIP at [6,7]; tail at 6
    # alloc key4 (2 slots): tail must skip past [6,7], wrap to 0
    mgr.alloc("key4", num_slots=2, token_count=32, model_id="m", pos_offset=160)
    # key4 should sit at slot 4
    assert mgr.store.get("key4") is not None
    assert mgr.store.get("key4").start_slot == 4


def test_save_and_load_state(tmp_path):
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=2, token_count=32, model_id="m", pos_offset=48)
    path = str(tmp_path / "daser.index")
    mgr.save(path)

    store2 = MetadataStore(total_slots=8)
    mgr2 = ChunkManager(total_slots=8, metadata_store=store2)
    mgr2.load(path)
    assert mgr2.head == mgr.head
    assert mgr2.tail == mgr.tail
    assert mgr2.store.get("key1") is not None
    assert mgr2.store.get("key2") is not None
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/server/test_chunk_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'daser.server.chunk_manager'`

- [ ] **Step 3: Implement ChunkManager**

Create `daser/server/chunk_manager.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.server.metadata_store import ChunkMeta, MetadataStore

logger = init_logger(__name__)


class ChunkManager:
    """Ring buffer manager for DaseR's slot-based KV store.

    Allocates contiguous slots from a fixed-size ring buffer. When the
    buffer is full, the oldest chunk is evicted automatically. Handles
    wrap-around at the end of the buffer by inserting SKIP blocks.

    Args:
        total_slots: total number of slots in the ring buffer.
        metadata_store: MetadataStore instance this manager operates on.
    """

    def __init__(self, total_slots: int, metadata_store: MetadataStore) -> None:
        self._total_slots = total_slots
        self._store = metadata_store
        self._head: int = 0  # next slot to write
        self._tail: int = 0  # oldest chunk's first slot

    @property
    def store(self) -> MetadataStore:
        """The underlying MetadataStore."""
        return self._store

    @property
    def head(self) -> int:
        """Index of the next slot to be written."""
        return self._head

    @property
    def tail(self) -> int:
        """Index of the oldest chunk's first slot."""
        return self._tail

    @property
    def free_slots(self) -> int:
        """Number of slots available without eviction."""
        if self._head >= self._tail:
            return self._total_slots - (self._head - self._tail)
        return self._tail - self._head

    def alloc(
        self,
        chunk_key: str,
        num_slots: int,
        token_count: int,
        model_id: str,
        pos_offset: int,
    ) -> int:
        """Allocate contiguous slots for a new chunk.

        Evicts oldest chunks as needed. Inserts a SKIP block and wraps
        head to 0 if there is not enough contiguous space at the end of
        the buffer.

        Args:
            chunk_key: unique identifier for the chunk.
            num_slots: number of contiguous slots required.
            token_count: number of tokens in this chunk.
            model_id: model identifier for cache invalidation.
            pos_offset: position encoding offset for this chunk.

        Returns:
            start_slot: index of the first allocated slot.

        Raises:
            ValueError: if num_slots > total_slots.
        """
        if num_slots > self._total_slots:
            raise ValueError(
                f"num_slots={num_slots} exceeds total_slots={self._total_slots}"
            )

        # Wrap if not enough contiguous space at end of buffer
        tail_space = self._total_slots - self._head
        if tail_space < num_slots and tail_space > 0:
            # Evict any chunks that overlap the SKIP region
            self._evict_range(self._head, tail_space)
            self._store.insert_skip(self._head, tail_space)
            logger.debug("[CHUNK] wrap: SKIP %d slots at %d", tail_space, self._head)
            self._head = 0

        # Evict until we have enough free space
        while self.free_slots < num_slots:
            self.evict_oldest()

        start_slot = self._head
        meta = ChunkMeta(
            chunk_key=chunk_key,
            start_slot=start_slot,
            num_slots=num_slots,
            token_count=token_count,
            pos_offset=pos_offset,
            model_id=model_id,
        )
        self._store.insert(meta)
        self._head = (self._head + num_slots) % self._total_slots
        logger.debug(
            "[CHUNK] alloc chunk_key=%s start=%d num=%d head=%d tail=%d",
            chunk_key,
            start_slot,
            num_slots,
            self._head,
            self._tail,
        )
        return start_slot

    def evict_oldest(self) -> None:
        """Evict the oldest chunk (or skip block) at tail and advance tail.

        Raises:
            RuntimeError: if tail == head (buffer is empty).
        """
        if self._tail == self._head and len(self._store) == 0:
            raise RuntimeError("evict_oldest called on empty ring buffer")

        entry = self._store.get_slot_entry(self._tail)
        if entry.kind == "chunk":
            assert entry.chunk_key is not None
            self._store.remove(entry.chunk_key)
            self._tail = (self._tail + entry.num_slots) % self._total_slots
            logger.debug(
                "[CHUNK] evict chunk_key=%s tail=%d", entry.chunk_key, self._tail
            )
        elif entry.kind == "skip":
            self._tail = (self._tail + entry.num_slots) % self._total_slots
            logger.debug("[CHUNK] skip block consumed, tail=%d", self._tail)
        else:
            # cont slot at tail is a bug — advance by 1 to recover
            logger.warning("[CHUNK] unexpected cont slot at tail=%d", self._tail)
            self._tail = (self._tail + 1) % self._total_slots

    def _evict_range(self, start: int, count: int) -> None:
        """Evict any chunks whose first slot falls within [start, start+count).

        Used before inserting a SKIP block during wrap-around.

        Args:
            start: first slot of the range.
            count: number of slots in the range.
        """
        for slot_id in range(start, start + count):
            entry = self._store.get_slot_entry(slot_id)
            if entry.kind == "chunk" and entry.chunk_key is not None:
                self._store.remove(entry.chunk_key)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist full ring buffer state (index + head/tail) to path.

        Args:
            path: absolute file path to write.
        """
        import io

        # Save index first into a bytes buffer
        self._store.save(path + ".tmp_store")
        with open(path + ".tmp_store", "rb") as f:
            index_bytes = f.read()
        import os
        os.unlink(path + ".tmp_store")

        payload = {
            "head": self._head,
            "tail": self._tail,
            "index": index_bytes,
        }
        with open(path, "wb") as f:
            f.write(msgpack.packb(payload, use_bin_type=True))
        logger.info("[CHUNK] state saved to %s (head=%d tail=%d)", path, self._head, self._tail)

    def load(self, path: str) -> None:
        """Restore ring buffer state from path, replacing current state.

        Args:
            path: absolute file path to read.
        """
        import tempfile
        import os

        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        self._head = payload["head"]
        self._tail = payload["tail"]

        # Restore index via a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".index")
        tmp.write(payload["index"])
        tmp.close()
        try:
            self._store.load(tmp.name)
        finally:
            os.unlink(tmp.name)

        logger.info("[CHUNK] state loaded from %s (head=%d tail=%d)", path, self._head, self._tail)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/server/test_chunk_manager.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (config + metadata_store + chunk_manager).

- [ ] **Step 6: Commit**

```bash
git add daser/server/chunk_manager.py tests/server/test_chunk_manager.py
git commit -m "feat: add ChunkManager ring buffer with wrap-around and eviction"
```

---

## Task 6: Lint and Type Check

**Files:** no new files.

- [ ] **Step 1: Run ruff**

```bash
ruff check daser/ tests/
```

Fix any reported issues. Common ones: missing blank lines, unused imports.

- [ ] **Step 2: Run ruff format**

```bash
ruff format daser/ tests/
```

- [ ] **Step 3: Run mypy**

```bash
mypy daser/
```

Fix any type errors. Common ones: `Optional` not handled, missing return types.

- [ ] **Step 4: Full test suite one more time**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "style: apply ruff format and fix mypy warnings in storage layer"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task |
|---|---|
| Project layout: pyproject.toml, daser/ package | Task 1 |
| daser/logging.py with init_logger + perf_logger | Task 2 |
| DaserConfig with resolved_slot_size | Task 3 |
| ChunkMeta, SlotEntry | Task 4 |
| MetadataStore: insert, remove, get, save, load | Task 4 |
| ChunkManager: ring buffer, alloc, evict, wrap | Task 5 |
| SKIP block handling | Task 5 (test_wrap_around, test_wrap_skip_advances_tail) |
| Index serialization (msgpack, no WAL) | Tasks 4 + 5 |

**Not in scope for Plan 1 (covered in later plans):**
- GDS Transfer Layer (Plan 2)
- IPC server + RetrievalIndex + PositionEncoder (Plan 3)
- DaserConnector + vLLM integration (Plan 4)

### Type Consistency Check

- `ChunkMeta` fields used identically in MetadataStore and ChunkManager ✓
- `SlotEntry.kind` Literal values ("chunk", "skip", "cont") consistent across both modules ✓
- `MetadataStore.insert_skip(start_slot, num_slots)` matches ChunkManager call site ✓
- `ChunkManager.alloc(chunk_key, num_slots, token_count, model_id, pos_offset)` matches test calls ✓
- `ChunkManager.save(path)` / `load(path)` match test calls ✓
