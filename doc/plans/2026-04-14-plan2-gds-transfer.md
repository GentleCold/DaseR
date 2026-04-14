# Plan 2: GDS Transfer Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the async NVMe↔GPU IO layer using kvikio (cuFile / compat-mode fallback). No vLLM dependency, no IPC — pure IO path.

**Architecture:** `GDSTransferLayer` wraps `kvikio.cufile.CuFile` and exposes `write_async` / `read_into_async` returning `asyncio.Future[int]`. Backend detection happens once at init (`kvikio.defaults.compat_mode()`). cupy zero-copy views bridge torch tensors to kvikio buffers.

**Tech Stack:** Python 3.10+, kvikio 25.10, cupy, torch, pytest. Venv: `source /data/zwt/vllm/bin/activate`. Test disk: `/data/zwt/daser_test/`.

---

## File Map

| File | Create/Modify | Purpose |
|------|---------------|---------|
| `daser/connector/__init__.py` | Create | Package marker |
| `daser/connector/gds_transfer.py` | Create | `GDSTransferLayer` |
| `tests/connector/__init__.py` | Create | Package marker |
| `tests/connector/test_gds_transfer.py` | Create | IO round-trip and async tests |

---

## Task 1: Package Marker + GDSTransferLayer

**Files:**
- Create: `daser/connector/__init__.py`
- Create: `daser/connector/gds_transfer.py`
- Create: `tests/connector/__init__.py`

### Step 1: Create package markers

`daser/connector/__init__.py`:
```python
# SPDX-License-Identifier: Apache-2.0
```

`tests/connector/__init__.py`: empty file.

### Step 2: Create `daser/connector/gds_transfer.py`

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import enum
import os
from typing import Optional

# Third Party
import cupy
import kvikio
import kvikio.cufile
import kvikio.defaults
import torch

# First Party
from daser.logging import init_logger, init_perf_logger

logger = init_logger(__name__)
perf = init_perf_logger(__name__)


class TransferBackend(enum.Enum):
    """Active IO backend for GDSTransferLayer."""

    GDS = "gds"       # cuFile GDS — direct NVMe↔GPU DMA, no CPU involvement
    COMPAT = "compat" # kvikio compat mode — POSIX thread-pool + CPU bounce buffer


class GDSTransferLayer:
    """Async NVMe↔GPU IO using kvikio (cuFile GDS or compat-mode fallback).

    Opens a pre-existing file for read+write. Exposes coroutine-compatible
    methods that wrap kvikio IOFuture in asyncio via run_in_executor so
    callers stay in a pure asyncio event loop.

    Backend is selected once at construction from kvikio.defaults.compat_mode():
    - CompatMode.OFF  → GDS path (direct DMA)
    - CompatMode.ON   → compat path (POSIX + CPU bounce, still async via thread pool)
    - CompatMode.AUTO → treated as COMPAT unless GDS actually activates

    Args:
        path: absolute path to the pre-allocated store file.
    """

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Store file not found: {path}")

        mode = kvikio.defaults.compat_mode()
        if mode == kvikio.CompatMode.OFF:
            self._backend = TransferBackend.GDS
        else:
            self._backend = TransferBackend.COMPAT

        self._file = kvikio.cufile.CuFile(path, "r+")
        logger.info("[GDS] backend=%s path=%s", self._backend.value, path)

    @property
    def backend(self) -> TransferBackend:
        """The active IO backend (immutable after init)."""
        return self._backend

    async def write_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Write from a GPU buffer to the store file at file_offset.

        Non-blocking: submits IO and suspends the coroutine until the
        IOFuture completes in the kvikio thread pool.

        Args:
            buf: cupy ndarray on device (or host in compat mode).
            file_offset: byte offset in the store file to write at.
            nbytes: bytes to write; defaults to full buf size.

        Returns:
            Number of bytes written.
        """
        loop = asyncio.get_event_loop()
        io_future = self._file.pwrite(buf, nbytes, file_offset)
        return await loop.run_in_executor(None, io_future.get)

    async def read_into_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Read from the store file into a GPU buffer at file_offset.

        Non-blocking: submits IO and suspends the coroutine until the
        IOFuture completes in the kvikio thread pool.

        Args:
            buf: pre-allocated cupy ndarray on device to read into.
            file_offset: byte offset in the store file to read from.
            nbytes: bytes to read; defaults to full buf size.

        Returns:
            Number of bytes read.
        """
        loop = asyncio.get_event_loop()
        io_future = self._file.pread(buf, nbytes, file_offset)
        return await loop.run_in_executor(None, io_future.get)

    def close(self) -> None:
        """Close the underlying kvikio file handle."""
        self._file.close()
        logger.debug("[GDS] file closed")

    def __enter__(self) -> "GDSTransferLayer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
```

### Step 3: Commit package init files (not the main module yet)

```bash
cd /home/zwt/daser_project/DaseR
git add daser/connector/__init__.py tests/connector/__init__.py
git commit -m "feat: add connector package markers"
```

---

## Task 2: Tests + Verify

**Files:**
- Create: `tests/connector/test_gds_transfer.py`

### Step 1: Write tests first (TDD)

Create `tests/connector/test_gds_transfer.py`:

```python
# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os

# Third Party
import cupy
import pytest
import torch

# First Party
from daser.connector.gds_transfer import GDSTransferLayer, TransferBackend

TEST_DIR = "/data/zwt/daser_test"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def store_file(tmp_path):
    """Pre-allocate a small store file for testing."""
    path = str(tmp_path / "test.store")
    # 4 MB is large enough for all tests; fill with zeros
    size = 4 * 1024 * 1024
    with open(path, "wb") as f:
        f.write(b"\x00" * size)
    return path


def test_backend_is_set(store_file):
    """Backend must be GDS or COMPAT after construction."""
    gds = GDSTransferLayer(store_file)
    assert gds.backend in (TransferBackend.GDS, TransferBackend.COMPAT)
    gds.close()


def test_write_and_read_roundtrip(store_file):
    """Write a GPU tensor, read it back, verify exact equality."""
    nbytes = 1024 * 1024  # 1 MB
    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    src_cp = cupy.asarray(src)

    gds = GDSTransferLayer(store_file)
    written = _run(gds.write_async(src_cp, file_offset=0))
    assert written == nbytes

    dst = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    dst_cp = cupy.asarray(dst)
    read = _run(gds.read_into_async(dst_cp, file_offset=0))
    assert read == nbytes

    assert torch.equal(src, dst)
    gds.close()


def test_multiple_offsets(store_file):
    """Write two non-overlapping regions and read both back correctly."""
    size = 512 * 1024  # 512 KB each
    t1 = torch.ones(size, dtype=torch.uint8, device="cuda")
    t2 = torch.full((size,), 2, dtype=torch.uint8, device="cuda")

    gds = GDSTransferLayer(store_file)
    _run(gds.write_async(cupy.asarray(t1), file_offset=0))
    _run(gds.write_async(cupy.asarray(t2), file_offset=size))

    r1 = torch.zeros(size, dtype=torch.uint8, device="cuda")
    r2 = torch.zeros(size, dtype=torch.uint8, device="cuda")
    _run(gds.read_into_async(cupy.asarray(r1), file_offset=0))
    _run(gds.read_into_async(cupy.asarray(r2), file_offset=size))

    assert torch.equal(t1, r1)
    assert torch.equal(t2, r2)
    gds.close()


def test_context_manager(store_file):
    """Context manager closes file without errors."""
    with GDSTransferLayer(store_file) as gds:
        assert gds.backend in (TransferBackend.GDS, TransferBackend.COMPAT)


def test_missing_file_raises():
    """FileNotFoundError raised when store file does not exist."""
    with pytest.raises(FileNotFoundError):
        GDSTransferLayer("/data/zwt/daser_test/nonexistent.store")
```

### Step 2: Run tests (expect import error before module is created)

```bash
cd /home/zwt/daser_project/DaseR && source /data/zwt/vllm/bin/activate
pytest tests/connector/test_gds_transfer.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError` for `daser.connector.gds_transfer`.

### Step 3: Verify `gds_transfer.py` is correct, then run

After `gds_transfer.py` is in place:

```bash
pytest tests/connector/test_gds_transfer.py -v
```

Expected: 5 tests pass.

### Step 4: Run ruff + full suite

```bash
ruff check daser/connector/ tests/connector/
ruff format daser/connector/ tests/connector/
pytest tests/ -v
```

Expected: 23 tests pass (18 old + 5 new).

### Step 5: Single commit

```bash
git add daser/connector/gds_transfer.py tests/connector/test_gds_transfer.py
git commit -m "feat: add GDSTransferLayer with kvikio async IO"
```

---

## Self-Review Checklist

| Spec Section | Covered |
|---|---|
| cuFile async primary path | `TransferBackend.GDS` when compat_mode=OFF |
| io_uring fallback (POSIX thread pool via kvikio compat) | `TransferBackend.COMPAT` |
| Backend selected once at startup, immutable | `__init__` sets `_backend`, no runtime switching |
| Backend logged at startup | `logger.info("[GDS] backend=...")` |
| asyncio-compatible (no threading mix) | `run_in_executor` wraps IOFuture.get |
| Data plane stays in vLLM process | GDSTransferLayer is imported only from connector/ |
