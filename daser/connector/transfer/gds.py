# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import enum
from typing import Optional

# Third Party
import cupy
import kvikio
import kvikio.cufile
import kvikio.defaults
import torch

# First Party
from daser.connector.transfer.base import BaseTransferLayer, TransferBackendName
from daser.connector.transfer.utils import as_torch_uint8, require_store_path
from daser.logging import init_logger

logger = init_logger(__name__)


class KvikIOTransferBackend(enum.Enum):
    """Active IO backend for GDSTransferLayer."""

    GDS = "gds"  # cuFile GDS — direct NVMe↔GPU DMA, no CPU involvement
    COMPAT = "compat"  # kvikio compat mode — POSIX thread-pool + CPU bounce buffer


class GDSTransferLayer(BaseTransferLayer):
    """Async NVMe↔GPU IO using kvikio (cuFile GDS or compat-mode fallback).

    Opens a pre-existing file for read+write. Exposes coroutine-compatible
    methods that wrap kvikio IOFuture in asyncio via run_in_executor so
    callers stay in a pure asyncio event loop.

    Backend is selected once at construction from kvikio.defaults.get("compat_mode"):
    - CompatMode.OFF  → GDS path (direct DMA)
    - CompatMode.ON   → compat path (POSIX + CPU bounce, still async via thread pool)
    - CompatMode.AUTO → treated as COMPAT unless GDS actually activates

    Args:
        path: absolute path to the pre-allocated store file.
        nthreads: kvikio thread-pool size used in compat mode (default 4).
            Increasing this overlaps GPU→CPU staging with disk IO.
            Ignored when GDS direct-DMA is active.
    """

    def __init__(self, path: str, nthreads: int = 4) -> None:
        super().__init__()
        require_store_path(path)

        mode = kvikio.defaults.get("compat_mode")
        if mode == kvikio.CompatMode.OFF:
            self._backend = KvikIOTransferBackend.GDS
        else:
            self._backend = KvikIOTransferBackend.COMPAT
            # In compat mode, kvikio uses a POSIX thread pool for IO.
            # Default is 1 thread which serialises all writes; 4 threads
            # overlaps GPU→CPU staging with disk IO on btrfs/NVMe workloads.
            kvikio.defaults.set("num_threads", nthreads)

        self._file = kvikio.cufile.CuFile(path, "r+")
        logger.info(
            "[GDS] backend=%s nthreads=%d path=%s", self._backend.name, nthreads, path
        )

    @property
    def backend_name(self) -> TransferBackendName:
        """Configured transfer backend name."""
        return TransferBackendName.GDS

    @property
    def backend(self) -> KvikIOTransferBackend:
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
        written = await loop.run_in_executor(None, io_future.get)
        self._record_l2_write(written)
        return written

    async def write_chunk_async(
        self,
        chunk_key: str,
        buf: torch.Tensor,
        file_offset: int,
        nbytes: int,
    ) -> int:
        """Write one logical chunk from a torch staging tensor.

        Args:
            chunk_key: cache key for the chunk; GDS does not maintain L1 state.
            buf: source torch tensor containing chunk bytes.
            file_offset: byte offset in the store file.
            nbytes: bytes to write.

        Returns:
            Number of bytes written.
        """
        del chunk_key
        return await self.write_async(
            cupy.asarray(as_torch_uint8(buf)),
            file_offset,
            nbytes,
        )

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
        read = await loop.run_in_executor(None, io_future.get)
        self._record_l2_read(read)
        return read

    async def read_chunk_into_async(
        self,
        chunk_key: str,
        buf: torch.Tensor,
        file_offset: int,
        nbytes: int,
        l2_durable: bool,
        protect_lookup: bool = False,
    ) -> int:
        """Read one logical chunk into a torch staging tensor.

        Args:
            chunk_key: cache key for the chunk; GDS does not maintain L1 state.
            buf: destination torch tensor for chunk bytes.
            file_offset: byte offset in the store file.
            nbytes: bytes to read.
            l2_durable: must be True because GDS reads only from durable L2.
            protect_lookup: ignored because GDS has no local L1 cache.

        Returns:
            Number of bytes read.

        Raises:
            RuntimeError: if L2 is not durable.
        """
        del chunk_key, protect_lookup
        if not l2_durable:
            raise RuntimeError("chunk is not durable in L2")
        return await self.read_into_async(
            cupy.asarray(as_torch_uint8(buf)),
            file_offset,
            nbytes,
        )

    def pin_chunks_for_lookup(self, chunk_keys: list[str]) -> None:
        """No-op lookup protection for direct GDS transfers.

        Args:
            chunk_keys: ignored cache keys.
        """
        del chunk_keys

    def release_lookup_pins(self, chunk_keys: list[str]) -> None:
        """No-op lookup release for direct GDS transfers.

        Args:
            chunk_keys: ignored cache keys.
        """
        del chunk_keys

    def close(self) -> None:
        """Close the underlying kvikio file handle."""
        self._file.close()
        logger.debug("[GDS] file closed")

    def stats(self):
        """Return GDS transfer counters.

        Returns:
            Transfer stats containing GDS L2 byte counters.

        Async/thread-safety:
            Reads immutable backend state only.
        """
        return self._base_stats()

    def __enter__(self) -> "GDSTransferLayer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
