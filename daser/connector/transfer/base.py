# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import enum
from typing import Optional, Protocol

# Third Party
import cupy
import torch


class TransferBackendName(enum.Enum):
    """Configured data-plane transfer backend."""

    GDS = "gds"
    IOURING_MEM = "iouring-mem"


@dataclass(frozen=True)
class TransferStats:
    """Runtime counters for a transfer layer.

    Attributes:
        l1_hits: number of L1 cache hits.
        l1_misses: number of L1 cache misses.
        l1_evictions: number of L1 cache evictions.
        l1_bytes: bytes currently held by L1.
        l2_read_bytes: bytes read from SSD L2.
        l2_write_bytes: bytes written to SSD L2.

    Async/thread-safety:
        Implementations should return a snapshot that can be read from the
        worker thread without mutating transfer state.
    """

    l1_hits: int = 0
    l1_misses: int = 0
    l1_evictions: int = 0
    l1_bytes: int = 0
    l2_read_bytes: int = 0
    l2_write_bytes: int = 0


@dataclass(frozen=True)
class TransferConfig:
    """Configuration used to construct a transfer layer.

    Attributes:
        backend_name: immutable backend selected by the server.
        store_path: preallocated L2 store file path.
        l1_cache_size: pinned host L1 capacity in bytes. Only backends with an
            L1 cache use this value.

    Async/thread-safety:
        Immutable value object passed from the worker thread during transfer
        initialization.
    """

    backend_name: TransferBackendName
    store_path: str
    l1_cache_size: int = 0


@dataclass(frozen=True)
class TransferCallbacks:
    """Server publication callbacks used by transfer implementations.

    Attributes:
        commit_chunk: async callback for durable GDS chunk publication.
        commit_l1: async callback for publishing worker L1 residency.
        commit_l2: async callback for publishing durable L2 residency.
        evict_l1: async callback for removing worker L1 residency.

    Async/thread-safety:
        Callbacks are invoked from the worker background asyncio loop. They
        should not perform synchronous blocking work.
    """

    commit_chunk: Callable[[str], object]
    commit_l1: Callable[[str], object]
    commit_l2: Callable[[str], object]
    evict_l1: Callable[[str], object]


class TransferLayer(Protocol):
    """Connector data-plane transfer interface.

    Implementations move whole contiguous KV chunks or spans between the
    worker's GPU staging buffers, optional host-memory cache, and SSD store.
    Backend selection is immutable after construction.
    """

    @property
    def backend_name(self) -> TransferBackendName:
        """Return the configured backend name."""
        ...

    @property
    def max_concurrent_chunk_reads(self) -> int:
        """Return the supported concurrent chunk-read budget.

        Returns:
            Maximum number of chunk-level read coroutines the connector should
            allow to be active at once.

        Async/thread-safety:
            Read on the worker thread after backend construction. The value
            must be immutable for the lifetime of the transfer layer.
        """
        ...

    async def write_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Write from a GPU-visible buffer to the transfer backend.

        Args:
            buf: source CuPy view.
            file_offset: byte offset in the logical L2 store.
            nbytes: bytes to write; defaults to the full buffer.

        Returns:
            Number of bytes accepted by the backend.

        Async/thread-safety:
            Must not block the worker event loop while storage I/O is pending.
        """
        ...

    async def write_chunk_async(
        self,
        chunk_key: str,
        buf: torch.Tensor,
        file_offset: int,
        nbytes: int,
    ) -> int:
        """Write one logical chunk from a GPU staging tensor.

        Args:
            chunk_key: cache key for the chunk being written.
            buf: source torch tensor containing chunk bytes.
            file_offset: byte offset in the logical L2 store.
            nbytes: bytes to write.

        Returns:
            Number of bytes accepted by the backend.

        Async/thread-safety:
            Must not block the worker event loop while storage I/O is pending.
        """
        ...

    async def read_into_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Read from the transfer backend into a GPU-visible buffer.

        Args:
            buf: destination CuPy view.
            file_offset: byte offset in the logical L2 store.
            nbytes: bytes to read; defaults to the full buffer.

        Returns:
            Number of bytes read.

        Async/thread-safety:
            Must not block the worker event loop while storage I/O is pending.
        """
        ...

    async def read_chunk_into_async(
        self,
        chunk_key: str,
        buf: torch.Tensor,
        file_offset: int,
        nbytes: int,
        l2_durable: bool,
        protect_lookup: bool = False,
    ) -> int:
        """Read one logical chunk into a GPU staging tensor.

        Args:
            chunk_key: cache key for the chunk being read.
            buf: destination torch tensor for chunk bytes.
            file_offset: byte offset in the logical L2 store.
            nbytes: bytes to read.
            l2_durable: whether the backend may read from L2 on miss.
            protect_lookup: keep local transfer state protected for the active
                scheduler lookup lease until release_lookup_pins.

        Returns:
            Number of bytes read.

        Async/thread-safety:
            Must not block the worker event loop while storage I/O is pending.
        """
        ...

    def pin_chunks_for_lookup(self, chunk_keys: list[str]) -> None:
        """Protect lookup-hit chunks in transfer-local state.

        Args:
            chunk_keys: cache keys returned by scheduler lookup.

        Async/thread-safety:
            Called on the worker thread before any store throttling for the
            current step. Backends without an L1 cache may no-op.
        """
        ...

    def release_lookup_pins(self, chunk_keys: list[str]) -> None:
        """Release transfer-local lookup protections.

        Args:
            chunk_keys: cache keys whose scheduler lookup lease ended.

        Async/thread-safety:
            Called on the worker thread after corresponding reads are safe.
        """
        ...

    def stats(self) -> TransferStats:
        """Return a snapshot of transfer counters."""
        ...

    def close(self) -> None:
        """Release files, pinned buffers, and backend resources."""
        ...


class BaseTransferLayer:
    """Common transfer-layer accounting helpers.

    Async/thread-safety:
        Counters are mutated by the owning transfer backend. Current connector
        usage serializes mutations on the worker background event loop.
    """

    def __init__(self) -> None:
        self._l2_read_bytes = 0
        self._l2_write_bytes = 0

    def _record_l2_read(self, nbytes: int) -> None:
        """Add bytes to the L2 read counter.

        Args:
            nbytes: number of bytes read from L2.
        """
        self._l2_read_bytes += nbytes

    def _record_l2_write(self, nbytes: int) -> None:
        """Add bytes to the L2 write counter.

        Args:
            nbytes: number of bytes written to L2.
        """
        self._l2_write_bytes += nbytes

    def _base_stats(self) -> TransferStats:
        """Return common L2 transfer counters."""
        return TransferStats(
            l2_read_bytes=self._l2_read_bytes,
            l2_write_bytes=self._l2_write_bytes,
        )
