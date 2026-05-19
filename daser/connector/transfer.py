# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import enum
from typing import Optional, Protocol

# Third Party
import cupy


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

    def stats(self) -> TransferStats:
        """Return a snapshot of transfer counters."""
        ...

    def close(self) -> None:
        """Release files, pinned buffers, and backend resources."""
        ...
