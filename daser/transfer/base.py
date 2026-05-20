# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
from typing import Any


class TransferMode(enum.Enum):
    """Server-selected transfer implementation."""

    GDS = "gds"
    IOURING = "iouring"


@dataclass
class TransferStats:
    """Common transfer counters.

    Attributes:
        l1_hits: number of reads served from the memory tier.
        l1_misses: number of reads not found in the memory tier.
        l2_reads: number of reads issued to the SSD tier.
        l2_writes: number of writes issued to the SSD tier.
    """

    l1_hits: int = 0
    l1_misses: int = 0
    l2_reads: int = 0
    l2_writes: int = 0


class TransferLayer(ABC):
    """Abstract server-owned KV transfer layer.

    Concrete implementations hide whether storage is GDS direct-to-GPU or
    io_uring through pinned host memory.

    Async/thread-safety:
        Methods are coroutine-compatible and should be called from the server
        asyncio event loop. Implementations may offload blocking syscalls to an
        executor.
    """

    @abstractmethod
    async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
        """Load bytes into ``dst``.

        Args:
            dst: writable buffer or GPU array.
            file_offset: L2 byte offset.
            nbytes: number of bytes to load.

        Returns:
            Number of bytes loaded.
        """

    @abstractmethod
    async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
        """Store bytes from ``src``.

        Args:
            src: readable buffer or GPU array.
            file_offset: L2 byte offset.
            nbytes: number of bytes to store.

        Returns:
            Number of bytes stored.
        """

    @abstractmethod
    def close(self) -> None:
        """Release file handles and backend resources."""
