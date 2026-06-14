# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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

    Capability surface:
        Backends advertise optional behavior through attributes and overridable
        methods rather than ad-hoc duck typing. ``coalesce_store_spans`` lets a
        backend opt into adjacent-span coalescing; ``stats`` and
        ``l1_bytes_used`` expose tiering counters; ``drain`` waits for
        background work; ``store_bytes_grouped``/``load_bytes_grouped`` execute
        multi-span batches and default to looping over the single-span methods.

    Async/thread-safety:
        Methods are coroutine-compatible and should be called from the server
        asyncio event loop. Implementations may offload blocking syscalls to an
        executor.
    """

    #: When True the server coalesces adjacent store spans before dispatch.
    coalesce_store_spans: bool = False

    @property
    def stats(self) -> TransferStats:
        """Return tiering counters for this backend.

        Returns:
            The backend's ``_stats`` snapshot when it maintains one, otherwise
            zeroed counters for backends without a memory tier.
        """
        stats = getattr(self, "_stats", None)
        return stats if stats is not None else TransferStats()

    @property
    def l1_bytes_used(self) -> int:
        """Return L1 memory-tier bytes currently resident.

        Returns:
            The backend's ``_l1_used`` counter, or 0 for backends without a
            memory tier.
        """
        return int(getattr(self, "_l1_used", 0))

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

    async def store_bytes_grouped(self, src: Any, spans: list[dict[str, int]]) -> int:
        """Store multiple spans from ``src``.

        Args:
            src: readable buffer or GPU array spanning all source offsets.
            spans: span dicts with ``source_offset``, ``nbytes``, ``file_offset``.

        Returns:
            Total number of bytes stored.

        Async/thread-safety:
            Default implementation loops over ``store_bytes``. Backends may
            override to batch the spans into a single backend call.
        """
        total = 0
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            total += await self.store_bytes(
                src[source_offset : source_offset + nbytes], file_offset, nbytes
            )
        return total

    async def load_bytes_grouped(self, dst: Any, spans: list[dict[str, int]]) -> int:
        """Load multiple spans into ``dst``.

        Args:
            dst: writable buffer or GPU array spanning all target offsets.
            spans: span dicts with ``target_offset``, ``nbytes``, ``file_offset``.

        Returns:
            Total number of bytes loaded.

        Async/thread-safety:
            Default implementation loops over ``load_bytes``. Backends may
            override to batch the spans into a single backend call.
        """
        total = 0
        view = memoryview(dst) if isinstance(dst, (bytearray, bytes)) else dst
        for span in spans:
            target_offset = int(span.get("target_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            total += await self.load_bytes(
                view[target_offset : target_offset + nbytes], file_offset, nbytes
            )
        return total

    async def drain(self) -> None:  # noqa: B027
        """Wait for any background transfer work to complete.

        Async/thread-safety:
            Default implementation is an intentional no-op for backends with no
            background work. Backends with async write-back override this.
        """

    @abstractmethod
    def close(self) -> None:
        """Release file handles and backend resources."""
