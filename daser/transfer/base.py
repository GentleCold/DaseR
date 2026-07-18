# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

TransferTier = Literal["l1", "mixed", "l2"]


@dataclass
class TransferStats:
    """Common transfer counters.

    Attributes:
        l1_hits: number of reads served from the memory tier.
        l1_misses: number of reads not found in the memory tier.
        l2_reads: number of reads issued to the SSD tier.
        l2_writes: number of writes issued to the SSD tier.
        prefetch_requests: number of host-tier prefetch operations.
        prefetch_l1_bytes: requested bytes already resident in L1.
        prefetch_l2_bytes: requested bytes read from L2.
    """

    l1_hits: int = 0
    l1_misses: int = 0
    l2_reads: int = 0
    l2_writes: int = 0
    prefetch_requests: int = 0
    prefetch_l1_bytes: int = 0
    prefetch_l2_bytes: int = 0


@dataclass(frozen=True)
class PrefetchResult:
    """Byte attribution for one host-tier prefetch operation."""

    requested_bytes: int
    l1_bytes: int
    l2_bytes: int


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

    async def prefetch_bytes_grouped(
        self,
        spans: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> PrefetchResult:
        """Promote storage spans into an optional host-memory tier.

        Args:
            spans: Storage spans containing ``file_offset`` and ``nbytes``.
            lease_id: Optional request identifier whose promoted bytes must
                remain stable until a later load consumes them.

        Returns:
            Byte attribution for the request.

        Raises:
            NotImplementedError: If this backend has no host-memory tier.

        Async/thread-safety:
            Implementations run on the server event loop and must await all
            blocking I/O through their normal async backend path.
        """
        raise NotImplementedError("transfer backend does not support host prefetch")

    async def classify_and_acquire_lease(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> TransferTier:
        """Classify host-tier residency and atomically lease an all-L1 window.

        Args:
            lease_id: Request identifier used by later load and release calls.
            spans: Storage spans containing ``file_offset`` and ``nbytes``.

        Returns:
            ``l1`` when every byte is resident, ``mixed`` for partial L1
            residency, or ``l2`` when no requested byte is resident.

        Raises:
            NotImplementedError: If this backend has no host-memory tier.

        Async/thread-safety:
            Classification and all-L1 lease acquisition must be atomic with
            respect to cache eviction and overlapping stores.
        """
        raise NotImplementedError("transfer backend does not support host leases")

    async def load_leased_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
        lease_id: str,
    ) -> int:
        """Load spans from bytes retained by a request lease.

        Args:
            dst: Writable destination buffer.
            spans: Span dicts with target offset, file offset, and byte count.
            lease_id: Request identifier returned through lookup admission.

        Returns:
            Total number of bytes loaded.

        Async/thread-safety:
            Backends with host leases must keep the lease alive after this call;
            the IPC boundary releases ranges only after destination sync.
        """
        return await self.load_bytes_grouped(dst, spans)

    async def release_lease_ranges(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> None:
        """Release successfully staged physical ranges from a request lease.

        Args:
            lease_id: Request identifier owning the retained bytes.
            spans: Physical storage ranges safe to release after staging sync.

        Returns:
            None.

        Async/thread-safety:
            Implementations must make repeated and overlapping releases
            idempotent.
        """
        return None

    async def release_lease(self, lease_id: str) -> None:
        """Release every remaining range owned by a request lease.

        Args:
            lease_id: Request identifier to clean up.

        Returns:
            None.

        Async/thread-safety:
            Implementations must make cleanup idempotent so cancellation can
            race with load completion safely.
        """
        return None

    async def drain(self) -> None:  # noqa: B027
        """Wait for any background transfer work to complete.

        Async/thread-safety:
            Default implementation is an intentional no-op for backends with no
            background work. Backends with async write-back override this.
        """

    @abstractmethod
    def close(self) -> None:
        """Release file handles and backend resources."""
