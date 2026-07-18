# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from dataclasses import dataclass
from typing import Any

# First Party
from daser.logging import init_logger
from daser.transfer.base import (
    PrefetchResult,
    TransferLayer,
    TransferStats,
    TransferTier,
)
from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.l1_cache import L1Cache, L1RangeHit
from daser.transfer.iouring.l2_engine import L2IoEngine
from daser.transfer.iouring.native import NativeIOUring
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice

logger = init_logger(__name__)

_DIRECT_IO_ALIGNMENT = 4096


@dataclass(frozen=True)
class _LeasedSliceRange:
    """Map one physical byte range to bytes retained in a pinned slice."""

    file_offset: int
    nbytes: int
    data: PinnedMemorySlice
    source_offset: int


@dataclass
class _RequestLease:
    """Retain the unconsumed physical ranges for one inference request."""

    remaining: list[tuple[int, int]]
    hits: list[_LeasedSliceRange]
    slice_ids: set[int]
    released: asyncio.Future[None]
    active_loads: int = 0
    release_all: bool = False


def _normalize_ranges(spans: list[dict[str, int]]) -> list[tuple[int, int]]:
    """Return sorted, merged positive physical ranges from transfer spans."""
    ranges = sorted(
        (int(span["file_offset"]), int(span["nbytes"]))
        for span in spans
        if int(span["nbytes"]) > 0
    )
    merged: list[tuple[int, int]] = []
    for start, size in ranges:
        end = start + size
        if merged and start <= merged[-1][0] + merged[-1][1]:
            previous_start, previous_size = merged[-1]
            merged[-1] = (
                previous_start,
                max(previous_start + previous_size, end) - previous_start,
            )
        else:
            merged.append((start, size))
    return merged


def _subtract_ranges(
    ranges: list[tuple[int, int]],
    removals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Subtract physical ranges, preserving any non-overlapping fragments."""
    result = list(ranges)
    for remove_start, remove_size in removals:
        remove_end = remove_start + remove_size
        next_result: list[tuple[int, int]] = []
        for start, size in result:
            end = start + size
            if end <= remove_start or remove_end <= start:
                next_result.append((start, size))
                continue
            if start < remove_start:
                next_result.append((start, remove_start - start))
            if remove_end < end:
                next_result.append((remove_end, end - remove_end))
        result = next_result
    return result


def _subtract_leased_hits(
    hits: list[_LeasedSliceRange],
    removals: list[tuple[int, int]],
) -> list[_LeasedSliceRange]:
    """Subtract consumed ranges while preserving pinned-slice source offsets."""
    result = list(hits)
    for remove_start, remove_size in removals:
        remove_end = remove_start + remove_size
        next_result: list[_LeasedSliceRange] = []
        for hit in result:
            start = hit.file_offset
            end = start + hit.nbytes
            if end <= remove_start or remove_end <= start:
                next_result.append(hit)
                continue
            if start < remove_start:
                next_result.append(
                    _LeasedSliceRange(
                        file_offset=start,
                        nbytes=remove_start - start,
                        data=hit.data,
                        source_offset=hit.source_offset,
                    )
                )
            if remove_end < end:
                next_result.append(
                    _LeasedSliceRange(
                        file_offset=remove_end,
                        nbytes=end - remove_end,
                        data=hit.data,
                        source_offset=hit.source_offset + remove_end - start,
                    )
                )
        result = next_result
    return result


class TieredIOUringTransferLayer(TransferLayer):
    """Async L1 pinned-memory + L2 SSD transfer layer.

    The implementation uses Linux io_uring for L2 positioned file I/O. GPU
    CUDA-IPC copies are added at the server IPC boundary; this layer owns the
    tiering and L2 persistence policy.

    Args:
        path: pre-allocated or creatable L2 store file.
        l1_bytes: maximum memory-tier bytes.
        l2_bytes: SSD-tier capacity.
        io_workers: number of native io_uring rings and executor threads used
            for L2 operations.

    Async/thread-safety:
        Public async methods serialize tier metadata with an asyncio lock.
        io_uring completion waits are offloaded with ``run_in_executor``.
    """

    coalesce_store_spans = True

    def __init__(
        self,
        path: str,
        l1_bytes: int,
        l2_bytes: int,
        io_workers: int = 8,
        skip_l2: bool = False,
    ) -> None:
        if l1_bytes <= 0:
            raise ValueError("l1_bytes must be positive")
        if not skip_l2 and l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if not skip_l2 and l1_bytes > l2_bytes:
            raise ValueError("l1_bytes must not exceed l2_bytes")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        self._l2: L2IoEngine | None = None
        if not skip_l2:
            self._l2 = L2IoEngine(path, l2_bytes, io_workers)
        self._l1_bytes = l1_bytes
        self._l2_bytes = l2_bytes
        self._pending_l2: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._pending_l2_buffers: dict[tuple[int, int], PinnedMemorySlice] = {}
        self._pending_l1_promotions: dict[int, asyncio.Future[None]] = {}
        self._pending_l1_promotion_epochs: dict[int, int] = {}
        self._cache_epoch = 0
        self._cache_mutations: list[tuple[int, int, int]] = []
        self._request_leases: dict[str, _RequestLease] = {}
        self._leased_slices: dict[int, tuple[PinnedMemorySlice, int]] = {}
        self._l1 = L1Cache(
            l1_bytes,
            alignment=_DIRECT_IO_ALIGNMENT,
            pinned_predicate=self._is_slice_pinned,
        )
        self._l2_errors: list[BaseException] = []
        self._lock = asyncio.Lock()
        self._stats = TransferStats()
        logger.info(
            "[TRANSFER:iouring] path=%s l1=%d l2=%d direct_io=%s "
            "io_workers=%d skip_l2=%s",
            path,
            l1_bytes,
            l2_bytes,
            not skip_l2,
            io_workers,
            skip_l2,
        )

    async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
        """Load bytes into a writable destination buffer.

        Args:
            dst: writable byte buffer.
            file_offset: byte offset in L2.
            nbytes: number of bytes to load.

        Returns:
            Number of bytes loaded.
        """
        self._check_range(file_offset, nbytes)
        pending: list[asyncio.Task[None]] = []
        l1_hits: list[L1RangeHit] = []
        misses: list[dict[str, int]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            l1_hits, misses = self._l1.resolve_subranges(
                target_offset=0,
                file_offset=file_offset,
                nbytes=nbytes,
            )
            if self._l2 is None and misses:
                self._stats.l1_misses += len(misses)
                raise KeyError(
                    "skip_l2 cache miss for range "
                    f"[{file_offset}, {file_offset + nbytes})"
                )
            if l1_hits:
                self._l1.record_hits(l1_hits)
                self._stats.l1_hits += 1 if self._l2 is None else len(l1_hits)
                self._copy_grouped_to_dst(
                    dst,
                    [
                        (
                            hit.target_offset,
                            hit.data,
                            hit.source_offset,
                            hit.nbytes,
                        )
                        for hit in l1_hits
                    ],
                )
            for miss in misses:
                self._stats.l1_misses += 1
                pending.extend(
                    self._find_pending_l2_locked(
                        int(miss["file_offset"]),
                        int(miss["nbytes"]),
                    )
                )

        if pending:
            await asyncio.gather(*pending)
        if misses:
            await self._load_l2_misses_grouped(dst, misses)
        return nbytes

    async def load_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
    ) -> int:
        """Load multiple spans, coalescing L1 hits into one destination copy.

        Args:
            dst: writable byte buffer.
            spans: span dicts with target_offset, file_offset, and nbytes.

        Returns:
            Number of bytes loaded.

        Async/thread-safety:
            Uses the same metadata lock as ``load_bytes``. L2 misses still use
            native io_uring through the executor.
        """
        total = 0
        merged_l1: list[tuple[int, PinnedMemorySlice, int, int]] = []
        misses: list[dict[str, int]] = []
        pending: list[asyncio.Task[None]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            for span in spans:
                target_offset = int(span.get("target_offset", 0))
                file_offset = int(span["file_offset"])
                nbytes = int(span["nbytes"])
                self._check_range(file_offset, nbytes)
                total += nbytes
                l1_hits, span_misses = self._l1.resolve_subranges(
                    target_offset=target_offset,
                    file_offset=file_offset,
                    nbytes=nbytes,
                )
                if self._l2 is None and span_misses:
                    self._stats.l1_misses += 1
                    raise KeyError(
                        "skip_l2 cache miss for range "
                        f"[{file_offset}, {file_offset + nbytes})"
                    )
                if l1_hits:
                    self._l1.record_hits(l1_hits)
                    self._stats.l1_hits += 1 if self._l2 is None else len(l1_hits)
                    merged_l1.extend(
                        (
                            hit.target_offset,
                            hit.data,
                            hit.source_offset,
                            hit.nbytes,
                        )
                        for hit in l1_hits
                    )
                for miss in span_misses:
                    self._stats.l1_misses += 1
                    pending.extend(
                        self._find_pending_l2_locked(
                            int(miss["file_offset"]),
                            int(miss["nbytes"]),
                        )
                    )
                    misses.append(
                        {
                            "target_offset": int(miss["target_offset"]),
                            "file_offset": int(miss["file_offset"]),
                            "nbytes": int(miss["nbytes"]),
                        }
                    )
            if merged_l1:
                self._copy_grouped_to_dst(dst, merged_l1)

        if pending:
            await asyncio.gather(*pending)
        if misses:
            await self._load_l2_misses_grouped(dst, misses)
        return total

    async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
        """Store bytes into L1 immediately and schedule L2 persistence.

        Args:
            src: readable byte buffer.
            file_offset: byte offset in L2.
            nbytes: number of bytes to store.

        Returns:
            Number of bytes stored.
        """
        self._check_range(file_offset, nbytes)
        key = (file_offset, nbytes)
        if self._l2 is None:
            while True:
                async with self._lock:
                    waiters = self._overlapping_lease_waiters_locked(
                        file_offset, nbytes
                    )
                    if not waiters:
                        hit = self._l1.find(file_offset)
                        if hit is not None:
                            hit_key, cached, target_offset = hit
                            if target_offset + nbytes <= len(cached):
                                self._copy_src_to_pinned_at(
                                    src, cached, target_offset, nbytes
                                )
                                self._l1.touch(hit_key)
                                self._record_cache_mutation_locked(file_offset, nbytes)
                                return nbytes
                        data = self._l1.reserve_or_raise(
                            key,
                            nbytes,
                            preserve_overlaps=True,
                        )
                        try:
                            self._copy_src_to_pinned_at(src, data, 0, nbytes)
                        except BaseException:
                            data.close()
                            raise
                        self._record_cache_mutation_locked(file_offset, nbytes)
                        self._l1.put(key, data)
                        return nbytes
                await asyncio.gather(*waiters)

        await self._wait_for_overlapping_leases(file_offset, nbytes)
        data = await self._reserve_l1_buffer(key, nbytes)
        try:
            self._copy_src_to_pinned(src, data, nbytes)
        except BaseException:
            data.close()
            raise
        try:
            while True:
                async with self._lock:
                    self._raise_l2_error_locked()
                    waiters = self._overlapping_lease_waiters_locked(
                        file_offset, nbytes
                    )
                    if not waiters:
                        previous = self._find_pending_l2_locked(file_offset, nbytes)
                        self._record_cache_mutation_locked(file_offset, nbytes)
                        self._l1.put(key, data)
                        task = self._schedule_l2_write_locked(
                            key,
                            file_offset,
                            data,
                            previous,
                        )
                        self._pending_l2[key] = task
                        self._pending_l2_buffers[key] = data
                        return nbytes
                await asyncio.gather(*waiters)
        except BaseException:
            async with self._lock:
                if not self._l1.contains_slice(data):
                    data.close()
                    self._l1.notify_pool_waiters()
            raise

    async def store_bytes_grouped(
        self,
        src: Any,
        spans: list[dict[str, Any]],
    ) -> int:
        """Store multiple spans through the L1/L2 tiering path.

        Args:
            src: readable byte buffer or CuPy ndarray.
            spans: span dicts with source_offset, file_offset, and nbytes.

        Returns:
            Total number of bytes stored.

        Async/thread-safety:
            In normal tiered mode this uses ``store_bytes`` for each span so L1
            visibility and asynchronous L2 scheduling keep the same ordering
            guarantees as single-span stores. In ``skip_l2`` mode it serializes
            L1 metadata once for the whole group because no pending L2 writer
            can retain evicted pool slices.
        """
        if self._l2 is None:
            return await self._store_bytes_grouped_l1_only(src, spans)

        total = 0
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            source = self._slice_src(src, source_offset, nbytes)
            total += await self.store_bytes(source, file_offset, nbytes)
        return total

    async def prefetch_bytes_grouped(
        self,
        spans: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> PrefetchResult:
        """Promote L2-missing portions of spans into the pinned L1 tier.

        Args:
            spans: Aligned storage spans containing ``file_offset`` and
                ``nbytes``.
            lease_id: Optional request identifier that blocks overlapping
                stores and retains promoted slices for the later GPU load.

        Returns:
            Requested bytes split between existing L1 data and L2 reads.

        Raises:
            NotImplementedError: If the L2 tier is disabled.

        Async/thread-safety:
            Metadata is protected by the transfer lock; io_uring reads are
            awaited through the existing executor-backed read path.
        """
        if self._l2 is None:
            raise NotImplementedError("prefetch requires the io_uring L2 tier")

        requested_ranges = _normalize_ranges(spans)
        requested_bytes = sum(size for _start, size in requested_ranges)
        if lease_id is not None and requested_bytes > self._l1_bytes:
            raise MemoryError(
                f"request lease needs {requested_bytes} bytes but L1 capacity is "
                f"{self._l1_bytes}"
            )
        l1_bytes = 0
        misses: list[dict[str, int]] = []
        pending: list[asyncio.Task[None]] = []
        try:
            async with self._lock:
                self._raise_l2_error_locked()
                if lease_id is not None:
                    self._replace_request_lease_locked(lease_id, requested_ranges, [])
                for file_offset, nbytes in requested_ranges:
                    self._check_range(file_offset, nbytes)
                    l1_hits, span_misses = self._l1.resolve_subranges(
                        target_offset=0,
                        file_offset=file_offset,
                        nbytes=nbytes,
                    )
                    if l1_hits:
                        self._l1.record_hits(l1_hits)
                        l1_bytes += sum(hit.nbytes for hit in l1_hits)
                        if lease_id is not None:
                            self._attach_l1_hits_to_lease_locked(lease_id, l1_hits)
                    for miss in span_misses:
                        pending.extend(
                            self._find_pending_l2_locked(
                                int(miss["file_offset"]),
                                int(miss["nbytes"]),
                            )
                        )
                        misses.append(
                            {
                                "target_offset": 0,
                                "file_offset": int(miss["file_offset"]),
                                "nbytes": int(miss["nbytes"]),
                            }
                        )

            if pending:
                await asyncio.gather(*set(pending))
            if misses:
                await self._load_l2_misses_grouped(None, misses, lease_id=lease_id)
            l2_bytes = sum(int(miss["nbytes"]) for miss in misses)
            async with self._lock:
                if lease_id is not None:
                    self._require_complete_lease_locked(lease_id)
                self._stats.prefetch_requests += 1
                self._stats.prefetch_l1_bytes += l1_bytes
                self._stats.prefetch_l2_bytes += l2_bytes
            return PrefetchResult(requested_bytes, l1_bytes, l2_bytes)
        except BaseException:
            if lease_id is not None:
                await self.release_lease(lease_id)
            raise

    async def classify_and_acquire_lease(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> TransferTier:
        """Classify exact spans and atomically retain an all-L1 request window."""
        requested_ranges = _normalize_ranges(spans)
        if not requested_ranges:
            return "l2"
        async with self._lock:
            self._raise_l2_error_locked()
            existing = self._request_leases.get(lease_id)
            if existing is not None:
                if existing.remaining == requested_ranges:
                    return "l1"
                self._release_request_lease_locked(lease_id)

            hits: list[L1RangeHit] = []
            l1_bytes = 0
            requested_bytes = 0
            has_miss = False
            for file_offset, nbytes in requested_ranges:
                self._check_range(file_offset, nbytes)
                requested_bytes += nbytes
                span_hits, misses = self._l1.resolve_subranges(
                    target_offset=0,
                    file_offset=file_offset,
                    nbytes=nbytes,
                )
                hits.extend(span_hits)
                l1_bytes += sum(hit.nbytes for hit in span_hits)
                has_miss = has_miss or bool(misses)
            if not has_miss:
                self._l1.record_hits(hits)
                self._replace_request_lease_locked(lease_id, requested_ranges, hits)
                return "l1"
            return "mixed" if l1_bytes else "l2"

    async def load_leased_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
        lease_id: str,
    ) -> int:
        """Copy request-leased L1 bytes without re-resolving cache metadata."""
        total = 0
        chunks: list[tuple[int, PinnedMemorySlice, int, int]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            lease = self._request_leases.get(lease_id)
            if lease is None:
                raise KeyError(f"unknown transfer lease: {lease_id}")
            lease.active_loads += 1
            try:
                for span in spans:
                    target_offset = int(span.get("target_offset", 0))
                    file_offset = int(span["file_offset"])
                    nbytes = int(span["nbytes"])
                    self._check_range(file_offset, nbytes)
                    total += nbytes
                    chunks.extend(
                        self._resolve_leased_range_locked(
                            lease,
                            target_offset,
                            file_offset,
                            nbytes,
                        )
                    )
                if chunks:
                    self._stats.l1_hits += len(chunks)
                    self._copy_grouped_to_dst(dst, chunks)
            except BaseException:
                lease.active_loads -= 1
                if lease.release_all and lease.active_loads == 0:
                    self._release_request_lease_locked(lease_id, force=True)
                raise
        return total

    async def release_lease_ranges(
        self,
        lease_id: str,
        spans: list[dict[str, int]],
    ) -> None:
        """Release staged physical ranges after the IPC destination is synced."""
        removals = _normalize_ranges(spans)
        async with self._lock:
            lease = self._request_leases.get(lease_id)
            if lease is None:
                return
            if lease.active_loads > 0:
                lease.active_loads -= 1
            if lease.release_all:
                if lease.active_loads == 0:
                    self._release_request_lease_locked(lease_id, force=True)
                return
            self._consume_lease_ranges_locked(lease_id, removals)

    async def release_lease(self, lease_id: str) -> None:
        """Idempotently release every remaining range for one request."""
        async with self._lock:
            self._release_request_lease_locked(lease_id)

    async def drain(self) -> None:
        """Wait until all pending L2 writes have completed.

        Async/thread-safety:
            Must be called from the owning asyncio event loop before shutdown
            when durable L2 contents are required.
        """
        if self._l2 is None:
            return None
        while True:
            async with self._lock:
                pending = list(self._pending_l2.values())
                self._raise_l2_error_locked()
            if not pending:
                return
            await asyncio.gather(*pending)

    def close(self) -> None:
        """Close the L2 file handle after pending writes are drained/cancelled."""
        for lease in self._request_leases.values():
            if not lease.released.done():
                lease.released.set_result(None)
        self._request_leases.clear()
        self._leased_slices.clear()
        for task in self._pending_l2.values():
            task.cancel()
        self._pending_l2.clear()
        for pending_buffer in self._pending_l2_buffers.values():
            if not self._l1.contains_slice(pending_buffer):
                pending_buffer.close()
        self._pending_l2_buffers.clear()
        if self._l2 is not None:
            self._l2.close()
        self._l1.close()

    def _is_slice_pinned(self, key: tuple[int, int], data: PinnedMemorySlice) -> bool:
        """Return whether an L2 writer or request lease still owns ``data``."""
        return (
            self._pending_l2_buffers.get(key) is data or id(data) in self._leased_slices
        )

    @property
    def l1_bytes_used(self) -> int:
        """Return resident L1 memory-tier bytes."""
        return self._l1.bytes_used

    def _check_range(self, file_offset: int, nbytes: int) -> None:
        """Validate an L2 byte range.

        Args:
            file_offset: byte offset in L2.
            nbytes: byte count.

        Raises:
            ValueError: when the range is invalid or exceeds L2.
        """
        if file_offset < 0 or nbytes < 0:
            raise ValueError("file_offset and nbytes must be non-negative")
        if self._l2 is None:
            if nbytes > self._l1_bytes:
                raise ValueError(
                    f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
                )
            return
        if file_offset + nbytes > self._l2_bytes:
            raise ValueError(
                f"range [{file_offset}, {file_offset + nbytes}) exceeds "
                f"L2 capacity {self._l2_bytes}"
            )
        if (
            file_offset % _DIRECT_IO_ALIGNMENT != 0
            or nbytes % _DIRECT_IO_ALIGNMENT != 0
        ):
            raise ValueError(
                "O_DIRECT io_uring ranges must be aligned to "
                f"{_DIRECT_IO_ALIGNMENT} bytes: offset={file_offset} nbytes={nbytes}"
            )

    def _find_pending_l2_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> list[asyncio.Task[None]]:
        """Return pending L2 writes overlapping the requested byte span."""
        end = file_offset + nbytes
        return [
            task
            for key, task in self._pending_l2.items()
            if key[0] < end and file_offset < key[0] + key[1]
        ]

    def _read_l2_into(
        self,
        file_offset: int,
        dst: PinnedMemorySlice,
        uring: NativeIOUring,
    ) -> int:
        """Blocking io_uring L2 read into pinned memory."""
        if self._l2 is None:
            raise RuntimeError("L2 reads are disabled when skip_l2 is true")
        return self._l2.read_into(file_offset, dst.view(), uring)

    def _write_l2(
        self,
        file_offset: int,
        data: PinnedMemorySlice,
        uring: NativeIOUring,
    ) -> None:
        """Blocking io_uring L2 write."""
        if self._l2 is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        written = self._l2.write(file_offset, data.view(), uring)
        if written != len(data):
            raise IOError(f"short io_uring write: {written} != {len(data)}")

    def _schedule_l2_write_locked(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: PinnedMemorySlice,
        previous: list[asyncio.Task[None]],
    ) -> asyncio.Task[None]:
        """Schedule one L2 write and start independent IO immediately."""
        if self._l2 is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        if previous:
            return asyncio.create_task(
                self._write_l2_async(key, file_offset, data, previous)
            )

        loop = asyncio.get_event_loop()
        uring = self._next_uring()
        future = loop.run_in_executor(
            self._l2.executor,
            self._write_l2,
            file_offset,
            data,
            uring,
        )
        return asyncio.create_task(self._track_l2_write(key, future))

    async def _track_l2_write(
        self,
        key: tuple[int, int],
        future: asyncio.Future[None],
    ) -> None:
        """Publish completion for an already-submitted L2 write."""
        current = asyncio.current_task()
        try:
            await future
            async with self._lock:
                self._stats.l2_writes += 1
        except BaseException as exc:
            async with self._lock:
                self._l2_errors.append(exc)
            raise
        finally:
            async with self._lock:
                if self._pending_l2.get(key) is current:
                    self._pending_l2.pop(key, None)
                    pending_buffer = self._pending_l2_buffers.pop(key, None)
                    if pending_buffer is not None:
                        self._close_unowned_slice_locked(pending_buffer)

    async def _write_l2_async(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: PinnedMemorySlice,
        previous: list[asyncio.Task[None]],
    ) -> None:
        """Persist one L2 write and publish completion to waiters."""
        current = asyncio.current_task()
        try:
            if previous:
                await asyncio.gather(*previous)
            loop = asyncio.get_event_loop()
            if self._l2 is None:
                raise RuntimeError("L2 writes are disabled when skip_l2 is true")
            await loop.run_in_executor(
                self._l2.executor,
                self._write_l2,
                file_offset,
                data,
                self._next_uring(),
            )
            async with self._lock:
                self._stats.l2_writes += 1
        except BaseException as exc:
            async with self._lock:
                self._l2_errors.append(exc)
            raise
        finally:
            async with self._lock:
                if self._pending_l2.get(key) is current:
                    self._pending_l2.pop(key, None)
                    pending_buffer = self._pending_l2_buffers.pop(key, None)
                    if pending_buffer is not None:
                        self._close_unowned_slice_locked(pending_buffer)

    def _raise_l2_error_locked(self) -> None:
        """Raise and clear the first asynchronous L2 write failure."""
        if not self._l2_errors:
            return
        error = self._l2_errors.pop(0)
        raise RuntimeError("asynchronous io_uring L2 write failed") from error

    async def _load_l2_misses_grouped(
        self,
        dst: Any | None,
        misses: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> None:
        """Read grouped L2 misses concurrently, then promote in request order."""
        start = 0
        while start < len(misses):
            batch = self._next_l2_miss_batch(misses, start)
            await self._load_l2_miss_batch(dst, batch, lease_id=lease_id)
            start += len(batch)

    async def _load_l2_miss_batch(
        self,
        dst: Any | None,
        misses: list[dict[str, int]],
        lease_id: str | None = None,
    ) -> None:
        """Read one bounded L2 miss batch and promote it to L1."""
        loop = asyncio.get_event_loop()
        if self._l2 is None:
            raise RuntimeError("L2 reads are disabled when skip_l2 is true")
        reads: list[tuple[dict[str, int], PinnedMemorySlice, int, int]] = []
        future_to_read: dict[
            asyncio.Future[int],
            tuple[dict[str, int], PinnedMemorySlice, int, int],
        ] = {}
        try:
            for span in misses:
                nbytes = int(span["nbytes"])
                key = (int(span["file_offset"]), nbytes)
                (
                    pinned,
                    promotion_id,
                    epoch,
                    pending_writes,
                ) = await self._reserve_l1_promotion(key, nbytes)
                reads.append((span, pinned, promotion_id, epoch))
                if pending_writes:
                    await asyncio.gather(*set(pending_writes))
                future = loop.run_in_executor(
                    self._l2.executor,
                    self._read_l2_into,
                    int(span["file_offset"]),
                    pinned,
                    self._next_uring(),
                )
                future_to_read[future] = (span, pinned, promotion_id, epoch)

            pending: set[asyncio.Future[int]] = set(future_to_read)
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                    span, pinned, promotion_id, epoch = future_to_read[future]
                    if dst is not None:
                        self._copy_grouped_to_dst(
                            dst,
                            [
                                (
                                    int(span["target_offset"]),
                                    pinned,
                                    0,
                                    int(span["nbytes"]),
                                )
                            ],
                        )
                    async with self._lock:
                        self._raise_l2_error_locked()
                        nbytes = int(span["nbytes"])
                        key = (int(span["file_offset"]), nbytes)
                        self._stats.l2_reads += 1
                        stale = self._promotion_is_stale_locked(
                            int(span["file_offset"]), nbytes, epoch
                        )
                        if stale:
                            pinned.close()
                            self._l1.notify_pool_waiters()
                        else:
                            self._l1.put(key, pinned)
                            if lease_id is not None:
                                self._attach_l1_hits_to_lease_locked(
                                    lease_id,
                                    [
                                        L1RangeHit(
                                            target_offset=0,
                                            key=key,
                                            data=pinned,
                                            source_offset=0,
                                            nbytes=nbytes,
                                        )
                                    ],
                                )
                        self._finish_l1_promotion_locked(promotion_id)
        finally:
            if future_to_read:
                await asyncio.gather(*future_to_read, return_exceptions=True)
            async with self._lock:
                live_buffers = self._l1.resident_slice_ids() | set(self._leased_slices)
                for _span, pinned, promotion_id, _epoch in reads:
                    self._finish_l1_promotion_locked(promotion_id)
                    if id(pinned) not in live_buffers:
                        pinned.close()
                        self._l1.notify_pool_waiters()

    def _next_l2_miss_batch(
        self,
        misses: list[dict[str, int]],
        start: int,
    ) -> list[dict[str, int]]:
        """Return a miss batch bounded by L1 capacity and io_uring count."""
        ring_count = self._l2.ring_count if self._l2 is not None else 1
        batch: list[dict[str, int]] = []
        batch_bytes = 0
        for span in misses[start:]:
            nbytes = int(span["nbytes"])
            if batch and (
                batch_bytes + nbytes > self._l1_bytes or len(batch) >= ring_count
            ):
                break
            batch.append(span)
            batch_bytes += nbytes
        return batch

    def _next_uring(self) -> NativeIOUring:
        """Return the next native io_uring instance for one L2 operation."""
        if self._l2 is None:
            raise RuntimeError("io_uring rings are disabled when skip_l2 is true")
        return self._l2.next_uring()

    def _copy_grouped_to_dst(
        self,
        dst: Any,
        chunks: list[tuple[int, PinnedMemorySlice, int, int]],
    ) -> None:
        """Copy source chunks into the destination without staging repacks."""
        copy_ops.copy_grouped_to_dst(dst, chunks)

    def _slice_src(self, src: Any, offset: int, nbytes: int) -> Any:
        """Return a readable source slice."""
        return copy_ops.slice_src(src, offset, nbytes)

    def _copy_src_to_pinned(
        self,
        src: Any,
        pinned: PinnedMemorySlice,
        nbytes: int,
    ) -> None:
        """Copy bytes from a CPU or CuPy source into pinned host memory."""
        copy_ops.copy_src_to_pinned(src, pinned, 0, nbytes)

    def _copy_src_to_pinned_at(
        self,
        src: Any,
        pinned: PinnedMemorySlice,
        target_offset: int,
        nbytes: int,
    ) -> None:
        """Copy bytes from a CPU or CUDA source into pinned host memory."""
        copy_ops.copy_src_to_pinned(src, pinned, target_offset, nbytes)

    async def _store_bytes_grouped_l1_only(
        self,
        src: Any,
        spans: list[dict[str, Any]],
    ) -> int:
        """Store grouped spans in L1 without scheduling L2 persistence."""
        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                waiters = {
                    waiter
                    for span in spans
                    for waiter in self._overlapping_lease_waiters_locked(
                        int(span["file_offset"]),
                        int(span["nbytes"]),
                    )
                }
                if not waiters:
                    total = 0
                    for span in spans:
                        source_offset = int(span.get("source_offset", 0))
                        nbytes = int(span["nbytes"])
                        file_offset = int(span["file_offset"])
                        self._check_range(file_offset, nbytes)
                        key = (file_offset, nbytes)
                        hit = self._l1.find(file_offset)
                        source = self._slice_src(src, source_offset, nbytes)
                        if hit is not None:
                            hit_key, cached, target_offset = hit
                            if target_offset + nbytes <= len(cached):
                                self._copy_src_to_pinned_at(
                                    source, cached, target_offset, nbytes
                                )
                                self._l1.touch(hit_key)
                                self._record_cache_mutation_locked(file_offset, nbytes)
                                total += nbytes
                                continue
                        data = self._l1.reserve_or_raise(
                            key,
                            nbytes,
                            preserve_overlaps=True,
                        )
                        try:
                            self._copy_src_to_pinned_at(source, data, 0, nbytes)
                        except BaseException:
                            data.close()
                            raise
                        self._record_cache_mutation_locked(file_offset, nbytes)
                        self._l1.put(key, data)
                        total += nbytes
                    return total
            await asyncio.gather(*waiters)

    async def _reserve_l1_buffer(
        self,
        key: tuple[int, int],
        nbytes: int,
    ) -> PinnedMemorySlice:
        """Reserve pinned L1 space, waiting for pending L2 victims if needed.

        Args:
            key: L1 byte-range key being inserted.
            nbytes: Number of logical bytes needed.

        Returns:
            A pinned slice leased from the preallocated pool.

        Async/thread-safety:
            Mutates L1 metadata under the transfer lock. If evicted victims are
            still owned by pending L2 writes, releases the lock and waits for
            those writes to free their pool slices.
        """
        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                data = self._l1.reserve(key, nbytes)
                if data is not None:
                    return data
                # Pool exhausted with no evictable victim: an in-flight L2
                # write still pins a slice. Wait for one to finish, then retry.
                wait_for: asyncio.Future[None] | None = next(
                    (task for task in self._pending_l2.values() if not task.done()),
                    None,
                )
                if wait_for is None:
                    wait_for = next(
                        (
                            future
                            for future in self._pending_l1_promotions.values()
                            if not future.done()
                        ),
                        None,
                    )
                if wait_for is None:
                    wait_for = asyncio.get_event_loop().create_future()
                    self._l1.register_pool_waiter(wait_for)
            await wait_for

    async def _reserve_l1_promotion(
        self,
        key: tuple[int, int],
        nbytes: int,
    ) -> tuple[
        PinnedMemorySlice,
        int,
        int,
        list[asyncio.Task[None]],
    ]:
        """Reserve and register a pinned slice for one L2 promotion.

        Args:
            key: L2 range being promoted.
            nbytes: Number of bytes to reserve.

        Returns:
            The pinned slice, reservation identifier, cache mutation epoch, and
            L2 writes that must finish before the read starts.

        Async/thread-safety:
            The reservation and its waiter are registered while the transfer
            lock is held. Other stores therefore cannot mistake an in-flight
            promotion for free pool capacity.
        """
        while True:
            async with self._lock:
                self._raise_l2_error_locked()
                data = self._l1.reserve(key, nbytes)
                if data is not None:
                    promotion_id = id(data)
                    self._pending_l1_promotions[promotion_id] = (
                        asyncio.get_event_loop().create_future()
                    )
                    self._pending_l1_promotion_epochs[promotion_id] = self._cache_epoch
                    pending_writes = self._find_pending_l2_locked(key[0], key[1])
                    return (
                        data,
                        promotion_id,
                        self._cache_epoch,
                        pending_writes,
                    )
                wait_for: asyncio.Future[None] | None = next(
                    (task for task in self._pending_l2.values() if not task.done()),
                    None,
                )
                if wait_for is None:
                    wait_for = next(
                        (
                            future
                            for future in self._pending_l1_promotions.values()
                            if not future.done()
                        ),
                        None,
                    )
                if wait_for is None:
                    wait_for = asyncio.get_event_loop().create_future()
                    self._l1.register_pool_waiter(wait_for)
            await wait_for

    def _record_cache_mutation_locked(self, file_offset: int, nbytes: int) -> None:
        """Record a newly published store for concurrent promotion checks."""
        self._cache_epoch += 1
        self._cache_mutations.append(
            (self._cache_epoch, file_offset, file_offset + nbytes)
        )

    def _promotion_is_stale_locked(
        self,
        file_offset: int,
        nbytes: int,
        epoch: int,
    ) -> bool:
        """Return whether a newer overlapping store invalidated a promotion."""
        end = file_offset + nbytes
        if self._l1.has_overlap(file_offset, nbytes):
            return True
        return any(
            mutation_epoch > epoch
            and mutation_start < end
            and file_offset < mutation_end
            for mutation_epoch, mutation_start, mutation_end in self._cache_mutations
        )

    def _finish_l1_promotion_locked(self, promotion_id: int) -> None:
        """Release one promotion reservation and wake blocked pool users."""
        waiter = self._pending_l1_promotions.pop(promotion_id, None)
        self._pending_l1_promotion_epochs.pop(promotion_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
        if self._pending_l1_promotion_epochs:
            oldest_epoch = min(self._pending_l1_promotion_epochs.values())
            self._cache_mutations = [
                mutation
                for mutation in self._cache_mutations
                if mutation[0] > oldest_epoch
            ]
        else:
            self._cache_mutations.clear()

    def _replace_request_lease_locked(
        self,
        lease_id: str,
        ranges: list[tuple[int, int]],
        hits: list[L1RangeHit],
    ) -> None:
        """Replace one request lease while the transfer metadata lock is held."""
        if not lease_id:
            raise ValueError("lease_id must not be empty")
        existing = self._request_leases.get(lease_id)
        if existing is not None and existing.active_loads:
            raise RuntimeError(f"cannot replace active transfer lease: {lease_id}")
        self._release_request_lease_locked(lease_id)
        self._request_leases[lease_id] = _RequestLease(
            remaining=list(ranges),
            hits=[],
            slice_ids=set(),
            released=asyncio.get_running_loop().create_future(),
        )
        self._attach_l1_hits_to_lease_locked(lease_id, hits)

    def _attach_l1_hits_to_lease_locked(
        self,
        lease_id: str,
        hits: list[L1RangeHit],
    ) -> None:
        """Retain resident slice references for one admitted request."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            raise RuntimeError(
                f"transfer lease was released during prefetch: {lease_id}"
            )
        new_ids: set[int] = set()
        for hit in hits:
            lease.hits.append(
                _LeasedSliceRange(
                    file_offset=hit.key[0] + hit.source_offset,
                    nbytes=hit.nbytes,
                    data=hit.data,
                    source_offset=hit.source_offset,
                )
            )
            data_id = id(hit.data)
            if data_id not in lease.slice_ids:
                new_ids.add(data_id)
                lease.slice_ids.add(data_id)
        lease.hits.sort(key=lambda item: item.file_offset)
        for data_id in new_ids:
            data = next(hit.data for hit in hits if id(hit.data) == data_id)
            existing = self._leased_slices.get(data_id)
            if existing is None:
                self._leased_slices[data_id] = (data, 1)
            else:
                self._leased_slices[data_id] = (existing[0], existing[1] + 1)

    def _require_complete_lease_locked(self, lease_id: str) -> None:
        """Raise unless retained slice ranges cover the whole request lease."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            raise RuntimeError(
                f"transfer lease was released during prefetch: {lease_id}"
            )
        uncovered = list(lease.remaining)
        for hit in lease.hits:
            uncovered = _subtract_ranges(
                uncovered,
                [(hit.file_offset, hit.nbytes)],
            )
        if uncovered:
            raise RuntimeError(f"prefetch did not retain complete lease: {lease_id}")

    def _resolve_leased_range_locked(
        self,
        lease: _RequestLease,
        target_offset: int,
        file_offset: int,
        nbytes: int,
    ) -> list[tuple[int, PinnedMemorySlice, int, int]]:
        """Resolve one load span exclusively through retained lease references."""
        chunks: list[tuple[int, PinnedMemorySlice, int, int]] = []
        cursor = file_offset
        end = file_offset + nbytes
        while cursor < end:
            covering = next(
                (
                    hit
                    for hit in lease.hits
                    if hit.file_offset <= cursor < hit.file_offset + hit.nbytes
                ),
                None,
            )
            if covering is None:
                raise KeyError(
                    f"request lease does not cover load range [{file_offset}, {end})"
                )
            covered = min(end, covering.file_offset + covering.nbytes) - cursor
            chunks.append(
                (
                    target_offset + cursor - file_offset,
                    covering.data,
                    covering.source_offset + cursor - covering.file_offset,
                    covered,
                )
            )
            cursor += covered
        return chunks

    def _consume_lease_ranges_locked(
        self,
        lease_id: str,
        removals: list[tuple[int, int]],
    ) -> None:
        """Remove staged ranges and release slice references no longer needed."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            return
        old_slice_ids = set(lease.slice_ids)
        lease.remaining = _subtract_ranges(lease.remaining, removals)
        lease.hits = _subtract_leased_hits(lease.hits, removals)
        lease.slice_ids = {id(hit.data) for hit in lease.hits}
        for data_id in old_slice_ids - lease.slice_ids:
            self._release_slice_reference_locked(data_id)
        if lease.remaining:
            return
        self._request_leases.pop(lease_id, None)
        if not lease.released.done():
            lease.released.set_result(None)
        self._l1.notify_pool_waiters()

    def _release_request_lease_locked(
        self,
        lease_id: str,
        *,
        force: bool = False,
    ) -> None:
        """Drop one complete lease and wake overlapping store waiters."""
        lease = self._request_leases.get(lease_id)
        if lease is None:
            return
        if lease.active_loads and not force:
            lease.release_all = True
            return
        self._request_leases.pop(lease_id, None)
        for data_id in lease.slice_ids:
            self._release_slice_reference_locked(data_id)
        if not lease.released.done():
            lease.released.set_result(None)
        self._l1.notify_pool_waiters()

    def _release_slice_reference_locked(self, data_id: int) -> None:
        """Release one request's reference to a retained pinned slice."""
        retained = self._leased_slices.get(data_id)
        if retained is None:
            return
        data, count = retained
        if count > 1:
            self._leased_slices[data_id] = (data, count - 1)
            return
        self._leased_slices.pop(data_id, None)
        self._close_unowned_slice_locked(data)

    def _close_unowned_slice_locked(self, data: PinnedMemorySlice) -> None:
        """Close a detached slice after its final writer/lease reference leaves."""
        if self._l1.contains_slice(data):
            return
        if any(buffer is data for buffer in self._pending_l2_buffers.values()):
            return
        if id(data) in self._leased_slices:
            return
        data.close()
        self._l1.notify_pool_waiters()

    def _overlapping_lease_waiters_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> list[asyncio.Future[None]]:
        """Return active request-lease waiters overlapping one store range."""
        end = file_offset + nbytes
        return [
            lease.released
            for lease in self._request_leases.values()
            if any(
                start < end and file_offset < start + size
                for start, size in lease.remaining
            )
        ]

    async def _wait_for_overlapping_leases(
        self,
        file_offset: int,
        nbytes: int,
    ) -> None:
        """Wait until no request lease protects an overlapping physical range."""
        while True:
            async with self._lock:
                waiters = self._overlapping_lease_waiters_locked(file_offset, nbytes)
            if not waiters:
                return
            await asyncio.gather(*waiters)
