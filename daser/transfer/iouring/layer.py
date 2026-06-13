# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import bisect
from collections import OrderedDict
import concurrent.futures
from dataclasses import dataclass
import os
import threading
from typing import Any

# First Party
from daser.logging import init_logger
from daser.replacement import LRUReplacementPolicy
from daser.transfer.base import TransferLayer, TransferStats
from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.native import NativeIOUring
from daser.transfer.iouring.pinned_pool import PinnedMemoryPool, PinnedMemorySlice

logger = init_logger(__name__)

_DIRECT_IO_ALIGNMENT = 4096


@dataclass(frozen=True)
class _L1RangeHit:
    """One L1-backed subrange inside a requested load span."""

    target_offset: int
    key: tuple[int, int]
    data: PinnedMemorySlice
    source_offset: int
    nbytes: int


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
        self._skip_l2 = skip_l2
        self._path = path
        self._fd: int | None = None
        self._urings: list[NativeIOUring] = []
        self._io_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._uring_lock = threading.Lock()
        self._next_uring_index = 0
        if not skip_l2:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a+b") as f:
                f.truncate(l2_bytes)
            self._fd = os.open(path, os.O_RDWR | os.O_DIRECT)
            self._urings = [NativeIOUring(entries=64) for _ in range(io_workers)]
            self._io_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=io_workers,
                thread_name_prefix="daser-iouring",
            )
        self._l1_bytes = l1_bytes
        self._l2_bytes = l2_bytes
        self._pool = PinnedMemoryPool(
            l1_bytes,
            alignment=_DIRECT_IO_ALIGNMENT,
        )
        self._l1: OrderedDict[tuple[int, int], PinnedMemorySlice] = OrderedDict()
        self._l1_starts: list[int] = []
        self._l1_by_start: dict[int, tuple[int, int]] = {}
        self._l1_used = 0
        self._policy = LRUReplacementPolicy[tuple[int, int]]()
        self._pending_l2: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._pending_l2_buffers: dict[tuple[int, int], PinnedMemorySlice] = {}
        self._pool_waiters: list[asyncio.Future[None]] = []
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
        l1_hits: list[_L1RangeHit] = []
        misses: list[dict[str, int]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            l1_hits, misses = self._resolve_l1_subranges_locked(
                target_offset=0,
                file_offset=file_offset,
                nbytes=nbytes,
            )
            if self._skip_l2 and misses:
                self._stats.l1_misses += len(misses)
                raise KeyError(
                    "skip_l2 cache miss for range "
                    f"[{file_offset}, {file_offset + nbytes})"
                )
            if l1_hits:
                self._record_l1_hits_locked(
                    l1_hits,
                    hit_count=1 if self._skip_l2 else None,
                )
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
                l1_hits, span_misses = self._resolve_l1_subranges_locked(
                    target_offset=target_offset,
                    file_offset=file_offset,
                    nbytes=nbytes,
                )
                if self._skip_l2 and span_misses:
                    self._stats.l1_misses += 1
                    raise KeyError(
                        "skip_l2 cache miss for range "
                        f"[{file_offset}, {file_offset + nbytes})"
                    )
                if l1_hits:
                    self._record_l1_hits_locked(
                        l1_hits,
                        hit_count=1 if self._skip_l2 else None,
                    )
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
        if self._skip_l2:
            async with self._lock:
                hit = self._find_l1_locked(file_offset)
                if hit is not None:
                    hit_key, cached, target_offset = hit
                    if target_offset + nbytes <= len(cached):
                        self._copy_src_to_pinned_at(src, cached, target_offset, nbytes)
                        self._policy.access(hit_key)
                        self._l1.move_to_end(hit_key)
                        return nbytes
                data = self._reserve_l1_buffer_locked_or_raise(
                    key,
                    nbytes,
                    preserve_overlaps=True,
                )
                try:
                    self._copy_src_to_pinned_at(src, data, 0, nbytes)
                except BaseException:
                    data.close()
                    raise
                self._put_l1_locked(key, data)
            return nbytes

        data = await self._reserve_l1_buffer(key, nbytes)
        try:
            self._copy_src_to_pinned(src, data, nbytes)
        except BaseException:
            data.close()
            raise
        async with self._lock:
            self._raise_l2_error_locked()
            previous = self._find_pending_l2_locked(file_offset, nbytes)
            self._put_l1_locked(key, data)
            task = self._schedule_l2_write_locked(
                key,
                file_offset,
                data,
                previous,
            )
            self._pending_l2[key] = task
            self._pending_l2_buffers[key] = data
        return nbytes

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
        if self._skip_l2:
            return await self._store_bytes_grouped_l1_only(src, spans)

        total = 0
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            source = self._slice_src(src, source_offset, nbytes)
            total += await self.store_bytes(source, file_offset, nbytes)
        return total

    async def drain(self) -> None:
        """Wait until all pending L2 writes have completed.

        Async/thread-safety:
            Must be called from the owning asyncio event loop before shutdown
            when durable L2 contents are required.
        """
        if self._skip_l2:
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
        for task in self._pending_l2.values():
            task.cancel()
        self._pending_l2.clear()
        for pending_buffer in self._pending_l2_buffers.values():
            if pending_buffer not in self._l1.values():
                pending_buffer.close()
        self._pending_l2_buffers.clear()
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=True)
        for uring in self._urings:
            uring.close()
        if self._fd is not None:
            os.close(self._fd)
        self._pool.close()

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
        if self._skip_l2:
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

    def _put_l1_locked(self, key: tuple[int, int], data: PinnedMemorySlice) -> None:
        """Insert bytes into L1 after dropping overlapping ranges."""
        self._drop_overlapping_l1_locked(key[0], key[1])
        self._insert_l1_entry_locked(key, data)

    def _insert_l1_entry_locked(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
    ) -> None:
        """Insert one non-overlapping L1 entry and enforce capacity."""
        if len(data) > self._l1_bytes:
            return
        self._l1[key] = data
        self._insert_l1_index_locked(key)
        self._l1.move_to_end(key)
        self._policy.insert(key)
        self._l1_used += len(data)
        self._notify_pool_waiters_locked()
        while self._l1_used > self._l1_bytes:
            victim = self._policy.evict()
            if victim is None:
                break
            removed = self._l1.pop(victim, None)
            self._remove_l1_index_locked(victim)
            if removed is not None:
                self._l1_used -= len(removed)
                self._release_l1_buffer_locked(victim, removed)

    def _drop_overlapping_l1_locked(
        self,
        file_offset: int,
        nbytes: int,
        *,
        preserve_remainder: bool = False,
    ) -> None:
        """Remove L1 entries that overlap a newly written byte range."""
        end = file_offset + nbytes
        victims = [
            key for key in self._l1 if key[0] < end and file_offset < key[0] + key[1]
        ]
        for victim in victims:
            removed = self._l1.pop(victim, None)
            self._remove_l1_index_locked(victim)
            self._policy.remove(victim)
            preserved = (
                self._preserve_non_overlapping_l1_bytes(
                    victim,
                    removed,
                    file_offset,
                    end,
                )
                if preserve_remainder and removed is not None
                else []
            )
            if removed is not None:
                self._l1_used -= len(removed)
                self._release_l1_buffer_locked(victim, removed)
            for preserved_key, payload in preserved:
                self._put_preserved_l1_fragment_locked(preserved_key, payload)

    def _preserve_non_overlapping_l1_bytes(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
        overlap_start: int,
        overlap_end: int,
    ) -> list[tuple[tuple[int, int], bytes]]:
        """Return old L1 fragments that fall outside an overwrite range."""
        key_start, key_size = key
        key_end = key_start + key_size
        fragments: list[tuple[tuple[int, int], bytes]] = []
        view = data.view()
        if key_start < overlap_start:
            keep = overlap_start - key_start
            fragments.append(((key_start, keep), bytes(view[:keep])))
        if overlap_end < key_end:
            source_offset = overlap_end - key_start
            keep = key_end - overlap_end
            fragments.append(
                (
                    (overlap_end, keep),
                    bytes(view[source_offset : source_offset + keep]),
                )
            )
        return fragments

    def _put_preserved_l1_fragment_locked(
        self,
        key: tuple[int, int],
        payload: bytes,
    ) -> None:
        """Insert one preserved fragment copied out of an overwritten L1 range."""
        if not payload:
            return
        data, wait_for = self._reserve_l1_buffer_locked(
            key,
            len(payload),
            drop_overlaps=False,
        )
        if data is None:
            if wait_for is not None:
                raise RuntimeError("unexpected pending L2 write preserving L1 range")
            raise MemoryError(
                f"could not preserve {len(payload)} L1 bytes from overwritten range"
            )
        try:
            data.view()[: len(payload)] = payload
        except BaseException:
            data.close()
            raise
        self._insert_l1_entry_locked(key, data)

    def _find_l1_locked(
        self,
        file_offset: int,
    ) -> tuple[tuple[int, int], PinnedMemorySlice, int] | None:
        """Return the cached L1 range containing the requested start offset."""
        idx = bisect.bisect_right(self._l1_starts, file_offset) - 1
        if idx < 0:
            return None
        start = self._l1_starts[idx]
        key = self._l1_by_start.get(start)
        if key is None:
            return None
        data = self._l1.get(key)
        if data is None:
            return None
        if file_offset < key[0] + key[1]:
            return key, data, file_offset - key[0]
        return None

    def _resolve_l1_subranges_locked(
        self,
        target_offset: int,
        file_offset: int,
        nbytes: int,
    ) -> tuple[list[_L1RangeHit], list[dict[str, int]]]:
        """Split a load span into cached L1 slices and uncached gaps.

        Args:
            target_offset: destination byte offset corresponding to
                ``file_offset``.
            file_offset: L2 byte offset where the requested span starts.
            nbytes: requested byte count.

        Returns:
            A pair of L1 hit slices and L2 miss gaps in ascending file-offset
            order. Metadata is read under ``_lock`` and no buffers are mutated.

        Async/thread-safety:
            Must be called with the transfer metadata lock held.
        """
        hits: list[_L1RangeHit] = []
        misses: list[dict[str, int]] = []
        request_end = file_offset + nbytes
        cursor = file_offset
        while cursor < request_end:
            hit = self._find_l1_locked(cursor)
            if hit is not None:
                key, data, source_offset = hit
                covered = min(key[0] + key[1], request_end) - cursor
                hits.append(
                    _L1RangeHit(
                        target_offset=target_offset + (cursor - file_offset),
                        key=key,
                        data=data,
                        source_offset=source_offset,
                        nbytes=covered,
                    )
                )
                cursor += covered
                continue

            next_idx = bisect.bisect_left(self._l1_starts, cursor)
            next_start = (
                self._l1_starts[next_idx]
                if next_idx < len(self._l1_starts)
                else request_end
            )
            gap_end = min(next_start, request_end)
            if gap_end <= cursor:
                gap_end = request_end
            misses.append(
                {
                    "target_offset": target_offset + (cursor - file_offset),
                    "file_offset": cursor,
                    "nbytes": gap_end - cursor,
                }
            )
            cursor = gap_end
        return hits, misses

    def _record_l1_hits_locked(
        self,
        hits: list[_L1RangeHit],
        hit_count: int | None = None,
    ) -> None:
        """Update replacement state and stats for L1 hit slices.

        Args:
            hits: L1 slices returned by ``_resolve_l1_subranges_locked``.

        Returns:
            None.

        Async/thread-safety:
            Must be called with the transfer metadata lock held.
        """
        for hit in hits:
            self._policy.access(hit.key)
            self._l1.move_to_end(hit.key)
        self._stats.l1_hits += len(hits) if hit_count is None else hit_count

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
        if self._fd is None:
            raise RuntimeError("L2 reads are disabled when skip_l2 is true")
        return uring.read_into(self._fd, file_offset, dst.view())

    def _write_l2(
        self,
        file_offset: int,
        data: PinnedMemorySlice,
        uring: NativeIOUring,
    ) -> None:
        """Blocking io_uring L2 write."""
        if self._fd is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        written = uring.write(self._fd, file_offset, data.view())
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
        if self._io_executor is None:
            raise RuntimeError("L2 writes are disabled when skip_l2 is true")
        if previous:
            return asyncio.create_task(
                self._write_l2_async(key, file_offset, data, previous)
            )

        loop = asyncio.get_event_loop()
        uring = self._next_uring()
        future = loop.run_in_executor(
            self._io_executor,
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
                    if (
                        pending_buffer is not None
                        and self._l1.get(key) is not pending_buffer
                    ):
                        pending_buffer.close()

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
            if self._io_executor is None:
                raise RuntimeError("L2 writes are disabled when skip_l2 is true")
            await loop.run_in_executor(
                self._io_executor,
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
                    if (
                        pending_buffer is not None
                        and self._l1.get(key) is not pending_buffer
                    ):
                        pending_buffer.close()

    def _raise_l2_error_locked(self) -> None:
        """Raise and clear the first asynchronous L2 write failure."""
        if not self._l2_errors:
            return
        error = self._l2_errors.pop(0)
        raise RuntimeError("asynchronous io_uring L2 write failed") from error

    async def _load_l2_misses_grouped(
        self,
        dst: Any,
        misses: list[dict[str, int]],
    ) -> None:
        """Read grouped L2 misses concurrently, then promote in request order."""
        start = 0
        while start < len(misses):
            batch = self._next_l2_miss_batch(misses, start)
            await self._load_l2_miss_batch(dst, batch)
            start += len(batch)

    async def _load_l2_miss_batch(
        self,
        dst: Any,
        misses: list[dict[str, int]],
    ) -> None:
        """Read one bounded L2 miss batch and promote it to L1."""
        loop = asyncio.get_event_loop()
        if self._io_executor is None:
            raise RuntimeError("L2 reads are disabled when skip_l2 is true")
        reads: list[tuple[dict[str, int], PinnedMemorySlice]] = []
        try:
            for span in misses:
                nbytes = int(span["nbytes"])
                key = (int(span["file_offset"]), nbytes)
                pinned = await self._reserve_l1_buffer(key, nbytes)
                reads.append((span, pinned))

            await asyncio.gather(
                *(
                    loop.run_in_executor(
                        self._io_executor,
                        self._read_l2_into,
                        int(span["file_offset"]),
                        pinned,
                        self._next_uring(),
                    )
                    for span, pinned in reads
                )
            )

            self._copy_grouped_to_dst(
                dst,
                [
                    (
                        int(span["target_offset"]),
                        pinned,
                        0,
                        int(span["nbytes"]),
                    )
                    for span, pinned in reads
                ],
            )
            async with self._lock:
                self._raise_l2_error_locked()
                for span, pinned in reads:
                    nbytes = int(span["nbytes"])
                    key = (int(span["file_offset"]), nbytes)
                    self._stats.l2_reads += 1
                    self._put_l1_locked(key, pinned)
        except BaseException:
            live_buffers = set()
            async with self._lock:
                live_buffers = {id(buffer) for buffer in self._l1.values()}
            for _span, pinned in reads:
                if id(pinned) not in live_buffers:
                    pinned.close()
            raise

    def _next_l2_miss_batch(
        self,
        misses: list[dict[str, int]],
        start: int,
    ) -> list[dict[str, int]]:
        """Return a miss batch bounded by L1 capacity and io_uring count."""
        batch: list[dict[str, int]] = []
        batch_bytes = 0
        for span in misses[start:]:
            nbytes = int(span["nbytes"])
            if batch and (
                batch_bytes + nbytes > self._l1_bytes or len(batch) >= len(self._urings)
            ):
                break
            batch.append(span)
            batch_bytes += nbytes
        return batch

    def _next_uring(self) -> NativeIOUring:
        """Return the next native io_uring instance for one L2 operation."""
        if not self._urings:
            raise RuntimeError("io_uring rings are disabled when skip_l2 is true")
        with self._uring_lock:
            uring = self._urings[self._next_uring_index]
            self._next_uring_index = (self._next_uring_index + 1) % len(self._urings)
            return uring

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
        total = 0
        async with self._lock:
            self._raise_l2_error_locked()
            for span in spans:
                source_offset = int(span.get("source_offset", 0))
                nbytes = int(span["nbytes"])
                file_offset = int(span["file_offset"])
                self._check_range(file_offset, nbytes)
                key = (file_offset, nbytes)
                hit = self._find_l1_locked(file_offset)
                source = self._slice_src(src, source_offset, nbytes)
                if hit is not None:
                    hit_key, cached, target_offset = hit
                    if target_offset + nbytes <= len(cached):
                        self._copy_src_to_pinned_at(
                            source, cached, target_offset, nbytes
                        )
                        self._policy.access(hit_key)
                        self._l1.move_to_end(hit_key)
                        total += nbytes
                        continue
                data = self._reserve_l1_buffer_locked_or_raise(
                    key,
                    nbytes,
                    preserve_overlaps=True,
                )
                try:
                    self._copy_src_to_pinned_at(source, data, 0, nbytes)
                except BaseException:
                    data.close()
                    raise
                self._put_l1_locked(key, data)
                total += nbytes
        return total

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
                data, wait_for = self._reserve_l1_buffer_locked(key, nbytes)
                if data is not None:
                    return data
                if wait_for is None:
                    wait_for = asyncio.get_event_loop().create_future()
                    self._pool_waiters.append(wait_for)
            if wait_for is None:
                raise MemoryError(
                    f"could not reserve {nbytes} pinned L1 bytes from "
                    f"{self._l1_bytes} byte pool"
                )
            await wait_for

    def _reserve_l1_buffer_locked(
        self,
        key: tuple[int, int],
        nbytes: int,
        *,
        drop_overlaps: bool = True,
        preserve_overlaps: bool = False,
    ) -> tuple[PinnedMemorySlice | None, asyncio.Task[None] | None]:
        """Try to reserve pinned L1 space for a new store or promoted load."""
        if nbytes > self._l1_bytes:
            raise ValueError(
                f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
            )
        if drop_overlaps:
            self._drop_overlapping_l1_locked(
                key[0],
                key[1],
                preserve_remainder=preserve_overlaps,
            )
        data = self._pool.allocate(nbytes)
        while data is None:
            victim = self._policy.evict()
            if victim is None:
                pending = next(iter(self._pending_l2.values()), None)
                return None, pending
            removed = self._l1.pop(victim, None)
            self._remove_l1_index_locked(victim)
            if removed is not None:
                self._l1_used -= len(removed)
                self._release_l1_buffer_locked(victim, removed)
            data = self._pool.allocate(nbytes)
        return data, None

    def _reserve_l1_buffer_locked_or_raise(
        self,
        key: tuple[int, int],
        nbytes: int,
        *,
        preserve_overlaps: bool = False,
    ) -> PinnedMemorySlice:
        """Reserve pinned L1 space when no pending L2 writer can block reuse."""
        data, wait_for = self._reserve_l1_buffer_locked(
            key,
            nbytes,
            preserve_overlaps=preserve_overlaps,
        )
        if data is not None:
            return data
        if wait_for is not None:
            raise RuntimeError("unexpected pending L2 write in skip_l2 mode")
        raise MemoryError(
            f"could not reserve {nbytes} pinned L1 bytes from "
            f"{self._l1_bytes} byte pool"
        )

    def _release_l1_buffer_locked(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
    ) -> None:
        """Close an L1 buffer unless an L2 write still owns it."""
        if self._pending_l2_buffers.get(key) is data:
            return
        data.close()
        self._notify_pool_waiters_locked()

    def _insert_l1_index_locked(self, key: tuple[int, int]) -> None:
        """Add one L1 range to the start-offset lookup index."""
        start = key[0]
        existing = self._l1_by_start.get(start)
        if existing == key:
            return
        if existing is not None:
            self._remove_l1_index_locked(existing)
        bisect.insort(self._l1_starts, start)
        self._l1_by_start[start] = key

    def _remove_l1_index_locked(self, key: tuple[int, int]) -> None:
        """Remove one L1 range from the start-offset lookup index."""
        start = key[0]
        if self._l1_by_start.get(start) != key:
            return
        del self._l1_by_start[start]
        idx = bisect.bisect_left(self._l1_starts, start)
        if idx < len(self._l1_starts) and self._l1_starts[idx] == start:
            self._l1_starts.pop(idx)

    def _notify_pool_waiters_locked(self) -> None:
        """Wake tasks waiting for L1 pool metadata or free-space changes."""
        waiters = self._pool_waiters
        self._pool_waiters = []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)
