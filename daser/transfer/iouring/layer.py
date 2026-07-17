# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from typing import Any

# First Party
from daser.logging import init_logger
from daser.transfer.base import PrefetchResult, TransferLayer, TransferStats
from daser.transfer.iouring import copy_ops
from daser.transfer.iouring.l1_cache import L1Cache, L1RangeHit
from daser.transfer.iouring.l2_engine import L2IoEngine
from daser.transfer.iouring.native import NativeIOUring
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice

logger = init_logger(__name__)

_DIRECT_IO_ALIGNMENT = 4096


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
        self._l1 = L1Cache(
            l1_bytes,
            alignment=_DIRECT_IO_ALIGNMENT,
            pinned_predicate=self._is_pinned_by_l2,
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
            async with self._lock:
                hit = self._l1.find(file_offset)
                if hit is not None:
                    hit_key, cached, target_offset = hit
                    if target_offset + nbytes <= len(cached):
                        self._copy_src_to_pinned_at(src, cached, target_offset, nbytes)
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

        data = await self._reserve_l1_buffer(key, nbytes)
        try:
            self._copy_src_to_pinned(src, data, nbytes)
        except BaseException:
            data.close()
            raise
        async with self._lock:
            self._raise_l2_error_locked()
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
        self, spans: list[dict[str, int]]
    ) -> PrefetchResult:
        """Promote L2-missing portions of spans into the pinned L1 tier.

        Args:
            spans: Aligned storage spans containing ``file_offset`` and
                ``nbytes``.

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

        requested_bytes = 0
        l1_bytes = 0
        misses: list[dict[str, int]] = []
        pending: list[asyncio.Task[None]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            for span in spans:
                file_offset = int(span["file_offset"])
                nbytes = int(span["nbytes"])
                self._check_range(file_offset, nbytes)
                requested_bytes += nbytes
                l1_hits, span_misses = self._l1.resolve_subranges(
                    target_offset=0,
                    file_offset=file_offset,
                    nbytes=nbytes,
                )
                if l1_hits:
                    self._l1.record_hits(l1_hits)
                    l1_bytes += sum(hit.nbytes for hit in l1_hits)
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
            await self._load_l2_misses_grouped(None, misses)
        l2_bytes = sum(int(miss["nbytes"]) for miss in misses)
        async with self._lock:
            self._stats.prefetch_requests += 1
            self._stats.prefetch_l1_bytes += l1_bytes
            self._stats.prefetch_l2_bytes += l2_bytes
        return PrefetchResult(requested_bytes, l1_bytes, l2_bytes)

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

    def _is_pinned_by_l2(self, key: tuple[int, int], data: PinnedMemorySlice) -> bool:
        """Return whether ``data`` is still owned by an in-flight L2 write."""
        return self._pending_l2_buffers.get(key) is data

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
        dst: Any | None,
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
        dst: Any | None,
        misses: list[dict[str, int]],
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
                        self._finish_l1_promotion_locked(promotion_id)
        finally:
            if future_to_read:
                await asyncio.gather(*future_to_read, return_exceptions=True)
            async with self._lock:
                live_buffers = self._l1.resident_slice_ids()
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
        total = 0
        async with self._lock:
            self._raise_l2_error_locked()
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
