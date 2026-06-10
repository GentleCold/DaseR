# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import bisect
from collections import OrderedDict
from typing import Any

# First Party
from daser.logging import init_logger
from daser.replacement import LRUReplacementPolicy
from daser.transfer.base import TransferLayer, TransferStats
from daser.transfer.iouring.pinned_pool import PinnedMemoryPool, PinnedMemorySlice

logger = init_logger(__name__)

_L1_ALIGNMENT = 4096


class L1OnlyTransferLayer(TransferLayer):
    """Async pinned-memory transfer layer with no L2 persistence.

    Args:
        l1_bytes: maximum memory-tier bytes.

    Async/thread-safety:
        Public async methods serialize L1 metadata through an asyncio lock.
        No file descriptors or background L2 writes are created.
    """

    coalesce_store_spans = True

    def __init__(self, l1_bytes: int) -> None:
        if l1_bytes <= 0:
            raise ValueError("l1_bytes must be positive")
        self._l1_bytes = l1_bytes
        self._pool = PinnedMemoryPool(l1_bytes, alignment=_L1_ALIGNMENT)
        self._l1: OrderedDict[tuple[int, int], PinnedMemorySlice] = OrderedDict()
        self._l1_starts: list[int] = []
        self._l1_by_start: dict[int, tuple[int, int]] = {}
        self._l1_used = 0
        self._policy = LRUReplacementPolicy[tuple[int, int]]()
        self._lock = asyncio.Lock()
        self.stats = TransferStats()
        logger.info("[TRANSFER:l1-only] l1=%d", l1_bytes)

    async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
        """Load bytes from L1 into a writable destination buffer.

        Args:
            dst: writable byte buffer or CUDA-like destination.
            file_offset: logical byte offset used as the L1 key.
            nbytes: number of bytes to load.

        Returns:
            Number of bytes loaded.

        Raises:
            KeyError: when the requested range is not resident in L1.
        """
        self._check_range(file_offset, nbytes)
        async with self._lock:
            chunks = self._find_l1_chunks_locked(file_offset, nbytes)
            if chunks is None:
                self.stats.l1_misses += 1
                raise KeyError(
                    "L1-only cache miss for range "
                    f"[{file_offset}, {file_offset + nbytes})"
                )
            for _target_offset, hit_key, _cached, _source_offset, _nbytes in chunks:
                self._policy.access(hit_key)
                self._l1.move_to_end(hit_key)
            self.stats.l1_hits += 1
            if len(chunks) == 1:
                _target_offset, _hit_key, cached, source_offset, _chunk_nbytes = chunks[
                    0
                ]
                self._copy_pinned_to_dst(dst, cached, source_offset, nbytes)
            else:
                copy_chunks = [
                    (target_offset, cached, source_offset, chunk_nbytes)
                    for (
                        target_offset,
                        _key,
                        cached,
                        source_offset,
                        chunk_nbytes,
                    ) in chunks
                ]
                self._copy_grouped_to_dst(dst, copy_chunks)
        return nbytes

    def _find_l1_chunks_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> list[tuple[int, tuple[int, int], PinnedMemorySlice, int, int]] | None:
        """Return adjacent resident L1 chunks covering a byte range."""
        end = file_offset + nbytes
        cursor = file_offset
        chunks: list[tuple[int, tuple[int, int], PinnedMemorySlice, int, int]] = []
        while cursor < end:
            hit = self._find_l1_locked(cursor, 1)
            if hit is None:
                return None
            key, cached, source_offset = hit
            key_end = key[0] + key[1]
            chunk_nbytes = min(end, key_end) - cursor
            if chunk_nbytes <= 0:
                return None
            chunks.append(
                (
                    cursor - file_offset,
                    key,
                    cached,
                    source_offset,
                    chunk_nbytes,
                )
            )
            cursor += chunk_nbytes
        return chunks

    async def load_bytes_grouped(
        self,
        dst: Any,
        spans: list[dict[str, int]],
    ) -> int:
        """Load multiple spans from L1 into a destination buffer.

        Args:
            dst: writable byte buffer or CUDA-like destination.
            spans: span dicts with target_offset, file_offset, and nbytes.

        Returns:
            Total number of bytes loaded.

        Raises:
            KeyError: when any requested span is not resident in L1.
        """
        total = 0
        chunks: list[tuple[int, PinnedMemorySlice, int, int]] = []
        accessed: list[tuple[int, int]] = []
        async with self._lock:
            for span in spans:
                target_offset = int(span.get("target_offset", 0))
                file_offset = int(span["file_offset"])
                nbytes = int(span["nbytes"])
                self._check_range(file_offset, nbytes)
                total += nbytes
                span_chunks = self._find_l1_chunks_locked(file_offset, nbytes)
                if span_chunks is None:
                    self.stats.l1_misses += 1
                    raise KeyError(
                        "L1-only cache miss for range "
                        f"[{file_offset}, {file_offset + nbytes})"
                    )
                for (
                    relative_target,
                    hit_key,
                    cached,
                    source_offset,
                    chunk_nbytes,
                ) in span_chunks:
                    accessed.append(hit_key)
                    chunks.append(
                        (
                            target_offset + relative_target,
                            cached,
                            source_offset,
                            chunk_nbytes,
                        )
                    )
            for hit_key in accessed:
                self._policy.access(hit_key)
                self._l1.move_to_end(hit_key)
            self.stats.l1_hits += len(spans)
            self._copy_grouped_to_dst(dst, chunks)
        return total

    async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
        """Store bytes in L1 memory without scheduling L2 persistence.

        Args:
            src: readable byte buffer or CUDA-like source.
            file_offset: logical byte offset used as the L1 key.
            nbytes: number of bytes to store.

        Returns:
            Number of bytes stored.
        """
        self._check_range(file_offset, nbytes)
        key = (file_offset, nbytes)
        async with self._lock:
            hit = self._find_l1_locked(file_offset, nbytes)
            if hit is not None:
                hit_key, cached, target_offset = hit
                self._copy_src_to_pinned_at(src, cached, target_offset, nbytes)
                self._policy.access(hit_key)
                self._l1.move_to_end(hit_key)
                return nbytes
            data = self._reserve_l1_buffer_locked(key, nbytes)
            try:
                self._copy_src_to_pinned_at(src, data, 0, nbytes)
            except BaseException:
                data.close()
                raise
            self._put_l1_locked(key, data)
        return nbytes

    async def store_bytes_grouped(
        self,
        src: Any,
        spans: list[dict[str, Any]],
    ) -> int:
        """Store multiple spans in L1 memory.

        Args:
            src: readable byte buffer or CUDA-like source.
            spans: span dicts with source_offset, file_offset, and nbytes.

        Returns:
            Total number of bytes stored.

        Async/thread-safety:
            Calls ``store_bytes`` for each span so replacement ordering matches
            single-span stores.
        """
        total = 0
        for span in spans:
            source_offset = int(span.get("source_offset", 0))
            nbytes = int(span["nbytes"])
            file_offset = int(span["file_offset"])
            source = self._slice_src(src, source_offset, nbytes)
            total += await self.store_bytes(source, file_offset, nbytes)
        return total

    async def drain(self) -> None:
        """Return immediately because L1-only mode has no background writes."""
        return None

    def close(self) -> None:
        """Release all L1 pinned memory resources."""
        for data in self._l1.values():
            data.close()
        self._l1.clear()
        self._l1_starts.clear()
        self._l1_by_start.clear()
        self._pool.close()

    def _check_range(self, file_offset: int, nbytes: int) -> None:
        """Validate a logical byte range.

        Args:
            file_offset: logical byte offset.
            nbytes: byte count.

        Raises:
            ValueError: when the range is negative or too large for L1.
        """
        if file_offset < 0 or nbytes < 0:
            raise ValueError("file_offset and nbytes must be non-negative")
        if nbytes > self._l1_bytes:
            raise ValueError(
                f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
            )

    def _put_l1_locked(self, key: tuple[int, int], data: PinnedMemorySlice) -> None:
        """Insert a range into L1 and evict old ranges until within capacity."""
        self._l1[key] = data
        self._insert_l1_index_locked(key)
        self._l1.move_to_end(key)
        self._policy.insert(key)
        self._l1_used += len(data)
        while self._l1_used > self._l1_bytes:
            victim = self._policy.evict()
            if victim is None:
                break
            removed = self._l1.pop(victim, None)
            self._remove_l1_index_locked(victim)
            if removed is not None:
                self._l1_used -= len(removed)
                removed.close()

    def _drop_overlapping_l1_locked(self, file_offset: int, nbytes: int) -> None:
        """Remove L1 entries overlapping a new store range."""
        end = file_offset + nbytes
        victims = [
            key for key in self._l1 if key[0] < end and file_offset < key[0] + key[1]
        ]
        for victim in victims:
            removed = self._l1.pop(victim, None)
            self._remove_l1_index_locked(victim)
            self._policy.remove(victim)
            if removed is not None:
                self._l1_used -= len(removed)
                removed.close()

    def _find_l1_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> tuple[tuple[int, int], PinnedMemorySlice, int] | None:
        """Return an L1 range covering the requested byte span."""
        end = file_offset + nbytes
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
        if end <= key[0] + key[1]:
            return key, data, file_offset - key[0]
        return None

    def _reserve_l1_buffer_locked(
        self,
        key: tuple[int, int],
        nbytes: int,
    ) -> PinnedMemorySlice:
        """Reserve pinned memory for one L1 range."""
        if nbytes > self._l1_bytes:
            raise ValueError(
                f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
            )
        self._drop_overlapping_l1_locked(key[0], key[1])
        data = self._pool.allocate(nbytes)
        while data is None:
            victim = self._policy.evict()
            if victim is None:
                raise MemoryError(
                    f"could not reserve {nbytes} pinned L1 bytes from "
                    f"{self._l1_bytes} byte pool"
                )
            removed = self._l1.pop(victim, None)
            self._remove_l1_index_locked(victim)
            if removed is not None:
                self._l1_used -= len(removed)
                removed.close()
            data = self._pool.allocate(nbytes)
        return data

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

    def _copy_pinned_to_dst(
        self,
        dst: Any,
        data: PinnedMemorySlice,
        source_offset: int,
        nbytes: int,
    ) -> None:
        """Copy pinned host bytes into a CPU or CUDA destination."""
        target = self._slice_dst(dst, 0, nbytes)
        dst_ptr = self._cuda_array_ptr(target)
        if dst_ptr is not None:
            from cupy.cuda import runtime

            runtime.memcpyAsync(
                dst_ptr,
                data.ptr_at(source_offset),
                nbytes,
                runtime.memcpyHostToDevice,
                0,
            )
            return
        if hasattr(target, "set"):
            import numpy

            host = numpy.frombuffer(
                data.view()[source_offset : source_offset + nbytes],
                dtype=numpy.uint8,
                count=nbytes,
            )
            target.set(host)
            return
        memoryview(dst).cast("B")[:nbytes] = data.view()[
            source_offset : source_offset + nbytes
        ]

    def _copy_grouped_to_dst(
        self,
        dst: Any,
        chunks: list[tuple[int, PinnedMemorySlice, int, int]],
    ) -> None:
        """Copy source chunks into the destination without staging repacks."""
        if not chunks:
            return
        first_target = self._slice_dst(dst, chunks[0][0], chunks[0][3])
        if self._cuda_array_ptr(first_target) is not None:
            self._copy_grouped_to_cuda_dst(dst, chunks)
            return

        for target_offset, data, source_offset, nbytes in self._coalesce_copy_chunks(
            chunks
        ):
            target = self._slice_dst(dst, target_offset, nbytes)
            if hasattr(target, "set"):
                import numpy

                host = numpy.frombuffer(
                    data.view()[source_offset : source_offset + nbytes],
                    dtype=numpy.uint8,
                    count=nbytes,
                )
                target.set(host)
                continue
            dst_view = memoryview(dst).cast("B")
            dst_view[target_offset : target_offset + nbytes] = data.view()[
                source_offset : source_offset + nbytes
            ]

    def _copy_grouped_to_cuda_dst(
        self,
        dst: Any,
        chunks: list[tuple[int, PinnedMemorySlice, int, int]],
    ) -> None:
        """Copy grouped pinned ranges into a CUDA destination."""
        from cupy.cuda import runtime

        ordered = sorted(chunks, key=lambda item: item[0])
        merged: list[tuple[int, int, int]] = []
        for target_offset, data, source_offset, nbytes in ordered:
            source_ptr = data.ptr_at(source_offset)
            if not merged:
                merged.append((target_offset, source_ptr, nbytes))
                continue
            prev_target, prev_source, prev_nbytes = merged[-1]
            if (
                target_offset == prev_target + prev_nbytes
                and source_ptr == prev_source + prev_nbytes
            ):
                merged[-1] = (prev_target, prev_source, prev_nbytes + nbytes)
                continue
            merged.append((target_offset, source_ptr, nbytes))

        for target_offset, source_ptr, nbytes in merged:
            target = self._slice_dst(dst, target_offset, nbytes)
            dst_ptr = self._cuda_array_ptr(target)
            if dst_ptr is None:
                raise TypeError("grouped CUDA copy target lost CUDA array interface")
            runtime.memcpyAsync(
                dst_ptr,
                source_ptr,
                nbytes,
                runtime.memcpyHostToDevice,
                0,
            )

    def _coalesce_copy_chunks(
        self,
        chunks: list[tuple[int, PinnedMemorySlice, int, int]],
    ) -> list[tuple[int, PinnedMemorySlice, int, int]]:
        """Merge adjacent L1-hit copies with contiguous source and target."""
        ordered = sorted(chunks, key=lambda item: item[0])
        merged: list[tuple[int, PinnedMemorySlice, int, int]] = []
        for target_offset, data, source_offset, nbytes in ordered:
            if not merged:
                merged.append((target_offset, data, source_offset, nbytes))
                continue
            prev_target, prev_data, prev_source, prev_nbytes = merged[-1]
            if (
                prev_data is data
                and target_offset == prev_target + prev_nbytes
                and source_offset == prev_source + prev_nbytes
            ):
                merged[-1] = (
                    prev_target,
                    prev_data,
                    prev_source,
                    prev_nbytes + nbytes,
                )
                continue
            merged.append((target_offset, data, source_offset, nbytes))
        return merged

    def _slice_dst(self, dst: Any, offset: int, nbytes: int) -> Any:
        """Return a writable destination slice."""
        if hasattr(dst, "set"):
            try:
                return dst[offset : offset + nbytes]
            except (TypeError, KeyError, IndexError):
                if offset == 0:
                    return dst
                raise
        if isinstance(dst, bytearray | memoryview):
            return memoryview(dst).cast("B")[offset : offset + nbytes]
        try:
            return dst[offset : offset + nbytes]
        except (TypeError, KeyError, IndexError):
            pass
        return memoryview(dst).cast("B")[offset : offset + nbytes]

    def _slice_src(self, src: Any, offset: int, nbytes: int) -> Any:
        """Return a readable source slice."""
        if hasattr(src, "get"):
            return src[offset : offset + nbytes]
        try:
            return src[offset : offset + nbytes]
        except (TypeError, KeyError, IndexError):
            pass
        return memoryview(src).cast("B")[offset : offset + nbytes]

    def _cuda_array_ptr(self, dst: Any) -> int | None:
        """Return a CUDA device pointer for a CuPy-like array destination."""
        data = getattr(dst, "data", None)
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            return None
        return int(ptr)

    def _copy_src_to_pinned_at(
        self,
        src: Any,
        pinned: PinnedMemorySlice,
        target_offset: int,
        nbytes: int,
    ) -> None:
        """Copy bytes from a CPU or CUDA source into pinned host memory."""
        if hasattr(src, "data") and getattr(src.data, "ptr", None) is not None:
            from cupy.cuda import runtime

            runtime.memcpy(
                pinned.ptr_at(target_offset),
                int(src.data.ptr),
                nbytes,
                runtime.memcpyDeviceToHost,
            )
            return
        pinned.view()[target_offset : target_offset + nbytes] = memoryview(src).cast(
            "B"
        )[:nbytes]
