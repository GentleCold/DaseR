# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections import OrderedDict
import concurrent.futures
import os
from typing import Any

# First Party
from daser.logging import init_logger
from daser.replacement import LRUReplacementPolicy
from daser.transfer.base import TransferLayer, TransferStats
from daser.transfer.native_iouring import NativeIOUring
from daser.transfer.pinned_memory import PinnedMemoryBuffer

logger = init_logger(__name__)


class IOUringPinnedTransferLayer(TransferLayer):
    """Async L1 pinned-memory + L2 SSD transfer layer.

    The implementation uses Linux io_uring for L2 positioned file I/O. GPU
    CUDA-IPC copies are added at the server IPC boundary; this layer owns the
    tiering and L2 persistence policy.

    Args:
        path: pre-allocated or creatable L2 store file.
        l1_bytes: maximum memory-tier bytes.
        l2_bytes: SSD-tier capacity.

    Async/thread-safety:
        Public async methods serialize tier metadata with an asyncio lock.
        io_uring completion waits are offloaded with ``run_in_executor``.
    """

    def __init__(self, path: str, l1_bytes: int, l2_bytes: int) -> None:
        if l1_bytes <= 0:
            raise ValueError("l1_bytes must be positive")
        if l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if l1_bytes > l2_bytes:
            raise ValueError("l1_bytes must not exceed l2_bytes")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a+b") as f:
            f.truncate(l2_bytes)

        self._path = path
        self._fd = os.open(path, os.O_RDWR)
        self._uring = NativeIOUring(entries=64)
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="daser-iouring",
        )
        self._l1_bytes = l1_bytes
        self._l2_bytes = l2_bytes
        self._l1: OrderedDict[tuple[int, int], PinnedMemoryBuffer] = OrderedDict()
        self._l1_used = 0
        self._policy = LRUReplacementPolicy[tuple[int, int]]()
        self._pending_l2: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._l2_errors: list[BaseException] = []
        self._lock = asyncio.Lock()
        self.stats = TransferStats()
        logger.info(
            "[TRANSFER:iouring-pinned] path=%s l1=%d l2=%d",
            path,
            l1_bytes,
            l2_bytes,
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
        key = (file_offset, nbytes)
        pending: list[asyncio.Task[None]] = []
        async with self._lock:
            self._raise_l2_error_locked()
            hit = self._find_l1_locked(file_offset, nbytes)
            if hit is not None:
                hit_key, cached, source_offset = hit
                self._policy.access(hit_key)
                self._l1.move_to_end(hit_key)
                self.stats.l1_hits += 1
                self._copy_pinned_to_dst(dst, cached, source_offset, nbytes)
                return nbytes
            self.stats.l1_misses += 1
            pending = self._find_pending_l2_locked(file_offset, nbytes)

        if pending:
            await asyncio.gather(*pending)
        loop = asyncio.get_event_loop()
        pinned = PinnedMemoryBuffer(nbytes)
        try:
            await loop.run_in_executor(
                self._io_executor,
                self._read_l2_into,
                file_offset,
                pinned,
            )
        except BaseException:
            pinned.close()
            raise
        async with self._lock:
            self._raise_l2_error_locked()
            self.stats.l2_reads += 1
            self._copy_pinned_to_dst(dst, pinned, 0, nbytes)
            self._put_l1_locked(key, pinned)
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
        merged_l1: list[tuple[int, PinnedMemoryBuffer, int, int]] = []
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
                hit = self._find_l1_locked(file_offset, nbytes)
                if hit is None:
                    self.stats.l1_misses += 1
                    pending.extend(self._find_pending_l2_locked(file_offset, nbytes))
                    misses.append(
                        {
                            "target_offset": target_offset,
                            "file_offset": file_offset,
                            "nbytes": nbytes,
                        }
                    )
                    continue
                hit_key, cached, source_offset = hit
                self._policy.access(hit_key)
                self._l1.move_to_end(hit_key)
                self.stats.l1_hits += 1
                merged_l1.append(
                    (
                        target_offset,
                        cached,
                        source_offset,
                        nbytes,
                    )
                )
            if merged_l1:
                self._copy_grouped_to_dst(dst, merged_l1)

        if pending:
            await asyncio.gather(*pending)
        for span in misses:
            target_offset = int(span["target_offset"])
            nbytes = int(span["nbytes"])
            dst_slice = self._slice_dst(dst, target_offset, nbytes)
            await self.load_bytes(dst_slice, int(span["file_offset"]), nbytes)
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
        data = self._copy_from_src(src, nbytes)
        key = (file_offset, nbytes)
        async with self._lock:
            self._raise_l2_error_locked()
            previous = self._find_pending_l2_locked(file_offset, nbytes)
            self._put_l1_locked(key, PinnedMemoryBuffer.from_bytes(data))
            task = self._schedule_l2_write_locked(
                key,
                file_offset,
                bytes(data),
                previous,
            )
            self._pending_l2[key] = task
        return nbytes

    async def drain(self) -> None:
        """Wait until all pending L2 writes have completed.

        Async/thread-safety:
            Must be called from the owning asyncio event loop before shutdown
            when durable L2 contents are required.
        """
        while True:
            async with self._lock:
                pending = list(self._pending_l2.values())
                self._raise_l2_error_locked()
            if not pending:
                return
            await asyncio.gather(*pending)

    def close(self) -> None:
        """Close the L2 file handle after pending writes are drained/cancelled."""
        self._io_executor.shutdown(wait=True)
        self._uring.close()
        os.close(self._fd)

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
        if file_offset + nbytes > self._l2_bytes:
            raise ValueError(
                f"range [{file_offset}, {file_offset + nbytes}) exceeds "
                f"L2 capacity {self._l2_bytes}"
            )

    def _put_l1_locked(self, key: tuple[int, int], data: PinnedMemoryBuffer) -> None:
        """Insert bytes into L1 and evict until capacity is respected."""
        self._drop_overlapping_l1_locked(key[0], key[1])
        if len(data) > self._l1_bytes:
            data.close()
            return
        self._l1[key] = data
        self._l1.move_to_end(key)
        self._policy.insert(key)
        self._l1_used += len(data)
        while self._l1_used > self._l1_bytes:
            victim = self._policy.evict()
            if victim is None:
                break
            removed = self._l1.pop(victim, None)
            if removed is not None:
                self._l1_used -= len(removed)
                removed.close()

    def _drop_overlapping_l1_locked(self, file_offset: int, nbytes: int) -> None:
        """Remove L1 entries that overlap a newly written byte range."""
        end = file_offset + nbytes
        victims = [
            key for key in self._l1 if key[0] < end and file_offset < key[0] + key[1]
        ]
        for victim in victims:
            removed = self._l1.pop(victim, None)
            self._policy.remove(victim)
            if removed is not None:
                self._l1_used -= len(removed)
                removed.close()

    def _find_l1_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> tuple[tuple[int, int], PinnedMemoryBuffer, int] | None:
        """Return a cached L1 range covering the requested byte span."""
        end = file_offset + nbytes
        for key, data in self._l1.items():
            start, length = key
            if start <= file_offset and end <= start + length:
                return key, data, file_offset - start
        return None

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

    def _read_l2(self, file_offset: int, nbytes: int) -> bytes:
        """Blocking io_uring L2 read."""
        return self._uring.read(self._fd, file_offset, nbytes)

    def _read_l2_into(
        self,
        file_offset: int,
        dst: PinnedMemoryBuffer,
    ) -> int:
        """Blocking io_uring L2 read into pinned memory."""
        return self._uring.read_into(self._fd, file_offset, dst.view())

    def _write_l2(self, file_offset: int, data: bytes) -> None:
        """Blocking io_uring L2 write."""
        written = self._uring.write(self._fd, file_offset, data)
        if written != len(data):
            raise IOError(f"short io_uring write: {written} != {len(data)}")

    def _schedule_l2_write_locked(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: bytes,
        previous: list[asyncio.Task[None]],
    ) -> asyncio.Task[None]:
        """Schedule one L2 write and start independent IO immediately."""
        if previous:
            return asyncio.create_task(
                self._write_l2_async(key, file_offset, data, previous)
            )

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._io_executor,
            self._write_l2,
            file_offset,
            data,
        )
        return asyncio.create_task(self._track_l2_write(key, future))

    async def _track_l2_write(
        self,
        key: tuple[int, int],
        future: asyncio.Future[None],
    ) -> None:
        """Publish completion for an already-submitted L2 write."""
        try:
            await future
            async with self._lock:
                self.stats.l2_writes += 1
        except BaseException as exc:
            async with self._lock:
                self._l2_errors.append(exc)
            raise
        finally:
            async with self._lock:
                if self._pending_l2.get(key) is asyncio.current_task():
                    self._pending_l2.pop(key, None)

    async def _write_l2_async(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: bytes,
        previous: list[asyncio.Task[None]],
    ) -> None:
        """Persist one L2 write and publish completion to waiters."""
        try:
            if previous:
                await asyncio.gather(*previous)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._io_executor,
                self._write_l2,
                file_offset,
                data,
            )
            async with self._lock:
                self.stats.l2_writes += 1
        except BaseException as exc:
            async with self._lock:
                self._l2_errors.append(exc)
            raise
        finally:
            async with self._lock:
                if self._pending_l2.get(key) is asyncio.current_task():
                    self._pending_l2.pop(key, None)

    def _raise_l2_error_locked(self) -> None:
        """Raise and clear the first asynchronous L2 write failure."""
        if not self._l2_errors:
            return
        error = self._l2_errors.pop(0)
        raise RuntimeError("asynchronous io_uring L2 write failed") from error

    def _copy_to_dst(
        self,
        dst: Any,
        data: bytes | bytearray | memoryview,
        nbytes: int,
    ) -> None:
        """Copy bytes into a writable destination."""
        if hasattr(dst, "set"):
            import numpy

            host = numpy.frombuffer(data, dtype=numpy.uint8, count=nbytes)
            dst[:nbytes].set(host)
            return
        memoryview(dst)[:nbytes] = memoryview(data)[:nbytes]

    def _copy_pinned_to_dst(
        self,
        dst: Any,
        data: PinnedMemoryBuffer,
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
        chunks: list[tuple[int, PinnedMemoryBuffer, int, int]],
    ) -> None:
        """Copy source chunks into the destination without staging repacks."""
        if not chunks:
            return
        for target_offset, data, source_offset, nbytes in chunks:
            target = self._slice_dst(dst, target_offset, nbytes)
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
                continue
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

    def _slice_dst(self, dst: Any, offset: int, nbytes: int) -> Any:
        """Return a writable destination slice."""
        if hasattr(dst, "set"):
            return dst[offset : offset + nbytes]
        try:
            return dst[offset : offset + nbytes]
        except (TypeError, KeyError, IndexError):
            pass
        return memoryview(dst).cast("B")[offset : offset + nbytes]

    def _cuda_array_ptr(self, dst: Any) -> int | None:
        """Return a CUDA device pointer for a CuPy-like array destination."""
        data = getattr(dst, "data", None)
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            return None
        return int(ptr)

    def _copy_from_src(self, src: Any, nbytes: int) -> bytearray:
        """Copy bytes from a CPU or CuPy source into host memory.

        Args:
            src: readable byte buffer or CuPy ndarray.
            nbytes: number of bytes to copy.

        Returns:
            Host bytearray containing the source bytes.
        """
        if hasattr(src, "get"):
            return bytearray(src[:nbytes].get().tobytes())
        return bytearray(memoryview(src)[:nbytes])
