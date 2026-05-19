# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections import OrderedDict
import os
from typing import Any

# First Party
from daser.logging import init_logger
from daser.replacement import LRUReplacementPolicy
from daser.transfer.base import TransferLayer, TransferStats

logger = init_logger(__name__)


class IOUringPinnedTransferLayer(TransferLayer):
    """Async L1 pinned-memory + L2 SSD transfer layer.

    The Python implementation uses executor-backed positioned file I/O as the
    portable fallback while keeping the public contract named for the target
    io_uring backend. GPU CUDA-IPC copies are added at the server IPC boundary;
    this layer owns the tiering and L2 persistence policy.

    Args:
        path: pre-allocated or creatable L2 store file.
        l1_bytes: maximum memory-tier bytes.
        l2_bytes: SSD-tier capacity.

    Async/thread-safety:
        Public async methods serialize tier metadata with an asyncio lock.
        Blocking file I/O is offloaded with ``run_in_executor``.
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
        self._l1_bytes = l1_bytes
        self._l2_bytes = l2_bytes
        self._l1: OrderedDict[tuple[int, int], bytearray] = OrderedDict()
        self._l1_used = 0
        self._policy = LRUReplacementPolicy[tuple[int, int]]()
        self._pending_l2: dict[tuple[int, int], asyncio.Task[None]] = {}
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
        pending = None
        async with self._lock:
            hit = self._find_l1_locked(file_offset, nbytes)
            if hit is not None:
                hit_key, cached, source_offset = hit
                self._policy.access(hit_key)
                self._l1.move_to_end(hit_key)
                self.stats.l1_hits += 1
                self._copy_to_dst(dst, memoryview(cached)[source_offset:], nbytes)
                return nbytes
            self.stats.l1_misses += 1
            pending = self._find_pending_l2_locked(file_offset, nbytes)

        if pending is not None:
            await pending
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._read_l2, file_offset, nbytes)
        async with self._lock:
            self.stats.l2_reads += 1
            self._put_l1_locked(key, bytearray(data))
            self._copy_to_dst(dst, data, nbytes)
        return nbytes

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
            previous = self._pending_l2.get(key)
            self._put_l1_locked(key, data)
        if previous is not None:
            await previous
        task = asyncio.create_task(self._write_l2_async(key, file_offset, bytes(data)))
        async with self._lock:
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
            if not pending:
                return
            await asyncio.gather(*pending)

    def close(self) -> None:
        """Close the L2 file handle after pending writes are drained/cancelled."""
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

    def _put_l1_locked(self, key: tuple[int, int], data: bytearray) -> None:
        """Insert bytes into L1 and evict until capacity is respected."""
        old = self._l1.pop(key, None)
        if old is not None:
            self._l1_used -= len(old)
            self._policy.remove(key)
        if len(data) > self._l1_bytes:
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

    def _find_l1_locked(
        self,
        file_offset: int,
        nbytes: int,
    ) -> tuple[tuple[int, int], bytearray, int] | None:
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
    ) -> asyncio.Task[None] | None:
        """Return a pending L2 write covering the requested byte span."""
        end = file_offset + nbytes
        for key, task in self._pending_l2.items():
            start, length = key
            if start <= file_offset and end <= start + length:
                return task
        return None

    def _read_l2(self, file_offset: int, nbytes: int) -> bytes:
        """Blocking positioned L2 read."""
        data = os.pread(self._fd, nbytes, file_offset)
        if len(data) != nbytes:
            raise IOError(f"short read: {len(data)} != {nbytes}")
        return data

    def _write_l2(self, file_offset: int, data: bytes) -> None:
        """Blocking positioned L2 write."""
        written = os.pwrite(self._fd, data, file_offset)
        if written != len(data):
            raise IOError(f"short write: {written} != {len(data)}")

    async def _write_l2_async(
        self,
        key: tuple[int, int],
        file_offset: int,
        data: bytes,
    ) -> None:
        """Persist one L2 write and publish completion to waiters."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_l2, file_offset, data)
            async with self._lock:
                self.stats.l2_writes += 1
        finally:
            async with self._lock:
                if self._pending_l2.get(key) is asyncio.current_task():
                    self._pending_l2.pop(key, None)

    def _copy_to_dst(self, dst: Any, data: bytes | bytearray, nbytes: int) -> None:
        """Copy bytes into a writable destination."""
        if hasattr(dst, "set"):
            import numpy

            host = numpy.frombuffer(data, dtype=numpy.uint8, count=nbytes)
            dst[:nbytes].set(host)
            return
        memoryview(dst)[:nbytes] = memoryview(data)[:nbytes]

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
