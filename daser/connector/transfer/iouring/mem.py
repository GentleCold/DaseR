# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections.abc import Callable
import inspect
from typing import Optional

# Third Party
import cupy
import torch

# First Party
from daser.connector.transfer.base import (
    BaseTransferLayer,
    TransferBackendName,
    TransferStats,
)
from daser.connector.transfer.iouring.engine import FileIOEngine, NativeIOUringEngine
from daser.connector.transfer.iouring.l1_cache import PinnedL1Cache
from daser.connector.transfer.utils import (
    as_torch_uint8,
    copy_tensor,
    maybe_await,
    require_store_path,
)
from daser.logging import init_logger

logger = init_logger(__name__)


class IOUringMemTransferLayer(BaseTransferLayer):
    """Transfer backend using pinned host L1 plus SSD-memory I/O.

    Args:
        path: preallocated store file path.
        l1_cache_size: pinned host L1 capacity in bytes.
        allocator: optional host buffer allocator for tests.
        io_engine: optional SSD-memory I/O engine for tests.
        commit_l1: optional callback after L1 write is complete.
        commit_l2: optional callback after L2 write is durable.
        evict_l1: optional callback after L1 eviction.

    Async/thread-safety:
        Public coroutines are submitted on the worker background asyncio loop.
        L1 cache access is expected to be serialized by that loop.
    """

    def __init__(
        self,
        path: str,
        l1_cache_size: int,
        allocator: Callable[[int], torch.Tensor] | None = None,
        io_engine: FileIOEngine | None = None,
        commit_l1: Callable[[str], object] | None = None,
        commit_l2: Callable[[str], object] | None = None,
        evict_l1: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        require_store_path(path)
        self._io = io_engine or NativeIOUringEngine(path)
        self._commit_l1 = commit_l1
        self._commit_l2 = commit_l2
        self._cache = PinnedL1Cache(
            capacity_bytes=l1_cache_size,
            allocator=allocator,
            on_evict=self._wrap_evict_callback(evict_l1),
        )
        logger.info("[IOURING] path=%s l1_cache_size=%d", path, l1_cache_size)

    @property
    def backend_name(self) -> TransferBackendName:
        """Configured transfer backend name."""
        return TransferBackendName.IOURING_MEM

    @property
    def max_concurrent_chunk_reads(self) -> int:
        """Return the chunk-read concurrency budget for L1-backed reads.

        Returns:
            One active read, so an L2 miss can occupy L1, copy to GPU staging,
            and release its load pin before the next miss reserves L1.

        Async/thread-safety:
            Immutable after construction and safe to read from the worker
            thread.
        """
        return 1

    async def write_chunk_async(
        self,
        chunk_key: str,
        buf: torch.Tensor | cupy.ndarray,
        file_offset: int,
        nbytes: int,
    ) -> int:
        """Write a chunk to L1 first, then persist it to L2.

        Args:
            chunk_key: chunk cache key.
            buf: source tensor or CuPy view.
            file_offset: byte offset in L2 store.
            nbytes: bytes to write.

        Returns:
            Number of bytes accepted.
        """
        src = as_torch_uint8(buf)
        entry = self._cache.reserve(
            chunk_key,
            nbytes,
            durable=False,
            durable_pin=True,
        )
        copy_tensor(entry.buffer, src, nbytes)
        if self._commit_l1 is not None:
            await maybe_await(self._commit_l1(chunk_key))
        written = await self._io.pwrite_from(entry.buffer, file_offset, nbytes)
        self._record_l2_write(written)
        self._cache.mark_durable(chunk_key)
        self._cache.release_durable_pin(chunk_key)
        if self._commit_l2 is not None:
            await maybe_await(self._commit_l2(chunk_key))
        return nbytes

    async def read_chunk_into_async(
        self,
        chunk_key: str,
        buf: torch.Tensor | cupy.ndarray,
        file_offset: int,
        nbytes: int,
        l2_durable: bool,
        protect_lookup: bool = False,
    ) -> int:
        """Read a chunk into a GPU-visible buffer, preferring L1.

        Args:
            chunk_key: chunk cache key.
            buf: destination tensor or CuPy view.
            file_offset: byte offset in L2 store.
            nbytes: bytes to read.
            l2_durable: whether L2 fallback is legal.
            protect_lookup: keep the L1 entry pinned for the active scheduler
                lookup lease until ``release_lookup_pins`` is called.

        Returns:
            Number of bytes read.

        Raises:
            RuntimeError: if L1 misses and L2 is not durable.
        """
        dst = as_torch_uint8(buf)
        entry = self._cache.pin_for_load(chunk_key)
        if entry is not None:
            if protect_lookup:
                entry.lookup_pin_count += 1
            try:
                copy_tensor(dst, entry.buffer, nbytes)
                return nbytes
            finally:
                self._cache.release_load_pin(chunk_key)
        if not l2_durable:
            raise RuntimeError(f"chunk is not durable in L2: {chunk_key}")
        entry = self._cache.reserve(chunk_key, nbytes, durable=True)
        entry.load_pin_count += 1
        if protect_lookup:
            entry.lookup_pin_count += 1
        try:
            read = await self._io.pread_into(entry.buffer, file_offset, nbytes)
            self._record_l2_read(read)
            copy_tensor(dst, entry.buffer, read)
            return read
        finally:
            self._cache.release_load_pin(chunk_key)

    def pin_chunks_for_lookup(self, chunk_keys: list[str]) -> None:
        """Protect lookup-hit chunks from local L1 eviction.

        Args:
            chunk_keys: cache keys returned by scheduler lookup.

        Async/thread-safety:
            Called on the worker thread before background store throttling can
            evict L1 entries for the current forward step.
        """
        for chunk_key in chunk_keys:
            self._cache.pin_for_lookup(chunk_key)

    def release_lookup_pins(self, chunk_keys: list[str]) -> None:
        """Release local L1 lookup pins after load/release.

        Args:
            chunk_keys: cache keys whose scheduler lookup lease ended.

        Async/thread-safety:
            Called on the worker thread after the corresponding load finishes.
        """
        for chunk_key in chunk_keys:
            self._cache.release_lookup_pin(chunk_key)

    async def write_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Reject unchunked writes for the L1-managed io_uring backend.

        Args:
            buf: source CuPy view.
            file_offset: byte offset in L2 store.
            nbytes: bytes to write; defaults to full buffer.

        Returns:
            This method does not return normally.

        Raises:
            RuntimeError: always. Call ``write_chunk_async`` so io_uring uses
                L1-managed buffers rather than temporary host staging memory.
        """
        del buf, file_offset, nbytes
        raise RuntimeError("io_uring memory transfer requires chunked writes")

    async def read_into_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Reject unchunked reads for the L1-managed io_uring backend.

        Args:
            buf: destination CuPy view.
            file_offset: byte offset in L2 store.
            nbytes: bytes to read; defaults to full buffer.

        Returns:
            This method does not return normally.

        Raises:
            RuntimeError: always. Call ``read_chunk_into_async`` so io_uring
                uses L1-managed buffers rather than temporary host staging
                memory.
        """
        del buf, file_offset, nbytes
        raise RuntimeError("io_uring memory transfer requires chunked reads")

    def stats(self) -> TransferStats:
        """Return transfer counters."""
        l1 = self._cache.stats()
        return TransferStats(
            l1_hits=l1.l1_hits,
            l1_misses=l1.l1_misses,
            l1_evictions=l1.l1_evictions,
            l1_bytes=l1.l1_bytes,
            l2_read_bytes=self._base_stats().l2_read_bytes,
            l2_write_bytes=self._base_stats().l2_write_bytes,
        )

    def close(self) -> None:
        """Close the underlying SSD-memory IO engine."""
        self._io.close()

    def _wrap_evict_callback(
        self, callback: Callable[[str], object] | None
    ) -> Callable[[str], None] | None:
        """Wrap a possibly-async eviction callback for PinnedL1Cache.

        Args:
            callback: sync or async eviction callback.

        Returns:
            Synchronous callback suitable for PinnedL1Cache.
        """
        if callback is None:
            return None

        def _wrapped(chunk_key: str) -> None:
            result = callback(chunk_key)
            if inspect.isawaitable(result):
                task = asyncio.create_task(result)
                task.add_done_callback(self._log_evict_result)

        return _wrapped

    @staticmethod
    def _log_evict_result(task: asyncio.Task) -> None:
        """Log asynchronous L1 eviction callback failures.

        Args:
            task: callback task created by ``_wrap_evict_callback``.
        """
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[IOURING] evict_l1 callback skipped: %s", exc)
