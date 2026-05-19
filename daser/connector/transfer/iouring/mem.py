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
        try:
            entry = self._cache.reserve(chunk_key, nbytes, durable=True)
        except MemoryError:
            host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=dst.is_cuda)
            read = await self._io.pread_into(host, file_offset, nbytes)
            self._record_l2_read(read)
            copy_tensor(dst, host, read)
            return read
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

    async def read_chunk_host_async(
        self,
        chunk_key: str,
        file_offset: int,
        nbytes: int,
        l2_durable: bool,
    ) -> torch.Tensor:
        """Return a host L1 buffer for a chunk, filling from L2 on miss.

        Args:
            chunk_key: chunk cache key.
            file_offset: byte offset in L2 store.
            nbytes: bytes to read.
            l2_durable: whether L2 fallback is legal.

        Returns:
            CPU uint8 tensor containing the requested chunk bytes.

        Raises:
            RuntimeError: if L1 misses and L2 is not durable.
        """
        entry = self._cache.pin_for_load(chunk_key)
        if entry is not None:
            return entry.buffer[:nbytes]
        if not l2_durable:
            raise RuntimeError(f"chunk is not durable in L2: {chunk_key}")
        entry = self._cache.reserve(chunk_key, nbytes, durable=True)
        entry.load_pin_count += 1
        read = await self._io.pread_into(entry.buffer, file_offset, nbytes)
        self._record_l2_read(read)
        return entry.buffer[:read]

    def release_chunk_host(self, chunk_key: str) -> None:
        """Release a host buffer returned by ``read_chunk_host_async``.

        Args:
            chunk_key: chunk cache key whose load pin should be released.

        Async/thread-safety:
            Called by the worker thread after synchronous H2D copy completes.
            Access is serialized by vLLM connector execution.
        """
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
        """Compatibility write path that persists directly to L2.

        Args:
            buf: source CuPy view.
            file_offset: byte offset in L2 store.
            nbytes: bytes to write; defaults to full buffer.

        Returns:
            Number of bytes written.
        """
        src = as_torch_uint8(buf)
        count = int(nbytes if nbytes is not None else src.numel())
        host = torch.empty(count, dtype=torch.uint8, pin_memory=src.is_cuda)
        copy_tensor(host, src, count)
        written = await self._io.pwrite_from(host, file_offset, count)
        self._record_l2_write(written)
        return written

    async def read_into_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Compatibility read path that reads directly from L2.

        Args:
            buf: destination CuPy view.
            file_offset: byte offset in L2 store.
            nbytes: bytes to read; defaults to full buffer.

        Returns:
            Number of bytes read.
        """
        dst = as_torch_uint8(buf)
        count = int(nbytes if nbytes is not None else dst.numel())
        host = torch.empty(count, dtype=torch.uint8, pin_memory=dst.is_cuda)
        read = await self._io.pread_into(host, file_offset, count)
        self._record_l2_read(read)
        copy_tensor(dst, host, read)
        return read

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
                asyncio.create_task(result)

        return _wrapped
