# SPDX-License-Identifier: Apache-2.0

"""SSD-to-host-memory I/O engines for the io_uring transfer backend."""

# Standard
import asyncio
import os

# Third Party
import torch

# First Party
from daser.connector.transfer.iouring.uring import NativeIOUring


class FileIOEngine:
    """Async interface for SSD to host-memory I/O engines.

    Async/thread-safety:
        Implementations must be coroutine-compatible and avoid blocking the
        worker event loop while storage operations are pending.
    """

    async def pread_into(self, dst: torch.Tensor, file_offset: int, nbytes: int) -> int:
        """Read bytes from the store into a host tensor.

        Args:
            dst: destination uint8 CPU tensor.
            file_offset: byte offset in the store file.
            nbytes: number of bytes to read.

        Returns:
            Number of bytes read.
        """
        raise NotImplementedError

    async def pwrite_from(
        self, src: torch.Tensor, file_offset: int, nbytes: int
    ) -> int:
        """Write bytes from a host tensor into the store.

        Args:
            src: source uint8 CPU tensor.
            file_offset: byte offset in the store file.
            nbytes: number of bytes to write.

        Returns:
            Number of bytes written.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close owned file descriptors or native resources."""
        raise NotImplementedError


class NativeIOUringEngine(FileIOEngine):
    """Native io_uring engine for SSD to host-memory I/O.

    Args:
        path: preallocated store file path.
        queue_depth: io_uring queue depth.

    Async/thread-safety:
        Operations execute in the event-loop default executor and are serialized
        by a lock because the minimal NativeIOUring wrapper owns one SQ/CQ pair.
    """

    def __init__(self, path: str, queue_depth: int = 64) -> None:
        self._fd = os.open(path, os.O_RDWR)
        self._ring = NativeIOUring(entries=queue_depth)
        self._lock = asyncio.Lock()

    async def pread_into(self, dst: torch.Tensor, file_offset: int, nbytes: int) -> int:
        """Read bytes into a CPU tensor using native io_uring."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None,
                self._ring.read_into,
                self._fd,
                dst,
                file_offset,
                nbytes,
            )

    async def pwrite_from(
        self, src: torch.Tensor, file_offset: int, nbytes: int
    ) -> int:
        """Write bytes from a CPU tensor using native io_uring."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None,
                self._ring.write_from,
                self._fd,
                src,
                file_offset,
                nbytes,
            )

    def close(self) -> None:
        """Close the native io_uring and store file descriptor."""
        self._ring.close()
        os.close(self._fd)


class PreadPwriteTestEngine(FileIOEngine):
    """Test-only async engine using `os.pread`/`os.pwrite`.

    Production IOUringMemTransferLayer construction does not instantiate this
    engine. Unit tests may inject it to exercise L1 logic without requiring
    kernel io_uring support.

    Args:
        path: preallocated store file path.
    """

    def __init__(self, path: str) -> None:
        self._fd = os.open(path, os.O_RDWR)

    async def pread_into(self, dst: torch.Tensor, file_offset: int, nbytes: int) -> int:
        """Read bytes into a CPU tensor using `os.pread` in an executor."""
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, os.pread, self._fd, nbytes, file_offset)
        view = dst[: len(data)]
        view.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        return len(data)

    async def pwrite_from(
        self, src: torch.Tensor, file_offset: int, nbytes: int
    ) -> int:
        """Write bytes from a CPU tensor using `os.pwrite` in an executor."""
        data = memoryview(src[:nbytes].contiguous().numpy())
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, os.pwrite, self._fd, data, file_offset)

    def close(self) -> None:
        """Close the test file descriptor."""
        os.close(self._fd)
