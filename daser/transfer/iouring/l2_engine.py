# SPDX-License-Identifier: Apache-2.0

"""io_uring positioned I/O engine for the L2 SSD tier.

This engine owns the O_DIRECT file descriptor, the pool of native io_uring
rings, and the executor used to offload blocking completion waits. It holds no
tiering or L1 state: callers pass absolute byte offsets and memoryviews. Write
scheduling and the pending-write bookkeeping that ties a write to an L1 pool
slice stay in the transfer-layer orchestrator.
"""

# Standard
import concurrent.futures
import os
import threading

# First Party
from daser.transfer.iouring.native import NativeIOUring


class L2IoEngine:
    """Round-robin io_uring positioned reads/writes against the L2 store file.

    Args:
        path: pre-allocated or creatable L2 store file.
        l2_bytes: SSD-tier capacity; the file is truncated to this size.
        io_workers: number of native io_uring rings and executor threads.

    Async/thread-safety:
        Ring selection is serialized by an internal lock. Blocking syscalls are
        meant to be offloaded onto ``executor`` by the caller, which awaits the
        completion on its event loop.
    """

    def __init__(self, path: str, l2_bytes: int, io_workers: int) -> None:
        if l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a+b") as f:
            f.truncate(l2_bytes)
        self._fd = os.open(path, os.O_RDWR | os.O_DIRECT)
        self._urings = [NativeIOUring(entries=64) for _ in range(io_workers)]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="daser-iouring",
        )
        self._uring_lock = threading.Lock()
        self._next_uring_index = 0

    @property
    def ring_count(self) -> int:
        """Return the number of io_uring rings (used to bound read batches)."""
        return len(self._urings)

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the executor used to offload blocking io_uring waits."""
        return self._executor

    def next_uring(self) -> NativeIOUring:
        """Return the next native io_uring instance for one L2 operation."""
        with self._uring_lock:
            uring = self._urings[self._next_uring_index]
            self._next_uring_index = (self._next_uring_index + 1) % len(self._urings)
            return uring

    def read_into(self, file_offset: int, dst: memoryview, uring: NativeIOUring) -> int:
        """Blocking io_uring L2 read into ``dst``.

        Args:
            file_offset: byte offset in the L2 store file.
            dst: writable memoryview to fill.
            uring: ring to submit on (from ``next_uring``).

        Returns:
            Number of bytes read.
        """
        return uring.read_into(self._fd, file_offset, dst)

    def write(self, file_offset: int, data: memoryview, uring: NativeIOUring) -> int:
        """Blocking io_uring L2 write of ``data``.

        Args:
            file_offset: byte offset in the L2 store file.
            data: readable memoryview to persist.
            uring: ring to submit on (from ``next_uring``).

        Returns:
            Number of bytes written.
        """
        return uring.write(self._fd, file_offset, data)

    def close(self) -> None:
        """Shut down the executor, close rings, and close the file descriptor."""
        self._executor.shutdown(wait=True)
        for uring in self._urings:
            uring.close()
        os.close(self._fd)
