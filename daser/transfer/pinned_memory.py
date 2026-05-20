# SPDX-License-Identifier: Apache-2.0


class PinnedMemoryBuffer:
    """CUDA page-locked host byte buffer used by the transfer L1 tier.

    Args:
        size: Number of bytes to allocate and lock.

    Async/thread-safety:
        The buffer is synchronous and not internally synchronized. Callers must
        serialize concurrent mutation.
    """

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._size = size
        import cupy

        self._memory = cupy.cuda.alloc_pinned_memory(size if size > 0 else 1)
        self._closed = False

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PinnedMemoryBuffer":
        """Allocate a pinned buffer and initialize it from bytes.

        Args:
            data: Source byte data.

        Returns:
            Pinned memory buffer containing a copy of ``data``.

        Thread-safety:
            Allocates and initializes a new independent buffer.
        """
        view = memoryview(data).cast("B")
        buf = cls(len(view))
        buf.view()[: len(view)] = view
        return buf

    def view(self) -> memoryview:
        """Return a byte-addressable memoryview for the pinned buffer.

        Returns:
            Writable byte memoryview over the live pinned allocation.

        Thread-safety:
            The returned view is not synchronized; callers must serialize
            concurrent mutation.
        """
        return memoryview(self._memory).cast("B")[: self._size]

    def ptr_at(self, offset: int = 0) -> int:
        """Return the host pointer for a byte offset into the pinned buffer.

        Args:
            offset: Byte offset inside the allocation.

        Returns:
            Integer host pointer suitable for CUDA runtime copy APIs.

        Thread-safety:
            Safe to call concurrently while the buffer is open. The returned
            pointer must not be used after ``close()``.
        """
        if offset < 0 or offset > self._size:
            raise ValueError("offset is outside pinned buffer")
        return int(self._memory.ptr) + offset

    def to_bytes(self) -> bytes:
        """Return a bytes copy of the pinned buffer contents.

        Returns:
            Immutable bytes containing the buffer contents.

        Thread-safety:
            Reads the current contents without internal locking.
        """
        return bytes(self.view())

    def close(self) -> None:
        """Release the CUDA pinned pages.

        Returns:
            None.

        Thread-safety:
            Call when no other thread is reading or writing the buffer.
        """
        if self._closed:
            return
        self._memory = None
        self._closed = True

    def __len__(self) -> int:
        """Return the pinned buffer size in bytes.

        Returns:
            Allocated logical size in bytes.

        Thread-safety:
            Safe to call concurrently because the size is immutable.
        """
        return self._size
