# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable


def _align_up(value: int, alignment: int) -> int:
    """Round a byte count up to an alignment boundary."""
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


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


class PinnedMemorySlice:
    """Logical byte slice leased from a pinned memory pool.

    Args:
        owner: Backing pinned allocation.
        offset: Byte offset inside ``owner``.
        size: Logical slice size in bytes.
        release: Callback invoked once when ``close`` releases the lease.

    Async/thread-safety:
        The slice is not internally synchronized. Pool metadata mutation is
        protected by the owning pool; callers must still serialize concurrent
        reads and writes to the returned memoryview.
    """

    def __init__(
        self,
        owner: PinnedMemoryBuffer,
        offset: int,
        size: int,
        release: Callable[[int, int], None],
    ) -> None:
        if offset < 0 or size < 0 or offset + size > len(owner):
            raise ValueError("slice is outside pinned buffer")
        self._owner = owner
        self._offset = offset
        self._size = size
        self._release = release
        self._closed = False

    def view(self) -> memoryview:
        """Return a byte-addressable memoryview for the leased slice.

        Returns:
            Writable byte memoryview over the live pinned slice.

        Thread-safety:
            The returned view is not synchronized. It must not be used after
            ``close`` returns.
        """
        if self._closed:
            raise RuntimeError("pinned slice is closed")
        return self._owner.view()[self._offset : self._offset + self._size]

    def ptr_at(self, offset: int = 0) -> int:
        """Return the host pointer for a byte offset into the slice.

        Args:
            offset: Byte offset inside this slice.

        Returns:
            Integer host pointer suitable for CUDA runtime copy APIs.

        Thread-safety:
            Safe while the slice remains open. The returned pointer must not be
            used after ``close``.
        """
        if offset < 0 or offset > self._size:
            raise ValueError("offset is outside pinned slice")
        if self._closed:
            raise RuntimeError("pinned slice is closed")
        return self._owner.ptr_at(self._offset + offset)

    def close(self) -> None:
        """Release the slice back to the owning pool.

        Returns:
            None.

        Thread-safety:
            Call only after no thread or CUDA operation is using the slice.
        """
        if self._closed:
            return
        self._closed = True
        self._release(self._offset, self._size)

    def __len__(self) -> int:
        """Return the logical slice size in bytes."""
        return self._size


class PinnedMemoryPool:
    """Fixed-capacity page-locked memory pool for transfer L1 entries.

    Args:
        capacity: Total logical bytes available for leases.
        alignment: Byte alignment used for each lease.

    Async/thread-safety:
        Pool metadata mutation is synchronous. Callers should invoke
        ``allocate`` and ``close`` from the transfer layer's metadata lock.
    """

    def __init__(self, capacity: int, alignment: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        self._alignment = alignment
        self._capacity = _align_up(capacity, alignment)
        self._memory = PinnedMemoryBuffer(self._capacity)
        self._free: list[tuple[int, int]] = [(0, self._capacity)]
        self._closed = False

    def allocate(self, size: int) -> PinnedMemorySlice | None:
        """Lease a pinned slice from the pool.

        Args:
            size: Logical byte count requested.

        Returns:
            A pinned slice, or ``None`` when no contiguous free range can hold
            the aligned request.

        Thread-safety:
            Mutates pool free-list metadata and should be serialized by the
            caller.
        """
        if self._closed:
            raise RuntimeError("pinned pool is closed")
        if size < 0:
            raise ValueError("size must be non-negative")
        aligned = _align_up(size if size > 0 else 1, self._alignment)
        for idx, (offset, free_size) in enumerate(self._free):
            if free_size < aligned:
                continue
            remaining = free_size - aligned
            if remaining:
                self._free[idx] = (offset + aligned, remaining)
            else:
                self._free.pop(idx)
            return PinnedMemorySlice(
                owner=self._memory,
                offset=offset,
                size=size,
                release=self._release,
            )
        return None

    def close(self) -> None:
        """Release the backing pinned allocation.

        Returns:
            None.

        Thread-safety:
            Call after all leases have been returned and no transfer is using
            the pool.
        """
        if self._closed:
            return
        self._closed = True
        self._free = []
        self._memory.close()

    def _release(self, offset: int, size: int) -> None:
        """Return an aligned slice to the free list."""
        aligned = _align_up(size if size > 0 else 1, self._alignment)
        self._free.append((offset, aligned))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for free_offset, free_size in self._free:
            if not merged:
                merged.append((free_offset, free_size))
                continue
            prev_offset, prev_size = merged[-1]
            prev_end = prev_offset + prev_size
            if free_offset <= prev_end:
                merged[-1] = (
                    prev_offset,
                    max(prev_end, free_offset + free_size) - prev_offset,
                )
            else:
                merged.append((free_offset, free_size))
        self._free = merged
