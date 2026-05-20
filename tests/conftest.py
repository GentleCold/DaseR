# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest


class _CpuPinnedMemoryBuffer:
    """Test-only byte buffer used when the CI CuPy shim has no CUDA runtime.

    Args:
        size: Number of bytes to allocate.

    Async/thread-safety:
        The buffer is synchronous and not internally synchronized.
    """

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._size = size
        self._data = bytearray(size)

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview
    ) -> "_CpuPinnedMemoryBuffer":
        """Create a test buffer initialized from bytes.

        Args:
            data: Source byte data.

        Returns:
            Test buffer containing a copy of ``data``.

        Async/thread-safety:
            Allocates a new independent buffer.
        """
        view = memoryview(data).cast("B")
        buf = cls(len(view))
        buf.view()[:] = view
        return buf

    def view(self) -> memoryview:
        """Return a byte memoryview over the test buffer.

        Returns:
            Writable byte memoryview.

        Async/thread-safety:
            The returned view is not synchronized.
        """
        return memoryview(self._data)

    def ptr_at(self, offset: int = 0) -> int:
        """Return no device pointer because this helper is CPU-only.

        Args:
            offset: Byte offset requested by the caller.

        Raises:
            RuntimeError: Always, because CPU-only tests must not use CUDA copy
                paths.
        """
        raise RuntimeError(f"CPU test buffer has no CUDA pointer at offset {offset}")

    def close(self) -> None:
        """Release the test buffer.

        Returns:
            None.

        Async/thread-safety:
            Call when no other code is using views into the buffer.
        """
        self._data = bytearray()

    def __len__(self) -> int:
        """Return the current buffer size.

        Returns:
            Buffer size in bytes.

        Async/thread-safety:
            Safe for CPU test use.
        """
        return self._size


@pytest.fixture(autouse=True)
def _patch_pinned_memory_for_cpu_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch transfer tests when the installed CuPy package has no CUDA API.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Async/thread-safety:
        Applies a per-test module attribute patch before test execution.
    """
    try:
        import cupy

        alloc_pinned_memory = cupy.cuda.alloc_pinned_memory
    except (ImportError, AttributeError):
        alloc_pinned_memory = None
    if alloc_pinned_memory is not None:
        return

    import daser.transfer.iouring.pinned_pool as pinned_memory

    monkeypatch.setattr(
        pinned_memory,
        "PinnedMemoryBuffer",
        _CpuPinnedMemoryBuffer,
    )
