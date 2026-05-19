# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any


@dataclass
class CudaIPCBuffer:
    """Opened CUDA IPC memory as a byte-addressable CuPy array.

    Attributes:
        array: CuPy uint8 ndarray covering the remote allocation.
        ptr: CUDA device pointer returned by ``ipcOpenMemHandle``.

    Async/thread-safety:
        The opened handle is process-local and must be closed by the same
        process after transfer operations finish.
    """

    array: Any
    ptr: int

    def close(self) -> None:
        """Close the CUDA IPC memory handle."""
        from cupy.cuda import runtime  # Third Party

        runtime.ipcCloseMemHandle(self.ptr)


def open_cuda_ipc_buffer(handle: bytes, nbytes: int) -> CudaIPCBuffer:
    """Open a CUDA IPC handle as a CuPy uint8 ndarray.

    Args:
        handle: Raw 64-byte CUDA IPC memory handle.
        nbytes: Number of bytes in the exported allocation.

    Returns:
        CudaIPCBuffer containing a byte array view and close method.
    """
    import cupy  # Third Party
    from cupy.cuda import runtime  # Third Party

    ptr = runtime.ipcOpenMemHandle(handle)
    owner = object()
    memory = cupy.cuda.UnownedMemory(ptr, nbytes, owner)
    memptr = cupy.cuda.MemoryPointer(memory, 0)
    array = cupy.ndarray((nbytes,), dtype=cupy.uint8, memptr=memptr)
    return CudaIPCBuffer(array=array, ptr=ptr)


def export_cuda_ipc_handle(array: Any) -> bytes:
    """Export a CuPy-compatible array's base pointer as a CUDA IPC handle.

    Args:
        array: CuPy ndarray or object exposing ``.data.ptr``.

    Returns:
        Raw CUDA IPC memory handle bytes.
    """
    from cupy.cuda import runtime  # Third Party

    return runtime.ipcGetMemHandle(array.data.ptr)
