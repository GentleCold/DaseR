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
        owns_handle: True when this process opened an IPC handle and must close it.

    Async/thread-safety:
        The opened handle is process-local and must be closed by the same
        process after transfer operations finish.
    """

    array: Any
    ptr: int
    owns_handle: bool = True

    def close(self) -> None:
        """Close the CUDA IPC memory handle."""
        import cupy  # Third Party
        from cupy.cuda import runtime  # Third Party

        if self.owns_handle:
            with cupy.cuda.Device(int(self.array.device.id)):
                runtime.ipcCloseMemHandle(self.ptr)


def open_cuda_ipc_buffer(
    handle: bytes,
    nbytes: int,
    device_id: int | None = None,
    local_ptr: int | None = None,
    allocation_offset: int = 0,
) -> CudaIPCBuffer:
    """Open a CUDA IPC handle as a CuPy uint8 ndarray.

    Args:
        handle: Raw 64-byte CUDA IPC memory handle.
        nbytes: Number of bytes in the exported allocation.
        device_id: CUDA device ordinal for the exporting allocation. When
            provided, the receiver initializes and selects that device before
            opening the IPC handle.
        local_ptr: raw device pointer to use when exporter and receiver are in
            the same process.
        allocation_offset: byte offset from the opened allocation base to the
            exported tensor view.

    Returns:
        CudaIPCBuffer containing a byte array view and close method.
    """
    import cupy  # Third Party
    from cupy.cuda import runtime  # Third Party

    if device_id is not None:
        cupy.cuda.Device(device_id).use()
    if allocation_offset < 0:
        raise ValueError("allocation_offset must be non-negative")
    owns_handle = local_ptr is None
    ptr = local_ptr if local_ptr is not None else runtime.ipcOpenMemHandle(handle)
    owner = object()
    memory = cupy.cuda.UnownedMemory(ptr, nbytes + allocation_offset, owner)
    memptr = cupy.cuda.MemoryPointer(memory, 0)
    if allocation_offset:
        memptr = cupy.cuda.MemoryPointer(memory, allocation_offset)
    array = cupy.ndarray((nbytes,), dtype=cupy.uint8, memptr=memptr)
    return CudaIPCBuffer(array=array, ptr=ptr, owns_handle=owns_handle)


def export_cuda_ipc_handle(array: Any) -> bytes:
    """Export a CuPy-compatible array's base pointer as a CUDA IPC handle.

    Args:
        array: CuPy ndarray or object exposing ``.data.ptr``.

    Returns:
        Raw CUDA IPC memory handle bytes.
    """
    from cupy.cuda import runtime  # Third Party

    return runtime.ipcGetMemHandle(array.data.ptr)


def cuda_array_pointer(array: Any) -> int:
    """Return the raw device pointer for a CuPy-compatible array.

    Args:
        array: CuPy ndarray or compatible object exposing ``.data.ptr``.

    Returns:
        Raw CUDA device pointer as an integer.
    """
    return int(array.data.ptr)


def cuda_array_device_id(array: Any) -> int:
    """Return the CUDA device ordinal for a CuPy-compatible array.

    Args:
        array: CuPy ndarray or compatible object exposing ``.device.id``.

    Returns:
        CUDA device ordinal.
    """
    return int(array.device.id)
