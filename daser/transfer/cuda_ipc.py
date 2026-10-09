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
    pci_bus_id: str | None = None,
    local_ptr: int | None = None,
    allocation_offset: int = 0,
) -> CudaIPCBuffer:
    """Open a CUDA IPC handle as a CuPy uint8 ndarray.

    Args:
        handle: Raw 64-byte CUDA IPC memory handle.
        nbytes: Number of bytes in the exported allocation.
        pci_bus_id: PCI bus ID of the GPU holding the exported allocation.
            When provided, the receiver selects its own ordinal for that GPU
            before opening the IPC handle, so exporter and receiver may see
            different ``CUDA_VISIBLE_DEVICES`` sets.
        local_ptr: raw device pointer to use when exporter and receiver are in
            the same process.
        allocation_offset: byte offset from the opened allocation base to the
            exported tensor view.

    Returns:
        CudaIPCBuffer containing a byte array view and close method.
    """
    import cupy  # Third Party
    from cupy.cuda import runtime  # Third Party

    if pci_bus_id is not None:
        cupy.cuda.Device(cuda_device_for_pci_bus_id(pci_bus_id)).use()
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


def cuda_array_pci_bus_id(array: Any) -> str:
    """Return the PCI bus ID of the GPU holding a CuPy-compatible array.

    Device ordinals are local to each process's ``CUDA_VISIBLE_DEVICES``, so
    an exporter and a receiver that see different device sets disagree on
    them. The PCI bus ID names the same physical GPU in every process.

    Args:
        array: CuPy ndarray or compatible object exposing ``.device.id``.

    Returns:
        PCI bus ID string such as ``0000:38:00.0``.
    """
    from cupy.cuda import runtime  # Third Party

    return str(runtime.deviceGetPCIBusId(int(array.device.id)))


def cuda_device_for_pci_bus_id(pci_bus_id: str) -> int:
    """Return this process's CUDA device ordinal for a PCI bus ID.

    Args:
        pci_bus_id: PCI bus ID reported by :func:`cuda_array_pci_bus_id`.

    Returns:
        Local CUDA device ordinal.

    Raises:
        cupy.cuda.runtime.CUDARuntimeError: the GPU is not visible to this
            process.
    """
    from cupy.cuda import runtime  # Third Party

    return int(runtime.deviceGetByPCIBusId(pci_bus_id))


def cuda_allocation_base_and_offset(device_ptr: int) -> tuple[int, int]:
    """Return the CUDA allocation base and byte offset for a tensor pointer.

    Args:
        device_ptr: CUDA device pointer exported through IPC.

    Returns:
        Tuple of allocation base pointer and byte offset. When the CUDA driver
        query is unavailable, the pointer itself is used as the base.

    Async/thread-safety:
        Read-only CUDA driver query safe during worker transfer preparation.
    """
    try:
        from cuda.bindings import driver as cuda_driver

        result, base_ptr, _allocation_size = cuda_driver.cuMemGetAddressRange(
            device_ptr
        )
        if result == cuda_driver.CUresult.CUDA_SUCCESS:
            base = int(base_ptr)
            return base, int(device_ptr) - base
    except Exception:  # noqa: BLE001
        pass
    return int(device_ptr), 0
