# SPDX-License-Identifier: Apache-2.0

"""Stateless copy/marshalling helpers for the io_uring transfer layer.

These functions move bytes between CPU buffers, CuPy device arrays, and the
pinned host slices used by the L1 tier. They hold no tiering state and take no
locks, so they are safe to call from either the event loop or executor threads
as long as the caller owns the buffers passed in.
"""

# Standard
from typing import Any

# First Party
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice

#: One grouped copy chunk: (target_offset, source_slice, source_offset, nbytes).
CopyChunk = tuple[int, PinnedMemorySlice, int, int]


def destination_copy_event(dst: Any, chunks: list[CopyChunk]) -> Any | None:
    """Create an unrecorded event for a CUDA destination copy, or return None.

    Args:
        dst: Byte-addressable destination with an optional private copy stream.
        chunks: Nonempty source/destination ranges about to be copied.

    Returns:
        CUDA event on the destination device, or None for synchronous CPU copies.

    Async/thread-safety:
        Call on the transfer event loop before submission. The caller must retain
        sources, record this event after submission and poll it before release.
    """
    if not chunks:
        return None
    target = slice_dst(dst, chunks[0][0], chunks[0][3])
    if cuda_array_ptr(target) is None:
        return None
    import cupy

    with cupy.cuda.Device(int(target.device.id)):
        interprocess = bool(getattr(dst, "copy_event_interprocess", False))
        # Only the explicit deferred path crosses the process boundary. Raw
        # and synchronous transfers keep their existing event behavior.
        return cupy.cuda.Event(
            disable_timing=True,
            interprocess=interprocess,
        )


def record_destination_copy_event(dst: Any, event: Any) -> None:
    """Record a completion event after copies on the destination's stream.

    Args:
        dst: Destination that supplied the private copy stream, or stream zero.
        event: Event created by destination_copy_event before submission.

    Returns:
        None.

    Async/thread-safety:
        Call on the same transfer loop immediately after the grouped submission.
        Recording is nonblocking and must also run after a partial copy failure.
    """
    import cupy

    target = slice_dst(dst, 0, 0)
    with cupy.cuda.Device(int(target.device.id)):
        event.record(cupy.cuda.ExternalStream(int(getattr(dst, "copy_stream_ptr", 0))))
    retain = getattr(dst, "record_copy_event", None)
    if retain is not None:
        retain(event)


def cuda_array_ptr(dst: Any) -> int | None:
    """Return a CUDA device pointer for a CuPy-like array destination.

    Args:
        dst: candidate destination object.

    Returns:
        Integer device pointer when ``dst`` exposes the CuPy ``data.ptr``
        interface, otherwise None.
    """
    data = getattr(dst, "data", None)
    ptr = getattr(data, "ptr", None)
    if ptr is None:
        return None
    return int(ptr)


def slice_dst(dst: Any, offset: int, nbytes: int) -> Any:
    """Return a writable destination slice for ``[offset, offset + nbytes)``.

    Args:
        dst: writable buffer or CuPy ndarray.
        offset: byte offset into ``dst``.
        nbytes: number of bytes the slice must cover.

    Returns:
        A sliced view appropriate for the destination type.
    """
    if hasattr(dst, "set"):
        try:
            return dst[offset : offset + nbytes]
        except (TypeError, KeyError, IndexError):
            if offset == 0:
                return dst
            raise
    if isinstance(dst, bytearray | memoryview):
        return memoryview(dst).cast("B")[offset : offset + nbytes]
    try:
        return dst[offset : offset + nbytes]
    except (TypeError, KeyError, IndexError):
        pass
    return memoryview(dst).cast("B")[offset : offset + nbytes]


def slice_src(src: Any, offset: int, nbytes: int) -> Any:
    """Return a readable source slice for ``[offset, offset + nbytes)``.

    Args:
        src: readable buffer or CuPy ndarray.
        offset: byte offset into ``src``.
        nbytes: number of bytes the slice must cover.

    Returns:
        A sliced view appropriate for the source type.
    """
    if hasattr(src, "get"):
        return src[offset : offset + nbytes]
    try:
        return src[offset : offset + nbytes]
    except (TypeError, KeyError, IndexError):
        pass
    return memoryview(src).cast("B")[offset : offset + nbytes]


def copy_src_to_pinned(
    src: Any,
    pinned: PinnedMemorySlice,
    target_offset: int,
    nbytes: int,
) -> None:
    """Copy bytes from a CPU or CuPy source into pinned host memory.

    Args:
        src: readable byte buffer or CuPy ndarray.
        pinned: destination slice leased from the L1 pool.
        target_offset: byte offset into ``pinned`` to write at.
        nbytes: number of bytes to copy.
    """
    if hasattr(src, "data") and getattr(src.data, "ptr", None) is not None:
        import cupy
        from cupy.cuda import runtime

        with cupy.cuda.Device(int(src.device.id)):
            runtime.memcpy(
                pinned.ptr_at(target_offset),
                int(src.data.ptr),
                nbytes,
                runtime.memcpyDeviceToHost,
            )
        return
    pinned.view()[target_offset : target_offset + nbytes] = memoryview(src).cast("B")[
        :nbytes
    ]


def coalesce_copy_chunks(chunks: list[CopyChunk]) -> list[CopyChunk]:
    """Merge adjacent copies sharing one source slice and contiguous offsets.

    Args:
        chunks: grouped copy chunks to merge.

    Returns:
        Chunks sorted by target offset with adjacent runs merged.
    """
    ordered = sorted(chunks, key=lambda item: item[0])
    merged: list[CopyChunk] = []
    for target_offset, data, source_offset, nbytes in ordered:
        if not merged:
            merged.append((target_offset, data, source_offset, nbytes))
            continue
        prev_target, prev_data, prev_source, prev_nbytes = merged[-1]
        if (
            prev_data is data
            and target_offset == prev_target + prev_nbytes
            and source_offset == prev_source + prev_nbytes
        ):
            merged[-1] = (prev_target, prev_data, prev_source, prev_nbytes + nbytes)
            continue
        merged.append((target_offset, data, source_offset, nbytes))
    return merged


def copy_grouped_to_cuda_dst(dst: Any, chunks: list[CopyChunk]) -> None:
    """Copy grouped pinned ranges into a CUDA destination.

    Args:
        dst: CuPy ndarray destination.
        chunks: grouped copy chunks from the L1 tier.
    """
    import cupy
    from cupy.cuda import runtime

    ordered = sorted(chunks, key=lambda item: item[0])
    merged: list[tuple[int, int, int]] = []
    for target_offset, data, source_offset, nbytes in ordered:
        source_ptr = data.ptr_at(source_offset)
        if not merged:
            merged.append((target_offset, source_ptr, nbytes))
            continue
        prev_target, prev_source, prev_nbytes = merged[-1]
        if (
            target_offset == prev_target + prev_nbytes
            and source_ptr == prev_source + prev_nbytes
        ):
            merged[-1] = (prev_target, prev_source, prev_nbytes + nbytes)
            continue
        merged.append((target_offset, source_ptr, nbytes))

    stream_ptr = int(getattr(dst, "copy_stream_ptr", 0))
    for target_offset, source_ptr, nbytes in merged:
        target = slice_dst(dst, target_offset, nbytes)
        dst_ptr = cuda_array_ptr(target)
        if dst_ptr is None:
            raise TypeError("grouped CUDA copy target lost CUDA array interface")
        with cupy.cuda.Device(int(target.device.id)):
            runtime.memcpyAsync(
                dst_ptr,
                source_ptr,
                nbytes,
                runtime.memcpyHostToDevice,
                stream_ptr,
            )


def copy_grouped_to_dst(dst: Any, chunks: list[CopyChunk]) -> None:
    """Copy source chunks into the destination without staging repacks.

    Args:
        dst: writable buffer or CuPy ndarray destination.
        chunks: grouped copy chunks from the L1 tier.
    """
    if not chunks:
        return
    first_target = slice_dst(dst, chunks[0][0], chunks[0][3])
    if cuda_array_ptr(first_target) is not None:
        copy_grouped_to_cuda_dst(dst, chunks)
        return

    for target_offset, data, source_offset, nbytes in coalesce_copy_chunks(chunks):
        target = slice_dst(dst, target_offset, nbytes)
        if hasattr(target, "set"):
            import numpy

            host = numpy.frombuffer(
                data.view()[source_offset : source_offset + nbytes],
                dtype=numpy.uint8,
                count=nbytes,
            )
            target.set(host)
            continue
        dst_view = memoryview(dst).cast("B")
        dst_view[target_offset : target_offset + nbytes] = data.view()[
            source_offset : source_offset + nbytes
        ]
