# SPDX-License-Identifier: Apache-2.0

"""Explicitly built, allocation-free native encoding for offline experiments."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any
import weakref

import numpy as np

from daser.compression.format import (
    CODEBOOK_ENTRIES,
    IO_ALIGNMENT,
    ONLINE_TILE_SCALARS,
    CompressedStoreGeometry,
    PlaneDescriptor,
    SlotHeader,
    SlotMode,
    digest_bytes,
)


def _load_library() -> ctypes.CDLL:
    path = Path(
        os.environ.get(
            "DASER_CPU_CODEC_LIBRARY",
            Path(__file__).with_name("libdaser_cpu_codec.so"),
        )
    )
    try:
        library = ctypes.CDLL(str(path.resolve()))
    except OSError as exc:
        raise RuntimeError(
            "Build the CPU codec explicitly with make -C daser/compression, "
            "or set DASER_CPU_CODEC_LIBRARY to the built shared library"
        ) from exc
    library.daser_cpu_encoder_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.daser_cpu_encoder_create.restype = ctypes.c_void_p
    library.daser_cpu_encoder_destroy.argtypes = [ctypes.c_void_p]
    library.daser_cpu_encoder_destroy.restype = None
    library.daser_cpu_encoder_workspace_bytes.argtypes = [ctypes.c_void_p]
    library.daser_cpu_encoder_workspace_bytes.restype = ctypes.c_uint64
    library.daser_cpu_encoder_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.daser_cpu_encoder_encode.restype = ctypes.c_uint32
    return library


def _byte_view(buffer: Any, *, writable: bool) -> np.ndarray:
    view = memoryview(buffer)
    if not view.c_contiguous:
        raise ValueError("buffers must be C-contiguous")
    if writable and view.readonly:
        raise ValueError("output must be writable")
    # frombuffer retains the exporter, accepts readonly input, and never copies.
    return np.frombuffer(view, dtype=np.uint8)


@dataclass(frozen=True)
class CpuEncodedSlot:
    """Mode and valid output bytes; the caller retains ownership of the buffer."""

    mode: SlotMode
    stored_length: int


class CpuSlotEncoder:
    """Reusable synchronous encoder, owned by one worker thread at a time.

    Args:
        geometry (CompressedStoreGeometry): Complete canonical slot geometry,
            using 256-scalar tiles and the master's one-page header.
        codebooks (bytes): Fifteen distinct high bytes per layer/K-or-V plane.

    Async/thread-safety:
        Native encoding releases the GIL. Use one instance per pool worker and
        call from an executor, never an asyncio hot path. Concurrent calls or
        close during encoding raise RuntimeError. No internal threads are used.
    """

    def __init__(self, geometry: CompressedStoreGeometry, codebooks: bytes) -> None:
        """Validate geometry/tables and allocate a private native workspace.

        Args:
            geometry (CompressedStoreGeometry): Full-slot BF16 wire geometry.
            codebooks (bytes): Immutable plane-major high-byte tables.

        Returns:
            None.

        Async/thread-safety:
            Synchronous startup allocation and shared-library loading.
        """
        if geometry.tile_scalars != ONLINE_TILE_SCALARS:
            raise ValueError("CPU experiment requires 256-scalar tiles")
        if geometry.slot_size >= 2**32 or geometry.num_slots > 2**64:
            raise ValueError("geometry exceeds the native wire integer limits")
        if 120 + geometry.plane_count * 48 >= IO_ALIGNMENT:
            raise ValueError("plane descriptor table exceeds the master header")
        if len(codebooks) != geometry.plane_count * CODEBOOK_ENTRIES:
            raise ValueError("codebooks do not match geometry")
        for start in range(0, len(codebooks), CODEBOOK_ENTRIES):
            if (
                len(set(codebooks[start : start + CODEBOOK_ENTRIES]))
                != CODEBOOK_ENTRIES
            ):
                raise ValueError("each codebook must contain 15 distinct bytes")
        tiles = (geometry.plane_scalars + 255) // 256
        header = SlotHeader(
            slot_id=0,
            raw_length=geometry.slot_size,
            stored_length=0,
            tile_scalars=geometry.tile_scalars,
            num_layers=geometry.num_layers,
            codebook_hash=digest_bytes(codebooks),
            # Match the GPU producer's 0x01 + 31 zero-byte placeholder.
            # Reference decode remains hash-checking by default; experimental
            # callers must explicitly opt out and compare restored bytes.
            raw_hash=b"\x01" + bytes(31),
            descriptors=tuple(
                PlaneDescriptor(
                    p // 2, p % 2, geometry.plane_scalars, tiles, *([0] * 7)
                )
                for p in range(geometry.plane_count)
            ),
        ).pack()
        self._geometry = geometry
        self._lock = threading.Lock()
        self._library = _load_library()
        self._context = self._library.daser_cpu_encoder_create(
            geometry.plane_count,
            geometry.plane_scalars,
            geometry.slot_size,
            ctypes.c_char_p(codebooks),
            ctypes.c_char_p(header),
        )
        if not self._context:
            raise MemoryError("could not allocate CPU codec workspace")
        self._finalizer = weakref.finalize(
            self, self._library.daser_cpu_encoder_destroy, self._context
        )
        self._workspace_bytes = int(
            self._library.daser_cpu_encoder_workspace_bytes(self._context)
        )

    @property
    def workspace_bytes(self) -> int:
        """Return native bytes per instance, excluding caller buffers/allocator.

        Async/thread-safety:
            Immutable integer; safe to read from any thread.
        """
        return self._workspace_bytes

    def encode_into(self, raw: Any, output: Any, *, slot_id: int) -> CpuEncodedSlot:
        """Encode a complete canonical slot into a disjoint preallocated buffer.

        Args:
            raw (buffer): Contiguous CPU bytes in [layer, K/V, token, head, dim]
                order, exactly geometry.slot_size bytes. Readonly is supported.
            output (buffer): Writable contiguous CPU capacity of at least one
                raw slot. Its entire range must be disjoint from raw.
            slot_id (int): Logical ID in [0, geometry.num_slots).

        Returns:
            CpuEncodedSlot: Mode and length; only output[:stored_length] is
            valid. RAW copies the original canonical bytes. No content hash is
            computed, exactly as for GPU-produced records.

        Async/thread-safety:
            Synchronous, GIL-releasing native work; buffers must remain stable
            until return. Rejects simultaneous use of this instance.
        """
        if not 0 <= slot_id < self._geometry.num_slots:
            raise ValueError("slot_id is outside the compressed store")
        source = _byte_view(raw, writable=False)
        target = _byte_view(output, writable=True)
        if source.nbytes != self._geometry.slot_size:
            raise ValueError("raw slot length does not match geometry")
        if target.nbytes < self._geometry.slot_size:
            raise ValueError("output must hold a complete raw slot")
        src, dst = source.ctypes.data, target.ctypes.data
        if src < dst + target.nbytes and dst < src + source.nbytes:
            raise ValueError("input and output must not overlap")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("encoder is already in use")
        try:
            if not self._finalizer.alive:
                raise RuntimeError("encoder is closed")
            mode = ctypes.c_uint32()
            length = self._library.daser_cpu_encoder_encode(
                self._context, src, dst, slot_id, ctypes.byref(mode)
            )
            return CpuEncodedSlot(SlotMode(mode.value), int(length))
        finally:
            self._lock.release()

    def close(self) -> None:
        """Release native workspace, idempotently, returning None.

        Async/thread-safety:
            Synchronous; raises RuntimeError if this encoder is in use.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("encoder is already in use")
        try:
            self._finalizer()
        finally:
            self._lock.release()
