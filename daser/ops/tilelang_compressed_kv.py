# SPDX-License-Identifier: Apache-2.0
"""TileLang fused online compressed-KV store and load operators.

The operators in this module deliberately use the existing variable-length
record format.  Store is exposed as a small kernel bundle because a global
prefix is required before variable-length records can be compacted; all
stages are launched on the caller's stream and no host round trip occurs
between them.  Load is a single kernel because its record offsets and lengths
are already known when the server returns a read plan.
"""

# Do not enable ``from __future__ import annotations``. TileLang inspects the
# concrete annotations while constructing the prim_func.

from dataclasses import dataclass
from typing import Any

import tilelang
import tilelang.language as T

_STORE_THREADS = 1024
_COMPACT_THREADS = 256
_HEADER_BYTES = 4096
_SLOT_FIXED_HEADER_BYTES = 120
_DESCRIPTOR_BYTES = 48
_CODEBOOK_ENTRIES = 15


@dataclass(frozen=True)
class TileLangStoreKernels:
    """Compiled TileLang stages used by the online store pipeline."""

    encode: Any
    prefix: Any
    layout: Any
    compact: Any
    raw_restore: Any


_STORE_CACHE: dict[tuple[int, int, int, int, int], TileLangStoreKernels] = {}
_LOAD_CACHE: dict[tuple[int, int, int, int], Any] = {}


def compile_store_kernels(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
) -> TileLangStoreKernels:
    """Compile the online TileLang store stages for one KV geometry.

    Args:
        num_blocks: Number of physical vLLM KV blocks in the cache.
        num_planes: Number of layer/K-or-V planes.
        plane_scalars: BF16 scalars in one plane.
        tile_scalars: Scalar quantum used by the packed record.

    Returns:
        Cached encode, prefix, layout, compact, and raw-restore kernels.

    Raises:
        ValueError: If the geometry is not positive or exceeds the supported
            one-tile shared-memory envelope.

    Async/thread-safety:
        Compilation is synchronous and intended for registration-time
        warmup. The process-local cache is only mutated during that phase.
    """
    if min(num_blocks, num_planes, plane_scalars, tile_scalars) <= 0:
        raise ValueError("TileLang store geometry must be positive")
    tiles_per_plane = (plane_scalars + tile_scalars - 1) // tile_scalars
    if tiles_per_plane > 1024:
        raise ValueError("TileLang store supports at most 1024 tiles per plane")
    key = (num_blocks, num_planes, plane_scalars, tile_scalars, tiles_per_plane)
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached
    kernels = TileLangStoreKernels(
        encode=tilelang.compile(
            _build_store_encode(
                num_blocks=num_blocks,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                tile_scalars=tile_scalars,
                tiles_per_plane=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
        prefix=tilelang.compile(
            _build_store_prefix(
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                max_tiles=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
        layout=tilelang.compile(
            _build_store_layout(
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                max_tiles=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
        compact=tilelang.compile(
            _build_store_compact(
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                max_tiles=tiles_per_plane,
            ),
            target="cuda",
            execution_backend="cython",
        ),
        raw_restore=tilelang.compile(
            _build_raw_restore(
                num_blocks=num_blocks,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
            ),
            target="cuda",
            execution_backend="cython",
        ),
    )
    _STORE_CACHE[key] = kernels
    return kernels


def compile_load_kernel(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    staging_bytes: int,
) -> Any:
    """Compile the single-launch TileLang fused compressed load kernel.

    Args:
        num_blocks: Number of destination vLLM KV blocks.
        num_planes: Number of layer/K-or-V planes.
        plane_scalars: BF16 scalars in one plane.
        tile_scalars: Scalar quantum encoded in each record.
        staging_bytes: Representative staging capacity used for warmup metadata;
            the compiled kernel accepts the actual extent at runtime.

    Returns:
        Cached TileLang JIT kernel. Slot count and staging extent remain runtime
        dimensions, so request metadata does not trigger compilation.

    Raises:
        ValueError: If geometry is not positive or the tile count is too large.

    Async/thread-safety:
        Compilation is synchronous and should happen before request traffic.
        The returned kernel launches asynchronously on the current PyTorch
        stream.
    """
    if min(num_blocks, num_planes, plane_scalars, tile_scalars, staging_bytes) <= 0:
        raise ValueError("TileLang load geometry must be positive")
    tiles_per_plane = (plane_scalars + tile_scalars - 1) // tile_scalars
    if tiles_per_plane > 1024:
        raise ValueError("TileLang load supports at most 1024 tiles per plane")
    # The staging extent is a runtime tensor dimension. Keep the argument in
    # the public API for callers that document their pool capacity, but do not
    # specialize the JIT cache on that capacity.
    key = (num_blocks, num_planes, plane_scalars, tile_scalars)
    cached = _LOAD_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = tilelang.compile(
        _build_load(
            num_blocks=num_blocks,
            num_planes=num_planes,
            plane_scalars=plane_scalars,
            tile_scalars=tile_scalars,
            tiles_per_plane=tiles_per_plane,
            staging_bytes=staging_bytes,
        ),
        target="cuda",
        execution_backend="cython",
    )
    _LOAD_CACHE[key] = kernel
    return kernel


def _build_store_encode(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    tiles_per_plane: int,
) -> Any:
    """Build one tile-per-block store encoder."""
    n_slots = T.dynamic("N")
    scratch_bytes = T.dynamic("R")
    count_items = T.dynamic("C")

    @T.prim_func
    def main(
        kv_bits: T.Tensor((num_blocks, num_planes, plane_scalars), "uint16"),
        block_ids: T.Tensor((n_slots,), "int32"),
        lookup: T.Tensor((num_planes * 256,), "uint8"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        tile_counts: T.Tensor((count_items,), "uint32"),
        slot_stride: T.int32,
        plane_record_bytes: T.int32,
        tile_escape_capacity: T.int32,
        block_tokens: T.int32,
        valid_token_count: T.int32,
    ):
        del slot_stride
        with T.Kernel(n_slots, num_planes, tiles_per_plane, threads=_STORE_THREADS) as (
            bx,
            by,
            bz,
        ):
            warp_counts = T.alloc_shared((32,), "uint32")
            warp_prefix = T.alloc_shared((32,), "uint32")
            for tx in T.Parallel(_STORE_THREADS):
                slot = bx
                plane = by
                tile = bz
                lane = tx & 31
                warp = tx // 32
                scalar = tile * tile_scalars + tx
                active = scalar < plane_scalars
                block_id = block_ids[slot]
                row_scalars = plane_scalars // block_tokens
                valid = active and (
                    slot * block_tokens + scalar // row_scalars < valid_token_count
                )
                bits = (
                    T.cast(kv_bits[block_id, plane, scalar], T.uint16) if valid else 0
                )
                high = (bits >> 8) & 255
                code = T.cast(lookup[plane * 256 + high], T.uint8) if valid else 15
                low_base = (
                    slot * (4096 + num_planes * plane_record_bytes)
                    + 4096
                    + plane * plane_record_bytes
                )
                if active:
                    scratch[low_base + scalar] = T.cast(bits & 255, T.uint8)
                if active and (tx & 1) == 0:
                    next_scalar = scalar + 1
                    next_active = next_scalar < plane_scalars
                    next_valid = next_active and (
                        slot * block_tokens + next_scalar // row_scalars
                        < valid_token_count
                    )
                    next_bits = (
                        T.cast(kv_bits[block_id, plane, next_scalar], T.uint16)
                        if next_valid
                        else 0
                    )
                    next_high = (next_bits >> 8) & 255
                    next_code = (
                        T.cast(lookup[plane * 256 + next_high], T.uint8)
                        if next_valid
                        else 15
                    )
                    scratch[low_base + plane_scalars + scalar // 2] = T.cast(
                        code | (next_code << 4), T.uint8
                    )

                # Invalid tail scalars are encoded as zero-bit escapes, matching
                # the existing format's canonical zero fill for partial slots.
                mask = T.ballot(active and code == 15)
                if lane == 0:
                    warp_counts[warp] = T.popcount(mask)
                T.sync_threads()
                if tx == 0:
                    running = T.alloc_var(T.uint32, init=0)
                    for index in range(32):
                        warp_prefix[index] = running
                        running = running + warp_counts[index]
                    count_base = (slot * num_planes + plane) * tiles_per_plane
                    tile_counts[count_base + tile] = running
                T.sync_threads()
                if active and code == 15:
                    lower = (1 << lane) - 1
                    rank = warp_prefix[warp] + T.popcount(mask & lower)
                    if rank < tile_escape_capacity:
                        prefix_base = (
                            low_base + plane_scalars + (plane_scalars + 1) // 2
                        )
                        escape_base = prefix_base + 4 * (tiles_per_plane + 1)
                        scratch[escape_base + tile * tile_escape_capacity + rank] = (
                            T.cast(bits >> 8, T.uint8)
                        )

    return main


def _build_store_prefix(*, num_planes: int, plane_scalars: int, max_tiles: int) -> Any:
    """Build per-plane cumulative escape prefixes and totals."""
    n_slots = T.dynamic("N")
    count_items = T.dynamic("C")
    row_items = T.dynamic("Q")
    scratch_bytes = T.dynamic("R")

    @T.prim_func
    def main(
        tile_counts: T.Tensor((count_items,), "uint32"),
        totals: T.Tensor((row_items,), "uint32"),
        overflow: T.Tensor((n_slots,), "uint32"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        tile_escape_capacity: T.int32,
        fixed_escape_bytes: T.int32,
        plane_record_bytes: T.int32,
    ):
        with T.Kernel(n_slots * num_planes, threads=1) as bx:
            for _tx in T.Parallel(1):
                row = bx
                slot = row // num_planes
                plane = row - slot * num_planes
                row_count_base = row * max_tiles
                prefix_base = (
                    slot * (4096 + num_planes * plane_record_bytes)
                    + 4096
                    + plane * plane_record_bytes
                    + plane_scalars
                    + (plane_scalars + 1) // 2
                )
                running = T.alloc_var(T.uint32, init=0)
                for byte in range(4):
                    scratch[prefix_base + byte] = 0
                for tile in range(max_tiles):
                    count = tile_counts[row_count_base + tile]
                    if count > tile_escape_capacity:
                        overflow[slot] = 1
                    running = running + count
                    scratch[prefix_base + (tile + 1) * 4] = T.cast(
                        running & 255, T.uint8
                    )
                    scratch[prefix_base + (tile + 1) * 4 + 1] = T.cast(
                        (running >> 8) & 255, T.uint8
                    )
                    scratch[prefix_base + (tile + 1) * 4 + 2] = T.cast(
                        (running >> 16) & 255, T.uint8
                    )
                    scratch[prefix_base + (tile + 1) * 4 + 3] = T.cast(
                        (running >> 24) & 255, T.uint8
                    )
                totals[row] = running
                if running > fixed_escape_bytes:
                    overflow[slot] = 1

    return main


def _build_store_layout(*, num_planes: int, plane_scalars: int, max_tiles: int) -> Any:
    """Build compact variable-length offsets from device totals."""
    n_slots = T.dynamic("N")
    row_items = T.dynamic("Q")
    payload_items = T.dynamic("B")

    @T.prim_func
    def main(
        totals: T.Tensor((row_items,), "uint32"),
        overflow: T.Tensor((n_slots,), "uint32"),
        source_offsets: T.Tensor((row_items,), "int64"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((payload_items,), "int64"),
        slot_offsets: T.Tensor((n_slots,), "int64"),
        slot_stride: T.int32,
        scratch_plane_record_bytes: T.int32,
        scratch_slot_stride: T.int32,
    ):
        payload_base = plane_scalars + (plane_scalars + 1) // 2 + 4 * (max_tiles + 1)
        with T.Kernel(1, threads=1):
            for _tx in T.Parallel(1):
                staging_cursor = T.alloc_var(T.int64, init=0)
                for slot in range(n_slots):
                    row_base = slot * num_planes
                    raw = T.alloc_var(
                        T.int32, init=T.cast(overflow[slot] != 0, T.int32)
                    )
                    record_cursor = T.alloc_var(T.int64, init=_HEADER_BYTES)
                    if raw == 0:
                        for plane in range(num_planes):
                            candidate_payload = payload_base + totals[row_base + plane]
                            candidate_record = (candidate_payload + 4095) // 4096 * 4096
                            if record_cursor + candidate_record > slot_stride:
                                raw = 1
                            record_cursor = record_cursor + candidate_record

                    slot_offsets[slot] = staging_cursor
                    if raw != 0:
                        overflow[slot] = 1
                        for plane in range(num_planes):
                            row = row_base + plane
                            source_offsets[row] = 0
                            destination_offsets[row] = 0
                            payload_bytes[row] = 0
                        staging_cursor = staging_cursor + slot_stride
                    else:
                        overflow[slot] = 0
                        record_cursor = _HEADER_BYTES
                        for plane in range(num_planes):
                            row = row_base + plane
                            layout_payload = payload_base + totals[row]
                            source_offsets[row] = (
                                slot * scratch_slot_stride
                                + _HEADER_BYTES
                                + plane * scratch_plane_record_bytes
                            )
                            destination_offsets[row] = staging_cursor + record_cursor
                            payload_bytes[row] = layout_payload
                            record_cursor = record_cursor + (
                                (layout_payload + 4095) // 4096 * 4096
                            )
                        staging_cursor = staging_cursor + record_cursor

    return main


def _build_store_compact(*, num_planes: int, plane_scalars: int, max_tiles: int) -> Any:
    """Compact tile-local escape streams into canonical record order."""
    row_items = T.dynamic("Q")
    source_bytes = T.dynamic("R")
    destination_bytes = T.dynamic("S")

    @T.prim_func
    def main(
        source: T.Tensor((source_bytes,), "uint8"),
        destination: T.Tensor((destination_bytes,), "uint8"),
        source_offsets: T.Tensor((row_items,), "int64"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((row_items,), "int64"),
        tile_escape_capacity: T.int32,
    ):
        base_bytes = plane_scalars + (plane_scalars + 1) // 2 + 4 * (max_tiles + 1)
        with T.Kernel(row_items, threads=_COMPACT_THREADS) as bx:
            for tx in T.Parallel(_COMPACT_THREADS):
                row = bx
                source_base = source_offsets[row]
                destination_base = destination_offsets[row]
                payload = payload_bytes[row]
                if payload >= base_bytes:
                    for index in range(tx, base_bytes, _COMPACT_THREADS):
                        destination[destination_base + index] = source[
                            source_base + index
                        ]
                    prefix_base = source_base + plane_scalars + (plane_scalars + 1) // 2
                    source_escape = source_base + base_bytes
                    destination_escape = destination_base + base_bytes
                    escape_bytes = payload - base_bytes
                    for tile in range(max_tiles):
                        begin = (
                            T.cast(source[prefix_base + tile * 4], T.uint32)
                            | (
                                T.cast(source[prefix_base + tile * 4 + 1], T.uint32)
                                << 8
                            )
                            | (
                                T.cast(source[prefix_base + tile * 4 + 2], T.uint32)
                                << 16
                            )
                            | (
                                T.cast(source[prefix_base + tile * 4 + 3], T.uint32)
                                << 24
                            )
                        )
                        end = (
                            T.cast(source[prefix_base + (tile + 1) * 4], T.uint32)
                            | (
                                T.cast(
                                    source[prefix_base + (tile + 1) * 4 + 1], T.uint32
                                )
                                << 8
                            )
                            | (
                                T.cast(
                                    source[prefix_base + (tile + 1) * 4 + 2], T.uint32
                                )
                                << 16
                            )
                            | (
                                T.cast(
                                    source[prefix_base + (tile + 1) * 4 + 3], T.uint32
                                )
                                << 24
                            )
                        )
                        if begin < escape_bytes:
                            count = T.min(end - begin, escape_bytes - begin)
                            tile_source = source_escape + tile * tile_escape_capacity
                            for index in range(tx, count, _COMPACT_THREADS):
                                destination[destination_escape + begin + index] = (
                                    source[tile_source + index]
                                )

    return main


def _build_raw_restore(*, num_blocks: int, num_planes: int, plane_scalars: int) -> Any:
    """Build the raw overflow copy stage."""
    n_slots = T.dynamic("N")
    destination_items = T.dynamic("S")

    @T.prim_func
    def main(
        kv_bits: T.Tensor((num_blocks, num_planes, plane_scalars), "uint16"),
        block_ids: T.Tensor((n_slots,), "int32"),
        overflow: T.Tensor((n_slots,), "uint32"),
        destination: T.Tensor((destination_items,), "uint8"),
        slot_offsets: T.Tensor((n_slots,), "int64"),
        slot_stride: T.int32,
    ):
        with T.Kernel(n_slots, threads=_COMPACT_THREADS) as bx:
            for tx in T.Parallel(_COMPACT_THREADS):
                slot = bx
                if overflow[slot] != 0:
                    block_id = block_ids[slot]
                    base = slot_offsets[slot]
                    for index in range(
                        tx, num_planes * plane_scalars, _COMPACT_THREADS
                    ):
                        value = kv_bits[
                            block_id, index // plane_scalars, index % plane_scalars
                        ]
                        destination[base + index * 2] = T.cast(value & 255, T.uint8)
                        destination[base + index * 2 + 1] = T.cast(value >> 8, T.uint8)

    return main


def _build_load(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    tiles_per_plane: int,
    staging_bytes: int,
) -> Any:
    """Build the single-launch raw/packed decoder."""
    n_slots = T.dynamic("N")
    staging_extent = T.dynamic("S")

    @T.prim_func
    def main(
        staging: T.Tensor((staging_extent,), "uint8"),
        staging_offsets: T.Tensor((n_slots,), "int64"),
        block_ids: T.Tensor((n_slots,), "int32"),
        modes: T.Tensor((n_slots,), "int32"),
        codebooks: T.Tensor((num_planes * _CODEBOOK_ENTRIES,), "uint8"),
        dst: T.Tensor((num_blocks, num_planes, plane_scalars), "uint16"),
    ):
        with T.Kernel(n_slots, num_planes, tiles_per_plane, threads=_STORE_THREADS) as (
            bx,
            by,
            bz,
        ):
            warp_counts = T.alloc_shared((32,), "uint32")
            warp_prefix = T.alloc_shared((32,), "uint32")
            for tx in T.Parallel(_STORE_THREADS):
                slot = bx
                plane = by
                tile = bz
                lane = tx & 31
                warp = tx // 32
                scalar = tile * tile_scalars + tx
                active = scalar < plane_scalars
                slot_base = staging_offsets[slot]
                block_id = block_ids[slot]
                mode = modes[slot]
                if mode == 0:
                    byte_base = slot_base + plane * plane_scalars * 2 + scalar * 2
                    raw_low = T.cast(staging[byte_base], T.uint16) if active else 0
                    raw_high = T.cast(staging[byte_base + 1], T.uint16) if active else 0
                    dst[block_id, plane, scalar] = (
                        raw_low | (raw_high << 8) if active else 0
                    )
                else:
                    # The fixed header page is 4 KiB, but its descriptor table
                    # starts immediately after the 120-byte struct.  Using
                    # the page boundary here silently read zero padding and
                    # made every compressed plane decode from offset zero.
                    descriptor = (
                        slot_base + _SLOT_FIXED_HEADER_BYTES + plane * _DESCRIPTOR_BYTES
                    )
                    low_offset = (
                        T.cast(staging[descriptor + 20], T.uint32)
                        | (T.cast(staging[descriptor + 21], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 22], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 23], T.uint32) << 24)
                    )
                    symbol_offset = (
                        T.cast(staging[descriptor + 24], T.uint32)
                        | (T.cast(staging[descriptor + 25], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 26], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 27], T.uint32) << 24)
                    )
                    prefix_offset = (
                        T.cast(staging[descriptor + 28], T.uint32)
                        | (T.cast(staging[descriptor + 29], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 30], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 31], T.uint32) << 24)
                    )
                    escape_offset = (
                        T.cast(staging[descriptor + 32], T.uint32)
                        | (T.cast(staging[descriptor + 33], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 34], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 35], T.uint32) << 24)
                    )
                    prefix_base = slot_base + prefix_offset
                    prefix_value = (
                        T.cast(staging[prefix_base + tile * 4], T.uint32)
                        | (T.cast(staging[prefix_base + tile * 4 + 1], T.uint32) << 8)
                        | (T.cast(staging[prefix_base + tile * 4 + 2], T.uint32) << 16)
                        | (T.cast(staging[prefix_base + tile * 4 + 3], T.uint32) << 24)
                    )
                    symbol = (
                        T.cast(
                            staging[slot_base + symbol_offset + scalar // 2], T.uint8
                        )
                        if active
                        else 0
                    )
                    code = (symbol >> ((scalar & 1) * 4)) & 15
                    mask = T.ballot(active and code == 15)
                    if lane == 0:
                        warp_counts[warp] = T.popcount(mask)
                    T.sync_threads()
                    if tx == 0:
                        running = T.alloc_var(T.uint32, init=0)
                        for index in range(32):
                            warp_prefix[index] = running
                            running = running + warp_counts[index]
                    T.sync_threads()
                    decoded_high = (
                        T.cast(
                            staging[
                                slot_base
                                + escape_offset
                                + prefix_value
                                + warp_prefix[warp]
                                + T.popcount(mask & ((1 << lane) - 1))
                            ],
                            T.uint16,
                        )
                        if code == 15 and active
                        else T.cast(
                            codebooks[plane * _CODEBOOK_ENTRIES + code], T.uint16
                        )
                    )
                    decoded_low = (
                        T.cast(staging[slot_base + low_offset + scalar], T.uint16)
                        if active
                        else 0
                    )
                    dst[block_id, plane, scalar] = (
                        decoded_low | (decoded_high << 8) if active else 0
                    )

    return main


def clear_tilelang_codec_cache() -> None:
    """Clear process-local TileLang codec kernels for tests.

    Returns:
        None.

    Async/thread-safety:
        Call only while no worker can launch a codec kernel.
    """
    _STORE_CACHE.clear()
    _LOAD_CACHE.clear()


__all__ = [
    "TileLangStoreKernels",
    "clear_tilelang_codec_cache",
    "compile_load_kernel",
    "compile_store_kernels",
]
