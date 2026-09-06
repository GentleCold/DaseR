# SPDX-License-Identifier: Apache-2.0
"""TileLang fused online compressed-KV store and load operators.

The operators in this module deliberately use the existing variable-length
record format. Store is exposed as a small kernel bundle because a global
prefix is required before variable-length records can be emitted; all stages
are launched on the caller's stream and no host round trip occurs between
them. The final store stage reads the live KV cache directly after layout
planning and emits the slot header in the same launch, avoiding a full-sized
intermediate payload copy and a per-slot host-to-device header copy. Load is a
single kernel because its record offsets and lengths are already known when
the server returns a read plan.
"""

# Do not enable ``from __future__ import annotations``. TileLang inspects the
# concrete annotations while constructing the prim_func.

from dataclasses import dataclass
from typing import Any

import tilelang
import tilelang.language as T

_COUNT_THREADS = 256
_COUNT_WARPS = _COUNT_THREADS // 32
_STORE_THREADS = 1024
# Compact assigns one warp to each 1024-scalar tile.
_COMPACT_THREADS = 256
_COMPACT_WARPS = _COMPACT_THREADS // 32
# Decode uses fewer threads because each escape ballot is warp-local.  Four
# warps cover one tile through multiple scalar groups while reducing block-wide
# barrier and register overhead on the load critical path.
_LOAD_THREADS = 128
_LOAD_WARPS = _LOAD_THREADS // 32
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
        Cached fused encode, layout, and compact kernels.

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
                num_blocks=num_blocks,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                tile_scalars=tile_scalars,
                max_tiles=tiles_per_plane,
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
    """Build the single-read encode pass.

    The first pass reads each selected KV scalar once and only counts escapes.
    Payload emission is deferred until layout has produced variable-length
    destinations. The compact pass then reads the live KV cache directly and
    writes the final payload, avoiding a full low/symbol/escape scratch round
    trip.

    One CUDA block owns a slot/plane row and its warps independently scan
    tiles.  This keeps the output tile-count layout unchanged while avoiding
    hundreds of thousands of very short blocks for production KV geometry.
    """
    n_slots = T.dynamic("N")
    count_items = T.dynamic("C")
    scratch_bytes = T.dynamic("R")

    @T.prim_func
    def main(
        kv_bits: T.Tensor((num_blocks, num_planes, plane_scalars), "uint16"),
        block_ids: T.Tensor((n_slots,), "int32"),
        lookup: T.Tensor((num_planes * 256,), "uint8"),
        tile_counts: T.Tensor((count_items,), "uint32"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        block_tokens: T.int32,
        valid_token_count: T.int32,
        tile_escape_capacity: T.int32,
        scratch_plane_record_bytes: T.int32,
    ):
        with T.Kernel(n_slots, num_planes, threads=_COUNT_THREADS) as (bx, by):
            for tx in T.Parallel(_COUNT_THREADS):
                slot = bx
                plane = by
                lane = tx & 31
                warp = tx // 32
                block_id = block_ids[slot]
                row_scalars = plane_scalars // block_tokens
                for tile_group in range(
                    (tiles_per_plane + _COUNT_WARPS - 1) // _COUNT_WARPS
                ):
                    tile = tile_group * _COUNT_WARPS + warp
                    lane_count = T.alloc_var(T.uint32, init=0)
                    escape_before = T.alloc_var(T.uint32, init=0)
                    for scalar_group in range((tile_scalars + 31) // 32):
                        scalar = tile * tile_scalars + scalar_group * 32 + lane
                        active = tile < tiles_per_plane and scalar < plane_scalars
                        valid = active and (
                            slot * block_tokens + scalar // row_scalars
                            < valid_token_count
                        )
                        bits = (
                            T.cast(kv_bits[block_id, plane, scalar], T.uint16)
                            if valid
                            else 0
                        )
                        high = (bits >> 8) & 255
                        code = (
                            T.cast(lookup[plane * 256 + high], T.uint8) if valid else 15
                        )
                        mask = T.ballot(active and code == 15)
                        escape_before = escape_before + T.popcount(mask)
                        # Invalid tail scalars count as zero-bit escapes,
                        # matching the format's canonical partial-slot fill.
                        lane_count = lane_count + T.cast(
                            active and code == 15, T.uint32
                        )
                    for offset_index in range(5):
                        lane_count = lane_count + T.shfl_down(
                            lane_count, 16 >> offset_index
                        )
                    if lane == 0 and tile < tiles_per_plane:
                        count_base = (slot * num_planes + plane) * tiles_per_plane
                        tile_counts[count_base + tile] = lane_count

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


def _build_store_compact(
    *,
    num_blocks: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    max_tiles: int,
) -> Any:
    """Emit packed records, headers, and raw-overflow slots in one launch.

    The layout planner marks overflow slots with zero payload descriptors and
    assigns them a raw slot base. One block owns a slot/plane row and each warp
    advances through independent tiles, reducing the launch grid from one
    block per tile to one block per plane while preserving the persisted
    record format.  The plane-zero block also serializes the canonical 4 KiB
    header after layout planning.  Header bytes are written by the same CUDA
    launch as payload bytes, so the caller does not need a pageable or pinned
    host metadata copy on the hot path.
    """
    row_items = T.dynamic("Q")
    slot_items = T.dynamic("N")
    destination_bytes = T.dynamic("S")
    scratch_bytes = T.dynamic("R")

    @T.prim_func
    def main(
        kv_bits: T.Tensor((num_blocks, num_planes, plane_scalars), "uint16"),
        block_ids: T.Tensor((slot_items,), "int32"),
        logical_slots: T.Tensor((slot_items,), "int64"),
        overflow: T.Tensor((slot_items,), "uint32"),
        destination: T.Tensor((destination_bytes,), "uint8"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((row_items,), "int64"),
        totals: T.Tensor((row_items,), "uint32"),
        slot_offsets: T.Tensor((slot_items,), "int64"),
        lookup: T.Tensor((num_planes * 256,), "uint8"),
        codebook_hash: T.Tensor((32,), "uint8"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        tile_escape_capacity: T.int32,
        scratch_plane_record_bytes: T.int32,
        block_tokens: T.int32,
        valid_token_count: T.int32,
        slot_stride: T.int32,
    ):
        base_bytes = plane_scalars + (plane_scalars + 1) // 2 + 4 * (max_tiles + 1)
        symbol_bytes = (plane_scalars + 1) // 2
        prefix_bytes = 4 * (max_tiles + 1)
        with T.Kernel(slot_items, num_planes, threads=_COMPACT_THREADS) as (bx, by):
            for tx in T.Parallel(_COMPACT_THREADS):
                slot = bx
                plane = by
                lane = T.cast(tx & 31, T.int32)
                warp = T.cast(tx // 32, T.int32)
                row = slot * num_planes + plane
                block_id = block_ids[slot]
                if overflow[slot] != 0:
                    if plane == 0:
                        raw_base = slot_offsets[slot]
                        for chunk in range(
                            (num_planes * plane_scalars + _COMPACT_THREADS - 1)
                            // _COMPACT_THREADS
                        ):
                            index = chunk * _COMPACT_THREADS + tx
                            if index < num_planes * plane_scalars:
                                value = kv_bits[
                                    block_id,
                                    index // plane_scalars,
                                    index % plane_scalars,
                                ]
                                destination[raw_base + index * 2] = T.cast(
                                    value & 255, T.uint8
                                )
                                destination[raw_base + index * 2 + 1] = T.cast(
                                    value >> 8, T.uint8
                                )
                else:
                    # The compact launch owns the header as well as payload
                    # bytes.  Header fields use the little-endian wire
                    # layout from ``compression.format``; clearing the page
                    # first preserves the canonical zero-filled tail emitted
                    # by ``SlotHeader.pack``.
                    if plane == 0:
                        header_base = slot_offsets[slot]
                        for header_index in range(_HEADER_BYTES // _COMPACT_THREADS):
                            destination[
                                header_base + header_index * _COMPACT_THREADS + tx
                            ] = 0
                        T.sync_threads()
                        if tx == 0:
                            destination[header_base + 0] = 68
                            destination[header_base + 1] = 75
                            destination[header_base + 2] = 86
                            destination[header_base + 3] = 83
                            destination[header_base + 4] = 76
                            destination[header_base + 5] = 79
                            destination[header_base + 6] = 84
                            destination[header_base + 7] = 49
                            for byte in range(4):
                                destination[header_base + 8 + byte] = T.cast(
                                    (1 >> (byte * 8)) & 255, T.uint8
                                )
                                destination[header_base + 12 + byte] = T.cast(
                                    (_HEADER_BYTES >> (byte * 8)) & 255, T.uint8
                                )
                                destination[header_base + 16 + byte] = T.cast(
                                    (_HEADER_BYTES >> (byte * 8)) & 255, T.uint8
                                )
                                destination[header_base + 20 + byte] = T.cast(
                                    (tile_scalars >> (byte * 8)) & 255, T.uint8
                                )
                            for byte in range(8):
                                destination[header_base + 24 + byte] = T.cast(
                                    (logical_slots[slot] >> (byte * 8)) & 255,
                                    T.uint8,
                                )
                                destination[header_base + 40 + byte] = T.cast(
                                    (slot_stride >> (byte * 8)) & 255, T.uint8
                                )
                            for byte in range(4):
                                destination[header_base + 32 + byte] = T.cast(
                                    (num_planes // 2) >> (byte * 8),
                                    T.uint8,
                                )
                                destination[header_base + 36 + byte] = T.cast(
                                    (num_planes >> (byte * 8)) & 255, T.uint8
                                )
                            for byte in range(32):
                                destination[header_base + 56 + byte] = codebook_hash[
                                    byte
                                ]
                            destination[header_base + 88] = 1
                            stored_length = T.alloc_var(T.int64, init=0)
                            for descriptor_plane in range(num_planes):
                                descriptor_row = slot * num_planes + descriptor_plane
                                descriptor_base = (
                                    header_base
                                    + _SLOT_FIXED_HEADER_BYTES
                                    + descriptor_plane * _DESCRIPTOR_BYTES
                                )
                                relative_offset = (
                                    destination_offsets[descriptor_row] - header_base
                                )
                                payload_length = payload_bytes[descriptor_row]
                                record_length = (payload_length + 4095) // 4096 * 4096
                                stored_length = relative_offset + record_length
                                for byte in range(2):
                                    destination[descriptor_base + byte] = T.cast(
                                        (descriptor_plane // 2 >> (byte * 8)) & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 2 + byte] = T.cast(
                                        (descriptor_plane % 2 >> (byte * 8)) & 255,
                                        T.uint8,
                                    )
                                for byte in range(4):
                                    destination[descriptor_base + 4 + byte] = T.cast(
                                        (plane_scalars >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 8 + byte] = T.cast(
                                        (max_tiles >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 12 + byte] = T.cast(
                                        (relative_offset >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 16 + byte] = T.cast(
                                        (record_length >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 20 + byte] = T.cast(
                                        (relative_offset >> (byte * 8)) & 255, T.uint8
                                    )
                                    destination[descriptor_base + 24 + byte] = T.cast(
                                        (relative_offset + plane_scalars) >> (byte * 8)
                                        & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 28 + byte] = T.cast(
                                        (relative_offset + plane_scalars + symbol_bytes)
                                        >> (byte * 8)
                                        & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 32 + byte] = T.cast(
                                        (relative_offset + base_bytes) >> (byte * 8)
                                        & 255,
                                        T.uint8,
                                    )
                                    destination[descriptor_base + 36 + byte] = T.cast(
                                        totals[descriptor_row] >> (byte * 8) & 255,
                                        T.uint8,
                                    )
                            for byte in range(8):
                                destination[header_base + 48 + byte] = T.cast(
                                    (stored_length >> (byte * 8)) & 255, T.uint8
                                )
                    destination_base = destination_offsets[row]
                    payload = payload_bytes[row]
                    if payload >= base_bytes:
                        scratch_plane_base = (
                            slot * (4096 + num_planes * scratch_plane_record_bytes)
                            + 4096
                            + plane * scratch_plane_record_bytes
                        )
                        scratch_prefix_base = (
                            scratch_plane_base + plane_scalars + symbol_bytes
                        )
                        if tx < prefix_bytes:
                            destination[
                                destination_base + plane_scalars + symbol_bytes + tx
                            ] = scratch[scratch_prefix_base + tx]
                        for tile_group in range(
                            (max_tiles + _COMPACT_WARPS - 1) // _COMPACT_WARPS
                        ):
                            tile = tile_group * _COMPACT_WARPS + warp
                            tile_begin = tile * tile_scalars
                            # Each lane scans the same tile-local scalar groups
                            # in lockstep. Keeping the preceding-group escape
                            # count in a lane-local accumulator avoids a
                            # block-wide prefix or a shuffle dependency in the
                            # emit pass.
                            escape_before = T.alloc_var(T.uint32, init=0)
                            for scalar_group in range((tile_scalars + 31) // 32):
                                scalar = tile_begin + scalar_group * 32 + lane
                                active = tile < max_tiles and scalar < plane_scalars
                                valid = active and (
                                    slot * block_tokens
                                    + scalar // (plane_scalars // block_tokens)
                                    < valid_token_count
                                )
                                bits = (
                                    T.cast(kv_bits[block_id, plane, scalar], T.uint16)
                                    if valid
                                    else 0
                                )
                                high = (bits >> 8) & 255
                                code = (
                                    T.cast(lookup[plane * 256 + high], T.uint8)
                                    if valid
                                    else 15
                                )
                                next_code = T.shfl_down(T.cast(code, T.uint32), 1)
                                symbol = T.cast(
                                    code | (T.cast(next_code, T.uint8) << 4), T.uint8
                                )
                                if active:
                                    destination[destination_base + scalar] = T.cast(
                                        bits & 255, T.uint8
                                    )
                                if active and (lane & 1) == 0:
                                    destination[
                                        destination_base + plane_scalars + scalar // 2
                                    ] = symbol

                                mask = T.ballot(active and code == 15)
                                prefix_value = (
                                    (
                                        T.cast(
                                            scratch[scratch_prefix_base + tile * 4],
                                            T.uint32,
                                        )
                                        | (
                                            T.cast(
                                                scratch[
                                                    scratch_prefix_base + tile * 4 + 1
                                                ],
                                                T.uint32,
                                            )
                                            << 8
                                        )
                                        | (
                                            T.cast(
                                                scratch[
                                                    scratch_prefix_base + tile * 4 + 2
                                                ],
                                                T.uint32,
                                            )
                                            << 16
                                        )
                                        | (
                                            T.cast(
                                                scratch[
                                                    scratch_prefix_base + tile * 4 + 3
                                                ],
                                                T.uint32,
                                            )
                                            << 24
                                        )
                                    )
                                    if tile < max_tiles
                                    else 0
                                )
                                if active and code == 15:
                                    lower_mask = mask & (
                                        T.cast(0xFFFFFFFF, T.uint32) >> (lane ^ 31)
                                    )
                                    current_bit = T.cast((mask >> lane) & 1, T.uint32)
                                    rank = (
                                        prefix_value
                                        + escape_before
                                        + T.popcount(lower_mask)
                                        - current_bit
                                    )
                                    destination[
                                        destination_base + base_bytes + rank
                                    ] = T.cast(bits >> 8, T.uint8)
                                group_count = T.popcount(mask)
                                escape_before = escape_before + group_count

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
        with T.Kernel(n_slots, num_planes, tiles_per_plane, threads=_LOAD_THREADS) as (
            bx,
            by,
            bz,
        ):
            warp_counts = T.alloc_shared((_LOAD_WARPS,), "uint32")
            warp_prefix = T.alloc_shared((_LOAD_WARPS,), "uint32")
            descriptor_offsets = T.alloc_shared((4,), "uint32")
            for tx in T.Parallel(_LOAD_THREADS):
                slot = bx
                plane = by
                tile = bz
                lane = tx & 31
                warp = tx // 32
                slot_base = staging_offsets[slot]
                block_id = block_ids[slot]
                mode = modes[slot]
                if mode != 0 and tx < 4:
                    descriptor = (
                        slot_base + _SLOT_FIXED_HEADER_BYTES + plane * _DESCRIPTOR_BYTES
                    )
                    descriptor_offsets[0] = (
                        T.cast(staging[descriptor + 20], T.uint32)
                        | (T.cast(staging[descriptor + 21], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 22], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 23], T.uint32) << 24)
                    )
                    descriptor_offsets[1] = (
                        T.cast(staging[descriptor + 24], T.uint32)
                        | (T.cast(staging[descriptor + 25], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 26], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 27], T.uint32) << 24)
                    )
                    descriptor_offsets[2] = (
                        T.cast(staging[descriptor + 28], T.uint32)
                        | (T.cast(staging[descriptor + 29], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 30], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 31], T.uint32) << 24)
                    )
                    descriptor_offsets[3] = (
                        T.cast(staging[descriptor + 32], T.uint32)
                        | (T.cast(staging[descriptor + 33], T.uint32) << 8)
                        | (T.cast(staging[descriptor + 34], T.uint32) << 16)
                        | (T.cast(staging[descriptor + 35], T.uint32) << 24)
                    )
                T.sync_threads()
                low_offset = descriptor_offsets[0] if mode != 0 else 0
                symbol_offset = descriptor_offsets[1] if mode != 0 else 0
                prefix_offset = descriptor_offsets[2] if mode != 0 else 0
                escape_offset = descriptor_offsets[3] if mode != 0 else 0
                prefix_value = T.alloc_var(T.uint32, init=0)
                if mode != 0:
                    prefix_base = slot_base + prefix_offset
                    prefix_value = (
                        T.cast(staging[prefix_base + tile * 4], T.uint32)
                        | (T.cast(staging[prefix_base + tile * 4 + 1], T.uint32) << 8)
                        | (T.cast(staging[prefix_base + tile * 4 + 2], T.uint32) << 16)
                        | (T.cast(staging[prefix_base + tile * 4 + 3], T.uint32) << 24)
                    )
                for scalar_group in range(
                    (tile_scalars + _LOAD_THREADS - 1) // _LOAD_THREADS
                ):
                    scalar = tile * tile_scalars + scalar_group * _LOAD_THREADS + tx
                    active = scalar < plane_scalars
                    if mode == 0:
                        byte_base = slot_base + plane * plane_scalars * 2 + scalar * 2
                        raw_low = T.cast(staging[byte_base], T.uint16) if active else 0
                        raw_high = (
                            T.cast(staging[byte_base + 1], T.uint16) if active else 0
                        )
                        dst[block_id, plane, scalar] = (
                            raw_low | (raw_high << 8) if active else 0
                        )
                        continue
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
                        for index in range(_LOAD_WARPS):
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
