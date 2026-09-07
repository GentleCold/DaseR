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
_ESCAPE_PACKED_FLAG = 0x80
_ESCAPE_3BIT_FLAG = 0x40


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
        raw_counts: T.Tensor((count_items,), "uint32"),
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
                    raw_lane_count = T.alloc_var(T.uint32, init=0)
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
                        mask = T.ballot(active and code >= 7)
                        escape_before = escape_before + T.popcount(mask)
                        # Invalid tail scalars count as zero-bit escapes,
                        # matching the format's canonical partial-slot fill.
                        lane_count = lane_count + T.cast(active and code >= 7, T.uint32)
                        # Online mode uses the narrow three-bit escape stream:
                        # entries 7..13 remain secondary codes and entries
                        # 14..15 are emitted as raw high bytes.  Counting the
                        # full raw set here lets the layout stage size that
                        # stream without a host round trip.
                        raw_lane_count = raw_lane_count + T.cast(
                            active and code >= 14, T.uint32
                        )
                    for offset_index in range(5):
                        lane_count = lane_count + T.shfl_down(
                            lane_count, 16 >> offset_index
                        )
                        raw_lane_count = raw_lane_count + T.shfl_down(
                            raw_lane_count, 16 >> offset_index
                        )
                    if lane == 0 and tile < tiles_per_plane:
                        count_base = (slot * num_planes + plane) * tiles_per_plane
                        tile_counts[count_base + tile] = lane_count
                        raw_counts[count_base + tile] = raw_lane_count

    return main


def _build_store_prefix(*, num_planes: int, plane_scalars: int, max_tiles: int) -> Any:
    """Build per-plane cumulative escape prefixes and slot totals.

    The fixed scratch argument ``tile_escape_capacity`` remains in the kernel
    ABI for warm-cache compatibility, but only the complete plane total is an
    overflow decision.  Compact writes escapes directly into the variable
    destination, so tile-local distribution does not constrain the envelope.
    """
    n_slots = T.dynamic("N")
    count_items = T.dynamic("C")
    row_items = T.dynamic("Q")
    scratch_bytes = T.dynamic("R")

    @T.prim_func
    def main(
        tile_counts: T.Tensor((count_items,), "uint32"),
        raw_counts: T.Tensor((count_items,), "uint32"),
        totals: T.Tensor((row_items,), "uint32"),
        raw_totals: T.Tensor((row_items,), "uint32"),
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
                raw_prefix_base = prefix_base + 4 * (max_tiles + 1)
                running = T.alloc_var(T.uint32, init=0)
                raw_running = T.alloc_var(T.uint32, init=0)
                for byte in range(4):
                    scratch[prefix_base + byte] = 0
                    scratch[raw_prefix_base + byte] = 0
                for tile in range(max_tiles):
                    count = tile_counts[row_count_base + tile]
                    raw_count = raw_counts[row_count_base + tile]
                    # ``tile_escape_capacity`` belongs to the fixed scratch
                    # geometry, but the compact writer emits escapes into the
                    # variable-length destination using the cumulative prefix
                    # below.  A single tile may therefore contain more than
                    # the average escape budget while the complete plane still
                    # fits its fixed envelope.  Treating that shape as raw
                    # fallback discarded otherwise valid compression and was
                    # the source of whole-batch overflow spikes.  Only the
                    # slot-wide envelope check is authoritative.
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
                    raw_running = raw_running + raw_count
                    scratch[raw_prefix_base + (tile + 1) * 4] = T.cast(
                        raw_running & 255, T.uint8
                    )
                    scratch[raw_prefix_base + (tile + 1) * 4 + 1] = T.cast(
                        (raw_running >> 8) & 255, T.uint8
                    )
                    scratch[raw_prefix_base + (tile + 1) * 4 + 2] = T.cast(
                        (raw_running >> 16) & 255, T.uint8
                    )
                    scratch[raw_prefix_base + (tile + 1) * 4 + 3] = T.cast(
                        (raw_running >> 24) & 255, T.uint8
                    )
                totals[row] = running
                raw_totals[row] = raw_running
                escape_bytes_3 = (running * 3 + 7) // 8 + raw_running
                escape_bytes_4 = (running + 1) // 2 + raw_running
                if (
                    escape_bytes_3 > fixed_escape_bytes
                    and escape_bytes_4 > fixed_escape_bytes
                ):
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
        raw_totals: T.Tensor((row_items,), "uint32"),
        overflow: T.Tensor((n_slots,), "uint32"),
        symbol_bits: T.Tensor((n_slots,), "int32"),
        source_offsets: T.Tensor((row_items,), "int64"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((payload_items,), "int64"),
        slot_offsets: T.Tensor((n_slots,), "int64"),
        slot_stride: T.int32,
        scratch_plane_record_bytes: T.int32,
        scratch_slot_stride: T.int32,
    ):
        prefix_bytes_3 = 8 * (max_tiles + 1)
        payload_base_3 = plane_scalars + (plane_scalars * 3 + 7) // 8 + prefix_bytes_3
        with T.Kernel(1, threads=1):
            for _tx in T.Parallel(1):
                staging_cursor = T.alloc_var(T.int64, init=0)
                for slot in range(n_slots):
                    row_base = slot * num_planes
                    raw = T.alloc_var(
                        T.int32, init=T.cast(overflow[slot] != 0, T.int32)
                    )
                    record_cursor = T.alloc_var(T.int64, init=_HEADER_BYTES)
                    record_cursor_3 = T.alloc_var(T.int64, init=_HEADER_BYTES)
                    if raw == 0:
                        for plane in range(num_planes):
                            candidate_payload_3 = (
                                payload_base_3
                                + (totals[row_base + plane] * 3 + 7) // 8
                                + raw_totals[row_base + plane]
                            )
                            record_cursor_3 = record_cursor_3 + candidate_payload_3
                            if record_cursor_3 > slot_stride:
                                raw = 1
                        if raw == 0:
                            # Internal value 5 means a three-bit main stream
                            # with a three-bit escape stream.  The persisted
                            # header keeps the main width and carries the
                            # escape width in a dedicated flag.
                            symbol_bits[slot] = 5
                            record_cursor = record_cursor_3

                    slot_offsets[slot] = staging_cursor
                    if raw != 0:
                        overflow[slot] = 1
                        symbol_bits[slot] = 0
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
                            symbol_bytes = (plane_scalars * 3 + 7) // 8
                            layout_payload = (
                                plane_scalars
                                + symbol_bytes
                                + prefix_bytes_3
                                + (totals[row] * 3 + 7) // 8
                                + raw_totals[row]
                            )
                            source_offsets[row] = (
                                slot * scratch_slot_stride
                                + _HEADER_BYTES
                                + plane * scratch_plane_record_bytes
                            )
                            destination_offsets[row] = staging_cursor + record_cursor
                            payload_bytes[row] = layout_payload
                            record_cursor = record_cursor + layout_payload
                        staging_cursor = (
                            staging_cursor + (record_cursor + 4095) // 4096 * 4096
                        )

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
    escape_word_items = T.dynamic("W")

    @T.prim_func
    def main(
        kv_bits: T.Tensor((num_blocks, num_planes, plane_scalars), "uint16"),
        block_ids: T.Tensor((slot_items,), "int32"),
        logical_slots: T.Tensor((slot_items,), "int64"),
        overflow: T.Tensor((slot_items,), "uint32"),
        destination: T.Tensor((destination_bytes,), "uint8"),
        escape_words: T.Tensor((escape_word_items,), "uint32"),
        destination_offsets: T.Tensor((row_items,), "int64"),
        payload_bytes: T.Tensor((row_items,), "int64"),
        totals: T.Tensor((row_items,), "uint32"),
        raw_totals: T.Tensor((row_items,), "uint32"),
        slot_offsets: T.Tensor((slot_items,), "int64"),
        symbol_bits: T.Tensor((slot_items,), "int32"),
        lookup: T.Tensor((num_planes * 256,), "uint8"),
        codebook_hash: T.Tensor((32,), "uint8"),
        scratch: T.Tensor((scratch_bytes,), "uint8"),
        tile_escape_capacity: T.int32,
        scratch_plane_record_bytes: T.int32,
        block_tokens: T.int32,
        valid_token_count: T.int32,
        slot_stride: T.int32,
    ):
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
                    # The online layout planner emits only the new packed
                    # format here (``symbol_bits == 5``: 3-bit symbols plus
                    # 3-bit escape tokens).  Keep these byte counts static in
                    # the TileLang builder.  Making them depend on a device
                    # tensor caused the generated kernel to select the legacy
                    # 4-bit symbol width for header offsets even though the
                    # layout planner had reserved the 3-bit payload.
                    packed_escape = True
                    escape_symbol_bits = 3
                    prefix_bytes = 8 * (max_tiles + 1)
                    symbol_bytes = (plane_scalars * 3 + 7) // 8
                    base_bytes = plane_scalars + symbol_bytes + prefix_bytes
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
                            destination[
                                header_base
                                + _SLOT_FIXED_HEADER_BYTES
                                + num_planes * _DESCRIPTOR_BYTES
                            ] = T.cast(
                                3 | _ESCAPE_PACKED_FLAG | _ESCAPE_3BIT_FLAG,
                                T.uint8,
                            )
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
                                record_length = payload_length
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
                                    escape_count = (
                                        totals[descriptor_row]
                                        if packed_escape
                                        else raw_totals[descriptor_row]
                                    )
                                    destination[descriptor_base + 36 + byte] = T.cast(
                                        escape_count >> (byte * 8) & 255, T.uint8
                                    )
                            for byte in range(8):
                                destination[header_base + 48 + byte] = T.cast(
                                    (
                                        (stored_length + 4095) // 4096 * 4096
                                        >> (byte * 8)
                                    )
                                    & 255,
                                    T.uint8,
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
                            scratch_plane_base
                            + plane_scalars
                            + (plane_scalars + 1) // 2
                        )
                        scratch_raw_prefix_base = scratch_prefix_base + 4 * (
                            max_tiles + 1
                        )
                        max_escape_code_bytes = max(
                            (plane_scalars + 1) // 2,
                            (plane_scalars * 3 + 7) // 8,
                        )
                        max_escape_words = (max_escape_code_bytes + 3) // 4
                        escape_word_base = row * max_escape_words
                        escape_code_bytes = (totals[row] * escape_symbol_bits + 7) // 8
                        if packed_escape:
                            # TileLang requires statically typed loop domains.
                            # Iterate over the maximum packed escape extent and
                            # predicate the runtime tail instead of using the
                            # uint32 total as a range bound.
                            for escape_word_group in range(
                                (max_escape_words + _COMPACT_THREADS - 1)
                                // _COMPACT_THREADS
                            ):
                                escape_word = escape_word_group * _COMPACT_THREADS + tx
                                if escape_word < max_escape_words:
                                    escape_words[escape_word_base + escape_word] = 0
                            T.sync_threads()
                            for escape_group in range(
                                (max_escape_code_bytes + _COMPACT_THREADS - 1)
                                // _COMPACT_THREADS
                            ):
                                escape_index = escape_group * _COMPACT_THREADS + tx
                                if escape_index < escape_code_bytes:
                                    destination[
                                        destination_base + base_bytes + escape_index
                                    ] = 0
                            T.sync_threads()
                        for prefix_group in range(
                            (prefix_bytes + _COMPACT_THREADS - 1) // _COMPACT_THREADS
                        ):
                            prefix_index = prefix_group * _COMPACT_THREADS + tx
                            if prefix_index < prefix_bytes:
                                destination[
                                    destination_base
                                    + plane_scalars
                                    + symbol_bytes
                                    + prefix_index
                                ] = scratch[scratch_prefix_base + prefix_index]
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
                            raw_escape_before = T.alloc_var(T.uint32, init=0)
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
                                if active:
                                    destination[destination_base + scalar] = T.cast(
                                        bits & 255, T.uint8
                                    )
                                mask = T.ballot(active and code >= 7)
                                raw_mask = T.ballot(active and code >= 14)
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
                                raw_prefix_value = (
                                    (
                                        T.cast(
                                            scratch[scratch_raw_prefix_base + tile * 4],
                                            T.uint32,
                                        )
                                        | (
                                            T.cast(
                                                scratch[
                                                    scratch_raw_prefix_base
                                                    + tile * 4
                                                    + 1
                                                ],
                                                T.uint32,
                                            )
                                            << 8
                                        )
                                        | (
                                            T.cast(
                                                scratch[
                                                    scratch_raw_prefix_base
                                                    + tile * 4
                                                    + 2
                                                ],
                                                T.uint32,
                                            )
                                            << 16
                                        )
                                        | (
                                            T.cast(
                                                scratch[
                                                    scratch_raw_prefix_base
                                                    + tile * 4
                                                    + 3
                                                ],
                                                T.uint32,
                                            )
                                            << 24
                                        )
                                    )
                                    if tile < max_tiles
                                    else 0
                                )
                                if active and code >= 7:
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
                                    if packed_escape:
                                        raw_sentinel = 7
                                        escape_code = (
                                            code - 7 if code < 14 else raw_sentinel
                                        )
                                        escape_bit_offset = rank * escape_symbol_bits
                                        escape_byte = escape_bit_offset // 8
                                        escape_word = escape_bit_offset // 32
                                        escape_word_shift = escape_bit_offset & 31
                                        # Escape words are cleared before the
                                        # tile scan. Atomic additions are safe
                                        # because adjacent 3-bit tokens occupy
                                        # disjoint bit ranges; the final copy
                                        # emits only the required byte extent.
                                        T.atomic_add(
                                            escape_words[
                                                escape_word_base + escape_word
                                            ],
                                            T.cast(escape_code, T.uint32)
                                            << escape_word_shift,
                                        )
                                        if escape_word_shift > 29:
                                            T.atomic_add(
                                                escape_words[
                                                    escape_word_base + escape_word + 1
                                                ],
                                                T.cast(escape_code, T.uint32)
                                                >> (32 - escape_word_shift),
                                            )
                                        if code >= 14:
                                            raw_lower_mask = raw_mask & (
                                                T.cast(0xFFFFFFFF, T.uint32)
                                                >> (lane ^ 31)
                                            )
                                            raw_current_bit = T.cast(
                                                (raw_mask >> lane) & 1, T.uint32
                                            )
                                            raw_rank = (
                                                raw_prefix_value
                                                + raw_escape_before
                                                + T.popcount(raw_lower_mask)
                                                - raw_current_bit
                                            )
                                            destination[
                                                destination_base
                                                + base_bytes
                                                + escape_code_bytes
                                                + raw_rank
                                            ] = T.cast(bits >> 8, T.uint8)
                                    else:
                                        destination[
                                            destination_base + base_bytes + rank
                                        ] = T.cast(bits >> 8, T.uint8)
                                group_count = T.popcount(mask)
                                escape_before = escape_before + group_count
                                raw_escape_before = raw_escape_before + T.popcount(
                                    raw_mask
                                )

                        if packed_escape:
                            # Expand the complete words only after every warp
                            # has emitted its ranks.
                            T.sync_threads()
                            for escape_byte_group in range(
                                (max_escape_code_bytes + _COMPACT_THREADS - 1)
                                // _COMPACT_THREADS
                            ):
                                escape_byte = escape_byte_group * _COMPACT_THREADS + tx
                                if escape_byte < escape_code_bytes:
                                    escape_word = escape_byte // 4
                                    byte_shift = (escape_byte & 3) * 8
                                    destination[
                                        destination_base + base_bytes + escape_byte
                                    ] = T.cast(
                                        escape_words[escape_word_base + escape_word]
                                        >> byte_shift,
                                        T.uint8,
                                    )

                    # Every non-overflow online record uses the packed
                    # three-bit symbol stream selected by the layout pass.
                    if packed_escape:
                        symbol_bytes_3 = (plane_scalars * 3 + 7) // 8
                        for byte_group in range(
                            (symbol_bytes_3 + _COMPACT_THREADS - 1) // _COMPACT_THREADS
                        ):
                            symbol_index = byte_group * _COMPACT_THREADS + tx
                            if symbol_index < symbol_bytes_3:
                                first_scalar = symbol_index * 8 // 3
                                byte_begin = symbol_index * 8
                                packed = T.alloc_var(T.uint32, init=0)
                                for code_index in range(4):
                                    scalar = first_scalar + code_index
                                    if scalar < plane_scalars:
                                        valid = (
                                            slot * block_tokens
                                            + scalar // (plane_scalars // block_tokens)
                                            < valid_token_count
                                        )
                                        bits = (
                                            kv_bits[block_id, plane, scalar]
                                            if valid
                                            else 0
                                        )
                                        high = (bits >> 8) & 255
                                        code = T.alloc_var(T.uint32, init=7)
                                        if valid:
                                            code = T.cast(
                                                lookup[plane * 256 + high], T.uint32
                                            )
                                        code = 7 if code >= 7 else code
                                        for code_bit in range(3):
                                            global_bit = scalar * 3 + code_bit
                                            if (
                                                global_bit >= byte_begin
                                                and global_bit < byte_begin + 8
                                            ):
                                                packed = packed | (
                                                    ((code >> code_bit) & 1)
                                                    << (global_bit - byte_begin)
                                                )
                                destination[
                                    destination_base + plane_scalars + symbol_index
                                ] = T.cast(packed & 255, T.uint8)

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
            warp_raw_counts = T.alloc_shared((_LOAD_WARPS,), "uint32")
            warp_raw_prefix = T.alloc_shared((_LOAD_WARPS,), "uint32")
            descriptor_offsets = T.alloc_shared((4,), "uint32")
            symbol_bits_shared = T.alloc_shared((1,), "uint32")
            escape_packed_shared = T.alloc_shared((1,), "uint32")
            escape_symbol_bits_shared = T.alloc_shared((1,), "uint32")
            escape_count_shared = T.alloc_shared((1,), "uint32")
            for tx in T.Parallel(_LOAD_THREADS):
                slot = bx
                plane = by
                tile = bz
                lane = tx & 31
                warp = tx // 32
                slot_base = staging_offsets[slot]
                block_id = block_ids[slot]
                mode = modes[slot]
                if mode != 0 and tx == 0:
                    flags_offset = (
                        slot_base
                        + _SLOT_FIXED_HEADER_BYTES
                        + num_planes * _DESCRIPTOR_BYTES
                    )
                    flags = T.cast(staging[flags_offset], T.uint32)
                    escape_packed_shared[0] = flags & _ESCAPE_PACKED_FLAG
                    escape_symbol_bits_shared[0] = (
                        3 if (flags & _ESCAPE_3BIT_FLAG) != 0 else 4
                    )
                    symbol_bits_shared[0] = flags & 0x3F
                    if symbol_bits_shared[0] == 0:
                        symbol_bits_shared[0] = 4
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
                    if tx == 0:
                        escape_count_shared[0] = (
                            T.cast(staging[descriptor + 36], T.uint32)
                            | (T.cast(staging[descriptor + 37], T.uint32) << 8)
                            | (T.cast(staging[descriptor + 38], T.uint32) << 16)
                            | (T.cast(staging[descriptor + 39], T.uint32) << 24)
                        )
                T.sync_threads()
                low_offset = descriptor_offsets[0] if mode != 0 else 0
                symbol_offset = descriptor_offsets[1] if mode != 0 else 0
                prefix_offset = descriptor_offsets[2] if mode != 0 else 0
                escape_offset = descriptor_offsets[3] if mode != 0 else 0
                prefix_value = T.alloc_var(T.uint32, init=0)
                raw_prefix_value = T.alloc_var(T.uint32, init=0)
                if mode != 0:
                    prefix_base = slot_base + prefix_offset
                    prefix_value = (
                        T.cast(staging[prefix_base + tile * 4], T.uint32)
                        | (T.cast(staging[prefix_base + tile * 4 + 1], T.uint32) << 8)
                        | (T.cast(staging[prefix_base + tile * 4 + 2], T.uint32) << 16)
                        | (T.cast(staging[prefix_base + tile * 4 + 3], T.uint32) << 24)
                    )
                    if escape_packed_shared[0] != 0:
                        raw_prefix_base = prefix_base + 4 * (tiles_per_plane + 1)
                        raw_prefix_value = (
                            T.cast(staging[raw_prefix_base + tile * 4], T.uint32)
                            | (
                                T.cast(
                                    staging[raw_prefix_base + tile * 4 + 1], T.uint32
                                )
                                << 8
                            )
                            | (
                                T.cast(
                                    staging[raw_prefix_base + tile * 4 + 2], T.uint32
                                )
                                << 16
                            )
                            | (
                                T.cast(
                                    staging[raw_prefix_base + tile * 4 + 3], T.uint32
                                )
                                << 24
                            )
                        )
                escape_group_prefix = T.alloc_var(T.uint32, init=0)
                raw_escape_group_prefix = T.alloc_var(T.uint32, init=0)
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
                    code = T.alloc_var(T.uint32, init=0)
                    escape_code = T.alloc_var(T.uint32, init=15)
                    if symbol_bits_shared[0] == 3:
                        bit_offset = scalar * 3
                        symbol_byte = bit_offset // 8
                        symbol_shift = bit_offset & 7
                        packed = (
                            T.cast(
                                staging[slot_base + symbol_offset + symbol_byte],
                                T.uint32,
                            )
                            | (
                                T.cast(
                                    staging[
                                        slot_base + symbol_offset + symbol_byte + 1
                                    ],
                                    T.uint32,
                                )
                                << 8
                            )
                            if active
                            else 0
                        )
                        code = (packed >> symbol_shift) & 7
                        escape_code = 7
                    else:
                        symbol = (
                            T.cast(
                                staging[slot_base + symbol_offset + scalar // 2],
                                T.uint8,
                            )
                            if active
                            else 0
                        )
                        code = (symbol >> ((scalar & 1) * 4)) & 15
                    mask = T.ballot(active and code == escape_code)
                    if lane == 0:
                        warp_counts[warp] = T.popcount(mask)
                    T.sync_threads()
                    if tx == 0:
                        running = T.alloc_var(T.uint32, init=0)
                        for index in range(_LOAD_WARPS):
                            warp_prefix[index] = running
                            running = running + warp_counts[index]
                    T.sync_threads()
                    token_rank = (
                        prefix_value
                        + escape_group_prefix
                        + warp_prefix[warp]
                        + T.popcount(mask & ((1 << lane) - 1))
                    )
                    packed_escape_value = T.alloc_var(T.uint32, init=15)
                    if escape_packed_shared[0] != 0 and code == escape_code and active:
                        if escape_symbol_bits_shared[0] == 3:
                            escape_bit_offset = token_rank * 3
                            escape_byte = escape_bit_offset // 8
                            escape_shift = escape_bit_offset & 7
                            packed = T.alloc_var(T.uint32, init=0)
                            packed = T.cast(
                                staging[slot_base + escape_offset + escape_byte],
                                T.uint32,
                            )
                            packed = packed | (
                                T.cast(
                                    staging[
                                        slot_base + escape_offset + escape_byte + 1
                                    ],
                                    T.uint32,
                                )
                                << 8
                            )
                            packed_escape_value = (packed >> escape_shift) & 7
                        else:
                            packed_escape_value = (
                                T.cast(
                                    staging[
                                        slot_base + escape_offset + token_rank // 2
                                    ],
                                    T.uint32,
                                )
                                >> ((token_rank & 1) * 4)
                            ) & 15
                    raw_mask = T.ballot(
                        escape_packed_shared[0] != 0
                        and code == escape_code
                        and packed_escape_value
                        == ((1 << escape_symbol_bits_shared[0]) - 1)
                        and active
                    )
                    if lane == 0:
                        warp_raw_counts[warp] = T.popcount(raw_mask)
                    T.sync_threads()
                    if tx == 0:
                        raw_running = T.alloc_var(T.uint32, init=0)
                        for index in range(_LOAD_WARPS):
                            warp_raw_prefix[index] = raw_running
                            raw_running = raw_running + warp_raw_counts[index]
                    T.sync_threads()
                    raw_rank = (
                        raw_prefix_value
                        + raw_escape_group_prefix
                        + warp_raw_prefix[warp]
                        + T.popcount(raw_mask & ((1 << lane) - 1))
                    )
                    escape_code_bytes = (
                        escape_count_shared[0] * escape_symbol_bits_shared[0] + 7
                    ) // 8
                    packed_decoded_high = T.if_then_else(
                        code == escape_code
                        and packed_escape_value
                        == ((1 << escape_symbol_bits_shared[0]) - 1)
                        and active,
                        T.cast(
                            staging[
                                slot_base + escape_offset + escape_code_bytes + raw_rank
                            ],
                            T.uint16,
                        ),
                        T.if_then_else(
                            code == escape_code and active,
                            T.cast(
                                codebooks[
                                    plane * _CODEBOOK_ENTRIES + 7 + packed_escape_value
                                ],
                                T.uint16,
                            ),
                            T.cast(
                                codebooks[plane * _CODEBOOK_ENTRIES + code],
                                T.uint16,
                            ),
                        ),
                    )
                    unpacked_decoded_high = T.if_then_else(
                        code == escape_code and active,
                        T.cast(
                            staging[slot_base + escape_offset + token_rank],
                            T.uint16,
                        ),
                        T.cast(
                            codebooks[plane * _CODEBOOK_ENTRIES + code],
                            T.uint16,
                        ),
                    )
                    decoded_high = T.if_then_else(
                        escape_packed_shared[0] != 0,
                        packed_decoded_high,
                        unpacked_decoded_high,
                    )
                    decoded_low = (
                        T.cast(staging[slot_base + low_offset + scalar], T.uint16)
                        if active
                        else 0
                    )
                    dst[block_id, plane, scalar] = (
                        decoded_low | (decoded_high << 8) if active else 0
                    )
                    # ``warp_prefix`` covers only the current 128-thread
                    # group. Carry its total into the next group so ranks
                    # remain global across the complete tile.
                    escape_group_prefix = (
                        escape_group_prefix
                        + warp_prefix[_LOAD_WARPS - 1]
                        + warp_counts[_LOAD_WARPS - 1]
                    )
                    raw_escape_group_prefix = (
                        raw_escape_group_prefix
                        + warp_raw_prefix[_LOAD_WARPS - 1]
                        + warp_raw_counts[_LOAD_WARPS - 1]
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
