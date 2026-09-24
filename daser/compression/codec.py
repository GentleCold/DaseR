# SPDX-License-Identifier: Apache-2.0

"""CPU reference codec for strict-lossless BF16 byte-plane slots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from daser.compression.format import (
    CODEBOOK_ENTRIES,
    ESCAPE_SYMBOL_BITS,
    IO_ALIGNMENT,
    KV_PLANES,
    SYMBOL_BITS,
    CompressedStoreGeometry,
    PlaneDescriptor,
    SlotHeader,
    SlotMode,
    align_up,
    digest_bytes,
)

# Qwen3-8B's K and V planes have different high-byte neighborhoods.  The
# calibration artifact keeps all K palettes followed by all V palettes because
# that is the natural layer-wise form used while fitting the tables.  The
# online cache layout is instead interleaved as ``K(layer 0), V(layer 0), ...``;
# ``default_online_codebooks`` explicitly converts between those two orders.
# The secondary entries are also fixed per plane.  The narrow three-bit escape
# stream still carries their indices, but keeping the frequent tail values in
# the table avoids writing them as raw bytes.  This is a startup constant, not
# a per-slot calibration pass, so it adds no request-path work or metadata.
_QWEN3_8B_PRIMARY_CODEBOOKS: tuple[tuple[int, ...], ...] = (
    (191, 63, 62, 190, 64, 192, 61),
    (191, 63, 190, 62, 64, 192, 189),
    (191, 63, 190, 62, 64, 192, 189),
    (63, 191, 62, 190, 64, 192, 61),
    (63, 191, 62, 190, 64, 192, 189),
    (63, 191, 192, 64, 190, 62, 189),
    (63, 191, 62, 190, 64, 192, 189),
    (63, 191, 190, 62, 64, 192, 61),
    (63, 191, 190, 62, 64, 192, 61),
    (191, 63, 190, 62, 192, 64, 189),
    (63, 191, 190, 62, 192, 64, 189),
    (63, 191, 62, 190, 64, 192, 61),
    (63, 191, 190, 62, 64, 192, 189),
    (63, 191, 62, 190, 64, 192, 61),
    (191, 63, 190, 62, 192, 64, 189),
    (191, 63, 190, 62, 192, 64, 61),
    (191, 63, 190, 62, 192, 64, 189),
    (63, 191, 62, 190, 192, 64, 61),
    (63, 191, 62, 190, 64, 192, 61),
    (191, 63, 190, 62, 192, 64, 189),
    (63, 191, 62, 190, 192, 64, 189),
    (63, 191, 62, 190, 64, 192, 61),
    (63, 191, 62, 190, 192, 64, 189),
    (191, 63, 62, 190, 192, 64, 61),
    (191, 63, 62, 190, 192, 64, 61),
    (63, 191, 62, 190, 64, 192, 61),
    (191, 63, 190, 62, 192, 64, 189),
    (191, 63, 190, 62, 64, 192, 189),
    (63, 191, 62, 190, 192, 64, 61),
    (63, 191, 62, 190, 64, 192, 61),
    (63, 191, 62, 190, 192, 64, 61),
    (63, 191, 62, 190, 192, 64, 61),
    (191, 63, 190, 62, 64, 192, 189),
    (63, 191, 62, 190, 192, 64, 189),
    (63, 191, 62, 190, 64, 192, 61),
    (63, 191, 190, 62, 64, 192, 189),
    (188, 60, 187, 59, 61, 189, 186),
    (61, 189, 60, 188, 187, 59, 190),
    (61, 189, 188, 60, 59, 187, 190),
    (189, 61, 188, 60, 190, 62, 59),
    (189, 61, 190, 62, 188, 60, 187),
    (189, 61, 190, 62, 188, 60, 59),
    (61, 189, 190, 62, 60, 188, 59),
    (190, 62, 189, 61, 188, 60, 191),
    (190, 62, 61, 189, 191, 63, 188),
    (62, 190, 61, 189, 63, 191, 188),
    (190, 62, 191, 63, 189, 61, 188),
    (190, 62, 189, 61, 191, 63, 188),
    (62, 190, 61, 189, 191, 63, 60),
    (62, 190, 61, 189, 63, 191, 60),
    (190, 62, 189, 61, 63, 191, 188),
    (62, 190, 189, 61, 63, 191, 188),
    (62, 190, 63, 191, 61, 189, 60),
    (62, 190, 63, 191, 61, 189, 60),
    (190, 62, 191, 63, 189, 61, 188),
    (190, 62, 191, 63, 189, 61, 60),
    (190, 62, 191, 63, 189, 61, 60),
    (63, 191, 62, 190, 61, 189, 188),
    (63, 191, 62, 190, 61, 189, 192),
    (63, 191, 62, 190, 61, 189, 64),
    (191, 63, 190, 62, 64, 192, 61),
    (63, 191, 190, 62, 64, 192, 189),
    (191, 63, 190, 62, 192, 64, 189),
    (63, 191, 64, 192, 62, 190, 61),
    (63, 191, 192, 64, 62, 190, 61),
    (191, 63, 192, 64, 190, 62, 61),
    (191, 63, 64, 192, 190, 62, 61),
    (192, 64, 63, 191, 190, 62, 61),
    (192, 64, 63, 191, 190, 62, 193),
    (192, 64, 63, 191, 193, 65, 62),
    (192, 64, 191, 63, 193, 65, 190),
    (192, 64, 63, 191, 62, 190, 61),
)

# Unlike the primary calibration artifact above, these entries are stored in
# physical interleaved ``K(layer 0), V(layer 0), ...`` order to match the
# online lookup tensor directly.
_QWEN3_8B_SECONDARY_CODEBOOKS: tuple[tuple[int, ...], ...] = (
    (189, 188, 60, 193, 65, 187, 67),
    (58, 185, 57, 62, 184, 190, 56),
    (61, 188, 60, 193, 65, 187, 59),
    (58, 186, 62, 57, 185, 56, 184),
    (61, 188, 60, 65, 193, 187, 59),
    (62, 186, 58, 185, 57, 184, 56),
    (189, 188, 60, 65, 187, 59, 193),
    (187, 186, 58, 57, 185, 56, 184),
    (61, 193, 188, 60, 65, 187, 59),
    (59, 186, 58, 57, 185, 56, 184),
    (61, 65, 193, 60, 188, 66, 187),
    (187, 186, 58, 63, 185, 57, 191),
    (61, 188, 60, 65, 193, 187, 59),
    (187, 186, 58, 191, 63, 185, 57),
    (189, 188, 60, 65, 59, 187, 193),
    (59, 187, 63, 58, 186, 57, 185),
    (189, 60, 188, 65, 194, 193, 187),
    (60, 187, 59, 186, 58, 57, 185),
    (61, 188, 60, 193, 59, 187, 58),
    (60, 187, 59, 186, 58, 57, 185),
    (61, 188, 60, 193, 65, 194, 187),
    (60, 59, 187, 186, 58, 185, 57),
    (189, 65, 193, 60, 188, 59, 187),
    (60, 59, 187, 58, 186, 57, 185),
    (61, 60, 188, 193, 65, 59, 187),
    (188, 59, 187, 186, 58, 57, 185),
    (189, 188, 60, 59, 187, 193, 65),
    (188, 59, 187, 186, 58, 185, 57),
    (61, 193, 60, 188, 65, 59, 187),
    (60, 187, 59, 186, 58, 185, 57),
    (189, 60, 188, 65, 59, 187, 193),
    (60, 59, 187, 186, 58, 57, 185),
    (61, 60, 188, 193, 65, 59, 187),
    (188, 59, 187, 58, 186, 185, 57),
    (189, 188, 60, 187, 59, 65, 193),
    (188, 59, 187, 58, 186, 192, 185),
    (189, 188, 60, 65, 193, 187, 59),
    (60, 59, 187, 186, 58, 185, 57),
    (61, 188, 60, 193, 59, 187, 65),
    (188, 59, 187, 64, 192, 186, 58),
    (61, 188, 60, 59, 187, 65, 193),
    (188, 59, 187, 64, 192, 186, 58),
    (189, 60, 188, 65, 187, 59, 193),
    (60, 59, 187, 64, 192, 186, 58),
    (61, 188, 60, 65, 193, 59, 187),
    (188, 60, 64, 187, 59, 58, 186),
    (189, 60, 188, 193, 65, 59, 187),
    (60, 188, 192, 59, 187, 58, 186),
    (189, 60, 188, 59, 187, 65, 58),
    (189, 188, 60, 187, 59, 186, 58),
    (189, 188, 60, 65, 193, 187, 59),
    (61, 188, 60, 59, 187, 58, 186),
    (61, 60, 188, 59, 187, 193, 186),
    (61, 188, 60, 187, 59, 186, 58),
    (61, 188, 60, 65, 187, 59, 58),
    (189, 188, 60, 187, 59, 58, 186),
    (189, 60, 188, 65, 193, 187, 59),
    (189, 60, 188, 187, 59, 193, 186),
    (189, 188, 60, 187, 59, 65, 193),
    (189, 60, 188, 193, 65, 59, 187),
    (189, 60, 188, 65, 193, 59, 187),
    (189, 60, 188, 193, 65, 59, 187),
    (189, 188, 60, 59, 187, 193, 65),
    (189, 65, 193, 60, 188, 187, 59),
    (61, 60, 188, 193, 65, 187, 59),
    (65, 189, 61, 188, 60, 187, 59),
    (61, 60, 188, 187, 59, 65, 193),
    (190, 61, 189, 188, 60, 187, 59),
    (189, 188, 60, 65, 193, 59, 187),
    (62, 189, 61, 188, 60, 59, 187),
    (61, 193, 188, 60, 65, 187, 59),
    (189, 193, 65, 188, 60, 187, 59),
)

_GENERIC_ONLINE_CODEBOOK = (
    63,
    191,
    62,
    190,
    64,
    192,
    61,
    189,
    60,
    188,
    59,
    187,
    58,
    186,
    57,
)


def _complete_online_codebook(
    primary: tuple[int, ...],
    *,
    secondary: tuple[int, ...] | None = None,
    raw_sentinel: int | None = None,
) -> tuple[int, ...]:
    """Fill a primary table while reserving the last entry for raw escapes.

    The packed three-bit escape stream treats codebook entry fourteen as a raw
    byte rather than as a secondary three-bit symbol. ``secondary`` supplies
    the seven frequent extension values before the generic fallback. A caller
    can therefore reserve a deliberately rare value for that entry without
    changing the persisted format.
    """
    values = list(primary)
    target = CODEBOOK_ENTRIES - (1 if raw_sentinel is not None else 0)
    candidates = (
        *(secondary or ()),
        *_GENERIC_ONLINE_CODEBOOK,
    )
    for value in candidates:
        if value not in values:
            values.append(value)
        if len(values) == target:
            break
    if raw_sentinel is not None:
        if raw_sentinel in values:
            raise ValueError("raw sentinel duplicates an online codebook entry")
        values.append(raw_sentinel)
    if len(values) != CODEBOOK_ENTRIES:
        raise ValueError("online codebook does not contain fifteen entries")
    return tuple(values)


@dataclass(frozen=True)
class EncodedSlot:
    """Result of choosing compressed or explicit raw mode for one slot.

    Attributes:
        slot_id: Fixed-envelope logical slot ID.
        mode: COMPRESSED when the encoded bytes fit, otherwise RAW.
        payload: Aligned bytes written at the start of the envelope.
        raw_hash: SHA-256 of the original slot bytes.
        encoded_hash: SHA-256 of the stored payload.
    """

    slot_id: int
    mode: SlotMode
    payload: bytes
    raw_hash: bytes
    encoded_hash: bytes


def default_online_codebooks(geometry: CompressedStoreGeometry) -> bytes:
    """Return a deterministic codebook for online stores.

    Online serving cannot pause for an activation calibration pass.  Qwen3-8B
    uses a fixed per-plane primary palette learned from representative BF16
    KV activations; this keeps the seven values representable by the 3-bit
    stream aligned with each K/V plane's sign and exponent neighborhood.  For
    other geometries, the model-independent palette remains the conservative
    fallback. Values outside the primary table remain lossless escapes.

    Args:
        geometry: KV geometry whose plane count determines the output size.

    Returns:
        Plane-major 15-entry codebook bytes.

    Async/thread-safety:
        Pure startup computation; safe to call from any thread.
    """
    if (
        geometry.num_layers == 36
        and geometry.num_kv_heads == 8
        and geometry.head_dim == 128
        and geometry.plane_count == len(_QWEN3_8B_PRIMARY_CODEBOOKS)
    ):
        # ``_QWEN3_8B_PRIMARY_CODEBOOKS`` is stored as two layer-major blocks
        # (all K tables, then all V tables), while the slot codec walks the
        # physical KV layout in per-layer K/V order.  Keeping this conversion
        # at the one codebook boundary prevents a silent loss of compression
        # on every V plane and preserves the decoder's existing plane index.
        layer_count = geometry.num_layers
        values = b"".join(
            bytes(
                _complete_online_codebook(
                    _QWEN3_8B_PRIMARY_CODEBOOKS[
                        layer if kv == 0 else layer_count + layer
                    ],
                    secondary=_QWEN3_8B_SECONDARY_CODEBOOKS[layer * KV_PLANES + kv],
                    # Entry fourteen is emitted as a raw byte by the narrow
                    # escape stream.  Zero is outside the dense BF16
                    # high-byte neighborhoods used by valid Qwen3 KV values,
                    # so reserving it avoids turning a common secondary value
                    # into an eight-bit escape without changing the decoder.
                    raw_sentinel=0,
                )
            )
            for layer in range(layer_count)
            for kv in range(KV_PLANES)
        )
        return values
    values = bytes(_GENERIC_ONLINE_CODEBOOK)
    return values * geometry.plane_count


def encode_slot(
    raw_slot: bytes | bytearray | memoryview,
    *,
    slot_id: int,
    geometry: CompressedStoreGeometry,
    codebooks: bytes,
) -> EncodedSlot:
    """Encode one raw BF16 slot into its fixed envelope.

    This is the byte-exact reference for the online GPU packer. The low byte
    of every scalar is stored verbatim. High bytes map to a plane-local
    three-bit code; symbol 7 escapes to a packed three-bit secondary code or,
    for bytes outside the table, a raw byte. The complete slot payload is
    4 KiB aligned for O_DIRECT slot reads; plane records are packed
    back-to-back inside that slot because they are never submitted as
    independent physical reads.

    Args:
        raw_slot: Production slot-major BF16 bytes.
        slot_id: Logical fixed-envelope slot ID.
        geometry: Exact model and store geometry.
        codebooks: Plane-major static tables.

    Returns:
        EncodedSlot. Incompressible payloads use explicit RAW mode.

    Raises:
        ValueError: If inputs violate the format geometry.

    Async/thread-safety:
        CPU-bound synchronous operation using only local arrays.
    """
    _validate_codec_inputs(raw_slot, geometry, codebooks)
    if slot_id < 0 or slot_id >= geometry.num_slots:
        raise ValueError("slot_id is outside the compressed store")
    raw_bytes = bytes(raw_slot)
    raw_hash = digest_bytes(raw_bytes)
    codebook_hash = digest_bytes(codebooks)
    tables = np.frombuffer(codebooks, dtype=np.uint8).reshape(
        geometry.plane_count, CODEBOOK_ENTRIES
    )
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)

    encoded_planes: list[
        tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]
    ] = []
    for plane in range(geometry.plane_count):
        plane_start = plane * geometry.plane_bytes
        plane_bytes = raw[plane_start : plane_start + geometry.plane_bytes]
        low = np.ascontiguousarray(plane_bytes[0::2])
        high = np.ascontiguousarray(plane_bytes[1::2])
        codes = _encode_high_bytes(high, tables[plane])
        encoded_planes.append((low, high, codes))

    records: list[tuple[PlaneDescriptor, bytes, bytes, bytes, bytes]] = []
    cursor = IO_ALIGNMENT
    escape_code = (1 << SYMBOL_BITS) - 1
    raw_code = (1 << ESCAPE_SYMBOL_BITS) - 1
    for plane, (low, high, codes) in enumerate(encoded_planes):
        # Primary entries 0-6 are direct symbols; entries 7-13 become escape
        # tokens, and entry 14 or an unmapped byte is stored raw.
        escape_mask = codes >= escape_code
        raw_mask = codes >= escape_code + raw_code
        prefixes = _escape_prefixes(escape_mask, geometry.tile_scalars)
        escape_codes = np.where(
            raw_mask[escape_mask], raw_code, codes[escape_mask] - escape_code
        ).astype(np.uint8, copy=False)
        escapes = (
            _pack_codes(escape_codes).tobytes()
            + np.ascontiguousarray(high[raw_mask]).tobytes()
        )
        escape_count = len(escape_codes)
        prefix_bytes = (
            prefixes.astype("<u4", copy=False).tobytes()
            + _escape_prefixes(raw_mask, geometry.tile_scalars)
            .astype("<u4", copy=False)
            .tobytes()
        )
        symbols = _pack_codes(np.minimum(codes, escape_code))

        low_bytes = low.tobytes()
        symbol_bytes = symbols.tobytes()
        record_offset = cursor
        low_offset = record_offset
        symbol_offset = low_offset + len(low_bytes)
        prefix_offset = symbol_offset + len(symbol_bytes)
        escape_offset = prefix_offset + len(prefix_bytes)
        record_length = escape_offset + len(escapes) - record_offset
        layer, kv = divmod(plane, KV_PLANES)
        descriptor = PlaneDescriptor(
            layer=layer,
            kv=kv,
            scalar_count=geometry.plane_scalars,
            tile_count=len(prefixes) - 1,
            record_offset=record_offset,
            record_length=record_length,
            low_offset=low_offset,
            symbol_offset=symbol_offset,
            prefix_offset=prefix_offset,
            escape_offset=escape_offset,
            escape_count=escape_count,
        )
        records.append((descriptor, low_bytes, symbol_bytes, prefix_bytes, escapes))
        cursor += record_length

    stored_length = align_up(cursor)
    if stored_length > geometry.slot_size:
        return EncodedSlot(
            slot_id=slot_id,
            mode=SlotMode.RAW,
            payload=raw_bytes,
            raw_hash=raw_hash,
            encoded_hash=raw_hash,
        )

    header = SlotHeader(
        slot_id=slot_id,
        raw_length=geometry.slot_size,
        stored_length=stored_length,
        tile_scalars=geometry.tile_scalars,
        num_layers=geometry.num_layers,
        codebook_hash=codebook_hash,
        raw_hash=raw_hash,
        descriptors=tuple(record[0] for record in records),
    )
    payload = bytearray(stored_length)
    payload[:IO_ALIGNMENT] = header.pack()
    for descriptor, low, symbols, prefixes, escapes in records:
        payload[descriptor.low_offset : descriptor.low_offset + len(low)] = low
        payload[descriptor.symbol_offset : descriptor.symbol_offset + len(symbols)] = (
            symbols
        )
        payload[descriptor.prefix_offset : descriptor.prefix_offset + len(prefixes)] = (
            prefixes
        )
        payload[descriptor.escape_offset : descriptor.escape_offset + len(escapes)] = (
            escapes
        )
    encoded = bytes(payload)
    return EncodedSlot(
        slot_id=slot_id,
        mode=SlotMode.COMPRESSED,
        payload=encoded,
        raw_hash=raw_hash,
        encoded_hash=digest_bytes(encoded),
    )


def decode_slot(
    encoded: bytes | bytearray | memoryview,
    *,
    mode: SlotMode,
    slot_id: int,
    geometry: CompressedStoreGeometry,
    codebooks: bytes,
    verify_hash: bool = True,
) -> bytes:
    """Reference-decode one encoded slot byte-for-byte.

    Args:
        encoded: Stored slot bytes, excluding the unused envelope tail.
        mode: Slot mode recorded with the encoded bytes.
        slot_id: Expected fixed-envelope logical slot.
        geometry: Exact model and store geometry.
        codebooks: Static plane-major high-byte tables.
        verify_hash: Recompute and validate the header raw hash when true.

    Returns:
        Original slot-major BF16 bytes.

    Raises:
        ValueError: On malformed metadata, truncation, or hash mismatch.

    Async/thread-safety:
        CPU-bound synchronous reference implementation. Production GPU restore
        uses the same descriptor contract without calling this function.
    """
    _validate_codebooks(geometry, codebooks)
    if mode is SlotMode.RAW:
        if len(encoded) != geometry.slot_size:
            raise ValueError("raw-mode slot length does not match geometry")
        return bytes(encoded)
    if mode is not SlotMode.COMPRESSED:
        raise ValueError("unknown compressed slot mode")
    header = SlotHeader.parse(
        encoded,
        expected_geometry=geometry,
        expected_slot_id=slot_id,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    tables = np.frombuffer(codebooks, dtype=np.uint8).reshape(
        geometry.plane_count, CODEBOOK_ENTRIES
    )
    source = memoryview(encoded)
    decoded = bytearray(geometry.slot_size)
    escape_code = (1 << SYMBOL_BITS) - 1
    raw_code = (1 << ESCAPE_SYMBOL_BITS) - 1
    for plane, descriptor in enumerate(header.descriptors):
        scalar_count = descriptor.scalar_count
        low = np.frombuffer(
            source[descriptor.low_offset : descriptor.low_offset + scalar_count],
            dtype=np.uint8,
        )
        symbol_length = (scalar_count * SYMBOL_BITS + 7) // 8
        symbols = np.frombuffer(
            source[descriptor.symbol_offset : descriptor.symbol_offset + symbol_length],
            dtype=np.uint8,
        )
        codes = _unpack_codes(symbols, scalar_count)
        escape_mask = codes == escape_code
        high: NDArray[np.uint8] = np.empty(scalar_count, dtype=np.uint8)
        coded_mask = ~escape_mask
        high[coded_mask] = tables[plane, codes[coded_mask]]
        if int(np.count_nonzero(escape_mask)) != descriptor.escape_count:
            raise ValueError("compressed plane escape token count mismatch")
        escape_code_bytes = (descriptor.escape_count * ESCAPE_SYMBOL_BITS + 7) // 8
        packed_escape_codes = _unpack_codes(
            np.frombuffer(
                source[
                    descriptor.escape_offset : descriptor.escape_offset
                    + escape_code_bytes
                ],
                dtype=np.uint8,
            ),
            descriptor.escape_count,
        )
        escape_indices = np.flatnonzero(escape_mask)
        raw_mask = packed_escape_codes == raw_code
        raw_count = int(np.count_nonzero(raw_mask))
        raw_escapes = np.frombuffer(
            source[
                descriptor.escape_offset + escape_code_bytes : descriptor.escape_offset
                + escape_code_bytes
                + raw_count
            ],
            dtype=np.uint8,
        )
        high[escape_indices[~raw_mask]] = tables[
            plane, escape_code + packed_escape_codes[~raw_mask]
        ]
        high[escape_indices[raw_mask]] = raw_escapes
        plane_start = plane * geometry.plane_bytes
        plane_view = memoryview(decoded)[
            plane_start : plane_start + geometry.plane_bytes
        ]
        interleaved = np.frombuffer(plane_view, dtype=np.uint8)
        interleaved[0::2] = low
        interleaved[1::2] = high
    result = bytes(decoded)
    if verify_hash and digest_bytes(result) != header.raw_hash:
        raise ValueError("decoded slot hash mismatch")
    return result


def _validate_codec_inputs(
    raw_slot: bytes | bytearray | memoryview,
    geometry: CompressedStoreGeometry,
    codebooks: bytes,
) -> None:
    if len(raw_slot) != geometry.slot_size:
        raise ValueError("raw slot length does not match geometry")
    _validate_codebooks(geometry, codebooks)


def _validate_codebooks(
    geometry: CompressedStoreGeometry,
    codebooks: bytes,
) -> None:
    expected = geometry.plane_count * CODEBOOK_ENTRIES
    if len(codebooks) != expected:
        raise ValueError(
            f"codebooks contain {len(codebooks)} bytes, expected {expected}"
        )
    tables = np.frombuffer(codebooks, dtype=np.uint8).reshape(
        geometry.plane_count, CODEBOOK_ENTRIES
    )
    if any(len(np.unique(table)) != CODEBOOK_ENTRIES for table in tables):
        raise ValueError("each codebook must contain 15 distinct bytes")


def _encode_high_bytes(
    high: NDArray[np.uint8],
    codebook: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    lookup: NDArray[np.uint8] = np.full(256, CODEBOOK_ENTRIES, dtype=np.uint8)
    lookup[codebook] = np.arange(CODEBOOK_ENTRIES, dtype=np.uint8)
    return lookup[high]


def _pack_codes(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack three-bit codes into a dense little-endian bit stream."""
    packed = np.zeros((len(codes) * SYMBOL_BITS + 7) // 8, dtype=np.uint8)
    for index, code in enumerate(codes):
        bit_offset = index * SYMBOL_BITS
        byte_offset = bit_offset // 8
        shift = bit_offset & 7
        value = int(code) << shift
        packed[byte_offset] |= value & 0xFF
        if shift > 5:
            packed[byte_offset + 1] |= (value >> 8) & 0xFF
    return packed


def _unpack_codes(symbols: NDArray[np.uint8], scalar_count: int) -> NDArray[np.uint8]:
    """Unpack a dense little-endian three-bit symbol stream."""
    codes = np.empty(scalar_count, dtype=np.uint8)
    for index in range(scalar_count):
        bit_offset = index * SYMBOL_BITS
        byte_offset = bit_offset // 8
        shift = bit_offset & 7
        value = int(symbols[byte_offset])
        if shift > 5:
            value |= int(symbols[byte_offset + 1]) << 8
        codes[index] = (value >> shift) & 0x07
    return codes


def _escape_prefixes(
    escape_mask: NDArray[np.bool_], tile_scalars: int
) -> NDArray[np.uint32]:
    scalar_count = len(escape_mask)
    tile_count = (scalar_count + tile_scalars - 1) // tile_scalars
    cumulative = np.cumsum(escape_mask, dtype=np.uint32)
    prefixes: NDArray[np.uint32] = np.zeros(tile_count + 1, dtype=np.uint32)
    for tile in range(tile_count):
        end = min((tile + 1) * tile_scalars, scalar_count)
        prefixes[tile + 1] = cumulative[end - 1]
    return prefixes


__all__ = [
    "EncodedSlot",
    "default_online_codebooks",
    "decode_slot",
    "encode_slot",
]
