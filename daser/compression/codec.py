# SPDX-License-Identifier: Apache-2.0

"""Offline strict-lossless BF16 byte-plane codec."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from daser.compression.format import (
    CODEBOOK_ENTRIES,
    IO_ALIGNMENT,
    KV_PLANES,
    CompressedSlotRef,
    CompressedStoreGeometry,
    CompressedStoreIndex,
    PlaneDescriptor,
    SlotHeader,
    SlotMode,
    align_up,
    digest_bytes,
)
from daser.logging import init_logger

logger = init_logger(__name__)


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


def calibrate_codebooks(
    raw_slots: Iterable[bytes | bytearray | memoryview],
    geometry: CompressedStoreGeometry,
) -> bytes:
    """Fit one deterministic 15-entry high-byte codebook per layer/K/V plane.

    Args:
        raw_slots: Calibration-only raw slots in production slot-major order.
        geometry: Exact slot and model geometry.

    Returns:
        Plane-major codebook bytes with ``num_layers * 2 * 15`` entries.

    Raises:
        ValueError: If no slots are provided or a slot has the wrong length.

    Async/thread-safety:
        CPU-bound synchronous offline operation. It does not mutate inputs and
        is safe across callers that provide independent iterables.
    """
    counts: NDArray[np.uint64] = np.zeros((geometry.plane_count, 256), dtype=np.uint64)
    slot_count = 0
    for raw_slot in raw_slots:
        if len(raw_slot) != geometry.slot_size:
            raise ValueError("calibration slot length does not match geometry")
        slot = np.frombuffer(raw_slot, dtype=np.uint8)
        for plane in range(geometry.plane_count):
            start = plane * geometry.plane_bytes
            high = slot[start + 1 : start + geometry.plane_bytes : 2]
            counts[plane] += np.bincount(high, minlength=256).astype(np.uint64)
        slot_count += 1
    if slot_count == 0:
        raise ValueError("at least one calibration slot is required")

    byte_values: NDArray[np.int64] = np.arange(256, dtype=np.int64)
    codebooks: NDArray[np.uint8] = np.empty(
        (geometry.plane_count, CODEBOOK_ENTRIES), dtype=np.uint8
    )
    for plane in range(geometry.plane_count):
        # lexsort uses the final key as primary: count descending, byte ascending.
        order = np.lexsort((byte_values, -counts[plane].astype(np.int64)))
        codebooks[plane] = order[:CODEBOOK_ENTRIES]
    return codebooks.tobytes()


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

    The low byte of every scalar is stored verbatim. High bytes map to a
    plane-local four-bit code; symbol 15 stores the original byte in an escape
    stream. The complete slot payload is 4 KiB aligned for O_DIRECT slot reads;
    plane records are packed back-to-back inside that slot because they are
    never submitted as independent physical reads.

    Args:
        raw_slot: Production slot-major BF16 bytes.
        slot_id: Logical fixed-envelope slot ID.
        geometry: Exact model and store geometry.
        codebooks: Calibration-only plane-major static tables.

    Returns:
        EncodedSlot. Incompressible payloads use explicit RAW mode.

    Raises:
        ValueError: If inputs violate the format geometry.

    Async/thread-safety:
        CPU-bound synchronous offline operation using only local arrays.
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

    symbol_bits, escape_symbol_bits = _select_symbol_bits(encoded_planes, geometry)
    records: list[tuple[PlaneDescriptor, bytes, bytes, bytes, bytes]] = []
    cursor = IO_ALIGNMENT
    for plane, (low, high, codes) in enumerate(encoded_planes):
        escape_mask = (
            codes >= (1 << symbol_bits) - 1
            if symbol_bits == 3
            else codes == CODEBOOK_ENTRIES
        )
        escape_packed = symbol_bits == 3
        prefixes = _escape_prefixes(escape_mask, geometry.tile_scalars)
        if escape_packed:
            raw_code = (1 << escape_symbol_bits) - 1
            escape_codes = np.where(
                codes[escape_mask] >= (CODEBOOK_ENTRIES - (4 - escape_symbol_bits)),
                raw_code,
                codes[escape_mask] - 7,
            ).astype(np.uint8, copy=False)
            escape_symbols = _pack_codes(escape_codes, escape_symbol_bits).tobytes()
            raw_escapes = np.ascontiguousarray(
                high[codes >= (CODEBOOK_ENTRIES - (4 - escape_symbol_bits))]
            ).tobytes()
            escapes = escape_symbols + raw_escapes
            escape_count = len(escape_codes)
            raw_prefixes = _escape_prefixes(
                codes >= (CODEBOOK_ENTRIES - (4 - escape_symbol_bits)),
                geometry.tile_scalars,
            )
            prefix_bytes = (
                prefixes.astype("<u4", copy=False).tobytes()
                + raw_prefixes.astype("<u4", copy=False).tobytes()
            )
        else:
            escapes = np.ascontiguousarray(high[escape_mask]).tobytes()
            escape_count = len(escapes)
            prefix_bytes = prefixes.astype("<u4", copy=False).tobytes()
        symbols = _pack_codes(codes, symbol_bits)

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
        symbol_bits=symbol_bits,
        escape_packed=escape_packed,
        escape_symbol_bits=escape_symbol_bits,
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
    """Reference-decode one indexed slot byte-for-byte.

    Args:
        encoded: Indexed stored bytes, excluding the unused envelope tail.
        mode: Explicit side-index slot mode.
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
    for plane, descriptor in enumerate(header.descriptors):
        scalar_count = descriptor.scalar_count
        low = np.frombuffer(
            source[descriptor.low_offset : descriptor.low_offset + scalar_count],
            dtype=np.uint8,
        )
        symbol_length = (scalar_count * header.symbol_bits + 7) // 8
        symbols = np.frombuffer(
            source[descriptor.symbol_offset : descriptor.symbol_offset + symbol_length],
            dtype=np.uint8,
        )
        codes = _unpack_codes(symbols, scalar_count, header.symbol_bits)
        escape_mask = codes == (1 << header.symbol_bits) - 1
        high: NDArray[np.uint8] = np.empty(scalar_count, dtype=np.uint8)
        coded_mask = ~escape_mask
        high[coded_mask] = tables[plane, codes[coded_mask]]
        if header.escape_packed:
            if int(np.count_nonzero(escape_mask)) != descriptor.escape_count:
                raise ValueError("compressed plane escape token count mismatch")
            escape_code_bytes = (
                descriptor.escape_count * header.escape_symbol_bits + 7
            ) // 8
            packed_escape_codes = _unpack_codes(
                np.frombuffer(
                    source[
                        descriptor.escape_offset : descriptor.escape_offset
                        + escape_code_bytes
                    ],
                    dtype=np.uint8,
                ),
                descriptor.escape_count,
                header.escape_symbol_bits,
            )
            escape_indices = np.flatnonzero(escape_mask)
            raw_mask = packed_escape_codes == ((1 << header.escape_symbol_bits) - 1)
            raw_count = int(np.count_nonzero(raw_mask))
            raw_escapes = np.frombuffer(
                source[
                    descriptor.escape_offset
                    + escape_code_bytes : descriptor.escape_offset
                    + escape_code_bytes
                    + raw_count
                ],
                dtype=np.uint8,
            )
            high[escape_indices[~raw_mask]] = tables[
                plane, 7 + packed_escape_codes[~raw_mask]
            ]
            high[escape_indices[raw_mask]] = raw_escapes
        else:
            escapes = np.frombuffer(
                source[
                    descriptor.escape_offset : descriptor.escape_offset
                    + descriptor.escape_count
                ],
                dtype=np.uint8,
            )
            if int(np.count_nonzero(escape_mask)) != descriptor.escape_count:
                raise ValueError("compressed plane escape count mismatch")
            high[escape_mask] = escapes
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


def build_compressed_store(
    raw_store_path: str | os.PathLike[str],
    compressed_store_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
    *,
    geometry: CompressedStoreGeometry,
    model_hash: bytes,
    codebooks: bytes,
) -> CompressedStoreIndex:
    """Offline-convert a raw DaseR store into fixed compressed envelopes.

    Args:
        raw_store_path: Source production ``daser.store`` snapshot.
        compressed_store_path: New fixed-envelope data file; it must differ
            from the source path.
        index_path: Destination ``daser.compressed.index`` path.
        geometry: Exact source/store/model geometry.
        model_hash: SHA-256 of the model configuration identity.
        codebooks: Calibration-only static plane-major tables.

    Returns:
        Validated immutable side index written beside the destination store.

    Raises:
        ValueError: On path aliasing, source size, geometry, or codec mismatch.

    Async/thread-safety:
        Synchronous single-writer offline operation. It must run while the
        source snapshot is immutable and outside the server event loop.
    """
    _validate_codebooks(geometry, codebooks)
    source_path = Path(raw_store_path).resolve()
    destination_path = Path(compressed_store_path).resolve()
    if source_path == destination_path:
        raise ValueError("compressed store destination must differ from source")
    expected_bytes = geometry.num_slots * geometry.slot_size
    if source_path.stat().st_size != expected_bytes:
        raise ValueError("raw store size does not match compressed geometry")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[CompressedSlotRef] = []
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        delete=False,
    ) as output:
        temporary_path = Path(output.name)
        try:
            with source_path.open("rb") as source:
                output.truncate(expected_bytes)
                for slot_id in range(geometry.num_slots):
                    raw_slot = source.read(geometry.slot_size)
                    if len(raw_slot) != geometry.slot_size:
                        raise ValueError("raw store ended inside a slot")
                    encoded = encode_slot(
                        raw_slot,
                        slot_id=slot_id,
                        geometry=geometry,
                        codebooks=codebooks,
                    )
                    output.seek(slot_id * geometry.slot_size)
                    output.write(encoded.payload)
                    entries.append(
                        CompressedSlotRef(
                            slot_id=slot_id,
                            mode=encoded.mode,
                            file_offset=slot_id * geometry.slot_size,
                            stored_length=len(encoded.payload),
                            raw_hash=encoded.raw_hash,
                            encoded_hash=encoded.encoded_hash,
                        )
                    )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    index = CompressedStoreIndex(geometry, model_hash, codebooks, entries)
    index.write(index_path)
    logger.info(
        "[INDEX] built compressed store slots=%d raw_bytes=%d stored_bytes=%d",
        geometry.num_slots,
        expected_bytes,
        sum(entry.stored_length for entry in entries),
    )
    return index


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


def _pack_nibbles(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    packed: NDArray[np.uint8] = np.zeros((len(codes) + 1) // 2, dtype=np.uint8)
    packed[:] = codes[0::2]
    odd = codes[1::2]
    packed[: len(odd)] |= odd << 4
    return packed


def _pack_codes(codes: NDArray[np.uint8], symbol_bits: int) -> NDArray[np.uint8]:
    """Pack the selected 3-bit or historical 4-bit symbol stream."""
    if symbol_bits == 4:
        return _pack_nibbles(codes)
    if symbol_bits != 3:
        raise ValueError("symbol_bits must be 3 or 4")
    packed = np.zeros((len(codes) * symbol_bits + 7) // 8, dtype=np.uint8)
    for index, code in enumerate(codes):
        if symbol_bits == 3 and code >= (1 << symbol_bits) - 1:
            code = 7
        bit_offset = index * symbol_bits
        byte_offset = bit_offset // 8
        shift = bit_offset & 7
        value = int(code) << shift
        packed[byte_offset] |= value & 0xFF
        if shift > 5:
            packed[byte_offset + 1] |= (value >> 8) & 0xFF
    return packed


def _unpack_nibbles(symbols: NDArray[np.uint8], scalar_count: int) -> NDArray[np.uint8]:
    codes: NDArray[np.uint8] = np.empty(scalar_count, dtype=np.uint8)
    codes[0::2] = symbols & 0x0F
    codes[1::2] = symbols[: scalar_count // 2] >> 4
    return codes


def _unpack_codes(
    symbols: NDArray[np.uint8], scalar_count: int, symbol_bits: int
) -> NDArray[np.uint8]:
    """Unpack a dense little-endian 3-bit or 4-bit symbol stream."""
    if symbol_bits == 4:
        return _unpack_nibbles(symbols, scalar_count)
    if symbol_bits != 3:
        raise ValueError("symbol_bits must be 3 or 4")
    codes = np.empty(scalar_count, dtype=np.uint8)
    for index in range(scalar_count):
        bit_offset = index * symbol_bits
        byte_offset = bit_offset // 8
        shift = bit_offset & 7
        value = int(symbols[byte_offset])
        if shift > 5:
            value |= int(symbols[byte_offset + 1]) << 8
        codes[index] = (value >> shift) & 0x07
    return codes


def _select_symbol_bits(
    planes: list[tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]],
    geometry: CompressedStoreGeometry,
) -> tuple[int, int]:
    """Choose main and packed-escape widths for the smallest slot payload.

    The 3-bit main stream can use either a four-bit escape token (eight
    secondary entries plus a raw sentinel) or a three-bit token (seven
    secondary entries plus a raw sentinel).  The latter saves one bit per
    escape and only promotes the fifteenth codebook entry to a raw byte.
    """
    prefix_bytes = 4 * (
        (geometry.plane_scalars + geometry.tile_scalars - 1) // geometry.tile_scalars
        + 1
    )
    candidate_lengths: dict[tuple[int, int], int] = {}
    for bits, escape_bits in ((3, 3), (3, 4), (4, 4)):
        symbol_bytes = (geometry.plane_scalars * bits + 7) // 8
        cursor = IO_ALIGNMENT
        for low, _high, codes in planes:
            if bits == 3:
                token_count = int(np.count_nonzero(codes >= 7))
                raw_threshold = CODEBOOK_ENTRIES - (4 - escape_bits)
                raw_count = int(np.count_nonzero(codes >= raw_threshold))
                escape_bytes = (token_count * escape_bits + 7) // 8 + raw_count
                candidate_prefix_bytes = prefix_bytes * 2
            else:
                escape_bytes = int(np.count_nonzero(codes == CODEBOOK_ENTRIES))
                candidate_prefix_bytes = prefix_bytes
            payload = len(low) + symbol_bytes + candidate_prefix_bytes + escape_bytes
            # Plane records are contiguous inside one slot.  Only the final
            # slot extent is aligned because physical reads never target a
            # plane record independently.
            cursor += payload
        candidate_lengths[(bits, escape_bits)] = align_up(cursor)
    return min(candidate_lengths, key=lambda choice: candidate_lengths[choice])


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
    "build_compressed_store",
    "calibrate_codebooks",
    "default_online_codebooks",
    "decode_slot",
    "encode_slot",
]
