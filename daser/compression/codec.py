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
    """Return a deterministic model-independent codebook for online stores.

    Online serving cannot pause for an activation calibration pass.  The
    high-byte values below cover the two dense BF16 exponent/sign
    neighbourhoods observed in normalized attention KV values (the positive
    and negative bands around ``1.0``). Values outside the table remain
    lossless escapes. The same table is derived for every plane so the server
    can publish it once at startup without scanning live KV memory.

    Args:
        geometry: KV geometry whose plane count determines the output size.

    Returns:
        Plane-major 15-entry codebook bytes.

    Async/thread-safety:
        Pure startup computation; safe to call from any thread.
    """
    values = bytes((63, 191, 62, 190, 64, 192, 61, 189, 60, 188, 59, 187, 58, 186, 57))
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
    stream. Each plane record and the complete returned payload are 4 KiB
    aligned for independent validation and O_DIRECT slot reads.

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

    records: list[tuple[PlaneDescriptor, bytes, bytes, bytes, bytes]] = []
    cursor = IO_ALIGNMENT
    for plane in range(geometry.plane_count):
        plane_start = plane * geometry.plane_bytes
        plane_bytes = raw[plane_start : plane_start + geometry.plane_bytes]
        low = np.ascontiguousarray(plane_bytes[0::2])
        high = np.ascontiguousarray(plane_bytes[1::2])
        codes = _encode_high_bytes(high, tables[plane])
        symbols = _pack_nibbles(codes)
        escape_mask = codes == CODEBOOK_ENTRIES
        escapes = np.ascontiguousarray(high[escape_mask])
        prefixes = _escape_prefixes(escape_mask, geometry.tile_scalars)

        low_bytes = low.tobytes()
        symbol_bytes = symbols.tobytes()
        prefix_bytes = prefixes.astype("<u4", copy=False).tobytes()
        escape_bytes = escapes.tobytes()
        record_offset = cursor
        low_offset = record_offset
        symbol_offset = low_offset + len(low_bytes)
        prefix_offset = symbol_offset + len(symbol_bytes)
        escape_offset = prefix_offset + len(prefix_bytes)
        record_length = align_up(
            escape_offset + len(escape_bytes) - record_offset,
            IO_ALIGNMENT,
        )
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
            escape_count=len(escape_bytes),
        )
        records.append(
            (descriptor, low_bytes, symbol_bytes, prefix_bytes, escape_bytes)
        )
        cursor += record_length

    stored_length = cursor
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
        symbol_length = (scalar_count + 1) // 2
        symbols = np.frombuffer(
            source[descriptor.symbol_offset : descriptor.symbol_offset + symbol_length],
            dtype=np.uint8,
        )
        codes = _unpack_nibbles(symbols, scalar_count)
        escapes = np.frombuffer(
            source[
                descriptor.escape_offset : descriptor.escape_offset
                + descriptor.escape_count
            ],
            dtype=np.uint8,
        )
        escape_mask = codes == CODEBOOK_ENTRIES
        if int(np.count_nonzero(escape_mask)) != descriptor.escape_count:
            raise ValueError("compressed plane escape count mismatch")
        high: NDArray[np.uint8] = np.empty(scalar_count, dtype=np.uint8)
        coded_mask = ~escape_mask
        high[coded_mask] = tables[plane, codes[coded_mask]]
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


def _unpack_nibbles(symbols: NDArray[np.uint8], scalar_count: int) -> NDArray[np.uint8]:
    codes: NDArray[np.uint8] = np.empty(scalar_count, dtype=np.uint8)
    codes[0::2] = symbols & 0x0F
    codes[1::2] = symbols[: scalar_count // 2] >> 4
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
    "build_compressed_store",
    "calibrate_codebooks",
    "default_online_codebooks",
    "decode_slot",
    "encode_slot",
]
