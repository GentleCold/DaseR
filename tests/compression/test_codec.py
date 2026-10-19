# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import random

import numpy as np
import pytest

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import (
    CODEBOOK_ENTRIES,
    FORMAT_VERSION,
    IO_ALIGNMENT,
    ONLINE_TILE_SCALARS,
    SlotHeader,
    digest_bytes,
    slot_header_bytes,
)


def _geometry(num_slots: int = 2) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=num_slots,
        slot_size=2 * 2 * 128 * 4 * 127 * 2,
        block_tokens=128,
        num_layers=2,
        num_kv_heads=4,
        head_dim=127,
        tile_scalars=1024,
    )


def _slot(
    geometry: CompressedStoreGeometry,
    *,
    seed: int,
    escaped_high_byte: int | None = None,
) -> bytes:
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
    raw[1::2] = common[
        rng.integers(0, len(common), geometry.slot_size // 2, dtype=np.uint8)
    ]
    if escaped_high_byte is not None:
        raw[1::2][::997] = escaped_high_byte
    return raw.tobytes()


def test_reference_codec_is_byte_exact_with_escapes_and_tail_tile() -> None:
    geometry = _geometry()
    evaluation = _slot(geometry, seed=2, escaped_high_byte=0x7E)
    codebooks = default_online_codebooks(geometry)

    encoded = encode_slot(
        evaluation,
        slot_id=1,
        geometry=geometry,
        codebooks=codebooks,
    )

    assert encoded.mode is SlotMode.COMPRESSED
    assert len(encoded.payload) < geometry.slot_size
    assert len(encoded.payload) % IO_ALIGNMENT == 0
    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=1,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert header.descriptors[0].scalar_count % geometry.tile_scalars == 512
    # Only the complete slot is an O_DIRECT transfer. Plane records are
    # intentionally packed back-to-back so their internal padding is not
    # counted as stored KV bytes.
    assert any(
        descriptor.record_offset % IO_ALIGNMENT for descriptor in header.descriptors[1:]
    )
    assert any(
        descriptor.record_length % IO_ALIGNMENT for descriptor in header.descriptors
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=1,
            geometry=geometry,
            codebooks=codebooks,
        )
        == evaluation
    )


def test_three_bit_codec_treats_unrepresentable_symbols_as_escapes() -> None:
    """Three-bit packing must preserve codebook entries 7 through 14."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    table = np.frombuffer(codebooks, dtype=np.uint8)[:CODEBOOK_ENTRIES]
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = 17
    high = np.full(geometry.slot_size // 2, table[0], dtype=np.uint8)
    high[::997] = table[10]
    raw[1::2] = high

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )

    assert encoded.mode is SlotMode.COMPRESSED
    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=0,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_three_bit_escape_stream_round_trips_the_eighth_secondary_entry() -> None:
    """The narrow escape stream promotes codebook entry fourteen to raw bytes."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    table = np.frombuffer(codebooks, dtype=np.uint8)[:CODEBOOK_ENTRIES]
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = 23
    high = np.full(geometry.slot_size // 2, table[0], dtype=np.uint8)
    high[::997] = table[14]
    raw[1::2] = high

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )

    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=0,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_online_codebook_is_generic_for_all_geometries() -> None:
    """The startup default is model-independent and plane-major."""
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=36 * 2 * 128 * 8 * 128 * 2,
        block_tokens=128,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
    )

    tables = np.frombuffer(default_online_codebooks(geometry), dtype=np.uint8).reshape(
        geometry.plane_count, CODEBOOK_ENTRIES
    )

    assert tables.shape == (72, CODEBOOK_ENTRIES)
    assert all(len(np.unique(row)) == CODEBOOK_ENTRIES for row in tables)
    assert np.all(tables == tables[0])
    assert tables[0].tolist() == [
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
    ]


def test_incompressible_slot_uses_explicit_raw_mode() -> None:
    geometry = _geometry(num_slots=1)
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8).tobytes()
    codebooks = default_online_codebooks(geometry)

    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)

    assert encoded.mode is SlotMode.RAW
    assert encoded.payload == raw
    assert (
        decode_slot(
            encoded.payload,
            mode=SlotMode.RAW,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw
    )


def test_slot_header_rejects_corrupt_descriptor() -> None:
    geometry = _geometry(num_slots=1)
    slot = _slot(geometry, seed=30)
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(slot, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    payload = bytearray(encoded.payload)
    payload[128:136] = b"\xff" * 8

    with pytest.raises(ValueError):
        decode_slot(
            payload,
            mode=SlotMode.COMPRESSED,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )


def test_unknown_format_version_is_rejected() -> None:
    """Only version-1 records parse; other header versions are refused."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    raw = _slot(geometry, seed=45)
    payload = bytearray(
        encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks).payload
    )
    assert SlotHeader.parse(payload).format_version == FORMAT_VERSION
    payload[8] = 2
    with pytest.raises(ValueError, match="version"):
        SlotHeader.parse(payload, expected_geometry=geometry)


def _layer_geometry(num_layers: int) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=1,
        slot_size=num_layers * 2 * 16 * 8 * 128 * 2,
        block_tokens=16,
        num_layers=num_layers,
        num_kv_heads=8,
        head_dim=128,
        tile_scalars=ONLINE_TILE_SCALARS,
    )


@pytest.mark.parametrize(
    ("num_layers", "header_pages"), [(41, 1), (42, 2), (48, 2), (64, 2)]
)
def test_deep_models_round_trip_with_multi_page_headers(
    num_layers: int, header_pages: int
) -> None:
    # One page holds 82 plane descriptors; deeper models spill to more pages.
    geometry = _layer_geometry(num_layers)
    assert geometry.header_bytes == slot_header_bytes(2 * num_layers)
    assert geometry.header_bytes == header_pages * IO_ALIGNMENT
    codebooks = default_online_codebooks(geometry)
    slot = _slot(geometry, seed=num_layers)
    encoded = encode_slot(slot, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    header = SlotHeader.parse(encoded.payload, expected_geometry=geometry)
    assert header.descriptors[0].record_offset == geometry.header_bytes
    assert len(header.descriptors) == 2 * num_layers
    decoded = decode_slot(
        encoded.payload,
        mode=encoded.mode,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
    )
    assert decoded == slot


def test_parse_rejects_a_header_length_that_does_not_fit_the_planes() -> None:
    geometry = _layer_geometry(48)
    encoded = encode_slot(
        _slot(geometry, seed=1),
        slot_id=0,
        geometry=geometry,
        codebooks=default_online_codebooks(geometry),
    )
    payload = bytearray(encoded.payload)
    # header_bytes is the third little-endian u32 of the fixed header.
    payload[12:16] = IO_ALIGNMENT.to_bytes(4, "little")
    with pytest.raises(ValueError, match="header length"):
        SlotHeader.parse(bytes(payload), expected_geometry=geometry)


@pytest.mark.parametrize(
    ("corruption", "with_geometry"),
    [("zero_tile", False), ("escape_bounds", False), ("escape_bounds", True)],
)
def test_parse_validates_bounds_before_using_untrusted_header_fields(
    corruption: str, with_geometry: bool
) -> None:
    """Malformed geometry and streams raise ValueError before division/indexing."""
    geometry = _geometry()
    encoded = encode_slot(
        _slot(geometry, seed=3),
        slot_id=0,
        geometry=geometry,
        codebooks=default_online_codebooks(geometry),
    )
    header = SlotHeader.parse(encoded.payload, expected_geometry=geometry)
    if corruption == "zero_tile":
        header = replace(header, tile_scalars=0)
    else:
        descriptor = replace(
            header.descriptors[0],
            record_length=len(encoded.payload) + IO_ALIGNMENT,
            escape_offset=len(encoded.payload) + 1,
            escape_count=1,
        )
        header = replace(header, descriptors=(descriptor, *header.descriptors[1:]))
    packed_header = header.pack()
    payload = packed_header + encoded.payload[len(packed_header) :]
    with pytest.raises(ValueError):
        SlotHeader.parse(payload, expected_geometry=geometry if with_geometry else None)


def test_seeded_header_mutations_preserve_parser_error_contract() -> None:
    """Mutated fixed fields and descriptors either validate or raise ValueError."""
    geometry = _layer_geometry(1)
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(
        _slot(geometry, seed=11), slot_id=0, geometry=geometry, codebooks=codebooks
    )
    rng = random.Random(3019)
    for _ in range(1000):
        payload = bytearray(encoded.payload)
        # For two planes the first 256 bytes cover the fixed header, all
        # descriptors and layout flags; padding mutations add little coverage.
        position = rng.randrange(256)
        payload[position] ^= 1 << rng.randrange(8)
        try:
            header = SlotHeader.parse(
                payload,
                expected_geometry=geometry,
                expected_slot_id=0,
                expected_codebook_hash=digest_bytes(codebooks),
            )
        except ValueError:
            continue
        assert header.raw_length == geometry.slot_size
        assert header.tile_scalars == geometry.tile_scalars
        assert len(header.descriptors) == geometry.plane_count
        assert all(
            0
            < item.record_offset
            < item.record_offset + item.record_length
            <= header.stored_length
            <= len(payload)
            for item in header.descriptors
        )
