# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from daser.compression import (
    CODEC_ID,
    CompressedStoreGeometry,
    SlotMode,
    SlotPublication,
    codec_identity_digest,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import (
    CODEBOOK_ENTRIES,
    IO_ALIGNMENT,
    SlotHeader,
    digest_bytes,
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


def test_slot_publication_round_trip_validates_stored_and_restored_bytes() -> None:
    """The immutable contract rejects stale bytes before and after decode."""
    geometry = _geometry()
    raw = _slot(geometry, seed=7, escaped_high_byte=0x7E)
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(
        raw,
        slot_id=1,
        geometry=geometry,
        codebooks=codebooks,
    )

    publication = SlotPublication.from_encoded(
        slot_id=encoded.slot_id,
        mode=encoded.mode,
        raw_length=geometry.slot_size,
        stored_payload=encoded.payload,
        source_digest=encoded.raw_hash,
        codebook_hash=digest_bytes(codebooks),
        tile_scalars=geometry.tile_scalars,
    )
    reloaded = SlotPublication.from_payload(publication.to_payload())
    expected_codec_digest = codec_identity_digest(
        codebook_hash=digest_bytes(codebooks),
        tile_scalars=geometry.tile_scalars,
    )

    reloaded.validate_reload(
        encoded.payload,
        expected_slot_id=1,
        expected_codec_digest=expected_codec_digest,
    )
    reloaded.validate_restored(raw)
    assert reloaded.codec_id == CODEC_ID

    with pytest.raises(ValueError, match="stored digest"):
        reloaded.validate_reload(
            encoded.payload[:-1] + bytes([encoded.payload[-1] ^ 1]),
            expected_slot_id=1,
            expected_codec_digest=expected_codec_digest,
        )
    with pytest.raises(ValueError, match="restored digest"):
        reloaded.validate_restored(raw[:-1] + bytes([raw[-1] ^ 1]))


def test_slot_publication_rejects_identity_and_incomplete_payload() -> None:
    """A restart cannot accept a different codec or a partial carrier."""
    geometry = _geometry(num_slots=1)
    raw = _slot(geometry, seed=9, escaped_high_byte=0x7E)
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(
        raw,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
    )
    payload = SlotPublication.from_encoded(
        slot_id=0,
        mode=encoded.mode,
        raw_length=geometry.slot_size,
        stored_payload=encoded.payload,
        source_digest=encoded.raw_hash,
        codebook_hash=digest_bytes(codebooks),
        tile_scalars=geometry.tile_scalars,
    ).to_payload()

    with pytest.raises(ValueError, match="codec identity"):
        SlotPublication.from_payload(
            {**payload, "codec_digest": bytes(32)}
        ).validate_reload(
            encoded.payload,
            expected_slot_id=0,
            expected_codec_digest=codec_identity_digest(
                codebook_hash=digest_bytes(codebooks),
                tile_scalars=geometry.tile_scalars,
            ),
        )
    incomplete = dict(payload)
    del incomplete["stored_digest"]
    with pytest.raises(ValueError, match="invalid slot publication payload"):
        SlotPublication.from_payload(incomplete)


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
    publication = SlotPublication.from_encoded(
        slot_id=encoded.slot_id,
        mode=encoded.mode,
        raw_length=geometry.slot_size,
        stored_payload=encoded.payload,
        source_digest=encoded.raw_hash,
        codebook_hash=digest_bytes(codebooks),
        tile_scalars=geometry.tile_scalars,
    )
    publication.validate_reload(
        encoded.payload,
        expected_slot_id=0,
        expected_codec_digest=codec_identity_digest(
            codebook_hash=digest_bytes(codebooks),
            tile_scalars=geometry.tile_scalars,
        ),
    )
    publication.validate_restored(raw)
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
