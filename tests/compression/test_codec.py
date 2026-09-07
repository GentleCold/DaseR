# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from daser.compression import (
    CompressedStoreGeometry,
    CompressedStoreIndex,
    SlotMode,
    build_compressed_store,
    calibrate_codebooks,
    decode_slot,
    encode_slot,
)
from daser.compression.codec import default_online_codebooks
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
    calibration = _slot(geometry, seed=1)
    evaluation = _slot(geometry, seed=2, escaped_high_byte=0x7E)
    codebooks = calibrate_codebooks([calibration], geometry)

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
    assert header.symbol_bits == 3
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
    assert header.symbol_bits == 3
    assert header.escape_packed
    assert header.escape_symbol_bits == 3
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


def test_qwen3_online_codebook_uses_plane_specific_primary_values() -> None:
    """The production Qwen3 geometry gets distinct K/V primary palettes."""
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
    assert tables[0, :7].tolist() == [191, 63, 62, 190, 64, 192, 61]
    assert tables[1, :7].tolist() == [188, 60, 187, 59, 61, 189, 186]
    assert tables[2, :7].tolist() == [191, 63, 190, 62, 64, 192, 189]
    assert tables[3, :7].tolist() == [61, 189, 60, 188, 187, 59, 190]
    assert tables[0, 7:14].tolist() == [189, 188, 60, 193, 65, 187, 67]
    assert tables[1, 7:14].tolist() == [58, 185, 57, 62, 184, 190, 56]
    assert tables[0, :7].tolist() != tables[1, :7].tolist()
    assert np.all(tables[:, -1] == 0)


def test_incompressible_slot_uses_explicit_raw_mode() -> None:
    geometry = _geometry(num_slots=1)
    calibration = _slot(geometry, seed=3)
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8).tobytes()
    codebooks = calibrate_codebooks([calibration], geometry)

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


def test_store_builder_preserves_envelopes_and_round_trips_index(
    tmp_path: Path,
) -> None:
    geometry = _geometry()
    slots = [_slot(geometry, seed=index + 10) for index in range(2)]
    codebooks = calibrate_codebooks([slots[0]], geometry)
    raw_path = tmp_path / "raw.store"
    compressed_path = tmp_path / "daser.store"
    index_path = tmp_path / "daser.compressed.index"
    raw_path.write_bytes(b"".join(slots))

    built = build_compressed_store(
        raw_path,
        compressed_path,
        index_path,
        geometry=geometry,
        model_hash=digest_bytes(b"model-config"),
        codebooks=codebooks,
    )
    loaded = CompressedStoreIndex.load(
        index_path,
        expected_geometry=geometry,
        expected_model_hash=digest_bytes(b"model-config"),
    )

    assert compressed_path.stat().st_size == geometry.num_slots * geometry.slot_size
    assert loaded.codebooks == built.codebooks
    with compressed_path.open("rb") as handle:
        for expected_slot, ref in zip(slots, loaded.resolve_slots(0, 2), strict=True):
            assert ref.file_offset == ref.slot_id * geometry.slot_size
            handle.seek(ref.file_offset)
            stored = handle.read(ref.stored_length)
            assert digest_bytes(stored) == ref.encoded_hash
            assert (
                decode_slot(
                    stored,
                    mode=ref.mode,
                    slot_id=ref.slot_id,
                    geometry=geometry,
                    codebooks=loaded.codebooks,
                )
                == expected_slot
            )


@pytest.mark.parametrize("corruption", ["truncated", "codebook"])
def test_side_index_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    geometry = _geometry(num_slots=1)
    slot = _slot(geometry, seed=20)
    codebooks = calibrate_codebooks([slot], geometry)
    raw_path = tmp_path / "raw.store"
    compressed_path = tmp_path / "daser.store"
    index_path = tmp_path / "daser.compressed.index"
    raw_path.write_bytes(slot)
    build_compressed_store(
        raw_path,
        compressed_path,
        index_path,
        geometry=geometry,
        model_hash=digest_bytes(b"model-config"),
        codebooks=codebooks,
    )
    payload = bytearray(index_path.read_bytes())
    if corruption == "truncated":
        del payload[-1]
    else:
        payload[IO_ALIGNMENT] ^= 0xFF
    index_path.write_bytes(payload)

    with pytest.raises(ValueError):
        CompressedStoreIndex.load(index_path)


def test_slot_header_rejects_corrupt_descriptor() -> None:
    geometry = _geometry(num_slots=1)
    slot = _slot(geometry, seed=30)
    codebooks = calibrate_codebooks([slot], geometry)
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
