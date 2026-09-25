# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from daser.compression import (
    CompressedStoreGeometry,
    calibrate_codebooks,
    decode_slot,
    encode_slot,
    write_calibration_artifact,
)
from daser.compression.format import CODEBOOK_ENTRIES, KV_PLANES


def _geometry() -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=2,
        slot_size=2 * 2 * 128 * 2 * 8 * 2,
        block_tokens=128,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
    )


def _slot(geometry: CompressedStoreGeometry, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    high_values = np.asarray(
        [0x3E, 0x3F, 0x40, 0xBF, 0xBE, 0x41, 0x42, 0x43],
        dtype=np.uint8,
    )
    raw[1::2] = high_values[rng.integers(0, len(high_values), geometry.slot_size // 2)]
    return raw.tobytes()


def test_calibration_binds_model_identity_and_round_trips(tmp_path) -> None:
    geometry = _geometry()
    slots = [_slot(geometry, seed=1), _slot(geometry, seed=2)]

    artifact = calibrate_codebooks(
        slots,
        model_id="example/model",
        geometry=geometry,
        source_sha256="source-digest",
    )
    metadata = artifact.metadata()
    tables = np.frombuffer(artifact.codebooks, dtype=np.uint8).reshape(
        geometry.plane_count, CODEBOOK_ENTRIES
    )

    assert metadata["model_id"] == "example/model"
    assert metadata["plane_count"] == geometry.num_layers * KV_PLANES
    assert metadata["layout"].startswith("plane-major")
    assert metadata["source_sha256"] == "source-digest"
    assert tables.shape == (geometry.plane_count, CODEBOOK_ENTRIES)
    assert all(len(np.unique(row)) == CODEBOOK_ENTRIES for row in tables)

    encoded = encode_slot(
        slots[0], slot_id=0, geometry=geometry, codebooks=artifact.codebooks
    )
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=artifact.codebooks,
        )
        == slots[0]
    )

    write_calibration_artifact(artifact, tmp_path)
    assert (tmp_path / "codebooks.bin").read_bytes() == artifact.codebooks
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata
