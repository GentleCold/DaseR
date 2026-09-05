# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from daser.compression import CompressedStoreIndex, decode_slot
from daser.compression.__main__ import main


def test_cli_builds_byte_exact_read_only_store(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 8,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    slot_a = bytes(range(256)) * 16
    slot_b = bytes(reversed(range(256))) * 16
    raw_store = raw_dir / "daser.store"
    raw_store.write_bytes(slot_a + slot_b)
    raw_index = raw_dir / "daser.index"
    raw_index.write_bytes(b"control-index")
    output_dir = tmp_path / "compressed"

    result = main(
        [
            "--raw-store",
            str(raw_store),
            "--raw-index",
            str(raw_index),
            "--output-dir",
            str(output_dir),
            "--model-path",
            str(model_path),
            "--block-tokens",
            "128",
            "--calibration-slot",
            "0",
        ]
    )

    assert result == 0
    assert (output_dir / "daser.index").read_bytes() == b"control-index"
    assert (output_dir / "daser.store").stat().st_size == len(slot_a + slot_b)
    index = CompressedStoreIndex.load(output_dir / "daser.compressed.index")
    with (output_dir / "daser.store").open("rb") as handle:
        for expected, ref in zip(
            (slot_a, slot_b), index.resolve_slots(0, 2), strict=True
        ):
            handle.seek(ref.file_offset)
            encoded = handle.read(ref.stored_length)
            assert (
                decode_slot(
                    encoded,
                    mode=ref.mode,
                    slot_id=ref.slot_id,
                    geometry=index.geometry,
                    codebooks=index.codebooks,
                )
                == expected
            )
