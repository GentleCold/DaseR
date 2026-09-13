# SPDX-License-Identifier: Apache-2.0
"""Public structural contracts for progressive packed payload copies."""

from dataclasses import replace

import numpy as np
import pytest

from daser.compression import (
    CompressedStoreGeometry,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import IO_ALIGNMENT, SlotHeader, layer_group_copy_ends


@pytest.fixture(scope="module")
def payload() -> bytes:
    """Provide real encoded five-layer planes with unaligned exception tails."""
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=5 * 2 * 128 * 8 * 128 * 2,
        block_tokens=128,
        num_layers=5,
        num_kv_heads=8,
        head_dim=128,
        tile_scalars=256,
    )
    books = default_online_codebooks(geometry)
    values = np.empty((geometry.plane_count, geometry.plane_scalars, 2), dtype=np.uint8)
    values[:, :, 0] = 37
    for plane in range(geometry.plane_count):
        values[plane, :, 1] = books[plane * 15]
        values[plane, ::997, 1] = 0x7E
    return encode_slot(
        values.tobytes(), slot_id=0, geometry=geometry, codebooks=books
    ).payload


@pytest.mark.parametrize("group_layers", [1, 2, 5, 8])
def test_groups_copy_every_required_plane_without_repacking(
    payload: bytes,
    group_layers: int,
) -> None:
    header = SlotHeader.parse(payload)
    ends = layer_group_copy_ends(
        payload[:IO_ALIGNMENT],
        stored_length=len(payload),
        num_layers=5,
        layers_per_group=group_layers,
    )
    target = bytearray(len(payload))
    begin = 0
    for group, end in enumerate(ends):
        target[begin:end] = payload[begin:end]
        visible_layers = min(5, (group + 1) * group_layers)
        for descriptor in header.descriptors[: visible_layers * 2]:
            record_end = descriptor.record_offset + descriptor.record_length
            assert record_end <= end
            assert (
                target[descriptor.record_offset : record_end]
                == payload[descriptor.record_offset : record_end]
            )
        assert end % 128 == 0
        begin = end
    assert target == payload
    assert len(ends) == (5 + group_layers - 1) // group_layers


@pytest.mark.parametrize(
    "failure", ["gap", "low_overrun", "wrong_layer", "late_overrun"]
)
def test_invalid_descriptor_anywhere_in_group_is_rejected(
    payload: bytes, failure: str
) -> None:
    header = SlotHeader.parse(payload)
    descriptors = list(header.descriptors)
    # Corrupt a K plane,which is not a group endpoint. Validating only each
    # group's final V descriptor would incorrectly accept these failures.
    original = descriptors[2]
    changes = {
        "gap": {"record_offset": original.record_offset + 1},
        "low_overrun": {"low_offset": len(payload)},
        "wrong_layer": {"layer": 0},
        "late_overrun": {"record_length": len(payload)},
    }[failure]
    descriptors[2] = replace(original, **changes)
    page = replace(header, descriptors=tuple(descriptors)).pack()
    with pytest.raises(ValueError, match="descriptor"):
        layer_group_copy_ends(
            page, stored_length=len(payload), num_layers=5, layers_per_group=2
        )


@pytest.mark.parametrize("failure", ["truncated", "length", "layers", "group"])
def test_index_and_complete_header_are_required(payload: bytes, failure: str) -> None:
    with pytest.raises(ValueError):
        layer_group_copy_ends(
            payload[: IO_ALIGNMENT - 1]
            if failure == "truncated"
            else payload[:IO_ALIGNMENT],
            stored_length=len(payload) + (4096 if failure == "length" else 0),
            num_layers=6 if failure == "layers" else 5,
            layers_per_group=0 if failure == "group" else 2,
        )
