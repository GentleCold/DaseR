# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    calibrate_codebooks,
    default_online_codebooks,
    encode_slot,
)
from daser.ops.compressed_kv import (
    FusedCompressedKVDecoder,
    FusedOnlineKVPacker,
    warm_fused_online_kv_packer,
)


def _geometry() -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=2,
        slot_size=2 * 2 * 128 * 4 * 127 * 2,
        block_tokens=128,
        num_layers=2,
        num_kv_heads=4,
        head_dim=127,
    )


def _slot(geometry: CompressedStoreGeometry, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
    raw[1::2] = common[
        rng.integers(0, len(common), geometry.slot_size // 2, dtype=np.uint8)
    ]
    raw[1::2][::997] = 0x7E
    return raw.tobytes()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_decoder_restores_mixed_slots_byte_exact() -> None:
    geometry = _geometry()
    raw_compressed = _slot(geometry, 1)
    raw_fallback = _slot(geometry, 2)
    codebooks = calibrate_codebooks([raw_compressed], geometry)
    encoded = encode_slot(
        raw_compressed,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
    )
    assert encoded.mode is SlotMode.COMPRESSED
    staging_bytes = encoded.payload + raw_fallback
    staging = torch.from_numpy(
        np.frombuffer(staging_bytes, dtype=np.uint8).copy()
    ).cuda()
    with torch.inference_mode():
        destination = torch.zeros(
            4,
            geometry.num_layers,
            2,
            geometry.block_tokens,
            geometry.num_kv_heads,
            geometry.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        decoder = FusedCompressedKVDecoder(
            kv_cache=destination,
            codebooks=codebooks,
            tile_scalars=geometry.tile_scalars,
            ring_depth=2,
            max_slots_per_buffer=2,
        )
    assert destination.is_inference()
    stream = torch.cuda.Stream()

    launches = [
        (0, [1, 3]),
        (1, [0, 2]),
        (0, [2, 0]),
    ]
    for buffer_index, block_ids in launches:
        restored = decoder.decode(
            staging=staging,
            staging_offsets=[0, len(encoded.payload)],
            block_ids=block_ids,
            modes=[1, 0],
            buffer_index=buffer_index,
            stream=stream,
        )
        stream.synchronize()
        assert restored == 2

    expected = [raw_fallback, raw_compressed, raw_compressed, raw_fallback]
    for block_id, raw in enumerate(expected):
        assert destination[block_id].view(torch.uint8).cpu().numpy().tobytes() == raw


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_online_packer_restores_escapes_and_raw_overflow_byte_exact() -> None:
    """Single-read packing preserves tile escapes beside raw fallback slots."""
    geometry = _geometry()
    raw_compressed = _slot(geometry, 21)
    raw_fallback_array = np.frombuffer(_slot(geometry, 22), dtype=np.uint8).copy()
    raw_fallback_array[1::2] = 0x7E
    raw_fallback = raw_fallback_array.tobytes()
    codebooks = default_online_codebooks(geometry)
    source = (
        torch.from_numpy(
            np.frombuffer(raw_compressed + raw_fallback, dtype=np.uint16).copy()
        )
        .to("cuda")
        .view(torch.bfloat16)
        .reshape(
            geometry.num_slots,
            geometry.num_layers,
            2,
            geometry.block_tokens,
            geometry.num_kv_heads,
            geometry.head_dim,
        )
    )
    warm_fused_online_kv_packer(
        source,
        max_slots_per_buffer=geometry.num_slots,
        tile_scalars=geometry.tile_scalars,
    )
    packer = FusedOnlineKVPacker(
        kv_cache=source,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        max_slots_per_buffer=geometry.num_slots,
    )
    staging = torch.empty(
        geometry.num_slots * geometry.slot_size,
        dtype=torch.uint8,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    packed = packer.pack_into(
        staging=staging,
        block_ids=[0, 1],
        logical_slots=[0, 1],
        slot_stride=geometry.slot_size,
        stream=stream,
    )
    stream.synchronize()

    assert [slot.mode for slot in packed] == [SlotMode.COMPRESSED, SlotMode.RAW]
    destination = torch.zeros(
        4,
        geometry.num_layers,
        2,
        geometry.block_tokens,
        geometry.num_kv_heads,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    decoder = FusedCompressedKVDecoder(
        kv_cache=destination,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        ring_depth=1,
        max_slots_per_buffer=geometry.num_slots,
    )
    decoder.decode(
        staging=staging,
        staging_offsets=[slot.source_offset for slot in packed],
        block_ids=[1, 3],
        modes=[int(slot.mode) for slot in packed],
        buffer_index=0,
        stream=stream,
    )
    stream.synchronize()

    assert destination[1].view(torch.uint8).cpu().numpy().tobytes() == raw_compressed
    assert destination[3].view(torch.uint8).cpu().numpy().tobytes() == raw_fallback
