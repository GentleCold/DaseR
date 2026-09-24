# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from pathlib import Path

import cupy
import numpy as np
import pytest
import torch

from daser.compression import (
    CompressedStoreGeometry,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import IO_ALIGNMENT
from daser.ops.compressed_kv import FusedCompressedKVDecoder
from daser.transfer.iouring.native import NativeIOUring


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


def _aligned_pinned_view(nbytes: int) -> tuple[object, memoryview]:
    allocation = cupy.cuda.alloc_pinned_memory(nbytes + IO_ALIGNMENT - 1)
    view = memoryview(allocation).cast("B")
    address = ctypes.addressof(ctypes.c_char.from_buffer(view))
    offset = (-address) % IO_ALIGNMENT
    aligned = view[offset : offset + nbytes]
    assert ctypes.addressof(ctypes.c_char.from_buffer(aligned)) % IO_ALIGNMENT == 0
    return allocation, aligned


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_iouring_h2d_fused_restore_is_byte_exact(tmp_path: Path) -> None:
    geometry = _geometry()
    raw_slots = [_slot(geometry, seed) for seed in (11, 12)]
    codebooks = default_online_codebooks(geometry)
    encoded = [
        encode_slot(raw, slot_id=slot_id, geometry=geometry, codebooks=codebooks)
        for slot_id, raw in enumerate(raw_slots)
    ]
    # Each record starts at its fixed slot envelope, as online stores place it.
    store_path = tmp_path / "daser.store"
    with store_path.open("wb") as handle:
        handle.truncate(geometry.num_slots * geometry.slot_size)
        for slot_id, slot in enumerate(encoded):
            handle.seek(slot_id * geometry.slot_size)
            handle.write(slot.payload)
    transferred_bytes = sum(len(slot.payload) for slot in encoded)
    pinned_owner, pinned = _aligned_pinned_view(transferred_bytes)
    offsets: list[int] = []
    cursor = 0
    fd = os.open(store_path, os.O_RDONLY | os.O_DIRECT)
    ring = NativeIOUring(entries=8)
    try:
        for slot_id, slot in enumerate(encoded):
            nbytes = len(slot.payload)
            offsets.append(cursor)
            target = pinned[cursor : cursor + nbytes]
            file_offset = slot_id * geometry.slot_size
            assert ring.read_into(fd, file_offset, target) == nbytes
            cursor += nbytes
    finally:
        ring.close()
        os.close(fd)
    assert cursor == transferred_bytes
    assert transferred_bytes < geometry.num_slots * geometry.slot_size

    staging = torch.empty(transferred_bytes, dtype=torch.uint8, device="cuda")
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
    stream = torch.cuda.Stream()
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    host_array = np.frombuffer(pinned, dtype=np.uint8, count=transferred_bytes)
    with external_stream:
        cupy.asarray(staging).set(host_array, stream=external_stream)
    restored = decoder.decode(
        staging=staging,
        staging_offsets=offsets,
        block_ids=[1, 3],
        modes=[int(slot.mode) for slot in encoded],
        buffer_index=0,
        stream=stream,
    )
    stream.synchronize()

    assert restored == geometry.num_slots
    assert destination[1].view(torch.uint8).cpu().numpy().tobytes() == raw_slots[0]
    assert destination[3].view(torch.uint8).cpu().numpy().tobytes() == raw_slots[1]
    assert pinned_owner is not None
