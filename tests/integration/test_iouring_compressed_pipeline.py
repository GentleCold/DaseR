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
    build_compressed_store,
    calibrate_codebooks,
)
from daser.compression.format import IO_ALIGNMENT, digest_bytes
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
    codebooks = calibrate_codebooks([raw_slots[0]], geometry)
    raw_path = tmp_path / "raw.store"
    store_path = tmp_path / "daser.store"
    index_path = tmp_path / "daser.compressed.index"
    raw_path.write_bytes(b"".join(raw_slots))
    index = build_compressed_store(
        raw_path,
        store_path,
        index_path,
        geometry=geometry,
        model_hash=digest_bytes(b"integration-model"),
        codebooks=codebooks,
    )
    refs = index.resolve_slots(0, geometry.num_slots)
    transferred_bytes = sum(ref.stored_length for ref in refs)
    pinned_owner, pinned = _aligned_pinned_view(transferred_bytes)
    offsets: list[int] = []
    cursor = 0
    fd = os.open(store_path, os.O_RDONLY | os.O_DIRECT)
    ring = NativeIOUring(entries=8)
    try:
        for ref in refs:
            offsets.append(cursor)
            target = pinned[cursor : cursor + ref.stored_length]
            assert ring.read_into(fd, ref.file_offset, target) == ref.stored_length
            cursor += ref.stored_length
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
        codebooks=index.codebooks,
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
        modes=[int(ref.mode) for ref in refs],
        buffer_index=0,
        stream=stream,
    )
    stream.synchronize()

    assert restored == geometry.num_slots
    assert destination[1].view(torch.uint8).cpu().numpy().tobytes() == raw_slots[0]
    assert destination[3].view(torch.uint8).cpu().numpy().tobytes() == raw_slots[1]
    assert pinned_owner is not None
