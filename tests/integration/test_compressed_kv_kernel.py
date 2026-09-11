# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
import torch

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    calibrate_codebooks,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.format import SlotHeader, digest_bytes
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
@pytest.mark.parametrize("tile_scalars", [512, 1024, 2048])
def test_fused_decoder_restores_mixed_slots_byte_exact(tile_scalars: int) -> None:
    geometry = replace(_geometry(), tile_scalars=tile_scalars)
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
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("tile_scalars", [256, 1024])
@pytest.mark.parametrize("segmented", [False, True])
def test_fused_decoder_fanout_preserves_mixed_sources_and_ring_reuse(
    monkeypatch: pytest.MonkeyPatch,
    online: bool,
    tile_scalars: int,
    segmented: bool,
) -> None:
    """Interleaved aliases restore exact KV to distinct targets without new JIT."""
    import tilelang

    geometry = replace(_geometry(), tile_scalars=tile_scalars)
    if tile_scalars == 256:
        # Leave a partial tile and fewer than four tiles in the final CTA.
        geometry = CompressedStoreGeometry(
            num_slots=2,
            slot_size=8 * 2 * 128 * 61 * 2,
            block_tokens=128,
            num_layers=8,
            num_kv_heads=1,
            head_dim=61,
            tile_scalars=tile_scalars,
        )
    packed_raw, fallback_raw = _slot(geometry, 51), _slot(geometry, 52)
    codebooks = calibrate_codebooks([packed_raw], geometry)
    if online:
        source = (
            torch.from_numpy(np.frombuffer(packed_raw, dtype=np.uint16).copy())
            .view(torch.bfloat16)
            .reshape(
                1,
                geometry.num_layers,
                2,
                geometry.block_tokens,
                geometry.num_kv_heads,
                geometry.head_dim,
            )
            .cuda()
        )
        warm_fused_online_kv_packer(source, 1, tile_scalars=tile_scalars)
        packer = FusedOnlineKVPacker(
            kv_cache=source,
            codebooks=codebooks,
            tile_scalars=geometry.tile_scalars,
            max_slots_per_buffer=1,
        )
        encoded_staging = torch.empty(
            geometry.slot_size, dtype=torch.uint8, device="cuda"
        )
        encode_stream = torch.cuda.Stream()
        records = packer.pack_into(
            staging=encoded_staging,
            block_ids=[0],
            logical_slots=[0],
            slot_stride=geometry.slot_size,
            stream=encode_stream,
        )
        encode_stream.synchronize()
        assert records[0].mode is SlotMode.COMPRESSED
        payload = encoded_staging[: records[0].stored_length].cpu().numpy().tobytes()
    else:
        encoded = encode_slot(
            packed_raw, slot_id=0, geometry=geometry, codebooks=codebooks
        )
        assert encoded.mode is SlotMode.COMPRESSED
        payload = encoded.payload
    staging = torch.from_numpy(
        np.frombuffer(payload + fallback_raw, dtype=np.uint8).copy()
    ).cuda()
    destination = torch.zeros(
        7,
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
        max_slots_per_buffer=6,
        online=online,
    )

    def no_compile(*args: object, **kwargs: object) -> None:
        raise AssertionError("fanout changed the compile geometry")

    monkeypatch.setattr(tilelang, "compile", no_compile)
    stream = torch.cuda.Stream()
    blocks = [6, 2, 5, 1, 3, 0]
    for ring in (0, 1, 0):
        if segmented:
            destination.zero_()
            torch.cuda.synchronize()
            plan = decoder.prepare(
                staging=staging,
                staging_offsets=[len(payload), 0] * 3,
                block_ids=blocks,
                modes=[0, 1] * 3,
                buffer_index=ring,
                stream=stream,
            )
            assert plan is not None
            assert plan.remaining_sources == 2
            assert plan.destination_count == 6
            assert plan.submit_next(1) == 1
            stream.synchronize()
            # The first unique source is raw and fans out to three disjoint
            # targets. The packed source must not be launched by preparation
            # or by this first segment, even across metadata ring reuse.
            for index, block in enumerate(blocks):
                if index % 2:
                    assert not torch.count_nonzero(destination[block])
                else:
                    assert (
                        destination[block].view(torch.uint8).cpu().numpy().tobytes()
                        == fallback_raw
                    )
            assert plan.submit_next(8) == 1
            assert plan.remaining_sources == 0
            assert plan.submit_next(8) == 0
            restored = plan.destination_count
        else:
            restored = decoder.decode(
                staging=staging,
                staging_offsets=[len(payload), 0] * 3,
                block_ids=blocks,
                modes=[0, 1] * 3,
                buffer_index=ring,
                stream=stream,
            )
        stream.synchronize()
        assert restored == 6
        for index, block in enumerate(blocks):
            expected = fallback_raw if index % 2 == 0 else packed_raw
            assert (
                destination[block].view(torch.uint8).cpu().numpy().tobytes() == expected
            )
    assert not torch.count_nonzero(destination[4])
    assert (
        decoder.decode(
            staging=staging,
            staging_offsets=[0],
            block_ids=[4],
            modes=[1],
            buffer_index=0,
            stream=stream,
        )
        == 1
    )
    stream.synchronize()
    assert destination[4].view(torch.uint8).cpu().numpy().tobytes() == packed_raw
    with pytest.raises(ValueError, match="distinct"):
        decoder.decode(
            staging=staging,
            staging_offsets=[0, len(payload)],
            block_ids=[0, 0],
            modes=[1, 0],
            buffer_index=0,
            stream=stream,
        )
    with pytest.raises(ValueError, match="conflicting modes"):
        decoder.decode(
            staging=staging,
            staging_offsets=[0, 0],
            block_ids=[0, 1],
            modes=[1, 0],
            buffer_index=0,
            stream=stream,
        )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("tile_scalars", "escape_pattern"),
    [
        (256, "sparse"),
        (256, "secondary"),
        (256, "clustered_raw"),
        (512, "sparse"),
        (513, "sparse"),
        (1024, "sparse"),
        (2048, "sparse"),
    ],
)
@pytest.mark.parametrize("online", [False, True])
def test_online_packer_restores_escapes_and_raw_overflow_byte_exact(
    tile_scalars: int,
    escape_pattern: str,
    online: bool,
) -> None:
    """Preserve escapes and raw slots across aligned and partial tile words."""
    geometry = replace(_geometry(), tile_scalars=tile_scalars)
    if escape_pattern != "sparse":
        # A full escape stream exercises the final allocated scratch word.
        # Thirty-one tiles leave one inactive warp and a partial tail tile.
        geometry = CompressedStoreGeometry(
            num_slots=2,
            slot_size=8 * 2 * 128 * 61 * 2,
            block_tokens=128,
            num_layers=8,
            num_kv_heads=1,
            head_dim=61,
            tile_scalars=tile_scalars,
        )
    raw_compressed = _slot(geometry, 21)
    raw_fallback_array = np.frombuffer(_slot(geometry, 22), dtype=np.uint8).copy()
    raw_fallback_array[1::2] = 0x7E
    raw_fallback = raw_fallback_array.tobytes()
    codebooks = default_online_codebooks(geometry)
    if escape_pattern != "sparse":
        raw_array = (
            np.frombuffer(raw_compressed, dtype=np.uint8)
            .copy()
            .reshape(geometry.plane_count, geometry.plane_scalars, 2)
        )
        for plane in range(geometry.plane_count):
            if escape_pattern == "secondary":
                raw_array[plane, :, 1] = codebooks[plane * 15 + 13]
            else:
                book = codebooks[plane * 15 : (plane + 1) * 15]
                raw_symbol = next(value for value in range(256) if value not in book)
                # Adjacent groups cover a full eight-symbol raw escape word,
                # mixed primary/secondary/raw ranks, and empty escape words.
                # The odd period shifts these patterns through every lane and
                # byte boundary without making the whole slot incompressible.
                pattern = np.array(
                    [raw_symbol] * 8
                    + [book[0], raw_symbol, book[13], raw_symbol, book[6], book[7]]
                    + [book[0]] * 51,
                    dtype=np.uint8,
                )
                raw_array[plane, :, 1] = np.resize(pattern, geometry.plane_scalars)
        raw_compressed = raw_array.tobytes()
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
    compressed_bytes = staging[packed[0].source_offset :]
    header = SlotHeader.parse(
        compressed_bytes.cpu().numpy().tobytes(),
        expected_geometry=geometry,
        expected_slot_id=packed[0].logical_slot,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert header.stored_length == packed[0].stored_length
    # An independent CPU reader checks that the host span, persisted header,
    # and GPU encoder agree; GPU roundtrip alone could share an offset bug.
    assert (
        decode_slot(
            compressed_bytes[: packed[0].stored_length].cpu().numpy().tobytes(),
            mode=packed[0].mode,
            slot_id=packed[0].logical_slot,
            geometry=geometry,
            codebooks=codebooks,
            verify_hash=False,
        )
        == raw_compressed
    )
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
        online=online,
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


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("tile_scalars", "num_layers", "head_dim"),
    [(256, 2, 127), (1024, 2, 127), (256, 18, 129)],
)
def test_online_packer_handles_production_batch_geometry_and_partial_tail(
    tile_scalars: int,
    num_layers: int,
    head_dim: int,
) -> None:
    """Restore a mixed, non-contiguous production-shaped batch exactly.

    FAST requests can submit dozens of logical blocks in one store/load batch,
    and vLLM physical block IDs are not required to be contiguous.  The final
    prompt block may also contain fewer than ``block_tokens`` valid tokens;
    online packing must zero the invalid tail while preserving an incompressible
    raw-fallback slot elsewhere in the same batch.
    """
    geometry = CompressedStoreGeometry(
        num_slots=64,
        slot_size=num_layers * 2 * 128 * 4 * head_dim * 2,
        block_tokens=128,
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=head_dim,
        tile_scalars=tile_scalars,
    )
    source_block_ids = [1 + ((index * 17) % 95) for index in range(48)]
    assert len(set(source_block_ids)) == len(source_block_ids)
    raw_slots: list[bytes] = []
    for index in range(96):
        if index == source_block_ids[7]:
            raw = np.full(geometry.slot_size, 0x7E, dtype=np.uint8)
        else:
            raw = np.empty(geometry.slot_size, dtype=np.uint8)
            rng = np.random.default_rng(1000 + index)
            raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
            common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
            raw[1::2] = common[
                rng.integers(0, len(common), geometry.slot_size // 2, dtype=np.uint8)
            ]
        raw_slots.append(raw.tobytes())

    codebooks = default_online_codebooks(geometry)
    source = (
        torch.from_numpy(np.frombuffer(b"".join(raw_slots), dtype=np.uint16).copy())
        .to("cuda")
        .view(torch.bfloat16)
        .reshape(
            96,
            geometry.num_layers,
            2,
            geometry.block_tokens,
            geometry.num_kv_heads,
            geometry.head_dim,
        )
    )
    warm_fused_online_kv_packer(
        source,
        max_slots_per_buffer=len(source_block_ids),
        tile_scalars=geometry.tile_scalars,
    )
    packer = FusedOnlineKVPacker(
        kv_cache=source,
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
        max_slots_per_buffer=len(source_block_ids),
    )
    staging = torch.empty(
        len(source_block_ids) * geometry.slot_size,
        dtype=torch.uint8,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    valid_tokens = (len(source_block_ids) - 1) * geometry.block_tokens + 127
    packed = packer.pack_into(
        staging=staging,
        block_ids=source_block_ids,
        logical_slots=list(range(len(source_block_ids))),
        slot_stride=geometry.slot_size,
        stream=stream,
        valid_token_count=valid_tokens,
    )
    stream.synchronize()

    assert packed[7].mode is SlotMode.RAW
    assert any(slot.mode is SlotMode.COMPRESSED for slot in packed)
    destination_block_ids = [3 + ((index * 19) % 119) for index in range(48)]
    assert len(set(destination_block_ids)) == len(destination_block_ids)
    destination = torch.zeros(
        128,
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
        max_slots_per_buffer=len(source_block_ids),
        online=tile_scalars == 256,
    )
    decoder.decode(
        staging=staging,
        staging_offsets=[slot.source_offset for slot in packed],
        block_ids=destination_block_ids,
        modes=[int(slot.mode) for slot in packed],
        buffer_index=1,
        stream=stream,
    )
    stream.synchronize()

    for index, (source_id, destination_id) in enumerate(
        zip(source_block_ids, destination_block_ids, strict=True)
    ):
        expected = np.frombuffer(raw_slots[source_id], dtype=np.uint8).copy()
        if (
            index == len(source_block_ids) - 1
            and packed[index].mode is SlotMode.COMPRESSED
        ):
            valid_tail_tokens = (
                valid_tokens - (len(source_block_ids) - 1) * geometry.block_tokens
            )
            # Slots are laid out plane-major, so a partial final block has an
            # invalid-token tail in every layer/KV plane rather than only in
            # the final contiguous fraction of the slot.
            plane_bytes = geometry.plane_scalars * 2
            valid_plane_bytes = valid_tail_tokens * (
                plane_bytes // geometry.block_tokens
            )
            for plane in range(geometry.plane_count):
                plane_start = plane * plane_bytes
                expected[
                    plane_start + valid_plane_bytes : plane_start + plane_bytes
                ] = 0
        actual = destination[destination_id].view(torch.uint8).cpu().numpy().reshape(-1)
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(actual != expected)[0])
            begin = max(0, mismatch - 8)
            end = min(len(expected), mismatch + 8)
            raise AssertionError(
                f"slot {index} (source {source_id}, destination {destination_id}) "
                f"mode={packed[index].mode.name} mismatch at {mismatch}; "
                f"expected={expected[begin:end].tolist()} "
                f"actual={actual[begin:end].tolist()}"
            )
