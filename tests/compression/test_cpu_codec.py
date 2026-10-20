# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from daser.compression import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from daser.compression.cpu import CpuSlotEncoder
from daser.compression.format import SlotHeader


def _geometry() -> CompressedStoreGeometry:
    # Eight planes pairs make the slot 4 KiB aligned despite a 128-scalar tail.
    return CompressedStoreGeometry(
        num_slots=4,
        slot_size=8 * 2 * 128 * 61 * 2,
        block_tokens=128,
        num_layers=8,
        num_kv_heads=1,
        head_dim=61,
        tile_scalars=256,
    )


def _raw(geometry: CompressedStoreGeometry, kind: str, books: bytes) -> bytes:
    rng = np.random.default_rng(314)
    raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8)
    if kind == "raw":
        return raw.tobytes()
    high = raw[1::2].reshape(geometry.plane_count, geometry.plane_scalars)
    for plane, row in enumerate(high):
        table = np.frombuffer(books, np.uint8)[plane * 15 : (plane + 1) * 15]
        row[:] = table[0] if kind in ("primary", "uneven") else table[7]
        if kind == "mixed":
            # All high-byte values, both packed-stream boundary crossings, and
            # escapes directly around a tile boundary. Lows remain arbitrary.
            row[247:503] = np.arange(256, dtype=np.uint8)
            row[::11] = table[13]
            row[::17] = table[14]
    if kind == "uneven":
        # A single plane's worst-case escape stream still fits the complete
        # slot. CPU admission must not copy the GPU's per-plane escape cap.
        high[0] = 255
    if kind == "mixed":
        # +/-0, +/-infinity, quiet/signalling NaNs and +/-subnormals.
        special = np.array(
            [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 1, 0x8001, 0xFFFF],
            dtype="<u2",
        )
        raw[: special.nbytes] = special.view(np.uint8)
    return raw.tobytes()


@pytest.fixture(scope="module", autouse=True)
def cpu_library(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Explicitly compile the native test library; no runtime JIT is involved."""
    if os.environ.get("DASER_CPU_CODEC_LIBRARY"):
        yield
        return
    if not shutil.which("c++"):
        pytest.skip("a C++17 compiler is required for the native CPU codec")
    target = tmp_path_factory.mktemp("cpu-codec") / "codec.so"
    source = Path(__file__).resolve().parents[2] / "daser/compression/cpu_codec.cpp"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(target),
        ],
        check=True,
    )
    os.environ["DASER_CPU_CODEC_LIBRARY"] = str(target)
    try:
        yield
    finally:
        del os.environ["DASER_CPU_CODEC_LIBRARY"]


@pytest.mark.parametrize("kind", ["primary", "secondary", "mixed", "uneven", "raw"])
@pytest.mark.parametrize("custom_books", [False, True])
def test_reference_wire_contract_and_reuse(kind: str, custom_books: bool) -> None:
    """Protect packed tails/prefixes and stale-buffer reuse against the oracle."""
    geometry = _geometry()
    books = default_online_codebooks(geometry)
    if custom_books:
        books = b"".join(
            bytes((value + plane) % 256 for value in range(15))
            for plane in range(geometry.plane_count)
        )
    encoder = CpuSlotEncoder(geometry, books)
    output = bytearray([0xAB]) * (geometry.slot_size + 37)
    try:
        for variant in [kind, "raw", kind]:
            raw = _raw(geometry, variant, books)
            result = encoder.encode_into(raw, output, slot_id=3)
            payload = bytes(output[: result.stored_length])
            assert output[geometry.slot_size :] == bytes([0xAB]) * 37
            reference = encode_slot(raw, slot_id=3, geometry=geometry, codebooks=books)
            assert result.mode == reference.mode
            if variant == "uneven":
                assert result.mode == SlotMode.COMPRESSED
            expected = bytearray(reference.payload)
            if result.mode == SlotMode.COMPRESSED:
                expected[88:120] = b"\x01" + bytes(31)
                header = SlotHeader.parse(payload, expected_geometry=geometry)
                assert header.raw_hash == b"\x01" + bytes(31)
                with pytest.raises(ValueError, match="hash mismatch"):
                    decode_slot(
                        payload,
                        mode=result.mode,
                        slot_id=3,
                        geometry=geometry,
                        codebooks=books,
                    )
            assert payload == expected
            assert (
                decode_slot(
                    payload,
                    mode=result.mode,
                    slot_id=3,
                    geometry=geometry,
                    codebooks=books,
                    verify_hash=False,
                )
                == raw
            )
        assert 0 < encoder.workspace_bytes < geometry.slot_size
    finally:
        encoder.close()


@pytest.mark.parametrize("escape_remainder", range(8))
def test_packed_words_cross_tiles_with_unaligned_buffers(
    escape_remainder: int,
) -> None:
    """Protect all secondary bit offsets, short tails and three-byte writes."""
    geometry = _geometry()
    books = default_online_codebooks(geometry)
    raw = np.frombuffer(_raw(geometry, "primary", books), np.uint8).copy()
    high = raw[1::2].reshape(geometry.plane_count, geometry.plane_scalars)
    for plane, row in enumerate(high):
        table = np.frombuffer(books, np.uint8)[plane * 15 : (plane + 1) * 15]
        row[:] = table[np.arange(len(row)) % 7]
        for tile, begin in enumerate(range(0, len(row), 256)):
            count = min(256, len(row) - begin)
            escapes = min(count, (8 + escape_remainder + 17 * tile) % 257)
            row[begin : begin + escapes] = table[7 + np.arange(escapes) % 7]
            row[begin : begin + escapes : 5] = table[14]
    encoder = CpuSlotEncoder(geometry, books)
    # Neither buffer promises machine-word alignment. Guards protect caller
    # bounds; whole-payload comparison protects adjacent packed/prefix streams.
    source = memoryview(b"!" + raw.tobytes())[1:]
    guarded = bytearray([0xA5]) * (geometry.slot_size + 2)
    output = memoryview(guarded)[1:-1]
    try:
        result = encoder.encode_into(source, output, slot_id=1)
        assert result.mode == SlotMode.COMPRESSED
        expected = bytearray(
            encode_slot(
                raw.tobytes(), slot_id=1, geometry=geometry, codebooks=books
            ).payload
        )
        expected[88:120] = b"\x01" + bytes(31)
        assert output[: result.stored_length].tobytes() == expected
        assert guarded[0] == guarded[-1] == 0xA5
    finally:
        encoder.close()


@pytest.mark.integration
@pytest.mark.parametrize("chunk_slots", [None, 1, 3])
def test_cpu_batch_pipeline_layout_order_and_buffer_reuse(
    chunk_slots: int | None,
) -> None:
    """Protect partial copy groups and disjoint buffer reuse across full runs."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from benchmarks.compression.bench_cpu_codec import CpuBatch
    from daser.ops.kv_layout import allocate_cross_layer_kv_cache

    geometry = replace(_geometry(), num_slots=8)
    books = default_online_codebooks(geometry)
    kinds = ["mixed", "raw", "secondary", "uneven"] * 2
    originals = [_raw(geometry, kind, books) for kind in kinds]
    corpus = np.frombuffer(b"".join(originals), dtype="<u2").copy()
    corpus = corpus.reshape(8, 8, 2, 128, 1, 61)
    source = allocate_cross_layer_kv_cache(
        8, 8, 128, 1, 61, dtype=torch.bfloat16, device="cuda"
    )
    source.copy_(torch.from_numpy(corpus).view(torch.bfloat16))
    torch.cuda.synchronize()
    runner = CpuBatch(corpus, source, geometry, books, 4, 2, True, chunk_slots)
    oracle = CpuSlotEncoder(geometry, books)
    output = bytearray(geometry.slot_size)
    try:
        # A different caller stream must not move copies away from the stream
        # the runner fences before CPU threads read their pinned input slices.
        with torch.cuda.stream(torch.cuda.Stream()):
            for start in (0, 4, 0):
                if start == 4:
                    records, trace = runner.profile(start)
                    for copy in trace["copies"]:
                        for index in range(
                            copy["first"], copy["first"] + copy["count"]
                        ):
                            assert trace["slots"][index]["start_ms"] >= copy["done_ms"]
                else:
                    records = runner.run(start)
                for index, record in enumerate(records):
                    expected = oracle.encode_into(
                        originals[start + index], output, slot_id=start + index
                    )
                    assert record == expected
                    assert (
                        runner.outputs[index][: record.stored_length]
                        == output[: expected.stored_length]
                    )
        assert runner.memory()["host_pinned_input_bytes"] == 4 * geometry.slot_size
    finally:
        runner.close()
        oracle.close()


def test_buffer_bounds_lifetime_and_geometry() -> None:
    geometry = _geometry()
    books = default_online_codebooks(geometry)
    encoder = CpuSlotEncoder(geometry, books)
    raw = bytearray(geometry.slot_size)
    output = bytearray(geometry.slot_size)
    with pytest.raises(ValueError, match="overlap"):
        encoder.encode_into(raw, raw, slot_id=0)
    overlap = bytearray(geometry.slot_size * 2)
    with pytest.raises(ValueError, match="overlap"):
        encoder.encode_into(
            memoryview(overlap)[1 : geometry.slot_size + 1],
            memoryview(overlap)[: geometry.slot_size],
            slot_id=0,
        )
    with pytest.raises(ValueError, match="contiguous"):
        encoder.encode_into(memoryview(overlap)[::2], output, slot_id=0)
    with pytest.raises(ValueError, match="writable"):
        encoder.encode_into(raw, bytes(output), slot_id=0)
    with pytest.raises(ValueError, match="length"):
        encoder.encode_into(raw[:-1], output, slot_id=0)
    with pytest.raises(ValueError, match="complete raw slot"):
        encoder.encode_into(raw, output[:-1], slot_id=0)
    for slot_id in (-1, geometry.num_slots):
        with pytest.raises(ValueError, match="slot_id"):
            encoder.encode_into(raw, output, slot_id=slot_id)
    encoder.close()
    encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode_into(raw, output, slot_id=0)
    with pytest.raises(ValueError, match="distinct"):
        CpuSlotEncoder(geometry, bytes(len(books)))
    with pytest.raises(ValueError, match="256"):
        CpuSlotEncoder(replace(geometry, tile_scalars=1024), books)


def test_independent_workers_and_all_bf16_bit_patterns() -> None:
    geometry = replace(
        _geometry(), block_tokens=256, slot_size=_geometry().slot_size * 2
    )
    books = default_online_codebooks(geometry)

    def run(worker: int) -> bytes:
        encoder = CpuSlotEncoder(geometry, books)
        raw = np.resize(np.arange(65536, dtype="<u2"), geometry.slot_size // 2)
        raw = np.roll(raw, worker * 991).tobytes()
        output = bytearray(geometry.slot_size)
        try:
            result = encoder.encode_into(raw, output, slot_id=worker)
            assert result.mode == SlotMode.RAW
            assert bytes(output) == raw
            return bytes(output)
        finally:
            encoder.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(set(pool.map(run, range(4)))) == 4


@pytest.mark.integration
def test_existing_gpu_decoder_mixed_cpu_slots_and_reuse() -> None:
    """Use the unchanged GPU consumer with compact mixed RAW/compressed records."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from daser.ops.compressed_kv import FusedCompressedKVDecoder
    from daser.ops.kv_layout import allocate_cross_layer_kv_cache

    geometry = _geometry()
    books = default_online_codebooks(geometry)
    encoder = CpuSlotEncoder(geometry, books)
    destination = allocate_cross_layer_kv_cache(
        4, 8, 128, 1, 61, dtype=torch.bfloat16, device="cuda"
    )
    decoder = FusedCompressedKVDecoder(
        kv_cache=destination,
        codebooks=books,
        tile_scalars=256,
        ring_depth=1,
        max_slots_per_buffer=4,
    )
    stream = torch.cuda.current_stream()
    staging = torch.empty(4 * geometry.slot_size, dtype=torch.uint8, device="cuda")
    output = bytearray(geometry.slot_size)
    try:
        for kinds in [
            ["mixed", "raw", "uneven", "primary"],
            ["raw", "primary", "mixed", "secondary"],
        ]:
            payloads, offsets, modes, originals = [], [], [], []
            for slot, kind in enumerate(kinds):
                raw = _raw(geometry, kind, books)
                result = encoder.encode_into(raw, output, slot_id=slot)
                offsets.append(sum(map(len, payloads)))
                payloads.append(bytes(output[: result.stored_length]))
                originals.append(raw)
                modes.append(int(result.mode))
            packed = np.frombuffer(b"".join(payloads), np.uint8).copy()
            staging[: len(packed)].copy_(torch.from_numpy(packed))
            decoder.decode(
                staging=staging,
                staging_offsets=offsets,
                block_ids=[2, 0, 3, 1],
                modes=modes,
                buffer_index=0,
                stream=stream,
            )
            stream.synchronize()
            for slot, block in enumerate([2, 0, 3, 1]):
                actual = destination[block].contiguous().view(torch.uint8).cpu().numpy()
                assert actual.tobytes() == originals[slot]
    finally:
        encoder.close()
