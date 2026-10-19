# SPDX-License-Identifier: Apache-2.0
"""Exercise public load ownership with real CUDA copies and packed restores."""

import asyncio
import time
from typing import Any

import numpy as np
import pytest
import torch

from daser.compression import (
    CompressedStoreGeometry,
    default_online_codebooks,
    encode_slot,
)
from daser.connector.metadata import CompressedLoadSlot, ReqLoadSpec
from daser.connector.worker.load import LoadPipeline
from daser.connector.worker.memory import FixedCudaStagingPool
from daser.ops.compressed_kv import FusedCompressedKVDecoder, PreparedKVRestore
from daser.ops.kv_layout import allocate_cross_layer_kv_cache


class MemoryLoadClient:
    """Supply immutable pinned bytes through the production async IPC boundary."""

    def __init__(self, pool: FixedCudaStagingPool, payload: bytes) -> None:
        self.pool = pool
        self.host = torch.empty(len(payload), dtype=torch.uint8, pin_memory=True)
        self.host.copy_(torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy()))
        self.calls: list[int] = []

    async def init_transfer(self) -> None:
        """Accept startup on the load loop; no external service is needed."""

    async def register_staging_cuda(self, **kwargs: Any) -> None:
        """Accept exported buffer registration on the load loop."""
        assert kwargs["direction"] == "load"

    async def transfer_load_registered_cuda(self, **kwargs: Any) -> dict[str, Any]:
        """Return only after real H2D finishes, retaining the pinned source."""
        buffer_index = int(kwargs["buffer_index"])
        target = self.pool.buffer(buffer_index)
        torch.cuda.set_device(target.device)
        stream = torch.cuda.Stream(device=target.device)
        self.calls.append(buffer_index)
        with torch.cuda.stream(stream):
            for span in kwargs["spans"]:
                start = int(span["target_offset"])
                offset = int(span["file_offset"])
                nbytes = int(span["nbytes"])
                target[start : start + nbytes].copy_(
                    self.host[offset : offset + nbytes], non_blocking=True
                )
            event = torch.cuda.Event()
            event.record(stream)
        while not event.query():
            await asyncio.sleep(0)
        return {}

    async def close(self) -> None:
        """Close after the public pipeline shutdown has drained every load."""


def wait_finished(pipeline: LoadPipeline, request: str) -> None:
    """Poll public completion with a bounded timeout on the test's main thread."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if request in pipeline.collect_finished():
            return
        time.sleep(0.001)
    raise AssertionError("load did not complete")


def _packed_fixture(
    num_slots: int,
) -> tuple[CompressedStoreGeometry, Any, list[bytes], list[Any], bytes]:
    """Encode mixed raw/packed slots and their back-to-back load references."""
    geometry = CompressedStoreGeometry(
        num_slots=num_slots,
        slot_size=2 * 2 * 128 * 32 * 2,
        num_layers=2,
        block_tokens=128,
        num_kv_heads=1,
        head_dim=32,
    )
    rng = np.random.default_rng(123)
    raw_slots = []
    for index in range(geometry.num_slots):
        raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8)
        if index % 3:
            raw[1::2] = rng.choice(
                np.array([0x3E, 0x3F, 0xBF], dtype=np.uint8), geometry.slot_size // 2
            )
            raw[1::1994] = 0x7E
        raw_slots.append(raw.tobytes())
    codebooks = default_online_codebooks(geometry)
    encoded = [
        encode_slot(raw, slot_id=index, geometry=geometry, codebooks=codebooks)
        for index, raw in enumerate(raw_slots)
    ]
    refs = []
    offset = 0
    for index, slot in enumerate(encoded):
        refs.append(
            CompressedLoadSlot(
                index,
                "raw" if int(slot.mode) == 0 else "compressed",
                offset,
                len(slot.payload),
            )
        )
        offset += len(slot.payload)
    return geometry, codebooks, raw_slots, refs, b"".join(s.payload for s in encoded)


def _full_spec(geometry: CompressedStoreGeometry, refs: list[Any]) -> ReqLoadSpec:
    """Load every fixture slot into the destination block of the same index."""
    return ReqLoadSpec(
        chunk_key="mixed",
        start_slot=0,
        num_slots=geometry.num_slots,
        block_ids=list(range(geometry.num_slots)),
        file_offset=0,
        token_count=geometry.num_slots * geometry.block_tokens,
        compressed_slots=refs,
    )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("step_fence", [False, True])
def test_packed_restore_waits_for_queued_model_stream_writes(
    monkeypatch: pytest.MonkeyPatch, step_fence: bool
) -> None:
    """A restore lands after model-stream work queued before its step began.

    Under vLLM async scheduling a block handed to a load can still be written
    by kernels of its previous owner. The model stream here sleeps, then
    overwrites the destination; the restore must not finish before that write.
    """
    geometry, codebooks, raw_slots, refs, payload = _packed_fixture(4)
    pool = FixedCudaStagingPool(torch.device("cuda:0"), 4 * geometry.slot_size, 2)
    client = MemoryLoadClient(pool, payload)
    monkeypatch.setattr("daser.connector.worker.load.IPCClientAsync", lambda _: client)
    destination = allocate_cross_layer_kv_cache(
        geometry.num_slots, 2, 128, 1, 32, dtype=torch.bfloat16, device="cuda"
    ).zero_()
    torch.cuda.synchronize()
    pipeline = LoadPipeline("memory-test", client_count=2)
    pipeline.configure(
        kv_caches={"cross": destination},
        layer_names=["a", "b"],
        local_slot_size=geometry.slot_size,
        rank_stride_bytes=0,
        tp_rank=0,
        staging_pool=pool,
        load_key_scale=1.0,
        load_value_scale=1.0,
        rope_delta_scale=1.0,
        rope_base=10000.0,
        rope_rotary_dim=0,
        rope_is_neox_style=True,
    )
    pipeline.configure_compression(
        storage_format="compressed-online",
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
    )
    pipeline.initialize_transfer()
    # vLLM runs the model on a dedicated non-default stream; the legacy default
    # stream would implicitly order the restore and hide a missing fence.
    model_stream = torch.cuda.Stream()
    try:
        # Warm up every kernel used below: CUDA lazy module loading synchronizes
        # the device on a kernel's first launch, which would mask the race.
        with torch.cuda.stream(model_stream):
            torch.cuda._sleep(1000)  # noqa: SLF001 - no public GPU spin
            destination.view(torch.uint8)[:1].fill_(0)
            pipeline.start({"warm": _full_spec(geometry, refs)})
        wait_finished(pipeline, "warm")
        torch.cuda.synchronize()
        with torch.cuda.stream(model_stream):
            # The previous owner's write, still queued behind ~1 s of GPU spin.
            torch.cuda._sleep(1_000_000_000)  # noqa: SLF001 - no public GPU spin
            destination.view(torch.uint8).fill_(0xFF)
            if step_fence:
                pipeline.begin_step(True)
            pipeline.start({"late": _full_spec(geometry, refs)})
        wait_finished(pipeline, "late")
        assert pipeline.take_invalid_block_ids() == set()
        torch.cuda.synchronize()
        assert destination.view(torch.uint8).cpu().numpy().tobytes() == (
            b"".join(raw_slots)
        )
    finally:
        pipeline.shutdown()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("failure", [None, "prepare", "restore"])
def test_packed_pipeline_drains_failures_and_reuses_rings(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Mixed slots stay byte exact across batches, failures and repeated reuse."""
    geometry, codebooks, raw_slots, refs, payload = _packed_fixture(9)
    assert {ref.mode for ref in refs} == {"raw", "compressed"}
    pool = FixedCudaStagingPool(torch.device("cuda:0"), 2 * geometry.slot_size, 2)
    client = MemoryLoadClient(pool, payload)
    monkeypatch.setattr("daser.connector.worker.load.IPCClientAsync", lambda _: client)
    destination = allocate_cross_layer_kv_cache(
        geometry.num_slots, 2, 128, 1, 32, dtype=torch.bfloat16, device="cuda"
    ).zero_()
    torch.cuda.synchronize()
    pipeline = LoadPipeline("memory-test", client_count=2)
    pipeline.configure(
        kv_caches={"cross": destination},
        layer_names=["a", "b"],
        local_slot_size=geometry.slot_size,
        rank_stride_bytes=0,
        tp_rank=0,
        staging_pool=pool,
        load_key_scale=1.0,
        load_value_scale=1.0,
        rope_delta_scale=1.0,
        rope_base=10000.0,
        rope_rotary_dim=0,
        rope_is_neox_style=True,
    )
    pipeline.configure_compression(
        storage_format="compressed-online",
        codebooks=codebooks,
        tile_scalars=geometry.tile_scalars,
    )
    pipeline.initialize_transfer()
    original_prepare = FusedCompressedKVDecoder.prepare
    original_submit = PreparedKVRestore.submit
    prepare_calls = 0
    restore_calls = 0

    def prepare(decoder: FusedCompressedKVDecoder, **kwargs: Any) -> Any:
        nonlocal prepare_calls
        plan = original_prepare(decoder, **kwargs)
        prepare_calls += 1
        if failure == "prepare" and prepare_calls == 2:
            raise RuntimeError("injected failure after metadata GPU submission")
        return plan

    def submit(plan: PreparedKVRestore) -> int:
        nonlocal restore_calls
        restored = original_submit(plan)
        restore_calls += 1
        if failure == "restore" and restore_calls == 2:
            raise RuntimeError("injected failure after decode GPU submission")
        return restored

    monkeypatch.setattr(FusedCompressedKVDecoder, "prepare", prepare)
    monkeypatch.setattr(PreparedKVRestore, "submit", submit)
    try:
        for repeat in range(3):
            request = f"request-{repeat}"
            pipeline.start(
                {
                    request: ReqLoadSpec(
                        chunk_key="mixed",
                        start_slot=0,
                        num_slots=geometry.num_slots,
                        block_ids=list(range(geometry.num_slots)),
                        file_offset=0,
                        token_count=geometry.num_slots * geometry.block_tokens,
                        compressed_slots=refs,
                    )
                }
            )
            wait_finished(pipeline, request)
            invalid = pipeline.take_invalid_block_ids()
            assert pool.available == pool.depth
            if repeat == 0 and failure is not None:
                assert invalid == set(range(geometry.num_slots))
            else:
                assert invalid == set()
                assert destination.view(torch.uint8).cpu().numpy().tobytes() == (
                    b"".join(raw_slots)
                )
        assert len(client.calls) >= 9
        assert set(client.calls) == {0, 1}
    finally:
        pipeline.shutdown()
    assert pool.available == pool.depth
