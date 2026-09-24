# SPDX-License-Identifier: Apache-2.0
"""Public deferred-store cancellation contracts, without CUDA traffic."""

import asyncio
import time
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("cupy")

from daser.connector.ipc_client import IPCClientAsync
from daser.connector.metadata import DaserConnectorMeta, ReqStoreSpec
from daser.connector.worker.runtime import WorkerRuntime
from daser.connector.worker.store import StorePipeline


def collect_until_ready(pipeline: StorePipeline) -> set[str]:
    """Poll a bounded test until a send completes or an async failure surfaces."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        completed = pipeline.collect_finished(set())
        if completed:
            return completed
        time.sleep(0.001)
    raise AssertionError("test pipeline did not complete")


def test_cancel_unsent_store_releases_identity_without_stale_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canceled source specs cannot later be saved, even when an ID is resumed."""
    released: list[tuple[str, int, int]] = []

    async def release(
        self: IPCClientAsync, chunk_key: str, start_slot: int, num_slots: int
    ) -> None:
        del self
        released.append((chunk_key, start_slot, num_slots))

    monkeypatch.setattr(IPCClientAsync, "release_chunk_writer", release)
    pipeline = StorePipeline("unused-test-socket")
    try:
        pipeline.queue_finished(
            {"req:store:0": ReqStoreSpec("old", 9, 1, [10], 4096, 128)}
        )
        pipeline.cancel_pending({"missing", "req"})
        pipeline.cancel_pending({"req"})
        assert pipeline.collect_finished({"req"}) == set()
        pipeline.queue_finished(
            {"req:store:0": ReqStoreSpec("resumed", 15, 1, [12], 8192, 128)}
        )
        assert pipeline.collect_finished({"req"}) == set()
        assert collect_until_ready(pipeline) == {"req"}
        assert pipeline.collect_finished(set()) == set()
    finally:
        pipeline.shutdown()
    assert released == [("old", 9, 1)]


def test_submitted_save_is_preserved_by_cancellation() -> None:
    """Submission transfers lifetime ownership; cancellation cannot drop it."""
    pipeline = StorePipeline("unused-test-socket")
    try:
        pipeline.queue_finished({"req": ReqStoreSpec("live", 1, 1, [4], 4096, 128)})
        assert pipeline.collect_finished({"req"}) == set()
        pipeline.cancel_pending({"req"})
        # An unconfigured pipeline has no KV batch, so this exercises the
        # request lifecycle without starting CUDA or a storage connection.
        assert collect_until_ready(pipeline) == {"req"}
        assert pipeline.collect_finished(set()) == set()
    finally:
        pipeline.shutdown()


def test_preemption_without_forward_releases_store_without_completing_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public runtime hook handles cancellation even without another save."""
    released: list[tuple[str, int, int]] = []

    async def release(
        self: IPCClientAsync, chunk_key: str, start_slot: int, num_slots: int
    ) -> None:
        del self
        released.append((chunk_key, start_slot, num_slots))

    monkeypatch.setattr(IPCClientAsync, "release_chunk_writer", release)
    runtime = WorkerRuntime(
        socket_path="unused-test-socket",
        transfer_mode="iouring",
        skip_l2=True,
        tp_size=1,
        tp_rank=0,
        server_tp_size=1,
        slot_size=4096,
        store_path="",
        rank_stride_bytes=4096,
        rope_base=10000.0,
        rope_rotary_dim=0,
        rope_is_neox_style=True,
        rope_delta_scale=1.0,
        load_key_scale=1.0,
        load_value_scale=1.0,
        kv_cache_config=None,
    )
    try:
        runtime.bind_connector_metadata(
            DaserConnectorMeta(
                reqs_to_store={
                    "req:store:0": ReqStoreSpec("old", 9, 1, [10], 4096, 128),
                }
            )
        )
        runtime.wait_for_save()
        runtime.clear_connector_metadata()
        runtime.handle_preemptions(DaserConnectorMeta(cancelled_store_req_ids={"req"}))
        # vLLM's no-forward path can collect completion without invoking save.
        # A later request finish must never submit this recycled source.
        assert runtime.get_finished({"req"}) == (None, None)
        assert runtime.get_finished(set()) == (None, None)
    finally:
        runtime.shutdown()
    assert released == [("old", 9, 1)]


@pytest.mark.parametrize("observe_in_shutdown", [False, True])
def test_release_error_is_visible_and_other_claims_are_attempted(
    monkeypatch: pytest.MonkeyPatch, observe_in_shutdown: bool
) -> None:
    """Writer cleanup must not suppress errors or abandon other allocations."""
    attempted: set[str] = set()

    async def release(
        self: IPCClientAsync, chunk_key: str, start_slot: int, num_slots: int
    ) -> None:
        del self, start_slot, num_slots
        attempted.add(chunk_key)
        await asyncio.sleep(0)
        if chunk_key == "bad":
            raise RuntimeError("release failed")

    monkeypatch.setattr(IPCClientAsync, "release_chunk_writer", release)
    pipeline = StorePipeline("unused-test-socket")
    pipeline.queue_finished(
        {
            "req:store:0": ReqStoreSpec("bad", 1, 1, [4], 4096, 128),
            "req:store:1": ReqStoreSpec("good", 2, 1, [5], 8192, 128),
        }
    )
    pipeline.cancel_pending({"req"})
    if observe_in_shutdown:
        with pytest.raises(RuntimeError, match="release failed"):
            pipeline.shutdown()
    else:
        try:
            with pytest.raises(RuntimeError, match="release failed"):
                collect_until_ready(pipeline)
        finally:
            pipeline.shutdown()
    assert attempted == {"bad", "good"}


@pytest.mark.asyncio
async def test_async_release_preserves_allocation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public async client must send the generation's full identity."""
    payloads: list[dict[str, Any]] = []

    async def call(self: IPCClientAsync, payload: dict[str, Any]) -> dict[str, Any]:
        del self
        payloads.append(payload)
        return {"ok": True}

    monkeypatch.setattr(IPCClientAsync, "call", call)
    client = IPCClientAsync("unused-test-socket")
    await client.release_chunk_writer("chunk", 25, 2)
    await client.close()
    assert payloads == [
        {
            "op": "release_chunk_writer",
            "chunk_key": "chunk",
            "start_slot": 25,
            "num_slots": 2,
        }
    ]
