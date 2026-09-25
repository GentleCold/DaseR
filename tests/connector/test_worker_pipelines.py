# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import Future
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("cupy")

import torch

from daser.connector.metadata import (
    CompressedLoadSlot,
    ReqLoadSpec,
    ReqStoreSpec,
    StoreWriteSpan,
)
from daser.connector.worker.load import LoadPipeline
from daser.connector.worker.memory import FixedCudaStagingPool
from daser.connector.worker.store import (
    StagedStoreBatch,
    StorePipeline,
)


class _ManualFuture:
    def __init__(self) -> None:
        self.complete = False

    def done(self) -> bool:
        return self.complete

    def result(self, timeout: float) -> None:
        del timeout


class _GateFuture:
    def __init__(self) -> None:
        self.complete = False

    def done(self) -> bool:
        return self.complete

    def result(self, timeout: float) -> None:
        del timeout
        if not self.complete:
            raise AssertionError("future result observed before completion")


def _store_spec(key: str, blocks: list[int]) -> ReqStoreSpec:
    return ReqStoreSpec(key, 0, len(blocks), blocks, 0, len(blocks))


def test_store_pipeline_dispatches_finished_saves_in_fifo_order() -> None:
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
    pipeline._pending_writer_releases = []  # noqa: SLF001
    pipeline._staging_pool = SimpleNamespace(depth=2)  # noqa: SLF001
    submitted: list[_ManualFuture] = []

    def submit(save: Any) -> None:
        future = _ManualFuture()
        submitted.append(future)
        save.future = future

    pipeline._submit_save = submit  # type: ignore[method-assign]  # noqa: SLF001
    pipeline.queue_finished(
        {req: _store_spec(req, [index]) for index, req in enumerate(("a", "b", "c"))},
    )

    assert pipeline.collect_finished(set()) == set()
    assert submitted == []
    assert pipeline.collect_finished({"a", "b", "c"}) == set()
    assert len(submitted) == 3
    submitted[0].complete = True
    assert pipeline.collect_finished(set()) == {"a"}
    assert len(submitted) == 3
    submitted[1].complete = True
    submitted[2].complete = True
    assert pipeline.collect_finished(set()) == {"b", "c"}


def test_packed_store_releases_request_after_snapshot_before_transfer() -> None:
    """Packed completion is emitted while its transfer future is pending."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
    pipeline._pending_writer_releases = []  # noqa: SLF001
    pipeline._online_packer = object()  # noqa: SLF001
    submitted: list[_GateFuture] = []

    def submit(save: Any) -> None:
        future = _GateFuture()
        submitted.append(future)
        save.future = future
        save.snapshot_done = True

    pipeline._submit_save = submit  # type: ignore[method-assign]  # noqa: SLF001
    pipeline.queue_finished({"req": _store_spec("req", [0])})

    # The first poll preserves the worker dispatch boundary.
    assert pipeline.collect_finished({"req"}) == set()
    assert pipeline.collect_finished(set()) == {"req"}
    assert "req" in pipeline._pending_finished_saves  # noqa: SLF001

    submitted[0].complete = True
    assert pipeline.collect_finished(set()) == set()
    assert "req" not in pipeline._pending_finished_saves  # noqa: SLF001


def test_packed_store_groups_finished_requests_by_staging_capacity() -> None:
    """One completion wave is greedily packed into bounded FIFO groups."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
    pipeline._pending_writer_releases = []  # noqa: SLF001
    pipeline._online_packer = object()  # noqa: SLF001
    pipeline._kv_caches = {}  # noqa: SLF001
    pipeline._staging_pool = SimpleNamespace(depth=1)  # noqa: SLF001
    pipeline._local_slot_size = 16  # noqa: SLF001
    pipeline._rank_stride_bytes = 0  # noqa: SLF001
    pipeline._tp_rank = 0  # noqa: SLF001
    pipeline._staging_bytes = 64  # noqa: SLF001
    pipeline._online_pack_batch_slots = 85  # noqa: SLF001
    submitted: list[list[str]] = []

    def submit_group(saves: tuple[Any, ...]) -> None:
        submitted.append([next(iter(save.reqs_to_store)) for save in saves])
        for save in saves:
            save.future = _ManualFuture()

    pipeline._submit_packed_save_group = submit_group  # type: ignore[method-assign]  # noqa: SLF001
    pipeline.queue_finished(
        {
            "a": _store_spec("a", [0, 1]),
            "b": _store_spec("b", [2, 3]),
            "c": _store_spec("c", [4]),
        }
    )
    pipeline._kv_caches = {"layer": torch.empty(1)}  # noqa: SLF001

    assert pipeline.collect_finished({"a", "b", "c"}) == set()
    assert submitted == [["a", "b"], ["c"]]


def test_packed_store_group_uses_latest_producer_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merged pack waits for the newest event, regardless of FIFO age."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
    pipeline._pending_writer_releases = []  # noqa: SLF001
    pipeline._online_packer = object()  # noqa: SLF001
    pipeline._kv_caches = {"layer": torch.empty(1)}  # noqa: SLF001
    pipeline._producer_event_order = 0  # noqa: SLF001
    events = [object(), object(), object()]
    captured: list[tuple[Any, list[str]]] = []

    monkeypatch.setattr(
        "daser.connector.worker.store.record_cuda_event",
        lambda _tensor: events.pop(0),
    )

    async def run_group(merged: Any, event: Any, members: tuple[Any, ...]) -> None:
        del members
        captured.append((event, list(merged.reqs_to_store)))

    def submit(coro: Any) -> Future[None]:
        future: Future[None] = Future()
        try:
            asyncio.run(coro)
            future.set_result(None)
        except BaseException as exc:
            future.set_exception(exc)
        return future

    pipeline._run_bounded_save_group = run_group  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._submit = submit  # type: ignore[method-assign]  # noqa: SLF001
    pipeline.queue_finished({"a": _store_spec("a", [0])})
    pipeline.queue_finished({"b": _store_spec("b", [1])})
    newest_event = events[0]
    pipeline.queue_finished({"a:store:0": _store_spec("a:store:0", [2])})
    saves = tuple(pipeline._pending_finished_saves.values())  # noqa: SLF001

    pipeline._submit_packed_save_group(saves)  # noqa: SLF001

    assert captured == [(newest_event, ["a", "a:store:0", "b"])]
    assert all(save.future.done() for save in saves)


@pytest.mark.asyncio
async def test_packed_store_holds_lease_until_transfer_finishes() -> None:
    """Staging ownership outlives snapshot notification and transfer overlap."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._online_packer = object()  # noqa: SLF001
    pipeline._store_capacity = 1  # noqa: SLF001
    pipeline._plan_finished_save = lambda _save: [  # type: ignore[method-assign]  # noqa: SLF001
        ([0], [StoreWriteSpan(0, 16, 0, "req", 0, 1, 0, 1)])
    ]
    transfer_started = asyncio.Event()
    transfer_release = asyncio.Event()
    released: list[int] = []

    class Lease:
        view = torch.empty(1)

        def release(self) -> None:
            released.append(1)

    async def stage(
        block_ids: list[int],
        spans: list[Any],
        event: Any,
    ) -> StagedStoreBatch:
        del block_ids, spans, event
        return StagedStoreBatch(torch.empty(1), [], Lease())

    async def write(staged: StagedStoreBatch) -> list[str]:
        del staged
        transfer_started.set()
        await transfer_release.wait()
        return ["req"]

    pipeline._stage_batch_acquired = stage  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._write_cuda_buffer = write  # type: ignore[method-assign]  # noqa: SLF001
    save = SimpleNamespace(snapshot_done=False)
    task = asyncio.create_task(pipeline._store_finished_save(save, None))  # noqa: SLF001

    await transfer_started.wait()
    await asyncio.sleep(0)
    assert save.snapshot_done is True
    assert released == []

    transfer_release.set()
    await task
    assert released == [1]


@pytest.mark.asyncio
async def test_packed_store_group_failure_releases_lease_and_snapshots_members() -> (
    None
):
    """A failed grouped transfer neither retains staging nor source blocks."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._online_packer = object()  # noqa: SLF001
    pipeline._store_capacity = 1  # noqa: SLF001
    pipeline._plan_finished_save = lambda _save: [  # type: ignore[method-assign]  # noqa: SLF001
        ([0, 1], [StoreWriteSpan(0, 32, 0, "group", 0, 2, 0, 2)])
    ]
    released: list[int] = []

    class Lease:
        view = torch.empty(1)

        def release(self) -> None:
            released.append(1)

    async def stage(
        block_ids: list[int],
        spans: list[Any],
        event: Any,
    ) -> StagedStoreBatch:
        del block_ids, spans, event
        return StagedStoreBatch(torch.empty(1), [], Lease())

    async def write(staged: StagedStoreBatch) -> list[str]:
        del staged
        raise RuntimeError("store failed")

    pipeline._stage_batch_acquired = stage  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._write_cuda_buffer = write  # type: ignore[method-assign]  # noqa: SLF001
    members = (
        SimpleNamespace(snapshot_done=False),
        SimpleNamespace(snapshot_done=False),
    )

    with pytest.raises(RuntimeError, match="store failed"):
        await pipeline._store_finished_save(  # noqa: SLF001
            SimpleNamespace(snapshot_done=False),
            None,
            snapshot_targets=members,
        )

    assert all(member.snapshot_done for member in members)
    assert released == [1]


def test_packed_store_captures_latest_producer_event_at_queue_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred packed stores retain the event from the latest worker step."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
    pipeline._pending_writer_releases = []  # noqa: SLF001
    pipeline._online_packer = object()  # noqa: SLF001
    pipeline._kv_caches = {"layer": torch.empty(1)}  # noqa: SLF001
    events = [object(), object()]

    def capture_event(tensor: torch.Tensor) -> object | None:
        if tensor is pipeline._kv_caches["layer"]:  # noqa: SLF001
            return events.pop(0)
        return None

    monkeypatch.setattr(
        "daser.connector.worker.store.record_cuda_event",
        capture_event,
    )

    pipeline.queue_finished({"req": _store_spec("req", [0])})
    first = pipeline._pending_finished_saves["req"].producer_event  # noqa: SLF001
    pipeline.queue_finished({"req:store:0": _store_spec("req:store:0", [1])})
    second = pipeline._pending_finished_saves["req"].producer_event  # noqa: SLF001

    assert first is not second
    assert second is not None
    assert pipeline._pending_finished_saves["req"].reqs_to_store.keys() == {  # noqa: SLF001
        "req",
        "req:store:0",
    }


@pytest.mark.asyncio
async def test_store_dispatcher_bounds_and_orders_background_saves() -> None:
    """Finished saves run FIFO while respecting the staging depth."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._store_capacity = 1  # noqa: SLF001
    pipeline._store_semaphore = None  # noqa: SLF001
    active = 0
    max_active = 0
    order: list[str] = []

    async def save(save: Any, event: Any) -> None:
        del event
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        order.append(save.req_id)
        await asyncio.sleep(0)
        active -= 1

    pipeline._store_finished_save = save  # type: ignore[method-assign]  # noqa: SLF001
    saves = [SimpleNamespace(req_id=req_id) for req_id in ("a", "b", "c")]
    await asyncio.gather(
        *(pipeline._run_bounded_save(save, None) for save in saves)  # noqa: SLF001
    )

    assert order == ["a", "b", "c"]
    assert max_active == 1


def test_store_pipeline_streams_request_larger_than_pool_depth() -> None:
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
    pipeline._pending_writer_releases = []  # noqa: SLF001
    pipeline._staging_pool = SimpleNamespace(depth=1)  # noqa: SLF001
    pipeline._store_capacity = 1  # noqa: SLF001
    pipeline._store_semaphore = None  # noqa: SLF001
    pipeline._kv_caches = {"layer": torch.empty(1)}  # noqa: SLF001
    pipeline._local_slot_size = 16  # noqa: SLF001
    pipeline._rank_stride_bytes = 0  # noqa: SLF001
    pipeline._tp_rank = 0  # noqa: SLF001
    pipeline._tp_size = 1  # noqa: SLF001
    pipeline._staging_bytes = 32  # noqa: SLF001
    released: list[int] = []
    writes: list[int] = []

    class Lease:
        view = torch.empty(1)

        def release(self) -> None:
            released.append(1)

    def stage(block_ids: list[int], spans: list[Any], event: Any) -> StagedStoreBatch:
        del event
        return StagedStoreBatch(torch.empty(1), spans, Lease())

    async def write(staged: StagedStoreBatch) -> list[str]:
        writes.append(sum(span.nbytes for span in staged.spans) // 16)
        return ["large"]

    def submit(coro: Any) -> Future[None]:
        future: Future[None] = Future()
        try:
            asyncio.run(coro)
            future.set_result(None)
        except BaseException as exc:
            future.set_exception(exc)
        return future

    pipeline._stage_batch = stage  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._write_cuda_buffer = write  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._submit = submit  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._kv_caches = {}  # noqa: SLF001
    pipeline.queue_finished({"large": _store_spec("large", [0, 1, 2, 3, 4])})
    pipeline._kv_caches = {"layer": torch.empty(1)}  # noqa: SLF001

    assert pipeline.collect_finished({"large"}) == set()
    assert pipeline.collect_finished(set()) == {"large"}
    assert writes == [2, 2, 1]
    assert len(released) == 3


@pytest.mark.asyncio
async def test_packed_store_overlaps_next_stage_with_transfer() -> None:
    """Packed batches use the second staging lease while the first transfers."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._online_packer = object()  # noqa: SLF001
    pipeline._store_capacity = 2  # noqa: SLF001
    batches = [
        (
            [0],
            [StoreWriteSpan(0, 16, 0, "first", 0, 1, 0, 1)],
        ),
        (
            [1],
            [StoreWriteSpan(0, 16, 16, "second", 1, 1, 1, 1)],
        ),
    ]
    pipeline._plan_finished_save = lambda _save: batches  # type: ignore[method-assign]  # noqa: SLF001
    stage_started: list[int] = []
    transfer_started: list[int] = []
    released: list[int] = []

    class Lease:
        def __init__(self, index: int) -> None:
            self.index = index
            self.view = torch.empty(1)

        def release(self) -> None:
            released.append(self.index)

    async def stage(
        block_ids: list[int],
        spans: list[Any],
        event: Any,
    ) -> StagedStoreBatch:
        del spans, event
        index = block_ids[0]
        stage_started.append(index)
        await asyncio.sleep(0)
        return StagedStoreBatch(torch.empty(1), [], Lease(index))

    async def write(staged: StagedStoreBatch) -> list[str]:
        index = staged.lease.index
        transfer_started.append(index)
        await asyncio.sleep(0)
        return []

    pipeline._stage_batch_acquired = stage  # type: ignore[method-assign]  # noqa: SLF001
    pipeline._write_cuda_buffer = write  # type: ignore[method-assign]  # noqa: SLF001
    await pipeline._store_finished_save(SimpleNamespace(), None)  # noqa: SLF001

    assert stage_started == [0, 1]
    assert transfer_started == [0, 1]
    assert released == [0, 1]


class _LoadClient:
    def __init__(self, fail_offset: int | None = None) -> None:
        self.calls: list[int] = []
        self.lease_ids: list[str | None] = []
        self.fail_offset = fail_offset

    async def transfer_load_registered_cuda(self, **kwargs: Any) -> dict[str, Any]:
        offset = int(kwargs["spans"][0]["file_offset"])
        self.calls.append(offset)
        self.lease_ids.append(kwargs.get("lease_id"))
        if offset == self.fail_offset:
            raise RuntimeError("load failed")
        return {
            "transfer_open_ms": 1.0,
            "transfer_load_ms": 2.0,
            "transfer_sync_ms": 3.0,
            "transfer_stats_delta": {"l1_hits": 4, "l1_misses": 5, "l2_reads": 6},
        }

    async def close(self) -> None:
        return None


def _load_spec(key: str, blocks: list[int], offset: int = 0) -> ReqLoadSpec:
    return ReqLoadSpec(key, offset // 16, len(blocks), blocks, offset, len(blocks))


def _packed_load_spec(
    key: str,
    blocks: list[int],
    offset: int = 0,
    *,
    lease_id: str = "",
    pos_offset: int = 0,
) -> ReqLoadSpec:
    return ReqLoadSpec(
        chunk_key=key,
        start_slot=offset // 16,
        num_slots=len(blocks),
        block_ids=blocks,
        file_offset=offset,
        token_count=len(blocks),
        pos_offset=pos_offset,
        lease_id=lease_id,
        compressed_slots=[
            CompressedLoadSlot(
                slot_id=offset // 16 + index,
                mode="compressed",
                file_offset=offset + index * 8,
                stored_length=8,
            )
            for index in range(len(blocks))
        ],
    )


def _load_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    client: _LoadClient,
    staging_pool: FixedCudaStagingPool | None = None,
) -> LoadPipeline:
    monkeypatch.setattr(
        "daser.connector.worker.load.copy_staging_to_kv_cache",
        lambda **kwargs: 1,
    )
    pipeline = LoadPipeline("unused.sock", client_count=2)
    pipeline._clients = [client, client]  # type: ignore[assignment]  # noqa: SLF001
    pipeline.configure(
        kv_caches={"layer": torch.empty(1)},
        layer_names=["layer"],
        local_slot_size=16,
        rank_stride_bytes=0,
        tp_rank=0,
        staging_pool=staging_pool or FixedCudaStagingPool(torch.device("cpu"), 32, 2),
        load_key_scale=1.0,
        load_value_scale=1.0,
        rope_delta_scale=1.0,
        rope_base=10000.0,
        rope_rotary_dim=0,
        rope_is_neox_style=True,
    )
    pipeline._staging_registered = True  # noqa: SLF001
    return pipeline


def _wait_finished(pipeline: LoadPipeline, expected: set[str]) -> set[str]:
    deadline = time.monotonic() + 2.0
    finished: set[str] = set()
    while time.monotonic() < deadline and finished != expected:
        finished.update(pipeline.collect_finished())
        time.sleep(0.005)
    return finished


def test_load_pipeline_handles_empty_and_multibatch_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _LoadClient()
    pipeline = _load_pipeline(monkeypatch, client)
    try:
        pipeline.start(
            {
                "empty": _load_spec("empty", []),
                "large:load:0": ReqLoadSpec(
                    **{
                        **vars(_load_spec("large", [0, 1, 2, 3, 4])),
                        "lease_id": "large",
                    }
                ),
            }
        )
        assert _wait_finished(pipeline, {"empty", "large"}) == {"empty", "large"}
        assert client.calls == [0, 32, 64]
        assert client.lease_ids == ["large", "large", "large"]
    finally:
        pipeline.shutdown()


def test_load_failure_invalidates_only_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _LoadClient(fail_offset=0)
    pipeline = _load_pipeline(monkeypatch, client)
    try:
        pipeline.start(
            {
                "bad": _load_spec("bad", [7], 0),
                "good": _load_spec("good", [8], 16),
            }
        )
        assert _wait_finished(pipeline, {"bad", "good"}) == {"bad", "good"}
        assert pipeline.take_invalid_block_ids() == {7}
        assert sorted(client.calls) == [0, 16]
    finally:
        pipeline.shutdown()


def test_load_transfer_error_returns_staging_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a failed read, every fixed staging lane remains usable."""
    client = _LoadClient(fail_offset=0)
    pipeline = _load_pipeline(monkeypatch, client)
    try:
        pipeline.start({"bad": _load_spec("bad", [7])})
        assert _wait_finished(pipeline, {"bad"}) == {"bad"}
        assert pipeline.take_invalid_block_ids() == {7}
        pipeline.start(
            {
                "next_a": _load_spec("next_a", [8], 16),
                "next_b": _load_spec("next_b", [9], 32),
            }
        )
        assert _wait_finished(pipeline, {"next_a", "next_b"}) == {"next_a", "next_b"}
        assert pipeline.take_invalid_block_ids() == set()
        assert sorted(client.calls) == [0, 16, 32]
    finally:
        pipeline.shutdown()


@pytest.mark.parametrize("shutdown_during_drain", [False, True])
def test_lookahead_failure_waits_for_sibling_before_invalidation(
    monkeypatch: pytest.MonkeyPatch, shutdown_during_drain: bool
) -> None:
    """A failed middle batch retains ownership until speculative IO finishes."""
    sibling_started = threading.Event()
    failure_raised = threading.Event()
    release_sibling = threading.Event()

    class Client(_LoadClient):
        async def transfer_load_registered_cuda(self, **kwargs: Any) -> dict[str, Any]:
            offset = int(kwargs["spans"][0]["file_offset"])
            if offset == 32:
                while not sibling_started.is_set():
                    await asyncio.sleep(0.001)
                failure_raised.set()
                raise RuntimeError("middle batch failed")
            if offset == 64:
                sibling_started.set()
                while not release_sibling.is_set():
                    await asyncio.sleep(0.001)
            return await super().transfer_load_registered_cuda(**kwargs)

    pool = FixedCudaStagingPool(torch.device("cpu"), 32, 2)
    pipeline = _load_pipeline(monkeypatch, Client(), pool)
    stopped = threading.Event()
    stop_thread: threading.Thread | None = None

    def stop() -> None:
        pipeline.shutdown()
        stopped.set()

    try:
        pipeline.start({"failed": _load_spec("failed", list(range(6)))})
        assert failure_raised.wait(timeout=2)
        # Give the dispatcher opportunities to process the injected error.
        # Completion includes errors, so both public surfaces must stay empty
        # until the sibling stops writing the request's staging allocation.
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            assert pipeline.collect_finished() == set()
            assert pipeline.take_invalid_block_ids() == set()
            time.sleep(0.001)
        assert pool.available < pool.depth
        if shutdown_during_drain:
            stop_thread = threading.Thread(target=stop)
            stop_thread.start()
            assert not stopped.wait(timeout=0.05)
        release_sibling.set()
        if stop_thread is not None:
            stop_thread.join(timeout=3)
            assert stopped.is_set()
        else:
            assert _wait_finished(pipeline, {"failed"}) == {"failed"}
            assert pipeline.take_invalid_block_ids() == set(range(6))
            # Reuse both lanes after failure, protecting against the stale
            # active-batch cleanup that can lose a promoted lookahead lease.
            pipeline.start({"next": _load_spec("next", [6, 7, 8, 9], 96)})
            assert _wait_finished(pipeline, {"next"}) == {"next"}
            assert pipeline.take_invalid_block_ids() == set()
        assert pool.available == pool.depth
    finally:
        release_sibling.set()
        if stop_thread is not None:
            stop_thread.join(timeout=3)
        else:
            pipeline.shutdown()


def test_packed_load_coalesces_exact_sources_and_completes_all_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One step reads an immutable source once and restores every destination."""
    client = _LoadClient()
    pipeline = _load_pipeline(monkeypatch, client)
    restore_ranges: list[list[Any]] = []

    def restore(state: Any) -> tuple[int, int, None]:
        restore_ranges.append(state.per_req_ranges)
        return sum(len(item[3].block_ids) for item in state.per_req_ranges), 1, None

    pipeline._restore_batch = restore  # type: ignore[method-assign]  # noqa: SLF001
    requests = {
        req_id: _packed_load_spec("shared", [index * 3, index * 3 + 1, index * 3 + 2])
        for index, req_id in enumerate(("a", "b", "c", "d"))
    }
    try:
        pipeline.start(requests)

        assert len({id(load.future) for load in pipeline._pending.values()}) == 1  # noqa: SLF001
        assert _wait_finished(pipeline, set(requests)) == set(requests)
        assert client.calls == [0]
        assert len(restore_ranges) == 1
        assert len(restore_ranges[0]) == 4
    finally:
        pipeline.shutdown()


def test_packed_load_group_failure_invalidates_all_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared transfer failure completes and invalidates every group member."""
    client = _LoadClient(fail_offset=0)
    pipeline = _load_pipeline(monkeypatch, client)
    requests = {
        "a": _packed_load_spec("shared", [1, 2]),
        "b": _packed_load_spec("shared", [3, 4]),
    }
    try:
        pipeline.start(requests)

        assert _wait_finished(pipeline, set(requests)) == set(requests)
        assert client.calls == [0]
        assert pipeline.take_invalid_block_ids() == {1, 2, 3, 4}
    finally:
        pipeline.shutdown()


@pytest.mark.parametrize("fail_shared", [False, True])
def test_delayed_load_failure_keeps_other_requests_progressing(
    monkeypatch: pytest.MonkeyPatch, fail_shared: bool
) -> None:
    """Delayed transfers preserve errors and let another request make progress."""
    release = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    submissions: list[int] = []

    class DelayedClient(_LoadClient):
        async def transfer_load_registered_cuda(self, **kwargs: Any) -> dict[str, Any]:
            submissions.append(int(kwargs["spans"][0]["file_offset"]))
            first_started.set()
            if len(submissions) >= 2:
                second_started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return await super().transfer_load_registered_cuda(**kwargs)

    client = DelayedClient(fail_offset=0 if fail_shared else None)
    pipeline = _load_pipeline(monkeypatch, client)
    restored: list[int] = []

    def restore(state: Any) -> tuple[int, int, None]:
        blocks = [block for item in state.per_req_ranges for block in item[3].block_ids]
        restored.extend(blocks)
        return len(blocks), 1, None

    pipeline._restore_batch = restore  # type: ignore[method-assign]  # noqa: SLF001
    try:
        pipeline.start({"a": _packed_load_spec("shared", [1, 2, 3])})
        assert first_started.wait(timeout=2)
        pipeline.start(
            {
                "b": _packed_load_spec("other", [4, 5, 6], offset=64),
            }
        )
        # The unrelated request proves that the loop has processed the late
        # worker call while the first source transfer remains blocked.
        assert second_started.wait(timeout=2)
        assert pipeline.collect_finished() == set()
        release.set()
        assert _wait_finished(pipeline, {"a", "b"}) == {"a", "b"}
        assert sorted(submissions) == [0, 64]
        assert sorted(restored) == ([4, 5, 6] if fail_shared else list(range(1, 7)))
        assert pipeline.take_invalid_block_ids() == (
            {1, 2, 3} if fail_shared else set()
        )
    finally:
        release.set()
        pipeline.shutdown()


def test_late_load_after_restore_submission_uses_an_independent_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never mutate decoder metadata after launch or complete targets early."""
    client = _LoadClient()
    pipeline = _load_pipeline(monkeypatch, client)
    restored: list[list[int]] = []
    first_restore = threading.Event()
    second_restore = threading.Event()
    release = threading.Event()

    def restore(state: Any) -> tuple[int, int, Any]:
        restored.append(
            [block for item in state.per_req_ranges for block in item[3].block_ids]
        )
        first_restore.set()
        if len(restored) >= 2:
            second_restore.set()
        return len(restored[-1]), 1, SimpleNamespace(query=release.is_set)

    pipeline._restore_batch = restore  # type: ignore[method-assign]  # noqa: SLF001
    try:
        pipeline.start({"a": _packed_load_spec("shared", [1, 2])})
        assert first_restore.wait(timeout=2)
        pipeline.start({"b": _packed_load_spec("shared", [3, 4])})
        assert second_restore.wait(timeout=2)
        assert pipeline.collect_finished() == set()
        release.set()
        assert _wait_finished(pipeline, {"a", "b"}) == {"a", "b"}
        assert client.calls == [0, 0]
        assert restored == [[1, 2], [3, 4]]
        # Finished also includes failed requests; readiness requires success
        # after the actual asynchronous event-completion path is consumed.
        assert pipeline.take_invalid_block_ids() == set()
    finally:
        release.set()
        pipeline.shutdown()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_packed_load_spec("shared", [1]), _packed_load_spec("other", [2])),
        (
            _packed_load_spec("shared", [1], pos_offset=0),
            _packed_load_spec("shared", [2], pos_offset=1),
        ),
        (_packed_load_spec("shared", [1]), _packed_load_spec("shared", [1])),
        (
            _packed_load_spec("shared", [1], lease_id="a"),
            _packed_load_spec("shared", [2], lease_id="b"),
        ),
        (_load_spec("shared", [1]), _load_spec("shared", [2])),
    ],
)
def test_load_coalescing_rejects_nonidentical_or_owned_requests(
    monkeypatch: pytest.MonkeyPatch,
    first: ReqLoadSpec,
    second: ReqLoadSpec,
) -> None:
    """Source mismatches, overlaps, leases, and raw requests stay independent."""
    client = _LoadClient()
    pipeline = _load_pipeline(monkeypatch, client)
    pipeline._restore_batch = lambda _state: (1, 1, None)  # type: ignore[method-assign]  # noqa: SLF001
    try:
        pipeline.start({"a": first, "b": second})

        assert len({id(load.future) for load in pipeline._pending.values()}) == 2  # noqa: SLF001
        assert _wait_finished(pipeline, {"a", "b"}) == {"a", "b"}
        assert len(client.calls) == 2
    finally:
        pipeline.shutdown()


def test_compressed_decoder_metadata_capacity_follows_destination_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fan-out metadata grows independently from fixed staging bytes."""
    captured: dict[str, Any] = {}

    def decoder(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("daser.connector.worker.load.FusedCompressedKVDecoder", decoder)
    pipeline = LoadPipeline.__new__(LoadPipeline)
    pipeline._staging_pool = SimpleNamespace(buffer_bytes=32, depth=2)  # noqa: SLF001
    pipeline._kv_caches = {"layer": torch.empty((10, 1, 2, 1, 1, 1))}  # noqa: SLF001
    pipeline._local_slot_size = 16  # noqa: SLF001

    pipeline.configure_compression(  # noqa: SLF001
        storage_format="compressed-online",
        codebooks=b"codebook",
        tile_scalars=1024,
    )

    assert captured["max_slots_per_buffer"] == 10
    assert pipeline._staging_pool.buffer_bytes == 32  # noqa: SLF001


@pytest.mark.parametrize(("batch_slots", "expected"), [(2, 2), (85, 4)])
def test_online_pack_batch_is_capped_by_config_and_staging_buffer(
    batch_slots: int, expected: int
) -> None:
    """One codec launch covers the smaller of the configured cap and a buffer."""
    pipeline = StorePipeline("unused.sock", online_pack_batch_slots=batch_slots)
    try:
        assert pipeline.max_online_pack_slots == 0
        pipeline.configure(
            kv_caches={"layer": torch.empty(1)},
            layer_names=["layer"],
            layer_idx_map={"layer": 0},
            local_slot_size=16,
            rank_stride_bytes=0,
            tp_rank=0,
            tp_size=1,
            staging_bytes=64,
            staging_pool=FixedCudaStagingPool(torch.device("cpu"), 64, 2),
        )
        assert pipeline.max_online_pack_slots == expected
    finally:
        pipeline.shutdown()


def test_online_pack_batch_slots_must_be_positive() -> None:
    """A zero codec batch is rejected before any store thread starts."""
    with pytest.raises(ValueError, match="online_pack_batch_slots"):
        StorePipeline("unused.sock", online_pack_batch_slots=0)


@pytest.mark.asyncio
async def test_load_restore_event_is_polled_without_stream_synchronize() -> None:
    """Restore completion waits on a non-blocking CUDA-event query."""
    pipeline = LoadPipeline.__new__(LoadPipeline)
    queries = 0

    class Event:
        def query(self) -> bool:
            nonlocal queries
            queries += 1
            return queries >= 2

    assert await pipeline._wait_for_cuda_event(Event()) >= 0.0  # noqa: SLF001
    assert queries == 2


@pytest.mark.asyncio
async def test_load_dispatcher_waits_for_restore_after_transfer_completion() -> None:
    """Dispatcher progress follows CUDA restore instead of a stale IPC future."""
    pipeline = LoadPipeline.__new__(LoadPipeline)
    transfer_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)
    restore_future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
    state = SimpleNamespace(
        active=SimpleNamespace(
            future=transfer_future,
            restore_future=restore_future,
        )
    )

    asyncio.get_running_loop().call_later(0.001, restore_future.set_result, 0.0)
    await asyncio.wait_for(
        pipeline._wait_for_completion([state]),  # noqa: SLF001
        timeout=0.1,
    )

    assert restore_future.done()
