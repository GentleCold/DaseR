# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import Future
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("cupy")

import torch

from daser.connector.metadata import ReqLoadSpec, ReqStoreSpec
from daser.connector.worker.load import LoadPipeline
from daser.connector.worker.memory import FixedCudaStagingPool
from daser.connector.worker.store import StagedStoreBatch, StorePipeline


class _ManualFuture:
    def __init__(self) -> None:
        self.complete = False

    def done(self) -> bool:
        return self.complete

    def result(self, timeout: float) -> None:
        del timeout


def _store_spec(key: str, blocks: list[int]) -> ReqStoreSpec:
    return ReqStoreSpec(key, 0, len(blocks), blocks, 0, len(blocks))


def test_store_pipeline_dispatches_finished_saves_in_fifo_order() -> None:
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
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


def test_packed_store_captures_latest_producer_event_at_queue_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred packed stores retain the event from the latest worker step."""
    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._pending_finished_saves = {}  # noqa: SLF001
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
        ([0], [_store_spec("first", [0])]),
        ([1], [_store_spec("second", [1])]),
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
        block_ids: list[int], spans: list[Any], event: Any
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


def _load_pipeline(
    monkeypatch: pytest.MonkeyPatch, client: _LoadClient
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
        staging_pool=FixedCudaStagingPool(torch.device("cpu"), 32, 2),
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
    transfer_future: Future[None] = Future()
    transfer_future.set_result(None)
    restore_future: Future[float] = Future()
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
