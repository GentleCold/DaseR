# SPDX-License-Identifier: Apache-2.0
"""Hold unleased source pages until delayed destination copies complete."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from daser.transfer.iouring import TieredIOUringTransferLayer, copy_ops


class DeferredDestination:
    """Model a DMA queue whose source pointers are consumed only at completion."""

    def __init__(self) -> None:
        self.data = bytearray(4096)
        self.chunks: list[copy_ops.CopyChunk] = []
        self.submitted = asyncio.Event()
        self.ready = False

    @property
    def done(self) -> bool:
        """Return whether the controlled copy has consumed its source pages."""
        return self.ready

    def complete(self) -> None:
        """Consume borrowed source pages, then publish device completion."""
        if self.ready:
            return
        for offset, source, source_offset, size in self.chunks:
            self.data[offset : offset + size] = source.view()[
                source_offset : source_offset + size
            ]
        self.ready = True


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("outcome", ["success", "cancel", "submit_error"])
def test_delayed_h2d_preserves_unleased_source_during_pool_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resident: bool, outcome: str
) -> None:
    """L1 and L2 sources survive replacement and cancellation until DMA ends."""
    original_copy = copy_ops.copy_grouped_to_dst

    def event(dst: Any, chunks: list[copy_ops.CopyChunk]) -> Any | None:
        return dst if isinstance(dst, DeferredDestination) else None

    def enqueue(dst: Any, chunks: list[copy_ops.CopyChunk]) -> None:
        if isinstance(dst, DeferredDestination):
            dst.chunks.extend(chunks)
            if outcome == "submit_error":
                raise RuntimeError("partial DMA submission")
        else:
            original_copy(dst, chunks)

    def record(dst: DeferredDestination, completion: DeferredDestination) -> None:
        assert dst is completion
        dst.submitted.set()

    monkeypatch.setattr(copy_ops, "destination_copy_event", event)
    monkeypatch.setattr(copy_ops, "copy_grouped_to_dst", enqueue)
    monkeypatch.setattr(copy_ops, "record_destination_copy_event", record)

    async def scenario() -> None:
        path = tmp_path / "copy-lifetime.store"
        path.write_bytes(b"a" * 4096 + b"b" * 4096)
        layer = TieredIOUringTransferLayer(
            path=str(path),
            l1_bytes=4096,
            l2_bytes=8192,
            coalesce_load_misses=True,
        )
        if resident:
            await layer.load_bytes(bytearray(4096), 0, 4096)
        dst = DeferredDestination()
        loading = asyncio.create_task(
            layer.load_bytes_grouped(
                dst, [{"file_offset": 0, "target_offset": 0, "nbytes": 4096}]
            )
        )
        replacing: asyncio.Task[int] | None = None
        try:
            await asyncio.wait_for(dst.submitted.wait(), timeout=5)
            if outcome == "cancel":
                loading.cancel()
            replacing = asyncio.create_task(layer.store_bytes(b"x" * 4096, 0, 4096))
            await asyncio.sleep(0.01)
            assert not loading.done()
            assert not replacing.done()
            dst.complete()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(loading, timeout=5)
            elif outcome == "submit_error":
                with pytest.raises(RuntimeError, match="partial DMA submission"):
                    await asyncio.wait_for(loading, timeout=5)
            else:
                assert await asyncio.wait_for(loading, timeout=5) == 4096
            assert await asyncio.wait_for(replacing, timeout=5) == 4096
            assert dst.data == b"a" * 4096
            later = bytearray(4096)
            assert await layer.load_bytes(later, 0, 4096) == 4096
            assert later == b"x" * 4096
        finally:
            dst.complete()
            await asyncio.gather(
                loading, *([replacing] if replacing else []), return_exceptions=True
            )
            await layer.drain()
            layer.close()

    asyncio.run(scenario())


def test_failed_copy_drains_slow_sibling_before_load_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial enqueue failure cannot end a load with another DMA pending."""
    events: list[DeferredDestination] = []
    original_copy = copy_ops.copy_grouped_to_dst

    def event(dst: Any, chunks: list[copy_ops.CopyChunk]) -> Any | None:
        if not isinstance(dst, DeferredDestination):
            return None
        completion = DeferredDestination()
        completion.data = dst.data
        events.append(completion)
        return completion

    def enqueue(dst: Any, chunks: list[copy_ops.CopyChunk]) -> None:
        if not isinstance(dst, DeferredDestination):
            original_copy(dst, chunks)
            return
        events[-1].chunks.extend(chunks)
        if len(events) == 1:
            raise RuntimeError("first copy submission failed")

    def record(dst: DeferredDestination, completion: DeferredDestination) -> None:
        completion.submitted.set()
        if len(events) == 1:
            completion.complete()
        else:
            dst.submitted.set()

    monkeypatch.setattr(copy_ops, "destination_copy_event", event)
    monkeypatch.setattr(copy_ops, "copy_grouped_to_dst", enqueue)
    monkeypatch.setattr(copy_ops, "record_destination_copy_event", record)

    async def scenario() -> None:
        path = tmp_path / "sibling-copy.store"
        path.write_bytes(b"a" * 4096 + b"-" * 4096 + b"b" * 4096)
        layer = TieredIOUringTransferLayer(
            path=str(path),
            l1_bytes=8192,
            l2_bytes=12288,
            coalesce_load_misses=True,
        )
        dst = DeferredDestination()
        dst.data = bytearray(8192)
        loading = asyncio.create_task(
            layer.load_bytes_grouped(
                dst,
                [
                    {"file_offset": 0, "target_offset": 0, "nbytes": 4096},
                    {"file_offset": 8192, "target_offset": 4096, "nbytes": 4096},
                ],
            )
        )
        try:
            await asyncio.wait_for(dst.submitted.wait(), timeout=5)
            await asyncio.sleep(0.01)
            assert len(events) == 2
            assert not loading.done()
            events[1].complete()
            with pytest.raises(RuntimeError, match="first copy submission failed"):
                await asyncio.wait_for(loading, timeout=5)
            assert dst.data == b"a" * 4096 + b"b" * 4096
        finally:
            for completion in events:
                completion.complete()
            await asyncio.gather(loading, return_exceptions=True)
            await layer.drain()
            layer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("submit_error", [False, True])
def test_deferred_l1_submission_retains_sources_until_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, submit_error: bool
) -> None:
    """Deferred L1 returns before DMA, preserves source pages, and reports errors."""
    original_copy = copy_ops.copy_grouped_to_dst
    original_slice = copy_ops.slice_dst
    original_ptr = copy_ops.cuda_array_ptr

    def event(dst: Any, chunks: list[copy_ops.CopyChunk]) -> Any | None:
        return dst if isinstance(dst, DeferredDestination) else None

    def enqueue(dst: Any, chunks: list[copy_ops.CopyChunk]) -> None:
        if isinstance(dst, DeferredDestination):
            dst.chunks.extend(chunks)
            if submit_error:
                raise RuntimeError("partial DMA submission")
        else:
            original_copy(dst, chunks)

    def record(dst: DeferredDestination, completion: DeferredDestination) -> None:
        assert dst is completion
        dst.submitted.set()

    monkeypatch.setattr(copy_ops, "destination_copy_event", event)
    monkeypatch.setattr(copy_ops, "copy_grouped_to_dst", enqueue)
    monkeypatch.setattr(copy_ops, "record_destination_copy_event", record)
    monkeypatch.setattr(
        copy_ops,
        "slice_dst",
        lambda dst, offset, size: (
            dst
            if isinstance(dst, DeferredDestination)
            else original_slice(dst, offset, size)
        ),
    )
    monkeypatch.setattr(
        copy_ops,
        "cuda_array_ptr",
        lambda dst: 1 if isinstance(dst, DeferredDestination) else original_ptr(dst),
    )

    async def scenario() -> None:
        path = tmp_path / "deferred-l1.store"
        path.write_bytes(b"a" * 4096 + b"b" * 4096)
        layer = TieredIOUringTransferLayer(
            path=str(path), l1_bytes=4096, l2_bytes=8192, coalesce_load_misses=True
        )
        dst = DeferredDestination()
        submission = None
        replacing = None
        try:
            spans = [{"file_offset": 0, "target_offset": 0, "nbytes": 4096}]
            assert await layer.enqueue_l1_load_grouped(dst, spans) is None
            await layer.load_bytes(bytearray(4096), 0, 4096)
            submission_task = asyncio.create_task(
                layer.enqueue_l1_load_grouped(dst, spans)
            )
            await asyncio.wait_for(dst.submitted.wait(), timeout=5)
            if submit_error:
                assert not submission_task.done()
            else:
                submission = await submission_task
                assert submission is not None
                assert submission.bytes == 4096
                assert not submission.completion.done()
            replacing = asyncio.create_task(layer.store_bytes(b"x" * 4096, 0, 4096))
            await asyncio.sleep(0.01)
            assert not replacing.done()
            dst.complete()
            if submit_error:
                with pytest.raises(RuntimeError, match="partial DMA submission"):
                    await submission_task
            else:
                await submission.completion
            assert await replacing == 4096
            assert dst.data == b"a" * 4096
        finally:
            dst.complete()
            await asyncio.gather(
                *([submission.completion] if submission else []),
                *([submission_task] if "submission_task" in locals() else []),
                *([replacing] if replacing else []),
                return_exceptions=True,
            )
            await layer.drain()
            layer.close()

    asyncio.run(scenario())
