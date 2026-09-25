# SPDX-License-Identifier: Apache-2.0
"""Protect pinned store destinations while executor copies are still running."""

import asyncio
from pathlib import Path
import threading
from typing import Any

import pytest

from daser.transfer.iouring import TieredIOUringTransferLayer, copy_ops
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice


@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "cancel_twice"])
def test_grouped_store_drains_snapshot_before_releasing_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A pending copy survives cancellation and wakes stores waiting for its pages."""
    started = threading.Event()
    release = threading.Event()
    ended = threading.Event()
    original = copy_ops.copy_src_to_pinned

    def copy(
        src: Any, pinned: PinnedMemorySlice, target_offset: int, nbytes: int
    ) -> None:
        if bytes(src[:1]) == b"a":
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test copy was not released")
            # Access the actual public view only when the simulated DMA ends.
            # Closing/recycling the allocation before this point is invalid.
            assert len(pinned.view()) == nbytes
            original(src, pinned, target_offset, nbytes)
            ended.set()
            if outcome == "error":
                raise RuntimeError("snapshot copy failed after writing")
        else:
            original(src, pinned, target_offset, nbytes)

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            path=str(tmp_path / "store-lifetime.store"),
            l1_bytes=4096,
            l2_bytes=8192,
            coalesce_load_misses=True,
        )
        first = [{"source_offset": 0, "file_offset": 0, "nbytes": 4096, "packed": True}]
        second = [
            {"source_offset": 0, "file_offset": 4096, "nbytes": 4096, "packed": True}
        ]
        storing = asyncio.create_task(layer.store_bytes_grouped(b"a" * 4096, first))
        replacing: asyncio.Task[int] | None = None
        try:
            assert await asyncio.to_thread(started.wait, 5)
            replacing = asyncio.create_task(
                layer.store_bytes_grouped(b"b" * 4096, second)
            )
            await asyncio.sleep(0.01)
            assert not replacing.done()
            if outcome.startswith("cancel"):
                storing.cancel()
                await asyncio.sleep(0.01)
                assert not storing.done()
                if outcome == "cancel_twice":
                    storing.cancel()
                    await asyncio.sleep(0.01)
                    assert not storing.done()
            assert not ended.is_set()
            release.set()
            if outcome.startswith("cancel"):
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(storing, 5)
            elif outcome == "error":
                with pytest.raises(RuntimeError, match="snapshot copy failed"):
                    await asyncio.wait_for(storing, 5)
            else:
                assert await asyncio.wait_for(storing, 5) == 4096
            assert ended.is_set()
            assert await asyncio.wait_for(replacing, 5) == 4096
            restored = bytearray(4096)
            assert await layer.load_bytes(restored, 4096, 4096) == 4096
            assert restored == b"b" * 4096
        finally:
            release.set()
            await asyncio.gather(
                storing, *([replacing] if replacing else []), return_exceptions=True
            )
            await layer.drain()
            layer.close()

    asyncio.run(scenario())


def test_grouped_store_streams_through_smaller_l1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contiguous source larger than L1 remains storable without a larger pool."""
    copy_sizes: list[int] = []
    original = copy_ops.copy_src_to_pinned

    def copy(
        src: Any, pinned: PinnedMemorySlice, target_offset: int, nbytes: int
    ) -> None:
        copy_sizes.append(nbytes)
        original(src, pinned, target_offset, nbytes)

    monkeypatch.setattr(copy_ops, "copy_src_to_pinned", copy)

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            path=str(tmp_path / "small-pool.store"),
            l1_bytes=8192,
            l2_bytes=24576,
            coalesce_load_misses=True,
        )
        source = b"a" * 4096 + b"b" * 4096 + b"c" * 4096
        spans = [
            {
                "source_offset": i * 4096,
                "file_offset": i * 8192,
                "nbytes": 4096,
                "packed": True,
            }
            for i in range(3)
        ]
        try:
            assert await asyncio.wait_for(
                layer.store_bytes_grouped(source, spans), 5
            ) == len(source)
            assert copy_sizes == [8192, 4096]
            await layer.drain()
            for i in range(3):
                restored = bytearray(4096)
                assert await layer.load_bytes(restored, i * 8192, 4096) == 4096
                assert restored == source[i * 4096 : (i + 1) * 4096]
        finally:
            await layer.drain()
            layer.close()

    asyncio.run(scenario())
