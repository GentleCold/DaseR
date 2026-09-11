# SPDX-License-Identifier: Apache-2.0
"""Sparse read completion and buffer lifetime through the public transfer API."""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
import threading

import pytest

from daser.transfer.iouring import TieredIOUringTransferLayer
from daser.transfer.iouring.native import NativeIOUring


@pytest.mark.parametrize("cancel", [False, True])
def test_packed_batch_copies_ready_extent_before_slow_read_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """Early CQEs copy immediately; cancellation drains the remaining IO."""
    original = NativeIOUring.read_batch_into
    release = threading.Event()
    started = threading.Event()

    def delayed(
        self: NativeIOUring,
        fd: int,
        reads: Sequence[tuple[int, memoryview]],
        on_complete: Callable[[int, int], None] | None = None,
    ) -> int:
        count = original(self, fd, reads[:1], on_complete)
        started.set()
        assert release.wait(timeout=10)

        def complete(index: int, nbytes: int) -> None:
            if on_complete is not None:
                on_complete(index + 1, nbytes)

        return count + original(self, fd, reads[1:], complete)

    monkeypatch.setattr(NativeIOUring, "read_batch_into", delayed)

    async def scenario() -> None:
        path = tmp_path / "sparse.store"
        second_offset = 8192
        path.write_bytes(b"a" * 4096 + bytes(second_offset - 4096) + b"b" * 4096)
        layer = TieredIOUringTransferLayer(
            path=str(path),
            l1_bytes=8192,
            l2_bytes=12288,
            coalesce_load_misses=True,
        )
        dst = bytearray(8192)
        loading = asyncio.create_task(
            layer.load_bytes_grouped(
                dst,
                [
                    {"file_offset": 0, "target_offset": 0, "nbytes": 4096},
                    {
                        "file_offset": second_offset,
                        "target_offset": 4096,
                        "nbytes": 4096,
                    },
                ],
            )
        )
        try:

            async def wait_for_first_copy() -> None:
                while not started.is_set() or dst[:4096] != b"a" * 4096:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_for_first_copy(), timeout=5)
            assert not loading.done()
            assert dst[4096:] == bytes(4096)
            if cancel:
                loading.cancel()
                await asyncio.sleep(0.02)
                assert not loading.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await loading
            else:
                assert await loading == 8192
                assert dst == b"a" * 4096 + b"b" * 4096
                assert layer.stats.l2_reads == 2
            # The cancelled request's buffer leases must be usable again.
            restored = bytearray(4096)
            assert await layer.load_bytes(restored, second_offset, 4096) == 4096
            assert restored == b"b" * 4096
        finally:
            release.set()
            await asyncio.gather(loading, return_exceptions=True)
            layer.close()

    asyncio.run(scenario())


def test_ready_packed_reads_copy_before_stale_promotions_are_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent store preserves new L1 bytes and the older captured read."""
    original = NativeIOUring.read_batch_into
    captured = threading.Event()
    release = threading.Event()

    def delayed(
        self: NativeIOUring,
        fd: int,
        reads: Sequence[tuple[int, memoryview]],
        on_complete: Callable[[int, int], None] | None = None,
    ) -> int:
        result = original(self, fd, reads)
        captured.set()
        assert release.wait(timeout=10)
        if on_complete is not None:
            for index, (_offset, view) in enumerate(reads):
                on_complete(index, len(view))
        return result

    monkeypatch.setattr(NativeIOUring, "read_batch_into", delayed)

    async def scenario() -> None:
        path = tmp_path / "stale-batch.store"
        path.write_bytes(b"a" * 4096 + bytes(4096) + b"b" * 4096 + bytes(4096))
        layer = TieredIOUringTransferLayer(
            path=str(path),
            l1_bytes=16384,
            l2_bytes=16384,
            coalesce_load_misses=True,
        )
        spans = [
            {"file_offset": 0, "target_offset": 0, "nbytes": 4096},
            {"file_offset": 8192, "target_offset": 4096, "nbytes": 4096},
        ]
        dst = bytearray(8192)
        loading = asyncio.create_task(layer.load_bytes_grouped(dst, spans))
        try:
            assert await asyncio.to_thread(captured.wait, timeout=5)
            await layer.store_bytes(b"x" * 4096, 0, 4096)
            await layer.store_bytes(b"y" * 4096, 8192, 4096)
            release.set()
            assert await asyncio.wait_for(loading, timeout=5) == 8192
            assert dst == b"a" * 4096 + b"b" * 4096
            await layer.drain()
            restored = bytearray(8192)
            assert await layer.load_bytes_grouped(restored, spans) == 8192
            assert restored == b"x" * 4096 + b"y" * 4096
            assert layer.stats.l2_reads == 2
        finally:
            release.set()
            await asyncio.gather(loading, return_exceptions=True)
            await layer.drain()
            layer.close()

    asyncio.run(scenario())
