# SPDX-License-Identifier: Apache-2.0
"""Sparse read completion and buffer lifetime through the public transfer API."""

# Standard
import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
import threading

# Third Party
import pytest

# First Party
from daser.transfer import Move
from daser.transfer.iouring import TieredIOUringTransferLayer
from daser.transfer.iouring.native import NativeIOUring
from tests.transfer.store_files import create_store_file

PAGE = 4096


@pytest.mark.parametrize("cancel", [False, True])
def test_batch_copies_ready_read_before_slow_read_finishes(
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
        if len(reads) == 1:
            return count
        started.set()
        assert release.wait(timeout=10)

        def complete(index: int, nbytes: int) -> None:
            if on_complete is not None:
                on_complete(index + 1, nbytes)

        return count + original(self, fd, reads[1:], complete)

    monkeypatch.setattr(NativeIOUring, "read_batch_into", delayed)

    async def scenario() -> None:
        path = tmp_path / "sparse.store"
        path.write_bytes(b"a" * PAGE + bytes(PAGE) + b"b" * PAGE)
        layer = TieredIOUringTransferLayer(
            create_store_file(str(path), 3 * PAGE),
            l1_bytes=0,
            staging_bytes=2 * PAGE,
            l2_bytes=3 * PAGE,
        )
        dst = bytearray(2 * PAGE)
        moves = [Move(0, PAGE, l2_offset=0), Move(PAGE, PAGE, l2_offset=2 * PAGE)]
        loading = asyncio.create_task(layer.load(dst, moves, "l2"))
        try:

            async def wait_for_first_copy() -> None:
                while not started.is_set() or dst[:PAGE] != b"a" * PAGE:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_for_first_copy(), timeout=5)
            assert not loading.done()
            assert dst[PAGE:] == bytes(PAGE)
            if cancel:
                # A later cancellation must not detach the executor wrapper
                # after one CQE has already exposed a completed sibling.
                for _ in range(3):
                    loading.cancel()
                    await asyncio.sleep(0.02)
                    assert not loading.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await loading
            else:
                await loading
                assert dst == b"a" * PAGE + b"b" * PAGE
                assert layer.stats.l2_reads == 2
            # The cancelled load returned every staging slice.
            assert layer.debug_state()["staging_used_bytes"] == 0
            restored = bytearray(PAGE)
            await layer.load(restored, [Move(0, PAGE, l2_offset=2 * PAGE)], "l2")
            assert restored == b"b" * PAGE
        finally:
            release.set()
            await asyncio.gather(loading, return_exceptions=True)
            await layer.close()

    asyncio.run(scenario())
