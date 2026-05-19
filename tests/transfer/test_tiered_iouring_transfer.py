# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import threading
import time

# First Party
from daser.transfer.iouring_pinned import IOUringPinnedTransferLayer


class DelayedWriteTransferLayer(IOUringPinnedTransferLayer):
    """Test transfer layer that can pause selected L2 writes."""

    def __init__(
        self,
        path: str,
        l1_bytes: int,
        l2_bytes: int,
        delayed_offsets: set[int],
    ) -> None:
        super().__init__(path=path, l1_bytes=l1_bytes, l2_bytes=l2_bytes)
        self.delayed_offsets = delayed_offsets
        self.release_write = threading.Event()
        self.write_started = threading.Event()

    def _write_l2(self, file_offset: int, data: bytes) -> None:
        """Pause configured writes before delegating to the real L2 writer."""
        if file_offset in self.delayed_offsets:
            self.delayed_offsets.remove(file_offset)
            self.write_started.set()
            self.release_write.wait(timeout=5.0)
        super()._write_l2(file_offset, data)


def _run(coro: object) -> object:
    """Run a coroutine on the current test event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def test_iouring_pinned_load_hits_l1_before_l2(tmp_path) -> None:
    """Stored data is readable from L1 immediately before L2 persistence."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=16,
        l2_bytes=64,
    )
    src = bytearray(b"abcdefgh")
    dst = bytearray(8)

    written = _run(layer.store_bytes(src, file_offset=0, nbytes=8))
    loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=8))

    assert written == 8
    assert loaded == 8
    assert bytes(dst) == b"abcdefgh"
    assert layer.stats.l1_hits == 1
    assert layer.stats.l2_reads == 0
    layer.close()


def test_iouring_pinned_load_hits_l1_subrange(tmp_path) -> None:
    """Loads can hit a subrange of a larger cached L1 store span."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=16,
        l2_bytes=64,
    )
    src = bytearray(b"abcdefghijklmnop")
    dst = bytearray(4)

    _run(layer.store_bytes(src, file_offset=0, nbytes=16))
    loaded = _run(layer.load_bytes(dst, file_offset=4, nbytes=4))

    assert loaded == 4
    assert bytes(dst) == b"efgh"
    assert layer.stats.l1_hits == 1
    assert layer.stats.l2_reads == 0
    layer.close()


def test_iouring_pinned_promotes_l2_miss_to_l1(tmp_path) -> None:
    """L1 eviction falls back to L2 and promotes the bytes back into L1."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=8,
        l2_bytes=64,
    )

    _run(layer.store_bytes(bytearray(b"aaaaaaaa"), file_offset=0, nbytes=8))
    _run(layer.store_bytes(bytearray(b"bbbbbbbb"), file_offset=8, nbytes=8))

    dst = bytearray(8)
    loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=8))

    assert loaded == 8
    assert bytes(dst) == b"aaaaaaaa"
    assert layer.stats.l1_misses == 1
    assert layer.stats.l2_reads == 1
    layer.close()


def test_iouring_pinned_parallel_l2_loads_use_independent_offsets(tmp_path) -> None:
    """Concurrent L2 loads read their requested byte ranges exactly."""

    async def scenario() -> None:
        block_size = 256 * 1024
        block_count = 128
        path = str(tmp_path / "daser.store")
        layer = IOUringPinnedTransferLayer(
            path=path,
            l1_bytes=block_size,
            l2_bytes=block_size * block_count,
        )
        try:
            for i in range(block_count):
                await layer.store_bytes(
                    bytearray([i]) * block_size,
                    file_offset=i * block_size,
                    nbytes=block_size,
                )
        finally:
            layer.close()

        layer = IOUringPinnedTransferLayer(
            path=path,
            l1_bytes=block_size,
            l2_bytes=block_size * block_count,
        )
        try:

            async def load_block(i: int) -> tuple[int, bytes]:
                dst = bytearray(block_size)
                await layer.load_bytes(
                    dst,
                    file_offset=i * block_size,
                    nbytes=block_size,
                )
                return i, bytes(dst)

            results = await asyncio.gather(*(load_block(i) for i in range(block_count)))
        finally:
            layer.close()

        for i, data in results:
            assert data == bytes([i]) * block_size

    _run(scenario())


def test_iouring_pinned_store_returns_after_l1_before_l2_flush(tmp_path) -> None:
    """Store returns once L1 is readable while L2 persistence continues."""

    async def scenario() -> None:
        layer = DelayedWriteTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=8,
            l2_bytes=32,
            delayed_offsets={0},
        )
        try:
            start = time.perf_counter()
            written = await asyncio.wait_for(
                layer.store_bytes(bytearray(b"abcdefgh"), file_offset=0, nbytes=8),
                timeout=0.2,
            )
            elapsed = time.perf_counter() - start
            assert written == 8
            assert elapsed < 0.2
            await asyncio.wait_for(
                asyncio.to_thread(layer.write_started.wait), timeout=1.0
            )

            dst = bytearray(8)
            loaded = await layer.load_bytes(dst, file_offset=0, nbytes=8)
            assert loaded == 8
            assert bytes(dst) == b"abcdefgh"
        finally:
            layer.release_write.set()
            layer.close()

    _run(scenario())


def test_iouring_pinned_load_waits_for_pending_l2_write_after_l1_eviction(
    tmp_path,
) -> None:
    """L2 fallback waits for a pending write covering the requested range."""

    async def scenario() -> None:
        layer = DelayedWriteTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=8,
            l2_bytes=32,
            delayed_offsets={0},
        )
        try:
            await layer.store_bytes(bytearray(b"aaaaaaaa"), file_offset=0, nbytes=8)
            await asyncio.wait_for(
                asyncio.to_thread(layer.write_started.wait), timeout=1.0
            )
            await layer.store_bytes(bytearray(b"bbbbbbbb"), file_offset=8, nbytes=8)

            dst = bytearray(8)
            load_task = asyncio.create_task(
                layer.load_bytes(dst, file_offset=0, nbytes=8)
            )
            await asyncio.sleep(0.05)
            assert not load_task.done()

            layer.release_write.set()
            loaded = await asyncio.wait_for(load_task, timeout=1.0)
            assert loaded == 8
            assert bytes(dst) == b"aaaaaaaa"
        finally:
            layer.release_write.set()
            layer.close()

    _run(scenario())


def test_iouring_pinned_overwrite_waits_for_previous_l2_write(tmp_path) -> None:
    """A same-span rewrite keeps newer L1 data visible before old L2 drains."""

    async def scenario() -> None:
        layer = DelayedWriteTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=8,
            l2_bytes=32,
            delayed_offsets={0},
        )
        try:
            await layer.store_bytes(bytearray(b"aaaaaaaa"), file_offset=0, nbytes=8)
            await asyncio.wait_for(
                asyncio.to_thread(layer.write_started.wait), timeout=1.0
            )
            await layer.store_bytes(bytearray(b"bbbbbbbb"), file_offset=0, nbytes=8)

            dst = bytearray(8)
            assert await layer.load_bytes(dst, file_offset=0, nbytes=8) == 8
            assert bytes(dst) == b"bbbbbbbb"

            await layer.store_bytes(bytearray(b"cccccccc"), file_offset=8, nbytes=8)
            load_task = asyncio.create_task(
                layer.load_bytes(bytearray(8), file_offset=0, nbytes=8)
            )
            await asyncio.sleep(0.05)
            assert not load_task.done()

            layer.release_write.set()
            assert await asyncio.wait_for(load_task, timeout=1.0) == 8
            await layer.drain()

            after_drain = bytearray(8)
            assert await layer.load_bytes(after_drain, file_offset=0, nbytes=8) == 8
            assert bytes(after_drain) == b"bbbbbbbb"
        finally:
            layer.release_write.set()
            layer.close()

    _run(scenario())


def test_iouring_pinned_rejects_l2_overflow(tmp_path) -> None:
    """Writes beyond the configured L2 capacity are rejected."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=8,
        l2_bytes=8,
    )

    try:
        _run(layer.store_bytes(bytearray(b"xx"), file_offset=7, nbytes=2))
    except ValueError as exc:
        assert "exceeds L2 capacity" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    finally:
        layer.close()
