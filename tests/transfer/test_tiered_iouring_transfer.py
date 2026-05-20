# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
import threading
import time

from daser.transfer.iouring_pinned import IOUringPinnedTransferLayer

# First Party
import daser.transfer.native_iouring as native_iouring
from daser.transfer.native_iouring import NativeIOUring


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


class GroupedCopyProbe(IOUringPinnedTransferLayer):
    """Test transfer layer that records grouped destination copies."""

    def __init__(self, path: str, l1_bytes: int, l2_bytes: int) -> None:
        super().__init__(path=path, l1_bytes=l1_bytes, l2_bytes=l2_bytes)
        self.grouped_copy_calls = 0

    def _copy_grouped_to_dst(
        self,
        dst: object,
        chunks: list[tuple[int, memoryview]],
    ) -> None:
        """Record grouped copies before delegating to the production helper."""
        self.grouped_copy_calls += 1
        super()._copy_grouped_to_dst(dst, chunks)


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


def test_iouring_pinned_l2_uses_native_iouring(tmp_path, monkeypatch) -> None:
    """L2 persistence and reload use native io_uring instead of pread/pwrite."""
    calls = {"read_into": 0, "write": 0}
    original_read_into = NativeIOUring.read_into
    original_write = NativeIOUring.write

    def forbidden_pread(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("os.pread must not be used")

    def forbidden_pwrite(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("os.pwrite must not be used")

    def tracked_read_into(
        self: NativeIOUring,
        fd: int,
        file_offset: int,
        dst: memoryview,
    ) -> int:
        calls["read_into"] += 1
        return original_read_into(self, fd, file_offset, dst)

    def tracked_write(
        self: NativeIOUring,
        fd: int,
        file_offset: int,
        data: bytes,
    ) -> int:
        calls["write"] += 1
        return original_write(self, fd, file_offset, data)

    monkeypatch.setattr("os.pread", forbidden_pread)
    monkeypatch.setattr("os.pwrite", forbidden_pwrite)
    monkeypatch.setattr(NativeIOUring, "read_into", tracked_read_into)
    monkeypatch.setattr(NativeIOUring, "write", tracked_write)

    path = str(tmp_path / "daser.store")
    layer = IOUringPinnedTransferLayer(path=path, l1_bytes=8, l2_bytes=64)
    try:
        _run(layer.store_bytes(bytearray(b"abcdefgh"), file_offset=0, nbytes=8))
        _run(layer.store_bytes(bytearray(b"ijklmnop"), file_offset=8, nbytes=8))
        _run(layer.drain())
    finally:
        layer.close()

    layer = IOUringPinnedTransferLayer(path=path, l1_bytes=8, l2_bytes=64)
    try:
        dst = bytearray(8)
        assert _run(layer.load_bytes(dst, file_offset=0, nbytes=8)) == 8
        assert bytes(dst) == b"abcdefgh"
    finally:
        layer.close()
    assert calls == {"read_into": 1, "write": 2}


def test_native_iouring_splits_large_positioned_io(tmp_path, monkeypatch) -> None:
    """Native io_uring splits reads and writes above the kernel IO cap."""
    monkeypatch.setattr(native_iouring, "_MAX_RW_COUNT", 5)
    path = tmp_path / "split.store"
    path.write_bytes(b"\x00" * 16)
    fd = os.open(path, os.O_RDWR)
    uring = NativeIOUring(entries=8)
    try:
        assert uring.write(fd, 1, b"abcdefghijkl") == 12
        assert uring.read(fd, 1, 12) == b"abcdefghijkl"
        assert path.read_bytes()[1:13] == b"abcdefghijkl"
    finally:
        uring.close()
        os.close(fd)


def test_native_iouring_read_into(tmp_path) -> None:
    """Native io_uring can read directly into caller-owned buffers."""
    path = tmp_path / "read_into.store"
    path.write_bytes(b"abcdefghijkl")
    fd = os.open(path, os.O_RDWR)
    uring = NativeIOUring(entries=8)
    dst = bytearray(6)
    try:
        assert uring.read_into(fd, 3, memoryview(dst)) == 6
        assert bytes(dst) == b"defghi"
    finally:
        uring.close()
        os.close(fd)


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


def test_iouring_pinned_grouped_load_batches_l1_hits(tmp_path) -> None:
    """Grouped L1 loads batch host-to-destination copies."""
    layer = GroupedCopyProbe(
        path=str(tmp_path / "daser.store"),
        l1_bytes=16,
        l2_bytes=64,
    )

    try:
        _run(layer.store_bytes(bytearray(b"abcdefgh"), file_offset=0, nbytes=8))
        _run(layer.store_bytes(bytearray(b"ijklmnop"), file_offset=8, nbytes=8))
        dst = bytearray(16)

        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": 8},
                    {"target_offset": 8, "file_offset": 8, "nbytes": 8},
                ],
            )
        )

        assert loaded == 16
        assert bytes(dst) == b"abcdefghijklmnop"
        assert layer.stats.l1_hits == 2
        assert layer.stats.l2_reads == 0
        assert layer.grouped_copy_calls == 1
    finally:
        layer.close()


def test_iouring_pinned_grouped_load_supports_sliceable_cuda_wrapper(tmp_path) -> None:
    """Grouped L1 loads can target CUDA wrapper objects from IPC."""

    class TargetSlice:
        def __init__(self, parent: "SliceableTarget", start: int, stop: int) -> None:
            self._parent = parent
            self._start = start
            self._stop = stop

        def set(self, data: object) -> None:
            """Copy numpy-backed data into the parent buffer."""
            self._parent.data[self._start : self._stop] = memoryview(data).cast("B")

    class SliceableTarget:
        def __init__(self, size: int) -> None:
            self.data = bytearray(size)

        def __getitem__(self, item: slice) -> TargetSlice:
            """Return a settable slice without exposing ``set`` on self."""
            return TargetSlice(self, int(item.start or 0), int(item.stop or 0))

    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=16,
        l2_bytes=64,
    )
    try:
        _run(layer.store_bytes(bytearray(b"abcdefgh"), file_offset=0, nbytes=8))
        _run(layer.store_bytes(bytearray(b"ijklmnop"), file_offset=8, nbytes=8))
        dst = SliceableTarget(16)

        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": 8},
                    {"target_offset": 8, "file_offset": 8, "nbytes": 8},
                ],
            )
        )

        assert loaded == 16
        assert bytes(dst.data) == b"abcdefghijklmnop"
    finally:
        layer.close()


def test_iouring_pinned_write_invalidates_overlapping_l1_ranges(tmp_path) -> None:
    """A subrange write invalidates wider cached L1 entries that overlap it."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=32,
        l2_bytes=64,
    )
    try:
        _run(layer.store_bytes(bytearray(b"abcdefghijklmnop"), 0, 16))
        _run(layer.store_bytes(bytearray(b"WXYZ"), 4, 4))

        dst = bytearray(4)
        assert _run(layer.load_bytes(dst, 4, 4)) == 4
        assert bytes(dst) == b"WXYZ"
    finally:
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
    """L2 reload waits for a pending write covering the requested range."""

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
