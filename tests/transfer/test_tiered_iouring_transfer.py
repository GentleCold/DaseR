# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
import threading
import time

# Third Party
import pytest

from daser.transfer.iouring import TieredIOUringTransferLayer

# First Party
import daser.transfer.iouring.native as native_iouring
from daser.transfer.iouring.native import NativeIOUring

ALIGNMENT = 4096


class DelayedWriteTransferLayer(TieredIOUringTransferLayer):
    """Test transfer layer that can pause selected L2 writes."""

    def __init__(
        self,
        path: str,
        l1_bytes: int,
        l2_bytes: int,
        delayed_offsets: set[int],
    ) -> None:
        super().__init__(
            path=path,
            l1_bytes=l1_bytes,
            l2_bytes=l2_bytes,
        )
        self.delayed_offsets = delayed_offsets
        self.release_write = threading.Event()
        self.write_started = threading.Event()

    def _write_l2(
        self,
        file_offset: int,
        data: object,
        uring: NativeIOUring,
    ) -> None:
        """Pause configured writes before delegating to the real L2 writer."""
        if file_offset in self.delayed_offsets:
            self.delayed_offsets.remove(file_offset)
            self.write_started.set()
            self.release_write.wait(timeout=5.0)
        super()._write_l2(file_offset, data, uring)


class GroupedCopyProbe(TieredIOUringTransferLayer):
    """Test transfer layer that records grouped destination copies."""

    def __init__(self, path: str, l1_bytes: int, l2_bytes: int) -> None:
        super().__init__(
            path=path,
            l1_bytes=l1_bytes,
            l2_bytes=l2_bytes,
        )
        self.grouped_copy_calls = 0

    def _copy_grouped_to_dst(
        self,
        dst: object,
        chunks: list[tuple[int, object, int, int]],
    ) -> None:
        """Record grouped copies before delegating to the production helper."""
        self.grouped_copy_calls += 1
        super()._copy_grouped_to_dst(dst, chunks)


class RecordingReadTransferLayer(TieredIOUringTransferLayer):
    """Test transfer layer that records L2 read offsets and sizes."""

    def __init__(self, path: str, l1_bytes: int, l2_bytes: int) -> None:
        super().__init__(
            path=path,
            l1_bytes=l1_bytes,
            l2_bytes=l2_bytes,
        )
        self.reads: list[tuple[int, int]] = []

    def _read_l2_into(
        self,
        file_offset: int,
        dst: object,
        uring: NativeIOUring,
    ) -> int:
        """Record L2 reads before delegating to the production helper."""
        self.reads.append((file_offset, len(dst)))  # type: ignore[arg-type]
        return super()._read_l2_into(file_offset, dst, uring)


class L2BatchCapacityProbe(TieredIOUringTransferLayer):
    """Test transfer layer exposing the grouped L2 miss batch capacity."""

    def load_l2_miss_batch_capacity(self) -> int:
        """Return the production grouped L2 miss batch capacity."""
        return super()._load_l2_miss_batch_capacity()


def _run(coro: object) -> object:
    """Run a coroutine on the current test event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _block(byte: bytes, size: int = ALIGNMENT) -> bytearray:
    """Return one aligned-size payload block filled with ``byte``."""
    return bytearray(byte * size)


def test_iouring_load_hits_l1_before_l2(tmp_path) -> None:
    """Stored data is readable from L1 immediately before L2 persistence."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 2,
    )
    src = _block(b"a")
    dst = bytearray(ALIGNMENT)

    written = _run(layer.store_bytes(src, file_offset=0, nbytes=ALIGNMENT))
    loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT))

    assert written == ALIGNMENT
    assert loaded == ALIGNMENT
    assert bytes(dst) == bytes(src)
    assert layer.stats.l1_hits == 1
    assert layer.stats.l2_reads == 0
    layer.close()


def test_iouring_l2_uses_native_iouring(tmp_path, monkeypatch) -> None:
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
    layer = TieredIOUringTransferLayer(
        path=path, l1_bytes=ALIGNMENT, l2_bytes=ALIGNMENT * 3
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        _run(layer.drain())
    finally:
        layer.close()

    layer = TieredIOUringTransferLayer(
        path=path, l1_bytes=ALIGNMENT, l2_bytes=ALIGNMENT * 3
    )
    try:
        dst = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT)) == ALIGNMENT
        assert bytes(dst) == bytes(_block(b"a"))
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


def test_iouring_direct_io_aligned_roundtrip(tmp_path) -> None:
    """Production io_uring mode uses O_DIRECT for aligned L2 ranges."""
    path = str(tmp_path / "direct.store")
    try:
        layer = TieredIOUringTransferLayer(
            path=path,
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 2,
        )
    except OSError as exc:
        pytest.skip(f"filesystem does not support O_DIRECT in this test: {exc}")

    try:
        src = _block(b"a")
        assert (
            _run(layer.store_bytes(src, file_offset=0, nbytes=ALIGNMENT)) == ALIGNMENT
        )
        _run(layer.drain())
    finally:
        layer.close()

    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 2,
    )
    try:
        dst = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT)) == ALIGNMENT
        assert bytes(dst) == bytes(_block(b"a"))
        with pytest.raises(ValueError, match="O_DIRECT"):
            _run(layer.store_bytes(bytearray(b"x"), file_offset=1, nbytes=1))
    finally:
        layer.close()


def test_iouring_load_hits_l1_subrange(tmp_path) -> None:
    """Loads can hit a subrange of a larger cached L1 store span."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 2,
        l2_bytes=ALIGNMENT * 3,
    )
    src = _block(b"a") + _block(b"b")
    dst = bytearray(ALIGNMENT)

    _run(layer.store_bytes(src, file_offset=0, nbytes=ALIGNMENT * 2))
    loaded = _run(layer.load_bytes(dst, file_offset=ALIGNMENT, nbytes=ALIGNMENT))

    assert loaded == ALIGNMENT
    assert bytes(dst) == bytes(_block(b"b"))
    assert layer.stats.l1_hits == 1
    assert layer.stats.l2_reads == 0
    layer.close()


def test_iouring_grouped_load_batches_l1_hits(tmp_path) -> None:
    """Grouped L1 loads batch host-to-destination copies."""
    layer = GroupedCopyProbe(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 2,
        l2_bytes=ALIGNMENT * 3,
    )

    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        dst = bytearray(ALIGNMENT * 2)

        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )

        assert loaded == ALIGNMENT * 2
        assert bytes(dst) == bytes(_block(b"a") + _block(b"b"))
        assert layer.stats.l1_hits == 2
        assert layer.stats.l2_reads == 0
        assert layer.grouped_copy_calls == 1
    finally:
        layer.close()


def test_iouring_grouped_load_supports_sliceable_cuda_wrapper(tmp_path) -> None:
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

    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 2,
        l2_bytes=ALIGNMENT * 3,
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        dst = SliceableTarget(ALIGNMENT * 2)

        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )

        assert loaded == ALIGNMENT * 2
        assert bytes(dst.data) == bytes(_block(b"a") + _block(b"b"))
    finally:
        layer.close()


def test_iouring_write_invalidates_overlapping_l1_ranges(tmp_path) -> None:
    """A subrange write invalidates wider cached L1 entries that overlap it."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 2,
        l2_bytes=ALIGNMENT * 3,
    )
    try:
        _run(layer.store_bytes(_block(b"a") + _block(b"b"), 0, ALIGNMENT * 2))
        _run(layer.store_bytes(_block(b"w"), ALIGNMENT, ALIGNMENT))

        dst = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(dst, ALIGNMENT, ALIGNMENT)) == ALIGNMENT
        assert bytes(dst) == bytes(_block(b"w"))
    finally:
        layer.close()


def test_iouring_promotes_l2_miss_to_l1(tmp_path) -> None:
    """L1 eviction falls back to L2 and promotes the bytes back into L1."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 3,
    )

    _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
    _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))

    dst = bytearray(ALIGNMENT)
    loaded = _run(layer.load_bytes(dst, file_offset=ALIGNMENT, nbytes=ALIGNMENT))

    assert loaded == ALIGNMENT
    assert bytes(dst) == bytes(_block(b"b"))
    assert layer.stats.l1_misses == 1
    assert layer.stats.l2_reads == 1
    layer.close()


def test_iouring_grouped_l2_miss_does_not_pollute_l1_scan(tmp_path) -> None:
    """Grouped scan misses do not evict later warm L1 ranges."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 3,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        for idx, byte in enumerate((b"a", b"b", b"c", b"d")):
            _run(
                layer.store_bytes(
                    _block(byte),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            )
        _run(layer.drain())

        first = bytearray(ALIGNMENT)
        assert (
            _run(
                layer.load_bytes_grouped(
                    first,
                    [
                        {
                            "target_offset": 0,
                            "file_offset": ALIGNMENT * 3,
                            "nbytes": ALIGNMENT,
                        }
                    ],
                )
            )
            == ALIGNMENT
        )
        assert bytes(first) == bytes(_block(b"d"))

        second = bytearray(ALIGNMENT)
        assert (
            _run(
                layer.load_bytes_grouped(
                    second,
                    [{"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT}],
                )
            )
            == ALIGNMENT
        )
        assert bytes(second) == bytes(_block(b"a"))
        assert layer.stats.l1_hits == 1
        assert layer.stats.l2_reads == 1
    finally:
        layer.close()


def test_iouring_grouped_l2_misses_coalesce_contiguous_reads(
    tmp_path,
) -> None:
    """Contiguous grouped L2 misses are read through one larger IO."""
    path = str(tmp_path / "daser.store")
    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        for idx, byte in enumerate((b"a", b"b", b"c")):
            _run(
                layer.store_bytes(
                    _block(byte),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            )
        _run(layer.drain())
    finally:
        layer.close()

    layer = RecordingReadTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        dst = bytearray(ALIGNMENT * 3)
        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                    {
                        "target_offset": ALIGNMENT * 2,
                        "file_offset": ALIGNMENT * 2,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )

        assert loaded == ALIGNMENT * 3
        assert bytes(dst[:ALIGNMENT]) == bytes(_block(b"a"))
        assert bytes(dst[ALIGNMENT : ALIGNMENT * 2]) == bytes(_block(b"b"))
        assert bytes(dst[ALIGNMENT * 2 :]) == bytes(_block(b"c"))
        assert layer.reads == [(0, ALIGNMENT * 3)]
        assert layer.stats.l2_reads == 3
    finally:
        layer.close()


def test_iouring_grouped_l2_misses_coalesce_file_contiguous_targets(
    tmp_path,
) -> None:
    """File-contiguous L2 misses coalesce even with non-contiguous targets."""
    path = str(tmp_path / "daser.store")
    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        for idx, byte in enumerate((b"a", b"b", b"c")):
            _run(
                layer.store_bytes(
                    _block(byte),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            )
        _run(layer.drain())
    finally:
        layer.close()

    layer = RecordingReadTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        dst = bytearray(ALIGNMENT * 3)
        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {
                        "target_offset": ALIGNMENT * 2,
                        "file_offset": 0,
                        "nbytes": ALIGNMENT,
                    },
                    {
                        "target_offset": 0,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT * 2,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )

        assert loaded == ALIGNMENT * 3
        assert bytes(dst[:ALIGNMENT]) == bytes(_block(b"b"))
        assert bytes(dst[ALIGNMENT : ALIGNMENT * 2]) == bytes(_block(b"c"))
        assert bytes(dst[ALIGNMENT * 2 :]) == bytes(_block(b"a"))
        assert layer.reads == [(0, ALIGNMENT * 3)]
        assert layer.stats.l2_reads == 3
    finally:
        layer.close()


def test_iouring_grouped_l2_batch_not_limited_by_uring_count(
    tmp_path,
) -> None:
    """More spans than io_uring workers can still become one sequential read."""

    path = str(tmp_path / "daser.store")
    block_count = 12
    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * block_count,
    )
    try:
        for idx in range(block_count):
            _run(
                layer.store_bytes(
                    _block(bytes([idx])),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            )
        _run(layer.drain())
    finally:
        layer.close()

    layer = RecordingReadTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * block_count,
    )
    try:
        dst = bytearray(ALIGNMENT * block_count)
        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {
                        "target_offset": idx * ALIGNMENT,
                        "file_offset": idx * ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    }
                    for idx in range(block_count)
                ],
            )
        )

        assert loaded == ALIGNMENT * block_count
        for idx in range(block_count):
            start = idx * ALIGNMENT
            end = start + ALIGNMENT
            assert bytes(dst[start:end]) == bytes(_block(bytes([idx])))
        assert layer.reads == [(0, ALIGNMENT * block_count)]
        assert layer.stats.l2_reads == block_count
    finally:
        layer.close()


def test_iouring_grouped_l2_batch_uses_bounded_read_window(tmp_path) -> None:
    """Grouped L2 miss scans keep the transient read window bounded."""
    layer = L2BatchCapacityProbe(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT,
    )
    try:
        assert layer.load_l2_miss_batch_capacity() == (256 << 20)
    finally:
        layer.close()


def test_iouring_grouped_l2_misses_can_exceed_l1_capacity(tmp_path) -> None:
    """Grouped scan misses use bounded transient buffers outside L1."""

    async def scenario() -> None:
        path = str(tmp_path / "daser.store")
        layer = TieredIOUringTransferLayer(
            path=path,
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 3,
        )
        try:
            await layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT)
            await layer.store_bytes(
                _block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT
            )
            await layer.drain()
        finally:
            layer.close()

        layer = TieredIOUringTransferLayer(
            path=path,
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 3,
        )
        try:
            dst = bytearray(ALIGNMENT * 2)
            loaded = await asyncio.wait_for(
                layer.load_bytes_grouped(
                    dst,
                    [
                        {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                        {
                            "target_offset": ALIGNMENT,
                            "file_offset": ALIGNMENT,
                            "nbytes": ALIGNMENT,
                        },
                    ],
                ),
                timeout=2.0,
            )
            assert loaded == ALIGNMENT * 2
            assert bytes(dst) == bytes(_block(b"a") + _block(b"b"))
            assert layer.stats.l1_misses == 2
            assert layer.stats.l2_reads == 2
        finally:
            layer.close()

    _run(scenario())


def test_iouring_store_scan_does_not_evict_existing_l1_when_full(tmp_path) -> None:
    """Sequential store scans keep existing L1 entries and persist overflow to L2."""
    path = str(tmp_path / "daser.store")
    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 2,
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))

        first = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(first, file_offset=0, nbytes=ALIGNMENT))
        assert bytes(first) == bytes(_block(b"a"))
        assert layer.stats.l1_hits == 1
        assert layer.stats.l2_reads == 0

        second = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(second, file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        assert bytes(second) == bytes(_block(b"b"))
        assert layer.stats.l2_reads == 1
    finally:
        layer.close()


def test_iouring_store_overflow_load_hits_bounded_spill_cache(tmp_path) -> None:
    """Store overflow remains in a bounded memory spill cache for warm loads."""
    path = str(tmp_path / "daser.store")
    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 3,
        spill_bytes=ALIGNMENT,
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))

        second = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(second, file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        assert bytes(second) == bytes(_block(b"b"))
        assert layer.stats.l2_reads == 0

        _run(
            layer.store_bytes(
                _block(b"c"),
                file_offset=ALIGNMENT * 2,
                nbytes=ALIGNMENT,
            )
        )
        reloaded = bytearray(ALIGNMENT)
        assert _run(layer.load_bytes(reloaded, file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        assert bytes(reloaded) == bytes(_block(b"b"))
        assert layer.stats.l2_reads == 1
    finally:
        layer.close()


def test_iouring_store_overflow_spill_owns_pinned_buffer(tmp_path, monkeypatch) -> None:
    """Overflow spill keeps the transient pinned buffer instead of copying it."""
    path = str(tmp_path / "daser.store")
    layer = TieredIOUringTransferLayer(
        path=path,
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT * 2,
        spill_bytes=ALIGNMENT,
    )
    closed_sizes: list[int] = []

    def record_close(self: object) -> None:
        closed_sizes.append(len(self))  # type: ignore[arg-type]

    monkeypatch.setattr(
        "daser.transfer.iouring.pinned_pool.PinnedMemoryBuffer.close",
        record_close,
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))

        assert closed_sizes == []
    finally:
        layer.close()


def test_iouring_parallel_l2_loads_use_independent_offsets(tmp_path) -> None:
    """Concurrent L2 loads read their requested byte ranges exactly."""

    async def scenario() -> None:
        block_size = 256 * 1024
        block_count = 128
        path = str(tmp_path / "daser.store")
        layer = TieredIOUringTransferLayer(
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

        layer = TieredIOUringTransferLayer(
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


def test_iouring_store_returns_after_l1_before_l2_flush(tmp_path) -> None:
    """Store returns once L1 is readable while L2 persistence continues."""

    async def scenario() -> None:
        layer = DelayedWriteTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 2,
            delayed_offsets={0},
        )
        try:
            start = time.perf_counter()
            written = await asyncio.wait_for(
                layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT),
                timeout=0.2,
            )
            elapsed = time.perf_counter() - start
            assert written == ALIGNMENT
            assert elapsed < 0.2
            await asyncio.wait_for(
                asyncio.to_thread(layer.write_started.wait), timeout=1.0
            )

            dst = bytearray(ALIGNMENT)
            loaded = await layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT)
            assert loaded == ALIGNMENT
            assert bytes(dst) == bytes(_block(b"a"))
        finally:
            layer.release_write.set()
            layer.close()

    _run(scenario())


def test_iouring_store_waits_for_pending_l2_overlap_before_overwrite(
    tmp_path,
) -> None:
    """Overwriting a pending L2 range applies backpressure before persisting."""

    async def scenario() -> None:
        layer = DelayedWriteTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 2,
            delayed_offsets={0},
        )
        try:
            await layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT)
            await asyncio.wait_for(
                asyncio.to_thread(layer.write_started.wait), timeout=1.0
            )
            store_task = asyncio.create_task(
                layer.store_bytes(_block(b"b"), file_offset=0, nbytes=ALIGNMENT)
            )
            await asyncio.sleep(0.05)
            assert not store_task.done()

            layer.release_write.set()
            assert await asyncio.wait_for(store_task, timeout=1.0) == ALIGNMENT
            await layer.drain()

            dst = bytearray(ALIGNMENT)
            assert await layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"b"))
        finally:
            layer.release_write.set()
            layer.close()

    _run(scenario())


def test_iouring_overwrite_waits_for_previous_l2_pool_owner(tmp_path) -> None:
    """A same-span rewrite waits until the old pending writer releases L1 memory."""

    async def scenario() -> None:
        layer = DelayedWriteTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 2,
            delayed_offsets={0},
        )
        try:
            await layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT)
            await asyncio.wait_for(
                asyncio.to_thread(layer.write_started.wait), timeout=1.0
            )
            overwrite = asyncio.create_task(
                layer.store_bytes(_block(b"b"), file_offset=0, nbytes=ALIGNMENT)
            )
            await asyncio.sleep(0.05)
            assert not overwrite.done()

            layer.release_write.set()
            assert await asyncio.wait_for(overwrite, timeout=1.0) == ALIGNMENT
            await layer.drain()

            after_drain = bytearray(ALIGNMENT)
            assert (
                await layer.load_bytes(after_drain, file_offset=0, nbytes=ALIGNMENT)
                == ALIGNMENT
            )
            assert bytes(after_drain) == bytes(_block(b"b"))
        finally:
            layer.release_write.set()
            layer.close()

    _run(scenario())


def test_iouring_rejects_l2_overflow(tmp_path) -> None:
    """Writes beyond the configured L2 capacity are rejected."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT,
        l2_bytes=ALIGNMENT,
    )

    try:
        _run(layer.store_bytes(_block(b"x"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))
    except ValueError as exc:
        assert "exceeds L2 capacity" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    finally:
        layer.close()
