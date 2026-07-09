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
from daser.transfer.iouring.pinned_pool import PinnedMemorySlice

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


class L2ReadProbe(TieredIOUringTransferLayer):
    """Test transfer layer that records L2 read byte ranges."""

    def __init__(self, path: str, l1_bytes: int, l2_bytes: int) -> None:
        super().__init__(
            path=path,
            l1_bytes=l1_bytes,
            l2_bytes=l2_bytes,
        )
        self.l2_read_ranges: list[tuple[int, int]] = []

    def _read_l2_into(
        self,
        file_offset: int,
        dst: object,
        uring: NativeIOUring,
    ) -> int:
        """Record L2 reads before delegating to the production helper."""
        self.l2_read_ranges.append((file_offset, len(dst)))
        return super()._read_l2_into(file_offset, dst, uring)


class L2WriteBatchProbe(TieredIOUringTransferLayer):
    """Test transfer layer that records grouped L2 write batches."""

    def __init__(self, path: str, l1_bytes: int, l2_bytes: int) -> None:
        super().__init__(
            path=path,
            l1_bytes=l1_bytes,
            l2_bytes=l2_bytes,
        )
        self.l2_write_batches: list[list[tuple[int, int]]] = []

    def _write_l2_batch(
        self,
        entries: list[tuple[tuple[int, int], int, PinnedMemorySlice]],
        uring: NativeIOUring,
    ) -> None:
        """Record grouped L2 writes before delegating to production IO."""
        self.l2_write_batches.append(
            [(file_offset, len(data)) for _key, file_offset, data in entries]
        )
        super()._write_l2_batch(entries, uring)


class DelayedL2ReadProbe(TieredIOUringTransferLayer):
    """Test transfer layer that pauses selected L2 reads."""

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
        self.release_read = threading.Event()
        self.read_started = threading.Event()
        self.copy_offsets: list[int] = []

    def _read_l2_into(
        self,
        file_offset: int,
        dst: object,
        uring: NativeIOUring,
    ) -> int:
        """Pause configured reads before delegating to the real L2 reader."""
        if file_offset in self.delayed_offsets:
            self.delayed_offsets.remove(file_offset)
            self.read_started.set()
            self.release_read.wait(timeout=5.0)
        return super()._read_l2_into(file_offset, dst, uring)

    def _copy_grouped_to_dst(
        self,
        dst: object,
        chunks: list[tuple[int, object, int, int]],
    ) -> None:
        """Record destination copy offsets before delegating."""
        self.copy_offsets.extend(int(chunk[0]) for chunk in chunks)
        super()._copy_grouped_to_dst(dst, chunks)


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
        dst = bytearray(12)
        assert uring.read_into(fd, 1, memoryview(dst)) == 12
        assert bytes(dst) == b"abcdefghijkl"
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


def test_iouring_load_stitches_adjacent_l1_entries(tmp_path) -> None:
    """Single-span loads can stitch adjacent L1 entries without L2 reads."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 3,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        dst = bytearray(ALIGNMENT * 2)

        loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT * 2))

        assert loaded == ALIGNMENT * 2
        assert bytes(dst) == bytes(_block(b"a") + _block(b"b"))
        assert layer.stats.l1_hits == 2
        assert layer.stats.l1_misses == 0
        assert layer.stats.l2_reads == 0
    finally:
        layer.close()


def test_iouring_grouped_load_stitches_adjacent_l1_entries(tmp_path) -> None:
    """A requested span can be satisfied by adjacent L1 entries."""
    layer = TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 3,
        l2_bytes=ALIGNMENT * 4,
    )
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT))
        dst = bytearray(ALIGNMENT * 2)

        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [{"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT * 2}],
            )
        )

        assert loaded == ALIGNMENT * 2
        assert bytes(dst) == bytes(_block(b"a") + _block(b"b"))
        assert layer.stats.l1_hits == 2
        assert layer.stats.l1_misses == 0
        assert layer.stats.l2_reads == 0
    finally:
        layer.close()


def test_iouring_grouped_load_reads_only_l1_gaps_from_l2(tmp_path) -> None:
    """Mixed L1/L2 grouped loads read only missing subranges from L2."""

    async def scenario() -> None:
        path = str(tmp_path / "daser.store")
        layer = L2ReadProbe(
            path=path,
            l1_bytes=ALIGNMENT * 3,
            l2_bytes=ALIGNMENT * 4,
        )
        try:
            await layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT)
            await layer.store_bytes(
                _block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT
            )
            await layer.store_bytes(
                _block(b"c"), file_offset=ALIGNMENT * 2, nbytes=ALIGNMENT
            )
            await layer.drain()
        finally:
            layer.close()

        layer = L2ReadProbe(
            path=path,
            l1_bytes=ALIGNMENT * 3,
            l2_bytes=ALIGNMENT * 4,
        )
        try:
            await layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT)
            await layer.store_bytes(
                _block(b"c"), file_offset=ALIGNMENT * 2, nbytes=ALIGNMENT
            )
            dst = bytearray(ALIGNMENT * 3)

            loaded = await layer.load_bytes_grouped(
                dst,
                [{"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT * 3}],
            )

            assert loaded == ALIGNMENT * 3
            assert bytes(dst) == bytes(_block(b"a") + _block(b"b") + _block(b"c"))
            assert layer.stats.l1_hits == 2
            assert layer.stats.l1_misses == 1
            assert layer.stats.l2_reads == 1
            assert layer.l2_read_ranges == [(ALIGNMENT, ALIGNMENT)]
        finally:
            layer.close()

    _run(scenario())


def test_iouring_grouped_load_batches_l1_hits(tmp_path) -> None:
    """Grouped L1 loads batch host-to-destination copies."""
    layer = GroupedCopyProbe(
        path=str(tmp_path / "daser.store"),
        l1_bytes=ALIGNMENT * 3,
        l2_bytes=ALIGNMENT * 4,
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


def test_iouring_store_admission_leaves_promotion_headroom(tmp_path) -> None:
    """Tiered stores keep a small promotion headroom below full L1 capacity."""

    async def scenario() -> None:
        layer = TieredIOUringTransferLayer(
            path=str(tmp_path / "daser.store"),
            l1_bytes=ALIGNMENT * 20,
            l2_bytes=ALIGNMENT * 24,
        )
        try:
            for idx in range(20):
                await layer.store_bytes(
                    _block(bytes([idx])),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            await layer.drain()
            assert layer.l1_bytes_used == ALIGNMENT * 19
        finally:
            layer.close()

    _run(scenario())


def test_iouring_grouped_store_trims_request_suffix_first(tmp_path) -> None:
    """Forward physical stores keep prefix slots resident after high-water trim."""

    async def scenario() -> None:
        layer = L2ReadProbe(
            path=str(tmp_path / "daser.store"),
            l1_bytes=ALIGNMENT * 5,
            l2_bytes=ALIGNMENT * 8,
        )
        try:
            src = b"".join(_block(byte) for byte in (b"a", b"b", b"c", b"d", b"e"))
            await layer.store_bytes_grouped(
                bytearray(src),
                [
                    {
                        "source_offset": idx * ALIGNMENT,
                        "file_offset": idx * ALIGNMENT,
                        "nbytes": ALIGNMENT,
                        "chunk_key": "req",
                        "start_slot": 0,
                        "num_slots": 5,
                    }
                    for idx in range(5)
                ],
            )
            await layer.drain()

            assert layer.l1_bytes_used == ALIGNMENT * 4
            dst = bytearray(ALIGNMENT)
            for idx, byte in enumerate((b"a", b"b", b"c", b"d")):
                await layer.load_bytes(dst, idx * ALIGNMENT, ALIGNMENT)
                assert bytes(dst) == bytes(_block(byte))
            await layer.load_bytes(dst, ALIGNMENT * 4, ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"e"))
            assert layer.l2_read_ranges == [(ALIGNMENT * 4, ALIGNMENT)]
        finally:
            layer.close()

    _run(scenario())


def test_iouring_grouped_store_batches_adjacent_l2_writes(tmp_path) -> None:
    """Grouped stores keep L1 slots separate while batching adjacent L2 writes."""

    async def scenario() -> None:
        layer = L2WriteBatchProbe(
            path=str(tmp_path / "daser.store"),
            l1_bytes=ALIGNMENT * 8,
            l2_bytes=ALIGNMENT * 8,
        )
        try:
            src = b"".join(_block(byte) for byte in (b"a", b"b", b"c", b"d"))
            stored = await layer.store_bytes_grouped(
                bytearray(src),
                [
                    {
                        "source_offset": idx * ALIGNMENT,
                        "file_offset": idx * ALIGNMENT,
                        "nbytes": ALIGNMENT,
                        "chunk_key": "req",
                        "start_slot": 0,
                        "num_slots": 4,
                    }
                    for idx in range(4)
                ],
            )
            await layer.drain()

            assert stored == ALIGNMENT * 4
            assert layer.l2_write_batches == [
                [(idx * ALIGNMENT, ALIGNMENT) for idx in range(4)]
            ]

            dst = bytearray(ALIGNMENT * 4)
            await layer.load_bytes_grouped(
                dst,
                [
                    {
                        "target_offset": idx * ALIGNMENT,
                        "file_offset": idx * ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    }
                    for idx in range(4)
                ],
            )
            assert bytes(dst) == bytes(src)
        finally:
            layer.close()

    _run(scenario())


def test_iouring_l2_promotion_uses_headroom_then_evicts(tmp_path) -> None:
    """L2 misses promote into headroom and evict old residents once full."""

    async def scenario() -> None:
        path = str(tmp_path / "daser.store")
        writer = TieredIOUringTransferLayer(
            path=path,
            l1_bytes=ALIGNMENT * 5,
            l2_bytes=ALIGNMENT * 8,
        )
        try:
            for idx, byte in enumerate((b"a", b"b", b"c", b"d", b"e", b"f")):
                await writer.store_bytes(
                    _block(byte),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            await writer.drain()
        finally:
            writer.close()

        layer = L2ReadProbe(
            path=path,
            l1_bytes=ALIGNMENT * 5,
            l2_bytes=ALIGNMENT * 8,
        )
        try:
            for idx, byte in enumerate((b"a", b"b", b"c", b"d")):
                await layer.store_bytes(
                    _block(byte),
                    file_offset=idx * ALIGNMENT,
                    nbytes=ALIGNMENT,
                )
            await layer.drain()
            assert layer.l1_bytes_used == ALIGNMENT * 4

            dst = bytearray(ALIGNMENT)
            await layer.load_bytes(dst, file_offset=ALIGNMENT * 4, nbytes=ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"e"))
            assert layer.l1_bytes_used == ALIGNMENT * 5
            assert layer.l2_read_ranges == [(ALIGNMENT * 4, ALIGNMENT)]

            await layer.load_bytes(dst, file_offset=ALIGNMENT * 5, nbytes=ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"f"))
            assert layer.l1_bytes_used == ALIGNMENT * 5
            assert layer.l2_read_ranges == [
                (ALIGNMENT * 4, ALIGNMENT),
                (ALIGNMENT * 5, ALIGNMENT),
            ]

            await layer.load_bytes(dst, file_offset=ALIGNMENT, nbytes=ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"b"))
            assert layer.l2_read_ranges == [
                (ALIGNMENT * 4, ALIGNMENT),
                (ALIGNMENT * 5, ALIGNMENT),
            ]

            await layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"a"))
            assert layer.l2_read_ranges == [
                (ALIGNMENT * 4, ALIGNMENT),
                (ALIGNMENT * 5, ALIGNMENT),
            ]

            await layer.load_bytes(dst, file_offset=ALIGNMENT * 4, nbytes=ALIGNMENT)
            assert bytes(dst) == bytes(_block(b"e"))
            assert layer.l2_read_ranges == [
                (ALIGNMENT * 4, ALIGNMENT),
                (ALIGNMENT * 5, ALIGNMENT),
                (ALIGNMENT * 4, ALIGNMENT),
            ]
        finally:
            layer.close()

    _run(scenario())


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
    loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT))

    assert loaded == ALIGNMENT
    assert bytes(dst) == bytes(_block(b"a"))
    assert layer.stats.l1_misses == 1
    assert layer.stats.l2_reads == 1
    layer.close()


def test_iouring_grouped_l2_misses_are_bounded_by_l1_capacity(tmp_path) -> None:
    """Grouped L2 misses make progress when the request is larger than L1."""

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


def test_iouring_l2_miss_batch_copies_each_read_as_it_completes(tmp_path) -> None:
    """A completed L2 miss should copy to destination before slower reads finish."""

    async def scenario() -> None:
        path = str(tmp_path / "daser.store")
        writer = TieredIOUringTransferLayer(
            path=path,
            l1_bytes=ALIGNMENT,
            l2_bytes=ALIGNMENT * 2,
        )
        try:
            await writer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT)
            await writer.store_bytes(
                _block(b"b"),
                file_offset=ALIGNMENT,
                nbytes=ALIGNMENT,
            )
            await writer.drain()
        finally:
            writer.close()

        layer = DelayedL2ReadProbe(
            path=path,
            l1_bytes=ALIGNMENT * 2,
            l2_bytes=ALIGNMENT * 2,
            delayed_offsets={0},
        )
        try:
            dst = bytearray(ALIGNMENT * 2)
            task = asyncio.create_task(
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
            assert await asyncio.to_thread(layer.read_started.wait, timeout=2.0)
            deadline = time.perf_counter() + 2.0
            while not layer.copy_offsets and time.perf_counter() < deadline:
                await asyncio.sleep(0.01)
            assert layer.copy_offsets == [ALIGNMENT]
            layer.release_read.set()
            loaded = await asyncio.wait_for(task, timeout=2.0)
            assert loaded == ALIGNMENT * 2
            assert bytes(dst) == bytes(_block(b"a") + _block(b"b"))
        finally:
            layer.release_read.set()
            layer.close()

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


def test_iouring_store_waits_for_pending_l2_victim_before_reusing_pool(
    tmp_path,
) -> None:
    """Evicting a pending L2 buffer applies backpressure instead of allocating."""

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
                layer.store_bytes(_block(b"b"), file_offset=ALIGNMENT, nbytes=ALIGNMENT)
            )
            await asyncio.sleep(0.05)
            assert not store_task.done()

            layer.release_write.set()
            assert await asyncio.wait_for(store_task, timeout=1.0) == ALIGNMENT
            await layer.drain()
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
