# SPDX-License-Identifier: Apache-2.0
"""Concurrent demand loads share pending physical reads without sharing ownership."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import threading

import pytest

from daser.transfer.iouring import TieredIOUringTransferLayer
from daser.transfer.iouring.native import NativeIOUring

_BLOCK = 4096
_TIMEOUT = 10


@dataclass
class _ReadProbe:
    """Record submitted extents and hold the first real SSD read on its thread."""

    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    ranges: list[tuple[int, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _install_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_first: bool = False,
    capture_before_pause: bool = False,
) -> _ReadProbe:
    """Gate a real native read and track both single and batch public APIs."""
    probe = _ReadProbe()
    read_into = NativeIOUring.read_into
    read_batch_into = NativeIOUring.read_batch_into

    def record(ranges: list[tuple[int, int]]) -> bool:
        with probe.lock:
            first = not probe.ranges
            probe.ranges.extend(ranges)
        return first

    def pause(first: bool) -> None:
        if first:
            probe.started.set()
            if not probe.release.wait(_TIMEOUT):
                raise TimeoutError("the test did not release the native read")
            if fail_first:
                raise OSError("synthetic demand read failure")

    def single(ring: NativeIOUring, fd: int, file_offset: int, dst: memoryview) -> int:
        first = record([(file_offset, len(dst))])
        if capture_before_pause:
            result = read_into(ring, fd, file_offset, dst)
            pause(first)
            return result
        pause(first)
        return read_into(ring, fd, file_offset, dst)

    def batch(
        ring: NativeIOUring,
        fd: int,
        reads: Sequence[tuple[int, memoryview]],
        on_complete: Callable[[int, int], None] | None = None,
    ) -> int:
        first = record([(offset, len(dst)) for offset, dst in reads])
        # Tests that need an old-byte snapshot use a single extent. Pausing
        # before callbacks here keeps all batch destinations owned by the IO.
        pause(first)
        return read_batch_into(ring, fd, reads, on_complete)

    monkeypatch.setattr(NativeIOUring, "read_into", single)
    monkeypatch.setattr(NativeIOUring, "read_batch_into", batch)
    return probe


def _cold_layer(
    tmp_path: Path, *, l1_blocks: int = 8, l1_accounting: str = "stored"
) -> TieredIOUringTransferLayer:
    """Seed an aligned SSD store without admitting its contents into L1."""
    path = tmp_path / "demand-coalescing.store"
    path.write_bytes(b"".join(bytes([97 + index]) * _BLOCK for index in range(16)))
    return TieredIOUringTransferLayer(
        path=str(path),
        l1_bytes=l1_blocks * _BLOCK,
        l2_bytes=16 * _BLOCK,
        io_workers=4,
        l1_accounting=l1_accounting,
    )


async def _load(
    layer: TieredIOUringTransferLayer,
    dst: bytearray,
    *,
    grouped: bool,
    file_offset: int = 0,
) -> int:
    """Use either public demand API for the same physical extent."""
    if grouped:
        return await layer.load_bytes_grouped(
            dst,
            [{"file_offset": file_offset, "target_offset": 0, "nbytes": len(dst)}],
        )
    return await layer.load_bytes(dst, file_offset, len(dst))


async def _wait_for_reader(probe: _ReadProbe) -> None:
    """Wait off-loop until the executor owns the first native destination."""
    assert await asyncio.to_thread(probe.started.wait, _TIMEOUT)


async def _let_followers_wait() -> None:
    """Allow newly scheduled callers to reach the pending-read wait path."""
    # The native owner is gated, so this delay cannot accidentally turn a
    # concurrent test into an ordinary warm-cache hit. No latency is asserted.
    await asyncio.sleep(0.02)


async def _finish(
    layer: TieredIOUringTransferLayer,
    probe: _ReadProbe,
    tasks: Sequence[asyncio.Task[object]],
) -> None:
    """Stop every caller and drain IO before releasing the test store buffers."""
    probe.release.set()
    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), _TIMEOUT
            )
        await layer.drain()
    finally:
        layer.close()


@pytest.mark.parametrize("owner_grouped", [False, True], ids=["scalar", "grouped"])
@pytest.mark.parametrize("follower_grouped", [False, True], ids=["scalar", "grouped"])
def test_concurrent_loads_reuse_one_physical_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_grouped: bool,
    follower_grouped: bool,
) -> None:
    """Separate scalar/grouped callers read a shared cold range only once."""
    probe = _install_probe(monkeypatch)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path)
        destinations = [bytearray(_BLOCK) for _ in range(5)]
        tasks = [
            asyncio.create_task(_load(layer, destinations[0], grouped=owner_grouped))
        ]
        try:
            await _wait_for_reader(probe)
            tasks.extend(
                asyncio.create_task(_load(layer, dst, grouped=follower_grouped))
                for dst in destinations[1:]
            )
            await _let_followers_wait()
            assert all(not task.done() for task in tasks)
            probe.release.set()
            assert await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT) == [
                _BLOCK
            ] * len(tasks)
            assert all(dst == b"a" * _BLOCK for dst in destinations)
            assert probe.ranges == [(0, _BLOCK)]
            assert layer.stats.l2_reads == 1
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


def test_partial_overlap_preserves_targets_and_reads_only_uncovered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follower reuses overlap and scatters bytes to its own target layout."""
    probe = _install_probe(monkeypatch)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path)
        owner_dst = bytearray(2 * _BLOCK)
        follower_dst = bytearray(b"?" * (4 * _BLOCK))
        tasks = [asyncio.create_task(layer.load_bytes(owner_dst, 0, 2 * _BLOCK))]
        try:
            await _wait_for_reader(probe)
            tasks.append(
                asyncio.create_task(
                    layer.load_bytes_grouped(
                        follower_dst,
                        [
                            {
                                "file_offset": _BLOCK,
                                "target_offset": 0,
                                "nbytes": 2 * _BLOCK,
                            },
                            {
                                "file_offset": 0,
                                "target_offset": 3 * _BLOCK,
                                "nbytes": _BLOCK,
                            },
                        ],
                    )
                )
            )
            await _let_followers_wait()
            assert not tasks[1].done()
            probe.release.set()
            assert await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT) == [
                2 * _BLOCK,
                3 * _BLOCK,
            ]
            assert owner_dst == b"a" * _BLOCK + b"b" * _BLOCK
            assert follower_dst == b"".join(
                value * _BLOCK for value in [b"b", b"c", b"?", b"a"]
            )
            assert sorted(probe.ranges) == [(0, 2 * _BLOCK), (2 * _BLOCK, _BLOCK)]
            assert layer.stats.l2_reads == 2
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


def test_cold_coalesced_read_preserves_scattered_targets_and_raw_charges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Share a cold extent while preserving each caller's layout and raw charges.

    Args:
        tmp_path: Path for the temporary aligned SSD store.
        monkeypatch: Pytest fixture used to gate the public native read API.

    Returns:
        None. Assertions cover data placement, read reuse and capacity eviction.

    Async/thread-safety:
        One event loop owns the layer; a thread event gates the real native read.
        All callers and IO are drained before releasing the layer.
    """
    probe = _install_probe(monkeypatch)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path, l1_blocks=6, l1_accounting="raw")
        owner_dst = bytearray(b"?" * (4 * _BLOCK))
        follower_dst = bytearray(b"?" * (4 * _BLOCK))
        # Adjacent physical pages must share one read even though their target
        # addresses are scattered. Unequal raw-equivalent charges ensure that
        # preserving the extent's physical size alone cannot pass this test.
        owner_spans = [
            {
                "file_offset": 0,
                "target_offset": 0,
                "nbytes": _BLOCK,
                "accounted_nbytes": 2 * _BLOCK,
            },
            {
                "file_offset": _BLOCK,
                "target_offset": 3 * _BLOCK,
                "nbytes": _BLOCK,
                "accounted_nbytes": 3 * _BLOCK,
            },
        ]
        follower_spans = [
            dict(owner_spans[0], target_offset=2 * _BLOCK),
            dict(owner_spans[1], target_offset=0),
        ]
        tasks = [asyncio.create_task(layer.load_bytes_grouped(owner_dst, owner_spans))]
        try:
            await _wait_for_reader(probe)
            tasks.append(
                asyncio.create_task(
                    layer.load_bytes_grouped(follower_dst, follower_spans)
                )
            )
            await _let_followers_wait()
            assert all(not task.done() for task in tasks)
            probe.release.set()
            assert await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT) == [
                2 * _BLOCK,
                2 * _BLOCK,
            ]
            owner_expected = b"a" * _BLOCK + b"?" * (2 * _BLOCK) + b"b" * _BLOCK
            assert owner_dst == owner_expected
            assert follower_dst == b"".join(
                value * _BLOCK for value in [b"b", b"?", b"a", b"?"]
            )
            assert probe.ranges == [(0, 2 * _BLOCK)]
            assert layer.stats.l2_reads == 1
            assert layer.l1_bytes_used == 5 * _BLOCK

            # The next page fits physically alongside the two-page parent,
            # but its charge exceeds the remaining raw-equivalent capacity.
            # Eviction must therefore force the original extent back to SSD.
            later = bytearray(_BLOCK)
            assert (
                await layer.load_bytes_grouped(
                    later,
                    [
                        {
                            "file_offset": 2 * _BLOCK,
                            "target_offset": 0,
                            "nbytes": _BLOCK,
                            "accounted_nbytes": 2 * _BLOCK,
                        }
                    ],
                )
                == _BLOCK
            )
            assert later == b"c" * _BLOCK
            assert layer.l1_bytes_used == 2 * _BLOCK
            reread = bytearray(b"?" * (4 * _BLOCK))
            assert await layer.load_bytes_grouped(reread, owner_spans) == 2 * _BLOCK
            assert reread == owner_expected
            assert probe.ranges == [
                (0, 2 * _BLOCK),
                (2 * _BLOCK, _BLOCK),
                (0, 2 * _BLOCK),
            ]
            assert layer.stats.l2_reads == 3
            assert layer.l1_bytes_used == 5 * _BLOCK
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("prefetch_first", [False, True], ids=["demand", "prefetch"])
def test_demand_and_prefetch_share_pending_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefetch_first: bool
) -> None:
    """Demand and leased prefetch join the same read in either arrival order."""
    probe = _install_probe(monkeypatch)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path)
        dst = bytearray(_BLOCK)
        spans = [{"file_offset": 0, "nbytes": _BLOCK}]
        tasks: list[asyncio.Task[object]] = []
        try:
            if prefetch_first:
                tasks.append(
                    asyncio.create_task(layer.prefetch_bytes_grouped(spans, "lease"))
                )
            else:
                tasks.append(asyncio.create_task(layer.load_bytes(dst, 0, _BLOCK)))
            await _wait_for_reader(probe)
            if prefetch_first:
                tasks.append(asyncio.create_task(layer.load_bytes(dst, 0, _BLOCK)))
            else:
                tasks.append(
                    asyncio.create_task(layer.prefetch_bytes_grouped(spans, "lease"))
                )
            await _let_followers_wait()
            probe.release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT)
            assert dst == b"a" * _BLOCK
            leased_dst = bytearray(_BLOCK)
            assert (
                await layer.load_leased_bytes_grouped(
                    leased_dst, [{**spans[0], "target_offset": 0}], "lease"
                )
                == _BLOCK
            )
            assert leased_dst == dst
            assert probe.ranges == [(0, _BLOCK)]
            assert layer.stats.l2_reads == 1
        finally:
            await layer.release_lease("lease")
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


def test_cancelling_follower_keeps_owner_and_other_followers_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling one caller must not cancel the shared completion or its IO."""
    probe = _install_probe(monkeypatch)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path)
        destinations = [bytearray(_BLOCK) for _ in range(3)]
        tasks = [asyncio.create_task(layer.load_bytes(destinations[0], 0, _BLOCK))]
        try:
            await _wait_for_reader(probe)
            tasks.extend(
                asyncio.create_task(layer.load_bytes(dst, 0, _BLOCK))
                for dst in destinations[1:]
            )
            await _let_followers_wait()
            tasks[1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(tasks[1], _TIMEOUT)
            assert not tasks[0].done()
            assert not tasks[2].done()
            probe.release.set()
            assert await asyncio.wait_for(
                asyncio.gather(tasks[0], tasks[2]), _TIMEOUT
            ) == [_BLOCK, _BLOCK]
            assert destinations[0] == destinations[2] == b"a" * _BLOCK
            assert probe.ranges == [(0, _BLOCK)]
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("owner_outcome", ["cancel", "io_error"])
def test_failed_owner_releases_followers_and_pool_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner_outcome: str
) -> None:
    """A failed owner leaves followers able to retry and reuse a one-page pool."""
    probe = _install_probe(monkeypatch, fail_first=owner_outcome == "io_error")

    async def scenario() -> None:
        layer = _cold_layer(tmp_path, l1_blocks=1)
        owner_dst = bytearray(_BLOCK)
        follower_dst = bytearray(_BLOCK)
        tasks = [asyncio.create_task(layer.load_bytes(owner_dst, 0, _BLOCK))]
        try:
            await _wait_for_reader(probe)
            tasks.append(asyncio.create_task(layer.load_bytes(follower_dst, 0, _BLOCK)))
            await _let_followers_wait()
            if owner_outcome == "cancel":
                tasks[0].cancel()
                await _let_followers_wait()
                assert not tasks[0].done()
                assert not tasks[1].done()
            probe.release.set()
            expected_error = (
                asyncio.CancelledError if owner_outcome == "cancel" else OSError
            )
            with pytest.raises(expected_error):
                await asyncio.wait_for(tasks[0], _TIMEOUT)
            assert await asyncio.wait_for(tasks[1], _TIMEOUT) == _BLOCK
            assert follower_dst == b"a" * _BLOCK
            replacement = bytearray(_BLOCK)
            assert (
                await asyncio.wait_for(
                    layer.load_bytes(replacement, _BLOCK, _BLOCK), _TIMEOUT
                )
                == _BLOCK
            )
            assert replacement == b"b" * _BLOCK
            assert layer.l1_bytes_used == _BLOCK
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


def test_tiny_l1_allows_overlapping_and_disjoint_loads_to_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared reads and unrelated pool waiters make progress with one page."""
    probe = _install_probe(monkeypatch)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path, l1_blocks=1)
        offsets = [0, 0, _BLOCK, 2 * _BLOCK, 0]
        destinations = [bytearray(_BLOCK) for _ in offsets]
        tasks = [asyncio.create_task(layer.load_bytes(destinations[0], 0, _BLOCK))]
        try:
            await _wait_for_reader(probe)
            tasks.extend(
                asyncio.create_task(layer.load_bytes(dst, offset, _BLOCK))
                for dst, offset in zip(destinations[1:], offsets[1:], strict=True)
            )
            await _let_followers_wait()
            probe.release.set()
            assert await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT) == [
                _BLOCK
            ] * len(tasks)
            assert [bytes(dst) for dst in destinations] == [
                bytes([97 + offset // _BLOCK]) * _BLOCK for offset in offsets
            ]
            assert layer.l1_bytes_used <= _BLOCK
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


def test_duplicate_grouped_spans_do_not_wait_on_their_own_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate physical spans in one request share data at distinct targets."""
    probe = _install_probe(monkeypatch)
    probe.release.set()

    async def scenario() -> None:
        layer = _cold_layer(tmp_path)
        dst = bytearray(b"?" * (3 * _BLOCK))
        tasks = [
            asyncio.create_task(
                layer.load_bytes_grouped(
                    dst,
                    [
                        {"file_offset": 0, "target_offset": 0, "nbytes": _BLOCK},
                        {
                            "file_offset": 0,
                            "target_offset": 2 * _BLOCK,
                            "nbytes": _BLOCK,
                        },
                    ],
                )
            )
        ]
        try:
            assert await asyncio.wait_for(tasks[0], _TIMEOUT) == 2 * _BLOCK
            assert dst == b"a" * _BLOCK + b"?" * _BLOCK + b"a" * _BLOCK
            assert probe.ranges == [(0, _BLOCK)]
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())


def test_waiting_load_resolves_current_l1_after_overlapping_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follower uses newer published bytes instead of a stale owner's buffer."""
    probe = _install_probe(monkeypatch, capture_before_pause=True)

    async def scenario() -> None:
        layer = _cold_layer(tmp_path)
        old_dst = bytearray(_BLOCK)
        current_dst = bytearray(_BLOCK)
        tasks = [asyncio.create_task(layer.load_bytes(old_dst, 0, _BLOCK))]
        try:
            await _wait_for_reader(probe)
            tasks.append(asyncio.create_task(layer.load_bytes(current_dst, 0, _BLOCK)))
            await _let_followers_wait()
            assert await layer.store_bytes(b"n" * _BLOCK, 0, _BLOCK) == _BLOCK
            probe.release.set()
            assert await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT) == [
                _BLOCK,
                _BLOCK,
            ]
            assert old_dst == b"a" * _BLOCK
            assert current_dst == b"n" * _BLOCK
            reread = bytearray(_BLOCK)
            assert await layer.load_bytes(reread, 0, _BLOCK) == _BLOCK
            assert reread == current_dst
            assert probe.ranges == [(0, _BLOCK)]
        finally:
            await _finish(layer, probe, tasks)

    asyncio.run(scenario())
