# SPDX-License-Identifier: Apache-2.0

"""io_uring data mover over a pinned L1 arena, a staging pool and the L2 file.

The server owns every location: ``l1_offset`` indexes the arena,
``l2_offset`` the store file. This layer only keeps transient state (in-flight
IO, staging slices, copy events). It never caches by address, never evicts
and never orders writes: the server pins an extent while IO uses it and only
hands over extents whose bytes are valid.
"""

# Standard
import asyncio
from collections.abc import Awaitable, Coroutine, Sequence
import time
from typing import Any

# Third Party
import numpy

# First Party
from daser.logging import init_logger
from daser.transfer.base import (
    LoadSource,
    Move,
    StoreHandle,
    TransferLayer,
    TransferStats,
)
from daser.transfer.iouring import copy_ops, pinned_pool
from daser.transfer.iouring.l2_engine import L2IoEngine
from daser.transfer.iouring.native import NativeIOUring
from daser.transfer.iouring.staging import StagingPool
from daser.transfer.moves import (
    DIRECT_IO_ALIGNMENT,
    MoveKind,
    buffer_nbytes,
    coalesce_moves,
    split_move,
    validate_moves,
)
from daser.transfer.tasks import consume_exception, resolve, settle

logger = init_logger(__name__)

#: Reads up to this size share one native submission batch; larger reads are
#: striped (or issued alone) and occupy a ring by themselves.
_BATCHED_READ_BYTES = 128 * 1024 * 1024
#: Reads per native submission batch (the ring depth).
_READ_BATCH_COUNT = 64
#: Largest arena snapshot group: a long store is snapshotted group by group,
#: and each group's L2 writes start while the next group is still copied.
_SNAPSHOT_GROUP_BYTES = 128 * 1024 * 1024
#: Largest staging piece; several pieces keep IO in flight through the pool.
_STAGING_PIECE_BYTES = 32 * 1024 * 1024
#: Poll interval for CUDA copy events.
_EVENT_POLL_S = 0.0005


def _stripe_extents(nbytes: int, stripe_bytes: int) -> list[tuple[int, int]]:
    """Split one aligned read into submission-sized ``(offset, size)`` parts.

    Args:
        nbytes: Read length in bytes.
        stripe_bytes: Target part length; parts grow beyond it when the read
            would otherwise need more than one ring submission batch.

    Returns:
        Contiguous parts covering ``[0, nbytes)``, each a multiple of the
        direct-IO alignment except possibly the last.
    """
    per_part = max(stripe_bytes, -(-nbytes // _READ_BATCH_COUNT))
    per_part = -(-per_part // DIRECT_IO_ALIGNMENT) * DIRECT_IO_ALIGNMENT
    return [
        (offset, min(per_part, nbytes - offset))
        for offset in range(0, nbytes, per_part)
    ]


def _snapshot_groups(moves: list[Move]) -> list[list[Move]]:
    """Cut arena-bound store moves into snapshot groups in source order.

    Args:
        moves: Moves with ``l1_offset``.

    Returns:
        Groups of at most ``_SNAPSHOT_GROUP_BYTES`` each, ordered by
        ``buffer_offset``; a longer move is split across groups.
    """
    groups: list[list[Move]] = []
    group_bytes = 0
    for move in sorted(moves, key=lambda move: move.buffer_offset):
        for piece in split_move(move, _SNAPSHOT_GROUP_BYTES):
            if not groups or group_bytes + piece.nbytes > _SNAPSHOT_GROUP_BYTES:
                groups.append([])
                group_bytes = 0
            groups[-1].append(piece)
            group_bytes += piece.nbytes
    return groups


def _snapshot_runs(
    buffer: Any, runs: list[tuple[int, copy_ops.HostMemory, int, int]]
) -> None:
    """Copy ``(buffer_offset, host, host_offset, nbytes)`` runs into host memory.

    Async/thread-safety:
        Blocking (CUDA ``memcpy`` for device buffers); run off the loop.
    """
    for buffer_offset, host, host_offset, nbytes in runs:
        source = copy_ops.slice_src(buffer, buffer_offset, nbytes)
        copy_ops.copy_src_to_pinned(source, host, host_offset, nbytes)


def _read_exact(
    l2: L2IoEngine, file_offset: int, data: memoryview, uring: NativeIOUring
) -> int:
    """Blocking L2 read that fails on a short read.

    Returns:
        Bytes read.
    """
    read = l2.read_into(file_offset, data, uring)
    if read != len(data):
        raise IOError(f"short io_uring read: {read} != {len(data)}")
    return read


class TieredIOUringTransferLayer(TransferLayer):
    """io_uring transfer backend with a pinned L1 arena and staging pool.

    Args:
        path: Existing L2 store file of at least ``l2_bytes`` bytes, created
            by the server; unused when ``skip_l2``.
        l1_bytes: Pinned arena size (multiple of 4096); 0 builds no arena.
        staging_bytes: Pinned staging pool size (positive multiple of 4096)
            for L2 bytes without an arena extent; must be 0 when ``skip_l2``.
        l2_bytes: L2 file bytes addressable by ``l2_offset``.
        io_workers: Native io_uring rings and executor threads.
        skip_l2: Build no L2 engine; only arena stores and loads work.
        l2_read_stripe_bytes: Split a read larger than 128 MiB into parts of
            about this size submitted together on one ring (multiple of 4096;
            0 keeps one serial read).
        l2_write_gate: Research switch; L2 writes pause between 16 MiB pieces
            while an L2 read is in native IO (see ``L2IoEngine``).

    Async/thread-safety:
        Call every method from one event loop. Native IO waits and D2H
        snapshots run on executor threads; H2D copies are polled. Concurrent
        operations on the same extent are the caller's responsibility.
    """

    def __init__(
        self,
        path: str,
        *,
        l1_bytes: int,
        staging_bytes: int,
        l2_bytes: int,
        io_workers: int = 8,
        skip_l2: bool = False,
        l2_read_stripe_bytes: int = 0,
        l2_write_gate: bool = False,
    ) -> None:
        for name, value in (
            ("l1_bytes", l1_bytes),
            ("staging_bytes", staging_bytes),
            ("l2_read_stripe_bytes", l2_read_stripe_bytes),
        ):
            if value < 0 or value % DIRECT_IO_ALIGNMENT:
                raise ValueError(
                    f"{name} must be a non-negative multiple of {DIRECT_IO_ALIGNMENT}"
                )
        if skip_l2:
            if not l1_bytes:
                raise ValueError("skip_l2 needs a positive l1_bytes")
            if staging_bytes:
                raise ValueError("skip_l2 has no staging pool; staging_bytes must be 0")
        elif l2_bytes <= 0 or not staging_bytes:
            raise ValueError("an L2 tier needs positive l2_bytes and staging_bytes")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        self._l1_bytes = l1_bytes
        self._l2_bytes = 0 if skip_l2 else l2_bytes
        self._stripe = l2_read_stripe_bytes
        self._stats = TransferStats()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._kinds: dict[str, int] = {}
        self._errors: list[BaseException] = []
        self._close_task: asyncio.Task[None] | None = None
        self._l2: L2IoEngine | None = None
        self._arena: pinned_pool.PinnedMemoryBuffer | None = None
        self._staging: StagingPool | None = None
        try:
            if not skip_l2:
                self._l2 = L2IoEngine(path, l2_bytes, io_workers, l2_write_gate)
            if l1_bytes:
                self._arena = pinned_pool.PinnedMemoryBuffer(l1_bytes)
            if staging_bytes:
                self._staging = StagingPool(staging_bytes)
        except BaseException:
            self._release_resources()
            raise
        logger.info(
            "[TRANSFER:iouring] l1=%d staging=%d l2=%d workers=%d stripe=%d gate=%s",
            l1_bytes,
            staging_bytes,
            self._l2_bytes,
            io_workers,
            l2_read_stripe_bytes,
            l2_write_gate,
        )

    @property
    def stats(self) -> TransferStats:
        """Return the live cumulative counters."""
        return self._stats

    @property
    def l2_timing_us(self) -> dict[str, int]:
        """Return cumulative L2 queue and IO microseconds (empty without L2)."""
        return {} if self._l2 is None else self._l2.timing_us

    @property
    def l2_read_contention(self) -> dict[tuple[int, bool], tuple[int, int, int]]:
        """Return L2 read jobs by contention met (empty without L2)."""
        return {} if self._l2 is None else self._l2.read_contention

    def debug_state(self) -> dict[str, Any]:
        """Return in-flight operation counts and staging occupancy.

        Returns:
            Counts per operation kind (``store``, ``l2_write``, ``load``,
            ``read``, ``copy``, ``runs``), staging bytes used and waiters.
        """
        state: dict[str, Any] = dict(self._kinds)
        state["closing"] = self._close_task is not None
        if self._staging is not None:
            state["staging_used_bytes"] = self._staging.used_bytes
            state["staging_waiters"] = self._staging.waiters
        return state

    async def store(self, buffer: Any, moves: Sequence[Move]) -> StoreHandle:
        """Snapshot moves into the arena and/or staging and write L2 copies.

        Args:
            buffer: Readable staging buffer (CuPy-like or bytes-like).
            moves: Ranges with ``buffer_offset`` and a tier offset each.

        Returns:
            Handle: ``staged`` once every source byte is in host memory
            (arena copies valid), ``persisted`` once every L2 write ended.

        Raises:
            ValueError: For an invalid move; nothing is submitted.
            RuntimeError: If the layer is closing.
        """
        self._check_open()
        self._validate(moves, "store", buffer)
        loop = asyncio.get_running_loop()
        handle = StoreHandle(loop.create_future(), loop.create_future())
        handle.staged.add_done_callback(consume_exception)
        handle.persisted.add_done_callback(consume_exception)
        self._spawn("store", self._run_store(buffer, list(moves), handle))
        return handle

    async def load(
        self, buffer: Any, moves: Sequence[Move], source: LoadSource
    ) -> None:
        """Copy arena or L2 ranges into ``buffer``.

        Args:
            buffer: Writable destination (CuPy-like or bytes-like).
            moves: Ranges to read; with ``source="l2"`` a move that has
                ``l1_offset`` fills that arena extent with the same read.
            source: ``"l1"`` or ``"l2"``.

        Raises:
            ValueError: For an invalid move; nothing is submitted.
            IOError: On a failed or short read.
            RuntimeError: If the layer is closing.

        Async/thread-safety:
            Returns once every byte reached ``buffer``; cancellation is
            deferred until submitted IO and copies ended.
        """
        if source not in ("l1", "l2"):
            raise ValueError(f"load source must be 'l1' or 'l2', not {source!r}")
        self._check_open()
        self._validate(moves, source, buffer)
        if not moves:
            return
        if source == "l1":
            body = self._load_arena(buffer, list(moves))
        else:
            body = self._read_l2(buffer, list(moves))
        await settle([self._spawn("load", body)])

    async def promote(self, moves: Sequence[Move]) -> None:
        """Read L2 ranges into arena extents without a GPU copy.

        Args:
            moves: Ranges with ``l2_offset`` and ``l1_offset``.

        Raises:
            NotImplementedError: If the layer has no arena or no L2.
            ValueError: For an invalid move; nothing is submitted.
            IOError: On a failed or short read.

        Async/thread-safety:
            Cancellation is deferred until the submitted reads ended.
        """
        self._check_open()
        if self._arena is None or self._l2 is None:
            raise NotImplementedError("promotion needs both an L1 arena and L2")
        self._validate(moves, "promote", None)
        if moves:
            await settle([self._spawn("load", self._read_l2(None, list(moves)))])

    async def demote(self, moves: Sequence[Move]) -> None:
        """Write arena extents to their L2 ranges.

        Args:
            moves: Ranges with ``l1_offset`` (source) and ``l2_offset``.

        Raises:
            NotImplementedError: If the layer has no arena or no L2.
            ValueError: For an invalid move; nothing is submitted.
            IOError: On a failed or short write.

        Async/thread-safety:
            Runs on the L2 write executor (and its write gate); cancellation
            is deferred until the submitted writes ended.
        """
        self._check_open()
        if self._arena is None or self._l2 is None:
            raise NotImplementedError("demotion needs both an L1 arena and L2")
        self._validate(moves, "promote", None)
        runs = coalesce_moves(list(moves), ("l2", "l1"))
        if runs:
            await settle([self._spawn("l2_write", self._write(runs))])

    async def load_runs(
        self,
        buffer: Any,
        l1_offsets: Any,
        targets: Any,
        lengths: Any,
    ) -> None:
        """Copy many arena runs into a flat CUDA destination in one batch.

        Args:
            buffer: Flat CuPy byte destination, optionally exposing
                ``copy_stream_ptr``.
            l1_offsets: ``int64`` array of absolute arena offsets.
            targets: ``int64`` array of destination byte offsets.
            lengths: ``int64`` array of positive run lengths.

        Raises:
            NotImplementedError: If the layer has no arena.
            ValueError: If the arrays are empty or unequal, or a run falls
                outside the arena or the destination buffer.

        Async/thread-safety:
            Returns once the copies completed on the device; cancellation is
            deferred until then.
        """
        self._check_open()
        if self._arena is None:
            raise NotImplementedError("transfer layer has no L1 arena")
        offsets = numpy.asarray(l1_offsets, dtype=numpy.int64)
        sizes = numpy.asarray(lengths, dtype=numpy.int64)
        dst_offsets = numpy.asarray(targets, dtype=numpy.int64)
        if not (len(offsets) == len(sizes) == len(dst_offsets)) or not len(offsets):
            raise ValueError("layer runs must be nonempty and of equal length")
        if (
            int(sizes.min()) <= 0
            or int(offsets.min()) < 0
            or int((offsets + sizes).max()) > self._l1_bytes
        ):
            raise ValueError("layer runs exceed the L1 arena")
        dst_bytes = buffer_nbytes(buffer)
        if int(dst_offsets.min()) < 0 or (
            dst_bytes is not None and int((dst_offsets + sizes).max()) > dst_bytes
        ):
            raise ValueError("layer runs exceed the destination buffer")
        sources = offsets.astype(numpy.uint64) + numpy.uint64(self._arena.ptr_at(0))
        await settle(
            [self._spawn("runs", self._copy_runs(buffer, dst_offsets, sources, sizes))]
        )

    async def drain(self) -> None:
        """Wait for every in-flight operation, then raise a store failure.

        Raises:
            RuntimeError: For the first store failure since the previous
                ``drain``; later failures of that window are logged only.

        Async/thread-safety:
            Cancelling ``drain`` does not cancel the work it waits for.
        """
        await self._wait_idle()
        if self._errors:
            first, *rest = self._errors
            self._errors.clear()
            for error in rest:
                logger.warning("[TRANSFER:iouring] further store failure: %r", error)
            raise RuntimeError("asynchronous io_uring store failed") from first

    async def close(self) -> None:
        """Drain all work, then close L2 and free the pinned memory.

        Async/thread-safety:
            Idempotent; concurrent callers share one shutdown. New operations
            raise ``RuntimeError`` once close began. Store failures still
            pending are logged, not raised.
        """
        if self._close_task is None:
            self._close_task = asyncio.get_running_loop().create_task(self._run_close())
        await asyncio.shield(self._close_task)

    def _check_open(self) -> None:
        """Raise if close began."""
        if self._close_task is not None:
            raise RuntimeError("io_uring transfer layer is closing")

    def _validate(self, moves: Sequence[Move], kind: MoveKind, buffer: Any) -> None:
        """Validate a move batch against this layer's regions."""
        validate_moves(
            moves,
            kind,
            buffer_bytes=None if buffer is None else buffer_nbytes(buffer),
            l1_bytes=self._l1_bytes,
            l2_bytes=self._l2_bytes,
        )

    def _spawn(self, kind: str, body: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Start a tracked task that ``drain`` and ``close`` wait for."""
        task = asyncio.get_running_loop().create_task(body)
        self._tasks.add(task)
        self._kinds[kind] = self._kinds.get(kind, 0) + 1

        def done(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            self._kinds[kind] -= 1

        task.add_done_callback(done)
        return task

    async def _wait_idle(self) -> None:
        """Wait until no tracked task remains, without cancelling any."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))

    async def _run_store(
        self, buffer: Any, moves: list[Move], handle: StoreHandle
    ) -> None:
        """Body of one store: snapshots, then L2 writes, then the handle.

        Arena moves are snapshotted in bounded groups; a group's L2 writes are
        submitted before the next group is copied, so D2H and SSD overlap.
        """
        writes: list[asyncio.Task[Any]] = []
        error: BaseException | None = None
        try:
            arena_moves = [move for move in moves if move.l1_offset is not None]
            staged_moves = [move for move in moves if move.l1_offset is None]
            for group in _snapshot_groups(arena_moves):
                await self._snapshot_arena(buffer, group)
                runs = coalesce_moves(
                    [move for move in group if move.l2_offset is not None],
                    ("l2", "l1"),
                )
                if runs:
                    # One write job set per group: per-run jobs would queue
                    # ahead of reads on the shared executor.
                    writes.append(self._spawn("l2_write", self._write(runs)))
            if staged_moves:
                await self._store_staged(buffer, staged_moves, writes)
        except BaseException as exc:
            error = exc
        resolve(handle.staged, error)
        try:
            await settle(writes)
        except BaseException as exc:
            error = error or exc
        resolve(handle.persisted, error)
        if isinstance(error, asyncio.CancelledError):
            raise error
        if error is not None:
            self._errors.append(error)
            logger.warning("[TRANSFER:iouring] store failed: %r", error)

    async def _snapshot_arena(self, buffer: Any, moves: list[Move]) -> None:
        """Copy arena-bound moves from ``buffer`` into the arena."""
        assert self._arena is not None
        arena = self._arena
        runs = [
            (run.buffer_offset, arena, run.l1_offset, run.nbytes)
            for run in coalesce_moves(moves, ("buffer", "l1"))
            if run.l1_offset is not None
        ]
        await self._snapshot(buffer, runs)

    async def _store_staged(
        self, buffer: Any, moves: list[Move], writes: list[asyncio.Task[Any]]
    ) -> None:
        """Stream arena-less moves through staging slices into L2.

        Pieces are reserved in groups; a group is snapshotted with one D2H
        call and each piece's write releases its slice. Only the first piece
        of a group may wait for space, because a waiter must hold no slice
        whose IO is not yet submitted.
        """
        assert self._staging is not None
        staging = self._staging
        piece_bytes = min(_STAGING_PIECE_BYTES, staging.capacity)
        pieces = [
            piece
            for run in coalesce_moves(moves, ("buffer", "l2"))
            for piece in split_move(run, piece_bytes)
        ]
        group: list[tuple[Move, pinned_pool.PinnedMemorySlice]] = []
        try:
            for piece in pieces:
                data = staging.try_reserve(piece.nbytes)
                if data is None:
                    await self._flush_staged(buffer, group, writes)
                    data = await self._reserve_waiting(piece.nbytes)
                group.append((piece, data))
            await self._flush_staged(buffer, group, writes)
        finally:
            for _piece, data in group:
                staging.release(data)

    async def _flush_staged(
        self,
        buffer: Any,
        group: list[tuple[Move, pinned_pool.PinnedMemorySlice]],
        writes: list[asyncio.Task[Any]],
    ) -> None:
        """Snapshot a staging group and hand each slice to its L2 write."""
        if not group:
            return
        await self._snapshot(
            buffer,
            [(piece.buffer_offset, data, 0, piece.nbytes) for piece, data in group],
        )
        for piece, data in group:
            writes.append(self._spawn("l2_write", self._write([piece], data)))
        group.clear()

    async def _snapshot(
        self, buffer: Any, runs: list[tuple[int, copy_ops.HostMemory, int, int]]
    ) -> None:
        """Run one D2H snapshot of ``runs`` on a worker thread."""
        # Copy helpers select the CUDA device explicitly and need no caller
        # context variables. Submit directly without an intermediate task;
        # settle still defers repeated cancellation until the real copy ends.
        await settle(
            [
                asyncio.get_running_loop().run_in_executor(
                    None, _snapshot_runs, buffer, runs
                )
            ]
        )
        self._stats.d2h_copies += 1
        self._stats.d2h_bytes += sum(run[3] for run in runs)

    async def _write(
        self,
        runs: list[Move],
        owned: pinned_pool.PinnedMemorySlice | None = None,
    ) -> None:
        """Persist ``runs`` at their ``l2_offset``; release ``owned`` after.

        Sources are the runs' arena extents, or the single run's ``owned``
        staging slice.
        """
        assert self._l2 is not None
        writes = []
        for run in runs:
            assert run.l2_offset is not None
            if owned is not None:
                view = owned.view()
            else:
                assert self._arena is not None and run.l1_offset is not None
                view = self._arena.view()[run.l1_offset : run.l1_offset + run.nbytes]
            writes.append((run.l2_offset, view))
        try:
            await self._l2.write_batch(writes)
            self._stats.l2_writes += len(runs)
            self._stats.l2_write_bytes += sum(run.nbytes for run in runs)
        finally:
            if owned is not None and self._staging is not None:
                self._staging.release(owned)

    async def _load_arena(self, buffer: Any, moves: list[Move]) -> None:
        """Copy arena ranges into ``buffer``."""
        assert self._arena is not None
        arena = self._arena
        chunks: list[copy_ops.CopyChunk] = [
            (move.buffer_offset, arena, move.l1_offset, move.nbytes)
            for move in moves
            if move.l1_offset is not None
        ]
        copy = self._submit_copy(buffer, chunks)
        if copy is not None:
            await settle([copy])

    async def _read_l2(self, buffer: Any | None, moves: list[Move]) -> None:
        """Read L2 moves into the arena or staging, then copy to ``buffer``.

        A move with ``l1_offset`` reads straight into its arena extent; one
        without reads through a staging slice. Reads run in batches bounded
        by the ring count; each read's copy starts at its own completion.
        """
        assert self._l2 is not None
        keys = ("l2", "l1") if buffer is None else ("l2", "buffer", "l1")
        merged = coalesce_moves(moves, keys)
        if self._staging is not None:
            piece_bytes = min(_STAGING_PIECE_BYTES, self._staging.capacity)
            merged = [
                piece
                for move in merged
                for piece in (
                    [move]
                    if move.l1_offset is not None
                    else split_move(move, piece_bytes)
                )
            ]
        capacity = self._l2.ring_count * _READ_BATCH_COUNT
        start = 0
        while start < len(merged):
            cost = 0
            end = start
            while end < len(merged):
                piece_cost = (
                    1
                    if merged[end].nbytes <= _BATCHED_READ_BYTES
                    else _READ_BATCH_COUNT
                )
                if end > start and cost + piece_cost > capacity:
                    break
                cost += piece_cost
                end += 1
            await self._read_batch(buffer, merged[start:end])
            start = end

    async def _read_batch(self, buffer: Any | None, moves: list[Move]) -> None:
        """Submit one bounded batch of reads and wait for all of them."""
        assert self._l2 is not None
        l2 = self._l2
        loop = asyncio.get_running_loop()
        jobs: list[Awaitable[Any]] = []
        finishers: list[asyncio.Task[Any]] = []
        queued: list[tuple[Move, memoryview, pinned_pool.PinnedMemorySlice | None]] = []
        unsubmitted: pinned_pool.PinnedMemorySlice | None = None

        def submit_queued() -> None:
            if not queued:
                return
            job, completions = l2.submit_read_batch(
                [(move.l2_offset, view) for move, view, _data in queued]
            )
            # From here the finishers own the slices, not the error path.
            submitted = list(queued)
            queued.clear()
            jobs.append(job)
            for (move, _view, data), completion in zip(
                submitted, completions, strict=True
            ):
                finishers.append(
                    self._spawn(
                        "read", self._finish_read(buffer, move, data, completion)
                    )
                )

        error: BaseException | None = None
        try:
            for move in moves:
                assert move.l2_offset is not None
                data: pinned_pool.PinnedMemorySlice | None = None
                if move.l1_offset is None:
                    assert self._staging is not None
                    data = self._staging.try_reserve(move.nbytes)
                    if data is None:
                        # Wait only once everything held is in flight.
                        submit_queued()
                        data = await self._reserve_waiting(move.nbytes)
                    unsubmitted = data
                    view = data.view()
                else:
                    assert self._arena is not None
                    view = self._arena.view()[
                        move.l1_offset : move.l1_offset + move.nbytes
                    ]
                if move.nbytes <= _BATCHED_READ_BYTES:
                    queued.append((move, view, data))
                    unsubmitted = None
                    if len(queued) >= _READ_BATCH_COUNT:
                        submit_queued()
                    continue
                submit_queued()
                if self._stripe:
                    job, parts = l2.submit_read_batch(
                        [
                            (move.l2_offset + offset, view[offset : offset + size])
                            for offset, size in _stripe_extents(
                                move.nbytes, self._stripe
                            )
                        ]
                    )
                    jobs.extend(parts)
                else:
                    job = loop.run_in_executor(
                        l2.executor,
                        l2.timed("read", _read_exact),
                        l2,
                        move.l2_offset,
                        view,
                        l2.next_uring(),
                    )
                unsubmitted = None
                finishers.append(
                    self._spawn("read", self._finish_read(buffer, move, data, job))
                )
            submit_queued()
        except BaseException as exc:
            error = exc
            # Slices still queued or reserved never reached the kernel.
            for _move, _view, data in queued:
                if data is not None and self._staging is not None:
                    self._staging.release(data)
            queued.clear()
            if unsubmitted is not None and self._staging is not None:
                self._staging.release(unsubmitted)
        try:
            await settle([*jobs, *finishers])
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error

    async def _finish_read(
        self,
        buffer: Any | None,
        move: Move,
        data: pinned_pool.PinnedMemorySlice | None,
        completion: asyncio.Future[int],
    ) -> None:
        """Account one read, copy its bytes to ``buffer``, release staging.

        The read is awaited through ``settle``: cancelling this task must
        neither cancel the IO future nor release ``data`` while the kernel
        may still write into it.
        """
        try:
            await settle([completion])
            nbytes = completion.result()
            if nbytes != move.nbytes:
                raise IOError(f"short io_uring read: {nbytes} != {move.nbytes}")
            self._stats.l2_reads += 1
            self._stats.l2_read_bytes += nbytes
            if buffer is None:
                return
            if data is None:
                assert self._arena is not None and move.l1_offset is not None
                chunk: copy_ops.CopyChunk = (
                    move.buffer_offset,
                    self._arena,
                    move.l1_offset,
                    move.nbytes,
                )
            else:
                chunk = (move.buffer_offset, data, 0, move.nbytes)
            copy = self._submit_copy(buffer, [chunk])
            if copy is not None:
                await settle([copy])
        finally:
            if data is not None and self._staging is not None:
                self._staging.release(data)

    async def _reserve_waiting(self, nbytes: int) -> pinned_pool.PinnedMemorySlice:
        """Reserve staging, waiting for in-flight IO, and count the wait."""
        assert self._staging is not None
        start = time.perf_counter()
        data = await self._staging.reserve(nbytes)
        self._stats.staging_waits += 1
        self._stats.staging_wait_us += int((time.perf_counter() - start) * 1e6)
        return data

    def _submit_copy(
        self, buffer: Any, chunks: list[copy_ops.CopyChunk]
    ) -> asyncio.Task[None] | None:
        """Copy host chunks into ``buffer``.

        Returns:
            None after a synchronous CPU copy, else a tracked task that ends
            when the CUDA copy event completed (raising a submit failure).
        """
        event = copy_ops.destination_copy_event(buffer, chunks)
        nbytes = sum(chunk[3] for chunk in chunks)
        if event is None:
            copy_ops.copy_grouped_to_dst(buffer, chunks)
            self._stats.dst_copies += 1
            self._stats.dst_copy_bytes += nbytes
            return None
        error: BaseException | None = None
        try:
            runs = copy_ops.copy_grouped_to_dst(buffer, chunks)
            self._stats.dst_copies += 1
            self._stats.dst_copy_runs += runs
            self._stats.dst_copy_bytes += nbytes
        except BaseException as exc:
            error = exc
        finally:
            # A partial submission may still read the sources: record the
            # event even after a failure and wait for it before raising.
            copy_ops.record_destination_copy_event(buffer, event)
        return self._spawn("copy", self._await_event(event, error))

    async def _copy_runs(
        self, buffer: Any, targets: Any, sources: Any, sizes: Any
    ) -> None:
        """Queue pointer runs into ``buffer`` and wait for the copy event."""
        event = copy_ops.copy_pointer_runs_to_cuda_dst(buffer, targets, sources, sizes)
        self._stats.dst_copies += 1
        self._stats.dst_copy_runs += len(sizes)
        self._stats.dst_copy_bytes += int(sizes.sum())
        await self._await_event(event, None)

    async def _await_event(self, event: Any, error: BaseException | None) -> None:
        """Poll a CUDA event until done, then raise ``error`` if any.

        A cancellation is deferred until the event completed, because the
        copy keeps reading its host sources until then.
        """
        cancellation: asyncio.CancelledError | None = None
        while not event.done:
            try:
                await asyncio.sleep(_EVENT_POLL_S)
            except asyncio.CancelledError as exc:
                cancellation = exc
        if error is not None:
            raise error
        if cancellation is not None:
            raise cancellation

    async def _run_close(self) -> None:
        """Shutdown body shared by every ``close`` caller."""
        await self._wait_idle()
        for error in self._errors:
            logger.warning("[TRANSFER:iouring] unreported store failure: %r", error)
        self._errors.clear()
        if self._l2 is not None:
            l2, self._l2 = self._l2, None
            await asyncio.to_thread(l2.close)
        self._release_resources()
        logger.debug("[TRANSFER:iouring] closed")

    def _release_resources(self) -> None:
        """Close the L2 engine (if still open) and free pinned memory."""
        if self._l2 is not None:
            self._l2.close()
            self._l2 = None
        if self._staging is not None:
            self._staging.close()
            self._staging = None
        if self._arena is not None:
            self._arena.close()
            self._arena = None
