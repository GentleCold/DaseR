# SPDX-License-Identifier: Apache-2.0

"""Chunk commit/eviction/ownership state for the DaseR control plane."""

# Standard
import asyncio


class ChunkLifecycle:
    """Track commit, write-ownership, eviction, and commit-waiter state.

    ServerCore keeps several parallel sets in lockstep: which chunk keys are
    committed and visible to lookup, which have an active store writer, and
    which were evicted. This class owns those sets and the commit-waiter
    futures so the transitions stay consistent in one place.

    Async/thread-safety:
        All methods run on the server asyncio event loop. Commit waiters are
        completed from the same loop by ``mark_committed``.
    """

    def __init__(self) -> None:
        self._committed: set[str] = set()
        self._write_owners: set[str] = set()
        self._evicted: set[str] = set()
        self._commit_waiters: dict[str, set[asyncio.Future[None]]] = {}
        self._commit_shards: dict[str, tuple[int, set[int]]] = {}
        self._publishing: set[str] = set()
        self._written_ranges: dict[tuple[str, int], list[tuple[int, int]]] = {}
        self._write_expectations: dict[tuple[str, int], tuple[int, int]] = {}
        self._complete_write_ranks: set[tuple[str, int]] = set()

    def is_committed(self, chunk_key: str) -> bool:
        """Return whether ``chunk_key`` is committed and visible to lookup."""
        return chunk_key in self._committed

    def is_evicted(self, chunk_key: str) -> bool:
        """Return whether ``chunk_key`` was evicted."""
        return chunk_key in self._evicted

    @property
    def write_owners(self) -> set[str]:
        """Return the live write-owner key set (used for reuse predicates)."""
        return self._write_owners

    @property
    def committed(self) -> set[str]:
        """Return the committed key set (used for reuse predicates)."""
        return self._committed

    def mark_write_owner(self, chunk_key: str) -> None:
        """Record that a store writer claimed ``chunk_key``."""
        self._write_owners.add(chunk_key)

    def mark_committed(self, chunk_key: str) -> None:
        """Mark ``chunk_key`` committed, claim ownership, and wake waiters."""
        self._committed.add(chunk_key)
        self._write_owners.add(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_written_ranges(chunk_key)
        self._notify_commit_waiters(chunk_key)

    def record_written_range(
        self,
        chunk_key: str,
        tp_rank: int,
        expected_start: int,
        expected_end: int,
        range_start: int,
        range_end: int,
    ) -> bool:
        """Record an accepted transfer range and report complete coverage.

        Args:
            chunk_key: Cache key whose bytes were transferred.
            tp_rank: Tensor-parallel rank owning the range.
            expected_start: First file byte required for this rank's chunk.
            expected_end: Exclusive file byte after this rank's chunk.
            range_start: First file byte accepted by the transfer.
            range_end: Exclusive file byte accepted by the transfer.

        Returns:
            True once, when this rank has covered the complete expected range.

        Raises:
            ValueError: if the range is invalid or the expected range changes.

        Async/thread-safety:
            Runs on the server event loop. Ranges are merged in memory and do
            not perform blocking I/O.
        """
        if expected_start < 0 or expected_end <= expected_start:
            raise ValueError("invalid expected write range")
        if range_start < expected_start or range_end > expected_end:
            raise ValueError("write range exceeds the allocated chunk")
        if range_end <= range_start:
            raise ValueError("write range must be non-empty")
        identity = (chunk_key, tp_rank)
        expected = self._write_expectations.setdefault(
            identity, (expected_start, expected_end)
        )
        if expected != (expected_start, expected_end):
            raise ValueError("inconsistent expected write range")
        if identity in self._complete_write_ranks:
            return False

        ranges = self._written_ranges.setdefault(identity, [])
        ranges.append((range_start, range_end))
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._written_ranges[identity] = merged
        if (
            len(merged) == 1
            and merged[0][0] <= expected_start
            and merged[0][1] >= expected_end
        ):
            self._complete_write_ranks.add(identity)
            return True
        return False

    def record_commit_shard(self, chunk_key: str, tp_rank: int, tp_size: int) -> bool:
        """Record one TP rank and return whether this call should publish.

        Args:
            chunk_key: Cache key for the pending chunk.
            tp_rank: Tensor-parallel rank that completed its store.
            tp_size: Required number of tensor-parallel ranks.

        Returns:
            True exactly once when all distinct ranks have arrived.

        Raises:
            ValueError: if rank metadata is invalid or changes mid-commit.

        Async/thread-safety:
            Runs on the server event loop. ``_publishing`` prevents another
            request from publishing while the retrieval insert awaits.
        """
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP rank {tp_rank} for size {tp_size}")
        if chunk_key in self._committed or chunk_key in self._publishing:
            return False
        expected_size, ranks = self._commit_shards.setdefault(
            chunk_key, (tp_size, set())
        )
        if expected_size != tp_size:
            raise ValueError(
                f"inconsistent TP size for chunk: {tp_size} != {expected_size}"
            )
        ranks.add(tp_rank)
        if len(ranks) != tp_size:
            return False
        self._publishing.add(chunk_key)
        return True

    def abort_publish(self, chunk_key: str) -> None:
        """Allow a failed retrieval-index publish to be retried."""
        self._publishing.discard(chunk_key)

    def mark_evicted(self, chunk_key: str) -> None:
        """Drop committed/owner state for ``chunk_key`` and record eviction."""
        self._committed.discard(chunk_key)
        self._write_owners.discard(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_written_ranges(chunk_key)
        self._evicted.add(chunk_key)

    def discard_owner(self, chunk_key: str) -> None:
        """Release a store-writer claim without committing or evicting."""
        self._write_owners.discard(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_written_ranges(chunk_key)

    def discard(self, chunk_key: str) -> None:
        """Drop committed and write-owner state without recording eviction."""
        self._committed.discard(chunk_key)
        self._write_owners.discard(chunk_key)
        self._commit_shards.pop(chunk_key, None)
        self._publishing.discard(chunk_key)
        self._clear_written_ranges(chunk_key)

    def _clear_written_ranges(self, chunk_key: str) -> None:
        """Remove transfer coverage state for a chunk."""
        identities = [
            identity
            for identity in self._write_expectations
            if identity[0] == chunk_key
        ]
        for identity in identities:
            self._write_expectations.pop(identity, None)
            self._written_ranges.pop(identity, None)
            self._complete_write_ranks.discard(identity)

    async def wait_for_committed(
        self,
        chunk_keys: list[str],
        timeout_s: float,
    ) -> None:
        """Wait until every key in ``chunk_keys`` is committed.

        Args:
            chunk_keys: keys to await; duplicates and already-committed keys
                are ignored.
            timeout_s: maximum seconds to wait.

        Raises:
            TimeoutError: if any key is still uncommitted at timeout.
        """
        pending = [
            key for key in dict.fromkeys(chunk_keys) if key not in self._committed
        ]
        if not pending:
            return

        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[None]] = {}
        for key in pending:
            future: asyncio.Future[None] = loop.create_future()
            self._commit_waiters.setdefault(key, set()).add(future)
            futures[key] = future
        try:
            await asyncio.wait_for(
                asyncio.gather(*futures.values()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("timed out waiting for committed chunks") from exc
        finally:
            for key, future in futures.items():
                waiters = self._commit_waiters.get(key)
                if waiters is None:
                    continue
                waiters.discard(future)
                if not waiters:
                    self._commit_waiters.pop(key, None)

    def _notify_commit_waiters(self, chunk_key: str) -> None:
        """Complete and clear any commit waiters for ``chunk_key``."""
        waiters = self._commit_waiters.pop(chunk_key, set())
        for future in waiters:
            if not future.done():
                future.set_result(None)
