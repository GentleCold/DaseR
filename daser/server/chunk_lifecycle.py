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
        self._notify_commit_waiters(chunk_key)

    def mark_evicted(self, chunk_key: str) -> None:
        """Drop committed/owner state for ``chunk_key`` and record eviction."""
        self._committed.discard(chunk_key)
        self._write_owners.discard(chunk_key)
        self._evicted.add(chunk_key)

    def discard_owner(self, chunk_key: str) -> None:
        """Release a store-writer claim without committing or evicting."""
        self._write_owners.discard(chunk_key)

    def discard(self, chunk_key: str) -> None:
        """Drop committed and write-owner state without recording eviction."""
        self._committed.discard(chunk_key)
        self._write_owners.discard(chunk_key)

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
