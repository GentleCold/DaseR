# SPDX-License-Identifier: Apache-2.0

"""Range-keyed pinned-host LRU cache for the io_uring L1 tier.

This cache maps L2 byte ranges to pinned-memory slices and enforces an LRU
capacity bound. It is backend-agnostic and holds no io_uring or asyncio state:
the transfer-layer orchestrator owns the metadata lock and calls these methods
with it held. Because an in-flight L2 write can pin a pool slice, the cache
asks the orchestrator whether a slice is still pinned through an injected
predicate before closing it on eviction.
"""

# Standard
import bisect
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

# First Party
from daser.replacement import LRUReplacementPolicy
from daser.transfer.iouring.pinned_pool import PinnedMemoryPool, PinnedMemorySlice


@dataclass(frozen=True)
class L1RangeHit:
    """One L1-backed subrange inside a requested load span."""

    target_offset: int
    key: tuple[int, int]
    data: PinnedMemorySlice
    source_offset: int
    nbytes: int


class L1Cache:
    """Pinned-host LRU cache keyed by ``(file_offset, nbytes)`` byte ranges.

    Args:
        l1_bytes: maximum resident bytes in the memory tier.
        alignment: pinned-pool allocation alignment for O_DIRECT compatibility.
        pinned_predicate: returns True when a ``(key, slice)`` pair is still
            owned by an in-flight L2 write and must not be closed on eviction.

    Async/thread-safety:
        Not internally synchronized. All methods assume the orchestrator's
        metadata lock is held; ``register_pool_waiter`` futures are resolved by
        ``notify_pool_waiters`` after space frees up.
    """

    def __init__(
        self,
        l1_bytes: int,
        alignment: int,
        pinned_predicate: Callable[[tuple[int, int], PinnedMemorySlice], bool],
    ) -> None:
        self._l1_bytes = l1_bytes
        self._pool = PinnedMemoryPool(l1_bytes, alignment=alignment)
        self._entries: OrderedDict[tuple[int, int], PinnedMemorySlice] = OrderedDict()
        self._starts: list[int] = []
        self._by_start: dict[int, tuple[int, int]] = {}
        self._used = 0
        self._policy = LRUReplacementPolicy[tuple[int, int]]()
        self._pool_waiters: list[object] = []
        self._is_pinned = pinned_predicate

    @property
    def capacity_bytes(self) -> int:
        """Return the configured L1 capacity in bytes."""
        return self._l1_bytes

    @property
    def bytes_used(self) -> int:
        """Return resident L1 bytes."""
        return self._used

    def get(self, key: tuple[int, int]) -> PinnedMemorySlice | None:
        """Return the resident slice for ``key`` or None."""
        return self._entries.get(key)

    def contains_slice(self, data: PinnedMemorySlice) -> bool:
        """Return whether ``data`` is currently a resident L1 slice."""
        return data in self._entries.values()

    def resident_slice_ids(self) -> set[int]:
        """Return ``id()`` of every resident slice for liveness checks."""
        return {id(buffer) for buffer in self._entries.values()}

    def close(self) -> None:
        """Release the pinned pool backing this cache."""
        self._pool.close()

    def find(
        self, file_offset: int
    ) -> tuple[tuple[int, int], PinnedMemorySlice, int] | None:
        """Return the cached range containing ``file_offset``.

        Args:
            file_offset: L2 byte offset to locate.

        Returns:
            ``(key, slice, source_offset)`` when a resident range covers the
            offset, otherwise None.
        """
        idx = bisect.bisect_right(self._starts, file_offset) - 1
        if idx < 0:
            return None
        start = self._starts[idx]
        key = self._by_start.get(start)
        if key is None:
            return None
        data = self._entries.get(key)
        if data is None:
            return None
        if file_offset < key[0] + key[1]:
            return key, data, file_offset - key[0]
        return None

    def resolve_subranges(
        self,
        target_offset: int,
        file_offset: int,
        nbytes: int,
    ) -> tuple[list[L1RangeHit], list[dict[str, int]]]:
        """Split a load span into cached slices and uncached gaps.

        Args:
            target_offset: destination byte offset matching ``file_offset``.
            file_offset: L2 byte offset where the requested span starts.
            nbytes: requested byte count.

        Returns:
            A pair of L1 hit slices and L2 miss gaps in ascending file-offset
            order.
        """
        hits: list[L1RangeHit] = []
        misses: list[dict[str, int]] = []
        request_end = file_offset + nbytes
        cursor = file_offset
        while cursor < request_end:
            hit = self.find(cursor)
            if hit is not None:
                key, data, source_offset = hit
                covered = min(key[0] + key[1], request_end) - cursor
                hits.append(
                    L1RangeHit(
                        target_offset=target_offset + (cursor - file_offset),
                        key=key,
                        data=data,
                        source_offset=source_offset,
                        nbytes=covered,
                    )
                )
                cursor += covered
                continue

            next_idx = bisect.bisect_left(self._starts, cursor)
            next_start = (
                self._starts[next_idx] if next_idx < len(self._starts) else request_end
            )
            gap_end = min(next_start, request_end)
            if gap_end <= cursor:
                gap_end = request_end
            misses.append(
                {
                    "target_offset": target_offset + (cursor - file_offset),
                    "file_offset": cursor,
                    "nbytes": gap_end - cursor,
                }
            )
            cursor = gap_end
        return hits, misses

    def record_hits(self, hits: list[L1RangeHit]) -> None:
        """Refresh LRU recency for hit slices.

        Args:
            hits: slices returned by ``resolve_subranges``.
        """
        for hit in hits:
            self._policy.access(hit.key)
            self._entries.move_to_end(hit.key)

    def touch(self, key: tuple[int, int]) -> None:
        """Refresh LRU recency for one resident key (used on in-place stores)."""
        self._policy.access(key)
        self._entries.move_to_end(key)

    def put(self, key: tuple[int, int], data: PinnedMemorySlice) -> None:
        """Insert bytes into L1 after dropping overlapping ranges.

        Args:
            key: ``(file_offset, nbytes)`` range key.
            data: pinned slice holding the range's bytes.
        """
        self.drop_overlapping(key[0], key[1])
        self._insert_entry(key, data)

    def reserve(
        self,
        key: tuple[int, int],
        nbytes: int,
        *,
        drop_overlaps: bool = True,
        preserve_overlaps: bool = False,
    ) -> PinnedMemorySlice | None:
        """Try to reserve pinned space for a store or promoted load.

        Args:
            key: range key being inserted.
            nbytes: logical bytes needed.
            drop_overlaps: drop resident ranges overlapping ``key`` first.
            preserve_overlaps: keep the non-overlapping remainder of dropped
                ranges when ``drop_overlaps`` is set.

        Returns:
            A pinned slice, or None when the pool is exhausted and no further
            victim can be evicted (the caller must wait for an in-flight L2
            write to free its slice, then retry).

        Raises:
            ValueError: if ``nbytes`` exceeds the L1 capacity.
        """
        if nbytes > self._l1_bytes:
            raise ValueError(
                f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
            )
        if drop_overlaps:
            self.drop_overlapping(key[0], key[1], preserve_remainder=preserve_overlaps)
        data = self._pool.allocate(nbytes)
        while data is None:
            victim = self._policy.evict()
            if victim is None:
                return None
            removed = self._entries.pop(victim, None)
            self._remove_index(victim)
            if removed is not None:
                self._used -= len(removed)
                self.release(victim, removed)
            data = self._pool.allocate(nbytes)
        return data

    def reserve_or_raise(
        self,
        key: tuple[int, int],
        nbytes: int,
        *,
        preserve_overlaps: bool = False,
    ) -> PinnedMemorySlice:
        """Reserve pinned space when no in-flight L2 writer can block reuse.

        Args:
            key: range key being inserted.
            nbytes: logical bytes needed.
            preserve_overlaps: keep non-overlapping remainders of dropped ranges.

        Returns:
            A pinned slice.

        Raises:
            MemoryError: if the pool cannot satisfy the request.
        """
        data = self.reserve(key, nbytes, preserve_overlaps=preserve_overlaps)
        if data is None:
            raise MemoryError(
                f"could not reserve {nbytes} pinned L1 bytes from "
                f"{self._l1_bytes} byte pool"
            )
        return data

    def release(self, key: tuple[int, int], data: PinnedMemorySlice) -> None:
        """Close an evicted L1 slice unless an L2 write still owns it.

        Args:
            key: range key being released.
            data: slice removed from the cache.
        """
        if self._is_pinned(key, data):
            return
        data.close()
        self.notify_pool_waiters()

    def register_pool_waiter(self, waiter: object) -> None:
        """Register a future to wake when pool space or metadata changes."""
        self._pool_waiters.append(waiter)

    def notify_pool_waiters(self) -> None:
        """Wake futures waiting for L1 pool metadata or free-space changes."""
        waiters = self._pool_waiters
        self._pool_waiters = []
        for waiter in waiters:
            if not waiter.done():  # type: ignore[attr-defined]
                waiter.set_result(None)  # type: ignore[attr-defined]

    def drop_overlapping(
        self,
        file_offset: int,
        nbytes: int,
        *,
        preserve_remainder: bool = False,
    ) -> None:
        """Remove L1 entries overlapping a newly written byte range.

        Args:
            file_offset: start of the overwritten range.
            nbytes: length of the overwritten range.
            preserve_remainder: re-insert non-overlapping fragments of dropped
                ranges as fresh L1 entries.
        """
        end = file_offset + nbytes
        victims = [
            key
            for key in self._entries
            if key[0] < end and file_offset < key[0] + key[1]
        ]
        for victim in victims:
            removed = self._entries.pop(victim, None)
            self._remove_index(victim)
            self._policy.remove(victim)
            preserved = (
                self._preserve_non_overlapping(victim, removed, file_offset, end)
                if preserve_remainder and removed is not None
                else []
            )
            if removed is not None:
                self._used -= len(removed)
                self.release(victim, removed)
            for preserved_key, payload in preserved:
                self._put_preserved_fragment(preserved_key, payload)

    def _insert_entry(self, key: tuple[int, int], data: PinnedMemorySlice) -> None:
        """Insert one non-overlapping entry and enforce capacity."""
        if len(data) > self._l1_bytes:
            return
        self._entries[key] = data
        self._insert_index(key)
        self._entries.move_to_end(key)
        self._policy.insert(key)
        self._used += len(data)
        self.notify_pool_waiters()
        while self._used > self._l1_bytes:
            victim = self._policy.evict()
            if victim is None:
                break
            removed = self._entries.pop(victim, None)
            self._remove_index(victim)
            if removed is not None:
                self._used -= len(removed)
                self.release(victim, removed)

    def _preserve_non_overlapping(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
        overlap_start: int,
        overlap_end: int,
    ) -> list[tuple[tuple[int, int], bytes]]:
        """Return old L1 fragments that fall outside an overwrite range."""
        key_start, key_size = key
        key_end = key_start + key_size
        fragments: list[tuple[tuple[int, int], bytes]] = []
        view = data.view()
        if key_start < overlap_start:
            keep = overlap_start - key_start
            fragments.append(((key_start, keep), bytes(view[:keep])))
        if overlap_end < key_end:
            source_offset = overlap_end - key_start
            keep = key_end - overlap_end
            fragments.append(
                (
                    (overlap_end, keep),
                    bytes(view[source_offset : source_offset + keep]),
                )
            )
        return fragments

    def _put_preserved_fragment(self, key: tuple[int, int], payload: bytes) -> None:
        """Insert one fragment copied out of an overwritten L1 range."""
        if not payload:
            return
        data = self.reserve(key, len(payload), drop_overlaps=False)
        if data is None:
            raise MemoryError(
                f"could not preserve {len(payload)} L1 bytes from overwritten range"
            )
        try:
            data.view()[: len(payload)] = payload
        except BaseException:
            data.close()
            raise
        self._insert_entry(key, data)

    def _insert_index(self, key: tuple[int, int]) -> None:
        """Add one range to the start-offset lookup index."""
        start = key[0]
        existing = self._by_start.get(start)
        if existing == key:
            return
        if existing is not None:
            self._remove_index(existing)
        bisect.insort(self._starts, start)
        self._by_start[start] = key

    def _remove_index(self, key: tuple[int, int]) -> None:
        """Remove one range from the start-offset lookup index."""
        start = key[0]
        if self._by_start.get(start) != key:
            return
        del self._by_start[start]
        idx = bisect.bisect_left(self._starts, start)
        if idx < len(self._starts) and self._starts[idx] == start:
            self._starts.pop(idx)
