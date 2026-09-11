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
from daser.replacement import LRUReplacementPolicy, ReplacementPolicy
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
        replacement_policy: Optional policy for physical allocation victims.
            Omission retains LRU; the cache still owns all release barriers.

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
        replacement_policy: ReplacementPolicy[int] | None = None,
    ) -> None:
        self._l1_bytes = l1_bytes
        self._pool = PinnedMemoryPool(l1_bytes, alignment=alignment)
        self._entries: OrderedDict[tuple[int, int], PinnedMemorySlice] = OrderedDict()
        # Writer completion asks about a particular slice, not a byte range.
        # Keep its identity alongside the range index so every small packed
        # write does not scan all resident entries on the server event loop.
        self._slice_ids: set[int] = set()
        self._starts: list[int] = []
        self._by_start: dict[int, tuple[int, int]] = {}
        self._used = 0
        # A grouped D2H copy has independently addressable file ranges, but
        # its pool storage is reclaimed only after the last child closes.
        # Track replacement at that physical ownership boundary: evicting a
        # cold child of a hot allocation alone cannot make room for a load.
        self._allocation_keys: dict[int, dict[tuple[int, int], None]] = {}
        self._policy: ReplacementPolicy[int] = (
            LRUReplacementPolicy[int]()
            if replacement_policy is None
            else replacement_policy
        )
        self._pool_waiters: list[object] = []
        self._is_pinned = pinned_predicate

    @property
    def bytes_used(self) -> int:
        """Return resident L1 bytes."""
        return self._used

    def get(self, key: tuple[int, int]) -> PinnedMemorySlice | None:
        """Return the resident slice for ``key`` or None."""
        return self._entries.get(key)

    def contains_slice(self, data: PinnedMemorySlice) -> bool:
        """Return whether ``data`` is currently a resident L1 slice."""
        return id(data) in self._slice_ids

    def resident_slice_ids(self) -> set[int]:
        """Return ``id()`` of every resident slice for liveness checks."""
        return self._slice_ids.copy()

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

    def has_overlap(self, file_offset: int, nbytes: int) -> bool:
        """Return whether any resident range overlaps a requested span.

        Args:
            file_offset: Start of the requested byte range.
            nbytes: Number of requested bytes.

        Returns:
            ``True`` when at least one resident L1 range overlaps the span.
        """
        if nbytes <= 0:
            return False
        end = file_offset + nbytes
        index = max(0, bisect.bisect_right(self._starts, file_offset) - 1)
        while index < len(self._starts):
            start = self._starts[index]
            if start >= end:
                break
            key = self._by_start.get(start)
            if key is not None and key[0] + key[1] > file_offset:
                return True
            index += 1
        return False

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
            self._policy.access(hit.data.allocation_id)
            self._entries.move_to_end(hit.key)

    def touch(self, key: tuple[int, int]) -> None:
        """Refresh LRU recency for one resident key (used on in-place stores)."""
        self._policy.access(self._entries[key].allocation_id)
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
            if not self._evict_allocation():
                return None
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

    def reserve_untracked(self, nbytes: int) -> PinnedMemorySlice | None:
        """Reserve pool space without publishing a range-keyed entry.

        Args:
            nbytes: Total bytes for a temporary grouped allocation.

        Returns:
            A pinned slice, or ``None`` when all evictable pool space is
            currently retained by in-flight transfer owners.

        Raises:
            ValueError: If ``nbytes`` exceeds the L1 capacity.

        Thread-safety:
            Metadata and pool ownership must be protected by the transfer
            layer's asyncio lock. The caller must publish child entries with
            ``put_reserved_group`` or close the returned slice on failure.
        """
        if nbytes > self._l1_bytes:
            raise ValueError(
                f"range {nbytes} bytes exceeds L1 capacity {self._l1_bytes}"
            )
        data = self._pool.allocate(nbytes)
        while data is None:
            if not self._evict_allocation():
                return None
            data = self._pool.allocate(nbytes)
        return data

    def put_reserved_group(
        self,
        entries: list[tuple[tuple[int, int], PinnedMemorySlice]],
    ) -> None:
        """Publish entries backed by one previously reserved pool slice.

        Args:
            entries: Non-overlapping L1 keys and child slices sharing a
                grouped allocation.

        Thread-safety:
            Must be called while the transfer layer metadata lock is held.
            The grouped reservation already accounted for pool capacity, so
            entries are inserted without repeating overlap eviction.
        """
        for key, data in entries:
            self._insert_entry(key, data)

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
            preserve_remainder: re-insert non-overlapping fragments as child
                slices sharing the original physical allocation. The pool
                reclaims that allocation only after its final owner closes.

        Thread-safety:
            Requires the transfer metadata lock. Retained bytes are immutable
            while resident or borrowed by an asynchronous transfer.
        """
        end = file_offset + nbytes
        # Resident ranges are disjoint and already indexed by start offset.
        # Only the predecessor can begin before the overwrite and still
        # overlap it. Snapshot keys before mutating the index: preserving a
        # boundary fragment may insert entries.
        # Scanning every resident for every small packed store otherwise
        # turns publishing a batch into quadratic work on the event loop.
        first = max(0, bisect.bisect_right(self._starts, file_offset) - 1)
        stop = bisect.bisect_left(self._starts, end)
        victims = [
            self._by_start[start]
            for start in self._starts[first:stop]
            if file_offset < start + self._by_start[start][1]
        ]
        for victim in victims:
            removed = self._pop_entry(victim)
            preserved = (
                self._preserve_non_overlapping(victim, removed, file_offset, end)
                if preserve_remainder and removed is not None
                else []
            )
            if removed is not None:
                self._used -= len(removed)
                self.release(victim, removed)
            for preserved_key, fragment in preserved:
                self._insert_entry(preserved_key, fragment)

    def _insert_entry(self, key: tuple[int, int], data: PinnedMemorySlice) -> None:
        """Insert one non-overlapping entry and enforce capacity."""
        if len(data) > self._l1_bytes:
            return
        self._entries[key] = data
        self._slice_ids.add(id(data))
        self._insert_index(key)
        self._entries.move_to_end(key)
        allocation = data.allocation_id
        self._allocation_keys.setdefault(allocation, {})[key] = None
        self._policy.insert(allocation)
        self._used += len(data)
        self.notify_pool_waiters()
        while self._used > self._l1_bytes:
            if not self._evict_allocation():
                break

    def _evict_allocation(self) -> bool:
        """Drop all resident children of the least recently used allocation."""
        allocation = self._policy.evict()
        if allocation is None:
            return False
        for key in tuple(self._allocation_keys[allocation]):
            removed = self._pop_entry(key)
            if removed is not None:
                self._used -= len(removed)
                # An external writer/lease can outlive residency. Release
                # retains that child's ownership until its existing barrier;
                # replacement never makes in-flight pool bytes reusable.
                self.release(key, removed)
        return True

    def _pop_entry(self, key: tuple[int, int]) -> PinnedMemorySlice | None:
        """Detach one resident from both indexes before releasing its ownership."""
        removed = self._entries.pop(key, None)
        self._remove_index(key)
        if removed is not None:
            self._slice_ids.remove(id(removed))
            allocation = removed.allocation_id
            keys = self._allocation_keys[allocation]
            del keys[key]
            if not keys:
                del self._allocation_keys[allocation]
                self._policy.remove(allocation)
        return removed

    def _preserve_non_overlapping(
        self,
        key: tuple[int, int],
        data: PinnedMemorySlice,
        overlap_start: int,
        overlap_end: int,
    ) -> list[tuple[tuple[int, int], PinnedMemorySlice]]:
        """Borrow untouched fragments before releasing the old resident."""
        key_start, key_size = key
        key_end = key_start + key_size
        fragments: list[tuple[tuple[int, int], PinnedMemorySlice]] = []
        # Payload copies under the metadata lock can stall unrelated loads
        # for hundreds of milliseconds. Child leases retain immutable bytes
        # without copying or reserving again. Holes remain charged to the
        # original allocation until all children and external owners close;
        # bytes_used measures residency, not physical pool availability.
        if key_start < overlap_start:
            keep = overlap_start - key_start
            fragments.append(((key_start, keep), data.subslice(0, keep)))
        if overlap_end < key_end:
            source_offset = overlap_end - key_start
            keep = key_end - overlap_end
            fragments.append(
                (
                    (overlap_end, keep),
                    data.subslice(source_offset, keep),
                )
            )
        return fragments

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
