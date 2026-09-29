# SPDX-License-Identifier: Apache-2.0
"""FIFO-safe placement of online records in a reclaimable byte arena."""

from dataclasses import dataclass
from typing import Any

from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import ChunkMeta


def _free_gaps(
    occupied: list[tuple[int, int]], *, start: int, end: int
) -> list[list[int]]:
    """Return the free ``[start, end)`` intervals of an arena in address order.

    Args:
        occupied: Live byte intervals, in any order; they may touch.
        start: Inclusive arena start.
        end: Exclusive arena end.

    Returns:
        Mutable ``[gap_start, gap_end]`` pairs sorted by address.
    """
    gaps: list[list[int]] = []
    cursor = start
    for occupied_start, occupied_end in sorted(occupied):
        if occupied_start > cursor:
            gaps.append([cursor, min(occupied_start, end)])
        cursor = max(cursor, occupied_end)
    if cursor < end:
        gaps.append([cursor, end])
    return gaps


def _reserve_first_fit(gaps: list[list[int]], nbytes: int) -> int | None:
    """Carve ``nbytes`` from the first gap that fits and return its offset.

    Args:
        gaps: Address-ordered free intervals; updated in place.
        nbytes: Aligned bytes to reserve.

    Returns:
        Start offset of the reservation, or None when no gap is large enough.
    """
    for index, gap in enumerate(gaps):
        if gap[1] - gap[0] < nbytes:
            continue
        offset = gap[0]
        gap[0] += nbytes
        if gap[0] == gap[1]:
            del gaps[index]
        return offset
    return None


class PackedArenaFull(MemoryError):
    """Raised when a packed batch does not fit the arena's free byte ranges.

    Attributes:
        deficit_bytes: Lower bound on bytes that must be reclaimed before a
            retry of the same batch can succeed.
    """

    def __init__(self, deficit_bytes: int, message: str) -> None:
        """Record the reclaim target.

        Args:
            deficit_bytes: Positive number of bytes to free before retrying.
            message: Human-readable error message.
        """
        super().__init__(message)
        self.deficit_bytes = deficit_bytes


@dataclass(frozen=True)
class _Placement:
    """Retain the exact allocation generation and its immutable representation."""

    owner: ChunkMeta
    file_offset: int
    nbytes: int
    mode: str


class OnlinePackedLayout:
    """Keep packed records contiguous while reclaiming evicted owners.

    One server event loop owns this bounded metadata map. No method performs
    IO or suspends; transfer and publication remain the caller's responsibility.
    """

    def __init__(self) -> None:
        self._placements: dict[tuple[int, int], _Placement] = {}

    def compact(
        self,
        spans: list[dict[str, Any]],
        manager: ChunkManager,
        *,
        local_slot_size: int,
        rank_base: int = 0,
        arena_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Place source-contiguous records into free physical byte ranges.

        Args:
            spans: Packed spans carrying current logical allocation metadata.
            manager: Public owner of current allocation identities and FIFO tail.
            local_slot_size: Positive aligned byte size of one raw local slot.
            rank_base: Nonnegative start of this rank's physical lane.
            arena_size: Physical byte capacity of this rank lane. When omitted,
                derive it from the manager's logical slot count for unit tests.

        Returns:
            Copied spans with server-assigned physical file offsets.

        Raises:
            ValueError: On stale ownership, invalid slot geometry, duplicate
                records or a changed representation of an existing generation.
            PackedArenaFull: When the new records do not fit the free byte
                ranges; no placement is changed, so the call can be retried
                after reclaiming ``deficit_bytes``.

        Async/thread-safety:
            Call on the server event loop, without yielding between validation
            and transfer reservation. Failed transfer keeps its placement for
            an idempotent retry; this method does not publish cache visibility.
        """
        if (
            local_slot_size <= 0
            or local_slot_size % 4096
            or rank_base < 0
            or rank_base % 4096
        ):
            raise ValueError("invalid packed layout geometry")
        if arena_size is None:
            arena_size = manager.total_slots * local_slot_size
        if arena_size <= 0 or arena_size % 4096:
            raise ValueError("invalid packed arena size")
        arena_end = rank_base + arena_size
        result = [dict(span) for span in spans]
        records: list[tuple[int, int, ChunkMeta, _Placement | None]] = []
        seen: set[int] = set()
        stale_keys: list[tuple[int, int]] = []
        occupied: list[tuple[int, int]] = []
        for key, placement in self._placements.items():
            if key[0] != rank_base:
                continue
            owner = manager.store.get(placement.owner.chunk_key)
            if owner is not placement.owner:
                stale_keys.append(key)
                continue
            end = placement.file_offset + placement.nbytes
            if placement.file_offset < rank_base or end > arena_end:
                raise ValueError("remembered packed placement is outside its arena")
            occupied.append((placement.file_offset, end))
        # Validate the whole call before changing any remembered placement.
        for index, span in enumerate(result):
            if not bool(span.get("packed", False)):
                continue
            owner = manager.store.get(str(span.get("chunk_key", "")))
            start = int(span.get("start_slot", -1))
            count = int(span.get("num_slots", 0))
            logical = int(span.get("logical_slot_start", -1))
            relative = 0 if count == 1 else logical
            if (
                owner is None
                or owner.start_slot != start
                or owner.num_slots != count
                or logical < 0
                or not 0 <= relative < count
                or int(span.get("logical_slot_count", 0)) != 1
            ):
                raise ValueError("packed layout requires a current slot allocation")
            slot = start + relative
            nbytes = int(span["nbytes"])
            if not 0 < nbytes <= local_slot_size or nbytes % 4096:
                raise ValueError("packed record must fit one aligned raw slot")
            if slot in seen:
                raise ValueError("packed layout repeats a logical slot")
            seen.add(slot)
            prior = self._placements.get((rank_base, slot))
            if prior is not None and prior.owner is not owner:
                prior = None
            if prior is not None and (
                prior.nbytes != nbytes
                or prior.mode != str(span.get("mode", "compressed"))
            ):
                raise ValueError("packed retry changes an assigned representation")
            if prior is not None and not (
                rank_base <= prior.file_offset
                and prior.file_offset + prior.nbytes <= arena_end
            ):
                raise ValueError("packed retry placement is outside its arena")
            records.append((index, slot, owner, prior))

        pending: dict[tuple[int, int], _Placement] = {}
        run: list[tuple[int, int, ChunkMeta]] = []
        # Build the free list once per call; every run carves from it, so
        # placement cost does not grow with repeated scans of the arena.
        gaps = _free_gaps(occupied, start=rank_base, end=arena_end)
        unplaced_bytes = sum(
            int(result[index]["nbytes"])
            for index, _slot, _owner, prior in records
            if prior is None
        )

        def arena_full(nbytes: int) -> PackedArenaFull:
            free_bytes = sum(gap_end - gap_start for gap_start, gap_end in gaps)
            return PackedArenaFull(
                max(nbytes, unplaced_bytes - free_bytes),
                f"packed arena exhausted: need={nbytes} "
                f"unplaced={unplaced_bytes} free={free_bytes}",
            )

        def finish_run() -> None:
            nonlocal unplaced_bytes
            if not run:
                return
            total_bytes = sum(int(result[index]["nbytes"]) for index, _, _ in run)
            cursor = _reserve_first_fit(gaps, total_bytes)
            offsets = []
            if cursor is not None:
                for index, _, _ in run:
                    offsets.append(cursor)
                    cursor += int(result[index]["nbytes"])
                unplaced_bytes -= total_bytes
            else:
                # Fragmented arena: place records individually.
                for index, _, _ in run:
                    nbytes = int(result[index]["nbytes"])
                    offset = _reserve_first_fit(gaps, nbytes)
                    if offset is None:
                        raise arena_full(nbytes)
                    offsets.append(offset)
                    unplaced_bytes -= nbytes
            for (index, slot, owner), cursor in zip(run, offsets, strict=True):
                span = result[index]
                nbytes = int(span["nbytes"])
                span["file_offset"] = cursor
                pending[(rank_base, slot)] = _Placement(
                    owner, cursor, nbytes, str(span.get("mode", "compressed"))
                )
            run.clear()

        for index, slot, owner, prior in records:
            if prior is not None:
                finish_run()
                result[index]["file_offset"] = prior.file_offset
                continue
            if run:
                prev_index, prev_slot, _prev_owner = run[-1]
                prev_span = result[prev_index]
                if index != prev_index + 1 or int(
                    result[index]["source_offset"]
                ) != int(prev_span["source_offset"]) + int(prev_span["nbytes"]):
                    finish_run()
            run.append((index, slot, owner))
        finish_run()
        for key in stale_keys:
            self._placements.pop(key, None)
        self._placements.update(pending)
        return result

    def owned_bytes(self, owner: ChunkMeta, *, rank_base: int = 0) -> int:
        """Return packed arena bytes placed for one allocation generation.

        Args:
            owner: Chunk metadata of the allocation, typically captured just
                before it is evicted.
            rank_base: Start of the rank lane to inspect.

        Returns:
            Bytes that become reclaimable once ``owner`` is no longer current.

        Async/thread-safety:
            Pure lookup on the server event loop; costs O(owner.num_slots).
        """
        total = 0
        for slot in range(owner.start_slot, owner.start_slot + owner.num_slots):
            placement = self._placements.get((rank_base, slot))
            if placement is not None and placement.owner is owner:
                total += placement.nbytes
        return total
