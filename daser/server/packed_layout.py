# SPDX-License-Identifier: Apache-2.0
"""FIFO-safe placement of online records in a reclaimable byte arena."""

from dataclasses import dataclass
from typing import Any

from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import ChunkMeta


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

        def finish_run() -> None:
            if not run:
                return
            total_bytes = sum(int(result[index]["nbytes"]) for index, _, _ in run)
            try:
                cursor = self._find_free_extent(
                    occupied,
                    start=rank_base,
                    end=arena_end,
                    nbytes=total_bytes,
                )
                offsets = []
                for index, _, _ in run:
                    nbytes = int(result[index]["nbytes"])
                    offsets.append(cursor)
                    cursor += nbytes
                occupied.extend(
                    [
                        (offset, offset + int(result[index]["nbytes"]))
                        for offset, (index, _, _) in zip(offsets, run, strict=True)
                    ]
                )
            except MemoryError:
                trial_occupied = list(occupied)
                offsets = []
                for index, _, _ in run:
                    nbytes = int(result[index]["nbytes"])
                    offset = self._find_free_extent(
                        trial_occupied,
                        start=rank_base,
                        end=arena_end,
                        nbytes=nbytes,
                    )
                    offsets.append(offset)
                    trial_occupied.append((offset, offset + nbytes))
                occupied[:] = trial_occupied
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

    @staticmethod
    def _find_free_extent(
        occupied: list[tuple[int, int]],
        *,
        start: int,
        end: int,
        nbytes: int,
    ) -> int:
        """Return the first aligned gap large enough for one new run.

        Args:
            occupied: Existing live or tentatively reserved byte intervals.
            start: Inclusive arena start.
            end: Exclusive arena end.
            nbytes: Aligned bytes required by the run.

        Returns:
            The first byte offset that can hold the complete run.

        Raises:
            MemoryError: If the arena has insufficient free bytes.

        Async/thread-safety:
            Pure CPU planning on the server event loop; it never blocks.
        """
        if nbytes <= 0 or nbytes % 4096:
            raise ValueError("packed extent size must be positive and aligned")
        cursor = start
        for occupied_start, occupied_end in sorted(occupied):
            if occupied_end <= cursor:
                continue
            if occupied_start >= cursor + nbytes:
                return cursor
            cursor = max(cursor, occupied_end)
        if cursor + nbytes <= end:
            return cursor
        raise MemoryError(
            f"packed arena exhausted: need={nbytes} free_end={max(0, end - cursor)}"
        )
