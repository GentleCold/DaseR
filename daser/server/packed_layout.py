# SPDX-License-Identifier: Apache-2.0
"""FIFO-safe placement of online records in a reclaimable byte arena."""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import ChunkMeta


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


@dataclass(eq=False)
class _Placement:
    """Retain the exact allocation generation and its immutable representation.

    ``writers`` counts open write holds; a placement dropped while held is
    ``retired`` and keeps its bytes until the last hold ends.
    """

    owner: ChunkMeta
    file_offset: int
    nbytes: int
    mode: str
    writers: int = 0
    retired: bool = False


def _span_slot(span: dict[str, Any]) -> int:
    """Return the logical slot a single-slot packed span describes."""
    start = int(span.get("start_slot", -1))
    count = int(span.get("num_slots", 0))
    return start + (0 if count == 1 else int(span.get("logical_slot_start", -1)))


class _FreeExtents:
    """Address-ordered free byte ranges of one rank lane.

    Adjacent ranges are always coalesced, so the list length equals the number
    of holes between live records.
    """

    def __init__(self, start: int, end: int) -> None:
        """Start with the whole lane ``[start, end)`` free."""
        self.start = start
        self.end = end
        self.free_bytes = end - start
        self._starts: list[int] = [start]
        self._ends: list[int] = [end]

    def reserve(self, nbytes: int) -> int | None:
        """Carve ``nbytes`` from the first range that fits; return its offset."""
        for index, (start, end) in enumerate(
            zip(self._starts, self._ends, strict=True)
        ):
            if end - start < nbytes:
                continue
            if end - start == nbytes:
                del self._starts[index]
                del self._ends[index]
            else:
                self._starts[index] = start + nbytes
            self.free_bytes -= nbytes
            return start
        return None

    def free(self, offset: int, nbytes: int) -> None:
        """Return ``[offset, offset + nbytes)`` and merge it with neighbours."""
        end = offset + nbytes
        index = bisect_left(self._starts, offset)
        merge_prev = index > 0 and self._ends[index - 1] == offset
        merge_next = index < len(self._starts) and self._starts[index] == end
        if merge_prev and merge_next:
            self._ends[index - 1] = self._ends[index]
            del self._starts[index]
            del self._ends[index]
        elif merge_prev:
            self._ends[index - 1] = end
        elif merge_next:
            self._starts[index] = offset
        else:
            self._starts.insert(index, offset)
            self._ends.insert(index, end)
        self.free_bytes += nbytes


class OnlinePackedLayout:
    """Keep packed records contiguous while reclaiming evicted owners.

    One server event loop owns this bounded metadata map. No method performs
    IO or suspends; transfer and publication remain the caller's responsibility.
    Free space is tracked incrementally: callers report chunks that left the
    metadata store through ``release``, so placement cost does not grow with
    the number of live records. Bytes under an open write hold stay reserved
    after their owner leaves, so a later record cannot be placed over bytes a
    late write may still target.
    """

    def __init__(self) -> None:
        self._placements: dict[tuple[int, int], _Placement] = {}
        self._keys_by_chunk: dict[str, set[tuple[int, int]]] = {}
        self._arenas: dict[int, _FreeExtents] = {}
        self._holds: dict[int, list[tuple[int, _Placement]]] = {}
        self._next_hold = 0

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
                records, a changed representation of an existing generation or
                a lane whose arena size changes.
            PackedArenaFull: When the new records do not fit the free byte
                ranges. No live placement is changed, so the call can be
                retried after reclaiming ``deficit_bytes``.

        Async/thread-safety:
            Call on the server event loop, without yielding between validation
            and transfer reservation. Failed transfer keeps its placement for
            an idempotent retry; this method does not publish cache visibility.
            Costs O(len(spans) x free holes), independent of live records.
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
        arena = self._arena(rank_base, arena_size)
        result = [dict(span) for span in spans]
        records: list[tuple[int, int, ChunkMeta, _Placement | None]] = []
        seen: set[int] = set()
        stale: list[tuple[tuple[int, int], _Placement]] = []
        # Validate the whole call before changing any remembered placement.
        for index, span in enumerate(result):
            if not bool(span.get("packed", False)):
                continue
            owner = manager.store.get(str(span.get("chunk_key", "")))
            start = int(span.get("start_slot", -1))
            count = int(span.get("num_slots", 0))
            logical = int(span.get("logical_slot_start", -1))
            slot = _span_slot(span)
            relative = slot - start
            if (
                owner is None
                or owner.start_slot != start
                or owner.num_slots != count
                or logical < 0
                or not 0 <= relative < count
                or int(span.get("logical_slot_count", 0)) != 1
            ):
                raise ValueError("packed layout requires a current slot allocation")
            nbytes = int(span["nbytes"])
            if not 0 < nbytes <= local_slot_size or nbytes % 4096:
                raise ValueError("packed record must fit one aligned raw slot")
            if slot in seen:
                raise ValueError("packed layout repeats a logical slot")
            seen.add(slot)
            prior = self._placements.get((rank_base, slot))
            if prior is not None and prior.owner is not owner:
                # An earlier generation of this slot that was never released.
                stale.append(((rank_base, slot), prior))
                prior = None
            if prior is not None and (
                prior.nbytes != nbytes
                or prior.mode != str(span.get("mode", "compressed"))
            ):
                raise ValueError("packed retry changes an assigned representation")
            records.append((index, slot, owner, prior))
        for key, placement in stale:
            self._drop(key, placement)

        pending: dict[tuple[int, int], _Placement] = {}
        run: list[tuple[int, int, ChunkMeta]] = []
        unplaced_bytes = sum(
            int(result[index]["nbytes"])
            for index, _slot, _owner, prior in records
            if prior is None
        )

        def reserve_run() -> list[int] | None:
            total_bytes = sum(int(result[index]["nbytes"]) for index, _, _ in run)
            cursor = arena.reserve(total_bytes)
            if cursor is not None:
                offsets = []
                for index, _, _ in run:
                    offsets.append(cursor)
                    cursor += int(result[index]["nbytes"])
                return offsets
            # Fragmented arena: place records individually.
            offsets = []
            for index, _, _ in run:
                nbytes = int(result[index]["nbytes"])
                offset = arena.reserve(nbytes)
                if offset is None:
                    # ``offsets`` holds only the records reserved so far.
                    for (prev_index, _, _), prev in zip(run, offsets, strict=False):
                        arena.free(prev, int(result[prev_index]["nbytes"]))
                    return None
                offsets.append(offset)
            return offsets

        def finish_run() -> None:
            nonlocal unplaced_bytes
            if not run:
                return
            offsets = reserve_run()
            if offsets is None:
                # Roll back earlier runs so a failed call reserves nothing.
                for placement in pending.values():
                    arena.free(placement.file_offset, placement.nbytes)
                free_bytes = arena.free_bytes
                need = int(result[run[0][0]]["nbytes"])
                raise PackedArenaFull(
                    max(need, unplaced_bytes - free_bytes),
                    f"packed arena exhausted: need={need} "
                    f"unplaced={unplaced_bytes} free={free_bytes}",
                )
            for (index, slot, owner), cursor in zip(run, offsets, strict=True):
                span = result[index]
                nbytes = int(span["nbytes"])
                span["file_offset"] = cursor
                unplaced_bytes -= nbytes
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
                prev_index, _prev_slot, _prev_owner = run[-1]
                prev_span = result[prev_index]
                if index != prev_index + 1 or int(
                    result[index]["source_offset"]
                ) != int(prev_span["source_offset"]) + int(prev_span["nbytes"]):
                    finish_run()
            run.append((index, slot, owner))
        finish_run()
        for key, placement in pending.items():
            self._placements[key] = placement
            self._keys_by_chunk.setdefault(placement.owner.chunk_key, set()).add(key)
        return result

    def begin_writes(self, spans: list[dict[str, Any]], *, rank_base: int = 0) -> int:
        """Keep the bytes of placed packed spans reserved until a write ends.

        Args:
            spans: Spans returned by ``compact`` for one transfer. Non-packed
                spans are ignored.
            rank_base: Start of the rank lane the spans were placed in.

        Returns:
            Hold token to pass to ``end_writes`` exactly once.

        Raises:
            ValueError: If a packed span does not match a current placement.

        Async/thread-safety:
            Call on the server event loop right after ``compact``, before
            yielding, so no release can run in between.
        """
        held: list[tuple[int, _Placement]] = []
        for span in spans:
            if not bool(span.get("packed", False)):
                continue
            placement = self._placements.get((rank_base, _span_slot(span)))
            if placement is None or placement.file_offset != int(span["file_offset"]):
                raise ValueError("write hold requires a current packed placement")
            held.append((rank_base, placement))
        for _base, placement in held:
            placement.writers += 1
        token = self._next_hold
        self._next_hold += 1
        self._holds[token] = held
        return token

    def end_writes(self, token: int) -> None:
        """End a write hold and free bytes whose owner left meanwhile.

        Args:
            token: Value returned by ``begin_writes``.

        Raises:
            KeyError: If the token is unknown or already ended.

        Async/thread-safety:
            Call on the server event loop once the transfer layer has the
            write ordered (returned or failed).
        """
        for rank_base, placement in self._holds.pop(token):
            placement.writers -= 1
            if placement.retired and placement.writers == 0:
                self._arenas[rank_base].free(placement.file_offset, placement.nbytes)

    def release(self, chunk_key: str, manager: ChunkManager) -> None:
        """Reclaim the bytes of a chunk that is no longer a current allocation.

        Args:
            chunk_key: Key that left the metadata store.
            manager: Public owner of current allocation identities.

        Async/thread-safety:
            Call on the server event loop after the chunk is removed from
            ``manager.store``. Placements of a newer generation under the same
            key are kept. Costs O(slots of the chunk).
        """
        current = manager.store.get(chunk_key)
        for key in list(self._keys_by_chunk.get(chunk_key, ())):
            placement = self._placements.get(key)
            if placement is not None and placement.owner is not current:
                self._drop(key, placement)

    def free_bytes(self, *, rank_base: int = 0) -> int:
        """Return unreserved bytes of a rank lane.

        Args:
            rank_base: Start of the rank lane to inspect.

        Returns:
            Free bytes, or 0 before the lane's first placement.

        Async/thread-safety:
            Pure lookup on the server event loop.
        """
        arena = self._arenas.get(rank_base)
        return 0 if arena is None else arena.free_bytes

    def _arena(self, rank_base: int, arena_size: int) -> _FreeExtents:
        """Return the free-extent list of a lane, creating it on first use."""
        arena = self._arenas.get(rank_base)
        if arena is None:
            arena = _FreeExtents(rank_base, rank_base + arena_size)
            self._arenas[rank_base] = arena
        elif arena.end - arena.start != arena_size:
            raise ValueError("packed arena size changed for a rank lane")
        return arena

    def _drop(self, key: tuple[int, int], placement: _Placement) -> None:
        """Forget one placement and return its bytes unless a write holds them."""
        del self._placements[key]
        chunk_key = placement.owner.chunk_key
        keys = self._keys_by_chunk.get(chunk_key)
        if keys is not None:
            keys.discard(key)
            if not keys:
                del self._keys_by_chunk[chunk_key]
        if placement.writers:
            placement.retired = True
            return
        self._arenas[key[0]].free(placement.file_offset, placement.nbytes)
