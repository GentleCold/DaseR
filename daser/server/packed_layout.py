# SPDX-License-Identifier: Apache-2.0
"""FIFO-safe placement of online records within existing ring envelopes."""

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
    """Keep packed records contiguous without reclaiming live FIFO data.

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
    ) -> list[dict[str, Any]]:
        """Tail-align new source-contiguous runs and reuse retry placements.

        Args:
            spans: Spans already validated inside their raw allocation bounds.
            manager: Public owner of current allocation identities and FIFO tail.
            local_slot_size: Positive aligned byte size of one raw local slot.
            rank_base: Nonnegative start of this rank's physical lane.

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
        if local_slot_size <= 0 or local_slot_size % 4096 or rank_base < 0:
            raise ValueError("invalid packed layout geometry")
        result = [dict(span) for span in spans]
        records: list[tuple[int, int, ChunkMeta, _Placement | None]] = []
        seen: set[int] = set()
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
            records.append((index, slot, owner, prior))

        pending: dict[tuple[int, int], _Placement] = {}
        run: list[tuple[int, int, ChunkMeta]] = []

        def finish_run() -> None:
            if not run:
                return
            # Each record fits one raw slot, so this backward placement never
            # precedes its own slot. FIFO invalidates it before any later slot
            # holding its bytes can be reused. Do not change this to head-align.
            cursor = rank_base + (run[-1][1] + 1) * local_slot_size
            for index, slot, owner in reversed(run):
                span = result[index]
                nbytes = int(span["nbytes"])
                cursor -= nbytes
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
                if (
                    index != prev_index + 1
                    or slot != prev_slot + 1
                    or slot == manager.tail_slot
                    or int(result[index]["source_offset"])
                    != int(prev_span["source_offset"]) + int(prev_span["nbytes"])
                ):
                    finish_run()
            run.append((index, slot, owner))
        finish_run()
        self._placements.update(pending)
        return result
