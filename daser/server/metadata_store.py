# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict, dataclass, field
import time
from typing import Literal, Optional

# Third Party
import msgpack

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)


@dataclass
class ChunkMeta:
    """Metadata for one cached KV chunk.

    Attributes:
        chunk_key: xxh3_128(token_ids) or document ID identifying this chunk.
        start_slot: index of the first slot in daser.store.
        num_slots: number of contiguous slots occupied.
        token_count: number of tokens whose KV is stored.
        pos_offset: position encoding offset applied at load time.
        model_id: model identifier, prevents cross-model reuse.
        created_at: unix timestamp of insertion.
        doc_ids: list of doc_ids that reference this chunk (empty when
            the chunk belongs to no registered document). Serves as the
            back-pointer used by cascading eviction.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    created_at: float = 0.0
    doc_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class SlotEntry:
    """Describes one logical unit in the ring buffer's slot_map.

    kind="chunk" marks the first slot of a real KV chunk.
    kind="skip"  marks the first slot of a SKIP block (wrap padding).
    kind="cont"  marks continuation slots (positions 1..n-1 of a chunk/skip).

    For kind="chunk" and kind="skip", num_slots is the total size of the unit.
    For kind="cont", num_slots is unused (set to 0).

    Attributes:
        kind: one of "chunk", "skip", "cont".
        chunk_key: set for kind="chunk"; None otherwise.
        num_slots: total slots in this unit (only meaningful for first slot).
    """

    kind: Literal["chunk", "skip", "cont"]
    chunk_key: Optional[str] = None
    num_slots: int = 0


class MetadataStore:
    """In-memory index for DaseR's ring buffer.

    Maintains two structures:
    - chunk_index: maps chunk_key → ChunkMeta for O(1) lookup and removal.
    - slot_map: list[SlotEntry] indexed by slot_id, for ring buffer traversal.

    Can be serialized to / deserialized from a msgpack file.

    Args:
        total_slots: total number of slots in the ring buffer (must match
                     ChunkManager's total_slots).
    """

    def __init__(self, total_slots: int) -> None:
        self._total_slots = total_slots
        self._chunk_index: dict[str, ChunkMeta] = {}
        self._slot_map: list[SlotEntry] = [
            SlotEntry(kind="cont") for _ in range(total_slots)
        ]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, meta: ChunkMeta) -> None:
        """Insert a chunk into the index and mark its slots in slot_map.

        The first slot gets kind="chunk", remaining slots get kind="cont".

        Args:
            meta: ChunkMeta to insert.

        Raises:
            ValueError: if chunk_key already exists.
        """
        if meta.chunk_key in self._chunk_index:
            raise ValueError(f"chunk_key already exists: {meta.chunk_key}")
        self._chunk_index[meta.chunk_key] = meta
        self._slot_map[meta.start_slot] = SlotEntry(
            kind="chunk", chunk_key=meta.chunk_key, num_slots=meta.num_slots
        )
        for i in range(1, meta.num_slots):
            self._slot_map[meta.start_slot + i] = SlotEntry(kind="cont")
        logger.debug(
            "[INDEX] insert chunk_key=%s start_slot=%d num_slots=%d",
            meta.chunk_key,
            meta.start_slot,
            meta.num_slots,
        )

    def insert_skip(self, start_slot: int, num_slots: int) -> None:
        """Mark a range of slots as SKIP (wrap-around padding).

        Args:
            start_slot: first slot of the SKIP block.
            num_slots: number of slots in the block.
        """
        self._slot_map[start_slot] = SlotEntry(
            kind="skip", chunk_key=None, num_slots=num_slots
        )
        for i in range(1, num_slots):
            self._slot_map[start_slot + i] = SlotEntry(kind="cont")
        logger.debug(
            "[INDEX] insert_skip start_slot=%d num_slots=%d",
            start_slot,
            num_slots,
        )

    def remove(self, chunk_key: str) -> None:
        """Remove a chunk from the index.

        Does not clear slot_map entries — ChunkManager advances tail past them.

        Args:
            chunk_key: key of the chunk to remove.

        Raises:
            KeyError: if chunk_key is not found.
        """
        if chunk_key not in self._chunk_index:
            raise KeyError(f"chunk_key not found: {chunk_key}")
        del self._chunk_index[chunk_key]
        logger.debug("[INDEX] remove chunk_key=%s", chunk_key)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, chunk_key: str) -> Optional[ChunkMeta]:
        """Look up a chunk by key.

        Args:
            chunk_key: the key to look up.

        Returns:
            ChunkMeta if found, None otherwise.
        """
        return self._chunk_index.get(chunk_key)

    def get_slot_entry(self, slot_id: int) -> SlotEntry:
        """Return the SlotEntry at the given slot index.

        Args:
            slot_id: slot index in [0, total_slots).

        Returns:
            SlotEntry for that slot.
        """
        return self._slot_map[slot_id]

    def __len__(self) -> int:
        """Return number of chunks currently stored."""
        return len(self._chunk_index)

    def iter_chunks(self):
        """Yield every stored ChunkMeta.

        Returns:
            Iterator over ChunkMeta values.
        """
        return iter(self._chunk_index.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to a msgpack file.

        Writes ring buffer state (head, tail) are not stored here — they
        are passed in by ChunkManager. Call ChunkManager.save() instead
        to persist the full state.

        Args:
            path: absolute path to write the msgpack file.
        """
        payload = {
            "total_slots": self._total_slots,
            "chunk_index": {k: asdict(v) for k, v in self._chunk_index.items()},
            "slot_map": [
                {"kind": e.kind, "chunk_key": e.chunk_key, "num_slots": e.num_slots}
                for e in self._slot_map
            ],
        }
        with open(path, "wb") as f:
            f.write(msgpack.packb(payload, use_bin_type=True))
        logger.info("[INDEX] saved %d chunks to %s", len(self._chunk_index), path)

    def load(self, path: str) -> None:
        """Deserialize the index from a msgpack file, replacing current state.

        Args:
            path: absolute path to the msgpack file.
        """
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        self._total_slots = payload["total_slots"]
        self._chunk_index = {
            k: ChunkMeta(**v) for k, v in payload["chunk_index"].items()
        }
        self._slot_map = [
            SlotEntry(
                kind=e["kind"],
                chunk_key=e["chunk_key"],
                num_slots=e["num_slots"],
            )
            for e in payload["slot_map"]
        ]
        logger.info("[INDEX] loaded %d chunks from %s", len(self._chunk_index), path)
