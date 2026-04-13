# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import tempfile

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.server.metadata_store import ChunkMeta, MetadataStore

logger = init_logger(__name__)


class ChunkManager:
    """Ring buffer manager for DaseR's slot-based KV store.

    Allocates contiguous slots from a fixed-size ring buffer. When the
    buffer is full, the oldest chunk is evicted automatically. Handles
    wrap-around at the end of the buffer by inserting SKIP blocks.

    Args:
        total_slots: total number of slots in the ring buffer.
        metadata_store: MetadataStore instance this manager operates on.
    """

    def __init__(self, total_slots: int, metadata_store: MetadataStore) -> None:
        self._total_slots = total_slots
        self._store = metadata_store
        self._head: int = 0  # next slot to write
        self._tail: int = 0  # oldest chunk's first slot

    @property
    def store(self) -> MetadataStore:
        """The underlying MetadataStore."""
        return self._store

    @property
    def head(self) -> int:
        """Index of the next slot to be written."""
        return self._head

    @property
    def tail(self) -> int:
        """Index of the oldest chunk's first slot."""
        return self._tail

    @property
    def free_slots(self) -> int:
        """Number of slots currently available without eviction.

        When head == tail the buffer is either completely empty or completely
        full.  We disambiguate by checking whether any chunks are stored.
        """
        if self._head == self._tail:
            return 0 if len(self._store) > 0 else self._total_slots
        if self._head > self._tail:
            return self._total_slots - (self._head - self._tail)
        return self._tail - self._head

    def alloc(
        self,
        chunk_key: str,
        num_slots: int,
        token_count: int,
        model_id: str,
        pos_offset: int,
    ) -> int:
        """Allocate contiguous slots for a new chunk.

        Evicts oldest chunks as needed. Inserts a SKIP block and wraps
        head to 0 if there is not enough contiguous space at the end of
        the buffer.

        Args:
            chunk_key: unique identifier for the chunk.
            num_slots: number of contiguous slots required.
            token_count: number of tokens in this chunk.
            model_id: model identifier for cache invalidation.
            pos_offset: position encoding offset for this chunk.

        Returns:
            start_slot: index of the first allocated slot.

        Raises:
            ValueError: if num_slots > total_slots.
        """
        if num_slots > self._total_slots:
            raise ValueError(
                f"num_slots={num_slots} exceeds total_slots={self._total_slots}"
            )

        # Wrap if not enough contiguous space at end of buffer
        tail_space = self._total_slots - self._head
        if 0 < tail_space < num_slots:
            # Evict any chunk whose first slot falls in the about-to-be-skipped region
            self._evict_range(self._head, tail_space)
            self._store.insert_skip(self._head, tail_space)
            logger.debug("[CHUNK] wrap: SKIP %d slots at %d", tail_space, self._head)
            self._head = 0

        # Evict until we have enough free space
        while self.free_slots < num_slots:
            self.evict_oldest()

        start_slot = self._head
        meta = ChunkMeta(
            chunk_key=chunk_key,
            start_slot=start_slot,
            num_slots=num_slots,
            token_count=token_count,
            pos_offset=pos_offset,
            model_id=model_id,
        )
        self._store.insert(meta)
        self._head += num_slots
        if self._head == self._total_slots:
            self._head = 0
        logger.debug(
            "[CHUNK] alloc chunk_key=%s start=%d num=%d head=%d tail=%d",
            chunk_key,
            start_slot,
            num_slots,
            self._head,
            self._tail,
        )
        return start_slot

    def evict_oldest(self) -> None:
        """Evict the oldest chunk (or skip block) at tail and advance tail.

        Raises:
            RuntimeError: if the buffer is empty.
        """
        if self._tail == self._head and len(self._store) == 0:
            raise RuntimeError("evict_oldest called on empty ring buffer")

        entry = self._store.get_slot_entry(self._tail)
        if entry.kind == "chunk":
            if entry.chunk_key is None:
                raise RuntimeError(
                    f"slot_map invariant violated: kind='chunk' at tail={self._tail} "
                    "but chunk_key is None"
                )
            self._store.remove(entry.chunk_key)
            self._tail = (self._tail + entry.num_slots) % self._total_slots
            logger.debug(
                "[CHUNK] evict chunk_key=%s tail=%d", entry.chunk_key, self._tail
            )
        elif entry.kind == "skip":
            self._tail = (self._tail + entry.num_slots) % self._total_slots
            logger.debug("[CHUNK] skip block consumed, tail=%d", self._tail)
        else:
            # cont slot at tail is a bug — advance by 1 to recover
            logger.warning("[CHUNK] unexpected cont slot at tail=%d", self._tail)
            self._tail = (self._tail + 1) % self._total_slots

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist full ring buffer state (index + head/tail) to path.

        Args:
            path: absolute file path to write.
        """
        tmp_store_path = path + ".tmp_store"
        self._store.save(tmp_store_path)
        with open(tmp_store_path, "rb") as f:
            index_bytes = f.read()
        os.unlink(tmp_store_path)

        payload = {
            "head": self._head,
            "tail": self._tail,
            "index": index_bytes,
        }
        with open(path, "wb") as f:
            f.write(msgpack.packb(payload, use_bin_type=True))
        logger.info(
            "[CHUNK] state saved to %s (head=%d tail=%d)", path, self._head, self._tail
        )

    def load(self, path: str) -> None:
        """Restore ring buffer state from path, replacing current state.

        Args:
            path: absolute file path to read.
        """
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        self._head = payload["head"]
        self._tail = payload["tail"]

        # Restore index via a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".index")
        tmp.write(payload["index"])
        tmp.close()
        try:
            self._store.load(tmp.name)
        finally:
            os.unlink(tmp.name)

        logger.info(
            "[CHUNK] state loaded from %s (head=%d tail=%d)",
            path,
            self._head,
            self._tail,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_range(self, start: int, count: int) -> None:
        """Evict any chunks whose first slot falls within [start, start+count).

        Used before inserting a SKIP block during wrap-around.

        Args:
            start: first slot of the range.
            count: number of slots in the range.
        """
        for slot_id in range(start, start + count):
            entry = self._store.get_slot_entry(slot_id)
            if entry.kind == "chunk" and entry.chunk_key is not None:
                # Guard against stale slot_map entries from a prior ring cycle:
                # remove() leaves kind="chunk" in slot_map; same slot may appear
                # live here in the next cycle even though it was already freed.
                if self._store.get(entry.chunk_key) is not None:
                    self._store.remove(entry.chunk_key)
