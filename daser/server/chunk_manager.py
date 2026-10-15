# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import tempfile
from typing import TYPE_CHECKING

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.server.metadata_store import ChunkMeta, MetadataStore

if TYPE_CHECKING:
    from daser.server.doc_registry import DocRegistry

logger = init_logger(__name__)

_SNAPSHOT_VERSION = 1


class ChunkManager:
    """Ring buffer manager for DaseR's slot-based KV store.

    Allocates contiguous slots from a fixed-size ring buffer. When the
    buffer is full, the oldest chunk is evicted automatically. Handles
    wrap-around at the end of the buffer by inserting SKIP blocks.

    Args:
        total_slots: total number of slots in the ring buffer.
        metadata_store: MetadataStore instance this manager operates on.
        doc_registry: optional DocRegistry; when provided, evictions
            cascade into the registry so DocEntry.cached_mask stays in
            sync with the ring buffer. Omitting it keeps chunk-only
            behavior for tests and the legacy server entry point.
    """

    def __init__(
        self,
        total_slots: int,
        metadata_store: MetadataStore,
        doc_registry: "DocRegistry | None" = None,
        physical_slots: int | None = None,
    ) -> None:
        self._total_slots = total_slots
        self._physical_slots = physical_slots or total_slots
        if self._physical_slots <= 0 or self._physical_slots > total_slots:
            raise ValueError("physical_slots must be within logical slot capacity")
        self._store = metadata_store
        self._doc_registry = doc_registry
        self._head: int = 0  # next slot to write
        self._tail: int = 0  # oldest chunk's first slot
        self._evicted_chunk_keys: list[str] = []

    @property
    def total_slots(self) -> int:
        """Return total ring-buffer slot capacity."""
        return self._total_slots

    @property
    def physical_slots(self) -> int:
        """Return raw-envelope slots represented by the physical L2 lane."""
        return self._physical_slots

    @property
    def tail_slot(self) -> int:
        """Return the next FIFO reclamation slot for server layout planning.

        Returns:
            Logical slot index at the ring's oldest allocation or wrap padding.

        Async/thread-safety:
            Read on the owning server event loop; does not advance the ring.
        """
        return self._tail

    @property
    def doc_registry(self) -> "DocRegistry | None":
        """Return the attached DocRegistry if any."""
        return self._doc_registry

    @property
    def store(self) -> MetadataStore:
        """The underlying MetadataStore."""
        return self._store

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
            self._notify_eviction(entry.chunk_key)
            self._store.remove(entry.chunk_key)
            self._evicted_chunk_keys.append(entry.chunk_key)
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

    def drain_evicted_chunk_keys(self) -> list[str]:
        """Return and clear chunk keys evicted by automatic ring operations.

        The ring allocator does not own any retrieval index. Server-level
        coordinators consume this list after allocation and remove the same
        keys from their retrieval index, keeping metadata and lookup state in
        sync without coupling storage allocation to a concrete index.

        Returns:
            List of chunk keys evicted since the previous drain.
        """
        evicted = list(self._evicted_chunk_keys)
        self._evicted_chunk_keys.clear()
        return evicted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        cache_reuse_mode: str | None = None,
        *,
        committed_keys: set[str],
    ) -> None:
        """Persist ring state and an explicit publication manifest to path.

        Args:
            path (str): absolute file path to write.
            cache_reuse_mode (str | None): cache reuse mode that created the
                persisted metadata.
            committed_keys (set[str]): keys published by ServerCore. Allocated
                keys outside this set are discarded when restoring the snapshot.

        Returns:
            None.

        Async/thread-safety:
            Synchronous shutdown-only I/O on the owning server event loop.
            The caller must capture publication state on that same loop without
            yielding before this call. Does not mutate live allocation state.
        """
        tmp_store_path = path + ".tmp_store"
        tmp_path = path + ".tmp"
        try:
            self._store.save(tmp_store_path)
            with open(tmp_store_path, "rb") as f:
                index_bytes = f.read()

            payload = {
                "snapshot_version": _SNAPSHOT_VERSION,
                "committed_keys": sorted(committed_keys),
                "head": self._head,
                "tail": self._tail,
                "index": index_bytes,
                "cache_reuse_mode": cache_reuse_mode,
                "doc_registry": (
                    self._doc_registry.to_dict()
                    if self._doc_registry is not None
                    else {}
                ),
            }
            with open(tmp_path, "wb") as f:
                f.write(msgpack.packb(payload, use_bin_type=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            for cleanup_path in (tmp_store_path, tmp_path):
                if os.path.exists(cleanup_path):
                    os.unlink(cleanup_path)
        logger.info(
            "[CHUNK] state saved to %s (head=%d tail=%d docs=%d)",
            path,
            self._head,
            self._tail,
            len(self._doc_registry) if self._doc_registry is not None else 0,
        )

    def load(
        self,
        path: str,
        expected_cache_reuse_mode: str | None = None,
    ) -> None:
        """Restore ring state, retaining only explicitly committed allocations.

        Args:
            path (str): absolute file path to read.
            expected_cache_reuse_mode (str | None): startup cache reuse mode.
                When set, metadata from a different mode is rejected.

        Returns:
            None.

        Raises:
            ValueError: if the snapshot lacks a supported publication manifest
                or does not match the requested cache reuse mode.

        Async/thread-safety:
            Synchronous startup-only I/O; call before serving requests or
            rebuilding the retrieval index. Requires exclusive access.
        """
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        # Older snapshots cannot distinguish allocated space from published KV.
        # Reject them before changing any state; guessing would expose unwritten
        # or partial payloads as cache hits. The startup caller cold-starts.
        version = payload.get("snapshot_version")
        if type(version) is not int or version != _SNAPSHOT_VERSION:
            raise ValueError("unsupported snapshot version; cold start is required")
        committed_keys = payload.get("committed_keys")
        if not isinstance(committed_keys, list) or not all(
            isinstance(key, str) for key in committed_keys
        ):
            raise ValueError("invalid committed keys; cold start is required")
        committed = set(committed_keys)

        saved_mode = payload.get("cache_reuse_mode")
        if expected_cache_reuse_mode is not None and saved_mode is None:
            raise ValueError(
                "cache reuse mode missing from index; cold start is required"
            )
        if (
            expected_cache_reuse_mode is not None
            and saved_mode != expected_cache_reuse_mode
        ):
            raise ValueError(
                "cache reuse mode mismatch: "
                f"index was saved with {saved_mode!r}, "
                f"server started with {expected_cache_reuse_mode!r}"
            )

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

        if self._doc_registry is not None:
            self._doc_registry.load_dict(payload.get("doc_registry", {}) or {})

        # Keep physical offsets and FIFO geometry intact. Removing an allocation
        # leaves a SKIP span, so a pending hole is reclaimed when the ring tail
        # reaches it without shifting any committed payload. Restore documents
        # first so their cached masks are cleared through the normal eviction API.
        for meta in list(self._store.iter_chunks()):
            if meta.chunk_key not in committed:
                self._notify_eviction(meta.chunk_key)
                self._store.remove(meta.chunk_key)
        if len(self._store) == 0:
            self._head = self._tail = 0

        logger.info(
            "[CHUNK] state loaded from %s (head=%d tail=%d docs=%d)",
            path,
            self._head,
            self._tail,
            len(self._doc_registry) if self._doc_registry is not None else 0,
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
                # Slots behind the tail keep their last entry. Only evict the
                # chunk if this slot is still its live head.
                meta = self._store.get(entry.chunk_key)
                if meta is not None and meta.start_slot == slot_id:
                    self._notify_eviction(entry.chunk_key)
                    self._store.remove(entry.chunk_key)
                    self._evicted_chunk_keys.append(entry.chunk_key)

    def _notify_eviction(self, chunk_key: str) -> None:
        """Forward a chunk eviction to the attached DocRegistry.

        Reads the ChunkMeta so the doc_ids referencing this chunk are
        known, then flips their cached_mask bits. Silently no-ops when
        no DocRegistry is attached.

        Args:
            chunk_key: chunk that is about to be removed from the store.
        """
        if self._doc_registry is None:
            return
        meta = self._store.get(chunk_key)
        if meta is None:
            return
        for doc_id in list(meta.doc_ids):
            self._doc_registry.mark_chunk_evicted(doc_id, chunk_key)
