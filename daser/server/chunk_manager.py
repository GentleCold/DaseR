# SPDX-License-Identifier: Apache-2.0
"""Chunk ids, chunk metadata ownership and index persistence."""

# Standard
from dataclasses import asdict, dataclass
import os
from typing import Any

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.server.doc_registry import DocRegistry
from daser.server.metadata_store import ChunkMeta, MetadataStore
from daser.server.tiers import L2Placement

logger = init_logger(__name__)

#: Persisted index layout; files of another version require a cold start.
_INDEX_FORMAT = 2


@dataclass(frozen=True)
class IndexSnapshot:
    """Persisted server index (raw chunks with an L2 copy only).

    Attributes:
        cache_reuse_mode: Cache reuse mode that produced the chunks.
        next_slot: Next unassigned slot id.
        tier_grain: Tier allocation grain in bytes.
        l2_lane_bytes: Bytes of each TP rank's L2 lane.
        tensor_parallel_size: Number of L2 lanes.
        metas: Chunk metadata of the persisted chunks.
        l2: L2 placement of every persisted chunk.
        docs: Serialized document registry (``DocRegistry.to_dict``).
    """

    cache_reuse_mode: str | None
    next_slot: int
    tier_grain: int
    l2_lane_bytes: int
    tensor_parallel_size: int
    metas: list[ChunkMeta]
    l2: list[L2Placement]
    docs: dict[str, dict[str, Any]]


class ChunkManager:
    """Chunk id allocator and owner of chunk and document metadata.

    Slot ids are identities, not addresses: each allocation takes the next
    ``num_slots`` ids and ids are never reused, so a delayed write or load
    naming a removed allocation can never alias a later one. Byte capacity
    lives in ``daser.server.tiers``.

    Args:
        metadata_store: MetadataStore instance this manager operates on.
        doc_registry: optional DocRegistry persisted with the index.

    Async/thread-safety:
        Not thread-safe; owned by the server event loop.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        doc_registry: DocRegistry | None = None,
    ) -> None:
        self._store = metadata_store
        self._doc_registry = doc_registry
        self._next_slot = 0
        # slot id -> (chunk key, slot index inside the chunk), every live slot.
        self._slots: dict[int, tuple[str, int]] = {}
        for meta in metadata_store.iter_chunks():
            self._next_slot = max(self._next_slot, meta.start_slot + meta.num_slots)
            self._map_slots(meta)

    @property
    def doc_registry(self) -> DocRegistry | None:
        """Return the attached DocRegistry if any."""
        return self._doc_registry

    @property
    def store(self) -> MetadataStore:
        """The underlying MetadataStore."""
        return self._store

    @property
    def next_slot(self) -> int:
        """Return the id the next allocation starts at."""
        return self._next_slot

    def alloc(
        self,
        chunk_key: str,
        num_slots: int,
        token_count: int,
        model_id: str,
        pos_offset: int,
        parent_key: str = "",
    ) -> ChunkMeta:
        """Assign fresh slot ids to a new chunk and index its metadata.

        Args:
            chunk_key: unique identifier for the chunk.
            num_slots: number of slot ids (KV blocks) the chunk covers.
            token_count: number of tokens in this chunk.
            model_id: model identifier for cache invalidation.
            pos_offset: position encoding offset for this chunk.
            parent_key: previous chunk of the prefix chain, or empty.

        Returns:
            The inserted chunk metadata.

        Raises:
            ValueError: if ``num_slots < 1`` or the key is already indexed.
        """
        if num_slots < 1:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        meta = ChunkMeta(
            chunk_key=chunk_key,
            start_slot=self._next_slot,
            num_slots=num_slots,
            token_count=token_count,
            pos_offset=pos_offset,
            model_id=model_id,
            parent_key=parent_key,
        )
        self._store.insert(meta)
        self._next_slot += num_slots
        self._map_slots(meta)
        logger.debug(
            "[CHUNK] alloc chunk_key=%s start=%d num=%d",
            chunk_key,
            meta.start_slot,
            num_slots,
        )
        return meta

    def remove(self, chunk_key: str) -> ChunkMeta:
        """Remove a chunk from the index; its slot ids are never reused.

        Args:
            chunk_key: key of an indexed chunk.

        Returns:
            The removed metadata.

        Raises:
            KeyError: if the chunk is not indexed.
        """
        meta = self._store.remove(chunk_key)
        for slot in range(meta.start_slot, meta.start_slot + meta.num_slots):
            del self._slots[slot]
        return meta

    def resolve_slot(self, slot_id: int) -> tuple[ChunkMeta, int] | None:
        """Return the live chunk owning a slot id.

        Args:
            slot_id: Slot id from an allocation or lookup result.

        Returns:
            ``(meta, slot_index)`` with ``slot_index`` relative to
            ``meta.start_slot``, or None when no indexed chunk owns the id.
        """
        owner = self._slots.get(slot_id)
        if owner is None:
            return None
        meta = self._store.get(owner[0])
        assert meta is not None
        return meta, owner[1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, snapshot: IndexSnapshot) -> None:
        """Atomically write an index snapshot to ``path``.

        Args:
            path: absolute file path to write.
            snapshot: State to persist (built by ``ServerCore.save_index``).

        Async/thread-safety:
            Blocking file IO with fsync; call at shutdown only.
        """
        payload = {
            "format": _INDEX_FORMAT,
            "cache_reuse_mode": snapshot.cache_reuse_mode,
            "next_slot": snapshot.next_slot,
            "tier_grain": snapshot.tier_grain,
            "l2_lane_bytes": snapshot.l2_lane_bytes,
            "tensor_parallel_size": snapshot.tensor_parallel_size,
            "metas": [asdict(meta) for meta in snapshot.metas],
            "l2": [[p.chunk_key, p.offset, list(p.record_sizes)] for p in snapshot.l2],
            "docs": snapshot.docs,
        }
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(msgpack.packb(payload, use_bin_type=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        logger.info(
            "[CHUNK] state saved to %s (chunks=%d docs=%d)",
            path,
            len(snapshot.metas),
            len(snapshot.docs),
        )

    @staticmethod
    def read_snapshot(
        path: str, expected_cache_reuse_mode: str | None = None
    ) -> IndexSnapshot:
        """Read an index snapshot written by ``save``.

        Args:
            path: absolute file path to read.
            expected_cache_reuse_mode: When set, a snapshot saved by another
                (or an unknown) cache reuse mode is rejected.

        Returns:
            The decoded snapshot.

        Raises:
            ValueError: on an unknown format or a reuse-mode mismatch.

        Async/thread-safety:
            Blocking file IO; call at startup only.
        """
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)
        if not isinstance(payload, dict) or payload.get("format") != _INDEX_FORMAT:
            raise ValueError("unsupported index format; cold start is required")
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
        return IndexSnapshot(
            cache_reuse_mode=saved_mode,
            next_slot=int(payload["next_slot"]),
            tier_grain=int(payload["tier_grain"]),
            l2_lane_bytes=int(payload["l2_lane_bytes"]),
            tensor_parallel_size=int(payload["tensor_parallel_size"]),
            metas=[ChunkMeta(**fields) for fields in payload["metas"]],
            l2=[
                L2Placement(
                    chunk_key=str(key),
                    offset=int(offset),
                    record_sizes=tuple(int(size) for size in sizes),
                )
                for key, offset, sizes in payload["l2"]
            ],
            docs=dict(payload.get("docs") or {}),
        )

    def validate_snapshot(self, snapshot: IndexSnapshot) -> None:
        """Check that ``restore`` would accept a snapshot.

        Args:
            snapshot: Snapshot to check.

        Raises:
            ValueError: if this manager already indexes chunks, keys repeat,
                slot ranges are invalid, overlap, or reach ``next_slot``, or
                the attached document registry cannot decode the snapshot.
        """
        if len(self._store) or self._next_slot:
            raise ValueError("index restore requires an empty chunk manager")
        ranges = []
        keys = set()
        for meta in snapshot.metas:
            if meta.chunk_key in keys:
                raise ValueError(f"duplicate restored chunk: {meta.chunk_key}")
            keys.add(meta.chunk_key)
            end = meta.start_slot + meta.num_slots
            if meta.start_slot < 0 or meta.num_slots < 1 or end > snapshot.next_slot:
                raise ValueError(f"invalid slot range for chunk: {meta.chunk_key}")
            ranges.append((meta.start_slot, end))
        ranges.sort()
        for (_start, end), (next_start, _end) in zip(ranges, ranges[1:], strict=False):
            if end > next_start:
                raise ValueError("restored slot ranges overlap")
        if self._doc_registry is not None:
            # Decode before publishing any chunk or slot identity. A rejected
            # document must leave both the live manager and registry intact;
            # validating only during restore would leave a half-loaded index.
            try:
                DocRegistry().load_dict(snapshot.docs)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid snapshot document registry") from exc

    def restore(self, snapshot: IndexSnapshot) -> None:
        """Index the chunks and documents of a snapshot.

        Documents referencing chunks the snapshot does not contain are marked
        evicted for those chunks.

        Args:
            snapshot: Snapshot accepted by ``validate_snapshot``.

        Raises:
            ValueError: if ``validate_snapshot`` rejects it (nothing changes).
        """
        self.validate_snapshot(snapshot)
        for meta in snapshot.metas:
            self._store.insert(meta)
            self._map_slots(meta)
        self._next_slot = snapshot.next_slot
        if self._doc_registry is not None:
            self._doc_registry.load_dict(snapshot.docs)
            for entry in self._doc_registry.all_entries():
                for key in entry.chunk_keys:
                    if self._store.get(key) is None:
                        self._doc_registry.mark_chunk_evicted(entry.doc_id, key)
        logger.info(
            "[CHUNK] state restored (chunks=%d next_slot=%d)",
            len(snapshot.metas),
            snapshot.next_slot,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_slots(self, meta: ChunkMeta) -> None:
        """Record ``meta`` as the owner of its slot ids."""
        for index in range(meta.num_slots):
            self._slots[meta.start_slot + index] = (meta.chunk_key, index)
