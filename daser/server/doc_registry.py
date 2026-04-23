# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict, dataclass, field
import time
from typing import Optional


@dataclass
class DocEntry:
    """Registered document metadata.

    Attributes:
        doc_id: unique identifier assigned at register_doc time.
        title: user-supplied display title.
        created_at: unix timestamp of registration.
        token_count: number of tokens in the original document.
        chunk_keys: chunk_keys of this doc's KV chunks, in upload order;
            the ordering is never reshuffled so that prompt reconstruction
            is deterministic.
        cached_mask: list[bool], same length as chunk_keys; True when the
            chunk is still resident in the ring buffer, False once it has
            been evicted. Stored as list[bool] rather than a raw bitset
            for readable msgpack serialization.
        status: "ready", "partial", "evicted", or "failed".
        tokens: optional full token sequence, saved so inference can
            reconstruct the concatenated prompt without re-tokenizing.
        error: optional human-readable failure reason.
    """

    doc_id: str
    title: str
    created_at: float = 0.0
    token_count: int = 0
    chunk_keys: list[str] = field(default_factory=list)
    cached_mask: list[bool] = field(default_factory=list)
    status: str = "ready"
    tokens: Optional[list[int]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()


class DocRegistry:
    """In-memory doc_id → DocEntry index, persisted alongside daser.index.

    The registry is the single source of truth for document metadata. It
    is updated in two places:
    - register_doc / evict_doc: explicit admin ops.
    - ChunkManager.evict: flips cached_mask entries when a referenced
      chunk is evicted from the ring buffer.

    Thread-safety: callers run inside the asyncio event loop of
    IPCServer; no additional locking is provided.
    """

    def __init__(self) -> None:
        self._docs: dict[str, DocEntry] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, entry: DocEntry) -> None:
        """Register a new document entry.

        Args:
            entry: DocEntry to insert.

        Raises:
            ValueError: if doc_id already exists.
        """
        if entry.doc_id in self._docs:
            raise ValueError(f"doc_id already exists: {entry.doc_id}")
        if not entry.cached_mask:
            entry.cached_mask = [True] * len(entry.chunk_keys)
        self._docs[entry.doc_id] = entry

    def remove(self, doc_id: str) -> Optional[DocEntry]:
        """Remove a document entry.

        Args:
            doc_id: doc to remove.

        Returns:
            The removed DocEntry, or None if doc_id is unknown.
        """
        return self._docs.pop(doc_id, None)

    def mark_chunk_evicted(self, doc_id: str, chunk_key: str) -> None:
        """Flip the cached_mask bit for (doc_id, chunk_key) to False.

        Also flips status to "evicted" if every chunk for this doc is
        now gone. Silently no-ops if doc_id or chunk_key is unknown.

        Args:
            doc_id: document whose cached_mask should be updated.
            chunk_key: chunk that was evicted from the ring buffer.
        """
        entry = self._docs.get(doc_id)
        if entry is None:
            return
        for i, key in enumerate(entry.chunk_keys):
            if key == chunk_key and i < len(entry.cached_mask):
                entry.cached_mask[i] = False
        if entry.status == "ready" and not any(entry.cached_mask):
            entry.status = "evicted"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, doc_id: str) -> Optional[DocEntry]:
        """Return the DocEntry for doc_id, or None if unknown."""
        return self._docs.get(doc_id)

    def all_entries(self) -> list[DocEntry]:
        """Return every DocEntry (snapshot order is insertion order)."""
        return list(self._docs.values())

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # Serialization helpers (used by ChunkManager.save / load)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict]:
        """Return a plain-dict representation for msgpack serialization."""
        return {k: asdict(v) for k, v in self._docs.items()}

    def load_dict(self, payload: dict[str, dict]) -> None:
        """Replace current state with a previously serialized dict.

        Args:
            payload: dict[doc_id, DocEntry fields] produced by to_dict.
        """
        self._docs = {k: DocEntry(**v) for k, v in payload.items()}
