# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import math
from typing import Any, Optional

# First Party
from daser.logging import init_logger
from daser.position.base import PositionEncoder
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
from daser.server.chunk_manager import ChunkManager
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


@dataclass(frozen=True)
class ChunkInfo:
    """Public chunk metadata returned by cache lookup operations.

    Attributes:
        chunk_key: cache key for the token sequence.
        start_slot: first slot occupied in the ring buffer.
        num_slots: number of contiguous slots occupied.
        token_count: number of tokens covered by the chunk.
        pos_offset: position offset assigned by the position encoder.
        target_token_start: token offset in the current prompt where this
            chunk should be loaded.
        model_id: model identifier used for reuse isolation.
        file_offset: byte offset in the KV store file.
        residency: current chunk residency state.
        l2_durable: True when the chunk has a durable L2 copy.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    file_offset: int
    target_token_start: int = 0
    residency: str = "l2_only"
    l2_durable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a msgpack/JSON-safe representation.

        Returns:
            Plain dict containing all chunk fields.
        """
        return {
            "chunk_key": self.chunk_key,
            "start_slot": self.start_slot,
            "num_slots": self.num_slots,
            "token_count": self.token_count,
            "pos_offset": self.pos_offset,
            "model_id": self.model_id,
            "file_offset": self.file_offset,
            "target_token_start": self.target_token_start,
            "residency": self.residency,
            "l2_durable": self.l2_durable,
        }


@dataclass(frozen=True)
class Allocation:
    """Slot allocation returned to the connector.

    Attributes:
        chunk_key: cache key for the allocated chunk.
        start_slot: first allocated slot.
        num_slots: number of allocated slots.
        file_offset: byte offset in the KV store file.
        pos_offset: position offset assigned by the position encoder.
        token_count: number of tokens covered by this allocation.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    file_offset: int
    pos_offset: int
    token_count: int

    def to_dict(self, include_chunk_key: bool = True) -> dict[str, Any]:
        """Return a msgpack/JSON-safe representation.

        Args:
            include_chunk_key: include chunk_key in the output when True.

        Returns:
            Plain dict containing allocation fields.
        """
        payload: dict[str, Any] = {
            "start_slot": self.start_slot,
            "num_slots": self.num_slots,
            "file_offset": self.file_offset,
            "pos_offset": self.pos_offset,
        }
        if include_chunk_key:
            payload["chunk_key"] = self.chunk_key
            payload["token_count"] = self.token_count
        return payload


@dataclass(frozen=True)
class MatchAndAllocResult:
    """Result for combined lookup and conditional allocation.

    Attributes:
        chunks: matching chunks when lookup hits.
        alloc: allocation info when lookup misses and allocation occurs.
    """

    chunks: list[ChunkInfo]
    alloc: Optional[Allocation]

    def to_dict(self) -> dict[str, Any]:
        """Return a msgpack-safe response payload.

        Returns:
            Dict with chunks and optional alloc.
        """
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "alloc": self.alloc.to_dict() if self.alloc is not None else None,
        }


@dataclass(frozen=True)
class DocumentRegistration:
    """Result for registering a document.

    Attributes:
        chunk_count_cached: number of document chunks currently cached.
    """

    chunk_count_cached: int


@dataclass(frozen=True)
class DocumentSummary:
    """Summary returned by document listing.

    Attributes:
        doc_id: unique document identifier.
        title: user-facing title.
        token_count: number of original document tokens.
        chunk_count_total: number of chunk keys referenced by the document.
        chunk_count_cached: number of referenced chunks still cached.
        status: document cache status.
        created_at: unix timestamp when the document was registered.
    """

    doc_id: str
    title: str
    token_count: int
    chunk_count_total: int
    chunk_count_cached: int
    status: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation.

        Returns:
            Plain dict containing summary fields.
        """
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "token_count": self.token_count,
            "chunk_count_total": self.chunk_count_total,
            "chunk_count_cached": self.chunk_count_cached,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DeleteDocumentResult:
    """Result for deleting a document.

    Attributes:
        chunks_evicted: number of chunks removed because no other doc used them.
    """

    chunks_evicted: int


class ServerCore:
    """Shared control-plane core for DaseR server APIs.

    The core owns all mutable server-side cache and document state. The
    HTTP server calls document-level methods directly, while the IPC server
    calls connector cache methods.

    Args:
        chunk_manager: ring-buffer allocator and persistence coordinator.
        retrieval_index: cache lookup index.
        position_encoder: position-offset policy.
        slot_size: bytes per KV slot.
        block_tokens: tokens per vLLM block.

    Async/thread-safety:
        Intended to run on the DaseR server asyncio event loop. Methods do
        not perform blocking I/O. Callers should serialize access through
        the single event loop.
    """

    def __init__(
        self,
        chunk_manager: ChunkManager,
        retrieval_index: RetrievalIndex,
        position_encoder: PositionEncoder,
        slot_size: int,
        block_tokens: int = 16,
    ) -> None:
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._slot_size = slot_size
        self._block_tokens = block_tokens

    @property
    def chunk_manager(self) -> ChunkManager:
        """Return the owned ChunkManager."""
        return self._cm

    async def rebuild_retrieval_index(self) -> None:
        """Reinsert restored chunks into the retrieval index.

        Async/thread-safety:
            Run during startup before serving requests.
        """
        for meta in list(self._cm.store.iter_chunks()):
            if meta.l2_durable:
                await self._ri.insert(meta)

    async def lookup(
        self, tokens: list[int], model_id: str, pin: bool = False
    ) -> list[ChunkInfo]:
        """Look up cached chunks for token IDs.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            pin: when True, protect returned chunks until release_chunks.

        Returns:
            List of matching chunks, possibly empty.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        matches = await self._ri.lookup(tokens, model_id)
        chunks = [self._chunk_info(match) for match in matches]
        if pin:
            for chunk in chunks:
                meta = self._cm.store.get(chunk.chunk_key)
                if meta is not None:
                    meta.pin_count += 1
                    meta.lease_expires_at = 0.0
        return chunks

    async def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str
    ) -> Allocation:
        """Allocate slots for a new chunk or reuse a compatible allocation.

        Args:
            chunk_key: cache key for the token sequence.
            token_count: number of tokens in this chunk.
            model_id: model identifier.

        Returns:
            Allocation metadata for connector writes.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        num_slots = math.ceil(token_count / self._block_tokens)
        meta = await self._alloc_or_get_chunk(
            chunk_key=chunk_key,
            token_count=token_count,
            num_slots=num_slots,
            model_id=model_id,
        )
        return self._allocation(meta, token_count=token_count, num_slots=num_slots)

    async def match_and_alloc(
        self, tokens: list[int], chunk_key: str, model_id: str
    ) -> MatchAndAllocResult:
        """Run lookup and allocate a future store target on miss.

        Args:
            tokens: full prompt token IDs.
            chunk_key: key for the aligned prefix; empty disables allocation.
            model_id: model identifier.

        Returns:
            Lookup chunks on hit, or allocation on miss.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        chunks = await self.lookup(tokens, model_id, pin=True)
        if chunks:
            return MatchAndAllocResult(chunks=chunks, alloc=None)
        if not chunk_key:
            return MatchAndAllocResult(chunks=[], alloc=None)
        aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        if aligned == 0:
            return MatchAndAllocResult(chunks=[], alloc=None)
        num_slots = math.ceil(aligned / self._block_tokens)
        meta = await self._alloc_or_get_chunk(
            chunk_key=chunk_key,
            token_count=aligned,
            num_slots=num_slots,
            model_id=model_id,
        )
        return MatchAndAllocResult(
            chunks=[],
            alloc=self._allocation(meta, token_count=aligned, num_slots=num_slots),
        )

    async def commit_chunk(self, chunk_key: str) -> None:
        """Mark a chunk as committed and visible to lookup.

        Args:
            chunk_key: cache key for the chunk.

        Raises:
            ValueError: if the chunk was not allocated.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            raise ValueError(f"chunk_key not found: {chunk_key}")
        meta.residency = "l2_only"
        meta.l2_durable = True
        meta.backend = "gds"
        if meta.pin_count > 0:
            meta.pin_count -= 1
        await self._ri.insert(meta)
        logger.debug("[CORE] commit_chunk key=%s", chunk_key[:8])

    async def commit_l1(self, chunk_key: str) -> None:
        """Record that a chunk's bytes are available in worker L1 memory.

        Args:
            chunk_key: cache key for the chunk.

        Raises:
            ValueError: if the chunk was not allocated.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            raise ValueError(f"chunk_key not found: {chunk_key}")
        meta.residency = "l1_only"
        meta.l2_durable = False
        meta.backend = "iouring-mem"
        logger.debug("[CORE] commit_l1 key=%s", chunk_key[:8])

    async def commit_l2(self, chunk_key: str) -> None:
        """Mark a chunk's SSD L2 copy as durable.

        Args:
            chunk_key: cache key for the chunk.

        Raises:
            ValueError: if the chunk was not allocated.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            raise ValueError(f"chunk_key not found: {chunk_key}")
        meta.l2_durable = True
        release_count = 1
        if meta.residency == "l1_only":
            meta.residency = "l1_l2"
            await self._ri.insert(meta)
        elif meta.residency == "allocated":
            meta.residency = "l2_only"
            await self._ri.insert(meta)
        for _ in range(release_count):
            if meta.pin_count <= 0:
                break
            meta.pin_count -= 1
        logger.debug("[CORE] commit_l2 key=%s", chunk_key[:8])

    async def release_chunks(self, chunk_keys: list[str]) -> None:
        """Release lookup/load pins for chunks.

        Args:
            chunk_keys: chunk keys to release.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        for chunk_key in chunk_keys:
            meta = self._cm.store.get(chunk_key)
            if meta is not None and meta.pin_count > 0:
                meta.pin_count -= 1
        logger.debug("[CORE] release_chunks count=%d", len(chunk_keys))

    async def evict_l1(self, chunk_key: str) -> None:
        """Record worker-side L1 eviction for a durable, unpinned chunk.

        Args:
            chunk_key: chunk key whose L1 copy was evicted.

        Raises:
            ValueError: if eviction would drop the only readable copy or a
                protected in-flight load.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            return
        if not meta.l2_durable and meta.residency == "l1_only":
            meta.residency = "allocated"
            logger.debug("[CORE] evict_l1 before l2 commit key=%s", chunk_key[:8])
            return
        if meta.pin_count > 0:
            raise ValueError(f"chunk is pinned: {chunk_key}")
        if not meta.l2_durable:
            raise ValueError(f"chunk is not durable in L2: {chunk_key}")
        if meta.residency == "l1_l2":
            meta.residency = "l2_only"
        logger.debug("[CORE] evict_l1 key=%s", chunk_key[:8])

    async def evict_chunk(self, chunk_key: str) -> None:
        """Evict a chunk from retrieval and metadata state.

        Args:
            chunk_key: cache key for the chunk.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        await self._ri.remove(chunk_key)
        meta = self._cm.store.get(chunk_key)
        if meta is not None:
            self._mark_chunk_evicted_in_docs(meta)
            self._cm.store.remove(chunk_key)
        logger.debug("[CORE] evict_chunk key=%s", chunk_key[:8])

    async def register_document(
        self,
        doc_id: str,
        title: str,
        chunk_keys: list[str],
        token_count: int,
        tokens: Optional[list[int]] = None,
    ) -> DocumentRegistration:
        """Register a document and attach it to cached chunks.

        Args:
            doc_id: unique document identifier.
            title: user-facing title.
            chunk_keys: chunk keys in document order.
            token_count: number of original document tokens.
            tokens: optional full token sequence for prompt reconstruction.

        Returns:
            Document registration result.

        Raises:
            RuntimeError: if no DocRegistry is attached.
            ValueError: if doc_id already exists.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        registry = self._require_doc_registry()
        if registry.get(doc_id) is not None:
            raise ValueError(f"doc_id already exists: {doc_id}")

        cached_mask: list[bool] = []
        for key in chunk_keys:
            meta = self._cm.store.get(key)
            if meta is None:
                cached_mask.append(False)
                continue
            if doc_id not in meta.doc_ids:
                meta.doc_ids.append(doc_id)
            cached_mask.append(True)

        entry = DocEntry(
            doc_id=doc_id,
            title=title,
            token_count=token_count,
            chunk_keys=list(chunk_keys),
            cached_mask=cached_mask,
            status="ready" if any(cached_mask) else "evicted",
            tokens=list(tokens) if tokens is not None else None,
        )
        registry.insert(entry)
        cached = sum(cached_mask)
        logger.info(
            "[CORE] register_document doc_id=%s chunks=%d cached=%d",
            doc_id,
            len(chunk_keys),
            cached,
        )
        return DocumentRegistration(chunk_count_cached=cached)

    async def list_documents(self) -> list[DocumentSummary]:
        """Return summaries for all registered documents.

        Returns:
            Document summaries in registry order.

        Async/thread-safety:
            Reads in-memory state on the server event loop.
        """
        registry = self._cm.doc_registry
        if registry is None:
            return []
        summaries: list[DocumentSummary] = []
        for entry in registry.all_entries():
            summaries.append(
                DocumentSummary(
                    doc_id=entry.doc_id,
                    title=entry.title,
                    token_count=entry.token_count,
                    chunk_count_total=len(entry.chunk_keys),
                    chunk_count_cached=sum(entry.cached_mask),
                    status=entry.status,
                    created_at=entry.created_at,
                )
            )
        return summaries

    async def get_document(self, doc_id: str) -> Optional[DocEntry]:
        """Return a registered document.

        Args:
            doc_id: document identifier.

        Returns:
            DocEntry when found, otherwise None.

        Async/thread-safety:
            Reads in-memory state on the server event loop.
        """
        registry = self._cm.doc_registry
        if registry is None:
            return None
        return registry.get(doc_id)

    async def delete_document(self, doc_id: str) -> DeleteDocumentResult:
        """Delete a document and evict chunks it solely references.

        Args:
            doc_id: document identifier.

        Returns:
            Deletion result.

        Raises:
            RuntimeError: if no DocRegistry is attached.
            ValueError: if doc_id is unknown.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        registry = self._require_doc_registry()
        entry = registry.remove(doc_id)
        if entry is None:
            raise ValueError(f"doc_id not found: {doc_id}")

        chunks_evicted = 0
        for key in entry.chunk_keys:
            if self._detach_doc_from_chunk(doc_id, key):
                await self._ri.remove(key)
                chunks_evicted += 1
        logger.info(
            "[CORE] delete_document doc_id=%s chunks_evicted=%d",
            doc_id,
            chunks_evicted,
        )
        return DeleteDocumentResult(chunks_evicted=chunks_evicted)

    async def _drain_ring_evictions(self) -> None:
        """Remove ring-evicted chunks from retrieval index.

        Async/thread-safety:
            Runs on the server event loop after allocation.
        """
        for chunk_key in self._cm.drain_evicted_chunk_keys():
            await self._ri.remove(chunk_key)
            logger.debug("[CORE] removed auto-evicted chunk key=%s", chunk_key[:8])

    def _require_doc_registry(self) -> DocRegistry:
        """Return the attached DocRegistry or raise a public operation error."""
        registry = self._cm.doc_registry
        if registry is None:
            raise RuntimeError("doc registry not enabled")
        return registry

    def _mark_chunk_evicted_in_docs(self, meta: ChunkMeta) -> None:
        """Mark a chunk as evicted in every referencing document.

        Args:
            meta: chunk metadata being evicted.
        """
        registry = self._cm.doc_registry
        if registry is None:
            return
        for doc_id in list(meta.doc_ids):
            registry.mark_chunk_evicted(doc_id, meta.chunk_key)

    def _detach_doc_from_chunk(self, doc_id: str, chunk_key: str) -> bool:
        """Detach a document reference and remove unreferenced chunks.

        Args:
            doc_id: document being deleted.
            chunk_key: referenced chunk key.

        Returns:
            True when the chunk became unreferenced and was removed.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            return False
        if doc_id in meta.doc_ids:
            meta.doc_ids.remove(doc_id)
        if meta.doc_ids:
            return False
        self._cm.store.remove(chunk_key)
        return True

    async def _alloc_or_get_chunk(
        self,
        chunk_key: str,
        token_count: int,
        num_slots: int,
        model_id: str,
    ) -> ChunkMeta:
        """Allocate a chunk or reuse a compatible in-flight allocation.

        Args:
            chunk_key: cache key for the token sequence.
            token_count: number of tokens covered by the chunk.
            num_slots: number of slots required.
            model_id: model identifier.

        Returns:
            Existing or newly allocated chunk metadata.

        Raises:
            ValueError: if an existing chunk has incompatible metadata.
            RuntimeError: if allocation does not populate metadata.
        """
        existing = self._cm.store.get(chunk_key)
        if existing is not None:
            if (
                existing.token_count != token_count
                or existing.num_slots != num_slots
                or existing.model_id != model_id
            ):
                raise ValueError(
                    f"chunk_key already exists with incompatible metadata: {chunk_key}"
                )
            return existing

        pos_offset = self._pe.assign_offset(chunk_key, token_count)
        start_slot = self._cm.alloc(
            chunk_key=chunk_key,
            num_slots=num_slots,
            token_count=token_count,
            model_id=model_id,
            pos_offset=pos_offset,
        )
        await self._drain_ring_evictions()
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            raise RuntimeError(
                f"allocation succeeded at slot {start_slot} but metadata is missing"
            )
        meta.pin_count += 1
        return meta

    def _chunk_info(self, match: RetrievalMatch) -> ChunkInfo:
        """Convert a RetrievalMatch to ChunkInfo."""
        meta = match.meta
        pos_offset = self._pe.get_offset(meta, match.target_token_start)
        return ChunkInfo(
            chunk_key=meta.chunk_key,
            start_slot=meta.start_slot,
            num_slots=meta.num_slots,
            token_count=meta.token_count,
            pos_offset=pos_offset,
            model_id=meta.model_id,
            file_offset=meta.start_slot * self._slot_size,
            target_token_start=match.target_token_start,
            residency=meta.residency,
            l2_durable=meta.l2_durable,
        )

    def _allocation(
        self, meta: ChunkMeta, token_count: int, num_slots: int
    ) -> Allocation:
        """Convert ChunkMeta to Allocation."""
        return Allocation(
            chunk_key=meta.chunk_key,
            start_slot=meta.start_slot,
            num_slots=num_slots,
            file_offset=meta.start_slot * self._slot_size,
            pos_offset=meta.pos_offset,
            token_count=token_count,
        )
