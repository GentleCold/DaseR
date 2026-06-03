# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
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
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    token_count: int
    pos_offset: int
    model_id: str
    file_offset: int
    target_token_start: int = 0

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
        self._evicted_chunk_keys: set[str] = set()
        self._commit_requests = 0
        self._late_evicted_commits = 0
        self._lookup_requests = 0
        self._lookup_hits = 0
        self._committed_chunk_keys: set[str] = set()
        self._commit_waiters: dict[str, set[asyncio.Future[None]]] = {}

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
            await self._ri.insert(meta)
            self._committed_chunk_keys.add(meta.chunk_key)

    async def lookup(self, tokens: list[int], model_id: str) -> list[ChunkInfo]:
        """Look up cached chunks for token IDs.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.

        Returns:
            List of matching chunks, possibly empty.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        matches = await self._ri.lookup(tokens, model_id)
        self._lookup_requests += 1
        if matches:
            self._lookup_hits += 1
        return [self._chunk_info(match) for match in matches]

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
        chunks = await self.lookup(tokens, model_id)
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
            if chunk_key in self._evicted_chunk_keys:
                self._commit_requests += 1
                self._late_evicted_commits += 1
                logger.debug(
                    "[CORE] ignore late commit for evicted key=%s",
                    chunk_key[:8],
                )
                return
            raise ValueError(f"chunk_key not found: {chunk_key}")
        await self._ri.insert(meta)
        self._committed_chunk_keys.add(chunk_key)
        self._notify_commit_waiters(chunk_key)
        self._commit_requests += 1
        logger.debug("[CORE] commit_chunk key=%s", chunk_key[:8])

    def is_chunk_committed(self, chunk_key: str) -> bool:
        """Return whether a chunk key has been committed.

        Args:
            chunk_key: cache key to check.

        Returns:
            True after ``commit_chunk`` has published the chunk and before it is
            evicted or removed.

        Async/thread-safety:
            Reads in-memory state on the server event loop. It performs no
            blocking I/O.
        """
        return chunk_key in self._committed_chunk_keys

    async def wait_for_committed_chunks(
        self,
        chunk_keys: list[str],
        timeout_s: float,
    ) -> None:
        """Wait until all chunk keys have been committed.

        Args:
            chunk_keys: cache keys to wait for.
            timeout_s: maximum wait time in seconds.

        Raises:
            TimeoutError: if any chunk key is still uncommitted at timeout.

        Async/thread-safety:
            Must run on the server event loop. Waiters are completed by
            ``commit_chunk`` on the same event loop.
        """
        pending = [
            key
            for key in dict.fromkeys(chunk_keys)
            if key not in self._committed_chunk_keys
        ]
        if not pending:
            return

        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[None]] = {}
        for key in pending:
            future: asyncio.Future[None] = loop.create_future()
            self._commit_waiters.setdefault(key, set()).add(future)
            futures[key] = future
        try:
            await asyncio.wait_for(
                asyncio.gather(*futures.values()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("timed out waiting for committed chunks") from exc
        finally:
            for key, future in futures.items():
                waiters = self._commit_waiters.get(key)
                if waiters is None:
                    continue
                waiters.discard(future)
                if not waiters:
                    self._commit_waiters.pop(key, None)

    async def commit_stats(self) -> dict[str, int]:
        """Return connector commit counters for benchmark synchronization.

        Returns:
            Dict containing total processed commit requests and the subset that
            arrived after the chunk had already been evicted.

        Async/thread-safety:
            Reads in-memory counters on the server event loop.
        """
        return {
            "commit_requests": self._commit_requests,
            "late_evicted_commits": self._late_evicted_commits,
            "lookup_requests": self._lookup_requests,
            "lookup_hits": self._lookup_hits,
        }

    async def live_allocations(self, allocations: list[dict[str, Any]]) -> list[str]:
        """Return chunk keys that still own their allocated slot ranges.

        Args:
            allocations: Dicts with ``chunk_key``, ``start_slot``, and
                ``num_slots`` fields.

        Returns:
            Chunk keys whose current metadata still matches the supplied slot
            allocation.

        Async/thread-safety:
            Reads in-memory metadata on the server event loop.
        """
        live: list[str] = []
        for alloc in allocations:
            chunk_key = str(alloc.get("chunk_key", ""))
            if not chunk_key:
                continue
            if self.is_current_allocation(
                chunk_key=chunk_key,
                start_slot=int(alloc.get("start_slot", -1)),
                num_slots=int(alloc.get("num_slots", 0)),
            ):
                live.append(chunk_key)
        return live

    def is_current_allocation(
        self,
        chunk_key: str,
        start_slot: int,
        num_slots: int,
    ) -> bool:
        """Return whether a delayed write still targets the live chunk.

        Args:
            chunk_key: chunk key associated with the write span.
            start_slot: first slot the connector was told to write.
            num_slots: number of slots allocated for the chunk.

        Returns:
            True when the chunk still exists and still owns the same slot range.

        Async/thread-safety:
            Reads in-memory metadata on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        return (
            meta is not None
            and meta.start_slot == start_slot
            and meta.num_slots == num_slots
        )

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
        self._committed_chunk_keys.discard(chunk_key)
        self._evicted_chunk_keys.add(chunk_key)
        logger.debug("[CORE] evict_chunk key=%s", chunk_key[:8])

    async def register_document(
        self,
        doc_id: str,
        title: str,
        chunk_keys: list[str],
        token_count: int,
        tokens: Optional[list[int]] = None,
        text: Optional[str] = None,
    ) -> DocumentRegistration:
        """Register a document and attach it to cached chunks.

        Args:
            doc_id: unique document identifier.
            title: user-facing title.
            chunk_keys: chunk keys in document order.
            token_count: number of original document tokens.
            tokens: optional full token sequence for prompt reconstruction.
            text: optional original document text for UI inspection.

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
            text=text,
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
            self._committed_chunk_keys.discard(chunk_key)
            self._evicted_chunk_keys.add(chunk_key)
            logger.debug("[CORE] removed auto-evicted chunk key=%s", chunk_key[:8])

    def _notify_commit_waiters(self, chunk_key: str) -> None:
        """Wake coroutines waiting for a chunk commit."""
        waiters = self._commit_waiters.pop(chunk_key, set())
        for future in waiters:
            if not future.done():
                future.set_result(None)

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
            self._committed_chunk_keys.discard(chunk_key)
            return False
        if doc_id in meta.doc_ids:
            meta.doc_ids.remove(doc_id)
        if meta.doc_ids:
            return False
        self._cm.store.remove(chunk_key)
        self._committed_chunk_keys.discard(chunk_key)
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
