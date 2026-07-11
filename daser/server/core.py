# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import math
from typing import Any, Optional

# First Party
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.position.base import PositionEncoder
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
from daser.server.chunk_lifecycle import ChunkLifecycle
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
        skipped: True when this allocation references an already committed
            identical chunk and the connector should not write KV again.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    file_offset: int
    pos_offset: int
    token_count: int
    skipped: bool = False

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
            "skipped": self.skipped,
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
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._slot_size = slot_size
        self._block_tokens = block_tokens
        self._metrics = metrics_registry or REGISTRY
        self._lifecycle = ChunkLifecycle()
        self._commit_requests = 0
        self._late_evicted_commits = 0
        self._lookup_requests = 0
        self._lookup_hits = 0
        self._record_capacity_metrics()

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
            self._lifecycle.mark_committed(meta.chunk_key)

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
        chunks = [self._chunk_info(match) for match in matches]
        self._metrics.counter(
            "daser_cache_lookup_total",
            "Cache lookup requests by result.",
        ).inc(labels={"result": "hit" if chunks else "miss"})
        self._metrics.counter(
            "daser_cache_requested_tokens_total",
            "Prompt tokens checked for cache reuse.",
        ).inc(len(tokens))
        self._metrics.counter(
            "daser_cache_matched_tokens_total",
            "Prompt tokens matched by cache lookup.",
        ).inc(sum(chunk.token_count for chunk in chunks))
        if chunks:
            self._metrics.histogram(
                "daser_cache_prefix_reuse_tokens",
                "Tokens reused per cache hit.",
                buckets=(16, 64, 128, 256, 512, 1024, 2048, 4096),
            ).observe(sum(chunk.token_count for chunk in chunks))
        self._record_capacity_metrics()
        return chunks

    async def record_external_prefix_cache(self, queries: int, hits: int) -> None:
        """Record vLLM-equivalent external prefix cache token counters.

        Args:
            queries: Number of prompt tokens vLLM queried through the KV
                connector external prefix cache path.
            hits: Number of queried tokens vLLM accepted as external prefix
                cache hits.

        Returns:
            None.

        Async/thread-safety:
            Performs no blocking I/O and should run on the server event loop.
        """
        queries = max(0, int(queries))
        hits = max(0, min(int(hits), queries))
        self._metrics.counter(
            "daser_external_prefix_cache_queries_total",
            "External prefix cache queries from DaseR KV connector, "
            "in terms of queried tokens.",
        ).inc(queries)
        self._metrics.counter(
            "daser_external_prefix_cache_hits_total",
            "External prefix cache hits from DaseR KV connector, "
            "in terms of cached tokens accepted by vLLM.",
        ).inc(hits)

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
        allocations = await self.alloc_chunks(
            [{"chunk_key": chunk_key, "token_count": token_count}],
            model_id,
        )
        return allocations[0]

    async def alloc_chunks(
        self,
        chunks: list[dict[str, Any]],
        model_id: str,
    ) -> list[Allocation]:
        """Allocate slots for multiple chunks in one server event-loop turn.

        Args:
            chunks: chunk descriptors with ``chunk_key`` and ``token_count``.
            model_id: model identifier.

        Returns:
            Allocation metadata in the same order as ``chunks``.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        allocations: list[Allocation] = []
        for chunk in chunks:
            chunk_key = str(chunk["chunk_key"])
            token_count = int(chunk["token_count"])
            num_slots = self._slots_for(token_count)
            should_write = not self._has_store_owner(chunk_key, token_count, model_id)
            meta = await self._alloc_or_get_chunk(
                chunk_key=chunk_key,
                token_count=token_count,
                num_slots=num_slots,
                model_id=model_id,
            )
            if should_write:
                self._lifecycle.mark_write_owner(chunk_key)
            allocations.append(
                self._allocation(
                    meta,
                    token_count=token_count,
                    num_slots=num_slots,
                    skipped=not should_write,
                )
            )
        return allocations

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
        num_slots = self._slots_for(aligned)
        should_write = not self._has_store_owner(chunk_key, aligned, model_id)
        meta = await self._alloc_or_get_chunk(
            chunk_key=chunk_key,
            token_count=aligned,
            num_slots=num_slots,
            model_id=model_id,
        )
        if should_write:
            self._lifecycle.mark_write_owner(chunk_key)
        return MatchAndAllocResult(
            chunks=[],
            alloc=self._allocation(
                meta,
                token_count=aligned,
                num_slots=num_slots,
                skipped=not should_write,
            ),
        )

    async def commit_chunk(
        self, chunk_key: str, tp_rank: int = 0, tp_size: int = 1
    ) -> None:
        """Mark a chunk as committed and visible to lookup.

        Args:
            chunk_key: cache key for the chunk.
            tp_rank: tensor-parallel rank whose shard finished storing.
            tp_size: total tensor-parallel ranks required before publication.

        Raises:
            ValueError: if the chunk was not allocated.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            if self._lifecycle.is_evicted(chunk_key):
                self._commit_requests += 1
                self._late_evicted_commits += 1
                self._metrics.counter(
                    "daser_cache_late_evicted_commits_total",
                    "Commit requests ignored because the allocation was evicted.",
                ).inc()
                logger.debug(
                    "[CORE] ignore late commit for evicted key=%s",
                    chunk_key[:8],
                )
                return
            raise ValueError(f"chunk_key not found: {chunk_key}")
        self._commit_requests += 1
        if not self._lifecycle.record_commit_shard(chunk_key, tp_rank, tp_size):
            return
        try:
            await self._ri.insert(meta)
        except BaseException:
            self._lifecycle.abort_publish(chunk_key)
            raise
        self._lifecycle.mark_committed(chunk_key)
        self._metrics.counter(
            "daser_cache_committed_chunks_total",
            "Chunks committed and published to the retrieval index.",
        ).inc()
        self._record_capacity_metrics()
        logger.debug(
            "[CORE] commit_chunk key=%s tp_rank=%d tp_size=%d",
            chunk_key[:8],
            tp_rank,
            tp_size,
        )

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
        return self._lifecycle.is_committed(chunk_key)

    def is_chunk_reusable(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
    ) -> bool:
        """Return whether an identical committed chunk can be reused.

        Args:
            chunk_key: cache key to check.
            token_count: number of tokens expected for this chunk.
            model_id: model identifier expected for reuse isolation.

        Returns:
            True when a committed metadata entry exists for the same key,
            token count, slot count, and model. False for misses, in-flight
            uncommitted allocations, and incompatible metadata.

        Async/thread-safety:
            Reads in-memory state on the server event loop. It performs no
            blocking I/O.
        """
        return self._meta_matches(
            chunk_key, token_count, model_id, self._lifecycle.committed
        )

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
        await self._lifecycle.wait_for_committed(chunk_keys, timeout_s)

    async def wait_for_pending_chunks(self, timeout_s: float) -> None:
        """Wait for currently allocated store writers to commit.

        Args:
            timeout_s: maximum seconds to wait.

        Raises:
            TimeoutError: if any pending chunk remains uncommitted at timeout.

        Async/thread-safety:
            Snapshots lifecycle state and waits without blocking the server
            event loop. Writers registered after the snapshot are not included.
        """
        pending = list(self._lifecycle.write_owners - self._lifecycle.committed)
        await self._lifecycle.wait_for_committed(pending, timeout_s)

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

    async def release_chunk_writer(
        self,
        chunk_key: str,
        start_slot: int,
        num_slots: int,
    ) -> bool:
        """Release an uncommitted writer claim for a canceled store.

        Args:
            chunk_key: chunk key associated with the canceled store.
            start_slot: first slot the connector was told to write.
            num_slots: number of slots allocated for the chunk.

        Returns:
            True when an uncommitted writer claim was released, False when the
            chunk was already committed or no longer matches the allocation.

        Async/thread-safety:
            Performs in-memory mutation on the server event loop.
        """
        if self._lifecycle.is_committed(chunk_key):
            return False
        if not self.is_current_allocation(chunk_key, start_slot, num_slots):
            return False
        self._lifecycle.discard_owner(chunk_key)
        return True

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
        self._lifecycle.mark_evicted(chunk_key)
        self._metrics.counter(
            "daser_cache_evicted_chunks_total",
            "Chunks evicted from cache metadata.",
        ).inc(labels={"reason": "explicit"})
        self._record_capacity_metrics()
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
            self._lifecycle.mark_evicted(chunk_key)
            self._metrics.counter(
                "daser_cache_evicted_chunks_total",
                "Chunks evicted from cache metadata.",
            ).inc(labels={"reason": "capacity"})
            logger.debug("[CORE] removed auto-evicted chunk key=%s", chunk_key[:8])
        self._record_capacity_metrics()

    def _record_capacity_metrics(self) -> None:
        """Publish current byte capacity gauges."""
        total_slots = self._cm.total_slots
        used_slots = total_slots - self._cm.free_slots
        self._metrics.gauge(
            "daser_store_l2_bytes_capacity",
            "Total L2 store capacity in bytes.",
        ).set(total_slots * self._slot_size)
        self._metrics.gauge(
            "daser_store_l2_bytes_used",
            "Currently used L2 store bytes.",
        ).set(used_slots * self._slot_size)

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
            self._lifecycle.discard(chunk_key)
            return False
        if doc_id in meta.doc_ids:
            meta.doc_ids.remove(doc_id)
        if meta.doc_ids:
            return False
        self._cm.store.remove(chunk_key)
        self._lifecycle.discard(chunk_key)
        return True

    def _has_store_owner(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
    ) -> bool:
        """Return whether a compatible writer already owns a chunk key.

        Args:
            chunk_key: cache key to inspect.
            token_count: number of tokens expected for this chunk.
            model_id: model identifier expected for reuse isolation.

        Returns:
            True for committed chunks and for uncommitted allocations whose
            first writer has already claimed the store target.
        """
        return self._meta_matches(
            chunk_key, token_count, model_id, self._lifecycle.write_owners
        )

    def _slots_for(self, token_count: int) -> int:
        """Return the number of KV slots needed for ``token_count`` tokens."""
        return math.ceil(token_count / self._block_tokens)

    def _meta_matches(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
        membership: set[str],
    ) -> bool:
        """Return whether stored metadata matches and is in ``membership``.

        Args:
            chunk_key: cache key to inspect.
            token_count: expected token count.
            model_id: expected model identifier.
            membership: set the key must belong to (committed or write-owner).

        Returns:
            True when a metadata entry exists for the key, is in ``membership``,
            and matches the token count, slot count, and model.
        """
        meta = self._cm.store.get(chunk_key)
        if meta is None or chunk_key not in membership:
            return False
        return (
            meta.token_count == token_count
            and meta.num_slots == self._slots_for(token_count)
            and meta.model_id == model_id
        )

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
        self,
        meta: ChunkMeta,
        token_count: int,
        num_slots: int,
        skipped: bool = False,
    ) -> Allocation:
        """Convert ChunkMeta to Allocation."""
        return Allocation(
            chunk_key=meta.chunk_key,
            start_slot=meta.start_slot,
            num_slots=num_slots,
            file_offset=meta.start_slot * self._slot_size,
            pos_offset=meta.pos_offset,
            token_count=token_count,
            skipped=skipped,
        )
