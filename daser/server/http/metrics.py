# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics collector for DaseR Stage A observability.

The ``MetricsCollector`` owns counters, gauges, and histograms registered
with the ``prometheus_client`` library. It is injected into ``build_http_app``
as an optional parameter so existing tests and deployments that do not enable
metrics remain unaffected.
"""

# Standard
from typing import TYPE_CHECKING

# Third Party
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from daser.server.core import ServerCore


class MetricsCollector:
    """Prometheus metrics collector for the DaseR HTTP server.

    Each collector instance owns an independent ``CollectorRegistry`` so
    multiple inspectors or test fixtures can co-exist without duplicate
    metric errors.

    Args:
        enabled: when False, ``export_metrics`` and route hooks are no-ops.

    Async/thread-safety:
        Each instance uses its own ``CollectorRegistry``. Prometheus metrics
        use internal locks and are safe for concurrent calls from FastAPI
        endpoints.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        reg = CollectorRegistry(auto_describe=True)

        histogram_buckets = (
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            float("inf"),
        )

        self.http_requests_total = Counter(
            "daser_http_requests_total",
            "Total HTTP requests served",
            ["method", "endpoint", "status"],
            registry=reg,
        )
        self.chunk_hits_total = Counter(
            "daser_chunk_hits_total",
            "Chunk cache hits by document",
            ["doc_id"],
            registry=reg,
        )
        self.chunk_misses_total = Counter(
            "daser_chunk_misses_total",
            "Chunk cache misses by document",
            ["doc_id"],
            registry=reg,
        )
        self.lookup_total = Counter(
            "daser_lookup_total",
            "Server-side cache lookups by result",
            ["status"],
            registry=reg,
        )
        self.ttft_seconds = Histogram(
            "daser_ttft_seconds",
            "Time-to-first-token for inference requests",
            buckets=histogram_buckets,
            registry=reg,
        )
        self.inference_latency_seconds = Histogram(
            "daser_inference_latency_seconds",
            "End-to-end inference wall latency",
            buckets=histogram_buckets,
            registry=reg,
        )
        self.document_prefill_seconds = Histogram(
            "daser_document_prefill_seconds",
            "Document upload prefill latency",
            buckets=histogram_buckets,
            registry=reg,
        )
        self.cache_chunks = Gauge(
            "daser_cache_chunks",
            "Number of chunks currently stored in the ring buffer",
            registry=reg,
        )
        self.cache_free_slots = Gauge(
            "daser_cache_free_slots",
            "Number of free slots in the ring buffer",
            registry=reg,
        )
        self.cache_total_slots = Gauge(
            "daser_cache_total_slots",
            "Total slot capacity of the ring buffer",
            registry=reg,
        )
        self.eviction_total = Counter(
            "daser_eviction_total",
            "Chunk evictions by reason",
            ["reason"],
            registry=reg,
        )
        self.documents_total = Gauge(
            "daser_documents_total",
            "Number of registered documents",
            registry=reg,
        )

        self._registry = reg
        self._seen_eviction_counts: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        """Return whether metrics collection is active.

        Returns:
            ``True`` when route hooks should record metrics.

        Async/thread-safety:
            Reads immutable configuration.
        """
        return self._enabled

    def export_metrics(self) -> bytes:
        """Return the current Prometheus text-format metrics payload.

        Returns:
            Bytes suitable for a ``text/plain; version=0.0.4`` response body.
            When the collector is disabled the payload contains only a disabled
            marker comment.

        Async/thread-safety:
            Delegates to prometheus_client, which guards registry collection
            with its own locks.
        """
        if not self._enabled:
            return b"# metrics disabled\n"
        return generate_latest(self._registry)

    def state_snapshot(self, core: "ServerCore") -> None:
        """Sync point-in-time gauges from the shared server core.

        Args:
            core: active ``ServerCore`` instance whose ``ChunkManager`` and
                ``DocRegistry`` provide the current cache and document state.

        Returns:
            None.

        Async/thread-safety:
            Reads in-memory state on the FastAPI event loop. The same
            event-loop ownership used by ``ServerCore`` guarantees exclusive
            access.
        """
        if not self._enabled:
            return
        cm = core.chunk_manager
        self.cache_chunks.set(len(cm.store))
        self.cache_free_slots.set(cm.free_slots)
        self.cache_total_slots.set(cm.total_slots)
        registry = getattr(cm, "doc_registry", None)
        if registry is not None:
            self.documents_total.set(len(registry))
        else:
            self.documents_total.set(0)
        for reason, count in core.eviction_stats().items():
            previous = self._seen_eviction_counts.get(reason, 0)
            if count > previous:
                self.eviction_total.labels(reason=reason).inc(count - previous)
                self._seen_eviction_counts[reason] = count

    def record_lookup_hit(self, doc_id: str) -> None:
        """Record a chunk cache hit for one document.

        Args:
            doc_id: document identifier that owns the cached chunk.

        Returns:
            None.

        Async/thread-safety:
            Updates a prometheus_client counter guarded by its internal lock.
        """
        if not self._enabled:
            return
        self.chunk_hits_total.labels(doc_id=doc_id).inc()

    def record_lookup_miss(self, doc_id: str) -> None:
        """Record a chunk cache miss for one document.

        Args:
            doc_id: document identifier referenced by the lookup.

        Returns:
            None.

        Async/thread-safety:
            Updates a prometheus_client counter guarded by its internal lock.
        """
        if not self._enabled:
            return
        self.chunk_misses_total.labels(doc_id=doc_id).inc()

    def record_http_request(self, method: str, endpoint: str, status: int) -> None:
        """Record one HTTP request.

        Args:
            method: HTTP method name.
            endpoint: route template, for example ``"/documents/{doc_id}"``.
            status: HTTP response status code.

        Returns:
            None.

        Async/thread-safety:
            Updates a prometheus_client counter guarded by its internal lock.
        """
        if not self._enabled:
            return
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status),
        ).inc()

    def record_eviction(self, reason: str) -> None:
        """Record a chunk eviction.

        Args:
            reason: human-readable cause, for example ``"ring"`` or
                ``"document_delete"`` in Stage A.

        Returns:
            None.

        Async/thread-safety:
            Updates a prometheus_client counter guarded by its internal lock.
        """
        if not self._enabled:
            return
        self.eviction_total.labels(reason=reason).inc()

    def record_lookup_result(self, hit: bool) -> None:
        """Record a server-side cache lookup result.

        Args:
            hit: True when at least one chunk matched the lookup prompt.

        Returns:
            None.

        Async/thread-safety:
            Updates a prometheus_client counter guarded by its internal lock.
        """
        if not self._enabled:
            return
        status = "hit" if hit else "miss"
        self.lookup_total.labels(status=status).inc()

    def record_inference(self, ttft_ms: float, latency_ms: float) -> None:
        """Record inference latency for a single completion request.

        Args:
            ttft_ms: time-to-first-token in milliseconds.
            latency_ms: end-to-end wall latency in milliseconds.

        Returns:
            None.

        Async/thread-safety:
            Updates prometheus_client histograms guarded by internal locks.
        """
        if not self._enabled:
            return
        self.ttft_seconds.observe(ttft_ms / 1000.0)
        self.inference_latency_seconds.observe(latency_ms / 1000.0)

    def record_document_prefill(self, prefill_ms: float) -> None:
        """Record document upload prefill latency.

        Args:
            prefill_ms: time spent in prefill during upload, in milliseconds.

        Returns:
            None.

        Async/thread-safety:
            Updates a prometheus_client histogram guarded by its internal lock.
        """
        if not self._enabled:
            return
        self.document_prefill_seconds.observe(prefill_ms / 1000.0)
