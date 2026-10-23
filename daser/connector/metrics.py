# SPDX-License-Identifier: Apache-2.0
"""FragPack allocation telemetry over vLLM's public connector stats channel."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from prometheus_client import Gauge
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

_GEOMETRY = {
    "blocks": "Allocated FragPack blocks per worker, including reserved blocks.",
    "block_tokens": "Token slots per FragPack block.",
    "layers": "Attention layers in this worker's FragPack store.",
    "kv_heads": "KV heads per layer on this worker.",
    "page_bytes": "Allocated FragPack page bytes per block-layer on this worker.",
    "raw_page_bytes": "BF16 bytes for this block-layer geometry, not a baseline run.",
}


class DaserConnectorStats(KVConnectorStats):
    """Latest capacity snapshot per TP rank, serialized by vLLM.

    Async/thread-safety:
        Owned by one worker or statistics aggregation thread at a time.
        Aggregation replaces each rank's snapshot rather than summing repeated
        observations. TP ranks have different memory shards but share logical
        block ids, so block counts must never be added across ranks.
    """

    def reset(self) -> None:
        """Clear observations; called by the owning statistics thread.

        Returns:
            None. Does not mutate any input snapshot supplied to aggregate.
        """
        self.data = {}

    def aggregate(self, other: KVConnectorStats) -> DaserConnectorStats:
        """Merge the most recent allocation for each worker.

        Args:
            other: Later worker or interval statistics.

        Returns:
            This accumulator, updated in place on its owning thread. Input
            snapshots are copied so subsequent aggregation cannot alter them.
        """
        if other.data.get("capacity"):
            self.data.setdefault("capacity", {}).update(
                deepcopy(other.data["capacity"])
            )
        return self

    def reduce(self) -> dict[str, int | float]:
        """Summarize the latest allocations for the owning text logger.

        Returns:
            Minimum per-rank block count and summed per-rank tensor bytes.
            Reads host values only; no I/O or synchronization.
        """
        ranks = list(self.data.get("capacity", {}).values())
        if not ranks:
            return {}
        return {
            "fragpack_blocks_per_rank_min": min(r["blocks"] for r in ranks),
            "fragpack_tensor_bytes": sum(sum(r["memory"].values()) for r in ranks),
        }

    def is_empty(self) -> bool:
        """Return whether the owning thread has observed any capacity snapshot.

        Returns:
            True before the first initialized store is observed. No I/O.
        """
        return not self.data.get("capacity")


class DaserPromMetrics(KVConnectorPromMetrics):
    """Expose worker allocations from vLLM's API-server metrics endpoint.

    Args:
        vllm_config: Serving configuration supplied by vLLM.
        metric_types: Public vLLM metric factories, including its Gauge type.
        labelnames: vLLM's model and engine label names.
        per_engine_labelvalues: Corresponding values keyed by engine index.

    Async/thread-safety:
        Construct and observe on vLLM's statistics thread. Worker processes
        never register Prometheus collectors or perform telemetry I/O.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        gauge = metric_types[Gauge]
        labels = [*labelnames, "tp_rank"]
        self._geometry = {
            key: gauge(
                name=f"daser_fragpack_{key}",
                documentation=description,
                labelnames=labels,
                multiprocess_mode="mostrecent",
            )
            for key, description in _GEOMETRY.items()
        }
        self._memory = gauge(
            name="daser_fragpack_memory_bytes",
            documentation="Persistent FragPack tensor bytes by worker and component; "
            "excludes weights, allocator reserve, graphs and per-step temporaries.",
            labelnames=[*labels, "component"],
            multiprocess_mode="mostrecent",
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        """Publish a host-only allocation snapshot without inventing zero data.

        Args:
            transfer_stats_data: Serialized DaserConnectorStats observations.
            engine_idx: Engine whose workers produced the observations.

        Returns:
            None. Gauges retain the latest allocation between observations.

        Async/thread-safety:
            Called on the API server's statistics thread; no GPU access.
        """
        for rank, capacity in transfer_stats_data.get("capacity", {}).items():
            labels = [*self.per_engine_labelvalues[engine_idx], rank]
            for key, metric in self._geometry.items():
                metric.labels(*labels).set(capacity[key])
            for component, nbytes in capacity["memory"].items():
                self._memory.labels(*labels, component).set(nbytes)
