# SPDX-License-Identifier: Apache-2.0
"""Capacity telemetry preserves rank identity and reports allocations, not sums."""

from functools import partial
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", minversion="0.30.0")

from prometheus_client import (  # noqa: E402
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from daser.attention.capacity import FragPackCapacity  # noqa: E402
from daser.connector.daser_connector import DaserConnector  # noqa: E402
from daser.connector.metrics import DaserConnectorStats  # noqa: E402


def _snapshot(blocks: int, arena: int) -> dict:
    """Return serializable allocation data for one worker."""
    return FragPackCapacity(
        blocks=blocks,
        block_tokens=128,
        layers=2,
        kv_heads=1,
        page_bytes=47616,
        raw_page_bytes=65536,
        memory={"arena": arena, "overflow": 32768, "connector_staging": 65536},
    ).to_dict()


def test_capacity_aggregation_replaces_observations_and_preserves_ranks() -> None:
    """Repeated intervals must not multiply capacity or collapse TP workers."""
    first = DaserConnector.build_kv_connector_stats(
        {"capacity": {"0": _snapshot(10, 952320)}}
    )
    second = DaserConnectorStats({"capacity": {"1": _snapshot(10, 952320)}})
    acc = DaserConnector.build_kv_connector_stats()
    acc.aggregate(first).aggregate(second).aggregate(first)
    assert set(acc.to_dict()["capacity"]) == {"0", "1"}
    assert acc.reduce()["fragpack_blocks_per_rank_min"] == 10
    assert acc.reduce()["fragpack_tensor_bytes"] == 2 * (952320 + 32768 + 65536)

    acc.aggregate(DaserConnectorStats({"capacity": {"0": _snapshot(8, 761856)}}))
    assert acc.reduce()["fragpack_blocks_per_rank_min"] == 8
    assert first.to_dict()["capacity"]["0"]["blocks"] == 10
    acc.reset()
    assert acc.is_empty()
    assert acc.reduce() == {}
    assert not first.is_empty()


def test_vllm_metrics_hook_exports_latest_capacity_per_engine_and_rank() -> None:
    """Public connector hooks expose parseable gauges without fabricated zeros."""
    registry = CollectorRegistry()
    collector = DaserConnector.build_prom_metrics(
        SimpleNamespace(kv_transfer_config=None),
        {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        },
        ["model_name", "engine"],
        {0: ["demo", "0"], 1: ["demo", "1"]},
    )
    collector.observe({})
    assert (
        registry.get_sample_value(
            "daser_fragpack_blocks",
            {"model_name": "demo", "engine": "0", "tp_rank": "0"},
        )
        is None
    )

    data = {"capacity": {"0": _snapshot(10, 952320), "1": _snapshot(10, 952320)}}
    collector.observe(data)
    collector.observe(data)
    collector.observe({"capacity": {"0": _snapshot(7, 666624)}}, engine_idx=1)
    labels = {"model_name": "demo", "engine": "0", "tp_rank": "1"}
    assert registry.get_sample_value("daser_fragpack_blocks", labels) == 10
    assert (
        registry.get_sample_value(
            "daser_fragpack_memory_bytes", {**labels, "component": "arena"}
        )
        == 952320
    )
    assert (
        registry.get_sample_value(
            "daser_fragpack_blocks", {**labels, "engine": "1", "tp_rank": "0"}
        )
        == 7
    )

    collector.observe({"capacity": {"1": _snapshot(8, 761856)}})
    assert registry.get_sample_value("daser_fragpack_blocks", labels) == 8
    assert b"# TYPE daser_fragpack_memory_bytes gauge" in generate_latest(registry)


def test_capacity_snapshot_does_not_share_mutable_payloads() -> None:
    """Connector additions must not change the attention-owned snapshot."""
    capacity = FragPackCapacity(10, 128, 2, 1, 47616, 65536, {"arena": 952320})
    wire = capacity.to_dict()
    wire["memory"]["connector_staging"] = 65536
    assert capacity.memory == {"arena": 952320}
