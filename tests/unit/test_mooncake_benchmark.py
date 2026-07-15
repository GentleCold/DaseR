# SPDX-License-Identifier: Apache-2.0
"""Tests for the Mooncake production trace benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mooncake.benchmark import (
    MOONCAKE_BLOCK_TOKENS,
    PromptBuilder,
    TraceRequest,
    load_trace,
    scheduled_offset_seconds,
    select_requests,
    summarise,
)


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_trace_validates_order_and_block_count(tmp_path: Path) -> None:
    """Rows must be source ordered and contain one hash per 512-token block."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "timestamp": 0,
                "input_length": 513,
                "output_length": 4,
                "hash_ids": [10, 11],
            },
            {
                "timestamp": 3,
                "input_length": 512,
                "output_length": 2,
                "hash_ids": [10],
            },
        ],
    )

    requests = load_trace(trace)

    assert [request.index for request in requests] == [0, 1]
    assert requests[0].hash_ids == (10, 11)

    _write_trace(
        trace,
        [
            {
                "timestamp": 3,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [1],
            },
            {
                "timestamp": 2,
                "input_length": 512,
                "output_length": 1,
                "hash_ids": [2],
            },
        ],
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        load_trace(trace)


def test_load_trace_selects_contiguous_rebased_window(tmp_path: Path) -> None:
    """A source window preserves row IDs and relative arrival gaps."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "timestamp": index * 10,
                "input_length": 1,
                "output_length": 1,
                "hash_ids": [index],
            }
            for index in range(4)
        ],
    )

    requests = load_trace(trace, max_requests=2, start_request=1)

    assert [request.index for request in requests] == [1, 2]
    assert [request.timestamp_ms for request in requests] == [0, 10]


def test_load_trace_filters_context_stratum_before_limit(tmp_path: Path) -> None:
    """Context filtering keeps source order and limits retained rows."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "timestamp": index * 10,
                "input_length": length,
                "output_length": 1,
                "hash_ids": [index] * ((length + 511) // 512),
            }
            for index, length in enumerate((10, 600, 20, 700))
        ],
    )

    requests = load_trace(trace, max_requests=2, min_context_tokens=500)

    assert [request.index for request in requests] == [1, 3]
    assert [request.timestamp_ms for request in requests] == [0, 20]


def test_prompt_builder_preserves_full_and_partial_shared_blocks() -> None:
    """Equal hash prefixes yield equal tokens, including a partial last block."""
    builder = PromptBuilder(vocab_size=1024, special_token_ids={0, 1, 2})
    first = TraceRequest(0, 0, 700, 1, (10, 11))
    second = TraceRequest(1, 0, 900, 1, (10, 11))
    different = TraceRequest(2, 0, 700, 1, (10, 12))

    first_prompt = builder.build(first)
    second_prompt = builder.build(second)
    different_prompt = builder.build(different)

    assert len(first_prompt) == 700
    assert first_prompt == second_prompt[:700]
    assert (
        first_prompt[:MOONCAKE_BLOCK_TOKENS] == different_prompt[:MOONCAKE_BLOCK_TOKENS]
    )
    assert (
        first_prompt[MOONCAKE_BLOCK_TOKENS:] != different_prompt[MOONCAKE_BLOCK_TOKENS:]
    )
    assert not ({0, 1, 2} & set(first_prompt))


def test_arrival_scaling_preserves_same_timestamp_bursts() -> None:
    """Scaling changes time units without separating simultaneous arrivals."""
    assert scheduled_offset_seconds(3000, 1.0) == 3.0
    assert scheduled_offset_seconds(3000, 2.0) == 1.5
    assert scheduled_offset_seconds(3000, 4.0) == scheduled_offset_seconds(3000, 4.0)
    with pytest.raises(ValueError, match="positive"):
        scheduled_offset_seconds(0, 0)


def test_context_selection_and_summary_report_skipped_token_mass() -> None:
    """Overflow rows remain visible and successful latency excludes failures."""
    requests = [
        TraceRequest(0, 0, 1000, 10, (1, 2)),
        TraceRequest(1, 1, 2000, 100, (3, 4, 5, 6)),
    ]
    selection = select_requests(requests, max_model_len=1500)

    summary = summarise(
        selection,
        [
            {
                "error": None,
                "ttft_ms": 10,
                "latency_ms": 20,
                "client_queue_ms": 1,
                "arrival_to_first_token_ms": 12,
                "arrival_to_completion_ms": 22,
            },
            {
                "error": "failed",
                "ttft_ms": 0,
                "latency_ms": 0,
                "client_queue_ms": 0,
                "arrival_to_first_token_ms": 0,
                "arrival_to_completion_ms": 0,
            },
        ],
        wall_seconds=2,
        time_scale=1,
        prefix_cache_metrics={
            "vllm:prefix_cache_queries_total": 100,
            "vllm:prefix_cache_hits_total": 60,
        },
    )

    assert [request.index for request in selection.eligible] == [0]
    assert [request.index for request in selection.skipped] == [1]
    assert summary["skipped_context_requests"] == 1
    assert summary["skipped_input_tokens"] == 2000
    assert summary["completed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["ttft_ms"]["p99"] == 10
    assert summary["arrival_to_first_token_ms"]["p99"] == 12
    assert summary["prefix_cache"]["local_hit_rate"] == 0.6


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"timestamp": 0, "input_length": 0, "output_length": 1, "hash_ids": []},
            "positive",
        ),
        (
            {"timestamp": 0, "input_length": 513, "output_length": 1, "hash_ids": [1]},
            "expected 2 hash IDs",
        ),
        (
            {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [-1]},
            "non-negative",
        ),
    ],
)
def test_load_trace_rejects_invalid_rows(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    """Invalid lengths and hash metadata fail before replay begins."""
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, [row])

    with pytest.raises(ValueError, match=message):
        load_trace(trace)
