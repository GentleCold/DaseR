# SPDX-License-Identifier: Apache-2.0
"""Tests for TraceLab wall-clock-gap preparation."""

import csv

from benchmarks.tracelab.prepare_trace import (
    prepare_sessions,
    select_dense_windows,
    select_pause_windows,
    select_sessions,
    write_trace,
)


def test_prepare_trace_preserves_lengths_and_clamps_overlapping_gap(tmp_path) -> None:
    rows = [
        ("a", 100, 10, 0, 1_000, 2_000),
        ("a", 20, 5, 110, 1_900, 2_500),
        ("b", 200, 10, 0, 3_000, 3_500),
        ("b", 30, 5, 210, 5_000, 5_500),
    ]
    sessions = prepare_sessions(rows)
    selected = select_sessions(sessions, max_sessions=2, max_model_len=1_000, seed=1)
    output = tmp_path / "trace.csv"
    write_trace(selected, output, arrival_rate=1.0, seed=2)

    with output.open(newline="", encoding="utf-8") as file:
        emitted = list(csv.DictReader(file))

    by_input = {int(row["input_len"]): row for row in emitted}
    assert by_input[100]["tool_wait_after_ms"] == "0"
    assert by_input[200]["tool_wait_after_ms"] == "1500"
    assert by_input[20]["prefix_len"] == "110"
    assert sum(session.negative_gaps for session in selected) == 1

    windows = select_pause_windows(
        sessions,
        max_sessions=1,
        max_model_len=1_000,
        window_rounds=2,
        min_pause_s=1.0,
        max_pause_s=2.0,
        max_prefix_tokens=1_000,
    )
    assert [round_.input_len for round_ in windows[0].rounds] == [200, 30]


def test_select_dense_windows_rejects_overlapping_and_long_gaps() -> None:
    sessions = prepare_sessions(
        [
            ("dense", 10, 1, 70_000, 1_000, 1_100),
            ("dense", 10, 1, 71_000, 1_200, 1_300),
            ("dense", 10, 1, 72_000, 1_400, 1_500),
            ("dense", 10, 1, 90_000, 5_000, 5_100),
            ("overlap", 10, 1, 100_000, 1_000, 1_200),
            ("overlap", 10, 1, 110_000, 1_100, 1_300),
            ("overlap", 10, 1, 120_000, 1_400, 1_500),
        ]
    )

    windows = select_dense_windows(
        sessions,
        max_sessions=2,
        max_model_len=300_000,
        window_rounds=3,
        max_gap_s=0.5,
        min_prefix_tokens=65_536,
        max_prefix_tokens=200_000,
    )

    assert len(windows) == 1
    assert windows[0].source_id == "dense"
    assert [round_.prefix_len for round_ in windows[0].rounds] == [
        70_000,
        71_000,
        72_000,
    ]
    assert windows[0].negative_gaps == 0
