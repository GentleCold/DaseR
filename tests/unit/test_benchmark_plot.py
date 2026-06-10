# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark comparison plotting helpers."""

from benchmarks.plot_benchmark_comparison import _point_from_phase


def test_plot_point_includes_all_request_elapsed_seconds() -> None:
    """Plot points expose all-request elapsed wall time in seconds."""
    point = _point_from_phase(
        "Chunk",
        "LMCache",
        "Warm",
        {
            "summary": {
                "ttft_ms_mean": 10.0,
                "all_requests_elapsed_ms": 2500.0,
            },
            "requests": [{"ttft_ms": 8.0}, {"ttft_ms": 12.0}],
        },
    )

    assert point.all_requests_elapsed_s == 2.5


def test_plot_point_uses_phase_elapsed_fallback() -> None:
    """Existing result files can still be plotted using phase elapsed time."""
    point = _point_from_phase(
        "Prefix",
        "DaseR",
        "Warm",
        {
            "summary": {
                "ttft_ms_mean": 10.0,
                "phase_elapsed_ms": 3000.0,
            },
            "requests": [{"ttft_ms": 10.0}],
        },
    )

    assert point.all_requests_elapsed_s == 3.0
