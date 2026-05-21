# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``examples/service_demo/metrics.py``."""

# Standard
import importlib.util
from pathlib import Path

# First Party
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_METRICS = _REPO_ROOT / "examples" / "service_demo" / "metrics.py"
_spec = importlib.util.spec_from_file_location("service_demo_metrics", _EXAMPLE_METRICS)
assert _spec and _spec.loader
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)

contiguous_covered_tokens = _metrics.contiguous_covered_tokens
summarize_cache_hits = _metrics.summarize_cache_hits
trial_stats = _metrics.trial_stats


def test_contiguous_covered_tokens_stops_at_gap() -> None:
    hits = [
        {"target_token_start": 0, "token_count": 16},
        {"target_token_start": 32, "token_count": 16},
    ]
    assert contiguous_covered_tokens(hits, prompt_tokens=100) == 16


def test_summarize_cache_hits_estimates_with_layers() -> None:
    hits = [
        {
            "target_token_start": 0,
            "token_count": 16,
            "num_slots": 1,
            "pos_offset": 0,
        },
        {
            "target_token_start": 16,
            "token_count": 144,
            "num_slots": 9,
            "pos_offset": 16,
        },
    ]
    summary = summarize_cache_hits(hits, block_tokens=16, num_layers=36)
    assert summary["hit_count"] == 2
    assert summary["estimated_gds_reads"] == 2
    assert summary["estimated_layer_index_copies"] == 72
    assert summary["estimated_rope_block_ops"] == 10 * 36


def test_trial_stats() -> None:
    stats = trial_stats([100.0, 120.0, 110.0])
    assert stats["median_ms"] == 110.0
    assert stats["min_ms"] == 100.0
