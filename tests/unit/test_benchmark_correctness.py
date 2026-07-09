# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark exact text/token correctness."""

# Standard
from types import SimpleNamespace

# First Party
from benchmarks.utils.metrics import (
    contains_accuracy,
    request_text_exact_match,
)


def test_contains_accuracy_ignores_errors() -> None:
    """Answer containment ignores failed requests."""
    results = [
        SimpleNamespace(sample_id=1, generated_text="the answer is Paris", error=None),
        SimpleNamespace(sample_id=2, generated_text="", error="failed"),
    ]

    assert contains_accuracy(results, {1: ["Paris"], 2: ["London"]}) == 1.0


def test_contains_accuracy_returns_none_without_answerable_results() -> None:
    """Answer containment is None when the workload has no answer labels."""
    results = [
        SimpleNamespace(sample_id=1, generated_text="free-form answer", error=None),
    ]

    assert contains_accuracy(results, {}) is None


def test_request_text_exact_match_compares_cold_and_warm_results() -> None:
    """Service benchmark correctness compares IMDB cold/warm generated text."""
    cold = [
        SimpleNamespace(sample_id=1, generated_text="positive", error=None),
        SimpleNamespace(sample_id=2, generated_text="negative", error=None),
        SimpleNamespace(sample_id=3, generated_text="", error="timeout"),
    ]
    warm = [
        SimpleNamespace(sample_id=1, generated_text="positive", error=None),
        SimpleNamespace(sample_id=2, generated_text="mixed", error=None),
        SimpleNamespace(sample_id=3, generated_text="", error=None),
    ]

    result = request_text_exact_match(cold, warm)

    assert result["total"] == 2
    assert result["matches"] == 1
    assert result["mismatches"] == 1
    assert result["accuracy"] == 0.5
    assert result["mismatch_details"] == [
        {
            "sample_id": 2,
            "cold_text": "negative",
            "warm_text": "mixed",
        }
    ]
