# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark softmax probability matching."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from benchmarks.bench_e2e_daser_vs_lmcache import correctness_check


class _FlatLogits:
    def __init__(self, values: list[float]) -> None:
        self.start_indices = [0]
        self.end_indices = [len(values)]
        self.logprobs = values


def _output(logits: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=[10],
                logprobs=_FlatLogits(logits),
            )
        ],
    )


def test_correctness_accepts_equal_softmax_probabilities() -> None:
    """Outputs pass when logits differ but softmax probabilities are equal."""
    result = correctness_check(
        "test",
        [_output([1.0, 2.0, 3.0])],
        [_output([11.0, 12.0, 13.0])],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 0
    assert result["total"] == 1
    assert result["max_prob_abs_diff"] == pytest.approx(0.0)


def test_correctness_rejects_probability_diff_above_tolerance() -> None:
    """Outputs mismatch only when probability max_abs_diff exceeds tolerance."""
    result = correctness_check(
        "test",
        [_output([0.0, 0.0])],
        [_output([10.0, 0.0])],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["indices"] == [0]
    assert result["max_prob_abs_diff"] == pytest.approx(0.4999546)


def test_correctness_rejects_missing_logits() -> None:
    """Outputs mismatch when full logits are unavailable."""
    result = correctness_check(
        "test",
        [_output([1.0, 2.0, 3.0])],
        [SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[SimpleNamespace()])],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["max_prob_abs_diff"] == float("inf")
