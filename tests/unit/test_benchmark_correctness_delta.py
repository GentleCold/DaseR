# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark correctness delta matching."""

# Standard
from types import SimpleNamespace

# First Party
from benchmarks.bench_e2e_daser_vs_lmcache import correctness_check


class _Logprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def _output(token_id: int, logprobs: dict[int, float]) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=[token_id],
                logprobs=[{tid: _Logprob(value) for tid, value in logprobs.items()}],
            )
        ],
    )


def test_correctness_delta_accepts_token_difference_within_logprob_tolerance() -> None:
    """Token ID equality is not required when logprob deltas are small."""
    result = correctness_check(
        "test",
        [_output(10, {10: -0.10, 11: -0.12})],
        [_output(11, {10: -0.11, 11: -0.13})],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 0
    assert result["total"] == 1


def test_correctness_delta_rejects_logprob_difference_above_tolerance() -> None:
    """Outputs mismatch only when comparable logprob deltas exceed tolerance."""
    result = correctness_check(
        "test",
        [_output(10, {10: -0.10, 11: -0.12})],
        [_output(11, {10: -0.40, 11: -0.43})],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["indices"] == [0]


def test_correctness_delta_does_not_cross_compare_token_ids() -> None:
    """Only sampled logprob deltas matter when generated token IDs differ."""
    result = correctness_check(
        "test",
        [_output(10, {10: -0.10})],
        [_output(11, {11: -0.12})],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 0
