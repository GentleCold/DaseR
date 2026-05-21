# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark text correctness and top-k mismatch diagnostics."""

# Standard
from types import SimpleNamespace

# First Party
from benchmarks.bench_e2e_daser_vs_lmcache import correctness_check


class _Logprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def _output(
    token_id: int,
    text: str,
    topk: dict[int, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=[token_id],
                text=text,
                logprobs=(
                    [{tid: _Logprob(value) for tid, value in (topk or {}).items()}]
                    if topk is not None
                    else None
                ),
            )
        ],
    )


def test_correctness_accepts_exact_text_and_tokens() -> None:
    """Outputs pass when generated text and token IDs are identical."""
    result = correctness_check(
        "test",
        [_output(10, " yes")],
        [_output(10, " yes")],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 0
    assert result["allowed_mismatches"] == 0
    assert result["strict_mismatches"] == 0
    assert result["total"] == 1


def test_correctness_allows_close_topk_token_mismatch() -> None:
    """Text mismatch is allowed when both tokens are close top-k contenders."""
    result = correctness_check(
        "test",
        [_output(10, " yes", {10: -0.50, 11: -0.56, 12: -1.0})],
        [_output(11, " no", {11: -0.51, 10: -0.55, 12: -1.0})],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["allowed_mismatches"] == 1
    assert result["strict_mismatches"] == 0
    assert result["allowed_indices"] == [0]
    assert result["mismatch_details"][0]["reason"] == "topk_close_margin"


def test_correctness_rejects_large_margin_mismatch() -> None:
    """Text mismatch is strict when top1 is clearly separated from top2."""
    result = correctness_check(
        "test",
        [_output(10, " yes", {10: -0.10, 11: -1.00, 12: -1.5})],
        [_output(11, " no", {11: -0.20, 10: -1.10, 12: -1.5})],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["allowed_mismatches"] == 0
    assert result["strict_mismatches"] == 1
    assert result["strict_indices"] == [0]


def test_correctness_rejects_token_missing_from_peer_topk() -> None:
    """Text mismatch is strict when chosen tokens are not in peer top-k."""
    result = correctness_check(
        "test",
        [_output(10, " yes", {10: -0.50, 12: -0.55, 13: -1.0})],
        [_output(11, " no", {11: -0.51, 12: -0.54, 13: -1.0})],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["allowed_mismatches"] == 0
    assert result["strict_mismatches"] == 1
