# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark exact text/token correctness."""

# Standard
from types import SimpleNamespace

# First Party
from benchmarks.bench_common import (
    correctness_check,
    correctness_check_with_visibility,
)


def _output(token_id: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=[token_id],
                text=text,
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
    assert result["total"] == 1
    assert result["indices"] == []


def test_correctness_rejects_text_mismatch() -> None:
    """Outputs fail when text differs even if token IDs match."""
    result = correctness_check(
        "test",
        [_output(10, " yes")],
        [_output(10, " no")],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["indices"] == [0]
    assert result["mismatch_details"][0]["cold_token_ids"] == [10]
    assert result["mismatch_details"][0]["warm_token_ids"] == [10]
    assert result["mismatch_details"][0]["cold_text"] == " yes"
    assert result["mismatch_details"][0]["warm_text"] == " no"


def test_correctness_rejects_token_mismatch() -> None:
    """Outputs fail when token IDs differ even if decoded text matches."""
    result = correctness_check(
        "test",
        [_output(10, " yes")],
        [_output(11, " yes")],
        [[1, 2, 3]],
        64,
    )

    assert result["mismatches"] == 1
    assert result["indices"] == [0]
    assert result["mismatch_details"][0]["cold_token_ids"] == [10]
    assert result["mismatch_details"][0]["warm_token_ids"] == [11]


def test_correctness_splits_visible_hit_mismatches() -> None:
    """Visible-hit counters count exact mismatches only for visible prompts."""
    result = correctness_check_with_visibility(
        "test",
        [_output(10, " yes"), _output(20, " maybe")],
        [_output(11, " yes"), _output(21, " no")],
        [[1, 2, 3], [4, 5, 6]],
        64,
        [True, False],
    )

    assert result["mismatches"] == 2
    assert result["visible_total"] == 1
    assert result["visible_mismatches"] == 1
