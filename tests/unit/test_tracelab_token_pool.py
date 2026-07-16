# SPDX-License-Identifier: Apache-2.0
"""Tests for TraceLab model-native token-pool preparation."""

from array import array

import pytest

from benchmarks.tracelab.prepare_token_pool import (
    build_token_pool,
    write_token_pool,
)


class _Tokenizer:
    vocab_size = 16
    all_special_ids = [16]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [int(value) for value in text.split()]


def test_build_and_write_token_pool(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1 2 3\n\n4 5\n", encoding="utf-8")

    token_ids = build_token_pool(corpus, _Tokenizer(), limit=4)
    output = tmp_path / "tokens.u32"
    write_token_pool(token_ids, output)

    values = array("I")
    with output.open("rb") as file:
        values.fromfile(file, 4)
    assert token_ids == [1, 2, 3, 4]
    assert values.tolist() == token_ids


def test_build_token_pool_drops_declared_special_token(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("15 16\n", encoding="utf-8")

    assert build_token_pool(corpus, _Tokenizer(), limit=10) == [15]


def test_build_token_pool_rejects_unknown_out_of_range_token(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("15 17\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        build_token_pool(corpus, _Tokenizer(), limit=10)
