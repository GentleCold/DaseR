# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic multi-prefix benchmark manifest."""

from __future__ import annotations

from benchmarks.multiprefix_bench import build_request_manifest


class _Tokenizer:
    all_special_ids = [0, 1]

    def __len__(self) -> int:
        return 1000


def test_manifest_uses_independent_prefix_families() -> None:
    """Each family shares only its own block-aligned prefix."""
    manifest = build_request_manifest(
        tokenizer=_Tokenizer(),
        families=3,
        requests_per_family=2,
        prefix_tokens=8,
        suffix_tokens=4,
        block_tokens=4,
        seed=42,
    )

    requests = manifest["requests"]
    assert manifest["total_requests"] == 6
    assert manifest["expected_hit_tokens"] == 24
    assert manifest["expected_hit_rate"] == 24 / 72
    assert requests[0]["prompt"][:8] == requests[1]["prompt"][:8]
    assert requests[0]["prompt"][:8] != requests[2]["prompt"][:8]
    assert requests[0]["prompt"][8:] != requests[1]["prompt"][8:]


def test_manifest_rejects_unaligned_lengths() -> None:
    """Prefix and suffix lengths must map to whole KV blocks."""
    try:
        build_request_manifest(
            tokenizer=_Tokenizer(),
            families=1,
            requests_per_family=2,
            prefix_tokens=7,
            suffix_tokens=4,
            block_tokens=4,
            seed=42,
        )
    except ValueError as exc:
        assert "block aligned" in str(exc)
    else:
        raise AssertionError("unaligned prompt lengths must be rejected")
