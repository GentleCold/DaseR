# SPDX-License-Identifier: Apache-2.0
"""Helpers for summarizing HTTP ``trace_cache`` hits in the service demo.

These summaries approximate connector load cost for issue #35 investigations.
They reflect ``ServerCore.lookup`` (control plane), not vLLM scheduler
``extra_tokens`` or worker debug logs.
"""

# Standard
import math
import statistics
from typing import Any


def contiguous_covered_tokens(hits: list[dict[str, Any]], prompt_tokens: int) -> int:
    """Return tokens covered by a contiguous prefix of block-aligned hits.

    Args:
        hits: ``cache_hits`` entries from ``POST /infer`` with
            ``target_token_start`` and ``token_count``.
        prompt_tokens: total prompt length (caps the covered range).

    Returns:
        Number of prompt tokens in the longest contiguous hit prefix starting
        at offset 0. Non-contiguous hits after a gap stop the scan.
    """
    if not hits:
        return 0
    covered_until = 0
    for hit in sorted(hits, key=lambda item: int(item.get("target_token_start", 0))):
        start = int(hit.get("target_token_start", 0))
        count = int(hit.get("token_count", 0))
        end = start + count
        if end <= covered_until:
            continue
        if start > covered_until:
            break
        covered_until = min(end, prompt_tokens)
    return covered_until


def summarize_cache_hits(
    hits: list[dict[str, Any]],
    *,
    block_tokens: int = 16,
    num_layers: int | None = None,
) -> dict[str, Any]:
    """Summarize cache hits for demo reporting and JSON export.

    Args:
        hits: ``cache_hits`` list from ``POST /infer`` when ``trace_cache`` is
            enabled.
        block_tokens: vLLM block size used for slot estimates.
        num_layers: optional model layer count; when set, adds connector load
            estimates (one GDS read per hit, per-layer copies, per-block RoPE).

    Returns:
        Summary dict safe for JSON serialization.
    """
    target_starts = [int(h.get("target_token_start", 0)) for h in hits]
    token_lengths = [int(h.get("token_count", 0)) for h in hits]
    num_slots_list = [int(h.get("num_slots", 0)) for h in hits]
    if hits and not any(num_slots_list):
        num_slots_list = [math.ceil(max(t, 0) / block_tokens) for t in token_lengths]
    reused_tokens = sum(token_lengths)
    num_slots_sum = sum(num_slots_list)
    summary: dict[str, Any] = {
        "hit_count": len(hits),
        "reused_token_sum": reused_tokens,
        "num_slots_sum": num_slots_sum,
        "target_token_starts": target_starts,
        "token_lengths": token_lengths,
        "num_slots": num_slots_list,
        "estimated_gds_reads": len(hits),
    }
    if num_layers is not None and num_layers > 0:
        summary["estimated_layer_index_copies"] = len(hits) * num_layers
        has_pos_offset = any(int(h.get("pos_offset", 0)) != 0 for h in hits) or any(
            target_starts
        )
        if has_pos_offset:
            summary["estimated_rope_block_ops"] = num_slots_sum * num_layers
    return summary


def trial_stats(values: list[float]) -> dict[str, float]:
    """Compute mean and median for a list of millisecond samples.

    Args:
        values: non-empty list of durations in milliseconds.

    Returns:
        Dict with ``mean_ms``, ``median_ms``, ``min_ms``, ``max_ms``.

    Raises:
        ValueError: if ``values`` is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }
