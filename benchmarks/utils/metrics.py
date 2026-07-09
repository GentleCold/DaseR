# SPDX-License-Identifier: Apache-2.0
"""Benchmark metrics and correctness helpers."""

from __future__ import annotations

import re
from typing import Any


def contains_accuracy(
    results: list[Any], answers_by_id: dict[int, list[str]]
) -> float | None:
    """Compute answer-containment accuracy for request results.

    Args:
        results: Objects with ``sample_id``, ``generated_text``, and ``error``.
        answers_by_id: Mapping from sample ID to acceptable answers.

    Returns:
        Fraction of successful answerable requests containing at least one
        answer, or None when no answerable successful request exists.

    Thread-safety:
        Pure function.
    """
    ok = [
        result
        for result in results
        if getattr(result, "error", None) is None
        and answers_by_id.get(int(getattr(result, "sample_id", -1)), [])
    ]
    if not ok:
        return None
    hits = 0
    for result in ok:
        generated = str(getattr(result, "generated_text", "")).strip().lower()
        answers = answers_by_id.get(int(getattr(result, "sample_id", -1)), [])
        if any(str(answer).strip().lower() in generated for answer in answers):
            hits += 1
    return hits / len(ok)


def request_text_exact_match(
    cold_results: list[Any],
    warm_results: list[Any],
) -> dict[str, Any]:
    """Compare successful cold/warm request generated text exactly.

    Args:
        cold_results: Cold-phase request results with sample_id, generated_text,
            and error attributes.
        warm_results: Warm-phase request results with sample_id, generated_text,
            and error attributes.

    Returns:
        Exact-match counters and mismatch details. Requests with an error in
        either phase are excluded from the denominator.

    Thread-safety:
        Pure function.
    """
    warm_by_id = {
        int(getattr(result, "sample_id", -1)): result for result in warm_results
    }
    matches = 0
    mismatches = 0
    mismatch_details: list[dict[str, Any]] = []
    for cold in cold_results:
        sample_id = int(getattr(cold, "sample_id", -1))
        warm = warm_by_id.get(sample_id)
        if warm is None:
            continue
        if getattr(cold, "error", None) is not None:
            continue
        if getattr(warm, "error", None) is not None:
            continue
        cold_text = str(getattr(cold, "generated_text", ""))
        warm_text = str(getattr(warm, "generated_text", ""))
        if cold_text == warm_text:
            matches += 1
            continue
        mismatches += 1
        mismatch_details.append(
            {
                "sample_id": sample_id,
                "cold_text": cold_text,
                "warm_text": warm_text,
            }
        )
    total = matches + mismatches
    return {
        "total": total,
        "matches": matches,
        "mismatches": mismatches,
        "accuracy": matches / total if total > 0 else None,
        "mismatch_details": mismatch_details,
    }


def extract_prometheus_counters(text: str) -> dict[str, float]:
    """Extract numeric Prometheus samples by metric name.

    Args:
        text: Prometheus text exposition.

    Returns:
        Mapping from metric name to summed sample value. Labels are ignored so
        multi-engine samples naturally aggregate.

    Thread-safety:
        Pure function.
    """
    counters: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+eE0-9.]+)",
            line,
        )
        if match is None:
            continue
        name, labels, value = match.groups()
        counters[name] = counters.get(name, 0.0) + float(value)
        if labels:
            labeled_name = f"{name}{labels}"
            counters[labeled_name] = counters.get(labeled_name, 0.0) + float(value)
    return counters


def compute_metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    """Return non-negative deltas between two metric snapshots.

    Args:
        before: Metric values before a benchmark phase.
        after: Metric values after a benchmark phase.

    Returns:
        Delta mapping. Counter resets are treated as the after value.

    Thread-safety:
        Pure function.
    """
    delta: dict[str, float] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0.0)
        delta[key] = (
            after_value - before_value if after_value >= before_value else after_value
        )
    return delta


def hit_ratio_from_metrics(
    metrics: dict[str, float], hits_key: str, queries_key: str
) -> float | None:
    """Compute a hit ratio from explicit hit/query counters.

    Args:
        metrics: Metric mapping.
        hits_key: Counter key for hits.
        queries_key: Counter key for query denominator.

    Returns:
        Hit ratio, or None if the denominator is absent or zero.

    Thread-safety:
        Pure function.
    """
    queries = metrics.get(queries_key, 0.0)
    if queries <= 0:
        return None
    return metrics.get(hits_key, 0.0) / queries


def first_available_hit_ratio(
    metrics: dict[str, float], key_pairs: tuple[tuple[str, str], ...]
) -> float | None:
    """Compute the first available hit ratio from candidate metric pairs.

    Args:
        metrics: Metric mapping.
        key_pairs: Ordered ``(hits_key, queries_key)`` candidates.

    Returns:
        First ratio whose denominator exists and is non-zero, otherwise None.

    Thread-safety:
        Pure function.
    """
    for hits_key, queries_key in key_pairs:
        ratio = hit_ratio_from_metrics(metrics, hits_key, queries_key)
        if ratio is not None:
            return ratio
    return None


def extract_lmcache_status_metrics(status: dict[str, Any]) -> dict[str, float]:
    """Extract LMCache hit counters from status JSON when present.

    Args:
        status: ``/status`` payload from LMCache MP HTTP server.

    Returns:
        Flattened counters understood by the benchmark reporter.

    Thread-safety:
        Pure function.
    """
    prefetch = _find_mapping(status, "prefetch_controller") or {}
    requested = _first_number(
        prefetch,
        (
            "requested_tokens",
            "total_requested_tokens",
            "prefetch_requested_tokens",
            "num_requested_tokens",
        ),
    )
    hits = _first_number(
        prefetch,
        (
            "hit_tokens",
            "total_hit_tokens",
            "prefetch_hit_tokens",
            "found_tokens",
        ),
    )
    metrics: dict[str, float] = {}
    if requested is not None:
        metrics["lmcache_prefetch_requested_tokens"] = requested
    if hits is not None:
        metrics["lmcache_prefetch_hit_tokens"] = hits
    return metrics


def _find_mapping(value: Any, key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        found = value.get(key)
        if isinstance(found, dict):
            return found
        for child in value.values():
            nested = _find_mapping(child, key)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for child in value:
            nested = _find_mapping(child, key)
            if nested is not None:
                return nested
    return None


def _first_number(value: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, int | float):
            return float(raw)
    return None
