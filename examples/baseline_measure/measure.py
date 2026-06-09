# SPDX-License-Identifier: Apache-2.0
"""Baseline measurement script for DaseR Stage A observability.

Runs a sequence of document uploads and inference requests against a
running DaseR HTTP server and reports cache-hit ratio, TTFT, and latency
distributions. This script is standalone and not imported by DaseR.

Usage::

    python examples/baseline_measure/measure.py \\
        --service-url http://127.0.0.1:2026 \\
        --samples 20

Requirements: ``httpx`` (already a DaseR dependency).
"""

# Standard
import argparse
import json
import statistics
import time
from typing import Any


def _upload(client: Any, service_url: str, title: str, text: str) -> dict:
    """Upload a single document.

    Args:
        client: ``httpx.Client`` connected to the DaseR server.
        service_url: base URL of the DaseR HTTP API.
        title: display title for the document.
        text: raw document text.

    Returns:
        The JSON response body from ``POST /documents``.

    Raises:
        RuntimeError: if the upload fails.
    """
    resp = client.post(
        f"{service_url}/documents",
        json={"title": title, "text": text},
        timeout=600.0,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"upload failed: {resp.status_code} {resp.text}")
    return resp.json()


def _infer(
    client: Any,
    service_url: str,
    doc_ids: list[str],
    task: str,
    use_kv_cache: bool = True,
) -> tuple[dict, float]:
    """Run one inference request.

    Args:
        client: ``httpx.Client`` connected to the DaseR server.
        service_url: base URL of the DaseR HTTP API.
        doc_ids: ordered document identifiers.
        task: user task text.
        use_kv_cache: when False, sets ``daser_skip_load`` to bypass KV cache.

    Returns:
        ``(response_json, wall_seconds)``.
    """
    t0 = time.time()
    resp = client.post(
        f"{service_url}/infer",
        json={
            "doc_ids": doc_ids,
            "task": task,
            "use_kv_cache": use_kv_cache,
            "trace_cache": True,
            "gen_params": {"max_tokens": 80, "temperature": 0.0, "stop": ["\n\n"]},
        },
        timeout=600.0,
    )
    elapsed = time.time() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"infer failed: {resp.status_code} {resp.text}")
    return resp.json(), elapsed


def _run_baseline(args: argparse.Namespace) -> None:
    """Upload two documents and compare KV-cached vs non-cached inference.

    Args:
        args: parsed CLI arguments.
    """
    import httpx

    client = httpx.Client(base_url=args.service_url, timeout=600.0)

    print(f"=== DaseR Baseline Measurement ({args.samples} samples) ===")
    print(f"  service: {args.service_url}")

    # Health check
    try:
        health = client.get("/health").json()
        print(f"  health:  {json.dumps(health)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  health:  UNREACHABLE ({exc})")
        return

    # Upload documents
    print("\n--- Upload ---")
    doc_a = _upload(
        client,
        args.service_url,
        "Document A",
        "DaseR is a RAG-native KV cache service for large language model "
        "inference. It integrates with vLLM through KVConnectorBase_V1 and "
        "stores attention KV tensors on NVMe using NVIDIA cuFile (GDS) or "
        "io_uring as a fallback. The NVMe ring buffer uses fixed-size slots "
        "and chunk metadata for cache management.",
    )
    doc_b = _upload(
        client,
        args.service_url,
        "Document B",
        "The DaseR HTTP RAG API handles document upload, listing, and "
        "inference. Upload tokenizes text, creates block-aligned chunks, "
        "prewarms each chunk through vLLM, commits chunk_keys in ServerCore, "
        "and reuses existing ChunkMeta entries for duplicate documents.",
    )
    print(f"  doc_a: {doc_a['doc_id'][:8]}... ({doc_a['chunk_count']} chunks)")
    print(f"  doc_b: {doc_b['doc_id'][:8]}... ({doc_b['chunk_count']} chunks)")

    task = (
        "Write exactly two short sentences. "
        "Sentence 1 must start with DaseR and summarize the cache service. "
        "Sentence 2 must start with The HTTP RAG API and summarize the "
        "upload/inference layer."
    )

    # Warm-up: one no-KV request to ensure vLLM is primed
    print("\n--- Warm-up ---")
    _infer(
        client,
        args.service_url,
        [doc_a["doc_id"], doc_b["doc_id"]],
        task,
        use_kv_cache=False,
    )
    print("  warm-up complete")

    # Collect samples
    print(f"\n--- Sampling ({args.samples} rounds) ---")
    kv_load_ttfts: list[float] = []
    kv_load_latencies: list[float] = []
    kv_load_hits: list[int] = []
    no_kv_ttfts: list[float] = []
    no_kv_latencies: list[float] = []

    for i in range(args.samples):
        # With KV cache
        result, wall = _infer(
            client,
            args.service_url,
            [doc_a["doc_id"], doc_b["doc_id"]],
            task,
            use_kv_cache=True,
        )
        kv_load_ttfts.append(float(result.get("ttft_ms", 0.0)))
        kv_load_latencies.append(float(result.get("latency_ms", 0.0)))
        kv_load_hits.append(len(result.get("cache_hits", [])))

        # Without KV cache
        result, wall = _infer(
            client,
            args.service_url,
            [doc_a["doc_id"], doc_b["doc_id"]],
            task,
            use_kv_cache=False,
        )
        no_kv_ttfts.append(float(result.get("ttft_ms", 0.0)))
        no_kv_latencies.append(float(result.get("latency_ms", 0.0)))

        if (i + 1) % 5 == 0:
            print(f"  ... {i + 1}/{args.samples}")

    # Report
    print("\n=== Results (ms) ===")
    _print_row("KV-load TTFT", kv_load_ttfts)
    _print_row("KV-load latency", kv_load_latencies)
    _print_row("No-KV TTFT", no_kv_ttfts)
    _print_row("No-KV latency", no_kv_latencies)

    avg_hits = statistics.mean(kv_load_hits) if kv_load_hits else 0.0
    print(f"\n  Avg cache hits (KV-load): {avg_hits:.1f}")
    if kv_load_hits:
        hit_pct = sum(1 for h in kv_load_hits if h > 0) / len(kv_load_hits) * 100
        print(f"  Hit rate: {hit_pct:.1f}%")
        ttft_speedup = (
            statistics.mean(no_kv_ttfts) / statistics.mean(kv_load_ttfts)
            if kv_load_ttfts
            else 0.0
        )
        print(f"  TTFT speedup: {ttft_speedup:.2f}x")


def _print_row(label: str, values: list[float]) -> None:
    """Print a statistics row.

    Args:
        label: metric name.
        values: list of measured values in milliseconds.
    """
    if not values:
        print(f"  {label:20s}  (no data)")
        return
    sorted_vals = sorted(values)
    p50 = sorted_vals[len(sorted_vals) // 2]
    p99 = sorted_vals[min(len(sorted_vals) - 1, (len(sorted_vals) * 99) // 100)]
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"  {label:20s}  avg={avg:7.1f}  p50={p50:7.1f}  p99={p99:7.1f}  std={std:6.1f}"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:2026")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="number of inference rounds (default: 20)",
    )
    args = parser.parse_args()
    _run_baseline(args)


if __name__ == "__main__":
    main()
