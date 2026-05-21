# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo of the DaseR HTTP RAG API.

Flow:
    1. Upload two short documents via POST /documents.
    2. List docs via GET /documents.
    3. Fetch one doc via GET /documents/{doc_id}.
    4. Run inference against both docs via POST /infer.
    5. Delete the first doc.
    6. Confirm delete with GET /documents.

Use ``--benchmark`` for issue #35-style TTFT comparison (see README.md).

Prereqs: see README.md in this directory. Both ``vllm serve`` and
``python -m daser.server`` must be running first; this script drives
the public HTTP API only.
"""

# Standard
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

# Third Party
import httpx


def _load_metrics_module() -> Any:
    """Load ``metrics.py`` from this directory without package installation.

    Returns:
        Loaded metrics module.

    Async/thread-safety:
        Called once at import time in the demo process only.
    """
    metrics_path = Path(__file__).resolve().parent / "metrics.py"
    spec = importlib.util.spec_from_file_location("service_demo_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load metrics from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()
contiguous_covered_tokens = _metrics.contiguous_covered_tokens
summarize_cache_hits = _metrics.summarize_cache_hits
trial_stats = _metrics.trial_stats

DOC_A = (
    "DaseR is a RAG-native KV cache service for large language model inference. "
    "It integrates with vLLM through the KVConnectorBase_V1 interface and stores "
    "attention KV tensors directly onto NVMe storage using NVIDIA cuFile (GDS), "
    "or io_uring as a compatibility fallback. The ring buffer on NVMe is "
    "organised as a sequence of fixed-size slots; each chunk occupies a "
    "contiguous range of slots and is persisted together with its metadata. "
    "Ring buffer eviction happens transparently to the RAG API: when a "
    "chunk is overwritten, the DocRegistry on the control plane flips its "
    "cached_mask bit so the document list can still report the doc's chunks as "
    "partially or wholly evicted without losing the document entry."
)

DOC_B = (
    "The DaseR HTTP RAG API sits above the control plane. It handles document "
    "upload, listing and inference requests from end users. When a document is "
    "uploaded, the API tokenises the text, splits the token sequence into "
    "block-aligned chunks, and runs each chunk through vLLM with a one-token "
    "completion. That forward pass is enough to make the DaserConnector save "
    "the chunk's KV to DaseR. After every chunk is committed, ServerCore "
    "binds the new doc_id to the chunk_keys. "
    "Because chunks are keyed by the hash of their tokens, re-uploading the "
    "same document merely adds the new doc_id to the existing ChunkMeta "
    "doc_ids list, avoiding any duplicate KV write."
)

TTFT_GEN_PARAMS: dict[str, Any] = {
    "max_tokens": 1,
    "temperature": 0.0,
}
E2E_GEN_PARAMS: dict[str, Any] = {
    "max_tokens": 80,
    "temperature": 0.0,
    "stop": ["\n\n"],
}


def _j(resp: httpx.Response) -> dict[str, Any]:
    """Return parsed JSON and surface errors clearly.

    Args:
        resp: HTTP response from the DaseR service.

    Returns:
        Parsed JSON body.

    Raises:
        SystemExit: when the status code is 4xx/5xx.
    """
    if resp.status_code >= 400:
        sys.exit(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def _print_cache_hits(hits: list[dict[str, Any]]) -> None:
    """Print cache hit details returned by trace_cache.

    Args:
        hits: ``cache_hits`` from ``POST /infer``.
    """
    if not hits:
        print("cache_hits: []")
        return
    print("cache_hits:")
    for hit in hits:
        print(
            "  - key={key} target={target} tokens={tokens} slots={slots} "
            "pos_offset={offset}".format(
                key=str(hit.get("chunk_key", ""))[:8],
                target=hit.get("target_token_start"),
                tokens=hit.get("token_count"),
                slots=hit.get("num_slots"),
                offset=hit.get("pos_offset"),
            )
        )


def _upload_docs(client: httpx.Client) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upload the demo documents to one service.

    Args:
        client: HTTP client targeting one DaseR server.

    Returns:
        Tuple of upload response dicts for doc A and doc B.
    """
    doc_a = _j(
        client.post("/documents", json={"title": "DaseR overview", "text": DOC_A})
    )
    doc_b = _j(
        client.post("/documents", json={"title": "Service layer", "text": DOC_B})
    )
    return doc_a, doc_b


def _infer(
    client: httpx.Client,
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
    task: str,
    *,
    trace_cache: bool,
    gen_params: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Run inference over docA/docB and return response plus wall time.

    Args:
        client: HTTP client for the DaseR server.
        doc_a: upload response for document A.
        doc_b: upload response for document B.
        task: user task text.
        trace_cache: when True, request ``cache_hits`` in the response.
        gen_params: OpenAI-style generation parameters for vLLM.

    Returns:
        Tuple of infer JSON body and client wall time in milliseconds.
    """
    t0 = time.time()
    result = _j(
        client.post(
            "/infer",
            json={
                "doc_ids": [doc_a["doc_id"], doc_b["doc_id"]],
                "task": task,
                "trace_cache": trace_cache,
                "gen_params": gen_params,
            },
        )
    )
    return result, (time.time() - t0) * 1000


def _build_infer_record(
    result: dict[str, Any],
    wall_ms: float,
    *,
    block_tokens: int,
    num_layers: Optional[int],
) -> dict[str, Any]:
    """Build a JSON-serializable infer measurement record.

    Args:
        result: ``POST /infer`` response body.
        wall_ms: client-side wall time in milliseconds.
        block_tokens: block size for cache summaries.
        num_layers: optional layer count for load estimates.

    Returns:
        Measurement dict including HTTP and cache summaries.
    """
    hits = result.get("cache_hits") or []
    prompt_tokens = int(result.get("prompt_tokens", 0))
    record: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(result.get("completion_tokens", 0)),
        "server_latency_ms": float(result.get("latency_ms", 0.0)),
        "client_wall_ms": wall_ms,
        "contiguous_hit_tokens": contiguous_covered_tokens(hits, prompt_tokens),
    }
    if hits:
        record["cache_summary"] = summarize_cache_hits(
            hits,
            block_tokens=block_tokens,
            num_layers=num_layers,
        )
    return record


def _print_infer_result(
    label: str,
    result: dict[str, Any],
    wall_ms: float,
    *,
    block_tokens: int,
    num_layers: Optional[int],
) -> None:
    """Print one inference result in a comparable format.

    Args:
        label: section label for stdout.
        result: infer response body.
        wall_ms: client wall time in milliseconds.
        block_tokens: block size for summaries.
        num_layers: optional layer count for estimates.
    """
    print(f"\n==> {label}")
    print("answer:")
    print(result.get("text", ""))
    print(
        "metrics: prompt_tokens={prompt} completion_tokens={completion} "
        "server_latency_ms={latency:.1f} client_wall_ms={wall:.1f}".format(
            prompt=int(result.get("prompt_tokens", 0)),
            completion=int(result.get("completion_tokens", 0)),
            latency=float(result.get("latency_ms", 0.0)),
            wall=wall_ms,
        )
    )
    hits = result.get("cache_hits")
    if hits is not None:
        _print_cache_hits(hits)
        summary = summarize_cache_hits(
            hits, block_tokens=block_tokens, num_layers=num_layers
        )
        covered = contiguous_covered_tokens(hits, int(result.get("prompt_tokens", 0)))
        print(
            "cache_summary: hits={hits} reused_tokens={tokens} "
            "contiguous_prefix_tokens={covered} gds_reads~={reads}".format(
                hits=summary["hit_count"],
                tokens=summary["reused_token_sum"],
                covered=covered,
                reads=summary["estimated_gds_reads"],
            )
        )
        if "estimated_rope_block_ops" in summary:
            print(
                "connector_estimates: layer_copies~={copies} rope_ops~={rope}".format(
                    copies=summary.get("estimated_layer_index_copies"),
                    rope=summary["estimated_rope_block_ops"],
                )
            )


def _run_compare(args: argparse.Namespace) -> None:
    """Run baseline and chunk-reuse services and print answer differences.

    Args:
        args: parsed CLI arguments.
    """
    baseline = httpx.Client(base_url=args.baseline_url, timeout=600.0)
    chunk_reuse = httpx.Client(base_url=args.chunk_reuse_url, timeout=600.0)
    gen_params = TTFT_GEN_PARAMS if args.ttft_only else E2E_GEN_PARAMS
    num_layers = args.num_layers

    print("==> baseline health")
    print(json.dumps(_j(baseline.get("/health")), indent=2))
    print("\n==> chunk-reuse health")
    print(json.dumps(_j(chunk_reuse.get("/health")), indent=2))

    print("\n==> upload docs to baseline")
    base_a, base_b = _upload_docs(baseline)
    print(json.dumps({"doc_a": base_a, "doc_b": base_b}, indent=2))

    print("\n==> upload docs to chunk-reuse")
    reuse_a, reuse_b = _upload_docs(chunk_reuse)
    print(json.dumps({"doc_a": reuse_a, "doc_b": reuse_b}, indent=2))

    baseline_result, baseline_wall = _infer(
        baseline,
        base_a,
        base_b,
        args.task,
        trace_cache=True,
        gen_params=gen_params,
    )
    reuse_result, reuse_wall = _infer(
        chunk_reuse,
        reuse_a,
        reuse_b,
        args.task,
        trace_cache=True,
        gen_params=gen_params,
    )

    _print_infer_result(
        "BASELINE",
        baseline_result,
        baseline_wall,
        block_tokens=args.block_tokens,
        num_layers=num_layers,
    )
    _print_infer_result(
        "CHUNK_REUSE",
        reuse_result,
        reuse_wall,
        block_tokens=args.block_tokens,
        num_layers=num_layers,
    )


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run repeated TTFT-oriented measurements and optional JSON export.

    Args:
        args: parsed CLI arguments.
    """
    baseline = httpx.Client(base_url=args.baseline_url, timeout=600.0)
    chunk_reuse = httpx.Client(base_url=args.chunk_reuse_url, timeout=600.0)
    gen_params = TTFT_GEN_PARAMS if args.ttft_only else E2E_GEN_PARAMS
    num_layers = args.num_layers

    print("==> benchmark upload (baseline)")
    base_a, base_b = _upload_docs(baseline)
    print("==> benchmark upload (chunk-reuse)")
    reuse_a, reuse_b = _upload_docs(chunk_reuse)

    trials: list[dict[str, Any]] = []
    baseline_server_ms: list[float] = []
    reuse_server_ms: list[float] = []

    for trial in range(args.trials):
        print(f"\n==> trial {trial + 1}/{args.trials}")
        b_result, b_wall = _infer(
            baseline,
            base_a,
            base_b,
            args.task,
            trace_cache=True,
            gen_params=gen_params,
        )
        r_result, r_wall = _infer(
            chunk_reuse,
            reuse_a,
            reuse_b,
            args.task,
            trace_cache=True,
            gen_params=gen_params,
        )
        b_record = _build_infer_record(
            b_result, b_wall, block_tokens=args.block_tokens, num_layers=num_layers
        )
        r_record = _build_infer_record(
            r_result, r_wall, block_tokens=args.block_tokens, num_layers=num_layers
        )
        baseline_server_ms.append(b_record["server_latency_ms"])
        reuse_server_ms.append(r_record["server_latency_ms"])
        trials.append(
            {
                "trial": trial,
                "baseline": b_record,
                "chunk_reuse": r_record,
            }
        )
        delta = r_record["server_latency_ms"] - b_record["server_latency_ms"]
        print(
            "  baseline server_latency_ms={b:.1f} "
            "chunk_reuse={r:.1f} delta={d:+.1f}".format(
                b=b_record["server_latency_ms"],
                r=r_record["server_latency_ms"],
                d=delta,
            )
        )

    b_stats = trial_stats(baseline_server_ms)
    r_stats = trial_stats(reuse_server_ms)
    summary = {
        "baseline_server_latency_ms": b_stats,
        "chunk_reuse_server_latency_ms": r_stats,
        "delta_median_ms": r_stats["median_ms"] - b_stats["median_ms"],
        "delta_mean_ms": r_stats["mean_ms"] - b_stats["mean_ms"],
    }

    print("\n==> benchmark summary (server_latency_ms)")
    print(json.dumps(summary, indent=2))

    if args.json_out:
        payload = {
            "measurement": "ttft" if args.ttft_only else "e2e",
            "config": {
                "baseline_url": args.baseline_url,
                "chunk_reuse_url": args.chunk_reuse_url,
                "trials": args.trials,
                "block_tokens": args.block_tokens,
                "num_layers": num_layers,
                "gen_params": gen_params,
            },
            "upload": {
                "baseline": {"doc_a": base_a, "doc_b": base_b},
                "chunk_reuse": {"doc_a": reuse_a, "doc_b": reuse_b},
            },
            "trials": trials,
            "summary": summary,
            "notes": {
                "server_latency_ms": (
                    "HTTP /infer wraps vllm.completion; with max_tokens=1 this "
                    "approximates TTFT plus one decode step."
                ),
                "cache_hits": (
                    "Control-plane lookup only; correlate connector logs for "
                    "GDS reads and GPU copies."
                ),
            },
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n==> wrote {out_path}")


def main() -> None:
    """CLI entry point for the service demo and benchmark modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8080")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8080")
    parser.add_argument("--chunk-reuse-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run one infer on baseline (prefix) and chunk-reuse servers.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run repeated infer trials and print TTFT/E2E summary stats.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of infer trials in --benchmark mode (default: 5).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Write benchmark payload to this JSON file.",
    )
    parser.add_argument(
        "--ttft-only",
        action="store_true",
        help="Use max_tokens=1 (default in --benchmark).",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Use max_tokens=80 with stop; disables default TTFT gen params.",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=16,
        help="Block size for cache hit summaries (default: 16).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Model layer count for connector load estimates (e.g. 36).",
    )
    parser.add_argument(
        "--task",
        default=(
            "Write exactly two short sentences. Sentence 1 must start with DaseR "
            "and summarize the cache service. Sentence 2 must start with The "
            "HTTP RAG API and summarize the upload/inference layer."
        ),
    )
    args = parser.parse_args()

    if args.benchmark and args.compare_baseline:
        sys.exit("Use either --benchmark or --compare-baseline, not both.")
    if args.benchmark:
        if not args.e2e:
            args.ttft_only = True
        _run_benchmark(args)
        return
    if args.compare_baseline:
        if not args.e2e and not args.ttft_only:
            args.ttft_only = True
        _run_compare(args)
        return

    client = httpx.Client(base_url=args.service_url, timeout=600.0)
    gen_params = TTFT_GEN_PARAMS if args.ttft_only else E2E_GEN_PARAMS

    print("==> health")
    print(json.dumps(_j(client.get("/health")), indent=2))

    print("\n==> upload doc A")
    doc_a, doc_b = _upload_docs(client)
    print(json.dumps(doc_a, indent=2))

    print("\n==> upload doc B")
    print(json.dumps(doc_b, indent=2))

    print("\n==> list docs")
    print(json.dumps(_j(client.get("/documents")), indent=2))

    print("\n==> get doc A")
    print(json.dumps(_j(client.get(f"/documents/{doc_a['doc_id']}")), indent=2))

    print("\n==> infer over both docs")
    inf, wall_ms = _infer(
        client,
        doc_a,
        doc_b,
        args.task,
        trace_cache=True,
        gen_params=gen_params,
    )
    print(json.dumps(inf, indent=2))
    print(f"(wall clock: {wall_ms:.1f} ms)")
    hits = inf.get("cache_hits") or []
    if hits:
        print(
            json.dumps(
                summarize_cache_hits(
                    hits, block_tokens=args.block_tokens, num_layers=args.num_layers
                ),
                indent=2,
            )
        )

    print("\n==> delete doc A")
    print(json.dumps(_j(client.delete(f"/documents/{doc_a['doc_id']}")), indent=2))

    print("\n==> list docs (after delete)")
    print(json.dumps(_j(client.get("/documents")), indent=2))


if __name__ == "__main__":
    main()
