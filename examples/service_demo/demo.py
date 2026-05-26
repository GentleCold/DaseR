# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo of the DaseR HTTP RAG API.

Flow:
    1. Upload two short documents via POST /documents.
    2. List docs via GET /documents.
    3. Fetch one doc via GET /documents/{doc_id}.
    4. Run inference against both docs via POST /infer.
    5. Delete the first doc.
    6. Confirm delete with GET /documents.

Prereqs: see README.md in this directory. Both ``vllm serve`` and
``python -m daser.server`` must be running first; this script drives
the public HTTP API only.
"""

# Standard
import argparse
import json
import sys
import time

# Third Party
import httpx

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


def _j(resp: httpx.Response) -> dict:
    """Return parsed JSON and surface errors clearly."""
    if resp.status_code >= 400:
        sys.exit(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def _print_cache_hits(hits: list[dict]) -> None:
    """Print cache hit details returned by trace_cache."""
    if not hits:
        print("cache_hits: []")
        return
    print("cache_hits:")
    for hit in hits:
        print(
            "  - key={key} target={target} tokens={tokens} pos_offset={offset}".format(
                key=str(hit.get("chunk_key", ""))[:8],
                target=hit.get("target_token_start"),
                tokens=hit.get("token_count"),
                offset=hit.get("pos_offset"),
            )
        )


def _upload_docs(client: httpx.Client) -> tuple[dict, dict]:
    """Upload the demo documents to one service."""
    doc_a = _j(
        client.post("/documents", json={"title": "DaseR overview", "text": DOC_A})
    )
    doc_b = _j(
        client.post("/documents", json={"title": "Service layer", "text": DOC_B})
    )
    return doc_a, doc_b


def _infer(
    client: httpx.Client,
    doc_a: dict,
    doc_b: dict,
    task: str,
    trace_cache: bool,
) -> tuple[dict, float]:
    """Run inference over docA/docB and return response plus wall time."""
    t0 = time.time()
    result = _j(
        client.post(
            "/infer",
            json={
                "doc_ids": [doc_a["doc_id"], doc_b["doc_id"]],
                "task": task,
                "trace_cache": trace_cache,
                "gen_params": {"max_tokens": 80, "temperature": 0.0, "stop": ["\n\n"]},
            },
        )
    )
    return result, (time.time() - t0) * 1000


def _print_infer_result(label: str, result: dict, wall_ms: float) -> None:
    """Print one inference result in a comparable format."""
    print(f"\n==> {label}")
    print("answer:")
    print(result.get("text", ""))
    print(
        "metrics: prompt_tokens={prompt} completion_tokens={completion} "
        "latency_ms={latency:.1f} wall_ms={wall:.1f}".format(
            prompt=result.get("prompt_tokens", 0),
            completion=result.get("completion_tokens", 0),
            latency=float(result.get("latency_ms", 0.0)),
            wall=wall_ms,
        )
    )
    if "cache_hits" in result:
        _print_cache_hits(result["cache_hits"])


def _run_compare(args: argparse.Namespace) -> None:
    """Run baseline and chunk-reuse services and print answer differences."""
    baseline = httpx.Client(base_url=args.baseline_url, timeout=600.0)
    chunk_reuse = httpx.Client(base_url=args.chunk_reuse_url, timeout=600.0)

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
        baseline, base_a, base_b, args.task, trace_cache=False
    )
    reuse_result, reuse_wall = _infer(
        chunk_reuse, reuse_a, reuse_b, args.task, trace_cache=True
    )

    _print_infer_result("BASELINE", baseline_result, baseline_wall)
    _print_infer_result("CHUNK_REUSE", reuse_result, reuse_wall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:2026")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:2026")
    parser.add_argument("--chunk-reuse-url", default="http://127.0.0.1:8081")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument(
        "--task",
        default=(
            "Write exactly two short sentences. Sentence 1 must start with DaseR "
            "and summarize the cache service. Sentence 2 must start with The "
            "HTTP RAG API and summarize the upload/inference layer."
        ),
    )
    args = parser.parse_args()

    if args.compare_baseline:
        _run_compare(args)
        return

    client = httpx.Client(base_url=args.service_url, timeout=600.0)

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
    inf, wall_ms = _infer(client, doc_a, doc_b, args.task, trace_cache=True)
    print(json.dumps(inf, indent=2))
    print(f"(wall clock: {wall_ms:.1f} ms)")

    print("\n==> delete doc A")
    print(json.dumps(_j(client.delete(f"/documents/{doc_a['doc_id']}")), indent=2))

    print("\n==> list docs (after delete)")
    print(json.dumps(_j(client.get("/documents")), indent=2))


if __name__ == "__main__":
    main()
