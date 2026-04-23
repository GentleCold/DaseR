# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo of the DaseR service layer.

Flow:
    1. Upload two short documents via POST /documents.
    2. List docs via GET /documents.
    3. Fetch one doc via GET /documents/{doc_id}.
    4. Run inference against both docs via POST /infer.
    5. Delete the first doc.
    6. Confirm delete with GET /documents.

Prereqs: see README.md in this directory. Both ``vllm serve`` and
``python -m daser.service`` must be running first; this script drives
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
    "Ring buffer eviction happens transparently to the service layer: when a "
    "chunk is overwritten, the DocRegistry on the control plane flips its "
    "cached_mask bit so list_docs can still report the doc's chunks as "
    "partially or wholly evicted without losing the document entry."
)

DOC_B = (
    "The DaseR service layer sits above the control plane. It handles document "
    "upload, listing and inference requests from end users. When a document is "
    "uploaded, the service tokenises the text, splits the token sequence into "
    "block-aligned chunks, and runs each chunk through vLLM with a one-token "
    "completion. That forward pass is enough to make the DaserConnector save "
    "the chunk's KV to DaseR. After every chunk is committed, the service "
    "sends a register_doc RPC that binds the new doc_id to the chunk_keys. "
    "Because chunks are keyed by the hash of their tokens, re-uploading the "
    "same document merely adds the new doc_id to the existing ChunkMeta "
    "doc_ids list, avoiding any duplicate KV write."
)


def _j(resp: httpx.Response) -> dict:
    """Return parsed JSON and surface errors clearly."""
    if resp.status_code >= 400:
        sys.exit(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8080")
    parser.add_argument("--task", default="Summarize both documents in two sentences.")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.service_url, timeout=600.0)

    print("==> health")
    print(json.dumps(_j(client.get("/health")), indent=2))

    print("\n==> upload doc A")
    doc_a = _j(
        client.post("/documents", json={"title": "DaseR overview", "text": DOC_A})
    )
    print(json.dumps(doc_a, indent=2))

    print("\n==> upload doc B")
    doc_b = _j(
        client.post("/documents", json={"title": "Service layer", "text": DOC_B})
    )
    print(json.dumps(doc_b, indent=2))

    print("\n==> list docs")
    print(json.dumps(_j(client.get("/documents")), indent=2))

    print("\n==> get doc A")
    print(json.dumps(_j(client.get(f"/documents/{doc_a['doc_id']}")), indent=2))

    print("\n==> infer over both docs")
    t0 = time.time()
    inf = _j(
        client.post(
            "/infer",
            json={
                "doc_ids": [doc_a["doc_id"], doc_b["doc_id"]],
                "task": args.task,
                "gen_params": {"max_tokens": 128, "temperature": 0.0},
            },
        )
    )
    print(json.dumps(inf, indent=2))
    print(f"(wall clock: {(time.time() - t0) * 1000:.1f} ms)")

    print("\n==> delete doc A")
    print(json.dumps(_j(client.delete(f"/documents/{doc_a['doc_id']}")), indent=2))

    print("\n==> list docs (after delete)")
    print(json.dumps(_j(client.get("/documents")), indent=2))


if __name__ == "__main__":
    main()
