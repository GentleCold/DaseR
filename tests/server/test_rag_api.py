# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
from fastapi.testclient import TestClient

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.metadata_store import MetadataStore
from daser.server.rag_api import RAGAPIConfig, build_rag_api

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def make_core() -> ServerCore:
    """Create a ServerCore for RAG API tests."""
    store = MetadataStore(total_slots=64)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=64,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


class FakeTokenizer:
    """Simple deterministic tokenizer for RAG API tests."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        return {"input_ids": [ord(ch) for ch in text]}


class FakeVLLMClient:
    """Fake vLLM client that records calls."""

    def __init__(self, fail_prefill: bool = False) -> None:
        self.fail_prefill = fail_prefill
        self.prefills: list[list[int]] = []
        self.completions: list[tuple[list[int], dict[str, Any] | None]] = []

    async def close(self) -> None:
        """Close fake client."""

    async def health(self) -> bool:
        """Return healthy state."""
        return True

    async def prefill(self, tokens: list[int]) -> None:
        """Record a prefill call."""
        if self.fail_prefill:
            raise RuntimeError("prefill failed")
        self.prefills.append(list(tokens))

    async def completion(
        self,
        tokens: list[int],
        gen_params: dict[str, Any] | None = None,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a completion call and return OpenAI-style data."""
        self.completions.append((list(tokens), gen_params, kv_transfer_params))
        return {
            "choices": [{"text": "answer"}],
            "usage": {"completion_tokens": 3},
        }


def _make_client(
    vllm: FakeVLLMClient | None = None,
) -> tuple[TestClient, FakeVLLMClient]:
    core = make_core()
    fake_vllm = vllm or FakeVLLMClient()
    app = build_rag_api(
        RAGAPIConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
            chunk_blocks=1,
            system_prompt="S:",
            doc_separator="|",
            task_separator="? ",
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    return TestClient(app), fake_vllm


def test_upload_document_prefills_and_registers() -> None:
    client, vllm = _make_client()

    resp = client.post("/documents", json={"title": "doc", "text": "abcdefgh"})

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] == 2
    assert body["chunk_count_cached"] == 0
    assert vllm.prefills == [[97, 98, 99, 100], [101, 102, 103, 104]]

    docs = client.get("/documents").json()
    assert len(docs) == 1
    assert docs[0]["title"] == "doc"


def test_upload_prefill_failure_does_not_register_document() -> None:
    client, _ = _make_client(FakeVLLMClient(fail_prefill=True))

    resp = client.post("/documents", json={"title": "doc", "text": "abcd"})

    assert resp.status_code == 502
    assert client.get("/documents").json() == []


def test_document_get_and_delete() -> None:
    client, _ = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]

    doc = client.get(f"/documents/{doc_id}")
    assert doc.status_code == 200
    assert "tokens" not in doc.json()

    deleted = client.delete(f"/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_infer_rebuilds_prompt_and_forwards_gen_params() -> None:
    client, vllm = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]

    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_id],
            "task": "go",
            "gen_params": {"max_tokens": 7},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "answer"
    assert vllm.completions == [
        (
            [83, 58, 97, 98, 99, 100, 63, 32, 103, 111],
            {"max_tokens": 7},
            {"daser_skip_save": True},
        )
    ]
