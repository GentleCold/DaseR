# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from typing import Any
import warnings

# Third Party
from fastapi.testclient import TestClient

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.http import HTTPServerConfig, build_http_app
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def make_core() -> ServerCore:
    """Create a ServerCore for HTTP server tests."""
    return make_core_with_index(PrefixHashIndex(block_tokens=BLOCK_TOKENS))


def make_core_with_index(retrieval_index: Any) -> ServerCore:
    """Create a ServerCore with a custom retrieval index."""
    store = MetadataStore(total_slots=64)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=64,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=retrieval_index,
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


class FakeTokenizer:
    """Simple deterministic tokenizer for RAG API tests."""

    pad_token_id = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        return {"input_ids": [ord(ch) for ch in text]}


class FakeVLLMClient:
    """Fake vLLM client that records calls."""

    def __init__(
        self,
        fail_prefill: bool = False,
        commit_core: ServerCore | None = None,
        model_id: str = "m",
    ) -> None:
        self.fail_prefill = fail_prefill
        self.commit_core = commit_core
        self.model_id = model_id
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
        if self.commit_core is not None:
            key = _hash_tokens(tokens)
            await self.commit_core.alloc_chunk(
                key,
                token_count=len(tokens),
                model_id=self.model_id,
            )
            await self.commit_core.commit_chunk(key)

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
) -> tuple[TestClient, FakeVLLMClient, ServerCore]:
    core = make_core()
    fake_vllm = vllm or FakeVLLMClient()
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
            system_prompt="S:",
            doc_separator="|",
            task_separator="? ",
            answer_separator="! ",
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    return TestClient(app), fake_vllm, core


def test_build_http_app_uses_non_deprecated_lifespan() -> None:
    """Constructing the app should not use FastAPI's deprecated on_event API."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _make_client()

    messages = [str(warning.message) for warning in caught]
    assert not any("on_event is deprecated" in message for message in messages)


def test_upload_document_prefills_and_registers() -> None:
    client, vllm, _ = _make_client()

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
    client, _, _ = _make_client(FakeVLLMClient(fail_prefill=True))

    resp = client.post("/documents", json={"title": "doc", "text": "abcd"})

    assert resp.status_code == 502
    assert client.get("/documents").json() == []


def test_document_get_and_delete() -> None:
    client, _, _ = _make_client()
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
    client, vllm, _ = _make_client()
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
            [83, 58, 97, 98, 99, 100, 63, 32, 103, 111, 33, 32],
            {"max_tokens": 7},
            {"daser_skip_save": True},
        )
    ]


def test_prefix_mode_infer_keeps_document_tail_tokens() -> None:
    client, vllm, _ = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcde"}).json()[
        "doc_id"
    ]

    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_id],
            "task": "go",
        },
    )

    assert resp.status_code == 200
    assert vllm.prefills == [[97, 98, 99, 100]]
    assert vllm.completions[0][0] == [
        83,
        58,
        97,
        98,
        99,
        100,
        101,
        63,
        32,
        103,
        111,
        33,
        32,
    ]


def test_infer_trace_cache_returns_lookup_hits() -> None:
    client, _, core = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]
    prompt_prefix = [83, 58, 97, 98, 99, 100]
    key = _hash_tokens(prompt_prefix[:4])

    asyncio.get_event_loop().run_until_complete(
        core.alloc_chunk(key, token_count=4, model_id="m")
    )
    asyncio.get_event_loop().run_until_complete(core.commit_chunk(key))

    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_id],
            "task": "go",
            "trace_cache": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hits"][0]["chunk_key"] == key
    assert body["cache_hits"][0]["target_token_start"] == 0


def test_chunk_reuse_infer_uses_contiguous_prewarmed_padded_segments() -> None:
    core = make_core_with_index(ChunkReuseIndex(block_tokens=BLOCK_TOKENS))
    fake_vllm = FakeVLLMClient(commit_core=core)
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=BLOCK_TOKENS,
            system_prompt="S:",
            doc_separator="|",
            task_separator="? ",
            answer_separator="! ",
            align_document_chunks=True,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    client = TestClient(app)

    doc_a = client.post("/documents", json={"title": "a", "text": "abcde"}).json()
    doc_b = client.post("/documents", json={"title": "b", "text": "fghi"}).json()
    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_a["doc_id"], doc_b["doc_id"]],
            "task": "go",
            "trace_cache": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    system_key = _hash_tokens([83, 58, 32, 32])
    doc_a_key = _hash_tokens([97, 98, 99, 100, 101, 32, 32, 32])
    sep_key = _hash_tokens([124, 32, 32, 32])
    doc_b_key = _hash_tokens([102, 103, 104, 105])
    assert [hit["chunk_key"] for hit in body["cache_hits"]] == [
        system_key,
        doc_a_key,
        sep_key,
        doc_b_key,
    ]
    assert [hit["target_token_start"] for hit in body["cache_hits"]] == [
        0,
        4,
        12,
        16,
    ]
    assert fake_vllm.prefills == [
        [97, 98, 99, 100, 101, 32, 32, 32],
        [102, 103, 104, 105],
        [83, 58, 32, 32],
        [124, 32, 32, 32],
    ]
    assert fake_vllm.completions[0][0] == [
        83,
        58,
        32,
        32,
        97,
        98,
        99,
        100,
        101,
        32,
        32,
        32,
        124,
        32,
        32,
        32,
        102,
        103,
        104,
        105,
        63,
        32,
        103,
        111,
        33,
        32,
    ]
    assert fake_vllm.completions[0][2] == {"daser_skip_save": True}


def test_chunk_reuse_padding_prefers_tokenizer_pad_token_id() -> None:
    class PadTokenizer(FakeTokenizer):
        pad_token_id = 0

    core = make_core_with_index(ChunkReuseIndex(block_tokens=BLOCK_TOKENS))
    fake_vllm = FakeVLLMClient(commit_core=core)
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=BLOCK_TOKENS,
            system_prompt="S:",
            doc_separator="|",
            task_separator="? ",
            answer_separator="! ",
            align_document_chunks=True,
        ),
        core,
        tokenizer=PadTokenizer(),
        vllm=fake_vllm,
    )
    client = TestClient(app)

    resp = client.post("/documents", json={"title": "doc", "text": "abcde"})

    assert resp.status_code == 201
    assert fake_vllm.prefills == [[97, 98, 99, 100, 101, 0, 0, 0]]


def test_chunk_reuse_uses_one_block_aligned_chunk_per_prompt_segment() -> None:
    core = make_core_with_index(ChunkReuseIndex(block_tokens=BLOCK_TOKENS))
    fake_vllm = FakeVLLMClient(commit_core=core)
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=BLOCK_TOKENS,
            system_prompt="S:",
            doc_separator="|",
            task_separator="? ",
            answer_separator="! ",
            align_document_chunks=True,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    client = TestClient(app)

    doc_a = client.post("/documents", json={"title": "a", "text": "abcde"}).json()
    doc_b = client.post("/documents", json={"title": "b", "text": "fghi"}).json()
    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_a["doc_id"], doc_b["doc_id"]],
            "task": "go",
            "trace_cache": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    system = [83, 58, 32, 32]
    doc_a_tokens = [97, 98, 99, 100, 101, 32, 32, 32]
    separator = [124, 32, 32, 32]
    doc_b_tokens = [102, 103, 104, 105]
    assert fake_vllm.prefills == [
        doc_a_tokens,
        doc_b_tokens,
        system,
        separator,
    ]
    assert [hit["chunk_key"] for hit in body["cache_hits"]] == [
        _hash_tokens(system),
        _hash_tokens(doc_a_tokens),
        _hash_tokens(separator),
        _hash_tokens(doc_b_tokens),
    ]
    assert [hit["target_token_start"] for hit in body["cache_hits"]] == [
        0,
        4,
        12,
        16,
    ]
    assert fake_vllm.completions[0][0] == [
        *system,
        *doc_a_tokens,
        *separator,
        *doc_b_tokens,
        63,
        32,
        103,
        111,
        33,
        32,
    ]
