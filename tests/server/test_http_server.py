# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from typing import Any
import warnings

# Third Party
from fastapi.testclient import TestClient

# First Party
from daser.connector.helpers import ROLLING_PREFIX_SEED, hash_tokens, rolling_prefix_key
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.http import HTTPServerConfig, build_http_app
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


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
    chat_template_kwargs: list[dict[str, Any]] = []

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        return {"input_ids": [ord(ch) for ch in text]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        """Return a compact deterministic chat template for tests."""
        assert tokenize is False
        self.chat_template_kwargs.append(kwargs)
        rendered = ""
        for message in messages:
            rendered += f"<|{message['role']}|>{message['content']}"
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


class FakeVLLMClient:
    """Fake vLLM client that records calls."""

    def __init__(
        self,
        fail_prefill: bool = False,
        commit_core: ServerCore | None = None,
        model_id: str = "m",
        commit_delay_s: float = 0.0,
        commit_in_background: bool = False,
    ) -> None:
        self.fail_prefill = fail_prefill
        self.commit_core = commit_core
        self.model_id = model_id
        self.commit_delay_s = commit_delay_s
        self.commit_in_background = commit_in_background
        self.prefills: list[list[int]] = []
        self.completions: list[tuple[list[int], dict[str, Any] | None]] = []
        self.completion_ttft_ms = 12.5

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
            if self.commit_in_background:
                asyncio.create_task(self._commit_tokens(tokens))
            else:
                await self._commit_tokens(tokens)

    async def _commit_tokens(self, tokens: list[int]) -> None:
        """Simulate connector allocation and commit for prefetched tokens."""
        if self.commit_core is None:
            return
        if self.commit_delay_s > 0:
            await asyncio.sleep(self.commit_delay_s)
        key = hash_tokens(tokens)
        alloc = await self.commit_core.alloc_chunk(
            key,
            token_count=len(tokens),
            model_id=self.model_id,
        )
        if alloc.skipped:
            return
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

    async def completion_with_ttft(
        self,
        tokens: list[int],
        gen_params: dict[str, Any] | None = None,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Record a completion call and return fake TTFT."""
        result = await self.completion(tokens, gen_params, kv_transfer_params)
        return result, self.completion_ttft_ms


def _make_client(
    vllm: FakeVLLMClient | None = None,
) -> tuple[TestClient, FakeVLLMClient, ServerCore]:
    core = make_core()
    fake_vllm = vllm or FakeVLLMClient(commit_core=core)
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
            system_prompt="S:",
            doc_separator="|",
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    return TestClient(app), fake_vllm, core


def test_drain_endpoint_waits_for_core_transfer_work() -> None:
    """POST /drain exposes a benchmark-safe transfer drain primitive."""
    calls = 0

    async def drain_transfer() -> None:
        nonlocal calls
        calls += 1

    core = make_core()
    fake_vllm = FakeVLLMClient(commit_core=core)
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
        drain_transfer=drain_transfer,
    )
    client = TestClient(app)

    response = client.post("/drain")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == 1


def _ids(text: str) -> list[int]:
    """Return fake-tokenizer IDs for text."""
    return [ord(ch) for ch in text]


def _chat_prefix(system_prompt: str = "S:") -> list[int]:
    """Return fake chat-template tokens before document content."""
    return _ids(f"<|system|>{system_prompt}<|user|>Documents:\n")


def _doc_separator() -> list[int]:
    """Return fake chat-template document separator tokens."""
    return _ids("\n|\n")


def _chat_suffix(task: str = "go") -> list[int]:
    """Return fake chat-template tokens after document content."""
    return _ids(f"\n\nTask: {task}<|assistant|>")


def test_build_http_app_uses_non_deprecated_lifespan() -> None:
    """Constructing the app should not use FastAPI's deprecated on_event API."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _make_client()

    messages = [str(warning.message) for warning in caught]
    assert not any("on_event is deprecated" in message for message in messages)


def test_web_ui_index_served() -> None:
    """The built-in Web UI should be served from the HTTP root."""
    client, _, _ = _make_client()

    resp = client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "DaseR 控制台" in resp.text


def test_health_does_not_expose_server_metrics() -> None:
    """Cache counters should be exposed through /metrics, not /health JSON."""
    client, _, _ = _make_client()

    resp = client.get("/health")

    assert resp.status_code == 200
    assert "metrics" not in resp.json()


def test_web_ui_static_assets_served() -> None:
    """Static UI assets should be available from the packaged app."""
    client, _, _ = _make_client()

    css = client.get("/ui/static/styles.css")
    js = client.get("/ui/static/app.js")
    logo = client.get("/ui/static/daser-icon.png")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert logo.status_code == 200
    assert logo.content.startswith(b"\x89PNG")


def test_web_ui_metrics_include_ttft() -> None:
    """The Web UI should render the TTFT value returned by /infer."""
    client, _, _ = _make_client()

    js = client.get("/ui/static/app.js")

    assert js.status_code == 200
    assert '["TTFT", formatMs(result.ttft_ms)]' in js.text


def test_web_ui_result_panel_aligns_with_inference_panel() -> None:
    """The result panel should start on the same grid row as online inference."""
    client, _, _ = _make_client()

    css = client.get("/ui/static/styles.css")

    assert css.status_code == 200
    assert (
        'grid-template-areas:\n    "upload documents"\n    "infer result";' in css.text
    )
    assert ".infer-panel {\n  grid-area: infer;\n}" in css.text
    assert ".result-panel {\n  display: grid;\n  grid-area: result;" in css.text
    assert "grid-row: span 2;" not in css.text


def test_web_ui_document_titles_stay_on_one_line() -> None:
    """Document card titles should use one line and truncate when needed."""
    client, _, _ = _make_client()

    css = client.get("/ui/static/styles.css")

    assert css.status_code == 200
    assert ".doc-title {\n  display: flex;" in css.text
    assert "  flex: 1 1 auto;" in css.text
    assert ".doc-title input {\n  width: auto;\n  flex: 0 0 auto;\n}" in css.text
    assert (
        ".doc-title span {\n"
        "  min-width: 0;\n"
        "  overflow: hidden;\n"
        "  text-overflow: ellipsis;\n"
        "  white-space: nowrap;\n"
        "}" in css.text
    )


def test_web_ui_document_preview_can_be_toggled_off() -> None:
    """Clicking the visible document again should clear the preview."""
    client, _, _ = _make_client()

    js = client.get("/ui/static/app.js")

    assert js.status_code == 200
    assert "previewDocId: null" in js.text
    assert (
        'view.textContent = state.previewDocId === doc.doc_id ? "取消查看" : "查看";'
        in js.text
    )
    assert "function resetDocumentPreview()" in js.text
    assert "if (state.previewDocId === docId) {" in js.text
    assert "resetDocumentPreview();" in js.text


def test_web_ui_cache_hit_details_stay_inside_result_panel() -> None:
    """Cache hit details should scroll within their result-panel row."""
    client, _, _ = _make_client()

    css = client.get("/ui/static/styles.css")

    assert css.status_code == 200
    assert (
        "grid-template-rows: auto auto auto minmax(180px, 1fr) "
        "minmax(140px, 220px);" in css.text
    )
    assert ".result-panel {\n  display: grid;" in css.text
    assert "  overflow: hidden;" in css.text
    assert (
        ".cache-details {\n"
        "  display: grid;\n"
        "  grid-template-rows: auto minmax(0, 1fr);\n"
        "  overflow: hidden;\n"
        "}" in css.text
    )
    assert ".cache-hits {\n  min-height: 0;" in css.text
    assert "  height: 100%;" in css.text
    assert "  overflow-y: auto;" in css.text
    assert "  overscroll-behavior: contain;" in css.text


def test_chunk_reuse_lifespan_prewarms_fixed_segments() -> None:
    """Chunk reuse should prefill fixed prompt segments before serving traffic."""
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
            align_document_chunks=True,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert fake_vllm.prefills == [
            [*_chat_prefix(), 32],
            [*_doc_separator(), 32],
        ]
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
    prefix = [*_chat_prefix(), 32]
    separator = [*_doc_separator(), 32]
    assert body["cache_hits"][0]["chunk_key"] == hash_tokens(prefix)
    assert body["cache_hits"][2]["chunk_key"] == hash_tokens(separator)
    assert fake_vllm.prefills == [
        prefix,
        separator,
        [97, 98, 99, 100, 101, 32, 32, 32],
        [102, 103, 104, 105],
    ]


def test_chunk_reuse_lifespan_skips_restored_fixed_segments() -> None:
    """Startup should not prefill fixed chat segments already in the index."""
    core = make_core_with_index(ChunkReuseIndex(block_tokens=BLOCK_TOKENS))
    fake_vllm = FakeVLLMClient(commit_core=core)
    prefix = [*_chat_prefix(), 32]
    prefix_key = hash_tokens(prefix)
    asyncio.get_event_loop().run_until_complete(
        core.alloc_chunk(prefix_key, token_count=len(prefix), model_id="m")
    )
    asyncio.get_event_loop().run_until_complete(core.commit_chunk(prefix_key))
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=BLOCK_TOKENS,
            system_prompt="S:",
            doc_separator="|",
            align_document_chunks=True,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert fake_vllm.prefills == [[*_doc_separator(), 32]]


def test_upload_document_prefills_and_registers() -> None:
    client, vllm, _ = _make_client()

    resp = client.post("/documents", json={"title": "doc", "text": "abcdefgh"})

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] == 2
    assert body["chunk_count_cached"] == 2
    assert vllm.prefills == [[97, 98, 99, 100], [101, 102, 103, 104]]

    docs = client.get("/documents").json()
    assert len(docs) == 1
    assert docs[0]["title"] == "doc"
    assert docs[0]["chunk_count_cached"] == 2


def test_upload_document_skips_prefill_for_identical_committed_chunks() -> None:
    """Uploading the same document twice should reuse committed chunks."""
    client, vllm, _ = _make_client()

    first = client.post("/documents", json={"title": "first", "text": "abcdefgh"})
    second = client.post("/documents", json={"title": "second", "text": "abcdefgh"})

    assert first.status_code == 201
    assert second.status_code == 201
    assert vllm.prefills == [[97, 98, 99, 100], [101, 102, 103, 104]]
    docs = client.get("/documents").json()
    assert len(docs) == 2
    assert docs[0]["chunk_count_cached"] == 2
    assert docs[1]["chunk_count_cached"] == 2


def test_upload_document_rejects_prefix_mode() -> None:
    core = make_core()
    fake_vllm = FakeVLLMClient()
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=BLOCK_TOKENS,
            system_prompt="S:",
            doc_separator="|",
            cache_reuse_mode="prefix",
            align_document_chunks=False,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    client = TestClient(app)

    resp = client.post("/documents", json={"title": "doc", "text": "abcd"})

    assert resp.status_code == 400
    assert resp.json()["detail"] == ("document upload requires chunk cache reuse mode")
    assert fake_vllm.prefills == []


def test_upload_document_waits_for_committed_store() -> None:
    core = make_core()
    fake_vllm = FakeVLLMClient(
        commit_core=core,
        commit_delay_s=0.05,
        commit_in_background=True,
    )
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=BLOCK_TOKENS,
            system_prompt="S:",
            doc_separator="|",
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )

    with TestClient(app) as client:
        resp = client.post("/documents", json={"title": "doc", "text": "abcd"})

    assert resp.status_code == 201
    assert resp.json()["chunk_count_cached"] == 1


def test_upload_document_times_out_when_store_never_commits(monkeypatch: Any) -> None:
    monkeypatch.setattr("daser.server.http.app._DOCUMENT_STORE_SYNC_TIMEOUT_S", 0.01)
    client, _, _ = _make_client(FakeVLLMClient())

    resp = client.post("/documents", json={"title": "doc", "text": "abcd"})

    assert resp.status_code == 504
    assert resp.json()["detail"] == (
        "DaseR store sync timed out before document registration"
    )
    assert client.get("/documents").json() == []


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
    assert doc.json()["text"] == "abcd"

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
    body = resp.json()
    assert body["text"] == "answer"
    assert body["ttft_ms"] == 12.5
    assert body["prompt_preview"] == (
        "<|system|>S:<|user|>Documents:\n<doc>\n\nTask: go<|assistant|>"
    )
    assert vllm.completions == [
        (
            [
                *[ord(ch) for ch in "<|system|>S:<|user|>Documents:\n"],
                97,
                98,
                99,
                100,
                *[ord(ch) for ch in "\n\nTask: go<|assistant|>"],
            ],
            {"max_tokens": 7},
            {"daser_skip_save": True},
        )
    ]


def test_infer_chat_template_disables_thinking_before_tokenization() -> None:
    core = make_core()
    fake_vllm = FakeVLLMClient(commit_core=core)
    tokenizer = FakeTokenizer()
    app = build_http_app(
        HTTPServerConfig(
            vllm_base_url="http://vllm",
            model="m",
            tokenizer="fake",
            block_tokens=4,
            system_prompt="S:",
            doc_separator="|",
        ),
        core,
        tokenizer=tokenizer,
        vllm=fake_vllm,
    )
    client = TestClient(app)
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]

    resp = client.post("/infer", json={"doc_ids": [doc_id], "task": "go"})

    assert resp.status_code == 200
    assert tokenizer.chat_template_kwargs
    assert all(
        kwargs.get("enable_thinking") is False
        for kwargs in tokenizer.chat_template_kwargs
    )


def test_infer_prompt_preview_replaces_documents_with_titles() -> None:
    """Prompt preview should show structure without dumping full doc text."""
    client, _, _ = _make_client()
    doc_a = client.post("/documents", json={"title": "alpha", "text": "abcd"}).json()
    doc_b = client.post("/documents", json={"title": "beta", "text": "efgh"}).json()

    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_a["doc_id"], doc_b["doc_id"]],
            "task": "go",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["prompt_preview"] == (
        "<|system|>S:<|user|>Documents:\n<alpha>\n|\n<beta>\n\nTask: go<|assistant|>"
    )


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
        *[ord(ch) for ch in "<|system|>S:<|user|>Documents:\n"],
        97,
        98,
        99,
        100,
        101,
        *[ord(ch) for ch in "\n\nTask: go<|assistant|>"],
    ]


def test_infer_trace_cache_returns_lookup_hits() -> None:
    client, _, core = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]
    key = first_rolling_key(_chat_prefix())

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


def test_infer_traces_cache_by_default() -> None:
    """Inference responses include cache hit details without trace_cache input."""
    client, _, core = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]
    key = first_rolling_key(_chat_prefix())
    asyncio.get_event_loop().run_until_complete(
        core.alloc_chunk(key, token_count=4, model_id="m")
    )
    asyncio.get_event_loop().run_until_complete(core.commit_chunk(key))

    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_id],
            "task": "go",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_enabled"] is True
    assert body["cache_hits"][0]["chunk_key"] == key


def test_infer_can_disable_kv_cache_load() -> None:
    """use_kv_cache=False keeps the prompt but disables DaseR lookup/load."""
    client, vllm, _ = _make_client()
    doc_id = client.post("/documents", json={"title": "doc", "text": "abcd"}).json()[
        "doc_id"
    ]

    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc_id],
            "task": "go",
            "use_kv_cache": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_enabled"] is False
    assert body["cache_hits"] == []
    assert vllm.completions[0][2] == {
        "daser_skip_save": True,
        "daser_skip_load": True,
    }


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
    prefix = [*_chat_prefix(), 32]
    separator = [*_doc_separator(), 32]
    system_key = hash_tokens(prefix)
    doc_a_key = hash_tokens([97, 98, 99, 100, 101, 32, 32, 32])
    sep_key = hash_tokens(separator)
    doc_b_key = hash_tokens([102, 103, 104, 105])
    assert [hit["chunk_key"] for hit in body["cache_hits"]] == [
        system_key,
        doc_a_key,
        sep_key,
        doc_b_key,
    ]
    assert [hit["target_token_start"] for hit in body["cache_hits"]] == [
        0,
        len(prefix),
        len(prefix) + 8,
        len(prefix) + 8 + len(separator),
    ]
    assert fake_vllm.prefills == [
        [97, 98, 99, 100, 101, 32, 32, 32],
        [102, 103, 104, 105],
        [*_chat_prefix(), 32],
        [*_doc_separator(), 32],
    ]
    assert fake_vllm.completions[0][0] == [
        *prefix,
        97,
        98,
        99,
        100,
        101,
        32,
        32,
        32,
        *separator,
        102,
        103,
        104,
        105,
        *_chat_suffix(),
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
    system = [*_chat_prefix(), 32]
    doc_a_tokens = [97, 98, 99, 100, 101, 32, 32, 32]
    separator = [*_doc_separator(), 32]
    doc_b_tokens = [102, 103, 104, 105]
    assert fake_vllm.prefills == [
        doc_a_tokens,
        doc_b_tokens,
        system,
        separator,
    ]
    assert [hit["chunk_key"] for hit in body["cache_hits"]] == [
        hash_tokens(system),
        hash_tokens(doc_a_tokens),
        hash_tokens(separator),
        hash_tokens(doc_b_tokens),
    ]
    assert [hit["target_token_start"] for hit in body["cache_hits"]] == [
        0,
        len(system),
        len(system) + len(doc_a_tokens),
        len(system) + len(doc_a_tokens) + len(separator),
    ]
    assert fake_vllm.completions[0][0] == [
        *system,
        *doc_a_tokens,
        *separator,
        *doc_b_tokens,
        *_chat_suffix(),
    ]


def test_chunk_reuse_repeated_separator_keeps_hits_contiguous_before_suffix() -> None:
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
            align_document_chunks=True,
        ),
        core,
        tokenizer=FakeTokenizer(),
        vllm=fake_vllm,
    )
    client = TestClient(app)

    docs = [
        client.post("/documents", json={"title": "a", "text": "abcd"}).json(),
        client.post("/documents", json={"title": "b", "text": "efgh"}).json(),
        client.post("/documents", json={"title": "c", "text": "ijkl"}).json(),
    ]
    resp = client.post(
        "/infer",
        json={
            "doc_ids": [doc["doc_id"] for doc in docs],
            "task": "go",
            "trace_cache": True,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    prefix = [*_chat_prefix(), 32]
    doc_a_tokens = [97, 98, 99, 100]
    separator = [*_doc_separator(), 32]
    doc_b_tokens = [101, 102, 103, 104]
    doc_c_tokens = [105, 106, 107, 108]
    assert [hit["chunk_key"] for hit in body["cache_hits"]] == [
        hash_tokens(prefix),
        hash_tokens(doc_a_tokens),
        hash_tokens(separator),
        hash_tokens(doc_b_tokens),
        hash_tokens(separator),
        hash_tokens(doc_c_tokens),
    ]
    hit_starts = [hit["target_token_start"] for hit in body["cache_hits"]]
    hit_ends = [
        hit["target_token_start"] + hit["token_count"] for hit in body["cache_hits"]
    ]
    assert hit_starts == [0, *hit_ends[:-1]]
    assert hit_ends[-1] == len(fake_vllm.completions[0][0]) - len(_chat_suffix())
