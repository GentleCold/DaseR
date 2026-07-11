# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
import uuid

# Third Party
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# First Party
from daser.config import (
    CACHE_REUSE_PREFIX,
    DEFAULT_CACHE_REUSE_MODE,
)
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.server.core import ServerCore
from daser.server.doc_registry import DocEntry
from daser.server.http.chunker import Chunker, TokenChunk
from daser.server.http.vllm_client import VLLMClient

logger = init_logger(__name__)

_DOCUMENT_STORE_SYNC_TIMEOUT_S = 300.0


@dataclass
class HTTPServerConfig:
    """Runtime configuration for the HTTP server.

    Attributes:
        vllm_base_url: URL of the ``vllm serve`` instance.
        model: model identifier vLLM is serving.
        tokenizer: HuggingFace tokenizer name/path.
        block_tokens: vLLM block size.
        system_prompt: fixed prefix before document prompts.
        doc_separator: separator inserted between documents.
        cache_reuse_mode: cache reuse strategy selected by the DaseR server.
        align_document_chunks: when True, insert padding tokens before each
            document so document chunks begin on vLLM block boundaries.
        trust_remote_code: allow tokenizer repository Python code.
    """

    vllm_base_url: str
    model: str
    tokenizer: str
    block_tokens: int = 16
    system_prompt: str = (
        "You are a helpful assistant answering questions using "
        "the following documents.\n\n"
    )
    doc_separator: str = "\n\n---\n\n"
    cache_reuse_mode: str = DEFAULT_CACHE_REUSE_MODE
    align_document_chunks: bool = False
    transfer_mode: str = "iouring"
    trust_remote_code: bool = False


class UploadRequest(BaseModel):
    """Request body for ``POST /documents``."""

    title: str = Field(..., description="Display title for the document")
    text: str = Field(..., description="Raw document text")


class InferRequest(BaseModel):
    """Request body for ``POST /infer``."""

    doc_ids: list[str] = Field(..., description="Doc IDs to include in the prompt")
    task: str = Field(..., description="User task appended after documents")
    use_kv_cache: bool = Field(
        default=True,
        description="Use DaseR KV cache lookup/load for this inference request",
    )
    trace_cache: bool = Field(
        default=True,
        description="Include control-plane cache lookup details for this prompt",
    )
    gen_params: Optional[dict[str, Any]] = Field(
        default=None, description="OpenAI-style generation parameters"
    )


@dataclass
class PromptSegment:
    """Prompt segment with display text and token IDs.

    Attributes:
        label: segment label used for fixed-cache warmup.
        text: human-readable segment text.
        tokens: token IDs for the segment.
        fixed: True when the segment is reusable fixed prompt structure.
    """

    label: str
    text: str
    tokens: list[int]
    fixed: bool = True


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text without adding special tokens.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        text: input text.

    Returns:
        Token ID list.
    """
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render messages with the tokenizer chat template.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        messages: chat messages with ``role`` and ``content`` keys.
        add_generation_prompt: whether to append the assistant generation
            prefix according to the model template.

    Returns:
        Rendered prompt text.

    Async/thread-safety:
        Pure synchronous tokenizer call; safe to run on the event loop for the
        small fixed strings used by the HTTP service.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        )
    body = ""
    for message in messages:
        body += f"{message['role']}: {message['content']}\n"
    if add_generation_prompt:
        body += "assistant: "
    return body


def _split_rendered_once(rendered: str, marker: str) -> tuple[str, str]:
    """Split rendered template text at a marker.

    Args:
        rendered: full rendered template text.
        marker: unique marker inserted into message content.

    Returns:
        ``(before, after)`` around the marker.

    Raises:
        ValueError: if the marker is missing or appears more than once.
    """
    if rendered.count(marker) != 1:
        raise ValueError(f"chat template marker {marker!r} must appear exactly once")
    before, after = rendered.split(marker, 1)
    return before, after


def _chat_prompt_static_parts(
    tokenizer: Any,
    system_prompt: str,
    task: str,
) -> tuple[str, str]:
    """Return chat-template text around the document region.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        system_prompt: system message text.
        task: user task text.

    Returns:
        ``(user_prefix, user_suffix)`` where documents should be inserted
        between the two pieces.
    """
    docs_marker = "__DASER_DOCUMENTS__"
    user_content = f"Documents:\n{docs_marker}\n\nTask: {task}"
    rendered = _render_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        add_generation_prompt=True,
    )
    return _split_rendered_once(rendered, docs_marker)


def _build_prompt_segments(
    tokenizer: Any,
    system_prompt: str,
    doc_separator: str,
    task: str,
    docs: list[DocEntry],
) -> tuple[list[PromptSegment], str]:
    """Build chat-template prompt segments with document-token placeholders.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        system_prompt: system message text.
        doc_separator: separator text between document chunks.
        task: user task text.
        docs: ordered documents to insert in the user message.

    Returns:
        ``(segments, preview)``. The segment list is suitable for token-level
        prompt construction after replacing document placeholders with each
        document's cached tokens. The preview replaces document text with
        titles.
    """
    user_prefix, user_suffix = _chat_prompt_static_parts(
        tokenizer,
        system_prompt,
        task,
    )
    segments: list[PromptSegment] = [
        PromptSegment("chat_prefix", user_prefix, _tokenize(tokenizer, user_prefix))
    ]
    preview = user_prefix
    for i, doc in enumerate(docs):
        if i > 0:
            separator_text = f"\n{doc_separator}\n"
            segments.append(
                PromptSegment(
                    "doc_separator",
                    separator_text,
                    _tokenize(tokenizer, separator_text),
                )
            )
            preview += separator_text
        segments.append(
            PromptSegment("document", f"<{doc.title}>", list(doc.tokens or []), False)
        )
        preview += f"<{doc.title}>"
    segments.append(
        PromptSegment(
            "chat_suffix",
            user_suffix,
            _tokenize(tokenizer, user_suffix),
            False,
        )
    )
    preview += user_suffix
    return segments, preview


def _doc_to_public_dict(entry: DocEntry) -> dict[str, Any]:
    """Convert a DocEntry to the public HTTP representation.

    Args:
        entry: document registry entry.

    Returns:
        Dict that omits internal full-token storage.
    """
    return {
        "doc_id": entry.doc_id,
        "title": entry.title,
        "created_at": entry.created_at,
        "token_count": entry.token_count,
        "chunk_keys": list(entry.chunk_keys),
        "cached_mask": list(entry.cached_mask),
        "status": entry.status,
        "text": entry.text,
        "error": entry.error,
    }


def _resolve_pad_token(tokenizer: Any) -> int:
    """Return the tokenizer pad token, deriving a fallback from a space token.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.

    Returns:
        Pad token ID to use for block and chunk alignment.
    """
    tokenizer_pad_token = getattr(tokenizer, "pad_token_id", None)
    if tokenizer_pad_token is not None:
        return int(tokenizer_pad_token)
    pad_tokens = _tokenize(tokenizer, " ")
    return pad_tokens[0] if pad_tokens else 0


def _document_chunks(
    chunker: Chunker,
    tokens: list[int],
    pad_token: int,
    align_document_chunks: bool,
) -> list[TokenChunk]:
    """Build cacheable document chunks for the configured reuse mode.

    Args:
        chunker: document chunker.
        tokens: document token IDs.
        pad_token: token used for alignment padding.
        align_document_chunks: when True, return one block-aligned chunk
            for the whole document segment.

    Returns:
        Token chunks to prefill and register.
    """
    if align_document_chunks and tokens:
        return [chunker.single_chunk(tokens, pad_token)]
    return chunker.chunk(tokens)


def _tokens_from_chunks(chunks: list[TokenChunk]) -> list[int]:
    """Flatten chunk token lists.

    Args:
        chunks: token chunks in prompt order.

    Returns:
        Flattened token IDs.
    """
    return [token for chunk in chunks for token in chunk.tokens]


def _segment_tokens(
    chunker: Chunker,
    tokens: list[int],
    pad_token: int,
    align_document_chunks: bool,
) -> list[int]:
    """Return prompt segment tokens with optional block-boundary padding.

    Args:
        chunker: document chunker.
        tokens: segment token IDs.
        pad_token: token used for alignment padding.
        align_document_chunks: pad to a block boundary when True.

    Returns:
        Segment tokens ready to append to a prompt.
    """
    if align_document_chunks:
        return chunker.pad_to_block_boundary(tokens, pad_token)
    return list(tokens)


async def _prefill_chunks(
    vllm: VLLMClient,
    chunks: list[TokenChunk],
    label: str,
) -> list[str]:
    """Prefill chunks through vLLM and return their cache keys.

    Args:
        vllm: vLLM HTTP client.
        chunks: fixed-size token chunks to prefill.
        label: log/error label for the prefill operation.

    Returns:
        Chunk keys in input order.

    Async/thread-safety:
        Runs async HTTP requests sequentially on the server event loop.
    """
    chunk_keys: list[str] = []
    for i, chunk in enumerate(chunks):
        try:
            await vllm.prefill(chunk.tokens)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[HTTP] prefill failed for %s chunk %d: %s", label, i, exc)
            raise HTTPException(
                status_code=502, detail=f"vLLM prefill failed: {exc}"
            ) from exc
        chunk_keys.append(chunk.chunk_key)
    return chunk_keys


async def _prefill_uncached_chunks(
    vllm: VLLMClient,
    core: ServerCore,
    chunks: list[TokenChunk],
    model_id: str,
    label: str,
) -> list[str]:
    """Prefill only chunks that are not already committed in DaseR.

    Args:
        vllm: vLLM HTTP client.
        core: server core used to check committed chunks.
        chunks: fixed-size token chunks to prefill or reuse.
        model_id: model identifier for reuse isolation.
        label: log/error label for the prefill operation.

    Returns:
        Chunk keys in input order, including reused chunks.

    Async/thread-safety:
        Runs async HTTP requests sequentially on the server event loop. It
        reads committed chunk state through ServerCore's public interface.
    """
    chunk_keys: list[str] = []
    scheduled_keys: set[str] = set()
    for chunk in chunks:
        chunk_keys.append(chunk.chunk_key)
        if chunk.chunk_key in scheduled_keys:
            continue
        if core.is_chunk_reusable(chunk.chunk_key, len(chunk.tokens), model_id):
            continue
        await _prefill_chunks(vllm, [chunk], label)
        scheduled_keys.add(chunk.chunk_key)
    return chunk_keys


async def _wait_for_committed_chunks(
    core: ServerCore,
    chunks: list[TokenChunk],
) -> None:
    """Wait until prefetched chunks are committed and visible to lookup.

    Args:
        core: server core used for public lookup synchronization.
        chunks: chunks whose KV was just prefetched through vLLM.

    Raises:
        HTTPException: if the chunks do not become visible before the timeout.

    Async/thread-safety:
        Waits on the server core commit primitive from the FastAPI event loop.
        It does not access connector internals or block the event loop.
    """
    chunk_keys = [chunk.chunk_key for chunk in chunks]
    try:
        await core.wait_for_committed_chunks(
            chunk_keys,
            timeout_s=_DOCUMENT_STORE_SYNC_TIMEOUT_S,
        )
    except TimeoutError as exc:
        uncommitted_keys = [
            key for key in chunk_keys if not core.is_chunk_committed(key)
        ]
        keys = ", ".join(key[:8] for key in uncommitted_keys)
        logger.error("[HTTP] document store sync timed out keys=%s", keys)
        raise HTTPException(
            status_code=504,
            detail="DaseR store sync timed out before document registration",
        ) from exc


async def _prewarm_fixed_segments(
    cfg: HTTPServerConfig,
    tokenizer: Any,
    chunker: Chunker,
    pad_token: int,
    vllm: VLLMClient,
    core: ServerCore,
    prewarmed_fixed_segments: set[str],
) -> None:
    """Prefill fixed RAG segments during startup for chunk reuse mode.

    Args:
        cfg: HTTP server runtime configuration.
        tokenizer: tokenizer used to build fixed segment token IDs.
        chunker: chunker used to pad fixed segments to block boundaries.
        pad_token: token ID used for chunk padding.
        vllm: vLLM HTTP client used to prefill KV.
        core: server core used to detect already-restored fixed chunks.
        prewarmed_fixed_segments: mutable set of fixed segment cache keys.

    Returns:
        None.

    Async/thread-safety:
        Runs on FastAPI lifespan startup before request traffic is served.
        It performs sequential async HTTP calls to vLLM.
    """
    if not cfg.align_document_chunks:
        return
    chat_prefix, _ = _chat_prompt_static_parts(
        tokenizer,
        cfg.system_prompt,
        "",
    )
    for label, text in (
        ("chat_prefix", chat_prefix),
        ("doc_separator", f"\n{cfg.doc_separator}\n"),
    ):
        segment_tokens = _tokenize(tokenizer, text)
        if not segment_tokens:
            continue
        chunk = chunker.single_chunk(segment_tokens, pad_token)
        if chunk.chunk_key in prewarmed_fixed_segments:
            continue
        if await core.lookup(chunk.tokens, cfg.model):
            prewarmed_fixed_segments.add(chunk.chunk_key)
            continue
        await _prefill_chunks(vllm, [chunk], label)
        prewarmed_fixed_segments.add(chunk.chunk_key)


def build_http_app(
    cfg: HTTPServerConfig,
    core: ServerCore,
    tokenizer: Any | None = None,
    vllm: VLLMClient | None = None,
    metrics_registry: MetricsRegistry | None = None,
    drain_transfer: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Construct the HTTP server app.

    Args:
        cfg: HTTP server runtime configuration.
        core: shared server core.
        tokenizer: optional tokenizer override for tests.
        vllm: optional vLLM client override for tests.
        metrics_registry: optional Prometheus metrics registry.
        drain_transfer: optional callback that waits for server-owned transfer
            background work.

    Returns:
        FastAPI instance ready for uvicorn.
    """
    metrics = metrics_registry or REGISTRY
    if tokenizer is None:
        # Third Party
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer, trust_remote_code=cfg.trust_remote_code
        )
    if vllm is None:
        vllm = VLLMClient(base_url=cfg.vllm_base_url, model=cfg.model)
    pad_token = _resolve_pad_token(tokenizer)
    prewarmed_fixed_segments: set[str] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Prewarm fixed segments and close the vLLM client on shutdown.

        Args:
            _: FastAPI app instance supplied by FastAPI.

        Yields:
            None while the app is running.

        Async/thread-safety:
            Runs on FastAPI's lifespan task. Fixed chunk prefill is awaited
            before serving traffic; ``vllm.close`` is awaited during shutdown
            and should not be called concurrently elsewhere.
        """
        await _prewarm_fixed_segments(
            cfg,
            tokenizer,
            chunker,
            pad_token,
            vllm,
            core,
            prewarmed_fixed_segments,
        )
        yield
        await vllm.close()

    app = FastAPI(title="DaseR Server", version="0.1.0", lifespan=lifespan)
    static_dir = resources.files("daser.server.http").joinpath("static")
    app.mount(
        "/ui/static",
        StaticFiles(directory=str(static_dir)),
        name="daser-ui-static",
    )
    chunker = Chunker(block_tokens=cfg.block_tokens)
    metrics.gauge("daser_up", "DaseR HTTP server liveness.").set(1.0)
    metrics.gauge("daser_info", "Static server configuration.").set(
        1.0, labels={"mode": cfg.cache_reuse_mode, "transfer": cfg.transfer_mode}
    )

    @app.get("/", include_in_schema=False)
    async def web_ui() -> FileResponse:
        """Serve the built-in DaseR Web UI."""
        return FileResponse(str(static_dir.joinpath("index.html")))

    async def _ensure_fixed_segment_cached(label: str, tokens: list[int]) -> None:
        """Prefill a fixed RAG segment once in chunk reuse mode."""
        if not cfg.align_document_chunks:
            return
        if not tokens:
            return
        chunk = chunker.single_chunk(tokens, pad_token)
        if chunk.chunk_key in prewarmed_fixed_segments:
            return
        if await core.lookup(chunk.tokens, cfg.model):
            prewarmed_fixed_segments.add(chunk.chunk_key)
            return
        await _prefill_chunks(vllm, [chunk], label)
        prewarmed_fixed_segments.add(chunk.chunk_key)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Return server and vLLM liveness state."""
        vllm_ok = await vllm.health()
        metrics.gauge("daser_vllm_health_up", "vLLM health as seen by DaseR.").set(
            1.0 if vllm_ok else 0.0
        )
        return {
            "status": "ok" if vllm_ok else "degraded",
            "vllm": vllm_ok,
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Return Prometheus exposition text for DaseR metrics."""
        return Response(
            metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/drain")
    async def drain_endpoint() -> dict[str, bool]:
        """Wait for transfer work and allocated stores to commit."""
        if drain_transfer is not None:
            await drain_transfer()
        try:
            await core.wait_for_pending_chunks(timeout_s=_DOCUMENT_STORE_SYNC_TIMEOUT_S)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="DaseR drain timed out waiting for pending stores",
            ) from exc
        return {"ok": True}

    @app.post("/documents", status_code=201)
    async def upload_document(req: UploadRequest) -> dict[str, Any]:
        """Upload a document, prefill chunk KV, and register it."""
        if cfg.cache_reuse_mode == CACHE_REUSE_PREFIX:
            raise HTTPException(
                status_code=400,
                detail="document upload requires chunk cache reuse mode",
            )
        tokens = _tokenize(tokenizer, req.text)
        chunks = _document_chunks(
            chunker,
            tokens,
            pad_token,
            cfg.align_document_chunks,
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="document is empty or shorter than one cacheable chunk",
            )

        chunk_keys: list[str] = []
        t0 = time.time()
        chunk_keys = await _prefill_uncached_chunks(
            vllm,
            core,
            chunks,
            cfg.model,
            "document",
        )
        await _wait_for_committed_chunks(core, chunks)
        prefill_ms = (time.time() - t0) * 1000
        prompt_tokens = (
            _tokens_from_chunks(chunks) if cfg.align_document_chunks else tokens
        )

        doc_id = str(uuid.uuid4())
        try:
            result = await core.register_document(
                doc_id=doc_id,
                title=req.title,
                chunk_keys=chunk_keys,
                token_count=len(tokens),
                tokens=prompt_tokens,
                text=req.text,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[HTTP] register_document failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"DaseR register_document failed: {exc}"
            ) from exc

        logger.info(
            "[HTTP] uploaded doc_id=%s chunks=%d cached=%d prefill_ms=%.1f",
            doc_id,
            len(chunk_keys),
            result.chunk_count_cached,
            prefill_ms,
        )
        return {
            "doc_id": doc_id,
            "status": "ready",
            "chunk_count": len(chunk_keys),
            "chunk_count_cached": result.chunk_count_cached,
            "prefill_ms": prefill_ms,
        }

    @app.get("/documents")
    async def list_documents() -> list[dict[str, Any]]:
        """List registered documents."""
        summaries = await core.list_documents()
        return [summary.to_dict() for summary in summaries]

    @app.get("/documents/{doc_id}")
    async def get_document(doc_id: str) -> dict[str, Any]:
        """Return public metadata for one document."""
        doc = await core.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="doc not found")
        return _doc_to_public_dict(doc)

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> dict[str, Any]:
        """Delete a document and release its chunk references."""
        try:
            result = await core.delete_document(doc_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="doc not found") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"DaseR delete_document: {exc}"
            ) from exc
        return {"ok": True, "chunks_evicted": result.chunks_evicted}

    @app.post("/infer")
    async def infer(req: InferRequest) -> dict[str, Any]:
        """Run inference on a chat-template prompt with cached documents."""
        if not req.doc_ids:
            raise HTTPException(status_code=400, detail="doc_ids must not be empty")

        docs: list[DocEntry] = []
        for doc_id in req.doc_ids:
            doc = await core.get_document(doc_id)
            if doc is None:
                raise HTTPException(status_code=404, detail=f"doc not found: {doc_id}")
            if not doc.tokens:
                raise HTTPException(
                    status_code=409,
                    detail=f"doc {doc_id} has no cached tokens for prompt rebuild",
                )
            docs.append(doc)

        prompt_segments, prompt_preview = _build_prompt_segments(
            tokenizer,
            cfg.system_prompt,
            cfg.doc_separator,
            req.task,
            docs,
        )
        prompt_tokens: list[int] = []
        for segment in prompt_segments:
            if segment.fixed:
                await _ensure_fixed_segment_cached(segment.label, segment.tokens)
            prompt_tokens.extend(
                _segment_tokens(
                    chunker,
                    segment.tokens,
                    pad_token,
                    cfg.align_document_chunks and segment.fixed,
                )
            )

        cache_hits: list[dict[str, Any]] = []
        if req.use_kv_cache and req.trace_cache:
            cache_hits = [
                chunk.to_dict() for chunk in await core.lookup(prompt_tokens, cfg.model)
            ]

        # Tell the connector to skip persisting this request's KV. The
        # /infer prompt is system + doc tokens + task; doc chunks are
        # already cached during /documents upload, and the task suffix
        # is single-use, so re-caching the combined prompt only burns
        # ring-buffer space and GDS write bandwidth.
        t0 = time.time()
        try:
            kv_transfer_params: dict[str, Any] = {"daser_skip_save": True}
            if not req.use_kv_cache:
                kv_transfer_params["daser_skip_load"] = True
            result, ttft_ms = await vllm.completion_with_ttft(
                prompt_tokens,
                req.gen_params,
                kv_transfer_params=kv_transfer_params,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[HTTP] completion failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"vLLM completion: {exc}"
            ) from exc
        elapsed_ms = (time.time() - t0) * 1000

        text = ""
        if result.get("choices"):
            text = result["choices"][0].get("text", "")
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens", 0))

        response = {
            "text": text,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "latency_ms": elapsed_ms,
            "ttft_ms": ttft_ms,
            "cache_enabled": req.use_kv_cache,
            "cache_hits": cache_hits,
            "prompt_preview": prompt_preview,
        }
        return response

    return app
