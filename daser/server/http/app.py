# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
import time
from typing import Any, AsyncIterator, Optional
import uuid

# Third Party
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# First Party
from daser.logging import init_logger
from daser.server.core import ServerCore
from daser.server.doc_registry import DocEntry
from daser.server.http.chunker import Chunker, TokenChunk
from daser.server.http.vllm_client import VLLMClient

logger = init_logger(__name__)


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
        task_separator: separator inserted before task text.
        answer_separator: separator inserted after task text before generation.
        align_document_chunks: when True, insert padding tokens before each
            document so document chunks begin on vLLM block boundaries.
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
    task_separator: str = "\n\n---\nTask: "
    answer_separator: str = "\nAnswer: "
    align_document_chunks: bool = False


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


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text without adding special tokens.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        text: input text.

    Returns:
        Token ID list.
    """
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


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


def build_http_app(
    cfg: HTTPServerConfig,
    core: ServerCore,
    tokenizer: Any | None = None,
    vllm: VLLMClient | None = None,
) -> FastAPI:
    """Construct the HTTP server app.

    Args:
        cfg: HTTP server runtime configuration.
        core: shared server core.
        tokenizer: optional tokenizer override for tests.
        vllm: optional vLLM client override for tests.

    Returns:
        FastAPI instance ready for uvicorn.
    """
    if tokenizer is None:
        # Third Party
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if vllm is None:
        vllm = VLLMClient(base_url=cfg.vllm_base_url, model=cfg.model)
    pad_token = _resolve_pad_token(tokenizer)
    prewarmed_fixed_segments: set[str] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close the vLLM client when the app shuts down.

        Args:
            _: FastAPI app instance supplied by FastAPI.

        Yields:
            None while the app is running.

        Async/thread-safety:
            Runs on FastAPI's lifespan task; ``vllm.close`` is awaited during
            shutdown and should not be called concurrently elsewhere.
        """
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
        await _prefill_chunks(vllm, [chunk], label)
        prewarmed_fixed_segments.add(chunk.chunk_key)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Return server and vLLM liveness state."""
        vllm_ok = await vllm.health()
        return {
            "status": "ok" if vllm_ok else "degraded",
            "vllm": vllm_ok,
        }

    @app.post("/documents", status_code=201)
    async def upload_document(req: UploadRequest) -> dict[str, Any]:
        """Upload a document, prefill chunk KV, and register it."""
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
        chunk_keys = await _prefill_chunks(vllm, chunks, "document")
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
        """Run inference on concatenated document tokens and task."""
        if not req.doc_ids:
            raise HTTPException(status_code=400, detail="doc_ids must not be empty")

        system_tokens = _tokenize(tokenizer, cfg.system_prompt)
        separator_tokens = _tokenize(tokenizer, cfg.doc_separator)
        task_prefix_tokens = _tokenize(tokenizer, cfg.task_separator)
        answer_prefix_tokens = _tokenize(tokenizer, cfg.answer_separator)
        if cfg.align_document_chunks:
            await _ensure_fixed_segment_cached("system", system_tokens)
            if len(req.doc_ids) > 1:
                await _ensure_fixed_segment_cached("separator", separator_tokens)
        prompt_tokens = _segment_tokens(
            chunker,
            system_tokens,
            pad_token,
            cfg.align_document_chunks,
        )
        prompt_preview = cfg.system_prompt

        for i, doc_id in enumerate(req.doc_ids):
            doc = await core.get_document(doc_id)
            if doc is None:
                raise HTTPException(status_code=404, detail=f"doc not found: {doc_id}")
            if not doc.tokens:
                raise HTTPException(
                    status_code=409,
                    detail=f"doc {doc_id} has no cached tokens for prompt rebuild",
                )
            if i > 0:
                prompt_tokens.extend(
                    _segment_tokens(
                        chunker,
                        separator_tokens,
                        pad_token,
                        cfg.align_document_chunks,
                    )
                )
            prompt_tokens.extend(doc.tokens)
            if i > 0:
                prompt_preview += cfg.doc_separator
            prompt_preview += f"<{doc.title}>"

        prompt_tokens.extend(task_prefix_tokens)
        prompt_tokens.extend(_tokenize(tokenizer, req.task))
        prompt_tokens.extend(answer_prefix_tokens)
        prompt_preview += f"{cfg.task_separator}{req.task}{cfg.answer_separator}"

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
            result = await vllm.completion(
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
            "cache_enabled": req.use_kv_cache,
            "cache_hits": cache_hits,
            "prompt_preview": prompt_preview,
        }
        return response

    return app
