# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import time
from typing import Any, Optional
import uuid

# Third Party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# First Party
from daser.connector.ipc_client import IPCClientAsync
from daser.logging import init_logger
from daser.service.chunker import Chunker
from daser.service.vllm_client import VLLMClient

logger = init_logger(__name__)


@dataclass
class ServiceConfig:
    """Runtime configuration for the service layer.

    Attributes:
        vllm_base_url: URL of the ``vllm serve`` instance.
        model: model identifier vLLM is serving.
        tokenizer: HuggingFace tokenizer name / path; used to tokenize
            uploaded documents and task prompts.
        socket_path: DaseR IPC socket path.
        block_tokens: vLLM block size (must match the server).
        chunk_blocks: blocks per chunk (granularity/IPC trade-off).
        system_prompt: fixed [SYS] block inserted before doc prompts.
        doc_separator: text inserted between concatenated doc chunks.
        task_separator: text inserted before the task prompt.
    """

    vllm_base_url: str
    model: str
    tokenizer: str
    socket_path: str
    block_tokens: int = 16
    chunk_blocks: int = 16
    system_prompt: str = (
        "You are a helpful assistant answering questions using "
        "the following documents.\n\n"
    )
    doc_separator: str = "\n\n---\n\n"
    task_separator: str = "\n\n---\nTask: "


class UploadRequest(BaseModel):
    """POST /documents body."""

    title: str = Field(..., description="Display title for the document")
    text: str = Field(..., description="Raw document text")


class InferRequest(BaseModel):
    """POST /infer body."""

    doc_ids: list[str] = Field(..., description="Doc IDs to include in the prompt")
    task: str = Field(..., description="User task / question appended after documents")
    gen_params: Optional[dict[str, Any]] = Field(
        default=None, description="OpenAI-style generation parameters"
    )


def _tokenize(tokenizer, text: str) -> list[int]:
    """Tokenize ``text`` without adding special tokens.

    Centralised so every call site keeps identical behavior, which
    matters because ``chunk_key`` is a hash of the exact token sequence.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        text: input string.

    Returns:
        Token ID list.
    """
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_service_app(cfg: ServiceConfig) -> FastAPI:
    """Construct the FastAPI app wired to the given ServiceConfig.

    The app owns three long-lived helpers (IPC client, vLLM client,
    tokenizer) and a Chunker. Handlers are written as closures so the
    helpers can be captured without making them module-level globals.

    Args:
        cfg: ServiceConfig produced by the entry point.

    Returns:
        FastAPI instance ready to be passed to uvicorn.
    """
    # Third Party
    from transformers import AutoTokenizer

    app = FastAPI(title="DaseR Service", version="0.1.0")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    chunker = Chunker(block_tokens=cfg.block_tokens, chunk_blocks=cfg.chunk_blocks)
    ipc = IPCClientAsync(cfg.socket_path)
    vllm = VLLMClient(base_url=cfg.vllm_base_url, model=cfg.model)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await vllm.close()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness probe: reports vLLM reachability."""
        vllm_ok = await vllm.health()
        return {
            "status": "ok" if vllm_ok else "degraded",
            "vllm": vllm_ok,
        }

    @app.post("/documents", status_code=201)
    async def upload_document(req: UploadRequest) -> dict[str, Any]:
        """Upload a document, prefill chunk KV, then register with DaseR."""
        tokens = _tokenize(tokenizer, req.text)
        chunks = chunker.chunk(tokens)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"document is shorter than one chunk "
                    f"({chunker.chunk_tokens} tokens)"
                ),
            )

        chunk_keys: list[str] = []
        t0 = time.time()
        for i, chunk in enumerate(chunks):
            try:
                await vllm.prefill(chunk.tokens)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[SERVICE] prefill failed for chunk %d: %s", i, exc)
                raise HTTPException(
                    status_code=502, detail=f"vLLM prefill failed: {exc}"
                ) from exc
            chunk_keys.append(chunk.chunk_key)
        prefill_ms = (time.time() - t0) * 1000

        doc_id = str(uuid.uuid4())
        try:
            resp = await ipc.register_doc(
                doc_id=doc_id,
                title=req.title,
                chunk_keys=chunk_keys,
                token_count=len(tokens),
                tokens=tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SERVICE] register_doc failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"DaseR register_doc failed: {exc}"
            ) from exc

        logger.info(
            "[SERVICE] uploaded doc_id=%s chunks=%d cached=%d prefill_ms=%.1f",
            doc_id,
            len(chunk_keys),
            resp.get("chunk_count_cached", 0),
            prefill_ms,
        )
        return {
            "doc_id": doc_id,
            "status": "ready",
            "chunk_count": len(chunk_keys),
            "chunk_count_cached": resp.get("chunk_count_cached", len(chunk_keys)),
            "prefill_ms": prefill_ms,
        }

    @app.get("/documents")
    async def list_documents() -> list[dict[str, Any]]:
        """List every registered document."""
        try:
            return await ipc.list_docs()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"DaseR list_docs: {exc}"
            ) from exc

    @app.get("/documents/{doc_id}")
    async def get_document(doc_id: str) -> dict[str, Any]:
        """Return the DocEntry for ``doc_id``."""
        try:
            doc = await ipc.get_doc(doc_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"DaseR get_doc: {exc}"
            ) from exc
        if not doc:
            raise HTTPException(status_code=404, detail="doc not found")
        # Don't leak the full tokens array over the public endpoint;
        # it's internal state used by /infer.
        doc = {k: v for k, v in doc.items() if k != "tokens"}
        return doc

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> dict[str, Any]:
        """Unregister a document and cascade-evict its chunks."""
        try:
            return await ipc.evict_doc(doc_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"DaseR evict_doc: {exc}"
            ) from exc

    @app.post("/infer")
    async def infer(req: InferRequest) -> dict[str, Any]:
        """Run inference on concatenated doc tokens + task prompt."""
        if not req.doc_ids:
            raise HTTPException(status_code=400, detail="doc_ids must not be empty")

        prompt_tokens: list[int] = _tokenize(tokenizer, cfg.system_prompt)
        separator_tokens = _tokenize(tokenizer, cfg.doc_separator)
        task_prefix_tokens = _tokenize(tokenizer, cfg.task_separator)

        for i, doc_id in enumerate(req.doc_ids):
            try:
                doc = await ipc.get_doc(doc_id)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502, detail=f"DaseR get_doc({doc_id}): {exc}"
                ) from exc
            if not doc:
                raise HTTPException(status_code=404, detail=f"doc not found: {doc_id}")
            doc_tokens = doc.get("tokens")
            if not doc_tokens:
                raise HTTPException(
                    status_code=409,
                    detail=f"doc {doc_id} has no cached tokens for prompt rebuild",
                )
            if i > 0:
                prompt_tokens.extend(separator_tokens)
            prompt_tokens.extend(doc_tokens)

        prompt_tokens.extend(task_prefix_tokens)
        prompt_tokens.extend(_tokenize(tokenizer, req.task))

        t0 = time.time()
        try:
            result = await vllm.completion(prompt_tokens, req.gen_params)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SERVICE] completion failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"vLLM completion: {exc}"
            ) from exc
        elapsed_ms = (time.time() - t0) * 1000

        text = ""
        completion_tokens = 0
        if result.get("choices"):
            text = result["choices"][0].get("text", "")
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens", 0))

        return {
            "text": text,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "latency_ms": elapsed_ms,
        }

    return app
