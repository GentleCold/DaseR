# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import time
from typing import Any, Optional

# Third Party
import httpx

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)


class VLLMClient:
    """Thin async HTTP client for a ``vllm serve`` endpoint.

    Provides two kinds of calls the service layer needs:
    - ``prefill``: fires a short completion that runs the tokens through
      the model's forward pass so the DaserConnector's save path records
      them as a chunk on NVMe. Uses the smallest possible ``max_tokens``
      that the OpenAI-compatible API accepts.
    - ``completion``: regular completion used by the inference endpoint.

    Both paths talk to the ``/v1/completions`` endpoint because
    ``prompt_token_ids`` is the easiest way to feed pre-tokenized text.

    Args:
        base_url: vLLM base URL, e.g. ``http://127.0.0.1:8001``.
        model: model name to pass to the OpenAI API (must match the
            name vLLM serves under; typically the model path).
        timeout: per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "VLLMClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client if it was opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``/v1/completions`` and return the JSON body.

        Args:
            payload: request body for the OpenAI-compatible endpoint.

        Returns:
            Parsed JSON response.
        """
        client = self._client
        if client is None:
            client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
            self._client = client
        resp = await client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def prefill(self, tokens: list[int]) -> None:
        """Run a prefill-only pass over ``tokens`` so DaseR caches them.

        vLLM's OpenAI layer rejects ``max_tokens=0``, so we request
        exactly one decoded token and discard it. The single extra
        decoded token does not affect the cached KV chunks because
        those are keyed on the original prompt prefix.

        Args:
            tokens: token IDs to run through the model.
        """
        payload = {
            "model": self._model,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        await self._post_completions(payload)

    async def completion(
        self,
        tokens: list[int],
        gen_params: Optional[dict[str, Any]] = None,
        kv_transfer_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run a normal completion for the supplied tokens.

        Args:
            tokens: token IDs forming the prompt.
            gen_params: optional OpenAI-style generation parameters
                (max_tokens, temperature, top_p, ...). Unknown keys
                are forwarded untouched; vLLM decides what to accept.
            kv_transfer_params: optional per-request KV-transfer hints
                forwarded to vLLM as the top-level ``kv_transfer_params``
                field. vLLM exposes the dict on ``request.kv_transfer_params``
                so the connector can adjust per-request KV behavior (e.g.
                skipping persistence of task-prompt KV). When ``None``
                the field is omitted to preserve the prior request shape.

        Returns:
            Parsed OpenAI-format completion response.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": tokens,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if gen_params:
            payload.update(gen_params)
        if kv_transfer_params is not None:
            payload["kv_transfer_params"] = kv_transfer_params
        return await self._post_completions(payload)

    async def completion_with_ttft(
        self,
        tokens: list[int],
        gen_params: Optional[dict[str, Any]] = None,
        kv_transfer_params: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], float]:
        """Run a streaming completion and measure time to first token.

        Args:
            tokens: token IDs forming the prompt.
            gen_params: optional OpenAI-style generation parameters.
            kv_transfer_params: optional per-request KV-transfer hints
                forwarded to vLLM.

        Returns:
            ``(completion_response, ttft_ms)`` where ``completion_response``
            uses the non-streaming OpenAI completion shape expected by the
            service layer, and ``ttft_ms`` is measured from request submission
            to the first non-empty streamed text fragment.

        Async/thread-safety:
            Uses the client's asyncio HTTP session and must be awaited from an
            event loop. Timing is local wall-clock process time.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": tokens,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if gen_params:
            payload.update(gen_params)
        if kv_transfer_params is not None:
            payload["kv_transfer_params"] = kv_transfer_params

        client = self._client
        if client is None:
            client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
            self._client = client

        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        start = time.perf_counter()
        first_token_at: float | None = None
        async with client.stream("POST", "/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                if chunk.get("usage") is not None:
                    usage = dict(chunk["usage"])
                for choice in chunk.get("choices", []):
                    fragment = str(choice.get("text", ""))
                    if not fragment:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(fragment)

        end = time.perf_counter()
        ttft_ms = ((first_token_at or end) - start) * 1000
        return {
            "choices": [{"text": "".join(text_parts)}],
            "usage": usage,
        }, ttft_ms

    async def health(self) -> bool:
        """Return True when vLLM answers HTTP 200 at ``/health``.

        Failures bubble up as False rather than raising so the service
        ``/health`` endpoint can report partial availability.

        Returns:
            True on HTTP 200, False otherwise.
        """
        client = self._client
        if client is None:
            client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
            self._client = client
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SERVICE] vLLM health check failed: %s", exc)
            return False

    async def list_models(self) -> list[str]:
        """Return model IDs reported by vLLM's OpenAI-compatible API.

        Returns:
            List of model IDs from ``GET /v1/models``.

        Async/thread-safety:
            Uses the client's asyncio HTTP session and must be awaited from an
            event loop.
        """
        client = self._client
        if client is None:
            client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
            self._client = client
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        body = resp.json()
        return [str(model["id"]) for model in body.get("data", [])]
