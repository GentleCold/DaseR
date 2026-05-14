# SPDX-License-Identifier: Apache-2.0

# Standard
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

    @property
    def model(self) -> str:
        """Model identifier used by this client."""
        return self._model

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
    ) -> dict[str, Any]:
        """Run a normal completion for the supplied tokens.

        Args:
            tokens: token IDs forming the prompt.
            gen_params: optional OpenAI-style generation parameters
                (max_tokens, temperature, top_p, ...). Unknown keys
                are forwarded untouched; vLLM decides what to accept.

        Returns:
            Parsed OpenAI-format completion response.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": tokens,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
        }
        if gen_params:
            payload.update(gen_params)
        return await self._post_completions(payload)

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
