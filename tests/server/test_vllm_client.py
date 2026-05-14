# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the VLLMClient HTTP payload shape.

Asserts the exact payload sent to ``/v1/completions`` so the contract
between the RAG API and vLLM stays stable. In particular,
``kv_transfer_params`` must round-trip into the request body when set
and be absent when not — vLLM ignores unknown fields but a typo would
silently disable the skip-save path.

Run with:
    python -m pytest tests/server/test_vllm_client.py -xvs
"""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from daser.server.vllm_client import VLLMClient


class _CapturingClient:
    """Drop-in for httpx.AsyncClient that records POST payloads."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def post(self, url: str, json: dict[str, Any]):  # noqa: A002
        self.posts.append((url, json))

        class _Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"choices": [{"text": "ok"}], "usage": {}}

        return _Resp()

    async def aclose(self) -> None:
        return None


def _make_client() -> tuple[VLLMClient, _CapturingClient]:
    vllm = VLLMClient(base_url="http://localhost:8000", model="dummy")
    fake = _CapturingClient()
    vllm._client = fake  # noqa: SLF001 — test-only injection
    return vllm, fake


@pytest.mark.asyncio
async def test_completion_omits_kv_transfer_params_by_default() -> None:
    vllm, fake = _make_client()

    await vllm.completion([1, 2, 3])

    assert len(fake.posts) == 1
    _, body = fake.posts[0]
    assert "kv_transfer_params" not in body, (
        "Default completion call must not include kv_transfer_params "
        "so existing vLLM endpoints see the same request shape."
    )


@pytest.mark.asyncio
async def test_completion_forwards_kv_transfer_params() -> None:
    vllm, fake = _make_client()

    await vllm.completion(
        [1, 2, 3],
        kv_transfer_params={"daser_skip_save": True},
    )

    _, body = fake.posts[0]
    assert body.get("kv_transfer_params") == {"daser_skip_save": True}


@pytest.mark.asyncio
async def test_completion_merges_gen_params_and_kv_transfer_params() -> None:
    vllm, fake = _make_client()

    await vllm.completion(
        [1, 2, 3],
        gen_params={"max_tokens": 8, "temperature": 0.1},
        kv_transfer_params={"daser_skip_save": True},
    )

    _, body = fake.posts[0]
    assert body["max_tokens"] == 8
    assert body["temperature"] == pytest.approx(0.1)
    assert body["kv_transfer_params"] == {"daser_skip_save": True}
