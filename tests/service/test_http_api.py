# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
from httpx import ASGITransport, AsyncClient
import pytest

# First Party
from daser.service.http_api import ServiceConfig, build_service_app


class FakeTokenizer:
    """Whitespace tokenizer that maps known strings to fixed token IDs."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        mapping = {
            "SYS": [100],
            "|": [200],
            "TASK": [300],
            "question": [400],
        }
        if text in mapping:
            return {"input_ids": mapping[text]}
        return {"input_ids": [int(v) for v in text.split()]}


class FakeIPC:
    """Minimal async IPC fake for service /infer tests."""

    def __init__(self) -> None:
        self.lookup_calls: list[dict[str, Any]] = []
        self.prompt_plan_calls: list[dict[str, Any]] = []

    async def get_doc(self, doc_id: str) -> dict[str, Any]:
        docs = {
            "doc-a": {"tokens": [1, 2, 3, 4]},
            "doc-b": {"tokens": [5, 6, 7, 8]},
        }
        return docs[doc_id]

    async def lookup_doc_chunks(
        self,
        doc_ids: list[str],
        doc_start_offsets: list[int],
        model_id: str,
    ) -> dict[str, Any]:
        self.lookup_calls.append(
            {
                "doc_ids": doc_ids,
                "doc_start_offsets": doc_start_offsets,
                "model_id": model_id,
            }
        )
        return {"chunks": [{"chunk_key": "k"}], "missing": []}

    async def register_prompt_plan(
        self,
        prompt_key: str,
        model_id: str,
        chunks: list[dict[str, Any]],
        missing: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.prompt_plan_calls.append(
            {
                "prompt_key": prompt_key,
                "model_id": model_id,
                "chunks": chunks,
                "missing": missing,
            }
        )
        return {"ok": True}


class FakeVLLM:
    """Minimal async vLLM fake that records completion prompts."""

    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    async def close(self) -> None:
        return

    async def completion(
        self,
        tokens: list[int],
        gen_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.prompts.append(tokens)
        return {
            "choices": [{"text": "ok"}],
            "usage": {"completion_tokens": 1},
        }


@pytest.mark.asyncio
async def test_doc_rope_infer_calls_lookup_doc_chunks() -> None:
    ipc = FakeIPC()
    vllm = FakeVLLM()
    cfg = ServiceConfig(
        vllm_base_url="http://unused",
        model="m",
        tokenizer="unused",
        socket_path="/tmp/unused.sock",
        system_prompt="SYS",
        doc_separator="|",
        task_separator="TASK",
        infer_cache_mode="doc-rope",
    )
    app = build_service_app(
        cfg,
        tokenizer=FakeTokenizer(),
        ipc=ipc,
        vllm=vllm,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/infer",
            json={"doc_ids": ["doc-a", "doc-b"], "task": "question"},
        )

    assert resp.status_code == 200
    assert ipc.lookup_calls == [
        {
            "doc_ids": ["doc-a", "doc-b"],
            "doc_start_offsets": [1, 6],
            "model_id": "m",
        }
    ]
    assert len(ipc.prompt_plan_calls) == 1
    assert ipc.prompt_plan_calls[0]["chunks"] == [{"chunk_key": "k"}]
    assert vllm.prompts == [[100, 1, 2, 3, 4, 200, 5, 6, 7, 8, 300, 400]]
    assert resp.json()["cache_mode"] == "doc-rope"
    assert resp.json()["cache_chunks"] == 1
    assert resp.json()["cache_missing"] == 0
