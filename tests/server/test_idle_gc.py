# SPDX-License-Identifier: Apache-2.0
"""Idle GC must cover overlapping requests, transfers and lifecycle failures."""

from argparse import Namespace
import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import gc

import pytest
from starlette.types import ASGIApp, Receive, Scope, Send

from daser.server.__main__ import run_server
from daser.server.idle_gc import IdleGCMiddleware


@asynccontextmanager
async def running(
    handler: ASGIApp, is_idle: Callable[[], bool]
) -> AsyncIterator[IdleGCMiddleware]:
    """Run the real middleware lifespan around a controlled HTTP handler."""
    ready = asyncio.Event()
    stop = asyncio.Event()

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            ready.set()
            await stop.wait()
        else:
            await handler(scope, receive, send)

    middleware = IdleGCMiddleware(app, is_idle, interval_s=0.001)
    lifespan = asyncio.create_task(middleware({"type": "lifespan"}, unused, unused))
    await asyncio.wait_for(ready.wait(), timeout=1)
    try:
        yield middleware
    finally:
        stop.set()
        await lifespan


async def unused(*_args: object) -> None:
    """Fail if a controlled test app unexpectedly uses ASGI messaging."""
    raise AssertionError("unexpected ASGI messaging")


async def test_overlapping_requests_defer_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finishing one request cannot collect while another remains active."""
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    collected = asyncio.Event()
    generations: list[int] = []

    async def handler(scope: Scope, _receive: Receive, _send: Send) -> None:
        lane = scope["lane"]
        entered[lane].set()
        await release[lane].wait()

    def collect(generation: int) -> int:
        generations.append(generation)
        collected.set()
        return 0

    monkeypatch.setattr(gc, "get_count", lambda: (701, 0, 11))
    monkeypatch.setattr(gc, "collect", collect)
    originally_enabled = gc.isenabled()
    async with running(handler, lambda: True) as middleware:
        assert not gc.isenabled()
        requests = [
            asyncio.create_task(middleware({"type": "http", "lane": i}, unused, unused))
            for i in range(2)
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 1)
            await asyncio.sleep(0.01)
            assert generations == []
            release[0].set()
            await requests[0]
            await asyncio.sleep(0.01)
            assert generations == []
            release[1].set()
            await requests[1]
            await asyncio.wait_for(collected.wait(), 1)
            assert set(generations) == {2}
        finally:
            for event in release:
                event.set()
            await asyncio.gather(*requests, return_exceptions=True)
    assert gc.isenabled() == originally_enabled


async def test_ipc_and_background_transfers_defer_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HTTP scopes does not imply that the IPC/transfer plane is idle."""
    idle = False
    collected = asyncio.Event()

    def collect(generation: int) -> int:
        assert idle
        assert generation == 0
        collected.set()
        return 0

    monkeypatch.setattr(gc, "get_count", lambda: (701, 0, 0))
    monkeypatch.setattr(gc, "collect", collect)
    async with running(unused, lambda: idle):
        await asyncio.sleep(0.01)
        assert not collected.is_set()
        idle = True
        await asyncio.wait_for(collected.wait(), 1)


async def test_cancelled_request_releases_gc_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must not leave GC disabled from a leaked activity count."""
    entered = asyncio.Event()
    collected = asyncio.Event()

    async def handler(_scope: Scope, _receive: Receive, _send: Send) -> None:
        entered.set()
        await asyncio.Future()

    def collect(_generation: int) -> int:
        collected.set()
        return 0

    monkeypatch.setattr(gc, "get_count", lambda: (701, 0, 0))
    monkeypatch.setattr(gc, "collect", collect)
    async with running(handler, lambda: True) as middleware:
        request = asyncio.create_task(middleware({"type": "http"}, unused, unused))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0.01)
            assert not collected.is_set()
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        await asyncio.wait_for(collected.wait(), 1)


@pytest.mark.parametrize("enabled", [False, True])
async def test_failed_lifespan_restores_original_gc_state(enabled: bool) -> None:
    """Startup errors must survive cancellation of a not-yet-started collector."""
    originally_enabled = gc.isenabled()
    (gc.enable if enabled else gc.disable)()

    async def fail(_scope: Scope, _receive: Receive, _send: Send) -> None:
        assert not gc.isenabled()
        raise ValueError("startup failed")

    try:
        middleware = IdleGCMiddleware(fail, lambda: True)
        with pytest.raises(ValueError, match="startup failed"):
            await middleware({"type": "lifespan"}, unused, unused)
        assert gc.isenabled() == enabled
    finally:
        (gc.enable if originally_enabled else gc.disable)()


async def test_cancelled_lifespan_restores_gc_state() -> None:
    """Cancellation of the whole application still joins its collector."""
    ready = asyncio.Event()
    originally_enabled = gc.isenabled()

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        ready.set()
        await asyncio.Future()

    middleware = IdleGCMiddleware(app, lambda: False)
    lifespan = asyncio.create_task(middleware({"type": "lifespan"}, unused, unused))
    try:
        await asyncio.wait_for(ready.wait(), 1)
        assert not gc.isenabled()
    finally:
        lifespan.cancel()
        (outcome,) = await asyncio.gather(lifespan, return_exceptions=True)
    assert isinstance(outcome, asyncio.CancelledError)
    assert gc.isenabled() == originally_enabled


async def test_server_startup_failure_restores_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server startup must disable GC before IPC and restore it on failure."""
    originally_enabled = gc.isenabled()

    async def unavailable(_url: str) -> str:
        assert not gc.isenabled()
        raise ValueError("vLLM unavailable")

    monkeypatch.setattr("daser.server.__main__._read_vllm_model_id", unavailable)
    with pytest.raises(ValueError, match="vLLM unavailable"):
        await run_server(Namespace(gc_policy="idle", vllm_base_url="http://unused"))
    assert gc.isenabled() == originally_enabled
