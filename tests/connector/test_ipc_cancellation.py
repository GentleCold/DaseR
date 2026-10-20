# SPDX-License-Identifier: Apache-2.0
"""Worker RPC cancellation must preserve frame and buffer ownership."""

import asyncio
from collections.abc import AsyncIterator
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from daser.connector.ipc_client import IPCClientAsync
from daser.ipc_protocol import pack_frame, read_frame, write_frame


@asynccontextmanager
async def fragmented_peer(
    path: Path, split: int, disconnect: bool = False
) -> AsyncIterator[tuple[str, asyncio.Event, asyncio.Event, list[int]]]:
    """Hold the first response at a selected frame boundary."""
    received = asyncio.Event()
    release = asyncio.Event()
    seen: list[int] = []
    writers: set[asyncio.StreamWriter] = set()
    tasks: set[asyncio.Task[None]] = set()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        writers.add(writer)
        try:
            while True:
                message = await read_frame(reader)
                request_id = int(message["id"])
                seen.append(request_id)
                if request_id == 1:
                    response = pack_frame({"id": request_id})
                    writer.write(response[:split])
                    await writer.drain()
                    received.set()
                    await release.wait()
                    if disconnect:
                        return
                    writer.write(response[split:])
                    await writer.drain()
                else:
                    await write_frame(writer, {"id": request_id})
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writers.discard(writer)
            tasks.discard(task)
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    server = await asyncio.start_unix_server(serve, path=str(path))
    try:
        yield str(path), received, release, seen
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        for writer in list(writers):
            writer.close()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*list(tasks)), 1)


@pytest.mark.parametrize("split", [0, 2, 5])
@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_cancelled_rpc_drains_its_frame_before_reusing_client(
    tmp_path: Path, split: int, cancellations: int, disconnect: bool
) -> None:
    """Cancellation at either header/body boundary cannot corrupt the next RPC."""
    async with fragmented_peer(tmp_path / "cancel.sock", split, disconnect) as peer:
        path, received, release, seen = peer
        client = IPCClientAsync(path)
        first = asyncio.create_task(client.call({"op": "echo", "id": 1}))
        second: asyncio.Task[dict] | None = None
        try:
            await asyncio.wait_for(received.wait(), 1)
            # Let readexactly consume a complete header before cancelling a
            # partial body. Retrying a whole read_frame here loses framing.
            await asyncio.sleep(0.002)
            for _ in range(cancellations):
                first.cancel()
                await asyncio.sleep(0)
            assert not first.done()
            second = asyncio.create_task(client.call({"op": "echo", "id": 2}))
            await asyncio.sleep(0)
            assert not second.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 1)
            assert await asyncio.wait_for(second, 1) == {"id": 2}
            assert seen == [1, 2]
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (first, second) if task is not None),
                return_exceptions=True,
            )
            await client.close()


async def test_cancelling_a_lock_waiter_does_not_touch_the_active_rpc(
    tmp_path: Path,
) -> None:
    """An unsent cancellation cannot retire another request's transport."""
    async with fragmented_peer(tmp_path / "waiter.sock", 0) as peer:
        path, received, release, seen = peer
        client = IPCClientAsync(path)
        first = asyncio.create_task(client.call({"op": "echo", "id": 1}))
        try:
            await asyncio.wait_for(received.wait(), 1)
            waiter = asyncio.create_task(client.call({"op": "echo", "id": 2}))
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release.set()
            assert await asyncio.wait_for(first, 1) == {"id": 1}
            assert await client.call({"op": "echo", "id": 3}) == {"id": 3}
            assert seen == [1, 3]
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
            await client.close()


async def test_cancelled_write_drain_does_not_resend_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flow-control cancellation still drains exactly the submitted exchange."""
    entered = asyncio.Event()
    release = asyncio.Event()
    writes: list[bytes] = []
    reader = asyncio.StreamReader()
    reader.feed_data(pack_frame({"ok": True}))

    class Writer:
        def is_closing(self) -> bool:
            return False

        def write(self, data: bytes) -> None:
            writes.append(data)

        async def drain(self) -> None:
            entered.set()
            await release.wait()

        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    writer = Writer()

    async def connect(_path: str) -> tuple[asyncio.StreamReader, Writer]:
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", connect)
    client = IPCClientAsync("unused")
    request = asyncio.create_task(client.call({"op": "store"}))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(3):
            request.cancel()
            await asyncio.sleep(0)
        assert not request.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, 1)
        assert writes == [pack_frame({"op": "store"})]
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)
        await client.close()
