# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
from typing import Any

from daser.ipc_protocol import read_frame, write_frame

# First Party
from daser.logging import init_logger
from daser.server.core import ServerCore

logger = init_logger(__name__)


class IPCServer:
    """IPC server over Unix socket + msgpack.

    This server is the internal IPC interface for vLLM DaserConnector. It only
    exposes connector cache operations and delegates all business logic to
    ServerCore.

    Args:
        socket_path: Unix socket path.
        core: shared DaseR server core.

    Async/thread-safety:
        Must be started and stopped from the server asyncio event loop.
    """

    def __init__(self, socket_path: str, core: ServerCore) -> None:
        self._socket_path = socket_path
        self._core = core
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Start listening on the Unix socket.

        Async/thread-safety:
            Removes a stale socket path and starts an asyncio Unix server.
        """
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("[IPC] listening on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the server and remove the socket file.

        Async/thread-safety:
            Closes the asyncio server on the current event loop.
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        logger.info("[IPC] server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one connector connection with multiple frames.

        Args:
            reader: request stream reader.
            writer: response stream writer.

        Async/thread-safety:
            Runs one coroutine per client connection.
        """
        try:
            while True:
                try:
                    msg = await read_frame(reader)
                except asyncio.IncompleteReadError:
                    return
                response = await self._dispatch(msg)
                await write_frame(writer, response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] error handling request: %s", exc)
            try:
                await write_frame(writer, {"error": str(exc)})
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one decoded connector request.

        Args:
            msg: request dict containing an ``op`` key.

        Returns:
            Response dict suitable for msgpack encoding.

        Async/thread-safety:
            Calls ServerCore on the same asyncio event loop.
        """
        try:
            op = msg.get("op")
            if op == "lookup":
                chunks = await self._core.lookup(msg["tokens"], msg["model_id"])
                return {"chunks": [chunk.to_dict() for chunk in chunks]}
            if op == "alloc_chunk":
                alloc = await self._core.alloc_chunk(
                    msg["chunk_key"], int(msg["token_count"]), msg["model_id"]
                )
                return alloc.to_dict(include_chunk_key=False)
            if op == "match_and_alloc":
                result = await self._core.match_and_alloc(
                    msg["tokens"], msg.get("chunk_key", ""), msg["model_id"]
                )
                return result.to_dict()
            if op == "commit_chunk":
                await self._core.commit_chunk(msg["chunk_key"])
                return {"ok": True}
            if op == "evict_chunk":
                await self._core.evict_chunk(msg["chunk_key"])
                return {"ok": True}
            return {"error": f"unknown op: {op}"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] request failed: %s", exc)
            return {"error": str(exc)}
