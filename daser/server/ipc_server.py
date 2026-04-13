# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import math
import os
from typing import Any

# Third Party
import msgpack

# First Party
from daser.logging import init_logger
from daser.position.base import PositionEncoder
from daser.retrieval.base import RetrievalIndex
from daser.server.chunk_manager import ChunkManager

logger = init_logger(__name__)

_HEADER_SIZE = 4


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one length-prefixed msgpack frame from reader.

    Args:
        reader: asyncio stream reader.

    Returns:
        Decoded dict.
    """
    header = await reader.readexactly(_HEADER_SIZE)
    length = int.from_bytes(header, "big")
    data = await reader.readexactly(length)
    return msgpack.unpackb(data, raw=False)


async def _write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Write one length-prefixed msgpack frame to writer.

    Args:
        writer: asyncio stream writer.
        payload: dict to encode and send.
    """
    data = msgpack.packb(payload, use_bin_type=True)
    header = len(data).to_bytes(_HEADER_SIZE, "big")
    writer.write(header + data)
    await writer.drain()


class IPCServer:
    """Asyncio Unix socket server for DaseR's control plane.

    Handles four message types from DaserConnector:
    - lookup:      {op, tokens, model_id} → {chunks: list[dict]}
    - alloc_chunk: {op, chunk_key, token_count, model_id} →
                   {start_slot, file_offset, pos_offset}
    - commit_chunk: {op, chunk_key} → {ok: true}
    - evict_chunk:  {op, chunk_key} → {ok: true}

    Wire protocol: 4-byte big-endian length prefix + msgpack body.

    Args:
        socket_path: Unix socket path.
        chunk_manager: ChunkManager for ring buffer alloc/evict.
        retrieval_index: RetrievalIndex for lookup/insert/remove.
        position_encoder: PositionEncoder for offset assignment.
        slot_size: bytes per slot (for computing file_offset).
        block_tokens: tokens per vLLM block (for computing num_slots).
    """

    def __init__(
        self,
        socket_path: str,
        chunk_manager: ChunkManager,
        retrieval_index: RetrievalIndex,
        position_encoder: PositionEncoder,
        slot_size: int,
        block_tokens: int = 16,
    ) -> None:
        self._socket_path = socket_path
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._slot_size = slot_size
        self._block_tokens = block_tokens
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("[IPC] listening on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the server and remove the socket file."""
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
        """Handle one client connection (one request per connection).

        Args:
            reader: stream reader.
            writer: stream writer.
        """
        try:
            msg = await _read_frame(reader)
            op = msg.get("op")
            if op == "lookup":
                response = await self._handle_lookup(msg)
            elif op == "alloc_chunk":
                response = await self._handle_alloc_chunk(msg)
            elif op == "commit_chunk":
                response = await self._handle_commit_chunk(msg)
            elif op == "evict_chunk":
                response = await self._handle_evict_chunk(msg)
            else:
                response = {"error": f"unknown op: {op}"}
            await _write_frame(writer, response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] error handling request: %s", exc)
            try:
                await _write_frame(writer, {"error": str(exc)})
            except Exception:
                pass
        finally:
            writer.close()

    async def _handle_lookup(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a lookup request.

        Args:
            msg: request dict with tokens and model_id.

        Returns:
            Response dict with list of matching chunk dicts.
        """
        tokens: list[int] = msg["tokens"]
        model_id: str = msg["model_id"]
        chunks = await self._ri.lookup(tokens, model_id)
        return {
            "chunks": [
                {
                    "chunk_key": m.chunk_key,
                    "start_slot": m.start_slot,
                    "num_slots": m.num_slots,
                    "token_count": m.token_count,
                    "pos_offset": m.pos_offset,
                    "model_id": m.model_id,
                    "file_offset": m.start_slot * self._slot_size,
                }
                for m in chunks
            ]
        }

    async def _handle_alloc_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an alloc_chunk request.

        Args:
            msg: request dict with chunk_key, token_count, and model_id.

        Returns:
            Response dict with start_slot, file_offset, and pos_offset.
        """
        chunk_key: str = msg["chunk_key"]
        token_count: int = msg["token_count"]
        model_id: str = msg["model_id"]

        num_slots = math.ceil(token_count / self._block_tokens)
        pos_offset = self._pe.assign_offset(chunk_key, token_count)
        start_slot = self._cm.alloc(
            chunk_key=chunk_key,
            num_slots=num_slots,
            token_count=token_count,
            model_id=model_id,
            pos_offset=pos_offset,
        )
        file_offset = start_slot * self._slot_size
        logger.debug(
            "[IPC] alloc_chunk key=%s start=%d offset=%d pos=%d",
            chunk_key[:8],
            start_slot,
            file_offset,
            pos_offset,
        )
        return {
            "start_slot": start_slot,
            "file_offset": file_offset,
            "pos_offset": pos_offset,
        }

    async def _handle_commit_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a commit_chunk request.

        Args:
            msg: request dict with chunk_key.

        Returns:
            Response dict with ok=True or error.
        """
        chunk_key: str = msg["chunk_key"]
        meta = self._cm.store.get(chunk_key)
        if meta is None:
            return {"error": f"chunk_key not found: {chunk_key}"}
        await self._ri.insert(meta)
        logger.debug("[IPC] commit_chunk key=%s", chunk_key[:8])
        return {"ok": True}

    async def _handle_evict_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an evict_chunk request.

        Args:
            msg: request dict with chunk_key.

        Returns:
            Response dict with ok=True.
        """
        chunk_key: str = msg["chunk_key"]
        await self._ri.remove(chunk_key)
        meta = self._cm.store.get(chunk_key)
        if meta is not None:
            self._cm.store.remove(chunk_key)
        logger.debug("[IPC] evict_chunk key=%s", chunk_key[:8])
        return {"ok": True}
