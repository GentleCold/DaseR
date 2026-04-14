# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import socket
from typing import Any

# Third Party
import msgpack

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

_HEADER_SIZE = 4


def _pack(payload: dict[str, Any]) -> bytes:
    """Encode payload as length-prefixed msgpack frame."""
    data = msgpack.packb(payload, use_bin_type=True)
    return len(data).to_bytes(_HEADER_SIZE, "big") + data


def _unpack(raw: bytes) -> dict[str, Any]:
    """Decode a raw msgpack bytes object to a dict."""
    return msgpack.unpackb(raw, raw=False)


def _recv_exact(s: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a blocking socket.

    Args:
        s: connected socket.
        n: number of bytes to receive.

    Returns:
        Exactly n bytes.

    Raises:
        ConnectionError: if the connection closes before n bytes arrive.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before receiving all bytes")
        buf.extend(chunk)
    return bytes(buf)


class IPCClientSync:
    """Synchronous blocking IPC client for scheduler-side calls.

    Uses a raw blocking Unix socket for each call. Target RTT < 0.1 ms
    (Unix socket, same machine). A new connection is opened per call so
    no persistent connection state is needed.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the response (blocking).

        Args:
            payload: dict with "op" and any required fields.

        Returns:
            Response dict from the server.

        Raises:
            RuntimeError: if the server returns an error response.
            TimeoutError: if the server does not respond within 30 seconds.
        """
        raw = _pack(payload)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(30.0)
            s.connect(self._path)
            s.sendall(raw)
            header = _recv_exact(s, _HEADER_SIZE)
            length = int.from_bytes(header, "big")
            data = _recv_exact(s, length)
        result = _unpack(data)
        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    def lookup(self, tokens: list[int], model_id: str) -> list[dict[str, Any]]:
        """Look up cached chunks for the given token sequence.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.

        Returns:
            List of chunk dicts (may be empty).
        """
        resp = self.call({"op": "lookup", "tokens": tokens, "model_id": model_id})
        return resp.get("chunks", [])

    def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str
    ) -> dict[str, Any]:
        """Allocate a slot for a new chunk.

        Args:
            chunk_key: SHA256 hex of the token IDs.
            token_count: number of tokens in the chunk.
            model_id: model identifier.

        Returns:
            Dict with start_slot, file_offset, pos_offset.
        """
        return self.call(
            {
                "op": "alloc_chunk",
                "chunk_key": chunk_key,
                "token_count": token_count,
                "model_id": model_id,
            }
        )

    def commit_chunk(self, chunk_key: str) -> None:
        """Mark a chunk as committed (GDS write complete).

        Args:
            chunk_key: SHA256 hex of the chunk's token IDs.
        """
        self.call({"op": "commit_chunk", "chunk_key": chunk_key})

    def evict_chunk(self, chunk_key: str) -> None:
        """Evict a chunk from the DaseR index.

        Args:
            chunk_key: SHA256 hex of the chunk's token IDs.
        """
        self.call({"op": "evict_chunk", "chunk_key": chunk_key})


class IPCClientAsync:
    """Asyncio IPC client for worker-side calls.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request asynchronously and return the response.

        Args:
            payload: dict with "op" and any required fields.

        Returns:
            Response dict from the server.

        Raises:
            RuntimeError: if the server returns an error response.
        """
        reader, writer = await asyncio.open_unix_connection(self._path)
        try:
            data = msgpack.packb(payload, use_bin_type=True)
            header = len(data).to_bytes(_HEADER_SIZE, "big")
            writer.write(header + data)
            await writer.drain()

            resp_header = await reader.readexactly(_HEADER_SIZE)
            resp_len = int.from_bytes(resp_header, "big")
            resp_data = await reader.readexactly(resp_len)
        finally:
            writer.close()
            await writer.wait_closed()

        result = _unpack(resp_data)
        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    async def commit_chunk(self, chunk_key: str) -> None:
        """Async: mark a chunk as committed.

        Args:
            chunk_key: SHA256 hex of the token IDs.
        """
        await self.call({"op": "commit_chunk", "chunk_key": chunk_key})
