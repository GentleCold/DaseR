# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import socket
import threading
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

    Uses a persistent blocking Unix socket that is connected lazily on
    first call and reused for subsequent RPCs. On any transport error
    the socket is reset and the call is retried once so a restarted
    server does not leave the client wedged.

    Thread-safety: one scheduler thread at a time is assumed; a lock
    serialises access so that interleaved calls from worker threads do
    not corrupt the framing.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30.0)
        s.connect(self._path)
        return s

    def _reset(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

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
        with self._lock:
            for attempt in range(2):
                if self._sock is None:
                    self._sock = self._connect()
                try:
                    self._sock.sendall(raw)
                    header = _recv_exact(self._sock, _HEADER_SIZE)
                    length = int.from_bytes(header, "big")
                    data = _recv_exact(self._sock, length)
                    break
                except (ConnectionError, OSError, BrokenPipeError) as exc:
                    self._reset()
                    if attempt == 1:
                        raise RuntimeError(f"[IPC] transport failure: {exc}") from exc
        result = _unpack(data)
        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    def close(self) -> None:
        """Close the persistent socket if open."""
        with self._lock:
            self._reset()

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

    def match_and_alloc(
        self, tokens: list[int], chunk_key: str, model_id: str
    ) -> dict[str, Any]:
        """Combined lookup + alloc in one RPC.

        On a cache hit the server returns the matching chunks and no
        allocation; on a miss it allocates a slot for the block-aligned
        prefix and returns the allocation info. Either way the scheduler
        gets both possible futures in a single round trip.

        Args:
            tokens: full prompt token IDs.
            chunk_key: client-computed hash of the block-aligned prefix;
                empty string disables the fallback allocation.
            model_id: model identifier.

        Returns:
            Dict with "chunks" (list[dict]) and "alloc" (dict|None).
        """
        return self.call(
            {
                "op": "match_and_alloc",
                "tokens": tokens,
                "chunk_key": chunk_key,
                "model_id": model_id,
            }
        )

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
            chunk_key: xxh3_128 hex of the token IDs.
        """
        await self.call({"op": "commit_chunk", "chunk_key": chunk_key})

    async def register_doc(
        self,
        doc_id: str,
        title: str,
        chunk_keys: list[str],
        token_count: int,
        tokens: list[int] | None = None,
    ) -> dict[str, Any]:
        """Async: register a document with its chunk_keys.

        Args:
            doc_id: unique document identifier.
            title: display title.
            chunk_keys: ordered chunk_keys belonging to this document.
            token_count: total token count of the original document.
            tokens: optional full token sequence, stored for prompt
                reconstruction during inference.

        Returns:
            Server response dict.
        """
        payload: dict[str, Any] = {
            "op": "register_doc",
            "doc_id": doc_id,
            "title": title,
            "chunk_keys": chunk_keys,
            "token_count": token_count,
        }
        if tokens is not None:
            payload["tokens"] = tokens
        return await self.call(payload)

    async def list_docs(self) -> list[dict[str, Any]]:
        """Async: list all registered documents.

        Returns:
            List of doc summary dicts.
        """
        resp = await self.call({"op": "list_docs"})
        return resp.get("docs", [])

    async def get_doc(self, doc_id: str) -> dict[str, Any]:
        """Async: fetch the full DocEntry for doc_id.

        Args:
            doc_id: document identifier.

        Returns:
            Doc dict (chunk_keys, cached_mask, tokens, status, ...).
        """
        resp = await self.call({"op": "get_doc", "doc_id": doc_id})
        return resp.get("doc", {})

    async def evict_doc(self, doc_id: str) -> dict[str, Any]:
        """Async: evict a document and its solely-referenced chunks.

        Args:
            doc_id: document identifier.

        Returns:
            Server response dict.
        """
        return await self.call({"op": "evict_doc", "doc_id": doc_id})
