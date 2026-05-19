# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import socket
import threading
from typing import Any

# First Party
from daser.ipc_protocol import pack_frame, read_frame, recv_frame
from daser.logging import init_logger

logger = init_logger(__name__)


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
        raw = pack_frame(payload)
        with self._lock:
            for attempt in range(2):
                if self._sock is None:
                    self._sock = self._connect()
                try:
                    self._sock.sendall(raw)
                    result = recv_frame(self._sock)
                    break
                except (ConnectionError, OSError, BrokenPipeError) as exc:
                    self._reset()
                    if attempt == 1:
                        raise RuntimeError(f"[IPC] transport failure: {exc}") from exc
        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    def close(self) -> None:
        """Close the persistent socket if open."""
        with self._lock:
            self._reset()

    def get_runtime_config(self) -> dict[str, Any]:
        """Return DaseR runtime configuration owned by the server.

        Returns:
            Runtime config dict containing store_path, slot_size, block_tokens,
            and model_id.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other scheduler
            calls.
        """
        resp = self.call({"op": "get_runtime_config"})
        config = resp.get("runtime_config", {})
        if not isinstance(config, dict):
            raise RuntimeError("[IPC] invalid runtime_config response")
        return config

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

    def release_chunks(self, chunk_keys: list[str]) -> None:
        """Release server-side lookup/load pins for chunks.

        Args:
            chunk_keys: chunk keys to release.
        """
        self.call({"op": "release_chunks", "chunk_keys": chunk_keys})

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
            writer.write(pack_frame(payload))
            await writer.drain()
            result = await read_frame(reader)
        finally:
            writer.close()
            await writer.wait_closed()

        if "error" in result:
            raise RuntimeError(f"[IPC] server error: {result['error']}")
        return result

    async def commit_chunk(self, chunk_key: str) -> None:
        """Async: mark a chunk as committed.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
        """
        await self.call({"op": "commit_chunk", "chunk_key": chunk_key})

    async def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str
    ) -> dict[str, Any]:
        """Async: allocate an L2 ring slot for a new chunk.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
            token_count: number of tokens in the chunk.
            model_id: model identifier.

        Returns:
            Dict with start_slot, file_offset, and pos_offset.
        """
        return await self.call(
            {
                "op": "alloc_chunk",
                "chunk_key": chunk_key,
                "token_count": token_count,
                "model_id": model_id,
            }
        )

    async def commit_l1(self, chunk_key: str) -> None:
        """Async: publish a chunk available in worker L1 memory.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
        """
        await self.call({"op": "commit_l1", "chunk_key": chunk_key})

    async def commit_l2(self, chunk_key: str) -> None:
        """Async: mark a chunk's SSD L2 copy durable.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
        """
        await self.call({"op": "commit_l2", "chunk_key": chunk_key})

    async def release_chunks(self, chunk_keys: list[str]) -> None:
        """Async: release server-side lookup/load pins.

        Args:
            chunk_keys: chunk keys to release.
        """
        await self.call({"op": "release_chunks", "chunk_keys": chunk_keys})

    async def evict_l1(self, chunk_key: str) -> None:
        """Async: notify server that worker evicted a chunk from L1.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
        """
        await self.call({"op": "evict_l1", "chunk_key": chunk_key})
