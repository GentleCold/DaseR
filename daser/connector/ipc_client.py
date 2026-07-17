# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import contextlib
import socket
import threading
from typing import Any

# First Party
from daser.ipc_protocol import pack_frame, read_frame, recv_frame
from daser.logging import init_logger

logger = init_logger(__name__)


def _raise_on_error(result: dict[str, Any]) -> dict[str, Any]:
    """Raise when the server returned an error frame, else return the result.

    Args:
        result: decoded response frame from the server.

    Returns:
        The same ``result`` dict when it carries no ``error`` field.

    Raises:
        RuntimeError: if the response contains an ``error`` field.
    """
    if "error" in result:
        raise RuntimeError(f"[IPC] server error: {result['error']}")
    return result


class _IPCClientBase:
    """Shared connection policy for the sync and async IPC clients.

    Holds the socket path and the common retry/error contract. Concrete
    subclasses provide the transport-specific ``call`` (blocking socket vs
    asyncio streams); both reset their connection and retry once on transport
    failure so a restarted server does not wedge the client.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    #: Number of attempts (one retry) before surfacing a transport failure.
    _MAX_ATTEMPTS = 2

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path


class IPCClientSync(_IPCClientBase):
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
        super().__init__(socket_path)
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
            for attempt in range(self._MAX_ATTEMPTS):
                if self._sock is None:
                    self._sock = self._connect()
                try:
                    self._sock.sendall(raw)
                    result = recv_frame(self._sock)
                    break
                except (ConnectionError, OSError, BrokenPipeError) as exc:
                    self._reset()
                    if attempt == self._MAX_ATTEMPTS - 1:
                        raise RuntimeError(f"[IPC] transport failure: {exc}") from exc
        return _raise_on_error(result)

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

    def lookup(
        self,
        tokens: list[int],
        model_id: str,
        external_prefix_queries: int | None = None,
        num_computed_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        """Look up cached chunks for the given token sequence.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            external_prefix_queries: optional vLLM external prefix query token
                count to record on the server using the same lookup result.
            num_computed_tokens: tokens already computed locally by vLLM.

        Returns:
            List of chunk dicts (may be empty).
        """
        payload: dict[str, Any] = {
            "op": "lookup",
            "tokens": tokens,
            "model_id": model_id,
        }
        if external_prefix_queries is not None:
            payload["external_prefix_queries"] = int(external_prefix_queries)
            payload["num_computed_tokens"] = int(num_computed_tokens)
        resp = self.call(payload)
        return resp.get("chunks", [])

    def record_external_prefix_cache(self, queries: int, hits: int) -> None:
        """Record vLLM-equivalent external prefix cache counters.

        Args:
            queries: Number of queried prompt tokens in vLLM's external prefix
                cache accounting.
            hits: Number of queried tokens accepted as external prefix hits.

        Returns:
            None.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other scheduler
            calls.
        """
        self.call(
            {
                "op": "record_external_prefix_cache",
                "queries": int(queries),
                "hits": int(hits),
            }
        )

    def alloc_chunk(
        self, chunk_key: str, token_count: int, model_id: str
    ) -> dict[str, Any]:
        """Allocate a slot for a new chunk.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
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

    def alloc_chunks(
        self,
        chunks: list[dict[str, Any]],
        model_id: str,
    ) -> list[dict[str, Any]]:
        """Allocate slots for multiple chunks in one IPC call.

        Args:
            chunks: chunk descriptors with chunk_key and token_count.
            model_id: model identifier.

        Returns:
            List of allocation dicts with chunk_key, start_slot, file_offset,
            and pos_offset.
        """
        resp = self.call(
            {
                "op": "alloc_chunks",
                "chunks": chunks,
                "model_id": model_id,
            }
        )
        allocations = resp.get("allocations", [])
        if not isinstance(allocations, list):
            raise RuntimeError("[IPC] invalid alloc_chunks response")
        return [dict(alloc) for alloc in allocations]

    def commit_chunk(self, chunk_key: str, tp_rank: int = 0, tp_size: int = 1) -> None:
        """Mark a chunk as committed (GDS write complete).

        Args:
            chunk_key: xxh3_128 hex of the chunk's token IDs.
            tp_rank: tensor-parallel rank whose shard was stored.
            tp_size: total tensor-parallel ranks required for publication.
        """
        self.call(
            {
                "op": "commit_chunk",
                "chunk_key": chunk_key,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        )

    def transfer_drain(self) -> None:
        """Wait for server-owned transfer-layer background work.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other scheduler
            calls.
        """
        self.call({"op": "transfer_drain"})

    def transfer_prefetch(self, spans: list[dict[str, int]]) -> dict[str, int]:
        """Synchronously promote storage spans into the host-memory tier.

        Args:
            spans: Storage spans containing ``file_offset`` and ``nbytes``.

        Returns:
            Requested, L1-resident, and L2-read byte counts.

        Thread-safety:
            Intended for a dedicated scheduler prefetch thread because the RPC
            waits for all required L2 reads.
        """
        response = self.call({"op": "transfer_prefetch", "spans": spans})
        fields = ("requested_bytes", "l1_bytes", "l2_bytes")
        if any(field not in response for field in fields):
            raise RuntimeError("[IPC] invalid transfer_prefetch response")
        return {field: int(response[field]) for field in fields}

    def commit_stats(self) -> dict[str, int]:
        """Return server-side connector commit counters.

        Returns:
            Dict containing processed commit counters.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other scheduler
            calls.
        """
        resp = self.call({"op": "commit_stats"})
        stats = resp.get("commit_stats", {})
        if not isinstance(stats, dict):
            raise RuntimeError("[IPC] invalid commit_stats response")
        return {str(k): int(v) for k, v in stats.items()}

    def live_allocations(self, allocations: list[dict[str, int | str]]) -> set[str]:
        """Return chunk keys that still own their server allocation.

        Args:
            allocations: Dicts with chunk_key, start_slot, and num_slots.

        Returns:
            Set of live chunk keys.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other scheduler
            calls.
        """
        resp = self.call({"op": "live_allocations", "allocations": allocations})
        chunk_keys = resp.get("chunk_keys", [])
        if not isinstance(chunk_keys, list):
            raise RuntimeError("[IPC] invalid live_allocations response")
        return {str(key) for key in chunk_keys}

    def evict_chunk(self, chunk_key: str) -> None:
        """Evict a chunk from the DaseR index.

        Args:
            chunk_key: xxh3_128 hex of the chunk's token IDs.
        """
        self.call({"op": "evict_chunk", "chunk_key": chunk_key})

    def release_chunk_writer(
        self,
        chunk_key: str,
        start_slot: int,
        num_slots: int,
    ) -> None:
        """Release an uncommitted store writer claim.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
            start_slot: first allocated slot for the pending store.
            num_slots: number of slots allocated for the pending store.
        """
        self.call(
            {
                "op": "release_chunk_writer",
                "chunk_key": chunk_key,
                "start_slot": start_slot,
                "num_slots": num_slots,
            }
        )


class IPCClientAsync(_IPCClientBase):
    """Asyncio IPC client for worker-side calls.

    Args:
        socket_path: Unix socket path of the DaseR server.
    """

    def __init__(self, socket_path: str) -> None:
        super().__init__(socket_path)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Return a persistent async connection, opening it on first use."""
        if self._reader is None or self._writer is None or self._writer.is_closing():
            self._reader, self._writer = await asyncio.open_unix_connection(self._path)
        return self._reader, self._writer

    async def _reset(self) -> None:
        """Close the persistent async connection if it is open."""
        if self._writer is not None:
            with contextlib.suppress(ConnectionError, OSError):
                self._writer.close()
                await self._writer.wait_closed()
        self._reader = None
        self._writer = None

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request asynchronously and return the response.

        Args:
            payload: dict with "op" and any required fields.

        Returns:
            Response dict from the server.

        Raises:
            RuntimeError: if the server returns an error response.
        """
        raw = pack_frame(payload)
        async with self._lock:
            for attempt in range(self._MAX_ATTEMPTS):
                try:
                    reader, writer = await self._connect()
                    writer.write(raw)
                    await writer.drain()
                    result = await read_frame(reader)
                    break
                except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
                    await self._reset()
                    if attempt == self._MAX_ATTEMPTS - 1:
                        raise RuntimeError(f"[IPC] transport failure: {exc}") from exc

        return _raise_on_error(result)

    async def close(self) -> None:
        """Close the persistent async connection."""
        async with self._lock:
            await self._reset()

    async def commit_chunk(
        self, chunk_key: str, tp_rank: int = 0, tp_size: int = 1
    ) -> None:
        """Async: mark a chunk as committed.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
            tp_rank: tensor-parallel rank whose shard was stored.
            tp_size: total tensor-parallel ranks required for publication.
        """
        await self.call(
            {
                "op": "commit_chunk",
                "chunk_key": chunk_key,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        )

    async def commit_chunks(
        self, chunk_keys: list[str], tp_rank: int = 0, tp_size: int = 1
    ) -> None:
        """Async: mark multiple chunks as committed in one RPC.

        Args:
            chunk_keys: xxh3_128 hex chunk keys.
            tp_rank: tensor-parallel rank whose shards were stored.
            tp_size: total tensor-parallel ranks required for publication.
        """
        await self.call(
            {
                "op": "commit_chunks",
                "chunk_keys": chunk_keys,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        )

    async def transfer_drain(self) -> None:
        """Async: wait for server-owned transfer-layer background work.

        Async/thread-safety:
            Serializes with other calls on the persistent async connection.
        """
        await self.call({"op": "transfer_drain"})

    async def init_transfer(self) -> None:
        """Async: initialize the server-owned transfer layer.

        Async/thread-safety:
            Serializes with other calls on the persistent async connection.
        """
        await self.call({"op": "init_transfer"})

    async def transfer_store_bytes(
        self,
        data: bytes,
        spans: list[dict[str, int]],
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> list[str]:
        """Store bytes through the server-owned transfer layer.

        Args:
            data: source bytes.
            spans: byte spans containing source_offset, nbytes, and file_offset.
            tp_rank: tensor-parallel rank that owns the stored shard.
            tp_size: total tensor-parallel ranks required before publication.

        Async/thread-safety:
            Opens a short-lived async IPC connection for this request.
        """
        resp = await self.call(
            {
                "op": "transfer_store",
                "payload": {"data": data},
                "spans": spans,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        )
        chunk_keys = resp.get("chunk_keys", [])
        if not isinstance(chunk_keys, list):
            raise RuntimeError("[IPC] invalid transfer_store chunk_keys response")
        return [str(key) for key in chunk_keys]

    async def transfer_load_bytes(self, spans: list[dict[str, int]]) -> bytes:
        """Load bytes through the server-owned transfer layer.

        Args:
            spans: byte spans containing target_offset, nbytes, and file_offset.

        Returns:
            Loaded bytes in target-offset order.
        """
        resp = await self.call(
            {
                "op": "transfer_load",
                "payload": {"return_data": True},
                "spans": spans,
            }
        )
        data = resp.get("data", b"")
        if not isinstance(data, bytes):
            raise RuntimeError("[IPC] invalid transfer_load data response")
        return data

    async def transfer_store_cuda(
        self,
        cuda_ipc_handle: bytes,
        nbytes: int,
        device_id: int,
        device_ptr: int,
        allocation_base_ptr: int,
        allocation_offset: int,
        producer_pid: int,
        spans: list[dict[str, Any]],
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> list[str]:
        """Store from a CUDA IPC buffer through the server transfer layer.

        Args:
            cuda_ipc_handle: exported CUDA IPC memory handle.
            nbytes: byte size of the exported allocation.
            device_id: CUDA device ordinal for the exported allocation.
            device_ptr: raw device pointer for same-process server harnesses.
            allocation_base_ptr: base pointer of the CUDA allocation owning
                ``device_ptr``.
            allocation_offset: byte offset of ``device_ptr`` from
                ``allocation_base_ptr``.
            producer_pid: process ID that exported the pointer.
            spans: byte spans containing source_offset, nbytes, and file_offset.
            tp_rank: tensor-parallel rank that owns the stored shard.
            tp_size: total tensor-parallel ranks required before publication.
        """
        resp = await self.call(
            {
                "op": "transfer_store",
                "payload": {
                    "cuda_ipc_handle": cuda_ipc_handle,
                    "nbytes": nbytes,
                    "device_id": device_id,
                    "device_ptr": device_ptr,
                    "allocation_base_ptr": allocation_base_ptr,
                    "allocation_offset": allocation_offset,
                    "producer_pid": producer_pid,
                },
                "spans": spans,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        )
        chunk_keys = resp.get("chunk_keys", [])
        if not isinstance(chunk_keys, list):
            raise RuntimeError("[IPC] invalid transfer_store chunk_keys response")
        return [str(key) for key in chunk_keys]

    async def transfer_load_cuda(
        self,
        cuda_ipc_handle: bytes,
        nbytes: int,
        device_id: int,
        device_ptr: int,
        allocation_base_ptr: int,
        allocation_offset: int,
        producer_pid: int,
        spans: list[dict[str, int]],
    ) -> dict[str, Any]:
        """Load into a CUDA IPC buffer through the server transfer layer.

        Args:
            cuda_ipc_handle: exported CUDA IPC memory handle.
            nbytes: byte size of the exported allocation.
            device_id: CUDA device ordinal for the exported allocation.
            device_ptr: raw device pointer for same-process server harnesses.
            allocation_base_ptr: base pointer of the CUDA allocation owning
                ``device_ptr``.
            allocation_offset: byte offset of ``device_ptr`` from
                ``allocation_base_ptr``.
            producer_pid: process ID that exported the pointer.
            spans: byte spans containing target_offset, nbytes, and file_offset.

        Returns:
            Server response including transferred bytes and optional timing
            counters.
        """
        return await self.call(
            {
                "op": "transfer_load",
                "payload": {
                    "cuda_ipc_handle": cuda_ipc_handle,
                    "nbytes": nbytes,
                    "device_id": device_id,
                    "device_ptr": device_ptr,
                    "allocation_base_ptr": allocation_base_ptr,
                    "allocation_offset": allocation_offset,
                    "producer_pid": producer_pid,
                },
                "spans": spans,
            }
        )

    async def register_load_staging_cuda(
        self,
        buffer_index: int,
        cuda_ipc_handle: bytes,
        allocation_bytes: int,
        device_id: int,
        device_ptr: int,
        allocation_base_ptr: int,
        allocation_offset: int,
        producer_pid: int,
    ) -> None:
        """Register one fixed load staging CUDA allocation with the server.

        Args:
            buffer_index: Worker-local fixed staging buffer index.
            cuda_ipc_handle: exported CUDA IPC memory handle.
            allocation_bytes: byte size of the CUDA allocation to map.
            device_id: CUDA device ordinal for the exported allocation.
            device_ptr: raw device pointer for same-process server harnesses.
            allocation_base_ptr: base pointer of the CUDA allocation owning
                ``device_ptr``.
            allocation_offset: byte offset of ``device_ptr`` from
                ``allocation_base_ptr``.
            producer_pid: process ID that exported the pointer.

        Async/thread-safety:
            Serializes with other calls on the dedicated async client
            connection. Intended for worker initialization before hot-path
            cache-hit loads.
        """
        await self.call(
            {
                "op": "register_load_staging",
                "payload": {
                    "buffer_index": int(buffer_index),
                    "cuda_ipc_handle": cuda_ipc_handle,
                    "allocation_bytes": int(allocation_bytes),
                    "device_id": int(device_id),
                    "device_ptr": int(device_ptr),
                    "allocation_base_ptr": int(allocation_base_ptr),
                    "allocation_offset": int(allocation_offset),
                    "producer_pid": int(producer_pid),
                },
            }
        )

    async def transfer_load_registered_cuda(
        self,
        buffer_index: int,
        producer_pid: int,
        nbytes: int,
        spans: list[dict[str, int]],
    ) -> dict[str, Any]:
        """Load into a previously registered fixed CUDA staging buffer.

        Args:
            buffer_index: Worker-local fixed staging buffer index registered
                through ``register_load_staging_cuda``.
            producer_pid: Process ID that registered the staging buffer.
            nbytes: logical bytes to write for this transfer.
            spans: byte spans containing target_offset, nbytes, and file_offset.

        Returns:
            Server response including transferred bytes and timing counters.

        Async/thread-safety:
            Runs on the worker load event loop and avoids per-load CUDA IPC
            handle export/open payloads on the hot path.
        """
        return await self.call(
            {
                "op": "transfer_load",
                "payload": {
                    "load_staging_buffer_index": int(buffer_index),
                    "producer_pid": int(producer_pid),
                    "nbytes": int(nbytes),
                },
                "spans": spans,
            }
        )
