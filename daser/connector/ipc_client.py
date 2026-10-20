# SPDX-License-Identifier: Apache-2.0

# Standard
import array
import asyncio
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import socket
import threading
from typing import Any, Literal

# First Party
from daser.ipc_protocol import exchange_frame, pack_frame, recv_frame
from daser.logging import init_logger

logger = init_logger(__name__)


def _pack_lookup_tokens(tokens: list[int]) -> bytes:
    """Pack lookup IDs once for compact IPC and server-side hashing.

    Args:
        tokens: Prompt token IDs.

    Returns:
        Native signed-int bytes consumed directly by retrieval indexes.
    """
    return bytes(array.array("i", tokens))


@dataclass(frozen=True)
class PrefetchLookupResult:
    """Validated scheduler lookup result with exact host-tier admission state."""

    chunks: list[dict[str, Any]]
    spans: list[dict[str, int]]
    tier: Literal["l1", "mixed", "l2"] | None


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
        timeout: Per-call socket timeout in seconds; ``None`` waits for the
            server indefinitely. A timed-out RPC may still be running on the
            server, so only callers whose RPCs are bounded should set one.
    """

    def __init__(self, socket_path: str, timeout: float | None = 30.0) -> None:
        super().__init__(socket_path)
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._timeout = timeout

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
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
            RuntimeError: if the server does not respond within the client
                timeout (reported as a transport failure after one retry).
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
        defer_pending: bool | None = None,
    ) -> list[dict[str, Any]] | None:
        """Look up cached chunks for the given token sequence.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            external_prefix_queries: optional vLLM external prefix query token
                count to record on the server using the same lookup result.
            num_computed_tokens: tokens already computed locally by vLLM.
            defer_pending: None keeps the server's pending-writer wait. True
                asks the server not to wait but to report a lookup that a
                pending writer may extend; False asks for an immediate answer.

        Returns:
            List of chunk dicts (may be empty), or None when ``defer_pending``
            is True and the server reported a pending writer.
        """
        chunks, _ = self.lookup_versioned(
            tokens,
            model_id,
            external_prefix_queries=external_prefix_queries,
            num_computed_tokens=num_computed_tokens,
            defer_pending=defer_pending,
        )
        return chunks

    def lookup_versioned(
        self,
        tokens: list[int],
        model_id: str,
        external_prefix_queries: int | None = None,
        num_computed_tokens: int = 0,
        defer_pending: bool | None = None,
    ) -> tuple[list[dict[str, Any]] | None, int | None]:
        """Look up cached chunks and the index epoch the result belongs to.

        Args:
            tokens: prompt token IDs.
            model_id: model identifier.
            external_prefix_queries: see :meth:`lookup`.
            num_computed_tokens: see :meth:`lookup`.
            defer_pending: see :meth:`lookup`.

        Returns:
            ``(chunks, epoch)``. ``chunks`` is as returned by :meth:`lookup`;
            ``epoch`` is the server index epoch read before the lookup, or
            None when the server deferred the lookup. While
            :meth:`index_epoch` still returns this value, no chunk in
            ``chunks`` has been removed from the index.
        """
        payload: dict[str, Any] = {
            "op": "lookup",
            "token_bytes": _pack_lookup_tokens(tokens),
            "model_id": model_id,
        }
        if external_prefix_queries is not None:
            payload["external_prefix_queries"] = int(external_prefix_queries)
            payload["num_computed_tokens"] = int(num_computed_tokens)
        if defer_pending is not None:
            payload["defer_pending"] = bool(defer_pending)
        resp = self.call(payload)
        if resp.get("pending"):
            return None, None
        epoch = resp.get("index_epoch")
        return resp.get("chunks", []), None if epoch is None else int(epoch)

    def index_epoch(self) -> int:
        """Return the server index epoch, which advances on every removal.

        Returns:
            Current epoch; compare with the one from :meth:`lookup_versioned`.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other calls; the
            request carries no tokens and the server answers without a lookup.
        """
        return int(self.call({"op": "index_epoch"})["index_epoch"])

    def lookup_with_prefetch(
        self,
        lease_id: str,
        tokens: list[int],
        model_id: str,
        external_prefix_queries: int,
        num_computed_tokens: int = 0,
    ) -> PrefetchLookupResult:
        """Look up chunks and lease the L1 copies of their external window.

        Args:
            lease_id: Base vLLM request ID naming the server lease.
            tokens: Prompt token IDs sent to the retrieval index.
            model_id: Model identifier.
            external_prefix_queries: vLLM external-prefix query token count.
            num_computed_tokens: Tokens already resident in vLLM's local cache.

        Returns:
            Validated chunks, the window's ``{start_slot, num_slots, nbytes}``
            spans (``nbytes`` per rank), and the tier classification:
            ``l1`` when every leased chunk already has an L1 copy, else
            ``mixed`` or ``l2`` (``transfer_prefetch`` promotes the rest).

        Thread-safety:
            Uses the lock-protected scheduler IPC connection. The server makes
            classification and lease acquisition atomic.
        """
        response = self.call(
            {
                "op": "lookup_prefetch",
                "lease_id": lease_id,
                "token_bytes": _pack_lookup_tokens(tokens),
                "model_id": model_id,
                "external_prefix_queries": int(external_prefix_queries),
                "num_computed_tokens": int(num_computed_tokens),
            }
        )
        chunks = response.get("chunks", [])
        spans = response.get("spans", [])
        tier = response.get("tier")
        if not isinstance(chunks, list) or not all(
            isinstance(chunk, dict) for chunk in chunks
        ):
            raise RuntimeError("[IPC] invalid lookup_prefetch chunks response")
        if not isinstance(spans, list) or not all(
            isinstance(span, dict)
            and "start_slot" in span
            and "num_slots" in span
            and "nbytes" in span
            for span in spans
        ):
            raise RuntimeError("[IPC] invalid lookup_prefetch spans response")
        if tier not in (None, "l1", "mixed", "l2"):
            raise RuntimeError("[IPC] invalid lookup_prefetch tier response")
        if bool(spans) != (tier is not None):
            raise RuntimeError("[IPC] incomplete lookup_prefetch response")
        if not chunks and spans:
            raise RuntimeError(
                "[IPC] empty lookup_prefetch response has transfer state"
            )
        return PrefetchLookupResult(
            chunks=[dict(chunk) for chunk in chunks],
            spans=[
                {
                    "start_slot": int(span["start_slot"]),
                    "num_slots": int(span["num_slots"]),
                    "nbytes": int(span["nbytes"]),
                }
                for span in spans
            ],
            tier=tier,
        )

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
        self, chunk_key: str, token_count: int, model_id: str, parent_key: str = ""
    ) -> dict[str, Any]:
        """Allocate a slot for a new chunk.

        Args:
            chunk_key: xxh3_128 hex of the token IDs.
            token_count: number of tokens in the chunk.
            model_id: model identifier.
            parent_key: previous chunk of the prefix chain, or empty.

        Returns:
            Dict with start_slot, num_slots, pos_offset and skipped.
        """
        return self.call(
            {
                "op": "alloc_chunk",
                "chunk_key": chunk_key,
                "token_count": token_count,
                "model_id": model_id,
                "parent_key": parent_key,
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
            List of allocation dicts with chunk_key, start_slot, num_slots,
            token_count, pos_offset and skipped.
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

    def transfer_drain(self) -> None:
        """Wait for server-owned transfer-layer background work.

        Thread-safety:
            Uses the same lock-protected blocking RPC path as other scheduler
            calls.
        """
        self.call({"op": "transfer_drain"})

    def transfer_prefetch(self, lease_id: str) -> dict[str, int]:
        """Synchronously bring a lease's chunks into the host-memory tier.

        Args:
            lease_id: Request lease from ``lookup_with_prefetch``.

        Returns:
            Requested, already-L1 and promoted (L2-read) byte counts.

        Raises:
            RuntimeError: If the lease is gone or a chunk could not be brought
                into L1; the caller releases the lease and loads without it.

        Thread-safety:
            Intended for a dedicated scheduler prefetch thread because the RPC
            waits for the lease budget and every promotion read.
        """
        response = self.call({"op": "transfer_prefetch", "lease_id": lease_id})
        fields = ("requested_bytes", "l1_bytes", "l2_bytes")
        if any(field not in response for field in fields):
            raise RuntimeError("[IPC] invalid transfer_prefetch response")
        return {field: int(response[field]) for field in fields}

    def release_transfer_lease(self, lease_id: str) -> None:
        """Idempotently release the L1 pins of one request lease.

        Args:
            lease_id: Base vLLM request ID to clean up.

        Returns:
            None.

        Thread-safety:
            Uses the lock-protected scheduler IPC connection and may race with
            worker load completion safely.
        """
        self.call({"op": "release_transfer_lease", "lease_id": lease_id})

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
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            with contextlib.suppress(ConnectionError, OSError):
                writer.close()
                await writer.wait_closed()

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request asynchronously and return the response.

        Args:
            payload: dict with "op" and any required fields.

        Returns:
            Response dict from the server.

        Raises:
            RuntimeError: if the server returns an error response.
            asyncio.CancelledError: After the submitted request's reply is
                drained or the transport fails.

        Async/thread-safety:
            Serializes complete exchanges on the persistent connection. A task
            cancelled while waiting for the lock sends nothing. Once submitted,
            repeated cancellation is deferred until the response or transport
            failure; the cancelled exchange is never replayed.
        """
        raw = pack_frame(payload)
        async with self._lock:
            for attempt in range(self._MAX_ATTEMPTS):
                try:
                    reader, writer = await self._connect()
                    result = await exchange_frame(reader, writer, raw)
                    break
                except asyncio.CancelledError:
                    # Retire the cancelled exchange's transport before another
                    # caller acquires the lock. _reset clears references before
                    # awaiting close, so another cancellation cannot expose a
                    # half-retired stream to the next call.
                    await self._reset()
                    raise
                except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
                    await self._reset()
                    if attempt == self._MAX_ATTEMPTS - 1:
                        raise RuntimeError(f"[IPC] transport failure: {exc}") from exc

        return _raise_on_error(result)

    async def close(self) -> None:
        """Close the persistent async connection."""
        async with self._lock:
            await self._reset()

    async def transfer_drain(self) -> None:
        """Async: wait for server-owned transfer-layer background work.

        Async/thread-safety:
            Serializes with other calls on the persistent async connection.
        """
        await self.call({"op": "transfer_drain"})

    async def release_chunk_writer(
        self, chunk_key: str, start_slot: int, num_slots: int
    ) -> None:
        """Release a canceled writer only for its original allocation identity.

        Args:
            chunk_key: Hash of the pending chunk's token IDs.
            start_slot: First logical slot in the original allocation.
            num_slots: Number of slots in that allocation.
        Returns:
            None; server or transport failures propagate to the caller.
        Async/thread-safety:
            Awaits the existing public RPC on the client's owning event loop.
        """
        await self.call(
            {
                "op": "release_chunk_writer",
                "chunk_key": chunk_key,
                "start_slot": start_slot,
                "num_slots": num_slots,
            }
        )

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
            spans: store spans ``{source_offset, nbytes, chunk_key,
                start_slot, num_slots, slot_index, packed, mode}``.
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

    async def transfer_load_bytes(
        self,
        spans: list[dict[str, int]],
        tp_rank: int = 0,
    ) -> bytes:
        """Load bytes through the server-owned transfer layer.

        Args:
            spans: load spans ``{target_offset, start_slot, num_slots,
                nbytes}``.
            tp_rank: tensor-parallel rank whose lane is read.

        Returns:
            Loaded bytes in target-offset order.
        """
        resp = await self.call(
            {
                "op": "transfer_load",
                "payload": {"return_data": True},
                "spans": spans,
                "tp_rank": tp_rank,
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
        device_pci_bus_id: str,
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
            device_pci_bus_id: PCI bus ID of the GPU holding the exported
                allocation; device ordinals differ between processes.
            device_ptr: raw device pointer for same-process server harnesses.
            allocation_base_ptr: base pointer of the CUDA allocation owning
                ``device_ptr``.
            allocation_offset: byte offset of ``device_ptr`` from
                ``allocation_base_ptr``.
            producer_pid: process ID that exported the pointer.
            spans: store spans ``{source_offset, nbytes, chunk_key,
                start_slot, num_slots, slot_index, packed, mode}``.
            tp_rank: tensor-parallel rank that owns the stored shard.
            tp_size: total tensor-parallel ranks required before publication.
        """
        resp = await self.call(
            {
                "op": "transfer_store",
                "payload": {
                    "cuda_ipc_handle": cuda_ipc_handle,
                    "nbytes": nbytes,
                    "device_pci_bus_id": device_pci_bus_id,
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
        device_pci_bus_id: str,
        device_ptr: int,
        allocation_base_ptr: int,
        allocation_offset: int,
        producer_pid: int,
        spans: list[dict[str, int]],
        tp_rank: int = 0,
    ) -> dict[str, Any]:
        """Load into a CUDA IPC buffer through the server transfer layer.

        Args:
            cuda_ipc_handle: exported CUDA IPC memory handle.
            nbytes: byte size of the exported allocation.
            device_pci_bus_id: PCI bus ID of the GPU holding the exported
                allocation; device ordinals differ between processes.
            device_ptr: raw device pointer for same-process server harnesses.
            allocation_base_ptr: base pointer of the CUDA allocation owning
                ``device_ptr``.
            allocation_offset: byte offset of ``device_ptr`` from
                ``allocation_base_ptr``.
            producer_pid: process ID that exported the pointer.
            spans: load spans ``{target_offset, start_slot, num_slots,
                nbytes}``.
            tp_rank: tensor-parallel rank whose lane is read.

        Returns:
            Server response including transferred bytes, ``l1_segments`` /
            ``l2_segments`` and timing counters.
        """
        request: dict[str, Any] = {
            "op": "transfer_load",
            "payload": {
                "cuda_ipc_handle": cuda_ipc_handle,
                "nbytes": nbytes,
                "device_pci_bus_id": device_pci_bus_id,
                "device_ptr": device_ptr,
                "allocation_base_ptr": allocation_base_ptr,
                "allocation_offset": allocation_offset,
                "producer_pid": producer_pid,
            },
            "spans": spans,
            "tp_rank": tp_rank,
        }
        return await self.call(request)

    async def transfer_load_runs(
        self,
        buffer_index: int,
        producer_pid: int,
        start_slots: Any,
        intra_offsets: Any,
        target_offsets: Any,
        nbytes: Any,
        lease_ids: Sequence[str],
        tp_rank: int = 0,
    ) -> dict[str, Any]:
        """Copy runs of leased L1 records into a registered CUDA staging buffer.

        Args:
            buffer_index: Worker-local registered load staging buffer index.
            producer_pid: Process ID that registered the staging buffer.
            start_slots: Slot id of each run's record (int64-convertible).
            intra_offsets: Byte offset of each run inside its record.
            target_offsets: Destination offsets inside the staging buffer.
            nbytes: Run lengths.
            lease_ids: Request leases from ``lookup_prefetch``; together
                they pin every named slot in L1.
            tp_rank: tensor-parallel rank whose lane is read.

        Returns:
            Server response with ``bytes`` and ``transfer_ms`` after the
            copies completed on the device.

        Async/thread-safety:
            Serializes with other calls on this client's connection; use one
            client per staging buffer to overlap groups.
        """
        import numpy

        runs = numpy.stack(
            [
                numpy.asarray(start_slots, dtype=numpy.int64),
                numpy.asarray(intra_offsets, dtype=numpy.int64),
                numpy.asarray(target_offsets, dtype=numpy.int64),
                numpy.asarray(nbytes, dtype=numpy.int64),
            ]
        )
        return await self.call(
            {
                "op": "transfer_load_runs",
                "payload": {
                    "load_staging_buffer_index": int(buffer_index),
                    "producer_pid": int(producer_pid),
                },
                "lease_ids": [str(lease_id) for lease_id in lease_ids],
                "tp_rank": int(tp_rank),
                "runs": runs.tobytes(),
            }
        )

    async def release_transfer_lease(self, lease_id: str) -> dict[str, Any]:
        """Release the L1 pins of one request lease.

        Args:
            lease_id: Request lease from ``lookup_prefetch``.

        Returns:
            Server response.

        Async/thread-safety:
            Serializes with other calls on this client's connection.
        """
        return await self.call({"op": "release_transfer_lease", "lease_id": lease_id})

    async def register_staging_cuda(
        self,
        direction: str,
        buffer_index: int,
        cuda_ipc_handle: bytes,
        allocation_bytes: int,
        device_pci_bus_id: str,
        device_ptr: int,
        allocation_base_ptr: int,
        allocation_offset: int,
        producer_pid: int,
    ) -> None:
        """Register one fixed staging CUDA allocation with the server.

        Args:
            direction: Staging ownership, either ``load`` or ``store``.
            buffer_index: Worker-local fixed staging buffer index.
            cuda_ipc_handle: exported CUDA IPC memory handle.
            allocation_bytes: byte size of the CUDA allocation to map.
            device_pci_bus_id: PCI bus ID of the GPU holding the exported
                allocation; device ordinals differ between processes.
            device_ptr: raw device pointer for same-process server harnesses.
            allocation_base_ptr: base pointer of the CUDA allocation owning
                ``device_ptr``.
            allocation_offset: byte offset of ``device_ptr`` from
                ``allocation_base_ptr``.
            producer_pid: process ID that exported the pointer.

        Async/thread-safety:
            Serializes with other calls on the dedicated async client
            connection. Intended for worker initialization before hot-path
            transfers. Invalid directions raise ValueError before sending.
        """
        if direction not in {"load", "store"}:
            raise ValueError("staging direction must be load or store")
        await self.call(
            {
                "op": f"register_{direction}_staging",
                "payload": {
                    "buffer_index": int(buffer_index),
                    "cuda_ipc_handle": cuda_ipc_handle,
                    "allocation_bytes": int(allocation_bytes),
                    "device_pci_bus_id": str(device_pci_bus_id),
                    "device_ptr": int(device_ptr),
                    "allocation_base_ptr": int(allocation_base_ptr),
                    "allocation_offset": int(allocation_offset),
                    "producer_pid": int(producer_pid),
                },
            }
        )

    async def transfer_store_registered_cuda(
        self,
        buffer_index: int,
        producer_pid: int,
        spans: list[dict[str, Any]],
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> list[str]:
        """Store completed regions from an initialized fixed staging buffer.

        Args:
            buffer_index: Worker-local registered store buffer index (int).
            producer_pid: Exporting worker process ID (int).
            spans: Region byte mappings (list[dict]); source offsets are
                relative to the full registered buffer, not a compact view.
            tp_rank: Rank owning this shard (int).
            tp_size: Number of ranks required to publish a chunk (int).

        Returns:
            Accepted chunk keys (list[str]).

        Async/thread-safety:
            Uses this client's serialized async connection. The worker must
            retain its staging lease until this call completes.
        """
        response = await self.call(
            {
                "op": "transfer_store",
                "payload": {
                    "store_staging_buffer_index": buffer_index,
                    "producer_pid": producer_pid,
                },
                "spans": spans,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        )
        chunk_keys = response.get("chunk_keys", [])
        if not isinstance(chunk_keys, list):
            raise RuntimeError("[IPC] invalid transfer_store chunk_keys response")
        return [str(key) for key in chunk_keys]

    async def transfer_load_registered_cuda(
        self,
        buffer_index: int,
        producer_pid: int,
        nbytes: int,
        spans: list[dict[str, int]],
        tp_rank: int = 0,
    ) -> dict[str, Any]:
        """Load into a previously registered fixed CUDA staging buffer.

        Args:
            buffer_index: Worker-local fixed staging buffer index registered
                through ``register_staging_cuda(direction="load")``.
            producer_pid: Process ID that registered the staging buffer.
            nbytes: logical bytes to write for this transfer.
            spans: load spans ``{target_offset, start_slot, num_slots,
                nbytes}``.
            tp_rank: tensor-parallel rank whose lane is read.

        Returns:
            Server response including transferred bytes, ``l1_segments`` /
            ``l2_segments`` and timing counters.

        Async/thread-safety:
            Runs on the worker load event loop and avoids per-load CUDA IPC
            handle export/open payloads on the hot path.
        """
        request: dict[str, Any] = {
            "op": "transfer_load",
            "payload": {
                "load_staging_buffer_index": int(buffer_index),
                "producer_pid": int(producer_pid),
                "nbytes": int(nbytes),
            },
            "spans": spans,
            "tp_rank": tp_rank,
        }
        return await self.call(request)
