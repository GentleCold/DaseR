# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections import OrderedDict
import contextlib
from dataclasses import asdict
import os
import threading
import time
from typing import Any

from daser.ipc_protocol import read_frame, write_frame

# First Party
from daser.logging import init_logger
from daser.server.core import ServerCore
from daser.transfer import TransferLayer
from daser.transfer.cuda_ipc import open_cuda_ipc_buffer
from daser.transfer.iouring import TieredIOUringTransferLayer

logger = init_logger(__name__)


_CUDA_IPC_CACHE_LIMIT = 16


def _coalesce_transfer_spans(spans: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Merge adjacent transfer spans without changing byte contents.

    Args:
        spans: transfer spans with source_offset, file_offset, and nbytes.

    Returns:
        Coalesced spans sorted by source and file offset.
    """
    normalized = [
        {
            "source_offset": int(span.get("source_offset", 0)),
            "file_offset": int(span["file_offset"]),
            "nbytes": int(span["nbytes"]),
        }
        for span in spans
        if int(span["nbytes"]) > 0
    ]
    normalized.sort(key=lambda span: (span["source_offset"], span["file_offset"]))
    if not normalized:
        return []

    merged = [normalized[0]]
    for span in normalized[1:]:
        prev = merged[-1]
        prev_source_end = prev["source_offset"] + prev["nbytes"]
        prev_file_end = prev["file_offset"] + prev["nbytes"]
        if (
            span["source_offset"] == prev_source_end
            and span["file_offset"] == prev_file_end
        ):
            prev["nbytes"] += span["nbytes"]
        else:
            merged.append(span)
    return merged


class IPCServer:
    """IPC server over Unix socket + msgpack.

    This server is the internal IPC interface for vLLM DaserConnector. It only
    exposes connector cache operations and delegates all business logic to
    ServerCore.

    Args:
        socket_path: Unix socket path.
        core: shared DaseR server core.
        runtime_config: connector runtime values returned by
            ``get_runtime_config``.

    Async/thread-safety:
        Must be started and stopped from the server asyncio event loop.
    """

    def __init__(
        self,
        socket_path: str,
        core: ServerCore,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._core = core
        self._runtime_config = runtime_config or {}
        self._server: asyncio.AbstractServer | None = None
        self._transfer: TransferLayer | None = None
        self._transfer_lock = threading.Lock()
        self._cuda_ipc_cache: OrderedDict[
            tuple[int, int, int, int | None], "_CachedCudaArray"
        ] = OrderedDict()

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

    async def initialize_transfer(self) -> None:
        """Eagerly create the transfer layer off the event loop.

        Offloads the blocking ``_ensure_transfer`` call (which may allocate
        pinned memory pools, open io_uring rings, etc.) to a thread so the
        server is fully provisioned before the first inference request.

        Async/thread-safety:
            Must be called from the server asyncio event loop during startup.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_transfer)

    async def stop(self) -> None:
        """Stop the server and remove the socket file.

        Async/thread-safety:
            Closes the asyncio server on the current event loop.
        """
        await self.stop_accepting()
        await self.close()
        logger.info("[IPC] server stopped")

    async def stop_accepting(self) -> None:
        """Stop accepting new IPC connections and remove the socket path.

        Async/thread-safety:
            Closes the asyncio listener on the current event loop. Existing
            transfer resources remain open until ``close`` is called.
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        logger.info("[IPC] server stopped accepting")

    async def close(self) -> None:
        """Drain and close transfer resources owned by the IPC server.

        Async/thread-safety:
            Runs on the server asyncio event loop after new IPC work has been
            rejected.
        """
        if self._transfer is not None:
            drain = getattr(self._transfer, "drain", None)
            if drain is not None:
                await drain()
            self._transfer.close()
            self._transfer = None
        self._close_cuda_ipc_cache()
        logger.info("[IPC] resources closed")

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
            with contextlib.suppress(Exception):
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
            if op == "get_runtime_config":
                return {"runtime_config": dict(self._runtime_config)}
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
            if op == "commit_chunks":
                for chunk_key in msg.get("chunk_keys", []):
                    await self._core.commit_chunk(chunk_key)
                return {"ok": True}
            if op == "commit_stats":
                return {"commit_stats": await self._core.commit_stats()}
            if op == "live_allocations":
                live = await self._core.live_allocations(
                    list(msg.get("allocations", []))
                )
                return {"chunk_keys": live}
            if op == "transfer_drain":
                transfer = self._transfer
                if transfer is not None:
                    drain = getattr(transfer, "drain", None)
                    if drain is not None:
                        await drain()
                return {"ok": True}
            if op == "init_transfer":
                self._ensure_transfer()
                return {"ok": True}
            if op == "transfer_store":
                return await self._transfer_store(msg)
            if op == "transfer_load":
                return await self._transfer_load(msg)
            if op == "evict_chunk":
                await self._core.evict_chunk(msg["chunk_key"])
                return {"ok": True}
            return {"error": f"unknown op: {op}"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] request failed: %s", exc)
            return {"error": str(exc)}

    async def _transfer_store(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Store one or more spans through the server-owned transfer layer.

        Args:
            msg: IPC request with ``payload`` and ``spans``.

        Returns:
            Response dict with total bytes stored.

        Async/thread-safety:
            Runs on the IPC event loop and awaits transfer-layer operations.
        """
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        transfer = self._ensure_transfer()
        total = 0
        stored_chunk_keys: list[str] = []
        buffer = self._payload_buffer(payload)
        try:
            live_spans: list[dict[str, Any]] = []
            for span in spans:
                nbytes = int(span["nbytes"])
                file_offset = int(span["file_offset"])
                chunk_key = str(span.get("chunk_key", ""))
                if chunk_key:
                    live = self._core.is_current_allocation(
                        chunk_key=chunk_key,
                        start_slot=int(span.get("start_slot", -1)),
                        num_slots=int(span.get("num_slots", 0)),
                    )
                    if not live:
                        logger.debug(
                            "[IPC] skip stale transfer_store key=%s offset=%d bytes=%d",
                            chunk_key[:8],
                            file_offset,
                            nbytes,
                        )
                        continue
                    stored_chunk_keys.append(chunk_key)
                live_spans.append(span)

            store_spans = (
                _coalesce_transfer_spans(live_spans)
                if bool(getattr(transfer, "coalesce_store_spans", False))
                else live_spans
            )
            grouped_store = getattr(transfer, "store_bytes_grouped", None)
            if grouped_store is not None:
                total = await grouped_store(buffer, store_spans)
            else:
                for span in store_spans:
                    source_offset = int(span.get("source_offset", 0))
                    nbytes = int(span["nbytes"])
                    file_offset = int(span["file_offset"])
                    src = buffer[source_offset : source_offset + nbytes]
                    total += await transfer.store_bytes(src, file_offset, nbytes)
        finally:
            if isinstance(buffer, _UncachedCudaArray):
                buffer.close()
        return {"ok": True, "bytes": total, "chunk_keys": stored_chunk_keys}

    async def _transfer_load(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Load one or more spans through the server-owned transfer layer.

        Args:
            msg: IPC request with ``payload`` and ``spans``.

        Returns:
            Response dict with total bytes loaded and optional bytes data.

        Async/thread-safety:
            Runs on the IPC event loop and awaits transfer-layer operations.
        """
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        transfer = self._ensure_transfer()
        total_size = self._payload_size(payload, spans)
        open_ms = 0.0
        if payload.get("return_data"):
            buffer: Any = bytearray(total_size)
        else:
            open_start = time.perf_counter()
            buffer = self._payload_buffer(payload)
            open_ms = (time.perf_counter() - open_start) * 1000

        total = 0
        load_ms = 0.0
        sync_ms = 0.0
        close_ms = 0.0
        try:
            stats_before = getattr(transfer, "stats", None)
            before = asdict(stats_before) if stats_before is not None else {}
            started = time.perf_counter()
            load_start = time.perf_counter()
            grouped_load = getattr(transfer, "load_bytes_grouped", None)
            if grouped_load is not None:
                total = await grouped_load(buffer, spans)
            else:
                for span in spans:
                    target_offset = int(span.get("target_offset", 0))
                    nbytes = int(span["nbytes"])
                    file_offset = int(span["file_offset"])
                    if isinstance(buffer, bytearray):
                        dst = memoryview(buffer)[target_offset : target_offset + nbytes]
                    else:
                        dst = buffer[target_offset : target_offset + nbytes]
                    total += await transfer.load_bytes(dst, file_offset, nbytes)
            load_ms = (time.perf_counter() - load_start) * 1000
            synchronize = getattr(buffer, "synchronize", None)
            if synchronize is not None:
                sync_start = time.perf_counter()
                synchronize()
                sync_ms = (time.perf_counter() - sync_start) * 1000
            elapsed_ms = (time.perf_counter() - started) * 1000
            stats_after = getattr(transfer, "stats", None)
            after = asdict(stats_after) if stats_after is not None else {}
            stats_delta = {
                key: int(after.get(key, 0)) - int(before.get(key, 0))
                for key in set(before) | set(after)
            }
            logger.debug(
                "[IPC] transfer_load timing: spans=%d bytes=%d total_size=%d "
                "open_ms=%.3f load_ms=%.3f sync_ms=%.3f elapsed_ms=%.3f "
                "stats_delta=%s",
                len(spans),
                total,
                total_size,
                open_ms,
                load_ms,
                sync_ms,
                elapsed_ms,
                stats_delta,
            )
            response: dict[str, Any] = {"ok": True, "bytes": total}
            if payload.get("return_data"):
                response["data"] = bytes(buffer)
            else:
                response["transfer_ms"] = elapsed_ms
                response["transfer_open_ms"] = open_ms
                response["transfer_load_ms"] = load_ms
                response["transfer_sync_ms"] = sync_ms
                response["transfer_stats_delta"] = stats_delta
            return response
        finally:
            if isinstance(buffer, _UncachedCudaArray):
                close = getattr(buffer, "close", None)
                close_start = time.perf_counter()
                close()
                close_ms = (time.perf_counter() - close_start) * 1000
                logger.info(
                    "[IPC] transfer_load close timing: bytes=%d close_ms=%.3f",
                    total,
                    close_ms,
                )

    def _ensure_transfer(self) -> TransferLayer:
        """Return the server-owned transfer layer, creating it on first use.

        Protected by a threading lock so that concurrent calls from the
        eager ``initialize_transfer`` thread-pool path and the event-loop
        IPC request path cannot create duplicate transfer layer instances.
        """
        if self._transfer is not None:
            return self._transfer
        with self._transfer_lock:
            if self._transfer is not None:
                return self._transfer
            mode = str(self._runtime_config.get("transfer_mode", "gds"))
            path = str(self._runtime_config.get("store_path", ""))
            if mode == "gds":
                from daser.transfer.gds import GDSTransferLayer

                self._transfer = GDSTransferLayer(path)
            elif mode == "iouring":
                l2_bytes = int(
                    self._runtime_config.get(
                        "l2_size_bytes",
                        self._runtime_config.get("total_store_bytes", 0),
                    )
                )
                if l2_bytes <= 0:
                    slot_size = int(self._runtime_config.get("slot_size", 0))
                    total_slots = int(self._runtime_config.get("total_slots", 0))
                    l2_bytes = slot_size * total_slots
                self._transfer = TieredIOUringTransferLayer(
                    path=path,
                    l1_bytes=int(self._runtime_config.get("l1_size_bytes", l2_bytes)),
                    l2_bytes=l2_bytes,
                )
            else:
                raise ValueError(f"unknown transfer_mode: {mode}")
        return self._transfer

    def _payload_buffer(self, payload: dict[str, Any]) -> Any:
        """Return a byte-addressable buffer for an IPC transfer payload."""
        if "data" in payload:
            return bytearray(payload["data"])
        if "cuda_ipc_handle" in payload:
            local_ptr = None
            producer_pid = int(payload.get("producer_pid", -1))
            device_ptr = int(payload["device_ptr"])
            nbytes = int(payload["nbytes"])
            device_id = int(payload["device_id"]) if "device_id" in payload else None
            allocation_offset = int(payload.get("allocation_offset", 0))
            allocation_base_ptr = int(
                payload.get("allocation_base_ptr", device_ptr - allocation_offset)
            )
            if producer_pid == os.getpid():
                local_ptr = allocation_base_ptr
            if local_ptr is None:
                key = (
                    producer_pid,
                    allocation_base_ptr,
                    nbytes + allocation_offset,
                    device_id,
                )
                cached = self._cuda_ipc_cache.get(key)
                if cached is None:
                    self._evict_cuda_ipc_cache_if_needed()
                    opened = open_cuda_ipc_buffer(
                        handle=payload["cuda_ipc_handle"],
                        nbytes=nbytes,
                        device_id=device_id,
                        local_ptr=None,
                        allocation_offset=allocation_offset,
                    )
                    cached = _CachedCudaArray(opened)
                    self._cuda_ipc_cache[key] = cached
                else:
                    self._cuda_ipc_cache.move_to_end(key)
                return cached
            opened = open_cuda_ipc_buffer(
                handle=payload["cuda_ipc_handle"],
                nbytes=nbytes,
                device_id=device_id,
                local_ptr=local_ptr,
                allocation_offset=allocation_offset,
            )
            return _UncachedCudaArray(opened)
        raise ValueError("transfer payload requires data or cuda_ipc_handle")

    def _evict_cuda_ipc_cache_if_needed(self) -> None:
        """Evict one cached CUDA IPC mapping when the cache is full."""
        if len(self._cuda_ipc_cache) < _CUDA_IPC_CACHE_LIMIT:
            return
        _key, cached = self._cuda_ipc_cache.popitem(last=False)
        cached.close()

    def _close_cuda_ipc_cache(self) -> None:
        """Close all cached CUDA IPC mappings."""
        for cached in self._cuda_ipc_cache.values():
            cached.close()
        self._cuda_ipc_cache.clear()

    def _payload_size(
        self, payload: dict[str, Any], spans: list[dict[str, Any]]
    ) -> int:
        """Return destination payload size for transfer_load."""
        if "nbytes" in payload:
            return int(payload["nbytes"])
        max_end = 0
        for span in spans:
            max_end = max(
                max_end,
                int(span.get("target_offset", 0)) + int(span["nbytes"]),
            )
        return max_end


class _CachedCudaArray:
    """Sliceable wrapper for a cached CUDA IPC buffer."""

    def __init__(self, opened: Any) -> None:
        self._opened = opened

    def __getitem__(self, item: Any) -> Any:
        """Return a CuPy array slice."""
        return self._opened.array[item]

    def synchronize(self) -> None:
        """Synchronize CUDA writes issued through the opened array."""
        from cupy.cuda import runtime

        runtime.streamSynchronize(0)

    def close(self) -> None:
        """Close the CUDA IPC handle."""
        self._opened.close()


class _UncachedCudaArray(_CachedCudaArray):
    """Sliceable wrapper that owns a one-shot CUDA IPC buffer."""
