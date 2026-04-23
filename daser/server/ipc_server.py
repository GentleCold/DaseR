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
from daser.server.doc_registry import DocEntry, DocRegistry

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
        doc_registry: optional DocRegistry; when provided the server
            also handles register_doc / list_docs / get_doc / evict_doc
            ops used by the service layer.
    """

    def __init__(
        self,
        socket_path: str,
        chunk_manager: ChunkManager,
        retrieval_index: RetrievalIndex,
        position_encoder: PositionEncoder,
        slot_size: int,
        block_tokens: int = 16,
        doc_registry: "DocRegistry | None" = None,
    ) -> None:
        self._socket_path = socket_path
        self._cm = chunk_manager
        self._ri = retrieval_index
        self._pe = position_encoder
        self._slot_size = slot_size
        self._block_tokens = block_tokens
        self._doc_registry = doc_registry
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
        """Handle one client connection with multiple request frames.

        The connection stays open until the client closes it or an
        unrecoverable error occurs. This lets IPCClientSync amortise
        socket setup across many scheduler RPCs.

        Args:
            reader: stream reader.
            writer: stream writer.
        """
        try:
            while True:
                try:
                    msg = await _read_frame(reader)
                except asyncio.IncompleteReadError:
                    return
                op = msg.get("op")
                if op == "lookup":
                    response = await self._handle_lookup(msg)
                elif op == "alloc_chunk":
                    response = await self._handle_alloc_chunk(msg)
                elif op == "match_and_alloc":
                    response = await self._handle_match_and_alloc(msg)
                elif op == "commit_chunk":
                    response = await self._handle_commit_chunk(msg)
                elif op == "evict_chunk":
                    response = await self._handle_evict_chunk(msg)
                elif op == "register_doc":
                    response = await self._handle_register_doc(msg)
                elif op == "list_docs":
                    response = await self._handle_list_docs(msg)
                elif op == "get_doc":
                    response = await self._handle_get_doc(msg)
                elif op == "evict_doc":
                    response = await self._handle_evict_doc(msg)
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
            await writer.wait_closed()

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
            Response dict with start_slot, num_slots, file_offset, pos_offset.
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
            "num_slots": num_slots,
            "file_offset": file_offset,
            "pos_offset": pos_offset,
        }

    async def _handle_match_and_alloc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a combined lookup + (conditional) alloc request.

        Looks up tokens for a cache hit. If a hit is found, returns the
        matching chunks and no allocation. If no hit, allocates a new
        chunk for the block-aligned prefix and returns the allocation
        info (so the caller can use it as a future store target).

        Args:
            msg: request dict with tokens, model_id, chunk_key (hash of
                the block-aligned prefix prepared client-side).

        Returns:
            Dict with "chunks" and optional "alloc" keys.
        """
        tokens: list[int] = msg["tokens"]
        model_id: str = msg["model_id"]
        chunk_key: str = msg.get("chunk_key", "")

        chunks = await self._ri.lookup(tokens, model_id)
        if chunks:
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
                ],
                "alloc": None,
            }

        if not chunk_key:
            return {"chunks": [], "alloc": None}
        aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        if aligned == 0:
            return {"chunks": [], "alloc": None}
        num_slots = math.ceil(aligned / self._block_tokens)
        pos_offset = self._pe.assign_offset(chunk_key, aligned)
        start_slot = self._cm.alloc(
            chunk_key=chunk_key,
            num_slots=num_slots,
            token_count=aligned,
            model_id=model_id,
            pos_offset=pos_offset,
        )
        return {
            "chunks": [],
            "alloc": {
                "chunk_key": chunk_key,
                "start_slot": start_slot,
                "num_slots": num_slots,
                "file_offset": start_slot * self._slot_size,
                "pos_offset": pos_offset,
                "token_count": aligned,
            },
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
            if self._doc_registry is not None:
                for doc_id in list(meta.doc_ids):
                    self._doc_registry.mark_chunk_evicted(doc_id, chunk_key)
            self._cm.store.remove(chunk_key)
        logger.debug("[IPC] evict_chunk key=%s", chunk_key[:8])
        return {"ok": True}

    async def _handle_register_doc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Register a new document and attach its chunk_keys.

        Links the doc_id to every already-committed chunk in
        ``chunk_keys`` and records the DocEntry in DocRegistry.
        Chunks that are no longer in the ring buffer are accepted —
        their cached_mask bit is set to False up front.

        Args:
            msg: ``doc_id``, ``title``, ``chunk_keys``, ``token_count``,
                optional ``tokens``.

        Returns:
            ``{"ok": True, "chunk_count_cached": int}`` on success,
            otherwise ``{"error": ...}``.
        """
        if self._doc_registry is None:
            return {"error": "doc registry not enabled"}

        doc_id: str = msg["doc_id"]
        title: str = msg.get("title", "")
        chunk_keys: list[str] = msg.get("chunk_keys", [])
        token_count: int = int(msg.get("token_count", 0))
        tokens = msg.get("tokens")

        if self._doc_registry.get(doc_id) is not None:
            return {"error": f"doc_id already exists: {doc_id}"}

        cached_mask: list[bool] = []
        for key in chunk_keys:
            meta = self._cm.store.get(key)
            if meta is None:
                cached_mask.append(False)
                continue
            if doc_id not in meta.doc_ids:
                meta.doc_ids.append(doc_id)
            cached_mask.append(True)

        entry = DocEntry(
            doc_id=doc_id,
            title=title,
            token_count=token_count,
            chunk_keys=list(chunk_keys),
            cached_mask=cached_mask,
            status="ready" if any(cached_mask) else "evicted",
            tokens=list(tokens) if tokens is not None else None,
        )
        self._doc_registry.insert(entry)
        logger.info(
            "[IPC] register_doc doc_id=%s chunks=%d cached=%d",
            doc_id,
            len(chunk_keys),
            sum(cached_mask),
        )
        return {"ok": True, "chunk_count_cached": sum(cached_mask)}

    async def _handle_list_docs(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Return a summary of every registered document.

        Args:
            msg: unused.

        Returns:
            ``{"docs": [...]}`` where each entry carries doc_id, title,
            token_count, chunk_count_total, chunk_count_cached, status
            and created_at.
        """
        if self._doc_registry is None:
            return {"docs": []}
        docs: list[dict[str, Any]] = []
        for entry in self._doc_registry.all_entries():
            docs.append(
                {
                    "doc_id": entry.doc_id,
                    "title": entry.title,
                    "token_count": entry.token_count,
                    "chunk_count_total": len(entry.chunk_keys),
                    "chunk_count_cached": sum(entry.cached_mask),
                    "status": entry.status,
                    "created_at": entry.created_at,
                }
            )
        return {"docs": docs}

    async def _handle_get_doc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Return the full DocEntry for ``doc_id``.

        Args:
            msg: ``doc_id``.

        Returns:
            ``{"doc": {...}}`` with chunk_keys, cached_mask, tokens, etc.,
            or ``{"error": ...}`` when the doc is unknown.
        """
        if self._doc_registry is None:
            return {"error": "doc registry not enabled"}
        doc_id: str = msg["doc_id"]
        entry = self._doc_registry.get(doc_id)
        if entry is None:
            return {"error": f"doc_id not found: {doc_id}"}
        return {
            "doc": {
                "doc_id": entry.doc_id,
                "title": entry.title,
                "created_at": entry.created_at,
                "token_count": entry.token_count,
                "chunk_keys": list(entry.chunk_keys),
                "cached_mask": list(entry.cached_mask),
                "status": entry.status,
                "tokens": (list(entry.tokens) if entry.tokens is not None else None),
                "error": entry.error,
            }
        }

    async def _handle_evict_doc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Unregister a document and evict chunks it solely references.

        For each chunk_key in the DocEntry:
          - remove ``doc_id`` from its ``ChunkMeta.doc_ids``;
          - if no other doc references that chunk anymore, evict it
            from the ring buffer (retrieval index + metadata store).

        Args:
            msg: ``doc_id``.

        Returns:
            ``{"ok": True, "chunks_evicted": int}`` or ``{"error": ...}``.
        """
        if self._doc_registry is None:
            return {"error": "doc registry not enabled"}
        doc_id: str = msg["doc_id"]
        entry = self._doc_registry.remove(doc_id)
        if entry is None:
            return {"error": f"doc_id not found: {doc_id}"}

        chunks_evicted = 0
        for key in entry.chunk_keys:
            meta = self._cm.store.get(key)
            if meta is None:
                continue
            if doc_id in meta.doc_ids:
                meta.doc_ids.remove(doc_id)
            if not meta.doc_ids:
                await self._ri.remove(key)
                self._cm.store.remove(key)
                chunks_evicted += 1
        logger.info(
            "[IPC] evict_doc doc_id=%s chunks_evicted=%d", doc_id, chunks_evicted
        )
        return {"ok": True, "chunks_evicted": chunks_evicted}
