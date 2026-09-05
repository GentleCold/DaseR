# SPDX-License-Identifier: Apache-2.0

import asyncio

import msgpack
import pytest

from daser.compression import (
    CompressedSlotRef,
    CompressedStoreGeometry,
    CompressedStoreIndex,
    SlotMode,
)
from daser.compression.format import digest_bytes
from daser.connector.helpers import ROLLING_PREFIX_SEED, rolling_prefix_key
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.ipc.server import IPCServer
from daser.server.metadata_store import MetadataStore


def _compressed_index() -> CompressedStoreIndex:
    geometry = CompressedStoreGeometry(
        num_slots=4,
        slot_size=4096,
        block_tokens=128,
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
    )
    zero_hash = digest_bytes(bytes(geometry.slot_size))
    entries = [
        CompressedSlotRef(
            slot_id=slot_id,
            mode=SlotMode.COMPRESSED,
            file_offset=slot_id * geometry.slot_size,
            stored_length=4096,
            raw_hash=zero_hash,
            encoded_hash=zero_hash,
        )
        for slot_id in range(geometry.num_slots)
    ]
    codebooks = bytes(range(15)) * geometry.plane_count
    return CompressedStoreIndex(
        geometry,
        digest_bytes(b"model"),
        codebooks,
        entries,
    )


def _core() -> ServerCore:
    metadata = MetadataStore(total_slots=4)
    manager = ChunkManager(total_slots=4, metadata_store=metadata)
    return ServerCore(
        chunk_manager=manager,
        retrieval_index=PrefixHashIndex(block_tokens=128),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=4096,
        block_tokens=128,
    )


async def _send(socket_path: str, payload: dict[str, object]) -> dict[str, object]:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        body = msgpack.packb(payload, use_bin_type=True)
        writer.write(len(body).to_bytes(4, "big") + body)
        await writer.drain()
        length = int.from_bytes(await reader.readexactly(4), "big")
        return msgpack.unpackb(await reader.readexactly(length), raw=False)
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_lookup_attaches_slot_refs_and_mutations_fail(tmp_path) -> None:
    core = _core()
    tokens = list(range(128))
    key = rolling_prefix_key(ROLLING_PREFIX_SEED, tokens)
    await core.alloc_chunk(key, len(tokens), "model")
    await core.commit_chunk(key)
    socket_path = str(tmp_path / "compressed.sock")
    server = IPCServer(
        socket_path,
        core,
        runtime_config={
            "storage_format": "compressed-read-only",
            "block_tokens": 128,
        },
        compressed_store_index=_compressed_index(),
    )
    await server.start()
    try:
        lookup = await _send(
            socket_path,
            {"op": "lookup", "tokens": tokens, "model_id": "model"},
        )
        slots = lookup["chunks"][0]["compressed_slots"]
        assert slots == [
            {
                "slot_id": 0,
                "mode": "compressed",
                "file_offset": 0,
                "stored_length": 4096,
            }
        ]
        allocation = await _send(
            socket_path,
            {
                "op": "alloc_chunk",
                "chunk_key": "new",
                "token_count": 128,
                "model_id": "model",
            },
        )
        assert "compressed-read-only" in allocation["error"]
        store = await _send(
            socket_path,
            {"op": "transfer_store", "payload": {"data": b""}, "spans": []},
        )
        assert "compressed-read-only" in store["error"]
    finally:
        await server.stop()
