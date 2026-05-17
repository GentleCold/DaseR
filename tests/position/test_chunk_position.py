# SPDX-License-Identifier: Apache-2.0

# Standard
import time

# First Party
from daser.position.chunk_position import ChunkPositionEncoder
from daser.server.metadata_store import ChunkMeta


def _meta(pos_offset: int) -> ChunkMeta:
    return ChunkMeta(
        chunk_key="test",
        start_slot=0,
        num_slots=1,
        token_count=16,
        pos_offset=pos_offset,
        model_id="m",
        created_at=time.time(),
    )


def test_assign_offset_uses_initial_offset() -> None:
    enc = ChunkPositionEncoder(initial_offset=0)
    assert enc.assign_offset("key", 16) == 0


def test_get_offset_returns_meta_offset_without_target() -> None:
    enc = ChunkPositionEncoder(initial_offset=0)
    meta = _meta(pos_offset=64)
    assert enc.get_offset(meta) == 64


def test_get_offset_returns_target_token_start_when_provided() -> None:
    enc = ChunkPositionEncoder(initial_offset=0)
    meta = _meta(pos_offset=0)
    assert enc.get_offset(meta, target_token_start=128) == 128
