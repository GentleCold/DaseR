# SPDX-License-Identifier: Apache-2.0

# Standard
import time

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
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


def test_default_offset_is_zero():
    enc = FixedOffsetEncoder()
    assert enc.assign_offset("key", 16) == 0


def test_custom_fixed_offset():
    enc = FixedOffsetEncoder(fixed_offset=512)
    assert enc.assign_offset("key", 16) == 512


def test_get_offset_returns_meta_pos_offset():
    enc = FixedOffsetEncoder(fixed_offset=0)
    meta = _meta(pos_offset=128)
    assert enc.get_offset(meta) == 128
