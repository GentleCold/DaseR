# SPDX-License-Identifier: Apache-2.0

# Third Party
import cupy
import pytest
import torch

# First Party
from daser.connector.daser_connector import (
    DaserConnectorMeta,
    ReqLoadSpec,
    ReqStoreSpec,
    _block_ids_for_chunk,
    _contiguous_prefix_tokens,
    hash_tokens,
)
from daser.connector.gds_transfer import GDSTransferLayer

BLOCK_TOKENS = 4
NUM_LAYERS = 2

pytestmark = pytest.mark.integration


def test_dataclasses_instantiate():
    """DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec all instantiate cleanly."""
    spec_load = ReqLoadSpec("k", 0, 1, [0], 0, 16, 0, 0)
    spec_store = ReqStoreSpec("k", 0, 1, [0], 0, 16)
    meta = DaserConnectorMeta(
        reqs_to_load={"r": spec_load}, reqs_to_store={"r2": spec_store}
    )
    assert "r" in meta.reqs_to_load
    assert "r2" in meta.reqs_to_store
    assert spec_load.target_token_start == 0
    assert spec_load.pos_offset == 0


def test_block_ids_for_chunk_uses_target_token_start():
    block_ids = [10, 11, 12, 13, 14]
    assert _block_ids_for_chunk(
        block_ids=block_ids,
        target_token_start=8,
        num_slots=2,
        block_tokens=4,
    ) == [12, 13]


def test_block_ids_for_chunk_returns_empty_for_out_of_range():
    block_ids = [10, 11]
    assert (
        _block_ids_for_chunk(
            block_ids=block_ids,
            target_token_start=8,
            num_slots=1,
            block_tokens=4,
        )
        == []
    )


def test_block_ids_for_prefix_chunk_respects_accepted_token_limit():
    block_ids = [10, 11, 12]
    assert _block_ids_for_chunk(
        block_ids=block_ids,
        target_token_start=0,
        num_slots=3,
        block_tokens=4,
        max_tokens=8,
    ) == [10, 11]


def test_block_ids_for_non_prefix_chunk_can_map_without_prefix_credit():
    block_ids = [10, 11, 12, 13]
    assert _block_ids_for_chunk(
        block_ids=block_ids,
        target_token_start=8,
        num_slots=1,
        block_tokens=4,
        max_tokens=None,
    ) == [12]


def test_contiguous_prefix_tokens_handles_partially_computed_prefix():
    chunks = [{"target_token_start": 0, "token_count": 96}]
    assert _contiguous_prefix_tokens(chunks, num_computed_tokens=16) == 80


def test_contiguous_prefix_tokens_stops_at_gap():
    chunks = [
        {"target_token_start": 16, "token_count": 16},
        {"target_token_start": 48, "token_count": 16},
    ]
    assert _contiguous_prefix_tokens(chunks, num_computed_tokens=16) == 16


def test_hash_tokens_deterministic():
    tokens = [1, 2, 3, 4]
    assert hash_tokens(tokens) == hash_tokens(tokens)
    assert hash_tokens(tokens) != hash_tokens([1, 2, 3, 5])


@pytest.mark.asyncio
async def test_gds_roundtrip_with_kv_tensor(tmp_path):
    """Write a KV tensor to NVMe via GDS, read it back, verify equality."""
    store_path = str(tmp_path / "test.store")
    kv = torch.randint(
        0, 256, (2, 4, BLOCK_TOKENS, 8), dtype=torch.uint8, device="cuda"
    )
    size = 4 * 1024 * 1024
    with open(store_path, "wb") as f:
        f.write(b"\x00" * size)

    gds = GDSTransferLayer(store_path)
    data = kv[:, 0].contiguous()
    cp = cupy.asarray(data)
    await gds.write_async(cp, file_offset=0)

    recv = torch.zeros_like(kv[:, 0])
    cp_recv = cupy.asarray(recv)
    await gds.read_into_async(cp_recv, file_offset=0)
    assert torch.equal(kv[:, 0], recv)
    gds.close()
