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
    hash_tokens,
)
from daser.connector.gds_transfer import GDSTransferLayer

BLOCK_TOKENS = 4
NUM_LAYERS = 2


def test_dataclasses_instantiate():
    """DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec all instantiate cleanly."""
    spec_load = ReqLoadSpec("k", 0, 1, [0], 0, 16)
    spec_store = ReqStoreSpec("k", 0, 1, [0], 0, 16)
    meta = DaserConnectorMeta(
        reqs_to_load={"r": spec_load}, reqs_to_store={"r2": spec_store}
    )
    assert "r" in meta.reqs_to_load
    assert "r2" in meta.reqs_to_store


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
