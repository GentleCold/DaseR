# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import cupy
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from daser.connector.daser_connector import DaserConnector
from daser.connector.helpers import hash_tokens
from daser.connector.metadata import (
    DaserConnectorMeta,
    ReqLoadSpec,
    ReqStoreSpec,
    StoreChunkWrite,
)
from daser.connector.scheduler import (
    _block_ids_for_chunk,
    _contiguous_prefix_tokens,
)
from daser.connector.transfer import GDSTransferLayer, TransferBackendName
from daser.connector.worker import (
    DEFAULT_ROPE_DELTA_SCALE,
    _apply_rope_delta_to_key_block,
    _build_store_chunk_writes,
    _build_store_write_spans,
    _coerce_save_staging_for_transfer,
    _copy_host_staging_to_contiguous_kv_cache,
    _copy_kv_cache_to_staging,
    _copy_staging_to_kv_cache,
    _merge_load_staging,
)

BLOCK_TOKENS = 4
NUM_LAYERS = 2

pytestmark = pytest.mark.integration


class _RuntimeConfigProbe(DaserConnector):
    """Test connector exposing runtime config state through public properties."""

    @property
    def runtime_state(self):
        return (
            self._store_path,
            self._slot_size,
            self._block_tokens,
            self._model_id,
        )


class _WorkerProbe(DaserConnector):
    """Worker-side probe with minimal state for lazy GDS setup tests."""

    def __init__(self, store_path: str, transfer_factory=None) -> None:
        self._meta = DaserConnectorMeta(reqs_to_load={"req": object()})
        self._transfer = None
        self._transfer_backend_name = TransferBackendName.GDS
        self._transfer_factory = transfer_factory
        self._store_path = store_path
        self._slot_size = 1024
        self._block_tokens = 4
        self._layer_names = []

    def _refresh_runtime_config(self) -> None:
        return

    def _build_transfer_layer(self):
        if self._transfer_factory is None:
            return super()._build_transfer_layer()
        return self._transfer_factory()

    @property
    def transfer_backend(self):
        return self._transfer


class _SchedulerProbe(DaserConnector):
    """Scheduler-side probe that can emulate deferred runtime config."""

    def __init__(self, ipc_client) -> None:
        self._runtime_config_ready = False
        self._block_tokens = 16
        self._model_id = "default"
        self._req_tokens = {}
        self._pending_loads = {}
        self._pending_alloc = {}
        self._ipc_sync = ipc_client

    def _refresh_runtime_config(self) -> None:
        self._runtime_config_ready = True
        self._model_id = "served-model"


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


def test_connector_allows_runtime_config_from_ipc(monkeypatch, tmp_path):
    """Worker startup can begin with socket_path only and fill config by IPC."""

    class DummyIPCClient:
        def __init__(self, socket_path):
            self.socket_path = socket_path

        def get_runtime_config(self):
            return {
                "store_path": str(tmp_path / "daser.store"),
                "slot_size": 1024,
                "block_tokens": 4,
                "model_id": "served-model",
            }

    class DummyBase:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._role = role

    class DummyConfig:
        kv_connector_extra_config = {"socket_path": "/tmp/daser.sock"}

    class DummyVLLMConfig:
        kv_transfer_config = DummyConfig()
        model_config = None

    monkeypatch.setattr(
        "daser.connector.daser_connector.IPCClientSync",
        DummyIPCClient,
    )
    monkeypatch.setattr(
        "daser.connector.daser_connector.KVConnectorBase_V1.__init__",
        DummyBase.__init__,
    )

    connector = _RuntimeConfigProbe(
        DummyVLLMConfig(),
        role=KVConnectorRole.SCHEDULER,
    )

    assert connector.runtime_state == (
        str(tmp_path / "daser.store"),
        1024,
        4,
        "served-model",
    )


def test_start_load_kv_initializes_gds_after_server_creates_store(
    monkeypatch, tmp_path
):
    """Worker load path retries GDS setup after deferred server startup."""
    store_path = tmp_path / "daser.store"
    store_path.write_bytes(b"\0" * 4096)
    created_paths = []

    class DummyGDS:
        def __init__(self, path):
            created_paths.append(path)

        @property
        def backend_name(self):
            return type("Backend", (), {"value": "dummy"})()

    def build_transfer():
        return DummyGDS(str(store_path))

    connector = _WorkerProbe(str(store_path), transfer_factory=build_transfer)

    connector.start_load_kv(forward_context=object())

    assert created_paths == [str(store_path)]
    assert isinstance(connector.transfer_backend, DummyGDS)


def test_scheduler_refreshes_runtime_config_before_lookup(monkeypatch):
    """Scheduler uses server-provided model_id after deferred server startup."""
    seen_model_ids = []

    class DummyIPCClient:
        def match_and_alloc(self, tokens, chunk_key, model_id):
            seen_model_ids.append(model_id)
            return {"chunks": []}

    class DummyRequest:
        request_id = "request-1"
        prompt_token_ids = list(range(16))
        kv_transfer_params = {}

    connector = _SchedulerProbe(DummyIPCClient())

    connector.get_num_new_matched_tokens(DummyRequest(), num_computed_tokens=0)

    assert seen_model_ids == ["served-model"]


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


def test_block_ids_for_non_prefix_chunk_respects_external_token_limit():
    block_ids = [10, 11, 12, 13]
    assert (
        _block_ids_for_chunk(
            block_ids=block_ids,
            target_token_start=12,
            num_slots=1,
            block_tokens=4,
            max_tokens=12,
        )
        == []
    )


def test_contiguous_prefix_tokens_handles_partially_computed_prefix():
    chunks = [{"target_token_start": 0, "token_count": 96}]
    assert _contiguous_prefix_tokens(chunks, num_computed_tokens=16) == 80


def test_contiguous_prefix_tokens_stops_at_gap():
    chunks = [
        {"target_token_start": 16, "token_count": 16},
        {"target_token_start": 48, "token_count": 16},
    ]
    assert _contiguous_prefix_tokens(chunks, num_computed_tokens=16) == 16


def test_contiguous_prefix_tokens_covers_padded_rag_segments():
    chunks = [
        {"target_token_start": 0, "token_count": 4},
        {"target_token_start": 4, "token_count": 4},
        {"target_token_start": 8, "token_count": 4},
        {"target_token_start": 12, "token_count": 4},
        {"target_token_start": 16, "token_count": 4},
    ]

    assert _contiguous_prefix_tokens(chunks, num_computed_tokens=0) == 20


def _rotate_neox_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    rotary_dim: int,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device)
            / rotary_dim
        )
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = freqs.cos().unsqueeze(-2).to(x.dtype)
    sin = freqs.sin().unsqueeze(-2).to(x.dtype)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1, x2 = torch.chunk(x_rot, 2, dim=-1)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)


def test_apply_rope_delta_rotates_key_block_to_target_positions():
    raw = torch.randn(4, 2, 8, dtype=torch.float32)
    source_positions = torch.arange(4)
    target_positions = source_positions + 12
    stored = _rotate_neox_reference(raw, source_positions, base=10000.0, rotary_dim=8)
    expected = _rotate_neox_reference(raw, target_positions, base=10000.0, rotary_dim=8)

    actual = stored.clone()
    _apply_rope_delta_to_key_block(
        actual,
        delta=12,
        rope_base=10000.0,
        rotary_dim=8,
        is_neox_style=True,
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_connector_default_rope_delta_scale_moves_to_target_position():
    """Default load-time RoPE relocation uses the target-start delta."""
    assert round(64 * DEFAULT_ROPE_DELTA_SCALE) == 64


def test_apply_rope_delta_leaves_non_rotary_tail_unchanged():
    raw = torch.randn(4, 2, 8, dtype=torch.float32)
    actual = raw.clone()

    _apply_rope_delta_to_key_block(
        actual,
        delta=7,
        rope_base=10000.0,
        rotary_dim=4,
        is_neox_style=True,
    )

    assert torch.equal(actual[..., 4:], raw[..., 4:])
    assert not torch.equal(actual[..., :4], raw[..., :4])


def test_update_state_after_alloc_skips_chunks_beyond_external_prefix():
    class MockConnector:
        def __init__(self) -> None:
            self._block_tokens = BLOCK_TOKENS
            self._pending_loads = {
                "req": {
                    "0": {
                        "chunk_key": "a",
                        "num_slots": 1,
                        "target_token_start": 0,
                    },
                    "1": {
                        "chunk_key": "b",
                        "num_slots": 1,
                        "target_token_start": 4,
                    },
                    "2": {
                        "chunk_key": "c",
                        "num_slots": 1,
                        "target_token_start": 8,
                    },
                }
            }
            self._pending_alloc = {}

        @property
        def pending_loads(self) -> dict:
            return self._pending_loads

    class MockRequest:
        request_id = "req"

    class MockBlock:
        def __init__(self, block_id: int) -> None:
            self.block_id = block_id

    class MockBlocks:
        blocks = ([MockBlock(10), MockBlock(11), MockBlock(12)],)

    connector = MockConnector()

    DaserConnector.update_state_after_alloc(
        connector, MockRequest(), MockBlocks(), num_external_tokens=8
    )

    chunks = connector.pending_loads["req"]
    assert chunks["0"]["block_ids"] == [10]
    assert chunks["1"]["block_ids"] == [11]
    assert "2" not in chunks


def test_hash_tokens_deterministic():
    tokens = [1, 2, 3, 4]
    assert hash_tokens(tokens) == hash_tokens(tokens)
    assert hash_tokens(tokens) != hash_tokens([1, 2, 3, 5])


def test_build_store_chunk_writes_captures_async_metadata():
    """Chunk write descriptors do not depend on mutable connector state."""
    specs = {
        "req-a": ReqStoreSpec("chunk-a", 10, 2, [1, 2], 1000, 8),
        "req-b": ReqStoreSpec("chunk-b", 20, 1, [3], 2000, 4),
    }

    writes = _build_store_chunk_writes(
        reqs_to_store=specs,
        req_slot_ranges={"req-a": (0, 2), "req-b": (2, 3)},
        slot_size=128,
    )

    assert [write.chunk_key for write in writes] == ["chunk-a", "chunk-b"]
    offsets = [
        (write.source_offset, write.nbytes, write.file_offset) for write in writes
    ]
    assert offsets == [
        (0, 256, 1000),
        (256, 128, 2000),
    ]


def test_iouring_chunk_transfer_keeps_torch_save_staging(monkeypatch):
    """Chunk-aware transfer writes can consume torch staging without CuPy wrapping."""
    staging = torch.zeros(8, dtype=torch.uint8)

    class ChunkTransfer:
        async def write_chunk_async(self) -> None:
            return None

    def fail_asarray(_):
        raise AssertionError("cupy.asarray should not be called")

    monkeypatch.setattr(cupy, "asarray", fail_asarray)

    coerced = _coerce_save_staging_for_transfer(staging, ChunkTransfer())

    assert coerced is staging


def test_gds_span_transfer_wraps_save_staging_with_cupy(monkeypatch):
    """Span-only transfer writes still receive a CuPy-compatible staging view."""
    staging = torch.zeros(8, dtype=torch.uint8)
    sentinel = object()
    seen = []

    class SpanTransfer:
        async def write_async(self) -> None:
            return None

    def fake_asarray(value):
        seen.append(value)
        return sentinel

    monkeypatch.setattr(cupy, "asarray", fake_asarray)

    coerced = _coerce_save_staging_for_transfer(staging, SpanTransfer())

    assert coerced is sentinel
    assert seen == [staging]


@pytest.mark.asyncio
async def test_chunk_writes_are_serialized_to_limit_l1_durable_pins():
    """Chunk-aware writes complete one at a time to avoid filling L1 with pins."""
    staging = torch.arange(12, dtype=torch.uint8)
    active = 0
    max_active = 0
    seen: list[str] = []

    class ChunkTransfer:
        async def write_chunk_async(self, chunk_key, buf, file_offset, nbytes):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            seen.append(chunk_key)
            await asyncio.sleep(0)
            active -= 1
            return nbytes

    class Probe(DaserConnector):
        def __init__(self) -> None:
            self._transfer = ChunkTransfer()

        async def run_write_and_commit(self):
            await self._write_and_commit(
                staging,
                spans=[],
                chunk_writes=writes,
                commit_keys=[],
            )

    writes = [
        StoreChunkWrite("a", 0, 4, 0),
        StoreChunkWrite("b", 4, 4, 4),
        StoreChunkWrite("c", 8, 4, 8),
    ]

    await Probe().run_write_and_commit()

    assert seen == ["a", "b", "c"]
    assert max_active == 1


def test_copy_staging_to_kv_cache_batches_by_layer():
    """Slot-major staging bytes are restored into arbitrary KV block IDs."""
    block_ids = [5, 1, 7]
    layer_names = ["layer.0", "layer.1"]
    kv_caches = {
        name: torch.zeros((2, 10, 2, 2), dtype=torch.bfloat16) for name in layer_names
    }
    layer_shape = kv_caches["layer.0"][:, block_ids[0]].shape
    layer_size = kv_caches["layer.0"][:, block_ids[0]].nbytes
    slot_size = layer_size * len(layer_names)
    staging = torch.empty(len(block_ids) * slot_size, dtype=torch.uint8)

    for slot_i in range(len(block_ids)):
        for layer_idx in range(len(layer_names)):
            offset = slot_i * slot_size + layer_idx * layer_size
            value = float((slot_i + 1) * 100 + layer_idx)
            layer = torch.full(layer_shape, value, dtype=torch.bfloat16)
            staging[offset : offset + layer_size].copy_(
                layer.reshape(-1).view(torch.uint8)
            )

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches=kv_caches,
        layer_names=layer_names,
        block_ids=block_ids,
        slot_size=slot_size,
    )

    assert copies == len(layer_names)
    for slot_i, block_id in enumerate(block_ids):
        for layer_idx, layer_name in enumerate(layer_names):
            expected = torch.full(
                layer_shape,
                float((slot_i + 1) * 100 + layer_idx),
                dtype=torch.bfloat16,
            )
            assert torch.equal(kv_caches[layer_name][:, block_id], expected)


def test_merge_load_staging_combines_requests_in_block_order():
    first = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    second = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
    merged, block_ids = _merge_load_staging(
        [(first, [10, 11]), (second, [7, 8])],
        slot_size=2,
    )

    assert merged.tolist() == [5, 6, 7, 8, 1, 2, 3, 4]
    assert block_ids == [7, 8, 10, 11]


def test_copy_host_staging_to_contiguous_kv_cache_avoids_gpu_staging():
    """Contiguous block IDs can be copied directly from host staging."""
    block_ids = [3, 4, 5]
    layer_names = ["layer.0", "layer.1"]
    kv_caches = {
        name: torch.zeros((2, 10, 2, 2), dtype=torch.bfloat16) for name in layer_names
    }
    layer_shape = kv_caches["layer.0"][:, block_ids[0]].shape
    layer_size = kv_caches["layer.0"][:, block_ids[0]].nbytes
    slot_size = layer_size * len(layer_names)
    staging = torch.empty(len(block_ids) * slot_size, dtype=torch.uint8)

    for slot_i in range(len(block_ids)):
        for layer_idx in range(len(layer_names)):
            offset = slot_i * slot_size + layer_idx * layer_size
            value = float((slot_i + 1) * 100 + layer_idx)
            layer = torch.full(layer_shape, value, dtype=torch.bfloat16)
            staging[offset : offset + layer_size].copy_(
                layer.reshape(-1).view(torch.uint8)
            )

    copies = _copy_host_staging_to_contiguous_kv_cache(
        staging=staging,
        kv_caches=kv_caches,
        layer_names=layer_names,
        block_ids=block_ids,
        slot_size=slot_size,
    )

    assert copies == len(layer_names)
    for slot_i, block_id in enumerate(block_ids):
        for layer_idx, layer_name in enumerate(layer_names):
            expected = torch.full(
                layer_shape,
                float((slot_i + 1) * 100 + layer_idx),
                dtype=torch.bfloat16,
            )
            assert torch.equal(kv_caches[layer_name][:, block_id], expected)


def test_copy_kv_cache_to_staging_batches_by_layer():
    """KV cache blocks are packed into slot-major staging bytes."""
    block_ids = [5, 1, 7]
    num_layers = 2
    kv_layer = torch.zeros((2, 10, 2, 2), dtype=torch.bfloat16)
    layer_shape = kv_layer[:, block_ids[0]].shape
    layer_size = kv_layer[:, block_ids[0]].nbytes
    slot_size = layer_size * num_layers
    staging = torch.zeros(len(block_ids) * slot_size, dtype=torch.uint8)

    for slot_i, block_id in enumerate(block_ids):
        kv_layer[:, block_id].fill_(float((slot_i + 1) * 10))

    _copy_kv_cache_to_staging(
        staging=staging,
        kv_layer=kv_layer,
        layer_idx=1,
        block_ids=block_ids,
        num_layers=num_layers,
        slot_size=slot_size,
    )

    for slot_i in range(len(block_ids)):
        layer0_offset = slot_i * slot_size
        layer1_offset = layer0_offset + layer_size
        layer0 = staging[layer0_offset : layer0_offset + layer_size].view(
            torch.bfloat16
        )
        layer1 = staging[layer1_offset : layer1_offset + layer_size].view(
            torch.bfloat16
        )
        expected = torch.full(
            layer_shape,
            float((slot_i + 1) * 10),
            dtype=torch.bfloat16,
        )
        assert torch.equal(layer0, torch.zeros_like(layer0))
        assert torch.equal(layer1.view(layer_shape), expected)


def test_copy_kv_cache_to_staging_handles_contiguous_blocks():
    """Contiguous block IDs are packed into the same slot-major format."""
    block_ids = [2, 3, 4]
    num_layers = 2
    kv_layer = torch.zeros((2, 8, 2, 2), dtype=torch.bfloat16)
    layer_shape = kv_layer[:, block_ids[0]].shape
    layer_size = kv_layer[:, block_ids[0]].nbytes
    slot_size = layer_size * num_layers
    staging = torch.zeros(len(block_ids) * slot_size, dtype=torch.uint8)

    for slot_i, block_id in enumerate(block_ids):
        kv_layer[:, block_id].fill_(float((slot_i + 1) * 10))

    _copy_kv_cache_to_staging(
        staging=staging,
        kv_layer=kv_layer,
        layer_idx=1,
        block_ids=block_ids,
        num_layers=num_layers,
        slot_size=slot_size,
    )

    for slot_i in range(len(block_ids)):
        layer1_offset = slot_i * slot_size + layer_size
        layer1 = staging[layer1_offset : layer1_offset + layer_size].view(
            torch.bfloat16
        )
        expected = torch.full(
            layer_shape,
            float((slot_i + 1) * 10),
            dtype=torch.bfloat16,
        )
        assert torch.equal(layer1.view(layer_shape), expected)


def test_copy_kv_cache_to_staging_accepts_precomputed_block_index():
    """Non-contiguous block IDs can reuse a caller-provided device index."""
    block_ids = [5, 1, 7]
    num_layers = 2
    kv_layer = torch.zeros((2, 10, 2, 2), dtype=torch.bfloat16)
    layer_shape = kv_layer[:, block_ids[0]].shape
    layer_size = kv_layer[:, block_ids[0]].nbytes
    slot_size = layer_size * num_layers
    staging = torch.zeros(len(block_ids) * slot_size, dtype=torch.uint8)
    block_index = torch.tensor(block_ids, dtype=torch.long, device=kv_layer.device)

    for slot_i, block_id in enumerate(block_ids):
        kv_layer[:, block_id].fill_(float((slot_i + 1) * 10))

    _copy_kv_cache_to_staging(
        staging=staging,
        kv_layer=kv_layer,
        layer_idx=1,
        block_ids=block_ids,
        num_layers=num_layers,
        slot_size=slot_size,
        block_index=block_index,
    )

    for slot_i in range(len(block_ids)):
        layer1_offset = slot_i * slot_size + layer_size
        layer1 = staging[layer1_offset : layer1_offset + layer_size].view(
            torch.bfloat16
        )
        expected = torch.full(
            layer_shape,
            float((slot_i + 1) * 10),
            dtype=torch.bfloat16,
        )
        assert torch.equal(layer1.view(layer_shape), expected)


def test_build_store_write_spans_coalesces_adjacent_requests():
    """Adjacent request slices with adjacent store slots become one pwrite."""
    reqs_to_store = {
        "r0": ReqStoreSpec("k0", 10, 2, [4, 5], 0, 8),
        "r1": ReqStoreSpec("k1", 12, 1, [6], 0, 4),
        "r2": ReqStoreSpec("k2", 20, 2, [7, 8], 0, 8),
    }
    req_slot_ranges = {
        "r0": (0, 2),
        "r1": (2, 3),
        "r2": (3, 5),
    }

    spans = _build_store_write_spans(reqs_to_store, req_slot_ranges, slot_size=32)

    assert [(s.source_offset, s.nbytes, s.file_offset) for s in spans] == [
        (0, 96, 320),
        (96, 64, 640),
    ]


def test_step_staging_packs_multiple_requests_with_one_layer_copy():
    """A combined block list can pack all request KV into step-major staging."""
    req_block_ids = {
        "r0": [5, 1],
        "r1": [7],
    }
    all_block_ids = [block_id for ids in req_block_ids.values() for block_id in ids]
    num_layers = 2
    kv_layer = torch.zeros((2, 10, 2, 2), dtype=torch.bfloat16)
    layer_shape = kv_layer[:, all_block_ids[0]].shape
    layer_size = kv_layer[:, all_block_ids[0]].nbytes
    slot_size = layer_size * num_layers
    staging = torch.zeros(len(all_block_ids) * slot_size, dtype=torch.uint8)

    for slot_i, block_id in enumerate(all_block_ids):
        kv_layer[:, block_id].fill_(float((slot_i + 1) * 10))

    _copy_kv_cache_to_staging(
        staging=staging,
        kv_layer=kv_layer,
        layer_idx=0,
        block_ids=all_block_ids,
        num_layers=num_layers,
        slot_size=slot_size,
    )

    for slot_i in range(len(all_block_ids)):
        offset = slot_i * slot_size
        actual = staging[offset : offset + layer_size].view(torch.bfloat16)
        expected = torch.full(
            layer_shape,
            float((slot_i + 1) * 10),
            dtype=torch.bfloat16,
        )
        assert torch.equal(actual.view(layer_shape), expected)


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
