# SPDX-License-Identifier: Apache-2.0

# Third Party
import cupy
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from daser.connector.daser_connector import DaserConnector
from daser.connector.helpers import hash_tokens
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.scheduler import (
    SchedulerConnectorMixin,
    _block_ids_for_chunk,
    _contiguous_prefix_tokens,
    _trim_chunk_to_external_window,
)
from daser.connector.staging import (
    DEFAULT_PENDING_STORE_STAGING_BYTES,
    DEFAULT_ROPE_DELTA_SCALE,
    DEFAULT_STORE_STAGING_BYTES,
    MIN_STORE_STAGING_BYTES,
    CudaStagingPool,
)
from daser.connector.staging import (
    apply_rope_delta_to_key_block as _apply_rope_delta_to_key_block,
)
from daser.connector.staging import (
    build_load_copy_runs as _build_load_copy_runs,
)
from daser.connector.staging import (
    build_load_read_batches as _build_load_read_batches,
)
from daser.connector.staging import (
    build_load_read_plan as _build_load_read_plan,
)
from daser.connector.staging import (
    build_staging_store_batches as _build_staging_store_batches,
)
from daser.connector.staging import (
    copy_staging_to_kv_cache as _copy_staging_to_kv_cache,
)
from daser.connector.staging import (
    derive_store_staging_limits as _derive_store_staging_limits,
)
from daser.connector.staging import (
    record_cuda_event as _record_cuda_event,
)
from daser.connector.staging import (
    synchronize_cuda_tensor as _synchronize_cuda_tensor,
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
    """Worker-side probe with minimal state for transfer readiness tests."""

    def __init__(self, store_path: str) -> None:
        self._meta = DaserConnectorMeta(reqs_to_load={"req": object()})
        self._transfer_ready = False
        self._store_path = store_path
        self._slot_size = 1024
        self._block_tokens = 4
        self._layer_names = []
        self._transfer_mode = "gds"

    def _refresh_runtime_config(self) -> None:
        return

    @property
    def transfer_ready(self):
        return self._transfer_ready


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


class _AllocatingSchedulerProbe(SchedulerConnectorMixin):
    """Minimal scheduler probe that records allocation RPCs."""

    def __init__(self) -> None:
        self._block_tokens = BLOCK_TOKENS
        self._slot_size = 32
        self._pending_loads = {}
        self._pending_stores = {}
        self._pending_alloc = {}
        self._req_tokens = {}
        self._model_id = "m"
        self.alloc_calls: list[tuple[str, int, str]] = []
        self._ipc_sync = self

    def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
        """Record an allocation call and return server-style metadata."""
        self.alloc_calls.append((chunk_key, token_count, model_id))
        return {
            "start_slot": 5,
            "file_offset": 160,
            "pos_offset": 0,
        }

    def live_allocations(self, allocations: list[dict]) -> set[str]:
        """Return allocations whose chunk key starts with ``live``."""
        return {
            str(alloc["chunk_key"])
            for alloc in allocations
            if str(alloc["chunk_key"]).startswith("live")
        }

    def seed_pending_store(
        self, req_id: str, chunk_key: str, token_count: int, block_ids: list[int]
    ) -> None:
        """Seed pending scheduler state for a store allocation test."""
        self._req_tokens[req_id] = [1] * token_count
        self._pending_alloc[req_id] = type(
            "Pending",
            (),
            {
                "chunk_key": chunk_key,
                "token_count": token_count,
                "block_ids": block_ids,
            },
        )()

    def record_cached_blocks(self, scheduler_output) -> None:
        """Expose cached-block recording through a public test helper."""
        self._record_cached_store_blocks(scheduler_output)

    def seed_pending_store_spec(self, req_id: str, spec: ReqStoreSpec) -> None:
        """Seed a ready pending store entry for connector-meta packaging."""
        self._pending_stores[req_id] = {
            "chunk_key": spec.chunk_key,
            "start_slot": spec.start_slot,
            "num_slots": spec.num_slots,
            "block_ids": spec.block_ids,
            "file_offset": spec.file_offset,
            "token_count": spec.token_count,
        }

    @property
    def pending_state(self) -> tuple[dict, dict]:
        """Return pending allocation and store state for assertions."""
        return self._pending_alloc, self._pending_stores


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
    """Worker load path marks server transfer ready after deferred startup."""
    store_path = tmp_path / "daser.store"
    store_path.write_bytes(b"\0" * 4096)

    connector = _WorkerProbe(str(store_path))

    connector.start_load_kv(forward_context=object())

    assert connector.transfer_ready is True


def test_scheduler_refreshes_runtime_config_before_lookup(monkeypatch):
    """Scheduler uses server-provided model_id after deferred server startup."""
    seen_model_ids = []

    class DummyIPCClient:
        def lookup(self, tokens, model_id):
            seen_model_ids.append(model_id)
            return []

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


def test_trim_chunk_to_external_window_skips_local_prefix_slots():
    """External load windows trim chunk slots that vLLM already computed."""
    chunk = {
        "chunk_key": "k0",
        "start_slot": 100,
        "num_slots": 4,
        "file_offset": 3200,
        "token_count": 16,
        "target_token_start": 0,
    }

    ok = _trim_chunk_to_external_window(
        chunk=chunk,
        block_ids=[10, 11, 12, 13],
        external_start=4,
        num_external_tokens=8,
        block_tokens=4,
        slot_size=32,
    )

    assert ok
    assert chunk["start_slot"] == 101
    assert chunk["file_offset"] == 3232
    assert chunk["num_slots"] == 2
    assert chunk["token_count"] == 8
    assert chunk["target_token_start"] == 4
    assert chunk["block_ids"] == [10, 11]


def test_update_state_after_alloc_single_hit_uses_external_window():
    """Single-prefix hit maps only the external suffix vLLM requested."""

    class MockConnector(SchedulerConnectorMixin):
        def __init__(self) -> None:
            self._block_tokens = BLOCK_TOKENS
            self._slot_size = 32
            self._pending_loads = {
                "req": {
                    "chunk_key": "k0",
                    "start_slot": 100,
                    "num_slots": 4,
                    "file_offset": 3200,
                    "token_count": 16,
                    "target_token_start": 0,
                    "num_computed_tokens": 4,
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
        blocks = ([MockBlock(10), MockBlock(11), MockBlock(12), MockBlock(13)],)

    connector = MockConnector()

    DaserConnector.update_state_after_alloc(
        connector,
        MockRequest(),
        MockBlocks(),
        num_external_tokens=8,
    )

    chunk = connector.pending_loads["req"]
    assert chunk["start_slot"] == 101
    assert chunk["file_offset"] == 3232
    assert chunk["num_slots"] == 2
    assert chunk["block_ids"] == [10, 11]


def test_update_state_after_alloc_multi_hit_trims_each_chunk_to_external_window():
    """Multi-chunk hits use the same absolute external token window."""

    class MockConnector(SchedulerConnectorMixin):
        def __init__(self) -> None:
            self._block_tokens = BLOCK_TOKENS
            self._slot_size = 32
            self._pending_loads = {
                "req": {
                    "0": {
                        "chunk_key": "a",
                        "start_slot": 100,
                        "num_slots": 2,
                        "file_offset": 3200,
                        "token_count": 8,
                        "target_token_start": 0,
                        "num_computed_tokens": 4,
                    },
                    "1": {
                        "chunk_key": "b",
                        "start_slot": 200,
                        "num_slots": 2,
                        "file_offset": 6400,
                        "token_count": 8,
                        "target_token_start": 8,
                        "num_computed_tokens": 4,
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
        blocks = ([MockBlock(10), MockBlock(11), MockBlock(12), MockBlock(13)],)

    connector = MockConnector()

    DaserConnector.update_state_after_alloc(
        connector,
        MockRequest(),
        MockBlocks(),
        num_external_tokens=8,
    )

    chunks = connector.pending_loads["req"]
    assert chunks["0"]["start_slot"] == 101
    assert chunks["0"]["num_slots"] == 1
    assert chunks["0"]["block_ids"] == [10]
    assert chunks["1"]["start_slot"] == 200
    assert chunks["1"]["num_slots"] == 1
    assert chunks["1"]["block_ids"] == [11]


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
            self._slot_size = 32
            self._pending_loads = {
                "req": {
                    "0": {
                        "chunk_key": "a",
                        "start_slot": 0,
                        "num_slots": 1,
                        "file_offset": 0,
                        "token_count": 4,
                        "target_token_start": 0,
                    },
                    "1": {
                        "chunk_key": "b",
                        "start_slot": 1,
                        "num_slots": 1,
                        "file_offset": 32,
                        "token_count": 4,
                        "target_token_start": 4,
                    },
                    "2": {
                        "chunk_key": "c",
                        "start_slot": 2,
                        "num_slots": 1,
                        "file_offset": 64,
                        "token_count": 4,
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


def test_record_cached_store_blocks_allocates_when_chunked_prefill_completes():
    """Chunked-prefill cached steps allocate once all store blocks are known."""
    connector = _AllocatingSchedulerProbe()
    tokens = [1] * 12
    key = hash_tokens(tokens)
    connector.seed_pending_store("req", key, 12, [10, 11])

    class Cached:
        req_ids = ["req"]
        new_block_ids = [([12],)]
        resumed_req_ids = set()

    class Output:
        scheduled_cached_reqs = Cached()

    connector.record_cached_blocks(Output())

    pending_alloc, pending_stores = connector.pending_state
    assert connector.alloc_calls == [(key, 12, "m")]
    assert pending_alloc == {}
    assert pending_stores["req"]["block_ids"] == [10, 11, 12]
    assert pending_stores["req"]["chunk_key"] == key


def test_record_cached_store_blocks_appends_resumed_incremental_blocks():
    """Resumed chunked-prefill steps may report only newly allocated blocks."""
    connector = _AllocatingSchedulerProbe()
    tokens = [1] * 16
    key = hash_tokens(tokens)
    connector.seed_pending_store("req", key, 16, [10, 11])

    class Cached:
        req_ids = ["req"]
        new_block_ids = [([12, 13],)]
        resumed_req_ids = {"req"}

    class Output:
        scheduled_cached_reqs = Cached()

    connector.record_cached_blocks(Output())

    pending_alloc, pending_stores = connector.pending_state
    assert connector.alloc_calls == [(key, 16, "m")]
    assert pending_alloc == {}
    assert pending_stores["req"]["block_ids"] == [10, 11, 12, 13]


def test_record_cached_store_blocks_replaces_resumed_full_block_list():
    """Resumed chunked-prefill steps may also report the full block list."""
    connector = _AllocatingSchedulerProbe()
    tokens = [1] * 16
    key = hash_tokens(tokens)
    connector.seed_pending_store("req", key, 16, [10, 11])

    class Cached:
        req_ids = ["req"]
        new_block_ids = [([10, 11, 12, 13],)]
        resumed_req_ids = {"req"}

    class Output:
        scheduled_cached_reqs = Cached()

    connector.record_cached_blocks(Output())

    pending_alloc, pending_stores = connector.pending_state
    assert connector.alloc_calls == [(key, 16, "m")]
    assert pending_alloc == {}
    assert pending_stores["req"]["block_ids"] == [10, 11, 12, 13]


def test_filter_live_store_specs_drops_stale_allocations():
    """Scheduler drops store specs whose server allocation was already reused."""
    connector = _AllocatingSchedulerProbe()
    specs = {
        "a": ReqStoreSpec("live-key", 0, 1, [1], 0, 4),
        "b": ReqStoreSpec("stale-key", 1, 1, [2], 32, 4),
    }
    for req_id, spec in specs.items():
        connector.seed_pending_store_spec(req_id, spec)

    class Output:
        num_scheduled_tokens = {"a": 4, "b": 4}
        scheduled_cached_reqs = None

    meta = connector.build_connector_meta(Output())

    assert meta.reqs_to_store == {"a": specs["a"]}


def test_hash_tokens_deterministic():
    tokens = [1, 2, 3, 4]
    assert hash_tokens(tokens) == hash_tokens(tokens)
    assert hash_tokens(tokens) != hash_tokens([1, 2, 3, 5])


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


def test_build_staging_store_batches_caps_gpu_staging_bytes():
    """Store batches preserve chunk metadata while bounding staging size."""
    reqs_to_store = {
        "r0": ReqStoreSpec("k0", 10, 3, [5, 1, 7], 0, 12),
        "r1": ReqStoreSpec("k1", 20, 1, [8], 0, 4),
    }

    batches = _build_staging_store_batches(
        reqs_to_store=reqs_to_store,
        slot_size=32,
        max_batch_bytes=64,
    )

    assert [block_ids for block_ids, _ in batches] == [[5, 1], [7, 8]]
    assert [
        [
            (
                span.chunk_key,
                span.source_offset,
                span.nbytes,
                span.file_offset,
                span.start_slot,
                span.num_slots,
            )
            for span in spans
        ]
        for _, spans in batches
    ] == [
        [("k0", 0, 64, 320, 10, 3)],
        [("k0", 0, 32, 384, 10, 3), ("k1", 32, 32, 640, 20, 1)],
    ]


def test_derive_store_staging_limits_scale_with_vram(monkeypatch):
    """GPU staging caps consider device size and currently free VRAM."""

    class Props:
        def __init__(self, total_memory: int) -> None:
            self.total_memory = total_memory

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: Props(24 << 30),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (12 << 30, 24 << 30),
    )
    small_batch, small_pending = _derive_store_staging_limits(torch.device("cuda"))
    assert small_batch == max(
        MIN_STORE_STAGING_BYTES,
        min((24 << 30) // 160, (12 << 30) // 32),
    )
    assert small_pending == max(
        small_batch,
        min((24 << 30) // 80, (12 << 30) // 16),
    )

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: Props(80 << 30),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (64 << 30, 80 << 30),
    )
    large_batch, large_pending = _derive_store_staging_limits(torch.device("cuda"))
    assert large_batch == DEFAULT_STORE_STAGING_BYTES
    assert large_pending == DEFAULT_PENDING_STORE_STAGING_BYTES

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (8 << 30, 80 << 30),
    )
    tight_batch, tight_pending = _derive_store_staging_limits(torch.device("cuda"))
    assert tight_batch == (8 << 30) // 32
    assert tight_pending == (8 << 30) // 16


def test_cuda_staging_pool_reuses_preallocated_buffer():
    """Staging pool reuses its init-time allocation after release."""
    pool = CudaStagingPool(
        device=torch.device("cpu"),
        initial_bytes=128,
        max_buffer_bytes=256,
    )

    lease = pool.acquire(64)
    first_tensor = lease.tensor
    assert lease.view.numel() == 64
    lease.release()

    second = pool.acquire(32)
    assert second.tensor is first_tensor
    assert second.view.numel() == 32


def test_build_load_read_batches_splits_large_steps_by_staging_cap():
    """Load staging plans are bounded without changing block order."""
    reqs_to_load = {
        "r0": ReqLoadSpec("k0", 10, 3, [4, 5, 6], 0, 12),
        "r1": ReqLoadSpec("k1", 20, 1, [7], 0, 4),
    }

    batches = _build_load_read_batches(
        reqs_to_load=reqs_to_load,
        slot_size=32,
        max_batch_bytes=64,
    )

    assert [total_bytes for total_bytes, _, _ in batches] == [64, 64]
    assert [
        [spec.block_ids for _, _, spec in per_req] for _, _, per_req in batches
    ] == [[[4, 5]], [[6], [7]]]
    assert [[span["file_offset"] for span in spans] for _, spans, _ in batches] == [
        [320],
        [384, 640],
    ]


def test_synchronize_cuda_tensor_skips_cpu_tensor(monkeypatch):
    """CPU staging does not touch CUDA synchronization helpers."""

    def fail_current_stream(*args, **kwargs):
        raise AssertionError("CPU tensors must not synchronize CUDA streams")

    monkeypatch.setattr(torch.cuda, "current_stream", fail_current_stream)

    _synchronize_cuda_tensor(torch.empty(4))


def test_record_cuda_event_skips_cpu_tensor(monkeypatch):
    """CPU staging does not allocate CUDA events for deferred saves."""

    def fail_event(*args, **kwargs):
        raise AssertionError("CPU tensors must not create CUDA events")

    monkeypatch.setattr(torch.cuda, "Event", fail_event)

    assert _record_cuda_event(torch.empty(4)) is None


def test_record_cuda_event_uses_current_producer_stream(monkeypatch):
    """Deferred saves record the producer thread's current CUDA stream."""
    tensor = torch.empty(4, device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        event = _record_cuda_event(tensor)

    assert event is not None
    stream.synchronize()
    event.synchronize()


def test_build_load_read_plan_batches_requests_into_one_staging_buffer():
    """Load spans target one combined staging tensor while preserving req ranges."""
    reqs_to_load = {
        "r0": ReqLoadSpec("k0", 10, 2, [4, 5], 0, 8),
        "r1": ReqLoadSpec("k1", 20, 1, [6], 0, 4),
    }

    total_bytes, spans, per_req = _build_load_read_plan(reqs_to_load, slot_size=32)

    assert total_bytes == 96
    assert spans == [
        {"target_offset": 0, "nbytes": 64, "file_offset": 320},
        {"target_offset": 64, "nbytes": 32, "file_offset": 640},
    ]
    assert [(start, end, spec.chunk_key) for start, end, spec in per_req] == [
        (0, 64, "k0"),
        (64, 96, "k1"),
    ]


def test_build_load_copy_runs_merges_same_transform_ranges():
    """Load copy runs merge adjacent requests only when transforms match."""
    reqs_to_load = {
        "r0": ReqLoadSpec("k0", 10, 2, [4, 5], 0, 8, pos_offset=0),
        "r1": ReqLoadSpec("k1", 20, 1, [6], 0, 4, pos_offset=0),
        "r2": ReqLoadSpec("k2", 30, 1, [7], 0, 4, pos_offset=16),
    }
    _, _, per_req = _build_load_read_plan(reqs_to_load, slot_size=32)

    runs = _build_load_copy_runs(per_req)

    assert [(run.start, run.end, run.block_ids, run.pos_offset) for run in runs] == [
        (0, 96, [4, 5, 6], 0),
        (96, 128, [7], 16),
    ]


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

    from daser.transfer.gds import GDSTransferLayer

    gds = GDSTransferLayer(store_path)
    data = kv[:, 0].contiguous()
    cp = cupy.asarray(data)
    await gds.write_async(cp, file_offset=0)

    recv = torch.zeros_like(kv[:, 0])
    cp_recv = cupy.asarray(recv)
    await gds.read_into_async(cp_recv, file_offset=0)
    assert torch.equal(kv[:, 0], recv)
    gds.close()
