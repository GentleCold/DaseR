# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import cupy
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from daser.connector.daser_connector import DaserConnector
from daser.connector.helpers import (
    ROLLING_PREFIX_SEED,
    PendingStore,
    hash_tokens,
    rolling_prefix_key,
    rolling_prefix_keys,
)
from daser.connector.metadata import (
    DaserConnectorMeta,
    ReqLoadSpec,
    ReqStoreSpec,
    StoreWriteSpan,
)
from daser.connector.scheduler.lifecycle import RequestLifecycle
from daser.connector.scheduler.planning import (
    _block_ids_for_chunk,
    _contiguous_prefix_tokens,
    _trim_chunk_to_external_window,
)
from daser.connector.scheduler.reuse import PrefixReuseStrategy
from daser.connector.worker.load import (
    build_load_copy_runs as _build_load_copy_runs,
)
from daser.connector.worker.load import (
    build_load_read_batches as _build_load_read_batches,
)
from daser.connector.worker.load import (
    build_load_read_plan as _build_load_read_plan,
)
from daser.connector.worker.memory import (
    DEFAULT_STAGING_BUDGET_BYTES,
    DEFAULT_STORE_STAGING_BYTES,
    MIN_STORE_STAGING_BYTES,
    FixedCudaStagingPool,
    derive_staging_layout,
)
from daser.connector.worker.runtime import WorkerRuntime
from daser.connector.worker.staging import (
    DEFAULT_ROPE_DELTA_SCALE,
)
from daser.connector.worker.staging import (
    apply_rope_delta_to_key_block as _apply_rope_delta_to_key_block,
)
from daser.connector.worker.staging import (
    copy_staging_to_kv_cache as _copy_staging_to_kv_cache,
)
from daser.connector.worker.staging import (
    record_cuda_event as _record_cuda_event,
)
from daser.connector.worker.staging import (
    synchronize_cuda_tensor as _synchronize_cuda_tensor,
)
from daser.connector.worker.store import (
    StagedStoreBatch,
    StorePipeline,
)
from daser.connector.worker.store import (
    build_staging_store_batches as _build_staging_store_batches,
)

BLOCK_TOKENS = 4
NUM_LAYERS = 2

pytestmark = pytest.mark.integration


def rolling_keys(tokens: list[int], block_tokens: int) -> list[str]:
    """Return expected rolling-prefix keys for test assertions."""
    keys: list[str] = []
    key = ROLLING_PREFIX_SEED
    aligned = (len(tokens) // block_tokens) * block_tokens
    for start in range(0, aligned, block_tokens):
        key = rolling_prefix_key(key, tokens[start : start + block_tokens])
        keys.append(key)
    return keys


def test_rolling_prefix_keys_match_single_step_helper() -> None:
    """Batched rolling-prefix keys preserve the existing key sequence."""
    tokens = list(range(32))

    assert rolling_prefix_keys(tokens, block_tokens=8) == rolling_keys(
        tokens,
        block_tokens=8,
    )


def test_scheduler_defers_request_until_prefetch_completes(monkeypatch) -> None:
    """The scheduler retries a request after asynchronous L2-to-L1 promotion."""

    class LookupIPC:
        lookup_calls = 0

        def lookup(self, tokens, model_id, **kwargs):
            del tokens, model_id, kwargs
            self.lookup_calls += 1
            return [
                {
                    "chunk_key": "cached",
                    "start_slot": 1,
                    "num_slots": 2,
                    "file_offset": 32,
                    "token_count": 8,
                    "target_token_start": 0,
                    "pos_offset": 0,
                }
            ]

    prefetch_calls = []

    def fake_prefetch(socket_path, spans):
        prefetch_calls.append((socket_path, spans))
        return {"requested_bytes": 64, "l1_bytes": 32, "l2_bytes": 32}

    monkeypatch.setattr(
        "daser.connector.scheduler.lifecycle._prefetch_external_spans",
        fake_prefetch,
    )
    ipc = LookupIPC()
    lifecycle = RequestLifecycle(
        ipc_client=ipc,
        socket_path="/unused/daser.sock",
        block_tokens=4,
        slot_size=32,
        model_id="model",
        cache_reuse_mode="chunk",
        runtime_config_ready=True,
        prefetch_max_requests=1,
    )
    request = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=list(range(12)),
        kv_transfer_params={"daser_skip_save": True},
    )
    try:
        assert lifecycle.get_num_new_matched_tokens(request, 0) == (None, True)
        for _ in range(100):
            result = lifecycle.get_num_new_matched_tokens(request, 0)
            if result != (None, True):
                break
        assert result == (8, True)
        assert prefetch_calls == [
            ("/unused/daser.sock", [{"file_offset": 32, "nbytes": 64}])
        ]
        assert ipc.lookup_calls == 1
    finally:
        lifecycle.shutdown()


class _SchedulerProbe(RequestLifecycle):
    """Scheduler-side probe that can emulate deferred runtime config."""

    def __init__(self, ipc_client) -> None:
        self._runtime_config_ready = False
        self._block_tokens = 16
        self._slot_size = 32
        self._model_id = "default"
        self._req_tokens = {}
        self._pending_loads = {}
        self._pending_alloc = {}
        self._ipc_sync = ipc_client
        self.refresh_count = 0

    def _refresh_runtime_config(self) -> None:
        self.refresh_count += 1
        self._runtime_config_ready = True
        self._model_id = "served-model"

    def mark_runtime_ready(self, model_id: str) -> None:
        """Seed runtime config state through a public test helper."""
        self._runtime_config_ready = True
        self._model_id = model_id


class _AllocatingSchedulerProbe(RequestLifecycle):
    """Minimal scheduler probe that records allocation RPCs."""

    def __init__(self) -> None:
        self._block_tokens = BLOCK_TOKENS
        self._slot_size = 32
        self._init_reuse_strategy()
        self._pending_loads = {}
        self._pending_stores = {}
        self._pending_alloc = {}
        self._pending_async_saves = set()
        self._req_tokens = {}
        self._model_id = "m"
        self.alloc_calls: list[tuple[str, int, str]] = []
        self.released_allocations: list[tuple[str, int, int]] = []
        self._ipc_sync = self

    def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
        """Record an allocation call and return server-style metadata."""
        self.alloc_calls.append((chunk_key, token_count, model_id))
        return {
            "start_slot": 5,
            "file_offset": 160,
            "pos_offset": 0,
        }

    def alloc_chunks(self, chunks: list[dict], model_id: str) -> list[dict]:
        """Record a batched allocation call and return server-style metadata."""
        self.alloc_calls.append(("batch", len(chunks), model_id))
        return [
            {
                "chunk_key": str(chunk["chunk_key"]),
                "start_slot": 20 + idx,
                "file_offset": (20 + idx) * self._slot_size,
                "pos_offset": 0,
            }
            for idx, chunk in enumerate(chunks)
        ]

    def live_allocations(self, allocations: list[dict]) -> set[str]:
        """Return allocations whose chunk key starts with ``live``."""
        return {
            str(alloc["chunk_key"])
            for alloc in allocations
            if str(alloc["chunk_key"]).startswith("live")
        }

    def release_chunk_writer(
        self,
        chunk_key: str,
        start_slot: int,
        num_slots: int,
    ) -> None:
        """Record release calls for discarded pending stores."""
        self.released_allocations.append((chunk_key, start_slot, num_slots))

    def seed_pending_store(
        self, req_id: str, chunk_key: str, token_count: int, block_ids: list[int]
    ) -> None:
        """Seed pending scheduler state for a store allocation test."""
        self._req_tokens[req_id] = [1] * token_count
        self._pending_alloc[req_id] = PendingStore(
            chunk_key=chunk_key,
            token_count=token_count,
            block_ids=block_ids,
        )

    def record_cached_blocks(self, scheduler_output) -> None:
        """Expose cached-block recording through a public test helper."""
        self._record_cached_store_blocks(scheduler_output)

    def seed_tokens(self, req_id: str, tokens: list[int]) -> None:
        """Seed request tokens for scheduler allocation tests."""
        self._req_tokens[req_id] = tokens

    def maybe_allocate_store_for_test(self, req_id: str) -> None:
        """Expose pending-store allocation through a public test helper."""
        self._maybe_allocate_pending_store(req_id, self._pending_alloc[req_id])

    def use_prefix_reuse_strategy(self) -> None:
        """Switch this test probe to prefix cache reuse."""
        self._cache_reuse_strategy = PrefixReuseStrategy(self._block_tokens)

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

    def request_finished_for_test(self, req_id: str):
        """Expose scheduler request-finished behavior through a test request."""

        class _Request:
            request_id = req_id

        return self.request_finished(_Request(), [])

    def update_connector_output_for_test(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
    ):
        """Expose connector output handling for async save completions."""

        class _Output:
            def __init__(
                self,
                sending_ids: set[str] | None,
                recving_ids: set[str] | None,
            ) -> None:
                self.finished_sending = sending_ids
                self.finished_recving = recving_ids

        self.update_connector_output(_Output(finished_sending, finished_recving))

    def has_req_tokens(self, req_id: str) -> bool:
        """Return whether scheduler token state is still held for a request."""
        return req_id in self._req_tokens

    @property
    def pending_state(self) -> tuple[dict, dict]:
        """Return pending allocation and store state for assertions."""
        return self._pending_alloc, self._pending_stores


class _AllocChunkOnlyIPC:
    """IPC shim that exposes only alloc_chunk for fallback allocation tests."""

    def __init__(self, owner: _AllocatingSchedulerProbe) -> None:
        self._owner = owner

    def alloc_chunk(
        self,
        chunk_key: str,
        token_count: int,
        model_id: str,
    ) -> dict:
        """Forward single-chunk allocation calls to the owner probe."""
        return self._owner.alloc_chunk(chunk_key, token_count, model_id)


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


def test_connector_requests_cross_layer_nhd_layout() -> None:
    """DaseR asks vLLM for block-major cross-layer KV cache layout."""
    assert DaserConnector.get_required_kvcache_layout(object()) == "NHD"
    assert DaserConnector.prefer_cross_layer_blocks.fget(object()) is True


def test_connector_records_cache_reuse_mode_from_runtime_config(monkeypatch, tmp_path):
    """Scheduler should follow the server-selected cache reuse strategy."""

    class DummyIPCClient:
        def __init__(self, socket_path):
            self.socket_path = socket_path

        def get_runtime_config(self):
            return {
                "store_path": str(tmp_path / "daser.store"),
                "slot_size": 1024,
                "block_tokens": 4,
                "model_id": "served-model",
                "cache_reuse_mode": "prefix",
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

    connector = DaserConnector(
        DummyVLLMConfig(),
        role=KVConnectorRole.SCHEDULER,
    )

    assert isinstance(  # noqa: SLF001
        connector._request_lifecycle._reuse_strategy(),  # noqa: SLF001
        PrefixReuseStrategy,
    )


def test_staging_layout_respects_available_cuda_headroom(monkeypatch) -> None:
    """Combined staging pools stay within available CUDA headroom."""
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=80 << 30),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: ((4 << 30), 80 << 30),
    )

    buffer_bytes, load_depth, store_depth, allocated = derive_staging_layout(
        torch.device("cuda"),
        local_slot_size=64 << 20,
        max_load_inflight=8,
        reserve_bytes=1 << 30,
    )

    assert (buffer_bytes, load_depth, store_depth) == ((4 << 30) // 10, 5, 2)
    assert allocated == buffer_bytes * 7
    assert allocated <= (4 << 30) - (1 << 30)


def test_worker_transfer_ready_allows_skip_l2_without_store_path() -> None:
    """L1-only mode has no store path but still has a valid transfer config."""

    class Pipeline:
        def __init__(self) -> None:
            self.initialized = False

        def configure_rank_geometry(self, *args: int) -> None:
            del args

        def initialize_transfer(self) -> None:
            self.initialized = True

    connector = WorkerRuntime.__new__(WorkerRuntime)
    connector._transfer_ready = False  # noqa: SLF001
    connector._pipelines_initialized = False  # noqa: SLF001
    connector._store_path = ""  # noqa: SLF001
    connector._slot_size = 1024  # noqa: SLF001
    connector._local_slot_size = 1024  # noqa: SLF001
    connector._tp_size = 1  # noqa: SLF001
    connector._server_tp_size = 1  # noqa: SLF001
    connector._tp_rank = 0  # noqa: SLF001
    connector._rank_stride_bytes = 0  # noqa: SLF001
    connector._transfer_mode = "iouring"  # noqa: SLF001
    connector._skip_l2 = True  # noqa: SLF001
    connector._refresh_runtime_config = lambda: None  # noqa: SLF001
    connector._load_pipeline = Pipeline()  # noqa: SLF001
    connector._store_pipeline = Pipeline()  # noqa: SLF001

    assert connector._ensure_transfer_ready() is True  # noqa: SLF001
    assert connector._transfer_ready is True  # noqa: SLF001
    assert connector._pipelines_initialized is True  # noqa: SLF001
    assert connector._load_pipeline.initialized is True  # noqa: SLF001
    assert connector._store_pipeline.initialized is True  # noqa: SLF001


def test_worker_transfer_ready_propagates_refreshed_tp_geometry() -> None:
    """Delayed TP geometry must reach pipelines before transfer initialization."""

    class Pipeline:
        def __init__(self) -> None:
            self.geometry: tuple[int, ...] | None = None
            self.initialized = False

        def configure_rank_geometry(self, *args: int) -> None:
            self.geometry = args

        def initialize_transfer(self) -> None:
            self.initialized = True

    connector = WorkerRuntime.__new__(WorkerRuntime)
    connector._transfer_ready = False  # noqa: SLF001
    connector._pipelines_initialized = False  # noqa: SLF001
    connector._store_path = ""  # noqa: SLF001
    connector._slot_size = 0  # noqa: SLF001
    connector._local_slot_size = 512  # noqa: SLF001
    connector._tp_size = 2  # noqa: SLF001
    connector._server_tp_size = 1  # noqa: SLF001
    connector._tp_rank = 1  # noqa: SLF001
    connector._rank_stride_bytes = 0  # noqa: SLF001
    connector._transfer_mode = "iouring"  # noqa: SLF001
    connector._skip_l2 = True  # noqa: SLF001

    def refresh() -> None:
        connector._slot_size = 1024  # noqa: SLF001
        connector._server_tp_size = 2  # noqa: SLF001
        connector._rank_stride_bytes = 8192  # noqa: SLF001

    connector._refresh_runtime_config = refresh  # noqa: SLF001
    connector._load_pipeline = Pipeline()  # noqa: SLF001
    connector._store_pipeline = Pipeline()  # noqa: SLF001

    assert connector._ensure_transfer_ready() is True  # noqa: SLF001
    assert connector._load_pipeline.geometry == (8192, 1)  # noqa: SLF001
    assert connector._store_pipeline.geometry == (8192, 1, 2)  # noqa: SLF001
    assert connector._load_pipeline.initialized is True  # noqa: SLF001
    assert connector._store_pipeline.initialized is True  # noqa: SLF001


def test_worker_runtime_refreshes_l1_only_transfer_config(monkeypatch) -> None:
    """Deferred worker config refresh must propagate L1-only transfer settings."""

    class DummyIPCClient:
        def __init__(self, socket_path: str) -> None:
            self.socket_path = socket_path

        def get_runtime_config(self) -> dict[str, object]:
            return {
                "store_path": "",
                "slot_size": 1024,
                "tensor_parallel_size": 2,
                "rank_stride_bytes": 512,
                "transfer_mode": "iouring",
                "skip_l2": True,
            }

        def close(self) -> None:
            return

    monkeypatch.setattr(
        "daser.connector.worker.runtime.IPCClientSync",
        DummyIPCClient,
    )
    connector = WorkerRuntime.__new__(WorkerRuntime)
    connector._socket_path = "/unused/daser.sock"  # noqa: SLF001
    connector._store_path = ""  # noqa: SLF001
    connector._slot_size = 0  # noqa: SLF001
    connector._server_tp_size = 1  # noqa: SLF001
    connector._rank_stride_bytes = 0  # noqa: SLF001
    connector._transfer_mode = "gds"  # noqa: SLF001
    connector._skip_l2 = False  # noqa: SLF001

    connector._refresh_runtime_config()  # noqa: SLF001

    assert connector._slot_size == 1024  # noqa: SLF001
    assert connector._server_tp_size == 2  # noqa: SLF001
    assert connector._rank_stride_bytes == 512  # noqa: SLF001
    assert connector._transfer_mode == "iouring"  # noqa: SLF001
    assert connector._skip_l2 is True  # noqa: SLF001


def test_request_lifecycle_rebuilds_prefix_keys_after_block_size_refresh() -> None:
    """Deferred geometry refresh must rebuild same-mode rolling-prefix keys."""

    class DummyIPCClient:
        def get_runtime_config(self) -> dict[str, object]:
            return {
                "slot_size": 1024,
                "block_tokens": 128,
                "model_id": "served-model",
                "cache_reuse_mode": "prefix",
            }

    lifecycle = RequestLifecycle(
        ipc_client=DummyIPCClient(),
        block_tokens=16,
        slot_size=0,
        model_id="default",
        cache_reuse_mode="prefix",
        runtime_config_ready=False,
    )
    lifecycle._refresh_runtime_config()  # noqa: SLF001
    tokens = list(range(256))
    pending = lifecycle._reuse_strategy().prepare_store(tokens, 256)  # noqa: SLF001

    assert pending is not None
    pending.block_ids = [7, 8]
    plan = lifecycle._reuse_strategy().plan_store(  # noqa: SLF001
        "req",
        pending,
        tokens,
        set(),
    )
    assert [intent.chunk_key for intent in plan.intents] == rolling_keys(tokens, 128)


def test_scheduler_runtime_config_is_owned_by_request_lifecycle(monkeypatch):
    """Scheduler readiness is refreshed directly on its lifecycle owner."""

    class DummyIPCClient:
        def __init__(self, socket_path):
            self.socket_path = socket_path

        def get_runtime_config(self):
            return {
                "store_path": "",
                "slot_size": 1024,
                "block_tokens": 4,
                "model_id": "served-model",
                "cache_reuse_mode": "prefix",
                "transfer_mode": "iouring",
                "skip_l2": True,
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

    connector = DaserConnector(
        DummyVLLMConfig(),
        role=KVConnectorRole.SCHEDULER,
    )

    lifecycle = connector._request_lifecycle  # noqa: SLF001
    assert lifecycle._runtime_config_ready is True  # noqa: SLF001
    assert lifecycle._model_id == "served-model"  # noqa: SLF001


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


def test_scheduler_refreshes_runtime_config_after_lookup_transport_failure():
    """A restarted DaseR server is rediscovered after one failed lookup."""

    class DummyIPCClient:
        def __init__(self) -> None:
            self.calls = 0
            self.model_ids: list[str] = []

        def lookup(self, _tokens, model_id):
            self.calls += 1
            self.model_ids.append(model_id)
            if self.calls == 1:
                raise RuntimeError("transport failure")
            return []

    class DummyRequest:
        request_id = "request-1"
        prompt_token_ids = list(range(16))
        kv_transfer_params = {"daser_skip_save": True}

    ipc_client = DummyIPCClient()
    connector = _SchedulerProbe(ipc_client)
    connector.mark_runtime_ready("old-model")

    first = connector.get_num_new_matched_tokens(
        DummyRequest(),
        num_computed_tokens=0,
    )
    second = connector.get_num_new_matched_tokens(
        DummyRequest(),
        num_computed_tokens=0,
    )

    assert first == (0, False)
    assert second == (0, False)
    assert connector.refresh_count == 1
    assert ipc_client.model_ids == ["old-model", "served-model"]


def test_scheduler_still_loads_from_daser_when_vllm_prefix_cache_is_enabled():
    """DaseR loads stay active when vLLM has no local prefix-cache hit."""

    class DummyIPCClient:
        def __init__(self) -> None:
            self.lookups = []

        def lookup(self, tokens, model_id):
            self.lookups.append((list(tokens), model_id))
            return [
                {
                    "chunk_key": "hit",
                    "start_slot": 0,
                    "num_slots": 2,
                    "file_offset": 0,
                    "token_count": 32,
                    "target_token_start": 0,
                    "pos_offset": 0,
                }
            ]

    class DummyRequest:
        request_id = "request-1"
        prompt_token_ids = list(range(32))
        kv_transfer_params = {}

    ipc_client = DummyIPCClient()
    connector = _SchedulerProbe(ipc_client)
    connector.mark_runtime_ready("served-model")

    assert connector.get_num_new_matched_tokens(
        DummyRequest(),
        num_computed_tokens=0,
    ) == (31, True)
    assert ipc_client.lookups == [(list(range(32)), "served-model")]
    assert "request-1" in connector._pending_loads  # noqa: SLF001
    pending_store = connector._pending_alloc["request-1"]  # noqa: SLF001
    assert pending_store.token_count == 32


def test_scheduler_reports_async_load_without_vllm_prefix_cache():
    """DaseR reports async load intent when vLLM prefix cache is off."""

    class DummyIPCClient:
        def lookup(self, tokens, model_id):
            del tokens, model_id
            return [
                {
                    "chunk_key": "hit",
                    "start_slot": 0,
                    "num_slots": 2,
                    "file_offset": 0,
                    "token_count": 32,
                    "target_token_start": 0,
                    "pos_offset": 0,
                }
            ]

    class DummyRequest:
        request_id = "request-1"
        prompt_token_ids = list(range(32))
        kv_transfer_params = {}

    connector = _SchedulerProbe(DummyIPCClient())
    connector.mark_runtime_ready("served-model")

    assert connector.get_num_new_matched_tokens(
        DummyRequest(),
        num_computed_tokens=0,
    ) == (31, True)


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
    assert chunk["block_ids"] == [11, 12]


def test_update_state_after_alloc_single_hit_uses_external_window():
    """Single-prefix hit maps the external suffix onto absolute request blocks."""

    class MockConnector(RequestLifecycle):
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

    RequestLifecycle.update_state_after_alloc(
        connector,
        MockRequest(),
        MockBlocks(),
        num_external_tokens=8,
    )

    chunk = connector.pending_loads["req"]
    assert chunk["start_slot"] == 101
    assert chunk["file_offset"] == 3232
    assert chunk["num_slots"] == 2
    assert chunk["block_ids"] == [11, 12]


def test_lookup_sends_external_prefix_query_metric_hint():
    """Connector sends vLLM external prefix query count with lookup RPC."""

    class MockIPC:
        def __init__(self) -> None:
            self.lookup_calls: list[tuple[list[int], str, int | None, int]] = []

        def lookup(
            self,
            tokens,
            model_id,
            external_prefix_queries=None,
            num_computed_tokens=0,
        ):
            self.lookup_calls.append(
                (
                    list(tokens),
                    model_id,
                    external_prefix_queries,
                    num_computed_tokens,
                )
            )
            return [
                {
                    "chunk_key": "k0",
                    "start_slot": 100,
                    "num_slots": 3,
                    "file_offset": 3200,
                    "token_count": 12,
                    "target_token_start": 0,
                    "pos_offset": 0,
                }
            ]

    class MockConnector(RequestLifecycle):
        def __init__(self) -> None:
            self._runtime_config_ready = True
            self._block_tokens = BLOCK_TOKENS
            self._slot_size = 32
            self._model_id = "m"
            self._cache_reuse_strategy = PrefixReuseStrategy(self._block_tokens)
            self._pending_loads = {
                "req": {
                    "chunk_key": "k0",
                    "start_slot": 100,
                    "num_slots": 3,
                    "file_offset": 3200,
                    "token_count": 12,
                    "target_token_start": 0,
                    "num_computed_tokens": 4,
                }
            }
            self._pending_alloc = {}
            self._pending_stores = {}
            self._req_tokens = {"req": list(range(20))}
            self._ipc_sync = MockIPC()

        @property
        def lookup_calls(self) -> list[tuple[list[int], str, int | None, int]]:
            return self._ipc_sync.lookup_calls

    class MockRequest:
        request_id = "req"
        prompt_token_ids = list(range(20))
        kv_transfer_params = {"daser_skip_save": True}

    connector = MockConnector()

    assert connector.get_num_new_matched_tokens(MockRequest(), 4) == (8, True)
    assert connector.lookup_calls == [(list(range(20)), "m", 16, 4)]


def test_update_state_after_alloc_multi_hit_trims_each_chunk_to_external_window():
    """Multi-chunk hits map onto absolute request block positions."""

    class MockConnector(RequestLifecycle):
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

    RequestLifecycle.update_state_after_alloc(
        connector,
        MockRequest(),
        MockBlocks(),
        num_external_tokens=8,
    )

    chunks = connector.pending_loads["req"]
    assert chunks["0"]["start_slot"] == 101
    assert chunks["0"]["num_slots"] == 1
    assert chunks["0"]["block_ids"] == [11]
    assert chunks["1"]["start_slot"] == 200
    assert chunks["1"]["num_slots"] == 1
    assert chunks["1"]["block_ids"] == [12]


def test_contiguous_prefix_tokens_handles_partially_computed_prefix():
    chunks = [{"target_token_start": 0, "token_count": 96}]
    assert _contiguous_prefix_tokens(chunks, num_computed_tokens=16) == 80


def test_single_non_prefix_chunk_does_not_credit_missing_prefix():
    """A lone chunk hit can only provide external tokens at its target offset."""

    class MockConnector(_SchedulerProbe):
        def __init__(self) -> None:
            super().__init__(ipc_client=self)
            self._runtime_config_ready = True
            self._block_tokens = BLOCK_TOKENS

        def lookup(self, tokens, model_id):
            return [
                {
                    "chunk_key": "doc",
                    "start_slot": 5,
                    "num_slots": 1,
                    "file_offset": 160,
                    "token_count": 4,
                    "target_token_start": 4,
                    "pos_offset": 4,
                }
            ]

        @property
        def pending_loads(self) -> dict:
            return self._pending_loads

    class MockRequest:
        request_id = "req"
        prompt_token_ids = list(range(12))
        kv_transfer_params = {"daser_skip_save": True}

    connector = MockConnector()

    assert connector.get_num_new_matched_tokens(MockRequest(), 0) == (0, False)
    assert connector.pending_loads == {}


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


def _apply_rope_delta_reference(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> None:
    """Apply a local PyTorch RoPE delta reference for tests."""
    if not is_neox_style:
        raise NotImplementedError("test reference only covers NeoX RoPE")
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=key_block.device)
            / rotary_dim
        )
    )
    freqs = delta * inv_freq
    compute = key_block[..., :rotary_dim].float()
    cos = freqs.cos().view(*([1] * (compute.dim() - 1)), -1)
    sin = freqs.sin().view(*([1] * (compute.dim() - 1)), -1)
    x1, x2 = torch.chunk(compute, 2, dim=-1)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    key_block[..., :rotary_dim].copy_(rotated.to(key_block.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rope_delta_rotates_key_block_to_target_positions():
    pytest.importorskip("tilelang")
    raw = torch.randn(4, 2, 8, dtype=torch.float32, device="cuda")
    source_positions = torch.arange(4, device="cuda")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rope_delta_rotates_key_block_batch_to_target_positions():
    pytest.importorskip("tilelang")
    raw = torch.randn(3, 4, 2, 8, dtype=torch.float32, device="cuda")
    source_positions = torch.arange(4, device="cuda")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rope_delta_leaves_non_rotary_tail_unchanged():
    pytest.importorskip("tilelang")
    raw = torch.randn(4, 2, 8, dtype=torch.float32, device="cuda")
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


def test_apply_rope_delta_raises_when_tilelang_unavailable(monkeypatch):
    import builtins

    from daser.ops import rope_apply

    class FakeCudaTensor:
        shape = (4, 2, 8)
        dtype = torch.float32
        device = torch.device("cuda")

        def is_contiguous(self) -> bool:
            return True

        def dim(self) -> int:
            return len(self.shape)

        def numel(self) -> int:
            return 64

        def reshape(self, *shape: int) -> "FakeCudaTensor":
            return self

    real_import = builtins.__import__

    def missing_tilelang(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tilelang":
            raise ImportError("No module named 'tilelang'")
        return real_import(name, globals, locals, fromlist, level)

    rope_apply.clear_rope_apply_cache()
    monkeypatch.setattr(builtins, "__import__", missing_tilelang)

    with pytest.raises(ImportError, match="tilelang"):
        rope_apply.apply_rope_delta_to_key_block(
            FakeCudaTensor(),
            delta=7,
            rope_base=10000.0,
            rotary_dim=8,
            is_neox_style=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rope_delta_tilelang_matches_naive_cuda():
    pytest.importorskip("tilelang")
    from daser.ops import rope_apply

    raw = torch.randn(4, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
    expected = raw.clone()
    _apply_rope_delta_reference(
        expected,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )
    actual = raw.clone()

    rope_apply.clear_rope_apply_cache()
    rope_apply.apply_rope_delta_to_key_block(
        actual,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )
    torch.cuda.synchronize(actual.device)

    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_apply_rope_delta_tilelang_reuses_dynamic_kernel_across_shapes(monkeypatch):
    from daser.ops import rope_apply

    class FakeCudaTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = torch.bfloat16
            self.device = torch.device("cuda")

        def is_contiguous(self) -> bool:
            return True

        def dim(self) -> int:
            return len(self.shape)

        def numel(self) -> int:
            numel = 1
            for extent in self.shape:
                numel *= extent
            return numel

        def stride(self, dim: int | None = None):
            strides = []
            stride = 1
            for extent in reversed(self.shape):
                strides.insert(0, stride)
                stride *= extent
            if dim is None:
                return tuple(strides)
            return strides[dim]

        def reshape(self, *shape: int) -> "FakeCudaTensor":
            return FakeCudaTensor(shape)

    compile_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(fn)

        def wrapped(key_block, delta, rope_base):
            assert key_block.dim() == 2

        return wrapped

    fake_tilelang = type("FakeTileLang", (), {"compile": staticmethod(fake_compile)})
    monkeypatch.setitem(__import__("sys").modules, "tilelang", fake_tilelang)
    monkeypatch.setattr(
        rope_apply,
        "_build_tilelang_kernel",
        lambda **kwargs: object(),
    )

    rope_apply.clear_rope_apply_cache()
    first = FakeCudaTensor((4, 16, 2, 128))
    second = FakeCudaTensor((11, 16, 2, 128))

    rope_apply.apply_rope_delta_to_key_block(
        first,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )
    rope_apply.apply_rope_delta_to_key_block(
        second,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )

    assert len(compile_calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rope_delta_tilelang_dynamic_kernel_matches_multiple_shapes_cuda():
    pytest.importorskip("tilelang")
    from daser.ops import rope_apply

    rope_apply.clear_rope_apply_cache()
    for block_count in (4, 11):
        raw = torch.randn(
            block_count,
            16,
            2,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        expected = raw.clone()
        _apply_rope_delta_reference(
            expected,
            delta=128,
            rope_base=1000000.0,
            rotary_dim=128,
            is_neox_style=True,
        )
        actual = raw.clone()

        rope_apply.apply_rope_delta_to_key_block(
            actual,
            delta=128,
            rope_base=1000000.0,
            rotary_dim=128,
            is_neox_style=True,
        )
        torch.cuda.synchronize(actual.device)

        assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_rope_delta_tilelang_matches_cross_layer_staging_cuda():
    """TileLang rotates only K inside full contiguous cross-layer staging KV."""
    pytest.importorskip("tilelang")
    from daser.ops.rope_apply import (
        apply_rope_delta_to_kv_key_block,
        clear_rope_apply_cache,
    )

    clear_rope_apply_cache()
    raw = torch.randn(
        4,
        3,
        2,
        16,
        2,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = raw.clone()
    _apply_rope_delta_reference(
        expected[:, :, 0],
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )
    actual = raw.clone()

    apply_rope_delta_to_kv_key_block(
        actual,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )
    torch.cuda.synchronize(actual.device)

    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_restore_cross_layer_kv_cache_table_tilelang_matches_reference_cuda():
    """Fused TileLang table restore copies V and rotates K into cross-layer KV."""
    pytest.importorskip("tilelang")
    from daser.ops.rope_apply import (
        build_rope_delta_tables,
        clear_rope_apply_cache,
        restore_cross_layer_kv_cache_table,
    )

    clear_rope_apply_cache()
    staging_kv = torch.randn(
        4,
        3,
        2,
        16,
        2,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = staging_kv.clone()
    _apply_rope_delta_reference(
        expected[:, :, 0],
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
        is_neox_style=True,
    )
    actual = torch.empty_like(staging_kv)
    cos_table, sin_table = build_rope_delta_tables(
        staging_kv.device,
        delta=128,
        rope_base=1000000.0,
        rotary_dim=128,
    )

    restore_cross_layer_kv_cache_table(
        staging_kv,
        actual,
        cos_table=cos_table,
        sin_table=sin_table,
        rotary_dim=128,
        is_neox_style=True,
    )
    torch.cuda.synchronize(actual.device)

    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_register_kv_caches_warms_dynamic_rope_apply_once(monkeypatch):
    from daser.connector.worker import runtime as worker

    class Probe(WorkerRuntime):
        def __init__(self) -> None:
            self._slot_size = 0
            self._tp_size = 1
            self._server_tp_size = 1
            self._tp_rank = 0
            self._rank_stride_bytes = 0
            self._rope_rotary_dim = 8
            self._rope_base = 10000.0
            self._rope_is_neox_style = True

        def _ensure_transfer_ready(self) -> bool:
            return False

        def _configure_pipelines(self, sample: torch.Tensor) -> int:
            del sample
            return 1

    calls = []
    monkeypatch.setattr(
        worker,
        "_warm_rope_apply_backends",
        lambda **kwargs: calls.append(kwargs),
    )

    kv_cache = torch.zeros(2, 3, 4, 5, 8, dtype=torch.float32)
    Probe().register_kv_caches({"layer": kv_cache})

    assert calls == [
        {
            "device": kv_cache.device,
            "dtype": kv_cache.dtype,
            "block_tokens": 4,
            "heads": 5,
            "head_dim": 8,
            "rotary_dim": 8,
            "rope_base": 10000.0,
            "is_neox_style": True,
        }
    ]


def test_register_cross_layers_kv_cache_preserves_layer_order(monkeypatch):
    """Worker registration keeps vLLM layer names for slot-major staging."""
    from daser.connector.worker import runtime as worker

    class Group:
        layer_names = ["layer.0", "layer.1"]

    class Config:
        kv_cache_groups = [Group()]

    class Probe(WorkerRuntime):
        def __init__(self) -> None:
            self._kv_cache_config = Config()
            self._slot_size = 0
            self._tp_size = 1
            self._server_tp_size = 1
            self._tp_rank = 0
            self._rank_stride_bytes = 0
            self._rope_rotary_dim = 8
            self._rope_base = 10000.0
            self._rope_is_neox_style = True
            self._transfer_ready = True

        def _ensure_transfer_ready(self) -> bool:
            return True

        def _init_server_transfer(self) -> None:
            return

        def _configure_pipelines(self, sample: torch.Tensor) -> int:
            del sample
            return 1

        @property
        def registration_state(self):
            return (
                self._layer_names,
                self._layer_idx_map,
                self._kv_caches,
                self._slot_size,
            )

    monkeypatch.setattr(worker, "_warm_rope_apply_backends", lambda **kwargs: None)

    kv_cache = torch.zeros((8, 2, 2, 4, 2, 8), dtype=torch.float32)
    probe = Probe()
    probe.register_cross_layers_kv_cache(kv_cache, attn_backend=object)
    layer_names, layer_idx_map, kv_caches, slot_size = probe.registration_state

    assert layer_names == ["layer.0", "layer.1"]
    assert layer_idx_map == {"layer.0": 0, "layer.1": 1}
    assert kv_caches["__cross_layers__"] is kv_cache
    assert slot_size == kv_cache[0].nbytes


def test_update_state_after_alloc_skips_chunks_beyond_external_prefix():
    class MockConnector(RequestLifecycle):
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

    RequestLifecycle.update_state_after_alloc(
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


def test_chunk_store_allocation_skips_committed_duplicate() -> None:
    """Chunk reuse should not enqueue stores for an already committed key."""

    class DuplicateSchedulerProbe(_AllocatingSchedulerProbe):
        def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
            self.alloc_calls.append((chunk_key, token_count, model_id))
            return {
                "start_slot": 5,
                "file_offset": 160,
                "pos_offset": 0,
                "skipped": True,
            }

    connector = DuplicateSchedulerProbe()
    tokens = [1] * 12
    key = hash_tokens(tokens)
    connector.seed_pending_store("req", key, 12, [10, 11, 12])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert connector.alloc_calls == [(key, 12, "m")]
    assert pending_alloc == {}
    assert pending_stores == {}


def test_prefix_mode_stores_computed_blocks_as_individual_slots():
    """Rolling prefix mode allocates one store spec for each computed slot."""

    class PrefixSchedulerProbe(_AllocatingSchedulerProbe):
        def __init__(self) -> None:
            super().__init__()
            self.use_prefix_reuse_strategy()

        def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
            slot = len(self.alloc_calls) + 20
            self.alloc_calls.append((chunk_key, token_count, model_id))
            return {
                "start_slot": slot,
                "file_offset": slot * self._slot_size,
                "pos_offset": 0,
            }

    tokens = list(range(12))
    keys = rolling_keys(tokens, BLOCK_TOKENS)
    connector = PrefixSchedulerProbe()
    connector.seed_pending_store("req", keys[-1], 12, [10, 11, 12])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert connector.alloc_calls == [("batch", len(keys), "m")]
    assert sorted(pending_stores) == ["req:store:0", "req:store:1", "req:store:2"]
    for slot_i, key in enumerate(keys):
        alloc = pending_stores[f"req:store:{slot_i}"]
        assert alloc["chunk_key"] == key
        assert alloc["token_count"] == BLOCK_TOKENS
        assert alloc["num_slots"] == 1
        assert alloc["block_ids"] == [10 + slot_i]


def test_prefix_mode_allocates_contiguous_slots_in_one_batch():
    """Rolling prefix mode batches server allocation for ready slot runs."""

    tokens = list(range(12))
    keys = rolling_keys(tokens, BLOCK_TOKENS)
    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    connector.seed_pending_store("req", keys[-1], 12, [10, 11, 12])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert connector.alloc_calls == [("batch", 3, "m")]
    assert sorted(pending_stores) == ["req:store:0", "req:store:1", "req:store:2"]
    assert [
        (
            pending_stores[f"req:store:{slot_i}"]["chunk_key"],
            pending_stores[f"req:store:{slot_i}"]["start_slot"],
            pending_stores[f"req:store:{slot_i}"]["file_offset"],
            pending_stores[f"req:store:{slot_i}"]["block_ids"],
        )
        for slot_i in range(3)
    ] == [
        (keys[0], 20, 20 * BLOCK_TOKENS * 8, [10]),
        (keys[1], 21, 21 * BLOCK_TOKENS * 8, [11]),
        (keys[2], 22, 22 * BLOCK_TOKENS * 8, [12]),
    ]


def test_prefix_mode_allocates_slots_incrementally_before_full_prompt_ready():
    """Rolling prefix stores publish each computed slot as block IDs arrive."""

    class PrefixSchedulerProbe(_AllocatingSchedulerProbe):
        def __init__(self) -> None:
            super().__init__()
            self.use_prefix_reuse_strategy()
            self._ipc_sync = _AllocChunkOnlyIPC(self)

        def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
            slot = len(self.alloc_calls) + 20
            self.alloc_calls.append((chunk_key, token_count, model_id))
            return {
                "start_slot": slot,
                "file_offset": slot * self._slot_size,
                "pos_offset": 0,
            }

    tokens = list(range(12))
    key0, key1, key2 = rolling_keys(tokens, BLOCK_TOKENS)
    connector = PrefixSchedulerProbe()
    connector.seed_pending_store("req", "", 12, [10])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert sorted(pending_alloc) == ["req"]
    assert sorted(pending_stores) == ["req:store:0"]
    assert pending_stores["req:store:0"]["chunk_key"] == key0
    assert pending_stores["req:store:0"]["block_ids"] == [10]

    pending_alloc["req"].block_ids.extend([11])
    connector.maybe_allocate_store_for_test("req")

    assert sorted(pending_alloc) == ["req"]
    assert sorted(pending_stores) == ["req:store:0", "req:store:1"]
    assert pending_stores["req:store:1"]["chunk_key"] == key1
    assert pending_stores["req:store:1"]["block_ids"] == [11]

    pending_alloc["req"].block_ids.extend([12])
    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert sorted(pending_stores) == ["req:store:0", "req:store:1", "req:store:2"]
    assert pending_stores["req:store:2"]["chunk_key"] == key2
    assert pending_stores["req:store:2"]["block_ids"] == [12]


def test_prefix_store_allocation_advances_rolling_key_incrementally(monkeypatch):
    """Prefix store allocation should not rebuild the whole key list repeatedly."""

    class PrefixSchedulerProbe(_AllocatingSchedulerProbe):
        def __init__(self) -> None:
            super().__init__()
            self.use_prefix_reuse_strategy()

    calls: list[tuple[list[int], int, str | None, int]] = []

    monkeypatch.setattr(
        "daser.connector.scheduler.reuse.rolling_prefix_keys",
        lambda tokens, block_tokens, initial_key=None, start_slot=0, **kwargs: (
            calls.append((list(tokens), block_tokens, initial_key, start_slot))
            or rolling_prefix_keys(
                tokens,
                block_tokens,
                initial_key=initial_key,
                start_slot=start_slot,
                **kwargs,
            )
        ),
    )

    tokens = list(range(12))
    key0 = rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])
    key1 = rolling_prefix_key(key0, tokens[BLOCK_TOKENS : BLOCK_TOKENS * 2])
    key2 = rolling_prefix_key(key1, tokens[BLOCK_TOKENS * 2 : BLOCK_TOKENS * 3])
    connector = PrefixSchedulerProbe()
    connector.seed_pending_store("req", key2, 12, [10, 11, 12])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    _, pending_stores = connector.pending_state
    assert calls == [(tokens, BLOCK_TOKENS, ROLLING_PREFIX_SEED, 0)]
    assert connector.alloc_calls == [("batch", 3, "m")]
    assert pending_stores["req:store:2"]["chunk_key"] == key2


def test_request_finished_keeps_request_until_store_finishes() -> None:
    """Scheduler should hold finished requests while worker stores KV."""
    connector = _AllocatingSchedulerProbe()
    connector.seed_tokens("req", [1] * 8)
    connector.seed_pending_store_spec(
        "req",
        ReqStoreSpec(
            chunk_key="live-store",
            start_slot=0,
            num_slots=2,
            block_ids=[10, 11],
            file_offset=0,
            token_count=8,
        ),
    )

    class Output:
        num_scheduled_tokens = {"req": 8}

    meta = connector.build_connector_meta(Output())

    assert list(meta.reqs_to_store) == ["req"]
    assert connector.request_finished_for_test("req") == (True, None)
    assert connector.has_req_tokens("req")

    connector.update_connector_output_for_test({"req"})

    assert not connector.has_req_tokens("req")
    assert connector.pending_state == ({}, {})


def test_request_finished_keeps_prefix_store_request_until_store_finishes() -> None:
    """Synthetic prefix store IDs should hold their base request lifecycle."""
    connector = _AllocatingSchedulerProbe()
    connector.seed_tokens("req", [1] * 8)
    connector.seed_pending_store_spec(
        "req:store:0",
        ReqStoreSpec(
            chunk_key="live-store-0",
            start_slot=0,
            num_slots=1,
            block_ids=[10],
            file_offset=0,
            token_count=4,
        ),
    )
    connector.seed_pending_store_spec(
        "req:store:1",
        ReqStoreSpec(
            chunk_key="live-store-1",
            start_slot=1,
            num_slots=1,
            block_ids=[11],
            file_offset=32,
            token_count=4,
        ),
    )

    class Output:
        num_scheduled_tokens = {"req": 8}

    meta = connector.build_connector_meta(Output())

    assert sorted(meta.reqs_to_store) == ["req:store:0", "req:store:1"]
    assert connector.request_finished_for_test("req") == (True, None)

    connector.update_connector_output_for_test({"req"})

    assert not connector.has_req_tokens("req")
    assert connector.pending_state == ({}, {})


def test_prefix_store_allocation_skips_committed_duplicate_slot() -> None:
    """Rolling-prefix stores should not enqueue writes for committed slots."""

    class DuplicatePrefixSchedulerProbe(_AllocatingSchedulerProbe):
        def __init__(self) -> None:
            super().__init__()
            self.use_prefix_reuse_strategy()
            self._ipc_sync = _AllocChunkOnlyIPC(self)

        def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
            slot = len(self.alloc_calls)
            self.alloc_calls.append((chunk_key, token_count, model_id))
            return {
                "start_slot": 20 + slot,
                "file_offset": (20 + slot) * self._slot_size,
                "pos_offset": 0,
                "skipped": slot == 0,
            }

    tokens = list(range(8))
    keys = rolling_keys(tokens, BLOCK_TOKENS)
    connector = DuplicatePrefixSchedulerProbe()
    connector.seed_pending_store("req", keys[-1], 8, [10, 11])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert connector.alloc_calls == [(key, BLOCK_TOKENS, "m") for key in keys]
    assert sorted(pending_stores) == ["req:store:1"]
    assert pending_stores["req:store:1"]["chunk_key"] == keys[1]


def test_prefix_store_batch_allocation_skips_committed_duplicate_slot() -> None:
    """Batched rolling-prefix allocation should skip duplicate slot stores."""

    class DuplicateBatchPrefixSchedulerProbe(_AllocatingSchedulerProbe):
        def __init__(self) -> None:
            super().__init__()
            self.use_prefix_reuse_strategy()

        def alloc_chunks(self, chunks: list[dict], model_id: str) -> list[dict]:
            self.alloc_calls.append(("batch", len(chunks), model_id))
            return [
                {
                    "chunk_key": str(chunk["chunk_key"]),
                    "start_slot": 20 + idx,
                    "file_offset": (20 + idx) * self._slot_size,
                    "pos_offset": 0,
                    "skipped": idx == 0,
                }
                for idx, chunk in enumerate(chunks)
            ]

    tokens = list(range(8))
    keys = rolling_keys(tokens, BLOCK_TOKENS)
    connector = DuplicateBatchPrefixSchedulerProbe()
    connector.seed_pending_store("req", keys[-1], 8, [10, 11])
    connector.seed_tokens("req", tokens)

    connector.maybe_allocate_store_for_test("req")

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert connector.alloc_calls == [("batch", len(keys), "m")]
    assert sorted(pending_stores) == ["req:store:1"]
    assert pending_stores["req:store:1"]["chunk_key"] == keys[1]


def test_prefix_mode_builds_one_store_spec_per_slot():
    """Connector metadata keeps rolling-prefix store work slot granular."""

    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    specs = {
        "req:store:0": ReqStoreSpec("live-a", 20, 1, [10], 640, 4),
        "req:store:1": ReqStoreSpec("live-b", 21, 1, [11], 672, 4),
    }
    for req_id, spec in specs.items():
        connector.seed_pending_store_spec(req_id, spec)

    class Output:
        num_scheduled_tokens = {"req": 8}
        scheduled_cached_reqs = None

    meta = connector.build_connector_meta(Output())

    assert meta.reqs_to_store == specs


def test_prefix_mode_merges_adjacent_load_specs_for_one_request() -> None:
    """Rolling-prefix load metadata should coalesce adjacent slot hits."""

    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    connector._pending_loads = {  # noqa: SLF001
        "req": {
            "0": {
                "chunk_key": "slot-a",
                "start_slot": 20,
                "num_slots": 1,
                "block_ids": [10],
                "file_offset": 640,
                "token_count": BLOCK_TOKENS,
                "target_token_start": 0,
                "pos_offset": 0,
            },
            "1": {
                "chunk_key": "slot-b",
                "start_slot": 21,
                "num_slots": 1,
                "block_ids": [11],
                "file_offset": 672,
                "token_count": BLOCK_TOKENS,
                "target_token_start": BLOCK_TOKENS,
                "pos_offset": 0,
            },
            "2": {
                "chunk_key": "slot-c",
                "start_slot": 22,
                "num_slots": 1,
                "block_ids": [12],
                "file_offset": 704,
                "token_count": BLOCK_TOKENS,
                "target_token_start": 2 * BLOCK_TOKENS,
                "pos_offset": 0,
            },
        }
    }

    class Output:
        num_scheduled_tokens = {"req": 12}
        scheduled_cached_reqs = None
        scheduled_new_reqs = []

    meta = connector.build_connector_meta(Output())

    assert list(meta.reqs_to_load) == ["req"]
    assert meta.reqs_to_load["req"] == ReqLoadSpec(
        chunk_key="slot-a",
        start_slot=20,
        num_slots=3,
        block_ids=[10, 11, 12],
        file_offset=640,
        token_count=12,
        target_token_start=0,
        pos_offset=0,
    )


def test_prefix_mode_uses_base_request_load_ids_for_split_load_specs() -> None:
    """Split load specs should still finish under the original request ID."""
    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    connector._pending_loads = {  # noqa: SLF001
        "req": {
            "0": {
                "chunk_key": "slot-a",
                "start_slot": 20,
                "num_slots": 1,
                "block_ids": [10],
                "file_offset": 640,
                "token_count": BLOCK_TOKENS,
                "target_token_start": 0,
                "pos_offset": 0,
            },
            "1": {
                "chunk_key": "slot-c",
                "start_slot": 22,
                "num_slots": 1,
                "block_ids": [12],
                "file_offset": 704,
                "token_count": BLOCK_TOKENS,
                "target_token_start": 2 * BLOCK_TOKENS,
                "pos_offset": 0,
            },
        }
    }

    class Output:
        num_scheduled_tokens = {"req": 12}
        scheduled_cached_reqs = None
        scheduled_new_reqs = []

    meta = connector.build_connector_meta(Output())

    assert sorted(meta.reqs_to_load) == ["req:load:0", "req:load:1"]


def test_build_connector_meta_includes_waiting_async_loads() -> None:
    """Async load specs must be sent after vLLM moves requests to waiting."""
    connector = _AllocatingSchedulerProbe()
    connector._pending_loads = {  # noqa: SLF001
        "req": {
            "chunk_key": "hit",
            "start_slot": 20,
            "num_slots": 1,
            "block_ids": [10],
            "file_offset": 640,
            "token_count": BLOCK_TOKENS,
            "target_token_start": 0,
            "pos_offset": 0,
        }
    }

    class Output:
        num_scheduled_tokens = {}
        scheduled_cached_reqs = None
        scheduled_new_reqs = []

    meta = connector.build_connector_meta(Output())

    assert list(meta.reqs_to_load) == ["req"]
    assert meta.reqs_to_load["req"].block_ids == [10]
    assert connector._pending_loads == {}  # noqa: SLF001


def test_prefix_mode_defers_uncomputed_slot_store_specs():
    """Rolling-prefix slot stores are published only after the slot is computed."""

    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    specs = {
        "req:store:0": ReqStoreSpec("live-a", 20, 1, [10], 640, 4),
        "req:store:1": ReqStoreSpec("live-b", 21, 1, [11], 672, 4),
        "req:store:2": ReqStoreSpec("live-c", 22, 1, [12], 704, 4),
    }
    for req_id, spec in specs.items():
        connector.seed_pending_store_spec(req_id, spec)

    class Cached:
        req_ids = ["req"]
        new_block_ids = []
        resumed_req_ids = set()
        num_computed_tokens = [0]

    class Output:
        num_scheduled_tokens = {"req": 4}
        scheduled_cached_reqs = Cached()
        scheduled_new_reqs = []

    meta = connector.build_connector_meta(Output())

    assert meta.reqs_to_store == {"req:store:0": specs["req:store:0"]}
    _, pending_stores = connector.pending_state
    assert "req:store:1" in pending_stores
    assert "req:store:2" in pending_stores


def test_build_connector_meta_drops_preempted_pending_store_state():
    """Preempted requests free KV blocks, so pending store block IDs are stale."""

    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    connector.seed_pending_store(
        "req",
        "unused",
        8,
        [10],
    )
    connector.seed_pending_store_spec(
        "req",
        ReqStoreSpec("live-whole", 20, 1, [10], 640, 4),
    )
    connector.seed_pending_store_spec(
        "req:store:0",
        ReqStoreSpec("live-slot", 21, 1, [10], 672, 4),
    )

    class Output:
        num_scheduled_tokens = {"req": 4}
        scheduled_cached_reqs = None
        scheduled_new_reqs = []
        preempted_req_ids = {"req"}

    meta = connector.build_connector_meta(Output())

    assert meta.reqs_to_store == {}
    assert connector.pending_state == ({}, {})


def test_build_connector_meta_releases_preempted_pending_store_writer():
    """Dropping an uncommitted store should release its server write claim."""
    connector = _AllocatingSchedulerProbe()
    connector.seed_pending_store_spec(
        "req",
        ReqStoreSpec("k0", 10, 2, [4, 5], 320, 8),
    )

    class Output:
        num_scheduled_tokens = {}
        scheduled_cached_reqs = None
        scheduled_new_reqs = []
        preempted_req_ids = {"req"}

    connector.build_connector_meta(Output())

    assert connector.released_allocations == [("k0", 10, 2)]


def test_prefix_mode_hit_tracks_store_from_first_missing_slot():
    """Warm prefix hits track suffix stores even when GPU coverage is longer."""

    class MockIPCClient:
        def lookup(self, tokens, model_id):
            del tokens, model_id
            return [
                {
                    "chunk_key": "hit-0",
                    "start_slot": 100,
                    "num_slots": 1,
                    "file_offset": 3200,
                    "token_count": 4,
                    "target_token_start": 0,
                    "pos_offset": 0,
                }
            ]

    class MockConnector(RequestLifecycle):
        def __init__(self) -> None:
            self._runtime_config_ready = True
            self._block_tokens = BLOCK_TOKENS
            self._cache_reuse_strategy = PrefixReuseStrategy(self._block_tokens)
            self._slot_size = 32
            self._model_id = "m"
            self._ipc_sync = MockIPCClient()
            self._pending_loads = {}
            self._pending_stores = {}
            self._pending_alloc = {}
            self._req_tokens = {}

        @property
        def pending_alloc(self) -> dict:
            return self._pending_alloc

    class MockRequest:
        request_id = "req"
        prompt_token_ids = list(range(12))
        kv_transfer_params = {}

    connector = MockConnector()
    assert connector.get_num_new_matched_tokens(MockRequest(), 0) == (4, True)
    pending = connector.pending_alloc["req"]
    assert pending.chunk_key == ""
    assert pending.token_count == 12
    assert pending.start_slot_index == 1
    assert pending.rolling_key == "hit-0"
    assert pending.rolling_slot_index == 1

    gpu_hit_connector = MockConnector()
    assert gpu_hit_connector.get_num_new_matched_tokens(MockRequest(), 8) == (0, False)
    assert gpu_hit_connector.pending_alloc["req"].start_slot_index == 1


def test_prefix_mode_hit_still_allocates_missing_slot_stores_after_load():
    """Prefix-hit requests still store newly computed suffix slots."""

    class PrefixSchedulerProbe(_AllocatingSchedulerProbe):
        def __init__(self) -> None:
            super().__init__()
            self.use_prefix_reuse_strategy()
            self._pending_loads = {
                "req": {
                    "chunk_key": "hit-0",
                    "start_slot": 100,
                    "num_slots": 1,
                    "file_offset": 3200,
                    "token_count": BLOCK_TOKENS,
                    "target_token_start": 0,
                    "num_computed_tokens": 0,
                }
            }

        def alloc_chunk(self, chunk_key: str, token_count: int, model_id: str) -> dict:
            slot = len(self.alloc_calls) + 20
            self.alloc_calls.append((chunk_key, token_count, model_id))
            return {
                "start_slot": slot,
                "file_offset": slot * self._slot_size,
                "pos_offset": 0,
            }

    tokens = list(range(12))
    key0, key1, key2 = rolling_keys(tokens, BLOCK_TOKENS)
    connector = PrefixSchedulerProbe()
    connector.seed_tokens("req", tokens)
    connector._pending_alloc["req"] = PendingStore(  # noqa: SLF001
        chunk_key="",
        token_count=12,
        start_slot_index=1,
        rolling_key=key0,
        rolling_slot_index=1,
    )

    class MockRequest:
        request_id = "req"

    class MockBlock:
        def __init__(self, block_id: int) -> None:
            self.block_id = block_id

    class MockBlocks:
        blocks = ([MockBlock(10), MockBlock(11), MockBlock(12)],)

    connector.update_state_after_alloc(
        MockRequest(),
        MockBlocks(),
        num_external_tokens=BLOCK_TOKENS,
    )

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert sorted(pending_stores) == ["req:store:1", "req:store:2"]
    assert connector.alloc_calls == [("batch", 2, "m")]
    assert pending_stores["req:store:1"]["chunk_key"] == key1
    assert pending_stores["req:store:2"]["chunk_key"] == key2
    assert pending_stores["req:store:1"]["block_ids"] == [11]
    assert pending_stores["req:store:2"]["block_ids"] == [12]


def test_finished_recving_keeps_pending_suffix_stores() -> None:
    """Async load completion should not drop pending stores for new suffix KV."""
    connector = _AllocatingSchedulerProbe()
    connector.seed_tokens("req", list(range(8)))
    connector.seed_pending_store_spec(
        "req:store:1",
        ReqStoreSpec(
            chunk_key="suffix",
            start_slot=1,
            num_slots=1,
            block_ids=[11],
            file_offset=32,
            token_count=8,
        ),
    )
    connector._pending_loads["req:load:0"] = {  # noqa: SLF001
        "chunk_key": "hit",
        "start_slot": 0,
        "num_slots": 1,
        "file_offset": 0,
        "token_count": 4,
    }

    connector.update_connector_output_for_test(finished_recving={"req"})

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert sorted(pending_stores) == ["req:store:1"]
    assert connector._pending_loads == {}  # noqa: SLF001
    assert not connector.has_req_tokens("req")


def test_finished_recving_keeps_tokens_until_suffix_allocation() -> None:
    """Async prefix loads retain tokens needed to key later suffix blocks."""
    tokens = list(range(12))
    key0, key1, key2 = rolling_keys(tokens, BLOCK_TOKENS)
    connector = _AllocatingSchedulerProbe()
    connector.use_prefix_reuse_strategy()
    connector.seed_tokens("req", tokens)
    connector._pending_alloc["req"] = PendingStore(  # noqa: SLF001
        chunk_key="",
        token_count=12,
        start_slot_index=1,
        rolling_key=key0,
        rolling_slot_index=1,
        block_ids=[10],
    )

    connector.update_connector_output_for_test(finished_recving={"req"})
    assert connector.has_req_tokens("req")

    class Cached:
        req_ids = ["req"]
        new_block_ids = [([11, 12],)]
        resumed_req_ids: set[str] = set()

    class Output:
        scheduled_cached_reqs = Cached()

    connector.record_cached_blocks(Output())

    pending_alloc, pending_stores = connector.pending_state
    assert pending_alloc == {}
    assert connector.alloc_calls == [("batch", 2, "m")]
    assert pending_stores["req:store:1"]["chunk_key"] == key1
    assert pending_stores["req:store:2"]["chunk_key"] == key2


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


def test_stale_filtered_store_does_not_hold_finished_request() -> None:
    """Scheduler should only wait for stores actually sent to workers."""
    connector = _AllocatingSchedulerProbe()
    connector.seed_tokens("req", [1] * 4)
    connector.seed_pending_store_spec(
        "req",
        ReqStoreSpec("stale-key", 0, 1, [10], 0, 4),
    )

    class Output:
        num_scheduled_tokens = {"req": 4}
        scheduled_cached_reqs = None

    meta = connector.build_connector_meta(Output())

    assert meta.reqs_to_store == {}
    assert connector.request_finished_for_test("req") == (False, None)


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


def test_copy_staging_to_kv_cache_batches_rope_transform(monkeypatch):
    """Load-time RoPE relocation is applied once over all loaded layers."""
    block_ids = [1, 2, 3]
    layer_names = ["layer.0", "layer.1"]
    kv_caches = {
        name: torch.zeros((2, 8, 4, 2, 8), dtype=torch.float32) for name in layer_names
    }
    layer_shape = kv_caches[layer_names[0]][:, block_ids[0]].shape
    layer_size = kv_caches["layer.0"][:, block_ids[0]].nbytes
    slot_size = layer_size * len(layer_names)
    staging = torch.empty(len(block_ids) * slot_size, dtype=torch.uint8)

    for slot_i in range(len(block_ids)):
        for layer_idx in range(len(layer_names)):
            layer = torch.full(
                layer_shape,
                float((slot_i + 1) * 100 + layer_idx),
                dtype=torch.float32,
            )
            start = slot_i * slot_size + layer_idx * layer_size
            staging[start : start + layer_size].copy_(
                layer.reshape(-1).view(torch.uint8)
            )

    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_rope(
        kv_block: torch.Tensor,
        delta: int,
        rope_base: float,
        rotary_dim: int,
        is_neox_style: bool,
    ) -> None:
        calls.append((tuple(kv_block.shape), delta))
        kv_block[:, :, 0, ..., :rotary_dim].add_(10.0)

    monkeypatch.setattr(
        "daser.connector.worker.staging.apply_rope_delta_to_kv_key_block",
        fake_rope,
    )

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches=kv_caches,
        layer_names=layer_names,
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=5,
        rope_rotary_dim=8,
    )

    assert copies == len(layer_names)
    assert calls == [((len(block_ids), len(layer_names), 2, 4, 2, 8), 5)]
    for slot_i, block_id in enumerate(block_ids):
        for layer_idx, layer_name in enumerate(layer_names):
            key = kv_caches[layer_name][0, block_id]
            value = kv_caches[layer_name][1, block_id]
            base = float((slot_i + 1) * 100 + layer_idx)
            assert torch.equal(key[..., :8], torch.full_like(key[..., :8], base + 10))
            assert torch.equal(value, torch.full_like(value, base))


def test_copy_staging_to_kv_cache_skips_rope_for_zero_offset(monkeypatch):
    """Position-zero chunks should copy without dispatching RoPE relocation."""
    block_ids = [1, 2]
    layer_names = ["layer.0", "layer.1"]
    kv_caches = {
        name: torch.zeros((2, 8, 4, 2, 8), dtype=torch.float32) for name in layer_names
    }
    layer_shape = kv_caches[layer_names[0]][:, block_ids[0]].shape
    layer_size = kv_caches[layer_names[0]][:, block_ids[0]].nbytes
    slot_size = layer_size * len(layer_names)
    staging = torch.empty(len(block_ids) * slot_size, dtype=torch.uint8)
    for slot_i in range(len(block_ids)):
        for layer_idx in range(len(layer_names)):
            layer = torch.full(
                layer_shape,
                float((slot_i + 1) * 100 + layer_idx),
                dtype=torch.float32,
            )
            start = slot_i * slot_size + layer_idx * layer_size
            staging[start : start + layer_size].copy_(
                layer.reshape(-1).view(torch.uint8)
            )

    def fail_rope(*args, **kwargs):
        raise AssertionError("RoPE should not run for pos_offset=0")

    monkeypatch.setattr(
        "daser.connector.worker.staging.apply_rope_delta_to_key_block",
        fail_rope,
    )

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches=kv_caches,
        layer_names=layer_names,
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=0,
        rope_rotary_dim=8,
    )

    assert copies == len(layer_names)


def test_copy_staging_to_cross_layer_kv_cache_uses_single_bulk_copy(monkeypatch):
    """Cross-layer vLLM KV layout should load with one bulk staging copy."""
    block_ids = [1, 2, 3]
    num_layers = 2
    cross_kv = torch.zeros((8, num_layers, 2, 4, 2, 8), dtype=torch.float32)
    layer_shape = cross_kv[block_ids[0], 0].shape
    layer_size = cross_kv[block_ids[0], 0].nbytes
    slot_size = layer_size * num_layers
    staging = torch.empty(len(block_ids) * slot_size, dtype=torch.uint8)
    for slot_i in range(len(block_ids)):
        for layer_idx in range(num_layers):
            layer = torch.full(
                layer_shape,
                float((slot_i + 1) * 100 + layer_idx),
                dtype=torch.float32,
            )
            start = slot_i * slot_size + layer_idx * layer_size
            staging[start : start + layer_size].copy_(
                layer.reshape(-1).view(torch.uint8)
            )

    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_rope(kv_block, delta, rope_base, rotary_dim, is_neox_style):
        calls.append((tuple(kv_block.shape), delta))
        kv_block[:, :, 0, ..., :rotary_dim].add_(10)

    monkeypatch.setattr(
        "daser.connector.worker.staging._apply_rope_delta_with_tables",
        fake_rope,
    )

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches={"__cross_layers__": cross_kv},
        layer_names=["layer.0", "layer.1"],
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=7,
        rope_rotary_dim=8,
    )

    assert copies == 1
    assert calls == [((len(block_ids), num_layers, 2, 4, 2, 8), 7)]
    for slot_i, block_id in enumerate(block_ids):
        for layer_idx in range(num_layers):
            key = cross_kv[block_id, layer_idx, 0]
            value = cross_kv[block_id, layer_idx, 1]
            base = float((slot_i + 1) * 100 + layer_idx)
            assert torch.equal(key[..., :8], torch.full_like(key[..., :8], base + 10))
            assert torch.equal(value, torch.full_like(value, base))


def test_copy_staging_to_cross_layer_kv_cache_rotates_target_for_small_runs(
    monkeypatch,
):
    """Small cross-layer loads should copy first and rotate destination K."""
    block_ids = list(range(16))
    num_layers = 2
    cross_kv = torch.zeros((20, num_layers, 2, 4, 2, 8), dtype=torch.float32)
    layer_size = cross_kv[block_ids[0], 0].nbytes
    slot_size = layer_size * num_layers
    staging = torch.ones(len(block_ids) * slot_size, dtype=torch.uint8)
    calls: list[tuple[int, tuple[int, ...], int]] = []

    def fake_rope(
        kv_block: torch.Tensor,
        delta: int,
        rope_base: float,
        rotary_dim: int,
        is_neox_style: bool,
    ) -> None:
        calls.append((int(kv_block.data_ptr()), tuple(kv_block.shape), delta))
        kv_block[:, :, 0, ..., :rotary_dim].add_(3.0)

    monkeypatch.setattr(
        "daser.connector.worker.staging._apply_rope_delta_with_tables",
        fake_rope,
    )

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches={"__cross_layers__": cross_kv},
        layer_names=["layer.0", "layer.1"],
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=11,
        rope_rotary_dim=8,
    )

    dst_view = cross_kv[block_ids[0] : block_ids[-1] + 1]
    assert copies == 1
    assert calls == [
        (
            int(dst_view.data_ptr()),
            (len(block_ids), num_layers, 2, 4, 2, 8),
            11,
        )
    ]


def test_copy_staging_to_cross_layer_kv_cache_prefers_table_rope_for_small_runs(
    monkeypatch,
):
    """Small cross-layer loads should use the table RoPE backend when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TileLang table RoPE path")
    device = torch.device("cuda")
    block_ids = list(range(16))
    num_layers = 2
    cross_kv = torch.zeros(
        (20, num_layers, 2, 4, 2, 8),
        dtype=torch.float32,
        device=device,
    )
    layer_size = cross_kv[block_ids[0], 0].nbytes
    slot_size = layer_size * num_layers
    staging = torch.ones(len(block_ids) * slot_size, dtype=torch.uint8, device=device)
    table_calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_table_rope(
        kv_block: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        rotary_dim: int,
        is_neox_style: bool,
    ) -> None:
        table_calls.append(
            (tuple(kv_block.shape), tuple(cos_table.shape), tuple(sin_table.shape))
        )
        kv_block[:, :, 0, ..., :rotary_dim].add_(5.0)

    def fail_rope(*args, **kwargs):
        raise AssertionError("legacy RoPE path should not run")

    monkeypatch.setattr(
        "daser.connector.worker.staging.apply_rope_delta_to_kv_key_block_table",
        fake_table_rope,
    )
    monkeypatch.setattr(
        "daser.connector.worker.staging._apply_rope_delta_to_kv_key_block",
        fail_rope,
    )
    monkeypatch.setattr("daser.connector.worker.staging._rope_table_cache", {})

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches={"__cross_layers__": cross_kv},
        layer_names=["layer.0", "layer.1"],
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=11,
        rope_rotary_dim=8,
    )

    assert copies == 1
    assert table_calls == [((len(block_ids), num_layers, 2, 4, 2, 8), (4,), (4,))]


def test_copy_staging_to_cross_layer_kv_cache_prefers_fused_table_restore(
    monkeypatch,
):
    """Large cross-layer loads should use fused restore with cached RoPE tables."""
    block_ids = list(range(32))
    num_layers = 2
    cross_kv = torch.zeros((40, num_layers, 2, 4, 2, 8), dtype=torch.float32)
    layer_size = cross_kv[block_ids[0], 0].nbytes
    slot_size = layer_size * num_layers
    staging = torch.ones(len(block_ids) * slot_size, dtype=torch.uint8)
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def fake_table_restore(
        src_kv: torch.Tensor,
        dst_kv: torch.Tensor,
        delta: int,
        rope_base: float,
        rotary_dim: int,
        is_neox_style: bool,
    ) -> bool:
        calls.append((tuple(src_kv.shape), tuple(dst_kv.shape), delta))
        dst_kv.copy_(src_kv)
        return True

    monkeypatch.setattr(
        "daser.connector.worker.staging._restore_cross_layer_with_tables",
        fake_table_restore,
    )

    copies = _copy_staging_to_kv_cache(
        staging=staging,
        kv_caches={"__cross_layers__": cross_kv},
        layer_names=["layer.0", "layer.1"],
        block_ids=block_ids,
        slot_size=slot_size,
        pos_offset=11,
        rope_rotary_dim=8,
    )

    assert copies == 1
    assert calls == [
        (
            (len(block_ids), num_layers, 2, 4, 2, 8),
            (len(block_ids), num_layers, 2, 4, 2, 8),
            11,
        )
    ]


def test_copy_staging_to_cross_layer_kv_cache_raises_failed_fused_restore(
    monkeypatch,
):
    """A failing TileLang fused restore should surface instead of falling back."""
    block_ids = list(range(32))
    num_layers = 2
    cross_kv = torch.zeros((40, num_layers, 2, 4, 2, 8), dtype=torch.float32)
    layer_size = cross_kv[block_ids[0], 0].nbytes
    slot_size = layer_size * num_layers
    staging = torch.ones(len(block_ids) * slot_size, dtype=torch.uint8)

    def failing_restore(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(
        "daser.connector.worker.staging._restore_cross_layer_with_tables",
        failing_restore,
    )

    with pytest.raises(RuntimeError, match="backend unavailable"):
        _copy_staging_to_kv_cache(
            staging=staging,
            kv_caches={"__cross_layers__": cross_kv},
            layer_names=["layer.0", "layer.1"],
            block_ids=block_ids,
            slot_size=slot_size,
            pos_offset=11,
            rope_rotary_dim=8,
        )


def test_build_staging_store_batches_caps_gpu_staging_bytes():
    """Store batches preserve chunk metadata while bounding staging size."""
    reqs_to_store = {
        "r0": ReqStoreSpec("k0", 10, 3, [5, 1, 7], 320, 12),
        "r1": ReqStoreSpec("k1", 20, 1, [8], 640, 4),
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


def test_build_staging_store_batches_uses_spec_file_offset():
    """Store spans honor server-provided file offsets instead of recomputing."""
    reqs_to_store = {
        "r0": ReqStoreSpec("k0", 10, 2, [5, 6], 2048, 8),
    }

    batches = _build_staging_store_batches(
        reqs_to_store=reqs_to_store,
        slot_size=32,
        max_batch_bytes=32,
    )

    assert [[span.file_offset for span in spans] for _block_ids, spans in batches] == [
        [2048],
        [2080],
    ]


@pytest.mark.asyncio
async def test_store_cuda_export_selects_staged_buffer_device(monkeypatch) -> None:
    """Background CUDA IPC export must select the TP rank's staged device."""
    from daser.connector.worker import store as store_module

    selected_devices: list[torch.device] = []
    transferred: list[dict] = []

    class Client:
        async def transfer_store_cuda(self, **kwargs):
            transferred.append(kwargs)
            return []

    pipeline = StorePipeline.__new__(StorePipeline)
    pipeline._client = Client()  # noqa: SLF001
    pipeline._tp_rank = 0  # noqa: SLF001
    pipeline._tp_size = 1  # noqa: SLF001
    buffer = SimpleNamespace(device=torch.device("cuda:1"), nbytes=32)
    staged = StagedStoreBatch(buffer=buffer, spans=[], lease=object())
    cupy_buffer = object()

    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    monkeypatch.setattr(store_module.cupy, "asarray", lambda tensor: cupy_buffer)
    monkeypatch.setattr(store_module, "cuda_array_pointer", lambda array: 4096)
    monkeypatch.setattr(
        store_module,
        "cuda_allocation_base_and_offset",
        lambda pointer: (pointer, 0),
    )
    monkeypatch.setattr(store_module, "export_cuda_ipc_handle", lambda array: b"ipc")
    monkeypatch.setattr(store_module, "cuda_array_device_id", lambda array: 1)

    await pipeline._write_cuda_buffer(staged)  # noqa: SLF001

    assert selected_devices == [torch.device("cuda:1")]
    assert transferred[0]["device_id"] == 1


def test_tensor_parallel_rank_lanes_are_contiguous_and_disjoint() -> None:
    """Each TP rank maps a logical slot run into its own contiguous lane."""
    local_slot_size = 32
    rank_stride = 10 * local_slot_size
    start_slot = 3

    rank_0 = start_slot * local_slot_size
    rank_1 = rank_stride + start_slot * local_slot_size

    assert rank_0 == 3 * local_slot_size
    assert rank_1 == rank_stride + 3 * local_slot_size
    assert rank_0 + 2 * local_slot_size <= rank_1


def test_derive_staging_layout_scales_with_vram(monkeypatch):
    """One budget preserves balanced pools and the explicit 6 GiB ceiling."""

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
    small_batch, small_load, small_store, small_total = derive_staging_layout(
        torch.device("cuda"), 64 << 20, 8, 1 << 30
    )
    assert small_batch == max(
        MIN_STORE_STAGING_BYTES,
        min((24 << 30) // 50, (12 << 30) // 10),
    )
    assert (small_load, small_store) == (2, 2)
    assert small_total == small_batch * 4

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
    large_batch, large_load, large_store, large_total = derive_staging_layout(
        torch.device("cuda"), 64 << 20, 8, 1 << 30
    )
    assert large_batch == DEFAULT_STORE_STAGING_BYTES
    assert (large_load, large_store) == (2, 2)
    assert large_total == DEFAULT_STAGING_BUDGET_BYTES

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (8 << 30, 80 << 30),
    )
    tight_batch, tight_load, tight_store, tight_total = derive_staging_layout(
        torch.device("cuda"), 64 << 20, 8, 1 << 30
    )
    assert tight_batch == (8 << 30) // 10
    assert (tight_load, tight_store) == (5, 2)
    assert tight_total == tight_batch * 7


def test_store_cuda_staging_pool_reuses_preallocated_buffer():
    """Store staging pool reuses its init-time allocation after release."""
    pool = FixedCudaStagingPool(
        device=torch.device("cpu"),
        buffer_bytes=128,
        depth=1,
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
        "r0": ReqLoadSpec("k0", 10, 3, [4, 5, 6], 320, 12),
        "r1": ReqLoadSpec("k1", 20, 1, [7], 640, 4),
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


def test_build_load_read_batches_uses_spec_file_offset():
    """Load spans honor server-provided file offsets instead of recomputing."""
    reqs_to_load = {
        "r0": ReqLoadSpec("k0", 10, 2, [4, 5], 4096, 8),
    }

    batches = _build_load_read_batches(
        reqs_to_load=reqs_to_load,
        slot_size=32,
        max_batch_bytes=32,
    )

    assert [[span["file_offset"] for span in spans] for _, spans, _ in batches] == [
        [4096],
        [4128],
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
        "r0": ReqLoadSpec("k0", 10, 2, [4, 5], 320, 8),
        "r1": ReqLoadSpec("k1", 20, 1, [6], 640, 4),
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


def test_build_load_read_plan_deduplicates_identical_source_reads():
    """Multiple requests for the same source chunk should share one read span."""
    reqs_to_load = {
        "r0": ReqLoadSpec("k0", 10, 2, [4, 5], 320, 8, pos_offset=0),
        "r1": ReqLoadSpec("k0", 10, 2, [8, 9], 320, 8, pos_offset=0),
    }

    total_bytes, spans, per_req = _build_load_read_plan(reqs_to_load, slot_size=32)

    assert total_bytes == 64
    assert spans == [{"target_offset": 0, "nbytes": 64, "file_offset": 320}]
    assert [(start, end, spec.block_ids) for start, end, spec in per_req] == [
        (0, 64, [4, 5]),
        (0, 64, [8, 9]),
    ]


def test_build_staging_store_batches_deduplicates_identical_chunk_writes():
    """Multiple store specs for one allocation should produce one write span."""
    reqs_to_store = {
        "r0": ReqStoreSpec("k0", 10, 2, [4, 5], 320, 8),
        "r1": ReqStoreSpec("k0", 10, 2, [8, 9], 320, 8),
    }

    batches = _build_staging_store_batches(
        reqs_to_store=reqs_to_store,
        slot_size=32,
        max_batch_bytes=128,
    )

    assert len(batches) == 1
    block_ids, spans = batches[0]
    assert block_ids == [4, 5]
    assert spans == [StoreWriteSpan(0, 64, 320, "k0", 10, 2)]


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
