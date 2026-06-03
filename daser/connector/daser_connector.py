# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import threading
from typing import TYPE_CHECKING, Any

# Third Party
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

# First Party
from daser.connector.helpers import PendingStore
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.reuse import CacheReuseStrategy, build_cache_reuse_strategy
from daser.connector.scheduler import (
    SchedulerConnectorMixin,
    _block_ids_for_chunk,
    _contiguous_prefix_tokens,
    _trim_chunk_to_external_window,
)
from daser.connector.staging import (
    DEFAULT_ROPE_DELTA_SCALE,
)
from daser.connector.staging import (
    apply_rope_delta_to_key_block as _apply_rope_delta_to_key_block,
)
from daser.connector.staging import (
    build_load_read_plan as _build_load_read_plan,
)
from daser.connector.staging import (
    copy_staging_to_kv_cache as _copy_staging_to_kv_cache,
)
from daser.connector.worker import WorkerConnectorMixin
from daser.logging import init_logger

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_ROPE_DELTA_SCALE",
    "DaserConnector",
    "DaserConnectorMeta",
    "ReqLoadSpec",
    "ReqStoreSpec",
    "_apply_rope_delta_to_key_block",
    "_build_load_read_plan",
    "_block_ids_for_chunk",
    "_contiguous_prefix_tokens",
    "_copy_staging_to_kv_cache",
    "_trim_chunk_to_external_window",
]


class DaserConnector(
    SchedulerConnectorMixin,
    WorkerConnectorMixin,
    KVConnectorBase_V1,
):
    """vLLM KVConnectorBase_V1 implementation backed by DaseR.

    The entrypoint remains in this module for vLLM's
    ``kv_connector_module_path``. Scheduler-role behavior lives in
    ``daser.connector.scheduler`` and worker-role behavior lives in
    ``daser.connector.worker``.

    Args:
        vllm_config: full VllmConfig from vLLM.
        role: KVConnectorRole.SCHEDULER or KVConnectorRole.WORKER.
        kv_cache_config: optional KV cache configuration (unused).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        extra: dict[str, Any] = {}
        if (
            hasattr(vllm_config, "kv_transfer_config")
            and vllm_config.kv_transfer_config is not None
        ):
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}

        self._socket_path: str = extra.get("socket_path", "/tmp/daser.sock")
        self._store_path: str = ""
        self._slot_size: int = 0
        self._block_tokens: int = 16
        self._model_id: str = "default"
        self._cache_reuse_strategy: CacheReuseStrategy
        self._set_cache_reuse_strategy(str(extra.get("cache_reuse_mode", "chunk")))
        cache_config = getattr(vllm_config, "cache_config", None)
        self._vllm_prefix_caching_enabled = bool(
            getattr(cache_config, "enable_prefix_caching", False)
        )
        self._runtime_config_ready = False
        self._rope_base: float = 10000.0
        self._rope_rotary_dim: int = 0
        self._rope_is_neox_style: bool = True
        self._rope_delta_scale: float = float(
            extra.get("rope_delta_scale", DEFAULT_ROPE_DELTA_SCALE)
        )
        self._load_key_scale: float = float(extra.get("load_key_scale", 1.0))
        self._load_value_scale: float = float(extra.get("load_value_scale", 1.0))
        self._init_rope_config(vllm_config)

        if role == KVConnectorRole.SCHEDULER:
            self._ipc_sync = IPCClientSync(self._socket_path)
            self._refresh_runtime_config()
            self._pending_loads: dict[str, dict[str, Any]] = {}
            self._pending_stores: dict[str, dict[str, Any]] = {}
            self._pending_alloc: dict[str, PendingStore] = {}
            self._req_tokens: dict[str, list[int]] = {}
        else:
            self._transfer_ready = False
            self._transfer_mode = str(extra.get("transfer_mode", "iouring"))
            self._ipc_load_async = IPCClientAsync(self._socket_path)
            self._ipc_store_async = IPCClientAsync(self._socket_path)
            self._ipc_async = self._ipc_store_async
            self._kv_caches: dict[str, torch.Tensor] = {}
            self._layer_names: list[str] = []
            self._layer_idx_map: dict[str, int] = {}
            self._meta: DaserConnectorMeta | None = None
            self._save_futures: list = []
            self._pending_save_staging_bytes = 0
            self._store_staging_bytes = 0
            self._pending_store_staging_limit_bytes = 0
            self._staging_pool = None
            self._pending_commits: set[str] = set()
            self._load_loop = asyncio.new_event_loop()
            self._store_loop = asyncio.new_event_loop()
            self._bg_loop = self._store_loop
            self._load_thread = threading.Thread(
                target=self._run_load_loop,
                daemon=True,
                name="daser-load-io",
            )
            self._store_thread = threading.Thread(
                target=self._run_store_loop,
                daemon=True,
                name="daser-store-io",
            )
            self._load_thread.start()
            self._store_thread.start()

        logger.info("[CONNECTOR] role=%s socket=%s", role.name, self._socket_path)

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """Request vLLM cross-layer KV cache blocks for bulk chunk transfers.

        Returns:
            True so vLLM stores all layers for a block contiguously when the
            selected attention backend supports it.

        Async/thread-safety:
            Pure config property read during vLLM worker initialization.
        """
        return True

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Return the vLLM KV cache layout required by DaseR.

        Args:
            vllm_config: vLLM runtime config.

        Returns:
            ``"NHD"`` so cross-layer FlashAttention layout is
            ``[blocks, layers, 2, block, heads, head_dim]``, matching DaseR's
            slot-major staging order.

        Async/thread-safety:
            Class-level config helper with no mutable state.
        """
        return "NHD"

    def _refresh_runtime_config(self) -> None:
        """Refresh server-owned runtime config over IPC when available."""
        client = getattr(self, "_ipc_sync", None)
        owns_client = client is None
        if client is None:
            client = IPCClientSync(self._socket_path)
        try:
            config = client.get_runtime_config()
        except Exception as exc:  # noqa: BLE001
            logger.info("[CONNECTOR] runtime config unavailable: %s", exc)
            return
        finally:
            if owns_client:
                client.close()

        self._store_path = str(config.get("store_path", self._store_path))
        self._slot_size = int(config.get("slot_size", self._slot_size))
        self._block_tokens = int(config.get("block_tokens", self._block_tokens))
        self._model_id = str(config.get("model_id", self._model_id))
        self._set_cache_reuse_strategy(str(config["cache_reuse_mode"]))
        self._runtime_config_ready = bool(self._store_path and self._slot_size)
        self._transfer_mode = str(
            config.get("transfer_mode", getattr(self, "_transfer_mode", "iouring"))
        )
        logger.info(
            "[CONNECTOR] runtime config store=%s slot_size=%d block_tokens=%d "
            "model=%s transfer=%s",
            self._store_path,
            self._slot_size,
            self._block_tokens,
            self._model_id,
            getattr(self, "_transfer_mode", "iouring"),
        )

    def _set_cache_reuse_strategy(self, cache_reuse_mode: str) -> None:
        """Set scheduler cache reuse strategy.

        Args:
            cache_reuse_mode: either ``"chunk"`` or ``"prefix"``.
        """
        self._cache_reuse_strategy = build_cache_reuse_strategy(
            cache_reuse_mode,
            self._block_tokens,
        )

    def _discard_pending_request(self, req_id: str) -> None:
        """Clear scheduler-side pending state for a request.

        Args:
            req_id: vLLM request ID.
        """
        self._pending_loads.pop(req_id, None)
        self._pending_stores.pop(req_id, None)
        self._pending_alloc.pop(req_id, None)

    def _init_rope_config(self, vllm_config: "VllmConfig") -> None:
        """Extract default RoPE settings from vLLM model config.

        Args:
            vllm_config: vLLM runtime config passed to the connector.
        """
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return
        try:
            head_size = int(model_config.get_head_size())
        except Exception:  # noqa: BLE001
            logger.warning("[CONNECTOR] could not infer RoPE head size")
            return

        hf_text_config = getattr(model_config, "hf_text_config", None)
        rope_parameters = getattr(hf_text_config, "rope_parameters", None) or {}
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        model_type = str(getattr(hf_text_config, "model_type", ""))
        if "qwen" in model_type and "rope_theta" not in rope_parameters:
            rope_base = 1000000.0
        else:
            rope_base = float(rope_parameters.get("rope_theta", 10000.0))
        partial = float(rope_parameters.get("partial_rotary_factor", 1.0))
        rotary_dim = int(head_size * partial)

        self._rope_base = rope_base
        self._rope_rotary_dim = rotary_dim
        self._rope_is_neox_style = True
        logger.info(
            "[CONNECTOR] rope base=%s rotary_dim=%d neox=%s",
            self._rope_base,
            self._rope_rotary_dim,
            self._rope_is_neox_style,
        )
        logger.info(
            "[CONNECTOR] load tuning rope_delta_scale=%s key_scale=%s value_scale=%s",
            self._rope_delta_scale,
            self._load_key_scale,
            self._load_value_scale,
        )
