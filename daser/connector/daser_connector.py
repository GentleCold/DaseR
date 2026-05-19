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
from daser.connector.helpers import PendingStore, hash_tokens
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.scheduler import (
    SchedulerConnectorMixin,
    _block_ids_for_chunk,
    _contiguous_prefix_tokens,
)
from daser.connector.worker import (
    DEFAULT_ROPE_DELTA_SCALE,
    WorkerConnectorMixin,
    _apply_rope_delta_to_key_block,
    _build_store_write_spans,
    _copy_kv_cache_to_staging,
    _copy_staging_to_kv_cache,
)
from daser.logging import init_logger

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_ROPE_DELTA_SCALE",
    "DaserConnector",
    "DaserConnectorMeta",
    "ReqLoadSpec",
    "ReqStoreSpec",
    "_apply_rope_delta_to_key_block",
    "_block_ids_for_chunk",
    "_build_store_write_spans",
    "_contiguous_prefix_tokens",
    "_copy_kv_cache_to_staging",
    "_copy_staging_to_kv_cache",
    "hash_tokens",
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
        self._runtime_config_ready = False
        self._rope_base: float = 10000.0
        self._rope_rotary_dim: int = 0
        self._rope_is_neox_style: bool = True
        self._rope_delta_scale: float = float(
            extra.get("rope_delta_scale", DEFAULT_ROPE_DELTA_SCALE)
        )
        self._load_key_scale: float = float(extra.get("load_key_scale", 1.0))
        self._load_value_scale: float = float(extra.get("load_value_scale", 1.0))
        self._max_inflight_store_bytes: int = int(
            extra.get("max_inflight_store_bytes", 1 << 30)
        )
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
            self._transfer_mode = str(extra.get("transfer_mode", "gds"))
            self._ipc_async = IPCClientAsync(self._socket_path)
            self._kv_caches: dict[str, torch.Tensor] = {}
            self._layer_names: list[str] = []
            self._layer_idx_map: dict[str, int] = {}
            self._meta: DaserConnectorMeta | None = None
            self._store_futures: list = []
            self._pending_commits: set[str] = set()
            self._save_all_block_ids: list[int] = []
            self._save_req_slot_ranges: dict[str, tuple[int, int]] = {}
            self._save_step_staging: torch.Tensor | None = None
            self._inflight_store_bytes: int = 0
            self._bg_loop = asyncio.new_event_loop()
            self._bg_thread = threading.Thread(
                target=self._run_bg_loop,
                daemon=True,
                name="daser-io",
            )
            self._bg_thread.start()

        logger.info("[CONNECTOR] role=%s socket=%s", role.name, self._socket_path)

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
        self._runtime_config_ready = bool(self._store_path and self._slot_size)
        self._transfer_mode = str(
            config.get("transfer_mode", getattr(self, "_transfer_mode", "gds"))
        )
        logger.info(
            "[CONNECTOR] runtime config store=%s slot_size=%d block_tokens=%d "
            "model=%s transfer=%s",
            self._store_path,
            self._slot_size,
            self._block_tokens,
            self._model_id,
            getattr(self, "_transfer_mode", "gds"),
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
