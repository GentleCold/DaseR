# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

# First Party
from daser.connector.ipc_client import IPCClientSync
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.request_lifecycle import RequestLifecycle
from daser.connector.scheduler import SchedulerConnectorMixin
from daser.connector.staging import (
    DEFAULT_ROPE_DELTA_SCALE,
)
from daser.connector.worker import WorkerConnectorMixin
from daser.connector.worker_runtime import WorkerRuntime
from daser.logging import init_logger

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_ROPE_DELTA_SCALE",
    "DaserConnector",
    "DaserConnectorMeta",
    "ReqLoadSpec",
    "ReqStoreSpec",
]


def _extract_rope_config(vllm_config: "VllmConfig") -> tuple[float, int, bool]:
    """Extract worker RoPE geometry from the vLLM model config."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return 10000.0, 0, True
    try:
        head_size = int(model_config.get_head_size())
    except Exception:  # noqa: BLE001
        logger.warning("[CONNECTOR] could not infer RoPE head size")
        return 10000.0, 0, True
    hf_text_config = getattr(model_config, "hf_text_config", None)
    rope_parameters = getattr(hf_text_config, "rope_parameters", None) or {}
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}
    model_type = str(getattr(hf_text_config, "model_type", ""))
    rope_base = (
        1000000.0
        if "qwen" in model_type and "rope_theta" not in rope_parameters
        else float(rope_parameters.get("rope_theta", 10000.0))
    )
    partial = float(rope_parameters.get("partial_rotary_factor", 1.0))
    return rope_base, int(head_size * partial), True


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

        socket_path = str(extra.get("socket_path", "/tmp/daser.sock"))
        if role == KVConnectorRole.SCHEDULER:
            self._request_lifecycle = RequestLifecycle(
                ipc_client=IPCClientSync(socket_path),
                block_tokens=16,
                slot_size=0,
                model_id="default",
                cache_reuse_mode=str(extra.get("cache_reuse_mode", "chunk")),
                runtime_config_ready=False,
            )
            self._request_lifecycle.refresh_runtime_config()
        else:
            parallel_config = getattr(vllm_config, "parallel_config", None)
            tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
            tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
            rope_base, rope_rotary_dim, rope_is_neox_style = _extract_rope_config(
                vllm_config
            )
            self._worker_runtime = WorkerRuntime(
                socket_path=socket_path,
                transfer_mode=str(extra.get("transfer_mode", "iouring")),
                skip_l2=bool(extra.get("skip_l2", False)),
                tp_size=tp_size,
                tp_rank=tp_rank,
                server_tp_size=1,
                slot_size=0,
                store_path="",
                rank_stride_bytes=0,
                rope_base=rope_base,
                rope_rotary_dim=rope_rotary_dim,
                rope_is_neox_style=rope_is_neox_style,
                rope_delta_scale=float(
                    extra.get("rope_delta_scale", DEFAULT_ROPE_DELTA_SCALE)
                ),
                load_key_scale=float(extra.get("load_key_scale", 1.0)),
                load_value_scale=float(extra.get("load_value_scale", 1.0)),
                kv_cache_config=kv_cache_config,
            )

        logger.info("[CONNECTOR] role=%s socket=%s", role.name, socket_path)

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
