# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
import json
import os

BLOCK_TOKENS = 16


@dataclass(frozen=True)
class ModelGeometry:
    """KV cache geometry derived from a HuggingFace model config.

    Attributes:
        num_kv_heads: number of KV attention heads.
        head_dim: per-head hidden dimension.
        num_layers: transformer layer count.
        dtype_bytes: bytes per scalar element.
    """

    num_kv_heads: int
    head_dim: int
    num_layers: int
    dtype_bytes: int

    @property
    def slot_size(self) -> int:
        """Return bytes required for one vLLM KV block across all layers."""
        return (
            self.num_kv_heads
            * self.head_dim
            * 2  # K and V
            * self.num_layers
            * BLOCK_TOKENS
            * self.dtype_bytes
        )


def _dtype_bytes(dtype: object) -> int:
    """Return storage bytes for a HuggingFace dtype string.

    Args:
        dtype: raw dtype value from model config.

    Returns:
        Number of bytes per element.
    """
    dtype_str = str(dtype or "").lower()
    if "float32" in dtype_str or "fp32" in dtype_str:
        return 4
    return 2


def model_geometry_from_path(model_path: str) -> ModelGeometry:
    """Derive KV cache geometry from ``model_path/config.json``.

    Args:
        model_path: HuggingFace model directory containing ``config.json``.

    Returns:
        ModelGeometry used by both DaseR server and vLLM connector config.

    Raises:
        ValueError: if required model geometry fields are missing or invalid.
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        payload = json.load(f)

    hidden_size = int(payload.get("hidden_size", 0))
    num_attention_heads = int(payload.get("num_attention_heads", 0))
    num_kv_heads = int(
        payload.get("num_key_value_heads")
        or payload.get("multi_query_group_num")
        or num_attention_heads
    )
    num_layers = int(payload.get("num_hidden_layers") or payload.get("n_layer") or 0)
    head_dim = int(payload.get("head_dim") or payload.get("kv_channels") or 0)
    if head_dim == 0 and hidden_size > 0 and num_attention_heads > 0:
        head_dim = hidden_size // num_attention_heads

    missing = [
        name
        for name, value in [
            ("num_key_value_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("num_hidden_layers", num_layers),
        ]
        if value <= 0
    ]
    if missing:
        raise ValueError(
            f"model config {config_path} is missing valid KV geometry: "
            + ", ".join(missing)
        )

    return ModelGeometry(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dtype_bytes=_dtype_bytes(payload.get("torch_dtype")),
    )


@dataclass
class DaserConfig:
    """Top-level configuration for DaseR.

    Paths are derived from ``store_dir`` so vLLM connector configuration can
    reuse the server's resolved parameters without duplicating CLI flags.

    Attributes:
        model_path: HuggingFace model path used by vLLM and tokenizer loading.
        store_dir: directory containing DaseR data and generated config files.
        total_store_bytes: total KV store capacity.
        total_slots: number of fixed-size slots in the ring buffer.
        ipc_socket_path: Unix socket path for Connector <-> Server IPC.
        log_level: log level string passed to init_logger.
        block_tokens: tokens per vLLM block (default 16).
        model_id: identifier string for the model, used to prevent
                  cross-model cache reuse.
        cache_reuse_mode: retrieval/position strategy selected at startup.
    """

    model_path: str = ""
    store_dir: str = "/mnt/xfs/daser"
    total_store_bytes: int = 10 * 1024 * 1024 * 1024
    ipc_socket_path: str = "/tmp/daser.sock"
    log_level: str = "INFO"

    block_tokens: int = BLOCK_TOKENS
    cache_reuse_mode: str = "prefix"

    @property
    def store_path(self) -> str:
        """Absolute path to the KV data file."""
        return os.path.join(self.store_dir, "daser.store")

    @property
    def index_path(self) -> str:
        """Absolute path to the serialized metadata index."""
        return os.path.join(self.store_dir, "daser.index")

    @property
    def connector_config_path(self) -> str:
        """Path where server writes reusable vLLM connector config."""
        return os.path.join(self.store_dir, "daser.connector.json")

    @property
    def vllm_launch_script_path(self) -> str:
        """Path where server writes a runnable vLLM launch script."""
        return os.path.join(self.store_dir, "vllm-serve-daser.sh")

    @property
    def model_id(self) -> str:
        """Model identifier used for cache isolation."""
        return self.model_path

    @property
    def total_slots(self) -> int:
        """Return total ring-buffer slots available in the store."""
        return self.total_store_bytes // self.resolved_slot_size()

    @property
    def aligned_store_bytes(self) -> int:
        """Return store capacity rounded down to a whole number of slots."""
        return self.total_slots * self.resolved_slot_size()

    def resolved_slot_size(self) -> int:
        """Return bytes per KV slot derived from the model config.

        Returns:
            Slot size in bytes.
        """
        return model_geometry_from_path(self.model_path).slot_size

    def connector_extra_config(self) -> dict[str, object]:
        """Return vLLM ``kv_connector_extra_config`` values.

        Returns:
            Dict that vLLM can pass to ``DaserConnector``.
        """
        return {
            "socket_path": self.ipc_socket_path,
            "store_path": self.store_path,
            "slot_size": self.resolved_slot_size(),
            "block_tokens": self.block_tokens,
            "model_id": self.model_id,
        }

    def connector_config(self) -> dict[str, object]:
        """Return full vLLM ``--kv-transfer-config`` JSON payload.

        Returns:
            Dict accepted by vLLM's ``--kv-transfer-config`` argument.
        """
        return {
            "kv_connector": "DaserConnector",
            "kv_connector_module_path": "daser.connector.daser_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": self.connector_extra_config(),
        }
