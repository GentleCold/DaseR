# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass


@dataclass
class DaserConfig:
    """Top-level configuration for DaseR.

    slot_size is computed automatically from model params when it is 0.
    All paths should be absolute.

    Attributes:
        store_path: absolute path to the KV data file (daser.store).
        index_path: absolute path to the serialized index (daser.index).
        total_slots: number of fixed-size slots in the ring buffer.
        slot_size: size of one slot in bytes. 0 means compute from model params.
        ipc_socket_path: Unix socket path for Connector <-> Server IPC.
        log_level: log level string passed to init_logger.
        perf_log_enabled: whether to activate PerfLogger output.
        num_kv_heads: number of KV attention heads (for slot_size computation).
        head_dim: attention head dimension (for slot_size computation).
        num_layers: number of transformer layers (for slot_size computation).
        block_tokens: tokens per vLLM block (default 16).
        dtype_bytes: bytes per element, e.g. 2 for bf16.
        model_id: identifier string for the model, used to prevent
                  cross-model cache reuse.
    """

    store_path: str = "/mnt/xfs/daser.store"
    index_path: str = "/mnt/xfs/daser.index"
    total_slots: int = 1024
    slot_size: int = 0
    ipc_socket_path: str = "/tmp/daser.sock"
    log_level: str = "INFO"
    perf_log_enabled: bool = False

    # Model params used only when slot_size == 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_layers: int = 0
    block_tokens: int = 16
    dtype_bytes: int = 2
    model_id: str = "default"

    def resolved_slot_size(self) -> int:
        """Return slot_size, computing it from model params if slot_size is 0.

        Returns:
            Slot size in bytes.

        Raises:
            ValueError: if slot_size is 0 and any model param is 0.
        """
        if self.slot_size > 0:
            return self.slot_size
        for param, name in [
            (self.num_kv_heads, "num_kv_heads"),
            (self.head_dim, "head_dim"),
            (self.num_layers, "num_layers"),
        ]:
            if param == 0:
                raise ValueError(
                    f"slot_size is 0 but {name} is also 0; "
                    "provide either slot_size or all model params"
                )
        return (
            self.num_kv_heads
            * self.head_dim
            * 2  # K and V
            * self.num_layers
            * self.block_tokens
            * self.dtype_bytes
        )
