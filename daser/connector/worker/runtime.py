# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

if TYPE_CHECKING:
    # Third Party
    from vllm.attention import AttentionMetadata
    from vllm.forward_context import ForwardContext

# First Party
from daser.connector.ipc_client import IPCClientSync
from daser.connector.metadata import (
    DaserConnectorMeta,
)
from daser.connector.worker.load import LoadPipeline
from daser.connector.worker.memory import (
    FixedCudaStagingPool,
    derive_staging_layout,
)
from daser.connector.worker.staging import (
    CROSS_LAYER_KV_CACHE_KEY,
    FUSED_RESTORE_MIN_SLOTS,
)
from daser.connector.worker.store import StorePipeline
from daser.logging import init_logger
from daser.ops.rope_apply import (
    apply_rope_delta_to_key_block as _apply_rope_delta_to_key_block,
)
from daser.ops.rope_apply import (
    apply_rope_delta_to_kv_key_block_table,
    restore_cross_layer_kv_cache_table,
)

logger = init_logger(__name__)

_ROPE_WARMUP_BLOCKS = 1
_LOAD_REQUEST_MAX_INFLIGHT = 8
_LOAD_STAGING_RESERVE_BYTES = 1 << 30


def _local_slot_bytes(connector: Any) -> int:
    """Return per-rank slot bytes, falling back for TP=1 test probes."""
    local_slot_size = int(getattr(connector, "_local_slot_size", 0))
    return local_slot_size or int(connector._slot_size)  # noqa: SLF001


def _validate_tp_layout(
    local_slot_size: int,
    storage_slot_size: int,
    tp_size: int,
    server_tp_size: int,
    tp_rank: int,
    rank_stride_bytes: int = 0,
) -> None:
    """Validate worker KV geometry against the server-owned TP layout.

    Args:
        local_slot_size: Slot bytes measured from the worker KV tensor.
        storage_slot_size: Aggregate slot bytes reported by the server.
        tp_size: vLLM worker tensor-parallel size.
        server_tp_size: Tensor-parallel size reported by the server.
        tp_rank: Current vLLM tensor-parallel rank.
        rank_stride_bytes: Byte distance between server-owned rank lanes.

    Raises:
        ValueError: if rank counts or slot geometry do not match.

    Async/thread-safety:
        Pure startup validation called before request traffic.
    """
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid TP rank {tp_rank} for size {tp_size}")
    if not storage_slot_size:
        return
    if server_tp_size != tp_size:
        raise ValueError(
            f"vLLM TP size {tp_size} does not match DaseR TP size {server_tp_size}"
        )
    if local_slot_size * tp_size != storage_slot_size:
        raise ValueError(
            "worker KV slot geometry does not match DaseR storage layout: "
            f"local={local_slot_size} tp={tp_size} storage={storage_slot_size}"
        )
    if tp_size > 1 and rank_stride_bytes <= 0:
        raise ValueError("DaseR runtime config is missing TP rank stride")


def _warm_rope_apply_backends(
    device: torch.device,
    dtype: torch.dtype,
    block_tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    rope_base: float,
    is_neox_style: bool,
) -> None:
    """Warm dynamic-shape RoPE apply operators.

    Args:
        device: device that owns the worker KV cache.
        dtype: KV cache dtype.
        block_tokens: tokens per cache block.
        heads: number of KV heads.
        head_dim: per-head dimension.
        rotary_dim: number of dimensions covered by RoPE.
        rope_base: RoPE theta/base.
        is_neox_style: True for split-half rotation, False for interleaved.

    Async/thread-safety:
        Runs synchronously during worker KV cache registration, before request
        traffic starts. It launches CUDA work on the current stream. TileLang
        failures are surfaced to avoid silently entering a slow restore path.
    """
    if device.type != "cuda" or rotary_dim <= 0 or head_dim < rotary_dim:
        return
    sample = torch.empty(
        (_ROPE_WARMUP_BLOCKS, block_tokens, heads, head_dim),
        dtype=dtype,
        device=device,
    )
    _apply_rope_delta_to_key_block(
        sample,
        delta=1,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )
    torch.cuda.synchronize(device)


def _warm_cross_layer_restore_backends(
    device: torch.device,
    dtype: torch.dtype,
    layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    rope_base: float,
    is_neox_style: bool,
) -> None:
    """Warm cross-layer staging restore TileLang kernels.

    Args:
        device: device that owns the worker KV cache.
        dtype: KV cache dtype.
        layers: number of model KV layers.
        block_tokens: tokens per cache block.
        heads: number of KV heads.
        head_dim: per-head dimension.
        rotary_dim: number of dimensions covered by RoPE.
        rope_base: RoPE theta/base.
        is_neox_style: True for split-half rotation, False for interleaved.

    Async/thread-safety:
        Runs synchronously during worker KV cache registration, before request
        traffic starts. TileLang import/compile failures are surfaced to avoid
        silently entering a slow restore path.
    """
    if device.type != "cuda" or rotary_dim <= 0 or head_dim < rotary_dim:
        return
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    freqs = inv_freq
    cos_table = freqs.cos().contiguous()
    sin_table = freqs.sin().contiguous()
    for blocks, use_fused_restore in (
        (_ROPE_WARMUP_BLOCKS, False),
        (FUSED_RESTORE_MIN_SLOTS, True),
    ):
        sample = torch.empty(
            blocks,
            layers,
            2,
            block_tokens,
            heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        if use_fused_restore:
            dst = torch.empty_like(sample)
            restore_cross_layer_kv_cache_table(
                sample,
                dst,
                cos_table=cos_table,
                sin_table=sin_table,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            )
        else:
            apply_rope_delta_to_kv_key_block_table(
                sample,
                cos_table=cos_table,
                sin_table=sin_table,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            )
    torch.cuda.synchronize(device)


class WorkerRuntime:
    """Own worker KV layout, step metadata, pipelines, and completion state.

    Async/thread-safety:
        Public methods are called on vLLM worker threads. Blocking NVMe work is
        submitted to the runtime's load or store pipeline loop.
    """

    def __init__(
        self,
        *,
        socket_path: str,
        transfer_mode: str,
        skip_l2: bool,
        tp_size: int,
        tp_rank: int,
        server_tp_size: int,
        slot_size: int,
        store_path: str,
        rank_stride_bytes: int,
        rope_base: float,
        rope_rotary_dim: int,
        rope_is_neox_style: bool,
        rope_delta_scale: float,
        load_key_scale: float,
        load_value_scale: float,
        kv_cache_config: Any,
    ) -> None:
        self._socket_path = socket_path
        self._transfer_mode = transfer_mode
        self._skip_l2 = skip_l2
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._server_tp_size = server_tp_size
        self._slot_size = slot_size
        self._local_slot_size = 0
        self._store_path = store_path
        self._rank_stride_bytes = rank_stride_bytes
        self._rope_base = rope_base
        self._rope_rotary_dim = rope_rotary_dim
        self._rope_is_neox_style = rope_is_neox_style
        self._rope_delta_scale = rope_delta_scale
        self._load_key_scale = load_key_scale
        self._load_value_scale = load_value_scale
        self._kv_cache_config = kv_cache_config
        self._role = KVConnectorRole.WORKER
        self._transfer_ready = False
        self._pipelines_initialized = False
        self._load_pipeline = LoadPipeline(socket_path, _LOAD_REQUEST_MAX_INFLIGHT)
        self._store_pipeline = StorePipeline(socket_path)
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._layer_idx_map: dict[str, int] = {}
        self._meta: DaserConnectorMeta | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register the per-layer KV cache tensors.

        Args:
            kv_caches: dict mapping layer_name -> KV tensor.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())
        self._layer_idx_map = {name: idx for idx, name in enumerate(self._layer_names)}
        sample = next(iter(kv_caches.values()), None)
        if sample is not None:
            logger.info(
                "[CONNECTOR] register_kv_caches: %d layers, first shape=%s dtype=%s",
                len(kv_caches),
                sample.shape,
                sample.dtype,
            )

        if self._layer_names and sample is not None:
            num_blocks = sample.shape[1] if sample.dim() >= 2 else 1
            layer_size = sample.nbytes // num_blocks
            local_slot_size = layer_size * len(self._layer_names)
            tp_size = self._tp_size
            _validate_tp_layout(
                local_slot_size,
                self._slot_size,
                tp_size,
                self._server_tp_size,
                self._tp_rank,
                self._rank_stride_bytes,
            )
            self._local_slot_size = local_slot_size
            if self._slot_size == 0:
                self._slot_size = local_slot_size * tp_size
            logger.info(
                "[CONNECTOR] registered local_slot_size=%d from %d layers",
                self._local_slot_size,
                len(self._layer_names),
            )

        if sample is not None:
            self._configure_pipelines(sample)
            if sample.dim() >= 5:
                _warm_rope_apply_backends(
                    device=sample.device,
                    dtype=sample.dtype,
                    block_tokens=int(sample.shape[-3]),
                    heads=int(sample.shape[-2]),
                    head_dim=int(sample.shape[-1]),
                    rotary_dim=self._rope_rotary_dim,
                    rope_base=self._rope_base,
                    is_neox_style=self._rope_is_neox_style,
                )

        self._init_server_transfer()

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type[Any],
    ) -> None:
        """Register vLLM's cross-layer KV cache tensor.

        Args:
            kv_cache: vLLM tensor whose logical layout starts with
                ``[blocks, layers, 2, block_tokens, heads, head_dim]`` for the
                NHD layout DaseR requests.
            attn_backend: Attention backend that created ``kv_cache``.

        Async/thread-safety:
            Called once during worker initialization before request traffic.
        """
        kv_cache_config = self._kv_cache_config
        layer_names: list[str] = []
        if kv_cache_config is not None:
            for group in getattr(kv_cache_config, "kv_cache_groups", []):
                layer_names.extend(list(getattr(group, "layer_names", [])))
        if not layer_names:
            layer_count = int(kv_cache.shape[1]) if kv_cache.dim() >= 2 else 0
            layer_names = [f"layer.{idx}" for idx in range(layer_count)]
        self._kv_caches = {CROSS_LAYER_KV_CACHE_KEY: kv_cache}
        self._layer_names = layer_names
        self._layer_idx_map = {name: idx for idx, name in enumerate(self._layer_names)}
        if kv_cache.dim() < 6:
            logger.warning(
                "[CONNECTOR] cross-layer KV cache has unsupported shape=%s",
                tuple(kv_cache.shape),
            )
            return
        layer_size = kv_cache[0, 0].nbytes
        local_slot_size = layer_size * len(self._layer_names)
        tp_size = self._tp_size
        _validate_tp_layout(
            local_slot_size,
            self._slot_size,
            tp_size,
            self._server_tp_size,
            self._tp_rank,
            self._rank_stride_bytes,
        )
        self._local_slot_size = local_slot_size
        if self._slot_size == 0:
            self._slot_size = local_slot_size * tp_size
        logger.info(
            "[CONNECTOR] registered cross-layer local_slot_size=%d from %d layers",
            self._local_slot_size,
            len(self._layer_names),
        )
        load_staging_depth = self._configure_pipelines(kv_cache)
        logger.info(
            "[CONNECTOR] register_cross_layers_kv_cache: layers=%d shape=%s "
            "dtype=%s load_request_max_inflight=%d load_staging_depth=%d",
            len(self._layer_names),
            tuple(kv_cache.shape),
            kv_cache.dtype,
            _LOAD_REQUEST_MAX_INFLIGHT,
            load_staging_depth,
        )
        _warm_rope_apply_backends(
            device=kv_cache.device,
            dtype=kv_cache.dtype,
            block_tokens=int(kv_cache.shape[-3]),
            heads=int(kv_cache.shape[-2]),
            head_dim=int(kv_cache.shape[-1]),
            rotary_dim=self._rope_rotary_dim,
            rope_base=self._rope_base,
            is_neox_style=self._rope_is_neox_style,
        )
        _warm_cross_layer_restore_backends(
            device=kv_cache.device,
            dtype=kv_cache.dtype,
            layers=int(kv_cache.shape[1]),
            block_tokens=int(kv_cache.shape[-3]),
            heads=int(kv_cache.shape[-2]),
            head_dim=int(kv_cache.shape[-1]),
            rotary_dim=self._rope_rotary_dim,
            rope_base=self._rope_base,
            is_neox_style=self._rope_is_neox_style,
        )
        self._init_server_transfer()

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        self._meta = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        self._meta = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Submit cache-hit requests to the load pipeline.

        Args:
            forward_context: vLLM forward context for this step.
            **kwargs: Additional vLLM hook arguments, currently unused.

        Async/thread-safety:
            Called on the vLLM worker thread. Transfer and restore execute on
            the load pipeline thread.
        """
        del forward_context, kwargs
        if self._meta is None or not self._meta.reqs_to_load:
            return
        reqs_to_load = dict(self._meta.reqs_to_load)
        if not self._ensure_transfer_ready():
            self._load_pipeline.mark_failed(
                reqs_to_load,
                "server transfer config is not ready",
            )
            return
        self._load_pipeline.start(reqs_to_load)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Return after request-level load completion restored every layer.

        Args:
            layer_name: Layer reported by vLLM; no per-layer wait is required.

        Async/thread-safety:
            Called on the vLLM worker thread.
        """
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit this layer's KV blocks for server-owned transfer.

        Args:
            layer_name: name of the current attention layer.
            kv_layer: full KV cache tensor for this layer.
            attn_metadata: attention metadata (not directly used).
        """
        if self._meta is None or not self._meta.reqs_to_store:
            return
        if not self._ensure_transfer_ready():
            return

        if layer_name not in self._layer_idx_map:
            logger.warning(
                "[CONNECTOR] save_kv_layer: unknown layer %s, skipping", layer_name
            )

    def wait_for_save(self) -> None:
        """Queue stores until vLLM reports request completion."""
        if self._meta is None:
            return
        reqs_to_store = dict(self._meta.reqs_to_store)
        commit_keys = {
            spec.chunk_key for spec in reqs_to_store.values() if spec.block_ids
        }
        if commit_keys:
            self._store_pipeline.queue_finished(reqs_to_store, commit_keys)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Collect completed background transfers after a worker step.

        Args:
            finished_req_ids: Request IDs that vLLM finished in this step.

        Returns:
            Finished-saving request IDs and finished async-loading request IDs.
        """
        finished_recving = self._load_pipeline.collect_finished()
        finished_sending = self._store_pipeline.collect_finished(finished_req_ids)
        return finished_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return and clear block IDs whose async load failed.

        Returns:
            vLLM block IDs that should be treated as invalid.
        """
        return self._load_pipeline.take_invalid_block_ids()

    def shutdown(self) -> None:
        """Stop the background IO loop."""
        if self._role != KVConnectorRole.WORKER:
            return
        self._load_pipeline.shutdown()
        self._store_pipeline.shutdown()

    def _ensure_transfer_ready(self) -> bool:
        """Refresh config and initialize both pipeline transfer clients."""
        if not self._transfer_ready:
            self._refresh_runtime_config()
            if not self._slot_size or (not self._skip_l2 and not self._store_path):
                logger.warning(
                    "[CONNECTOR] server transfer config is not ready; start DaseR "
                    "server before sending requests",
                )
                return False

            _validate_tp_layout(
                _local_slot_bytes(self),
                self._slot_size,
                self._tp_size,
                self._server_tp_size,
                self._tp_rank,
                self._rank_stride_bytes,
            )
            self._load_pipeline.configure_rank_geometry(
                self._rank_stride_bytes,
                self._tp_rank,
            )
            self._store_pipeline.configure_rank_geometry(
                self._rank_stride_bytes,
                self._tp_rank,
                self._tp_size,
            )
            self._transfer_ready = True
            logger.info("[CONNECTOR] server transfer mode=%s", self._transfer_mode)

        if not self._pipelines_initialized:
            self._load_pipeline.initialize_transfer()
            self._store_pipeline.initialize_transfer()
            self._pipelines_initialized = True
        return True

    def _refresh_runtime_config(self) -> None:
        """Refresh worker-owned storage geometry directly over sync IPC."""
        client = IPCClientSync(self._socket_path)
        try:
            config = client.get_runtime_config()
        except Exception as exc:  # noqa: BLE001
            logger.info("[CONNECTOR] runtime config unavailable: %s", exc)
            return
        finally:
            client.close()
        self._store_path = str(config.get("store_path", self._store_path))
        self._slot_size = int(config.get("slot_size", self._slot_size))
        self._server_tp_size = int(
            config.get("tensor_parallel_size", self._server_tp_size)
        )
        self._rank_stride_bytes = int(
            config.get("rank_stride_bytes", self._rank_stride_bytes)
        )
        self._skip_l2 = bool(config.get("skip_l2", self._skip_l2))
        self._transfer_mode = str(config.get("transfer_mode", self._transfer_mode))

    def _init_server_transfer(self) -> None:
        """Initialize both pipeline-owned transfer clients.

        Async/thread-safety:
            Called on the worker thread after KV-cache registration. Each
            pipeline performs its initialization on its private event loop.
        """
        self._ensure_transfer_ready()

    def _configure_pipelines(self, sample: torch.Tensor) -> int:
        """Configure load and store pipelines from one finalized KV layout.

        Args:
            sample: Representative registered KV-cache tensor.

        Returns:
            Number of preallocated load staging buffers.

        Async/thread-safety:
            Called once on the worker thread during KV-cache registration.
        """
        staging_bytes, load_depth, store_depth, allocated_bytes = derive_staging_layout(
            sample.device,
            self._local_slot_size,
            _LOAD_REQUEST_MAX_INFLIGHT,
            _LOAD_STAGING_RESERVE_BYTES,
        )
        store_pool = FixedCudaStagingPool(
            device=sample.device,
            buffer_bytes=staging_bytes,
            depth=store_depth,
        )
        load_pool = FixedCudaStagingPool(
            device=sample.device,
            buffer_bytes=staging_bytes,
            depth=load_depth,
        )
        self._store_pipeline.configure(
            kv_caches=self._kv_caches,
            layer_names=self._layer_names,
            layer_idx_map=self._layer_idx_map,
            local_slot_size=self._local_slot_size,
            rank_stride_bytes=self._rank_stride_bytes,
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            staging_bytes=staging_bytes,
            staging_pool=store_pool,
        )
        self._load_pipeline.configure(
            kv_caches=self._kv_caches,
            layer_names=self._layer_names,
            local_slot_size=self._local_slot_size,
            rank_stride_bytes=self._rank_stride_bytes,
            tp_rank=self._tp_rank,
            staging_pool=load_pool,
            load_key_scale=self._load_key_scale,
            load_value_scale=self._load_value_scale,
            rope_delta_scale=self._rope_delta_scale,
            rope_base=self._rope_base,
            rope_rotary_dim=self._rope_rotary_dim,
            rope_is_neox_style=self._rope_is_neox_style,
        )
        logger.info(
            "[CONNECTOR] preallocated staging buffer_bytes=%d total_bytes=%d "
            "load_depth=%d store_depth=%d",
            staging_bytes,
            allocated_bytes,
            load_pool.depth,
            store_pool.depth,
        )
        return load_pool.depth
