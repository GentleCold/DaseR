# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
import os
import threading
from typing import TYPE_CHECKING, Any

# Third Party
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata

# First Party
from daser.attention.capacity import FragPackCapacity, get_capacity
from daser.compression import CompressedStoreGeometry, default_online_codebooks
from daser.compression.attn_page import blob_slot_bytes
from daser.compression.format import ONLINE_TILE_SCALARS, digest_bytes
from daser.config import (
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_FRAGPACK,
    STORAGE_FORMAT_RAW,
)
from daser.connector.helpers import base_req_id
from daser.connector.ipc_client import IPCClientSync
from daser.connector.metadata import (
    DaserConnectorMeta,
)
from daser.connector.worker.load import LoadPipeline
from daser.connector.worker.memory import (
    DEFAULT_STAGING_BUDGET_BYTES,
    FixedCudaStagingPool,
    derive_staging_layout,
)
from daser.connector.worker.staging import (
    CROSS_LAYER_KV_CACHE_KEY,
    FUSED_RESTORE_MIN_SLOTS,
)
from daser.connector.worker.store import (
    DEFAULT_ONLINE_PACK_BATCH_SLOTS,
    StorePipeline,
)
from daser.logging import init_logger
from daser.ops.compressed_kv import warm_fused_online_kv_packer
from daser.ops.hbm_kv import page_bytes
from daser.ops.kv_layout import allocate_cross_layer_kv_cache, cross_layer_kv_view
from daser.ops.rope_apply import (
    apply_rope_delta_to_kv_key_block_table,
    restore_cross_layer_kv_cache_table,
)

logger = init_logger(__name__)

_ROPE_WARMUP_BLOCKS = 1
_LOAD_REQUEST_MAX_INFLIGHT = 8
_LOAD_STAGING_RESERVE_BYTES = 1 << 30
_TRANSFER_WARMUP_RETRY_S = 0.2
_TRANSFER_WARMUP_JOIN_TIMEOUT_S = 125.0


def forward_skips_layer_waits(forward_context: ForwardContext | None) -> bool:
    """Return whether this step's forward will not call ``wait_for_layer_load``.

    A full CUDA graph replay skips the Python layer hooks, and a step without
    attention metadata (no forward, or vLLM's model runner V2 entering a full
    graph) runs no attention layer at all.

    Args:
        forward_context: Context passed to ``start_load_kv``.

    Returns:
        True when per-layer waits cannot order the forward after the restore.
    """
    if forward_context is None or forward_context.attn_metadata is None:
        return True
    return forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL


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
    num_layers: int = 0,
) -> None:
    """Validate worker KV geometry against the server-owned TP layout.

    Args:
        local_slot_size: Slot bytes measured from the worker KV tensor.
        storage_slot_size: Aggregate slot bytes reported by the server.
        tp_size: vLLM worker tensor-parallel size.
        server_tp_size: Tensor-parallel size reported by the server.
        tp_rank: Current vLLM tensor-parallel rank.
        num_layers: KV layers the worker registered (0 = unknown); used to
            name the layer counts when slot sizes differ by whole layers.

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
        detail = ""
        layer_bytes = local_slot_size // num_layers if num_layers else 0
        if layer_bytes and storage_slot_size % (layer_bytes * tp_size) == 0:
            # Drafter (EAGLE) layers share the target's KV shape, so a
            # whole-layer difference is a layer-count mismatch.
            server_layers = storage_slot_size // (layer_bytes * tp_size)
            detail = (
                f"; vLLM registered {num_layers} KV layers but DaseR sized "
                f"slots for {server_layers}: start the DaseR server with "
                f"--extra-kv-layers {num_layers - server_layers} for layers "
                "beyond the model config (for example a speculative drafter)"
            )
        raise ValueError(
            "worker KV slot geometry does not match DaseR storage layout: "
            f"local={local_slot_size} tp={tp_size} storage={storage_slot_size}" + detail
        )


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
            dst = allocate_cross_layer_kv_cache(
                blocks,
                layers,
                block_tokens,
                heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
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
        rope_base: float,
        rope_rotary_dim: int,
        rope_is_neox_style: bool,
        rope_delta_scale: float,
        load_key_scale: float,
        load_value_scale: float,
        storage_format: str | None = None,
        online_pack_batch_slots: int = DEFAULT_ONLINE_PACK_BATCH_SLOTS,
        staging_budget_bytes: int = DEFAULT_STAGING_BUDGET_BYTES,
    ) -> None:
        if staging_budget_bytes <= 0:
            raise ValueError("staging_budget_bytes must be positive")
        self._staging_budget_bytes = int(staging_budget_bytes)
        self._allocated_staging_bytes = 0
        self._reported_capacity: FragPackCapacity | None = None
        self._capacity_stats: dict[str, Any] = {}
        self._socket_path = socket_path
        self._transfer_mode = transfer_mode
        self._skip_l2 = skip_l2
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._server_tp_size = server_tp_size
        self._slot_size = slot_size
        self._local_slot_size = 0
        self._store_path = store_path
        self._rope_base = rope_base
        self._rope_rotary_dim = rope_rotary_dim
        self._rope_is_neox_style = rope_is_neox_style
        self._rope_delta_scale = rope_delta_scale
        self._load_key_scale = load_key_scale
        self._load_value_scale = load_value_scale
        self._declared_storage_format = storage_format
        self._storage_format = storage_format or STORAGE_FORMAT_RAW
        self._compression_configured = False
        self._compression_codebook_hash: bytes | None = None
        self._compression_tile_scalars: int | None = None
        self._role = KVConnectorRole.WORKER
        self._transfer_ready = False
        self._pipelines_initialized = False
        self._transfer_init_lock = threading.Lock()
        self._transfer_warmup_stop = threading.Event()
        self._transfer_warmup_thread: threading.Thread | None = None
        self._transfer_warmup_error: BaseException | None = None
        self._load_pipeline = LoadPipeline(socket_path, _LOAD_REQUEST_MAX_INFLIGHT)
        self._store_pipeline = StorePipeline(
            socket_path, online_pack_batch_slots=online_pack_batch_slots
        )
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._layer_idx_map: dict[str, int] = {}
        self._meta: DaserConnectorMeta | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register vLLM's per-layer KV views as one cross-layer cache.

        Args:
            kv_caches: dict mapping layer_name -> ``[blocks, heads, tokens,
                2 * head_dim]`` views of vLLM's shared ``BLNHC`` buffer.

        Raises:
            ValueError: If the views do not form one block-outermost buffer or
                the slot geometry does not match the DaseR server.

        Async/thread-safety:
            Called once during worker initialization before request traffic.
        """
        if self._storage_format == STORAGE_FORMAT_FRAGPACK:
            self._register_fragpack_kv_caches(kv_caches)
            return
        layer_names, kv_cache = cross_layer_kv_view(kv_caches)
        self._kv_caches = {CROSS_LAYER_KV_CACHE_KEY: kv_cache}
        self._layer_names = layer_names
        self._layer_idx_map = {name: idx for idx, name in enumerate(layer_names)}
        local_slot_size = int(kv_cache[0].nbytes)
        _validate_tp_layout(
            local_slot_size,
            self._slot_size,
            self._tp_size,
            self._server_tp_size,
            self._tp_rank,
        )
        self._local_slot_size = local_slot_size
        if self._slot_size == 0:
            self._slot_size = local_slot_size * self._tp_size
        load_staging_depth = self._configure_pipelines(kv_cache)
        logger.info(
            "[CONNECTOR] register_kv_caches: layers=%d shape=%s dtype=%s "
            "local_slot_size=%d load_request_max_inflight=%d "
            "load_staging_depth=%d",
            len(layer_names),
            tuple(kv_cache.shape),
            kv_cache.dtype,
            local_slot_size,
            _LOAD_REQUEST_MAX_INFLIGHT,
            load_staging_depth,
        )
        _warm_cross_layer_restore_backends(
            device=kv_cache.device,
            dtype=kv_cache.dtype,
            layers=int(kv_cache.shape[1]),
            block_tokens=int(kv_cache.shape[3]),
            heads=int(kv_cache.shape[4]),
            head_dim=int(kv_cache.shape[5]),
            rotary_dim=self._rope_rotary_dim,
            rope_base=self._rope_base,
            is_neox_style=self._rope_is_neox_style,
        )
        if self._storage_format == STORAGE_FORMAT_COMPRESSED_ONLINE:
            warm_fused_online_kv_packer(
                kv_cache,
                max_slots_per_buffer=self._store_pipeline.max_online_pack_slots,
                tile_scalars=ONLINE_TILE_SCALARS,
            )
            self._prewarm_online_compression(kv_cache)
        self._init_server_transfer()

    def _register_fragpack_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register FragPack attention's per-layer page arenas.

        The pipelines never address these tensors: blocks move as blobs
        through the HBM store (``daser.ops.fragpack_codec``). They only provide the
        device, the model-stream fence, and the slot geometry.

        Args:
            kv_caches: dict mapping layer_name -> one layer's block pages, in
                layer order.

        Raises:
            ValueError: If the pages do not hold whole FragPack block-layers or the
                slot geometry does not match the DaseR server.

        Async/thread-safety:
            Called once during worker initialization before request traffic.
        """
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches)
        self._layer_idx_map = {name: i for i, name in enumerate(self._layer_names)}
        sample = next(iter(kv_caches.values()))
        page = sample.stride(0) * sample.element_size()
        kv_heads = page // page_bytes(1)
        if kv_heads < 1 or page % 16:
            raise ValueError(f"KV page of {page} B is not an FragPack block-layer page")
        local_slot_size = blob_slot_bytes(len(kv_caches), kv_heads)
        _validate_tp_layout(
            local_slot_size,
            self._slot_size,
            self._tp_size,
            self._server_tp_size,
            self._tp_rank,
        )
        self._local_slot_size = local_slot_size
        if self._slot_size == 0:
            self._slot_size = local_slot_size * self._tp_size
        load_staging_depth = self._configure_pipelines(sample)
        logger.info(
            "[CONNECTOR] register_kv_caches fragpack: layers=%d kv_heads=%d page=%d "
            "local_slot_size=%d load_staging_depth=%d",
            len(kv_caches),
            kv_heads,
            page,
            local_slot_size,
            load_staging_depth,
        )
        self._init_server_transfer()

    def capacity_stats(self) -> dict[str, Any]:
        """Return this worker's actual FragPack allocation for vLLM telemetry.

        Returns:
            Capacity snapshots keyed by TP rank, or an empty dictionary for
            raw/external-only storage and before attention creates its store.

        Async/thread-safety:
            Called by the worker's statistics hook after model execution.
            Reads host metadata only; performs no I/O or CUDA synchronization.
        """
        if self._storage_format != STORAGE_FORMAT_FRAGPACK:
            return {}
        capacity = get_capacity()
        if capacity is None:
            return {}
        if capacity is not self._reported_capacity:
            snapshot = capacity.to_dict()
            snapshot["memory"]["connector_staging"] = self._allocated_staging_bytes
            self._capacity_stats = {"capacity": {str(self._tp_rank): snapshot}}
            self._reported_capacity = capacity
        return self._capacity_stats

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.

        Async/thread-safety:
            Called on the vLLM worker thread after the step's state updates and
            before its forward launch, which is where restores are fenced.
            Packed slots the TP ranks stored differently are resolved to
            this rank's mode in place; the metadata object is per worker.
        """
        for spec in connector_metadata.reqs_to_load.values():
            if any(slot.rank_modes for slot in spec.compressed_slots):
                spec.compressed_slots = [
                    slot.for_rank(self._tp_rank) for slot in spec.compressed_slots
                ]
        self._meta = connector_metadata
        self._load_pipeline.begin_step(bool(connector_metadata.reqs_to_load))

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
            the load pipeline thread. A layer-wise load in a step whose
            forward skips the layer hooks blocks here until fully restored.
        """
        del kwargs
        meta = self._meta
        reqs_to_load = dict(meta.reqs_to_load) if meta is not None else {}
        layerwise_ids = meta.layerwise_req_ids if meta is not None else set()
        layerwise = {
            spec_id: spec
            for spec_id, spec in reqs_to_load.items()
            if base_req_id(spec_id) in layerwise_ids
        }
        for spec_id in layerwise:
            del reqs_to_load[spec_id]
        if (layerwise or reqs_to_load) and not self._ensure_transfer_ready():
            # A layer-wise spec is still started so its failure invalidates
            # the synchronously scheduled blocks and releases its lease.
            self._load_pipeline.start_layerwise(layerwise)
            self._load_pipeline.mark_failed(
                reqs_to_load,
                "server transfer config is not ready",
            )
            return
        # Called every step: this also retires the previous layer-wise state.
        self._load_pipeline.start_layerwise(layerwise)
        if layerwise and forward_skips_layer_waits(forward_context):
            # vLLM picks a full graph only for uniform decode batches, so this
            # is a layer-wise hit that left one token to compute: restore the
            # whole step before the replay instead of per layer.
            self._load_pipeline.wait_for_all_layers()
        if reqs_to_load:
            self._load_pipeline.start(reqs_to_load)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Order one layer's attention after this step's layer-wise restore.

        Args:
            layer_name: Attention layer about to run.

        Async/thread-safety:
            Called on the vLLM worker thread inside the forward pass. Returns
            immediately unless the step carries a layer-wise load.
        """
        layer = self._layer_idx_map.get(layer_name)
        if layer is not None:
            self._load_pipeline.wait_for_layer(layer)

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

    def handle_preemptions(self, kv_connector_metadata: DaserConnectorMeta) -> None:
        """Cancel unsent snapshots before their original KV blocks are reused.

        Args:
            kv_connector_metadata: DaserConnectorMeta for the upcoming step.

        Returns:
            None. Writer releases run on the store event loop.

        Async/thread-safety:
            Called on the worker thread before either model execution or
            remote restore can overwrite preempted blocks. This hook also
            runs when vLLM skips forward and therefore skips wait_for_save.
        """
        self._store_pipeline.cancel_pending(
            kv_connector_metadata.cancelled_store_req_ids
        )

    def wait_for_save(self) -> None:
        """Queue stores until vLLM reports request completion."""
        if self._meta is None:
            return
        if self._meta.reqs_to_store:
            self._store_pipeline.queue_finished(self._meta.reqs_to_store)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Collect completed background transfers after a worker step.

        Args:
            finished_req_ids: Request IDs that vLLM finished in this step.

        Returns:
            Finished-saving request IDs and finished async-loading request IDs.

        Async/thread-safety:
            Called on the worker thread while this step's metadata is bound,
            including no-forward steps that skip ``wait_for_save``.
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
        self._stop_transfer_warmup()
        self._load_pipeline.shutdown()
        self._store_pipeline.shutdown()

    def _ensure_transfer_ready(self) -> bool:
        """Refresh config and initialize both pipeline transfer clients.

        The lock makes foreground request hooks and startup warmup mutually
        exclusive. A request that races warmup waits for the one shared
        initialization instead of registering duplicate CUDA IPC handles.
        """
        warmup_error = getattr(self, "_transfer_warmup_error", None)
        if warmup_error is not None:
            raise RuntimeError("background transfer warmup failed") from warmup_error
        lock = getattr(self, "_transfer_init_lock", None)
        if lock is None:
            # Minimal test probes can bypass __init__; keep this helper usable
            # without weakening synchronization for real workers.
            lock = threading.Lock()
            self._transfer_init_lock = lock
        with lock:
            warmup_error = getattr(self, "_transfer_warmup_error", None)
            if warmup_error is not None:
                raise RuntimeError(
                    "background transfer warmup failed"
                ) from warmup_error
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
                    len(getattr(self, "_layer_names", ())),
                )
                self._load_pipeline.configure_rank_geometry(self._tp_rank)
                self._store_pipeline.configure_rank_geometry(
                    self._tp_rank,
                    self._tp_size,
                )
                self._transfer_ready = True
                logger.info("[CONNECTOR] server transfer mode=%s", self._transfer_mode)

            if not self._pipelines_initialized:
                self._set_transfer_cuda_device()
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
        self._skip_l2 = bool(config.get("skip_l2", self._skip_l2))
        self._transfer_mode = str(config.get("transfer_mode", self._transfer_mode))
        server_storage_format = str(config.get("storage_format", STORAGE_FORMAT_RAW))
        if (
            getattr(self, "_declared_storage_format", None) is not None
            and server_storage_format != self._declared_storage_format
        ):
            raise ValueError(
                "connector storage_format does not match DaseR server: "
                f"{self._declared_storage_format} != {server_storage_format}"
            )
        self._storage_format = server_storage_format
        codebooks = config.get("compressed_codebooks", b"")
        if not isinstance(codebooks, bytes):
            raise ValueError("compressed codebooks must be a byte payload")
        tile_scalars = int(config.get("compressed_tile_scalars", ONLINE_TILE_SCALARS))
        if getattr(self, "_compression_configured", False):
            if self._storage_format != STORAGE_FORMAT_COMPRESSED_ONLINE:
                return
            if tile_scalars != self._compression_tile_scalars:
                raise ValueError(
                    "server compressed tile geometry changed after configuration"
                )
            if digest_bytes(codebooks) == self._compression_codebook_hash:
                return
            # Prewarm binds the generic palette. A server started with
            # calibrated codebooks rebinds them here, before the first
            # transfer; the compiled kernels take codebooks as runtime data.
        load_pipeline = getattr(self, "_load_pipeline", None)
        store_pipeline = getattr(self, "_store_pipeline", None)
        if load_pipeline is None or store_pipeline is None:
            return
        load_pipeline.configure_compression(
            storage_format=self._storage_format,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
        )
        store_pipeline.configure_compression(
            storage_format=self._storage_format,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
        )
        self._compression_configured = True
        self._compression_codebook_hash = digest_bytes(codebooks)
        self._compression_tile_scalars = tile_scalars

    def _init_server_transfer(self) -> None:
        """Initialize both pipeline-owned transfer clients.

        Async/thread-safety:
            Called on the worker thread after KV-cache registration. Each
            pipeline performs its initialization on its private event loop.
        """
        if self._ensure_transfer_ready():
            return
        # vLLM may register its KV cache before DaseR has bound its socket. A
        # daemon retry keeps staging registration out of the first request's
        # TTFT while still allowing the normal hook to make progress when the
        # server is deliberately started later.
        if hasattr(self, "_transfer_warmup_stop"):
            self._start_transfer_warmup()

    def _start_transfer_warmup(self) -> None:
        """Start one background retry loop for deferred transfer setup.

        Async/thread-safety:
            The loop performs startup-only sync IPC and CUDA IPC export on a
            dedicated thread. It never handles request metadata or submits
            load/store work.
        """
        thread = self._transfer_warmup_thread
        if thread is not None and thread.is_alive():
            return
        stop = self._transfer_warmup_stop
        stop.clear()
        self._transfer_warmup_error = None
        thread = threading.Thread(
            target=self._transfer_warmup_loop,
            daemon=True,
            name="daser-transfer-warmup",
        )
        self._transfer_warmup_thread = thread
        thread.start()

    def _transfer_warmup_loop(self) -> None:
        """Retry deferred transfer setup until the server is ready or shutdown."""
        stop = self._transfer_warmup_stop
        while not stop.is_set():
            # Checking the filesystem avoids repeated long IPC connect waits
            # while the server process is still starting.
            if not os.path.exists(self._socket_path):
                stop.wait(_TRANSFER_WARMUP_RETRY_S)
                continue
            try:
                if self._ensure_transfer_ready():
                    logger.info("[CONNECTOR] background transfer warmup complete")
                    return
            except ValueError as exc:
                # Geometry and compression mismatches are permanent contract
                # violations; surface them instead of retrying forever.
                self._transfer_warmup_error = exc
                logger.error("[CONNECTOR] background transfer warmup failed: %s", exc)
                return
            except Exception as exc:  # noqa: BLE001
                # Server startup can expose a socket before its IPC handlers
                # are ready. Keep retrying transient transport failures.
                self._transfer_ready = False
                self._pipelines_initialized = False
                logger.info("[CONNECTOR] transfer warmup retry: %s", exc)
            stop.wait(_TRANSFER_WARMUP_RETRY_S)

    def _stop_transfer_warmup(self) -> None:
        """Stop and join the startup transfer warmup thread."""
        stop = getattr(self, "_transfer_warmup_stop", None)
        thread = getattr(self, "_transfer_warmup_thread", None)
        if stop is None or thread is None:
            return
        stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=_TRANSFER_WARMUP_JOIN_TIMEOUT_S)
        self._transfer_warmup_thread = None

    def _set_transfer_cuda_device(self) -> None:
        """Select the registered KV tensor's CUDA device for IPC export."""
        sample = next(iter(getattr(self, "_kv_caches", {}).values()), None)
        if sample is not None and sample.device.type == "cuda":
            torch.cuda.set_device(sample.device)

    def _prewarm_online_compression(self, kv_cache: torch.Tensor) -> None:
        """Construct online pack and restore codecs before serving traffic.

        Args:
            kv_cache: Registered cross-layer BF16 KV tensor.

        Async/thread-safety:
            Called on the vLLM worker thread during cache registration, before
            the API endpoint accepts requests.  The server later supplies the
            same deterministic online codebook over IPC; runtime refresh only
            validates its digest and does not rebuild CUDA modules on a request
            path.
        """
        geometry = CompressedStoreGeometry(
            num_slots=int(kv_cache.shape[0]),
            slot_size=int(kv_cache[0].nbytes),
            block_tokens=int(kv_cache.shape[3]),
            num_layers=int(kv_cache.shape[1]),
            num_kv_heads=int(kv_cache.shape[4]),
            head_dim=int(kv_cache.shape[5]),
            dtype_bytes=int(kv_cache.element_size()),
            tile_scalars=ONLINE_TILE_SCALARS,
        )
        codebooks = default_online_codebooks(geometry)
        self._load_pipeline.configure_compression(
            storage_format=STORAGE_FORMAT_COMPRESSED_ONLINE,
            codebooks=codebooks,
            tile_scalars=geometry.tile_scalars,
        )
        self._store_pipeline.configure_compression(
            storage_format=STORAGE_FORMAT_COMPRESSED_ONLINE,
            codebooks=codebooks,
            tile_scalars=geometry.tile_scalars,
        )
        self._compression_codebook_hash = digest_bytes(codebooks)
        self._compression_tile_scalars = geometry.tile_scalars
        self._compression_configured = True
        logger.info(
            "[CONNECTOR] prewarmed online compression slots=%d tile_scalars=%d",
            int(kv_cache.shape[0]),
            geometry.tile_scalars,
        )

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
            self._staging_budget_bytes,
        )
        load_pool = FixedCudaStagingPool(
            device=sample.device,
            buffer_bytes=staging_bytes,
            depth=load_depth,
        )
        store_pool = FixedCudaStagingPool(
            device=sample.device,
            buffer_bytes=staging_bytes,
            depth=store_depth,
        )
        self._store_pipeline.configure(
            kv_caches=self._kv_caches,
            layer_names=self._layer_names,
            local_slot_size=self._local_slot_size,
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            staging_bytes=staging_bytes,
            staging_pool=store_pool,
        )
        self._load_pipeline.configure(
            kv_caches=self._kv_caches,
            layer_names=self._layer_names,
            local_slot_size=self._local_slot_size,
            tp_rank=self._tp_rank,
            staging_pool=load_pool,
            load_key_scale=self._load_key_scale,
            load_value_scale=self._load_value_scale,
            rope_delta_scale=self._rope_delta_scale,
            rope_base=self._rope_base,
            rope_rotary_dim=self._rope_rotary_dim,
            rope_is_neox_style=self._rope_is_neox_style,
        )
        self._allocated_staging_bytes = (
            load_pool.buffer_bytes * load_pool.depth
            + store_pool.buffer_bytes * store_pool.depth
        )
        self._reported_capacity = None
        logger.info(
            "[CONNECTOR] preallocated staging buffer_bytes=%d total_bytes=%d "
            "load_depth=%d store_depth=%d",
            staging_bytes,
            allocated_bytes,
            load_pool.depth,
            store_pool.depth,
        )
        return load_pool.depth
