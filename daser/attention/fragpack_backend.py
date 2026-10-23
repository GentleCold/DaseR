# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend that keeps the KV cache FragPack-compressed in HBM.

Each paged KV block holds, per layer, one page of
:func:`~daser.ops.hbm_kv.page_bytes` that its ``2 H`` FragPack records share (23.25
KiB per record on average instead of 32 KiB raw), so the same KV memory holds
1.38x as many blocks. :class:`~daser.ops.hbm_kv.HbmKvStore`
manages the records over the engine's cache; decode attention decompresses
records inside the attention kernel
(:func:`~daser.ops.fused_attn_decode.decode_attention`), and prefill decodes
the context pages into a BF16 workspace and runs FlashAttention over it.

Per step: :meth:`FragPackMetadataBuilder.build` calls ``HbmKvStore.prepare``
(eagerly, also before a FULL CUDA-graph replay); per layer the impl writes the
new tokens, attends, and encodes the blocks closed this step; the last layer
commits. The store needs the ids of every block a running request may still
write, which vLLM does not pass to attention backends, so the model runner's
``execute_model`` is wrapped to mirror per-request block lists from the
scheduler output (:class:`OpenBlocks`).

Requirements: V2 model runner, BF16 KV, head size 128, ``H * 8`` query heads,
block size 128, SM90, no speculative decoding, a TP size that divides the KV
heads. Calibrated tables are read from ``DASER_FRAGPACK_TABLES`` (see
:mod:`daser.compression.attn_tables`); each TP rank uses its own heads' tables.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import os
from typing import ClassVar

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

from daser.attention.capacity import FragPackCapacity, publish_capacity
from daser.compression.attn_tables import load_tables
from daser.logging import init_logger
from daser.ops.fused_attn_decode import (
    GQA,
    RecordTables,
    decode_attention,
    decode_units,
    workspace_elems,
)
from daser.ops.hbm_kv import (
    BLOCK_TOKENS,
    HEAD_DIM,
    HbmKvStore,
    page_bytes,
    register_store,
)

logger = init_logger(__name__)

_POLL_EVERY = 16
_LOG_EVERY = 4096
# Raw (incompressible) records; DASER_FRAGPACK_OVERFLOW. Real KV almost never
# falls back (~1 in 10^6 records), but a degenerate prompt (one token
# repeated) falls back ~96%: 8192 holds such a 2048-token prompt (~5.9k
# records) cached alongside real traffic, at 256 MiB.
_OVERFLOW_RECORDS = 8192


class OpenBlocks:
    """Per-request block lists mirrored from scheduler outputs.

    Before a step, the only block of a request that can own a staging slot is
    the one holding its first uncomputed token, if earlier steps wrote part of
    it. :meth:`live_blocks` lists exactly those; the store's GC frees the slot
    of every other block. A superset is not safe: a block freed while open and
    reallocated ahead of its new owner's write position would keep its stale
    slot, so one request would hold two and exhaust staging.
    """

    def __init__(self) -> None:
        self.blocks: dict[str, list[int]] = {}
        self.computed: dict[str, int] = {}

    def update(self, so: object) -> None:
        """Apply one scheduler output (same order as the V2 model runner).

        Args:
            so: ``SchedulerOutput`` of a real (non-dummy) step.

        Returns:
            None.
        """
        for req_id in so.finished_req_ids | (so.preempted_req_ids or set()):
            self.blocks.pop(req_id, None)
            self.computed.pop(req_id, None)
        for new in so.scheduled_new_reqs:
            self.blocks[new.req_id] = list(new.block_ids[0])
            self.computed[new.req_id] = new.num_computed_tokens
        updates = so.block_table_updates or {}
        for req_id, block_ids in updates.items():
            if req_id in self.blocks:
                self.blocks[req_id] = list(block_ids[0])
        cached = so.scheduled_cached_reqs
        for req_id, nct, new_ids in zip(
            cached.req_ids,
            cached.num_computed_tokens,
            cached.new_block_ids,
            strict=True,
        ):
            if req_id not in self.blocks:
                continue
            self.computed[req_id] = nct
            if new_ids is not None and req_id not in updates:
                if req_id in cached.resumed_req_ids:
                    self.blocks[req_id] = list(new_ids[0])
                else:
                    self.blocks[req_id].extend(new_ids[0])

    def live_blocks(self) -> np.ndarray:
        """Return int32 ids of the partially written block of each request.

        Returns:
            One id per request whose computed length is not block aligned.
        """
        open_ids = []
        for req_id, blocks in self.blocks.items():
            index, offset = divmod(self.computed[req_id], BLOCK_TOKENS)
            if offset and index < len(blocks):
                open_ids.append(blocks[index])
        return np.asarray(open_ids, dtype=np.int32)


_TRACKER = OpenBlocks()


@functools.cache
def _install_tracker() -> None:
    """Wrap ``GPUModelRunner.execute_model`` (V2) to feed :data:`_TRACKER`."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.execute_model

    @functools.wraps(original)
    def execute_model(
        self: GPUModelRunner, scheduler_output: object, *args: object, **kwargs: object
    ) -> object:
        if not kwargs.get("dummy_run", False):
            _TRACKER.update(scheduler_output)
        return original(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model


@dataclass
class PrefillGroup:
    """Prefill rows attended in one workspace pass.

    Attributes:
        src_blocks: int32 ``[P]`` blocks decoded into workspace pages ``0..P``.
        block_table: int32 ``[R, max_pages]`` workspace page ids per row.
        cu_seqlens_q: int32 ``[R + 1]`` query offsets relative to ``tok0``.
        seqused_k: int32 ``[R]`` context lengths.
        tok0: First token of the group in the batch.
        tok1: One past the last token.
        max_seqlen_q: Longest query of the group.
        max_seqlen_k: Longest context of the group.
    """

    src_blocks: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    tok0: int
    tok1: int
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class FragPackMetadata:
    """Per-step metadata shared by every layer.

    Attributes:
        store: The KV store.
        layer_tables: Record tables per layer index.
        layer_index: Layer name to index.
        last_layer: Index of the layer that commits the step.
        slot_mapping: int64 slot ids of the batch tokens.
        num_tokens: Tokens with a slot (rows of ``key`` to write).
        num_decodes: Leading single-token rows.
        decode_block_table: int32 ``[num_decodes, max_pages]``.
        decode_seq_lens: int32 ``[num_decodes]``.
        units: Decode CTAs per KV head.
        decode_workspace: ``(opart, ml, plan)`` of the decode kernel; the
            first layer of a step writes the plan, the others reuse it.
        max_close: Grid bound of the per-layer encode.
        prefills: Prefill groups.
        wk: BF16 prefill K workspace ``[W, 128, H, 128]``.
        wv: BF16 prefill V workspace.
    """

    store: HbmKvStore
    layer_tables: list[RecordTables]
    layer_index: dict[str, int]
    last_layer: int
    slot_mapping: torch.Tensor
    num_tokens: int
    num_decodes: int
    decode_block_table: torch.Tensor
    decode_seq_lens: torch.Tensor
    units: int
    decode_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    max_close: int
    prefills: list[PrefillGroup]
    wk: torch.Tensor
    wv: torch.Tensor


class FragPackMetadataBuilder(AttentionMetadataBuilder[FragPackMetadata]):
    """Builds :class:`FragPackMetadata` and owns the store and workspaces.

    Args:
        kv_cache_spec: KV spec of the layer group.
        layer_names: Attention layers of the group.
        vllm_config: Engine config.
        device: CUDA device.

    Async/thread-safety:
        Used from the model runner thread only.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not vllm_config.use_v2_model_runner:
            raise ValueError("FragPack attention needs the V2 model runner")
        if vllm_config.speculative_config is not None:
            raise ValueError("FragPack attention does not support speculative decoding")
        if vllm_config.cache_config.block_size != BLOCK_TOKENS:
            raise ValueError(f"FragPack attention needs --block-size {BLOCK_TOKENS}")
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        _install_tracker()

        sched = vllm_config.scheduler_config
        capture = vllm_config.compilation_config.max_cudagraph_capture_size or 0
        self.max_tokens = max(sched.max_num_batched_tokens, capture)
        self.kv_heads = kv_cache_spec.num_kv_heads
        order = sorted(layer_names, key=extract_layer_index)
        self.layer_names = order
        self.layer_index = {name: i for i, name in enumerate(order)}

        props = torch.cuda.get_device_properties(device)
        self.num_sms = props.multi_processor_count
        self.max_reqs = sched.max_num_seqs
        opart, ml, plan = workspace_elems(
            self.kv_heads,
            decode_units(max(self.max_reqs, capture), self.kv_heads, self.num_sms),
        )
        f32 = dict(dtype=torch.float32, device=device)
        self.decode_workspace = (
            torch.empty(opart, **f32),
            torch.empty(ml, **f32),
            torch.empty(plan, dtype=torch.int32, device=device),
        )
        self.workspace_pages = -(
            -vllm_config.model_config.max_model_len // BLOCK_TOKENS
        )
        ws = (self.workspace_pages, BLOCK_TOKENS, self.kv_heads, HEAD_DIM)
        self.wk = torch.empty(ws, dtype=torch.bfloat16, device=device)
        self.wv = torch.empty(ws, dtype=torch.bfloat16, device=device)

        path = os.environ.get("DASER_FRAGPACK_TABLES")
        if not path:
            raise ValueError(
                "DASER_FRAGPACK_TABLES must point at calibrated FragPack tables (.npz)"
            )
        # vLLM gives TP rank r the KV heads [r * H, (r + 1) * H).
        rank = get_tensor_model_parallel_rank()
        heads = slice(rank * self.kv_heads, (rank + 1) * self.kv_heads)
        self.tables = load_tables(path, device, heads)
        if self.tables.plut.shape[2] != self.kv_heads:
            raise ValueError(
                f"FragPack tables {path} do not cover KV heads "
                f"{heads.start}..{heads.stop}"
            )
        self.store: HbmKvStore | None = None
        self.store_key: tuple[int, int] | None = None
        self.layer_tables: list[RecordTables] = []
        self.steps = 0

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FragPackMetadata:
        """Prepare the store for this step and build the metadata.

        Args:
            common_prefix_len: Unused (no cascade attention).
            common_attn_metadata: Batch metadata.
            fast_build: Unused.

        Returns:
            Metadata for every layer of the step.

        Raises:
            StoreError: If the store dropped KV data since the last poll.

        Async/thread-safety:
            Enqueues the store's prepare on the current stream.
        """
        return self._build(common_attn_metadata, capture=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FragPackMetadata:
        """Build capture metadata without touching store state.

        Args:
            common_attn_metadata: Dummy batch metadata.

        Returns:
            Metadata whose launch shapes match replays of this batch shape.
        """
        return self._build(common_attn_metadata, capture=True)

    def _build(self, cam: CommonAttentionMetadata, capture: bool) -> FragPackMetadata:
        store = self._store()
        num_reqs = cam.num_reqs
        nd, npf, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        qsl = cam.query_start_loc_cpu.numpy().astype(np.int64)
        seq = cam.seq_lens_cpu_upper_bound.numpy().astype(np.int64)[:num_reqs]
        qlen = qsl[1 : num_reqs + 1] - qsl[:num_reqs]
        num_tokens = min(int(qsl[num_reqs]), cam.slot_mapping.shape[0])

        # Blocks receiving their last token: one per crossed 128 boundary.
        closes = int(
            np.sum(seq // BLOCK_TOKENS - np.maximum(seq - qlen, 0) // BLOCK_TOKENS)
        )
        max_close = min(max(closes, nd), self.max_tokens)
        if not capture:
            store.prepare(cam.slot_mapping, num_tokens, _TRACKER.live_blocks())
            self.steps += 1
            if self.steps % _POLL_EVERY == 0:
                stats = store.poll()
                if stats is not None and self.steps % _LOG_EVERY == 0:
                    logger.info("FragPack store: %s", stats)

        prefills: list[PrefillGroup] = []
        if npf and not capture:
            prefills = self._prefill_groups(cam, nd, num_reqs, qsl, seq)
        return FragPackMetadata(
            store=store,
            layer_tables=self.layer_tables,
            layer_index=self.layer_index,
            last_layer=len(self.layer_names) - 1,
            slot_mapping=cam.slot_mapping,
            num_tokens=num_tokens,
            num_decodes=nd,
            decode_block_table=cam.block_table_tensor[:nd],
            decode_seq_lens=cam.seq_lens[:nd],
            units=decode_units(max(nd, 1), self.kv_heads, self.num_sms),
            decode_workspace=self.decode_workspace,
            max_close=max_close,
            prefills=prefills,
            wk=self.wk,
            wv=self.wv,
        )

    def _store(self) -> HbmKvStore:
        layers = get_layers_from_vllm_config(
            self.vllm_config, Attention, self.layer_names
        )
        caches = [layers[name].kv_cache for name in self.layer_names]
        if caches[0].numel() == 0:
            raise RuntimeError(
                "FragPack attention built metadata before the KV cache was bound"
            )
        key = (caches[0].data_ptr(), caches[0].shape[0])
        if self.store is None or key != self.store_key:
            # Every running request holds at most one block open across steps.
            self.store = HbmKvStore(
                caches,
                self.kv_heads,
                self.tables,
                self.max_tokens,
                staging_slots=self.max_reqs,
                overflow_records=int(
                    os.environ.get("DASER_FRAGPACK_OVERFLOW", _OVERFLOW_RECORDS)
                ),
            )
            self.store_key = key
            register_store(self.store)
            self.layer_tables = [
                RecordTables(*self.store.layer_tables(i)) for i in range(len(caches))
            ]
            # Capture sizes once at allocation, including auxiliary memory.
            # The connector forwards this host-only snapshot through vLLM's
            # public statistics channel; no telemetry RPC runs in attention.
            memory = self.store.memory_usage()
            memory["prefill_workspace"] = sum(
                t.numel() * t.element_size() for t in (self.wk, self.wv)
            )
            memory["decode_workspace"] = sum(
                t.numel() * t.element_size() for t in self.decode_workspace
            )
            publish_capacity(
                FragPackCapacity(
                    blocks=self.store.num_blocks,
                    block_tokens=BLOCK_TOKENS,
                    layers=self.store.num_layers,
                    kv_heads=self.kv_heads,
                    page_bytes=self.store.arena_stride[0],
                    raw_page_bytes=2 * self.kv_heads * BLOCK_TOKENS * HEAD_DIM * 2,
                    memory=memory,
                )
            )
            logger.info(
                "FragPack store: %d layers x %d blocks, %d B/block-layer (raw %d), "
                "%d staging slots, %d overflow records, %.0f MiB staging+scratch"
                "+overflow, %.0f MiB prefill workspace",
                len(caches),
                self.store.num_blocks,
                page_bytes(self.kv_heads),
                2 * self.kv_heads * BLOCK_TOKENS * HEAD_DIM * 2,
                self.store.staging.shape[0],
                self.store.overflow.shape[0],
                (
                    self.store.staging.numel()
                    + self.store.scratch.numel()
                    + self.store.overflow.numel()
                )
                / 2**20,
                2 * self.wk.numel() * self.wk.element_size() / 2**20,
            )
        return self.store

    def _prefill_groups(
        self,
        cam: CommonAttentionMetadata,
        first: int,
        num_reqs: int,
        qsl: np.ndarray,
        seq: np.ndarray,
    ) -> list[PrefillGroup]:
        pages = -(-seq // BLOCK_TOKENS)
        groups: list[PrefillGroup] = []
        start, used = first, 0
        for r in range(first, num_reqs):
            if used + pages[r] > self.workspace_pages and r > start:
                groups.append(self._prefill_group(cam, start, r, qsl, seq, pages))
                start, used = r, 0
            used += int(pages[r])
        groups.append(self._prefill_group(cam, start, num_reqs, qsl, seq, pages))
        return groups

    def _prefill_group(
        self,
        cam: CommonAttentionMetadata,
        r0: int,
        r1: int,
        qsl: np.ndarray,
        seq: np.ndarray,
        pages: np.ndarray,
    ) -> PrefillGroup:
        n = r1 - r0
        pg = pages[r0:r1]
        offs = np.concatenate(([0], np.cumsum(pg)))
        total, width = int(offs[-1]), int(pg.max())
        rows = np.repeat(np.arange(r0, r1), pg)
        cols = np.arange(total) - np.repeat(offs[:-1], pg)
        table = np.zeros((n, width), dtype=np.int64)
        for i in range(n):
            table[i, : pg[i]] = np.arange(offs[i], offs[i + 1])
        host = np.concatenate(
            (rows, cols, table.ravel(), qsl[r0 : r1 + 1] - qsl[r0], seq[r0:r1])
        )
        dev = torch.from_numpy(host).pin_memory().to(self.device, non_blocking=True)
        a, b, c, d = (
            total,
            2 * total,
            2 * total + n * width,
            2 * total + n * width + n + 1,
        )
        src = cam.block_table_tensor[dev[:a], dev[a:b]].to(torch.int32)
        return PrefillGroup(
            src_blocks=src,
            block_table=dev[b:c].view(n, width).to(torch.int32),
            cu_seqlens_q=dev[c:d].to(torch.int32),
            seqused_k=dev[d:].to(torch.int32),
            tok0=int(qsl[r0]),
            tok1=int(qsl[r1]),
            max_seqlen_q=int((qsl[r0 + 1 : r1 + 1] - qsl[r0:r1]).max()),
            max_seqlen_k=int(seq[r0:r1].max()),
        )


class FragPackAttentionImpl(AttentionImpl[FragPackMetadata]):
    """Decoder self-attention over the FragPack HBM store.

    Args:
        num_heads: Query heads.
        head_size: Head size (128).
        scale: Softmax scale.
        num_kv_heads: KV heads (``num_heads / 8``).
        alibi_slopes: Must be None.
        sliding_window: Must be None.
        kv_cache_dtype: ``auto`` or ``bfloat16``.
        logits_soft_cap: Must be None or 0.
        attn_type: Must be decoder.
        kv_sharing_target_layer_name: Must be None.
        **kwargs: Ignored.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs: object,
    ) -> None:
        num_kv_heads = num_kv_heads or num_heads
        if num_heads != GQA * num_kv_heads or head_size != HEAD_DIM:
            raise ValueError(
                f"FragPack attention needs {GQA}x GQA and head size {HEAD_DIM}"
            )
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise ValueError("FragPack attention supports plain causal attention only")
        if (
            attn_type != AttentionType.DECODER
            or kv_sharing_target_layer_name is not None
        ):
            raise ValueError(
                "FragPack attention supports unshared decoder self-attention only"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.fa_version = get_flash_attn_version()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FragPackMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write this layer's new KV, attend, and encode closed blocks.

        Args:
            layer: The attention layer.
            query: BF16 ``[T, Hq, 128]``.
            key: BF16 ``[T, H, 128]``.
            value: BF16 ``[T, H, 128]``.
            kv_cache: Unused (the store addresses the cache directly).
            attn_metadata: Step metadata, None in profiling runs.
            output: BF16 ``[T, Hq, 128]`` output buffer.
            output_scale: Unsupported.
            output_block_scale: Unsupported.

        Returns:
            ``output``.

        Async/thread-safety:
            Enqueues kernels on the current stream; CUDA-graph safe for
            decode-only batches.
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "FragPack attention does not fuse output quantization"
            )
        md = attn_metadata
        if md is None:
            return output.zero_()
        li = md.layer_index[layer.layer_name]
        store = md.store
        n = md.num_tokens
        store.write(li, key[:n], value[:n], md.slot_mapping[:n])
        nd = md.num_decodes
        if nd:
            decode_attention(
                query[:nd],
                md.layer_tables[li],
                md.decode_block_table,
                md.decode_seq_lens,
                self.scale,
                md.units,
                output[:nd],
                md.decode_workspace,
                replan=li == 0,
            )
        for g in md.prefills:
            store.gather(li, g.src_blocks, md.wk, md.wv)
            flash_attn_varlen_func(
                q=query[g.tok0 : g.tok1],
                k=md.wk,
                v=md.wv,
                out=output[g.tok0 : g.tok1],
                cu_seqlens_q=g.cu_seqlens_q,
                max_seqlen_q=g.max_seqlen_q,
                seqused_k=g.seqused_k,
                max_seqlen_k=g.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=g.block_table,
                fa_version=self.fa_version,
            )
        store.encode(li, md.max_close)
        if li == md.last_layer:
            store.commit()
        return output


class FragPackAttentionBackend(AttentionBackend):
    """FragPack-compressed HBM KV cache backend (``--attention-backend CUSTOM``)."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Return the only block size the kernels handle (128)."""
        return [BLOCK_TOKENS]

    @staticmethod
    def get_name() -> str:
        """Return the registry name."""
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[FragPackAttentionImpl]:
        """Return the impl class."""
        return FragPackAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FragPackMetadataBuilder]:
        """Return the metadata builder class."""
        return FragPackMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """Return the supported head sizes."""
        return [HEAD_DIM]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Size a block as one FragPack record page per layer.

        Args:
            spec: Dense KV spec.

        Returns:
            Spec with one head slot of ``page_bytes(H) / 128`` bytes per token.
        """
        if spec.state_content_bytes is not None:
            return spec
        return replace(
            spec,
            num_head_slots=1,
            state_content_bytes=page_bytes(spec.num_kv_heads) // BLOCK_TOKENS,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        """Return the layer-major layout (each layer's blocks contiguous)."""
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def supports_compute_capability(cls, capability: object) -> bool:
        """Return True on SM90 (TMA bulk copies, FA3 prefill)."""
        return capability.major == 9


__all__ = [
    "FragPackAttentionBackend",
    "FragPackAttentionImpl",
    "FragPackMetadata",
    "FragPackMetadataBuilder",
    "PrefillGroup",
]
