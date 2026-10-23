# SPDX-License-Identifier: Apache-2.0
"""HBM-resident FragPack KV store over a paged KV cache.

The paged KV cache allocated by the serving engine is used as the arena: row
``b`` of layer ``l`` is one page shared by the block's ``2 H`` records, packed
back to back in encode order (each record is its dense part plus a variable
part of at most ``VAR_CAP``). :func:`page_bytes` sizes the page for an
average variable part of ``VAR_SHARE``, so the cache needs ``2 H * (DENSE +
VAR_SHARE)`` per block-layer instead of the raw ``2 H * 32768``. A record is
located through per-layer tables ``rec_addr[L, NB, 2, H]`` (absolute device
address) and ``rec_len`` (bytes past the dense part; raw records use
``RAW_VAR_LEN``), so a record can live in its block's page, in staging while
its block is still filling, or raw in a small overflow pool when it does not
code or its page is full.

A block is *open* from its first token until its last (offset 127). Open
blocks hold raw BF16 records in kernel order in a staging slot:

* persistent slots ``[S_p, L, 2, H, 32768]`` for blocks that stay open across
  steps (decode tails, chunked prefill boundaries);
* transient scratch ``[S_t, 2, H, 32768]`` for blocks that open and close in
  the same step; one layer's worth, reused layer after layer in stream order.

Per step, :meth:`HbmKvStore.prepare` (once, before the forward) releases the
staging of blocks no request references any more, claims staging for blocks
receiving offset 0, lists the blocks receiving offset 127, and points claimed
blocks' records at their staging. Per layer, :meth:`write` stores the new
tokens, attention reads through the tables, and :meth:`encode` codes the
closed blocks into their arena pages. :meth:`commit` (after the last layer)
releases the closed blocks' staging. Every launch shape depends only on host
values fixed per batch shape, so the per-layer ops are CUDA-graph safe.

External tiers move closed blocks as block blobs (``attn_page.BLOB_MAGIC``):
:meth:`HbmKvStore.export_blocks` copies a block's records out, and
:meth:`HbmKvStore.import_blocks` installs them into another block on any
stream. Overflow slots are a bitmap, so imports may allocate them while the
model stream encodes. The serving engine's attention backend creates the
store; :func:`register_store` / :func:`active_store` hand it to the KV
connector without either importing the other.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from daser.compression import attn_page as ap
from daser.compression.attn_tables import DeviceTables
from daser.logging import init_logger
from daser.ops.fragpack_ext import extension

logger = init_logger(__name__)

BLOCK_TOKENS = 128
HEAD_DIM = 128
# Page budget per record past the dense part. Calibrated KV averages ~550 B of
# variable part (VAR_CAP 2048); at 1280 a block-layer page spills none of a
# real-trace sample, at 1024 about 2% of records.
VAR_SHARE = 1280
RAW_RECORD = 32768
RAW_VAR_LEN = RAW_RECORD - ap.DENSE_BYTES  # rec_len of a raw record

# Counter slots, mirrored from hbm_kv.cu.
_ERR_STAGING = 4
_ERR_OVERFLOW = 5
_FALLBACKS = 6
_ENCODED = 7
_SPILLS = 8
_N_COUNTERS = 16
_N_TAGS = 4
# Import status bits, mirrored from hbm_kv.cu.
_BLOB_FAILURES = {1: "bad header", 2: "bad lengths", 4: "overflow full"}

_ACTIVE: HbmKvStore | None = None


def page_bytes(kv_heads: int) -> int:
    """Arena page bytes per block-layer for ``kv_heads`` KV heads.

    Args:
        kv_heads: KV heads ``H``.

    Returns:
        ``2 H * (DENSE_BYTES + VAR_SHARE)``.

    Async/thread-safety:
        Pure function.
    """
    return 2 * kv_heads * (ap.DENSE_BYTES + VAR_SHARE)


@dataclass(frozen=True)
class StoreStats:
    """Cumulative store counters.

    Attributes:
        encoded: Records encoded (all layers).
        fallbacks: Records stored raw in the overflow pool (includes spills).
        spills: Coded records stored raw because their block-layer page was
            full.
        err_staging: Claims dropped because staging was exhausted.
        err_overflow: Raw records dropped because the overflow pool was full.
    """

    encoded: int
    fallbacks: int
    spills: int
    err_staging: int
    err_overflow: int


class StoreError(RuntimeError):
    """Raised when the store dropped KV data (staging or overflow exhausted)."""


def register_store(store: HbmKvStore) -> None:
    """Publish the process's store to the KV connector.

    Args:
        store: The store attention now reads; replaces any earlier one.

    Returns:
        None.

    Async/thread-safety:
        Call from the forward thread when the store is (re)created.
    """
    global _ACTIVE
    _ACTIVE = store


def active_store() -> HbmKvStore:
    """Return the store published by :func:`register_store`.

    Returns:
        The current store.

    Raises:
        RuntimeError: If attention has not created one yet (the FragPack backend
            creates it on its first metadata build).

    Async/thread-safety:
        Lock-free read; the reference is replaced only on KV re-binding.
    """
    if _ACTIVE is None:
        raise RuntimeError(
            "no FragPack KV store: the FragPack attention backend is not active"
        )
    return _ACTIVE


def blob_failures(status: int) -> str:
    """Describe an import status word.

    Args:
        status: Nonzero bit set returned for one block by an import.

    Returns:
        Comma-separated failure names.

    Async/thread-safety:
        Pure function.
    """
    return ", ".join(name for bit, name in _BLOB_FAILURES.items() if status & bit)


class HbmKvStore:
    """FragPack KV store whose arena is the engine's paged KV cache.

    Args:
        kv_caches: One uint8-viewable tensor per layer, in layer order. The
            page of block ``b`` at layer ``l`` is row ``b``: it starts at
            ``kv_caches[l].data_ptr() + b * stride`` and spans ``stride =
            kv_caches[l].stride(0) * element_size`` bytes (normally
            :func:`page_bytes`; a 16-byte multiple).
        kv_heads: KV heads ``H``.
        tables: Calibrated lookup tables for every layer.
        max_tokens: Most tokens in one step (sizes the per-step lists and
            transient scratch).
        staging_slots: Persistent staging slots ``S_p``, i.e. blocks that may
            stay open across steps at once (at least one per running request).
        overflow_records: Raw records that may be held in the overflow pool.

    Async/thread-safety:
        All methods enqueue on the current CUDA stream and must be called from
        the thread that drives the forward pass.
    """

    def __init__(
        self,
        kv_caches: list[torch.Tensor],
        kv_heads: int,
        tables: DeviceTables,
        max_tokens: int,
        staging_slots: int,
        overflow_records: int,
    ) -> None:
        dev = kv_caches[0].device
        self.num_layers = len(kv_caches)
        self.kv_heads = kv_heads
        self.num_blocks = kv_caches[0].shape[0]
        self.tables = tables
        if tables.rank_of.shape[:3] != (self.num_layers, 2, kv_heads):
            raise ValueError(
                f"tables {tuple(tables.rank_of.shape)} do not match "
                f"L={self.num_layers}, H={kv_heads}"
            )
        self.arena_base: list[int] = []
        self.arena_stride: list[int] = []
        # Identifies the codec tables, so blobs coded with other tables are
        # rejected on import.
        self.table_digest = int.from_bytes(
            hashlib.blake2b(
                tables.rank_of.cpu().numpy().tobytes(), digest_size=8
            ).digest(),
            "little",
            signed=True,
        )
        for cache in kv_caches:
            stride = cache.stride(0) * cache.element_size()
            if cache.shape[0] != self.num_blocks or stride % 16:
                raise ValueError(
                    f"cache {tuple(cache.shape)} stride {stride} is not a "
                    "16-byte multiple"
                )
            self.arena_base.append(cache.data_ptr())
            self.arena_stride.append(stride)

        L, NB, H = self.num_layers, self.num_blocks, kv_heads
        i32 = dict(dtype=torch.int32, device=dev)
        transient = max_tokens // BLOCK_TOKENS + 2
        self.counters = torch.zeros(_N_COUNTERS, **i32)
        self.counters[0] = staging_slots
        self.wslot = torch.full((NB,), -1, **i32)
        self.tags = torch.zeros((_N_TAGS, NB), **i32)
        self.free_stack = torch.arange(staging_slots, **i32)
        # One bit per overflow slot (set = in use); the tail past
        # ``overflow_records`` is preset so it is never handed out.
        words = -(-overflow_records // 32)
        bits = np.zeros(words * 32, dtype=bool)
        bits[overflow_records:] = True
        self.ovf_bits = torch.from_numpy(
            np.packbits(bits, bitorder="little").view(np.int32).copy()
        ).to(dev)
        self.claim_list = torch.zeros(max_tokens, **i32)
        self.close_list = torch.zeros(max_tokens, **i32)
        # Until a block is first claimed its records read one all-zero raw
        # record, so attention over a never-written block (the engine's null
        # block, dummy runs) stays in bounds.
        self.null_record = torch.zeros(RAW_RECORD, dtype=torch.uint8, device=dev)
        self.rec_addr = torch.full(
            (L, NB, 2, H), self.null_record.data_ptr(), dtype=torch.int64, device=dev
        )
        self.rec_len = torch.full((L, NB, 2, H), RAW_VAR_LEN, **i32)
        self.staging = torch.zeros(
            (staging_slots, L, 2, H, RAW_RECORD), dtype=torch.uint8, device=dev
        )
        self.scratch = torch.zeros(
            (transient, 2, H, RAW_RECORD), dtype=torch.uint8, device=dev
        )
        self.overflow = torch.zeros(
            (overflow_records, RAW_RECORD), dtype=torch.uint8, device=dev
        )
        self.fill = torch.zeros((L, NB), **i32)
        self.live = torch.zeros(NB, **i32)
        i64 = dict(dtype=torch.int64, device=dev)
        self.arena_base_dev = torch.tensor(self.arena_base, **i64)
        self.arena_stride_dev = torch.tensor(self.arena_stride, **i64)
        self._poll_host = torch.zeros(_N_COUNTERS, dtype=torch.int32, pin_memory=True)
        self._poll_event: torch.cuda.Event | None = None
        self._step = 0
        self._state = (
            self.counters,
            self.wslot,
            self.tags,
            self.free_stack,
            self.ovf_bits,
            self.claim_list,
            self.close_list,
            self.rec_addr,
            self.rec_len,
            self.staging,
            self.scratch,
            self.overflow,
            self.fill,
        )

    def memory_usage(self) -> dict[str, int]:
        """Report persistent GPU tensor bytes by allocation component.

        Returns:
            Byte counts for the arena, staging, scratch, overflow, metadata
            and codebooks. Arena rows use their actual strides. Attention
            and connector workspaces are owned and reported by their callers.

        Async/thread-safety:
            Reads tensor shapes only, with no CUDA synchronization or I/O.
            Call after construction on the owning worker thread. The values
            are allocation sizes, not live record occupancy or allocator use.
        """
        metadata = (
            self.counters,
            self.wslot,
            self.tags,
            self.free_stack,
            self.ovf_bits,
            self.claim_list,
            self.close_list,
            self.null_record,
            self.rec_addr,
            self.rec_len,
            self.fill,
            self.live,
            self.arena_base_dev,
            self.arena_stride_dev,
        )
        return {
            "arena": self.num_blocks * sum(self.arena_stride),
            "staging": self.staging.numel() * self.staging.element_size(),
            "scratch": self.scratch.numel() * self.scratch.element_size(),
            "overflow": self.overflow.numel() * self.overflow.element_size(),
            "metadata": sum(t.numel() * t.element_size() for t in metadata),
            "codebooks": sum(
                t.numel() * t.element_size()
                for t in (self.tables.rank_of, self.tables.plut, self.tables.slut)
            ),
        }

    def prepare(
        self, slot_mapping: torch.Tensor, num_tokens: int, live_blocks: np.ndarray
    ) -> None:
        """Start a step: GC, claim staging, list closing blocks.

        Args:
            slot_mapping: int64 ``[>= num_tokens]`` slot ids (negative = pad).
            num_tokens: Tokens to scan.
            live_blocks: int32 ids of the blocks that keep their staging
                slot: the partially written block of each running request.
                Every other block's slot is freed.

        Returns:
            None.

        Async/thread-safety:
            Enqueues one host-to-device copy and one kernel. Call once per
            step before the first :meth:`write`, outside CUDA graph capture.
        """
        self._step = self._step % 0x7FFFFFFE + 1
        n_live = int(live_blocks.size)
        if n_live:
            # A fresh pinned buffer per copy (torch's caching host allocator
            # recycles it only after the copy completes).
            host = torch.from_numpy(
                np.ascontiguousarray(live_blocks, dtype=np.int32)
            ).pin_memory()
            self.live[:n_live].copy_(host, non_blocking=True)
        extension().hbm_prepare(
            *self._state, slot_mapping, num_tokens, self.live, n_live, self._step
        )

    def write(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store one layer's new K / V tokens into their blocks' staging.

        Args:
            layer: Layer index.
            key: BF16 ``[N, H, 128]`` (rows / heads may be strided).
            value: BF16 ``[N, H, 128]``.
            slot_mapping: int64 ``[N]`` slot ids (negative = skip).

        Returns:
            None.

        Async/thread-safety:
            Enqueues one kernel; CUDA-graph safe.
        """
        extension().hbm_write(*self._state, key, value, slot_mapping, layer)

    def encode(self, layer: int, max_close: int) -> None:
        """Code this layer's records of every block closed this step.

        Args:
            layer: Layer index.
            max_close: Upper bound on blocks closed this step (grid size).

        Returns:
            None.

        Async/thread-safety:
            Enqueues one kernel; CUDA-graph safe for a fixed ``max_close``.
        """
        extension().hbm_encode(
            *self._state,
            self.tables.rank_of,
            layer,
            self.arena_base[layer],
            self.arena_stride[layer],
            max_close,
        )

    def commit(self) -> None:
        """Release the staging of blocks closed this step (after the last layer).

        Returns:
            None.

        Async/thread-safety:
            Enqueues one kernel; CUDA-graph safe.
        """
        extension().hbm_commit(*self._state)

    def gather(
        self, layer: int, blocks: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor
    ) -> None:
        """Decode blocks of one layer into token-major BF16 pages.

        Args:
            layer: Layer index.
            blocks: int32 ``[P]`` block ids.
            wk: BF16 ``[>= P, 128, H, 128]`` K pages (page ``i`` = ``blocks[i]``).
            wv: BF16 ``[>= P, 128, H, 128]`` V pages.

        Returns:
            None.

        Async/thread-safety:
            Enqueues one kernel.
        """
        extension().hbm_gather(
            blocks,
            self.rec_addr[layer],
            self.rec_len[layer],
            self.tables.plut[layer],
            self.tables.slut[layer],
            wk,
            wv,
            self.kv_heads,
        )

    def layer_tables(
        self, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rec_addr, rec_len, plut, slut)`` views of one layer.

        Args:
            layer: Layer index.

        Returns:
            Views shaped ``[NB, 2, H]``, ``[NB, 2, H]``, ``[2, H, 2]``, ``[2, H, 16]``.

        Async/thread-safety:
            Pure view creation.
        """
        return (
            self.rec_addr[layer],
            self.rec_len[layer],
            self.tables.plut[layer],
            self.tables.slut[layer],
        )

    @property
    def blob_stride(self) -> int:
        """Return the bytes reserved per block blob (all records raw)."""
        return ap.blob_slot_bytes(self.num_layers, self.kv_heads)

    def export_blocks(
        self, blocks: torch.Tensor, out: torch.Tensor, stride: int
    ) -> torch.Tensor:
        """Copy closed blocks' records into block blobs.

        Args:
            blocks: int32 ``[n]`` block ids, each closed (all 128 tokens
                encoded) and not reclaimed until the copy completes.
            out: Contiguous 16-byte aligned uint8 buffer; blob ``i`` is
                written at ``i * stride``.
            stride: Bytes between blobs, at least :attr:`blob_stride` and a
                16-byte multiple.

        Returns:
            int64 ``[n]`` device tensor of blob lengths, each rounded up to
            ``attn_page.BLOB_ALIGN``.

        Async/thread-safety:
            Enqueues one kernel on the current stream, which must be ordered
            after the forward that closed the blocks.
        """
        lengths = torch.empty(blocks.numel(), dtype=torch.int64, device=out.device)
        extension().hbm_export(
            blocks,
            self.rec_addr,
            self.rec_len,
            out,
            stride,
            lengths,
            self.table_digest,
        )
        return lengths

    def import_blocks(
        self, src: torch.Tensor, offsets: torch.Tensor, blocks: torch.Tensor
    ) -> torch.Tensor:
        """Install block blobs into blocks of this store.

        Coded records are packed into each block's arena pages and raw ones
        are placed in overflow slots; records the blocks held in overflow are
        freed only after that, so a failed layer keeps tables that own their
        slots. A blob that fails validation (magic, geometry, codec tables,
        lengths) or finds the overflow pool full sets its status and leaves
        the affected layers' tables unchanged.

        Args:
            src: Contiguous 16-byte aligned uint8 buffer holding the blobs.
            offsets: int64 ``[n]`` byte offsets of the blobs in ``src``.
            blocks: int32 ``[n]`` distinct destination blocks, none open or
                read by attention until the import completes.

        Returns:
            int32 ``[n]`` device status, zero for an installed block (see
            :func:`blob_failures`). A failed block must be recomputed.

        Async/thread-safety:
            Enqueues one kernel on the current stream; may run on a stream
            other than the forward's, since overflow slots are allocated
            atomically and the blocks are disjoint from the forward's.
        """
        status = torch.zeros(blocks.numel(), dtype=torch.int32, device=src.device)
        extension().hbm_import(
            *self._state,
            src,
            offsets,
            blocks,
            status,
            self.arena_base_dev,
            self.arena_stride_dev,
            self.table_digest,
        )
        return status

    def poll(self) -> StoreStats | None:
        """Asynchronously snapshot the counters; raise if data was dropped.

        Each call reads the snapshot taken by the previous call (if its copy
        has finished) and enqueues a new one, so it never blocks the host.

        Returns:
            The previous snapshot's stats, or None if none is ready.

        Raises:
            StoreError: If staging or the overflow pool was exhausted.

        Async/thread-safety:
            Non-blocking; call from the forward thread outside graph capture.
        """
        stats = None
        if self._poll_event is not None:
            if not self._poll_event.query():
                return None
            stats = self._stats_from(self._poll_host.numpy())
        self._poll_host.copy_(self.counters, non_blocking=True)
        self._poll_event = torch.cuda.Event()
        self._poll_event.record()
        return stats

    def stats(self) -> StoreStats:
        """Synchronously read the counters.

        Returns:
            Current stats.

        Raises:
            StoreError: If staging or the overflow pool was exhausted.

        Async/thread-safety:
            Blocks on the current stream; for tests and shutdown only.
        """
        return self._stats_from(self.counters.cpu().numpy())

    def _stats_from(self, c: np.ndarray) -> StoreStats:
        stats = StoreStats(
            encoded=int(c[_ENCODED]),
            fallbacks=int(c[_FALLBACKS]),
            spills=int(c[_SPILLS]),
            err_staging=int(c[_ERR_STAGING]),
            err_overflow=int(c[_ERR_OVERFLOW]),
        )
        if stats.err_staging or stats.err_overflow:
            raise StoreError(
                f"FragPack store dropped KV: staging={stats.err_staging} "
                f"overflow={stats.err_overflow} "
                "(staging must cover max_num_seqs; raise DASER_FRAGPACK_OVERFLOW)"
            )
        return stats


__all__ = [
    "BLOCK_TOKENS",
    "HEAD_DIM",
    "RAW_RECORD",
    "VAR_SHARE",
    "HbmKvStore",
    "StoreError",
    "StoreStats",
    "active_store",
    "blob_failures",
    "page_bytes",
    "register_store",
]
