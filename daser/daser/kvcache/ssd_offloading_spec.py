from __future__ import annotations

import os
from collections import OrderedDict
import ctypes

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager, PrepareStoreOutput
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec

from .ssd_index import SSDIndex, SSDIndexMeta

logger = init_logger(__name__)


def _kv_print(msg: str) -> None:
    # Print-based observability requested by user.
    # Can be disabled by setting DASER_KVCACHE_PRINT=0.
    # print(msg, flush=True)
    pass


@dataclass
class SSDLoadStoreSpec(LoadStoreSpec):
    """Load/store spec for SSD-resident KV blocks.

    slot_ids are stable identifiers into a fixed-size slot file.
    """

    slot_ids: np.ndarray
    block_bytes: int
    data_file: str

    @staticmethod
    def medium() -> str:
        return "SSD"


class SSDBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("slot_id", ctypes.c_int64)]  # type: ignore

    def __init__(self, slot_id: int):
        super().__init__()
        self.slot_id = int(slot_id)


class SSDBackend(Backend):
    """Backend that allocates fixed-size slots within a single file."""

    def __init__(self, *, cache_dir: Path, block_bytes: int, max_blocks: int):
        super().__init__(block_size=block_bytes, medium=SSDLoadStoreSpec.medium())
        self.cache_dir = cache_dir
        self.block_bytes = block_bytes
        self.max_blocks = max_blocks

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.cache_dir / "kv_blocks.bin"
        # File-backed index (no sqlite). Stored under cache_dir/index/.
        self.index = SSDIndex(self.cache_dir / "index", SSDIndexMeta(block_bytes=block_bytes))

        # Ensure data file exists and is large enough for max_blocks.
        if not self.data_file.exists():
            self.data_file.write_bytes(b"")
        target_size = self.block_bytes * self.max_blocks
        current_size = self.data_file.stat().st_size
        if current_size < target_size:
            with open(self.data_file, "r+b") as f:
                f.truncate(target_size)

        # Slot allocator state
        self._next_fresh_slot = self._infer_next_fresh_slot()
        self._free_slots: set[int] = set()

    def _infer_next_fresh_slot(self) -> int:
        # Best-effort: infer upper bound from existing mappings and free slots.
        max_slot = -1
        for _bh, slot_id in self.index.iter_ready():
            if slot_id > max_slot:
                max_slot = slot_id
        return max_slot + 1

    def close(self) -> None:
        self.index.close()

    def get_num_free_blocks(self):
        # free slots + remaining fresh slots
        # NOTE: We currently approximate and only account for remaining fresh
        # slots. The manager will handle 'cannot store' by returning None.
        return self.max_blocks - self._next_fresh_slot

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        n = len(block_hashes)
        if n == 0:
            return []

        # First reuse from in-memory free slots (we do not persist this).
        reused: list[int] = []
        while self._free_slots and len(reused) < n:
            reused.append(self._free_slots.pop())
        fresh_needed = n - len(reused)

        if self._next_fresh_slot + fresh_needed > self.max_blocks:
            raise RuntimeError(
                f"SSDBackend out of slots: requested={n}, next_fresh={self._next_fresh_slot}, max={self.max_blocks}"
            )

        fresh = list(range(self._next_fresh_slot, self._next_fresh_slot + fresh_needed))
        self._next_fresh_slot += fresh_needed

        slot_ids = reused + fresh

        blocks: list[BlockStatus] = []
        for bh, slot_id in zip(block_hashes, slot_ids):
            blocks.append(SSDBlockStatus(slot_id))
        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, SSDBlockStatus)
        self._free_slots.add(int(block.slot_id))

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        slot_ids: list[int] = []
        for block in blocks:
            assert isinstance(block, SSDBlockStatus)
            slot_ids.append(int(block.slot_id))
        return SSDLoadStoreSpec(
            slot_ids=np.array(slot_ids, dtype=np.int64),
            block_bytes=self.block_bytes,
            data_file=str(self.data_file),
        )


class PersistentLRUOffloadingManager(OffloadingManager):
    """LRU manager with persistence via SSDIndex.

    This is intentionally minimal (no ARC yet) to keep correctness simple.
    """

    def __init__(self, backend: SSDBackend, enable_events: bool = False):
        self.backend = backend
        self.blocks: OrderedDict[BlockHash, SSDBlockStatus] = OrderedDict()
        self.enable_events = enable_events

        # Warm-load ready mappings from disk.
        for bh_bytes, slot_id in self.backend.index.iter_ready():
            bh = BlockHash(bh_bytes)
            st = SSDBlockStatus(slot_id)
            st.ref_cnt = 0
            self.blocks[bh] = st

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for bh in block_hashes:
            st = self.blocks.get(bh)
            if st is None or not st.is_ready:
                break
            hit_count += 1
        if hit_count > 0:
            _kv_print(f"[SSDKV] HIT prefix_blocks={hit_count}")
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for bh in block_hashes:
            st = self.blocks[bh]
            assert st.is_ready
            st.ref_cnt += 1
            blocks.append(st)
        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for bh in reversed(list(block_hashes)):
            if self.blocks.get(bh):
                self.blocks.move_to_end(bh)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for bh in block_hashes:
            st = self.blocks[bh]
            assert st.ref_cnt > 0
            st.ref_cnt -= 1

    def prepare_store(self, block_hashes: Iterable[BlockHash]) -> PrepareStoreOutput | None:
        # filter blocks already stored
        to_store = [bh for bh in block_hashes if bh not in self.blocks]
        if not to_store:
            # Return an empty spec
            empty = SSDLoadStoreSpec(slot_ids=np.array([], dtype=np.int64), block_bytes=self.backend.block_bytes, data_file=str(self.backend.data_file))
            return PrepareStoreOutput(block_hashes_to_store=[], store_spec=empty, block_hashes_evicted=[])

        # Evict if needed: only evict blocks not in use.
        # Instead, attempt allocation and evict until it succeeds.
        to_evict: list[BlockHash] = []
        while True:
            try:
                allocated = self.backend.allocate_blocks(to_store)
                break
            except RuntimeError:
                # evict one LRU ready block with ref_cnt==0
                evicted_one = False
                for bh, st in self.blocks.items():
                    if st.ref_cnt == 0:
                        to_evict.append(bh)
                        self.backend.free(st)
                        # Remove from persistent mapping so it won't be restored next run.
                        self.backend.index.delete(bytes(bh))
                        del self.blocks[bh]
                        evicted_one = True
                        self.backend.index.commit()
                        break
                if not evicted_one:
                    return None

        assert len(allocated) == len(to_store)
        for bh, st in zip(to_store, allocated):
            assert isinstance(st, SSDBlockStatus)
            self.blocks[bh] = st

        store_spec = self.backend.get_load_store_spec(to_store, allocated)
        return PrepareStoreOutput(
            block_hashes_to_store=to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        if success:
            for bh in block_hashes:
                st = self.blocks.get(bh)
                if st is None:
                    continue
                if not st.is_ready:
                    st.ref_cnt = 0
                    # Persist only when the KV write succeeded.
                    self.backend.index.set(bytes(bh), int(st.slot_id))
            self.backend.index.commit()
        else:
            for bh in block_hashes:
                st = self.blocks.get(bh)
                if st is None:
                    continue
                if not st.is_ready:
                    self.backend.free(st)
                    self.backend.index.delete(bytes(bh))
                    del self.blocks[bh]
            self.backend.index.commit()


class SSDGPUTransferHandler(OffloadingHandler):
    """GPU <-> SSD transfers using kvikio when available.

    Notes:
    - This handler currently assumes offloaded_block_size == gpu_block_size (factor=1).
    - It operates on a single cross-layer KV cache tensor.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor,
        *,
        block_bytes: int,
        num_gpu_blocks: int,
    ):
        self.kv_cache = kv_cache
        self.block_bytes = block_bytes
        self.num_gpu_blocks = int(num_gpu_blocks)
        self._pending: dict[int, torch.cuda.Event] = {}

        # Determine the 'num_blocks' dimension by looking for the dimension
        # that changes when selecting blocks; for cross-layer tensors this is
        # typically 1 or 2 depending on layout. We default to the first dim
        # whose size equals the KV cache manager's num_blocks at runtime.
        self._num_blocks_dim: int | None = None

    def _resolve_num_blocks_dim(self, expected_num_blocks: int) -> int:
        if self._num_blocks_dim is not None:
            return self._num_blocks_dim
        for dim, sz in enumerate(self.kv_cache.shape):
            if sz == expected_num_blocks:
                self._num_blocks_dim = dim
                return dim
        # fallback: common layout puts num_blocks at dim=2
        self._num_blocks_dim = min(2, self.kv_cache.ndim - 1)
        return self._num_blocks_dim

    @staticmethod
    def _as_u8_1d_view(t: torch.Tensor) -> torch.Tensor:
        # Reinterpret raw bytes without dtype conversion.
        if not t.is_contiguous():
            t = t.contiguous()
        return t.view(torch.uint8).reshape(-1)

    def _validate_block_bytes(self, one_block: torch.Tensor) -> None:
        actual = int(one_block.numel() * one_block.element_size())
        if actual != int(self.block_bytes):
            raise ValueError(
                f"KV block byte-size mismatch: tensor={actual} bytes, spec={self.block_bytes} bytes. "
                "This usually means the cache dir is being reused across incompatible (model/dtype/layout/block_size)."
            )

    def _pwrite_blocks(self, path: str, slot_ids: np.ndarray, packed_blocks: torch.Tensor) -> None:
        # packed_blocks is shaped [N, ...] on CUDA.
        n = int(slot_ids.size)
        if n == 0:
            return
        try:
            from kvikio import CuFile  # type: ignore
            _kv_print(
                f"[SSDKV] STORE write blocks={n} bytes_per_block={self.block_bytes} via=kvikio file={path}"
            )
            with CuFile(path, "r+b") as f:
                for i, slot in enumerate(slot_ids.tolist()):
                    offset = int(slot) * int(self.block_bytes)
                    blk = packed_blocks[i]
                    self._validate_block_bytes(blk)
                    buf = self._as_u8_1d_view(blk)
                    f.pwrite(buf, offset)  # type: ignore[attr-defined]
        except Exception as e:
            _kv_print(
                f"[SSDKV] STORE write blocks={n} bytes_per_block={self.block_bytes} via=posix file={path} (kvikio_unavailable={type(e).__name__})"
            )
            with open(path, "r+b", buffering=0) as f:
                for i, slot in enumerate(slot_ids.tolist()):
                    offset = int(slot) * int(self.block_bytes)
                    blk = packed_blocks[i]
                    self._validate_block_bytes(blk)
                    buf = self._as_u8_1d_view(blk).cpu().numpy().tobytes()
                    f.seek(offset)
                    f.write(buf)

    def _pread_blocks(self, path: str, slot_ids: np.ndarray, out_packed: torch.Tensor) -> None:
        # out_packed is shaped [N, ...] on CUDA.
        n = int(slot_ids.size)
        if n == 0:
            return
        try:
            from kvikio import CuFile  # type: ignore
            _kv_print(
                f"[SSDKV] LOAD read blocks={n} bytes_per_block={self.block_bytes} via=kvikio file={path}"
            )
            with CuFile(path, "r+b") as f:
                for i, slot in enumerate(slot_ids.tolist()):
                    offset = int(slot) * int(self.block_bytes)
                    blk = out_packed[i]
                    self._validate_block_bytes(blk)
                    buf = self._as_u8_1d_view(blk)
                    f.pread(buf, offset)  # type: ignore[attr-defined]
        except Exception as e:
            _kv_print(
                f"[SSDKV] LOAD read blocks={n} bytes_per_block={self.block_bytes} via=posix file={path} (kvikio_unavailable={type(e).__name__})"
            )
            with open(path, "rb", buffering=0) as f:
                for i, slot in enumerate(slot_ids.tolist()):
                    offset = int(slot) * int(self.block_bytes)
                    f.seek(offset)
                    raw = f.read(int(self.block_bytes))
                    if len(raw) != int(self.block_bytes):
                        raise IOError(
                            f"Short read from SSD cache file {path}: expected={self.block_bytes}, got={len(raw)}"
                        )
                    arr = np.frombuffer(raw, dtype=np.uint8).copy()
                    cpu = torch.from_numpy(arr)
                    gpu = cpu.to(device="cuda", non_blocking=True)
                    dst = self._as_u8_1d_view(out_packed[i])
                    dst.copy_(gpu)

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec

        stream = torch.cuda.current_stream()
        ev = torch.cuda.Event()

        if isinstance(src, GPULoadStoreSpec) and isinstance(dst, SSDLoadStoreSpec):
            # GPU -> SSD (store)
            _kv_print(
                f"[SSDKV] STORE job={job_id} blocks={int(dst.slot_ids.size)} file={dst.data_file}"
            )
            gpu_block_ids = torch.from_numpy(src.block_ids).to(device="cuda", non_blocking=True)
            dim = self._resolve_num_blocks_dim(expected_num_blocks=self.num_gpu_blocks)
            gathered = torch.index_select(self.kv_cache, dim=dim, index=gpu_block_ids)
            packed = gathered.movedim(dim, 0).contiguous() if dim != 0 else gathered.contiguous()
            with torch.cuda.stream(stream):
                self._pwrite_blocks(dst.data_file, dst.slot_ids, packed)
                ev.record(stream)

        elif isinstance(src, SSDLoadStoreSpec) and isinstance(dst, GPULoadStoreSpec):
            # SSD -> GPU (load)
            _kv_print(
                f"[SSDKV] LOAD job={job_id} blocks={int(src.slot_ids.size)} file={src.data_file}"
            )
            gpu_block_ids = torch.from_numpy(dst.block_ids).to(device="cuda", non_blocking=True)
            dim = self._resolve_num_blocks_dim(expected_num_blocks=self.num_gpu_blocks)

            # Allocate a packed buffer [N, ...] matching one-block slice.
            one = torch.index_select(self.kv_cache, dim=dim, index=gpu_block_ids[:1])
            one_packed = one.movedim(dim, 0) if dim != 0 else one
            block_shape = tuple(one_packed.shape[1:])
            out_packed = torch.empty(
                (gpu_block_ids.numel(),) + block_shape,
                device="cuda",
                dtype=self.kv_cache.dtype,
            )

            with torch.cuda.stream(stream):
                self._pread_blocks(src.data_file, src.slot_ids, out_packed)
                to_scatter = out_packed.movedim(0, dim).contiguous() if dim != 0 else out_packed
                self.kv_cache.index_copy_(dim, gpu_block_ids, to_scatter)
                ev.record(stream)

        else:
            raise TypeError(f"Unsupported transfer spec types: {type(src)} -> {type(dst)}")

        self._pending[job_id] = ev
        return True

    def get_finished(self) -> list[TransferResult]:
        finished: list[TransferResult] = []
        for job_id, ev in list(self._pending.items()):
            if ev.query():
                finished.append((job_id, True))
                del self._pending[job_id]
        return finished

    def wait(self, job_ids: set[int]) -> None:
        for jid in job_ids:
            ev = self._pending.get(jid)
            if ev is not None:
                ev.synchronize()


class SSDOffloadingSpec(OffloadingSpec):
    """SSD-backed OffloadingSpec for OffloadingConnector.

    Configure via kv_transfer_config.kv_connector_extra_config, e.g.:
      {
        "spec_name": "SSDOffloadingSpec",
        "spec_module_path": "daser.kvcache.ssd_offloading_spec",
        "ssd_cache_dir": "/data/$USER/vllm_kv_ssd",
        "max_blocks": 500000,
      }

    Current limitations:
    - Only supports offloaded_block_size == gpu_block_size (block_size_factor=1).
    - Optimized for the cross-layer KV cache tensor path.
    """

    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)

        if self.offloaded_block_size != self.gpu_block_size:
            raise ValueError(
                "SSDOffloadingSpec currently requires offloaded_block_size == gpu_block_size "
                f"(got offloaded_block_size={self.offloaded_block_size}, gpu_block_size={self.gpu_block_size})."
            )

        cache_dir = self.extra_config.get("ssd_cache_dir")
        if not cache_dir:
            cache_dir = os.path.join("/tmp", os.getenv("USER", "unknown"), "vllm_kv_ssd")
        self.cache_dir = Path(os.path.expandvars(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        max_blocks = int(self.extra_config.get("max_blocks", 100000))
        self.max_blocks = max_blocks

        if kv_cache_config is None:
            raise ValueError("SSDOffloadingSpec requires kv_cache_config")

        # We assume group 0 (current OffloadingConnector limitation).
        group = kv_cache_config.kv_cache_groups[0]
        num_layers_in_group = len(group.layer_names)

        # bytes per block across all layers for that group
        block_bytes = num_layers_in_group * group.kv_cache_spec.page_size_bytes
        self.block_bytes = int(block_bytes)

        logger.info(
            "SSDOffloadingSpec initialized: cache_dir=%s, block_bytes=%d, max_blocks=%d",
            str(self.cache_dir),
            self.block_bytes,
            self.max_blocks,
        )

        self.num_gpu_blocks = int(kv_cache_config.num_blocks)

    def get_manager(self) -> OffloadingManager:
        backend = SSDBackend(cache_dir=self.cache_dir, block_bytes=self.block_bytes, max_blocks=self.max_blocks)
        return PersistentLRUOffloadingManager(backend=backend)

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        # Prefer cross-layer tensor; OffloadingConnector sets prefer_cross_layer_blocks.
        if len(kv_caches) != 1:
            raise ValueError(
                "SSDOffloadingSpec currently supports exactly one KV cache tensor (cross-layer path). "
                f"Got {list(kv_caches.keys())}"
            )

        kv_cache = next(iter(kv_caches.values()))
        handler = SSDGPUTransferHandler(
            kv_cache=kv_cache,
            block_bytes=self.block_bytes,
            num_gpu_blocks=self.num_gpu_blocks,
        )

        # GPU -> SSD store
        yield (GPULoadStoreSpec, SSDLoadStoreSpec, handler)
        # SSD -> GPU load
        yield (SSDLoadStoreSpec, GPULoadStoreSpec, handler)
