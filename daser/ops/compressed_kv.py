# SPDX-License-Identifier: Apache-2.0

"""Fused strict-lossless decode from GPU staging into cross-layer KV cache."""

from dataclasses import dataclass
import time
from typing import Any

import cupy
import numpy as np
import torch

from daser.compression.format import (
    CODEBOOK_ENTRIES,
    IO_ALIGNMENT,
    CompressedStoreGeometry,
    PlaneDescriptor,
    SlotHeader,
    SlotMode,
    align_up,
    digest_bytes,
)
from daser.logging import init_logger

_SLOT_HEADER_BYTES = IO_ALIGNMENT
_SLOT_FIXED_HEADER_BYTES = 120
_PLANE_DESCRIPTOR_BYTES = 48
_THREADS = 256
_WARPS = _THREADS // 32
_UNVERIFIED_SLOT_HASH = b"\x01" + b"\x00" * 31
_ONLINE_PACK_CACHE: dict[tuple[object, int, int, int, int], object] = {}
_ONLINE_COUNT_CACHE: dict[int, object] = {}
_ONLINE_CUDA_MODULE_CACHE: dict[int, object] = {}

logger = init_logger(__name__)

_CUDA_SOURCE = r"""
#define CODEBOOK_ENTRIES 15

extern "C" __global__ void fused_decode_layout(
    const unsigned char* __restrict__ src,
    const long long* __restrict__ slot_offsets,
    const int* __restrict__ block_ids,
    const int* __restrict__ modes,
    const unsigned char* __restrict__ codebooks,
    unsigned short* __restrict__ dst,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_scalars,
    int tiles_per_plane,
    int num_dst_blocks) {
  const int linear = (int)blockIdx.x;
  const int tile = linear % tiles_per_plane;
  const int plane = (linear / tiles_per_plane) % num_planes;
  const int slot = linear / (tiles_per_plane * num_planes);
  if (slot >= num_slots) return;

  const int block_id = block_ids[slot];
  if (block_id < 0 || block_id >= num_dst_blocks) return;
  const int tid = (int)threadIdx.x;
  const int tile_begin = tile * tile_scalars;
  const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
  const long long slot_base = slot_offsets[slot];
  const long long dst_base =
      ((long long)block_id * num_planes + plane) * plane_scalars;

  if (modes[slot] == 0) {
    const unsigned short* raw =
        reinterpret_cast<const unsigned short*>(src + slot_base);
    const long long raw_base = (long long)plane * plane_scalars;
    for (int scalar = tile_begin + tid; scalar < tile_end;
         scalar += blockDim.x) {
      dst[dst_base + scalar] = raw[raw_base + scalar];
    }
    return;
  }

  const unsigned char* descriptor = src + slot_base + 120 + plane * 48;
  const unsigned int* fields =
      reinterpret_cast<const unsigned int*>(descriptor + 4);
  const unsigned int low_offset = fields[4];
  const unsigned int symbol_offset = fields[5];
  const unsigned int prefix_offset = fields[6];
  const unsigned int escape_offset = fields[7];
  const unsigned char* low = src + slot_base + low_offset;
  const unsigned char* symbols = src + slot_base + symbol_offset;
  const unsigned int* prefixes = reinterpret_cast<const unsigned int*>(
      src + slot_base + prefix_offset);
  const unsigned char* escapes = src + slot_base + escape_offset;
  const unsigned int tile_escape_start = prefixes[tile];
  __shared__ unsigned int warp_counts[8];
  __shared__ unsigned int warp_prefix[8];
  __shared__ unsigned int group_escape_base;
  __shared__ unsigned int group_escape_count;
  if (tid == 0) group_escape_base = 0;
  __syncthreads();

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int groups = (tile_scalars + blockDim.x - 1) / blockDim.x;
  for (int group = 0; group < groups; ++group) {
    const int scalar = tile_begin + group * blockDim.x + tid;
    const bool active = scalar < tile_end;
    unsigned char code = 0;
    if (active) {
      const unsigned char packed = symbols[scalar >> 1];
      code = (scalar & 1) ? (packed >> 4) : (packed & 15);
    }
    const unsigned mask = __ballot_sync(0xffffffffu, active && code == 15);
    if (lane == 0) warp_counts[warp] = __popc(mask);
    __syncthreads();
    if (tid == 0) {
      unsigned int running = 0;
      #pragma unroll
      for (int index = 0; index < 8; ++index) {
        warp_prefix[index] = running;
        running += warp_counts[index];
      }
      group_escape_count = running;
    }
    __syncthreads();
    if (active) {
      unsigned char high;
      if (code == 15) {
        const unsigned lower = lane == 0 ? 0u : ((1u << lane) - 1u);
        const unsigned int rank =
            group_escape_base + warp_prefix[warp] + __popc(mask & lower);
        high = escapes[tile_escape_start + rank];
      } else {
        high = codebooks[plane * 15 + code];
      }
      dst[dst_base + scalar] =
          (unsigned short)low[scalar] | ((unsigned short)high << 8);
    }
    __syncthreads();
    if (tid == 0) group_escape_base += group_escape_count;
    __syncthreads();
  }
}

extern "C" __global__ void count_online_escapes(
    const unsigned short* __restrict__ src,
    const int* __restrict__ block_ids,
    const unsigned char* __restrict__ lookup,
    unsigned int* __restrict__ counts,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_scalars,
    int tiles_per_plane,
    int max_tiles,
    int block_tokens,
    int valid_token_count,
    int full_valid) {
  // One warp owns one tile.  The previous implementation made every warp in
  // a block walk the same tile and synchronized the whole block once per
  // tile.  Qwen3-8B has 128 tiles per plane, so those repeated barriers kept
  // the count pass resident on the device while vLLM was trying to run the
  // next generation step.  Keeping eight independent tiles in flight per
  // block preserves the output layout without a cross-warp barrier.
  const int plane_linear = (int)blockIdx.x;
  const int plane = plane_linear % num_planes;
  const int slot = plane_linear / num_planes;
  if (slot >= num_slots) return;
  const int block_id = block_ids[slot];
  if (block_id < 0) {
    for (int tile = (int)threadIdx.x; tile < tiles_per_plane;
         tile += blockDim.x) {
      counts[plane_linear * max_tiles + tile] = 0;
    }
    return;
  }
  const int row_scalars = plane_scalars / block_tokens;
  const long long src_base =
      ((long long)block_id * num_planes + plane) * plane_scalars;
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  for (int tile_group = 0; tile_group < (tiles_per_plane + 7) / 8;
       ++tile_group) {
    const int tile = tile_group * 8 + warp;
    if (tile >= tiles_per_plane) continue;
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    unsigned int total = 0;
    for (int scalar_group = tile_begin + lane * 4;
         scalar_group < tile_end;
         scalar_group += 128) {
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int scalar = scalar_group + index;
        const bool active = scalar < tile_end;
        bool valid = active;
        if (!full_valid && valid) {
          const int token = scalar / row_scalars;
          valid = slot * block_tokens + token < valid_token_count;
        }
        const unsigned int high =
            valid ? (src[src_base + scalar] >> 8) : 0u;
        const unsigned mask = __ballot_sync(
            0xffffffffu, valid && lookup[plane * 256 + high] == 15);
        if (lane == 0) total += __popc(mask);
      }
    }
    if (lane == 0) counts[plane_linear * max_tiles + tile] = total;
  }
}

extern "C" __global__ void fused_online_pack(
    const unsigned short* __restrict__ src,
    const int* __restrict__ block_ids,
    const long long* __restrict__ low_offsets,
    const long long* __restrict__ symbol_offsets,
    const long long* __restrict__ escape_offsets,
    const unsigned int* __restrict__ prefixes,
    const unsigned char* __restrict__ lookup,
    unsigned char* __restrict__ dst,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_scalars,
    int max_tiles,
    int block_tokens,
    int valid_token_count) {
  const int slot = (int)blockIdx.x;
  const int plane = (int)blockIdx.y;
  if (slot >= num_slots || plane >= num_planes) return;
  const int block_id = block_ids[slot];
  if (block_id < 0) return;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tile_count = (plane_scalars + tile_scalars - 1) / tile_scalars;
  const int row_scalars = plane_scalars / block_tokens;
  const long long src_base =
      ((long long)block_id * num_planes + plane) * plane_scalars;
  const int prefix_index_base = (slot * num_planes + plane) * (max_tiles + 1);
  const long long low_base = low_offsets[slot * num_planes + plane];
  const long long symbol_base = symbol_offsets[slot * num_planes + plane];
  const long long escape_base = escape_offsets[slot * num_planes + plane];
  const long long prefix_output_base =
      low_base + plane_scalars + (plane_scalars + 1) / 2;

  // Prefixes are small but numerous (one per tile for every plane). Writing
  // them here avoids hundreds of Python-issued CUDA copies per store batch;
  // the same stream already orders this metadata before the decoder can read
  // the record from the staging buffer.
  unsigned int* prefix_output =
      reinterpret_cast<unsigned int*>(dst + prefix_output_base);
  for (int prefix_index = tid; prefix_index <= tile_count;
       prefix_index += blockDim.x) {
    prefix_output[prefix_index] = prefixes[prefix_index_base + prefix_index];
  }

  // One warp owns one tile.  Four values per lane keep the ballot/rank work
  // small while preserving scalar order in the nibble and escape streams.
  for (int tile_group = 0; tile_group < (tile_count + 7) / 8;
       ++tile_group) {
    const int tile = tile_group * 8 + warp;
    if (tile >= tile_count) continue;
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    const unsigned int tile_escape_start = prefixes[prefix_index_base + tile];
    unsigned int escape_group_base = 0;

    for (int scalar_base = tile_begin; scalar_base < tile_end;
         scalar_base += 128) {
      unsigned short bits[4] = {0, 0, 0, 0};
      unsigned char codes[4] = {CODEBOOK_ENTRIES, CODEBOOK_ENTRIES,
                                CODEBOOK_ENTRIES, CODEBOOK_ENTRIES};
      bool active[4] = {false, false, false, false};
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int scalar = scalar_base + lane * 4 + index;
        active[index] = scalar < tile_end;
        if (active[index]) {
          const int token = scalar / row_scalars;
          const bool valid = slot * block_tokens + token < valid_token_count;
          if (valid) bits[index] = src[src_base + scalar];
          codes[index] = lookup[plane * 256 + (bits[index] >> 8)];
        }
      }

      const unsigned int lower = (1u << lane) - 1u;
      unsigned int masks[4];
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        masks[index] = __ballot_sync(
            0xffffffffu, active[index] && codes[index] == CODEBOOK_ENTRIES);
      }
      unsigned int ranks[4];
      unsigned int running = escape_group_base;
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        ranks[index] = running + __popc(masks[index] & lower);
        running += __popc(masks[index]);
      }

      const int first_scalar = scalar_base + lane * 4;
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        if (active[index]) {
          dst[low_base + first_scalar + index] =
              (unsigned char)(bits[index] & 0xff);
        }
      }
      if (first_scalar < tile_end) {
        dst[symbol_base + first_scalar / 2] =
            (unsigned char)(codes[0] | (codes[1] << 4));
      }
      if (first_scalar + 2 < tile_end) {
        dst[symbol_base + (first_scalar + 2) / 2] =
            (unsigned char)(codes[2] | (codes[3] << 4));
      }
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int scalar = first_scalar + index;
        if (active[index] && codes[index] == CODEBOOK_ENTRIES) {
          dst[escape_base + tile_escape_start + ranks[index]] =
              (unsigned char)(bits[index] >> 8);
        }
      }
      escape_group_base = running;
    }
  }
}
"""


def _online_packer_kernel(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    block_tokens: int,
) -> object:
    """Compile or return the process-local CUDA online pack kernel.

    The online writer is kept in the same RawModule as escape counting so the
    two kernels share one per-device compilation and module lifetime.  Unlike
    the decoder, this path is launched with a two-dimensional slot/plane grid
    and takes all layout metadata as runtime inputs.
    """
    device_index = int(device.index or 0)
    key = (device_index, dtype, num_planes, plane_scalars, tile_scalars, block_tokens)
    cached = _ONLINE_PACK_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        module = _ONLINE_CUDA_MODULE_CACHE.get(device_index)
        if module is None:
            with cupy.cuda.Device(device_index):
                module = cupy.RawModule(
                    code=_CUDA_SOURCE,
                    options=("--std=c++14",),
                    name_expressions=(
                        "count_online_escapes",
                        "fused_online_pack",
                    ),
                )
            _ONLINE_CUDA_MODULE_CACHE[device_index] = module
        kernel = module.get_function("fused_online_pack")
    except Exception as exc:
        raise RuntimeError(
            "CUDA online pack compilation failed for "
            f"dtype={dtype}, num_planes={num_planes}, plane_scalars={plane_scalars}"
        ) from exc
    _ONLINE_PACK_CACHE[key] = kernel
    return kernel


def _online_count_kernel(device: torch.device) -> object:
    """Compile or return the CUDA escape-count kernel for one device."""
    index = int(device.index or 0)
    cached = _ONLINE_COUNT_CACHE.get(index)
    if cached is not None:
        return cached
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        with cupy.cuda.Device(index):
            module = cupy.RawModule(
                code=_CUDA_SOURCE,
                options=("--std=c++14",),
                name_expressions=(
                    "count_online_escapes",
                    "fused_online_pack",
                ),
            )
        _ONLINE_CUDA_MODULE_CACHE[index] = module
    kernel = module.get_function("count_online_escapes")
    _ONLINE_COUNT_CACHE[index] = kernel
    return kernel


def _launch_online_pack(
    kernel: object,
    *,
    kv_bits: torch.Tensor,
    block_ids: torch.Tensor,
    low_offsets: torch.Tensor,
    symbol_offsets: torch.Tensor,
    escape_offsets: torch.Tensor,
    prefixes: torch.Tensor,
    lookup: torch.Tensor,
    staging: torch.Tensor,
    num_slots: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    max_tiles: int,
    block_tokens: int,
    valid_token_count: int,
    stream: torch.cuda.Stream,
) -> None:
    """Launch the CUDA online writer on the caller-owned CUDA stream.

    Args:
        kernel: Compiled ``fused_online_pack`` RawKernel.
        kv_bits: Flattened BF16 source cache viewed as uint16.
        block_ids: Device block IDs, with ``-1`` for raw fallback slots.
        low_offsets: Per-slot/plane low-byte destinations in staging.
        symbol_offsets: Per-slot/plane nibble destinations in staging.
        escape_offsets: Per-slot/plane escape destinations in staging.
        prefixes: Per-slot/plane tile escape prefixes.
        lookup: Plane-major high-byte to nibble lookup table.
        staging: CUDA byte staging buffer receiving packed records.
        num_slots: Number of active slot records.
        num_planes: Number of layer/K/V planes.
        plane_scalars: Scalar count in one plane.
        tile_scalars: Scalar count in one independent codec tile.
        max_tiles: Capacity of each prefix row.
        block_tokens: Tokens represented by one source block.
        valid_token_count: Valid token extent across the selected slots.
        stream: CUDA stream that orders metadata copies and the writer.

    Async/thread-safety:
        The launch is asynchronous with respect to the host. The caller owns
        the tensors and must retain them until ``stream`` reaches completion.
    """
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_slots, num_planes),
            (_THREADS,),
            (
                cupy.asarray(kv_bits),
                cupy.asarray(block_ids),
                cupy.asarray(low_offsets),
                cupy.asarray(symbol_offsets),
                cupy.asarray(escape_offsets),
                cupy.asarray(prefixes),
                cupy.asarray(lookup),
                cupy.asarray(staging),
                np.int32(num_slots),
                np.int32(num_planes),
                np.int32(plane_scalars),
                np.int32(tile_scalars),
                np.int32(max_tiles),
                np.int32(block_tokens),
                np.int32(valid_token_count),
            ),
        )


def warm_fused_online_kv_packer(
    kv_cache: torch.Tensor, max_slots_per_buffer: int, tile_scalars: int = 1024
) -> None:
    """Compile and launch the online packer before serving traffic.

    Args:
        kv_cache: Worker-owned contiguous BF16 KV cache.
        max_slots_per_buffer: Maximum packed records in one store staging lease.
        tile_scalars: Dynamic codec tile size.

    Raises:
        ValueError: If the cache geometry is unsupported.
        RuntimeError: If TileLang cannot compile or launch the representative
            kernel.
    """
    if (
        kv_cache.device.type != "cuda"
        or kv_cache.dim() != 6
        or not kv_cache.is_contiguous()
        or kv_cache.dtype is not torch.bfloat16
    ):
        raise ValueError("online packer requires contiguous CUDA BF16 KV cache")
    if max_slots_per_buffer <= 0 or tile_scalars <= 0:
        raise ValueError("online packer geometry must be positive")
    num_planes = int(kv_cache.shape[1]) * 2
    plane_scalars = int(np.prod(kv_cache.shape[3:]))
    block_tokens = int(kv_cache.shape[3])
    kernel = _online_packer_kernel(
        device=kv_cache.device,
        dtype=kv_cache.dtype,
        num_planes=num_planes,
        plane_scalars=plane_scalars,
        tile_scalars=tile_scalars,
        block_tokens=block_tokens,
    )
    max_tiles = (plane_scalars + tile_scalars - 1) // tile_scalars
    device = kv_cache.device
    try:
        with torch.cuda.device(device), torch.inference_mode(False):
            ids = torch.full(
                (max_slots_per_buffer,), -1, dtype=torch.int32, device=device
            )
            zeros = torch.zeros(max_slots_per_buffer, dtype=torch.int64, device=device)
            prefixes = torch.zeros(
                max_slots_per_buffer * num_planes * (max_tiles + 1),
                dtype=torch.uint32,
                device=device,
            )
            lookup = torch.zeros(num_planes * 256, dtype=torch.uint8, device=device)
            output = torch.empty(1, dtype=torch.uint8, device=device)
            _launch_online_pack(
                kernel,
                kv_bits=kv_cache.view(torch.uint16).reshape(
                    int(kv_cache.shape[0]), num_planes, plane_scalars
                ),
                block_ids=ids,
                low_offsets=zeros,
                symbol_offsets=zeros,
                escape_offsets=zeros,
                prefixes=prefixes,
                lookup=lookup,
                staging=output,
                num_slots=max_slots_per_buffer,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                tile_scalars=tile_scalars,
                max_tiles=max_tiles,
                block_tokens=block_tokens,
                valid_token_count=max_slots_per_buffer * block_tokens,
                stream=torch.cuda.current_stream(device),
            )
            torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            "TileLang online packer warmup failed for "
            f"dtype={kv_cache.dtype}, num_planes={num_planes}, "
            f"plane_scalars={plane_scalars}, max_slots={max_slots_per_buffer}"
        ) from exc


@dataclass(frozen=True)
class OnlinePackedSlot:
    """Metadata for one packed payload written into a staging buffer."""

    logical_slot: int
    mode: SlotMode
    source_offset: int
    stored_length: int


class FusedOnlineKVPacker:
    """Encode live KV blocks directly into a CUDA store staging buffer."""

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        codebooks: bytes,
        tile_scalars: int,
        max_slots_per_buffer: int,
    ) -> None:
        if (
            kv_cache.device.type != "cuda"
            or kv_cache.dim() != 6
            or not kv_cache.is_contiguous()
            or kv_cache.dtype is not torch.bfloat16
        ):
            raise ValueError("online packer requires contiguous CUDA BF16 KV cache")
        self._device = kv_cache.device
        self._num_blocks = int(kv_cache.shape[0])
        self._num_planes = int(kv_cache.shape[1]) * 2
        self._plane_scalars = int(np.prod(kv_cache.shape[3:]))
        self._tile_scalars = int(tile_scalars)
        self._max_slots = int(max_slots_per_buffer)
        self._geometry = CompressedStoreGeometry(
            num_slots=self._num_blocks,
            slot_size=int(kv_cache[0].nbytes),
            block_tokens=int(kv_cache.shape[3]),
            num_layers=int(kv_cache.shape[1]),
            num_kv_heads=int(kv_cache.shape[4]),
            head_dim=int(kv_cache.shape[5]),
            tile_scalars=self._tile_scalars,
        )
        if len(codebooks) != self._num_planes * CODEBOOK_ENTRIES:
            raise ValueError("online codebooks do not match KV geometry")
        self._codebook_hash = digest_bytes(codebooks)
        self._kv_bits = kv_cache.view(torch.uint16).reshape(
            self._num_blocks, self._num_planes, self._plane_scalars
        )
        lookup = np.full((self._num_planes, 256), CODEBOOK_ENTRIES, dtype=np.uint8)
        tables = np.frombuffer(codebooks, dtype=np.uint8).reshape(
            self._num_planes, CODEBOOK_ENTRIES
        )
        lookup[np.arange(self._num_planes)[:, None], tables] = np.arange(
            CODEBOOK_ENTRIES, dtype=np.uint8
        )
        self._lookup = torch.from_numpy(lookup.reshape(-1).copy()).to(self._device)
        self._kernel = _ONLINE_PACK_CACHE.get(
            (
                int(kv_cache.device.index or 0),
                kv_cache.dtype,
                self._num_planes,
                self._plane_scalars,
                self._tile_scalars,
                int(kv_cache.shape[3]),
            )
        )
        if self._kernel is None:
            raise RuntimeError("online packer was not warmed before configuration")
        self._diagnostic_emitted = False
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        prefix_count = self._max_slots * self._num_planes * (max_tiles + 1)
        count_count = self._max_slots * self._num_planes * max_tiles
        with torch.inference_mode(False):
            self._host_block_ids = torch.empty(
                self._max_slots, dtype=torch.int32, pin_memory=True
            )
            self._host_launch_ids = torch.empty(
                self._max_slots, dtype=torch.int32, pin_memory=True
            )
            self._host_low_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._host_symbol_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._host_escape_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._host_prefixes = torch.empty(
                prefix_count, dtype=torch.uint32, pin_memory=True
            )
            self._host_counts = torch.empty(
                count_count, dtype=torch.uint32, pin_memory=True
            )
            self._host_headers = torch.empty(
                (self._max_slots, IO_ALIGNMENT),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._device_block_ids = torch.empty(
                self._max_slots, dtype=torch.int32, device=self._device
            )
            self._device_launch_ids = torch.empty(
                self._max_slots, dtype=torch.int32, device=self._device
            )
            self._device_low_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                device=self._device,
            )
            self._device_symbol_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                device=self._device,
            )
            self._device_escape_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                device=self._device,
            )
            self._device_prefixes = torch.empty(
                prefix_count, dtype=torch.uint32, device=self._device
            )
            self._device_counts = torch.empty(
                count_count, dtype=torch.uint32, device=self._device
            )

    def pack_into(
        self,
        *,
        staging: torch.Tensor,
        block_ids: list[int],
        logical_slots: list[int],
        slot_stride: int,
        stream: torch.cuda.Stream,
        valid_token_count: int | None = None,
    ) -> list[OnlinePackedSlot]:
        """Pack selected live blocks into fixed-stride staging regions.

        Args:
            staging: Worker-owned CUDA byte staging buffer.
            block_ids: Physical vLLM blocks in logical order.
            logical_slots: DaseR slot IDs corresponding to ``block_ids``.
            slot_stride: Raw rank-local slot bytes and staging region stride.
            stream: Store CUDA stream ordering the pack and metadata copies.
            valid_token_count: Optional valid token extent for a partial tail.

        Returns:
            Actual mode, source offset, and stored length for every slot.
        """
        if not block_ids or len(block_ids) != len(logical_slots):
            raise ValueError("online packer block metadata is invalid")
        if len(block_ids) > self._max_slots or slot_stride != self._geometry.slot_size:
            raise ValueError("online packer staging geometry is invalid")
        if staging.device != self._device or staging.dtype is not torch.uint8:
            raise ValueError("online packer staging must be a CUDA byte tensor")
        if any(block < 0 or block >= self._num_blocks for block in block_ids):
            raise ValueError("online packer block ID is invalid")
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        slot_count = len(block_ids)
        effective_valid_tokens = (
            slot_count * self._geometry.block_tokens
            if valid_token_count is None
            else int(valid_token_count)
        )
        if not 0 < effective_valid_tokens <= slot_count * self._geometry.block_tokens:
            raise ValueError("online packer valid token extent is invalid")
        with torch.cuda.device(self._device):
            host_ids = self._host_block_ids[:slot_count]
            host_ids.copy_(torch.as_tensor(block_ids, dtype=torch.int32))
            device_ids = self._device_block_ids[:slot_count]
            device_counts = self._device_counts[
                : slot_count * self._num_planes * max_tiles
            ]
            count_kernel = _online_count_kernel(self._device)
            count_grid = slot_count * self._num_planes
            full_valid = int(
                effective_valid_tokens == slot_count * self._geometry.block_tokens
            )
            count_started = time.perf_counter()
            with torch.cuda.stream(stream):
                device_ids.copy_(host_ids, non_blocking=True)
                external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
                with external_stream:
                    count_kernel(
                        (count_grid,),
                        (_THREADS,),
                        (
                            cupy.asarray(self._kv_bits),
                            cupy.asarray(device_ids),
                            cupy.asarray(self._lookup),
                            cupy.asarray(device_counts),
                            np.int32(slot_count),
                            np.int32(self._num_planes),
                            np.int32(self._plane_scalars),
                            np.int32(self._tile_scalars),
                            np.int32(max_tiles),
                            np.int32(max_tiles),
                            np.int32(self._geometry.block_tokens),
                            np.int32(effective_valid_tokens),
                            np.int32(full_valid),
                        ),
                    )
                self._host_counts[: slot_count * self._num_planes * max_tiles].copy_(
                    device_counts, non_blocking=True
                )
            stream.synchronize()
            count_sync_ms = (time.perf_counter() - count_started) * 1000
            count_host = (
                self._host_counts[: slot_count * self._num_planes * max_tiles]
                .numpy()
                .reshape(slot_count, self._num_planes, max_tiles)
            )
            prefix_cpu = (
                self._host_prefixes[: slot_count * self._num_planes * (max_tiles + 1)]
                .numpy()
                .reshape(slot_count, self._num_planes, max_tiles + 1)
            )
            prefix_cpu.fill(0)
            prefix_cpu[..., 1:] = np.cumsum(count_host, axis=-1, dtype=np.uint32)
            logger.debug(
                "[PACK] count synchronization slots=%d full_valid=%s elapsed_ms=%.3f",
                slot_count,
                bool(full_valid),
                count_sync_ms,
            )

            plans: list[tuple[SlotMode, int, tuple[PlaneDescriptor, ...]]] = []
            for slot_index in range(len(block_ids)):
                cursor = IO_ALIGNMENT
                descriptors: list[PlaneDescriptor] = []
                for plane in range(self._num_planes):
                    tile_count = max_tiles
                    escape_count = int(prefix_cpu[slot_index, plane, tile_count])
                    record_offset = cursor
                    symbol_length = (self._plane_scalars + 1) // 2
                    prefix_length = (tile_count + 1) * 4
                    escape_offset = (
                        record_offset
                        + self._plane_scalars
                        + symbol_length
                        + prefix_length
                    )
                    record_length = align_up(
                        escape_offset + escape_count - record_offset
                    )
                    layer, kv = divmod(plane, 2)
                    descriptors.append(
                        PlaneDescriptor(
                            layer=layer,
                            kv=kv,
                            scalar_count=self._plane_scalars,
                            tile_count=tile_count,
                            record_offset=record_offset,
                            record_length=record_length,
                            low_offset=record_offset,
                            symbol_offset=record_offset + self._plane_scalars,
                            prefix_offset=(
                                record_offset + self._plane_scalars + symbol_length
                            ),
                            escape_offset=escape_offset,
                            escape_count=escape_count,
                        )
                    )
                    cursor += record_length
                if cursor > slot_stride:
                    plans.append((SlotMode.RAW, slot_stride, tuple()))
                else:
                    plans.append((SlotMode.COMPRESSED, cursor, tuple(descriptors)))

            if not self._diagnostic_emitted:
                compressed_count = sum(
                    mode is SlotMode.COMPRESSED for mode, _length, _descriptors in plans
                )
                logger.info(
                    "[PACK] first batch slots=%d max_slots=%d fixed_escape_bytes=%d "
                    "lengths=%s modes=%s compressed=%d",
                    slot_count,
                    self._max_slots,
                    self._plane_scalars * self._num_planes,
                    [length for _mode, length, _descriptors in plans],
                    [mode.name.lower() for mode, _length, _descriptors in plans],
                    compressed_count,
                )
                self._diagnostic_emitted = True

            slot_bases = np.arange(slot_count, dtype=np.int64) * slot_stride
            low_offsets = (
                self._host_low_offsets[: slot_count * self._num_planes]
                .numpy()
                .reshape(slot_count, self._num_planes)
            )
            symbol_offsets = (
                self._host_symbol_offsets[: slot_count * self._num_planes]
                .numpy()
                .reshape(slot_count, self._num_planes)
            )
            escape_offsets = (
                self._host_escape_offsets[: slot_count * self._num_planes]
                .numpy()
                .reshape(slot_count, self._num_planes)
            )
            low_offsets.fill(0)
            symbol_offsets.fill(0)
            escape_offsets.fill(0)
            for index, (mode, _length, descriptors) in enumerate(plans):
                if mode is SlotMode.COMPRESSED:
                    for plane, descriptor in enumerate(descriptors):
                        base = int(slot_bases[index])
                        low_offsets[index, plane] = base + descriptor.low_offset
                        symbol_offsets[index, plane] = base + descriptor.symbol_offset
                        escape_offsets[index, plane] = base + descriptor.escape_offset
            with torch.cuda.stream(stream):
                device_low_offsets = self._device_low_offsets[
                    : slot_count * self._num_planes
                ]
                device_low_offsets.copy_(
                    self._host_low_offsets[: slot_count * self._num_planes],
                    non_blocking=True,
                )
                device_symbol_offsets = self._device_symbol_offsets[
                    : slot_count * self._num_planes
                ]
                device_symbol_offsets.copy_(
                    self._host_symbol_offsets[: slot_count * self._num_planes],
                    non_blocking=True,
                )
                device_escape_offsets = self._device_escape_offsets[
                    : slot_count * self._num_planes
                ]
                device_escape_offsets.copy_(
                    self._host_escape_offsets[: slot_count * self._num_planes],
                    non_blocking=True,
                )
                device_prefixes = self._device_prefixes[
                    : slot_count * self._num_planes * (max_tiles + 1)
                ]
                device_prefixes.copy_(
                    self._host_prefixes[
                        : slot_count * self._num_planes * (max_tiles + 1)
                    ],
                    non_blocking=True,
                )
                host_launch_ids = self._host_launch_ids[:slot_count]
                for index, (mode, length, descriptors) in enumerate(plans):
                    base = int(slot_bases[index])
                    if mode is SlotMode.RAW:
                        host_launch_ids[index] = -1
                        raw_block = (
                            self._kv_bits[block_ids[index]]
                            .contiguous()
                            .view(torch.uint8)
                            .reshape(-1)
                        )
                        staging[base : base + slot_stride].copy_(raw_block)
                        continue
                    host_launch_ids[index] = block_ids[index]
                    header = SlotHeader(
                        slot_id=int(logical_slots[index]),
                        raw_length=slot_stride,
                        stored_length=length,
                        tile_scalars=self._tile_scalars,
                        num_layers=self._geometry.num_layers,
                        codebook_hash=self._codebook_hash,
                        raw_hash=_UNVERIFIED_SLOT_HASH,
                        descriptors=descriptors,
                    ).pack()
                    self._host_headers[index].copy_(
                        torch.frombuffer(bytearray(header), dtype=torch.uint8)
                    )
                    staging[base : base + IO_ALIGNMENT].copy_(
                        self._host_headers[index], non_blocking=True
                    )
                device_launch_ids = self._device_launch_ids[:slot_count]
                device_launch_ids.copy_(host_launch_ids, non_blocking=True)
                _launch_online_pack(
                    self._kernel,
                    kv_bits=self._kv_bits,
                    block_ids=device_launch_ids,
                    low_offsets=device_low_offsets,
                    symbol_offsets=device_symbol_offsets,
                    escape_offsets=device_escape_offsets,
                    prefixes=device_prefixes,
                    lookup=self._lookup,
                    staging=staging,
                    num_slots=slot_count,
                    num_planes=self._num_planes,
                    plane_scalars=self._plane_scalars,
                    tile_scalars=self._tile_scalars,
                    max_tiles=max_tiles,
                    block_tokens=self._geometry.block_tokens,
                    valid_token_count=effective_valid_tokens,
                    stream=stream,
                )
        return [
            OnlinePackedSlot(
                logical_slot=int(logical_slots[index]),
                mode=mode,
                source_offset=int(slot_bases[index]),
                stored_length=length,
            )
            for index, (mode, length, _descriptors) in enumerate(plans)
        ]


@dataclass
class _MetadataRing:
    """Persistent pinned/device launch metadata for one staging buffer."""

    host_offsets: torch.Tensor
    host_blocks: torch.Tensor
    host_modes: torch.Tensor
    device_offsets: torch.Tensor
    device_blocks: torch.Tensor
    device_modes: torch.Tensor


class FusedCompressedKVDecoder:
    """Own a compiled decoder and persistent metadata for fixed staging slots.

    Args:
        kv_cache: Contiguous cross-layer tensor with layout
            ``[blocks, layers, 2, tokens, heads, dim]``.
        codebooks: Plane-major static 15-entry high-byte tables.
        tile_scalars: Codec tile size encoded in the side index.
        ring_depth: Number of independently leased load staging buffers.
        max_slots_per_buffer: Maximum slot records described by one launch.

    Async/thread-safety:
        Constructed before request traffic. ``decode`` is called only on the
        LoadPipeline thread; metadata rings are independently indexed by the
        staging buffer lease.
    """

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        codebooks: bytes,
        tile_scalars: int,
        ring_depth: int,
        max_slots_per_buffer: int,
    ) -> None:
        if kv_cache.device.type != "cuda" or kv_cache.dim() != 6:
            raise ValueError("compressed restore requires a 6D CUDA KV cache")
        if not kv_cache.is_contiguous() or kv_cache.dtype is not torch.bfloat16:
            raise ValueError("compressed restore requires contiguous BF16 KV cache")
        if ring_depth <= 0 or max_slots_per_buffer <= 0 or tile_scalars <= 0:
            raise ValueError("compressed decoder ring geometry must be positive")
        self._kv_cache = kv_cache
        self._num_blocks = int(kv_cache.shape[0])
        self._num_layers = int(kv_cache.shape[1])
        self._num_planes = self._num_layers * 2
        self._plane_scalars = int(np.prod(kv_cache.shape[3:]))
        self._tile_scalars = tile_scalars
        self._tiles_per_plane = (self._plane_scalars + tile_scalars - 1) // tile_scalars
        expected_codebooks = self._num_planes * CODEBOOK_ENTRIES
        if len(codebooks) != expected_codebooks:
            raise ValueError("compressed codebooks do not match KV layer geometry")
        with cupy.cuda.Device(kv_cache.device.index or 0):
            module = cupy.RawModule(
                code=_CUDA_SOURCE,
                options=("--std=c++14",),
                name_expressions=("fused_decode_layout",),
            )
            self._kernel = module.get_function("fused_decode_layout")
            self._codebooks = cupy.asarray(np.frombuffer(codebooks, dtype=np.uint8))
        self._rings = tuple(
            self._allocate_metadata(kv_cache.device, max_slots_per_buffer)
            for _ in range(ring_depth)
        )

    def decode(
        self,
        *,
        staging: torch.Tensor,
        staging_offsets: list[int],
        block_ids: list[int],
        modes: list[int],
        buffer_index: int,
        stream: torch.cuda.Stream,
    ) -> int:
        """Launch fused decode/layout into the registered vLLM KV tensor.

        Args:
            staging: GPU byte tensor filled by the server transfer layer.
            staging_offsets: Start of each indexed slot record in staging.
            block_ids: Matching destination physical vLLM blocks.
            modes: Zero for raw slots and one for compressed slots.
            buffer_index: Fixed staging/metadata ring index.
            stream: LoadPipeline CUDA stream ordered after transfer completion.

        Returns:
            Number of logical slots restored by this launch.

        Raises:
            ValueError: If metadata lengths, capacity, or block IDs are invalid.

        Async/thread-safety:
            Called on one load thread. The caller retains the staging lease and
            synchronizes ``stream`` before reusing its ring index.
        """
        slot_count = len(staging_offsets)
        if not (slot_count == len(block_ids) == len(modes)):
            raise ValueError("compressed restore metadata lengths do not match")
        if slot_count == 0:
            return 0
        try:
            ring = self._rings[buffer_index]
        except IndexError as exc:
            raise ValueError("compressed metadata ring index is invalid") from exc
        if slot_count > ring.host_offsets.numel():
            raise ValueError("compressed restore exceeds metadata ring capacity")
        for index, value in enumerate(staging_offsets):
            ring.host_offsets[index] = value
        for index, value in enumerate(block_ids):
            ring.host_blocks[index] = value
        for index, value in enumerate(modes):
            ring.host_modes[index] = value
        with torch.cuda.stream(stream):
            ring.device_offsets[:slot_count].copy_(
                ring.host_offsets[:slot_count], non_blocking=True
            )
            ring.device_blocks[:slot_count].copy_(
                ring.host_blocks[:slot_count], non_blocking=True
            )
            ring.device_modes[:slot_count].copy_(
                ring.host_modes[:slot_count], non_blocking=True
            )
        grid = slot_count * self._num_planes * self._tiles_per_plane
        external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
        with external_stream:
            self._kernel(
                (grid,),
                (_THREADS,),
                (
                    cupy.asarray(staging),
                    cupy.asarray(ring.device_offsets),
                    cupy.asarray(ring.device_blocks),
                    cupy.asarray(ring.device_modes),
                    self._codebooks,
                    cupy.asarray(self._kv_cache),
                    np.int32(slot_count),
                    np.int32(self._num_planes),
                    np.int32(self._plane_scalars),
                    np.int32(self._tile_scalars),
                    np.int32(self._tiles_per_plane),
                    np.int32(self._num_blocks),
                ),
            )
        return slot_count

    @staticmethod
    def _allocate_metadata(device: torch.device, capacity: int) -> _MetadataRing:
        # vLLM registers KV caches under InferenceMode, while this metadata is
        # updated later on the load thread. Explicitly create normal tensors so
        # thread-local InferenceMode state at construction cannot make the ring
        # immutable outside that context.
        with torch.inference_mode(False):
            host_offsets = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            host_blocks = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
            host_modes = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
            return _MetadataRing(
                host_offsets=host_offsets,
                host_blocks=host_blocks,
                host_modes=host_modes,
                device_offsets=torch.empty(capacity, dtype=torch.int64, device=device),
                device_blocks=torch.empty(capacity, dtype=torch.int32, device=device),
                device_modes=torch.empty(capacity, dtype=torch.int32, device=device),
            )


def compressed_slot_metadata(
    per_req_ranges: list[Any],
) -> tuple[list[int], list[int], list[int]]:
    """Flatten batch restore ranges into kernel slot metadata.

    Args:
        per_req_ranges: Read-plan ranges containing ReqLoadSpec values.

    Returns:
        Staging offsets, destination block IDs, and integer slot modes.

    Async/thread-safety:
        Pure CPU planning safe on the load pipeline thread.
    """
    offsets: list[int] = []
    block_ids: list[int] = []
    modes: list[int] = []
    for item in per_req_ranges:
        start, _end, spec = item if len(item) == 3 else (item[0], item[1], item[3])
        cursor = int(start)
        if len(spec.compressed_slots) != len(spec.block_ids):
            raise ValueError("compressed slot metadata does not match block IDs")
        for slot, block_id in zip(spec.compressed_slots, spec.block_ids, strict=True):
            offsets.append(cursor)
            block_ids.append(int(block_id))
            modes.append(0 if slot.mode == "raw" else 1)
            cursor += slot.stored_length
    return offsets, block_ids, modes


__all__ = [
    "FusedCompressedKVDecoder",
    "FusedOnlineKVPacker",
    "OnlinePackedSlot",
    "compressed_slot_metadata",
    "warm_fused_online_kv_packer",
]
