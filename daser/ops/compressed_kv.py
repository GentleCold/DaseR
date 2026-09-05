# SPDX-License-Identifier: Apache-2.0

"""Fused strict-lossless decode from GPU staging into cross-layer KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cupy
import numpy as np
import torch

from daser.compression.format import CODEBOOK_ENTRIES, IO_ALIGNMENT

_SLOT_HEADER_BYTES = IO_ALIGNMENT
_SLOT_FIXED_HEADER_BYTES = 120
_PLANE_DESCRIPTOR_BYTES = 48
_THREADS = 256

_CUDA_SOURCE = r"""
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
        const unsigned int rank = group_escape_base + warp_prefix[warp]
            + __popc(mask & lower);
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
"""


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


__all__ = ["FusedCompressedKVDecoder", "compressed_slot_metadata"]
