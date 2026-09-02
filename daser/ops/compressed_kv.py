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
_ONLINE_THREADS = 32
_ONLINE_FIXED_THREADS = 256
# Bound each online codec launch so a large first-turn prefix yields to vLLM
# between batches.  The raw staging lease remains unchanged; this only limits
# the amount of GPU work submitted by one pack launch.
ONLINE_PACK_BATCH_SLOTS = 4
_WARPS = _THREADS // 32
_UNVERIFIED_SLOT_HASH = b"\x01" + b"\x00" * 31
_ONLINE_PACK_CACHE: dict[tuple[object, int, int, int, int], object] = {}
_ONLINE_FIXED_PACK_CACHE: dict[tuple[object, int, int, int, int], object] = {}
_ONLINE_VARIABLE_PACK_CACHE: dict[int, object] = {}
_ONLINE_COUNT_CACHE: dict[int, object] = {}
_ONLINE_REDUCE_CACHE: dict[int, object] = {}
_ONLINE_PREFIX_CACHE: dict[int, object] = {}
_ONLINE_LAYOUT_CACHE: dict[int, object] = {}
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
  const int warps_per_block = blockDim.x / 32;
  for (int tile = warp; tile < tiles_per_plane; tile += warps_per_block) {
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    unsigned int total = 0;
    // Count escapes with a warp-local reduction.  Counting does not need
    // ballot-derived ranks; each lane can accumulate its own scalar matches
    // and reduce them once, leaving ballots to the writer kernel where they
    // are required to address the variable-length escape stream.
    for (int scalar = tile_begin + lane; scalar < tile_end; scalar += 32) {
      bool valid = true;
      if (!full_valid) {
        const int token = scalar / row_scalars;
        valid = slot * block_tokens + token < valid_token_count;
      }
      const unsigned int high =
          valid ? (src[src_base + scalar] >> 8) : 0u;
      total += valid && lookup[plane * 256 + high] == 15;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    if (lane == 0) {
      counts[plane_linear * max_tiles + tile] = total;
    }
  }
}

extern "C" __global__ void reduce_online_escape_counts(
    const unsigned int* __restrict__ counts,
    unsigned int* __restrict__ totals,
    int num_rows,
    int max_tiles) {
  const int row = (int)blockIdx.x;
  if (row >= num_rows || threadIdx.x != 0) return;
  const unsigned int base = (unsigned int)row * (unsigned int)max_tiles;
  unsigned int total = 0;
  for (int tile = 0; tile < max_tiles; ++tile) {
    total += counts[base + (unsigned int)tile];
  }
  totals[row] = total;
}

extern "C" __global__ void build_online_prefix(
    const unsigned int* __restrict__ counts,
    unsigned int* __restrict__ prefixes,
    int num_rows,
    int max_tiles) {
  // Each row is only a few hundred bytes.  A single thread avoids launching
  // a general-purpose scan (and its temporary int64 tensor) for every store
  // batch while keeping the result resident for the writer kernel.
  const int row = (int)blockIdx.x;
  if (row >= num_rows || threadIdx.x != 0) return;
  const unsigned int count_base = (unsigned int)row * (unsigned int)max_tiles;
  const unsigned int prefix_base = (unsigned int)row * (unsigned int)(max_tiles + 1);
  unsigned int running = 0;
  prefixes[prefix_base] = 0;
  for (int tile = 0; tile < max_tiles; ++tile) {
    running += counts[count_base + (unsigned int)tile];
    prefixes[prefix_base + (unsigned int)tile + 1u] = running;
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
  const int warps_per_block = blockDim.x / 32;
  for (int tile = warp; tile < tile_count; tile += warps_per_block) {
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

// The fixed-scratch writer combines escape counting, prefix construction, and
// payload emission in one launch.  The old variable-length path had to
// synchronize after a count pass so Python could compute every plane offset;
// that synchronization is particularly expensive when vLLM is concurrently
// running a prefill.  The scratch envelope is compacted into a variable-length
// record before transfer, so unused escape capacity never reaches storage.  A
// second tiny restore kernel handles the rare incompressible slot.
#define MAX_ONLINE_TILES 1024

extern "C" __global__ void fused_online_pack_fixed(
    const unsigned short* __restrict__ src,
    const int* __restrict__ block_ids,
    const unsigned char* __restrict__ lookup,
    unsigned char* __restrict__ dst,
    unsigned int* __restrict__ totals,
    unsigned int* __restrict__ overflow,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_scalars,
    int tiles_per_plane,
    int block_tokens,
    int valid_token_count,
    int full_valid,
    int slot_stride,
    int plane_record_bytes,
    int fixed_escape_bytes) {
  const int slot = (int)blockIdx.x;
  const int plane = (int)blockIdx.y;
  if (slot >= num_slots || plane >= num_planes || tiles_per_plane > MAX_ONLINE_TILES) {
    return;
  }
  const int block_id = block_ids[slot];
  if (block_id < 0) return;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps_per_block = blockDim.x / 32;
  const int row_scalars = plane_scalars / block_tokens;
  const long long src_base =
      ((long long)block_id * num_planes + plane) * plane_scalars;
  const long long slot_base = (long long)slot * slot_stride;
  const long long record_base =
      slot_base + 4096LL + (long long)plane * plane_record_bytes;

  __shared__ unsigned int tile_counts[MAX_ONLINE_TILES];
  __shared__ unsigned int tile_prefix[MAX_ONLINE_TILES + 1];
  __shared__ unsigned int plane_overflow;
  for (int index = tid; index < tiles_per_plane; index += blockDim.x) {
    tile_counts[index] = 0;
  }
  if (tid == 0) plane_overflow = 0;
  __syncthreads();

  // Count one tile per warp and reduce only within that warp.  This preserves
  // parallelism for the 256-tile Qwen geometry without a block barrier per
  // tile.
  for (int tile = warp; tile < tiles_per_plane; tile += warps_per_block) {
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    unsigned int total = 0;
    for (int scalar = tile_begin + lane; scalar < tile_end; scalar += 32) {
      bool valid = true;
      if (!full_valid) {
        const int token = scalar / row_scalars;
        valid = slot * block_tokens + token < valid_token_count;
      }
      const unsigned int high = valid ? (src[src_base + scalar] >> 8) : 0u;
      total += valid && lookup[plane * 256 + high] == CODEBOOK_ENTRIES;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    if (lane == 0) tile_counts[tile] = total;
  }
  __syncthreads();

  if (tid == 0) {
    unsigned int running = 0;
    tile_prefix[0] = 0;
    for (int tile = 0; tile < tiles_per_plane; ++tile) {
      running += tile_counts[tile];
      tile_prefix[tile + 1] = running;
    }
    totals[slot * num_planes + plane] = running;
    plane_overflow = running > (unsigned int)fixed_escape_bytes;
    if (plane_overflow) overflow[slot] = 1;
  }
  __syncthreads();
  if (plane_overflow) return;

  const long long low_base = record_base;
  const long long symbol_base = low_base + plane_scalars;
  const long long prefix_base = symbol_base + (plane_scalars + 1) / 2;
  const long long escape_base = prefix_base + 4LL * (tiles_per_plane + 1);
  for (int index = tid; index <= tiles_per_plane; index += blockDim.x) {
    reinterpret_cast<unsigned int*>(dst + prefix_base)[index] = tile_prefix[index];
  }

  for (int tile = warp; tile < tiles_per_plane; tile += warps_per_block) {
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    const unsigned int tile_escape_start = tile_prefix[tile];
    unsigned int escape_group_base = 0;
    for (int scalar_base = tile_begin; scalar_base < tile_end;
         scalar_base += 128) {
      unsigned short bits[4] = {0, 0, 0, 0};
      unsigned char codes[4] = {0, 0, 0, 0};
      bool active[4] = {false, false, false, false};
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int scalar = scalar_base + lane * 4 + index;
        active[index] = scalar < tile_end;
        if (active[index]) {
          const int token = scalar / row_scalars;
          const bool valid = full_valid ||
              slot * block_tokens + token < valid_token_count;
          if (valid) {
            bits[index] = src[src_base + scalar];
            codes[index] = lookup[plane * 256 + (bits[index] >> 8)];
          }
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

// Encode each tile once and keep its escape bytes in a fixed tile-local
// scratch segment.  The previous fixed writer counted every scalar and then
// reread the complete source plane to emit low/symbol/escape streams.  This
// variant emits low bytes, symbols, and tile-local escapes during the count
// pass; compaction later reorders only the escape bytes into the canonical
// cumulative-prefix layout.  A tile that exceeds its bounded scratch segment
// is marked raw so no truncated payload can be published.
extern "C" __global__ void fused_online_pack_fixed_single_read(
    const unsigned short* __restrict__ src,
    const int* __restrict__ block_ids,
    const unsigned char* __restrict__ lookup,
    unsigned char* __restrict__ dst,
    unsigned int* __restrict__ totals,
    unsigned int* __restrict__ overflow,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_scalars,
    int tiles_per_plane,
    int block_tokens,
    int valid_token_count,
    int full_valid,
    int scratch_slot_stride,
    int scratch_plane_record_bytes,
    int tile_escape_capacity) {
  const int slot = (int)blockIdx.x;
  const int plane = (int)blockIdx.y;
  if (slot >= num_slots || plane >= num_planes ||
      tiles_per_plane > MAX_ONLINE_TILES) {
    return;
  }
  const int block_id = block_ids[slot];
  if (block_id < 0) return;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps_per_block = blockDim.x / 32;
  const int row_scalars = plane_scalars / block_tokens;
  const long long src_base =
      ((long long)block_id * num_planes + plane) * plane_scalars;
  const long long slot_base = (long long)slot * scratch_slot_stride;
  const long long record_base =
      slot_base + 4096LL + (long long)plane * scratch_plane_record_bytes;
  const long long low_base = record_base;
  const long long symbol_base = low_base + plane_scalars;
  const long long prefix_base = symbol_base + (plane_scalars + 1) / 2;
  const long long escape_base = prefix_base + 4LL * (tiles_per_plane + 1);

  __shared__ unsigned int tile_counts[MAX_ONLINE_TILES];
  __shared__ unsigned int tile_prefix[MAX_ONLINE_TILES + 1];
  __shared__ unsigned int plane_overflow;
  for (int index = tid; index < tiles_per_plane; index += blockDim.x) {
    tile_counts[index] = 0;
  }
  if (tid == 0) plane_overflow = 0;
  __syncthreads();

  // One warp owns one tile.  Each lane handles four adjacent scalars so
  // ballot ranks preserve the canonical scalar order in the escape stream.
  for (int tile = warp; tile < tiles_per_plane; tile += warps_per_block) {
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    unsigned int escape_group_base = 0;
    for (int scalar_base = tile_begin; scalar_base < tile_end;
         scalar_base += 128) {
      unsigned short bits[4] = {0, 0, 0, 0};
      unsigned char codes[4] = {0, 0, 0, 0};
      bool active[4] = {false, false, false, false};
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int scalar = scalar_base + lane * 4 + index;
        active[index] = scalar < tile_end;
        if (active[index]) {
          const int token = scalar / row_scalars;
          const bool valid = full_valid ||
              slot * block_tokens + token < valid_token_count;
          if (valid) {
            bits[index] = src[src_base + scalar];
            codes[index] = lookup[plane * 256 + (bits[index] >> 8)];
          }
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
        if (active[index] && codes[index] == CODEBOOK_ENTRIES &&
            ranks[index] < (unsigned int)tile_escape_capacity) {
          dst[escape_base + (long long)tile * tile_escape_capacity +
              ranks[index]] = (unsigned char)(bits[index] >> 8);
        }
      }
      escape_group_base = running;
    }
    if (lane == 0) {
      tile_counts[tile] = escape_group_base;
      if (escape_group_base > (unsigned int)tile_escape_capacity) {
        atomicExch(&plane_overflow, 1u);
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    unsigned int running = 0;
    tile_prefix[0] = 0;
    for (int tile = 0; tile < tiles_per_plane; ++tile) {
      running += tile_counts[tile];
      tile_prefix[tile + 1] = running;
    }
    totals[slot * num_planes + plane] = running;
    if (plane_overflow) overflow[slot] = 1;
  }
  __syncthreads();

  // Prefixes are part of the persisted record.  They are written even for an
  // overflow slot because the raw restore path overwrites the complete slot.
  for (int index = tid; index <= tiles_per_plane; index += blockDim.x) {
    reinterpret_cast<unsigned int*>(dst + prefix_base)[index] =
        tile_prefix[index];
  }
}

extern "C" __global__ void restore_online_raw_overflow(
    const unsigned short* __restrict__ src,
    const int* __restrict__ block_ids,
    const unsigned int* __restrict__ overflow,
    unsigned char* __restrict__ dst,
    const long long* __restrict__ slot_offsets,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int slot_stride) {
  const int slot = (int)blockIdx.x;
  if (slot >= num_slots || overflow[slot] == 0) return;
  const int block_id = block_ids[slot];
  if (block_id < 0) return;
  const int tid = (int)threadIdx.x;
  const int raw_scalars = num_planes * plane_scalars;
  const unsigned short* raw = src + (long long)block_id * raw_scalars;
  unsigned short* output = reinterpret_cast<unsigned short*>(
      dst + slot_offsets[slot]);
  for (int index = tid; index < raw_scalars; index += blockDim.x) {
    output[index] = raw[index];
  }
}

// Build the variable-length destination layout after the fixed writer has
// produced per-plane escape totals.  The records are ordered slot-major, so a
// single lightweight thread can scan the small batch and publish all source,
// destination, and payload spans without a host round trip.  ``overflow`` is
// extended to cover records whose complete compressed payload would exceed the
// raw slot envelope; the existing raw restore kernel then handles both cases.
extern "C" __global__ void build_online_fixed_layout(
    const unsigned int* __restrict__ totals,
    unsigned int* __restrict__ overflow,
    long long* __restrict__ source_offsets,
    long long* __restrict__ destination_offsets,
    long long* __restrict__ payload_bytes,
    long long* __restrict__ slot_offsets,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int max_tiles,
    int slot_stride,
    int scratch_plane_record_bytes,
    int scratch_slot_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const long long payload_base =
      (long long)plane_scalars + (plane_scalars + 1LL) / 2LL +
      4LL * ((long long)max_tiles + 1LL);
  long long staging_cursor = 0;
  for (int slot = 0; slot < num_slots; ++slot) {
    const long long row_base = (long long)slot * num_planes;
    bool raw = overflow[slot] != 0;
    long long record_cursor = 4096;
    if (!raw) {
      for (int plane = 0; plane < num_planes; ++plane) {
        const long long payload =
            payload_base + (long long)totals[row_base + plane];
        const long long record = (payload + 4095LL) & ~4095LL;
        if (record_cursor + record > (long long)slot_stride) {
          raw = true;
          break;
        }
        record_cursor += record;
      }
    }
    slot_offsets[slot] = staging_cursor;
    if (raw) {
      overflow[slot] = 1;
      staging_cursor += (long long)slot_stride;
      for (int plane = 0; plane < num_planes; ++plane) {
        const long long row = row_base + plane;
        source_offsets[row] = 0;
        destination_offsets[row] = 0;
        payload_bytes[row] = 0;
      }
      continue;
    }

    overflow[slot] = 0;
    record_cursor = 4096;
    for (int plane = 0; plane < num_planes; ++plane) {
      const long long row = row_base + plane;
      const long long payload = payload_base + (long long)totals[row];
      source_offsets[row] =
          (long long)slot * scratch_slot_stride + 4096LL +
          (long long)plane * scratch_plane_record_bytes;
      destination_offsets[row] = staging_cursor + record_cursor;
      payload_bytes[row] = payload;
      record_cursor += (payload + 4095LL) & ~4095LL;
    }
    staging_cursor += record_cursor;
  }
}

extern "C" __global__ void compact_online_fixed(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const long long* __restrict__ src_offsets,
    const long long* __restrict__ dst_offsets,
    const long long* __restrict__ payload_bytes,
    int num_slots,
    int num_planes) {
  const int linear = (int)blockIdx.x;
  const int plane = linear % num_planes;
  const int slot = linear / num_planes;
  if (slot >= num_slots) return;
  const long long source = src_offsets[linear];
  const long long target = dst_offsets[linear];
  const long long bytes = payload_bytes[linear];
  for (long long index = (long long)threadIdx.x; index < bytes;
       index += blockDim.x) {
    dst[target + index] = src[source + index];
  }
}

// Compact a single-read fixed record.  Low bytes, symbols, and prefixes are
// already contiguous; only the escape stream is tile-local in the scratch
// envelope and must be gathered according to its cumulative prefixes.
extern "C" __global__ void compact_online_fixed_tiled(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const long long* __restrict__ src_offsets,
    const long long* __restrict__ dst_offsets,
    const long long* __restrict__ payload_bytes,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_count,
    int tile_escape_capacity) {
  const int linear = (int)blockIdx.x;
  const int plane = linear % num_planes;
  const int slot = linear / num_planes;
  if (slot >= num_slots) return;
  const long long source = src_offsets[linear];
  const long long target = dst_offsets[linear];
  const long long payload = payload_bytes[linear];
  if (payload <= 0) return;
  const long long symbol_bytes = (plane_scalars + 1) / 2;
  const long long prefix_bytes = 4LL * (tile_count + 1);
  const long long base_bytes = plane_scalars + symbol_bytes + prefix_bytes;
  if (payload < base_bytes) return;
  for (long long index = (long long)threadIdx.x; index < base_bytes;
       index += blockDim.x) {
    dst[target + index] = src[source + index];
  }
  const unsigned int* prefixes = reinterpret_cast<const unsigned int*>(
      src + source + plane_scalars + symbol_bytes);
  const long long source_escape = source + base_bytes;
  const long long target_escape = target + base_bytes;
  const long long escape_bytes = payload - base_bytes;
  for (int tile = 0; tile < tile_count; ++tile) {
    const unsigned int begin = prefixes[tile];
    const unsigned int end = prefixes[tile + 1];
    if (begin >= (unsigned int)escape_bytes) continue;
    const unsigned int count = min(
        end - begin, (unsigned int)escape_bytes - begin);
    const long long tile_source =
        source_escape + (long long)tile * tile_escape_capacity;
    for (unsigned int index = (unsigned int)threadIdx.x; index < count;
         index += blockDim.x) {
      dst[target_escape + begin + index] = src[tile_source + index];
    }
  }
}

// Emit a counted record directly into its final variable-length location.
// ``count_online_escapes`` has already populated the per-tile counts and the
// host has planned one aligned record offset per plane.  Keeping the prefix
// scan in shared memory avoids a fixed-envelope write followed by a second
// device-to-device compaction pass.
extern "C" __global__ void fused_online_pack_variable(
    const unsigned short* __restrict__ src,
    const int* __restrict__ block_ids,
    const unsigned char* __restrict__ lookup,
    const unsigned int* __restrict__ counts,
    const unsigned int* __restrict__ raw_modes,
    unsigned char* __restrict__ dst,
    const long long* __restrict__ plane_offsets,
    int num_slots,
    int num_planes,
    int plane_scalars,
    int tile_scalars,
    int tiles_per_plane,
    int block_tokens,
    int valid_token_count,
    int full_valid,
    int max_tiles) {
  const int slot = (int)blockIdx.x;
  const int plane = (int)blockIdx.y;
  if (slot >= num_slots || plane >= num_planes ||
      tiles_per_plane > MAX_ONLINE_TILES || raw_modes[slot] != 0) {
    return;
  }
  const int block_id = block_ids[slot];
  if (block_id < 0) return;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps_per_block = blockDim.x / 32;
  const int row_scalars = plane_scalars / block_tokens;
  const long long src_base =
      ((long long)block_id * num_planes + plane) * plane_scalars;
  const unsigned int row = (unsigned int)slot * (unsigned int)num_planes +
      (unsigned int)plane;
  __shared__ unsigned int tile_prefix[MAX_ONLINE_TILES + 1];
  if (tid == 0) {
    unsigned int running = 0;
    tile_prefix[0] = 0;
    const unsigned int count_base = row * (unsigned int)max_tiles;
    for (int tile = 0; tile < tiles_per_plane; ++tile) {
      running += counts[count_base + (unsigned int)tile];
      tile_prefix[tile + 1] = running;
    }
  }
  __syncthreads();

  const long long low_base = plane_offsets[row];
  const long long symbol_base = low_base + plane_scalars;
  const long long prefix_base = symbol_base + (plane_scalars + 1) / 2;
  const long long escape_base = prefix_base + 4LL * (tiles_per_plane + 1);
  for (int index = tid; index <= tiles_per_plane; index += blockDim.x) {
    reinterpret_cast<unsigned int*>(dst + prefix_base)[index] = tile_prefix[index];
  }

  for (int tile = warp; tile < tiles_per_plane; tile += warps_per_block) {
    const int tile_begin = tile * tile_scalars;
    const int tile_end = min(tile_begin + tile_scalars, plane_scalars);
    const unsigned int tile_escape_start = tile_prefix[tile];
    unsigned int escape_group_base = 0;
    for (int scalar_base = tile_begin; scalar_base < tile_end;
         scalar_base += 128) {
      unsigned short bits[4] = {0, 0, 0, 0};
      unsigned char codes[4] = {0, 0, 0, 0};
      bool active[4] = {false, false, false, false};
      #pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int scalar = scalar_base + lane * 4 + index;
        active[index] = scalar < tile_end;
        if (active[index]) {
          const int token = scalar / row_scalars;
          const bool valid = full_valid ||
              slot * block_tokens + token < valid_token_count;
          if (valid) {
            bits[index] = src[src_base + scalar];
            codes[index] = lookup[plane * 256 + (bits[index] >> 8)];
          }
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
                        "reduce_online_escape_counts",
                        "build_online_prefix",
                        "fused_online_pack",
                        "fused_online_pack_fixed",
                        "fused_online_pack_fixed_single_read",
                        "fused_online_pack_variable",
                        "restore_online_raw_overflow",
                        "build_online_fixed_layout",
                        "compact_online_fixed",
                        "compact_online_fixed_tiled",
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


def _online_fixed_packer_kernel(device: torch.device) -> object:
    """Compile or return the one-launch fixed-envelope pack kernel."""
    index = int(device.index or 0)
    cached = _ONLINE_FIXED_PACK_CACHE.get((index,))
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
                    "reduce_online_escape_counts",
                    "build_online_prefix",
                    "fused_online_pack",
                    "fused_online_pack_fixed",
                    "fused_online_pack_fixed_single_read",
                    "fused_online_pack_variable",
                    "restore_online_raw_overflow",
                    "build_online_fixed_layout",
                    "compact_online_fixed",
                    "compact_online_fixed_tiled",
                ),
            )
        _ONLINE_CUDA_MODULE_CACHE[index] = module
    kernel = module.get_function("fused_online_pack_fixed_single_read")
    _ONLINE_FIXED_PACK_CACHE[(index,)] = kernel
    return kernel


def _online_variable_packer_kernel(device: torch.device) -> object:
    """Return the direct variable-length online writer kernel."""
    index = int(device.index or 0)
    cached = _ONLINE_VARIABLE_PACK_CACHE.get(index)
    if cached is not None:
        return cached
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        _online_fixed_packer_kernel(device)
        module = _ONLINE_CUDA_MODULE_CACHE[index]
    kernel = module.get_function("fused_online_pack_variable")
    _ONLINE_VARIABLE_PACK_CACHE[index] = kernel
    return kernel


def _online_raw_restore_kernel(device: torch.device) -> object:
    """Return the overflow raw-copy kernel from the shared CUDA module."""
    index = int(device.index or 0)
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        _online_fixed_packer_kernel(device)
        module = _ONLINE_CUDA_MODULE_CACHE[index]
    return module.get_function("restore_online_raw_overflow")


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
                    "reduce_online_escape_counts",
                    "build_online_prefix",
                    "fused_online_pack",
                    "fused_online_pack_fixed",
                    "fused_online_pack_fixed_single_read",
                    "fused_online_pack_variable",
                    "restore_online_raw_overflow",
                    "build_online_fixed_layout",
                    "compact_online_fixed",
                    "compact_online_fixed_tiled",
                ),
            )
        _ONLINE_CUDA_MODULE_CACHE[index] = module
    kernel = module.get_function("count_online_escapes")
    _ONLINE_COUNT_CACHE[index] = kernel
    return kernel


def _online_reduce_kernel(device: torch.device) -> object:
    """Return the device reduction kernel for per-plane escape totals."""
    index = int(device.index or 0)
    cached = _ONLINE_REDUCE_CACHE.get(index)
    if cached is not None:
        return cached
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        _online_fixed_packer_kernel(device)
        module = _ONLINE_CUDA_MODULE_CACHE[index]
    kernel = module.get_function("reduce_online_escape_counts")
    _ONLINE_REDUCE_CACHE[index] = kernel
    return kernel


def _online_prefix_kernel(device: torch.device) -> object:
    """Compile or return the tiny device-resident prefix scan kernel."""
    index = int(device.index or 0)
    cached = _ONLINE_PREFIX_CACHE.get(index)
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
                    "reduce_online_escape_counts",
                    "build_online_prefix",
                    "fused_online_pack",
                    "fused_online_pack_fixed",
                    "fused_online_pack_fixed_single_read",
                    "fused_online_pack_variable",
                    "restore_online_raw_overflow",
                    "build_online_fixed_layout",
                    "compact_online_fixed",
                    "compact_online_fixed_tiled",
                ),
            )
        _ONLINE_CUDA_MODULE_CACHE[index] = module
    kernel = module.get_function("build_online_prefix")
    _ONLINE_PREFIX_CACHE[index] = kernel
    return kernel


def _launch_online_prefix(
    kernel: object,
    *,
    counts: torch.Tensor,
    prefixes: torch.Tensor,
    num_rows: int,
    max_tiles: int,
    stream: torch.cuda.Stream,
) -> None:
    """Build per-tile escape prefixes on the caller-owned CUDA stream.

    Args:
        kernel: Compiled ``build_online_prefix`` RawKernel.
        counts: Device row-major escape counts.
        prefixes: Device row-major output prefix table.
        num_rows: Number of slot/plane rows represented by ``counts``.
        max_tiles: Number of count entries per row.
        stream: Store stream ordering the metadata for the pack kernel.

    Async/thread-safety:
        The launch is asynchronous with respect to the host. The caller must
        retain both tensors until ``stream`` reaches completion.
    """
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_rows,),
            (1,),
            (
                cupy.asarray(counts),
                cupy.asarray(prefixes),
                np.int32(num_rows),
                np.int32(max_tiles),
            ),
        )


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
            (_ONLINE_THREADS,),
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


def _launch_online_fixed_pack(
    kernel: object,
    *,
    kv_bits: torch.Tensor,
    block_ids: torch.Tensor,
    lookup: torch.Tensor,
    staging: torch.Tensor,
    totals: torch.Tensor,
    overflow: torch.Tensor,
    num_slots: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    max_tiles: int,
    block_tokens: int,
    valid_token_count: int,
    plane_record_bytes: int,
    tile_escape_capacity: int,
    scratch_slot_stride: int,
    stream: torch.cuda.Stream,
) -> None:
    """Launch the single-read fixed-envelope online pack kernel."""
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_slots, num_planes),
            (_ONLINE_FIXED_THREADS,),
            (
                cupy.asarray(kv_bits),
                cupy.asarray(block_ids),
                cupy.asarray(lookup),
                cupy.asarray(staging),
                cupy.asarray(totals),
                cupy.asarray(overflow),
                np.int32(num_slots),
                np.int32(num_planes),
                np.int32(plane_scalars),
                np.int32(tile_scalars),
                np.int32(max_tiles),
                np.int32(block_tokens),
                np.int32(valid_token_count),
                np.int32(valid_token_count == num_slots * block_tokens),
                np.int32(scratch_slot_stride),
                np.int32(plane_record_bytes),
                np.int32(tile_escape_capacity),
            ),
        )


def _launch_online_fixed_layout(
    kernel: object,
    *,
    totals: torch.Tensor,
    overflow: torch.Tensor,
    source_offsets: torch.Tensor,
    destination_offsets: torch.Tensor,
    payload_bytes: torch.Tensor,
    slot_offsets: torch.Tensor,
    num_slots: int,
    num_planes: int,
    plane_scalars: int,
    max_tiles: int,
    slot_stride: int,
    scratch_plane_record_bytes: int,
    scratch_slot_stride: int,
    stream: torch.cuda.Stream,
) -> None:
    """Launch the device-resident variable-layout planner.

    Args:
        kernel: Compiled ``build_online_fixed_layout`` RawKernel.
        totals: Device per-slot/plane escape totals.
        overflow: Device raw-fallback flags, updated for envelope overflow.
        source_offsets: Device scratch payload offsets to fill.
        destination_offsets: Device final staging offsets to fill.
        payload_bytes: Device unpadded payload lengths to fill.
        slot_offsets: Device final staging base for each slot.
        num_slots: Active records in this batch.
        num_planes: KV planes per record.
        plane_scalars: Scalar count in one plane.
        max_tiles: Codec tiles in one plane.
        slot_stride: Raw bytes reserved for one slot.
        scratch_plane_record_bytes: Fixed scratch plane stride.
        scratch_slot_stride: Fixed scratch slot stride.
        stream: CUDA stream ordering the layout before restore/compact.

    Async/thread-safety:
        The launch is asynchronous with respect to the host. All output
        tensors remain owned by the packer until ``stream`` is synchronized.
    """
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (1,),
            (1,),
            (
                cupy.asarray(totals),
                cupy.asarray(overflow),
                cupy.asarray(source_offsets),
                cupy.asarray(destination_offsets),
                cupy.asarray(payload_bytes),
                cupy.asarray(slot_offsets),
                np.int32(num_slots),
                np.int32(num_planes),
                np.int32(plane_scalars),
                np.int32(max_tiles),
                np.int32(slot_stride),
                np.int32(scratch_plane_record_bytes),
                np.int32(scratch_slot_stride),
            ),
        )


def _launch_online_reduce(
    kernel: object,
    *,
    counts: torch.Tensor,
    totals: torch.Tensor,
    num_rows: int,
    max_tiles: int,
    stream: torch.cuda.Stream,
) -> None:
    """Reduce per-tile escape counts into one total per plane on CUDA."""
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_rows,),
            (1,),
            (
                cupy.asarray(counts),
                cupy.asarray(totals),
                np.int32(num_rows),
                np.int32(max_tiles),
            ),
        )


def _launch_online_variable_pack(
    kernel: object,
    *,
    kv_bits: torch.Tensor,
    block_ids: torch.Tensor,
    lookup: torch.Tensor,
    counts: torch.Tensor,
    raw_modes: torch.Tensor,
    staging: torch.Tensor,
    plane_offsets: torch.Tensor,
    num_slots: int,
    num_planes: int,
    plane_scalars: int,
    tile_scalars: int,
    max_tiles: int,
    block_tokens: int,
    valid_token_count: int,
    stream: torch.cuda.Stream,
) -> None:
    """Launch the direct variable-length online writer."""
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_slots, num_planes),
            (_THREADS,),
            (
                cupy.asarray(kv_bits),
                cupy.asarray(block_ids),
                cupy.asarray(lookup),
                cupy.asarray(counts),
                cupy.asarray(raw_modes),
                cupy.asarray(staging),
                cupy.asarray(plane_offsets),
                np.int32(num_slots),
                np.int32(num_planes),
                np.int32(plane_scalars),
                np.int32(tile_scalars),
                np.int32(max_tiles),
                np.int32(block_tokens),
                np.int32(valid_token_count),
                np.int32(valid_token_count == num_slots * block_tokens),
                np.int32(max_tiles),
            ),
        )


def _launch_online_raw_restore(
    kernel: object,
    *,
    kv_bits: torch.Tensor,
    block_ids: torch.Tensor,
    overflow: torch.Tensor,
    staging: torch.Tensor,
    slot_offsets: torch.Tensor,
    num_slots: int,
    num_planes: int,
    plane_scalars: int,
    slot_stride: int,
    stream: torch.cuda.Stream,
) -> None:
    """Restore overflow slots at their planned variable staging offsets.

    Args:
        kernel: Compiled raw-overflow CUDA kernel.
        kv_bits: Flattened BF16 source cache viewed as uint16.
        block_ids: Device block IDs in slot order.
        overflow: Device flags selecting raw fallback slots.
        staging: Destination staging byte buffer.
        slot_offsets: Device byte base for every logical staging slot.
        num_slots: Number of active slots.
        num_planes: Number of KV planes per slot.
        plane_scalars: Scalar count in one plane.
        slot_stride: Raw bytes reserved for one slot.
        stream: CUDA stream ordering the copy.

    Async/thread-safety:
        The launch is asynchronous; callers retain all tensors until the stream
        is synchronized before exporting the staging buffer.
    """
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_slots,),
            (_THREADS,),
            (
                cupy.asarray(kv_bits),
                cupy.asarray(block_ids),
                cupy.asarray(overflow),
                cupy.asarray(staging),
                cupy.asarray(slot_offsets),
                np.int32(num_slots),
                np.int32(num_planes),
                np.int32(plane_scalars),
                np.int32(slot_stride),
            ),
        )


def _launch_online_compact(
    kernel: object,
    *,
    source: torch.Tensor,
    destination: torch.Tensor,
    source_offsets: torch.Tensor,
    destination_offsets: torch.Tensor,
    payload_bytes: torch.Tensor,
    num_slots: int,
    num_planes: int,
    stream: torch.cuda.Stream,
) -> None:
    """Compact fixed-scratch plane payloads into variable-length records.

    Args:
        kernel: Compiled ``compact_online_fixed`` RawKernel.
        source: Device byte buffer containing fixed-envelope records.
        destination: Device byte staging buffer receiving compact records.
        source_offsets: Device byte offsets for each fixed plane payload.
        destination_offsets: Device byte offsets for each compact plane payload.
        payload_bytes: Device payload lengths, excluding aligned record padding.
        num_slots: Number of slot records in this batch.
        num_planes: Number of layer/K-or-V planes per slot.
        stream: CUDA stream ordering fixed writes and compaction.

    Async/thread-safety:
        The launch is asynchronous with respect to the host.  ``source`` and
        ``destination`` must remain alive until ``stream`` reaches completion.
    """
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_slots * num_planes,),
            (_THREADS,),
            (
                cupy.asarray(source),
                cupy.asarray(destination),
                cupy.asarray(source_offsets),
                cupy.asarray(destination_offsets),
                cupy.asarray(payload_bytes),
                np.int32(num_slots),
                np.int32(num_planes),
            ),
        )


def _launch_online_compact_tiled(
    kernel: object,
    *,
    source: torch.Tensor,
    destination: torch.Tensor,
    source_offsets: torch.Tensor,
    destination_offsets: torch.Tensor,
    payload_bytes: torch.Tensor,
    num_slots: int,
    num_planes: int,
    plane_scalars: int,
    tile_count: int,
    tile_escape_capacity: int,
    stream: torch.cuda.Stream,
) -> None:
    """Launch compaction for tile-local escape scratch records.

    Args:
        kernel: Compiled ``compact_online_fixed_tiled`` RawKernel.
        source: Device byte buffer containing fixed single-read records.
        destination: Device byte staging buffer receiving compact records.
        source_offsets: Device byte offsets for scratch plane payloads.
        destination_offsets: Device byte offsets for compact plane payloads.
        payload_bytes: Device payload lengths, excluding aligned padding.
        num_slots: Number of slot records in the batch.
        num_planes: Number of KV planes per slot.
        plane_scalars: Scalar count in each plane.
        tile_count: Number of codec tiles in each plane.
        tile_escape_capacity: Fixed scratch bytes reserved per tile.
        stream: CUDA stream ordering the copy.

    Async/thread-safety:
        The launch is asynchronous with respect to the host. Both buffers and
        metadata must remain alive until ``stream`` reaches completion.
    """
    external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    with external_stream:
        kernel(
            (num_slots * num_planes,),
            (_THREADS,),
            (
                cupy.asarray(source),
                cupy.asarray(destination),
                cupy.asarray(source_offsets),
                cupy.asarray(destination_offsets),
                cupy.asarray(payload_bytes),
                np.int32(num_slots),
                np.int32(num_planes),
                np.int32(plane_scalars),
                np.int32(tile_count),
                np.int32(tile_escape_capacity),
            ),
        )


def _online_compact_kernel(device: torch.device) -> object:
    """Return the compact kernel from the process-local CUDA module."""
    index = int(device.index or 0)
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        _online_fixed_packer_kernel(device)
        module = _ONLINE_CUDA_MODULE_CACHE[index]
    return module.get_function("compact_online_fixed")


def _online_fixed_layout_kernel(device: torch.device) -> object:
    """Return the device-resident fixed-layout planner kernel."""
    index = int(device.index or 0)
    cached = _ONLINE_LAYOUT_CACHE.get(index)
    if cached is not None:
        return cached
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        _online_fixed_packer_kernel(device)
        module = _ONLINE_CUDA_MODULE_CACHE[index]
    kernel = module.get_function("build_online_fixed_layout")
    _ONLINE_LAYOUT_CACHE[index] = kernel
    return kernel


def _online_compact_tiled_kernel(device: torch.device) -> object:
    """Return the single-read tile-local compaction kernel."""
    index = int(device.index or 0)
    module = _ONLINE_CUDA_MODULE_CACHE.get(index)
    if module is None:
        _online_fixed_packer_kernel(device)
        module = _ONLINE_CUDA_MODULE_CACHE[index]
    return module.get_function("compact_online_fixed_tiled")


def _fixed_envelope_geometry(
    *, slot_stride: int, num_planes: int, plane_scalars: int, max_tiles: int
) -> tuple[int, int, int] | None:
    """Return fixed plane bytes, escape capacity, and stored slot bytes."""
    payload_base = plane_scalars + (plane_scalars + 1) // 2 + 4 * (max_tiles + 1)
    available = slot_stride - IO_ALIGNMENT
    if available <= 0 or num_planes <= 0:
        return None
    plane_bytes = (available // num_planes // IO_ALIGNMENT) * IO_ALIGNMENT
    fixed_escape = plane_bytes - payload_base
    if fixed_escape <= 0:
        return None
    return plane_bytes, fixed_escape, IO_ALIGNMENT + num_planes * plane_bytes


def _fixed_single_read_geometry(
    *, slot_stride: int, num_planes: int, plane_scalars: int, max_tiles: int
) -> tuple[int, int, int] | None:
    """Return tile scratch capacity and the padded single-read envelope.

    The persisted envelope keeps the original ``fixed_escape`` capacity.  The
    internal scratch plane reserves one equal-sized segment per tile, rounded
    to the normal alignment, so the first pass can emit escape bytes without a
    second source read.  Any tile that exceeds its segment is restored raw.
    """
    envelope = _fixed_envelope_geometry(
        slot_stride=slot_stride,
        num_planes=num_planes,
        plane_scalars=plane_scalars,
        max_tiles=max_tiles,
    )
    if envelope is None or max_tiles <= 0:
        return None
    _plane_bytes, fixed_escape, _stored_length = envelope
    tile_capacity = (fixed_escape + max_tiles - 1) // max_tiles
    payload_base = plane_scalars + (plane_scalars + 1) // 2 + 4 * (max_tiles + 1)
    scratch_plane_bytes = align_up(payload_base + tile_capacity * max_tiles)
    scratch_slot_bytes = IO_ALIGNMENT + num_planes * scratch_plane_bytes
    return tile_capacity, scratch_plane_bytes, scratch_slot_bytes


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
    prefix_kernel = _online_prefix_kernel(kv_cache.device)
    count_kernel = _online_count_kernel(kv_cache.device)
    reduce_kernel = _online_reduce_kernel(kv_cache.device)
    fixed_kernel = _online_fixed_packer_kernel(kv_cache.device)
    variable_kernel = _online_variable_packer_kernel(kv_cache.device)
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
            counts = torch.zeros(
                max_slots_per_buffer * num_planes * max_tiles,
                dtype=torch.uint32,
                device=device,
            )
            lookup = torch.zeros(num_planes * 256, dtype=torch.uint8, device=device)
            output = torch.empty(
                int(kv_cache[0].nbytes), dtype=torch.uint8, device=device
            )
            warm_stream = torch.cuda.current_stream(device)
            _launch_online_prefix(
                prefix_kernel,
                counts=counts,
                prefixes=prefixes,
                num_rows=max_slots_per_buffer * num_planes,
                max_tiles=max_tiles,
                stream=warm_stream,
            )
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
                stream=warm_stream,
            )
            envelope = _fixed_envelope_geometry(
                slot_stride=int(kv_cache[0].nbytes),
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                max_tiles=max_tiles,
            )
            if envelope is None:
                raise ValueError("online packer cannot fit a fixed record envelope")
            plane_record_bytes, _fixed_escape_bytes, _stored_length = envelope
            single_read_geometry = _fixed_single_read_geometry(
                slot_stride=int(kv_cache[0].nbytes),
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                max_tiles=max_tiles,
            )
            if single_read_geometry is None:
                raise ValueError("online packer cannot fit single-read scratch")
            tile_escape_capacity, scratch_plane_bytes, scratch_slot_bytes = (
                single_read_geometry
            )
            # The production store path bounds one codec launch to a small
            # batch even when the staging lease can hold many more raw slots.
            # Warm that actual launch shape so the first live store does not
            # pay CUDA's large scratch allocation or a new grid specialization.
            warm_fixed_slots = min(max_slots_per_buffer, ONLINE_PACK_BATCH_SLOTS)
            fixed_ids = torch.zeros(warm_fixed_slots, dtype=torch.int32, device=device)
            fixed_output = torch.empty(
                warm_fixed_slots * scratch_slot_bytes,
                dtype=torch.uint8,
                device=device,
            )
            fixed_compact_output = torch.empty(
                warm_fixed_slots * int(kv_cache[0].nbytes),
                dtype=torch.uint8,
                device=device,
            )
            fixed_totals = torch.zeros(
                warm_fixed_slots * num_planes, dtype=torch.uint32, device=device
            )
            fixed_overflow = torch.zeros(
                warm_fixed_slots, dtype=torch.uint32, device=device
            )
            _launch_online_fixed_pack(
                fixed_kernel,
                kv_bits=kv_cache.view(torch.uint16).reshape(
                    int(kv_cache.shape[0]), num_planes, plane_scalars
                ),
                block_ids=fixed_ids,
                lookup=lookup,
                staging=fixed_output,
                totals=fixed_totals,
                overflow=fixed_overflow,
                num_slots=warm_fixed_slots,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                tile_scalars=tile_scalars,
                max_tiles=max_tiles,
                block_tokens=block_tokens,
                valid_token_count=warm_fixed_slots * block_tokens,
                plane_record_bytes=scratch_plane_bytes,
                tile_escape_capacity=tile_escape_capacity,
                scratch_slot_stride=scratch_slot_bytes,
                stream=warm_stream,
            )
            compact_payload_base = (
                plane_scalars + (plane_scalars + 1) // 2 + 4 * (max_tiles + 1)
            )
            compact_rows = warm_fixed_slots * num_planes
            compact_source_offsets = torch.tensor(
                [
                    slot * scratch_slot_bytes
                    + IO_ALIGNMENT
                    + plane * scratch_plane_bytes
                    for slot in range(warm_fixed_slots)
                    for plane in range(num_planes)
                ],
                dtype=torch.int64,
                device=device,
            )
            compact_destination_offsets = torch.tensor(
                [
                    slot * int(kv_cache[0].nbytes)
                    + IO_ALIGNMENT
                    + plane * plane_record_bytes
                    for slot in range(warm_fixed_slots)
                    for plane in range(num_planes)
                ],
                dtype=torch.int64,
                device=device,
            )
            compact_lengths = torch.full(
                (compact_rows,), compact_payload_base, dtype=torch.int64, device=device
            )
            _launch_online_compact_tiled(
                _online_compact_tiled_kernel(device),
                source=fixed_output,
                destination=fixed_compact_output,
                source_offsets=compact_source_offsets,
                destination_offsets=compact_destination_offsets,
                payload_bytes=compact_lengths,
                num_slots=warm_fixed_slots,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                tile_count=max_tiles,
                tile_escape_capacity=tile_escape_capacity,
                stream=warm_stream,
            )
            variable_offsets = torch.zeros(
                max_slots_per_buffer * num_planes,
                dtype=torch.int64,
                device=device,
            )
            variable_modes = torch.zeros(
                max_slots_per_buffer, dtype=torch.uint32, device=device
            )
            variable_totals = torch.zeros(
                max_slots_per_buffer * num_planes,
                dtype=torch.uint32,
                device=device,
            )
            with cupy.cuda.ExternalStream(warm_stream.cuda_stream):
                count_kernel(
                    (max_slots_per_buffer * num_planes,),
                    (_ONLINE_THREADS,),
                    (
                        cupy.asarray(
                            kv_cache.view(torch.uint16).reshape(
                                int(kv_cache.shape[0]), num_planes, plane_scalars
                            )
                        ),
                        cupy.asarray(ids),
                        cupy.asarray(lookup),
                        cupy.asarray(counts),
                        np.int32(max_slots_per_buffer),
                        np.int32(num_planes),
                        np.int32(plane_scalars),
                        np.int32(tile_scalars),
                        np.int32(max_tiles),
                        np.int32(max_tiles),
                        np.int32(block_tokens),
                        np.int32(max_slots_per_buffer * block_tokens),
                        np.int32(1),
                    ),
                )
            _launch_online_reduce(
                reduce_kernel,
                counts=counts,
                totals=variable_totals,
                num_rows=max_slots_per_buffer * num_planes,
                max_tiles=max_tiles,
                stream=warm_stream,
            )
            _launch_online_variable_pack(
                variable_kernel,
                kv_bits=kv_cache.view(torch.uint16).reshape(
                    int(kv_cache.shape[0]), num_planes, plane_scalars
                ),
                block_ids=ids,
                lookup=lookup,
                counts=counts,
                raw_modes=variable_modes,
                staging=output,
                plane_offsets=variable_offsets,
                num_slots=max_slots_per_buffer,
                num_planes=num_planes,
                plane_scalars=plane_scalars,
                tile_scalars=tile_scalars,
                max_tiles=max_tiles,
                block_tokens=block_tokens,
                valid_token_count=max_slots_per_buffer * block_tokens,
                stream=warm_stream,
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
        self._fixed_kernel = _ONLINE_FIXED_PACK_CACHE.get(
            (int(kv_cache.device.index or 0),)
        )
        self._variable_kernel = _ONLINE_VARIABLE_PACK_CACHE.get(
            int(kv_cache.device.index or 0)
        )
        self._count_kernel = _ONLINE_COUNT_CACHE.get(int(kv_cache.device.index or 0))
        self._reduce_kernel = _ONLINE_REDUCE_CACHE.get(int(kv_cache.device.index or 0))
        self._diagnostic_emitted = False
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        envelope = _fixed_envelope_geometry(
            slot_stride=self._geometry.slot_size,
            num_planes=self._num_planes,
            plane_scalars=self._plane_scalars,
            max_tiles=max_tiles,
        )
        if envelope is None or max_tiles > 1024:
            self._fixed_plane_record_bytes = 0
            self._fixed_escape_bytes = 0
            self._fixed_stored_length = 0
            self._fixed_tile_escape_capacity = 0
            self._fixed_scratch_plane_record_bytes = 0
            self._fixed_scratch_slot_stride = 0
        else:
            (
                self._fixed_plane_record_bytes,
                self._fixed_escape_bytes,
                self._fixed_stored_length,
            ) = envelope
            single_read_geometry = _fixed_single_read_geometry(
                slot_stride=self._geometry.slot_size,
                num_planes=self._num_planes,
                plane_scalars=self._plane_scalars,
                max_tiles=max_tiles,
            )
            if single_read_geometry is None:
                self._fixed_tile_escape_capacity = 0
                self._fixed_scratch_plane_record_bytes = 0
                self._fixed_scratch_slot_stride = 0
            else:
                (
                    self._fixed_tile_escape_capacity,
                    self._fixed_scratch_plane_record_bytes,
                    self._fixed_scratch_slot_stride,
                ) = single_read_geometry
        prefix_count = self._max_slots * self._num_planes * (max_tiles + 1)
        count_count = self._max_slots * self._num_planes * max_tiles
        total_count = self._max_slots * self._num_planes
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
            self._host_plane_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._host_slot_offsets = torch.empty(
                self._max_slots,
                dtype=torch.int64,
                pin_memory=True,
            )
            # Planning stays device-resident until the compact per-plane
            # totals are known.  The host only needs those totals to build
            # variable-length descriptors; the full tile prefix table is
            # consumed by the pack kernel from ``_device_prefixes``.
            self._host_totals = torch.empty(
                total_count, dtype=torch.uint32, pin_memory=True
            )
            self._host_overflow = torch.empty(
                self._max_slots, dtype=torch.uint32, pin_memory=True
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
            self._device_plane_offsets = torch.empty(
                self._max_slots * self._num_planes,
                dtype=torch.int64,
                device=self._device,
            )
            self._device_slot_offsets = torch.empty(
                self._max_slots,
                dtype=torch.int64,
                device=self._device,
            )
            self._device_prefixes = torch.empty(
                prefix_count, dtype=torch.uint32, device=self._device
            )
            self._device_counts = torch.empty(
                count_count, dtype=torch.uint32, device=self._device
            )
            self._device_totals = torch.empty(
                total_count, dtype=torch.uint32, device=self._device
            )
            self._device_overflow = torch.empty(
                self._max_slots, dtype=torch.uint32, device=self._device
            )
            # The production store path caps a codec launch at
            # ``ONLINE_PACK_BATCH_SLOTS``.  Allocate that fixed-scratch
            # envelope while the packer is constructed, before request
            # traffic starts.  A lazy allocation here can call CUDA's backing
            # allocator on the first completed prefill and turn an otherwise
            # asynchronous store into a visible TTFT spike (the Qwen3
            # geometry needs about 302 MiB for sixteen slots).
            fixed_scratch_slots = min(self._max_slots, ONLINE_PACK_BATCH_SLOTS)
            if self._fixed_scratch_slot_stride > 0 and fixed_scratch_slots > 0:
                self._fixed_scratch = torch.empty(
                    fixed_scratch_slots * self._fixed_scratch_slot_stride,
                    dtype=torch.uint8,
                    device=self._device,
                )
                self._fixed_scratch_slots = fixed_scratch_slots
            else:
                self._fixed_scratch = None
                self._fixed_scratch_slots = 0
            compact_count = self._max_slots * self._num_planes
            self._host_compact_source = torch.empty(
                compact_count, dtype=torch.int64, pin_memory=True
            )
            self._host_compact_destination = torch.empty(
                compact_count, dtype=torch.int64, pin_memory=True
            )
            self._host_compact_bytes = torch.empty(
                compact_count, dtype=torch.int64, pin_memory=True
            )
            self._device_compact_source = torch.empty(
                compact_count, dtype=torch.int64, device=self._device
            )
            self._device_compact_destination = torch.empty(
                compact_count, dtype=torch.int64, device=self._device
            )
            self._device_compact_bytes = torch.empty(
                compact_count, dtype=torch.int64, device=self._device
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
        if self._fixed_kernel is not None and self._fixed_escape_bytes > 0:
            # The fixed writer performs count, prefix construction, and payload
            # emission in one launch.  It then compacts only the live payload
            # bytes, which is faster on the serving geometry than the direct
            # variable writer's separate count/reduce and write launches.  The
            # fixed envelope remains an internal scratch representation, so
            # transfer bytes and on-disk records stay variable-length.
            return self._pack_fixed_into(
                staging=staging,
                block_ids=block_ids,
                logical_slots=logical_slots,
                slot_stride=slot_stride,
                stream=stream,
                valid_token_count=effective_valid_tokens,
            )
        if (
            self._variable_kernel is not None
            and self._count_kernel is not None
            and self._reduce_kernel is not None
            and max_tiles <= 1024
        ):
            return self._pack_variable_into(
                staging=staging,
                block_ids=block_ids,
                logical_slots=logical_slots,
                slot_stride=slot_stride,
                stream=stream,
                valid_token_count=effective_valid_tokens,
            )
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
                        (_ONLINE_THREADS,),
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
                device_prefixes = self._device_prefixes[
                    : slot_count * self._num_planes * (max_tiles + 1)
                ]
                _launch_online_prefix(
                    _online_prefix_kernel(self._device),
                    counts=device_counts,
                    prefixes=device_prefixes,
                    num_rows=slot_count * self._num_planes,
                    max_tiles=max_tiles,
                    stream=stream,
                )
                self._host_totals[: slot_count * self._num_planes].copy_(
                    device_prefixes.view(slot_count, self._num_planes, max_tiles + 1)[
                        ..., -1
                    ].reshape(-1),
                    non_blocking=True,
                )
            stream.synchronize()
            count_sync_ms = (time.perf_counter() - count_started) * 1000
            totals_host = self._host_totals[: slot_count * self._num_planes].numpy()
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
                    total_index = slot_index * self._num_planes + plane
                    escape_count = int(totals_host[total_index])
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

    def _pack_variable_into(
        self,
        *,
        staging: torch.Tensor,
        block_ids: list[int],
        logical_slots: list[int],
        slot_stride: int,
        stream: torch.cuda.Stream,
        valid_token_count: int,
    ) -> list[OnlinePackedSlot]:
        """Count and emit packed records directly at their final offsets.

        Args:
            staging: CUDA byte buffer receiving variable-length records.
            block_ids: Physical source blocks in logical order.
            logical_slots: DaseR slot IDs written into record headers.
            slot_stride: Raw bytes reserved for one source slot.
            stream: Store CUDA stream ordering count, write, and headers.
            valid_token_count: Number of valid prompt tokens in this batch.

        Returns:
            Packed slot metadata with raw fallback for records that do not fit.

        Async/thread-safety:
            Called under the store pipeline staging lock. The single stream
            synchronization publishes totals to the host. The caller owns the
            final synchronization that publishes staging bytes to IPC.
        """
        slot_count = len(block_ids)
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        row_count = slot_count * self._num_planes
        with torch.cuda.device(self._device):
            host_ids = self._host_block_ids[:slot_count]
            host_ids.copy_(torch.as_tensor(block_ids, dtype=torch.int32))
            device_ids = self._device_block_ids[:slot_count]
            device_counts = self._device_counts[: row_count * max_tiles]
            device_totals = self._device_totals[:row_count]
            count_kernel = self._count_kernel or _online_count_kernel(self._device)
            reduce_kernel = self._reduce_kernel or _online_reduce_kernel(self._device)
            variable_kernel = self._variable_kernel or _online_variable_packer_kernel(
                self._device
            )
            full_valid = int(
                valid_token_count == slot_count * self._geometry.block_tokens
            )
            with torch.cuda.stream(stream):
                device_ids.copy_(host_ids, non_blocking=True)
                external_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
                with external_stream:
                    count_kernel(
                        (row_count,),
                        (_ONLINE_THREADS,),
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
                            np.int32(valid_token_count),
                            np.int32(full_valid),
                        ),
                    )
                _launch_online_reduce(
                    reduce_kernel,
                    counts=device_counts,
                    totals=device_totals,
                    num_rows=row_count,
                    max_tiles=max_tiles,
                    stream=stream,
                )
                self._host_totals[:row_count].copy_(device_totals, non_blocking=True)
            stream.synchronize()

            totals_host = self._host_totals[:row_count].numpy()
            plans: list[tuple[SlotMode, int, tuple[PlaneDescriptor, ...]]] = []
            for slot_index in range(slot_count):
                cursor = IO_ALIGNMENT
                descriptors: list[PlaneDescriptor] = []
                for plane in range(self._num_planes):
                    escape_count = int(
                        totals_host[slot_index * self._num_planes + plane]
                    )
                    symbol_length = (self._plane_scalars + 1) // 2
                    prefix_length = 4 * (max_tiles + 1)
                    payload_length = (
                        self._plane_scalars
                        + symbol_length
                        + prefix_length
                        + escape_count
                    )
                    record_offset = cursor
                    record_length = align_up(payload_length)
                    escape_offset = (
                        record_offset
                        + self._plane_scalars
                        + symbol_length
                        + prefix_length
                    )
                    descriptors.append(
                        PlaneDescriptor(
                            layer=plane // 2,
                            kv=plane % 2,
                            scalar_count=self._plane_scalars,
                            tile_count=max_tiles,
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

            # Place records back-to-back in the exported staging buffer.  Every
            # record length is 4 KiB aligned and bounded by ``slot_stride``;
            # raw fallback therefore consumes exactly one raw slot while
            # compressed records leave their unused envelope bytes available to
            # the next slot.  The decoder receives these bases through its
            # per-slot metadata, so descriptors remain relative to each base.
            slot_bases = np.empty(slot_count, dtype=np.int64)
            staging_cursor = 0
            for index, (_mode, length, _descriptors) in enumerate(plans):
                slot_bases[index] = staging_cursor
                staging_cursor += length
            if staging_cursor > slot_count * slot_stride:
                raise RuntimeError("online packed staging layout exceeds capacity")

            host_modes = self._host_overflow[:slot_count].numpy()
            host_modes.fill(0)
            host_plane_offsets = self._host_plane_offsets[:row_count].numpy()
            host_plane_offsets.fill(0)
            host_slot_offsets = self._host_slot_offsets[:slot_count].numpy()
            host_slot_offsets[:] = slot_bases
            for slot_index, (mode, _length, descriptors) in enumerate(plans):
                if mode is SlotMode.COMPRESSED:
                    slot_base = int(slot_bases[slot_index])
                    for plane, descriptor in enumerate(descriptors):
                        host_plane_offsets[slot_index * self._num_planes + plane] = (
                            slot_base + descriptor.record_offset
                        )
                else:
                    host_modes[slot_index] = 1

            device_plane_offsets = self._device_plane_offsets[:row_count]
            device_modes = self._device_overflow[:slot_count]
            with torch.cuda.stream(stream):
                device_plane_offsets.copy_(
                    self._host_plane_offsets[:row_count], non_blocking=True
                )
                device_slot_offsets = self._device_slot_offsets[:slot_count]
                device_slot_offsets.copy_(
                    self._host_slot_offsets[:slot_count], non_blocking=True
                )
                device_modes.copy_(self._host_overflow[:slot_count], non_blocking=True)
                if bool(np.any(host_modes)):
                    _launch_online_raw_restore(
                        _online_raw_restore_kernel(self._device),
                        kv_bits=self._kv_bits,
                        block_ids=device_ids,
                        overflow=device_modes,
                        staging=staging,
                        slot_offsets=device_slot_offsets,
                        num_slots=slot_count,
                        num_planes=self._num_planes,
                        plane_scalars=self._plane_scalars,
                        slot_stride=slot_stride,
                        stream=stream,
                    )
                _launch_online_variable_pack(
                    variable_kernel,
                    kv_bits=self._kv_bits,
                    block_ids=device_ids,
                    lookup=self._lookup,
                    counts=device_counts,
                    raw_modes=device_modes,
                    staging=staging,
                    plane_offsets=device_plane_offsets,
                    num_slots=slot_count,
                    num_planes=self._num_planes,
                    plane_scalars=self._plane_scalars,
                    tile_scalars=self._tile_scalars,
                    max_tiles=max_tiles,
                    block_tokens=self._geometry.block_tokens,
                    valid_token_count=valid_token_count,
                    stream=stream,
                )
                for index, (mode, length, descriptors) in enumerate(plans):
                    if mode is not SlotMode.COMPRESSED:
                        continue
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
                    header_tensor = torch.frombuffer(
                        bytearray(header), dtype=torch.uint8
                    )
                    base = int(slot_bases[index])
                    staging[base : base + IO_ALIGNMENT].copy_(
                        header_tensor, non_blocking=True
                    )
        compressed_count = sum(mode is SlotMode.COMPRESSED for mode, _, _ in plans)
        packed_bytes = sum(
            length
            for mode, length, _descriptors in plans
            if mode is SlotMode.COMPRESSED
        )
        logger.debug(
            "[PACK] direct variable slots=%d compressed=%d packed_bytes=%d overflow=%d",
            slot_count,
            compressed_count,
            packed_bytes,
            slot_count - compressed_count,
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

    def _pack_fixed_into(
        self,
        *,
        staging: torch.Tensor,
        block_ids: list[int],
        logical_slots: list[int],
        slot_stride: int,
        stream: torch.cuda.Stream,
        valid_token_count: int,
    ) -> list[OnlinePackedSlot]:
        """Pack records through a fixed scratch envelope and compact them.

        Args:
            staging: CUDA byte buffer receiving slot records.
            block_ids: Physical source blocks in logical order.
            logical_slots: DaseR slot IDs written into record headers.
            slot_stride: Raw bytes reserved for one source slot.
            stream: CUDA stream ordering source reads and staging writes.
            valid_token_count: Number of valid prompt tokens in this batch.

        Returns:
            Packed slot metadata with raw mode for overflow records.  Compressed
            records retain their actual variable-length payloads in ``staging``.

        Async/thread-safety:
            Called under the store pipeline staging lock. The stream is
            synchronized once after this method returns by the caller, after
            the host has planned variable record spans and headers.
        """
        slot_count = len(block_ids)
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        row_count = slot_count * self._num_planes
        with torch.cuda.device(self._device):
            # The fixed writer is intentionally isolated from the exported
            # staging allocation.  Allocate only for the active batch because
            # ``max_slots_per_buffer`` can describe a much larger pool.
            scratch_slot_stride = self._fixed_scratch_slot_stride
            scratch_plane_record_bytes = self._fixed_scratch_plane_record_bytes
            tile_escape_capacity = self._fixed_tile_escape_capacity
            if (
                scratch_slot_stride <= 0
                or scratch_plane_record_bytes <= 0
                or tile_escape_capacity <= 0
            ):
                raise RuntimeError("single-read fixed scratch geometry is invalid")
            if (
                self._fixed_scratch is None
                or self._fixed_scratch_slots < slot_count
                or self._fixed_scratch.numel() < slot_count * scratch_slot_stride
            ):
                self._fixed_scratch = torch.empty(
                    slot_count * scratch_slot_stride,
                    dtype=torch.uint8,
                    device=self._device,
                )
                self._fixed_scratch_slots = slot_count
            scratch = self._fixed_scratch[: slot_count * scratch_slot_stride]

            host_ids = self._host_block_ids[:slot_count]
            host_ids.copy_(torch.as_tensor(block_ids, dtype=torch.int32))
            device_ids = self._device_block_ids[:slot_count]
            device_totals = self._device_totals[:row_count]
            device_overflow = self._device_overflow[:slot_count]
            device_sources = self._device_compact_source[:row_count]
            device_destinations = self._device_compact_destination[:row_count]
            device_bytes = self._device_compact_bytes[:row_count]
            device_slot_offsets = self._device_slot_offsets[:slot_count]
            with torch.cuda.stream(stream):
                device_ids.copy_(host_ids, non_blocking=True)
                # Keep initialization on the same stream as the writer.  A
                # zero launched on the worker's current stream could race the
                # fixed kernel when the store stream is independent.
                device_overflow.zero_()
                _launch_online_fixed_pack(
                    self._fixed_kernel,
                    kv_bits=self._kv_bits,
                    block_ids=device_ids,
                    lookup=self._lookup,
                    staging=scratch,
                    totals=device_totals,
                    overflow=device_overflow,
                    num_slots=slot_count,
                    num_planes=self._num_planes,
                    plane_scalars=self._plane_scalars,
                    tile_scalars=self._tile_scalars,
                    max_tiles=max_tiles,
                    block_tokens=self._geometry.block_tokens,
                    valid_token_count=valid_token_count,
                    plane_record_bytes=scratch_plane_record_bytes,
                    tile_escape_capacity=tile_escape_capacity,
                    scratch_slot_stride=scratch_slot_stride,
                    stream=stream,
                )
                # Keep variable-length layout planning on the same CUDA
                # stream as the writer.  This lets compaction start
                # immediately after the single-read kernel instead of waiting
                # for Python to consume totals and copy descriptor arrays back
                # to the device.
                _launch_online_fixed_layout(
                    _online_fixed_layout_kernel(self._device),
                    totals=device_totals,
                    overflow=device_overflow,
                    source_offsets=device_sources,
                    destination_offsets=device_destinations,
                    payload_bytes=device_bytes,
                    slot_offsets=device_slot_offsets,
                    num_slots=slot_count,
                    num_planes=self._num_planes,
                    plane_scalars=self._plane_scalars,
                    max_tiles=max_tiles,
                    slot_stride=slot_stride,
                    scratch_plane_record_bytes=scratch_plane_record_bytes,
                    scratch_slot_stride=scratch_slot_stride,
                    stream=stream,
                )
                self._host_totals[:row_count].copy_(device_totals, non_blocking=True)
                self._host_overflow[:slot_count].copy_(
                    device_overflow, non_blocking=True
                )
                # The raw restore kernel checks the device fallback flags, so
                # it can be launched unconditionally while the compact kernel
                # skips rows whose payload length is zero.
                _launch_online_raw_restore(
                    _online_raw_restore_kernel(self._device),
                    kv_bits=self._kv_bits,
                    block_ids=device_ids,
                    overflow=device_overflow,
                    staging=staging,
                    slot_offsets=device_slot_offsets,
                    num_slots=slot_count,
                    num_planes=self._num_planes,
                    plane_scalars=self._plane_scalars,
                    slot_stride=slot_stride,
                    stream=stream,
                )
                _launch_online_compact_tiled(
                    _online_compact_tiled_kernel(self._device),
                    source=scratch,
                    destination=staging,
                    source_offsets=device_sources,
                    destination_offsets=device_destinations,
                    payload_bytes=device_bytes,
                    num_slots=slot_count,
                    num_planes=self._num_planes,
                    plane_scalars=self._plane_scalars,
                    tile_count=max_tiles,
                    tile_escape_capacity=tile_escape_capacity,
                    stream=stream,
                )

            # Totals and fallback flags are the only metadata needed by the
            # host to construct self-describing headers.  The device has
            # already laid out and compacted all payload bytes by this point.
            stream.synchronize()
            overflow_host = self._host_overflow[:slot_count].numpy().copy()
            totals_host = self._host_totals[:row_count].numpy()
            plans: list[tuple[SlotMode, int, tuple[PlaneDescriptor, ...]]] = []
            for slot_index in range(slot_count):
                if overflow_host[slot_index]:
                    plans.append((SlotMode.RAW, slot_stride, tuple()))
                    continue
                cursor = IO_ALIGNMENT
                descriptors: list[PlaneDescriptor] = []
                for plane in range(self._num_planes):
                    escape_count = int(
                        totals_host[slot_index * self._num_planes + plane]
                    )
                    symbol_length = (self._plane_scalars + 1) // 2
                    prefix_length = 4 * (max_tiles + 1)
                    payload_length = (
                        self._plane_scalars
                        + symbol_length
                        + prefix_length
                        + escape_count
                    )
                    record_offset = cursor
                    record_length = align_up(payload_length)
                    escape_offset = (
                        record_offset
                        + self._plane_scalars
                        + symbol_length
                        + prefix_length
                    )
                    descriptors.append(
                        PlaneDescriptor(
                            layer=plane // 2,
                            kv=plane % 2,
                            scalar_count=self._plane_scalars,
                            tile_count=max_tiles,
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
                    raise RuntimeError("device and host fixed layout planners disagree")
                plans.append((SlotMode.COMPRESSED, cursor, tuple(descriptors)))

            # Recompute the compact slot bases for the returned IPC spans. The
            # device planner uses the same slot-major scan; checking capacity
            # here keeps malformed geometry from exposing an out-of-bounds
            # mapping even though payload compaction has already completed.
            slot_bases = np.empty(slot_count, dtype=np.int64)
            staging_cursor = 0
            for index, (_mode, length, _descriptors) in enumerate(plans):
                slot_bases[index] = staging_cursor
                staging_cursor += length
            if staging_cursor > slot_count * slot_stride:
                raise RuntimeError("online packed staging layout exceeds capacity")
            self._host_slot_offsets[:slot_count].numpy()[:] = slot_bases
            with torch.cuda.stream(stream):
                for index, (mode, length, descriptors) in enumerate(plans):
                    if mode is not SlotMode.COMPRESSED:
                        continue
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
                    # Keep the host source pinned and enqueue the tiny header
                    # copy on the pack stream.  The previous fixed-scratch
                    # path passed a pageable bytearray with ``non_blocking``
                    # disabled, which made every slot header a synchronous
                    # host-to-device handoff after the codec kernels had
                    # already completed.  Reusing the persistent pinned ring
                    # preserves the header bytes while allowing the outer
                    # staging event to wait once for all headers together.
                    self._host_headers[index].copy_(
                        torch.frombuffer(bytearray(header), dtype=torch.uint8)
                    )
                    base = int(slot_bases[index])
                    staging[base : base + IO_ALIGNMENT].copy_(
                        self._host_headers[index], non_blocking=True
                    )
        compressed_count = sum(mode is SlotMode.COMPRESSED for mode, _, _ in plans)
        packed_bytes = sum(
            length
            for mode, length, _descriptors in plans
            if mode is SlotMode.COMPRESSED
        )
        logger.debug(
            "[PACK] fixed scratch slots=%d compressed=%d packed_bytes=%d "
            "scratch_bytes=%d escape_capacity=%d overflow=%d",
            slot_count,
            compressed_count,
            packed_bytes,
            slot_count * slot_stride,
            self._fixed_escape_bytes,
            slot_count - compressed_count,
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
        # The decoder metadata is small, but this method runs once for every
        # cache-hit batch.  Per-element tensor assignment takes the Python
        # interpreter lock for every slot and creates a host-side dispatch
        # point before the three asynchronous H2D copies.  Convert each list
        # once into its declared dtype and copy through the contiguous NumPy
        # view of the persistent pinned tensor instead.
        np.copyto(
            ring.host_offsets[:slot_count].numpy(),
            np.asarray(staging_offsets, dtype=np.int64),
        )
        np.copyto(
            ring.host_blocks[:slot_count].numpy(),
            np.asarray(block_ids, dtype=np.int32),
        )
        np.copyto(
            ring.host_modes[:slot_count].numpy(),
            np.asarray(modes, dtype=np.int32),
        )
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
