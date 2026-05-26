# SPDX-License-Identifier: Apache-2.0
"""TileLang RoPE delta kernels for cached K blocks."""

# Do not enable ``from __future__ import annotations`` in this module.
# TileLang reads function annotations at kernel-build time and needs concrete
# closure values for tensor shapes.

# Standard
from collections.abc import Callable
from typing import Any

# Third Party
import torch

TileLangFn = Callable[[torch.Tensor, int, float], None]
TileLangCacheKey = tuple[Any, int, int, bool]

_kernel_cache: dict[TileLangCacheKey, TileLangFn] = {}


def clear_tilelang_rope_cache() -> None:
    """Clear cached TileLang RoPE kernels.

    Returns:
        None.

    Async/thread-safety:
        Intended for tests and benchmarks. Do not call while another thread is
        applying RoPE with the TileLang backend.
    """
    _kernel_cache.clear()


def apply_rope_delta_to_key_block_tilelang(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Apply RoPE delta in place with a cached TileLang CUDA kernel.

    Args:
        key_block: Contiguous K cache block with shape
            ``[..., block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. ``key_block`` is modified in place.

    Async/thread-safety:
        Launches a CUDA kernel on the current stream. The kernel cache is
        process-wide and safe for normal single worker-thread use.
    """
    if not key_block.is_contiguous():
        raise ValueError("TileLang RoPE backend requires a contiguous key block")
    if key_block.device.type != "cuda":
        raise ValueError("TileLang RoPE backend requires a CUDA tensor")
    if key_block.dim() < 3:
        raise ValueError("TileLang RoPE backend requires at least 3 dimensions")
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError("TileLang RoPE backend requires a positive even rotary_dim")
    if key_block.shape[-1] < rotary_dim:
        raise ValueError("rotary_dim must not exceed head_dim")

    head_dim = int(key_block.shape[-1])
    n_groups = int(key_block.numel() // head_dim)
    flat = key_block.reshape(n_groups, head_dim)
    kernel = _get_tilelang_kernel(
        key_block.dtype,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    kernel(flat, int(delta), float(rope_base))


def _get_tilelang_kernel(
    dtype: Any,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> TileLangFn:
    """Return a cached dynamic-shape TileLang kernel for the requested layout."""
    key = (dtype, head_dim, rotary_dim, is_neox_style)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    import tilelang

    kernel = tilelang.compile(
        _build_tilelang_kernel(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
            dtype=_tilelang_dtype(dtype),
        ),
        target="cuda",
        execution_backend="cython",
    )
    _kernel_cache[key] = kernel
    return kernel


def _tilelang_dtype(dtype: Any) -> str:
    """Return the TileLang dtype string for a torch dtype."""
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"unsupported TileLang RoPE dtype: {dtype}")


def _build_tilelang_kernel(
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
    dtype: str,
) -> object:
    """Build a TileLang in-place RoPE delta kernel with dynamic batch extent."""
    import tilelang.language as T

    n_groups = T.dynamic("N")
    half = rotary_dim // 2
    total_pairs = n_groups * half
    threads = 256

    @T.prim_func
    def main(
        key_block: T.Tensor((n_groups, head_dim), dtype),
        delta: T.int32,
        rope_base: T.float32,
    ):
        with T.Kernel(
            T.ceildiv(total_pairs, threads),
            threads=threads,
        ) as bx:
            for tx in T.Parallel(threads):
                pair_linear = bx * threads + tx
                if pair_linear < total_pairs:
                    group = pair_linear // half
                    pair = pair_linear - group * half
                    angle = T.cast(delta, T.float32) / T.pow(
                        rope_base,
                        T.cast(pair * 2, T.float32) / T.cast(rotary_dim, T.float32),
                    )
                    cos = T.cos(angle)
                    sin = T.sin(angle)
                    if is_neox_style:
                        offset_1 = pair
                        offset_2 = pair + half
                    else:
                        offset_1 = pair * 2
                        offset_2 = offset_1 + 1
                    value_1 = T.cast(key_block[group, offset_1], T.float32)
                    value_2 = T.cast(key_block[group, offset_2], T.float32)
                    key_block[group, offset_1] = T.cast(
                        value_1 * cos - value_2 * sin,
                        dtype,
                    )
                    key_block[group, offset_2] = T.cast(
                        value_2 * cos + value_1 * sin,
                        dtype,
                    )

    return main
