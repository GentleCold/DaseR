# SPDX-License-Identifier: Apache-2.0
"""TileLang RoPE delta operators for cached K blocks."""

# Do not enable ``from __future__ import annotations`` in this module.
# TileLang reads function annotations at kernel-build time and needs concrete
# closure values for tensor shapes.

# Standard
from collections.abc import Callable
from typing import Any

# Third Party
import torch

TileLangFn = Callable[[torch.Tensor, int, float], None]
TileLangTableFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]
TileLangRestoreTableFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    None,
]
TileLangCacheKey = tuple[Any, int, int, bool]
TileLangKVCacheKey = tuple[Any, tuple[int, ...], int, bool]
TileLangRestoreCacheKey = tuple[Any, tuple[int, ...], int, bool]

_kernel_cache: dict[TileLangCacheKey, TileLangFn] = {}
_kv_table_kernel_cache: dict[TileLangKVCacheKey, TileLangTableFn] = {}
_restore_table_kernel_cache: dict[TileLangRestoreCacheKey, TileLangRestoreTableFn] = {}


def clear_rope_apply_cache() -> None:
    """Clear cached RoPE operator kernels.

    Returns:
        None.

    Async/thread-safety:
        Intended for tests and benchmarks. Do not call while another thread is
        applying RoPE.
    """
    _kernel_cache.clear()
    _kv_table_kernel_cache.clear()
    _restore_table_kernel_cache.clear()


def apply_rope_delta_to_key_block(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Rotate an already-RoPE'd K block by a relative position delta.

    Args:
        key_block: Contiguous K cache block with shape
            ``[..., block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. ``key_block`` is modified in place.

    Raises:
        ValueError: if ``rotary_dim`` exceeds the head dimension, or the block
            is not a contiguous CUDA tensor with the expected shape.

    Async/thread-safety:
        Launches a CUDA kernel on the current stream. The kernel cache is
        process-wide and safe for normal single worker-thread use.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if key_block.shape[-1] < rotary_dim:
        raise ValueError("rotary_dim must not exceed head_dim")
    if key_block.device.type != "cuda":
        raise ValueError("TileLang RoPE apply requires a CUDA tensor")
    if key_block.dim() < 3:
        raise ValueError("TileLang RoPE apply requires at least 3 dimensions")
    if rotary_dim % 2 != 0:
        raise ValueError("TileLang RoPE apply requires an even rotary_dim")
    if not key_block.is_contiguous():
        raise ValueError("TileLang RoPE apply requires a contiguous key block")

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


def apply_rope_delta_to_kv_key_block(
    kv_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Rotate K entries inside a full KV staging block by a RoPE delta.

    Args:
        kv_block: Full KV tensor with shape
            ``[blocks, layers, 2, block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. Only the key slice ``kv_block[:, :, 0]`` is modified in place.

    Raises:
        ValueError: if ``rotary_dim`` exceeds the head dimension.

    Async/thread-safety:
        Launches CUDA work on the current PyTorch stream. The cosine/sine
        tables are built once per call and passed to the TileLang table kernel.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if kv_block.shape[-1] < rotary_dim:
        raise ValueError("rotary_dim must not exceed head_dim")

    cos_table, sin_table = build_rope_delta_tables(
        kv_block.device,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
    )
    apply_rope_delta_to_kv_key_block_table(
        kv_block,
        cos_table=cos_table,
        sin_table=sin_table,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def apply_rope_delta_to_kv_key_block_table(
    kv_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Apply RoPE delta to K entries using precomputed cosine/sine tables.

    Args:
        kv_block: Contiguous KV tensor with shape
            ``[blocks, layers, 2, block_tokens, heads, head_dim]``.
        cos_table: Contiguous fp32 cosine table with ``rotary_dim // 2`` values.
        sin_table: Contiguous fp32 sine table with ``rotary_dim // 2`` values.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. Only ``kv_block[:, :, 0]`` is modified in place.

    Async/thread-safety:
        Launches one CUDA kernel on the current stream and uses a process-wide
        kernel cache for normal single worker-thread use.
    """
    if kv_block.device.type != "cuda":
        raise ValueError("TileLang table RoPE backend requires a CUDA tensor")
    if not kv_block.is_contiguous():
        raise ValueError("TileLang table RoPE backend requires contiguous staging")
    if kv_block.dim() != 6 or kv_block.shape[2] != 2:
        raise ValueError("TileLang table RoPE backend requires [blocks,layers,2,...]")
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(
            "TileLang table RoPE backend requires a positive even rotary_dim"
        )
    if kv_block.shape[-1] < rotary_dim:
        raise ValueError("rotary_dim must not exceed head_dim")
    expected = rotary_dim // 2
    if cos_table.shape != (expected,) or sin_table.shape != (expected,):
        raise ValueError("RoPE tables must have shape [rotary_dim // 2]")
    if (
        cos_table.device != kv_block.device
        or sin_table.device != kv_block.device
        or cos_table.dtype != torch.float32
        or sin_table.dtype != torch.float32
        or not cos_table.is_contiguous()
        or not sin_table.is_contiguous()
    ):
        raise ValueError("RoPE tables must be contiguous fp32 tensors on the KV device")

    kernel = _get_tilelang_kv_table_kernel(
        kv_block.dtype,
        tuple(int(dim) for dim in kv_block.shape),
        rotary_dim,
        is_neox_style,
    )
    kernel(kv_block, cos_table, sin_table)


def restore_cross_layer_kv_cache_table(
    staging_kv: torch.Tensor,
    dst_kv: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Copy staging KV to cross-layer KV cache using precomputed RoPE tables.

    Args:
        staging_kv: Contiguous source KV tensor with shape
            ``[blocks, layers, 2, block_tokens, heads, head_dim]``.
        dst_kv: Contiguous destination KV tensor with the same shape.
        cos_table: Contiguous fp32 cosine table with ``rotary_dim // 2`` values.
        sin_table: Contiguous fp32 sine table with ``rotary_dim // 2`` values.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. ``dst_kv`` receives copied V/non-rotary data and rotated K data.

    Async/thread-safety:
        Launches one CUDA kernel on the current stream and uses a process-wide
        kernel cache for normal single worker-thread use.
    """
    if staging_kv.device.type != "cuda" or dst_kv.device.type != "cuda":
        raise ValueError("TileLang table fused restore requires CUDA tensors")
    if not staging_kv.is_contiguous() or not dst_kv.is_contiguous():
        raise ValueError("TileLang table fused restore requires contiguous tensors")
    if staging_kv.shape != dst_kv.shape:
        raise ValueError("source and destination KV shapes must match")
    if staging_kv.dtype != dst_kv.dtype:
        raise ValueError("source and destination KV dtypes must match")
    if staging_kv.dim() != 6 or staging_kv.shape[2] != 2:
        raise ValueError("TileLang table fused restore requires [blocks,layers,2,...]")
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(
            "TileLang table fused restore requires a positive even rotary_dim"
        )
    if staging_kv.shape[-1] < rotary_dim:
        raise ValueError("rotary_dim must not exceed head_dim")
    expected = rotary_dim // 2
    if cos_table.shape != (expected,) or sin_table.shape != (expected,):
        raise ValueError("RoPE tables must have shape [rotary_dim // 2]")
    if (
        cos_table.device != staging_kv.device
        or sin_table.device != staging_kv.device
        or cos_table.dtype != torch.float32
        or sin_table.dtype != torch.float32
        or not cos_table.is_contiguous()
        or not sin_table.is_contiguous()
    ):
        raise ValueError("RoPE tables must be contiguous fp32 tensors on the KV device")

    kernel = _get_tilelang_restore_table_kernel(
        staging_kv.dtype,
        tuple(int(dim) for dim in staging_kv.shape),
        rotary_dim,
        is_neox_style,
    )
    kernel(staging_kv, dst_kv, cos_table, sin_table)


def build_rope_delta_tables(
    device: Any,
    delta: int,
    rope_base: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fp32 cosine/sine tables for a RoPE position delta.

    Args:
        device: CUDA device that will consume the tables.
        delta: relative RoPE position delta.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.

    Returns:
        ``(cos, sin)`` contiguous fp32 tensors with shape
        ``[rotary_dim // 2]``.

    Async/thread-safety:
        Allocates tensors on ``device`` and performs regular PyTorch tensor
        operations on the current stream.
    """
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    freqs = int(delta) * inv_freq
    return freqs.cos().contiguous(), freqs.sin().contiguous()


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


def _get_tilelang_restore_table_kernel(
    dtype: Any,
    shape: tuple[int, ...],
    rotary_dim: int,
    is_neox_style: bool,
) -> TileLangRestoreTableFn:
    """Return a cached fused restore kernel using precomputed RoPE tables."""
    key = (dtype, shape[1:], rotary_dim, is_neox_style)
    cached = _restore_table_kernel_cache.get(key)
    if cached is not None:
        return cached

    import tilelang

    kernel = tilelang.compile(
        _build_tilelang_restore_table_kernel(
            static_shape=shape[1:],
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
            dtype=_tilelang_dtype(dtype),
        ),
        target="cuda",
        execution_backend="cython",
    )
    _restore_table_kernel_cache[key] = kernel
    return kernel


def _get_tilelang_kv_table_kernel(
    dtype: Any,
    shape: tuple[int, ...],
    rotary_dim: int,
    is_neox_style: bool,
) -> TileLangTableFn:
    """Return a cached TileLang kernel using precomputed RoPE tables."""
    key = (dtype, shape[1:], rotary_dim, is_neox_style)
    cached = _kv_table_kernel_cache.get(key)
    if cached is not None:
        return cached

    import tilelang

    kernel = tilelang.compile(
        _build_tilelang_kv_table_kernel(
            static_shape=shape[1:],
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
            dtype=_tilelang_dtype(dtype),
        ),
        target="cuda",
        execution_backend="cython",
    )
    _kv_table_kernel_cache[key] = kernel
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


def _build_tilelang_restore_table_kernel(
    static_shape: tuple[int, ...],
    rotary_dim: int,
    is_neox_style: bool,
    dtype: str,
) -> object:
    """Build a fused restore kernel that reads precomputed trig tables."""
    import tilelang.language as T

    n_blocks = T.dynamic("N")
    layers = static_shape[0]
    block_tokens = static_shape[2]
    heads = static_shape[3]
    head_dim = static_shape[4]
    half = rotary_dim // 2
    groups_per_block = layers * block_tokens * heads
    non_rotary_dims = head_dim - rotary_dim
    units_per_group = head_dim + non_rotary_dims + half
    total_units = n_blocks * groups_per_block * units_per_group
    threads = 256

    @T.prim_func
    def main(
        staging_kv: T.Tensor((n_blocks, *static_shape), dtype),
        dst_kv: T.Tensor((n_blocks, *static_shape), dtype),
        cos_table: T.Tensor((half,), "float32"),
        sin_table: T.Tensor((half,), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(total_units, threads),
            threads=threads,
        ) as bx:
            for tx in T.Parallel(threads):
                unit_linear = bx * threads + tx
                if unit_linear < total_units:
                    group = unit_linear // units_per_group
                    op_idx = unit_linear - group * units_per_group
                    block_idx = group // groups_per_block
                    rem_0 = group - block_idx * groups_per_block
                    layer_idx = rem_0 // (block_tokens * heads)
                    rem_1 = rem_0 - layer_idx * block_tokens * heads
                    token_idx = rem_1 // heads
                    head_idx = rem_1 - token_idx * heads
                    if op_idx < head_dim:
                        dst_kv[
                            block_idx,
                            layer_idx,
                            1,
                            token_idx,
                            head_idx,
                            op_idx,
                        ] = staging_kv[
                            block_idx,
                            layer_idx,
                            1,
                            token_idx,
                            head_idx,
                            op_idx,
                        ]
                    elif op_idx < head_dim + non_rotary_dims:
                        dim_idx = rotary_dim + op_idx - head_dim
                        dst_kv[
                            block_idx,
                            layer_idx,
                            0,
                            token_idx,
                            head_idx,
                            dim_idx,
                        ] = staging_kv[
                            block_idx,
                            layer_idx,
                            0,
                            token_idx,
                            head_idx,
                            dim_idx,
                        ]
                    else:
                        pair = op_idx - head_dim - non_rotary_dims
                        if is_neox_style:
                            offset_1 = pair
                            offset_2 = pair + half
                        else:
                            offset_1 = pair * 2
                            offset_2 = offset_1 + 1
                        cos = cos_table[pair]
                        sin = sin_table[pair]
                        value_1 = T.cast(
                            staging_kv[
                                block_idx,
                                layer_idx,
                                0,
                                token_idx,
                                head_idx,
                                offset_1,
                            ],
                            T.float32,
                        )
                        value_2 = T.cast(
                            staging_kv[
                                block_idx,
                                layer_idx,
                                0,
                                token_idx,
                                head_idx,
                                offset_2,
                            ],
                            T.float32,
                        )
                        dst_kv[
                            block_idx,
                            layer_idx,
                            0,
                            token_idx,
                            head_idx,
                            offset_1,
                        ] = T.cast(value_1 * cos - value_2 * sin, dtype)
                        dst_kv[
                            block_idx,
                            layer_idx,
                            0,
                            token_idx,
                            head_idx,
                            offset_2,
                        ] = T.cast(value_2 * cos + value_1 * sin, dtype)

    return main


def _build_tilelang_kv_table_kernel(
    static_shape: tuple[int, ...],
    rotary_dim: int,
    is_neox_style: bool,
    dtype: str,
) -> object:
    """Build a full-KV RoPE kernel that reads precomputed trig tables."""
    import tilelang.language as T

    n_blocks = T.dynamic("N")
    layers = static_shape[0]
    block_tokens = static_shape[2]
    heads = static_shape[3]
    half = rotary_dim // 2
    groups_per_block = layers * block_tokens * heads
    total_pairs = n_blocks * groups_per_block * half
    threads = 256

    @T.prim_func
    def main(
        kv_block: T.Tensor((n_blocks, *static_shape), dtype),
        cos_table: T.Tensor((half,), "float32"),
        sin_table: T.Tensor((half,), "float32"),
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
                    block_idx = group // groups_per_block
                    rem_0 = group - block_idx * groups_per_block
                    layer_idx = rem_0 // (block_tokens * heads)
                    rem_1 = rem_0 - layer_idx * block_tokens * heads
                    token_idx = rem_1 // heads
                    head_idx = rem_1 - token_idx * heads
                    if is_neox_style:
                        offset_1 = pair
                        offset_2 = pair + half
                    else:
                        offset_1 = pair * 2
                        offset_2 = offset_1 + 1
                    cos = cos_table[pair]
                    sin = sin_table[pair]
                    value_1 = T.cast(
                        kv_block[
                            block_idx,
                            layer_idx,
                            0,
                            token_idx,
                            head_idx,
                            offset_1,
                        ],
                        T.float32,
                    )
                    value_2 = T.cast(
                        kv_block[
                            block_idx,
                            layer_idx,
                            0,
                            token_idx,
                            head_idx,
                            offset_2,
                        ],
                        T.float32,
                    )
                    kv_block[
                        block_idx,
                        layer_idx,
                        0,
                        token_idx,
                        head_idx,
                        offset_1,
                    ] = T.cast(value_1 * cos - value_2 * sin, dtype)
                    kv_block[
                        block_idx,
                        layer_idx,
                        0,
                        token_idx,
                        head_idx,
                        offset_2,
                    ] = T.cast(value_2 * cos + value_1 * sin, dtype)

    return main
