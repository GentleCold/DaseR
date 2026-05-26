# SPDX-License-Identifier: Apache-2.0
"""Production RoPE delta operators for cached K blocks."""

from __future__ import annotations

# Third Party
import torch

# First Party
from daser.ops.rope_apply_tilelang import (
    apply_rope_delta_to_key_block_tilelang,
    apply_rope_delta_to_kv_key_block_table_tilelang,
    clear_tilelang_rope_cache,
)

RopeApplyBackend = str


def clear_rope_apply_cache() -> None:
    """Clear cached RoPE operator kernels.

    Returns:
        None.

    Async/thread-safety:
        Intended for tests and benchmarks. Do not call while another thread is
        applying RoPE.
    """
    clear_tilelang_rope_cache()


def clear_rope_apply_compile_cache() -> None:
    """Compatibility alias for clearing cached RoPE kernels.

    Returns:
        None.

    Async/thread-safety:
        Intended for existing benchmarks and tests. There is no production
        torch.compile backend.
    """
    clear_rope_apply_cache()


def apply_rope_delta_to_key_block(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
    backend: RopeApplyBackend = "tilelang",
) -> None:
    """Rotate an already-RoPE'd K block by a relative position delta.

    Args:
        key_block: K cache block with shape ``[..., block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.
        backend: only ``"tilelang"`` and ``"auto"`` are accepted. ``"auto"``
            maps to the TileLang production path.

    Returns:
        None. ``key_block`` is modified in place.

    Raises:
        ValueError: if ``backend`` is not a production backend.
        ImportError, RuntimeError: if TileLang is unavailable or kernel launch
            fails.

    Async/thread-safety:
        Launches CUDA work on the current PyTorch stream. TileLang's kernel
        cache is process-wide and intended for normal single worker-thread use.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if key_block.shape[-1] < rotary_dim:
        return
    if backend not in ("auto", "tilelang"):
        raise ValueError(f"unknown RoPE apply backend: {backend}")

    apply_rope_delta_to_key_block_tilelang(
        key_block,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def apply_rope_delta_to_kv_key_block(
    kv_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
    backend: RopeApplyBackend = "tilelang",
) -> None:
    """Rotate K entries inside a full KV staging block by a RoPE delta.

    Args:
        kv_block: Full KV tensor with shape
            ``[blocks, layers, 2, block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.
        backend: only ``"tilelang"`` and ``"auto"`` are accepted. ``"auto"``
            maps to the TileLang production path.

    Returns:
        None. Only the key slice ``kv_block[:, :, 0]`` is modified in place.

    Raises:
        ValueError: if ``backend`` is not a production backend.
        ImportError, RuntimeError: if TileLang is unavailable or kernel launch
            fails.

    Async/thread-safety:
        Launches CUDA work on the current PyTorch stream. The cosine/sine
        tables are built once per call and passed to the TileLang table kernel.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if kv_block.shape[-1] < rotary_dim:
        return
    if backend not in ("auto", "tilelang"):
        raise ValueError(f"unknown RoPE apply backend: {backend}")

    cos_table, sin_table = build_rope_delta_tables(
        kv_block.device,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
    )
    apply_rope_delta_to_kv_key_block_table_tilelang(
        kv_block,
        cos_table=cos_table,
        sin_table=sin_table,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def build_rope_delta_tables(
    device: torch.device,
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
