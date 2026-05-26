# SPDX-License-Identifier: Apache-2.0
"""RoPE delta application operators for cached K blocks."""

from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import Any, Literal

# Third Party
import torch

# First Party
from daser.logging import init_logger
from daser.ops.rope_apply_tilelang import (
    apply_rope_delta_to_key_block_tilelang,
    clear_tilelang_rope_cache,
)

logger = init_logger(__name__)

RopeApplyBackend = Literal["auto", "tilelang", "compile", "naive"]

_CompiledFn = Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor]
_compile_cache: dict[tuple[Any, Any, int, int, bool], _CompiledFn] = {}
_compile_disabled = False
_compile_warning_emitted = False
_tilelang_disabled = False
_tilelang_warning_emitted = False


def clear_rope_apply_compile_cache() -> None:
    """Clear cached compiled RoPE functions.

    Returns:
        None.

    Async/thread-safety:
        Intended for tests and benchmark setup. Do not call while another
        thread is applying RoPE with the compiled backend.
    """
    global _compile_disabled, _compile_warning_emitted
    global _tilelang_disabled, _tilelang_warning_emitted
    _compile_cache.clear()
    clear_tilelang_rope_cache()
    _compile_disabled = False
    _compile_warning_emitted = False
    _tilelang_disabled = False
    _tilelang_warning_emitted = False


def apply_rope_delta_to_key_block(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
    backend: RopeApplyBackend = "auto",
) -> None:
    """Rotate an already-RoPE'd K block by a relative position delta.

    Args:
        key_block: K cache block with shape ``[..., block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.
        backend: ``auto`` tries TileLang, then naive; ``tilelang`` and
            ``compile`` force the respective fast-path attempt before fallback;
            ``naive`` skips fast paths.

    Returns:
        None. ``key_block`` is modified in place.

    Async/thread-safety:
        Performs GPU tensor operations on the current PyTorch stream. The
        compile cache is shared process-wide and is safe for normal single
        worker-thread use; tests should clear it between monkeypatched runs.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if key_block.shape[-1] < rotary_dim:
        return
    if backend not in ("auto", "tilelang", "compile", "naive"):
        raise ValueError(f"unknown RoPE apply backend: {backend}")

    if backend == "naive":
        apply_rope_delta_to_key_block_naive(
            key_block,
            delta=delta,
            rope_base=rope_base,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
        return

    if backend in ("auto", "tilelang") and _can_use_tilelang_backend(
        key_block, rotary_dim
    ):
        try:
            _apply_tilelang(
                key_block,
                delta=delta,
                rope_base=rope_base,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            )
            return
        except Exception as exc:  # noqa: BLE001
            _disable_tilelang_once(exc)

    if backend == "tilelang":
        apply_rope_delta_to_key_block_naive(
            key_block,
            delta=delta,
            rope_base=rope_base,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
        return

    if backend == "compile" and _can_use_compiled_backend(key_block, rotary_dim):
        try:
            _apply_compiled(
                key_block,
                delta=delta,
                rope_base=rope_base,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            )
            return
        except Exception as exc:  # noqa: BLE001
            _disable_compile_once(exc)

    apply_rope_delta_to_key_block_naive(
        key_block,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def apply_rope_delta_to_key_block_naive(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Naively rotate an already-RoPE'd K block by a relative position delta.

    Args:
        key_block: K cache block with shape ``[..., block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. ``key_block`` is modified in place.

    Async/thread-safety:
        Performs tensor work on the current PyTorch stream and has no shared
        mutable state.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if key_block.shape[-1] < rotary_dim:
        return

    rotated = _rope_delta_kernel(
        key_block[..., :rotary_dim],
        _build_freqs(key_block.device, delta, rope_base, rotary_dim),
        is_neox_style,
    )
    key_block[..., :rotary_dim].copy_(rotated.to(key_block.dtype))


def _can_use_compiled_backend(key_block: torch.Tensor, rotary_dim: int) -> bool:
    """Return whether the compiled backend should be attempted."""
    if _compile_disabled:
        return False
    if not hasattr(torch, "compile"):
        return False
    if key_block.device.type != "cuda":
        return False
    if not key_block.is_contiguous():
        return False
    if key_block.dim() < 3:
        return False
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        return False
    return key_block.shape[-1] >= rotary_dim


def _can_use_tilelang_backend(key_block: torch.Tensor, rotary_dim: int) -> bool:
    """Return whether the TileLang backend should be attempted."""
    if _tilelang_disabled:
        return False
    if key_block.device.type != "cuda":
        return False
    if not key_block.is_contiguous():
        return False
    if key_block.dim() < 3:
        return False
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        return False
    return key_block.shape[-1] >= rotary_dim


def _apply_tilelang(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Apply RoPE delta with TileLang when the optional backend is available."""
    apply_rope_delta_to_key_block_tilelang(
        key_block,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def _apply_compiled(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Apply RoPE delta with a cached compiled tensor function."""
    compiled = _get_compiled_fn(
        key_block.dtype,
        key_block.device,
        rotary_dim,
        is_neox_style,
    )
    rotated = compiled(
        key_block[..., :rotary_dim],
        _build_freqs(key_block.device, delta, rope_base, rotary_dim),
        is_neox_style,
    )
    key_block[..., :rotary_dim].copy_(rotated.to(key_block.dtype))


def _get_compiled_fn(
    dtype: Any,
    device: Any,
    rotary_dim: int,
    is_neox_style: bool,
) -> _CompiledFn:
    """Return a cached dynamic compiled RoPE delta function."""
    key = (dtype, device, rotary_dim, is_neox_style)
    compiled = _compile_cache.get(key)
    if compiled is not None:
        return compiled
    compiled = torch.compile(_rope_delta_kernel, fullgraph=True, dynamic=True)
    _compile_cache[key] = compiled
    return compiled


def _rope_delta_kernel(
    rot: torch.Tensor,
    freqs: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Return the rotated RoPE dimensions for one K block."""
    compute = rot.float()
    cos = freqs.cos().view(*([1] * (compute.dim() - 1)), -1)
    sin = freqs.sin().view(*([1] * (compute.dim() - 1)), -1)

    if is_neox_style:
        x1, x2 = torch.chunk(compute, 2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    x1 = compute[..., ::2]
    x2 = compute[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return rotated.flatten(-2)


def _build_freqs(
    device: Any,
    delta: int,
    rope_base: float,
    rotary_dim: int,
) -> torch.Tensor:
    """Build fp32 RoPE delta frequencies for a relative position offset."""
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    return delta * inv_freq


def _disable_compile_once(exc: Exception) -> None:
    """Disable the compiled backend after the first runtime failure."""
    global _compile_disabled, _compile_warning_emitted
    _compile_disabled = True
    if _compile_warning_emitted:
        return
    _compile_warning_emitted = True
    logger.warning("[ROPE] torch.compile backend disabled: %s", exc)


def _disable_tilelang_once(exc: Exception) -> None:
    """Disable the TileLang backend after the first runtime failure."""
    global _tilelang_disabled, _tilelang_warning_emitted
    _tilelang_disabled = True
    if _tilelang_warning_emitted:
        return
    _tilelang_warning_emitted = True
    logger.warning("[ROPE] TileLang backend disabled: %s", exc)
