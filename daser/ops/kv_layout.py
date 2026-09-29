# SPDX-License-Identifier: Apache-2.0

"""Views between vLLM's block-outermost KV cache and DaseR slot order.

DaseR requests vLLM's ``BLNHC`` KV cache layout. vLLM then allocates one
shared buffer whose physical order is ``[blocks, layers, tokens, heads, 2,
head_dim]`` and hands the connector one ``[blocks, heads, tokens, 2 *
head_dim]`` view per layer. DaseR slots keep the plane-major order
``[layers, 2, tokens, heads, head_dim]`` so each (layer, K/V) plane is one
contiguous codec record. The connector therefore works on a strided logical
view ``[blocks, layers, 2, tokens, heads, head_dim]`` over vLLM's buffer, and
kernels that address raw memory use the contiguous physical view.
"""

import torch

# Physical [B, L, N, H, 2, D] -> logical [B, L, 2, N, H, D].
_PHYSICAL_TO_LOGICAL = (0, 1, 4, 2, 3, 5)
# Logical [B, L, 2, N, H, D] -> physical [B, L, N, H, 2, D].
_LOGICAL_TO_PHYSICAL = (0, 1, 3, 4, 2, 5)


def cross_layer_kv_view(
    kv_caches: dict[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    """Build the logical cross-layer view over vLLM's per-layer KV views.

    Args:
        kv_caches: vLLM per-layer tensors shaped ``[blocks, heads, tokens,
            2 * head_dim]`` for the ``BLNHC`` layout.

    Returns:
        ``(layer_names, kv_cache)``: layer names in their physical order inside
        vLLM's shared buffer, which is also the DaseR slot layer order, and a
        strided ``[blocks, layers, 2, tokens, heads, head_dim]`` view that
        aliases vLLM's KV memory.

    Raises:
        ValueError: If the per-layer views are not evenly spaced slices of one
            block-outermost ``BLNHC`` buffer.

    Async/thread-safety:
        Pure metadata work; call once during KV cache registration.
    """
    if not kv_caches:
        raise ValueError("KV cache registration requires at least one layer")
    layer_names = sorted(kv_caches, key=lambda name: kv_caches[name].storage_offset())
    layers = [kv_caches[name] for name in layer_names]
    first = layers[0]
    if first.dim() != 4 or first.shape[-1] % 2:
        raise ValueError(f"unsupported vLLM KV cache shape {tuple(first.shape)}")
    for layer in layers[1:]:
        if (
            layer.shape != first.shape
            or layer.stride() != first.stride()
            or layer.dtype != first.dtype
            or layer.device != first.device
            or layer.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
        ):
            raise ValueError("vLLM KV layers must be views of one shared buffer")
    num_blocks, heads, tokens, packed_dim = (int(dim) for dim in first.shape)
    head_dim = packed_dim // 2
    block_stride, head_stride, token_stride, _ = first.stride()
    layer_stride = (
        layers[1].storage_offset() - first.storage_offset() if len(layers) > 1 else 0
    )
    for idx, layer in enumerate(layers):
        if layer.storage_offset() != first.storage_offset() + idx * layer_stride:
            raise ValueError("vLLM KV layers must be evenly spaced")
    if len(layers) == 1:
        layer_stride = tokens * heads * packed_dim
    logical = torch.as_strided(
        first,
        size=(num_blocks, len(layers), 2, tokens, heads, head_dim),
        stride=(block_stride, layer_stride, head_dim, token_stride, head_stride, 1),
    )
    physical_kv_view(logical)
    return layer_names, logical


def physical_kv_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return the contiguous ``[blocks, layers, tokens, heads, 2, head_dim]`` view.

    Args:
        kv_cache: Logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view.

    Returns:
        The same memory permuted into its contiguous physical order.

    Raises:
        ValueError: If ``kv_cache`` is not a logical view of a contiguous
            block-outermost ``BLNHC`` buffer.
    """
    if kv_cache.dim() != 6 or kv_cache.shape[2] != 2:
        raise ValueError(f"unsupported cross-layer KV shape {tuple(kv_cache.shape)}")
    physical = kv_cache.permute(_LOGICAL_TO_PHYSICAL)
    if not physical.is_contiguous():
        raise ValueError("cross-layer KV cache must be a contiguous BLNHC buffer")
    return physical


def allocate_cross_layer_kv_cache(
    num_blocks: int,
    num_layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate a logical cross-layer KV view with vLLM's ``BLNHC`` memory.

    Args:
        num_blocks: Number of KV blocks.
        num_layers: Number of attention layers.
        block_tokens: Tokens per block.
        heads: KV heads per layer.
        head_dim: Scalars per head.
        dtype: KV element type.
        device: Allocation device.

    Returns:
        A logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view over a
        freshly allocated physical ``[blocks, layers, tokens, heads, 2,
        head_dim]`` buffer.
    """
    physical = torch.empty(
        num_blocks,
        num_layers,
        block_tokens,
        heads,
        2,
        head_dim,
        dtype=dtype,
        device=device,
    )
    return physical.permute(_PHYSICAL_TO_LOGICAL)


def vllm_layer_views(kv_cache: torch.Tensor) -> list[torch.Tensor]:
    """Return vLLM-shaped per-layer views of a logical cross-layer cache.

    Args:
        kv_cache: Logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view
            of a contiguous ``BLNHC`` buffer.

    Returns:
        One ``[blocks, heads, tokens, 2 * head_dim]`` view per layer, matching
        what vLLM passes to ``register_kv_caches``.
    """
    physical = physical_kv_view(kv_cache)
    num_blocks, num_layers, tokens, heads, _, head_dim = physical.shape
    packed = physical.reshape(num_blocks, num_layers, tokens, heads, 2 * head_dim)
    return [packed[:, idx].transpose(1, 2) for idx in range(num_layers)]


__all__ = [
    "allocate_cross_layer_kv_cache",
    "cross_layer_kv_view",
    "physical_kv_view",
    "vllm_layer_views",
]
