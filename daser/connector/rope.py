# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch


def apply_rope_delta(
    key: torch.Tensor,
    source_start: int,
    target_start: int,
    block_tokens: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Rotate a RoPE-encoded key tensor from source to target positions.

    Standard RoPE rotations compose by angle addition. A key cached at
    ``source_start + token_i`` can therefore be moved to
    ``target_start + token_i`` by applying the angle delta between the two
    positions. The helper treats the last dimension as the RoPE head dimension
    and the second-to-last dimension as tokens.

    Args:
        key: RoPE-encoded key tensor. The last dimension must be even.
        source_start: absolute source position of the first token.
        target_start: absolute target position of the first token.
        block_tokens: expected number of token positions in this block.
        rope_theta: RoPE base theta from model config.

    Returns:
        A rotated tensor with the same shape, dtype, and device as ``key``.

    Raises:
        ValueError: if the tensor shape is unsupported.

    Async/thread-safety:
        Pure tensor operation. It does not mutate ``key`` and performs no I/O.
    """
    if key.dim() < 2:
        raise ValueError("key tensor must have at least token and head dimensions")
    head_dim = key.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    token_count = key.shape[-2]
    if token_count > block_tokens:
        raise ValueError("token dimension cannot exceed block_tokens")
    if source_start == target_start:
        return key.clone()

    half_dim = head_dim // 2
    positions = torch.arange(token_count, device=key.device, dtype=torch.float32)
    delta_positions = (target_start - source_start) + positions.new_zeros(token_count)
    dims = torch.arange(0, half_dim, device=key.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (dims / half_dim))
    angles = delta_positions[:, None] * inv_freq[None, :]
    cos = torch.cos(angles).to(dtype=key.dtype)
    sin = torch.sin(angles).to(dtype=key.dtype)

    while cos.dim() < key.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    first = key[..., :half_dim]
    second = key[..., half_dim:]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.cat((rotated_first, rotated_second), dim=-1)
