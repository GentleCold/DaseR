# SPDX-License-Identifier: Apache-2.0

"""Shared transfer-layer helper functions."""

# Standard
import inspect
import os

# Third Party
import cupy
import torch


def require_store_path(path: str) -> None:
    """Validate that a transfer store path exists.

    Args:
        path: preallocated store file path.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Store file not found: {path}")


def as_torch_uint8(buf: torch.Tensor | cupy.ndarray) -> torch.Tensor:
    """Return a flattened uint8 torch view for a torch tensor or CuPy array.

    Args:
        buf: torch tensor or CuPy array.

    Returns:
        Flattened uint8 torch tensor sharing storage with ``buf`` when possible.
    """
    if isinstance(buf, torch.Tensor):
        tensor = buf
    else:
        tensor = torch.as_tensor(buf, device=f"cuda:{buf.device.id}")
    return tensor.view(torch.uint8).flatten()


def copy_tensor(dst: torch.Tensor, src: torch.Tensor, nbytes: int) -> None:
    """Copy nbytes between uint8 tensor views.

    Args:
        dst: destination tensor.
        src: source tensor.
        nbytes: number of bytes to copy.

    Async/thread-safety:
        Synchronous tensor copy on the caller's thread.
    """
    dst[:nbytes].copy_(src[:nbytes], non_blocking=dst.is_cuda or src.is_cuda)


async def maybe_await(value: object) -> None:
    """Await ``value`` when it is awaitable.

    Args:
        value: callback return value.
    """
    if inspect.isawaitable(value):
        await value
