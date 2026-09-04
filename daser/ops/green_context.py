# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA Green Context streams for codec/prefill isolation.

CUDA Green Context is intentionally exposed as a small optional wrapper.  The
normal PyTorch primary context and stream remain untouched; callers choose the
returned external stream only for work that is safe to isolate.  The driver
handles are kept alive for the lifetime of the worker because TileLang and
PyTorch may retain stream-associated events after the last launch.
"""

import ctypes
from dataclasses import dataclass
from typing import Any

import torch


class _CuDevResource(ctypes.Structure):
    """ABI-compatible CUDA driver SM resource union."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("padding", ctypes.c_ubyte * 92),
        ("external", ctypes.c_ubyte * 48),
    ]


def parse_green_context_sm_count(value: str | None, env_name: str) -> int:
    """Parse an optional non-negative Green Context SM count.

    Args:
        value: Environment value. Empty or missing values disable the feature.
        env_name: Name included in validation errors.

    Returns:
        Requested SM count, or ``0`` when disabled.

    Raises:
        ValueError: If ``value`` is not a non-negative integer.

    Async/thread-safety:
        Pure startup parsing with no shared state.
    """
    if value is None or not value.strip():
        return 0
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a non-negative integer") from exc
    if count < 0:
        raise ValueError(f"{env_name} must be a non-negative integer")
    return count


def _bind_driver_function(driver: Any, name: str, *argtypes: Any) -> Any:
    """Bind one CUDA driver function with an explicit ctypes ABI.

    Raises:
        RuntimeError: If the installed driver does not expose ``name``. Green
            Context support is version-gated by the CUDA driver, so convert
            the low-level attribute failure into a startup diagnostic.
    """
    try:
        function = getattr(driver, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"CUDA driver does not expose {name}; Green Context requires "
            "a CUDA 12.4 or newer driver"
        ) from exc
    function.restype = ctypes.c_int
    function.argtypes = list(argtypes)
    return function


def _driver_error(driver: Any, code: int, operation: str) -> RuntimeError:
    """Build an actionable CUDA driver error without optional helpers."""
    get_name = getattr(driver, "cuGetErrorName", None)
    if get_name is None:
        return RuntimeError(f"{operation} failed with CUDA driver code {code}")
    get_name.restype = ctypes.c_int
    get_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    name = ctypes.c_char_p()
    if get_name(code, ctypes.byref(name)) != 0 or not name.value:
        return RuntimeError(f"{operation} failed with CUDA driver code {code}")
    return RuntimeError(
        f"{operation} failed with CUDA driver code {code} ({name.value.decode()})"
    )


@dataclass
class GreenContextStream:
    """Own one CUDA Green Context and its PyTorch external stream.

    Args:
        stream: PyTorch external stream bound to the Green Context.
        provisioned_sms: Number of SMs provisioned by the driver.
        driver: Loaded CUDA driver library retained for handle lifetime.
        context_handle: Opaque Green Context handle.
        stream_handle: Opaque CUDA stream handle.

    Async/thread-safety:
        The owning pipeline must submit and close this stream from one worker
        lifecycle. ``close`` synchronizes the stream and may block during
        shutdown; driver context destruction is deferred to process teardown.
    """

    stream: torch.cuda.ExternalStream
    provisioned_sms: int
    driver: Any
    context_handle: ctypes.c_void_p
    stream_handle: ctypes.c_void_p
    _closed: bool = False

    def close(self) -> None:
        """Synchronize pending codec work before worker teardown.

        Async/thread-safety:
            Synchronous shutdown-only operation. No other thread may submit
            work to ``stream`` while this method runs.
        """
        if self._closed:
            return
        self.stream.synchronize()
        self._closed = True


def create_green_context(*, device: torch.device, sm_count: int) -> GreenContextStream:
    """Create an SM-partitioned CUDA stream for optional codec work.

    Args:
        device: CUDA device hosting the vLLM KV cache.
        sm_count: Positive minimum SM count requested for the partition.

    Returns:
        Worker-owned Green Context stream descriptor.

    Raises:
        ValueError: If ``device`` is not CUDA or ``sm_count`` is non-positive.
        RuntimeError: If the installed CUDA driver cannot create the context.

    Async/thread-safety:
        Startup-only synchronous operation. The returned stream is safe for
        the owning pipeline's serialized CUDA submissions.
    """
    if device.type != "cuda":
        raise ValueError("green context requires a CUDA device")
    if sm_count <= 0:
        raise ValueError("green context SM count must be positive")
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError("CUDA driver library is unavailable") from exc

    cu_init = _bind_driver_function(driver, "cuInit", ctypes.c_uint)
    cu_device_get = _bind_driver_function(
        driver,
        "cuDeviceGet",
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    )
    cu_get_resource = _bind_driver_function(
        driver,
        "cuDeviceGetDevResource",
        ctypes.c_int,
        ctypes.POINTER(_CuDevResource),
        ctypes.c_int,
    )
    cu_split = _bind_driver_function(
        driver,
        "cuDevSmResourceSplitByCount",
        ctypes.POINTER(_CuDevResource),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(_CuDevResource),
        ctypes.POINTER(_CuDevResource),
        ctypes.c_uint,
        ctypes.c_uint,
    )
    cu_generate_desc = _bind_driver_function(
        driver,
        "cuDevResourceGenerateDesc",
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_CuDevResource),
        ctypes.c_uint,
    )
    cu_create = _bind_driver_function(
        driver,
        "cuGreenCtxCreate",
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint,
    )
    cu_stream_create = _bind_driver_function(
        driver,
        "cuGreenCtxStreamCreate",
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_int,
    )

    code = cu_init(0)
    if code != 0:
        raise _driver_error(driver, code, "cuInit")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    driver_device = ctypes.c_int()
    code = cu_device_get(ctypes.byref(driver_device), int(device_index))
    if code != 0:
        raise _driver_error(driver, code, "cuDeviceGet")

    source = _CuDevResource()
    code = cu_get_resource(driver_device.value, ctypes.byref(source), 1)
    if code != 0:
        raise _driver_error(driver, code, "cuDeviceGetDevResource")
    groups = (_CuDevResource * 1)()
    remaining = _CuDevResource()
    group_count = ctypes.c_uint(1)
    code = cu_split(
        groups,
        ctypes.byref(group_count),
        ctypes.byref(source),
        ctypes.byref(remaining),
        0,
        sm_count,
    )
    if code != 0:
        raise _driver_error(driver, code, "cuDevSmResourceSplitByCount")
    if group_count.value != 1:
        raise RuntimeError(
            "cuDevSmResourceSplitByCount returned an unexpected group count: "
            f"requested={sm_count} groups={group_count.value}"
        )
    provisioned_sms = int.from_bytes(bytes(groups[0].external[:4]), "little")

    descriptor = ctypes.c_void_p()
    code = cu_generate_desc(ctypes.byref(descriptor), groups, 1)
    if code != 0:
        raise _driver_error(driver, code, "cuDevResourceGenerateDesc")
    context_handle = ctypes.c_void_p()
    code = cu_create(ctypes.byref(context_handle), descriptor, driver_device.value, 1)
    if code != 0:
        raise _driver_error(driver, code, "cuGreenCtxCreate")
    stream_handle = ctypes.c_void_p()
    code = cu_stream_create(ctypes.byref(stream_handle), context_handle, 1, 0)
    if code != 0:
        destroy_context = _bind_driver_function(
            driver, "cuGreenCtxDestroy", ctypes.c_void_p
        )
        destroy_context(context_handle)
        raise _driver_error(driver, code, "cuGreenCtxStreamCreate")

    torch.cuda.set_device(device)
    return GreenContextStream(
        stream=torch.cuda.ExternalStream(stream_handle.value, device=device),
        provisioned_sms=provisioned_sms,
        driver=driver,
        context_handle=context_handle,
        stream_handle=stream_handle,
    )


__all__ = [
    "GreenContextStream",
    "create_green_context",
    "parse_green_context_sm_count",
]
