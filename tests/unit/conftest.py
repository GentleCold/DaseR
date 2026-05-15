# SPDX-License-Identifier: Apache-2.0
"""Unit-test stubs for optional vLLM/GPU runtime dependencies."""

# Standard
import enum
import importlib.util
import sys
from types import ModuleType
from typing import Any


def _ensure_module(name: str) -> ModuleType:
    """Return an existing module or create a lightweight stub module.

    Args:
        name: Fully-qualified module name.

    Returns:
        Existing or newly-created module.

    Async/thread-safety:
        Mutates ``sys.modules`` during pytest collection in a single process.
    """
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    return module


def _install_optional_runtime_stubs() -> None:
    """Install minimal stubs for optional GPU/vLLM dependencies if absent.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        Intended for pytest collection startup before tests run concurrently.
    """
    if importlib.util.find_spec("torch") is None:
        torch = _ensure_module("torch")
        torch.Tensor = object
        torch.uint8 = object()

    if importlib.util.find_spec("cupy") is None:
        cupy = _ensure_module("cupy")
        cupy.ndarray = object

        def _asarray(value: Any) -> Any:
            return value

        cupy.asarray = _asarray

    if importlib.util.find_spec("kvikio") is None:
        kvikio = _ensure_module("kvikio")
        cufile = _ensure_module("kvikio.cufile")
        defaults = _ensure_module("kvikio.defaults")
        kvikio.cufile = cufile
        kvikio.defaults = defaults

        class CompatMode(enum.Enum):
            """Minimal kvikio compat-mode enum for importing gds_transfer."""

            OFF = "off"
            ON = "on"
            AUTO = "auto"

        class CuFile:
            """Placeholder CuFile that should never be constructed in unit tests."""

            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("kvikio CuFile stub is import-only")

        def _get(_: str) -> CompatMode:
            return CompatMode.AUTO

        def _set(_: str, __: Any) -> None:
            return None

        kvikio.CompatMode = CompatMode
        cufile.CuFile = CuFile
        defaults.get = _get
        defaults.set = _set

    if importlib.util.find_spec("vllm") is None:
        base = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
        _ensure_module("vllm")
        _ensure_module("vllm.distributed")
        _ensure_module("vllm.distributed.kv_transfer")
        _ensure_module("vllm.distributed.kv_transfer.kv_connector")
        _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1")

        class KVConnectorMetadata:
            """Minimal metadata base for scheduler-side unit tests."""

        class KVConnectorRole(enum.Enum):
            """Minimal connector roles for scheduler-side unit tests."""

            SCHEDULER = "scheduler"
            WORKER = "worker"

        class KVConnectorBase_V1:
            """Minimal connector base for scheduler-side unit tests."""

            def __init__(
                self,
                vllm_config: Any,
                role: KVConnectorRole,
                kv_cache_config: Any = None,
            ) -> None:
                """Store role and metadata for unit-test subclasses.

                Args:
                    vllm_config: unused vLLM config placeholder.
                    role: connector role.
                    kv_cache_config: unused KV cache config placeholder.

                Async/thread-safety:
                    Mutates only this instance.
                """
                self._role = role
                self._connector_metadata: KVConnectorMetadata | None = None

            def bind_connector_metadata(
                self, connector_metadata: KVConnectorMetadata
            ) -> None:
                """Bind connector metadata.

                Args:
                    connector_metadata: metadata object.

                Returns:
                    None.

                Async/thread-safety:
                    Mutates only this instance.
                """
                self._connector_metadata = connector_metadata

            def clear_connector_metadata(self) -> None:
                """Clear connector metadata.

                Args:
                    None.

                Returns:
                    None.

                Async/thread-safety:
                    Mutates only this instance.
                """
                self._connector_metadata = None

        base.KVConnectorBase_V1 = KVConnectorBase_V1
        base.KVConnectorMetadata = KVConnectorMetadata
        base.KVConnectorRole = KVConnectorRole


_install_optional_runtime_stubs()
