# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.forward_context import ForwardContext

from daser.connector.metadata import DaserConnectorMeta


class WorkerConnectorMixin:
    """Adapt vLLM worker hooks to the worker runtime interface."""

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register per-layer KV tensors with the worker runtime."""
        self._worker_runtime.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type[Any],
    ) -> None:
        """Register vLLM's cross-layer KV tensor with the worker runtime."""
        self._worker_runtime.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Bind one scheduler metadata step to the worker runtime."""
        super().bind_connector_metadata(connector_metadata)
        self._worker_runtime.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        """Clear the current metadata step from the worker runtime."""
        super().clear_connector_metadata()
        self._worker_runtime.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Submit scheduler-selected cache loads through the load pipeline."""
        self._worker_runtime.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Observe the runtime's request-level load completion contract."""
        self._worker_runtime.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Forward one vLLM save hook to the worker runtime."""
        self._worker_runtime.save_kv_layer(
            layer_name,
            kv_layer,
            attn_metadata,
            **kwargs,
        )

    def wait_for_save(self) -> None:
        """Defer step stores through the worker runtime."""
        self._worker_runtime.wait_for_save()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return completed store and load request IDs from the runtime."""
        return self._worker_runtime.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return and clear runtime load-error block IDs."""
        return self._worker_runtime.get_block_ids_with_load_errors()

    def shutdown(self) -> None:
        """Drain and stop the worker runtime when initialized."""
        runtime = getattr(self, "_worker_runtime", None)
        if runtime is not None:
            runtime.shutdown()
