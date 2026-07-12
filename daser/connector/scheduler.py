# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.request import Request

from daser.connector.metadata import DaserConnectorMeta
from daser.connector.scheduler_planning import (
    _base_req_id,
    _block_ids_for_chunk,
    _computed_tokens_after_step,
    _contiguous_prefix_tokens,
    _get_kv_transfer_flag,
    _matches_request_or_store_id,
    _store_slot_index,
    _trim_chunk_to_external_window,
)

__all__ = [
    "SchedulerConnectorMixin",
    "_base_req_id",
    "_block_ids_for_chunk",
    "_computed_tokens_after_step",
    "_contiguous_prefix_tokens",
    "_get_kv_transfer_flag",
    "_matches_request_or_store_id",
    "_store_slot_index",
    "_trim_chunk_to_external_window",
]


class SchedulerConnectorMixin:
    """Adapt vLLM scheduler hooks to the request lifecycle interface."""

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return DaseR cache credit for one request."""
        return self._request_lifecycle.get_num_new_matched_tokens(
            request,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Bind allocated vLLM blocks to pending lifecycle work."""
        self._request_lifecycle.update_state_after_alloc(
            request,
            blocks,
            num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> DaserConnectorMeta:
        """Build worker metadata from request lifecycle state."""
        return self._request_lifecycle.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return whether worker save completion still holds request blocks."""
        return self._request_lifecycle.request_finished(request, block_ids)

    def update_connector_output(self, connector_output: Any) -> None:
        """Apply worker transfer completions to request lifecycle state."""
        self._request_lifecycle.update_connector_output(connector_output)
