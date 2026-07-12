# SPDX-License-Identifier: Apache-2.0

# Standard
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.request import Request

# First Party
from daser.connector.helpers import base_req_id
from daser.connector.metadata import ReqLoadSpec
from daser.logging import init_logger

logger = init_logger(__name__)


def _base_req_id(req_id: str) -> str:
    """Compatibility wrapper for tests importing the scheduler-private helper."""
    return base_req_id(req_id)


def _store_slot_index(req_id: str) -> int | None:
    """Return the rolling-prefix store slot encoded in a synthetic request ID.

    Args:
        req_id: vLLM request ID or ``<req_id>:store:<slot>`` synthetic ID.

    Returns:
        Slot index for synthetic store IDs, or None for regular IDs.
    """
    if ":store:" not in req_id:
        return None
    try:
        return int(req_id.rsplit(":store:", 1)[1])
    except ValueError:
        return None


def _matches_request_or_store_id(req_id: str, base_req_id: str) -> bool:
    """Return whether ``req_id`` belongs to a base request.

    Args:
        req_id: vLLM request ID or synthetic connector work ID.
        base_req_id: Base vLLM request ID to match.

    Returns:
        True when ``req_id`` is the base request or one of its synthetic
        store/load entries.
    """
    return (
        req_id == base_req_id
        or req_id.startswith(f"{base_req_id}:store:")
        or req_id.startswith(f"{base_req_id}:load:")
    )


def _computed_tokens_after_step(
    scheduler_output: "SchedulerOutput",
) -> dict[str, int]:
    """Return per-request token counts that are valid after this step.

    Args:
        scheduler_output: vLLM SchedulerOutput for this step.

    Returns:
        Mapping from request ID to ``num_computed_tokens + scheduled_tokens``.
        Falls back to the scheduled token count when older or test scheduler
        outputs do not expose prior computed-token metadata.
    """
    scheduled = dict(getattr(scheduler_output, "num_scheduled_tokens", {}))
    computed_after = {req_id: int(tokens) for req_id, tokens in scheduled.items()}

    for req_data in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        req_id = str(getattr(req_data, "req_id", ""))
        if req_id in scheduled:
            computed_after[req_id] = int(
                getattr(req_data, "num_computed_tokens", 0)
            ) + int(scheduled[req_id])

    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached_reqs is not None:
        req_ids = getattr(cached_reqs, "req_ids", [])
        prior_counts = getattr(cached_reqs, "num_computed_tokens", [])
        for req_id, prior in zip(req_ids, prior_counts, strict=False):
            req_id = str(req_id)
            if req_id in scheduled:
                computed_after[req_id] = int(prior) + int(scheduled[req_id])

    return computed_after


def _get_kv_transfer_flag(request: "Request", key: str) -> Any:
    """Return ``request.kv_transfer_params[key]`` if present, else ``None``.

    Args:
        request: vLLM ``Request`` or compatible object.
        key: connector-specific flag name to extract.

    Returns:
        The value under ``key``, or ``None`` when absent.
    """
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    return params.get(key)


def _block_ids_for_chunk(
    block_ids: list[int],
    target_token_start: int,
    num_slots: int,
    block_tokens: int,
    max_tokens: int | None = None,
) -> list[int]:
    """Return vLLM block IDs for a chunk's target prompt range.

    Args:
        block_ids: all block IDs allocated to the request.
        target_token_start: token offset where the chunk starts in the prompt.
        num_slots: number of blocks/slots covered by the chunk.
        block_tokens: tokens per vLLM block.
        max_tokens: optional upper bound on accepted external tokens.

    Returns:
        Slice of block_ids for the chunk, or an empty list when the range
        is not block-aligned or exceeds the allocated blocks.
    """
    if target_token_start % block_tokens != 0:
        return []
    target_block_start = target_token_start // block_tokens
    effective_slots = num_slots
    if max_tokens is not None:
        remaining_tokens = max_tokens - target_token_start
        if remaining_tokens <= 0:
            return []
        effective_slots = min(num_slots, math.ceil(remaining_tokens / block_tokens))
    target_block_end = target_block_start + effective_slots
    if target_block_start < 0 or target_block_end > len(block_ids):
        return []
    return block_ids[target_block_start:target_block_end]


def _trim_chunk_to_external_window(
    chunk: dict[str, Any],
    block_ids: list[int],
    external_start: int,
    num_external_tokens: int,
    block_tokens: int,
    slot_size: int,
) -> bool:
    """Trim chunk metadata to the external token interval vLLM requested.

    Args:
        chunk: Mutable chunk metadata returned by the server.
        block_ids: Full vLLM block allocation for the request.
        external_start: Token offset where external KV loading begins.
        num_external_tokens: Number of tokens accepted from the connector.
        block_tokens: Tokens per vLLM block.
        slot_size: Bytes per DaseR slot.

    Returns:
        True when the chunk still covers at least one whole KV block.
    """
    if external_start % block_tokens != 0 or num_external_tokens <= 0:
        return False
    target_start = int(chunk.get("target_token_start", 0))
    target_end = target_start + int(chunk["token_count"])
    external_end = external_start + num_external_tokens
    load_start = max(target_start, external_start)
    load_end = min(target_end, external_end)
    load_start = ((load_start + block_tokens - 1) // block_tokens) * block_tokens
    load_end = ((load_end + block_tokens - 1) // block_tokens) * block_tokens
    load_end = min(load_end, target_end)
    if load_end <= load_start:
        return False

    skip_slots = (load_start - target_start) // block_tokens
    num_slots = (load_end - load_start) // block_tokens
    if load_start < external_start:
        return False
    block_start = load_start // block_tokens
    block_end = block_start + num_slots
    if block_start < 0 or block_end > len(block_ids):
        return False

    chunk["start_slot"] = int(chunk["start_slot"]) + skip_slots
    chunk["file_offset"] = int(chunk["file_offset"]) + skip_slots * slot_size
    chunk["num_slots"] = num_slots
    chunk["token_count"] = num_slots * block_tokens
    chunk["target_token_start"] = load_start
    chunk["block_ids"] = block_ids[block_start:block_end]
    return bool(chunk["block_ids"])


def _contiguous_prefix_tokens(
    chunks: list[dict[str, Any]], num_computed_tokens: int
) -> int:
    """Return external tokens covered contiguously after computed tokens.

    Args:
        chunks: server chunk payloads with target_token_start and token_count.
        num_computed_tokens: tokens vLLM already has locally.

    Returns:
        Number of additional contiguous prefix tokens covered by chunks.
    """
    covered_until = num_computed_tokens
    for chunk in sorted(
        chunks,
        key=lambda item: int(item.get("target_token_start", 0)),
    ):
        target_start = int(chunk.get("target_token_start", 0))
        token_count = int(chunk["token_count"])
        target_end = target_start + token_count
        if target_end <= covered_until:
            continue
        if target_start > covered_until:
            break
        covered_until = target_end
    return covered_until - num_computed_tokens


def _load_spec_from_chunk(chunk: dict[str, Any]) -> ReqLoadSpec:
    """Build a worker load specification from scheduler chunk metadata.

    Args:
        chunk: Chunk metadata returned by the server and annotated with vLLM
            block IDs during allocation.

    Returns:
        ReqLoadSpec consumed by the worker load path.

    Async/thread-safety:
        Pure scheduler-thread helper; it does not mutate connector state.
    """
    return ReqLoadSpec(
        chunk_key=str(chunk["chunk_key"]),
        start_slot=int(chunk["start_slot"]),
        num_slots=int(chunk["num_slots"]),
        block_ids=list(chunk["block_ids"]),
        file_offset=int(chunk["file_offset"]),
        token_count=int(chunk["token_count"]),
        target_token_start=int(chunk.get("target_token_start", 0)),
        pos_offset=int(chunk.get("pos_offset", 0)),
    )


def _merge_adjacent_load_specs(
    specs: list[ReqLoadSpec],
    slot_size: int,
) -> list[ReqLoadSpec]:
    """Merge adjacent load specs that describe one continuous KV byte range.

    Args:
        specs: Load specs for one request in prompt order.
        slot_size: Bytes represented by one DaseR KV slot.

    Returns:
        Coalesced load specs. Chunk keys from the first spec in a run are kept
        only as diagnostics; the worker load path addresses data by byte range.

    Async/thread-safety:
        Pure scheduler-thread helper; it does not mutate connector state.
    """
    merged: list[ReqLoadSpec] = []
    for spec in specs:
        if not spec.block_ids:
            continue
        if not merged:
            merged.append(spec)
            continue
        prev = merged[-1]
        prev_slots = len(prev.block_ids)
        adjacent = (
            prev.pos_offset == spec.pos_offset
            and prev.start_slot + prev_slots == spec.start_slot
            and prev.file_offset + prev_slots * slot_size == spec.file_offset
            and prev.target_token_start + prev.token_count == spec.target_token_start
        )
        if not adjacent:
            merged.append(spec)
            continue
        merged[-1] = ReqLoadSpec(
            chunk_key=prev.chunk_key,
            start_slot=prev.start_slot,
            num_slots=prev_slots + len(spec.block_ids),
            block_ids=[*prev.block_ids, *spec.block_ids],
            file_offset=prev.file_offset,
            token_count=prev.token_count + spec.token_count,
            target_token_start=prev.target_token_start,
            pos_offset=prev.pos_offset,
        )
    return merged
