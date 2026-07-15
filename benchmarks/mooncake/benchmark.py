# SPDX-License-Identifier: Apache-2.0
"""Replay Mooncake FAST'25 traces against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any, Iterable, Sequence

import httpx

from benchmarks.utils.datasets import BenchmarkSample
from benchmarks.utils.loadgen import RequestResult, vllm_completion_stream
from benchmarks.utils.metrics import compute_metric_delta, extract_prometheus_counters

MOONCAKE_BLOCK_TOKENS = 512
_BLOCK_ID_DIGITS = 8
_WRITER_STOP = object()


@dataclass(frozen=True)
class TraceRequest:
    """One validated Mooncake request.

    Args:
        index: Zero-based source row index.
        timestamp_ms: Arrival offset in milliseconds.
        input_length: Prompt length in tokens.
        output_length: Requested decode length in tokens.
        hash_ids: Remapped 512-token prefix block hashes.

    Thread-safety:
        Immutable and safe to share across replay tasks.
    """

    index: int
    timestamp_ms: float
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


@dataclass(frozen=True)
class TraceSelection:
    """Context-limit partition of selected trace requests.

    Args:
        selected: Contiguous source prefix selected for the run.
        eligible: Requests that fit the configured model context.
        skipped: Requests rejected by the context limit.

    Thread-safety:
        Immutable containers of immutable requests.
    """

    selected: tuple[TraceRequest, ...]
    eligible: tuple[TraceRequest, ...]
    skipped: tuple[TraceRequest, ...]


class PromptBuilder:
    """Build deterministic token-ID prompts from Mooncake block hashes.

    Args:
        vocab_size: Model vocabulary size.
        special_token_ids: Token IDs that must not appear in synthetic prompts.

    Thread-safety:
        Safe in the asyncio replay, where calls occur on one event-loop thread.
        The internal LRU cache is not intended for concurrent native threads.
    """

    def __init__(self, vocab_size: int, special_token_ids: Iterable[int]) -> None:
        excluded = set(special_token_ids)
        self._tokens = tuple(
            token_id for token_id in range(vocab_size) if token_id not in excluded
        )
        if len(self._tokens) < 2:
            raise ValueError("tokenizer must expose at least two ordinary token IDs")
        self._max_hash_id = len(self._tokens) ** _BLOCK_ID_DIGITS
        self._blocks: OrderedDict[int, tuple[int, ...]] = OrderedDict()

    def build(self, request: TraceRequest) -> list[int]:
        """Construct the exact-length prompt represented by a trace row.

        Args:
            request: Validated Mooncake request.

        Returns:
            Integer token IDs whose 512-token blocks follow ``hash_ids``.

        Thread-safety:
            See the class-level note. Returned lists are request-owned.
        """

        prompt: list[int] = []
        for hash_id in request.hash_ids:
            prompt.extend(self._block(hash_id))
        del prompt[request.input_length :]
        return prompt

    def _block(self, hash_id: int) -> tuple[int, ...]:
        cached = self._blocks.get(hash_id)
        if cached is not None:
            self._blocks.move_to_end(hash_id)
            return cached
        if hash_id >= self._max_hash_id:
            raise ValueError(
                f"hash ID {hash_id} exceeds deterministic encoding capacity"
            )
        base = len(self._tokens)
        value = hash_id
        block = []
        for _ in range(_BLOCK_ID_DIGITS):
            block.append(self._tokens[value % base])
            value //= base
        state = (hash_id + 1) & ((1 << 64) - 1)
        for position in range(_BLOCK_ID_DIGITS, MOONCAKE_BLOCK_TOKENS):
            state = (state * 6364136223846793005 + 1442695040888963407 + position) & (
                (1 << 64) - 1
            )
            block.append(self._tokens[state % base])
        result = tuple(block)
        self._blocks[hash_id] = result
        if len(self._blocks) > 8192:
            self._blocks.popitem(last=False)
        return result


def load_trace(
    path: Path,
    max_requests: int = 0,
    start_request: int = 0,
    min_context_tokens: int = 0,
) -> list[TraceRequest]:
    """Load and validate a contiguous window of a Mooncake JSONL trace.

    Args:
        path: Local Mooncake JSONL path.
        max_requests: Maximum rows to load; zero loads the complete trace.
        start_request: Zero-based source row where the window begins.
        min_context_tokens: Minimum input-plus-output tokens to retain.

    Returns:
        Validated requests in source order.

    Raises:
        ValueError: If the limit or any selected row violates the trace schema.

    Thread-safety:
        Performs synchronous planning-time file IO before replay starts.
    """

    if max_requests < 0 or start_request < 0 or min_context_tokens < 0:
        raise ValueError("request limits must be non-negative")
    requests: list[TraceRequest] = []
    previous_timestamp = -1.0
    first_timestamp = 0.0
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start_request:
                continue
            if max_requests and len(requests) >= max_requests:
                break
            if not line.strip():
                raise ValueError(f"row {index}: blank lines are not valid requests")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {index}: invalid JSON: {exc.msg}") from exc
            request = _parse_request(index, payload)
            if request.input_length + request.output_length < min_context_tokens:
                continue
            if request.timestamp_ms < previous_timestamp:
                raise ValueError(f"row {index}: timestamps must be non-decreasing")
            previous_timestamp = request.timestamp_ms
            if not requests:
                first_timestamp = request.timestamp_ms
            requests.append(
                replace(request, timestamp_ms=request.timestamp_ms - first_timestamp)
            )
    if not requests:
        raise ValueError("trace selection is empty")
    return requests


def select_requests(
    requests: Sequence[TraceRequest], max_model_len: int
) -> TraceSelection:
    """Partition requests by the model context limit without changing lengths.

    Args:
        requests: Validated source-order requests.
        max_model_len: Maximum prompt plus output token count.

    Returns:
        Selected, eligible, and skipped immutable request sequences.

    Thread-safety:
        Pure function.
    """

    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    selected = tuple(requests)
    eligible: list[TraceRequest] = []
    skipped: list[TraceRequest] = []
    for request in selected:
        destination = (
            eligible
            if request.input_length + request.output_length <= max_model_len
            else skipped
        )
        destination.append(request)
    return TraceSelection(
        selected=selected,
        eligible=tuple(eligible),
        skipped=tuple(skipped),
    )


def scheduled_offset_seconds(timestamp_ms: float, time_scale: float) -> float:
    """Convert a source timestamp to a scaled replay offset.

    Args:
        timestamp_ms: Original arrival offset in milliseconds.
        time_scale: Replay acceleration factor; one preserves source timing.

    Returns:
        Replay offset in seconds.

    Thread-safety:
        Pure function.
    """

    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be finite and positive")
    return timestamp_ms / 1000.0 / time_scale


def summarise(
    selection: TraceSelection,
    results: Sequence[dict[str, Any]],
    wall_seconds: float,
    time_scale: float,
    prefix_cache_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Summarize trace coverage and successful request latency.

    Args:
        selection: Context-limit partition used for the replay.
        results: Completed eligible request result dictionaries.
        wall_seconds: Measured replay wall duration.
        time_scale: Configured arrival acceleration factor.
        prefix_cache_metrics: Optional vLLM local/external cache counter deltas.

    Returns:
        JSON-serializable aggregate metrics.

    Thread-safety:
        Pure function over immutable snapshots.
    """

    successful = [result for result in results if result.get("error") is None]
    ttfts = [float(result["ttft_ms"]) for result in successful]
    latencies = [float(result["latency_ms"]) for result in successful]
    client_queues = [float(result["client_queue_ms"]) for result in successful]
    arrival_ttfts = [
        float(result["arrival_to_first_token_ms"]) for result in successful
    ]
    arrival_latencies = [
        float(result["arrival_to_completion_ms"]) for result in successful
    ]
    return {
        "selected_requests": len(selection.selected),
        "eligible_requests": len(selection.eligible),
        "skipped_context_requests": len(selection.skipped),
        "skipped_input_tokens": sum(row.input_length for row in selection.skipped),
        "skipped_output_tokens": sum(row.output_length for row in selection.skipped),
        "completed_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "wall_seconds": wall_seconds,
        "time_scale": time_scale,
        "achieved_completion_rps": (
            len(successful) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "ttft_ms": _latency_summary(ttfts),
        "latency_ms": _latency_summary(latencies),
        "client_queue_ms": _latency_summary(client_queues),
        "arrival_to_first_token_ms": _latency_summary(arrival_ttfts),
        "arrival_to_completion_ms": _latency_summary(arrival_latencies),
        "prefix_cache": _prefix_cache_summary(prefix_cache_metrics or {}),
    }


async def replay(
    *,
    selection: TraceSelection,
    prompt_builder: PromptBuilder,
    server_url: str,
    served_model_name: str,
    output_path: Path,
    max_inflight: int,
    timeout_seconds: float,
    time_scale: float,
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    """Replay eligible requests and incrementally write completion records.

    Args:
        selection: Validated context-limit partition.
        prompt_builder: Deterministic token prompt builder.
        server_url: OpenAI-compatible server base URL.
        served_model_name: Model name sent in completion bodies.
        output_path: JSONL request-result destination.
        max_inflight: Maximum concurrent HTTP requests.
        timeout_seconds: Per-request HTTP timeout.
        time_scale: Arrival acceleration factor.
        seed: Generation seed.

    Returns:
        Completion-order result dictionaries and measured wall duration.

    Async/thread-safety:
        Uses one asyncio event loop for scheduling and a dedicated writer
        thread for result IO. Prompt construction is serialized per task after
        admission and does not run in the writer thread.
    """

    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    scheduled_offset_seconds(0, time_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _ResultWriter(output_path)
    for request in selection.skipped:
        writer.write(_skipped_record(request))

    outer_sem = asyncio.Semaphore(max_inflight)
    started_at = time.perf_counter()
    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(
        max_connections=max_inflight,
        max_keepalive_connections=max_inflight,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [
                asyncio.create_task(
                    _run_request(
                        request=request,
                        prompt_builder=prompt_builder,
                        client=client,
                        server_url=server_url,
                        served_model_name=served_model_name,
                        outer_sem=outer_sem,
                        started_at=started_at,
                        timeout_seconds=timeout_seconds,
                        time_scale=time_scale,
                        seed=seed,
                        writer=writer,
                    )
                )
                for request in selection.eligible
            ]
            results = list(await asyncio.gather(*tasks))
    finally:
        writer.close()
    return results, time.perf_counter() - started_at


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Mooncake benchmark CLI arguments.

    Args:
        argv: Optional arguments without the executable name.

    Returns:
        Parsed argparse namespace.

    Thread-safety:
        Pure except for argparse's process-level error handling.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--server-url", default="http://127.0.0.1:8001")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--start-request", type=int, default=0)
    parser.add_argument("--min-context-tokens", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overflow", choices=("skip", "error"), default="skip")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    """Run the benchmark from parsed CLI arguments.

    Args:
        args: Namespace returned by :func:`parse_args`.

    Returns:
        Zero on complete success, one when any eligible request fails.

    Async/thread-safety:
        Owns one replay event loop and one result-writer thread.
    """

    _reject_tmp_output(args.output_dir)
    requests = load_trace(
        args.trace,
        args.max_requests,
        args.start_request,
        args.min_context_tokens,
    )
    selection = select_requests(requests, args.max_model_len)
    if args.overflow == "error" and selection.skipped:
        raise ValueError(
            f"{len(selection.skipped)} selected requests exceed max model length"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_builder = PromptBuilder(
        vocab_size=len(tokenizer),
        special_token_ids=tokenizer.all_special_ids,
    )
    results_path = args.output_dir / "requests.jsonl"
    summary_path = args.output_dir / "summary.json"
    before_metrics = await _metrics_snapshot(args.server_url.rstrip("/"))
    results, wall_seconds = await replay(
        selection=selection,
        prompt_builder=prompt_builder,
        server_url=args.server_url.rstrip("/"),
        served_model_name=args.served_model_name,
        output_path=results_path,
        max_inflight=args.max_inflight,
        timeout_seconds=args.timeout,
        time_scale=args.time_scale,
        seed=args.seed,
    )
    after_metrics = await _metrics_snapshot(args.server_url.rstrip("/"))
    summary = summarise(
        selection,
        results,
        wall_seconds,
        args.time_scale,
        compute_metric_delta(before_metrics, after_metrics),
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed_requests"] else 0


def main(argv: list[str] | None = None) -> None:
    """Run the Mooncake benchmark CLI.

    Args:
        argv: Optional arguments without the executable name.

    Async/thread-safety:
        Creates and owns the process event loop.
    """

    raise SystemExit(asyncio.run(main_async(parse_args(argv))))


def _parse_request(index: int, payload: Any) -> TraceRequest:
    if not isinstance(payload, dict):
        raise ValueError(f"row {index}: request must be a JSON object")
    timestamp = _number(payload.get("timestamp"), index, "timestamp")
    input_length = _integer(payload.get("input_length"), index, "input_length")
    output_length = _integer(payload.get("output_length"), index, "output_length")
    raw_hash_ids = payload.get("hash_ids")
    if not isinstance(raw_hash_ids, list):
        raise ValueError(f"row {index}: hash_ids must be a list")
    hash_ids = tuple(
        _integer(hash_id, index, f"hash_ids[{position}]")
        for position, hash_id in enumerate(raw_hash_ids)
    )
    if timestamp < 0:
        raise ValueError(f"row {index}: timestamp must be non-negative")
    if input_length <= 0 or output_length <= 0:
        raise ValueError(f"row {index}: input and output lengths must be positive")
    expected_blocks = math.ceil(input_length / MOONCAKE_BLOCK_TOKENS)
    if len(hash_ids) != expected_blocks:
        raise ValueError(
            f"row {index}: expected {expected_blocks} hash IDs, got {len(hash_ids)}"
        )
    if any(hash_id < 0 for hash_id in hash_ids):
        raise ValueError(f"row {index}: hash IDs must be non-negative")
    return TraceRequest(
        index=index,
        timestamp_ms=timestamp,
        input_length=input_length,
        output_length=output_length,
        hash_ids=hash_ids,
    )


def _integer(value: Any, index: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"row {index}: {field} must be an integer")
    return value


def _number(value: Any, index: int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {index}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row {index}: {field} must be finite")
    return result


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": statistics.mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _prefix_cache_summary(metrics: dict[str, float]) -> dict[str, float]:
    local_queries = metrics.get("vllm:prefix_cache_queries_total", 0.0)
    local_hits = metrics.get("vllm:prefix_cache_hits_total", 0.0)
    external_queries = metrics.get("vllm:external_prefix_cache_queries_total", 0.0)
    external_hits = metrics.get("vllm:external_prefix_cache_hits_total", 0.0)
    return {
        "local_query_tokens": local_queries,
        "local_hit_tokens": local_hits,
        "local_hit_rate": local_hits / local_queries if local_queries else 0.0,
        "external_query_tokens": external_queries,
        "external_hit_tokens": external_hits,
        "external_hit_rate": (
            external_hits / external_queries if external_queries else 0.0
        ),
    }


async def _metrics_snapshot(server_url: str) -> dict[str, float]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{server_url}/metrics")
            response.raise_for_status()
    except httpx.HTTPError:
        return {}
    return extract_prometheus_counters(response.text)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


async def _run_request(
    *,
    request: TraceRequest,
    prompt_builder: PromptBuilder,
    client: httpx.AsyncClient,
    server_url: str,
    served_model_name: str,
    outer_sem: asyncio.Semaphore,
    started_at: float,
    timeout_seconds: float,
    time_scale: float,
    seed: int,
    writer: _ResultWriter,
) -> dict[str, Any]:
    scheduled_offset = scheduled_offset_seconds(request.timestamp_ms, time_scale)
    await asyncio.sleep(max(0.0, started_at + scheduled_offset - time.perf_counter()))
    queued_at = time.perf_counter()
    async with outer_sem:
        admitted_at = time.perf_counter()
        prompt = prompt_builder.build(request)
        sent_at = admitted_at

        def record_send() -> None:
            nonlocal sent_at
            sent_at = time.perf_counter()

        result = await vllm_completion_stream(
            client,
            server_url,
            BenchmarkSample(
                sample_id=request.index,
                dataset="mooncake-toolagent",
                context="",
                question="",
                answers=[],
            ),
            prompt,
            {
                "model": served_model_name,
                "max_tokens": request.output_length,
                "temperature": 0.0,
                "ignore_eos": True,
                "seed": seed,
            },
            asyncio.Semaphore(1),
            timeout_seconds,
            on_send=record_send,
        )
    record = _result_record(
        request=request,
        result=result,
        scheduled_offset=scheduled_offset,
        queued_at=queued_at,
        admitted_at=admitted_at,
        sent_at=sent_at,
        completed_at=time.perf_counter(),
        started_at=started_at,
    )
    writer.write(record)
    return record


def _result_record(
    *,
    request: TraceRequest,
    result: RequestResult,
    scheduled_offset: float,
    queued_at: float,
    admitted_at: float,
    sent_at: float,
    completed_at: float,
    started_at: float,
) -> dict[str, Any]:
    payload = asdict(result)
    payload.pop("generated_text", None)
    payload.pop("queue_ms", None)
    target_at = started_at + scheduled_offset
    arrival_to_send_ms = max(0.0, sent_at - target_at) * 1000.0
    return {
        "status": "completed" if result.error is None else "failed",
        "trace_index": request.index,
        "source_timestamp_ms": request.timestamp_ms,
        "input_length": request.input_length,
        "output_length": request.output_length,
        "scheduled_offset_s": scheduled_offset,
        "queue_enter_offset_s": queued_at - started_at,
        "admitted_offset_s": admitted_at - started_at,
        "send_offset_s": sent_at - started_at,
        "completion_offset_s": completed_at - started_at,
        "client_queue_ms": (admitted_at - queued_at) * 1000.0,
        "arrival_to_send_ms": arrival_to_send_ms,
        "arrival_to_first_token_ms": arrival_to_send_ms + result.ttft_ms,
        "arrival_to_completion_ms": max(0.0, completed_at - target_at) * 1000.0,
        **payload,
    }


def _skipped_record(request: TraceRequest) -> dict[str, Any]:
    return {
        "status": "skipped_context_limit",
        "trace_index": request.index,
        "source_timestamp_ms": request.timestamp_ms,
        "input_length": request.input_length,
        "output_length": request.output_length,
    }


def _reject_tmp_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError("benchmark output must use approved scratch, not /tmp")


class _ResultWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError("result writer failed") from self._error
        self._queue.put(record)

    def close(self) -> None:
        self._queue.put(_WRITER_STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("result writer failed") from self._error

    def _run(self) -> None:
        try:
            with self._path.open("w", encoding="utf-8") as output:
                while True:
                    record = self._queue.get()
                    if record is _WRITER_STOP:
                        break
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
        except BaseException as exc:
            self._error = exc


if __name__ == "__main__":
    main()
