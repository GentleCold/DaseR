# SPDX-License-Identifier: Apache-2.0
"""Export real TraceLab session gaps for the official closed-loop runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from benchmarks.utils.constants import slot_size_for_block_tokens

_FIELDS = (
    "id",
    "input_len",
    "output_len",
    "arrival_time",
    "round_idx",
    "tool_wait_after_ms",
    "prefix_len",
)
_BLOCK_TOKENS = 16


@dataclass(frozen=True)
class Round:
    """One replayable TraceLab round.

    Args:
        input_len: Newly appended prompt tokens.
        output_len: Trace-observed decode tokens.
        prefix_len: Reusable historical prompt tokens.
        first_event_ms: Wall-clock timestamp of the first round event.
        last_event_ms: Wall-clock timestamp of the last round event.

    Thread-safety:
        Immutable and safe to share between preparation threads.
    """

    input_len: int
    output_len: int
    prefix_len: int
    first_event_ms: int
    last_event_ms: int


@dataclass(frozen=True)
class PreparedSession:
    """A validated closed-loop session and its source gap diagnostics.

    Args:
        source_id: Stable composite TraceLab session identity.
        rounds: Source-ordered replay rounds.
        negative_gaps: Adjacent event boundaries that overlap in the trace.

    Thread-safety:
        Immutable and safe to share between preparation threads.
    """

    source_id: str
    rounds: tuple[Round, ...]
    negative_gaps: int


def prepare_sessions(rows: Iterable[tuple[Any, ...]]) -> list[PreparedSession]:
    """Group ordered DuckDB rows into validated TraceLab sessions.

    Args:
        rows: Query rows containing composite identity, token counts, and event
            bounds in the order returned by :func:`load_rows`.

    Returns:
        Sessions with at least two rounds and valid event timestamps.

    Thread-safety:
        Pure with respect to caller-owned rows.
    """

    grouped: dict[str, list[Round]] = {}
    for source_id, input_len, output_len, prefix_len, first_ms, last_ms in rows:
        if first_ms is None or last_ms is None:
            continue
        grouped.setdefault(str(source_id), []).append(
            Round(
                input_len=max(1, int(input_len or 0)),
                output_len=max(1, int(output_len or 0)),
                prefix_len=max(0, int(prefix_len or 0)),
                first_event_ms=int(first_ms),
                last_event_ms=int(last_ms),
            )
        )

    sessions: list[PreparedSession] = []
    for source_id, rounds in grouped.items():
        if len(rounds) < 2:
            continue
        negative_gaps = sum(
            right.first_event_ms < left.last_event_ms
            for left, right in zip(rounds, rounds[1:], strict=False)
        )
        sessions.append(PreparedSession(source_id, tuple(rounds), negative_gaps))
    return sessions


def select_sessions(
    sessions: list[PreparedSession],
    *,
    max_sessions: int,
    max_model_len: int,
    seed: int,
) -> list[PreparedSession]:
    """Select whole sessions without truncating or rescaling trace rounds.

    Args:
        sessions: Candidate sessions.
        max_sessions: Maximum number of sessions to return.
        max_model_len: Model context limit for prompt plus output tokens.
        seed: Deterministic shuffle seed.

    Returns:
        Up to ``max_sessions`` sessions whose every round fits the model.

    Thread-safety:
        Does not mutate ``sessions`` or global random state.
    """

    if max_sessions <= 0 or max_model_len <= 0:
        raise ValueError("session and model limits must be positive")
    eligible = [
        session
        for session in sessions
        if all(
            round_.prefix_len + round_.input_len + round_.output_len <= max_model_len
            for round_ in session.rounds
        )
    ]
    random.Random(seed).shuffle(eligible)
    return eligible[:max_sessions]


def select_pause_windows(
    sessions: list[PreparedSession],
    *,
    max_sessions: int,
    max_model_len: int,
    window_rounds: int,
    min_pause_s: float,
    max_pause_s: float,
    max_prefix_tokens: int,
) -> list[PreparedSession]:
    """Select long-prefix windows around real, bounded TraceLab pauses.

    Args:
        sessions: Candidate complete sessions.
        max_sessions: Maximum number of windows to return.
        max_model_len: Model context limit for prompt plus output tokens.
        window_rounds: Maximum contiguous rounds retained per session.
        min_pause_s: Minimum source pause represented by each window.
        max_pause_s: Maximum source pause represented by each window.
        max_prefix_tokens: Largest pause-frontier prefix to admit.

    Returns:
        Deterministic windows ordered by decreasing pause-frontier prefix.

    Thread-safety:
        Does not mutate caller-owned sessions.
    """

    if (
        max_sessions <= 0
        or max_model_len <= 0
        or window_rounds < 2
        or min_pause_s < 0
        or max_pause_s < min_pause_s
        or max_prefix_tokens <= 0
    ):
        raise ValueError("invalid pause-window limits")
    candidates: list[tuple[int, PreparedSession]] = []
    for session in sessions:
        best: tuple[int, int] | None = None
        for index, (left, right) in enumerate(
            zip(session.rounds, session.rounds[1:], strict=False)
        ):
            gap_s = max(0, right.first_event_ms - left.last_event_ms) / 1000.0
            context_tokens = right.prefix_len + right.input_len + right.output_len
            if (
                min_pause_s <= gap_s <= max_pause_s
                and left.prefix_len <= max_prefix_tokens
                and context_tokens <= max_model_len
                and (best is None or left.prefix_len > best[0])
            ):
                best = (left.prefix_len, index)
        if best is None:
            continue
        prefix_tokens, pause_index = best
        before = window_rounds // 2
        start = max(0, pause_index + 1 - before)
        end = min(len(session.rounds), start + window_rounds)
        start = max(0, end - window_rounds)
        window = session.rounds[start:end]
        window_gaps_s = [
            max(0, right.first_event_ms - left.last_event_ms) / 1000.0
            for left, right in zip(window, window[1:], strict=False)
        ]
        if max(window_gaps_s, default=0.0) <= max_pause_s and all(
            round_.prefix_len + round_.input_len + round_.output_len <= max_model_len
            for round_ in window
        ):
            negative_gaps = sum(
                right.first_event_ms < left.last_event_ms
                for left, right in zip(window, window[1:], strict=False)
            )
            candidates.append(
                (
                    prefix_tokens,
                    PreparedSession(session.source_id, window, negative_gaps),
                )
            )
    candidates.sort(key=lambda item: (-item[0], item[1].source_id))
    return [session for _, session in candidates[:max_sessions]]


def select_dense_windows(
    sessions: list[PreparedSession],
    *,
    max_sessions: int,
    max_model_len: int,
    window_rounds: int,
    max_gap_s: float,
    min_prefix_tokens: int,
    max_prefix_tokens: int,
) -> list[PreparedSession]:
    """Select long-prefix windows with only short, non-overlapping gaps.

    Args:
        sessions: Candidate complete sessions.
        max_sessions: Maximum number of windows to return.
        max_model_len: Model context limit for prompt plus output tokens.
        window_rounds: Exact number of contiguous rounds per window.
        max_gap_s: Largest source gap allowed inside a window.
        min_prefix_tokens: Minimum peak prefix required in a window.
        max_prefix_tokens: Largest prefix admitted in a window.

    Returns:
        One deterministic best window per selected session, ordered by total
        reusable prefix tokens.

    Thread-safety:
        Does not mutate caller-owned sessions.
    """

    if (
        max_sessions <= 0
        or max_model_len <= 0
        or window_rounds < 2
        or max_gap_s < 0
        or min_prefix_tokens < 0
        or max_prefix_tokens < min_prefix_tokens
    ):
        raise ValueError("invalid dense-window limits")
    max_gap_ms = max_gap_s * 1000.0
    candidates: list[tuple[int, PreparedSession]] = []
    for session in sessions:
        best: tuple[int, tuple[Round, ...]] | None = None
        for start in range(len(session.rounds) - window_rounds + 1):
            window = session.rounds[start : start + window_rounds]
            prefixes = [round_.prefix_len for round_ in window]
            gaps_ms = [
                right.first_event_ms - left.last_event_ms
                for left, right in zip(window, window[1:], strict=False)
            ]
            if (
                max(prefixes) < min_prefix_tokens
                or max(prefixes) > max_prefix_tokens
                or any(gap_ms < 0 or gap_ms > max_gap_ms for gap_ms in gaps_ms)
                or any(
                    round_.prefix_len + round_.input_len + round_.output_len
                    > max_model_len
                    for round_ in window
                )
            ):
                continue
            prefix_total = sum(prefixes)
            if best is None or prefix_total > best[0]:
                best = (prefix_total, window)
        if best is not None:
            prefix_total, window = best
            candidates.append(
                (prefix_total, PreparedSession(session.source_id, window, 0))
            )
    candidates.sort(key=lambda item: (-item[0], item[1].source_id))
    return [session for _, session in candidates[:max_sessions]]


def write_trace(
    sessions: list[PreparedSession],
    output: Path,
    *,
    arrival_rate: float,
    seed: int,
) -> None:
    """Write official TraceLab closed-loop CSV using real inter-round gaps.

    Args:
        sessions: Selected sessions.
        output: Destination CSV path.
        arrival_rate: Poisson session arrival rate in sessions per second.
        seed: Deterministic arrival seed.

    Returns:
        ``None``.

    Thread-safety:
        Performs synchronous planning-time file IO. The destination must not be
        written concurrently.
    """

    if not sessions:
        raise ValueError("at least one session is required")
    if not math.isfinite(arrival_rate) or arrival_rate <= 0:
        raise ValueError("arrival_rate must be finite and positive")
    rng = random.Random(seed)
    arrival_ms = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDS)
        writer.writeheader()
        for session_id, session in enumerate(sessions):
            if session_id:
                arrival_ms += rng.expovariate(arrival_rate) * 1000.0
            for round_idx, round_ in enumerate(session.rounds):
                gap_ms = 0
                if round_idx + 1 < len(session.rounds):
                    gap_ms = max(
                        0,
                        session.rounds[round_idx + 1].first_event_ms
                        - round_.last_event_ms,
                    )
                writer.writerow(
                    {
                        "id": session_id,
                        "input_len": round_.input_len,
                        "output_len": round_.output_len,
                        "arrival_time": f"{arrival_ms:.6f}",
                        "round_idx": round_idx,
                        "tool_wait_after_ms": gap_ms,
                        "prefix_len": round_.prefix_len,
                    }
                )


def load_rows(database: Path, provider: str) -> list[tuple[Any, ...]]:
    """Load source-ordered token and real event-bound rows from TraceLab.

    Args:
        database: TraceLab DuckDB path.
        provider: ``all``, ``claude``, or ``codex``.

    Returns:
        Query rows accepted by :func:`prepare_sessions`.

    Thread-safety:
        Opens a read-only database connection for this call.
    """

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required only for trace preparation; run this script "
            "from TraceLab's uv environment"
        ) from exc
    where = "" if provider == "all" else "WHERE r.provider = ?"
    params = [] if provider == "all" else [provider]
    query = f"""
        SELECT
            concat(r.project, '|', r.session_id, '|', r.session_file) AS source_id,
            r.newly_append_tokens,
            r.output_tokens,
            r.prefix_tokens,
            min(epoch_ms(t.timestamp)) AS first_event_ms,
            max(epoch_ms(t.timestamp)) AS last_event_ms
        FROM rounds r
        LEFT JOIN timing_events t USING (round_pk)
        {where}
        GROUP BY r.project, r.session_id, r.session_file, r.round_index,
                 r.round_pk, r.ingest_seq, r.newly_append_tokens,
                 r.output_tokens, r.prefix_tokens
        ORDER BY min(r.ingest_seq), r.round_index, r.round_pk
    """
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(query, params).fetchall()
    finally:
        connection.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse TraceLab preparation arguments.

    Args:
        argv: Optional argument vector without the executable name.

    Returns:
        Validated command-line arguments.

    Thread-safety:
        Pure except for ``argparse`` error handling.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1_048_576)
    parser.add_argument("--max-sessions", type=int, default=12)
    parser.add_argument("--arrival-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pause-window-rounds", type=int, default=0)
    parser.add_argument("--dense-window-rounds", type=int, default=0)
    parser.add_argument("--min-pause-seconds", type=float, default=30.0)
    parser.add_argument("--max-pause-seconds", type=float, default=300.0)
    parser.add_argument("--max-dense-gap-seconds", type=float, default=1.0)
    parser.add_argument("--min-dense-prefix-tokens", type=int, default=65_536)
    parser.add_argument("--max-prefix-tokens", type=int, default=260_000)
    parser.add_argument("--provider", choices=("all", "claude", "codex"), default="all")
    args = parser.parse_args(argv)
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if args.pause_window_rounds and args.dense_window_rounds:
        parser.error("pause and dense window modes are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    """Prepare one replay CSV and print its capacity summary.

    Args:
        argv: Optional argument vector without the executable name.

    Returns:
        ``None``.

    Thread-safety:
        Performs synchronous DuckDB and output-file IO as a standalone command.
    """

    args = parse_args(argv)
    sessions = prepare_sessions(load_rows(args.db, args.provider))
    if args.pause_window_rounds:
        selected = select_pause_windows(
            sessions,
            max_sessions=args.max_sessions,
            max_model_len=args.max_model_len,
            window_rounds=args.pause_window_rounds,
            min_pause_s=args.min_pause_seconds,
            max_pause_s=args.max_pause_seconds,
            max_prefix_tokens=args.max_prefix_tokens,
        )
    elif args.dense_window_rounds:
        selected = select_dense_windows(
            sessions,
            max_sessions=args.max_sessions,
            max_model_len=args.max_model_len,
            window_rounds=args.dense_window_rounds,
            max_gap_s=args.max_dense_gap_seconds,
            min_prefix_tokens=args.min_dense_prefix_tokens,
            max_prefix_tokens=args.max_prefix_tokens,
        )
    else:
        selected = select_sessions(
            sessions,
            max_sessions=args.max_sessions,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )
    write_trace(selected, args.out, arrival_rate=args.arrival_rate, seed=args.seed + 1)
    bytes_per_token = (
        slot_size_for_block_tokens(args.model, _BLOCK_TOKENS, args.tensor_parallel_size)
        // _BLOCK_TOKENS
    )
    peak_tokens = sum(
        max(round_.prefix_len for round_ in session.rounds) for session in selected
    )
    summary = {
        "sessions": len(selected),
        "rounds": sum(len(session.rounds) for session in selected),
        "negative_gaps_clamped": sum(session.negative_gaps for session in selected),
        "sum_session_peak_tokens": peak_tokens,
        "sum_session_peak_kv_gib": peak_tokens * bytes_per_token / 1024**3,
        "bytes_per_token": bytes_per_token,
        "output": str(args.out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
