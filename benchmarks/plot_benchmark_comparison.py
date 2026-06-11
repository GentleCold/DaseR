# SPDX-License-Identifier: Apache-2.0
"""Plot benchmark comparison figures from unified benchmark results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeriesPoint:
    """One plotted benchmark series point.

    Args:
        mode: Cache reuse mode label.
        backend: Backend label.
        phase: Benchmark phase label.
        mean_ttft_ms: Mean time to first token in milliseconds.
        p99_ttft_ms: P99 time to first token in milliseconds.
        all_requests_elapsed_s: Wall-clock time until all requests complete in seconds.
        hit_rate: External-prefix hit rate, if available.
        answer_accuracy: LongBench answer containment accuracy.
        exact_accuracy: Cold/warm exact generated-text accuracy, if available.

    Thread-safety:
        Immutable value object.
    """

    mode: str
    backend: str
    phase: str
    mean_ttft_ms: float
    p99_ttft_ms: float
    all_requests_elapsed_s: float
    hit_rate: float | None
    answer_accuracy: float | None
    exact_accuracy: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.

    Thread-safety:
        Pure aside from reading process argv.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-run", required=True, help="Chunk-mode run root.")
    parser.add_argument("--prefix-run", required=True, help="Prefix-mode run root.")
    parser.add_argument(
        "--out-dir",
        default="benchmarks/figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--name",
        default="longbench_out128_comparison",
        help="Output file stem.",
    )
    parser.add_argument(
        "--show-answer-accuracy",
        action="store_true",
        help="Include LongBench answer accuracy. Intended for output-128 runs.",
    )
    return parser.parse_args()


def load_points(mode: str, run_root: Path) -> list[SeriesPoint]:
    """Load plot points from one benchmark run root.

    Args:
        mode: Human-readable reuse mode label.
        run_root: Directory containing backend results.

    Returns:
        Plot points for baseline and warm service phases.

    Thread-safety:
        Reads JSON files and keeps no shared state.
    """
    points: list[SeriesPoint] = []
    for backend in ("vllm", "lmcache", "daser"):
        path = run_root / backend / "results.json"
        data = json.loads(path.read_text())
        result = data["result"]
        if backend == "vllm":
            points.append(
                _point_from_phase(mode, "vLLM", "Baseline", result["baseline"])
            )
            continue

        exact_accuracy = _exact_accuracy(result)
        label = "LMCache" if backend == "lmcache" else "DaseR"
        if mode == "Chunk" and backend == "daser":
            phase = result["warm"]
        else:
            phase = result["warm"]
        points.append(
            _point_from_phase(
                mode,
                label,
                "Warm",
                phase,
                exact_accuracy=exact_accuracy,
            )
        )
    return points


def plot(
    points: list[SeriesPoint],
    out_dir: Path,
    name: str,
    show_answer_accuracy: bool = False,
) -> list[Path]:
    """Render a publication-style benchmark summary figure.

    Args:
        points: Benchmark points.
        out_dir: Output directory.
        name: Output file stem.
        show_answer_accuracy: Whether to include answer-accuracy bars.

    Returns:
        Paths of generated image files.

    Thread-safety:
        Uses matplotlib global state and is not thread-safe.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    modes = ["Chunk", "Prefix"]
    backends = ["vLLM", "LMCache", "DaseR"]
    palette = {"vLLM": "#4A5568", "LMCache": "#2B6CB0", "DaseR": "#2F855A"}

    panel_count = 3 if show_answer_accuracy else 2
    fig = plt.figure(
        figsize=(12.0, 4.4 if panel_count == 2 else 4.8),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(1, panel_count)
    ax_ttft = fig.add_subplot(grid[0, 0])
    ax_elapsed = fig.add_subplot(grid[0, 1])

    _ttft_bars(
        ax_ttft,
        points,
        modes,
        backends,
        palette,
    )
    _grouped_bars(
        ax_elapsed,
        points,
        modes,
        backends,
        palette,
        value=lambda point: point.all_requests_elapsed_s,
        ylabel="Elapsed Time (s)",
        title="All Requests Completed",
        fmt="{:.1f}",
    )
    if show_answer_accuracy:
        ax_accuracy = fig.add_subplot(grid[0, 2])
        _answer_accuracy_bars(ax_accuracy, points, modes, backends, palette)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[backend], label=backend)
        for backend in backends
    ]
    handles.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="#2D3748",
            hatch="////",
            label="P99 TTFT",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(
        "LongBench Benchmark: Baseline vs. Cache Reuse",
        y=1.08,
        fontsize=15,
        fontweight="bold",
    )

    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _point_from_phase(
    mode: str,
    backend: str,
    phase_name: str,
    phase: dict[str, Any],
    exact_accuracy: float | None = None,
) -> SeriesPoint:
    summary = phase["summary"]
    requests = phase.get("requests", [])
    return SeriesPoint(
        mode=mode,
        backend=backend,
        phase=phase_name,
        mean_ttft_ms=float(summary.get("ttft_ms_mean", 0.0)),
        p99_ttft_ms=_percentile(
            [float(request.get("ttft_ms", 0.0)) for request in requests], 99
        ),
        all_requests_elapsed_s=_elapsed_seconds(summary),
        hit_rate=_optional_float(summary.get("vllm_external_prefix_cache_hit_rate")),
        answer_accuracy=_optional_float(summary.get("answer_contains_accuracy")),
        exact_accuracy=exact_accuracy,
    )


def _exact_accuracy(result: dict[str, Any]) -> float | None:
    correctness = result.get("correctness", {})
    exact = correctness.get("cold_warm_exact_match", {})
    return _optional_float(exact.get("accuracy"))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _elapsed_seconds(summary: dict[str, Any]) -> float:
    elapsed_ms = summary.get(
        "all_requests_elapsed_ms",
        summary.get("phase_elapsed_ms", 0.0),
    )
    return float(elapsed_ms) / 1000.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * percentile / 100) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _configure_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": [
                "JetBrains Mono",
                "IBM Plex Mono",
                "DejaVu Sans Mono",
                "Liberation Mono",
            ],
            "axes.edgecolor": "#2D3748",
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
        }
    )


def _ttft_bars(
    ax: Any,
    points: list[SeriesPoint],
    modes: list[str],
    backends: list[str],
    palette: dict[str, str],
) -> None:
    width = 0.11
    x_positions = list(range(len(modes)))
    by_key = {(point.mode, point.backend): point for point in points}
    metric_offsets = (-width / 2, width / 2)
    backend_offsets = (-width * 2.4, 0.0, width * 2.4)
    for backend_offset, backend in zip(backend_offsets, backends, strict=True):
        for metric_offset, attr, hatch, alpha in (
            (metric_offsets[0], "mean_ttft_ms", "", 0.96),
            (metric_offsets[1], "p99_ttft_ms", "////", 0.70),
        ):
            values = [
                getattr(by_key[(mode, backend)], attr)
                if (mode, backend) in by_key
                else 0.0
                for mode in modes
            ]
            bars = ax.bar(
                [x + backend_offset + metric_offset for x in x_positions],
                values,
                width=width,
                color=palette[backend],
                edgecolor="#2D3748" if hatch else "white",
                linewidth=0.5,
                hatch=hatch,
                alpha=alpha,
            )
            if attr == "mean_ttft_ms":
                _label_bars(ax, bars, fmt="{:.0f}")
    ax.set_xticks(x_positions, modes)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Mean and P99 TTFT", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def _grouped_bars(
    ax: Any,
    points: list[SeriesPoint],
    modes: list[str],
    backends: list[str],
    palette: dict[str, str],
    value: Any,
    ylabel: str,
    title: str,
    fmt: str = "{:.0f}",
) -> None:
    width = 0.23
    x_positions = list(range(len(modes)))
    by_key = {(point.mode, point.backend): point for point in points}
    for offset, backend in zip((-width, 0.0, width), backends, strict=True):
        values = [
            value(by_key[(mode, backend)]) if (mode, backend) in by_key else 0.0
            for mode in modes
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=palette[backend],
            edgecolor="white",
            linewidth=0.7,
        )
        _label_bars(ax, bars, fmt=fmt)
    ax.set_xticks(x_positions, modes)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def _answer_accuracy_bars(
    ax: Any,
    points: list[SeriesPoint],
    modes: list[str],
    backends: list[str],
    palette: dict[str, str],
) -> None:
    width = 0.23
    x_positions = list(range(len(modes)))
    by_key = {(point.mode, point.backend): point for point in points}
    for offset, backend in zip((-width, 0.0, width), backends, strict=True):
        values = []
        for mode in modes:
            point = by_key.get((mode, backend))
            metric = point.answer_accuracy if point else None
            values.append(0.0 if metric is None else metric * 100)
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=palette[backend],
            edgecolor="white",
            linewidth=0.7,
        )
        _label_bars(ax, bars, fmt="{:.0f}%")
    ax.set_xticks(x_positions, modes)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Answer Accuracy (%)")
    ax.set_title("LongBench Answer Accuracy", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def _hit_bars(
    ax: Any,
    points: list[SeriesPoint],
    modes: list[str],
    palette: dict[str, str],
) -> None:
    width = 0.28
    x_positions = list(range(len(modes)))
    by_key = {(point.mode, point.backend): point for point in points}
    for offset, backend in zip(
        (-width / 2, width / 2), ("LMCache", "DaseR"), strict=True
    ):
        values = []
        for mode in modes:
            point = by_key.get((mode, backend))
            values.append(
                0.0 if point is None or point.hit_rate is None else point.hit_rate * 100
            )
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=palette[backend],
            edgecolor="white",
            linewidth=0.7,
        )
        _label_bars(ax, bars, fmt="{:.1f}%")
    ax.set_xticks(x_positions, modes)
    ax.set_ylim(0, 105)
    ax.set_ylabel("External Prefix Hit Rate (%)")
    ax.set_title("Warm Cache Hit Rate", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def _label_bars(ax: Any, bars: Any, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    points = [
        *load_points("Chunk", Path(args.chunk_run)),
        *load_points("Prefix", Path(args.prefix_run)),
    ]
    outputs = plot(
        points,
        Path(args.out_dir),
        args.name,
        show_answer_accuracy=args.show_answer_accuracy,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
