# SPDX-License-Identifier: Apache-2.0
"""Standalone profiling for E2E KV cache benchmarks.

Provides GPU-precise phase timing via ``torch.cuda.Event``, optional
Chrome-trace export via ``torch.profiler``, and structured comparison
reports. All profiling is external to ``daser.connector`` — zero
modifications to the connector code are needed.

Usage::

    from benchmarks.profiler import ProfilerContext, ProfilerReport

    daser_ctx = ProfilerContext("DaseR", trace_dir="/tmp/traces")
    with daser_ctx.phase("cold_generate"):
        outputs = llm.generate(prompts, params)

    ProfilerReport(daser=daser_ctx, lmcache=lmcache_ctx).print()
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any

# Third Party
import torch

# First Party
from daser.perf import load_histograms_json

_METRIC_DISPLAY_ORDER = [
    "ipc_sync_match_and_alloc",
    "ipc_sync_alloc_chunk",
    "load_gds_wait",
    "gds_read",
    "gds_write",
    "load_gpu_copy",
    "save_gpu_copy",
    "save_gds_write",
    "save_ipc_commit",
    "ipc_sync_commit_chunk",
    "ipc_async_commit_chunk",
]


@dataclass
class PhaseResult:
    """Timing and histogram data for one benchmark phase.

    Attributes:
        label: Human-readable phase name (e.g. ``"cold_generate"``).
        wall_elapsed_s: Wall-clock elapsed seconds.
        gpu_elapsed_ms: GPU-stream elapsed milliseconds (0 if no CUDA).
        histograms: Per-operation histogram dict loaded from connector export.
        cache_stats: Optional cache hit/miss counters.
    """

    label: str
    wall_elapsed_s: float
    gpu_elapsed_ms: float = 0.0
    histograms: dict[str, Any] = field(default_factory=dict)
    cache_stats: dict[str, Any] | None = None


class ProfilerContext:
    """Collects timed phases for one system (DaseR or LMCache).

    Uses ``torch.cuda.Event`` for GPU-stream timing and optionally wraps
    each phase with ``torch.profiler.profile()`` for Chrome-trace export.

    Args:
        name: System label used in report headers and NVTX ranges.
        trace_dir: If set, export Chrome traces to this directory.
    """

    def __init__(self, name: str, trace_dir: str | None = None) -> None:
        self.name = name
        self.trace_dir = trace_dir
        self.phases: list[PhaseResult] = []
        self._trace_counter = 0

    def measure_generate(
        self,
        label: str,
        llm: Any,
        prompts: list[Any],
        params: Any,
        histogram_path: str | None = None,
        cache_path: str | None = None,
    ) -> Any:
        """Time one ``llm.generate()`` call and return its outputs.

        Records wall-clock time, GPU-stream time (via CUDA events), and
        an NVTX range marker. When ``trace_dir`` is configured, wraps the
        call with ``torch.profiler.profile()`` and exports a Chrome trace.

        Args:
            label: Phase label, e.g. ``"cold_generate"``.
            llm: vLLM ``LLM`` instance.
            prompts: List of ``TokensPrompt`` objects.
            params: vLLM ``SamplingParams``.
            histogram_path: Path to connector histogram JSON to load after
                the phase completes (if any).
            cache_path: Path to cache statistics JSON (if any).

        Returns:
            The result of ``llm.generate(prompts, params)``.
        """
        nvtx_range = f"{self.name}/{label}"
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(nvtx_range)

        t0 = time.perf_counter()
        start_ev.record()

        if self.trace_dir is not None:
            trace_path = os.path.join(
                self.trace_dir,
                f"{self.name.lower()}_{label}_{self._trace_counter:02d}.json",
            )
            self._trace_counter += 1
            os.makedirs(self.trace_dir, exist_ok=True)

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            ) as prof:
                outputs = llm.generate(prompts, params)
            try:
                prof.export_chrome_trace(trace_path)
            except Exception:
                pass
        else:
            outputs = llm.generate(prompts, params)

        end_ev.record()
        wall_elapsed = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        gpu_elapsed_ms = 0.0
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                gpu_elapsed_ms = start_ev.elapsed_time(end_ev)
            except Exception:
                pass

        histograms: dict[str, Any] = {}
        if histogram_path is not None:
            histograms = load_histograms_json(histogram_path)

        cache_stats: dict[str, Any] | None = None
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                cache_stats = json.load(f)

        self.phases.append(
            PhaseResult(
                label=label,
                wall_elapsed_s=wall_elapsed,
                gpu_elapsed_ms=gpu_elapsed_ms,
                histograms=histograms,
                cache_stats=cache_stats,
            )
        )
        return outputs


class ProfilerReport:
    """Side-by-side comparison report for DaseR vs LMCache profiling.

    Args:
        daser: ``ProfilerContext`` from the DaseR benchmark run.
        lmcache: ``ProfilerContext`` from the LMCache benchmark run.
        prompt_tokens: Total prompt tokens (for tok/s computation).
        cold_histograms: DaseR cold-pass histogram dict loaded after
            LLM shutdown (connector exports on shutdown).
        warm_histograms: DaseR warm-pass histogram dict.
    """

    def __init__(
        self,
        daser: ProfilerContext,
        lmcache: ProfilerContext,
        prompt_tokens: int = 0,
        cold_histograms: dict[str, Any] | None = None,
        warm_histograms: dict[str, Any] | None = None,
    ) -> None:
        self._daser = daser
        self._lmcache = lmcache
        self._prompt_tokens = prompt_tokens
        self._cold_hist = cold_histograms or {}
        self._warm_hist = warm_histograms or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print(self) -> None:
        """Print the full profiling report."""
        self._print_phase_timing()
        self._print_daser_histograms()
        self._print_summary()

    # ------------------------------------------------------------------
    # Phase timing
    # ------------------------------------------------------------------

    def _print_phase_timing(self) -> None:
        """Wall-clock + GPU timing table per phase."""
        all_labels: list[str] = []
        for ctx in (self._daser, self._lmcache):
            for p in ctx.phases:
                if p.label not in all_labels:
                    all_labels.append(p.label)

        has_gpu = any(
            p.gpu_elapsed_ms > 0
            for ctx in (self._daser, self._lmcache)
            for p in ctx.phases
        )

        print("\n" + "=" * 80)
        print("Phase Timing (wall-clock)")
        print("-" * 80)
        header = f"{'Phase':<24}{self._daser.name + '(s)':>16}\
            {self._lmcache.name + '(s)':>16}{'D/L Ratio':>16}"
        print(header)
        print("-" * 80)

        for label in all_labels:
            d_val = self._find_wall(label, self._daser)
            l_val = self._find_wall(label, self._lmcache)
            ratio = f"{d_val / l_val:.2f}x" if (d_val and l_val and l_val > 0) else "—"
            print(
                f"{label:<24}"
                f"{self._fmt_s(d_val):>16}"
                f"{self._fmt_s(l_val):>16}"
                f"{ratio:>16}"
            )

        if has_gpu:
            print()
            print("-" * 80)
            print("Phase Timing (GPU stream)")
            print("-" * 80)
            header_gpu = f"{'Phase':<24}{self._daser.name + '(ms)':>16}\
                {self._lmcache.name + '(ms)':>16}{'D/L Ratio':>16}"
            print(header_gpu)
            print("-" * 80)
            for label in all_labels:
                d_val = self._find_gpu(label, self._daser)
                l_val = self._find_gpu(label, self._lmcache)
                ratio = (
                    f"{d_val / l_val:.2f}x" if (d_val and l_val and l_val > 0) else "—"
                )
                print(
                    f"{label:<24}"
                    f"{self._fmt_ms(d_val):>16}"
                    f"{self._fmt_ms(l_val):>16}"
                    f"{ratio:>16}"
                )

        print("=" * 80)

    # ------------------------------------------------------------------
    # DaseR histograms
    # ------------------------------------------------------------------

    def _print_daser_histograms(self) -> None:
        """Print DaseR per-operation histograms, separated by cold/warm."""
        if self._cold_hist or self._warm_hist:
            cold_hist, warm_hist = self._cold_hist, self._warm_hist
        else:
            cold_hist, warm_hist = self._split_cold_warm_histograms()
        if cold_hist:
            self._print_histogram_section(
                "DaseR Connector Latency Profile (Cold Pass)", cold_hist
            )
        if warm_hist:
            self._print_histogram_section(
                "DaseR Connector Latency Profile (Warm Pass)", warm_hist
            )

    def _split_cold_warm_histograms(self) -> tuple[dict, dict]:
        """Split DaseR phases into cold and warm aggregate histograms."""
        cold: dict[str, Any] = {}
        warm: dict[str, Any] = {}
        for phase in self._daser.phases:
            if not phase.histograms:
                continue
            target = cold if "cold" in phase.label.lower() else warm
            for name, h in phase.histograms.items():
                if name not in target:
                    target[name] = dict(h)
                else:
                    existing = target[name]
                    existing["count"] = existing.get("count", 0) + h.get("count", 0)
                    if existing["count"] > 0 and h.get("count", 0) > 0:
                        w1 = existing["count"]
                        w2 = h["count"]
                        total = w1 + w2
                        existing["mean"] = round(
                            (existing.get("mean", 0) * w1 + h.get("mean", 0) * w2)
                            / total,
                            4,
                        )
        return cold, warm

    def _print_histogram_section(self, title: str, histograms: dict) -> None:
        """Print a single histogram table section."""
        print(f"\n{'─' * 80}")
        print(title)
        print(f"{'─' * 80}")
        header = f"{'Operation':<30}{'p50':>10}{'p95':>10}{'p99':>10}{'count':>8}"
        print(header)
        print("-" * 80)

        printed: set[str] = set()
        for name in _METRIC_DISPLAY_ORDER:
            h = histograms.get(name)
            if h is None or h.get("count", 0) == 0:
                continue
            self._print_hist_row(name, h)
            printed.add(name)

        for name in sorted(histograms.keys()):
            if name in printed:
                continue
            h = histograms[name]
            if h.get("count", 0) == 0:
                continue
            self._print_hist_row(name, h)

        print(f"{'─' * 80}")

    @staticmethod
    def _print_hist_row(name: str, h: dict) -> None:
        unit = h.get("unit", "")
        print(
            f"{name:<30}"
            f"{h.get('p50', 0):>7.2f}{unit}  "
            f"{h.get('p95', 0):>7.2f}{unit}  "
            f"{h.get('p99', 0):>7.2f}{unit}  "
            f"{h.get('count', 0):>8}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Print cold vs warm speedup comparison."""
        d_cold = self._find_wall_with("cold_generate", self._daser)
        d_warm = self._find_wall_with("warm_generate", self._daser)
        l_cold = self._find_wall_with("cold_generate", self._lmcache)
        l_warm = self._find_wall_with("warm_generate", self._lmcache)

        d_cold_tps = self._prompt_tokens / d_cold if d_cold and d_cold > 0 else None
        d_warm_tps = self._prompt_tokens / d_warm if d_warm and d_warm > 0 else None
        l_cold_tps = self._prompt_tokens / l_cold if l_cold and l_cold > 0 else None
        l_warm_tps = self._prompt_tokens / l_warm if l_warm and l_warm > 0 else None

        d_speedup = d_cold / d_warm if (d_cold and d_warm and d_warm > 0) else None
        l_speedup = l_cold / l_warm if (l_cold and l_warm and l_warm > 0) else None

        print(f"\n{'─' * 80}")
        print("Cold vs Warm Summary")
        print(f"{'─' * 80}")
        header = (
            f"{'Metric':<22}"
            f"{'DaseR Cold':>14}"
            f"{'DaseR Warm':>14}"
            f"{'LMCache Cold':>14}"
            f"{'LMCache Warm':>14}"
        )
        print(header)
        print("-" * 80)

        print(
            f"{'Elapsed':<22}"
            f"{self._fmt_s(d_cold):>14}"
            f"{self._fmt_s(d_warm):>14}"
            f"{self._fmt_s(l_cold):>14}"
            f"{self._fmt_s(l_warm):>14}"
        )
        print(
            f"{'tok/s':<22}"
            f"{self._fmt_tps(d_cold_tps):>14}"
            f"{self._fmt_tps(d_warm_tps):>14}"
            f"{self._fmt_tps(l_cold_tps):>14}"
            f"{self._fmt_tps(l_warm_tps):>14}"
        )
        print(
            f"{'Warm/Cold Speedup':<22}"
            f"{self._fmt_speedup(d_speedup):>14}"
            f"{'—':>14}"
            f"{self._fmt_speedup(l_speedup):>14}"
            f"{'—':>14}"
        )

        if d_warm_tps and l_warm_tps and l_warm_tps > 0:
            print("-" * 80)
            print(
                f"DaseR warm tok/s / LMCache warm tok/s = "
                f"{d_warm_tps / l_warm_tps:.2f}x"
            )

        print(f"{'─' * 80}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_wall(label: str, ctx: ProfilerContext) -> float | None:
        for p in ctx.phases:
            if p.label == label:
                return p.wall_elapsed_s
        return None

    @staticmethod
    def _find_gpu(label: str, ctx: ProfilerContext) -> float | None:
        for p in ctx.phases:
            if p.label == label and p.gpu_elapsed_ms > 0:
                return p.gpu_elapsed_ms
        return None

    @staticmethod
    def _find_wall_with(substring: str, ctx: ProfilerContext) -> float | None:
        for p in ctx.phases:
            if substring in p.label:
                return p.wall_elapsed_s
        return None

    @staticmethod
    def _fmt_s(v: float | None) -> str:
        return f"{v:.2f}s" if v is not None else "N/A"

    @staticmethod
    def _fmt_ms(v: float | None) -> str:
        return f"{v:.2f}ms" if v is not None else "N/A"

    @staticmethod
    def _fmt_tps(v: float | None) -> str:
        return f"{v:,.0f}" if v is not None else "N/A"

    @staticmethod
    def _fmt_speedup(v: float | None) -> str:
        return f"{v:.2f}x" if v is not None else "N/A"


def export_trace_json(phases: list[PhaseResult], path: str) -> None:
    """Export phase timing data as a simple JSON trace file.

    This is a lightweight alternative to full Chrome traces produced by
    ``torch.profiler``. The resulting JSON contains one duration event
    per phase and can be loaded in ``chrome://tracing`` or Perfetto.

    Args:
        phases: List of PhaseResult from a ProfilerContext.
        path: File path to write the JSON trace to.
    """
    events: list[dict[str, Any]] = []
    t0 = 0.0
    for i, p in enumerate(phases):
        start_us = int(t0 * 1_000_000)
        dur_us = int(p.wall_elapsed_s * 1_000_000)
        events.append(
            {
                "name": p.label,
                "cat": "benchmark",
                "ph": "X",
                "ts": start_us,
                "dur": dur_us,
                "pid": 0,
                "tid": i,
                "args": {
                    "wall_s": round(p.wall_elapsed_s, 4),
                    "gpu_ms": round(p.gpu_elapsed_ms, 2),
                },
            }
        )
        t0 += p.wall_elapsed_s

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f, indent=2)
