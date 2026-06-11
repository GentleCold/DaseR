# SPDX-License-Identifier: Apache-2.0
"""Dataset abstractions for benchmark workloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random

from benchmarks.utils.constants import DEFAULT_IMDB_QUESTION


@dataclass(frozen=True)
class BenchmarkSample:
    """One benchmark sample normalized across datasets.

    Args:
        sample_id: Stable integer ID within a loaded workload.
        dataset: Dataset name.
        context: Document/context text.
        question: Task or question text.
        answers: Acceptable answers for contains scoring.

    Thread-safety:
        Immutable value object; safe to share between threads.
    """

    sample_id: int
    dataset: str
    context: str
    question: str
    answers: list[str]


class BenchmarkDataset(ABC):
    """Abstract benchmark dataset loader."""

    @abstractmethod
    def load(self) -> list[BenchmarkSample]:
        """Load normalized benchmark samples.

        Returns:
            List of BenchmarkSample objects.

        Thread-safety:
            Implementations perform local file reads and keep no global state.
        """


class ImdbDataset(BenchmarkDataset):
    """Load IMDB CSV reviews as benchmark samples."""

    def __init__(
        self,
        path: str | Path,
        max_samples: int = 200,
        question: str = DEFAULT_IMDB_QUESTION,
    ) -> None:
        """Initialize the IMDB loader.

        Args:
            path: CSV path with a ``review`` column.
            max_samples: Maximum number of reviews to load; 0 means all.
            question: Shared task text for every review.
        """
        self._path = Path(path)
        self._max_samples = max_samples
        self._question = question

    def load(self) -> list[BenchmarkSample]:
        """Load IMDB reviews as benchmark samples."""
        if not self._path.is_file():
            raise FileNotFoundError(f"IMDB CSV not found: {self._path}")
        samples: list[BenchmarkSample] = []
        with self._path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                review = row.get("review", "").strip()
                if not review:
                    continue
                samples.append(
                    BenchmarkSample(
                        sample_id=len(samples),
                        dataset="imdb",
                        context=review,
                        question=self._question,
                        answers=[],
                    )
                )
                if self._max_samples > 0 and len(samples) >= self._max_samples:
                    break
        return samples


class LongBenchDataset(BenchmarkDataset):
    """Load one or more LongBench JSONL files."""

    def __init__(
        self,
        data_dir: str | Path,
        datasets: list[str] | None = None,
        max_samples: int = 0,
    ) -> None:
        """Initialize the LongBench loader.

        Args:
            data_dir: Directory containing ``*.jsonl`` files.
            datasets: Dataset names without extension; None means defaults are
                discovered from the directory.
            max_samples: Maximum samples per dataset; 0 means all.
        """
        self._data_dir = Path(data_dir)
        self._datasets = datasets
        self._max_samples = max_samples

    def load(self) -> list[BenchmarkSample]:
        """Load LongBench JSONL records as benchmark samples."""
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"LongBench dir not found: {self._data_dir}")
        names = self._datasets or sorted(
            path.stem for path in self._data_dir.glob("*.jsonl")
        )
        samples: list[BenchmarkSample] = []
        for name in names:
            path = self._data_dir / f"{name}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"LongBench JSONL not found: {path}")
            loaded_for_dataset = 0
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    answers = rec.get("answers", [])
                    if isinstance(answers, str):
                        answers = [answers]
                    samples.append(
                        BenchmarkSample(
                            sample_id=len(samples),
                            dataset=name,
                            context=str(rec.get("context", "")),
                            question=str(rec.get("input", "")),
                            answers=[str(answer) for answer in answers],
                        )
                    )
                    loaded_for_dataset += 1
                    if (
                        self._max_samples > 0
                        and loaded_for_dataset >= self._max_samples
                    ):
                        break
        return samples


def interleave_samples(
    samples: list[BenchmarkSample],
    seed: int = 42,
) -> list[BenchmarkSample]:
    """Interleave samples by dataset to reduce queue-order bias.

    Args:
        samples: Input sample list.
        seed: Shuffle seed within each dataset.

    Returns:
        Interleaved sample list.

    Thread-safety:
        Pure function.
    """
    rng = random.Random(seed)
    by_dataset: dict[str, list[BenchmarkSample]] = {}
    for sample in samples:
        by_dataset.setdefault(sample.dataset, []).append(sample)
    for dataset_samples in by_dataset.values():
        rng.shuffle(dataset_samples)

    result: list[BenchmarkSample] = []
    indices = {name: 0 for name in by_dataset}
    while True:
        added = False
        for name in sorted(by_dataset):
            idx = indices[name]
            dataset_samples = by_dataset[name]
            if idx < len(dataset_samples):
                result.append(dataset_samples[idx])
                indices[name] = idx + 1
                added = True
        if not added:
            return result


def dedup_by_context(samples: list[BenchmarkSample]) -> list[BenchmarkSample]:
    """Keep the first sample for each unique context.

    Args:
        samples: Input sample list.

    Returns:
        Deduplicated samples preserving first occurrence order.

    Thread-safety:
        Pure function.
    """
    seen: set[str] = set()
    result: list[BenchmarkSample] = []
    for sample in samples:
        if sample.context in seen:
            continue
        seen.add(sample.context)
        result.append(sample)
    return result
