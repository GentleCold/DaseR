# SPDX-License-Identifier: Apache-2.0
"""Dataset abstractions for benchmark workloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
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
    """Abstract benchmark dataset loader.

    Subclasses own their dataset name, their CLI arguments, and how to build
    themselves from parsed args, so adding a dataset needs no edits to the
    benchmark runner or load generator: implement the three hooks and register
    the class with ``register_dataset``.
    """

    #: Registry key; also the value accepted by ``--dataset``.
    name: str = ""

    @abstractmethod
    def load(self) -> list[BenchmarkSample]:
        """Load normalized benchmark samples.

        Returns:
            List of BenchmarkSample objects.

        Thread-safety:
            Implementations perform local file reads and keep no global state.
        """

    @classmethod  # noqa: B027
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register this dataset's CLI arguments on ``parser``.

        Args:
            parser: argument parser shared by all datasets. Implementations add
                only their own options and tolerate being called alongside
                sibling datasets, so option names must not collide.
        """

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> "BenchmarkDataset":
        """Build a dataset instance from parsed CLI args.

        Args:
            args: parsed argparse namespace containing this dataset's options
                plus the shared ``max_samples``.

        Returns:
            A ready-to-load dataset instance.
        """


_DATASET_REGISTRY: dict[str, type[BenchmarkDataset]] = {}


def register_dataset(cls: type[BenchmarkDataset]) -> type[BenchmarkDataset]:
    """Register a dataset class under its ``name`` for CLI dispatch.

    Args:
        cls: dataset class with a non-empty ``name``.

    Returns:
        The same class, so this can be used as a decorator.

    Raises:
        ValueError: if ``name`` is empty or already registered.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty name")
    if cls.name in _DATASET_REGISTRY:
        raise ValueError(f"dataset name already registered: {cls.name}")
    _DATASET_REGISTRY[cls.name] = cls
    return cls


def dataset_names() -> tuple[str, ...]:
    """Return registered dataset names in registration order."""
    return tuple(_DATASET_REGISTRY)


def add_dataset_cli_args(
    parser: argparse.ArgumentParser, default: str | None = None
) -> None:
    """Add ``--dataset`` plus every registered dataset's own CLI args.

    Args:
        parser: parser to extend with the dataset selector and per-dataset
            options.
        default: default dataset name; falls back to the first registered.
    """
    names = dataset_names()
    parser.add_argument(
        "--dataset",
        choices=names,
        default=default if default is not None else (names[0] if names else None),
    )
    for cls in _DATASET_REGISTRY.values():
        cls.add_cli_args(parser)


def build_dataset(args: argparse.Namespace) -> BenchmarkDataset:
    """Build the selected dataset from parsed args via the registry.

    Args:
        args: parsed argparse namespace with a ``dataset`` selector.

    Returns:
        The dataset instance for ``args.dataset``.

    Raises:
        ValueError: if ``args.dataset`` is not registered.
    """
    cls = _DATASET_REGISTRY.get(args.dataset)
    if cls is None:
        raise ValueError(f"unknown dataset: {args.dataset}")
    return cls.from_args(args)


@register_dataset
class ImdbDataset(BenchmarkDataset):
    """Load IMDB CSV reviews as benchmark samples."""

    name = "imdb"

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

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register the IMDB CSV path option."""
        parser.add_argument("--imdb", help="IMDB CSV path with a review column")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ImdbDataset":
        """Build an IMDB dataset from parsed args."""
        if not getattr(args, "imdb", None):
            raise ValueError("--imdb is required for --dataset imdb")
        return cls(args.imdb, max_samples=args.max_samples)

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


@register_dataset
class LongBenchDataset(BenchmarkDataset):
    """Load one or more LongBench JSONL files."""

    name = "longbench"

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

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register the LongBench directory and dataset-subset options."""
        parser.add_argument(
            "--longbench-dir", help="Directory containing LongBench *.jsonl files"
        )
        parser.add_argument(
            "--datasets",
            default=None,
            help="Comma-separated LongBench dataset names; default discovers all",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LongBenchDataset":
        """Build a LongBench dataset from parsed args."""
        if not getattr(args, "longbench_dir", None):
            raise ValueError("--longbench-dir is required for --dataset longbench")
        datasets = None
        if getattr(args, "datasets", None):
            datasets = [
                item.strip() for item in args.datasets.split(",") if item.strip()
            ]
        return cls(args.longbench_dir, datasets=datasets, max_samples=args.max_samples)

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
