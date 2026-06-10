# SPDX-License-Identifier: Apache-2.0
"""Accuracy baseline: chunk-mode KV reuse vs full prefill on LongBench QA.

Drives the production DaseR HTTP service (``/documents`` + ``/infer``) for
multi-document QA samples. Each sample's passages are uploaded as separate
documents (independently prefilled chunks), then the same assembled prompt is
answered twice with identical greedy decoding:

- ``full``:  ``use_kv_cache=false`` -> full prefill (accuracy upper bound).
- ``reuse``: ``use_kv_cache=true``  -> concatenated chunk KV, RoPE-relocated,
  no cross-attention repair (the lossy path under measurement).

Reported per dataset: LongBench-style QA F1 for both modes, the F1 gap,
extracted-answer F1 (bold span / first sentence, separating factual errors
from verbosity-induced format drift), gold-answer recall, normalized answer
agreement, chunk-hit token coverage, and TTFT. Accuracy means are computed
over samples whose reuse request verifiably loaded chunk KV; samples that
silently fell back to full recompute are excluded and reported separately.

Usage:
    python benchmarks/bench_accuracy_baseline.py \\
        --daser-url http://127.0.0.1:2046 \\
        --tokenizer /path/to/model \\
        --data-dir /path/to/daser-baseline/data \\
        --datasets hotpotqa,2wikimqa,musique \\
        --num-samples 50 \\
        --out /path/to/daser-baseline/results/baseline.json

The DaseR server must run with ``--cache-reuse-mode chunk`` and vLLM must be
started with prefix caching disabled (the standard DaseR deployment).
"""

from __future__ import annotations

# Standard
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import string
import sys
import time
from typing import Any

# Third Party
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

PASSAGE_SPLIT_RE = re.compile(r"(?=Passage \d+:\n)")
BOLD_SPAN_RE = re.compile(r"\*\*(.+?)\*\*")
TASK_INSTRUCTION = (
    "Answer the question based on the given documents. Only give me the "
    "answer and do not output any other words.\n\nQuestion: {question}"
)
HTTP_TIMEOUT_S = 600.0
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_SAMPLES = 50
DEFAULT_MAX_CONTEXT_TOKENS = 24000


@dataclass
class SampleResult:
    """Accuracy comparison result for one QA sample.

    Attributes:
        sample_id: LongBench record ``_id``.
        question: QA question text.
        gold_answers: reference answers.
        num_docs: number of uploaded passage documents.
        prompt_tokens: assembled prompt length reported by ``/infer``.
        full_text: answer text from the full-prefill mode.
        reuse_text: answer text from the chunk-reuse mode.
        full_f1: QA F1 of the full-prefill answer.
        reuse_f1: QA F1 of the chunk-reuse answer.
        full_f1_extracted: QA F1 of the extracted full-prefill answer span.
        reuse_f1_extracted: QA F1 of the extracted chunk-reuse answer span.
        full_recall: gold-answer substring recall of full prefill (0/1).
        reuse_recall: gold-answer substring recall of chunk reuse (0/1).
        agreement: normalized exact match between the two mode answers.
        reuse_coverage: fraction of prompt tokens served from cached chunks.
        full_ttft_ms: TTFT of the full-prefill request.
        reuse_ttft_ms: TTFT of the chunk-reuse request.
        reuse_loaded: whether the reuse path actually loaded cached KV,
            inferred from reuse TTFT being meaningfully below full TTFT.
        error: failure description when the sample was skipped.
    """

    sample_id: str
    question: str
    gold_answers: list[str]
    num_docs: int = 0
    prompt_tokens: int = 0
    full_text: str = ""
    reuse_text: str = ""
    full_f1: float = 0.0
    reuse_f1: float = 0.0
    full_f1_extracted: float = 0.0
    reuse_f1_extracted: float = 0.0
    full_recall: float = 0.0
    reuse_recall: float = 0.0
    agreement: bool = False
    reuse_coverage: float = 0.0
    full_ttft_ms: float = 0.0
    reuse_ttft_ms: float = 0.0
    reuse_loaded: bool = False
    error: str = ""


@dataclass
class DatasetSummary:
    """Aggregated accuracy metrics for one dataset.

    Accuracy means (`full_f1` .. `agreement_rate`) are computed only over
    samples whose reuse request verifiably loaded chunk KV (``reuse_loaded``),
    so silent full-recompute fallbacks cannot dilute the comparison.

    Attributes:
        dataset: dataset name.
        samples: number of scored samples (no error).
        samples_loaded: scored samples whose reuse path actually loaded KV;
            accuracy means below are over this subset.
        skipped: number of skipped samples (errors / over-length).
        full_f1: mean QA F1 of full prefill.
        reuse_f1: mean QA F1 of chunk reuse.
        f1_gap: ``full_f1 - reuse_f1``.
        full_f1_extracted: mean QA F1 of the extracted full-prefill answer.
        reuse_f1_extracted: mean QA F1 of the extracted chunk-reuse answer.
        f1_gap_extracted: ``full_f1_extracted - reuse_f1_extracted``.
        full_recall: mean gold-answer substring recall of full prefill.
        reuse_recall: mean gold-answer substring recall of chunk reuse.
        recall_gap: ``full_recall - reuse_recall``.
        agreement_rate: fraction of samples with matching normalized answers.
        mean_coverage: mean chunk-hit token coverage over scored samples.
        loaded_rate: fraction of scored samples with a verified KV load.
        mean_full_ttft_ms: mean full-prefill TTFT over scored samples.
        mean_reuse_ttft_ms: mean chunk-reuse TTFT over scored samples.
        results: per-sample records.
    """

    dataset: str
    samples: int = 0
    samples_loaded: int = 0
    skipped: int = 0
    full_f1: float = 0.0
    reuse_f1: float = 0.0
    f1_gap: float = 0.0
    full_f1_extracted: float = 0.0
    reuse_f1_extracted: float = 0.0
    f1_gap_extracted: float = 0.0
    full_recall: float = 0.0
    reuse_recall: float = 0.0
    recall_gap: float = 0.0
    agreement_rate: float = 0.0
    mean_coverage: float = 0.0
    loaded_rate: float = 0.0
    mean_full_ttft_ms: float = 0.0
    mean_reuse_ttft_ms: float = 0.0
    results: list[SampleResult] = field(default_factory=list)


def normalize_answer(text: str) -> str:
    """Normalize a QA answer for comparison (LongBench convention).

    Args:
        text: raw answer text.

    Returns:
        Lower-cased text without articles, punctuation, or extra whitespace.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Token-level QA F1 against the best-matching gold answer.

    Args:
        prediction: model answer text.
        gold_answers: list of acceptable reference answers.

    Returns:
        Maximum F1 over the gold answers, in [0, 1].
    """
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_recall(prediction: str, gold_answers: list[str]) -> float:
    """Return 1.0 when any gold answer appears in the prediction.

    Substring containment after normalization. Robust to answer verbosity,
    so it isolates factual correctness from instruction-following format
    (terse vs verbose) drift that token-level F1 conflates.

    Args:
        prediction: model answer text.
        gold_answers: list of acceptable reference answers.

    Returns:
        1.0 if a normalized gold answer is contained in the normalized
        prediction, else 0.0.
    """
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in pred_norm:
            return 1.0
    return 0.0


def extract_short_answer(text: str) -> str:
    """Extract the concise answer span from a possibly verbose response.

    Degraded chunk-reuse answers tend to wrap the correct answer in
    explanatory prose (often bolding the answer itself). Scoring the
    extracted span separates factual errors from verbosity-induced
    token-F1 dilution.

    Args:
        text: raw model answer text.

    Returns:
        The first bold ``**span**`` when present, otherwise the first
        sentence of the first line.
    """
    match = BOLD_SPAN_RE.search(text)
    if match:
        return match.group(1)
    first_line = text.strip().split("\n", 1)[0]
    return first_line.split(". ", 1)[0]


def split_passages(context: str) -> list[str]:
    """Split a LongBench context into passage documents.

    Args:
        context: raw context text, typically ``Passage N:`` delimited.

    Returns:
        Non-empty passage texts; the whole context when no markers exist.
    """
    parts = [part.strip() for part in PASSAGE_SPLIT_RE.split(context)]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [context.strip()] if context.strip() else []
    return parts


def load_samples(path: Path, limit: int) -> list[dict[str, Any]]:
    """Load LongBench JSONL records.

    Args:
        path: dataset JSONL path.
        limit: maximum number of records (0 = all).

    Returns:
        Parsed records in file order.
    """
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


class DaserClient:
    """Thin synchronous client for the DaseR HTTP service.

    Args:
        base_url: DaseR HTTP server base URL.

    Async/thread-safety:
        Uses one ``requests.Session`` from a single benchmark thread.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def health(self) -> bool:
        """Return whether DaseR and vLLM report healthy."""
        resp = self._session.get(f"{self._base_url}/health", timeout=30)
        resp.raise_for_status()
        return resp.json().get("status") == "ok"

    def upload_document(self, title: str, text: str) -> str:
        """Upload one document and return its doc_id."""
        resp = self._session.post(
            f"{self._base_url}/documents",
            json={"title": title, "text": text},
            timeout=HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        return str(resp.json()["doc_id"])

    def delete_document(self, doc_id: str) -> None:
        """Delete one document, ignoring missing IDs."""
        try:
            self._session.delete(
                f"{self._base_url}/documents/{doc_id}",
                timeout=HTTP_TIMEOUT_S,
            )
        except requests.RequestException as exc:  # pragma: no cover - cleanup
            logger.warning("delete_document %s failed: %s", doc_id[:8], exc)

    def infer(
        self,
        doc_ids: list[str],
        task: str,
        use_kv_cache: bool,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Run one ``/infer`` request with greedy decoding.

        Args:
            doc_ids: documents to include in the prompt.
            task: task text appended after the documents.
            use_kv_cache: enable DaseR chunk KV loading.
            max_new_tokens: generation cap.

        Returns:
            ``/infer`` response payload.
        """
        resp = self._session.post(
            f"{self._base_url}/infer",
            json={
                "doc_ids": doc_ids,
                "task": task,
                "use_kv_cache": use_kv_cache,
                "trace_cache": use_kv_cache,
                "gen_params": {
                    "max_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            },
            timeout=HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()


def hit_token_coverage(response: dict[str, Any]) -> float:
    """Return the fraction of prompt tokens covered by cache hits.

    Args:
        response: ``/infer`` response with ``cache_hits`` trace entries.

    Returns:
        Covered tokens / prompt tokens, in [0, 1].
    """
    prompt_tokens = int(response.get("prompt_tokens", 0))
    if prompt_tokens <= 0:
        return 0.0
    covered = sum(
        int(hit.get("token_count", 0)) for hit in response.get("cache_hits", [])
    )
    return min(1.0, covered / prompt_tokens)


def run_sample(
    client: DaserClient,
    record: dict[str, Any],
    max_new_tokens: int,
    delete_docs: bool,
) -> SampleResult:
    """Measure one sample in both modes.

    Args:
        client: DaseR HTTP client.
        record: LongBench QA record.
        max_new_tokens: generation cap per request.
        delete_docs: delete uploaded documents after the sample. Keeping
            them avoids delete/ring-eviction interplay and lets identical
            passages dedup across samples.

    Returns:
        Populated SampleResult; ``error`` is set on failure.
    """
    result = SampleResult(
        sample_id=str(record.get("_id", "")),
        question=str(record.get("input", "")),
        gold_answers=[str(ans) for ans in record.get("answers", [])],
    )
    passages = split_passages(str(record.get("context", "")))
    if not passages:
        result.error = "empty context"
        return result

    doc_ids: list[str] = []
    try:
        for i, passage in enumerate(passages):
            doc_ids.append(
                client.upload_document(f"{result.sample_id}-p{i:02d}", passage)
            )
        result.num_docs = len(doc_ids)
        task = TASK_INSTRUCTION.format(question=result.question)

        full = client.infer(
            doc_ids, task, use_kv_cache=False, max_new_tokens=max_new_tokens
        )
        reuse = client.infer(
            doc_ids, task, use_kv_cache=True, max_new_tokens=max_new_tokens
        )
    except requests.RequestException as exc:
        result.error = f"http error: {exc}"
        return result
    finally:
        if delete_docs:
            for doc_id in doc_ids:
                client.delete_document(doc_id)

    result.prompt_tokens = int(full.get("prompt_tokens", 0))
    result.full_text = str(full.get("text", "")).strip()
    result.reuse_text = str(reuse.get("text", "")).strip()
    result.full_f1 = qa_f1_score(result.full_text, result.gold_answers)
    result.reuse_f1 = qa_f1_score(result.reuse_text, result.gold_answers)
    result.full_f1_extracted = qa_f1_score(
        extract_short_answer(result.full_text), result.gold_answers
    )
    result.reuse_f1_extracted = qa_f1_score(
        extract_short_answer(result.reuse_text), result.gold_answers
    )
    result.full_recall = answer_recall(result.full_text, result.gold_answers)
    result.reuse_recall = answer_recall(result.reuse_text, result.gold_answers)
    result.agreement = normalize_answer(result.full_text) == normalize_answer(
        result.reuse_text
    )
    result.reuse_coverage = hit_token_coverage(reuse)
    result.full_ttft_ms = float(full.get("ttft_ms") or 0.0)
    result.reuse_ttft_ms = float(reuse.get("ttft_ms") or 0.0)
    # A genuine chunk-reuse load skips most prefill, so its TTFT is well
    # below full prefill. When reuse TTFT is not clearly lower, the reuse
    # path silently fell back to full recompute (e.g. evicted prefix), which
    # makes the accuracy comparison meaningless for that sample.
    result.reuse_loaded = result.reuse_ttft_ms < 0.85 * result.full_ttft_ms
    return result


def summarize(dataset: str, results: list[SampleResult]) -> DatasetSummary:
    """Aggregate per-sample results for one dataset.

    Accuracy means are computed only over samples with a verified chunk-KV
    load (``reuse_loaded``); coverage, loaded rate, and TTFT means are over
    all scored samples.

    Args:
        dataset: dataset name.
        results: all per-sample records including failures.

    Returns:
        DatasetSummary with means as described above.
    """
    summary = DatasetSummary(dataset=dataset, results=results)
    scored = [r for r in results if not r.error]
    loaded = [r for r in scored if r.reuse_loaded]
    summary.samples = len(scored)
    summary.samples_loaded = len(loaded)
    summary.skipped = len(results) - len(scored)
    if not scored:
        return summary
    n = float(len(scored))
    summary.mean_coverage = sum(r.reuse_coverage for r in scored) / n
    summary.loaded_rate = len(loaded) / n
    summary.mean_full_ttft_ms = sum(r.full_ttft_ms for r in scored) / n
    summary.mean_reuse_ttft_ms = sum(r.reuse_ttft_ms for r in scored) / n
    if not loaded:
        return summary
    m = float(len(loaded))
    summary.full_f1 = sum(r.full_f1 for r in loaded) / m
    summary.reuse_f1 = sum(r.reuse_f1 for r in loaded) / m
    summary.f1_gap = summary.full_f1 - summary.reuse_f1
    summary.full_f1_extracted = sum(r.full_f1_extracted for r in loaded) / m
    summary.reuse_f1_extracted = sum(r.reuse_f1_extracted for r in loaded) / m
    summary.f1_gap_extracted = summary.full_f1_extracted - summary.reuse_f1_extracted
    summary.full_recall = sum(r.full_recall for r in loaded) / m
    summary.reuse_recall = sum(r.reuse_recall for r in loaded) / m
    summary.recall_gap = summary.full_recall - summary.reuse_recall
    summary.agreement_rate = sum(1 for r in loaded if r.agreement) / m
    return summary


def print_report(summaries: list[DatasetSummary]) -> None:
    """Print the aggregate accuracy comparison table.

    Args:
        summaries: per-dataset aggregates.
    """
    print("\n" + "=" * 118)
    print("ACCURACY BASELINE — chunk-mode KV reuse vs full prefill (greedy)")
    print("=" * 118)
    header = (
        f"{'Dataset':<12} {'N':>4} {'Full F1':>8} {'Reuse F1':>9} {'F1 gap':>8} "
        f"{'ExtF1 gap':>10} {'Full Rec':>9} {'Reuse Rec':>10} {'Rec gap':>8} "
        f"{'Agree':>7} {'Loaded':>7}"
    )
    print(header)
    print("-" * 118)
    for s in summaries:
        print(
            f"{s.dataset:<12} {s.samples_loaded:>4} {s.full_f1:>8.4f} "
            f"{s.reuse_f1:>9.4f} {s.f1_gap:>+8.4f} "
            f"{s.f1_gap_extracted:>+10.4f} {s.full_recall:>9.4f} "
            f"{s.reuse_recall:>10.4f} {s.recall_gap:>+8.4f} "
            f"{s.agreement_rate:>6.1%} {s.loaded_rate:>6.1%}"
        )
    print("-" * 118)
    total = sum(s.samples_loaded for s in summaries)
    if total:
        full = sum(s.full_f1 * s.samples_loaded for s in summaries) / total
        reuse = sum(s.reuse_f1 * s.samples_loaded for s in summaries) / total
        full_r = sum(s.full_recall * s.samples_loaded for s in summaries) / total
        reuse_r = sum(s.reuse_recall * s.samples_loaded for s in summaries) / total
        ext_gap = sum(s.f1_gap_extracted * s.samples_loaded for s in summaries) / total
        print(
            f"{'OVERALL':<12} {total:>4} {full:>8.4f} {reuse:>9.4f} "
            f"{full - reuse:>+8.4f} {ext_gap:>+10.4f} {full_r:>9.4f} "
            f"{reuse_r:>10.4f} {full_r - reuse_r:>+8.4f}"
        )
    print("=" * 118)
    print(
        "Note: N counts samples with a verified chunk-KV load; accuracy "
        "means exclude silent full-recompute fallbacks.\n'ExtF1 gap' scores "
        "the extracted answer span (bold/first sentence), separating factual "
        "errors from verbosity drift."
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daser-url", default="http://127.0.0.1:2046")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer path for context-length filtering",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--datasets", default="hotpotqa,2wikimqa,musique")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--delete-docs",
        action="store_true",
        help="Delete uploaded documents after each sample (default: keep)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Third Party
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = DaserClient(args.daser_url)
    if not client.health():
        raise SystemExit("DaseR /health is not ok; start the services first")

    summaries: list[DatasetSummary] = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        path = Path(args.data_dir) / f"{dataset}.jsonl"
        records = load_samples(path, 0)
        results: list[SampleResult] = []
        scored = 0
        for record in records:
            if scored >= args.num_samples:
                break
            context_tokens = len(
                tokenizer.encode(
                    str(record.get("context", "")), add_special_tokens=False
                )
            )
            if context_tokens > args.max_context_tokens:
                continue
            t0 = time.time()
            result = run_sample(client, record, args.max_new_tokens, args.delete_docs)
            results.append(result)
            if not result.error:
                scored += 1
            logger.info(
                "[%s %d/%d] docs=%d tokens=%d full_f1=%.3f reuse_f1=%.3f "
                "cov=%.0f%% %.1fs %s",
                dataset,
                scored,
                args.num_samples,
                result.num_docs,
                result.prompt_tokens,
                result.full_f1,
                result.reuse_f1,
                result.reuse_coverage * 100,
                time.time() - t0,
                result.error or "",
            )
        summaries.append(summarize(dataset, results))

    print_report(summaries)
    if args.out:
        payload = {
            "config": {
                "daser_url": args.daser_url,
                "datasets": args.datasets,
                "num_samples": args.num_samples,
                "max_context_tokens": args.max_context_tokens,
                "max_new_tokens": args.max_new_tokens,
                "tokenizer": args.tokenizer,
                "decoding": "greedy",
                "delete_docs": args.delete_docs,
            },
            "summaries": [asdict(s) for s in summaries],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nJSON results written to {out_path}")


if __name__ == "__main__":
    main()
