import argparse
import os
from typing import Any, Dict, List, Optional
import pandas  # type: ignore

TASK_INSTRUCTION = (
    '\nGiven the above film review, answer whether the sentiment is "positive" or "negative". '
    'Respond ONLY with "positive" or "negative", in all lower case.\n'
)


def _pick_existing_data_path(preferred: str) -> str:
    if os.path.exists(preferred):
        return preferred
    fallback = "/data/zwt/imdb.csv"
    if os.path.exists(fallback):
        print(
            f"[WARN] {preferred} 不存在，自动改用 {fallback}。"
            "（如果你必须使用 /data/imdb.csv，请把文件放到该路径，或用 --data-path 指定。）"
        )
        return fallback
    raise FileNotFoundError(
        f"找不到数据文件：{preferred}（fallback={fallback} 也不存在）"
    )


def _ensure_cuda_visible_devices(requested: Optional[str]) -> None:
    """Must run before importing ray/torch/vllm."""
    if requested:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
        return

    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not current:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4"
        return


def _ensure_cuda_device_order() -> None:
    # Helps avoid vLLM warnings / surprises on heterogeneous GPU nodes.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _count_visible_gpus() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return 0
    return len([x for x in cvd.split(",") if x.strip()])


def _normalize_sentiment(text: str) -> str:
    t = (text or "").strip().lower()
    if "positive" in t and "negative" in t:
        # If model rambled, pick the first occurrence.
        return "positive" if t.index("positive") < t.index("negative") else "negative"
    if "positive" in t:
        return "positive"
    if "negative" in t:
        return "negative"
    # Fall back to raw (still lowercased) so user can see failures.
    return t


class VLLMSentimentPredictor:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_tokens: int,
    ):
        from vllm import LLM, SamplingParams  # type: ignore
        import ray  # type: ignore

        gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])  # type: ignore[attr-defined]
        print(f"[Predictor] Ray actor sees GPU ids: {gpu_ids}")
        print(f"[Predictor] Loading model: {model}")
        self._llm = LLM(
            model=model,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

    def __call__(self, batch):
        prompts: List[str] = batch["prompt"].astype(str).tolist()
        outputs = self._llm.generate(prompts, self._sampling_params)
        raw = [o.outputs[0].text if o.outputs else "" for o in outputs]
        pred = [_normalize_sentiment(x) for x in raw]
        batch = batch.copy()
        batch["prediction"] = pred
        batch["raw_output"] = raw
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray Data + vLLM IMDb sentiment example"
    )
    parser.add_argument(
        "--data-path",
        default="/data/imdb.csv",
        help="CSV path containing IMDb reviews (expects a 'review' column).",
    )
    parser.add_argument(
        "--model",
        default="/data/zwt/model/models/Qwen/Qwen3-8B",
        help="vLLM model path or HF model id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of prompts to run.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Override CUDA_VISIBLE_DEVICES (default: keep env; if unset, use 1,3,4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Ray Data batch size passed into vLLM generate().",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of Ray actors (one model replica per GPU).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4,
        help="Max new tokens for sentiment output (small is enough).",
    )
    args = parser.parse_args()

    # IMPORTANT: set CUDA_VISIBLE_DEVICES BEFORE importing ray.
    _ensure_cuda_visible_devices(args.cuda_visible_devices)
    _ensure_cuda_device_order()

    visible_gpu_count = _count_visible_gpus()
    if visible_gpu_count != 3:
        print(
            f"[WARN] 当前 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}，"
            f"可见 GPU 数量={visible_gpu_count}（期望 3 张卡：1,3,4）。"
        )

    data_path = _pick_existing_data_path(args.data_path)

    import ray

    print(f"[Init] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        object_store_memory=34359738368,
    )
    print(f"[Init] Ray cluster resources: {ray.cluster_resources()}")

    print(f"[Data] Reading CSV: {data_path}")
    ds = ray.data.read_csv(data_path)

    # Pick the review text column.
    cols = ds.schema().names
    if "review" in cols:
        text_col = "review"
    else:
        # fallback: first string-ish column
        text_col = cols[0]
        print(f"[WARN] 找不到 'review' 列，改用第一列: {text_col}")

    print(f"[Data] Building prompts from column: {text_col}")

    def build_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
        review = "" if row.get(text_col) is None else str(row.get(text_col))
        return {
            "review": review,
            "prompt": review + TASK_INSTRUCTION,
        }

    ds = ds.map(build_prompt)
    ds = ds.limit(args.limit)
    print(f"[Data] Limited to first {args.limit} prompts")

    visible = _count_visible_gpus()
    if visible < 1:
        raise RuntimeError("没有可用 GPU（CUDA_VISIBLE_DEVICES 为空）")

    # Use data-parallel replicas (1 GPU per actor). This works for any model and
    # matches the requirement of using GPUs 1,3,4 concurrently.
    concurrency = min(args.concurrency, visible)
    if concurrency < 1:
        concurrency = 1

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.7"))

    print(
        "[Infer] Starting vLLM inference via Ray Data map_batches: "
        f"replicas={concurrency} (1 GPU each), batch_size={args.batch_size}"
    )

    result_ds = ds.map_batches(
        VLLMSentimentPredictor,
        fn_constructor_kwargs={
            "model": args.model,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_mem_util,
            "max_tokens": args.max_tokens,
        },
        batch_format="pandas",
        batch_size=args.batch_size,
        num_gpus=1,
        concurrency=concurrency,
        compute=ray.data.ActorPoolStrategy(min_size=concurrency, max_size=concurrency),
    )

    print("[Infer] Materializing and collecting results...")
    results = result_ds.take_all()

    print(f"\n=== 推理完成，共 {len(results)} 条 ===")
    for i, r in enumerate(results):
        review_head = (r.get("review", "") or "").replace("\n", " ")[:120]
        print(f"{i:04d}\tpred={r.get('prediction')}\treview={review_head}")

    ray.shutdown()


if __name__ == "__main__":
    main()
