import argparse
import os
from typing import Any, Dict, List, Optional
import pandas  # type: ignore

TASK_INSTRUCTION = (
    '\nGiven the above film review, answer whether the sentiment is "positive" or "negative". '
    'Respond ONLY with "positive" or "negative", in all lower case.\n'
)


def _pick_existing_data_path(preferred: str) -> str:
    """
    检查首选数据路径是否存在。如果不存在，尝试使用备用路径。

    Args:
        preferred (str): 首选的数据文件路径。

    Returns:
        str: 最终使用的数据文件路径。

    Raises:
        FileNotFoundError: 如果首选路径和备用路径都不存在。
    """
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
    """
    确保设置了 CUDA_VISIBLE_DEVICES 环境变量。

    必须在导入 ray/torch/vllm 之前运行此函数。

    Args:
        requested (Optional[str]): 请求使用的 CUDA_VISIBLE_DEVICES 字符串。
                                   如果提供，它将覆盖当前的环境变量。
                                   如果为 None 且环境变量未设置，默认使用 "1,3,4"。
    """
    if requested:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
        return

    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not current:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4"
        return


def _ensure_cuda_device_order() -> None:
    """
    将 CUDA_DEVICE_ORDER 设置为 PCI_BUS_ID 以确保一致的 GPU 索引。

    这有助于避免在异构 GPU 节点上出现 vLLM 警告或意外行为。
    """
    # Helps avoid vLLM warnings / surprises on heterogeneous GPU nodes.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _count_visible_gpus() -> int:
    """
    计算 CUDA_VISIBLE_DEVICES 环境变量中指定的可见 GPU 数量。

    Returns:
        int: 可见 GPU 的数量。如果未设置 CUDA_VISIBLE_DEVICES，则返回 0。
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return 0
    return len([x for x in cvd.split(",") if x.strip()])


def _normalize_sentiment(text: str) -> str:
    """
    将模型输出标准化为 "positive" 或 "negative"。

    Args:
        text (str): 模型生成的原始文本。

    Returns:
        str: 如果找到关键词，返回 "positive" 或 "negative"；否则返回小写的原始文本。
    """
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
    """
    使用 vLLM 进行情感分析预测的类。
    该类旨在作为 Ray Data 的 Actor 使用（通过 map_batches）。
    """
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_tokens: int,
    ):
        """
        初始化 vLLM 引擎。

        Args:
            model (str): 模型路径或 Hugging Face 模型 ID。
            tensor_parallel_size (int): 每个实例使用的 GPU 数量（张量并行度）。
            gpu_memory_utilization (float): GPU 显存使用率（0.0 到 1.0）。
            max_tokens (int): 生成的最大 token 数量。
        """
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
        """
        对一批数据执行推理。

        Args:
            batch (pandas.DataFrame): 包含 'prompt' 列的 Pandas DataFrame。

        Returns:
            pandas.DataFrame: 包含预测结果 ('prediction') 和原始输出 ('raw_output') 的新 DataFrame。
        """
        prompts: List[str] = batch["prompt"].astype(str).tolist()
        outputs = self._llm.generate(prompts, self._sampling_params)
        raw = [o.outputs[0].text if o.outputs else "" for o in outputs]
        pred = [_normalize_sentiment(x) for x in raw]
        batch = batch.copy()
        batch["prediction"] = pred
        batch["raw_output"] = raw
        return batch


def main() -> None:
    \"\"\"
    主函数：配置参数，初始化 Ray，加载数据，并运行分布式推理管道。
    \"\"\"
    # 1. 配置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Ray Data + vLLM IMDb sentiment example"
    )
    # 输入数据集路径
    parser.add_argument(
        "--data-path",
        default="/data/imdb.csv",
        help="CSV path containing IMDb reviews (expects a 'review' column).",
    )
    # vLLM 模型路径或 Hugging Face ID
    parser.add_argument(
        "--model",
        default="/data/zwt/model/models/Qwen/Qwen3-8B",
        help="vLLM model path or HF model id.",
    )
    # 限制处理的数据条数，方便测试
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of prompts to run.",
    )
    # 手动指定可见的 GPU 设备
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Override CUDA_VISIBLE_DEVICES (default: keep env; if unset, use 1,3,4).",
    )
    # Ray Data 传递给 vLLM generate() 的批次大小
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Ray Data batch size passed into vLLM generate().",
    )
    # 并发度：即启动多少个 Ray Actor (模型副本)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of Ray actors (one model replica per GPU).",
    )
    # 模型生成的最大 token 数（情感分析只需要回复 positive/negative，所以设置很小）
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4,
        help="Max new tokens for sentiment output (small is enough).",
    )
    args = parser.parse_args()

    # 2. 环境变量配置：必须在导入 ray/vllm 之前设置 CUDA_VISIBLE_DEVICES
    _ensure_cuda_visible_devices(args.cuda_visible_devices)
    _ensure_cuda_device_order()

    # 检查并打印当前可见 GPU 信息
    visible_gpu_count = _count_visible_gpus()
    if visible_gpu_count != 3:
        print(
            f"[WARN] 当前 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}，"
            f"可见 GPU 数量={visible_gpu_count}（期望 3 张卡：1,3,4）。"
        )

    # 验证数据路径
    data_path = _pick_existing_data_path(args.data_path)

    import ray

    # 3. 初始化 Ray 运行时环境
    print(f"[Init] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        object_store_memory=34359738368, # 分配 32GB 对象存储内存
    )
    print(f"[Init] Ray cluster resources: {ray.cluster_resources()}")

    # 4. 数据加载与预处理
    print(f"[Data] Reading CSV: {data_path}")
    ds = ray.data.read_csv(data_path)

    # 动态识别文本列名
    cols = ds.schema().names
    if "review" in cols:
        text_col = "review"
    else:
        # 如果找不到 "review" 列，默认使用第一列
        text_col = cols[0]
        print(f"[WARN] 找不到 'review' 列，改用第一列: {text_col}")

    print(f"[Data] Building prompts from column: {text_col}")

    # 定义构建 Prompt 的转换函数
    def build_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
        review = "" if row.get(text_col) is None else str(row.get(text_col))
        return {
            "review": review,
            "prompt": review + TASK_INSTRUCTION,
        }

    # 使用 Ray Data 的 map 操作对数据集进行转换
    ds = ds.map(build_prompt)
    # 应用数据条数限制
    ds = ds.limit(args.limit)
    print(f"[Data] Limited to first {args.limit} prompts")

    # 再次确认 GPU 可用性
    visible = _count_visible_gpus()
    if visible < 1:
        raise RuntimeError("没有可用 GPU（CUDA_VISIBLE_DEVICES 为空）")

    # 计算最终推理的并发度（不能超过物理可见的 GPU 数量）
    concurrency = min(args.concurrency, visible)
    if concurrency < 1:
        concurrency = 1

    # 获取 GPU 显存占比设置
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.7"))

    print(
        "[Infer] Starting vLLM inference via Ray Data map_batches: "
        f"replicas={concurrency} (1 GPU each), batch_size={args.batch_size}"
    )

    # 5. 分布式推理：使用 map_batches 将数据分块交给 vLLM Actor 进行处理
    result_ds = ds.map_batches(
        VLLMSentimentPredictor, # 推理类
        fn_constructor_kwargs={ # 传递给类构造函数的参数
            "model": args.model,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_mem_util,
            "max_tokens": args.max_tokens,
        },
        batch_format="pandas",
        batch_size=args.batch_size,
        num_gpus=1, # 每个推理副本占用 1 张显卡
        concurrency=concurrency,
        # 使用 Actor 线程池策略来持久化 vLLM 实例
        compute=ray.data.ActorPoolStrategy(min_size=concurrency, max_size=concurrency),
    )

    # 6. 结果收集：触发实际计算并将结果拉取到本地
    print("[Infer] Materializing and collecting results...")
    results = result_ds.take_all()

    # 打印部分结果以供验证
    print(f"\n=== 推理完成，共 {len(results)} 条 ===")
    for i, r in enumerate(results):
        review_head = (r.get("review", "") or "").replace("\n", " ")[:120]
        print(f"{i:04d}\tpred={r.get('prediction')}\treview={review_head}")

    # 7. 优雅关闭 Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
