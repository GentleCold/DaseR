from __future__ import annotations

"""情感分类任务（sentiment）。

该模块提供一个“任务定义 + Predictor 实现”的最小闭环：
- `SentimentTask`：定义输入列、输出列、以及如何从原始样本构造 prompt。
- `SentimentPredictor`：Ray Actor（用于 map_batches），内部持有 vLLM 引擎，对 prompt 批量生成输出。

你可以把它当作后续抽象更多任务（摘要、抽取、打分等）的参考模板：
- 任务对象只负责“如何把 row 变成 prompt / 如何命名输出字段”。
- predictor 只负责“如何做推理”。
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas  # type: ignore

from .base import TextTaskBase


DEFAULT_SENTIMENT_INSTRUCTION = (
    '\nGiven the above film review, answer whether the sentiment is "positive" or "negative". '
    'Respond ONLY with "positive" or "negative", in all lower case.\n'
)


def normalize_sentiment(text: str) -> str:
    """把模型输出归一化为 "positive" 或 "negative"。

    vLLM/LLM 偶尔会输出多余文本（例如解释原因、重复标签等），这里做一个轻量的兜底：
- 同时包含 positive/negative 时，取最先出现的那个。
- 只包含其中一个时直接返回。
- 都不包含时返回原始小写文本，方便定位模型不遵循指令的问题。
    """

    t = (text or "").strip().lower()
    if "positive" in t and "negative" in t:
        return "positive" if t.index("positive") < t.index("negative") else "negative"
    if "positive" in t:
        return "positive"
    if "negative" in t:
        return "negative"
    return t


@dataclass(frozen=True)
class SentimentTask(TextTaskBase):
    """情感分类任务的“数据契约”。

    该对象不依赖 Ray/vLLM，仅描述输入/输出字段与 prompt 构造方式。

    作为文本任务（TextTaskBase）的一个具体实例：
    - 从 text_col 读取原始文本（默认 IMDb 的 "review"）
    - 生成 prompt_col（默认 "prompt"）
    - 预测输出写入 prediction_col/raw_output_col
    """

    # TextTaskBase 字段：text_col / prompt_col / instruction
    text_col: str = "review"
    prompt_col: str = "prompt"
    instruction: str = DEFAULT_SENTIMENT_INSTRUCTION

    # 业务输出字段：为保持兼容，仍然使用 review_out_col 作为原文输出列名。
    review_out_col: str = "review"
    prediction_col: str = "prediction"
    raw_output_col: str = "raw_output"

    @property
    def text_out_col(self) -> str:
        """TextTaskBase 需要的“原文本输出列名”。

        这里映射到 review_out_col，保证现有 example.py 的打印逻辑不需要改。
        """

        return self.review_out_col


class SentimentPredictor:
    """基于 vLLM 的情感分类 Predictor（Ray Actor）。

    使用方式：
- 由 Ray Data 的 `map_batches(..., compute=ActorPoolStrategy(...))` 构造并复用。
- 每个 Actor 内部初始化一次 vLLM LLM 引擎，随后对多个 batch 复用，避免重复加载模型。

    输入/输出：
- 输入 batch 必须包含 prompt_col 指定的列。
- 输出会新增 prediction_col 与 raw_output_col。
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_tokens: int,
        prompt_col: str = "prompt",
        prediction_col: str = "prediction",
        raw_output_col: str = "raw_output",
    ):
        """初始化 vLLM 引擎与推理参数。

        Args:
            model: 模型路径或 HF 模型 ID。
            tensor_parallel_size: 张量并行度。
            gpu_memory_utilization: 显存占用比例。
            max_tokens: 最大生成 token 数。
            prompt_col: 输入 prompt 列名。
            prediction_col: 输出预测列名。
            raw_output_col: 输出原始文本列名。
        """

        from vllm import LLM, SamplingParams  # type: ignore

        import ray  # type: ignore

        gpu_ids = (
            ray.get_runtime_context()  # type: ignore[attr-defined]
            .get_accelerator_ids()
            .get("GPU", [])
        )  # type: ignore[attr-defined]
        print(f"[Predictor] Ray actor sees GPU ids: {gpu_ids}")
        print(f"[Predictor] Loading model: {model}")

        self._prompt_col = prompt_col
        self._prediction_col = prediction_col
        self._raw_output_col = raw_output_col

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

    def __call__(self, batch: pandas.DataFrame) -> pandas.DataFrame:
        """对一个 batch 执行推理。

        Args:
            batch: pandas.DataFrame（由 Ray Data 以 batch_format='pandas' 提供）。

        Returns:
            增加 prediction/raw_output 列的新 DataFrame。
        """

        prompts: List[str] = batch[self._prompt_col].astype(str).tolist()
        outputs = self._llm.generate(prompts, self._sampling_params)
        raw = [o.outputs[0].text if o.outputs else "" for o in outputs]
        pred = [normalize_sentiment(x) for x in raw]

        batch = batch.copy()
        batch[self._prediction_col] = pred
        batch[self._raw_output_col] = raw
        return batch
