from __future__ import annotations

"""IMDb 情感分析示例（基于 daser 抽象）。

这个文件刻意保持“只做组装，不做实现细节”：
- 用 config 对象描述环境 / Ray / 数据集 / vLLM / 推理分发参数
- 用 task + ops + pipeline 拼装出一条 Ray Data 流水线
- 运行并打印部分结果

如果你要迁移到别的任务：通常只需要替换 task 与 predictor，以及少量配置。
"""

from daser.config import CudaConfig, DatasetConfig, InferenceConfig, RayConfig, VLLMConfig
from daser.datasets import read_csv
from daser.env import configure_cuda
from daser.ops import ActorPoolInferenceOp, LimitOp, MapRowsOp
from daser.pipeline import RayDataPipeline
from daser.ray_runtime import init_ray, shutdown_ray
from daser.tasks.sentiment import SentimentPredictor, SentimentTask


# ===== 示例配置（无 argparse；不做 fallback 防御） =====
# CUDA/Ray/Dataset/VLLM/Inference 分开配置，便于复用与替换。
CUDA = CudaConfig(cuda_visible_devices="1,3,4")
RAY = RayConfig(object_store_memory=32 * 1024 * 1024 * 1024)
DATASET = DatasetConfig(path="/data/zwt/imdb.csv", text_col="review", limit=1000)
VLLM = VLLMConfig(
    model="/data/zwt/model/models/Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_tokens=4,
)
INFER = InferenceConfig(batch_size=32, concurrency=3, num_gpus=1)


def main() -> None:
    """示例入口：初始化环境与 Ray，组装 pipeline 并执行。"""

    # 1) 先设置 CUDA 环境变量（需在 import vLLM/Ray 相关前完成）。
    configure_cuda(CUDA)

    # 2) 初始化 Ray。
    init_ray(RAY)

    # 3) 读取数据并定义任务契约。
    ds = read_csv(DATASET.path)
    task = SentimentTask(text_col=DATASET.text_col)

    # 4) 组装流水线：构造 prompt -> limit -> ActorPool 推理。
    pipeline = (
        RayDataPipeline()
        .add(MapRowsOp(task.build_prompt_row))
        .add(LimitOp(DATASET.limit))
        .add(
            ActorPoolInferenceOp(
                predictor_cls=SentimentPredictor,
                predictor_kwargs={
                    "model": VLLM.model,
                    "tensor_parallel_size": VLLM.tensor_parallel_size,
                    "gpu_memory_utilization": VLLM.gpu_memory_utilization,
                    "max_tokens": VLLM.max_tokens,
                    "prompt_col": task.prompt_col,
                    "prediction_col": task.prediction_col,
                    "raw_output_col": task.raw_output_col,
                },
                batch_size=INFER.batch_size,
                concurrency=INFER.concurrency,
                num_gpus=INFER.num_gpus,
            )
        )
    )

    result_ds = pipeline.run(ds)
    results = result_ds.take_all()

    print(f"\n=== 推理完成，共 {len(results)} 条 ===")
    for i, r in enumerate(results):
        review_head = (r.get(task.review_out_col, "") or "").replace("\n", " ")[:120]
        print(f"{i:04d}\tpred={r.get(task.prediction_col)}\treview={review_head}")

    shutdown_ray()


if __name__ == "__main__":
    main()
