"""DaseR：面向 Ray Data + vLLM 的轻量封装。

这里提供一套“可组合”的最小抽象：
- config：集中管理配置对象
- env：环境变量（CUDA）配置
- ray_runtime：Ray init/shutdown
- datasets：数据集读取入口
- ops/pipeline：算子与流水线
- tasks：任务定义与对应 predictor
"""

from .config import CudaConfig, DatasetConfig, InferenceConfig, RayConfig, VLLMConfig
from .env import configure_cuda
from .pipeline import RayDataPipeline

__all__ = [
    "CudaConfig",
    "DatasetConfig",
    "InferenceConfig",
    "RayConfig",
    "VLLMConfig",
    "configure_cuda",
    "RayDataPipeline",
]
