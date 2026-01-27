from __future__ import annotations

"""配置模块（config）。

这里集中定义一组“纯数据”的配置对象，用于描述一次 Ray Data + vLLM 推理任务需要的参数。

设计目标：
- 配置与逻辑解耦：业务逻辑只依赖这些配置对象，而不是散落在脚本里的常量。
- 可组合：不同任务/数据集/模型可以自由组合出不同 pipeline。
- 不做防御性编程：配置正确与否由调用方保证（符合你当前的工程取向）。

说明：
- 这些 dataclass 均为 frozen（不可变），避免运行中被意外修改导致行为漂移。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaConfig:
    """CUDA 环境相关配置。

    Attributes:
        cuda_visible_devices: 对应环境变量 CUDA_VISIBLE_DEVICES，例如 "0,1,2"。
        cuda_device_order: 对应环境变量 CUDA_DEVICE_ORDER，常用 "PCI_BUS_ID"。
    """

    cuda_visible_devices: str
    cuda_device_order: str = "PCI_BUS_ID"


@dataclass(frozen=True)
class RayConfig:
    """Ray Runtime 初始化配置。

    Attributes:
        object_store_memory: Ray 对象存储内存（bytes）。
        include_dashboard: 是否启动 dashboard（通常在离线批处理里关掉）。
        ignore_reinit_error: 重复 init 时是否忽略错误（脚本/Notebook 更方便）。
    """

    object_store_memory: int
    include_dashboard: bool = False
    ignore_reinit_error: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    """数据集读取与预处理相关配置。

    Attributes:
        path: CSV 路径。
        text_col: 文本列名（本示例期望为 IMDb 的 review 列）。
        limit: 截断数据条数，用于快速试跑/调试。
    """

    path: str
    text_col: str = "review"
    limit: int = 1000


@dataclass(frozen=True)
class VLLMConfig:
    """vLLM 推理引擎配置。

    Attributes:
        model: 模型路径或 HuggingFace 模型 ID。
        tensor_parallel_size: 张量并行度（每个实例使用多少张 GPU）。
        gpu_memory_utilization: vLLM 显存占用比例（0.0~1.0）。
        max_tokens: 最大生成 token 数（情感二分类通常很小）。
    """

    model: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.7
    max_tokens: int = 4


@dataclass(frozen=True)
class InferenceConfig:
    """Ray Data 推理分发配置。

    Attributes:
        batch_size: 传入 predictor 的批大小（影响吞吐/显存）。
        concurrency: Actor 池大小（通常约等于模型副本数）。
        num_gpus: 每个 Actor 占用的 GPU 数（本例为 1）。
    """

    batch_size: int = 32
    concurrency: int = 3
    num_gpus: int = 1
