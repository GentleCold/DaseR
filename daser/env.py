from __future__ import annotations

"""环境变量配置（env）。

该模块负责在导入 ray/torch/vllm 之前设置必要的环境变量。

为什么单独抽出来：
- 这类设置对导入顺序敏感，放在脚本里容易被无意改动。
- 多个示例/任务共用同一套环境配置逻辑。
"""

import os

from .config import CudaConfig


def configure_cuda(cfg: CudaConfig) -> None:
    """设置 CUDA 相关环境变量。

    注意：必须在导入 ray/torch/vllm 之前调用。

    Args:
        cfg: CUDA 配置。
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    os.environ["CUDA_DEVICE_ORDER"] = cfg.cuda_device_order
