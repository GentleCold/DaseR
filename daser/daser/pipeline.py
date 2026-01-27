from __future__ import annotations

"""Pipeline 组合器（pipeline）。

`RayDataPipeline` 是一个极简的流水线执行器：
- 维护一串 `DatasetOp`（ds -> ds）的算子；
- `run()` 按顺序执行它们；
- 不引入复杂的 DAG/优化逻辑，保持可读性与可控性。

为什么需要它：
- 让 example/业务脚本更像“声明式组装”，而不是一堆命令式调用混在一起。
- 便于复用：同一套 ops 可以在多个任务之间共享。
"""

from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

from .ops import DatasetOp, RayDatasetLike


D = TypeVar("D", bound=RayDatasetLike)


@dataclass
class RayDataPipeline(Generic[D]):
    """Ray Data 算子流水线。

    Attributes:
        ops: 算子列表；每个算子都接受并返回同一种 Dataset 类型。
    """

    ops: List[DatasetOp[D]] = field(default_factory=list)

    def add(self, op: DatasetOp[D]) -> "RayDataPipeline[D]":
        """追加一个算子并返回 self（方便链式调用）。"""
        self.ops.append(op)
        return self

    def run(self, ds: D) -> D:
        """执行流水线。

        Args:
            ds: 输入 Dataset。

        Returns:
            经过所有算子处理后的 Dataset。
        """
        for op in self.ops:
            ds = op(ds)
        return ds
