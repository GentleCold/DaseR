from __future__ import annotations

"""算子（operators / ops）。

本模块提供一组可组合的 Dataset 变换算子（op），用于拼装 Ray Data pipeline。

抽象思路：
- `DatasetOp`：把“对数据集做一次变换”抽象成可调用对象（ds -> ds）。
- `RayDataPipeline`（见 pipeline.py）负责按顺序执行这些算子。
- 具体任务只需要选择/组合算子，而不用把 map/limit/map_batches 逻辑散落在脚本里。

类型说明：
- 我们用 `RayDatasetLike` Protocol 仅用于静态类型提示，表达“这个对象至少有 map/limit/map_batches”。
- 由于 Ray Dataset 的类型在不同版本/安装方式下不稳定，这里不强依赖其具体类型。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol, TypeVar


T = TypeVar("T", bound="RayDatasetLike")


class RayDatasetLike(Protocol):
    """最小 Dataset 接口约束（用于类型提示）。

    只声明本项目会用到的三个方法：
    - map：按行变换（row -> row）
    - limit：截断数据
    - map_batches：按批推理/处理（常用于 Actor 推理）
    """

    def map(self: T, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> T: ...

    def limit(self: T, n: int) -> T: ...

    def map_batches(self: T, *args: Any, **kwargs: Any) -> T: ...


D = TypeVar("D", bound=RayDatasetLike)


class DatasetOp(Protocol[D]):
    """Dataset 算子协议：对 Dataset 做一次变换。"""

    def __call__(self, ds: D) -> D: ...


@dataclass(frozen=True)
class MapRowsOp:
    """行级 map 算子。

    典型用途：从原始行构造 prompt 字段、做字段清洗/重命名等。

    Attributes:
        fn: 形如 `row -> new_row` 的函数。
    """

    fn: Callable[[Dict[str, Any]], Dict[str, Any]]

    def __call__(self, ds: D) -> D:
        return ds.map(self.fn)


@dataclass(frozen=True)
class LimitOp:
    """limit 算子：截断数据集前 N 条。"""

    limit: int

    def __call__(self, ds: D) -> D:
        return ds.limit(self.limit)


@dataclass(frozen=True)
class ActorPoolInferenceOp:
    """ActorPool 推理算子（Ray Data map_batches 封装）。

    该算子会把 Dataset 分成 batch，并用 ActorPoolStrategy 复用 predictor 实例（通常内部持有 vLLM 引擎）。

    约定：
    - predictor_cls 必须是一个可被 Ray 构造的 class，并且可调用（__call__ 接受 pandas.DataFrame）。
    - predictor_kwargs 会传给 predictor_cls 的构造函数。

    Attributes:
        predictor_cls: 推理 Actor 类。
        predictor_kwargs: 构造参数。
        batch_size: 每个 batch 的大小。
        concurrency: ActorPool 大小（副本数）。
        num_gpus: 每个 actor 占用的 GPU 数。
    """

    predictor_cls: type
    predictor_kwargs: Dict[str, Any]
    batch_size: int
    concurrency: int
    num_gpus: int = 1

    def __call__(self, ds: D) -> D:
        import ray  # type: ignore

        return ds.map_batches(
            self.predictor_cls,
            fn_constructor_kwargs=self.predictor_kwargs,
            batch_format="pandas",
            batch_size=self.batch_size,
            num_gpus=self.num_gpus,
            concurrency=self.concurrency,
            compute=ray.data.ActorPoolStrategy(  # type: ignore[attr-defined]
                min_size=self.concurrency,
                max_size=self.concurrency,
            ),
        )
