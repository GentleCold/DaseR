from __future__ import annotations

"""数据集入口（datasets）。

这里放“数据集读取”这类 I/O 相关的最薄封装，避免把 `ray.data.read_*` 直接散落在各处。

说明：
- 当前只封装了 CSV 读取；后续可以扩展 read_parquet/read_json 等。
- 本模块默认 Ray 已经由调用方 init。
"""


def read_csv(path: str):
    """读取 CSV 并返回 Ray Dataset。

    Args:
        path: CSV 文件路径。

    Returns:
        Ray Dataset（具体类型由 Ray 版本决定，这里不强绑定）。

    注意：调用前应先完成 `ray.init()`。
    """

    import ray  # type: ignore

    return ray.data.read_csv(path)  # type: ignore[attr-defined]
