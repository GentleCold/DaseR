from __future__ import annotations

"""Ray 运行时封装（ray_runtime）。

该模块只做两件事：初始化与关闭 Ray。

约定：
- 不在这里做资源探测/回退/告警；调用方应确保配置正确。
- 通过 config 对象传参，避免在业务代码中散落 Ray init 参数。
"""

from .config import RayConfig


def init_ray(cfg: RayConfig) -> None:
    """初始化 Ray。

    Args:
        cfg: Ray 初始化参数。
    """

    import ray  # type: ignore

    ray.init(  # type: ignore[attr-defined]
        ignore_reinit_error=cfg.ignore_reinit_error,
        include_dashboard=cfg.include_dashboard,
        object_store_memory=cfg.object_store_memory,
    )


def shutdown_ray() -> None:
    """关闭 Ray（释放资源）。"""

    import ray  # type: ignore

    ray.shutdown()  # type: ignore[attr-defined]
