"""任务集合（tasks）。

每个 task 模块通常包含两部分：
- Task：描述数据契约（输入列/输出列/如何构造 prompt）
- Predictor：Ray Actor，负责实际推理
"""

from .sentiment import SentimentPredictor, SentimentTask

__all__ = ["SentimentPredictor", "SentimentTask"]
