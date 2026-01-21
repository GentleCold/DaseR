from __future__ import annotations

"""任务抽象基类（tasks.base）。

当前 `daser/tasks/*` 里每个任务模块往往会包含两类东西：

1) Task（任务定义 / 数据契约）
   - 描述输入数据的字段约定（例如文本列名、图像列名）。
   - 描述需要构造的模型输入（例如 prompt 字段）。
   - 描述输出字段命名（例如 prediction/raw_output）。

2) Predictor（执行器 / 推理器）
   - 在 Ray Data 的 map_batches + ActorPoolStrategy 下运行。
   - 负责加载模型并对 batch 执行推理。

为了避免每个具体任务都从零开始定义“如何从 row 取文本/图像、如何拼 prompt”，
这里增加一层更抽象的基类：

- TextTaskBase：处理文本输入的任务（例如情感分类、摘要、抽取）。
- ImageTaskBase：处理图像输入的任务（例如图像分类、VQA、多模态对话）。

注意：
- 这里的抽象只定义“数据契约/字段/row 变换”的共性，不强绑定某个模型或库。
- 为保持项目风格，本模块不做额外防御性检查；调用方保证 row 中字段存在/类型可转。
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TextTaskBase:
    """文本任务基类。

    子类通常只需要：
    - 选择/覆盖列名（text_col/prompt_col）
    - 提供 instruction（或更复杂的 prompt 模板）
    - 定义输出列命名（例如 prediction_col/raw_output_col）

    约定：
    - `build_prompt_row()` 输入为 Ray Dataset 的一行（dict）。
    - 输出为一个新的 dict，用于进入后续算子或 predictor。

    Attributes:
        text_col: 从输入 row 读取文本的列名。
        prompt_col: 输出 prompt 的列名。
        instruction: 拼接在原文本后的指令模板。
    """

    text_col: str = "text"
    prompt_col: str = "prompt"
    instruction: str = ""

    @property
    def text_out_col(self) -> str:
        """输出中承载原始文本的列名。

        默认与 text_col 同名；子类可覆写为不同命名（例如 IMDb review）。
        """

        return self.text_col

    def build_prompt(self, text: str) -> str:
        """从原始文本构造 prompt。

        默认策略：`text + instruction`。
        子类可覆写以实现更复杂的模板（例如 system/user 角色、few-shot 等）。
        """

        return text + self.instruction

    def build_prompt_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """把输入 row 转换为包含 prompt 的新 row。"""

        text = "" if row.get(self.text_col) is None else str(row.get(self.text_col))
        return {
            self.text_out_col: text,
            self.prompt_col: self.build_prompt(text),
        }


@dataclass(frozen=True)
class ImageTaskBase:
    """图像任务基类（占位抽象）。

    多数多模态模型会要求输入包含图像（路径/bytes/array）以及可选的文本 prompt。
    由于不同数据格式差异较大，这里只提供最小字段约定，供后续具体任务扩展。

    Attributes:
        image_col: 输入 row 中图像字段列名（可以是 path/bytes/array，取决于你的数据管线）。
        prompt_col: 输出/传给 predictor 的 prompt 列名。
        instruction: 可选的文本指令（例如 VQA 问题模板）。
    """

    image_col: str = "image"
    prompt_col: str = "prompt"
    instruction: str = ""

    @property
    def image_out_col(self) -> str:
        """输出中承载图像数据的列名。默认与 image_col 同名。"""

        return self.image_col

    def build_prompt_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """把输入 row 转换为 predictor 可消费的 row（图像 + 可选 prompt）。"""

        image = row.get(self.image_col)
        return {
            self.image_out_col: image,
            self.prompt_col: self.instruction,
        }
