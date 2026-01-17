# Python 代码风格指南

## 1. 通用格式
- **标准**: 遵循 [PEP 8](https://peps.python.org/pep-0008/) 进行所有代码格式化。
- **行长**: 限制每行最大 88 个字符 (Black 默认) 或 120 个字符（如果为了可读性必须）。
- **缩进**: 使用 4 个空格作为缩进层级。禁止使用 Tab。
- **导入**:
  - 在各个部分内按字母顺序排序：标准库、第三方库、本地应用 (Isort 标准)。
  - 优先使用绝对导入 (例如 `from mypackage.module import MyClass`) 而非相对导入。

## 2. 类型提示 (严格)
- **强制提示**: 所有函数参数和返回值必须包含类型提示 (Type Hints)。
- **泛型**: 使用 `typing.List`, `typing.Dict`, `typing.Optional` 等，或在 Python 3.9+ 上使用标准集合类型 (`list`, `dict`)。
- **Any**: 尽可能避免使用 `Any`。如果需要动态类型，必须在文档中说明原因。

## 3. 命名约定
- **变量/函数**: `snake_case` (蛇形命名)
- **类/异常**: `PascalCase` (帕斯卡命名/大驼峰)
- **常量**: `UPPER_CASE_WITH_UNDERSCORES` (全大写带下划线)
- **私有成员**: `_leading_underscore` (前导下划线)

## 4. 文档 (Docstrings)
- **格式**: 使用 [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) 文档字符串。
- **内容**:
  - **描述**: 函数/类的功能摘要。
  - **参数 (Args)**: 每个参数的描述。
  - **返回 (Returns)**: 返回值的描述。
  - **抛出 (Raises)**: 抛出的异常列表。

## 5. DaseR 特定规范
- **Dataclasses**: 配置对象必须使用 `@dataclass(frozen=True)`。
- **Ray Actors**: 修改状态的 Actor 方法必须清晰记录文档。
