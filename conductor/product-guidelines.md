# Product Guidelines (产品准则)

## 1. Engineering Standards (工程标准)

### Code Style (代码风格)
- **Type Hints**: 所有函数签名（Arguments & Return Types）必须包含严格的 Python 类型提示（Type Hints）。
- **Configuration**: 禁止硬编码。所有配置参数必须通过 `dataclass` 定义的 Config 对象传递（如 `CudaConfig`, `RayConfig`）。
- **Imports**: 保持清晰的导入结构，避免循环依赖。使用 `from __future__ import annotations` 以支持前向引用。

### Documentation (文档与注释)
- **Language**: 所有代码注释、文档字符串（Docstrings）以及与开发者的交互必须使用 **中文**。
- **Docstrings**: 每个模块、类和公共方法必须包含文档字符串，说明其设计意图、参数含义和返回值。

## 2. Architectural Principles (架构原则)

### Modularity (模块化)
- **Decoupling**: 严格遵循分层架构：
    - `Config`: 纯数据，无逻辑。
    - `Task`: 定义数据契约和业务转换（如 Prompt 构建），无副作用。
    - `Ops`: 封装 Ray Data 操作符。
    - `Pipeline`: 负责组装，不包含具体业务逻辑。
    - `Predictor`: 封装有状态的推理引擎（Ray Actor）。
- **Ray Data First**: 优先使用 Ray Data 的原生操作符（如 `map_batches`）而不是手动编写循环或多进程代码。

### Resource Management (资源管理)
- **Explicit GPU Control**: 必须通过 `daser.env.configure_cuda` 和 `CudaConfig` 显式管理可见设备，禁止依赖隐式的环境状态。
- **Statelessness**: 尽量保持算子（Ops）和任务（Task）无状态。必须维护状态时（如加载模型），应封装在 Ray Actor 中并由 `ActorPoolStrategy` 管理。

## 3. Interaction & Usage (交互与使用)
- **Developer First**: 提供清晰的错误提示和类型检查，帮助开发者快速定位配置错误。
- **Simplicity**: 对外暴露的接口（如 `RayDataPipeline`）应尽可能简洁，隐藏底层的分布式复杂性。
