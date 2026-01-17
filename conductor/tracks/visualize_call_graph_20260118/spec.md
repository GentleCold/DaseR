# Specification: 基于 Mermaid 的代码函数调用可视化

## 1. 目标 (Goal)
创建一个 `mermaid/` 目录，并在其中生成反映 `daser` 核心组件（Config, Task, Ops, Pipeline）之间函数调用关系和类依赖关系的 Mermaid 图表文件。

## 2. 范围 (Scope)
- **分析对象**:
    - `daser/pipeline.py` (RayDataPipeline)
    - `daser/ops.py` (ActorPoolInferenceOp 等)
    - `daser/tasks/sentiment.py` (SentimentTask, SentimentPredictor)
    - `daser/config.py` (配置类依赖)
- **输出产物**:
    - `mermaid/class_diagram.mmd`: 核心类及其关系的类图。
    - `mermaid/sequence_diagram.mmd`: `example.py` 中典型的流水线执行时序图（从初始化到推理结果）。
    - `mermaid/README.md`: 说明如何渲染和查看这些图表。

## 3. 需求 (Requirements)
- **准确性**: 图表必须准确反映代码中的继承、组合和调用关系。
- **可读性**: 避免生成过于复杂的全量调用图，专注于核心业务流程。
- **工具化**: 如果可能，提供一个脚本或指令来自动更新这些图（可选，本次主要关注手动/半自动生成）。

## 4. 成功标准 (Success Criteria)
- `mermaid/` 目录存在。
- 包含至少两个有效的 Mermaid 文件（类图和时序图）。
- 图表内容与当前代码逻辑一致。
