# Implementation Plan: 基于 Mermaid 的代码函数调用可视化

## Phase 1: 环境准备与分析
- [x] Task: 创建 `mermaid/` 目录。
- [x] Task: 深入分析 `daser/pipeline.py` 和 `example.py` 的执行流程。
- [x] Task: 深入分析 `daser/ops.py` 和 `daser/tasks/` 的类结构。

## Phase 2: 生成类图 (Class Diagram)
- [x] Task: 编写 `mermaid/class_diagram.mmd`。
    - [x] Subtask: 定义 Config 类 (`CudaConfig`, `RayConfig` 等)。
    - [x] Subtask: 定义核心接口 (`RayDataPipeline`).
    - [x] Subtask: 定义算子类 (`MapRowsOp`, `ActorPoolInferenceOp`).
    - [x] Subtask: 定义任务与预测器 (`SentimentTask`, `SentimentPredictor`).
    - [x] Subtask: 描绘它们之间的组合与继承关系。

## Phase 3: 生成时序图 (Sequence Diagram)
- [ ] Task: 编写 `mermaid/sequence_diagram.mmd`。
    - [ ] Subtask: 描绘 `example.py` 的 `main` 函数启动流程。
    - [ ] Subtask: 描绘 `RayDataPipeline.run` 内部的数据流转。
    - [ ] Subtask: 描绘 `ActorPoolInferenceOp` 如何分发任务给 `SentimentPredictor`。

## Phase 4: 文档与验证
- [ ] Task: 创建 `mermaid/README.md`，说明图表含义及渲染方法。
- [ ] Task: 验证生成的 Mermaid 代码是否语法正确（使用在线编辑器或 VS Code 插件预览）。
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
