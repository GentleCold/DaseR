# 项目工作流 (Workflow)

本文档定义了 DaseR 项目的标准开发流程。

## 1. 开发流程 (Development Cycle)

Conductor 使用基于 **Track (轨道)** 的开发模式。每个功能、Bug 修复或重构都应作为一个独立的 Track 进行。

### 阶段 1: 规划 (Plan)
1.  **新建 Track**: 使用 `/conductor:newTrack` 启动新任务。
2.  **定义需求 (Spec)**: 在 `spec.md` 中明确“做什么”和“为什么做”。
3.  **制定计划 (Plan)**: 在 `plan.md` 中将任务拆解为可执行的细粒度步骤。

### 阶段 2: 执行 (Implement)
1.  **执行步骤**: 按照 `plan.md` 的顺序逐个执行任务。
2.  **测试驱动 (TDD)**: (推荐) 在编写功能代码前先编写测试。
3.  **原子提交**: 每一小步完成后进行 Git 提交。

### 阶段 3: 验证 (Verify)
1.  **代码审查**: 检查代码是否符合 `code_styleguides/` 中的规范。
2.  **测试通过**: 确保所有新旧测试通过。
3.  **覆盖率**: 目标代码覆盖率 > 80%。

### 阶段 4: 完成 (Finalize)
1.  **更新文档**: 确保相关文档已更新。
2.  **合并**: 将代码合并到主分支（如果使用分支开发）。
3.  **关闭 Track**: 更新 `tracks.md` 状态为 Completed。

## 2. 版本控制规范 (Git Standards)

- **提交信息**: 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
    - `feat: ...` (新功能)
    - `fix: ...` (修复)
    - `docs: ...` (文档)
    - `refactor: ...` (重构)
    - `chore: ...` (杂务)
- **分支策略**:
    - 主分支: `main` 或 `master`
    - 功能分支: `feature/<track-id>` (可选，视团队规模而定)

## 3. 测试规范 (Testing Standards)

- **框架**: `pytest`
- **位置**: 测试文件应位于 `tests/` 目录下，或与源码文件同级（以 `test_` 开头）。
- **运行**: 使用 `pytest` 命令运行测试。
