# Product Definition (产品定义)

## Initial Concept (初始概念)
DaseR是为算法工程师提供一个基于Ray，Ray Data和vLLM的高性能的大规模的数据处理框架，支持数据的流水线处理。

## Project Vision (项目愿景)
DaseR 旨在成为算法工程师的首选工具，用于构建和管理大规模数据处理与大模型（LLM）推理任务。通过深度集成 Ray 生态系统（特别是 Ray Data）与 vLLM 高性能推理引擎，DaseR 提供了一套模块化、可扩展且易于维护的流水线框架，解决了大规模数据处理中的性能瓶颈与工程复杂性问题。

## Core Value Proposition (核心价值主张)
1.  **高性能流水线**: 利用 Ray Data 的分布式流式处理能力，实现数据加载、预处理与推理的并行化，最大化 GPU 利用率。
2.  **模块化架构**: 将配置（Config）、任务逻辑（Task）、算子（Ops）与编排（Pipeline）完全解耦，支持灵活复用与快速扩展。
3.  **vLLM 集成**: 内置对 vLLM 的优化支持，提供开箱即用的高吞吐量大模型推理能力。
4.  **开发体验**: 提供类型安全（Type-hinting）、配置即代码（Dataclass Config）的现代化开发体验，降低分布式系统的上手门槛。

## Target Users (目标用户)
- **算法工程师**: 需要处理海量数据（TB级）并进行 LLM 离线推理。
- **ML 平台工程师**: 需要构建稳定、可复用的模型推理基础设施。
- **Ray 开发者**: 希望学习或定制 Ray 分布式应用架构的高级用户。

## Key Features (核心功能)
- **RayDataPipeline**: 链式调用的流水线编排器，支持自定义算子注入。
- **ActorPoolInferenceOp**: 封装了 Actor Pool 管理逻辑的通用推理算子，自动处理批处理与并发。
- **Task Abstraction**: 标准化的任务定义接口，隔离数据契约与业务逻辑（如 Prompt 构建）。
- **Unified Configuration**: 基于 Dataclass 的集中式配置管理，涵盖 CUDA、Ray Runtime 及 vLLM 参数。
