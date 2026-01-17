# Technology Stack (技术栈)

## Core Technologies (核心技术)
- **Programming Language**: Python (3.8+)
    - 特性使用: Type Hints, Dataclasses
- **Distributed Framework**: Ray (`2.53.0`)
    - **Ray Data**: 核心数据处理与流式推理引擎。
    - **Ray Core**: 底层 Actor 管理与任务调度。
- **Inference Engine**: vLLM
    - 用途: 高吞吐量大语言模型 (LLM) 推理。

## Infrastructure & Environment (基础设施与环境)
- **Operating System**: Linux
- **Hardware Acceleration**: NVIDIA GPU (CUDA)
- **Environment Management**:
    - `daser.env.configure_cuda`: 显式 CUDA 设备控制。

## Key Dependencies (关键依赖)
- `ray[default]==2.53.0`
- `vLLM`
- `pandas` (通常用于 Ray Data 的底层数据交换，待确认但极有可能)
- `numpy`

## Development Tools (开发工具)
- **Version Control**: Git
- **IDE**: VS Code (推荐配置位于 `.vscode/`)

## Project Structure (项目结构)
- **Framework Package**: `daser/`
- **Configuration**: Dataclasses (`daser/config.py`)
- **Pipeline Orchestration**: Ray Data DAG (`daser/pipeline.py`)
