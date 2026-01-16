# DaseR 项目上下文

## 项目概览
DaseR 是一个专注于 **Ray** 框架使用和定制的开发工作区，特别是与 **vLLM** 交互以进行大模型推理。该项目演示了如何构建模块化、可复用的 Ray Data 数据处理和推理管道。最近的重构引入了分层架构，将配置、任务定义、算子（Ops）和流水线编排解耦，便于扩展和维护。

## 关键文件与目录

- **`daser/`**: 核心 Python 包，包含模块化的推理框架组件：
    - **`config.py`**: 定义 `CudaConfig`, `RayConfig`, `VLLMConfig` 等配置类，集中管理环境与推理参数。
    - **`pipeline.py`**: 提供 `RayDataPipeline` 类，用于以链式调用的方式组装数据处理和推理步骤。
    - **`ops.py`**: 封装 Ray Data 的操作符，如 `MapRowsOp`, `LimitOp` 和核心的 `ActorPoolInferenceOp`（用于分布式推理）。
    - **`tasks/`**: 包含具体任务的实现。
        - **`sentiment.py`**: 情感分析任务示例，定义了 `SentimentTask`（数据契约与 Prompt 构建）和 `SentimentPredictor`（封装 vLLM 的 Ray Actor）。
    - **`ray_runtime.py`**: 负责 Ray 集群的初始化 (`init_ray`) 和关闭 (`shutdown_ray`)。
    - **`env.py`**: 处理环境变量配置，如 `CUDA_VISIBLE_DEVICES`。
    - **`datasets.py`**: 数据集加载工具，如 `read_csv`。
- **`example.py`**: 使用 `daser` 包构建情感分析流水线的示例脚本。它展示了如何通过配置对象、任务定义和 Pipeline 组装来运行批量推理，而不包含底层实现细节。
- **`ray/`**: 指向 `ray-project/ray` 仓库的 git 子模块。这用于检出 Ray 源代码以进行本地修改。
- **`README.md`**: 包含基本的设置说明，特别是安装特定版本的 Ray (`2.53.0`) 和设置开发环境。

## 设置与使用

### 1. 环境设置
项目需要特定版本的 Ray。

```bash
pip install -U "ray[default]==2.53.0"
```

要使用 Ray 源代码（确保首先初始化子模块）：

```bash
# 如果目录为空，则初始化子模块
git submodule update --init --recursive

# 设置开发环境
cd ray
python python/ray/setup-dev.py -y
```

### 2. 运行示例
`example.py` 脚本执行情感分析。由于使用了新的配置类，参数调整直接在脚本内的配置对象中进行，不再依赖命令行参数（如有需要可自行扩展 argparse）。

```bash
python example.py
```

默认配置：
- 数据路径：`/data/zwt/imdb.csv`
- 模型路径：`/data/zwt/model/models/Qwen/Qwen3-8B`
- 限制条数：1000
- CUDA 设备：1,3,4

## 开发规范

- **模块化设计**: 项目采用分层架构：
    - **Config**: 纯数据类，管理所有可变参数。
    - **Task**: 定义数据契约（输入输出列名）和 Prompt 构建逻辑，不含推理代码。
    - **Predictor**: 封装具体的推理引擎（如 vLLM），作为 Ray Actor 运行。
    - **Pipeline**: 负责组装 Task 和 Predictor，通过 Ops 串联。
- **类型提示**: Python 代码严格使用 `typing` 模块进行函数签名。
- **GPU 管理**: 通过 `daser.env.configure_cuda` 和 `CudaConfig` 显式管理可见 GPU。
- **Ray Data**: 核心数据流基于 Ray Data (`ray.data`)，利用 `map_batches` 配合 `ActorPoolStrategy` 实现高效的 GPU 批处理推理。


## 其他

#### 交互语言                                                
-  **Gemini交互语言**:我是一名来自中国的开发工程师，请在Gemini中使用中文与我交流。  
-  **代码注释语言**: 我是一名来自中国的开发工程师，请务必使用中文进行代码注释。