# DaseR 代码可视化

本目录包含反映 DaseR 核心架构和执行流程的 Mermaid 图表。

## 包含的图表

1.  **[类图 (Class Diagram)](./class_diagram.mmd)**
    - 展示了配置类 (`Config`)、任务定义 (`Task`)、算子 (`Ops`) 和流水线 (`Pipeline`) 之间的静态关系。
    - 关键展示了 `RayDataPipeline` 如何组合 `DatasetOp`，以及 `ActorPoolInferenceOp` 如何封装 `SentimentPredictor`。

2.  **[时序图 (Sequence Diagram)](./sequence_diagram.mmd)**
    - 展示了 `example.py` 运行时的完整时序。
    - 涵盖了从 Ray 初始化、Pipeline 组装到分布式推理执行的数据流。

## 如何查看

### 在 VS Code 中查看 (推荐)
安装官方插件 **Markdown Preview Mermaid Support** 或 **Mermaid Preview**，直接打开 `.mmd` 文件即可预览。

### 在线查看
将 `.mmd` 文件的内容复制到 [Mermaid Live Editor](https://mermaid.live/) 中即可渲染。

### 命令行渲染
如果安装了 `mermaid-cli`:
```bash
mmdc -i class_diagram.mmd -o class_diagram.png
mmdc -i sequence_diagram.mmd -o sequence_diagram.png
```
