# DaseR 系统设计文档（中文）

DaseR 是面向 LLM 推理的 RAG-native KV 缓存服务。它以独立进程运行，通过 `KVConnectorBase_V1` 接口与 vLLM 集成，利用 NVIDIA cuFile（GDS）或 io_uring 将 KV 张量直接存储到 NVMe。

## 目录

| 文档 | 内容 |
|------|------|
| [整体方案](architecture.md) | 进程拓扑、关键设计决策、启动与关机 |
| [组件详解](components.md) | 组件职责、存储布局、可插拔接口、IPC 协议 |
| [数据流程](flows.md) | KV Store（写入）与 KV Load（加载）完整流程 |
| [服务层](serve.md) | 用户侧 RAG 服务层设计：文档上传 / 列举 / 推理 |