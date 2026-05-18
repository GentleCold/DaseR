# DaseR 系统设计文档

DaseR 是面向 LLM 推理的 RAG-native KV cache 服务。它通过 vLLM
`KVConnectorBase_V1` 接入推理流程，在 vLLM worker 进程内执行 KV 数据面
IO，在 DaseR server 进程内维护 HTTP server、IPC server 和控制面状态。

## 目录

| 文档 | 内容 |
|------|------|
| [整体架构](architecture.md) | 进程拓扑、HTTP/IPC server 边界、启动流程和关键设计决策 |
| [组件详解](components.md) | 组件职责、存储布局、可插拔接口和 IPC 协议 |
| [数据流程](flows.md) | 文档上传、推理、KV store 和 KV load 的完整流程 |
