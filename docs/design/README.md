# DaseR 系统设计文档

DaseR 是面向 LLM 推理的 RAG-native KV cache 服务。它通过 vLLM
`KVConnectorBase_V1` 接入推理流程。vLLM worker 持有 KV tensor、CUDA staging
和可选 GPU codec；DaseR server 持有 HTTP/IPC 接口、控制面、CUDA IPC mapping
和 `TransferLayer`，执行 L1/L2 字节搬运。

## 目录

| 文档 | 内容 |
|------|------|
| [整体架构](architecture.md) | 进程拓扑、HTTP/IPC server 边界、启动流程和关键设计决策 |
| [组件详解](components.md) | 组件职责、存储布局、可插拔接口和 IPC 协议 |
| [数据流程](flows.md) | 文档上传、推理、KV store 和 KV load 的完整流程 |
| [可观测性指标优化](metrics-optimization.md) | Prometheus 指标、L1/L2 容量、vLLM 集成和 Dashboard 设计 |
| [异步 H2D 源内存生命周期](async-h2d-source-lifetime.md) | server extent pin、transfer staging slice 与 CUDA completion 的释放边界 |
| [抢占与延迟保存](deferred-store-preemption.md) | 取消未提交 store、释放写者和 worker completion 契约 |
| [在线无损 KV codec](word-parallel-online-codec.md) | word-parallel 编解码、layer-major staging 与验证 |
| [在途 load 合并](inflight-load-coalescing.md) | 同前缀在途外部 load 时后到请求等待并复用本地前缀命中（`coalesce_loads`，默认关） |
| [独立 L1/L2 分层与叶子优先淘汰](tier-residency.md) | TLSF 分配、叶子优先 LRU、write-through / exclusive store、promotion / demotion、lease 和持久化 |
