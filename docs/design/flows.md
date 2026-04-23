# 数据流程

## KV Store 流程（Cache Miss，新写入）

当 vLLM 调度器发现某个请求在 DaseR 中没有命中缓存，走 **store 路径**，将新计算出的 KV 持久化到 NVMe。

### 阶段一：Scheduler — 查找未命中并分配 slot

```mermaid
sequenceDiagram
    participant S as vLLM Scheduler
    participant IPC as DaseR IPCServer
    participant CM as ChunkManager
    participant MS as MetadataStore

    S->>S: get_num_new_matched_tokens()<br/>aligned = (len(tokens) // block_tokens) * block_tokens<br/>chunk_key = xxh3_128(tokens[:aligned])
    S->>IPC: lookup(tokens, model_id)
    IPC-->>S: chunks=[] (miss)

    S->>S: update_state_after_alloc()<br/>block_ids = vLLM 分配的 blocks
    S->>IPC: alloc_chunk(chunk_key, token_count, model_id)
    IPC->>CM: alloc(chunk_key, num_slots, ...)
    CM->>CM: 1. 检查末尾连续空间是否足够<br/>2. 不足则插 SKIP 块，head 回绕到 0<br/>3. 循环驱逐最老 chunk 直到 free_slots 够<br/>4. start_slot = head
    CM->>MS: insert(ChunkMeta)
    Note over MS: chunk 状态：已分配、未提交<br/>RetrievalIndex 中不可见
    CM->>CM: head += num_slots
    CM-->>IPC: start_slot
    IPC-->>S: {start_slot, num_slots, file_offset, pos_offset}

    S->>S: _pending_stores[req_id] = alloc
    S->>S: build_connector_meta()<br/>→ DaserConnectorMeta(reqs_to_store) → 传入 Worker
```

### 阶段二：Worker — Forward Pass 中逐层异步写入

```mermaid
sequenceDiagram
    participant W as vLLM Worker
    participant BG as daser-io (bg asyncio loop)
    participant KV as kvikio / NVMe

    W->>W: bind_connector_metadata(meta)<br/>_store_futures=[], _pending_commits=set()

    loop 每一层 attention layer
        W->>W: save_kv_layer(layer_name, kv_layer)<br/>layer_idx = _layer_idx_map[layer_name]<br/>file_offset = (start_slot+slot_i)*slot_size + layer_idx*layer_size<br/>data = kv_layer[:, block_id].contiguous()<br/>cp_data = cupy.asarray(data)
        W->>BG: run_coroutine_threadsafe(gds.write_async(cp_data, offset))
        BG->>KV: kvikio.pwrite(cp_data, nbytes, offset)
        Note over BG,KV: 异步执行，不阻塞 Worker
        W->>W: _store_futures.append(future)
        W->>W: 继续执行下一层 attention 计算
    end
```

> 每层的 GDS write 提交到后台 `daser-io` 线程的 asyncio loop，与 forward pass 并发执行，不阻塞推理。

### 阶段三：Worker — 等待写完并两阶段提交

```mermaid
sequenceDiagram
    participant W as vLLM Worker
    participant BG as daser-io (bg asyncio loop)
    participant KV as kvikio / NVMe
    participant IPC as DaseR IPCServer
    participant RI as RetrievalIndex

    W->>W: wait_for_save()
    loop 等待所有 GDS write future
        W->>BG: future.result() — 阻塞等待
        BG->>KV: IOFuture.get()
        KV-->>BG: 写入完成
        BG-->>W: bytes written
    end
    W->>W: _store_futures.clear()

    W->>BG: run_coroutine_threadsafe(_commit_all()).result()
    loop 每个 pending_commit key
        BG->>IPC: ipc_async.commit_chunk(chunk_key)
        IPC->>RI: insert(ChunkMeta)
        Note over RI: chunk 现在对 lookup 可见
        IPC-->>BG: {ok: true}
    end
    W->>W: _pending_commits.clear()
```

> `commit_chunk` 是两阶段提交的第二步——GDS 写完成后 chunk 才通过 `RetrievalIndex.insert()` 对外可见，防止部分写入的数据被读到。

---

## KV Load 流程（Cache Hit，加载缓存）

当 vLLM 调度器发现某个请求在 DaseR 中有命中缓存，走 **load 路径**，将 NVMe 上的 KV 读回 GPU。

### 阶段一：Scheduler — 查找命中

```mermaid
sequenceDiagram
    participant S as vLLM Scheduler
    participant IPC as DaseR IPCServer
    participant RI as RetrievalIndex

    S->>S: get_num_new_matched_tokens()<br/>aligned = (len(tokens) // block_tokens) * block_tokens

    loop 从最长前缀向短试（步长 block_tokens）
        S->>IPC: lookup(tokens[:n], model_id)
        IPC->>RI: lookup(tokens[:n], model_id)
        RI->>RI: key = xxh3_128(tokens[:n])<br/>meta = _index.get(key)
        alt 命中
            RI-->>IPC: [ChunkMeta]
            IPC-->>S: {chunks: [ChunkMeta]}
            Note over S: 返回最长前缀命中，退出循环
        else 未命中
            RI-->>IPC: []
            IPC-->>S: {chunks: []}
            Note over S: n -= block_tokens，继续尝试
        end
    end

    S->>S: extra_tokens = meta.token_count - num_computed_tokens<br/>_pending_loads[req_id] = chunk<br/>return (extra_tokens, is_async=False)

    S->>S: update_state_after_alloc()<br/>block_ids = vLLM 分配的 blocks<br/>chunk["block_ids"] = block_ids[:num_needed]

    S->>S: build_connector_meta()<br/>→ DaserConnectorMeta(reqs_to_load) → 传入 Worker
```

> `is_async=False`：vLLM 在同一调度步内执行 forward pass，KV 必须在 forward 开始前完全就绪。

### 阶段二：Worker — 并发读取所有层（start_load_kv，同步阻塞）

```mermaid
sequenceDiagram
    participant W as vLLM Worker
    participant BG as daser-io (bg asyncio loop)
    participant KV as kvikio / NVMe

    W->>W: start_load_kv(forward_context)

    loop 所有层 × 所有 block（构建任务列表）
        W->>W: file_offset = (start_slot+slot_i)*slot_size + layer_idx*layer_size<br/>buf = torch.empty(nbytes, uint8, device=GPU)<br/>cp_buf = cupy.asarray(buf)  ← 零拷贝共享 GPU 内存<br/>all_reads.append(gds.read_into_async(cp_buf, file_offset))<br/>all_targets.append((buf, kv_tensor, block_id, shape))
    end

    W->>BG: run_coroutine_threadsafe(asyncio.gather(*all_reads)).result(timeout=120s)
    Note over W,BG: 阻塞等待，直到所有读取完成

    par 并发执行所有读取
        BG->>KV: kvikio.pread(cp_buf_0, nbytes, offset_0)
        KV-->>BG: GDS DMA 完成
    and
        BG->>KV: kvikio.pread(cp_buf_1, nbytes, offset_1)
        KV-->>BG: GDS DMA 完成
    and
        BG->>KV: kvikio.pread(cp_buf_N, nbytes, offset_N)
        KV-->>BG: GDS DMA 完成
    end

    BG-->>W: 全部读取完成

    loop 每个 (buf, kv_tensor, block_id, shape)
        W->>W: src = buf.view(kv_tensor.dtype).view(shape)<br/>kv_tensor[:, block_id].copy_(src)  ← GPU→GPU，同 CUDA stream
    end

    Note over W: start_load_kv 同步返回<br/>KV cache 已完全就绪
```

> 所有层、所有 block 的读取通过 `asyncio.gather` 并发提交，充分利用 NVMe 队列深度。`start_load_kv` 全量同步完成后返回，`wait_for_layer_load` 为 no-op，保证 CUDA graph 兼容性。

### 阶段三：Forward Pass

```mermaid
sequenceDiagram
    participant W as vLLM Worker

    Note over W: KV cache 已由 start_load_kv 完全填充
    W->>W: model.forward(...)<br/>attention(q, k, v)<br/>— 命中的 KV 来自 DaseR<br/>— 未命中的 token 走正常 prefill
    W->>W: wait_for_layer_load(layer_name)
    Note over W: no-op，所有加载已在 start_load_kv 完成<br/>FULL CUDA graph 模式下此钩子不被调用
```
