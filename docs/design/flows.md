# 数据流程

本文档描述当前 io_uring/L1/L2 主路径。vLLM worker 持有 CUDA context、KV cache
和 staging buffer；DaseR server 通过 IPC 打开 staging mapping，并由
`TransferLayer` 执行 transfer orchestration。GDS 仅是可选兼容 backend，不改变
下面的 connector-to-server IPC 边界。

## 文档上传流程

```mermaid
sequenceDiagram
    participant U as User
    participant HTTP as HTTP server
    participant V as vLLM HTTP API
    participant DC as DaserConnector
    participant IPC as IPC server
    participant C as ServerCore

    U->>HTTP: POST /documents(title, text)
    HTTP->>HTTP: tokenize + chunk + hash chunk keys
    loop each cacheable chunk
        HTTP->>V: prefill(chunk tokens)
        V->>DC: scheduler lookup / alloc
        DC->>IPC: match_and_alloc / alloc_chunk
        IPC->>C: allocate ring-buffer slots
        V->>DC: worker save KV
        DC->>IPC: commit_chunk
        IPC->>C: publish chunk to RetrievalIndex
    end
    HTTP->>C: register_document(doc_id, chunk_keys, tokens)
    C-->>HTTP: cached count
    HTTP-->>U: 201 doc_id
```

上传失败时 HTTP server 返回 `502`，不会注册文档。已经成功 commit 的 KV
chunk 可以保留在 ring buffer 中，后续相同内容仍可复用。

---

## 推理流程

```mermaid
sequenceDiagram
    participant U as User
    participant HTTP as HTTP server
    participant C as ServerCore
    participant V as vLLM HTTP API
    participant DC as DaserConnector
    participant IPC as IPC server

    U->>HTTP: POST /infer(doc_ids, task)
    HTTP->>C: get_document(doc_id) for each doc
    HTTP->>HTTP: rebuild prompt tokens
    opt trace_cache=true
        HTTP->>C: lookup(prompt_tokens, model_id)
        C-->>HTTP: cache hit details
    end
    HTTP->>V: completion(prompt tokens, daser_skip_save=true)
    V->>DC: scheduler lookup cached chunks
    DC->>IPC: match_and_alloc / lookup
    IPC->>C: lookup chunks
    DC->>DC: worker load KV from daser.store
    V-->>HTTP: completion
    HTTP-->>U: text + usage + latency
```

`/infer` 会向 vLLM 传 `kv_transfer_params={"daser_skip_save": true}`。文档
chunk 已在上传时缓存，task suffix 通常是一次性的，因此推理请求不再把整个
拼接 prompt 重新写入 DaseR。

---

## KV Store 流程

### 阶段一：Scheduler 查找 miss 并分配 slot

```mermaid
sequenceDiagram
    participant S as Scheduler hook
    participant SL as RequestLifecycle
    participant IPC as IPC server
    participant C as ServerCore
    participant CM as ChunkManager

    S->>SL: get_num_new_matched_tokens(request)
    SL->>SL: full_aligned = floor(len(tokens) / block_tokens) * block_tokens
    SL->>SL: reuse strategy builds store intent
    SL->>IPC: match_and_alloc(prefix, "", model_id)
    IPC->>C: match_and_alloc(...)
    C-->>IPC: chunks=[] / alloc=null
    IPC-->>SL: miss
    SL->>SL: track PendingStore(chunk_key, token_count)

    S->>SL: update_state_after_alloc(block_ids)
    SL->>IPC: alloc_chunk(chunk_key, token_count, model_id)
    IPC->>C: alloc_chunk(...)
    C->>CM: allocate slots, evict old chunks if needed
    CM-->>C: start_slot, num_slots
    C-->>IPC: file_offset, pos_offset
    IPC-->>SL: allocation
    S->>SL: build_connector_meta(scheduler_output)
    SL-->>S: reqs_to_store
```

`alloc_chunk` 只预留 metadata，chunk 还不会进入 `RetrievalIndex`。

### 阶段二：Worker staging 和批量写入

```mermaid
sequenceDiagram
    participant W as Worker hook
    participant WR as WorkerRuntime
    participant BG as StorePipeline / daser-store-io
    participant IPC as IPC server
    participant C as ServerCore
    participant TL as TransferLayer

    W->>WR: bind_connector_metadata(reqs_to_store)
    loop each attention layer
        W->>WR: save_kv_layer(layer_name, kv_layer)
    end
    W->>WR: wait_for_save()
    WR->>BG: defer/submit reqs_to_store
    BG->>BG: split reqs into bounded staging batches
    loop each staging batch
        BG->>BG: lease bounded GPU staging view
        BG->>BG: copy selected block KV into slot-major staging
        BG->>BG: record producer CUDA event
        BG->>BG: wait producer event
        BG->>IPC: transfer_store(cuda_ipc_handle, spans)
        IPC->>TL: store_bytes_grouped(staging slices, file_offset)
        TL-->>IPC: bytes accepted
        IPC-->>BG: stored chunk keys
    end
    IPC->>C: record accepted ranges after transfer completion
    C->>C: commit complete chunk coverage through existing TP quorum
```

`wait_for_save` 只把当前 worker step 的 KV store 意图加入待完成 FIFO，不会
立即读取 live KV cache。`get_finished` 收到 vLLM 的 `finished_req_ids` 后，
按请求进入 FIFO 的顺序提交全部已完成请求。后台 `daser-store-io` loop 使用
staging pool depth 的异步信号量限制实际 snapshot/transfer 并发；前一个 store
释放 staging lease 后，队列中的下一个 store 自动进入，不依赖后续 vLLM
connector step 或 benchmark dummy poll。`shutdown` 会阻塞等待所有 pending
store future 完成。

### 阶段三：Commit 发布

```mermaid
sequenceDiagram
    participant IPC as IPC server
    participant C as ServerCore
    participant RI as RetrievalIndex

    IPC->>C: record_store_ranges(chunk spans)
    C->>C: commit complete coverage through TP quorum
    C->>RI: insert(ChunkMeta)
    C-->>IPC: transfer response
```

commit 完成后 chunk 才能被 lookup 命中。iouring 路径在 bytes 已进入 L1、
L2 后台写入已调度后即可 commit；这里的 commit 表示缓存发布，不表示 SSD
写入已完成或具备断电持久性。

---

## KV Load 流程

### 阶段一：Scheduler 查找 hit

```mermaid
sequenceDiagram
    participant S as Scheduler hook
    participant SL as RequestLifecycle
    participant IPC as IPC server
    participant C as ServerCore
    participant RI as RetrievalIndex

    S->>SL: get_num_new_matched_tokens(request)
    SL->>IPC: match_and_alloc(prefix, "", model_id)
    IPC->>C: lookup(prefix, model_id)
    C->>RI: lookup(tokens, model_id)
    RI-->>C: matched chunks
    C-->>IPC: chunks
    IPC-->>SL: chunks
    SL->>SL: compute extra_tokens and pending_loads
    S->>SL: update_state_after_alloc(block_ids)
    SL->>SL: map chunk target ranges to vLLM block ids
    S->>SL: build_connector_meta(scheduler_output)
    SL-->>S: reqs_to_load
```

`PrefixHashIndex` 返回 rolling-prefix 的连续 slot 命中；
`ChunkReuseIndex` 可以返回多个 block-aligned chunks。Scheduler 会确保返回给
vLLM 的 external tokens 是连续可用的前缀范围。

### 阶段二：Worker 一次性加载

```mermaid
sequenceDiagram
    participant W as Worker hook
    participant WR as WorkerRuntime
    participant BG as LoadPipeline / daser-load-io
    participant IPC as IPC server
    participant TL as TransferLayer

    W->>WR: start_load_kv(forward_context)
    WR->>BG: submit reqs_to_load
    BG->>BG: split spans into bounded staging batches
    loop each load batch
        BG->>BG: lease GPU uint8 staging view
        BG->>BG: export CUDA IPC handle for staging
        BG->>IPC: transfer_load(cuda_ipc_handle, spans)
        IPC->>TL: load_bytes_grouped(staging slices, file_offset)
        TL-->>IPC: bytes read
        IPC-->>BG: ok
        BG-->>W: read complete
        loop each copy run and layer
            W->>W: copy staging bytes back into vLLM KV cache blocks
            opt pos_offset / load scale configured
                W->>W: apply key/value scale and RoPE delta
            end
        end
        W->>W: release staging lease
    end
```

`start_load_kv` 返回前 KV cache 已就绪。`wait_for_layer_load(layer_name)` 是
no-op，因为 FULL CUDA graph replay 不保证逐层 Python hook 执行。

---

## 关机持久化流程

```mermaid
sequenceDiagram
    participant OS as SIGTERM/SIGINT
    participant Main as daser.server
    participant C as ServerCore
    participant IPC as IPC server

    OS->>Main: stop_event
    Main->>Main: stop uvicorn HTTP server
    Main->>IPC: stop_accepting()
    Main->>C: save_snapshot(daser.index)
    C->>C: capture committed keys and save versioned ring state
    Main->>IPC: close()
    IPC->>IPC: drain and close transfer resources
```

raw 存储模式下，`ServerCore.save_snapshot()` 将当前 committed key 集合传给
`ChunkManager.save()`。`daser.index` 同时保存 `snapshot_version=1`、
`committed_keys`、ring-buffer head/tail、完整 `MetadataStore` 和 `DocRegistry`。
快照中的 allocation metadata 可能包含尚未提交的 chunk，恢复资格由
`committed_keys` 明确决定。

启动时先校验快照版本和 committed key 列表，再恢复状态。未提交 allocation
会被移除，其 slot 转为 SKIP，文档中对应的 cached mask 置为 false；保留 chunk
的物理位置不变。随后 `ServerCore.rebuild_retrieval_index()` 只重建保留下来的
committed chunk。旧格式快照缺少发布状态，不能安全推断哪些 bytes 可复用，
因此拒绝加载并冷启动。`--skip-l2` 和 `compressed-online` 仍不保存或恢复索引。

收到 SIGTERM/SIGINT 后，server 停止 HTTP 服务和 IPC 新连接入口，保存快照，
再等待已提交给 transfer 的后台工作并关闭资源。该流程不等待 connector 侧所有
尚未提交的 store。未提交 chunk 即使已有部分 SSD bytes，也不会在恢复后命中。
已经 commit 但尚未绑定文档的 orphan chunk 可以保留并在后续上传时复用。
快照记录的是缓存发布状态；此流程不提供事务式数据与索引提交或断电恢复保证。
