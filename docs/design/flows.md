# 数据流程

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
    participant S as vLLM Scheduler
    participant IPC as IPC server
    participant C as ServerCore
    participant CM as ChunkManager

    S->>S: get_num_new_matched_tokens(request)
    S->>S: full_aligned = floor(len(tokens) / block_tokens) * block_tokens
    S->>S: store_key = xxh3_128(tokens[:full_aligned])
    S->>IPC: match_and_alloc(prefix, "", model_id)
    IPC->>C: match_and_alloc(...)
    C-->>IPC: chunks=[] / alloc=null
    IPC-->>S: miss
    S->>S: track PendingStore(chunk_key, token_count)

    S->>S: update_state_after_alloc(block_ids)
    S->>IPC: alloc_chunk(chunk_key, token_count, model_id)
    IPC->>C: alloc_chunk(...)
    C->>CM: allocate slots, evict old chunks if needed
    CM-->>C: start_slot, num_slots
    C-->>IPC: file_offset, pos_offset
    IPC-->>S: allocation
    S->>S: build_connector_meta(reqs_to_store)
```

`alloc_chunk` 只预留 metadata，chunk 还不会进入 `RetrievalIndex`。

### 阶段二：Worker staging 和批量写入

```mermaid
sequenceDiagram
    participant W as vLLM Worker
    participant BG as daser-io loop
    participant IPC as IPC server
    participant TL as TransferLayer

    W->>W: bind_connector_metadata(reqs_to_store)
    loop each attention layer
        W->>W: save_kv_layer(layer_name, kv_layer)
    end
    W->>W: wait_for_save()
    W->>W: split reqs into bounded staging batches
    loop each staging batch
        W->>W: lease bounded GPU staging view
        W->>W: copy selected block KV into slot-major staging
        W->>W: record producer CUDA event
        W->>BG: run_coroutine_threadsafe(_write_cuda_buffer)
        BG->>BG: wait producer event
        BG->>IPC: transfer_store(cuda_ipc_handle, spans)
        IPC->>TL: store_bytes_grouped(staging slices, file_offset)
        TL-->>IPC: bytes accepted
        IPC-->>BG: stored chunk keys
    end
    BG->>IPC: commit_chunks(...) after all batch futures complete
```

`wait_for_save` 在当前 worker step 内完成 KV -> staging snapshot，然后把
server transfer 交给后台 `daser-io` loop。未完成 batch 的 staging lease 由
future 持有，完成后归还 `CudaStagingPool`；`shutdown` 会阻塞等待所有 pending
store future 完成。

### 阶段三：Commit 发布

```mermaid
sequenceDiagram
    participant IPC as IPC server
    participant C as ServerCore
    participant RI as RetrievalIndex

    IPC->>C: commit_chunk(chunk_key)
    C->>RI: insert(ChunkMeta)
    C-->>IPC: ok
```

commit 完成后 chunk 才能被 lookup 命中。

---

## KV Load 流程

### 阶段一：Scheduler 查找 hit

```mermaid
sequenceDiagram
    participant S as vLLM Scheduler
    participant IPC as IPC server
    participant C as ServerCore
    participant RI as RetrievalIndex

    S->>S: get_num_new_matched_tokens(request)
    S->>IPC: match_and_alloc(prefix, "", model_id)
    IPC->>C: lookup(prefix, model_id)
    C->>RI: lookup(tokens, model_id)
    RI-->>C: matched chunks
    C-->>IPC: chunks
    IPC-->>S: chunks
    S->>S: compute extra_tokens and pending_loads
    S->>S: update_state_after_alloc(block_ids)
    S->>S: map chunk target ranges to vLLM block ids
    S->>S: build_connector_meta(reqs_to_load)
```

`PrefixHashIndex` 通常返回一个最长前缀 chunk；`ChunkReuseIndex` 可以返回多个
block-aligned chunks。Scheduler 会确保返回给 vLLM 的 external tokens 是
连续可用的前缀范围。

### 阶段二：Worker 一次性加载

```mermaid
sequenceDiagram
    participant W as vLLM Worker
    participant BG as daser-io loop
    participant IPC as IPC server
    participant TL as TransferLayer

    W->>W: start_load_kv(forward_context)
    W->>W: split spans into bounded staging batches
    loop each load batch
        W->>W: lease GPU uint8 staging view
        W->>W: export CUDA IPC handle for staging
        W->>BG: transfer_load(cuda_ipc_handle, spans).result(timeout=120s)
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
    Main->>C: chunk_manager.save(daser.index)
    Main->>IPC: close()
    IPC->>IPC: drain and close transfer resources
```

`daser.index` 保存 ring-buffer head/tail、`MetadataStore` 和 `DocRegistry`。
启动时恢复 metadata 后，`ServerCore.rebuild_retrieval_index()` 会从 committed
chunk metadata 重建检索索引。

关机采用 fast consistent stop。收到 SIGTERM/SIGINT 后，server 先停止 HTTP
和 IPC 新请求入口，再保存当时已经 commit 的 chunk metadata 和已经完成注册的
文档。connector 侧尚未完成的后台写入或尚未到达 `commit_chunk` 的数据不会进入
`daser.index`；即使底层 `daser.store` 已有部分字节，重启后也不会被检索索引命中。
已经 commit 但还没有绑定到文档的 orphan chunk 会作为普通 cache chunk 保留，
后续再次上传相同内容时可以通过相同 chunk key 复用。
