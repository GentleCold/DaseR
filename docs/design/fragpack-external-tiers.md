# FragPack 记录的外部层存储（`--storage-format fragpack`）

[HBM 常驻压缩 KV](hbm-resident-kv.md) 让 vLLM 的 KV cache 本身就是 FragPack 记录。
DaseR 的 L1/L2 不再重新编码：connector 把已满 block 的记录原样拷成一个
block blob 存下，命中时再把 blob 装回目标 block。GPU、L1、L2 三层存的是同一份
无损字节，store 路径不做压缩，load 路径不做解压。

代码：`daser/ops/hbm_kv.cu`（`blob_export_kernel` / `blob_import_kernel`）、
`daser/ops/hbm_kv.py`（`export_blocks` / `import_blocks`、store 注册）、
`daser/ops/fragpack_codec.py`（connector 侧适配）、`daser/config.py`
（`STORAGE_FORMAT_FRAGPACK` 槽位大小）。

## Block blob

一个 block（全部层）一个 blob，4 KiB 对齐：

| 偏移 | 内容 |
|------|------|
| 0 | u32 magic `0x424B5046`（"FPKB"）、u32 version、u32 L、u32 H |
| 16 | i64 blob 字节数（head + 全部记录，不含尾部 pad） |
| 24 | i64 码表 digest（`rank_of` 的 blake2b-64） |
| 64 | i32 `rec_len[L, 2, H]`（可变部分字节，raw 为 `RAW_VAR_LEN`） |
| head | 记录按 (layer, K/V, head) 顺序紧排，每条 `22528 + rec_len` 字节 |

`head = round_up(64 + 8LH, 4096)`。server 的槽位大小取最大 blob（全部记录 raw）：
`blob_slot_bytes(L, H) = head + 2LH × 32768`，比 raw 槽位只多一个 4 KiB head，
因此 connector 永远放得下，不需要 raw 回退记录（tombstone）。实际存储长度
是压缩后的 blob 长度，server 按 packed 格式（与 `compressed-online` 相同的
变长记录、L1 按存储字节计费）管理。

## 数据流

store（`StorePipeline`，store 流）：等待产生这些 block 的 forward 事件 →
`FragPackBlockExporter.pack_into` 每 block 一个 CTA 列（grid `n × L`）把记录从
`rec_addr` 拷到 staging 的 `i × slot_stride` → 读回 blob 长度 → 按
`OnlinePackedSlot` 交给原有的 packed store 路径。DaseR 只存已满 block，已满即
已编码，`rec_addr` 指向 arena page 或 overflow；导出期间 block 在
`finished_sending` 之前不会被 vLLM 回收。

load（`LoadPipeline`，restore 流）：server 把 blob 读进 staging →
`FragPackBlockImporter.prepare` 上传偏移和目标 block → `submit` 在 restore 流上
`import_blocks`。每 (block, layer) 一个 CTA：

1. 所有 CTA 都校验整个 head（magic、版本、L/H、digest、声明长度、每条
   `rec_len` 合法且总和一致），各层结论一致；
2. thread 0 放置本层记录：可编码记录按顺序紧排进该 block-layer 的 page，
   放不下的和 raw 记录从 overflow 分配；
3. 拷贝记录，再更新 `rec_addr` / `rec_len` / `fill`，最后释放该 block 旧的
   overflow 记录。

状态位：1 head 不合法，2 长度不合法，4 overflow 满。restore 完成事件之后
`_finish_batch` 读状态，任一 block 失败就让整批失败，vLLM 按 invalid block
重算。

## 并发与正确性

- overflow 池从栈改为原子位图（`ovf_alloc` 用 `atomicOr` + `__ffs`，
  `ovf_free` 用 `atomicAnd`），import 在 restore 流上分配、model 流上 encode
  同时分配也安全。
- 先分配新记录、后释放旧记录：失败的层保留原表，原表仍拥有它的 overflow
  槽，之后重算时 `prepare` 的 claim 正常归还，不会重复释放。代价是同一 block
  重复导入时 overflow 需要同时容纳两份 raw 记录。
- load 的目标 block 不会是 open / live：DaseR 只加载整 block，计算进度保持
  128 对齐，目标 block 不在本 step 的 `slot_mapping` 中被写，也不在 live
  列表里。目标 block 上一 owner 留下的 staging 槽由下一次 `prepare` 的 GC
  回收。
- 读顺序：restore 流先等 `begin_step` 的 model 流 fence（`prepare` 之后），
  请求在 restore 事件完成后才被调度读这些 block。
- digest 防止用别的标定码表编码的 blob 被装进来（结果会解错而非报错）。

## 约束

- `--storage-format fragpack` 要求 vLLM 用 `--attention-backend CUSTOM`（FragPack
  backend）；connector 对 fragpack 不声明 KV layout，由 backend 选 `LBNHC`。
- 不支持 layer-wise load（blob 按 block 整体装入，connector 初始化时拒绝
  `layerwise_load`）；不支持 `--compressed-codebooks`（码表在 vLLM 侧，由
  `DASER_FRAGPACK_TABLES` 指定）。
- 其余与 packed 格式相同：io_uring、prefix 复用、TP 1、block 128、BF16。
- FragPack backend 创建 store 后经 `hbm_kv.register_store` 发布，connector
  经 `active_store()` 取用数据；容量监控另外读取 `daser.attention.capacity`
  的公开主机端快照 API，不导入 attention backend 或调用 CUDA 执行路径。
