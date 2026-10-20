# 独立 L1/L2 分层、叶子优先淘汰与 server 索引

本文描述 server 如何为 chunk 在 L1（pinned host）和 L2（SSD 上的 `daser.store`）
分配位置、选择淘汰对象，以及 lookup 如何决定从哪一层读。raw 与
compressed-online 共用同一套放置和淘汰路径。所有放置、淘汰和 L1 驻留决定都在
`daser/server/`；transfer layer 只在给定位置之间搬运字节。

## 模型

1. chunk（`ChunkMeta`，身份为 `chunk_key` + 不复用的 `start_slot`）在每一层
   最多一个副本：L1 中一段 pinned extent，和/或 L2 中一段文件 extent。
2. chunk 已 commit、且至少一层的副本有效时为**存在**；一层副本由所有 TP rank
   共用，只有所有 rank 的字节都就位后才算有效。prefix lookup 返回最长的存在
   前缀，chunk reuse 还可匹配其他对齐的 chunk。每次规划优先读完整 L1，否则读
   完整 L2；不能用各层不同 rank 的部分副本拼成一个可读副本。
3. 每层有自己的字节分配器（TLSF，粒度 `G`）和自己的叶子优先 LRU 堆，两层共享
   同一棵 chunk 树。chunk 在第 T 层可淘汰，当且仅当它未被固定，并且另一层有
   有效副本，或它是 chunk 树的叶子。
4. 淘汰 chunk 的最后一个副本即把它移出索引；只淘汰一个副本时仅释放该 extent。
5. 默认启用 L2 时，store 对 L2 write-through；L1 能接纳时把 D2H snapshot 留作
   L1 副本。L2 命中按同一接纳规则 promote 进 L1。`--skip-l2` 只写 L1；
   研究开关 `DASER_EXCLUSIVE_TIERS=1` 使新 store 优先只写 L1，读回仍保留 L2
   副本（见「独占分层」）。

驻留按 chunk 粒度：一层要么持有 chunk 的全部 record，要么一个都没有。每层副本
因此是一段连续 extent，不存在部分 record 状态，候选规则是逐 chunk 的谓词。

## 身份与寻址

- `start_slot` 是单调递增的 id（`next_slot` 计数器，每次分配前进 `num_slots`，
  随索引持久化），只是身份，不是物理地址。slot id 永不复用，
  `(chunk_key, start_slot, num_slots)` 不会与之后的分配混淆。没有槽位空闲区间、
  `SlotsExhausted` 和逻辑槽位超额供给；容量只按每层字节计。
- record 是 chunk 的一个 slot 在一个 rank lane 上的字节。raw record 长度为
  `local_slot_size`；压缩 record 长度为 codec 给出的 `stored_length`（已 4 KiB
  对齐）。
- 层内 chunk extent：record 在 4 KiB 偏移上首尾相接
  （`record_offset[i] = sum(sizes[:i])`），extent 长度为
  `round_up(sum(sizes), G)`。只有 extent 起点和长度按 `G` 对齐。
- TP：每层均分为 `tp` 条 lane。chunk 在某层的副本有一个所有 rank 共用的
  lane 内偏移，rank r 的字节位于 `r * lane_bytes + offset`。compressed-online
  仍限 TP=1。
- 所有位置由 server 解析。connector 不计算文件或 L1 偏移。

## 分配器与 store 文件

`TlsfAllocator`（`daser/server/allocator.py`）两层共用：

- 内部以 `G` 为单位，对外 API 以字节计，所有 offset、长度、size class 边界和
  切分点都是 `G` 的倍数，剩余空洞始终可用于 O_DIRECT 与 GDS。`G` 默认 64 KiB，
  启动时校验为 2 的幂且是 4096 的倍数（`--tier-grain`）。
- 两级分离空闲链表（一级为 log2 class，二级 8 档），位图为 Python int；空闲块按
  `start -> length` 与 `end -> start` 索引，相邻合并 O(1)。
- `reserve(length)` 先做 good fit（class 向上取整，位图扫描），未命中再在本
  class 内找精确适配，因此存在可容纳的空洞时不会报告失败。`release`、`take`
  （重建用）、`free_bytes`、`holes`、`largest_free`。
- 每个粒度单位一字节的空闲图（默认粒度下每 64 GiB 容量 1 MiB）：`release` 用它
  拒绝重复释放（可释放已预留 extent 的任意子区间），`take` 用它找覆盖区间的空闲
  块。检查是 C 层字节扫描，开销与 extent / 覆盖空闲块长度成正比，与空洞数无关。

store 文件（`daser/server/store_file.py`）：新建空文件后先设 NOCOW
（`FS_IOC_SETFLAGS` + `FS_NOCOW_FL`，失败只记日志），再 `fallocate` 到 L2 大小；
已有文件校验大小后沿用。btrfs 上 CoW 文件在反复覆盖后碎成上万个 extent，写入
吞吐明显低于 NOCOW 预分配文件。`L2IoEngine` 只打开已有文件，不 truncate。

## 层状态

`TierLayout(grain, l1_lane_bytes, l2_lane_bytes, tensor_parallel_size,
l1_record_bytes)` 是启动时固定的几何（lane 为 0 即该层关闭；`l1_record_bytes`
是 raw L1 记账下每个 record 的最小计费）。`TierIndex`（`daser/server/tiers.py`）
持有 `chunk_key -> ChunkTiers`：

```python
@dataclass(eq=False)
class TierCopy:
    chunk_key: str
    tier: str
    offset: int            # lane 内偏移，G 的倍数
    length: int            # extent 字节数，G 的倍数
    valid_ranks: int       # 字节已就位的 rank（位掩码）
    io_pins: int           # 使用该 extent 的在途读写
    attached: bool         # 被淘汰/移除后为 False，io_pins 归零时释放 extent
    promotion: asyncio.Future | None  # 仅 L1，L2->L1 读进行中

@dataclass
class ChunkTiers:
    record_sizes: tuple[int, ...]
    l1: TierCopy | None
    l2: TierCopy | None
    lease_pins: int        # 固定 L1 副本的请求 lease
```

所有 rank 都有效的副本称为**完整**副本。调用方拿到的句柄持有自己固定的副本：
`StorePlacement`、`Promotion(source, target)`、`Lease`，迟到的报告因此不会碰到
更新的放置。`TierIndex(layout, blocked, on_lost)` 中 `blocked(key)` 是 core 的
未提交写者判断，`on_lost(key)` 在容量淘汰拿走 chunk 最后一个副本时回调。操作
只在 event loop 上调用：

- `place_store([(key, sizes)])`：为每个 chunk 放 L2 extent（启用 L2 时）和 L1
  extent（可接纳时）。整批先尝试一次连续 L2 预留，再切成每个 chunk 各自对齐、
  可独立释放的 extent；放不下时逐 chunk 回退。L2 不足时淘汰 L2 候选直到放下，
  仍失败抛 `TierFull(tier, nbytes, io_blocked)` 并回滚整批。L1 不足时淘汰 L1
  候选，仍失败则该 chunk 跳过 L1（计入 `l1_admission_skips`）；L2 关闭时改为抛
  `TierFull`。放置按 chunk 幂等：后到的 TP rank 与同一 allocation 的写者重试复用
  已有副本（包括 L1 跳过）。返回的每个副本带一个调用方的 io pin。
- `mark_valid(copy, rank)`、`discard_incomplete(placement)`（store 失败后丢弃不
  完整副本）、`pin_io / unpin_io(copy)`、`io_released(tier)`。
- `begin_promotion(key)`：按接纳规则分配 L1，两份副本都 io pin，并挂上
  `promotion` future；`finish_promotion(promotion, ok)`。
- `pin_lease(key) -> Lease | None`、`unpin_lease(lease)`。
- `read_plan(key)`：完整 L1 则 L1；有在途 promotion 则等它；否则完整 L2；都没有
  抛 `ChunkUnavailable`。
- `remove(key)`，持久化用的 `snapshot_l2 / validate_l2_placements /
  restore_l2`，以及 `stats(tier)`。
- 淘汰一个副本时，另一层不完整的副本一并摘除（它永远无法服务读）；两层都没有
  副本的 chunk 移出索引。

## 淘汰树

`EvictionTree`（`daser/server/eviction.py`）每个 chunk 一个节点，记录
`parent_key`（前缀链上一个块的 key，由 connector 的 store intent 携带）、最近访问
`tick` 和孤儿标记。每层一个惰性最小堆，条目为 `(priority, sequence, key)`；
`priority` 综合孤儿、queue hint 和访问 tick，`sequence` 用于稳定排序。

默认模式的第 T 层候选谓词（独占模式另见下文）：

```
T 层副本存在且有效，且 io_pins_T == 0
且 非 (T == L1 且 lease_pins > 0)
且 未被固定、没有未提交写者
且 (另一层副本有效  或  chunk 是 chunk 树的叶子)
```

- 叶子指没有任何子节点，包括未提交的子节点，因此在写的子节点让祖先保持驻留。
- recency 是一份共享的 tick，在 alloc、commit 和 lookup 命中时沿链刷新（祖先比
  后代旧，链尾最新）；两个堆只在成员上不同。
- 堆是惰性的：出堆时校验，在可能使 chunk 成为候选的事件上重新入堆（副本变为
  有效、另一层副本得失、最后一个子节点离开、unpin、io unpin、lease 释放）。
  touch 只更新 tick，已入堆的条目留在原位，出堆时若优先级已变就按当前优先级重新
  入堆，因此命中路径每个节点 O(1)；优先级降低（标为孤儿）时立即重新入堆。每层每个
  节点至多约一个有效条目加一个过期条目，超出时压缩堆。
- 不变量：淘汰不会制造孤儿。非叶子 chunk 只在另一层仍有副本时才失去一层副本。
  孤儿只来自显式淘汰、文档删除或分配时父节点已不在；孤儿不可达，最先淘汰；父 key
  再次存入时子节点自动重新挂上。
- 分配失败且唯一的阻碍是 `io_pins`（时间短）时，调用方等待该层的 `io_released`
  事件后重试；lease 固定和未提交写者不等待。

### 排队提示（queue hint，默认不用）

闭环 max-concurrency 负载下，排队发生在客户端 gateway：P 个会话中 C 个在 vLLM
里，其余 P − C 个请求在 gateway 的 FIFO 里按顺序等放行。会话轮转时 LRU 正好
淘汰最快要用的前缀（循环访问），gateway 的队列却已经给出了接下来的访问顺序。

- gateway 在请求入队时 `PUT /queue_hint/{request_id}`（body 为 prompt token
  的本机字节序 int32，与 IPC 线格式相同），请求结束时 `DELETE /queue_hint/{request_id}`。
  提示必须保留到请求结束：放行即释放时，在跑请求的 chunk 变成未提示的 MRU，排在所有
  排队提示之下，lookup 与 load 之间被淘汰，load 失败（R227 实测 `slot has no
  committed chunk`）。在跑请求最早到达，seq 最小，因此最晚淘汰。
  server 用 `RetrievalIndex.chain_keys` 把 token 换成前缀链 key，调用
  `EvictionTree.want(request_id, keys)`，树按到达顺序给每个提示一个递增 `seq`；
  释放时 `unwant(request_id)`。
- 提示按 key 记录（key → 想要它的 seq 集合），不要求 key 已存在：提示期间才 commit
  的 chunk 加入树时直接继承提示。
- 优先级：孤儿 < 未提示（按 tick）< 被提示。被提示节点的优先级为
  `_HINT_BASE − min_seq`，`_HINT_BASE` 大于任何 tick，所以被提示的块比任何未提示
  块晚淘汰；被提示的块之间，`seq` 最大的（离放行最远）先淘汰。这是按可见
  FIFO 顺序保护近期需求的启发式，并不掌握完整未来访问序列。
  孤儿不可达，提示也救不了，仍最先淘汰。
- `want` 只抬高优先级，沿用惰性堆（出堆时纠正）。`unwant` 先按前缀顺序给不再被
  任何请求提示的 key 一个新 tick（请求刚用过），再立即重新入堆（优先级降低）。
- 叶子优先不变：同一请求的链上祖先与后代 seq 相同，候选谓词仍只允许叶子或有另一层
  副本的节点被淘汰。
- 开销：每个排队或在跑请求 O(prompt 块数) 的 key，入队和结束各一次 O(块数)；不进 vLLM
  调度路径。HTTP 与 IPC 共用 server 事件循环，hash 一个 100k token 的 prompt 约
  毫秒级。
- 指标：`daser_queue_hints_total` / `daser_queue_hint_keys_total`（收到的提示数与
  key 数）、`daser_queue_hint_requests`（gauge，排队提示数）、
  `daser_queue_hint_keys`（gauge，被提示 key 数）、
  `daser_queue_hint_evicted_copies_total{tier}`（淘汰了仍被提示的副本，说明容量
  连队列视野都装不下）。
- 只用于排队在 server 可见的 gateway 前面的部署；不发提示时行为与原 LRU 完全
  相同。

## Store 流程

1. scheduler `alloc_chunks` 分配 id、写入 `ChunkMeta`、加入树节点；分配阶段不检查
   容量。
2. worker `transfer_store` 只发送 staging 侧 span 和 chunk 身份（见下文 IPC
   线格式）；server 调用 `ServerCore.place_store([StoreRequest(chunk_key,
   start_slot, num_slots, record_sizes, record_modes)])`，得到每个 chunk 的
   `StorePlacement`（allocation 已不存在时为 None，写入直接丢弃，回复不列出该
   chunk；只有在途 IO 挡住空间时等待 `io_released` 后重试），每层每请求生成一次
   transfer 调用。raw chunk 的 slot 可以分多次请求到达：同一 rank 的重复 placement
   用 `release_placement` 释放 pin，最后补齐的请求报告结果。packed chunk 必须在
   一次请求中带齐全部 record 及其 `mode`。
3. 有 L1 extent 的 record：D2H 进 L1 后调用 `store_staged(placement, rank, tp)`，
   同一份字节异步写 L2，写完后调用 `store_persisted`，之前 L1 extent 一直 io
   pin。没有 L1 extent 的 record 经 staging pool 写 L2。失败时改调
   `store_failed`：丢弃不完整副本，未 commit 且没有副本的 chunk 移出索引
   （`reason="store_failed"`），重试时重新分配。
4. 所有 rank 的 L1 副本在 `store_staged` 中变为有效后即可发布，不等待 L2。
   没有完整 L1 副本时要等 L2 完整才发布，TP 发布状态由 `ChunkLifecycle` 维护。
   每个 rank 的 store 结束（`store_persisted`，无 L2 时为 `store_staged`）
   另计一次 commit 请求；这个结束计数不是发布的唯一触发点。
   chunk 离开索引后才结束的 store 计入
   `late_evicted_commits`，只释放 pin。
5. 两阶段提交不变：发布前 chunk 不可被 lookup 命中。取消的写者
   （`release_chunk_writer`）同样移除没有完整副本的 chunk。

## Load、promotion 与 readahead

- load span 用 `(start_slot, num_slots)` 指名，`ServerCore.plan_load` 展开为
  `LoadSegment(chunk_key, slot_index, record_sizes, lane_offset, copy)`，按
  `read_plan` 选层并 io pin，读完用 `release_load` 释放。span 可以从 chunk 中间
  开始（scheduler 裁剪）或覆盖多个相邻 chunk；`ChunkManager` 维护每个存活 slot
  的 `slot_id -> (chunk_key, slot_index)` 映射，O(slots) 解析。slot id 不再对应
  已 commit 的 chunk 时抛 `ChunkUnavailable`，不保留任何 pin。
- L2 读且 L1 可接纳时，同一次读把字节放进新 L1 extent 再 H2D，L1 副本随之有效；
  不可接纳时经 staging pool 读。
- 流式 load：`transfer_load` 先把 span 拆成逐 chunk 的段，校验字节数并算好
  buffer offset，此时不 pin。没有在途 promotion 的 chunk
  （`ServerCore.promotion_pending`）立即 pin，作为第一批发出（各层
  `transfer.load` 加本批 promote-on-read）。之后每批只等一个仍在 promote 的
  chunk，并带上届时已就绪的其余 chunk。这样已在 L1 的 chunk 与 readahead 的
  SSD 读重叠，而不是全部等到最后一个 promotion 完成才开始拷贝。每批的 promote-on-read 在本批读完时就
  `finish_promotion`，之后等待它们的 load 不必等整个请求结束。请求返回前等
  所有批次结束，再统一 `release_load`；任一批失败时其余批仍跑完，返回第一个
  错误。`daser_load_phase_seconds_total{phase="resolve"}` 记阻塞在规划（等
  promotion）上的时间，`io` 记其余墙钟时间，即未被等待覆盖的拷贝时间。
- readahead（两层都启用、lookup 带 `external_prefix_queries` 时）：lookup 命中
  后，server 对外部窗口中只有 L2 副本的 chunk 调用 `begin_promotions`，按
  `DASER_LOAD_READAHEAD`（每片字节，默认 1 GiB）切片、最多
  `DASER_LOAD_READAHEAD_WINDOW`（默认 4）片在途，交给 `transfer.promote`，计入
  `daser_readahead_bytes_total{outcome}`。promotion 期间到达的 load 等待其
  future，重复 lookup 不会重复读。promote 进来的 chunk 是普通 LRU 条目（lookup
  刷新为 MRU），不被固定，读之前可能再次被淘汰。
- prefix lookup 把相邻 chunk 合并成一个 run（`chunk_key` 为最后一个 chunk，
  `start_slot` 为第一个）。`compressed_slots`、readahead 和 prefetch lease 用
  `ServerCore.resolve_slots(start_slot, num_slots)` 把 run 拆回各 chunk，不能把
  run 当作单个 chunk。

## 独占分层（`DASER_EXCLUSIVE_TIERS=1`，默认关）

默认 write-through 为每个新 store 分配 L2，两层又共用同一份 recency，实际工作集
往往接近包含式缓存；但 L2 可独立淘汰副本，因此 L1 并非严格的 L2 子集。
R227（P16 C4）的命中接近 64 GiB 包含式模拟，同一访问序列下 96 GiB 并集模拟更高。
独占模式通过 L1 优先写入与异步降级减少新数据的重复驻留，读回仍可同时驻留两层。
两层都启用时才可开启，否则启动报错。

- store：新 chunk 先尝试 L1；L1 接纳的只放 L1（不写 L2，`store_staged` 即结束该
  rank 的 store），不接纳的只放 L2（同默认路径）。
- promotion 成功后保留 L2 源副本：promote 进来的 chunk 是干净的，离开 L1 时直接丢
  L1 副本，不再降级写回；L2 副本由 L2 照常淘汰（淘汰后 chunk 回到脏）。readahead
  把外部窗口提进 L1 后紧接着就被 load 走，若摘源，每次 readahead 都换来一次 L2 读 +
  一次降级写（R233 P16 C4：保留源后 L2 写 −63%、L2 读 io −40%，ΣTTFT 1.06×，命中
  不变）。独占只针对 store：新写入只落一层，读回的块两层各一份。
- 只在 L1 的 chunk 称为**脏**。L1 候选谓词改为「L2 副本完整，或 chunk 是孤儿」：脏的
  非孤儿副本不能同步淘汰，必须先降级；孤儿不可达，直接丢弃。L2 候选谓词不变。
- 降级（demotion）：树多一个 `demote` 堆，候选为 L1 完整、无 io pin、无 L2 副本、
  非孤儿、无未提交写者、且没有脏子节点的 chunk，按同一优先级出堆（未提示的 LRU 先、
  提示过的最后）。没有脏子节点这一条让链从尾部开始降级：链头比链尾旧，按 LRU 会先
  降级，进 L2 后是非叶子，在子节点离开前不是 L2 候选；R228 实测 L2 被这类块填满，
  不被 L1 接纳的 store 报 `TierFull`、请求失败。子节点得到 L2 副本或失去 L1 副本时
  刷新父节点。
  `begin_demotions(max_bytes)` 为出堆的 chunk 预留 L2 extent，只允许淘汰优先级严格
  低于被降级 chunk 的 L2 候选（不为冷块挤掉更热或被提示的 L2 块）。挤不下说明它是两层
  里最冷的块。下面的子树/叶子丢弃不看热度，而 lookup 不 pin 命中的块（load 在之后
  才来；R233 frawxk 因此 load 报 "no committed chunk"），所以 `touch_path`（lookup
  命中）给命中链上的块记 30 s 的 hold：hold 中的块不可被降级阶段的子树/叶子丢弃
  删掉，降级改为
  `stalled`、下批再试；普通 LRU 淘汰不受影响（刚 touch 的块本就最热）。曾试过丢弃前先
  回收 L2 里的重复副本（两层都完整的块），但回收让块变脏、缺口不变，prefetch 下形成
  降级↔回收循环（R233 cmpxpz：回收 23 万次、降级 25 万次 vs 无 prefetch 2.7 万），已删。
  它的子孙只能经它被 lookup 命中，价值不高于它，但同优先级挡在下限外：
  一条 queue hint 用同一个 seq 提示整条前缀，touch 又让链尾的 tick 比链头新。丢它会让
  子孙成为孤儿，留它在 L1 则降级停摆（R228 frawxq：L1 干净字节 0、L1 不再接纳），所以
  先从两层丢它子树里最深的叶子（计 `daser_tier_demotion_subtree_drops_total`），直到
  放得下；叶子自身挤不下直接从 L1 丢弃（可能是最后一份，计 `dropped`）。子树里都被
  pin 着时跳过它、本批之后重新入堆（计 `stalled`），跳过 32 个后结束本批。
  两份副本 io pin，IPC server 调 `transfer.demote`（L1 extent 写到 L2 extent，经 L2 写门），
  `finish_demotion(ok)` 成功时 L2 副本对所有 rank 有效，L1 副本变为干净、可同步淘汰；
  失败时丢弃 L2 副本。
- 干净余量：TierIndex 记 L1 干净字节（两层都完整的 chunk 的 L1 extent）与在途降级
  字节，`demotion_deficit = max(0, 目标 − (L1 空闲 + 干净 + 在途))`。目标默认 L1 lane
  的 1/8，可用 `DASER_EXCLUSIVE_CLEAN`（字节）调整。每次 tier 变化、store 发布、load /
  lease / promotion 放下 pin 后，core 若有缺口就回调 IPC server（脏副本要等 io pin 放下
  且写者提交后才可降级）；server 按 readahead 分片大小（默认 1 GiB，所有 rank 合计）
  成批发起降级，最多 4 片在途，一批结束再检查一次。余量
  不够时 store 跳过 L1 直接写 L2、promotion 被跳过（照常从 L2 经 staging 读），不会
  阻塞。
- 兜底：L1 不接纳、L2 也放不下时，store 从 `demote` 堆丢掉最冷的、无 lease 的脏
  叶子（最后一份，计 `dropped`），再试 L1；没有可丢的才报 `TierFull`。这个 store
  紧急回收路径不检查上述 lookup hold，因此 hold 不是从 lookup 到 load 的驻留保证。
- 写入量：L1 接纳的新 chunk 在异步降级时才写 L2；降级成功使 L1 副本变干净，
  并不立即删除 L1。只要 L2 副本仍完整，再次淘汰 L1 就不必重复写；L2 被独立淘汰后
  仍可能再次降级。
- 持久化：只有 L2 副本进 `daser.index`，独占模式下脏 chunk 在重启后丢失（compressed-online
  本就易失）。
- 指标：`daser_tier_demotions_total{outcome}`、`daser_tier_demoted_bytes_total`、
  gauge `daser_tier_l1_clean_bytes`、`daser_tier_demoting_bytes`。

## Prefetch lease（`--daser-prefetch-max-requests`，默认 0）

- `lookup_prefetch`（必须带 `lease_id`，同 id 的旧 lease 先释放）对外部窗口的
  每个 chunk `pin_lease`，没有 L1 副本的记为待 promote；回复
  `{chunks, spans, tier}`，`spans` 为 `{start_slot, num_slots, nbytes}`（`nbytes`
  为单 rank 字节），`tier = l1 | mixed | l2`；没有 L1 层、窗口为空或某个 chunk
  已离开时不取 lease，`tier` 为 None、`spans` 为空。
- `transfer_prefetch` promote 待读 chunk 并等待完成，回复
  `{ok, requested_bytes, l1_bytes, l2_bytes}`（所有 rank 合计）。lease 不存在或
  某个 chunk 无法进入 L1 时回复 `{"error"}`（计
  `daser_prefetch_operations_total{status="incomplete"}`），scheduler 释放 lease
  后照常 load。`--prefetch-lease-budget` 限制所有 lease 固定的 L1 字节：只有一个
  lease 时允许超出该预算，否则超预算的 lease 等待更早的 lease 释放。
  预算接纳不保证物理 L1 能容纳窗口，promotion 仍可能失败。
- 独占分层下不发 lease：`lookup_prefetch` 像 `lookup` 一样启动 readahead，回复
  `tier = None`，scheduler 不 defer、照常 load。lease 在请求等待期间固定整个外部
  窗口，独占 L1 能同步腾出的只有干净余量（L1 的 1/8），窗口远大于它：R233 cmpxpm
  189 个 lease 0 个完成、L1 接纳跳过 13 万次（无 prefetch 3 千），ΣTTFT 1062 vs 784。
  为 lease 抬高降级目标也不行，降级会越过 L2 下限删热子树（cmpxp：subtree drop
  3515 vs 32，命中 .66 vs .77）。layerwise 依赖 lease，独占下同样退回普通 load。
- leased load 和 layerwise runs 从被固定的 L1 副本读。`transfer_load` 不带
  `lease_id`，由 `plan_load` 自然选中 L1 副本。
- `release_transfer_lease` 解除固定，幂等。

## IPC 线格式

所有偏移由 server 解析；connector 只发 staging 侧偏移和 slot 身份。

| 项 | 字段 |
|---|---|
| `ChunkInfo` / `Allocation` | 不含 `file_offset` |
| `compressed_slots`（仅 compressed-online） | 每个 slot 一项 `{slot_id, mode, stored_length}`；run 内所有 chunk 的 slot 依次列出 |
| store 请求 | `payload`、`spans`、`tp_rank`、`tp_size` |
| store span | `{source_offset, nbytes, chunk_key, start_slot, num_slots, slot_index, packed, mode?}`；raw span 覆盖整数个 record，packed span 是一个 record 且带 `mode` |
| store 回复 | `{ok, bytes, chunk_keys}`；`chunk_keys` 为本请求实际放置的 chunk，过期 allocation 被跳过 |
| load 请求 | `payload`、`spans`、`tp_rank` |
| load span | `{target_offset, start_slot, num_slots, nbytes}`；server 校验 `nbytes` 等于所覆盖 record 之和 |
| load 回复 | `{ok, bytes, l1_segments, l2_segments}`，外加 `data`（`return_data`）或耗时与 `transfer_stats_delta` |
| `transfer_load_runs` | `payload`（已注册 load staging）、`lease_ids`（本 step 每个请求一个）、`tp_rank`、`runs`：`(4, n)` int64 表 `(start_slots, intra_offsets, targets, lengths)` 的字节；`intra_offsets` 相对所指 slot 起点，run 不能越出该 record，slot 必须在某个 lease 内；回复 `{ok, bytes, transfer_ms}` |
| runtime config | 删除 `rank_stride_bytes`、`total_slots`、`l2_budget_bytes`、`total_store_bytes`、`bip_enabled`；新增 `tier_grain`、`l1_enabled`、`l2_enabled`、`l1_lane_bytes`、`l2_lane_bytes`、`staging_bytes` |

store 回复时机：本请求所有 chunk 都有 L1 extent 时在 staging 完成后回复，L2
写在后台进行，失败只记 warning 并丢弃 L2 副本；有 chunk 只写 L2 时等 L2 写完再
回复，失败时回复错误并移除该 chunk。scheduler 规划：trim = `start_slot += skip`
（packed 同时切 `compressed_slots`），合并只看 `start_slot` 连续。

## 容量

- `l2_lane = round_down(l2_size / tp, G)`，文件大小 = `l2_lane * tp`，即
  `--l2-size` 按粒度向下取整。
- `l1_lane = round_down((l1_size - staging) / tp, G)`，pinned 总量
  `l1_lane * tp + staging <= --l1-size`。两层都启用时
  `staging = max(G, round_down(min(256 MiB, l1_size / 8), G))`，承载未接纳 L1 的
  store 和未 promote 的读；只有一层时 staging 为 0。
- 启动报错：iouring 下 `--l1-size <= 0`；启用的层 lane 放不下一个按 `G` 取整的
  raw record；prefetch lease 预算超过 `--l1-size`。
- 两层独立，不再要求 `--l1-size <= --l2-size`。
- runtime config 发布 `l1_enabled`、`l2_enabled`、`tier_grain`、`l1_lane_bytes`、
  `l2_lane_bytes`、`staging_bytes`，`l2_size_bytes` 为 store 文件大小；
  `total_slots`、`rank_stride_bytes`、`l2_budget_bytes`、`total_store_bytes`、
  `bip_enabled` 已删除。codec 几何不依赖槽位数（`CompressedStoreGeometry.num_slots`
  可为 None）。
- `--skip-l2` = 关闭 L2：不建文件，L1 必须接纳，否则 store 失败。
- `gds` = 关闭 L1：没有 pinned pool，lease、readahead、prefetch 请求在启动时或
  回复中拒绝。
- raw L1 记账（消融 arm）把 L1 extent 按 raw 等价长度分配。

## Transfer layer

transfer layer 是数据搬运层，只接受显式位置，只保留瞬态状态（在途 IO、staging、
promotion 读）：

```python
@dataclass(frozen=True)
class Move:
    buffer_offset: int     # staging buffer 一侧
    nbytes: int
    l1_offset: int | None  # pinned arena 绝对偏移
    l2_offset: int | None  # 文件绝对偏移
```

`store`（返回 `StoreHandle`，`staged` 在源字节被消费、L1 副本有效时完成，
`persisted` 在 L2 写完时完成）、`load(buffer, moves, source)`、`promote(moves)`、
`demote(moves)`（arena extent 写到 L2，独占分层用）、`load_runs`、`drain`、`close`。iouring 持有一块 pinned arena（L1 偏移索引其中）和
一个小的 staging pool；staging 的等待只依赖在途 IO，不会死锁。extent 在
`io_pins > 0` 时不会被复用，load 不会读到尚未有效的 extent，因此 transfer 层不
需要写后写排序、promotion 过期检测或按地址的 lease。GDS 只实现 L2 子集：
`store`（`staged == persisted`）、`load(source="l2")`、`drain`、`close`。

## 持久化

只有 raw + L2 持久化；compressed-online 和 `--skip-l2` 的索引易失，启动时冷启动。
`ServerCore.save_index` 写 `daser.index`（格式 2）：有完整 L2 副本的已 commit
chunk 的元数据（含 `parent_key`）、文档表、`next_slot`、每个 chunk 的 L2 偏移和
record 长度，以及粒度、L2 lane 大小和 TP。只在 L1 的 chunk 不保存，恢复后文档把
它们视为已淘汰。`load_index` 先校验格式、复用模式、粒度、lane、TP、slot 区间和
extent，全部通过才修改状态；用 `take` 重建 L2 分配器，L1 为空，淘汰树由索引重建。
任何失败都不改变 core，server 换一个空 core 冷启动。文件大小由
`ensure_store_file` 校验。

## 指标

每层：已用字节、空闲字节、空洞数、最大空闲 extent、淘汰（字节、次数）、为腾出
空间而多淘汰的字节、L1 接纳跳过次数。每次 tier 变化后只刷新 O(1) 的 gauge（容量、
已用、空洞数）；最大空闲 extent 要扫描一个 size class，只由 `tier_stats` 按需返回。

## 已知限制

- L1 由叶子优先 LRU 替换，不再有扫描抵抗的插入策略。
- readahead 的字节在被读之前不固定，store 压力大时可能被淘汰，代价是一次 L2 读。
- 驻留按 chunk 粒度：只读 chunk 一部分的 L2 load 不 promote 该 chunk。
- 每个 chunk 因粒度取整浪费不到 `G`。
- 独占分层下，后代仍在 L1 的 L2-only 非叶子不是 L2 候选；降级可能通过丢弃
  冷子树/叶子腾空间，也可能被 pin 或 lookup hold 阻挡而 stalled。store 的紧急
  回收还可能丢掉唯一的脏副本，因此不能把独占模式视为持久写回存储。
- TP ranks 并发 load 同一 L2-only chunk 时，若某 rank 规划时在等另一个 chunk 的
  promotion，期间另一 rank 已 promote 该 chunk（含全部 lane），前者的
  `begin_promotions` 返回 None，仍从 L2 读自己的 lane：多一次 L2 读，数据正确
  （L2 副本有效且 io pin）。不重新规划，因为 `LoadSegment` 不带绝对 slot。
- 分多次请求到达的 raw chunk，若后续请求永远不来且 worker 没有调用
  `release_chunk_writer`（worker 崩溃、IPC 丢失），server 的分片进度记录一直留
  到该 allocation 的下一次 store 或 `release_chunk_writer`；只占少量内存，计数见
  stalldump `server-ipc` 状态的 `chunk_stores`。正常取消路径都会调用 `release_chunk_writer`。
