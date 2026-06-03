# LongBench 性能对比: vLLM vs LMCache vs DaseR

**日期**: 2026-06-03
**配置**: Qwen3-8B, GPU 2 (H800 81GB), gpu_util=0.85, max_num_seqs=32, max_inflight=32
**数据集**: 2wikimqa, hotpotqa_e, 2wikimqa_e, musique, triviaqa (各20样本, dedup后100样本)
**DaseR**: transfer_mode=iouring, L1=256GiB, L2=300GiB

## TTFT (Time to First Token) — mean (ms)

| Dataset    | vLLM   | LMCache | DaseR-chunk | DaseR-prefix |
|------------|--------|---------|-------------|--------------|
| 2wikimqa   | 21,963 | 15,055  | 83,623      | 80,543       |
| 2wikimqa_e | 22,223 | 15,160  | 83,654      | 81,540       |
| hotpotqa_e | 22,428 | 15,422  | 77,702      | 80,196       |
| musique    | 23,383 | 15,921  | 78,127      | 77,525       |
| triviaqa   | 23,902 | 15,881  | 78,346      | 77,333       |

### vs vLLM 对比

| Dataset    | LMCache   | DaseR-chunk | DaseR-prefix |
|------------|-----------|-------------|--------------|
| 2wikimqa   | -31.5%    | +280.8%     | +266.7%      |
| 2wikimqa_e | -31.8%    | +276.4%     | +266.9%      |
| hotpotqa_e | -31.2%    | +246.5%     | +257.6%      |
| musique    | -31.9%    | +234.1%     | +231.5%      |
| triviaqa   | -33.6%    | +227.8%     | +223.5%      |

### TTFT P50 / P99

| Dataset    | vLLM P50/P99 | LMCache P50/P99 | DaseR-chunk P50/P99 | DaseR-prefix P50/P99 |
|------------|--------------|-----------------|---------------------|----------------------|
| 2wikimqa   | 22,272 / 42,409 | 15,110 / 27,991 | 79,431 / 129,439 | 79,271 / 122,411 |
| 2wikimqa_e | 22,387 / 42,476 | 15,198 / 28,143 | 86,569 / 129,334 | 80,913 / 131,233 |
| hotpotqa_e | 22,609 / 42,739 | 15,393 / 28,283 | 77,627 / 129,206 | 79,038 / 131,110 |
| musique    | 23,806 / 43,541 | 15,970 / 28,763 | 78,104 / 124,204 | 76,065 / 126,212 |
| triviaqa   | 23,988 / 44,239 | 16,239 / 29,261 | 76,764 / 120,633 | 77,018 / 122,643 |

## 准确率 (Accuracy — Contains)

| Dataset    | vLLM  | LMCache | DaseR-chunk | DaseR-prefix |
|------------|-------|---------|-------------|--------------|
| 2wikimqa   | 60%   | 60%     | 65%         | 65%          |
| 2wikimqa_e | 80%   | 75%     | 80%         | 80%          |
| hotpotqa_e | 55%   | 55%     | 65%         | 65%          |
| musique    | 70%   | 70%     | 60%         | 60%          |
| triviaqa   | 100%  | 100%    | 95%         | 95%          |
| **总体**   | **73%** | **72%** | **73%**     | **73%**      |

## 总耗时

| 模式           | 耗时   | 说明                        |
|----------------|--------|-----------------------------|
| vLLM           | 78s    | 纯推理                      |
| LMCache        | 191s   | 冷启动预填 + 热启动推理       |
| DaseR-chunk    | 598s   | 文档上传(预填) + 推理         |
| DaseR-prefix   | 591s   | 文档上传(预填) + 推理         |

## DaseR Profiling

| 指标                   | chunk    | prefix   |
|------------------------|----------|----------|
| TTFT (client avg)      | 80,291 ms | 79,427 ms |
| Server latency (avg)   | 88,815 ms | 87,825 ms |
| TTFT − server latency  | -8,524 ms | -8,397 ms |
| Cache hit chunks       | 200/200   | 200/200   |
| Cache hit rate         | 100.0%    | 100.0%    |

## 结论

- **准确率**: 所有模式一致 (72-73%)，KV Cache 不影响生成质量。
- **LMCache 最优**: TTFT 比 vLLM 降低 ~32%，通过缓存 KV 避免重复 prefill。
- **DaseR chunk vs prefix**: 两种模式性能几乎一致（差异 <2%），TTFT 均为 vLLM 的 3-4x、LMCache 的 5x。说明瓶颈不在 cache reuse 策略，而在底层数据路径。

