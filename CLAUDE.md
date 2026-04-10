- 代码注释和与用户交流请用中文

- 只关注 `LMCache`、`vllm`、`analyse` 文件夹和 `CLAUDE.md`，不要管其他文件

- 如果要运行 python 代码，请先使用 `source /data/zwt/vllm/bin/activate` 命令激活 python 环境

---

## 项目背景

项目针对的是 **Data + Task 作为 prompt** 的特殊场景：

- **Data**：使用 `/data/zwt/imdb.csv` 文件中的 `review` 列
- **Task**：多种多样，例如：
  ```
  Given the above film review, answer whether the sentiment is "positive" or "negative".
  Respond ONLY with "positive" or "negative", in all lower case.
  ```

工作流分为两个阶段（两阶段相互独立，Warm 阶段不依赖 Cold 阶段所在进程，不受 GPU 显存影响）：

1. **Cold 阶段**：单独推理 Data 部分，将 Data 的 KV Cache 写入 SSD（`/data/zwt/lmcache_kv/`）
2. **Warm 阶段**：prompt = Data + Task，从 SSD 加载 Data 的 KV Cache 到 GPU，加速推理

---

## 模型与硬件

- **模型**：固定使用 `/data/zwt/model/models/Qwen/Qwen3-8B`
- **GPU**：0、2、3、4 卡为 H800-80G，1 卡为 RTX4090-20G；实际测试只需用其中一张即可
- 还需注意设置`CUDA_DEVICE_ORDER=PCI_BUS_ID`，不然可能导致卡定位不准确

---

## LMCache 配置要点

- **不开启** Local CPU（`local_cpu: false`）
- **开启** async_load（`enable_async_loading: true`）
- KV Cache 存储路径：`/data/zwt/lmcache_kv/`
- 使用 `local_disk` 作为 Storage Backend
- 关键配置项（`LMCacheEngineConfig`）位于 `LMCache/lmcache/v1/config.py`

---

## 关键代码路径

### LMCache

| 路径 | 说明 |
|------|------|
| `LMCache/lmcache/v1/config.py` | 引擎配置（`LMCacheEngineConfig`） |
| `LMCache/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` | 文件系统 L2 适配器（SSD 存储） |
| `LMCache/lmcache/v1/manager.py` | 缓存引擎管理器 |
| `LMCache/tests/v1/` | 单元测试目录 |

### vllm KV Connector

| 路径 | 说明 |
|------|------|
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | OffloadingConnector（Cold/Warm 两阶段使用） |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | LMCacheConnector |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/` | vLLM-LMCache 集成适配层 |

### analyse（测试与分析脚本，当前为空）

- 后续在此目录中编写端到端测试、性能分析等代码

---

## 开发规范

### TDD

请遵循 TDD 开发原则：先写测试，再写实现。

### LMCache 代码风格（修改 LMCache 时遵守）

1. **License 头**：所有 Python 文件第一行须为 `# SPDX-License-Identifier: Apache-2.0`

2. **Import 顺序**（用注释分节）：
   ```python
   # Standard
   import os

   # Third Party
   import torch

   # First Party
   from lmcache.v1.config import LMCacheEngineConfig

   # Local
   from .utils import helper
   ```

3. **类型注解**：所有函数/方法参数和返回值均须有类型注解

4. **Docstring**：公共函数/方法须有 docstring，涵盖功能、参数、返回值、异常

5. **封装**：不访问其他类的私有成员（`_` 前缀）

### LMCache 测试运行

```bash
# 运行标准测试套件
pytest -xvs --ignore=tests/disagg \
  --ignore=tests/v1/test_nixl_storage.py \
  --ignore=tests/v1/multiprocess/ \
  --ignore=tests/v1/distributed/ \
  --ignore=tests/skipped \
  --ignore=tests/v1/storage_backend/test_eic.py

# 运行单个测试文件
pytest -xvs tests/v1/test_cache_engine.py
```

---

## 环境配置（参见 README.md）

```bash
# 安装依赖
pip install -U "ray[default]==2.53.0"
pip install lmcache==0.3.12

# vllm 可编辑安装
cd vllm
TMPDIR=/data/zwt/tmp_cache VLLM_PRECOMPILED_WHEEL_LOCATION="/home/zwt/vllm-0.14.0.whl" VLLM_USE_PRECOMPILED=1 pip install --editable .

# flash_attn 报错时重装
pip install flash-attn==2.7.3 --no-build-isolation
```
