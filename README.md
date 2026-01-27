# 环境配置

```
pip install -U "ray[default]==2.53.0"
pip install lmcache==0.3.12
```

运行以下命令可以对ray的python代码进行源码修改：

```
cd ray
python python/ray/setup-dev.py -y
```

运行以下命令可以对vllm的python代码进行源码修改：

如果安装报错空间不足可以修改TMPDIR位置

```
cd vllm
TMPDIR=/data/zwt/tmp_cache VLLM_PRECOMPILED_WHEEL_LOCATION="/home/zwt/vllm-0.14.0.whl" VLLM_USE_PRECOMPILED=1 pip install --editable .
```

如果运行时flash_attn报错，重装flash_attn：

```
pip install flash-attn==2.7.3 --no-build-isolation
```
