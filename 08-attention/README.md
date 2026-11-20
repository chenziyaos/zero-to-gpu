# 08-attention

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/08-attention.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

GPU-08 注意力内核的配套练习：分块融合与在线 softmax 的 HBM 流量账本。

## 说明

注意力是大模型最昂贵的算子。标准三段式（Q @ K.T → softmax → softmax @ V）要读写 N² 的中间矩阵，长序列下几乎全时间都花在搬 HBM；FlashAttention 通过分块融合与在线 softmax 把中间矩阵限制在片上 SRAM，削减 HBM 流量 60% 以上，关键瓶颈转到指数单元与共享内存。

本目录代码以 CPU 模拟器形式提供：用 Python 纯列表实现朴素注意力与在线 softmax，与标准注意力对比流量和时间，复现 FlashAttention 论文的 IO 分析与下界。

## 文件与环境

| 文件 | 环境 | 做什么 | 对应文档章节 |
|-----|-----|--------|-----------|
| `sim_attn.py` | Python 3.10+ | 注意力 HBM 流量账本：标准 vs FA1 分块 | GPU-08 第 1、2、3 章 |
| `sim_online_softmax.py` | Python 3.10+ | 在线 softmax 精度与块分割、IO 下界分析 | GPU-08 第 2 章 |
| `test_attn.py` | Python 3.10+ | 单元测试（6 个用例，验证 FA1 论文数字） | 全篇 |

无额外依赖，仅使用标准库与 `common/specs.py`。

## 运行命令与期望输出

### 1. 标准注意力流量（H100，N=8192, d=128, 32 头）

```bash
python3 sim_attn.py naive --seq-len 8192 --head-dim 128 --num-heads 32 --gpu h100-sxm
```

期望：17.38 GB HBM，5.19 ms（H100 带宽受限）

```
标准三段式注意力
  序列长度 N = 8192, 头维度 d = 128, 头数 = 32
  HBM 读 13.02 GB, 写 4.36 GB, 总 17.38 GB
  带宽受限（3.35 TB/s）：5188.4 μs（5.19 ms）
  瓶颈：带宽
```

### 2. FA1 分块融合（block_size=256）

```bash
python3 sim_attn.py blocked --seq-len 8192 --head-dim 128 --num-heads 32 --block-size 256 --gpu h100-sxm
```

期望：4.50 GB HBM（削减 74%），1.34 ms（3.87x 加速）

```
分块融合注意力（FA1）
  序列长度 N = 8192, 头维度 d = 128, 头数 = 32
  HBM 读 4.43 GB, 写 0.07 GB, 总 4.50 GB
  带宽受限（3.35 TB/s）：1342.2 μs（1.34 ms）
  瓶颈：带宽
```

### 3. 对比视图

```bash
python3 sim_attn.py savings --seq-len 8192 --head-dim 128 --num-heads 32 --block-size 256 --gpu h100-sxm
```

期望：标准 5.19 ms vs FA1 1.34 ms

```
对比：标准 vs FA1 分块
  标准三段式：17.38 GB HBM
  FA1 分块：  4.50 GB HBM
  削减：      12.88 GB (74.1%)
  @ 3.35 TB/s HBM 带宽：
    标准：5188.4 μs (5.19 ms)
    FA1：1342.2 μs (1.34 ms)
    加速：3.87x
```

### 4. 在线 softmax 成本分析

```bash
python3 sim_online_softmax.py scaling --seq-len 8192 --head-dim 128 --block-size 256
```

期望：32.3 μs exp 成本 < 1% 总时间

```
在线 softmax 重缩放分析
  块数：32
  每 token 重缩放次数：31
  exp 计算成本（H100 MUFU, 16 exp/cycle）：32.3 μs
  占总注意力时间比例（估计 ~5ms）：0.6%
```

### 5. 块大小推荐（SRAM 约束）

```bash
python3 sim_online_softmax.py blocks --seq-len 8192 --head-dim 128 --sram-kb 96 --gpu h100-sxm
```

期望：H100 96 KB SRAM 限制块大小最多 82

```
推荐块大小
  最大块大小（SRAM 限制）：82
  block_size=64: 128 块，重缩放 127x/token，exp ~130.6 μs
```

### 6. IO 下界（Dao et al. 2022）

```bash
python3 sim_online_softmax.py io-lower --seq-len 8192 --head-dim 128 --sram-kb 96 --gpu h100-sxm
```

期望：下界 0.00 GB（表示 FA1 已接近理论最优）

```
FlashAttention IO 下界（Dao et al. 2022）
  标准三段式：0.41 GB
  IO 下界：  0.00 GB
  最多能改进：194x
```

### 7. 测试

```bash
python3 test_attn.py
```

期望：6 个测试全部 pass

```
✓ Test 1: naive_attn 基本计算 OK
✓ Test 2: 流量削减 3.9x OK
✓ Test 3: H100 IO 下界 17.4 GB -> 0.0672 GB OK
✓ Test 4: 在线 softmax 块统计 OK
✓ Test 5: exp 成本 32.3 μs OK
✓ Test 6: 时间单调性 OK

6 tests passed ✓
```
