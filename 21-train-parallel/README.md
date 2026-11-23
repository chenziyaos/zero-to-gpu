# 21-train-parallel

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-gpu/blob/main/colab/21-train-parallel.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-21](../book/21-train-parallel.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-21 训练并行：TP、PP、DP、ZeRO 与 FP8 的显存账、通信账和气泡。本章回答：给定集群和模型，如何选择并行方案；不同并行的显存和通信特性如何。

| 文件 | 环境 | 做什么 | 对应文档章节 |
|---|---|---|---|
| `sim_memory.py` | CPU | 根据模型配置和并行度，计算参数、梯度、优化器、激活的显存分布 | 第 2 章 |
| `sim_parallel.py` | CPU | 给定集群和模型，枚举 TP/PP/DP/ZeRO 组合，估算每步计算、通信、气泡时间与 MFU | 第 3、4 章 |
| `test_parallel.py` | CPU | 验证 ZeRO 论文的显存数字、1F1B 气泡公式、TP 通信和 MFU 基线 | 全章 |

## 运行

```bash
cd zero-to-gpu/21-train-parallel

# 测试
python3 test_parallel.py

# 模型显存（8B 模型、H100、序列 8192）
python3 sim_memory.py model --params 8 --hidden 4096 --layers 32 --seq 8192 --gpu h100-sxm

# 并行方案比较（DP vs ZeRO-3）
python3 sim_memory.py parallel --params 8 --ngpus 64 --strategy dp --gpu h100-sxm
python3 sim_memory.py parallel --params 8 --ngpus 64 --strategy zero3 --gpu h100-sxm

# 单个并行方案分析（TP=8, PP=2, DP=4）
python3 sim_parallel.py scheme --params 8 --seq 8192 --layers 32 --ngpus 64 --tp 8 --pp 2 --gpu h100-sxm

# 搜索最优并行方案（64 卡，遍历 TP 度数 1-8、PP 度数 1-4）
python3 sim_parallel.py search --params 8 --seq 8192 --layers 32 --ngpus 64 --tp-max 8 --pp-max 4 --gpu h100-sxm
```

## 期望输出

### test_parallel.py

```
6/6 通过
✓ ZeRO-3 显存
✓ 1F1B 气泡
✓ TP all-reduce
✓ DP all-reduce
✓ MFU 基线
✓ 模型状态显存
```

## 版本

CUDA 13.4 Update 2，PyTorch 2.15.0。
