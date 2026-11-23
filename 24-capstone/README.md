# 24-capstone

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-gpu/blob/main/colab/24-capstone.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-24](../book/24-capstone.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-24 综合：从选卡到上线的训推方案计算器。

## 文件说明

| 文件 | 环境 | 功能 | 对应章节 |
|---|---|---|---|
| `sim_plan_train.py` | CPU | 训练计划：给定模型参数、总 token 数、GPU 型号与卡数，估算工期、卡时与每百万 token 成本 | 第 2 章 |
| `sim_plan_infer.py` | CPU | 推理计划：给定 QPS、序列长度、SLO，估算需要的卡数与成本 | 第 3 章 |
| `sim_calibrate.py` | CPU | 校准工具：对比估算值与公开数据（Llama 3、DeepSeek-V3 等）的差异 | 第 4 章 |
| `test_capstone.py` | CPU | 单元测试：验证公式与校准 | 全部 |

## 运行命令

### 训练计划
```bash
# 基础用法：70B 模型，1T token，512 张 H100，MFU 50%
python3 sim_plan_train.py plan --model 70b --tokens 1t --gpu h100-sxm --num-gpu 512

# 显示时间账单（计算 vs 检查点 vs 故障）
python3 sim_plan_train.py deficit --model 70b --tokens 1t --gpu h100-sxm --num-gpu 512

# 自定义参数
python3 sim_plan_train.py plan --model 405b --tokens 15t --gpu h100-sxm --num-gpu 16128 \
  --mfu 0.35 --checkpoint-hours 12 --fault-factor 1.08 --cost-per-hour 12
```

### 推理计划
```bash
# 基础用法：70B 模型，100 QPS，4096 token 平均长度
python3 sim_plan_infer.py plan --model 70b --qps 100 --length 4096 --gpu h100-sxm

# 校准实际观测值
python3 sim_plan_infer.py calibrate --model 70b --qps 100 --length 4096 --gpu h100-sxm \
  --actual-cards 8 --actual-ttft-ms 480 --actual-tpot-ms 45
```

### 校准与验证
```bash
# 列出所有公开训练数据
python3 sim_calibrate.py list

# 验证 Llama 3 405B 估算
python3 sim_calibrate.py verify llama3-405b --mfu 0.35 --checkpoint-oh 0.15 --fault-factor 1.05
```

### 测试
```bash
python3 test_capstone.py
```

## 期望输出

### sim_plan_train.py 输出示例
```
Training Plan
================================================================================
Model                              70b (70.0B params)
Tokens                              1t (1.00T tokens)
GPU                           H100 SXM (peak BF16 989 TFLOPS)
Count                              512 cards
Peak throughput                 506.37 PF/s (all GPUs)
MFU target                        50.0%
Effective FP/s                  253.18 PF/s

Total FLOPs                  420000.00 EF
Training time                    19.20 days (no overhead)
Checkpoint OH                     9.60 days
Fault factor                      1.08x (→ 8.0% loss)
Total time                       31.10 days

Card-hours                      382204 GPU-hours
Cost/M tokens     $               4.59 (@ $12.0/GPU-h)
Deficit vs 100%                 224.0%
```

### sim_calibrate.py 验证示例
```
Verification: Llama 3 405B
================================================================================

Model & Hardware
  Model                     405B params
  Total tokens              15.36T
  GPU                       H100 SXM
  Count                     16128

Time Accounting
  Reported                  168.0 days
  Estimated (MFU=35%)        93.4 days (-44.4 %)
  Error                     -74.6 days
  ...
```

## 版本

- CUDA 13.4 Update 2（驱动 R615 及以上）
- PyTorch 2.15.0
- NCCL 2.32.3
- 参考数据核对于 2026 年 11 月 4 日

## 文献

1. Llama 3 技术报告（2024-04）：405B 训练卡数、工期、MFU
2. DeepSeek-V3 后训练报告（2024-12）：H800 卡时数据
3. MegaScale（2402.15627）：集群训练实证数据
4. How To Scale Your Model：显存与通信账本
