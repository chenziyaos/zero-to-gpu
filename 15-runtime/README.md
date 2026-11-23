# GPU-15 · PyTorch 运行时 · 配套代码

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-gpu/blob/main/colab/15-runtime.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-15](../book/15-runtime.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

## 一句话

GPU 等 CPU 发射：Python 和框架开销让 kernel 之间出现空隙，torch.compile 融合与 CUDA Graph 消除 launch 等待、多流隐藏拷贝与计算的串行。

## 文件清单

| 文件 | 环境 | 做什么 | 对应文档章节 |
|---|---|---|---|
| `sim_launch.py` | Python 3.10+（纯标准库） | 模拟 eager、CUDA Graph、融合三套发射模型及时间账 | 第 2-3 章 |
| `sim_compile.py` | Python 3.10+（纯标准库） | 典型训练块 Inductor 融合前后的发射次数和 HBM 字节 | 第 5 章 |
| `sim_streams.py` | Python 3.10+（纯标准库） | 三段流水（H2D、SM、D2H）重叠和 SM 忙时计算 | 第 6 章 |
| `test_runtime.py` | Python 3.10+（纯标准库） | 25 个断言（6 个 eager bubble、3 个公开数据、16 个物理检查）| 全章 |
| `check_runtime.py` | PyTorch 2.15+（可选） | 只读核对版本、设备、CUDA Graph/compile 能力 | 第 9 章 |
| `gpu_runtime.py` | PyTorch 2.15+ + CUDA 13.4+（需要真卡） | GPU 上测量 launch、Graph、同步、多流的真实时间 | 第 8 章 |

## 快速开始

### 1. 模拟器（无需 GPU）

```bash
# eager 发射气泡：2000 个 1 us 小核，5 us 发射间隔
python3 sim_launch.py eager --kernels 2000 --kernel-us 1 --launch-us 5

# CUDA Graph 成本摊薄：录制一次，重放多次
python3 sim_launch.py graph --kernels 2000 --kernel-us 1 --launch-us 5

# 三套方案对比（eager vs Graph vs fusion）
python3 sim_launch.py compare

# 典型训练块 compile 前后账单
python3 sim_compile.py typical --batch 128 --seq 1024 --d 4096 --ff 14336

# 多流流水效率（均衡三段）
python3 sim_streams.py compare
```

### 2. 测试（无需 GPU）

```bash
python3 test_runtime.py
```

### 3. 真卡测量（可选，需要 GPU 和 PyTorch）

```bash
python3 gpu_runtime.py all
```

## 核心假设（参数标定）

| 参数 | 值 | 出处 |
|---|---|---|
| HBM 有效带宽（H100） | 2.68 TB/s（3.35 TB/s × 80%） | H100 产品页 |
| 发射间隔 | 5 us | PyTorch CUDA semantics |
| Guard 检查 | 2 us | torch.compile 实测 |
| PCIe Gen5 x16 | 63 GB/s 单向 | PCIe 5.0 规格 |
| 融合开销削减 | 30% | Inductor + 文献 |

## 版本基线

- PyTorch: 2.15.0+
- CUDA: 13.4 Update 2+
- NCCL: 2.32.3+

## 注

**未在真卡上复测。期望值来自 PyTorch 官方文档、NVIDIA 规格、公开论文与 H100 标称参数。**
