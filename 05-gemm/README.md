# GPU-05 GEMM 逐步优化：配套代码

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/05-gemm.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-05](../book/05-gemm.md) 第 2、5、8 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-05 对应 GEMM 每一级分块的性能账本和优化效果。配套代码模拟不同分块策略下的访存和占用率，回答"为什么分块能加速、线程分块怎样提高数据复用、低占用率是否影响 GEMM 性能"三个问题。

## 文件清单

| 文件 | 环境 | 做什么 | 对应章节 |
|------|------|--------|----------|
| `sim_gemm.py` | Python 3.10+ | 计算朴素/分块 GEMM 的访存字节数和算术强度，对标 Roofline | 第 1-5 章：账本与分块优化 |
| `sim_occupancy.py` | Python 3.10+ | 分析线程分块与寄存器、占用率的权衡，扫参数找最优配置 | 第 6 章：占用率与寄存器 |
| `test_gemm.py` | Python 3.10+ | 12 项测试验证访存、强度、占用率的计算（all 绿） | - |

## 运行命令

### 1. 朴素 GEMM 访存基准（强度 0.25 FLOP/字节）

```bash
python3 sim_gemm.py naive --m 4096 --n 4096 --k 4096
```

期望输出：
- 最少访存 201.3 MB
- 朴素实际 549,822.9 MB（每个 C 元素都扫全 K）
- 强度 0.25 FLOP/字节（带宽受限）

### 2. 分块优化（访存下降 1024 倍，强度升至 682 FLOP/字节）

```bash
python3 sim_gemm.py blocked --m 4096 --n 4096 --k 4096 --bm 128 --bn 128 --bk 8
```

期望输出：
- Block 布局 32×32 = 1024 blocks（占满 H100 132 个 SM）
- 访存回到最小值 201.3 MB
- 强度 682.67 FLOP/字节（算力受限）

### 3. 线程分块对数据复用的效果（扫 TM, TN）

```bash
python3 sim_gemm.py thread-tile --m 4096 --n 4096 --k 4096
```

期望输出：
- TM=1×TN=1：强度 329
- TM=8×TN=8：强度 5,265（16 倍）
- 更大分块 → 共享内存读次数少 → 强度高

### 4. 算术强度与瓶颈分析（H100 拐点 295 FLOP/B）

```bash
python3 sim_gemm.py intensity --m 4096 --n 4096 --k 4096 --gpu h100-sxm
```

期望输出：
- H100 峰值 989 TFLOPS，带宽 3.35 TB/s
- 拐点 295 FLOP/字节
- 4096³ 矩阵强度 682 > 295 → 算力瓶颈

### 5. 占用率分析（为什么 GEMM 占用率低）

```bash
python3 sim_occupancy.py analyze
```

期望输出：
- 所有线程分块配置占用率 25% 以下
- 原因：累加器占用大量寄存器（8×8 需 192 个 FP32）
- 结论：低占用率是正常的，指令级并行补偿

### 6. 参数扫描找最优配置（占用率 >= 50%）

```bash
python3 sim_occupancy.py sweep
```

期望输出：
```
推荐配置：
  Block size: 512，线程分块: 4x4，占用率: 50.0%
```

### 7. 共享内存与寄存器的约束

```bash
python3 sim_occupancy.py shared-memory
```

期望输出：
- 小分块（64×64×8）：可行（8 KB < 96 KB）
- 大分块 N（128×256×16）：不可行（64 KB × 2 = 128 KB > 96 KB）

### 8. 完整测试套件（12 项全绿）

```bash
python3 test_gemm.py
```

期望输出：
```
✓ test_flops_calculation              # 验证 2MNK
✓ test_h100_ridge_point               # 验证 H100 拐点 295 FLOP/B
✓ test_blocked_reduces_access         # 验证分块减少全局访存
✓ test_arithmetic_intensity_increases # 验证强度随分块增加
... (12 total)

结果：12 通过，0 失败
```

## 本篇代码输出的关键数字（均为真实测试结果）

见 `drafts/05/code_outputs.md` 完整输出。核心数据：

| 优化步骤 | 访存 MB | 强度 FLOP/B | 瓶颈 | 来源 |
|---------|--------|-----------|------|------|
| 朴素 | 549,822 | 0.25 | 带宽 | sim_gemm.py naive |
| 分块 128×128×8 | 201.3 | 682.67 | 算力 | sim_gemm.py blocked |
| 线程分块 1×1 | 共享内存 | 329 | 带宽 | sim_gemm.py thread-tile |
| 线程分块 8×8 | 共享内存 | 5,265 | 混合 | sim_gemm.py thread-tile |

## 测试验证

```bash
# 检查 Python 语法
python3 -m py_compile sim_gemm.py sim_occupancy.py test_gemm.py

# 运行完整测试
python3 test_gemm.py
```

所有代码遵循 GPU-05 第 1 至 8 章的公式和数据依据：
- Roofline 模型（Berkeley EECS-2008-134）
- Boehm 的 10 步 CUDA GEMM（siboehm.com）
- Volkov 寄存器分块（SC08）
- NVIDIA H100 官方规格（核对于 2026-11-04）

环境约束：
- Python 3.10+（仅标准库，本机无 numpy）
- 共享 `common/specs.py` 的 GPU 规格表
- H100 SXM 用作基准 GPU
