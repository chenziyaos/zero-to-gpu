# GPU-14 Profiling：配套代码

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/14-profiling.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

性能分析的核心思想：先用 Nsight Systems 看整体时间线找空隙，再用 Nsight Compute 深挖 kernel 内的瓶颈。本目录包含两个 CSV 解析器、四段示例负载和一份分析脚本。

## 文件清单

| 文件 | 环境 | 做什么 | 对应文档章节 |
|------|------|--------|-----------|
| sim_prof_parse.py | Python 3.10+ (仅标准库) | nsys/ncu CSV 解析，五分类、Roofline 账本 | GPU-14 第 2、3、5、6 章 |
| sim_gpu_util.py | Python 3.10+ (仅标准库) | GPU-Util 误导现象演示 | GPU-14 第 3 章 |
| test_prof.py | Python 3.10+ (仅标准库) | ≥8 个断言，复现 H100 脊点 295 FLOP/byte | 所有章节 |
| sample_nsys_kernels.csv | 任何机器 | 示例 nsys 输出：3 个 GEMM + 1 个 scale + 2 个 memcpy | 实验素材 |
| sample_ncu_metrics.csv | 任何机器 | 示例 ncu 指标：五分类演示 | 实验素材 |
| gpu_workload.py | CUDA 13.4+ | 四段刻意写坏的负载生成器（仅 CPU 部分） | GPU-14 第 10 章案例 |
| profile.sh | CUDA 13.4 + nsys/ncu | 一键抓 nsys + ncu 报告（未在真卡测试） | GPU-14 第 7 章 |

## 运行命令

### 解析 nsys 时间线（CPU 只需 3 秒）

```bash
python3 sim_prof_parse.py nsys --csv sample_nsys_kernels.csv
```

期望输出：
```
设备利用率分析
总时间线长度        :      6.559 s
Kernel 总时间       :      4.759 s ( 72.6%)
Memcpy 总时间       :      1.800 s ( 27.4%)
GPU 设备忙闲比      : 100.0%

按 Kernel 名统计
volta_fp32_gemm_kernel                          3  4191500.0 us
volta_fp32_scale_kernel                         1   567800.0 us

间隙分类
最大 kernel 间隙    :   777800.0 us
判断: 发射受限(间隙 > 100 us)
```

### 分析 ncu 指标五分类（CPU 即时）

```bash
python3 sim_prof_parse.py ncu --csv sample_ncu_metrics.csv --gpu h100-sxm
```

期望输出：
```
五分类瓶颈分析 (GPU: h100-sxm)
TENSOR:
  volta_fp32_gemm_1: SOL-Compute 68.5%, 带宽 2246 GB/s
LATENCY:
  volta_fp32_scale_1: Occupancy 42.3% 不足

Roofline 利用率
H100 SXM Roofline 脊点: 295 FLOP/byte
```

### GPU-Util 现象演示（CPU 即时）

```bash
python3 sim_gpu_util.py scenario --num-kernel 1 --sm-occupancy 0.01
```

期望输出：
```
GPU 型号              : h100-sxm (132 SMs)
总活跃 SM 数          : 1 / 132
GPU-Util（报告值）   : 100.0%
SM Activity（真实值）:   0.8%
```

```bash
python3 sim_gpu_util.py mislead
```

演示四种误导现象：单 kernel 100% GPU-Util、多卡平均 GPU-Util、NUMA 碎片等。

### 运行完整测试

```bash
python3 test_prof.py
```

期望：所有 ≥8 个断言通过，包括：
- nsys CSV 解析和时间线计算
- ncu 五分类判定（TENSOR / MEM / LATENCY / COMPUTE / LAUNCH）
- Roofline 脊点复现（H100 约 295 FLOP/byte）
- GPU-Util 虚假现象验证（1 SM → 100% GPU-Util, 0.76% SM Activity）

## 核心公式与数据

### 设备忙闲比（时间线分析）
```
GPU-Util = (Kernel 时间 + Memcpy 时间) / 总时间线长度 × 100%
```

### 五分类瓶颈判定（nccompute metrics）
| 类别 | 判定条件 | 优化方向 |
|-----|---------|--------|
| MEM | SOL-Memory > 80% | 减少 DRAM 访存，增加数据复用 |
| COMPUTE | SOL-Compute > 50%, SOL-Memory < 50% | 增加 SM 并行或 Tensor Core |
| TENSOR | SOL-Compute > 50%, 高带宽 | Tensor Core 已饱和，计算受限 |
| LATENCY | SM Occupancy < 50% | 提高占用率，隐藏访存延迟 |
| LAUNCH | Kernel 数多、平均时长短 | 合并小核，用 CUDA Graph |

### Roofline 脊点（H100 SXM）
```
Ridge = 峰值算力 / 显存带宽 = 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte
算术强度 > 295 → 受 SM 计算限制
算术强度 < 295 → 受 DRAM 带宽限制
```

## 样例数据说明

sample_nsys_kernels.csv：
- 3 × volta_fp32_gemm_kernel (1.3~1.5 ms 各)
- 1 × volta_fp32_scale_kernel (0.57 ms)
- 2 × memcpy (0.8 ms HtoD, 1.0 ms DtoH)
- 总时间线 6.6 s，设备忙闲比 100%（顺序执行无空隙）
- 最大 kernel 间隙 778 us，触发"发射受限"判定

sample_ncu_metrics.csv：
- GEMM_1 & _2: 高计算占用 (68-71%)，中等内存 (28-31%)，分类为 TENSOR
- SCALE_1: 低计算占用 (15%)，高内存占用 (79%)，分类为 LATENCY（占用率仅 42%）
- Stall reason 主要为 MemDep (42-73%)，其次 LG Throttle (5-12%)

## GPU 侧脚本说明

未在本仓库真卡上复测。

- **gpu_workload.py** - 四段刻意写坏的 kernel：
  1. launch_heavy: 2000 个 1 us 微核，间隙 10-20 us
  2. memory_bound: 矩阵转置无复用，行冲突
  3. compute_latency: 低占用、高依赖链
  4. good_kernel: 参考实现，双缓冲

- **profile.sh** - 一键 nsys + ncu 脚本，依赖 CUDA 13.4 + nsys + ncu

需要在 H100 或兼容卡上运行。输出 CSV 后用 sim_prof_parse.py 解析。
