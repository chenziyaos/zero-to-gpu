# GPU-03 CUDA 编程模型与并行执行

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/03-cuda.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-03](../book/03-cuda.md) 第 11 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

对应文档《GPU-03｜硬件｜CUDA 编程模型：线程层级、stream 与编译链》。

## 概述

本章代码模型化了主机与设备间的同步点、grid 配置的效率、H2D/D2H 拷贝与计算的重叠，以及 launch 开销的量化。所有计算器都可以在笔记本上运行（Python 3.10+ 标准库），无需 GPU。

## 文件清单

| 文件 | 环境 | 做什么 | 对应章节 |
|-----|-----|--------|---------|
| `sim_launch.py` | Python 3.10+ | Grid 配置与 wave 效率；H2D/D2H 带宽对比；流重叠模拟；launch 开销基准 | 第 2、6、8 章 |
| `test_launch.py` | Python 3.10+ | 10 项断言：grid 数学、Harris 带宽实测、Gray launch 开销 | 全章 |

## 运行方式

### grid 子命令：block 分配与 wave 效率

```bash
python3 sim_launch.py grid --n 1000000 --block 256 --sms 132
```

输出：grid block 总数、wave 数、最后一波的利用率、空闲槽位数。

**示例**：1M 元素，块大小 256，H100（132 SM，8 resident/SM）：

```
n=1000000, block_size=256, 132 SM, 8 resident/SM
  → 3907 blocks, 4 waves, last wave 70.0% utilized
  → 317 empty slots per full waves

Grid-stride: 4224 blocks (32 × SM), fill efficiently regardless of n

Configuration checks:
  grid=(70000, 1, 1) block=(1, 1, 1):   ✓ 合法
  grid=(1, 70000, 1) block=(256, 1, 1):   × gridDim.y=70000超出[1,65535]
  grid=(1, 1, 1) block=(1024, 2, 1):   × 每块线程数2048超过1024
  grid=(1, 1, 1) block=(32, 1, 65):   × blockDim.z=65超出[1,64] ; 每块线程数2080超过1024
```

### index 子命令：线程坐标映射

```bash
python3 sim_launch.py index --bx 4 --by 2 --tx 8 --ty 4
```

输出：2D grid 和 block 下的线程全局坐标与 warp 编号。

**示例**：4×2 grid，8×4 block（每 block 32 线程）：

```
gridDim=(4,2), blockDim=(8,4), row_width=32

Examples of thread to global coordinate mapping (行主序):
  blockIdx  threadIdx   (x, y)    global_idx   warp

  (0,0)    (7,0)      (7,0)    7             0
  (0,0)    (0,1)      (0,1)    32            1
  (1,0)    (0,0)      (8,0)    8             0
  (3,1)    (7,3)      (31,7)   255           7
```

### xfer 子命令：H2D/D2H 带宽对比

```bash
python3 sim_launch.py xfer --mib 256
```

对标 Harris（2016）在 T4 上的实测值。

**示例**：256 MiB 传输：

```
256 MiB transfer:
  pinned   @ 5.77 GB/s → 0.05 ms
  pageable @ 2.31 GB/s → 0.12 ms (差 2.50倍)

  PCIe 3.0 x16 单向 64 GB/s → 4.00 ms
  PCIe 4.0 x16 单向 128 GB/s → 2.00 ms
  PCIe 5.0 x16 单向 256 GB/s → 1.00 ms
```

### overlap 子命令：H2D/compute/D2H 并行模拟

```bash
python3 sim_launch.py overlap
```

离散事件模拟器，展示不同 stream、engine 配置下的重叠效果。

**示例**（4 个 stream，各 H2D+kernel+D2H）：

```
整块 H2D 1.0, kernel 1.0, D2H 1.0, 切成 4 个 stream
  单块：H2D 0.25, kernel 0.25, D2H 0.25

Serial time (all sequential) = 3.00 (h + k + d)

Scenario                                           V1       V2
--------------------------------------------------------------------
1 copy engine, single queue                     6.00     6.00
2 copy engines, single queue + Fermi signal (前一kernel全做完才放下一个)     6.00     6.00
2 copy engines, single queue (无Fermi信号)           3.00     3.00
2 copy engines, Hyper-Q                         3.00     3.00
1 copy engine, Hyper-Q                          3.00     3.00
```

### launch 子命令：Gray CUDA Graph 基准复现

```bash
python3 sim_launch.py launch
```

V100 上 20 kernel/step × 1000 步的开销对比。

**示例**：

```
20 kernels/step, 1000 steps, kernel base 2.9 us (V100):

1. 每个 kernel 后 sync：
   每 kernel  9.6 us
   总计      200.0 ms
   开销占比  96.7%

2. 每步 sync 一次：
   每 kernel  3.8 us
   总计      80.0 ms
   开销占比  93.7%

3. CUDA Graph：
   每 kernel  3.4 us
   总计      68.0 ms
   开销占比  93.1%

Graph 建图 400 us，每步省 0.4 us，
  5263 步回本
```

### 测试运行

```bash
python3 test_launch.py
```

验证 10 项物理约束和公开数字复现。

**示例**（全绿）：

```
✓ test_grid_efficiency_monotonicity: wave utilization ∈ [0, 1]
✓ test_grid_config_matches_cuda_spec: all configs valid
✓ test_wave_efficiency_full_utilization: perfect fits at 100%
✓ test_partial_wave_efficiency: empty slots computed correctly
✓ test_harris_pinned_transfer_physics: pinned 0.05 ms, pageable 0.12 ms (0.40× difference)
✓ test_pcie_bandwidth_scaling: PCIe 3/4/5 bandwidth monotonic
✓ test_overlap_monotonicity_with_engines: 1 engine 6.00, 2 engines 6.00
✓ test_overlap_lower_bound: sequential 3.00, actual 3.00, ideal overlap 1.00
✓ test_gray_launch_overhead_scaling: kernel 2900 us, overhead 9600 us, ratio 3.3×
✓ test_grid_ceil_arithmetic: ceil(n / block_size) correct

10/10 tests passed
```

## 核心数字

所有数字来自以下来源，核对于 2026 年 11 月 4 日：

- **H100 SXM**：132 SM，8 resident blocks/SM，3.35 TB/s HBM，450 GB/s NVLink（单向）
- **Harris（2016）T4 实测**：Pinned 5.77 GB/s，Pageable 2.31 GB/s（2.5× 差异）
- **Gray et al.（2019）CUDA Graph**：V100 kernel 2.9 us，launch overhead 3.8–9.6 us
- **PCIe**：Gen3 @ x16 = 64 GB/s，Gen5 @ x16 = 256 GB/s（理论值）

## 延伸阅读

- NVIDIA CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-programming-guide/
- Gray et al. "GPU performance prediction for arbitrary compute kernels"（CUDA Graph paper）：https://arxiv.org/abs/1810.08226
- Harris "Pinned memory and host-device transfer"（2016）：https://developer.nvidia.com/blog/
