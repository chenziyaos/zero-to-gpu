# 04-memory：访存与归约

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/04-memory.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

GPU-04「访存与归约：合并访存、共享内存与 warp 归约」的配套代码。

模拟和分析 GPU 全局内存的 sector 合并、共享内存的 bank 冲突，以及矩阵转置和并行归约的逐步优化。

## 文件一览

| 文件 | 环境 | 做什么 | 文档章节 |
|------|------|--------|---------|
| `sim_memory.py` | Python 3.10+ | CPU 模拟器：sector、bank 冲突、转置、归约性能估计 | 第 1-7 章 |
| `test_memory.py` | Python 3.10+ | 单元测试（16 个测试，验证论文数据） | 第 1-7 章 |
| `gpu_launch.py` | Python 3.10+ | GPU 内核启动配置与性能计算器 | 第 1-7 章 |
| `coalesce.cu` | CUDA 13+、Volta+ | 全局内存合并访存示例（三种步长模式） | 第 1-2 章 |
| `transpose.cu` | CUDA 13+、Volta+ | Harris 转置三版本（naive、padded、xor_swizzle） | 第 5 章 |
| `reduce.cu` | CUDA 13+、Volta+ | Harris 归约七步优化（从朴素到 warp shuffle） | 第 6-7 章 |
| `ncu_mem.sh` | Nsight Compute 2026.3.1+ | 分析脚本：运行 ncu 采集内存指标 | 第 1-7 章 |

## 快速开始

### 单元测试

```bash
cd repo/zero-to-gpu/04-memory
python3 test_memory.py
```

预期输出：16 个 ok

```
ok   test_coalesce_continuous
ok   test_coalesce_stride_2
ok   test_coalesce_stride_32
ok   test_row_utilization
ok   test_col_utilization_stride_1
ok   test_bank_no_conflict
ok   test_bank_32_way_conflict
ok   test_bank_gcd_16
ok   test_transpose_naive_vs_swizzle
ok   test_transpose_padded_middle
ok   test_reduce_stage_monotonicity
ok   test_reduce_dtype_size
ok   test_sector_size_constant
ok   test_shared_banks_constant
ok   test_warp_size_constant
ok   test_coalesce_boundary_aligned
```

### Sector 合并分析

```bash
# 连续读：4 个 sector，100% 利用率
python3 sim_memory.py coalesce --pattern row --m 128 --n 32 --stride 1
```

预期输出：
```
模式         row
矩阵            128x32    
步长         1
有效字节            128 B
实际搬运            128 B
Sector 数          4
利用率           100.0%
```

```bash
# 按列读：125 个 sector，3.2% 利用率（低效）
python3 sim_memory.py coalesce --pattern col --m 128 --n 32 --stride 1
```

预期输出：
```
模式         col
矩阵            128x32    
步长         1
有效字节            128 B
实际搬运           4000 B
Sector 数        125
利用率             3.2%
```

### Bank 冲突分析

```bash
# 步长 1：无冲突
python3 sim_memory.py bank --stride 1
# 步长 16：16 路冲突
python3 sim_memory.py bank --stride 16
# 步长 32：32 路冲突（最坏）
python3 sim_memory.py bank --stride 32
```

预期输出：
```
步长         1
Bank 冲突路数              1
状态         无冲突
步长         16
Bank 冲突路数             16
状态         16 路冲突（会导致 16x 延迟）
步长         32
Bank 冲突路数             32
状态         32 路冲突（会导致 32x 延迟）
```

### Harris 转置性能估计

```bash
python3 sim_memory.py transpose --m 4096 --n 4096
```

预期输出（H100 基准）：
```
优化阶段                       GB/s         时间 (ms)
--------------------------------------------------
naive                      0.37         183.631
shared_padded              1.51          44.517
xor_swizzle                2.86          23.430

矩阵尺寸：4096 x 4096
总数据量：0.07 GB
```

核心观察：从 naive 到 xor_swizzle，性能提升约 7.7 倍，主要是消掉共享内存的 32 路写冲突。

### 归约七步性能估计

```bash
python3 sim_memory.py reduce --n 4194304 --dtype bf16
```

预期输出（H100 基准）：
```
阶段                    GB/s         时间 (us)
---------------------------------------------
stage1                2.86         17572.4
stage4                0.90          9274.3

元素数：4194304
数据类型：bf16
总数据量：0.008 GB
```

核心观察：前三步受带宽限制（2.86 GB/s），第 4 步后（stage4）指令成为瓶颈（0.90 GB/s），性能平台期。

### GPU 启动配置与规格查询

```bash
python3 gpu_launch.py
```

预期输出：
```
GPU 规格                                                       h100-sxm
--------------------------------------------------------------------------------
架构                             Hopper
SM 数                           132
BF16 峰值                             989 TFLOPS
HBM 带宽                             3.35 TB/s
NVLink 双向带宽                         900 GB/s
PCIe 双向带宽                           128 GB/s

推荐启动配置                                                      
--------------------------------------------------------------------------------
memory          block=256  grid=  3907  shared_mem=     0B
compute         block=128  grid=  7813  shared_mem=     0B
reduce          block=256  grid=  3907  shared_mem=  1024B

带宽利用率目标                                                     
--------------------------------------------------------------------------------
连续访问 (stride=1)                100% (理想)
步长 2                           ~50%
步长 8                           ~12.5%

Bank 冲突目标（共享内存）                                             
--------------------------------------------------------------------------------
无冲突 (stride=1)                 1x 延迟
4 路冲突                          4x 延迟
32 路冲突（最坏）                     32x 延迟（应避免）
```

## GPU 端编译与分析（需要 GPU 机器）

### 编译 CUDA 内核

```bash
# 编译全局内存合并访存示例
nvcc -O3 -std=c++17 -arch=sm_80 coalesce.cu -o coalesce

# 编译 Harris 转置三版本
nvcc -O3 -std=c++17 -arch=sm_80 transpose.cu -o transpose

# 编译 Harris 归约七步优化
nvcc -O3 -std=c++17 -arch=sm_80 reduce.cu -o reduce
```

### 用 Nsight Compute 分析

```bash
# 分析 transpose 的 bank 冲突和 sector 利用率
./ncu_mem.sh ./transpose

# 分析 reduce 的访存模式演进
./ncu_mem.sh ./reduce
```

关键指标：
- **Sector Efficiency**：实际读字节数 / 搬运字节数（% ）；目标 >80%
- **Request Size / Sector Size**：越接近 1 越好；>2 表示浪费
- **Bank Conflicts**：共享内存冲突数；应接近 0

## 与文档的对应关系

- **第 1-2 章**（导读、全局内存访问模式）：`sim_memory.py coalesce`
- **第 3 章**（数据布局与向量化）：`sim_memory.py coalesce`（不同布局）
- **第 4-5 章**（共享内存、bank 冲突、转置案例）：`sim_memory.py bank` 和 `transpose.cu`
- **第 6-7 章**（归约七步）：`sim_memory.py reduce` 和 `reduce.cu`
- **第 8 章**（异步拷贝）：`gpu_launch.py`（配置建议）
- **第 10 章**（融合与验证）：`ncu_mem.sh`（实际测量）

## 注意事项

1. **CPU 模拟的局限性**：`sim_memory.py` 是纯 Python 模型，不包含实际的 kernel 执行、L1/L2 缓存效应、功耗墙降频。绝对数值仅供参考；相对趋势可信。

2. **GPU 端代码未在本仓库复测**：`.cu` 和 `.sh` 文件假设 Volta（SM 7.0）及以上、CUDA 13.4 及以上。实际性能因卡的代次和驱动版本而异。

3. **Volta 及以后的同步**：Volta（Compute Capability 7.0）之后，`volatile` 不能保证 warp 间的隐式同步，改用 `__syncthreads()` 或 `__syncwarp()`（Kepler+ 可用）。

4. **warp shuffle** 需要 Kepler（SM 3.0）及以上；题目假设 Volta+。

5. **Nsight Compute 版本**：2026.3.1 及以上才有完整的 memory workload 和 bank conflict 指标。

## 参考文献

1. NVIDIA CUDA 编程指南：Global Memory Access Patterns（第 5.3.2 节）
2. Harris, M. (2008). "Efficient Matrix Transpose in CUDA Without Bank Conflicts"
3. Harris, M. (2007). "Parallel Reduction"（NVIDIA GPU Computing Gems 第 2 卷）
4. NVIDIA Nsight Compute 文档：Memory Metrics
5. stas00/ml-engineering：GPU 加速器规格表（核对于 2026-11-02）

## 许可与共享

本代码目录属于「从 0 开始 GPU 优化」系列教材配套。代码注释以中文为主；Python/CUDA/shell 关键字和库函数名保持英文。
