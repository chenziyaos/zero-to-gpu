# 07-dsl

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/07-dsl.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-07](../book/07-dsl.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-07 Triton、CUTLASS 与 CuTe DSL 的配套练习。

## 文件表

| 文件 | 环境 | 做什么 | 对应文档章节 |
|------|------|--------|-----------|
| `sim_dsl.py` | Python 3.10+ stdlib | Triton 块调度、CuTe 布局代数、GEMM 配置空间过滤、Swizzle bank conflict 分析 | GPU-07 第 1-9 章 |
| `sim_autotune.py` | Python 3.10+ stdlib | 共享内存用量、持久化 kernel 网格分析、Autotune 配置扫描 | GPU-07 第 3-4 章 |
| `test_dsl.py` | Python 3.10+ stdlib | 13 个单元测试：块分组、布局代数、配置过滤、Swizzle 等 | — |
| `gpu_triton_mm.py` | PyTorch 2.15+, Triton 2.2+ | Triton 矩阵乘 kernel 框架（stub，需在 GPU 上运行） | GPU-07 第 2-3 章 |
| `gpu_cute_add.py` | NVIDIA CUTLASS 3.x, nvidia-cutlass-dsl | CuTe DSL 布局和 swizzle 示例（stub） | GPU-07 第 6-8 章 |
| `bench_dsl.sh` | NVIDIA GPU + PyTorch + Triton | Triton 和 CuTe 的 benchmark 脚本（stub） | GPU-07 第 10 章 |

## 运行方式

### CPU 模拟（无需 GPU）

```bash
# 块调度效果（Triton 教程：9×9 矩阵，GROUP_SIZE_M=8）
python3 sim_dsl.py block-order --m 1152 --n 1152 --k 1024 --group-size-m 8 --block-size-m 128 --block-size-n 128

# CuTe 布局代数演示
python3 sim_dsl.py layout-algebra --shape "256:16,128:256" --coalesce

# GEMM 自动调优配置空间（H100，4096×4096×4096）
python3 sim_dsl.py config-search --gpu h100-sxm --m 4096 --n 4096 --k 4096 --dtype f32

# Swizzle bank conflict 分析
python3 sim_dsl.py swizzle --smem-bytes 96000 --dtype f32

# 共享内存用量
python3 sim_autotune.py smem-usage --block-m 128 --block-n 128 --block-k 32 --dtype f32

# 持久化 kernel 网格分析
python3 sim_autotune.py persistent-grid --m 4096 --n 4096 --k 4096 --block-m 64 --block-n 64

# Autotune 配置过滤（H100，512×512×512）
python3 sim_autotune.py autotune-sweep --gpu h100-sxm --m 512 --n 512 --k 512

# 运行所有测试
python3 test_dsl.py
```

## 期望输出

### 块调度分析

```
块调度分析 (1152×1152×1024)
  num_blocks_m          9
  num_blocks_n          9
  total_blocks_naive    81
  blocks_first_wave     9
  num_groups            2
  L2_improvement_%      88.9%
```

说明：Triton 官方矩阵乘教程用 GROUP_SIZE_M 分组，将 9×9=81 块分成 2 组，第一波（56 个 SM）只需启动 9 块。

### 配置空间过滤

```
GEMM 自动调优配置空间 (h100-sxm, 4096×4096×4096, f32)
  max_smem_mb           96.0
  theoretical_configs   64
  valid_configs         64
  filter_ratio          100.0%
```

说明：64 个理论配置中，64 个满足 H100 的 96KB 共享内存限制。对于更大的矩阵或更小的 SMEM，过滤比例会更低。

### Autotune 过滤示例

```
Autotune 配置扫描 (h100-sxm, 512×512×512)
  all_candidate_configs  54
  valid_configs_count    24
  filter_ratio           44.4%
```

说明：Triton 官方持久化教程提到 24 个配置在 96KB SMEM 下可行；在更严格的约束下（如 RTX 4090 的 96KB），只有 ~6 个配置可行。

### 测试输出

```
运行 GPU-07 DSL 性能模型测试...
✓ test_block_order_basic passed
✓ test_block_order_published passed
✓ test_layout_algebra_volume passed
✓ test_layout_algebra_contiguity passed
✓ test_layout_algebra_coalesce passed
✓ test_config_search_filter passed
✓ test_config_search_published passed
✓ test_smem_usage_bounds passed
✓ test_smem_usage_scaling passed
✓ test_swizzle_basic passed
✓ test_swizzle_conflict_reduction passed
✓ test_autotune_sweep_published passed
✓ test_numerical_consistency passed

✓ 全部 13 个测试通过
```

## GPU 运行（需要硬件）

`gpu_triton_mm.py`、`gpu_cute_add.py` 和 `bench_dsl.sh` 需要 Ampere 及以后的 NVIDIA GPU（A100、H100 等）。

在有 GPU 的机器上：

```bash
# 需要环装环境
pip install torch==2.15+ triton==2.2+

# 运行 benchmark
bash bench_dsl.sh --m 4096 --n 4096 --k 4096 --dtype f32
```

## 源代码和文献

CPU 模拟中引用的数字来自：

1. Triton 官方矩阵乘教程（块分组、持久化）
   - https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
   - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

2. NVIDIA Hopper 白皮书（Swizzle、共享内存结构）

3. stas00/ml-engineering 的 GPU 加速器数据表（MAMF 和规格）

本代码仅在 CPU 上模拟上述约束；GPU 侧代码（.cu、Triton kernel）未在真卡上复测，期望值来自文献。
