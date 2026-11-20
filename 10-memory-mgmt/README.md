# GPU-10 显存管理 — PyTorch 缓存分配器与 OOM

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/10-memory-mgmt.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

PyTorch CUDA caching allocator 的工作机制模拟（CPU 端），以及训练显存预算与 OOM 重试链分析。核心演示：块对齐、两池设计、碎片化、切分合并、重试流程。

## 文件清单

| 文件 | 环境 | 功能 | 对应文档 |
|------|------|------|---------|
| `sim_alloc.py` | Python 3.10+ | 模拟分配器块管理、对齐、取整、池选择、切分合并 | GPU-10 第 2-4 章 |
| `sim_oom_retry.py` | Python 3.10+ | 显存预算计算、OOM 重试流程、模型规模扫描 | GPU-10 第 4、8 章 |
| `test_memory_mgmt.py` | Python 3.10+ | 15 项单元测试（论文与文献复现） | 全章 |

## 运行命令与期望输出

### 1. 单条分配请求（块对齐、池选择、新段创建）

```bash
python3 sim_alloc.py allocate --request-bytes 1048576
```

期望：1 MiB 请求 → 小池（≤1 MiB 阈值） → 2 MiB 段 → 50% 利用率

### 2. 分配序列与释放（追踪碎片演变）

```bash
python3 sim_alloc.py sequence --pattern "1M,2M,1M"
```

期望：1 MiB 进小池，2 MiB 进大池，演示两段并存与顺序释放

### 3. 显存预算：7B 模型单卡预算

```bash
python3 sim_oom_retry.py membudget --params-billion 7 --layers 32 --seq 2048 --dmodel 4096
```

期望：参数 14 GB + 激活 1 GB + 梯度 14 GB + Adam 28 GB = 57 GB（H100 80GB SXM 可行）

### 4. 显存预算：带 checkpointing（激活节省 70%）

```bash
python3 sim_oom_retry.py membudget --params-billion 7 --layers 32 --seq 2048 --dmodel 4096 --checkpointing
```

期望：激活降至 0.3 GB 左右，总预算降至 56 GB 以下

### 5. OOM 重试流程（直接成功）

```bash
python3 sim_oom_retry.py retry --total-reserved-gb 40 --request-gb 5
```

期望：可用 40 GB ≥ 请求 5 GB，0 次重试

### 6. 碎片压力测试（重复随机分配与释放）

```bash
python3 sim_alloc.py fragstress --iterations 100 --seed 42
```

期望：观察碎片率随迭代增长；200 次迭代后可能达到 20-30% 碎片

### 7. 完整测试套件（15 项单元测试）

```bash
python3 test_memory_mgmt.py
```

期望：全部通过
- 块对齐与池选择（物理约束）
- 单次分配、切分、合并（状态转换）
- 激活与参数显存计算（论文复现）
- OOM 重试流程（成功/失败分支）
- 碎片率单调性与合理范围检查（ZeRO-3 30% 阈值验证）

## 核心常量与出处

（均来自 PyTorch c10/cuda/CUDACachingAllocator.cpp，核对于 2026-11-04）

- **BLOCK_LO = 512 B**：块对齐单位
- **SMALL_POOL_THRESHOLD = 1 MiB**：池分界
- **SMALL_BUFFER_SIZE = 2 MiB**：小池段大小
- **LARGE_BUFFER_SIZE = 20 MiB**：大池段大小
- **LARGE_BLOCK_THRESHOLD_PCT = 50%**：OOM 时释放"超大块"的阈值百分比（保守估计）

## 关键论文与文献

1. [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) — memory_reserved vs memory_allocated
2. [c10/cuda/CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp) — 分配器实现
3. [ZeRO-3 论文 (Rajbhandari et al. 2021)](https://arxiv.org/abs/1910.02054) — 显存碎片 30%+ 导致 OOM 观察
4. [stas00/ml-engineering](https://github.com/stas00/ml-engineering) — 加速器峰值数据
