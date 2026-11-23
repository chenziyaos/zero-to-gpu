# 12-sharing

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-gpu/blob/main/colab/12-sharing.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-12](../book/12-sharing.md) 第 9 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-12 MIG、MPS 与容器的配套代码。对应文档 GPU-12 第 2-3 章（MIG 硬件与 MPS SM 分区）。

## 文件清单

| 文件 | 环境 | 做什么 | 对应文档 |
|------|------|--------|---------|
| `sim_mig_plan.py` | Python 3.10+ 标准库 | MIG profile 支持列表、混合需求放置、显存 slice 分配方案 | 第 2-3 章 MIG 硬件与放置 |
| `sim_mps_calc.py` | Python 3.10+ 标准库 | MPS SM 分区、块大小、舍入开销、多进程吞吐估计 | 第 6-7 章 MPS 控制与分区 |
| `test_sharing.py` | Python 3.10+ 标准库 | 单元测试（12 个，6 个复现公开数字 + 6 个物理约束） | 验证码 |

## 运行方式

### MIG 计算器

```bash
# 列出所有支持的 H100 profile
python3 sim_mig_plan.py avail

# 检查混合 profile 需求能否放下（2 个 3g.20gb + 1 个 1g.5gb）
python3 sim_mig_plan.py check --demand 3g.20gb 3g.20gb 1g.5gb

# 检查超限情况（8 个 1g.5gb，超过算力上限 7）
python3 sim_mig_plan.py check --demand 1g.5gb 1g.5gb 1g.5gb 1g.5gb 1g.5gb 1g.5gb 1g.5gb 1g.5gb

# 计算理论上限
python3 sim_mig_plan.py ceiling
```

### MPS 计算器

```bash
# 2 进程均分 H100（每个 50%）
python3 sim_mps_calc.py partition --processes 2 --percentage 50 50

# 4 进程不均匀分配（40%, 30%, 20%, 10%）
python3 sim_mps_calc.py partition --processes 4 --percentage 40 30 20 10

# 计算 4 进程时的块大小
python3 sim_mps_calc.py chunk --max-processes 4

# 分析舍入开销（理想 50 SM × 2）
python3 sim_mps_calc.py overhead --sm 132 --allocations 50 50

# 估计 2 进程的相对吞吐
python3 sim_mps_calc.py utilization --processes 2
```

### 单元测试

```bash
python3 test_sharing.py
```

## 期望输出

### MIG avail
```
H100 80GB 支持的 MIG profile（来源：NVIDIA MIG User Guide Hopper，Table 5）
Profile      Compute         Memory               Size  
------------------------------------------------------------
1g.5gb     compute=1/7  mem=1/8   5GB
2g.10gb    compute=2/7  mem=2/8  10GB
3g.20gb    compute=3/7  mem=2/8  10GB
4g.20gb    compute=4/7  mem=4/8  20GB
6g.40gb    compute=6/7  mem=4/8  20GB
7g.80gb    compute=7/7  mem=8/8  80GB
```

### MPS partition (2 进程 50% 各)
```
H100 MPS SM 分区（132 SM，2 进程）
块大小: 66.0 SM

进程     目标%      理想SM     块数     实际SM     实际%      浪费      
------------------------------------------------------------
     1    50.0%    66.0      1    66.0    50.0%     0.0
     2    50.0%    66.0      1    66.0    50.0%     0.0
------------------------------------------------------------
总计       100.0%   132.0 -   132.0   100.0%
```

### 单元测试
```
全部 12 个测试通过 ✓
- 复现 NVIDIA 官方 MIG profile 规格表
- 复现 NVIDIA MPS 块大小算法
- 物理约束验证（显存连续性、算力上限、单调性、数量级）
```
