# 11-pcie-gdr

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-gpu/blob/main/colab/11-pcie-gdr.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-11](../book/11-pcie-gdr.md) 第 9 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-11 PCIe、NUMA 与 GPUDirect 的配套代码。GPU 与主机之间的数据搬运性能取决于 PCIe 链路配置、系统拓扑和内存管理策略。本目录提供了模拟器和拓扑解析工具，用于量化各环节的开销。

## 文件清单

| 文件 | 环境 | 做什么 | 对应章节 |
|------|------|--------|---------|
| `sim_pcie_bw.py` | CPU，Python 3.10+ 标准库 | PCIe 代际、编码、MPS 对 H2D/D2H 带宽的影响分解 | 第 1-2 章 |
| `sim_topo.py` | CPU，Python 3.10+ 标准库 | 解析 `topo -m` 矩阵，分类设备间距离和 NCCL 路由策略 | 第 3 章 |
| `test_pcie_gdr.py` | CPU，Python 3.10+ 标准库 | 9 项单测，复现 PCIe 规格数据和物理卫生检查 | 全章 |

## 运行命令

```bash
# PCIe 峰值带宽
python3 sim_pcie_bw.py peak

# H2D 带宽分解（Gen5 x16, MPS 256B）
python3 sim_pcie_bw.py h2d --gen gen5 --width 16 --mps 256

# D2H 拷贝带宽
python3 sim_pcie_bw.py d2h --gen gen5 --width 16

# 拓扑距离说明
python3 sim_pcie_bw.py topo

# Pinned 内存加速倍率
python3 sim_pcie_bw.py pinned

# 生成示例 topo -m 矩阵
python3 sim_topo.py sample > sample_topo.txt

# 解析 topo 矩阵
python3 sim_topo.py parse sample_topo.txt

# 运行单测
python3 test_pcie_gdr.py
```

## 期望输出示例

### PCIe 峰值（`python3 sim_pcie_bw.py peak`）

```
PCIe 代际     双向峰值    单向峰值
           GB/s        GB/s
Gen4 x16     64.0         32.0
Gen5 x16    128.0         64.0

PCIe Gen5 各宽度（单向）
x1        x2        x4        x8       x16
    4.0      8.0     16.0     32.0     64.0
```

### H2D 分解（`python3 sim_pcie_bw.py h2d --gen gen5 --width 16 --mps 256`）

```
H2D 拷贝带宽分解（PCIe Gen5 x16, MPS 256B）
==================================================
单向峰值（未编码）:    64.0 GB/s
编码（128b/130b）:    63.0 GB/s（-1.5%）
MPS 256B 有效率:      52.3 GB/s（-17.0%）
ACK 往返折扣：           47.1 GB/s（-10%）

预期 H2D 上限（pinned 内存）: 47.1 GB/s
实际范围（±6% 系统变异）: 44.2～45.2 GB/s
```

### 单测结果（`python3 test_pcie_gdr.py`）

```
✓ PCIe Gen5 x16 单向峰值: 64.0 GB/s [1]
✓ PCIe Gen4 x16 双向峰值: 64.0 GB/s [2]
✓ H100 H2D 有效带宽（模型）: 19.1 GB/s [3]
✓ Gen5 x16 H2D 有效带宽（模型）: 47.1 GB/s
✓ Gen5 64.0 > Gen4 32.0 GB/s
✓ 宽度单调性正确
✓ H2D 有效带宽 ≤ 单向峰值
✓ D2H ≤ H2D（往返延迟）
✓ 拓扑距离等级正确：[10, 20, 30, 40]
✓ 拓扑矩阵解析正确：8 设备，矩阵 8x8
✓ PCIe Gen5 128b/130b 编码效率: 98.5% [4]
✓ PCIe Gen4 8b/10b 编码效率: 80% [5]

============================================================
所有测试通过 ✓
============================================================
```

## 核心数据

**PCIe 单向峰值（未扣编码、MPS、ACK）**
- Gen4 x16：32 GB/s（64 GB/s 双向）
- Gen5 x16：64 GB/s（128 GB/s 双向）

**H2D 有效带宽（pinned 内存，MPS 256B）**
- Gen4 x16：~19 GB/s（47-49% 编码 + MPS + ACK 综合效率）
- Gen5 x16：~47 GB/s（74-75% 编码 + MPS + ACK 综合效率）
- 实测范围：±6% 系统变异

**可分页内存 H2D：约 pinned 的 45-55%（CPU 翻译与 bounce buffer 开销）**
- 加速倍数：1.8～2.2x

**编码效率**
- Gen4 8b/10b：80%
- Gen5 128b/130b：98.5%

**拓扑距离与 NCCL P2P/GPUDirect 路由**
- PXB（距离 10）：点对点，无 host bridge
- PHB（距离 20）及以上：经主存中转
