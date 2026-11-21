# 22-moe-ep

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/22-moe-ep.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-22](../book/22-moe-ep.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-22 MoE 与专家并行（Expert Parallelism）的配套练习。

## 文件清单

| 文件 | 用途 | 运行环境 | 对应章节 |
|------|------|---------|---------|
| `sim_a2a.py` | All-to-All 通信量与延迟计算器 | CPU（Python 3.10+） | 第 2-4 章（通信分析） |
| `sim_imbalance.py` | 专家负载不均与均衡策略模拟 | CPU（Python 3.10+） | 第 5-7 章（负载均衡） |
| `test_a2a.py` | sim_a2a.py 的单测 | CPU（Python 3.10+） | 验证通信公式 |
| `test_imbalance.py` | sim_imbalance.py 的单测 | CPU（Python 3.10+） | 验证 Zipf 分布和均衡策略 |

## 使用方式

### All-to-All 通信

计算 MoE 的 dispatch 和 combine 阶段的通信字节数与延迟。

```bash
# 基础示例：计算 dispatch/combine 字节
python3 sim_a2a.py dispatch --hidden 4096 --top-k 2 --batch 32 --ep-rank 8

# 扫参：看 batch 增长时通信开销占比
python3 sim_a2a.py overhead --hidden 4096 --batch-min 1 --batch-max 128 --ep-rank 8

# 多节点路由：DeepSeek-V3 风格节点受限
python3 sim_a2a.py noderoute --hidden 4096 --batch 64 --gpus-per-node 8 --num-nodes 4

# 估延迟：单节点 vs 多节点
python3 sim_a2a.py latency --hidden 4096 --top-k 2 --batch 32 --ep-rank 8
```

### 专家负载不均

模拟 Zipf 分布下的专家激活偏斜和各种均衡策略的改善。

```bash
# Zipf 分布的负载分析
python3 sim_imbalance.py zipf --num-experts 128 --num-tokens 10240 --alpha 1.5

# 各均衡策略对比：辅助损失、偏置、冗余专家
python3 sim_imbalance.py rebalance --num-experts 128 --num-tokens 10240

# 负载不均统计
python3 sim_imbalance.py imbalance --distribution zipf --alpha 1.5 --num-experts 128
```

## 期望输出

### dispatch 示例

```
指标                   数值
----------------------------------------
hidden               4096
top-k                2
batch tokens         32
EP rank 数            8
----------------------------------------
dispatch 字节          0.50 MB
combine 字节           0.50 MB
总计                   1.00 MB

NVLink4 延迟           0.00 us
```

### zipf 示例

```
Zipf 参数              1.5
专家数                  128
总 token 数            10240
平均每专家                80
-----------------------------------
最大负载                 4193
最小负载                 0
平均负载                 80.00
max/mean 比           52.41

热专家（top 10）  激活数
--------------------
专家 0       4193
专家 1       1473
...
```

## 单测

```bash
python3 test_a2a.py
python3 test_imbalance.py
```

两个测试都通过后显示 `✓✓✓ 所有测试通过 ✓✓✓`。

## 依赖

- Python 3.10+（标准库，无额外包）
- `common/specs.py`（共用 GPU 规格表）
