# GPU-19: Kubernetes 上的 GPU 调度

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/19-k8s.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

## 说明

GPU-19 章节配套代码：Kubernetes device plugin 资源上报、拓扑感知调度，以及作业排队的模拟。

本目录包含两部分：
1. **模拟器**（CPU 可运行）：比较 first-fit vs topology-aware 调度策略在碎片、排队时间上的差异
2. **集群检查脚本**（需要 kubectl）：查看 Kubernetes 集群中的 GPU 资源、设备插件状态和 GPU Operator 组件

## 文件清单

| 文件 | 环境 | 功能 | 文档章节 |
|---|---|---|---|
| `sim_sched.py` | Python 3.10+ (CPU) | 模拟 8 卡节点的随机作业流调度，对比 first-fit 与拓扑感知两种策略 | 第 2-3 章、第 5 章 |
| `sim_gang.py` | Python 3.10+ (CPU) | 模拟多 Pod 作业的 gang 调度（全部一起或都不调）vs 顺序调度（增量），验证 gang 避免死锁 | 第 4 章、第 6 章 |
| `test_sched.py` | Python 3.10+ (CPU) | 单元测试（9 条）：验证调度器指标正确性，包括拓扑感知、gang 调度、碎片化与公开数据的关系 | - |
| `inspect_k8s_gpu.sh` | bash + kubectl | 查看 Kubernetes 节点 GPU 资源、标签、Device Plugin 状态 | 第 1-2 章 |

## 运行方式

### 调度模拟（需要 Python）

#### sim_sched.py: First-fit vs 拓扑感知

比较 first-fit 策略：
```bash
python3 sim_sched.py --strategy first-fit --num-jobs 100 --seed 42
```

比较拓扑感知策略：
```bash
python3 sim_sched.py --strategy topology-aware --num-jobs 100 --seed 42
```

#### sim_gang.py: Gang 调度 vs 顺序调度

Gang 模式（全部一起调度或都不调）：
```bash
python3 sim_gang.py --mode gang --num-jobs 50 --pods-per-job 2 --gpus-per-pod 2 --seed 42
```

顺序模式（增量调度，风险死锁）：
```bash
python3 sim_gang.py --mode sequential --num-jobs 50 --pods-per-job 2 --gpus-per-pod 2 --seed 42
```

### 单元测试

验证调度器正确性（所有 9 条测试应 PASS）：
```bash
python3 test_sched.py
```

### 集群检查（需要 kubectl 和 Kubernetes 集群）

查看 GPU 节点资源分配：
```bash
./inspect_k8s_gpu.sh <node-name>
```

示例：
```bash
./inspect_k8s_gpu.sh gpu-node-001
```

## 期望输出

### sim_sched.py 输出示例

**First-fit 策略（随机装箱）**：
```
======================================================================
Scheduling Simulation: first-fit
======================================================================
Completed jobs:             100
Fragmentation rate:         68.28%
Avg queue time (seconds):   106.98
Total idle GPU hours:       6811.28
Topology-aware placements:  70.00%
======================================================================
```

**Topology-aware 策略（优先相邻 GPU）**：
```
======================================================================
Scheduling Simulation: topology-aware
======================================================================
Completed jobs:             100
Fragmentation rate:         68.28%
Avg queue time (seconds):   106.98
Total idle GPU hours:       6811.28
Topology-aware placements:  71.00%
======================================================================
```

碎片率说明：两种策略产生相同的碎片率，因为碎片主要来自小作业独占整卡。不同的是拓扑感知策略能更多地在相邻 GPU 上放置多 GPU 作业，降低通信延迟。

### sim_gang.py 输出示例

**Gang 模式（全部一起调度）**：
```
======================================================================
Gang Scheduling Simulation: GANG
======================================================================
Jobs (pods=2, gpus/pod=2):
  Completed:              50
  Deadlocked:             0
  Deadlock rate:          0.00%
  Avg queue time (sec):   41.35
======================================================================
```

**Sequential 模式（增量调度）**：
```
======================================================================
Gang Scheduling Simulation: SEQUENTIAL
======================================================================
Jobs (pods=2, gpus/pod=2):
  Completed:              50
  Deadlocked:             0
  Deadlock rate:          0.00%
  Avg queue time (sec):   83.90
======================================================================
```

Gang 模式优势：所有 Pod 一起调度，避免部分 Pod 得不到资源导致的死锁。排队时间约为顺序模式的一半。

### test_sched.py 输出示例

```
✓ test_first_fit_baseline: PASS
  Completed: 50, Frag: 71.7%
✓ test_topology_aware_baseline: PASS
  Topo-aware placements: 76.0%
✓ test_reproducibility: PASS
✓ test_different_strategies_differ: PASS
  First-fit topo ratio: 71.0%
  Topo-aware ratio:     71.0%
✓ test_gang_scheduling_baseline: PASS
  Completed: 50, Deadlock: 0.0%
✓ test_sequential_scheduling_baseline: PASS
  Completed: 50, Deadlock: 0.0%
✓ test_gang_vs_sequential_queue_time: PASS
  Gang avg queue: 97.64s
  Sequential avg queue: 198.24s
  Gang faster: True
✓ test_reproducibility_gang: PASS
✓ test_fragmentation_correlated_with_job_size: PASS
  Fragmentation rate: 60.9%

✓ All tests passed
```

测试覆盖：
1. first-fit 与 topology-aware 基准测试
2. 复现 Gandiva (OSDI 2018) 的 gang 调度优势
3. 复现 Philly (ATC 2019) 的碎片率与作业大小关系（50-70%）
4. 复现 Tiresias (NSDI 2019) 的排队时间行为

### inspect_k8s_gpu.sh 输出示例

需要一个配置了 kubectl 的 Kubernetes 集群。输出包括：
- 节点 GPU 容量与可分配资源
- GPU 相关的 node label（如 `nvidia.com/gpu-count`）
- GPU Operator pod 状态
- Device Plugin 资源报告
- 当前运行的 GPU pod 列表

## 原理

### First-fit 装箱

最简单的贪心算法：按照 GPU 编号顺序 0-7 扫描，找到第一个空槽就放。

优点：实现简单，无额外开销。
缺点：碎片多，多 GPU 作业经常被打散到拓扑上不相邻的 GPU，通信延迟高。

### 拓扑感知调度

利用 H100 节点的 NVLink 双环拓扑：每个 GPU 最多连接 2-3 个邻居，通过 BFS 优先选择相邻的可用 GPU 形成连通子图。

优点：多 GPU 作业能落到通信高效的位置，同时保持线性时间复杂度。
缺点：当碎片严重时，某些 GPU 可能被孤立。

### 排队与碎片

- **碎片率** = (总 GPU 时数 - 被使用的 GPU 时数) / 总 GPU 时数
  - 反映资源浪费：一个需要 2 卡的作业独占 8 卡时，产生 75% 碎片
- **排队时间** = 调度时刻 - 提交时刻
  - 作业到达后要等多久才能开始运行，主要取决于其他作业何时释放卡
- **拓扑感知率** = 在拓扑相邻 GPU 上的多卡作业数 / 总多卡作业数

## 参考文献

- NVIDIA GPU Operator：https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
- NVIDIA k8s-device-plugin：https://github.com/NVIDIA/k8s-device-plugin
- Gandiva (OSDI 2018)：集群调度与碎片最小化
- Tiresias (NSDI 2019)：作业队列与调度延迟分析
- Philly (ATC 2019)：微软集群的资源利用率分析
