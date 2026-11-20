# GPU-09 驱动、CUDA 兼容与 Xid

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/09-driver.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

驱动版本与 CUDA 版本的兼容性检查、Xid 错误诊断、GPU 节点环境验证。

## 文件清单

| 文件 | 环境 | 做什么 | 对应文档章节 |
|------|------|-------|-----------|
| `sim_compat.py` | 本地 Python 3.10+ | 检查驱动与 libcudart 版本兼容性，判断是否需要 cuda-compat 包 | 第 3-7 章：兼容模式 |
| `sim_xid.py` | 本地 Python 3.10+ | 解析 dmesg 中的 Xid 错误代码，给出诊断和建议动作 | 第 9-10 章：Xid 分诊 |
| `test_driver.py` | 本地 Python 3.10+ | 11 条单元测试：兼容性逻辑、Xid 解析、代码库完整性 | 测试基础设施 |
| `check_node.sh` | GPU 节点（需要 root） | 检查节点驱动版本、GPU 卡信息、显存、温度、dmesg 中的错误 | 第 8、11 章：上线前核对 |

## 运行方式

### 1. CUDA 版本兼容检查

```bash
# 精确版本匹配（无兼容性问题）
python3 sim_compat.py check --driver 550 --cudart 12.4

# 跨大版本，使用 cuda-compat 包
python3 sim_compat.py check --driver 550 --cudart 13.0 --compat-pkg 13.0-2

# 常见版本组合矩阵
python3 sim_compat.py jumps

# Jetson 卡特殊规则
python3 sim_compat.py jetson --jetpack 6.0 --container-cuda 12.2
```

### 2. Xid 错误诊断

```bash
# 查看 Xid 错误代码目录
python3 sim_xid.py catalog

# 过滤特定分类（如 memory、thermal、compute）
python3 sim_xid.py catalog --filter memory

# 生成示例诊断（用于测试）
python3 sim_xid.py simulate --code 5 --gpu-id 0

# 从日志文件解析 Xid
python3 sim_xid.py parse --log /var/log/kern.log

# 从管道读取日志
grep Xid /var/log/messages | python3 sim_xid.py parse --stdin
```

### 3. 单元测试

```bash
python3 test_driver.py
```

输出 11 条测试结果，验证兼容性逻辑、Xid 解析、代码库完整性。测试源引用官方 NVIDIA 文档和实际故障案例。

### 4. 节点环境检查（生产机器）

```bash
# 快速检查：驱动版本、GPU 卡数、显存、温度
./check_node.sh

# 完整诊断：包括 dmesg 中的错误和 ECC 统计（需要 root）
sudo ./check_node.sh --full
```

## 期望输出示例

### sim_compat.py：后向兼容场景

```
驱动 R550 + libcudart 12.0： ✓ backward
  理由：libcudart 12.0 低于驱动支持的最高版本 12.4，后向兼容
  实际 CUDA 版本：12.0
  限制：
    · 没有 PTX JIT，只能运行 SASS 代码
    · 个别新接口可能返回 cudaErrorCallRequiresNewerDriver
    · 部分 Hopper 新特性不可用
```

### sim_xid.py：显存错误诊断

```
GPU #0 Xid 5：GR: Graphics Exception on SM
═══════════════════════════════════════
分类：memory  严重程度：critical

含义：
  Graphics RAM（DRAM）访问故障，可能是显存故障或驱动映射错误

常见原因：
  · 显存物理故障（坏块）
  · 驱动显存管理 bug，错误地分配地址
  · 超频导致的访存失败

建议的第一步：
  1. 立即下线该 GPU，进行显存诊断
  2. 跑 CUDA memtest 排查物理故障
  3. 如持续触发，RMA 显卡
```

### test_driver.py：测试通过

```
运行 GPU-09 驱动/CUDA 兼容性测试...

✓ 测试 1：后向兼容
✓ 测试 2：精确匹配
✓ 测试 3：同大版本前向兼容
✓ 测试 4：需要 compat 包
✓ 测试 5：使用 cuda-compat 包
✓ 测试 6：版本单调性
✓ 测试 7：Xid 日志解析
✓ 测试 8：Xid 代码库覆盖
✓ 测试 9：Xid 5 诊断
✓ 测试 10：Xid 24 诊断
✓ 测试 11：未知 Xid 代码

═══════════════════════════════════════
全部 11 条测试通过 ✓
═══════════════════════════════════════
```

## 核心数字

- **R550 驱动支持最高 CUDA 12.4**（基于 2026-11-02 官方文档）
- **后向兼容无 PTX JIT**：同大版本内向下兼容，但失去即时 JIT 编译能力
- **cuda-compat 包仅支持数据中心卡**：不支持消费级卡（RTX 系列）
- **常见 Xid 代码 15 个**：5（显存）、24（超时）、29（过温）、48（ECC 双比特）等
- **默认 kernel 超时 5 秒**：H100 的看门狗计时器（Xid 24）

## 参考文献

- NVIDIA CUDA Compatibility：https://docs.nvidia.com/deploy/cuda-compatibility/
- NVIDIA CUDA Toolkit Release Notes：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- NVIDIA Tesla Driver Architecture：https://docs.nvidia.com/datacenter/tesla/drivers/
