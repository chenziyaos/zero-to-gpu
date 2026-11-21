# GPU-16 · 单机验收与调优 · 配套代码

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/16-node-burnin.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-16](../book/16-node-burnin.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

单机验收把 H100/A100/B200 八卡机器从清点、固件、静态体检、微基准、互联通信、长测六层漏斗拆开，每层落盘一份观测 JSON 后汇总给判定引擎，输出 PASS（0）或 FAIL（1）的整机结论。六条合格线用比例口径（九成、八成五、八成、零容忍）跨代稳定，从判定、分析到基线沉淀一体化。

## 文件清单

| 文件 | 环境 | 做什么 | 对应正文 |
|---|---|---|---|
| `sim_burnin.py` | 无 GPU，Python 3.10+ 标准库 | 读观测 JSON（compute/memory/links/pcie/xids/burnin），按六条合格线判决，输出 PASS/CONDITIONAL/FAIL | GPU-16 第 2 章 |
| `make_samples.py` | 无 GPU | 生成四份样例观测 JSON：健康机、单卡故障、链路+Xid、温度+短测 | 第 1 章 1.3 节 |
| `test_burnin.py` | 无 GPU | 17 个用例覆盖六条合格线边界、卡间极差、Xid 分类、聚合逻辑 | 第 2、6 章 |
| `samples/` | 无 GPU | 四份演示用 JSON 文件 | 第 1 章 |

## 运行命令

### 生成样例
```bash
cd repo/zero-to-gpu/16-node-burnin
python3 make_samples.py
# 输出：samples/healthy_pass.json, fail_gpu5_slow.json, fail_link_xid.json, warn_temp_short_burn.json
```

### 判决单份观测
```bash
python3 sim_burnin.py report samples/healthy_pass.json
python3 sim_burnin.py report samples/fail_gpu5_slow.json  # exit 1
```

### 运行全部测试
```bash
python3 test_burnin.py
# 输出：17/17 tests passed
```

## 期望输出

### 健康机
```
$ python3 sim_burnin.py report samples/healthy_pass.json
机型 H100-SXM  结论 PASS  FAIL 0  WARN 0
  [PASS       ] BF16 TFLOPS 卡间极差     实测 1.5%                 要求 <= 5.0%
```

### 单卡故障
```
$ python3 sim_burnin.py report samples/fail_gpu5_slow.json; echo "exit=$?"
机型 H100-SXM  结论 FAIL  FAIL 2  WARN 0
  [FAIL       ] GPU5 BF16 TFLOPS     实测 850.0                要求 >= 890
  [FAIL       ] BF16 TFLOPS 卡间极差     实测 7.5%                 要求 <= 5.0%
exit=1
```

### 互联+致命 Xid
```
$ python3 sim_burnin.py report samples/fail_link_xid.json
机型 H100-SXM  结论 FAIL  FAIL 3  WARN 0
  [PASS       ] BF16 TFLOPS 卡间极差     实测 1.5%                 要求 <= 5.0%
  [FAIL       ] GPU2 NVLink P2P 单向 实测 320.0 GB/s            要求 >= 360 GB/s
  [FAIL       ] all-reduce busbw（单向容纳线） 实测 300.0 GB/s 要求 >= 360 GB/s
  [FAIL       ] 致命 Xid               实测 [79]                 要求 0
```

## 核心合格线（H100-SXM）

- **计算**：BF16 GEMM ≥ 890 TF（标称 989 TF 的九成）+ 卡间极差 ≤ 5%
- **显存**：HBM 带宽 ≥ 2.85 TB/s（标称 3.35 TB/s 的八成五）+ 容量 ≥ 78.04 GB（98%）+ ECC 双位零容忍 + 温度 < 83°C（WARN 78-83）
- **互联**：NVLink P2P 单向（p2pBandwidthLatencyTest 或 nvbandwidth 测量）PASS ≥ 360 GB/s (0.80 × 450) / WARN 337.5-360 / FAIL < 337.5；all-reduce busbw PASS ≥ 360 GB/s / WARN 252-360 / FAIL < 252（NVLS 可推高作为上界）
- **PCIe**：Gen5 x16 无降速降宽 + 带宽 ≥ 50.4 GB/s
- **错误**：致命 Xid（48/64/79/95/119/120，需 reset/RMA）零容忍 + 告警 Xid（13/31/43/45/63/74/92/94，app-level）记录
- **burn-in**：≥ 24 小时 + 掉卡/ECC双位/降频/NCCL失败四个计数全零

## 整机聚合规则

- 任一 FAIL → 整机 FAIL（拒收）
- 无 FAIL 但有 WARN → CONDITIONAL（附条件进 burn-in）
- 全 PASS → PASS（放行）

## 与系列其他文档的分工

单机到此交付完毕。从 GPU-17 开始进入集群层：NVLink 与 NVSwitch 后续由 GPU-13 讲跨机带宽，GPU-18 讲 NCCL 把带宽用满，GPU-19 讲 k8s 调度把任务送对位置。故障与老化的长周期监控见 GPU-20 可靠性与可观测。
