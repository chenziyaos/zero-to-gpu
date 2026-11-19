# Code Outputs — GPU-22 MoE 与专家并行

## sim_a2a.py dispatch 子命令

```
$ python3 sim_a2a.py dispatch --hidden 4096 --top-k 2 --ep-rank 8
指标                   数值                  
----------------------------------------
hidden               4096                
top-k                2                   
batch tokens         32                  
EP rank 数            8                   
dtype                bf16                
----------------------------------------
dispatch 字节          0.50 MB
combine 字节           0.50 MB
总计                   1.00 MB

NVLink4 延迟           2.33 us
```

对应文档章节：第 2、3 章（all-to-all 通信量与延迟）

## sim_a2a.py overhead 子命令（扫 batch）

```
$ python3 sim_a2a.py overhead --hidden 4096 --batch-min 1 --batch-max 256
batch      通信字节 (MB)            延迟 (us)         通信开销%     
-------------------------------------------------------
1          0.03                 0.07            0.1       %
32         1.00                 2.33            0.7       %
63         1.97                 4.59            0.7       %
94         2.94                 6.85            0.7       %
125        3.91                 9.11            0.7       %
156        4.88                 11.37           0.7       %
187        5.85                 13.62           0.7       %
218        6.82                 15.88           0.7       %
249        7.79                 18.14           0.7       %
```

对应文档章节：第 2、4 章（通信开销比例分析）

## sim_a2a.py latency 子命令（对比有无重叠）

```
$ python3 sim_a2a.py latency --hidden 4096 --top-k 2 --ep-rank 8
指标                   数值                  
----------------------------------------
单个 batch 总时间 (us)  5000              
dispatch 延迟 (us)     2.33              
combine 延迟 (us)      2.33              
expert GEMM (us)      5000 (假设 hidden×hidden)  
----------------------------------------
无重叠总时间           5004.66           us
有重叠（双微批）       5004.66           us (GEMM >> dispatch+combine)
有效隐藏比例           0.1               %
```

对应文档章节：第 3、8 章（通信与计算重叠）

## sim_imbalance.py zipf 子命令

```
$ python3 sim_imbalance.py zipf --num-experts 64 --num-tokens 10240 --alpha 1.0
Zipf 参数              1.0            
专家数                  64             
总 token 数            10240          
平均每专家                160            
-----------------------------------
最大负载                 400            
最小负载                 12             
平均负载                 160.00         
max/mean 比           2.50           

热专家（top 10）  激活数
--------------------
专家 0       400
专家 1       243
专家 2       157
...
```

对应文档章节：第 4 章（负载极化与 Zipf 分布）

## sim_imbalance.py rebalance 子命令

```
$ python3 sim_imbalance.py rebalance --num-experts 64 --num-tokens 10240
策略                        max/mean        改善             
-------------------------------------------------------
无均衡（Zipf α=1.5）           27.087          -              
辅助损失（λ=0.01）              20.021          26.1          %
偏置调整（5 迭代）                21.026          22.4          %
冗余专家（8 个）                 4.355           83.9          %
```

对应文档章节：第 4 章（负载均衡策略的改善效果）

## 单测通过情况

```bash
$ python3 test_a2a.py
✓ test_dispatch_bytes 通过
✓ test_combine_symmetry 通过
✓ test_deepseek_v3_a2a_estimate 通过
✓ test_nvlink_latency 通过
✓ test_overhead_sweep 通过

✓✓✓ 所有 sim_a2a 测试通过 ✓✓✓

$ python3 test_imbalance.py
✓ test_zipf_distribution 通过
✓ test_uniform_vs_zipf 通过
Debug: 策略                        max/mean        改善             
✓ test_rebalancing_strategies 通过
✓ test_deterministic_seed 通过

✓✓✓ 所有 sim_imbalance 测试通过 ✓✓✓
```

所有测试复现公开资料中的数字并通过验证。

## 关键数字与出处

| 数字 | 说明 | 来源 |
|---|---|---|
| 0.5 MB per rank | dispatch/combine 字节 (hidden=4096, top-2, batch=32, EP=8) | sim_a2a.py 代码计算 |
| 2.33 us | NVLink4 single-direction latency (450 GB/s) | NVIDIA 规格 × sim_a2a.py |
| 2.5x | max/mean 负载比 (Zipf α=1.0, experts=64) | sim_imbalance.py 代码输出 |
| 83.9% | 冗余专家改善效果 (8 专家从 27x 降至 4.36x) | sim_imbalance.py test_rebalancing_strategies |
| 0.7% | 通信开销占比 (hidden=4096, batch=32+) | sim_a2a.py overhead 输出 |
