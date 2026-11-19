# 13-nvlink

GPU-13 NVLink 与 NVSwitch 配套代码。单机内部的四层互连与环形 all-reduce 的带宽模型。

## 文件清单

| 文件 | 环境 | 做什么 | 对应文档章节 |
|------|------|--------|-----------|
| `sim_nvlink.py` | CPU Python 3.10+ | NVLink 六代单链路与单卡带宽（单向/双向）、环形 all-reduce 的 algbw/busbw 与延迟项、topo 矩阵解析 | 第 2-7 章 |
| `sim_nccl_check.py` | CPU Python 3.10+ | 解析 nccl-tests 日志，对标理论值，判定 PASS/WARN/FAIL | 第 9 章 |
| `test_nvlink.py` | CPU Python 3.10+ | 单元测试：规格值、理论值、单调性、拓扑解析 | 验收 |
| `topo_sample_8gpu_nvswitch.txt` | 无 GPU | H100 HGX 样例：8 卡全互联 NVLink | 第 4 章 |
| `topo_sample_8gpu_hybrid.txt` | 无 GPU | 混合拓扑样例：前 4 卡 NVLink，后 4 卡通过 PCIe | 第 5 章 |
| `nccl_test_h100_8gpu_good.log` | 无 GPU | 合成样例（非实测）：全 NVLink，busbw 382.8 GB/s，约为 450 GB/s 天花板的 85% | 第 9 章 |
| `nccl_test_h100_8gpu_pcie_hybrid.log` | 无 GPU | 合成样例（非实测）：环上一跳退到 PCIe Gen5 x16，busbw 52.4 GB/s | 第 5、9 章 |

## 运行方式

```bash
python3 sim_nvlink.py specs                                   # per-link / per-GPU bandwidth, one-way and bidirectional
python3 sim_nvlink.py ring --gen nvlink4 --n 8                # ring all-reduce over NVSwitch
python3 sim_nvlink.py ring --n 8 --slowest pcie5 --alpha-us 5 --msg 1e8   # a PCIe hop in the ring
python3 sim_nvlink.py topo-parse topo_sample_8gpu_hybrid.txt
python3 sim_nccl_check.py threshold
python3 sim_nccl_check.py parse nccl_test_h100_8gpu_good.log
python3 test_nvlink.py
```

口径：
- 厂商标的 600 / 900 / 1800 GB/s 是单卡双向合计；算时间账一律用单向（A100 300、H100 450、B200 900 GB/s）。
- nccl-tests 的 busbw = algbw × 2(n−1)/n，直接和环上最慢一跳的单向带宽比；8 卡 H100 走 NVSwitch 时天花板是 450 GB/s，algbw 天花板是 450 × 8/14 ≈ 257 GB/s。
- PASS ≥ 80%、WARN 50%–80%、FAIL < 50% 是本仓库的约定阈值，不是厂商数字；上线前用自己的基线替换。

## 期望输出

以下输出来自本机真实运行（无 GPU）：

### specs

```
$ python3 sim_nvlink.py specs
gen       GPU      link/dir  links    GPU/dir  GPU bidir
--------------------------------------------------------
nvlink1   P100      20 GB/s      4    80 GB/s   160 GB/s
nvlink2   V100      25 GB/s      6   150 GB/s   300 GB/s
nvlink3   A100      25 GB/s     12   300 GB/s   600 GB/s
nvlink4   H100      25 GB/s     18   450 GB/s   900 GB/s
nvlink5   B200      50 GB/s     18   900 GB/s  1800 GB/s
nvlink6   Rubin         n/a    n/a  1800 GB/s  3600 GB/s

Vendor headline numbers (600 / 900 / 1800 GB/s) are bidirectional; use GPU/dir for budgets.
```

### ring（NVSwitch）

```
$ python3 sim_nvlink.py ring --gen nvlink4 --n 8
ring all-reduce, slowest hop: nvlink4 (450 GB/s per direction per GPU), alpha 0.0 us/step
 GPUs      size     algbw     busbw        time  busbw/B
----------------------------------------------------------
    2    1.0 GB     450.0     450.0     2222 us  100.0%
    4    1.0 GB     300.0     450.0     3333 us  100.0%
    8    1.0 GB     257.1     450.0     3889 us  100.0%
```

### ring（环上有一跳 PCIe Gen5，100 MB，每步 5 µs）

```
$ python3 sim_nvlink.py ring --n 8 --slowest pcie5 --alpha-us 5 --msg 1e8
ring all-reduce, slowest hop: pcie5, alpha 5.0 us/step
 GPUs      size     algbw     busbw        time  busbw/B
----------------------------------------------------------
    2  100.0 MB      62.6      62.6     1597 us   99.4%
    4  100.0 MB      41.5      62.2     2411 us   98.8%
    8  100.0 MB      35.1      61.5     2848 us   97.5%
```

### topo-parse（混合拓扑）

```
$ python3 sim_nvlink.py topo-parse topo_sample_8gpu_hybrid.txt
topo -m 矩阵解析：8 个 GPU

    距离 类型                    链路对数       占比
--------------------------------------------------
     1  NVLink                   24     42.9%
    10  PCIe (PIX/PXB)           32     57.1%

NVLink 链路对：24 / 56（42.9%）
PCIe 或更慢：32 / 56（57.1%）

平均每卡 NVLink 直连数：3.0
建议：
⚠ 部分 NVLink：24 对链路走 NVLink，其余经 PCIe/NUMA。
  环形 all-reduce 会按最慢的链路（PCIe）算带宽。
```

### threshold

```
$ python3 sim_nccl_check.py threshold
hop                ceiling    PASS >=    WARN >=
--------------------------------------------------
nvlink3           300 GB/s      240.0      150.0
nvlink4           450 GB/s      360.0      225.0
nvlink5           900 GB/s      720.0      450.0
pcie-gen5-x16      63 GB/s       50.4       31.5
```

### parse（全 NVLink 合成样例）

```
$ python3 sim_nccl_check.py parse nccl_test_h100_8gpu_good.log
NCCL-tests 日志：nccl_test_h100_8gpu_good.log

配置：
  algorithm: RING
  gpu_count: 8
  message_sizes: ['1073MB'] ...
  protocol: Simple

推断预期 busbw（nvlink4，8 卡）：450 GB/s

           size      algbw      busbw       效率       判定
-----------------------------------------------------------------
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS
              ?       218.8       382.8     85.1%      PASS

总结：
  结果：8 PASS，0 WARN，0 FAIL（共 8 次测试）
  busbw 范围：382.8 - 382.8 GB/s
  ✓ 所有 PASS：NVLink 配置正常
```

### parse（PCIe 一跳合成样例）

```
$ python3 sim_nccl_check.py parse nccl_test_h100_8gpu_pcie_hybrid.log
NCCL-tests 日志：nccl_test_h100_8gpu_pcie_hybrid.log

配置：
  algorithm: RING
  gpu_count: 8
  message_sizes: ['1073MB'] ...
  protocol: Simple

推断预期 busbw（nvlink4，8 卡）：450 GB/s

           size      algbw      busbw       效率       判定
-----------------------------------------------------------------
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL
              ?        29.9        52.4     11.6%      FAIL

总结：
  结果：0 PASS，0 WARN，8 FAIL（共 8 次测试）
  busbw 范围：52.4 - 52.4 GB/s
  ✗ 有 FAIL：建议检查拓扑（topo -m）和 Fabric Manager 配置
```

### 单元测试

```
$ python3 test_nvlink.py
✓ NVLink4 规格值 (H100)
✓ NVLink5 规格值 (B200)
✓ Ring all-reduce 8 卡理论值 (H100)
✓ Ring 性能随 GPU 数单调增
✓ Ring 延迟项
✓ 不同代数可扩展性
✓ Topo 矩阵解析
✓ NCCL 日志解析
✓ NCCL 结果判定
✓ 示例日志自洽
✓ 消息大小 vs busbw

测试结果：11 通过，0 失败（共 11 项）
```
