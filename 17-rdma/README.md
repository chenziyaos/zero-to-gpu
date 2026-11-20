# 17-rdma

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/17-rdma.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

GPU-17 RDMA 网络的配套练习。先用 CPU 计算器把跨机这一跳的带宽账算清：轨道优化网络里每卡能拿多少、收敛比和 PXN 怎样改变 all-to-all；再看 ECMP 哈希碰撞和 incast 下 PFC 暂停各让通信等多久；最后在真机上用 perftest 和 GID 检查确认链路本身达标。

| 文件 | 环境 | 做什么 |
|---|---|---|
| `sim_fabric.py` | CPU | 轨道优化 leaf-spine 的 leaf 布局、二分带宽，轨道对齐 ring 与 all-to-all 的每卡可用带宽，扫收敛比 |
| `sim_ecmp.py` | CPU | N 条大象流哈希到 M 条上行时的碰撞与每流有效带宽（固定随机种子） |
| `sim_pfc.py` | CPU | incast 下 PFC 暂停占比、受害流吞吐与 DCQCN 对照，PAUSE 帧最长暂停时间 |
| `test_fabric.py` `test_pfc.py` `test_ecmp.py` | CPU | 单元测试 |
| `gpu_perftest_bw.sh` | 两台 RDMA 机器 | ib_write_bw 两机带宽测试 |
| `show_gids.sh` | RDMA 机器 | 列出各端口 GID、RoCE 版本与链路状态 |

## 运行

```bash
cd zero-to-gpu/17-rdma
python3 test_fabric.py && python3 test_pfc.py && python3 test_ecmp.py
python3 sim_fabric.py fabric --nodes 32
python3 sim_fabric.py sweep --nodes 128
python3 sim_fabric.py sweep --nodes 128 --no-pxn
python3 sim_ecmp.py --flows 16 --uplinks 8
python3 sim_pfc.py incast
python3 sim_pfc.py quanta
```

## 期望输出

- `sim_fabric.py fabric --nodes 32`：64 口 leaf、1:1 时每条轨道 1 台 leaf，ring 和 all-to-all 每卡都是 50.0 GB/s。
- `sim_fabric.py sweep --nodes 128`：收敛比 1:1、2:1、4:1 下轨道对齐 ring 都是 50.0 GB/s，all-to-all 分别为 50.0、36.9、20.6 GB/s；关掉 PXN 后 2:1 和 4:1 降到 26.1 和 13.1 GB/s。
- `sim_ecmp.py --flows 16 --uplinks 8`：理想每流 25.00 GB/s，典型 11.96 GB/s，效率 47.8%；每流发 1 GB 理想 40.0 ms、典型 83.63 ms。
- `sim_pfc.py incast`：400 Gb/s、Xoff 512 KB 时，N=8 每发送端 6.2 GB/s、上游端口暂停 88%，受害流在 PFC 下只剩 6.2 GB/s，DCQCN 下 43.8 GB/s。
- `sim_pfc.py quanta`：400 Gb/s 时单个 PAUSE 帧最长暂停 83.9 us。

以上数字均为本机实跑结果。
