# zero-to-gpu

“从 0 开始 GPU 优化”飞书系列的配套代码。每章一个目录，目录名和文档章节号一一对应，索引页是 GPU-00。

## 版本基线

- CUDA 13.4 Update 2，驱动分支 R615；13.x 应用可在 R580 及以上驱动运行，13.4 新特性要求 R615
- PyTorch 2.15.0（默认 wheel 基于 CUDA 13.2），NCCL 2.32.3，CUTLASS 4.8.1，Nsight Compute 2026.3.1
- CUDA 13 最低支持 Turing（sm_75），编译时按自己的卡改 `-arch`
- CPU 练习只依赖 Python 3.10 及以上和 numpy

## 文件命名约定

| 前缀或后缀 | 含义 | 运行环境 |
|---|---|---|
| `sim_*.py` | 模拟器或计算器，复现硬件和系统的关键账本 | CPU |
| `gpu_*.py` | 需要 GPU 的 Python 实验 | GPU |
| `*.cu` | CUDA 源文件，`nvcc -O3 -arch=sm_XX` 编译 | GPU |
| `*.sh` | 启动、压测或验收脚本 | GPU |
| `test_*.py` | 模拟器的单元测试，`python3 -m pytest` 或直接运行 | CPU |

## 目录

| 目录 | 章节 | 内容 |
|---|---|---|
| `common/` | 全系列 | GPU 规格表 |
| `01-arch/` | GPU-01 GPU 架构 | 设备属性读取、SM/warp/occupancy 计算器（CPU）、显存与共享内存微基准（GPU） |
| `02-ledger/` | GPU-02 性能账本 | Roofline 与 MFU 计算器：算术强度、拐点、kernel 理论上限、训练 MFU（CPU），瓶颈与融合实验（GPU） |
| `03-cuda/` | GPU-03 CUDA 编程模型 | vec_add、stream 与 event、PTX/SASS 对照 |
| `04-memory/` | GPU-04 访存与归约 | 合并访存、bank conflict、warp shuffle 归约的对比 |
| `05-gemm/` | GPU-05 GEMM 逐步优化 | 从朴素到寄存器分块的逐步 GEMM，CPU 上验证分块正确性 |
| `06-tensor-core/` | GPU-06 Tensor Core 与低精度 | mma/wgmma 示例、FP8 与 MX 缩放模拟（CPU） |
| `07-dsl/` | GPU-07 Triton、CUTLASS 与 CuTe DSL | 同一个 kernel 的三种写法 |
| `08-attention/` | GPU-08 注意力内核 | 分块注意力与在线 softmax 的 CPU 参考实现、HBM 读写账本 |
| `09-driver/` | GPU-09 驱动、CUDA 兼容与 Xid | 版本兼容检查脚本、Xid 速查 |
| `10-memory-mgmt/` | GPU-10 显存管理 | caching allocator 模拟器、memory snapshot 分析 |
| `11-pcie-gdr/` | GPU-11 PCIe、NUMA 与 GPUDirect | PCIe 带宽账、拓扑解析、IOMMU 检查 |
| `12-sharing/` | GPU-12 MIG、MPS 与容器 | MIG 切分方案计算器 |
| `13-nvlink/` | GPU-13 NVLink 与 NVSwitch | 单机 all-reduce 带宽账、nccl-tests 结果解析 |
| `14-profiling/` | GPU-14 Profiling | nsys/ncu 抓取脚本、报告解析 |
| `15-runtime/` | GPU-15 PyTorch 运行时 | launch 开销与 CUDA Graph 收益模拟、torch.compile 对比 |
| `16-node-burnin/` | GPU-16 单机验收与调优 | 验收脚本与合格线 |
| `17-rdma/` | GPU-17 RDMA 网络 | 拓扑与带宽计算器、ECMP 哈希极化模拟 |
| `18-nccl/` | GPU-18 NCCL | ring/tree 通信量与延迟模型、协议选择模拟 |
| `19-k8s/` | GPU-19 Kubernetes 上的 GPU | 拓扑感知调度模拟 |
| `20-reliability/` | GPU-20 可靠性与可观测 | 故障注入模拟、checkpoint 间隔计算 |
| `21-train-parallel/` | GPU-21 训练并行 | TP/PP/DP/ZeRO 显存与通信账本、并行方案搜索 |
| `22-moe-ep/` | GPU-22 MoE 与专家并行 | all-to-all 通信量与负载不均模拟 |
| `23-inference/` | GPU-23 推理系统 | KV cache 账本、PD 配比计算器 |
| `24-capstone/` | GPU-24 综合：从选卡到上线 | 选卡、并行、吞吐与成本的综合计算器 |

24 个目录都已填满：每个目录都有 `sim_*.py` 模拟器和 `test_*.py` 单元测试，在只有 CPU 的机器上就能跑完；需要真卡的脚本以 `gpu_` 开头或是 `.cu`、`.sh` 文件，文件头写明依赖版本。每个目录的 README 给出运行命令和本机实跑的期望输出。

全系列共用一条主线：GPU 优化就是把等待一层层消掉。每篇处理时间账 T = T_compute + W_mem + W_host + W_comm + W_idle 里的一项，配套代码负责把这一项的天花板、实际值和消除前后的差算出来。

## 快速试跑

一次跑完全部单元测试：

```bash
cd zero-to-gpu
for d in [0-2][0-9]-*; do (cd $d && for t in test_*.py; do python3 $t > /dev/null && echo "ok   $d/$t" || echo "FAIL $d/$t"; done); done
```


```bash
cd zero-to-gpu
python3 common/specs.py          # 打印各代 GPU 规格和 BF16 拐点
```
