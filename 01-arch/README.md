# 01-arch

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/01-arch.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

GPU-01 GPU 架构的配套练习。先用 CPU 计算器把 occupancy、在途字节和 wave 量化这三本账算清楚，再到真卡上跑微基准，拿实测数和论文数对照。

| 文件 | 环境 | 做什么 | 对应文档章节 |
|---|---|---|---|
| `sim_occupancy.py` | CPU | 按 CC 7.0 至 12.0 的资源上限算每 SM 驻留块数、warp 数和瓶颈 | 第 7 章 |
| `test_occupancy.py` | CPU | 复现 Best Practices Guide 的 37 寄存器例子等 5 个用例 | 第 7 章 |
| `sim_inflight.py` | CPU | Little 定律：吃满 HBM 每 SM 需要多少字节在路上 | 第 8 章 |
| `sim_wave.py` | CPU | wave 量化：块数不是 SM 数整数倍时的利用率上限 | 第 2 章 |
| `gpu_props.py` | GPU | 打印本机 SM 数、寄存器、共享内存、L2 | 第 1 章 |
| `gpu_occupancy.cu` | GPU | 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 和计算器对照 | 第 7 章 |
| `gpu_pchase.cu` | GPU | P-chase 测 L1、L2、HBM 延迟曲线 | 第 9 章 |
| `gpu_bandwidth.cu` | GPU | copy 和 read 带宽，标量和 float4、不同块数对比 | 第 8、9 章 |

## 运行

```bash
cd zero-to-gpu/01-arch
python3 test_occupancy.py
python3 sim_occupancy.py --cc 9.0 --threads 256 --regs 64 --smem 48
python3 sim_occupancy.py --cc 9.0 --sweep-regs --threads 128
python3 sim_inflight.py --gpu h100-sxm
python3 sim_wave.py

# 下面需要 GPU，-arch 按自己的卡改：A100 sm_80，H100 sm_90，B200 sm_100
nvcc -O3 -arch=sm_90 -Xptxas -v -o occ gpu_occupancy.cu && ./occ
nvcc -O3 -arch=sm_90 -o pchase gpu_pchase.cu && ./pchase > pchase.txt
nvcc -O3 -arch=sm_90 -o bw gpu_bandwidth.cu && ./bw
```

## 期望输出

`sim_occupancy.py --cc 9.0 --threads 256 --regs 64 --smem 48`：

```text
  按 warps      限制最多   8 块
  按 blocks     限制最多  32 块
  按 registers  限制最多   4 块
  按 shared_mem 限制最多   4 块
  驻留 4 块 = 32 warp，理论 occupancy 50.0%，瓶颈 registers、shared_mem
```

`sim_inflight.py --gpu h100-sxm`（延迟 480 周期、时钟 1.75 GHz 的粗估）：每 SM 约 6.8 KB 在路上，标量 float 加载要占满约 85% 的线程，float4 只要约 21%。

`sim_wave.py`：A100 108 SM、256x128 分块、M=2304 时，N=1536 一波 100%，N=1664 两波、利用率上限 54.2%。

真卡上的对照数（周期，来自 Luo et al., arXiv 2402.13499 的 P-chase 实测）：

| 层级 | A100 PCIe | H800 PCIe | RTX 4090 |
|---|---|---|---|
| 共享内存 | 29.0 | 29.0 | 30.1 |
| L1 | 37.9 | 40.7 | 43.4 |
| L2 | 261.5 | 263 | 273 |
| 全局内存 | 466.3 | 478.8 | 541.5 |

`gpu_bandwidth.cu` 在大数组上应达到标称带宽的九成上下：同一篇论文实测 A100 PCIe 1407 GB/s，H800 PCIe 1862 GB/s。每 SM 只放 1 个 128 线程块时带宽会明显掉下来，用 `sim_inflight.py` 的账解释。

## 版本

CUDA 13.4 Update 2，PyTorch 2.15.0。CUDA 13 最低支持 Turing（sm_75）。
