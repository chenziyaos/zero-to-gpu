# 02-ledger

GPU-02 性能账本的配套练习。先在 CPU 上把每个算子的浮点数和字节数算清楚，判断它落在 Roofline 的哪一侧；再到真卡上测出本机的两根线，用三个小实验看开销、带宽、算力三类瓶颈各长什么样，最后用 ncu 拍一张 Speed Of Light。

| 文件 | 环境 | 做什么 | 对应文档章节 |
|---|---|---|---|
| `sim_roofline.py` | CPU | 拐点、常见算子强度、GEMM 与分块强度、decode 线性层扫 token 数、注意力 decode 与 prefill、训练每 token 运算量与 MFU | 第 1、2、3、7、8 章 |
| `sim_regime.py` | CPU | batch 翻倍判瓶颈、n 个逐元素算子融合的流量账、kernel 固定开销占比 | 第 4、5 章 |
| `test_roofline.py` | CPU | 复现 NVIDIA 性能指南表 1、How To Scale Your Model 的拐点 295、PaLM 附录 B 的 MT-NLG MFU 29.7% 等 11 个用例 | 全章 |
| `gpu_mamf.py` | GPU | MAMF 式矩阵乘形状扫描和拷贝带宽，得到实测拐点 | 第 3 章 |
| `gpu_regimes.py` | GPU | repeat（带宽到算力的拐弯）、batch（开销平台）、fuse（融合减半流量） | 第 4、5 章 |
| `ncu_roofline.sh` | GPU | ncu 采 SpeedOfLight 和分层 Roofline，打印吞吐摘要 | 第 6 章 |

## 运行

```bash
cd zero-to-gpu/02-ledger
python3 test_roofline.py
python3 sim_roofline.py ridge
python3 sim_roofline.py ops --gpu h100-sxm
python3 sim_roofline.py gemm --m 4096 --n 4096 --k 4096
python3 sim_roofline.py decode --gpu h100-sxm
python3 sim_roofline.py decode --wbytes 0.5
python3 sim_roofline.py attn --ctx 32768
python3 sim_roofline.py mfu --tps 9000
python3 sim_regime.py regime --t1 1.00 --t2 1.10
python3 sim_regime.py fuse --n 4
python3 sim_regime.py launch --us 5

# 下面需要 GPU 和 PyTorch 2.x
python3 gpu_mamf.py
python3 gpu_regimes.py repeat
python3 gpu_regimes.py batch
python3 gpu_regimes.py fuse
./ncu_roofline.sh python3 gpu_regimes.py fuse
```

## 期望输出

`sim_roofline.py ridge`：理论拐点用厂商 BF16 稠密峰值除以 HBM 带宽，实测拐点用 MSMF 除以带宽的九成。

```text
GPU                      BF16    HBM     拐点 │   MSMF   带宽×0.9     实测拐点
A100 80GB SXM             312  2.039    153 │    271     1.84      148
H100 SXM                  989   3.35    295 │    786     3.02      261
B200（DGX B200 / 8）       2250      8    281 │   1439     7.20      200
```

`sim_roofline.py gemm`：4096 方阵 137.4 GFLOP、搬运 96 MiB，强度 1365，算力时间 139.0 us 远大于访存时间 30.0 us。

`sim_roofline.py decode`：H100 上 d=4096、f=14336 的线性层，1 个 token 强度 1、只用到峰值 0.3%；256 个 token 时间几乎不涨（35.1 到 37.9 us），强度 237；越过拐点约需 295 个 token，权重换成 4 bit（`--wbytes 0.5`）约 74 个。

`sim_roofline.py attn`：32 个 query 头、8 个 kv 头时 decode 注意力强度恒为 4，与上下文长度无关；32K prefill 强度 8192。

`sim_roofline.py mfu`：默认按 8B 稠密模型（32 层、32 头、head_dim 128、序列 8192）在 H100 上记账，6N 为 48.2 GFLOP，注意力项 12LHQT 为 12.9 GFLOP、占 21%；MFU 30%、40%、50% 分别要求每卡每秒 4859、6478、8098 个 token，`--tps 9000` 对应 MFU 55.6%。

`sim_regime.py fuse --n 4`：64M 个 BF16 元素，分开跑读写 1024 MiB、下限 320.5 us，融合后 256 MiB、80.1 us。

`sim_regime.py launch --us 5`：每个 kernel 固定 5 us 时，1M 元素的加法开销占 72.7%，16M 元素降到 14.3%。

真卡上的对照数：

| 实验 | 参考结果 | 出处 |
|---|---|---|
| `gpu_mamf.py` BF16 | A100 SXM 约 271 TFLOPS，H100 SXM 约 853，B200 约 1737 | stas00/ml-engineering 的 MAMF 表 |
| `gpu_regimes.py repeat` | r 较小时时间几乎不变，r 越过本机 FP32 非 Tensor Core 拐点后随 r 线性上涨 | Horace He，Making Deep Learning Go Brrrr From First Principles |
| `gpu_regimes.py fuse` | 融合后约 2 倍加速 | 同上 |
| `gpu_regimes.py batch` | 小 batch 段翻倍时间只涨 10% 左右，属于开销受限 | 同上 |

## 版本

CUDA 13.4 Update 2，PyTorch 2.15.0，Nsight Compute 2026.3.1。ncu 需要性能计数器权限，容器里通常要加 `--cap-add SYS_ADMIN`，或由管理员设置驱动参数 `NVreg_RestrictProfilingToAdminUsers=0`。
