# GPU-06 Tensor Core 与低精度

配套代码：Tensor Core 峰值、片段布局、浮点格式与低精度量化的 CPU 模拟器和单元测试。

## 文件清单

| 文件 | 环境 | 做什么 | 对应文档章节 |
|------|------|--------|-------------|
| sim_tc.py | Python 3.10+ | Tensor Core 指令峰值、片段寄存器占用、GEMM 强度 | 第 1-5 章 |
| sim_quantize.py | Python 3.10+ | FP8 和 FP4 格式规范、混合精度噪声、缩放因子开销 | 第 6-10 章 |
| test_tc.py | Python 3.10+ | 13 个单元测试，复现 NVIDIA 官方峰值和 FP8 规格 | 全部 |

## 运行方式

### 查看 Tensor Core 峰值

```bash
# H100 BF16 峰值
python3 sim_tc.py peak

# H100 FP8 峰值
python3 sim_tc.py peak --dtype fp8

# B200 BF16 峰值
python3 sim_tc.py peak --gpu b200 --dtype bf16
```

### 查看片段布局与寄存器占用

```bash
# mma.sync 16×8×16 的标准片段
python3 sim_tc.py fragment --api mma.sync

# wgmma（Hopper）的共享内存占用
python3 sim_tc.py fragment --api wgmma

# tcgen05（Blackwell）的 Tensor Memory 占用
python3 sim_tc.py fragment --api tcgen05
```

### 计算 GEMM 的强度与瓶颈

```bash
# 4K×4K×4K GEMM 的强度和 Roofline 拐点
python3 sim_tc.py strength

# 定制矩阵大小与分块
python3 sim_tc.py strength --m 8192 --n 8192 --k 4096 --tile-m 256 --tile-n 256
```

### 浮点格式与低精度

```bash
# 列出所有格式的范围与精度
python3 sim_quantize.py format

# 检查特定值是否在 FP8-E5M2 范围内
python3 sim_quantize.py format --fmt fp8-e5m2 --check-range 1e-5 1e4

# 混合精度的量化噪声（FP8 权重 + FP8 激活）
python3 sim_quantize.py noise

# 缩放因子的显存开销（向量级 vs 块级）
python3 sim_quantize.py overhead --params 671 --granule vector

# FP8 的下溢与过流风险
python3 sim_quantize.py deadzone --fmt fp8-e5m2 --grad-scale 256.0

# DeepSeek-V3 FP8 配方（显存估算）
python3 sim_quantize.py deepseekmoe --seq 8192 --vocab 102400 --num-layers 160
```

### 运行单元测试

```bash
# 运行所有 13 个测试（复现 NVIDIA 官方数字）
python3 test_tc.py
```

## 期望输出

### sim_tc.py peak

```
GPU                  架构             SM  TC/SM      峰值（TFLOPS）       精度
----------------------------------------------------------------------
H100 SXM             Hopper        132      4             989     bf16
```

### sim_tc.py fragment --api mma.sync

```
API               mma.sync
Shape (M×N×K)      16×8×16
A 寄存器                    8
B 寄存器                    4
C 寄存器                    4
总计（32-bit）              16
总计（字节）                  64
```

### sim_quantize.py format

```
格式                  位宽      范围（log10）            最大值            最小值
----------------------------------------------------------------------
fp8-e4m3             8            6.5      4.480e+02      1.520e-04
  Ampere 口径
fp8-e5m2             8            9.0      5.734e+04      6.100e-05
  Hopper 口径
```

### test_tc.py

```
运行 13 个单元测试...

✓ H100 BF16 峰值：989 TFLOPS
✓ H100 FP8 峰值：1979 TFLOPS
✓ B200 BF16 峰值：2250 TFLOPS
...
结果：13 passed，0 failed
```

## 代码特点

- **标准库 Python**：CPU 部分只用 Python 3.10+ 标准库，无需 numpy。
- **实测数字**：所有峰值和格式规范来自 NVIDIA 官方规格和 CUTLASS 库（核对于 2026-11-04）。
- **物理约束验证**：单元测试覆盖 Roofline 拐点单调性、量化噪声单调性、累加精度约束等。
- **DeepSeek-V3 配方**：包含 671B 参数 FP8 训练的显存估算与缩放策略。
