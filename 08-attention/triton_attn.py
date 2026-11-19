"""FlashAttention-2 Triton 实现（需要 GPU）。对应 GPU-08 第 5 章。

本文件是教学用代码框架，未在本仓库真卡复测。
期望环境：NVIDIA Ampere 及以后（A100、H100、L40、L4、B100 等），PyTorch 2.1.0+，Triton 2.1.0+。

核心步骤（Dao et al. 2023 "FlashAttention-2"）：
  1. 外层分块：扫 Q（行块），每块计算与所有 K/V（列块）的注意力
  2. 内层并行：扫 K/V 块时，warp 间分工：某些 warp 计算 S、softmax，其他 warp 计算输出
  3. 在线 softmax：记 m（行最大）、l（行和），换块时用 exp(m_old - m_new) 重缩放
  4. 循环顺序改进：外扫 Q，内扫 K，最后对 V 做向量化读取

论文数字（H100, BF16）：
  N=1024, d=64：146 TFLOPs（标准 ~ 40 TFLOPs）
  N=4096, d=128：190 TFLOPs
  相对峰值 150-200 TFLOPS / 989 TFLOPS = 15%-20% MFU

实现提示（参考 https://github.com/openai/flash-attention）：
  - 用 tl.reduce(axis=1) 计算 max_qk、sum_qk 用于在线 softmax
  - 块大小常选 (block_M, block_N) = (128, 128) 或 (64, 256)
  - SRAM 需要 ~96KB（H100），否则调度失败
  - 可选：debug_mask 用 tl.debug_print() 在运行时检查中间值

数值稳定性（Milakov & Gimelshein 2019）：
  - exp(x - m) 避免溢出；使用 tl.maximum 更新 m
  - l（行和）逐块累加时用 exp(m_old - m_new) 改正旧结果
  - 最后 output /= l 一次；BF16 精度下误差 < 1e-3
"""
# 未实现：此框架需要真实卡和 Triton 环境
