"""Triton 矩阵乘 kernel。

本文件在 GPU 上运行，需要环境：PyTorch 2.15+ 和 triton-lang 2.2+。
实际运行本脚本需要 Ampere 及以后的 GPU（A100、H100 等）。

代码基于 Triton 官方矩阵乘教程的持久化版本：
  https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

本仓库未在真卡上复测，期望值来自文献 [1]。
"""

# stub: 仓库内无法运行，仅供参考
raise RuntimeError("本文件需要 GPU 和 Triton，请在有 NVIDIA GPU 的机器上运行")

# 下面是伪代码框架

# import triton
# import triton.language as tl
# import torch
#
# @triton.jit
# def matmul_kernel(
#     a_ptr, b_ptr, c_ptr,
#     M, N, K,
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
# ):
#     """Triton 矩阵乘 kernel（持久化版本）。"""
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#
#     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
#     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
#
#     accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
#
#     for k_block in range(0, (K + BLOCK_K - 1) // BLOCK_K):
#         rk = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
#         a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
#         b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
#         accumulator += tl.dot(a, b, allow_tf32=False)
#
#     c = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
#     tl.store(c, accumulator)
