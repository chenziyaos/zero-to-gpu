"""在线 softmax 精度与分块切割：给定块大小和精度要求，推算注意力分块方案（CPU）。

对应 GPU-08 第 2 章：在线 softmax 是分块融合注意力的基石。

核心算法（Milakov & Gimelshein 2019；Dao et al. 改进）：
  设第 i 块的行最大为 m_i，行和为 l_i。一行 softmax：
    exp(x_j) / (sum_k exp(x_k)) = exp(x_j - m_global) / l_global
  不知道 m_global 时逐块计算：
    m_new = max(m_old, m_i)
    l_new = exp(m_old - m_new) * l_old + sum_j exp(S_ij - m_new)
    O_new = exp(m_old - m_new) * O_old + exp(S - m_new) @ V_i

  精度：数值稳定的两阶段归一化（online softmax），结果与一次性 softmax 相同。
  但受硬件舍入限制：fp32 精度下误差 < 2e-7；fp16 下 < 1e-3。

子命令：
  scaling    给定块大小和精度需求，计算重缩放因子（exp 计算量预估）
  blocks     推荐的块分割方案（以 SRAM 容量和 exp 吞吐平衡）
  io-lower   FlashAttention IO 下界分析

用法：
  python3 sim_online_softmax.py scaling --seq-len 8192 --head-dim 128 --block-size 256
  python3 sim_online_softmax.py blocks --seq-len 8192 --head-dim 128 --sram-kb 96 --gpu h100-sxm
  python3 sim_online_softmax.py io-lower --seq-len 8192 --head-dim 128 --gpu h100-sxm
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get  # noqa: E402


def online_softmax_stats(seq_len: int, head_dim: int, block_size: int) -> dict:
    """给定块大小，分析在线 softmax 的重缩放成本。

    输入：
      seq_len：序列长度（每个 token 一行）
      head_dim：向量维度 d（softmax 后还需 @ V）
      block_size：每个块的 tokens 数（对应 K/V 的行数）

    输出：
      num_blocks：总块数（以序列长度分块）
      rescales_per_token：每个 token 经历的重缩放次数（决定 exp 计算量）
      output_elements：softmax 输出元素数（N×d）
      m_updates：行最大更新次数（每块一次）
    """
    num_blocks = (seq_len + block_size - 1) // block_size

    # 每个 Q token（行）需要遍历所有 KV 块，重缩放 (num_blocks - 1) 次
    # 首个块初始化 m、l，后续块每块都需要一次 exp(m_old - m_new)
    rescales_per_token = max(0, num_blocks - 1)
    total_rescales = seq_len * rescales_per_token

    output_elements = seq_len * head_dim

    # 行最大更新：每个块都可能改变 m
    m_updates = seq_len * num_blocks

    return {
        'num_blocks': num_blocks,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'block_size': block_size,
        'rescales_per_token': rescales_per_token,
        'total_rescales': total_rescales,
        'output_elements': output_elements,
        'm_updates': m_updates,
    }


def exp_cost_estimate(seq_len: int, num_blocks: int, exp_tflops: float = 16.0) -> float:
    """exp 计算的时间预估（H100 上 MUFU，每 SM 每时钟 16 次）。

    H100 MUFU（Multi-Function Unit）峰值：每 SM 16 exp/time。
    132 SM，3.67 GHz 时钟：峰值约 16 × 132 × 3.67 / 1e9 ≈ 79.7 GigaExp/s。

    简化：exp 计算量（元素数）/ 吞吐 = 时间。
    """
    # 在线 softmax：每个 token 需要 (num_blocks - 1) 次 exp（重缩放），每次标量 exp
    # 加上 softmax 本体：exp(S - m) 共 N×block_size 次，sum 和 normalize 每个 token 一次
    rescale_exps = seq_len * max(0, num_blocks - 1)
    softmax_exps = seq_len * num_blocks  # 每块内的 exp(S_ij - m)，共 N 行
    total_exps = rescale_exps + softmax_exps

    # 按 exp/ns 换算时间（粗糙）
    time_us = total_exps / (exp_tflops * 1e9) * 1e6
    return time_us


def cmd_scaling(args):
    """分析重缩放成本。"""
    stats = online_softmax_stats(args.seq_len, args.head_dim, args.block_size)
    print(f"\n在线 softmax 重缩放分析")
    print(f"  参数：N={stats['seq_len']}, d={stats['head_dim']}, block_size={stats['block_size']}")
    print(f"  块数：{stats['num_blocks']}")
    print(f"  每 token 重缩放次数：{stats['rescales_per_token']}")
    print(f"  总重缩放操作数：{stats['total_rescales']}")
    print(f"  行最大（m）更新：{stats['m_updates']}")
    print(f"  输出元素数（N×d）：{stats['output_elements']}")

    # exp 开销估计
    exp_us = exp_cost_estimate(stats['seq_len'], stats['num_blocks'], exp_tflops=16)
    print(f"\n  exp 计算成本（H100 MUFU, 16 exp/cycle）：{exp_us:.1f} μs")
    print(f"  占总注意力时间比例（估计 ~5ms）：{100 * exp_us / 5000:.1f}%")


def cmd_blocks(args):
    """推荐块大小。"""
    # SRAM（共享内存）容量：给定 SRAM_KB，计算最大块大小
    # Q 块 (block_size, head_dim)、K 块 (block_size, head_dim)、V 块 (block_size, head_dim)
    # 分数矩阵 (block_size, block_size)、输出缓冲 (block_size, head_dim)
    # 总：3 × block_size × head_dim + block_size² + block_size × head_dim (FP32/BF16: 4 或 2 字节)

    sram_bytes = args.sram_kb * 1024
    dtype_bytes = 2  # BF16

    # 简化：SRAM 大约用来放 Q_block、K_block、V_block、S、output 缓冲
    # block_size × d（Q）+ block_size × d（K）+ block_size × d（V）+
    # block_size × block_size（S）+ block_size × d（O）
    # = block_size × (4d + block_size) × dtype_bytes

    # 解 block_size² + 4d × block_size - SRAM_bytes/dtype_bytes = 0
    # 一元二次方程
    a = 1
    b = 4 * args.head_dim
    c = -sram_bytes / dtype_bytes
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        print(f"SRAM 不足（{args.sram_kb} KB 装不下 head_dim={args.head_dim} 的一个块）")
        return
    block_size_max = (-b + math.sqrt(discriminant)) / (2 * a)
    block_size_max = int(block_size_max)

    print(f"\n推荐块大小")
    print(f"  SRAM：{args.sram_kb} KB")
    print(f"  head_dim：{args.head_dim}")
    print(f"  序列长度：{args.seq_len}")
    print(f"\n  最大块大小（SRAM 限制）：{block_size_max}")

    # 推荐块大小：256 或 512（权衡 SRAM 和重缩放开销）
    for block_size in [64, 128, 256, 512]:
        if block_size > block_size_max:
            continue
        stats = online_softmax_stats(args.seq_len, args.head_dim, block_size)
        exp_us = exp_cost_estimate(args.seq_len, stats['num_blocks'], exp_tflops=16)
        print(f"\n  block_size={block_size}: {stats['num_blocks']} 块，重缩放 {stats['rescales_per_token']}x/token，exp ~{exp_us:.1f} μs")


def cmd_io_lower(args):
    """FlashAttention IO 下界分析（Dao et al. 2022）。

    核心公式（FA1 论文定理 1）：
      注意力的 IO 下界 = Θ(Nd + N²/M) 字节，其中 M = SRAM 大小（元素数）。
      对于 M >> d，主要项是 O(N²/M)。

    实例数字：H100 上 96 KB SRAM，N=8192, d=128：
      下界约 5.6 MB（比标准的 16.25 GB 少 ~3000 倍）。
    """
    gpu = get(args.gpu)
    sram_bytes = args.sram_kb * 1024
    dtype_bytes = 2

    M = sram_bytes // dtype_bytes  # SRAM 元素数（BF16）

    # 下界公式：(Nd + N²/M) × dtype_bytes
    io_lower = (args.seq_len * args.head_dim + args.seq_len * args.seq_len / M) * dtype_bytes

    # 对比：标准写法和下界
    naive_io = (2 * args.seq_len * args.head_dim +
                3 * args.seq_len * args.seq_len) * dtype_bytes

    print(f"\nFlashAttention IO 下界（Dao et al. 2022）")
    print(f"  GPU：{gpu.name}, SRAM：{args.sram_kb} KB, N={args.seq_len}, d={args.head_dim}")
    print(f"\n  标准三段式：{naive_io / 1e9:.2f} GB")
    print(f"  IO 下界：  {io_lower / 1e9:.2f} GB")
    print(f"  最多能改进：{naive_io / io_lower:.0f}x")

    # 带宽下的时间
    naive_time_us = (naive_io / (gpu.hbm_tbs * 1e12)) * 1e6
    io_lower_time_us = (io_lower / (gpu.hbm_tbs * 1e12)) * 1e6
    print(f"\n  @ {gpu.hbm_tbs} TB/s HBM 带宽：")
    print(f"    标准：{naive_time_us:.1f} μs ({naive_time_us/1000:.2f} ms)")
    print(f"    下界：{io_lower_time_us:.1f} μs ({io_lower_time_us/1000:.2f} ms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # 为每个子命令添加相同的参数
    for subparser in [subparsers.add_parser('scaling', help='重缩放成本分析'),
                      subparsers.add_parser('blocks', help='推荐块大小'),
                      subparsers.add_parser('io-lower', help='IO 下界分析')]:
        subparser.add_argument('--seq-len', type=int, default=8192, help='序列长度')
        subparser.add_argument('--head-dim', type=int, default=128, help='头维度')
        subparser.add_argument('--block-size', type=int, default=256, help='块大小')
        subparser.add_argument('--sram-kb', type=int, default=96, help='共享内存大小（KB）')
        subparser.add_argument('--gpu', type=str, default='h100-sxm', help='GPU 型号')

    args = parser.parse_args()

    if args.cmd == 'scaling':
        cmd_scaling(args)
    elif args.cmd == 'blocks':
        cmd_blocks(args)
    elif args.cmd == 'io-lower':
        cmd_io_lower(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
