"""注意力内核 HBM 流量账本：标准写法、分块融合和在线 softmax（CPU）。对应 GPU-08 第 1、2、3 章。

核心对比：
  标准三段式（Q @ K.T、softmax、结果 @ V）：N² 矩阵往返 HBM，HBM 流量随序列长度 O(N²) 扩展。
  分块融合 + 在线 softmax：中间矩阵只活在 SRAM，HBM 流量变为 O(N×d)，只需读一遍 Q/K/V、写一遍输出。

公式（FlashAttention-1 论文，Dao et al. 2022）：
  朴素：QK 矩阵 (N, N)、softmax 分数 (N, N)、AV 结果 (N, d) → 读写 2N² + 2Nd 字节。
  FA1：分块循环访问，N² 矩阵只在块级 SRAM 驻留，总 HBM 访问 (4 + q_len/blocksize) × Nd 字节（最坏常数）。
  精确数字见 Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"（2022）。

在线 softmax（Milakov & Gimelshein, 2019；DA 改进）：
  记 m_i（行最大）和 l_i（行和），换块时用 exp(m_old - m_new) 重缩放旧结果，最后除 l 一次。
  结果和整行 softmax 数值相同（精度受硬件舍入限制）。

子命令：
  naive       标准三段式的 HBM 流量和时间账本
  blocked     分块融合（FA1 口径）的 HBM 流量
  savings     两种方案对比：流量、时间、加速比

用法：
  python3 sim_attn.py naive --seq-len 8192 --head-dim 128 --num-heads 32
  python3 sim_attn.py blocked --seq-len 8192 --head-dim 128 --num-heads 32 --block-size 256
  python3 sim_attn.py savings --seq-len 8192 --head-dim 128 --num-heads 32 --block-size 256 --gpu h100-sxm
"""
import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get  # noqa: E402


@dataclass
class AttnAccount:
    """注意力算子的流量和时间账。"""
    seq_len: int
    head_dim: int
    num_heads: int
    hbm_reads_gb: float
    hbm_writes_gb: float
    hbm_total_gb: float

    def time_us(self, bw_tbs: float) -> float:
        """给定 HBM 带宽下的执行时间（微秒）。带宽以 TB/s 计。"""
        return self.hbm_total_gb / (bw_tbs * 1000) * 1e6

    def ops_if_compute(self, bf16_tflops: float) -> float:
        """算力瓶颈下的计算时间（微秒）。"""
        # 注意力总浮点数：2×N×N×d（QK 和 softmax@V 各 2 FLOP/elem，忽略 exp/softmax 开销）
        flops = 2 * self.seq_len * self.seq_len * self.head_dim
        return flops / (bf16_tflops * 1e12) * 1e6


def naive_attn_hbm(seq_len: int, head_dim: int, num_heads: int, dtype_bytes: int = 2) -> AttnAccount:
    """标准三段式注意力的 HBM 流量。

    三个 kernel：
      1. Q @ K.T ：读 Q(N, d), K(N, d)，写 S(N, N)。读 2Nd，写 N²。
      2. softmax(S) ：读写 S(N, N)。读写 N²。
      3. softmax(S) @ V ：读 S(N, N), V(N, d)，写 O(N, d)。读 N² + Nd，写 Nd。

    总：读 2Nd + 2N² + N² + Nd + Nd = 2Nd + 3N²；写 N² + Nd。
    按 dtype_bytes 计字节（BF16 = 2 字节）。
    """
    reads_bytes = (2 * seq_len * head_dim + 3 * seq_len * seq_len) * dtype_bytes
    writes_bytes = (seq_len * seq_len + seq_len * head_dim) * dtype_bytes
    total_bytes = reads_bytes + writes_bytes

    reads_gb = reads_bytes / 1e9
    writes_gb = writes_bytes / 1e9
    total_gb = total_bytes / 1e9

    # 批次维度（num_heads 视为批量）
    reads_gb *= num_heads
    writes_gb *= num_heads
    total_gb *= num_heads

    return AttnAccount(seq_len, head_dim, num_heads, reads_gb, writes_gb, total_gb)


def blocked_attn_hbm(seq_len: int, head_dim: int, num_heads: int,
                     block_size: int, dtype_bytes: int = 2) -> AttnAccount:
    """分块融合注意力（FA1）的 HBM 流量。

    假设：
      - 外层循环扫 K（block_size × block_size 块），内层扫 Q（block_size）。
      - 共 ceil(N/block_size)² 个 KV 块，每块处理 N/block_size 个 Q 块。
      - 每个 Q 块读一遍（N/B 次），共读 Nd × (N/B) × num_Q_blocks = Nd × N / B 次。

    简化估计（FA1 论文的 FLOPS 分析）：
      读：Q 一遍、K (N/B) 倍、V (N/B) 倍、O 写一遍。
      约 (2 + 2×N/B) × N × d 字节读，N×d 字节写。

    更保守的估计（块大小 B = block_size）：
      读 Nd + Nd×(N/B) + Nd×(N/B) + Nd = Nd×(3 + 2N/B)
      写 Nd
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    reuses_per_q_block = num_blocks

    # Q、O 各一遍；K、V 因外层循环每块被读 num_blocks 次
    reads_bytes = (2 * seq_len * head_dim  # Q + O 初读
                   + 2 * seq_len * head_dim * reuses_per_q_block) * dtype_bytes  # K、V 重读
    writes_bytes = seq_len * head_dim * dtype_bytes
    total_bytes = reads_bytes + writes_bytes

    reads_gb = reads_bytes / 1e9 * num_heads
    writes_gb = writes_bytes / 1e9 * num_heads
    total_gb = total_bytes / 1e9 * num_heads

    return AttnAccount(seq_len, head_dim, num_heads, reads_gb, writes_gb, total_gb)


def print_account(label: str, acc: AttnAccount, bw_tbs: float = None, bf16_tflops: float = None):
    """输出一份账本。"""
    print(f"\n{label}")
    print(f"  序列长度 N = {acc.seq_len}, 头维度 d = {acc.head_dim}, 头数 = {acc.num_heads}")
    print(f"  HBM 读 {acc.hbm_reads_gb:.2f} GB, 写 {acc.hbm_writes_gb:.2f} GB, 总 {acc.hbm_total_gb:.2f} GB")
    if bw_tbs:
        t_mem_us = acc.time_us(bw_tbs)
        print(f"  带宽受限（{bw_tbs} TB/s）：{t_mem_us:.1f} μs（{t_mem_us/1000:.2f} ms）")
    if bf16_tflops:
        t_math_us = acc.ops_if_compute(bf16_tflops)
        print(f"  算力受限（{bf16_tflops} TFLOPS BF16）：{t_math_us:.1f} μs（{t_math_us/1000:.2f} ms）")
    if bw_tbs and bf16_tflops:
        bound = "算力" if acc.ops_if_compute(bf16_tflops) >= acc.time_us(bw_tbs) else "带宽"
        print(f"  瓶颈：{bound}")


def cmd_naive(args):
    """标准三段式注意力账本。"""
    acc = naive_attn_hbm(args.seq_len, args.head_dim, args.num_heads)
    print_account("标准三段式注意力", acc, args.bw_tbs, args.bf16_tflops)


def cmd_blocked(args):
    """分块融合注意力账本。"""
    acc = blocked_attn_hbm(args.seq_len, args.head_dim, args.num_heads, args.block_size)
    print_account("分块融合注意力（FA1）", acc, args.bw_tbs, args.bf16_tflops)


def cmd_savings(args):
    """对比两种方案。"""
    naive = naive_attn_hbm(args.seq_len, args.head_dim, args.num_heads)
    blocked = blocked_attn_hbm(args.seq_len, args.head_dim, args.num_heads, args.block_size)

    print("\n对比：标准 vs FA1 分块")
    print(f"  参数：N={args.seq_len}, d={args.head_dim}, heads={args.num_heads}, block_size={args.block_size}")
    print(f"\n  标准三段式：{naive.hbm_total_gb:.2f} GB HBM")
    print(f"  FA1 分块：  {blocked.hbm_total_gb:.2f} GB HBM")
    print(f"  削减：      {naive.hbm_total_gb - blocked.hbm_total_gb:.2f} GB ({100*(1-blocked.hbm_total_gb/naive.hbm_total_gb):.1f}%)")

    if args.bw_tbs:
        naive_t = naive.time_us(args.bw_tbs)
        blocked_t = blocked.time_us(args.bw_tbs)
        print(f"\n  @ {args.bw_tbs} TB/s HBM 带宽：")
        print(f"    标准：{naive_t:.1f} μs ({naive_t/1000:.2f} ms)")
        print(f"    FA1：{blocked_t:.1f} μs ({blocked_t/1000:.2f} ms)")
        print(f"    加速：{naive_t/blocked_t:.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # 为每个子命令添加相同的参数
    for subparser in [subparsers.add_parser('naive', help='标准三段式流量账'),
                      subparsers.add_parser('blocked', help='FA1 分块融合流量账'),
                      subparsers.add_parser('savings', help='对比标准 vs FA1')]:
        subparser.add_argument('--bw-tbs', type=float, default=3.35, help='HBM 带宽 TB/s（默认 H100 3.35）')
        subparser.add_argument('--bf16-tflops', type=float, default=989, help='BF16 峰值 TFLOPS（默认 H100 989）')
        subparser.add_argument('--seq-len', type=int, default=8192, help='序列长度（默认 8192）')
        subparser.add_argument('--head-dim', type=int, default=128, help='头维度（默认 128）')
        subparser.add_argument('--num-heads', type=int, default=32, help='注意力头数（默认 32）')
        subparser.add_argument('--block-size', type=int, default=256, help='分块大小（默认 256）')
        subparser.add_argument('--gpu', type=str, help='GPU 型号（覆盖 --bw-tbs 和 --bf16-tflops）')

    args = parser.parse_args()

    if args.gpu:
        gpu = get(args.gpu)
        args.bw_tbs = gpu.hbm_tbs
        args.bf16_tflops = gpu.bf16_tflops

    if args.cmd == 'naive':
        cmd_naive(args)
    elif args.cmd == 'blocked':
        cmd_blocked(args)
    elif args.cmd == 'savings':
        cmd_savings(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
