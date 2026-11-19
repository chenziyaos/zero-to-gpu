"""GEMM 分块与访存模式模拟器：计算朴素到分块 GEMM 各阶段的访存和性能。

对应 GPU-05 第 1 到 8 章。核心问题：
1. 朴素 GEMM 需要从全局内存读多少次
2. 分块（tile）怎样降低全局内存读写次数
3. 线程分块（thread tile）怎样降低共享内存的读写
4. 占用率和寄存器压力的权衡

关键参数：
  M, N, K     矩阵尺寸（C = A·B，其中 A[M,K], B[K,N], C[M,N]）
  block_m, block_n, block_k  block 分块大小
  thread_m, thread_n  单线程计算的 C tile 大小
  block_tile_m, block_tile_n  block 分块后的 C tile 大小

公式依据：
  全局内存读写字节数（朴素）= 2MK + 2NK + 2MN（读 A、B，写 C）
  全局内存读写字节数（分块）= 2MN + K(M+N)（KM 次读 A，KN 次读 B，一次写 C）
  共享内存读次数（无分块） = 2K（每个 C 元素需要读一整行 A 和一整列 B）
  共享内存读次数（线程分块） = K / (thread_m * thread_n)（数据复用）

单位约定：
  时间    微秒 (us)
  带宽    GB/s
  容量    KB（共享内存）、MB（全局内存）
  强度    FLOP/字节

用法：
  python3 sim_gemm.py naive --m 4096 --n 4096 --k 4096
  python3 sim_gemm.py blocked --m 4096 --n 4096 --k 4096 --bm 128 --bn 128 --bk 8
  python3 sim_gemm.py thread-tile --m 4096 --n 4096 --k 4096 --tm 8 --tn 8
  python3 sim_gemm.py intensity --m 4096 --n 4096 --k 4096
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# Hopper (H100) shared memory constants (in bytes)
SMEM_PER_SM = 228 * 1024  # Total shared memory per SM
SMEM_PER_BLOCK_DEFAULT = 48 * 1024  # Default per-block limit without opt-in
SMEM_PER_BLOCK_OPTIN = 227 * 1024  # Max per-block with cudaFuncSetAttribute opt-in


@dataclass
class GEMMState:
    """单个 GEMM 的计算与访存状态。"""
    m: int
    n: int
    k: int
    dtype_bytes: int = 4  # FP32 默认 4 字节，可改 BF16 = 2

    def flops(self) -> int:
        """GEMM 总浮点运算数 = 2MNK（乘积和）。"""
        return 2 * self.m * self.n * self.k

    def min_gbytes(self) -> Tuple[float, str]:
        """最少理论访存字节数：每个 A[i,p] 和 B[p,j] 只需从 HBM 读一次。
        返回（字节数 MB，描述）。"""
        # A: M×K，每个元素读 1 次 = M*K*dtype
        # B: K×N，每个元素读 1 次 = K*N*dtype
        # C: M×N，每个元素写 1 次 = M*N*dtype（实际只写，也计入）
        a_bytes = self.m * self.k * self.dtype_bytes
        b_bytes = self.k * self.n * self.dtype_bytes
        c_bytes = self.m * self.n * self.dtype_bytes
        total = a_bytes + b_bytes + c_bytes
        return total / 1e6, f'A[{self.m}x{self.k}] + B[{self.k}x{self.n}] + C[{self.m}x{self.n}]'

    def naive_gbytes(self) -> Tuple[float, str]:
        """朴素 GEMM 每个 C[i,j] 需要读一整行 A[i,:] 和一整列 B[:,j]。
        现实中需要多次加载。返回（字节数 MB，描述）。"""
        # 最坏情况：每次计算 C[i,j] += A[i,p] * B[p,j] 都从全局读
        # = M*N*K 次读 A + M*N*K 次读 B + M*N 次写 C
        a_reads = self.m * self.n * self.k * self.dtype_bytes
        b_reads = self.m * self.n * self.k * self.dtype_bytes
        c_writes = self.m * self.n * self.dtype_bytes
        total = a_reads + b_reads + c_writes
        return total / 1e6, f'无缓存，每个 C 元素遍历全 K'

    def blocked_gbytes(self, block_m: int, block_n: int, block_k: int) -> Tuple[float, str]:
        """分块 GEMM：将 C 分成 (block_m × block_n) 的 tile，K 分成 block_k 段。
        每个 tile 需要的 A、B 只读一次，共 K/block_k 个计算阶段。"""
        # A: 在所有 tile 中共读 M*K 字节（沿 N 维复用）
        # B: 在所有 tile 中共读 K*N 字节（沿 M 维复用）
        # C: 写 M*N 字节
        a_bytes = self.m * self.k * self.dtype_bytes
        b_bytes = self.k * self.n * self.dtype_bytes
        c_bytes = self.m * self.n * self.dtype_bytes
        total = a_bytes + b_bytes + c_bytes

        # 但实际上分块后访存额外开销（tile 边界处理等）
        num_tiles_m = (self.m + block_m - 1) // block_m
        num_tiles_n = (self.n + block_n - 1) // block_n
        num_k_tiles = (self.k + block_k - 1) // block_k

        desc = (f'分块 {num_tiles_m}x{num_tiles_n}x{num_k_tiles}，'
                f'tile({block_m}x{block_n}x{block_k})，'
                f'每个 K 段重复加载 {num_tiles_m * num_tiles_n} 个 tile')
        return total / 1e6, desc

    def thread_tile_reduction(self, thread_m: int, thread_n: int, block_k: int = 8, use_optin: bool = True) -> Tuple[float, str]:
        """线程分块：单线程计算 thread_m × thread_n 个 C 元素，数据复用增加。
        共享内存读次数大幅降低。"""
        # 假设 block size = 128 threads（8×16）
        block_m = min(128, self.m)  # 典型 128
        block_n = min(128, self.n)

        # 共享内存大小约束（H100 Hopper）
        shmem_limit_bytes = SMEM_PER_BLOCK_OPTIN if use_optin else SMEM_PER_BLOCK_DEFAULT
        shmem_a = block_m * block_k * self.dtype_bytes
        shmem_b = block_k * block_n * self.dtype_bytes
        shmem_used = shmem_a + shmem_b

        # 检查可行性
        limit_name = "opt-in limit" if use_optin else "default limit"
        if shmem_used > shmem_limit_bytes:
            return 0, f'超过共享内存{limit_name}：{shmem_used/1024:.1f} KB > {shmem_limit_bytes/1024:.1f} KB'

        # 共享内存读次数（线程分块后）
        # 每个线程读 block_k 次（A），block_k 次（B），但跨 thread_m*thread_n 个线程分摊
        shared_reads_per_iter = (shmem_a + shmem_b) / self.dtype_bytes  # 单位：元素

        # 全 K 阶段的共享内存总读次数
        num_k_iters = (self.k + block_k - 1) // block_k
        total_shmem_reads = shared_reads_per_iter * num_k_iters * self.dtype_bytes

        intensity = self.flops() / (total_shmem_reads / 1e9) if total_shmem_reads > 0 else 0

        desc = (f'线程分块 {thread_m}x{thread_n}，共享内存 {shmem_used/1024:.1f} KB（{limit_name}），'
                f'读次数约 {total_shmem_reads/1e6:.1f} MB')
        return intensity, desc


def cmd_naive(args):
    """子命令：朴素 GEMM（无优化）。"""
    state = GEMMState(args.m, args.n, args.k, args.dtype_bytes)
    flops = state.flops()
    min_bytes, min_desc = state.min_gbytes()
    naive_bytes, naive_desc = state.naive_gbytes()

    print(f'{"朴素 GEMM":20} {"M":>6}x{"N":>6}x{"K":>6}  {"浮点运算":>12}  {"最少访存":>12}  {"实际访存":>12}')
    print('-' * 95)
    print(f'{"矩阵尺寸":20} {state.m:6}x{state.n:6}x{state.k:6}  {flops:12,.0f}  {min_bytes:12.1f} MB  {naive_bytes:12.1f} MB')
    print(f'\n最少访存描述：{min_desc}')
    print(f'朴素访存描述：{naive_desc}')
    print(f'\n算术强度（最少）：{flops / (min_bytes * 1e6):.2f} FLOP/字节')
    print(f'算术强度（朴素）：{flops / (naive_bytes * 1e6):.2f} FLOP/字节')


def cmd_blocked(args):
    """子命令：分块 GEMM（tile 级别）。"""
    state = GEMMState(args.m, args.n, args.k, args.dtype_bytes)
    flops = state.flops()
    min_bytes, _ = state.min_gbytes()

    print(f'{"分块 GEMM":20} {"M":>6}x{"N":>6}x{"K":>6}  {"BM":>6}x{"BN":>6}x{"BK":>6}  {"访存 MB":>12}  {"强度":>10}')
    print('-' * 100)

    blocked_bytes, desc = state.blocked_gbytes(args.bm, args.bn, args.bk)
    intensity = flops / (blocked_bytes * 1e6)

    print(f'{"分块参数":20} {state.m:6}x{state.n:6}x{state.k:6}  {args.bm:6}x{args.bn:6}x{args.bk:6}  {blocked_bytes:12.1f}  {intensity:10.2f}')
    print(f'\n分块描述：{desc}')

    # 计算 block 级别的性能指标
    num_blocks_m = (state.m + args.bm - 1) // args.bm
    num_blocks_n = (state.n + args.bn - 1) // args.bn
    num_blocks = num_blocks_m * num_blocks_n
    print(f'\nBlock 布局：{num_blocks_m} × {num_blocks_n} = {num_blocks} blocks')
    print(f'访存（相对朴素）：{blocked_bytes / min_bytes:.1%}')


def cmd_thread_tile(args):
    """子命令：线程分块对共享内存读次数的影响。"""
    state = GEMMState(args.m, args.n, args.k, args.dtype_bytes)
    flops = state.flops()

    print(f'{"线程分块":20} {"M":>6}x{"N":>6}x{"K":>6}  {"TM":>4}x{"TN":>4}  {"数据复用":>12}  {"性能指标":>15}')
    print('-' * 95)

    # 扫描多个线程分块大小
    for tm in [1, 2, 4, 8, 16]:
        for tn in [1, 2, 4, 8, 16]:
            if tm * tn > 256:  # 每个线程最多 256 个累加器寄存器
                continue
            intensity, desc = state.thread_tile_reduction(tm, tn, args.bk)
            print(f'{"线程分块":20} {state.m:6}x{state.n:6}x{state.k:6}  {tm:4}x{tn:4}  {intensity:12.2f}')

    print(f'\n线程分块增加数据复用：TM×TN 越大，每个元素重用次数越多')
    print(f'约束条件：TM×TN×累加器字节 <= 256 KB（寄存器）')


def cmd_intensity(args):
    """子命令：不同分块大小下的算术强度与瓶颈分析。"""
    state = GEMMState(args.m, args.n, args.k, args.dtype_bytes)
    gpu = get(args.gpu)

    # H100 基准数据
    peak_tflops = gpu.bf16_tflops if args.dtype_bytes == 2 else gpu.bf16_tflops
    bw_tbs = gpu.hbm_tbs
    ridge = peak_tflops / bw_tbs  # 拐点

    print(f'{"算术强度分析":20} GPU: {gpu.name}')
    print(f'{"":20} 峰值: {peak_tflops:.0f} TFLOPS，带宽: {bw_tbs:.2f} TB/s，拐点: {ridge:.0f} FLOP/B')
    print('-' * 100)
    print(f'{"分块配置":30} {"全局访存 MB":>15} {"算术强度":>12} {"瓶颈":>15}')
    print('-' * 100)

    flops = state.flops()

    # 几个典型分块
    configs = [
        (1, 1, 1, '朴素（无分块）'),
        (64, 64, 8, '小分块'),
        (128, 128, 8, '中分块'),
        (256, 256, 16, '大分块'),
    ]

    for bm, bn, bk, label in configs:
        if bm > state.m or bn > state.n or bk > state.k:
            continue
        bytes_mb, _ = state.blocked_gbytes(bm, bn, bk)
        intensity = flops / (bytes_mb * 1e6)
        bound = '算力' if intensity >= ridge else '带宽'
        print(f'{label:30} {bytes_mb:15.1f}  {intensity:12.2f}  {bound:>15}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # naive 子命令
    p_naive = subparsers.add_parser('naive', help='朴素 GEMM（无优化）')
    p_naive.add_argument('--m', type=int, default=4096, help='矩阵 M 维度')
    p_naive.add_argument('--n', type=int, default=4096, help='矩阵 N 维度')
    p_naive.add_argument('--k', type=int, default=4096, help='矩阵 K 维度')
    p_naive.add_argument('--dtype-bytes', type=int, default=4, help='数据类型字节数（4=FP32, 2=BF16）')
    p_naive.set_defaults(func=cmd_naive)

    # blocked 子命令
    p_blocked = subparsers.add_parser('blocked', help='分块 GEMM（tile 优化）')
    p_blocked.add_argument('--m', type=int, default=4096)
    p_blocked.add_argument('--n', type=int, default=4096)
    p_blocked.add_argument('--k', type=int, default=4096)
    p_blocked.add_argument('--bm', type=int, default=128, help='Block M 维度')
    p_blocked.add_argument('--bn', type=int, default=128, help='Block N 维度')
    p_blocked.add_argument('--bk', type=int, default=8, help='Block K 维度（共享内存分块）')
    p_blocked.add_argument('--dtype-bytes', type=int, default=4)
    p_blocked.set_defaults(func=cmd_blocked)

    # thread-tile 子命令
    p_thread = subparsers.add_parser('thread-tile', help='线程分块（数据复用分析）')
    p_thread.add_argument('--m', type=int, default=4096)
    p_thread.add_argument('--n', type=int, default=4096)
    p_thread.add_argument('--k', type=int, default=4096)
    p_thread.add_argument('--bk', type=int, default=8)
    p_thread.add_argument('--dtype-bytes', type=int, default=4)
    p_thread.set_defaults(func=cmd_thread_tile)

    # intensity 子命令
    p_intensity = subparsers.add_parser('intensity', help='算术强度与瓶颈分析')
    p_intensity.add_argument('--m', type=int, default=4096)
    p_intensity.add_argument('--n', type=int, default=4096)
    p_intensity.add_argument('--k', type=int, default=4096)
    p_intensity.add_argument('--gpu', type=str, default='h100-sxm', help='GPU 型号')
    p_intensity.add_argument('--dtype-bytes', type=int, default=4)
    p_intensity.set_defaults(func=cmd_intensity)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
