"""寄存器与占用率的权衡：GEMM 中累加器、共享内存对占用率的影响。

对应 GPU-05 第 6 章。核心问题：
1. 线程分块越大，累加器使用越多，SM 上驻留的 warp 越少（占用率降低）
2. 占用率低是否意味着性能差？对于 GEMM，答案是否定的，因为存在指令级并行
3. 给定 block size，怎样平衡累加器与占用率

关键公式（基于 Hopper/H100）：
  寄存器文件大小    = 256 KB / SM
  共享内存大小      = 228 KB / SM (可配置，单个 block 最多 227 KB with opt-in，48 KB without)
  线程寄存器占用    = ceil(寄存器总数 / 256)  [单位：256 元素块]
  warp 占用        = SM寄存器 / 线程寄存器 / 32 threads/warp
  block 驻留warp数  = min(SM_count * warp_占用, ceil(threads_per_block / 32))

单位约定：
  寄存器  4 字节单位（元素数）
  SM     个数
  warp   32 线程
  占用率  %
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# Hopper (H100) shared memory and warp constants
SMEM_PER_SM = 228 * 1024  # Total shared memory per SM
SMEM_PER_BLOCK_DEFAULT = 48 * 1024  # Default per-block limit without opt-in
SMEM_PER_BLOCK_OPTIN = 227 * 1024  # Max per-block with cudaFuncSetAttribute opt-in
MAX_WARPS_PER_SM = 64  # Hopper supports up to 64 warps per SM


@dataclass
class GEMMRegisterAnalysis:
    """GEMM 线程分块与寄存器使用分析。"""
    # H100 规格（其他 GPU 可配置）
    sm_count: int = 132  # H100
    regs_per_sm: int = 256 * 1024 // 4  # 256 KB / 4 字节 = 65536 个寄存器元素
    max_warps_per_sm: int = MAX_WARPS_PER_SM  # Hopper: 64 warps per SM
    max_threads_per_block: int = 1024

    @staticmethod
    def regs_per_thread_for_tile(thread_m: int, thread_n: int, dtype_bytes: int = 4) -> int:
        """线程块（thread tile）中的累加器寄存器数。
        假设每个累加器 FP32 = 4 字节，BF16 = 2 字节（但按 FP32 对齐）。

        包含：
        - thread_m × thread_n 个累加器（C 矩阵）
        - thread_m × block_k 个寄存器用于 A 元素缓存（估计）
        - thread_n × block_k 个寄存器用于 B 元素缓存（估计）
        """
        # 累加器：thread_m * thread_n * 4 字节 / 4 = thread_m * thread_n 元素
        accumulators = thread_m * thread_n

        # A/B 缓存（近似：存几个元素在寄存器中）
        a_cache = thread_m * 2  # 每次读 2 个 A 元素
        b_cache = thread_n * 2  # 每次读 2 个 B 元素

        total = (accumulators + a_cache + b_cache) * (dtype_bytes // 2)  # 按字节换算
        return total

    def occupancy_analysis(self, block_size: int, thread_m: int, thread_n: int,
                          block_k: int = 8, dtype_bytes: int = 4) -> Tuple[int, int, float, str]:
        """计算给定配置下的占用率和性能指标。

        返回：（每个block驻留warp数，每个SM驻留warp数，占用率%，描述）
        """
        # 每个线程的寄存器数
        regs_per_thread = self.regs_per_thread_for_tile(thread_m, thread_n, dtype_bytes)

        # 寄存器向上对齐到 32 的倍数（GPU 的量子化）
        regs_per_thread_rounded = ((regs_per_thread + 31) // 32) * 32

        # 一个 warp（32 线程）的总寄存器数
        regs_per_warp = regs_per_thread_rounded * 32

        # SM 里能驻留多少个 warp（受寄存器限制）
        warps_per_sm_reg = self.regs_per_sm // regs_per_warp if regs_per_warp > 0 else 32
        warps_per_sm = min(warps_per_sm_reg, self.max_warps_per_sm)

        # Block 能装多少个 warp
        warps_per_block = (block_size + 31) // 32

        # Block 驻留多少个 warp（受 SM 驻留限制）
        block_warps = min(warps_per_block, warps_per_sm)

        # 占用率 = block_warps / max_warps_per_sm
        occupancy_pct = (block_warps / self.max_warps_per_sm) * 100

        desc = (f'线程分块 {thread_m}x{thread_n}，寄存器 {regs_per_thread}→{regs_per_thread_rounded}，'
                f'warps {block_warps}/{self.max_warps_per_sm}，占用率 {occupancy_pct:.1f}%')

        return block_warps, warps_per_sm, occupancy_pct, desc

    def compute_capability_impact(self, block_size: int, thread_m: int, thread_n: int) -> str:
        """评估占用率对计算能力的影响。

        GEMM 中低占用率不一定导致性能差，因为有指令级并行补偿。
        """
        _, _, occ, _ = self.occupancy_analysis(block_size, thread_m, thread_n)

        if occ >= 75:
            return '高占用率，充分隐藏延迟'
        elif occ >= 50:
            return '中等占用率，指令级并行补偿'
        elif occ >= 25:
            return '低占用率，GEMM 中正常（累加器占用大量寄存器）'
        else:
            return '极低占用率，性能可能受影响'


def cmd_analyze(args):
    """子命令：占用率分析。"""
    analyzer = GEMMRegisterAnalysis()

    print(f'{"GEMM 占用率分析":25} GPU: H100 SXM')
    print(f'{"":25} SM: {analyzer.sm_count}，寄存器/SM: {analyzer.regs_per_sm/1024:.0f} K，最大 warp/SM: {analyzer.max_warps_per_sm}')
    print('-' * 120)
    print(f'{"线程分块":15} {"Block":>8} {"每线程寄存器":>15} {"块内Warp":>10} {"占用率":>10} {"评估":>25}')
    print('-' * 120)

    # 扫描不同线程分块大小
    block_size = 256  # 16×16 = 256 threads（典型配置）

    for tm in [1, 2, 4, 8]:
        for tn in [1, 2, 4, 8]:
            tile_threads = tm * tn
            if tile_threads > 256:
                continue

            warps, warps_sm, occ, desc = analyzer.occupancy_analysis(block_size, tm, tn)
            impact = analyzer.compute_capability_impact(block_size, tm, tn)

            regs_per_thread = analyzer.regs_per_thread_for_tile(tm, tn)
            print(f'{tm}x{tn:6}       {block_size:8}  {regs_per_thread:15}  {warps:10}  {occ:9.1f}%  {impact:>25}')


def cmd_sweep(args):
    """子命令：扫描参数空间找最优配置。"""
    analyzer = GEMMRegisterAnalysis()

    print(f'{"参数空间扫描":25} 目标：高性能且占用率 >= 50%')
    print('-' * 130)
    print(f'{"线程分块":15} {"Block":>8} {"浮点数":>12} {"寄存器":>12} {"占用率":>10} {"可行性":>15}')
    print('-' * 130)

    best_config = None
    best_score = 0

    for block_size in [128, 256, 512]:
        for tm in [1, 2, 4, 8, 16]:
            for tn in [1, 2, 4, 8, 16]:
                if tm * tn > 256 or tm * tn > block_size // 4:
                    continue

                warps, _, occ, _ = analyzer.occupancy_analysis(block_size, tm, tn)
                regs_per_thread = analyzer.regs_per_thread_for_tile(tm, tn)

                # 评分：高占用率 + 大数据复用（tm*tn）
                score = occ * (1 + tm * tn / 100)

                feasible = '✓' if occ >= 50 else '✗'

                if (tm, tn) in [(1, 1), (2, 2), (4, 4), (8, 8)]:
                    print(f'{tm}x{tn:6}       {block_size:8}  {tm*tn:12}  {regs_per_thread:12}  {occ:9.1f}%  {feasible:>15}')

                if occ >= 50 and score > best_score:
                    best_score = score
                    best_config = (block_size, tm, tn, occ)

    print('\n推荐配置：')
    if best_config:
        block_size, tm, tn, occ = best_config
        print(f'  Block size: {block_size}，线程分块: {tm}x{tn}，占用率: {occ:.1f}%')


def cmd_shared_memory(args):
    """子命令：共享内存与寄存器的权衡。"""
    print(f'{"共享内存与寄存器权衡":25} H100: {SMEM_PER_SM/1024:.0f} KB 共享内存 / SM')
    print(f'{"单block限制":25} 默认: {SMEM_PER_BLOCK_DEFAULT/1024:.0f} KB，opt-in: {SMEM_PER_BLOCK_OPTIN/1024:.0f} KB')
    print('-' * 120)
    print(f'{"Block 分块":15} {"单缓冲 KB":>15} {"双缓冲 KB":>15} {"寄存器 K":>12} {"可行（opt-in）":>20}')
    print('-' * 120)

    # 典型的 GEMM 共享内存使用
    configs = [
        (64, 64, 8, '小分块'),
        (128, 128, 8, '中分块'),
        (256, 128, 8, '大分块 M'),
        (128, 256, 16, '大分块 N'),
        (128, 256, 32, '超大分块'),
    ]

    for block_m, block_n, block_k, label in configs:
        # 共享内存用于 A 和 B 的双缓冲
        dtype_bytes = 4  # FP32
        shmem_a = block_m * block_k * dtype_bytes
        shmem_b = block_k * block_n * dtype_bytes
        shmem_single = shmem_a + shmem_b

        # 加上双缓冲的额外开销
        shmem_double_buffer = shmem_single * 2

        # 寄存器用于累加器（估计）
        regs_accum = block_m * block_n * dtype_bytes

        # 检查可行性：单个block使用opt-in限制
        feasible = '✓' if shmem_double_buffer <= SMEM_PER_BLOCK_OPTIN else '✗'
        print(f'{label:15} {shmem_single/1024:15.1f}  {shmem_double_buffer/1024:15.1f}  {regs_accum/1024:12.1f}  {feasible:>20}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # analyze 子命令
    p_analyze = subparsers.add_parser('analyze', help='占用率分析')
    p_analyze.set_defaults(func=cmd_analyze)

    # sweep 子命令
    p_sweep = subparsers.add_parser('sweep', help='参数空间扫描')
    p_sweep.set_defaults(func=cmd_sweep)

    # shared-memory 子命令
    p_shmem = subparsers.add_parser('shared-memory', help='共享内存与寄存器权衡')
    p_shmem.set_defaults(func=cmd_shared_memory)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
