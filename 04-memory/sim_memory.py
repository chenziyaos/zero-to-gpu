"""GPU 内存访问模式模拟器：sector 合并、bank 冲突、warp shuffle 归约。
对应 GPU-04 第 1 至 7 章。

核心原理：
- 全局内存：warp 的 32 个地址被装进 32 字节 sector 里；不对齐和跨越会增加 sector 数。
  sector 数 = ceil(max_addr - min_addr + 4) / 32（4 字节 float）
- 共享内存：32 个 bank，bank_id = addr % 128 / 4（BF16）或 addr % 128 / 2（FP8），冲突路数 = gcd(步长, 32)
- 归约：树形、warp shuffle、CTA 三级；后两级用原子或共享内存消掉访存，指令成为瓶颈

子命令：
  coalesce  矩阵读写模式的 sector 数与利用率
  bank      共享内存 bank 冲突路数（按步长）
  transpose Harris 转置从朴素到优化的吞吐估计
  reduce    归约七步的吞吐与瓶颈

用法：
  python3 sim_memory.py coalesce --pattern row --m 128 --n 32 --stride 1
  python3 sim_memory.py bank --stride 1 --tile-size 32
  python3 sim_memory.py transpose --m 4096 --n 4096
  python3 sim_memory.py reduce --n 4194304 --dtype bf16

数字取自：
  [1] NVIDIA CUDA 编程指南：Global Memory Access Pattern（核对于 2026-11-02）
  [2] Efficient Matrix Transpose in CUDA Without Bank Conflicts（Harris，核对于 2026-11-02）
  [3] Parallel Reduction（Harris，核对于 2026-11-02）
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# H100 性能参数，来源 NVIDIA 官方规格（核对于 2026-11-02）
H100_HBM_BW_TBS = 3.35  # 单位 TB/s，对应 3350 GB/s
COPY_BW_ACTUAL = H100_HBM_BW_TBS * 0.90  # 实测拷贝带宽约为标称值的九成
WARP_SIZE = 32
SECTOR_SIZE = 32  # 字节
SHARED_BANKS = 32
SHARED_BANK_WIDTH = 4  # 字节，假设 BF16
WARP_THREADS = 32


def coalesce_sectors(addresses):
    """计算 warp 的多个地址需要多少个 sector。

    地址连续性：32 个线程的地址如果都在同一个 32 字节 sector 内，只需 1 个 sector。
    步长为 s 时，span = max_addr - min_addr = 31*s；sector 数 = ceil((span + 4) / 32)。
    """
    if not addresses:
        return 0
    min_addr = min(addresses)
    max_addr = max(addresses)
    span = max_addr - min_addr + 4  # 4 字节（float 宽度）
    sectors = math.ceil(span / SECTOR_SIZE)
    return sectors


def coalesce_utilization(pattern, m, n, stride):
    """矩阵访问模式的带宽利用率。

    Args:
        pattern: 'row' 按行、'col' 按列、'diagonal' 对角线
        m, n: 矩阵维度
        stride: 访存步长（元素数）

    Returns:
        (有效字节数, 实际搬运字节数, 利用率 %)
    """
    elem_bytes = 4  # float32 / BF16
    useful_bytes = WARP_THREADS * elem_bytes  # 32 个 float = 128 字节

    if pattern == 'row':
        # 按行读：32 个连续元素
        addresses = [i * stride * elem_bytes for i in range(WARP_THREADS)]
    elif pattern == 'col':
        # 按列读：步长为 n（矩阵宽度）
        addresses = [i * n * stride * elem_bytes for i in range(WARP_THREADS)]
    elif pattern == 'diagonal':
        # 对角线：每个线程跳 n+1 个元素
        addresses = [i * (n + 1) * stride * elem_bytes for i in range(WARP_THREADS)]
    else:
        raise ValueError(f'未知模式 {pattern}')

    sectors = coalesce_sectors(addresses)
    transferred_bytes = sectors * SECTOR_SIZE
    utilization = (useful_bytes / transferred_bytes) * 100 if transferred_bytes > 0 else 0

    return useful_bytes, transferred_bytes, utilization


def bank_conflicts(stride, tile_size=32):
    """共享内存的 bank 冲突路数。

    冲突路数 = gcd(步长, 32)。
    步长 1：1 路（无冲突），步长 32：32 路冲突。
    """
    conflicts = math.gcd(stride, SHARED_BANKS)
    return conflicts


def transpose_bandwidth_estimate(m, n, tile_size=32, optimization='naive'):
    """Harris 转置的吞吐率估计（GB/s）。

    从论文 "Efficient Matrix Transpose in CUDA Without Bank Conflicts"：
    - Naive：按行读、按列写，造成 32 路写冲突，性能很差
    - Shared memory + padding：读无冲突，写有冲突，性能约 18.8 GB/s（M2050）
    - 消写冲突（XOR swizzle 或填充）：1 路冲突，性能约 99.5 GB/s（M2050）

    我们用相对倍数，绝对值受计算的 overhead 和同步影响。
    """
    elem_bytes = 4
    total_bytes = m * n * elem_bytes

    if optimization == 'naive':
        # 一列读（连续）+ 一行写（跨 tile）= 一行 sector + tile_size 行 sector
        # 平均每个 tile：读 1 行 = 4 个 sector（128 elem），写 tile_size 列 = 128 sector
        avg_sectors_per_tile = 1 + tile_size
        estimated_fraction = 4 / avg_sectors_per_tile  # 相对完美的比例
    elif optimization == 'shared_padded':
        # 读无冲突，写有 1 路冲突（tile_size）
        avg_sectors_per_tile = 1 + 1
        estimated_fraction = 0.5  # 相对 naive
    elif optimization == 'xor_swizzle':
        # 读 1 路 + 写 1 路（消冲突）
        avg_sectors_per_tile = 1 + 1
        estimated_fraction = 0.95  # 接近理论上限（消掉冲突，但有同步 overhead）
    else:
        raise ValueError(f'未知优化 {optimization}')

    # 相对于 memcpy 的性能（实测 memcpy 约 HBM 带宽的 90%）
    theoretical_gb_s = COPY_BW_ACTUAL * estimated_fraction
    time_us = total_bytes / (theoretical_gb_s * 1e9) * 1e6

    return theoretical_gb_s, time_us


def reduce_bandwidth_estimate(n, dtype='bf16', level='stage1'):
    """归约七步的吞吐估计（GB/s）。

    参考 Harris 的论文 "Parallel Reduction"：
    Stage 1: 互相加（没有分支，所有线程活跃，全带宽）
    Stage 2: 分支（一半线程读，一半等待）
    Stage 3: 更多分支
    Stage 4 onwards: 访存不再是瓶颈，变成地址计算和同步

    Args:
        n: 元素数
        dtype: 'bf16' 或 'fp32'
        level: 'stage1'（第 4 步前访存瓶颈） 或 'stage4'（后三步已不受访存限制）
    """
    elem_bytes = {'bf16': 2, 'fp32': 4}[dtype]

    if level == 'stage1':
        # 前三步：所有线程活跃，全带宽；累计 3 次读，访存约 6N 字节
        bytes_accessed = 6 * n * elem_bytes
        gb_s = COPY_BW_ACTUAL * 0.95  # 满带宽，accounting for 一些 overhead
    elif level == 'stage4':
        # 后三步：指令主导，访存已不重要，但同步和地址计算成为瓶颈
        # 这个阶段的性能由计算能力和指令吞吐决定，不是带宽
        bytes_accessed = n * elem_bytes  # 只算最后的写回
        gb_s = COPY_BW_ACTUAL * 0.3  # 因为并不是全时间都在访存
    else:
        raise ValueError(f'未知阶段 {level}')

    time_us = bytes_accessed / (gb_s * 1e9) * 1e6
    return gb_s, time_us


def main():
    parser = argparse.ArgumentParser(
        description='GPU 内存访问模式模拟器',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # coalesce 子命令
    p = subparsers.add_parser('coalesce', help='矩阵读写模式的 sector 合并')
    p.add_argument('--pattern', default='row', choices=['row', 'col', 'diagonal'], help='访问模式')
    p.add_argument('--m', type=int, default=128, help='矩阵行数')
    p.add_argument('--n', type=int, default=32, help='矩阵列数')
    p.add_argument('--stride', type=int, default=1, help='访存步长（元素）')
    p.set_defaults(func=lambda a: cmd_coalesce(a))

    # bank 子命令
    p = subparsers.add_parser('bank', help='共享内存 bank 冲突路数')
    p.add_argument('--stride', type=int, default=1, help='访存步长（元素）')
    p.add_argument('--tile-size', type=int, default=32, help='tile 尺寸')
    p.set_defaults(func=lambda a: cmd_bank(a))

    # transpose 子命令
    p = subparsers.add_parser('transpose', help='Harris 转置的性能估计')
    p.add_argument('--m', type=int, default=4096, help='矩阵行数')
    p.add_argument('--n', type=int, default=4096, help='矩阵列数')
    p.set_defaults(func=lambda a: cmd_transpose(a))

    # reduce 子命令
    p = subparsers.add_parser('reduce', help='归约七步的吞吐估计')
    p.add_argument('--n', type=int, default=4194304, help='元素数')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp32'], help='数据类型')
    p.set_defaults(func=lambda a: cmd_reduce(a))

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


def cmd_coalesce(args):
    """coalesce 子命令的处理函数"""
    useful, transferred, util = coalesce_utilization(args.pattern, args.m, args.n, args.stride)
    sectors = math.ceil(transferred / SECTOR_SIZE)

    print(f'{"模式":<10} {args.pattern}')
    print(f'{"矩阵":<10} {args.m:>6}x{args.n:<6}')
    print(f'{"步长":<10} {args.stride}')
    print(f'{"有效字节":<10} {useful:>8} B')
    print(f'{"实际搬运":<10} {transferred:>8} B')
    print(f'{"Sector 数":<10} {sectors:>8}')
    print(f'{"利用率":<10} {util:>8.1f}%')


def cmd_bank(args):
    """bank 子命令的处理函数"""
    conflicts = bank_conflicts(args.stride, args.tile_size)

    print(f'{"步长":<10} {args.stride}')
    print(f'{"Bank 冲突路数":<15} {conflicts:>8}')
    if conflicts == 1:
        print(f'{"状态":<10} 无冲突')
    else:
        print(f'{"状态":<10} {conflicts} 路冲突（会导致 {conflicts}x 延迟）')


def cmd_transpose(args):
    """transpose 子命令的处理函数"""
    print(f'{"优化阶段":<20} {"GB/s":>10} {"时间 (ms)":>15}')
    print('-' * 50)

    for opt in ['naive', 'shared_padded', 'xor_swizzle']:
        gb_s, time_ms = transpose_bandwidth_estimate(args.m, args.n, optimization=opt)
        time_ms = time_ms / 1000
        print(f'{opt:<20} {gb_s:>10.2f} {time_ms:>15.3f}')

    print(f'\n矩阵尺寸：{args.m} x {args.n}')
    print(f'总数据量：{args.m * args.n * 4 / 1e9:.2f} GB')


def cmd_reduce(args):
    """reduce 子命令的处理函数"""
    print(f'{"阶段":<15} {"GB/s":>10} {"时间 (us)":>15}')
    print('-' * 45)

    for level in ['stage1', 'stage4']:
        gb_s, time_us = reduce_bandwidth_estimate(args.n, args.dtype, level)
        print(f'{level:<15} {gb_s:>10.2f} {time_us:>15.1f}')

    print(f'\n元素数：{args.n}')
    print(f'数据类型：{args.dtype}')
    print(f'总数据量：{args.n * (2 if args.dtype == "bf16" else 4) / 1e9:.3f} GB')


if __name__ == '__main__':
    main()
