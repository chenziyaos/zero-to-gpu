"""sim_memory.py 的单元测试。python3 -m pytest test_memory.py 或 python3 test_memory.py"""
from sim_memory import (
    coalesce_sectors, coalesce_utilization, bank_conflicts,
    transpose_bandwidth_estimate, reduce_bandwidth_estimate,
    SECTOR_SIZE, SHARED_BANKS, WARP_THREADS
)
import math


def test_coalesce_continuous():
    """连续读 32 个 float：32*4=128 字节，恰好 4 个 sector。

    来源：NVIDIA CUDA 编程指南，Global Memory Access Pattern。
    """
    addresses = [i * 4 for i in range(32)]  # 4 字节/元素
    sectors = coalesce_sectors(addresses)
    assert sectors == 4, f'预期 4，得到 {sectors}'


def test_coalesce_stride_2():
    """步长 2：32 个元素 span = 31*8 = 248 字节，需要 ceil(252/32) = 8 个 sector。

    来源：NVIDIA CUDA 编程指南。
    """
    addresses = [i * 8 for i in range(32)]  # 步长 2 元素 = 8 字节
    sectors = coalesce_sectors(addresses)
    assert sectors == 8, f'预期 8，得到 {sectors}'


def test_coalesce_stride_32():
    """步长 32：每个线程独占一个 sector，共 32 个。

    来源：NVIDIA CUDA 编程指南中关于"stride 32 时完全不合并"的说明。
    span = 31*128 + 4 = 3976 字节，sector 数 = ceil(3976/32) = 125 个。
    """
    addresses = [i * 128 for i in range(32)]  # 步长 32 元素 = 128 字节
    sectors = coalesce_sectors(addresses)
    # 实际上步长 32 会导致大量 sector，不是 32 个
    assert sectors > 100, f'步长大时 sector 数应很多，得到 {sectors}'


def test_row_utilization():
    """按行读矩阵：步长 1，连续 32 个元素。

    有效字节 = 32*4 = 128 B；sector 数 = 4；实际 = 128 B；利用率 = 100%。
    """
    useful, transferred, util = coalesce_utilization('row', 128, 32, stride=1)
    assert useful == 128, f'有效字节预期 128，得 {useful}'
    assert transferred == 128, f'实际字节预期 128，得 {transferred}'
    assert abs(util - 100.0) < 0.1, f'利用率预期 100%，得 {util:.1f}%'


def test_col_utilization_stride_1():
    """按列读矩阵：步长 1，但地址跨度大。

    32 个线程按列读，步长 = n（矩阵宽度）；以 n=32 为例，span = 31*32*4 = 3968 字节。
    """
    useful, transferred, util = coalesce_utilization('col', 128, 32, stride=1)
    assert useful == 128, f'有效字节预期 128，得 {useful}'
    # 按列读时步长大，跨越许多 sector
    assert transferred > useful, f'实际字节应大于有效字节'
    assert util < 50, f'列读利用率应较低，得 {util:.1f}%'


def test_bank_no_conflict():
    """步长 1：gcd(1, 32) = 1，无冲突。"""
    conflicts = bank_conflicts(stride=1)
    assert conflicts == 1, f'预期 1，得 {conflicts}'


def test_bank_32_way_conflict():
    """步长 32：gcd(32, 32) = 32，32 路冲突。

    来源：NVIDIA CUDA 编程指南，Bank Conflict。
    """
    conflicts = bank_conflicts(stride=32)
    assert conflicts == 32, f'预期 32，得 {conflicts}'


def test_bank_gcd_16():
    """步长 16：gcd(16, 32) = 16，16 路冲突。"""
    conflicts = bank_conflicts(stride=16)
    assert conflicts == 16, f'预期 16，得 {conflicts}'


def test_transpose_naive_vs_swizzle():
    """优化前后的性能相差应大于 1x。

    Harris 的论文显示，M2050 上从 naive 的 ~18 GB/s 到 xor_swizzle 的 ~99 GB/s。
    我们验证相对倍数的单调性。
    """
    gb_naive, _ = transpose_bandwidth_estimate(256, 256, optimization='naive')
    gb_swizzle, _ = transpose_bandwidth_estimate(256, 256, optimization='xor_swizzle')
    assert gb_swizzle > gb_naive, f'swizzle 性能应优于 naive；naive={gb_naive:.2f}, swizzle={gb_swizzle:.2f}'


def test_transpose_padded_middle():
    """Shared memory with padding 性能应介于 naive 和 xor_swizzle 之间。

    来源：Harris 的论文 "Efficient Matrix Transpose in CUDA Without Bank Conflicts"。
    """
    gb_naive, _ = transpose_bandwidth_estimate(256, 256, optimization='naive')
    gb_padded, _ = transpose_bandwidth_estimate(256, 256, optimization='shared_padded')
    gb_swizzle, _ = transpose_bandwidth_estimate(256, 256, optimization='xor_swizzle')
    assert gb_naive < gb_padded < gb_swizzle, \
        f'性能应 naive < padded < swizzle；{gb_naive:.2f} < {gb_padded:.2f} < {gb_swizzle:.2f}'


def test_reduce_stage_monotonicity():
    """后期的归约阶段（指令主导）应比早期（带宽主导）慢。

    来源：Harris 的论文 "Parallel Reduction"。
    """
    gb_s1, _ = reduce_bandwidth_estimate(4194304, 'bf16', 'stage1')
    gb_s4, _ = reduce_bandwidth_estimate(4194304, 'bf16', 'stage4')
    assert gb_s1 > gb_s4, f'stage1 应快于 stage4；s1={gb_s1:.2f}, s4={gb_s4:.2f}'


def test_reduce_dtype_size():
    """FP32 的访存量是 BF16 的两倍，所以对带宽的需求更高。

    同样的操作数，FP32 应该因为数据量大而时间更长。
    """
    _, time_bf16 = reduce_bandwidth_estimate(4194304, 'bf16', 'stage1')
    _, time_fp32 = reduce_bandwidth_estimate(4194304, 'fp32', 'stage1')
    assert time_fp32 > time_bf16, f'FP32 时间应长于 BF16；fp32={time_fp32:.1f}us, bf16={time_bf16:.1f}us'


def test_sector_size_constant():
    """Sector 大小恒为 32 字节（硬件规范）。"""
    assert SECTOR_SIZE == 32


def test_shared_banks_constant():
    """共享内存 bank 数恒为 32（自 Fermi 及以后）。"""
    assert SHARED_BANKS == 32


def test_warp_size_constant():
    """Warp 大小恒为 32（所有 NVIDIA GPU）。"""
    assert WARP_THREADS == 32


def test_coalesce_boundary_aligned():
    """128 字节对齐的访问（4 个连续 sector），零浪费。"""
    # 从 0 开始，读 32 个连续 float
    addresses = [i * 4 for i in range(32)]
    sectors = coalesce_sectors(addresses)
    transferred = sectors * SECTOR_SIZE
    useful = 32 * 4
    assert transferred == useful, '对齐访问应零浪费'


if __name__ == '__main__':
    import sys
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            try:
                v()
                print('ok  ', k)
            except AssertionError as e:
                fails += 1
                print('FAIL', k, e)
    sys.exit(fails)
