"""GEMM 模拟器测试：验证访存计算和性能指标的正确性。

测试 sim_gemm.py 和 sim_occupancy.py 中的计算公式，包括：
1. 朴素 GEMM 的访存字节数（依据：Roofline 论文，Berkeley）
2. 分块优化的效果（依据：Boehm 的 10 步 CUDA GEMM）
3. 线程分块数据复用（依据：Volkov 2008，寄存器分块）
4. 占用率与寄存器使用（依据：NVIDIA CUDA 编程指南）
"""
import sys
import os

# 导入被测模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get as get_gpu

# 添加当前目录以导入 sim_gemm 和 sim_occupancy
import sim_gemm
import sim_occupancy


def test_flops_calculation():
    """验证 FLOP 计算：2MNK 公式。

    依据：对称矩阵乘 C[i,j] = sum(A[i,k] * B[k,j])，k=0..K-1
    每个 C 元素 = K 次乘法 + K-1 次加法 ≈ 2K 次操作
    总计 M*N 个 C 元素，所以总 FLOP = 2*M*N*K
    """
    state = sim_gemm.GEMMState(m=1024, n=1024, k=1024)
    expected_flops = 2 * 1024 * 1024 * 1024
    actual_flops = state.flops()
    assert actual_flops == expected_flops, f'FLOP 计算错误：{actual_flops} != {expected_flops}'
    print('✓ test_flops_calculation')


def test_min_gbytes():
    """验证最少访存字节数（理想情况）。

    依据：每个矩阵元素最少从 HBM 读一次（写一次 C）。
    M=4096, N=4096, K=4096, FP32:
    A[M×K] = 4M*K 字节，B[K×N] = 4K*N 字节，C[M×N] = 4M*N 字节
    = 4*(M*K + K*N + M*N) = 4*K*(M+N) + 4*M*N
    """
    state = sim_gemm.GEMMState(m=4096, n=4096, k=4096, dtype_bytes=4)
    min_bytes_mb, _ = state.min_gbytes()

    # 理论值
    a_bytes = 4096 * 4096 * 4
    b_bytes = 4096 * 4096 * 4
    c_bytes = 4096 * 4096 * 4
    expected_mb = (a_bytes + b_bytes + c_bytes) / 1e6

    assert abs(min_bytes_mb - expected_mb) < 0.1, f'最小访存计算错误：{min_bytes_mb} vs {expected_mb}'
    print('✓ test_min_gbytes')


def test_naive_gbytes_larger_than_min():
    """验证朴素 GEMM 的访存 >= 最小访存（现实总是更坏）。

    朴素 GEMM 每次计算都从全局读，所以访存远大于最小值。
    """
    state = sim_gemm.GEMMState(m=1024, n=1024, k=1024)
    min_bytes, _ = state.min_gbytes()
    naive_bytes, _ = state.naive_gbytes()

    assert naive_bytes >= min_bytes, f'朴素访存应 >= 最小值：{naive_bytes} < {min_bytes}'
    assert naive_bytes / min_bytes > 10, '朴素访存应远大于最小值'
    print('✓ test_naive_gbytes_larger_than_min')


def test_blocked_reduces_access():
    """验证分块能减少全局访存。

    依据：Boehm 论文，分块后全局访存主要是加载 A 和 B 一次，
    而不是每个 C 元素都重复加载。
    """
    state = sim_gemm.GEMMState(m=2048, n=2048, k=2048)
    naive_bytes, _ = state.naive_gbytes()
    blocked_bytes, _ = state.blocked_gbytes(block_m=128, block_n=128, block_k=8)

    assert blocked_bytes <= naive_bytes, f'分块访存应 <= 朴素：{blocked_bytes} > {naive_bytes}'
    print('✓ test_blocked_reduces_access')


def test_arithmetic_intensity_increases():
    """验证分块优化后算术强度增加。

    算术强度 = FLOP / 访存字节数
    分块减少访存，强度增加。
    """
    state = sim_gemm.GEMMState(m=4096, n=4096, k=4096)
    flops = state.flops()

    naive_bytes, _ = state.naive_gbytes()
    intensity_naive = flops / (naive_bytes * 1e6)

    blocked_bytes, _ = state.blocked_gbytes(block_m=128, block_n=128, block_k=8)
    intensity_blocked = flops / (blocked_bytes * 1e6)

    assert intensity_blocked > intensity_naive, f'分块强度应更高：{intensity_blocked} <= {intensity_naive}'
    print('✓ test_arithmetic_intensity_increases')


def test_h100_ridge_point():
    """验证 H100 的 Roofline 拐点。

    依据：NVIDIA 官方规格
    H100 BF16：989 TFLOPS，3.35 TB/s
    拐点 = 峰值算力 / 带宽 = 989 / 3.35 ≈ 295 FLOP/字节
    """
    gpu = get_gpu('h100-sxm')
    ridge = gpu.ridge(dtype='bf16')

    # 允许 5% 误差
    expected_ridge = 989 / 3.35
    assert abs(ridge - expected_ridge) / expected_ridge < 0.05, f'H100 拐点计算错误：{ridge} vs {expected_ridge}'
    print('✓ test_h100_ridge_point')


def test_thread_tile_reuse_increases():
    """验证线程分块增加数据复用。

    依据：Volkov 寄存器分块论文
    线程分块 TM×TN 越大，每个 A、B 元素被访问的次数越多。
    """
    state = sim_gemm.GEMMState(m=4096, n=4096, k=4096)

    # TM=1,TN=1 时数据复用最少
    intensity1, desc1 = state.thread_tile_reduction(thread_m=1, thread_n=1, block_k=8)

    # TM=8,TN=8 时数据复用增加
    intensity2, desc2 = state.thread_tile_reduction(thread_m=8, thread_n=8, block_k=8)

    # 更大的线程分块应该有更好的强度（数据复用）
    assert intensity2 >= intensity1, f'较大分块强度应更好：{intensity2} < {intensity1}'
    print('✓ test_thread_tile_reuse_increases')


def test_occupancy_register_tradeoff():
    """验证占用率与寄存器的权衡。

    依据：NVIDIA CUDA 编程指南第 5 章
    累加器越多，占用率越低。
    """
    analyzer = sim_occupancy.GEMMRegisterAnalysis()

    # TM=1, TN=1 时寄存器少，占用率高
    _, _, occ_small, _ = analyzer.occupancy_analysis(block_size=256, thread_m=1, thread_n=1)

    # TM=8, TN=8 时寄存器多，占用率低
    _, _, occ_large, _ = analyzer.occupancy_analysis(block_size=256, thread_m=8, thread_n=8)

    assert occ_small >= occ_large, f'占用率应随累加器增加而降低：{occ_small} < {occ_large}'
    print('✓ test_occupancy_register_tradeoff')


def test_register_calculation_monotonic():
    """验证寄存器使用随线程分块单调增加。"""
    analyzer = sim_occupancy.GEMMRegisterAnalysis()

    prev_regs = 0
    for tm in [1, 2, 4, 8]:
        regs = analyzer.regs_per_thread_for_tile(thread_m=tm, thread_n=tm)
        assert regs >= prev_regs, f'寄存器应单调增加：{regs} < {prev_regs}'
        prev_regs = regs

    print('✓ test_register_calculation_monotonic')


def test_occupancy_reasonable_range():
    """验证占用率在合理范围内（0-100%）。"""
    analyzer = sim_occupancy.GEMMRegisterAnalysis()

    for tm in range(1, 17):
        for tn in range(1, 17):
            if tm * tn > 256:
                continue
            _, _, occ, _ = analyzer.occupancy_analysis(block_size=256, thread_m=tm, thread_n=tn)
            assert 0 <= occ <= 100, f'占用率超出范围：{occ}%'

    print('✓ test_occupancy_reasonable_range')


def test_small_matrix_special_case():
    """验证小矩阵的特殊情况（M<block_m 等）。"""
    state = sim_gemm.GEMMState(m=64, n=64, k=64)

    # 不应崩溃
    flops = state.flops()
    assert flops == 2 * 64 * 64 * 64

    min_bytes, _ = state.min_gbytes()
    assert min_bytes > 0

    blocked_bytes, _ = state.blocked_gbytes(block_m=128, block_n=128, block_k=8)
    assert blocked_bytes > 0

    print('✓ test_small_matrix_special_case')


def test_zero_dimension_handled():
    """验证零维度或极端情况的处理。"""
    # 极小矩阵
    state = sim_gemm.GEMMState(m=1, n=1, k=1)
    assert state.flops() == 2

    min_bytes, _ = state.min_gbytes()
    assert min_bytes > 0

    print('✓ test_zero_dimension_handled')


def test_shared_memory_h100_limits():
    """验证 H100 共享内存限制（228 KB per SM, 227 KB per block with opt-in, 48 KB default）。

    依据：NVIDIA Hopper 架构规范
    """
    # 验证常数定义
    assert sim_gemm.SMEM_PER_SM == 228 * 1024, 'SMEM_PER_SM 应为 228 KB'
    assert sim_gemm.SMEM_PER_BLOCK_OPTIN == 227 * 1024, 'SMEM_PER_BLOCK_OPTIN 应为 227 KB'
    assert sim_gemm.SMEM_PER_BLOCK_DEFAULT == 48 * 1024, 'SMEM_PER_BLOCK_DEFAULT 应为 48 KB'

    # 128×128×8 FP32 双缓冲分块
    # 单缓冲: 128*8*4 + 8*128*4 = 4096 + 4096 = 8192 B
    # 双缓冲: 2 * 8192 = 16384 B = 16 KB (fits in opt-in limit)
    bm, bn, bk = 128, 128, 8
    dtype_bytes = 4
    shmem_ab_single = (bm * bk + bk * bn) * dtype_bytes
    shmem_double = shmem_ab_single * 2

    assert shmem_double <= sim_gemm.SMEM_PER_BLOCK_OPTIN, \
        f'128×128×8 双缓冲 ({shmem_double/1024:.1f} KB) 应适应 opt-in 限制'

    # 验证线程分块检查可行性
    state = sim_gemm.GEMMState(m=4096, n=4096, k=4096)
    intensity, desc = state.thread_tile_reduction(thread_m=8, thread_n=8, block_k=8, use_optin=True)
    assert intensity > 0, f'128×128×8 应在 opt-in 限制内可行，但返回: {desc}'

    print('✓ test_shared_memory_h100_limits')


def run_all_tests():
    """运行所有测试。"""
    tests = [
        test_flops_calculation,
        test_min_gbytes,
        test_naive_gbytes_larger_than_min,
        test_blocked_reduces_access,
        test_arithmetic_intensity_increases,
        test_h100_ridge_point,
        test_thread_tile_reuse_increases,
        test_occupancy_register_tradeoff,
        test_register_calculation_monotonic,
        test_occupancy_reasonable_range,
        test_small_matrix_special_case,
        test_zero_dimension_handled,
        test_shared_memory_h100_limits,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f'✗ {test.__name__}: {e}')
            failed += 1
        except Exception as e:
            print(f'✗ {test.__name__}: {type(e).__name__}: {e}')
            failed += 1

    print(f'\n结果：{passed} 通过，{failed} 失败')
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
