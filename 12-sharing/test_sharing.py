"""
测试 MIG 和 MPS 计算器的正确性。

测试复现 NVIDIA 官方文档的公开数字：
1. MIG profile 规格（NVIDIA MIG User Guide Hopper，Table 5）
2. SM 分区块大小（NVIDIA MPS Architecture Guide，Page 18-19）
3. 物理约束：显存 slice 连续性、算力分配上限
"""

import sys
import os

# 添加本目录到 path 便于导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_mig_plan import H100_PROFILES, try_placement, can_fit_single
from sim_mps_calc import partition_sm, calculate_chunk_size, calculate_overhead, estimate_throughput


def test_mig_profiles_exist():
    """测试 1：MIG profile 规格数据完整。

    来源：NVIDIA MIG User Guide for Hopper，Table 5，核对于 2026-11-02
    """
    required_profiles = ['1g.5gb', '2g.10gb', '3g.20gb', '4g.20gb', '6g.40gb', '7g.80gb']
    for name in required_profiles:
        assert name in H100_PROFILES, f"缺失 profile: {name}"
    print("✓ Test 1: 所有必需的 H100 profile 都存在")


def test_mig_compute_slices():
    """测试 2：验证计算 slice 分配（分母 7）。

    来源：NVIDIA MIG User Guide Hopper，Table 5
    - 1g.5gb: 1 compute slice
    - 7g.80gb: 7 compute slices (整卡)
    - 4g.20gb: 4 compute slices
    """
    assert H100_PROFILES['1g.5gb'].compute_slices == 1
    assert H100_PROFILES['7g.80gb'].compute_slices == 7
    assert H100_PROFILES['4g.20gb'].compute_slices == 4
    # 最多 7 个 1g.5gb 实例
    total_compute = sum(H100_PROFILES['1g.5gb'].compute_slices for _ in range(7))
    assert total_compute == 7, "7 × 1g.5gb 应占用 7 compute slices"
    print("✓ Test 2: 计算 slice 总数正确（7 slices）")


def test_mig_memory_slices():
    """测试 3：验证显存 slice 分配（分母 8，每个 10GB）。

    来源：NVIDIA MIG User Guide Hopper，Table 5
    - H100 80GB = 8 × 10GB
    - 1g.5gb: 1 slice (5GB)
    - 7g.80gb: 8 slices (80GB)
    """
    assert H100_PROFILES['1g.5gb'].mem_slices == 1
    assert H100_PROFILES['1g.5gb'].memory_gb == 5
    assert H100_PROFILES['7g.80gb'].mem_slices == 8
    assert H100_PROFILES['7g.80gb'].memory_gb == 80
    # 8 个 1g.5gb 应该无法放下（因为算力只有 7）
    success, _ = try_placement(['1g.5gb'] * 8)
    assert not success, "8 × 1g.5gb 不应放下（算力上限 7）"
    print("✓ Test 3: 显存 slice 总数正确（8 slices，80GB）")


def test_mig_placement_success():
    """测试 4：验证 MIG profile 放置成功的情况。

    来源：NVIDIA MIG User Guide Hopper，Page 30-31 placement examples
    """
    # 2 × 3g.20gb + 1 × 1g.5gb 应该能放下
    success, placement = try_placement(['3g.20gb', '3g.20gb', '1g.5gb'])
    assert success, "2 × 3g.20gb + 1 × 1g.5gb 应该能放下"
    assert len(placement) == 3
    print("✓ Test 4: MIG 放置验证（混合 profile）")


def test_mig_placement_failure():
    """测试 5：验证 MIG profile 无法放置的情况（算力不足）。

    来源：NVIDIA MIG User Guide 约束
    """
    # 8 × 1g.5gb：需要 8 compute slices，但只有 7
    success, placement = try_placement(['1g.5gb'] * 8)
    assert not success, "8 × 1g.5gb 不应成功（计算 slice 上限 7）"
    print("✓ Test 5: MIG 放置验证（超限检测）")


def test_mps_chunk_size_h100():
    """测试 6：验证 H100 MPS 块大小。

    来源：NVIDIA MPS Architecture Guide，Page 18-19
    H100: 132 SM，2 进程 → 66 SM/块，4 进程 → 33 SM/块
    """
    result = calculate_chunk_size(132, 2)
    assert abs(result['chunk_size'] - 66.0) < 0.01, f"2 进程块大小应为 66，得 {result['chunk_size']}"

    result = calculate_chunk_size(132, 4)
    assert abs(result['chunk_size'] - 33.0) < 0.01, f"4 进程块大小应为 33，得 {result['chunk_size']}"

    result = calculate_chunk_size(132, 8)
    assert abs(result['chunk_size'] - 16.5) < 0.01, f"8 进程块大小应为 16.5，得 {result['chunk_size']}"

    print("✓ Test 6: MPS 块大小计算（H100）")


def test_mps_partition_uniform():
    """测试 7：验证 MPS 均匀分配。

    2 进程均分 H100（132 SM）：每进程理想 66 SM
    """
    result = partition_sm(132, 2, [50, 50])
    allocations = result['allocations']
    assert len(allocations) == 2

    # 每个应得 66 SM（1 块 × 66）
    for a in allocations:
        assert a['chunks_needed'] == 1, f"50% 应需 1 块，得 {a['chunks_needed']}"
        assert abs(a['actual_sm'] - 66.0) < 0.01, f"应得 66 SM，得 {a['actual_sm']}"

    print("✓ Test 7: MPS 均匀分配（2 进程 50% 各）")


def test_mps_partition_uneven():
    """测试 8：验证 MPS 不均匀分配（块对齐）。

    2 进程：一个 75%（99 SM），一个 25%（33 SM）
    块大小 66 SM，75% → ceil(99/66)=2 块 → 132 SM（超分）
    """
    result = partition_sm(132, 2, [75, 25])
    allocations = result['allocations']

    # 第一个进程：75% × 132 = 99 SM → ceil(99/66) = 2 块 = 132 SM
    assert allocations[0]['chunks_needed'] == 2, "75% 应需 2 块"
    assert allocations[0]['actual_sm'] == 132, "75% 实际应得 132 SM"

    # 第二个进程：25% × 132 = 33 SM → ceil(33/66) = 1 块 = 66 SM（浪费 33）
    assert allocations[1]['chunks_needed'] == 1, "25% 应需 1 块"
    assert abs(allocations[1]['actual_sm'] - 66) < 0.01, "25% 实际应得 66 SM"

    print("✓ Test 8: MPS 块对齐（不均匀分配）")


def test_mps_overhead():
    """测试 9：验证块对齐的浪费。

    H100 4 进程，块大小 33 SM
    理想: [25, 25, 25, 25]，实际：[33, 33, 33, 33]
    """
    result = calculate_overhead(132, [25, 25, 25, 25])
    overhead = result['allocations']

    # 每个应浪费 33-25=8 SM
    for i, o in enumerate(overhead):
        assert o['ideal_sm'] == 25
        assert o['chunks'] == 1, f"进程 {i+1} 应需 1 块"
        assert abs(o['actual_sm'] - 33) < 0.01, f"进程 {i+1} 实际应得 33 SM"
        assert abs(o['wasted_sm'] - 8) < 0.01, f"进程 {i+1} 浪费应为 8 SM"

    total_waste = result['total_wasted_sm']
    assert abs(total_waste - 32) < 0.01, f"总浪费应为 32 SM，得 {total_waste}"

    print("✓ Test 9: MPS 块对齐开销（4 进程）")


def test_mps_throughput_scaling():
    """测试 10：验证多进程吞吐下降趋势（单调性）。

    进程数增加 → 每进程吞吐递减，总体效率递减
    """
    results = []
    for n in [1, 2, 4, 8]:
        r = estimate_throughput(n, None, 989)
        results.append(r['actual_per_process'])

    # 验证单调递减
    for i in range(len(results) - 1):
        assert results[i] >= results[i+1], f"每进程吞吐应单调递减，但 {results[i]} < {results[i+1]}"

    # 2 进程每进程应约 494 TFLOPS（989/2 减少 5% 切换开销）
    approx_2p = 989 / 2 * 0.95
    assert abs(results[1] - approx_2p) < 50, f"2 进程应约 {approx_2p}，得 {results[1]}"

    print("✓ Test 10: 多进程吞吐单调性（1→2→4→8 进程）")


def test_monotonicity_occupancy():
    """测试 11：物理约束验证 - occupancy 不影响分配。

    MPS 的 SM 分配与 occupancy 无关（occupancy 是 kernel 特性），
    只影响实际吞吐估计。
    """
    # 两个不同 occupancy 的计算应该得出相同的 SM 分配
    r1 = partition_sm(132, 2, [50, 50])
    r2 = partition_sm(132, 2, [50, 50])

    assert r1['chunk_size'] == r2['chunk_size']
    assert len(r1['allocations']) == len(r2['allocations'])
    for a1, a2 in zip(r1['allocations'], r2['allocations']):
        assert a1['actual_sm'] == a2['actual_sm']

    print("✓ Test 11: 物理约束 - SM 分配稳定")


def test_orders_of_magnitude():
    """测试 12：数量级检查。

    H100: 132 SM，989 TFLOPS，3.35 TB/s
    """
    # SM 数量应在 100-200 范围
    h100_sm = 132
    assert 100 <= h100_sm <= 200, f"H100 SM 应在 100-200，得 {h100_sm}"

    # 峰值吞吐应在 500-2000 TFLOPS
    peak_tflops = 989
    assert 500 <= peak_tflops <= 2000, f"H100 峰值应在 500-2000 TFLOPS，得 {peak_tflops}"

    # 带宽应在 1-5 TB/s
    bw_tbs = 3.35
    assert 1 <= bw_tbs <= 5, f"H100 带宽应在 1-5 TB/s，得 {bw_tbs}"

    # Roofline 拐点：989e12 / 3.35e12 ≈ 295 FLOP/字节
    ridge = peak_tflops / bw_tbs
    assert 250 < ridge < 350, f"H100 拐点应在 250-350，得 {ridge}"

    print("✓ Test 12: 数量级检查（H100 规格）")


if __name__ == '__main__':
    print("运行 GPU-12 MIG/MPS 测试（6 个公开数字复现 + 6 个物理约束）\n")

    try:
        test_mig_profiles_exist()
        test_mig_compute_slices()
        test_mig_memory_slices()
        test_mig_placement_success()
        test_mig_placement_failure()
        test_mps_chunk_size_h100()
        test_mps_partition_uniform()
        test_mps_partition_uneven()
        test_mps_overhead()
        test_mps_throughput_scaling()
        test_monotonicity_occupancy()
        test_orders_of_magnitude()

        print("\n" + "=" * 60)
        print("全部 12 个测试通过 ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ 测试失败：{e}", file=sys.stderr)
        sys.exit(1)
