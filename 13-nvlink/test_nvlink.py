"""单元测试：NVLink 带宽计算与 NCCL 结果验证。

测试用例覆盖：
  1. 已发表论文中的具体数字（引用与来源）
  2. 物理约束与单调性检查
  3. 拓扑解析与边界情况

运行方式：python3 test_nvlink.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

from sim_nvlink import ring_allreduce, gpu_gbs, NVLINK_GENS, SLOW_HOPS, parse_topo_matrix
from sim_nccl_check import parse_nccl_log_file, judge_result, EXPECTED_BUSBW


def test_nvlink_specs_h100():
    """H100 NVLink4: 18 links x 25 GB/s per direction = 450 GB/s per direction, 900 GB/s bidirectional."""
    s = NVLINK_GENS['nvlink4']
    assert s['link_gbs'] == 25 and s['links'] == 18
    assert gpu_gbs('nvlink4') == 450
    assert 2 * gpu_gbs('nvlink3') == 600, 'A100 headline 600 GB/s is bidirectional'


def test_nvlink5_specs_b200():
    """B200 NVLink5: 18 links x 50 GB/s per direction = 900 GB/s per direction, 1.8 TB/s bidirectional."""
    assert gpu_gbs('nvlink5') == 900
    assert 2 * gpu_gbs('nvlink5') == 1800


def test_ring_allreduce_8gpu_h100():
    """8 x H100 over NVSwitch: busbw ceiling = 450 GB/s, algbw = 450 x 8 / 14 = 257.1 GB/s."""
    r = ring_allreduce(8, 450, msg_size_bytes=1e9)
    assert abs(r['busbw_gbs'] - 450) < 1e-6
    assert abs(r['algbw_gbs'] - 450 * 8 / 14) < 0.01
    assert abs(r['time_us'] - 1.75e9 / 450e9 * 1e6) < 0.01
    r2 = ring_allreduce(8, 450, msg_size_bytes=2e9)
    assert abs(r2['time_us'] / r['time_us'] - 2.0) < 1e-9


def test_ring_allreduce_monotonicity():
    """With a fixed hop bandwidth, algbw falls toward B/2 as n grows while busbw stays at B."""
    algbw = {n: ring_allreduce(n, 450)['algbw_gbs'] for n in (2, 4, 8, 16)}
    assert algbw[2] > algbw[4] > algbw[8] > algbw[16] > 225
    assert all(abs(ring_allreduce(n, 450)['busbw_gbs'] - 450) < 1e-6 for n in (2, 4, 8, 16))


def test_ring_latency_term():
    """Per-step latency lowers busbw for small messages and vanishes for large ones."""
    small = ring_allreduce(8, 450, msg_size_bytes=1e6, alpha_us=5)
    large = ring_allreduce(8, 450, msg_size_bytes=1e10, alpha_us=5)
    assert small['busbw_gbs'] < 0.1 * 450
    assert large['busbw_gbs'] > 0.99 * 450


def test_ring_allreduce_scalability():
    """busbw scales with the slowest hop: NVLink5 / NVLink4 = 2, a PCIe Gen5 hop caps at 63."""
    ratio = ring_allreduce(8, gpu_gbs('nvlink5'))['busbw_gbs'] / ring_allreduce(8, gpu_gbs('nvlink4'))['busbw_gbs']
    assert abs(ratio - 2.0) < 1e-9
    assert abs(ring_allreduce(8, SLOW_HOPS['pcie5'])['busbw_gbs'] - 63) < 1e-6


def test_topo_matrix_parse():
    """测试 topo 矩阵解析。

    使用临时样例矩阵文件。
    """
    sample_topo = """# 样例矩阵
GPU 0   GPU 1   GPU 2   GPU 3   GPU 4   GPU 5   GPU 6   GPU 7
X       1       1       1       10      10      10      10
1       X       1       1       10      10      10      10
1       1       X       1       10      10      10      10
1       1       1       X       10      10      10      10
10      10      10      10      X       1       1       1
10      10      10      10      1       X       1       1
10      10      10      10      1       1       X       1
10      10      10      10      1       1       1       X
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_topo)
        fname = f.name

    try:
        devices, matrix = parse_topo_matrix(fname)

        # 验证解析结果
        assert len(devices) == 8, f'应解析 8 个设备，得 {len(devices)}'
        assert len(matrix) == 8, f'矩阵应为 8x8，得 {len(matrix)}x{len(matrix[0])}'

        # 验证特定距离
        assert matrix[0][1] == 1, f'GPU0-GPU1 应为 1（NVLink），得 {matrix[0][1]}'
        assert matrix[0][4] == 10, f'GPU0-GPU4 应为 10（PCIe），得 {matrix[0][4]}'

        # 验证对称性
        for i in range(8):
            for j in range(8):
                assert matrix[i][j] == matrix[j][i], \
                    f'矩阵应对称：matrix[{i}][{j}]={matrix[i][j]} vs matrix[{j}][{i}]={matrix[j][i]}'

    finally:
        os.unlink(fname)


def test_nccl_log_parse():
    """测试 NCCL 日志解析。"""
    sample_log = """[GPU0]  size 1073741824, count 1024, type float, op sum, root 0, in_place 0, algorithm RING, protocol LL
#       out-of-place
#                                     bytes  count      type   op     time     algbw  busbw
  rank= 0 nranks= 8 cudaDev= 0  root= 0 avg= 1234567890 algbw=218.76 busbw=382.83
  rank= 1 nranks= 8 cudaDev= 1  root= 0 avg= 1234567890 algbw=218.76 busbw=382.83
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(sample_log)
        fname = f.name

    try:
        config, results = parse_nccl_log_file(fname)

        # 验证解析
        assert config.get('gpu_count') == 8, f'应检测 8 卡，得 {config.get("gpu_count")}'
        assert len(results) >= 2, f'应解析 >=2 行结果，得 {len(results)}'

        # 验证第一条数据
        assert abs(results[0]['algbw'] - 218.76) < 0.01, \
            f'algbw 应为 218.76，得 {results[0]["algbw"]}'
        assert abs(results[0]['busbw'] - 382.83) < 0.01, \
            f'busbw 应为 382.83，得 {results[0]["busbw"]}'

    finally:
        os.unlink(fname)


def test_judge_result():
    """PASS >= 80%, WARN 50-80%, FAIL < 50% of the expected ceiling."""
    assert judge_result(400, 500)[0] == 'PASS'
    assert judge_result(300, 500)[0] == 'WARN'
    assert judge_result(200, 500)[0] == 'FAIL'
    assert judge_result(382.83, EXPECTED_BUSBW[('nvlink4', 8)])[0] == 'PASS'
    assert judge_result(52.4, EXPECTED_BUSBW[('nvlink4', 8)])[0] == 'FAIL'


def test_sample_logs():
    """Shipped synthetic logs are self-consistent: busbw = algbw x 2(n-1)/n."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ('nccl_test_h100_8gpu_good.log', 'nccl_test_h100_8gpu_pcie_hybrid.log'):
        _, results = parse_nccl_log_file(os.path.join(here, name))
        r = results[0]
        assert abs(r['busbw'] - r['algbw'] * 1.75) < 0.05, name


def test_nvlink_throughput_consistency():
    """验证消息大小不影响 busbw（只影响时间）。"""
    result1 = ring_allreduce(8, 450, msg_size_bytes=1e9)
    result2 = ring_allreduce(8, 450, msg_size_bytes=2e9)

    # busbw（单位 GB/s）应相同
    assert abs(result1['busbw_gbs'] - result2['busbw_gbs']) < 0.01, \
        f'不同消息大小下 busbw 应相同：{result1["busbw_gbs"]:.1f} vs {result2["busbw_gbs"]:.1f}'

    # 时间应按消息大小比例缩放
    time_ratio = result2['time_us'] / result1['time_us']
    assert abs(time_ratio - 2.0) < 0.01, f'时间比应为 2.0，得 {time_ratio:.2f}'


def run_all_tests():
    """运行所有测试，汇总结果。"""
    tests = [
        ('NVLink4 规格值 (H100)', test_nvlink_specs_h100),
        ('NVLink5 规格值 (B200)', test_nvlink5_specs_b200),
        ('Ring all-reduce 8 卡理论值 (H100)', test_ring_allreduce_8gpu_h100),
        ('Ring 性能随 GPU 数单调增', test_ring_allreduce_monotonicity),
        ('Ring 延迟项', test_ring_latency_term),
        ('不同代数可扩展性', test_ring_allreduce_scalability),
        ('Topo 矩阵解析', test_topo_matrix_parse),
        ('NCCL 日志解析', test_nccl_log_parse),
        ('NCCL 结果判定', test_judge_result),
        ('示例日志自洽', test_sample_logs),
        ('消息大小 vs busbw', test_nvlink_throughput_consistency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f'✓ {test_name}')
            passed += 1
        except AssertionError as e:
            print(f'✗ {test_name}')
            print(f'  {e}')
            failed += 1
        except Exception as e:
            print(f'✗ {test_name} (异常)')
            print(f'  {e}')
            failed += 1

    print()
    print(f'测试结果：{passed} 通过，{failed} 失败（共 {len(tests)} 项）')
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
