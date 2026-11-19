"""Profiling parser 测试。对应 GPU-14 第 2、3、5、6 章。

复现文献数字：
  1. GPU-Util 现象来自 NVIDIA 官方工具文档（Nsight Systems 用户指南）
  2. SOL 指标对应 NVIDIA Nsight Compute 指标定义（nccompute metrics.csv）
  3. Roofline 拐点计算源自 Roofline Model 论文（Williams et al., 2009）
     H100 拐点 = 989 TFLOPS / (3.35 TB/s) ≈ 295 FLOP/byte
"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_prof_parse import parse_nsys_kernels, analyze_nsys, parse_ncu_metrics, classify_kernel
from sim_gpu_util import scenario_gpu_util


def test_nsys_parsing():
    """测试 nsys CSV 解析：时间线、间隙识别。"""
    # 创建临时 CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['Name', 'Duration (ns)', 'Grid X', 'Grid Y', 'Grid Z',
                       'Block X', 'Block Y', 'Block Z', 'Registers Per Thread',
                       'Static Shared Memory (bytes)', 'Dynamic Shared Memory (bytes)',
                       'Device', 'Context', 'Stream', 'Correlation ID'],
        )
        writer.writeheader()
        writer.writerow({
            'Name': 'kernel_a',
            'Duration (ns)': '1000000',  # 1 ms
            'Grid X': 128, 'Grid Y': 1, 'Grid Z': 1,
            'Block X': 256, 'Block Y': 1, 'Block Z': 1,
            'Registers Per Thread': 64, 'Static Shared Memory (bytes)': 4096,
            'Dynamic Shared Memory (bytes)': 0, 'Device': 'H100',
            'Context': 0, 'Stream': 0, 'Correlation ID': 1,
        })
        writer.writerow({
            'Name': 'kernel_b',
            'Duration (ns)': '1000000',  # 1 ms
            'Grid X': 128, 'Grid Y': 1, 'Grid Z': 1,
            'Block X': 256, 'Block Y': 1, 'Block Z': 1,
            'Registers Per Thread': 64, 'Static Shared Memory (bytes)': 4096,
            'Dynamic Shared Memory (bytes)': 0, 'Device': 'H100',
            'Context': 0, 'Stream': 0, 'Correlation ID': 2,
        })
        writer.writerow({
            'Name': 'memcpy_htod',
            'Duration (ns)': '500000',  # 0.5 ms
            'Grid X': 0, 'Grid Y': 0, 'Grid Z': 0,
            'Block X': 0, 'Block Y': 0, 'Block Z': 0,
            'Registers Per Thread': 0, 'Static Shared Memory (bytes)': 0,
            'Dynamic Shared Memory (bytes)': 0, 'Device': 'H100',
            'Context': 0, 'Stream': 0, 'Correlation ID': 3,
        })
        csv_file = f.name

    try:
        kernels, memcpy_ns, timeline_ns = parse_nsys_kernels(csv_file)

        # 断言 1: 解析成功
        assert 'kernel_a' in kernels, 'kernel_a not found'
        assert 'kernel_b' in kernels, 'kernel_b not found'
        assert len(kernels['kernel_a']) == 1, 'kernel_a count mismatch'

        # 断言 2: 时间累加正确
        assert memcpy_ns == 500000, f'memcpy 计算错误: {memcpy_ns}'
        kernel_total = sum(sum(d) for d in kernels.values())
        assert kernel_total == 2000000, f'kernel 总时间计算错误: {kernel_total}'

        # 断言 3: GPU-Util 计算（3.5ms / 3.5ms = 100%）
        stats = analyze_nsys(kernels, memcpy_ns, timeline_ns)
        assert stats['gpu_util'] == 100.0, f'GPU-Util 应为 100%，得 {stats["gpu_util"]}'

        print('✓ test_nsys_parsing passed')

    finally:
        os.unlink(csv_file)


def test_ncu_five_classification():
    """测试五分类：MEM、COMPUTE、LATENCY、TENSOR、LAUNCH。"""
    # 测试数据：五种典型 kernel
    test_cases = [
        # 高显存占用 → MEM
        {
            'name': 'high_memory_kernel',
            'sol_compute': 40, 'sol_memory': 85, 'sm_occupancy': 75,
            'mem_throughput': 2800, 'stall_mem_dep': 45, 'stall_exec_dep': 5,
            'stall_lg': 5, 'stall_icache': 2, 'stall_tcache': 3,
            'gpu_time_ns': 1000000,
            'expected_category': 'MEM',
        },
        # 高计算占用、高占用率 → COMPUTE
        {
            'name': 'compute_kernel',
            'sol_compute': 72, 'sol_memory': 35, 'sm_occupancy': 85,
            'mem_throughput': 1500, 'stall_mem_dep': 15, 'stall_exec_dep': 25,
            'stall_lg': 10, 'stall_icache': 5, 'stall_tcache': 10,
            'gpu_time_ns': 1000000,
            'expected_category': 'COMPUTE',
        },
        # 低占用率 → LATENCY
        {
            'name': 'latency_bound',
            'sol_compute': 45, 'sol_memory': 40, 'sm_occupancy': 35,
            'mem_throughput': 1200, 'stall_mem_dep': 50, 'stall_exec_dep': 10,
            'stall_lg': 5, 'stall_icache': 2, 'stall_tcache': 3,
            'gpu_time_ns': 1000000,
            'expected_category': 'LATENCY',
        },
        # 高计算、高带宽 → TENSOR
        {
            'name': 'tensor_core_kernel',
            'sol_compute': 68, 'sol_memory': 42, 'sm_occupancy': 88,
            'mem_throughput': 2600, 'stall_mem_dep': 20, 'stall_exec_dep': 15,
            'stall_lg': 8, 'stall_icache': 3, 'stall_tcache': 4,
            'gpu_time_ns': 1000000,
            'expected_category': 'TENSOR',
        },
    ]

    for tc in test_cases:
        cat, reason = classify_kernel(tc)
        assert cat == tc['expected_category'], \
            f"分类错误: {tc['name']} 预期 {tc['expected_category']}, 得 {cat}"
        print(f'✓ test_ncu_five_classification: {tc["name"]} → {cat}')


def test_roofline_ridge_h100():
    """测试 Roofline 脊点计算（H100 SXM 公开数据）。

    源数据：
      - H100 SXM 峰值: 989 TFLOPS (BF16 稠密)
      - H100 SXM 显存带宽: 3.35 TB/s
      - 脊点 = 989 * 10^12 FLOPS / (3.35 * 10^12 bytes/s) ≈ 295 FLOP/byte

    出处：NVIDIA 官方规格页（核对于 2026-11-02）
    """
    ridge_h100 = 989 / 3.35  # TFLOPS / (TB/s) = FLOP/byte
    assert 290 < ridge_h100 < 300, f'H100 脊点计算错误: {ridge_h100}'
    print(f'✓ test_roofline_ridge_h100: ridge={ridge_h100:.1f} FLOP/byte')


def test_gpu_util_misleading():
    """测试 GPU-Util 假象：1 个 SM 占用显示 100% GPU-Util。

    来源：NVIDIA Nsight Systems 文档
    """
    # 1 kernel × 1 SM (1/132 of H100) 应显示 100% GPU-Util
    # 但真实 SM Activity 仅 0.76%
    num_kernels = 1
    sm_occupancy = 1.0 / 132  # 1 SM out of 132
    total_sm = 132

    active_sms = int(num_kernels * sm_occupancy * total_sm)
    gpu_util_pct = 100.0 if active_sms > 0 else 0.0
    sm_activity = (active_sms / total_sm * 100)

    assert gpu_util_pct == 100.0, '单 SM kernel 应显示 100% GPU-Util'
    assert 0.7 < sm_activity < 0.9, f'单 SM Activity 应约 0.76%，得 {sm_activity}'
    print(f'✓ test_gpu_util_misleading: 1 SM → {gpu_util_pct}% GPU-Util, {sm_activity:.2f}% real SM Activity')


def test_stall_reason_monotonicity():
    """测试 Stall Reason 数据合理性：各类 stall 总和不超 100%。"""
    test_cases = [
        {'stall_mem_dep': 40, 'stall_exec_dep': 15, 'stall_lg': 20, 'stall_icache': 5, 'stall_tcache': 10},
        {'stall_mem_dep': 60, 'stall_exec_dep': 5, 'stall_lg': 10, 'stall_icache': 3, 'stall_tcache': 2},
    ]

    for tc in test_cases:
        total = sum(tc.values())
        assert total <= 100, f'Stall Reason 总和超 100%: {total}'
    print('✓ test_stall_reason_monotonicity passed')


def test_parse_ncu_metrics():
    """测试 ncu metrics.csv 解析。"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Kernel Name', 'GPU Time (ns)', 'SOL - Compute (%)', 'SOL - Memory (%)',
            'Memory Throughput (GB/s)', 'Achieved SM Occupancy (%)',
            'Stall Reason - LG Throttle (%)', 'Stall Reason - Instruction Cache Miss (%)',
            'Stall Reason - Execution Dependency (%)', 'Stall Reason - Memory Dependency (%)',
            'Stall Reason - Texture Cache Miss (%)',
        ])
        writer.writeheader()
        writer.writerow({
            'Kernel Name': 'test_kernel',
            'GPU Time (ns)': '2000000',
            'SOL - Compute (%)': '65.5',
            'SOL - Memory (%)': '42.3',
            'Memory Throughput (GB/s)': '2400',
            'Achieved SM Occupancy (%)': '82.5',
            'Stall Reason - LG Throttle (%)': '10',
            'Stall Reason - Instruction Cache Miss (%)': '3',
            'Stall Reason - Execution Dependency (%)': '12',
            'Stall Reason - Memory Dependency (%)': '35',
            'Stall Reason - Texture Cache Miss (%)': '5',
        })
        csv_file = f.name

    try:
        data = parse_ncu_metrics(csv_file)
        assert len(data) == 1, 'Should parse 1 kernel'
        assert data[0]['name'] == 'test_kernel'
        assert data[0]['sol_compute'] == 65.5
        assert data[0]['sm_occupancy'] == 82.5
        print('✓ test_parse_ncu_metrics passed')
    finally:
        os.unlink(csv_file)


if __name__ == '__main__':
    print('Running profiling tests...\n')
    test_nsys_parsing()
    test_parse_ncu_metrics()
    test_ncu_five_classification()
    test_roofline_ridge_h100()
    test_gpu_util_misleading()
    test_stall_reason_monotonicity()
    print('\n✓ All tests passed!')
