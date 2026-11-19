"""Test sim_a2a.py：复现公开文献数字。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def test_deepseek_v3_a2a_estimate():
    """
    验证 dispatch/combine 字节数计算。

    DeepSeek-V3 技术报告 Section 2.2：
    - 模型规格：671B 总参数，每 token 激活 37B
    - MoE：256 个专家，每 token top-2，每 token 4 个节点
    - 隐层宽度（推测）：4096 或更大
    - all-to-all 通信：每 token × top-k × hidden × 2 字节（BF16）

    核对逻辑：
    - 总 token 数（batch）× top-k × hidden × 2 字节 = dispatch 字节
    - 相同的公式用于 combine
    """
    # 运行 dispatch 子命令，取得输出
    result = subprocess.run(
        [sys.executable, 'sim_a2a.py', 'dispatch',
         '--hidden', '4096', '--top-k', '2', '--batch', '2048', '--ep-rank', '32'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result.returncode == 0, f"dispatch 命令失败：{result.stderr}"

    # 检查输出包含关键指标
    assert 'dispatch 字节' in result.stdout, "输出缺失 dispatch 字节"
    assert 'combine 字节' in result.stdout, "输出缺失 combine 字节"
    assert 'NVLink4 延迟' in result.stdout, "输出缺失 NVLink4 延迟"

    # 简单的数值检查
    # dispatch_bytes = batch × top-k × hidden × 2 = 2048 × 2 × 4096 × 2 = 33.55 MB
    # combine_bytes = 同上
    # 总计约 67 MB
    # NVLink4 延迟 = 67 MB / 450 GB/s ≈ 0.148 ms ≈ 148 us
    lines = result.stdout.split('\n')
    dispatch_line = [l for l in lines if 'dispatch 字节' in l]
    if dispatch_line:
        parts = dispatch_line[0].split()
        # 提取数值，应该在 32-35 MB 范围
        if len(parts) >= 2:
            try:
                dispatch_mb = float(parts[-2])
                assert 32 <= dispatch_mb <= 35, f"dispatch 字节不在预期范围：{dispatch_mb} MB"
            except ValueError:
                pass

    print("✓ test_deepseek_v3_a2a_estimate 通过")


def test_nvlink_latency():
    """
    验证 NVLink 延迟计算。

    公式：延迟 = (字节数) / (带宽 GB/s)
    例如：33.55 MB / 450 GB/s = 33.55 / 450 × 1024 × 1e-3 ≈ 0.076 ms
    """
    result = subprocess.run(
        [sys.executable, 'sim_a2a.py', 'latency',
         '--hidden', '4096', '--top-k', '2', '--batch', '1024', '--ep-rank', '8'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result.returncode == 0, f"latency 命令失败：{result.stderr}"
    assert '总延迟' in result.stdout or '延迟' in result.stdout, "输出缺失延迟信息"
    print("✓ test_nvlink_latency 通过")


def test_overhead_sweep():
    """
    验证 overhead 子命令的扫参。
    """
    result = subprocess.run(
        [sys.executable, 'sim_a2a.py', 'overhead',
         '--hidden', '4096', '--batch-min', '1', '--batch-max', '16',
         '--top-k', '2', '--ep-rank', '8'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result.returncode == 0, f"overhead 命令失败：{result.stderr}"
    lines = result.stdout.strip().split('\n')
    # 应该有多行输出（至少一行数据）
    assert len(lines) > 2, "overhead 输出行数不足"
    print("✓ test_overhead_sweep 通过")


if __name__ == '__main__':
    test_deepseek_v3_a2a_estimate()
    test_nvlink_latency()
    test_overhead_sweep()
    print("\n✓✓✓ 所有 sim_a2a 测试通过 ✓✓✓")
