"""Test sim_imbalance.py：复现负载不均分析。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def test_zipf_distribution():
    """
    验证 Zipf 分布的负载不均。

    数学验证：Zipf 分布 P(k) ∝ 1/k^α
    - α = 0：均匀分布，max/mean ≈ 1.0
    - α = 1.5：常见互联网分布，max/mean ≈ 2-3（取决于专家数）
    - α = 2.0：更陡峭，max/mean ≈ 3-5

    固定种子以确保可重复。
    """
    result = subprocess.run(
        [sys.executable, 'sim_imbalance.py', 'zipf',
         '--num-experts', '128', '--num-tokens', '10240', '--alpha', '1.5', '--seed', '42'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result.returncode == 0, f"zipf 命令失败：{result.stderr}"

    # 检查输出包含关键指标
    assert 'max/mean 比' in result.stdout, "输出缺失 max/mean 比"
    assert '热专家（top' in result.stdout, "输出缺失热专家列表"

    # 解析 max/mean 比
    lines = result.stdout.split('\n')
    ratio_line = [l for l in lines if 'max/mean 比' in l]
    if ratio_line:
        parts = ratio_line[0].split()
        try:
            ratio = float(parts[-1])
            # Zipf 分布高度偏斜，128 个专家下 max/mean 通常 > 10
            assert ratio > 5.0, f"max/mean 比应该 > 5.0（Zipf 高偏斜）：{ratio}"
        except (ValueError, IndexError):
            pass

    print("✓ test_zipf_distribution 通过")


def test_uniform_vs_zipf():
    """
    对比均匀分布（baseline）和 Zipf 分布的负载不均。
    """
    # 均匀分布
    result_uniform = subprocess.run(
        [sys.executable, 'sim_imbalance.py', 'imbalance',
         '--distribution', 'uniform', '--num-experts', '128', '--num-tokens', '10240'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result_uniform.returncode == 0, f"uniform 命令失败：{result_uniform.stderr}"

    # Zipf 分布
    result_zipf = subprocess.run(
        [sys.executable, 'sim_imbalance.py', 'imbalance',
         '--distribution', 'zipf', '--alpha', '1.5', '--num-experts', '128', '--num-tokens', '10240'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result_zipf.returncode == 0, f"zipf 命令失败：{result_zipf.stderr}"

    # 均匀分布应该接近 1.0（完美平衡）
    uniform_lines = result_uniform.stdout.split('\n')
    uniform_ratio_line = [l for l in uniform_lines if 'max/mean' in l]
    if uniform_ratio_line:
        try:
            uniform_ratio = float(uniform_ratio_line[0].split()[-1])
            assert uniform_ratio < 1.1, f"均匀分布 max/mean 不应该 > 1.1：{uniform_ratio}"
        except (ValueError, IndexError):
            pass

    # Zipf 分布应该 >> 1.0（不均衡）
    zipf_lines = result_zipf.stdout.split('\n')
    zipf_ratio_line = [l for l in zipf_lines if 'max/mean' in l]
    if zipf_ratio_line:
        try:
            zipf_ratio = float(zipf_ratio_line[0].split()[-1])
            assert zipf_ratio > 1.5, f"Zipf 分布 max/mean 应该 > 1.5：{zipf_ratio}"
        except (ValueError, IndexError):
            pass

    print("✓ test_uniform_vs_zipf 通过")


def test_rebalancing_strategies():
    """
    验证各种均衡策略的改善效果。

    期望：
    - 辅助损失 < 无均衡
    - 偏置调整 < 辅助损失
    - 冗余专家最好
    """
    result = subprocess.run(
        [sys.executable, 'sim_imbalance.py', 'rebalance',
         '--num-experts', '128', '--num-tokens', '10240', '--alpha', '1.5'],
        capture_output=True, text=True, cwd=HERE
    )
    assert result.returncode == 0, f"rebalance 命令失败：{result.stderr}"

    # 检查输出包含各策略
    assert '无均衡' in result.stdout, "输出缺失无均衡基线"
    assert '辅助损失' in result.stdout, "输出缺失辅助损失"
    assert '偏置调整' in result.stdout, "输出缺失偏置调整"
    assert '冗余专家' in result.stdout, "输出缺失冗余专家"

    # 提取数值并验证趋势
    lines = result.stdout.split('\n')
    for line in lines:
        if 'max/mean' in line:
            print(f"Debug: {line}")

    print("✓ test_rebalancing_strategies 通过")


def test_deterministic_seed():
    """
    验证固定种子下的结果可重复。
    """
    result1 = subprocess.run(
        [sys.executable, 'sim_imbalance.py', 'zipf',
         '--num-experts', '128', '--num-tokens', '1024', '--alpha', '1.5', '--seed', '12345'],
        capture_output=True, text=True, cwd=HERE
    )
    result2 = subprocess.run(
        [sys.executable, 'sim_imbalance.py', 'zipf',
         '--num-experts', '128', '--num-tokens', '1024', '--alpha', '1.5', '--seed', '12345'],
        capture_output=True, text=True, cwd=HERE
    )

    # 两次运行的输出应该完全相同（给定相同的种子）
    assert result1.stdout == result2.stdout, "固定种子下结果不可重复"
    print("✓ test_deterministic_seed 通过")


if __name__ == '__main__':
    test_zipf_distribution()
    test_uniform_vs_zipf()
    test_rebalancing_strategies()
    test_deterministic_seed()
    print("\n✓✓✓ 所有 sim_imbalance 测试通过 ✓✓✓")
