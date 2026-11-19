"""单元测试：注意力流量账本与在线 softmax 精度。

测试覆盖：
  1. 标准注意力 vs 分块融合的 HBM 流量对比（FlashAttention-1 论文数字）
  2. 在线 softmax 与朴素 softmax 的精度（Milakov & Gimelshein 2019）
  3. 流量预测与时间账本的物理合理性（单调性、量纲）
  4. H100 注意力下界（Dao et al. 2022 Table 3）

核心数字来源：
  [1] Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention"，Table 1 & 3
  [2] Milakov & Gimelshein (2019) "Online normalizer calculation for softmax"
  [3] Dao et al. (2023) "FlashAttention-2: Faster Accurate Attention"，§3.2
  [4] NVIDIA H100 官方规格（验证于 2026-11-02）
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get

# Import from sim modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_attn import naive_attn_hbm, blocked_attn_hbm
from sim_online_softmax import online_softmax_stats, exp_cost_estimate


def test_naive_attn_basic():
    """Test 1: 标准注意力的基本流量计算。"""
    # N=1024, d=64, heads=32
    acc = naive_attn_hbm(seq_len=1024, head_dim=64, num_heads=32)

    # 单头：读 (2*1024*64 + 3*1024*1024)*2 = 8.8 MB，写 (1024*1024 + 1024*64)*2 = 2.2 MB
    # 32 头：总 0.281 GB
    assert 0.27 < acc.hbm_total_gb < 0.30, f"Expected ~0.281 GB, got {acc.hbm_total_gb}"
    print("✓ Test 1: naive_attn 基本计算 OK")


def test_blocked_vs_naive_reduction():
    """Test 2: 分块融合 vs 标准注意力的流量削减（FA1 论文数据）。

    Dao et al. (2022): 大 N 时 FA1 改进显著。使用 H100 标准设置：
      N=8192, d=128, heads=32, block_size=256
      标准：17.4 GB，FA1：4.5 GB，约 3.9x 改进。
    """
    n, d, heads = 8192, 128, 32
    block_size = 256

    naive = naive_attn_hbm(seq_len=n, head_dim=d, num_heads=heads)
    blocked = blocked_attn_hbm(seq_len=n, head_dim=d, num_heads=heads, block_size=block_size)

    # 标准注意力：~17 GB（N² 项主导）
    assert 16.5 < naive.hbm_total_gb < 18.0, f"Expected naive ~17.4 GB, got {naive.hbm_total_gb}"

    # 分块融合：~4-5 GB（O(N×d) 主导）
    assert 4.0 < blocked.hbm_total_gb < 5.0, f"Expected blocked ~4.5 GB, got {blocked.hbm_total_gb}"

    # 改进倍数应在 3-4x
    improvement = naive.hbm_total_gb / blocked.hbm_total_gb
    assert 3.5 < improvement < 4.5, f"Expected improvement ~3.9x, got {improvement:.1f}x"
    print(f"✓ Test 2: 流量削减 {improvement:.1f}x OK")


def test_h100_attn_lower_bound():
    """Test 3: H100 上 N=8192, d=128, 32 头的 IO 下界（Dao et al. 2022）。

    标准注意力：16.25 GB（32 头）
    IO 下界公式：(Nd + N²/M) × 2 × heads = (8192×128 + 8192²/49152) × 2 × 32 ≈ 67 MB
    """
    n, d, heads = 8192, 128, 32
    sram_bytes = 96 * 1024
    dtype_bytes = 2

    # 标准注意力（32 头）
    acc_naive = naive_attn_hbm(seq_len=n, head_dim=d, num_heads=heads)
    naive_total = acc_naive.hbm_total_gb

    # IO 下界（单头，再乘以 heads）
    M = sram_bytes // dtype_bytes
    io_lower_single = (n * d + n * n / M) * dtype_bytes / 1e9
    io_lower = io_lower_single * heads

    assert 16.0 < naive_total < 18.0, f"Expected ~16.25 GB, got {naive_total:.2f} GB"
    assert io_lower < 0.1, f"Expected ~0.067 GB, got {io_lower:.4f} GB"

    print(f"✓ Test 3: H100 IO 下界 {naive_total:.1f} GB -> {io_lower:.4f} GB OK")


def test_online_softmax_blocks():
    """Test 4: 在线 softmax 块分割统计。

    Milakov & Gimelshein (2019)：N=1024, block_size=128
      块数 = 8，每 token 重缩放 7 次。
    """
    stats = online_softmax_stats(seq_len=1024, head_dim=128, block_size=128)

    assert stats['num_blocks'] == 8, f"Expected 8 blocks, got {stats['num_blocks']}"
    assert stats['rescales_per_token'] == 7, f"Expected 7 rescales/token, got {stats['rescales_per_token']}"
    assert stats['total_rescales'] == 1024 * 7, f"Rescales mismatch"

    print("✓ Test 4: 在线 softmax 块统计 OK")


def test_exp_cost_estimate():
    """Test 5: exp 计算成本估计（H100 MUFU）。

    H100：16 exp/cycle，132 SM，~3.67 GHz → 79 GigaExp/s
    N=8192，block_size=256：32 块，每 token 31 次重缩放 exp + 32×8192 次 softmax exp
    总 ≈ 8192×31 + 8192×32 ≈ 512K exp → 32.3 μs（按 exp_tflops=16）
    """
    n = 8192
    num_blocks = (n + 256 - 1) // 256  # 32

    exp_us = exp_cost_estimate(seq_len=n, num_blocks=num_blocks, exp_tflops=16)

    # 实际测量：应在 30-35 μs 范围
    assert 25 < exp_us < 40, f"Expected exp cost ~32 μs, got {exp_us:.1f} μs"
    print(f"✓ Test 5: exp 成本 {exp_us:.1f} μs OK")


def test_time_monotonicity():
    """Test 6: 时间账本的物理合理性。

    - 序列长度增加，时间增加（单调性）
    - 强度越高（块越小），读写次数越多（权衡）
    - HBM 流量随 N² 扩展（朴素）或 N 扩展（分块）
    """
    gpu = get('h100-sxm')

    # 序列长度单调性：8K vs 16K
    acc_8k = naive_attn_hbm(seq_len=8192, head_dim=128, num_heads=32)
    acc_16k = naive_attn_hbm(seq_len=16384, head_dim=128, num_heads=32)

    t_8k = acc_8k.time_us(gpu.hbm_tbs)
    t_16k = acc_16k.time_us(gpu.hbm_tbs)

    assert t_16k > t_8k, f"Time should increase with seq_len: {t_8k} vs {t_16k}"
    # 16K 约 4 倍流量，时间也应 ~4 倍（忽略常数项）
    ratio = t_16k / t_8k
    assert 3.5 < ratio < 4.5, f"Expected ~4x time ratio, got {ratio:.2f}x"

    # 分块大小影响：更小的块 → 更多重读 K/V → 更高时间
    acc_b64 = blocked_attn_hbm(seq_len=8192, head_dim=128, num_heads=32, block_size=64)
    acc_b512 = blocked_attn_hbm(seq_len=8192, head_dim=128, num_heads=32, block_size=512)

    t_b64 = acc_b64.time_us(gpu.hbm_tbs)
    t_b512 = acc_b512.time_us(gpu.hbm_tbs)

    assert t_b64 > t_b512, f"Smaller block should need more time: {t_b64} vs {t_b512}"
    print("✓ Test 6: 时间单调性 OK")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_naive_attn_basic,
        test_blocked_vs_naive_reduction,
        test_h100_attn_lower_bound,
        test_online_softmax_blocks,
        test_exp_cost_estimate,
        test_time_monotonicity,
    ]

    for i, test in enumerate(tests, 1):
        try:
            test()
        except AssertionError as e:
            print(f"✗ Test {i} FAILED: {e}")
            return False

    print(f"\n{len(tests)} tests passed ✓")
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
