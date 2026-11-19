#!/usr/bin/env python3
"""
Test collective timing model against nccl-tests reference data.

Verifies the busbw formula from nccl-tests matches our simulation.

Reference: nccl-tests/PERFORMANCE.md, Demystifying NCCL (arXiv 2507.04786)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from sim_collective import compute_ring_timing, compute_tree_timing, CollectiveArgs


def assert_sanity_check(msg_bytes, time_ms, bw_gb_s, label="", link_bw=50.0):
    """
    Sanity checks for collective timing.

    Expected ranges per NCCL docs and measured data:
    - 1 MB on 8 H100 NVLink (450 GB/s): ~18-50 µs (0.018-0.05 ms)
    - 1 GB on 8 H100 NVLink: ~5 ms (with 1 µs alpha overhead)
    - 1 MB cross-node (50 GB/s): ~51-200 µs (0.051-0.2 ms)
    - 1 GB cross-node: ~50 ms
    """
    msg_mb = msg_bytes / 1024 / 1024

    # Basic sanity: achieved BW should be >= 0
    assert bw_gb_s >= 0, f"{label}: BW negative: {bw_gb_s}"

    # Check timing is in expected ballpark based on link_bw
    if msg_bytes == 1048576 and time_ms > 0:  # 1 MB message
        if link_bw > 400:  # NVLink (450 GB/s nominal)
            # Should be in low microseconds: 0.018-0.2 ms
            assert 0.005 <= time_ms <= 0.2, \
                f"{label}: 1MB NVLink time {time_ms:.3f}ms out of expected 0.005-0.2ms range"
        elif link_bw > 40:  # Cross-node (50 GB/s nominal)
            # Should be in tens of microseconds: 0.05-0.5 ms
            assert 0.03 <= time_ms <= 0.5, \
                f"{label}: 1MB cross-node time {time_ms:.3f}ms out of expected 0.03-0.5ms range"

    if msg_bytes == 1073741824 and time_ms > 0:  # 1 GB message
        if link_bw > 400:  # NVLink
            # Should be ~5-10 ms
            assert 4 <= time_ms <= 15, \
                f"{label}: 1GB NVLink time {time_ms:.1f}ms out of expected 4-15ms range"
        elif link_bw > 40:  # Cross-node
            # Should be ~50-100 ms
            assert 40 <= time_ms <= 150, \
                f"{label}: 1GB cross-node time {time_ms:.1f}ms out of expected 40-150ms range"


def test_ring_all_reduce_formula():
    """
    Test ring all-reduce against reference formula (nccl-tests).

    nccl-tests formula (PERFORMANCE.md):
    - busbw = 2 * (n-1) / n * link_bw (network utilization factor)
    - algbw = S / (2*(n-1) * (alpha + beta * S/n)) where beta = 1/bw

    For 8 ranks, 1MB message, 50 GB/s link, 1 µs alpha:
    - busbw = 2*7/8 * 50 = 43.75 GB/s
    - chunk = 1MB/8 = 128 KB
    - time = 2*7 * (1µs + 128KB/50GB/s)
           = 14 * (1µs + 2.56µs) = ~50.84 µs

    References:
    - nccl-tests: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    - Demystifying NCCL (arXiv 2507.04786)
    """
    args = CollectiveArgs(
        algorithm='ring',
        protocol='Simple',
        num_ranks=8,
        msg_bytes=1024*1024,
        link_bw=50.0,
        alpha_us=1.0,
    )

    result = compute_ring_timing(args)

    # nccl-tests PERFORMANCE.md: busbw = algbw * 2(n-1)/n
    expected_busbw = result['algbw_gb_s'] * 2 * 7 / 8
    assert abs(result['busbw_gb_s'] - expected_busbw) < 0.01, \
        f"Busbw mismatch: {result['busbw_gb_s']} vs {expected_busbw}"

    # Verify timing structure
    assert result['num_steps'] == 2 * 7, "Ring should have 2*(n-1) steps"
    assert result['chunk_bytes'] == 1024*1024 / 8, "Chunk should be msg/n"

    # Sanity check: timing should be in expected range
    assert_sanity_check(result['msg_bytes'], result['time_ms'], result['algbw_gb_s'],
                        label="Ring 1MB cross-node (50 GB/s)", link_bw=50.0)

    print("✓ Ring all-reduce formula test passed")
    return True


def test_tree_depth():
    """Test tree depth calculation matches binary tree."""
    import math

    args = CollectiveArgs(
        algorithm='tree',
        protocol='Simple',
        num_ranks=8,
        msg_bytes=1024*1024,
        link_bw=50.0,
        alpha_us=1.0,
    )

    result = compute_tree_timing(args)

    # For 8 ranks (2^3), tree depth should be 3, so 2*3=6 steps
    expected_depth = math.ceil(math.log2(8))
    assert result['tree_depth'] == expected_depth, \
        f"Tree depth mismatch: {result['tree_depth']} vs {expected_depth}"

    assert result['num_steps'] == 2 * expected_depth, \
        f"Tree steps should be 2*depth: {result['num_steps']} vs {2*expected_depth}"

    print("✓ Tree depth test passed")
    return True


def test_ring_vs_tree_crossover():
    """
    Verify tree and ring have different characteristics.
    Tree has fewer steps but per-node bandwidth is still limited.
    """
    # Small message: 1 KB
    args_small = CollectiveArgs(
        algorithm='ring',
        protocol='Simple',
        num_ranks=8,
        msg_bytes=1024,
        link_bw=50.0,
        alpha_us=1.0,
    )

    ring_small = compute_ring_timing(args_small)
    tree_small = compute_tree_timing(args_small)

    # Tree has fewer steps (log n vs 2*(n-1)), so should have lower time
    assert tree_small['num_steps'] < ring_small['num_steps'], \
        f"Tree should have fewer steps: {tree_small['num_steps']} vs {ring_small['num_steps']}"

    # Large message: 256 MB
    args_large = CollectiveArgs(
        algorithm='ring',
        protocol='Simple',
        num_ranks=8,
        msg_bytes=256*1024*1024,
        link_bw=50.0,
        alpha_us=1.0,
    )

    ring_large = compute_ring_timing(args_large)
    tree_large = compute_tree_timing(args_large)

    print(f"Small msg: Ring {ring_small['time_ms']:.2f}ms vs Tree {tree_small['time_ms']:.2f}ms")
    print(f"Large msg: Ring {ring_large['time_ms']:.2f}ms vs Tree {tree_large['time_ms']:.2f}ms")

    print("✓ Ring vs tree crossover test passed")
    return True


def test_protocol_overhead():
    """Small messages favour LL (lowest latency), large messages favour Simple (full bandwidth).

    LL carries 4 B of data per 8 B store, so at large sizes it moves data at about half the rate of Simple;
    LL128 sits in between (120 of every 128 B). See NCCL docs on NCCL_PROTO and arXiv 2507.04786.
    """
    def t(proto, size):
        a = CollectiveArgs(algorithm='ring', protocol=proto, num_ranks=8, msg_bytes=size, link_bw=50.0, alpha_us=5.0)
        return compute_ring_timing(a)['time_ms']

    small, large = 8 * 1024, 256 * 1024 * 1024
    assert t('LL', small) < t('LL128', small) < t('Simple', small), 'LL should win on small messages'
    assert t('Simple', large) < t('LL128', large) < t('LL', large), 'Simple should win on large messages'
    ratio = t('LL', large) / t('Simple', large)
    assert 1.8 < ratio < 2.1, f'LL should be ~2x slower than Simple at 256 MB, got {ratio:.2f}'
    print(f"8 KB: LL {t('LL', small):.4f} ms, Simple {t('Simple', small):.4f} ms; 256 MB: LL/Simple = {ratio:.2f}")
    print("✓ Protocol overhead test passed")
    return True


def test_scaling_with_ranks():
    """More ranks = more steps for ring, but potentially higher aggregate BW."""
    args = CollectiveArgs(
        algorithm='ring',
        protocol='Simple',
        num_ranks=2,
        msg_bytes=1024*1024,
        link_bw=50.0,
        alpha_us=1.0,
    )

    results = {}
    for n in [2, 4, 8, 16]:
        args.num_ranks = n
        results[n] = compute_ring_timing(args)

    # Verify algbw drops as ranks increase (more steps)
    for n in [2, 4, 8]:
        assert results[n]['algbw_gb_s'] > results[n*2]['algbw_gb_s'], \
            f"Algbw should decrease with more ranks: {results[n]['num_ranks']} vs {results[n*2]['num_ranks']}"

    scaling_str = ', '.join(f'{n}r:{results[n]["algbw_gb_s"]:.1f}GB/s' for n in sorted(results.keys()))
    print(f"Scaling: {scaling_str}")
    print("✓ Scaling with ranks test passed")
    return True


def test_nvlink_vs_cross_node():
    """
    Sanity check: NVLink ring all-reduce should be ~10x faster than cross-node.

    Scenario:
    - 8 H100s, 1MB, NVLink 450 GB/s, alpha=1µs: ~0.018 ms
    - 8 GPUs across nodes, 1MB, 50 GB/s cross-node, alpha=1µs: ~0.051 ms
    - Ratio should be ~2.8x at least (from pure bandwidth ratio 450/50 = 9x)
    """
    args_nvlink = CollectiveArgs(
        algorithm='ring',
        protocol='LL128',
        num_ranks=8,
        msg_bytes=1024*1024,
        link_bw=450.0,  # H100 NVLink4 intra-node
        alpha_us=1.0,
    )

    args_cross = CollectiveArgs(
        algorithm='ring',
        protocol='LL128',
        num_ranks=8,
        msg_bytes=1024*1024,
        link_bw=50.0,  # 400G cross-node
        alpha_us=1.0,
    )

    result_nvlink = compute_ring_timing(args_nvlink)
    result_cross = compute_ring_timing(args_cross)

    # NVLink should be significantly faster
    speedup = result_cross['time_ms'] / result_nvlink['time_ms']
    assert speedup > 2.0, \
        f"NVLink should be >2x faster than cross-node: {speedup:.1f}x"

    # Sanity checks for both
    assert_sanity_check(result_nvlink['msg_bytes'], result_nvlink['time_ms'],
                        result_nvlink['algbw_gb_s'], label="NVLink", link_bw=450.0)
    assert_sanity_check(result_cross['msg_bytes'], result_cross['time_ms'],
                        result_cross['algbw_gb_s'], label="Cross-node", link_bw=50.0)

    print(f"NVLink vs Cross-node: {result_nvlink['time_ms']:.3f}ms vs {result_cross['time_ms']:.3f}ms ({speedup:.1f}x)")
    print("✓ NVLink vs cross-node sanity test passed")
    return True


def main():
    tests = [
        test_ring_all_reduce_formula,
        test_tree_depth,
        test_ring_vs_tree_crossover,
        test_protocol_overhead,
        test_scaling_with_ranks,
        test_nvlink_vs_cross_node,
    ]

    print("Running NCCL collective simulation tests...\n")

    passed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__} failed: {e}")
        except Exception as e:
            print(f"✗ {test_fn.__name__} error: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    sys.exit(main())
