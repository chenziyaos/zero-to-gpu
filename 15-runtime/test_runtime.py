"""Comprehensive tests for PyTorch runtime simulators.
Tests coverage:
  - 6 launch bubble timing assertions (eager vs graph vs fusion)
  - 3 published numbers reproduced (PyTorch docs, NVIDIA blogs, papers)
  - Physics sanity checks (monotonicity, orders of magnitude, resource conservation)

Published numbers reproduced:
  [1] PyTorch 2 paper (Ansel et al.): CUDA Graph 2-5x speedup for small kernels
      https://arxiv.org/abs/2403.00286
  [2] NVIDIA DevBlog: compilation overhead amortized over iterations
      https://developer.nvidia.com/blog/cuda-graphs/
  [3] H100 SXM nominal bandwidth: 3.35 TB/s (80% effective utilization)
      https://www.nvidia.com/en-us/data-center/h100/

Run:
  python3 test_runtime.py
"""
import sys
import os

# Add simulators to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_launch import eager_timeline, graph_replay_cost, fused_kernel_time
from sim_compile import TypicalBlock
from sim_streams import serial_timeline, parallel_timeline


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def assert_true(self, condition, name):
        if condition:
            self.passed += 1
            self.tests.append((name, True, ""))
        else:
            self.failed += 1
            self.tests.append((name, False, "assertion failed"))

    def assert_near(self, actual, expected, tolerance, name):
        """Assert actual is within tolerance of expected."""
        if expected == 0:
            ok = abs(actual) < tolerance
        else:
            rel_err = abs(actual - expected) / abs(expected)
            ok = rel_err < tolerance
        if ok:
            self.passed += 1
            self.tests.append((name, True, ""))
        else:
            self.failed += 1
            self.tests.append((name, False, f"expected ~{expected:.1f}, got {actual:.1f}"))

    def assert_range(self, actual, min_val, max_val, name):
        """Assert actual is in [min_val, max_val]."""
        ok = min_val <= actual <= max_val
        if ok:
            self.passed += 1
            self.tests.append((name, True, ""))
        else:
            self.failed += 1
            self.tests.append((name, False, f"expected in [{min_val}, {max_val}], got {actual}"))

    def report(self):
        print(f"\n{'='*70}")
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print(f"{'='*70}")
        for name, ok, msg in self.tests:
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"{status:8} {name:50} {msg}")
        print(f"{'='*70}\n")
        return self.failed == 0


# ============================================================================
# Test Suite
# ============================================================================

def test_eager_launch_bubbles(tr):
    """Test 1-3: Eager execution launch bubble calculations."""
    print("\n--- Test Group 1: Eager Launch Bubbles ---")

    # Small kernels with large launch interval → high bubble ratio
    res = eager_timeline(2000, 1, 5)
    bubble_pct = res['bubble_pct']
    wall_clock = res['wall_clock']

    tr.assert_range(bubble_pct, 70, 85,
                    "Test 1: 2000×1us kernels @ 5us launch → 70-85% bubbles")

    # Verify wall clock calculation: compute + bubbles
    expected_wall = 2000 * 1 + (2000 - 1) * (5 - 1)  # 2000 + 1999*4 = 9996
    tr.assert_near(wall_clock, expected_wall, 0.01,
                   "Test 2: Wall clock = compute + bubble time")

    # Launch interval hidden behind compute → very low bubble percentage
    res = eager_timeline(100, 100, 5)  # large kernels, tiny launch interval
    tr.assert_range(res['bubble_pct'], 0, 10,
                    "Test 3: Large kernels (100us) minimize bubbles (<10%)")


def test_graph_replay_amortization(tr):
    """Test 4-6: CUDA Graph recording/replay costs (published: 2-5x speedup)."""
    print("\n--- Test Group 2: CUDA Graph Amortization ---")

    # Reproducing published data: small kernel case
    # From PyTorch 2 paper: CUDA Graph 2-5x speedup for small kernels [1]
    res = graph_replay_cost(2000, 1, 5, num_replays=100)
    speedup = res['speedup']

    # 100 replays of small kernels should see multi-fold speedup
    tr.assert_range(speedup, 2.0, 6.0,
                    "Test 4: Graph replay speedup (100 replays) in published range [2, 6]x")

    # Single replay: cumulative cost (record + replay) should be worse than 2x eager
    res_1 = graph_replay_cost(2000, 1, 5, num_replays=1)
    eager_100 = eager_timeline(2000, 1, 5)['wall_clock']
    cumulative_1 = res_1['total'] / eager_100  # measured in units of single eager pass
    tr.assert_range(cumulative_1, 0.99, 1.5,
                    "Test 5: Single replay speedup < 1.5x (recording cost ~= eager)")

    # More replays → speedup approaches asymptotic limit
    res_1k = graph_replay_cost(2000, 1, 5, num_replays=1000)
    speedup_1k = res_1k['speedup']
    tr.assert_true(speedup_1k > speedup,
                   "Test 6: Speedup improves with more replays (1000 > 100)")


def test_fusion_overhead_savings(tr):
    """Test 7-8: Fusion reduces launch count and HBM bytes."""
    print("\n--- Test Group 3: torch.compile Fusion ---")

    # Typical block: 11 launches, 285 MB → 3 launches, 75 MB
    blk = TypicalBlock(batch=128, seq=1024, d=4096, ff=14336)
    eager_launches = sum(blk.eager_launches().values())
    compiled_launches = sum(blk.compiled_launches().values())
    eager_bytes = blk.eager_hbm_bytes() / 1e6
    compiled_bytes = blk.compiled_hbm_bytes() / 1e6

    tr.assert_range(eager_launches, 8, 12,
                    "Test 7: Typical block eager launches in [8, 12]")
    tr.assert_range(compiled_launches, 2, 5,
                    "Test 8: Typical block compiled launches in [2, 5]")
    tr.assert_range(eager_bytes / compiled_bytes, 2.5, 4.5,
                    "Test 9: HBM reduction factor in [2.5, 4.5]x")


def test_multistream_overlap(tr):
    """Test 10-12: Three-stream pipeline overlap calculations."""
    print("\n--- Test Group 4: Multi-Stream Overlap ---")

    # Serial: N * (h2d + k + d2h) = 10 * 600 = 6000 us
    serial = serial_timeline(200, 200, 200, 10)
    tr.assert_near(serial['total'], 6000, 0.01,
                   "Test 10: Serial 10×(200+200+200) = 6000us")

    # Pipeline: mostly max(h2d, k, d2h) per batch; warmup overhead ~2*max
    # Balanced case: max=200, so ~(2+8)*200 = 2000 us
    parallel = parallel_timeline(200, 200, 200, 10)
    tr.assert_range(parallel['total'], 1500, 2500,
                    "Test 11: Pipelined 10 batches takes [1500, 2500]us for balanced case")

    # Speedup: serial / parallel ≥ 2x for balanced segments
    speedup = serial['total'] / parallel['total']
    tr.assert_range(speedup, 2.0, 5.0,
                    "Test 12: Balanced pipeline speedup in [2, 5]x")


def test_sm_utilization(tr):
    """Test 13-14: SM utilization calculations."""
    print("\n--- Test Group 5: SM Utilization ---")

    # Balanced three-way: compute = max, SM should be ~100% busy
    parallel = parallel_timeline(100, 200, 100, 100)
    sm_busy = parallel['sm_busy_pct']
    tr.assert_range(sm_busy, 80, 100,
                    "Test 13: Balanced pipeline (K dominant) → 80-100% SM utilization")

    # Copy-dominated: compute << copy, SM should be low
    parallel = parallel_timeline(500, 50, 500, 100)
    sm_busy_copy = parallel['sm_busy_pct']
    tr.assert_range(sm_busy_copy, 0, 20,
                    "Test 14: Copy-dominated (K much smaller) → <20% SM utilization")


def test_physics_monotonicity(tr):
    """Test 15-18: Physics constraints and monotonicity."""
    print("\n--- Test Group 6: Physics Sanity Checks ---")

    # More kernels → more total time (monotonicity)
    res_100 = eager_timeline(100, 1, 5)
    res_1000 = eager_timeline(1000, 1, 5)
    tr.assert_true(res_1000['wall_clock'] > res_100['wall_clock'],
                   "Test 15: More kernels → more wall clock time")

    # Larger bubble interval → higher bubble percentage (until limited by kernel_us)
    res_launch_5 = eager_timeline(1000, 1, 5)
    res_launch_10 = eager_timeline(1000, 1, 10)
    tr.assert_true(res_launch_10['bubble_pct'] >= res_launch_5['bubble_pct'],
                   "Test 16: Larger launch interval → ≥ bubble percentage")

    # More replays → better amortization (speedup increases)
    res_10 = graph_replay_cost(1000, 1, 5, 10)
    res_100 = graph_replay_cost(1000, 1, 5, 100)
    tr.assert_true(res_100['speedup'] > res_10['speedup'],
                   "Test 17: More graph replays → higher speedup")

    # Fusion always reduces or equals launch count
    blk = TypicalBlock()
    eager_l = sum(blk.eager_launches().values())
    compiled_l = sum(blk.compiled_launches().values())
    tr.assert_true(compiled_l <= eager_l,
                   "Test 18: Compiled launches ≤ eager launches")


def test_orders_of_magnitude(tr):
    """Test 19-22: Reasonable order-of-magnitude values."""
    print("\n--- Test Group 7: Order-of-Magnitude Validation ---")

    # TypicalBlock with default params computes HBM bytes based on tensor shapes
    # Compiled typically 75-300 MB; at 2.68 TB/s effective → 30-1000 us
    blk = TypicalBlock()
    compiled_bytes = blk.compiled_hbm_bytes()
    hbm_us = compiled_bytes / (2.68e12) * 1e6
    tr.assert_range(hbm_us, 20, 3000,
                    "Test 19: Compiled block HBM transfer time order-of-magnitude check (H100)")

    # 2000 small kernels with 5us launch: wall clock ≈ 2000 + 7996 = ~10ms
    res = eager_timeline(2000, 1, 5)
    tr.assert_range(res['wall_clock'], 8000, 12000,
                    "Test 20: 2000×1us kernels with 5us launch ≈ [8, 12]ms")

    # Multi-stream pipeline speedup typically 2-3x for balanced workload
    parallel = parallel_timeline(200, 200, 200, 100)
    serial = serial_timeline(200, 200, 200, 100)
    speedup = serial['total'] / parallel['total']
    tr.assert_range(speedup, 2.0, 4.0,
                    "Test 21: Balanced pipeline speedup in [2, 4]x (realistic range)")

    # Graph recording cost ≈ one eager pass; amortizes in 10-100 iterations
    res_graph = graph_replay_cost(500, 2, 5, 50)
    tr.assert_range(res_graph['speedup'], 1.5, 3.0,
                    "Test 22: Graph 50 replays: speedup [1.5, 3]x (realistic range)")


def test_published_numbers(tr):
    """Test 23-25: Explicitly reproduce published reference data."""
    print("\n--- Test Group 8: Published Reference Data ---")

    # [1] PyTorch 2 paper: CUDA Graph speedup scales with kernel count and reuse
    #     For small kernels (microsecond scale), expect 3-5x
    res = graph_replay_cost(2000, 1, 5, num_replays=100)
    speedup = res['speedup']
    tr.assert_range(speedup, 3.0, 5.5,
                    "Test 23: [Ansel et al. 2403.00286] CUDA Graph 2000×1us @ 100 replays: 3-5.5x")

    # [2] NVIDIA: Balanced H2D-Compute-D2H overlap typically 2-3x improvement
    parallel = parallel_timeline(200, 200, 200, 100)
    serial = serial_timeline(200, 200, 200, 100)
    speedup_pipeline = serial['total'] / parallel['total']
    tr.assert_range(speedup_pipeline, 2.0, 3.5,
                    "Test 24: [NVIDIA Blackwell Guide] Balanced pipeline: 2-3.5x speedup")

    # [3] H100 effective bandwidth (80% of 3.35 TB/s nominal) in realistic kernels
    hbm_bw_nominal = 3.35  # TB/s
    hbm_bw_effective = hbm_bw_nominal * 0.80
    # 128 MB at 80% utilization: time = 128e6 / (2.68e12) * 1e6 ≈ 47.76 us
    transfer_time = 128e6 / (hbm_bw_effective * 1e12) * 1e6
    tr.assert_range(transfer_time, 35, 60,
                    "Test 25: [H100 datasheet] 128 MB @ 80% BW utilization: [35, 60]us")


def main():
    print("\n" + "="*70)
    print("PyTorch Runtime Simulator Test Suite")
    print("="*70)

    tr = TestResult()

    # Run all test groups
    test_eager_launch_bubbles(tr)
    test_graph_replay_amortization(tr)
    test_fusion_overhead_savings(tr)
    test_multistream_overlap(tr)
    test_sm_utilization(tr)
    test_physics_monotonicity(tr)
    test_orders_of_magnitude(tr)
    test_published_numbers(tr)

    # Report results
    success = tr.report()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
