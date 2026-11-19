#!/usr/bin/env python3
"""GPU 负载生成器：四段刻意写坏的负载用于 profiling 练习。
对应 GPU-14 第 10 章：一次完整分析。

未在本仓库真卡上复测；期望值来自以下文献：
  - NVIDIA Nsight Compute 用户指南第 6 章"Metrics"
  - NVIDIA CUDA C++ Best Practices Guide 第 3 章"Performance"
  - Roofline Model 论文附件中的 DRAM 访存模式

需要版本：
  - CUDA 13.4 Update 2 或更新（compute capability 7.0+）
  - cupy 或 pycuda（这里用 pycuda 示例）
  - Python 3.10+

四段负载（按问题严重程度排序）：
  1. launch_heavy - 2000 个微内核，平均 1 us，间隙 10-20 us
  2. memory_bound - 单个大 kernel，缺乏数据复用，受 HBM 带宽限制
  3. compute_latency - 低占用率 kernel，延迟藏不住
  4. good_kernel - 参考实现，对照组
"""
import sys


def launch_heavy():
    """
    2000 个 1 us 微内核，占 kernel 时间 ~2.8%，却留下 3-20 us 的发射间隙。
    特征：
    - kernel 数多（>1000）
    - 平均时长短（<5 us）
    - 每个 kernel 间隙 > 100 us
    """
    print("Workload 1: Launch-heavy (2000 tiny kernels)")
    print("Expected: 2000 kernels × 1 us + 2000 × 10 us launch gap = ~22 ms wall time")
    print("GPU-Util: 100%, but SM Activity < 10%, kernel time only 2.8%")
    return None  # 需要真卡运行


def memory_bound():
    """
    单个大 kernel，缺乏数据复用：
      - N×N 矩阵转置（in-place），步长不连续
      - 触发 bank conflict 和缓存行污染
      - SOL-Memory > 85%, SOL-Compute < 20%
    """
    print("Workload 2: Memory-bound (matrix transpose, non-coalesced)")
    print("Expected: SOL-Memory ~85%, Memory throughput ~2000 GB/s vs H100 3350 GB/s max")
    print("Bottleneck: HBM DRAM bandwidth, not SM compute")
    return None


def compute_latency():
    """
    低占用率 kernel，延迟藏不住：
      - 每个 thread 只加载 1-2 个元素，立即计算，存回
      - 没有足够的在途(in-flight)访存隐藏延迟
      - SM Occupancy < 40%, warp stall 因 dependency
    """
    print("Workload 3: Latency-bound (low occupancy, high dependency stalls)")
    print("Expected: SM Occupancy ~30%, stall reason MemDep ~60%")
    print("Fix: Increase occupancy via loop unrolling or software pipelining")
    return None


def good_kernel():
    """
    参考实现：
      - 充分的 warp 驻留数（>50%）
      - 低延迟依赖链（通过 loop unrolling 和双缓冲）
      - SOL-Compute ~70%, SOL-Memory ~40% → 接近 ridge point, 计算受限
    """
    print("Workload 4: Reference (well-tuned GEMM with double-buffering)")
    print("Expected: SM Occupancy ~80%, stall reasons balanced (<25% each)")
    print("Roofline: intensity ~250 FLOP/byte, near ridge for H100")
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python3 gpu_workload.py [1|2|3|4]")
        print("  1 - launch_heavy")
        print("  2 - memory_bound")
        print("  3 - compute_latency")
        print("  4 - good_kernel")
        return

    workload_id = sys.argv[1]
    workloads = {
        '1': launch_heavy,
        '2': memory_bound,
        '3': compute_latency,
        '4': good_kernel,
    }

    if workload_id not in workloads:
        print(f"Unknown workload: {workload_id}")
        return

    workloads[workload_id]()


if __name__ == '__main__':
    main()
