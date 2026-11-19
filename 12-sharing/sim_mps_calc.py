"""MPS（Multi-Process Service）SM 分区计算器：计算 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 的分配。

MPS 让多个进程的 kernel 共享同一张卡的 SM 和 cache，通过静态 SM 分区（exclusive thread percentage）
或动态分区来控制。这个计算器帮助规划 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE，即每个进程最多能用多少 SM。

核心公式（来源：NVIDIA MPS Architecture Guide，Page 18-19）：
- 单个进程的 SM 分区 = CUDA_MPS_ACTIVE_THREAD_PERCENTAGE / 100 × 总 SM 数
- 块大小（chunk）= 总 SM 数 / 允许的最大进程数
- 每个进程获得的块数 = ceil(分配的 SM % / 块大小)
- 注意：分配是向上舍入到块的倍数，所以多个进程的分配之和可能超过 100%

H100 有 132 个 SM，NVSwitch 连接到每个 SM 的 warp scheduler。

子命令：
  partition       静态分区：给定进程数和 SM 比例，计算每个进程的 SM 数
  chunk           块大小计算：给定进程数，块大小多少
  overhead        向上舍入开销：计算与理想分配的差异
  utilization     吞吐模型：给定进程数和占用率，估计相对性能下降

用法：
  python3 sim_mps_calc.py partition --processes 2 --percentage 50
  python3 sim_mps_calc.py chunk --max-processes 4
  python3 sim_mps_calc.py overhead --sm 132 --allocations 50 50
  python3 sim_mps_calc.py utilization --processes 2 --occupancy 25
"""
import argparse
import math
import sys
from typing import List


def partition_sm(total_sm: int, num_processes: int, percentages: List[int]) -> dict:
    """
    计算静态 SM 分区。

    参数：
      total_sm: GPU 的总 SM 数
      num_processes: 进程数
      percentages: 每个进程的目标百分比

    返回：实际 SM 数、块大小、舍入结果
    """
    if len(percentages) != num_processes:
        raise ValueError("percentages 数量必须等于 num_processes")

    if sum(percentages) > 100:
        print(f"WARNING: 总百分比 {sum(percentages)}% > 100%（会相互干扰）", file=sys.stderr)

    chunk_size = total_sm / num_processes
    allocations = []

    for i, pct in enumerate(percentages):
        ideal_sm = pct / 100.0 * total_sm
        chunks_needed = math.ceil(ideal_sm / chunk_size)
        actual_sm = chunks_needed * chunk_size
        allocations.append({
            'process': i + 1,
            'target_pct': pct,
            'ideal_sm': ideal_sm,
            'chunk_size': chunk_size,
            'chunks_needed': chunks_needed,
            'actual_sm': actual_sm,
            'actual_pct': 100.0 * actual_sm / total_sm,
            'wasted_sm': actual_sm - ideal_sm,
        })

    return {
        'total_sm': total_sm,
        'num_processes': num_processes,
        'chunk_size': chunk_size,
        'allocations': allocations,
        'total_allocated': sum(a['actual_sm'] for a in allocations),
        'total_utilization': 100.0 * sum(a['ideal_sm'] for a in allocations) / total_sm,
        'actual_utilization': 100.0 * sum(a['actual_sm'] for a in allocations) / total_sm,
    }


def calculate_chunk_size(total_sm: int, num_processes: int) -> dict:
    """计算块大小（最小分配单位）。"""
    chunk_size = total_sm / num_processes
    return {
        'total_sm': total_sm,
        'num_processes': num_processes,
        'chunk_size': chunk_size,
        'num_chunks': num_processes,
        'min_allocation': chunk_size,
        'note': f"最小分配 {chunk_size:.1f} SM（1 块），最大分配 {total_sm} SM（{num_processes} 块）"
    }


def calculate_overhead(total_sm: int, allocations: List[int]) -> dict:
    """计算由于块对齐导致的舍入开销。"""
    num_processes = len(allocations)
    chunk_size = total_sm / num_processes
    overhead = []

    for i, ideal_sm in enumerate(allocations):
        chunks = math.ceil(ideal_sm / chunk_size)
        actual_sm = chunks * chunk_size
        waste = actual_sm - ideal_sm
        overhead.append({
            'process': i + 1,
            'ideal_sm': ideal_sm,
            'chunks': chunks,
            'actual_sm': actual_sm,
            'wasted_sm': waste,
            'waste_pct': 100.0 * waste / ideal_sm if ideal_sm > 0 else 0,
        })

    total_waste = sum(o['wasted_sm'] for o in overhead)
    total_ideal = sum(allocations)

    return {
        'chunk_size': chunk_size,
        'allocations': overhead,
        'total_wasted_sm': total_waste,
        'total_ideal_sm': total_ideal,
        'overhead_pct': 100.0 * total_waste / total_ideal if total_ideal > 0 else 0,
    }


def estimate_throughput(num_processes: int, occupancy: float, base_tflops: float = 989.0) -> dict:
    """
    估计多进程下的相对吞吐下降。

    简化模型：假设 kernel 的 occupancy 相同，多进程共享 SM 导致上下文切换开销。
    实际性能取决于 kernel 特性、访存行为和调度。

    参数：
      num_processes: 进程数
      occupancy: 平均 occupancy（0-100，%）
      base_tflops: 单进程的理论峰值（H100 默认 989 BF16 TFLOPS）

    来源：NVIDIA MPS 性能分析（一般情况下，N 个进程会有 5-15% 的切换开销）
    """
    # 粗略模型：N 个进程共享，理论吞吐 1/N，但实际由于缓存局部性和调度延迟有损失
    ideal_per_process = base_tflops / num_processes
    # 切换开销估计（5-15% 损失，取中值）
    overhead_factor = 1.0 - (0.05 + 0.05 * (num_processes - 1) / 7.0)  # 2 进程 5%，8 进程 15%
    actual_per_process = ideal_per_process * overhead_factor

    return {
        'num_processes': num_processes,
        'base_tflops': base_tflops,
        'ideal_per_process': ideal_per_process,
        'switch_overhead_pct': (1 - overhead_factor) * 100,
        'actual_per_process': actual_per_process,
        'total_actual': actual_per_process * num_processes,
        'efficiency': 100.0 * actual_per_process * num_processes / base_tflops,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # partition 子命令
    partition_parser = subparsers.add_parser('partition', help='静态 SM 分区')
    partition_parser.add_argument('--processes', type=int, required=True, help='进程数')
    partition_parser.add_argument('--percentage', nargs='+', type=int, help='每个进程的目标百分比（不指定则均分）')
    partition_parser.add_argument('--sm', type=int, default=132, help='GPU 总 SM 数（默认 H100=132）')

    # chunk 子命令
    chunk_parser = subparsers.add_parser('chunk', help='块大小计算')
    chunk_parser.add_argument('--max-processes', type=int, required=True, help='最大进程数')
    chunk_parser.add_argument('--sm', type=int, default=132, help='GPU 总 SM 数')

    # overhead 子命令
    overhead_parser = subparsers.add_parser('overhead', help='舍入开销')
    overhead_parser.add_argument('--sm', type=int, default=132, help='GPU 总 SM 数')
    overhead_parser.add_argument('--allocations', nargs='+', type=float, required=True, help='每个进程的理想 SM 数')

    # utilization 子命令
    util_parser = subparsers.add_parser('utilization', help='吞吐估计')
    util_parser.add_argument('--processes', type=int, required=True, help='进程数')
    util_parser.add_argument('--occupancy', type=float, help='平均 occupancy (pct, optional)')
    util_parser.add_argument('--base-tflops', type=float, default=989, help='single process peak (default H100=989)')

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == 'partition':
        if args.percentage:
            if len(args.percentage) != args.processes:
                print("ERROR: --percentage 数量必须等于 --processes", file=sys.stderr)
                return
            percentages = args.percentage
        else:
            percentages = [100 // args.processes] * args.processes

        result = partition_sm(args.sm, args.processes, percentages)
        print(f"H100 MPS SM 分区（{args.sm} SM，{args.processes} 进程）")
        print(f"块大小: {result['chunk_size']:.1f} SM")
        print()
        print(f"{'进程':6} {'目标%':8} {'理想SM':8} {'块数':6} {'实际SM':8} {'实际%':8} {'浪费':8}")
        print("-" * 60)
        for a in result['allocations']:
            print(f"{a['process']:6} {a['target_pct']:7.1f}% {a['ideal_sm']:7.1f} {a['chunks_needed']:6} "
                  f"{a['actual_sm']:7.1f} {a['actual_pct']:7.1f}% {a['wasted_sm']:7.1f}")
        print("-" * 60)
        print(f"{'总计':6} {sum(a['target_pct'] for a in result['allocations']):7.1f}% "
              f"{sum(a['ideal_sm'] for a in result['allocations']):7.1f} - "
              f"{result['total_allocated']:7.1f} {100.0*result['total_allocated']/args.sm:7.1f}%")

    elif args.cmd == 'chunk':
        result = calculate_chunk_size(args.sm, args.max_processes)
        print(f"块大小计算（{args.sm} SM，支持 {args.max_processes} 进程）")
        print(f"块大小: {result['chunk_size']:.1f} SM")
        print(result['note'])

    elif args.cmd == 'overhead':
        result = calculate_overhead(args.sm, args.allocations)
        print(f"舍入开销分析（块大小 {result['chunk_size']:.1f} SM）")
        print()
        print(f"{'进程':6} {'理想SM':10} {'块数':6} {'实际SM':10} {'浪费':10} {'浪费%':8}")
        print("-" * 60)
        for a in result['allocations']:
            print(f"{a['process']:6} {a['ideal_sm']:10.1f} {a['chunks']:6} {a['actual_sm']:10.1f} "
                  f"{a['wasted_sm']:10.1f} {a['waste_pct']:7.1f}%")
        print("-" * 60)
        print(f"总浪费：{result['total_wasted_sm']:.1f} SM / {result['total_ideal_sm']:.1f} SM "
              f"= {result['overhead_pct']:.1f}%")

    elif args.cmd == 'utilization':
        result = estimate_throughput(args.processes, args.occupancy, args.base_tflops)
        print(f"多进程吞吐估计（{args.processes} 进程，基线 {args.base_tflops} TFLOPS）")
        print(f"理想每进程: {result['ideal_per_process']:.1f} TFLOPS")
        print(f"切换开销: {result['switch_overhead_pct']:.1f}%")
        print(f"实际每进程: {result['actual_per_process']:.1f} TFLOPS")
        print(f"总实际吞吐: {result['total_actual']:.1f} TFLOPS")
        print(f"卡级效率: {result['efficiency']:.1f}%")


if __name__ == '__main__':
    main()
