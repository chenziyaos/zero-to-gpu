"""CUDA kernel launch overhead and eager vs CUDA Graph vs compile time accounting.
Corresponding to GPU-15, sections 2-3: eager launch storms, CUDA Graph recording/replay costs.

Core model:
  T_eager = sum(kernel_time) + (num_kernels - 1) * launch_interval
  If launch_interval > kernel_time: bubbles = (num_kernels - 1) * (launch_interval - kernel_time)

  T_graph = T_record + (num_replays) * T_replay
    where T_record ≈ T_eager (full launch sequence + sync)
    and T_replay ≈ num_kernels * replay_cost (roughly launch_interval / 4)

  T_fused = 1 * fused_kernel_time + launch_interval (one big kernel, one launch)

Parameters (H100 SXM public measurements, unit: microseconds):
  - launch_interval: 5 us per PyTorch/CUDA runtime launch [1][2]
  - fused_epilogue_reduction: 30% time savings typical for elementwise+reduction fused chains [3]
  - record_sync_cost: included in eager re-execution during graph recording

Subcommands:
  eager    show launch bubble cost for varying kernel size and count
  graph    show cumulative speedup of repeated graph replays
  fusion   show fused kernel time vs eager chain
  compare  side-by-side comparison of eager, graph, fusion regimes

Usage:
  python3 sim_launch.py eager --kernels 2000 --kernel-us 1 --launch-us 5
  python3 sim_launch.py graph --kernels 2000 --kernel-us 1 --launch-us 5 --replays 1000
  python3 sim_launch.py compare

[1] PyTorch CUDA semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
[2] CUDA Runtime API overhead measurements: ~3-10 us dispatch, ~5 us launch [NVIDIA tuning guides]
[3] torch.compile & TorchInductor fusion: elementwise+reduction typically 30-40% overhead reduction
"""
import argparse
import sys


def eager_timeline(num_kernels, kernel_us, launch_us):
    """Eager execution: launch + compute overlap model.
    GPU is busy for kernel_time, CPU spends launch_us between launches.
    If launch_us > kernel_us, bubbles appear.
    """
    compute_time = num_kernels * kernel_us
    if launch_us > kernel_us:
        # Each gap between kernels creates a bubble
        bubble_per_gap = launch_us - kernel_us
        total_bubbles = (num_kernels - 1) * bubble_per_gap
        wall_clock = compute_time + total_bubbles
    else:
        # Launch hidden behind compute
        wall_clock = compute_time + (num_kernels - 1) * launch_us

    bubble_fraction = 0.0 if wall_clock == 0 else max(0, (wall_clock - compute_time) / wall_clock)
    return {'wall_clock': wall_clock, 'compute': compute_time, 'bubbles': wall_clock - compute_time,
            'bubble_pct': bubble_fraction * 100}


def graph_replay_cost(num_kernels, kernel_us, launch_us, num_replays):
    """CUDA Graph: record once, replay many times.
    Record cost ≈ eager (full launch sequence)
    Replay cost ≈ num_kernels * (launch_us / 4) + kernel_us per kernel
    """
    # Recording is like one eager pass
    record = eager_timeline(num_kernels, kernel_us, launch_us)['wall_clock']

    # Replay: graph submission is one operation (few us), then kernels execute with tight packing
    # Replay submission cost is roughly 1-2 us, but the key saving is: no per-kernel launch overhead
    # inside the graph. Each kernel still takes its time, but packed tightly.
    # Model: replay cost ≈ sum(kernel_time) + graph_overhead
    # Graph overhead is typically much smaller than per-kernel launch cost
    graph_submit_us = 1.0  # one graph submit per replay
    replay_cost = (num_kernels * kernel_us) + graph_submit_us

    total_time = record + num_replays * replay_cost
    eager_equiv = (num_replays + 1) * eager_timeline(num_kernels, kernel_us, launch_us)['wall_clock']
    speedup = eager_equiv / total_time if total_time > 0 else 1.0

    return {'record': record, 'replay_cost': replay_cost, 'total': total_time,
            'eager_equiv': eager_equiv, 'speedup': speedup}


def fused_kernel_time(num_kernels, kernel_us, fused_overhead_fraction=0.3):
    """Fusion: merge N small kernels into 1, save launch overhead but add compilation cost.
    Typical fusion reduces elementwise+reduction chains by 30-40%.
    Assume: fused_time ≈ sum(kernel_time) * (1 - overhead_fraction) + 1 launch
    """
    total_compute = num_kernels * kernel_us
    # Fusion overhead: typically 30% of total compute saved by merging accesses
    fused_compute = total_compute * (1 - fused_overhead_fraction)
    fused_wall = fused_compute + 5  # one launch
    return fused_wall


def cmd_eager(args):
    """Show launch bubble cost for eager execution."""
    print(f"\nEager execution timeline (kernel: {args.kernel_us} us, launch: {args.launch_us} us)")
    print(f"{'Kernels':>8} {'Wall(us)':>10} {'Compute':>10} {'Bubbles':>10} {'Bubble%':>8}")
    print("-" * 56)

    # If specific kernels count provided, show only that; else show sweep
    if hasattr(args, 'kernels') and args.kernels != 2000:
        nk_list = [args.kernels]
    else:
        nk_list = [100, 500, 1000, 2000, 5000]

    for nk in nk_list:
        res = eager_timeline(nk, args.kernel_us, args.launch_us)
        print(f"{nk:8} {res['wall_clock']:10.0f} {res['compute']:10.0f} {res['bubbles']:10.0f} {res['bubble_pct']:7.1f}%")


def cmd_graph(args):
    """Show graph recording and replay cost amortization."""
    print(f"\nCUDA Graph recording & replay (kernels: {args.kernels}, kernel: {args.kernel_us} us)")
    print(f"{'Replays':>10} {'Total(us)':>12} {'Speedup':>8} {'Vs Eager':>12}")
    print("-" * 50)

    eager_base = eager_timeline(args.kernels, args.kernel_us, args.launch_us)['wall_clock']

    for rep in [1, 2, 5, 10, 100, 1000, 10000]:
        res = graph_replay_cost(args.kernels, args.kernel_us, args.launch_us, rep)
        eager_total = (rep + 1) * eager_base
        print(f"{rep:10} {res['total']:12.0f} {res['speedup']:8.2f}x {eager_total/res['total']:12.2f}x")


def cmd_fusion(args):
    """Show fused kernel time vs eager launch chain."""
    print(f"\nFusion speedup: eager vs fused (kernel: {args.kernel_us} us, launch: {args.launch_us} us)")
    print(f"{'Kernels':>8} {'Eager(us)':>10} {'Fused(us)':>10} {'Speedup':>8}")
    print("-" * 45)

    for nk in [10, 100, 500, 2000]:
        eager_res = eager_timeline(nk, args.kernel_us, args.launch_us)
        fused = fused_kernel_time(nk, args.kernel_us, fused_overhead_fraction=0.3)
        speedup = eager_res['wall_clock'] / fused if fused > 0 else 1.0
        print(f"{nk:8} {eager_res['wall_clock']:10.0f} {fused:10.0f} {speedup:8.2f}x")


def cmd_compare(args):
    """Side-by-side comparison of eager, graph, fusion."""
    nk = args.kernels
    kernel_us = 1
    launch_us = 5

    print(f"\nThree regimes: eager vs CUDA Graph vs fusion")
    print(f"Scenario: {nk} kernels × {kernel_us} us each, {launch_us} us launch interval\n")

    eager_res = eager_timeline(nk, kernel_us, launch_us)
    graph_res = graph_replay_cost(nk, kernel_us, launch_us, 100)
    fused_res = fused_kernel_time(nk, kernel_us, fused_overhead_fraction=0.3)

    print(f"Eager execution (100 steps):")
    print(f"  Wall clock:    {eager_res['wall_clock'] * 100:8.0f} us")
    print(f"  Compute time:  {eager_res['compute'] * 100:8.0f} us")
    print(f"  Bubbles:       {eager_res['bubbles'] * 100:8.0f} us ({eager_res['bubble_pct']:.1f}%)")

    print(f"\nCUDA Graph (record + 100 replays):")
    print(f"  Record:        {graph_res['record']:8.0f} us")
    print(f"  Replay (each): {graph_res['replay_cost']:8.0f} us")
    print(f"  Total:         {graph_res['total']:8.0f} us")
    print(f"  Speedup vs eager: {eager_res['wall_clock'] * 100 / graph_res['total']:6.2f}x")

    print(f"\nFusion (compiled into 1 kernel, 100 steps):")
    print(f"  Per-step:      {fused_res:8.0f} us")
    print(f"  Total (100):   {fused_res * 100:8.0f} us")
    print(f"  Speedup vs eager: {eager_res['wall_clock'] * 100 / (fused_res * 100):6.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd')

    # eager subcommand
    p_eager = subparsers.add_parser('eager', help='Show launch bubble costs')
    p_eager.add_argument('--kernels', type=int, default=2000,
                         help='Number of kernels, default 2000')
    p_eager.add_argument('--kernel-us', type=float, default=1,
                         help='Kernel execution time (us), default 1')
    p_eager.add_argument('--launch-us', type=float, default=5,
                         help='Launch interval (us), default 5')
    p_eager.set_defaults(func=cmd_eager)

    # graph subcommand
    p_graph = subparsers.add_parser('graph', help='Show graph recording/replay costs')
    p_graph.add_argument('--kernels', type=int, default=2000,
                         help='Number of kernels, default 2000')
    p_graph.add_argument('--kernel-us', type=float, default=1,
                         help='Kernel execution time (us), default 1')
    p_graph.add_argument('--launch-us', type=float, default=5,
                         help='Launch interval (us), default 5')
    p_graph.add_argument('--replays', type=int, default=1000,
                         help='Number of replays (unused; cmd shows multiple), default 1000')
    p_graph.set_defaults(func=cmd_graph)

    # fusion subcommand
    p_fusion = subparsers.add_parser('fusion', help='Show fusion speedup')
    p_fusion.add_argument('--kernel-us', type=float, default=1,
                          help='Original kernel time (us), default 1')
    p_fusion.add_argument('--launch-us', type=float, default=5,
                          help='Launch interval (us), default 5')
    p_fusion.set_defaults(func=cmd_fusion)

    # compare subcommand
    p_compare = subparsers.add_parser('compare', help='Side-by-side comparison')
    p_compare.add_argument('--kernels', type=int, default=2000,
                           help='Number of kernels, default 2000')
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
