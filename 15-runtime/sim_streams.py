"""Multi-stream H2D-Compute-D2H overlap and SM utilization.
Corresponding to GPU-15, section 6: H2D copy, compute, D2H copy pipeline.

Core model:
  Three independent hardware queues: H2D (PCIe), SM (compute), D2H (PCIe).

  Single stream (serialized):
    T = N * (h2d_time + compute_time + d2h_time)

  Three-stream pipeline (H2D in stream_copy1, compute in stream_main, D2H in stream_copy2):
    Throughput: max(h2d_time, compute_time, d2h_time) per batch at steady state
    T ≈ (N+2) * max(h2d, compute, d2h)
    SM utilization: compute_time / max(...) across N steady-state batches

Parameters (H100 SXM, unit: microseconds, 128 MiB per batch typical):
  - PCIe Gen5 x16: ~63 GB/s per direction; 128 MiB @ 63 GB/s ≈ 2 ms [1]
  - GPU computation: 100 us to 2 ms depending on kernel size [2]
  - Balanced three-way: aim for similar h2d, compute, d2h times to maximize utilization

Subcommands:
  serial     show serial (no pipelining) timeline
  parallel   show three-stream pipelined timeline
  compare    side-by-side comparison of serial vs pipeline
  utilization  show SM busy percentage for different segment length ratios

Usage:
  python3 sim_streams.py serial --h2d 200 --compute 200 --d2h 200 --batches 10
  python3 sim_streams.py parallel --h2d 200 --compute 200 --d2h 200 --batches 10
  python3 sim_streams.py compare

[1] PCIe Gen5 x16 bandwidth: 64 GB/s nominal, ~50-63 GB/s effective per direction
    128 MiB / 63 GB/s ≈ 2030 us
[2] GPU kernel execution: 100 us (small), 500 us (medium), 2000 us (large)
[3] NVIDIA Blackwell Tuning Guide: overlap strategies with copy engines
    https://docs.nvidia.com/cuda/blackwell-tuning-guide/
"""
import argparse
import sys


def serial_timeline(h2d_us, compute_us, d2h_us, num_batches):
    """Serial execution: H→K→D repeats N times."""
    per_batch = h2d_us + compute_us + d2h_us
    total = num_batches * per_batch
    return {'total': total, 'per_batch': per_batch, 'h2d': h2d_us * num_batches,
            'compute': compute_us * num_batches, 'd2h': d2h_us * num_batches}


def parallel_timeline(h2d_us, compute_us, d2h_us, num_batches):
    """Three-stream pipelined execution.
    After warmup (2 batches), overlap: H2D(n), compute(n), D2H(n-1) in parallel.
    Throughput = max(h2d, compute, d2h) per batch.
    Total ≈ initial pipeline fill + steady state.
    """
    if num_batches == 0:
        return {'total': 0, 'max_segment': 0, 'sm_busy_pct': 0}

    max_seg = max(h2d_us, compute_us, d2h_us)
    # Warmup: 2 batches to fill pipeline
    # Batch 0: H2D(0) happens
    # Batch 1: H2D(1) + compute(0) in parallel, then D2H(0)
    # After batch 2, steady state: max_seg per batch

    warmup_cost = 2 * max_seg  # simplified: first 2 batches to fill pipeline
    steady_cost = (num_batches - 2) * max_seg if num_batches > 2 else 0
    total = warmup_cost + steady_cost if num_batches > 0 else max_seg

    # SM utilization: fraction of time SM is executing (not stalled on H2D or D2H)
    # In perfectly balanced pipeline, SM busy time = compute_us per cycle, cycle = max_seg
    sm_busy_pct = (compute_us / max_seg) * 100 if max_seg > 0 else 0

    return {'total': total, 'max_segment': max_seg, 'sm_busy_pct': sm_busy_pct}


def cmd_serial(args):
    """Show serial (no pipelining) timeline."""
    res = serial_timeline(args.h2d, args.compute, args.d2h, args.batches)
    print(f"\nSerial execution (no pipelining)")
    print(f"  H2D time: {args.h2d} us/batch")
    print(f"  Compute:  {args.compute} us/batch")
    print(f"  D2H time: {args.d2h} us/batch")
    print(f"  Batches:  {args.batches}")
    print(f"\nResults:")
    print(f"  Per-batch cycle: {res['per_batch']:.0f} us")
    print(f"  Total wall clock: {res['total']:.0f} us")
    print(f"  Breakdown: H2D {res['h2d']:.0f} us, Compute {res['compute']:.0f} us, D2H {res['d2h']:.0f} us")


def cmd_parallel(args):
    """Show pipelined timeline."""
    res = parallel_timeline(args.h2d, args.compute, args.d2h, args.batches)
    print(f"\nThree-stream pipelined execution")
    print(f"  H2D time: {args.h2d} us/batch")
    print(f"  Compute:  {args.compute} us/batch")
    print(f"  D2H time: {args.d2h} us/batch")
    print(f"  Batches:  {args.batches}")
    print(f"\nResults:")
    print(f"  Bottleneck segment: {res['max_segment']:.0f} us")
    print(f"  SM utilization: {res['sm_busy_pct']:.1f}%")
    print(f"  Total wall clock: {res['total']:.0f} us")


def cmd_compare(args):
    """Side-by-side comparison of serial vs pipelined."""
    h2d = args.h2d
    compute = args.compute
    d2h = args.d2h
    batches = args.batches

    serial_res = serial_timeline(h2d, compute, d2h, batches)
    parallel_res = parallel_timeline(h2d, compute, d2h, batches)

    print(f"\nComparison: Serial vs Three-Stream Pipeline")
    print(f"Scenario: H2D={h2d} us, compute={compute} us, D2H={d2h} us, {batches} batches\n")

    print(f"Serial execution:")
    print(f"  Total wall clock: {serial_res['total']:.0f} us")
    print(f"  Per-batch cycle:  {serial_res['per_batch']:.0f} us")

    print(f"\nThree-stream pipeline:")
    print(f"  Total wall clock: {parallel_res['total']:.0f} us")
    print(f"  Bottleneck:       {parallel_res['max_segment']:.0f} us/batch")
    print(f"  SM utilization:   {parallel_res['sm_busy_pct']:.1f}%")

    speedup = serial_res['total'] / parallel_res['total'] if parallel_res['total'] > 0 else 1.0
    print(f"\nSpeedup (serial / pipeline): {speedup:.2f}x")

    # Identify bottleneck
    max_idx = max([(h2d, 'H2D'), (compute, 'Compute'), (d2h, 'D2H')], key=lambda x: x[0])
    print(f"Bottleneck: {max_idx[1]} ({max_idx[0]} us)")


def cmd_utilization(args):
    """Show SM utilization for varying segment length ratios."""
    print(f"\nSM utilization across segment length ratios (total per batch: {args.total} us)\n")
    print(f"{'H2D':>8} {'Compute':>10} {'D2H':>8} {'Max Seg':>10} {'SM %':>8} {'Speedup':>8}")
    print("-" * 60)

    # Keep total fixed, vary compute fraction
    base_seg = args.total // 3  # balanced case
    for compute_pct in [20, 33, 50, 75, 100]:
        compute = int(args.total * compute_pct / 100)
        h2d = (args.total - compute) // 2
        d2h = args.total - compute - h2d

        parallel_res = parallel_timeline(h2d, compute, d2h, 100)
        serial_res = serial_timeline(h2d, compute, d2h, 100)
        speedup = serial_res['total'] / parallel_res['total'] if parallel_res['total'] > 0 else 1.0

        print(f"{h2d:8} {compute:10} {d2h:8} {parallel_res['max_segment']:10.0f} "
              f"{parallel_res['sm_busy_pct']:8.1f} {speedup:8.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd')

    # serial subcommand
    p_serial = subparsers.add_parser('serial', help='Show serial timeline')
    p_serial.add_argument('--h2d', type=float, default=200, help='H2D time (us)')
    p_serial.add_argument('--compute', type=float, default=200, help='Compute time (us)')
    p_serial.add_argument('--d2h', type=float, default=200, help='D2H time (us)')
    p_serial.add_argument('--batches', type=int, default=10, help='Number of batches')
    p_serial.set_defaults(func=cmd_serial)

    # parallel subcommand
    p_parallel = subparsers.add_parser('parallel', help='Show pipelined timeline')
    p_parallel.add_argument('--h2d', type=float, default=200, help='H2D time (us)')
    p_parallel.add_argument('--compute', type=float, default=200, help='Compute time (us)')
    p_parallel.add_argument('--d2h', type=float, default=200, help='D2H time (us)')
    p_parallel.add_argument('--batches', type=int, default=10, help='Number of batches')
    p_parallel.set_defaults(func=cmd_parallel)

    # compare subcommand
    p_compare = subparsers.add_parser('compare', help='Side-by-side comparison')
    p_compare.add_argument('--h2d', type=float, default=200, help='H2D time (us)')
    p_compare.add_argument('--compute', type=float, default=200, help='Compute time (us)')
    p_compare.add_argument('--d2h', type=float, default=200, help='D2H time (us)')
    p_compare.add_argument('--batches', type=int, default=10, help='Number of batches')
    p_compare.set_defaults(func=cmd_compare)

    # utilization subcommand
    p_util = subparsers.add_parser('utilization', help='Show SM utilization sweep')
    p_util.add_argument('--total', type=int, default=600, help='Total per-batch time (us)')
    p_util.set_defaults(func=cmd_utilization)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
