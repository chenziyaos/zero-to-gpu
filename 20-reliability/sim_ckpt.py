#!/usr/bin/env python3
"""
GPU-20: Checkpoint interval optimization using Young and Daly formulas.

Core formulas:
  Young (1974): τ* = sqrt(2*C*T_MTBF), where C = checkpoint write time
  Daly (2006): τ* = sqrt(2*T_MTBF / (1 + sqrt(1 + (2*R) / T_MTBF))),
               where R = recovery time (read + restart overhead)

ETTR (Effective Training Time Ratio) = usable_time / total_time, accounts for
checkpoint overhead and downtime from failures.

Reference data point: Llama 3 54-day training had 419 interruptions on 16384 H100s,
per Llama 3 Herd of Models paper (arXiv 2407.21783).
"""

import sys
import os
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

def young_interval(checkpoint_time, mtbf):
    """Young 1974: optimal checkpoint interval minimizes E[total time]."""
    if checkpoint_time <= 0 or mtbf <= 0:
        return 0
    return math.sqrt(2 * checkpoint_time * mtbf)

def daly_interval(mtbf, recovery_time, checkpoint_time):
    """
    Daly 2006: refined formula accounting for recovery time.
    Returns τ that minimizes E[T_total] with checkpoint and recovery overhead.
    """
    if mtbf <= 0 or recovery_time < 0 or checkpoint_time <= 0:
        return 0

    denom = 1 + math.sqrt(1 + (2 * recovery_time) / mtbf)
    tau = math.sqrt((2 * mtbf) / denom)
    return tau

def expected_time_to_result(
    total_steps,
    checkpoint_interval_steps,
    checkpoint_time_sec,
    mtbf_sec,
    recovery_time_sec,
    step_time_sec=0.1
):
    """
    Calculate ETTR (Effective Training Time Ratio) and expected completion time.

    Args:
      total_steps: training steps to run
      checkpoint_interval_steps: how many steps between checkpoints
      checkpoint_time_sec: wall-clock time to write a checkpoint
      mtbf_sec: Mean Time Between Failures for this job
      recovery_time_sec: time to read checkpoint and resume (includes restarts)
      step_time_sec: average wall-clock per step

    Returns:
      dict with ETTR, expected total time, num_failures, etc.
    """
    if checkpoint_interval_steps <= 0:
        checkpoint_interval_steps = total_steps

    num_checkpoints = max(1, (total_steps + checkpoint_interval_steps - 1) // checkpoint_interval_steps)
    num_intervals = max(1, num_checkpoints)

    # Expected failures follow Poisson: E[failures] = total_time / MTBF
    # But total_time depends on failures, so we solve iteratively

    # First estimate: no failures
    base_compute = total_steps * step_time_sec
    checkpoint_overhead = num_checkpoints * checkpoint_time_sec
    est_total_1 = base_compute + checkpoint_overhead

    # Now estimate failures in that time
    avg_interval_time = (checkpoint_interval_steps * step_time_sec) + checkpoint_time_sec
    expected_failures = est_total_1 / mtbf_sec

    # Second pass: include recovery time
    est_total_2 = base_compute + checkpoint_overhead + (expected_failures * recovery_time_sec)

    # Converge once more
    expected_failures = est_total_2 / mtbf_sec
    final_total = base_compute + checkpoint_overhead + (expected_failures * recovery_time_sec)

    ettr = base_compute / final_total if final_total > 0 else 0

    return {
        'total_time_sec': final_total,
        'compute_time_sec': base_compute,
        'checkpoint_time_sec': checkpoint_overhead,
        'recovery_time_sec': expected_failures * recovery_time_sec,
        'expected_failures': expected_failures,
        'num_checkpoints': num_checkpoints,
        'ettr': ettr,
        'goodput_ratio': ettr
    }

def scan_intervals(
    total_steps,
    checkpoint_time_sec,
    mtbf_sec,
    recovery_time_sec,
    step_time_sec=0.1
):
    """Scan checkpoint intervals from 1 to total_steps, find optimal."""
    results = []

    for interval in [1, 10, 50, 100, 500, 1000, 5000, 10000]:
        if interval > total_steps:
            continue
        r = expected_time_to_result(
            total_steps,
            interval,
            checkpoint_time_sec,
            mtbf_sec,
            recovery_time_sec,
            step_time_sec
        )
        results.append((interval, r))

    # Also add Young and Daly optimal
    young_tau = young_interval(checkpoint_time_sec, mtbf_sec)
    daly_tau = daly_interval(mtbf_sec, recovery_time_sec, checkpoint_time_sec)

    for tau_name, tau_steps in [('Young', young_tau / step_time_sec), ('Daly', daly_tau / step_time_sec)]:
        tau_steps = max(1, int(tau_steps))
        if tau_steps <= total_steps:
            r = expected_time_to_result(
                total_steps,
                tau_steps,
                checkpoint_time_sec,
                mtbf_sec,
                recovery_time_sec,
                step_time_sec
            )
            results.append((f'{tau_name}({tau_steps}))', r))

    return results

def main():
    parser = argparse.ArgumentParser(
        description='Checkpoint interval optimization: Young and Daly formulas'
    )
    parser.add_argument('--total-steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--ckpt-time', type=float, default=600.0,
                       help='Checkpoint write time (seconds)')
    parser.add_argument('--mtbf', type=float, default=86400.0,
                       help='Mean Time Between Failures (seconds); e.g., 86400 = 1 day')
    parser.add_argument('--recovery-time', type=float, default=300.0,
                       help='Recovery time including read and restart (seconds)')
    parser.add_argument('--step-time', type=float, default=0.1,
                       help='Average wall-clock time per step (seconds)')
    parser.add_argument('--interval', type=int, default=None,
                       help='Fixed checkpoint interval (steps). If omitted, scan to find optimal.')
    parser.add_argument('--scale', type=int, default=1,
                       help='Scale MTBF by 1/N (for N-card job, MTBF shrinks)')

    args = parser.parse_args()

    # Adjust MTBF for cluster size
    job_mtbf = args.mtbf / args.scale

    print(f"Configuration:")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Step time: {args.step_time} s")
    print(f"  Checkpoint write time: {args.ckpt_time} s")
    print(f"  MTBF (single card): {args.mtbf} s ({args.mtbf/3600:.1f} hours)")
    print(f"  Job MTBF ({args.scale} cards): {job_mtbf} s ({job_mtbf/3600:.1f} hours)")
    print(f"  Recovery time: {args.recovery_time} s")
    print()

    if args.interval is not None:
        r = expected_time_to_result(
            args.total_steps,
            args.interval,
            args.ckpt_time,
            job_mtbf,
            args.recovery_time,
            args.step_time
        )
        print(f"Fixed interval: {args.interval} steps")
        print(f"  Total time: {r['total_time_sec']/3600:.2f} hours")
        print(f"  Compute: {r['compute_time_sec']/3600:.2f} h, "
              f"Checkpoint: {r['checkpoint_time_sec']/3600:.2f} h, "
              f"Recovery: {r['recovery_time_sec']/3600:.2f} h")
        print(f"  Expected failures: {r['expected_failures']:.1f}")
        print(f"  ETTR (goodput): {r['ettr']*100:.1f}%")
    else:
        print("Scanning checkpoint intervals...")
        print()
        results = scan_intervals(
            args.total_steps,
            args.ckpt_time,
            job_mtbf,
            args.recovery_time,
            args.step_time
        )

        results.sort(key=lambda x: x[1]['ettr'], reverse=True)

        print(f"{'Interval':>12} {'Total (h)':>10} {'Compute':>8} {'Ckpt':>8} {'Recovery':>10} {'Failures':>8} {'ETTR':>7}")
        print("-" * 75)
        for interval, r in results:
            print(f"{str(interval):>12} {r['total_time_sec']/3600:>10.2f} "
                  f"{r['compute_time_sec']/3600:>8.2f} {r['checkpoint_time_sec']/3600:>8.2f} "
                  f"{r['recovery_time_sec']/3600:>10.2f} {r['expected_failures']:>8.1f} "
                  f"{r['ettr']*100:>6.1f}%")

        best_interval, best_r = results[0]
        print()
        print(f"Optimal checkpoint interval: {best_interval} steps")
        print(f"  Expected total time: {best_r['total_time_sec']/3600:.2f} hours")
        print(f"  ETTR (goodput): {best_r['ettr']*100:.1f}%")

if __name__ == '__main__':
    main()
