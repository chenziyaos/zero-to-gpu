#!/usr/bin/env python3
"""
GPU-20: Straggler impact in synchronous training.

In synchronous training (all ranks must finish each step before moving to next),
one slow node stalls the entire job. This simulator computes the slowdown and
goodput loss from stragglers.

Scenario: N nodes with heterogeneous step times. All-reduce synchronizes at the
end of each step. The job rate is limited by the slowest node.

Key metric: Relative slowdown = max_step_time / avg_step_time
  If avg = 100ms and max = 120ms, slowdown = 1.2x or 20% loss.
"""

import sys
import os
import argparse
import random
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

def straggler_impact(
    num_nodes,
    nominal_step_time_ms,
    straggler_slowdown_ratio,
    num_stragglers,
    num_steps=1000
):
    """
    Simulate training with stragglers.

    Args:
      num_nodes: total nodes
      nominal_step_time_ms: expected step time on a good node
      straggler_slowdown_ratio: how much slower stragglers are (e.g., 1.2 = 20% slower)
      num_stragglers: how many nodes are slow
      num_steps: steps to simulate

    Returns:
      dict with throughput metrics
    """
    if num_stragglers > num_nodes:
        num_stragglers = num_nodes

    # Assign straggler status
    good_nodes = num_nodes - num_stragglers
    good_step_time = nominal_step_time_ms
    straggler_step_time = nominal_step_time_ms * straggler_slowdown_ratio

    # Simulate
    total_time_no_stragglers = 0
    total_time_with_stragglers = 0

    for step in range(num_steps):
        # All-reduce synchronization: bottleneck is slowest node
        ideal_time = good_step_time
        # In synchronous training, all nodes sync at each step
        if num_stragglers == 0:
            actual_time = good_step_time
        else:
            actual_time = straggler_step_time  # Limited by slowest node
        total_time_no_stragglers += ideal_time
        total_time_with_stragglers += actual_time

    throughput_no_straggler = num_steps / (total_time_no_stragglers / 1000)
    throughput_with_straggler = num_steps / (total_time_with_stragglers / 1000)
    slowdown = total_time_with_stragglers / total_time_no_stragglers if total_time_no_stragglers > 0 else 1.0
    goodput_loss = (1 - 1 / slowdown) * 100 if slowdown > 0 else 0

    max_step_time = straggler_step_time if num_stragglers > 0 else good_step_time

    return {
        'max_step_time_ms': max_step_time,
        'avg_step_time_ms': (good_nodes * good_step_time + num_stragglers * straggler_step_time) / num_nodes,
        'total_time_no_stragglers_ms': total_time_no_stragglers,
        'total_time_with_stragglers_ms': total_time_with_stragglers,
        'slowdown': slowdown,
        'throughput_no_straggler': throughput_no_straggler,
        'throughput_with_straggler': throughput_with_straggler,
        'goodput_loss_percent': goodput_loss
    }

def main():
    parser = argparse.ArgumentParser(
        description='Straggler impact in synchronous training'
    )
    parser.add_argument('--num-nodes', type=int, default=8,
                       help='Number of training nodes')
    parser.add_argument('--step-time', type=float, default=100.0,
                       help='Nominal step time (milliseconds)')
    parser.add_argument('--stragglers', type=int, default=1,
                       help='Number of straggler nodes')
    parser.add_argument('--straggler-slowdown', type=float, default=1.5,
                       help='Straggler slowdown multiplier (e.g., 1.5 = 50% slower)')
    parser.add_argument('--num-steps', type=int, default=1000,
                       help='Number of steps to simulate')
    parser.add_argument('--scan', action='store_true',
                       help='Scan over different numbers of stragglers')

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Total nodes: {args.num_nodes}")
    print(f"  Nominal step time: {args.step_time} ms")
    print(f"  Straggler slowdown: {args.straggler_slowdown}x ({(args.straggler_slowdown-1)*100:.0f}% slower)")
    print()

    if args.scan:
        print(f"Scanning number of stragglers (0 to {min(5, args.num_nodes)}):")
        print()
        print(f"{'Stragglers':>10} {'Max Step (ms)':>14} {'Slowdown':>10} {'Goodput Loss':>12}")
        print("-" * 50)

        for num_stragglers in range(min(6, args.num_nodes + 1)):
            result = straggler_impact(
                args.num_nodes,
                args.step_time,
                args.straggler_slowdown,
                num_stragglers,
                args.num_steps
            )
            print(f"{num_stragglers:>10} {result['max_step_time_ms']:>14.1f} "
                  f"{result['slowdown']:>10.3f} {result['goodput_loss_percent']:>11.1f}%")
    else:
        result = straggler_impact(
            args.num_nodes,
            args.step_time,
            args.straggler_slowdown,
            args.stragglers,
            args.num_steps
        )

        print(f"Results ({args.num_steps} steps):")
        print(f"  Max step time: {result['max_step_time_ms']:.1f} ms (avg: {result['avg_step_time_ms']:.1f} ms)")
        print(f"  Slowdown: {result['slowdown']:.3f}x")
        print(f"  Goodput loss: {result['goodput_loss_percent']:.1f}%")
        print(f"  Throughput without stragglers: {result['throughput_no_straggler']:.1f} steps/sec")
        print(f"  Throughput with stragglers: {result['throughput_with_straggler']:.1f} steps/sec")

if __name__ == '__main__':
    main()
