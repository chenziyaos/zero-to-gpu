#!/usr/bin/env python3
"""
GPU-18: Compute-Communication Overlap & Bubbles

Simulates gradient computation and collective communication overlap in DDP training.
Determines what fraction of all-reduce communication can be hidden behind backward compute.

Model: backward pass proceeds layer-by-layer (or bucket-by-bucket). As each layer's
gradients are ready, an all-reduce can start. If backward is still running, communication
happens in parallel; if backward completes before all-reduce starts, communication sits
on the critical path.

Bucketizing: Group multiple layers into buckets. Larger buckets mean waiting longer before
starting communication (more overlap potential), but also more data per all-reduce.

Key insight: Overlap quality depends on:
1. Compute time per layer/bucket vs all-reduce time per bucket
2. Degree of pipelining (sequential vs parallel bucket processing)
3. Number of buckets (trade-off: more buckets = less overlap but sooner comm start)

For DDP with typical settings (25 MB default bucket, ~50-100 ms all-reduce per bucket),
achieving 80-95% comm hiding requires 8-16 buckets for a 32-layer model.

References:
- PyTorch DDP implementation: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/
- MegaScale (arXiv 2402.15627): Large-scale training with gradient bucketing
- NCCL Overlap in PyTorch: https://pytorch.org/docs/stable/notes/ddp.html

Software: Python 3.10+
"""

import argparse
import sys
from dataclasses import dataclass


@dataclass
class OverlapScenario:
    """Backward pass and all-reduce scenario."""
    num_layers: int  # layers in model
    compute_time_ms: float  # backward compute time per layer in ms
    comm_time_ms: float  # all-reduce communication time in ms
    num_buckets: int  # how many gradient buckets


def compute_overlap_timeline(scenario: OverlapScenario) -> dict:
    """
    Compute the critical path when overlapping backward + all-reduce.

    Scenario: backward pass produces gradients layer by layer.
    As soon as a layer's gradients are ready, we can start all-reducing them.
    But if backward is still running, comm is stalled on critical path.

    Bucketizing: group layers into buckets and communicate bucket-wise.
    Smaller buckets (more buckets) = communicate sooner = less overlap.
    Larger buckets (fewer buckets) = communicate later = more overlap.

    Timeline (bucket model):
    - Bucket i starts computing at: i * (total_compute / num_buckets)
    - Bucket i finishes computing at: (i+1) * (total_compute / num_buckets)
    - Bucket i can start comm immediately after compute
    - Bucket i comm takes: comm_time
    - Critical path: max(compute_end_time, comm_end_time)
    """
    total_compute = scenario.compute_time_ms * scenario.num_layers
    bucket_compute = total_compute / scenario.num_buckets
    bucket_comm = scenario.comm_time_ms

    # Build timeline
    exposed_comm = 0.0  # communication NOT hidden by compute
    max_time = 0.0
    last_comm_end = 0.0

    for i in range(scenario.num_buckets):
        compute_start = i * bucket_compute
        compute_end = (i + 1) * bucket_compute

        # Comm can start after compute finishes, but not before last comm
        comm_start = max(compute_end, last_comm_end)
        comm_end = comm_start + bucket_comm

        # This comm is exposed if it starts before previous bucket's compute finished
        # OR if this is the last bucket and compute already done
        if comm_start > compute_end:
            # Comm fully happens after compute; it's on critical path
            exposed_comm += bucket_comm
        else:
            # Comm overlaps with compute; expose the tail if any
            overlap_window = compute_end - comm_start
            if overlap_window < bucket_comm:
                exposed_comm += bucket_comm - overlap_window

        last_comm_end = comm_end
        max_time = max(max_time, comm_end)

    # Total without overlap: compute + all communictions
    total_no_overlap = total_compute + scenario.num_buckets * bucket_comm

    # With overlap, critical path is limited
    critical_path = max(total_compute, last_comm_end)

    # Actual exposed comm: the part that couldn't be hidden
    # (conservative: last bucket's comm is always on critical path)
    exposed_comm_final = bucket_comm
    for i in range(scenario.num_buckets - 1):
        comm_end_i = (i + 1) * bucket_compute + (i + 1) * bucket_comm
        if i == scenario.num_buckets - 2:
            # Check if next (last) bucket's compute extends past prev comm
            last_bucket_compute_start = (scenario.num_buckets - 1) * bucket_compute
            if last_bucket_compute_start < comm_end_i:
                # Last bucket's compute overlaps previous comm
                exposed_comm_final = 0.0
            else:
                exposed_comm_final = bucket_comm

    # Simpler model: pipelined all-reduce
    # Assumes perfect pipelining: each bucket comm starts when its compute ends
    critical_path_pipelined = total_compute + bucket_comm * (scenario.num_buckets - 1 + 1)
    # No, better: in ideal overlap, all buckets stream through
    # Bucket 0 compute + all bucket comms = bucket_compute + bucket_comm * num_buckets
    # But next bucket can start compute before bucket 0 comm is done
    # Realistic: compute_of_bucket_1 might overlap with comm_of_bucket_0

    # Conservative model: compute first, then all-reduce all at once
    no_overlap_time = total_compute + scenario.num_buckets * bucket_comm

    # Best case: perfect pipelining, first layer blocks on first bucket comm
    # Subsequent layers fully hidden
    best_overlap_time = total_compute + bucket_comm

    return {
        'num_layers': scenario.num_layers,
        'num_buckets': scenario.num_buckets,
        'total_compute_ms': total_compute,
        'total_comm_ms': scenario.num_buckets * bucket_comm,
        'total_no_overlap_ms': no_overlap_time,
        'best_overlap_ms': best_overlap_time,
        'exposed_comm_pct': (bucket_comm / (best_overlap_time - total_compute)) * 100
                            if best_overlap_time > total_compute else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Simulate compute-communication overlap in DDP'
    )
    parser.add_argument(
        'command',
        choices=['timeline', 'bucket-sweep', 'example'],
        help='Simulation type'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=24,
        help='Number of transformer layers (default: 24)'
    )
    parser.add_argument(
        '--compute-time-ms',
        type=float,
        default=5.0,
        help='Backward compute time per layer in ms (default: 5)'
    )
    parser.add_argument(
        '--comm-time-ms',
        type=float,
        default=50.0,
        help='Total all-reduce time in ms (default: 50)'
    )
    parser.add_argument(
        '--num-buckets',
        type=int,
        default=4,
        help='Number of gradient buckets (default: 4)'
    )

    args = parser.parse_args()

    scenario = OverlapScenario(
        num_layers=args.num_layers,
        compute_time_ms=args.compute_time_ms,
        comm_time_ms=args.comm_time_ms,
        num_buckets=args.num_buckets,
    )

    if args.command == 'timeline':
        result = compute_overlap_timeline(scenario)
        print(f"Overlap Timeline (gradient buckets)")
        print(f"Layers:              {result['num_layers']}")
        print(f"Buckets:             {result['num_buckets']}")
        print(f"Compute per layer:   {args.compute_time_ms:.1f} ms")
        print(f"Total compute:       {result['total_compute_ms']:.1f} ms")
        print(f"Total comm:          {result['total_comm_ms']:.1f} ms (all-reduce per bucket)")
        print(f"\nNo overlap:          {result['total_no_overlap_ms']:.1f} ms")
        print(f"Best overlap:        {result['best_overlap_ms']:.1f} ms")
        print(f"Speedup:             {result['total_no_overlap_ms'] / result['best_overlap_ms']:.2f}x")

    elif args.command == 'bucket-sweep':
        print(f"Effect of bucket count on critical path")
        print(f"{'Buckets':>8} {'Total (ms)':>12} {'Speedup':>10} {'Comm hidden %':>15}")
        print("-" * 50)

        for num_b in [1, 2, 4, 8, 12, 24]:
            scenario.num_buckets = num_b
            result = compute_overlap_timeline(scenario)
            speedup = result['total_no_overlap_ms'] / result['best_overlap_ms']
            # Comm hidden: (total_comm - exposed_on_critical_path) / total_comm
            comm_hidden = (result['total_comm_ms'] - max(0, result['best_overlap_ms'] - scenario.compute_time_ms * scenario.num_layers)) / result['total_comm_ms'] * 100
            print(f"{num_b:>8} {result['best_overlap_ms']:>12.1f} {speedup:>10.2f}x "
                  f"{comm_hidden:>15.1f}%")

    elif args.command == 'example':
        # Realistic Llama 7B example
        # 32 transformer layers, ~100-200 ms backward per GPU for medium batch
        # ~50-100 ms all-reduce per tensor on typical cluster
        print(f"Example: Llama 7B-like training with DDP")
        print(f"(32 layers, ~150 ms backward, ~80 ms all-reduce per 8-rank group)\n")

        scenario = OverlapScenario(
            num_layers=32,
            compute_time_ms=5.0,  # 32 * 5 = 160 ms total
            comm_time_ms=80.0,
            num_buckets=8,
        )

        result = compute_overlap_timeline(scenario)
        print(f"{'Scenario':20} {'Total time (ms)':>16} {'Speedup vs no-overlap':>22}")
        print("-" * 60)

        for num_b in [1, 4, 8, 16, 32]:
            scenario.num_buckets = num_b
            result = compute_overlap_timeline(scenario)
            speedup = result['total_no_overlap_ms'] / result['best_overlap_ms']
            print(f"Buckets={num_b:<14} {result['best_overlap_ms']:>16.1f} {speedup:>22.2f}x")


if __name__ == '__main__':
    main()
