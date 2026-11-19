#!/usr/bin/env python3
"""
GPU-17 RDMA Network: ECMP Hash Collision Simulation

Simulate N parallel elephant flows (large messages) competing on M uplinks
in a RoCE fabric using ECMP (Equal-Cost Multi-Path) routing. Compute:
- Hash collision probability (multiple flows hitting same uplink)
- Effective available bandwidth per flow
- Impact on all-reduce latency

Reference:
  [1] Meta RoCE for Distributed AI Training (SIGCOMM 2024):
      engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/
  [2] DCQCN (SIGCOMM 2015): research.microsoft.com/en-us/projects/dcqcn/
  [3] Facebook Clos Networks (SIGCOMM 2016): engineering.fb.com/open-source/fboss/

Constants audited: 2026-11-04
  - 400 Gbps network card: 50 GB/s single direction
  - H100 cluster baseline: 32 machines × 16 GPUs = 512 GPUs
  - Bisection bandwidth (32 machines, 1 uplink each): 16 × 50 = 800 GB/s
  - All-reduce reduces this to each GPU: 800 / 256 = 3.125 GB/s per GPU (ideal)
  - ECMP typically uses 5-tuple hash: [IP_src, IP_dst, port_src, port_dst, protocol]
"""

import argparse
import random
from collections import defaultdict
from typing import List, Tuple

def simulate_ecmp_collisions(
    num_flows: int,
    num_uplinks: int,
    random_seed: int = 42,
    num_samples: int = 10000
) -> dict:
    """
    Simulate ECMP hash collision for num_flows competing over num_uplinks.

    Each sample represents one batch of flows independently hashed to uplinks.
    For all-reduce with N GPUs communicating across bisection:
      - num_flows ≈ number of parallel all-reduce messages
      - num_uplinks ≈ bisection link count

    Return collision statistics: max flows per uplink, distribution, effective BW.
    """
    random.seed(random_seed)

    collisions_per_sample = []
    max_on_uplink_per_sample = []

    for _ in range(num_samples):
        # Hash each flow to a random uplink (simulating 5-tuple hash)
        uplink_assignments = defaultdict(int)
        for flow_id in range(num_flows):
            uplink = random.randint(0, num_uplinks - 1)
            uplink_assignments[uplink] += 1

        # Count collisions: uplinks with >1 flow
        collision_count = sum(1 for count in uplink_assignments.values() if count > 1)
        max_flows = max(uplink_assignments.values()) if uplink_assignments else 0

        collisions_per_sample.append(collision_count)
        max_on_uplink_per_sample.append(max_flows)

    # Compute statistics
    avg_collisions = sum(collisions_per_sample) / len(collisions_per_sample)
    max_collision = max(max_on_uplink_per_sample)
    avg_max_on_uplink = sum(max_on_uplink_per_sample) / len(max_on_uplink_per_sample)

    # Theoretical baseline (Poisson approximation): λ = num_flows / num_uplinks
    # Expected collisions ≈ num_uplinks × (1 - e^(-λ)) (at least one flow)
    # But "collision" here means uplink with >1 flow, so:
    # P(k flows on one uplink) = Poisson(λ) choose k
    lam = num_flows / num_uplinks
    theoretical_empty = num_uplinks * (1.0 - (1.0 - 1.0 / num_uplinks) ** num_flows)
    theoretical_max_expected = num_flows  # Worst case

    return {
        "num_flows": num_flows,
        "num_uplinks": num_uplinks,
        "collision_count_avg": avg_collisions,
        "collision_count_max": max(collisions_per_sample),
        "max_flows_on_uplink_avg": avg_max_on_uplink,
        "max_flows_on_uplink_max": max_collision,
        "lambda": lam,
        "theoretical_expected_collisions": theoretical_empty,
        "samples": num_samples
    }

def estimate_effective_bandwidth(
    num_flows: int,
    num_uplinks: int,
    total_uplink_bandwidth_gb_s: float,
    random_seed: int = 42
) -> dict:
    """
    Estimate effective bandwidth per flow under ECMP collision.

    With K flows hashed to M uplinks:
    - Best case: even distribution, each flow gets (total_bw / K) GB/s
    - Worst case: all K flows hit same uplink, each gets (total_bw / M / K) GB/s
    - Typical case: some collision, use simulation to estimate
    """
    collision_stats = simulate_ecmp_collisions(
        num_flows, num_uplinks, random_seed, num_samples=10000
    )

    # Bandwidth per uplink
    bw_per_uplink = total_uplink_bandwidth_gb_s / num_uplinks

    # Ideal: no collision
    ideal_bw_per_flow = total_uplink_bandwidth_gb_s / num_flows

    # With collision: average max flows per uplink determines bottleneck
    # If max is M, flows on that uplink each get (bw_per_uplink / M)
    max_flows_expected = collision_stats["max_flows_on_uplink_avg"]
    typical_bw_per_flow = bw_per_uplink / max_flows_expected if max_flows_expected > 0 else 0

    # Worst case observed in simulation
    max_flows_worst = collision_stats["max_flows_on_uplink_max"]
    worst_bw_per_flow = bw_per_uplink / max_flows_worst if max_flows_worst > 0 else 0

    return {
        "num_flows": num_flows,
        "num_uplinks": num_uplinks,
        "total_bandwidth_gb_s": total_uplink_bandwidth_gb_s,
        "ideal_bw_per_flow_gb_s": ideal_bw_per_flow,
        "typical_bw_per_flow_gb_s": typical_bw_per_flow,
        "worst_case_bw_per_flow_gb_s": worst_bw_per_flow,
        "ideal_all_reduce_latency_1gb_ms": 1000.0 / ideal_bw_per_flow if ideal_bw_per_flow > 0 else float('inf'),
        "typical_all_reduce_latency_1gb_ms": 1000.0 / typical_bw_per_flow if typical_bw_per_flow > 0 else float('inf'),
        "worst_case_all_reduce_latency_1gb_ms": 1000.0 / worst_bw_per_flow if worst_bw_per_flow > 0 else float('inf'),
        "collision_stats": collision_stats
    }

def main():
    parser = argparse.ArgumentParser(
        description="ECMP hash collision and effective bandwidth simulation"
    )
    parser.add_argument("--flows", type=int, default=16,
                       help="Number of parallel flows (all-reduce messages)")
    parser.add_argument("--uplinks", type=int, default=8,
                       help="Number of uplinks (bisection links)")
    parser.add_argument("--total-bw", type=float, default=400.0,
                       help="Total bisection bandwidth (GB/s)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for simulation")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Number of simulation samples")

    args = parser.parse_args()

    print("=== ECMP Hash Collision & Effective Bandwidth ===\n")

    # Scenario: 512 GPUs (32 machines × 16 GPUs), all-reduce
    # Uplinks: 32 machines × 1 uplink per machine = 32
    # But all-reduce ring: only use subset of links
    # Simplify: 16 messages (half cluster) on 8 uplinks

    print(f"Configuration:")
    print(f"  Parallel flows:      {args.flows}")
    print(f"  Uplinks available:   {args.uplinks}")
    print(f"  Total bandwidth:     {args.total_bw:.1f} GB/s")
    print(f"  Random seed:         {args.seed}\n")

    bw_result = estimate_effective_bandwidth(
        args.flows, args.uplinks, args.total_bw, args.seed
    )

    print(f"Collision Analysis:")
    cs = bw_result["collision_stats"]
    print(f"  Avg collided uplinks: {cs['collision_count_avg']:.1f} / {args.uplinks}")
    print(f"  Max flows on uplink (avg): {cs['max_flows_on_uplink_avg']:.2f}")
    print(f"  Max flows on uplink (worst): {cs['max_flows_on_uplink_max']}")
    print(f"  Poisson lambda: {cs['lambda']:.2f}\n")

    print(f"Effective Bandwidth per Flow:")
    print(f"  Ideal (no collision): {bw_result['ideal_bw_per_flow_gb_s']:.2f} GB/s")
    print(f"  Typical (avg case):  {bw_result['typical_bw_per_flow_gb_s']:.2f} GB/s")
    print(f"  Worst (max collision): {bw_result['worst_case_bw_per_flow_gb_s']:.2f} GB/s\n")

    print(f"Time to send 1 GB per flow:")
    print(f"  Ideal:  {bw_result["ideal_all_reduce_latency_1gb_ms"]:.1f} ms")
    print(f"  Typical: {bw_result['typical_all_reduce_latency_1gb_ms']:.2f} ms")
    print(f"  Worst: {bw_result['worst_case_all_reduce_latency_1gb_ms']:.2f} ms\n")

    efficiency = bw_result['typical_bw_per_flow_gb_s'] / bw_result['ideal_bw_per_flow_gb_s']
    print(f"Efficiency: {efficiency:.1%} of ideal\n")

if __name__ == "__main__":
    main()
