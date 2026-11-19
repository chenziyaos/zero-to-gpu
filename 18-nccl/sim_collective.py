#!/usr/bin/env python3
"""
GPU-18 NCCL: Algorithms, Protocols & Hang Diagnosis

Simulates collective communication timing using alpha-beta model.
Covers ring and tree algorithms with Simple, LL, and LL128 protocols.
Computes theoretical bandwidth (busbw) and effective bandwidth (algbw).

Reference: Demystifying NCCL (arXiv 2507.04786), nccl-tests PERFORMANCE.md
Software: Python 3.10+
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class CollectiveArgs:
    """Collective communication parameters."""
    algorithm: str  # 'ring', 'tree'
    protocol: str   # 'Simple', 'LL', 'LL128'
    num_ranks: int  # number of GPUs in collective
    msg_bytes: int  # message size in bytes
    link_bw: float = 450.0  # GB/s network bandwidth, single direction
    alpha_us: float = 1.0  # per-hop latency in microseconds
    # Typical values (per NCCL docs / Demystifying NCCL arXiv 2507.04786):
    # - NVLink4 intra-node: 450 GB/s, alpha ~1 µs per-hop (optimized), ~5-10 µs conservative
    # - 400G cross-node: 50 GB/s, alpha ~10-20 µs per-hop
    # Alpha-beta model: T_ring = 2(p-1) * (alpha + beta * S/p)
    # where beta = 1 / bandwidth, S = message size, p = number of ranks
    # Refs: Patarasuk & Yuan 2009 "Bandwidth Optimal All-Reduce"
    #       Thakur et al. 2005 "Optimization of Collective Communication"
    #       NCCL: https://github.com/NVIDIA/nccl/blob/master/doc/ALGORITHMS.md



# Per-protocol bandwidth efficiency and per-step latency relative to Simple.
# Bandwidth: LL packs 4 B data + 4 B flag per 8 B store (50%); LL128 carries 120 B per 128 B line (93.75%).
# Latency ratios follow NCCL's tuning model for ring over NVLink (src/graph/tuning.cc hwLat: LL 0.6, LL128 1.9,
# Simple 3.4 us), checked 2026-11-04; --alpha-us is interpreted as the Simple per-step latency.
PROTOCOLS = {
    'Simple': (1.0, 1.0),
    'LL': (0.5, 0.6 / 3.4),
    'LL128': (120 / 128, 1.9 / 3.4),
}


def protocol_params(name):
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {name}")
    return PROTOCOLS[name]

def compute_ring_timing(args: CollectiveArgs) -> dict:
    """
    Ring all-reduce timing using alpha-beta model.

    Ring algorithm: two phases, each with (n-1) sequential hops.
    - Reduce-scatter: rank i sends chunk to i+1 (in ring), receives from i-1
    - All-gather: rank i broadcasts its reduced chunk to all

    Formula (Patarasuk & Yuan 2009):
    T_ring = 2*(n-1) * (alpha + beta * S/(n*B))
    where S=message size, B=bandwidth, n=ranks, beta=1/B*1e9 (to get ns/byte)

    Effective bandwidth (algbw) = S / T_ring
    Bus bandwidth (busbw) = algbw * 2(n-1)/n (nccl-tests PERFORMANCE.md); ideal busbw approaches link_bw

    Protocol modes (per Demystifying NCCL arXiv 2507.04786):
    - Simple: full link bandwidth, highest per-step latency (memory fences + flags per chunk)
    - LL: 8-byte stores carry 4 bytes of data + 4-byte flag, so ~50% bandwidth, lowest latency
    - LL128: 128-byte lines carry 120 bytes of data, ~94% bandwidth, latency in between

    The effective_bw accounts for protocol inefficiency, which improves algbw
    but doesn't change the theoretical max (busbw).

    References:
    - Patarasuk & Yuan 2009: "Bandwidth Optimal All-Reduce Algorithms"
    - Thakur et al. 2005: "Optimization of Collective Communication"
    - Demystifying NCCL (arXiv 2507.04786)
    - NCCL Algorithms: https://github.com/NVIDIA/nccl/blob/master/doc/ALGORITHMS.md
    """
    n = args.num_ranks
    msg_bytes = args.msg_bytes
    chunk_bytes = msg_bytes / n

    protocol_efficiency, alpha_scale = protocol_params(args.protocol)

    effective_bw = args.link_bw * protocol_efficiency
    beta_sec_per_gb = 1.0 / effective_bw  # seconds per GB
    alpha_sec = args.alpha_us * alpha_scale / 1e6  # per-step latency of this protocol, seconds

    # Ring: reduce-scatter + all-gather, 2*(n-1) sequential hops
    num_steps = 2 * (n - 1)
    chunk_gb = chunk_bytes / 1e9
    step_time_sec = alpha_sec + beta_sec_per_gb * chunk_gb
    total_time_sec = num_steps * step_time_sec

    # Busbw (bus bandwidth): theoretical max if protocol used full link BW
    # For all-reduce with n ranks, only 2(n-1)/n links are active on average
    # (out of n possible pairs), so max aggregate = 2(n-1)/n * link_bw

    # Algbw (algorithm bandwidth): actual achieved bandwidth
    msg_gb = msg_bytes / 1e9
    algbw_gb_s = msg_gb / total_time_sec if total_time_sec > 0 else 0
    # nccl-tests convention: busbw = algbw × 2(n-1)/n, comparable to the per-GPU link bandwidth
    busbw_gb_s = algbw_gb_s * 2 * (n - 1) / n

    return {
        'algorithm': 'ring',
        'protocol': args.protocol,
        'num_ranks': n,
        'msg_bytes': msg_bytes,
        'chunk_bytes': chunk_bytes,
        'num_steps': num_steps,
        'time_ms': total_time_sec * 1000,
        'busbw_gb_s': busbw_gb_s,
        'algbw_gb_s': algbw_gb_s,
        'efficiency': busbw_gb_s / args.link_bw if args.link_bw > 0 else 0,
    }


def compute_tree_timing(args: CollectiveArgs) -> dict:
    """
    Binary tree all-reduce timing using alpha-beta model.

    Tree algorithm: two phases on a binary tree with depth log(n).
    - Reduce: tree converges to root, ~log(n) levels
    - Broadcast: root broadcasts to leaves, ~log(n) levels
    - Total: ~2*log(n) hops

    Formula (Thakur et al. 2005):
    T_tree = 2*log2(n) * (alpha + beta * S / B)
    where S=message size, B=bandwidth, alpha per-hop latency
    Note: Tree assumes full message size at each hop (conservative);
    some implementations reduce message size in reduce phase.

    Effective bandwidth and protocol efficiency as for ring.
    Bus bandwidth (busbw) = algbw * 2(n-1)/n, same factor as ring (nccl-tests)

    Tree vs Ring trade-offs:
    - Small messages (<512 KB): tree wins on latency (log n vs 2(n-1))
    - Large messages: both approach 2·S/B; ring keeps a small edge (busbw factor 2(n-1)/n vs 2)

    Reference: Thakur et al. 2005, "Optimization of Collective Communication"
    """
    import math
    n = args.num_ranks
    msg_bytes = args.msg_bytes

    protocol_efficiency, alpha_scale = protocol_params(args.protocol)

    effective_bw = args.link_bw * protocol_efficiency
    beta_sec_per_gb = 1.0 / effective_bw
    alpha_sec = args.alpha_us * alpha_scale / 1e6

    # Tree: log(n) levels in reduce, log(n) in broadcast
    tree_depth = math.ceil(math.log2(n))
    num_steps = 2 * tree_depth

    # Pipelined double binary tree (what NCCL uses): latency grows with 2·log2(n) hops, while the message
    # streams through in chunks, so the bandwidth term is ~2·S/B like ring (Sanders et al. 2009; NCCL 2.4 blog)
    total_time_sec = num_steps * alpha_sec + 2 * beta_sec_per_gb * (msg_bytes / 1e9)


    # Algbw
    msg_gb = msg_bytes / 1e9
    algbw_gb_s = msg_gb / total_time_sec if total_time_sec > 0 else 0
    busbw_gb_s = algbw_gb_s * 2 * (n - 1) / n  # same nccl-tests factor for all all-reduce algorithms

    return {
        'algorithm': 'tree',
        'protocol': args.protocol,
        'num_ranks': n,
        'msg_bytes': msg_bytes,
        'tree_depth': tree_depth,
        'num_steps': num_steps,
        'time_ms': total_time_sec * 1000,
        'busbw_gb_s': busbw_gb_s,
        'algbw_gb_s': algbw_gb_s,
        'efficiency': busbw_gb_s / args.link_bw if args.link_bw > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='NCCL collective timing simulator (ring/tree, Simple/LL/LL128)'
    )
    parser.add_argument(
        'algorithm',
        choices=['ring', 'tree', 'compare'],
        help='Collective algorithm'
    )
    parser.add_argument(
        '--protocol',
        default='Simple',
        choices=['Simple', 'LL', 'LL128'],
        help='Communication protocol (default: Simple)'
    )
    parser.add_argument(
        '--num-ranks',
        type=int,
        default=8,
        help='Number of ranks (default: 8)'
    )
    parser.add_argument(
        '--msg-bytes',
        type=int,
        default=1024*1024,
        help='Message size in bytes (default: 1MB)'
    )
    parser.add_argument(
        '--msg-range',
        action='store_true',
        help='Scan message sizes from 1KB to 1GB'
    )
    parser.add_argument(
        '--nvlink-bw',
        type=float,
        default=450.0,
        help='NVLink bandwidth in GB/s single direction (default: 450 for H100)'
    )
    parser.add_argument(
        '--link-bw',
        type=float,
        default=450.0,
        help='Link bandwidth in GB/s single direction (default: 450 for H100 NVLink4)'
    )

    args = parser.parse_args()

    # Create collective args
    coll_args = CollectiveArgs(
        algorithm=args.algorithm,
        protocol=args.protocol,
        num_ranks=args.num_ranks,
        msg_bytes=args.msg_bytes,
        link_bw=args.link_bw,
    )

    if args.msg_range:
        # Scan message sizes
        print(f"Algorithm comparison across message sizes (protocol: {args.protocol}, ranks: {args.num_ranks})")
        print(f"{'Size (MB)':>12} {'Ring (ms)':>12} {'Tree (ms)':>12} {'Ring algbw':>12} {'Tree algbw':>12}")
        print("-" * 62)

        sizes_kb = [1, 10, 100, 1024, 10*1024, 100*1024, 1024*1024]
        for size_kb in sizes_kb:
            msg_bytes = size_kb * 1024
            coll_args.msg_bytes = msg_bytes

            ring_result = compute_ring_timing(coll_args)
            tree_result = compute_tree_timing(coll_args)

            print(f"{size_kb/1024:>12.1f} {ring_result['time_ms']:>12.2f} "
                  f"{tree_result['time_ms']:>12.2f} {ring_result['algbw_gb_s']:>12.2f} "
                  f"{tree_result['algbw_gb_s']:>12.2f}")

    elif args.algorithm == 'compare':
        # Compare all algorithms and protocols at fixed message size
        print(f"Collective comparison at {args.msg_bytes/1024:.0f} KB, {args.num_ranks} ranks")
        print(f"{'Algorithm':>10} {'Protocol':>10} {'Time (ms)':>12} {'Algbw (GB/s)':>14} {'Busbw (GB/s)':>14}")
        print("-" * 62)

        for algo in ['ring', 'tree']:
            for proto in ['Simple', 'LL', 'LL128']:
                coll_args.algorithm = algo
                coll_args.protocol = proto

                if algo == 'ring':
                    result = compute_ring_timing(coll_args)
                else:
                    result = compute_tree_timing(coll_args)

                print(f"{algo:>10} {proto:>10} {result['time_ms']:>12.2f} "
                      f"{result['algbw_gb_s']:>14.2f} {result['busbw_gb_s']:>14.2f}")

    else:
        # Single run
        if args.algorithm == 'ring':
            result = compute_ring_timing(coll_args)
        else:
            result = compute_tree_timing(coll_args)

        # Pretty print
        print(f"\nCollective: {args.algorithm.upper()} all-reduce")
        print(f"Protocol:    {args.protocol}")
        print(f"Ranks:       {result['num_ranks']}")
        print(f"Message:     {result['msg_bytes'] / 1024:.1f} KB")
        print(f"Steps:       {result['num_steps']}")
        if 'chunk_bytes' in result:
            print(f"Chunk:       {result['chunk_bytes'] / 1024:.1f} KB")
        print(f"\nTiming:      {result['time_ms']:.3f} ms")
        print(f"Algbw:       {result['algbw_gb_s']:.2f} GB/s (actual)")
        print(f"Busbw:       {result['busbw_gb_s']:.2f} GB/s (algbw × 2(n-1)/n, compare with link {args.link_bw:g} GB/s)")
        print(f"Efficiency:  {result['efficiency']*100:.1f}%")


if __name__ == '__main__':
    main()
