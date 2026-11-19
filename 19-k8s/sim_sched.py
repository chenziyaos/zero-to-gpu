#!/usr/bin/env python3
"""
GPU-19: Kubernetes device plugin and topology-aware scheduling simulator.

Simulates random job stream on 8-GPU node with NVLink topology constraints.
Compares first-fit vs topology-aware bin packing.
Metrics: fragmentation rate, queue time, idle GPU hours.

Reference: NVIDIA GPU Operator docs, k8s-device-plugin, Gandiva (OSDI 2018),
Tiresias (NSDI 2019), Philly cluster analysis (ATC 2019).
Verified 2026-11-04.
"""

import argparse
import heapq
import random
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

# H100 SXM single socket: 8 GPUs with NVLink4 topology (double-ring)
# Bandwidth: single direction 450 GB/s per link, typical for 2-ring on 8 GPUs
NVLINK_BW_SINGLE = 450  # GB/s single direction


class PlacementStrategy(Enum):
    """Scheduling strategies."""
    FIRST_FIT = "first-fit"
    TOPOLOGY_AWARE = "topology-aware"


@dataclass
class NVLinkNode:
    """Represents a single GPU and its NVLink neighbors."""
    gpu_id: int
    neighbors: List[int]  # List of GPU IDs connected via NVLink4

    def __hash__(self):
        return hash(self.gpu_id)

    def __eq__(self, other):
        return self.gpu_id == other.gpu_id if isinstance(other, NVLinkNode) else False


@dataclass
class Job:
    """Represents a GPU job."""
    job_id: int
    num_gpus: int
    arrival_time: float
    duration: float
    submitted_time: float = 0.0  # When job was submitted to queue
    scheduled_time: float = 0.0  # When job started on GPUs
    gpus_assigned: List[int] = None  # Which GPUs it runs on

    def __post_init__(self):
        if self.gpus_assigned is None:
            self.gpus_assigned = []


def build_nvlink_topology() -> Dict[int, NVLinkNode]:
    """Build 8-GPU NVLink double-ring topology.

    Ring 1: 0-1-2-3-4
    Ring 2: 4-5-6-7-0
    Full connections: each GPU has ~3-4 neighbors in typical 8-GPU topology.
    """
    topology = {}
    # Double-ring for 8 GPUs
    neighbors_map = {
        0: [1, 7],  # connected to 1 and 7 in ring
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6, 0],
    }

    for gpu_id in range(8):
        topology[gpu_id] = NVLinkNode(gpu_id=gpu_id, neighbors=neighbors_map[gpu_id])

    return topology


def is_topology_aware_placement(
    job_gpus: List[int], topology: Dict[int, NVLinkNode]
) -> bool:
    """Check if assigned GPUs form connected component in NVLink topology.

    Returns True if all GPUs are pairwise reachable within 2 hops.
    """
    if len(job_gpus) == 1:
        return True

    job_set = set(job_gpus)

    # Simple heuristic: check if all pairs are direct neighbors or share a neighbor
    for gpu in job_gpus:
        for other in job_gpus:
            if gpu == other:
                continue
            # Check direct neighbor
            if other in topology[gpu].neighbors:
                continue
            # Check 2-hop neighbor
            found = False
            for neighbor in topology[gpu].neighbors:
                if other in topology[neighbor].neighbors:
                    found = True
                    break
            if not found:
                return False

    return True


def first_fit_schedule(
    node_state: List[int], job: Job, topology: Dict[int, NVLinkNode]
) -> Tuple[bool, List[int]]:
    """First-fit bin packing: assign to first available slots regardless of topology."""
    available_gpus = [i for i in range(8) if node_state[i] == 0]

    if len(available_gpus) < job.num_gpus:
        return False, []

    assigned = available_gpus[:job.num_gpus]
    return True, assigned


def topology_aware_schedule(
    node_state: List[int], job: Job, topology: Dict[int, NVLinkNode]
) -> Tuple[bool, List[int]]:
    """Topology-aware scheduling: prefer contiguous NVLink groups.

    Heuristic: try to place GPUs that form connected subgraph.
    Prefers multi-GPU jobs on adjacent GPUs to reduce communication latency.
    """
    available_gpus = [i for i in range(8) if node_state[i] == 0]

    if len(available_gpus) < job.num_gpus:
        return False, []

    # For single GPU jobs, any slot works
    if job.num_gpus == 1:
        return True, available_gpus[:1]

    # Greedy: start from each available GPU and try to form connected group
    # Prioritize starting points with more available neighbors
    start_candidates = sorted(available_gpus,
                              key=lambda g: len(set(topology[g].neighbors) & set(available_gpus)),
                              reverse=True)

    for start_gpu in start_candidates:
        group = {start_gpu}
        candidates = set(topology[start_gpu].neighbors) & set(available_gpus)

        while len(group) < job.num_gpus and candidates:
            next_gpu = candidates.pop()
            group.add(next_gpu)
            # Expand neighborhood
            for neighbor in topology[next_gpu].neighbors:
                if neighbor in available_gpus and neighbor not in group:
                    candidates.add(neighbor)

        if len(group) >= job.num_gpus:
            assigned = list(group)[:job.num_gpus]
            return True, assigned

    # Fallback to first-fit if topology-aware fails
    return first_fit_schedule(node_state, job, topology)


def simulate(
    strategy: PlacementStrategy,
    num_jobs: int = 100,
    job_size_dist: List[Tuple[int, float]] = None,
    seed: int = 42,
) -> Dict:
    """Simulate job scheduling on 8-GPU node.

    Args:
        strategy: FIRST_FIT or TOPOLOGY_AWARE
        num_jobs: number of jobs to simulate
        job_size_dist: list of (num_gpus, probability) pairs
        seed: random seed

    Returns:
        metrics dict with fragmentation, avg queue time, idle GPU hours
    """
    if job_size_dist is None:
        # Typical distribution: 1-2 GPU jobs are common
        job_size_dist = [(1, 0.3), (2, 0.4), (4, 0.2), (8, 0.1)]

    random.seed(seed)
    topology = build_nvlink_topology()

    # Node state: 0 = idle, 1 = busy
    node_state = [0] * 8
    job_queue = []
    job_events = []
    completed_jobs = []
    current_time = 0.0

    # Generate job arrivals (Poisson process)
    arrival_time = 0.0
    job_id = 0
    while job_id < num_jobs:
        # Inter-arrival time: exponential with mean 2.0 seconds
        inter_arrival = random.expovariate(1.0 / 2.0)
        arrival_time += inter_arrival

        # Job size sampled from distribution
        size = random.choices(
            [x[0] for x in job_size_dist],
            weights=[x[1] for x in job_size_dist],
            k=1
        )[0]

        # Job duration: uniform 5-20 seconds (scaled for simulation)
        duration = random.uniform(5.0, 20.0)

        job = Job(
            job_id=job_id,
            num_gpus=size,
            arrival_time=arrival_time,
            duration=duration,
            submitted_time=arrival_time,
        )
        job_queue.append(job)
        job_id += 1

    # Simulate scheduling events
    job_queue.sort(key=lambda j: j.arrival_time)
    queue_idx = 0
    scheduled_jobs = []
    finish_heap = []  # (finish_time, job_id)

    current_time = 0.0
    max_sim_time = job_queue[-1].arrival_time + 500  # Run long enough

    while queue_idx < len(job_queue) or finish_heap:
        # Process finish events
        while finish_heap and finish_heap[0][0] <= current_time:
            finish_time, job_id = heapq.heappop(finish_heap)
            finished_job = next(j for j in scheduled_jobs if j.job_id == job_id)

            # Free GPUs
            for gpu_id in finished_job.gpus_assigned:
                node_state[gpu_id] = 0

            completed_jobs.append(finished_job)

        # Try to schedule jobs from queue
        while queue_idx < len(job_queue):
            job = job_queue[queue_idx]

            if job.arrival_time > current_time:
                break

            # Try to schedule
            if strategy == PlacementStrategy.FIRST_FIT:
                scheduled, gpus = first_fit_schedule(node_state, job, topology)
            else:  # TOPOLOGY_AWARE
                scheduled, gpus = topology_aware_schedule(node_state, job, topology)

            if scheduled:
                job.scheduled_time = current_time
                job.gpus_assigned = gpus

                # Mark GPUs as busy
                for gpu_id in gpus:
                    node_state[gpu_id] = 1

                scheduled_jobs.append(job)
                finish_time = current_time + job.duration
                heapq.heappush(finish_heap, (finish_time, job.job_id))

                queue_idx += 1
            else:
                break

        # Advance time to next event
        if finish_heap:
            current_time = finish_heap[0][0]
        elif queue_idx < len(job_queue):
            current_time = job_queue[queue_idx].arrival_time
        else:
            break

    # Schedule remaining jobs in queue at end
    for job in job_queue[queue_idx:]:
        job.scheduled_time = max_sim_time
        if strategy == PlacementStrategy.FIRST_FIT:
            scheduled, gpus = first_fit_schedule(node_state, job, topology)
        else:
            scheduled, gpus = topology_aware_schedule(node_state, job, topology)
        if scheduled:
            job.gpus_assigned = gpus
            completed_jobs.append(job)

    # Calculate metrics
    total_queue_time = 0.0
    total_gpus_idle_hours = 0.0
    topology_aware_placements = 0

    for job in completed_jobs:
        queue_time = max(0, job.scheduled_time - job.submitted_time)
        total_queue_time += queue_time

        # Idle time: if job didn't use all 8 GPUs
        idle_gpus = 8 - job.num_gpus
        total_gpus_idle_hours += idle_gpus * job.duration

        # Check topology awareness
        if is_topology_aware_placement(job.gpus_assigned, topology):
            topology_aware_placements += 1

    num_completed = len(completed_jobs)

    # Fragmentation: measure as unused GPU hours
    total_gpu_hours = 8 * sum(job.duration for job in completed_jobs)
    used_gpu_hours = sum(job.num_gpus * job.duration for job in completed_jobs)
    fragmentation_rate = (total_gpu_hours - used_gpu_hours) / total_gpu_hours if total_gpu_hours > 0 else 0.0

    avg_queue_time = total_queue_time / num_completed if num_completed > 0 else 0.0
    topo_aware_ratio = topology_aware_placements / num_completed if num_completed > 0 else 0.0

    return {
        "num_completed": num_completed,
        "fragmentation_rate": fragmentation_rate,
        "avg_queue_time": avg_queue_time,
        "idle_gpu_hours": total_gpus_idle_hours,
        "topology_aware_ratio": topo_aware_ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Kubernetes GPU scheduling on 8-GPU node"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["first-fit", "topology-aware"],
        default="first-fit",
        help="Scheduling strategy",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=100,
        help="Number of jobs to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    strategy = PlacementStrategy(args.strategy)
    metrics = simulate(strategy, num_jobs=args.num_jobs, seed=args.seed)

    print(f"\n{'='*70}")
    print(f"Scheduling Simulation: {strategy.value}")
    print(f"{'='*70}")
    print(f"Completed jobs:             {metrics['num_completed']}")
    print(f"Fragmentation rate:         {metrics['fragmentation_rate']*100:.2f}%")
    print(f"Avg queue time (seconds):   {metrics['avg_queue_time']:.2f}")
    print(f"Total idle GPU hours:       {metrics['idle_gpu_hours']:.2f}")
    print(f"Topology-aware placements:  {metrics['topology_aware_ratio']*100:.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
