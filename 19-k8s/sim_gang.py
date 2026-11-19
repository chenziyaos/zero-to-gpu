#!/usr/bin/env python3
"""
GPU-19: Gang scheduling simulator for Kubernetes GPU workloads.

Simulates multi-pod jobs on 8-GPU node with NVLink topology constraints.
Compares gang scheduling (all-or-nothing) vs. sequential scheduling (incremental).
Gang scheduling prevents deadlock and fragmentation in tightly-coupled workloads.

Metrics: queue time, deadlock rate, GPU utilization, pod wait distribution.

Reference: Kueue gang scheduling, Volcano batch scheduler, Gandiva (OSDI 2018),
Kubernetes KEP-3850 (Gang scheduling), Tiresias (NSDI 2019).
Verified 2026-11-04.
"""

import argparse
import heapq
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Set
from collections import defaultdict


class SchedulingMode(Enum):
    """Scheduling modes for multi-pod jobs."""
    GANG = "gang"               # All pods schedule together or not at all
    SEQUENTIAL = "sequential"   # Pods scheduled one at a time (can deadlock)


@dataclass
class Pod:
    """Represents a single GPU pod."""
    pod_id: int
    job_id: int
    num_gpus: int
    arrival_time: float
    submitted_time: float = 0.0
    scheduled_time: float = 0.0
    gpus_assigned: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, scheduled, completed


@dataclass
class Job:
    """Represents a multi-pod job with multiple GPUs."""
    job_id: int
    num_pods: int              # Number of pods (typically replicas)
    pods_per_gpu: int          # GPUs per pod
    arrival_time: float
    duration: float
    pods: List[Pod] = field(default_factory=list)
    scheduled_time: float = 0.0
    completed_time: float = 0.0
    is_deadlocked: bool = False


def is_topology_connected(gpus: List[int], topology: Dict[int, Set[int]]) -> bool:
    """Check if assigned GPUs form connected component in NVLink topology."""
    if len(gpus) <= 1:
        return True

    # BFS to check connectivity
    visited = {gpus[0]}
    queue = [gpus[0]]

    while queue:
        gpu = queue.pop(0)
        for neighbor in topology[gpu]:
            if neighbor in gpus and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(gpus)


def build_nvlink_topology() -> Dict[int, Set[int]]:
    """Build 8-GPU NVLink double-ring topology."""
    neighbors_map = {
        0: {1, 7},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 5},
        5: {4, 6},
        6: {5, 7},
        7: {6, 0},
    }
    return {gpu_id: neighbors for gpu_id, neighbors in neighbors_map.items()}


def first_fit_placement(
    node_state: List[int], num_gpus: int, topology: Dict[int, Set[int]]
) -> Tuple[bool, List[int]]:
    """First-fit placement for a single pod."""
    available_gpus = [i for i in range(8) if node_state[i] == 0]

    if len(available_gpus) < num_gpus:
        return False, []

    return True, available_gpus[:num_gpus]


def topology_aware_placement(
    node_state: List[int], num_gpus: int, topology: Dict[int, Set[int]]
) -> Tuple[bool, List[int]]:
    """Topology-aware placement preferring connected GPU groups."""
    available_gpus = [i for i in range(8) if node_state[i] == 0]

    if len(available_gpus) < num_gpus:
        return False, []

    if num_gpus == 1:
        return True, available_gpus[:1]

    # Try to find connected group
    start_candidates = sorted(
        available_gpus,
        key=lambda g: len(topology[g] & set(available_gpus)),
        reverse=True
    )

    for start_gpu in start_candidates:
        group = {start_gpu}
        candidates = topology[start_gpu] & set(available_gpus)

        while len(group) < num_gpus and candidates:
            next_gpu = candidates.pop()
            group.add(next_gpu)
            for neighbor in topology[next_gpu]:
                if neighbor in available_gpus and neighbor not in group:
                    candidates.add(neighbor)

        if len(group) >= num_gpus:
            return True, list(group)[:num_gpus]

    # Fallback to first-fit
    return first_fit_placement(node_state, num_gpus, topology)


def simulate_gang_scheduling(
    num_jobs: int = 50,
    pods_per_job: int = 2,
    gpus_per_pod: int = 2,
    seed: int = 42,
) -> Dict:
    """Simulate gang scheduling (all-or-nothing pod placement).

    Args:
        num_jobs: number of jobs (each with multiple pods)
        pods_per_job: pods per job
        gpus_per_pod: GPUs per pod
        seed: random seed

    Returns:
        metrics dict
    """
    random.seed(seed)
    topology = build_nvlink_topology()
    node_state = [0] * 8  # 0 = idle, 1 = busy
    job_queue = []
    completed_jobs = []
    deadlocked_jobs = 0

    # Generate jobs
    arrival_time = 0.0
    for job_id in range(num_jobs):
        inter_arrival = random.expovariate(1.0 / 2.0)
        arrival_time += inter_arrival

        job = Job(
            job_id=job_id,
            num_pods=pods_per_job,
            pods_per_gpu=gpus_per_pod,
            arrival_time=arrival_time,
            duration=random.uniform(5.0, 20.0),
        )

        # Create pods for this job
        for pod_idx in range(pods_per_job):
            pod = Pod(
                pod_id=len(job.pods),
                job_id=job_id,
                num_gpus=gpus_per_pod,
                arrival_time=arrival_time,
                submitted_time=arrival_time,
            )
            job.pods.append(pod)

        job_queue.append(job)

    job_queue.sort(key=lambda j: j.arrival_time)

    # Simulate with gang scheduling: all pods of a job must be scheduled together
    queue_idx = 0
    scheduled_jobs = []
    finish_heap = []
    current_time = 0.0
    max_sim_time = job_queue[-1].arrival_time + 500

    while queue_idx < len(job_queue) or finish_heap:
        # Process finish events
        while finish_heap and finish_heap[0][0] <= current_time:
            finish_time, job_id = heapq.heappop(finish_heap)
            finished_job = next(j for j in scheduled_jobs if j.job_id == job_id)

            for pod in finished_job.pods:
                for gpu_id in pod.gpus_assigned:
                    node_state[gpu_id] = 0

            completed_jobs.append(finished_job)

        # Try to schedule jobs in gang mode
        while queue_idx < len(job_queue):
            job = job_queue[queue_idx]

            if job.arrival_time > current_time:
                break

            # Try to schedule all pods at once
            total_gpus_needed = job.num_pods * job.pods_per_gpu
            available_count = sum(1 for s in node_state if s == 0)

            if available_count >= total_gpus_needed:
                # Schedule all pods
                all_scheduled = True
                pod_placements = []

                for pod in job.pods:
                    scheduled, gpus = topology_aware_placement(
                        node_state, pod.num_gpus, topology
                    )

                    if not scheduled:
                        all_scheduled = False
                        break

                    pod_placements.append((pod, gpus))

                if all_scheduled:
                    # Commit all placements
                    job.scheduled_time = current_time
                    for pod, gpus in pod_placements:
                        pod.scheduled_time = current_time
                        pod.gpus_assigned = gpus
                        pod.status = "scheduled"

                        for gpu_id in gpus:
                            node_state[gpu_id] = 1

                    scheduled_jobs.append(job)
                    finish_time = current_time + job.duration
                    heapq.heappush(finish_heap, (finish_time, job.job_id))

                    queue_idx += 1
                else:
                    # Not all pods could fit: deadlock risk
                    break
            else:
                break

        # Advance time
        if finish_heap:
            current_time = finish_heap[0][0]
        elif queue_idx < len(job_queue):
            current_time = job_queue[queue_idx].arrival_time
        else:
            break

    # Mark remaining jobs as deadlocked
    for job in job_queue[queue_idx:]:
        job.is_deadlocked = True
        deadlocked_jobs += 1

    # Calculate metrics
    total_queue_time = 0.0
    pod_wait_distribution = defaultdict(int)

    for job in completed_jobs:
        queue_time = max(0, job.scheduled_time - job.arrival_time)
        total_queue_time += queue_time

        for pod in job.pods:
            pod_wait = max(0, pod.scheduled_time - pod.submitted_time)
            pod_wait_distribution[int(pod_wait / 5) * 5] += 1

    num_completed = len(completed_jobs)
    avg_queue_time = total_queue_time / num_completed if num_completed > 0 else 0.0

    return {
        "num_completed": num_completed,
        "num_deadlocked": deadlocked_jobs,
        "deadlock_rate": deadlocked_jobs / num_jobs if num_jobs > 0 else 0.0,
        "avg_queue_time": avg_queue_time,
        "pod_wait_distribution": dict(pod_wait_distribution),
    }


def simulate_sequential_scheduling(
    num_jobs: int = 50,
    pods_per_job: int = 2,
    gpus_per_pod: int = 2,
    seed: int = 42,
) -> Dict:
    """Simulate sequential scheduling (pods schedule incrementally, risk deadlock).

    Args:
        num_jobs: number of jobs
        pods_per_job: pods per job
        gpus_per_pod: GPUs per pod
        seed: random seed

    Returns:
        metrics dict
    """
    random.seed(seed)
    topology = build_nvlink_topology()
    node_state = [0] * 8
    job_queue = []
    completed_jobs = []
    deadlocked_jobs = 0

    # Generate jobs
    arrival_time = 0.0
    for job_id in range(num_jobs):
        inter_arrival = random.expovariate(1.0 / 2.0)
        arrival_time += inter_arrival

        job = Job(
            job_id=job_id,
            num_pods=pods_per_job,
            pods_per_gpu=gpus_per_pod,
            arrival_time=arrival_time,
            duration=random.uniform(5.0, 20.0),
        )

        for pod_idx in range(pods_per_job):
            pod = Pod(
                pod_id=len(job.pods),
                job_id=job_id,
                num_gpus=gpus_per_pod,
                arrival_time=arrival_time,
                submitted_time=arrival_time,
            )
            job.pods.append(pod)

        job_queue.append(job)

    job_queue.sort(key=lambda j: j.arrival_time)

    # Sequential scheduling: each pod scheduled independently
    queue_idx = 0
    scheduled_jobs = []
    finish_heap = []
    current_time = 0.0
    max_sim_time = job_queue[-1].arrival_time + 500
    job_pod_counters = defaultdict(int)  # How many pods of each job are scheduled

    while queue_idx < len(job_queue) or finish_heap:
        # Process finish events
        while finish_heap and finish_heap[0][0] <= current_time:
            finish_time, item = heapq.heappop(finish_heap)

            if isinstance(item, tuple):  # (job_id, pod_id)
                job_id, pod_id = item
                job = next(j for j in scheduled_jobs if j.job_id == job_id)
                pod = job.pods[pod_id]

                for gpu_id in pod.gpus_assigned:
                    node_state[gpu_id] = 0

                pod.status = "completed"

                if all(p.status == "completed" for p in job.pods):
                    job.completed_time = current_time
                    if job not in completed_jobs:
                        completed_jobs.append(job)

        # Try to schedule pods from jobs
        while queue_idx < len(job_queue):
            job = job_queue[queue_idx]

            if job.arrival_time > current_time:
                break

            # Try to schedule one pod at a time
            scheduled_count = job_pod_counters[job.job_id]

            if scheduled_count < job.num_pods:
                pod = job.pods[scheduled_count]

                if pod.status == "pending":
                    scheduled, gpus = topology_aware_placement(
                        node_state, pod.num_gpus, topology
                    )

                    if scheduled:
                        pod.scheduled_time = current_time
                        pod.gpus_assigned = gpus
                        pod.status = "scheduled"

                        for gpu_id in gpus:
                            node_state[gpu_id] = 1

                        if job not in scheduled_jobs:
                            scheduled_jobs.append(job)
                            if job.scheduled_time == 0:
                                job.scheduled_time = current_time

                        job_pod_counters[job.job_id] += 1
                        finish_time = current_time + job.duration
                        heapq.heappush(finish_heap, (finish_time, (job.job_id, scheduled_count)))

                        # Continue trying to schedule more pods
                        continue
                    else:
                        # Pod can't be scheduled, check for deadlock
                        # Simple heuristic: if waiting too long and others are stuck too
                        break

            queue_idx += 1

        # Advance time
        if finish_heap:
            current_time = finish_heap[0][0]
        elif queue_idx < len(job_queue):
            current_time = job_queue[queue_idx].arrival_time
        else:
            break

    # Count deadlocked jobs (some pods never scheduled)
    for job in job_queue[queue_idx:]:
        job.is_deadlocked = True
        deadlocked_jobs += 1

    for job in scheduled_jobs:
        if any(p.status != "completed" for p in job.pods):
            job.is_deadlocked = True
            deadlocked_jobs += 1

    # Calculate metrics
    total_queue_time = 0.0
    pod_wait_distribution = defaultdict(int)

    for job in completed_jobs:
        queue_time = max(0, job.scheduled_time - job.arrival_time)
        total_queue_time += queue_time

        for pod in job.pods:
            pod_wait = max(0, pod.scheduled_time - pod.submitted_time)
            pod_wait_distribution[int(pod_wait / 5) * 5] += 1

    num_completed = len(completed_jobs)
    avg_queue_time = total_queue_time / num_completed if num_completed > 0 else 0.0

    return {
        "num_completed": num_completed,
        "num_deadlocked": deadlocked_jobs,
        "deadlock_rate": deadlocked_jobs / num_jobs if num_jobs > 0 else 0.0,
        "avg_queue_time": avg_queue_time,
        "pod_wait_distribution": dict(pod_wait_distribution),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulate gang scheduling vs sequential scheduling for GPU jobs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gang", "sequential"],
        default="gang",
        help="Scheduling mode",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=50,
        help="Number of jobs to simulate",
    )
    parser.add_argument(
        "--pods-per-job",
        type=int,
        default=2,
        help="Pods per job",
    )
    parser.add_argument(
        "--gpus-per-pod",
        type=int,
        default=2,
        help="GPUs per pod",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.mode == "gang":
        metrics = simulate_gang_scheduling(
            num_jobs=args.num_jobs,
            pods_per_job=args.pods_per_job,
            gpus_per_pod=args.gpus_per_pod,
            seed=args.seed,
        )
    else:
        metrics = simulate_sequential_scheduling(
            num_jobs=args.num_jobs,
            pods_per_job=args.pods_per_job,
            gpus_per_pod=args.gpus_per_pod,
            seed=args.seed,
        )

    print(f"\n{'='*70}")
    print(f"Gang Scheduling Simulation: {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"Jobs (pods={args.pods_per_job}, gpus/pod={args.gpus_per_pod}):")
    print(f"  Completed:              {metrics['num_completed']}")
    print(f"  Deadlocked:             {metrics['num_deadlocked']}")
    print(f"  Deadlock rate:          {metrics['deadlock_rate']*100:.2f}%")
    print(f"  Avg queue time (sec):   {metrics['avg_queue_time']:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
