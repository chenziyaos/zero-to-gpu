#!/usr/bin/env python3
"""
GPU-20: Failure simulation and MTBF calculation.

Scenario: N-card training job with per-card annual failure rate. Failures follow
Poisson process; a single card failure halts the entire job (all-or-nothing).

Key insight: Job-level MTBF shrinks linearly with scale.
  If λ_card = failures per card per year, then
  λ_job = N * λ_card (job fails as soon as any card fails).

Reference: Llama 3 training on 16384 H100s: 419 interruptions in 54 days.
  This implies job MTBF ≈ (54 days * 24 hours) / 419 ≈ 3.08 hours,
  or ~7876 job-level interruptions per year, ~0.48 per card per year (419 / 16384 / 54 * 365).

Monte Carlo: simulate one full training run with random failures, count actual interruptions.
"""

import sys
import os
import argparse
import random
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

def job_mtbf_from_card_failure_rate(num_cards, annual_card_failures):
    """
    Compute job-level MTBF given per-card annual failure rate.
    Assumes failures are independent and Poisson.
    """
    job_annual_failures = num_cards * annual_card_failures
    job_mtbf_hours = 365 * 24 / job_annual_failures if job_annual_failures > 0 else float('inf')
    return job_mtbf_hours * 3600  # Return in seconds

def simulate_training_run(
    total_steps,
    step_time_sec,
    checkpoint_interval_steps,
    checkpoint_time_sec,
    mtbf_sec,
    recovery_time_sec,
    seed=42
):
    """
    Monte Carlo: simulate one training run with random failures.
    Failures occur according to Poisson process with rate 1/MTBF.

    Returns: dict with actual_failures, total_time, ettr, etc.
    """
    random.seed(seed)

    failures = 0
    completed_steps = 0
    total_time = 0.0
    last_checkpoint_step = 0

    while completed_steps < total_steps:
        # How many steps until next checkpoint?
        steps_to_checkpoint = checkpoint_interval_steps - (completed_steps - last_checkpoint_step)
        if completed_steps + steps_to_checkpoint > total_steps:
            steps_to_checkpoint = total_steps - completed_steps

        # Compute time for this segment
        segment_time = steps_to_checkpoint * step_time_sec

        # Check if a failure happens during this segment (Poisson process)
        failure_prob = 1 - math.exp(-segment_time / mtbf_sec)
        if random.random() < failure_prob:
            # Failure occurred, lose work back to last checkpoint
            failures += 1
            total_time += segment_time  # We spent this time and then failed
            total_time += recovery_time_sec  # Recovery + restart
            # Steps stay the same (rolled back to last checkpoint)
        else:
            # No failure, we complete the segment
            total_time += segment_time
            completed_steps += steps_to_checkpoint

            # If we just completed a checkpoint boundary, add checkpoint time
            if (completed_steps % checkpoint_interval_steps == 0 or completed_steps >= total_steps):
                total_time += checkpoint_time_sec
                last_checkpoint_step = completed_steps

    ettr = (total_steps * step_time_sec) / total_time if total_time > 0 else 0

    return {
        'failures': failures,
        'total_time_sec': total_time,
        'ettr': ettr,
        'completed_steps': completed_steps
    }

def main():
    parser = argparse.ArgumentParser(
        description='Failure simulation: job-level MTBF and training interruptions'
    )
    parser.add_argument('--num-cards', type=int, default=8,
                       help='Number of GPU cards in cluster')
    parser.add_argument('--card-annual-failures', type=float, default=0.5,
                       help='Expected failures per card per year (e.g., 0.5 = 1 failure per 2 years)')
    parser.add_argument('--total-steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--step-time', type=float, default=0.1,
                       help='Average wall-clock time per step (seconds)')
    parser.add_argument('--ckpt-interval', type=int, default=1000,
                       help='Checkpoint interval (steps)')
    parser.add_argument('--ckpt-time', type=float, default=600.0,
                       help='Checkpoint write time (seconds)')
    parser.add_argument('--recovery-time', type=float, default=300.0,
                       help='Recovery time (seconds)')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of Monte Carlo runs')
    parser.add_argument('--llama3', action='store_true',
                       help='Use Llama 3 parameters: 16384 cards, 419 failures in 54 days')

    args = parser.parse_args()

    if args.llama3:
        # Llama 3: 16384 cards, 419 failures over 54 days
        num_cards = 16384
        failures_54days = 419
        training_days = 54
        training_hours = training_days * 24
        job_mtbf_hours = training_hours / failures_54days
        expected_annual_failures = 365 * 24 / job_mtbf_hours
        card_annual_failures = expected_annual_failures / num_cards
        print(f"Llama 3 parameters:")
        print(f"  {num_cards} cards, {failures_54days} failures in {training_days} days")
        print(f"  Job MTBF: {job_mtbf_hours:.2f} hours")
        print(f"  Card annual failures: {card_annual_failures:.4f}")
        print()
    else:
        num_cards = args.num_cards
        card_annual_failures = args.card_annual_failures

    job_mtbf_sec = job_mtbf_from_card_failure_rate(num_cards, card_annual_failures)
    job_mtbf_hours = job_mtbf_sec / 3600

    print(f"Configuration:")
    print(f"  {num_cards} cards")
    print(f"  Card annual failure rate: {card_annual_failures:.4f} (1 failure per {1/card_annual_failures:.1f} years)")
    print(f"  Job MTBF: {job_mtbf_hours:.2f} hours")
    print(f"  Total steps: {args.total_steps:,}, step time: {args.step_time} s")
    print(f"  Checkpoint interval: {args.ckpt_interval:,} steps, write time: {args.ckpt_time} s")
    print(f"  Recovery time: {args.recovery_time} s")
    print()

    print(f"Running {args.num_runs} Monte Carlo simulations...")
    print()

    failures_list = []
    ettr_list = []
    total_times = []

    for run in range(args.num_runs):
        result = simulate_training_run(
            args.total_steps,
            args.step_time,
            args.ckpt_interval,
            args.ckpt_time,
            job_mtbf_sec,
            args.recovery_time,
            seed=42 + run
        )
        failures_list.append(result['failures'])
        ettr_list.append(result['ettr'])
        total_times.append(result['total_time_sec'])

    avg_failures = sum(failures_list) / len(failures_list)
    avg_ettr = sum(ettr_list) / len(ettr_list)
    avg_total_time = sum(total_times) / len(total_times)

    print(f"Results over {args.num_runs} runs:")
    print(f"  Average failures per run: {avg_failures:.1f}")
    print(f"  Average ETTR (goodput): {avg_ettr*100:.1f}%")
    print(f"  Average total time: {avg_total_time/3600:.2f} hours")
    print(f"  Min failures: {min(failures_list)}, Max: {max(failures_list)}")
    print(f"  Min ETTR: {min(ettr_list)*100:.1f}%, Max: {max(ettr_list)*100:.1f}%")

if __name__ == '__main__':
    main()
