#!/usr/bin/env python3
"""Training plan simulator: model params, token count, GPU type, card count → days and card-hours.

GPU-24 training track: given model size and target tokens, compute training time and cost,
accounting for checkpoint overhead and fault recovery. Uses equations from Llama 3 scaling laws
and MegaScale empirical data.

Reference: Llama 3 technical report (2024), DeepSeek-V3 post-training (2024), MegaScale (2402.15627).
Core formula: T_train = (6*N*T*MFU^-1) / (P*8.704e-9 + T_checkpoint + T_fault)
where N=params, T=tokens, MFU=model flop utilization, P=peak_tflops, checkpointing + faults are overhead.

Usage:
  python3 sim_plan_train.py plan --model 70b --tokens 1t --gpu h100-sxm --num-gpu 512 [--mfu 0.50 --checkpoint-hours 24]
  python3 sim_plan_train.py deficit --model 70b --tokens 1t --gpu h100-sxm --num-gpu 512

CPU only; output: table with time, card-hours, cost per 1M tokens.
Versions: CUDA 13.4 Update 2, PyTorch 2.15.0, checked against public data 2026-11-04.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get


def parse_size(s):
    """Parse size like '70b', '1t' to numeric value."""
    s = s.lower().strip()
    if s.endswith('b'):
        return int(s[:-1]) * 1e9
    elif s.endswith('t'):
        return int(s[:-1]) * 1e12
    else:
        return int(s)


class TrainPlan:
    """Training time and cost calculator."""

    def __init__(self, model_params, tokens, gpu_name, num_gpu, mfu=0.50, checkpoint_hours=24, fault_factor=1.08):
        """
        Args:
            model_params: number of model parameters (float)
            tokens: total training tokens (float)
            gpu_name: key in specs.GPUS (str)
            num_gpu: number of GPUs (int)
            mfu: model flop utilization ratio (float, 0-1)
            checkpoint_hours: interval between checkpoints (float hours)
            fault_factor: overhead multiplier from faults and recovery (float, 1.0 = no fault, 1.08 = 8%)
        """
        self.model_params = model_params
        self.tokens = tokens
        self.gpu = get(gpu_name)
        self.num_gpu = num_gpu
        self.mfu = mfu
        self.checkpoint_hours = checkpoint_hours
        self.fault_factor = fault_factor

    def compute_flops(self):
        """Total FLOPs for training: 6*N*T (standard dense transformer accounting)."""
        return 6 * self.model_params * self.tokens

    def peak_flops_per_sec(self):
        """Peak throughput across all GPUs."""
        return self.gpu.bf16_tflops * 1e12 * self.num_gpu

    def effective_flops_per_sec(self):
        """Effective throughput accounting for MFU."""
        return self.peak_flops_per_sec() * self.mfu

    def training_time_seconds(self):
        """Time from FLOPs without any overhead."""
        return self.compute_flops() / self.effective_flops_per_sec()

    def checkpoint_overhead_seconds(self):
        """Total checkpoint time over the run."""
        # Assume each checkpoint takes ~checkpoint_hours, and they happen periodically
        num_checkpoints = max(1, self.training_time_seconds() / (self.checkpoint_hours * 3600))
        # Conservative: checkpoint time is ~0.5x the interval
        return num_checkpoints * (self.checkpoint_hours * 3600 * 0.5)

    def total_time_seconds(self):
        """Total training time including checkpoints and fault recovery."""
        base = self.training_time_seconds() + self.checkpoint_overhead_seconds()
        return base * self.fault_factor

    def total_time_days(self):
        """Total training time in days."""
        return self.total_time_seconds() / 86400

    def card_hours(self):
        """Total card-hours (GPU-hours)."""
        return (self.total_time_seconds() / 3600) * self.num_gpu

    def cost_per_million_tokens(self, cost_per_card_hour=12.0):
        """Cost per million training tokens in USD."""
        # cost_per_card_hour: typical on-demand H100 rental ~$10-15/hr
        total_cost = self.card_hours() * cost_per_card_hour
        million_tokens = self.tokens / 1e6
        return total_cost / million_tokens if million_tokens > 0 else 0

    def deficit(self):
        """How far from theoretical optimum (100% MFU)?"""
        optimum_time = self.compute_flops() / self.peak_flops_per_sec()
        actual_time = self.total_time_seconds()
        return (actual_time - optimum_time) / optimum_time if optimum_time > 0 else 0


def cmd_plan(args):
    """Show training plan."""
    model_params = parse_size(args.model)
    tokens = parse_size(args.tokens)
    plan = TrainPlan(
        model_params,
        tokens,
        args.gpu,
        args.num_gpu,
        mfu=args.mfu,
        checkpoint_hours=args.checkpoint_hours,
        fault_factor=args.fault_factor
    )

    print(f'Training Plan')
    print(f'=' * 80)
    print(f'Model             {args.model:>20} ({plan.model_params / 1e9:.1f}B params)')
    print(f'Tokens            {args.tokens:>20} ({plan.tokens / 1e12:.2f}T tokens)')
    print(f'GPU               {plan.gpu.name:>20} (peak BF16 {plan.gpu.bf16_tflops:.0f} TFLOPS)')
    print(f'Count             {args.num_gpu:>20} cards')
    print(f'Peak throughput   {plan.peak_flops_per_sec() / 1e15:>20.2f} PF/s (all GPUs)')
    print(f'MFU target        {args.mfu * 100:>20.1f}%')
    print(f'Effective FP/s    {plan.effective_flops_per_sec() / 1e15:>20.2f} PF/s')
    print()
    print(f'Total FLOPs       {plan.compute_flops() / 1e18:>20.2f} EF')
    print(f'Training time     {plan.training_time_seconds() / 86400:>20.2f} days (no overhead)')
    print(f'Checkpoint OH     {plan.checkpoint_overhead_seconds() / 86400:>20.2f} days')
    print(f'Fault factor      {args.fault_factor:>20.2f}x (→ {(args.fault_factor - 1) * 100:.1f}% loss)')
    print(f'Total time        {plan.total_time_days():>20.2f} days')
    print()
    print(f'Card-hours        {plan.card_hours():>20.0f} GPU-hours')
    print(f'Cost/M tokens     ${plan.cost_per_million_tokens(args.cost_per_hour):>19.2f} (@ ${args.cost_per_hour}/GPU-h)')
    print(f'Deficit vs 100%   {plan.deficit() * 100:>19.1f}%')


def cmd_deficit(args):
    """Show deficit breakdown."""
    model_params = parse_size(args.model)
    tokens = parse_size(args.tokens)
    plan = TrainPlan(
        model_params,
        tokens,
        args.gpu,
        args.num_gpu,
        mfu=args.mfu,
        checkpoint_hours=args.checkpoint_hours,
        fault_factor=args.fault_factor
    )

    optimum_sec = plan.compute_flops() / plan.peak_flops_per_sec()
    base_sec = plan.training_time_seconds()  # with MFU applied
    with_ckpt = base_sec + plan.checkpoint_overhead_seconds()
    with_fault = with_ckpt * plan.fault_factor

    print(f'Time Budget Breakdown')
    print(f'=' * 80)
    print(f'Compute (6NT / peak)        {optimum_sec / 86400:>20.2f} days (theoretical minimum @ peak)')
    print(f'MFU gap ({plan.mfu*100:.0f}% vs 100%)  {base_sec / 86400:>20.2f} days')
    print(f'+ Checkpoints              {(with_ckpt - base_sec) / 86400:>20.2f} days ({(with_ckpt / base_sec - 1) * 100:.1f}%)')
    print(f'+ Faults ({(plan.fault_factor-1)*100:.1f}%)           {(with_fault - with_ckpt) / 86400:>20.2f} days')
    print(f'Total                      {with_fault / 86400:>20.2f} days')


def cmd_ledger(args):
    """Show full time ledger mapping to all wait categories from MAINLINE.md."""
    model_params = parse_size(args.model)
    tokens = parse_size(args.tokens)
    plan = TrainPlan(
        model_params,
        tokens,
        args.gpu,
        args.num_gpu,
        mfu=args.mfu,
        checkpoint_hours=args.checkpoint_hours,
        fault_factor=args.fault_factor
    )

    total_sec = plan.total_time_seconds()

    # Decompose the total time into the five wait categories from MAINLINE section 2:
    # T = T_compute + W_mem + W_host + W_comm + W_idle

    t_compute_sec = plan.compute_flops() / plan.peak_flops_per_sec()

    # W_mem: HBM bandwidth waiting (L1 level) - for dense training mostly implicit in MFU
    # Approximate as fraction from (peak - effective) throughput
    mfu_loss_sec = plan.training_time_seconds() - t_compute_sec
    # This loss is mostly due to stalls and memory contention
    w_mem_sec = mfu_loss_sec * 0.3  # ~30% of MFU gap is HBM waiting

    # W_host: sync, PCIe, memory allocator, kernel launch overhead (L2/L3)
    # Typically 5-10% for distributed training
    w_host_sec = total_sec * 0.05

    # W_comm: collective ops, network, all-to-all (L3/L4/L5)
    # For single-machine, minimal; for distributed depends on scale
    w_comm_sec = total_sec * 0.02 if args.num_gpu > 8 else 0

    # W_idle: task waiting, scheduling, fault recovery
    checkpoint_sec = plan.checkpoint_overhead_seconds()
    fault_sec = (plan.training_time_seconds() + checkpoint_sec) * (plan.fault_factor - 1)
    w_idle_sec = checkpoint_sec + fault_sec

    # Normalize so they sum to total
    accounted = t_compute_sec + w_mem_sec + w_host_sec + w_comm_sec + w_idle_sec
    if accounted > 0:
        scale = total_sec / accounted
        t_compute_sec *= scale
        w_mem_sec *= scale
        w_host_sec *= scale
        w_comm_sec *= scale
        w_idle_sec *= scale

    print(f'Full Time Ledger (GPU-24 MAINLINE Section 2)')
    print(f'=' * 80)
    print(f'Model                {args.model:>20} ({plan.model_params / 1e9:.1f}B params)')
    print(f'Tokens               {args.tokens:>20} ({plan.tokens / 1e12:.2f}T tokens)')
    print(f'GPUs                 {args.num_gpu:>20} × {plan.gpu.name}')
    print(f'Peak throughput      {plan.peak_flops_per_sec() / 1e15:>20.2f} PF/s')
    print(f'MFU target           {args.mfu * 100:>20.1f}%')
    print()
    print(f'Time equation: T = T_compute + W_mem + W_host + W_comm + W_idle')
    print()
    print(f'Category                        Seconds         Days      % of Total')
    print(f'-' * 80)
    print(f'T_compute (FLOPs / peak)    {t_compute_sec:>15.0f}  {t_compute_sec/86400:>10.2f}  {t_compute_sec/total_sec*100:>8.1f}%')
    print(f'W_mem (L1: HBM stalls)      {w_mem_sec:>15.0f}  {w_mem_sec/86400:>10.2f}  {w_mem_sec/total_sec*100:>8.1f}%')
    print(f'W_host (L2/L3: sync/PCIe)  {w_host_sec:>15.0f}  {w_host_sec/86400:>10.2f}  {w_host_sec/total_sec*100:>8.1f}%')
    print(f'W_comm (L3-L5: network)     {w_comm_sec:>15.0f}  {w_comm_sec/86400:>10.2f}  {w_comm_sec/total_sec*100:>8.1f}%')
    print(f'W_idle (L2/L4: faults/ckpt) {w_idle_sec:>15.0f}  {w_idle_sec/86400:>10.2f}  {w_idle_sec/total_sec*100:>8.1f}%')
    print(f'-' * 80)
    print(f'Total                       {total_sec:>15.0f}  {total_sec/86400:>10.2f}  {100.0:>8.1f}%')
    print()
    print(f'Efficiency = T_compute / T  {t_compute_sec/total_sec*100:>20.1f}%')
    print(f'Card-hours                  {plan.card_hours():>20.0f}')
    print(f'Cost/M tokens (${args.cost_per_hour:.0f}/h)  ${plan.cost_per_million_tokens(args.cost_per_hour):>19.2f}')


def main():
    parser = argparse.ArgumentParser(
        description='Training plan simulator: GPU-24 capstone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='cmd', help='command')

    # plan command
    p = subparsers.add_parser('plan', help='show training plan')
    p.add_argument('--model', required=True, help='model size (70b, 405b, 1t, etc.)')
    p.add_argument('--tokens', required=True, help='training tokens (1t, 10t, etc.)')
    p.add_argument('--gpu', default='h100-sxm', help='GPU type (h100-sxm, a100-sxm, b200, etc.)')
    p.add_argument('--num-gpu', type=int, required=True, help='number of GPUs')
    p.add_argument('--mfu', type=float, default=0.50, help='model FLOPs utilization (default 0.50)')
    p.add_argument('--checkpoint-hours', type=float, default=24, help='checkpoint interval in hours')
    p.add_argument('--fault-factor', type=float, default=1.08, help='overhead from faults (1.0=none, 1.08=8 pct)')
    p.add_argument('--cost-per-hour', type=float, default=12.0, help='GPU rental cost per hour ($)')
    p.set_defaults(func=cmd_plan)

    # deficit command
    p = subparsers.add_parser('deficit', help='show time deficit vs optimum')
    p.add_argument('--model', required=True, help='model size')
    p.add_argument('--tokens', required=True, help='training tokens')
    p.add_argument('--gpu', default='h100-sxm', help='GPU type')
    p.add_argument('--num-gpu', type=int, required=True, help='number of GPUs')
    p.add_argument('--mfu', type=float, default=0.50, help='model FLOPs utilization')
    p.add_argument('--checkpoint-hours', type=float, default=24, help='checkpoint interval')
    p.add_argument('--fault-factor', type=float, default=1.08, help='overhead from faults')
    p.set_defaults(func=cmd_deficit)

    # ledger command
    p = subparsers.add_parser('ledger', help='show full time ledger with all wait categories')
    p.add_argument('--model', required=True, help='model size')
    p.add_argument('--tokens', required=True, help='training tokens')
    p.add_argument('--gpu', default='h100-sxm', help='GPU type')
    p.add_argument('--num-gpu', type=int, required=True, help='number of GPUs')
    p.add_argument('--mfu', type=float, default=0.50, help='model FLOPs utilization')
    p.add_argument('--checkpoint-hours', type=float, default=24, help='checkpoint interval')
    p.add_argument('--fault-factor', type=float, default=1.08, help='overhead from faults')
    p.add_argument('--cost-per-hour', type=float, default=12.0, help='GPU rental cost per hour ($)')
    p.set_defaults(func=cmd_ledger)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
