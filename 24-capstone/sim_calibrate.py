#!/usr/bin/env python3
"""Calibration table: estimated vs actual training metrics for published models.

GPU-24 calibration track: builds a lookup table of public LLM training runs,
comparing our estimation (MFU, days, card-hours) against reported actuals.
Helps validate the sim_plan_train.py model and calibrate MFU assumptions.

Datasets:
  - Llama 3 405B (Meta 2024): 15.6T tokens, 16128 H100-SXM, 30.84M GPU-hours → ~79.75 days (HF model card)
  - DeepSeek-V3 (DeepSeek 2024): 14.8T tokens, 2048 H800, 2.664M GPU-hours → ~54.4 days pretraining (arxiv 2412.19437)

Reference: Llama 3.1 405B model card, DeepSeek-V3 technical report (arXiv 2412.19437).
TPU runs such as PaLM are left out: specs.py has no TPU entry, so their MFU cannot be back-solved here.

Usage:
  python3 sim_calibrate.py list
  python3 sim_calibrate.py verify llama3-405b [--mfu 0.50 --fault-factor 1.08]

CPU only; outputs comparison table with error metrics.
"""
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get


@dataclass
class PublicRun:
    """Published training run baseline."""
    name: str
    model_params_b: float        # Billions
    total_tokens_t: float        # Trillions
    gpu_type: str                # specs key
    num_gpu: int
    reported_days: float         # Days from paper
    reported_card_hours: float   # Card-hours if given, else None
    source: str                  # Paper / blog URL snippet


RUNS = [
    PublicRun(
        name='Llama 3 405B',
        model_params_b=405,
        total_tokens_t=15.6,  # 15T from paper, 15.6T exact; HF model card: ~15T
        gpu_type='h100-sxm',
        num_gpu=16128,
        reported_days=79.75,  # 30.84M GPU-hours / (16128 * 24) = ~79.75 days from HF model card; NOT 168 (source: HuggingFace meta-llama/Llama-3.1-405B model card https://huggingface.co/meta-llama/Llama-3.1-405B)
        reported_card_hours=30.84e6,  # 30.84M GPU-hours on H100-80GB from HF model card
        source='HuggingFace meta-llama/Llama-3.1-405B model card, 30.84M GPU-hours'
    ),
    PublicRun(
        name='DeepSeek-V3 (MoE: 37B active / 671B total)',
        model_params_b=37,  # 37B activated parameters per token (MoE: only active params count for FLOPs in 6NT formula)
        total_tokens_t=14.8,
        gpu_type='h800',
        num_gpu=2048,  # Actual 2048 H800 from arxiv 2412.19437
        reported_days=54.4,  # 2.664M GPU-hours / (2048 * 24) = ~54.4 days pretraining; NOT 163 (source: arxiv 2412.19437 Table 1: 14.8T tokens, 2.664M GPU-hours)
        reported_card_hours=2.664e6,  # 2.664M GPU-hours pretraining from arxiv 2412.19437
        source='DeepSeek-V3 arxiv 2412.19437: 37B activated params (MoE), 14.8T tokens, 2.664M pretraining GPU-hours on 2048 H800'
    ),
]


def compute_flops(model_b, tokens_t):
    """Standard dense transformer: 6 * N * T FLOPs."""
    return 6 * (model_b * 1e9) * (tokens_t * 1e12)


def estimate_days(model_b, tokens_t, gpu_type, num_gpu, mfu=0.5, checkpoint_oh=0.15, fault_factor=1.05):
    """Estimate training time using sim_plan_train logic.

    Note: checkpoint_oh and fault_factor are separate from the 6*N*T FLOPs calculation.
    In practice, Llama 3 and DeepSeek report days that include these overheads.
    Conservative defaults: 15% checkpoint overhead, 5% fault recovery.
    """
    gpu = get(gpu_type)
    flops = compute_flops(model_b, tokens_t)
    peak_flops = gpu.bf16_tflops * 1e12 * num_gpu
    effective_flops = peak_flops * mfu
    base_seconds = flops / effective_flops
    # Checkpoint overhead: assume checkpoint_oh fraction of base time
    ckpt_seconds = base_seconds * checkpoint_oh
    total_seconds = (base_seconds + ckpt_seconds) * fault_factor
    return total_seconds / 86400


def estimate_mfu(model_b, tokens_t, gpu_type, num_gpu, reported_days):
    """Back-solve MFU from reported days."""
    gpu = get(gpu_type)
    flops = compute_flops(model_b, tokens_t)
    reported_seconds = reported_days * 86400
    peak_flops = gpu.bf16_tflops * 1e12 * num_gpu
    effective_flops = flops / reported_seconds
    return effective_flops / peak_flops


def cmd_list():
    """List all public runs."""
    print('Public Training Runs for Calibration')
    print('=' * 100)
    print(f'{"Model":20} {"Params":>8} {"Tokens":>8} {"GPU":>12} {"#GPU":>7} {"Days":>8} {"MFU*":>6} {"Source":>30}')
    print('-' * 100)
    for run in RUNS:
        mfu = estimate_mfu(run.model_params_b, run.total_tokens_t, run.gpu_type,
                          run.num_gpu, run.reported_days)
        print(f'{run.name:20} {run.model_params_b:>8.0f}B {run.total_tokens_t:>8.2f}T {get(run.gpu_type).name:>12} '
              f'{run.num_gpu:>7} {run.reported_days:>8.0f} {mfu * 100:>5.1f}% {run.source[:28]:>30}')
    print()
    print('* MFU solved backwards from reported days; includes checkpoint and fault overhead.')


def cmd_verify(name, mfu=0.50, checkpoint_oh=0.5, fault_factor=1.08):
    """Verify estimation against one public run."""
    run = None
    for r in RUNS:
        if name.lower() in r.name.lower():
            run = r
            break
    if not run:
        print(f'Not found: {name}. Available: {", ".join(r.name for r in RUNS)}')
        return

    gpu = get(run.gpu_type)

    # Back-solve actual MFU from reported days
    actual_mfu = estimate_mfu(run.model_params_b, run.total_tokens_t, run.gpu_type,
                             run.num_gpu, run.reported_days)

    # Estimate with given MFU
    est_days = estimate_days(run.model_params_b, run.total_tokens_t, run.gpu_type,
                            run.num_gpu, mfu=mfu, checkpoint_oh=checkpoint_oh, fault_factor=fault_factor)

    # Breakdown
    flops = compute_flops(run.model_params_b, run.total_tokens_t)
    peak_flops = gpu.bf16_tflops * 1e12 * run.num_gpu

    compute_days = (flops / (peak_flops * actual_mfu)) / 86400  # Theoretical at actual MFU
    base_days_est = (flops / (peak_flops * mfu)) / 86400

    est_ckpt_days = base_days_est * checkpoint_oh
    est_fault_days = (base_days_est + est_ckpt_days) * (fault_factor - 1)

    card_hours_reported = run.reported_days * 24 * run.num_gpu if run.reported_card_hours is None else run.reported_card_hours
    card_hours_est = est_days * 24 * run.num_gpu

    cost_est = card_hours_est * 12.0 / 1e6 * (run.total_tokens_t * 1e12)  # $ per token

    print(f'Verification: {run.name}')
    print(f'=' * 80)
    print()
    print(f'Model & Hardware')
    print(f'  Model                     {run.model_params_b:.0f}B params')
    print(f'  Total tokens              {run.total_tokens_t:.2f}T')
    print(f'  GPU                       {gpu.name}')
    print(f'  Count                     {run.num_gpu}')
    print()
    print(f'Time Accounting')
    print(f'  Reported                  {run.reported_days:.1f} days')
    print(f'  Estimated (MFU={mfu:.0%})     {est_days:.1f} days ({(est_days - run.reported_days) / run.reported_days * 100:+.1f} %)')
    print(f'  Error                     {est_days - run.reported_days:+.1f} days')
    print()
    print(f'MFU Breakdown')
    print(f'  Peak throughput           {peak_flops / 1e15:.1f} PF/s')
    print(f'  Actual (back-solved)      {actual_mfu * 100:.1f}%')
    print(f'  Estimate input            {mfu * 100:.1f}%')
    print()
    print(f'Overhead Attribution (estimated)')
    print(f'  Base compute              {base_days_est:.1f} days')
    print(f'  Checkpoint ({checkpoint_oh:.0%})           {est_ckpt_days:.1f} days')
    print(f'  Fault recovery            {est_fault_days:.1f} days')
    print(f'  Total                     {est_days:.1f} days')
    print()
    print(f'Card-hours & Cost')
    print(f'  Reported card-hours       {card_hours_reported:,.0f}')
    print(f'  Estimated card-hours      {card_hours_est:,.0f} ({(card_hours_est - card_hours_reported) / card_hours_reported * 100:+.1f} %)')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print('Commands:')
        print('  python3 sim_calibrate.py list')
        print('  python3 sim_calibrate.py verify llama3-405b [--mfu 0.50 --checkpoint-oh 0.5 --fault-factor 1.08]')
        return

    cmd = sys.argv[1].lower()
    if cmd == 'list':
        cmd_list()
    elif cmd == 'verify':
        if len(sys.argv) < 3:
            print('Usage: sim_calibrate.py verify <model_name>')
            return
        name = sys.argv[2]
        # Parse optional kwargs
        kwargs = {}
        i = 3
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                key = sys.argv[i][2:].replace('-', '_')
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    kwargs[key] = float(sys.argv[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        cmd_verify(name, **kwargs)
    else:
        print(f'Unknown command: {cmd}')


if __name__ == '__main__':
    main()
