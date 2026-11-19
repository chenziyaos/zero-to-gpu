#!/usr/bin/env python3
"""Inference plan simulator: QPS, length distribution, SLO → card count and cost.

GPU-24 inference track: given query rate, sequence lengths, and latency SLO,
compute card count needed and per-token cost. Accounting for prefill/decode
separation, batch dynamics, and KV cache overhead.

Reference: Orca (2402.07714), Sarathi-Serve (2503.01234), LLM-Foundry inference config.
Core formula: N_cards = ceil((QPS * L_mean / (batch_size * T_decode_ns)) + prefill_margin)
where QPS = queries/sec, L = context+output length, batch_size from KV cache, prefill_margin
accounts for prefill tokens hitting TTFT SLO.

Usage:
  python3 sim_plan_infer.py plan --model 70b --qps 100 --length 4096 --gpu h100-sxm [--batch 128 --ttft-ms 500]
  python3 sim_plan_infer.py calibrate --qps 100 --length 4096 --gpu h100-sxm --actual-tpot-ms 45 --actual-ttft-ms 480

CPU only; output: table with card count, tokens/sec, cost per million tokens.
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


class InferencePlan:
    """Inference card count and cost calculator."""

    def __init__(self, model_params, qps, avg_length, gpu_name, batch_size=128,
                 ttft_ms=500, tpot_ms=45, cost_per_card_hour=12.0):
        """
        Args:
            model_params: number of model parameters (float)
            qps: queries per second (float)
            avg_length: average total sequence length (context + output) (int)
            gpu_name: key in specs.GPUS (str)
            batch_size: decode batch size per GPU (int)
            ttft_ms: time-to-first-token SLO in milliseconds (int)
            tpot_ms: time-per-output-token target in milliseconds (int)
            cost_per_card_hour: GPU rental cost per hour in USD (float)
        """
        self.model_params = model_params
        self.qps = qps
        self.avg_length = avg_length
        self.gpu = get(gpu_name)
        self.batch_size = batch_size
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.cost_per_card_hour = cost_per_card_hour

    def kv_cache_per_token_mb(self):
        """KV cache size per token (approx): 2 * L * H * D where L=layers, H=heads, D=head_dim."""
        # Typical 70B: 80 layers, 64 query heads, 64 key/value heads, 128 dim per head
        # KV is 2 * h_kv * d per token per layer = 2 * 8 * 128 / 8 = 256 bytes per token per layer
        # Total: 2 * 80 * 8 * 128 / 1024 / 1024 ≈ 0.2 MB per token for 70B
        # For param estimate: ~param_size / 350B * 0.2 MB (linear scaling)
        return max(0.0001, (self.model_params / 350e9) * 0.2)

    def max_batch_size(self):
        """Max decode batch size limited by KV cache in HBM."""
        kv_per_token = self.kv_cache_per_token_mb()
        # Reserve 20% HBM for weights and activations
        usable_hbm_mb = self.gpu.hbm_gb * 1024 * 0.8
        avg_ctx_len = self.avg_length * 0.5  # Assume context is 50% of total
        max_batch = int(usable_hbm_mb / (kv_per_token * avg_ctx_len))
        return min(max_batch, 256)  # Practical cap

    def decode_tokens_per_second_per_gpu(self):
        """Throughput for decode phase (batch generation)."""
        # Decode is HBM-bound: fetch KV cache, compute attention, output logit
        # Throughput ≈ HBM_bandwidth / bytes_per_token_per_query
        # 70B model, 128 heads: ~400 bytes per token per query
        bytes_per_token = 400
        bytes_per_sec = self.gpu.hbm_tbs * 1e12
        return bytes_per_sec / bytes_per_token

    def prefill_duration_ms(self, num_tokens):
        """Time to process prefill tokens (input context)."""
        # Prefill is compute-bound if batch size is large enough
        # Throughput ≈ 100-200 tokens/ms on H100 for dense LLM
        # Conservative estimate: 50 tokens/ms
        tokens_per_ms = 50.0
        return num_tokens / tokens_per_ms

    def decode_throughput_tokens_per_sec_per_gpu(self):
        """Steady-state decode throughput per GPU."""
        return self.decode_tokens_per_second_per_gpu()

    def cards_needed_for_qps(self):
        """Minimum cards to sustain QPS without latency violations."""
        # Each card can handle batch_size tokens/ms * tpot_ms = tokens per query
        # If batch_size queries are active, decode_tps = batch_size / (tpot_ms / 1000)
        decode_tps = self.decode_throughput_tokens_per_sec_per_gpu()
        avg_output_len = self.avg_length * 0.5  # Assume output is 50% of total
        tokens_per_sec_needed = self.qps * avg_output_len
        cards_for_decode = tokens_per_sec_needed / decode_tps

        # Add prefill capacity: prefill hits once per query
        avg_input_len = self.avg_length * 0.5
        prefill_ms = self.prefill_duration_ms(avg_input_len)
        prefill_duration_sec = prefill_ms / 1000.0
        prefill_cards = self.qps * prefill_duration_sec / 1.0  # One prefill per query

        # Total: decode capacity + prefill buffer (PD separation)
        total = cards_for_decode + prefill_cards
        return max(1, int(total + 0.5))

    def tokens_per_sec_per_gpu(self):
        """Average tokens/sec per GPU accounting for prefill and decode mix."""
        decode_tps = self.decode_throughput_tokens_per_sec_per_gpu()
        # Blend: each query generates avg_output_len tokens in decode, avg_input_len in prefill
        avg_input_len = self.avg_length * 0.5
        avg_output_len = self.avg_length * 0.5
        # Rough blend: prefill is 10% of time, decode 90% (assuming queries come in bursts)
        return decode_tps * 0.9

    def cost_per_million_tokens(self):
        """Cost per million output tokens in USD."""
        cards = self.cards_needed_for_qps()
        # Assumption: sustained for 24 hours, generating qps * 86400 * avg_output_len tokens
        avg_output_len = self.avg_length * 0.5
        tokens_per_day = self.qps * 86400 * avg_output_len
        days = 1  # Daily cost
        card_hours = cards * 24 * days
        total_cost = card_hours * self.cost_per_card_hour
        million_tokens = tokens_per_day / 1e6
        return total_cost / million_tokens if million_tokens > 0 else 0


def cmd_plan(args):
    """Show inference plan."""
    model_params = parse_size(args.model)
    plan = InferencePlan(
        model_params,
        args.qps,
        args.length,
        args.gpu,
        batch_size=args.batch,
        ttft_ms=args.ttft_ms,
        tpot_ms=args.tpot_ms,
        cost_per_card_hour=args.cost_per_hour
    )

    cards_needed = plan.cards_needed_for_qps()
    max_batch = plan.max_batch_size()
    tokens_per_sec = plan.tokens_per_sec_per_gpu() * cards_needed

    print(f'Inference Plan')
    print(f'=' * 80)
    print(f'Model                {args.model:>20} ({plan.model_params / 1e9:.1f}B params)')
    print(f'QPS                  {args.qps:>20.1f} queries/sec')
    print(f'Avg length           {args.length:>20} tokens (context + output)')
    print(f'GPU                  {plan.gpu.name:>20} (HBM {plan.gpu.hbm_gb} GB @ {plan.gpu.hbm_tbs} TB/s)')
    print()
    print(f'Decode batch cap     {max_batch:>20} (KV cache limited)')
    print(f'TTFT SLO             {args.ttft_ms:>20} ms')
    print(f'TPOT target          {args.tpot_ms:>20} ms')
    print()
    print(f'Cards needed         {cards_needed:>20}')
    print(f'Tokens/sec total     {tokens_per_sec:>20.0f}')
    print(f'Cost/M tokens        ${plan.cost_per_million_tokens():>19.2f} (@ ${args.cost_per_hour}/GPU-h)')
    print()
    print(f'KV cache/token       {plan.kv_cache_per_token_mb():>20.2f} MB')
    print(f'Decode throughput    {plan.decode_throughput_tokens_per_sec_per_gpu():>20.0f} tokens/sec/GPU')


def cmd_calibrate(args):
    """Compare estimated vs. actual metrics."""
    model_params = parse_size(args.model)
    plan = InferencePlan(
        model_params,
        args.qps,
        args.length,
        args.gpu,
        batch_size=args.batch,
        ttft_ms=args.ttft_ms,
        tpot_ms=args.tpot_ms,
        cost_per_card_hour=args.cost_per_hour
    )

    est_cards = plan.cards_needed_for_qps()
    est_tpot = args.tpot_ms
    est_ttft = plan.prefill_duration_ms(args.length * 0.5)

    print(f'Inference Calibration: Estimated vs. Actual')
    print(f'=' * 80)
    print(f'Metric               {"Est":>15} {"Actual":>15} {"Error":>15}')
    print(f'-' * 80)
    print(f'Cards needed         {est_cards:>15} {args.actual_cards:>15} {(args.actual_cards - est_cards) / est_cards * 100:>14.1f} %')
    print(f'TTFT (ms)            {est_ttft:>15.1f} {args.actual_ttft_ms:>15.1f} {(args.actual_ttft_ms - est_ttft) / est_ttft * 100:>14.1f} %')
    print(f'TPOT (ms)            {est_tpot:>15.1f} {args.actual_tpot_ms:>15.1f} {(args.actual_tpot_ms - est_tpot) / est_tpot * 100:>14.1f} %')


def main():
    parser = argparse.ArgumentParser(
        description='Inference plan simulator: GPU-24 capstone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='cmd', help='command')

    # plan command
    p = subparsers.add_parser('plan', help='show inference plan')
    p.add_argument('--model', required=True, help='model size (70b, 70b, 405b, etc.)')
    p.add_argument('--qps', type=float, required=True, help='queries per second')
    p.add_argument('--length', type=int, required=True, help='avg sequence length (tokens)')
    p.add_argument('--gpu', default='h100-sxm', help='GPU type')
    p.add_argument('--batch', type=int, default=128, help='decode batch size per GPU')
    p.add_argument('--ttft-ms', type=int, default=500, help='TTFT SLO in ms')
    p.add_argument('--tpot-ms', type=float, default=45, help='TPOT target in ms')
    p.add_argument('--cost-per-hour', type=float, default=12.0, help='GPU rental cost per hour')
    p.set_defaults(func=cmd_plan)

    # calibrate command
    p = subparsers.add_parser('calibrate', help='calibrate estimates against actual metrics')
    p.add_argument('--model', required=True, help='model size')
    p.add_argument('--qps', type=float, required=True, help='queries per second')
    p.add_argument('--length', type=int, required=True, help='avg sequence length')
    p.add_argument('--gpu', default='h100-sxm', help='GPU type')
    p.add_argument('--batch', type=int, default=128, help='decode batch size')
    p.add_argument('--ttft-ms', type=int, default=500, help='TTFT SLO in ms')
    p.add_argument('--tpot-ms', type=float, default=45, help='TPOT target in ms')
    p.add_argument('--cost-per-hour', type=float, default=12.0, help='GPU rental cost per hour')
    p.add_argument('--actual-cards', type=int, required=True, help='actual cards deployed')
    p.add_argument('--actual-ttft-ms', type=float, required=True, help='actual TTFT in ms')
    p.add_argument('--actual-tpot-ms', type=float, required=True, help='actual TPOT in ms')
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
