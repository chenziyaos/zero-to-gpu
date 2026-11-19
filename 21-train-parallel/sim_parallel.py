#!/usr/bin/env python3
"""GPU-21 并行方案搜索：给定集群和模型，枚举 TP/PP/DP/ZeRO 组合，计算每步时间与 MFU。

正确的物理模型（自 2026-11-05）：

  令 T = 全局 batch 总 tokens 数，ngpus = tp·pp·dp，H = 隐层维，L = 层数，N = 总参数。

  1. 每步 FLOP = 6·N·T（前向 + 反向，不含优化器）

  2. 计算时间（上界）：T_compute = 6·N·T / (ngpus·peak_bf16) / kernel_eff
     kernel_eff = 核心利用率，默认 0.8；可用 --kernel-eff 覆盖

  3. PP 流水气泡（1F1B）：m = 微批数 (default 8·pp)
     bubble_time = (pp−1)/m × T_compute

  4. TP all-reduce 通信（Megatron 4-agg per layer）：
     每层每 GPU 每微批激活字节 = T/dp × H × 2（fp16/bf16）
     4 all-reduce/layer（2 fwd + 2 bwd），L/pp 层这个 stage 的
     TP 通信量 = 4 × (L/pp) × (T/dp × H × 2)
     TP 时间/GPU = vol / (nvlink_gbs/2) × ring_factor，ring_factor = 2(tp−1)/tp
     假设未重叠（exposed）

  5. DP all-reduce 通信（ring，gradient）：
     梯度字节数 = 2·N/(tp·pp)（每 GPU 那片参数的梯度）
     DP 时间/GPU = 2(dp−1)/dp × bytes / B_dp，B_dp = 50 GB/s
     重叠比例 = overlap（default 0.5），暴露时间 = DP 时间 × (1 - overlap)

  6. 总时间 = T_compute + bubble_time + TP 时间 + 暴露 DP 时间

  7. MFU = 6·N·T / (ngpus·peak_bf16·总时间)

使用：
  python3 sim_parallel.py search --gbs-tokens 4M --seq 8192 --layers 32 --ngpus 64 --tp-max 8 --pp-max 4
  python3 sim_parallel.py scheme --params 8 --gbs-tokens 4M --ngpus 64 --tp 8 --pp 2 --gpu h100-sxm
"""

import argparse
import sys
import os
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get


def calc_mfu(params_b, gbs_tokens, seq_len, num_layers, tp_deg, pp_deg, dp_deg, gpu_name,
             kernel_eff=0.8, microbatches=None, overlap=0.5, hidden_dim=None):
    """估计一个并行方案的 MFU 与模型显存。

    Args:
        params_b: 模型参数（十亿）
        gbs_tokens: 全局 batch 总 tokens 数（十亿，例 4 代表 4B）
        seq_len: 序列长度（仅用于计算隐层维，若已提供 hidden_dim 则忽略）
        num_layers: 模型层数
        tp_deg, pp_deg, dp_deg: 三维并行度
        gpu_name: GPU 规格名称
        kernel_eff: 核心利用率（默认 0.8）
        microbatches: 微批数（默认 8·pp）
        overlap: DP all-reduce 重叠比例（默认 0.5）
        hidden_dim: 隐层维度（若 None，从 params 反推）

    Returns:
        包含详细时间分解和显存信息的字典
    """
    gpu = get(gpu_name)

    # 参数转换
    params = params_b * 1e9  # 参数个数
    # gbs_tokens 可能是原始数字（tokens）或字符串（"4M", "1B"）
    # 如果是原始数字，需要判断单位（1e6 表示百万，1e9 表示十亿）
    if isinstance(gbs_tokens, str):
        s = gbs_tokens.upper().strip()
        if s.endswith('M'):
            gbs_toks = float(s[:-1]) * 1e6
        elif s.endswith('B'):
            gbs_toks = float(s[:-1]) * 1e9
        else:
            gbs_toks = float(s)
    else:
        # 假设是十亿为单位的数字或原始 tokens 数
        gbs_toks = float(gbs_tokens)
        if gbs_toks < 1e6:  # 若小于百万，假设单位是十亿
            gbs_toks *= 1e9

    # 推导隐层维（若未明确）
    if hidden_dim is None:
        # 简化假设：N ≈ 12·L·H^2（Transformer 标准配置）
        # 所以 H ≈ sqrt(N / (12·L))
        hidden_dim = int((params / (12 * num_layers)) ** 0.5)

    ngpus = tp_deg * pp_deg * dp_deg

    # 1. 计算每步总 FLOP（前向 + 反向，全球所有 GPU）
    flop_per_step = 6 * params * gbs_toks

    # 2. 计算时间（微秒）
    # 每 GPU 分摊：flop_per_step / ngpus
    # 每 GPU 每秒 TFLOP：gpu.bf16_tflops
    # 计算时间 = (flop_per_step / ngpus) / (gpu.bf16_tflops * 1e12) / kernel_eff
    compute_time_us = (flop_per_step / ngpus) / (gpu.bf16_tflops * 1e12) / kernel_eff * 1e6

    # 3. PP 气泡（1F1B）
    if microbatches is None:
        microbatches = max(8 * pp_deg, 1)
    bubble_ratio = (pp_deg - 1) / microbatches if pp_deg > 1 else 0
    bubble_time_us = bubble_ratio * compute_time_us

    # 4. TP all-reduce 通信
    # 每层激活字节 = (gbs_toks/dp_deg) × hidden_dim × 2
    # 4 all-reduce/layer，L/pp 层，ring_factor = 2(tp-1)/tp
    if tp_deg > 1:
        activation_bytes_per_layer = (gbs_toks / dp_deg) * hidden_dim * 2
        tp_allreduce_vol = 4 * (num_layers / pp_deg) * activation_bytes_per_layer
        ring_factor_tp = 2 * (tp_deg - 1) / tp_deg if tp_deg > 1 else 0
        tp_allreduce_vol *= ring_factor_tp
        # 单向 NVLink 带宽
        nvlink_bw_bs = gpu.nvlink_gbs / 2
        tp_time_us = tp_allreduce_vol / (nvlink_bw_bs * 1e9) * 1e6
    else:
        tp_time_us = 0

    # 5. DP all-reduce 通信
    if dp_deg > 1:
        grad_bytes = 2 * params / (tp_deg * pp_deg)
        ring_factor_dp = 2 * (dp_deg - 1) / dp_deg
        dp_allreduce_vol = grad_bytes * ring_factor_dp
        b_dp = 50e9  # 50 GB/s RDMA
        dp_time_us = dp_allreduce_vol / b_dp * 1e6
        dp_exposed_us = dp_time_us * (1 - overlap)
    else:
        dp_time_us = 0
        dp_exposed_us = 0

    # 6. 总时间
    total_time_us = compute_time_us + bubble_time_us + tp_time_us + dp_exposed_us

    # 7. MFU = 6·N·T / (ngpus·peak_bf16·总时间)
    # total_time_us is in microseconds, convert to seconds: total_time_us * 1e-6
    mfu = (flop_per_step / (ngpus * gpu.bf16_tflops * 1e12 * total_time_us * 1e-6)) * 100

    # 8. Model-state memory per GPU, mixed-precision Adam (ZeRO paper accounting):
    #    bf16 param 2 B + bf16 grad 2 B + fp32 master/m/v 12 B = 16 B per parameter.
    #    Each GPU holds N/(tp*pp) parameters before ZeRO sharding over the dp group.
    #    ZeRO-1: 2 + 2 + 12/dp; ZeRO-2: 2 + 14/dp; ZeRO-3: 16/dp (activations not included).
    params_shard = params / (tp_deg * pp_deg)
    model_state_gb = params_shard * 16 / 1e9
    zero1_gb = params_shard * (4 + 12.0 / dp_deg) / 1e9
    zero2_gb = params_shard * (2 + 14.0 / dp_deg) / 1e9
    zero3_gb = params_shard * 16.0 / dp_deg / 1e9

    budget_gb = gpu.hbm_gb * 0.8  # leave 20% for activations and fragmentation
    feasible_dp = model_state_gb <= budget_gb
    feasible_zero1 = zero1_gb <= budget_gb
    feasible_zero2 = zero2_gb <= budget_gb
    feasible_zero3 = zero3_gb <= budget_gb

    if feasible_dp:
        needs_strategy = "纯DP"
    elif feasible_zero1:
        needs_strategy = "ZeRO-1"
    elif feasible_zero2:
        needs_strategy = "ZeRO-2"
    elif feasible_zero3:
        needs_strategy = "ZeRO-3"
    else:
        needs_strategy = "放不下"

    return {
        'flop': flop_per_step,
        'compute_us': compute_time_us,
        'tp_us': tp_time_us,
        'bubble_ratio': bubble_ratio,
        'bubble_time_us': bubble_time_us,
        'dp_us': dp_time_us,
        'dp_exposed_us': dp_exposed_us,
        'total_us': total_time_us,
        'mfu': mfu,
        'hidden_dim': hidden_dim,
        'model_state_gb': model_state_gb,
        'zero1_gb': zero1_gb,
        'zero2_gb': zero2_gb,
        'zero3_gb': zero3_gb,
        'feasible_dp': feasible_dp,
        'feasible_zero1': feasible_zero1,
        'feasible_zero2': feasible_zero2,
        'feasible_zero3': feasible_zero3,
        'needs_strategy': needs_strategy,
        'gpu_hbm_gb': gpu.hbm_gb,
    }


def scheme_detail(params_b, gbs_tokens, seq_len, num_layers, num_gpus, tp_deg, pp_deg, gpu_name,
                  kernel_eff=0.8, microbatches=None, overlap=0.5, hidden_dim=None):
    """打印一个方案的详细分析。"""
    if num_gpus % (tp_deg * pp_deg) != 0:
        print(f"错误：卡数 {num_gpus} 不能被 TP({tp_deg}) × PP({pp_deg}) = {tp_deg*pp_deg} 整除")
        return

    dp_deg = num_gpus // (tp_deg * pp_deg)
    gpu = get(gpu_name)

    result = calc_mfu(params_b, gbs_tokens, seq_len, num_layers, tp_deg, pp_deg, dp_deg, gpu_name,
                      kernel_eff=kernel_eff, microbatches=microbatches, overlap=overlap, hidden_dim=hidden_dim)

    print(f"{'GPU':12} {gpu.name}")
    print(f"{'模型参数':12} {params_b:.1f}B")
    print(f"{'全局 tokens':12} {gbs_tokens:.0e}" if isinstance(gbs_tokens, float) and gbs_tokens >= 1e6 else f"{'全局 tokens':12} {gbs_tokens}B")
    print(f"{'序列长度':12} {seq_len}")
    print(f"{'层数':12} {num_layers}")
    print(f"{'隐层维':12} {result['hidden_dim']}")
    print(f"{'总卡数':12} {num_gpus}")
    print(f"{'并行方案':12} TP={tp_deg}, PP={pp_deg}, DP={dp_deg}")
    print(f"{'核心利用率':12} {kernel_eff:.1%}")
    print(f"{'微批数':12} {microbatches if microbatches else 8*pp_deg}")
    print(f"{'DP 重叠':12} {overlap:.1%}")
    print()
    print(f"{'每步 FLOP':12} {result["flop"]:.2e}")
    print(f"{'计算时间':12} {result["compute_us"]:.1f} us ({result["compute_us"]/1000:.2f} ms)")
    print(f"{'TP 通信':12} {result["tp_us"]:.1f} us ({result["tp_us"]/1000:.2f} ms)")
    print(f"{'PP 气泡比':12} {result["bubble_ratio"]:.1%}")
    print(f"{'PP 气泡时间':12} {result["bubble_time_us"]:.1f} us ({result["bubble_time_us"]/1000:.2f} ms)")
    print(f"{'DP 通信':12} {result["dp_us"]:.1f} us ({result["dp_us"]/1000:.2f} ms, {overlap:.1%} 重叠)")
    print(f"{'DP 暴露':12} {result["dp_exposed_us"]:.1f} us ({result["dp_exposed_us"]/1000:.2f} ms)")
    print(f"{'总时间 (est)':12} {result["total_us"]:.1f} us ({result["total_us"]/1000:.2f} ms)")
    print(f"{'MFU':12} {result["mfu"]:.1f}%")
    print()
    print("显存分析（模型状态，混合精度 Adam 优化器）：")
    print(f"  纯 DP（无分片）：{result['model_state_gb']:.1f} GB ({result['model_state_gb']/result['gpu_hbm_gb']*100:.1f}% GPU)")
    print(f"  ZeRO-1（优化器分片）：{result['zero1_gb']:.1f} GB ({result['zero1_gb']/result['gpu_hbm_gb']*100:.1f}% GPU)")
    print(f"  ZeRO-2（梯度+优化器分片）：{result['zero2_gb']:.1f} GB ({result['zero2_gb']/result['gpu_hbm_gb']*100:.1f}% GPU)")
    print(f"  ZeRO-3（全分片）：{result['zero3_gb']:.1f} GB ({result['zero3_gb']/result['gpu_hbm_gb']*100:.1f}% GPU)")
    print(f"  需要的分片：{result['needs_strategy']}（模型状态 ≤ 80% 显存，未计激活）")
    print()


def search_schemes(params_b, gbs_tokens, seq_len, num_layers, num_gpus, tp_max, pp_max, gpu_name,
                   kernel_eff=0.8, microbatches=None, overlap=0.5, hidden_dim=None):
    """搜索可行的并行方案，输出前 N 个 MFU 最高的。"""
    candidates = []

    for tp_deg in range(1, min(tp_max + 1, num_gpus + 1)):
        for pp_deg in range(1, min(pp_max + 1, num_gpus // tp_deg + 1)):
            if num_gpus % (tp_deg * pp_deg) != 0:
                continue
            dp_deg = num_gpus // (tp_deg * pp_deg)

            result = calc_mfu(params_b, gbs_tokens, seq_len, num_layers, tp_deg, pp_deg, dp_deg, gpu_name,
                            kernel_eff=kernel_eff, microbatches=microbatches, overlap=overlap, hidden_dim=hidden_dim)
            candidates.append({
                'tp': tp_deg,
                'pp': pp_deg,
                'dp': dp_deg,
                **result
            })

    # 按 MFU 倒序排列
    candidates.sort(key=lambda x: x['mfu'], reverse=True)

    print(f"{'TP':4} {'PP':4} {'DP':4} {'MFU':>7} {'模型显存':>10} {'策略':>8} {'计算':>8} {'通信总':>9} {'气泡%':>7}")
    print("-" * 78)
    for i, c in enumerate(candidates[:10]):
        print(
            f"{c['tp']:4d} {c['pp']:4d} {c['dp']:4d} "
            f"{c['mfu']:7.1f}% {c['model_state_gb']:10.1f}GB "
            f"{c['needs_strategy']:>8} {c['compute_us']/1000:8.2f}ms "
            f"{(c['tp_us'] + c['dp_us'])/1000:9.2f}ms {c['bubble_ratio']:7.1%}"
        )
    print()
    print(f"共 {len(candidates)} 个可行方案。")
    print(f"注：模型显存 = 纯 DP 不分片的显存，策略标记所需分片级别（纯DP/ZeRO-1/ZeRO-2/ZeRO-3）以保持 ≤80% GPU 容量。")


def main():
    parser = argparse.ArgumentParser(description='GPU-21 并行方案搜索')
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # scheme 子命令
    p_scheme = subparsers.add_parser('scheme', help='分析单个方案')
    p_scheme.add_argument('--params', type=float, default=8, help='模型参数（十亿）')
    p_scheme.add_argument('--gbs-tokens', type=str, default='4M', help='全局 batch tokens（例 4M, 1B）')
    p_scheme.add_argument('--seq', type=int, default=8192, help='序列长度')
    p_scheme.add_argument('--layers', type=int, default=32, help='层数')
    p_scheme.add_argument('--hidden', type=int, default=None, help='隐层维（默认自动推导）')
    p_scheme.add_argument('--ngpus', type=int, default=64, help='总 GPU 数')
    p_scheme.add_argument('--tp', type=int, default=8, help='张量并行度')
    p_scheme.add_argument('--pp', type=int, default=2, help='流水线并行度')
    p_scheme.add_argument('--gpu', default='h100-sxm', help='GPU 规格')
    p_scheme.add_argument('--kernel-eff', type=float, default=0.8, help='核心利用率')
    p_scheme.add_argument('--microbatches', type=int, default=None, help='微批数')
    p_scheme.add_argument('--overlap', type=float, default=0.5, help='DP all-reduce 重叠比例')

    # search 子命令
    p_search = subparsers.add_parser('search', help='搜索并行方案')
    p_search.add_argument('--params', type=float, default=8, help='模型参数（十亿）')
    p_search.add_argument('--gbs-tokens', type=str, default='4M', help='全局 batch tokens')
    p_search.add_argument('--seq', type=int, default=8192, help='序列长度')
    p_search.add_argument('--layers', type=int, default=32, help='层数')
    p_search.add_argument('--hidden', type=int, default=None, help='隐层维')
    p_search.add_argument('--ngpus', type=int, default=64, help='总 GPU 数')
    p_search.add_argument('--tp-max', type=int, default=8, help='张量并行最大度')
    p_search.add_argument('--pp-max', type=int, default=4, help='流水线并行最大度')
    p_search.add_argument('--gpu', default='h100-sxm', help='GPU 规格')
    p_search.add_argument('--kernel-eff', type=float, default=0.8, help='核心利用率')
    p_search.add_argument('--microbatches', type=int, default=None, help='微批数')
    p_search.add_argument('--overlap', type=float, default=0.5, help='DP 重叠比例')

    args = parser.parse_args()

    # 解析 gbs_tokens（支持 4M, 1B 等格式）
    def parse_gbs(s):
        if isinstance(s, (int, float)):
            return float(s)
        s = s.upper().strip()
        if s.endswith('M'):
            return float(s[:-1]) * 1e6
        elif s.endswith('B'):
            return float(s[:-1]) * 1e9
        else:
            return float(s)

    if args.cmd == 'scheme':
        gbs = parse_gbs(args.gbs_tokens)
        scheme_detail(args.params, gbs, args.seq, args.layers, args.ngpus, args.tp, args.pp, args.gpu,
                     kernel_eff=args.kernel_eff, microbatches=args.microbatches, overlap=args.overlap,
                     hidden_dim=args.hidden)
    elif args.cmd == 'search':
        gbs = parse_gbs(args.gbs_tokens)
        search_schemes(args.params, gbs, args.seq, args.layers, args.ngpus, args.tp_max, args.pp_max, args.gpu,
                      kernel_eff=args.kernel_eff, microbatches=args.microbatches, overlap=args.overlap,
                      hidden_dim=args.hidden)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
