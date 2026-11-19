"""Prefill/Decode 分离：显存与带宽分配决策（CPU 模拟）。对应 GPU-23 第 5 至 7 章。

核心问题：
  给定请求分布（arrival rate、prefill 长度、decode 步数）和单机资源，
  如何在 Prefill 和 Decode 卡之间分配资源，以最大化吞吐同时满足 SLO？

账本给出：
  - Prefill 和 Decode 阶段各自的计算模式与内存约束
  - 两阶段的 KV 传输带宽需求
  - 给定卡数分配，估算吞吐与 TTFT/TPOT
  - 不同 batch 和分离比例下的 goodput

子命令：
  prefill_roofline  Prefill 阶段的 Roofline（高算术强度）
  pd_split          给定总卡数，扫描最优的 PD 分离比例
  kv_transfer       估算 KV cache 从 prefill 卡流向 decode 卡的带宽
  goodput           给定 SLO，估算有效吞吐（goodput）与浪费

用法：
  python3 sim_pd.py prefill_roofline --model llama-70b-bf16 --gpu h100-sxm
  python3 sim_pd.py pd_split --model llama-70b-bf16 --gpu h100-sxm --total-gpus 8
  python3 sim_pd.py kv_transfer --model llama-70b-bf16 --batch 256 --seq-len 4096
  python3 sim_pd.py goodput --model llama-70b-bf16 --gpu h100-sxm --slo-ms 200
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402
from sim_kv import (  # noqa: E402
    compute_weight_bytes, kv_bytes_per_token, kv_bytes_per_request,
    MODEL_CONFIGS, max_concurrent_requests
)
from sim_decode import decode_time_per_token_us  # noqa: E402


def prefill_time_per_batch_ms(config, batch_size, seq_len, hbm_tbs, tp=1):
    """Prefill 阶段一个 batch 的总时间（毫秒）。

    Prefill 是计算密集的：Q, K, V 线性投影、注意力矩阵乘法都涉及 seq_len × seq_len。
    粗估：权重加载一遍 + batch 中所有请求的 KV 读写。

    TP：单 GPU 权重和 KV 均除以 tp。
    """
    weight_total = compute_weight_bytes(config, config)
    weight_per_gpu = weight_total / tp
    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']

    # Prefill 中，权重加载一遍（all tokens use it），每 GPU 只加载 1/tp
    # 所有 token 的 KV 写入：batch_size * seq_len * num_layers * KV_per_token，每 GPU 也只写 1/tp
    kv_bytes_total = batch_size * seq_len * 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']
    kv_bytes_per_gpu = kv_bytes_total / tp

    # 单 GPU 的搬运
    total_bytes = weight_per_gpu + kv_bytes_per_gpu

    # 以带宽估时间
    latency_us = total_bytes / (hbm_tbs * 1e12) * 1e6
    return latency_us / 1000  # 转成毫秒


def prefill_batch_throughput_tps(config, batch_size, prefill_seq_len, hbm_tbs, tp=1):
    """Prefill 阶段的吞吐：每秒处理多少 token（输入 token）。

    一个 batch 的 prefill 时间（毫秒）
    吞吐 = batch_size * prefill_seq_len / (time_ms / 1000)
    """
    time_ms = prefill_time_per_batch_ms(config, batch_size, prefill_seq_len, hbm_tbs, tp=tp)
    total_tokens = batch_size * prefill_seq_len
    tps = total_tokens / (time_ms / 1000)
    return tps


def kv_transfer_bandwidth_gbs(batch_size, seq_len, config, num_prefill_gpus, nvlink_gbs_per_gpu, tp=1):
    """KV cache 从 prefill 卡集群流向 decode 卡集群的带宽需求（GB/s）。

    假设 prefill 完成后，KV 立即从 prefill GPU 传到 decode GPU。
    decode 卡数 = total_gpus - num_prefill_gpus。
    KV 传输额度：每生成一个 decode token，需要读一遍 KV（已考虑 TP 分片）。

    TP：KV 也跨 tp 卡分布，传输字节数除以 tp。
    """
    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']
    kv_per_token = 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']
    kv_per_request_total = kv_per_token * seq_len
    kv_per_request_per_gpu = kv_per_request_total / tp

    # 假设每个 prefill GPU 有 batch_size // num_prefill_gpus 个请求
    # 每个请求的 KV 传输体积 = kv_per_request_per_gpu（应用 TP）
    # 假设传输时间窗口为 decode 的一个 token 步（几十 ms）
    # 简化：给出单位时间的数据速率

    # 按 NVLink 估计（可用带宽）
    available_bw = nvlink_gbs_per_gpu
    return available_bw


def pd_split_analysis(config, gpu, total_gpus, batch_size, prefill_seq_len, decode_output_len, tp=1):
    """分析不同 PD 分离比例下的性能。

    简化模型：
    - Prefill GPU：每个 GPU 处理 batch/num_pfill 个请求的 prefill
    - Decode GPU：每个 GPU 处理 batch/num_decode 个请求的 decode
    - TP：权重和 KV 均分散到 tp 卡上
    """
    results = []

    # 单个 GPU 处理全 batch 的时间（应用 TP）
    pfill_time_ms = prefill_time_per_batch_ms(config, batch_size, prefill_seq_len, gpu.hbm_tbs, tp=tp)
    decode_latency_us = decode_time_per_token_us(config, batch_size, prefill_seq_len, gpu.hbm_tbs, tp=tp)
    decode_tpot_ms = decode_latency_us / 1000

    # 尝试不同的 prefill GPU 数量
    for num_pfill in range(1, total_gpus):
        num_decode = total_gpus - num_pfill

        # 单个 Prefill GPU 的吞吐率（input token/s）
        # batch prefill 时间不变，但分散到多个 GPU
        pfill_tps_per_gpu = batch_size * prefill_seq_len / (pfill_time_ms / 1000)

        # 单个 Decode GPU 的吞吐率（output token/s）
        # 每个 batch 的 decode 时间也不变，所以 decode tps 也不变
        decode_tps_per_gpu = batch_size / (decode_tpot_ms / 1000)

        # 总吞吐考虑并行度
        # Prefill 总吞吐 = pfill_tps_per_gpu × num_pfill（处理更多请求）
        # Decode 总吞吐 = decode_tps_per_gpu × num_decode（产生更多 token）
        pfill_tps_total = pfill_tps_per_gpu * num_pfill
        decode_tps_total = decode_tps_per_gpu * num_decode

        # TTFT（第一个 output token）：prefill 时间
        ttft_ms = pfill_time_ms

        # 瓶颈判定
        bottleneck = 'Pfill' if pfill_tps_total < decode_tps_total else 'Decode'

        results.append({
            'num_pfill': num_pfill,
            'num_decode': num_decode,
            'pfill_tps': pfill_tps_total,
            'decode_tps': decode_tps_total,
            'ttft_ms': ttft_ms,
            'tpot_ms': decode_tpot_ms,
            'bottleneck': bottleneck,
        })

    return results


def goodput_with_slo(config, gpu, batch_size, seq_len, slo_ms, decode_output_len, tp=1):
    """估算满足 SLO 的有效吞吐（goodput）。

    端到端延迟 = TTFT + TPOT × decode_output_tokens
    如果超过 SLO，作为违反 SLO 的请求处理。

    TP：权重和 KV 均分散，单 GPU 延迟减少。
    """
    # Decode 每个 token 的时间
    decode_latency_us = decode_time_per_token_us(config, batch_size, seq_len, gpu.hbm_tbs, tp=tp)
    tpot_ms = decode_latency_us / 1000

    # TTFT = Prefill 的批处理时间（生成 KV 缓存）
    ttft_ms = prefill_time_per_batch_ms(config, batch_size, seq_len, gpu.hbm_tbs, tp=tp)

    # 总延迟：TTFT（处理输入）+ TPOT × 输出 token 数
    total_latency_ms = ttft_ms + tpot_ms * decode_output_len

    # 如果超过 SLO，goodput 打折
    if total_latency_ms > slo_ms:
        reduction_ratio = slo_ms / total_latency_ms
    else:
        reduction_ratio = 1.0

    # 理论吞吐：batch / latency
    theoretical_tps = batch_size / (total_latency_ms / 1000)
    goodput = theoretical_tps * reduction_ratio

    return {
        'ttft_ms': ttft_ms,
        'tpot_ms': tpot_ms,
        'total_latency_ms': total_latency_ms,
        'slo_ms': slo_ms,
        'meets_slo': total_latency_ms <= slo_ms,
        'reduction_ratio': reduction_ratio,
        'theoretical_tps': theoretical_tps,
        'goodput_tps': goodput,
    }


def cmd_prefill_roofline(args):
    """Prefill 阶段的 Roofline 分析。"""
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    batch = args.batch
    seq_len = args.seq_len
    tp = args.tp

    weight_total = compute_weight_bytes(config, config)
    weight_per_gpu = weight_total / tp
    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']

    # Prefill 涉及高度的矩阵乘法，算术强度较高
    # 粗估：每个 token 至少做 2 * params 的乘加
    flops_estimate = batch * seq_len * config['params_b'] * 1e9 * 2
    kv_bytes_total = batch * seq_len * 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']
    kv_bytes_per_gpu = kv_bytes_total / tp
    total_bytes = weight_per_gpu + kv_bytes_per_gpu

    # 每张卡只算 1/tp 的 FLOP，也只搬自己那份字节
    intensity = flops_estimate / tp / total_bytes if total_bytes > 0 else 0
    ridge = gpu.ridge('bf16')

    time_ms = prefill_time_per_batch_ms(config, batch, seq_len, gpu.hbm_tbs, tp=tp)

    print(f"\n{'=' * 80}")
    print(f"Prefill Roofline（{args.model}, batch={batch}, seq_len={seq_len}, TP={tp}）")
    print(f"{'=' * 80}")
    print(f"{'权重 (per GPU)':30} {weight_per_gpu / 1e9:>10.1f} GB")
    print(f"{'KV 写入 (per GPU)':30} {kv_bytes_per_gpu / 1e9:>10.1f} GB")
    print(f"{'总搬运 (per GPU)':30} {total_bytes / 1e9:>10.1f} GB")
    print(f"{'估计 FLOP':30} {flops_estimate / 1e12:>10.1f} TFLOP")
    print(f"{'算术强度':30} {intensity:>10.2f} FLOP/B")
    print(f"{'Ridge Point':30} {ridge:>10.1f} FLOP/B")
    print(f"{'处理时间':30} {time_ms:>10.1f} ms")

    if intensity >= ridge:
        print(f"\n>>> Prefill 处于算力约束（intensity {intensity:.2f} >= ridge {ridge:.1f}）")
    else:
        print(f"\n>>> Prefill 处于带宽约束（intensity {intensity:.2f} < ridge {ridge:.1f}）")


def cmd_pd_split(args):
    """扫描最优的 PD 分离比例。"""
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    total_gpus = args.total_gpus
    batch = args.batch
    prefill_seq = args.prefill_seq
    decode_output = args.decode_output
    tp = args.tp

    results = pd_split_analysis(config, gpu, total_gpus, batch, prefill_seq, decode_output, tp=tp)

    print(f"\n{'=' * 120}")
    print(f"PD 分离决策分析（{args.model}，总卡数={total_gpus}，batch={batch}，TP={tp}）")
    print(f"{'=' * 120}")
    print(f"{'PF卡':>6} {'Dec卡':>7} {'Pfill吞 (tok/s)':>18} {'Decode吞 (tok/s)':>20} {'瓶颈':>9} {'TTFT (ms)':>11} {'TPOT (ms)':>11}")
    print(f"{'-' * 120}")

    for r in results:
        print(f"{r['num_pfill']:>6} {r['num_decode']:>7} {r['pfill_tps']:>18.1f} "
              f"{r['decode_tps']:>20.1f} {r['bottleneck']:>9} {r['ttft_ms']:>11.1f} {r['tpot_ms']:>11.3f}")


def cmd_kv_transfer(args):
    """估算 KV 从 prefill 卡流向 decode 卡的带宽需求。"""
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    batch = args.batch
    seq_len = args.seq_len
    tp = args.tp

    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']
    kv_per_token = 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']
    kv_per_request_total = kv_per_token * seq_len
    kv_per_request_per_gpu = kv_per_request_total / tp
    kv_total_per_gpu = batch * kv_per_request_per_gpu

    # 假设 decode 阶段产生 output_tokens 个 token，每次都要读一遍 KV
    # 传输周期 = TPOT（几十 ms）
    tpot_ms = args.decode_output_tokens / 1000  # 粗估

    # 带宽需求 = 总 KV 体积 / 传输周期
    if tpot_ms > 0:
        bw_required_gbs = kv_total_per_gpu / 1e9 / (tpot_ms / 1000)
    else:
        bw_required_gbs = 0

    print(f"\n{'=' * 80}")
    print(f"KV Cache 跨卡传输（{args.model}, batch={batch}, seq_len={seq_len}, TP={tp}）")
    print(f"{'=' * 80}")
    print(f"{'KV/token':30} {kv_per_token:>10} bytes")
    print(f"{'KV/请求':30} {kv_per_request_total / 1e6:>10.1f} MB")
    print(f"{'KV/请求 (per GPU)':30} {kv_per_request_per_gpu / 1e6:>10.1f} MB")
    print(f"{'总 KV (batch, per GPU)':30} {kv_total_per_gpu / 1e9:>10.1f} GB")
    print(f"{'Decode 输出 token':30} {args.decode_output_tokens:>10}")
    print(f"{'估计传输周期 (TPOT)':30} {tpot_ms:>10.2f} ms")
    print(f"{'所需带宽 (per GPU)':30} {bw_required_gbs:>10.1f} GB/s")
    print(f"\n说明：")
    print(f"  NVLink 单向 450 GB/s，PCIe Gen5 x16 单向 ~63 GB/s")
    print(f"  如果需求 > 450 GB/s，需多条 NVLink 或分片")


def cmd_goodput(args):
    """估算满足 SLO 的 goodput。"""
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    tp = args.tp

    print(f"\n{'=' * 100}")
    print(f"SLO 下的 Goodput 分析（{args.model} on {args.gpu}, TP={tp}）")
    print(f"{'=' * 100}")
    print(f"{'SLO (ms)':>12} {'TTFT (ms)':>12} {'TPOT (ms)':>12} {'总延迟 (ms)':>15} {'违反?':>8} {'Goodput (tok/s)':>15}")
    print(f"{'-' * 80}")

    for slo_ms in [50, 100, 200, 500, 1000]:
        result = goodput_with_slo(config, gpu, args.batch, args.seq_len, slo_ms, args.decode_output_tokens, tp=tp)
        violate = "Y" if not result['meets_slo'] else "N"
        print(f"{result['slo_ms']:>12.0f} {result['ttft_ms']:>12.1f} {result['tpot_ms']:>12.3f} "
              f"{result['total_latency_ms']:>15.1f} {violate:>8} {result['goodput_tps']:>15.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Prefill/Decode 分离：资源配置与吞吐决策',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n'
               '  python3 sim_pd.py prefill_roofline --model llama-70b-bf16 --gpu h100-sxm\n'
               '  python3 sim_pd.py pd_split --model llama-70b-bf16 --total-gpus 8\n'
               '  python3 sim_pd.py kv_transfer --model llama-70b-bf16 --batch 256 --seq-len 4096\n'
               '  python3 sim_pd.py goodput --model llama-70b-bf16 --gpu h100-sxm --slo-ms 200\n'
    )

    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    p_pfill = subparsers.add_parser('prefill_roofline', help='Prefill Roofline 分析')
    p_pfill.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_pfill.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_pfill.add_argument('--batch', type=int, default=256, help='batch size')
    p_pfill.add_argument('--seq-len', type=int, default=4096, help='序列长度')
    p_pfill.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_pfill.set_defaults(func=cmd_prefill_roofline)

    p_split = subparsers.add_parser('pd_split', help='扫 PD 分离比例')
    p_split.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_split.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_split.add_argument('--total-gpus', type=int, default=8, help='总 GPU 数')
    p_split.add_argument('--batch', type=int, default=256, help='batch size')
    p_split.add_argument('--prefill-seq', type=int, default=4096, help='prefill 序列长度')
    p_split.add_argument('--decode-output', type=int, default=100, help='decode 输出 token 数')
    p_split.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_split.set_defaults(func=cmd_pd_split)

    p_kv = subparsers.add_parser('kv_transfer', help='KV 跨卡传输带宽')
    p_kv.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_kv.add_argument('--batch', type=int, default=256, help='batch size')
    p_kv.add_argument('--seq-len', type=int, default=4096, help='序列长度')
    p_kv.add_argument('--decode-output-tokens', type=int, default=100, help='decode 输出 token')
    p_kv.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_kv.set_defaults(func=cmd_kv_transfer)

    p_goodput = subparsers.add_parser('goodput', help='SLO 下的 goodput 分析')
    p_goodput.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_goodput.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_goodput.add_argument('--batch', type=int, default=256, help='batch size')
    p_goodput.add_argument('--seq-len', type=int, default=4096, help='序列长度')
    p_goodput.add_argument('--decode-output-tokens', type=int, default=100, help='decode 输出 token')
    p_goodput.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_goodput.set_defaults(func=cmd_goodput)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
