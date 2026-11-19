"""Decode 阶段延迟与吞吐分析（CPU 模拟）。对应 GPU-23 第 4 章。

核心公式：
  Decode 时间/token = (权重字节 + KV 字节) / HBM 带宽
  Token throughput (tok/s) = batch_size / (time_per_token)
  TPOT (Time Per Output Token) = time_per_token / batch_size

账本给出：
  - 按 batch size 和上下文长度扫描 decode 延迟
  - 计算端到端 tokens/s 与 TPOT
  - 对比不同 batch 下延迟与吞吐的权衡
  - 权重对延迟的主导作用（带宽约束）

子命令：
  latency   扫 batch 和 seq_len，给出每 token 延迟和吞吐
  compare   对比不同模型的 TPOT
  roofline  画出 decode 的 roofline 分析
  scaling   扫 batch，观察吞吐饱和点

用法：
  python3 sim_decode.py latency --model llama-70b-bf16 --gpu h100-sxm
  python3 sim_decode.py compare --gpu h100-sxm --seq-len 16384
  python3 sim_decode.py roofline --model llama-70b-bf16 --batch 256
  python3 sim_decode.py scaling --model llama-70b-bf16 --gpu h100-sxm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402
from sim_kv import compute_weight_bytes, kv_bytes_per_token, MODEL_CONFIGS  # noqa: E402


def tp_allreduce_cost_us(batch_size, hidden_size, tp, num_layers=80):
    """Tensor Parallel all-reduce 通信成本（微秒）。

    公式：
    - 消息大小 = batch_size * hidden_size * 2 bytes（假设数据类型为 BF16）
    - 每层需要 2 次 all-reduce（一次在注意力后，一次在 FFN 后）
    - Ring all-reduce 延迟 = 2 * (tp - 1) / tp * msg_bytes / (ring_bandwidth GB/s) + latency
    - Ring 带宽：~450 GB/s（NVLink）
    - All-reduce 延迟：~8 µs 每次（假设）

    TP度数为 1 时，无通信成本。
    """
    if tp <= 1:
        return 0.0

    msg_bytes = batch_size * hidden_size * 2  # BF16
    ring_bw_gbs = 450  # NVLink 单向带宽
    allreduce_latency_us = 8.0

    # 每层 2 次 all-reduce，共 num_layers 层
    msg_time_us = 2 * (tp - 1) / tp * msg_bytes / (ring_bw_gbs * 1e9) * 1e6
    latency_time_us = 2 * num_layers * allreduce_latency_us

    return msg_time_us + latency_time_us


def decode_time_per_token_us(config, batch_size, seq_len, hbm_tbs, tp=1):
    """Decode 每个 token 的生成时间（微秒）。

    公式：时间 = (权重字节/tp + KV字节/tp) / HBM带宽 + TP通信成本

    权重：单 GPU 的分片权重 = 总权重 / tp
    KV：KV 缓存也分散在 TP 卡上，每卡读取 1/tp 的 KV
    通信：TP all-reduce（2次/层 × num_layers）

    TP度数为 1 时，无分片开销。
    """
    weight_total = compute_weight_bytes(config, config)
    weight_per_gpu = weight_total / tp

    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']

    # 每个 decode step 需要搬运的 KV：batch_size * seq_len * 2(K+V) * num_layers * num_kv_heads * head_dim * dtype
    # 这个 KV 也跨 tp 卡分布，每卡只读取 1/tp
    kv_bytes_total_per_gpu = batch_size * seq_len * 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes'] / tp

    # 单 GPU 的搬运字节
    total_bytes = weight_per_gpu + kv_bytes_total_per_gpu

    # 计算延迟：带宽延迟 + 通信延迟
    bw_latency_us = total_bytes / (hbm_tbs * 1e12) * 1e6
    comm_latency_us = tp_allreduce_cost_us(batch_size, config['hidden_size'], tp, num_layers)

    return bw_latency_us + comm_latency_us


def tokens_per_second(batch_size, latency_us):
    """根据延迟计算吞吐（tokens/s）。

    在 batch_size 个请求中，每个要生成 1 个 token，耗时 latency_us。
    所以 tokens/s = batch_size / (latency_us / 1e6)
    """
    return batch_size / (latency_us / 1e6)


def tpot_ms(latency_us):
    """TPOT：Time Per Output Token（毫秒）。

    在自回归解码中，生成一个 token 的时间。
    """
    return latency_us / 1000


def cmd_latency(args):
    """扫 batch 和 seq_len，给出延迟和吞吐。"""
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    tp = args.tp

    print(f"\n{'=' * 110}")
    print(f"Decode 阶段延迟与吞吐分析：{args.model} on {args.gpu} (TP={tp})")
    print(f"{'=' * 110}")

    # 为不同 seq_len 分别打表
    for seq_len in [2048, 4096, 8192, 16384]:
        print(f"\n--- seq_len = {seq_len} ---")
        print(f"{'batch':>8} {'延迟/token (us)':>18} {'延迟/token (ms)':>18} {'吞吐 (tok/s)':>14} {'TPOT (ms)':>12}")
        print(f"{'-' * 75}")

        for batch in [1, 4, 16, 64, 256]:
            latency_us = decode_time_per_token_us(config, batch, seq_len, gpu.hbm_tbs, tp=tp)
            latency_ms = latency_us / 1000
            tps = tokens_per_second(batch, latency_us)
            tpot = tpot_ms(latency_us)

            print(f"{batch:>8} {latency_us:>18.1f} {latency_ms:>18.3f} {tps:>14.1f} {tpot:>12.3f}")


def cmd_compare(args):
    """对比不同模型的 TPOT。"""
    gpu = get(args.gpu)
    batch = args.batch
    seq_len = args.seq_len

    print(f"\n{'=' * 95}")
    print(f"模型对比：TPOT (batch={batch}, seq_len={seq_len})")
    print(f"{'=' * 95}")
    print(f"{'模型':20} {'TP':>4} {'TPOT (ms)':>15} {'吞吐 (tok/s)':>15}")
    print(f"{'-' * 77}")

    # 8B with TP=1, 70B with TP=8, Qwen with TP=1
    model_tp_configs = [
        ('llama-7b-bf16', 1),
        ('llama-70b-bf16', 8),
        ('qwen-14b-bf16', 1),
    ]

    for model_name, tp in model_tp_configs:
        if model_name not in MODEL_CONFIGS:
            continue
        config = MODEL_CONFIGS[model_name]
        latency_us = decode_time_per_token_us(config, batch, seq_len, gpu.hbm_tbs, tp=tp)
        tpot = tpot_ms(latency_us)
        tps = tokens_per_second(batch, latency_us)

        print(f"{model_name:20} {tp:>4} {tpot:>15.3f} {tps:>15.1f}")


def cmd_roofline(args):
    """画出 decode roofline：带宽约束下的算术强度。

    由于 decode 主要是权重和 KV 的搬运，算术强度很低。
    """
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    batch = args.batch
    seq_len = args.seq_len

    weight = compute_weight_bytes(config, config)
    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']
    kv_bytes = batch * seq_len * 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']

    # 算术强度 = 算力 / 内存字节
    # decode 中，权重作 1 次乘加（每字节对应一定的 FLOP），KV 大部分是搬运查询
    # 粗估：权重参数量 * 2 (乘加) / 权重字节 = 2 / dtype_bytes ≈ 1 FLOP/byte (BF16)
    # KV 基本是搬运，0.1 FLOP/byte

    # 每个 token：线性层 2·N FLOP；注意力 QKᵀ 与 PV 每层 4·ctx·d FLOP（d = 头数 × 头维度）
    flops_from_weight = batch * (config['params_b'] * 1e9 * 2 + 4 * num_layers * seq_len * config['hidden_size'])
    # 按 TP 切分后每张卡只搬自己那份字节、算自己那份 FLOP
    tp = getattr(args, 'tp', 1) or 1
    weight, kv_bytes, flops_from_weight = weight / tp, kv_bytes / tp, flops_from_weight / tp
    intensity = flops_from_weight / (weight + kv_bytes)

    peak_tflops = gpu.bf16_tflops
    hw_peak_us = (weight + kv_bytes) / (gpu.hbm_tbs * 1e12) * 1e6

    print(f"\n{'=' * 80}")
    print(f"Decode Roofline（{args.model}, batch={batch}, seq_len={seq_len}）")
    print(f"{'=' * 80}")
    print(f"{'权重（每卡）':30} {weight / 1e9:>10.2f} GB")
    print(f"{'KV cache（每卡）':30} {kv_bytes / 1e9:>10.2f} GB")
    print(f"{'总搬运（每卡）':30} {(weight + kv_bytes) / 1e9:>10.2f} GB")
    print(f"{'算术强度':30} {intensity:>10.2f} FLOP/B")
    print(f"{'HBM 带宽':30} {gpu.hbm_tbs:>10.2f} TB/s")
    print(f"{'峰值算力':30} {peak_tflops:>10.0f} TFLOP/s")
    print(f"{'带宽下限（每步）':30} {hw_peak_us / 1000:>10.2f} ms")

    ridge_point = gpu.bf16_tflops * 1e12 / (gpu.hbm_tbs * 1e12)
    print(f"{'Roofline 拐点':30} {ridge_point:>10.2f} FLOP/B")

    if intensity < ridge_point:
        print(f"\n>>> 该配置处于带宽约束，算术强度 {intensity:.2f} < 拐点 {ridge_point:.2f}")
    else:
        print(f"\n>>> 该配置处于算力约束，算术强度 {intensity:.2f} >= 拐点 {ridge_point:.2f}")


def cmd_scaling(args):
    """扫 batch，观察吞吐饱和与延迟增长。"""
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    seq_len = args.seq_len

    print(f"\n{'=' * 90}")
    print(f"Decode 吞吐扫描：{args.model} on {args.gpu} (seq_len={seq_len})")
    print(f"{'=' * 90}")
    print(f"{'batch':>8} {'延迟/token (ms)':>18} {'吞吐 (tok/s)':>14} {'吞吐/batch':>14} {'利用率':>10}")
    print(f"{'-' * 64}")

    # 峰值带宽对应的理论最大吞吐
    weight = compute_weight_bytes(config, config)
    head_dim = config['hidden_size'] // config['num_heads']

    for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        latency_us = decode_time_per_token_us(config, batch, seq_len, gpu.hbm_tbs)
        tps = tokens_per_second(batch, latency_us)
        tpot = tpot_ms(latency_us)

        # 理论最大：只受权重带宽限制的情况
        max_latency_us = weight / (gpu.hbm_tbs * 1e12) * 1e6
        max_tps = 1 / (max_latency_us / 1e6)  # 单 token 的 token/s

        util = min(100.0, (max_latency_us / latency_us) * 100)  # 相对饱和度估计

        print(f"{batch:>8} {tpot:>18.3f} {tps:>14.1f} {batch/tpot:>14.1f} {util:>9.1f}%")


def decode_time_per_request_ms(config, batch_size, seq_len, output_tokens, hbm_tbs):
    """一个请求从 prefill 到 decode 完成的端到端时间（毫秒）。

    这是简化估计，用于 PD 分离决策。实际需要考虑 pipeline。
    """
    # 每个 decode token 的时间
    time_per_token_us = decode_time_per_token_us(config, batch_size, seq_len, hbm_tbs)
    # 生成 output_tokens 所需总时间
    total_us = time_per_token_us * output_tokens
    return total_us / 1000


def main():
    parser = argparse.ArgumentParser(
        description='Decode 阶段延迟与吞吐分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n'
               '  python3 sim_decode.py latency --model llama-70b-bf16 --gpu h100-sxm\n'
               '  python3 sim_decode.py compare --gpu h100-sxm\n'
               '  python3 sim_decode.py scaling --model llama-70b-bf16 --gpu h100-sxm\n'
    )

    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    p_latency = subparsers.add_parser('latency', help='扫 batch 和 seq_len')
    p_latency.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_latency.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_latency.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_latency.set_defaults(func=cmd_latency)

    p_compare = subparsers.add_parser('compare', help='对比不同模型 TPOT')
    p_compare.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_compare.add_argument('--batch', type=int, default=256, help='batch size')
    p_compare.add_argument('--seq-len', type=int, default=16384, help='序列长度')
    p_compare.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_compare.set_defaults(func=cmd_compare)

    p_roofline = subparsers.add_parser('roofline', help='Decode Roofline 分析')
    p_roofline.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_roofline.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_roofline.add_argument('--batch', type=int, default=256, help='batch size')
    p_roofline.add_argument('--seq-len', type=int, default=16384, help='序列长度')
    p_roofline.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_roofline.set_defaults(func=cmd_roofline)

    p_scaling = subparsers.add_parser('scaling', help='扫 batch 看吞吐')
    p_scaling.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_scaling.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_scaling.add_argument('--seq-len', type=int, default=16384, help='序列长度')
    p_scaling.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_scaling.set_defaults(func=cmd_scaling)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
