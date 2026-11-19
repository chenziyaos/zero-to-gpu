"""KV cache 显存账本与并发估算（CPU）。对应 GPU-23 第 2 至 3 章。

核心公式：
  KV 字节数 (单个请求) = 2 * seq_len * num_kv_heads * head_dim * dtype_bytes
  最大并发 = (可用显存 - 权重 - 激活值) / (KV 字节/请求)

账本给出：
  - 按模型配置计算每个 token 的 KV 字节量
  - 给定可用显存，估算最大并发请求与 batch size
  - 对比 GQA / MLA 的 KV cache 压缩效果

子命令：
  config    解析模型配置，给出 KV 字节/token/请求
  capacity  给定显存和 batch，估算最大上下文和并发
  compare   对比不同 KV 压缩方法
  scaling   扫 batch，看并发与吞吐的关系

用法：
  python3 sim_kv.py config --model llama-70b-bf16
  python3 sim_kv.py capacity --gpu h100-sxm --batch 256
  python3 sim_kv.py compare --model llama-70b-bf16 --seq-len 16384
  python3 sim_kv.py scaling --gpu h100-sxm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402


# 常见大模型配置（单位与含义见注释）
MODEL_CONFIGS = {
    # Llama 3.1 系列
    # 来源：meta-llama/Llama-3.1-8B HF model card (核对于 2026-11-04)
    # 精确参数量：8.03B
    'llama-8b-bf16': {
        'params_b': 8.03,    # 精确参数量，十亿（HF 模型卡）
        'num_layers': 32,    # 层数
        'hidden_size': 4096, # d，隐层维度
        'num_heads': 32,     # num_q_heads
        'num_kv_heads': 8,   # GQA，每个 KV head 对应 4 个 Q head
        'vocab': 128256,
        'dtype_bytes': 2,    # BF16
    },
    # 别名，兼容旧代码
    'llama-7b-bf16': {
        'params_b': 8.03,
        'num_layers': 32,
        'hidden_size': 4096,
        'num_heads': 32,
        'num_kv_heads': 8,
        'vocab': 128256,
        'dtype_bytes': 2,
    },
    # Llama-3.1-70B HF model card (核对于 2026-11-04)
    # 精确参数量：70.6B
    'llama-70b-bf16': {
        'params_b': 70.6,    # 精确参数量，十亿（HF 模型卡）
        'num_layers': 80,    # 80 层
        'hidden_size': 8192,
        'num_heads': 64,
        'num_kv_heads': 8,   # GQA，每个 KV head 对应 8 个 Q head
        'vocab': 128256,
        'dtype_bytes': 2,
    },
    # Qwen 系列
    # 来源：Qwen2.5-14B HF config.json (核对于 2026-11-04)
    'qwen-14b-bf16': {
        'params_b': 14,
        'num_layers': 48,    # 48 层
        'hidden_size': 5120,
        'num_heads': 40,
        'num_kv_heads': 40,  # MHA (num_heads == num_kv_heads)
        'vocab': 152064,
        'dtype_bytes': 2,
    },
    # OPT-13B: vLLM 论文用例
    # 来源：facebook/opt-13b HF config.json (核对于 2026-11-04)
    'opt-13b-fp32': {
        'params_b': 13,
        'num_layers': 40,    # 40 层
        'hidden_size': 5120,
        'num_heads': 40,
        'num_kv_heads': 40,  # MHA
        'vocab': 50272,
        'dtype_bytes': 4,    # FP32
    },
    # DeepSeek-V3: MLA 结构（简化模型，实际 KV 缓存更复杂）
    # 来源：deepseek-ai/DeepSeek-V3 技术文档 (核对于 2026-11-04)
    'deepseek-v3-bf16': {
        'params_b': 671,     # 总参数（包括 MoE）
        'num_layers': 61,    # 61 层
        'hidden_size': 4096,
        'num_heads': 128,
        'num_kv_heads': 128,
        'vocab': 100000,
        'dtype_bytes': 2,
        'note': 'MLA: 实际 KV 缓存使用 kv_lora_rank 512 + qk_rope_head_dim 64，远小于标准 MHA',
    },
}


def compute_weight_bytes(model_name, config):
    """计算模型权重字节数。

    使用精确参数量（来自 HuggingFace 模型卡）：
    - Llama-3.1-8B: 8.03 B
    - Llama-3.1-70B: 70.6 B
    """
    return int(config['params_b'] * 1e9 * config['dtype_bytes'])


def kv_bytes_per_token(config):
    """单个 token 的 KV cache 字节数（单个请求）。

    公式：2 (K + V) * num_layers * num_kv_heads * head_dim * dtype_bytes
    其中 head_dim = hidden_size / num_heads（使用 num_q_heads）
    每层都有自己的 K 和 V 缓存。

    例：Llama 70B (BF16)：80 层 * 8 KV 头 * 128 head_dim * 2 字节 * 2(K+V) = 327,680 bytes/token ≈ 320 KiB
    """
    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']
    # 每层都有 K cache + V cache（两份），总共 num_layers 层
    kv_bytes = 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']
    return kv_bytes


def kv_bytes_per_request(config, seq_len):
    """单个请求的总 KV cache 字节数。"""
    return seq_len * kv_bytes_per_token(config)


def weight_and_activation_bytes(config, batch_size, seq_len):
    """权重和激活值的显存占用（粗估）。

    权重 = compute_weight_bytes()
    激活值 ~= 批量激活缓存 + 临时计算缓存，粗估为权重的 30%-50%
    """
    weight = compute_weight_bytes(config, config)
    # decode 阶段激活值占用较小，粗估权重的 20%；prefill 会更大
    activation = weight * 0.2
    return int(weight + activation)


def max_concurrent_requests(hbm_gb, batch_size, config, seq_len):
    """给定 HBM、batch size 和请求配置，估算最大并发请求数。

    可用显存 = 总 HBM - 权重 - 激活值
    最大并发 = 可用显存 / (KV 字节/请求 * seq_len)

    假设 batch_size 内所有请求长度相同。
    """
    hbm_bytes = hbm_gb * 1e9
    weight_act = weight_and_activation_bytes(config, batch_size, seq_len)
    available = hbm_bytes - weight_act

    if available <= 0:
        return 0, available

    kv_per_req = kv_bytes_per_request(config, seq_len)
    if kv_per_req == 0:
        return 0, available

    # 最多能放多少个这样的请求
    max_req = int(available / kv_per_req)
    return max_req, available


def decode_latency_us(config, batch_size, seq_len, hbm_bw_tbs):
    """Decode 阶段一个 token 的生成延迟（微秒）。

    公式：延迟 = (权重字节 + KV 字节) / (HBM 带宽)
    其中 KV 字节 = batch_size * seq_len * 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    假设权重搬运一遍，KV 读取 batch_size 个请求 × seq_len 个位置。
    """
    weight = compute_weight_bytes(config, config)
    head_dim = config['hidden_size'] // config['num_heads']
    num_layers = config['num_layers']

    # 每个 decode step 需要搬运的 KV：batch_size * seq_len * 2(K+V) * num_layers * KV_dim
    kv_bytes_total = batch_size * seq_len * 2 * num_layers * config['num_kv_heads'] * head_dim * config['dtype_bytes']

    total_bytes = weight + kv_bytes_total
    latency_us = total_bytes / (hbm_bw_tbs * 1e12) * 1e6  # 转成微秒
    return latency_us


def print_config_summary(name, config, tp=1):
    """打印模型配置摘要。

    Args:
        name: 模型名
        config: 模型配置字典
        tp: Tensor Parallel 度数（权重和 KV 字节均除以 tp）
    """
    weight = compute_weight_bytes(config, config)
    weight_per_gpu = weight / tp
    kv_per_token = kv_bytes_per_token(config)
    kv_per_token_per_gpu = kv_per_token / tp

    print(f"\n{'=' * 70}")
    print(f"模型: {name} (TP={tp})")
    print(f"{'=' * 70}")
    print(f"{'参数量':20} {config['params_b']:>8.2f} B")
    print(f"{'隐层维度 (d)':20} {config['hidden_size']:>8}")
    print(f"{'Query heads':20} {config['num_heads']:>8}")
    print(f"{'KV heads':20} {config['num_kv_heads']:>8} (GQA 压缩比 {config['num_heads']/config['num_kv_heads']:.1f}x)")
    print(f"{'权重总计':20} {weight / 1e9:>8.2f} GB")
    print(f"{'权重 / GPU':20} {weight_per_gpu / 1e9:>8.2f} GB")
    print(f"{'KV / token':20} {kv_per_token:>8} bytes")
    print(f"{'KV / token / GPU':20} {kv_per_token_per_gpu:>8.0f} bytes")


def cmd_config(args):
    """打印模型配置和 KV 账本。"""
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}，可选：{', '.join(MODEL_CONFIGS)}")
        return

    config = MODEL_CONFIGS[args.model]
    tp = args.tp
    print_config_summary(args.model, config, tp=tp)

    print(f"\n{'=' * 70}")
    print(f"KV Cache 按上下文长度的占用（整个模型）:")
    print(f"{'=' * 70}")
    print(f"{'seq_len':>10} {'KV/请求 (MB)':>15} {'1K请求 (GB)':>15}")
    print(f"{'-' * 42}")

    for seq_len in [2048, 4096, 8192, 16384, 32768]:
        kv_req = kv_bytes_per_request(config, seq_len)
        kv_1k = kv_req * 1000 / 1e9
        print(f"{seq_len:>10} {kv_req / 1e6:>15.1f} {kv_1k:>15.1f}")


def cmd_capacity(args):
    """估算给定显存下的最大并发（按单 GPU，应用 TP）。"""
    gpu = get(args.gpu)
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    tp = args.tp
    print_config_summary(args.model, config, tp=tp)

    print(f"\n{'=' * 70}")
    print(f"可用显存分析（单 GPU，TP={tp}）")
    print(f"{'=' * 70}")
    print(f"{'总 HBM':20} {gpu.hbm_gb:>8.1f} GB")

    weight_total = compute_weight_bytes(config, config)
    weight_per_gpu = weight_total / tp
    print(f"{'权重 (per GPU)':20} {weight_per_gpu / 1e9:>8.2f} GB")

    # 假设 decode 阶段激活较小，估为权重的 15%
    activation_per_gpu = weight_per_gpu * 0.15
    print(f"{'激活值 (估)':20} {activation_per_gpu / 1e9:>8.2f} GB")

    used_per_gpu = (weight_per_gpu + activation_per_gpu) / 1e9
    available_per_gpu = gpu.hbm_gb - used_per_gpu

    print(f"{'已用 (权重+激活)':20} {used_per_gpu:>8.2f} GB")
    print(f"{'可用 (KV缓存)':20} {available_per_gpu:>8.2f} GB")

    if available_per_gpu < 0:
        print(f"\n⚠️  警告：模型不适合该 GPU！权重和激活已超出 HBM。")
        return

    print(f"\n{'=' * 70}")
    print(f"最大并发 (batch={args.batch}, TP={tp}):")
    print(f"{'=' * 70}")
    print(f"{'seq_len':>10} {'最大并发':>12} {'KV 用量 (GB)':>14} {'利用率 %':>10}")
    print(f"{'-' * 50}")

    for seq_len in [2048, 4096, 8192, 16384]:
        # 每个请求的 KV 字节数（整个模型）
        kv_per_req = kv_bytes_per_request(config, seq_len)
        # 可以容纳的最大并发请求数
        max_req = int(available_per_gpu * 1e9 / kv_per_req)
        kv_used = max_req * kv_per_req / 1e9
        util = (kv_used / available_per_gpu * 100) if available_per_gpu > 0 else 0
        print(f"{seq_len:>10} {max_req:>12} {kv_used:>14.2f} {util:>9.1f}%")


def cmd_compare(args):
    """对比不同 KV 压缩方法（GQA vs MLA）。"""
    if args.model not in MODEL_CONFIGS:
        print(f"未知模型 {args.model}")
        return

    config = MODEL_CONFIGS[args.model]
    base_kv = kv_bytes_per_request(config, args.seq_len)

    print(f"\n{'=' * 70}")
    print(f"KV Cache 压缩效果对比 (seq_len={args.seq_len})")
    print(f"{'=' * 70}")

    print(f"{'方法':20} {'KV/请求 (MB)':>15} {'压缩比':>10} {'显存节省':>12}")
    print(f"{'-' * 60}")

    # 原始 (MHA)
    mha_kv = 2 * config['num_heads'] * (config['hidden_size'] // config['num_heads']) * config['dtype_bytes'] * args.seq_len
    print(f"{'MHA (基准)':20} {mha_kv / 1e6:>15.1f} {'1.0x':>10} {0:>11}%")

    # GQA (当前配置)
    print(f"{'GQA':20} {base_kv / 1e6:>15.1f} {mha_kv/base_kv:>10.1f}x {100*(1-base_kv/mha_kv):>10.1f}%")

    # MLA 估计（压缩 KV 到 512 维）
    if 'MLA' in config.get('note', ''):
        mla_kv = 2 * 512 * config['dtype_bytes'] * args.seq_len  # 低秩投影
        print(f"{'MLA':20} {mla_kv / 1e6:>15.1f} {mha_kv/mla_kv:>10.1f}x {100*(1-mla_kv/mha_kv):>10.1f}%")

    # INT4 量化 (4 bit)
    int4_kv = base_kv / 4  # 4 bit = 0.5 byte
    print(f"{'GQA + INT4':20} {int4_kv / 1e6:>15.1f} {mha_kv/int4_kv:>10.1f}x {100*(1-int4_kv/mha_kv):>10.1f}%")


def cmd_scaling(args):
    """扫 batch size，观察吞吐与延迟的关系。"""
    gpu = get(args.gpu)

    # 用 Llama 70B 作为默认
    config = MODEL_CONFIGS['llama-70b-bf16']
    print_config_summary('llama-70b-bf16 (用于扫描)', config)

    print(f"\n{'=' * 80}")
    print(f"Decode 阶段延迟与吞吐 (seq_len=16384)")
    print(f"{'=' * 80}")
    print(f"{'batch':>8} {'延迟 (ms)':>12} {'吞吐 (tok/s)':>14} {'并发':>8} {'KV占比%':>10}")
    print(f"{'-' * 54}")

    seq_len = 16384
    weight = compute_weight_bytes(config, config)
    hbm_available = gpu.hbm_gb * 1e9 - (weight + weight * 0.15)

    for batch in [1, 4, 16, 64, 256]:
        latency_us = decode_latency_us(config, batch, seq_len, gpu.hbm_tbs)
        latency_ms = latency_us / 1000
        tps = batch / (latency_us / 1e6)  # tokens per second

        kv_used = batch * kv_bytes_per_request(config, seq_len) / 1e9
        kv_pct = (kv_used / (hbm_available / 1e9) * 100) if hbm_available > 0 else 0

        print(f"{batch:>8} {latency_ms:>12.2f} {tps:>14.1f} {batch:>8} {kv_pct:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='KV cache 显存账本与并发估算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n  python3 sim_kv.py config --model llama-70b-bf16\n'
               '  python3 sim_kv.py capacity --gpu h100-sxm --batch 256\n'
               '  python3 sim_kv.py scaling --gpu h100-sxm\n'
    )

    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    p_config = subparsers.add_parser('config', help='打印模型配置和 KV 账本')
    p_config.add_argument('--model', default='llama-70b-bf16',
                          help=f'模型名，可选：{", ".join(MODEL_CONFIGS.keys())}')
    p_config.add_argument('--tp', type=int, default=8,
                          help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_config.set_defaults(func=cmd_config)

    p_capacity = subparsers.add_parser('capacity', help='估算最大并发')
    p_capacity.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_capacity.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_capacity.add_argument('--batch', type=int, default=256, help='batch size')
    p_capacity.add_argument('--tp', type=int, default=8,
                            help='Tensor Parallel 度数（8B 默认 1，70B 默认 8）')
    p_capacity.set_defaults(func=cmd_capacity)

    p_compare = subparsers.add_parser('compare', help='对比 KV 压缩方法')
    p_compare.add_argument('--model', default='llama-70b-bf16', help='模型名')
    p_compare.add_argument('--seq-len', type=int, default=16384, help='序列长度')
    p_compare.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数')
    p_compare.set_defaults(func=cmd_compare)

    p_scaling = subparsers.add_parser('scaling', help='扫 batch，看吞吐变化')
    p_scaling.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    p_scaling.add_argument('--tp', type=int, default=8, help='Tensor Parallel 度数')
    p_scaling.set_defaults(func=cmd_scaling)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
