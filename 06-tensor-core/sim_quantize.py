"""低精度格式与缩放：FP8、FP4 的精度账和缩放粒度（对应 GPU-06 第 6-10 章）。

核心公式：
  1. 浮点表示范围 = 2^exp_bias × (1 + 小数部分)
  2. FP8 Ampere 口径：E4M3（指数 4 位、尾数 3 位）范围 [2^-7, 2^8)，最小非零 2^-10
  3. FP8 H100 口径：E5M2（指数 5 位、尾数 2 位）范围 [2^-15, 2^16)，最小非零 2^-17
  4. FP4（NVIDIA MXFP4）：两级缩放，共享指数块 + 元素级重新缩放
  5. 混合精度：累加 FP32，权重和激活可用 FP8；量化噪声 σ = 范围 / (√12 × 样本大小)
  6. 缩放因子粒度：向量级（K 方向）、行级（M 方向）或块级（根据数据分布）

数据来源：
  - NVIDIA FP8 技术文档，E4M3/E5M2 规格（核对于 2026-11-04）
  - DeepSeek-V3 论文附录，FP8 训练配置（arXiv:2412.19437）
  - CUTLASS 库 float_e4m3.h、float_e5m2.h（主分支 2026-11）

子命令：
  format      各格式的范围、精度和量化误差
  overhead    缩放因子的显存开销
  noise       混合精度的量化噪声估计
  deadzone    FP8 的下溢和过流风险评估
  deepseekmoe DeepSeek-V3 FP8 配方（3 层缩放 + 块级梯度归约）

用法：
  python3 sim_quantize.py format
  python3 sim_quantize.py format --fmt fp8-e4m3 --check-range 1e-7 1e3
  python3 sim_quantize.py noise --weight-fmt fp8 --act-fmt fp8 --batch 64 --dim 4096
  python3 sim_quantize.py deadzone --fmt fp8 --grad-scale 100.0
  python3 sim_quantize.py deepseekmoe --seq 8192 --vocab 102400 --num-layers 160
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# FP8 格式规格
# 来源：NVIDIA A100/H100 技术文档及 CUTLASS 库（核对于 2026-11-04）
FORMATS = {
    'fp32': {
        'bits': 32,
        'exp_bits': 8,
        'mant_bits': 23,
        'exp_bias': 127,
        'max': 3.4e38,
        'min': 1.4e-45,
        'range_approx': 90,  # log10(max/min)
    },
    'fp16': {
        'bits': 16,
        'exp_bits': 5,
        'mant_bits': 10,
        'exp_bias': 15,
        'max': 6.5e4,
        'min': 6.1e-5,
        'range_approx': 15,
    },
    'bf16': {
        'bits': 16,
        'exp_bits': 8,
        'mant_bits': 7,
        'exp_bias': 127,
        'max': 3.4e38,
        'min': 1.2e-38,
        'range_approx': 90,
    },
    'fp8-e4m3': {
        'bits': 8,
        'exp_bits': 4,
        'mant_bits': 3,
        'exp_bias': 7,
        'max': 448.0,
        'min': 1.52e-4,
        'range_approx': 6.5,
        'note': 'Ampere 口径',
    },
    'fp8-e5m2': {
        'bits': 8,
        'exp_bits': 5,
        'mant_bits': 2,
        'exp_bias': 15,
        'max': 57344.0,
        'min': 6.1e-5,
        'range_approx': 21,
        'note': 'Hopper 口径',
    },
}

# 量化噪声（均匀量化的高斯模型）
# σ_quantize = 范围 / (√12 × 有效比特数)


def quantization_noise(num_levels):
    """均匀量化的噪声标准差（一阶）。"""
    return 1.0 / (math.sqrt(12) * num_levels)


def cmd_format(args):
    """浮点格式的表示范围和量化误差。"""
    print(f'{"格式":15} {"位宽":>6} {"范围（log10）":>14} {"最大值":>14} {"最小值":>14}')
    print('-' * 70)

    for fmt_name in ['fp32', 'fp16', 'bf16', 'fp8-e4m3', 'fp8-e5m2']:
        fmt = FORMATS.get(fmt_name)
        if fmt:
            if args.check_range and args.fmt != fmt_name:
                continue
            if args.fmt and args.fmt != fmt_name:
                continue

            log_range = math.log10(fmt['max'] / fmt['min'])
            print(f'{fmt_name:15} {fmt["bits"]:6d} {log_range:14.1f} {fmt["max"]:14.3e} {fmt["min"]:14.3e}')

            if 'note' in fmt:
                print(f'  {fmt["note"]}')

            # 验证某个值的存储性
            if args.check_range:
                val_min, val_max = args.check_range
                in_range = val_min >= fmt['min'] and val_max <= fmt['max']
                result = '✓ 可表示' if in_range else '✗ 超出范围'
                print(f'  [{val_min:.1e}, {val_max:.1e}]: {result}')


def cmd_overhead(args):
    """缩放因子的显存开销。"""
    total_params = args.params  # 百万
    scale_granule = args.granule  # 'vector' (K 方向), 'row' (M 方向), 'block' (粗粒度)

    # 假设矩阵形状 [M, K]，缩放因子是 FP32（4 字节）
    # 向量级：每 K 个元素一个，共 K / granule_size 个
    # 行级：每 M 个元素一个，共 M / granule_size 个

    if scale_granule == 'vector':
        # 假设 K = sqrt(params)，缩放数 ~ sqrt(params)
        num_scales = int(math.sqrt(total_params * 1e6))
    elif scale_granule == 'row':
        # 假设 M = sqrt(params)，缩放数 ~ sqrt(params)
        num_scales = int(math.sqrt(total_params * 1e6))
    elif scale_granule == 'block':
        # 块粒度：假设块大小 128×128，缩放数 ~ params / 128
        num_scales = int(total_params * 1e6 / (128 * 128))
    else:
        num_scales = int(total_params * 1e6)

    bytes_per_scale = 4  # FP32
    total_overhead_gb = num_scales * bytes_per_scale / 1e9

    print(f'{"参数":20} {"值":>25}')
    print('-' * 50)
    print(f'{"总参数（M）":20} {total_params:>25}')
    print(f'{"缩放粒度":20} {scale_granule:>25}')
    print(f'{"缩放数量":20} {num_scales:>25}')
    print(f'{"每个缩放（字节）":20} {bytes_per_scale:>25}')
    print(f'{"总开销（GB）":20} {total_overhead_gb:>25.3f}')
    print(f'{"相对模型（%）":20} {(total_overhead_gb/(total_params*2/1000))*100:>25.2f}')


def cmd_noise(args):
    """混合精度的量化噪声。"""
    fmt_w = args.weight_fmt
    fmt_a = args.act_fmt
    batch = args.batch
    dim = args.dim

    fmt_w_spec = FORMATS.get(fmt_w, {})
    fmt_a_spec = FORMATS.get(fmt_a, {})

    if not fmt_w_spec or not fmt_a_spec:
        print(f'ERROR: 未知格式 {fmt_w} 或 {fmt_a}')
        return

    # 权重和激活噪声（假设数据在 [-1, 1] 范围，缩放到格式的max）
    # 噪声 σ ~ 格式范围 / (√12 × 有效位数)
    num_levels_w = 2 ** fmt_w_spec['mant_bits']
    num_levels_a = 2 ** fmt_a_spec['mant_bits']

    noise_w = 1.0 / num_levels_w  # 相对噪声
    noise_a = 1.0 / num_levels_a

    # 梯度噪声（反向传播）
    # 梯度 = dL/dW ~ dL/dA × dA/dW
    # 梯度噪声 ~ 损失梯度噪声 × 激活梯度噪声
    # 简化：σ_grad ~ σ_act × batch
    noise_grad = noise_a * math.sqrt(batch)

    # 误差界估计（Kolmogorov-Arnold）
    # 对于 LLM，相对误差 ~ σ / E[|value|] ~ σ / √dim（假设 iid）
    error_bound = (noise_w + noise_grad) / math.sqrt(dim)

    print(f'{"指标":25} {"值":>20}')
    print('-' * 50)
    print(f'{"权重格式":25} {fmt_w:>20}')
    print(f'{"激活格式":25} {fmt_a:>20}')
    print(f'{"Batch 大小":25} {batch:>20}')
    print(f'{"隐层维度":25} {dim:>20}')
    print()
    print(f'{"权重相对噪声":25} {noise_w:>20.6f}')
    print(f'{"激活相对噪声":25} {noise_a:>20.6f}')
    print(f'{"梯度相对噪声":25} {noise_grad:>20.6f}')
    print(f'{"误差界（相对）":25} {error_bound:>20.6f}')

    # 判断是否可训练
    if error_bound < 0.01:
        print(f'{"结论":25} {"✓ 可训练（相对误差 <1%）":>20}')
    elif error_bound < 0.05:
        print(f'{"结论":25} {"⚠ 可训练但需关注精度":>20}')
    else:
        print(f'{"结论":25} {"✗ 误差过大，不推荐":>20}')


def cmd_deadzone(args):
    """FP8 的下溢和过流。"""
    fmt_spec = FORMATS.get(args.fmt, {})

    if not fmt_spec:
        print(f'ERROR: 未知格式 {args.fmt}')
        return

    grad_scale = args.grad_scale
    fmt_max = fmt_spec['max']
    fmt_min = fmt_spec['min']

    # 梯度经过缩放因子后的范围
    grad_min_scaled = grad_scale
    grad_max_scaled = grad_scale

    # 判断溢出风险
    overflow_risk = 'SAFE' if grad_max_scaled < fmt_max else 'RISK'
    underflow_risk = 'SAFE' if grad_min_scaled > fmt_min else 'RISK'

    print(f'{"参数":25} {"值":>20}')
    print('-' * 50)
    print(f'{"格式":25} {args.fmt:>20}')
    print(f'{"缩放因子（S）":25} {grad_scale:>20.2e}')
    print(f'{"格式最大值":25} {fmt_max:>20.2e}')
    print(f'{"格式最小值":25} {fmt_min:>20.2e}')
    print()
    print(f'{"过流风险":25} {overflow_risk:>20}')
    print(f'{"下溢风险":25} {underflow_risk:>20}')

    # 推荐缩放范围
    safe_scale_max = fmt_max / 10.0
    safe_scale_min = fmt_min * 10.0
    print(f'{"推荐缩放范围":25} {f"[{safe_scale_min:.2e}, {safe_scale_max:.2e}]":>20}')


def cmd_deepseekmoe(args):
    """DeepSeek-V3 FP8 配方（arXiv:2412.19437）。"""
    seq_len = args.seq
    vocab = args.vocab
    num_layers = args.num_layers

    # DeepSeek-V3：671B 参数，160 层，3968 隐层维度
    hidden_dim = 3968 if not args.hidden_dim else args.hidden_dim
    num_attn_heads = 128 if not args.num_heads else args.num_heads
    num_experts = 256 if not args.num_experts else args.num_experts

    # FP8 训练三件套：
    # 1. 权重量化（每层 vector 级缩放）
    # 2. 激活量化（每层 vector 级缩放）
    # 3. 梯度缩放（块级）

    # 显存消耗估算（前向 + 反向）
    # 参数：671B FP8 = 671 GB
    # 激活：seq × hidden × layers × 2（前后向）× FP32
    activation_bytes = seq_len * hidden_dim * num_layers * 2 * 4 / 1e9

    # 优化器状态（Adam：m + v）
    optimizer_bytes = 671 * 2  # FP32 m 和 v

    # 梯度缩放（每 128×128 块一个）
    blocks_per_layer = (hidden_dim * hidden_dim) / (128 * 128)
    scale_bytes = num_layers * blocks_per_layer * 4 / 1e9

    print(f'{"配置":25} {"值":>20}')
    print('-' * 50)
    print(f'{"模型参数（B）":25} {671:>20}')
    print(f'{"层数":25} {num_layers:>20}')
    print(f'{"隐层维度":25} {hidden_dim:>20}')
    print(f'{"序列长度":25} {seq_len:>20}')
    print(f'{"词汇表大小":25} {vocab:>20}')
    print()
    print(f'{"显存消耗":25} {"":>20}')
    print(f'{"参数（FP8，GB）":25} {671:>20}')
    print(f'{"激活（FP32，GB）":25} {activation_bytes:>20.1f}')
    print(f'{"优化器状态（GB）":25} {optimizer_bytes:>20.0f}')
    print(f'{"缩放因子（GB）":25} {scale_bytes:>20.3f}')

    # 总显存需求（假设 H100）
    total_hbm = 671 + activation_bytes + optimizer_bytes + scale_bytes
    print(f'{"总计（GB）":25} {total_hbm:>20.1f}')

    # 与 FP32 对比
    fp32_total = 671 * 4 + activation_bytes
    reduction = (1 - total_hbm / fp32_total) * 100
    print(f'{"相对 FP32 节省（%）":25} {reduction:>20.1f}')

    print()
    print('FP8 配方（DeepSeek-V3）：')
    print('  1. 权重：E5M2，vector 级缩放（K 方向，每 K/128 个缩放因子）')
    print('  2. 激活：E5M2，vector 级缩放（同权重）')
    print('  3. 梯度：E5M2，块级缩放（128×128，训练稳定性）')
    print('  4. 累加：FP32（保留全精度）')
    print('  5. 缩放因子：每步动态调整，基于梯度直方图')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # format 子命令
    sp = subparsers.add_parser('format', help='浮点格式')
    sp.add_argument('--fmt', default=None, help='指定格式')
    sp.add_argument('--check-range', nargs=2, type=float, default=None,
                    metavar=('MIN', 'MAX'), help='检查值范围是否可表示')
    sp.set_defaults(func=cmd_format)

    # overhead 子命令
    sp = subparsers.add_parser('overhead', help='缩放开销')
    sp.add_argument('--params', type=float, default=671, help='总参数数（百万）')
    sp.add_argument('--granule', default='vector', choices=['vector', 'row', 'block'],
                    help='缩放粒度')
    sp.set_defaults(func=cmd_overhead)

    # noise 子命令
    sp = subparsers.add_parser('noise', help='量化噪声')
    sp.add_argument('--weight-fmt', default='fp8-e5m2', help='权重格式')
    sp.add_argument('--act-fmt', default='fp8-e5m2', help='激活格式')
    sp.add_argument('--batch', type=int, default=64, help='Batch 大小')
    sp.add_argument('--dim', type=int, default=4096, help='隐层维度')
    sp.set_defaults(func=cmd_noise)

    # deadzone 子命令
    sp = subparsers.add_parser('deadzone', help='溢出风险')
    sp.add_argument('--fmt', default='fp8-e5m2', help='量化格式')
    sp.add_argument('--grad-scale', type=float, default=100.0, help='梯度缩放因子')
    sp.set_defaults(func=cmd_deadzone)

    # deepseekmoe 子命令
    sp = subparsers.add_parser('deepseekmoe', help='DeepSeek FP8 配方')
    sp.add_argument('--seq', type=int, default=8192, help='序列长度')
    sp.add_argument('--vocab', type=int, default=102400, help='词汇表大小')
    sp.add_argument('--num-layers', type=int, default=160, help='层数')
    sp.add_argument('--hidden-dim', type=int, default=None, help='隐层维度')
    sp.add_argument('--num-heads', type=int, default=None, help='注意力头数')
    sp.add_argument('--num-experts', type=int, default=None, help='专家数')
    sp.set_defaults(func=cmd_deepseekmoe)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
