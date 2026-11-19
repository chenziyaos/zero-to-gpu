#!/usr/bin/env python3
"""GPU-21 显存账本：计算训练中每种并行方案的参数、梯度、优化器、激活显存。

公式（混合精度 Adam 优化器，H100 SXM 口径）：
  参数: (FP16) P 字节
  梯度: (FP16) P 字节
  优化器状态: (FP32 m + v) 2×4P = 8P 字节

  不同并行下的分片:
  - DP (纯数据并行): 每卡 P + P + 8P = 10P 字节
  - ZeRO-1: 每卡 P + P + 8P/N = P(2 + 8/N)
  - ZeRO-2: 每卡 P + P/N + 8P/N = P(1 + 9/N)
  - ZeRO-3: 每卡 P/N + P/N + 8P/N = P(10/N)

  激活（简化）: ~S×H×L×2 (FP16) 其中 S=seq_len, H=hidden_dim, L=layers

使用：
  python3 sim_memory.py model --params 8 --hidden 4096 --layers 32 --seq 8192 --gpu h100-sxm
  python3 sim_memory.py parallel --params 8 --ngpus 8 --strategy dp --gpu h100-sxm
  python3 sim_memory.py parallel --params 8 --ngpus 8 --strategy zero3 --gpu h100-sxm
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get


def model_memory(params_b, hidden_dim, num_layers, seq_len, gpu_name):
    """计算单卡的显存需求。

    params_b: 模型参数，单位 B (Billion)
    """
    gpu = get(gpu_name)
    params_bytes = int(params_b * 1e9)

    # 训练参数
    param_mem = params_bytes * 2  # FP16 参数
    grad_mem = params_bytes * 2   # FP16 梯度
    optim_mem = params_bytes * 8  # FP32 m+v

    # 激活显存（粗略估计）：S×H×L (FP16)
    act_mem = seq_len * hidden_dim * num_layers * 2

    # KV Cache (预留，仅参考)
    kv_mem = seq_len * hidden_dim * num_layers * 2 * 2  # 2 for k,v

    total_no_act = param_mem + grad_mem + optim_mem
    total_with_act = total_no_act + act_mem

    print(f"{'GPU':12} {gpu.name}")
    print(f"{'模型参数':12} {params_b:.1f}B")
    print(f"{'隐层维度':12} {hidden_dim}")
    print(f"{'层数':12} {num_layers}")
    print(f"{'序列长度':12} {seq_len}")
    print(f"{'可用显存':12} {gpu.hbm_gb:.1f} GB")
    print()
    print(f"{'项':12} {'GB':>8} {'%':>6}")
    print(f"{'-'*28}")
    print(f"{'参数 (FP16)':12} {param_mem/1e9:8.2f} {param_mem/gpu.hbm_gb/1e9*100:6.1f}")
    print(f"{'梯度 (FP16)':12} {grad_mem/1e9:8.2f} {grad_mem/gpu.hbm_gb/1e9*100:6.1f}")
    print(f"{'优化器状态':12} {optim_mem/1e9:8.2f} {optim_mem/gpu.hbm_gb/1e9*100:6.1f}")
    print(f"{'激活':12} {act_mem/1e9:8.2f} {act_mem/gpu.hbm_gb/1e9*100:6.1f}")
    print(f"{'-'*28}")
    print(f"{'总计（不含激活）':12} {total_no_act/1e9:8.2f} {total_no_act/gpu.hbm_gb/1e9*100:6.1f}")
    print(f"{'总计（含激活）':12} {total_with_act/1e9:8.2f} {total_with_act/gpu.hbm_gb/1e9*100:6.1f}")
    print()
    print(f"说明：激活为粗略估计 S×H×L×2 (FP16)，不含后向激活和中间结果。")


def parallel_memory(params_b, num_gpus, strategy, gpu_name):
    """计算并行方案下每卡显存。

    strategy: 'dp', 'zero1', 'zero2', 'zero3'
    """
    gpu = get(gpu_name)
    params_bytes = int(params_b * 1e9)

    # 基础成本（混合精度 Adam）
    param_mem = params_bytes * 2      # FP16 参数
    grad_mem = params_bytes * 2        # FP16 梯度
    optim_mem = params_bytes * 8       # FP32 m+v

    strategies = {
        'dp': (param_mem + grad_mem + optim_mem, "纯数据并行，每卡完整副本"),
        'zero1': (
            param_mem + grad_mem + optim_mem / num_gpus,
            f"ZeRO-1，优化器状态分片到 {num_gpus} 卡"
        ),
        'zero2': (
            param_mem + (grad_mem + optim_mem) / num_gpus,
            f"ZeRO-2，梯度+优化器分片到 {num_gpus} 卡"
        ),
        'zero3': (
            (param_mem + grad_mem + optim_mem) / num_gpus,
            f"ZeRO-3，参数+梯度+优化器全部分片到 {num_gpus} 卡"
        ),
    }

    if strategy not in strategies:
        print(f"未知策略 {strategy}，可选：{', '.join(strategies.keys())}")
        return

    per_card_mem, desc = strategies[strategy]

    print(f"{'方案':12} {strategy.upper()}")
    print(f"{'GPU':12} {gpu.name}")
    print(f"{'总卡数':12} {num_gpus}")
    print(f"{'模型大小':12} {params_b:.1f}B")
    print(f"{'可用显存':12} {gpu.hbm_gb:.1f} GB")
    print()
    print(f"每卡成本：")
    print(f"  参数 (FP16)      {param_mem/1e9:.2f} GB")
    print(f"  梯度 (FP16)      {grad_mem/1e9:.2f} GB")
    print(f"  优化器状态       {optim_mem/1e9:.2f} GB")
    print()
    print(f"  {desc}")
    print(f"  每卡显存占用：    {per_card_mem/1e9:.2f} GB ({per_card_mem/gpu.hbm_gb/1e9*100:.1f}%)")
    print()

    # 节省比例
    if strategy != 'dp':
        dp_mem = param_mem + grad_mem + optim_mem
        ratio = dp_mem / per_card_mem
        print(f"相比纯 DP 的节省：{ratio:.1f}×")


def comm_volume(params_b, num_gpus, num_layers, tp_degree, pp_degree, dp_degree):
    """计算每步通信量。

    简化模型：
    - TP: 每层 6 次 all-reduce (weight gradients × 2 + weight updates × 2 等)
    - PP: 微批间的激活 all-gather / all-scatter
    - DP: 每步一次 all-reduce (梯度)
    """
    # 验证
    assert tp_degree * pp_degree * dp_degree == num_gpus, "度数乘积需等于总卡数"

    params_bytes = int(params_b * 1e9)

    # 粗略估计：每层通信量
    # TP: 3 次 all-gather + 3 次 reduce-scatter (每层)
    tp_vol_per_layer = params_bytes // tp_degree  # 权重 / TP 度数

    # PP: 激活和梯度通过流水
    # DP: 梯度 all-reduce (每层梯度)
    dp_vol_grad = params_bytes / dp_degree  # 每卡梯度

    print(f"{'总卡数':12} {num_gpus}")
    print(f"{'TP/PP/DP':12} {tp_degree}/{pp_degree}/{dp_degree}")
    print(f"{'模型大小':12} {params_b:.1f}B")
    print(f"{'层数':12} {num_layers}")
    print()

    if tp_degree > 1:
        total_tp_vol = tp_vol_per_layer * num_layers * 6  # 粗略
        print(f"TP 通信量（每步）：")
        print(f"  每层通信：~{tp_vol_per_layer/1e9:.2f} GB × 6 / {tp_degree} = {tp_vol_per_layer*6/1e9/tp_degree:.2f} GB")
        print(f"  总计 {num_layers} 层：~{total_tp_vol/1e9:.2f} GB")
        print()

    if pp_degree > 1:
        print(f"PP 通信量（气泡除外）：")
        print(f"  激活/梯度通过流水，减少气泡需微批数 m > {pp_degree}")
        print()

    if dp_degree > 1:
        print(f"DP all-reduce（每步末）：")
        print(f"  梯度汇聚 {dp_vol_grad/1e9:.2f} GB，目标带宽决定时间")
        print()


def main():
    parser = argparse.ArgumentParser(description='GPU-21 显存账本')
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # model 子命令
    p_model = subparsers.add_parser('model', help='计算模型显存需求')
    p_model.add_argument('--params', type=float, default=8, help='模型参数（B）')
    p_model.add_argument('--hidden', type=int, default=4096, help='隐层维度')
    p_model.add_argument('--layers', type=int, default=32, help='层数')
    p_model.add_argument('--seq', type=int, default=8192, help='序列长度')
    p_model.add_argument('--gpu', default='h100-sxm', help='GPU 型号')

    # parallel 子命令
    p_parallel = subparsers.add_parser('parallel', help='计算并行方案显存')
    p_parallel.add_argument('--params', type=float, default=8, help='模型参数（B）')
    p_parallel.add_argument('--ngpus', type=int, default=8, help='总 GPU 卡数')
    p_parallel.add_argument('--strategy', default='zero3', choices=['dp', 'zero1', 'zero2', 'zero3'])
    p_parallel.add_argument('--gpu', default='h100-sxm', help='GPU 型号')

    # comm 子命令
    p_comm = subparsers.add_parser('comm', help='计算通信量')
    p_comm.add_argument('--params', type=float, default=8)
    p_comm.add_argument('--ngpus', type=int, default=8)
    p_comm.add_argument('--layers', type=int, default=32)
    p_comm.add_argument('--tp', type=int, default=1)
    p_comm.add_argument('--pp', type=int, default=1)
    p_comm.add_argument('--dp', type=int, default=1)

    args = parser.parse_args()

    if args.cmd == 'model':
        model_memory(args.params, args.hidden, args.layers, args.seq, args.gpu)
    elif args.cmd == 'parallel':
        parallel_memory(args.params, args.ngpus, args.strategy, args.gpu)
    elif args.cmd == 'comm':
        comm_volume(args.params, args.ngpus, args.layers, args.tp, args.pp, args.dp)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
