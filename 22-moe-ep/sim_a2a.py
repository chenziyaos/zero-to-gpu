"""All-to-All 通信量与延迟：MoE 调度和专家并行的通信账（CPU）。对应 GPU-22 第 2 至 4 章。

核心：给定隐层宽度、top-k、EP 规模和节点受限路由，计算每 token 每 rank 的
dispatch 和 combine 字节数、延迟（按 NVLink/RDMA 分层）。

MoE all-to-all 通信时间 = (总发送字节) / (带宽 × rank 数)
DeepSeek-V3 节点受限路由：每 token 最多 4 个节点，跨节点 token 用 RDMA。

子命令：
  dispatch   给定 hidden、top-k、EP rank 数，算 dispatch/combine 字节
  overhead   扫 batch 和 hidden，看通信开销占比
  noderoute  DeepSeek-V3 风格节点受限路由的通信分布：机内 NVLink vs 跨机 RDMA
  latency    按 NVLink/RDMA 带宽估延迟，可选启用通信与计算重叠

用法：
  python3 sim_a2a.py dispatch --hidden 4096 --top-k 2 --ep-rank 8
  python3 sim_a2a.py overhead --hidden 4096 --batch-min 1 --batch-max 128
  python3 sim_a2a.py noderoute --hidden 4096 --top-k 2 --batch 64 --gpus-per-node 8
  python3 sim_a2a.py latency --hidden 4096 --top-k 2 --ep-rank 8 --overlap
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# NVLink 4（H100/H200/B100）带宽及 RDMA 规格
# 来源：NVIDIA 官方规格表（核对于 2026-11-04）
NVLINK4_BW_GB_S = 450  # 单向 GB/s；双向 900 GB/s（不用于 all-to-all 的单向）
RDMA_BW_GB_S = 50     # 400 Gb/s InfiniBand 转换约 50 GB/s，取实测口径


def dispatch_bytes(hidden, num_tokens, top_k, ep_rank, dtype='bf16'):
    """计算 all-to-all dispatch 阶段的总发送字节数。

    前向过程：
    1. 路由决策：hidden dims → top-k 个专家索引 + 权重
    2. dispatch all-to-all：把 token 按其目标专家分散到各 rank

    发送数据：每个 token 拆成 top-k 份给不同 rank，每份含：
    - 嵌入向量（dtype，hidden 维）
    - 路由权重（FP32，4 字节）

    假设 top-k 个专家均匀分布在 ep_rank 个 rank 上，
    每 token 期望从每个 rank 接收 (num_tokens × top_k / ep_rank) 份。

    每 rank 的发送字节 = (num_tokens × top_k / ep_rank) × (dtype_bytes × hidden + 4)
    所有 rank 总计 = num_tokens × top_k × dtype_bytes × hidden + num_tokens × top_k × 4

    参数 dtype：'bf16'（2 字节）、'fp8'（1 字节）、'fp4'（0.5 字节，不常见）
    """
    dtype_bytes = {'bf16': 2, 'fp8': 1, 'fp4': 0.5}[dtype]
    # 每 token top-k 个激活，每个激活 dtype_bytes × hidden
    embedding_bytes = num_tokens * top_k * dtype_bytes * hidden
    # 路由权重（FP32）
    weight_bytes = num_tokens * top_k * 4
    total_bytes = embedding_bytes + weight_bytes
    return total_bytes


def combine_bytes(hidden, num_tokens, top_k, ep_rank, dtype='bf16'):
    """计算 combine 阶段的总接收字节数。

    combine 是 dispatch 的逆：各 rank 汇聚从 N-1 个其他 rank 接收的专家输出。

    输出数据：每个 token 的 top-k 个专家的输出，需要汇总：
    - 每份输出（dtype，hidden 维）
    - 路由权重缩放系数（FP32，4 字节）

    总字节 = num_tokens × top_k × dtype_bytes × hidden + num_tokens × top_k × 4（近似）

    参数 dtype：'bf16'（2 字节）、'fp8'（1 字节）、'fp4'（0.5 字节）
    """
    dtype_bytes = {'bf16': 2, 'fp8': 1, 'fp4': 0.5}[dtype]
    output_bytes = num_tokens * top_k * dtype_bytes * hidden
    weight_bytes = num_tokens * top_k * 4
    total_bytes = output_bytes + weight_bytes
    return total_bytes


def a2a_latency_single_node(dispatch_b, combine_b, bw_gb_s, ep_rank):
    """单节点内 all-to-all（NVLink）延迟。

    all-to-all 阶段 1：各 rank 发送 dispatch_bytes，接收来自其他 ep_rank-1 个 rank 的数据
    阶段 2：各 rank 发送 combine_bytes（通常 dispatch_b ≈ combine_b，近似同时完成）

    Megatron 公式（一般化）：
    T_a2a = (N-1) × total_bytes / (P × BW)
    其中 N=rank 数、total_bytes=dispatch+combine、P=rank 数、BW=聚合带宽
    简化为 T = total_bytes / BW（假设充分流水、broadcast 前已同步）

    更精确的 ring all-to-all：每 rank 需要发送 total_bytes 共 N-1 轮，
    每轮 total_bytes/(N-1) 字节，所以 T = total_bytes / BW（聚合看）。
    """
    total_bytes = dispatch_b + combine_b
    # 转换为 GB 后除以 GB/s 得到秒，再转换为微秒
    # latency_s = total_bytes / (bw_gb_s * 1e9)
    latency_us = (total_bytes / (bw_gb_s * 1e9)) * 1e6
    return latency_us


def a2a_latency_multi_node(dispatch_b, combine_b, gpus_per_node, ep_rank,
                            nvlink_bw, rdma_bw, intra_frac=0.5):
    """跨节点 all-to-all（混合 NVLink 和 RDMA）延迟。

    假设 ep_rank = gpus_per_node × num_nodes。
    节点内通信（NVLink）和跨节点通信（RDMA）并行。

    简化模型：
    - 节点内：dispatch_b 的一部分走 NVLink
    - 跨节点：dispatch_b 的另一部分走 RDMA
    - 两者可以重叠，总延迟 ≈ max(T_nvlink, T_rdma)

    参数：
      intra_frac：dispatch 中节点内通信的比例（0 ≤ intra_frac ≤ 1）
    """
    intra_b = dispatch_b * intra_frac
    inter_b = dispatch_b * (1 - intra_frac)

    # 节点内 all-to-all（微秒）
    t_intra_us = (intra_b / (nvlink_bw * 1e9)) * 1e6

    # 跨节点 all-to-all（微秒）
    t_inter_us = (inter_b / (rdma_bw * 1e9)) * 1e6

    # combine 类似
    t_combine_intra = (combine_b * intra_frac / (nvlink_bw * 1e9)) * 1e6
    t_combine_inter = (combine_b * (1 - intra_frac) / (rdma_bw * 1e9)) * 1e6

    # 最坏情况：max(dispatch, combine)
    t_dispatch = max(t_intra_us, t_inter_us)
    t_combine = max(t_combine_intra, t_combine_inter)

    return t_dispatch + t_combine, t_dispatch, t_combine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='cmd')

    # dispatch 子命令
    dispatch_p = subparsers.add_parser('dispatch', help='计算 dispatch/combine 字节')
    dispatch_p.add_argument('--hidden', type=int, default=4096, help='隐层宽度')
    dispatch_p.add_argument('--top-k', type=int, default=2, help='top-k 专家数')
    dispatch_p.add_argument('--ep-rank', type=int, default=8, help='EP rank 数（同节点 GPU 数）')
    dispatch_p.add_argument('--batch', type=int, default=32, help='batch tokens 数')
    dispatch_p.add_argument('--dtype', choices=['bf16', 'fp8', 'fp4'], default='bf16', help='数据类型')

    # overhead 子命令
    overhead_p = subparsers.add_parser('overhead', help='扫 batch 看通信开销占比')
    overhead_p.add_argument('--hidden', type=int, default=4096)
    overhead_p.add_argument('--batch-min', type=int, default=1)
    overhead_p.add_argument('--batch-max', type=int, default=128)
    overhead_p.add_argument('--top-k', type=int, default=2)
    overhead_p.add_argument('--ep-rank', type=int, default=8)
    overhead_p.add_argument('--gpu', default='h100-sxm', help='GPU 规格')

    # noderoute 子命令（DeepSeek-V3 节点受限路由）
    noderoute_p = subparsers.add_parser('noderoute', help='节点受限路由：NVLink vs RDMA 分布')
    noderoute_p.add_argument('--hidden', type=int, default=4096)
    noderoute_p.add_argument('--top-k', type=int, default=2)
    noderoute_p.add_argument('--batch', type=int, default=64)
    noderoute_p.add_argument('--gpus-per-node', type=int, default=8)
    noderoute_p.add_argument('--num-nodes', type=int, default=4)
    noderoute_p.add_argument('--max-nodes-per-token', type=int, default=4, help='DeepSeek-V3 节点受限')

    # latency 子命令
    latency_p = subparsers.add_parser('latency', help='估延迟')
    latency_p.add_argument('--hidden', type=int, default=4096)
    latency_p.add_argument('--top-k', type=int, default=2)
    latency_p.add_argument('--ep-rank', type=int, default=8)
    latency_p.add_argument('--batch', type=int, default=32)
    latency_p.add_argument('--overlap', action='store_true', help='启用通信与计算重叠')
    latency_p.add_argument('--gpus-per-node', type=int, default=8)

    args = parser.parse_args()

    if args.cmd == 'dispatch':
        dispatch_b = dispatch_bytes(args.hidden, args.batch, args.top_k, args.ep_rank, args.dtype)
        combine_b = combine_bytes(args.hidden, args.batch, args.top_k, args.ep_rank, args.dtype)
        print(f"{'指标':<20} {'数值':<20}")
        print("-" * 40)
        print(f"{'hidden':<20} {args.hidden:<20}")
        print(f"{'top-k':<20} {args.top_k:<20}")
        print(f"{'batch tokens':<20} {args.batch:<20}")
        print(f"{'EP rank 数':<20} {args.ep_rank:<20}")
        print(f"{'dtype':<20} {args.dtype:<20}")
        print("-" * 40)
        print(f"{'dispatch 字节':<20} {dispatch_b / 1024 / 1024:.2f} MB")
        print(f"{'combine 字节':<20} {combine_b / 1024 / 1024:.2f} MB")
        print(f"{'总计':<20} {(dispatch_b + combine_b) / 1024 / 1024:.2f} MB")
        print()
        print(f"{'NVLink4 延迟':<20} {a2a_latency_single_node(dispatch_b, combine_b, NVLINK4_BW_GB_S, args.ep_rank):.2f} us")

    elif args.cmd == 'overhead':
        print(f"{'batch':<10} {'通信字节 (MB)':<20} {'延迟 (us)':<15} {'通信开销%':<10}")
        print("-" * 55)

        for batch in range(args.batch_min, args.batch_max + 1, max(1, (args.batch_max - args.batch_min) // 8)):
            dispatch_b = dispatch_bytes(args.hidden, batch, args.top_k, args.ep_rank, 'bf16')
            combine_b = combine_bytes(args.hidden, batch, args.top_k, args.ep_rank, 'bf16')
            total_b = dispatch_b + combine_b

            # all-to-all 通信延迟（微秒）
            comm_us = a2a_latency_single_node(dispatch_b, combine_b, NVLINK4_BW_GB_S, args.ep_rank)

            # 通信开销占比：假设单个 forward/backward token 在专家上需要 ~100 us 计算
            # 这是估算，实际取决于专家规模
            compute_us = max(100, batch * 10)  # 粗略估计
            overhead_pct = comm_us / (comm_us + compute_us) * 100

            print(f"{batch:<10} {total_b / 1024 / 1024:<20.2f} {comm_us:<15.2f} {overhead_pct:<10.1f}%")

    elif args.cmd == 'noderoute':
        # DeepSeek-V3 风格：每 token 最多 4 个节点
        num_nodes = args.num_nodes
        gpus_per_node = args.gpus_per_node
        ep_rank = num_nodes * gpus_per_node

        dispatch_b = dispatch_bytes(args.hidden, args.batch, args.top_k, ep_rank)
        combine_b = combine_bytes(args.hidden, args.batch, args.top_k, ep_rank)

        # 假设 token 均匀分布，每 token 跨 max_nodes_per_token 个节点
        # 节点内：(gpus_per_node - 1) × gpus_per_node 条边
        # 跨节点：(max_nodes_per_token - 1) × gpus_per_node × num_nodes 条边（近似）
        # 简化：按节点数比例分配
        nodes_touched_per_token = min(args.max_nodes_per_token, num_nodes)
        inter_node_frac = (nodes_touched_per_token - 1) / num_nodes if num_nodes > 1 else 0
        intra_frac = 1 - inter_node_frac

        t_total, t_dispatch, t_combine = a2a_latency_multi_node(
            dispatch_b, combine_b, gpus_per_node, ep_rank,
            NVLINK4_BW_GB_S, RDMA_BW_GB_S, intra_frac=intra_frac
        )

        print(f"{'参数':<30} {'数值':<20}")
        print("-" * 50)
        print(f"{'总节点数':<30} {num_nodes:<20}")
        print(f"{'每节点 GPU 数':<30} {gpus_per_node:<20}")
        print(f"{'总 EP rank 数':<30} {ep_rank:<20}")
        print(f"{'节点受限（max nodes/token）':<30} {args.max_nodes_per_token:<20}")
        print(f"{'预期跨节点比例':<30} {inter_node_frac*100:.1f}%")
        print("-" * 50)
        print(f"{'dispatch 字节':<30} {dispatch_b / 1024 / 1024:.2f} MB")
        print(f"{'combine 字节':<30} {combine_b / 1024 / 1024:.2f} MB")
        print(f"{'总通信字节':<30} {(dispatch_b + combine_b) / 1024 / 1024:.2f} MB")
        print("-" * 50)
        print(f"{'dispatch 延迟':<30} {t_dispatch:.2f} us")
        print(f"{'combine 延迟':<30} {t_combine:.2f} us")
        print(f"{'总延迟':<30} {t_total:.2f} us")

    elif args.cmd == 'latency':
        dispatch_b = dispatch_bytes(args.hidden, args.batch, args.top_k, args.ep_rank)
        combine_b = combine_bytes(args.hidden, args.batch, args.top_k, args.ep_rank)

        num_nodes = (args.ep_rank + args.gpus_per_node - 1) // args.gpus_per_node
        is_multinode = num_nodes > 1

        print(f"{'配置':<30} {'数值':<20}")
        print("-" * 50)
        print(f"{'隐层宽度':<30} {args.hidden:<20}")
        print(f"{'top-k':<30} {args.top_k:<20}")
        print(f"{'batch tokens':<30} {args.batch:<20}")
        print(f"{'EP rank 数':<30} {args.ep_rank:<20}")
        print(f"{'每节点 GPU 数':<30} {args.gpus_per_node:<20}")
        print(f"{'节点数':<30} {num_nodes:<20}")
        print("-" * 50)

        if is_multinode:
            intra_frac = 1.0 / num_nodes  # 粗略估计
            t_total, t_dispatch, t_combine = a2a_latency_multi_node(
                dispatch_b, combine_b, args.gpus_per_node, args.ep_rank,
                NVLINK4_BW_GB_S, RDMA_BW_GB_S, intra_frac=intra_frac
            )
            print(f"{'模式':<30} {'多节点（NVLink + RDMA）':<20}")
            print(f"{'dispatch 延迟':<30} {t_dispatch:.2f} us")
            print(f"{'combine 延迟':<30} {t_combine:.2f} us")
        else:
            t_total = a2a_latency_single_node(dispatch_b, combine_b, NVLINK4_BW_GB_S, args.ep_rank)
            print(f"{'模式':<30} {'单节点（NVLink）':<20}")

        print(f"{'总延迟':<30} {t_total:.2f} us")
        if args.overlap:
            print(f"{'通信与计算重叠系数':<30} {'0.8':<20}（假设）")
            effective_latency = t_total * 0.2
            print(f"{'有效延迟（重叠后）':<30} {effective_latency:.2f} us")


if __name__ == '__main__':
    main()
