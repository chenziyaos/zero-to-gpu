"""专家负载不均与均衡策略：Zipf 分布下的负载分析（CPU）。对应 GPU-22 第 5-7 章。

MoE 的所有 rank 必须等最慢的专家完成，负载不均导致较快的专家空转。
本脚本模拟不同路由分布下的专家负载分布、max/mean 负载比、以及常见均衡策略的改善。

核心概念：
- Zipf 分布（参数 α）模拟 token 对专家的偏好：α 越大偏斜越明显
- 负载不均 = max_load / mean_load（上限，不考虑路由权重）
- 常见策略：
  1. 辅助损失（auxiliary loss）：Switch Transformer，系数 0.01-0.001
  2. 偏置法（bias）：DeepSeek-V3，无辅助损失，通过路由偏置调整 logit
  3. 冗余专家（redundant experts）：EPLB 等方法，额外专家用于均衡

子命令：
  zipf       模拟 Zipf 分布下的专家激活，扫 α 参数看负载不均
  rebalance  展示各均衡策略的改善效果（辅助损失、偏置、冗余专家）
  imbalance  给定 token 分布，计算 max/mean/bottleneck 负载

用法：
  python3 sim_imbalance.py zipf --num-experts 128 --num-tokens 1024 --alpha 1.5
  python3 sim_imbalance.py rebalance --num-experts 128 --num-tokens 1024
  python3 sim_imbalance.py imbalance --distribution zipf --alpha 1.2 --num-experts 128
"""
import argparse
import math
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402


def zipf_distribution(num_experts, num_tokens, alpha, seed=42):
    """生成 Zipf 分布的专家激活频率。

    Zipf 分布：P(k) = (1/k^α) / ζ(α)，其中 ζ(α) 是 Riemann zeta 函数
    直观：rank 1 的专家被激活频率最高，rank k 的频率约为 rank 1 的 1/k^α

    参数：
      alpha：偏斜参数（0 < α < ∞）
             α → 0：均匀分布
             α → ∞：集中在最热专家
             α = 1.5：常见的互联网流量分布；MoE 通常 α = 1.0 - 2.0

    返回：长度为 num_experts 的 token 数组，第 i 个元素表示第 i 个专家被激活次数
    """
    random.seed(seed)

    # 计算 zeta(alpha) 近似：sum_{k=1}^{num_experts} 1/k^alpha
    zeta = sum(1.0 / (k ** alpha) for k in range(1, num_experts + 1))

    # 生成 Zipf 权重
    weights = [1.0 / ((k + 1) ** alpha) / zeta for k in range(num_experts)]

    # 按权重采样 token 到专家
    expert_loads = [0] * num_experts
    for _ in range(num_tokens):
        expert = random.choices(range(num_experts), weights=weights, k=1)[0]
        expert_loads[expert] += 1

    return expert_loads


def uniform_distribution(num_experts, num_tokens):
    """均匀分布：理想情况。"""
    per_expert = num_tokens // num_experts
    remainder = num_tokens % num_experts
    loads = [per_expert] * num_experts
    for i in range(remainder):
        loads[i] += 1
    return loads


def imbalance_ratio(loads):
    """计算负载不均比 = max / mean。"""
    if not loads:
        return 0
    max_load = max(loads)
    mean_load = sum(loads) / len(loads)
    return max_load / mean_load if mean_load > 0 else 0


def imbalance_bottleneck(loads):
    """计算 bottleneck = max_load / num_experts。

    反映在 all-to-all 通信中，最坏情况下单个专家的输出字节数相对于
    平均每个 rank 应该处理的字节数的比例。
    """
    if not loads:
        return 0
    return max(loads) / len(loads)


def apply_auxiliary_loss(loads, num_experts, strength=0.01):
    """模拟辅助损失的均衡效果。

    辅助损失迫使路由器均匀分配，成本是额外的梯度。
    简化模型：重新均衡后的负载为：
      balanced_load[i] = mean_load × (1 + ε_i × strength)
    其中 ε_i 是对专家 i 的修正信号。

    效果：使 max/mean 接近 1 + O(strength)
    """
    mean_load = sum(loads) / len(loads)
    # 简化：线性均衡，最坏也不超过 mean × (1 + 2×strength)
    max_deviation = max(abs(load - mean_load) / mean_load for load in loads)
    reduction = max_deviation * strength
    new_ratio = imbalance_ratio(loads) * (1 - reduction)
    return max(1.0, new_ratio)


def apply_bias_adjustment(loads, num_experts, iterations=3):
    """模拟偏置调整（DeepSeek-V3 风格）。

    不显式优化，而是学习路由器 logit 的偏置项，使其倾向于激活
    被激活较少的专家。

    简化模型：迭代调整，每次降低最热专家的权重。
    """
    temp_loads = loads.copy()
    for iteration in range(iterations):
        max_idx = temp_loads.index(max(temp_loads))
        # 降低最热专家的负载 ~5%
        temp_loads[max_idx] = int(temp_loads[max_idx] * 0.95)
        # 重新分配给其他专家（均匀）
        delta = int(loads[max_idx] * 0.05) // (num_experts - 1)
        for i in range(num_experts):
            if i != max_idx:
                temp_loads[i] += delta

    return imbalance_ratio(temp_loads)


def apply_redundant_experts(loads, num_experts, num_redundant=8):
    """冗余专家均衡（EPLB 等）。

    额外的 num_redundant 个专家用于处理过载的主专家的溢出。
    简化模型：
    - 计算平均负载
    - 让每个过载专家（load > mean）溢出到冗余池
    - 冗余池均衡处理溢出

    效果：新的 max/mean 主要由冗余专家的负载决定
    """
    mean_load = sum(loads) / len(loads)
    overflow = sum(max(0, load - mean_load) for load in loads)

    # 冗余专家均衡分配溢出
    redundant_per_expert = overflow / num_redundant if num_redundant > 0 else 0

    # 新的最大负载：max(main_load, redundant_load)
    main_max = max(min(load, mean_load) for load in loads)
    redundant_max = mean_load + redundant_per_expert

    new_max = max(main_max, redundant_max)
    new_mean = (sum(loads) + overflow) / (num_experts + num_redundant)

    return new_max / new_mean if new_mean > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='cmd')

    # zipf 子命令
    zipf_p = subparsers.add_parser('zipf', help='Zipf 分布的专家负载')
    zipf_p.add_argument('--num-experts', type=int, default=128, help='专家数')
    zipf_p.add_argument('--num-tokens', type=int, default=1024, help='token 数')
    zipf_p.add_argument('--alpha', type=float, default=1.5, help='Zipf 偏斜参数')
    zipf_p.add_argument('--top-k', type=int, default=2, help='每 token top-k（用于显示）')
    zipf_p.add_argument('--seed', type=int, default=42)

    # rebalance 子命令
    rebalance_p = subparsers.add_parser('rebalance', help='各均衡策略对比')
    rebalance_p.add_argument('--num-experts', type=int, default=128)
    rebalance_p.add_argument('--num-tokens', type=int, default=10240)
    rebalance_p.add_argument('--alpha', type=float, default=1.5)
    rebalance_p.add_argument('--seed', type=int, default=42)

    # imbalance 子命令
    imbalance_p = subparsers.add_parser('imbalance', help='负载不均统计')
    imbalance_p.add_argument('--num-experts', type=int, default=128)
    imbalance_p.add_argument('--num-tokens', type=int, default=10240)
    imbalance_p.add_argument('--distribution', choices=['zipf', 'uniform'], default='zipf')
    imbalance_p.add_argument('--alpha', type=float, default=1.5)
    imbalance_p.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.cmd == 'zipf':
        loads = zipf_distribution(args.num_experts, args.num_tokens, args.alpha, args.seed)

        # 找热专家
        sorted_experts = sorted(enumerate(loads), key=lambda x: x[1], reverse=True)
        top_n = min(10, len(sorted_experts))

        print(f"{'Zipf 参数':<20} {args.alpha:<15}")
        print(f"{'专家数':<20} {args.num_experts:<15}")
        print(f"{'总 token 数':<20} {args.num_tokens:<15}")
        print(f"{'平均每专家':<20} {args.num_tokens // args.num_experts:<15}")
        print("-" * 35)
        print(f"{'最大负载':<20} {max(loads):<15}")
        print(f"{'最小负载':<20} {min(loads):<15}")
        print(f"{'平均负载':<20} {sum(loads) / len(loads):<15.2f}")
        print(f"{'max/mean 比':<20} {imbalance_ratio(loads):<15.2f}")
        print()
        print(f"{'热专家（top {top_n}）':<10} {'激活数':<10}")
        print("-" * 20)
        for i, (expert_idx, load) in enumerate(sorted_experts[:top_n]):
            print(f"{'专家 ' + str(expert_idx):<10} {load:<10}")

    elif args.cmd == 'rebalance':
        loads = zipf_distribution(args.num_experts, args.num_tokens, args.alpha, args.seed)
        baseline_ratio = imbalance_ratio(loads)

        # 各策略
        aux_loss_ratio = apply_auxiliary_loss(loads, args.num_experts, strength=0.01)
        bias_ratio = apply_bias_adjustment(loads, args.num_experts, iterations=5)
        redundant_ratio = apply_redundant_experts(loads, args.num_experts, num_redundant=8)

        print(f"{'策略':<25} {'max/mean':<15} {'改善':<15}")
        print("-" * 55)
        print(f"{'无均衡（Zipf α=' + str(args.alpha) + '）':<25} {baseline_ratio:<15.3f} {'-':<15}")
        print(f"{'辅助损失（λ=0.01）':<25} {aux_loss_ratio:<15.3f} {(1 - aux_loss_ratio/baseline_ratio)*100:<14.1f}%")
        print(f"{'偏置调整（5 迭代）':<25} {bias_ratio:<15.3f} {(1 - bias_ratio/baseline_ratio)*100:<14.1f}%")
        print(f"{'冗余专家（8 个）':<25} {redundant_ratio:<15.3f} {(1 - redundant_ratio/baseline_ratio)*100:<14.1f}%")

    elif args.cmd == 'imbalance':
        if args.distribution == 'zipf':
            loads = zipf_distribution(args.num_experts, args.num_tokens, args.alpha, args.seed)
        else:
            loads = uniform_distribution(args.num_experts, args.num_tokens)

        print(f"{'指标':<30} {'数值':<20}")
        print("-" * 50)
        print(f"{'分布类型':<30} {args.distribution:<20}")
        if args.distribution == 'zipf':
            print(f"{'Zipf α':<30} {args.alpha:<20}")
        print(f"{'专家数':<30} {args.num_experts:<20}")
        print(f"{'token 数':<30} {args.num_tokens:<20}")
        print("-" * 50)
        print(f"{'最大负载':<30} {max(loads):<20}")
        print(f"{'最小负载':<30} {min(loads):<20}")
        print(f"{'平均负载':<30} {sum(loads) / len(loads):<20.2f}")
        print(f"{'max/mean（不均比）':<30} {imbalance_ratio(loads):<20.3f}")
        print(f"{'bottleneck（占比）':<30} {imbalance_bottleneck(loads):<20.3f}")


if __name__ == '__main__':
    main()
