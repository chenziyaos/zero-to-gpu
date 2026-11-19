"""OOM 重试链与 HBM 预算模拟。对应 GPU-10 第 4、8 章。

OOM 前的重试流程（c10/cuda/CUDACachingAllocator.cpp，emptyCache 及 malloc 路径）：
  1. 试图分配失败（cudaMalloc 返回 error）
  2. 尝试释放"超大块"（> 总预留的 50%？阈值见代码）
  3. 同步全部流，再释放所有空闲段
  4. 再试一次 malloc
  5. 仍失败才抛 RuntimeError

训练显存预算（第 8 章，来自 Transformer 推导）：
  模型参数：P 个参数
  激活：单样本单层激活通常 (2L * S * D) 字节（L 层，S seq, D 模型宽度）
  梯度：与参数同大小
  优化器状态：Adam 是参数的 2 倍（moment1, moment2）
  中间激活（checkpointing 影响）

子命令：
  retry      模拟 OOM 重试链：给定显存请求，演示重试次数与释放阶段
  membudget  训练显存预算计算器：P/L/S/D 参数入，输出单卡显存需求（不含 gradient）
  budget_sweep  扫参数空间，找出单卡 40GB H100 能放下的最大模型规模
  oom_pattern    解析 PyTorch OOM 报错字符串，提取碎片率和重试计数

用法：
  python3 sim_oom_retry.py retry --total-reserved-gb 40 --request-gb 5
  python3 sim_oom_retry.py membudget --params-billion 7 --layers 32 --seq 2048 --dmodel 4096
  python3 sim_oom_retry.py budget-sweep --gpu h100-sxm
  python3 sim_oom_retry.py oom-pattern --error-msg "CUDA out of memory..."
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# ===== PyTorch 显存预算常数 =====
# 来源：PyTorch 文档 + Transformer 显存分析
# 核对于 2026-11-04

# OOM 重试释放阈值：超过显存预留的多少百分比算"超大块"（保守估计）
LARGE_BLOCK_THRESHOLD_PCT = 0.50  # 50% 阈值估计，真实代码可能不同

# Transformer 训练显存（per token, per layer 近似）
# 来源：Chen et al. 2016, Korthikanti et al. ZeRO-3, Transformers 官方文档

def bytes_fmt(b):
    """格式化字节数为易读形式。"""
    for unit, divisor in [('GB', 1e9), ('MB', 1e6), ('KB', 1e3)]:
        if b >= divisor:
            return f"{b / divisor:.2f} {unit}"
    return f"{b} B"


@dataclass
class MemBudgetBreakdown:
    """显存预算分项。"""
    params: int  # 参数显存（byte）
    activations: int  # 激活显存
    gradients: int  # 梯度显存
    optimizer_state: int  # 优化器状态（Adam: 2x params）
    kv_cache: int  # KV cache（推理时才算，训练时通常 0）

    @property
    def total(self) -> int:
        return self.params + self.activations + self.gradients + self.optimizer_state + self.kv_cache


def estimate_model_params(params_billion: float, dtype_bytes: int = 2) -> int:
    """估计模型参数显存（纯参数，未计梯度）。

    Args:
        params_billion: 模型参数数（单位：十亿）
        dtype_bytes: 数据类型字节数（BF16/FP16 = 2, FP32 = 4）
    """
    return int(params_billion * 1e9 * dtype_bytes)


def estimate_activation_per_token_per_layer(
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int = 2
) -> int:
    """
    估计每层每 token 的激活显存。
    来源：Transformer 架构分析，y = x + attention(norm(x)) + ffn(norm(x))
    每层需要保存约 2 * S * D（S=seq_len, D=hidden_size）用于重算。

    简化：不计 attention 矩阵 (N^2 对于长序列太大)。
    """
    # 每层输入/输出激活
    return 2 * seq_len * hidden_size * dtype_bytes


def estimate_activation_total(
    layers: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int = 2,
    checkpointing: bool = False
) -> int:
    """
    总激活显存。

    Args:
        checkpointing: 若 True，只保存重新计算点而非全部激活，可减少 ~70%
    """
    per_layer = estimate_activation_per_token_per_layer(seq_len, hidden_size, dtype_bytes)
    total = layers * per_layer
    if checkpointing:
        total = int(total * 0.3)  # 保守估计：保留 30% 用于重新计算点
    return total


def estimate_oom_retry_sequence(
    total_reserved_gb: float,
    request_gb: float,
    num_segments: int = 10,
    avg_segment_frag: float = 0.2
) -> Tuple[int, str]:
    """
    模拟 OOM 重试流程，返回重试次数和详细过程。

    Args:
        total_reserved_gb: 显存总预留（GB）
        request_gb: 当前请求（GB）
        num_segments: 预估段数（用于 coalesce 收益估计）
        avg_segment_frag: 平均段碎片率

    Returns:
        (重试次数, 过程描述)
    """
    total_bytes = total_reserved_gb * 1e9
    request_bytes = request_gb * 1e9

    if request_bytes > total_bytes:
        return -1, "请求超过总显存，无法满足"

    steps = []
    available = total_bytes
    retry_count = 0

    # 第 1 步：直接尝试
    steps.append(f"第 1 次尝试：请求 {request_gb:.2f} GB，可用 {available / 1e9:.2f} GB")
    if available >= request_bytes:
        return 0, "\n".join(steps) + "\n成功（无需重试）"

    retry_count += 1

    # 第 2 步：释放超大块
    large_threshold = total_bytes * LARGE_BLOCK_THRESHOLD_PCT
    estimated_freed_large = large_threshold * 0.5  # 保守估计只能释放超大块池的 50%
    available += estimated_freed_large
    steps.append(f"重试 {retry_count}：释放超大块，回收 ~{estimated_freed_large / 1e9:.2f} GB，可用 {available / 1e9:.2f} GB")

    if available >= request_bytes:
        return retry_count, "\n".join(steps) + "\n成功"

    retry_count += 1

    # 第 3 步：同步所有流 + 释放所有空闲段
    # 收益：段合并 + 浪费空间回收
    estimated_freed_coalesce = (total_bytes * avg_segment_frag) * 0.7  # 70% 的碎片可能被回收
    available += estimated_freed_coalesce
    steps.append(f"重试 {retry_count}：同步流 + 合并段，回收 ~{estimated_freed_coalesce / 1e9:.2f} GB，可用 {available / 1e9:.2f} GB")

    if available >= request_bytes:
        return retry_count, "\n".join(steps) + "\n成功"

    return -1, "\n".join(steps) + f"\nOOM 失败：还差 {(request_bytes - available) / 1e9:.2f} GB"


def cmd_retry(args):
    """模拟 OOM 重试链。"""
    retry_count, narrative = estimate_oom_retry_sequence(
        args.total_reserved_gb,
        args.request_gb,
        num_segments=args.num_segments,
        avg_segment_frag=args.avg_frag
    )

    print(narrative)
    print()
    if retry_count >= 0:
        print(f"结论：经历 {retry_count} 次重试，最终分配成功")
    else:
        print("结论：OOM，无法通过重试满足请求")


def cmd_membudget(args):
    """训练显存预算计算器。"""
    dtype = 2 if args.dtype == 'bf16' else (4 if args.dtype == 'fp32' else 1)

    params = estimate_model_params(args.params_billion, dtype)
    activations = estimate_activation_total(
        args.layers,
        args.seq,
        args.dmodel,
        dtype,
        checkpointing=args.checkpointing
    )
    gradients = params  # 与参数同大小
    optimizer_state = params * 2 if args.optimizer == 'adam' else 0

    budget = MemBudgetBreakdown(
        params=params,
        activations=activations,
        gradients=gradients,
        optimizer_state=optimizer_state,
        kv_cache=0
    )

    print(f"显存预算（单卡，{args.dtype}）")
    print(f"{'':20} {'字节':>15} {'占比':>8}")
    print("-" * 50)
    print(f"{'参数':20} {bytes_fmt(budget.params):>15} {100*budget.params/budget.total:>7.1f}%")
    print(f"{'激活':20} {bytes_fmt(budget.activations):>15} {100*budget.activations/budget.total:>7.1f}%")
    print(f"{'梯度':20} {bytes_fmt(budget.gradients):>15} {100*budget.gradients/budget.total:>7.1f}%")
    print(f"{'优化器状态':20} {bytes_fmt(budget.optimizer_state):>15} {100*budget.optimizer_state/budget.total:>7.1f}%")
    print("-" * 50)
    print(f"{'总计':20} {bytes_fmt(budget.total):>15} {'100.0%':>8}")
    print()
    print(f"模型规格：{args.params_billion}B 参数，{args.layers} 层，序列长度 {args.seq}，隐藏维 {args.dmodel}")
    if args.checkpointing:
        print("激活已启用梯度检查点（~70% 节省）")


def cmd_budget_sweep(args):
    """扫参数空间，找出单卡能放下的最大模型。"""
    gpu = get(args.gpu)
    target_gb = gpu.hbm_gb * 0.8  # 使用 80% 留给 overhead

    print(f"{args.gpu} 显存 {gpu.hbm_gb}GB，目标使用 {target_gb:.0f}GB")
    print()
    print(f"{'模型大小':>8} {'层':>5} {'隐藏维':>8} {'序列长':>8} {'预算':>8} {'适配':>8}")
    print("-" * 60)

    for params_b in [1, 3, 7, 13, 20, 30]:
        for layers in [24, 32, 40]:
            for dmodel in [2048, 4096, 8192]:
                for seq in [512, 2048, 4096, 8192]:
                    budget = estimate_model_params(params_b, 2)
                    activations = estimate_activation_total(layers, seq, dmodel, 2, checkpointing=True)
                    gradients = budget
                    optimizer = budget * 2
                    total = budget + activations + gradients + optimizer

                    fits = "✓" if total <= target_gb * 1e9 else "✗"
                    if fits == "✓":
                        total_gb = total / 1e9
                        print(f"{params_b:>7}B {layers:>5} {dmodel:>8} {seq:>8} {total_gb:>7.1f}GB {fits:>8}")


def cmd_oom_pattern(args):
    """解析 OOM 报错字符串。"""
    msg = args.error_msg if args.error_msg else (
        "RuntimeError: CUDA out of memory. Tried to allocate 5.00 GiB "
        "(GPU 0; 79.99 GiB total capacity; 60.50 GiB already allocated; "
        "2.10 GiB free; 11.20 GiB reserved but unallocated; "
        "0 GiB unaccounted for)"
    )

    print("OOM 报错解析：")
    print(msg)
    print()

    # 简单正则式提取关键数字
    import re
    patterns = {
        '请求': r'Tried to allocate ([\d.]+) GiB',
        '总容量': r'total capacity; ([\d.]+) GiB',
        '已分配': r'already allocated; ([\d.]+) GiB',
        '空闲': r'([\d.]+) GiB free',
        '保留未分配': r'([\d.]+) GiB reserved but unallocated',
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, msg)
        if match:
            extracted[key] = float(match.group(1))

    if 'total capacity' in extracted and 'already allocated' in extracted:
        cap = extracted['总容量']
        alloc = extracted['已分配']
        frag = extracted['reserved but unallocated'] if '保留未分配' in extracted else 0
        free = extracted['free'] if 'free' in extracted else (cap - alloc)

        print(f"{'总容量':15} {cap:>8.2f} GB")
        print(f"{'已分配':15} {alloc:>8.2f} GB ({100*alloc/cap:.1f}%)")
        print(f"{'缓存（未分配）':15} {frag:>8.2f} GB ({100*frag/cap:.1f}%)")
        print(f"{'可用':15} {free:>8.2f} GB ({100*free/cap:.1f}%)")
        print(f"{'碎片率':15} {100*frag/(alloc+frag) if alloc+frag > 0 else 0:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    sp_retry = subparsers.add_parser('retry', help='OOM 重试流程')
    sp_retry.add_argument('--total-reserved-gb', type=float, default=80, help='总显存预留（GB）')
    sp_retry.add_argument('--request-gb', type=float, default=5, help='当前请求（GB）')
    sp_retry.add_argument('--num-segments', type=int, default=10, help='段数')
    sp_retry.add_argument('--avg-frag', type=float, default=0.2, help='平均碎片率')

    sp_budget = subparsers.add_parser('membudget', help='显存预算')
    sp_budget.add_argument('--params-billion', type=float, default=7, help='参数数（B）')
    sp_budget.add_argument('--layers', type=int, default=32, help='层数')
    sp_budget.add_argument('--seq', type=int, default=2048, help='序列长度')
    sp_budget.add_argument('--dmodel', type=int, default=4096, help='隐藏维')
    sp_budget.add_argument('--dtype', default='bf16', help='数据类型（bf16/fp32）')
    sp_budget.add_argument('--optimizer', default='adam', help='优化器（adam/sgd）')
    sp_budget.add_argument('--checkpointing', action='store_true', help='启用梯度检查点')
    sp_budget.add_argument('--gpu', default='h100-sxm', help='GPU 型号')

    sp_sweep = subparsers.add_parser('budget-sweep', help='参数空间扫描')
    sp_sweep.add_argument('--gpu', default='h100-sxm', help='GPU 型号')

    sp_oom = subparsers.add_parser('oom-pattern', help='解析 OOM 报错')
    sp_oom.add_argument('--error-msg', help='错误消息（可选，默认示例）')

    args = parser.parse_args()

    if args.cmd == 'retry':
        cmd_retry(args)
    elif args.cmd == 'membudget':
        cmd_membudget(args)
    elif args.cmd == 'budget-sweep':
        cmd_budget_sweep(args)
    elif args.cmd == 'oom-pattern':
        cmd_oom_pattern(args)


if __name__ == '__main__':
    main()
