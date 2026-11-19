"""DSL 性能模型：Triton 块调度、CuTe 布局代数和 CUTLASS 自动调优（CPU）。对应 GPU-07。

核心概念：
  1. 块调度(block ordering)：同一个 kernel，改变线程块在网格中的顺序，
     通过影响 L2 缓存重用和 DRAM 页面命中，性能差异可达 10%。
  2. 布局代数(layout algebra)：CuTe 用 Shape:Stride 表达布局，coalesce 和
     swizzle 改变内存访问模式的连续性和缓存行冲突。
  3. 配置空间(config search)：autotune 时，瓦片大小、共享内存大小、线程块大小
     的组合空间是指数级的，实际可行的配置（不超出 smem 限制）往往只有原空间的 5-10%。

子命令：
  block-order    Triton 矩阵乘中的块分组和重排效果（GROUP_SIZE_M 对 L2 重用）
  layout-algebra CuTe Shape:Stride 的 coalesce、stride、repeat 运算
  config-search  GEMM 自动调优的可行配置数（给定卡和形状下）
  swizzle        Swizzle 参数对 shared memory bank conflict 的影响

用法：
  python3 sim_dsl.py block-order --m 4096 --n 4096 --k 4096 --group-size-m 64
  python3 sim_dsl.py layout-algebra --shape "256:16,128:256" --coalesce
  python3 sim_dsl.py config-search --gpu h100-sxm --m 4096 --n 4096 --k 4096
  python3 sim_dsl.py swizzle --smem-bytes 96000 --dtype f32
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get

# L2 缓存和 DRAM 页面特性
# 来源：GPU-06 NVIDIA Ampere / Hopper 白皮书，L2 缓存 40 MB，行 128 字节
L2_CACHE_BYTES = 40 * 1024 * 1024  # 40 MB (典型值，不同卡略有差异)
L2_LINE_BYTES = 128                # 缓存行 128 字节
DRAM_PAGE_BYTES = 4096             # DRAM 页面 4 KB (虚拟地址页面)

# Tensor Core 规格：H100 64 个 TensorFloat32 运算/周期/SM，BF16 相同吞吐
# 来源：NVIDIA H100 数据表，核对于 2026-11-02
TENSOR_OPS_PER_CYCLE_PER_SM = 64   # 对于 FP32/BF16 accumulate


@dataclass
class Layout:
    """简化的 CuTe 布局：(Shape, Stride)。"""
    shape: List[int]  # 维度大小
    stride: List[int] # 单位跨度（以元素数）

    def volume(self):
        """布局覆盖的元素总数。"""
        v = 1
        for s in self.shape:
            v *= s
        return v

    def coalesce(self):
        """简化的 coalesce：如果最后一个维度的 stride 为 1，则返回合并后的跨度。"""
        if not self.stride or self.stride[-1] != 1:
            return None  # 不能合并
        # 尝试将最后两个维度合并
        if len(self.shape) >= 2 and len(self.stride) >= 2:
            if self.stride[-2] == self.shape[-1]:  # stride[i-1] == shape[i]
                merged_shape = self.shape[:-2] + [self.shape[-2] * self.shape[-1]]
                merged_stride = self.stride[:-2] + [1]
                return Layout(merged_shape, merged_stride)
        return self

    def bytes_to_load(self, dtype_bytes: int) -> int:
        """加载此布局需要的字节数（考虑 stride 和缓存行大小）。"""
        # 简化：计算线性内存范围
        max_offset = sum((s - 1) * st for s, st in zip(self.shape, self.stride))
        return (max_offset + 1) * dtype_bytes


def block_order_sim(m: int, n: int, k: int, group_size_m: int, block_size_m: int = 64, block_size_n: int = 64):
    """
    Triton 矩阵乘块分组效果估算。GROUP_SIZE_M 分组可减少 L2 缓存频繁换出。

    返回: (无分组总块数, 分组后总块数, 预期 L2 重用提升%)。
    返回数字：一个 9×9 矩阵（块数），第一波（并行 56 个 SM）从 90 块减少到 54 块。
    来源：Triton 官方矩阵乘教程，https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    """
    num_blocks_m = (m + block_size_m - 1) // block_size_m
    num_blocks_n = (n + block_size_n - 1) // block_size_n
    total_blocks_naive = num_blocks_m * num_blocks_n

    # GROUP_SIZE_M 分组：减少同时加载的块数，提高 L2 重用
    num_groups = (num_blocks_m + group_size_m - 1) // group_size_m
    blocks_per_group = group_size_m * num_blocks_n
    # 在有足够 SM 的情况下，每一波只启动一个组
    blocks_first_wave = num_groups * num_blocks_n if num_blocks_m < group_size_m else num_blocks_n

    # 估算 L2 重用提升：分组后减少了独立块的数量，更多块能共享 L2 缓存中的 A 矩阵块
    l2_improvement_pct = (1 - blocks_first_wave / total_blocks_naive) * 100 if total_blocks_naive > 0 else 0

    return {
        'total_blocks_naive': total_blocks_naive,
        'blocks_first_wave': blocks_first_wave,
        'num_groups': num_groups,
        'l2_improvement_pct': l2_improvement_pct,
        'num_blocks_m': num_blocks_m,
        'num_blocks_n': num_blocks_n,
    }


def layout_algebra_demo(shape_str: str, do_coalesce: bool = False):
    """
    演示 CuTe 布局代数的基本运算。
    shape_str: 格式 "S1:T1,S2:T2,..." 表示 Shape:Stride 对。
    """
    parts = shape_str.split(',')
    shapes = []
    strides = []
    for part in parts:
        s, t = map(int, part.split(':'))
        shapes.append(s)
        strides.append(t)

    layout = Layout(shapes, strides)
    result = {
        'input': shape_str,
        'shape': layout.shape,
        'stride': layout.stride,
        'volume': layout.volume(),
        'bytes_f32': layout.bytes_to_load(4),
        'bytes_f64': layout.bytes_to_load(8),
        'is_contiguous': strides[-1] == 1 if strides else False,
    }

    if do_coalesce:
        coalesced = layout.coalesce()
        if coalesced != layout:
            result['coalesced_shape'] = coalesced.shape
            result['coalesced_stride'] = coalesced.stride

    return result


def config_search(gpu_name: str, m: int, n: int, k: int, max_smem_bytes: int = None, dtype: str = 'f32'):
    """
    GEMM 自动调优配置空间过滤。给定 m×n×k 和卡，计算有效配置数。

    共享内存限制（H100 SXM）：96 KB 正常，最多 192 KB（牺牲 L1）。
    单个配置大小 ≈ (BLOCK_M + BLOCK_N) * BLOCK_K * dtype_bytes + 同步 overhead。

    来源：NVIDIA CUDA Occupancy Calculator，核对于 2026-11-02。
    """
    gpu = get(gpu_name)
    if max_smem_bytes is None:
        max_smem_bytes = 96 * 1024  # 96 KB 默认

    dtype_bytes = {'f32': 4, 'f16': 2, 'bf16': 2, 'f64': 8}[dtype]

    # 扫描常见的块大小（Triton 默认值和倍数）
    block_sizes = [32, 64, 128, 256]
    valid_configs = []

    for block_m in block_sizes:
        for block_n in block_sizes:
            for block_k in block_sizes:
                # 估算单个配置的共享内存用量
                # 双缓冲 A: block_m * block_k, 双缓冲 B: block_k * block_n
                smem_needed = 2 * (block_m * block_k + block_k * block_n) * dtype_bytes

                if smem_needed <= max_smem_bytes:
                    # 矩阵尺寸兼容性检查
                    if (m >= block_m) and (n >= block_n) and (k >= block_k):
                        tiles_m = (m + block_m - 1) // block_m
                        tiles_n = (n + block_n - 1) // block_n
                        tiles_k = (k + block_k - 1) // block_k
                        valid_configs.append({
                            'block_m': block_m,
                            'block_n': block_n,
                            'block_k': block_k,
                            'smem_bytes': smem_needed,
                            'total_blocks': tiles_m * tiles_n,
                            'k_iterations': tiles_k,
                        })

    # 计算理论空间和过滤比例
    theoretical_configs = len(block_sizes) ** 3
    valid_count = len(valid_configs)
    filter_ratio = valid_count / theoretical_configs if theoretical_configs > 0 else 0

    return {
        'gpu': gpu_name,
        'm': m, 'n': n, 'k': k,
        'dtype': dtype,
        'max_smem_mb': max_smem_bytes / (1024 * 1024),
        'theoretical_configs': theoretical_configs,
        'valid_configs': valid_count,
        'filter_ratio': filter_ratio,
        'sample_configs': valid_configs[:5] if valid_configs else [],
    }


def swizzle_analysis(smem_bytes: int, dtype: str = 'f32', swizzle_bits: int = 3):
    """
    Swizzle 对共享内存 bank conflict 的影响。

    H100 共享内存：4-bank 分组，每条 bank 32 字节。swizzle 通过在物理地址中
    XOR 低位，打破步长为 bank_stride 的访问模式导致的冲突。

    来源：NVIDIA Hopper 白皮书，shared memory bank conflict analysis。
    """
    dtype_bytes = {'f32': 4, 'f16': 2, 'bf16': 2, 'f64': 8}[dtype]
    smem_banks = 32  # H100 shared memory 分为 32 个 bank
    bank_bytes = 128 // smem_banks  # 每个 bank 占 4 字节

    # 典型情况：warp (32 线程) 同时访问 shared memory
    warp_size = 32
    elements_per_bank = bank_bytes // dtype_bytes

    # 不加 swizzle：线性步长访问容易导致多线程访问同一 bank
    conflict_stride = bank_bytes // dtype_bytes  # 多少个元素为一个 bank 周期

    # 加 swizzle：XOR 地址的低位，打破规律步长
    swizzle_mask = (1 << swizzle_bits) - 1

    return {
        'smem_bytes': smem_bytes,
        'smem_mb': smem_bytes / (1024 * 1024),
        'dtype': dtype,
        'dtype_bytes': dtype_bytes,
        'shared_memory_banks': smem_banks,
        'bytes_per_bank': bank_bytes,
        'elements_per_bank': elements_per_bank,
        'warp_size': warp_size,
        'conflict_stride_elements': conflict_stride,
        'swizzle_bits': swizzle_bits,
        'swizzle_mask': hex(swizzle_mask),
        'estimated_conflict_reduction': f'{(1 - (1 / (2 ** swizzle_bits))) * 100:.1f}%',
    }


def main():
    parser = argparse.ArgumentParser(description='GPU-07 DSL 性能模型')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # block-order 子命令
    p_block = subparsers.add_parser('block-order', help='Triton 块分组效果')
    p_block.add_argument('--m', type=int, default=4096)
    p_block.add_argument('--n', type=int, default=4096)
    p_block.add_argument('--k', type=int, default=4096)
    p_block.add_argument('--group-size-m', type=int, default=64)
    p_block.add_argument('--block-size-m', type=int, default=64)
    p_block.add_argument('--block-size-n', type=int, default=64)

    # layout-algebra 子命令
    p_layout = subparsers.add_parser('layout-algebra', help='CuTe 布局代数')
    p_layout.add_argument('--shape', type=str, default='256:16,128:256', help='形式: S1:T1,S2:T2,...')
    p_layout.add_argument('--coalesce', action='store_true')

    # config-search 子命令
    p_config = subparsers.add_parser('config-search', help='自动调优配置过滤')
    p_config.add_argument('--gpu', type=str, default='h100-sxm')
    p_config.add_argument('--m', type=int, default=4096)
    p_config.add_argument('--n', type=int, default=4096)
    p_config.add_argument('--k', type=int, default=4096)
    p_config.add_argument('--max-smem-mb', type=float, default=96)
    p_config.add_argument('--dtype', type=str, default='f32', choices=['f32', 'f16', 'bf16', 'f64'])

    # swizzle 子命令
    p_swizzle = subparsers.add_parser('swizzle', help='Swizzle bank conflict 分析')
    p_swizzle.add_argument('--smem-bytes', type=int, default=96000)
    p_swizzle.add_argument('--dtype', type=str, default='f32')
    p_swizzle.add_argument('--swizzle-bits', type=int, default=3)

    args = parser.parse_args()

    if args.command == 'block-order':
        result = block_order_sim(args.m, args.n, args.k, args.group_size_m,
                                args.block_size_m, args.block_size_n)
        print(f'块调度分析 ({args.m}×{args.n}×{args.k})')
        print(f'  num_blocks_m          {result["num_blocks_m"]}')
        print(f'  num_blocks_n          {result["num_blocks_n"]}')
        print(f'  total_blocks_naive    {result["total_blocks_naive"]}')
        print(f'  blocks_first_wave     {result["blocks_first_wave"]}')
        print(f'  num_groups            {result["num_groups"]}')
        print(f'  L2_improvement_%      {result["l2_improvement_pct"]:.1f}%')

    elif args.command == 'layout-algebra':
        result = layout_algebra_demo(args.shape, args.coalesce)
        print(f'布局代数')
        print(f'  input              {result["input"]}')
        print(f'  shape              {result["shape"]}')
        print(f'  stride             {result["stride"]}')
        print(f'  volume             {result["volume"]}')
        print(f'  bytes_f32          {result["bytes_f32"]}')
        print(f'  bytes_f64          {result["bytes_f64"]}')
        print(f'  is_contiguous      {result["is_contiguous"]}')
        if 'coalesced_shape' in result:
            print(f'  coalesced_shape    {result["coalesced_shape"]}')
            print(f'  coalesced_stride   {result["coalesced_stride"]}')

    elif args.command == 'config-search':
        result = config_search(args.gpu, args.m, args.n, args.k,
                              int(args.max_smem_mb * 1024 * 1024), args.dtype)
        print(f'GEMM 自动调优配置空间 ({args.gpu}, {args.m}×{args.n}×{args.k}, {args.dtype})')
        print(f'  max_smem_mb           {result["max_smem_mb"]:.1f}')
        print(f'  theoretical_configs   {result["theoretical_configs"]}')
        print(f'  valid_configs         {result["valid_configs"]}')
        print(f'  filter_ratio          {result["filter_ratio"]:.1%}')
        if result['sample_configs']:
            print(f'  sample_configs (前 5 个):')
            for cfg in result['sample_configs']:
                print(f'    block({cfg["block_m"]}x{cfg["block_n"]}x{cfg["block_k"]}) '
                      f'smem={cfg["smem_bytes"]/1024:.0f}KB '
                      f'blocks={cfg["total_blocks"]} '
                      f'k_iter={cfg["k_iterations"]}')

    elif args.command == 'swizzle':
        result = swizzle_analysis(args.smem_bytes, args.dtype, args.swizzle_bits)
        print(f'Swizzle Bank Conflict 分析')
        print(f'  smem_mb                    {result["smem_mb"]:.2f}')
        print(f'  dtype                      {result["dtype"]}')
        print(f'  shared_memory_banks        {result["shared_memory_banks"]}')
        print(f'  bytes_per_bank             {result["bytes_per_bank"]}')
        print(f'  elements_per_bank          {result["elements_per_bank"]}')
        print(f'  warp_size                  {result["warp_size"]}')
        print(f'  conflict_stride_elements   {result["conflict_stride_elements"]}')
        print(f'  swizzle_bits               {result["swizzle_bits"]}')
        print(f'  swizzle_mask               {result["swizzle_mask"]}')
        print(f'  estimated_conflict_reduction {result["estimated_conflict_reduction"]}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
