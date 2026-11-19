"""Triton Autotune 与持久化 kernel（CPU 估算）。对应 GPU-07 第 3-4 章。

核心指标：
  1. 瓦片大小与共享内存：大瓦片提升算力利用率，但共享内存容量有限
  2. 配置过滤：autotune 必须过滤掉超出 SMEM 限制的配置
  3. 持久化策略：一个线程块处理多个 k 瓦片，减少线程块启动开销
  4. Reset-to-zero 成本：原地更新 kernel 需要额外的内存屏障

子命令：
  smem-usage     瓦片大小 -> 共享内存用量
  persistent-grid 持久化 kernel 的网格大小和 k 循环次数
  autotune-sweep 完整自动调优过程（理想情况）

用法：
  python3 sim_autotune.py smem-usage --block-m 128 --block-n 128 --block-k 32 --dtype f32
  python3 sim_autotune.py persistent-grid --m 4096 --n 4096 --k 4096 --block-m 64 --block-n 64
  python3 sim_autotune.py autotune-sweep --gpu h100-sxm --m 512 --n 512 --k 512
"""
import argparse
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get

# 共享内存容量（H100 SXM）
SMEM_PER_BLOCK_BYTES = 96 * 1024  # 96 KB 默认，支持扩展到 192 KB


def smem_usage(block_m: int, block_n: int, block_k: int, dtype: str) -> Dict:
    """
    计算矩阵乘 kernel 的共享内存用量。

    双缓冲：
      - A 矩阵块 (BM, BK): 2 × BM × BK × dtype_bytes
      - B 矩阵块 (BK, BN): 2 × BK × BN × dtype_bytes

    来源：Triton GEMM tutorial，https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
    """
    dtype_bytes = {'f32': 4, 'f16': 2, 'bf16': 2, 'f64': 8}[dtype]

    # 双缓冲 A 和 B
    smem_a = 2 * block_m * block_k * dtype_bytes
    smem_b = 2 * block_k * block_n * dtype_bytes
    total_smem = smem_a + smem_b

    # 额外 overhead: 同步、工作队列等（估算 2KB）
    overhead = 2 * 1024
    total_with_overhead = total_smem + overhead

    return {
        'block_m': block_m,
        'block_n': block_n,
        'block_k': block_k,
        'dtype': dtype,
        'dtype_bytes': dtype_bytes,
        'smem_a_bytes': smem_a,
        'smem_b_bytes': smem_b,
        'smem_subtotal': total_smem,
        'overhead_bytes': overhead,
        'total_bytes': total_with_overhead,
        'total_mb': total_with_overhead / (1024 * 1024),
        'fits_default_96kb': total_with_overhead <= 96 * 1024,
        'fits_max_192kb': total_with_overhead <= 192 * 1024,
    }


def persistent_grid(m: int, n: int, k: int, block_m: int, block_n: int) -> Dict:
    """
    持久化 kernel：一个线程块循环处理多个 k 瓦片，而不是启动多个块。

    传统网格：Z 维（k 维）启动多个块，同步开销大。
    持久化：Z 维只启动 1，每个块循环 ceil(K / block_k) 次。

    来源：Triton 官方持久化矩阵乘教程。
    """
    tiles_m = (m + block_m - 1) // block_m
    tiles_n = (n + block_n - 1) // block_n

    # 传统方式：grid = (tiles_m, tiles_n, 1)
    traditional_blocks = tiles_m * tiles_n

    # 持久化方式：同样的块数，但循环处理所有 k
    persistent_blocks = tiles_m * tiles_n  # 块数相同
    # 每个块处理的 k 迭代数（后续会根据 block_k 计算）

    return {
        'm': m,
        'n': n,
        'k': k,
        'block_m': block_m,
        'block_n': block_n,
        'tiles_m': tiles_m,
        'tiles_n': tiles_n,
        'traditional_grid_blocks': traditional_blocks,
        'persistent_grid_blocks': persistent_blocks,
        'grid_reduction_ratio': traditional_blocks / persistent_blocks if persistent_blocks > 0 else 1.0,
    }


def autotune_sweep(gpu_name: str, m: int, n: int, k: int, max_configs: int = 24) -> Dict:
    """
    Triton autotune 的完整过程估算。

    参考 Triton 官方 persistent GEMM 教程的 24 个配置。在 FP16 下，
    最大的配置需要 384 KB 共享内存，RTX 4090 (96 KB) 只能运行其中 6 个。

    来源：https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
    """
    gpu = get(gpu_name)

    # 生成候选配置：block_m, block_n, num_stages, num_warps
    block_ms = [64, 128, 256]
    block_ns = [64, 128, 256]
    num_stages_list = [3, 4, 5]
    num_warps_list = [4, 8]

    all_configs = []
    for bm in block_ms:
        for bn in block_ns:
            for stages in num_stages_list:
                for warps in num_warps_list:
                    all_configs.append({
                        'block_m': bm,
                        'block_n': bn,
                        'block_k': 16,  # 固定为 16
                        'num_stages': stages,
                        'num_warps': warps,
                    })

    # 过滤：计算每个配置的 SMEM 用量（简化：只计算双缓冲部分）
    # num_stages 个 buffer，每个 stage 有 A 和 B
    dtype_bytes = 2  # FP16
    valid_configs = []

    for cfg in all_configs:
        # 估算共享内存（保守估计）
        smem_per_stage = (cfg['block_m'] * cfg['block_k'] +
                         cfg['block_k'] * cfg['block_n']) * dtype_bytes
        total_smem = cfg['num_stages'] * smem_per_stage + 2 * 1024  # overhead

        if total_smem <= 96 * 1024:  # 默认 96 KB
            valid_configs.append({**cfg, 'smem_bytes': total_smem})

    # 只保留前 max_configs 个
    valid_configs = valid_configs[:max_configs]

    return {
        'gpu': gpu_name,
        'm': m, 'n': n, 'k': k,
        'all_candidate_configs': len(all_configs),
        'valid_configs_count': len(valid_configs),
        'filter_ratio': len(valid_configs) / len(all_configs) if all_configs else 0,
        'sample_valid_configs': valid_configs[:5],
        'note': 'FP16 下，384KB 配置在 96KB SMEM 的卡上被过滤；'
                '持久化教程 24 个配置中有 6 个能在 RTX 4090 上运行。',
    }


def reset_to_zero_cost(m: int, n: int, threads_per_block: int = 128) -> Dict:
    """
    原地更新（in-place update）kernel 的 reset-to-zero 成本。

    Triton autotune 中，reset_to_zero=True 会在每个线程块启动时清零输出矩阵，
    防止累积错误。成本是额外的内存写操作。

    来源：Triton autotune 文档。
    """
    # 假设矩阵是 FP32 (4 字节)
    dtype_bytes = 4
    total_elements = m * n
    total_bytes = total_elements * dtype_bytes

    # 清零成本：每个线程块清零一部分
    num_blocks = (total_elements + threads_per_block - 1) // threads_per_block
    bytes_per_block = (threads_per_block * dtype_bytes)

    # 带宽成本估算：H100 HBM 3.35 TB/s
    h100_bw_tbs = 3.35
    h100_bw_gbs = h100_bw_tbs * 1000
    time_us = total_bytes / (h100_bw_gbs * 1e9) * 1e6

    return {
        'm': m,
        'n': n,
        'total_elements': total_elements,
        'total_bytes': total_bytes,
        'dtype_bytes': dtype_bytes,
        'threads_per_block': threads_per_block,
        'num_blocks_needed': num_blocks,
        'time_us_h100': time_us,
        'estimated_overhead_pct': (time_us / 100.0) * 100,  # 假设总时间 100 μs（示例）
    }


def main():
    parser = argparse.ArgumentParser(description='GPU-07 Autotune 与持久化分析')
    subparsers = parser.add_subparsers(dest='command')

    # smem-usage 子命令
    p_smem = subparsers.add_parser('smem-usage', help='共享内存用量')
    p_smem.add_argument('--block-m', type=int, default=128)
    p_smem.add_argument('--block-n', type=int, default=128)
    p_smem.add_argument('--block-k', type=int, default=32)
    p_smem.add_argument('--dtype', type=str, default='f32')

    # persistent-grid 子命令
    p_persist = subparsers.add_parser('persistent-grid', help='持久化网格')
    p_persist.add_argument('--m', type=int, default=4096)
    p_persist.add_argument('--n', type=int, default=4096)
    p_persist.add_argument('--k', type=int, default=4096)
    p_persist.add_argument('--block-m', type=int, default=64)
    p_persist.add_argument('--block-n', type=int, default=64)

    # autotune-sweep 子命令
    p_autotune = subparsers.add_parser('autotune-sweep', help='自动调优过程')
    p_autotune.add_argument('--gpu', type=str, default='h100-sxm')
    p_autotune.add_argument('--m', type=int, default=512)
    p_autotune.add_argument('--n', type=int, default=512)
    p_autotune.add_argument('--k', type=int, default=512)

    # reset-to-zero 子命令
    p_reset = subparsers.add_parser('reset-to-zero', help='原地更新成本')
    p_reset.add_argument('--m', type=int, default=4096)
    p_reset.add_argument('--n', type=int, default=4096)

    args = parser.parse_args()

    if args.command == 'smem-usage':
        result = smem_usage(args.block_m, args.block_n, args.block_k, args.dtype)
        print(f'共享内存用量分析')
        print(f'  block_m            {result["block_m"]}')
        print(f'  block_n            {result["block_n"]}')
        print(f'  block_k            {result["block_k"]}')
        print(f'  dtype              {result["dtype"]}')
        print(f'  smem_a_bytes       {result["smem_a_bytes"]}')
        print(f'  smem_b_bytes       {result["smem_b_bytes"]}')
        print(f'  smem_subtotal      {result["smem_subtotal"]}')
        print(f'  overhead_bytes     {result["overhead_bytes"]}')
        print(f'  total_bytes        {result["total_bytes"]}')
        print(f'  total_mb           {result["total_mb"]:.2f}')
        print(f'  fits_default_96kb  {result["fits_default_96kb"]}')
        print(f'  fits_max_192kb     {result["fits_max_192kb"]}')

    elif args.command == 'persistent-grid':
        result = persistent_grid(args.m, args.n, args.k, args.block_m, args.block_n)
        print(f'持久化 Kernel 网格分析 ({args.m}×{args.n}×{args.k})')
        print(f'  tiles_m                    {result["tiles_m"]}')
        print(f'  tiles_n                    {result["tiles_n"]}')
        print(f'  traditional_grid_blocks    {result["traditional_grid_blocks"]}')
        print(f'  persistent_grid_blocks     {result["persistent_grid_blocks"]}')
        print(f'  grid_reduction_ratio       {result["grid_reduction_ratio"]:.2f}x')

    elif args.command == 'autotune-sweep':
        result = autotune_sweep(args.gpu, args.m, args.n, args.k)
        print(f'Autotune 配置扫描 ({args.gpu}, {args.m}×{args.n}×{args.k})')
        print(f'  all_candidate_configs  {result["all_candidate_configs"]}')
        print(f'  valid_configs_count    {result["valid_configs_count"]}')
        print(f'  filter_ratio           {result["filter_ratio"]:.1%}')
        print(f'  note                   {result["note"]}')
        if result['sample_valid_configs']:
            print(f'  sample_valid_configs:')
            for cfg in result['sample_valid_configs']:
                print(f'    BM={cfg["block_m"]:3d} BN={cfg["block_n"]:3d} '
                      f'stages={cfg["num_stages"]} warps={cfg["num_warps"]} '
                      f'smem={cfg["smem_bytes"]/1024:.0f}KB')

    elif args.command == 'reset-to-zero':
        result = reset_to_zero_cost(args.m, args.n)
        print(f'Reset-to-zero 成本分析 ({args.m}×{args.n})')
        print(f'  total_bytes           {result["total_bytes"]}')
        print(f'  num_blocks_needed     {result["num_blocks_needed"]}')
        print(f'  time_us_h100          {result["time_us_h100"]:.3f}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
