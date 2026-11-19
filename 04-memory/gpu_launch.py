"""GPU 内核启动辅助工具：Nsight Compute 分析与性能量化。

本模块提供以下功能：
1. CUDA 核函数的配置建议（grid size、block size、shared memory）
2. 性能期望值的计算（基于论文与官方规格）
3. Nsight Compute 分析指标的解释

本仓库未在真卡上复测。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402


def recommend_block_config(kernel_type, problem_size, gpu_name='h100-sxm'):
    """根据内核类型和问题规模推荐 block 配置。

    Args:
        kernel_type: 'memory', 'compute', 'reduce'
        problem_size: 数据量或元素数
        gpu_name: GPU 型号

    Returns:
        dict: {'block_size': int, 'grid_size': int, 'shared_mem_bytes': int, 'notes': str}
    """
    gpu = get(gpu_name)

    if kernel_type == 'memory':
        # 访存优化的内核：优先最大化占用率，通常 block_size = 256
        block_size = 256
        grid_size = (problem_size + block_size - 1) // block_size
        shared_mem = 0  # 或者根据模式定制
        notes = 'memory 内核：block_size=256 通常最优；shared_mem 取决于具体优化模式'

    elif kernel_type == 'compute':
        # 计算优化的内核：平衡占用率与缓存利用
        block_size = 128
        grid_size = (problem_size + block_size - 1) // block_size
        shared_mem = 0
        notes = 'compute 内核：block_size=128；充分提供 shared memory（若需要）'

    elif kernel_type == 'reduce':
        # 归约内核：block_size 影响树高，通常 256-512
        block_size = 256
        grid_size = (problem_size + block_size - 1) // block_size
        shared_mem = block_size * 4  # float 预留
        notes = f'reduce 内核：block_size={block_size}；shared_mem={shared_mem} bytes'

    else:
        raise ValueError(f'未知内核类型 {kernel_type}')

    return {
        'block_size': block_size,
        'grid_size': grid_size,
        'shared_mem_bytes': shared_mem,
        'notes': notes
    }


def sector_efficiency(num_sectors_actual, num_requests, sector_size=32, request_bytes=32):
    """计算 sector 利用率。

    Sector Efficiency = (Request Size) / (num_sectors * sector_size) * 100%
    """
    useful_bytes = num_requests * request_bytes
    transferred_bytes = num_sectors_actual * sector_size
    if transferred_bytes == 0:
        return 0.0
    return (useful_bytes / transferred_bytes) * 100


def estimate_memory_time(num_bytes, bandwidth_tbs, dtype='bf16'):
    """估计内存操作的时间（微秒）。

    Args:
        num_bytes: 字节数
        bandwidth_tbs: 带宽，TB/s
        dtype: 数据类型（影响计算强度）

    Returns:
        float: 时间（微秒）
    """
    time_us = num_bytes / (bandwidth_tbs * 1e12) * 1e6
    return time_us


def estimate_flops_time(flops, peak_tflops):
    """估计计算操作的时间（微秒）。

    Args:
        flops: 浮点运算数
        peak_tflops: 峰值算力，TFLOPS

    Returns:
        float: 时间（微秒）
    """
    time_us = flops / (peak_tflops * 1e12) * 1e6
    return time_us


def summary(gpu_name='h100-sxm'):
    """打印 GPU 规格和推荐配置总结。"""
    gpu = get(gpu_name)

    print(f'{"GPU 规格":60} {gpu_name}')
    print('-' * 80)
    print(f'{"架构":<30} {gpu.arch}')
    print(f'{"SM 数":<30} {gpu.sm}')
    print(f'{"BF16 峰值":<30} {gpu.bf16_tflops:>8.0f} TFLOPS')
    print(f'{"HBM 带宽":<30} {gpu.hbm_tbs:>8.2f} TB/s')
    print(f'{"NVLink 双向带宽":<30} {gpu.nvlink_gbs:>8.0f} GB/s')
    print(f'{"PCIe 双向带宽":<30} {gpu.pcie_gbs:>8.0f} GB/s')
    print()

    print(f'{"推荐启动配置":60}')
    print('-' * 80)
    for ktype in ['memory', 'compute', 'reduce']:
        cfg = recommend_block_config(ktype, 1000000, gpu_name)
        print(f'{ktype:15} block={cfg["block_size"]:>3}  grid={cfg["grid_size"]:>6}  '
              f'shared_mem={cfg["shared_mem_bytes"]:>6}B')
    print()

    print(f'{"带宽利用率目标":60}')
    print('-' * 80)
    print(f'{"连续访问 (stride=1)":<30} 100% (理想)')
    print(f'{"步长 2":<30} ~50%')
    print(f'{"步长 8":<30} ~12.5%')
    print()

    print(f'{"Bank 冲突目标（共享内存）":60}')
    print('-' * 80)
    print(f'{"无冲突 (stride=1)":<30} 1x 延迟')
    print(f'{"4 路冲突":<30} 4x 延迟')
    print(f'{"32 路冲突（最坏）":<30} 32x 延迟（应避免）')


if __name__ == '__main__':
    summary()
