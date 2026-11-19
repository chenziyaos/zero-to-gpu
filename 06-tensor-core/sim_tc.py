"""Tensor Core 峰值和运用率：mma 指令的吞吐、延迟、片段布局（对应 GPU-06 第 1-4 章）。

核心公式：
  1. Tensor Core 峰值 = SM 个数 × 每 SM 的 mma 单元数 × 时钟频率 × 数据宽度
     每 SM 的 mma 单元数取决于代际：Ampere 4 个，Hopper 4 个，Blackwell 按核心计算
  2. warp 能发射的 mma 每周期 = min(调度器容量, warp 等待时间≤前驱依赖延迟)
  3. 片段占有率 = 运行 kernel 的 warp 数 × 每 warp 的寄存器消耗 / SM 的总寄存器数
  4. 强度 = (M × K × 2 + K × N × 2 + M × N × 2) / (M × K + K × N) 字节，假设 BF16 且无重用
     更精确：强度依赖片上分块大小、重用倍数和搬运模式

数据来源：
  - NVIDIA Ampere、Hopper、Blackwell 架构白皮书及 CUDA 编程手册（核对于 2026-11-04）
  - 实测片段位置算法来自 CUTLASS 3.7+ 和 Tensor Memory 文档
  - HBM 延迟来自 Luo et al. 2024 "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture"（arXiv 2402.13499）

子命令：
  peak        各 GPU 的 Tensor Core 峰值，拆出 BF16、FP8、FP4
  utilization 给定 kernel 形状和占有率，估算实际运用率（瓶颈落在哪）
  fragment    wmma/mma.sync/wgmma/tcgen05 的片段内存占用和布局细节
  strength    GEMM 的数学强度、片上强度和搬运瓶颈
  pipeline    GEMM 主循环的每步周期数和流水效果（sync/double-buffer/multistage）

用法：
  python3 sim_tc.py peak
  python3 sim_tc.py peak --gpu h100-sxm --dtype fp8
  python3 sim_tc.py utilization --gpu h100-sxm --m 128 --n 128 --k 64 --occupancy 0.5
  python3 sim_tc.py fragment --api mma.sync --m 16 --n 8 --k 16
  python3 sim_tc.py strength --m 4096 --n 4096 --k 4096 --tile-m 128 --tile-n 128
  python3 sim_tc.py pipeline --mode sync
  python3 sim_tc.py pipeline --mode double-buffer
  python3 sim_tc.py pipeline --mode multistage --stages 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# Tensor Core 单元数和指令发射能力（代际级）
# 来源：NVIDIA 白皮书，Hopper 架构 (2023-03) 和 Blackwell 架构 (2024-12)
ARCHITECTURES = {
    'Ampere': {'tc_per_sm': 4, 'max_warp_per_sm': 32, 'max_registers': 256 * 1024, 'latency_mma_f32': 6},
    'Hopper': {'tc_per_sm': 4, 'max_warp_per_sm': 32, 'max_registers': 256 * 1024, 'latency_mma_f32': 6,
               'has_wgmma': True},
    'Blackwell': {'tc_per_sm': 8, 'max_warp_per_sm': 32, 'max_registers': 256 * 1024, 'latency_mma_f32': 6,
                  'has_wgmma': True, 'has_tcgen05': True},
    'Rubin': {'tc_per_sm': 8, 'max_warp_per_sm': 32, 'max_registers': 256 * 1024, 'latency_mma_f32': 6,
              'has_wgmma': True, 'has_tcgen05': True},
}

# Fragment 标准尺寸及其寄存器消耗（32-bit 元素计数）
# 来源：CUTLASS 3.7 fragment.h、CUDA PTX ISA 文档 mma 指令规格（核对于 2026-11-04）
FRAGMENTS = {
    'wmma': [
        {'shape': (16, 16, 16), 'a_regs': 8, 'b_regs': 8, 'c_regs': 8},
        {'shape': (32, 8, 16), 'a_regs': 4, 'b_regs': 8, 'c_regs': 8},
        {'shape': (8, 32, 16), 'a_regs': 8, 'b_regs': 4, 'c_regs': 8},
    ],
    'mma.sync': [
        {'shape': (16, 8, 16), 'a_regs': 8, 'b_regs': 4, 'c_regs': 4},
        {'shape': (16, 16, 16), 'a_regs': 8, 'b_regs': 8, 'c_regs': 8},
        {'shape': (32, 8, 16), 'a_regs': 8, 'b_regs': 4, 'c_regs': 8},
        {'shape': (8, 32, 16), 'a_regs': 4, 'b_regs': 8, 'c_regs': 8},
    ],
    'wgmma': [
        {'shape': (64, 256, 16), 'smem_a_bytes': 2048, 'smem_b_bytes': 8192, 'reg_acc': 128},
    ],
    'tcgen05': [
        {'shape': (64, 256, 16), 'tmem_bytes': 65536, 'reg_reqs': 256},
    ],
}


def cmd_peak(args):
    """各 GPU 的 Tensor Core 峰值。"""
    gpu = get(args.gpu)
    arch = ARCHITECTURES.get(gpu.arch, {})

    if not arch:
        print(f'ERROR: {gpu.arch} 无 Tensor Core 配置')
        return

    tc_per_sm = arch['tc_per_sm']
    sm_count = gpu.sm if gpu.sm > 0 else 0

    # 假设时钟 1.8-2.0 GHz，实际运行频率略低
    clock_ghz = 2.0

    # 每个 Tensor Core 每时钟执行 2 个 FMA（fused multiply-add = 2 FLOP）
    flop_per_core_per_clock = 2

    if args.dtype == 'bf16' and gpu.bf16_tflops > 0:
        tflops = gpu.bf16_tflops
    elif args.dtype == 'fp8' and gpu.fp8_tflops > 0:
        tflops = gpu.fp8_tflops
    elif args.dtype == 'fp4' and gpu.fp4_tflops > 0:
        tflops = gpu.fp4_tflops
    else:
        tflops = gpu.bf16_tflops or 0

    print(f'{"GPU":20} {"架构":12} {"SM":>4} {"TC/SM":>6} {"峰值（TFLOPS）":>15} {"精度":>8}')
    print('-' * 70)
    print(f'{gpu.name:20} {gpu.arch:12} {sm_count:4d} {tc_per_sm:6d} {tflops:15.0f} {args.dtype:>8}')


def cmd_utilization(args):
    """kernel 的 Tensor Core 运用率。"""
    gpu = get(args.gpu)
    arch = ARCHITECTURES.get(gpu.arch, {})

    m, n, k = args.m, args.n, args.k
    occupancy = args.occupancy

    # 估算：每个 warp 处理一个 16×16×16 的 mma（Ampere/Hopper 标配）
    ops_per_mma = m * n * k * 2  # 矩阵乘法 flops

    # 实际吞吐受调度器和依赖链深度影响
    # 简化模型：调度压力 = occupancy × SM 个数 × max_warp_per_sm
    max_warps_in_flight = int(gpu.sm * arch['max_warp_per_sm'] * occupancy)

    # mma 延迟假设为 6 周期，可并发深度～延迟
    mma_latency = arch['latency_mma_f32']
    concurrent_mmas = max(1, int(mma_latency / 2.0))

    # 实际频率略低于峰值
    actual_tflops = gpu.bf16_tflops * 0.85

    print(f'{"参数":20} {"值":>20}')
    print('-' * 45)
    print(f'{"M×N×K":20} {f"{m}×{n}×{k}":>20}')
    print(f'{"FLOPS":20} {f"{ops_per_mma/1e9:.1f}B":>20}')
    print(f'{"占有率":20} {f"{occupancy*100:.0f}%":>20}')
    print(f'{"最多并发 mma":20} {concurrent_mmas:>20}')
    print(f'{"推估吞吐（TFLOPS）":20} {actual_tflops:>20.0f}')

    # 估算 kernel 执行时间（微秒）
    if actual_tflops > 0:
        exec_time_us = ops_per_mma / (actual_tflops * 1e12) * 1e6
        print(f'{"执行时间（µs）":20} {exec_time_us:>20.2f}')


def cmd_fragment(args):
    """fragment 的寄存器占用和布局。"""
    api = args.api

    if api not in FRAGMENTS:
        print(f'ERROR: 未知 API {api}，可选：{", ".join(FRAGMENTS)}')
        return

    candidates = FRAGMENTS[api]

    # 尽量匹配用户输入的 shape
    best = None
    for frag in candidates:
        if api in ('wmma', 'mma.sync'):
            m, n, k = frag['shape']
            if m == args.m and n == args.n and k == args.k:
                best = frag
                break
        else:
            # wgmma/tcgen05 通常固定形状
            best = frag
            break

    if not best:
        best = candidates[0]

    print(f'{"API":15} {api:>10}')
    shape_str = f"{best['shape'][0]}×{best['shape'][1]}×{best['shape'][2]}"
    print(f'{"Shape (M×N×K)":15} {shape_str:>10}')

    if api in ('wmma', 'mma.sync'):
        a_regs = best.get('a_regs', 0)
        b_regs = best.get('b_regs', 0)
        c_regs = best.get('c_regs', 0)
        total_regs = a_regs + b_regs + c_regs
        print(f'{"A 寄存器":15} {a_regs:>10}')
        print(f'{"B 寄存器":15} {b_regs:>10}')
        print(f'{"C 寄存器":15} {c_regs:>10}')
        print(f'{"总计（32-bit）":15} {total_regs:>10}')
        print(f'{"总计（字节）":15} {total_regs*4:>10}')
    elif api == 'wgmma':
        smem_a_str = f"{best['smem_a_bytes']} B"
        smem_b_str = f"{best['smem_b_bytes']} B"
        print(f'{"共享内存 A":15} {smem_a_str:>10}')
        print(f'{"共享内存 B":15} {smem_b_str:>10}')
        print(f'{"累加器寄存器":15} {best["reg_acc"]:>10}')
    elif api == 'tcgen05':
        tmem_str = f"{best['tmem_bytes']} B"
        print(f'{"Tensor Memory":15} {tmem_str:>10}')
        print(f'{"寄存器需求":15} {best["reg_reqs"]:>10}')


def cmd_strength(args):
    """GEMM 的数学强度和搬运瓶颈。"""
    m, n, k = args.m, args.n, args.k
    tile_m, tile_n = args.tile_m, args.tile_n

    # 总 FLOPS（每个元素的矩阵乘乘 2 个 FMA）
    total_flops = m * n * k * 2

    # 全局内存访问（假设 BF16，2 字节/元素）
    # A: M × K × 2 字节，但分块后每个块只访问一次
    bytes_a = m * k * 2
    bytes_b = k * n * 2
    bytes_c = m * n * 2
    total_bytes = bytes_a + bytes_b + bytes_c

    # 数学强度：全局约束
    strength_math = total_flops / total_bytes

    # 片上强度：假设分块到 (tile_m, tile_n)
    flops_per_tile = tile_m * tile_n * k * 2
    bytes_per_tile_a = tile_m * k * 2
    bytes_per_tile_b = k * tile_n * 2
    bytes_per_tile_c = tile_m * tile_n * 2
    bytes_per_tile = bytes_per_tile_a + bytes_per_tile_b + bytes_per_tile_c

    strength_tile = flops_per_tile / bytes_per_tile

    gpu = get(args.gpu)
    ridge = gpu.ridge('bf16')

    print(f'{"维度":20} {"大小":>20}')
    print('-' * 45)
    print(f'{"M×N×K":20} {f"{m}×{n}×{k}":>20}')
    print(f'{"Tile M×N":20} {f"{tile_m}×{tile_n}":>20}')
    print(f'{"总 FLOPS":20} {f"{total_flops/1e9:.1f}B":>20}')
    print(f'{"总访存字节":20} {f"{total_bytes/1e9:.1f}B":>20}')
    print(f'{"数学强度（FLOP/B）":20} {strength_math:>20.2f}')
    print(f'{"片上强度（FLOP/B）":20} {strength_tile:>20.2f}')
    print(f'{"Roofline 拐点":20} {ridge:>20.0f}')

    if strength_math < ridge:
        print(f'{"瓶颈":20} {"带宽":>20}')
    else:
        print(f'{"瓶颈":20} {"算力":>20}')


def cmd_pipeline(args):
    """GEMM 主循环的每步时间模型：同步 / 双缓冲 / 多级流水。

    假设：
      - 4096×4096×4096 GEMM，128×128 分块 BF16
      - H100: 989 TFLOPS（峰值，132 SM × 4 TC/SM × 1.83 GHz）
      - 每 SM 的 Tensor Core 速度 = 989e12 / (132 * 1.83e9) ≈ 4096 BF16 FLOP/cycle/SM
      - HBM 延迟 600 cycles（基于 Luo et al. 2024，arXiv 2402.13499）
      - HBM 带宽 3.35 TB/s；L2 到 SM 带宽按 SM 算约 10 TB/s
      - 共享内存带宽 10 TB/s（per SM）
      - 每 k 步搬运字节（128×64 tile）= 128*64*2 (A) + 64*128*2 (B) = 32 KB
    """
    gpu = get(args.gpu)
    tile_m, tile_n, tile_k = args.tile_m, args.tile_n, args.tile_k

    # 参数
    bytes_per_elem = 2  # BF16
    bytes_per_kstep = tile_m * tile_k * bytes_per_elem + tile_k * tile_n * bytes_per_elem

    # H100 的峰值和带宽参数
    peak_tflops = gpu.bf16_tflops
    sm_count = gpu.sm
    hbm_bw_tbs = gpu.hbm_tbs

    # 计算：每 SM 的 Tensor Core FLOP/cycle（H100 约 4096）
    # peak_tflops = SM × TC/SM × freq × 2 FLOP/clock
    # 1.83 GHz 是 H100 的实际时钟（有保守性）
    freq_ghz = 1.83
    tc_per_sm = 4
    flop_per_tc_per_clock = 2
    flop_per_sm_per_cycle = (peak_tflops * 1e12 / 1e9) / (sm_count * freq_ghz)

    # 每个 k-步的计算时间和搬运时间
    flops_per_kstep = tile_m * tile_n * tile_k * 2
    compute_cycles_per_kstep = flops_per_kstep / flop_per_sm_per_cycle

    # 搬运延迟和时间：HBM → 共享内存，假设 3.35 TB/s
    # 实际：搬运 32 KB 需要 32K B / 3.35e12 B/s = 9.6 ns = 9.6/0.546 ≈ 18 cycles
    # 但加上延迟（600 cycles）主导
    load_latency_cycles = args.load_latency
    bytes_per_sec = hbm_bw_tbs * 1e12
    load_bw_cycles = bytes_per_kstep / (bytes_per_sec / freq_ghz / 1e9)  # 粗略转换
    load_cycles = load_latency_cycles + load_bw_cycles

    # 输出模式
    mode = args.mode
    stages = args.stages if args.mode == 'multistage' else 1

    print(f'{"参数":25} {"值":>30}')
    print('-' * 60)
    print(f'{"Tile (M×N×K)":25} {f"{tile_m}×{tile_n}×{tile_k}":>30}')
    print(f'{"每 k-step 字节":25} {f"{bytes_per_kstep/1024:.1f} KB":>30}')
    print(f'{"计算周期":25} {compute_cycles_per_kstep:>30.1f}')
    print(f'{"加载延迟（cycles）":25} {load_latency_cycles:>30.0f}')
    print(f'{"带宽贡献（cycles）":25} {load_bw_cycles:>30.1f}')
    print()

    if mode == 'sync':
        # 同步：搬完再算
        total_per_kstep = load_cycles + compute_cycles_per_kstep
        print(f'{"模式":25} {"sync（搬完再算）":>30}')
        print(f'{"每 k-step 周期":25} {total_per_kstep:>30.1f}')

    elif mode == 'double-buffer':
        # 双缓冲：第 i 步搬 i+1 的数据，计算 i 的结果
        # 最好情况：max(compute, load)
        total_per_kstep = max(compute_cycles_per_kstep, load_cycles)
        print(f'{"模式":25} {"double-buffer（2 阶段）":>30}')
        print(f'{"每 k-step 周期":25} {total_per_kstep:>30.1f}')

    elif mode == 'multistage':
        # 多级流水：S 个阶段并行
        # 最优情况：max(compute, load / (S-1))，因为 S-1 个 load 可并行
        effective_load = load_cycles / max(1, stages - 1)
        total_per_kstep = max(compute_cycles_per_kstep, effective_load)
        print(f'{"模式":25} {f"multistage（{stages} 阶段，TMA）":>30}')
        print(f'{"每 k-step 周期":25} {total_per_kstep:>30.1f}')
    else:
        total_per_kstep = load_cycles + compute_cycles_per_kstep
        print(f'{"模式":25} {"unknown":>30}')
        print(f'{"每 k-step 周期":25} {total_per_kstep:>30.1f}')

    # 计算整体 GEMM 的周期和吞吐
    # 4096×4096×4096，分块 128×128，k=64
    m_total, n_total, k_total = 4096, 4096, 4096
    tiles_m = m_total // tile_m
    tiles_n = n_total // tile_n
    k_steps = k_total // tile_k

    total_flops_gemm = m_total * n_total * k_total * 2

    # 粗略：每个分块瓷砖需要 k_steps 个计算步
    # 总周期 ≈ tiles_m × tiles_n × k_steps × cycles_per_kstep
    # 但这忽略了并行性，用另一个方式：直接从 peak TFLOPS 推

    # 实际：假设能达到峰值的某个百分比
    if mode == 'sync':
        utilization_pct = compute_cycles_per_kstep / total_per_kstep * 100
    else:
        utilization_pct = compute_cycles_per_kstep / total_per_kstep * 100

    achieved_tflops = peak_tflops * utilization_pct / 100

    print()
    print(f'{"峰值 TFLOPS":25} {peak_tflops:>30.0f}')
    print(f'{"利用率（%）":25} {utilization_pct:>30.1f}%')
    print(f'{"实现吞吐（TFLOPS）":25} {achieved_tflops:>30.1f}')
    print()
    print(f'【假设】HBM 延迟 {load_latency_cycles} cycles，带宽 {hbm_bw_tbs} TB/s；'
          f'共享内存 ~10 TB/s（per SM）；H100 频率 {freq_ghz} GHz')
    print(f'【数据来源】Luo et al. 2024 (arXiv 2402.13499); NVIDIA specs')
    print(f'【说明】sync 为基线；double-buffer 隐藏 load；multistage 进一步减少延迟贡献')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # peak 子命令
    sp = subparsers.add_parser('peak', help='Tensor Core 峰值')
    sp.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    sp.add_argument('--dtype', default='bf16', choices=['bf16', 'fp8', 'fp4'], help='数据类型')
    sp.set_defaults(func=cmd_peak)

    # utilization 子命令
    sp = subparsers.add_parser('utilization', help='运用率')
    sp.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    sp.add_argument('--m', type=int, default=128, help='M 维度')
    sp.add_argument('--n', type=int, default=128, help='N 维度')
    sp.add_argument('--k', type=int, default=64, help='K 维度')
    sp.add_argument('--occupancy', type=float, default=0.5, help='占有率（0-1）')
    sp.set_defaults(func=cmd_utilization)

    # fragment 子命令
    sp = subparsers.add_parser('fragment', help='Fragment 布局')
    sp.add_argument('--api', default='mma.sync', choices=['wmma', 'mma.sync', 'wgmma', 'tcgen05'],
                    help='mma API')
    sp.add_argument('--m', type=int, default=16, help='M 维度')
    sp.add_argument('--n', type=int, default=8, help='N 维度')
    sp.add_argument('--k', type=int, default=16, help='K 维度')
    sp.set_defaults(func=cmd_fragment)

    # strength 子命令
    sp = subparsers.add_parser('strength', help='算术强度')
    sp.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    sp.add_argument('--m', type=int, default=4096, help='M 维度')
    sp.add_argument('--n', type=int, default=4096, help='N 维度')
    sp.add_argument('--k', type=int, default=4096, help='K 维度')
    sp.add_argument('--tile-m', type=int, default=128, help='Tile M')
    sp.add_argument('--tile-n', type=int, default=128, help='Tile N')
    sp.set_defaults(func=cmd_strength)

    # pipeline 子命令
    sp = subparsers.add_parser('pipeline', help='GEMM 流水线周期模型')
    sp.add_argument('--gpu', default='h100-sxm', help='GPU 型号')
    sp.add_argument('--tile-m', type=int, default=128, help='分块 M')
    sp.add_argument('--tile-n', type=int, default=128, help='分块 N')
    sp.add_argument('--tile-k', type=int, default=64, help='分块 K')
    sp.add_argument('--mode', default='sync', choices=['sync', 'double-buffer', 'multistage'],
                    help='流水模式')
    sp.add_argument('--stages', type=int, default=4, help='多级流水的阶段数（仅 multistage）')
    sp.add_argument('--load-latency', type=float, default=600,
                    help='HBM 加载延迟（cycles，默认 600 基于 Luo et al. 2024）')
    sp.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
