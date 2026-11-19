"""Occupancy 计算器（CPU）。对应 GPU-01 第 7 章。

给定 compute capability、每块线程数、每线程寄存器数和每块共享内存，算出每个 SM 能驻留几个块、
几个 warp，以及是哪一项资源先卡住。

各架构上限取自 CUDA Programming Guide 附录 Compute Capabilities 的表 30 至表 32，
和 Ampere、Hopper、Blackwell Tuning Guide（核对于 2026-11-02）。

寄存器的算法：
  1. 每个 warp 的寄存器数按 256 取整（CUDA C++ Best Practices Guide, Occupancy 一节）。
  2. SM 的寄存器文件分成 4 个分区，每个分区 16K 个 32 位寄存器，warp 固定落在一个分区里，
     所以先算每个分区能放几个 warp，再乘 4。这样才能复现 Best Practices Guide 里
     37 寄存器、320 线程块只能驻留 4 块（63%）的例子。
共享内存的算法：
  每块的用量加上 CUDA 为每块保留的 1 KB（CC 8.0 起），再按 128 字节取整。
  分配粒度因架构而异，以 Nsight Compute 的 Occupancy Calculator 为准。
  这里假设 kernel 已经通过 cudaFuncAttributeMaxDynamicSharedMemorySize 申请到该架构的最大 carveout。

用法：
  python3 sim_occupancy.py --cc 9.0 --threads 256 --regs 64 --smem 48
  python3 sim_occupancy.py --cc 9.0 --sweep-regs --threads 128
"""
import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str
    max_warps: int        # 每 SM 最多驻留 warp
    max_blocks: int       # 每 SM 最多驻留块
    regs: int             # 每 SM 32 位寄存器个数
    smem_sm_kb: int       # 每 SM 最大共享内存（最大 carveout）
    smem_block_kb: int    # 每块最大共享内存
    reserved_b: int       # CUDA 为每块保留的共享内存字节数
    smem_gran: int = 128  # 共享内存分配粒度，字节
    reg_gran: int = 256   # 每 warp 寄存器分配粒度
    max_regs_thread: int = 255


ARCHS = {
    '7.0': Arch('Volta V100', 64, 32, 65536, 96, 96, 0, 256),
    '7.5': Arch('Turing', 32, 16, 65536, 64, 64, 0, 256),
    '8.0': Arch('Ampere A100', 64, 32, 65536, 164, 163, 1024),
    '8.6': Arch('Ampere GA10x', 48, 16, 65536, 100, 99, 1024),
    '8.9': Arch('Ada', 48, 24, 65536, 100, 99, 1024),
    '9.0': Arch('Hopper H100/H200', 64, 32, 65536, 228, 227, 1024),
    '10.0': Arch('Blackwell B200/GB200', 64, 32, 65536, 228, 227, 1024),
    '12.0': Arch('Blackwell GeForce/RTX PRO', 48, 24, 65536, 100, 99, 1024),
}


def occupancy(cc, threads, regs, smem_kb=0.0):
    a = ARCHS[cc]
    if threads % 32:
        print(f'  提示：{threads} 不是 32 的倍数，最后一个 warp 有空线程')
    if regs > a.max_regs_thread:
        raise ValueError(f'每线程最多 {a.max_regs_thread} 个寄存器')
    if smem_kb > a.smem_block_kb:
        raise ValueError(f'每块共享内存最多 {a.smem_block_kb} KB')
    wpb = math.ceil(threads / 32)
    lim = {}
    lim['warps'] = a.max_warps // wpb
    lim['blocks'] = a.max_blocks
    if regs:
        per_warp = math.ceil(regs * 32 / a.reg_gran) * a.reg_gran
        warps_by_reg = 4 * ((a.regs // 4) // per_warp)
        lim['registers'] = warps_by_reg // wpb
    if smem_kb:
        per_block = math.ceil((smem_kb * 1024 + a.reserved_b) / a.smem_gran) * a.smem_gran
        lim['shared_mem'] = (a.smem_sm_kb * 1024) // per_block
    blocks = min(lim.values())
    warps = blocks * wpb
    limiter = [k for k, v in lim.items() if v == blocks]
    return dict(arch=a.name, blocks=blocks, warps=warps, occ=warps / a.max_warps, limits=lim, limiter=limiter)


def show(cc, threads, regs, smem):
    r = occupancy(cc, threads, regs, smem)
    print(f'CC {cc} {r["arch"]}: {threads} 线程/块, {regs} 寄存器/线程, {smem:g} KB 共享内存/块')
    for k, v in r['limits'].items():
        print(f'  按 {k:10} 限制最多 {v:3d} 块')
    print(f'  驻留 {r["blocks"]} 块 = {r["warps"]} warp，理论 occupancy {r["occ"]:.1%}，瓶颈 {"、".join(r["limiter"])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cc', default='9.0', choices=ARCHS)
    p.add_argument('--threads', type=int, default=256)
    p.add_argument('--regs', type=int, default=32)
    p.add_argument('--smem', type=float, default=0, help='每块共享内存，KB')
    p.add_argument('--sweep-regs', action='store_true')
    a = p.parse_args()
    if a.sweep_regs:
        print(f'CC {a.cc}，{a.threads} 线程/块，寄存器数和 occupancy 的阶梯：')
        last = None
        for regs in range(16, 256, 8):
            o = occupancy(a.cc, a.threads, regs, a.smem)['occ']
            if o != last:
                print(f'  {regs:3d} 寄存器起 -> {o:.1%}')
                last = o
    else:
        show(a.cc, a.threads, a.regs, a.smem)
