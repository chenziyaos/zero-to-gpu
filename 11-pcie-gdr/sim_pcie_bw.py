"""PCIe 带宽模型：链路规格、编码、MPS 与实测有效带宽（CPU）。对应 GPU-11 第 1-2 章。

PCIe 双向带宽如下，单向和双向标注清楚：
  Gen4 x16: 32 GT/s × 16 / 8 × 2 = 64 GB/s 双向（单向 32 GB/s），扣编码 8b/10b 后 51.2 GB/s
  Gen5 x16: 64 GT/s × 16 / 8 × 2 = 128 GB/s 双向（单向 63 GB/s），扣编码 128b/130b 后约 122 GB/s
  Gen5 x8:  64 GT/s × 8 / 8 × 2 = 64 GB/s 双向（单向 31.5 GB/s）
  Gen5 x4:  64 GT/s × 4 / 8 × 2 = 32 GB/s 双向（单向 15.75 GB/s）

编码包括：
  - Physical Layer (8b/10b 或 128b/130b)
  - Link Layer: DMA 读写时往返确认 (ACK)、流控
  - Data Link Layer: CRC、TLP 包头开销

DMA 写上限（未包含 ACK 回程）约为 80-90% 效率。单卡 H100 SXM 主机端的 H2D 实测约 50-53 GB/s；
主机到设备读（D2H）约 48-52 GB/s。参考资料见正文。

子命令：
  peak      各代 PCIe 链路的峰值双向和单向带宽
  h2d       H2D 拷贝上限：MPS 大小和编码低头
  d2h       D2H 拷贝上限：往返确认和丢包重传的影响
  topo      解析 topo -m 距离和决定带宽阶梯
  pinned    pinned 内存相比可分页内存的加速倍率（文献值与机器相关）

用法：
  python3 sim_pcie_bw.py peak
  python3 sim_pcie_bw.py h2d --gen 5 --width 16 --mps 256
  python3 sim_pcie_bw.py d2h --gen 5 --width 16
  python3 sim_pcie_bw.py topo
  python3 sim_pcie_bw.py pinned
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402


# PCIe 代际基础参数
# 来源：PCI-SIG 规范、NVIDIA GPU 产品页（核对于 2026-11-04）
# 单向完整带宽（未编码）；NVIDIA 规格表中的"X GB/s"通常是双向合计
# Gen4 x16: 32 GT/s × 16 lanes / (8 bits/byte) = 64 GB/s 双向 (32 GB/s per dir)
# Gen5 x16: 64 GT/s × 16 lanes / (8 bits/byte) = 128 GB/s 双向 (64 GB/s per dir, before encoding)
PCIE_SPECS = {
    'gen4': {
        'name': 'Gen4',
        'single_dir_gbs': 32,  # 单向未编码，GB/s
        'encoding': '8b/10b',
        'encoding_eff': 8 / 10,
    },
    'gen5': {
        'name': 'Gen5',
        'single_dir_gbs': 64,  # 单向未编码，GB/s
        'encoding': '128b/130b',
        'encoding_eff': 128 / 130,
    },
}

# PCIe MPS (Maximum Payload Size) 对 DMA 写的影响
# 来源：https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding.pdf
# 更大的 MPS 让效率更高，但增加切片开销；256 字节是常见配置
MPS_DATA = {
    64: {'tlp_bytes': 64 + 16, 'eff': 0.62},    # 小 MPS，包头开销大
    128: {'tlp_bytes': 128 + 16, 'eff': 0.75},
    256: {'tlp_bytes': 256 + 16, 'eff': 0.83},
    512: {'tlp_bytes': 512 + 16, 'eff': 0.87},
    1024: {'tlp_bytes': 1024 + 16, 'eff': 0.90},
    2048: {'tlp_bytes': 2048 + 16, 'eff': 0.92},
    4096: {'tlp_bytes': 4096 + 16, 'eff': 0.93},
}


def peak_bw_single(gen, width):
    """单向峰值带宽 GB/s（未扣编码）。"""
    spec = PCIE_SPECS.get(gen)
    if not spec:
        raise ValueError(f'未知代际 {gen}')
    # 标称单向 GB/s 乘以宽度比例
    # 例如 Gen5 x16: 64 GB/s，x8 则 32 GB/s
    return spec['single_dir_gbs'] * width / 16


def peak_bw(gen, width):
    """给定代际和宽度，双向峰值带宽 GB/s（未扣编码）。"""
    return peak_bw_single(gen, width) * 2


def h2d_effective_bw(gen, width, mps=256, ack_ratio=0.10):
    """H2D 有效带宽（扣编码、MPS、ACK 往返）。

    模型：raw_single × encoding_eff × mps_eff × (1 - ack_ratio)

    编码在物理层已扣：
    - Physical Layer (8b/10b 或 128b/130b)
    - TLP 包头开销（包含在 mps_eff）
    - ACK 往返（逻辑上不搬数据，但占链路）

    参数：
      gen: 'gen4' 或 'gen5'
      width: x1, x2, x4, x8, x16
      mps: Maximum Payload Size，字节；常见 256
      ack_ratio: ACK 往返占链路时间的比例，典型 8-12%
    """
    raw_single = peak_bw_single(gen, width)

    # 扣编码
    spec = PCIE_SPECS[gen]
    after_encoding = raw_single * spec['encoding_eff']

    # 扣 MPS 开销
    if mps not in MPS_DATA:
        raise ValueError(f'MPS {mps} 未配置，可选 {list(MPS_DATA.keys())}')
    mps_eff = MPS_DATA[mps]['eff']
    after_mps = after_encoding * mps_eff

    # 扣 ACK 开销（10% 较为保守）
    after_ack = after_mps * (1 - ack_ratio)

    return after_ack


def h2d_with_slack(gen, width, mps=256):
    """H2D 实际有效带宽范围（考虑系统变异）。
    返回 (min_gbps, typical_gbps, max_gbps) 三个参考值。
    系统变异: ±6% 范围内正常。
    """
    base = h2d_effective_bw(gen, width, mps)
    return (base * 0.94, base, base * 0.96)


def d2h_effective_bw(gen, width):
    """D2H 有效带宽。由于往返确认更多，实测效率略低于 H2D。
    参考资料：Neugebauer et al. 测得约 90-92% 相对于 H2D 的效率。
    """
    h2d = h2d_effective_bw(gen, width)
    return h2d * 0.91


def cmd_peak(args):
    """输出各代 PCIe 链路的峰值"""
    print('PCIe 代际     双向峰值    单向峰值')
    print('           GB/s        GB/s')
    for gen in ['gen4', 'gen5']:
        bid = peak_bw(gen, 16)
        uni = peak_bw_single(gen, 16)
        print(f'{PCIE_SPECS[gen]["name"]} x16  {bid:7.1f}      {uni:7.1f}')
    print()
    print('PCIe Gen5 各宽度（单向）')
    print('x1        x2        x4        x8       x16')
    for w in [1, 2, 4, 8, 16]:
        print(f'{peak_bw_single("gen5", w):7.1f}', end='  ')
    print()


def cmd_h2d(args):
    """H2D 拷贝带宽分析"""
    gen = args.gen
    width = args.width
    mps = args.mps

    raw_single = peak_bw_single(gen, width)
    spec = PCIE_SPECS[gen]
    after_encoding = raw_single * spec['encoding_eff']
    eff_h2d = h2d_effective_bw(gen, width, mps)
    min_bw, typ_bw, max_bw = h2d_with_slack(gen, width, mps)

    print(f'H2D 拷贝带宽分解（PCIe {PCIE_SPECS[gen]["name"]} x{width}, MPS {mps}B）')
    print('=' * 50)
    print(f'单向峰值（未编码）: {raw_single:7.1f} GB/s')
    print(f'编码（{spec["encoding"]}）: {after_encoding:7.1f} GB/s（-{(1-spec["encoding_eff"])*100:.1f}%）')
    mps_eff = MPS_DATA[mps]['eff']
    print(f'MPS {mps}B 有效率:   {after_encoding * mps_eff:7.1f} GB/s（-{(1-mps_eff)*100:.1f}%）')
    print(f'ACK 往返折扣：        {eff_h2d:7.1f} GB/s（-10%）')
    print()
    print(f'预期 H2D 上限（pinned 内存）: {eff_h2d:.1f} GB/s')
    print(f'实际范围（±6% 系统变异）: {min_bw:.1f}～{max_bw:.1f} GB/s')


def cmd_d2h(args):
    """D2H 拷贝带宽"""
    gen = args.gen
    width = args.width

    h2d = h2d_effective_bw(gen, width)
    d2h = d2h_effective_bw(gen, width)

    print(f'D2H 拷贝带宽（PCIe {PCIE_SPECS[gen]["name"]} x{width}）')
    print('=' * 50)
    print(f'H2D 有效带宽:      {h2d:.1f} GB/s')
    print(f'D2H 有效带宽:      {d2h:.1f} GB/s（-9% 往返确认开销）')


def cmd_topo(args):
    """topo -m 六级距离解释"""
    print('PCIe 拓扑距离与带宽阶梯')
    print('=' * 60)
    distances = [
        ('GPU', 'GPU', 'PXB', '单 PXB 内', '60-100 GB/s (NVLink)'),
        ('GPU', 'NIC', 'PXB', '单 PXB 内', '63 GB/s (Gen5 x16)'),
        ('GPU', 'CPU/MEM', 'PHB', '跨 CPU 桥', '31.5 GB/s (Gen5 x8)'),
        ('GPU', '远端 NUMA', 'NODE', '跨 socket', '8-15 GB/s'),
        ('GPU', '其他', 'SYS', '系统范围', '<8 GB/s'),
    ]
    print('距离级   常见配置         类型         典型场景           有效带宽')
    for src, dst, level, desc, bw in distances:
        print(f'{level:5}  {src:3}-{dst:3}          {desc:12}   {bw:20}')
    print()
    print('说明：')
    print('- PXB: PCIe Expansion Bus，单平面内不经 CPU')
    print('- PHB: PCIe Host Bridge，经过 CPU 的主桥')
    print('- NODE: 跨 NUMA socket，经 UPI/GMI')
    print('- 距离来源：topo -m 和 NCCL paths.cc')
    print('- NCCL 默认 P2P 和 GPUDirect 仅用到 PXB，超过则用主存中转')


def cmd_pinned(args):
    """Pinned 内存的加速倍率与模型预测"""
    print('Pinned 内存 vs 可分页内存（H2D）')
    print('=' * 50)
    print()

    # 基于模型计算各场景
    print('场景                             模型预测 H2D (pinned)    加速倍数')
    print('-' * 70)

    scenarios = [
        ('PCIe Gen4 x16, MPS 256B', 'gen4', 16, 256),
        ('PCIe Gen5 x16, MPS 256B', 'gen5', 16, 256),
        ('PCIe Gen5 x8, MPS 256B', 'gen5', 8, 256),
    ]

    # pageable 内存相比 pinned 的折扣因子（Neugebauer et al.）
    pageable_factor = 0.45  # pageable 约为 pinned 的 45-55%

    for label, gen, width, mps in scenarios:
        pinned_bw = h2d_effective_bw(gen, width, mps)
        pageable_bw = pinned_bw * pageable_factor
        speedup = pinned_bw / pageable_bw if pageable_bw > 0 else 0
        print(f'{label:30} ~{pinned_bw:5.1f} GB/s        {speedup:5.2f}x')

    print()
    print('原因：')
    print('- pinned 内存：预先锁定物理页、建立直接 IOMMU 映射，DMA 跳过翻译')
    print('- 可分页内存：CPU 动态翻译、bounce buffer 复制、staging 开销累积')
    print('- 跨 NUMA 时 IOMMU 翻译成本更高，pinned 优势显著')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    subparsers.add_parser('peak', help='各代 PCIe 峰值')

    h2d_parser = subparsers.add_parser('h2d', help='H2D 拷贝带宽分析')
    h2d_parser.add_argument('--gen', default='gen5', choices=['gen4', 'gen5'])
    h2d_parser.add_argument('--width', type=int, default=16)
    h2d_parser.add_argument('--mps', type=int, default=256)

    d2h_parser = subparsers.add_parser('d2h', help='D2H 拷贝带宽')
    d2h_parser.add_argument('--gen', default='gen5', choices=['gen4', 'gen5'])
    d2h_parser.add_argument('--width', type=int, default=16)

    subparsers.add_parser('topo', help='topo -m 距离说明')
    subparsers.add_parser('pinned', help='pinned 内存加速倍率')

    args = parser.parse_args()

    if args.cmd == 'peak':
        cmd_peak(args)
    elif args.cmd == 'h2d':
        cmd_h2d(args)
    elif args.cmd == 'd2h':
        cmd_d2h(args)
    elif args.cmd == 'topo':
        cmd_topo(args)
    elif args.cmd == 'pinned':
        cmd_pinned(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
