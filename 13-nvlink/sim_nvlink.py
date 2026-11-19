"""NVLink bandwidth and single-node ring all-reduce model (CPU). Companion to GPU-13.

Conventions (all per GPU, per direction unless stated otherwise):
  link_gbs   one NVLink link, one direction
  gpu_gbs    links x link_gbs, the number vendors quote as "900 GB/s" is bidirectional (2 x gpu_gbs)
Ring all-reduce over an NVSwitch fabric: every GPU can push gpu_gbs to its ring neighbour, so
  t      = 2(n-1)/n * S / B + 2(n-1) * alpha
  algbw  = S / t
  busbw  = algbw * 2(n-1)/n   (nccl-tests definition; compare it with B, the slowest hop's per-direction BW)

Subcommands:
  specs       NVLink generations: per-link and per-GPU bandwidth
  ring        n-GPU ring all-reduce: algbw, busbw and time
  topo-parse  parse an `nvidia-smi topo -m` matrix and count NVLink vs PCIe pairs

Usage:
  python3 sim_nvlink.py specs
  python3 sim_nvlink.py ring --n 8 --gen nvlink4
  python3 sim_nvlink.py ring --n 8 --gen nvlink4 --slowest pcie5
  python3 sim_nvlink.py topo-parse topo_sample_8gpu_hybrid.txt
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

# Per-link per-direction rate (GB/s) and links per GPU, from NVIDIA NVLink / product pages
# (checked 2026-11-04). nvlink6 is only published as 3.6 TB/s bidirectional per GPU.
NVLINK_GENS = {
    'nvlink1': {'gpu': 'P100', 'link_gbs': 20, 'links': 4},
    'nvlink2': {'gpu': 'V100', 'link_gbs': 25, 'links': 6},
    'nvlink3': {'gpu': 'A100', 'link_gbs': 25, 'links': 12},
    'nvlink4': {'gpu': 'H100', 'link_gbs': 25, 'links': 18},
    'nvlink5': {'gpu': 'B200', 'link_gbs': 50, 'links': 18},
    'nvlink6': {'gpu': 'Rubin', 'link_gbs': None, 'links': None, 'gpu_gbs': 1800},
}

# Per-direction bandwidth of a non-NVLink hop that can end up in the ring
SLOW_HOPS = {'pcie4': 32, 'pcie5': 63, 'pcie6': 126}

TOPO_BW = {
    1: 'NVLink',
    10: 'PCIe (PIX/PXB)',
    20: 'PCIe (PHB)',
    30: 'NUMA (NODE)',
    40: 'System (SYS)',
    255: 'None',
}


def gpu_gbs(gen):
    """Per-GPU per-direction NVLink bandwidth in GB/s."""
    s = NVLINK_GENS[gen]
    return s.get('gpu_gbs') or s['link_gbs'] * s['links']


def ring_allreduce(n, bw_gbs, msg_size_bytes=1e9, alpha_us=0.0):
    """Ring all-reduce over n GPUs whose slowest hop has bw_gbs per direction."""
    if n <= 1:
        return {'n': n, 'algbw_gbs': 0.0, 'busbw_gbs': 0.0, 'time_us': 0.0}
    factor = 2 * (n - 1) / n
    time_us = factor * msg_size_bytes / (bw_gbs * 1e9) * 1e6 + 2 * (n - 1) * alpha_us
    algbw = msg_size_bytes / (time_us * 1e-6) / 1e9
    busbw = algbw * factor
    return {
        'n': n,
        'bw_gbs': bw_gbs,
        'msg_size_bytes': msg_size_bytes,
        'algbw_gbs': algbw,
        'busbw_gbs': busbw,
        'time_us': time_us,
        'efficiency': busbw / bw_gbs,
    }


def cmd_specs(args):
    print(f'{"gen":9} {"GPU":6} {"link/dir":>10} {"links":>6} {"GPU/dir":>10} {"GPU bidir":>10}')
    print('-' * 56)
    for gen, s in NVLINK_GENS.items():
        if args.gen not in ('all', gen):
            continue
        link = f'{s["link_gbs"]} GB/s' if s['link_gbs'] else 'n/a'
        links = s['links'] or 'n/a'
        g = gpu_gbs(gen)
        print(f'{gen:9} {s["gpu"]:6} {link:>10} {links:>6} {g:>5} GB/s {2 * g:>5} GB/s')
    print()
    print('Vendor headline numbers (600 / 900 / 1800 GB/s) are bidirectional; use GPU/dir for budgets.')


def cmd_ring(args):
    if args.gen not in NVLINK_GENS:
        sys.exit(f'unknown NVLink generation: {args.gen}')
    bw = SLOW_HOPS[args.slowest] if args.slowest else gpu_gbs(args.gen)
    hop = args.slowest or f'{args.gen} ({bw} GB/s per direction per GPU)'
    print(f'ring all-reduce, slowest hop: {hop}, alpha {args.alpha_us} us/step')
    print(f'{"GPUs":>5} {"size":>9} {"algbw":>9} {"busbw":>9} {"time":>11} {"busbw/B":>8}')
    print('-' * 58)
    for n in [2, 4, 8, 16, 32]:
        if n > args.n_max:
            break
        r = ring_allreduce(n, bw, args.msg_bytes, args.alpha_us)
        size = f'{args.msg_bytes / 1e9:.1f} GB' if args.msg_bytes >= 1e9 else f'{args.msg_bytes / 1e6:.1f} MB'
        tm = f'{r["time_us"]:.0f} us' if r['time_us'] < 1e6 else f'{r["time_us"] / 1e6:.2f} s'
        print(f'{n:>5} {size:>9} {r["algbw_gbs"]:>9.1f} {r["busbw_gbs"]:>9.1f} {tm:>11} {r["efficiency"]:>7.1%}')


def parse_topo_matrix(filename):
    """解析 topo -m 矩阵文件。返回 (devices, matrix)。"""
    devices = []
    matrix = []

    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    if not lines:
        raise ValueError('矩阵文件为空')

    header = lines[0].split()
    for i, h in enumerate(header):
        if 'GPU' in h or 'NIC' in h:
            if i + 1 < len(header) and header[i+1].isdigit():
                devices.append(f'{h} {header[i+1]}')
            else:
                devices.append(f'{h} {len(devices)}')

    for line in lines[1:]:
        row = line.split()
        row_data = []
        for v in row:
            if v.isdigit() or v == 'X':
                try:
                    row_data.append(int(v) if v != 'X' else 1)
                except ValueError:
                    pass
        if row_data:
            matrix.append(row_data)

    return devices, matrix


def cmd_topo_parse(args):
    """解析 topo 矩阵，统计 NVLink 链路数"""
    try:
        devices, matrix = parse_topo_matrix(args.file)
    except Exception as e:
        print(f'错误：{e}', file=sys.stderr)
        sys.exit(1)

    n = len(matrix)

    # 距离统计
    dist_count = defaultdict(int)
    dist_pairs = defaultdict(list)

    for i in range(n):
        for j in range(n):
            if i != j:
                d = matrix[i][j]
                dist_count[d] += 1
                dist_pairs[d].append((i, j))

    print(f'topo -m 矩阵解析：{n} 个 GPU')
    print()
    print(f'{"距离":>6} {"类型":15} {"链路对数":>10} {"占比":>8}')
    print('-' * 50)

    total_pairs = n * (n - 1)

    for d in sorted(dist_count.keys()):
        count = dist_count[d]
        pct = 100.0 * count / total_pairs if total_pairs > 0 else 0
        bw_type = TOPO_BW.get(d, '?')
        print(f'{d:>6}  {bw_type:15}  {count:>10}  {pct:>7.1f}%')

    print()

    # NVLink 统计
    nvlink_count = sum(dist_count.get(d, 0) for d in [1])
    pcie_count = sum(dist_count.get(d, 0) for d in [10, 20, 30, 40])

    print(f'NVLink 链路对：{nvlink_count} / {total_pairs}（{100.0*nvlink_count/total_pairs:.1f}%）')
    print(f'PCIe 或更慢：{pcie_count} / {total_pairs}（{100.0*pcie_count/total_pairs:.1f}%）')
    print()

    if nvlink_count > 0:
        avg_nvlink_per_gpu = nvlink_count / n
        print(f'平均每卡 NVLink 直连数：{avg_nvlink_per_gpu:.1f}')

    print('建议：')
    if dist_count.get(1, 0) == n * (n - 1):
        print('✓ 全互联 NVLink：所有 GPU 之间都是 NVLink 直连。')
    elif dist_count.get(1, 0) > 0:
        print(f'⚠ 部分 NVLink：{nvlink_count} 对链路走 NVLink，其余经 PCIe/NUMA。')
        print('  环形 all-reduce 会按最慢的链路（PCIe）算带宽。')
    else:
        print('✗ 无 NVLink：所有链路都走 PCIe 或更慢，不适合密集通信。')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    specs_parser = subparsers.add_parser('specs', help='NVLink 规格表')
    specs_parser.add_argument('--gen', default='all', help='特定代数（如 nvlink4、nvlink5）')

    ring_parser = subparsers.add_parser('ring', help='环形 all-reduce 带宽模型')
    ring_parser.add_argument('--gen', default='nvlink4', help='NVLink generation (default nvlink4)')
    ring_parser.add_argument('--n', type=int, dest='n_max', default=8, help='max GPU count (default 8)')
    ring_parser.add_argument('--msg', type=float, dest='msg_bytes', default=1e9, help='message size in bytes (default 1e9)')
    ring_parser.add_argument('--alpha-us', type=float, default=0.0, help='per-step latency in us (default 0)')
    ring_parser.add_argument('--slowest', choices=sorted(SLOW_HOPS), default=None,
                             help='a PCIe hop that sits in the ring and caps it')

    topo_parser = subparsers.add_parser('topo-parse', help='解析 topo -m 矩阵')
    topo_parser.add_argument('file', help='topo -m 矩阵文件')

    args = parser.parse_args()

    if args.cmd == 'specs':
        cmd_specs(args)
    elif args.cmd == 'ring':
        cmd_ring(args)
    elif args.cmd == 'topo-parse':
        cmd_topo_parse(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
