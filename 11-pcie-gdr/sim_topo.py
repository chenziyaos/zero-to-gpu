"""topo -m 拓扑矩阵解析器与距离分类（CPU）。对应 GPU-11 第 3 章。

topo -m 输出设备间的六级距离：
  1 (default): 最好，主要是NVLink同一卡
  10 (PXB): PCIe同一张表，不过CPU host bridge
  20 (PHB): 过 PCIe host bridge，但同 CPU 芯片
  30 (NODE): 跨 NUMA node（socket 间 UPI/GMI）
  40 (SYS): 系统范围
  255: 无连接

NCCL 文档：P2P 和 GPUDirect 默认只走到 PXB（10），其他都走主存。

示例输入格式（topo -m 输出）：
  GPU 0   X    PIX  PXB  PHB  NODE
  GPU 1   PIX  X    PXB  PHB  NODE
  ...

子命令：
  parse   解析 topo -m 矩阵文件，分类距离和带宽
  sample  输出示例矩阵（可保存到文件后 parse 检验）
  stat    统计矩阵里的距离分布

用法：
  python3 sim_topo.py sample > sample_topo.txt
  python3 sim_topo.py parse sample_topo.txt
  python3 sim_topo.py parse /proc/driver/nvidia/gpus/0/topo/matrix
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

# topo -m 距离常量
# 来源：NVIDIA nvidia-smi 文档、NCCL paths.cc（核对于 2026-11-04）
TOPO_DISTANCES = {
    1: {'name': 'GPU', 'desc': 'NVLink 或 L2 缓存同级', 'nccl_use': 'NVLINK/FAST'},
    10: {'name': 'PXB', 'desc': 'PCIe 同平面，无 host bridge', 'nccl_use': 'P2P'},
    20: {'name': 'PHB', 'desc': 'PCIe host bridge，过 CPU', 'nccl_use': '主存中转'},
    30: {'name': 'NODE', 'desc': '跨 NUMA socket（UPI/GMI）', 'nccl_use': '主存中转'},
    40: {'name': 'SYS', 'desc': '系统范围', 'nccl_use': '主存中转'},
    255: {'name': 'NONE', 'desc': '无连接', 'nccl_use': '不支持'},
}


def parse_topo_matrix(filename):
    """解析 topo -m 矩阵文件。返回 (devices, matrix)。

    matrix[i][j] 是设备 i 到设备 j 的距离（整数）。
    """
    devices = []
    matrix = []

    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    # 第一行：设备列表
    if not lines:
        raise ValueError('矩阵文件为空')

    header = lines[0].split()
    # 通常格式是 "GPU 0  GPU 1  GPU 2" 或 "GPU  GPU  GPU"，取后面的数字或序号
    for i, h in enumerate(header):
        if 'GPU' in h or 'NIC' in h or 'CPU' in h:
            if i + 1 < len(header) and header[i+1].isdigit():
                devices.append(f'{h} {header[i+1]}')
            else:
                devices.append(f'{h} {len(devices)}')

    # 后续行：矩阵内容
    for line in lines[1:]:
        row = line.split()
        # 第一列可能是标签，后续是数字；去掉标签后取纯数字
        row_data = []
        for v in row:
            if v.isdigit() or v == 'X':
                try:
                    row_data.append(int(v) if v != 'X' else 1)
                except ValueError:
                    pass
        if row_data:
            matrix.append(row_data)

    if not matrix:
        raise ValueError('无法解析矩阵')

    # 检查矩阵尺寸
    n_devs = len([d for d in devices if d])
    if len(matrix) != n_devs or any(len(r) != n_devs for r in matrix):
        print(f'警告：矩阵大小 {len(matrix)}x{len(matrix[0]) if matrix else 0}，设备数 {n_devs}', file=sys.stderr)

    return devices, matrix


def cmd_parse(args):
    """解析并分析 topo 矩阵"""
    try:
        devices, matrix = parse_topo_matrix(args.file)
    except Exception as e:
        print(f'错误：{e}', file=sys.stderr)
        sys.exit(1)

    n = len(matrix)
    print(f'找到 {n} 个设备')
    print()

    # 距离统计
    dist_count = defaultdict(int)
    dist_pairs = defaultdict(list)

    for i in range(n):
        for j in range(n):
            if i != j:
                d = matrix[i][j]
                dist_count[d] += 1
                dist_pairs[d].append((i, j))

    # 输出设备列表
    print('设备列表：')
    for idx, dev in enumerate(devices):
        if dev:
            print(f'  [{idx}] {dev}')
    print()

    # 输出距离分布
    print('距离分布：')
    print(f'{"距离":6} {"级别":10} {"对数":6} {"说明"}')
    print('-' * 60)
    for d in sorted(dist_count.keys()):
        info = TOPO_DISTANCES.get(d, {'name': '?', 'desc': '未知'})
        print(f'{d:6} {info["name"]:10} {dist_count[d]:6}  {info["desc"]}')
    print()

    # 关键对列表（PXB 内的）
    if 10 in dist_pairs and dist_pairs[10]:
        print('PXB 内配对（可用 P2P）：')
        for i, j in sorted(dist_pairs[10])[:10]:
            print(f'  GPU {i} ↔ GPU {j}')
        if len(dist_pairs[10]) > 10:
            print(f'  ...还有 {len(dist_pairs[10]) - 10} 对')
        print()

    # 跨 host bridge 的配对
    if 20 in dist_pairs and dist_pairs[20]:
        print('跨 host bridge 配对（需主存中转）：')
        for i, j in sorted(dist_pairs[20])[:5]:
            print(f'  GPU {i} ↔ GPU {j}')
        if len(dist_pairs[20]) > 5:
            print(f'  ...还有 {len(dist_pairs[20]) - 5} 对')
        print()

    # 跨 NUMA 的配对
    if 30 in dist_pairs and dist_pairs[30]:
        print('跨 NUMA socket（NODE）：')
        for i, j in sorted(dist_pairs[30])[:5]:
            print(f'  GPU {i} ↔ GPU {j}')
        if len(dist_pairs[30]) > 5:
            print(f'  ...还有 {len(dist_pairs[30]) - 5} 对')
        print()

    # 推荐
    print('建议：')
    if dist_count.get(10, 0) == n * (n - 1):
        print('✓ 所有 GPU 在同一 PXB 内，P2P 和 GPUDirect RDMA 都可用。')
    else:
        print('⚠ GPU 跨越多个平面，部分配对需主存中转。检查：')
        print('  - 是否可用 NVLink 而不是 PCIe？')
        print('  - 跨 NUMA 的拷贝是否已绑核？')


def cmd_sample(args):
    """输出示例矩阵"""
    sample = """# 样例 topo -m 矩阵（8 卡 H100 DGX 单 NVSwitch）
# 格式：第一行设备列表，后续行距离矩阵；对角线为 X 或 1

GPU 0   GPU 1   GPU 2   GPU 3   GPU 4   GPU 5   GPU 6   GPU 7
X       1       1       1       10      10      10      10
1       X       1       1       10      10      10      10
1       1       X       1       10      10      10      10
1       1       1       X       10      10      10      10
10      10      10      10      X       1       1       1
10      10      10      10      1       X       1       1
10      10      10      10      1       1       X       1
10      10      10      10      1       1       1       X
"""
    print(sample, end='')


def cmd_stat(args):
    """统计矩阵"""
    try:
        devices, matrix = parse_topo_matrix(args.file)
    except Exception as e:
        print(f'错误：{e}', file=sys.stderr)
        sys.exit(1)

    n = len(matrix)
    dist_count = defaultdict(int)
    max_dist = 0
    min_dist = 255

    for i in range(n):
        for j in range(n):
            if i != j:
                d = matrix[i][j]
                dist_count[d] += 1
                max_dist = max(max_dist, d)
                min_dist = min(min_dist, d)

    print(f'统计：{n} 个设备')
    print(f'距离范围：{min_dist} 到 {max_dist}')
    print(f'配对总数：{n * (n - 1)}')
    print(f'平均距离：{sum(d * c for d, c in dist_count.items()) / sum(dist_count.values()):.1f}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    parse_parser = subparsers.add_parser('parse', help='解析 topo 矩阵文件')
    parse_parser.add_argument('file', help='topo -m 矩阵文件或 /proc/driver/nvidia/gpus/*/topo/matrix')

    subparsers.add_parser('sample', help='输出示例矩阵')

    stat_parser = subparsers.add_parser('stat', help='统计矩阵')
    stat_parser.add_argument('file', help='topo -m 矩阵文件')

    args = parser.parse_args()

    if args.cmd == 'parse':
        cmd_parse(args)
    elif args.cmd == 'sample':
        cmd_sample(args)
    elif args.cmd == 'stat':
        cmd_stat(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
