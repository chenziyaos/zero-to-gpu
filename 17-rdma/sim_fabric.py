"""轨道优化 leaf-spine 网络的带宽账（CPU）。对应 GPU-17 讲拓扑与收敛比的部分。

模型：每台机器 G 张卡、每卡一块网卡（单向 nic_gbps/8 GB/s）。轨道优化（rail-optimized）把所有机器的
第 i 号网卡接到同一组 leaf（第 i 条轨道）。leaf 端口一部分向下接网卡、一部分向上接 spine，
收敛比 oversub = 下行端口数 / 上行端口数（1 即无阻塞）。

每卡可用跨机带宽取决于流量有多少离开 leaf：
  - 只在 leaf 内的流量不受收敛比影响；
  - 离开 leaf 的流量共享上行，可用带宽 = 网卡带宽 / max(1, oversub × 离开 leaf 的流量占比)。
NCCL 的 ring 按轨道对齐建环；all-to-all 打开 PXN 时先经 NVLink 转到目的轨道的同号卡，
所以两者都只在轨道内走网络，离开 leaf 的只有同轨道里落在别的 leaf 上的那部分。
关掉 PXN（--no-pxn）时，跨轨道的目的地都要经 spine。

子命令：
  fabric   给定规模和收敛比，打印 leaf 布局、二分带宽和两种流量下的每卡可用带宽
  sweep    固定规模，扫收敛比 1、2、4

用法：
  python3 sim_fabric.py fabric --nodes 32
  python3 sim_fabric.py fabric --nodes 128 --oversub 2
  python3 sim_fabric.py fabric --nodes 128 --oversub 2 --no-pxn
  python3 sim_fabric.py sweep --nodes 128
"""
import argparse
import math


def leaf_layout(nodes, leaf_ports, oversub):
    """每台 leaf 的下行口、上行口，以及每条轨道需要几台 leaf。"""
    down = int(leaf_ports * oversub / (1 + oversub))
    up = leaf_ports - down
    return down, up, math.ceil(nodes / down)


def fabric(nodes, gpus_per_node=8, nic_gbps=400, oversub=1.0, leaf_ports=64, pxn=True):
    nic = nic_gbps / 8                       # 400G 网卡单向 50 GB/s
    down, up, lpr = leaf_layout(nodes, leaf_ports, oversub)
    per_leaf = min(nodes, down)
    gpus = nodes * gpus_per_node
    eff = oversub if lpr > 1 else 1.0        # 一条轨道只有一台 leaf 时，轨道内流量不上 spine
    # 轨道对齐 ring：每台 leaf 只有进出两条边离开 leaf，占 leaf 内全部边的 2/per_leaf
    ring_cross = 0.0 if lpr == 1 else min(1.0, 2 / per_leaf)
    # all-to-all：其他机器里落在别的 leaf 上的占比
    others = max(1, nodes - 1)
    a2a_cross = (nodes - per_leaf) / others
    if not pxn:
        # 跨轨道的目的地（占 (G-1)/G）都要经 spine
        a2a_cross = (gpus_per_node - 1) / gpus_per_node + a2a_cross / gpus_per_node
        eff = oversub
    return {
        'nodes': nodes, 'gpus': gpus, 'nic': nic, 'oversub': oversub, 'pxn': pxn,
        'down': down, 'up': up, 'leaves_per_rail': lpr, 'nodes_per_leaf': per_leaf,
        'bisection_tbs': gpus / 2 * nic / (oversub if lpr > 1 or not pxn else 1.0) / 1000,
        'ring': nic / max(1.0, eff * ring_cross),
        'a2a': nic / max(1.0, eff * a2a_cross),
        'a2a_cross': a2a_cross,
    }


def show(r):
    print(f"规模 {r['nodes']} 台 × {r['gpus'] // r['nodes']} 卡 = {r['gpus']} 卡，网卡单向 {r['nic']:.0f} GB/s，"
          f"收敛比 {r['oversub']:g}:1，PXN {'开' if r['pxn'] else '关'}")
    print(f"leaf 每台下行 {r['down']} 口、上行 {r['up']} 口；每条轨道 {r['leaves_per_rail']} 台 leaf，每台接 {r['nodes_per_leaf']} 台机器")
    print(f"二分带宽（单向）          {r['bisection_tbs']:7.1f} TB/s")
    print(f"每卡注入带宽              {r['nic']:7.1f} GB/s")
    print(f"每卡可用：轨道对齐 ring    {r['ring']:7.1f} GB/s")
    print(f"每卡可用：all-to-all       {r['a2a']:7.1f} GB/s（{r['a2a_cross']:.0%} 的跨机流量要上 spine）")


def main():
    ap = argparse.ArgumentParser(description='轨道优化 leaf-spine 网络的带宽账')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fabric', 'sweep'):
        p = sub.add_parser(name)
        p.add_argument('--nodes', type=int, default=32)
        p.add_argument('--gpus-per-node', type=int, default=8)
        p.add_argument('--nic-gbps', type=float, default=400)
        p.add_argument('--leaf-ports', type=int, default=64, help='leaf 端口数，按网卡速率计')
        p.add_argument('--no-pxn', action='store_true')
        if name == 'fabric':
            p.add_argument('--oversub', type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == 'fabric':
        show(fabric(a.nodes, a.gpus_per_node, a.nic_gbps, a.oversub, a.leaf_ports, not a.no_pxn))
        return
    print(f"{'收敛比':>6} {'leaf/轨道':>9} {'二分 TB/s':>10} {'ring GB/s':>10} {'a2a GB/s':>9}")
    for o in (1.0, 2.0, 4.0):
        r = fabric(a.nodes, a.gpus_per_node, a.nic_gbps, o, a.leaf_ports, not a.no_pxn)
        print(f"{o:>5g}:1 {r['leaves_per_rail']:>9} {r['bisection_tbs']:>10.1f} {r['ring']:>10.1f} {r['a2a']:>9.1f}")


if __name__ == '__main__':
    main()
