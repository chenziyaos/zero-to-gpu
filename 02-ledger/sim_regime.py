"""三类瓶颈的判别和融合收益的账（CPU）。对应 GPU-02 第 4、5 章。

  regime  输入 batch 翻倍前后的实测时间，判断是开销、带宽还是算力受限
  fuse    n 个逐元素算子分开跑和融合成一个 kernel，显存流量和时间下限各是多少
  launch  每个 kernel 的固定开销占比：小算子为什么被 launch 和 Python 拖住

用法：
  python3 sim_regime.py regime --t1 1.00 --t2 1.10
  python3 sim_regime.py fuse --n 4 --elems 67108864
  python3 sim_regime.py launch --us 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get  # noqa: E402


def regime(t1, t2):
    """batch 翻倍：时间几乎不变说明 GPU 在等主机侧（开销受限）；接近翻倍说明 GPU 已被喂满。
    阈值取 Horace He（2022）的说法：翻倍只多 10% 左右，基本可以断定是开销受限。"""
    r = t2 / t1
    if r < 1.3:
        return r, '开销受限：GPU 在等 Python、框架调度或 kernel launch，先上 CUDA Graph 或 torch.compile'
    if r < 1.8:
        return r, '混合区：一部分 kernel 已经喂满，一部分还被开销拖着，用 nsys 看时间线上的空隙'
    return r, '已喂满：再看 ncu 的 SOL，决定是带宽受限（去融合、降精度）还是算力受限（去用 Tensor Core）'


def fuse(n_ops, elems, b=2, bw_tbs=3.35):
    """n 个一元逐元素算子串联。分开跑每个算子读一次写一次；融合后只读一次写一次。"""
    sep = n_ops * 2 * elems * b
    fused = 2 * elems * b
    return sep, fused, sep / (bw_tbs * 1e12) * 1e6, fused / (bw_tbs * 1e12) * 1e6


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    q = sp.add_parser('regime'); q.add_argument('--t1', type=float, required=True); q.add_argument('--t2', type=float, required=True)
    q = sp.add_parser('fuse'); q.add_argument('--n', type=int, default=4); q.add_argument('--elems', type=int, default=64 * 2 ** 20)
    q.add_argument('--gpu', default='h100-sxm')
    q = sp.add_parser('launch'); q.add_argument('--us', type=float, default=5, help='每个 kernel 的固定开销，微秒')
    q.add_argument('--gpu', default='h100-sxm')
    a = p.parse_args()
    if a.cmd == 'regime':
        r, msg = regime(a.t1, a.t2)
        print(f'batch 翻倍，时间变成 {r:.2f} 倍 → {msg}')
    elif a.cmd == 'fuse':
        g = get(a.gpu)
        sep, fused, t_sep, t_fused = fuse(a.n, a.elems, bw_tbs=g.hbm_tbs)
        print(f'{g.name}，{a.n} 个逐元素算子，{a.elems / 2 ** 20:.0f}M 个 BF16 元素')
        print(f'  分开跑：读写 {sep / 2 ** 20:6.0f} MiB，下限 {t_sep:7.1f} us')
        print(f'  融合后：读写 {fused / 2 ** 20:6.0f} MiB，下限 {t_fused:7.1f} us，快 {t_sep / t_fused:.1f} 倍')
    else:
        g = get(a.gpu)
        print(f'{g.name}，每个 kernel 固定开销 {a.us:g} us；BF16 逐元素 a + b 的有效时间与开销占比')
        for n in (2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24, 2 ** 28):
            t = 3 * n * 2 / (g.hbm_tbs * 1e12) * 1e6
            print(f'  {n:>10} 元素：搬运 {t:9.2f} us，开销占 {a.us / (a.us + t):6.1%}')
