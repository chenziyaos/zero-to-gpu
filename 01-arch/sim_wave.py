"""Wave 量化：线程块个数不是 SM 数的整数倍时，最后一波吃不满（CPU）。对应 GPU-01 第 2 章。

例子来自 NVIDIA Deep Learning Performance 文档 Matrix Multiplication Background 的 Wave Quantization 一节：
A100 有 108 个 SM，强制 256x128 分块、每 SM 一块，M=2304 时 N=1536 正好 108 块一波，
N 再加一个分块宽度就变成两波，第二波只有 9 块。

用法：
  python3 sim_wave.py --sms 108 --m 2304 --tile-m 256 --tile-n 128 --n 1408 1536 1664 3072
  python3 sim_wave.py --sms 132 --m 4096 --n 4096 --tile-m 128 --tile-n 256 --per-sm 1
"""
import argparse
import math


def waves(m, n, tm, tn, sms, per_sm=1):
    tiles = math.ceil(m / tm) * math.ceil(n / tn)
    slots = sms * per_sm
    w = math.ceil(tiles / slots)
    eff = tiles / (w * slots)
    return tiles, w, eff


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sms', type=int, default=108)
    p.add_argument('--per-sm', type=int, default=1, help='每 SM 同时驻留的块数')
    p.add_argument('--m', type=int, default=2304)
    p.add_argument('--n', type=int, nargs='+', default=[1408, 1536, 1664, 3072, 3200])
    p.add_argument('--tile-m', type=int, default=256)
    p.add_argument('--tile-n', type=int, default=128)
    a = p.parse_args()
    print(f'{a.sms} SM x {a.per_sm} 块/SM，分块 {a.tile_m}x{a.tile_n}，M={a.m}')
    for n in a.n:
        t, w, e = waves(a.m, n, a.tile_m, a.tile_n, a.sms, a.per_sm)
        print(f'  N={n:5d}: {t:4d} 块, {w} 波, SM 利用率上限 {e:6.1%}')
