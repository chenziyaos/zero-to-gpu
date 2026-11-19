"""实测本机的矩阵乘上限（MAMF 式扫描）和显存拷贝带宽，得到实测 Roofline 的两根线（GPU）。对应 GPU-02 第 3 章。

做法参考 stas00/ml-engineering 的 mamf-finder：在一组形状上反复跑 torch.matmul，取最好的那个形状的 TFLOPS。
默认只扫一小组形状，几分钟跑完；要逼近论文表里的数，用 --full 扫更大的范围，并锁频后多跑几轮。
持续负载下的数（MSMF）要用 --seconds 让每个形状跑足够久，功耗墙生效后再计时。

用法：
  python3 gpu_mamf.py                  # 快速扫描，BF16
  python3 gpu_mamf.py --dtype fp16 --full
  python3 gpu_mamf.py --seconds 30     # 每个形状先烧 30 秒再计时，近似 MSMF
"""
import argparse
import itertools
import time

import torch


def bench(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


def matmul_tflops(m, n, k, dtype, iters, burn_s):
    a = torch.randn(m, k, device='cuda', dtype=dtype)
    b = torch.randn(k, n, device='cuda', dtype=dtype)
    c = torch.empty(m, n, device='cuda', dtype=dtype)
    fn = lambda: torch.matmul(a, b, out=c)  # noqa: E731
    t0 = time.time()
    while time.time() - t0 < burn_s:
        fn()
    t = bench(fn, iters)
    return 2 * m * n * k / t / 1e12


def copy_tbs(nbytes, iters=50):
    x = torch.empty(nbytes // 2, device='cuda', dtype=torch.bfloat16)
    y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), iters)
    return 2 * nbytes / t / 1e12   # 读一遍写一遍


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    p.add_argument('--full', action='store_true')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--seconds', type=float, default=0, help='每个形状先持续跑多少秒再计时')
    a = p.parse_args()
    dt = {'bf16': torch.bfloat16, 'fp16': torch.float16}[a.dtype]
    prop = torch.cuda.get_device_properties(0)
    print(f'# {prop.name}, {prop.multi_processor_count} SM, torch {torch.__version__}, {a.dtype}')
    sizes = range(1024, 20481, 1024) if a.full else (4096, 6144, 8192, 12288, 16384)
    best = (0, None)
    for m, n, k in itertools.product(sizes, sizes, (4096, 8192) if not a.full else sizes):
        try:
            tf = matmul_tflops(m, n, k, dt, a.iters, a.seconds)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        if tf > best[0]:
            best = (tf, (m, n, k))
            print(f'  new best {tf:7.1f} TFLOPS at M={m} N={n} K={k}')
    bw = max(copy_tbs(s) for s in (1 << 28, 1 << 30, 1 << 31))
    print(f'实测矩阵乘上限 {best[0]:.0f} TFLOPS（形状 {best[1]}），实测拷贝带宽 {bw:.2f} TB/s，实测拐点 {best[0] / bw:.0f} FLOP/字节')
    print('把这两个数填进 sim_roofline.py 的 MAMF 表，或用 --peak/--bw 参数画自己的 Roofline')
