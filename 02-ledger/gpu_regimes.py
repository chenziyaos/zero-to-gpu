"""三类瓶颈的现场实验（GPU）。对应 GPU-02 第 4、5 章。

repeat：Horace He（2022）的实验。对同一个张量连续做 r 次 x = x * 2（编译成一个 kernel），
        r 从 1 扫到 256。r 小时时间几乎不变（带宽受限），r 大时时间随 r 线性涨（算力受限），拐点就在本机的非 Tensor Core 拐点附近。
batch： 把一个小 MLP 的 batch 从 1 翻倍到 4096，看时间什么时候开始跟着涨，前面那段平台就是开销受限区。
fuse：  x.cos().cos() 分开跑和 torch.compile 融合后对比，融合后显存流量减半。

用法：
  python3 gpu_regimes.py repeat
  python3 gpu_regimes.py batch
  python3 gpu_regimes.py fuse
"""
import argparse

import torch


def bench(fn, iters=50):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3   # 微秒


def repeat():
    n = 1 << 26
    x = torch.randn(n, device='cuda')
    print(f'{n} 个 FP32 元素，每次读 4 字节写 4 字节；r 次乘法编译成一个 kernel')
    print(f'{"r":>5} {"us":>9} {"GB/s":>8} {"GFLOPS":>9}')
    for r in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        def f(x, r=r):
            for _ in range(r):
                x = x * 2
            return x
        g = torch.compile(f)
        t = bench(lambda: g(x))
        print(f'{r:5d} {t:9.1f} {8 * n / t / 1e3:8.0f} {r * n / t / 1e3:9.0f}')


def batch():
    d = 1024
    mlp = torch.nn.Sequential(*[torch.nn.Linear(d, d) for _ in range(8)]).cuda().half()
    print(f'8 层 {d}×{d} 线性层，FP16，eager')
    prev = None
    for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        x = torch.randn(bs, d, device='cuda', dtype=torch.half)
        with torch.no_grad():
            t = bench(lambda: mlp(x))
        tag = '' if prev is None else f'×{t / prev:.2f}'
        print(f'  batch {bs:5d}: {t:8.1f} us {tag}')
        prev = t


def fuse():
    x = torch.randn(1 << 26, device='cuda')
    eager = lambda: x.cos().cos()  # noqa: E731
    comp = torch.compile(lambda t: t.cos().cos())
    te, tc = bench(eager), bench(lambda: comp(x))
    print(f'eager {te:.1f} us，compile 融合 {tc:.1f} us，加速 {te / tc:.2f} 倍（理论约 2 倍）')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('exp', choices=['repeat', 'batch', 'fuse'])
    {'repeat': repeat, 'batch': batch, 'fuse': fuse}[p.parse_args().exp]()
