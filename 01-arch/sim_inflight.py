"""Little 定律：要吃满显存带宽，每个 SM 需要多少字节在路上（CPU）。对应 GPU-01 第 8 章。

在途字节 = 带宽 × 延迟。把它摊到每个 SM，再除以每个线程一次加载的字节数，
就得到需要同时发出的加载个数。和每 SM 最多 2048 个线程比一比，就知道
单靠多线程（TLP）够不够，还是必须靠每线程多发几条独立加载（ILP）或向量化。

默认参数：
  延迟 480 周期：Luo et al. (arXiv 2402.13499) 在 H800 PCIe 上用 P-chase 测得全局内存约 479 周期
  时钟 1.75 GHz：粗估值，只用来把周期换成纳秒，换自己的卡请用 nvidia-smi -q -d CLOCK 查
  带宽和 SM 数取 common/specs.py
这是数量级估算，用来判断该从哪个方向优化，不是精确预测。

用法：
  python3 sim_inflight.py --gpu h100-sxm
  python3 sim_inflight.py --gpu a100-sxm --lat 466 --clk 1.41
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get  # noqa: E402


def inflight(bw_tbs, lat_cycles, clk_ghz, sms):
    lat_ns = lat_cycles / clk_ghz
    total = bw_tbs * 1e12 * lat_ns * 1e-9
    return lat_ns, total, total / sms


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='h100-sxm')
    p.add_argument('--lat', type=float, default=480, help='全局内存延迟，周期')
    p.add_argument('--clk', type=float, default=1.75, help='SM 时钟，GHz')
    p.add_argument('--threads', type=int, default=2048, help='每 SM 最多线程')
    a = p.parse_args()
    g = get(a.gpu)
    lat_ns, total, per_sm = inflight(g.hbm_tbs, a.lat, a.clk, g.sm)
    print(f'{g.name}: {g.hbm_tbs} TB/s, {g.sm} SM, 延迟 {a.lat:g} 周期 = {lat_ns:.0f} ns')
    print(f'  整卡在途 {total / 1024:.0f} KB，每 SM {per_sm / 1024:.1f} KB')
    for width, name in ((4, 'float'), (8, 'float2'), (16, 'float4')):
        n = per_sm / width
        print(f'  每线程一次读 {width:2d} B（{name:6}）：每 SM 要 {n:5.0f} 个加载在路上，'
              f'占满 {a.threads} 线程的 {n / a.threads:5.1%}')
