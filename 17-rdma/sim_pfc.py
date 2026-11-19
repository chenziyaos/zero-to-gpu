"""incast 下的 PFC 暂停与 DCQCN 对照（CPU）。对应 GPU-17 讲 PFC、ECN 与 DCQCN 的部分。

场景：N 个发送端以线速 R 同时发往同一个出口（出口速率也是 R），交换机该端口缓冲 B，
到达 Xoff 阈值就向上游发 PFC PAUSE，低于 Xon 恢复。RoCE 靠 PFC 做到不丢包，代价有两个：
  1. 被暂停的是整个优先级队列，和 incast 发送端共用上游端口的无关流量（受害流）一起停下，
     这就是队头阻塞；暂停沿链路逐跳向上游扩散时成了 PFC 风暴。
  2. 暂停是开关式的，队列在 Xoff 与 Xon 之间来回振荡，每个周期都要发 PAUSE 帧。
DCQCN 在队列超过 Kmin 时按概率打 ECN 标记，接收端回 CNP，发送端降速；稳态下每个发送端收敛到 R/N，
队列停在 Kmin 附近，几乎不触发 PFC。

记账（流体近似，忽略链路往返的反应时间）：
  队列增长速率 = (N − 1)·R，填满 Xoff 用时 t_fill = Xoff / ((N − 1)·R)
  长期看出口满速，每个 incast 发送端平均只能拿 R/N，上游端口被暂停的时间占比 = 1 − 1/N
  受害流和某个 incast 发送端共用上游端口：PFC 下它跟着被暂停，平均吞吐 ≈ R_v · (1/N)；
  DCQCN 下不触发暂停，受害流拿到上游端口剩下的 R − R/N
  一个 PAUSE 帧最多暂停 65535 个 quanta，1 quanta = 512 bit 时间，400 Gb/s 时上限约 83.9 μs

子命令：
  incast   扫 N，对比 PFC 与 DCQCN 下的暂停占比、受害流吞吐和队列深度
  quanta   打印不同速率下单个 PAUSE 帧的最长暂停时间

用法：
  python3 sim_pfc.py incast --senders 2 4 8 16 32
  python3 sim_pfc.py incast --rate-gbps 400 --buffer-mb 32 --xoff-kb 512
  python3 sim_pfc.py quanta
"""
import argparse

QUANTA_BITS = 512
MAX_QUANTA = 65535


def pause_max_us(rate_gbps):
    """单个 PAUSE 帧的最长暂停时间，微秒。"""
    return MAX_QUANTA * QUANTA_BITS / (rate_gbps * 1e9) * 1e6


def incast(n, rate_gbps=400, xoff_kb=512, kmin_kb=100, victim_gbps=None):
    r = rate_gbps / 8                      # GB/s
    v = r if victim_gbps is None else victim_gbps / 8
    if n <= 1:
        return {'n': n, 't_fill_us': float('inf'), 'pause_frac': 0.0, 'per_sender': r,
                'victim_pfc': v, 'victim_dcqcn': v, 'q_pfc_kb': 0.0, 'q_dcqcn_kb': 0.0}
    t_fill = xoff_kb * 1e3 / ((n - 1) * r * 1e9) * 1e6
    per_sender = r / n
    pause_frac = 1 - 1 / n
    return {
        'n': n,
        't_fill_us': t_fill,
        'pause_frac': pause_frac,
        'per_sender': per_sender,
        'victim_pfc': min(v, r) * (1 - pause_frac),
        'victim_dcqcn': min(v, r - per_sender),
        'q_pfc_kb': xoff_kb,
        'q_dcqcn_kb': kmin_kb,
    }


def main():
    ap = argparse.ArgumentParser(description='incast 下的 PFC 暂停与 DCQCN 对照')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('incast')
    p.add_argument('--senders', type=int, nargs='+', default=[2, 4, 8, 16, 32])
    p.add_argument('--rate-gbps', type=float, default=400)
    p.add_argument('--xoff-kb', type=float, default=512, help='PFC Xoff 阈值（按端口、按优先级）')
    p.add_argument('--kmin-kb', type=float, default=100, help='DCQCN 开始打 ECN 标记的队列深度')
    sub.add_parser('quanta')
    a = ap.parse_args()
    if a.cmd == 'quanta':
        for g in (100, 200, 400, 800):
            print(f'{g:>4} Gb/s  单个 PAUSE 帧最长暂停 {pause_max_us(g):7.1f} us')
        return
    r = a.rate_gbps / 8
    print(f'出口 {a.rate_gbps:g} Gb/s（{r:.0f} GB/s），Xoff {a.xoff_kb:g} KB，DCQCN Kmin {a.kmin_kb:g} KB；受害流与一个 incast 发送端共用上游端口')
    print(f"{'N':>3} {'填满Xoff us':>11} {'每发送端 GB/s':>13} {'暂停占比':>8} {'受害流 PFC':>10} {'受害流 DCQCN':>12} {'队列 PFC/DCQCN KB':>18}")
    for n in a.senders:
        x = incast(n, a.rate_gbps, a.xoff_kb, a.kmin_kb)
        print(f"{n:>3} {x['t_fill_us']:>11.2f} {x['per_sender']:>13.1f} {x['pause_frac']:>8.0%} "
              f"{x['victim_pfc']:>10.1f} {x['victim_dcqcn']:>12.1f} {x['q_pfc_kb']:>9.0f}/{x['q_dcqcn_kb']:.0f}")


if __name__ == '__main__':
    main()
