"""Roofline 计算器：算术强度、拐点和 kernel 理论上限（CPU）。对应 GPU-02 第 2 至 6 章。

核心只有一行：可达算力 = min(峰值算力, 算术强度 × 带宽)。
算术强度 = 浮点运算次数 / 和显存之间搬运的字节数，单位 FLOP/字节。
拐点 = 峰值算力 / 带宽，强度低于拐点受带宽限制，高于拐点受算力限制。

子命令：
  ridge   各卡的标称拐点和实测拐点（MAMF 与带宽九成口径）
  ops     一组常见算子的强度、上限和瓶颈
  gemm    任意形状 GEMM 的账，含分块后的片上强度
  decode  解码阶段线性层：扫 batch，看什么时候从带宽受限变成算力受限
  attn    注意力的账：prefill 和 decode，MHA 与 GQA
  mfu     训练的账：每 token 运算量 6N + 12LHQT，目标 MFU 需要的吞吐

用法：
  python3 sim_roofline.py ridge
  python3 sim_roofline.py ops --gpu h100-sxm
  python3 sim_roofline.py gemm --m 4096 --n 4096 --k 4096
  python3 sim_roofline.py decode --d 4096 --f 14336
  python3 sim_roofline.py attn --ctx 8192 --heads 32 --kv-heads 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# 实测矩阵乘上限，BF16 稠密，单位 TFLOPS。
# 来源：stas00/ml-engineering，compute/accelerator/README.md 的 Maximum Achievable Matmul FLOPS 表（核对于 2026-11-02）
#   MAMF：短时满频下 mamf-finder 扫形状得到的最好值；MSMF：持续负载、受功耗墙限制后的值
MAMF = {
    'a100-sxm': {'mamf': 271, 'msmf': None},
    'h100-sxm': {'mamf': 853, 'msmf': 786},
    'b200': {'mamf': 1737, 'msmf': 1439},
}
BW_EFF = 0.90   # 实测拷贝带宽约为标称值的九成，见 GPU-01 第 8.2 节


def attainable(intensity, peak_tflops, bw_tbs):
    """Roofline：给定强度下的可达算力，TFLOPS。"""
    return min(peak_tflops, intensity * bw_tbs)


def bound(intensity, peak_tflops, bw_tbs):
    return '算力' if intensity >= peak_tflops / bw_tbs else '带宽'


def ledger(flops, nbytes, peak_tflops, bw_tbs):
    """一个 kernel 的账：强度、上限、两项时间（微秒）和瓶颈。"""
    t_math = flops / (peak_tflops * 1e12) * 1e6
    t_mem = nbytes / (bw_tbs * 1e12) * 1e6
    i = flops / nbytes
    return {'flops': flops, 'bytes': nbytes, 'intensity': i,
            'tflops': attainable(i, peak_tflops, bw_tbs),
            't_math_us': t_math, 't_mem_us': t_mem, 't_us': max(t_math, t_mem),
            'bound': '算力' if t_math >= t_mem else '带宽'}


# ---------- 算子记账 ----------

def gemm(m, n, k, b_in=2, b_out=2):
    """C[m,n] = A[m,k] @ B[k,n]：每个输入只读一次、输出写一次的理想账。"""
    return 2 * m * n * k, (m * k + k * n) * b_in + m * n * b_out


def tile_intensity(bm, bn, b_in=2):
    """分块 GEMM 的片上强度：每个 k 步读 (bm + bn) × bk 个元素，做 2 × bm × bn × bk 次运算。"""
    return 2 * bm * bn / ((bm + bn) * b_in)


def elementwise(n, flops_per_elem, n_in=1, n_out=1, b=2):
    """逐元素：每个元素读 n_in 个张量、写 n_out 个张量。"""
    return n * flops_per_elem, n * (n_in + n_out) * b


def linear(tokens, d, f, b_w=2, b_a=2):
    """线性层 y[tokens,f] = x[tokens,d] @ W[d,f]，权重字节 b_w，激活字节 b_a。"""
    return 2 * tokens * d * f, d * f * b_w + tokens * d * b_a + tokens * f * b_a


def attn_decode(batch, ctx, heads, kv_heads, head_dim, b_kv=2):
    """解码一步的注意力：每条序列读一遍自己的 K、V cache。
    运算：QK^T 和 PV 各 2 × ctx × head_dim 次，按 query 头计；字节：K 和 V 按 kv 头计。"""
    flops = 4 * batch * ctx * heads * head_dim
    nbytes = 2 * batch * ctx * kv_heads * head_dim * b_kv
    return flops, nbytes


def attn_prefill(batch, seq, heads, head_dim, causal=True, b=2):
    """FlashAttention 式 prefill：Q、K、V 各读一次，O 写一次，S×S 的分数矩阵不落显存。"""
    flops = 4 * batch * seq * seq * heads * head_dim // (2 if causal else 1)
    nbytes = 4 * batch * seq * heads * head_dim * b
    return flops, nbytes


# ---------- 子命令 ----------

def cmd_ridge(a):
    print(f'{"GPU":22} {"BF16":>6} {"HBM":>6} {"拐点":>6} │ {"MSMF":>6} {"带宽×0.9":>8} {"实测拐点":>8}')
    for key, g in GPUS.items():
        if not g.bf16_tflops:
            continue
        m = MAMF.get(key, {})
        eff = m.get('msmf') or m.get('mamf')
        bw = g.hbm_tbs * BW_EFF
        tail = f'{eff:6g} {bw:8.2f} {eff * 1e12 / (bw * 1e12):8.0f}' if eff else f'{"-":>6} {bw:8.2f} {"-":>8}'
        print(f'{g.name:22} {g.bf16_tflops:6g} {g.hbm_tbs:6g} {g.ridge():6.0f} │ {tail}')
    print('单位：TFLOPS、TB/s、FLOP/字节。MSMF 缺省时用 MAMF。')


def ops_table(gpu):
    g = get(gpu)
    peak, bw = g.bf16_tflops, g.hbm_tbs
    n = 64 * 2 ** 20
    rows = [
        ('ReLU，64M 元素', elementwise(n, 1, 1, 1)),
        ('a + b，64M 元素', elementwise(n, 1, 2, 1)),
        ('LayerNorm，8192×8192', elementwise(8192 * 8192, 8, 1, 1)),
        ('解码线性层，1 token', linear(1, 4096, 4096)),
        ('解码线性层，64 token', linear(64, 4096, 4096)),
        ('V100 指南：线性层 batch 512', linear(512, 1024, 4096)),
        ('GEMM 4096³', gemm(4096, 4096, 4096)),
        ('注意力 decode，MHA，8K 上下文', attn_decode(1, 8192, 32, 32, 128)),
        ('注意力 decode，GQA 8 组', attn_decode(1, 8192, 32, 4, 128)),
        ('注意力 prefill，8K，因果', attn_prefill(1, 8192, 32, 128)),
    ]
    return g, [(name, ledger(f, b, peak, bw)) for name, (f, b) in rows]


def cmd_ops(a):
    g, rows = ops_table(a.gpu)
    print(f'{g.name}：BF16 {g.bf16_tflops:g} TFLOPS，HBM {g.hbm_tbs:g} TB/s，拐点 {g.ridge():.0f} FLOP/字节')
    print(f'  {"算子":30} {"强度":>8} {"上限 TFLOPS":>11} {"占峰值":>7}  瓶颈')
    for name, r in rows:
        print(f'  {name:30} {r["intensity"]:8.2f} {r["tflops"]:11.1f} {r["tflops"] / g.bf16_tflops:7.1%}  {r["bound"]}')


def cmd_gemm(a):
    g = get(a.gpu)
    f, b = gemm(a.m, a.n, a.k, a.bytes, a.bytes)
    r = ledger(f, b, g.bf16_tflops, g.hbm_tbs)
    print(f'{g.name} GEMM M={a.m} N={a.n} K={a.k}，每元素 {a.bytes} 字节')
    print(f'  运算 {f / 1e9:.1f} GFLOP，搬运 {b / 2 ** 20:.1f} MiB，强度 {r["intensity"]:.0f} FLOP/字节（拐点 {g.ridge():.0f}）')
    print(f'  算力时间 {r["t_math_us"]:.1f} us，访存时间 {r["t_mem_us"]:.1f} us → {r["bound"]}受限，'
          f'理论下限 {r["t_us"]:.1f} us')
    for bm, bn in ((64, 64), (128, 128), (128, 256), (256, 256)):
        print(f'  分块 {bm}×{bn}：片上强度 {tile_intensity(bm, bn, a.bytes):6.1f} FLOP/字节（对 L2 和共享内存而言）')


def cmd_decode(a):
    g = get(a.gpu)
    print(f'{g.name} 线性层 d={a.d} f={a.f}，权重 {a.wbytes} 字节/元素；拐点 {g.ridge():.0f}')
    print(f'  {"tokens":>6} {"强度":>7} {"上限 TFLOPS":>11} {"占峰值":>7} {"时间 us":>8}  瓶颈')
    for t in (1, 8, 32, 64, 128, 256, 512, 1024, 4096):
        r = ledger(*linear(t, a.d, a.f, a.wbytes), g.bf16_tflops, g.hbm_tbs)
        print(f'  {t:6d} {r["intensity"]:7.1f} {r["tflops"]:11.1f} {r["tflops"] / g.bf16_tflops:7.1%} {r["t_us"]:8.1f}  {r["bound"]}')
    crit = g.ridge() * a.wbytes / 2
    print(f'  token 数远小于 d、f 时强度约等于 token 数 × 2 / 权重字节，越过拐点约需 {crit:.0f} 个 token')


def cmd_attn(a):
    g = get(a.gpu)
    grp = a.heads // a.kv_heads
    print(f'{g.name} 注意力：{a.heads} 个 query 头，{a.kv_heads} 个 kv 头（每组 {grp}），head_dim {a.head_dim}')
    for ctx in (1024, 8192, a.ctx):
        r = ledger(*attn_decode(a.batch, ctx, a.heads, a.kv_heads, a.head_dim), g.bf16_tflops, g.hbm_tbs)
        print(f'  decode ctx={ctx:6d}：强度 {r["intensity"]:5.1f}，KV 读 {r["bytes"] / 2 ** 20:7.1f} MiB，'
              f'{r["t_us"]:7.1f} us，{r["bound"]}受限')
    r = ledger(*attn_prefill(a.batch, a.ctx, a.heads, a.head_dim), g.bf16_tflops, g.hbm_tbs)
    print(f'  prefill seq={a.ctx}：强度 {r["intensity"]:7.0f}，{r["bound"]}受限')
    print(f'  decode 的强度只取决于每组 query 头数，和 batch、上下文长度无关')


def train_flops_per_token(params, layers, heads, head_dim, seq):
    """PaLM 附录 B：每个训练 token 的模型运算量 6N + 12·L·H·Q·T（注意力项按非因果计）。"""
    return 6 * params, 12 * layers * heads * head_dim * seq


def mfu(tokens_per_s, n_gpu, peak_tflops, dense, attn):
    """MFU = 实测每秒 token 数 × 每 token 运算量 / (卡数 × 峰值)。重算的运算不计入。"""
    return tokens_per_s * (dense + attn) / (n_gpu * peak_tflops * 1e12)


def cmd_mfu(a):
    g = get(a.gpu)
    dense, attn = train_flops_per_token(a.params, a.layers, a.heads, a.head_dim, a.seq)
    print(f'{g.name} 训练账：N={a.params / 1e9:g}B，L={a.layers}，H={a.heads}，Q={a.head_dim}，T={a.seq}')
    print(f'  每 token 运算：6N = {dense / 1e9:.1f} GFLOP，注意力 12LHQT = {attn / 1e9:.1f} GFLOP（占 {attn / (dense + attn):.0%}）')
    for target in (0.3, 0.4, 0.5):
        need = target * g.bf16_tflops * 1e12 / (dense + attn)
        print(f'  MFU {target:.0%} 需要每卡每秒 {need:8.0f} 个 token')
    if a.tps:
        print(f'  实测每卡每秒 {a.tps:g} 个 token → MFU {mfu(a.tps, 1, g.bf16_tflops, dense, attn):.1%}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('ridge')
    q = sp.add_parser('ops'); q.add_argument('--gpu', default='h100-sxm')
    q = sp.add_parser('gemm'); q.add_argument('--gpu', default='h100-sxm')
    q.add_argument('--m', type=int, default=4096); q.add_argument('--n', type=int, default=4096)
    q.add_argument('--k', type=int, default=4096); q.add_argument('--bytes', type=int, default=2)
    q = sp.add_parser('decode'); q.add_argument('--gpu', default='h100-sxm')
    q.add_argument('--d', type=int, default=4096); q.add_argument('--f', type=int, default=14336)
    q.add_argument('--wbytes', type=float, default=2, help='权重每元素字节：BF16 2，FP8/INT8 1，4 比特 0.5')
    q = sp.add_parser('attn'); q.add_argument('--gpu', default='h100-sxm')
    q.add_argument('--batch', type=int, default=1); q.add_argument('--ctx', type=int, default=32768)
    q.add_argument('--heads', type=int, default=32); q.add_argument('--kv-heads', type=int, default=8)
    q.add_argument('--head-dim', type=int, default=128)
    q = sp.add_parser('mfu'); q.add_argument('--gpu', default='h100-sxm')
    q.add_argument('--params', type=float, default=8.03e9); q.add_argument('--layers', type=int, default=32)
    q.add_argument('--heads', type=int, default=32); q.add_argument('--head-dim', type=int, default=128)
    q.add_argument('--seq', type=int, default=8192); q.add_argument('--tps', type=float, default=0, help='实测每卡每秒 token 数')
    a = p.parse_args()
    {'ridge': cmd_ridge, 'ops': cmd_ops, 'gemm': cmd_gemm, 'decode': cmd_decode, 'attn': cmd_attn,
     'mfu': cmd_mfu}[a.cmd](a)
