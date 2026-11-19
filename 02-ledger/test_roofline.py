"""sim_roofline 和 sim_regime 的单元测试。python3 -m pytest test_roofline.py 或 python3 test_roofline.py"""
from sim_roofline import attainable, gemm, linear, elementwise, attn_decode, tile_intensity, ledger, MAMF
from sim_regime import regime, fuse


def test_ridge_h100():
    # How To Scale Your Model 第 1 部分：H100 BF16 稠密 989 TFLOPS / 3.35 TB/s ≈ 295
    assert round(989 / 3.35) == 295


def test_v100_guide_table():
    # NVIDIA GPU Performance Background 表 1：线性层 4096 输出、1024 输入、batch 512 强度 315；batch 1 约 1；ReLU 0.25
    f, b = linear(512, 1024, 4096)
    assert round(f / b) == 315
    f, b = linear(1, 1024, 4096)
    assert round(f / b) == 1
    f, b = elementwise(10 ** 6, 1, 1, 1)
    assert f / b == 0.25


def test_dot_product_half():
    # 点积强度趋近 1/2：N 次乘加 = 2N 次运算，读 4N 字节
    n = 10 ** 6
    assert abs(2 * n / (4 * n) - 0.5) < 1e-12


def test_gemm_square():
    # 方阵 GEMM 强度 = 2N³ / (6N²) = N/3
    f, b = gemm(4096, 4096, 4096)
    assert abs(f / b - 4096 / 3) < 1e-6


def test_tile():
    # 分块强度 bm·bn/(bm+bn)，BF16 时 2 字节抵掉 2 倍运算
    assert tile_intensity(128, 128) == 64
    assert tile_intensity(256, 128) == 2 * 256 * 128 / (384 * 2)


def test_roofline_shape():
    assert attainable(1, 989, 3.35) == 3.35
    assert attainable(10000, 989, 3.35) == 989


def test_decode_intensity_equals_group():
    # decode 注意力强度 = query 头数 / kv 头数，与上下文长度无关
    for ctx in (1024, 32768):
        f, b = attn_decode(4, ctx, 32, 8, 128)
        assert f / b == 4
    f, b = attn_decode(1, 8192, 32, 32, 128)
    assert f / b == 1


def test_mamf_ratio():
    # H100 SXM MAMF 853、MSMF 786，相对 989 约 86% 与 79%
    assert round(MAMF['h100-sxm']['mamf'] / 989, 3) == 0.862
    assert round(MAMF['h100-sxm']['msmf'] / 989, 3) == 0.795


def test_regime_and_fuse():
    assert regime(1.0, 1.1)[1].startswith('开销')
    assert regime(1.0, 1.95)[1].startswith('已喂满')
    sep, fused, *_ = fuse(2, 1000)
    assert sep == 2 * fused      # Horace He：x.cos().cos() 融合后显存流量减半


def test_palm_mfu_mtnlg():
    # PaLM 附录 B：MT-NLG 530B 在 2240 张 A100 上每秒 65.43K token，不计注意力 MFU 29.7%
    from sim_roofline import mfu
    assert abs(mfu(65.43e3, 2240, 312, 6 * 530e9, 0) - 0.297) < 0.001


def test_ledger_bound():
    r = ledger(*gemm(4096, 4096, 4096), 989, 3.35)
    assert r['bound'] == '算力' and r['t_math_us'] > r['t_mem_us']


if __name__ == '__main__':
    import sys
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            try:
                v(); print('ok  ', k)
            except AssertionError as e:
                fails += 1; print('FAIL', k, e)
    sys.exit(fails)
