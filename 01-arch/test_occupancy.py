"""sim_occupancy 的单元测试。python3 -m pytest test_occupancy.py 或 python3 test_occupancy.py"""
from sim_occupancy import occupancy


def test_best_practices_guide_example():
    # CUDA C++ Best Practices Guide：CC 7.0，37 寄存器
    # 128 线程块驻留 12 块，occupancy 75%；320 线程块只驻留 4 块，occupancy 63%
    r = occupancy('7.0', 128, 37)
    assert r['blocks'] == 12 and abs(r['occ'] - 0.75) < 1e-9
    r = occupancy('7.0', 320, 37)
    assert r['blocks'] == 4 and round(r['occ'], 2) == 0.62


def test_32_regs_is_full():
    # 64K 寄存器 / 2048 线程 = 32，32 寄存器时满 occupancy
    for cc in ('8.0', '9.0', '10.0'):
        assert occupancy(cc, 256, 32)['occ'] == 1.0


def test_hopper_smem():
    # H100：每块 100 KB 共享内存，加 1 KB 保留后 228 KB 只放得下 2 块
    r = occupancy('9.0', 128, 32, 100)
    assert r['blocks'] == 2 and r['limiter'] == ['shared_mem']
    # 每块 227 KB 是上限，只能放 1 块
    assert occupancy('9.0', 128, 32, 227)['blocks'] == 1


def test_block_limit():
    # 32 线程的小块：H100 每 SM 最多 32 块，只有 32 个 warp，occupancy 50%
    r = occupancy('9.0', 32, 16)
    assert r['blocks'] == 32 and r['occ'] == 0.5 and 'blocks' in r['limiter']


def test_consumer_blackwell():
    # CC 12.0 每 SM 48 warp、100 KB 共享内存
    r = occupancy('12.0', 256, 32, 48)
    assert r['blocks'] == 2 and r['warps'] == 16


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f()
            print('ok', k)
