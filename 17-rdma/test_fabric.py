"""sim_fabric.py 的单元测试。python3 test_fabric.py"""
from sim_fabric import fabric, leaf_layout


def test_nic_rate():
    # 400 Gb/s 网卡单向 50 GB/s（ConnectX-7 规格页标称 400Gb/s 端口）
    assert fabric(32)['nic'] == 50.0


def test_single_leaf_per_rail_is_nonblocking():
    # 32 台机器、64 口 leaf 1:1：每条轨道一台 leaf 就够，轨道内流量不上 spine
    r = fabric(32)
    assert r['leaves_per_rail'] == 1 and r['ring'] == 50.0 and r['a2a'] == 50.0


def test_rail_ring_ignores_oversub():
    # 轨道对齐的 ring 每台 leaf 只有两条边离开 leaf，2:1 收敛下仍拿满网卡
    assert fabric(128, oversub=2.0)['ring'] == 50.0


def test_a2a_degrades_with_oversub():
    a = [fabric(128, oversub=o)['a2a'] for o in (1.0, 2.0, 4.0)]
    assert a[0] == 50.0 and a[0] > a[1] > a[2]


def test_pxn_keeps_traffic_in_rail():
    # 关掉 PXN 后跨轨道目的地都上 spine，2:1 收敛下 all-to-all 更差
    assert fabric(128, oversub=2.0, pxn=False)['a2a'] < fabric(128, oversub=2.0)['a2a']


def test_leaf_layout():
    assert leaf_layout(128, 64, 1.0) == (32, 32, 4)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok  ', name)
