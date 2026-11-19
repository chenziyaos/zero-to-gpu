"""sim_pfc.py 的单元测试。python3 test_pfc.py"""
from sim_pfc import incast, pause_max_us


def test_pause_quanta_400g():
    # IEEE 802.1Qbb：PAUSE 以 quanta 计，1 quanta = 512 bit 时间，最多 65535 个；400 Gb/s 时约 83.9 us
    assert abs(pause_max_us(400) - 83.886) < 0.01


def test_fair_share():
    assert incast(8)['per_sender'] == 50 / 8


def test_pause_fraction_grows_with_n():
    f = [incast(n)['pause_frac'] for n in (2, 4, 8, 16)]
    assert f == sorted(f) and f[0] == 0.5


def test_xoff_fills_faster_with_more_senders():
    assert incast(16)['t_fill_us'] < incast(4)['t_fill_us']


def test_dcqcn_protects_victim():
    x = incast(8)
    assert x['victim_dcqcn'] > 5 * x['victim_pfc']


def test_no_incast_no_pause():
    assert incast(1)['pause_frac'] == 0.0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok  ', name)
