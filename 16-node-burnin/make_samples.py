"""生成 4 份样例观测 JSON：验收各层结论的演示用例。
对应 GPU-16 第 1 章。

四个样例：
  1. healthy_pass.json      - 八卡全过，六类全 PASS
  2. fail_gpu5_slow.json    - 单卡算力不足，FAIL 单条线
  3. fail_link_xid.json     - 慢链 + 致命 Xid79，多条 FAIL + 聚合演示
  4. warn_temp_short_burn.json - 温度撞线 + burn-in 不足，WARN 演示

每份包含：compute（BF16/FP32 TFLOPS）、memory（HBM 带宽/容量/ECC/温度）、
links（NVLink P2P 单向/all-reduce busbw）、pcie、xids、burnin

参考值取自 H100-SXM：989 TF、3.35 TB/s、80 GB、900 GB/s 双向 NVLink (450 GB/s 单向)、
128 GB/s 双向 PCIe。
"""
import json
import os


def sample_healthy_pass():
    """健康机：八卡均匀合成样本（~780 TFLOPS per GPU）、无错误、24h burn-in。"""
    return {
        'model': 'H100-SXM',
        'compute': [
            {'gpu': i, 'bf16_tflops': 780 + (i - 4) * 1.5, 'fp32_tflops': 56.8 + (i - 4) * 0.1}
            for i in range(8)
        ],
        'memory': [
            {'gpu': i, 'hbm_bw_tbs': 2.92 + (i % 2) * 0.05, 'hbm_gb': 79.5, 'ecc_dbe': 0, 'max_temp_c': 75 + (i % 3)}
            for i in range(8)
        ],
        'links': [
            {'gpu': i, 'min_nvlink_gbs': 385.0 + (i * 0.3), 'busbw_gbs': 360.0}
            for i in range(8)
        ],
        'pcie': [
            {'gpu': i, 'link_gen': 5, 'link_width': 16, 'bw_gbs': 50.5}
            for i in range(8)
        ],
        'xids': [],
        'burnin': {
            'hours': 24.5,
            'cards_dropped': 0,
            'ecc_dbe_total': 0,
            'throttle_events': 0,
            'nccl_failures': 0
        }
    }


def sample_fail_gpu5_slow():
    """单卡算力不足：GPU5 的 BF16 TFLOPS 降到 650，低于 700 合格线。"""
    data = sample_healthy_pass()
    data['compute'][5]['bf16_tflops'] = 650.0  # FAIL 线是 700；另测极差 780/650=1.2 > 1.05 → also FAIL
    return data


def sample_fail_link_xid():
    """链路慢 + 致命 Xid：GPU2 NVLink P2P 单向 320 GB/s（< 360 PASS 线）+ Xid 79（致命）。"""
    data = sample_healthy_pass()
    data['links'][2]['min_nvlink_gbs'] = 320.0  # < 360 (0.80×450) PASS 线，属 FAIL 区
    data['links'][0]['busbw_gbs'] = 300.0       # < 360 (0.80×450) PASS 线，属 FAIL 区
    data['xids'] = [
        {'timestamp': '2026-11-05T10:30:15Z', 'code': 79, 'gpu': 2}  # 致命 Xid
    ]
    data['burnin']['ecc_dbe_total'] = 0
    return data


def sample_warn_temp_short_burn():
    """温度撞警告区 + burn-in 不足：GPU0 温度 80°C、burn-in 只跑了 12h。"""
    data = sample_healthy_pass()
    data['memory'][0]['max_temp_c'] = 80.0  # 78-83 WARN 区
    data['burnin']['hours'] = 12.0           # 低于 24h WARN 线
    return data


def main():
    samples = {
        'healthy_pass.json': sample_healthy_pass(),
        'fail_gpu5_slow.json': sample_fail_gpu5_slow(),
        'fail_link_xid.json': sample_fail_link_xid(),
        'warn_temp_short_burn.json': sample_warn_temp_short_burn(),
    }

    os.makedirs('samples', exist_ok=True)

    for name, data in samples.items():
        path = os.path.join('samples', name)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ {path}")


if __name__ == '__main__':
    main()
