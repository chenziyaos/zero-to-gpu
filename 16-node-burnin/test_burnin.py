"""验收判定引擎测试：覆盖六条合格线边界与聚合逻辑。

测试清单（17 个用例）：
  ✓ 计算线边界：BF16 过线 + 低于线 + 卡间极差
  ✓ 显存线边界：HBM 带宽、容量、温度
  ✓ ECC 双位零容忍
  ✓ NVLink 逐链线 + busbw 线
  ✓ PCIe 代际与协商
  ✓ 致命 Xid vs 告警 Xid
  ✓ burn-in 时长 + 四类故障
  ✓ 整机聚合：PASS / CONDITIONAL / FAIL
"""
import json
import subprocess
import sys
import os

# 测试数据工厂
def h100_base():
    """H100 基础观测（合成样本，NVLink P2P 单向）。"""
    return {
        'model': 'H100-SXM',
        'compute': [{'gpu': i, 'bf16_tflops': 780.0, 'fp32_tflops': 56.8} for i in range(8)],
        'memory': [{'gpu': i, 'hbm_bw_tbs': 2.92, 'hbm_gb': 79.5, 'ecc_dbe': 0, 'max_temp_c': 75.0} for i in range(8)],
        'links': [{'gpu': i, 'min_nvlink_gbs': 385.0, 'busbw_gbs': 811.0} for i in range(8)],  # P2P 单向，>= 360 PASS 线
        'pcie': [{'gpu': i, 'link_gen': 5, 'link_width': 16, 'bw_gbs': 51.2} for i in range(8)],
        'xids': [],
        'burnin': {'hours': 24.5, 'cards_dropped': 0, 'ecc_dbe_total': 0, 'throttle_events': 0, 'nccl_failures': 0}
    }


def run_judge(data, test_name):
    """运行判定引擎，解析输出。"""
    tmp_json = f'/tmp/test_{test_name}.json'
    try:
        with open(tmp_json, 'w') as f:
            json.dump(data, f)

        result = subprocess.run(
            ['python3', 'sim_burnin.py', 'report', tmp_json],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)


def extract_verdict(output):
    """从输出解析 PASS/FAIL/CONDITIONAL。"""
    for line in output.split('\n'):
        if '结论' in line:
            # 从 "结论 XXX" 中提取 XXX
            import re
            match = re.search(r'结论\s+(\w+)', line)
            if match:
                return match.group(1)
    return None


def test_1_compute_bf16_pass():
    """计算线：BF16 780 TF 过 700 TF 合格线。"""
    data = h100_base()
    code, output = run_judge(data, 'compute_bf16_pass')
    verdict = extract_verdict(output)
    assert verdict == 'PASS', f"Expected PASS, got {verdict}\n{output}"
    assert 'BF16 TFLOPS 卡间极差' in output, "Missing spread check"
    print("✓ test_1_compute_bf16_pass")


def test_2_compute_bf16_fail():
    """计算线：BF16 690 TF 低于 700 TF，FAIL。"""
    data = h100_base()
    data['compute'][3]['bf16_tflops'] = 690.0
    code, output = run_judge(data, 'compute_bf16_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert '[FAIL' in output and 'GPU3' in output, "Should show GPU3 failure"
    print("✓ test_2_compute_bf16_fail")


def test_3_compute_spread_fail():
    """计算线：卡间极差 7% > 5% 限，FAIL。"""
    data = h100_base()
    data['compute'][0]['bf16_tflops'] = 700.0
    data['compute'][1]['bf16_tflops'] = 750.0  # 极差 7.1%
    code, output = run_judge(data, 'compute_spread_fail')
    verdict = extract_verdict(output)
    # 极差 FAIL 会导致整体 FAIL
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'BF16 TFLOPS 卡间极差' in output, "Missing spread failure message"
    print("✓ test_3_compute_spread_fail")


def test_4_hbm_bw_fail():
    """显存线：HBM 带宽 2.40 TB/s < 2.85 TB/s 合格线，FAIL。"""
    data = h100_base()
    data['memory'][2]['hbm_bw_tbs'] = 2.40
    code, output = run_judge(data, 'hbm_bw_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'GPU2' in output and 'HBM 带宽' in output, "Should show GPU2 bandwidth failure"
    print("✓ test_4_hbm_bw_fail")


def test_5_hbm_cap_fail():
    """显存线：容量 76.8 GB < 78.04 GB (98%)，FAIL。"""
    data = h100_base()
    data['memory'][4]['hbm_gb'] = 76.8
    code, output = run_judge(data, 'hbm_cap_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'GPU4' in output and '显存容量' in output, "Should show GPU4 capacity failure"
    print("✓ test_5_hbm_cap_fail")


def test_6_ecc_dbe_fail():
    """显存线：ECC 双位错误零容忍，一条记录即 FAIL。"""
    data = h100_base()
    data['memory'][0]['ecc_dbe'] = 1
    code, output = run_judge(data, 'ecc_dbe_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'ECC 双位错误' in output and '[FAIL' in output, "Should show ECC failure"
    print("✓ test_6_ecc_dbe_fail")


def test_7_temp_fail():
    """显存线：温度 84°C >= 83°C，FAIL。"""
    data = h100_base()
    data['memory'][5]['max_temp_c'] = 84.0
    code, output = run_judge(data, 'temp_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'GPU5' in output and '结温' in output, "Should show GPU5 temperature failure"
    print("✓ test_7_temp_fail")


def test_8_temp_warn():
    """显存线：温度 81°C 在 78-83 WARN 区，CONDITIONAL。"""
    data = h100_base()
    data['memory'][1]['max_temp_c'] = 81.0
    code, output = run_judge(data, 'temp_warn')
    verdict = extract_verdict(output)
    assert verdict == 'CONDITIONAL', f"Expected CONDITIONAL, got {verdict}\n{output}"
    assert '[WARN' in output, "Should show WARN level"
    assert '结温' in output, "Should mention temperature"
    print("✓ test_8_temp_warn")


def test_9_nvlink_chain_fail():
    """互联线：NVLink P2P 单向 320 GB/s < 360 GB/s (0.80×450) PASS 线，FAIL。"""
    data = h100_base()
    data['links'][2]['min_nvlink_gbs'] = 320.0  # < 337.5 (0.75×450)，属 FAIL 区
    code, output = run_judge(data, 'nvlink_chain_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'GPU2' in output and 'NVLink' in output, "Should show GPU2 NVLink failure"
    print("✓ test_9_nvlink_chain_fail")


def test_10_busbw_fail():
    """互联线：all-reduce busbw 300 GB/s < 360 GB/s (0.80×450) PASS 线，FAIL。"""
    data = h100_base()
    data['links'][0]['busbw_gbs'] = 300.0  # < 252 (0.70×450)，属 FAIL 区
    code, output = run_judge(data, 'busbw_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'all-reduce busbw' in output and '[FAIL' in output, "Should show busbw failure"
    print("✓ test_10_busbw_fail")


def test_11_pcie_downgrade_fail():
    """PCIe 线：代际降级到 Gen4，FAIL。"""
    data = h100_base()
    data['pcie'][6]['link_gen'] = 4
    code, output = run_judge(data, 'pcie_downgrade_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'PCIe' in output and 'GPU6' in output, "Should show GPU6 PCIe failure"
    print("✓ test_11_pcie_downgrade_fail")


def test_12_xid_fatal_fail():
    """错误线：Xid 79（致命，需 RMA）零容忍，FAIL。"""
    data = h100_base()
    data['xids'] = [{'timestamp': '...', 'code': 79, 'gpu': 3}]
    code, output = run_judge(data, 'xid_fatal_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert '致命 Xid' in output and '79' in output, "Should show Xid 79 failure"
    print("✓ test_12_xid_fatal_fail")


def test_13_xid_warn_conditional():
    """错误线：Xid 31（app-level 可重试）给 WARN，CONDITIONAL。"""
    data = h100_base()
    data['xids'] = [{'timestamp': '...', 'code': 31, 'gpu': 0}]
    code, output = run_judge(data, 'xid_warn_conditional')
    verdict = extract_verdict(output)
    assert verdict == 'CONDITIONAL', f"Expected CONDITIONAL, got {verdict}\n{output}"
    assert '告警 Xid' in output, "Should show warning Xid"
    assert '[WARN' in output, "Should show WARN level"
    print("✓ test_13_xid_warn_conditional")


def test_14_burnin_short_warn():
    """burn-in 线：12 小时 < 24 小时，WARN → CONDITIONAL。"""
    data = h100_base()
    data['burnin']['hours'] = 12.0
    code, output = run_judge(data, 'burnin_short_warn')
    verdict = extract_verdict(output)
    assert verdict == 'CONDITIONAL', f"Expected CONDITIONAL, got {verdict}\n{output}"
    assert '时长' in output, "Should mention burn-in time"
    assert '[WARN' in output, "Should show WARN level"
    print("✓ test_14_burnin_short_warn")


def test_15_burnin_drop_fail():
    """burn-in 线：掉卡计数 1 > 0，FAIL。"""
    data = h100_base()
    data['burnin']['cards_dropped'] = 1
    code, output = run_judge(data, 'burnin_drop_fail')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert '掉卡' in output and '[FAIL' in output, "Should show card drop failure"
    print("✓ test_15_burnin_drop_fail")


def test_16_combined_fail_warn():
    """聚合逻辑：多个 WARN + 一个 FAIL → FAIL。"""
    data = h100_base()
    data['memory'][0]['max_temp_c'] = 80.0  # WARN
    data['burnin']['hours'] = 12.0           # WARN
    data['compute'][1]['bf16_tflops'] = 690.0  # FAIL（低于 700 线）
    code, output = run_judge(data, 'combined_fail_warn')
    verdict = extract_verdict(output)
    assert verdict == 'FAIL', f"Expected FAIL, got {verdict}\n{output}"
    assert 'FAIL 1' in output or 'FAIL 2' in output, "Should show FAIL count"
    print("✓ test_16_combined_fail_warn")


def test_17_exit_code():
    """exit 码：PASS → 0，FAIL → 1。"""
    data = h100_base()
    code, output = run_judge(data, 'exit_code_pass')
    assert code == 0, f"Expected exit code 0 for PASS, got {code}"

    data_fail = h100_base()
    data_fail['compute'][0]['bf16_tflops'] = 690.0  # 低于 700 线
    code, output = run_judge(data_fail, 'exit_code_fail')
    assert code == 1, f"Expected exit code 1 for FAIL, got {code}"
    print("✓ test_17_exit_code")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tests = [
        test_1_compute_bf16_pass,
        test_2_compute_bf16_fail,
        test_3_compute_spread_fail,
        test_4_hbm_bw_fail,
        test_5_hbm_cap_fail,
        test_6_ecc_dbe_fail,
        test_7_temp_fail,
        test_8_temp_warn,
        test_9_nvlink_chain_fail,
        test_10_busbw_fail,
        test_11_pcie_downgrade_fail,
        test_12_xid_fatal_fail,
        test_13_xid_warn_conditional,
        test_14_burnin_short_warn,
        test_15_burnin_drop_fail,
        test_16_combined_fail_warn,
        test_17_exit_code,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
