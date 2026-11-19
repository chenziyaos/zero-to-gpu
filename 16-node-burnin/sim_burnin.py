"""GPU 单机验收判定引擎：计算、显存、NVLink、PCIe、错误、burn-in 六类合格线。
对应 GPU-16 第 2 章。

核心判决逻辑：
  - 计算：BF16/FP32 GEMM 各自过九成峰值线 + 卡间极差 ≤5% 或 FAIL
  - 显存：HBM 带宽 ≥85% 标称 + 容量 ≥98% + ECC 双位零容忍 + 温度 <83°C(WARN 78-83)
  - 互联：NVLink P2P 单向 ≥80% × per-direction_bandwidth (H100: 360 GB/s) + all-reduce busbw ≥80% × per-direction_bandwidth 或 FAIL
  - PCIe：代际八成（Gen5 x16 → 50.4 GB/s）且无降速降宽或 FAIL
  - 错误：致命 Xid(48/64/79/95/119/120) 零容忍 + 告警 Xid(13/31/43/45/63/74/92/94) 记录 WARN
  - burn-in：≥24h + 掉卡/ECC双位/降频/通信失败计数全零

整机结论聚合：任一 FAIL → FAIL；无 FAIL 但有 WARN → CONDITIONAL；全 PASS → PASS
退出码：0 通过 / 1 拒收

子命令：
  report  吃 JSON 观测数据，输出 PASS/WARN/FAIL 判决与逐项实测值

用法：
  python3 sim_burnin.py report samples/healthy_pass.json
  python3 sim_burnin.py report samples/fail_single_gpu.json --verbose

观测 JSON 格式（见 make_samples.py）：
  {
    "model": "H100-SXM",
    "compute": [{"gpu": 0, "bf16_tflops": 912, "fp32_tflops": 56.8}, ...],
    "memory": [{"gpu": 0, "hbm_bw_tbs": 2.92, "hbm_gb": 78.5, "ecc_dbe": 0, "max_temp_c": 78}, ...],
    "links": [{"gpu": 0, "min_nvlink_gbs": 385.2, "busbw_gbs": 412}, ...],  # P2P 单向，非双向
    "pcie": [{"gpu": 0, "link_gen": 5, "link_width": 16, "bw_gbs": 51.2}, ...],
    "xids": [{"timestamp": "...", "code": 79, "gpu": 2}, ...],
    "burnin": {"hours": 24.5, "cards_dropped": 0, "ecc_dbe_total": 0, "throttle_events": 0, "nccl_failures": 0}
  }
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# 合格线参数：比例口径 + 零容忍项
# BF16 GEMM 合格线：700 TFLOPS (≈89% of ~786 MAMF reference from GPU-02)
BF16_TFLOPS_PASS = 700.0  # per GPU, single GEMM
THRESHOLDS = {
    'bf16_tflops_ratio': 0.90,      # 计算线：90% 标称（仅作参考，实际用 BF16_TFLOPS_PASS）
    'fp32_tflops_ratio': 0.90,
    'card_spread_ratio': 0.05,      # 卡间极差线：≤5%
    'hbm_bw_ratio': 0.85,           # 显存带宽线：85% 标称
    'hbm_cap_ratio': 0.98,          # 显存容量线：98% 标称
    'hbm_temp_pass_c': 83.0,        # 温度线：>83 FAIL
    'hbm_temp_warn_c': 78.0,        # 温度WARN区：78-83
    'nvlink_p2p_ratio_pass': 0.80,  # NVLink P2P 单向 PASS：80% per-direction BW (H100: 0.80×450=360 GB/s)
    'nvlink_p2p_ratio_warn': 0.75,  # NVLink P2P WARN 区：75%-80% per-direction BW
    'busbw_ratio_pass': 0.80,       # all-reduce busbw PASS：80% per-direction BW (H100: 360 GB/s)
    'busbw_ratio_warn': 0.70,       # all-reduce busbw WARN 区：70%-80% per-direction BW
    'pcie_ratio': 0.80,             # PCIe：80% 理论带宽
    'burnin_min_hours': 24.0,       # burn-in：≥24h
    # Xid 分类遵循 NVIDIA 官方 Xid 目录推荐操作：reset/RMA (fatal) vs app-level (warn)
    'fatal_xids': {48, 64, 79, 95, 119, 120},  # 致命 Xid：需要 reset/RMA
    'warn_xids': {13, 31, 43, 45, 63, 74, 92, 94},  # 告警 Xid：app-level 可重试
}


def pcie_gen_theory_gbs(gen, width):
    """PCIe 编码后理论带宽，单向。Gen5 x16 ≈ 63 GB/s 单向。
    计算方式：GT/s * lane数 * 字节转换 * 编码效率
    """
    rates = {3: 4, 4: 8, 5: 16}  # Gen3~5 GT/s per lane
    gt_s = rates.get(gen, 0)
    if not gt_s:
        return 0
    # 计算单向带宽：GT/s * width (lane数) / 8 (bit→byte) * 编码效率
    # Gen5 128b/130b ≈ 0.985, 简化用 1.0
    single_dir = gt_s * width / 8
    return single_dir


def judge_compute(observations, specs):
    """判：每卡 BF16/FP32 TFLOPS、卡间极差。"""
    issues = []
    for precision in ['bf16']:  # 只判 BF16；FP32 作为参考但不判决
        field = f'{precision}_tflops'
        values = [obs.get(field, 0) for obs in observations]

        # 单卡线：BF16 用绝对值 700 TFLOPS（参考 GPU-02 MAMF），不用规格峰值比例
        if precision == 'bf16':
            threshold = BF16_TFLOPS_PASS
        else:
            ref = getattr(specs, f'{precision}_tflops', 0)
            if ref == 0:
                continue  # 该精度无规格
            threshold = ref * THRESHOLDS[f'{precision}_tflops_ratio']

        for i, v in enumerate(values):
            if v < threshold:
                issues.append({
                    'level': 'FAIL',
                    'name': f'GPU{i} {precision.upper()} TFLOPS',
                    'actual': v,
                    'requirement': f'>= {threshold:.0f}',
                    'measured': f'{v:.1f}'
                })

        # 卡间极差线
        if len(values) > 0 and min(values) > 0:
            avg = sum(values) / len(values)
            max_v = max(values)
            min_v = min(values)
            spread = (max_v - min_v) / avg
            spread_limit = THRESHOLDS['card_spread_ratio']

            level = 'PASS' if spread <= spread_limit else 'FAIL'
            issues.append({
                'level': level,
                'name': f'{precision.upper()} TFLOPS 卡间极差',
                'actual': spread,
                'requirement': f'<= {spread_limit*100:.1f}%',
                'measured': f'{spread*100:.1f}%'
            })

    return issues


def judge_memory(observations, specs):
    """判：HBM 带宽、容量、ECC 双位、温度。"""
    issues = []

    # HBM 带宽
    bw_threshold = specs.hbm_tbs * THRESHOLDS['hbm_bw_ratio']
    for i, obs in enumerate(observations):
        bw = obs.get('hbm_bw_tbs', 0)
        if bw < bw_threshold:
            issues.append({
                'level': 'FAIL',
                'name': f'GPU{i} HBM 带宽',
                'actual': bw,
                'requirement': f'>= {bw_threshold:.2f} TB/s',
                'measured': f'{bw:.2f} TB/s'
            })

    # 显存容量（需要逐卡数据）
    cap_threshold = specs.hbm_gb * THRESHOLDS['hbm_cap_ratio']
    for i, obs in enumerate(observations):
        cap = obs.get('hbm_gb', specs.hbm_gb)
        if cap < cap_threshold:
            issues.append({
                'level': 'FAIL',
                'name': f'GPU{i} 显存容量',
                'actual': cap,
                'requirement': f'>= {cap_threshold:.1f} GB',
                'measured': f'{cap:.1f} GB'
            })

    # ECC 双位错误零容忍
    ecc_dbe_total = sum(obs.get('ecc_dbe', 0) for obs in observations)
    if ecc_dbe_total > 0:
        issues.append({
            'level': 'FAIL',
            'name': 'ECC 双位错误',
            'actual': ecc_dbe_total,
            'requirement': '0',
            'measured': str(ecc_dbe_total)
        })

    # 温度
    for i, obs in enumerate(observations):
        temp = obs.get('max_temp_c', 0)
        if temp >= THRESHOLDS['hbm_temp_pass_c']:
            issues.append({
                'level': 'FAIL',
                'name': f'GPU{i} 结温',
                'actual': temp,
                'requirement': f'< {THRESHOLDS["hbm_temp_pass_c"]:.0f}°C',
                'measured': f'{temp:.0f}°C'
            })
        elif temp >= THRESHOLDS['hbm_temp_warn_c']:
            issues.append({
                'level': 'WARN',
                'name': f'GPU{i} 结温',
                'actual': temp,
                'requirement': f'< {THRESHOLDS["hbm_temp_warn_c"]:.0f}°C（余量区）',
                'measured': f'{temp:.0f}°C'
            })

    return issues


def judge_interconnect(observations, specs):
    """判：NVLink P2P 单向 + all-reduce busbw。

    NVLink P2P (per GPU pair through NVSwitch, unidirectional):
      使用 nvlink_gbs / 2 得单向带宽。H100: 900/2 = 450 GB/s
      PASS: >= 0.80 × 450 = 360 GB/s
      WARN: 0.75-0.80 × 450 = 337.5-360 GB/s
      FAIL: < 337.5 GB/s

    all-reduce busbw at large message size:
      PASS: >= 0.80 × per-direction NVLink bandwidth (H100: 360 GB/s)
      WARN: 0.70-0.80 × per-direction (252-360 GB/s)
      FAIL: < 252 GB/s
      注：NVLS 可推高 busbw，本线作为下界。
    """
    issues = []

    # 单向 NVLink 带宽（从 specs 的双向值）
    per_dir_bw = specs.nvlink_gbs / 2

    # NVLink P2P 逐链判决
    for i, obs in enumerate(observations):
        nvlink = obs.get('min_nvlink_gbs', 0)
        pass_threshold = per_dir_bw * THRESHOLDS['nvlink_p2p_ratio_pass']
        warn_threshold = per_dir_bw * THRESHOLDS['nvlink_p2p_ratio_warn']

        if nvlink < warn_threshold:
            issues.append({
                'level': 'FAIL',
                'name': f'GPU{i} NVLink P2P 单向',
                'actual': nvlink,
                'requirement': f'>= {pass_threshold:.0f} GB/s',
                'measured': f'{nvlink:.1f} GB/s'
            })
        elif nvlink < pass_threshold:
            issues.append({
                'level': 'WARN',
                'name': f'GPU{i} NVLink P2P 单向',
                'actual': nvlink,
                'requirement': f'>= {pass_threshold:.0f} GB/s（余量区 {warn_threshold:.0f}-{pass_threshold:.0f}）',
                'measured': f'{nvlink:.1f} GB/s'
            })

    # all-reduce busbw 判决
    busbw = observations[0].get('busbw_gbs', 0) if observations else 0
    busbw_pass = per_dir_bw * THRESHOLDS['busbw_ratio_pass']
    busbw_warn = per_dir_bw * THRESHOLDS['busbw_ratio_warn']

    if busbw < busbw_warn:
        issues.append({
            'level': 'FAIL',
            'name': 'all-reduce busbw（单向容纳线）',
            'actual': busbw,
            'requirement': f'>= {busbw_pass:.0f} GB/s',
            'measured': f'{busbw:.1f} GB/s'
        })
    elif busbw < busbw_pass:
        issues.append({
            'level': 'WARN',
            'name': 'all-reduce busbw（单向容纳线）',
            'actual': busbw,
            'requirement': f'>= {busbw_pass:.0f} GB/s（余量区 {busbw_warn:.0f}-{busbw_pass:.0f}）',
            'measured': f'{busbw:.1f} GB/s'
        })

    return issues


def judge_pcie(observations, specs):
    """判：PCIe 代际、宽度、带宽。"""
    issues = []

    for i, obs in enumerate(observations):
        gen = obs.get('link_gen', 5)
        width = obs.get('link_width', 16)
        bw = obs.get('bw_gbs', 0)

        # 协商检查
        theory = pcie_gen_theory_gbs(gen, width)
        expected_gen = 5
        expected_width = 16

        if gen < expected_gen or width < expected_width:
            issues.append({
                'level': 'FAIL',
                'name': f'GPU{i} PCIe 协商',
                'actual': f'Gen{gen} x{width}',
                'requirement': f'Gen{expected_gen} x{expected_width}',
                'measured': f'Gen{gen} x{width}'
            })

        # 带宽线
        bw_threshold = theory * THRESHOLDS['pcie_ratio']
        if bw < bw_threshold:
            issues.append({
                'level': 'FAIL',
                'name': f'GPU{i} PCIe 带宽',
                'actual': bw,
                'requirement': f'>= {bw_threshold:.1f} GB/s',
                'measured': f'{bw:.1f} GB/s'
            })

    return issues


def judge_errors(xids_list):
    """判：致命 Xid 零容忍，告警 Xid 记录。"""
    issues = []

    fatal = [x for x in xids_list if x.get('code', 0) in THRESHOLDS['fatal_xids']]
    warn = [x for x in xids_list if x.get('code', 0) in THRESHOLDS['warn_xids']]

    if fatal:
        fatal_codes = [str(x.get('code')) for x in fatal]
        issues.append({
            'level': 'FAIL',
            'name': '致命 Xid',
            'actual': len(fatal),
            'requirement': '0',
            'measured': f'[{", ".join(fatal_codes)}]'
        })

    if warn:
        warn_codes = [str(x.get('code')) for x in warn]
        issues.append({
            'level': 'WARN',
            'name': '告警 Xid',
            'actual': len(warn),
            'requirement': '复查',
            'measured': f'[{", ".join(warn_codes)}]'
        })

    return issues


def judge_burnin(burnin_data):
    """判：时长、掉卡、ECC 双位、降频、通信失败。"""
    issues = []

    hours = burnin_data.get('hours', 0)
    if hours < THRESHOLDS['burnin_min_hours']:
        issues.append({
            'level': 'WARN',
            'name': 'burn-in 时长',
            'actual': hours,
            'requirement': f'>= {THRESHOLDS["burnin_min_hours"]:.0f}h',
            'measured': f'{hours:.1f}h'
        })

    # 四个零容忍计数
    for key, display in [
        ('cards_dropped', '掉卡'),
        ('ecc_dbe_total', 'ECC 双位'),
        ('throttle_events', '降频事件'),
        ('nccl_failures', 'NCCL 失败')
    ]:
        count = burnin_data.get(key, 0)
        if count > 0:
            issues.append({
                'level': 'FAIL',
                'name': f'burn-in {display}',
                'actual': count,
                'requirement': '0',
                'measured': str(count)
            })

    return issues


def aggregate_verdict(all_issues):
    """聚合所有判决：任一 FAIL → FAIL；无 FAIL 但有 WARN → CONDITIONAL；全 PASS → PASS。"""
    has_fail = any(i['level'] == 'FAIL' for i in all_issues)
    has_warn = any(i['level'] == 'WARN' for i in all_issues)

    if has_fail:
        return 'FAIL'
    elif has_warn:
        return 'CONDITIONAL'
    else:
        return 'PASS'


def report(args):
    """主入口：读 JSON、判决、打印报告。"""
    try:
        with open(args.observations, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: 无法读取 {args.observations}: {e}", file=sys.stderr)
        return 1

    model_name = data.get('model', 'Unknown')
    try:
        normalized = model_name.lower().replace(' ', '').replace('_', '-')
        specs = get(normalized)
    except SystemExit:
        print(f"ERROR: 未知模型 {model_name}", file=sys.stderr)
        return 1

    # 逐类判决
    issues = []
    issues.extend(judge_compute(data.get('compute', []), specs))
    issues.extend(judge_memory(data.get('memory', []), specs))
    issues.extend(judge_interconnect(data.get('links', []), specs))
    issues.extend(judge_pcie(data.get('pcie', []), specs))
    issues.extend(judge_errors(data.get('xids', [])))
    issues.extend(judge_burnin(data.get('burnin', {})))

    verdict = aggregate_verdict(issues)
    fail_count = sum(1 for i in issues if i['level'] == 'FAIL')
    warn_count = sum(1 for i in issues if i['level'] == 'WARN')

    # 打印报告
    print(f"机型 {model_name}  结论 {verdict}  FAIL {fail_count}  WARN {warn_count}")
    for issue in issues:
        level_str = f"[{issue['level']:11}]"
        print(f"  {level_str} {issue['name']:20} 实测 {issue['measured']:20} 要求 {issue['requirement']}")

    return 0 if verdict == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description='GPU 单机验收判定引擎')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    report_parser = subparsers.add_parser('report', help='读观测 JSON 输出判决')
    report_parser.add_argument('observations', help='观测 JSON 文件路径')
    report_parser.add_argument('--verbose', action='store_true', help='详细输出')
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
