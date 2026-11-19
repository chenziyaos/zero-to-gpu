"""NCCL-tests 日志解析器：针对单机 ring all-reduce 的带宽验证（CPU）。对应 GPU-13 第 9 章。

nccl-tests 的 all_reduce 子测试给出两列带宽数据：
  1. algbw（算法带宽）：应用看到的吞吐，等于消息大小 / 耗时，单位 GB/s
  2. busbw: algbw x 2(n-1)/n for all-reduce; compare it with per-direction link bandwidth

判定规则：
  - PASS: busbw >= 80% of the slowest hop's per-direction bandwidth
  - WARN: 50%-80%, often a PCIe hop or contention
  - FAIL: < 50%, check topo -m and Fabric Manager

子命令：
  parse    解析单条 nccl-tests 日志文件
  batch    批量扫描目录下所有 nccl-test 日志
  threshold  测试 busbw 阈值判定

用法：
  python3 sim_nccl_check.py parse nccl_test_h100_8gpu.log --expected-bw 450
  python3 sim_nccl_check.py batch /path/to/logs/
  python3 sim_nccl_check.py threshold --gen nvlink4 --n 8
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# NCCL-tests 输出示例行格式
# [GPU0]  size 1073741824, count 1024, type float, op sum, root 0, in_place 0, algorithm RING, protocol LL
#   rank= 0 nranks= 8 cudaDev= 0  root= 0 avg= 1234567890 algbw=123.45 busbw=234.56

NCCL_LOG_PATTERN = re.compile(
    r'algbw\s*=\s*([\d.]+)\s+busbw\s*=\s*([\d.]+)',
    re.IGNORECASE
)

# Expected busbw ceiling for a single-node ring all-reduce = per-direction bandwidth of the slowest hop
# (nccl-tests busbw is already normalised by 2(n-1)/n, so it is compared with link bandwidth directly)
EXPECTED_BUSBW = {
    ('nvlink3', 8): 300,   # A100: 12 links x 25 GB/s
    ('nvlink4', 8): 450,   # H100: 18 links x 25 GB/s
    ('nvlink5', 8): 900,   # B200: 18 links x 50 GB/s
    ('pcie-gen5-x16', 8): 63,
}
PASS_RATIO, WARN_RATIO = 0.80, 0.50  # project convention, not a vendor number


def parse_nccl_log_file(filename):
    """解析 nccl-tests 日志文件。返回 (config, results)。

    config: {'gpu_count': int, 'message_size': int, 'algorithm': str, ...}
    results: [{'size': int, 'algbw': float, 'busbw': float}, ...]
    """
    config = {}
    results = []

    try:
        with open(filename, 'r') as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f'无法读取文件 {filename}: {e}')

    # 提取配置行
    for line in content.split('\n'):
        # 提取 GPU 数
        if 'nranks=' in line:
            m = re.search(r'nranks\s*=\s*(\d+)', line)
            if m:
                config['gpu_count'] = int(m.group(1))

        # 提取消息大小
        if 'size' in line and line.startswith('[GPU'):
            m = re.search(r'size\s+(\d+)', line)
            if m:
                size = int(m.group(1))
                if 'message_sizes' not in config:
                    config['message_sizes'] = []
                if size not in config['message_sizes']:
                    config['message_sizes'].append(size)

        # 提取算法与协议
        if 'algorithm' in line:
            m = re.search(r'algorithm\s+(\w+)', line)
            if m:
                config['algorithm'] = m.group(1)
            m = re.search(r'protocol\s+(\w+)', line)
            if m:
                config['protocol'] = m.group(1)

        # 提取 algbw 和 busbw
        match = NCCL_LOG_PATTERN.search(line)
        if match:
            algbw = float(match.group(1))
            busbw = float(match.group(2))
            results.append({'algbw': algbw, 'busbw': busbw, 'line': line.strip()})

    return config, results


def judge_result(busbw, expected_bw, algbw=None):
    """判定 NCCL 测试结果：PASS / WARN / FAIL。

    PASS: busbw >= PASS_RATIO x expected
    WARN: WARN_RATIO..PASS_RATIO
    FAIL: below WARN_RATIO
    """
    if expected_bw <= 0:
        return 'UNKNOWN', 0.0

    efficiency = 100.0 * busbw / expected_bw
    if efficiency >= PASS_RATIO * 100:
        return 'PASS', efficiency
    elif efficiency >= WARN_RATIO * 100:
        return 'WARN', efficiency
    else:
        return 'FAIL', efficiency


def cmd_parse(args):
    """解析单个 nccl-tests 日志"""
    try:
        config, results = parse_nccl_log_file(args.file)
    except Exception as e:
        print(f'错误：{e}', file=sys.stderr)
        sys.exit(1)

    print(f'NCCL-tests 日志：{os.path.basename(args.file)}')
    print()
    print('配置：')
    for k, v in sorted(config.items()):
        if k == 'message_sizes':
            print(f'  {k}: {[f"{s//1e6:.0f}MB" if s >= 1e6 else f"{s//1e3}KB" for s in sorted(v)[:5]]} ...')
        else:
            print(f'  {k}: {v}')
    print()

    if not results:
        print('未找到带宽数据')
        return

    gpu_count = config.get('gpu_count', 0)
    expected_bw = args.expected_bw

    # 如果未指定预期带宽，尝试从配置推断
    if expected_bw is None:
        nvlink_gen = args.nvlink_gen
        key = (nvlink_gen, gpu_count)
        if key in EXPECTED_BUSBW:
            expected_bw = EXPECTED_BUSBW[key]
            print(f'推断预期 busbw（{nvlink_gen}，{gpu_count} 卡）：{expected_bw} GB/s')
        else:
            expected_bw = 400  # 缺省保守估计
            print(f'⚠ 无法推断预期值，使用缺省 {expected_bw} GB/s')
    else:
        print(f'指定预期 busbw：{expected_bw} GB/s')

    print()
    print(f'{"size":>15} {"algbw":>10} {"busbw":>10} {"效率":>8} {"判定":>8}')
    print('-' * 65)

    statuses = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
    busbw_min, busbw_max = float('inf'), 0

    for r in results:
        algbw = r['algbw']
        busbw = r['busbw']

        busbw_min = min(busbw_min, busbw)
        busbw_max = max(busbw_max, busbw)

        status, eff = judge_result(busbw, expected_bw, algbw)
        statuses[status] += 1

        # 尝试从日志行提取消息大小
        size_str = '?'
        m = re.search(r'size\s+(\d+)', r['line'])
        if m:
            size = int(m.group(1))
            if size >= 1e9:
                size_str = f'{size/1e9:.1f}GB'
            elif size >= 1e6:
                size_str = f'{size/1e6:.0f}MB'
            else:
                size_str = f'{size//1e3}KB'

        print(f'{size_str:>15}  {algbw:>10.1f}  {busbw:>10.1f}  {eff:>7.1f}%  {status:>8}')

    print()
    print('总结：')
    print(f'  结果：{statuses["PASS"]} PASS，{statuses["WARN"]} WARN，{statuses["FAIL"]} FAIL（共 {len(results)} 次测试）')
    print(f'  busbw 范围：{busbw_min:.1f} - {busbw_max:.1f} GB/s')

    if statuses['FAIL'] > 0:
        print('  ✗ 有 FAIL：建议检查拓扑（topo -m）和 Fabric Manager 配置')
    elif statuses['WARN'] > 0:
        print('  ⚠ 有 WARN：busbw 低于预期的 80%，可能有小范围 PCIe 或调度问题')
    elif statuses['PASS'] == len(results):
        print('  ✓ 所有 PASS：NVLink 配置正常')


def cmd_batch(args):
    """批量扫描日志目录"""
    log_dir = Path(args.directory)
    if not log_dir.is_dir():
        print(f'目录不存在：{args.directory}', file=sys.stderr)
        sys.exit(1)

    log_files = list(log_dir.glob('*.log')) + list(log_dir.glob('*nccl*'))
    if not log_files:
        print(f'未找到日志文件：{log_dir}/*.log', file=sys.stderr)
        return

    print(f'扫描 {len(log_files)} 个日志文件...')
    print()
    print(f'{"文件":30} {"GPU数":>6} {"状态":>8} {"MIN":>10} {"AVG":>10} {"MAX":>10}')
    print('-' * 75)

    for log_file in sorted(log_files)[:10]:  # 最多显示 10 个
        try:
            config, results = parse_nccl_log_file(str(log_file))
            if not results:
                continue

            gpu_count = config.get('gpu_count', 0)
            busbw_vals = [r['busbw'] for r in results]
            busbw_min, busbw_max = min(busbw_vals), max(busbw_vals)
            busbw_avg = sum(busbw_vals) / len(busbw_vals)

            # 判定
            key = ('nvlink4', gpu_count)
            expected_bw = EXPECTED_BUSBW.get(key, 400)
            status = judge_result(busbw_avg, expected_bw)[0]

            fname = log_file.name[:30]
            print(f'{fname:30}  {gpu_count:>6}  {status:>8}  {busbw_min:>10.1f}  '
                  f'{busbw_avg:>10.1f}  {busbw_max:>10.1f}')
        except Exception as e:
            pass


def cmd_threshold(args):
    """Expected busbw ceiling and PASS/WARN lines per generation (8 GPUs)."""
    print(f'{"hop":15} {"ceiling":>10} {"PASS >=":>10} {"WARN >=":>10}')
    print('-' * 50)
    for (gen, n), bw in EXPECTED_BUSBW.items():
        print(f'{gen:15} {bw:>5} GB/s {bw * PASS_RATIO:>10.1f} {bw * WARN_RATIO:>10.1f}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    parse_parser = subparsers.add_parser('parse', help='解析单个 nccl-tests 日志')
    parse_parser.add_argument('file', help='日志文件路径')
    parse_parser.add_argument('--expected-bw', type=float, default=None, help='期望 busbw（GB/s）')
    parse_parser.add_argument('--nvlink-gen', default='nvlink4', help='NVLink 代数（用于推断期望值）')

    batch_parser = subparsers.add_parser('batch', help='批量扫描日志目录')
    batch_parser.add_argument('directory', help='日志目录')

    subparsers.add_parser('threshold', help='显示预期 busbw 阈值表')

    args = parser.parse_args()

    if args.cmd == 'parse':
        cmd_parse(args)
    elif args.cmd == 'batch':
        cmd_batch(args)
    elif args.cmd == 'threshold':
        cmd_threshold(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
