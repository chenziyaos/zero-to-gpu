"""GPU-Util vs SM Activity 分析器（CPU-only）。
对应 GPU-14 第 3 章：GPU-Util 不等于干活。

核心原理：
  GPU-Util = (至少一个 SM 在活动的时间 / 总时间) × 100%

  反直觉现象：一个 kernel 占用 1 个 SM 出现 100% GPU-Util；8 卡中 1 卡一个 kernel 跑在 1 SM 也显示 100%。
  这因为：GPU-Util 只看"有没有在动"，不看"多少卡/SM 在动"。

  真正的饱和度指标：
  1. SM Activity = (平均活跃 SM 数 / 总 SM 数) × 100%  → 数据并行度
  2. Device Throughput = 总 flops / 总时间 vs 峰值算力   → 利用率
  3. Memory Throughput = 显存 GB/s / 最大 GB/s          → 带宽用满度

子命令：
  scenario   - 场景计算：给定 kernel 个数、SM 占用率，算 GPU-Util 错觉
  timeline   - 从时间线数据逆推 SM Activity
  mislead    - 演示 GPU-Util 四种误导现象

用法：
  python3 sim_gpu_util.py scenario --num-kernel 1 --sm-occupancy 0.01
  python3 sim_gpu_util.py timeline --file timeline.txt --total-sm 132
  python3 sim_gpu_util.py mislead
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402


def scenario_gpu_util(num_kernels, sm_occupancy, total_sm=132, gpu='h100-sxm'):
    """给定场景，计算 GPU-Util 和真实 SM Activity。

    参数：
      num_kernels: 并发 kernel 数
      sm_occupancy: 每个 kernel 占用的 SM 比例 (0..1)
      total_sm: 总 SM 数（H100 是 132）
    """
    gpu_obj = get(gpu)
    total_sm = gpu_obj.sm

    active_sms = int(num_kernels * sm_occupancy * total_sm)
    gpu_util_pct = 100.0 if active_sms > 0 else 0.0  # GPU-Util: 只要有 SM 动就 100%
    sm_activity = (active_sms / total_sm * 100) if total_sm > 0 else 0.0

    print(f'{"GPU-Util vs SM Activity 场景分析":^60}')
    print(f'{"-"*60}')
    print(f'GPU 型号              : {gpu} ({total_sm} SMs)')
    print(f'并发 kernel 数        : {num_kernels}')
    print(f'每个 kernel 占用 SM % : {sm_occupancy*100:.1f}%')
    print()
    print(f'总活跃 SM 数          : {active_sms} / {total_sm}')
    print(f'GPU-Util（报告值）   : {gpu_util_pct:5.1f}%')
    print(f'SM Activity（真实值）: {sm_activity:5.1f}%')
    print()

    if sm_activity < 50:
        print(f'警告: SM Activity 仅 {sm_activity:.1f}%，数据并行度不足')
        print(f'建议: 增加 batch、并行 kernel、或用 MPS 承载多个作业')
    else:
        print(f'正常: SM Activity {sm_activity:.1f}%，设备利用充分')
    print()


def misleading_cases():
    """演示 GPU-Util 的四种误导现象。"""
    print(f'{"GPU-Util 四种误导现象":^60}')
    print(f'{"-"*60}')
    print()

    # 现象 1: 一个 kernel，一个 SM，100% GPU-Util
    print('现象 1: 1 kernel × 1 SM → 100% GPU-Util')
    print('-' * 60)
    scenario_gpu_util(num_kernels=1, sm_occupancy=1/132)
    print()

    # 现象 2: 两个小 kernel，几乎零占用，仍 100% GPU-Util
    print('现象 2: 2 kernel × 0.1% SM → 100% GPU-Util（看似饱和，实际 0.2%）')
    print('-' * 60)
    scenario_gpu_util(num_kernels=2, sm_occupancy=0.001)
    print()

    # 现象 3: 整卡利用，仍不到 100%（应该满）
    print('现象 3: 系列小任务，SM 总占用 100%，但 CPU 调度开销高 → GPU-Util 低于 100%')
    print('-' * 60)
    scenario_gpu_util(num_kernels=50, sm_occupancy=1.0/50)
    print()

    # 现象 4: DGX 多卡，1 卡用 100%，整机 GPU-Util 显示 12.5%（错觉）
    print('现象 4: DGX H100（8 卡）其中 1 卡 100% 利用 → 整机 GPU-Util 显示 12.5%')
    print('-' * 60)
    num_gpus = 8
    active_gpus = 1
    gpu_util_per_card = 100.0
    multi_gpu_util = (active_gpus / num_gpus * 100)  # 这是 NVIDIA 工具有时的报告方式
    print(f'单卡 GPU-Util        : {gpu_util_per_card:.1f}%')
    print(f'多卡平均 GPU-Util    : {multi_gpu_util:.1f}% ({active_gpus}/{num_gpus} 卡)')
    print(f'实际含义             : 仅 1/{num_gpus} 张卡在工作')
    print()


def timeline_analysis(timeline_file, total_sm=132):
    """从时间线文件逆推 SM Activity。

    文件格式：每行一个事件，格式为 "timestamp_ns kernel_name sm_count"
    """
    import json

    if not os.path.exists(timeline_file):
        print(f'文件不存在: {timeline_file}')
        return

    try:
        with open(timeline_file, 'r') as f:
            events = json.load(f)
    except Exception as e:
        print(f'解析失败: {e}')
        return

    # 简化：计算时间加权平均活跃 SM 数
    total_active_sm_time = 0
    total_time = 0

    for event in events:
        ts = event.get('timestamp_ns', 0)
        duration_ns = event.get('duration_ns', 0)
        sm_count = event.get('sm_count', 0)

        total_active_sm_time += sm_count * duration_ns
        total_time += duration_ns

    avg_active_sms = total_active_sm_time / total_time if total_time > 0 else 0
    sm_activity = (avg_active_sms / total_sm * 100) if total_sm > 0 else 0

    print(f'{"时间线分析":^60}')
    print(f'{"-"*60}')
    print(f'总时间                : {total_time/1e9:.3f} s')
    print(f'平均活跃 SM 数        : {avg_active_sms:.1f} / {total_sm}')
    print(f'SM Activity           : {sm_activity:.1f}%')
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # scenario 子命令
    scenario_parser = subparsers.add_parser('scenario', help='场景计算')
    scenario_parser.add_argument('--num-kernel', type=int, default=1, help='并发 kernel 数')
    scenario_parser.add_argument('--sm-occupancy', type=float, default=0.01,
                                  help='每个 kernel 占用 SM 比例')
    scenario_parser.add_argument('--gpu', default='h100-sxm', help='GPU 型号')

    # timeline 子命令
    timeline_parser = subparsers.add_parser('timeline', help='时间线分析')
    timeline_parser.add_argument('--file', required=True, help='时间线 JSON 文件')
    timeline_parser.add_argument('--total-sm', type=int, default=132, help='总 SM 数')

    # mislead 子命令
    mislead_parser = subparsers.add_parser('mislead', help='演示误导现象')

    args = parser.parse_args()

    if args.cmd == 'scenario':
        scenario_gpu_util(args.num_kernel, args.sm_occupancy, gpu=args.gpu)
    elif args.cmd == 'timeline':
        timeline_analysis(args.file, args.total_sm)
    elif args.cmd == 'mislead':
        misleading_cases()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
