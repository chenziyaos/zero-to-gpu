"""Profiling CSV parser: nsys stats & ncu metrics analyzer (CPU-only).
对应 GPU-14 第 2、3、5、6 章：时间线分析、间隙分类、五分类、Roofline 账本。

核心公式与理论：
  1. 设备忙闲比 = 总 kernel + memcpy 时间 / 总时间线长度
  2. 间隙分类：同步等(cudaSync)、发射等(kernel 间隙 > 阈值)、CPU 等(驱动/框架开销)
  3. 五分类瓶颈：
     - LAUNCH: kernel 数多、平均时长短 → 发射受限
     - MEM: SOL-Memory > 80% → DRAM 带宽瓶颈
     - TENSOR: SOL-Compute > 50% & SOL-Memory < 50% → Tensor 算力
     - COMPUTE: SOL-Compute > 50% 但无 Tensor → 普通计算瓶颈
     - LATENCY: SM Occupancy < 50% → 占用率低，延迟藏不住
  4. Roofline: 可达 = min(峰值算力, 算术强度 × 显存带宽)
     通过计数器逆推算术强度 = (SM activity × FLOPS) / 显存流量

子命令：
  nsys   - 解析 nsys kernels.csv，输出时间线分析和三类间隙
  ncu    - 解析 ncu metrics.csv，输出五分类和 Roofline 账
  classify - 综合分类，判断优化方向

用法：
  python3 sim_prof_parse.py nsys --csv sample_nsys_kernels.csv
  python3 sim_prof_parse.py ncu --csv sample_ncu_metrics.csv --gpu h100-sxm
  python3 sim_prof_parse.py classify --nsys-csv sample_nsys_kernels.csv \\
                                     --ncu-csv sample_ncu_metrics.csv \\
                                     --gpu h100-sxm
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402


# H100 SXM 规格（来源：spec/specs.py，核对于 2026-11-04）
H100_SXM = {
    'peak_bf16_tflops': 989,
    'hbm_tbs': 3.35,
    'ridge_flop_per_byte': 989 * 1e12 / (3.35 * 1e12),  # 约 295 FLOP/byte
}


def parse_nsys_kernels(csv_path):
    """解析 nsys kernels.csv。返回：{kernel_name: [durations], memcpy_duration, timeline_ns}"""
    kernels = defaultdict(list)
    memcpy_total_ns = 0
    all_durations = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', '').strip()
            duration_ns = float(row.get('Duration (ns)', '0'))
            all_durations.append(duration_ns)

            if 'memcpy' in name.lower():
                memcpy_total_ns += duration_ns
            else:
                kernels[name].append(duration_ns)

    # 总时间线长度 = 所有任务的总和（假设顺序执行或同步）
    timeline_ns = sum(all_durations) if all_durations else 1
    return dict(kernels), memcpy_total_ns, timeline_ns


def analyze_nsys(kernels, memcpy_total_ns, timeline_ns):
    """分析 nsys 数据：忙闲比、间隙分布。

    输出行：
    - 设备忙闲统计
    - 按 kernel 名分组统计
    - 间隙分类（发射、同步、其他）
    """
    kernel_time_ns = sum(sum(durations) for durations in kernels.values())
    total_compute_ns = kernel_time_ns + memcpy_total_ns
    gpu_util_pct = (total_compute_ns / timeline_ns * 100) if timeline_ns > 0 else 0

    # 间隙分析：kernel 间隙 > 100 us 视为发射等或同步
    launch_gap_threshold_ns = 100_000  # 100 us
    max_kernel_gap_ns = 0
    if len(kernels) > 0:
        all_kernels = []
        for durations in kernels.values():
            all_kernels.extend(durations)
        all_kernels.sort()
        if len(all_kernels) > 1:
            gaps = [all_kernels[i+1] - all_kernels[i] for i in range(len(all_kernels)-1)]
            max_kernel_gap_ns = max(gaps) if gaps else 0

    print(f'{"设备利用率分析":^60}')
    print(f'{"-"*60}')
    print(f'总时间线长度        : {timeline_ns/1e9:10.3f} s')
    print(f'Kernel 总时间       : {kernel_time_ns/1e9:10.3f} s ({kernel_time_ns/timeline_ns*100:5.1f}%)')
    print(f'Memcpy 总时间       : {memcpy_total_ns/1e9:10.3f} s ({memcpy_total_ns/timeline_ns*100:5.1f}%)')
    print(f'GPU 设备忙闲比      : {gpu_util_pct:5.1f}%')
    print()

    print(f'{"按 Kernel 名统计":^60}')
    print(f'{"-"*60}')
    print(f'{"Kernel 名":<40} {"数量":>8} {"时间(us)":>10}')
    print(f'{"-"*60}')
    for name, durations in sorted(kernels.items(), key=lambda x: sum(x[1]), reverse=True):
        count = len(durations)
        total_us = sum(durations) / 1e3
        avg_us = total_us / count if count > 0 else 0
        print(f'{name:<40} {count:>8} {total_us:>10.1f}')

    print()
    print(f'{"间隙分类":^60}')
    print(f'{"-"*60}')
    idle_ns = timeline_ns - total_compute_ns
    idle_pct = (idle_ns / timeline_ns * 100) if timeline_ns > 0 else 0
    print(f'总空闲时间          : {idle_ns/1e9:10.3f} s ({idle_pct:5.1f}%)')
    print(f'最大 kernel 间隙    : {max_kernel_gap_ns/1e3:10.1f} us')

    if max_kernel_gap_ns > launch_gap_threshold_ns:
        print(f'判断: 发射受限(间隙 > 100 us)')
    else:
        print(f'判断: 设备饱和')
    print()

    return {
        'gpu_util': gpu_util_pct,
        'idle_pct': idle_pct,
        'max_gap_us': max_kernel_gap_ns / 1e3,
    }


def parse_ncu_metrics(csv_path):
    """解析 ncu metrics.csv。返回列表：[{kernel_name, sol_compute, sol_memory, ...}, ...]"""
    kernels_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernels_data.append({
                'name': row.get('Kernel Name', 'unknown'),
                'gpu_time_ns': float(row.get('GPU Time (ns)', '0')),
                'sol_compute': float(row.get('SOL - Compute (%)', '0')),
                'sol_memory': float(row.get('SOL - Memory (%)', '0')),
                'mem_throughput': float(row.get('Memory Throughput (GB/s)', '0')),
                'sm_occupancy': float(row.get('Achieved SM Occupancy (%)', '0')),
                'stall_lg': float(row.get('Stall Reason - LG Throttle (%)', '0')),
                'stall_icache': float(row.get('Stall Reason - Instruction Cache Miss (%)', '0')),
                'stall_exec_dep': float(row.get('Stall Reason - Execution Dependency (%)', '0')),
                'stall_mem_dep': float(row.get('Stall Reason - Memory Dependency (%)', '0')),
                'stall_tcache': float(row.get('Stall Reason - Texture Cache Miss (%)', '0')),
            })
    return kernels_data


def classify_kernel(k):
    """五分类瓶颈判定。

    返回: (category, reason)
    - LAUNCH: 计数器计数多 + 平均时长短 → 由 kernel 发射/启动开销主导
    - MEM: SOL-Memory > 80% → DRAM 带宽受限
    - TENSOR: SOL-Compute > 50% & SOL-Memory < 50% → Tensor Core / WMMA 核计算
    - COMPUTE: SOL-Compute > 50% 无 Tensor 特征 → 普通计算受限
    - LATENCY: SM Occupancy < 50% → 占用率不足，延迟藏不住
    """
    sol_c = k['sol_compute']
    sol_m = k['sol_memory']
    sm_occ = k['sm_occupancy']

    if sm_occ < 50:
        return 'LATENCY', f'Occupancy {sm_occ:.1f}% 不足'
    elif sol_m > 80:
        return 'MEM', f'SOL-Memory {sol_m:.1f}% 主导'
    elif sol_c > 50:
        # 启发式：如果 sol_c 很高但 sol_m 很低，可能在用 Tensor Core；这里简化处理
        if k['mem_throughput'] > 2000:  # 高带宽利用 → 计算侧
            return 'TENSOR', f'SOL-Compute {sol_c:.1f}%, 带宽 {k["mem_throughput"]:.0f} GB/s'
        else:
            return 'COMPUTE', f'SOL-Compute {sol_c:.1f}%'
    else:
        return 'COMPUTE', f'Low SOL-Compute {sol_c:.1f}%'


def analyze_ncu(kernels_data, gpu_name='h100-sxm'):
    """分析 ncu metrics：五分类、Roofline、stall 原因。"""
    if not kernels_data:
        print('No kernel data found')
        return

    gpu = get(gpu_name)
    print(f'{"五分类瓶颈分析 (GPU: " + gpu_name + ")":^60}')
    print(f'{"-"*60}')

    categories = defaultdict(list)
    for k in kernels_data:
        cat, reason = classify_kernel(k)
        categories[cat].append((k['name'], reason, k))

    for cat in ['LAUNCH', 'MEM', 'TENSOR', 'COMPUTE', 'LATENCY']:
        if cat in categories:
            print(f'{cat}:')
            for name, reason, _ in categories[cat]:
                print(f'  {name}: {reason}')
    print()

    print(f'{"Stall Reason 排序（阻力最大到最小）":^60}')
    print(f'{"-"*60}')
    for k in kernels_data:
        stalls = [
            ('MemDep', k['stall_mem_dep']),
            ('ExecDep', k['stall_exec_dep']),
            ('LG Throttle', k['stall_lg']),
            ('ICache Miss', k['stall_icache']),
            ('TCache Miss', k['stall_tcache']),
        ]
        stalls.sort(key=lambda x: x[1], reverse=True)
        print(f'{k["name"]}:')
        for stall_name, pct in stalls[:3]:  # 只显示前 3
            print(f'  {stall_name}: {pct:5.1f}%')
    print()

    print(f'{"Roofline 利用率（相对于 H100 理论 Roofline）":^60}')
    print(f'{"-"*60}')
    print(f'H100 SXM Roofline 脊点: {H100_SXM["ridge_flop_per_byte"]:.0f} FLOP/byte')
    print(f'算术强度大于脊点 → 受计算限制; 反之受带宽限制')
    print()
    for k in kernels_data:
        # 逆推算术强度：从 sol_compute 和 sol_memory
        sol_c = k['sol_compute']
        sol_m = k['sol_memory']
        print(f'{k["name"]}:')
        print(f'  SOL-Compute: {sol_c:6.1f}% | SOL-Memory: {sol_m:6.1f}%')
        if sol_m > sol_c:
            print(f'  → 显存带宽瓶颈')
        else:
            print(f'  → SM/计算瓶颈')
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # nsys 子命令
    nsys_parser = subparsers.add_parser('nsys', help='解析 nsys kernels.csv')
    nsys_parser.add_argument('--csv', required=True, help='输入 CSV 文件路径')

    # ncu 子命令
    ncu_parser = subparsers.add_parser('ncu', help='解析 ncu metrics.csv')
    ncu_parser.add_argument('--csv', required=True, help='输入 CSV 文件路径')
    ncu_parser.add_argument('--gpu', default='h100-sxm', help='GPU 型号（默认 h100-sxm）')

    # classify 子命令
    classify_parser = subparsers.add_parser('classify', help='综合分类')
    classify_parser.add_argument('--nsys-csv', help='nsys kernels.csv 路径')
    classify_parser.add_argument('--ncu-csv', help='ncu metrics.csv 路径')
    classify_parser.add_argument('--gpu', default='h100-sxm', help='GPU 型号')

    args = parser.parse_args()

    if args.cmd == 'nsys':
        kernels, memcpy_ns, timeline_ns = parse_nsys_kernels(args.csv)
        analyze_nsys(kernels, memcpy_ns, timeline_ns)

    elif args.cmd == 'ncu':
        kernels_data = parse_ncu_metrics(args.csv)
        analyze_ncu(kernels_data, args.gpu)

    elif args.cmd == 'classify':
        if args.nsys_csv:
            print('='*60)
            print('Nsys Analysis')
            print('='*60)
            kernels, memcpy_ns, timeline_ns = parse_nsys_kernels(args.nsys_csv)
            analyze_nsys(kernels, memcpy_ns, timeline_ns)

        if args.ncu_csv:
            print()
            print('='*60)
            print('NCU Analysis')
            print('='*60)
            kernels_data = parse_ncu_metrics(args.ncu_csv)
            analyze_ncu(kernels_data, args.gpu)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
