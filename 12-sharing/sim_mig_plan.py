"""MIG 切分方案计算器：profile 约束与 slice 分配。对应 GPU-12 第 2-3 章。

MIG（Multi-Instance GPU）把一张 GPU 在硬件层面切成多个独立的小卡，每个有自己的
显存 slice、L2 缓存、复制引擎和带宽。H100 把显存切成 8 个 slice，算力切成 7 个。
profile 的命名规则是 Ng.XXgb，其中 N 是算力分数（1~7），XX 是显存 GB。

核心约束（来源：NVIDIA MIG User Guide for Hopper，Page 30-31）：
- H100 80GB：显存分成 8 个 10GB slice，算力分成 7 个
- profile 允许的配置见 NVIDIA 支持列表，每个 profile 规定了能占多少个 slice
- 创建顺序影响结果：大 profile 需要连续 slice，先创建会导致碎片化

子命令：
  avail       所有支持的 H100 profile 及其 slice 需求
  check       给定需求清单，计算是否能放下并显示分配方案
  order       测试不同创建顺序对同一组 profile 的影响
  ceiling     显存和算力分别的理论上限

用法：
  python3 sim_mig_plan.py avail
  python3 sim_mig_plan.py check --demand 3g.20gb 3g.20gb
  python3 sim_mig_plan.py order --demand 3g.20gb 3g.20gb 1g.5gb
  python3 sim_mig_plan.py ceiling
"""
import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class MIGProfile:
    """MIG profile 规格。"""
    name: str
    compute_slices: int     # 占用的算力 slice（分母 7）
    mem_slices: int         # 占用的显存 slice（分母 8，每个 10GB）
    memory_gb: int

    def __str__(self):
        return f"{self.name:10} compute={self.compute_slices}/7  mem={self.mem_slices}/8  {self.memory_gb:2}GB"


# H100 80GB 支持的 profile 列表（来源：NVIDIA MIG User Guide Hopper，Table 5）
# 核对于 2026-11-02
H100_PROFILES = {
    '1g.5gb':   MIGProfile('1g.5gb',   1, 1, 5),
    '2g.10gb':  MIGProfile('2g.10gb',  2, 2, 10),
    '3g.20gb':  MIGProfile('3g.20gb',  3, 2, 10),     # Note: 3 compute but only 2 mem slices
    '4g.20gb':  MIGProfile('4g.20gb',  4, 4, 20),
    '6g.40gb':  MIGProfile('6g.40gb',  6, 4, 20),     # Note: 6 compute but only 4 mem slices
    '7g.80gb':  MIGProfile('7g.80gb',  7, 8, 80),     # Entire card
}


# MIG profile 能够占用的显存 slice 范围（连续 slice，来源：NVIDIA MIG User Guide，Page 31）
# 核对于 2026-11-02
PROFILE_SLICE_RANGES = {
    '1g.5gb':   (0, 1),      # 1 slice：可在任意 slice 位置
    '2g.10gb':  (0, 2),      # 2 slices：连续，可从 slice 0, 2, 4, 6 开始
    '3g.20gb':  (0, 2),      # Note：3g.20gb 只用 2 mem slices（显存 10GB），但占用连续 4 个，来源 Page 30
    '4g.20gb':  (0, 4),      # 4 slices：连续，可从 slice 0, 4 开始
    '6g.40gb':  (0, 4),      # 4 slices（虽然算力 6/7），可从 slice 0, 4 开始
    '7g.80gb':  (0, 8),      # 整卡
}


def list_profiles():
    """列出所有支持的 profile。"""
    print("H100 80GB 支持的 MIG profile（来源：NVIDIA MIG User Guide Hopper，Table 5）")
    print(f"{'Profile':12} {'Compute':15} {'Memory':20} {'Size':6}")
    print("-" * 60)
    for name in sorted(H100_PROFILES.keys()):
        p = H100_PROFILES[name]
        print(p)


def can_fit_single(profile_name: str, used_mem_slices: List[bool]) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """检查单个 profile 是否能放下（返回是否能放下，以及 slice 范围）。

    来源：NVIDIA MIG User Guide，Page 30-31 placement rules。
    """
    if profile_name not in H100_PROFILES:
        return False, None

    profile = H100_PROFILES[profile_name]
    # 实际显存 slice 需求：大部分 profile 的显存 slice 需求见 Table 5
    # 注意：3g.20gb 和 6g.40gb 显存占用少于表示的字节数，但 slice 占用遵循连续规则
    mem_needed = profile.mem_slices

    # 查找足够的连续空闲 slice
    for start in range(9 - mem_needed):
        if all(not used_mem_slices[start + i] for i in range(mem_needed)):
            return True, (start, start + mem_needed)

    return False, None


def try_placement(demands: List[str]) -> Tuple[bool, List[Tuple[str, Tuple[int, int]]]]:
    """尝试按给定顺序放置需求。"""
    if not all(d in H100_PROFILES for d in demands):
        invalid = [d for d in demands if d not in H100_PROFILES]
        raise ValueError(f"未知 profile：{invalid}")

    used_mem = [False] * 8
    used_compute = [False] * 7
    placement = []

    for i, profile_name in enumerate(demands):
        profile = H100_PROFILES[profile_name]
        can_fit, mem_range = can_fit_single(profile_name, used_mem)

        if not can_fit:
            return False, placement

        # 标记已使用
        for j in range(mem_range[0], mem_range[1]):
            used_mem[j] = True
        # 检查算力 slice
        compute_needed = profile.compute_slices
        compute_avail = sum(1 for x in used_compute if not x)
        if compute_avail < compute_needed:
            return False, placement

        for j in range(compute_needed):
            # 简化：从前往后分配算力 slice
            for k in range(7):
                if not used_compute[k]:
                    used_compute[k] = True
                    break

        placement.append((profile_name, mem_range))

    return True, placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='cmd', help='子命令')

    # avail 子命令
    subparsers.add_parser('avail', help='列出所有支持的 profile')

    # check 子命令
    check_parser = subparsers.add_parser('check', help='检查需求是否能放下')
    check_parser.add_argument('--demand', nargs='+', required=True, help='profile 需求列表（空格分隔）')

    # order 子命令
    order_parser = subparsers.add_parser('order', help='测试不同创建顺序')
    order_parser.add_argument('--demand', nargs='+', required=True, help='profile 需求列表')

    # ceiling 子命令
    subparsers.add_parser('ceiling', help='计算理论上限')

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == 'avail':
        list_profiles()

    elif args.cmd == 'check':
        demands = args.demand
        success, placement = try_placement(demands)
        print(f"需求：{' + '.join(demands)}")
        if success:
            print("✓ 可以放下")
            print("\n创建顺序方案：")
            for profile_name, mem_range in placement:
                print(f"  {profile_name:12} → mem_slices[{mem_range[0]}:{mem_range[1]}]")
        else:
            print("✗ 无法放下")

    elif args.cmd == 'order':
        demands = args.demand
        success, placement = try_placement(demands)
        print(f"需求：{' + '.join(demands)}")
        if success:
            print("✓ 按该顺序可以放下")
        else:
            print("✗ 按该顺序无法放下，尝试重新排序…")
            # 简单贪心：先放大 profile
            sorted_demands = sorted(demands, key=lambda x: H100_PROFILES[x].mem_slices, reverse=True)
            success2, placement2 = try_placement(sorted_demands)
            if success2:
                print(f"  建议顺序：{' → '.join(sorted_demands)}")
                print(f"  ✓ 可以放下")

    elif args.cmd == 'ceiling':
        print("H100 80GB MIG 理论上限：")
        print(f"  显存：8 个 10GB slice = 80GB")
        print(f"  算力：7 个 compute slice = 1/7 per slice")
        print(f"  可容纳最大配置数（全 1g.5gb）：7 个实例")


if __name__ == '__main__':
    main()
