"""CUDA 版本兼容检查器：给定驱动和程序，判断是否能跑、用哪种兼容模式。对应 GPU-09 第 3 至 7 章。

核心思路：libcuda（驱动端）和 libcudart（程序端）由两个独立的版本决定。
- nvidia-smi 显示的 CUDA Version 是该驱动支持的最高运行时版本。
- 镜像里的 libcudart（编译时决定）和 cuda-compat 包版本决定程序能否跑。

三种兼容模式：
  1. 前向兼容：libcudart >= 驱动的最高支持版本，程序会自动降级到驱动支持的版本。
  2. 后向兼容：libcudart < 驱动最高支持版本，使用驱动原生用户态库，无 PTX JIT。
  3. cuda-compat 垫片：跨大版本时，cuda-compat 包（版本 > 驱动原生版本）垫在驱动上。

子命令：
  check     检查驱动版本和 libcudart 版本是否兼容，输出可用的兼容模式和限制。
  jumps     扫描常见驱动版本与程序版本的组合，展示何时需要升级或装 compat 包。
  jetson    特殊规则：Jetson 卡容器里的版本链更复杂，但同样遵循上述原理。

用法：
  python3 sim_compat.py check --driver 615 --cudart 13.4
  python3 sim_compat.py jumps
  python3 sim_compat.py jetson --container-cuda 13.4 --jetpack 6.0
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

# NVIDIA CUDA 驱动发布计划与支持的最高运行时版本。
# 来源：NVIDIA CUDA Toolkit Release Notes，https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
# 表："CUDA Toolkit and Corresponding Driver Versions"
# 核对日期：2026-11-05
# 基线：R615 支持 CUDA 13.4（LTSB）；R580 起支持 CUDA 13.x
DRIVER_CUDA_SUPPORT = {
    # R615（CUDA 13.4 Update 2，数据中心卡主线）
    615: {'cuda_max': 13.4, 'branch': 'LTSB', 'eol': '2027-06'},
    # R600-R612（CUDA 13.x 线）
    610: {'cuda_max': 13.3, 'branch': 'PB', 'eol': '2025-06'},
    600: {'cuda_max': 13.2, 'branch': 'PB', 'eol': '2025-06'},
    # R580-R590（CUDA 13.0 及过渡）
    590: {'cuda_max': 13.1, 'branch': 'PB', 'eol': '2025-06'},
    580: {'cuda_max': 13.0, 'branch': 'LTSB', 'eol': '2027-09'},
    # R550 及以后（CUDA 12 LTS 线）
    550: {'cuda_max': 12.4, 'branch': 'LTSB', 'eol': '2027-11'},
    551: {'cuda_max': 12.4, 'branch': 'PB', 'eol': '2025-11'},
    552: {'cuda_max': 12.4, 'branch': 'PB', 'eol': '2025-11'},
    # R540 到 R548（CUDA 12 过渡）
    548: {'cuda_max': 12.3, 'branch': 'PB', 'eol': '2024-11'},
    545: {'cuda_max': 12.1, 'branch': 'LTSB', 'eol': '2026-11'},
    # R530-R535（CUDA 11 到 12 过渡）
    535: {'cuda_max': 12.0, 'branch': 'LTSB', 'eol': '2025-11'},
    530: {'cuda_max': 11.8, 'branch': 'PB', 'eol': '2023-11'},
    # R520-R525（CUDA 11 晚期）
    525: {'cuda_max': 11.8, 'branch': 'LTSB', 'eol': '2024-11'},
    520: {'cuda_max': 11.8, 'branch': 'LTSB', 'eol': '2024-11'},
    # R515 及之前（CUDA 11 早期，现已停支）
    515: {'cuda_max': 11.7, 'branch': 'LTSB', 'eol': '2023-11'},
    511: {'cuda_max': 11.6, 'branch': 'NFB', 'eol': '2022-11'},
}

# CUDA 大版本之间的兼容跨度：同大版本间可以向前/向后兼容，跨大版本必须用 cuda-compat。
# CUDA X.y.z：X 是大版本，Y 是 major，Z 是 patch。同大版本（13.0～13.4）相互兼容。
# 来源：NVIDIA CUDA Programming Guide 和发行说明，https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
CUDA_MAJOR_VER = {
    11: {'eol': '2024-06', 'compat_pkg_avail': True},
    12: {'eol': '2025-06', 'compat_pkg_avail': True},
    13: {'eol': '2027-06', 'compat_pkg_avail': True},
}

@dataclass
class CompatResult:
    """兼容性检查结果。"""
    compatible: bool
    mode: str  # 'forward', 'backward', 'compat-pkg', 'incompatible'
    reason: str
    actual_cuda: str  # 程序实际会用的 CUDA 版本
    limitations: list[str]  # 已知限制

def parse_version(v: str) -> Tuple[int, int, int]:
    """解析版本号为 (major, minor, patch)，容错不完整版本号。

    处理格式如 '12.4', '12.4.1', '13.0-1'（compat 包格式）。
    """
    # 首先处理 compat 包的 '-' 分隔符（如 13.0-1），取主版本部分
    v = v.split('-')[0]

    parts = v.split('.')
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)

def version_tuple_from_float(v: float) -> Tuple[int, int]:
    """从浮点版本号生成 (major, minor) 元组。"""
    v_str = str(v)
    parts = v_str.split('.')
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return (major, minor)

def compare_versions(v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
    """比较两个 (major, minor) 版本，返回 -1/0/1。"""
    if v1[0] != v2[0]:
        return -1 if v1[0] < v2[0] else 1
    if v1[1] != v2[1]:
        return -1 if v1[1] < v2[1] else 1
    return 0

def check_compat(driver_ver: int, cudart_ver: str, compat_pkg_ver: Optional[str] = None) -> CompatResult:
    """
    检查驱动与程序 CUDA 运行时版本的兼容性。

    Args:
        driver_ver: 驱动版本号（如 550, 545）
        cudart_ver: libcudart 版本（如 "12.4", "11.8"）
        compat_pkg_ver: 可选，cuda-compat 包版本（如 "13.4-1"）

    Returns:
        CompatResult 包含兼容性判断、实际使用的 CUDA 版本和限制列表。
    """
    if driver_ver not in DRIVER_CUDA_SUPPORT:
        return CompatResult(
            compatible=False,
            mode='incompatible',
            reason=f'驱动版本 {driver_ver} 未知，仓库支持 {list(DRIVER_CUDA_SUPPORT.keys())}',
            actual_cuda='',
            limitations=['需要确认驱动版本号']
        )

    driver_info = DRIVER_CUDA_SUPPORT[driver_ver]
    max_cuda_supported = driver_info['cuda_max']
    cudart_major, cudart_minor, _ = parse_version(cudart_ver)
    cudart_version_str = f'{cudart_major}.{cudart_minor}'

    # 检查 libcudart 大版本
    libcudart_major = cudart_major
    if libcudart_major not in CUDA_MAJOR_VER:
        return CompatResult(
            compatible=False,
            mode='incompatible',
            reason=f'libcudart 大版本 {libcudart_major} 已过期或未来版本',
            actual_cuda='',
            limitations=['需要更新 CUDA 工具链']
        )

    # 获取驱动支持版本的 major 和 supported 版本的 (major, minor)
    driver_max_major, driver_max_minor = version_tuple_from_float(max_cuda_supported)
    driver_max_version_str = f'{driver_max_major}.{driver_max_minor}'
    cudart_version_tuple = (cudart_major, cudart_minor)
    driver_max_version_tuple = (driver_max_major, driver_max_minor)

    cmp = compare_versions(cudart_version_tuple, driver_max_version_tuple)

    # 情况 1：libcudart 版本 < 驱动最高支持版本，后向兼容
    if cmp < 0:
        return CompatResult(
            compatible=True,
            mode='backward',
            reason=f'libcudart {cudart_version_str} 低于驱动支持的最高版本 {max_cuda_supported}，后向兼容',
            actual_cuda=cudart_version_str,
            limitations=[
                '没有 PTX JIT，只能运行 SASS 代码',
                '个别新接口可能返回 cudaErrorCallRequiresNewerDriver',
                '部分 Hopper 新特性不可用',
            ]
        )

    # 情况 2：libcudart 版本 = 驱动最高支持版本，精确匹配
    if cmp == 0:
        return CompatResult(
            compatible=True,
            mode='forward',
            reason=f'libcudart {cudart_version_str} 完全匹配驱动支持版本',
            actual_cuda=cudart_version_str,
            limitations=['无已知限制']
        )

    # 情况 3：libcudart 版本 > 驱动最高支持版本，但同大版本，前向兼容
    if libcudart_major == driver_max_major:
        return CompatResult(
            compatible=True,
            mode='forward',
            reason=f'libcudart {cudart_version_str} 高于驱动支持版本 {max_cuda_supported}，同大版本前向兼容',
            actual_cuda=driver_max_version_str,
            limitations=[
                '程序会自动降级到驱动支持的最高版本',
                f'会降级到 CUDA {max_cuda_supported} 的功能集',
            ]
        )

    # 情况 4：libcudart 跨大版本（更高的大版本），需要 cuda-compat 包
    if libcudart_major > driver_max_major:
        if not CUDA_MAJOR_VER[libcudart_major]['compat_pkg_avail']:
            return CompatResult(
                compatible=False,
                mode='incompatible',
                reason=f'libcudart {cudart_version_str} 比驱动支持的版本 {max_cuda_supported} 高一个大版本，该 CUDA 版本无 compat 包',
                actual_cuda='',
                limitations=['需要升级驱动或更换程序的 CUDA 版本']
            )

        # 有 compat 包的情况
        if compat_pkg_ver:
            compat_major, compat_minor, _ = parse_version(compat_pkg_ver)
            compat_version_tuple = (compat_major, compat_minor)
            compat_cmp = compare_versions(compat_version_tuple, cudart_version_tuple)
            if compat_cmp >= 0:
                return CompatResult(
                    compatible=True,
                    mode='compat-pkg',
                    reason=f'使用 cuda-compat {compat_pkg_ver} 包垫在驱动 R{driver_ver} 上支持 libcudart {cudart_version_str}',
                    actual_cuda=cudart_version_str,
                    limitations=[
                        '仅支持数据中心卡（A100/H100/L40S 等），消费卡会报错',
                        f'cuda-compat 包版本必须 >= {cudart_version_str}',
                        '性能可能略低于驱动原生版本',
                    ]
                )
            else:
                return CompatResult(
                    compatible=False,
                    mode='incompatible',
                    reason=f'cuda-compat 包版本 {compat_pkg_ver} 低于程序需要的 {cudart_version_str}',
                    actual_cuda='',
                    limitations=['升级 cuda-compat 包版本']
                )
        else:
            return CompatResult(
                compatible=False,
                mode='incompatible',
                reason=f'libcudart {cudart_version_str} 比驱动支持的版本高一个大版本，需要安装 cuda-compat 包',
                actual_cuda='',
                limitations=[
                    f'安装 cuda-compat >= {cudart_version_str}',
                    '仅数据中心卡可用',
                ]
            )

    # 不应到达此处
    return CompatResult(
        compatible=False,
        mode='incompatible',
        reason='未知兼容性规则',
        actual_cuda='',
        limitations=['联系维护者']
    )


def cmd_check(args):
    """执行 check 子命令。"""
    result = check_compat(args.driver, args.cudart, args.compat_pkg)
    print(f'驱动 R{args.driver} + libcudart {args.cudart}：', end=' ')
    if result.compatible:
        print(f'✓ {result.mode}')
    else:
        print(f'✗ {result.mode}')
    print(f'  理由：{result.reason}')
    if result.actual_cuda:
        print(f'  实际 CUDA 版本：{result.actual_cuda}')
    print(f'  限制：')
    for lim in result.limitations:
        print(f'    · {lim}')


def cmd_jumps(args):
    """执行 jumps 子命令，展示常见升级场景。"""
    print('常见驱动与 CUDA 版本组合：')
    print()

    # 场景表：（驱动版本，libcudart 版本，可选 compat 包）
    scenarios = [
        (615, '13.4', None),   # CUDA 13.4 Update 2 基线
        (580, '13.0', None),   # R580 支持 CUDA 13.x 起点
        (610, '13.3', None),   # R610 与 13.3 版本匹配
        (550, '12.4', None),   # CUDA 12 LTS 线
        (545, '12.1', None),   # 后向兼容：12.1 < 12.1
        (550, '13.0', '13.0-2'),  # 跨大版本需要 compat（R550 仅支持 12.4）
        (515, '11.8', None),   # 已停支驱动 + 同大版本
    ]

    for driver, cudart, compat in scenarios:
        result = check_compat(driver, cudart, compat)
        status = '✓' if result.compatible else '✗'
        compat_str = f'（cuda-compat {compat}）' if compat else ''
        print(f'{status} 驱动 R{driver:3d} + libcudart {cudart} {compat_str}')
        print(f'  → {result.mode:15s} {result.reason}')
        if result.actual_cuda:
            print(f'     实际运行 CUDA {result.actual_cuda}')
        print()


def cmd_jetson(args):
    """执行 jetson 子命令。Jetson 卡的容器化更复杂。"""
    # Jetson 系列有独立的 CUDA 版本号与驱动绑定。
    # 来源：NVIDIA Jetson Platform Services，https://developer.nvidia.com/embedded/jetson-orin
    print(f'JetPack {args.jetpack}（驱动嵌入版本）：')
    jetpack_cuda_map = {
        '5.1': {'cuda': '11.4', 'driver': 'embedded'},
        '6.0': {'cuda': '12.2', 'driver': 'embedded'},
        '6.1': {'cuda': '12.2', 'driver': 'embedded'},
    }

    if args.jetpack not in jetpack_cuda_map:
        print(f'  未知 JetPack 版本 {args.jetpack}')
        return

    jp_info = jetpack_cuda_map[args.jetpack]
    print(f'  内置 CUDA：{jp_info["cuda"]}')
    print(f'  容器内若装了 cuda-toolkit {args.container_cuda}：')
    result = check_compat(615, args.container_cuda)  # 用 R615 作参考（CUDA 13.x 基线）
    print(f'    {result.mode}（基于 CUDA 兼容规则）')
    print(f'    注意：Jetson 容器与宿主驱动共享，需要验证容器内 nvidia-smi 输出')


def main():
    parser = argparse.ArgumentParser(
        description='CUDA 版本兼容检查器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command')

    # check 子命令
    check_parser = subparsers.add_parser('check', help='检查驱动与 libcudart 兼容性')
    check_parser.add_argument('--driver', type=int, required=True, help='驱动版本号（如 550）')
    check_parser.add_argument('--cudart', type=str, required=True, help='libcudart 版本（如 12.4）')
    check_parser.add_argument('--compat-pkg', type=str, help='可选，cuda-compat 包版本')
    check_parser.set_defaults(func=cmd_check)

    # jumps 子命令
    jumps_parser = subparsers.add_parser('jumps', help='常见版本组合矩阵')
    jumps_parser.set_defaults(func=cmd_jumps)

    # jetson 子命令
    jetson_parser = subparsers.add_parser('jetson', help='Jetson 卡特殊兼容规则')
    jetson_parser.add_argument('--jetpack', type=str, default='6.0', help='JetPack 版本')
    jetson_parser.add_argument('--container-cuda', type=str, default='13.4', help='容器内 CUDA 版本')
    jetson_parser.set_defaults(func=cmd_jetson)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
