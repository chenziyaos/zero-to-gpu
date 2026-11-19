"""Xid 错误诊断工具：从 dmesg 日志中解析 GPU 错误代码，给出可能的根因和建议。对应 GPU-09 第 9、10 章。

Xid（NVIDIA GPU X.id ecc or critical error）是驱动与 GPU 之间通信失败、硬件故障或固件问题的编码。
NVIDIA GPU 驱动在检测到严重硬件错误时会输出：
  NVRM: Xid <gpu_id>.<error_code>: <description>

每个 error_code 有固定的含义，指向显存错误、超时、ECC 故障、功耗过高、温度过高、链路失败等。

子命令：
  parse      解析 dmesg 中的 Xid 行，给出中文诊断和建议的第一步动作。
  catalog    列出常见的 Xid 代码及其含义（本地参考）。
  simulate   生成一个 Xid 日志示例，用于测试解析逻辑。

用法：
  python3 sim_xid.py parse --log /var/log/kern.log
  python3 sim_xid.py catalog --filter memory
  python3 sim_xid.py simulate --code 5
  grep "Xid" /var/log/messages | python3 sim_xid.py parse --stdin
"""
import argparse
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# Xid 错误代码目录，根据 NVIDIA 官方文档构建。
# 来源：https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
# 核对日期：2026-11-05
# 注：已移除标注为"Unused"或"Deprecated"的代码 (1,2,3,5,6,14,24,29)
XID_CATALOG = {
    8: {
        'name': 'GPU stopped processing',
        'category': 'timeout',
        'severity': 'high',
        'description': 'GPU 停止处理，通常源于 kernel 执行过长或系统过载',
        'common_causes': [
            '应用程序导致 GPU 停止响应',
            '系统资源不足导致调度延迟',
        ],
        'actions': [
            '1. 重启应用程序',
            '2. 检查系统资源使用情况',
            '3. 如问题持续，升级驱动',
        ],
        'sla_hours': 4,
    },
    11: {
        'name': 'Invalid or corrupted push buffer stream',
        'category': 'buffer',
        'severity': 'high',
        'description': '无效或损坏的命令缓冲流，推送缓冲区状态不一致',
        'common_causes': [
            '驱动或应用程序生成的命令缓冲格式错误',
            '显存损坏导致的缓冲区指针指向无效地址',
        ],
        'actions': [
            '1. 重启应用程序',
            '2. 检查显存健康状况',
            '3. 升级驱动到最新版本',
        ],
        'sla_hours': 2,
    },
    13: {
        'name': 'Graphics Engine Exception',
        'category': 'graphics',
        'severity': 'high',
        'description': '图形处理引擎异常，可能来自硬件故障、驱动问题、用户应用或系统内存损坏',
        'common_causes': [
            '用户应用程序逻辑错误',
            '驱动软件故障',
            '系统内存损坏或硬件故障',
        ],
        'actions': [
            '1. 重启应用程序或系统',
            '2. 使用调试工具或硬件诊断排查',
            '3. 升级驱动版本',
        ],
        'sla_hours': 2,
    },
    31: {
        'name': 'GPU memory page fault',
        'category': 'memory',
        'severity': 'critical',
        'description': 'GPU 内存页面故障，非法地址访问或页表错误',
        'common_causes': [
            '应用程序的指针错误',
            '显存映射错误',
        ],
        'actions': [
            '1. 检查应用程序内存访问的合法性',
            '2. 重启应用程序或调试',
            '3. 如持续触发，下线 GPU 进行诊断',
        ],
        'sla_hours': 2,
    },
    32: {
        'name': 'Invalid or corrupted push buffer stream (PBDMA)',
        'category': 'buffer',
        'severity': 'high',
        'description': '无效或损坏的命令缓冲流，PBDMA（推送缓冲直接内存访问）错误',
        'common_causes': [
            '命令缓冲格式错误',
            '内存故障导致的数据损坏',
        ],
        'actions': [
            '1. 重启应用程序',
            '2. 检查显存和系统内存',
            '3. 升级驱动',
        ],
        'sla_hours': 2,
    },
    43: {
        'name': 'GPU stopped processing',
        'category': 'recovery',
        'severity': 'medium',
        'description': 'GPU 停止处理，通常来自应用程序错误而非硬件问题',
        'common_causes': [
            '应用程序逻辑错误导致 GPU 重置',
            '应用程序被中断',
        ],
        'actions': [
            '1. 忽略此错误，或检查关联的其他 Xid 错误',
            '2. 重新运行应用程序',
        ],
        'sla_hours': 4,
    },
    45: {
        'name': 'Preemptive cleanup due to previous errors',
        'category': 'recovery',
        'severity': 'high',
        'description': '由于前面的错误而进行预抢占式清理，GPU 或应用程序在恢复中',
        'common_causes': [
            '前面的 Xid 错误触发了清理流程',
            '应用程序被中止或 GPU 重置',
        ],
        'actions': [
            '1. 查看此错误前的其他 Xid 错误代码',
            '2. 等待清理完成后重新运行',
            '3. 检查 GPU 是否恢复正常',
        ],
        'sla_hours': 2,
    },
    48: {
        'name': 'Double Bit ECC Error',
        'category': 'ecc_memory',
        'severity': 'critical',
        'description': 'ECC 无法纠正的双比特错误，显存数据已损坏，GPU 需要重置',
        'common_causes': [
            '显存物理故障',
            '严重的硬件降级',
        ],
        'actions': [
            '1. 重置 GPU',
            '2. 立即下线并进行完整诊断',
            '3. 准备硬件更换或 RMA',
        ],
        'sla_hours': 1,
    },
    63: {
        'name': 'ECC page retirement or row remapping recording event',
        'category': 'ecc_memory',
        'severity': 'low',
        'description': 'GPU 内存页面被自动隔离，行级故障被系统处理',
        'common_causes': [
            '显存某行出现错误被系统自动隔离',
            '硬件老化导致的故障自动修复',
        ],
        'actions': [
            '1. 忽略此错误，监控频率',
            '2. 启用 ECC 日志了解更多',
            '3. 长期规划硬件更换',
        ],
        'sla_hours': 24,
    },
    64: {
        'name': 'ECC page retirement or row remapper recording failure',
        'category': 'ecc_memory',
        'severity': 'critical',
        'description': 'GPU 内存行重映射失败，隔离机制本身出现问题',
        'common_causes': [
            '显存故障过于严重',
            '隔离容量已耗尽',
        ],
        'actions': [
            '1. 重置 GPU',
            '2. 立即下线进行诊断',
            '3. 准备硬件更换',
        ],
        'sla_hours': 1,
    },
    74: {
        'name': 'NVLINK Error',
        'category': 'nvlink',
        'severity': 'critical',
        'description': 'NVLink 连接错误，多 GPU 间通信失败',
        'common_causes': [
            'NVLink 物理连接故障',
            'NVLink 驱动管理故障',
        ],
        'actions': [
            '1. 检查 NVLink 物理连接',
            '2. 运行多 GPU 测试验证通信',
            '3. 咨询技术支持',
        ],
        'sla_hours': 1,
    },
    79: {
        'name': 'GPU has fallen off the bus',
        'category': 'pcie',
        'severity': 'critical',
        'description': 'GPU 从 PCIe 总线消失，PCIe 访问失败，通信中断',
        'common_causes': [
            'PCIe 物理连接断裂或接触不良',
            'PCIe 链路错误',
        ],
        'actions': [
            '1. 重启系统',
            '2. 检查 PCIe 插槽和连接',
            '3. 如问题持续，更换硬件',
        ],
        'sla_hours': 1,
    },
    92: {
        'name': 'High single-bit ECC error rate',
        'category': 'ecc_memory',
        'severity': 'high',
        'description': 'ECC 单比特错误频率过高，表明硬件或驱动问题',
        'common_causes': [
            '显存即将故障',
            '驱动或硬件降级',
        ],
        'actions': [
            '1. 忽略单发错误，监控频率',
            '2. 如频率上升，下线 GPU',
            '3. 咨询技术支持或准备更换',
        ],
        'sla_hours': 4,
    },
    94: {
        'name': 'Contained ECC error',
        'category': 'ecc_memory',
        'severity': 'medium',
        'description': '已包含的 ECC 错误，错误被隔离到单个应用程序',
        'common_causes': [
            '应用程序执行中发生单比特错误且被纠正',
            '显存老化导致的偶发错误',
        ],
        'actions': [
            '1. 重启应用程序',
            '2. 监控错误频率',
            '3. 长期关注显存健康',
        ],
        'sla_hours': 24,
    },
    95: {
        'name': 'Uncontained ECC error',
        'category': 'ecc_memory',
        'severity': 'critical',
        'description': '未包含的 ECC 错误，影响多个应用程序，GPU 需要重置',
        'common_causes': [
            '严重的显存故障',
            '驱动或硬件大范围故障',
        ],
        'actions': [
            '1. 重置 GPU',
            '2. 立即下线进行诊断',
            '3. 准备硬件更换',
        ],
        'sla_hours': 1,
    },
    119: {
        'name': 'GSP RPC Timeout',
        'category': 'firmware',
        'severity': 'high',
        'description': '固件 RPC 超时，固件无响应',
        'common_causes': [
            '固件故障',
            '驱动通信问题',
            '系统过载',
        ],
        'actions': [
            '1. 重置 GPU 或系统',
            '2. 升级驱动',
            '3. 检查系统资源',
        ],
        'sla_hours': 2,
    },
    120: {
        'name': 'GSP Error',
        'category': 'firmware',
        'severity': 'high',
        'description': '通用固件错误，GPU 需要重置',
        'common_causes': [
            '固件内部错误',
            '驱动 bug',
            '硬件故障',
        ],
        'actions': [
            '1. 重置 GPU 或系统',
            '2. 升级驱动到最新版本',
            '3. 联系技术支持',
        ],
        'sla_hours': 2,
    },
}

@dataclass
class XidDiagnosis:
    code: int
    gpu_id: int
    line_raw: str
    name: Optional[str]
    category: Optional[str]
    severity: Optional[str]
    description: Optional[str]
    common_causes: list[str]
    actions: list[str]
    is_critical: bool

def parse_xid_line(line: str) -> Optional[Tuple[int, int]]:
    """
    从 dmesg 行中提取 Xid 代码和 GPU ID。
    Expected format: 'NVRM: Xid <gpu_id>.<code>: <description>'
    也支持其他常见的 dmesg 日期前缀。
    """
    match = re.search(r'Xid\s+(\d+)\.(\d+):', line)
    if match:
        gpu_id = int(match.group(1))
        code = int(match.group(2))
        return gpu_id, code
    return None

def diagnose_xid(gpu_id: int, code: int, raw_line: str = '') -> XidDiagnosis:
    """根据 Xid 代码给出诊断。"""
    if code in XID_CATALOG:
        info = XID_CATALOG[code]
        return XidDiagnosis(
            code=code,
            gpu_id=gpu_id,
            line_raw=raw_line,
            name=info['name'],
            category=info['category'],
            severity=info['severity'],
            description=info['description'],
            common_causes=info['common_causes'],
            actions=info['actions'],
            is_critical=info['severity'] in ['critical', 'high']
        )
    else:
        return XidDiagnosis(
            code=code,
            gpu_id=gpu_id,
            line_raw=raw_line,
            name=f'Unknown Xid {code}',
            category='unknown',
            severity='unknown',
            description='此 Xid 代码未在本工具的数据库中',
            common_causes=['需要查阅官方 NVIDIA 文档或驱动日志'],
            actions=['1. 查看完整的 dmesg 上下文', '2. 升级驱动并重试', '3. 联系 NVIDIA 技术支持'],
            is_critical=False
        )

def format_diagnosis(diag: XidDiagnosis) -> str:
    """格式化诊断结果为易读的中文输出。"""
    lines = []
    lines.append(f'═══════════════════════════════════════')
    lines.append(f'GPU #{diag.gpu_id} Xid {diag.code}：{diag.name}')
    lines.append(f'═══════════════════════════════════════')
    lines.append(f'分类：{diag.category}  严重程度：{diag.severity}')
    lines.append(f'')
    lines.append(f'含义：')
    lines.append(f'  {diag.description}')
    lines.append(f'')
    lines.append(f'常见原因：')
    for cause in diag.common_causes:
        lines.append(f'  · {cause}')
    lines.append(f'')
    lines.append(f'建议的第一步：')
    for action in diag.actions:
        lines.append(f'  {action}')
    lines.append(f'')
    return '\n'.join(lines)

def cmd_parse(args):
    """执行 parse 子命令，从日志中提取并诊断 Xid。"""
    lines_to_parse = []

    if args.stdin:
        lines_to_parse = sys.stdin.readlines()
    elif args.log:
        try:
            with open(args.log, 'r') as f:
                lines_to_parse = f.readlines()
        except FileNotFoundError:
            print(f'错误：文件 {args.log} 不存在', file=sys.stderr)
            return

    xids_found = []
    for line in lines_to_parse:
        parsed = parse_xid_line(line)
        if parsed:
            gpu_id, code = parsed
            xids_found.append((gpu_id, code, line.strip()))

    if not xids_found:
        print('未发现 Xid 错误，日志正常。')
        return

    print(f'发现 {len(xids_found)} 条 Xid 错误：')
    print()

    critical_count = 0
    for gpu_id, code, line in xids_found:
        diag = diagnose_xid(gpu_id, code, line)
        print(format_diagnosis(diag))
        if diag.is_critical:
            critical_count += 1

    print(f'总结：{critical_count}/{len(xids_found)} 条为严重错误，需要立即处理。')


def cmd_catalog(args):
    """执行 catalog 子命令，列出 Xid 代码库。"""
    print('NVIDIA GPU Xid 错误代码目录')
    print('=' * 60)
    print()

    codes_to_show = XID_CATALOG.keys()
    if args.filter:
        codes_to_show = [c for c in codes_to_show
                         if args.filter.lower() in XID_CATALOG[c]['category'].lower()]

    for code in sorted(codes_to_show):
        info = XID_CATALOG[code]
        print(f'Xid {code:2d}：{info["name"]}')
        print(f'       分类：{info["category"]:15s} 严重：{info["severity"]:10s}')
        print(f'       {info["description"]}')
        print()


def cmd_simulate(args):
    """执行 simulate 子命令，生成示例 Xid 日志行。"""
    if args.code not in XID_CATALOG:
        print(f'错误：Xid {args.code} 不在目录中', file=sys.stderr)
        print(f'可用代码：{sorted(XID_CATALOG.keys())}', file=sys.stderr)
        return

    gpu_id = args.gpu_id
    info = XID_CATALOG[args.code]
    example_line = f'[某日期] NVRM: Xid {gpu_id}.{args.code}: {info["name"]}'

    print(f'示例日志行：')
    print(example_line)
    print()

    diag = diagnose_xid(gpu_id, args.code, example_line)
    print(format_diagnosis(diag))


def main():
    parser = argparse.ArgumentParser(
        description='GPU Xid 错误诊断工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command')

    # parse 子命令
    parse_parser = subparsers.add_parser('parse', help='解析 dmesg 日志中的 Xid')
    parse_parser.add_argument('--log', type=str, help='日志文件路径（如 /var/log/kern.log）')
    parse_parser.add_argument('--stdin', action='store_true', help='从标准输入读取日志')
    parse_parser.set_defaults(func=cmd_parse)

    # catalog 子命令
    catalog_parser = subparsers.add_parser('catalog', help='列出 Xid 代码目录')
    catalog_parser.add_argument('--filter', type=str, help='按分类筛选（如 memory、thermal、compute）')
    catalog_parser.set_defaults(func=cmd_catalog)

    # simulate 子命令
    simulate_parser = subparsers.add_parser('simulate', help='生成示例 Xid 日志')
    simulate_parser.add_argument('--code', type=int, required=True, help='Xid 代码号')
    simulate_parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID（默认 0）')
    simulate_parser.set_defaults(func=cmd_simulate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
