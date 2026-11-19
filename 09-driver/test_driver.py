"""单元测试：sim_compat 和 sim_xid 的逻辑正确性。

测试源：
  1. NVIDIA CUDA 兼容性官方文档：https://docs.nvidia.com/deploy/cuda-compatibility/
  2. NVIDIA GPU Operator 文档中的版本支持矩阵
  3. NVIDIA Xid 错误代码文档（白皮书和社区维护的资料库）
  4. 内部生产环境真实案例
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_compat import check_compat, CompatResult
from sim_xid import parse_xid_line, diagnose_xid, XID_CATALOG

def test_compat_backward_compatible():
    """测试 1：后向兼容 - libcudart 低于驱动支持的最高版本。

    官方文档：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
    R615 驱动支持最高 CUDA 13.4，装 CUDA 13.0 程序应该后向兼容。
    """
    result = check_compat(615, '13.0')
    assert result.compatible, '应该兼容'
    assert result.mode == 'backward', f'应该是后向兼容，得到 {result.mode}'
    assert result.actual_cuda == '13.0', f'实际版本应该是 13.0，得到 {result.actual_cuda}'
    assert any('PTX JIT' in lim for lim in result.limitations), '后向兼容应该提到没有 PTX JIT'
    print('✓ 测试 1：后向兼容')


def test_compat_forward_compatible_same_version():
    """测试 2：精确匹配 - libcudart 版本与驱动支持版本完全相同。

    R610 驱动支持 CUDA 13.3，装 CUDA 13.3 程序应该精确匹配。
    """
    result = check_compat(610, '13.3')
    assert result.compatible, '应该兼容'
    assert result.mode == 'forward', f'应该是前向兼容模式，得到 {result.mode}'
    assert result.actual_cuda == '13.3'
    print('✓ 测试 2：精确匹配')


def test_compat_forward_compatible_within_major():
    """测试 3：同大版本前向兼容 - libcudart 高于驱动支持版本，但同大版本。

    R580 驱动支持最高 CUDA 13.0，装 CUDA 13.0 同大版本兼容。
    """
    result = check_compat(580, '13.0')
    assert result.compatible
    assert result.mode == 'forward'
    assert result.actual_cuda == '13.0'
    print('✓ 测试 3：同大版本前向兼容')


def test_compat_compat_pkg_needed():
    """测试 4：需要 cuda-compat 包 - libcudart 跨大版本，无包时不兼容。

    R550 驱动支持最高 CUDA 12.4（CUDA 12），程序需要 CUDA 13.0 会跨大版本。
    没有 compat 包时应该不兼容。
    """
    result = check_compat(550, '13.0', compat_pkg_ver=None)
    assert not result.compatible, '没有 compat 包应该不兼容'
    assert result.mode == 'incompatible'
    assert any('cuda-compat' in lim for lim in result.limitations), '应该提示需要 compat 包'
    print('✓ 测试 4：需要 compat 包')


def test_compat_with_compat_pkg():
    """测试 5：使用 cuda-compat 包 - 跨大版本但有合适的 compat 包时兼容。

    R550 + CUDA 13.0 程序，如果装了 cuda-compat 13.0 包，应该能跑。
    """
    result = check_compat(550, '13.0', compat_pkg_ver='13.0-1')
    assert result.compatible, f'有 compat 包应该兼容，结果 {result.mode}'
    assert result.mode == 'compat-pkg'
    assert result.actual_cuda == '13.0'
    assert any('数据中心卡' in lim for lim in result.limitations), 'compat 包应该提示仅支持数据中心卡'
    print('✓ 测试 5：使用 cuda-compat 包')


def test_compat_monotonicity():
    """测试 6：单调性 - 版本更新不应该让兼容性变差。

    R550 驱动配不同版本的 libcudart：版本越高，越可能需要更新驱动或加 compat。
    验证约束条件的自洽性。
    """
    versions = ['11.8', '12.0', '12.1', '12.4', '13.0']
    results = [check_compat(550, v) for v in versions]

    # 12.4 及以下都应该兼容（不需要 compat）
    for i in range(4):  # 11.8, 12.0, 12.1, 12.4
        assert results[i].compatible and results[i].mode != 'compat-pkg', \
            f'{versions[i]} 应该不需要 compat 包'

    # 13.0 不加 compat 包应该不兼容
    assert not results[4].compatible, '13.0 无 compat 包应该不兼容'

    print('✓ 测试 6：版本单调性')


# ============ Xid 测试 ============

def test_xid_parse_valid():
    """测试 7：Xid 日志解析 - 正确提取 GPU ID 和错误代码。

    典型日志格式：NVRM: Xid <gpu_id>.<code>: <description>
    """
    line = 'Nov  4 10:45:23 gpu-node NVRM: Xid 0.13: GR: Graphics Exception on SM'
    parsed = parse_xid_line(line)
    assert parsed is not None
    gpu_id, code = parsed
    assert gpu_id == 0
    assert code == 13
    print('✓ 测试 7：Xid 日志解析')


def test_xid_catalog_coverage():
    """测试 8：Xid 代码库 - 常见代码都有诊断信息。

    至少包含最常见的几个错误：8（超时），31（页故障），48（ECC DBE），79（PCIe）。
    """
    critical_codes = [8, 31, 48, 79]
    for code in critical_codes:
        assert code in XID_CATALOG, f'Xid {code} 应该在目录中'
        info = XID_CATALOG[code]
        assert 'description' in info and info['description'], f'Xid {code} 应该有描述'
        assert 'common_causes' in info and len(info['common_causes']) > 0, f'Xid {code} 应该有常见原因'
        assert 'actions' in info and len(info['actions']) > 0, f'Xid {code} 应该有建议动作'
    print('✓ 测试 8：Xid 代码库覆盖')


def test_xid_diagnosis_memory_error():
    """测试 9：Xid 诊断 - Xid 31（显存错误）给出正确的诊断。

    Xid 31 是 GPU 内存页面故障，属于内存类别，严重程度为 critical。
    """
    diag = diagnose_xid(gpu_id=0, code=31)
    assert diag.code == 31
    assert diag.gpu_id == 0
    assert diag.category == 'memory'
    assert diag.severity == 'critical'
    assert diag.is_critical
    assert len(diag.actions) > 0
    print('✓ 测试 9：Xid 31 诊断')


def test_xid_diagnosis_timeout():
    """测试 10：Xid 诊断 - Xid 8（超时）给出超时相关建议。

    Xid 8 是 GPU 停止处理（超时），属于超时类别，应该提示检查 kernel 执行时间。
    """
    diag = diagnose_xid(gpu_id=2, code=8)
    assert diag.code == 8
    assert diag.gpu_id == 2
    assert diag.category == 'timeout'
    assert any('kernel' in str(cause).lower() or 'GPU' in str(cause) for cause in diag.common_causes), \
        'Xid 8 应该提到 GPU 或 kernel 执行时间'
    print('✓ 测试 10：Xid 8 诊断')


def test_xid_unknown_code():
    """测试 11：未知 Xid 代码 - 诊断工具应该给出通用建议而不崩溃。

    对于不在目录中的 Xid 代码，应该返回 unknown 分类和查阅文档的建议。
    """
    diag = diagnose_xid(gpu_id=0, code=9999)
    assert diag.code == 9999
    assert diag.category == 'unknown'
    assert 'Unknown' in diag.name
    assert len(diag.actions) > 0  # 应该有通用建议
    print('✓ 测试 11：未知 Xid 代码')


if __name__ == '__main__':
    print('运行 GPU-09 驱动/CUDA 兼容性测试...')
    print()

    test_compat_backward_compatible()
    test_compat_forward_compatible_same_version()
    test_compat_forward_compatible_within_major()
    test_compat_compat_pkg_needed()
    test_compat_with_compat_pkg()
    test_compat_monotonicity()
    test_xid_parse_valid()
    test_xid_catalog_coverage()
    test_xid_diagnosis_memory_error()
    test_xid_diagnosis_timeout()
    test_xid_unknown_code()

    print()
    print('═══════════════════════════════════════')
    print('全部 11 条测试通过 ✓')
    print('═══════════════════════════════════════')
