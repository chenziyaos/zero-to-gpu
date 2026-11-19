#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 节点检查脚本：验证驱动版本、CUDA 兼容性和 GPU 健康状态。
# 对应 GPU-09 第 8 到 10 章，用于上线前的驱动与 CUDA 环境核对。
#
# 依赖版本：nvidia-driver R580+（2026-11 基线）
# 权限要求：部分命令需要 root（如 dmesg -c），无权限时会跳过
# 本仓库未在真卡上复测，期望值来自 NVIDIA 官方文档和 GPU-09 数字校准。
#
# 用法：
#   ./check_node.sh                    # 快速检查
#   ./check_node.sh --full             # 完整诊断，包括 ECC 和详细日志

FULL_CHECK="${1:-}"

echo "=== GPU 节点环境检查 ==="
echo "核对日期：2026-11-02"
echo

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ 找不到 nvidia-smi，驱动可能未安装"
    exit 1
fi

echo "✓ nvidia-smi 可用"

# 1. 获取驱动版本
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "驱动版本：$driver_version"

# 2. 获取 CUDA Version（驱动支持的最高运行时）
cuda_version=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 || echo "unknown")
cuda_runtime_version=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "CUDA Version（驱动口径）：见 nvidia-smi 右上角"
nvidia-smi | grep "CUDA Version"

# 3. GPU 信息
echo
echo "GPU 卡信息："
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "  卡数：$gpu_count"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  型号：$gpu_name"

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,unit=MB | head -1)
echo "  显存：$gpu_memory"

# 4. 显存健康检查
echo
echo "显存状态："
for i in $(seq 0 $((gpu_count - 1))); do
    mem_free=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,unit=MB)
    mem_total=$(nvidia-smi -i "$i" --query-gpu=memory.total --format=csv,noheader,unit=MB)
    echo "  GPU $i：$mem_free / $mem_total"
done

# 5. 温度检查
echo
echo "GPU 温度："
for i in $(seq 0 $((gpu_count - 1))); do
    temp=$(nvidia-smi -i "$i" --query-gpu=temperature.gpu --format=csv,noheader,unit=C || echo "N/A")
    echo "  GPU $i：$temp"
done

# 6. 电源状态
echo
echo "电源管理："
pstate=$(nvidia-smi -pm 1 2>/dev/null && echo "已启用" || echo "未检查（需要 root）")
echo "  持久模式：$pstate"

# 7. PCIe 链路检查（如果可用）
echo
echo "PCIe 链路："
if command -v lspci &> /dev/null; then
    lspci | grep -i nvidia | head -3 || echo "  （无法获取 PCIe 信息）"
else
    echo "  （lspci 不可用）"
fi

# 8. dmesg 中的 Xid 和错误（需要 root）
if [ "$FULL_CHECK" = "--full" ]; then
    echo
    echo "近期 Xid 错误（最近 100 行）："
    if dmesg 2>/dev/null | grep -i "Xid\|NVRM" | tail -10; then
        echo "  （发现错误，请用 sim_xid.py 诊断）"
    else
        echo "  ✓ 无 Xid 错误记录"
    fi

    echo
    echo "ECC 错误统计（如支持）："
    for i in $(seq 0 $((gpu_count - 1))); do
        ecc_errors=$(nvidia-smi -i "$i" --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader 2>/dev/null || echo "N/A")
        if [ "$ecc_errors" != "N/A" ]; then
            echo "  GPU $i ECC 可纠正错误：$ecc_errors"
        fi
    done
fi

echo
echo "=== 检查完毕 ==="
echo "后续步骤："
echo "  1. 对比期望的驱动版本（R550+ 推荐）"
echo "  2. 如有 Xid 错误，用 python3 sim_xid.py parse --log /var/log/kern.log"
echo "  3. 验证容器内的 libcudart 版本与驱动兼容（见 GPU-09 第 3 至 7 章）"
