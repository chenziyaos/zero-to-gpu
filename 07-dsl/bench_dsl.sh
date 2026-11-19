#!/usr/bin/env bash
# Triton、CuTe DSL 和 CUTLASS 3.x 的矩阵乘 benchmark 脚本。
#
# 本脚本在 GPU 上运行，需要环境：
#   - NVIDIA GPU: Ampere (A100) 或 Hopper (H100) 及以后
#   - PyTorch 2.15+
#   - Triton 2.2+
#   - NVIDIA CUTLASS 3.x 和 nvidia-cutlass-dsl
#
# 使用方式：
#   bash bench_dsl.sh
#   bash bench_dsl.sh --m 8192 --n 8192 --k 8192 --triton-only
#
# 本仓库未在真卡上复测，期望值来自文献 [1]。

set -Eeuo pipefail

# 默认参数
M=${M:-4096}
N=${N:-4096}
K=${K:-4096}
DTYPE=${DTYPE:-"f32"}
TRITON_ONLY=${TRITON_ONLY:-0}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --m) M=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --k) K=$2; shift 2 ;;
        --dtype) DTYPE=$2; shift 2 ;;
        --triton-only) TRITON_ONLY=1; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "======== Triton 矩阵乘 Benchmark ========"
echo "矩阵大小: ${M}×${N}×${K}, dtype: ${DTYPE}"
echo ""

# Triton benchmark（Python）
if command -v python3 &> /dev/null; then
    echo "Triton kernel..."
    # python3 gpu_triton_mm.py --m $M --n $N --k $K --dtype $DTYPE
    echo "  [stub: 本仓库未实现真实 benchmark，需在 GPU 机器上手动运行]"
else
    echo "⚠ Python 未找到"
fi

# CuTe DSL benchmark（如果已安装 nvidia-cutlass-dsl）
if [ "$TRITON_ONLY" -eq 0 ]; then
    echo ""
    echo "CuTe DSL kernel..."
    # python3 gpu_cute_add.py --m $M --n $N --k $K
    echo "  [stub: 需要 nvidia-cutlass-dsl]"
fi

echo ""
echo "======== Benchmark 结束 ========"
echo "期望输出："
echo "  - Triton 矩阵乘：100 GEMM 运行的平均吞吐量（TFLOPS）"
echo "  - CuTe kernel：同一矩阵大小下的性能对比"
