#!/usr/bin/env bash
# One-click profiling script: nsys + ncu on a GPU workload.
# 对应 GPU-14 第 7 章：抓取脚本与权限排障。
# 未在本仓库真卡上复测; 期望值来自 NVIDIA Nsight 工具文档。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/profile_output"
WORKLOAD="${1:-gpu_workload.py}"
WORKLOAD_ID="${2:-1}"

mkdir -p "$OUTPUT_DIR"

echo "GPU Profiling Script"
echo "===================="
echo "Workload: $WORKLOAD (ID: $WORKLOAD_ID)"
echo "Output: $OUTPUT_DIR"
echo ""

# Check for required tools
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys not found. Install NVIDIA Nsight Systems."
    exit 1
fi

if ! command -v ncu &> /dev/null; then
    echo "Error: ncu not found. Install NVIDIA Nsight Compute."
    exit 1
fi

# Stage 1: Nsys timeline profiling (low overhead)
echo "Stage 1: Nsight Systems (timeline + statistics)"
NSYS_OUTPUT="${OUTPUT_DIR}/profile_nsys"
nsys profile \
    --trace=cuda,cudnn,nvtx \
    --stats=true \
    --output="${NSYS_OUTPUT}" \
    python3 "$WORKLOAD" "$WORKLOAD_ID" || true

# Export CSV from nsys
if [ -f "${NSYS_OUTPUT}.nsys-rep" ]; then
    echo "Exporting nsys kernels to CSV..."
    nsys export --type csv "${NSYS_OUTPUT}.nsys-rep" \
        --output-directory "$OUTPUT_DIR" || true
fi

echo "✓ Nsys output: ${OUTPUT_DIR}/"
echo ""

# Stage 2: NCU kernel replay (high overhead, only on selected kernels)
echo "Stage 2: Nsight Compute (detailed metrics, kernel replay)"
NCU_OUTPUT="${OUTPUT_DIR}/profile_ncu.csv"
ncu --set full \
    --csv \
    --output "$NCU_OUTPUT" \
    python3 "$WORKLOAD" "$WORKLOAD_ID" || true

echo "✓ NCU output: ${NCU_OUTPUT}"
echo ""

echo "Profiling complete. Outputs:"
echo "  - ${OUTPUT_DIR}/*_kernels.csv (nsys)"
echo "  - ${NCU_OUTPUT} (ncu metrics)"
echo ""
echo "Next: python3 sim_prof_parse.py classify \\"
echo "  --nsys-csv ${OUTPUT_DIR}/*_kernels.csv \\"
echo "  --ncu-csv ${NCU_OUTPUT} \\"
echo "  --gpu h100-sxm"
