#!/bin/bash
# 用 ncu 给一个程序拍 Speed Of Light 和分层 Roofline（GPU）。对应 GPU-02 第 6 章。
# 用法：./ncu_roofline.sh python3 gpu_regimes.py fuse
# 输出 roofline.ncu-rep，用 ncu-ui 打开看 GPU Speed Of Light Roofline Chart；命令行摘要直接打印
set -euo pipefail
[ $# -ge 1 ] || { echo "用法：$0 <程序及参数>"; exit 1; }
ncu --set roofline \
    --section SpeedOfLight \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --launch-count 20 \
    -o roofline -f "$@"
ncu --import roofline.ncu-rep --page details --section SpeedOfLight --print-units base | \
    grep -E "Kernel Name|^ +[a-z_]+.*\(|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|Duration" || true
