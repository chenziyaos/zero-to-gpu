#!/usr/bin/env bash
set -Eeuo pipefail

##
# GEMM benchmark 脚本框架。
#
# 用法：
#   ./benchmark.sh naive      # 测试朴素 GEMM
#   ./benchmark.sh blocked    # 测试分块 GEMM
#   ./benchmark.sh all        # 运行全部 CPU 模拟
#
# 环境：Python 3.10+
# GPU 侧脚本（.cu 编译）未在本仓库真卡复测。
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-all}" in
    naive)
        echo "=== 朴素 GEMM 基准（纯访存灾难） ==="
        python3 sim_gemm.py naive --m 4096 --n 4096 --k 4096
        ;;

    blocked)
        echo "=== 分块 GEMM 优化（访存下降 1024 倍） ==="
        python3 sim_gemm.py blocked --m 4096 --n 4096 --k 4096 --bm 128 --bn 128 --bk 8
        ;;

    thread-tile)
        echo "=== 线程分块数据复用分析 ==="
        python3 sim_gemm.py thread-tile --m 4096 --n 4096 --k 4096
        ;;

    intensity)
        echo "=== 算术强度与瓶颈（H100 拐点 295） ==="
        python3 sim_gemm.py intensity --m 4096 --n 4096 --k 4096 --gpu h100-sxm
        ;;

    occupancy)
        echo "=== 占用率分析 ==="
        python3 sim_occupancy.py analyze
        ;;

    sweep)
        echo "=== 参数空间扫描找最优配置 ==="
        python3 sim_occupancy.py sweep
        ;;

    shmem)
        echo "=== 共享内存约束分析 ==="
        python3 sim_occupancy.py shared-memory
        ;;

    test)
        echo "=== 运行完整测试套件（12 项） ==="
        python3 test_gemm.py
        ;;

    all)
        echo "=== 运行全部 CPU 模拟与测试 ==="
        "$0" naive
        echo
        "$0" blocked
        echo
        "$0" thread-tile | head -20
        echo
        "$0" intensity
        echo
        "$0" occupancy | head -20
        echo
        "$0" test
        ;;

    *)
        cat <<EOF
用法：./benchmark.sh [命令]

命令：
  naive        朴素 GEMM（访存灾难基准）
  blocked      分块优化（访存降 1024 倍）
  thread-tile  线程分块数据复用分析
  intensity    算术强度与瓶颈分析
  occupancy    占用率分析
  sweep        参数扫描找最优
  shmem        共享内存约束
  test         完整测试套件
  all          运行全部（默认）

所有命令运行 CPU 侧模拟器（无需 GPU）。
GPU 侧脚本（.cu）示例参见 sgemm_naive.cu（未在真卡复测）。
EOF
        exit 1
        ;;
esac
