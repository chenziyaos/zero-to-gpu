#!/usr/bin/env bash
set -Eeuo pipefail

# Nsight Compute profiling 脚本：分析访存、bank 冲突、sector 利用率。
#
# 依赖：Nsight Compute 2026.3.1 及以上，CUDA 13.4 及以上，GPU 计算能力 7.0+（Volta）
#
# 本仓库未在真卡上复测。期望值来自：
# [1] NVIDIA Nsight Compute 文档：Memory 指标集
# [2] Harris 的论文与 NVIDIA 官方教程
#
# 用法：
#   ./ncu_mem.sh <内核二进制>
#   例：./ncu_mem.sh ./transpose

if [[ $# -lt 1 ]]; then
    echo "用法：$0 <内核二进制> [参数...]"
    echo ""
    echo "示例："
    echo "  $0 ./transpose"
    echo "  $0 ./reduce"
    exit 1
fi

BINARY="$1"
shift

if [[ ! -f "$BINARY" ]]; then
    echo "错误：找不到二进制文件 $BINARY"
    exit 1
fi

# 检查 ncu 命令
if ! command -v ncu &> /dev/null; then
    echo "错误：Nsight Compute (ncu) 未找到"
    echo "请确保已安装 CUDA Toolkit 与 Nsight Compute"
    exit 1
fi

echo "=== GPU 内存分析 ==="
echo ""
echo "内核：$BINARY"
echo ""

# 运行 ncu，采集常见的内存指标
# metrics 选择：
#   smsp__inst_executed:sum                  执行的指令数
#   dram__bytes_read:sum                     DRAM 读字节数
#   dram__bytes_write:sum                    DRAM 写字节数
#   l1tex__m_read_sectors:sum                L1 读 sector 数
#   l1tex__m_read_requests:sum               L1 读请求数
#   l1tex__t_set_conflicts_sm:sum            共享内存冲突数

ncu --set memory_chart \
    --set memory_workload \
    --set sm__sass_thread_inst_executed \
    --metrics \
        dram__bytes_read,dram__bytes_write,\
        l1tex__m_read_sectors,l1tex__m_read_requests,\
        lts__t_sector_pipe_lgop_mem_read,\
        l1tex__t_set_conflicts_sm \
    "$BINARY" "$@"

echo ""
echo "=== 指标说明 ==="
echo "- DRAM Reads/Writes：从 HBM 读写的总字节数"
echo "- L1 Sectors：L1 缓存的 sector 数（应尽量少）"
echo "- L1 Requests：L1 请求数（实际访问次数）"
echo "- Sector-to-Request Ratio：越接近 1 越好（无浪费）"
echo "- Bank Conflicts (l1tex__t_set_conflicts_sm)：共享内存冲突次数（应尽量少）"
echo ""
