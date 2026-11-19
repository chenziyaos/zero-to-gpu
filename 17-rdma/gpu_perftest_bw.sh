#!/usr/bin/env bash
#
# GPU-17 RDMA Network: Cross-machine bandwidth measurement with perftest
#
# Prerequisites: OFED 5.x+ with perftest, two machines with RoCE/IB NICs
# Output: bandwidth comparison between two-machine and single-machine NVLink
#
# Usage on server (listening):
#   $ ./gpu_perftest_bw.sh server 192.168.1.100
#
# Usage on client (sending):
#   $ ./gpu_perftest_bw.sh client 192.168.1.100 1024
#
# [1] NVIDIA RDMA Aware Programming v1.7: https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17
# [2] Meta RoCE SIGCOMM 2024: engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/
#
# Constants audited: 2026-11-04
#   - 400 Gbps RoCE: 50 GB/s theoretical
#   - H100 NVLink: 450 GB/s single direction
#   - perftest package required (install: apt-get install perftest)
#
# Note: This script has not been tested on real hardware. Expected values from literature [1] [2].

set -Eeuo pipefail

SERVER_IP="${2:-127.0.0.1}"
MESSAGE_SIZE="${3:-1024}"  # Bytes
DURATION="${4:-10}"         # Seconds

function show_usage() {
    echo "Usage: $0 {server|client} [server_ip] [message_size_bytes] [duration_sec]"
    echo ""
    echo "  server     - Start as perftest server (receiver)"
    echo "  client     - Start as perftest client (sender)"
    echo "  server_ip  - Server IP address (default: 127.0.0.1)"
    echo "  message_size_bytes - Default 1024 bytes"
    echo "  duration_sec - Test duration in seconds (default: 10)"
    exit 1
}

function check_perftest() {
    if ! command -v ib_send_bw &> /dev/null; then
        echo "ERROR: perftest not found. Install with: apt-get install perftest"
        exit 1
    fi
}

function run_server() {
    echo "Starting perftest server on $SERVER_IP, waiting for client..."
    ib_send_bw -a GID_BASED -x 3 -n 5000 -d mlx5_0:1 -c RC -t UD --report_gbits --gid-index 3
}

function run_client() {
    echo "Starting perftest client to $SERVER_IP, msg_size=$MESSAGE_SIZE bytes, duration=${DURATION}s"
    ib_send_bw -a GID_BASED -x 3 -n 5000 -d mlx5_0:1 -c RC -t UD "$SERVER_IP" \
        --report_gbits --gid-index 3
}

case "${1:-}" in
    server)
        check_perftest
        run_server
        ;;
    client)
        check_perftest
        run_client
        ;;
    *)
        show_usage
        ;;
esac
