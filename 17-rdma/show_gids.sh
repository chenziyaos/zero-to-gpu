#!/usr/bin/env bash
#
# GPU-17 RDMA Network: Show Global Identifiers (GIDs) for RDMA fabric
#
# Display RDMA port configuration, link state, and available GIDs for RoCE.
# Used for verifying fabric connectivity and configuration before all-reduce tests.
#
# Reference:
#   [1] NVIDIA RDMA Aware Programming v1.7: https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17
#   [2] show_gids from ibutils package
#
# Usage:
#   $ ./show_gids.sh
#   $ ./show_gids.sh mlx5_0  (specific device)
#
# Output example (H100 + ConnectX-7 with RoCEv2):
#   Device: mlx5_0
#   Port:   1
#   Link State: ACTIVE
#   MTU: 4096
#   GID 0: fe80::0000:0000:0000:0001 (Link-Local)
#   GID 1: 0000:0000:0000:0000:0000:ffff:192.168.1.100 (RoCEv2)
#   GID 2: ...
#
# Note: This script is a wrapper around system RDMA tools and requires:
#   - OFED 5.x+ installation
#   - ibutils package (ibstat, show_gids commands)

set -Eeuo pipefail

DEVICE="${1:-}"

if [[ -z "$DEVICE" ]]; then
    echo "=== RDMA Devices ==="
    ibstat -l | while read dev; do
        echo ""
        echo "Device: $dev"
        ibstat "$dev" | grep -E "State:|Link|MTU|Rate"
    done
else
    echo "=== RDMA Device: $DEVICE ==="
    ibstat "$DEVICE"
fi

echo ""
echo "=== GID Mappings (for RoCE) ==="
if command -v show_gids &> /dev/null; then
    show_gids
else
    echo "NOTE: show_gids not found. Install OFED ibutils package."
    echo "Fallback: Display GID by device + port using sysfs."
    for device in /sys/class/infiniband/mlx5_*/ports/*/gid_entries/gid*; do
        if [[ -f "$device" ]]; then
            dev_name=$(basename $(dirname $(dirname $(dirname "$device"))))
            port=$(basename $(dirname $(dirname "$device")))
            gid_idx=$(basename "$device" | sed 's/gid//')
            gid=$(cat "$device")
            echo "$dev_name:$port / GID $gid_idx: $gid"
        fi
    done
fi
