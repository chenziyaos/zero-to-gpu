#!/usr/bin/env bash
set -Eeuo pipefail

# Inspect Kubernetes GPU resources and topology
# Read-only operations using kubectl
# Usage: ./inspect_k8s_gpu.sh [node-name]
#
# This script checks:
# - GPU node labels and resources
# - Device plugin status
# - GPU Operator components
# - Available GPUs per node
#
# Dependencies: kubectl configured to access cluster

NODE="${1:-}"

if [[ -z "$NODE" ]]; then
    echo "Usage: $0 <node-name>"
    echo ""
    echo "List all nodes with GPUs:"
    kubectl get nodes -L nvidia.com/gpu-count 2>/dev/null || echo "  (No GPU nodes found)"
    exit 1
fi

echo "==================================================================="
echo "Node: $NODE"
echo "==================================================================="

echo ""
echo "--- Node Info ---"
kubectl describe node "$NODE" | grep -A 50 "Allocatable:" | head -20

echo ""
echo "--- GPU Labels ---"
kubectl get node "$NODE" -o jsonpath='{.metadata.labels}' | grep -i gpu || echo "  (No GPU-related labels)"

echo ""
echo "--- Capacity & Allocatable ---"
kubectl get node "$NODE" -o jsonpath='{.status.capacity}' | grep -i gpu || echo "  (No GPU capacity info)"

echo ""
echo "--- NVIDIA GPU Operator Pods (gpu-operator namespace) ---"
kubectl get pods -n gpu-operator -o wide 2>/dev/null | head -20 || echo "  (GPU Operator not installed)"

echo ""
echo "--- Device Plugin Status ---"
kubectl describe node "$NODE" 2>/dev/null | grep -A 5 "nvidia.com" || echo "  (No nvidia.com resources)"

echo ""
echo "--- Pods using GPUs on this node ---"
kubectl get pods --all-namespaces --field-selector spec.nodeName="$NODE" \
    -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.nvidia\.com/gpu}{"\n"}{end}' \
    2>/dev/null | grep gpu || echo "  (No GPU pods on this node)"

echo ""
echo "==================================================================="
