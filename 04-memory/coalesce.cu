/* 全局内存合并访存的示例 CUDA 内核。
 *
 * 核心原理：32 个线程的地址需要尽量落在少数几个 32 字节 sector 内。
 * - 连续读：步长 1，4 个 sector（128 字节数据）
 * - 步长 2：8 个 sector（128 字节数据分散到 8 个 sector）
 * - 步长 32：125 个 sector（线程地址跨度太大）
 *
 * 编译：nvcc -O3 -std=c++17 -arch=sm_80 coalesce.cu -o coalesce
 * （Volta 及以后，CUDA 13 及以上）
 *
 * 本仓库未在真卡上复测。期望值来自 NVIDIA CUDA 编程指南第 5.3.2 节。
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void read_coalesced(const float *d_in, float *d_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_out[tid] = d_in[tid];  // 连续读，4 个 sector
    }
}

__global__ void read_stride2(const float *d_in, float *d_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 2 < n) {
        d_out[tid] = d_in[tid * 2];  // 步长 2，8 个 sector
    }
}

__global__ void read_stride32(const float *d_in, float *d_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 32 < n) {
        d_out[tid] = d_in[tid * 32];  // 步长 32，125 个 sector
    }
}

int main() {
    printf("全局内存合并访存示例\n");
    printf("（需要 GPU，本仓库未在真卡复测）\n");
    printf("\n");

    // 注意：这段代码需要 GPU，本机无法运行
    printf("示例内核：\n");
    printf("1. read_coalesced：连续读，理想情况\n");
    printf("2. read_stride2：步长 2，sector 数翻倍\n");
    printf("3. read_stride32：步长 32，sector 数爆炸\n");
    printf("\n");
    printf("用法（在 GPU 机器上）：\n");
    printf("  float *d_in, *d_out;\n");
    printf("  cudaMalloc(&d_in, n * sizeof(float));\n");
    printf("  cudaMalloc(&d_out, n * sizeof(float));\n");
    printf("  read_coalesced<<<(n + 255) / 256, 256>>>(d_in, d_out, n);\n");
    printf("  // 用 Nsight Compute 分析 Sector Efficiency 指标\n");

    return 0;
}
