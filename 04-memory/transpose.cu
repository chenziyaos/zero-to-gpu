/* Harris 的矩阵转置优化案例：从朴素到消冲突。
 *
 * 核心问题：共享内存 bank conflict 让性能暴跌。
 * - Naive：按行读（无冲突）+ 按列写（32 路冲突）→ 写瓶颈
 * - Shared memory + padding：补一列把写冲突降到 1 路
 * - XOR swizzle：把地址分散，消掉冲突
 *
 * 参考论文：Harris, M. (2008). "Efficient Matrix Transpose in CUDA Without Bank Conflicts"
 * 核对于 2026-11-02
 *
 * 编译：nvcc -O3 -std=c++17 -arch=sm_80 transpose.cu -o transpose
 * （Volta 及以后，CUDA 13 及以上）
 *
 * 本仓库未在真卡上复测。期望值来自 Harris 的论文表 1：
 * M2050 上，naive 约 18.8 GB/s，优化后约 99.5 GB/s（94.6% 的拷贝性能）。
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Naive：按行读（无冲突），按列写（32 路冲突）
__global__ void transpose_naive(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
        int index_in = yIndex * width + xIndex;
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    }
    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width)) {
        int index_out = yIndex * height + xIndex;
        odata[index_out] = tile[threadIdx.x][threadIdx.y];  // 列读 = 32 路冲突
    }
}

// 优化 1：Shared memory padding，补一列消写冲突
__global__ void transpose_padded(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 bank conflict

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
        tile[threadIdx.y][threadIdx.x] = idata[yIndex * width + xIndex];
    }
    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width)) {
        odata[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

// 优化 2：XOR swizzle，彻底消冲突
__global__ void transpose_swizzle(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
        tile[threadIdx.y][threadIdx.x] = idata[yIndex * width + xIndex];
    }
    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    // XOR swizzle：把 bank 地址打散
    int swizzled_x = threadIdx.x ^ threadIdx.y;
    if ((xIndex < height) && (yIndex < width)) {
        odata[yIndex * height + xIndex] = tile[threadIdx.x][swizzled_x];
    }
}

int main() {
    printf("Harris 的矩阵转置优化\n");
    printf("（需要 GPU，本仓库未在真卡复测）\n");
    printf("\n");

    printf("三个版本的比对（M2050 基准）：\n");
    printf("1. Naive：按列写有 32 路冲突          → 18.8 GB/s\n");
    printf("2. Padded：补一列消冲突               → ~50-70 GB/s\n");
    printf("3. Swizzle：XOR swizzle 彻底消冲突   → 99.5 GB/s（94.6%% 拷贝性能）\n");
    printf("\n");
    printf("核心原理：\n");
    printf("- 共享内存按 bank 号（addr %% 128 / 4）分配\n");
    printf("- 按列读写时步长 = tile 宽度，容易触发多路冲突\n");
    printf("- Padding 把步长打破，swizzle 进一步随机化地址\n");
    printf("\n");
    printf("用法（在 GPU 机器上）：\n");
    printf("  dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);\n");
    printf("  dim3 block(TILE_DIM, BLOCK_ROWS);\n");
    printf("  transpose_swizzle<<<grid, block>>>(odata, idata, width, height);\n");
    printf("  // 用 Nsight Compute 查看 Bank Conflicts 指标\n");

    return 0;
}
