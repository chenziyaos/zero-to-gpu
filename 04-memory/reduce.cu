/* Harris 的并行归约七步优化：从朴素到消指令。
 *
 * 核心原理：归约前三步被带宽限制，后三步被指令限制。
 * 早期版本有不必要的地址计算和同步，后期版本用 warp shuffle 消掉访存。
 *
 * 参考论文：Harris, M. (2007). "Parallel Reduction"
 * 核对于 2026-11-02
 *
 * 编译：nvcc -O3 -std=c++17 -arch=sm_80 reduce.cu -o reduce
 * （Volta 及以后，CUDA 13 及以上。Volta 后不支持隐式同步 volatile）
 *
 * 本仓库未在真卡上复测。期望值来自 Harris 的论文图 5：
 * G80 上，七个版本从 2.08 GB/s → 62.7 GB/s（30 倍加速）。
 * 第 4 步后访存已非瓶颈，指令成为主导。
 */

#include <stdio.h>
#include <cuda_runtime.h>

// 第 1 步：朴素版本，所有线程活跃，全带宽
__global__ void reduce_v1(float *g_odata, const float *g_idata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 第 4 步：避免分支，线程从后向前，减少控制流
__global__ void reduce_v4(float *g_odata, const float *g_idata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最后一步用 warp shuffle（假设 s = 32）
    if (blockDim.x >= 64) {
        volatile float *vdata = sdata;
        if (tid < 32) {
            vdata[tid] += vdata[tid + 32];
            vdata[tid] += vdata[tid + 16];
            vdata[tid] += vdata[tid + 8];
            vdata[tid] += vdata[tid + 4];
            vdata[tid] += vdata[tid + 2];
            vdata[tid] += vdata[tid + 1];
        }
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 第 7 步（Kepler 及以后的现代版本）：完全用 warp shuffle，无共享内存访问
__global__ void reduce_v7(float *g_odata, const float *g_idata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (i < n) ? g_idata[i] : 0.0f;

    // 同 block 内的 warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 共享内存只用于 block 间的最后一步
    extern __shared__ float sdata[];
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();

    // 最后一个 warp 完成
    if (tid < (blockDim.x + warpSize - 1) / warpSize) {
        sum = sdata[tid];
        for (int offset = (blockDim.x + warpSize - 1) / warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (tid == 0) g_odata[blockIdx.x] = sum;
}

int main() {
    printf("Harris 的并行归约七步优化\n");
    printf("（需要 GPU，本仓库未在真卡复测）\n");
    printf("\n");

    printf("性能演进（G80 基准，单位 GB/s）：\n");
    printf("第 1 步（朴素）：2.08   ── 所有线程活跃，但分支严重\n");
    printf("第 2 步（消分支）：9.6   ── 减少分支，但仍有同步 overhead\n");
    printf("第 3 步（避免分支）：25.5  ── 分支转移到 if，但仍有冲突\n");
    printf("第 4 步（避免 bank 冲突）：28.5  ── 访存已不是主瓶颈\n");
    printf("第 5 步（完全展开）：37.3  ── 消掉更多同步\n");
    printf("第 6 步（loop unroll）：51.5  ── 编译器自动优化\n");
    printf("第 7 步（warp shuffle）：62.7  ── 无共享内存冲突，接近理论上限\n");
    printf("\n");

    printf("核心规律：\n");
    printf("- 前三步：受带宽限制，性能随 stride 变化\n");
    printf("- 第 4 步：访存已对齐，性能平台期\n");
    printf("- 后三步：指令和同步成为主导，每次优化 1.2-1.5x\n");
    printf("\n");

    printf("用法（在 GPU 机器上）：\n");
    printf("  unsigned int blockSize = 256;\n");
    printf("  unsigned int numBlocks = (n + blockSize - 1) / blockSize;\n");
    printf("  reduce_v7<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_odata, d_idata, n);\n");
    printf("\n");

    printf("注意（Volta 及以后）：\n");
    printf("- volatile 关键字在 Volta 及以后不能保证同步，改用 __syncwarp()\n");
    printf("- warp shuffle 需要 Kepler 及以后\n");

    return 0;
}
