/**
 * GEMM 朴素实现（SGEMM, float32, C = A*B）。
 *
 * 对应 GPU-05 第 2 章：最基础的双层循环实现。
 *
 * 依赖：CUDA 13.4，不需要 cuBLAS（纯 kernel 实现）。
 * 版本依赖：NVIDIA RTX A6000 或 H100（compute capability >= 7.0）。
 *
 * 注：未在本仓库真卡复测，期望值来自文献 [26] (siboehm.com)。
 * 本代码保守可读，仅用于演示朴素写法及后续优化的起点。
 */

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * 最朴素的 SGEMM kernel：每个线程计算一个 C[i,j] 元素。
 *
 * Grid:  ceil(N/32) × ceil(M/32)，block 32×32 = 1024 threads
 * Block: threadIdx.x = j (0..31), threadIdx.y = i (0..31)
 *
 * 访存特性（性能灾难）：
 *   - 全局内存读：M*N*K 次读 A，M*N*K 次读 B（重复加载）
 *   - 全局内存写：M*N 次写 C
 *   - 总字节数：约 2*M*N*K * sizeof(float) ≈ 549 GB（对 4096³）
 *   - 实现的 FLOP/字节：0.25（最坏情况带宽受限）
 *
 * 理论性能下限：
 *   GPU-A6000 (11 TFLOPS, 0.576 TB/s) → 9 GB/s 带宽上限 / 2M*N*K / 4 = 309 GFLOP/s
 */
__global__ void sgemm_naive(float *A, float *B, float *C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= N) return;

    float sum = 0.0f;

    // K 次迭代，每次读一个 A[i,p] 和 B[p,j]
    for (int p = 0; p < K; ++p) {
        sum += A[i * K + p] * B[p * N + j];
    }

    C[i * N + j] = sum;
}


/**
 * Host wrapper: 分配内存、launch kernel、同步。
 */
void sgemm_naive_host(int M, int N, int K) {
    // 输入矩阵 A[M×K], B[K×N]，输出 C[M×N]
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_a);
    cudaMalloc(&d_B, size_b);
    cudaMalloc(&d_C, size_c);

    // TODO: 数据拷贝（H2D），kernel 调用，D2H 拷贝，计时

    // Launch kernel
    dim3 block(32, 32);  // 1024 threads
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    // 假设 A, B 已由 H2D 拷贝到 GPU
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // TODO: cudaDeviceSynchronize() 并测时

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
