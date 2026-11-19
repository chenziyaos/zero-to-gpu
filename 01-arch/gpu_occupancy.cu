// 用 CUDA 运行时 API 算 occupancy，和 sim_occupancy.py 的结果对照。
// cudaOccupancyMaxActiveBlocksPerMultiprocessor 读的是编译后 kernel 的真实寄存器数和共享内存，
// 所以先用 -Xptxas -v 看清每个 kernel 用了多少寄存器，再把同样的参数喂给 sim_occupancy.py。
//
// 编译：nvcc -O3 -arch=sm_90 -Xptxas -v -o occ gpu_occupancy.cu
// 运行：./occ
// 期望：下面三个 kernel 的寄存器数由编译器决定，每一行打印出的块数应当和
//       python3 sim_occupancy.py --cc 9.0 --threads T --regs R --smem S 给出的一致。
#include <cstdio>
#include <cuda_runtime.h>

__global__ void light(float* x) { x[blockIdx.x * blockDim.x + threadIdx.x] *= 2.f; }

// 用 launch_bounds 把寄存器压到每线程 64 个以内
__global__ void __launch_bounds__(256, 4) heavy(float* x) {
  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = x[threadIdx.x + i * 256];
#pragma unroll
  for (int k = 0; k < 8; ++k)
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] = acc[i] * acc[(i + k) & 31] + 1.f;
  float s = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) s += acc[i];
  x[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern __shared__ float dyn[];
__global__ void smem_user(float* x) {
  dyn[threadIdx.x] = x[threadIdx.x];
  __syncthreads();
  x[threadIdx.x] = dyn[blockDim.x - 1 - threadIdx.x];
}

template <class K>
void report(const char* name, K k, int threads, size_t smem) {
  cudaFuncAttributes a; cudaFuncGetAttributes(&a, k);
  if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int blocks = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, threads, smem);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int warps = blocks * ((threads + 31) / 32), maxw = p.maxThreadsPerMultiProcessor / 32;
  printf("%-10s threads %4d regs %3d smem %6zu B -> %2d blocks, %2d/%d warps, occupancy %.1f%%\n",
         name, threads, a.numRegs, smem + a.sharedSizeBytes, blocks, warps, maxw, 100.0 * warps / maxw);
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("# %s CC %d.%d, %d SM, regs/SM %d, smem/SM %zu KB, smem/block optin %zu KB\n", p.name, p.major, p.minor,
         p.multiProcessorCount, p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlockOptin / 1024);
  report("light", light, 256, 0);
  report("heavy", heavy, 256, 0);
  report("smem 48K", smem_user, 128, 48 * 1024);
  report("smem 100K", smem_user, 128, 100 * 1024);
  return 0;
}
