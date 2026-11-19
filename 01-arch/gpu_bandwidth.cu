// 显存带宽微基准：grid-stride 循环做 copy（读 1 写 1）和 read-only 求和，
// 对比标量 float 和向量化 float4 两种加载宽度，以及不同的线程块数。
// 报告的带宽按实际搬运字节算：copy 是 2 × N × 4 字节，read 是 N × 4 字节。
//
// 编译：nvcc -O3 -arch=sm_90 -o bw gpu_bandwidth.cu
// 运行：./bw            # 默认 1 GiB 数组
//
// 期望：大数组的 copy 和 read 能到标称带宽的九成上下。Luo et al. (arXiv 2402.13499) 实测
// A100 PCIe 1407 GB/s（约 90%）、H800 PCIe 1862 GB/s（约 91%）。块数太少（例如每 SM 1 块 128 线程）
// 时在途请求不够，带宽明显掉下来，这正是 sim_inflight.py 算的那本账。
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void copy1(const float* __restrict__ a, float* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void copy4(const float4* __restrict__ a, float4* __restrict__ b, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void read4(const float4* __restrict__ a, float* out, size_t n4) {
  float s = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i]; s += v.x + v.y + v.z + v.w;
  }
  if (s == 12345.f) *out = s;   // 几乎不会成立，只为留住读操作
}

template <class F>
float timeit(F f, int reps = 20) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  f(); CK(cudaDeviceSynchronize());
  cudaEventRecord(s);
  for (int i = 0; i < reps; ++i) f();
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  return ms / reps;
}

int main(int argc, char** argv) {
  size_t mb = argc > 1 ? atol(argv[1]) : 1024;
  size_t n = mb * 1024 * 1024 / sizeof(float), n4 = n / 4;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int sms = p.multiProcessorCount;
  printf("# %s, %d SM, array %zu MB\n", p.name, sms, mb);
  float *a, *b, *o;
  CK(cudaMalloc(&a, n * 4)); CK(cudaMalloc(&b, n * 4)); CK(cudaMalloc(&o, 4));
  CK(cudaMemset(a, 0, n * 4));
  double gb = n * 4.0 / 1e9;
  printf("# kernel   blocks/SM threads  GB/s\n");
  int cfg[][2] = {{1, 128}, {1, 1024}, {4, 256}, {8, 256}, {16, 128}};
  for (auto& c : cfg) {
    int blocks = sms * c[0], th = c[1];
    float t1 = timeit([&] { copy1<<<blocks, th>>>(a, b, n); });
    float t4 = timeit([&] { copy4<<<blocks, th>>>((float4*)a, (float4*)b, n4); });
    float tr = timeit([&] { read4<<<blocks, th>>>((float4*)a, o, n4); });
    printf("copy1      %3d      %4d  %7.0f\n", c[0], th, 2 * gb / (t1 / 1e3));
    printf("copy4      %3d      %4d  %7.0f\n", c[0], th, 2 * gb / (t4 / 1e3));
    printf("read4      %3d      %4d  %7.0f\n", c[0], th, gb / (tr / 1e3));
  }
  return 0;
}
