// P-chase 延迟微基准：一个线程沿着随机环形链表一路读下去，每次读的地址依赖上一次的结果，
// 所以读和读之间不能重叠，平均每跳的周期数就是这一级存储的访问延迟。
// 工作集从 8 KB 扫到 512 MB，曲线上的台阶对应 L1、L2、HBM。
// 方法来自 Mei & Chu (arXiv 1509.02308)、Jia et al. (arXiv 1804.06826)，Luo et al. (arXiv 2402.13499) 用同样方法测 Hopper。
//
// 编译：nvcc -O3 -arch=sm_90 -o pchase gpu_pchase.cu     （A100 用 sm_80，B200 用 sm_100）
// 运行：./pchase            # 默认 stride 128 B，一个缓存行一跳
//       ./pchase 32         # 改 stride，单位字节
//
// 期望（周期，H800 PCIe，Luo et al. 表格）：L1 约 41，L2 约 263，全局约 479。
// A100：L1 约 38，L2 约 262，全局约 466。工作集跨过 L2 容量后才会稳定到全局延迟；
// H100 的 L2 分两个分区，访问远端分区的延迟明显更高，曲线会在 25 MB 附近多一个小台阶。
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void chase(const unsigned* __restrict__ next, int iters, unsigned* out, long long* cycles) {
  unsigned j = 0;
  // 先走一遍预热，把能缓存的都放进缓存
  for (int i = 0; i < iters; ++i) j = next[j];
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) j = next[j];
  long long t1 = clock64();
  *out = j;            // 防止编译器把循环优化掉
  *cycles = t1 - t0;
}

int main(int argc, char** argv) {
  int stride = argc > 1 ? atoi(argv[1]) : 128;
  int sstep = stride / sizeof(unsigned);
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("# %s, L2 %d KB, stride %d B\n# size_KB  cycles_per_load\n", p.name, p.l2CacheSize / 1024, stride);
  std::mt19937 rng(42);
  for (size_t kb = 8; kb <= 512 * 1024; kb *= 2) {
    for (int sub = 0; sub < 2; ++sub) {               // 每个 2 的幂之间再插一个 1.5 倍的点
      size_t bytes = kb * 1024 * (sub ? 3 : 2) / 2;
      size_t n = bytes / sizeof(unsigned), nodes = n / sstep;
      if (nodes < 2) continue;
      // 节点按 stride 排开，再随机打乱访问顺序，躲开硬件预取
      std::vector<unsigned> order(nodes), h(n, 0);
      for (size_t i = 0; i < nodes; ++i) order[i] = i * sstep;
      std::shuffle(order.begin() + 1, order.end(), rng);
      for (size_t i = 0; i < nodes; ++i) h[order[i]] = order[(i + 1) % nodes];
      unsigned *d, *o; long long* c;
      CK(cudaMalloc(&d, n * sizeof(unsigned))); CK(cudaMalloc(&o, 4)); CK(cudaMalloc(&c, 8));
      CK(cudaMemcpy(d, h.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
      int iters = (int)std::min<size_t>(std::max<size_t>(nodes, 4096), 1 << 20);
      chase<<<1, 1>>>(d, iters, o, c);
      CK(cudaGetLastError());
      long long cyc; CK(cudaMemcpy(&cyc, c, 8, cudaMemcpyDeviceToHost));
      printf("%9zu  %8.1f\n", bytes / 1024, (double)cyc / iters);
      CK(cudaFree(d)); CK(cudaFree(o)); CK(cudaFree(c));
    }
  }
  return 0;
}
