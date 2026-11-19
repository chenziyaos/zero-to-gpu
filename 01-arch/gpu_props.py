"""打印本机 GPU 的架构参数，并按 common/specs.py 的口径算出峰值和拐点（GPU）。对应 GPU-01 第 1 章。

需要 PyTorch 和一张 NVIDIA GPU。字段名对应 cudaDeviceProp，PyTorch 只暴露其中一部分，
更全的字段用 CUDA samples 里的 deviceQuery，或者 nvidia-smi -q。

用法：python3 gpu_props.py
"""
import torch

if not torch.cuda.is_available():
    raise SystemExit('没有可用的 CUDA 设备；没有 GPU 时先跑 sim_occupancy.py 和 sim_inflight.py')

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'[{i}] {p.name}  CC {p.major}.{p.minor}')
    print(f'    SM 个数            {p.multi_processor_count}')
    print(f'    显存               {p.total_memory / 2**30:.1f} GiB')
    for k, label in (('max_threads_per_multi_processor', '每 SM 最多线程'),
                     ('regs_per_multiprocessor', '每 SM 寄存器'),
                     ('shared_memory_per_multiprocessor', '每 SM 共享内存（字节）'),
                     ('shared_memory_per_block_optin', '每块共享内存上限（字节）'),
                     ('L2_cache_size', 'L2（字节）'),
                     ('warp_size', 'warp 大小')):
        if hasattr(p, k):
            print(f'    {label:18} {getattr(p, k)}')
    print(f'    PyTorch 编译用 CUDA {torch.version.cuda}')
