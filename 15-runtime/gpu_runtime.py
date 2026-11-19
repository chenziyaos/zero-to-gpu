"""GPU-based measurements for CUDA launch, graph, and multi-stream.
Corresponding to GPU-15, section 8: real machine experiments.

This script requires a GPU with PyTorch installed.
Do NOT run on CPU: will exit gracefully with no failures.

Measurements:
  - launch:  Eager vs CUDA Graph vs torch.compile for small kernel sequences
  - streams: Serial vs three-stream pipeline H2D-Compute-D2H overlap
  - sync:    Impact of per-step synchronization (e.g., .item() polling)

Assumptions:
  GPU: H100 SXM or compatible (adjust --device if needed)
  PyTorch: 2.15.0+ with torch.compile and CUDA Graph support
  CUDA: 13.4+

Baseline expectations (H100 SXM):
  [1] CUDA Graph 2000×1us small kernels @ 100 replays: ~2-5x speedup vs eager
  [2] Three-stream pipeline (balanced): ~2-3x speedup vs serial
  [3] Per-step .item() synchronization: 1-2 orders of magnitude slowdown

Note: "未在本仓库真卡复测" — Expected values derived from public docs.
      Actual numbers vary by GPU generation, CUDA version, and thermal state.
"""
import sys
import os


def try_import_torch():
    """Try to import torch; exit gracefully if not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("Info: GPU not available; skipping measurements")
            sys.exit(0)
        return torch
    except ImportError:
        print("Info: PyTorch not installed; skipping GPU measurements")
        sys.exit(0)


def launch_experiment(torch, num_kernels=2000, kernel_us=1, num_runs=100):
    """Measure eager vs CUDA Graph for small kernel sequence."""
    if not torch.cuda.is_available():
        return None

    device = torch.device('cuda:0')
    tensor_size = 4096
    x = torch.ones(tensor_size, device=device, dtype=torch.bfloat16)
    s = torch.tensor(0.5, device=device, dtype=torch.bfloat16)

    def kernel_chain():
        """Simple kernel sequence: mul, add, mul, ..."""
        y = x.clone()
        for _ in range(num_kernels):
            y = y.mul_(s).add_(0.001)
        return y

    # Warmup
    for _ in range(5):
        kernel_chain()
    torch.cuda.synchronize()

    # Eager timing
    times_eager = []
    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        kernel_chain()
        t_end.record()
        torch.cuda.synchronize()
        elapsed = t_start.elapsed_time(t_end)
        times_eager.append(elapsed)

    eager_ms = sorted(times_eager)[len(times_eager) // 2]  # median

    # CUDA Graph timing
    try:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()

        # Warmup on stream
        with torch.cuda.stream(stream):
            for _ in range(3):
                kernel_chain()
        torch.cuda.current_stream().wait_stream(stream)

        # Record
        with torch.cuda.graph(graph):
            kernel_chain()

        times_graph = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            t_start.record()
            graph.replay()
            t_end.record()
            torch.cuda.synchronize()
            elapsed = t_start.elapsed_time(t_end)
            times_graph.append(elapsed)

        graph_ms = sorted(times_graph)[len(times_graph) // 2]
        graph_speedup = eager_ms / graph_ms if graph_ms > 0 else 1.0
    except Exception as e:
        print(f"CUDA Graph measurement failed: {e}")
        graph_ms = None
        graph_speedup = None

    # torch.compile timing (if available)
    try:
        compiled_chain = torch.compile(kernel_chain, backend='eager')  # CPU-side dispatch only
        times_compile = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            t_start.record()
            compiled_chain()
            t_end.record()
            torch.cuda.synchronize()
            elapsed = t_start.elapsed_time(t_end)
            times_compile.append(elapsed)
        compile_ms = sorted(times_compile)[len(times_compile) // 2]
        compile_speedup = eager_ms / compile_ms if compile_ms > 0 else 1.0
    except Exception as e:
        print(f"torch.compile measurement skipped: {e}")
        compile_ms = None
        compile_speedup = None

    return {
        'eager_ms': eager_ms,
        'graph_ms': graph_ms,
        'graph_speedup': graph_speedup,
        'compile_ms': compile_ms,
        'compile_speedup': compile_speedup,
    }


def sync_experiment(torch, num_kernels=2000, num_runs=20):
    """Measure impact of per-step synchronization (e.g., .item())."""
    if not torch.cuda.is_available():
        return None

    device = torch.device('cuda:0')
    x = torch.ones(num_kernels, device=device, dtype=torch.bfloat16)

    # Baseline: no per-step sync
    times_nosync = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        y = x.clone()
        for i in range(num_kernels):
            y = y.mul_(0.5).add_(0.001)
        t_end.record()
        torch.cuda.synchronize()
        elapsed = t_start.elapsed_time(t_end)
        times_nosync.append(elapsed)

    nosync_ms = sorted(times_nosync)[len(times_nosync) // 2]

    # With per-step .item() (forces sync)
    times_sync = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        y = x.clone()
        for i in range(num_kernels):
            y = y.mul_(0.5).add_(0.001)
            _ = y[0].item()  # Forces synchronization
        t_end.record()
        torch.cuda.synchronize()
        elapsed = t_start.elapsed_time(t_end)
        times_sync.append(elapsed)

    sync_ms = sorted(times_sync)[len(times_sync) // 2]
    sync_slowdown = sync_ms / nosync_ms if nosync_ms > 0 else 1.0

    return {
        'nosync_ms': nosync_ms,
        'sync_ms': sync_ms,
        'sync_slowdown': sync_slowdown,
    }


def streams_experiment(torch, h2d_size=128*1024*1024, kernel_us=100, num_batches=10):
    """Measure multi-stream H2D-Compute-D2H overlap."""
    if not torch.cuda.is_available():
        return None

    device = torch.device('cuda:0')
    host_tensor = torch.ones(h2d_size // 4, dtype=torch.float32)  # 4 bytes per float32
    device_tensor = torch.zeros(h2d_size // 4, device=device, dtype=torch.float32)

    # Serial: H2D → compute → D2H, repeated
    times_serial = []
    for _ in range(3):
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(num_batches):
            device_tensor.copy_(host_tensor, non_blocking=False)
            torch.cuda.current_stream().synchronize()
            device_tensor.mul_(0.5)
            torch.cuda.current_stream().synchronize()
            host_tensor_result = device_tensor.cpu()
        t_end.record()
        torch.cuda.synchronize()
        elapsed = t_start.elapsed_time(t_end)
        times_serial.append(elapsed)

    serial_ms = sorted(times_serial)[len(times_serial) // 2]

    # Parallel: three streams H2D, compute, D2H with overlapping
    try:
        stream_h2d = torch.cuda.Stream()
        stream_compute = torch.cuda.Stream()
        stream_d2h = torch.cuda.Stream()

        times_parallel = []
        for _ in range(3):
            torch.cuda.synchronize()
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            t_start.record()

            for b in range(num_batches + 2):
                if b < num_batches:
                    with torch.cuda.stream(stream_h2d):
                        device_tensor.copy_(host_tensor, non_blocking=True)
                if 0 < b < num_batches + 1:
                    with torch.cuda.stream(stream_compute):
                        stream_compute.wait_stream(stream_h2d)
                        device_tensor.mul_(0.5)
                        stream_h2d.wait_stream(stream_compute)
                if b > 0 and b <= num_batches:
                    with torch.cuda.stream(stream_d2h):
                        stream_d2h.wait_stream(stream_compute)
                        host_result = device_tensor.cpu()

            torch.cuda.synchronize()
            t_end.record()
            torch.cuda.synchronize()
            elapsed = t_start.elapsed_time(t_end)
            times_parallel.append(elapsed)

        parallel_ms = sorted(times_parallel)[len(times_parallel) // 2]
        pipeline_speedup = serial_ms / parallel_ms if parallel_ms > 0 else 1.0
    except Exception as e:
        print(f"Multi-stream experiment failed: {e}")
        parallel_ms = None
        pipeline_speedup = None

    return {
        'serial_ms': serial_ms,
        'parallel_ms': parallel_ms,
        'pipeline_speedup': pipeline_speedup,
    }


def main():
    torch = try_import_torch()
    if torch is None:
        return

    print("\n" + "="*70)
    print("GPU Runtime Measurements (H100 expected baseline)")
    print("="*70)

    if not torch.cuda.is_available():
        print("No CUDA GPU detected; skipping measurements")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Runtime: {torch.version.cuda}")

    print("\n--- Launch Experiment ---")
    try:
        launch_res = launch_experiment(torch, num_kernels=2000, kernel_us=1, num_runs=100)
        if launch_res:
            print(f"Eager (2000×1us kernels, 100 runs, median):")
            print(f"  Wall clock: {launch_res['eager_ms']:.2f} ms")
            if launch_res['graph_ms']:
                print(f"CUDA Graph (100 replays):")
                print(f"  Wall clock: {launch_res['graph_ms']:.2f} ms")
                print(f"  Speedup vs eager: {launch_res['graph_speedup']:.2f}x")
            if launch_res['compile_ms']:
                print(f"torch.compile (100 runs):")
                print(f"  Wall clock: {launch_res['compile_ms']:.2f} ms")
                print(f"  Speedup vs eager: {launch_res['compile_speedup']:.2f}x")
    except Exception as e:
        print(f"Launch experiment error: {e}")

    print("\n--- Synchronization Experiment ---")
    try:
        sync_res = sync_experiment(torch, num_kernels=2000, num_runs=20)
        if sync_res:
            print(f"2000 kernels, 20 runs (median):")
            print(f"  No per-step sync: {sync_res['nosync_ms']:.2f} ms")
            print(f"  Per-step .item(): {sync_res['sync_ms']:.2f} ms")
            print(f"  Slowdown: {sync_res['sync_slowdown']:.2f}x")
    except Exception as e:
        print(f"Sync experiment error: {e}")

    print("\n--- Multi-Stream Experiment ---")
    try:
        streams_res = streams_experiment(torch, h2d_size=128*1024*1024, kernel_us=100, num_batches=10)
        if streams_res:
            print(f"H2D-Compute-D2H pipeline (128 MiB, 10 batches, median of 3):")
            print(f"  Serial: {streams_res['serial_ms']:.2f} ms")
            if streams_res['parallel_ms']:
                print(f"  Pipelined: {streams_res['parallel_ms']:.2f} ms")
                print(f"  Speedup: {streams_res['pipeline_speedup']:.2f}x")
    except Exception as e:
        print(f"Streams experiment error: {e}")

    print("\n" + "="*70)
    print("Note: Expected baseline values (H100 SXM, PyTorch 2.15+, CUDA 13.4+):")
    print("  Launch: Graph 2-5x, compile depends on fusion potential")
    print("  Sync: Per-step .item() typically 10-100x slowdown")
    print("  Streams: Balanced pipeline 2-3x speedup")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
