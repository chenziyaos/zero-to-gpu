"""Environment check: PyTorch, CUDA versions and capabilities.
Corresponding to GPU-15, section 9: version baselines and feature detection.

This script is read-only: prints versions, environment, and capability flags.
No modifications to system state.

Baselines (GPU-15):
  PyTorch: 2.15.0
  CUDA Toolkit: 13.4 Update 2
  NCCL: 2.32.3
  Linux: 6.8

Version notes:
  [1] CUDA Graph: 11.x early → incomplete multi-stream/view support
      CUDA 12+ → condition nodes, loop nodes, queue updates
  [2] Inductor: fusion rules change every PyTorch minor version
      2.15 improved shape backfill and profiler attribution
  [3] caching allocator: expandable_segments default varies by platform/version
"""
import sys
import os


def check_pytorch():
    """Check PyTorch version and CUDA baseline."""
    try:
        import torch
        pt_version = torch.__version__
        print(f"PyTorch version: {pt_version}")

        # Try to get wheel CUDA baseline
        if hasattr(torch, 'cuda'):
            try:
                # Wheel baseline info (approximate)
                if "cu118" in pt_version or "cu121" in pt_version:
                    cuda_baseline = "CUDA 11.8/12.1 (wheel baseline)"
                elif "cu124" in pt_version:
                    cuda_baseline = "CUDA 12.4 (wheel baseline)"
                else:
                    cuda_baseline = "Unknown (check wheel tags)"
                print(f"Wheel CUDA baseline: {cuda_baseline}")
            except Exception as e:
                print(f"Could not determine wheel baseline: {e}")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False


def check_cuda():
    """Check CUDA runtime and toolkit versions."""
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA Runtime (via torch): {cuda_version}")
        else:
            print("CUDA not available")
            return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

    # Try to check CUDA toolkit via system command (best effort)
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"CUDA Toolkit (nvcc): {line.strip()}")
                    break
        else:
            print("nvcc not found or execution failed")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        print(f"Could not run nvcc: {e}")

    return True


def check_cudnn():
    """Check cuDNN version."""
    try:
        import torch
        if hasattr(torch.backends, 'cudnn'):
            cudnn_version = torch.backends.cudnn.version()
            print(f"cuDNN version: {cudnn_version}")
            return True
    except Exception as e:
        print(f"cuDNN check: {e}")
    return False


def check_nccl():
    """Check NCCL version."""
    try:
        # NCCL version check is not directly exposed in PyTorch
        # Try via environment or subprocess
        import subprocess
        result = subprocess.run(['python3', '-c', 'import torch; print(torch.cuda.nccl.version())'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"NCCL version: {result.stdout.strip()}")
            return True
    except Exception:
        pass

    # Fallback: check environment
    nccl_version = os.environ.get('NCCL_VERSION', 'Not set')
    print(f"NCCL_VERSION env: {nccl_version}")
    return False


def check_gpu_info():
    """Check available GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU detected")
            return False

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem_gb:.0f} GB HBM)")

        return True
    except Exception as e:
        print(f"GPU info check: {e}")
        return False


def check_capabilities():
    """Check torch.compile and CUDA Graph availability."""
    try:
        import torch
        print("\nCapability flags:")

        # torch.compile
        if hasattr(torch, 'compile'):
            print("  torch.compile: ✓ available")
        else:
            print("  torch.compile: ✗ not available")

        # CUDA Graph
        if hasattr(torch.cuda, 'CUDAGraph'):
            print("  CUDA Graph: ✓ available")
        else:
            print("  CUDA Graph: ✗ not available")

        # TF32
        if hasattr(torch.cuda, 'is_tf32_enabled'):
            tf32_enabled = torch.cuda.is_tf32_enabled()
            print(f"  TF32 enabled: {tf32_enabled}")
        else:
            print("  TF32: unknown")

        return True
    except Exception as e:
        print(f"Capability check: {e}")
        return False


def check_environment_vars():
    """Print relevant environment variables."""
    print("\nEnvironment variables:")
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
        'PYTORCH_CUDA_ALLOC_CONF',
        'NCCL_DEBUG',
        'NCCL_IB_DISABLE',
        'NCCL_SOCKET_IFNAME',
        'OMP_NUM_THREADS',
        'CUDA_LAUNCH_BLOCKING',
        'TORCH_HOME',
    ]

    for var in env_vars:
        value = os.environ.get(var, '(not set)')
        print(f"  {var:30} = {value}")


def check_numa():
    """Check NUMA configuration (Linux)."""
    try:
        import subprocess
        # Try numactl to get node info
        result = subprocess.run(['numactl', '--show'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nNUMA configuration (numactl --show):")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("\nNUMA info: numactl not available (no cross-socket topology data)")
    return False


def main():
    print("\n" + "="*70)
    print("PyTorch Runtime Environment Check")
    print("="*70 + "\n")

    print("Software versions:")
    check_pytorch()
    print()
    check_cuda()
    print()
    check_cudnn()
    print()
    check_nccl()

    print("\n" + "-"*70)
    print("\nHardware:")
    check_gpu_info()

    print("\n" + "-"*70)
    check_capabilities()

    print("\n" + "-"*70)
    check_environment_vars()

    print("\n" + "-"*70)
    check_numa()

    print("\n" + "="*70)
    print("Check complete. Compare versions against GPU-15 baselines:")
    print("  PyTorch: 2.15.0+")
    print("  CUDA: 13.4 Update 2 or later (CUDA 13.x minimum)")
    print("  NCCL: 2.32.3+")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
