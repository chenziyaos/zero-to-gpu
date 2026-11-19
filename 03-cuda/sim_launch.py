"""Launch overhead, grid configuration, and stream overlap: simulator for GPU-03.

Topics: launch latency, grid quantization, wave efficiency, overlap simulation.

Formulas:
  grid_blocks = ceil(n / block_size)
  waves = ceil(grid_blocks / (sm_count * resident_blocks_per_sm))
  wave_utilization = (grid_blocks % (sm_count * resident_blocks_per_sm)) / (sm_count * resident_blocks_per_sm)
  h2d_pinned_tbs = 5.77  # Harris measured on T4, GB/s
  h2d_pageable_tbs = 2.31  # Harris measured on T4, GB/s
  pcie5_x16_single = 64  # GB/s
  kernel_launch_overhead = 5-10 us (CPU-side), 1-2 us (GPU-side dispatch)
  V100 kernel 2.9 us base, overlap reduces total time to 1.5x longest stage + epsilon

Subcommands:
  grid       Grid configuration, block count, wave efficiency
  index      2D to 1D index mapping for threads
  xfer       Pinned vs pageable transfer time (Harris data)
  overlap    Discrete-event simulator for H2D/compute/D2H pipelining

Usage:
  python3 sim_launch.py grid --n 1000000 --block 256 --sms 132
  python3 sim_launch.py index --bx 4 --by 2 --tx 8 --ty 4
  python3 sim_launch.py xfer --mib 256 --pcie-gen 5
  python3 sim_launch.py overlap
  python3 sim_launch.py launch
"""
import argparse
import sys
import math
from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class SimState:
    """Discrete-event simulator state for H2D/kernel/D2H overlap."""
    time: float = 0.0
    queue: list = None  # min-heap of (time, engine_id, op_type)

    def __post_init__(self):
        if self.queue is None:
            self.queue = []


def grid_config(n, block_size, sm_count, resident_per_sm=8):
    """Compute grid configuration, waves, and efficiency."""
    grid_blocks = math.ceil(n / block_size)
    slots_per_wave = sm_count * resident_per_sm
    num_waves = math.ceil(grid_blocks / slots_per_wave)

    if grid_blocks > 0:
        last_wave_blocks = ((grid_blocks - 1) % slots_per_wave) + 1
        wave_util = last_wave_blocks / slots_per_wave
    else:
        wave_util = 0

    return {
        'n': n,
        'block_size': block_size,
        'grid_blocks': grid_blocks,
        'sm_count': sm_count,
        'resident_per_sm': resident_per_sm,
        'slots_per_wave': slots_per_wave,
        'num_waves': num_waves,
        'last_wave_util': wave_util,
        'empty_slots': slots_per_wave - last_wave_blocks if grid_blocks > 0 else 0,
    }


def cmd_grid(args):
    """Subcommand: grid configuration."""
    cfg = grid_config(args.n, args.block_size, args.sms, args.resident)
    print(f"n={cfg['n']}, block_size={cfg['block_size']}, {cfg['sm_count']} SM, "
          f"{cfg['resident_per_sm']} resident/SM")
    print(f"  → {cfg['grid_blocks']} blocks, {cfg['num_waves']} waves, "
          f"last wave {cfg['last_wave_util']*100:.1f}% utilized")
    print(f"  → {cfg['empty_slots']} empty slots per full waves")
    print()

    # Grid-stride suggestion
    grid_stride_blocks = 32 * cfg['sm_count']
    print(f"Grid-stride: {grid_stride_blocks} blocks (32 × SM), fill efficiently regardless of n")
    print()

    # Common errors
    tests = [
        (70000, 1, 1, 1, 1, "合法"),
        (1, 70000, 256, 1, 1, "gridDim.y=70000 超出 [1, 65535]"),
        (1, 1, 1024, 2, 1, "blockDim.x=1024, blockDim.y=2 → 2048 线程，超过 1024"),
        (1, 1, 32, 1, 65, "blockDim.z=65 超出 [1, 64]; 2080 线程超过 1024"),
    ]
    print("Configuration checks:")
    for test_args in tests:
        if len(test_args) == 6:
            gx, gy, bx, by, bz, msg = test_args
        else:
            raise ValueError(f"Invalid test tuple length: {len(test_args)}")

        blocks = bx * by * bz
        errors = []
        if gx > 2**31 - 1:
            errors.append("gridDim.x超界")
        if gy > 65535:
            errors.append(f"gridDim.y={gy}超出[1,65535]")
        if bx > 1024:
            errors.append(f"blockDim.x={bx}超出[1,1024]")
        if by > 1024:
            errors.append(f"blockDim.y={by}超出[1,1024]")
        if bz > 64:
            errors.append(f"blockDim.z={bz}超出[1,64]")
        if blocks > 1024:
            errors.append(f"每块线程数{blocks}超过1024")

        status = "  × " + " ; ".join(errors) if errors else "  ✓ " + msg
        print(f"  grid=({gx}, {gy}, 1) block=({bx}, {by}, {bz}): {status}")


def cmd_index(args):
    """Subcommand: 2D thread to 1D index mapping."""
    bx, by, tx, ty = args.bx, args.by, args.tx, args.ty
    row_width = bx * tx

    print(f"gridDim=({bx},{by}), blockDim=({tx},{ty}), row_width={row_width}")
    print()
    print("Examples of thread to global coordinate mapping (行主序):")
    print(f"  blockIdx  threadIdx   (x, y)    global_idx   warp")

    samples = [
        (0, 0, tx-1, 0),  # last x of first block, first y
        (0, 0, 0, 1),      # first x, second y
        (1, 0, 0, 0),      # first x of second block
        (bx-1, by-1, tx-1, ty-1),  # last of everything
    ]

    for bi, bj, ti, tj in samples:
        # Global coord
        gi = bi * tx + ti
        gj = bj * ty + tj
        # Linear index in row-major
        linear = gj * row_width + gi
        # Linear thread index in block
        block_linear = tj * tx + ti
        # Which warp in block (32 threads per warp)
        warp_in_block = block_linear // 32

        print(f"  ({bi},{bj})    ({ti},{tj})      ({gi},{gj})    {linear:6d}       {warp_in_block}")


def cmd_xfer(args):
    """Subcommand: pinned vs pageable transfer time."""
    mib = args.mib
    bytes_val = mib * 1024 * 1024

    # Harris measurements on T4
    pinned_tbs = 5.77  # GB/s
    pageable_tbs = 2.31  # GB/s

    t_pinned_ms = bytes_val / (pinned_tbs * 1e9)
    t_pageable_ms = bytes_val / (pageable_tbs * 1e9)
    ratio = t_pageable_ms / t_pinned_ms

    print(f"{mib} MiB transfer:")
    print(f"  pinned   @ {pinned_tbs:.2f} GB/s → {t_pinned_ms:.2f} ms")
    print(f"  pageable @ {pageable_tbs:.2f} GB/s → {t_pageable_ms:.2f} ms (差 {ratio:.2f}倍)")
    print()

    # Theoretical PCIe limits
    gen_bw = {3: 4, 4: 8, 5: 16}  # GB/s per direction
    x16_factor = 16

    for gen in [3, 4, 5]:
        single_dir_gbs = gen_bw[gen] * x16_factor
        t_pcie_ms = bytes_val / (single_dir_gbs * 1e9)
        print(f"  PCIe {gen}.0 x16 单向 {single_dir_gbs:.0f} GB/s → {t_pcie_ms:.2f} ms")


def cmd_overlap():
    """Subcommand: discrete-event simulator for H2D/compute/D2H overlap."""
    # Gray's benchmark: V100, 20 kernels per step, 1000 steps, 2.9 us per kernel

    # Scenarios with different engine configurations and submission order
    scenarios = [
        {
            'name': '1 copy engine, single queue',
            'copy_engines': 1,
            'hyper_q': False,
            'kernel_wait_signal': False,
        },
        {
            'name': '2 copy engines, single queue + Fermi signal (前一kernel全做完才放下一个)',
            'copy_engines': 2,
            'hyper_q': False,
            'kernel_wait_signal': True,
        },
        {
            'name': '2 copy engines, single queue (无Fermi信号)',
            'copy_engines': 2,
            'hyper_q': False,
            'kernel_wait_signal': False,
        },
        {
            'name': '2 copy engines, Hyper-Q',
            'copy_engines': 2,
            'hyper_q': True,
            'kernel_wait_signal': False,
        },
        {
            'name': '1 copy engine, Hyper-Q',
            'copy_engines': 1,
            'hyper_q': True,
            'kernel_wait_signal': False,
        },
    ]

    # Times normalized: h=d=k=1 unit each
    h, k, d = 1.0, 1.0, 1.0
    n_chunks = 4
    h_per_chunk = h / n_chunks
    d_per_chunk = d / n_chunks
    k_per_chunk = k / n_chunks

    print(f"整块 H2D {h}, kernel {k}, D2H {d}, 切成 {n_chunks} 个 stream")
    print(f"  单块：H2D {h_per_chunk}, kernel {k_per_chunk}, D2H {d_per_chunk}")
    print()

    # Submission order V1: by stream
    v1_order = []
    for i in range(n_chunks):
        v1_order.extend([('h2d', i), ('kern', i), ('d2h', i)])

    # Submission order V2: by operation type
    v2_order = []
    for i in range(n_chunks):
        v2_order.append(('h2d', i))
    for i in range(n_chunks):
        v2_order.append(('kern', i))
    for i in range(n_chunks):
        v2_order.append(('d2h', i))

    results = []
    for scenario in scenarios:
        t_v1 = sim_overlap(v1_order, scenario, h_per_chunk, k_per_chunk, d_per_chunk)
        t_v2 = sim_overlap(v2_order, scenario, h_per_chunk, k_per_chunk, d_per_chunk)
        results.append((scenario['name'], t_v1, t_v2))

    print("Serial time (all sequential) = 3.00 (h + k + d)")
    print()
    print(f"{'Scenario':<50} {'V1':>8} {'V2':>8}")
    print("-" * 68)
    for name, t_v1, t_v2 in results:
        print(f"{name:<50} {t_v1:8.2f} {t_v2:8.2f}")


def sim_overlap(order, scenario, h_dur, k_dur, d_dur):
    """Simulate timeline with given submission order and engine config."""
    copy_engines = scenario['copy_engines']
    hyper_q = scenario['hyper_q']
    kernel_wait = scenario['kernel_wait_signal']

    # Event queue: (time, priority, op_id, op_type, stream_id)
    # priority: lower = earlier (for ties)
    pq = []

    # Track engine availability and queue state
    h2d_engine_free = [0.0] * min(1, copy_engines)  # H2D shares one
    d2h_engine_free = [0.0] * min(1, copy_engines)  # D2H has separate if copy_engines >= 2
    if copy_engines >= 2:
        d2h_engine_free = [0.0]

    # Queues per stream
    queues = {i: [] for i in range(4)}
    last_kernel_end = 0.0

    # Submit all operations
    for op_type, stream_id in order:
        queues[stream_id].append(op_type)

    # Simple discrete event simulation
    # For each stream in order, schedule operations as engines become free
    time = 0.0
    h2d_free_time = 0.0
    d2h_free_time = 0.0
    kernel_free_time = 0.0

    op_counter = 0
    for op_type, stream_id in order:
        if op_type == 'h2d':
            # Schedule H2D on H2D engine
            start = max(time, h2d_free_time)
            end = start + h_dur
            h2d_free_time = end
            time = end
        elif op_type == 'kern':
            # Kernels must respect Fermi signal (if enabled)
            if kernel_wait and op_counter > 0 and order[op_counter - 1][0] == 'kern':
                # Previous kernel must finish before this one starts
                start = max(time, kernel_free_time)
            else:
                start = max(time, kernel_free_time)
            end = start + k_dur
            kernel_free_time = end
            last_kernel_end = end
            time = end
        elif op_type == 'd2h':
            # Schedule D2H on D2H engine (may be same as H2D if only 1 copy engine)
            # But D2H must respect kernel completion
            start = max(time, d2h_free_time, last_kernel_end)
            end = start + d_dur
            d2h_free_time = end
            time = end

        op_counter += 1

    return time


def cmd_launch():
    """Subcommand: launch overhead benchmark replication (Gray's CUDA Graph paper)."""
    # Gray's numbers: V100, 20 kernels/step, 1000 steps, 2.9 us per kernel
    kernel_time = 2.9e-6  # seconds
    num_kernels_per_step = 20
    num_steps = 1000

    # Each iteration overhead estimates (from Gray's paper)
    overhead_per_sync = 9.6e-6  # sec, each kernel after sync
    overhead_per_step = 3.8e-6  # sec per kernel, one sync per step
    overhead_graph = 3.4e-6     # sec per kernel with Graph
    graph_build = 400e-6        # sec to build graph

    total_sync_each = num_kernels_per_step * overhead_per_sync * num_steps
    total_sync_per_step = num_kernels_per_step * overhead_per_step * num_steps
    total_graph_launch = num_kernels_per_step * overhead_graph * num_steps
    graph_amortized = graph_build / (num_kernels_per_step)  # amortized per kernel

    print("20 kernels/step, 1000 steps, kernel base 2.9 us (V100):")
    print()

    print("1. 每个 kernel 后 sync：")
    print(f"   每 kernel  {overhead_per_sync*1e6:.1f} us")
    print(f"   总计      {total_sync_each*1e3:.1f} ms")
    print(f"   开销占比  {(total_sync_each - num_kernels_per_step*num_steps*kernel_time) / total_sync_each * 100:.1f}%")
    print()

    print("2. 每步 sync 一次：")
    print(f"   每 kernel  {overhead_per_step*1e6:.1f} us")
    print(f"   总计      {total_sync_per_step*1e3:.1f} ms")
    print(f"   开销占比  {(total_sync_per_step - num_kernels_per_step*num_steps*kernel_time) / total_sync_per_step * 100:.1f}%")
    print()

    print("3. CUDA Graph：")
    print(f"   每 kernel  {overhead_graph*1e6:.1f} us")
    print(f"   总计      {total_graph_launch*1e3:.1f} ms")
    print(f"   开销占比  {(total_graph_launch - num_kernels_per_step*num_steps*kernel_time) / total_graph_launch * 100:.1f}%")
    print()

    print(f"Graph 建图 {graph_build*1e6:.0f} us，每步省 {(overhead_per_step - overhead_graph)*1e6:.1f} us，")
    print(f"  {graph_build / ((overhead_per_step - overhead_graph) * num_kernels_per_step):.0f} 步回本")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='cmd', help='Subcommand')

    # grid
    p_grid = subparsers.add_parser('grid', help='Grid configuration and wave efficiency')
    p_grid.add_argument('--n', type=int, default=1000000, help='Total elements')
    p_grid.add_argument('--block', type=int, dest='block_size', default=256, help='Block size (threads)')
    p_grid.add_argument('--sms', type=int, default=132, help='SM count')
    p_grid.add_argument('--resident', type=int, default=8, help='Resident blocks per SM')
    p_grid.set_defaults(func=lambda a: cmd_grid(a))

    # index
    p_index = subparsers.add_parser('index', help='2D index mapping')
    p_index.add_argument('--bx', type=int, default=4, help='gridDim.x')
    p_index.add_argument('--by', type=int, default=2, help='gridDim.y')
    p_index.add_argument('--tx', type=int, default=8, help='blockDim.x')
    p_index.add_argument('--ty', type=int, default=4, help='blockDim.y')
    p_index.set_defaults(func=lambda a: cmd_index(a))

    # xfer
    p_xfer = subparsers.add_parser('xfer', help='Transfer time: pinned vs pageable')
    p_xfer.add_argument('--mib', type=int, default=256, help='Transfer size in MiB')
    p_xfer.add_argument('--pcie-gen', type=int, default=5, help='PCIe generation')
    p_xfer.set_defaults(func=lambda a: cmd_xfer(a))

    # overlap
    p_overlap = subparsers.add_parser('overlap', help='H2D/compute/D2H overlap scenarios')
    p_overlap.set_defaults(func=lambda a: cmd_overlap())

    # launch
    p_launch = subparsers.add_parser('launch', help='Launch overhead: Gray CUDA Graph benchmark')
    p_launch.set_defaults(func=lambda a: cmd_launch())

    args = parser.parse_args()
    if args.cmd:
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
