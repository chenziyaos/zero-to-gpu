"""PyTorch CUDA 缓存分配器模拟：块、池、切分、合并与碎片（CPU 模拟）。对应 GPU-10 第 2 至 4 章。

核心机制（c10/cuda/CUDACachingAllocator.cpp）：
  1. 所有请求向上取整到 512 字节的倍数（BLOCK_LO 对齐）。
  2. 请求分成两个池：≤1 MiB 进小池（small），>1 MiB 进大池（large）。
  3. 查找：在目标池里找首个 >= 请求大小的空闲块。找到复用；找不到向 CUDA 要新段。
  4. 小池一次申请 2 MiB 段（SmallBuffer），大池一次申请 20 MiB 段（LargeBuffer）。
  5. 复用时若块太大，从末尾切下多余部分放回空闲池。
  6. 释放时与相邻空闲块合并，缩小碎片。
  7. OOM 前重试：释放超大块、同步全流释放空闲段。

子命令：
  allocate   模拟单条分配请求，显示取整、选池、查找、切分过程
  sequence   模拟一系列请求，跟踪空闲池和碎片演变
  fragstress 病理性碎片：重复分配、释放不对称大小，观察碎片率
  poolstats  给定一组请求模式，统计两池的段数、有效利用率

用法：
  python3 sim_alloc.py allocate --request-bytes 1048576
  python3 sim_alloc.py sequence --pattern "1M,2M,1M,500K,2M"
  python3 sim_alloc.py fragstress --iterations 100 --seed 42
  python3 sim_alloc.py poolstats --pattern-file patterns.txt
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import GPUS, get  # noqa: E402

# ===== PyTorch CUDACachingAllocator 常量 =====
# 来源：pytorch/c10/cuda/CUDACachingAllocator.cpp 及 CUDA 文档
# 核对于 2026-11-04

BLOCK_LO = 512  # 块对齐到 512 B；见 CUDACachingAllocator.cpp kSmallSize, kLargeSize 附近
SMALL_POOL_THRESHOLD = 1 * 1024 * 1024  # 1 MiB 阈值；见 c10/cuda/CUDACachingAllocator.cpp
SMALL_BUFFER_SIZE = 2 * 1024 * 1024  # 小池一次申请 2 MiB 段
LARGE_BUFFER_SIZE = 20 * 1024 * 1024  # 大池一次申请 20 MiB 段

def round_up(size, alignment):
    """向上取整到对齐边界。"""
    return ((size + alignment - 1) // alignment) * alignment


@dataclass
class Block:
    """内存块。"""
    size: int  # 实际大小（已对齐）
    allocated: bool
    alloc_id: int = 0  # 分配 ID（用于追踪）
    segment_id: int = 0  # 所属段 ID


@dataclass
class Segment:
    """内存段。对应一次 cudaMalloc。"""
    segment_id: int
    size: int
    pool: str  # 'small' 或 'large'
    blocks: List[Block] = field(default_factory=list)

    def fragmentation(self):
        """段内碎片率：空闲字节 / 段大小。"""
        free_bytes = sum(b.size for b in self.blocks if not b.allocated)
        return free_bytes / self.size if self.size > 0 else 0.0


class CachingAllocator:
    """PyTorch 风格的缓存分配器。"""

    def __init__(self, gpu_name='h100-sxm'):
        self.gpu = get(gpu_name)
        self.segments: Dict[int, Segment] = {}
        self.free_blocks: Dict[str, List[Block]] = {'small': [], 'large': []}
        self.next_alloc_id = 1
        self.next_segment_id = 1
        self.total_allocated = 0
        self.total_reserved = 0
        self.num_alloc_retries = 0

    def choose_pool(self, aligned_size):
        """选择池。"""
        return 'small' if aligned_size <= SMALL_POOL_THRESHOLD else 'large'

    def allocate(self, requested_size: int, alloc_id_hint=None) -> Tuple[int, str, str]:
        """
        分配内存。
        返回：(分配ID, 池名, 所在段ID)
        """
        # 第 1 步：取整
        aligned_size = round_up(requested_size, BLOCK_LO)
        pool = self.choose_pool(aligned_size)

        # 第 2 步：查找现有段里的空闲块
        best_block = None
        best_segment_id = None
        best_block_size = float('inf')
        for seg_id, seg in self.segments.items():
            if seg.pool != pool:
                continue
            for i, block in enumerate(seg.blocks):
                if not block.allocated and block.size >= aligned_size:
                    if block.size < best_block_size:
                        best_block = (seg_id, i, block)
                        best_block_size = block.size
                        best_segment_id = seg_id

        if best_block is not None:
            seg_id, block_idx, best_blk_obj = best_block
            # 第 3 步：切分
            segment = self.segments[seg_id]
            block = segment.blocks[block_idx]
            if block.size > aligned_size:
                # 从末尾切出多余部分
                new_block = Block(
                    size=block.size - aligned_size,
                    allocated=False,
                    segment_id=seg_id
                )
                block.size = aligned_size
                segment.blocks.insert(block_idx + 1, new_block)
            block.allocated = True
            block.alloc_id = self.next_alloc_id
            self.total_allocated += aligned_size
            result_id = self.next_alloc_id
            self.next_alloc_id += 1
            return result_id, pool, f"seg{seg_id}"

        # 第 4 步：找不到，分配新段
        buffer_size = SMALL_BUFFER_SIZE if pool == 'small' else LARGE_BUFFER_SIZE
        if aligned_size > buffer_size:
            # 特大请求，段大小等于请求大小
            buffer_size = aligned_size

        new_segment = Segment(
            segment_id=self.next_segment_id,
            size=buffer_size,
            pool=pool,
            blocks=[Block(size=buffer_size, allocated=False)]
        )
        seg_id = self.next_segment_id
        self.segments[seg_id] = new_segment
        self.total_reserved += buffer_size
        self.next_segment_id += 1

        # 递归分配（从新段里分配）
        return self.allocate(requested_size, alloc_id_hint)

    def free(self, alloc_id: int):
        """释放分配。"""
        freed = False
        for seg in self.segments.values():
            for block in seg.blocks:
                if block.allocated and block.alloc_id == alloc_id:
                    self.total_allocated -= block.size
                    block.allocated = False
                    freed = True
                    break

        if freed:
            self.coalesce_segment()

    def coalesce_segment(self):
        """合并相邻空闲块。"""
        for seg in self.segments.values():
            blocks = seg.blocks
            merged = True
            while merged:
                merged = False
                for i in range(len(blocks) - 1):
                    if not blocks[i].allocated and not blocks[i+1].allocated:
                        blocks[i].size += blocks[i+1].size
                        blocks.pop(i+1)
                        merged = True
                        break

    def get_stats(self):
        """返回统计信息。"""
        fragmentation = {}
        total_free = 0
        for pool in ['small', 'large']:
            pool_segments = [s for s in self.segments.values() if s.pool == pool]
            pool_frag = sum(s.fragmentation() for s in pool_segments) / len(pool_segments) if pool_segments else 0.0
            fragmentation[pool] = pool_frag
            for seg in pool_segments:
                total_free += sum(b.size for b in seg.blocks if not b.allocated)

        return {
            'allocated': self.total_allocated,
            'reserved': self.total_reserved,
            'free': total_free,
            'num_segments': len(self.segments),
            'small_pool_segments': len([s for s in self.segments.values() if s.pool == 'small']),
            'large_pool_segments': len([s for s in self.segments.values() if s.pool == 'large']),
            'fragmentation': fragmentation,
        }


def cmd_allocate(args):
    """模拟单条分配请求。"""
    alloc = CachingAllocator(args.gpu)
    req = args.request_bytes

    aligned = round_up(req, BLOCK_LO)
    pool = 'small' if aligned <= SMALL_POOL_THRESHOLD else 'large'
    buffer_size = SMALL_BUFFER_SIZE if pool == 'small' else LARGE_BUFFER_SIZE

    print(f"{'请求':12} {req:>15,} 字节")
    print(f"{'对齐后':12} {aligned:>15,} 字节（对齐到 {BLOCK_LO} B）")
    print(f"{'选择池':12} {pool} 池（阈值 {SMALL_POOL_THRESHOLD} B）")
    print(f"{'段大小':12} {buffer_size:>15,} 字节")

    alloc_id, chosen_pool, seg_loc = alloc.allocate(req)
    stats = alloc.get_stats()

    print(f"{'分配ID':12} {alloc_id}")
    print(f"{'位置':12} {seg_loc}")
    print()
    print(f"{'已分配':12} {stats['allocated']:>15,} 字节")
    print(f"{'已保留':12} {stats['reserved']:>15,} 字节")
    print(f"{'浪费':12} {stats['reserved'] - stats['allocated']:>15,} 字节")
    print(f"{'利用率':12} {100 * stats['allocated'] / stats['reserved']:>14.1f}%")


def cmd_sequence(args):
    """模拟一系列请求。"""
    sizes_str = args.pattern
    sizes = []
    for s in sizes_str.split(','):
        s = s.strip()
        mult = 1
        if s.endswith('K'):
            mult = 1024
            s = s[:-1]
        elif s.endswith('M'):
            mult = 1024 * 1024
            s = s[:-1]
        elif s.endswith('G'):
            mult = 1024 * 1024 * 1024
            s = s[:-1]
        sizes.append(int(s) * mult)

    alloc = CachingAllocator(args.gpu)
    alloc_ids = []

    print(f"{'步':>3} {'操作':8} {'请求':>12} {'对齐':>12} {'池':8} {'已分配':>12} {'已保留':>12} {'利用率':>8}")
    print("-" * 100)

    for i, size in enumerate(sizes):
        alloc_id, pool, _ = alloc.allocate(size)
        alloc_ids.append(alloc_id)
        stats = alloc.get_stats()
        util = 100 * stats['allocated'] / stats['reserved'] if stats['reserved'] > 0 else 0
        aligned = round_up(size, BLOCK_LO)
        print(f"{i:3d} {'分配':8} {size:>12,} {aligned:>12,} {pool:8} {stats['allocated']:>12,} {stats['reserved']:>12,} {util:7.1f}%")

    print()
    print("按顺序释放：")
    for i, alloc_id in enumerate(alloc_ids):
        alloc.free(alloc_id)
        stats = alloc.get_stats()
        util = 100 * stats['allocated'] / stats['reserved'] if stats['reserved'] > 0 else 0
        print(f"{i:3d} {'释放':8} {'':>12} {'':>12} {'':8} {stats['allocated']:>12,} {stats['reserved']:>12,} {util:7.1f}%")


def cmd_fragstress(args):
    """病理性碎片：重复分配、释放不对称大小。"""
    random.seed(args.seed)
    alloc = CachingAllocator(args.gpu)

    sizes = [512 * 1024, 1024 * 1024, 2 * 1024 * 1024]  # 512K, 1M, 2M
    alloc_ids = []

    print(f"{'迭代':>3} {'操作':8} {'已分配':>12} {'已保留':>12} {'碎片(small)':>12} {'碎片(large)':>12}")
    print("-" * 80)

    for iteration in range(args.iterations):
        # 分配
        size = random.choice(sizes)
        alloc_id, _, _ = alloc.allocate(size)
        alloc_ids.append(alloc_id)

        # 随机释放一个（不一定是最后分配的）
        if len(alloc_ids) > 2 and random.random() < 0.5:
            idx = random.randint(0, len(alloc_ids) - 2)
            alloc.free(alloc_ids[idx])
            alloc_ids.pop(idx)

        stats = alloc.get_stats()
        if iteration % 10 == 0 or iteration == args.iterations - 1:
            frag_small = stats['fragmentation'].get('small', 0)
            frag_large = stats['fragmentation'].get('large', 0)
            print(f"{iteration:3d} {'':8} {stats['allocated']:>12,} {stats['reserved']:>12,} {frag_small:>11.1%} {frag_large:>11.1%}")

    print()
    final_stats = alloc.get_stats()
    print(f"最终已分配：{final_stats['allocated']:,}")
    print(f"最终已保留：{final_stats['reserved']:,}")
    print(f"浪费（碎片 + 缓存）：{final_stats['reserved'] - final_stats['allocated']:,}")


def cmd_poolstats(args):
    """给定一组请求，统计两池情况。"""
    # 简单生成：10 个小请求 + 5 个大请求
    requests = [100 * 1024 for _ in range(10)] + [10 * 1024 * 1024 for _ in range(5)]
    random.shuffle(requests)

    alloc = CachingAllocator(args.gpu)
    for req in requests:
        alloc.allocate(req)

    stats = alloc.get_stats()
    print(f"{'池':8} {'段数':>6} {'总大小':>15} {'已分配':>15} {'空闲':>15} {'碎片率':>8}")
    print("-" * 80)

    for pool in ['small', 'large']:
        segs = [s for s in alloc.segments.values() if s.pool == pool]
        if segs:
            seg_size = sum(s.size for s in segs)
            seg_alloc = sum(sum(b.size for b in s.blocks if b.allocated) for s in segs)
            seg_free = seg_size - seg_alloc
            frag = stats['fragmentation'].get(pool, 0)
            print(f"{pool:8} {len(segs):>6} {seg_size:>15,} {seg_alloc:>15,} {seg_free:>15,} {frag:>7.1%}")

    print()
    print(f"总计已分配：{stats['allocated']:,}")
    print(f"总计已保留：{stats['reserved']:,}")
    print(f"利用率：{100 * stats['allocated'] / stats['reserved']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', default='h100-sxm', help='GPU 型号（默认 h100-sxm）')
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    sp_alloc = subparsers.add_parser('allocate', help='单条分配请求')
    sp_alloc.add_argument('--request-bytes', type=int, default=1048576, help='请求大小（字节）')

    sp_seq = subparsers.add_parser('sequence', help='一系列请求')
    sp_seq.add_argument('--pattern', default='1M,2M,1M,500K,2M', help='大小模式，逗号分隔（支持 K/M/G 后缀）')

    sp_frag = subparsers.add_parser('fragstress', help='碎片压力测试')
    sp_frag.add_argument('--iterations', type=int, default=100, help='迭代次数')
    sp_frag.add_argument('--seed', type=int, default=42, help='随机种子')

    sp_pool = subparsers.add_parser('poolstats', help='两池统计')
    sp_pool.add_argument('--pattern-file', help='请求模式文件（可选）')

    args = parser.parse_args()

    if args.cmd == 'allocate':
        cmd_allocate(args)
    elif args.cmd == 'sequence':
        cmd_sequence(args)
    elif args.cmd == 'fragstress':
        cmd_fragstress(args)
    elif args.cmd == 'poolstats':
        cmd_poolstats(args)


if __name__ == '__main__':
    main()
