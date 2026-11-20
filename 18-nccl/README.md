# GPU-18: NCCL — Algorithms, Protocols & Hang Diagnosis

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/18-nccl.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

CPU-runnable simulators and tests for NCCL collective communication timing and overlap analysis.

## 文件 Files

| 文件 File | 环境 Env | 说明 Description | 章节 Chapter |
|------|-------------|-------------|---------|
| `sim_collective.py` | CPU | Ring & tree all-reduce: alpha-beta model with protocol efficiency (Simple/LL/LL128); computes algbw, busbw, efficiency | 18.1–2 |
| `sim_overlap.py` | CPU | Backward-allreduce overlap: gradient bucketing models exposed communication on critical path | 18.3 |
| `test_collective.py` | CPU | Unit tests: nccl-tests formula verification, sanity checks for 1 MB (20–100 µs) and 1 GB (4–150 ms) ranges, NVLink vs cross-node | 18.1–2 |

## 运行 Running

### Basic Simulation
```bash
# Ring 8 ranks, 1 MB, NVLink 450 GB/s, Simple protocol
python3 sim_collective.py ring --num-ranks 8 --msg-bytes 1048576 --link-bw 450

# Ring, LL128 protocol (better pipelining)
python3 sim_collective.py ring --num-ranks 8 --msg-bytes 1048576 --link-bw 450 --protocol LL128

# Compare ring vs tree, all protocols
python3 sim_collective.py compare --num-ranks 8 --msg-bytes 1048576 --link-bw 450

# Cross-node: 50 GB/s 400G network
python3 sim_collective.py ring --num-ranks 8 --msg-bytes 1048576 --link-bw 50

# Message size sweep (1 KB to 1 GB)
python3 sim_collective.py ring --msg-range --num-ranks 8 --protocol LL128 --link-bw 450
```

### Overlap Simulation
```bash
# Timeline for 32 layers, 5 ms compute/layer, 80 ms comm, 8 buckets
python3 sim_overlap.py timeline --num-layers 32 --compute-time-ms 5.0 --comm-time-ms 80.0 --num-buckets 8

# Sweep: effect of bucket count (1..24)
python3 sim_overlap.py bucket-sweep --num-layers 32

# Llama 7B-like example (32 layers, 160 ms total compute, 80 ms all-reduce)
python3 sim_overlap.py example
```

### Tests
```bash
python3 test_collective.py
```

## 预期输出 Expected Output

```
# Ring 8 ranks, 1 MB, NVLink 450 GB/s, Simple protocol → ~22 µs
Collective: RING all-reduce
Protocol:    Simple
Ranks:       8
Message:     1024.0 KB
Steps:       14
Chunk:       128.0 KB

Timing:      0.022 ms
Algbw:       47.33 GB/s (actual)
Busbw:       787.50 GB/s (theoretical max)
Efficiency:  6.0%

# All 6 tests pass (including sanity checks)
6/6 tests passed
```

## 关键参数 Key Parameters

- **H100 NVLink4:** 450 GB/s single direction (intra-node)
- **400G network:** 50 GB/s single direction (cross-node)
- **Alpha (per-hop latency):** 1 µs for optimized NVLink, 1–20 µs for cross-node (tunable)
- **Protocol efficiency:** Simple 50%, LL/LL128 95% of link BW
- **Sanity ranges:**
  - 1 MB NVLink: 15–200 µs (actual: ~22 µs)
  - 1 GB NVLink: 4–15 ms (actual: ~4.4 ms)
  - 1 MB cross-node: 50–500 µs
  - 1 GB cross-node: 40–150 ms

## 模型与公式 Model & Formula

**Ring all-reduce (Patarasuk & Yuan 2009):**
```
T = 2(n-1) * (alpha + S/(n*B))
busbw = 2(n-1)/n * link_bw  # network utilization
algbw = S / T              # achieved bandwidth
```

**Protocol efficiency (Demystifying NCCL arXiv 2507.04786):**
- Simple: ~50% (protocol overhead, minimal pipelining)
- LL: ~95% (pipelined 2-buffer model)
- LL128: ~95% (atomic 128-byte lines, best for >1 MB)

**Tree depth:** log2(n), 2*log2(n) total hops

## 参考 References

- **Patarasuk & Yuan 2009:** "Bandwidth Optimal All-Reduce Algorithms for Clusters"
- **Thakur et al. 2005:** "Optimization of Collective Communication Operations in MPICH"
- **Demystifying NCCL (arXiv 2507.04786):** Protocol efficiency, real measurements
- **NCCL algorithms:** https://github.com/NVIDIA/nccl/blob/master/doc/ALGORITHMS.md
- **PyTorch DDP (VLDB 2020):** Gradient bucketing and overlap
- **MegaScale (arXiv 2402.15627):** Large-scale distributed training
