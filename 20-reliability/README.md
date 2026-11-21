# GPU-20: Reliability & Observability — Checkpoint Interval, Failure Rates & Stragglers

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/20-reliability.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-20](../book/20-reliability.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

Large-scale GPU training faces three main reliability challenges: (1) hardware failures cause job interruption, (2) slow nodes stall synchronous training, (3) silent data corruption corrupts models. This directory contains simulators for checkpoint interval optimization (Young/Daly formulas), job-level failure rate estimation, and straggler impact analysis.

## Files

| File | Environment | Purpose | Docs Section |
|------|-------------|---------|--------|
| `sim_ckpt.py` | CPU-only, Python 3.10+ | Young (1974) and Daly (2006) checkpoint interval formulas; ETTR calculation | 2. Checkpoint Interval Optimization |
| `sim_failures.py` | CPU-only, Python 3.10+ | Job-level MTBF from per-card failure rates; Monte Carlo training simulation | 3. Failure Distribution at Scale |
| `sim_straggler.py` | CPU-only, Python 3.10+ | Straggler slowdown in synchronous all-reduce training | 4. Slow Nodes & Synchronization Stalls |
| `test_reliability.py` | CPU-only, Python 3.10+ unittest | Verify against Llama 3 (16384 H100s: 419 failures/54d) and formulas | All simulators |

## Quick Start

### Checkpoint Interval Optimization

Find the optimal checkpoint interval for a 100k-step job on 16 cards with 1.5h job MTBF:

```bash
python3 sim_ckpt.py --total-steps 100000 --ckpt-time 600 --mtbf 86400 \
                     --recovery-time 300 --step-time 0.1 --scale 16
```

Output shows Young and Daly optima and ETTR (goodput) at each interval.

### Failure Simulation (Llama 3 Replica)

Verify against Llama 3 training: 16384 H100s, 419 interruptions in 54 days:

```bash
python3 sim_failures.py --llama3
```

Expected: Job MTBF ≈ 3.09 hours, ~1 failure per 100k-step training run.

### Straggler Impact

Measure slowdown from 1-5 stragglers running 50% slower:

```bash
python3 sim_straggler.py --scan
```

Expected: 1 straggler at 1.5x slowdown = 33% goodput loss in 8-node training.

### Run Tests

```bash
python3 test_reliability.py -v
```

All tests verify against published references and Llama 3 data.

## Expected Output

Checkpoint interval scan (best interval ≈ 25455 steps at 1.5h job MTBF):
```
Optimal checkpoint interval: Young(25455)) steps
  Expected total time: 3.65 hours
  ETTR (goodput): 76.2%
```

Llama 3 replica (100 MC runs):
```
Average failures per run: 1.0
Average ETTR (goodput): 14.2%
```

Straggler scan (1.5x slowdown):
```
Stragglers  Slowdown  Goodput Loss
         1      1.500        33.3%
```

## Key Parameters & Formulas

**Young (1974):** τ* = √(2·C·T_MTBF)  
**Daly (2006):** τ* = √(2·T_MTBF / (1 + √(1 + 2·R/T_MTBF)))

- C = checkpoint write time (s)
- T_MTBF = job Mean Time Between Failures (s)
- R = recovery time including read & restart (s)
- τ* = optimal checkpoint interval in seconds

**Job MTBF scales inversely with cluster size:**
- λ_job = N · λ_card (job fails when any card fails)
- MTBF_job = MTBF_card / N

## Reference Data

- **Llama 3 (arXiv 2407.21783):** 16384 H100s, 54 days, 419 failures → 3.09 h job MTBF
- **MegaScale (NSDI 2024, arXiv 2402.15627):** 12288 H100s, 55.2% MFU with fault tolerance
- **Daly (2006, arXiv 0704.3859):** Checkpoint interval optimization
- **Young (1974):** Optimal checkpoint placement with failures
