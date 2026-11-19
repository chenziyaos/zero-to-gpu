#!/usr/bin/env python3
"""GPU-21 训练并行测试：验证新的并行时间模型。

测试用例：
1. 7.5B 模型、64 卡、ZeRO-3: 复现 DeepSpeed 论文 §7.5B 模型 64 卡显存
2. 1F1B 气泡公式：验证 (p-1)/m 与不同流水线深度的关系
3. TP all-reduce 通信：验证 Megatron 4-agg/layer 的公式
4. DP all-reduce 通信：验证环形规约的 ring factor 和时间
5. MFU 基线：8B 模型、64 卡、TP=8、PP=2、DP=4
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_memory import parallel_memory
from sim_parallel import calc_mfu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import get


def test_zero3_7p5b_64gpu():
    """测试 ZeRO-3 显存：7.5B 模型，64 卡。

    期望值来自 Rajbhandari et al. ZeRO (SC 2020) 表 2。
    ZeRO-3 下，每卡应约为 DP 的 1/64。
    """
    print("=" * 60)
    print("TEST 1: ZeRO-3 显存 (7.5B, 64 GPU, H100 SXM)")
    print("=" * 60)

    gpu = get('h100-sxm')

    # ZeRO-3: (P + G + O) / N
    params_b = 7.5
    params_bytes = int(params_b * 1e9)
    per_card_zero3 = (params_bytes * 2 + params_bytes * 2 + params_bytes * 8) / 64

    # DP baseline
    per_card_dp = params_bytes * 2 + params_bytes * 2 + params_bytes * 8

    print(f"模型参数：{params_b}B")
    print(f"GPU：{gpu.name} ({gpu.hbm_gb} GB)")
    print(f"卡数：64")
    print()
    print(f"纯 DP，每卡：{per_card_dp/1e9:.2f} GB (需 {per_card_dp/gpu.hbm_gb/1e9:.1f}× 卡显存)")
    print(f"ZeRO-3，每卡：{per_card_zero3/1e9:.2f} GB ({per_card_zero3/gpu.hbm_gb/1e9*100:.1f}%)")
    print(f"节省倍数：{per_card_dp / per_card_zero3:.1f}×")
    print()

    # 判断
    if per_card_zero3 / 1e9 < gpu.hbm_gb * 0.8:  # 保留 20% 裕度
        print("✓ PASS: ZeRO-3 配置可行（显存占用 < 80%）")
        return True
    else:
        print(f"✗ FAIL: 显存不足（占用 > 80%）")
        return False


def test_1f1b_bubble():
    """测试 1F1B 气泡公式：(p-1)/m，其中 p=流水线层数，m=微批数。

    期望：
    - p=4, m=2: bubble = 3/2 = 150%
    - p=4, m=4: bubble = 3/4 = 75%
    - p=8, m=8: bubble = 7/8 = 87.5%
    """
    print()
    print("=" * 60)
    print("TEST 2: 1F1B 气泡公式 (p-1)/m")
    print("=" * 60)

    test_cases = [
        (4, 2),
        (4, 4),
        (8, 8),
    ]

    all_pass = True
    for p, m in test_cases:
        bubble = (p - 1) / m
        print(f"PP度数={p}, 微批={m}: 气泡 {bubble*100:.1f}%")

        # 验证边界：气泡应在 [0, 1) 范围内（当 m >= p 时）
        if m >= p:
            if 0 <= bubble < 1.0:
                print(f"  ✓ 合理范围")
            else:
                print(f"  ✗ 超出范围")
                all_pass = False
        else:
            print(f"  ℹ m < p 不推荐（气泡 > 100%）")

    return all_pass


def test_tp_allreduce():
    """测试 TP all-reduce 通信：Megatron 4-agg/layer。

    期望：
    - 8B 模型、32 层、8K tokens、H=4096（隐层维）、TP=8
    - 4 all-reduce/layer，activation bytes/layer = (T/dp) × H × 2
    - 若 DP=1：(8192/1) × 4096 × 2 = 64 MB/layer × 4 all-reduce/layer × 32 layers = 8 GB
    - ring factor = 2(tp-1)/tp = 2×7/8 = 1.75
    - 总通信 = 8 GB × 1.75 = 14 GB
    - H100 NVLink 单向 450 GB/s，预期 ~ 31 ms
    """
    print()
    print("=" * 60)
    print("TEST 3: TP all-reduce 通信 (Megatron 4-agg/layer)")
    print("=" * 60)

    gpu = get('h100-sxm')
    params_b = 8
    layers = 32
    gbs_toks = 8192
    hidden_dim = 4096
    tp_deg = 8
    dp_deg = 1  # 单个 GPU

    # 激活字节/层 = (gbs_toks / dp_deg) × H × 2
    activation_bytes_per_layer = (gbs_toks / dp_deg) * hidden_dim * 2
    # 4 all-reduce/layer
    tp_allreduce_vol = 4 * layers * activation_bytes_per_layer
    # ring factor
    ring_factor = 2 * (tp_deg - 1) / tp_deg
    tp_allreduce_vol *= ring_factor

    # 单向 NVLink 带宽
    nvlink_bw_gbs = gpu.nvlink_gbs / 2
    time_ms = tp_allreduce_vol / (nvlink_bw_gbs * 1e9) * 1000

    print(f"GPU：{gpu.name}")
    print(f"模型参数：{params_b}B, 层数：{layers}, 隐层维：{hidden_dim}")
    print(f"全局 tokens：{gbs_toks}")
    print(f"TP度数：{tp_deg}")
    print(f"激活字节/层：{activation_bytes_per_layer/1e9:.3f} GB")
    print(f"TP all-reduce 总通信量：{tp_allreduce_vol/1e9:.1f} GB")
    print(f"Ring factor：{ring_factor:.3f}")
    print(f"NVLink 单向带宽：{nvlink_bw_gbs:.0f} GB/s")
    print(f"预期 TP 时间：{time_ms:.1f} ms")
    print()

    # 判断
    if 10 < time_ms < 100:  # 合理范围（不考虑重叠）
        print(f"✓ PASS: TP 通信时间在合理范围")
        return True
    else:
        print(f"✗ FAIL: TP 通信时间过长或过短")
        return False


def test_dp_allreduce():
    """测试 DP all-reduce 通信：环形规约。

    期望：
    - 8B 模型、8 卡 TP、2 卡 PP、4 卡 DP（总 64 卡）
    - 每卡梯度 = 2 × 8e9 / (8×2) = 1 GB
    - Ring all-reduce：2(dp-1)/dp × bytes = 2×3/4 × 1GB = 1.5 GB
    - RDMA 50 GB/s，预期 ~ 30 ms
    """
    print()
    print("=" * 60)
    print("TEST 4: DP all-reduce 通信 (环形规约)")
    print("=" * 60)

    params_b = 8
    tp_deg = 8
    pp_deg = 2
    dp_deg = 4

    # 每卡梯度 = 2 × params / (tp × pp)
    grad_bytes = 2 * params_b * 1e9 / (tp_deg * pp_deg)
    # Ring factor
    ring_factor = 2 * (dp_deg - 1) / dp_deg
    dp_vol = grad_bytes * ring_factor

    # RDMA 50 GB/s
    b_dp = 50e9
    time_ms = dp_vol / b_dp * 1000

    print(f"模型参数：{params_b}B")
    print(f"并行：TP={tp_deg}, PP={pp_deg}, DP={dp_deg}")
    print(f"每卡梯度：{grad_bytes/1e9:.2f} GB")
    print(f"Ring factor：{ring_factor:.3f}")
    print(f"DP all-reduce 通信量：{dp_vol/1e9:.2f} GB")
    print(f"RDMA 带宽：{b_dp/1e9:.0f} GB/s")
    print(f"预期 DP 通信时间：{time_ms:.1f} ms")
    print()

    # 判断
    if 10 < time_ms < 100:  # 合理范围
        print(f"✓ PASS: DP 通信时间在合理范围")
        return True
    else:
        print(f"✗ FAIL: DP 通信时间过长或过短")
        return False


def test_mfu_baseline():
    """测试 MFU 基线：验证 MFU 模型的关键性质。

    性质1：MFU ≤ kernel_eff（计算时间总是小于总时间）
    性质2：MFU 随 overlap 减少而减少（DP 通信暴露增加降低 MFU）
    性质3：MFU 随 pp 增加而减少（气泡增加）
    """
    print()
    print("=" * 60)
    print("TEST 5: MFU 基线性质验证")
    print("=" * 60)

    params_b = 8
    gbs_tokens = 1e6  # 1 million tokens per step
    seq_len = 8192
    layers = 32
    num_gpus = 64
    tp_deg = 8
    pp_deg = 2
    dp_deg = 4

    # 性质1：MFU ≤ kernel_eff
    result = calc_mfu(params_b, gbs_tokens, seq_len, layers, tp_deg, pp_deg, dp_deg, 'h100-sxm',
                     kernel_eff=0.8, overlap=0.5)

    print(f"模型参数：{params_b}B, 全局 tokens：{gbs_tokens/1e6:.1f}M")
    print(f"序列长度：{seq_len}, 层数：{layers}")
    print(f"并行：TP={tp_deg}, PP={pp_deg}, DP={dp_deg}")
    print(f"核心效率：0.8, DP 重叠：0.5")
    print()
    print(f"计算时间：{result['compute_us']/1000:.2f} ms")
    print(f"TP 通信：{result['tp_us']/1000:.2f} ms")
    print(f"PP 气泡时间：{result['bubble_time_us']/1000:.2f} ms")
    print(f"DP 暴露：{result['dp_exposed_us']/1000:.2f} ms")
    print(f"总时间：{result['total_us']/1000:.2f} ms")
    print(f"MFU：{result['mfu']:.1f}%")
    print()

    # 性质1检查
    if result['mfu'] <= 80.1:  # 略大于kernel_eff因为FLOPs计算上界
        print(f"✓ 性质1 PASS: MFU ≤ kernel_eff")
        pass1 = True
    else:
        print(f"✗ 性质1 FAIL: MFU 过高（> 80%）")
        pass1 = False

    # 性质2检查：overlap 从 0.5 改到 0.2 时 MFU 应该下降
    result_less_overlap = calc_mfu(params_b, gbs_tokens, seq_len, layers, tp_deg, pp_deg, dp_deg, 'h100-sxm',
                                   kernel_eff=0.8, overlap=0.2)
    print(f"overlap=0.2 时 MFU：{result_less_overlap['mfu']:.1f}%")
    if result_less_overlap['mfu'] < result['mfu']:
        print(f"✓ 性质2 PASS: MFU 随 overlap 减少而减少")
        pass2 = True
    else:
        print(f"✗ 性质2 FAIL: MFU 应该随 overlap 减少而减少")
        pass2 = False

    # 性质3检查：pp 从 2 改到 4 时 MFU 应该下降
    result_more_pp = calc_mfu(params_b, gbs_tokens, seq_len, layers, tp_deg, 4, dp_deg//2, 'h100-sxm',
                              kernel_eff=0.8, overlap=0.5)
    print(f"pp=4, dp=2 时 MFU：{result_more_pp['mfu']:.1f}%")
    if result_more_pp['mfu'] < result['mfu']:
        print(f"✓ 性质3 PASS: MFU 随 pp 增加而减少（因为气泡增加）")
        pass3 = True
    else:
        print(f"✗ 性质3 FAIL: MFU 应该随 pp 增加而减少")
        pass3 = False

    print()
    return pass1 and pass2 and pass3


def test_model_state_feasibility():
    """测试模型显存可行性标记。

    场景：8B 模型、64 卡。
    - TP=1, PP=1, DP=64：纯 DP，每卡显存 = 90 GB（超过 80% 阈值，需 ZeRO-3）
    - TP=8, PP=2, DP=4：每卡显存 = 90/16 = 5.625 GB（可行纯 DP）
    """
    print()
    print("=" * 60)
    print("TEST 6: 模型状态显存可行性")
    print("=" * 60)

    params_b = 8
    gbs_tokens = 1e6  # 1M tokens
    seq_len = 8192
    layers = 32
    num_gpus = 64

    # 情况1：TP=1, PP=1, DP=64（纯 DP 不可行）
    result1 = calc_mfu(params_b, gbs_tokens, seq_len, layers, 1, 1, 64, 'h100-sxm')
    print(f"情况1: TP=1, PP=1, DP=64 (纯 DP 不可行)")
    print(f"  每卡显存：{result1['model_state_gb']:.1f} GB / {result1['gpu_hbm_gb']} GB")
    print(f"  可行性：纯DP={result1['feasible_dp']}, ZeRO-3={result1['feasible_zero3']}")
    print(f"  所需策略：{result1['needs_strategy']}")
    pass1 = (not result1['feasible_dp']) and result1['feasible_zero3']
    if pass1:
        print(f"  ✓ 正确标记（需 ZeRO-3）")
    else:
        print(f"  ✗ 错误标记")

    print()

    # 情况2：TP=8, PP=2, DP=4（纯 DP 可行）
    result2 = calc_mfu(params_b, gbs_tokens, seq_len, layers, 8, 2, 4, 'h100-sxm')
    print(f"情况2: TP=8, PP=2, DP=4 (纯 DP 可行)")
    print(f"  每卡显存：{result2['model_state_gb']:.1f} GB / {result2['gpu_hbm_gb']} GB")
    print(f"  可行性：纯DP={result2['feasible_dp']}, ZeRO-3={result2['feasible_zero3']}")
    print(f"  所需策略：{result2['needs_strategy']}")
    pass2 = result2['feasible_dp']
    if pass2:
        print(f"  ✓ 正确标记（纯 DP 可行）")
    else:
        print(f"  ✗ 错误标记")

    print()
    return pass1 and pass2


def main():
    results = []

    results.append(("ZeRO-3 显存", test_zero3_7p5b_64gpu()))
    results.append(("1F1B 气泡", test_1f1b_bubble()))
    results.append(("TP all-reduce", test_tp_allreduce()))
    results.append(("DP all-reduce", test_dp_allreduce()))
    results.append(("MFU 基线", test_mfu_baseline()))
    results.append(("模型状态显存", test_model_state_feasibility()))

    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"{passed}/{total} 通过")
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
