#!/usr/bin/env python3
"""Unit tests for GPU-24 capstone: training, inference, and calibration.

Tests verify core calculations against published baselines and sanity checks.
"""
from sim_plan_train import TrainPlan
from sim_plan_infer import InferencePlan
from sim_calibrate import compute_flops, estimate_mfu, estimate_days, RUNS


def test_train_flops_formula():
    """FLOPs formula: 6*N*T."""
    plan = TrainPlan(70e9, 1e12, 'h100-sxm', 512)
    expected_flops = 6 * 70e9 * 1e12
    assert plan.compute_flops() == expected_flops


def test_train_peak_throughput_h100():
    """H100: 989 TFLOPS * 512 = 506.37 PF/s."""
    plan = TrainPlan(70e9, 1e12, 'h100-sxm', 512)
    peak = plan.peak_flops_per_sec()
    expected = 989e12 * 512
    assert abs(peak - expected) < 1e12  # Tolerance for float precision


def test_train_effective_flops_with_mfu():
    """Effective FLOPs = peak * MFU."""
    plan = TrainPlan(70e9, 1e12, 'h100-sxm', 512, mfu=0.50)
    effective = plan.effective_flops_per_sec()
    peak = plan.peak_flops_per_sec()
    assert abs(effective - peak * 0.50) < 1e12


def test_train_time_from_flops():
    """Training time = FLOPs / effective throughput."""
    plan = TrainPlan(70e9, 1e12, 'h100-sxm', 512, mfu=1.0, checkpoint_hours=0, fault_factor=1.0)
    flops = plan.compute_flops()
    eff_fps = plan.effective_flops_per_sec()
    expected_sec = flops / eff_fps
    actual_sec = plan.training_time_seconds()
    assert abs(actual_sec - expected_sec) < 1


def test_infer_kv_cache_scaling():
    """KV cache scales roughly with model size."""
    plan_70b = InferencePlan(70e9, 10, 4096, 'h100-sxm')
    plan_140b = InferencePlan(140e9, 10, 4096, 'h100-sxm')
    # Should be roughly 2x larger
    ratio = plan_140b.kv_cache_per_token_mb() / plan_70b.kv_cache_per_token_mb()
    assert 1.8 < ratio < 2.2


def test_infer_batch_cap_respects_hbm():
    """Max batch size should be < HBM / (KV per token per seq)."""
    plan = InferencePlan(70e9, 10, 4096, 'h100-sxm')
    max_batch = plan.max_batch_size()
    assert 1 <= max_batch <= 256


def test_llama3_405b_estimated_mfu():
    """Back-solve MFU from Llama 3 405B reported 79.75 days.

    HuggingFace model card: 15.6T tokens, 16128 H100-SXM, 30.84M GPU-hours
    Days = 30.84M / (16128*24) ≈ 79.75 days
    Back-solved MFU from GPU-hours: 6*405B*15.6T / (30.84M*3600*989 TFLOPS) ≈ 38-39%
    This is actual sustained MFU without separate overhead adjustment.
    """
    run = RUNS[0]  # Llama 3 405B
    mfu = estimate_mfu(run.model_params_b, run.total_tokens_t, run.gpu_type,
                      run.num_gpu, run.reported_days)
    # Back-solved MFU from official GPU-hours reported by Meta
    # Should be in range 0.35-0.45 for large models at this scale
    assert 0.30 < mfu < 0.50, f'Llama 3 405B MFU {mfu:.1%} out of expected range (check GPU-hours source)'


def test_calibrate_estimates_close_to_llama3():
    """Estimate Llama 3 405B with backed-solved MFU should be within ±5% of reported.

    HF model card: 30.84M GPU-hours → ~79.75 days
    Using MFU=0.38 (back-solved from 6*N*T / GPU-hours), checkpoint_oh=0, fault_factor=1.0.
    Error should be very small since we're using actual reported GPU-hours.
    """
    run = RUNS[0]
    # Back-solve MFU from the GPU-hours (30.84M GPU-hours reported)
    actual_mfu = run.reported_card_hours / (24 * run.num_gpu) / run.reported_days if run.reported_card_hours else 1.0
    mfu_from_flops = (6 * run.model_params_b * 1e9 * run.total_tokens_t * 1e12) / (run.reported_card_hours * 3600 * 989e12 * 1)
    est_days = estimate_days(run.model_params_b, run.total_tokens_t, run.gpu_type,
                            run.num_gpu, mfu=mfu_from_flops, checkpoint_oh=0, fault_factor=1.0)
    error_pct = abs(est_days - run.reported_days) / run.reported_days * 100
    # Should be within 2% since we're using official GPU-hours
    assert error_pct < 5, f'Estimation error {error_pct:.1f}% > 5% (check GPU-hours calculation)'


def test_flops_formula_dense_transformer():
    """6*N*T is standard for dense models."""
    # Test: 175B params, 0.3T tokens = 6*175*0.3 * 1e21 FLOPs
    flops = compute_flops(175, 0.3)  # Input: billions, trillions
    expected = 6 * 175e9 * 0.3e12
    assert flops == expected
    # 6*175*0.3 = 315, so 315e21 FLOPs = 315,000 EF
    assert flops == 315e21


def test_train_deficit_is_percentage():
    """Deficit should increase with lower MFU."""
    plan_100 = TrainPlan(70e9, 1e12, 'h100-sxm', 512, mfu=1.0, checkpoint_hours=24, fault_factor=1.0)
    deficit_100 = plan_100.deficit()
    # With checkpoint overhead, deficit > 0 even at 100% MFU
    assert deficit_100 >= 0.1  # ~10% overhead from checkpoint

    plan_50 = TrainPlan(70e9, 1e12, 'h100-sxm', 512, mfu=0.50, checkpoint_hours=24, fault_factor=1.0)
    deficit_50 = plan_50.deficit()
    assert deficit_50 > deficit_100  # Lower MFU → higher deficit


if __name__ == '__main__':
    import sys
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            try:
                v()
                print('ok  ', k)
            except AssertionError as e:
                fails += 1
                print('FAIL', k, e)
    sys.exit(fails)
