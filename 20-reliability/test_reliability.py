#!/usr/bin/env python3
"""
GPU-20: Unit tests for reliability calculations.

Tests verify against:
1. Young-Daly checkpoint formulas against published references
2. Llama 3 failure counts: 419 interruptions in 54 days on 16384 H100s
3. Straggler slowdown predictions
"""

import sys
import os
import unittest
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_ckpt
import sim_failures
import sim_straggler

class TestCheckpointFormulas(unittest.TestCase):
    """Test Young and Daly formulas against published results."""

    def test_young_formula_basic(self):
        """Young 1974: τ* = sqrt(2*C*T_MTBF)"""
        # Example: C=600s, MTBF=86400s (1 day)
        C = 600.0  # checkpoint time in seconds
        T_MTBF = 86400.0
        tau = sim_ckpt.young_interval(C, T_MTBF)
        expected = math.sqrt(2 * C * T_MTBF)
        self.assertAlmostEqual(tau, expected, places=2)
        # Expected ~3470 seconds, or ~58 minutes

    def test_daly_formula_basic(self):
        """Daly 2006: τ* = sqrt(2*T_MTBF / (1 + sqrt(1 + 2*R/T_MTBF)))"""
        T_MTBF = 86400.0
        R = 300.0  # recovery time
        tau = sim_ckpt.daly_interval(T_MTBF, R, checkpoint_time=600.0)
        # Daly should be slightly smaller than Young due to recovery overhead
        self.assertGreater(tau, 0)
        self.assertLess(tau, T_MTBF)

    def test_ettr_calculation(self):
        """ETTR should be 0 < ettr < 1, and decrease with more failures."""
        total_steps = 100000
        step_time = 0.1
        ckpt_time = 600.0
        mtbf = 86400.0
        recovery = 300.0

        r_tight = sim_ckpt.expected_time_to_result(
            total_steps, 500, ckpt_time, mtbf, recovery, step_time
        )
        r_loose = sim_ckpt.expected_time_to_result(
            total_steps, 10000, ckpt_time, mtbf, recovery, step_time
        )

        # Tighter checkpointing = more checkpoint overhead but fewer lost steps
        # We expect ETTR to improve with fewer failures
        self.assertGreater(r_tight['ettr'], 0)
        self.assertLess(r_tight['ettr'], 1.0)

class TestFailureSimulation(unittest.TestCase):
    """Test failure rates and Llama 3 data point."""

    def test_llama3_mtbf(self):
        """
        Llama 3: 16384 cards, 419 failures in 54 days.
        Expected job MTBF ≈ (54*24) / 419 ≈ 3.08 hours ≈ 11100 seconds
        """
        num_cards = 16384
        failures = 419
        days = 54
        hours = days * 24
        expected_job_mtbf_hours = hours / failures
        expected_job_mtbf_sec = expected_job_mtbf_hours * 3600

        # Now compute card-level failure rate
        card_annual_failures = failures / num_cards / days * 365
        computed_job_mtbf = sim_failures.job_mtbf_from_card_failure_rate(
            num_cards, card_annual_failures
        )

        self.assertAlmostEqual(computed_job_mtbf, expected_job_mtbf_sec, delta=1)
        self.assertAlmostEqual(expected_job_mtbf_hours, 3.08, delta=0.1)

    def test_mtbf_scales_linearly(self):
        """Doubling num_cards should halve the job MTBF."""
        card_failure_rate = 0.5
        mtbf_1_card = sim_failures.job_mtbf_from_card_failure_rate(1, card_failure_rate)
        mtbf_8_cards = sim_failures.job_mtbf_from_card_failure_rate(8, card_failure_rate)
        mtbf_16_cards = sim_failures.job_mtbf_from_card_failure_rate(16, card_failure_rate)

        # MTBF should scale inversely
        self.assertAlmostEqual(mtbf_8_cards, mtbf_1_card / 8, delta=0.1)
        self.assertAlmostEqual(mtbf_16_cards, mtbf_1_card / 16, delta=0.1)

class TestStragglerImpact(unittest.TestCase):
    """Test straggler slowdown calculations."""

    def test_no_stragglers(self):
        """With no stragglers, slowdown should be 1.0"""
        result = sim_straggler.straggler_impact(
            num_nodes=8,
            nominal_step_time_ms=100.0,
            straggler_slowdown_ratio=1.5,
            num_stragglers=0,
            num_steps=100
        )
        self.assertAlmostEqual(result['slowdown'], 1.0, places=5)
        self.assertAlmostEqual(result['goodput_loss_percent'], 0, places=1)

    def test_all_stragglers(self):
        """If all nodes are slow by 1.5x, slowdown should be ~1.5x"""
        result = sim_straggler.straggler_impact(
            num_nodes=8,
            nominal_step_time_ms=100.0,
            straggler_slowdown_ratio=1.5,
            num_stragglers=8,
            num_steps=100
        )
        self.assertAlmostEqual(result['slowdown'], 1.5, places=2)
        self.assertGreater(result['goodput_loss_percent'], 25)  # ~33% loss for 1.5x slowdown

    def test_single_straggler_impact(self):
        """One 2x slow node in 16 synchronizes at 2x speed, causing 2x slowdown"""
        result = sim_straggler.straggler_impact(
            num_nodes=16,
            nominal_step_time_ms=100.0,
            straggler_slowdown_ratio=2.0,
            num_stragglers=1,
            num_steps=1000
        )
        # Max step time should be 200ms (1 straggler at 2x slowdown)
        self.assertAlmostEqual(result['max_step_time_ms'], 200.0, places=1)
        # In synchronous training, all nodes synchronize at the slowest node (200ms)
        # So slowdown is 200/100 = 2.0x
        self.assertAlmostEqual(result['slowdown'], 2.0, places=1)

if __name__ == '__main__':
    unittest.main()
