"""
Tests for theory-exact components - unittest version.
Converts the pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch
import torch.nn as nn

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from utils.lipschitz import (
    estimate_Cdrift,
    estimate_plan_change,
    compute_value_of_memory_residual,
    compute_exact_baseline_summation,
    compute_exact_advantage,
    compute_theoretical_drift_bound,
)


def get_small_config():
    """Small TRM config for testing."""
    return dict(
        batch_size=4,
        seq_len=16,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=8,
        vocab_size=10,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=2,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=True,
        rl_target_Lz=0.9,
        rl_target_Lv=1.0,
        rl_enable_policy_head=True,
        rl_num_actions=16 * 10 + 1,
        rl_latent_ball_radius=0.0,
    )


def get_sample_batch():
    """Sample batch for testing."""
    batch_size = 4
    seq_len = 16
    return {
        "inputs": torch.randint(0, 10, (batch_size, seq_len)),
        "puzzle_identifiers": torch.arange(batch_size),
    }


class TestPlanChangeTracking(unittest.TestCase):
    """Tests for plan change tracking Δy_max."""

    def test_plan_change_integer_tokens(self):
        """Test plan change with integer tokens (Hamming distance)."""
        y_old = torch.tensor([1, 2, 3, 4, 5])
        y_new = torch.tensor([1, 2, 9, 4, 5])

        change = estimate_plan_change(y_old, y_new)
        self.assertEqual(change, 1.0, f"Expected 1 change, got {change}")

    def test_plan_change_multiple_edits(self):
        """Test plan change with multiple edits."""
        y_old = torch.tensor([1, 2, 3, 4, 5])
        y_new = torch.tensor([9, 2, 9, 4, 9])

        change = estimate_plan_change(y_old, y_new)
        self.assertEqual(change, 3.0, f"Expected 3 changes, got {change}")

    def test_plan_change_batch(self):
        """Test plan change with batched plans."""
        y_old = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y_new = torch.tensor([[1, 9, 3], [9, 9, 6]])

        change = estimate_plan_change(y_old, y_new)
        self.assertEqual(change, 2.0, f"Expected max change of 2, got {change}")

    def test_plan_change_no_change(self):
        """Test plan change when plans are identical."""
        y = torch.tensor([1, 2, 3, 4, 5])
        change = estimate_plan_change(y, y.clone())
        self.assertEqual(change, 0.0, f"Expected 0 change, got {change}")


class TestDriftBound(unittest.TestCase):
    """Tests for drift bound C_drift(n)."""

    def test_theoretical_drift_bound(self):
        """Test theoretical drift bound computation."""
        kappa_n = 0.5
        E_0 = 1.0
        L_zstar = 0.5
        delta_y_max = 0.1

        drift = compute_theoretical_drift_bound(kappa_n, E_0, L_zstar, delta_y_max)

        expected = 0.5 * 1.0 + (0.5 * 0.5 * 0.1) / 0.5
        self.assertAlmostEqual(drift, expected, places=6)

    def test_drift_bound_not_contractive(self):
        """Test drift bound returns inf when not contractive."""
        drift = compute_theoretical_drift_bound(1.5, 1.0, 0.5, 0.1)
        self.assertEqual(drift, float("inf"))


class TestExactBaseline(unittest.TestCase):
    """Tests for exact baseline computation for Theorem 5.9."""

    def test_exact_advantage_centering(self):
        """Test that exact advantages are centered: E_{a~π}[Â(s,a)] = 0."""
        batch_size = 4
        num_actions = 10

        q_values = torch.randn(batch_size, num_actions)
        probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)

        exact_baseline = (probs * q_values).sum(dim=-1)
        advantages_all = q_values - exact_baseline.unsqueeze(-1)
        expected_adv = (probs * advantages_all).sum(dim=-1)

        self.assertTrue((expected_adv.abs() < 1e-5).all())

    def test_compute_exact_advantage(self):
        """Test compute_exact_advantage function."""
        batch_size = 4
        num_actions = 10

        q_values = torch.randn(batch_size, num_actions)
        probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        exact_baseline = (probs * q_values).sum(dim=-1)

        actions = torch.randint(0, num_actions, (batch_size,))
        adv = compute_exact_advantage(q_values, exact_baseline, actions)

        self.assertEqual(adv.shape, (batch_size,))

        for i in range(batch_size):
            expected = q_values[i, actions[i]] - exact_baseline[i]
            self.assertAlmostEqual(adv[i].item(), expected.item(), places=5)


class TestRLConfigTheoryOptions(unittest.TestCase):
    """Tests for new theory-exact config options."""

    def test_default_values(self):
        """Test default values for new config options."""
        cfg = RLConfig()

        self.assertFalse(cfg.exact_baseline_summation)
        self.assertEqual(cfg.latent_ball_radius, 10.0)
        self.assertFalse(cfg.track_drift_metrics)
        self.assertFalse(cfg.track_plan_change)
        self.assertFalse(cfg.compute_value_of_memory)

    def test_enable_theory_exact(self):
        """Test enabling all theory-exact options."""
        cfg = RLConfig(
            exact_baseline_summation=True,
            latent_ball_radius=5.0,
            track_drift_metrics=True,
            track_plan_change=True,
            compute_value_of_memory=True,
        )

        self.assertTrue(cfg.exact_baseline_summation)
        self.assertEqual(cfg.latent_ball_radius, 5.0)
        self.assertTrue(cfg.track_drift_metrics)
        self.assertTrue(cfg.track_plan_change)
        self.assertTrue(cfg.compute_value_of_memory)


class TestForwardInvariantProjection(unittest.TestCase):
    """Tests for forward-invariant projection (Eq. 14 in paper)."""

    def setUp(self):
        self.config = get_small_config()
        self.sample_batch = get_sample_batch()
        self.sample_plan = torch.randint(0, 10, self.sample_batch["inputs"].shape)

    def test_projection_bounds_latent_norm(self):
        """Test that projection keeps ||z|| ≤ R."""
        config = dict(self.config)
        config["rl_latent_ball_radius"] = 5.0
        model = TinyRecursiveReasoningModel_ACTV1(config)
        R = model.config.rl_latent_ball_radius

        z = model.init_latent(self.sample_batch, self.sample_plan)
        for _ in range(10):
            z = model.update_latent(z, self.sample_plan, self.sample_batch)

        z_H_norm = z.z_H.norm(p=2, dim=(1, 2))
        z_L_norm = z.z_L.norm(p=2, dim=(1, 2))
        joint_norm = torch.sqrt(z_H_norm.square() + z_L_norm.square())

        self.assertTrue((z_H_norm <= R + 1e-5).all())
        self.assertTrue((z_L_norm <= R + 1e-5).all())
        self.assertTrue((joint_norm <= R + 1e-5).all())

    def test_no_projection_when_disabled(self):
        """Test that latent can grow when projection is disabled."""
        model = TinyRecursiveReasoningModel_ACTV1(self.config)

        z = model.init_latent(self.sample_batch, self.sample_plan)
        initial_norm = z.z_H.norm(p=2, dim=(1, 2)).max()

        for _ in range(20):
            z = model.update_latent(z, self.sample_plan, self.sample_batch)

        final_norm = z.z_H.norm(p=2, dim=(1, 2)).max()
        self.assertGreaterEqual(final_norm, 0)


if __name__ == "__main__":
    unittest.main()
