"""
Tests for theory metrics computation - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.lipschitz import (
    estimate_Cz,
    estimate_Lv,
    estimate_local_Lz,
    compute_unrolling_term_proxy,
)
from rl.upi_trm_trainer import compute_empirical_bellman_residual


def _tiny_trm_cfg(batch_size: int = 2, seq_len: int = 4, vocab_size: int = 16):
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=4,
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=4,
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
        rl_enable_contraction=False,
    )


def _dummy_batch(batch_size: int, seq_len: int, vocab_size: int, num_identifiers: int = 4):
    return {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, num_identifiers, (batch_size,)),
    }


class TestTheoryMetrics(unittest.TestCase):
    """Tests for theory metrics computation."""

    def test_estimate_Cz_returns_positive_value(self):
        """Test that C_z estimation returns a positive finite value."""
        torch.manual_seed(0)
        cfg = _tiny_trm_cfg()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        hat_Cz = estimate_Cz(model, x_batch, y_batch)

        self.assertIsInstance(hat_Cz, float)
        self.assertGreaterEqual(hat_Cz, 0.0)
        self.assertLess(hat_Cz, float("inf"))

    def test_estimate_local_Lz_returns_reasonable_value(self):
        """Test that local L_z estimation returns a reasonable value."""
        torch.manual_seed(1)
        cfg = _tiny_trm_cfg()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        z_n = model.eval_latent(x_batch, y_batch, n=2)
        batch = model._standardize_latent_batch(x_batch, y_batch)
        context = model._build_latent_context_with_plan(batch)

        hat_Lz = estimate_local_Lz(model.inner, z_n, context, num_samples=4)

        self.assertIsInstance(hat_Lz, float)
        self.assertGreaterEqual(hat_Lz, 0.0)
        # Without contraction, L_z can be >= 1, but should be finite
        self.assertLess(hat_Lz, 100.0)

    def test_estimate_Lv_returns_positive_value(self):
        """Test that L_v estimation returns a positive finite value."""
        torch.manual_seed(2)
        cfg = _tiny_trm_cfg()
        model = TinyRecursiveReasoningModel_ACTV1(cfg)
        model.eval()

        x_batch = _dummy_batch(cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
        y_batch = torch.zeros(cfg["batch_size"], cfg["seq_len"], dtype=torch.long)

        z_n = model.eval_latent(x_batch, y_batch, n=2)
        # Use flatten (view) to match model.used_value() behavior, NOT mean
        z_vec = z_n.z_H.view(z_n.z_H.shape[0], -1)  # [B, seq_len * hidden_size]

        batch = model._standardize_latent_batch(x_batch, y_batch)
        context = model._build_latent_context_with_plan(batch)
        input_embeddings = context["input_embeddings"]
        plan_embeddings = context["plan_embeddings"]
        # Use flatten (view) to match model.used_value() behavior, NOT pool
        x_embed = input_embeddings.view(input_embeddings.shape[0], -1)
        y_embed = plan_embeddings.view(plan_embeddings.shape[0], -1)
        combined_embed = torch.cat([x_embed, y_embed], dim=-1)

        hat_Lv = estimate_Lv(model.value_head, z_vec, combined_embed, num_samples=4)

        self.assertIsInstance(hat_Lv, float)
        self.assertGreaterEqual(hat_Lv, 0.0)
        self.assertLess(hat_Lv, float("inf"))

    def test_compute_unrolling_term_proxy(self):
        """Test unrolling term proxy computation."""
        # With L_z < 1, should get finite result
        result = compute_unrolling_term_proxy(hat_Lv=1.0, hat_Lz=0.5, hat_Cz=1.0, n=4)
        expected = 1.0 * (0.5 ** 4) * 1.0 / (1.0 - 0.5)
        self.assertLess(abs(result - expected), 1e-6)

        # With L_z >= 1, should get infinity
        result_inf = compute_unrolling_term_proxy(hat_Lv=1.0, hat_Lz=1.0, hat_Cz=1.0, n=4)
        self.assertEqual(result_inf, float("inf"))

    def test_compute_empirical_bellman_residual(self):
        """Test Bellman residual computation."""
        torch.manual_seed(3)
        batch_size = 8
        values = torch.randn(batch_size)
        rewards = torch.randn(batch_size)
        next_values = torch.randn(batch_size)
        dones = torch.zeros(batch_size, dtype=torch.bool)
        dones[0] = True  # First sample is terminal
        gamma = 0.99

        metrics = compute_empirical_bellman_residual(values, rewards, next_values, dones, gamma)

        self.assertIn("bellman_residual_mean", metrics)
        self.assertIn("bellman_residual_max", metrics)
        self.assertIn("bellman_residual_std", metrics)
        self.assertGreaterEqual(metrics["bellman_residual_mean"], 0.0)
        self.assertGreaterEqual(metrics["bellman_residual_max"], metrics["bellman_residual_mean"])

    def test_unrolling_term_decays_with_n(self):
        """Test that unrolling term decreases as n increases (for L_z < 1)."""
        hat_Lv = 2.0
        hat_Lz = 0.8
        hat_Cz = 1.5

        terms = [compute_unrolling_term_proxy(hat_Lv, hat_Lz, hat_Cz, n) for n in [1, 2, 4, 8]]

        # Should be monotonically decreasing
        for i in range(len(terms) - 1):
            self.assertGreater(terms[i], terms[i + 1], f"Expected {terms[i]} > {terms[i+1]}")


if __name__ == "__main__":
    unittest.main()
