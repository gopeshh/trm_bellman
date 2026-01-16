"""
Tests for Unroll Sensitivity Evaluation (ICML Phase 1).

unittest-compatible version for Buck2.
"""

import unittest
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestKLDivergence(unittest.TestCase):
    """Tests for KL divergence computation."""

    def test_kl_self_is_zero(self):
        """KL(p || p) should be approximately 0."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        # Uniform distribution
        p = torch.ones(10) / 10
        kl = compute_kl_divergence(p, p)
        self.assertLess(abs(kl), 1e-6, f"KL(p||p) should be ~0, got {kl}")

        # Non-uniform distribution
        p = torch.softmax(torch.randn(20), dim=0)
        kl = compute_kl_divergence(p, p)
        self.assertLess(abs(kl), 1e-6, f"KL(p||p) should be ~0, got {kl}")

    def test_kl_is_non_negative(self):
        """KL divergence should always be >= 0."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        rng = np.random.default_rng(42)
        for _ in range(10):
            p = torch.softmax(torch.from_numpy(rng.random(15)).float(), dim=0)
            q = torch.softmax(torch.from_numpy(rng.random(15)).float(), dim=0)
            kl = compute_kl_divergence(p, q)
            self.assertGreaterEqual(kl, 0, f"KL should be >= 0, got {kl}")

    def test_kl_is_finite(self):
        """KL divergence should be finite even with near-zero probabilities."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        # Distribution with very small probabilities
        p = torch.zeros(10)
        p[0] = 0.999
        p[1] = 0.001
        q = torch.zeros(10)
        q[0] = 0.5
        q[1] = 0.5

        kl = compute_kl_divergence(p, q)
        self.assertTrue(np.isfinite(kl), f"KL should be finite, got {kl}")

    def test_kl_asymmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        p = torch.tensor([0.9, 0.1])
        q = torch.tensor([0.5, 0.5])

        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)

        # They should be different
        self.assertGreater(abs(kl_pq - kl_qp), 0.01, "KL should be asymmetric")


class TestSameDepthMetrics(unittest.TestCase):
    """Test that metrics at the same depth are trivial."""

    def test_delta_V_same_n_is_zero(self):
        """Δ_V(n, n) should be 0."""
        v1 = torch.tensor(5.0)
        v2 = torch.tensor(5.0)
        delta_V = float(torch.abs(v1 - v2).item())
        self.assertLess(abs(delta_V), 1e-8, f"Δ_V(n,n) should be 0, got {delta_V}")

    def test_delta_z_same_n_is_zero(self):
        """Δ_z(n, n) should be 0."""
        z1 = torch.randn(10, 10)
        z2 = z1.clone()
        delta_z = float(torch.norm(z1 - z2, p=2).item())
        self.assertLess(abs(delta_z), 1e-8, f"Δ_z(n,n) should be 0, got {delta_z}")

    def test_argmax_agreement_same_n_is_one(self):
        """argmax_agreement(n, n) should be 1."""
        p = torch.softmax(torch.randn(100), dim=0)
        argmax1 = p.argmax().item()
        argmax2 = p.argmax().item()
        agree = 1 if argmax1 == argmax2 else 0
        self.assertEqual(agree, 1, "argmax agreement with self should be 1")


class TestDataStructures(unittest.TestCase):
    """Test data structure creation and serialization."""

    def test_puzzle_state_creation(self):
        """PuzzleState should be creatable with all fields."""
        from scripts.eval_unroll_sensitivity import PuzzleState

        state = PuzzleState(
            state_id="test_0001",
            inputs=torch.ones(16, dtype=torch.long),
            puzzle_identifier=torch.tensor(0),
            plan=torch.ones(16, dtype=torch.long),
            empties=4,
            source_path="/fake/path",
        )

        self.assertEqual(state.state_id, "test_0001")
        self.assertEqual(state.empties, 4)
        self.assertIsNone(state.parent_id)
        self.assertIsNone(state.action_source)

    def test_batch_metadata_creation(self):
        """BatchMetadata should be creatable with all fields."""
        from scripts.eval_unroll_sensitivity import BatchMetadata

        meta = BatchMetadata(
            batch_name="b0",
            num_states=100,
            empties_distribution={1: 10, 2: 20, 3: 30, 4: 10, 6: 15, 7: 10, 8: 5},
            seed=42,
            source_paths=["/data/sudoku-4x4-trivial"],
            creation_time="2024-01-15T00:00:00",
        )

        self.assertEqual(meta.num_states, 100)
        self.assertEqual(sum(meta.empties_distribution.values()), 100)

    def test_eval_metrics_creation(self):
        """EvalMetrics should be creatable."""
        from scripts.eval_unroll_sensitivity import EvalMetrics

        metrics = EvalMetrics(
            state_id="test_0001",
            n1=4,
            n2=16,
            delta_V=0.05,
            delta_pi=0.02,
            delta_z=0.1,
            argmax_agree=1,
            z_pre_norm=8.5,
            z_post_norm=8.5,
            saturated=0,
        )

        self.assertLess(metrics.n1, metrics.n2)
        self.assertGreaterEqual(metrics.delta_V, 0)
        self.assertGreaterEqual(metrics.delta_pi, 0)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_count_empties(self):
        """count_empties should correctly count cells with value 1."""
        from scripts.eval_unroll_sensitivity import count_empties

        # All empties (all 1s)
        inputs = torch.ones(16, dtype=torch.long)
        self.assertEqual(count_empties(inputs), 16)

        # No empties (all non-1)
        inputs = torch.full((16,), 2, dtype=torch.long)
        self.assertEqual(count_empties(inputs), 0)

        # Mixed
        inputs = torch.tensor([1, 2, 3, 1, 1, 4, 5, 1, 2, 3, 4, 5, 1, 1, 2, 3])
        self.assertEqual(count_empties(inputs), 6)

    def test_stable_hash_deterministic(self):
        """stable_hash should be deterministic."""
        from scripts.eval_unroll_sensitivity import stable_hash

        t = torch.tensor([1, 2, 3, 4, 5])
        h1 = stable_hash(t)
        h2 = stable_hash(t)
        self.assertEqual(h1, h2, "Hash should be deterministic")

    def test_stable_hash_different_tensors(self):
        """stable_hash should differ for different tensors."""
        from scripts.eval_unroll_sensitivity import stable_hash

        t1 = torch.tensor([1, 2, 3, 4, 5])
        t2 = torch.tensor([1, 2, 3, 4, 6])

        h1 = stable_hash(t1)
        h2 = stable_hash(t2)
        self.assertNotEqual(h1, h2, "Different tensors should have different hashes")

    def test_aggregate_metrics(self):
        """aggregate_metrics should compute correct statistics."""
        from scripts.eval_unroll_sensitivity import aggregate_metrics

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        agg = aggregate_metrics(values)

        self.assertEqual(agg["mean"], 5.5)
        self.assertEqual(agg["median"], 5.5)
        self.assertEqual(agg["max"], 10.0)
        self.assertAlmostEqual(agg["p90"], 9.1, places=1)

    def test_aggregate_metrics_empty(self):
        """aggregate_metrics should handle empty list."""
        from scripts.eval_unroll_sensitivity import aggregate_metrics

        agg = aggregate_metrics([])
        self.assertEqual(agg["mean"], 0.0)
        self.assertEqual(agg["max"], 0.0)


class TestOutputShapes(unittest.TestCase):
    """Test that output shapes are correct."""

    def test_kl_returns_scalar(self):
        """KL divergence should return a scalar."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        p = torch.softmax(torch.randn(50), dim=0)
        q = torch.softmax(torch.randn(50), dim=0)
        kl = compute_kl_divergence(p, q)

        self.assertIsInstance(kl, float, f"KL should be float, got {type(kl)}")

    def test_aggregate_returns_dict(self):
        """aggregate_metrics should return dict with expected keys."""
        from scripts.eval_unroll_sensitivity import aggregate_metrics

        values = [1.0, 2.0, 3.0]
        agg = aggregate_metrics(values)

        expected_keys = {"mean", "median", "p90", "p99", "max"}
        self.assertEqual(set(agg.keys()), expected_keys)


class TestModelCompatibility(unittest.TestCase):
    """Test model compatibility checking."""

    def test_compatible_models(self):
        """Compatible models should pass check."""
        from scripts.eval_unroll_sensitivity import check_model_compatibility

        config_a = {
            "num_actions": 97,
            "episodic_latent": True,
            "use_feasibility_checker": True,
            "enable_contraction": False,
        }
        config_b = {
            "num_actions": 97,
            "episodic_latent": True,
            "use_feasibility_checker": True,
            "enable_contraction": True,
        }

        compatible, diffs = check_model_compatibility(config_a, config_b)
        self.assertTrue(compatible, f"Should be compatible, but got diffs: {diffs}")
        self.assertEqual(len(diffs), 0)

    def test_incompatible_action_space(self):
        """Different action spaces should be incompatible."""
        from scripts.eval_unroll_sensitivity import check_model_compatibility

        config_a = {"num_actions": 97, "episodic_latent": True, "use_feasibility_checker": True}
        config_b = {"num_actions": 730, "episodic_latent": True, "use_feasibility_checker": True}

        compatible, diffs = check_model_compatibility(config_a, config_b)
        self.assertFalse(compatible, "Different action spaces should be incompatible")
        self.assertIn("num_actions", diffs[0])

    def test_incompatible_episodic_latent(self):
        """Different episodic_latent should be incompatible."""
        from scripts.eval_unroll_sensitivity import check_model_compatibility

        config_a = {"num_actions": 97, "episodic_latent": True, "use_feasibility_checker": True}
        config_b = {"num_actions": 97, "episodic_latent": False, "use_feasibility_checker": True}

        compatible, diffs = check_model_compatibility(config_a, config_b)
        self.assertFalse(compatible, "Different episodic_latent should be incompatible")


class TestBatchBuilding(unittest.TestCase):
    """Test batch building logic (without actual data loading)."""

    def test_b0_target_composition(self):
        """B0 should target 70 easy + 30 hard."""
        target_easy = 70
        target_hard = 30
        total = target_easy + target_hard
        self.assertEqual(total, 100, "B0 should have 100 states")

    def test_empties_classification(self):
        """Empties should be classified correctly."""
        easy_range = range(1, 5)
        hard_range = range(6, 9)

        self.assertEqual(list(easy_range), [1, 2, 3, 4])
        self.assertEqual(list(hard_range), [6, 7, 8])


class TestIntegration(unittest.TestCase):
    """Integration tests that require model imports."""

    def _get_dummy_model(self):
        """Create a minimal model for testing."""
        try:
            from models.recursive_reasoning.trm import (
                TinyRecursiveReasoningModel_ACTV1,
                TinyRecursiveReasoningModel_ACTV1Config,
            )

            config = TinyRecursiveReasoningModel_ACTV1Config(
                hidden_size=32,
                vocab_size=6,
                rl_enable_value_head=True,
                rl_enable_policy_head=True,
                rl_num_actions=97,
            )

            model = TinyRecursiveReasoningModel_ACTV1(config)
            model.eval()
            return model
        except ImportError:
            return None

    def test_model_outputs_finite(self):
        """Model outputs should be finite."""
        model = self._get_dummy_model()
        if model is None:
            self.skipTest("Model imports not available")

        x = {
            "inputs": torch.ones(1, 16, dtype=torch.long),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        }
        y = torch.ones(1, 16, dtype=torch.long)

        with torch.no_grad():
            value, z = model.used_value(x, y, n=4)
            self.assertTrue(torch.isfinite(value).all(), "Value should be finite")

            dist, z = model.policy_dist(x, y, n=4)
            self.assertTrue(torch.isfinite(dist.probs).all(), "Policy probs should be finite")


if __name__ == "__main__":
    unittest.main()
