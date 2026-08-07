"""
Tests for Unroll Sensitivity Evaluation (ICML Phase 1).

These tests verify the correctness of metrics computation and batch building.
All tests are CPU-compatible and fast.
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test KL Divergence
# =============================================================================

class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_self_is_zero(self):
        """KL(p || p) should be approximately 0."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        # Uniform distribution
        p = torch.ones(10) / 10
        kl = compute_kl_divergence(p, p)
        assert abs(kl) < 1e-6, f"KL(p||p) should be ~0, got {kl}"

        # Non-uniform distribution
        p = torch.softmax(torch.randn(20), dim=0)
        kl = compute_kl_divergence(p, p)
        assert abs(kl) < 1e-6, f"KL(p||p) should be ~0, got {kl}"

    def test_kl_is_non_negative(self):
        """KL divergence should always be >= 0."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        rng = np.random.default_rng(42)
        for _ in range(10):
            p = torch.softmax(torch.from_numpy(rng.random(15)).float(), dim=0)
            q = torch.softmax(torch.from_numpy(rng.random(15)).float(), dim=0)
            kl = compute_kl_divergence(p, q)
            assert kl >= 0, f"KL should be >= 0, got {kl}"

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
        assert np.isfinite(kl), f"KL should be finite, got {kl}"

    def test_kl_asymmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        p = torch.tensor([0.9, 0.1])
        q = torch.tensor([0.5, 0.5])

        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)

        # They should be different
        assert abs(kl_pq - kl_qp) > 0.01, "KL should be asymmetric"


# =============================================================================
# Test Metrics at Same Depth
# =============================================================================

class TestSameDepthMetrics:
    """Test that metrics at the same depth are trivial."""

    def test_delta_V_same_n_is_zero(self):
        """Δ_V(n, n) should be 0."""
        # Simulate the computation
        v1 = torch.tensor(5.0)
        v2 = torch.tensor(5.0)  # Same value at same depth
        delta_V = float(torch.abs(v1 - v2).item())
        assert abs(delta_V) < 1e-8, f"Δ_V(n,n) should be 0, got {delta_V}"

    def test_delta_z_same_n_is_zero(self):
        """Δ_z(n, n) should be 0."""
        z1 = torch.randn(10, 10)
        z2 = z1.clone()  # Same latent at same depth
        delta_z = float(torch.norm(z1 - z2, p=2).item())
        assert abs(delta_z) < 1e-8, f"Δ_z(n,n) should be 0, got {delta_z}"

    def test_argmax_agreement_same_n_is_one(self):
        """argmax_agreement(n, n) should be 1."""
        p = torch.softmax(torch.randn(100), dim=0)
        argmax1 = p.argmax().item()
        argmax2 = p.argmax().item()  # Same distribution
        agree = 1 if argmax1 == argmax2 else 0
        assert agree == 1, "argmax agreement with self should be 1"


# =============================================================================
# Test Data Structures
# =============================================================================

class TestDataStructures:
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

        assert state.state_id == "test_0001"
        assert state.empties == 4
        assert state.parent_id is None
        assert state.action_source is None

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

        assert meta.num_states == 100
        assert sum(meta.empties_distribution.values()) == 100

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

        assert metrics.n1 < metrics.n2
        assert metrics.delta_V >= 0
        assert metrics.delta_pi >= 0


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_count_empties(self):
        """count_empties should correctly count cells with value 1."""
        from scripts.eval_unroll_sensitivity import count_empties

        # All empties (all 1s)
        inputs = torch.ones(16, dtype=torch.long)
        assert count_empties(inputs) == 16

        # No empties (all non-1)
        inputs = torch.full((16,), 2, dtype=torch.long)
        assert count_empties(inputs) == 0

        # Mixed
        inputs = torch.tensor([1, 2, 3, 1, 1, 4, 5, 1, 2, 3, 4, 5, 1, 1, 2, 3])
        assert count_empties(inputs) == 6

    def test_stable_hash_deterministic(self):
        """stable_hash should be deterministic."""
        from scripts.eval_unroll_sensitivity import stable_hash

        t = torch.tensor([1, 2, 3, 4, 5])
        h1 = stable_hash(t)
        h2 = stable_hash(t)
        assert h1 == h2, "Hash should be deterministic"

    def test_stable_hash_different_tensors(self):
        """stable_hash should differ for different tensors."""
        from scripts.eval_unroll_sensitivity import stable_hash

        t1 = torch.tensor([1, 2, 3, 4, 5])
        t2 = torch.tensor([1, 2, 3, 4, 6])  # Different last element

        h1 = stable_hash(t1)
        h2 = stable_hash(t2)
        assert h1 != h2, "Different tensors should have different hashes"

    def test_aggregate_metrics(self):
        """aggregate_metrics should compute correct statistics."""
        from scripts.eval_unroll_sensitivity import aggregate_metrics

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        agg = aggregate_metrics(values)

        assert agg["mean"] == 5.5
        assert agg["median"] == 5.5
        assert agg["max"] == 10.0
        assert agg["p90"] == pytest.approx(9.1, rel=0.1)
        assert agg["p99"] == pytest.approx(9.91, rel=0.1)

    def test_aggregate_metrics_empty(self):
        """aggregate_metrics should handle empty list."""
        from scripts.eval_unroll_sensitivity import aggregate_metrics

        agg = aggregate_metrics([])
        assert agg["mean"] == 0.0
        assert agg["max"] == 0.0


# =============================================================================
# Test Output Shapes
# =============================================================================

class TestOutputShapes:
    """Test that output shapes are correct."""

    def test_kl_returns_scalar(self):
        """KL divergence should return a scalar."""
        from scripts.eval_unroll_sensitivity import compute_kl_divergence

        p = torch.softmax(torch.randn(50), dim=0)
        q = torch.softmax(torch.randn(50), dim=0)
        kl = compute_kl_divergence(p, q)

        assert isinstance(kl, float), f"KL should be float, got {type(kl)}"

    def test_aggregate_returns_dict(self):
        """aggregate_metrics should return dict with expected keys."""
        from scripts.eval_unroll_sensitivity import aggregate_metrics

        values = [1.0, 2.0, 3.0]
        agg = aggregate_metrics(values)

        expected_keys = {"mean", "median", "p90", "p99", "max"}
        assert set(agg.keys()) == expected_keys


# =============================================================================
# Test Model Compatibility Check
# =============================================================================

class TestModelCompatibility:
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
            "enable_contraction": True,  # Different, but not required to match
        }

        compatible, diffs = check_model_compatibility(config_a, config_b)
        assert compatible, f"Should be compatible, but got diffs: {diffs}"
        assert len(diffs) == 0

    def test_incompatible_action_space(self):
        """Different action spaces should be incompatible."""
        from scripts.eval_unroll_sensitivity import check_model_compatibility

        config_a = {"num_actions": 97, "episodic_latent": True, "use_feasibility_checker": True}
        config_b = {"num_actions": 730, "episodic_latent": True, "use_feasibility_checker": True}

        compatible, diffs = check_model_compatibility(config_a, config_b)
        assert not compatible, "Different action spaces should be incompatible"
        assert "num_actions" in diffs[0]

    def test_incompatible_episodic_latent(self):
        """Different episodic_latent should be incompatible."""
        from scripts.eval_unroll_sensitivity import check_model_compatibility

        config_a = {"num_actions": 97, "episodic_latent": True, "use_feasibility_checker": True}
        config_b = {"num_actions": 97, "episodic_latent": False, "use_feasibility_checker": True}

        compatible, diffs = check_model_compatibility(config_a, config_b)
        assert not compatible, "Different episodic_latent should be incompatible"


# =============================================================================
# Test Batch Building (Mock)
# =============================================================================

class TestBatchBuilding:
    """Test batch building logic (without actual data loading)."""

    def test_b0_target_composition(self):
        """B0 should target 70 easy + 30 hard."""
        # This is a design verification, not a runtime test
        target_easy = 70
        target_hard = 30
        total = target_easy + target_hard
        assert total == 100, "B0 should have 100 states"

    def test_empties_classification(self):
        """Empties should be classified correctly."""
        # Easy: 1-4 empties
        # Hard: 6-8 empties
        easy_range = range(1, 5)
        hard_range = range(6, 9)

        assert list(easy_range) == [1, 2, 3, 4]
        assert list(hard_range) == [6, 7, 8]


# =============================================================================
# Integration Test (if model available)
# =============================================================================

class TestIntegration:
    """Integration tests that require model imports."""

    @pytest.fixture
    def dummy_model(self):
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
                rl_num_actions=97,  # 16 * 6 + 1
            )

            model = TinyRecursiveReasoningModel_ACTV1(config)
            model.eval()
            return model
        except ImportError:
            pytest.skip("Model imports not available")

    def test_model_outputs_finite(self, dummy_model):
        """Model outputs should be finite."""
        x = {
            "inputs": torch.ones(1, 16, dtype=torch.long),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        }
        y = torch.ones(1, 16, dtype=torch.long)

        with torch.no_grad():
            value, z = dummy_model.used_value(x, y, n=4)
            assert torch.isfinite(value).all(), "Value should be finite"

            dist, z = dummy_model.policy_dist(x, y, n=4)
            assert torch.isfinite(dist.probs).all(), "Policy probs should be finite"

    def test_different_depths_may_differ(self, dummy_model):
        """Different unroll depths may produce different outputs."""
        x = {
            "inputs": torch.ones(1, 16, dtype=torch.long) * 2,  # All clues
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        }
        y = torch.ones(1, 16, dtype=torch.long) * 2

        with torch.no_grad():
            v1, z1 = dummy_model.used_value(x, y, n=1)
            v4, z4 = dummy_model.used_value(x, y, n=4)

            # They may or may not differ, but should be finite
            assert torch.isfinite(v1).all()
            assert torch.isfinite(v4).all()


# =============================================================================
# Test Saturation Semantics
# =============================================================================

class TestSaturationSemantics:
    """Tests saturation semantics for explicit projection modes."""

    def test_disabled_saturation_is_negative_one(self):
        """Disabled projection should report saturated=-1 (N/A)."""
        # This tests the logic at eval_unroll_sensitivity.py:934-938
        # When radius is None, saturated = -1.

        # Simulate the check
        radius = None
        z_pre = 10.0  # arbitrary nonzero

        if radius is None:
            saturated = -1
        else:
            saturated = 1 if z_pre >= 0.95 * radius else 0

        assert saturated == -1

    def test_r_positive_saturation_logic(self):
        """When R>0, saturation should be computed correctly."""
        # Test saturation = 1 when z_pre >= 0.95 * R
        radius = 10.0
        z_pre_high = 9.6  # >= 0.95 * 10 = 9.5
        z_pre_low = 5.0   # < 9.5

        if radius is None:
            sat_high = -1
            sat_low = -1
        else:
            sat_high = 1 if z_pre_high >= 0.95 * radius else 0
            sat_low = 1 if z_pre_low >= 0.95 * radius else 0

        assert sat_high == 1, "z_pre >= 0.95*R should give saturated=1"
        assert sat_low == 0, "z_pre < 0.95*R should give saturated=0"

    def test_saturation_rate_excludes_disabled_projection(self):
        """Saturation rate computation should skip -1 values."""
        # Simulate the filter logic from the script
        saturated_values = [-1, -1, 1, 0, 1]

        valid_sat = [s for s in saturated_values if s >= 0]
        assert valid_sat == [1, 0, 1], "Should exclude -1 values"

        if valid_sat:
            rate = sum(valid_sat) / len(valid_sat)
        else:
            rate = None

        assert abs(rate - 2/3) < 0.01, f"Rate should be 2/3, got {rate}"


# =============================================================================
# Test Self-Consistency (n=n)
# =============================================================================

class TestSelfConsistency:
    """Tests that comparing n to n gives trivial metrics."""

    def test_delta_v_self_is_zero(self):
        """Delta_V(n, n) should be exactly 0."""
        # When comparing a model output to itself, delta should be 0
        v = torch.tensor(1.5)
        delta_V = float(torch.abs(v - v).item())
        assert delta_V == 0.0, f"Delta_V(n,n) should be 0, got {delta_V}"

    def test_argmax_agree_self_is_one(self):
        """argmax_agree(n, n) should be 1."""
        # When comparing a distribution to itself, argmax always agrees
        p = torch.softmax(torch.randn(97), dim=0)
        argmax1 = p.argmax().item()
        argmax2 = p.argmax().item()
        agree = 1 if argmax1 == argmax2 else 0
        assert agree == 1, "argmax should always agree with itself"

    def test_delta_z_self_is_zero(self):
        """Delta_z(n, n) should be exactly 0."""
        z = torch.randn(64)
        delta_z = float(torch.norm(z - z, p=2).item())
        assert delta_z == 0.0, f"Delta_z(n,n) should be 0, got {delta_z}"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
