"""
Tests for theory-exact components that close the gap between paper and implementation.

These tests verify:
1. Forward-invariant projection
2. Exact baseline computation (Theorem 6.7)
3. Persistent slow-drift bound monitoring
4. Value-of-memory diagnostics
5. Plan-change tracking used by slow-drift premises
"""

import pytest
import torch
import torch.nn as nn

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.upi_trm_trainer import UPITrmTrainer
from utils.lipschitz import (
    estimate_Cdrift,
    estimate_plan_change,
    compute_value_of_memory_residual,
    compute_exact_baseline_summation,
    compute_exact_advantage,
    compute_theoretical_drift_bound,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_config():
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
        rl_num_actions=16 * 10 + 1,  # seq_len * vocab_size + STOP
        rl_latent_projection_mode="disabled",
        rl_latent_ball_radius=None,
    )


@pytest.fixture
def model_with_projection(small_config):
    """TRM with forward-invariant projection enabled."""
    config = dict(small_config)
    config["rl_latent_projection_mode"] = "enabled"
    config["rl_latent_ball_radius"] = 5.0  # Enable projection
    return TinyRecursiveReasoningModel_ACTV1(config)


@pytest.fixture
def model_without_projection(small_config):
    """TRM without forward-invariant projection."""
    return TinyRecursiveReasoningModel_ACTV1(small_config)


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    batch_size = 4
    seq_len = 16
    return {
        "inputs": torch.randint(0, 10, (batch_size, seq_len)),
        "puzzle_identifiers": torch.arange(batch_size),
    }


@pytest.fixture
def sample_plan(sample_batch):
    """Sample plan tensor."""
    return torch.randint(0, 10, sample_batch["inputs"].shape)


# =============================================================================
# Test: Forward-Invariant Projection
# =============================================================================

class TestForwardInvariantProjection:
    """Tests for forward-invariant projection (Eq. 14 in paper)."""
    
    def test_projection_bounds_latent_norm(self, model_with_projection, sample_batch, sample_plan):
        """Test that projection keeps ||z|| ≤ R."""
        model = model_with_projection
        R = model.config.rl_latent_ball_radius
        
        # Run multiple latent steps
        z = model.init_latent(sample_batch, sample_plan)
        for _ in range(10):
            z = model.update_latent(z, sample_plan, sample_batch)
        
        # Check norm bounds
        z_H_norm = z.z_H.norm(p=2, dim=(1, 2))  # [B]
        z_L_norm = z.z_L.norm(p=2, dim=(1, 2))  # [B]
        joint_norm = torch.sqrt(z_H_norm.square() + z_L_norm.square())
        
        assert (z_H_norm <= R + 1e-5).all(), f"z_H norm {z_H_norm.max()} exceeds R={R}"
        assert (z_L_norm <= R + 1e-5).all(), f"z_L norm {z_L_norm.max()} exceeds R={R}"
        assert (joint_norm <= R + 1e-5).all(), f"joint norm {joint_norm.max()} exceeds R={R}"
    
    def test_projection_is_1_lipschitz(self, model_with_projection, sample_batch, sample_plan):
        """Test that projection doesn't increase distance between latents."""
        model = model_with_projection
        
        # Create two slightly different latents
        z1 = model.init_latent(sample_batch, sample_plan)
        z2_H = z1.z_H + 0.1 * torch.randn_like(z1.z_H)
        z2_L = z1.z_L + 0.1 * torch.randn_like(z1.z_L)
        z2 = z1.__class__(z_H=z2_H, z_L=z2_L)
        
        # Distance before update
        dist_before = torch.sqrt(
            (z1.z_H - z2.z_H).pow(2).sum() + (z1.z_L - z2.z_L).pow(2).sum()
        )
        
        # Apply latent step (includes projection)
        z1_new = model.update_latent(z1, sample_plan, sample_batch)
        z2_new = model.update_latent(z2, sample_plan, sample_batch)
        
        dist_after = torch.sqrt(
            (z1_new.z_H - z2_new.z_H).pow(2).sum() + (z1_new.z_L - z2_new.z_L).pow(2).sum()
        )
        
        # Contraction should be preserved (with some tolerance for numerical errors)
        L_z = model.config.rl_target_Lz
        assert dist_after <= dist_before * L_z + 0.1, "Contraction violated by projection"
    
    def test_no_projection_when_disabled(self, model_without_projection, sample_batch, sample_plan):
        """Test that latent can grow when projection is disabled."""
        model = model_without_projection
        
        # Run many steps - latent might grow
        z = model.init_latent(sample_batch, sample_plan)
        initial_norm = z.z_H.norm(p=2, dim=(1, 2)).max()
        
        for _ in range(20):
            z = model.update_latent(z, sample_plan, sample_batch)
        
        # Just verify it runs without error - norm may or may not grow
        final_norm = z.z_H.norm(p=2, dim=(1, 2)).max()
        assert final_norm >= 0, "Norm should be non-negative"


# =============================================================================
# Test: Plan Change Tracking for Slow-Drift Premises
# =============================================================================

class TestPlanChangeTracking:
    """Tests for plan change tracking Δy_max."""
    
    def test_plan_change_integer_tokens(self):
        """Test plan change with integer tokens (Hamming distance)."""
        y_old = torch.tensor([1, 2, 3, 4, 5])
        y_new = torch.tensor([1, 2, 9, 4, 5])  # One change
        
        change = estimate_plan_change(y_old, y_new)
        assert change == 1.0, f"Expected 1 change, got {change}"
    
    def test_plan_change_multiple_edits(self):
        """Test plan change with multiple edits."""
        y_old = torch.tensor([1, 2, 3, 4, 5])
        y_new = torch.tensor([9, 2, 9, 4, 9])  # Three changes
        
        change = estimate_plan_change(y_old, y_new)
        assert change == 3.0, f"Expected 3 changes, got {change}"
    
    def test_plan_change_batch(self):
        """Test plan change with batched plans."""
        y_old = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y_new = torch.tensor([[1, 9, 3], [9, 9, 6]])  # 1 and 2 changes
        
        change = estimate_plan_change(y_old, y_new)
        assert change == 2.0, f"Expected max change of 2, got {change}"
    
    def test_plan_change_no_change(self):
        """Test plan change when plans are identical."""
        y = torch.tensor([1, 2, 3, 4, 5])
        change = estimate_plan_change(y, y.clone())
        assert change == 0.0, f"Expected 0 change, got {change}"


# =============================================================================
# Test: Persistent Slow-Drift Bound
# =============================================================================

class TestDriftBound:
    """Tests for drift bound C_drift(n)."""
    
    def test_theoretical_drift_bound(self):
        """Test theoretical drift bound computation."""
        kappa_n = 0.5  # L_z^n
        E_0 = 1.0
        L_zstar = 0.5
        delta_y_max = 0.1
        
        drift = compute_theoretical_drift_bound(kappa_n, E_0, L_zstar, delta_y_max)
        
        # Expected: κ_n * E_0 + (κ_n * L_{z*} * Δy_max) / (1 - κ_n)
        expected = 0.5 * 1.0 + (0.5 * 0.5 * 0.1) / 0.5
        assert abs(drift - expected) < 1e-6, f"Expected {expected}, got {drift}"
    
    def test_drift_bound_not_contractive(self):
        """Test drift bound returns inf when not contractive."""
        drift = compute_theoretical_drift_bound(1.5, 1.0, 0.5, 0.1)
        assert drift == float("inf"), "Should return inf when κ_n >= 1"
    
    def test_estimate_cdrift(self, model_without_projection, sample_batch, sample_plan):
        """Test empirical drift estimation."""
        model = model_without_projection
        
        # Initialize latent
        z = model.init_latent(sample_batch, sample_plan)
        
        # Drift from fresh init should be small
        drift = estimate_Cdrift(model, sample_batch, sample_plan, z, n=4)
        
        # With fresh init, drift should be near zero
        assert drift >= 0, "Drift should be non-negative"
        # Note: actual drift may be non-zero due to numerical precision


# =============================================================================
# Test: Value-of-Memory Diagnostic
# =============================================================================

class TestValueOfMemory:
    """Tests for value of memory computation."""
    
    def test_value_of_memory_residual(self, model_without_projection, sample_batch, sample_plan):
        """Test value of memory residual computation."""
        model = model_without_projection
        
        rewards = torch.zeros(sample_batch["inputs"].shape[0])
        dones = torch.zeros(sample_batch["inputs"].shape[0], dtype=torch.bool)
        
        # Get next values
        with torch.no_grad():
            v_next, _ = model.used_value(sample_batch, sample_plan, n=4)
        
        vom_metrics = compute_value_of_memory_residual(
            model=model,
            x_batch=sample_batch,
            y_batch=sample_plan,
            z_batch=None,
            n=4,
            rewards=rewards,
            next_values=v_next,
            dones=dones,
            gamma=0.99,
        )
        
        # Check all expected keys are present
        assert "v_memoryless_mean" in vom_metrics
        assert "v_persistent_mean" in vom_metrics
        assert "value_of_memory_mean" in vom_metrics
        assert "value_of_memory_max" in vom_metrics
        assert "bellman_residual_memoryless_mean" in vom_metrics
        
        # Value of memory should be non-negative
        assert vom_metrics["value_of_memory_mean"] >= 0
        assert vom_metrics["value_of_memory_max"] >= 0


# =============================================================================
# Test: Exact Baseline Computation (Theorem 6.7)
# =============================================================================

class TestExactBaseline:
    """Tests exact statewise centering used by Theorem 6.7."""
    
    def test_exact_advantage_centering(self):
        """Test that exact advantages are centered: E_{a~π}[Â(s,a)] = 0."""
        batch_size = 4
        num_actions = 10
        
        # Create mock Q-values and policy
        q_values = torch.randn(batch_size, num_actions)
        probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        
        # Exact baseline: E[Q] = Σ π(a) Q(a)
        exact_baseline = (probs * q_values).sum(dim=-1)
        
        # Compute advantages for all actions
        advantages_all = q_values - exact_baseline.unsqueeze(-1)
        
        # Expected advantage: E_{a~π}[Â(s,a)] = Σ π(a) Â(a) = 0
        expected_adv = (probs * advantages_all).sum(dim=-1)
        
        # Should be exactly zero (up to numerical precision)
        assert (expected_adv.abs() < 1e-5).all(), f"Centering violated: {expected_adv}"
    
    def test_compute_exact_advantage(self):
        """Test compute_exact_advantage function."""
        batch_size = 4
        num_actions = 10
        
        q_values = torch.randn(batch_size, num_actions)
        probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        exact_baseline = (probs * q_values).sum(dim=-1)
        
        # Sample some actions
        actions = torch.randint(0, num_actions, (batch_size,))
        
        # Compute advantages
        adv = compute_exact_advantage(q_values, exact_baseline, actions)
        
        # Verify shape
        assert adv.shape == (batch_size,), f"Wrong shape: {adv.shape}"
        
        # Verify values
        for i in range(batch_size):
            expected = q_values[i, actions[i]] - exact_baseline[i]
            assert abs(adv[i] - expected) < 1e-5, f"Wrong advantage at {i}"


# =============================================================================
# Test: RLConfig New Options
# =============================================================================

class TestRLConfigTheoryOptions:
    """Tests for new theory-exact config options."""
    
    def test_default_values(self):
        """Test default values for new config options."""
        cfg = RLConfig()
        
        # Most new options default to off/zero; projection defaults to enabled
        # with radius 10.0 under the current explicit configuration contract.
        assert cfg.exact_baseline_summation == False
        assert cfg.latent_ball_radius == 10.0  # ENABLED by default for theory alignment
        assert cfg.track_drift_metrics == False
        assert cfg.track_plan_change == False
        assert cfg.compute_value_of_memory == False
    
    def test_enable_theory_exact(self):
        """Test enabling all theory-exact options."""
        cfg = RLConfig(
            exact_baseline_summation=True,
            latent_ball_radius=5.0,
            track_drift_metrics=True,
            track_plan_change=True,
            compute_value_of_memory=True,
        )
        
        assert cfg.exact_baseline_summation == True
        assert cfg.latent_ball_radius == 5.0
        assert cfg.track_drift_metrics == True
        assert cfg.track_plan_change == True
        assert cfg.compute_value_of_memory == True


# =============================================================================
# Integration Test: Full Trainer with Theory Components
# =============================================================================

class TestTrainerTheoryIntegration:
    """Integration tests for trainer with theory components."""
    
    def test_trainer_with_checker_fn(self, model_without_projection):
        """Test trainer can be configured with checker function."""
        model = model_without_projection
        
        # Create minimal dataset
        class DummyDataset:
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return {
                    "inputs": torch.randint(0, 10, (16,)),
                    "puzzle_identifiers": torch.tensor(idx),
                    "initial_plan": torch.zeros(16, dtype=torch.long),
                    "solution": torch.randint(0, 10, (16,)),
                }
        
        dataset = DummyDataset()
        
        def dummy_checker(x, y):
            return 0.0
        
        env_cfg = PlanEditEnvConfig(max_edits=8, gamma=0.99, vocab_size=10)
        env = PlanEditEnv(dataset, dummy_checker, env_cfg)
        env.set_stop_action_id(16 * 10)  # seq_len * vocab_size
        
        rl_cfg = RLConfig(
            batch_size=4,
            max_edits=8,
            exact_baseline_summation=False,  # Don't actually run exact baseline
            track_drift_metrics=True,
            track_plan_change=True,
        )
        
        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))
        trainer.set_checker_fn(dummy_checker)
        
        assert trainer._checker_fn is not None
        assert trainer._checker_fn == dummy_checker
    
    def test_theory_stats_tracking(self, model_without_projection):
        """Test theory statistics are tracked properly."""
        model = model_without_projection
        
        class DummyDataset:
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return {
                    "inputs": torch.randint(0, 10, (16,)),
                    "puzzle_identifiers": torch.tensor(idx),
                    "initial_plan": torch.zeros(16, dtype=torch.long),
                    "solution": torch.randint(0, 10, (16,)),
                }
        
        dataset = DummyDataset()
        
        def dummy_checker(x, y):
            return 0.0
        
        env_cfg = PlanEditEnvConfig(max_edits=8, gamma=0.99, vocab_size=10)
        env = PlanEditEnv(dataset, dummy_checker, env_cfg)
        env.set_stop_action_id(16 * 10)
        
        rl_cfg = RLConfig(
            batch_size=4,
            max_edits=8,
            episodic_latent=True,  # Use episodic mode (simpler)
            track_plan_change=True,
        )
        
        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))
        
        # Collect some episodes
        for _ in range(2):
            trainer.collect_episode()
        
        # Check that plan changes were tracked
        stats = trainer.get_theory_stats()
        if trainer._plan_changes:
            assert "plan_change_mean" in stats
            assert "plan_change_max" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
