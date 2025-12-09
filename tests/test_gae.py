"""
Tests for Generalized Advantage Estimation (GAE).
This corresponds to the optional λ-returns mentioned in Section 6.2 of the paper.
"""
import torch

from rl.value_targets import compute_gae, compute_gae_trajectory


def test_gae_single_step_equals_td_error():
    """Test that single-step GAE with λ=0 equals TD error."""
    batch_size = 4
    rewards = torch.tensor([1.0, 2.0, 0.5, -0.5])
    values = torch.tensor([0.5, 1.0, 0.3, 0.2])
    next_values = torch.tensor([0.8, 0.9, 0.4, 0.1])
    dones = torch.zeros(batch_size, dtype=torch.bool)
    gamma = 0.99
    gae_lambda = 0.0  # λ=0 gives pure TD error

    gae = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda)

    # For λ=0, GAE = TD error = r + γV(s') - V(s)
    expected = rewards + gamma * next_values - values
    assert torch.allclose(gae, expected)


def test_gae_with_terminal_masks_bootstrap():
    """Test that GAE correctly masks bootstrap for terminal states."""
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([0.8, 100.0])  # Large next_value that should be masked
    dones = torch.tensor([False, True])
    gamma = 0.99
    gae_lambda = 0.95

    gae = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda)

    # For non-terminal with λ>0: TD_error * (1 + γλ)
    td_error_0 = 1.0 + 0.99 * 0.8 - 0.5  # = 1.292
    expected_0 = td_error_0 * (1.0 + gamma * gae_lambda)  # = 1.292 * 1.9405

    # For terminal: TD_error * 1 (mask zeros out the λ term)
    expected_1 = 2.0 - 1.0  # = 1.0

    assert abs(gae[0].item() - expected_0) < 1e-5
    assert abs(gae[1].item() - expected_1) < 1e-6


def test_gae_single_step_lambda_effect():
    """Test that non-zero λ amplifies advantages for non-terminal states."""
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.5])
    next_values = torch.tensor([0.6, 0.6])
    dones = torch.zeros(2, dtype=torch.bool)
    gamma = 0.99

    # Compare λ=0 (pure TD) vs λ=0.95 (GAE-approximated)
    gae_0 = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda=0.0)
    gae_95 = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda=0.95)

    # With λ>0, advantages should be scaled up for non-terminal transitions
    # gae_95 = gae_0 * (1 + γλ) = gae_0 * 1.9405
    assert torch.allclose(gae_95, gae_0 * (1 + gamma * 0.95), atol=1e-5)

    # Higher λ should give larger advantages
    assert (gae_95 > gae_0).all()


def test_gae_trajectory_simple_case():
    """Test GAE trajectory computation with a simple deterministic case."""
    # Simple 3-step trajectory with r=1 at each step, V=0 everywhere
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.zeros(3, dtype=torch.bool)
    gamma = 0.99
    gae_lambda = 0.95
    last_value = 0.0

    gae = compute_gae_trajectory(rewards, values, dones, gamma, gae_lambda, last_value)

    # With V=0 everywhere and r=1:
    # δ_t = r_t + γV_{t+1} - V_t = 1 + 0 - 0 = 1
    # A_2 = δ_2 = 1
    # A_1 = δ_1 + γλ * A_2 = 1 + 0.99 * 0.95 * 1 = 1.9405
    # A_0 = δ_0 + γλ * A_1 = 1 + 0.99 * 0.95 * 1.9405 ≈ 2.8245
    assert gae.shape == (3,)
    assert gae[2].item() < gae[1].item() < gae[0].item()  # Earlier steps have larger advantages


def test_gae_trajectory_with_early_terminal():
    """Test GAE trajectory with early termination."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.tensor([False, True, False])  # Terminal at step 1
    gamma = 0.99
    gae_lambda = 0.95
    last_value = 0.0

    gae = compute_gae_trajectory(rewards, values, dones, gamma, gae_lambda, last_value)

    # After terminal, GAE should reset
    # A_2 = δ_2 = 1 (fresh start after terminal)
    # A_1 = δ_1 = 1 (terminal, no future)
    # A_0 = δ_0 + γλ * A_1 * mask_0 = 1 + γλ * 1 = 1.9405
    assert gae.shape == (3,)


def test_gae_lambda_zero_equals_td():
    """Test that λ=0 GAE equals pure TD error."""
    rewards = torch.tensor([1.0, 2.0, 0.5])
    values = torch.tensor([0.5, 1.0, 0.3])
    dones = torch.zeros(3, dtype=torch.bool)
    gamma = 0.99
    gae_lambda = 0.0  # Pure TD
    last_value = 0.2

    gae = compute_gae_trajectory(rewards, values, dones, gamma, gae_lambda, last_value)

    # With λ=0, A_t = δ_t = r_t + γV_{t+1} - V_t
    values_extended = torch.cat([values, torch.tensor([last_value])])
    expected = rewards + gamma * values_extended[1:] - values
    assert torch.allclose(gae, expected, atol=1e-5)


def test_gae_lambda_one_approaches_monte_carlo():
    """Test that λ=1 GAE approaches Monte Carlo returns."""
    # With λ=1, GAE should give full returns minus baseline
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)  # Zero baseline
    dones = torch.zeros(3, dtype=torch.bool)
    gamma = 0.99
    gae_lambda = 1.0  # Full Monte Carlo
    last_value = 0.0

    gae = compute_gae_trajectory(rewards, values, dones, gamma, gae_lambda, last_value)

    # With V=0 and λ=1, A_t should equal discounted return from t
    # A_0 = 1 + γ*1 + γ²*1 = 1 + 0.99 + 0.9801 ≈ 2.9701
    # A_1 = 1 + γ*1 = 1.99
    # A_2 = 1
    expected_0 = 1.0 + gamma + gamma ** 2
    expected_1 = 1.0 + gamma
    expected_2 = 1.0

    assert abs(gae[0].item() - expected_0) < 1e-4
    assert abs(gae[1].item() - expected_1) < 1e-4
    assert abs(gae[2].item() - expected_2) < 1e-4

