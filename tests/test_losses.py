"""Unit tests for RL loss functions."""

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from rl.losses import ppo_policy_loss, value_bellman_residual_loss  # noqa: E402


@pytest.mark.unit
def test_value_bellman_residual_loss_basic():
    """Test basic BR loss computation."""
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    returns = torch.tensor([1.5, 2.5, 3.5, 4.5])

    loss = value_bellman_residual_loss(values, returns)

    # Expected loss: mean of (0.5^2, 0.5^2, 0.5^2, 0.5^2) = 0.25
    expected = torch.tensor(0.25)
    assert torch.allclose(loss, expected, atol=1e-6)


@pytest.mark.unit
def test_value_bellman_residual_loss_no_gradient_to_returns():
    """Test that BR loss doesn't propagate gradients to returns."""
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    returns = torch.tensor([1.5, 2.5, 3.5], requires_grad=True)

    loss = value_bellman_residual_loss(values, returns)
    loss.backward()

    # Values should have gradients
    assert values.grad is not None

    # Returns should NOT have gradients (due to detach)
    assert returns.grad is None


@pytest.mark.unit
def test_value_bellman_residual_loss_clipping():
    """Test value clipping in BR loss."""
    values = torch.tensor([100.0, -100.0, 5.0])
    returns = torch.tensor([1.0, 2.0, 3.0])

    loss = value_bellman_residual_loss(values, returns, clip_value=10.0)

    # Values should be clipped to [-10, 10]
    # Clipped: [10, -10, 5]
    # Targets: [1, 2, 3]
    # Errors: [9, -12, 2]
    # MSE: (81 + 144 + 4) / 3 = 76.333...
    expected_loss = (81.0 + 144.0 + 4.0) / 3.0
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-4)


@pytest.mark.unit
def test_ppo_policy_loss_no_change():
    """Test PPO loss when policy doesn't change (ratio = 1)."""
    log_probs = torch.tensor([0.0, -1.0, -2.0])
    old_log_probs = torch.tensor([0.0, -1.0, -2.0])
    advantages = torch.tensor([1.0, -1.0, 0.5])

    loss = ppo_policy_loss(log_probs, old_log_probs, advantages, epsilon=0.2)

    # When ratio = 1, loss = -mean(advantages)
    expected = -advantages.mean()
    assert torch.allclose(loss, expected, atol=1e-6)


@pytest.mark.unit
def test_ppo_policy_loss_moves_in_advantage_direction():
    """Test that PPO loss decreases when policy moves in advantage direction."""
    # Old policy log probs
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0])

    # Positive advantages: we want to increase probability
    advantages = torch.tensor([1.0, 1.0, 1.0])

    # Initial: same as old
    log_probs_same = torch.tensor([-1.0, -1.0, -1.0])
    loss_same = ppo_policy_loss(log_probs_same, old_log_probs, advantages)

    # Improved: increased log prob (higher probability) for positive advantages
    log_probs_better = torch.tensor([-0.5, -0.5, -0.5])
    loss_better = ppo_policy_loss(log_probs_better, old_log_probs, advantages)

    # Loss should decrease (become more negative or less positive)
    assert loss_better < loss_same, (
        f"Loss should decrease when moving in advantage direction: "
        f"{loss_better.item()} vs {loss_same.item()}"
    )


@pytest.mark.unit
def test_ppo_policy_loss_moves_against_advantage_direction():
    """Test that PPO loss increases when policy moves against advantage direction."""
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    advantages = torch.tensor([1.0, 1.0, 1.0])

    # Same as old
    log_probs_same = torch.tensor([-1.0, -1.0, -1.0])
    loss_same = ppo_policy_loss(log_probs_same, old_log_probs, advantages)

    # Worse: decreased log prob (lower probability) for positive advantages
    log_probs_worse = torch.tensor([-2.0, -2.0, -2.0])
    loss_worse = ppo_policy_loss(log_probs_worse, old_log_probs, advantages)

    # Loss should increase
    assert loss_worse > loss_same, "Loss should increase when moving against advantage direction"


@pytest.mark.unit
def test_ppo_policy_loss_clipping():
    """Test that PPO clips probability ratios."""
    old_log_probs = torch.tensor([-1.0, -1.0])
    advantages = torch.tensor([1.0, 1.0])
    epsilon = 0.2

    # Very large increase in probability (ratio >> 1 + epsilon)
    log_probs_huge = torch.tensor([2.0, 2.0])  # exp(2 - (-1)) = exp(3) ≈ 20.08

    loss = ppo_policy_loss(log_probs_huge, old_log_probs, advantages, epsilon=epsilon)

    # With clipping, ratio is limited to 1 + epsilon = 1.2
    # Clipped objective = 1.2 * advantages = 1.2 * [1, 1]
    # Loss = -mean([1.2, 1.2]) = -1.2
    expected = torch.tensor(-1.2)
    assert torch.allclose(
        loss, expected, atol=1e-4
    ), f"Expected clipped loss {expected.item()}, got {loss.item()}"


@pytest.mark.unit
def test_ppo_policy_loss_negative_advantages():
    """Test PPO with negative advantages (bad actions)."""
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    advantages = torch.tensor([-2.0, -2.0, -2.0])

    log_probs_same = torch.tensor([-1.0, -1.0, -1.0])
    loss_same = ppo_policy_loss(log_probs_same, old_log_probs, advantages)

    # Decrease probability (good for negative advantages)
    log_probs_better = torch.tensor([-2.0, -2.0, -2.0])
    loss_better = ppo_policy_loss(log_probs_better, old_log_probs, advantages)

    # Loss should decrease (become more negative)
    assert loss_better < loss_same, "Loss should decrease when reducing prob of bad actions"


@pytest.mark.unit
def test_ppo_policy_loss_with_entropy():
    """Test PPO loss with entropy regularization."""
    log_probs = torch.tensor([-1.0, -1.0, -1.0])
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    advantages = torch.tensor([1.0, 1.0, 1.0])
    entropy = torch.tensor([0.5, 0.5, 0.5])

    # Without entropy
    loss_no_entropy = ppo_policy_loss(log_probs, old_log_probs, advantages)

    # With entropy
    loss_with_entropy = ppo_policy_loss(
        log_probs, old_log_probs, advantages, entropy=entropy, beta=0.01
    )

    # Loss with entropy should be lower (entropy term is subtracted)
    # loss_with_entropy = loss_no_entropy - 0.01 * 0.5 = loss_no_entropy - 0.005
    expected_diff = 0.01 * entropy.mean()
    assert torch.allclose(loss_no_entropy - loss_with_entropy, expected_diff, atol=1e-6)


@pytest.mark.unit
def test_ppo_policy_loss_gradient_flow():
    """Test that PPO loss allows gradient flow to log_probs."""
    log_probs = torch.tensor([-1.0, -1.0], requires_grad=True)
    old_log_probs = torch.tensor([-1.0, -1.0])
    advantages = torch.tensor([1.0, 1.0])

    loss = ppo_policy_loss(log_probs, old_log_probs, advantages)
    loss.backward()

    # Should have gradients
    assert log_probs.grad is not None
    assert not torch.isnan(log_probs.grad).any()


@pytest.mark.unit
def test_ppo_policy_loss_symmetric_clipping():
    """Test that PPO clips symmetrically for positive and negative advantages."""
    old_log_probs = torch.tensor([-1.0])
    epsilon = 0.2

    # Test with positive advantage
    advantages_pos = torch.tensor([1.0])
    log_probs_high = torch.tensor([2.0])  # Very high ratio

    loss_pos = ppo_policy_loss(log_probs_high, old_log_probs, advantages_pos, epsilon=epsilon)

    # Test with negative advantage
    advantages_neg = torch.tensor([-1.0])
    log_probs_low = torch.tensor([-4.0])  # Very low ratio

    loss_neg = ppo_policy_loss(log_probs_low, old_log_probs, advantages_neg, epsilon=epsilon)

    # Both should apply clipping
    # For positive advantage: clipped to 1+eps = 1.2, loss = -1.2
    # For negative advantage: clipped to 1-eps = 0.8, loss = -0.8
    expected_pos = torch.tensor(-1.2)
    expected_neg = torch.tensor(-0.8)

    assert torch.allclose(loss_pos, expected_pos, atol=1e-4)
    assert torch.allclose(loss_neg, expected_neg, atol=1e-4)


@pytest.mark.unit
def test_ppo_policy_loss_batch_processing():
    """Test PPO loss with batch of different ratios and advantages."""
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0, -1.0])

    # Mix of scenarios
    log_probs = torch.tensor(
        [
            -1.0,  # ratio=1, no change
            -0.5,  # ratio>1, improve
            -2.0,  # ratio<1, worsen
            2.0,  # ratio>>1, will be clipped
        ]
    )

    advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])

    loss = ppo_policy_loss(log_probs, old_log_probs, advantages, epsilon=0.2)

    # Loss should be computed correctly across batch
    assert not torch.isnan(loss)
    assert loss.shape == torch.Size([])  # Scalar
