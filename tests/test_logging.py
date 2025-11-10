"""
Unit tests for RL logging utilities (Step 11).

Verify that:
1. RLLogger can log metrics to stdout and optionally W&B
2. Metric computation functions work correctly
3. Metrics have finite values and expected properties
"""
import pytest

torch = pytest.importorskip("torch")

from rl.logging import (  # noqa: E402
    RLLogger,
    compute_policy_entropy,
    compute_policy_kl,
    compute_ppo_clip_fraction,
    compute_score_delta,
    compute_value_residual,
)


def test_rl_logger_initialization():
    """Test RLLogger initialization."""
    # Without W&B
    logger = RLLogger(use_wandb=False, log_interval=5)
    assert logger.log_interval == 5
    assert logger.step == 0
    assert logger.use_wandb is False

    # With W&B (should gracefully degrade if not installed)
    logger_wandb = RLLogger(
        use_wandb=True, log_interval=10, project_name="test", run_name="test_run"
    )
    # If wandb is not installed, it should disable W&B
    assert logger_wandb.log_interval == 10


def test_rl_logger_logging():
    """Test that RLLogger can log metrics."""
    logger = RLLogger(use_wandb=False, log_interval=1)

    # Log some metrics
    metrics = {
        "loss": 1.5,
        "value": torch.tensor(2.5),
        "count": 10,
    }

    # Should not raise any errors
    logger.log(metrics, step=1)
    assert logger.step == 1

    # Log again
    logger.log(metrics)
    assert logger.step == 2


def test_rl_logger_interval():
    """Test that RLLogger only logs at specified intervals."""
    logger = RLLogger(use_wandb=False, log_interval=5)

    # Log at steps 1-4 should not print (interval=5)
    for i in range(1, 5):
        logger.log({"loss": 1.0}, step=i)

    # Log at step 5 should print
    logger.log({"loss": 1.0}, step=5)
    assert logger.step == 5


def test_compute_value_residual():
    """Test value residual computation."""
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    returns = torch.tensor([1.5, 2.5, 2.5, 3.5])

    residual = compute_value_residual(values, returns)

    # Should be the max absolute difference
    expected_max = torch.tensor(0.5)  # max(|1-1.5|, |2-2.5|, |3-2.5|, |4-3.5|)
    assert torch.isclose(residual, expected_max)


def test_compute_policy_kl():
    """Test policy KL divergence computation."""
    log_probs_old = torch.tensor([-1.0, -1.5, -2.0])
    log_probs_new = torch.tensor([-1.1, -1.4, -2.1])

    kl = compute_policy_kl(log_probs_new, log_probs_old)

    # KL should be positive (since we changed the policy)
    assert kl.item() > 0

    # KL should be finite
    assert torch.isfinite(kl)


def test_compute_policy_kl_zero():
    """Test that KL is zero when policies are identical."""
    log_probs = torch.tensor([-1.0, -1.5, -2.0])

    kl = compute_policy_kl(log_probs, log_probs)

    # KL should be zero when policies are identical
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_compute_policy_entropy():
    """Test policy entropy computation."""
    batch_size = 4
    num_cells = 81
    num_vals = 10

    # Create log probabilities (should sum to 1 when exponentiated)
    log_probs_pos = torch.log_softmax(torch.randn(batch_size, num_cells), dim=-1)
    log_probs_val = torch.log_softmax(torch.randn(batch_size, num_vals), dim=-1)

    entropy = compute_policy_entropy(log_probs_pos, log_probs_val)

    # Entropy should be positive
    assert entropy.item() > 0

    # Entropy should be finite
    assert torch.isfinite(entropy)


def test_compute_policy_entropy_uniform():
    """Test entropy for uniform distribution (should be maximal)."""
    batch_size = 2
    num_cells = 10
    num_vals = 5

    # Uniform distributions (log(1/N) for each element)
    log_probs_pos = torch.full((batch_size, num_cells), -torch.log(torch.tensor(num_cells)))
    log_probs_val = torch.full((batch_size, num_vals), -torch.log(torch.tensor(num_vals)))

    entropy = compute_policy_entropy(log_probs_pos, log_probs_val)

    # For uniform distribution, entropy = log(N)
    # H(pos) + H(val) ≈ log(10) + log(5)
    expected_entropy = torch.log(torch.tensor(num_cells)) + torch.log(torch.tensor(num_vals))

    assert torch.isclose(entropy, expected_entropy, rtol=1e-4)


def test_compute_ppo_clip_fraction():
    """Test PPO clip fraction computation."""
    # Case 1: No clipping (ratio ~1)
    log_probs_old = torch.tensor([-1.0, -1.5, -2.0, -2.5])
    log_probs_new = torch.tensor([-1.05, -1.55, -2.05, -2.55])  # Small change

    clip_frac = compute_ppo_clip_fraction(log_probs_new, log_probs_old, epsilon=0.2)

    # Clip fraction should be very low
    assert clip_frac.item() < 0.1

    # Case 2: High clipping (large ratio change)
    log_probs_new_large = torch.tensor([-0.5, -0.5, -0.5, -0.5])  # Large change

    clip_frac_large = compute_ppo_clip_fraction(log_probs_new_large, log_probs_old, epsilon=0.2)

    # Clip fraction should be high
    assert clip_frac_large.item() > 0.5


def test_compute_score_delta():
    """Test score delta computation."""
    scores_before = torch.tensor([0.5, 0.6, 0.7, 0.8])
    scores_after = torch.tensor([0.6, 0.7, 0.8, 0.85])

    delta = compute_score_delta(scores_before, scores_after)

    # Mean improvement should be positive
    expected_delta = (scores_after - scores_before).mean()
    assert torch.isclose(delta, expected_delta)

    # Should be approximately 0.0875
    assert delta.item() > 0


def test_compute_score_delta_negative():
    """Test score delta when scores decrease."""
    scores_before = torch.tensor([0.8, 0.7, 0.6, 0.5])
    scores_after = torch.tensor([0.7, 0.6, 0.5, 0.4])

    delta = compute_score_delta(scores_before, scores_after)

    # Mean improvement should be negative (scores decreased)
    assert delta.item() < 0


def test_metrics_are_finite():
    """Test that all metric functions return finite values."""
    batch_size = 4
    num_cells = 81
    num_vals = 10

    # Generate random inputs
    values = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    log_probs_old = torch.randn(batch_size)
    log_probs_new = torch.randn(batch_size)
    log_probs_pos = torch.log_softmax(torch.randn(batch_size, num_cells), dim=-1)
    log_probs_val = torch.log_softmax(torch.randn(batch_size, num_vals), dim=-1)
    scores_before = torch.rand(batch_size)
    scores_after = torch.rand(batch_size)

    # Compute all metrics
    residual = compute_value_residual(values, returns)
    kl = compute_policy_kl(log_probs_new, log_probs_old)
    entropy = compute_policy_entropy(log_probs_pos, log_probs_val)
    clip_frac = compute_ppo_clip_fraction(log_probs_new, log_probs_old, epsilon=0.2)
    delta = compute_score_delta(scores_before, scores_after)

    # All metrics should be finite
    assert torch.isfinite(residual)
    assert torch.isfinite(kl)
    assert torch.isfinite(entropy)
    assert torch.isfinite(clip_frac)
    assert torch.isfinite(delta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
