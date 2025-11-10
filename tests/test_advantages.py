"""Unit tests for advantage estimation functions."""

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from rl.advantages import center_advantages, compute_advantages  # noqa: E402


@pytest.mark.unit
def test_center_advantages_zero_mean():
    """Test that centered advantages have zero mean."""
    # Create advantages with non-zero mean
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    centered = center_advantages(advantages)

    # Check that mean is approximately zero
    assert torch.allclose(
        centered.mean(), torch.tensor(0.0), atol=1e-6
    ), f"Centered advantages should have zero mean, got {centered.mean().item()}"


@pytest.mark.unit
def test_center_advantages_unit_variance():
    """Test that centered advantages have unit variance (approximately)."""
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    centered = center_advantages(advantages)

    # Check that std is approximately 1.0
    assert torch.allclose(
        centered.std(unbiased=False), torch.tensor(1.0), atol=1e-5
    ), f"Centered advantages should have unit std, got {centered.std(unbiased=False).item()}"


@pytest.mark.unit
def test_center_advantages_batch():
    """Test centering with batch of advantages."""
    # Create a batch of advantages (T=5, B=3)
    advantages = torch.randn(5, 3)

    centered = center_advantages(advantages)

    # Mean across all elements should be zero
    assert torch.allclose(centered.mean(), torch.tensor(0.0), atol=1e-6)

    # Shape should be preserved
    assert centered.shape == advantages.shape


@pytest.mark.unit
def test_center_advantages_stability():
    """Test that centering is numerically stable with small variance."""
    # Advantages with very small variance
    advantages = torch.tensor([1.0, 1.0001, 1.0002, 0.9999])

    # Should not raise an error due to division by near-zero std
    centered = center_advantages(advantages, eps=1e-8)

    # Mean should still be zero
    assert torch.allclose(centered.mean(), torch.tensor(0.0), atol=1e-5)


@pytest.mark.unit
def test_compute_advantages_simple():
    """Test basic advantage computation."""
    returns = torch.tensor([10.0, 8.0, 6.0, 4.0])
    baseline = torch.tensor([9.0, 7.0, 5.0, 3.0])

    advantages = compute_advantages(returns, baseline)

    expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(advantages, expected, atol=1e-6)


@pytest.mark.unit
def test_compute_advantages_negative():
    """Test advantage computation with negative advantages."""
    returns = torch.tensor([1.0, 2.0, 3.0])
    baseline = torch.tensor([5.0, 6.0, 7.0])

    advantages = compute_advantages(returns, baseline)

    expected = torch.tensor([-4.0, -4.0, -4.0])
    assert torch.allclose(advantages, expected, atol=1e-6)


@pytest.mark.unit
def test_compute_advantages_batch():
    """Test advantage computation with batch dimension."""
    # (T=3, B=2)
    returns = torch.tensor([[10.0, 8.0], [6.0, 4.0], [2.0, 0.0]])
    baseline = torch.tensor([[9.0, 7.0], [5.0, 3.0], [1.0, -1.0]])

    advantages = compute_advantages(returns, baseline)

    expected = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(advantages, expected, atol=1e-6)


@pytest.mark.unit
def test_centered_advantages_pipeline():
    """Test full pipeline: compute advantages then center."""
    returns = torch.tensor([10.0, 15.0, 20.0, 25.0])
    baseline = torch.tensor([8.0, 12.0, 18.0, 24.0])

    # Compute advantages
    advantages = compute_advantages(returns, baseline)

    # Center them
    centered = center_advantages(advantages)

    # Centered should have zero mean
    assert torch.allclose(centered.mean(), torch.tensor(0.0), atol=1e-6)

    # Check values make sense
    # Original advantages: [2, 3, 2, 1]
    # Mean: 2.0, Std: ~0.816
    # Centered: ~[0, 1.225, 0, -1.225]
    assert centered.shape == advantages.shape
