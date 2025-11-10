"""Unit tests for ValueHead with EMA target network."""

import pytest

# Try to import torch, skip tests if not available
torch = pytest.importorskip("torch", reason="torch not installed")

from rl.value_head import ValueHead  # noqa: E402


@pytest.mark.unit
def test_value_head_initialization():
    """Test ValueHead initializes correctly with target network."""
    z_dim, x_dim, hidden = 64, 32, 128
    value_head = ValueHead(z_dim, x_dim, hidden)

    # Check network exists
    assert value_head.net is not None
    assert value_head.target is not None

    # Check target parameters don't require grad
    for p in value_head.target.parameters():
        assert not p.requires_grad

    # Check target is initialized as copy of online network
    for p, tp in zip(value_head.net.parameters(), value_head.target.parameters()):
        assert torch.allclose(p.data, tp.data), "Target should be initialized as copy"


@pytest.mark.unit
def test_value_head_forward():
    """Test forward pass produces correct output shape."""
    batch_size = 4
    z_dim, x_dim = 64, 32
    value_head = ValueHead(z_dim, x_dim)

    z = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, x_dim)

    values = value_head.forward(z, x)

    assert values.shape == (batch_size,), f"Expected shape ({batch_size},), got {values.shape}"
    assert not torch.isnan(values).any(), "Values contain NaN"
    assert not torch.isinf(values).any(), "Values contain Inf"


@pytest.mark.unit
def test_value_head_target_value():
    """Test target network forward pass."""
    batch_size = 4
    z_dim, x_dim = 64, 32
    value_head = ValueHead(z_dim, x_dim)

    z = torch.randn(batch_size, z_dim)
    x = torch.randn(batch_size, x_dim)

    target_values = value_head.target_value(z, x)

    assert target_values.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {target_values.shape}"
    assert not target_values.requires_grad, "Target values should not require grad"


@pytest.mark.unit
def test_update_target_moves_parameters_closer():
    """Test that update_target moves target parameters closer to online parameters.

    This is the key acceptance test: after update_target, target parameters
    should be closer to online parameters than before.
    """
    z_dim, x_dim = 64, 32
    value_head = ValueHead(z_dim, x_dim)

    # Modify online network parameters significantly
    with torch.no_grad():
        for p in value_head.net.parameters():
            p.add_(torch.randn_like(p) * 2.0)

    # Measure distance before update
    distances_before = []
    for p, tp in zip(value_head.net.parameters(), value_head.target.parameters()):
        distance = (p - tp).abs().mean().item()
        distances_before.append(distance)

    # Update target
    tau = 0.995
    value_head.update_target(tau)

    # Measure distance after update
    distances_after = []
    for p, tp in zip(value_head.net.parameters(), value_head.target.parameters()):
        distance = (p - tp).abs().mean().item()
        distances_after.append(distance)

    # Target should be closer to online after update
    for before, after in zip(distances_before, distances_after):
        assert after < before, f"Target should move closer: before={before:.6f}, after={after:.6f}"


@pytest.mark.unit
def test_update_target_ema_correctness():
    """Test that update_target implements correct EMA formula."""
    z_dim, x_dim = 64, 32
    value_head = ValueHead(z_dim, x_dim)

    # Store original target parameters
    original_target_params = [tp.clone() for tp in value_head.target.parameters()]

    # Modify online network
    with torch.no_grad():
        for p in value_head.net.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Update with tau
    tau = 0.9
    value_head.update_target(tau)

    # Verify EMA formula: θ_target = tau * θ_target + (1 - tau) * θ_online
    for p, tp, orig_tp in zip(
        value_head.net.parameters(),
        value_head.target.parameters(),
        original_target_params,
    ):
        expected = tau * orig_tp + (1 - tau) * p
        assert torch.allclose(tp, expected, atol=1e-6), "EMA update formula not correct"


@pytest.mark.unit
def test_update_target_no_grad():
    """Test that update_target doesn't build computation graph."""
    z_dim, x_dim = 64, 32
    value_head = ValueHead(z_dim, x_dim)

    # All parameters should track gradients for online network
    for p in value_head.net.parameters():
        assert p.requires_grad

    # Update target
    value_head.update_target()

    # Target parameters should still not require gradients
    for tp in value_head.target.parameters():
        assert not tp.requires_grad


@pytest.mark.unit
def test_value_head_different_tau():
    """Test update_target with different tau values."""
    z_dim, x_dim = 64, 32
    value_head = ValueHead(z_dim, x_dim)

    # Modify online network
    with torch.no_grad():
        for p in value_head.net.parameters():
            p.add_(torch.randn_like(p) * 1.0)

    # Test with smaller tau (faster update)
    tau_small = 0.5
    distances_small = []

    value_head_small = ValueHead(z_dim, x_dim)
    with torch.no_grad():
        for p, p_src in zip(value_head_small.net.parameters(), value_head.net.parameters()):
            p.copy_(p_src)

    value_head_small.update_target(tau_small)

    for p, tp in zip(value_head_small.net.parameters(), value_head_small.target.parameters()):
        distances_small.append((p - tp).abs().mean().item())

    # Test with larger tau (slower update)
    tau_large = 0.99
    distances_large = []

    value_head_large = ValueHead(z_dim, x_dim)
    with torch.no_grad():
        for p, p_src in zip(value_head_large.net.parameters(), value_head.net.parameters()):
            p.copy_(p_src)

    value_head_large.update_target(tau_large)

    for p, tp in zip(value_head_large.net.parameters(), value_head_large.target.parameters()):
        distances_large.append((p - tp).abs().mean().item())

    # Smaller tau should result in smaller distance (faster convergence)
    avg_distance_small = sum(distances_small) / len(distances_small)
    avg_distance_large = sum(distances_large) / len(distances_large)

    assert (
        avg_distance_small < avg_distance_large
    ), f"Smaller tau should converge faster: {avg_distance_small} vs {avg_distance_large}"
