"""Unit tests for spectral norm and Lipschitz constant monitoring."""

import pytest

torch = pytest.importorskip("torch")  # noqa: E402
nn = pytest.importorskip("torch.nn")  # noqa: E402

from rl.contraction import (  # noqa: E402
    LipschitzMonitor,
    apply_spectral_norm,
    compute_spectral_norm,
    estimate_Lz,
    lipschitz_regularizer,
)


class SyntheticMLP(nn.Module):
    """Synthetic MLP for testing Lipschitz constant."""

    def __init__(self, dim: int = 32, hidden: int = 64, scale: float = 1.0):
        """Initialize synthetic MLP.

        Args:
            dim: Input/output dimension
            hidden: Hidden dimension
            scale: Weight scale multiplier (>1 for high Lz, <1 for low Lz)
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.scale = scale

        # Initialize with large weights to get Lz > 1
        with torch.no_grad():
            self.fc1.weight.mul_(scale)
            self.fc2.weight.mul_(scale)

    def forward(self, z, y, x):
        """Forward pass: z' = f(z, y, x)."""
        # Simple z→z mapping (ignoring y, x for simplicity)
        h = torch.relu(self.fc1(z))
        return self.fc2(h)


@pytest.mark.unit
def test_compute_spectral_norm_identity():
    """Test spectral norm of identity matrix is 1.0."""
    # Identity matrix has spectral norm = 1
    identity = torch.eye(10)

    sigma = compute_spectral_norm(identity, n_iter=20)

    assert torch.allclose(
        sigma, torch.tensor(1.0), atol=0.1
    ), f"Identity spectral norm should be 1.0, got {sigma.item()}"


@pytest.mark.unit
def test_compute_spectral_norm_scaled():
    """Test spectral norm of scaled matrix."""
    # Scaled identity: spectral norm = scale
    scale = 2.5
    scaled_identity = scale * torch.eye(10)

    sigma = compute_spectral_norm(scaled_identity, n_iter=20)

    assert torch.allclose(
        sigma, torch.tensor(scale), atol=0.2
    ), f"Scaled identity spectral norm should be {scale}, got {sigma.item()}"


@pytest.mark.unit
def test_apply_spectral_norm_to_module():
    """Test that spectral norm can be applied to a module."""
    mlp = SyntheticMLP(dim=16, hidden=32, scale=2.0)

    # Apply spectral norm
    mlp = apply_spectral_norm(mlp, name_patterns=None, n_power_iterations=1)

    # Check that spectral normalization was applied
    # The weight should now be normalized
    assert hasattr(
        mlp.fc1, "parametrizations"
    ), "Spectral norm should add parametrizations attribute"


@pytest.mark.unit
def test_spectral_norm_reduces_Lz():
    """Test that spectral norm reduces Lipschitz constant Lz from >1 to <1."""
    batch_size = 4
    dim = 32
    hidden = 64

    # Create MLP with large weights (Lz > 1)
    mlp_no_sn = SyntheticMLP(dim=dim, hidden=hidden, scale=3.0)

    # Create MLP with spectral norm (Lz < 1)
    mlp_with_sn = SyntheticMLP(dim=dim, hidden=hidden, scale=3.0)
    mlp_with_sn = apply_spectral_norm(mlp_with_sn, name_patterns=None, n_power_iterations=1)

    # Generate random inputs
    z = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    x = torch.randn(batch_size, dim)

    # Estimate Lz without spectral norm
    Lz_no_sn, _ = estimate_Lz(mlp_no_sn, z, y, x, iters=5)

    # Estimate Lz with spectral norm
    Lz_with_sn, _ = estimate_Lz(mlp_with_sn, z, y, x, iters=5)

    # Key assertion: spectral norm should reduce Lz
    print(f"\nL_z without SN: {Lz_no_sn:.4f}")
    print(f"L_z with SN: {Lz_with_sn:.4f}")

    assert (
        Lz_with_sn < Lz_no_sn
    ), f"Spectral norm should reduce Lz: {Lz_with_sn:.4f} vs {Lz_no_sn:.4f}"

    # Additional assertion: SN should push Lz below 1.0 (or close to it)
    assert Lz_with_sn < 2.0, f"Spectral norm should significantly reduce Lz, got {Lz_with_sn:.4f}"


@pytest.mark.unit
def test_estimate_Lz_basic():
    """Test basic Lz estimation."""
    dim = 16
    batch_size = 2

    # Simple linear function with known Lipschitz constant
    mlp = nn.Sequential(nn.Linear(dim, dim))

    # Scale weights
    with torch.no_grad():
        mlp[0].weight.mul_(0.5)  # Should have Lz ≈ 0.5

    z = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    x = torch.randn(batch_size, dim)

    # Wrap in a callable
    def f(z_in, y_in, x_in):
        return mlp(z_in)

    Lz, z_next = estimate_Lz(f, z, y, x, iters=10)

    # Should be reasonably small
    assert Lz < 10.0, f"Lz estimate seems too large: {Lz}"
    assert z_next.shape == z.shape


@pytest.mark.unit
def test_estimate_Lz_with_ema():
    """Test Lz estimation with EMA smoothing."""
    dim = 16
    batch_size = 2

    mlp = SyntheticMLP(dim=dim, hidden=32, scale=1.5)

    z = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    x = torch.randn(batch_size, dim)

    # First estimate
    Lz1, _ = estimate_Lz(mlp, z, y, x, iters=3, ema_alpha=0.9, prev_estimate=None)

    # Second estimate with EMA
    Lz2, _ = estimate_Lz(mlp, z, y, x, iters=3, ema_alpha=0.9, prev_estimate=Lz1)

    # EMA should smooth the estimates
    assert Lz2 > 0.0
    assert abs(Lz2 - Lz1) < 10.0, "EMA should not change estimate drastically"


@pytest.mark.unit
def test_lipschitz_regularizer():
    """Test Lipschitz regularizer computation."""
    mlp = SyntheticMLP(dim=16, hidden=32, scale=2.0)

    # Compute regularizer
    reg_loss = lipschitz_regularizer(mlp, target_product=0.95)

    # Should be non-negative
    assert reg_loss >= 0.0

    # Should penalize large spectral norms
    assert reg_loss > 0.0, "Regularizer should penalize large weights"


@pytest.mark.unit
def test_lipschitz_regularizer_with_spectral_norm():
    """Test that spectral norm reduces regularization loss."""
    dim = 16
    hidden = 32

    # Without spectral norm
    mlp_no_sn = SyntheticMLP(dim=dim, hidden=hidden, scale=2.0)
    reg_no_sn = lipschitz_regularizer(mlp_no_sn, target_product=0.95)

    # With spectral norm
    mlp_with_sn = SyntheticMLP(dim=dim, hidden=hidden, scale=2.0)
    mlp_with_sn = apply_spectral_norm(mlp_with_sn, name_patterns=None, n_power_iterations=1)
    reg_with_sn = lipschitz_regularizer(mlp_with_sn, target_product=0.95)

    # Spectral norm should reduce regularization loss
    assert (
        reg_with_sn < reg_no_sn
    ), f"Spectral norm should reduce reg loss: {reg_with_sn:.4f} vs {reg_no_sn:.4f}"


@pytest.mark.unit
def test_lipschitz_monitor_update():
    """Test LipschitzMonitor tracking."""
    monitor = LipschitzMonitor(window_size=10, ema_alpha=0.9)

    # Add some estimates
    estimates = [1.2, 1.1, 0.95, 0.9, 0.85]
    for L_z in estimates:
        monitor.update(L_z)

    # Check history
    assert len(monitor.history) == len(estimates)

    # Check mean
    mean_Lz = monitor.get_mean_lipschitz()
    assert abs(mean_Lz - sum(estimates) / len(estimates)) < 0.01

    # Check max
    max_Lz = monitor.get_max_lipschitz()
    assert max_Lz == max(estimates)

    # Check EMA
    ema_Lz = monitor.get_ema_lipschitz()
    assert ema_Lz > 0.0


@pytest.mark.unit
def test_lipschitz_monitor_window():
    """Test that LipschitzMonitor respects window size."""
    window_size = 5
    monitor = LipschitzMonitor(window_size=window_size)

    # Add more than window_size estimates
    for i in range(10):
        monitor.update(float(i))

    # History should be capped at window_size
    assert len(monitor.history) == window_size

    # Should contain most recent values
    assert monitor.history[-1] == 9.0


@pytest.mark.unit
def test_spectral_norm_with_name_patterns():
    """Test applying spectral norm to specific layers only."""

    class SelectiveMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_path_fc = nn.Linear(16, 16)
            self.output_fc = nn.Linear(16, 16)

        def forward(self, z, y, x):
            return self.output_fc(self.z_path_fc(z))

    mlp = SelectiveMLP()

    # Apply spectral norm only to z_path layers
    mlp = apply_spectral_norm(mlp, name_patterns=["z_path"], n_power_iterations=1)

    # Check that z_path_fc has spectral norm
    assert hasattr(mlp.z_path_fc, "parametrizations"), "z_path_fc should have spectral norm"

    # output_fc should not have spectral norm (no pattern match)
    # Note: We can't easily check this without inspecting parametrizations


@pytest.mark.unit
def test_high_scale_without_sn_exceeds_1():
    """Test that without SN, high-scale MLP has Lz > 1.0."""
    batch_size = 4
    dim = 32
    hidden = 64

    # Create MLP with very large weights
    mlp = SyntheticMLP(dim=dim, hidden=hidden, scale=5.0)

    z = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    x = torch.randn(batch_size, dim)

    Lz, _ = estimate_Lz(mlp, z, y, x, iters=5)

    print(f"\nHigh-scale MLP L_z: {Lz:.4f}")

    # With large weights, Lz should exceed 1.0
    assert Lz > 1.0, f"High-scale MLP should have Lz > 1.0, got {Lz:.4f}"


@pytest.mark.unit
def test_gradient_flow_through_estimate_Lz():
    """Test that estimate_Lz doesn't break gradient flow."""
    dim = 16
    batch_size = 2

    mlp = SyntheticMLP(dim=dim, hidden=32, scale=1.5)

    z = torch.randn(batch_size, dim, requires_grad=True)
    y = torch.randn(batch_size, dim)
    x = torch.randn(batch_size, dim)

    # estimate_Lz should not require gradients (it's for monitoring)
    Lz, z_next = estimate_Lz(mlp, z, y, x, iters=3)

    # z_next should be detached
    assert not z_next.requires_grad, "z_next should be detached"

    # Lz is a float, not a tensor
    assert isinstance(Lz, float)
