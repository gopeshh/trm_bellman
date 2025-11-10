"""Spectral norm utilities and Lipschitz constant monitoring.

This module provides tools for enforcing and monitoring contraction properties
in neural networks through spectral normalization and Lipschitz constant tracking.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def apply_spectral_norm(
    module: nn.Module,
    name_patterns: Optional[List[str]] = None,
    n_power_iterations: int = 1,
) -> nn.Module:
    """Apply spectral normalization to layers matching name patterns.

    Constrains the spectral norm (largest singular value) of weight
    matrices to be at most 1, enforcing Lipschitz continuity on the z→z path.

    Args:
        module: Module to apply spectral norm to
        name_patterns: List of name patterns to match (e.g., ["z_path", "recurrent"])
                      If None, applies to all Linear and Conv layers
        n_power_iterations: Number of power iterations for estimation

    Returns:
        Module with spectral norm applied
    """
    for name, child in module.named_modules():
        # Check if this module should have spectral norm applied
        should_apply = False

        if name_patterns is None:
            # Apply to all Linear and Conv layers
            if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                should_apply = True
        else:
            # Apply only if name matches one of the patterns
            for pattern in name_patterns:
                if pattern in name:
                    if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        should_apply = True
                        break

        if should_apply:
            # Apply spectral normalization
            spectral_norm(child, name="weight", n_power_iterations=n_power_iterations)

    return module


def compute_spectral_norm(weight: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Compute spectral norm (largest singular value) of weight matrix.

    Uses power iteration method for efficient estimation.

    Args:
        weight: Weight tensor (out_features, in_features)
        n_iter: Number of power iterations

    Returns:
        Spectral norm (scalar tensor)
    """
    # Reshape weight for matrix operations
    if weight.dim() > 2:
        weight = weight.reshape(weight.size(0), -1)

    # Initialize random vector
    u = torch.randn(weight.size(0), device=weight.device, dtype=weight.dtype)
    u = u / u.norm()

    # Power iteration
    for _ in range(n_iter):
        v = weight.t() @ u
        v = v / v.norm()
        u = weight @ v
        u = u / u.norm()

    # Compute spectral norm: σ_max = u^T W v
    sigma = (u @ weight @ v).item()
    return torch.tensor(sigma, device=weight.device)


def estimate_Lz(
    f: nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    iters: int = 3,
    ema_alpha: float = 0.9,
    prev_estimate: Optional[float] = None,
) -> Tuple[float, torch.Tensor]:
    """Estimate Lipschitz constant L_z of f using power iteration on Jacobian.

    Uses Jacobian-vector products via autograd to estimate the largest singular
    value of ∂f/∂z, which bounds the Lipschitz constant.

    Args:
        f: Function/module f_θ(z, y, x) -> z'
        z: Internal state tensor (B, z_dim) with requires_grad=True
        y: Current output (B, y_dim)
        x: Input (B, x_dim)
        iters: Number of power iterations (default: 3)
        ema_alpha: EMA smoothing factor (default: 0.9)
        prev_estimate: Previous L_z estimate for EMA

    Returns:
        Tuple of (L_z_estimate, z_next)
    """
    # Ensure z requires gradients
    z = z.detach().requires_grad_(True)

    # Forward pass
    z_next = f(z, y, x)

    # Initialize random vector for power iteration
    v = torch.randn_like(z)
    v = v / v.norm()

    # Power iteration on Jacobian
    for _ in range(iters):
        # Compute Jacobian-vector product: J @ v
        (jvp,) = torch.autograd.grad(
            outputs=z_next,
            inputs=z,
            grad_outputs=v,
            create_graph=False,
            retain_graph=True,
        )

        # Normalize
        v = jvp / (jvp.norm() + 1e-8)

    # Compute final J @ v to get eigenvalue estimate
    (jvp,) = torch.autograd.grad(
        outputs=z_next,
        inputs=z,
        grad_outputs=v,
        create_graph=False,
        retain_graph=False,
    )

    # Estimate of largest singular value
    L_z_estimate = jvp.norm().item()

    # Apply EMA if previous estimate exists
    if prev_estimate is not None:
        L_z_estimate = ema_alpha * prev_estimate + (1 - ema_alpha) * L_z_estimate

    return L_z_estimate, z_next.detach()


def lipschitz_regularizer(
    module: nn.Module,
    target_product: float = 0.95,
) -> torch.Tensor:
    """Compute Lipschitz regularization loss.

    R_lip = Σ max(0, ||W||_2 - s_l)

    Penalizes spectral norms that exceed target values to encourage contraction.

    Args:
        module: Module with spectral-normalized layers
        target_product: Target cumulative product of spectral norms (default: 0.95)

    Returns:
        Regularization loss
    """
    reg_loss = torch.tensor(0.0, device=next(module.parameters()).device)
    count = 0

    for name, child in module.named_modules():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Get weight (possibly spectral normalized)
            weight = child.weight

            # Compute spectral norm
            sigma = compute_spectral_norm(weight, n_iter=5)

            # Target per-layer: target_product^(1/num_layers)
            # For simplicity, use target_product directly as threshold
            violation = torch.clamp(sigma - target_product, min=0.0)
            reg_loss = reg_loss + violation
            count += 1

    return reg_loss / max(count, 1)


class SpectralNormMonitor:
    """Monitor and log spectral norms of network layers.

    Tracks the spectral norms (Lipschitz constants) of weight matrices
    throughout training to ensure contraction properties.

    Attributes:
        modules: Dictionary of modules to monitor
        history: History of spectral norms
    """

    def __init__(self):
        """Initialize spectral norm monitor."""
        self.modules = {}
        self.history = {}

    def register_module(self, name: str, module: Any):
        """Register a module for monitoring.

        Args:
            name: Name identifier for module
            module: Module to monitor
        """
        pass

    def compute_all_norms(self) -> Dict[str, float]:
        """Compute spectral norms for all registered modules.

        Returns:
            Dictionary mapping module names to spectral norms
        """
        pass

    def log_norms(self, step: int):
        """Log current spectral norms and update history.

        Args:
            step: Current training step
        """
        pass

    def get_max_norm(self) -> float:
        """Get maximum spectral norm across all modules.

        Returns:
            Maximum spectral norm
        """
        pass


class LipschitzMonitor:
    """Monitor Lipschitz constant Lz of the meta-MDP dynamics.

    Tracks empirical Lipschitz constant of the z→z mapping using
    Jacobian power iteration estimates.

    Attributes:
        window_size: Size of moving average window
        history: History of Lz measurements
        ema_estimate: Current EMA estimate
        ema_alpha: EMA smoothing factor
    """

    def __init__(self, window_size: int = 1000, ema_alpha: float = 0.9):
        """Initialize Lipschitz monitor.

        Args:
            window_size: Size of moving average window
            ema_alpha: EMA smoothing factor
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.history: List[float] = []
        self.ema_estimate: Optional[float] = None

    def update(self, L_z: float):
        """Update monitor with new L_z estimate.

        Args:
            L_z: Lipschitz constant estimate
        """
        self.history.append(L_z)

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Update EMA
        if self.ema_estimate is None:
            self.ema_estimate = L_z
        else:
            self.ema_estimate = self.ema_alpha * self.ema_estimate + (1 - self.ema_alpha) * L_z

    def get_mean_lipschitz(self) -> float:
        """Get mean Lipschitz constant over recent history.

        Returns:
            Mean Lz
        """
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def get_max_lipschitz(self) -> float:
        """Get maximum Lipschitz constant over recent history.

        Returns:
            Max Lz
        """
        if not self.history:
            return 0.0
        return max(self.history)

    def get_ema_lipschitz(self) -> float:
        """Get EMA Lipschitz estimate.

        Returns:
            EMA Lz
        """
        return self.ema_estimate if self.ema_estimate is not None else 0.0


def enforce_lipschitz_constraint(
    module: Any,
    lipschitz_constant: float = 1.0,
    method: str = "spectral_norm",
):
    """Enforce Lipschitz constraint on module.

    Args:
        module: Module to constrain
        lipschitz_constant: Desired Lipschitz constant
        method: Method to use ('spectral_norm', 'weight_clipping', 'gradient_penalty')
    """
    pass


def gradient_penalty(
    model: Any,
    real_data: Any,
    fake_data: Any,
    lambda_gp: float = 10.0,
) -> Any:
    """Compute gradient penalty for Lipschitz constraint.

    Used in WGAN-GP style regularization.

    Args:
        model: Model to compute gradient penalty for
        real_data: Real data samples
        fake_data: Generated/fake data samples
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty loss
    """
    pass
