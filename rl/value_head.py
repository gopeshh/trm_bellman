"""Value head with EMA target network for stable bootstrapping.

This module implements Vψ (value function) with an exponential moving average (EMA)
target network for computing stable value targets during training.
"""

from copy import deepcopy

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Value function network that estimates state values.

    The value function Vψ(x, z) estimates the expected return from
    a given state, used for advantage estimation and bootstrapping.

    Uses an EMA target network for stable value targets as in Eq. (3).

    Attributes:
        z_dim: Dimension of internal state representations
        x_dim: Dimension of input
        hidden: Dimension of hidden layers
        net: Online value network
        target: Target network (EMA of online network)
    """

    def __init__(self, z_dim: int, x_dim: int, hidden: int = 256):
        """Initialize value head with EMA target.

        Args:
            z_dim: Dimension of internal state representations
            x_dim: Dimension of input
            hidden: Dimension of hidden layers (default: 256)
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + x_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.target = deepcopy(self.net)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, z_n: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute state value using online network.

        Args:
            z_n: Internal state tensor (B, z_dim)
            x: Input tensor (B, x_dim)

        Returns:
            Value estimate (B,)
        """
        return self.net(torch.cat([z_n, x], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def target_value(self, z_n: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using target network for stable bootstrapping.

        Args:
            z_n: Internal state tensor (B, z_dim)
            x: Input tensor (B, x_dim)

        Returns:
            Target value estimate (B,)
        """
        return self.target(torch.cat([z_n, x], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau: float = 0.995):
        """Update target network using exponential moving average.

        Updates target parameters:
            θ_target = tau * θ_target + (1 - tau) * θ_online

        Args:
            tau: EMA decay rate (default: 0.995)
        """
        for p, tp in zip(self.net.parameters(), self.target.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)
