"""Advantage estimation with centering for variance reduction.

This module implements centered advantages using K-step returns or
Generalized Advantage Estimation (GAE) for policy gradient methods.
"""

from typing import Any, Optional

import torch


def compute_advantages(
    returns: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """Compute advantages from returns and baseline.

    A(s,a) = Q(s,a) - V(s) = returns - baseline

    Args:
        returns: Computed returns (B,) or (T, B)
        baseline: Value estimates V(s) (B,) or (T, B)

    Returns:
        Advantages (B,) or (T, B)
    """
    return returns - baseline


def compute_gae(
    rewards: Any,
    values: Any,
    bootstrap_value: Any,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    dones: Optional[Any] = None,
) -> Any:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a bias-variance tradeoff controlled by lambda:
    - lambda=0: one-step TD (low variance, high bias)
    - lambda=1: Monte Carlo (high variance, low bias)

    Args:
        rewards: Reward sequence (T, B)
        values: Value estimates V(s_t) (T, B)
        bootstrap_value: Bootstrap value V(s_{T+1}) (B,)
        gamma: Discount factor
        lambda_: GAE lambda parameter (typically 0.95)
        dones: Optional done flags (T, B)

    Returns:
        GAE advantages (T, B)
    """
    pass


def center_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Center and normalize advantages for stability.

    Normalizes advantages to have zero mean and unit variance across batch.

    Args:
        advantages: Advantage tensor (T, B) or (B,)
        eps: Small constant for numerical stability

    Returns:
        Centered and normalized advantages
    """
    mean = advantages.mean()
    std = advantages.std() + eps
    return (advantages - mean) / std


def compute_returns(
    rewards: Any,
    bootstrap_value: Any,
    gamma: float = 0.99,
    dones: Optional[Any] = None,
) -> Any:
    """Compute discounted returns with bootstrapping.

    R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^K*V(s_{t+K})

    Args:
        rewards: Reward sequence (T, B)
        bootstrap_value: Bootstrap value (B,)
        gamma: Discount factor
        dones: Optional done flags (T, B)

    Returns:
        Discounted returns (T, B)
    """
    pass


class AdvantageEstimator:
    """Wrapper for advantage estimation with configurable method.

    Supports both K-step returns and GAE with centering.

    Attributes:
        method: 'k-step' or 'gae'
        gamma: Discount factor
        lambda_: GAE lambda (only for GAE method)
        center: Whether to center advantages
    """

    def __init__(
        self,
        method: str = "gae",
        gamma: float = 0.99,
        lambda_: float = 0.95,
        center: bool = True,
    ):
        """Initialize advantage estimator.

        Args:
            method: 'k-step' or 'gae'
            gamma: Discount factor
            lambda_: GAE lambda parameter
            center: Whether to center and normalize advantages
        """
        self.method = method
        self.gamma = gamma
        self.lambda_ = lambda_
        self.center = center

    def __call__(
        self,
        rewards: Any,
        values: Any,
        bootstrap_value: Any,
        dones: Optional[Any] = None,
    ) -> Any:
        """Compute advantages using configured method.

        Args:
            rewards: Reward sequence (T, B)
            values: Value estimates (T, B)
            bootstrap_value: Bootstrap value (B,)
            dones: Optional done flags (T, B)

        Returns:
            Computed advantages (T, B)
        """
        pass
