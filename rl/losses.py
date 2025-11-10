"""RL loss functions for policy and value optimization.

This module implements various RL losses:
- Value Bellman Residual (BR) loss
- PPO clipped policy loss
- TRPO-style trust region loss
- Entropy regularization
"""

from typing import Any, Dict, Optional, Tuple

import torch


def value_bellman_residual_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_value: Optional[float] = None,
) -> torch.Tensor:
    """Compute value function Bellman residual loss.

    L_br = (V(s) - stop_grad(G_K))^2

    Minimizes squared error between value predictions and returns,
    optionally with clipping for stability.

    Args:
        values: Value predictions V(s) (B,)
        returns: Computed returns G_K (B,)
        clip_value: Optional value clipping threshold

    Returns:
        Value loss scalar
    """
    # Detach returns to stop gradient (stop_grad)
    targets = returns.detach()

    if clip_value is not None:
        # Clip value predictions for stability
        values = torch.clamp(values, -clip_value, clip_value)

    # MSE loss
    loss = ((values - targets) ** 2).mean()
    return loss


def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    entropy: Optional[torch.Tensor] = None,
    beta: float = 0.001,
) -> torch.Tensor:
    """Compute PPO clipped policy loss with optional entropy regularization.

    L^CLIP(θ) = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * H[π]

    Args:
        log_probs: Current policy log probs log π(a|s) (B,)
        old_log_probs: Old policy log probs log π_old(a|s) (B,)
        advantages: Advantage estimates A(s,a) (B,)
        epsilon: PPO clipping parameter (typically 0.2)
        entropy: Optional policy entropy for regularization (B,)
        beta: Entropy regularization coefficient (default: 0.001)

    Returns:
        PPO loss scalar
    """
    # Compute probability ratio: π(a|s) / π_old(a|s)
    ratio = (log_probs - old_log_probs).exp()

    # Unclipped objective
    unclipped = ratio * advantages

    # Clipped objective
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    # PPO loss: negative because we want to maximize the objective
    loss = -torch.min(unclipped, clipped).mean()

    # Entropy regularization (encourage exploration)
    if entropy is not None:
        loss -= beta * entropy.mean()

    return loss


def trpo_surrogate_loss(
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
) -> Any:
    """Compute TRPO surrogate objective.

    L^TRPO(θ) = E[ratio * A] where ratio = π(a|s) / π_old(a|s)

    Args:
        log_probs: Current policy log probs (B,)
        old_log_probs: Old policy log probs (B,)
        advantages: Advantage estimates (B,)

    Returns:
        TRPO surrogate loss scalar (to maximize, so return negative)
    """
    pass


def entropy_loss(
    log_probs: Any,
    entropy: Optional[Any] = None,
) -> Any:
    """Compute entropy regularization term.

    Encourages exploration by maximizing policy entropy.

    Args:
        log_probs: Log probabilities (B,)
        entropy: Optional precomputed entropy (B,)

    Returns:
        Negative entropy (minimize to maximize entropy)
    """
    pass


def kl_divergence(log_probs: Any, old_log_probs: Any) -> Any:
    """Compute KL divergence between old and new policy.

    KL(π_old || π) for trust region monitoring.

    Args:
        log_probs: Current policy log probs (B,)
        old_log_probs: Old policy log probs (B,)

    Returns:
        Mean KL divergence
    """
    pass


class RLLoss:
    """Combined RL loss with configurable components.

    Combines value loss, policy loss, and entropy regularization
    with configurable weights.

    Attributes:
        policy_loss_type: 'ppo' or 'trpo'
        value_coef: Weight for value loss
        entropy_coef: Weight for entropy regularization
        ppo_epsilon: PPO clipping parameter
    """

    def __init__(
        self,
        policy_loss_type: str = "ppo",
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epsilon: float = 0.2,
        clip_value: Optional[float] = None,
    ):
        """Initialize RL loss.

        Args:
            policy_loss_type: 'ppo' or 'trpo'
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy regularization
            ppo_epsilon: PPO clipping epsilon
            clip_value: Optional value clipping
        """
        self.policy_loss_type = policy_loss_type
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epsilon = ppo_epsilon
        self.clip_value = clip_value

    def __call__(
        self,
        log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
        values: Any,
        returns: Any,
        entropy: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """Compute combined RL loss.

        Args:
            log_probs: Current policy log probs
            old_log_probs: Old policy log probs
            advantages: Advantages
            values: Value predictions
            returns: Target returns
            entropy: Policy entropy

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        pass
