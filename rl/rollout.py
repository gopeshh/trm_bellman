"""K-step rollout with bootstrapping for trajectory collection.

This module handles collecting K-step trajectories from the meta-MDP
using the current policy, with value bootstrapping for finite horizons.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@torch.no_grad()
def rollout_k(
    policy: Callable,
    value_target: Callable,
    env: Any,
    s0: Tuple[torch.Tensor, torch.Tensor],
    K: int,
    gamma: float,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Perform K-step rollout with bootstrapped target value.

    Collects a K-step trajectory using the policy, accumulates discounted
    rewards, and bootstraps with the target value function.

    The return is computed as:
        G = Σ_{t=0}^{K-1} γ^t * r_t + γ^K * V_target(s_K)

    Args:
        policy: Policy with sample(*s) -> (action, log_prob) method
        value_target: Target value function with (*s) -> value method
        env: Environment with step(s, a) -> (s', r, done, info) method
        s0: Initial state (x, y)
        K: Number of rollout steps
        gamma: Discount factor

    Returns:
        Tuple of (G, s) where:
            - G: Bootstrapped return (B,)
            - s: Final state (x, y)
    """
    s = s0
    G = 0.0
    pow_gamma = 1.0

    for _ in range(K):
        a, _ = policy.sample(*s)
        s_prime, r, done, info = env.step(s, a, s[0])  # env.step needs x
        G = G + pow_gamma * r
        pow_gamma *= gamma
        s = s_prime

        # Early termination if episode ends
        if done.any():
            break

    # Bootstrap with target value
    G = G + pow_gamma * value_target(*s)

    return G, s


class RolloutBuffer:
    """Buffer for storing trajectory data during rollouts.

    Stores states, actions, rewards, values, log_probs for K-step rollouts.

    Attributes:
        capacity: Maximum number of transitions to store
        device: Device for tensor storage
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        """Initialize rollout buffer.

        Args:
            capacity: Maximum buffer size
            device: Device for tensors
        """
        self.capacity = capacity
        self.device = device
        self.clear()

    def clear(self):
        """Clear all stored data."""
        pass

    def add(
        self,
        state: Any,
        action: Any,
        reward: Any,
        value: Any,
        log_prob: Any,
        done: Any,
    ):
        """Add transition to buffer.

        Args:
            state: State tensor
            action: Action tensor
            reward: Reward tensor
            value: Value estimate
            log_prob: Log probability of action
            done: Done flag
        """
        pass

    def get(self) -> Dict[str, Any]:
        """Retrieve all stored data as dictionary.

        Returns:
            Dictionary with keys: states, actions, rewards, values, log_probs, dones
        """
        pass

    def __len__(self) -> int:
        """Return number of stored transitions."""
        pass


class Rollout:
    """K-step rollout collector with value bootstrapping.

    Performs K-step rollouts in the meta-MDP, collecting trajectories
    for policy optimization with bootstrapped returns.

    Attributes:
        meta_mdp: Meta-MDP environment
        policy: Policy network
        value_fn: Value function network
        K: Number of rollout steps
    """

    def __init__(
        self,
        meta_mdp: Any,
        policy: Any,
        value_fn: Any,
        K: int = 16,
        gamma: float = 0.99,
    ):
        """Initialize rollout collector.

        Args:
            meta_mdp: Meta-MDP environment
            policy: Policy network for action selection
            value_fn: Value function for bootstrapping
            K: Number of steps per rollout
            gamma: Discount factor
        """
        self.meta_mdp = meta_mdp
        self.policy = policy
        self.value_fn = value_fn
        self.K = K
        self.gamma = gamma

    def collect(self, x: Any, num_rollouts: int = 1) -> RolloutBuffer:
        """Collect K-step rollouts.

        Args:
            x: Input tensor (B, input_dim)
            num_rollouts: Number of rollouts to collect

        Returns:
            RolloutBuffer with collected trajectories
        """
        pass

    def collect_single(self, x: Any, state: Any) -> Tuple[List[Any], ...]:
        """Collect single K-step rollout from given state.

        Args:
            x: Input tensor
            state: Initial state

        Returns:
            Tuple of (states, actions, rewards, values, log_probs, dones)
        """
        pass

    def bootstrap_returns(
        self,
        rewards: Any,
        values: Any,
        dones: Any,
        next_value: Any,
    ) -> Any:
        """Compute bootstrapped K-step returns.

        Args:
            rewards: Reward sequence (T, B)
            values: Value estimates (T, B)
            dones: Done flags (T, B)
            next_value: Bootstrap value (B,)

        Returns:
            Bootstrapped returns (T, B)
        """
        pass


def compute_k_step_returns(
    rewards: Any,
    bootstrap_value: Any,
    gamma: float = 0.99,
    dones: Optional[Any] = None,
) -> Any:
    """Compute K-step returns with bootstrapping.

    Args:
        rewards: Rewards (T, B)
        bootstrap_value: Value to bootstrap from (B,)
        gamma: Discount factor
        dones: Optional done flags (T, B)

    Returns:
        K-step returns (T, B)
    """
    pass
