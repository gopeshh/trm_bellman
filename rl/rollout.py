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
    s0: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    K: int,
    gamma: float,
    f_theta: Optional[Callable] = None,
    n_inner: int = 6,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Perform K-step rollout with bootstrapped target value.

    Collects a K-step trajectory using the policy, accumulates discounted
    rewards, and bootstraps with the target value function.

    The return is computed as:
        G = Σ_{t=0}^{K-1} γ^t * r_t + γ^K * V_target(s_K)

    Args:
        policy: Policy with sample(y, z_n, x) -> (action, log_prob) method
        value_target: Target value function with (z_n, x) -> value method
        env: Environment with step(s, a, x) -> (s', r, done, info) method
        s0: Initial state (x, y, z_n) tuple
        K: Number of rollout steps
        gamma: Discount factor
        f_theta: Optional model for recomputing z_n after edits
        n_inner: Number of inner reasoning steps for z_n recomputation

    Returns:
        Tuple of (G, s) where:
            - G: Bootstrapped return (B,)
            - s: Final state (x, y, z_n)
    """
    x, y, z_n = s0
    G = 0.0
    pow_gamma = 1.0

    for _ in range(K):
        # Sample action from policy with correct signature: (y, z_n, x)
        a, _ = policy.sample(y, z_n, x)

        # Take environment step (returns (x, y') state without z_n)
        s_prime_xy, r, done, info = env.step((x, y), a, x)
        x_prime, y_prime = s_prime_xy

        # Recompute z_n for new state if model provided
        if f_theta is not None:
            z_n_prime = f_theta(y_prime, x_prime, n=n_inner)
        else:
            # If no model, keep previous z_n (not ideal but allows testing)
            z_n_prime = z_n

        # Accumulate discounted reward
        G = G + pow_gamma * r
        pow_gamma *= gamma

        # Update state
        x, y, z_n = x_prime, y_prime, z_n_prime

        # Early termination if episode ends
        if done.any():
            break

    # Bootstrap with target value using correct signature: (z_n, x)
    G = G + pow_gamma * value_target(z_n, x)

    return G, (x, y, z_n)


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
