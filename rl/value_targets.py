"""
Value target computation for UPI-TRM.

This module provides functions for computing:
- K-step bootstrapped targets (Section 5 of paper)
- Generalized Advantage Estimation (GAE)
- Empirical Bellman residual monitoring
"""

from typing import Dict, Optional

import torch


def compute_k_step_bootstrapped_target(
    rewards_K: torch.Tensor,
    dones_K: torch.Tensor,
    steps_taken: torch.Tensor,
    v_K: torch.Tensor,
    gamma: float,
    K: int,
    exact_k_step_targets: bool = False,
    C_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute K-step bootstrapped targets with proper terminal masking.

    Implements the K-step value operator T_K^π from Section 5:
        G^(K)(s_0) = Σ_{k=0}^{K-1} γ^k r_k + γ^K V(s_K)

    Terminal rewards produced by PlanEditEnv already fold in the absorbing
    tail, so terminal samples do not bootstrap. ``C_max`` is retained only for
    backward-compatible callers and has no effect.

    Args:
        rewards_K: [batch_size, K] rewards for each step
        dones_K: [batch_size, K] done flags for each step
        steps_taken: [batch_size] actual number of steps taken (1 to K)
        v_K: [batch_size] V(s_K) bootstrap values
        gamma: Discount factor
        K: Maximum horizon
        exact_k_step_targets: If True, use γ^K (theory-exact).
                              If False, use γ^steps_taken (practical).
        C_max: Deprecated compatibility argument. Terminal rewards already
               include the return-equivalent absorbing-tail correction.

    Returns:
        [batch_size] K-step bootstrapped targets
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"`gamma` must lie in [0,1), got {gamma!r}.")

    batch_size = rewards_K.shape[0]

    # Mask rewards beyond steps_taken for each sample (defensive against improper padding)
    step_indices = torch.arange(K, device=rewards_K.device).unsqueeze(0)  # [1, K]
    valid_mask = step_indices < steps_taken.unsqueeze(1)  # [batch_size, K]
    rewards_K = rewards_K * valid_mask

    gammas = rewards_K.new_tensor([gamma**k for k in range(K)])
    reward_returns = (rewards_K * gammas).sum(dim=1)

    final_idx = (steps_taken - 1).clamp(min=0)
    batch_indices = torch.arange(batch_size, device=rewards_K.device)
    done_final = dones_K[batch_indices, final_idx]

    if exact_k_step_targets:
        incomplete = (steps_taken < K) & (~done_final)
        if bool(incomplete.any().item()):
            raise ValueError(
                "A fixed-K target requires K transitions unless the segment terminates."
            )

    bootstrap_factor = gamma ** steps_taken.to(v_K.dtype)
    v_bootstrap = torch.where(done_final, torch.zeros_like(v_K), v_K)
    return reward_returns + bootstrap_factor * v_bootstrap


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE) for a batch of independent transitions.

    This implements a single-step approximation of GAE(λ) from Schulman et al. (2016).
    For independent transitions (not full trajectories), we compute:
        A = δ * (1 + γλ(1-done))
    where δ = r + γV(s') - V(s) is the TD error.

    This approximation assumes the next advantage is approximately equal to the current
    TD error, which provides a λ-weighted blend between TD(0) and a 2-step lookahead.
    - When λ=0: Returns pure TD error (δ)
    - When λ=1: Returns δ * (1 + γ) for non-terminal states (approximates 2-step return)

    For proper multi-step GAE with full trajectory information, use compute_gae_trajectory.

    Args:
        rewards: [B] rewards for each transition
        values: [B] V(s) for each starting state
        next_values: [B] V(s') for each next state
        dones: [B] boolean done flags
        gamma: Discount factor
        gae_lambda: GAE λ parameter (0 = TD, 1 = Monte Carlo-like)

    Returns:
        advantages: [B] GAE-approximated advantages
    """
    mask = (~dones).float()
    next_values_masked = torch.where(dones, torch.zeros_like(next_values), next_values)
    td_error = rewards + gamma * next_values_masked - values
    
    # For single transitions, we approximate the GAE recursion:
    # A_t = δ_t + γλ * A_{t+1}
    # By assuming A_{t+1} ≈ δ_t (the TD error is a reasonable proxy),
    # we get: A_t ≈ δ_t * (1 + γλ) for non-terminal transitions.
    # This provides meaningful λ-blending without requiring full trajectories.
    gae_factor = 1.0 + gamma * gae_lambda * mask
    return td_error * gae_factor


def compute_gae_trajectory(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute GAE advantages for a full trajectory.

    This is the standard GAE computation that requires sequential trajectory data.
    The recursion is:
        A_t = δ_t + γλ(1-done_t) * A_{t+1}
    where δ_t = r_t + γ(1-done_t)V(s_{t+1}) - V(s_t)

    Args:
        rewards: [T] rewards for each timestep
        values: [T] V(s) for each state in trajectory
        dones: [T] boolean done flags
        gamma: Discount factor
        gae_lambda: GAE λ parameter
        last_value: V(s_T) for the final state after trajectory

    Returns:
        advantages: [T] GAE advantages
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    # Append last_value for bootstrapping
    values_extended = torch.cat([values, torch.tensor([last_value], device=values.device, dtype=values.dtype)])

    for t in reversed(range(T)):
        bootstrap_value = torch.where(
            dones[t],
            torch.zeros_like(values_extended[t + 1]),
            values_extended[t + 1],
        )
        next_gae = torch.where(dones[t], torch.zeros_like(gae), gae)
        delta = rewards[t] + gamma * bootstrap_value - values[t]
        gae = delta + gamma * gae_lambda * next_gae
        advantages[t] = gae

    return advantages


def compute_empirical_bellman_residual(
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> Dict[str, float]:
    """
    Compute empirical Bellman residual statistics for monitoring.

    The Bellman residual is |V(s) - (r + γV(s'))|, which measures how well
    the value function satisfies the Bellman equation.

    Args:
        values: [B] V(s) predictions
        rewards: [B] rewards
        next_values: [B] V(s') predictions
        dones: [B] done flags
        gamma: Discount factor

    Returns:
        Dictionary with residual statistics (mean, max, std)
    """
    next_values_masked = torch.where(dones, torch.zeros_like(next_values), next_values)
    td_target = rewards + gamma * next_values_masked
    residual = (values - td_target).abs()

    return {
        "bellman_residual_mean": float(residual.mean().item()),
        "bellman_residual_max": float(residual.max().item()),
        "bellman_residual_std": float(residual.std(unbiased=False).item()),
    }


def compute_td_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    centered: bool = False,
) -> torch.Tensor:
    """
    Compute simple 1-step TD advantage: A(s,a) = r + γV(s') - V(s).
    
    Args:
        rewards: [B] rewards
        values: [B] V(s) for starting states
        next_values: [B] V(s') for next states
        dones: [B] done flags
        gamma: Discount factor
        centered: If True, subtract batch mean for variance reduction
        
    Returns:
        advantages: [B] TD advantages
    """
    next_values_masked = torch.where(dones, torch.zeros_like(next_values), next_values)
    td_target = rewards + gamma * next_values_masked
    adv = td_target - values
    
    if centered:
        adv = adv - adv.mean()
    
    return adv
