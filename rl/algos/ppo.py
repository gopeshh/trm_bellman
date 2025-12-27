"""
Proximal Policy Optimization (PPO) baseline trainer for plan-space RL.

This module implements PPO for comparison with UPI-TRM. It reuses the same
environment, model interfaces, and value targets as UPI-TRM.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from rl.batch_utils import state_is_batched, prepare_batch_x, prepare_plan, normalize_puzzle_id
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.value_targets import compute_gae_trajectory


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Core PPO hyperparameters
    clip_eps: float = 0.2  # Clipping epsilon for policy ratio
    vf_coef: float = 0.5  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping norm

    # PPO training schedule
    num_steps: int = 128  # Steps per rollout before update
    num_epochs: int = 4  # Epochs per PPO update
    num_minibatches: int = 4  # Minibatches per epoch

    # Value function
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    normalize_advantages: bool = True  # Normalize advantages per minibatch
    clip_vf_loss: bool = False  # Clip value function loss (optional)

    # Learning rates
    policy_lr: float = 3e-4
    value_lr: float = 1e-4

    # Model
    inner_unroll_n: int = 4  # Latent unroll steps (for TRM backbone)

    # Logging
    log_interval: int = 10
    eval_interval: int = 50

    # Training
    num_train_steps: int = 10000
    batch_size: int = 64  # For minibatch sampling


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data collected by PPO."""

    # State data
    x_list: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    y_list: List[torch.Tensor] = field(default_factory=list)

    # Action and reward data
    actions: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)

    # Action masks for invalid action handling
    action_masks: List[Optional[torch.Tensor]] = field(default_factory=list)

    def add(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a transition to the buffer."""
        self.x_list.append(x)
        self.y_list.append(y)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_masks.append(action_mask)

    def clear(self) -> None:
        """Clear all stored data."""
        self.x_list.clear()
        self.y_list.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for plan-space RL.

    This trainer implements the standard PPO algorithm for comparison with UPI-TRM.
    It can be used with either the TRM backbone or a simpler no-recursion encoder.

    Key differences from UPI-TRM:
    - Uses clipped surrogate objective instead of CPI mixture
    - Multiple epochs over collected data
    - Standard GAE for advantage estimation (no exact baseline summation)
    - No theory-exact features (no contraction enforcement, etc.)

    Args:
        model: Neural network model with policy_dist() and used_value() methods
        env: PlanEditEnv environment
        config: PPOConfig with hyperparameters
        device: PyTorch device
    """

    def __init__(
        self,
        model: nn.Module,
        env: PlanEditEnv,
        config: PPOConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.env = env
        self.config = config
        self.device = device

        self.rollout_buffer = RolloutBuffer()

        # Training state
        self._train_step_count = 0
        self._episode_count = 0
        self.term_stats = {"stop": 0, "solved": 0, "budget": 0}

        # Optimizer for all model parameters
        self.optimizer = torch.optim.Adam([
            {"params": self._get_policy_params(), "lr": config.policy_lr},
            {"params": self._get_value_params(), "lr": config.value_lr},
        ])

        # Current episode state
        self._current_x: Optional[Dict[str, torch.Tensor]] = None
        self._current_y: Optional[torch.Tensor] = None
        self._episode_rewards: List[float] = []

    def _get_policy_params(self) -> List[nn.Parameter]:
        """Get policy head parameters."""
        params = []
        for name, param in self.model.named_parameters():
            if "edit_policy" in name and param.requires_grad:
                params.append(param)
        return params if params else list(self.model.parameters())

    def _get_value_params(self) -> List[nn.Parameter]:
        """Get value head and backbone parameters."""
        params = []
        for name, param in self.model.named_parameters():
            if "edit_policy" not in name and param.requires_grad:
                params.append(param)
        return params if params else []

    def _prepare_batch_x(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare state dict for model input."""
        return prepare_batch_x(x, self.device, batched=False)

    def _prepare_plan(self, y: Any) -> torch.Tensor:
        """Prepare plan tensor for model input."""
        return prepare_plan(y, self.device, batched=False)

    def _clone_state(self, state: Any) -> Any:
        """Clone state tensors for storage."""
        if isinstance(state, dict):
            return {k: (v.clone().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}
        if torch.is_tensor(state):
            return state.clone().cpu()
        return state

    def collect_rollouts(self, num_steps: int) -> None:
        """
        Collect num_steps of experience using current policy.

        Stores transitions in self.rollout_buffer for later training.
        """
        self.model.eval()
        self.rollout_buffer.clear()

        # Initialize episode if needed
        if self._current_x is None:
            self._current_x, self._current_y = self.env.reset()
            self._episode_rewards = []

        for _ in range(num_steps):
            x = self._current_x
            y = self._current_y

            # Prepare inputs
            batch_x = self._prepare_batch_x(x)
            batch_y = self._prepare_plan(y)

            # Get action mask
            action_mask = self.env.get_action_mask()
            if action_mask is not None:
                action_mask = action_mask.to(self.device)

            # Get policy distribution and value
            with torch.no_grad():
                dist, _ = self.model.policy_dist(
                    batch_x, batch_y,
                    n=self.config.inner_unroll_n,
                    action_mask=action_mask
                )
                value, _ = self.model.used_value(
                    batch_x, batch_y,
                    n=self.config.inner_unroll_n
                )

                # Sample action
                action = dist.sample().squeeze()
                log_prob = dist.log_prob(action)

            # Step environment
            (x_next, y_next), reward, done, info = self.env.step(action.item())
            self._episode_rewards.append(reward)

            # Store transition
            self.rollout_buffer.add(
                x=self._clone_state(x),
                y=self._clone_state(y),
                action=action.cpu(),
                log_prob=log_prob.cpu(),
                reward=reward,
                done=done,
                value=value.cpu(),
                action_mask=action_mask.cpu() if action_mask is not None else None,
            )

            # Update state
            self._current_x = x_next
            self._current_y = y_next

            # Handle episode end
            if done:
                self._episode_count += 1
                reason = info.get("done_reason", "unknown")
                if reason in self.term_stats:
                    self.term_stats[reason] += 1

                # Reset for next episode
                self._current_x, self._current_y = self.env.reset()
                self._episode_rewards = []

    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns for the rollout buffer.

        Returns:
            returns: [num_steps] tensor of discounted returns
            advantages: [num_steps] tensor of GAE advantages
        """
        rewards = torch.tensor(self.rollout_buffer.rewards, dtype=torch.float32)
        values = torch.stack(self.rollout_buffer.values).squeeze()
        dones = torch.tensor(self.rollout_buffer.dones, dtype=torch.bool)

        # Get bootstrap value for last state
        with torch.no_grad():
            batch_x = self._prepare_batch_x(self._current_x)
            batch_y = self._prepare_plan(self._current_y)
            last_value, _ = self.model.used_value(
                batch_x, batch_y,
                n=self.config.inner_unroll_n
            )
            last_value = last_value.cpu().item()

        # Compute GAE
        advantages = compute_gae_trajectory(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            last_value=last_value,
        )

        returns = advantages + values

        return returns, advantages

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected rollouts.

        Returns:
            Dictionary with loss statistics
        """
        self.model.train()

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Get old log probs
        old_log_probs = torch.stack(self.rollout_buffer.log_probs).to(self.device)
        actions = torch.stack(self.rollout_buffer.actions).to(self.device)
        old_values = torch.stack(self.rollout_buffer.values).squeeze().to(self.device)

        # Prepare batch data
        num_steps = len(self.rollout_buffer)
        batch_size = num_steps // self.config.num_minibatches
        batch_size = max(batch_size, 1)

        # Track losses across epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.config.num_epochs):
            # Random permutation for minibatch sampling
            indices = torch.randperm(num_steps)

            for start in range(0, num_steps, batch_size):
                end = min(start + batch_size, num_steps)
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_values = old_values[mb_indices]

                # Normalize advantages
                if self.config.normalize_advantages and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Prepare state batch
                mb_x = self._stack_x_batch([self.rollout_buffer.x_list[i] for i in mb_indices])
                mb_y = torch.stack([self.rollout_buffer.y_list[i] for i in mb_indices]).to(self.device)

                # Get action mask batch
                mb_action_masks = [self.rollout_buffer.action_masks[i] for i in mb_indices]
                if mb_action_masks[0] is not None:
                    mb_action_mask = torch.stack(mb_action_masks).to(self.device)
                else:
                    mb_action_mask = None

                # Forward pass
                dist, _ = self.model.policy_dist(
                    mb_x, mb_y,
                    n=self.config.inner_unroll_n,
                    action_mask=mb_action_mask
                )
                values, _ = self.model.used_value(mb_x, mb_y, n=self.config.inner_unroll_n)

                # Compute log probs and entropy
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.config.clip_vf_loss:
                    # Clipped value loss (optional)
                    v_clipped = mb_old_values + torch.clamp(
                        values.squeeze() - mb_old_values,
                        -self.config.clip_eps,
                        self.config.clip_eps
                    )
                    vf_loss1 = F.mse_loss(values.squeeze(), mb_returns)
                    vf_loss2 = F.mse_loss(v_clipped, mb_returns)
                    value_loss = torch.max(vf_loss1, vf_loss2)
                else:
                    value_loss = F.mse_loss(values.squeeze(), mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        self._train_step_count += 1

        return {
            "loss_policy": total_policy_loss / max(num_updates, 1),
            "loss_value": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "num_updates": num_updates,
        }

    def _stack_x_batch(self, x_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack a list of x dicts into a batched dict."""
        inputs = torch.stack([x["inputs"] for x in x_list], dim=0).to(self.device)
        puzzle_ids = torch.stack(
            [normalize_puzzle_id(x["puzzle_identifiers"]) for x in x_list], dim=0
        ).to(self.device)
        if puzzle_ids.dim() > 1:
            puzzle_ids = puzzle_ids.squeeze(-1)
        return {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    def train_step(self) -> Dict[str, float]:
        """
        Perform one PPO training step: collect rollouts + update.

        Returns:
            Dictionary with training statistics
        """
        # Collect rollouts
        self.collect_rollouts(self.config.num_steps)

        # Update policy and value function
        update_stats = self.update()

        return update_stats

    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return {
            "train_step": self._train_step_count,
            "episodes": self._episode_count,
            "term_stop": self.term_stats["stop"],
            "term_solved": self.term_stats["solved"],
            "term_budget": self.term_stats["budget"],
        }
