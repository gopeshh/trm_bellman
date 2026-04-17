"""
Advantage Actor-Critic (A2C) baseline trainer for plan-space RL.

This module implements A2C for comparison with UPI-TRM. It's a simpler
baseline than PPO with single-step updates and no clipping.

Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from rl.batch_utils import prepare_batch_x, prepare_plan, normalize_puzzle_id
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.value_targets import compute_gae_trajectory, compute_td_advantage


@dataclass
class A2CConfig:
    """Configuration for A2C training."""

    # Core A2C hyperparameters
    vf_coef: float = 0.5  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping norm

    # Training schedule
    num_steps: int = 5  # Steps per rollout before update (smaller than PPO)

    # Advantage estimation
    gamma: float = 0.99  # Discount factor
    use_gae: bool = True  # Use GAE or simple TD advantage
    gae_lambda: float = 0.95  # GAE lambda (if use_gae=True)
    normalize_advantages: bool = False  # Normalize advantages (less common in A2C)

    # Learning rates
    lr: float = 7e-4  # Single learning rate for simplicity

    # Model
    inner_unroll_n: int = 4  # Latent unroll steps (for TRM backbone)

    # Logging
    log_interval: int = 10
    eval_interval: int = 50

    # Training
    num_train_steps: int = 10000


@dataclass
class A2CRolloutBuffer:
    """Simple buffer for A2C rollouts."""

    x_list: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    y_list: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    action_masks: List[Optional[torch.Tensor]] = field(default_factory=list)

    def add(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a transition to the buffer."""
        self.x_list.append(x)
        self.y_list.append(y)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_masks.append(action_mask)

    def clear(self) -> None:
        """Clear all stored data."""
        self.x_list.clear()
        self.y_list.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class A2CTrainer:
    """
    Advantage Actor-Critic (A2C) trainer for plan-space RL.

    A2C is a simpler synchronous variant of A3C. It performs a single gradient
    update after collecting a small batch of experience (typically 5 steps).

    Key differences from PPO:
    - No clipping - uses vanilla policy gradient
    - Single gradient step per rollout (no epochs)
    - Typically smaller rollout sizes

    Key differences from UPI-TRM:
    - No CPI mixture updates
    - No exact baseline summation
    - No theory-exact features

    Args:
        model: Neural network model with policy_dist() and used_value() methods
        env: PlanEditEnv environment
        config: A2CConfig with hyperparameters
        device: PyTorch device
    """

    def __init__(
        self,
        model: nn.Module,
        env: PlanEditEnv,
        config: A2CConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.env = env
        self.config = config
        self.device = device

        self.rollout_buffer = A2CRolloutBuffer()

        # Training state
        self._train_step_count = 0
        self._episode_count = 0
        self.term_stats = {"stop": 0, "solved": 0, "budget": 0}

        # Single optimizer for all parameters
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=config.lr,
            alpha=0.99,
            eps=1e-5,
        )

        # Current episode state
        self._current_x: Optional[Dict[str, torch.Tensor]] = None
        self._current_y: Optional[torch.Tensor] = None
        self._episode_rewards: List[float] = []

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

    def _stack_scalar_buffer(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Stack scalar or singleton tensors into a flat 1D tensor."""
        return torch.stack(tensors).reshape(-1)

    def collect_rollouts(self, num_steps: int) -> torch.Tensor:
        """
        Collect num_steps of experience using current policy.

        Returns:
            last_value: Bootstrap value for the final state
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
                action = dist.sample().reshape(())
                value = value.reshape(())

            # Step environment
            (x_next, y_next), reward, done, info = self.env.step(action.item())
            self._episode_rewards.append(reward)

            # Store transition
            self.rollout_buffer.add(
                x=self._clone_state(x),
                y=self._clone_state(y),
                action=action.cpu(),
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

        # Get bootstrap value for last state
        with torch.no_grad():
            batch_x = self._prepare_batch_x(self._current_x)
            batch_y = self._prepare_plan(self._current_y)
            last_value, _ = self.model.used_value(
                batch_x, batch_y,
                n=self.config.inner_unroll_n
            )

        return last_value.cpu()

    def update(self, last_value: torch.Tensor) -> Dict[str, float]:
        """
        Perform A2C update on collected rollouts.

        Args:
            last_value: Bootstrap value for final state

        Returns:
            Dictionary with loss statistics
        """
        self.model.train()

        # Convert rollout data to tensors
        rewards = torch.tensor(self.rollout_buffer.rewards, dtype=torch.float32)
        values = self._stack_scalar_buffer(self.rollout_buffer.values)
        dones = torch.tensor(self.rollout_buffer.dones, dtype=torch.bool)
        actions = self._stack_scalar_buffer(self.rollout_buffer.actions).to(self.device)

        # Compute advantages
        if self.config.use_gae:
            advantages = compute_gae_trajectory(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                last_value=last_value.item(),
            )
            returns = advantages + values
        else:
            # Simple n-step return
            returns = self._compute_n_step_returns(rewards, dones, last_value.item())
            advantages = returns - values

        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages (optional)
        if self.config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare batch data
        x_batch = self._stack_x_batch(self.rollout_buffer.x_list)
        y_batch = torch.stack(self.rollout_buffer.y_list).to(self.device)

        # Get action mask batch
        if self.rollout_buffer.action_masks[0] is not None:
            action_mask = torch.stack(self.rollout_buffer.action_masks).to(self.device)
        else:
            action_mask = None

        # Forward pass
        dist, _ = self.model.policy_dist(
            x_batch, y_batch,
            n=self.config.inner_unroll_n,
            action_mask=action_mask
        )
        new_values, _ = self.model.used_value(x_batch, y_batch, n=self.config.inner_unroll_n)
        new_values = new_values.reshape(-1)

        # Compute log probs and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy loss (vanilla policy gradient)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns)

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

        self._train_step_count += 1

        return {
            "loss_policy": policy_loss.item(),
            "loss_value": value_loss.item(),
            "entropy": entropy.item(),
            "loss_total": loss.item(),
        }

    def _compute_n_step_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        last_value: float,
    ) -> torch.Tensor:
        """Compute n-step returns with bootstrapping."""
        T = len(rewards)
        returns = torch.zeros(T, dtype=torch.float32)

        R = last_value
        for t in reversed(range(T)):
            R = rewards[t] + self.config.gamma * R * (1 - dones[t].float())
            returns[t] = R

        return returns

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
        Perform one A2C training step: collect rollouts + update.

        Returns:
            Dictionary with training statistics
        """
        # Collect rollouts and get bootstrap value
        last_value = self.collect_rollouts(self.config.num_steps)

        # Update policy and value function
        update_stats = self.update(last_value)

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

    def evaluate_policy_metrics(
        self,
        env_cfg: PlanEditEnvConfig,
        dataset: Any,
        checker: Any,
        num_episodes: int = 50,
    ) -> Dict[str, float]:
        """
        Evaluate the policy using greedy rollouts, returning success rate and mean score.

        This method provides the same interface as UPITrmTrainer.evaluate_policy_metrics()
        for fair comparison between algorithms.

        Args:
            env_cfg: Environment configuration
            dataset: Dataset providing puzzle instances
            checker: Checker function (x, y) -> score
            num_episodes: Number of evaluation episodes (default 50)

        Returns:
            Dict with:
                - mean_score: Average final checker score
                - success_rate: Fraction of episodes that reached max score
                - eval_policy_mode: "greedy"
                - solved_count: Number of solved episodes
                - total_episodes: Total evaluation episodes
        """
        from rl.evaluator import evaluate_plan_policy_with_scores

        mean_score, success_rate, detailed_stats = evaluate_plan_policy_with_scores(
            model=self.model,
            dataset=dataset,
            checker=checker,
            env_cfg=env_cfg,
            num_episodes=num_episodes,
            inner_unroll_n=self.config.inner_unroll_n,
            episodic_latent=True,  # Baselines use episodic latent
            greedy=True,  # Always greedy for deterministic evaluation
        )

        result = {
            "mean_score": mean_score,
            "success_rate": success_rate,
            "eval_policy_mode": "greedy",
        }
        result.update(detailed_stats)
        return result
