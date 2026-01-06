"""
Deep Q-Network (DQN) and Double DQN baseline trainer for plan-space RL.

This module implements DQN and Double DQN for comparison with UPI-TRM.
Unlike PPO/A2C which are actor-critic methods, DQN is a pure value-based
method that learns Q(s, a) directly.

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from rl.batch_utils import prepare_batch_x, prepare_plan, normalize_puzzle_id
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig


@dataclass
class DQNConfig:
    """Configuration for DQN/Double DQN training."""

    # Core DQN hyperparameters
    double_dqn: bool = True  # Use Double DQN (recommended)
    gamma: float = 0.99  # Discount factor

    # Exploration
    epsilon_start: float = 1.0  # Initial exploration rate
    epsilon_end: float = 0.01  # Final exploration rate
    epsilon_decay_steps: int = 5000  # Steps to decay epsilon

    # Replay buffer
    buffer_size: int = 10000  # Replay buffer capacity
    batch_size: int = 64  # Minibatch size for training
    min_buffer_size: int = 500  # Minimum transitions before training starts

    # Target network
    target_update_freq: int = 100  # Steps between target network updates
    soft_update_tau: float = 0.0  # If > 0, use soft update instead of hard copy

    # Learning
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0  # Gradient clipping norm

    # Model
    inner_unroll_n: int = 4  # Latent unroll steps (for TRM backbone)

    # Training schedule
    train_freq: int = 4  # Train every N steps
    gradient_steps: int = 1  # Number of gradient updates per training
    num_train_steps: int = 10000

    # Logging
    log_interval: int = 100
    eval_interval: int = 500


@dataclass
class Transition:
    """A single transition in the replay buffer."""
    x: Dict[str, torch.Tensor]
    y: torch.Tensor
    action: int
    reward: float
    x_next: Dict[str, torch.Tensor]
    y_next: torch.Tensor
    done: bool
    action_mask: Optional[torch.Tensor] = None
    next_action_mask: Optional[torch.Tensor] = None


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and provides random sampling for training.
    Uses a deque for efficient O(1) append/pop operations.
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-network that wraps the base model to output Q-values for all actions.

    For TRM backbone: Uses the latent state z^(n) to compute Q(s, a) for each action.
    For NoRec backbone: Uses the encoder output directly.

    The Q-network outputs a vector of Q-values for all actions, enabling efficient
    action selection and target computation.
    """

    def __init__(self, base_model: nn.Module, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        self.num_actions = num_actions

        # Compute input dimension for Q-head
        # For TRM: z_H shape is [batch, seq_len, hidden_size], so flattened is seq_len * hidden_size
        if hasattr(base_model, 'config'):
            if hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            elif hasattr(base_model.config, 'hidden_dim'):
                hidden_size = base_model.config.hidden_dim
            else:
                hidden_size = hidden_dim

            # For TRM, we need seq_len * hidden_size
            if hasattr(base_model.config, 'seq_len'):
                input_dim = base_model.config.seq_len * hidden_size
            else:
                input_dim = hidden_size
        else:
            input_dim = hidden_dim

        # Store for potential debugging
        self._input_dim = input_dim

        # Q-value head: maps flattened latent to Q(s, a) for each action
        # Use a smaller intermediate layer since input is large
        intermediate_dim = min(256, input_dim // 2) if input_dim > 256 else hidden_dim
        self.q_head = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_actions),
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        n: int = 4,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions.

        Args:
            x: State dict with puzzle tensors
            y: Current plan tensor
            n: Number of latent unroll steps
            action_mask: Optional mask for valid actions (1=valid, 0=invalid)

        Returns:
            Q-values tensor of shape [batch_size, num_actions]
        """
        # Get latent representation from base model
        if hasattr(self.base_model, 'unroll_latent'):
            # TRM backbone: use latent state
            z_carry, _ = self.base_model.unroll_latent(x, y, n=n)
            # TRM returns a carry object with z_H attribute
            if hasattr(z_carry, 'z_H'):
                z = z_carry.z_H
            else:
                z = z_carry  # Fallback if it's already a tensor
        elif hasattr(self.base_model, 'encode'):
            # NoRec backbone: use encoder output
            z = self.base_model.encode(x, y)
        else:
            raise ValueError("Model must have unroll_latent() or encode() method")

        # Flatten z for Q-head input (z might be [batch, seq, hidden])
        if z.dim() == 3:
            z = z.view(z.shape[0], -1)  # Flatten to [batch, seq*hidden]
        elif z.dim() == 2:
            pass  # Already [batch, hidden]
        else:
            raise ValueError(f"Unexpected z dimensionality: {z.dim()}")

        # Compute Q-values
        q_values = self.q_head(z)

        # Mask invalid actions with large negative values
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0).expand(q_values.shape[0], -1)
            # Set invalid action Q-values to very negative
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)

        return q_values

    def get_latent(self, x: Dict[str, torch.Tensor], y: torch.Tensor, n: int = 4) -> torch.Tensor:
        """Get the latent representation for inspection/debugging."""
        if hasattr(self.base_model, 'unroll_latent'):
            z_carry, _ = self.base_model.unroll_latent(x, y, n=n)
            # TRM returns a carry object with z_H attribute
            if hasattr(z_carry, 'z_H'):
                z = z_carry.z_H
            else:
                z = z_carry
        elif hasattr(self.base_model, 'encode'):
            z = self.base_model.encode(x, y)
        else:
            raise ValueError("Model must have unroll_latent() or encode() method")

        if z.dim() == 3:
            z = z.view(z.shape[0], -1)
        return z


class DQNTrainer:
    """
    Deep Q-Network trainer for plan-space RL.

    This trainer implements DQN and Double DQN for comparison with UPI-TRM.
    It uses experience replay and a target network for stable training.

    Key differences from UPI-TRM:
    - Value-based method (learns Q(s,a) directly, no separate policy network)
    - Epsilon-greedy exploration instead of stochastic policy
    - Experience replay for sample efficiency
    - Target network for stable bootstrapping

    Args:
        model: Neural network model (TRM or NoRec encoder)
        env: PlanEditEnv environment
        config: DQNConfig with hyperparameters
        device: PyTorch device
    """

    def __init__(
        self,
        model: nn.Module,
        env: PlanEditEnv,
        config: DQNConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.config = config
        self.device = device

        # Number of actions from environment
        # PlanEditEnv uses stop_action_id convention (STOP is last action)
        if hasattr(env, 'stop_action_id') and env.stop_action_id is not None:
            self.num_actions = env.stop_action_id + 1
        else:
            raise ValueError(
                "DQNTrainer requires env.stop_action_id to be set. "
                "Call env.set_stop_action_id() before creating the trainer."
            )

        # Create Q-networks
        self.q_network = QNetwork(model, self.num_actions).to(device)

        # Create target network (copy of Q-network)
        self.target_network = QNetwork(
            self._clone_base_model(model),
            self.num_actions
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
        )

        # Training state
        self._train_step_count = 0
        self._env_step_count = 0
        self._episode_count = 0
        self._epsilon = config.epsilon_start
        self.term_stats = {"stop": 0, "solved": 0, "budget": 0}

        # Current episode state
        self._current_x: Optional[Dict[str, torch.Tensor]] = None
        self._current_y: Optional[torch.Tensor] = None
        self._current_action_mask: Optional[torch.Tensor] = None
        self._episode_rewards: List[float] = []

    def _clone_base_model(self, model: nn.Module) -> nn.Module:
        """Clone the base model for the target network."""
        import copy
        return copy.deepcopy(model)

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

    def _get_epsilon(self) -> float:
        """Get current exploration rate with linear decay."""
        progress = min(1.0, self._env_step_count / self.config.epsilon_decay_steps)
        epsilon = self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )
        return epsilon

    def select_action(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        greedy: bool = False,
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            x: State dict
            y: Current plan
            action_mask: Optional mask for valid actions
            greedy: If True, always select best action (for evaluation)

        Returns:
            Selected action index
        """
        epsilon = 0.0 if greedy else self._get_epsilon()

        if random.random() < epsilon:
            # Random action (respecting mask)
            if action_mask is not None:
                valid_actions = torch.where(action_mask.bool())[0]
                if len(valid_actions) > 0:
                    return random.choice(valid_actions.tolist())
            return random.randint(0, self.num_actions - 1)

        # Greedy action from Q-network
        batch_x = self._prepare_batch_x(x)
        batch_y = self._prepare_plan(y)

        with torch.no_grad():
            q_values = self.q_network(
                batch_x, batch_y,
                n=self.config.inner_unroll_n,
                action_mask=action_mask.to(self.device) if action_mask is not None else None,
            )
            return q_values.argmax(dim=-1).item()

    def collect_step(self) -> bool:
        """
        Collect one step of experience.

        Returns:
            True if episode ended, False otherwise
        """
        # Initialize episode if needed
        if self._current_x is None:
            self._current_x, self._current_y = self.env.reset()
            self._current_action_mask = self.env.get_action_mask()
            self._episode_rewards = []

        x = self._current_x
        y = self._current_y
        action_mask = self._current_action_mask

        # Select action
        action = self.select_action(x, y, action_mask)

        # Step environment
        (x_next, y_next), reward, done, info = self.env.step(action)
        next_action_mask = self.env.get_action_mask() if not done else None

        self._episode_rewards.append(reward)
        self._env_step_count += 1

        # Store transition
        transition = Transition(
            x=self._clone_state(x),
            y=self._clone_state(y),
            action=action,
            reward=reward,
            x_next=self._clone_state(x_next),
            y_next=self._clone_state(y_next),
            done=done,
            action_mask=action_mask.cpu() if action_mask is not None else None,
            next_action_mask=next_action_mask.cpu() if next_action_mask is not None else None,
        )
        self.replay_buffer.add(transition)

        # Update state
        self._current_x = x_next
        self._current_y = y_next
        self._current_action_mask = next_action_mask

        # Handle episode end
        if done:
            self._episode_count += 1
            reason = info.get("done_reason", "unknown")
            if reason in self.term_stats:
                self.term_stats[reason] += 1

            # Reset for next episode
            self._current_x, self._current_y = self.env.reset()
            self._current_action_mask = self.env.get_action_mask()
            self._episode_rewards = []
            return True

        return False

    def train_batch(self) -> Dict[str, float]:
        """
        Perform one gradient update on a batch from replay buffer.

        Returns:
            Dictionary with loss statistics
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {"loss_q": 0.0, "mean_q": 0.0}

        self.q_network.train()

        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Prepare batch tensors
        batch_x = self._stack_x_batch([t.x for t in batch])
        batch_y = torch.stack([t.y for t in batch]).to(self.device)
        batch_actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        batch_rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        batch_dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)

        batch_x_next = self._stack_x_batch([t.x_next for t in batch])
        batch_y_next = torch.stack([t.y_next for t in batch]).to(self.device)

        # Handle action masks
        if batch[0].next_action_mask is not None:
            batch_next_masks = torch.stack([
                t.next_action_mask if t.next_action_mask is not None
                else torch.ones(self.num_actions)
                for t in batch
            ]).to(self.device)
        else:
            batch_next_masks = None

        # Compute current Q-values
        current_q = self.q_network(batch_x, batch_y, n=self.config.inner_unroll_n)
        current_q = current_q.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use online network for action selection, target for evaluation
                next_q_online = self.q_network(
                    batch_x_next, batch_y_next,
                    n=self.config.inner_unroll_n,
                    action_mask=batch_next_masks,
                )
                best_actions = next_q_online.argmax(dim=-1, keepdim=True)

                next_q_target = self.target_network(
                    batch_x_next, batch_y_next,
                    n=self.config.inner_unroll_n,
                    action_mask=batch_next_masks,
                )
                next_q = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                # Standard DQN: use target network for both
                next_q_target = self.target_network(
                    batch_x_next, batch_y_next,
                    n=self.config.inner_unroll_n,
                    action_mask=batch_next_masks,
                )
                next_q = next_q_target.max(dim=-1).values

            # Zero out Q-values for terminal states
            next_q = next_q.masked_fill(batch_dones, 0.0)

            # Compute TD target
            target_q = batch_rewards + self.config.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn_utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss_q": loss.item(),
            "mean_q": current_q.mean().item(),
            "mean_target_q": target_q.mean().item(),
            "epsilon": self._get_epsilon(),
        }

    def update_target_network(self) -> None:
        """Update target network (hard copy or soft update)."""
        if self.config.soft_update_tau > 0:
            # Soft update: θ_target = τ * θ_online + (1 - τ) * θ_target
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.config.soft_update_tau * online_param.data +
                    (1 - self.config.soft_update_tau) * target_param.data
                )
        else:
            # Hard update: copy weights directly
            self.target_network.load_state_dict(self.q_network.state_dict())

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
        Perform one DQN training step: collect experience + train.

        Returns:
            Dictionary with training statistics
        """
        # Collect experience
        for _ in range(self.config.train_freq):
            self.collect_step()

        # Train on replay buffer
        train_stats = {}
        for _ in range(self.config.gradient_steps):
            stats = self.train_batch()
            for k, v in stats.items():
                train_stats[k] = train_stats.get(k, 0) + v / self.config.gradient_steps

        self._train_step_count += 1

        # Update target network periodically
        if self._train_step_count % self.config.target_update_freq == 0:
            self.update_target_network()

        return train_stats

    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return {
            "train_step": self._train_step_count,
            "env_steps": self._env_step_count,
            "episodes": self._episode_count,
            "epsilon": self._get_epsilon(),
            "buffer_size": len(self.replay_buffer),
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
        Evaluate the policy using greedy rollouts.

        Args:
            env_cfg: Environment configuration
            dataset: Dataset providing puzzle instances
            checker: Checker function (x, y) -> score
            num_episodes: Number of evaluation episodes

        Returns:
            Dict with mean_score, success_rate, etc.
        """
        self.q_network.eval()

        # Create evaluation environment
        eval_env = PlanEditEnv(
            dataset=dataset,
            checker=checker,
            config=env_cfg,
        )
        # Set stop action ID from main env
        eval_env.set_stop_action_id(self.env.stop_action_id)

        solved_count = 0
        total_scores = []

        for _ in range(num_episodes):
            x, y = eval_env.reset()
            done = False

            while not done:
                action_mask = eval_env.get_action_mask()
                action = self.select_action(x, y, action_mask, greedy=True)
                (x, y), reward, done, info = eval_env.step(action)

            final_score = checker(x, y)
            total_scores.append(final_score)

            # Check if solved (max score for Sudoku is 10.0)
            if final_score >= 10.0 - 1e-6:
                solved_count += 1

        mean_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        success_rate = solved_count / num_episodes if num_episodes > 0 else 0.0

        self.q_network.train()

        return {
            "mean_score": mean_score,
            "success_rate": success_rate,
            "eval_policy_mode": "greedy",
            "solved_count": solved_count,
            "total_episodes": num_episodes,
            "score_min": min(total_scores) if total_scores else 0.0,
            "score_max": max(total_scores) if total_scores else 0.0,
        }
