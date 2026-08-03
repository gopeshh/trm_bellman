"""
Deep Q-Network (DQN) and Double DQN baseline trainer for plan-space RL.

This module implements DQN and Double DQN for comparison with UPI-TRM.
Unlike PPO/A2C which are actor-critic methods, DQN is a pure value-based
method that learns Q(s, a) directly.

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from rl.batch_utils import prepare_batch_x, prepare_plan, stack_batch_states
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_utils import sudoku_is_solved, sudoku_get_stats


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
    target_update_freq: int = 100  # Env steps between target network updates
    soft_update_tau: float = 0.0  # If > 0, use soft update instead of hard copy

    # Learning
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0  # Gradient clipping norm
    dqn_n_step: int = 1  # n-step TD target horizon (1 reproduces vanilla DQN)

    # Model
    inner_unroll_n: int = 4  # Latent unroll steps (for TRM backbone)

    # Training schedule
    train_freq: int = 4  # Train every N steps
    gradient_steps: int = 1  # Number of gradient updates per training
    num_train_steps: int = 10000

    # Logging
    log_interval: int = 100
    eval_interval: int = 500

    # Evaluation
    # Episode count used by evaluate_policy_metrics() when the caller does
    # not pass an explicit num_episodes; build_trainer() should set this
    # from rl_cfg.eval_num_episodes so UPI-TRM and baselines stay in sync.
    eval_num_episodes: int = 50


def compute_epsilon_decay_steps(
    num_train_steps: int,
    exploration_fraction: float,
    train_freq: int,
) -> int:
    """Convert the configured exploration fraction into environment-step units."""
    return int(num_train_steps * train_freq * exploration_fraction)


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
    bootstrap_steps: int = 1


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions in a flat ring buffer and supports n-step aggregation
    at sampling time while respecting episode boundaries.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.size = 0
        self.next_index = 0
        self._next_in_episode: List[Optional[int]] = [None] * capacity
        self._current_episode_last_index: Optional[int] = None
        self._current_episode_indices: List[int] = []

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        idx = self.next_index
        self.buffer[idx] = transition
        self._next_in_episode[idx] = None

        if idx in self._current_episode_indices:
            self._current_episode_indices = [
                current_idx for current_idx in self._current_episode_indices if current_idx != idx
            ]
            if self._current_episode_last_index == idx:
                self._current_episode_last_index = (
                    self._current_episode_indices[-1] if self._current_episode_indices else None
                )

        if self._current_episode_last_index is not None:
            self._next_in_episode[self._current_episode_last_index] = idx
        self._current_episode_indices.append(idx)
        self._current_episode_last_index = idx

        if self.size < self.capacity:
            self.size += 1
        self.next_index = (self.next_index + 1) % self.capacity

        if transition.done:
            self._current_episode_last_index = None
            self._current_episode_indices = []

    def clear(self) -> None:
        """Reset the replay buffer."""
        self.buffer = [None] * self.capacity
        self.size = 0
        self.next_index = 0
        self._next_in_episode = [None] * self.capacity
        self._current_episode_last_index = None
        self._current_episode_indices = []

    def _valid_index_upper_bound(self) -> int:
        return self.capacity if self.size == self.capacity else self.size

    def _current_episode_unready_indices(self, n_step: int) -> set[int]:
        # Overwrites happen oldest-first, so any completed episode retained in
        # the ring buffer is kept as a suffix of that episode's time order.
        # That means a stored transition cannot have an in-episode successor
        # that was overwritten before it. The only forward links that may
        # still be unavailable are the tail links of the current in-progress
        # episode, which are blocked here for n-step sampling.
        if n_step <= 1 or not self._current_episode_indices:
            return set()
        return set(self._current_episode_indices[-(n_step - 1):])

    def build_n_step_transition(
        self,
        index: int,
        n_step: int,
        gamma: float,
    ) -> Transition:
        """Construct an n-step aggregate transition from a stored index."""
        if n_step <= 0:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        transition = self.buffer[index]
        if transition is None:
            raise IndexError(f"ReplayBuffer slot {index} is empty")

        total_reward = 0.0
        discount = 1.0
        steps = 0
        current_index: Optional[int] = index
        final_transition = transition

        while current_index is not None and steps < n_step:
            current_transition = self.buffer[current_index]
            if current_transition is None:
                break
            total_reward += discount * float(current_transition.reward)
            final_transition = current_transition
            steps += 1
            if current_transition.done:
                break
            current_index = self._next_in_episode[current_index]
            discount *= gamma

        return Transition(
            x=transition.x,
            y=transition.y,
            action=transition.action,
            reward=total_reward,
            x_next=final_transition.x_next,
            y_next=final_transition.y_next,
            done=final_transition.done,
            action_mask=transition.action_mask,
            next_action_mask=final_transition.next_action_mask,
            bootstrap_steps=steps,
        )

    def sample(
        self,
        batch_size: int,
        n_step: int = 1,
        gamma: float = 1.0,
    ) -> List[Transition]:
        """Sample a random batch of transitions with optional n-step aggregation."""
        if n_step <= 0:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        if self.size == 0:
            return []

        upper_bound = self._valid_index_upper_bound()
        blocked_indices = self._current_episode_unready_indices(n_step)
        ready_count = max(0, upper_bound - len(blocked_indices))
        if ready_count == 0:
            return []

        sample_size = min(batch_size, ready_count)
        if sample_size == ready_count:
            sampled_indices = [
                idx
                for idx in range(upper_bound)
                if self.buffer[idx] is not None and idx not in blocked_indices
            ]
        else:
            sampled_set = set()
            while len(sampled_set) < sample_size:
                idx = random.randrange(upper_bound)
                if idx in blocked_indices or self.buffer[idx] is None:
                    continue
                sampled_set.add(idx)
            sampled_indices = list(sampled_set)
        return [
            self.build_n_step_transition(idx, n_step=n_step, gamma=gamma)
            for idx in sampled_indices
        ]

    def __len__(self) -> int:
        return self.size


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
        self.base_model: Any = base_model
        self.num_actions = num_actions
        self._uses_trm_latent = (
            hasattr(base_model, "unroll_latent")
            and hasattr(base_model, "inner")
            and hasattr(base_model, "config")
            and hasattr(base_model.config, "seq_len")
        )

        # Compute input dimension for Q-head.
        # - TRM exposes the flattened latent state z_H with length seq_len + puzzle_emb_len.
        # - NoRec encoders already collapse to a single hidden vector.
        model_config: Any = getattr(base_model, "config", None)
        if model_config is not None:
            if hasattr(model_config, 'hidden_size'):
                hidden_size = int(model_config.hidden_size)
            elif hasattr(model_config, 'hidden_dim'):
                hidden_size = int(model_config.hidden_dim)
            else:
                hidden_size = hidden_dim

            if self._uses_trm_latent:
                puzzle_emb_len = 0
                inner = getattr(base_model, 'inner', None)
                if inner is not None:
                    puzzle_emb_len = int(getattr(inner, 'puzzle_emb_len', 0) or 0)
                input_dim = (int(model_config.seq_len) + puzzle_emb_len) * hidden_size
            elif hasattr(base_model, 'encode'):
                input_dim = hidden_size
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
        if self._uses_trm_latent:
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
                if action_mask.numel() != self.num_actions:
                    raise ValueError(
                        f"action mask has {action_mask.numel()} entries, "
                        f"expected {self.num_actions}"
                    )
                action_mask = action_mask.unsqueeze(0).expand(q_values.shape[0], -1)
            elif (
                action_mask.dim() != 2
                or tuple(action_mask.shape) != tuple(q_values.shape)
            ):
                raise ValueError(
                    "action mask must have shape [num_actions] or "
                    f"[batch_size, num_actions]; got {tuple(action_mask.shape)}, "
                    f"expected ({q_values.shape[0]}, {self.num_actions})"
                )
            action_mask = action_mask.to(device=q_values.device, dtype=torch.bool)
            invalid_rows = ~action_mask.any(dim=-1)
            if bool(invalid_rows.any().item()):
                rows = torch.nonzero(invalid_rows, as_tuple=False).reshape(-1).tolist()
                raise RuntimeError(
                    f"action mask has no valid actions for batch rows {rows}"
                )
            # Set invalid action Q-values to very negative
            q_values = q_values.masked_fill(~action_mask, -1e9)

        return q_values

    def get_latent(self, x: Dict[str, torch.Tensor], y: torch.Tensor, n: int = 4) -> torch.Tensor:
        """Get the latent representation for inspection/debugging."""
        if self._uses_trm_latent:
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
        self._last_target_update_env_step = 0
        self._episode_count = 0
        self._epsilon = config.epsilon_start
        self.term_stats = {"stop": 0, "solved": 0, "budget": 0}

        if self.config.dqn_n_step < 1:
            raise ValueError(f"dqn_n_step must be >= 1, got {self.config.dqn_n_step}")

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

        validated_mask = action_mask
        if action_mask is not None:
            if action_mask.dim() == 2 and action_mask.shape[0] == 1:
                validated_mask = action_mask.reshape(-1)
            elif action_mask.dim() != 1:
                raise ValueError(
                    "Single-state action mask must have shape [num_actions] or "
                    f"[1, num_actions], got {tuple(action_mask.shape)}"
                )
            if validated_mask.numel() != self.num_actions:
                raise ValueError(
                    f"action mask has {validated_mask.numel()} entries, "
                    f"expected {self.num_actions}"
                )
            validated_mask = validated_mask.to(dtype=torch.bool)
            if not bool(validated_mask.any().item()):
                raise RuntimeError("action mask has no valid actions")

        if random.random() < epsilon:
            # Random action (respecting mask)
            if validated_mask is not None:
                valid_actions = torch.where(validated_mask)[0]
                return random.choice(valid_actions.tolist())
            return random.randint(0, self.num_actions - 1)

        # Greedy action from Q-network
        batch_x = self._prepare_batch_x(x)
        batch_y = self._prepare_plan(y)

        with torch.no_grad():
            q_values = self.q_network(
                batch_x, batch_y,
                n=self.config.inner_unroll_n,
                action_mask=(
                    validated_mask.to(self.device)
                    if validated_mask is not None
                    else None
                ),
            )
            return int(q_values.argmax(dim=-1).item())

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
        if x is None or y is None:
            raise RuntimeError("DQN episode state was not initialized.")

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
        batch = self.replay_buffer.sample(
            self.config.batch_size,
            n_step=self.config.dqn_n_step,
            gamma=self.config.gamma,
        )
        if not batch:
            return {"loss_q": 0.0, "mean_q": 0.0, "mean_target_q": 0.0, "epsilon": self._get_epsilon()}

        # Prepare batch tensors
        batch_x = self._stack_x_batch([t.x for t in batch])
        batch_y = torch.stack([t.y for t in batch]).to(self.device)
        batch_actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        batch_rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        batch_dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
        batch_bootstrap_steps = torch.tensor(
            [t.bootstrap_steps for t in batch], dtype=torch.long, device=self.device
        )

        batch_x_next = self._stack_x_batch([t.x_next for t in batch])
        batch_y_next = torch.stack([t.y_next for t in batch]).to(self.device)

        # Handle action masks
        if any(t.action_mask is not None for t in batch):
            batch_action_masks = torch.stack([
                t.action_mask if t.action_mask is not None
                else torch.ones(self.num_actions, dtype=torch.bool)
                for t in batch
            ]).to(self.device)
        else:
            batch_action_masks = None

        if any(t.next_action_mask is not None for t in batch):
            batch_next_masks = torch.stack([
                t.next_action_mask if t.next_action_mask is not None
                else torch.ones(self.num_actions, dtype=torch.bool)
                for t in batch
            ]).to(self.device)
        else:
            batch_next_masks = None

        # Compute current Q-values
        current_q = self.q_network(
            batch_x,
            batch_y,
            n=self.config.inner_unroll_n,
            action_mask=batch_action_masks,
        )
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
            discounts = torch.pow(
                torch.full_like(batch_rewards, self.config.gamma),
                batch_bootstrap_steps.to(batch_rewards.dtype),
            )
            target_q = batch_rewards + discounts * next_q

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
        return stack_batch_states(x_list, self.device)

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
        train_stats: Dict[str, float] = {}
        for _ in range(self.config.gradient_steps):
            stats = self.train_batch()
            for k, v in stats.items():
                train_stats[k] = train_stats.get(k, 0.0) + v / self.config.gradient_steps

        self._train_step_count += 1

        # Update target network periodically in env-step units.
        if (
            self._env_step_count - self._last_target_update_env_step
            >= self.config.target_update_freq
        ):
            self.update_target_network()
            self._last_target_update_env_step = self._env_step_count

        return train_stats

    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return {
            "train_step": float(self._train_step_count),
            "env_steps": float(self._env_step_count),
            "episodes": float(self._episode_count),
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
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the DQN policy using epsilon=0 greedy Q-network rollouts.
        
        IMPORTANT (Task 3 fix): This method uses the Q-network's argmax(Q-values)
        for action selection, NOT model.policy_dist(). DQN is a value-based method
        and must be evaluated using the learned Q-function, not actor-critic interfaces.

        Args:
            env_cfg: Environment configuration
            dataset: Dataset providing puzzle instances
            checker: Checker function (x, y) -> score
            num_episodes: Number of evaluation episodes. If None, falls back
                to ``self.config.eval_num_episodes`` so UPI-TRM and baseline
                trainers use the same episode count when callers rely on
                defaults (the value that ``build_trainer`` threads in from
                ``rl_cfg.eval_num_episodes``).

        Returns:
            Dict with mean_score, success_rate, mean_return, invalid_action_rate,
            filled/violations/zero_cand, etc.
        """
        num_episodes = (
            num_episodes if num_episodes is not None else self.config.eval_num_episodes
        )
        self.q_network.eval()
        device = self.device
        
        # Create evaluation environment
        eval_env = PlanEditEnv(
            dataset=dataset,
            checker=checker,
            config=env_cfg,
            task_config=getattr(self.env, "task_config", None),
        )
        if eval_env.stop_action_id is None:
            eval_env.set_stop_action_id(stop_id=self.num_actions - 1)
        
        dataset_size = len(dataset)
        if dataset_size == 0:
            return {"mean_score": 0.0, "success_rate": 0.0, "eval_policy_mode": "q_greedy"}
        
        num_solved = 0
        total_score = 0.0
        episodes_ran = 0
        total_return = 0.0
        total_steps = 0
        total_invalid_actions = 0
        all_final_scores: List[float] = []
        all_initial_scores: List[float] = []
        max_possible_score: Optional[float] = None
        
        # Sudoku-specific tracking
        all_filled: List[int] = []
        all_violations: List[int] = []
        all_zero_cand: List[int] = []
        is_sudoku_task = False
        
        with torch.no_grad():
            for episode_idx in range(num_episodes):
                x, y = eval_env.reset(idx=episode_idx % dataset_size)
                done = False
                episode_return = 0.0
                episode_steps = 0
                episode_invalid_actions = 0
                
                # Track initial score before any edits
                initial_score = float(checker(x, y))
                all_initial_scores.append(initial_score)
                
                # Get optimal solution for computing max reward
                if isinstance(x, dict):
                    # NOTE: Use explicit None check instead of `or` to avoid
                    # "Boolean value of Tensor with more than one value is ambiguous"
                    # when solution is a multi-element tensor
                    optimal_plan = x.get("solution", None)
                    if optimal_plan is None:
                        optimal_plan = x.get("labels", None)
                else:
                    optimal_plan = None
                
                if optimal_plan is not None:
                    episode_max_reward = float(checker(x, optimal_plan))
                    if max_possible_score is None:
                        max_possible_score = episode_max_reward
                else:
                    episode_max_reward = None
                
                for _ in range(env_cfg.max_edits):
                    # Prepare inputs for Q-network
                    batch_x = self._prepare_batch_x(x)
                    batch_y = self._prepare_plan(y)
                    
                    # Get action mask
                    action_mask = eval_env.get_action_mask()
                    if action_mask is not None:
                        action_mask = action_mask.to(device)
                    
                    # === KEY FIX: Use Q-network for action selection, NOT policy_dist ===
                    # DQN selects actions via argmax(Q(s,a)) with epsilon=0 (greedy)
                    q_values = self.q_network(
                        batch_x, batch_y,
                        n=self.config.inner_unroll_n,
                        action_mask=action_mask,
                    )
                    action = q_values.argmax(dim=-1).item()

                    if action_mask is not None:
                        if action < 0 or action >= action_mask.numel() or not bool(action_mask[action].item()):
                            episode_invalid_actions += 1
                    
                    (x_next, y_next), reward, done, _ = eval_env.step(action)
                    episode_return += float(reward)
                    episode_steps += 1
                    x, y = x_next, y_next
                    
                    if done:
                        break
                
                final_score = float(checker(x, y))
                total_score += final_score
                all_final_scores.append(final_score)
                episodes_ran += 1
                total_return += episode_return
                total_steps += episode_steps
                total_invalid_actions += episode_invalid_actions
                
                # Get final plan tensor for Sudoku-specific checks
                if isinstance(y, torch.Tensor):
                    final_plan = y
                elif isinstance(y, dict) and "plan" in y:
                    final_plan = y["plan"]
                else:
                    final_plan = None
                
                # Track Sudoku-specific stats and use solution-independent success criterion
                if final_plan is not None and final_plan.numel() in (16, 81):
                    is_sudoku_task = True
                    total_cells, filled, violations, zero_cand = sudoku_get_stats(final_plan)
                    all_filled.append(filled)
                    all_violations.append(violations)
                    all_zero_cand.append(zero_cand)
                    
                    # Use sudoku_is_solved for solution-independent success
                    if sudoku_is_solved(final_plan):
                        num_solved += 1
                elif episode_max_reward is not None and abs(final_score - episode_max_reward) < 1e-6:
                    num_solved += 1
        
        mean_score = total_score / float(max(episodes_ran, 1))
        success_rate = num_solved / float(max(episodes_ran, 1))
        
        result: Dict[str, Any] = {
            "mean_score": mean_score,
            "success_rate": success_rate,
            "eval_policy_mode": "q_greedy",  # Indicates Q-network greedy, not policy_dist
            "solved_count": num_solved,
            "total_episodes": episodes_ran,
            "score_min": min(all_final_scores) if all_final_scores else 0.0,
            "score_max": max(all_final_scores) if all_final_scores else 0.0,
            "max_possible_score": max_possible_score,
            "initial_score_mean": sum(all_initial_scores) / len(all_initial_scores) if all_initial_scores else 0.0,
            "mean_return": total_return / float(max(episodes_ran, 1)),
            "mean_steps": total_steps / float(max(episodes_ran, 1)),
            "invalid_action_rate": total_invalid_actions / float(max(total_steps, 1)),
        }
        
        # Add Sudoku-specific stats if applicable
        if is_sudoku_task and all_filled:
            result["final_filled_mean"] = sum(all_filled) / len(all_filled)
            result["final_violations_mean"] = sum(all_violations) / len(all_violations)
            result["final_zero_cand_mean"] = sum(all_zero_cand) / len(all_zero_cand)
        
        return result
