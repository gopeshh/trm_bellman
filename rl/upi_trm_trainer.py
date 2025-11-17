from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig


@dataclass
class Transition:
    x: Any
    y: Any
    action: torch.Tensor
    reward: torch.Tensor
    x_next: Any
    y_next: Any
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def __len__(self) -> int:
        return len(self.storage)

    def sample_batch(self, batch_size: int) -> List[Transition]:
        assert len(self.storage) >= batch_size, "Not enough transitions in replay buffer"
        indices = torch.randint(low=0, high=len(self.storage), size=(batch_size,)).tolist()
        return [self.storage[i] for i in indices]


class UPITrmTrainer:
    """
    Unrolled Policy Iteration trainer for TinyRecursiveReasoningModel_ACTV1 in plan space.
    """

    def __init__(
        self,
        model: TinyRecursiveReasoningModel_ACTV1,
        env: PlanEditEnv,
        rl_cfg: RLConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.env = env
        self.env_config: PlanEditEnvConfig = env.config
        self.rl_cfg = rl_cfg
        self.device = device

        self.replay = ReplayBuffer(capacity=rl_cfg.replay_capacity)

        value_params: List[nn.Parameter] = []
        policy_params: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "value_head" in name:
                value_params.append(param)
            elif "edit_policy" in name:
                policy_params.append(param)
            else:
                value_params.append(param)
                policy_params.append(param)

        self.value_opt = torch.optim.Adam(value_params, lr=rl_cfg.value_lr)
        self.policy_opt = torch.optim.Adam(policy_params, lr=rl_cfg.policy_lr)

        self.target_model = TinyRecursiveReasoningModel_ACTV1(self._config_to_dict(self.model.config)).to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad_(False)
        self._hard_update_target()

    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        if hasattr(config, "model_dump"):
            return config.model_dump()
        return config.dict()

    def _hard_update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self) -> None:
        tau = self.rl_cfg.target_ema_tau
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target_model.parameters()):
                p_targ.data.mul_(tau).add_(p.data, alpha=1 - tau)

    def collect_episode(self) -> None:
        """
        Run a single episode in the plan-space env using the current policy_dist,
        store transitions in replay buffer.
        """

        self.model.eval()
        x, y = self.env.reset()
        done = False
        edit_budget = min(self.rl_cfg.max_edits, self.env_config.max_edits)

        while not done and self.env.step_count < edit_budget:
            batched = self._state_is_batched(x)
            batch_x = self._prepare_batch_x(x, batched=batched)
            batch_y = self._prepare_plan(y, batched=batched)

            dist = self.model.policy_dist(batch_x, batch_y, n=self.rl_cfg.inner_unroll_n)
            action = dist.sample()

            (x_next, y_next), reward, done, _ = self.env.step(action.item())

            transition = Transition(
                x=self._clone_state(x),
                y=self._clone_state(y),
                action=action.detach().cpu(),
                reward=torch.as_tensor(reward, dtype=torch.float32).view(1),
                x_next=self._clone_state(x_next),
                y_next=self._clone_state(y_next),
                done=torch.tensor([done], dtype=torch.bool),
            )
            self.replay.add(transition)

            x, y = x_next, y_next

    def _prepare_batch_x(self, x: Dict[str, torch.Tensor], batched: bool) -> Dict[str, torch.Tensor]:
        """
        Normalize env state dictionaries into the TRM batch dict format.
        Ensures we always return tensors shaped as:
          inputs  -> [B, seq_len]
          puzzle_identifiers -> [B]
        """
        if not isinstance(x, dict):
            raise TypeError("Expected environment state x to be a dict with tensor entries.")
        batch = {}
        for key in ("inputs", "puzzle_identifiers"):
            if key not in x:
                raise KeyError(f"Expected key `{key}` in environment state for TRM inputs.")
            tensor = x[key]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            if not batched:
                if tensor.ndim == 0 or key == "inputs":
                    tensor = tensor.unsqueeze(0)
            batch[key] = tensor.to(self.device)
        return batch

    def _prepare_plan(self, y: Any, batched: bool) -> torch.Tensor:
        """
        Convert plan objects to tensors shaped [B, seq_len].
        Currently treats plans as simple tensors mirroring the input tokens.
        """
        if isinstance(y, dict):
            plan = y.get("inputs")
            if plan is None:
                raise KeyError("Dictionary plan must contain an `inputs` tensor.")
        else:
            plan = y
        if not torch.is_tensor(plan):
            plan = torch.as_tensor(plan)
        if not batched:
            plan = plan.unsqueeze(0)
        return plan.to(self.device)

    def _stack_batch(
        self, transitions: List[Transition]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stack a list of Transition objects into batch tensors.
        Assumes x and x_next are dicts with at least "inputs" and "puzzle_identifiers".
        """

        inputs = torch.stack([t.x["inputs"] for t in transitions], dim=0).to(self.device)
        puzzle_ids = torch.stack([t.x["puzzle_identifiers"] for t in transitions], dim=0).to(self.device)
        x_batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

        inputs_next = torch.stack([t.x_next["inputs"] for t in transitions], dim=0).to(self.device)
        puzzle_ids_next = torch.stack([t.x_next["puzzle_identifiers"] for t in transitions], dim=0).to(self.device)
        x_next_batch = {"inputs": inputs_next, "puzzle_identifiers": puzzle_ids_next}

        y_batch = torch.stack([self._plan_tensor(t.y) for t in transitions], dim=0).to(self.device)
        y_next_batch = torch.stack([self._plan_tensor(t.y_next) for t in transitions], dim=0).to(self.device)

        actions = torch.stack([t.action for t in transitions], dim=0).to(self.device)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        rewards = torch.stack([t.reward for t in transitions], dim=0).to(self.device).squeeze(-1)
        dones = torch.stack([t.done for t in transitions], dim=0).to(self.device).squeeze(-1)

        return x_batch, y_batch, x_next_batch, y_next_batch, actions, rewards, dones

    def _plan_tensor(self, plan: Any) -> torch.Tensor:
        if isinstance(plan, dict):
            plan_tensor = plan.get("inputs")
            if plan_tensor is None:
                raise KeyError("Dictionary plan must contain an `inputs` tensor.")
        else:
            plan_tensor = plan
        if not torch.is_tensor(plan_tensor):
            plan_tensor = torch.as_tensor(plan_tensor)
        return plan_tensor

    def _clone_state(self, state: Any) -> Any:
        if isinstance(state, dict):
            return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        if torch.is_tensor(state):
            return state.clone()
        return state

    def _state_is_batched(self, x: Dict[str, torch.Tensor]) -> bool:
        """
        Heuristic to detect whether the env already produced a batch of states.
        PlanEditEnv currently returns single instances, but custom envs may batch.
        """
        if not isinstance(x, dict):
            return False
        inputs = x.get("inputs")
        puzzle_ids = x.get("puzzle_identifiers")
        if not (torch.is_tensor(inputs) and torch.is_tensor(puzzle_ids)):
            return False
        if inputs.ndim == 0:
            return False
        if puzzle_ids.ndim == 0:
            return False
        return inputs.shape[0] == puzzle_ids.shape[0]

    def value_update(self) -> float:
        """
        Perform one value-function update using K-step bootstrapped targets.
        Here we use a simplified 1-step TD version for code clarity.
        """

        if len(self.replay) < self.rl_cfg.batch_size:
            return 0.0

        transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
        x_batch, y_batch, x_next_batch, y_next_batch, _, rewards, dones = self._stack_batch(transitions)

        self.model.train()
        self.value_opt.zero_grad()

        with torch.no_grad():
            v_next = self.target_model.used_value(x_next_batch, y_next_batch, n=self.rl_cfg.inner_unroll_n)
            mask = (~dones).float()
            v_next = v_next * mask

        v_s = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
        td_target = rewards + self.rl_cfg.gamma * v_next
        loss_val = F.mse_loss(v_s, td_target.detach())
        loss_val.backward()
        self.value_opt.step()
        self._soft_update_target()

        return float(loss_val.item())

    def policy_update(self) -> float:
        """
        Perform one policy-gradient step using 1-step TD advantages.
        """

        if len(self.replay) < self.rl_cfg.batch_size:
            return 0.0

        transitions = self.replay.sample_batch(self.rl_cfg.batch_size)
        x_batch, y_batch, x_next_batch, y_next_batch, actions, rewards, dones = self._stack_batch(transitions)

        self.model.train()
        self.policy_opt.zero_grad()

        with torch.no_grad():
            v_s = self.model.used_value(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
            v_next = self.model.used_value(x_next_batch, y_next_batch, n=self.rl_cfg.inner_unroll_n)
            mask = (~dones).float()
            v_next = v_next * mask
            td_target = rewards + self.rl_cfg.gamma * v_next
            adv = td_target - v_s

        dist = self.model.policy_dist(x_batch, y_batch, n=self.rl_cfg.inner_unroll_n)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        loss_policy = -(log_prob * adv.detach()).mean() - self.rl_cfg.entropy_coef * entropy
        loss_policy.backward()
        self.policy_opt.step()

        return float(loss_policy.item())

    def train_step(self) -> Dict[str, float]:
        """
        One outer training step: collect data, then run value and policy updates.
        """

        for _ in range(self.rl_cfg.rollout_episodes_per_step):
            self.collect_episode()

        loss_val = self.value_update()
        loss_policy = self.policy_update()

        return {
            "loss_value": loss_val,
            "loss_policy": loss_policy,
        }

