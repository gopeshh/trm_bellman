from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    gym_api = import_module("gym")
except ImportError:  # pragma: no cover
    try:
        gym_api = import_module("gymnasium")
    except ImportError:  # pragma: no cover
        gym_api = None


MASKED_LOGIT_VALUE = -1e8
ObsType = Union[torch.Tensor, Dict[str, torch.Tensor]]


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _ensure_action_mask(action_mask: Any, *, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if action_mask is None:
        return None
    if torch.is_tensor(action_mask):
        mask = action_mask.to(device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(action_mask, device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    return mask


def apply_action_mask(logits: torch.Tensor, action_mask: Optional[Any]) -> torch.Tensor:
    mask = _ensure_action_mask(action_mask, batch_size=logits.shape[0], device=logits.device)
    if mask is None:
        return logits
    masked_logits = logits.masked_fill(~mask, MASKED_LOGIT_VALUE)
    all_masked = ~mask.any(dim=-1, keepdim=True)
    if all_masked.any():
        masked_logits = torch.where(all_masked.expand_as(masked_logits), torch.zeros_like(masked_logits), masked_logits)
    return masked_logits


def action_mask_from_obs(obs: ObsType) -> Optional[torch.Tensor]:
    if isinstance(obs, dict):
        return obs.get("action_mask")
    return None


def obs_to_device(obs: ObsType, device: torch.device) -> ObsType:
    if isinstance(obs, dict):
        return {
            key: (value.to(device) if torch.is_tensor(value) else torch.as_tensor(value, device=device))
            for key, value in obs.items()
        }
    if torch.is_tensor(obs):
        return obs.to(device)
    return torch.as_tensor(obs, device=device)


def flatten_time_env_obs(observations: Sequence[ObsType]) -> ObsType:
    if not observations:
        raise ValueError("Expected at least one observation")
    first = observations[0]
    if isinstance(first, dict):
        dict_observations: list[Dict[str, torch.Tensor]] = []
        for observation in observations:
            if not isinstance(observation, dict):
                raise TypeError("Cannot combine dictionary and tensor observations.")
            dict_observations.append(observation)
        stacked = {
            key: torch.stack([obs[key] for obs in dict_observations], dim=0)
            for key in first.keys()
        }
        return {
            key: value.reshape(-1, *value.shape[2:])
            for key, value in stacked.items()
        }
    tensor_observations: list[torch.Tensor] = []
    for observation in observations:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("Cannot combine tensor and dictionary observations.")
        tensor_observations.append(observation)
    stacked_tensor = torch.stack(tensor_observations, dim=0)
    return stacked_tensor.reshape(-1, *stacked_tensor.shape[2:])


@dataclass(frozen=True)
class SudokuBundle:
    dataset: Any
    eval_dataset: Any
    checker_fn: Any
    env_cfg: Any
    task_config: Any
    seq_len: int
    vocab_size: int
    num_identifiers: int
    num_actions: int
    checker_kind: str
    rl_cfg: Any
    train_split: str
    eval_split: str
    train_pool_sha256: Optional[str]
    eval_pool_sha256: Optional[str]
    eval_puzzle_id_offset: int


def build_sudoku_bundle(config: Mapping[str, Any]) -> SudokuBundle:
    from rl.config import RLConfig
    from rl.envs.plan_edit_env import PlanEditEnvConfig
    from rl.training_setup import (
        build_dataset_from_paths,
        offset_puzzle_identifiers,
        resolve_checker_from_dataset,
    )
    from utils.dataset_provenance import (
        dataset_input_sha256s,
        dataset_pool_sha256,
    )

    dataset_paths = config.get("dataset_paths")
    if dataset_paths is None:
        dataset_path = config.get("dataset_path")
        dataset_paths = [dataset_path] if dataset_path else None

    rl_cfg_kwargs: Dict[str, Any] = {
        "batch_size": int(config.get("batch_size", 1)),
        "max_edits": int(config.get("max_edits", 16)),
        "gamma": float(config.get("gamma", 0.99)),
    }
    for field in (
        "use_feasibility_checker",
        "use_progress_checker",
        "use_constraint_checker",
        "feasibility_violation_weight",
        "feasibility_zerocand_weight",
    ):
        if field in config:
            rl_cfg_kwargs[field] = config[field]
    rl_cfg = RLConfig(**rl_cfg_kwargs)

    batch_size = int(config.get("batch_size", 1))
    pool_size = int(config.get("pool_size", max(batch_size, 8)))
    train_split = str(config.get("train_split", "train"))
    eval_split = str(config.get("eval_split", "test"))
    eval_episodes = int(config.get("eval_episodes", 50))
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be positive")
    if dataset_paths and train_split == eval_split:
        raise ValueError(
            "CleanRL Sudoku training and evaluation must use different splits; "
            f"got {train_split!r} for both."
        )
    dataset, seq_len, vocab_size, num_identifiers = build_dataset_from_paths(
        dataset_paths=dataset_paths,
        pool_size=pool_size,
        split=train_split,
    )

    train_pool_sha256: Optional[str] = None
    eval_pool_sha256: Optional[str] = None
    eval_puzzle_id_offset = 0
    if dataset_paths:
        eval_pool_size = max(
            int(config.get("eval_pool_size", eval_episodes)),
            eval_episodes,
        )
        eval_dataset, eval_seq_len, eval_vocab_size, eval_num_identifiers = (
            build_dataset_from_paths(
                dataset_paths=dataset_paths,
                pool_size=eval_pool_size,
                split=eval_split,
            )
        )
        if (eval_seq_len, eval_vocab_size) != (seq_len, vocab_size):
            raise RuntimeError(
                "CleanRL Sudoku train/eval splits have incompatible shapes: "
                f"train={(seq_len, vocab_size)}, "
                f"eval={(eval_seq_len, eval_vocab_size)}."
            )
        if len(eval_dataset) < eval_episodes:
            raise RuntimeError(
                "CleanRL Sudoku evaluation split is smaller than eval_episodes; "
                f"refusing to repeat instances ({len(eval_dataset)} < {eval_episodes})."
            )
        overlap = set(dataset_input_sha256s(dataset)).intersection(
            dataset_input_sha256s(eval_dataset)
        )
        if overlap:
            raise RuntimeError(
                "CleanRL Sudoku train/eval pools overlap; refusing an in-sample "
                f"evaluation ({len(overlap)} duplicate inputs)."
            )
        eval_puzzle_id_offset = num_identifiers
        offset_puzzle_identifiers(eval_dataset, eval_puzzle_id_offset)
        num_identifiers = eval_puzzle_id_offset + eval_num_identifiers
        train_pool_sha256 = dataset_pool_sha256(dataset, len(dataset))
        eval_pool_sha256 = dataset_pool_sha256(eval_dataset, eval_episodes)
        print(
            f"[DATASET] train_split={train_split} train_samples={len(dataset)} "
            f"train_pool_sha256={train_pool_sha256}"
        )
        print(
            f"[DATASET] eval_split={eval_split} eval_samples={eval_episodes} "
            f"eval_pool_sha256={eval_pool_sha256} "
            f"eval_puzzle_id_offset={eval_puzzle_id_offset}"
        )
    else:
        eval_dataset = dataset
        train_split = "dummy"
        eval_split = "dummy"

    checker_resolution = resolve_checker_from_dataset(rl_cfg=rl_cfg, dataset=dataset, seq_len=seq_len)
    checker_fn = checker_resolution.checker_fn
    checker_kind = checker_resolution.checker_kind
    is_sudoku = checker_kind in {"solution", "constraint", "progress", "feasibility"}

    num_edit_actions = seq_len * vocab_size
    num_actions = num_edit_actions + 1

    env_cfg = PlanEditEnvConfig(
        max_edits=int(config.get("max_edits", 16)),
        gamma=float(config.get("gamma", 0.99)),
        reward_shaping=bool(config.get("reward_shaping", True)),
        vocab_size=vocab_size,
        solved_threshold=float(config["solved_threshold"]) if is_sudoku and config.get("solved_threshold") is not None else None,
        task_type="sudoku" if is_sudoku else "dummy",
        stop_action_mode=config.get("stop_action_mode", "noop"),
        stop_action_penalty=float(config.get("stop_action_penalty", -0.1)),
        fail_terminal_reward=float(config.get("fail_terminal_reward", 0.0)),
        solve_terminal_reward=float(config.get("solve_terminal_reward", 0.0)),
        disable_constraint_masking=bool(config.get("disable_constraint_masking", False)),
    )

    task_config = None
    try:
        from rl.task_config import get_task_config
        if is_sudoku:
            task_config = get_task_config(
                "sudoku",
                disable_constraint_masking=bool(config.get("disable_constraint_masking", False)),
            )
    except ImportError:
        pass

    return SudokuBundle(
        dataset=dataset,
        eval_dataset=eval_dataset,
        checker_fn=checker_fn,
        env_cfg=env_cfg,
        task_config=task_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_identifiers=num_identifiers,
        num_actions=num_actions,
        checker_kind=checker_kind,
        rl_cfg=rl_cfg,
        train_split=train_split,
        eval_split=eval_split,
        train_pool_sha256=train_pool_sha256,
        eval_pool_sha256=eval_pool_sha256,
        eval_puzzle_id_offset=eval_puzzle_id_offset,
    )


class GymPlanEditEnv:
    """Gym-compatible wrapper over PlanEditEnv for CleanRL rollout loops."""

    def __init__(
        self,
        dataset: Any,
        checker_fn: Any,
        env_cfg: Any,
        task_config: Any = None,
        seed: int = 0,
    ) -> None:
        from rl.envs.plan_edit_env import PlanEditEnv

        self.inner = PlanEditEnv(dataset=dataset, checker=checker_fn, config=env_cfg, task_config=task_config)
        num_edit_actions = env_cfg.vocab_size * self._guess_seq_len(dataset)
        num_actions = num_edit_actions + 1
        self.inner.set_stop_action_id(stop_id=num_actions - 1)
        self.num_actions = num_actions
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        if gym_api is not None:
            self.action_space = gym_api.spaces.Discrete(num_actions)

    def _guess_seq_len(self, dataset: Any) -> int:
        sample = dataset[0]
        if isinstance(sample, dict):
            return sample["inputs"].shape[-1]
        return sample.shape[-1]

    def _obs_from_state(self, x: Any, y: Any) -> Dict[str, np.ndarray]:
        if isinstance(x, dict):
            inputs = x["inputs"].detach().cpu().numpy().astype(np.float32).flatten()
            pid = x.get("puzzle_identifiers")
            if pid is not None:
                pid = pid.detach().cpu().numpy().astype(np.float32).flatten()
            else:
                pid = np.zeros(1, dtype=np.float32)
            remaining_edits = x.get("remaining_edits")
            if remaining_edits is not None:
                remaining_edits = (
                    remaining_edits.detach().cpu().numpy().astype(np.float32).reshape(-1)
                )
            else:
                remaining_edits = np.zeros(1, dtype=np.float32)
        else:
            inputs = x.detach().cpu().numpy().astype(np.float32).flatten()
            pid = np.zeros(1, dtype=np.float32)
            remaining_edits = np.zeros(1, dtype=np.float32)

        plan = y.detach().cpu().numpy().astype(np.float32).flatten()

        mask = self.inner.get_action_mask()
        if mask is not None:
            action_mask = mask.detach().cpu().numpy().astype(np.bool_)
        else:
            action_mask = np.ones(self.num_actions, dtype=np.bool_)

        return {
            "inputs": inputs,
            "plan": plan,
            "puzzle_identifiers": pid,
            "remaining_edits": remaining_edits,
            "action_mask": action_mask,
        }

    def reset(self, *, seed: Optional[int] = None) -> tuple:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        idx = self._rng.randint(0, len(self.inner.dataset))
        x, y = self.inner.reset(idx=idx)
        return self._obs_from_state(x, y), {}

    def step(self, action: int) -> tuple:
        (x, y), reward, done, info = self.inner.step(action)
        obs = self._obs_from_state(x, y)
        info = dict(info)
        return obs, float(reward), bool(done), False, info


def _build_trm_cfg(config: Mapping[str, Any], bundle: Any, enable_value_head: bool = True) -> dict:
    """Build TRM config dict from YAML config, matching upi_trm_train.py:813.

    RL-relevant fields are sourced from an RLConfig instance so that defaults
    (enable_contraction=True, target_Lz=0.9, latent_ball_radius=10.0, etc.)
    match the in-house baseline path exactly.
    """
    from rl.config import RLConfig

    rl_fields: Dict[str, Any] = {}
    for key in (
        "enable_contraction", "target_Lz", "target_Lv",
        "latent_ball_radius", "disable_value_head_norm",
    ):
        if key in config:
            rl_fields[key] = config[key]
    rl_cfg = RLConfig(
        batch_size=int(config.get("batch_size", 1)),
        max_edits=int(config.get("max_edits", 16)),
        gamma=float(config.get("gamma", 0.99)),
        **rl_fields,
    )

    hidden_size = int(config.get("hidden_size", 64))
    puzzle_emb_ndim = int(config.get("puzzle_emb_ndim", 0))
    return dict(
        batch_size=rl_cfg.batch_size,
        seq_len=bundle.seq_len,
        puzzle_emb_ndim=puzzle_emb_ndim,
        puzzle_emb_len=int(config.get("puzzle_emb_len", 0)) if puzzle_emb_ndim > 0 else 0,
        num_puzzle_identifiers=max(bundle.num_identifiers, rl_cfg.batch_size),
        vocab_size=bundle.vocab_size,
        H_cycles=int(config.get("h_cycles", 2)),
        L_cycles=int(config.get("l_cycles", 2)),
        H_layers=int(config.get("h_layers", 0)),
        L_layers=int(config.get("l_layers", 1)),
        hidden_size=hidden_size,
        expansion=2.0,
        num_heads=max(4, hidden_size // 16),
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=enable_value_head,
        rl_enable_contraction=rl_cfg.enable_contraction,
        rl_target_Lz=rl_cfg.target_Lz,
        rl_target_Lv=rl_cfg.target_Lv,
        rl_disable_value_head_norm=getattr(rl_cfg, "disable_value_head_norm", False),
        rl_enable_policy_head=True,
        rl_num_actions=bundle.num_actions,
        rl_latent_ball_radius=getattr(rl_cfg, "latent_ball_radius", 10.0),
    )


class TRMActorCritic(nn.Module):
    """Wraps a TRM model to expose the CleanRL actor-critic interface."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

        bundle = config["_bundle"]
        self.seq_len = bundle.seq_len
        self.vocab_size = bundle.vocab_size
        self.num_actions = bundle.num_actions
        self.inner_unroll_n = int(config.get("inner_unroll_n", 2))

        trm_cfg = _build_trm_cfg(config, bundle, enable_value_head=True)
        self.model = TinyRecursiveReasoningModel_ACTV1(trm_cfg)

    def _parse_obs(self, obs: ObsType) -> tuple:
        if not isinstance(obs, dict):
            raise TypeError("TRM actor-critic observations must be dictionaries.")
        inputs = obs["inputs"]
        plan = obs["plan"]
        pid = obs["puzzle_identifiers"]
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
            plan = plan.unsqueeze(0)
            pid = pid.unsqueeze(0)
        x = {"inputs": inputs.long(), "puzzle_identifiers": pid.long().squeeze(-1)}
        if "remaining_edits" in obs:
            x["remaining_edits"] = obs["remaining_edits"].long().reshape(-1)
        y = plan.long()
        return x, y

    def get_value(self, obs: ObsType) -> torch.Tensor:
        x, y = self._parse_obs(obs)
        value, _ = self.model.used_value(x, y, n=self.inner_unroll_n)
        return value.unsqueeze(-1)

    def get_policy_logits(self, obs: ObsType, action_mask: Optional[Any] = None) -> torch.Tensor:
        x, y = self._parse_obs(obs)
        if action_mask is None:
            action_mask = action_mask_from_obs(obs)
        if action_mask is not None:
            action_mask = action_mask.to(device=x["inputs"].device, dtype=torch.bool)
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0).expand(x["inputs"].shape[0], -1)
        dist, _ = self.model.policy_dist(x, y, n=self.inner_unroll_n, action_mask=action_mask)
        return dist.logits

    def get_action_and_value(
        self,
        obs: ObsType,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[Any] = None,
    ):
        x, y = self._parse_obs(obs)
        if action_mask is None:
            action_mask = action_mask_from_obs(obs)
        if action_mask is not None:
            action_mask = action_mask.to(device=x["inputs"].device, dtype=torch.bool)
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0).expand(x["inputs"].shape[0], -1)

        dist, z = self.model.policy_dist(x, y, n=self.inner_unroll_n, action_mask=action_mask)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value, _ = self.model.used_value(x, y, n=self.inner_unroll_n)
        return action, logprob, entropy, value.unsqueeze(-1)


class TRMQNetwork(nn.Module):
    """Wraps a TRM model using the in-house QNetwork from rl/algos/dqn.py.

    This reuses the exact same QNetwork class that the in-house DQNTrainer
    uses, ensuring identical Q-head architecture, initialization, and
    gradient flow.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
        from rl.algos.dqn import QNetwork

        bundle = config["_bundle"]
        self.seq_len = bundle.seq_len
        self.vocab_size = bundle.vocab_size
        self.num_actions = bundle.num_actions
        self.inner_unroll_n = int(config.get("inner_unroll_n", 2))

        trm_cfg = _build_trm_cfg(config, bundle, enable_value_head=False)
        base_model = TinyRecursiveReasoningModel_ACTV1(trm_cfg)
        self.q_net = QNetwork(base_model, bundle.num_actions)
        self.model = base_model

    def _parse_obs(self, obs: ObsType) -> tuple:
        if not isinstance(obs, dict):
            raise TypeError("TRM Q-network observations must be dictionaries.")
        inputs = obs["inputs"]
        plan = obs["plan"]
        pid = obs["puzzle_identifiers"]
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
            plan = plan.unsqueeze(0)
            pid = pid.unsqueeze(0)
        x = {"inputs": inputs.long(), "puzzle_identifiers": pid.long().squeeze(-1)}
        if "remaining_edits" in obs:
            x["remaining_edits"] = obs["remaining_edits"].long().reshape(-1)
        y = plan.long()
        return x, y

    def forward(self, obs: ObsType, action_mask: Optional[Any] = None) -> torch.Tensor:
        x, y = self._parse_obs(obs)
        if action_mask is None:
            action_mask = action_mask_from_obs(obs)
        if action_mask is not None:
            action_mask = action_mask.to(device=x["inputs"].device, dtype=torch.bool)
        return self.q_net(x, y, n=self.inner_unroll_n, action_mask=action_mask)


class _QNetworkEvalAdapter(nn.Module):
    """Wraps TRMQNetwork for the canonical evaluator which expects policy_dist().

    The evaluator calls model.policy_dist(x, y, n, action_mask) and uses
    dist.logits.argmax() for greedy action selection. This adapter routes
    that call through the Q-network's forward pass so eval scores the
    learned Q-values, not the untrained policy head.
    """

    def __init__(self, q_network: Any) -> None:
        super().__init__()
        self.q_network = q_network
        self.model = q_network.model
        self.config = q_network.model.config if hasattr(q_network.model, "config") else None

    def policy_dist(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        n: int = 2,
        action_mask: Optional[torch.Tensor] = None,
        z: Optional[Any] = None,
    ) -> tuple:
        if z is not None:
            raise ValueError(
                "Q-network evaluation supports episodic latent state only; "
                "persistent latent state would otherwise be silently discarded."
            )
        obs = {
            "inputs": x["inputs"].float(),
            "plan": y.float(),
            "puzzle_identifiers": x["puzzle_identifiers"].float().unsqueeze(-1) if x["puzzle_identifiers"].ndim == 1 else x["puzzle_identifiers"].float(),
            "action_mask": action_mask,
        }
        q_values = self.q_network(obs, action_mask=action_mask)
        dist = Categorical(logits=q_values)
        return dist, None


class CartPoleActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.critic_net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_net(obs)

    def get_policy_logits(self, obs: torch.Tensor, action_mask: Optional[Any] = None) -> torch.Tensor:
        logits = self.actor_net(obs)
        return apply_action_mask(logits, action_mask)

    def get_action_distribution(self, obs: torch.Tensor, action_mask: Optional[Any] = None) -> Categorical:
        return Categorical(logits=self.get_policy_logits(obs, action_mask=action_mask))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[Any] = None,
    ):
        logits = apply_action_mask(self.actor_net(obs), action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic_net(obs)


class CartPoleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, action_dim), std=1.0),
        )

    def forward(self, obs: torch.Tensor, action_mask: Optional[Any] = None) -> torch.Tensor:
        return apply_action_mask(self.network(obs), action_mask)
