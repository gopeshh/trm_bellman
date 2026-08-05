"""
CleanRL-style DQN for CartPole smoke tests and Sudoku TRM benchmarks.

Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

Modifications from upstream CleanRL dqn.py:
  - Agent class: CartPoleQNetwork (orthogonal init) or TRMQNetwork (TRM policy
    head logits as Q-values). See trm_adapter.py.
  - Double DQN: action selection uses online network argmax, value from target
    network (lines 387-397).
  - N-step returns: NStepBuffer accumulates gamma-discounted rewards over n
    steps with episode-aware flushing. Per-transition walk_len stored in
    Transition.n_steps; TD target uses gamma**n_steps (lines 77-120, 398-402).
  - Env dispatch: env_kind="cartpole" uses gym.make; env_kind="sudoku" uses
    GymPlanEditEnv wrapper (lines 306-318).
  - Evaluation: Sudoku uses evaluate_plan_policy_with_scores from
    rl/evaluator.py (lines 174-206).
  - Log format: eval lines print eval_success_rate= first for provenance
    script compatibility.
  - Replay buffer stores dict observations for Sudoku (lines 227-244).
"""

from __future__ import annotations

import argparse
import copy
from importlib import import_module
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from rl.cleanrl.trm_adapter import (
    CartPoleQNetwork,
    GymPlanEditEnv,
    TRMQNetwork,
    action_mask_from_obs,
    build_sudoku_bundle,
    obs_to_device,
)
from rl.cleanrl.ppo_trm import (
    ObsType,
    _config_float,
    _config_int,
    _compat_reset,
    _compat_step,
    _evaluate_sudoku,
    _output_dir,
    _seed_everything,
    _write_jsonl,
)

try:
    gym_api = import_module("gym")
except ImportError:  # pragma: no cover
    gym_api = import_module("gymnasium")


class Transition(NamedTuple):
    obs: Any
    action: int
    reward: float
    next_obs: Any
    done: bool
    n_steps: int = 1


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class NStepBuffer:
    """Accumulates n-step returns before pushing to the main replay buffer.

    Each push adds one transition. When n transitions have accumulated (or an
    episode ends), the oldest is flushed as a single n-step transition with
    discounted reward sum and the n-th-step next_obs/done.

    Episode boundaries (done=True) flush all remaining pending transitions
    with their actual (shorter-than-n) discounted returns and done=True.
    """

    def __init__(self, n: int, gamma: float, replay: ReplayBuffer) -> None:
        self.n = n
        self.gamma = gamma
        self.replay = replay
        self.pending: deque[Tuple[Any, int, float, Any, bool]] = deque()

    def push(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool, episode_done: bool = False) -> None:
        self.pending.append((obs, action, reward, next_obs, done))
        if episode_done:
            self._flush_all()
        elif len(self.pending) >= self.n:
            self._flush_oldest()

    def _flush_oldest(self) -> None:
        if not self.pending:
            return
        obs_0, action_0, _, _, _ = self.pending[0]
        n_step_reward = 0.0
        final_done = False
        final_next_obs = None
        walk_len = 0
        steps = min(self.n, len(self.pending))
        for i in range(steps):
            _, _, r, nobs, d = self.pending[i]
            n_step_reward += (self.gamma ** i) * r
            final_next_obs = nobs
            walk_len = i + 1
            if d:
                final_done = True
                break
        self.replay.push(Transition(obs_0, action_0, n_step_reward, final_next_obs, final_done, walk_len))
        self.pending.popleft()

    def _flush_all(self) -> None:
        while self.pending:
            self._flush_oldest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanRL DQN adapter for TRM baselines.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    config = dict(config)
    config["config_path"] = str(Path(config_path).resolve())
    return config


def _evaluate_cartpole_dqn(
    q_network: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    env_id = str(config.get("env_id", "CartPole-v1"))
    eval_episodes = _config_int(config, "eval_episodes", 20)
    episode_returns: List[float] = []
    for episode_idx in range(eval_episodes):
        env = gym_api.make(env_id)
        obs, _info = _compat_reset(env, seed=int(config.get("seed", 0)) + 10_000 + episode_idx)
        done = False
        truncated = False
        episode_return = 0.0
        while not (done or truncated):
            obs_tensor = torch.as_tensor(np.array(obs), device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(obs_tensor)
            action = int(q_values.argmax(dim=-1).item())
            obs, reward, done, truncated, _info = _compat_step(env, action)
            episode_return += float(reward)
        episode_returns.append(episode_return)
        if hasattr(env, "close"):
            env.close()
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    return {
        "mean_return": mean_return,
        "score_min": float(np.min(episode_returns)) if episode_returns else 0.0,
        "score_max": float(np.max(episode_returns)) if episode_returns else 0.0,
        "episodes": eval_episodes,
        "eval_policy_mode": "greedy",
    }


def _evaluate_sudoku_dqn(
    q_network: Any, bundle: Any, config: Mapping[str, Any], device: torch.device,
) -> Dict[str, Any]:
    from rl.evaluator import evaluate_plan_policy_with_scores
    from rl.cleanrl.trm_adapter import _QNetworkEvalAdapter

    eval_model = _QNetworkEvalAdapter(q_network)
    eval_episodes = _config_int(config, "eval_episodes", 50)
    inner_n = int(config.get("inner_unroll_n", 2))

    mean_score, success_rate, stats = evaluate_plan_policy_with_scores(
        model=eval_model,
        dataset=bundle.eval_dataset,
        checker=bundle.checker_fn,
        env_cfg=bundle.env_cfg,
        task_config=bundle.task_config,
        num_episodes=eval_episodes,
        inner_unroll_n=inner_n,
        episodic_latent=True,
        greedy=True,
    )

    return {
        "success_rate": success_rate,
        "mean_score": mean_score,
        "mean_return": stats.get("mean_return", 0.0),
        "solved_count": stats.get("solved_count", 0),
        "episodes": eval_episodes,
        "score_min": stats.get("score_min", 0.0),
        "score_max": stats.get("score_max", 0.0),
        "initial_score_mean": stats.get("initial_score_mean", 0.0),
        "invalid_action_rate": stats.get("invalid_action_rate", 0.0),
        "eval_policy_mode": "greedy",
        "eval_split": bundle.eval_split,
        "eval_pool_sha256": bundle.eval_pool_sha256,
    }


def _print_eval_dqn(step: int, metrics: Mapping[str, Any]) -> None:
    sr = metrics.get("success_rate")
    if sr is not None:
        print(
            f"[step {step:05d}] eval_success_rate={sr:.3f} "
            f"eval_mean_return={metrics['mean_return']:.3f} "
            f"score_range={metrics['score_min']:.2f}-{metrics['score_max']:.2f}"
        )
    else:
        print(
            f"[step {step:05d}] eval_mean_return={metrics['mean_return']:.3f} "
            f"score_range={metrics['score_min']:.2f}-{metrics['score_max']:.2f}"
        )


def _linear_schedule(start: float, end: float, fraction: float) -> float:
    return start + fraction * (end - start)


def _sample_random_action(
    num_actions: int,
    action_mask: Optional[torch.Tensor] = None,
) -> int:
    """Sample uniformly from the valid actions in ``action_mask``."""
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if action_mask is None:
        return random.randrange(num_actions)

    flat_mask = action_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    if flat_mask.numel() != num_actions:
        raise ValueError(
            f"action mask has {flat_mask.numel()} entries, expected {num_actions}"
        )
    valid_actions = torch.nonzero(flat_mask, as_tuple=False).reshape(-1).tolist()
    if not valid_actions:
        raise RuntimeError("Cannot sample an action because the action mask is empty")
    return int(random.choice(valid_actions))


def _compute_td_target(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    next_q: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Build DQN targets without arithmetically masking terminal Q-values."""
    bootstrap_q = torch.where(
        dones.to(dtype=torch.bool),
        torch.zeros_like(next_q),
        next_q,
    )
    return rewards + discounts * bootstrap_q


def _obs_to_buffer(obs: Any, env_kind: str) -> Any:
    """Convert observation to a format suitable for replay buffer storage."""
    if env_kind == "cartpole":
        return np.array(obs, dtype=np.float32)
    return {k: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for k, v in obs.items()}


def _batch_buffer_obs(transitions: List[Transition], device: torch.device, env_kind: str) -> Any:
    """Stack buffered observations into batched tensors."""
    if env_kind == "cartpole":
        return torch.as_tensor(np.array([t.obs for t in transitions]), device=device, dtype=torch.float32)
    keys = transitions[0].obs.keys()
    return {
        k: torch.as_tensor(np.stack([t.obs[k] for t in transitions]), device=device, dtype=torch.float32)
        for k in keys
    }


def _batch_buffer_next_obs(transitions: List[Transition], device: torch.device, env_kind: str) -> Any:
    if env_kind == "cartpole":
        return torch.as_tensor(np.array([t.next_obs for t in transitions]), device=device, dtype=torch.float32)
    keys = transitions[0].next_obs.keys()
    return {
        k: torch.as_tensor(np.stack([t.next_obs[k] for t in transitions]), device=device, dtype=torch.float32)
        for k in keys
    }


def run(config: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    seed = _config_int(config, "seed", 0)
    _seed_everything(seed, torch_deterministic=bool(config.get("torch_deterministic", True)))

    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    env_kind = str(config.get("env_kind", "cartpole")).lower()

    bundle = None
    if env_kind == "cartpole":
        env_id = str(config.get("env_id", "CartPole-v1"))
        env = gym_api.make(env_id)
    else:
        bundle = build_sudoku_bundle(config)
        env = GymPlanEditEnv(
            dataset=bundle.dataset, checker_fn=bundle.checker_fn,
            env_cfg=bundle.env_cfg, task_config=bundle.task_config,
            seed=seed,
        )

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_history_path = output_dir / "eval_history.jsonl"
    if eval_history_path.exists():
        eval_history_path.unlink()

    total_timesteps = _config_int(config, "total_timesteps", 100000)
    learning_rate = _config_float(config, "learning_rate", 2.5e-4)
    gamma = _config_float(config, "gamma", 0.99)
    buffer_size = _config_int(config, "buffer_size", 10000)
    batch_size = _config_int(config, "dqn_batch_size", _config_int(config, "batch_size", 64))
    learning_starts = _config_int(config, "learning_starts", 500)
    train_freq = _config_int(config, "train_freq", 4)
    target_update_freq = _config_int(config, "target_update_freq", 500)
    epsilon_start = _config_float(config, "epsilon_start", 1.0)
    epsilon_end = _config_float(config, "epsilon_end", 0.05)
    exploration_fraction = _config_float(config, "exploration_fraction", 0.5)
    max_grad_norm = _config_float(config, "max_grad_norm", 10.0)
    eval_interval = _config_int(config, "eval_interval", 5000)
    log_interval = _config_int(config, "log_interval", 1000)
    n_step = _config_int(config, "n_step", 1)

    if env_kind == "cartpole":
        obs_dim = _config_int(config, "obs_dim", 4)
        action_dim = _config_int(config, "action_dim", 2)
        hidden_dim = _config_int(config, "hidden_dim", 128)
        q_network = CartPoleQNetwork(obs_dim, action_dim, hidden_dim).to(device)
        target_network = CartPoleQNetwork(obs_dim, action_dim, hidden_dim).to(device)
    else:
        agent_config = dict(config)
        agent_config["_bundle"] = bundle
        q_network = TRMQNetwork(agent_config).to(device)
        target_network = TRMQNetwork(agent_config).to(device)

    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, eps=1e-5)
    replay = ReplayBuffer(buffer_size)
    nstep_buf = NStepBuffer(n=n_step, gamma=gamma, replay=replay) if n_step > 1 else None

    run_config = dict(config)
    run_config.pop("_bundle", None)
    run_config.update({"backend": "cleanrl", "algo": "dqn", "mode": "train", "device": str(device), "n_step": n_step})
    if bundle is not None:
        run_config.update(
            {
                "train_split": bundle.train_split,
                "eval_split": bundle.eval_split,
                "train_pool_sha256": bundle.train_pool_sha256,
                "eval_pool_sha256": bundle.eval_pool_sha256,
                "eval_puzzle_id_offset": bundle.eval_puzzle_id_offset,
            }
        )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n")

    if env_kind == "cartpole":
        obs, _info = _compat_reset(env, seed=seed)
    else:
        obs, _info = env.reset(seed=seed)

    start_time = time.time()
    eval_history: List[Dict[str, Any]] = []
    episode_return = 0.0
    episode_count = 0
    num_actions = env.action_space.n if hasattr(env, "action_space") and hasattr(env.action_space, "n") else (bundle.num_actions if bundle else 2)
    loss_val = 0.0
    q_mean = 0.0
    td_mean = 0.0

    for global_step in range(1, total_timesteps + 1):
        fraction = min(1.0, float(global_step) / max(1, int(total_timesteps * exploration_fraction)))
        epsilon = _linear_schedule(epsilon_start, epsilon_end, fraction)

        if random.random() < epsilon:
            if env_kind == "cartpole":
                action = _sample_random_action(num_actions)
            else:
                if not isinstance(obs, dict):
                    raise TypeError("Sudoku DQN observations must be dictionaries.")
                action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
                action = _sample_random_action(num_actions, action_mask)
        else:
            if env_kind == "cartpole":
                obs_tensor = torch.as_tensor(np.array(obs), device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_network(obs_tensor)
            else:
                obs_t = obs_to_device(obs, device)
                if not isinstance(obs_t, dict):
                    raise TypeError("Sudoku DQN observations must be dictionaries.")
                obs_t = {k: (v.unsqueeze(0) if torch.is_tensor(v) and v.ndim == 1 else v) for k, v in obs_t.items()}
                with torch.no_grad():
                    q_values = q_network(obs_t, action_mask=action_mask_from_obs(obs_t))
            action = int(q_values.argmax(dim=-1).item())

        if env_kind == "cartpole":
            next_obs, reward, terminated, truncated, info = _compat_step(env, action)
        else:
            next_obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        done_for_buffer = terminated
        obs_buf = _obs_to_buffer(obs, env_kind)
        next_obs_buf = _obs_to_buffer(next_obs, env_kind)

        if nstep_buf is not None:
            nstep_buf.push(obs_buf, action, reward, next_obs_buf, done_for_buffer, episode_done=terminated or truncated)
        else:
            replay.push(Transition(obs=obs_buf, action=action, reward=reward, next_obs=next_obs_buf, done=done_for_buffer))

        if terminated or truncated:
            episode_count += 1
            if env_kind == "cartpole":
                obs, _info = _compat_reset(env)
            else:
                obs, _info = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

        if global_step >= learning_starts and global_step % train_freq == 0 and len(replay) >= batch_size:
            batch = replay.sample(batch_size)
            b_obs = _batch_buffer_obs(batch, device, env_kind)
            b_actions = torch.as_tensor([t.action for t in batch], device=device, dtype=torch.long)
            b_rewards = torch.as_tensor([t.reward for t in batch], device=device, dtype=torch.float32)
            b_next_obs = _batch_buffer_next_obs(batch, device, env_kind)
            b_dones = torch.as_tensor([t.done for t in batch], device=device, dtype=torch.float32)
            b_n_steps = torch.as_tensor([t.n_steps for t in batch], device=device, dtype=torch.float32)

            with torch.no_grad():
                if env_kind == "cartpole":
                    target_q = target_network(b_next_obs)
                    online_q = q_network(b_next_obs)
                else:
                    next_mask = action_mask_from_obs(b_next_obs)
                    target_q = target_network(b_next_obs, action_mask=next_mask)
                    online_q = q_network(b_next_obs, action_mask=next_mask)
                best_actions = online_q.argmax(dim=1)
                target_max_q = target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                discount = gamma ** b_n_steps
                td_target = _compute_td_target(
                    b_rewards,
                    discount,
                    target_max_q,
                    b_dones,
                )

            if env_kind == "cartpole":
                current_q = q_network(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
            else:
                current_q = q_network(b_obs, action_mask=action_mask_from_obs(b_obs)).gather(1, b_actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(current_q, td_target)
            loss_val = float(loss.item())
            q_mean = float(current_q.mean().item())
            td_mean = float(td_target.mean().item())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"DQN loss is not finite at step {global_step}: {loss.item()}. "
                    f"Q range: [{current_q.min().item():.2f}, {current_q.max().item():.2f}], "
                    f"TD target range: [{td_target.min().item():.2f}, {td_target.max().item():.2f}]"
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), max_grad_norm)
            optimizer.step()

        if global_step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        if global_step % log_interval == 0 and global_step >= learning_starts:
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            print(
                f"[step {global_step:05d}] td_loss={loss_val:.6f} "
                f"mean_q={q_mean:.3f} td_target={td_mean:.3f} "
                f"epsilon={epsilon:.3f} buffer={len(replay)} episodes={episode_count} sps={sps}"
            )

        if eval_interval > 0 and global_step % eval_interval == 0:
            if env_kind == "cartpole":
                metrics = _evaluate_cartpole_dqn(q_network, config, device)
            else:
                metrics = _evaluate_sudoku_dqn(q_network, bundle, config, device)
            metrics["step"] = global_step
            eval_history.append(metrics)
            _write_jsonl(eval_history_path, metrics)
            _print_eval_dqn(global_step, metrics)

    if env_kind == "cartpole":
        final_eval = _evaluate_cartpole_dqn(q_network, config, device)
    else:
        final_eval = _evaluate_sudoku_dqn(q_network, bundle, config, device)
    if not eval_history or int(eval_history[-1]["step"]) != total_timesteps:
        final_eval["step"] = total_timesteps
        eval_history.append(final_eval)
        _write_jsonl(eval_history_path, final_eval)
        _print_eval_dqn(total_timesteps, final_eval)

    if hasattr(env, "close"):
        env.close()

    summary = {
        "backend": "cleanrl",
        "mode": "train",
        "t0_2_status": "complete",
        "t0_3_ready": True,
        "policy_consumes_action_mask": env_kind != "cartpole",
        "env": env_kind,
        "algo": "dqn",
        "algo_variant": config.get("algo_variant", "cleanrl_dqn"),
        "seed": seed,
        "dataset_dir": str(config.get("dataset_path", "")),
        "train_split": bundle.train_split if bundle else None,
        "eval_split": bundle.eval_split if bundle else None,
        "train_pool_sha256": bundle.train_pool_sha256 if bundle else None,
        "eval_pool_sha256": bundle.eval_pool_sha256 if bundle else None,
        "eval_puzzle_id_offset": bundle.eval_puzzle_id_offset if bundle else None,
        "action_space_n": bundle.num_actions if bundle else _config_int(config, "action_dim", 2),
        "train_steps": total_timesteps,
        "eval_freq": eval_interval,
        "eval_episodes": _config_int(config, "eval_episodes", 20),
        "device": str(device),
        "env_kind": env_kind,
        "final_eval": final_eval,
        "best_eval_mean_return": float(max((e["mean_return"] for e in eval_history), default=0.0)),
        "best_eval_success_rate": float(max((e.get("success_rate", 0.0) for e in eval_history), default=0.0)),
        "num_eval_points": len(eval_history),
        "artifacts": {
            "run_config_json": str(output_dir / "run_config.json"),
            "eval_history_jsonl": str(eval_history_path),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def main(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        args = _parse_args()
        config = _load_config(args.config)
        if args.seed is not None:
            config["seed"] = int(args.seed)
        if args.output_dir is not None:
            config["output_dir"] = args.output_dir
    return run(config)


if __name__ == "__main__":
    main()
