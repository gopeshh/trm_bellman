"""
CleanRL-style PPO for TRM baselines and CartPole smoke tests.

Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

Modifications from upstream CleanRL ppo.py:
  - Agent class: replaced with CartPoleActorCritic (separate nets + orthogonal
    init for CartPole) or TRMActorCritic (TRM model wrapper for Sudoku).
    See trm_adapter.py.
  - Env construction: BasicVectorEnv replaces gymnasium.vector.SyncVectorEnv;
    adds GymPlanEditEnv path for Sudoku (lines 186-203).
  - Truncation bootstrap: added final-value correction for truncated episodes
    using obs_to_device for both flat (CartPole) and dict (Sudoku) observations
    (lines 395-413).
  - Evaluation: CartPole uses _evaluate_cartpole (greedy argmax); Sudoku uses
    evaluate_plan_policy_with_scores from rl/evaluator.py for canonical
    deterministic evaluation with episode_idx % dataset_size (lines 256-288).
  - Log format: eval lines print eval_success_rate= first for provenance
    script compatibility (lines 295-305).
  - Summary: includes best_eval_success_rate alongside best_eval_mean_return.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from rl.cleanrl.trm_adapter import (
    CartPoleActorCritic,
    GymPlanEditEnv,
    TRMActorCritic,
    action_mask_from_obs,
    build_sudoku_bundle,
    flatten_time_env_obs,
    obs_to_device,
)

try:
    gym_api = import_module("gym")
except ImportError:  # pragma: no cover
    gym_api = import_module("gymnasium")


ObsType = Union[torch.Tensor, Dict[str, torch.Tensor]]


class _ValueAgent(Protocol):
    def get_value(self, obs: ObsType) -> torch.Tensor:
        ...


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanRL PPO adapter for TRM baselines.")
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


def _seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _compat_reset(env: Any, *, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    if seed is None:
        reset_result = env.reset()
    else:
        try:
            reset_result = env.reset(seed=seed)
        except TypeError:
            reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        return obs, dict(info)
    return reset_result, {}


def _compat_step(env: Any, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    step_result = env.step(int(action))
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)
    obs, reward, done, info = step_result
    truncated = bool(info.get("TimeLimit.truncated", False))
    terminated = bool(done and not truncated)
    return obs, float(reward), terminated, truncated, dict(info)


class BasicVectorEnv:
    def __init__(self, env_fns: Sequence[Any]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_action_space = getattr(self.envs[0], "action_space", None)

    def _stack_obs(self, observations: Sequence[Any]) -> Any:
        first = observations[0]
        if isinstance(first, dict):
            return {
                key: np.stack([np.asarray(obs[key]) for obs in observations], axis=0)
                for key in first.keys()
            }
        return np.stack([np.asarray(obs) for obs in observations], axis=0)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Any, List[Dict[str, Any]]]:
        observations = []
        infos: List[Dict[str, Any]] = []
        for idx, env in enumerate(self.envs):
            obs, info = _compat_reset(env, seed=None if seed is None else seed + idx)
            observations.append(obs)
            infos.append(info)
        return self._stack_obs(observations), infos

    def step(self, actions: Sequence[int]) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos: List[Dict[str, Any]] = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = _compat_step(env, int(action))
            if terminated or truncated:
                final_observation = obs
                final_info = dict(info)
                obs, reset_info = _compat_reset(env)
                info = dict(reset_info)
                info["final_observation"] = final_observation
                info["final_info"] = final_info
            observations.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)
        return (
            self._stack_obs(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminations, dtype=np.bool_),
            np.asarray(truncations, dtype=np.bool_),
            infos,
        )


def _output_dir(config: Mapping[str, Any]) -> Path:
    explicit = config.get("output_dir")
    if explicit is not None:
        return Path(explicit)
    output_root = Path(config.get("output_root", "results/cleanrl"))
    algo_variant = config.get("algo_variant", config.get("algo", "cleanrl_ppo"))
    seed = int(config.get("seed", 0))
    return output_root / str(algo_variant) / f"seed{seed}"


def _config_int(config: Mapping[str, Any], key: str, default: int) -> int:
    return int(config.get(key, default))


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    return float(config.get(key, default))


def _episode_done_flags(
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Return episode-boundary flags for rollout and GAE bookkeeping."""
    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated flags must have matching shapes")
    if terminated.ndim != 1:
        raise ValueError("boundary batches must be one-dimensional")
    return np.logical_or(terminated, truncated)


def _apply_truncation_bootstrap(
    rewards: torch.Tensor,
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: Sequence[Mapping[str, Any]],
    agent: _ValueAgent,
    device: torch.device,
    *,
    gamma: float,
) -> torch.Tensor:
    """Bootstrap a time-limit transition once, before cutting the GAE trace."""

    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated flags must have matching shapes")
    if rewards.numel() != terminated.size or len(infos) != terminated.size:
        raise ValueError(
            "reward, boundary flag, and info batches must have matching sizes"
        )

    corrected = rewards.clone()
    for index in range(terminated.size):
        if not truncated[index] or terminated[index]:
            continue
        final_obs = infos[index].get("final_observation")
        if final_obs is None:
            raise RuntimeError(
                "A truncated transition is missing final_observation; "
                "cannot compute its bootstrap value."
            )
        with torch.no_grad():
            final_obs_t = obs_to_device(final_obs, device)
            if isinstance(final_obs_t, dict):
                final_obs_t = {
                    key: (
                        value.unsqueeze(0)
                        if torch.is_tensor(value) and value.ndim == 1
                        else value
                    )
                    for key, value in final_obs_t.items()
                }
            elif final_obs_t.ndim == 1:
                final_obs_t = final_obs_t.unsqueeze(0)
            terminal_value = agent.get_value(final_obs_t).reshape(-1)
        if terminal_value.numel() != 1:
            raise ValueError(
                "Expected one bootstrap value per truncated transition, got "
                f"shape {tuple(terminal_value.shape)}"
            )
        corrected[index] += gamma * terminal_value.item()
    return corrected


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Compute GAE without carrying traces across episode boundaries."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for timestep in reversed(range(rewards.shape[0])):
        if timestep == rewards.shape[0] - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[timestep + 1]
            nextvalues = values[timestep + 1]
        delta = (
            rewards[timestep]
            + gamma * nextvalues * nextnonterminal
            - values[timestep]
        )
        lastgaelam = (
            delta
            + gamma * gae_lambda * nextnonterminal * lastgaelam
        )
        advantages[timestep] = lastgaelam
    return advantages


def _index_obs(obs: ObsType, indices: torch.Tensor) -> ObsType:
    if isinstance(obs, dict):
        return {key: value[indices] for key, value in obs.items()}
    return obs[indices]


def _build_envs(config: Mapping[str, Any]) -> Tuple[BasicVectorEnv, Optional[Any], str]:
    env_kind = str(config.get("env_kind", "cartpole")).lower()
    if env_kind == "cartpole":
        env_id = str(config.get("env_id", "CartPole-v1"))
        num_envs = _config_int(config, "num_envs", 4)

        def make_env(seed_offset: int):
            def thunk():
                try:
                    return gym_api.make(env_id)
                except Exception:
                    return gym_api.make(env_id, render_mode=None)
            return thunk

        return BasicVectorEnv([make_env(i) for i in range(num_envs)]), None, env_kind

    bundle = build_sudoku_bundle(config)
    num_envs = _config_int(config, "num_envs", 1)

    def make_sudoku_env(seed_offset: int):
        def thunk():
            return GymPlanEditEnv(
                dataset=bundle.dataset,
                checker_fn=bundle.checker_fn,
                env_cfg=bundle.env_cfg,
                task_config=bundle.task_config,
                seed=int(config.get("seed", 0)) + seed_offset,
            )
        return thunk

    return BasicVectorEnv([make_sudoku_env(i) for i in range(num_envs)]), bundle, env_kind


def _build_agent(config: Mapping[str, Any], bundle: Optional[Any], device: torch.device) -> Any:
    env_kind = str(config.get("env_kind", "cartpole")).lower()
    if env_kind == "cartpole":
        agent = CartPoleActorCritic(
            obs_dim=_config_int(config, "obs_dim", 4),
            action_dim=_config_int(config, "action_dim", 2),
            hidden_dim=_config_int(config, "hidden_dim", 64),
        )
    else:
        if bundle is None:
            raise RuntimeError("Sudoku bundle is required for TRM PPO.")
        agent_config = dict(config)
        agent_config["_bundle"] = bundle
        agent = TRMActorCritic(agent_config)
    return agent.to(device)


def _evaluate_cartpole(agent: CartPoleActorCritic, config: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
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
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            logits = agent.get_policy_logits(obs_tensor)
            action = int(logits.argmax(dim=-1).item())
            obs, reward, done, truncated, _info = _compat_step(env, action)
            episode_return += float(reward)
        episode_returns.append(episode_return)
        if hasattr(env, "close"):
            env.close()
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    return {
        "success_rate": 0.0,
        "mean_score": mean_return,
        "mean_return": mean_return,
        "invalid_action_rate": 0.0,
        "solved_count": 0,
        "episodes": eval_episodes,
        "score_min": float(np.min(episode_returns)) if episode_returns else 0.0,
        "score_max": float(np.max(episode_returns)) if episode_returns else 0.0,
        "initial_score_mean": 0.0,
        "eval_policy_mode": "greedy",
        "episode_returns": episode_returns,
    }


def _evaluate_sudoku(
    agent: nn.Module, bundle: Any, config: Mapping[str, Any], device: torch.device,
) -> Dict[str, Any]:
    from rl.evaluator import evaluate_plan_policy_with_scores

    trm_model = agent.model if hasattr(agent, "model") else agent
    eval_episodes = _config_int(config, "eval_episodes", 50)
    inner_n = int(config.get("inner_unroll_n", 2))

    mean_score, success_rate, stats = evaluate_plan_policy_with_scores(
        model=trm_model,
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
        "final_filled_mean": stats.get("final_filled_mean"),
        "final_violations_mean": stats.get("final_violations_mean"),
        "final_zero_cand_mean": stats.get("final_zero_cand_mean"),
    }


def _print_eval(step: int, metrics: Mapping[str, Any], env_kind: str) -> None:
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


def _write_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _validate_exact_interaction_budget(
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
) -> int:
    """Return the rollout batch size or reject a silently rounded budget."""

    batch_size = num_envs * num_steps
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if batch_size <= 0:
        raise ValueError("num_envs * num_steps must be positive")
    if total_timesteps % batch_size != 0:
        raise ValueError(
            "Exact interaction accounting requires total_timesteps to be "
            "divisible by num_envs * num_steps; got "
            f"{total_timesteps} % {batch_size} = {total_timesteps % batch_size}."
        )
    return batch_size


def run(config: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    seed = _config_int(config, "seed", 0)
    _seed_everything(seed, torch_deterministic=bool(config.get("torch_deterministic", True)))

    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    envs, bundle, env_kind = _build_envs(config)
    agent = _build_agent(config, bundle, device)

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_history_path = output_dir / "eval_history.jsonl"
    if eval_history_path.exists():
        eval_history_path.unlink()

    total_timesteps = _config_int(config, "total_timesteps", 100000)
    num_envs = envs.num_envs
    num_steps = _config_int(config, "num_steps", 125)
    batch_size = _validate_exact_interaction_budget(
        total_timesteps,
        num_envs,
        num_steps,
    )
    num_minibatches = _config_int(config, "num_minibatches", 4)
    minibatch_size = batch_size // max(num_minibatches, 1)
    update_epochs = _config_int(config, "update_epochs", 4)
    learning_rate = _config_float(config, "learning_rate", 2.5e-4)
    gamma = _config_float(config, "gamma", 0.99)
    gae_lambda = _config_float(config, "gae_lambda", 0.95)
    clip_coef = _config_float(config, "clip_coef", 0.2)
    ent_coef = _config_float(config, "ent_coef", 0.01)
    vf_coef = _config_float(config, "vf_coef", 0.5)
    max_grad_norm = _config_float(config, "max_grad_norm", 0.5)
    clip_vloss = bool(config.get("clip_vloss", True))
    norm_adv = bool(config.get("norm_adv", True))
    anneal_lr = bool(config.get("anneal_lr", True))

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    run_config = dict(config)
    run_config.update(
        {
            "backend": "cleanrl",
            "mode": "train",
            "batch_size_effective": batch_size,
            "minibatch_size": minibatch_size,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "total_timesteps": total_timesteps,
            "device": str(device),
            "policy_consumes_action_mask": env_kind != "cartpole",
        }
    )
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
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    next_obs_env, _infos = envs.reset(seed=seed)
    next_obs = obs_to_device(next_obs_env, device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)
    global_step = 0
    start_time = time.time()
    eval_interval = _config_int(config, "eval_interval", 5000)
    log_interval = _config_int(config, "log_interval", 5)
    num_updates = total_timesteps // batch_size
    eval_history: List[Dict[str, Any]] = []

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / max(num_updates, 1)
            optimizer.param_groups[0]["lr"] = frac * learning_rate

        obs_storage: List[ObsType] = []
        actions_storage: List[torch.Tensor] = []
        logprob_storage: List[torch.Tensor] = []
        reward_storage: List[torch.Tensor] = []
        done_storage: List[torch.Tensor] = []
        value_storage: List[torch.Tensor] = []

        for _step in range(num_steps):
            global_step += num_envs
            obs_storage.append(next_obs)
            done_storage.append(next_done)

            with torch.no_grad():
                action, logprob, _entropy, value = agent.get_action_and_value(
                    next_obs,
                    action_mask=action_mask_from_obs(next_obs),
                )

            actions_storage.append(action)
            logprob_storage.append(logprob)
            value_storage.append(value.view(-1))

            next_obs_env, reward, terminated, truncated, infos = envs.step(action.detach().cpu().numpy())
            reward_t = torch.as_tensor(reward, device=device, dtype=torch.float32)

            # Bootstrap value at truncation (agent is still alive, not terminal).
            # GAE then treats truncation as an episode boundary, so this value is
            # included exactly once and the reset state's trace cannot leak back.
            reward_t = _apply_truncation_bootstrap(
                reward_t,
                terminated,
                truncated,
                infos,
                agent,
                device,
                gamma=gamma,
            )

            reward_storage.append(reward_t)
            next_done = torch.as_tensor(
                _episode_done_flags(terminated, truncated),
                device=device,
                dtype=torch.float32,
            )
            next_obs = obs_to_device(next_obs_env, device)

            if eval_interval > 0 and global_step % eval_interval == 0:
                metrics = _evaluate_sudoku(agent, bundle, config, device) if env_kind != "cartpole" else _evaluate_cartpole(agent, config, device)
                metrics = dict(metrics)
                metrics["step"] = global_step
                eval_history.append(metrics)
                _write_jsonl(eval_history_path, metrics)
                _print_eval(global_step, metrics, env_kind)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)

        rewards = torch.stack(reward_storage)
        dones = torch.stack(done_storage)
        values = torch.stack(value_storage)
        advantages = _compute_gae(
            rewards,
            values,
            dones,
            next_done,
            next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        returns = advantages + values

        b_obs = flatten_time_env_obs(obs_storage)
        b_actions = torch.stack(actions_storage).reshape(-1)
        b_logprobs = torch.stack(logprob_storage).reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs: List[float] = []
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)

        for _epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = torch.as_tensor(b_inds[start:end], device=device, dtype=torch.long)
                mb_obs = _index_obs(b_obs, mb_inds)
                mb_actions = b_actions.long()[mb_inds]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs,
                    action=mb_actions,
                    action_mask=action_mask_from_obs(mb_obs),
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        if update % log_interval == 0:
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            print(
                f"[step {global_step:05d}] value_loss={v_loss.item():.6f} "
                f"policy_loss={pg_loss.item():.6f} "
                f"entropy={entropy_loss.item():.6f} approx_kl={approx_kl.item():.6f} "
                f"clipfrac={float(np.mean(clipfracs)) if clipfracs else 0.0:.6f} sps={sps}"
            )

    final_eval = _evaluate_sudoku(agent, bundle, config, device) if env_kind != "cartpole" else _evaluate_cartpole(agent, config, device)
    if not eval_history or int(eval_history[-1]["step"]) != int(global_step):
        final_eval_entry = dict(final_eval)
        final_eval_entry["step"] = int(global_step)
        eval_history.append(final_eval_entry)
        _write_jsonl(eval_history_path, final_eval_entry)
        _print_eval(int(global_step), final_eval_entry, env_kind)

    summary = {
        "backend": "cleanrl",
        "mode": "train",
        "t0_2_status": "complete",
        "t0_3_ready": True,
        "policy_consumes_action_mask": env_kind != "cartpole",
        "env": env_kind,
        "algo": config.get("algo", "ppo"),
        "algo_variant": config.get("algo_variant", config.get("algo", "ppo")),
        "seed": seed,
        "dataset_dir": str(config.get("dataset_path", "")),
        "train_split": bundle.train_split if bundle else None,
        "eval_split": bundle.eval_split if bundle else None,
        "train_pool_sha256": bundle.train_pool_sha256 if bundle else None,
        "eval_pool_sha256": bundle.eval_pool_sha256 if bundle else None,
        "eval_puzzle_id_offset": bundle.eval_puzzle_id_offset if bundle else None,
        "action_space_n": bundle.num_actions if bundle else _config_int(config, "action_dim", 2),
        "train_steps": global_step,
        "requested_train_steps": total_timesteps,
        "eval_freq": eval_interval,
        "eval_episodes": _config_int(config, "eval_episodes", 20),
        "device": str(device),
        "env_kind": env_kind,
        "final_eval": final_eval,
        "best_eval_mean_return": float(max((entry["mean_return"] for entry in eval_history), default=0.0)),
        "best_eval_success_rate": float(max((entry.get("success_rate", 0.0) for entry in eval_history), default=0.0)),
        "num_eval_points": len(eval_history),
        "artifacts": {
            "run_config_json": str(output_dir / "run_config.json"),
            "eval_history_jsonl": str(eval_history_path),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
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
