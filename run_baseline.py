#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, NamedTuple

from external_baselines import GymSudoku4x4Env, Sudoku4x4ExternalEnv
from utils.dataset_provenance import ordered_pool_sha256, sequence_input_sha256s

DEFAULT_SB3_HPARAMS = {
    "ppo": {
        "learning_rate": 1.0e-4,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_steps": 64,
        "batch_size": 16,
        "n_epochs": 4,
    },
    "a2c": {
        "learning_rate": 1.0e-4,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_steps": 32,
    },
    "dqn": {
        "learning_rate": 5.0e-4,
        "gamma": 0.99,
        "n_steps": 1,
        "buffer_size": 10000,
        "learning_starts": 200,
        "batch_size": 64,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 200,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10.0,
    },
}

_DQN_WITH_NSTEP_CLS = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrypoint for provenance-checked external baseline integration."
    )
    parser.add_argument("--algo", choices=("ppo", "a2c", "dqn"), required=True)
    parser.add_argument("--env", choices=("sudoku4x4",), required=True)
    parser.add_argument("--backend", choices=("auto", "sb3", "smoke"), default="auto")
    parser.add_argument("--mode", choices=("episode", "train"), default="episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--dataset-dir",
        default="data/sudoku-4x4-easy_6to8empties",
        help="Path to the hard 4x4 dataset root. The directory name is historical.",
    )
    parser.add_argument("--puzzle-index", type=int, default=None)
    parser.add_argument("--max-edits", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--dqn-buffer-size", type=int, default=None)
    parser.add_argument("--dqn-learning-starts", type=int, default=None)
    parser.add_argument("--dqn-batch-size", type=int, default=None)
    parser.add_argument("--dqn-train-freq", type=int, default=None)
    parser.add_argument("--dqn-gradient-steps", type=int, default=None)
    parser.add_argument("--dqn-target-update-interval", type=int, default=None)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=None)
    parser.add_argument("--dqn-exploration-initial-eps", type=float, default=None)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=None)
    parser.add_argument("--dqn-n-steps", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default="results/neurips2026/external_hard4x4",
        help="Root directory for baseline artifacts and summaries.",
    )
    return parser.parse_args()


def _algo_output_name(args: argparse.Namespace) -> str:
    if args.algo == "dqn" and args.dqn_n_steps is not None and args.dqn_n_steps > 1:
        return f"dqn_nstep{args.dqn_n_steps}"
    return args.algo


def _get_dqn_with_nstep_cls():
    global _DQN_WITH_NSTEP_CLS
    if _DQN_WITH_NSTEP_CLS is not None:
        return _DQN_WITH_NSTEP_CLS

    from collections import deque

    import numpy as np
    import torch as th
    from stable_baselines3 import DQN as SB3DQN
    from stable_baselines3.common.buffers import DictReplayBuffer
    from stable_baselines3.common.type_aliases import TensorDict
    from torch.nn import functional as F

    class NStepDictReplayBufferSamples(NamedTuple):
        observations: TensorDict
        actions: th.Tensor
        next_observations: TensorDict
        dones: th.Tensor
        rewards: th.Tensor
        discounts: th.Tensor

    class NStepDictReplayBuffer(DictReplayBuffer):
        def __init__(
            self,
            buffer_size: int,
            observation_space: Any,
            action_space: Any,
            device: Any = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            n_steps: int = 1,
            gamma: float = 0.99,
        ) -> None:
            super().__init__(
                buffer_size,
                observation_space,
                action_space,
                device=device,
                n_envs=n_envs,
                optimize_memory_usage=optimize_memory_usage,
                handle_timeout_termination=handle_timeout_termination,
            )
            if n_envs != 1:
                raise ValueError("NStepDictReplayBuffer only supports n_envs=1")
            if n_steps < 1:
                raise ValueError(f"n_steps must be >= 1, got {n_steps}")
            self.n_steps = int(n_steps)
            self.gamma = float(gamma)
            self.discounts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self._pending = [deque() for _ in range(self.n_envs)]

        def reset(self) -> None:
            super().reset()
            self.discounts.fill(0.0)
            self._pending = [deque() for _ in range(self.n_envs)]

        def _copy_obs_slice(self, obs: dict[str, np.ndarray], env_idx: int) -> dict[str, np.ndarray]:
            return {key: np.array(value[env_idx]).copy() for key, value in obs.items()}

        def _emit_transition(self, env_idx: int, horizon: int) -> None:
            pending = list(self._pending[env_idx])[:horizon]
            first = pending[0]
            last = pending[-1]

            reward = 0.0
            for step_idx, transition in enumerate(pending):
                reward += (self.gamma**step_idx) * float(transition["reward"])

            obs = {key: value[None, ...] for key, value in first["obs"].items()}
            next_obs = {key: value[None, ...] for key, value in last["next_obs"].items()}
            action = np.array([first["action"]], dtype=self.actions.dtype).reshape((1, self.action_dim))
            done = np.array([float(last["done"])], dtype=np.float32)
            reward_arr = np.array([reward], dtype=np.float32)

            super().add(
                obs,
                next_obs,
                action,
                reward_arr,
                done,
                [last["info"]],
            )
            pos = (self.pos - 1) % self.buffer_size
            self.discounts[pos, env_idx] = self.gamma**horizon

        def add(
            self,
            obs: dict[str, np.ndarray],
            next_obs: dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict[str, Any]],
        ) -> None:
            for env_idx in range(self.n_envs):
                self._pending[env_idx].append(
                    {
                        "obs": self._copy_obs_slice(obs, env_idx),
                        "next_obs": self._copy_obs_slice(next_obs, env_idx),
                        "action": int(np.array(action[env_idx]).item()),
                        "reward": float(np.array(reward[env_idx]).item()),
                        "done": bool(np.array(done[env_idx]).item()),
                        "info": dict(infos[env_idx]),
                    }
                )

                if self._pending[env_idx][-1]["done"]:
                    while self._pending[env_idx]:
                        self._emit_transition(env_idx, len(self._pending[env_idx]))
                        self._pending[env_idx].popleft()
                elif len(self._pending[env_idx]) >= self.n_steps:
                    self._emit_transition(env_idx, self.n_steps)
                    self._pending[env_idx].popleft()

        def _get_samples(self, batch_inds: np.ndarray, env: Any = None) -> NStepDictReplayBufferSamples:
            env_indices = np.zeros((len(batch_inds),), dtype=np.int64)
            obs_ = self._normalize_obs(
                {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                env,
            )
            next_obs_ = self._normalize_obs(
                {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()},
                env,
            )
            observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
            next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}
            dones = (
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1)
            rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
            discounts = self.discounts[batch_inds, env_indices].reshape(-1, 1)
            return NStepDictReplayBufferSamples(
                observations=observations,
                actions=self.to_torch(self.actions[batch_inds, env_indices]),
                next_observations=next_observations,
                dones=self.to_torch(dones),
                rewards=self.to_torch(rewards),
                discounts=self.to_torch(discounts),
            )

    class DQNWithNStep(SB3DQN):
        def __init__(self, *args, n_steps: int = 1, **kwargs) -> None:
            self.n_steps = int(n_steps) if n_steps is not None else 1
            gamma = float(kwargs.get("gamma", 0.99))
            if self.n_steps > 1:
                replay_buffer_kwargs = dict(kwargs.pop("replay_buffer_kwargs", {}))
                replay_buffer_kwargs.setdefault("n_steps", self.n_steps)
                replay_buffer_kwargs.setdefault("gamma", gamma)
                kwargs["replay_buffer_class"] = NStepDictReplayBuffer
                kwargs["replay_buffer_kwargs"] = replay_buffer_kwargs
            super().__init__(*args, **kwargs)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)

            losses = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                with th.no_grad():
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    if hasattr(replay_data, "discounts"):
                        discount = replay_data.discounts
                    else:
                        discount = th.full_like(replay_data.dones, self.gamma)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discount * next_q_values

                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))

    NStepDictReplayBufferSamples.__module__ = __name__
    NStepDictReplayBufferSamples.__qualname__ = "NStepDictReplayBufferSamples"
    NStepDictReplayBuffer.__module__ = __name__
    NStepDictReplayBuffer.__qualname__ = "NStepDictReplayBuffer"
    DQNWithNStep.__module__ = __name__
    DQNWithNStep.__qualname__ = "DQNWithNStep"
    globals()["NStepDictReplayBufferSamples"] = NStepDictReplayBufferSamples
    globals()["NStepDictReplayBuffer"] = NStepDictReplayBuffer
    globals()["DQNWithNStep"] = DQNWithNStep
    _DQN_WITH_NSTEP_CLS = DQNWithNStep
    return _DQN_WITH_NSTEP_CLS


def _try_load_sb3(algo: str):
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO as Algo
        elif algo == "a2c":
            from stable_baselines3 import A2C as Algo
        else:
            Algo = _get_dqn_with_nstep_cls()
        _patch_sb3_preprocess_obs()
        return Algo, None
    except ImportError as exc:  # pragma: no cover - depends on local packages
        return None, str(exc)


def _patch_sb3_preprocess_obs() -> None:
    import torch as th
    import torch.nn.functional as F
    from gym import spaces
    from stable_baselines3.common import policies as sb3_policies
    from stable_baselines3.common import preprocessing as sb3_preprocessing

    if getattr(sb3_preprocessing, "_buiksat_cpu_one_hot_patch", False):
        return

    def patched_preprocess_obs(
        obs: Any,
        observation_space: Any,
        normalize_images: bool = True,
    ) -> Any:
        if isinstance(observation_space, spaces.Dict):
            assert isinstance(obs, dict)
            return {
                key: patched_preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
                for key, _obs in obs.items()
            }

        if isinstance(observation_space, spaces.Box):
            device = obs.device
            obs_cpu = obs.cpu()
            if normalize_images and sb3_preprocessing.is_image_space(observation_space):
                return (obs_cpu.float() / 255.0).to(device)
            return obs_cpu.float().to(device)

        if isinstance(observation_space, spaces.Discrete):
            device = obs.device
            return F.one_hot(obs.long().cpu(), num_classes=int(observation_space.n)).float().to(device)

        if isinstance(observation_space, spaces.MultiDiscrete):
            device = obs.device
            obs_cpu = obs.long().cpu()
            pieces = [
                F.one_hot(obs_cpu[:, idx], num_classes=int(observation_space.nvec[idx])).float()
                for idx in range(len(observation_space.nvec))
            ]
            return th.cat(pieces, dim=-1).to(device)

        if isinstance(observation_space, spaces.MultiBinary):
            device = obs.device
            return obs.cpu().float().to(device)

        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")

    sb3_preprocessing.preprocess_obs = patched_preprocess_obs
    sb3_policies.preprocess_obs = patched_preprocess_obs
    sb3_preprocessing._buiksat_cpu_one_hot_patch = True


def _output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / _algo_output_name(args) / f"seed{args.seed}"


def _merge_sb3_hparams(args: argparse.Namespace) -> dict[str, Any]:
    defaults = dict(DEFAULT_SB3_HPARAMS[args.algo])
    overrides = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "ent_coef": args.entropy_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }
    if args.algo in ("ppo", "a2c"):
        overrides["n_steps"] = args.n_steps
    if args.algo == "ppo":
        overrides["batch_size"] = args.ppo_batch_size
        overrides["n_epochs"] = args.ppo_epochs
    elif args.algo == "dqn":
        overrides.update(
            {
                "n_steps": args.dqn_n_steps,
                "buffer_size": args.dqn_buffer_size,
                "learning_starts": args.dqn_learning_starts,
                "batch_size": args.dqn_batch_size,
                "train_freq": args.dqn_train_freq,
                "gradient_steps": args.dqn_gradient_steps,
                "target_update_interval": args.dqn_target_update_interval,
                "exploration_fraction": args.dqn_exploration_fraction,
                "exploration_initial_eps": args.dqn_exploration_initial_eps,
                "exploration_final_eps": args.dqn_exploration_final_eps,
            }
        )

    for key, value in overrides.items():
        if value is not None and key in defaults:
            defaults[key] = value
    return defaults


def _policy_mask_note() -> str:
    return (
        "Vanilla SB3 PPO/A2C/DQN does not apply logit masking from action_mask; "
        "the policy acts in the full no-mask action space and invalid edits are "
        "penalized by the environment. This matches the paper's hard-4x4 no-mask "
        "protocol and should be compared against the internal no-mask A2C baseline."
    )


def _reset_env(
    env: Any,
    seed: int,
    puzzle_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    info = {
        "puzzle_index": None,
        "observation_spec": env.core.observation_spec,
    }
    reset_kwargs: dict[str, Any] = {"seed": seed}
    if puzzle_index is not None:
        reset_kwargs["options"] = {"puzzle_index": puzzle_index}
    try:
        reset_result = env.reset(**reset_kwargs)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        reset_result = env.reset()

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs = reset_result
        info["puzzle_index"] = env.core._last_puzzle_index
    return obs, info


def _step_env(env: Any, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    step_result = env.step(int(action))
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, step_info = step_result
    else:
        obs, reward, done, step_info = step_result
        terminated = bool(done) and not bool(step_info.get("TimeLimit.truncated", False))
        truncated = bool(done) and not terminated
    return obs, float(reward), terminated, truncated, step_info


def _run_one_policy_episode(
    model: Any,
    env: Any,
    *,
    seed: int,
    puzzle_index: int | None,
    deterministic: bool,
) -> dict[str, Any]:
    obs, info = _reset_env(env, seed=seed, puzzle_index=puzzle_index)
    total_reward = 0.0
    invalid_actions = 0
    steps = 0
    terminated = False
    truncated = False
    solved = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, step_info = _step_env(env, int(action))
        total_reward += float(reward)
        invalid_actions += int(step_info["invalid_action"])
        steps += 1
        solved = bool(step_info["solved"])

    return {
        "puzzle_index": info["puzzle_index"],
        "return": total_reward,
        "steps": steps,
        "solved": solved,
        "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_actions / max(steps, 1),
    }


def _evaluate_model(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    if GymSudoku4x4Env is None:
        raise RuntimeError("GymSudoku4x4Env is unavailable because gym/gymnasium/numpy are missing")

    eval_env = GymSudoku4x4Env(
        dataset_dir=args.dataset_dir,
        split=args.eval_split,
        max_edits=args.max_edits,
    )
    dataset_size = len(eval_env.core.inputs_data)
    if dataset_size < args.eval_episodes:
        raise RuntimeError(
            "Evaluation split is smaller than eval_episodes; refusing to repeat "
            f"instances ({dataset_size} < {args.eval_episodes})."
        )
    pool_sha256 = ordered_pool_sha256(
        eval_env.core.inputs_data,
        eval_env.core.labels_data,
        args.eval_episodes,
    )
    episodes = []
    for episode_idx in range(args.eval_episodes):
        episodes.append(
            _run_one_policy_episode(
                model,
                eval_env,
                seed=args.seed + episode_idx,
                puzzle_index=episode_idx % dataset_size,
                deterministic=True,
            )
        )

    mean_return = sum(ep["return"] for ep in episodes) / len(episodes)
    success_rate = sum(1 for ep in episodes if ep["solved"]) / len(episodes)
    mean_steps = sum(ep["steps"] for ep in episodes) / len(episodes)
    invalid_action_rate = sum(ep["invalid_actions"] for ep in episodes) / max(
        sum(ep["steps"] for ep in episodes),
        1,
    )

    return {
        "episodes": len(episodes),
        "split": args.eval_split,
        "pool_sha256": pool_sha256,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
        "invalid_action_rate": invalid_action_rate,
    }


def _run_smoke_episode(args: argparse.Namespace) -> dict[str, Any]:
    env = Sudoku4x4ExternalEnv(
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_edits=args.max_edits,
    )
    rng = random.Random(args.seed)
    obs, info = env.reset(seed=args.seed, puzzle_index=args.puzzle_index)

    total_reward = 0.0
    invalid_actions = 0
    trajectory = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        valid_actions = [idx for idx, allowed in enumerate(obs["action_mask"]) if allowed]
        if not valid_actions:
            action = env.stop_action_id
        else:
            action = rng.choice(valid_actions)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        invalid_actions += int(step_info["invalid_action"])
        trajectory.append(
            {
                "step": step_info["steps"],
                "action": action,
                "reward": reward,
                "solved": step_info["solved"],
                "invalid_action": step_info["invalid_action"],
            }
        )

    return {
        "backend": "smoke",
        "mode": "episode",
        "backend_note": "stable_baselines3 unavailable; used a valid-action smoke policy to validate the env contract",
        "t0_2_status": "partial",
        "t0_2_note": "Environment contract validated, but a real SB3 PPO/A2C run is still pending package installation.",
        "policy_consumes_action_mask": False,
        "policy_mask_note": _policy_mask_note(),
        "env": args.env,
        "algo": args.algo,
        "algo_variant": _algo_output_name(args),
        "seed": args.seed,
        "split": args.split,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "puzzle_index": info["puzzle_index"],
        "action_space_n": info["action_space_n"],
        "observation_spec": info["observation_spec"],
        "steps": len(trajectory),
        "return": total_reward,
        "solved": trajectory[-1]["solved"] if trajectory else False,
        "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_actions / max(len(trajectory), 1),
        "trajectory": trajectory,
    }


def _run_sb3_episode(args: argparse.Namespace, algo_cls) -> dict[str, Any]:
    if GymSudoku4x4Env is None:
        raise RuntimeError("GymSudoku4x4Env is unavailable because gym/gymnasium/numpy are missing")

    env = GymSudoku4x4Env(
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_edits=args.max_edits,
    )
    hparams = _merge_sb3_hparams(args)
    policy = "MultiInputPolicy"
    model = algo_cls(
        policy,
        env,
        verbose=0,
        seed=args.seed,
        device=args.device,
        **hparams,
    )
    episode = _run_one_policy_episode(
        model,
        env,
        seed=args.seed,
        puzzle_index=args.puzzle_index,
        deterministic=False,
    )

    return {
        "backend": "sb3",
        "mode": "episode",
        "t0_2_status": "complete",
        "policy_consumes_action_mask": False,
        "policy_mask_note": _policy_mask_note(),
        "env": args.env,
        "algo": args.algo,
        "algo_variant": _algo_output_name(args),
        "seed": args.seed,
        "split": args.split,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "puzzle_index": episode["puzzle_index"],
        "action_space_n": env.core.action_space_n,
        "observation_spec": env.core.observation_spec,
        "steps": episode["steps"],
        "return": episode["return"],
        "solved": episode["solved"],
        "invalid_actions": episode["invalid_actions"],
        "invalid_action_rate": episode["invalid_action_rate"],
    }


def _run_sb3_training(args: argparse.Namespace, algo_cls) -> dict[str, Any]:
    if GymSudoku4x4Env is None:
        raise RuntimeError("GymSudoku4x4Env is unavailable because gym/gymnasium/numpy are missing")
    if args.split == args.eval_split:
        raise RuntimeError(
            "Training and evaluation must use different dataset splits; got "
            f"{args.split!r} for both."
        )

    from stable_baselines3.common.callbacks import BaseCallback

    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_history_path = output_dir / "eval_history.jsonl"
    hparams = _merge_sb3_hparams(args)

    train_env = GymSudoku4x4Env(
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_edits=args.max_edits,
    )
    eval_probe = GymSudoku4x4Env(
        dataset_dir=args.dataset_dir,
        split=args.eval_split,
        max_edits=args.max_edits,
    )
    overlap = set(sequence_input_sha256s(train_env.core.inputs_data)).intersection(
        sequence_input_sha256s(eval_probe.core.inputs_data)
    )
    if overlap:
        raise RuntimeError(
            "Training and evaluation pools overlap; refusing an in-sample "
            f"evaluation ({len(overlap)} duplicate inputs)."
        )
    model = algo_cls(
        "MultiInputPolicy",
        train_env,
        verbose=0,
        seed=args.seed,
        device=args.device,
        **hparams,
    )

    run_config = {
        "algo": args.algo,
        "algo_variant": _algo_output_name(args),
        "backend": "sb3",
        "mode": "train",
        "seed": args.seed,
        "train_split": args.split,
        "eval_split": args.eval_split,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "train_steps": args.train_steps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "checkpoint_freq": args.checkpoint_freq,
        "device": args.device,
        "sb3_hparams": hparams,
        "policy_consumes_action_mask": False,
        "policy_mask_note": _policy_mask_note(),
    }
    if args.algo == "dqn":
        run_config["dqn_n_steps"] = int(hparams["n_steps"])
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))
    if eval_history_path.exists():
        eval_history_path.unlink()

    history: list[dict[str, Any]] = []

    class BaselineEvalCallback(BaseCallback):
        def _on_step(self) -> bool:
            step = int(self.num_timesteps)
            if args.eval_freq > 0 and step % args.eval_freq == 0:
                metrics = _evaluate_model(args, self.model)
                metrics["step"] = step
                history.append(metrics)
                with eval_history_path.open("a") as f:
                    f.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(
                    f"[step {step:05d}] "
                    f"eval_success_rate={metrics['success_rate']:.3f} "
                    f"eval_mean_return={metrics['mean_return']:.3f} "
                    f"eval_invalid_action_rate={metrics['invalid_action_rate']:.3f}"
                )

            if args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0:
                checkpoint_base = output_dir / f"model_step_{step}"
                self.model.save(str(checkpoint_base))
                print(f"[step {step:05d}] checkpoint={checkpoint_base}.zip")
            return True

    callback = BaselineEvalCallback()
    model.learn(total_timesteps=args.train_steps, callback=callback)

    final_model_base = output_dir / "model_final"
    model.save(str(final_model_base))
    final_eval = _evaluate_model(args, model)
    if not history or history[-1]["step"] != args.train_steps:
        final_entry = dict(final_eval)
        final_entry["step"] = args.train_steps
        history.append(final_entry)
        with eval_history_path.open("a") as f:
            f.write(json.dumps(final_entry, sort_keys=True) + "\n")

    best_success_rate = max((entry["success_rate"] for entry in history), default=0.0)
    summary = {
        "backend": "sb3",
        "mode": "train",
        "t0_2_status": "complete",
        "t0_3_ready": True,
        "policy_consumes_action_mask": False,
        "policy_mask_note": _policy_mask_note(),
        "env": args.env,
        "algo": args.algo,
        "algo_variant": _algo_output_name(args),
        "seed": args.seed,
        "train_split": args.split,
        "eval_split": args.eval_split,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "action_space_n": train_env.core.action_space_n,
        "observation_spec": train_env.core.observation_spec,
        "train_steps": args.train_steps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "checkpoint_freq": args.checkpoint_freq,
        "device": args.device,
        "sb3_hparams": hparams,
        "final_eval": final_eval,
        "best_eval_success_rate": best_success_rate,
        "num_eval_points": len(history),
        "artifacts": {
            "run_config_json": str(output_dir / "run_config.json"),
            "eval_history_jsonl": str(eval_history_path),
            "final_model": f"{final_model_base}.zip",
        },
    }
    if args.algo == "dqn":
        summary["dqn_n_steps"] = int(hparams["n_steps"])
    return summary


def _save_summary(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "train_summary.json" if summary.get("mode") == "train" else "episode_summary.json"
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return output_path


def main() -> int:
    args = _parse_args()

    algo_cls, sb3_error = _try_load_sb3(args.algo)
    use_sb3 = algo_cls is not None and args.backend in ("auto", "sb3")

    if args.backend == "sb3" and not use_sb3:
        raise RuntimeError(f"Requested SB3 backend but it is unavailable: {sb3_error}")
    if args.mode == "train" and not use_sb3:
        raise RuntimeError("Training mode requires the SB3 backend")

    if use_sb3 and args.mode == "train":
        summary = _run_sb3_training(args, algo_cls)
    elif use_sb3:
        summary = _run_sb3_episode(args, algo_cls)
    else:
        summary = _run_smoke_episode(args)
        if sb3_error is not None:
            summary["sb3_import_error"] = sb3_error

    output_path = _save_summary(args, summary)

    print(f"[baseline] env={args.env} algo={args.algo} backend={summary['backend']}")
    print(
        "[baseline] action_space_n="
        f"{summary['action_space_n']} observation_keys="
        f"{','.join(summary['observation_spec'].keys())}"
    )
    if summary["mode"] == "train":
        final_eval = summary["final_eval"]
        print(
            f"[baseline] finished training: steps={summary['train_steps']} "
            f"final_eval_success_rate={final_eval['success_rate']:.3f} "
            f"final_eval_mean_return={final_eval['mean_return']:.3f}"
        )
        print(f"[baseline] t0.3_ready={summary.get('t0_3_ready', False)}")
    else:
        print(
            f"[baseline] finished one episode: steps={summary['steps']} "
            f"return={summary['return']:.3f} solved={summary['solved']}"
        )
    print(f"[baseline] t0.2_status={summary['t0_2_status']}")
    if "backend_note" in summary:
        print(f"[baseline] note: {summary['backend_note']}")
    if "t0_2_note" in summary:
        print(f"[baseline] note: {summary['t0_2_note']}")
    if "policy_mask_note" in summary:
        print(f"[baseline] note: {summary['policy_mask_note']}")
    print(f"[baseline] summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
