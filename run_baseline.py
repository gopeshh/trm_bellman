#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from external_baselines import GymSudoku4x4Env, Sudoku4x4ExternalEnv

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
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrypoint for trusted external baseline integration."
    )
    parser.add_argument("--algo", choices=("ppo", "a2c"), required=True)
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
    parser.add_argument(
        "--output-root",
        default="results/neurips2026/external_hard4x4",
        help="Root directory for baseline artifacts and summaries.",
    )
    return parser.parse_args()


def _try_load_sb3(algo: str):
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO as Algo
        else:
            from stable_baselines3 import A2C as Algo
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
    return Path(args.output_root) / args.algo / f"seed{args.seed}"


def _merge_sb3_hparams(args: argparse.Namespace) -> dict[str, Any]:
    defaults = dict(DEFAULT_SB3_HPARAMS[args.algo])
    overrides = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "ent_coef": args.entropy_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "n_steps": args.n_steps,
    }
    if args.algo == "ppo":
        overrides["batch_size"] = args.ppo_batch_size
        overrides["n_epochs"] = args.ppo_epochs

    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value
    return defaults


def _policy_mask_note() -> str:
    return (
        "Vanilla SB3 PPO/A2C does not apply logit masking from action_mask; "
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
    policy = "MultiInputPolicy"
    model = algo_cls(policy, env, verbose=0, seed=args.seed, device=args.device)
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
    return {
        "backend": "sb3",
        "mode": "train",
        "t0_2_status": "complete",
        "t0_3_ready": True,
        "policy_consumes_action_mask": False,
        "policy_mask_note": _policy_mask_note(),
        "env": args.env,
        "algo": args.algo,
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
