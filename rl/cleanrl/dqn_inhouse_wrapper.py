"""
In-house DQN wrapper for the benchmark harness.

This is NOT an independent CleanRL DQN implementation. It wraps the
existing DQNTrainer from rl/algos/dqn.py to produce benchmark-compatible
output artifacts. The DQN rows in the benchmark table should be labeled
as "in-house DQN via benchmark harness," not as CleanRL.

The CleanRL DQN training loop has an unresolved observation-encoding
divergence on Sudoku that causes Q-value explosion. This wrapper is the
fallback path until that is debugged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from rl.cleanrl.trm_adapter import build_sudoku_bundle, _build_trm_cfg
from rl.cleanrl.ppo_trm import (
    _config_float,
    _config_int,
    _output_dir,
    _seed_everything,
    _write_jsonl,
)


def run(config: Mapping[str, Any]) -> Dict[str, Any]:
    import dataclasses
    import torch

    config = dict(config)
    seed = _config_int(config, "seed", 0)
    _seed_everything(seed)

    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    bundle = build_sudoku_bundle(config)

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_history_path = output_dir / "eval_history.jsonl"
    if eval_history_path.exists():
        eval_history_path.unlink()

    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    from rl.algos.dqn import DQNTrainer, DQNConfig, compute_epsilon_decay_steps
    from rl.envs.plan_edit_env import PlanEditEnv
    from utils.seeding import set_global_seed

    set_global_seed(seed)

    trm_cfg = _build_trm_cfg(config, bundle, enable_value_head=False)
    model = TinyRecursiveReasoningModel_ACTV1(trm_cfg).to(device)

    env = PlanEditEnv(dataset=bundle.dataset, checker=bundle.checker_fn, config=bundle.env_cfg, task_config=bundle.task_config)
    num_actions = bundle.seq_len * bundle.vocab_size + 1
    env.set_stop_action_id(stop_id=num_actions - 1)

    total_env_steps = _config_int(config, "total_timesteps", 320000)
    train_freq = _config_int(config, "dqn_train_freq", _config_int(config, "train_freq", 4))
    exploration_fraction = _config_float(config, "exploration_fraction", 0.3)
    inner_n = _config_int(config, "inner_unroll_n", 2)

    dqn_cfg_kwargs = dict(
        gamma=_config_float(config, "gamma", 0.99),
        epsilon_start=_config_float(config, "epsilon_start", 1.0),
        epsilon_end=_config_float(config, "epsilon_end", 0.05),
        epsilon_decay_steps=compute_epsilon_decay_steps(
            num_train_steps=total_env_steps // train_freq,
            exploration_fraction=exploration_fraction,
            train_freq=train_freq,
        ),
        buffer_size=_config_int(config, "buffer_size", 10000),
        batch_size=_config_int(config, "dqn_batch_size", 64),
        min_buffer_size=_config_int(config, "learning_starts", 200),
        target_update_freq=_config_int(config, "target_update_freq", 200),
        learning_rate=_config_float(config, "learning_rate", 2.5e-4),
        max_grad_norm=_config_float(config, "max_grad_norm", 10.0),
        inner_unroll_n=inner_n,
        train_freq=train_freq,
        double_dqn=bool(config.get("double_dqn", True)),
    )

    available_fields = {f.name for f in dataclasses.fields(DQNConfig)}
    for field, value in [
        ("gradient_steps", 1),
        ("num_train_steps", total_env_steps // train_freq),
        ("log_interval", _config_int(config, "log_interval", 50)),
        ("eval_interval", _config_int(config, "eval_interval", 10000) // train_freq),
        ("eval_num_episodes", _config_int(config, "eval_episodes", 50)),
    ]:
        if field in available_fields:
            dqn_cfg_kwargs[field] = value

    dqn_cfg = DQNConfig(**dqn_cfg_kwargs)
    trainer = DQNTrainer(model=model, env=env, config=dqn_cfg, device=device)

    run_config = dict(config)
    run_config.update({
        "backend": "inhouse_dqn_wrapper",
        "mode": "train",
        "t0_2_status": "complete",
        "t0_3_ready": True,
        "device": str(device),
        "total_env_steps_budget": total_env_steps,
    })
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n")

    eval_interval_env_steps = _config_int(config, "eval_interval", 10000)
    log_interval_env_steps = _config_int(config, "log_interval", 1000)
    eval_history = []
    start_time = time.time()
    last_eval_env_step = 0
    last_log_env_step = 0

    while trainer._env_step_count < total_env_steps:
        metrics = trainer.train_step()
        env_steps = trainer._env_step_count

        if env_steps - last_log_env_step >= log_interval_env_steps:
            last_log_env_step = env_steps
            sps = int(env_steps / max(time.time() - start_time, 1e-6))
            print(
                f"[step {env_steps:05d}] td_loss={metrics.get('loss_q', 0.0):.6f} "
                f"mean_q={metrics.get('mean_q', 0.0):.3f} "
                f"epsilon={metrics.get('epsilon', 0.0):.3f} sps={sps}"
            )

        if env_steps - last_eval_env_step >= eval_interval_env_steps:
            last_eval_env_step = env_steps
            eval_metrics = trainer.evaluate_policy_metrics(
                env_cfg=bundle.env_cfg,
                dataset=bundle.dataset,
                checker=bundle.checker_fn,
            )
            eval_entry = {
                "step": env_steps,
                "success_rate": eval_metrics.get("success_rate", 0.0),
                "mean_score": eval_metrics.get("mean_score", 0.0),
                "mean_return": eval_metrics.get("mean_return", 0.0),
                "solved_count": eval_metrics.get("solved_count", 0),
                "score_min": eval_metrics.get("score_min", 0.0),
                "score_max": eval_metrics.get("score_max", 0.0),
                "eval_policy_mode": "greedy_q",
            }
            eval_history.append(eval_entry)
            _write_jsonl(eval_history_path, eval_entry)
            sr = eval_entry["success_rate"]
            mr = eval_entry["mean_return"]
            print(
                f"[step {env_steps:05d}] eval_success_rate={sr:.3f} "
                f"eval_mean_return={mr:.3f} "
                f"score_range={eval_entry['score_min']:.2f}-{eval_entry['score_max']:.2f}"
            )

    final_eval = eval_history[-1] if eval_history else {"success_rate": 0.0, "mean_return": 0.0}
    summary = {
        "backend": "inhouse_dqn_wrapper",
        "mode": "train",
        "t0_2_status": "complete",
        "t0_3_ready": True,
        "policy_consumes_action_mask": True,
        "env": "sudoku",
        "algo": "dqn",
        "algo_variant": config.get("algo_variant", "inhouse_dqn"),
        "seed": seed,
        "dataset_dir": str(config.get("dataset_path", "")),
        "action_space_n": num_actions,
        "train_steps": total_env_steps,
        "actual_env_steps": trainer._env_step_count,
        "eval_freq": eval_interval_env_steps,
        "eval_episodes": _config_int(config, "eval_episodes", 50),
        "device": str(device),
        "env_kind": "sudoku",
        "final_eval": final_eval,
        "best_eval_mean_return": float(max((e.get("mean_return", 0.0) for e in eval_history), default=0.0)),
        "best_eval_success_rate": float(max((e.get("success_rate", 0.0) for e in eval_history), default=0.0)),
        "num_eval_points": len(eval_history),
        "artifacts": {
            "run_config_json": str(output_dir / "run_config.json"),
            "eval_history_jsonl": str(eval_history_path),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary
