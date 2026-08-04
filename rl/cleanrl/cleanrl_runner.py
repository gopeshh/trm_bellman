"""
Unified dispatch entrypoint for CleanRL baselines.

Routes to ppo_trm, a2c_trm, or dqn_trm based on the 'algo' field in the
YAML config. This is the single binary that the benchmark harness calls.

Usage:
    cleanrl_runner --config configs/baselines/cleanrl_ppo_sudoku.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
if hasattr(torch.backends, "cuda"):
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanRL baseline runner (dispatch entrypoint).")
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


def run(config: Mapping[str, Any]) -> Dict[str, Any]:
    algo = str(config.get("algo", "ppo")).lower()

    if algo == "ppo":
        from rl.cleanrl.ppo_trm import run as ppo_run
        return ppo_run(config)
    elif algo == "a2c":
        from rl.cleanrl.a2c_trm import run as a2c_run
        return a2c_run(config)
    elif algo in ("dqn", "dqn_n5"):
        env_kind = str(config.get("env_kind", "cartpole")).lower()
        if env_kind == "cartpole":
            from rl.cleanrl.dqn_trm import run as dqn_run
            return dqn_run(config)
        else:
            n_step = int(config.get("n_step", 1))
            if n_step > 1:
                raise ValueError(
                    f"Sudoku DQN n-step={n_step} is not supported in the in-house wrapper. "
                    "DQNTrainer supports n-step returns through DQNConfig.dqn_n_step, "
                    "but this wrapper does not yet map n_step into that field. Use "
                    "n_step=1 or run the CleanRL DQN n-step CartPole smoke gate instead."
                )
            from rl.cleanrl.dqn_inhouse_wrapper import run as dqn_inhouse_run
            return dqn_inhouse_run(config)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Expected one of: ppo, a2c, dqn, dqn_n5")


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
