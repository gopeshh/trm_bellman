"""
CleanRL-style A2C as a thin PPO wrapper.

A2C is PPO with update_epochs=1, num_minibatches=1, and clip_coef large
enough to never clip (10.0). This ensures identical rollout collection,
GAE computation, truncation bootstrap, and evaluation code paths.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from rl.cleanrl.ppo_trm import run as ppo_run


def run(config: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    config["update_epochs"] = 1
    config["num_minibatches"] = 1
    config["clip_coef"] = 10.0
    config["clip_vloss"] = False
    config["norm_adv"] = False
    return ppo_run(config)


def main(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        import argparse
        import yaml
        from pathlib import Path

        parser = argparse.ArgumentParser(description="CleanRL A2C (thin PPO wrapper).")
        parser.add_argument("--config", required=True)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--output-dir", default=None)
        args = parser.parse_args()

        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        config = dict(config)
        config["config_path"] = str(Path(args.config).resolve())
        if args.seed is not None:
            config["seed"] = int(args.seed)
        if args.output_dir is not None:
            config["output_dir"] = args.output_dir
    return run(config)


if __name__ == "__main__":
    main()
