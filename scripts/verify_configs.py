#!/usr/bin/env python3
"""Validate the current RL config tree against repo invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.config import RLConfig


CONFIG_ROOT = PROJECT_ROOT / "configs"
RL_CONFIG_EXCLUDE_DIRS = {"pretrain"}
STABILITY_DIRS = {
    "exp2_contraction_sweep",
    "exp3_projection_ablation",
    "table3_hard_controlled",
}


def iter_rl_config_paths() -> list[Path]:
    return sorted(
        path
        for path in CONFIG_ROOT.rglob("*.yaml")
        if not RL_CONFIG_EXCLUDE_DIRS.intersection(path.parts)
    )


def load_yaml(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def load_rl_config(path: Path) -> RLConfig:
    return RLConfig(**load_yaml(path))


def validate_repo_invariants(path: Path, cfg_dict: dict) -> list[str]:
    issues: list[str] = []

    if (
        STABILITY_DIRS.intersection(path.parts)
        or "no_vhead_norm" in path.name
    ) and not cfg_dict.get("disable_value_head_norm", False):
        issues.append("stability config must set disable_value_head_norm=true")

    if "phase4_2x2_norm_ablation" in path.parts:
        if path.stem.endswith("_nv") and not cfg_dict.get("disable_value_head_norm", False):
            issues.append("phase4 *_nv configs must set disable_value_head_norm=true")
        if path.stem.endswith("_yv") and cfg_dict.get("disable_value_head_norm", True):
            issues.append("phase4 *_yv configs must set disable_value_head_norm=false")

    return issues


def verify_config(path: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if not path.exists():
        return False, [f"file not found: {path}"]

    try:
        cfg_dict = load_yaml(path)
    except yaml.YAMLError as exc:
        return False, [f"YAML parsing error: {exc}"]

    try:
        RLConfig(**cfg_dict)
    except Exception as exc:
        return False, [f"RLConfig validation error: {exc}"]

    issues.extend(validate_repo_invariants(path, cfg_dict))
    return len(issues) == 0, issues


def main() -> int:
    config_paths = iter_rl_config_paths()
    passed = 0
    failed = 0

    print("=" * 72)
    print("RL Config Validation Report")
    print("=" * 72)

    for config_path in config_paths:
        success, issues = verify_config(config_path)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {config_path.relative_to(PROJECT_ROOT)}")

        if success:
            passed += 1
            cfg = load_rl_config(config_path)
            flags = [
                f"algo={cfg.algorithm}",
                f"checker={'feasibility' if cfg.use_feasibility_checker else 'other'}",
                f"contraction={'on' if cfg.enable_contraction else 'off'}",
                f"disable_value_head_norm={cfg.disable_value_head_norm}",
            ]
            print(f"  {' | '.join(flags)}")
        else:
            failed += 1
            for issue in issues:
                print(f"  - {issue}")

    print("-" * 72)
    print(f"Summary: {passed} passed, {failed} failed, {len(config_paths)} checked")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
