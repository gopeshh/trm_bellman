#!/usr/bin/env python3
"""
Re-evaluate fixed UPI-TRM checkpoints through the baseline evaluation interface.

This script is intentionally narrow: it loads an existing family of checkpoints,
reconstructs the same dataset/checker/env bootstrap used by `upi_trm_train.py`,
and emits per-seed / aggregate metrics with the evaluator used by the
architecture-matched baselines.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


LINK_TREE_ROOT = Path(__file__).resolve().parents[1]
SUDOKU_CHECKER_KINDS = {"solution", "constraint", "progress", "feasibility"}


def _ensure_source_root_on_path() -> Path:
    candidates = [
        os.environ.get("TRM_BELLMAN_SOURCE_ROOT"),
        str(Path.cwd()),
        str(Path.cwd() / "buiksat_trm"),
        "/data/repos/fbsource/fbcode/buiksat_trm",
        "/home/buiksat/trm_bellman",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if (path / "puzzle_dataset.py").exists() and (path / "rl").exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path
    raise RuntimeError("Could not locate trm_bellman source root for reevaluation imports.")


PROJECT_ROOT = _ensure_source_root_on_path()

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnvConfig
from rl.evaluator import evaluate_plan_policy_with_scores
from rl.sudoku_checkers import (
    make_sudoku_feasibility_checker,
    sudoku_checker,
    sudoku_constraint_checker,
    sudoku_progress_checker,
)
from rl.task_config import get_task_config
from scripts.eval.unroll_sensitivity import load_model_for_eval


class OfflinePuzzleDataset:
    def __init__(
        self,
        *,
        samples: list[dict[str, Any]],
        seq_len: int,
        vocab_size: int,
        num_identifiers: int,
    ) -> None:
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_identifiers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate persistent-z UPI checkpoints through the baseline evaluator.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        required=True,
        help="Glob for the checkpoint family to evaluate.",
    )
    parser.add_argument(
        "--seed-regex",
        default=r"seed(\d+)",
        help="Regex used to infer the training seed from each checkpoint path.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional exact seed list to enforce before evaluation.",
    )
    parser.add_argument(
        "--config-yaml",
        required=True,
        help="Training config YAML used for these checkpoints.",
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        required=True,
        help="Dataset path(s) forwarded to build_dataset_from_paths().",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for per-seed CSV, aggregate JSON, and run metadata.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Optional override for evaluation episode count.",
    )
    return parser.parse_args()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _population_std(values: list[float]) -> float:
    return statistics.pstdev(values) if values else 0.0


def _load_rl_config(config_yaml: str) -> RLConfig:
    with open(config_yaml, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return RLConfig(**raw)


def _resolve_checkpoints(
    checkpoint_glob: str,
    seed_regex: str,
    expected_seeds: list[int] | None,
) -> list[tuple[int, Path]]:
    matches = sorted(Path(path) for path in glob.glob(checkpoint_glob))
    if not matches:
        raise FileNotFoundError(f"No checkpoints matched glob: {checkpoint_glob}")

    compiled = re.compile(seed_regex)
    by_seed: dict[int, Path] = {}
    for checkpoint_path in matches:
        match = compiled.search(str(checkpoint_path))
        if match is None:
            raise RuntimeError(
                f"Could not infer seed from checkpoint path {checkpoint_path} using regex {seed_regex!r}"
            )
        seed = int(match.group(1))
        if seed in by_seed:
            raise RuntimeError(
                f"Duplicate seed {seed} matched by {by_seed[seed]} and {checkpoint_path}"
            )
        by_seed[seed] = checkpoint_path

    ordered = sorted(by_seed.items(), key=lambda item: item[0])
    found_seeds = [seed for seed, _ in ordered]
    if expected_seeds is not None and found_seeds != expected_seeds:
        raise RuntimeError(
            f"Expected seeds {expected_seeds}, found {found_seeds} from {checkpoint_glob}"
        )
    return ordered


def _build_eval_bundle(
    rl_cfg: RLConfig,
    dataset_paths: list[str],
) -> tuple[Any, Any, PlanEditEnvConfig, Any, dict[str, Any]]:
    pool_size = max(rl_cfg.batch_size, 8)
    ds_cfg = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=dataset_paths,
        global_batch_size=pool_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    iterable = PuzzleDataset(ds_cfg, split="train")
    samples: list[dict[str, Any]] = []
    for _set_name, batch, _batch_idx in iterable:
        batch_inputs = batch["inputs"]
        batch_ids = batch["puzzle_identifiers"]
        batch_labels = batch.get("labels")
        batch_size = batch_inputs.shape[0]
        for idx in range(batch_size):
            sample: dict[str, Any] = {
                "inputs": batch_inputs[idx].clone(),
                "puzzle_identifiers": batch_ids[idx].clone(),
                "initial_plan": batch_inputs[idx].clone(),
            }
            if batch_labels is not None:
                sample["solution"] = batch_labels[idx].clone()
            samples.append(sample)
            if len(samples) >= pool_size:
                break
        if len(samples) >= pool_size:
            break

    if not samples:
        raise RuntimeError(f"Failed to materialize evaluation pool from dataset_paths={dataset_paths}")

    num_identifiers = int(
        torch.stack([sample["puzzle_identifiers"] for sample in samples]).max().item() + 1
    )
    dataset = OfflinePuzzleDataset(
        samples=samples,
        seq_len=samples[0]["inputs"].shape[-1],
        vocab_size=iterable.metadata.vocab_size,
        num_identifiers=max(num_identifiers, iterable.metadata.num_puzzle_identifiers),
    )

    seq_len = dataset.seq_len
    vocab_size = dataset.vocab_size

    use_feasibility_checker = getattr(rl_cfg, "use_feasibility_checker", False)
    use_progress_checker = getattr(rl_cfg, "use_progress_checker", False)
    use_constraint_checker = getattr(rl_cfg, "use_constraint_checker", False)
    w_v = getattr(rl_cfg, "feasibility_violation_weight", 2.0)
    w_z = getattr(rl_cfg, "feasibility_zerocand_weight", 5.0)

    if use_feasibility_checker and seq_len in (16, 81):
        checker_fn = make_sudoku_feasibility_checker(w_v=w_v, w_z=w_z)
        checker_kind = "feasibility"
    elif use_progress_checker and seq_len in (16, 81):
        checker_fn = sudoku_progress_checker
        checker_kind = "progress"
    elif use_constraint_checker and seq_len in (16, 81):
        checker_fn = sudoku_constraint_checker
        checker_kind = "constraint"
    else:
        checker_fn = sudoku_checker
        checker_kind = "solution"

    is_sudoku_checker = checker_kind in SUDOKU_CHECKER_KINDS

    env_cfg = PlanEditEnvConfig(
        max_edits=rl_cfg.max_edits,
        gamma=rl_cfg.gamma,
        reward_shaping=rl_cfg.reward_shaping,
        vocab_size=vocab_size,
        solved_threshold=rl_cfg.solved_threshold if is_sudoku_checker else None,
        task_type=getattr(rl_cfg, "task_name", "sudoku"),
        stop_action_mode=getattr(rl_cfg, "stop_action_mode", "noop"),
        stop_action_penalty=getattr(rl_cfg, "stop_action_penalty", -0.1),
        fail_terminal_reward=getattr(rl_cfg, "fail_terminal_reward", 0.0),
        solve_terminal_reward=getattr(rl_cfg, "solve_terminal_reward", 0.0),
        disable_constraint_masking=getattr(rl_cfg, "disable_constraint_masking", False),
    )

    task_config = None
    task_name = getattr(rl_cfg, "task_name", "sudoku")
    if is_sudoku_checker:
        task_config = get_task_config(
            task_name,
            disable_constraint_masking=getattr(rl_cfg, "disable_constraint_masking", False),
        )

    metadata = {
        "dataset_num_samples": len(dataset),
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_identifiers": dataset.num_identifiers,
        "checker_kind": checker_kind,
        "pool_size": pool_size,
        "task_name": task_name,
    }
    return dataset, checker_fn, env_cfg, task_config, metadata


def _evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    seed: int,
    config_yaml: str,
    dataset: Any,
    checker_fn: Any,
    env_cfg: PlanEditEnvConfig,
    task_config: Any,
    device: str,
    num_episodes: int,
    inner_unroll_n: int,
    episodic_latent: bool,
    checker_kind: str,
) -> dict[str, Any]:
    started_at = time.time()
    model, _model_cfg = load_model_for_eval(
        checkpoint_path=str(checkpoint_path),
        device=device,
        config_yaml_path=config_yaml,
    )

    mean_score, success_rate, details = evaluate_plan_policy_with_scores(
        model=model,
        dataset=dataset,
        checker=checker_fn,
        env_cfg=env_cfg,
        task_config=task_config,
        num_episodes=num_episodes,
        inner_unroll_n=inner_unroll_n,
        episodic_latent=episodic_latent,
        greedy=True,
    )
    elapsed_sec = time.time() - started_at

    return {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "success_rate": float(success_rate),
        "mean_score": float(mean_score),
        "mean_return": float(details.get("mean_return", 0.0)),
        "invalid_action_rate": float(details.get("invalid_action_rate", 0.0)),
        "mean_steps": float(details.get("mean_steps", 0.0)),
        "solved_count": int(details.get("solved_count", 0)),
        "total_episodes": int(details.get("total_episodes", 0)),
        "initial_score_mean": float(details.get("initial_score_mean", 0.0)),
        "score_min": float(details.get("score_min", 0.0)),
        "score_max": float(details.get("score_max", 0.0)),
        "eval_policy_mode": "greedy",
        "inner_unroll_n": inner_unroll_n,
        "episodic_latent": episodic_latent,
        "checker_kind": checker_kind,
        "wall_time_sec": elapsed_sec,
    }


def _write_per_seed_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_path = output_dir / "per_seed_metrics.csv"
    fieldnames = [
        "seed",
        "checkpoint_path",
        "success_rate",
        "mean_score",
        "mean_return",
        "invalid_action_rate",
        "mean_steps",
        "solved_count",
        "total_episodes",
        "initial_score_mean",
        "score_min",
        "score_max",
        "eval_policy_mode",
        "inner_unroll_n",
        "episodic_latent",
        "checker_kind",
        "wall_time_sec",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rates = [float(row["success_rate"]) for row in rows]
    mean_returns = [float(row["mean_return"]) for row in rows]
    invalid_rates = [float(row["invalid_action_rate"]) for row in rows]
    mean_scores = [float(row["mean_score"]) for row in rows]
    mean_steps = [float(row["mean_steps"]) for row in rows]

    return {
        "num_seeds": len(rows),
        "success_rate_mean": statistics.mean(success_rates),
        "success_rate_sample_std": _sample_std(success_rates),
        "success_rate_population_std": _population_std(success_rates),
        "mean_return_mean": statistics.mean(mean_returns),
        "mean_return_sample_std": _sample_std(mean_returns),
        "mean_return_population_std": _population_std(mean_returns),
        "invalid_action_rate_mean": statistics.mean(invalid_rates),
        "invalid_action_rate_sample_std": _sample_std(invalid_rates),
        "invalid_action_rate_population_std": _population_std(invalid_rates),
        "mean_score_mean": statistics.mean(mean_scores),
        "mean_score_sample_std": _sample_std(mean_scores),
        "mean_steps_mean": statistics.mean(mean_steps),
        "mean_steps_sample_std": _sample_std(mean_steps),
    }


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rl_cfg = _load_rl_config(args.config_yaml)
    checkpoints = _resolve_checkpoints(
        checkpoint_glob=args.checkpoint_glob,
        seed_regex=args.seed_regex,
        expected_seeds=args.expected_seeds,
    )
    dataset, checker_fn, env_cfg, task_config, bundle_meta = _build_eval_bundle(
        rl_cfg=rl_cfg,
        dataset_paths=args.dataset_paths,
    )

    num_episodes = args.num_episodes or rl_cfg.eval_num_episodes
    rows = []
    run_started_at = time.time()
    for seed, checkpoint_path in checkpoints:
        row = _evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            seed=seed,
            config_yaml=args.config_yaml,
            dataset=dataset,
            checker_fn=checker_fn,
            env_cfg=env_cfg,
            task_config=task_config,
            device=device,
            num_episodes=num_episodes,
            inner_unroll_n=rl_cfg.inner_unroll_n,
            episodic_latent=rl_cfg.episodic_latent,
            checker_kind=bundle_meta["checker_kind"],
        )
        rows.append(row)
        print(
            f"[reeval] seed={seed} success_rate={row['success_rate']:.3f} "
            f"mean_return={row['mean_return']:.3f} "
            f"invalid_action_rate={row['invalid_action_rate']:.3f}"
        )

    aggregate = _aggregate(rows)
    run_metadata = {
        "script": str(Path(__file__).resolve()),
        "repo_git_sha": _git_sha(),
        "checkpoint_glob": args.checkpoint_glob,
        "seed_regex": args.seed_regex,
        "expected_seeds": args.expected_seeds,
        "config_yaml": args.config_yaml,
        "dataset_paths": args.dataset_paths,
        "device": device,
        "num_episodes": num_episodes,
        "rl_cfg": {
            "batch_size": rl_cfg.batch_size,
            "max_edits": rl_cfg.max_edits,
            "gamma": rl_cfg.gamma,
            "inner_unroll_n": rl_cfg.inner_unroll_n,
            "episodic_latent": rl_cfg.episodic_latent,
            "eval_num_episodes": rl_cfg.eval_num_episodes,
            "disable_constraint_masking": getattr(rl_cfg, "disable_constraint_masking", False),
            "use_feasibility_checker": getattr(rl_cfg, "use_feasibility_checker", False),
            "feasibility_violation_weight": getattr(rl_cfg, "feasibility_violation_weight", None),
            "feasibility_zerocand_weight": getattr(rl_cfg, "feasibility_zerocand_weight", None),
        },
        "bundle": bundle_meta,
        "wall_time_sec": time.time() - run_started_at,
    }

    per_seed_csv = _write_per_seed_csv(output_dir, rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_metadata": run_metadata,
                "aggregate": aggregate,
                "per_seed": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"[reeval] per_seed_csv={per_seed_csv}")
    print(f"[reeval] summary_json={summary_path}")
    print(
        f"[reeval] aggregate success={aggregate['success_rate_mean']:.3f} "
        f"+/- {aggregate['success_rate_sample_std']:.3f} "
        f"return={aggregate['mean_return_mean']:.3f} "
        f"+/- {aggregate['mean_return_sample_std']:.3f} "
        f"invalid={aggregate['invalid_action_rate_mean']:.3f} "
        f"+/- {aggregate['invalid_action_rate_sample_std']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
