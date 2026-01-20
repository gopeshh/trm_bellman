#!/usr/bin/env python3
"""
Diagnostic script to reconcile the success rate discrepancy between:
- Training evaluation: ~88% success
- Exp5 evaluation: ~6-7% success

This script evaluates the same checkpoints using the exact training evaluator
and compares with the Exp5 evaluation protocol.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_and_config(checkpoint_path: str, config_yaml_path: str, device: str = "cuda"):
    """Load model from checkpoint with config."""
    from models.recursive_reasoning.trm import TRM

    # Load config
    with open(config_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Create model
    model = TRM(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_heads=config.get("num_heads", 2),
        num_layers=config.get("num_layers", 2),
        inner_unroll_n=config.get("inner_unroll_n", 2),
        num_actions=config["num_actions"],
        enable_contraction=config.get("enable_contraction", False),
        target_Lz=config.get("target_Lz", 0.9),
        disable_value_head_norm=config.get("disable_value_head_norm", True),
        latent_ball_radius=config.get("latent_ball_radius", 0.0),
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, config


def evaluate_with_training_protocol(
    model: torch.nn.Module,
    config: Dict[str, Any],
    data_dir: Path,
    device: str,
    n_episodes: int = 50,
    max_edits: int = 20,
    inner_unroll_n: int = 2,
) -> Dict[str, Any]:
    """Evaluate using exact training evaluation protocol."""
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
    from rl.evaluator import evaluate_plan_policy_with_scores
    from rl.sudoku_utils import sudoku_get_stats
    from rl.datasets import load_sudoku_dataset

    # Load the trivial dataset (same as training)
    trivial_dir = data_dir / "sudoku-4x4-trivial"
    train_dataset = load_sudoku_dataset(str(trivial_dir / "train"))

    # Create checker (same as training)
    def checker(x, y):
        if torch.is_tensor(y):
            plan = y
        elif isinstance(y, dict):
            plan = y.get("plan", y.get("labels"))
        else:
            plan = torch.as_tensor(y)
        total_cells, filled, violations, zero_cand = sudoku_get_stats(plan)
        return float(filled) - 2.0 * float(violations) - 5.0 * float(zero_cand)

    # Create environment config (same as training)
    env_cfg = PlanEditEnvConfig(
        max_edits=max_edits,
        gamma=0.99,
        vocab_size=config["vocab_size"],
        reward_shaping=True,
        stop_action_mode="disabled",
        task_type="sudoku",
    )

    # Use training evaluator
    mean_score, success_rate, detailed_stats = evaluate_plan_policy_with_scores(
        model=model,
        dataset=train_dataset,
        checker=checker,
        env_cfg=env_cfg,
        num_episodes=n_episodes,
        inner_unroll_n=inner_unroll_n,
        episodic_latent=True,
        greedy=True,
        use_sudoku_solved_criterion=True,
    )

    # Compute initial score distribution
    initial_scores = []
    for i in range(min(n_episodes, len(train_dataset))):
        sample = train_dataset[i]
        if isinstance(sample, dict):
            plan = sample.get("initial_plan", sample.get("inputs"))
        else:
            plan = sample
        if torch.is_tensor(plan):
            _, filled, violations, zero_cand = sudoku_get_stats(plan)
            initial_scores.append(float(filled) - 2.0 * float(violations) - 5.0 * float(zero_cand))

    return {
        "success_rate": success_rate,
        "mean_score": mean_score,
        "initial_score_mean": np.mean(initial_scores) if initial_scores else 0,
        "initial_score_std": np.std(initial_scores) if initial_scores else 0,
        "n_episodes": n_episodes,
    }


def evaluate_with_exp5_protocol(
    model: torch.nn.Module,
    config: Dict[str, Any],
    data_dir: Path,
    device: str,
    n_episodes: int = 100,
    max_steps: int = 20,
    n_unroll: int = 2,
) -> Dict[str, Any]:
    """Evaluate using the Exp5 evaluation protocol."""
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
    from rl.evaluator import evaluate_plan_policy_with_scores
    from rl.sudoku_utils import sudoku_get_stats

    # Load puzzles the way Exp5 does
    trivial_puzzles = []
    for path in data_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("sudoku-4x4"):
            continue
        if "6to8" in path.name or "hard" in path.name:
            continue

        for split in ["train", "test"]:
            split_dir = path / split
            if not split_dir.exists():
                continue
            inputs_path = split_dir / "all__inputs.npy"
            if not inputs_path.exists():
                continue

            inputs = np.load(inputs_path)
            for i in range(len(inputs)):
                inp = torch.from_numpy(inputs[i].astype(np.int64))
                empties = int((inp == 1).sum().item())
                if 1 <= empties <= 4:  # trivial
                    trivial_puzzles.append({
                        "inputs": inp,
                        "initial_plan": inp.clone(),
                        "puzzle_identifiers": torch.tensor(i, dtype=torch.long),
                        "solution": inp.clone(),
                    })

    # Create a dataset-like object
    class PuzzleListDataset:
        def __init__(self, puzzles):
            self.puzzles = puzzles

        def __len__(self):
            return len(self.puzzles)

        def __getitem__(self, idx):
            return self.puzzles[idx]

    # Sample puzzles
    rng = np.random.default_rng(42)
    if len(trivial_puzzles) > n_episodes:
        indices = rng.choice(len(trivial_puzzles), size=n_episodes, replace=False)
        eval_puzzles = [trivial_puzzles[i] for i in indices]
    else:
        eval_puzzles = trivial_puzzles

    dataset = PuzzleListDataset(eval_puzzles)

    # Create checker
    def checker(x, y):
        if torch.is_tensor(y):
            plan = y
        elif isinstance(y, dict):
            plan = y.get("plan", y.get("labels"))
        else:
            plan = torch.as_tensor(y)
        _, filled, violations, zero_cand = sudoku_get_stats(plan)
        return float(filled) - 2.0 * float(violations) - 5.0 * float(zero_cand)

    # Create environment config
    env_cfg = PlanEditEnvConfig(
        max_edits=max_steps,
        gamma=0.99,
        vocab_size=config["vocab_size"],
        reward_shaping=True,
        stop_action_mode="disabled",
        task_type="sudoku",
    )

    # Use training evaluator
    mean_score, success_rate, _ = evaluate_plan_policy_with_scores(
        model=model,
        dataset=dataset,
        checker=checker,
        env_cfg=env_cfg,
        num_episodes=len(eval_puzzles),
        inner_unroll_n=n_unroll,
        episodic_latent=True,
        greedy=True,
        use_sudoku_solved_criterion=True,
    )

    # Compute initial scores
    initial_scores = []
    for p in eval_puzzles[:n_episodes]:
        plan = p["initial_plan"]
        _, filled, violations, zero_cand = sudoku_get_stats(plan)
        initial_scores.append(float(filled) - 2.0 * float(violations) - 5.0 * float(zero_cand))

    return {
        "success_rate": success_rate,
        "mean_score": mean_score,
        "initial_score_mean": np.mean(initial_scores),
        "initial_score_std": np.std(initial_scores),
        "n_episodes": len(eval_puzzles),
        "n_trivial_puzzles_total": len(trivial_puzzles),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose success rate discrepancy")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--config_yaml", type=str, required=True, help="Config YAML path")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "data"

    print("=" * 70)
    print("SUCCESS RATE DISCREPANCY DIAGNOSTIC")
    print("=" * 70)
    print()

    results = []

    for ckpt_path in args.checkpoints:
        print(f"\n[Checkpoint] {ckpt_path}")
        print("-" * 60)

        model, config = load_model_and_config(ckpt_path, args.config_yaml, args.device)

        # Test with training protocol
        print("[Training Protocol] Evaluating...")
        training_result = evaluate_with_training_protocol(
            model, config, data_dir, args.device,
            n_episodes=50, max_edits=20, inner_unroll_n=2,
        )
        print(f"  Success: {training_result['success_rate']*100:.1f}%")
        print(f"  Mean Score: {training_result['mean_score']:.2f}")
        print(f"  Initial Score: {training_result['initial_score_mean']:.2f}±{training_result['initial_score_std']:.2f}")

        # Test with Exp5 protocol
        print("[Exp5 Protocol] Evaluating...")
        exp5_result = evaluate_with_exp5_protocol(
            model, config, data_dir, args.device,
            n_episodes=100, max_steps=20, n_unroll=2,
        )
        print(f"  Success: {exp5_result['success_rate']*100:.1f}%")
        print(f"  Mean Score: {exp5_result['mean_score']:.2f}")
        print(f"  Initial Score: {exp5_result['initial_score_mean']:.2f}±{exp5_result['initial_score_std']:.2f}")
        print(f"  Total Trivial Puzzles: {exp5_result['n_trivial_puzzles_total']}")

        results.append({
            "checkpoint": ckpt_path,
            "training_protocol": training_result,
            "exp5_protocol": exp5_result,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        train_success = r["training_protocol"]["success_rate"]
        exp5_success = r["exp5_protocol"]["success_rate"]
        discrepancy = abs(train_success - exp5_success)

        print(f"\n{r['checkpoint']}")
        print(f"  Training: {train_success*100:.1f}%")
        print(f"  Exp5: {exp5_success*100:.1f}%")
        print(f"  Discrepancy: {discrepancy*100:.1f}pp")

        if discrepancy > 0.1:
            print("  ⚠ SIGNIFICANT DISCREPANCY DETECTED")

    # Save results
    summary_path = out_path / "diagnostic_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
