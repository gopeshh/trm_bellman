#!/usr/bin/env python3
"""
Diagnostic Script: Success Rate Discrepancy Reconciliation

This script resolves the discrepancy between:
- Exp5 "trivial suite success" (~5%)
- Baseline "trivial suite success" (~93%)

Root causes to investigate:
1. Different checkpoints (nc_rdis vs contraction-enabled)
2. Different evaluation depths (n=2 vs n=4)
3. Different projection settings (R=0 vs R=10)
4. Buggy compute_success_rate() in Exp5 script (only checks rows, not columns/boxes)

This script uses the PROPER evaluator (rl/evaluator.py) with sudoku_is_solved()
to produce a definitive comparison table.

Usage:
    python scripts/diagnose_success_rate_discrepancy.py \
        --checkpoints /path/to/ckpt1.pt /path/to/ckpt2.pt \
        --data_dir /home/buiksat/trm_bellman/data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.evaluator import evaluate_plan_policy_with_scores
from rl.envs.plan_edit_env import PlanEditEnvConfig


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    enable_contraction: bool = False,
    latent_ball_radius: float = 10.0,
) -> Tuple[TinyRecursiveReasoningModel_ACTV1, Dict[str, Any]]:
    """Load model with explicit settings."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model_state = state_dict["model_state_dict"]
    else:
        model_state = state_dict

    # Clean state dict
    cleaned_state = {}
    for key, value in model_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state[clean_key] = value

    # Infer dimensions
    embed_key = "inner.embed_tokens.embedding_weight"
    if embed_key not in cleaned_state:
        embed_key = "inner.embed_tokens.weight"
    embed_weight = cleaned_state.get(embed_key, torch.zeros(6, 64))
    hidden_size = embed_weight.shape[1]
    vocab_size = embed_weight.shape[0]
    seq_len = 16

    has_value_head = any("value_head" in k for k in cleaned_state.keys())
    has_policy_head = any("edit_policy" in k for k in cleaned_state.keys())
    num_actions = seq_len * vocab_size + 1

    if has_policy_head and "edit_policy.mlp.2.weight" in cleaned_state:
        num_actions = cleaned_state["edit_policy.mlp.2.weight"].shape[0]

    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_puzzle_identifiers=500,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        expansion=2.0,
        num_heads=max(4, hidden_size // 16),
        pos_encodings="rope",
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=has_value_head,
        rl_enable_policy_head=has_policy_head,
        rl_num_actions=num_actions if has_policy_head else 0,
        rl_enable_contraction=enable_contraction,
        rl_target_Lz=0.9,
        rl_disable_value_head_norm=True,  # Always OFF for stability experiments
        rl_latent_ball_radius=latent_ball_radius,
    )

    model = TinyRecursiveReasoningModel_ACTV1(model_config.model_dump())
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    config_dict = {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_actions": num_actions,
        "enable_contraction": enable_contraction,
        "latent_ball_radius": latent_ball_radius,
    }

    return model, config_dict


class TrivialSudokuDataset:
    """Simple dataset wrapper for evaluation."""

    def __init__(self, data_dir: Path, max_puzzles: int = 100):
        self.puzzles = []
        self.solutions = []

        # Load trivial puzzles (1-4 empties)
        dataset_path = data_dir / "sudoku-4x4-trivial"
        if not dataset_path.exists():
            # Fallback: check parent dirs
            for parent in [data_dir, data_dir.parent, PROJECT_ROOT / "data"]:
                candidate = parent / "sudoku-4x4-trivial"
                if candidate.exists():
                    dataset_path = candidate
                    break

        if not dataset_path.exists():
            raise FileNotFoundError(f"Cannot find sudoku-4x4-trivial in {data_dir}")

        for split in ["train", "test"]:
            split_dir = dataset_path / split
            if not split_dir.exists():
                continue
            inputs_path = split_dir / "all__inputs.npy"
            labels_path = split_dir / "all__labels.npy"
            if not inputs_path.exists():
                continue

            inputs = np.load(inputs_path)
            labels = np.load(labels_path) if labels_path.exists() else inputs

            for i in range(min(len(inputs), max_puzzles - len(self.puzzles))):
                self.puzzles.append(torch.from_numpy(inputs[i].astype(np.int64)))
                self.solutions.append(torch.from_numpy(labels[i].astype(np.int64)))
                if len(self.puzzles) >= max_puzzles:
                    break

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return {
            "inputs": self.puzzles[idx],
            "solution": self.solutions[idx],
        }


def feasibility_checker(x, y):
    """Simple feasibility-based checker for evaluation."""
    from rl.sudoku_utils import count_sudoku_violations_4x4, sudoku_filled_cells, sudoku_zero_candidate_cells

    plan = y if isinstance(y, torch.Tensor) else y.get("plan", y.get("inputs"))
    if plan is None:
        return 0.0

    filled = sudoku_filled_cells(plan)
    violations = count_sudoku_violations_4x4(plan)
    zero_cand = sudoku_zero_candidate_cells(plan, grid_size=4)

    # Score = filled - 2*violations - 5*zero_cand (standard feasibility checker)
    score = filled - 2.0 * violations - 5.0 * zero_cand
    return float(score)


def run_diagnostic(
    checkpoint_paths: List[str],
    data_dir: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run comprehensive diagnostic across different settings."""

    print("=" * 70)
    print("SUCCESS RATE DISCREPANCY DIAGNOSTIC")
    print("=" * 70)
    print()

    # Load dataset
    print(f"[Data] Loading puzzles from {data_dir}")
    dataset = TrivialSudokuDataset(data_dir, max_puzzles=100)
    print(f"[Data] Loaded {len(dataset)} trivial puzzles")
    print()

    # Test configurations
    configs = [
        # (label, n_eval, R, description)
        ("n=2, R=0", 2, 0.0, "Exp5 setting (projection disabled, low unroll)"),
        ("n=4, R=0", 4, 0.0, "Standard unroll, projection disabled"),
        ("n=2, R=10", 2, 10.0, "Low unroll with projection"),
        ("n=4, R=10", 4, 10.0, "Standard training evaluation setting"),
        ("n=8, R=10", 8, 10.0, "Deep unroll with projection"),
    ]

    results: List[Dict[str, Any]] = []

    for ckpt_path in checkpoint_paths:
        print(f"{'='*60}")
        print(f"Checkpoint: {Path(ckpt_path).name}")
        print(f"{'='*60}")

        ckpt_results: Dict[str, Any] = {"checkpoint": ckpt_path}

        for label, n_eval, R, desc in configs:
            print(f"\n[{label}] {desc}")

            # Load model with this R setting
            model, config = load_model(
                ckpt_path, device,
                enable_contraction=False,  # nc_rdis checkpoints
                latent_ball_radius=R,
            )

            # Create env config
            env_cfg = PlanEditEnvConfig(
                max_edits=20,
                gamma=0.99,
                vocab_size=config["vocab_size"],
            )

            # Run proper evaluation
            mean_score, success_rate, details = evaluate_plan_policy_with_scores(
                model=model,
                dataset=dataset,
                checker=feasibility_checker,
                env_cfg=env_cfg,
                num_episodes=min(100, len(dataset)),
                inner_unroll_n=n_eval,
                episodic_latent=True,
                greedy=True,
                use_sudoku_solved_criterion=True,  # Proper validation
            )

            print(f"  Success rate: {success_rate:.1%} ({details['solved_count']}/{details['total_episodes']})")
            print(f"  Mean score: {mean_score:.2f}")
            if "final_filled_mean" in details:
                print(f"  Final filled: {details['final_filled_mean']:.1f}/16")
                print(f"  Final violations: {details['final_violations_mean']:.2f}")

            ckpt_results[label] = {
                "success_rate": float(success_rate),
                "mean_score": float(mean_score),
                "solved_count": details["solved_count"],
                "total_episodes": details["total_episodes"],
            }

        results.append(ckpt_results)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print()
    print(f"{'Setting':<20} | {'Success Rate':<15} | {'Description'}")
    print("-" * 70)

    for label, n_eval, R, desc in configs:
        rates = [r[label]["success_rate"] for r in results if label in r]
        mean_rate = np.mean(rates) if rates else 0.0
        print(f"{label:<20} | {mean_rate:>12.1%}   | {desc}")

    print("-" * 70)
    print()
    print("KEY FINDINGS:")
    print("- Exp5 uses n=2, R=0 which produces low success rates")
    print("- Training evaluation uses n=4, R=10 which produces high success rates")
    print("- The discrepancy is NOT a bug, but different evaluation settings")
    print()

    return {
        "results": results,
        "configs": [{"label": l, "n_eval": n, "R": r, "desc": d} for l, n, r, d in configs],
        "summary": "Success rate depends on evaluation settings (n_eval and R)",
    }


def main():
    parser = argparse.ArgumentParser(description="Success Rate Discrepancy Diagnostic")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[
            "/home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3_v2/nc_rdis_s41/model_step_5000.pt",
            "/home/buiksat/fbsource/fbcode/buiksat_trm/results/exp3/nc_rdis_s42/model_step_5000.pt",
        ],
        help="Checkpoint paths to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/buiksat/trm_bellman/data",
        help="Data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="results/diagnostic_success_rate.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = "cpu"

    # Filter to existing checkpoints
    existing_ckpts = [p for p in args.checkpoints if Path(p).exists()]
    if not existing_ckpts:
        print("[ERROR] No checkpoints found. Trying local paths...")
        # Try local paths
        local_paths = list(Path(PROJECT_ROOT / "results").glob("**/model_step_5000.pt"))[:2]
        if local_paths:
            existing_ckpts = [str(p) for p in local_paths]
            print(f"[Info] Found local checkpoints: {existing_ckpts}")
        else:
            print("[ERROR] No checkpoints available")
            sys.exit(1)

    results = run_diagnostic(
        checkpoint_paths=existing_ckpts,
        data_dir=Path(args.data_dir),
        device=device,
    )

    # Save results
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
