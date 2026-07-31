#!/usr/bin/env python3
"""
Random Baseline Evaluation for 4x4 Feasibility Sudoku.

Evaluates a uniform random policy over VALID masked actions.
This provides a fair baseline for comparison with learned policies.

Usage:
    buck2 run //buiksat_trm:eval_random_baseline -- \
        --dataset-path buiksat_trm/data/sudoku-4x4-trivial \
        --seeds 42 123 456 \
        --num-episodes 50 \
        --output buiksat_trm/results/plot_data/random_baseline_feasibility.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def load_puzzles(dataset_path: str, split: str = "train") -> List[Dict]:
    """Load puzzles from dataset directory (NumPy format)."""
    split_path = Path(dataset_path) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    # Load NumPy arrays
    inputs_path = split_path / "all__inputs.npy"
    labels_path = split_path / "all__labels.npy"

    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found: {inputs_path}")

    inputs = np.load(inputs_path)
    labels = np.load(labels_path) if labels_path.exists() else None

    puzzles = []
    for i in range(len(inputs)):
        puzzle = {"inputs": inputs[i].tolist()}
        if labels is not None:
            puzzle["solution"] = labels[i].tolist()
        puzzles.append(puzzle)

    return puzzles


def compute_sudoku_stats(plan: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Compute Sudoku statistics.
    Returns (total_cells, filled, violations, zero_cand).
    """
    if plan.numel() == 16:  # 4x4
        n = 4
        box_h, box_w = 2, 2
    elif plan.numel() == 81:  # 9x9
        n = 9
        box_h, box_w = 3, 3
    else:
        raise ValueError(f"Unknown Sudoku size: {plan.numel()}")

    grid = plan.view(n, n)

    # Count filled cells
    filled = int((grid != 0).sum().item())

    # Count violations
    violations = 0
    for i in range(n):
        # Row violations
        row = grid[i, :]
        row_vals = row[row != 0]
        violations += len(row_vals) - len(torch.unique(row_vals))

        # Column violations
        col = grid[:, i]
        col_vals = col[col != 0]
        violations += len(col_vals) - len(torch.unique(col_vals))

    # Box violations
    for box_r in range(n // box_h):
        for box_c in range(n // box_w):
            box = grid[box_r * box_h:(box_r + 1) * box_h,
                       box_c * box_w:(box_c + 1) * box_w].flatten()
            box_vals = box[box != 0]
            violations += len(box_vals) - len(torch.unique(box_vals))

    # Count zero-candidate cells (simplified: empty cells with no valid candidates)
    zero_cand = 0
    for i in range(n):
        for j in range(n):
            if grid[i, j] == 0:
                # Check what values are available
                used = set()
                used.update(grid[i, :].tolist())  # row
                used.update(grid[:, j].tolist())  # column
                # Box
                box_r, box_c = (i // box_h) * box_h, (j // box_w) * box_w
                used.update(grid[box_r:box_r + box_h, box_c:box_c + box_w].flatten().tolist())
                used.discard(0)
                if len(used) == n:  # No candidates available
                    zero_cand += 1

    return n * n, filled, violations, zero_cand


def is_solved(plan: torch.Tensor) -> bool:
    """Check if the Sudoku is solved (filled and no violations)."""
    total, filled, violations, _ = compute_sudoku_stats(plan)
    return filled == total and violations == 0


def feasibility_score(plan: torch.Tensor, w_v: float = 2.0, w_z: float = 5.0) -> float:
    """Compute feasibility score: filled - w_v * violations - w_z * zero_cand."""
    _, filled, violations, zero_cand = compute_sudoku_stats(plan)
    return float(filled - w_v * violations - w_z * zero_cand)


def get_valid_actions(plan: torch.Tensor, clue_mask: torch.Tensor) -> List[int]:
    """
    Get valid actions for a 4x4 Sudoku.
    Action space: 16 positions × 6 values (0-5 where 0=clear, 1-4=digits) + STOP = 97 actions.
    Mask: Cannot edit clue cells.
    """
    n = 4
    valid = []
    for pos in range(16):
        if clue_mask[pos] == 0:  # Not a clue cell
            for val in range(6):  # 0=clear, 1-4=digits, 5=unused but in action space
                if val <= 4:  # Valid values for 4x4
                    action = pos * 6 + val
                    valid.append(action)
    return valid


def apply_action(plan: torch.Tensor, action: int) -> torch.Tensor:
    """Apply an edit action to the plan."""
    if action == 96:  # STOP action
        return plan
    pos = action // 6
    val = action % 6
    new_plan = plan.clone()
    new_plan[pos] = val
    return new_plan


def evaluate_random_policy(
    puzzles: List[Dict],
    max_edits: int = 16,
    num_episodes: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate a uniform random policy over valid actions."""
    random.seed(seed)
    torch.manual_seed(seed)

    num_solved = 0
    total_score = 0.0
    all_filled = []
    all_violations = []
    all_zero_cand = []

    for ep_idx in range(num_episodes):
        puzzle = puzzles[ep_idx % len(puzzles)]
        clues = torch.tensor(puzzle["inputs"], dtype=torch.long)
        plan = clues.clone()  # Start with clues
        clue_mask = (clues != 0).long()  # 1 for clue cells

        for step in range(max_edits):
            # Check if already solved
            if is_solved(plan):
                break

            # Get valid actions (exclude editing clue cells)
            valid_actions = get_valid_actions(plan, clue_mask)

            if not valid_actions:
                break

            # Sample uniformly from valid actions
            action = random.choice(valid_actions)
            plan = apply_action(plan, action)

        # Compute final stats
        _, filled, violations, zero_cand = compute_sudoku_stats(plan)
        final_score = feasibility_score(plan)

        total_score += final_score
        all_filled.append(filled)
        all_violations.append(violations)
        all_zero_cand.append(zero_cand)

        if is_solved(plan):
            num_solved += 1

    # Compute means
    return {
        "success_rate": num_solved / num_episodes,
        "mean_score": total_score / num_episodes,
        "final_filled_mean": sum(all_filled) / len(all_filled),
        "final_violations_mean": sum(all_violations) / len(all_violations),
        "final_zero_cand_mean": sum(all_zero_cand) / len(all_zero_cand),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate random baseline for feasibility Sudoku")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--num-episodes", type=int, default=50, help="Episodes per seed")
    parser.add_argument("--max-edits", type=int, default=16, help="Max edits per episode")
    parser.add_argument("--output", type=str, default="results/plot_data/random_baseline_feasibility.csv")
    args = parser.parse_args()

    # Load puzzles
    puzzles = load_puzzles(args.dataset_path, split="train")
    print(f"Loaded {len(puzzles)} puzzles from {args.dataset_path}")

    # Evaluate for each seed
    results = []
    for seed in args.seeds:
        print(f"\nEvaluating random baseline with seed={seed}...")
        metrics = evaluate_random_policy(
            puzzles=puzzles,
            max_edits=args.max_edits,
            num_episodes=args.num_episodes,
            seed=seed,
        )
        metrics["seed"] = seed
        results.append(metrics)

        print(f"  success_rate: {metrics['success_rate']:.3f}")
        print(f"  mean_score: {metrics['mean_score']:.3f}")
        print(f"  filled_mean: {metrics['final_filled_mean']:.2f}")
        print(f"  violations_mean: {metrics['final_violations_mean']:.2f}")
        print(f"  zero_cand_mean: {metrics['final_zero_cand_mean']:.2f}")

    # Compute aggregate stats
    agg_success = sum(r["success_rate"] for r in results) / len(results)
    agg_score = sum(r["mean_score"] for r in results) / len(results)
    agg_filled = sum(r["final_filled_mean"] for r in results) / len(results)
    agg_violations = sum(r["final_violations_mean"] for r in results) / len(results)
    agg_zero_cand = sum(r["final_zero_cand_mean"] for r in results) / len(results)

    print(f"\n=== AGGREGATE RANDOM BASELINE ===")
    print(f"Mean success_rate: {agg_success:.3f}")
    print(f"Mean mean_score: {agg_score:.3f}")
    print(f"Mean filled: {agg_filled:.2f}")
    print(f"Mean violations: {agg_violations:.2f}")
    print(f"Mean zero_cand: {agg_zero_cand:.2f}")

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "success_rate", "mean_score",
            "final_filled_mean", "final_violations_mean", "final_zero_cand_mean"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seed": r["seed"],
                "success_rate": f"{r['success_rate']:.4f}",
                "mean_score": f"{r['mean_score']:.4f}",
                "final_filled_mean": f"{r['final_filled_mean']:.4f}",
                "final_violations_mean": f"{r['final_violations_mean']:.4f}",
                "final_zero_cand_mean": f"{r['final_zero_cand_mean']:.4f}",
            })
        # Add aggregate row
        writer.writerow({
            "seed": "mean",
            "success_rate": f"{agg_success:.4f}",
            "mean_score": f"{agg_score:.4f}",
            "final_filled_mean": f"{agg_filled:.4f}",
            "final_violations_mean": f"{agg_violations:.4f}",
            "final_zero_cand_mean": f"{agg_zero_cand:.4f}",
        })

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
