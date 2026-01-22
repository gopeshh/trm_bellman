#!/usr/bin/env python3
"""
Evaluate random policy on 4x4 Sudoku to establish baseline success rate.

This script is consistent with the training/eval harness used by PPO/A2C/DQN/UPI-TRM
baselines. It uses the same:
- Dataset loading format (NumPy arrays)
- PlanEditEnv with proper stop_action_id
- Feasibility checker with same weights (2.0 violations, 5.0 zero-cand)
- Success criterion: sudoku_is_solved (all filled, no violations)
- Action masking (only valid edit actions, excluding clue cells)

Usage:
    python scripts/eval_random_policy.py \\
        --dataset-path data/sudoku-4x4-ultra-easy \\
        --seeds 42 123 456 \\
        --num-episodes 50 \\
        --max-edits 16 \\
        --output results/plot_data/random_baseline.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_utils import (
    sudoku_is_solved,
    sudoku_get_stats,
    sudoku_filled_cells,
    count_sudoku_violations_4x4,
    sudoku_zero_candidate_cells,
)


def load_dataset_npy(
    dataset_path: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load puzzles from dataset directory (NumPy format).
    
    Matches the format expected by PlanEditEnv:
    - inputs: [seq_len] tensor of tokens (1=empty, 2-5=digits for 4x4)
    - puzzle_identifiers: scalar or [1] tensor
    - solution: [seq_len] tensor (optional, for max_score calculation)
    - initial_plan: [seq_len] tensor (copy of inputs, starts with clues)
    """
    split_path = Path(dataset_path) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    inputs_path = split_path / "all__inputs.npy"
    labels_path = split_path / "all__labels.npy"
    puzzle_ids_path = split_path / "all__puzzle_identifiers.npy"

    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found: {inputs_path}")

    inputs = np.load(inputs_path)
    labels = np.load(labels_path) if labels_path.exists() else None
    puzzle_ids = np.load(puzzle_ids_path) if puzzle_ids_path.exists() else None

    samples = []
    n = len(inputs) if max_samples is None else min(len(inputs), max_samples)
    
    for i in range(n):
        sample = {
            "inputs": torch.tensor(inputs[i], dtype=torch.long),
            "puzzle_identifiers": torch.tensor(
                puzzle_ids[i] if puzzle_ids is not None else i,
                dtype=torch.long
            ),
            "initial_plan": torch.tensor(inputs[i], dtype=torch.long),  # Start with clues
        }
        if labels is not None:
            sample["solution"] = torch.tensor(labels[i], dtype=torch.long)
        samples.append(sample)

    return samples


def sudoku_feasibility_checker(x: Dict, y: torch.Tensor, w_v: float = 2.0, w_z: float = 5.0) -> float:
    """
    Feasibility-aware Sudoku checker - matches upi_trm_train.sudoku_feasibility_checker.
    
    Score = filled - w_v * violations - w_z * zeroCand
    
    Args:
        x: Instance dict (unused, for API compatibility)
        y: Plan tensor [seq_len]
        w_v: Violation penalty weight (default: 2.0)
        w_z: Zero-candidate penalty weight (default: 5.0)
    
    Returns:
        Score where:
        - Maximum (solved): total_cells (16 for 4x4)
        - Initial: filled_cells (clue count)
        - Violations: filled - w_v * violations
        - Dead-end: heavily penalized by -w_z * zeroCand
    """
    plan = y.to(torch.long) if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
    total_cells = plan.numel()

    if total_cells == 16:
        filled = sudoku_filled_cells(plan, empty_token=1)
        violations = count_sudoku_violations_4x4(plan)
        zero_cand = sudoku_zero_candidate_cells(plan, grid_size=4)
    elif total_cells == 81:
        from rl.sudoku_utils import count_sudoku_violations_9x9
        filled = sudoku_filled_cells(plan, empty_token=1)
        violations = count_sudoku_violations_9x9(plan)
        zero_cand = sudoku_zero_candidate_cells(plan, grid_size=9)
    else:
        # Unknown grid size
        return 0.0

    score = float(filled - w_v * violations - w_z * zero_cand)
    return score


class OfflineDatasetWrapper:
    """Minimal dataset wrapper for PlanEditEnv compatibility."""
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def evaluate_random_policy(
    dataset: OfflineDatasetWrapper,
    env_cfg: PlanEditEnvConfig,
    checker_fn,
    num_episodes: int = 50,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a uniform random policy over valid (masked) actions.
    
    This matches baseline evaluation:
    - Uses PlanEditEnv with action masking
    - Samples uniformly from valid actions only
    - Uses info["solved"] as success criterion (sudoku_is_solved)
    - Tracks same metrics as baseline eval logs
    
    Args:
        dataset: Dataset of puzzle samples
        env_cfg: Environment configuration (must match baselines)
        checker_fn: Checker function (x, y) -> score
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        verbose: Print per-episode details
    
    Returns:
        Dict with success_rate, mean_score, and Sudoku-specific metrics
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = PlanEditEnv(dataset=dataset, checker=checker_fn, config=env_cfg)
    
    # Compute action space size
    # For 4x4 Sudoku: 16 positions × 6 values (0-5) = 96 edit actions + 1 STOP = 97
    # But baselines use stop_action_mode="disabled", so STOP is masked out
    sample = dataset[0]
    seq_len = sample["inputs"].shape[-1]
    vocab_size = env_cfg.vocab_size or 6  # Default for 4x4 Sudoku
    num_edit_actions = seq_len * vocab_size
    num_actions = num_edit_actions + 1  # +1 for STOP action
    
    # Set stop_action_id (must be done before stepping)
    env.set_stop_action_id(stop_id=num_actions - 1)
    
    # Track metrics
    num_solved = 0
    total_score = 0.0
    all_final_scores: List[float] = []
    all_initial_scores: List[float] = []
    all_filled: List[int] = []
    all_violations: List[int] = []
    all_zero_cand: List[int] = []
    
    for ep_idx in range(num_episodes):
        # Reset environment with specific puzzle
        x, y = env.reset(idx=ep_idx % len(dataset))
        done = False
        
        # Track initial score
        initial_score = float(checker_fn(x, y))
        all_initial_scores.append(initial_score)
        
        step_count = 0
        while not done and step_count < env_cfg.max_edits:
            # Get action mask (True = valid action)
            action_mask = env.get_action_mask()
            
            if action_mask is None:
                # No masking - sample from all actions
                action = random.randint(0, num_actions - 1)
            else:
                # Sample uniformly from valid actions only
                valid_actions = torch.where(action_mask)[0].tolist()
                if not valid_actions:
                    # No valid actions - force termination
                    break
                action = random.choice(valid_actions)
            
            # Step environment
            (x_next, y_next), reward, done, info = env.step(action)
            x, y = x_next, y_next
            step_count += 1
        
        # Get final metrics
        final_score = float(checker_fn(x, y))
        total_score += final_score
        all_final_scores.append(final_score)
        
        # Get Sudoku-specific stats
        final_plan = y if torch.is_tensor(y) else torch.tensor(y)
        if final_plan.numel() in (16, 81):
            total_cells, filled, violations, zero_cand = sudoku_get_stats(final_plan)
            all_filled.append(filled)
            all_violations.append(violations)
            all_zero_cand.append(zero_cand)
            
            # Check solved using same criterion as baselines
            solved = sudoku_is_solved(final_plan)
        else:
            solved = info.get("solved", False)
        
        if solved:
            num_solved += 1
        
        if verbose:
            print(f"  Episode {ep_idx}: score={final_score:.2f}, solved={solved}, steps={step_count}")
    
    # Compute aggregate metrics
    success_rate = num_solved / max(num_episodes, 1)
    mean_score = total_score / max(num_episodes, 1)
    
    result = {
        "seed": seed,
        "success_rate": success_rate,
        "mean_score": mean_score,
        "solved_count": num_solved,
        "num_episodes": num_episodes,
        "max_edits": env_cfg.max_edits,
        "score_min": min(all_final_scores) if all_final_scores else 0.0,
        "score_max": max(all_final_scores) if all_final_scores else 0.0,
        "initial_score_mean": sum(all_initial_scores) / len(all_initial_scores) if all_initial_scores else 0.0,
    }
    
    # Add Sudoku-specific stats if available
    if all_filled:
        result["final_filled_mean"] = sum(all_filled) / len(all_filled)
        result["final_violations_mean"] = sum(all_violations) / len(all_violations)
        result["final_zero_cand_mean"] = sum(all_zero_cand) / len(all_zero_cand)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate random policy on 4x4 Sudoku (consistent with baseline harness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (3 seeds, 50 episodes each)
    python scripts/eval_random_policy.py --dataset-path data/sudoku-4x4-ultra-easy
    
    # Run with specific seeds and output to CSV
    python scripts/eval_random_policy.py \\
        --dataset-path data/sudoku-4x4-ultra-easy \\
        --seeds 42 123 456 \\
        --output results/plot_data/random_baseline.csv
        
    # Quick test with fewer episodes
    python scripts/eval_random_policy.py \\
        --dataset-path data/sudoku-4x4-ultra-easy \\
        --num-episodes 10 --seeds 42
"""
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., data/sudoku-4x4-ultra-easy)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds to evaluate (default: 42 123 456)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per seed (default: 50)"
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=16,
        help="Maximum edits per episode (default: 16, matches baselines)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99, matches baselines)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=6,
        help="Vocabulary size for 4x4 Sudoku (default: 6: 0=pad, 1=empty, 2-5=digits)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (also saves .json alongside)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode details"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RANDOM BASELINE EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per seed: {args.num_episodes}")
    print(f"Max edits: {args.max_edits}")
    print(f"Gamma: {args.gamma}")
    print()

    # Load dataset
    samples = load_dataset_npy(args.dataset_path, split="train")
    dataset = OfflineDatasetWrapper(samples)
    print(f"Loaded {len(dataset)} puzzles from {args.dataset_path}")
    
    # Get sequence length from first sample
    seq_len = samples[0]["inputs"].shape[-1]
    print(f"Sequence length: {seq_len} (grid size: {int(seq_len**0.5)}x{int(seq_len**0.5)})")

    # Create environment config matching baselines
    # IMPORTANT: stop_action_mode="disabled" matches baseline configs
    env_cfg = PlanEditEnvConfig(
        max_edits=args.max_edits,
        gamma=args.gamma,
        reward_shaping=True,
        vocab_size=args.vocab_size,
        task_type="sudoku",
        stop_action_mode="disabled",  # Matches baseline configs
        stop_action_penalty=-0.1,
        fail_terminal_reward=-16.0,
        solve_terminal_reward=1.0,
    )
    print(f"Stop action mode: {env_cfg.stop_action_mode}")
    print()

    # Checker function matching baselines (w_v=2.0, w_z=5.0)
    def checker_fn(x, y):
        return sudoku_feasibility_checker(x, y, w_v=2.0, w_z=5.0)

    # Evaluate for each seed
    all_results = []
    for seed in args.seeds:
        print(f"Evaluating seed={seed}...")
        result = evaluate_random_policy(
            dataset=dataset,
            env_cfg=env_cfg,
            checker_fn=checker_fn,
            num_episodes=args.num_episodes,
            seed=seed,
            verbose=args.verbose,
        )
        result["dataset"] = args.dataset_path
        all_results.append(result)
        
        print(f"  success_rate: {result['success_rate']:.3f} ({result['solved_count']}/{result['num_episodes']})")
        print(f"  mean_score: {result['mean_score']:.3f}")
        if "final_filled_mean" in result:
            print(f"  filled_mean: {result['final_filled_mean']:.2f}")
            print(f"  violations_mean: {result['final_violations_mean']:.2f}")
            print(f"  zero_cand_mean: {result['final_zero_cand_mean']:.2f}")
        print()

    # Compute aggregate statistics
    agg_success = sum(r["success_rate"] for r in all_results) / len(all_results)
    agg_score = sum(r["mean_score"] for r in all_results) / len(all_results)
    
    # Compute std across seeds
    success_rates = [r["success_rate"] for r in all_results]
    success_std = (sum((r - agg_success)**2 for r in success_rates) / len(success_rates)) ** 0.5
    
    print("=" * 60)
    print("AGGREGATE RESULTS (mean ± std across seeds)")
    print("=" * 60)
    print(f"Success rate: {agg_success*100:.1f}% ± {success_std*100:.1f}%")
    print(f"Mean score: {agg_score:.3f}")
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        with open(output_path, "w", newline="") as f:
            fieldnames = [
                "seed", "success_rate", "mean_score", "solved_count", "num_episodes",
                "max_edits", "dataset", "score_min", "score_max", "initial_score_mean",
                "final_filled_mean", "final_violations_mean", "final_zero_cand_mean"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
            # Add aggregate row
            writer.writerow({
                "seed": "mean",
                "success_rate": agg_success,
                "mean_score": agg_score,
                "solved_count": sum(r["solved_count"] for r in all_results),
                "num_episodes": sum(r["num_episodes"] for r in all_results),
                "max_edits": args.max_edits,
                "dataset": args.dataset_path,
            })
        print(f"Saved CSV: {output_path}")
        
        # Save JSON (for easier programmatic access)
        json_path = output_path.with_suffix(".json")
        json_output = {
            "per_seed": all_results,
            "aggregate": {
                "success_rate_mean": agg_success,
                "success_rate_std": success_std,
                "mean_score": agg_score,
                "total_solved": sum(r["solved_count"] for r in all_results),
                "total_episodes": sum(r["num_episodes"] for r in all_results),
            },
            "config": {
                "dataset": args.dataset_path,
                "seeds": args.seeds,
                "num_episodes": args.num_episodes,
                "max_edits": args.max_edits,
                "gamma": args.gamma,
                "vocab_size": args.vocab_size,
                "stop_action_mode": env_cfg.stop_action_mode,
                "checker": "feasibility (w_v=2.0, w_z=5.0)",
            }
        }
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved JSON: {json_path}")
    
    print()
    print("NOTE: 'Random (action mask)' in Table 3 uses uniform sampling over")
    print("      valid actions only (excluding clue cells and STOP when disabled).")
    print("      Success = sudoku_is_solved (all filled, no violations).")


if __name__ == "__main__":
    main()
