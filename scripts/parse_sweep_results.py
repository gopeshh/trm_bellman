#!/usr/bin/env python3
"""Parse and analyze sweep results from RL ablation studies.

Usage:
    python scripts/parse_sweep_results.py runs/sweep_<timestamp>
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_csv_results(csv_path: str) -> List[Dict[str, Any]]:
    """Parse sweep results from CSV file.

    Args:
        csv_path: Path to sweep_results.csv

    Returns:
        List of result dictionaries
    """
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            result = {
                "run_id": int(row["run_id"]),
                "K": int(row["K"]),
                "n_inner": int(row["n_inner"]),
                "spectral_target_prod": float(row["spectral_target_prod"]),
                "ppo_clip": float(row["ppo_clip"]),
                "status": row["status"],
                "final_loss": float(row["final_loss"]) if row["final_loss"] != "N/A" else None,
                "best_accuracy": (
                    float(row["best_accuracy"]) if row["best_accuracy"] != "N/A" else None
                ),
                "steps_to_solution": (
                    int(row["steps_to_solution"]) if row["steps_to_solution"] != "N/A" else None
                ),
                "runtime_sec": int(row["runtime_sec"]),
            }
            results.append(result)

    return results


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics across all runs.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary of statistics
    """
    successful_runs = [r for r in results if r["status"] == "success"]
    failed_runs = [r for r in results if r["status"] == "failed"]

    stats = {
        "total_runs": len(results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "success_rate": len(successful_runs) / len(results) if results else 0.0,
    }

    if successful_runs:
        # Loss statistics
        losses = [r["final_loss"] for r in successful_runs if r["final_loss"] is not None]
        if losses:
            stats["loss"] = {
                "min": min(losses),
                "max": max(losses),
                "mean": sum(losses) / len(losses),
            }

        # Accuracy statistics
        accuracies = [r["best_accuracy"] for r in successful_runs if r["best_accuracy"] is not None]
        if accuracies:
            stats["accuracy"] = {
                "min": min(accuracies),
                "max": max(accuracies),
                "mean": sum(accuracies) / len(accuracies),
            }

        # Steps statistics
        steps = [
            r["steps_to_solution"] for r in successful_runs if r["steps_to_solution"] is not None
        ]
        if steps:
            stats["steps"] = {
                "min": min(steps),
                "max": max(steps),
                "mean": sum(steps) / len(steps),
            }

        # Runtime statistics
        runtimes = [r["runtime_sec"] for r in successful_runs]
        stats["runtime_sec"] = {
            "min": min(runtimes),
            "max": max(runtimes),
            "mean": sum(runtimes) / len(runtimes),
            "total": sum(runtimes),
        }

    return stats


def find_best_runs(
    results: List[Dict[str, Any]], metric: str = "best_accuracy", top_k: int = 5
) -> List[Dict[str, Any]]:
    """Find the best runs according to a metric.

    Args:
        results: List of result dictionaries
        metric: Metric to optimize ("best_accuracy", "final_loss", etc.)
        top_k: Number of top runs to return

    Returns:
        List of top-k result dictionaries
    """
    successful_runs = [r for r in results if r["status"] == "success" and r[metric] is not None]

    # Sort by metric (ascending for loss, descending for accuracy)
    if "loss" in metric:
        successful_runs.sort(key=lambda x: x[metric])
    else:
        successful_runs.sort(key=lambda x: x[metric], reverse=True)

    return successful_runs[:top_k]


def analyze_hyperparameter_effects(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the effect of each hyperparameter on performance.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with per-hyperparameter statistics
    """
    successful_runs = [r for r in results if r["status"] == "success"]

    effects = {}

    for param in ["K", "n_inner", "spectral_target_prod", "ppo_clip"]:
        param_values = defaultdict(list)

        for run in successful_runs:
            param_val = run[param]
            if run["final_loss"] is not None:
                param_values[param_val].append(run["final_loss"])

        # Compute mean loss for each parameter value
        param_stats = {}
        for val, losses in param_values.items():
            param_stats[str(val)] = {
                "mean_loss": sum(losses) / len(losses) if losses else None,
                "num_runs": len(losses),
            }

        effects[param] = param_stats

    return effects


def main():
    """Main function to parse and analyze sweep results."""
    parser = argparse.ArgumentParser(description="Parse RL ablation sweep results")
    parser.add_argument("sweep_dir", type=str, help="Path to sweep directory")
    parser.add_argument(
        "--metric",
        type=str,
        default="best_accuracy",
        choices=["best_accuracy", "final_loss", "steps_to_solution"],
        help="Metric to rank runs by",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top runs to display")

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    csv_path = sweep_dir / "sweep_results.csv"

    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("RL Ablation Sweep Analysis")
    print("=" * 80)
    print(f"Sweep directory: {sweep_dir}")
    print(f"Results file: {csv_path}")
    print()

    # Parse results
    print("Parsing results...")
    results = parse_csv_results(str(csv_path))
    print(f"✓ Loaded {len(results)} runs")
    print()

    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(results)

    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total runs: {stats['total_runs']}")
    print(f"Successful: {stats['successful_runs']}")
    print(f"Failed: {stats['failed_runs']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print()

    if "loss" in stats:
        print(
            f"Final Loss:   min={stats['loss']['min']:.4f}, "
            f"max={stats['loss']['max']:.4f}, mean={stats['loss']['mean']:.4f}"
        )
    if "accuracy" in stats:
        print(
            f"Accuracy:     min={stats['accuracy']['min']:.4f}, "
            f"max={stats['accuracy']['max']:.4f}, mean={stats['accuracy']['mean']:.4f}"
        )
    if "steps" in stats:
        print(
            f"Steps:        min={stats['steps']['min']}, "
            f"max={stats['steps']['max']}, mean={stats['steps']['mean']:.0f}"
        )
    if "runtime_sec" in stats:
        print(
            f"Runtime (s):  min={stats['runtime_sec']['min']}, "
            f"max={stats['runtime_sec']['max']}, mean={stats['runtime_sec']['mean']:.0f}, "
            f"total={stats['runtime_sec']['total']}"
        )
    print()

    # Find best runs
    print("=" * 80)
    print(f"Top {args.top_k} Runs (by {args.metric})")
    print("=" * 80)
    best_runs = find_best_runs(results, metric=args.metric, top_k=args.top_k)

    for i, run in enumerate(best_runs, 1):
        print(
            f"{i}. Run {run['run_id']}: K={run['K']}, n_inner={run['n_inner']}, "
            f"spectral={run['spectral_target_prod']}, ppo_clip={run['ppo_clip']}"
        )
        print(f"   Loss: {run['final_loss']:.4f}" if run["final_loss"] else "   Loss: N/A", end="")
        print(
            f", Accuracy: {run['best_accuracy']:.4f}"
            if run["best_accuracy"]
            else ", Accuracy: N/A",
            end="",
        )
        print(
            f", Steps: {run['steps_to_solution']}" if run["steps_to_solution"] else ", Steps: N/A"
        )
    print()

    # Analyze hyperparameter effects
    print("=" * 80)
    print("Hyperparameter Effects (Mean Loss)")
    print("=" * 80)
    effects = analyze_hyperparameter_effects(results)

    for param, param_stats in effects.items():
        print(f"\n{param}:")
        for val, stats_dict in sorted(param_stats.items()):
            mean_loss = stats_dict["mean_loss"]
            num_runs = stats_dict["num_runs"]
            if mean_loss is not None:
                print(f"  {val}: {mean_loss:.4f} (n={num_runs})")
    print()

    # Save JSON summary
    summary = {
        "sweep_dir": str(sweep_dir),
        "statistics": stats,
        "best_runs": best_runs[: args.top_k],
        "hyperparameter_effects": effects,
    }

    json_path = sweep_dir / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"✓ Summary saved to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
