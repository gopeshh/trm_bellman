#!/usr/bin/env python3
"""
Parse feasibility experiment logs and generate CSV plot data.

Usage:
    python scripts/parse_feasibility_logs.py --log-dir runs/feasibility --output-dir results/plot_data
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict
import csv


def parse_log_file(log_path: Path) -> dict:
    """Parse a single log file and extract metrics."""
    results = {
        "steps": [],
        "success_rates": [],
        "mean_scores": [],
        "solved_counts": [],
        "total_episodes": [],
    }

    final_success_rate = 0.0
    final_mean_score = 0.0
    peak_success_rate = 0.0
    steps_to_100 = None

    with open(log_path, "r") as f:
        content = f.read()

    # Find all eval lines
    # Pattern: [step XXXXX] eval_success_rate=X.XXX eval_mean_score=X.XXX ... [solved=X/Y, ...]
    pattern = r"\[step (\d+)\] eval_success_rate=([0-9.]+) eval_mean_score=([0-9.-]+).*\[solved=(\d+)/(\d+)"

    for match in re.finditer(pattern, content):
        step = int(match.group(1))
        success_rate = float(match.group(2))
        mean_score = float(match.group(3))
        solved = int(match.group(4))
        total = int(match.group(5))

        results["steps"].append(step)
        results["success_rates"].append(success_rate)
        results["mean_scores"].append(mean_score)
        results["solved_counts"].append(solved)
        results["total_episodes"].append(total)

        if success_rate > peak_success_rate:
            peak_success_rate = success_rate

        if success_rate >= 1.0 and steps_to_100 is None:
            steps_to_100 = step

        final_success_rate = success_rate
        final_mean_score = mean_score

    results["final_success_rate"] = final_success_rate
    results["final_mean_score"] = final_mean_score
    results["peak_success_rate"] = peak_success_rate
    results["steps_to_100"] = steps_to_100

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse feasibility experiment logs")
    parser.add_argument("--log-dir", default="runs/feasibility", help="Directory containing logs")
    parser.add_argument("--output-dir", default="results/plot_data", help="Output directory for CSVs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Algorithm directories
    algo_dirs = ["upi_trm", "ppo", "a2c", "dqn", "ablation_no_conservative", "ablation_no_contraction"]
    seeds = [42, 123, 456]

    all_results = []
    learning_curves = []

    for algo in algo_dirs:
        algo_path = log_dir / algo
        if not algo_path.exists():
            continue

        for seed in seeds:
            log_file = algo_path / f"{seed}.log"
            if not log_file.exists():
                continue

            print(f"Parsing {algo} seed={seed}...")
            results = parse_log_file(log_file)

            # Store summary
            all_results.append({
                "algorithm": algo,
                "seed": seed,
                "success_rate_final": results["final_success_rate"],
                "success_rate_peak": results["peak_success_rate"],
                "steps_to_100": results["steps_to_100"] if results["steps_to_100"] else "N/A",
                "mean_score_final": results["final_mean_score"],
            })

            # Store learning curve data
            for i, step in enumerate(results["steps"]):
                learning_curves.append({
                    "algorithm": algo,
                    "seed": seed,
                    "step": step,
                    "success_rate": results["success_rates"][i],
                    "mean_score": results["mean_scores"][i],
                    "solved_count": results["solved_counts"][i],
                    "total_episodes": results["total_episodes"][i],
                })

    # Write algorithm comparison CSV
    comparison_path = output_dir / "plot_data_feasibility_algorithm_comparison.csv"
    with open(comparison_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "seed", "success_rate_final", "success_rate_peak",
            "steps_to_100", "mean_score_final"
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Wrote {comparison_path}")

    # Write learning curves CSV
    curves_path = output_dir / "plot_data_feasibility_learning_curves.csv"
    with open(curves_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "seed", "step", "success_rate", "mean_score",
            "solved_count", "total_episodes"
        ])
        writer.writeheader()
        writer.writerows(learning_curves)
    print(f"Wrote {curves_path}")

    # Compute and write summary statistics
    summary = defaultdict(lambda: {"success_rates": [], "peak_rates": [], "mean_scores": []})
    for r in all_results:
        algo = r["algorithm"]
        summary[algo]["success_rates"].append(r["success_rate_final"])
        summary[algo]["peak_rates"].append(r["success_rate_peak"])
        summary[algo]["mean_scores"].append(r["mean_score_final"])

    summary_rows = []
    for algo, data in summary.items():
        n = len(data["success_rates"])
        if n == 0:
            continue

        def mean_std(vals):
            m = sum(vals) / len(vals)
            if len(vals) > 1:
                s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                s = 0.0
            return m, s

        sr_mean, sr_std = mean_std(data["success_rates"])
        pk_mean, pk_std = mean_std(data["peak_rates"])
        ms_mean, ms_std = mean_std(data["mean_scores"])

        summary_rows.append({
            "algorithm": algo,
            "n_seeds": n,
            "success_rate_final_mean": f"{sr_mean:.3f}",
            "success_rate_final_std": f"{sr_std:.3f}",
            "success_rate_peak_mean": f"{pk_mean:.3f}",
            "success_rate_peak_std": f"{pk_std:.3f}",
            "mean_score_final_mean": f"{ms_mean:.3f}",
            "mean_score_final_std": f"{ms_std:.3f}",
        })

    summary_path = output_dir / "plot_data_feasibility_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "n_seeds", "success_rate_final_mean", "success_rate_final_std",
            "success_rate_peak_mean", "success_rate_peak_std",
            "mean_score_final_mean", "mean_score_final_std"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    print("\n=== Summary ===")
    for row in summary_rows:
        print(f"{row['algorithm']}: final={row['success_rate_final_mean']}±{row['success_rate_final_std']}, "
              f"peak={row['success_rate_peak_mean']}±{row['success_rate_peak_std']}, "
              f"score={row['mean_score_final_mean']}±{row['mean_score_final_std']}")


if __name__ == "__main__":
    main()
