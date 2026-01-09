#!/usr/bin/env python3
"""
Parse feasibility experiment logs and generate CSV plot data.

Supports timestamped log files in multiple formats:
- {seed}.log (e.g., 42.log)
- {seed}_*.log (e.g., 42_20260108_223720.log)
- seed_{seed}_*.log (e.g., seed_42_20260108_223720.log)

Usage:
    python scripts/parse_feasibility_logs.py --log-dir runs/feasibility --output-dir results/plot_data
    python scripts/parse_feasibility_logs.py --all-matching  # Include all matching logs with run_id
"""

import argparse
import os
import re
import glob
from pathlib import Path
from collections import defaultdict
import csv


def find_logs_for_seed(algo_path: Path, seed: int, all_matching: bool = False) -> list:
    """
    Find log files for a given seed.

    Returns list of (log_path, run_id) tuples.
    - If all_matching=False: returns only the newest log
    - If all_matching=True: returns all matching logs

    Supports patterns:
    - {seed}.log (e.g., 42.log)
    - {seed}_*.log (e.g., 42_20260108_223720.log)
    - seed_{seed}_*.log (e.g., seed_42_20260108_223720.log)
    """
    # Pattern: {seed}.log or {seed}_*.log or seed_{seed}_*.log
    pattern1 = algo_path / f"{seed}.log"
    pattern2 = str(algo_path / f"{seed}_*.log")
    pattern3 = str(algo_path / f"seed_{seed}_*.log")

    logs = []

    # Check old-style log first
    if pattern1.exists():
        logs.append((pattern1, f"{seed}"))

    # Find timestamped logs (both {seed}_* and seed_{seed}_* patterns)
    timestamped = sorted(glob.glob(pattern2), key=os.path.getmtime, reverse=True)
    for log_path in timestamped:
        run_id = Path(log_path).stem  # e.g., "42_20260108_223720"
        logs.append((Path(log_path), run_id))

    # Find seed_ prefixed logs
    seed_prefixed = sorted(glob.glob(pattern3), key=os.path.getmtime, reverse=True)
    for log_path in seed_prefixed:
        run_id = Path(log_path).stem  # e.g., "seed_42_20260108_223720"
        logs.append((Path(log_path), run_id))

    if not all_matching and len(logs) > 0:
        # Return only the newest (first in sorted list by mtime)
        return [logs[0]]

    return logs


def parse_log_file(log_path: Path) -> dict:
    """Parse a single log file and extract metrics."""
    results = {
        "steps": [],
        "success_rates": [],
        "mean_scores": [],
        "solved_counts": [],
        "total_episodes": [],
        "filled_means": [],
        "violations_means": [],
        "zero_cand_means": [],
    }

    final_success_rate = 0.0
    final_mean_score = 0.0
    peak_success_rate = 0.0
    peak_step = 0
    steps_to_100 = None

    try:
        with open(log_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {log_path}: {e}")
        return results

    # Find all eval lines
    # Pattern: [step XXXXX] eval_success_rate=X.XXX eval_mean_score=X.XXX ... [solved=X/Y, ...]
    pattern = r"\[step (\d+)\] eval_success_rate=([0-9.]+) eval_mean_score=([0-9.-]+).*\[solved=(\d+)/(\d+)"

    # Also parse PROGRESS lines for filled/violations/zero_cand
    progress_pattern = r"\[step (\d+)\] PROGRESS: filled=([0-9.]+) violations=([0-9.]+) zero_cand=([0-9.]+)"

    # Build a dict of progress data by step
    progress_by_step = {}
    for match in re.finditer(progress_pattern, content):
        step = int(match.group(1))
        progress_by_step[step] = {
            "filled": float(match.group(2)),
            "violations": float(match.group(3)),
            "zero_cand": float(match.group(4)),
        }

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

        # Add progress data if available
        progress = progress_by_step.get(step, {})
        results["filled_means"].append(progress.get("filled"))
        results["violations_means"].append(progress.get("violations"))
        results["zero_cand_means"].append(progress.get("zero_cand"))

        if success_rate > peak_success_rate:
            peak_success_rate = success_rate
            peak_step = step

        if success_rate >= 1.0 and steps_to_100 is None:
            steps_to_100 = step

        final_success_rate = success_rate
        final_mean_score = mean_score

    results["final_success_rate"] = final_success_rate
    results["final_mean_score"] = final_mean_score
    results["peak_success_rate"] = peak_success_rate
    results["peak_step"] = peak_step
    results["steps_to_100"] = steps_to_100

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse feasibility experiment logs")
    parser.add_argument("--log-dir", default="runs/feasibility", help="Directory containing logs")
    parser.add_argument("--output-dir", default="results/plot_data", help="Output directory for CSVs")
    parser.add_argument("--all-matching", action="store_true",
                        help="Parse ALL matching logs (adds run_id column)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Algorithm directories
    algo_dirs = ["upi_trm", "ppo", "a2c", "dqn", "ablation_no_conservative", "ablation_no_contraction", "ablation_persistent_z", "ablation_persistent_z_no_contraction"]
    seeds = [42, 123, 456]

    all_results = []
    learning_curves = []

    # Track what we found
    found_logs = 0

    for algo in algo_dirs:
        algo_path = log_dir / algo
        if not algo_path.exists():
            continue

        for seed in seeds:
            logs = find_logs_for_seed(algo_path, seed, args.all_matching)
            if not logs:
                continue

            for log_file, run_id in logs:
                found_logs += 1
                print(f"Parsing {algo} seed={seed} run={run_id}...")
                results = parse_log_file(log_file)

                if not results["steps"]:
                    print(f"  Warning: No eval data found in {log_file}")
                    continue

                # Store summary
                row = {
                    "algorithm": algo,
                    "seed": seed,
                    "success_rate_final": results["final_success_rate"],
                    "success_rate_peak": results["peak_success_rate"],
                    "peak_step": results["peak_step"],
                    "steps_to_100": results["steps_to_100"] if results["steps_to_100"] else "N/A",
                    "mean_score_final": results["final_mean_score"],
                }
                if args.all_matching:
                    row["run_id"] = run_id
                all_results.append(row)

                # Store learning curve data
                for i, step in enumerate(results["steps"]):
                    curve_row = {
                        "algorithm": algo,
                        "seed": seed,
                        "step": step,
                        "success_rate": results["success_rates"][i],
                        "mean_score": results["mean_scores"][i],
                        "solved_count": results["solved_counts"][i],
                        "total_episodes": results["total_episodes"][i],
                        "filled_mean": results["filled_means"][i] if i < len(results["filled_means"]) else None,
                        "violations_mean": results["violations_means"][i] if i < len(results["violations_means"]) else None,
                        "zero_cand_mean": results["zero_cand_means"][i] if i < len(results["zero_cand_means"]) else None,
                    }
                    if args.all_matching:
                        curve_row["run_id"] = run_id
                    learning_curves.append(curve_row)

    print(f"\nFound {found_logs} log files")

    if not all_results:
        print("No results found!")
        return

    # Define fieldnames
    comparison_fields = ["algorithm", "seed", "success_rate_final", "success_rate_peak",
                         "peak_step", "steps_to_100", "mean_score_final"]
    curve_fields = ["algorithm", "seed", "step", "success_rate", "mean_score",
                    "solved_count", "total_episodes",
                    "filled_mean", "violations_mean", "zero_cand_mean"]

    if args.all_matching:
        comparison_fields.append("run_id")
        curve_fields.append("run_id")

    # Write algorithm comparison CSV
    comparison_path = output_dir / "plot_data_feasibility_algorithm_comparison.csv"
    with open(comparison_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Wrote {comparison_path}")

    # Write learning curves CSV
    curves_path = output_dir / "plot_data_feasibility_learning_curves.csv"
    with open(curves_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(learning_curves)
    print(f"Wrote {curves_path}")

    # Compute and write summary statistics (grouped by algorithm, ignoring run_id)
    summary = defaultdict(lambda: {"success_rates": [], "peak_rates": [], "mean_scores": []})

    # For summary, use only latest run per algo/seed combination
    seen = set()
    for r in all_results:
        key = (r["algorithm"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
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
