#!/usr/bin/env python3
"""
Plot results from contraction vs SGD tradeoff experiments.

Creates:
1. A table of final success rates
2. A learning curve plot (success rate vs steps)
3. A summary CSV

Usage:
    python scripts/plot_contraction_sgd_tradeoff.py \
        --results-dir results/plot_data_contraction_sgd_tradeoff_light \
        --output-dir results/plots_contraction_sgd_tradeoff_light
"""

import argparse
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def parse_log_file(log_path: str) -> Dict[str, Any]:
    """Parse a training log file and extract metrics."""
    results: Dict[str, Any] = {
        "final_success_rate": None,
        "final_mean_score": None,
        "final_step": None,
        "learning_curve": [],  # List of (step, success_rate) tuples
    }

    if not os.path.exists(log_path):
        return results

    with open(log_path, "r") as f:
        for line in f:
            # Look for evaluation results: "[eval] step=X success_rate=Y"
            # or "eval_success_rate=Y"
            if "eval_success_rate=" in line or ("success_rate=" in line and "eval" in line.lower()):
                try:
                    # Extract step number
                    step = None
                    step_match = re.search(r'step[=\s]+(\d+)', line)
                    if step_match:
                        step = int(step_match.group(1))

                    # Extract success rate
                    sr_match = re.search(r'(?:eval_)?success_rate=([0-9.]+)', line)
                    if sr_match:
                        success_rate = float(sr_match.group(1))
                        if step is not None:
                            results["learning_curve"].append((step, success_rate))
                        results["final_success_rate"] = success_rate
                        results["final_step"] = step

                    # Extract mean score
                    ms_match = re.search(r'(?:eval_)?mean_score=([0-9.]+)', line)
                    if ms_match:
                        results["final_mean_score"] = float(ms_match.group(1))

                except (ValueError, IndexError):
                    continue

    return results


def load_all_results(results_dir: str, seed: int = 42) -> Dict[str, Dict]:
    """Load results for all conditions for a given seed."""
    conditions = [
        "1_no_contraction",
        "2_weak_contraction",
        "3_standard_contraction",
        "4_scheduled_contraction",
    ]

    all_results = {}
    for cond in conditions:
        log_file = os.path.join(results_dir, f"{cond}_seed{seed}.log")
        if os.path.exists(log_file):
            all_results[cond] = parse_log_file(log_file)
        else:
            print(f"Warning: Log file not found: {log_file}")

    return all_results


def print_summary_table(all_results: Dict[str, Dict], seed: int) -> str:
    """Print and return a summary table of results."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"CONTRACTION vs SGD TRADEOFF RESULTS (seed={seed})")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Condition':<30} {'Success Rate':<15} {'Mean Score':<15}")
    lines.append("-" * 60)

    for cond, res in sorted(all_results.items()):
        sr = f"{res['final_success_rate']:.3f}" if res['final_success_rate'] is not None else "N/A"
        ms = f"{res['final_mean_score']:.3f}" if res['final_mean_score'] is not None else "N/A"
        lines.append(f"{cond:<30} {sr:<15} {ms:<15}")

    lines.append("-" * 60)
    lines.append("")

    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def plot_learning_curves(all_results: Dict[str, Dict], output_path: str, seed: int):
    """Create a learning curve plot showing success rate vs training steps."""
    plt.figure(figsize=(10, 6))

    # Define colors and labels
    style_map = {
        "1_no_contraction": {"color": "#2ecc71", "label": "No Contraction", "linestyle": "-"},
        "2_weak_contraction": {"color": "#3498db", "label": "Weak (Lz=0.99)", "linestyle": "--"},
        "3_standard_contraction": {"color": "#e74c3c", "label": "Standard (Lz=0.90)", "linestyle": "-"},
        "4_scheduled_contraction": {"color": "#9b59b6", "label": "Scheduled (off→on)", "linestyle": "-."},
    }

    for cond, res in sorted(all_results.items()):
        if not res["learning_curve"]:
            continue

        steps = [x[0] for x in res["learning_curve"]]
        success_rates = [x[1] for x in res["learning_curve"]]

        style = style_map.get(cond, {"color": "gray", "label": cond, "linestyle": "-"})
        plt.plot(steps, success_rates,
                 color=style["color"],
                 linestyle=style["linestyle"],
                 label=style["label"],
                 linewidth=2,
                 marker='o',
                 markersize=4,
                 alpha=0.8)

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.title(f"Contraction vs SGD Tradeoff (seed={seed})", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curve plot saved to: {output_path}")


def save_csv(all_results: Dict[str, Dict], output_path: str, seed: int):
    """Save results to CSV format."""
    with open(output_path, "w") as f:
        f.write("condition,seed,final_success_rate,final_mean_score,final_step\n")
        for cond, res in sorted(all_results.items()):
            sr = res['final_success_rate'] if res['final_success_rate'] is not None else ""
            ms = res['final_mean_score'] if res['final_mean_score'] is not None else ""
            step = res['final_step'] if res['final_step'] is not None else ""
            f.write(f"{cond},{seed},{sr},{ms},{step}\n")
    print(f"CSV saved to: {output_path}")


def generate_conclusion(all_results: Dict[str, Dict]) -> str:
    """Generate a finite-run comparison without causal attribution."""
    lines = []
    lines.append("FINITE-RUN COMPARISON")
    lines.append("=" * 50)

    # Get success rates
    no_contraction_sr = all_results.get("1_no_contraction", {}).get("final_success_rate")
    weak_sr = all_results.get("2_weak_contraction", {}).get("final_success_rate")
    standard_sr = all_results.get("3_standard_contraction", {}).get("final_success_rate")
    scheduled_sr = all_results.get("4_scheduled_contraction", {}).get("final_success_rate")

    if no_contraction_sr is not None and standard_sr is not None:
        diff = no_contraction_sr - standard_sr
        if diff > 0.05:
            lines.append(f"No-contraction final success is {diff:.1%} higher than standard-contraction success.")
        elif diff < -0.05:
            lines.append(f"Standard-contraction final success is {-diff:.1%} higher than no-contraction success.")
        else:
            lines.append(f"No-contraction and standard-contraction final success differ by {abs(diff):.1%}.")

    if weak_sr is not None and standard_sr is not None:
        diff = weak_sr - standard_sr
        if diff > 0.02:
            lines.append(f"Weak-contraction (0.99) final success is {diff:.1%} higher than standard (0.90).")
        elif diff < -0.02:
            lines.append(f"Standard-contraction (0.90) final success is {-diff:.1%} higher than weak (0.99).")

    if scheduled_sr is not None and standard_sr is not None:
        diff = scheduled_sr - standard_sr
        if diff > 0.02:
            lines.append(f"Scheduled-contraction final success is {diff:.1%} higher than always-on standard contraction.")

    lines.append("These single-seed log comparisons do not establish causality or a necessary SGD tradeoff.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plot contraction vs SGD tradeoff results")
    parser.add_argument("--results-dir", type=str,
                        default="results/plot_data_contraction_sgd_tradeoff_light",
                        help="Directory containing log files")
    parser.add_argument("--output-dir", type=str,
                        default="results/plots_contraction_sgd_tradeoff_light",
                        help="Directory for output plots")
    parser.add_argument("--seed", type=int, default=42, help="Seed to analyze")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    all_results = load_all_results(args.results_dir, args.seed)

    if not all_results:
        print(f"No results found in {args.results_dir}")
        return

    # Print summary table
    table_str = print_summary_table(all_results, args.seed)

    # Save table to file
    table_path = os.path.join(args.output_dir, f"summary_seed{args.seed}.txt")
    with open(table_path, "w") as f:
        f.write(table_str)

    # Plot learning curves
    plot_path = os.path.join(args.output_dir, f"learning_curves_seed{args.seed}.png")
    plot_learning_curves(all_results, plot_path, args.seed)

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"results_seed{args.seed}.csv")
    save_csv(all_results, csv_path, args.seed)

    # Generate and print conclusion
    conclusion = generate_conclusion(all_results)
    print(conclusion)

    conclusion_path = os.path.join(args.output_dir, f"conclusion_seed{args.seed}.txt")
    with open(conclusion_path, "w") as f:
        f.write(conclusion)


if __name__ == "__main__":
    main()
