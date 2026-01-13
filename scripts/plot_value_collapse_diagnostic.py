#!/usr/bin/env python3
"""
Diagnostic plot showing value function collapse in the failed 6-8 empties experiments.

This script parses log files and generates a figure showing:
1. Value function V(s) over training steps (should NOT saturate at min)
2. Target values over training steps
3. Success rate over time
4. Reward distribution over time

Usage:
    python scripts/plot_value_collapse_diagnostic.py \
        --log-dir results/plot_data_contraction_sgd_sudoku-4x4-easy_6to8empties \
        --output results/plots_contraction_sgd_sudoku-4x4-easy_6to8empties/value_collapse_diagnostic.png
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import matplotlib - if not available, we'll generate CSV data instead
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will output CSV data instead")


def parse_value_debug(log_file: str) -> Dict[str, List[Tuple[int, float]]]:
    """Parse VALUE_DEBUG lines to extract value function and target progression."""
    results = {
        "V_mean": [],      # (step, V(s) mean)
        "target_mean": [], # (step, target mean)
        "reward_mean": [], # (step, reward mean)
    }

    if not os.path.exists(log_file):
        return results

    # Pattern: [step XXXXX] VALUE_DEBUG: target(mean=X std=X min=X max=X) V(s)(mean=X std=X) reward(mean=X std=X)
    pattern = re.compile(
        r'\[step (\d+)\] VALUE_DEBUG: '
        r'target\(mean=([-\d.]+) std=([-\d.]+) min=([-\d.]+) max=([-\d.]+)\) '
        r'V\(s\)\(mean=([-\d.]+) std=([-\d.]+)\) '
        r'reward\(mean=([-\d.]+) std=([-\d.]+)\)'
    )

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                target_mean = float(match.group(2))
                v_mean = float(match.group(6))
                reward_mean = float(match.group(8))

                results["V_mean"].append((step, v_mean))
                results["target_mean"].append((step, target_mean))
                results["reward_mean"].append((step, reward_mean))

    return results


def parse_eval_results(log_file: str) -> List[Tuple[int, float, float]]:
    """Parse eval lines to extract (step, success_rate, mean_score)."""
    results = []

    if not os.path.exists(log_file):
        return results

    # Pattern: [step XXXXX] eval_success_rate=X.XXX eval_mean_score=X.XXX
    pattern = re.compile(
        r'\[step (\d+)\] eval_success_rate=([\d.]+) eval_mean_score=([\d.]+)'
    )

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                success_rate = float(match.group(2))
                mean_score = float(match.group(3))
                results.append((step, success_rate, mean_score))

    return results


def create_diagnostic_plot(log_dir: str, output_path: str):
    """Create a 2x2 diagnostic plot showing value function collapse."""

    # Find all log files
    log_files = list(Path(log_dir).glob("*.log"))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    # Parse data from all logs
    all_data = {}
    for log_file in log_files:
        condition = log_file.stem  # e.g., "1_no_contraction_seed42"
        all_data[condition] = {
            "value_debug": parse_value_debug(str(log_file)),
            "eval": parse_eval_results(str(log_file)),
        }

    if not HAS_MATPLOTLIB:
        # Output CSV data instead
        csv_path = output_path.replace('.png', '.csv').replace('.pdf', '.csv')
        with open(csv_path, 'w') as f:
            f.write("condition,step,V_mean,target_mean,reward_mean,success_rate,mean_score\n")
            for condition, data in all_data.items():
                vd = data["value_debug"]
                ev = data["eval"]

                # Create a map of step -> eval data
                eval_map = {step: (sr, ms) for step, sr, ms in ev}

                for i, (step, v_mean) in enumerate(vd["V_mean"]):
                    target_mean = vd["target_mean"][i][1] if i < len(vd["target_mean"]) else ""
                    reward_mean = vd["reward_mean"][i][1] if i < len(vd["reward_mean"]) else ""
                    sr, ms = eval_map.get(step, ("", ""))
                    f.write(f"{condition},{step},{v_mean},{target_mean},{reward_mean},{sr},{ms}\n")

        print(f"CSV data saved to {csv_path}")
        return

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Value Function Collapse Diagnostic\n(6-8 empties dataset - FAILED experiments)",
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10.colors

    # Plot 1: V(s) mean over time
    ax1 = axes[0, 0]
    for i, (condition, data) in enumerate(all_data.items()):
        vd = data["value_debug"]
        if vd["V_mean"]:
            steps, values = zip(*vd["V_mean"])
            label = condition.split("_seed")[0]
            ax1.plot(steps, values, label=label, color=colors[i % len(colors)], alpha=0.8)
    ax1.axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='Saturation limit (-20)')
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("V(s) Mean")
    ax1.set_title("Value Function Collapse\n(Saturates at -20, no learning signal)")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Target mean over time
    ax2 = axes[0, 1]
    for i, (condition, data) in enumerate(all_data.items()):
        vd = data["value_debug"]
        if vd["target_mean"]:
            steps, values = zip(*vd["target_mean"])
            label = condition.split("_seed")[0]
            ax2.plot(steps, values, label=label, color=colors[i % len(colors)], alpha=0.8)
    ax2.axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='Min value (-20)')
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Target Mean")
    ax2.set_title("TD Targets\n(All negative, value has nowhere to go but down)")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reward mean over time
    ax3 = axes[1, 0]
    for i, (condition, data) in enumerate(all_data.items()):
        vd = data["value_debug"]
        if vd["reward_mean"]:
            steps, values = zip(*vd["reward_mean"])
            label = condition.split("_seed")[0]
            ax3.plot(steps, values, label=label, color=colors[i % len(colors)], alpha=0.8)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Zero reward')
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Reward Mean")
    ax3.set_title("Per-Step Rewards\n(Always negative → discouraging exploration)")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Success rate over time
    ax4 = axes[1, 1]
    for i, (condition, data) in enumerate(all_data.items()):
        ev = data["eval"]
        if ev:
            steps, success_rates, _ = zip(*ev)
            label = condition.split("_seed")[0]
            ax4.plot(steps, success_rates, label=label, color=colors[i % len(colors)],
                    alpha=0.8, marker='o', markersize=2)
    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("Success Rate")
    ax4.set_title("Evaluation Success Rate\n(Stuck at 0% - no learning)")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plot saved to {output_path}")

    # Also save as PDF for paper quality
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate value collapse diagnostic plot")
    parser.add_argument("--log-dir", type=str,
                        default="results/plot_data_contraction_sgd_sudoku-4x4-easy_6to8empties",
                        help="Directory containing log files")
    parser.add_argument("--output", type=str,
                        default="results/plots_contraction_sgd_sudoku-4x4-easy_6to8empties/value_collapse_diagnostic.png",
                        help="Output path for the plot")
    args = parser.parse_args()

    create_diagnostic_plot(args.log_dir, args.output)


if __name__ == "__main__":
    main()
