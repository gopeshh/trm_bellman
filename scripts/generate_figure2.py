#!/usr/bin/env python3
"""Generate Figure 2: Learning curves for Table 3 baselines."""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Methods and seeds
METHODS = ["ppo", "a2c", "dqn", "upitrm"]
SEEDS = [42, 123, 456]

# Colors for methods (colorblind-friendly)
COLORS = {
    "ppo": "#E69F00",    # Orange
    "a2c": "#56B4E9",    # Sky blue
    "dqn": "#CC79A7",    # Reddish purple
    "upitrm": "#009E73", # Bluish green
    "random": "#999999", # Gray
}

LABELS = {
    "ppo": "PPO",
    "a2c": "A2C",
    "dqn": "DQN",
    "upitrm": "UPI-TRM (Ours)",
    "random": "Random Policy",
}

# Log file patterns - DQN uses _fixed suffix for rerun
LOG_PATTERNS = {
    "ppo": "ppo_s{seed}.log",
    "a2c": "a2c_s{seed}.log",
    "dqn": "dqn_s{seed}_fixed.log",
    "upitrm": "upitrm_s{seed}.log",
}


def parse_log(log_path: Path) -> tuple:
    """Parse eval_success_rate from log file.

    Returns:
        steps: List of training steps
        success_rates: List of success rates (0-100)
    """
    steps = []
    success_rates = []

    # Pattern: [step 00100] eval_success_rate=0.20
    pattern = re.compile(r"\[step\s+(\d+)\].*eval_success_rate=([0-9.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                rate = float(match.group(2)) * 100  # Convert to percentage
                steps.append(step)
                success_rates.append(rate)

    return steps, success_rates


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2 learning curves")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/home/buiksat/trm_bellman/results/table3_baselines_rerun_2026_01_21",
        help="Directory containing log files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output PDF path (defaults to log-dir/figure2_learning_curves.pdf)",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        default=True,
        help="Include random policy baseline",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    # Collect data
    data = {}

    for method in METHODS:
        data[method] = {"steps": None, "rates": []}

        for seed in SEEDS:
            log_name = LOG_PATTERNS[method].format(seed=seed)
            log_path = log_dir / log_name

            if not log_path.exists():
                print(f"Warning: {log_name} not found")
                continue

            steps, rates = parse_log(log_path)

            if not steps:
                print(f"Warning: No eval data in {log_name}")
                continue

            print(f"{log_name}: {len(steps)} eval points, final={rates[-1]:.1f}%")

            if data[method]["steps"] is None:
                data[method]["steps"] = steps
            data[method]["rates"].append(rates)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot random policy baseline (horizontal line at 52%)
    # With action masking on trivial 4x4 Sudoku (1-4 empties), random achieves ~52%
    if args.include_random:
        ax.axhline(
            y=52,
            color=COLORS["random"],
            linestyle="--",
            linewidth=1.5,
            label=LABELS["random"],
            alpha=0.7,
        )

    for method in METHODS:
        if not data[method]["rates"]:
            continue

        steps = np.array(data[method]["steps"])
        rates = np.array(data[method]["rates"])

        # Compute mean and std
        mean_rate = rates.mean(axis=0)
        std_rate = rates.std(axis=0)

        # Plot mean line
        ax.plot(
            steps,
            mean_rate,
            color=COLORS[method],
            linewidth=2,
            label=LABELS[method],
        )

        # Plot shaded std region
        ax.fill_between(
            steps,
            mean_rate - std_rate,
            mean_rate + std_rate,
            color=COLORS[method],
            alpha=0.2,
        )

    # Formatting
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Learning Curves on TRIVIAL 4×4 Sudoku", fontsize=14)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = log_dir / "figure2_learning_curves.pdf"

    # Save PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    # Also save PNG for quick preview (in same directory as PDF)
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
