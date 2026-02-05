#!/usr/bin/env python3
"""
Generate mean score learning curve plots for 9×9 Sudoku experiments.
Matches paper style from plot_table3_hard.py.

Creates fig_9x9_mean_score_vs_steps.pdf

Usage:
    python scripts/plot_9x9_mean_score_vs_steps.py
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def apply_paper_style():
    """Apply paper-quality styling: larger fonts, readable at single-column width."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'pdf.fonttype': 42,  # TrueType fonts for better PDF rendering
        'ps.fonttype': 42,
    })


# Algorithm config for 9×9 experiments
ALGO_CONFIG = {
    "upi_trm": {
        "name": "UPI-TRM",
        "color": "#1f77b4",  # blue
        "marker": "o"
    },
    "ppo": {
        "name": "PPO",
        "color": "#ff7f0e",  # orange
        "marker": "s"
    },
    "a2c": {
        "name": "A2C",
        "color": "#2ca02c",  # green
        "marker": "^"
    },
    "dqn": {
        "name": "DQN",
        "color": "#d62728",  # red
        "marker": "D"
    },
}


def parse_log_file(log_path: Path) -> List[Tuple[int, float]]:
    """
    Parse a training log file and extract (step, mean_score) pairs.

    Returns:
        List of (step, mean_score) tuples
    """
    data = []
    pattern = r"\[step (\d+)\].*eval_mean_score=([0-9.]+)"

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                mean_score = float(match.group(2))
                data.append((step, mean_score))

    return data


def aggregate_seeds(log_files: List[Path], min_steps: int = 20000) -> Dict[int, List[Tuple[int, float]]]:
    """
    Load data for each seed separately, filtering out incomplete runs.

    Args:
        log_files: List of log file paths
        min_steps: Minimum steps required to consider a run complete (default 40k for 50k runs)

    Returns:
        Dict mapping seed_idx -> list of (step, mean_score) tuples
    """
    all_data = {}
    for i, log_file in enumerate(log_files):
        data = parse_log_file(log_file)
        if data:
            max_step = max(d[0] for d in data)
            if max_step >= min_steps:
                all_data[i] = data
            else:
                print(f"  Skipping {log_file.name}: incomplete (max step {max_step} < {min_steps})")
    return all_data


def compute_mean_std(seed_data: Dict[int, List[Tuple[int, float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and std across seeds.

    Returns:
        (steps, mean, std) arrays
    """
    if not seed_data:
        return np.array([]), np.array([]), np.array([])

    # Convert to dicts for easier access
    data_dicts = {seed: dict(data) for seed, data in seed_data.items()}

    # Get common steps
    all_steps = set(data_dicts[list(data_dicts.keys())[0]].keys())
    for d in data_dicts.values():
        all_steps &= set(d.keys())

    steps = np.array(sorted(all_steps))

    # Compute mean and std
    values = np.array([[d[s] for s in steps] for d in data_dicts.values()])
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)

    return steps, mean, std


def main():
    apply_paper_style()

    # Input and output directories
    # Use absolute path to handle buck2 sandbox
    repo_root = Path("/home/buiksat/trm_bellman")
    results_dir = repo_root / "results" / "9x9_experiments_seed0"
    output_dir = Path.home() / "UPI_TRM" / "UPI_TRM_ICML" / "figures"

    # Define methods and their log file patterns
    # Use specific patterns to avoid matching intermediate/restart logs
    ppo_logs = (
        list(results_dir.glob("ppo_50k_s[0-9].log")) +
        list(results_dir.glob("ppo_25k_s[0-9].log"))
    )
    methods = {
        "upi_trm": list(results_dir.glob("upi_trm_50k_s[0-9].log")),
        "ppo": ppo_logs,
        "a2c": list(results_dir.glob("a2c_50k_s[0-9].log")),
        "dqn": list(results_dir.glob("dqn_50k_s[0-9].log")),
    }

    # Plotting order (UPI-TRM first, then baselines)
    algo_order = ["upi_trm", "ppo", "a2c", "dqn"]

    # Create figure - match paper style size
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # Collect all plotted scores for auto-scaling Y-axis
    all_scores = []

    for algo_key in algo_order:
        log_files = methods.get(algo_key, [])
        if not log_files:
            print(f"Warning: No log files found for {algo_key}")
            continue

        config = ALGO_CONFIG[algo_key]
        print(f"Processing {config['name']}: {len(log_files)} seeds")

        # Load data for each seed
        seed_data = aggregate_seeds(log_files)

        if not seed_data:
            print(f"  No data found")
            continue

        # Plot individual seed curves (thin, semi-transparent)
        for seed_idx, data in seed_data.items():
            steps = [d[0] for d in data]
            scores = [d[1] for d in data]
            all_scores.extend(scores)  # Collect for Y-axis scaling
            ax.plot(steps, scores, color=config["color"], alpha=0.18, linewidth=0.9)

        # Compute and plot mean curve (thick)
        steps, mean, std = compute_mean_std(seed_data)

        if len(steps) > 0:
            num_seeds = len(seed_data)
            ax.plot(
                steps,
                mean,
                color=config["color"],
                linewidth=3.0,
                marker=config["marker"],
                markersize=6,
                markevery=10,  # Marker frequency for 50k steps
                label=f"{config['name']} (S={num_seeds})"
            )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Score")
    ax.set_title(
        "9×9 Sudoku (50k steps)\n"
        "Mean Score vs Training Steps"
    )

    # Auto-scale Y-axis based on plotted data with padding
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)

        # Add padding (use larger padding for score range 0-81)
        padding = (max_score - min_score) * 0.05
        padding = max(padding, 2.0)  # At least 2 points padding
        ymin = max(0.0, min_score - padding)
        ymax = min(81.0, max_score + padding)

        # Enforce minimum range if data is too narrow
        if ymax - ymin < 10:
            mean_score = (min_score + max_score) / 2
            ymin = max(0.0, mean_score - 5)
            ymax = min(81.0, mean_score + 5)

        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 81)

    ax.set_xlim(0, 50000)
    ax.grid(True, alpha=0.3)

    # Legend below plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    ax.tick_params(axis="both", which="major")

    # Leave room for legend under axes
    fig.subplots_adjust(bottom=0.25)

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_9x9_mean_score_vs_steps.pdf"

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {str(output_path).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
