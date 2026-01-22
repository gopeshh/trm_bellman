#!/usr/bin/env python3
"""
Generate learning curve plots for Table 3 baselines.
Matches paper style from plot_feasibility_curves.py and plot_6to8empties_paper_style.py

Creates trivial_baselines_vs_no_contraction_success_vs_steps.pdf

Usage:
    python plot_table3_baselines.py --results-dir results/table3_baselines_rerun_evalfix_2026_01_22 \
                                     --output-dir figures/
"""

import argparse
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


# Algorithm config with correct naming
ALGO_CONFIG = {
    "persistent_nc": {
        "name": "UPI-TRM (Persistent-z, no contraction)",
        "color": "#e377c2",  # pink
        "marker": "P"
    },
    "episodic_nc": {
        "name": "UPI-TRM (Episodic-z, no contraction)",
        "color": "#8c564b",  # brown
        "marker": "<"
    },
    "episodic_c_clean": {
        "name": "UPI-TRM (Episodic-z, contraction)",
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

# Random baseline for trivial dataset (52%)
RANDOM_BASELINE = 0.52


def parse_log_file(log_path: Path) -> List[Tuple[int, float]]:
    """
    Parse a training log file and extract (step, success_rate) pairs.

    Returns:
        List of (step, success_rate) tuples
    """
    data = []
    pattern = r"\[step (\d+)\] eval_success_rate=(\d+\.\d+)"

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                success_rate = float(match.group(2))
                data.append((step, success_rate))

    return data


def aggregate_seeds(log_files: List[Path]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Load data for each seed separately.

    Returns:
        Dict mapping seed_idx -> list of (step, success_rate) tuples
    """
    all_data = {}
    for i, log_file in enumerate(log_files):
        data = parse_log_file(log_file)
        if data:
            all_data[i] = data
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
    parser = argparse.ArgumentParser(description="Generate Figure 2: Table 3 baselines learning curves")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing training logs (e.g., results/table3_baselines_rerun_evalfix_2026_01_22)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output figures (e.g., figures/)")
    parser.add_argument("--output-name", type=str,
                        default="trivial_baselines_vs_no_contraction_success_vs_steps",
                        help="Base name for output files (without extension)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum x-axis value (default: 5000)")
    args = parser.parse_args()

    apply_paper_style()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    # Define methods and their log file patterns
    methods = {
        "persistent_nc": list(results_dir.glob("persistent_nc_s*.log")),
        "episodic_nc": list(results_dir.glob("episodic_nc_s*.log")),
        "episodic_c_clean": list(results_dir.glob("episodic_c_clean_s*.log")),
        "ppo": list(results_dir.glob("ppo_s*.log")),
        "a2c": list(results_dir.glob("a2c_s*.log")),
        "dqn": list(results_dir.glob("dqn_s*.log")),
    }

    # Plotting order (UPI-TRM variants first, then baselines)
    algo_order = [
        "persistent_nc",
        "episodic_nc",
        "episodic_c_clean",
        "ppo",
        "a2c",
        "dqn",
    ]

    # Create figure - match paper style size
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

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
            rates = [d[1] for d in data]
            ax.plot(steps, rates, color=config["color"], alpha=0.18, linewidth=0.9)

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
                markevery=8,
                label=f"{config['name']} (S={num_seeds})"
            )

    # Add random baseline horizontal line
    ax.axhline(
        y=RANDOM_BASELINE,
        color='gray',
        linestyle='--',
        linewidth=2.5,
        alpha=0.7,
        label=f'Random ({RANDOM_BASELINE:.0%})'
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title("4×4 Sudoku (1–4 empties, T=16): Success Rate vs Training Steps\n(Feasibility Checker, 5k steps, 3 seeds)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, args.max_steps)
    ax.grid(True, alpha=0.3)

    # Legend below plot in two columns (matches paper style)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    ax.tick_params(axis="both", which="major")

    # Leave room for legend under axes
    fig.subplots_adjust(bottom=0.38)

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output_name}.pdf"

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {str(output_path).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
