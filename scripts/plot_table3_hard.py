#!/usr/bin/env python3
"""
Generate the hard-4x4 no-mask capability anchor figure.

Creates hard_4x4_baselines_success_vs_steps.pdf for the NeurIPS paper repo.
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


ALGO_CONFIG = {
    "m1_persistent_nc_nomask": {
        "name": "UPI-TRM (No Mask)",
        "color": "#e377c2",  # pink
        "marker": "P"
    },
    "m1_a2c_nomask": {
        "name": "A2C (No Mask)",
        "color": "#2ca02c",  # green
        "marker": "^"
    },
}

# Random baseline for hard dataset 6-8 empties (0% - computed via eval_random_baseline.py)
RANDOM_BASELINE = 0.0


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
    apply_paper_style()

    # Use the locked no-mask hard-4x4 capability runs.
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results" / "table3_hard_6to8"
    output_dir = Path.home() / "UPI_TRM" / "UPI_TRM_NIPS" / "figures"

    # Define methods and their log file patterns
    methods = {
        "m1_persistent_nc_nomask": list(results_dir.glob("m1_persistent_nc_nomask_s*.log")),
        "m1_a2c_nomask": list(results_dir.glob("m1_a2c_nomask_s*.log")),
    }

    # Plot UPI-TRM first, then the in-house A2C baseline.
    algo_order = [
        "m1_persistent_nc_nomask",
        "m1_a2c_nomask",
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
                markevery=20,  # Fewer markers for longer x-axis
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
    ax.set_title(
        "4×4 Sudoku (6–8 empties, no-mask protocol, T=16)\n"
        "Success Rate vs Training Steps\n"
        "(20k steps, greedy evaluation)"
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 20000)
    ax.grid(True, alpha=0.3)

    # Legend below plot in two columns (matches paper style)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    ax.tick_params(axis="both", which="major")

    # Leave room for legend under axes
    fig.subplots_adjust(bottom=0.28)

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hard_4x4_baselines_success_vs_steps.pdf"

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {str(output_path).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
