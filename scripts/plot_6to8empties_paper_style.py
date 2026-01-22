#!/usr/bin/env python3
"""
Generate paper-style plots for 6-8 empties experiments.
Matches the style of trivial_baselines_vs_no_contraction_success_vs_steps.png

Key style elements:
- "UPI-TRM (...)" prefix for all UPI-TRM variants
- "Episodic-z" / "Persistent-z" terminology (not "Reset-z")
- "S=3" for seed count (not "n=3" since n is z-iterations in paper)
- Legend below plot in two columns
- Larger fonts, thicker mean curves, faint per-seed lines
- Correct random baseline for 6-8 empties (0%, not 52%)
"""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def apply_paper_style():
    """Apply paper-quality styling matching trivial plot."""
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
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


# Algorithm config with CORRECT naming per user requirements
# - "UPI-TRM ..." prefix for all UPI-TRM based methods
# - "Episodic-z" and "Persistent-z" terminology
# - R=10 indicates projection radius
ALGO_CONFIG = {
    # UPI-TRM main variants (the 2x2 grid)
    "upi_trm": {
        "name": "UPI-TRM (Episodic-z, contraction, R=10)",
        "color": "#1f77b4",  # blue
        "marker": "o"
    },
    "ablation_persistent_z": {
        "name": "UPI-TRM (Persistent-z, contraction, R=10)",
        "color": "#9467bd",  # purple
        "marker": "v"
    },
    "no_contraction": {
        "name": "UPI-TRM (Episodic-z, no contraction, R=10)",
        "color": "#8c564b",  # brown
        "marker": "<"
    },
    "persistent_z_no_contraction": {
        "name": "UPI-TRM (Persistent-z, no contraction, R=10)",
        "color": "#e377c2",  # pink
        "marker": "P"
    },
    # Other ablations
    "ablation_no_conservative": {
        "name": "UPI-TRM (no conservative, R=10)",
        "color": "#ff7f0e",  # orange
        "marker": "^"
    },
    # Baselines (not UPI-TRM)
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
    "ppo": {
        "name": "PPO",
        "color": "#ff7f0e",  # orange
        "marker": "s"
    },
}

# Random baseline for 6-8 empties (computed via eval_random_baseline.py)
# 0% success rate with 100 episodes per seed, 3 seeds
RANDOM_BASELINE = {
    "success_rate": 0.0,  # 0% (much lower than trivial's 52%)
    "mean_score": -3.26,
}


def load_learning_curves(csv_path: Path) -> dict:
    """Load learning curve data from CSV."""
    data = defaultdict(lambda: defaultdict(lambda: {
        "steps": [],
        "success_rates": [],
        "mean_scores": [],
    }))

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algorithm"]
            seed = int(row["seed"])
            step = int(row["step"])
            success_rate = float(row["success_rate"])
            mean_score = float(row["mean_score"])

            data[algo][seed]["steps"].append(step)
            data[algo][seed]["success_rates"].append(success_rate)
            data[algo][seed]["mean_scores"].append(mean_score)

    return data


def compute_mean_curve(seed_data: dict, metric_key: str = "success_rates") -> tuple:
    """Compute mean curve across seeds."""
    if not seed_data:
        return [], []

    all_steps = set()
    for seed, d in seed_data.items():
        all_steps.update(d["steps"])
    steps = sorted(all_steps)

    if not steps:
        return [], []

    mean_values = []
    for step in steps:
        values = []
        for seed, d in seed_data.items():
            if step in d["steps"]:
                idx = d["steps"].index(step)
                if idx < len(d[metric_key]):
                    val = d[metric_key][idx]
                    if val is not None:
                        values.append(val)
        if values:
            mean_values.append(sum(values) / len(values))
        else:
            mean_values.append(None)

    valid = [(s, r) for s, r in zip(steps, mean_values) if r is not None]
    if not valid:
        return [], []
    steps, mean_values = zip(*valid)

    return list(steps), list(mean_values)


def plot_success_vs_steps(data: dict, output_dir: Path):
    """
    Plot success rate vs training steps.
    Matches style of trivial_baselines_vs_no_contraction_success_vs_steps.png
    """
    apply_paper_style()

    # Increased height to accommodate legend below x-axis label
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # Order algorithms for plotting (put best performers first in legend)
    algo_order = [
        "persistent_z_no_contraction",
        "no_contraction",
        "upi_trm",
        "ablation_persistent_z",
        "a2c",
        "dqn",
        "ablation_no_conservative",
        "ppo",
    ]

    for algo in algo_order:
        if algo not in data:
            continue

        seed_data = data[algo]
        config = ALGO_CONFIG.get(algo, {"name": algo, "color": "gray", "marker": "o"})

        # Plot individual seed curves (thin, semi-transparent)
        for seed, d in seed_data.items():
            ax.plot(d["steps"], d["success_rates"],
                    color=config["color"], alpha=0.18, linewidth=0.9)

        # Plot mean curve (thick)
        mean_steps, mean_rates = compute_mean_curve(seed_data, "success_rates")
        if mean_steps:
            num_seeds = len(seed_data)
            ax.plot(mean_steps, mean_rates,
                    color=config["color"], linewidth=3.0,
                    marker=config["marker"], markersize=6, markevery=8,
                    label=f"{config['name']} (S={num_seeds})")

    # Add random baseline horizontal line
    # For 6-8 empties: 0% (unlike trivial's 52%)
    random_pct = RANDOM_BASELINE["success_rate"]
    ax.axhline(y=random_pct, color='gray', linestyle='--',
               linewidth=2.5, alpha=0.7, label=f'Random ({random_pct:.0%})')

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title("4×4 Sudoku: Success Rate vs Training Steps\n(Feasibility Checker, 6-8 empties, 20k steps)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.3)

    # Legend below plot in two columns (matches trivial plot style)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    ax.tick_params(axis="both", which="major")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "feasibility_success_vs_steps.png"
    pdf_path = output_dir / "feasibility_success_vs_steps.pdf"

    # Leave room for legend under axes
    fig.subplots_adjust(bottom=0.38)
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    plt.close(fig)


def main():
    # Use hardcoded path for fbcode environment
    # Data is at ~/trm_bellman/results/... which links to ~/fbsource/fbcode/buiksat_trm/...
    import os
    home = os.path.expanduser("~")
    base_path = Path(home) / "trm_bellman"
    data_dir = base_path / "results/plot_data_6to8empties_all_ablations_20k"
    output_dir = base_path / "results/plots_6to8empties_all_ablations_20k"

    csv_path = data_dir / "plot_data_feasibility_learning_curves.csv"
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    data = load_learning_curves(csv_path)

    print(f"Found algorithms: {list(data.keys())}")
    for algo, seed_data in data.items():
        print(f"  {algo}: seeds {list(seed_data.keys())}")

    print("\nGenerating paper-style success vs steps plot...")
    plot_success_vs_steps(data, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
