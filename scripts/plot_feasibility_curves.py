#!/usr/bin/env python3
"""
Plot feasibility experiment learning curves.

Usage:
    python scripts/plot_feasibility_curves.py --input results/plot_data/plot_data_feasibility_learning_curves.csv --output-dir results/plots
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


# Algorithm display names and colors
ALGO_CONFIG = {
    "upi_trm": {"name": "UPI-TRM", "color": "#1f77b4", "marker": "o"},
    "ppo": {"name": "PPO", "color": "#ff7f0e", "marker": "s"},
    "a2c": {"name": "A2C", "color": "#2ca02c", "marker": "^"},
    "dqn": {"name": "DQN", "color": "#d62728", "marker": "D"},
    "ablation_no_conservative": {"name": "No Conservative", "color": "#9467bd", "marker": "v"},
    "ablation_no_contraction": {"name": "No Contraction", "color": "#8c564b", "marker": "<"},
}


def load_data(csv_path: Path) -> dict:
    """Load learning curve data from CSV."""
    data = defaultdict(lambda: defaultdict(lambda: {"steps": [], "success_rates": [], "mean_scores": []}))

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


def compute_mean_curve(seed_data: dict) -> tuple:
    """Compute mean curve across seeds with matching steps."""
    if not seed_data:
        return [], []

    # Find all unique steps and sort them
    all_steps = set()
    for seed, d in seed_data.items():
        all_steps.update(d["steps"])
    steps = sorted(all_steps)

    if not steps:
        return [], []

    # For each step, compute mean across seeds that have that step
    mean_rates = []
    for step in steps:
        rates = []
        for seed, d in seed_data.items():
            if step in d["steps"]:
                idx = d["steps"].index(step)
                rates.append(d["success_rates"][idx])
        if rates:
            mean_rates.append(sum(rates) / len(rates))
        else:
            mean_rates.append(None)

    # Filter out None values
    valid = [(s, r) for s, r in zip(steps, mean_rates) if r is not None]
    if not valid:
        return [], []
    steps, mean_rates = zip(*valid)

    return list(steps), list(mean_rates)


def plot_success_vs_steps(data: dict, output_dir: Path):
    """Plot success rate vs training steps."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot main algorithms first
    main_algos = ["upi_trm", "ppo", "a2c", "dqn"]

    for algo in main_algos:
        if algo not in data:
            continue

        seed_data = data[algo]
        config = ALGO_CONFIG.get(algo, {"name": algo, "color": "gray", "marker": "o"})

        # Plot individual seed curves (thin, semi-transparent)
        for seed, d in seed_data.items():
            ax.plot(d["steps"], d["success_rates"],
                    color=config["color"], alpha=0.3, linewidth=1)

        # Plot mean curve (thick)
        mean_steps, mean_rates = compute_mean_curve(seed_data)
        if mean_steps:
            ax.plot(mean_steps, mean_rates,
                    color=config["color"], linewidth=2.5,
                    marker=config["marker"], markersize=4, markevery=10,
                    label=f"{config['name']} (n={len(seed_data)})")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("4×4 Sudoku: Success Rate vs Training Steps\n(Feasibility Checker, trivial dataset)", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "feasibility_success_vs_steps.png"
    pdf_path = output_dir / "feasibility_success_vs_steps.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    plt.close(fig)


def plot_score_vs_steps(data: dict, output_dir: Path):
    """Plot mean score vs training steps."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    main_algos = ["upi_trm", "ppo", "a2c", "dqn"]

    for algo in main_algos:
        if algo not in data:
            continue

        seed_data = data[algo]
        config = ALGO_CONFIG.get(algo, {"name": algo, "color": "gray", "marker": "o"})

        # Plot individual seed curves (thin, semi-transparent)
        for seed, d in seed_data.items():
            ax.plot(d["steps"], d["mean_scores"],
                    color=config["color"], alpha=0.3, linewidth=1)

        # Compute mean score curve
        if not seed_data:
            continue

        all_steps = set()
        for seed, d in seed_data.items():
            all_steps.update(d["steps"])
        steps = sorted(all_steps)

        mean_scores = []
        for step in steps:
            scores = []
            for seed, d in seed_data.items():
                if step in d["steps"]:
                    idx = d["steps"].index(step)
                    scores.append(d["mean_scores"][idx])
            if scores:
                mean_scores.append(sum(scores) / len(scores))
            else:
                mean_scores.append(None)

        valid = [(s, sc) for s, sc in zip(steps, mean_scores) if sc is not None]
        if valid:
            steps, mean_scores = zip(*valid)
            ax.plot(steps, mean_scores,
                    color=config["color"], linewidth=2.5,
                    marker=config["marker"], markersize=4, markevery=10,
                    label=f"{config['name']} (n={len(seed_data)})")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("4×4 Sudoku: Mean Score vs Training Steps\n(Feasibility Checker, trivial dataset)", fontsize=14)
    ax.set_ylim(0, 16)
    ax.set_xlim(0, None)
    ax.axhline(y=16, color='green', linestyle='--', alpha=0.5, label='Perfect (16)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "feasibility_score_vs_steps.png"
    pdf_path = output_dir / "feasibility_score_vs_steps.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    plt.close(fig)


def plot_ablations(data: dict, output_dir: Path):
    """Plot ablation comparison vs UPI-TRM."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return

    ablation_algos = ["ablation_no_conservative", "ablation_no_contraction"]
    has_ablations = any(algo in data for algo in ablation_algos)

    if not has_ablations or "upi_trm" not in data:
        print("No ablation data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot UPI-TRM baseline
    for algo in ["upi_trm"] + ablation_algos:
        if algo not in data:
            continue

        seed_data = data[algo]
        config = ALGO_CONFIG.get(algo, {"name": algo, "color": "gray", "marker": "o"})

        mean_steps, mean_rates = compute_mean_curve(seed_data)
        if mean_steps:
            ax.plot(mean_steps, mean_rates,
                    color=config["color"], linewidth=2.5,
                    marker=config["marker"], markersize=4, markevery=10,
                    label=f"{config['name']} (n={len(seed_data)})")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("4×4 Sudoku: Ablation Study\n(Feasibility Checker, trivial dataset)", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "feasibility_ablations.png"
    pdf_path = output_dir / "feasibility_ablations.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot feasibility experiment learning curves")
    parser.add_argument("--input", default="results/plot_data/plot_data_feasibility_learning_curves.csv",
                        help="Input CSV file")
    parser.add_argument("--output-dir", default="results/plots", help="Output directory for plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run parse_feasibility_logs.py first to generate CSV data.")
        return

    print(f"Loading data from {input_path}...")
    data = load_data(input_path)

    print(f"Found algorithms: {list(data.keys())}")
    for algo, seed_data in data.items():
        print(f"  {algo}: seeds {list(seed_data.keys())}")

    print("\nGenerating plots...")
    plot_success_vs_steps(data, output_dir)
    plot_score_vs_steps(data, output_dir)
    plot_ablations(data, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
