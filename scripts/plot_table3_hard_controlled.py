#!/usr/bin/env python3
"""
Generate plots for Table 3 Hard CONTROLLED 2×2 experiment.

Deconfounds contraction vs projection effects on hard 4x4 Sudoku (6-8 empties).

2×2 Design:
  Factor A: enable_contraction ∈ {false, true}
  Factor B: latent_ball_radius R ∈ {0.0, 10.0}

Outputs:
  - table3_hard_controlled_2x2_bar.pdf: Bar chart with 2×2 layout
  - table3_hard_controlled_curves.pdf: Learning curves for all 4 cells

Usage:
    buck2 run //buiksat_trm:plot_table3_hard_controlled
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


# Cell configuration for 2×2 design
CELL_CONFIG = {
    "nc_r0": {
        "name": "No Contraction, R=0",
        "short_name": "NC, R=0",
        "color": "#9467bd",  # purple
        "marker": "o",
        "contraction": False,
        "radius": 0.0,
    },
    "nc_r10": {
        "name": "No Contraction, R=10",
        "short_name": "NC, R=10",
        "color": "#8c564b",  # brown
        "marker": "s",
        "contraction": False,
        "radius": 10.0,
    },
    "c_r0": {
        "name": "Contraction, R=0",
        "short_name": "C, R=0",
        "color": "#2ca02c",  # green
        "marker": "^",
        "contraction": True,
        "radius": 0.0,
    },
    "c_r10": {
        "name": "Contraction, R=10",
        "short_name": "C, R=10",
        "color": "#1f77b4",  # blue
        "marker": "D",
        "contraction": True,
        "radius": 10.0,
    },
}


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
    """Load data for each seed separately."""
    all_data = {}
    for i, log_file in enumerate(log_files):
        data = parse_log_file(log_file)
        if data:
            all_data[i] = data
    return all_data


def compute_mean_std(seed_data: Dict[int, List[Tuple[int, float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std across seeds."""
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


def get_final_success_rates(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Get final success rate for each cell and seed."""
    final_rates = {}

    for cell in CELL_CONFIG.keys():
        final_rates[cell] = {}
        for seed in [42, 123, 456]:
            log_file = results_dir / f"{cell}_s{seed}.log"
            if log_file.exists():
                data = parse_log_file(log_file)
                if data:
                    final_rates[cell][seed] = data[-1][1]  # Last success rate

    return final_rates


def plot_2x2_bar_chart(results_dir: Path, output_dir: Path):
    """Create 2×2 bar chart showing final success rates."""
    final_rates = get_final_success_rates(results_dir)

    # Compute means and stds for each cell
    cell_means = {}
    cell_stds = {}
    for cell in CELL_CONFIG.keys():
        if cell in final_rates and final_rates[cell]:
            values = list(final_rates[cell].values())
            cell_means[cell] = np.mean(values)
            cell_stds[cell] = np.std(values)
        else:
            cell_means[cell] = 0
            cell_stds[cell] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar positions: grouped by contraction, side-by-side by radius
    x = np.array([0, 1])  # Two groups: No Contraction, Contraction
    width = 0.35

    # R=0 bars (left in each group)
    r0_means = [cell_means["nc_r0"], cell_means["c_r0"]]
    r0_stds = [cell_stds["nc_r0"], cell_stds["c_r0"]]

    # R=10 bars (right in each group)
    r10_means = [cell_means["nc_r10"], cell_means["c_r10"]]
    r10_stds = [cell_stds["nc_r10"], cell_stds["c_r10"]]

    bars1 = ax.bar(x - width/2, r0_means, width, yerr=r0_stds,
                   label='R=0 (Projection OFF)', color='#d62728', capsize=5)
    bars2 = ax.bar(x + width/2, r10_means, width, yerr=r10_stds,
                   label='R=10 (Projection ON)', color='#1f77b4', capsize=5)

    # Labels and formatting
    ax.set_ylabel('Final Success Rate')
    ax.set_title('2×2 Controlled Experiment: Contraction × Projection\n(Hard 4×4 Sudoku, 6-8 empties, T=16, 20k steps)')
    ax.set_xticks(x)
    ax.set_xticklabels(['No Contraction', 'Contraction'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()

    # Save
    output_path = output_dir / "table3_hard_controlled_2x2_bar.pdf"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {str(output_path).replace('.pdf', '.png')}")


def plot_learning_curves(results_dir: Path, output_dir: Path):
    """Create learning curves plot for all 4 cells."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    cell_order = ["nc_r0", "nc_r10", "c_r0", "c_r10"]

    for cell_key in cell_order:
        config = CELL_CONFIG[cell_key]
        log_files = list(results_dir.glob(f"{cell_key}_s*.log"))

        if not log_files:
            print(f"Warning: No log files found for {cell_key}")
            continue

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
                markevery=20,
                label=f"{config['short_name']} (S={num_seeds})"
            )

    # Random baseline (0% for hard puzzles)
    ax.axhline(
        y=0.0,
        color='gray',
        linestyle='--',
        linewidth=2.5,
        alpha=0.7,
        label='Random (0%)'
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title("2×2 Controlled: Contraction × Projection\n(Hard 4×4 Sudoku, 6-8 empties, T=16)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 20000)
    ax.grid(True, alpha=0.3)

    # Legend below plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    ax.tick_params(axis="both", which="major")

    fig.subplots_adjust(bottom=0.30)

    # Save
    output_path = output_dir / "table3_hard_controlled_curves.pdf"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {str(output_path).replace('.pdf', '.png')}")


def print_summary_table(results_dir: Path):
    """Print 2×2 summary table to console."""
    final_rates = get_final_success_rates(results_dir)

    print("\n" + "=" * 60)
    print("2×2 Controlled Experiment Results")
    print("=" * 60)
    print("\nFinal Success Rates (mean ± std over 3 seeds):")
    print()
    print("                     R=0 (proj OFF)     R=10 (proj ON)")
    print("-" * 60)

    for contraction, label in [(False, "No Contraction"), (True, "Contraction   ")]:
        prefix = "nc" if not contraction else "c"

        r0_vals = list(final_rates.get(f"{prefix}_r0", {}).values())
        r10_vals = list(final_rates.get(f"{prefix}_r10", {}).values())

        r0_str = f"{np.mean(r0_vals):.3f} ± {np.std(r0_vals):.3f}" if r0_vals else "N/A"
        r10_str = f"{np.mean(r10_vals):.3f} ± {np.std(r10_vals):.3f}" if r10_vals else "N/A"

        print(f"{label}     {r0_str}         {r10_str}")

    print("-" * 60)

    # Main effects analysis
    print("\nMain Effects Analysis:")

    # Effect of contraction (averaging over R)
    nc_all = list(final_rates.get("nc_r0", {}).values()) + list(final_rates.get("nc_r10", {}).values())
    c_all = list(final_rates.get("c_r0", {}).values()) + list(final_rates.get("c_r10", {}).values())

    if nc_all and c_all:
        nc_mean = np.mean(nc_all)
        c_mean = np.mean(c_all)
        contraction_effect = c_mean - nc_mean
        print(f"  Contraction effect: {contraction_effect:+.3f} (C - NC)")

    # Effect of projection (averaging over contraction)
    r0_all = list(final_rates.get("nc_r0", {}).values()) + list(final_rates.get("c_r0", {}).values())
    r10_all = list(final_rates.get("nc_r10", {}).values()) + list(final_rates.get("c_r10", {}).values())

    if r0_all and r10_all:
        r0_mean = np.mean(r0_all)
        r10_mean = np.mean(r10_all)
        projection_effect = r10_mean - r0_mean
        print(f"  Projection effect:  {projection_effect:+.3f} (R10 - R0)")

    # Interaction effect
    if all([final_rates.get(k, {}) for k in ["nc_r0", "nc_r10", "c_r0", "c_r10"]]):
        nc_r0 = np.mean(list(final_rates["nc_r0"].values()))
        nc_r10 = np.mean(list(final_rates["nc_r10"].values()))
        c_r0 = np.mean(list(final_rates["c_r0"].values()))
        c_r10 = np.mean(list(final_rates["c_r10"].values()))

        # Interaction = (C_R10 - C_R0) - (NC_R10 - NC_R0)
        interaction = (c_r10 - c_r0) - (nc_r10 - nc_r0)
        print(f"  Interaction:        {interaction:+.3f}")

    print()


def main():
    apply_paper_style()

    results_dir = Path("/home/buiksat/trm_bellman/results/table3_hard_6to8_controlled")
    output_dir = Path("/home/buiksat/UPI_TRM/UPI_TRM_ICML/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the experiments first: scripts/run_table3_hard_controlled_4gpu.sh")
        return

    # Print summary table
    print_summary_table(results_dir)

    # Generate plots
    plot_2x2_bar_chart(results_dir, output_dir)
    plot_learning_curves(results_dir, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
