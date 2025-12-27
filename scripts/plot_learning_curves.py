#!/usr/bin/env python3
"""
Generate learning curve plots for ICML 2026 paper.

Creates publication-ready figures comparing UPI-TRM with baselines.

Usage:
    python scripts/plot_learning_curves.py --input artifacts/summary.parquet --output paper/figures/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ICML 2026 figure sizing
# Single column: 3.25 inches wide
# Double column: 6.75 inches wide
SINGLE_COLUMN_WIDTH = 3.25
DOUBLE_COLUMN_WIDTH = 6.75
FIGURE_HEIGHT = 2.5

# Color palette for different methods
METHOD_COLORS = {
    "upi-trm": "#2E86AB",      # Blue - our method
    "ppo-trm": "#A23B72",      # Purple - PPO baseline
    "a2c-trm": "#F18F01",      # Orange - A2C baseline
    "ppo-norec": "#C73E1D",    # Red - No recursion
    "a2c-norec": "#95190C",    # Dark red
}

METHOD_LABELS = {
    "upi-trm": "UPI-TRM (Ours)",
    "ppo-trm": "PPO + TRM",
    "a2c-trm": "A2C + TRM",
    "ppo-norec": "PPO + MLP",
    "a2c-norec": "A2C + MLP",
}


def setup_plotting_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def compute_confidence_interval(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compute mean and confidence interval for grouped data.

    Args:
        data: DataFrame with metrics
        x_col: Column for x-axis (usually 'step')
        y_col: Column for y-axis (metric)
        group_col: Column for grouping (usually 'seed')
        confidence: Confidence level (default 95%)

    Returns:
        DataFrame with x, mean, ci_lower, ci_upper columns
    """
    from scipy import stats

    grouped = data.groupby(x_col)[y_col]

    result = pd.DataFrame({
        "x": grouped.mean().index,
        "mean": grouped.mean().values,
        "std": grouped.std().values,
        "n": grouped.count().values,
    })

    # Compute CI using t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, result["n"] - 1)
    margin = t_critical * result["std"] / np.sqrt(result["n"])

    result["ci_lower"] = result["mean"] - margin
    result["ci_upper"] = result["mean"] + margin

    return result


def plot_learning_curves_main(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "eval_success_rate",
    y_label: str = "Success Rate (%)",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT),
    configs_to_plot: Optional[List[str]] = None,
) -> None:
    """
    Plot main comparison: UPI-TRM vs baselines.

    Args:
        df: Aggregated data from aggregate_runs.py
        output_path: Path to save figure
        metric: Metric column to plot
        y_label: Y-axis label
        title: Optional title
        figsize: Figure size (width, height) in inches
        configs_to_plot: Specific configs to include (None = auto-detect)
    """
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Auto-detect main configs if not specified
    if configs_to_plot is None:
        # Look for main algorithm configs
        configs_to_plot = []
        for config in df["config_name"].unique():
            config_lower = config.lower()
            # Include main comparison configs
            if any(key in config_lower for key in ["upi-trm", "ppo", "a2c"]):
                # Exclude ablations
                if "ablation" not in config_lower and "no_" not in config_lower:
                    configs_to_plot.append(config)

    # Sort to ensure consistent ordering
    configs_to_plot = sorted(configs_to_plot)

    for config in configs_to_plot:
        config_data = df[df["config_name"] == config]

        if config_data.empty or metric not in config_data.columns:
            print(f"Warning: No data for {config} with metric {metric}")
            continue

        # Drop NaN values
        config_data = config_data.dropna(subset=[metric])

        if config_data.empty:
            continue

        # Compute mean and CI
        stats_df = compute_confidence_interval(
            config_data, "step", metric, "seed"
        )

        # Determine color and label
        color = METHOD_COLORS.get(config.lower().split("_")[0], "#666666")
        label = METHOD_LABELS.get(config.lower().split("_")[0], config)

        # Plot mean line
        ax.plot(stats_df["x"], stats_df["mean"], label=label, color=color)

        # Plot confidence interval
        ax.fill_between(
            stats_df["x"],
            stats_df["ci_lower"],
            stats_df["ci_upper"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    ax.legend(loc="lower right", framealpha=0.9)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"))  # Also save PNG for preview
    plt.close()

    print(f"Saved: {output_path}")


def plot_learning_curves_ablations(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "eval_success_rate",
    y_label: str = "Success Rate (%)",
    figsize: Tuple[float, float] = (DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT * 1.5),
) -> None:
    """
    Plot ablation study: All 9 ablation configurations.

    Args:
        df: Aggregated data from aggregate_runs.py
        output_path: Path to save figure
        metric: Metric column to plot
        y_label: Y-axis label
        figsize: Figure size (width, height) in inches
    """
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Find ablation configs
    ablation_configs = [c for c in df["config_name"].unique()
                       if "ablation" in c.lower() or "no_" in c.lower()]

    # Add main theory-exact config for comparison
    main_configs = [c for c in df["config_name"].unique()
                   if "theory_exact" in c.lower() and "ablation" not in c.lower()]

    all_configs = main_configs + sorted(ablation_configs)

    # Generate colors using a colormap
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i / len(all_configs)) for i in range(len(all_configs))]

    for i, config in enumerate(all_configs):
        config_data = df[df["config_name"] == config]

        if config_data.empty or metric not in config_data.columns:
            continue

        config_data = config_data.dropna(subset=[metric])

        if config_data.empty:
            continue

        # Compute mean and CI
        stats_df = compute_confidence_interval(
            config_data, "step", metric, "seed"
        )

        # Create label
        if "ablation" in config.lower():
            label = config.replace("ablation_", "- ")
        elif "theory_exact" in config.lower():
            label = "Full (all features)"
        else:
            label = config

        # Highlight main result
        linewidth = 2 if "Full" in label else 1
        linestyle = "-" if "Full" in label else "--"

        ax.plot(
            stats_df["x"],
            stats_df["mean"],
            label=label,
            color=colors[i],
            linewidth=linewidth,
            linestyle=linestyle,
        )

        ax.fill_between(
            stats_df["x"],
            stats_df["ci_lower"],
            stats_df["ci_upper"],
            alpha=0.1,
            color=colors[i],
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(y_label)
    ax.set_title("Ablation Study: Effect of Theory-Exact Features")

    # Place legend outside plot
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.9,
        fontsize=6,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_multiple_metrics(
    df: pd.DataFrame,
    output_path: str,
    metrics: List[Tuple[str, str]],  # (column_name, display_label)
    figsize: Tuple[float, float] = (DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT * 2),
) -> None:
    """
    Plot multiple metrics in subplots.

    Args:
        df: Aggregated data
        output_path: Path to save figure
        metrics: List of (column_name, display_label) tuples
        figsize: Figure size
    """
    setup_plotting_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    main_configs = [c for c in df["config_name"].unique()
                   if "ablation" not in c.lower()][:5]

    cmap = plt.cm.get_cmap("tab10")

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        for i, config in enumerate(main_configs):
            config_data = df[df["config_name"] == config]

            if config_data.empty or metric_col not in config_data.columns:
                continue

            config_data = config_data.dropna(subset=[metric_col])

            if config_data.empty:
                continue

            stats_df = compute_confidence_interval(
                config_data, "step", metric_col, "seed"
            )

            ax.plot(
                stats_df["x"],
                stats_df["mean"],
                label=config,
                color=cmap(i / len(main_configs)),
            )
            ax.fill_between(
                stats_df["x"],
                stats_df["ci_lower"],
                stats_df["ci_upper"],
                alpha=0.2,
                color=cmap(i / len(main_configs)),
            )

        ax.set_ylabel(metric_label)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=6)

    axes[-1].set_xlabel("Training Steps")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate learning curve plots")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/summary.parquet",
        help="Input parquet file from aggregate_runs.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run scripts/aggregate_runs.py first to generate data.")
        return

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Configs: {df['config_name'].unique().tolist()}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate main comparison plot
    plot_learning_curves_main(
        df,
        str(output_dir / "learning_curves_main.pdf"),
        metric="eval_success_rate",
        y_label="Success Rate (%)",
    )

    # Generate ablation plot
    plot_learning_curves_ablations(
        df,
        str(output_dir / "learning_curves_ablations.pdf"),
        metric="eval_success_rate",
        y_label="Success Rate (%)",
    )

    # Generate multi-metric plot
    plot_multiple_metrics(
        df,
        str(output_dir / "learning_curves_multi.pdf"),
        metrics=[
            ("eval_success_rate", "Success Rate (%)"),
            ("eval_mean_score", "Mean Score"),
            ("policy_entropy", "Policy Entropy"),
        ],
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
