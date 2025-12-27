#!/usr/bin/env python3
"""
Generate ablation study bar chart for ICML 2026 paper.

Creates a bar chart showing final success rate for each ablation configuration.

Usage:
    python scripts/plot_ablations.py --input artifacts/summary.parquet --output paper/figures/ablation_bar_chart.pdf
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ICML figure sizing
SINGLE_COLUMN_WIDTH = 3.25
DOUBLE_COLUMN_WIDTH = 6.75
FIGURE_HEIGHT = 2.5

# Ablation labels (human-readable)
ABLATION_LABELS = {
    "shaped_theory_exact": "Full (All Features)",
    "ablation_no_exact_baseline": "- Exact Baseline",
    "ablation_no_contraction": "- Contraction",
    "ablation_no_conservative_mixture": "- Conservative Mixture",
    "ablation_no_projection": "- Projection",
    "ablation_no_k_step": "- K-step Targets",
    "ablation_no_shaped_rewards": "- Shaped Rewards",
    "ablation_no_theory_mixture": "- Theory Mixture",
    "sparse_theory_exact": "Sparse Rewards",
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
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.axis": "y",
    })


def get_final_metrics(
    df: pd.DataFrame,
    metric: str = "eval_success_rate",
    step_percentile: float = 0.95,
) -> pd.DataFrame:
    """
    Get final metric values for each config/seed.

    Args:
        df: Aggregated data from aggregate_runs.py
        metric: Metric column name
        step_percentile: Use metrics from this percentile of training
                        (0.95 = last 5% of training)

    Returns:
        DataFrame with config_name, seed, metric columns
    """
    results = []

    for config in df["config_name"].unique():
        config_data = df[df["config_name"] == config]

        if metric not in config_data.columns:
            continue

        # Get step threshold (last X% of training)
        max_step = config_data["step"].max()
        step_threshold = max_step * step_percentile

        # Get data from final portion of training
        final_data = config_data[config_data["step"] >= step_threshold]

        for seed in final_data["seed"].unique():
            seed_data = final_data[final_data["seed"] == seed]
            values = seed_data[metric].dropna()

            if len(values) > 0:
                results.append({
                    "config_name": config,
                    "seed": seed,
                    metric: values.mean(),  # Average over final portion
                })

    return pd.DataFrame(results)


def plot_ablation_bar_chart(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "eval_success_rate",
    y_label: str = "Final Success Rate (%)",
    title: str = "Ablation Study: Contribution of Each Feature",
    figsize: Tuple[float, float] = (SINGLE_COLUMN_WIDTH, FIGURE_HEIGHT * 1.2),
    horizontal: bool = True,
) -> None:
    """
    Create bar chart showing final performance for each ablation.

    Args:
        df: Aggregated data from aggregate_runs.py
        output_path: Path to save figure
        metric: Metric column to plot
        y_label: Axis label for metric
        title: Figure title
        figsize: Figure size in inches
        horizontal: If True, plot horizontal bars
    """
    setup_plotting_style()

    # Get final metrics
    final_df = get_final_metrics(df, metric)

    if final_df.empty:
        print(f"Error: No data found for metric {metric}")
        return

    # Compute mean and std per config
    stats = final_df.groupby("config_name")[metric].agg(["mean", "std", "count"])
    stats = stats.reset_index()

    # Sort by mean (descending)
    stats = stats.sort_values("mean", ascending=True if horizontal else False)

    # Create labels
    labels = []
    for config in stats["config_name"]:
        config_lower = config.lower()
        # Try to match to known labels
        matched = False
        for key, label in ABLATION_LABELS.items():
            if key in config_lower:
                labels.append(label)
                matched = True
                break
        if not matched:
            labels.append(config)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(stats))
    means = stats["mean"].values
    stds = stats["std"].values

    # Color bars: green for full, orange for ablations
    colors = []
    for label in labels:
        if "Full" in label:
            colors.append("#2E86AB")  # Blue for full model
        elif label.startswith("-"):
            colors.append("#F18F01")  # Orange for ablations
        else:
            colors.append("#95190C")  # Red for sparse baseline

    if horizontal:
        bars = ax.barh(x, means, xerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel(y_label)
        ax.set_xlim(left=0)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            width = bar.get_width()
            ax.annotate(
                f"{mean:.1f}",
                xy=(width + std + 1, bar.get_y() + bar.get_height() / 2),
                va="center",
                fontsize=6,
            )
    else:
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(y_label)
        ax.set_ylim(bottom=0)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(
                f"{mean:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                ha="center",
                fontsize=6,
            )

    ax.set_title(title)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_ablation_grouped_bars(
    df: pd.DataFrame,
    output_path: str,
    metrics: List[Tuple[str, str]] = None,
    figsize: Tuple[float, float] = (DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT),
) -> None:
    """
    Create grouped bar chart showing multiple metrics per ablation.

    Args:
        df: Aggregated data
        output_path: Path to save figure
        metrics: List of (column_name, display_label) tuples
        figsize: Figure size
    """
    setup_plotting_style()

    if metrics is None:
        metrics = [
            ("eval_success_rate", "Success Rate"),
            ("eval_mean_score", "Mean Score"),
        ]

    # Get final metrics for each
    all_stats = []
    for metric_col, metric_label in metrics:
        final_df = get_final_metrics(df, metric_col)
        if final_df.empty:
            continue

        stats = final_df.groupby("config_name")[metric_col].agg(["mean", "std"])
        stats = stats.reset_index()
        stats["metric"] = metric_label
        stats = stats.rename(columns={"mean": "value", "std": "error"})
        all_stats.append(stats)

    if not all_stats:
        print("Error: No data found for any metric")
        return

    combined_stats = pd.concat(all_stats, ignore_index=True)

    # Get unique configs sorted by first metric mean
    first_metric = metrics[0][0]
    config_order = get_final_metrics(df, first_metric).groupby("config_name")[first_metric].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    n_configs = len(config_order)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    x = np.arange(n_configs)

    colors = plt.cm.get_cmap("tab10")

    for i, (metric_col, metric_label) in enumerate(metrics):
        metric_stats = combined_stats[combined_stats["metric"] == metric_label]
        # Reorder to match config_order
        values = []
        errors = []
        for config in config_order:
            row = metric_stats[metric_stats["config_name"] == config]
            if len(row) > 0:
                values.append(row["value"].values[0])
                errors.append(row["error"].values[0])
            else:
                values.append(0)
                errors.append(0)

        offset = (i - n_metrics / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, yerr=errors, label=metric_label,
               capsize=2, color=colors(i))

    # Create labels
    labels = []
    for config in config_order:
        config_lower = config.lower()
        matched = False
        for key, label in ABLATION_LABELS.items():
            if key in config_lower:
                labels.append(label)
                matched = True
                break
        if not matched:
            labels.append(config[:15])  # Truncate long names

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation bar charts")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/summary.parquet",
        help="Input parquet file from aggregate_runs.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures/ablation_bar_chart.pdf",
        help="Output PDF file path",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_success_rate",
        help="Metric to plot",
    )
    parser.add_argument(
        "--horizontal",
        action="store_true",
        default=True,
        help="Use horizontal bars (default: True)",
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

    # Generate bar chart
    plot_ablation_bar_chart(
        df,
        args.output,
        metric=args.metric,
        horizontal=args.horizontal,
    )

    # Also generate grouped bars if multiple metrics available
    output_dir = Path(args.output).parent
    plot_ablation_grouped_bars(
        df,
        str(output_dir / "ablation_grouped_bars.pdf"),
    )

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
