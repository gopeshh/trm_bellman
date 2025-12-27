#!/usr/bin/env python3
"""
Generate LaTeX tables for ICML 2026 paper.

Creates publication-ready tables comparing UPI-TRM with baselines and ablations.

Usage:
    python scripts/make_tables.py --input artifacts/summary.parquet --output paper/tables/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Method display names
METHOD_NAMES = {
    "upi-trm": "UPI-TRM (Ours)",
    "ppo-trm": "PPO + TRM",
    "a2c-trm": "A2C + TRM",
    "ppo-norec": "PPO + MLP",
    "a2c-norec": "A2C + MLP",
    "shaped_theory_exact": "UPI-TRM (Full)",
    "sparse_theory_exact": "UPI-TRM (Sparse)",
}

# Ablation labels
ABLATION_LABELS = {
    "ablation_no_exact_baseline": "w/o Exact Baseline",
    "ablation_no_contraction": "w/o Contraction",
    "ablation_no_conservative_mixture": "w/o Conservative Mix.",
    "ablation_no_projection": "w/o Projection",
    "ablation_no_k_step": "w/o K-step Targets",
    "ablation_no_shaped_rewards": "w/o Shaped Rewards",
    "ablation_no_theory_mixture": "w/o Theory Mixture",
}


def get_final_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    step_percentile: float = 0.95,
) -> pd.DataFrame:
    """
    Get final metric values for each config/seed.

    Args:
        df: Aggregated data
        metrics: List of metric column names
        step_percentile: Use metrics from this percentile of training

    Returns:
        DataFrame with config_name, seed, and metric columns
    """
    results = []

    for config in df["config_name"].unique():
        config_data = df[df["config_name"] == config]

        max_step = config_data["step"].max()
        step_threshold = max_step * step_percentile

        final_data = config_data[config_data["step"] >= step_threshold]

        for seed in final_data["seed"].unique():
            seed_data = final_data[final_data["seed"] == seed]
            row = {"config_name": config, "seed": seed}

            for metric in metrics:
                if metric in seed_data.columns:
                    values = seed_data[metric].dropna()
                    if len(values) > 0:
                        row[metric] = values.mean()

            results.append(row)

    return pd.DataFrame(results)


def compute_stats(
    df: pd.DataFrame,
    metrics: List[str],
) -> pd.DataFrame:
    """
    Compute mean ± std for each config and metric.

    Args:
        df: DataFrame from get_final_metrics
        metrics: List of metric column names

    Returns:
        DataFrame with config_name and metric_mean/metric_std columns
    """
    stats = []

    for config in df["config_name"].unique():
        config_data = df[df["config_name"] == config]
        row = {"config_name": config, "n_seeds": len(config_data)}

        for metric in metrics:
            if metric in config_data.columns:
                values = config_data[metric].dropna()
                if len(values) > 0:
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std()

        stats.append(row)

    return pd.DataFrame(stats)


def format_value(mean: float, std: float, bold: bool = False, precision: int = 1) -> str:
    """Format mean ± std for LaTeX."""
    if np.isnan(mean) or np.isnan(std):
        return "---"

    value = f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"
    if bold:
        value = f"\\textbf{{{value}}}"
    return value


def generate_main_results_table(
    df: pd.DataFrame,
    output_path: str,
    metrics: List[Tuple[str, str]] = None,
) -> None:
    """
    Generate main results table: UPI-TRM vs baselines.

    Args:
        df: Aggregated data
        output_path: Path to save .tex file
        metrics: List of (column_name, display_name) tuples
    """
    if metrics is None:
        metrics = [
            ("eval_success_rate", "Success (\\%)"),
            ("eval_mean_score", "Score"),
            ("eval_mean_steps_to_solve", "Steps"),
        ]

    metric_cols = [m[0] for m in metrics]
    final_df = get_final_metrics(df, metric_cols)
    stats_df = compute_stats(final_df, metric_cols)

    # Find best for each metric (for bolding)
    best = {}
    for metric, _ in metrics:
        mean_col = f"{metric}_mean"
        if mean_col in stats_df.columns:
            best_idx = stats_df[mean_col].idxmax()
            best[metric] = stats_df.loc[best_idx, "config_name"]

    # Generate LaTeX
    n_cols = 1 + len(metrics)  # Method + metrics
    col_spec = "l" + "c" * len(metrics)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main Results: Comparison of UPI-TRM with baseline algorithms on 9$\\times$9 Sudoku. Mean $\\pm$ std over 3 seeds.}",
        "\\label{tab:main_results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header
    header = "Method"
    for _, display_name in metrics:
        header += f" & {display_name}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Sort by first metric (descending)
    first_metric = metrics[0][0]
    stats_df = stats_df.sort_values(f"{first_metric}_mean", ascending=False)

    # Data rows
    for _, row in stats_df.iterrows():
        config = row["config_name"]

        # Get method name
        config_lower = config.lower()
        method_name = None
        for key, name in METHOD_NAMES.items():
            if key in config_lower:
                method_name = name
                break
        if method_name is None:
            method_name = config[:20]

        line = method_name
        for metric, _ in metrics:
            mean = row.get(f"{metric}_mean", np.nan)
            std = row.get(f"{metric}_std", np.nan)
            is_best = best.get(metric) == config
            line += " & " + format_value(mean, std, bold=is_best)
        line += " \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")


def generate_ablation_table(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "eval_success_rate",
    metric_name: str = "Success (\\%)",
) -> None:
    """
    Generate ablation study table.

    Args:
        df: Aggregated data
        output_path: Path to save .tex file
        metric: Metric column to report
        metric_name: Display name for metric
    """
    final_df = get_final_metrics(df, [metric])
    stats_df = compute_stats(final_df, [metric])

    # Filter to ablation configs + full model
    ablation_configs = []
    full_config = None

    for config in stats_df["config_name"].unique():
        config_lower = config.lower()
        if "ablation" in config_lower:
            ablation_configs.append(config)
        elif "shaped_theory_exact" in config_lower or "theory_exact" in config_lower:
            if "sparse" not in config_lower:
                full_config = config

    if full_config is None:
        print("Warning: Could not find full model config")
        return

    # Get full model performance
    full_row = stats_df[stats_df["config_name"] == full_config].iloc[0]
    full_mean = full_row[f"{metric}_mean"]
    full_std = full_row[f"{metric}_std"]

    # Generate LaTeX
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation Study: Contribution of each theory-exact feature. " +
        f"$\\Delta$ shows change from full model ({full_mean:.1f}\\%).}",
        "\\label{tab:ablations}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Configuration & {metric_name} & $\\Delta$ \\\\",
        "\\midrule",
    ]

    # Full model first
    lines.append(f"\\textbf{{Full Model (All Features)}} & " +
                f"\\textbf{{{full_mean:.1f} $\\pm$ {full_std:.1f}}} & --- \\\\")
    lines.append("\\midrule")

    # Ablations sorted by delta
    ablation_data = []
    for config in ablation_configs:
        row = stats_df[stats_df["config_name"] == config]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        mean = row[f"{metric}_mean"]
        std = row[f"{metric}_std"]
        delta = mean - full_mean

        # Get label
        config_lower = config.lower()
        label = None
        for key, name in ABLATION_LABELS.items():
            if key in config_lower:
                label = name
                break
        if label is None:
            label = config[:25]

        ablation_data.append((label, mean, std, delta))

    # Sort by delta (most negative first = biggest impact)
    ablation_data.sort(key=lambda x: x[3])

    for label, mean, std, delta in ablation_data:
        delta_str = f"{delta:+.1f}" if not np.isnan(delta) else "---"
        if delta < -5:
            # Highlight large drops
            delta_str = f"\\textcolor{{red}}{{{delta_str}}}"
        lines.append(f"{label} & {mean:.1f} $\\pm$ {std:.1f} & {delta_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")


def generate_theory_metrics_table(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate table showing theory metrics (Lipschitz, contraction, etc.).

    Args:
        df: Aggregated data
        output_path: Path to save .tex file
    """
    theory_metrics = [
        ("theory_hat_Lz", "$\\hat{L}_z$"),
        ("theory_hat_Cz", "$\\hat{C}_z$"),
        ("theory_hat_Lv", "$\\hat{L}_v$"),
        ("theory_bellman_residual", "Bellman Res."),
        ("theory_unrolling_term", "Unroll Bias"),
    ]

    metric_cols = [m[0] for m in theory_metrics]
    final_df = get_final_metrics(df, metric_cols)
    stats_df = compute_stats(final_df, metric_cols)

    # Filter to configs with theory metrics
    has_metrics = stats_df[[f"{m[0]}_mean" for m in theory_metrics if f"{m[0]}_mean" in stats_df.columns]]
    if has_metrics.empty or has_metrics.isna().all().all():
        print("No theory metrics found in data")
        return

    valid_rows = stats_df.dropna(subset=[f"{m[0]}_mean" for m in theory_metrics[:1]], how="all")

    n_cols = 1 + len(theory_metrics)
    col_spec = "l" + "c" * len(theory_metrics)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Theory Metrics: Lipschitz constants and approximation errors. " +
        "$\\hat{L}_z < 1$ required for contraction.}",
        "\\label{tab:theory_metrics}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header
    header = "Configuration"
    for _, display_name in theory_metrics:
        header += f" & {display_name}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for _, row in valid_rows.iterrows():
        config = row["config_name"]

        # Get method name
        config_lower = config.lower()
        method_name = None
        for key, name in METHOD_NAMES.items():
            if key in config_lower:
                method_name = name
                break
        if method_name is None:
            method_name = config[:20]

        line = method_name
        for metric, _ in theory_metrics:
            mean = row.get(f"{metric}_mean", np.nan)
            std = row.get(f"{metric}_std", np.nan)

            if np.isnan(mean):
                line += " & ---"
            else:
                # Use 2 decimal places for theory metrics
                value = format_value(mean, std, precision=2)
                # Highlight if L_z >= 1 (bad)
                if "Lz" in metric and mean >= 1.0:
                    value = f"\\textcolor{{red}}{{{value}}}"
                line += f" & {value}"
        line += " \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for paper")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/summary.parquet",
        help="Input parquet file from aggregate_runs.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/tables",
        help="Output directory for .tex files",
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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    generate_main_results_table(
        df,
        str(output_dir / "main_results.tex"),
    )

    generate_ablation_table(
        df,
        str(output_dir / "ablations.tex"),
    )

    generate_theory_metrics_table(
        df,
        str(output_dir / "theory_metrics.tex"),
    )

    print(f"\nAll tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
