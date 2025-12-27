#!/usr/bin/env python3
"""
Aggregate experiment runs from WandB into a single summary file.

This script downloads run data from WandB, groups by config and seed,
and outputs a parquet file for use in figure/table generation.

Usage:
    python scripts/aggregate_runs.py --project UPI-TRM-ICML-Shaped-Rewards --output artifacts/summary.parquet
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Run: pip install wandb")
    sys.exit(1)


def get_run_history(
    run: "wandb.apis.public.Run",
    metrics: List[str],
    sample_interval: int = 1,
) -> pd.DataFrame:
    """
    Extract history dataframe from a WandB run.

    Args:
        run: WandB run object
        metrics: List of metric names to extract
        sample_interval: Sample every N steps (for large histories)

    Returns:
        DataFrame with columns: step, metric1, metric2, ...
    """
    history = run.scan_history(keys=["_step"] + metrics)
    rows = []
    for i, row in enumerate(history):
        if i % sample_interval == 0:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["step"] + metrics)

    df = pd.DataFrame(rows)
    df = df.rename(columns={"_step": "step"})
    return df


def extract_config_name(run: "wandb.apis.public.Run") -> str:
    """Extract a readable config name from run config."""
    config = run.config

    # Try common naming patterns
    if "config_name" in config:
        return config["config_name"]

    # Build name from key features
    parts = []

    # Algorithm
    algorithm = config.get("algorithm", "upi-trm")
    parts.append(algorithm)

    # Model type
    model_type = config.get("model_type", "trm")
    if model_type != "trm":
        parts.append(model_type)

    # Key features
    if config.get("exact_baseline_summation"):
        parts.append("exact-baseline")
    if config.get("theory_exact_mixture"):
        parts.append("theory-mixture")
    if config.get("enable_contraction"):
        parts.append("contraction")
    if config.get("reward_shaping"):
        parts.append("shaped")
    else:
        parts.append("sparse")

    # K value
    K = config.get("K", 1)
    parts.append(f"K{K}")

    return "_".join(parts)


def extract_seed(run: "wandb.apis.public.Run") -> int:
    """Extract seed from run config or name."""
    if "seed" in run.config:
        return int(run.config["seed"])

    # Try to parse from run name
    name = run.name.lower()
    if "seed" in name:
        import re
        match = re.search(r"seed[_-]?(\d+)", name)
        if match:
            return int(match.group(1))

    return 0  # Default seed


def aggregate_wandb_runs(
    project: str,
    entity: Optional[str] = None,
    output_path: str = "artifacts/summary.parquet",
    metrics: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    sample_interval: int = 1,
) -> pd.DataFrame:
    """
    Download and aggregate all runs from a WandB project.

    Args:
        project: WandB project name
        entity: WandB entity (username or team). If None, uses default.
        output_path: Path to save aggregated data
        metrics: List of metrics to extract. If None, uses defaults.
        filter_tags: Only include runs with these tags
        sample_interval: Sample every N steps to reduce data size

    Returns:
        Aggregated DataFrame
    """
    api = wandb.Api()

    # Build project path
    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project

    print(f"Fetching runs from: {project_path}")

    # Default metrics to extract
    if metrics is None:
        metrics = [
            # Evaluation metrics
            "eval_success_rate",
            "eval_mean_score",
            "eval_mean_steps_to_solve",

            # Training losses
            "train_loss_value",
            "train_loss_policy",
            "train_loss_total",

            # Theory metrics
            "theory_hat_Lz",
            "theory_hat_Cz",
            "theory_hat_Lv",
            "theory_unrolling_term",
            "theory_bellman_residual",

            # Policy metrics
            "policy_entropy",
            "policy_kl_divergence",

            # Episode stats
            "episode_return",
            "episode_length",
            "term_solved",
            "term_stop",
            "term_budget",
        ]

    # Fetch runs
    runs = api.runs(project_path)

    all_data = []

    for run in runs:
        # Skip crashed/unfinished runs
        if run.state not in ["finished", "running"]:
            print(f"  Skipping {run.name} (state: {run.state})")
            continue

        # Filter by tags if specified
        if filter_tags:
            if not any(tag in run.tags for tag in filter_tags):
                continue

        print(f"  Processing: {run.name}")

        # Extract config info
        config_name = extract_config_name(run)
        seed = extract_seed(run)

        # Get history
        history_df = get_run_history(run, metrics, sample_interval)

        if history_df.empty:
            print(f"    Warning: No history data for {run.name}")
            continue

        # Add metadata columns
        history_df["config_name"] = config_name
        history_df["seed"] = seed
        history_df["run_id"] = run.id
        history_df["run_name"] = run.name

        all_data.append(history_df)

    if not all_data:
        print("No runs found!")
        return pd.DataFrame()

    # Combine all runs
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by config, seed, step
    combined_df = combined_df.sort_values(["config_name", "seed", "step"])

    # Save to parquet
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_path, index=False)
    print(f"\nSaved aggregated data to: {output_path}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Configs: {combined_df['config_name'].nunique()}")
    print(f"  Seeds per config: {combined_df.groupby('config_name')['seed'].nunique().to_dict()}")

    return combined_df


def compute_summary_stats(
    df: pd.DataFrame,
    output_path: str = "artifacts/summary_stats.csv",
) -> pd.DataFrame:
    """
    Compute summary statistics per config at final training step.

    Args:
        df: Aggregated data from aggregate_wandb_runs
        output_path: Path to save summary CSV

    Returns:
        Summary DataFrame with mean/std per config
    """
    # Get final step for each run
    final_step = df.groupby(["config_name", "seed"])["step"].max().reset_index()
    final_step = final_step.rename(columns={"step": "final_step"})

    # Merge to get final metrics
    df = df.merge(final_step, on=["config_name", "seed"])
    final_df = df[df["step"] == df["final_step"]]

    # Compute stats per config
    numeric_cols = final_df.select_dtypes(include=["number"]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["step", "seed", "final_step"]]

    stats = []
    for config_name, group in final_df.groupby("config_name"):
        row = {"config_name": config_name, "n_seeds": len(group)}
        for col in numeric_cols:
            values = group[col].dropna()
            if len(values) > 0:
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
        stats.append(row)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_path, index=False)
    print(f"Saved summary stats to: {output_path}")

    return stats_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate WandB runs for paper figures")
    parser.add_argument(
        "--project",
        type=str,
        default="UPI-TRM-ICML-Shaped-Rewards",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity (username or team)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/summary.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Sample every N steps (for large histories)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Filter runs by tags",
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Also compute summary statistics",
    )

    args = parser.parse_args()

    df = aggregate_wandb_runs(
        project=args.project,
        entity=args.entity,
        output_path=args.output,
        filter_tags=args.tags,
        sample_interval=args.sample_interval,
    )

    if args.compute_stats and not df.empty:
        stats_path = args.output.replace(".parquet", "_stats.csv")
        compute_summary_stats(df, stats_path)


if __name__ == "__main__":
    main()
