#!/usr/bin/env python3
"""
Plot Unroll Sensitivity Results for ICML Experiment 1.

Generates publication-ready plots with error bars across seeds:
1. Δ_V vs depth multiplier
2. Δ_π vs depth multiplier
3. argmax agreement vs depth multiplier
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_summary_csv(csv_path: str) -> Dict[str, List[float]]:
    """Load summary CSV and extract metrics."""
    import csv

    metrics = {
        "delta_V": [],
        "delta_pi": [],
        "delta_z": [],
        "argmax_agree": [],
    }

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in metrics:
                if key in row:
                    metrics[key].append(float(row[key]))

    return metrics


def load_per_state_csv(csv_path: str) -> List[Dict]:
    """Load per-state CSV."""
    import csv

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k not in ['state_id'] else v for k, v in row.items()})

    return rows


def aggregate_across_seeds(
    base_dir: str,
    model_name: str,
    seeds: List[int],
    batch: str = "b0",
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Aggregate metrics across seeds.

    Returns:
        Dict mapping n2 -> {metric: {"mean": x, "std": y}}
    """
    base = Path(base_dir)

    # Collect per-state data across seeds
    all_data: Dict[int, Dict[str, List[float]]] = {}

    for seed in seeds:
        csv_path = base / f"seed{seed}" / f"{model_name}_{batch}_per_state.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue

        rows = load_per_state_csv(str(csv_path))

        for row in rows:
            n2 = int(row.get("n2", 0))
            if n2 == 0:
                continue

            if n2 not in all_data:
                all_data[n2] = {"delta_V": [], "delta_pi": [], "delta_z": [], "argmax_agree": []}

            for metric in ["delta_V", "delta_pi", "delta_z", "argmax_agree"]:
                if metric in row:
                    all_data[n2][metric].append(row[metric])

    # Compute mean ± std for each n2
    result: Dict[int, Dict[str, Dict[str, float]]] = {}
    for n2, metrics in all_data.items():
        result[n2] = {}
        for metric, values in metrics.items():
            if values:
                result[n2][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            else:
                result[n2][metric] = {"mean": 0.0, "std": 0.0}

    return result


def plot_metric_vs_depth(
    data_a: Dict[int, Dict],
    data_b: Dict[int, Dict],
    metric: str,
    n_train: int,
    out_path: str,
    ylabel: str,
    title: Optional[str] = None,
):
    """
    Plot a metric vs depth multiplier for Model A and B.
    """
    # Get common n2 values
    n2_values = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not n2_values:
        print(f"Warning: No common n2 values for {metric}")
        return

    # Compute multipliers
    mults = [n2 / n_train for n2 in n2_values]

    # Extract values
    means_a = [data_a[n2][metric]["mean"] for n2 in n2_values]
    stds_a = [data_a[n2][metric]["std"] for n2 in n2_values]
    means_b = [data_b[n2][metric]["mean"] for n2 in n2_values]
    stds_b = [data_b[n2][metric]["std"] for n2 in n2_values]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot with error bars
    ax.errorbar(mults, means_a, yerr=stds_a, label="No Contraction (A')",
                marker='o', capsize=3, linewidth=2, markersize=6)
    ax.errorbar(mults, means_b, yerr=stds_b, label="Contraction (B)",
                marker='s', capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel("Depth Multiplier (n / n_train)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale for delta metrics if needed
    if metric in ["delta_V", "delta_pi", "delta_z"]:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot unroll sensitivity results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base results directory (e.g., results/validation/exp1_v4)")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50",
                        help="Comma-separated list of seeds")
    parser.add_argument("--n_train", type=int, default=2,
                        help="Training depth")
    parser.add_argument("--out_dir", type=str, default="results/figures/exp1_v4",
                        help="Output directory for figures")
    parser.add_argument("--batch", type=str, default="b0",
                        help="Batch to plot (b0 or b1)")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Plot] Loading data from {args.results_dir}")
    print(f"[Plot] Seeds: {seeds}")

    # Aggregate across seeds
    data_a = aggregate_across_seeds(args.results_dir, "model_a", seeds, args.batch)
    data_b = aggregate_across_seeds(args.results_dir, "model_b", seeds, args.batch)

    if not data_a or not data_b:
        print("Error: Could not load data")
        return 1

    print(f"[Plot] Found depth values: {sorted(data_a.keys())}")

    # Generate plots
    plot_metric_vs_depth(
        data_a, data_b, "delta_V", args.n_train,
        str(out_dir / "unroll_sensitivity_deltaV.pdf"),
        ylabel=r"$\Delta_V$ (Value Instability)",
        title=f"Value Stability vs Unroll Depth ({args.batch.upper()})",
    )

    plot_metric_vs_depth(
        data_a, data_b, "delta_pi", args.n_train,
        str(out_dir / "unroll_sensitivity_deltaPi.pdf"),
        ylabel=r"$\Delta_\pi$ (Policy KL Divergence)",
        title=f"Policy Stability vs Unroll Depth ({args.batch.upper()})",
    )

    plot_metric_vs_depth(
        data_a, data_b, "argmax_agree", args.n_train,
        str(out_dir / "unroll_sensitivity_argmax.pdf"),
        ylabel="Argmax Agreement Rate",
        title=f"Action Consistency vs Unroll Depth ({args.batch.upper()})",
    )

    plot_metric_vs_depth(
        data_a, data_b, "delta_z", args.n_train,
        str(out_dir / "unroll_sensitivity_deltaZ.pdf"),
        ylabel=r"$\Delta_z$ (Latent Drift)",
        title=f"Latent Stability vs Unroll Depth ({args.batch.upper()})",
    )

    print(f"\n[Plot] All figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
