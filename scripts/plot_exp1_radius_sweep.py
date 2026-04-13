#!/usr/bin/env python3
"""
Plot Radius Sweep Results for ICML Experiment 1.

Generates publication-ready plots showing stability across different
projection radii, demonstrating contraction provides stability independent
of clipping.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


PER_STATE_METRIC_ALIASES = {
    "delta_V": ("delta_V",),
    "delta_pi": ("delta_pi",),
    "delta_z": ("delta_z",),
    "argmax_agree": ("argmax_agree",),
    "saturated": ("saturated", "saturation"),
}


def get_metric_value(row: Dict[str, str], metric: str):
    """Load a metric from a per-state CSV row, tolerating legacy names."""
    for key in PER_STATE_METRIC_ALIASES.get(metric, (metric,)):
        if key in row:
            return float(row[key])
    return None


def load_radius_data(
    results_dir: str,
    seeds: List[int],
    radii: List[float],
    model_name: str,
    batch: str = "b0",
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """
    Load metrics for each radius across seeds.

    Returns:
        Dict mapping R -> {metric: {"mean": x, "std": y}}
    """
    base = Path(results_dir)
    result = {}

    for R in radii:
        R_str = f"R{int(R)}" if R == int(R) else f"R{R}"

        all_metrics = {"delta_V": [], "delta_pi": [], "delta_z": [], "argmax_agree": [], "saturated": []}

        for seed in seeds:
            csv_path = base / f"seed{seed}" / R_str / f"{model_name}_{batch}_per_state.csv"
            if not csv_path.exists():
                # Try alternate path structure
                csv_path = base / R_str / f"seed{seed}" / f"{model_name}_{batch}_per_state.csv"
                if not csv_path.exists():
                    print(f"Warning: {csv_path} not found")
                    continue

            import csv
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for metric in all_metrics:
                        val = get_metric_value(row, metric)
                        if val is None:
                            continue
                        if metric == "saturated" and val < 0:
                            # Skip N/A saturation values
                            continue
                        all_metrics[metric].append(val)

        # Compute mean ± std
        result[R] = {}
        for metric, values in all_metrics.items():
            if values:
                result[R][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            else:
                result[R][metric] = {"mean": 0.0, "std": 0.0}

    return result


def plot_radius_sweep(
    data_a: Dict[float, Dict],
    data_b: Dict[float, Dict],
    metric: str,
    out_path: str,
    ylabel: str,
    title: Optional[str] = None,
    show_saturation: bool = True,
):
    """
    Plot metric vs radius for Model A and B.
    """
    radii = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not radii:
        print(f"Warning: No common radii for {metric}")
        return

    # Filter out R=0 for x-axis (use categorical)
    radii_labels = [f"R={int(R)}" if R > 0 else "R=0\n(disabled)" for R in radii]
    x = np.arange(len(radii))

    means_a = [data_a[R][metric]["mean"] for R in radii]
    stds_a = [data_a[R][metric]["std"] for R in radii]
    means_b = [data_b[R][metric]["mean"] for R in radii]
    stds_b = [data_b[R][metric]["std"] for R in radii]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    width = 0.35
    ax.bar(x - width/2, means_a, width, yerr=stds_a, label="No Contraction (A')",
           capsize=3, alpha=0.8)
    ax.bar(x + width/2, means_b, width, yerr=stds_b, label="Contraction (B)",
           capsize=3, alpha=0.8)

    ax.set_xlabel("Projection Radius", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(radii_labels)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add saturation annotation
    if show_saturation and "saturated" in data_a.get(radii[0], {}):
        ax2 = ax.twinx()
        sat_a = [data_a[R].get("saturated", {}).get("mean", 0) * 100 for R in radii]
        sat_b = [data_b[R].get("saturated", {}).get("mean", 0) * 100 for R in radii]
        ax2.plot(x - width/2, sat_a, 'o--', color='gray', alpha=0.5, markersize=4)
        ax2.plot(x + width/2, sat_b, 's--', color='gray', alpha=0.5, markersize=4)
        ax2.set_ylabel("Saturation Rate (%)", fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot radius sweep results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base results directory")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50",
                        help="Comma-separated list of seeds")
    parser.add_argument("--radii", type=str, default="10,30,100,0",
                        help="Comma-separated list of radii")
    parser.add_argument("--out_dir", type=str, default="results/figures/exp1_v4",
                        help="Output directory for figures")
    parser.add_argument("--batch", type=str, default="b0",
                        help="Batch to plot (b0 or b1)")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    radii = [float(r) for r in args.radii.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Plot] Loading data from {args.results_dir}")
    print(f"[Plot] Seeds: {seeds}, Radii: {radii}")

    # Load data for each model
    data_a = load_radius_data(args.results_dir, seeds, radii, "model_a", args.batch)
    data_b = load_radius_data(args.results_dir, seeds, radii, "model_b", args.batch)

    if not data_a or not data_b:
        print("Error: Could not load data")
        return 1

    # Generate plots
    plot_radius_sweep(
        data_a, data_b, "delta_z",
        str(out_dir / "radius_sweep_deltaZ.pdf"),
        ylabel=r"$\Delta_z$ (Latent Drift)",
        title=f"Latent Stability vs Projection Radius ({args.batch.upper()})",
    )

    plot_radius_sweep(
        data_a, data_b, "delta_V",
        str(out_dir / "radius_sweep_deltaV.pdf"),
        ylabel=r"$\Delta_V$ (Value Instability)",
        title=f"Value Stability vs Projection Radius ({args.batch.upper()})",
    )

    plot_radius_sweep(
        data_a, data_b, "argmax_agree",
        str(out_dir / "radius_sweep_argmax.pdf"),
        ylabel="Argmax Agreement Rate",
        title=f"Action Consistency vs Projection Radius ({args.batch.upper()})",
        show_saturation=False,
    )

    print(f"\n[Plot] All figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
