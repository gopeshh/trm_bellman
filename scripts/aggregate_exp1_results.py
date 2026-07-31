#!/usr/bin/env python3
"""
Aggregate Experiment 1 Results Across Seeds.

Produces mean±std summaries and machine-readable CSVs.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


PER_STATE_METRIC_ALIASES = {
    "delta_V": ("delta_V",),
    "delta_pi": ("delta_pi",),
    "delta_z": ("delta_z",),
    "argmax_agree": ("argmax_agree",),
    "saturated": ("saturated", "saturation"),
    "z_pre_norm": ("z_pre_norm",),
    "z_post_norm": ("z_post_norm",),
}


def get_metric_value(row: Dict, metric: str):
    """Load a metric from a per-state row, tolerating older column names."""
    for key in PER_STATE_METRIC_ALIASES.get(metric, (metric,)):
        if key in row:
            return row[key]
    return None


def format_mean_std(stats: Dict[str, float], na_when_empty: bool = False) -> str:
    """Format mean/std for markdown tables."""
    if na_when_empty and stats.get("n", 0) == 0:
        return "N/A"
    return f"{stats['mean']:.4f}±{stats['std']:.4f}"


def load_per_state_csv(csv_path: str) -> List[Dict]:
    """Load per-state CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def aggregate_unroll_sensitivity(
    results_dir: str,
    seeds: List[int],
    out_dir: str,
    batch: str = "b0",
    n_train: int = 2,
) -> Dict:
    """
    Aggregate unroll sensitivity results across seeds.
    """
    base = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect data: model -> n2 -> metric -> list of values
    data = {"model_a": {}, "model_b": {}}

    for model in ["model_a", "model_b"]:
        for seed in seeds:
            csv_path = base / f"seed{seed}" / f"{model}_{batch}_per_state.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                continue

            rows = load_per_state_csv(str(csv_path))
            for row in rows:
                n1 = int(row.get("n1", 0))
                n2 = int(row.get("n2", 0))
                if n1 != n_train or n2 == 0:
                    continue

                if n2 not in data[model]:
                    data[model][n2] = {
                        "delta_V": [], "delta_pi": [], "delta_z": [],
                        "argmax_agree": [], "saturated": [],
                        "z_pre_norm": [], "z_post_norm": [],
                    }

                for metric in data[model][n2]:
                    val = get_metric_value(row, metric)
                    if val is None:
                        continue
                    if metric == "saturated" and val < 0:
                        continue  # Skip N/A
                    data[model][n2][metric].append(val)

    # Compute aggregates
    summary = {"model_a": {}, "model_b": {}}
    for model in ["model_a", "model_b"]:
        for n2, metrics in data[model].items():
            summary[model][n2] = {}
            for metric, values in metrics.items():
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
                    summary[model][n2][metric] = {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "n": len(values),
                    }
                else:
                    summary[model][n2][metric] = {"mean": 0.0, "std": 0.0, "n": 0}

    # Write CSV
    csv_path = out / f"unroll_sensitivity_{batch}_aggregated.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n2", "metric", "mean", "std", "n"])
        for model in ["model_a", "model_b"]:
            for n2 in sorted(summary[model].keys()):
                for metric, stats in summary[model][n2].items():
                    writer.writerow([model, n2, metric, stats["mean"], stats["std"], stats["n"]])

    print(f"[Aggregate] Wrote {csv_path}")

    # Write markdown summary
    md_path = out / f"unroll_sensitivity_{batch}_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Unroll Sensitivity Summary ({batch.upper()})\n\n")
        f.write(f"**Seeds**: {seeds}\n\n")
        f.write(f"**Fixed training depth**: {n_train}\n\n")

        # Get common n2 values
        n2_values = sorted(set(summary["model_a"].keys()) & set(summary["model_b"].keys()))

        for metric in ["delta_V", "delta_pi", "delta_z", "argmax_agree"]:
            f.write(f"## {metric}\n\n")
            f.write("| n2 | Model A (mean±std) | Model B (mean±std) |\n")
            f.write("|----|--------------------|--------------------|")

            for n2 in n2_values:
                stats_a = summary["model_a"].get(n2, {}).get(metric, {"mean": 0, "std": 0})
                stats_b = summary["model_b"].get(n2, {}).get(metric, {"mean": 0, "std": 0})
                f.write(f"\n| {n2} | {stats_a['mean']:.4f}±{stats_a['std']:.4f} | {stats_b['mean']:.4f}±{stats_b['std']:.4f} |")

            f.write("\n\n")

    print(f"[Aggregate] Wrote {md_path}")

    return summary


def aggregate_radius_sweep(
    results_dir: str,
    seeds: List[int],
    radii: List[float],
    out_dir: str,
    batch: str = "b0",
    n_train: int = 2,
    n_eval: int = 8,
) -> Dict:
    """
    Aggregate radius sweep results across seeds.
    """
    base = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect data: model -> R -> metric -> list of values
    data = {"model_a": {}, "model_b": {}}

    for model in ["model_a", "model_b"]:
        for R in radii:
            R_str = f"R{int(R)}" if R == int(R) else f"R{R}"
            data[model][R] = {
                "delta_V": [], "delta_pi": [], "delta_z": [],
                "argmax_agree": [], "saturated": [],
            }

            for seed in seeds:
                # Try different path structures
                csv_paths = [
                    base / f"seed{seed}" / R_str / f"{model}_{batch}_per_state.csv",
                    base / R_str / f"seed{seed}" / f"{model}_{batch}_per_state.csv",
                    base / f"seed{seed}" / f"{R_str}" / f"{model}_{batch}_per_state.csv",
                ]

                csv_path = None
                for p in csv_paths:
                    if p.exists():
                        csv_path = p
                        break

                if not csv_path:
                    continue

                rows = load_per_state_csv(str(csv_path))
                for row in rows:
                    if (
                        int(row.get("n1", 0)) != n_train
                        or int(row.get("n2", 0)) != n_eval
                    ):
                        continue
                    for metric in data[model][R]:
                        val = get_metric_value(row, metric)
                        if val is None:
                            continue
                        if metric == "saturated" and val < 0:
                            continue
                        data[model][R][metric].append(val)

    # Compute aggregates
    summary = {"model_a": {}, "model_b": {}}
    for model in ["model_a", "model_b"]:
        for R, metrics in data[model].items():
            summary[model][R] = {}
            for metric, values in metrics.items():
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
                    summary[model][R][metric] = {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "n": len(values),
                    }
                else:
                    summary[model][R][metric] = {"mean": 0.0, "std": 0.0, "n": 0}

    # Write CSV
    csv_path = out / f"radius_sweep_{batch}_aggregated.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "radius", "metric", "mean", "std", "n"])
        for model in ["model_a", "model_b"]:
            for R in sorted(summary[model].keys()):
                for metric, stats in summary[model][R].items():
                    writer.writerow([model, R, metric, stats["mean"], stats["std"], stats["n"]])

    print(f"[Aggregate] Wrote {csv_path}")

    # Write markdown summary
    md_path = out / f"radius_sweep_{batch}_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Radius Sweep Summary ({batch.upper()})\n\n")
        f.write(f"**Seeds**: {seeds}\n\n")
        f.write(f"**Radii**: {radii}\n\n")
        f.write(f"**Fixed training depth**: {n_train}\n\n")
        f.write(f"**Fixed evaluation depth**: {n_eval}\n\n")

        for metric, label in [
            ("delta_V", "delta_V"),
            ("delta_z", "delta_z"),
            ("argmax_agree", "argmax_agree"),
            ("saturated", "saturation"),
        ]:
            f.write(f"## {label}\n\n")
            f.write("| Radius | Model A (mean±std) | Model B (mean±std) |\n")
            f.write("|--------|--------------------|--------------------|")

            for R in sorted(radii):
                stats_a = summary["model_a"].get(R, {}).get(metric, {"mean": 0, "std": 0})
                stats_b = summary["model_b"].get(R, {}).get(metric, {"mean": 0, "std": 0})
                R_label = f"R={int(R)}" if R > 0 else "disabled"
                show_na = metric == "saturated"
                f.write(
                    f"\n| {R_label} | {format_mean_std(stats_a, na_when_empty=show_na)} | "
                    f"{format_mean_std(stats_b, na_when_empty=show_na)} |"
                )

            f.write("\n\n")

    print(f"[Aggregate] Wrote {md_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment 1 results")
    parser.add_argument("--mode", choices=["unroll", "radius", "both"], default="both",
                        help="Which results to aggregate")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base results directory")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50",
                        help="Comma-separated list of seeds")
    parser.add_argument("--radii", type=str, default="10,100,0",
                        help="Comma-separated list of radii (for radius mode)")
    parser.add_argument("--out_dir", type=str, default="results/plot_data/exp1_v4",
                        help="Output directory")
    parser.add_argument("--batch", type=str, default="b0",
                        help="Batch to aggregate (b0 or b1)")
    parser.add_argument("--n-train", type=int, default=2,
                        help="Fixed n1 depth to aggregate (default: 2)")
    parser.add_argument("--radius-n-eval", type=int, default=8,
                        help="Fixed n2 depth for radius aggregation (default: 8)")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    radii = [float(r) for r in args.radii.split(",")]

    print(f"[Aggregate] Seeds: {seeds}")
    print(f"[Aggregate] Results dir: {args.results_dir}")
    print(f"[Aggregate] Output dir: {args.out_dir}")

    if args.mode in ["unroll", "both"]:
        aggregate_unroll_sensitivity(
            args.results_dir, seeds, args.out_dir, args.batch, args.n_train
        )

    if args.mode in ["radius", "both"]:
        aggregate_radius_sweep(
            args.results_dir,
            seeds,
            radii,
            args.out_dir,
            args.batch,
            args.n_train,
            args.radius_n_eval,
        )

    print("\n[Aggregate] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
