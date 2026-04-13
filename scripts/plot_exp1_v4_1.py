#!/usr/bin/env python3
"""
Plot Experiment 1 v4.1 Results with Reviewer-Defensible Formatting.

Fixes:
- Argmax agreement uses Wilson 95% CI (bars always in [0,1])
- Explicit MISMATCH vs INCREMENTAL labels
- Proper z_norm and saturation diagnostics
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Wilson Score CI (for argmax agreement)
# =============================================================================

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0

    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


# =============================================================================
# Data Loading
# =============================================================================

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


def aggregate_mismatch_data(
    base_dir: Path,
    seeds: List[int],
    n_train: int,
    batch: str,
) -> Dict[str, Dict[int, Dict]]:
    """
    Aggregate MISMATCH DRIFT data: Δ(n_train, n2) only.

    Returns:
        model -> n2 -> {metric: {"mean", "std", "values", "ci_lower", "ci_upper"}}
    """
    data = {"model_a": {}, "model_b": {}}

    for model in ["model_a", "model_b"]:
        raw_by_n2 = {}  # n2 -> metric -> list of values

        for seed in seeds:
            csv_path = base_dir / f"seed{seed}" / f"{model}_{batch}_per_state.csv"
            if not csv_path.exists():
                continue

            rows = load_per_state_csv(str(csv_path))
            for row in rows:
                n1 = int(row.get("n1", 0))
                n2 = int(row.get("n2", 0))

                # MISMATCH: only include pairs from n_train
                if n1 != n_train:
                    continue

                if n2 not in raw_by_n2:
                    raw_by_n2[n2] = {
                        "delta_V": [], "delta_pi": [], "delta_z": [],
                        "argmax_agree": []
                    }

                for metric in raw_by_n2[n2]:
                    if metric in row:
                        raw_by_n2[n2][metric].append(row[metric])

        # Compute stats
        for n2, metrics in raw_by_n2.items():
            data[model][n2] = {}
            for metric, values in metrics.items():
                if not values:
                    data[model][n2][metric] = {"mean": 0, "std": 0, "n": 0}
                    continue

                mean = np.mean(values)
                std = np.std(values)
                n = len(values)

                stats = {"mean": mean, "std": std, "n": n, "values": values}

                # Wilson CI for argmax
                if metric == "argmax_agree":
                    successes = int(sum(values))
                    lower, upper = wilson_ci(successes, n)
                    stats["ci_lower"] = lower
                    stats["ci_upper"] = upper

                data[model][n2][metric] = stats

    return data


def aggregate_incremental_data(
    base_dir: Path,
    seeds: List[int],
    n_train: int,
    batch: str,
) -> Tuple[Dict[str, Dict[int, Dict]], List[Tuple[int, int]]]:
    """
    Aggregate INCREMENTAL DRIFT data: consecutive pairs only.

    Returns:
        (model -> n2 -> {metric: stats}, list of (n1, n2) pairs)
    """
    data = {"model_a": {}, "model_b": {}}

    # First pass: find all depths
    all_n2_values = set()
    for model in ["model_a", "model_b"]:
        for seed in seeds:
            csv_path = base_dir / f"seed{seed}" / f"{model}_{batch}_per_state.csv"
            if not csv_path.exists():
                continue
            rows = load_per_state_csv(str(csv_path))
            for row in rows:
                n2 = int(row.get("n2", 0))
                if n2 > 0:
                    all_n2_values.add(n2)

    all_depths = sorted([n_train] + list(all_n2_values))
    consecutive_pairs = [(all_depths[i], all_depths[i+1]) for i in range(len(all_depths)-1)]
    consecutive_set = set(consecutive_pairs)

    # Second pass: collect data
    for model in ["model_a", "model_b"]:
        raw_by_n2 = {}

        for seed in seeds:
            csv_path = base_dir / f"seed{seed}" / f"{model}_{batch}_per_state.csv"
            if not csv_path.exists():
                continue

            rows = load_per_state_csv(str(csv_path))
            for row in rows:
                n1 = int(row.get("n1", 0))
                n2 = int(row.get("n2", 0))

                # INCREMENTAL: only consecutive pairs
                if (n1, n2) not in consecutive_set:
                    continue

                if n2 not in raw_by_n2:
                    raw_by_n2[n2] = {
                        "delta_V": [], "delta_pi": [], "delta_z": [],
                        "argmax_agree": []
                    }

                for metric in raw_by_n2[n2]:
                    if metric in row:
                        raw_by_n2[n2][metric].append(row[metric])

        # Compute stats
        for n2, metrics in raw_by_n2.items():
            data[model][n2] = {}
            for metric, values in metrics.items():
                if not values:
                    data[model][n2][metric] = {"mean": 0, "std": 0, "n": 0}
                    continue

                mean = np.mean(values)
                std = np.std(values)
                n = len(values)

                stats = {"mean": mean, "std": std, "n": n, "values": values}

                if metric == "argmax_agree":
                    successes = int(sum(values))
                    lower, upper = wilson_ci(successes, n)
                    stats["ci_lower"] = lower
                    stats["ci_upper"] = upper

                data[model][n2][metric] = stats

    return data, consecutive_pairs


def aggregate_radius_sweep(
    base_dir: Path,
    seeds: List[int],
    radii: List[float],
    n_train: int,
    batch: str,
) -> Dict[float, Dict[str, Dict]]:
    """
    Aggregate radius sweep data.

    Returns:
        R -> model -> {metric: stats}
    """
    data = {}

    for R in radii:
        R_str = f"R{int(R)}" if R == int(R) else f"R{R}"
        data[R] = {"model_a": {}, "model_b": {}}

        for model in ["model_a", "model_b"]:
            raw = {"delta_V": [], "delta_pi": [], "delta_z": [], "argmax_agree": []}

            for seed in seeds:
                csv_path = base_dir / f"seed{seed}" / R_str / f"{model}_{batch}_per_state.csv"
                if not csv_path.exists():
                    continue

                rows = load_per_state_csv(str(csv_path))
                for row in rows:
                    n1 = int(row.get("n1", 0))
                    # Only include mismatch drift (from n_train)
                    if n1 != n_train:
                        continue

                    for metric in raw:
                        if metric in row:
                            raw[metric].append(row[metric])

            # Compute stats
            for metric, values in raw.items():
                if not values:
                    data[R][model][metric] = {"mean": 0, "std": 0, "n": 0}
                    continue

                mean = np.mean(values)
                std = np.std(values)
                n = len(values)
                stats = {"mean": mean, "std": std, "n": n}

                if metric == "argmax_agree":
                    successes = int(sum(values))
                    lower, upper = wilson_ci(successes, n)
                    stats["ci_lower"] = lower
                    stats["ci_upper"] = upper

                data[R][model][metric] = stats

    return data


# =============================================================================
# Plotting Functions
# =============================================================================

def set_publication_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_mismatch_metric(
    data_a: Dict[int, Dict],
    data_b: Dict[int, Dict],
    metric: str,
    n_train: int,
    out_path: str,
    ylabel: str,
    title: str,
    use_wilson_ci: bool = False,
    log_scale: bool = True,
):
    """Plot MISMATCH DRIFT metric vs depth multiplier."""
    set_publication_style()

    n2_values = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not n2_values:
        print(f"Warning: No data for {metric}")
        return

    mults = [n2 / n_train for n2 in n2_values]

    means_a = [data_a[n2][metric]["mean"] for n2 in n2_values]
    means_b = [data_b[n2][metric]["mean"] for n2 in n2_values]

    if use_wilson_ci:
        # Use Wilson CI for error bars
        yerr_a_lower = [data_a[n2][metric]["mean"] - data_a[n2][metric].get("ci_lower", 0) for n2 in n2_values]
        yerr_a_upper = [data_a[n2][metric].get("ci_upper", 1) - data_a[n2][metric]["mean"] for n2 in n2_values]
        yerr_b_lower = [data_b[n2][metric]["mean"] - data_b[n2][metric].get("ci_lower", 0) for n2 in n2_values]
        yerr_b_upper = [data_b[n2][metric].get("ci_upper", 1) - data_b[n2][metric]["mean"] for n2 in n2_values]
        yerr_a = [yerr_a_lower, yerr_a_upper]
        yerr_b = [yerr_b_lower, yerr_b_upper]
    else:
        stds_a = [data_a[n2][metric]["std"] for n2 in n2_values]
        stds_b = [data_b[n2][metric]["std"] for n2 in n2_values]
        yerr_a = stds_a
        yerr_b = stds_b

    fig, ax = plt.subplots()

    ax.errorbar(mults, means_a, yerr=yerr_a, label="No Contraction (A')",
                marker='o', capsize=4, color='#E24A33')
    ax.errorbar(mults, means_b, yerr=yerr_b, label="Contraction (B)",
                marker='s', capsize=4, color='#348ABD')

    ax.set_xlabel(r"Depth Multiplier ($n_2 / n_\mathrm{train}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")

    if log_scale and metric in ["delta_V", "delta_pi", "delta_z"]:
        ax.set_yscale("log")

    if use_wilson_ci:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


def plot_incremental_metric(
    data_a: Dict[int, Dict],
    data_b: Dict[int, Dict],
    pairs: List[Tuple[int, int]],
    metric: str,
    out_path: str,
    ylabel: str,
    title: str,
    use_wilson_ci: bool = False,
    log_scale: bool = True,
):
    """Plot INCREMENTAL DRIFT metric vs consecutive pairs."""
    set_publication_style()

    # X-axis: pair labels
    pair_labels = [f"{n1}→{n2}" for (n1, n2) in pairs]
    x = np.arange(len(pairs))

    n2_values = [n2 for (n1, n2) in pairs]

    means_a = [data_a.get(n2, {}).get(metric, {"mean": 0})["mean"] for n2 in n2_values]
    means_b = [data_b.get(n2, {}).get(metric, {"mean": 0})["mean"] for n2 in n2_values]

    if use_wilson_ci:
        yerr_a_lower = []
        yerr_a_upper = []
        yerr_b_lower = []
        yerr_b_upper = []
        for n2 in n2_values:
            stats_a = data_a.get(n2, {}).get(metric, {"mean": 0, "ci_lower": 0, "ci_upper": 1})
            stats_b = data_b.get(n2, {}).get(metric, {"mean": 0, "ci_lower": 0, "ci_upper": 1})
            mean_a = stats_a.get("mean", 0)
            mean_b = stats_b.get("mean", 0)
            ci_lower_a = stats_a.get("ci_lower", mean_a)
            ci_upper_a = stats_a.get("ci_upper", mean_a)
            ci_lower_b = stats_b.get("ci_lower", mean_b)
            ci_upper_b = stats_b.get("ci_upper", mean_b)
            # Ensure non-negative error bars
            yerr_a_lower.append(max(0, mean_a - ci_lower_a))
            yerr_a_upper.append(max(0, ci_upper_a - mean_a))
            yerr_b_lower.append(max(0, mean_b - ci_lower_b))
            yerr_b_upper.append(max(0, ci_upper_b - mean_b))
        yerr_a = [yerr_a_lower, yerr_a_upper]
        yerr_b = [yerr_b_lower, yerr_b_upper]
    else:
        stds_a = [data_a.get(n2, {}).get(metric, {"std": 0})["std"] for n2 in n2_values]
        stds_b = [data_b.get(n2, {}).get(metric, {"std": 0})["std"] for n2 in n2_values]
        yerr_a = stds_a
        yerr_b = stds_b

    fig, ax = plt.subplots()

    width = 0.35
    ax.bar(x - width/2, means_a, width, yerr=yerr_a, label="No Contraction (A')",
           color='#E24A33', capsize=4, alpha=0.8)
    ax.bar(x + width/2, means_b, width, yerr=yerr_b, label="Contraction (B)",
           color='#348ABD', capsize=4, alpha=0.8)

    ax.set_xlabel("Consecutive Depth Pair")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.legend(loc="best")

    if log_scale and metric in ["delta_V", "delta_pi", "delta_z"]:
        ax.set_yscale("log")

    if use_wilson_ci:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


def plot_radius_sweep_metric(
    data: Dict[float, Dict[str, Dict]],
    radii: List[float],
    metric: str,
    out_path: str,
    ylabel: str,
    title: str,
    use_wilson_ci: bool = False,
    log_scale: bool = True,
):
    """Plot radius sweep metric."""
    set_publication_style()

    # X-axis: radius values
    x = np.arange(len(radii))
    labels = [f"R={int(R)}" if R > 0 else "disabled" for R in radii]

    means_a = [data[R]["model_a"].get(metric, {"mean": 0})["mean"] for R in radii]
    means_b = [data[R]["model_b"].get(metric, {"mean": 0})["mean"] for R in radii]

    if use_wilson_ci:
        yerr_a_lower = []
        yerr_a_upper = []
        yerr_b_lower = []
        yerr_b_upper = []
        for R in radii:
            stats_a = data[R]["model_a"].get(metric, {"mean": 0, "ci_lower": 0, "ci_upper": 1})
            stats_b = data[R]["model_b"].get(metric, {"mean": 0, "ci_lower": 0, "ci_upper": 1})
            mean_a = stats_a.get("mean", 0)
            mean_b = stats_b.get("mean", 0)
            ci_lower_a = stats_a.get("ci_lower", mean_a)
            ci_upper_a = stats_a.get("ci_upper", mean_a)
            ci_lower_b = stats_b.get("ci_lower", mean_b)
            ci_upper_b = stats_b.get("ci_upper", mean_b)
            # Ensure non-negative error bars
            yerr_a_lower.append(max(0, mean_a - ci_lower_a))
            yerr_a_upper.append(max(0, ci_upper_a - mean_a))
            yerr_b_lower.append(max(0, mean_b - ci_lower_b))
            yerr_b_upper.append(max(0, ci_upper_b - mean_b))
        yerr_a = [yerr_a_lower, yerr_a_upper]
        yerr_b = [yerr_b_lower, yerr_b_upper]
    else:
        stds_a = [data[R]["model_a"].get(metric, {"std": 0})["std"] for R in radii]
        stds_b = [data[R]["model_b"].get(metric, {"std": 0})["std"] for R in radii]
        yerr_a = stds_a
        yerr_b = stds_b

    fig, ax = plt.subplots()

    width = 0.35
    ax.bar(x - width/2, means_a, width, yerr=yerr_a, label="No Contraction (A')",
           color='#E24A33', capsize=4, alpha=0.8)
    ax.bar(x + width/2, means_b, width, yerr=yerr_b, label="Contraction (B)",
           color='#348ABD', capsize=4, alpha=0.8)

    ax.set_xlabel("Projection Radius")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="best")

    if log_scale and metric in ["delta_V", "delta_pi", "delta_z"]:
        ax.set_yscale("log")

    if use_wilson_ci:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Exp1 v4.1 with Wilson CI")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base results directory")
    parser.add_argument("--out_dir", type=str, default="results/figures/exp1_v4.1",
                        help="Output directory for figures")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50",
                        help="Comma-separated list of seeds")
    parser.add_argument("--radii", type=str, default="10,100,0",
                        help="Comma-separated list of radii")
    parser.add_argument("--n_train", type=int, default=2,
                        help="Training depth")
    parser.add_argument("--batch", type=str, default="b0",
                        help="Batch to plot (b0 or b1)")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    radii = [float(r) for r in args.radii.split(",")]
    base_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Results dir: {base_dir}")
    print(f"[Config] Output dir: {out_dir}")
    print(f"[Config] Seeds: {seeds}")
    print(f"[Config] Batch: {args.batch}")

    # =============================================================================
    # MISMATCH DRIFT plots
    # =============================================================================
    print("\n=== MISMATCH DRIFT plots ===")
    mismatch_data = aggregate_mismatch_data(base_dir, seeds, args.n_train, args.batch)

    for metric, ylabel, log in [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_pi", r"$\Delta_\pi$ (Policy KL)", True),
        ("delta_z", r"$\Delta_z$ (Latent Drift)", True),
    ]:
        plot_mismatch_metric(
            mismatch_data["model_a"], mismatch_data["model_b"],
            metric, args.n_train,
            str(out_dir / f"mismatch_{metric}_{args.batch}.pdf"),
            ylabel=ylabel,
            title=f"MISMATCH DRIFT: {metric} ({args.batch.upper()})",
            log_scale=log,
        )

    # Argmax with Wilson CI
    plot_mismatch_metric(
        mismatch_data["model_a"], mismatch_data["model_b"],
        "argmax_agree", args.n_train,
        str(out_dir / f"mismatch_argmax_{args.batch}.pdf"),
        ylabel="Argmax Agreement Rate",
        title=f"MISMATCH DRIFT: Action Consistency ({args.batch.upper()})",
        use_wilson_ci=True,
        log_scale=False,
    )

    # =============================================================================
    # INCREMENTAL DRIFT plots
    # =============================================================================
    print("\n=== INCREMENTAL DRIFT plots ===")
    incr_data, pairs = aggregate_incremental_data(base_dir, seeds, args.n_train, args.batch)

    for metric, ylabel, log in [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_pi", r"$\Delta_\pi$ (Policy KL)", True),
        ("delta_z", r"$\Delta_z$ (Latent Drift)", True),
    ]:
        plot_incremental_metric(
            incr_data["model_a"], incr_data["model_b"], pairs,
            metric,
            str(out_dir / f"incremental_{metric}_{args.batch}.pdf"),
            ylabel=ylabel,
            title=f"INCREMENTAL DRIFT: {metric} ({args.batch.upper()})",
            log_scale=log,
        )

    # Argmax with Wilson CI
    plot_incremental_metric(
        incr_data["model_a"], incr_data["model_b"], pairs,
        "argmax_agree",
        str(out_dir / f"incremental_argmax_{args.batch}.pdf"),
        ylabel="Argmax Agreement Rate",
        title=f"INCREMENTAL DRIFT: Action Consistency ({args.batch.upper()})",
        use_wilson_ci=True,
        log_scale=False,
    )

    # =============================================================================
    # RADIUS SWEEP plots
    # =============================================================================
    print("\n=== RADIUS SWEEP plots ===")
    radius_data = aggregate_radius_sweep(base_dir, seeds, radii, args.n_train, args.batch)

    for metric, ylabel, log in [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_z", r"$\Delta_z$ (Latent Drift)", True),
    ]:
        plot_radius_sweep_metric(
            radius_data, radii,
            metric,
            str(out_dir / f"radius_sweep_{metric}_{args.batch}.pdf"),
            ylabel=ylabel,
            title=f"Radius Sweep: {metric} ({args.batch.upper()})",
            log_scale=log,
        )

    # Argmax with Wilson CI
    plot_radius_sweep_metric(
        radius_data, radii,
        "argmax_agree",
        str(out_dir / f"radius_sweep_argmax_{args.batch}.pdf"),
        ylabel="Argmax Agreement Rate",
        title=f"Radius Sweep: Action Consistency ({args.batch.upper()})",
        use_wilson_ci=True,
        log_scale=False,
    )

    print(f"\n[Done] All figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
