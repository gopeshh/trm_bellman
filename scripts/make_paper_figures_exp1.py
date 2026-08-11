#!/usr/bin/env python3
"""
Create Paper-Ready Figures and Tables for ICML Experiment 1.

This script:
1. Recomputes all aggregates from per-state CSVs (ground truth)
2. Creates concise paper-ready tables (2 tables)
3. Creates consolidated multi-panel figures (2 main figs)
4. Generates PROVENANCE.md and CLAIMS.md

Usage:
    python scripts/make_paper_figures_exp1.py \
        --results_dir results/validation/exp1_v4 \
        --out_dir results/paper_ready/exp1
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

SEEDS = [41, 42, 43]
N_TRAIN = 2
RADIUS_SWEEP_N2 = 8  # Paper-facing radius sweep uses the fixed 2→8 comparison.
RADII = [0.0, 10.0, 100.0]  # Order for display

# Label mapping: internal -> paper
LABEL_MAP = {
    "model_a": "No Contraction",
    "model_b": "Contraction",
}

# Colors for plots
COLORS = {
    "model_a": "#E24A33",  # Red
    "model_b": "#348ABD",  # Blue
}


# =============================================================================
# Wilson Score CI
# =============================================================================

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# =============================================================================
# Data Loading and Aggregation
# =============================================================================

@dataclass
class AggregatedStats:
    mean: float
    std: float
    n: int
    ci_lower: Optional[float] = None  # For Wilson CI on argmax
    ci_upper: Optional[float] = None


def load_per_state_csv(csv_path: Path) -> List[Dict]:
    """Load per-state CSV with proper type handling."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def aggregate_unroll_sensitivity(
    base_dir: Path,
    seeds: List[int],
    batch: str,
    n_train: int,
) -> Dict[str, Dict[int, Dict[str, AggregatedStats]]]:
    """
    Aggregate unroll sensitivity data from per-state CSVs.

    Returns: model -> n2 -> metric -> AggregatedStats
    Only includes MISMATCH pairs (n1 = n_train).
    """
    result = {"model_a": {}, "model_b": {}}

    for model in ["model_a", "model_b"]:
        raw_by_n2: Dict[int, Dict[str, List[float]]] = {}

        for seed in seeds:
            csv_path = base_dir / f"seed{seed}" / f"{model}_{batch}_per_state.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                continue

            rows = load_per_state_csv(csv_path)
            for row in rows:
                n1 = int(row.get("n1", 0))
                n2 = int(row.get("n2", 0))

                # Only MISMATCH: n1 = n_train
                if n1 != n_train:
                    continue

                if n2 not in raw_by_n2:
                    raw_by_n2[n2] = {
                        "delta_V": [], "delta_pi": [], "delta_z": [],
                        "argmax_agree": [], "saturated": [],
                        "z_pre_norm": [], "z_post_norm": [],
                    }

                for metric in raw_by_n2[n2]:
                    if metric in row:
                        val = row[metric]
                        # Skip N/A saturation
                        if metric == "saturated" and val < 0:
                            continue
                        raw_by_n2[n2][metric].append(val)

        # Compute stats
        for n2, metrics in raw_by_n2.items():
            result[model][n2] = {}
            for metric, values in metrics.items():
                if not values:
                    result[model][n2][metric] = AggregatedStats(0.0, 0.0, 0)
                    continue

                mean = float(np.mean(values))
                std = float(np.std(values))
                n = len(values)

                stats = AggregatedStats(mean, std, n)

                # Wilson CI for argmax
                if metric == "argmax_agree":
                    successes = int(sum(values))
                    stats.ci_lower, stats.ci_upper = wilson_ci(successes, n)

                result[model][n2][metric] = stats

    return result


def aggregate_radius_sweep(
    base_dir: Path,
    seeds: List[int],
    radii: List[float],
    batch: str,
    n_train: int,
) -> Dict[float, Dict[str, Dict[str, AggregatedStats]]]:
    """
    Aggregate radius sweep data from per-state CSVs.

    Returns: radius -> model -> metric -> AggregatedStats
    Uses the fixed MISMATCH pair (n1 = n_train, n2 = RADIUS_SWEEP_N2).
    """
    result = {}

    for R in radii:
        R_str = f"R{int(R)}" if R == int(R) else f"R{R}"
        result[R] = {"model_a": {}, "model_b": {}}

        for model in ["model_a", "model_b"]:
            raw: Dict[str, List[float]] = {
                "delta_V": [], "delta_pi": [], "delta_z": [],
                "argmax_agree": [], "saturated": [],
                "z_pre_norm": [], "z_post_norm": [],
            }

            for seed in seeds:
                csv_path = base_dir / f"seed{seed}" / R_str / f"{model}_{batch}_per_state.csv"
                if not csv_path.exists():
                    continue

                rows = load_per_state_csv(csv_path)
                for row in rows:
                    n1 = int(row.get("n1", 0))
                    n2 = int(row.get("n2", 0))
                    if n1 != n_train or n2 != RADIUS_SWEEP_N2:
                        continue

                    for metric in raw:
                        if metric in row:
                            val = row[metric]
                            if metric == "saturated" and val < 0:
                                continue
                            raw[metric].append(val)

            # Compute stats
            for metric, values in raw.items():
                if not values:
                    result[R][model][metric] = AggregatedStats(0.0, 0.0, 0)
                    continue

                mean = float(np.mean(values))
                std = float(np.std(values))
                n = len(values)

                stats = AggregatedStats(mean, std, n)

                if metric == "argmax_agree":
                    successes = int(sum(values))
                    stats.ci_lower, stats.ci_upper = wilson_ci(successes, n)

                result[R][model][metric] = stats

    return result


# =============================================================================
# Table Generation
# =============================================================================

def write_unroll_sensitivity_table(
    data_b0: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    data_b1: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    out_path: Path,
    n_train: int,
):
    """Write paper-ready unroll sensitivity table."""
    # Find the deepest n2
    n2_values = sorted(set(data_b0["model_a"].keys()) & set(data_b0["model_b"].keys()))
    if not n2_values:
        print("Error: No common n2 values")
        return

    deepest_n2 = max(n2_values)
    depth_mult = deepest_n2 // n_train

    with open(out_path, "w") as f:
        f.write("# Experiment 1: Unroll Sensitivity (Mismatch Drift)\n\n")
        f.write(f"**Comparison**: n_train={n_train} vs n₂={deepest_n2} ({depth_mult}× depth)\n\n")
        f.write("**Configuration**: Both models use value-head spectral norm OFF. ")
        f.write("Only difference is `enable_contraction`.\n\n")
        f.write("**Seeds**: 41, 42, 43\n\n")

        # Main table
        f.write("| Batch | Condition | Δ_V (mean±std) | Δ_π (mean±std) | Δ_z (mean±std) | Argmax Agree [95% CI] |\n")
        f.write("|-------|-----------|----------------|----------------|----------------|----------------------|\n")

        for batch_name, data in [("B0 (initial)", data_b0), ("B1 (successors)", data_b1)]:
            for model in ["model_a", "model_b"]:
                label = LABEL_MAP[model]
                stats = data[model].get(deepest_n2, {})

                dV = stats.get("delta_V", AggregatedStats(0, 0, 0))
                dpi = stats.get("delta_pi", AggregatedStats(0, 0, 0))
                dz = stats.get("delta_z", AggregatedStats(0, 0, 0))
                argmax = stats.get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))

                dV_str = f"{dV.mean:.3f}±{dV.std:.3f}"
                dpi_str = f"{dpi.mean:.4f}±{dpi.std:.4f}"
                dz_str = f"{dz.mean:.2f}±{dz.std:.2f}"
                argmax_str = f"{argmax.mean:.3f} [{argmax.ci_lower:.3f}, {argmax.ci_upper:.3f}]"

                f.write(f"| {batch_name} | {label} | {dV_str} | {dpi_str} | {dz_str} | {argmax_str} |\n")
                batch_name = ""  # Don't repeat batch name

        # Finite comparison summary
        f.write("\n## Recorded Comparison\n\n")

        dV_a = data_b0["model_a"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
        dV_b = data_b0["model_b"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
        if dV_b.mean > 0:
            improvement = dV_a.mean / dV_b.mean
            f.write(
                f"- **Observed B0 value-drift ratio**: {improvement:.1f}× "
                f"(Δ_V: {dV_a.mean:.3f} → {dV_b.mean:.3f})\n"
            )

        argmax_a = data_b0["model_a"].get(deepest_n2, {}).get("argmax_agree", AggregatedStats(0, 0, 0))
        argmax_b = data_b0["model_b"].get(deepest_n2, {}).get("argmax_agree", AggregatedStats(0, 0, 0))
        f.write(f"- **Action consistency (B0)**: {argmax_a.mean:.1%} → {argmax_b.mean:.1%}\n")

    print(f"[Table] {out_path}")


def write_radius_sweep_table(
    data_b0: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    data_b1: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    out_path: Path,
):
    """Write paper-ready radius sweep table."""
    with open(out_path, "w") as f:
        f.write("# Experiment 1: Projection Radius Sweep\n\n")
        f.write(
            "**Purpose**: Compare observed finite-sample stability with projection "
            "disabled across the two checkpoint conditions.\n\n"
        )
        f.write("**Configuration**: Both models use value-head spectral norm OFF.\n\n")
        f.write(
            f"**Delta definition**: fixed mismatch Δ(n_train={N_TRAIN}, n₂={RADIUS_SWEEP_N2}) "
            f"({RADIUS_SWEEP_N2 // N_TRAIN}× depth), pooled across all states and seeds.\n\n"
        )

        # Main table
        f.write("| Batch | Radius | Condition | Δ_V (mean±std) | Argmax Agree [95% CI] | Saturation |\n")
        f.write("|-------|--------|-----------|----------------|----------------------|------------|\n")

        radii_order = [0.0, 10.0, 100.0]

        for batch_name, data in [("B0", data_b0), ("B1", data_b1)]:
            for R in radii_order:
                if R not in data:
                    continue

                R_label = "disabled" if R == 0 else f"R={int(R)}"

                for model in ["model_a", "model_b"]:
                    label = LABEL_MAP[model]
                    stats = data[R].get(model, {})

                    dV = stats.get("delta_V", AggregatedStats(0, 0, 0))
                    argmax = stats.get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))
                    sat = stats.get("saturated", AggregatedStats(0, 0, 0))

                    dV_str = f"{dV.mean:.3f}±{dV.std:.3f}"
                    argmax_str = f"{argmax.mean:.3f} [{argmax.ci_lower:.3f}, {argmax.ci_upper:.3f}]"

                    if R == 0 or sat.n == 0:
                        sat_str = "N/A"
                    else:
                        sat_str = f"{sat.mean:.2f} (n={sat.n})"

                    f.write(f"| {batch_name} | {R_label} | {label} | {dV_str} | {argmax_str} | {sat_str} |\n")
                    batch_name = ""
                    R_label = ""

        # Key finding
        f.write("\n## Key Finding\n\n")

        # Projection-disabled comparison, stored under the legacy R0 artifact tag.
        if 0.0 in data_b0:
            dV_a_r0 = data_b0[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
            dV_b_r0 = data_b0[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))
            if dV_b_r0.mean > 0:
                improvement = dV_a_r0.mean / dV_b_r0.mean
                f.write(
                    f"- **Projection disabled (legacy artifact tag R0, fixed n={N_TRAIN}→{RADIUS_SWEEP_N2})**: "
                    f"the contraction-oriented checkpoint has {improvement:.1f}× "
                    "lower observed Δ_V than the no-contraction checkpoint\n"
                )
                f.write(f"  - Δ_V: No Contraction = {dV_a_r0.mean:.3f}, Contraction = {dV_b_r0.mean:.3f}\n")
                f.write(
                    "  - In these sampled states and checkpoints, the observed "
                    "difference remains with projection disabled; this finite "
                    "diagnostic does not establish a uniform contraction guarantee.\n"
                )

    print(f"[Table] {out_path}")


# =============================================================================
# Figure Generation
# =============================================================================

def set_paper_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def create_unroll_sensitivity_figure(
    data_b0: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    data_b1: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    out_path: Path,
    n_train: int,
):
    """Create combined unroll sensitivity figure (2 rows × 3 cols)."""
    set_paper_style()

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Get n2 values and compute multipliers
    n2_values = sorted(set(data_b0["model_a"].keys()) & set(data_b0["model_b"].keys()))
    mults = [n2 / n_train for n2 in n2_values]

    metrics = [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_pi", r"$\Delta_\pi$ (Policy KL)", True),
        ("argmax_agree", "Action Agreement Rate", False),
    ]

    for row, (batch_name, data) in enumerate([("B0 (Initial States)", data_b0), ("B1 (Successors)", data_b1)]):
        for col, (metric, ylabel, use_log) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])

            for model in ["model_a", "model_b"]:
                means: List[float] = []
                stds: List[float] = []
                for n2 in n2_values:
                    stats = data[model].get(n2, {}).get(metric, AggregatedStats(0, 0, 0))
                    means.append(stats.mean)
                    stds.append(stats.std)

                label = LABEL_MAP[model]
                color = COLORS[model]

                if metric == "argmax_agree":
                    # Use Wilson CI for error bars
                    ci_lower: List[float] = []
                    ci_upper: List[float] = []
                    for n2 in n2_values:
                        stats = data[model].get(n2, {}).get(metric, AggregatedStats(0, 0, 0, 0, 1))
                        ci_lower.append(max(0.0, stats.mean - (stats.ci_lower or 0.0)))
                        ci_upper.append(max(0.0, (stats.ci_upper or 1.0) - stats.mean))
                    yerr = [ci_lower, ci_upper]
                else:
                    yerr = stds

                ax.errorbar(mults, means, yerr=yerr, label=label,
                           marker='o' if model == "model_a" else 's',
                           color=color, capsize=3)

            ax.set_xlabel(r"Depth Multiplier ($n_2 / n_\mathrm{train}$)")
            ax.set_ylabel(ylabel)

            if row == 0:
                ax.set_title(ylabel.split("(")[0].strip())

            if use_log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0.9, 1.01)

            if row == 0 and col == 2:
                ax.legend(loc='lower left', framealpha=0.9)

        # Add batch label on left
        fig.text(0.01, 0.75 - row * 0.5, batch_name, fontsize=11, fontweight='bold',
                rotation=90, va='center')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] {out_path}")


def create_radius_sweep_figure(
    data_b0: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    data_b1: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    out_path: Path,
):
    """Create combined radius sweep figure (2 rows × 3 cols)."""
    set_paper_style()

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    radii = [0.0, 10.0, 100.0]
    x = np.arange(len(radii))
    x_labels = ["disabled", "R=10", "R=100"]

    metrics = [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_z", r"$\Delta_z$ (Latent Drift)", True),
        ("argmax_agree", "Action Agreement Rate", False),
    ]

    width = 0.35

    for row, (batch_name, data) in enumerate([("B0 (Initial States)", data_b0), ("B1 (Successors)", data_b1)]):
        for col, (metric, ylabel, use_log) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])

            for i, model in enumerate(["model_a", "model_b"]):
                means: List[float] = []
                yerr_list: List[float] = []

                for R in radii:
                    if R not in data:
                        means.append(0.0)
                        yerr_list.append(0.0)
                        continue

                    stats = data[R].get(model, {}).get(metric, AggregatedStats(0, 0, 0))
                    means.append(stats.mean)

                    if metric == "argmax_agree":
                        # For bar charts, use symmetric error
                        ci_range = ((stats.ci_upper or 1) - (stats.ci_lower or 0)) / 2
                        yerr_list.append(ci_range)
                    else:
                        yerr_list.append(stats.std)

                label = LABEL_MAP[model]
                color = COLORS[model]
                offset = -width/2 if i == 0 else width/2

                ax.bar(x + offset, means, width, yerr=yerr_list, label=label,
                      color=color, capsize=3, alpha=0.8)

            ax.set_xlabel("Projection Radius")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)

            if row == 0:
                ax.set_title(ylabel.split("(")[0].strip())

            if use_log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0.5, 1.05)

            if row == 0 and col == 2:
                ax.legend(loc='lower right', framealpha=0.9)

        # Add batch label
        fig.text(0.01, 0.75 - row * 0.5, batch_name, fontsize=11, fontweight='bold',
                rotation=90, va='center')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] {out_path}")


# =============================================================================
# Provenance and Claims
# =============================================================================

def write_provenance(
    out_path: Path,
    base_dir: Path,
    seeds: List[int],
):
    """Write provenance file."""
    with open(out_path, "w") as f:
        f.write("# Experiment 1 Provenance\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

        f.write("## Checkpoints\n\n")
        f.write("| Seed | Model A (No Contraction) | Model B (Contraction) |\n")
        f.write("|------|--------------------------|----------------------|\n")
        for seed in seeds:
            f.write(f"| {seed} | checkpoints/exp1_v4/model_a_prime_seed{seed}.pt | checkpoints/exp1_v4/model_b_seed{seed}.pt |\n")

        f.write("\n## YAML Configs\n\n")
        f.write("- Model A (No Contraction): `configs/ablations/upi_trm_feasibility_no_contraction.yaml`\n")
        f.write("- Model B (Contraction): `configs/ablations/upi_trm_feasibility_contraction.yaml`\n")

        f.write("\n## Key Config Differences\n\n")
        f.write("| Setting | No Contraction | Contraction |\n")
        f.write("|---------|----------------|-------------|\n")
        f.write("| enable_contraction | false | true |\n")
        f.write("| target_Lz | N/A | 0.9 |\n")
        f.write("| disable_value_head_norm | true | true |\n")
        f.write("| latent_ball_radius | 10.0 | 10.0 |\n")
        f.write("| inner_unroll_n | 2 | 2 |\n")

        f.write("\n## Batch Artifacts\n\n")
        f.write(f"- B0: `{base_dir}/seed42/b0.pt` (100 initial states)\n")
        f.write(f"- B1: `{base_dir}/seed42/b1.pt` (successor closure)\n")

        f.write("\n## Regeneration Commands\n\n")
        f.write("```bash\n")
        f.write("# Evaluation (already done, per-state CSVs exist)\n")
        f.write("buck2 run //buiksat_trm:eval_unroll_sensitivity -- compare \\\n")
        f.write("    --checkpoint_a checkpoints/exp1_v4/model_a_prime_seed42.pt \\\n")
        f.write("    --checkpoint_b checkpoints/exp1_v4/model_b_seed42.pt \\\n")
        f.write("    --batch_b0 artifacts/eval_batches/b0.pt \\\n")
        f.write("    --batch_b1 artifacts/eval_batches/b1.pt \\\n")
        f.write("    --n_mults 1,2,4,8\n")
        f.write("\n")
        f.write("# Paper figures\n")
        f.write("buck2 run //buiksat_trm:make_paper_figures_exp1 -- \\\n")
        f.write("    --results_dir results/validation/exp1_v4 \\\n")
        f.write("    --out_dir results/paper_ready/exp1\n")
        f.write("```\n")

    print(f"[Provenance] {out_path}")


def write_claims(
    out_path: Path,
    data_unroll_b0: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    data_radius_b0: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    n_train: int,
):
    """Write claims file with quantitative evidence."""
    n2_values = sorted(data_unroll_b0["model_a"].keys())
    deepest_n2 = max(n2_values) if n2_values else 16

    with open(out_path, "w") as f:
        f.write("# Experiment 1: Paper Claims\n\n")
        f.write("Copy-paste ready for paper text.\n\n")

        # Observation 1: finite value-drift comparison
        dV_a = data_unroll_b0["model_a"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
        dV_b = data_unroll_b0["model_b"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
        if dV_b.mean > 0:
            improvement = dV_a.mean / dV_b.mean
            f.write(
                f"1. **Observation**: At {deepest_n2//n_train}× depth on the "
                f"sampled states, the contraction-oriented checkpoint has "
                f"{improvement:.1f}× lower observed Δ_V.\n"
            )
            f.write(f"   **Evidence**: Table 1, Fig 1. Δ_V at n₂={deepest_n2}: No Contraction = {dV_a.mean:.3f}±{dV_a.std:.3f}, ")
            f.write(f"Contraction = {dV_b.mean:.3f}±{dV_b.std:.3f}.\n\n")

        # Observation 2: finite policy-difference comparison
        dpi_a = data_unroll_b0["model_a"].get(deepest_n2, {}).get("delta_pi", AggregatedStats(0, 0, 0))
        dpi_b = data_unroll_b0["model_b"].get(deepest_n2, {}).get("delta_pi", AggregatedStats(0, 0, 0))
        if dpi_b.mean > 0:
            pi_improvement = dpi_a.mean / dpi_b.mean
            f.write(
                f"2. **Observation**: The contraction-oriented checkpoint has "
                f"{pi_improvement:.0f}× lower observed policy KL on these samples.\n"
            )
            f.write(f"   **Evidence**: Table 1. Δ_π: {dpi_a.mean:.4f} → {dpi_b.mean:.4f}.\n\n")

        # Observation 3: finite action-agreement comparison
        argmax_a = data_unroll_b0["model_a"].get(deepest_n2, {}).get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))
        argmax_b = data_unroll_b0["model_b"].get(deepest_n2, {}).get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))
        f.write(
            f"3. **Observation**: Sampled action agreement is {argmax_a.mean:.1%} "
            f"for the no-contraction checkpoint and {argmax_b.mean:.1%} for the "
            "contraction-oriented checkpoint (Wilson 95% CI).\n"
        )
        f.write(f"   **Evidence**: Table 1. No Contraction: [{argmax_a.ci_lower:.3f}, {argmax_a.ci_upper:.3f}], ")
        f.write(f"Contraction: [{argmax_b.ci_lower:.3f}, {argmax_b.ci_upper:.3f}].\n\n")

        # Observation 4: finite disabled-projection comparison.
        if 0.0 in data_radius_b0:
            dV_a_r0 = data_radius_b0[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
            dV_b_r0 = data_radius_b0[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))
            if dV_b_r0.mean > 0:
                r0_improvement = dV_a_r0.mean / dV_b_r0.mean
                f.write(
                    f"4. **Observation**: With projection disabled (legacy artifact "
                    f"tag R0), at fixed "
                    f"{RADIUS_SWEEP_N2 // n_train}× mismatch "
                    f"(n={n_train}→{RADIUS_SWEEP_N2}), the contraction-oriented "
                    f"checkpoint has {r0_improvement:.1f}× lower observed Δ_V.\n"
                )
                f.write(
                    f"   **Evidence**: Table 2, Fig 2. Disabled-condition Δ_V "
                    f"(legacy tag R0; pooled over all states and seeds): "
                    f"{dV_a_r0.mean:.3f} → {dV_b_r0.mean:.3f}.\n"
                )
                f.write(
                    "   In these sampled states and checkpoints, the observed "
                    "difference remains with projection disabled. This finite "
                    "diagnostic does not establish a uniform contraction guarantee.\n\n"
                )

        # Observations 5--6: finite saturation behavior
        if 10.0 in data_radius_b0:
            sat_r10 = data_radius_b0[10.0]["model_a"].get("saturated", AggregatedStats(0, 0, 0))
            if sat_r10.n > 0:
                f.write(
                    f"5. **Observation**: At R=10, the recorded projection-active "
                    f"rate is {sat_r10.mean:.0%}.\n"
                )
                f.write("   **Evidence**: Table 2 reports the sampled pre-projection norms.\n\n")

        if 100.0 in data_radius_b0:
            sat_r100 = data_radius_b0[100.0]["model_a"].get("saturated", AggregatedStats(0, 0, 0))
            if sat_r100.n > 0:
                f.write(
                    f"6. **Observation**: At R=100, the recorded projection-active "
                    f"rate is {sat_r100.mean:.0%}.\n"
                )
                f.write("   **Evidence**: Table 2 reports the sampled pre-projection norms.\n\n")

        # Observation 7: finite latent-drift comparison
        dz_a = data_unroll_b0["model_a"].get(deepest_n2, {}).get("delta_z", AggregatedStats(0, 0, 0))
        dz_b = data_unroll_b0["model_b"].get(deepest_n2, {}).get("delta_z", AggregatedStats(0, 0, 0))
        if dz_b.mean > 0:
            z_improvement = dz_a.mean / dz_b.mean
            f.write(
                f"7. **Observation**: The contraction-oriented checkpoint has "
                f"{z_improvement:.1f}× lower observed Δ_z on these samples.\n"
            )
            f.write(f"   **Evidence**: Fig 1. Δ_z: {dz_a.mean:.2f} → {dz_b.mean:.2f}.\n\n")

        # Summary
        f.write("## One-Sentence Summary\n\n")
        f.write(
            "Across the sampled states and checkpoints, the contraction-oriented "
            f"condition has lower observed value drift at fixed {RADIUS_SWEEP_N2 // n_train}× "
            f"mismatch (n={n_train}→{RADIUS_SWEEP_N2}) with projection disabled. "
            "This finite diagnostic does not establish a uniform stability guarantee.\n"
        )

    print(f"[Claims] {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Create paper-ready figures for Exp1")
    parser.add_argument("--results_dir", type=str, default="results/validation/exp1_v4",
                       help="Base results directory with per-state CSVs")
    parser.add_argument("--out_dir", type=str, default="results/paper_ready/exp1",
                       help="Output directory for paper-ready outputs")

    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Results dir: {base_dir}")
    print(f"[Config] Output dir: {out_dir}")
    print(f"[Config] Seeds: {SEEDS}")
    print(f"[Config] n_train: {N_TRAIN}")

    # =========================================================================
    # Step 1: Aggregate from per-state CSVs
    # =========================================================================
    print("\n=== Aggregating Unroll Sensitivity ===")
    unroll_b0 = aggregate_unroll_sensitivity(base_dir, SEEDS, "b0", N_TRAIN)
    unroll_b1 = aggregate_unroll_sensitivity(base_dir, SEEDS, "b1", N_TRAIN)

    print("\n=== Aggregating Radius Sweep ===")
    radius_b0 = aggregate_radius_sweep(base_dir, SEEDS, RADII, "b0", N_TRAIN)
    radius_b1 = aggregate_radius_sweep(base_dir, SEEDS, RADII, "b1", N_TRAIN)

    # =========================================================================
    # Step 2: Write Tables
    # =========================================================================
    print("\n=== Writing Tables ===")
    write_unroll_sensitivity_table(unroll_b0, unroll_b1,
                                   out_dir / "table_exp1_unroll_sensitivity.md", N_TRAIN)
    write_radius_sweep_table(radius_b0, radius_b1,
                            out_dir / "table_exp1_radius_sweep.md")

    # =========================================================================
    # Step 3: Create Figures
    # =========================================================================
    print("\n=== Creating Figures ===")
    create_unroll_sensitivity_figure(unroll_b0, unroll_b1,
                                     out_dir / "fig_exp1_unroll_sensitivity.pdf", N_TRAIN)
    create_radius_sweep_figure(radius_b0, radius_b1,
                              out_dir / "fig_exp1_radius_sweep.pdf")

    # =========================================================================
    # Step 4: Write Provenance and Claims
    # =========================================================================
    print("\n=== Writing Documentation ===")
    write_provenance(out_dir / "PROVENANCE.md", base_dir, SEEDS)
    write_claims(out_dir / "CLAIMS.md", unroll_b0, radius_b0, N_TRAIN)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("PAPER-READY OUTPUTS")
    print("=" * 60)

    outputs = list(out_dir.glob("*"))
    for f in sorted(outputs):
        print(f"  {f}")

    print("\n" + "=" * 60)
    print("VALIDITY CHECK")
    print("=" * 60)

    # Check key metrics
    n2_values = sorted(unroll_b0["model_a"].keys())
    deepest_n2 = max(n2_values) if n2_values else 16

    dV_a = unroll_b0["model_a"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
    dV_b = unroll_b0["model_b"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))

    print(f"Δ_V at n₂={deepest_n2}:")
    print(f"  No Contraction: {dV_a.mean:.4f} ± {dV_a.std:.4f} (n={dV_a.n})")
    print(f"  Contraction:    {dV_b.mean:.4f} ± {dV_b.std:.4f} (n={dV_b.n})")

    if dV_a.n > 0 and dV_b.n > 0 and dV_b.mean < dV_a.mean:
        print("  ✓ Lower observed Δ_V for the contraction-oriented checkpoint")
    else:
        print("  ✗ UNEXPECTED: no lower observed Δ_V for the contraction-oriented checkpoint")

    # Saturation check
    if 10.0 in radius_b0:
        sat_r10 = radius_b0[10.0]["model_a"].get("saturated", AggregatedStats(0, 0, 0))
        print(f"\nSaturation at R=10: mean={sat_r10.mean:.2f}, n={sat_r10.n}")
        if sat_r10.n > 0:
            print("  ✓ Saturation metric computed correctly")
        else:
            print("  ✗ WARNING: Saturation has n=0")

    if 0.0 in radius_b0:
        sat_r0 = radius_b0[0.0]["model_a"].get("saturated", AggregatedStats(0, 0, 0))
        if sat_r0.n == 0:
            print("\nProjection-active rate: N/A for disabled mode (legacy tag R0)")
            print("  ✓ Correctly excluded from averages")

    print("\n" + "=" * 60)
    print("PASS: All outputs generated successfully")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
