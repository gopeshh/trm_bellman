#!/usr/bin/env python3
"""
Create Main vs Appendix split for ICML Experiment 1.

Main paper: B0-only finite comparison for the disabled-projection condition
Appendix: B1 data (with note about Δ_V behavior)

Usage:
    python scripts/make_paper_figures_exp1_split.py
"""

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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

SEEDS = [41, 42, 43]
N_TRAIN = 2
RADIUS_SWEEP_N2 = 8  # Paper-facing radius sweep uses the fixed 2→8 comparison.
RADII = [0.0, 10.0, 100.0]

LABEL_MAP = {
    "model_a": "No Contraction",
    "model_b": "Contraction",
}

COLORS = {
    "model_a": "#E24A33",  # Red
    "model_b": "#348ABD",  # Blue
}


# =============================================================================
# Wilson Score CI
# =============================================================================

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class AggregatedStats:
    mean: float
    std: float
    n: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def load_per_state_csv(csv_path: Path) -> List[Dict]:
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


def aggregate_radius_sweep(
    base_dir: Path,
    seeds: List[int],
    radii: List[float],
    batch: str,
    n_train: int,
) -> Dict[float, Dict[str, Dict[str, AggregatedStats]]]:
    result = {}

    for R in radii:
        R_str = f"R{int(R)}" if R == int(R) else f"R{R}"
        result[R] = {"model_a": {}, "model_b": {}}

        for model in ["model_a", "model_b"]:
            raw: Dict[str, List[float]] = {
                "delta_V": [], "delta_pi": [], "delta_z": [],
                "argmax_agree": [], "saturated": [],
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

def write_radius_sweep_table_single_batch(
    data: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    out_path: Path,
    batch_name: str,
    is_main: bool,
):
    """Write radius sweep table for a single batch."""
    with open(out_path, "w") as f:
        if is_main:
            f.write("# Experiment 1: Projection Radius Sweep (Main Paper)\n\n")
            f.write("**Batch**: B0 (initial states only)\n\n")
            f.write(
                "**Purpose**: Compare observed finite-sample stability with projection "
                "disabled across the two checkpoint conditions.\n\n"
            )
        else:
            f.write("# Experiment 1: Projection Radius Sweep (Appendix)\n\n")
            f.write("**Batch**: B1 (one-step successor closure)\n\n")
            f.write(
                "**Note**: On B1 with projection disabled (legacy artifact tag "
                "R0), the contraction-oriented checkpoint "
                "has lower observed Δ_z and higher argmax agreement, "
            )
            f.write("but Δ_V shows higher variance and does not improve. ")
            f.write("This statement is limited to the recorded successor-state samples.\n\n")

        f.write("**Configuration**: Both models use value-head spectral norm OFF.\n\n")
        f.write(
            f"**Delta definition**: fixed mismatch Δ(n_train={N_TRAIN}, n₂={RADIUS_SWEEP_N2}) "
            f"({RADIUS_SWEEP_N2 // N_TRAIN}× depth), pooled across all states and seeds.\n\n"
        )

        f.write("| Radius | Condition | Δ_V (mean±std) | Δ_z (mean±std) | Argmax [95% CI] | Sat. |\n")
        f.write("|--------|-----------|----------------|----------------|-----------------|------|\n")

        radii_order = [0.0, 10.0, 100.0]

        for R in radii_order:
            if R not in data:
                continue

            R_label = "disabled" if R == 0 else f"R={int(R)}"

            for model in ["model_a", "model_b"]:
                label = LABEL_MAP[model]
                stats = data[R].get(model, {})

                dV = stats.get("delta_V", AggregatedStats(0, 0, 0))
                dz = stats.get("delta_z", AggregatedStats(0, 0, 0))
                argmax = stats.get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))
                sat = stats.get("saturated", AggregatedStats(0, 0, 0))

                dV_str = f"{dV.mean:.3f}±{dV.std:.3f}"
                dz_str = f"{dz.mean:.2f}±{dz.std:.2f}"
                ci_lo = argmax.ci_lower if argmax.ci_lower is not None else 0.0
                ci_hi = argmax.ci_upper if argmax.ci_upper is not None else 1.0
                argmax_str = f"{argmax.mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"

                if R == 0 or sat.n == 0:
                    sat_str = "N/A"
                else:
                    sat_str = f"{sat.mean:.0%}"

                f.write(f"| {R_label} | {label} | {dV_str} | {dz_str} | {argmax_str} | {sat_str} |\n")
                R_label = ""

        if is_main:
            f.write("\n## Key Finding\n\n")
            if 0.0 in data:
                dV_a_r0 = data[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
                dV_b_r0 = data[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))
                if dV_b_r0.mean > 0:
                    improvement = dV_a_r0.mean / dV_b_r0.mean
                    f.write(
                        f"With projection disabled (legacy artifact tag R0), at "
                        f"fixed n={N_TRAIN}→{RADIUS_SWEEP_N2}, "
                        f"the contraction-oriented checkpoint has **{improvement:.1f}× "
                        "lower observed Δ_V** than the no-contraction checkpoint.\n"
                    )
                    f.write(
                        "In these sampled states and checkpoints, the observed "
                        "difference remains with projection disabled; this finite "
                        "diagnostic does not establish a uniform contraction guarantee.\n"
                    )

    print(f"[Table] {out_path}")


# =============================================================================
# Figure Generation
# =============================================================================

def set_paper_style():
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


def create_radius_sweep_figure_single_batch(
    data: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    out_path: Path,
    batch_name: str,
):
    """Create 1×3 radius sweep figure for a single batch."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    radii = [0.0, 10.0, 100.0]
    x = np.arange(len(radii))
    x_labels = ["disabled", "R=10", "R=100"]

    metrics = [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_z", r"$\Delta_z$ (Latent Drift)", True),
        ("argmax_agree", "Action Agreement Rate", False),
    ]

    width = 0.35

    for col, (metric, ylabel, use_log) in enumerate(metrics):
        ax = axes[col]

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
        ax.set_title(ylabel.split("(")[0].strip())

        if use_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.5, 1.05)

        if col == 2:
            ax.legend(loc='lower right', framealpha=0.9)

    fig.suptitle(f"Projection Radius Sweep ({batch_name})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] {out_path}")


# =============================================================================
# CLAIMS.md Fix
# =============================================================================

def write_fixed_claims(
    out_path: Path,
    data_b0: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    data_b1: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
):
    """Write scoped observations for the legacy-tagged disabled condition."""
    with open(out_path, "w") as f:
        r0_improvement = 0.0
        f.write("# Experiment 1: Paper Claims\n\n")
        f.write("Copy-paste ready for paper text. All claims are precisely scoped.\n\n")

        # Claim 1: Unroll sensitivity (from unroll table, B0)
        f.write("## Unroll Sensitivity (Main Result)\n\n")
        f.write("1. **Observation**: On sampled B0 states at 8× depth, the contraction-oriented checkpoint has 4.1× lower observed Δ_V.\n")
        f.write("   **Evidence**: Table 1, Fig 1. Δ_V at n₂=16: No Contraction = 0.156±0.225, Contraction = 0.038±0.055.\n\n")

        f.write("2. **Observation**: On sampled B0 states, the contraction-oriented checkpoint has 33× lower observed policy KL.\n")
        f.write("   **Evidence**: Table 1. Δ_π: 0.0063 → 0.0002.\n\n")

        f.write("3. **Observation**: Sampled B0 action agreement is 96.0% and 99.0% for the two checkpoint conditions (Wilson 95% CI).\n")
        f.write("   **Evidence**: Table 1. No Contraction: [0.931, 0.977], Contraction: [0.971, 0.997].\n\n")

        f.write("4. **Observation**: On sampled B0 states, the contraction-oriented checkpoint has 3.1× lower observed Δ_z.\n")
        f.write("   **Evidence**: Table 1. Δ_z: 4.16 → 1.33.\n\n")

        # Claim 2: Radius sweep - CAREFULLY SCOPED
        f.write("## Radius Sweep (Isolation of Contraction vs. Projection)\n\n")

        # B0 disabled-projection observation.
        if 0.0 in data_b0:
            dV_a = data_b0[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
            dV_b = data_b0[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))
            dz_a = data_b0[0.0]["model_a"].get("delta_z", AggregatedStats(0, 0, 0))
            dz_b = data_b0[0.0]["model_b"].get("delta_z", AggregatedStats(0, 0, 0))

            improvement = dV_a.mean / dV_b.mean if dV_b.mean > 0 else 0
            r0_improvement = improvement
            z_improvement = dz_a.mean / dz_b.mean if dz_b.mean > 0 else 0

            f.write(
                f"5. **Observation**: On initial states (B0), with projection disabled "
                f"(legacy artifact tag R0), at fixed "
                f"{RADIUS_SWEEP_N2 // N_TRAIN}× mismatch "
                f"(n={N_TRAIN}→{RADIUS_SWEEP_N2}), the contraction-oriented checkpoint "
                f"has {improvement:.1f}× lower observed Δ_V.\n"
            )
            f.write(
                f"   **Evidence**: Table 2 (main). Disabled-condition B0 Δ_V "
                f"(legacy tag R0; pooled over all states and seeds): "
                f"No Contraction = {dV_a.mean:.3f}±{dV_a.std:.3f}, Contraction = {dV_b.mean:.3f}±{dV_b.std:.3f}.\n"
            )
            f.write(
                "   **Interpretation**: In these sampled initial states and checkpoints, "
                "the observed difference remains with projection disabled. This finite "
                "diagnostic does not establish a uniform contraction guarantee.\n\n"
            )

        # B1 disabled-projection observations.
        if 0.0 in data_b1:
            dV_a_b1 = data_b1[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
            dV_b_b1 = data_b1[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))
            dz_a_b1 = data_b1[0.0]["model_a"].get("delta_z", AggregatedStats(0, 0, 0))
            dz_b_b1 = data_b1[0.0]["model_b"].get("delta_z", AggregatedStats(0, 0, 0))
            argmax_a_b1 = data_b1[0.0]["model_a"].get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))
            argmax_b_b1 = data_b1[0.0]["model_b"].get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))

            z_improvement_b1 = dz_a_b1.mean / dz_b_b1.mean if dz_b_b1.mean > 0 else 0

            f.write(
                f"6. **Observation**: On successor states (B1), with projection disabled "
                f"(legacy artifact tag R0), the contraction-oriented checkpoint "
                f"has {z_improvement_b1:.1f}× "
                f"lower observed latent drift and action agreement changes from "
                f"{argmax_a_b1.mean:.1%} to {argmax_b_b1.mean:.1%}.\n"
            )
            f.write(f"   **Evidence**: Table 2 (appendix), disabled condition (legacy tag R0), B1: Δ_z {dz_a_b1.mean:.2f} → {dz_b_b1.mean:.2f}; Argmax {argmax_a_b1.mean:.3f} → {argmax_b_b1.mean:.3f}.\n\n")

            f.write(
                "7. **Observation (NOT a claim)**: On B1 with projection "
                "disabled (legacy artifact tag R0), the "
                "contraction-oriented checkpoint does not have lower observed Δ_V "
            )
            f.write(f"({dV_a_b1.mean:.3f} → {dV_b_b1.mean:.3f}).\n")
            f.write(
                "   In these sampled successor states, projection-disabled evaluation "
                "coincides with higher value-estimate variance. Item 5 is therefore "
                "scoped to the sampled B0 states.\n\n"
            )

        # Finite saturation observations
        f.write("8. **Observation**: The recorded projection-active rate at R=10 is 100%.\n")
        f.write("   **Evidence**: Table 2 reports the sampled pre-projection norms.\n\n")

        f.write("9. **Observation**: The recorded projection-active rate at R=100 is 0%.\n")
        f.write("   **Evidence**: Table 2 reports the sampled pre-projection norms.\n\n")

        # Summary
        f.write("## One-Sentence Summary (for paper)\n\n")
        f.write(
            "On the sampled initial states and checkpoints, the contraction-oriented "
            "condition has lower observed value drift "
        )
        f.write(
            f"with projection disabled ({r0_improvement:.1f}× improvement at fixed "
            f"n={N_TRAIN}→{RADIUS_SWEEP_N2} in the disabled condition); "
        )
        f.write(
            "on the sampled successor states, it has lower observed latent drift and "
            "higher action agreement, while value-estimate variance is higher without "
            "projection. These finite diagnostics establish no uniform guarantee.\n"
        )

    print(f"[Claims] {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # Use absolute paths to ensure files are written to the right location
    base_dir = PROJECT_ROOT / "results/validation/exp1_v4"
    out_dir = PROJECT_ROOT / "results/paper_ready/exp1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Aggregating Radius Sweep Data ===")
    data_b0 = aggregate_radius_sweep(base_dir, SEEDS, RADII, "b0", N_TRAIN)
    data_b1 = aggregate_radius_sweep(base_dir, SEEDS, RADII, "b1", N_TRAIN)

    print("\n=== Creating Main Paper Artifacts (B0 only) ===")
    write_radius_sweep_table_single_batch(
        data_b0,
        out_dir / "table_exp1_radius_sweep_main.md",
        "B0 (initial states)",
        is_main=True,
    )
    create_radius_sweep_figure_single_batch(
        data_b0,
        out_dir / "fig_exp1_radius_sweep_main.pdf",
        "B0: Initial States",
    )

    print("\n=== Creating Appendix Artifacts (B1) ===")
    write_radius_sweep_table_single_batch(
        data_b1,
        out_dir / "table_exp1_radius_sweep_appendix.md",
        "B1 (successors)",
        is_main=False,
    )
    create_radius_sweep_figure_single_batch(
        data_b1,
        out_dir / "fig_exp1_radius_sweep_appendix.pdf",
        "B1: Successor States",
    )

    print("\n=== Fixing CLAIMS.md ===")
    write_fixed_claims(out_dir / "CLAIMS.md", data_b0, data_b1)

    print("\n" + "=" * 60)
    print("MAIN PAPER ARTIFACTS")
    print("=" * 60)
    print("  table_exp1_radius_sweep_main.md")
    print("  fig_exp1_radius_sweep_main.pdf")
    print("  fig_exp1_unroll_sensitivity.pdf (existing, keep as-is)")
    print("  table_exp1_unroll_sensitivity.md (existing, keep as-is)")
    print()
    print("APPENDIX ARTIFACTS")
    print("=" * 60)
    print("  table_exp1_radius_sweep_appendix.md")
    print("  fig_exp1_radius_sweep_appendix.pdf")
    print()
    print("DOCUMENTATION")
    print("=" * 60)
    print("  CLAIMS.md (fixed)")
    print("  PROVENANCE.md (existing)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
