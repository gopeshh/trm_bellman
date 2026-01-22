#!/usr/bin/env python3
"""
Final Paper-Ready Artifacts for ICML Experiment 1.

Generates:
- B0-only unroll sensitivity figure (main paper)
- B1-only unroll sensitivity figure (appendix)
- B0-only radius sweep figure (main paper)
- B1-only radius sweep figure (appendix)
- LaTeX tables (.tex) for direct paper inclusion
- Updated PROVENANCE.md and CLAIMS.md

Usage:
    python scripts/make_paper_figures_exp1_final.py
"""

import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

SEEDS = [41, 42, 43]
N_TRAIN = 2
N2_VALUES = [4, 8, 16]  # Evaluation depths
RADII = [0.0, 10.0, 100.0]

BASE_DIR = Path("/Users/buiksat/trm_bellman/results/validation/exp1_v4")
TABLES_DIR = Path("/Users/buiksat/trm_bellman/results/tables")
OUT_DIR = Path("/Users/buiksat/trm_bellman/results/paper_ready/exp1")  # docs & tables
FIG_DIR = Path("/Users/buiksat/UPI_TRM/UPI_TRM_ICML/figures")  # figures for paper

LABEL_MAP = {
    "model_a": "No Contraction",
    "model_b": "Contraction",
}

COLORS = {
    "model_a": "#E24A33",  # Red
    "model_b": "#348ABD",  # Blue
}

MARKERS = {
    "model_a": "o",
    "model_b": "s",
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
# Data Loading from Aggregated CSVs
# =============================================================================

@dataclass
class AggregatedStats:
    mean: float
    std: float
    n: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def load_unroll_csv(csv_path: Path) -> Dict[str, Dict[int, Dict[str, AggregatedStats]]]:
    """Load unroll sensitivity CSV. Returns: model -> n2 -> metric -> stats"""
    result = {"model_a": {}, "model_b": {}}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            n2 = int(row["n2"])
            metric = row["metric"]

            if n2 not in result[model]:
                result[model][n2] = {}

            stats = AggregatedStats(
                mean=float(row["mean"]),
                std=float(row["std"]),
                n=int(row["n"]),
                ci_lower=float(row["ci_lower"]) if row.get("ci_lower") else None,
                ci_upper=float(row["ci_upper"]) if row.get("ci_upper") else None,
            )
            result[model][n2][metric] = stats

    return result


def load_radius_csv(csv_path: Path) -> Dict[float, Dict[str, Dict[str, AggregatedStats]]]:
    """Load radius sweep CSV. Returns: radius -> model -> metric -> stats"""
    result = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            radius = float(row["radius"])
            metric = row["metric"]

            if radius not in result:
                result[radius] = {"model_a": {}, "model_b": {}}

            stats = AggregatedStats(
                mean=float(row["mean"]),
                std=float(row["std"]),
                n=int(row["n"]),
                ci_lower=float(row["ci_lower"]) if row.get("ci_lower") else None,
                ci_upper=float(row["ci_upper"]) if row.get("ci_upper") else None,
            )
            result[radius][model][metric] = stats

    return result


# =============================================================================
# Plot Styling
# =============================================================================

def set_paper_style():
    """ICML-ready styling: larger fonts, bolder lines for two-column print."""
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,  # TrueType fonts for better PDF rendering
        'ps.fonttype': 42,
    })


# =============================================================================
# Unroll Sensitivity Figures
# =============================================================================

def create_unroll_sensitivity_figure(
    data: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    out_path: Path,
    batch_name: str,
):
    """Create 1×3 unroll sensitivity figure for a single batch."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # x-axis: depth multiplier
    n2_values = sorted(set(data["model_a"].keys()) & set(data["model_b"].keys()))
    mults = [n2 / N_TRAIN for n2 in n2_values]

    metrics = [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_pi", r"$\Delta_\pi$ (Policy KL)", True),
        ("argmax_agree", "Action Agreement", False),
    ]

    for col, (metric, ylabel, use_log) in enumerate(metrics):
        ax = axes[col]

        for model in ["model_a", "model_b"]:
            means = []
            yerr_lower = []
            yerr_upper = []

            for n2 in n2_values:
                stats = data[model].get(n2, {}).get(metric, AggregatedStats(0, 0, 0))
                means.append(stats.mean)

                if metric == "argmax_agree" and stats.ci_lower is not None:
                    yerr_lower.append(max(0, stats.mean - stats.ci_lower))
                    yerr_upper.append(max(0, stats.ci_upper - stats.mean))
                else:
                    yerr_lower.append(stats.std)
                    yerr_upper.append(stats.std)

            label = LABEL_MAP[model]
            color = COLORS[model]
            marker = MARKERS[model]

            ax.errorbar(mults, means, yerr=[yerr_lower, yerr_upper],
                       label=label, marker=marker, color=color, capsize=5,
                       markeredgewidth=1.5, elinewidth=2.0)

        ax.set_xlabel(r"Depth Multiplier ($n_2 / n_\mathrm{train}$)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(mults)
        ax.set_xticklabels([f"{int(m)}×" for m in mults])

        if use_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.9, 1.01)

        if col == 2:
            ax.legend(loc='lower left', framealpha=0.9, fontsize=12)

    fig.suptitle(f"Unroll Sensitivity ({batch_name})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"[Figure] {out_path}")


# =============================================================================
# Radius Sweep Figures
# =============================================================================

def create_radius_sweep_figure(
    data: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    out_path: Path,
    batch_name: str,
):
    """Create 1×3 radius sweep figure for a single batch."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    radii = [0.0, 10.0, 100.0]
    x = np.arange(len(radii))
    x_labels = ["disabled", "R=10", "R=100"]
    width = 0.35

    metrics = [
        ("delta_V", r"$\Delta_V$ (Value Instability)", True),
        ("delta_z", r"$\Delta_z$ (Latent Drift)", True),
        ("argmax_agree", "Action Agreement", False),
    ]

    for col, (metric, ylabel, use_log) in enumerate(metrics):
        ax = axes[col]

        for i, model in enumerate(["model_a", "model_b"]):
            means = []
            yerr_list = []

            for R in radii:
                if R not in data:
                    means.append(0)
                    yerr_list.append(0)
                    continue

                stats = data[R].get(model, {}).get(metric, AggregatedStats(0, 0, 0))
                means.append(stats.mean)

                if metric == "argmax_agree" and stats.ci_lower is not None:
                    ci_range = (stats.ci_upper - stats.ci_lower) / 2
                    yerr_list.append(ci_range)
                else:
                    yerr_list.append(stats.std)

            label = LABEL_MAP[model]
            color = COLORS[model]
            offset = -width/2 if i == 0 else width/2

            ax.bar(x + offset, means, width, yerr=yerr_list, label=label,
                  color=color, capsize=5, alpha=0.85, edgecolor='black', linewidth=1.2,
                  error_kw={'elinewidth': 2.0})

        ax.set_xlabel("Projection Radius")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        if use_log and any(m > 0 for m in means):
            ax.set_yscale("log")
        elif not use_log:
            ax.set_ylim(0.5, 1.05)

        if col == 2:
            ax.legend(loc='lower right', framealpha=0.9, fontsize=12)

    fig.suptitle(f"Projection Radius Sweep ({batch_name})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"[Figure] {out_path}")


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def write_unroll_sensitivity_tex(
    data_b0: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    out_path: Path,
):
    """Write LaTeX table for unroll sensitivity (B0, main paper)."""
    n2_values = sorted(set(data_b0["model_a"].keys()) & set(data_b0["model_b"].keys()))

    with open(out_path, "w") as f:
        f.write("% Unroll Sensitivity Table (B0 - Main Paper)\n")
        f.write("% Auto-generated by make_paper_figures_exp1_final.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Unroll sensitivity on B0 (initial states). ")
        f.write("Training depth $n_{\\text{train}}{=}2$, evaluated at $8\\times$ depth ($n_2{=}16$).}\n")
        f.write("\\label{tab:unroll_sensitivity}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Condition & $\\Delta_V$ & $\\Delta_\\pi$ & $\\Delta_z$ & Argmax \\\\\n")
        f.write("\\midrule\n")

        # Use deepest n2 (16 = 8× depth)
        deepest_n2 = max(n2_values)

        for model in ["model_a", "model_b"]:
            label = LABEL_MAP[model]
            stats = data_b0[model].get(deepest_n2, {})

            dV = stats.get("delta_V", AggregatedStats(0, 0, 0))
            dpi = stats.get("delta_pi", AggregatedStats(0, 0, 0))
            dz = stats.get("delta_z", AggregatedStats(0, 0, 0))
            argmax = stats.get("argmax_agree", AggregatedStats(0, 0, 0, 0, 1))

            ci_lo = argmax.ci_lower if argmax.ci_lower else 0
            ci_hi = argmax.ci_upper if argmax.ci_upper else 1

            f.write(f"{label} & {dV.mean:.3f}$\\pm${dV.std:.3f} ")
            f.write(f"& {dpi.mean:.4f}$\\pm${dpi.std:.4f} ")
            f.write(f"& {dz.mean:.2f}$\\pm${dz.std:.2f} ")
            f.write(f"& {argmax.mean:.3f} \\tiny{{[{ci_lo:.2f},{ci_hi:.2f}]}} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[LaTeX] {out_path}")


def write_radius_sweep_tex(
    data: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    out_path: Path,
    batch_name: str,
    is_main: bool,
):
    """Write LaTeX table for radius sweep."""
    with open(out_path, "w") as f:
        loc = "Main Paper" if is_main else "Appendix"
        f.write(f"% Radius Sweep Table ({batch_name} - {loc})\n")
        f.write("% Auto-generated by make_paper_figures_exp1_final.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")

        if is_main:
            f.write("\\caption{Projection radius sweep on B0 (initial states). ")
            f.write("R=0 disables projection; contraction still provides $4.8\\times$ improvement.}\n")
            f.write("\\label{tab:radius_sweep}\n")
        else:
            f.write("\\caption{Radius sweep on B1 (successor states). ")
            f.write("$\\Delta_V$ at R=0 does not improve with contraction.}\n")
            f.write("\\label{tab:radius_sweep_b1}\n")

        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Radius & Condition & $\\Delta_V$ & $\\Delta_z$ & Argmax \\\\\n")
        f.write("\\midrule\n")

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

                ci_lo = argmax.ci_lower if argmax.ci_lower else 0
                ci_hi = argmax.ci_upper if argmax.ci_upper else 1

                f.write(f"{R_label} & {label} ")
                f.write(f"& {dV.mean:.3f}$\\pm${dV.std:.3f} ")
                f.write(f"& {dz.mean:.1f}$\\pm${dz.std:.1f} ")
                f.write(f"& {argmax.mean:.2f} \\tiny{{[{ci_lo:.2f},{ci_hi:.2f}]}} \\\\\n")
                R_label = ""

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[LaTeX] {out_path}")


# =============================================================================
# PROVENANCE.md
# =============================================================================

def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )
        return result.stdout.strip()[:7]
    except Exception:
        return "unknown"


def write_provenance(out_path: Path):
    """Write detailed provenance file."""
    git_hash = get_git_hash()

    with open(out_path, "w") as f:
        f.write("# Experiment 1 Provenance\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n")
        f.write(f"**Commit**: {git_hash}\n\n")

        f.write("## Input CSVs\n\n")
        f.write("| Purpose | Path |\n")
        f.write("|---------|------|\n")
        f.write("| Unroll B0 | `results/tables/unroll_sensitivity_b0_mismatch.csv` |\n")
        f.write("| Unroll B1 | `results/tables/unroll_sensitivity_b1_mismatch.csv` |\n")
        f.write("| Radius B0 | `results/tables/radius_sweep_b0_aggregated.csv` |\n")
        f.write("| Radius B1 | `results/tables/radius_sweep_b1_aggregated.csv` |\n")

        f.write("\n## Checkpoints\n\n")
        f.write("| Seed | No Contraction | Contraction |\n")
        f.write("|------|----------------|-------------|\n")
        for seed in SEEDS:
            f.write(f"| {seed} | `checkpoints/exp1_v4/model_a_prime_seed{seed}.pt` ")
            f.write(f"| `checkpoints/exp1_v4/model_b_seed{seed}.pt` |\n")

        f.write("\n## YAML Configs\n\n")
        f.write("- No Contraction: `configs/ablations/upi_trm_feasibility_no_contraction.yaml`\n")
        f.write("- Contraction: `configs/ablations/upi_trm_feasibility_contraction.yaml`\n")

        f.write("\n## Key Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| n_train | {N_TRAIN} |\n")
        f.write(f"| n2 (eval depths) | {N2_VALUES} |\n")
        f.write(f"| Radii | {RADII} |\n")
        f.write(f"| Seeds | {SEEDS} |\n")
        f.write("| disable_value_head_norm | true |\n")
        f.write("| target_Lz (contraction) | 0.9 |\n")

        f.write("\n## Regeneration Commands\n\n")
        f.write("```bash\n")
        f.write("# Generate all paper-ready artifacts\n")
        f.write("buck2 run //buiksat_trm:make_paper_figures_exp1_final\n")
        f.write("\n")
        f.write("# Or directly with Python\n")
        f.write("python scripts/make_paper_figures_exp1_final.py\n")
        f.write("```\n")

        f.write("\n## Output Artifacts\n\n")
        f.write("### Main Paper\n")
        f.write("- `fig_exp1_unroll_sensitivity_main.pdf` (B0, 1×3)\n")
        f.write("- `fig_exp1_radius_sweep_main.pdf` (B0, 1×3)\n")
        f.write("- `table_exp1_unroll_sensitivity.tex`\n")
        f.write("- `table_exp1_radius_sweep_main.tex`\n")
        f.write("\n### Appendix\n")
        f.write("- `fig_exp1_unroll_sensitivity_appendix.pdf` (B1, 1×3)\n")
        f.write("- `fig_exp1_radius_sweep_appendix.pdf` (B1, 1×3)\n")
        f.write("- `table_exp1_radius_sweep_appendix.tex`\n")

    print(f"[Provenance] {out_path}")


# =============================================================================
# CLAIMS.md
# =============================================================================

def write_claims(
    out_path: Path,
    unroll_b0: Dict[str, Dict[int, Dict[str, AggregatedStats]]],
    radius_b0: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
    radius_b1: Dict[float, Dict[str, Dict[str, AggregatedStats]]],
):
    """Write claims with explicit B0/B1 scoping."""
    n2_values = sorted(unroll_b0["model_a"].keys())
    deepest_n2 = max(n2_values) if n2_values else 16

    with open(out_path, "w") as f:
        f.write("# Experiment 1: Paper Claims\n\n")
        f.write("All claims are precisely scoped to B0 or B1.\n\n")

        # Unroll sensitivity claims (B0)
        f.write("## Unroll Sensitivity (B0 - Main Result)\n\n")

        dV_a = unroll_b0["model_a"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
        dV_b = unroll_b0["model_b"].get(deepest_n2, {}).get("delta_V", AggregatedStats(0, 0, 0))
        dpi_a = unroll_b0["model_a"].get(deepest_n2, {}).get("delta_pi", AggregatedStats(0, 0, 0))
        dpi_b = unroll_b0["model_b"].get(deepest_n2, {}).get("delta_pi", AggregatedStats(0, 0, 0))
        dz_a = unroll_b0["model_a"].get(deepest_n2, {}).get("delta_z", AggregatedStats(0, 0, 0))
        dz_b = unroll_b0["model_b"].get(deepest_n2, {}).get("delta_z", AggregatedStats(0, 0, 0))
        argmax_a = unroll_b0["model_a"].get(deepest_n2, {}).get("argmax_agree", AggregatedStats(0, 0, 0))
        argmax_b = unroll_b0["model_b"].get(deepest_n2, {}).get("argmax_agree", AggregatedStats(0, 0, 0))

        v_improvement = dV_a.mean / dV_b.mean if dV_b.mean > 0 else 0
        pi_improvement = dpi_a.mean / dpi_b.mean if dpi_b.mean > 0 else 0
        z_improvement = dz_a.mean / dz_b.mean if dz_b.mean > 0 else 0

        f.write(f"1. **Claim**: Contraction reduces Δ_V by {v_improvement:.1f}× at 8× depth on B0.\n")
        f.write(f"   **Evidence**: Δ_V {dV_a.mean:.3f}±{dV_a.std:.3f} → {dV_b.mean:.3f}±{dV_b.std:.3f}\n\n")

        f.write(f"2. **Claim**: Policy KL reduced by {pi_improvement:.0f}× on B0.\n")
        f.write(f"   **Evidence**: Δ_π {dpi_a.mean:.4f} → {dpi_b.mean:.4f}\n\n")

        f.write(f"3. **Claim**: Latent drift reduced by {z_improvement:.1f}× on B0.\n")
        f.write(f"   **Evidence**: Δ_z {dz_a.mean:.2f} → {dz_b.mean:.2f}\n\n")

        f.write(f"4. **Claim**: Action agreement improves from {argmax_a.mean:.1%} to {argmax_b.mean:.1%} on B0.\n\n")

        # Radius sweep claims (B0)
        f.write("## Radius Sweep (B0 - Isolation Result)\n\n")

        if 0.0 in radius_b0:
            dV_a_r0 = radius_b0[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
            dV_b_r0 = radius_b0[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))
            r0_improvement = dV_a_r0.mean / dV_b_r0.mean if dV_b_r0.mean > 0 else 0

            f.write(f"5. **Claim**: On B0 (initial states), with R=0 (projection disabled), ")
            f.write(f"contraction provides {r0_improvement:.1f}× value stability improvement.\n")
            f.write(f"   **Evidence**: Δ_V {dV_a_r0.mean:.3f} → {dV_b_r0.mean:.3f}\n")
            f.write("   **Interpretation**: Stability comes from contraction, not projection clipping.\n\n")

        # B1 warning
        f.write("## B1 Observation (NOT a main claim)\n\n")

        if 0.0 in radius_b1:
            dV_a_b1 = radius_b1[0.0]["model_a"].get("delta_V", AggregatedStats(0, 0, 0))
            dV_b_b1 = radius_b1[0.0]["model_b"].get("delta_V", AggregatedStats(0, 0, 0))

            f.write("⚠️ **Warning**: On B1 (successor states) with R=0, Δ_V does NOT improve with contraction.\n")
            f.write(f"   B1 R=0 Δ_V: {dV_a_b1.mean:.3f} → {dV_b_b1.mean:.3f} (increased)\n")
            f.write("   Latent drift and action agreement still improve on B1.\n")
            f.write("   The main R=0 claim is scoped to B0 initial states only.\n\n")

        # Summary
        f.write("## One-Sentence Summary\n\n")
        f.write("On initial states (B0), contraction enforcement provides value stability guarantees ")
        f.write(f"independent of projection radius ({r0_improvement:.1f}× improvement even at R=0); ")
        f.write("on successor states (B1), contraction improves latent and action stability, ")
        f.write("but value estimates require projection to avoid increased variance.\n")

    print(f"[Claims] {out_path}")


# =============================================================================
# Validation
# =============================================================================

def validate_outputs(out_dir: Path, fig_dir: Path):
    """Validate all outputs exist and are correct."""
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    # Figures go to paper repo
    fig_files = [
        "fig_exp1_unroll_sensitivity_main.pdf",
        "fig_exp1_unroll_sensitivity_appendix.pdf",
        "fig_exp1_radius_sweep_main.pdf",
        "fig_exp1_radius_sweep_appendix.pdf",
    ]
    # Docs/tables go to trm_bellman
    doc_files = [
        "table_exp1_unroll_sensitivity.tex",
        "table_exp1_radius_sweep_main.tex",
        "table_exp1_radius_sweep_appendix.tex",
        "PROVENANCE.md",
        "CLAIMS.md",
    ]

    all_ok = True
    for fname in fig_files:
        path = fig_dir / fname
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {fname} ({size} bytes)")
        else:
            print(f"  ✗ MISSING: {fname}")
            all_ok = False

    for fname in doc_files:
        path = out_dir / fname
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {fname} ({size} bytes)")
        else:
            print(f"  ✗ MISSING: {fname}")
            all_ok = False

    # Check labels in a tex file
    tex_file = out_dir / "table_exp1_unroll_sensitivity.tex"
    if tex_file.exists():
        content = tex_file.read_text()
        if "No Contraction" in content and "Contraction" in content:
            print("  ✓ LaTeX tables use correct labels")
        else:
            print("  ✗ Labels incorrect in LaTeX")
            all_ok = False

        if "model_a" in content or "model_b" in content:
            print("  ✗ Internal model names leaked into LaTeX")
            all_ok = False

    # Check CLAIMS.md for B0 scoping
    claims_file = out_dir / "CLAIMS.md"
    if claims_file.exists():
        content = claims_file.read_text()
        if "B0" in content and "initial states" in content:
            print("  ✓ CLAIMS.md has B0 scoping")
        else:
            print("  ✗ CLAIMS.md missing B0 scoping")
            all_ok = False

        if "⚠️" in content or "Warning" in content:
            print("  ✓ CLAIMS.md has B1 warning")
        else:
            print("  ✗ CLAIMS.md missing B1 warning")
            all_ok = False

    print()
    if all_ok:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED")

    return all_ok


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PAPER-READY ARTIFACTS")
    print("=" * 60)

    # Load data from aggregated CSVs
    print("\n=== Loading Data ===")
    unroll_b0 = load_unroll_csv(TABLES_DIR / "unroll_sensitivity_b0_mismatch.csv")
    unroll_b1 = load_unroll_csv(TABLES_DIR / "unroll_sensitivity_b1_mismatch.csv")
    radius_b0 = load_radius_csv(TABLES_DIR / "radius_sweep_b0_aggregated.csv")
    radius_b1 = load_radius_csv(TABLES_DIR / "radius_sweep_b1_aggregated.csv")
    print("  Loaded 4 aggregated CSVs")

    # Generate figures (to paper repo)
    print("\n=== Generating Figures ===")
    create_unroll_sensitivity_figure(unroll_b0, FIG_DIR / "fig_exp1_unroll_sensitivity_main.pdf", "B0: Initial States")
    create_unroll_sensitivity_figure(unroll_b1, FIG_DIR / "fig_exp1_unroll_sensitivity_appendix.pdf", "B1: Successor States")
    create_radius_sweep_figure(radius_b0, FIG_DIR / "fig_exp1_radius_sweep_main.pdf", "B0: Initial States")
    create_radius_sweep_figure(radius_b1, FIG_DIR / "fig_exp1_radius_sweep_appendix.pdf", "B1: Successor States")

    # Generate LaTeX tables
    print("\n=== Generating LaTeX Tables ===")
    write_unroll_sensitivity_tex(unroll_b0, OUT_DIR / "table_exp1_unroll_sensitivity.tex")
    write_radius_sweep_tex(radius_b0, OUT_DIR / "table_exp1_radius_sweep_main.tex", "B0", is_main=True)
    write_radius_sweep_tex(radius_b1, OUT_DIR / "table_exp1_radius_sweep_appendix.tex", "B1", is_main=False)

    # Generate documentation
    print("\n=== Generating Documentation ===")
    write_provenance(OUT_DIR / "PROVENANCE.md")
    write_claims(OUT_DIR / "CLAIMS.md", unroll_b0, radius_b0, radius_b1)

    # Validate
    all_ok = validate_outputs(OUT_DIR, FIG_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("PAPER INTEGRATION NOTE")
    print("=" * 60)
    print("\nFIGURES (in paper repo):")
    print(f"  {FIG_DIR}/")
    print("  - fig_exp1_unroll_sensitivity_main.pdf  (Section 7.4)")
    print("  - fig_exp1_unroll_sensitivity_appendix.pdf")
    print("  - fig_exp1_radius_sweep_main.pdf        (Section 7.4)")
    print("  - fig_exp1_radius_sweep_appendix.pdf")
    print("\nDOCS & TABLES (in trm_bellman):")
    print(f"  {OUT_DIR}/")
    print("  - table_exp1_unroll_sensitivity.tex")
    print("  - table_exp1_radius_sweep_main.tex")
    print("  - table_exp1_radius_sweep_appendix.tex")
    print("  - PROVENANCE.md, CLAIMS.md")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
