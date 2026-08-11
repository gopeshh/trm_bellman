#!/usr/bin/env python3
"""
Exp2 Final: Generate paper-ready figures, tables, and documentation.

This script produces the canonical Exp2 bundle documenting:
1. Dial failure: spectral-norm targeting does NOT control L_z effectively
2. Finite projection comparison: recorded R=10 versus disabled diagnostics

Reads from existing artifacts (no checkpoint loading):
- results/paper_ready/exp2c/DIAGNOSTICS_exp2c_lite.json (Exp2c evaluation)

Outputs to results/paper_ready/exp2_final/:
- fig_exp2_dial_does_not_control_Lz.pdf
- fig_exp2_projection_is_primary_stabilizer.pdf (legacy artifact filename)
- table_exp2_dial_does_not_control_Lz.tex
- table_exp2_projection_effect.tex
- CLAIMS.md
- PROVENANCE.md
- AUDIT.md (placeholder for audit results)
- PAPER_INSERT_SNIPPET.tex
- summary.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not available, skipping figure generation")


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP2C_JSON = PROJECT_ROOT / "results/paper_ready/exp2c/DIAGNOSTICS_exp2c_lite.json"
OUT_DIR = PROJECT_ROOT / "results/paper_ready/exp2_final"  # docs & tables
FIG_DIR = Path(os.environ.get("UPI_TRM_FIG_DIR", str(PROJECT_ROOT / "figures")))

# ICML paper styling
PAPER_STYLE = {
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,
    'font.weight': 'medium',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
}

# Commit references
COMMITS = {
    "exp2": "a460efd",
    "exp2b": "0d32097",
    "exp2c": "db3de70",
}

# Target L_z values in sweep order
TARGET_LZ_VALUES = [0.9, 0.95, 0.99, 0.999]

# Dial range threshold
DIAL_RANGE_THRESHOLD = 0.08


# =============================================================================
# Data Loading
# =============================================================================

def load_exp2c_data() -> Dict[str, Any]:
    """Load Exp2c evaluation results."""
    if not EXP2C_JSON.exists():
        raise FileNotFoundError(f"Exp2c data not found: {EXP2C_JSON}")

    with open(EXP2C_JSON) as f:
        return json.load(f)


def extract_metrics_by_radius(data: Dict[str, Any]) -> Dict[float, List[Dict]]:
    """Group results by evaluation radius."""
    by_radius = {}
    for result in data["results"]:
        r = result["eval_radius"]
        if r not in by_radius:
            by_radius[r] = []
        by_radius[r].append(result)
    return by_radius


def compute_dial_metrics(results_at_radius: List[Dict]) -> Dict[str, Any]:
    """Compute dial-related metrics for a given radius."""
    # Sort by target_lz
    sorted_results = sorted(results_at_radius, key=lambda x: x["target_lz"])

    target_lz_list = [r["target_lz"] for r in sorted_results]
    L_preproj_means = [r["L_preproj_mean"] for r in sorted_results]
    L_preproj_stds = [r["L_preproj_std"] for r in sorted_results]
    L_postproj_means = [r["L_postproj_mean"] for r in sorted_results]
    L_postproj_stds = [r["L_postproj_std"] for r in sorted_results]
    proj_active_rates = [r["projection_active_rate"] for r in sorted_results]

    # Dial range
    L_preproj_spread = max(L_preproj_means) - min(L_preproj_means)
    dial_range_passes = L_preproj_spread >= DIAL_RANGE_THRESHOLD

    # Check monotonicity
    is_monotonic_decreasing = all(
        L_preproj_means[i] >= L_preproj_means[i+1]
        for i in range(len(L_preproj_means) - 1)
    )
    is_monotonic_increasing = all(
        L_preproj_means[i] <= L_preproj_means[i+1]
        for i in range(len(L_preproj_means) - 1)
    )
    is_monotonic = is_monotonic_decreasing or is_monotonic_increasing

    return {
        "target_lz": target_lz_list,
        "L_preproj_mean": L_preproj_means,
        "L_preproj_std": L_preproj_stds,
        "L_postproj_mean": L_postproj_means,
        "L_postproj_std": L_postproj_stds,
        "projection_active_rate": proj_active_rates,
        "L_preproj_spread": L_preproj_spread,
        "dial_range_passes": dial_range_passes,
        "is_monotonic": is_monotonic,
        "results": sorted_results,
    }


def compute_stability_comparison(by_radius: Dict[float, List[Dict]]) -> Dict[str, Any]:
    """Compare stability metrics between R=10 and R=disabled."""
    r10_results = by_radius.get(10.0, [])
    r_disabled_results = by_radius.get(0.0, [])  # 0.0 means disabled

    if not r10_results or not r_disabled_results:
        return {}

    # Aggregate across target_lz values
    def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
        delta_V_values = [r["delta_V_2_16"] for r in results]
        argmax_agree_values = [r["argmax_agree_2_16"] for r in results]
        delta_pi_values = [r["delta_pi_2_16"] for r in results]
        proj_active_values = [r["projection_active_rate"] for r in results]

        return {
            "delta_V_mean": np.mean(delta_V_values),
            "delta_V_min": np.min(delta_V_values),
            "delta_V_max": np.max(delta_V_values),
            "argmax_agree_mean": np.mean(argmax_agree_values),
            "argmax_agree_min": np.min(argmax_agree_values),
            "argmax_agree_max": np.max(argmax_agree_values),
            "delta_pi_mean": np.mean(delta_pi_values),
            "projection_active_mean": np.mean(proj_active_values),
        }

    return {
        "R10": aggregate_metrics(r10_results),
        "R_disabled": aggregate_metrics(r_disabled_results),
    }


# =============================================================================
# Figure Generation
# =============================================================================

def generate_dial_figure(dial_metrics: Dict[str, Any], out_path: Path):
    """Generate figure showing dial does not control L_z."""
    if not HAS_MATPLOTLIB:
        print(f"[Skip] {out_path.name} (matplotlib not available)")
        return

    # Apply ICML paper styling
    plt.rcParams.update(PAPER_STYLE)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Get metrics for R=10 and R=disabled
    target_lz = dial_metrics["R10"]["target_lz"]
    x = np.arange(len(target_lz))

    # Colors for ICML visibility
    color_disabled = '#2166AC'  # Strong blue
    color_r10 = '#D6604D'       # Strong coral/red

    # Panel A: L_preproj at R=disabled
    ax = axes[0]
    r_disabled = dial_metrics["R_disabled"]
    bars = ax.bar(x, r_disabled["L_preproj_mean"], yerr=r_disabled["L_preproj_std"],
           capsize=5, color=color_disabled, alpha=0.85, edgecolor='black', linewidth=1.5,
           error_kw={'linewidth': 2, 'capthick': 2})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in target_lz], fontweight='bold')
    ax.set_xlabel("Target $L_z$", fontweight='bold')
    ax.set_ylabel("$\\hat{L}_{\\mathrm{pre}}$", fontweight='bold')
    ax.set_title(f"(A) Pre-proj Lipschitz (R=∞)\nSpread: {r_disabled['L_preproj_spread']:.3f}", fontweight='bold')
    ax.axhline(y=0.5, color='#666666', linestyle='--', linewidth=2, alpha=0.7, label='Reference')
    ax.set_ylim(0.3, 0.6)
    ax.tick_params(width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Panel B: L_postproj at R=10
    ax = axes[1]
    r10 = dial_metrics["R10"]
    bars = ax.bar(x, r10["L_postproj_mean"], yerr=r10["L_postproj_std"],
           capsize=5, color=color_r10, alpha=0.85, edgecolor='black', linewidth=1.5,
           error_kw={'linewidth': 2, 'capthick': 2})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in target_lz], fontweight='bold')
    ax.set_xlabel("Target $L_z$", fontweight='bold')
    ax.set_ylabel("$\\hat{L}_{\\mathrm{post}}$", fontweight='bold')
    ax.set_title("(B) Post-proj Lipschitz (R=10)\nSaturates at ~0.23", fontweight='bold')
    ax.set_ylim(0.0, 0.35)
    ax.tick_params(width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Panel C: Projection active rate
    ax = axes[2]
    width = 0.35
    ax.bar(x - width/2, r10["projection_active_rate"], width,
           label='R=10', color=color_r10, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, r_disabled["projection_active_rate"], width,
           label='R=∞', color=color_disabled, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in target_lz], fontweight='bold')
    ax.set_xlabel("Target $L_z$", fontweight='bold')
    ax.set_ylabel("Projection Active Rate", fontweight='bold')
    ax.set_title("(C) Recorded Projection-Active Rate", fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.tick_params(width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"[Output] Saved: {out_path}")


def generate_stability_figure(stability_comp: Dict[str, Any], by_radius: Dict, out_path: Path):
    """Generate the finite R=10 versus disabled diagnostic figure."""
    if not HAS_MATPLOTLIB:
        print(f"[Skip] {out_path.name} (matplotlib not available)")
        return

    # Apply ICML paper styling
    plt.rcParams.update(PAPER_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Get per-target_lz data
    r10_results = sorted(by_radius.get(10.0, []), key=lambda x: x["target_lz"])
    r_disabled_results = sorted(by_radius.get(0.0, []), key=lambda x: x["target_lz"])

    target_lz = [r["target_lz"] for r in r10_results]
    x = np.arange(len(target_lz))
    width = 0.35

    # Colors for ICML visibility
    color_on = '#1B7837'   # Strong green (projection ON)
    color_off = '#C51B7D'  # Strong magenta/red (projection OFF)

    # Panel A: ΔV comparison
    ax = axes[0]
    r10_dv = [r["delta_V_2_16"] for r in r10_results]
    r_disabled_dv = [r["delta_V_2_16"] for r in r_disabled_results]

    ax.bar(x - width/2, r10_dv, width, label='R=10 (proj ON)', color=color_on, 
           alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, r_disabled_dv, width, label='R=∞ (proj OFF)', color=color_off, 
           alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in target_lz], fontweight='bold')
    ax.set_xlabel("Target $L_z$", fontweight='bold')
    ax.set_ylabel("$\\Delta V$ (n=2 → n=16)", fontweight='bold')
    ax.set_title("(A) Recorded $\\Delta V$", fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9, edgecolor='black')
    ax.set_yscale('log')
    ax.tick_params(width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Panel B: Argmax agreement comparison
    ax = axes[1]
    r10_agree = [r["argmax_agree_2_16"] * 100 for r in r10_results]
    r_disabled_agree = [r["argmax_agree_2_16"] * 100 for r in r_disabled_results]

    ax.bar(x - width/2, r10_agree, width, label='R=10 (proj ON)', color=color_on, 
           alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, r_disabled_agree, width, label='R=∞ (proj OFF)', color=color_off, 
           alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in target_lz], fontweight='bold')
    ax.set_xlabel("Target $L_z$", fontweight='bold')
    ax.set_ylabel("Argmax Agreement (%)", fontweight='bold')
    ax.set_title("(B) Recorded Agreement", fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9, edgecolor='black')
    ax.set_ylim(80, 105)
    ax.tick_params(width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3)  # Add horizontal space between panels
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"[Output] Saved: {out_path}")


# =============================================================================
# Table Generation
# =============================================================================

def generate_dial_table(dial_metrics: Dict[str, Any], out_path: Path):
    """Generate LaTeX table for dial results."""
    r10 = dial_metrics["R10"]
    r_disabled = dial_metrics["R_disabled"]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Finite contraction-target sweep. At R=10, the recorded projection-active rate is 100\% and $\hat{L}_{post}$ clusters near 0.23. With projection disabled, recorded $\hat{L}_{pre}$ has spread 0.064 and non-monotonic ordering across the sampled targets.}",
        r"\label{tab:exp2_dial_failure}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Target $L_z$ & $\hat{L}_{pre}$ (R=$\infty$) & $\hat{L}_{post}$ (R=10) & Proj Active (R=10) & Proj Active (R=$\infty$) \\",
        r"\midrule",
    ]

    for i, t_lz in enumerate(r10["target_lz"]):
        L_pre = f"{r_disabled['L_preproj_mean'][i]:.3f}$\\pm${r_disabled['L_preproj_std'][i]:.3f}"
        L_post = f"{r10['L_postproj_mean'][i]:.3f}$\\pm${r10['L_postproj_std'][i]:.3f}"
        proj_r10 = f"{r10['projection_active_rate'][i]*100:.0f}\\%"
        proj_inf = f"{r_disabled['projection_active_rate'][i]*100:.0f}\\%"
        lines.append(f"  {t_lz} & {L_pre} & {L_post} & {proj_r10} & {proj_inf} \\\\")

    lines.extend([
        r"\midrule",
        f"  \\textbf{{Spread}} & \\textbf{{{r_disabled['L_preproj_spread']:.3f}}} & -- & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    print(f"[Output] Saved: {out_path}")


def generate_projection_table(stability_comp: Dict[str, Any], by_radius: Dict, out_path: Path):
    """Generate LaTeX table for projection effect."""
    r10_results = sorted(by_radius.get(10.0, []), key=lambda x: x["target_lz"])
    r_disabled_results = sorted(by_radius.get(0.0, []), key=lambda x: x["target_lz"])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Finite diagnostic comparison of R=10 projection and disabled projection on the recorded sweep.}",
        r"\label{tab:exp2_projection_stabilizer}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Target $L_z$ & \multicolumn{2}{c}{$\Delta V$ (n=2$\to$16)} & \multicolumn{2}{c}{Argmax Agree (\%)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & R=10 & R=$\infty$ & R=10 & R=$\infty$ \\",
        r"\midrule",
    ]

    for i in range(len(r10_results)):
        t_lz = r10_results[i]["target_lz"]
        dv_r10 = r10_results[i]["delta_V_2_16"]
        dv_inf = r_disabled_results[i]["delta_V_2_16"]
        agree_r10 = r10_results[i]["argmax_agree_2_16"] * 100
        agree_inf = r_disabled_results[i]["argmax_agree_2_16"] * 100

        lines.append(f"  {t_lz} & {dv_r10:.2f} & {dv_inf:.2f} & {agree_r10:.1f} & {agree_inf:.1f} \\\\")

    # Summary row
    r10_agg = stability_comp["R10"]
    r_inf_agg = stability_comp["R_disabled"]

    lines.extend([
        r"\midrule",
        f"  \\textbf{{Mean}} & \\textbf{{{r10_agg['delta_V_mean']:.2f}}} & {r_inf_agg['delta_V_mean']:.2f} & \\textbf{{{r10_agg['argmax_agree_mean']*100:.1f}}} & {r_inf_agg['argmax_agree_mean']*100:.1f} \\\\",
        f"  \\textbf{{Range}} & [{r10_agg['delta_V_min']:.1f}, {r10_agg['delta_V_max']:.1f}] & [{r_inf_agg['delta_V_min']:.1f}, {r_inf_agg['delta_V_max']:.1f}] & [{r10_agg['argmax_agree_min']*100:.0f}, {r10_agg['argmax_agree_max']*100:.0f}] & [{r_inf_agg['argmax_agree_min']*100:.0f}, {r_inf_agg['argmax_agree_max']*100:.0f}] \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path.write_text("\n".join(lines))
    print(f"[Output] Saved: {out_path}")


# =============================================================================
# Documentation Generation
# =============================================================================

def generate_claims_md(dial_metrics: Dict, stability_comp: Dict, out_path: Path):
    """Generate CLAIMS.md with scoped claims."""
    r_disabled = dial_metrics["R_disabled"]
    r10_agg = stability_comp["R10"]
    r_inf_agg = stability_comp["R_disabled"]

    content = f"""# Exp2 Final: Finite Diagnostic Observations

**Generated**: {datetime.now().isoformat()}
**Status**: Finite diagnostic comparison

## Observation 1: Recorded Dial Behavior

**Statement**: Across these sampled checkpoints, changing the configured target does not produce a monotonic ordering of the recorded pre-projection estimate.

**Evidence**:
- $\\hat{{L}}_{{preproj}}$ varies only weakly across target $L_z$ values
- Spread: {r_disabled['L_preproj_spread']:.3f} (threshold for "dial works": $\\geq$ 0.08)
- Ordering is non-monotonic: {' → '.join(f'{t}' for t in r_disabled['target_lz'])} yields $\\hat{{L}}_{{preproj}}$ = {' → '.join(f'{v:.3f}' for v in r_disabled['L_preproj_mean'])}
- At R=10: the recorded projection-active rate is 100% and $\\hat{{L}}_{{postproj}}$ clusters near 0.23

**Scope**:
- Evaluated on B0 (initial states)
- target $L_z \\in$ {{{', '.join(str(t) for t in r_disabled['target_lz'])}}}
- 3 seeds per condition
- This finite sweep does not establish a global response to the configured target

## Observation 2: Finite Projection Comparison

**Statement**: On these sampled states and checkpoints, the R=10 condition has lower observed value drift and higher action agreement than the projection-disabled condition.

**Evidence**:
- Recorded $\\Delta V$ (n=2 → n=16): {r_inf_agg['delta_V_min']:.1f}–{r_inf_agg['delta_V_max']:.1f} (R=disabled) and {r10_agg['delta_V_min']:.1f}–{r10_agg['delta_V_max']:.1f} (R=10)
  - Means: {r_inf_agg['delta_V_mean']:.1f} and {r10_agg['delta_V_mean']:.1f} (~{r_inf_agg['delta_V_mean']/r10_agg['delta_V_mean']:.0f}× ratio)
- Recorded argmax agreement: {r_inf_agg['argmax_agree_min']*100:.0f}–{r_inf_agg['argmax_agree_max']*100:.0f}% and {r10_agg['argmax_agree_min']*100:.0f}–{r10_agg['argmax_agree_max']*100:.0f}%
  - Difference of means: {(r10_agg['argmax_agree_mean'] - r_inf_agg['argmax_agree_mean'])*100:.1f} percentage points

**Scope**:
- Evaluated on B0 (initial states)
- Comparison: R=10 (projection ON, 100% active) vs R=disabled (projection OFF, 0% active)
- Mismatch protocol: n_train=2, n_eval=16
- These finite diagnostics establish no causal effect, global contraction, or uniform stability guarantee

## Explicit Non-Claims

1. **NO global monotonicity claim**: The finite sweep records a non-monotonic ordering
2. **NO dial-control claim**: The finite sweep does not establish controllable contraction
3. **NO projection-vs-R monotonicity claim**: We did not sweep R values; only compared R=10 vs R=disabled

## Audit Requirements

This CLAIMS.md must be verified against summary.json by the audit script.
"""

    out_path.write_text(content)
    print(f"[Output] Saved: {out_path}")


def generate_provenance_md(out_path: Path):
    """Generate PROVENANCE.md with full provenance."""
    content = f"""# Exp2 Final: Provenance

**Generated**: {datetime.now().isoformat()}

## Commit References

| Experiment | Commit | Description |
|------------|--------|-------------|
| Exp2 | {COMMITS['exp2']} | Initial contraction sweep |
| Exp2b | {COMMITS['exp2b']} | Projection comparison diagnostics |
| Exp2c | {COMMITS['exp2c']} | Dial unmasking evaluation |

## Checkpoint Paths

All checkpoints from Exp2 contraction sweep:

```
checkpoints/exp2_contraction_sweep/
├── lz_0900/seed{{41,42,43}}/model_step_5000.pt
├── lz_095/seed{{41,42,43}}/model_step_5000.pt
├── lz_099/seed{{41,42,43}}/model_step_5000.pt
└── lz_0999/seed{{41,42,43}}/model_step_5000.pt
```

## Source Data Files

| File | Description |
|------|-------------|
| `results/paper_ready/exp2c/DIAGNOSTICS_exp2c_lite.json` | Exp2c evaluation results (primary source) |

## Regeneration Commands

```bash
# Generate figures and tables
buck2 run //buiksat_trm:make_paper_figures_exp2_final

# Run audit
buck2 run //buiksat_trm:audit_exp2_final_paper_ready
```

## Configuration

| Parameter | Value |
|-----------|-------|
| n_train | 2 |
| n_eval | 16 (deepest mismatch) |
| Eval radii | [10.0, 100.0, disabled (0.0)] |
| Batch | B0 (initial states) |
| Seeds | [41, 42, 43] per target $L_z$ |
| target $L_z$ values | [0.9, 0.95, 0.99, 0.999] |

## Key Configuration Notes

- `disable_value_head_norm: true` - Value-head spectral norm OFF (prevents collapse)
- `use_feasibility_checker: true` - Standard checker
- `episodic_latent: true` - Reset latent at episode boundaries
"""

    out_path.write_text(content)
    print(f"[Output] Saved: {out_path}")


def generate_paper_snippet(out_path: Path):
    """Generate PAPER_INSERT_SNIPPET.tex for appendix."""
    content = r"""\subsection{Experiment 2: Contraction Dial Evaluation}
\label{sec:exp2_contraction}

We investigate whether targeting spectral-norm-based contraction provides a reliable ``stability dial'' for controlling unroll sensitivity.

\paragraph{Setup.}
We train models with target Lipschitz constants $L_z^* \in \{0.9, 0.95, 0.99, 0.999\}$ using spectral normalization on the update function, with 3 seeds per condition. We evaluate achieved Lipschitz constants and unroll sensitivity metrics at two projection settings: $R=10$ (default) and $R=\infty$ (projection disabled).

\paragraph{Finding 1: finite dial behavior.}
Across the sampled checkpoints, the recorded pre-projection estimate has spread 0.064 and non-monotonic ordering across configured targets when projection is disabled (Table~\ref{tab:exp2_dial_failure}). At $R=10$, the recorded projection-active rate is 100\% and the post-projection estimate clusters near 0.23. This finite sweep does not establish a global response to the configured target.

\paragraph{Finding 2: finite projection comparison.}
On the sampled states and checkpoints, observed value drift $\Delta V$ is 7.7--14.8 with projection disabled and 0.5--2.2 at $R=10$ (Table~\ref{tab:exp2_projection_stabilizer}). Recorded argmax agreement is 88--91\% and 97--99\%, respectively. These finite diagnostics establish no causal effect, global contraction, or uniform stability guarantee.

\input{results/paper_ready/exp2_final/table_exp2_dial_does_not_control_Lz}

\input{results/paper_ready/exp2_final/table_exp2_projection_effect}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{results/paper_ready/exp2_final/fig_exp2_dial_does_not_control_Lz.pdf}
    \caption{Finite contraction-target sweep. (A) Recorded pre-projection estimate. (B) Recorded post-projection estimate at $R=10$. (C) Recorded projection-active rate.}
    \label{fig:exp2_dial_failure}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{results/paper_ready/exp2_final/fig_exp2_projection_is_primary_stabilizer.pdf}
    \caption{Finite R=10 versus disabled comparison. (A) Recorded $\Delta V$. (B) Recorded argmax agreement.}
    \label{fig:exp2_projection_stabilizer}
\end{figure}
"""

    out_path.write_text(content)
    print(f"[Output] Saved: {out_path}")


def generate_summary_json(dial_metrics: Dict, stability_comp: Dict, out_path: Path):
    """Generate machine-readable summary for audit."""
    r_disabled = dial_metrics["R_disabled"]
    r10 = dial_metrics["R10"]
    r10_agg = stability_comp["R10"]
    r_inf_agg = stability_comp["R_disabled"]

    summary = {
        "generated": datetime.now().isoformat(),
        "commits": COMMITS,
        "dial_metrics": {
            "target_lz_values": r_disabled["target_lz"],
            "L_preproj_at_R_disabled": {
                "means": r_disabled["L_preproj_mean"],
                "stds": r_disabled["L_preproj_std"],
                "spread": r_disabled["L_preproj_spread"],
                "dial_range_threshold": DIAL_RANGE_THRESHOLD,
                "dial_range_passes": r_disabled["dial_range_passes"],
                "is_monotonic": r_disabled["is_monotonic"],
            },
            "L_postproj_at_R10": {
                "means": r10["L_postproj_mean"],
                "stds": r10["L_postproj_std"],
            },
            "projection_active_rate_R10": r10["projection_active_rate"],
            "projection_active_rate_R_disabled": r_disabled["projection_active_rate"],
        },
        "stability_metrics": {
            "R10": {
                "delta_V_mean": r10_agg["delta_V_mean"],
                "delta_V_range": [r10_agg["delta_V_min"], r10_agg["delta_V_max"]],
                "argmax_agree_mean": r10_agg["argmax_agree_mean"],
                "argmax_agree_range": [r10_agg["argmax_agree_min"], r10_agg["argmax_agree_max"]],
                "projection_active_mean": r10_agg["projection_active_mean"],
            },
            "R_disabled": {
                "delta_V_mean": r_inf_agg["delta_V_mean"],
                "delta_V_range": [r_inf_agg["delta_V_min"], r_inf_agg["delta_V_max"]],
                "argmax_agree_mean": r_inf_agg["argmax_agree_mean"],
                "argmax_agree_range": [r_inf_agg["argmax_agree_min"], r_inf_agg["argmax_agree_max"]],
                "projection_active_mean": r_inf_agg["projection_active_mean"],
            },
        },
        "claims": {
            "dial_failure": True,
            "finite_projection_comparison": True,
            "dial_monotonic": False,
        },
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[Output] Saved: {out_path}")


def generate_audit_placeholder(out_path: Path):
    """Generate placeholder AUDIT.md (filled by audit script)."""
    content = f"""# Exp2 Final: Audit Results

**Generated**: {datetime.now().isoformat()}
**Status**: PENDING (run audit script to populate)

Run the audit script to populate this file:
```bash
buck2 run //buiksat_trm:audit_exp2_final_paper_ready
```
"""

    out_path.write_text(content)
    print(f"[Output] Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EXP2 FINAL: Paper-Ready Artifact Generation")
    print("=" * 60)

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[Load] Loading Exp2c data...")
    exp2c_data = load_exp2c_data()

    # Extract and compute metrics
    print("[Process] Computing metrics...")
    by_radius = extract_metrics_by_radius(exp2c_data)

    dial_metrics = {
        "R10": compute_dial_metrics(by_radius.get(10.0, [])),
        "R_disabled": compute_dial_metrics(by_radius.get(0.0, [])),
    }

    stability_comp = compute_stability_comparison(by_radius)

    # Print summary
    print("\n[Summary]")
    print(f"  L_preproj spread (R=disabled): {dial_metrics['R_disabled']['L_preproj_spread']:.3f}")
    print(f"  Dial range passes (>=0.08): {dial_metrics['R_disabled']['dial_range_passes']}")
    print(f"  Is monotonic: {dial_metrics['R_disabled']['is_monotonic']}")
    print(f"  Projection active (R=10): {np.mean(dial_metrics['R10']['projection_active_rate'])*100:.0f}%")
    print(f"  Projection active (R=disabled): {np.mean(dial_metrics['R_disabled']['projection_active_rate'])*100:.0f}%")

    # Generate outputs - figures go to paper repo
    print("\n[Generate] Creating figures (to paper repo)...")
    generate_dial_figure(dial_metrics, FIG_DIR / "fig_exp2_dial_does_not_control_Lz.pdf")
    # Preserve the established artifact filename for downstream readers. The
    # generated caption and claims are finite-sample comparisons, not causal.
    generate_stability_figure(
        stability_comp,
        by_radius,
        FIG_DIR / "fig_exp2_projection_is_primary_stabilizer.pdf",
    )

    print("\n[Generate] Creating tables...")
    generate_dial_table(dial_metrics, OUT_DIR / "table_exp2_dial_does_not_control_Lz.tex")
    generate_projection_table(stability_comp, by_radius, OUT_DIR / "table_exp2_projection_effect.tex")

    print("\n[Generate] Creating documentation...")
    generate_claims_md(dial_metrics, stability_comp, OUT_DIR / "CLAIMS.md")
    generate_provenance_md(OUT_DIR / "PROVENANCE.md")
    generate_paper_snippet(OUT_DIR / "PAPER_INSERT_SNIPPET.tex")
    generate_summary_json(dial_metrics, stability_comp, OUT_DIR / "summary.json")
    generate_audit_placeholder(OUT_DIR / "AUDIT.md")

    print("\n" + "=" * 60)
    print("EXP2 FINAL: Generation Complete")
    print("=" * 60)
    print(f"\nFigures (paper repo): {FIG_DIR}")
    print(f"  - fig_exp2_dial_does_not_control_Lz.pdf")
    print("  - fig_exp2_projection_is_primary_stabilizer.pdf (legacy filename)")
    print(f"\nDocs & tables (trm_bellman): {OUT_DIR}")
    print(f"  - table_exp2_*.tex, CLAIMS.md, PROVENANCE.md")
    print("\nNext step: Run audit with")
    print("  buck2 run //buiksat_trm:audit_exp2_final_paper_ready")


if __name__ == "__main__":
    main()
