#!/usr/bin/env python3
"""
Figure generation for Exp4: L_preproj vs Argmax Agreement (n₂=8) scatter plot.

Usage:
    python scripts/generate_exp4_figures.py \
        --summary_json results/paper_ready/exp4_projection_free_dial_final/summary.json \
        --out_dir results/paper_ready/exp4_projection_free_dial_final

Outputs:
    - fig_exp4_dial_scatter.pdf: L_preproj vs Argmax Agreement (n₂=8) scatter with Spearman ρ and 95% CI
    - fig_exp4_scale_comparison.pdf: Bar chart of metrics by scale factor (supplement)
    - fig_exp4_dial_latex.tex: LaTeX table of results
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load summary.json from exp4 evaluation."""
    with open(summary_path, "r") as f:
        return json.load(f)


def generate_scatter_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate L_preproj vs Argmax Agreement (n₂=8) scatter plot with Spearman ρ and 95% CI."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("[Warning] matplotlib not available, skipping scatter plot")
        return

    all_results = summary.get("all_results", [])
    if not all_results:
        print("[Warning] No results to plot")
        return

    # Extract data
    L_preproj = [r["L_preproj"] for r in all_results]
    argmax_b0 = [r["argmax_b0_8x"] for r in all_results]
    argmax_b1 = [r["argmax_b1_8x"] for r in all_results]
    seeds = [r["checkpoint_seed"] for r in all_results]
    scales = [r["scale"] for r in all_results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Color by seed
    unique_seeds = sorted(set(seeds))
    seed_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    seed_to_color = {
        seed: seed_colors[index % len(seed_colors)]
        for index, seed in enumerate(unique_seeds)
    }

    # Marker by scale
    unique_scales = sorted(set(scales), reverse=True)
    markers = ['o', 's', '^', 'D']
    scale_to_marker = {s: m for s, m in zip(unique_scales, markers)}

    # B0 scatter
    for i, (lp, aa, seed, scale) in enumerate(zip(L_preproj, argmax_b0, seeds, scales)):
        ax1.scatter(lp, aa, c=[seed_to_color[seed]], marker=scale_to_marker[scale],
                   s=100, edgecolors='black', linewidth=0.5)

    # Trend line for B0 - use bootstrap CI if available, otherwise just ρ (no i.i.d. p-values)
    from scipy.stats import spearmanr
    rho_b0, _ = spearmanr(L_preproj, argmax_b0)
    z = np.polyfit(L_preproj, argmax_b0, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(L_preproj), max(L_preproj), 100)

    # Check for bootstrap CI data (cluster-valid)
    mono = summary.get("monotonicity", {})
    boot_b0 = mono.get("boot_aa_b0", {})
    boot_b1 = mono.get("boot_aa_b1", {})

    if boot_b0 and "rho_ci_lower" in boot_b0:
        # Use cluster bootstrap CI (statistically valid for repeated-measures)
        ci_lo = boot_b0.get("rho_ci_lower", 0)
        ci_hi = boot_b0.get("rho_ci_upper", 0)
        label_b0 = f'Trend (ρ={rho_b0:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}])'
    else:
        # No bootstrap data - show ρ only, do NOT show i.i.d. p-values
        label_b0 = f'Trend (ρ={rho_b0:.3f})'

    ax1.plot(x_line, p(x_line), 'k--', alpha=0.5, label=label_b0)

    ax1.set_xlabel('L_preproj (Lipschitz estimate)')
    ax1.set_ylabel('Argmax Agreement (n₂=8)')
    ax1.set_title('B0: Initial States')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # B1 scatter
    for i, (lp, aa, seed, scale) in enumerate(zip(L_preproj, argmax_b1, seeds, scales)):
        ax2.scatter(lp, aa, c=[seed_to_color[seed]], marker=scale_to_marker[scale],
                   s=100, edgecolors='black', linewidth=0.5)

    # Trend line for B1 - use bootstrap CI if available
    rho_b1, _ = spearmanr(L_preproj, argmax_b1)
    z = np.polyfit(L_preproj, argmax_b1, 1)
    p = np.poly1d(z)

    if boot_b1 and "rho_ci_lower" in boot_b1:
        ci_lo = boot_b1.get("rho_ci_lower", 0)
        ci_hi = boot_b1.get("rho_ci_upper", 0)
        label_b1 = f'Trend (ρ={rho_b1:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}])'
    else:
        label_b1 = f'Trend (ρ={rho_b1:.3f})'

    ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, label=label_b1)

    ax2.set_xlabel('L_preproj (Lipschitz estimate)')
    ax2.set_ylabel('Argmax Agreement (n₂=8)')
    ax2.set_title('B1: Successor States')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Legend for seeds and scales
    legend_elements = []
    for seed in unique_seeds:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=seed_to_color[seed],
                      markersize=10, label=f'Seed {seed}')
        )
    for scale in unique_scales:
        legend_elements.append(
            plt.Line2D([0], [0], marker=scale_to_marker[scale], color='gray',
                      markersize=10, label=f'Scale {scale:.2f}')
        )

    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), ncol=len(unique_seeds) + len(unique_scales),
               fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save
    pdf_path = out_path / "fig_exp4_dial_scatter.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    png_path = out_path / "fig_exp4_dial_scatter.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {png_path}")

    plt.close()


def generate_latex_table(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate LaTeX table of results by scale."""
    scale_sums = summary.get("scale_summaries", [])
    mono = summary.get("monotonicity", {})

    # Extract bootstrap results (new format) or fall back to old format
    boot_b0 = mono.get("boot_aa_b0", {})
    boot_b1 = mono.get("boot_aa_b1", {})

    ci_lo_b0 = 0.0
    ci_hi_b0 = 0.0
    ci_lo_b1 = 0.0
    ci_hi_b1 = 0.0
    p_b0 = 1.0
    p_b1 = 1.0

    # Check which format we have
    if boot_b0:
        # New bootstrap format with CIs
        rho_b0 = boot_b0.get("rho_point", 0)
        ci_lo_b0 = boot_b0.get("rho_ci_lower", 0)
        ci_hi_b0 = boot_b0.get("rho_ci_upper", 0)
        rho_b1 = boot_b1.get("rho_point", 0)
        ci_lo_b1 = boot_b1.get("rho_ci_lower", 0)
        ci_hi_b1 = boot_b1.get("rho_ci_upper", 0)
        use_bootstrap = True
    else:
        # Legacy format with p-values
        rho_b0 = mono.get("rho_aa_b0", 0)
        rho_b1 = mono.get("rho_aa_b1", 0)
        p_b0 = mono.get("p_aa_b0", 1)
        p_b1 = mono.get("p_aa_b1", 1)
        use_bootstrap = False

    content = r"""\begin{table}[t]
\centering
\caption{Exp4: Inference-time contraction dial results. %s}
\label{tab:exp4-dial}
\begin{tabular}{lccccc}
\toprule
Scale & $\hat{L}_{\text{preproj}}$ & Argmax (n2=8) B0 & Argmax (n2=8) B1 & $\Delta V$ (n2=8) B0 & Entropy \\
\midrule
""" % summary.get("decision", "")

    for s in scale_sums:
        content += f"{s['scale']:.2f} & "
        content += f"${s['L_preproj_mean']:.3f} \\pm {s['L_preproj_std']:.3f}$ & "
        content += f"${s['argmax_b0_8x_mean']:.3f} \\pm {s['argmax_b0_8x_std']:.3f}$ & "
        content += f"${s['argmax_b1_8x_mean']:.3f} \\pm {s['argmax_b1_8x_std']:.3f}$ & "
        content += f"${s['delta_V_b0_8x_mean']:.3f}$ & "
        content += f"${s['entropy_train_mean']:.2f}$ \\\\\n"

    if use_bootstrap:
        content += r"""\midrule
\multicolumn{6}{l}{\small Spearman $\rho$ (B0 Argmax, n2=8): $\rho=%.3f$ [95\%% CI: %.3f, %.3f]} \\
\multicolumn{6}{l}{\small Spearman $\rho$ (B1 Argmax, n2=8): $\rho=%.3f$ [95\%% CI: %.3f, %.3f]} \\
\bottomrule
\end{tabular}
\end{table}
""" % (rho_b0, ci_lo_b0, ci_hi_b0, rho_b1, ci_lo_b1, ci_hi_b1)
    else:
        content += r"""\midrule
\multicolumn{6}{l}{\small Spearman $\rho$ (B0 Argmax, n2=8): $\rho=%.3f$ ($p=%.4f$)} \\
\multicolumn{6}{l}{\small Spearman $\rho$ (B1 Argmax, n2=8): $\rho=%.3f$ ($p=%.4f$)} \\
\bottomrule
\end{tabular}
\end{table}
""" % (rho_b0, p_b0, rho_b1, p_b1)

    tex_path = out_path / "fig_exp4_dial_latex.tex"
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"Saved: {tex_path}")


def generate_scale_comparison(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate bar chart comparing metrics across scales."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[Warning] matplotlib not available, skipping bar chart")
        return

    scale_sums = summary.get("scale_summaries", [])
    if not scale_sums:
        return

    scales = [s["scale"] for s in scale_sums]
    L_preproj = [s["L_preproj_mean"] for s in scale_sums]
    L_preproj_std = [s["L_preproj_std"] for s in scale_sums]
    argmax_b0 = [s["argmax_b0_8x_mean"] for s in scale_sums]
    argmax_b0_std = [s["argmax_b0_8x_std"] for s in scale_sums]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(scales))
    width = 0.35

    # L_preproj by scale
    ax1.bar(x, L_preproj, width, yerr=L_preproj_std, capsize=3, color='steelblue')
    ax1.set_xlabel('Scale Factor')
    ax1.set_ylabel('L_preproj')
    ax1.set_title('Lipschitz Estimate by Scale')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s:.2f}' for s in scales])
    ax1.grid(True, alpha=0.3, axis='y')

    # Argmax agreement by scale
    ax2.bar(x, argmax_b0, width, yerr=argmax_b0_std, capsize=3, color='darkgreen')
    ax2.set_xlabel('Scale Factor')
    ax2.set_ylabel('Argmax Agreement (n₂=8)')
    ax2.set_title('Policy Stability by Scale')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s:.2f}' for s in scales])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    pdf_path = out_path / "fig_exp4_scale_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Exp4 figures")
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path to summary.json from exp4 evaluation",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (defaults to same as summary.json)",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        return 1

    out_path = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_path.mkdir(parents=True, exist_ok=True)

    summary = load_summary(str(summary_path))

    print(f"Generating figures from: {summary_path}")
    print(f"Output directory: {out_path}")
    print(f"Decision: {summary.get('decision', 'unknown')}")
    print()

    generate_scatter_plot(summary, out_path)
    generate_latex_table(summary, out_path)
    generate_scale_comparison(summary, out_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
