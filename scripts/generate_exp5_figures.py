#!/usr/bin/env python3
"""
Figure generation for Exp5: Stability–Expressivity Tradeoff Curve.

Produces:
- fig_exp5_tradeoff_curve.pdf: X=stability (argmax@8×), Y=success rate
- table_exp5_tradeoff_curve.tex: LaTeX table of results

Usage:
    python scripts/generate_exp5_figures.py \
        --summary_json results/paper_ready/exp5_tradeoff_curve/summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load summary.json."""
    with open(summary_path, "r") as f:
        return json.load(f)


def generate_tradeoff_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate stability vs expressivity tradeoff plot.
    
    Axes:
    - X: Argmax agreement @ 8× depth mismatch (n_train=2 → n_eval=16)
    - Y: Episode success rate [%] (solved within T steps)
    
    Caption should specify: suite definition, #episodes, horizon T, projection status.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as mticker
        matplotlib.use('Agg')
    except ImportError:
        print("[Warning] matplotlib not available, skipping plot")
        return

    all_results = summary.get("all_results", [])
    if not all_results:
        print("[Warning] No results to plot")
        return

    # Extract data
    scales = sorted(set(r["scale"] for r in all_results), reverse=True)

    # Colors by scale (consistent across points)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scales)))
    scale_to_color = {s: c for s, c in zip(scales, colors)}
    markers = ['o', 's', '^', 'D']
    scale_to_marker = {s: m for s, m in zip(scales, markers)}

    # ICML single-column: 3.25 in wide
    fig, ax = plt.subplots(figsize=(3.25, 2.2), constrained_layout=True)

    # Convert success to percentage
    x_all = np.array([r["argmax_b0_8x"] for r in all_results])
    y_all = np.array([100.0 * r["success_trivial"] for r in all_results])

    # Plot individual points (per seed/checkpoint)
    for r in all_results:
        ax.scatter(
            r["argmax_b0_8x"], 100.0 * r["success_trivial"],
            c=[scale_to_color[r["scale"]]],
            marker=scale_to_marker[r["scale"]],
            s=28, alpha=0.7, edgecolors='black', linewidth=0.4,
            zorder=3
        )

    # Plot scale means with error bars (gray, behind points)
    scale_sums = summary.get("scale_summaries", [])
    for s in scale_sums:
        ax.errorbar(
            s["argmax_b0_8x_mean"], 100.0 * s["success_trivial_mean"],
            xerr=s["argmax_b0_8x_std"], yerr=100.0 * s["success_trivial_std"],
            fmt='none', color='0.4', alpha=0.5, capsize=2, elinewidth=0.8,
            zorder=2
        )

    # Direct text annotations instead of legend (cleaner for single-column)
    for s in scale_sums:
        xm = s["argmax_b0_8x_mean"]
        ym = 100.0 * s["success_trivial_mean"]
        ax.annotate(
            f's={s["scale"]:.2f}',
            (xm, ym),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color='0.2',
            zorder=4
        )

    # Axis labels: measured quantities only, no interpretive framing
    ax.set_xlabel(r'Argmax agreement @ 8$\times$ depth mismatch', fontsize=8)
    ax.set_ylabel('Episode success rate [%]', fontsize=8)

    ax.grid(True, alpha=0.3, linewidth=0.6)

    # Tight y-axis limits based on data
    ymin, ymax = y_all.min(), y_all.max()
    pad = max(0.5, 0.15 * (ymax - ymin))  # at least 0.5 pp padding
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    # Tight x-axis limits
    xmin, xmax = x_all.min(), x_all.max()
    ax.set_xlim(xmin - 0.02, xmax + 0.02)

    # Tick font sizes for single-column
    ax.tick_params(axis='both', labelsize=7)

    # Save
    pdf_path = out_path / "fig_exp5_tradeoff_curve.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    png_path = out_path / "fig_exp5_tradeoff_curve.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {png_path}")

    plt.close()


def generate_L_vs_metrics_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate L_preproj vs stability/success dual-axis plot.
    
    Shows how achieved Lipschitz constant relates to both stability and success.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as mticker
        matplotlib.use('Agg')
    except ImportError:
        return

    scale_sums = summary.get("scale_summaries", [])
    if not scale_sums:
        return

    # Sort by scale (descending)
    scale_sums = sorted(scale_sums, key=lambda x: -x["scale"])

    L_preproj = np.array([s["L_preproj_mean"] for s in scale_sums])
    stability = np.array([s["argmax_b0_8x_mean"] for s in scale_sums])
    success_pct = np.array([100.0 * s["success_trivial_mean"] for s in scale_sums])
    scales = [s["scale"] for s in scale_sums]

    # ICML single-column width
    fig, ax1 = plt.subplots(figsize=(3.25, 2.2), constrained_layout=True)

    # Plot stability on left axis
    color1 = 'steelblue'
    ax1.set_xlabel(r'$\hat{L}_{\mathrm{preproj}}$ (achieved Lipschitz)', fontsize=8)
    ax1.set_ylabel(r'Argmax agreement @ 8$\times$', color=color1, fontsize=8)
    line1, = ax1.plot(L_preproj, stability, 'o-', color=color1, linewidth=1.5, markersize=5)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=7)
    ax1.tick_params(axis='x', labelsize=7)

    # Create second y-axis for success
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.set_ylabel('Episode success rate [%]', color=color2, fontsize=8)
    line2, = ax2.plot(L_preproj, success_pct, 's--', color=color2, linewidth=1.5, markersize=5)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=7)

    # Zoom success axis to observed range
    pad = max(0.5, 0.15 * (success_pct.max() - success_pct.min()))
    ax2.set_ylim(success_pct.min() - pad, success_pct.max() + pad)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(4))

    # Add scale annotations (on stability line)
    for lp, stab, s in zip(L_preproj, stability, scales):
        ax1.annotate(f's={s}', (lp, stab), textcoords="offset points", xytext=(3, 3), fontsize=6)

    # Compact legend
    ax1.legend([line1, line2], ['Stability', 'Success'],
               loc='lower right', fontsize=6, frameon=True, borderpad=0.3)

    ax1.grid(True, alpha=0.3, linewidth=0.6)

    pdf_path = out_path / "fig_exp5_L_vs_metrics.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    plt.close()


def generate_latex_table(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate LaTeX table."""
    scale_sums = summary.get("scale_summaries", [])
    gates = summary.get("gates", {})

    content = r"""\begin{table}[t]
\centering
\caption{Exp5: Stability–Expressivity Tradeoff. %s}
\label{tab:exp5-tradeoff}
\begin{tabular}{lcccc}
\toprule
Scale & $\hat{L}_{\text{preproj}}$ & Stability (argmax@8$\times$) & Success (trivial) & $\Delta V$@8$\times$ \\
\midrule
""" % summary.get("decision", "")

    for s in sorted(scale_sums, key=lambda x: -x["scale"]):
        content += f"{s['scale']:.2f} & "
        content += f"${s['L_preproj_mean']:.3f} \\pm {s['L_preproj_std']:.3f}$ & "
        content += f"${s['argmax_b0_8x_mean']:.3f} \\pm {s['argmax_b0_8x_std']:.3f}$ & "
        content += f"${s['success_trivial_mean']:.3f} \\pm {s['success_trivial_std']:.3f}$ & "
        content += f"${s['delta_V_b0_8x_mean']:.3f}$ \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex_path = out_path / "table_exp5_tradeoff_curve.tex"
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Exp5 figures")
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path to summary.json",
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
    print()

    generate_tradeoff_plot(summary, out_path)
    generate_L_vs_metrics_plot(summary, out_path)
    generate_latex_table(summary, out_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
