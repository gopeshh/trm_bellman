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
    """Generate stability vs expressivity tradeoff plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
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
    seeds = sorted(set(r["checkpoint_seed"] for r in all_results))

    # Colors and markers by scale
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scales)))
    scale_to_color = {s: c for s, c in zip(scales, colors)}
    markers = ['o', 's', '^', 'D']
    scale_to_marker = {s: m for s, m in zip(scales, markers)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot individual points
    for r in all_results:
        ax.scatter(
            r["argmax_b0_8x"], r["success_trivial"],
            c=[scale_to_color[r["scale"]]],
            marker=scale_to_marker[r["scale"]],
            s=80, alpha=0.7, edgecolors='black', linewidth=0.5
        )

    # Plot scale means with error bars
    scale_sums = summary.get("scale_summaries", [])
    for s in scale_sums:
        ax.errorbar(
            s["argmax_b0_8x_mean"], s["success_trivial_mean"],
            xerr=s["argmax_b0_8x_std"], yerr=s["success_trivial_std"],
            fmt='none', color='gray', alpha=0.5, capsize=3
        )

    # Connect means with line
    means_x = [s["argmax_b0_8x_mean"] for s in sorted(scale_sums, key=lambda x: -x["scale"])]
    means_y = [s["success_trivial_mean"] for s in sorted(scale_sums, key=lambda x: -x["scale"])]
    ax.plot(means_x, means_y, 'k--', alpha=0.5, linewidth=1)

    # Labels
    ax.set_xlabel('Stability: Argmax Agreement @ 4× Depth Mismatch', fontsize=12)
    ax.set_ylabel('Expressivity: Success Rate (trivial)', fontsize=12)
    ax.set_title('Stability–Expressivity Tradeoff Curve', fontsize=14)

    # Legend for scales
    legend_elements = []
    for scale in scales:
        legend_elements.append(
            plt.Line2D([0], [0], marker=scale_to_marker[scale], color='w',
                      markerfacecolor=scale_to_color[scale],
                      markersize=10, label=f'Scale {scale:.2f}')
        )
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.7, 1.0])
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    # Save
    pdf_path = out_path / "fig_exp5_tradeoff_curve.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    png_path = out_path / "fig_exp5_tradeoff_curve.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {png_path}")

    plt.close()


def generate_L_vs_metrics_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate L_preproj vs stability/success dual-axis plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    scale_sums = summary.get("scale_summaries", [])
    if not scale_sums:
        return

    # Sort by scale (descending)
    scale_sums = sorted(scale_sums, key=lambda x: -x["scale"])

    L_preproj = [s["L_preproj_mean"] for s in scale_sums]
    stability = [s["argmax_b0_8x_mean"] for s in scale_sums]
    success = [s["success_trivial_mean"] for s in scale_sums]
    scales = [s["scale"] for s in scale_sums]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot stability on left axis
    color1 = 'steelblue'
    ax1.set_xlabel('$\\hat{L}_{\\text{preproj}}$ (Achieved Lipschitz)', fontsize=12)
    ax1.set_ylabel('Stability (Argmax Agreement)', color=color1, fontsize=12)
    line1, = ax1.plot(L_preproj, stability, 'o-', color=color1, linewidth=2, markersize=8, label='Stability')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create second y-axis for success
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.set_ylabel('Success Rate', color=color2, fontsize=12)
    line2, = ax2.plot(L_preproj, success, 's--', color=color2, linewidth=2, markersize=8, label='Success')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add scale annotations
    for i, (lp, stab, succ, s) in enumerate(zip(L_preproj, stability, success, scales)):
        ax1.annotate(f's={s}', (lp, stab), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Combined legend
    ax1.legend([line1, line2], ['Stability', 'Success'], loc='center right')

    ax1.set_title('Stability–Expressivity vs Achieved Lipschitz', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

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
