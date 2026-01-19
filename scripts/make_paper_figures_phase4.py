#!/usr/bin/env python3
"""
Phase 4: Figure Generation for 2×2 Norm Ablation.

Produces:
- fig_phase4_2x2_norm_ablation.pdf: 2×2 grid showing condition effects
- table_phase4_2x2_norm_ablation.tex: LaTeX table

Usage:
    buck2 run //buiksat_trm:make_paper_figures_phase4 -- \
        --summary_json results/paper_ready/phase4_2x2_norm_ablation/summary.json
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


def generate_2x2_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate 2×2 grid plot showing condition effects."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plot")
        return

    aggregates = summary.get("aggregates", [])
    if not aggregates:
        print("[Warning] No aggregates to plot")
        return

    # Create 2×2 matrix of results
    # Rows: z→z contraction (OFF, ON)
    # Cols: value-head norm (OFF, ON)
    matrix_labels = [
        ["nc_nv", "nc_yv"],  # Contraction OFF
        ["yc_nv", "yc_yv"],  # Contraction ON
    ]

    # Get data by condition
    data_by_cond = {a["condition"]: a for a in aggregates}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Phase 4: 2×2 Norm Ablation", fontsize=14, fontweight='bold')

    metrics = [
        ("var_V_mean", "Var(V)", "steelblue"),
        ("argmax_4x_mean", "Argmax@4×", "darkgreen"),
        ("success_trivial_mean", "Success (trivial)", "darkorange"),
    ]

    for i, contraction in enumerate(["OFF", "ON"]):
        for j, vhead in enumerate(["OFF", "ON"]):
            ax = axes[i, j]
            cond = matrix_labels[i][j]

            if cond in data_by_cond:
                data = data_by_cond[cond]

                # Bar chart of metrics
                x = np.arange(len(metrics))
                values = [data.get(m[0], 0) for m in metrics]
                stds = [data.get(m[0].replace("_mean", "_std"), 0) for m in metrics]
                colors = [m[2] for m in metrics]

                bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
                ax.errorbar(x, values, yerr=stds, fmt='none', color='black', capsize=3)

                ax.set_xticks(x)
                ax.set_xticklabels([m[1] for m in metrics], rotation=45, ha='right')
                ax.set_ylim(0, max(1.0, max(values) * 1.2))

            ax.set_title(f"C={contraction}, V={vhead}", fontsize=11)
            ax.grid(True, alpha=0.3)

    # Add row/column labels
    axes[0, 0].set_ylabel("z→z Contraction OFF", fontsize=10)
    axes[1, 0].set_ylabel("z→z Contraction ON", fontsize=10)

    plt.tight_layout()

    pdf_path = out_path / "fig_phase4_2x2_norm_ablation.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    png_path = out_path / "fig_phase4_2x2_norm_ablation.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {png_path}")

    plt.close()


def generate_bar_comparison(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate grouped bar chart comparing conditions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    aggregates = summary.get("aggregates", [])
    if not aggregates:
        return

    conditions = [a["condition"] for a in aggregates]
    labels = [a["label"] for a in aggregates]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot 1: Var(V)
    ax = axes[0]
    vals = [a["var_V_mean"] for a in aggregates]
    stds = [a["var_V_std"] for a in aggregates]
    x = np.arange(len(conditions))
    ax.bar(x, vals, yerr=stds, color='steelblue', alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Var(V)")
    ax.set_title("Value Variance")
    ax.grid(True, alpha=0.3)

    # Plot 2: Argmax Agreement
    ax = axes[1]
    vals = [a["argmax_4x_mean"] for a in aggregates]
    stds = [a["argmax_4x_std"] for a in aggregates]
    ax.bar(x, vals, yerr=stds, color='darkgreen', alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Argmax Agreement @4×")
    ax.set_title("Policy Stability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 3: Success Rate
    ax = axes[2]
    vals = [a["success_trivial_mean"] for a in aggregates]
    stds = [a["success_trivial_std"] for a in aggregates]
    ax.bar(x, vals, yerr=stds, color='darkorange', alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Success Rate")
    ax.set_title("Trivial Suite")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Phase 4: Norm Ablation Comparison", fontsize=12, fontweight='bold')
    plt.tight_layout()

    pdf_path = out_path / "fig_phase4_bar_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    plt.close()


def generate_latex_table(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate LaTeX table."""
    aggregates = summary.get("aggregates", [])

    content = r"""\begin{table}[t]
\centering
\caption{Phase 4: 2×2 Norm Ablation across 3 seeds. C=z$\to$z contraction, V=value-head spectral norm.}
\label{tab:phase4-2x2-ablation}
\begin{tabular}{lcccccc}
\toprule
Condition & C & V & Var($V$) & Argmax@4$\times$ & Success (trivial) \\
\midrule
"""

    for a in aggregates:
        c_status = "ON" if a["enable_contraction"] else "OFF"
        v_status = "OFF" if a["disable_value_head_norm"] else "ON"
        content += f"{a['label']} & {c_status} & {v_status} & "
        content += f"${a['var_V_mean']:.3f} \\pm {a['var_V_std']:.3f}$ & "
        content += f"${a['argmax_4x_mean']:.3f} \\pm {a['argmax_4x_std']:.3f}$ & "
        content += f"${a['success_trivial_mean']:.3f} \\pm {a['success_trivial_std']:.3f}$ \\\\\n"

    content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex_path = out_path / "table_phase4_2x2_norm_ablation.tex"
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 4 figures")
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

    generate_2x2_plot(summary, out_path)
    generate_bar_comparison(summary, out_path)
    generate_latex_table(summary, out_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
