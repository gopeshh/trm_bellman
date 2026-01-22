#!/usr/bin/env python3
"""
Figure generation for Exp4 with ICML paper-ready styling (bolder, more visible).
Matches style from plot_table3_hard.py

Usage:
    python scripts/generate_exp4_figures_paper_style.py \
        --summary_json results/paper_ready/exp4_projection_free_dial_final/summary.json \
        --out_dir /Users/buiksat/UPI_TRM/UPI_TRM_ICML/figures
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def apply_paper_style():
    """Apply paper-quality styling: larger fonts, readable at single-column width."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'pdf.fonttype': 42,  # TrueType fonts for better PDF rendering
        'ps.fonttype': 42,
        'font.weight': 'medium',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load summary.json from exp4 evaluation."""
    with open(summary_path, "r") as f:
        return json.load(f)


def generate_scatter_plot(summary: Dict[str, Any], out_path: Path) -> None:
    """Generate L_preproj vs Argmax Agreement scatter plot with ICML-ready styling."""
    apply_paper_style()
    
    all_results = summary.get("all_results", [])
    if not all_results:
        print("[Warning] No results to plot")
        return

    # Extract data - need to handle the nested structure
    L_preproj = []
    argmax_b0 = []
    argmax_b1 = []
    seeds = []
    scales = []
    
    for r in all_results:
        L_preproj.append(r["L_preproj"])
        # Handle both old and new format
        if "argmax_b0_8x" in r:
            argmax_b0.append(r["argmax_b0_8x"])
            argmax_b1.append(r["argmax_b1_8x"])
        else:
            argmax_b0.append(r["metrics_b0"]["argmax_agree@8x"])
            argmax_b1.append(r["metrics_b1"]["argmax_agree@8x"])
        seeds.append(r.get("seed", r.get("checkpoint_seed", 42)))
        scales.append(r["scale"])

    # Create figure with two subplots - sized for ICML single column
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Color by scale - use distinct, bold colors
    unique_scales = sorted(set(scales), reverse=True)
    scale_colors = {
        1.0: '#d62728',   # red
        0.85: '#ff7f0e',  # orange
        0.7: '#2ca02c',   # green
        0.55: '#1f77b4',  # blue
    }
    # Fallback colors
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i, s in enumerate(unique_scales):
        if s not in scale_colors:
            scale_colors[s] = fallback_colors[i % len(fallback_colors)]

    # Marker by seed
    unique_seeds = sorted(set(seeds))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    seed_to_marker = {s: m for s, m in zip(unique_seeds, markers)}

    # B0 scatter - smaller markers, transparent to show overlapping
    for lp, aa, seed, scale in zip(L_preproj, argmax_b0, seeds, scales):
        ax1.scatter(lp, aa, 
                   c=scale_colors[scale], 
                   marker=seed_to_marker[seed],
                   s=120,  # Smaller markers for overlap visibility
                   edgecolors='black', 
                   linewidth=1.5,
                   alpha=0.6,  # More transparent to show overlap
                   zorder=3)

    # Trend line for B0
    rho_b0, _ = spearmanr(L_preproj, argmax_b0)
    z = np.polyfit(L_preproj, argmax_b0, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(L_preproj) - 0.02, max(L_preproj) + 0.02, 100)
    
    ax1.plot(x_line, p(x_line), 'k--', linewidth=3.0, alpha=0.7, 
             label=f'ρ = {rho_b0:.2f}')

    ax1.set_xlabel(r'$L_z^{\mathrm{pre}}$ (Lipschitz estimate)', fontweight='bold')
    ax1.set_ylabel('Argmax Agreement', fontweight='bold')
    ax1.set_title('B0: Initial States', fontweight='bold', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_ylim(0.55, 0.78)

    # B1 scatter - smaller markers, transparent
    for lp, aa, seed, scale in zip(L_preproj, argmax_b1, seeds, scales):
        ax2.scatter(lp, aa, 
                   c=scale_colors[scale], 
                   marker=seed_to_marker[seed],
                   s=120,
                   edgecolors='black', 
                   linewidth=1.5,
                   alpha=0.6,
                   zorder=3)

    # Trend line for B1
    rho_b1, _ = spearmanr(L_preproj, argmax_b1)
    z = np.polyfit(L_preproj, argmax_b1, 1)
    p = np.poly1d(z)
    
    ax2.plot(x_line, p(x_line), 'k--', linewidth=3.0, alpha=0.7,
             label=f'ρ = {rho_b1:.2f}')

    ax2.set_xlabel(r'$L_z^{\mathrm{pre}}$ (Lipschitz estimate)', fontweight='bold')
    ax2.set_ylabel('Argmax Agreement', fontweight='bold')
    ax2.set_title('B1: Successor States', fontweight='bold', fontsize=14)
    ax2.legend(loc='lower left', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linewidth=0.8)
    ax2.set_ylim(0.55, 0.75)

    # Create legend for scales and seeds below the plot
    legend_elements = []
    # Scale legend (colors)
    for scale in unique_scales:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=scale_colors[scale],
                      markeredgecolor='black',
                      markeredgewidth=1.5,
                      markersize=12, label=f'λ={scale:.2f}')
        )
    # Add separator space
    legend_elements.append(
        plt.Line2D([0], [0], marker='', color='w', label='')
    )
    # Seed legend (markers)
    for seed in unique_seeds:
        legend_elements.append(
            plt.Line2D([0], [0], marker=seed_to_marker[seed], color='w',
                      markerfacecolor='gray',
                      markeredgecolor='black',
                      markeredgewidth=1.5,
                      markersize=12, label=f'Seed {seed}')
        )

    fig.legend(handles=legend_elements, 
               loc='upper center',
               bbox_to_anchor=(0.5, 0.02), 
               ncol=len(unique_scales) + 1 + len(unique_seeds),
               fontsize=10,
               frameon=False,
               handletextpad=0.3,
               columnspacing=0.8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # Save
    pdf_path = out_path / "fig_exp4_dial_scatter.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.02)
    print(f"Saved: {pdf_path}")

    png_path = out_path / "fig_exp4_dial_scatter.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300, pad_inches=0.02)
    print(f"Saved: {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Exp4 figures (paper style)")
    parser.add_argument(
        "--summary_json",
        type=str,
        default="results/paper_ready/exp4_projection_free_dial_final/summary.json",
        help="Path to summary.json from exp4 evaluation",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory",
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

    generate_scatter_plot(summary, out_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())

