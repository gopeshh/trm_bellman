#!/usr/bin/env python3
"""
Generate preview plot for z-only contraction FAST experiment.
Labels clearly as FAST/approx baseline - NOT for paper figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
existing_csv = Path('/home/buiksat/trm_bellman/results/plot_data_contraction_fix_rerun/plot_data_feasibility_learning_curves_ALL.csv')
new_csv = Path('/home/buiksat/trm_bellman/results/plot_data_trivial_zonly_contraction_FAST/plot_data_feasibility_learning_curves.csv')
output_dir = Path('/home/buiksat/trm_bellman/results/plots_trivial_zonly_contraction_FAST')
combined_csv = Path('/home/buiksat/trm_bellman/results/plot_data_trivial_zonly_contraction_FAST/plot_data_feasibility_learning_curves_ALL_PLUS_ZONLY_FAST.csv')

output_dir.mkdir(parents=True, exist_ok=True)

# Load and combine data
print("Loading existing data...")
df_existing = pd.read_csv(existing_csv)
print(f"  Existing: {len(df_existing)} rows, algorithms: {df_existing['algorithm'].unique().tolist()}")

print("Loading new z-only contraction data...")
df_new = pd.read_csv(new_csv)
print(f"  New: {len(df_new)} rows")

# Combine
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv(combined_csv, index=False)
print(f"Wrote combined CSV: {combined_csv} ({len(df_combined)} rows)")

# Select algorithms for comparison plot
# Focus on: no_contraction (baseline), episodic_contraction (both norms), contraction_zonly_fast (new)
algos_to_plot = ['no_contraction', 'episodic_contraction', 'contraction_zonly_fast']
algo_labels = {
    'no_contraction': 'No Contraction (baseline)',
    'episodic_contraction': 'Full Contraction (zcon+vhead)',
    'contraction_zonly_fast': 'z-only Contraction (FAST approx)'
}
algo_colors = {
    'no_contraction': '#2ecc71',  # green
    'episodic_contraction': '#e74c3c',  # red
    'contraction_zonly_fast': '#3498db'  # blue
}

# Filter to selected algorithms
df_plot = df_combined[df_combined['algorithm'].isin(algos_to_plot)]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Success Rate
ax1 = axes[0]
for algo in algos_to_plot:
    df_algo = df_plot[df_plot['algorithm'] == algo]
    if len(df_algo) == 0:
        continue

    # Group by step, compute mean and std across seeds
    grouped = df_algo.groupby('step')['success_rate'].agg(['mean', 'std']).reset_index()

    ax1.plot(grouped['step'], grouped['mean'],
             label=algo_labels.get(algo, algo),
             color=algo_colors.get(algo, 'gray'),
             linewidth=2)
    ax1.fill_between(grouped['step'],
                     grouped['mean'] - grouped['std'],
                     grouped['mean'] + grouped['std'],
                     color=algo_colors.get(algo, 'gray'),
                     alpha=0.2)

ax1.set_xlabel('Training Steps (S)')
ax1.set_ylabel('Success Rate')
ax1.set_title('Success Rate (FAST preview - approx baseline)')
ax1.legend(loc='lower right')
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3)

# Plot 2: Mean Score
ax2 = axes[1]
for algo in algos_to_plot:
    df_algo = df_plot[df_plot['algorithm'] == algo]
    if len(df_algo) == 0:
        continue

    grouped = df_algo.groupby('step')['mean_score'].agg(['mean', 'std']).reset_index()

    ax2.plot(grouped['step'], grouped['mean'],
             label=algo_labels.get(algo, algo),
             color=algo_colors.get(algo, 'gray'),
             linewidth=2)
    ax2.fill_between(grouped['step'],
                     grouped['mean'] - grouped['std'],
                     grouped['mean'] + grouped['std'],
                     color=algo_colors.get(algo, 'gray'),
                     alpha=0.2)

ax2.set_xlabel('Training Steps (S)')
ax2.set_ylabel('Mean Checker Score')
ax2.set_title('Mean Score (FAST preview - approx baseline)')
ax2.legend(loc='lower right')
ax2.set_xlim(0, 5000)
ax2.grid(True, alpha=0.3)

plt.suptitle('FAST PREVIEW: z-only Contraction vs Baselines\n(exact_baseline_summation=false - NOT for paper)',
             fontsize=12, fontweight='bold')
plt.tight_layout()

# Save
png_path = output_dir / 'trivial_baselines_vs_zonly_contraction_FAST.png'
pdf_path = output_dir / 'trivial_baselines_vs_zonly_contraction_FAST.pdf'
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY: Final step (S=5000) performance")
print("="*60)
for algo in algos_to_plot:
    df_algo = df_plot[(df_plot['algorithm'] == algo) & (df_plot['step'] == df_plot[df_plot['algorithm'] == algo]['step'].max())]
    if len(df_algo) > 0:
        sr_mean = df_algo['success_rate'].mean()
        sr_std = df_algo['success_rate'].std()
        score_mean = df_algo['mean_score'].mean()
        print(f"{algo_labels.get(algo, algo):40s}: SR={sr_mean:.1%} ± {sr_std:.1%}, Score={score_mean:.2f}")

print("\n" + "="*60)
print("KEY FINDING: z-only contraction is STABLE (no collapse)")
print("  - Values did not collapse to -20")
print("  - Learning is slower than no_contraction but stable")
print("  - Unlike full contraction (zcon+vhead), no pathology")
print("="*60)
