#!/usr/bin/env python3
"""Generate plots for 6-8 empties all ablations experiments."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path("results/plot_data_6to8empties_all_ablations_20k")
OUTPUT_DIR = Path("results/plots_6to8empties_all_ablations_20k")

ALGO_CONFIG = {
    "persistent_z_no_contraction": {"name": "Persistent + No Contraction", "color": "#2ca02c", "marker": "s"},
    "no_contraction": {"name": "Episodic + No Contraction", "color": "#1f77b4", "marker": "o"},
    "ablation_no_conservative": {"name": "No Conservative (α=1)", "color": "#ff7f0e", "marker": "^"},
    "upi_trm": {"name": "UPI-TRM (Episodic + Contraction)", "color": "#d62728", "marker": "D"},
    "ablation_persistent_z": {"name": "Persistent + Contraction", "color": "#9467bd", "marker": "v"},
    "a2c": {"name": "A2C Baseline", "color": "#8c564b", "marker": "x"},
    "dqn": {"name": "DQN Baseline", "color": "#7f7f7f", "marker": "+"},
}

def plot_learning_curves():
    """Plot learning curves for all algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for algo, config in ALGO_CONFIG.items():
        csv_path = DATA_DIR / f"{algo}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        # Group by step and compute mean/std across seeds
        grouped = df.groupby('step').agg({
            'success_rate': ['mean', 'std'],
            'mean_score': ['mean', 'std']
        }).reset_index()

        steps = grouped['step']

        # Success rate plot
        mean_sr = grouped[('success_rate', 'mean')] * 100
        std_sr = grouped[('success_rate', 'std')] * 100
        axes[0].plot(steps, mean_sr, label=config['name'], color=config['color'],
                    marker=config['marker'], markevery=20, linewidth=1.5)
        axes[0].fill_between(steps, mean_sr - std_sr, mean_sr + std_sr,
                            alpha=0.2, color=config['color'])

        # Mean score plot
        mean_score = grouped[('mean_score', 'mean')]
        std_score = grouped[('mean_score', 'std')]
        axes[1].plot(steps, mean_score, label=config['name'], color=config['color'],
                    marker=config['marker'], markevery=20, linewidth=1.5)
        axes[1].fill_between(steps, mean_score - std_score, mean_score + std_score,
                            alpha=0.2, color=config['color'])

    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('4×4 Sudoku (6-8 Empties): Success Rate')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 75)

    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Mean Score')
    axes[1].set_title('4×4 Sudoku (6-8 Empties): Mean Score')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=9.0, color='gray', linestyle='--', alpha=0.5, label='Initial')
    axes[1].axhline(y=16.0, color='green', linestyle='--', alpha=0.5, label='Solved')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = OUTPUT_DIR / f"learning_curves.{ext}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {output_path}")

    plt.close()

def plot_final_bar_chart():
    """Plot bar chart of final success rates."""
    summary_path = DATA_DIR / "summary.csv"
    if not summary_path.exists():
        print("No summary.csv found")
        return

    df = pd.read_csv(summary_path)

    # Calculate mean and std per algorithm
    stats = df.groupby('algorithm').agg({
        'final_success_rate': ['mean', 'std']
    }).reset_index()
    stats.columns = ['algorithm', 'mean', 'std']

    # Sort by mean success rate
    stats = stats.sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stats))
    colors = [ALGO_CONFIG.get(algo, {}).get('color', 'gray') for algo in stats['algorithm']]

    bars = ax.bar(x, stats['mean'] * 100, yerr=stats['std'] * 100,
                  color=colors, capsize=5, alpha=0.8)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
        ax.text(i, mean * 100 + std * 100 + 2, f'{mean*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('4×4 Sudoku (6-8 Empties): Final Success Rate @ 20k Steps\n(n=3 seeds, error bars = 1 std)')

    # Use short names
    short_names = [ALGO_CONFIG.get(algo, {}).get('name', algo).replace(' + ', '\n')
                   for algo in stats['algorithm']]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)

    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = OUTPUT_DIR / f"final_bar_chart.{ext}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {output_path}")

    plt.close()

def plot_contraction_comparison():
    """Plot focused comparison: contraction vs no-contraction."""
    fig, ax = plt.subplots(figsize=(8, 5))

    comparison_algos = [
        ("persistent_z_no_contraction", "Persistent + No Contraction"),
        ("no_contraction", "Episodic + No Contraction"),
        ("ablation_persistent_z", "Persistent + Contraction"),
        ("upi_trm", "Episodic + Contraction"),
    ]

    for algo, name in comparison_algos:
        csv_path = DATA_DIR / f"{algo}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        grouped = df.groupby('step').agg({'success_rate': ['mean', 'std']}).reset_index()

        steps = grouped['step']
        mean_sr = grouped[('success_rate', 'mean')] * 100
        std_sr = grouped[('success_rate', 'std')] * 100

        config = ALGO_CONFIG.get(algo, {})
        ax.plot(steps, mean_sr, label=name, color=config.get('color', 'gray'),
                marker=config.get('marker', 'o'), markevery=20, linewidth=2)
        ax.fill_between(steps, mean_sr - std_sr, mean_sr + std_sr, alpha=0.2,
                       color=config.get('color', 'gray'))

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Contraction Ablation: 4×4 Sudoku (6-8 Empties)\nNo Contraction enables learning; Contraction blocks it')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 75)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = OUTPUT_DIR / f"contraction_comparison.{ext}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {output_path}")

    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating learning curves...")
    plot_learning_curves()

    print("Generating final bar chart...")
    plot_final_bar_chart()

    print("Generating contraction comparison...")
    plot_contraction_comparison()

    print("Done!")

if __name__ == "__main__":
    main()
