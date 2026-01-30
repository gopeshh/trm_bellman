#!/usr/bin/env python3
"""Plot training progress from UPI-TRM log file."""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_path):
    """Extract step, success_rate, and mean_score from log file."""
    steps = []
    success_rates = []
    mean_scores = []

    pattern = r'\[step (\d+)\] eval_success_rate=([\d.]+) eval_mean_score=([\d.]+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                steps.append(int(match.group(1)))
                success_rates.append(float(match.group(2)) * 100)  # Convert to percentage
                mean_scores.append(float(match.group(3)))

    return np.array(steps), np.array(success_rates), np.array(mean_scores)

def plot_training_progress(log_path, output_path):
    """Create dual-axis plot of training progress."""
    steps, success_rates, mean_scores = parse_log_file(log_path)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot mean score on left y-axis
    color1 = '#2563eb'  # Blue
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Score (out of 81)', color=color1, fontsize=12)
    line1 = ax1.plot(steps, mean_scores, color=color1, linewidth=2, label='Mean Score', marker='o', markersize=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(25, 65)
    ax1.axhline(y=26.84, color=color1, linestyle='--', alpha=0.5, label='Initial Score (26.84)')

    # Create second y-axis for success rate
    ax2 = ax1.twinx()
    color2 = '#16a34a'  # Green
    ax2.set_ylabel('Success Rate (%)', color=color2, fontsize=12)
    line2 = ax2.plot(steps, success_rates, color=color2, linewidth=2, label='Success Rate', marker='s', markersize=3)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-1, 15)

    # Title and grid
    plt.title('UPI-TRM 9x9 Sudoku Training Progress (50k steps, seed=0)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Add annotations for key milestones
    # First solve
    first_solve_idx = np.where(success_rates > 0)[0][0]
    ax2.annotate(f'First solve\n(step {steps[first_solve_idx]})',
                 xy=(steps[first_solve_idx], success_rates[first_solve_idx]),
                 xytext=(steps[first_solve_idx] + 3000, success_rates[first_solve_idx] + 3),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9)

    # Peak success rate
    peak_idx = np.argmax(success_rates)
    ax2.annotate(f'Peak: {success_rates[peak_idx]:.0f}%\n(step {steps[peak_idx]})',
                 xy=(steps[peak_idx], success_rates[peak_idx]),
                 xytext=(steps[peak_idx] - 5000, success_rates[peak_idx] + 2),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print summary stats
    print(f"\nSummary:")
    print(f"  Steps: {steps[0]} - {steps[-1]}")
    print(f"  Mean Score: {mean_scores[0]:.2f} → {mean_scores[-1]:.2f} (peak: {mean_scores.max():.2f} at step {steps[np.argmax(mean_scores)]})")
    print(f"  Success Rate: {success_rates[0]:.1f}% → {success_rates[-1]:.1f}% (peak: {success_rates.max():.1f}% at step {steps[np.argmax(success_rates)]})")

if __name__ == '__main__':
    log_path = '/home/buiksat/trm_bellman/results/9x9_experiments_seed0/upi_trm_50k_s0.log'
    output_path = '/home/buiksat/trm_bellman/results/9x9_experiments_seed0/upi_trm_50k_training_progress.png'
    plot_training_progress(log_path, output_path)
