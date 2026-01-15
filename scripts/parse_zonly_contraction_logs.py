#!/usr/bin/env python3
"""
Parse FAST preview logs for z-only contraction experiment.
Creates CSV in the same format as the existing plot_data_feasibility_learning_curves.csv
"""

import re
import os
from pathlib import Path

def parse_log(log_path: str, algorithm: str, seed: int) -> list:
    """Parse a log file and extract eval metrics."""
    rows = []

    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return rows

    with open(log_path, 'r') as f:
        content = f.read()

    # Pattern to match eval lines:
    # [step 00100] eval_success_rate=0.080 eval_mean_score=9.800 ...
    # [step 00100] PROGRESS: filled=14.72 violations=2.46 zero_cand=0.06

    eval_pattern = re.compile(
        r'\[step (\d+)\] eval_success_rate=([0-9.]+) eval_mean_score=([0-9.]+) '
        r'.*?\[solved=(\d+)/(\d+),'
    )

    progress_pattern = re.compile(
        r'\[step (\d+)\] PROGRESS: filled=([0-9.]+) violations=([0-9.]+) zero_cand=([0-9.]+)'
    )

    # Find all eval results
    eval_matches = {}
    for m in eval_pattern.finditer(content):
        step = int(m.group(1))
        eval_matches[step] = {
            'success_rate': float(m.group(2)),
            'mean_score': float(m.group(3)),
            'solved_count': int(m.group(4)),
            'total_episodes': int(m.group(5)),
        }

    # Find all progress results
    progress_matches = {}
    for m in progress_pattern.finditer(content):
        step = int(m.group(1))
        progress_matches[step] = {
            'filled_mean': float(m.group(2)),
            'violations_mean': float(m.group(3)),
            'zero_cand_mean': float(m.group(4)),
        }

    # Combine and create rows
    for step in sorted(eval_matches.keys()):
        e = eval_matches[step]
        p = progress_matches.get(step, {'filled_mean': 0, 'violations_mean': 0, 'zero_cand_mean': 0})

        rows.append({
            'algorithm': algorithm,
            'seed': seed,
            'step': step,
            'success_rate': e['success_rate'],
            'mean_score': e['mean_score'],
            'solved_count': e['solved_count'],
            'total_episodes': e['total_episodes'],
            'filled_mean': p['filled_mean'],
            'violations_mean': p['violations_mean'],
            'zero_cand_mean': p['zero_cand_mean'],
        })

    return rows


def main():
    results_dir = Path('/home/buiksat/trm_bellman/results/plot_data_trivial_zonly_contraction_FAST')
    algorithm = 'contraction_zonly_fast'

    all_rows = []

    for seed in [42, 123, 456]:
        log_path = results_dir / f'zonly_contraction_fast_seed{seed}.log'
        print(f"Parsing {log_path}...")
        rows = parse_log(str(log_path), algorithm, seed)
        all_rows.extend(rows)
        print(f"  Found {len(rows)} eval points")

    if not all_rows:
        print("No data found!")
        return

    # Write CSV
    csv_path = results_dir / 'plot_data_feasibility_learning_curves.csv'
    header = 'algorithm,seed,step,success_rate,mean_score,solved_count,total_episodes,filled_mean,violations_mean,zero_cand_mean'

    with open(csv_path, 'w') as f:
        f.write(header + '\n')
        for row in all_rows:
            line = ','.join([
                str(row['algorithm']),
                str(row['seed']),
                str(row['step']),
                str(row['success_rate']),
                str(row['mean_score']),
                str(row['solved_count']),
                str(row['total_episodes']),
                str(row['filled_mean']),
                str(row['violations_mean']),
                str(row['zero_cand_mean']),
            ])
            f.write(line + '\n')

    print(f"Wrote {len(all_rows)} rows to {csv_path}")

    # Print summary
    print("\nSummary (final step for each seed):")
    for seed in [42, 123, 456]:
        seed_rows = [r for r in all_rows if r['seed'] == seed]
        if seed_rows:
            final = seed_rows[-1]
            print(f"  Seed {seed}: step={final['step']}, success_rate={final['success_rate']:.3f}, mean_score={final['mean_score']:.2f}")
        else:
            print(f"  Seed {seed}: no data")


if __name__ == '__main__':
    main()
