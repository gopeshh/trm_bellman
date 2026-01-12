#!/usr/bin/env python3
"""Parse 6-8 empties all ablations logs and generate CSVs for plotting."""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path("runs/feasibility_6to8empties_all_ablations_20k")
OUTPUT_DIR = Path("results/plot_data_6to8empties_all_ablations_20k")

ALGORITHMS = [
    "upi_trm",
    "no_contraction",
    "ablation_persistent_z",
    "persistent_z_no_contraction",
    "ablation_no_conservative",
    "a2c",
    "dqn",
]

SEEDS = [42, 123, 456]

def parse_log(log_path):
    """Parse a single log file and extract eval metrics."""
    data = []
    pattern = re.compile(
        r"\[step (\d+)\] eval_success_rate=([0-9.]+) eval_mean_score=([0-9.-]+)"
    )

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    step = int(match.group(1))
                    success_rate = float(match.group(2))
                    mean_score = float(match.group(3))
                    data.append({
                        'step': step,
                        'success_rate': success_rate,
                        'mean_score': mean_score
                    })
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return data

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    final_results = []

    for algo in ALGORITHMS:
        algo_dir = LOG_DIR / algo
        if not algo_dir.exists():
            print(f"Warning: {algo_dir} does not exist")
            continue

        for seed in SEEDS:
            # Find log file for this seed
            log_files = list(algo_dir.glob(f"{seed}_*.log"))
            if not log_files:
                print(f"Warning: No log for {algo} seed={seed}")
                continue

            log_path = log_files[0]
            data = parse_log(log_path)

            if data:
                key = f"{algo}_seed{seed}"
                all_results[key] = data

                # Get final result
                final = data[-1]
                final_results.append({
                    'algorithm': algo,
                    'seed': seed,
                    'final_step': final['step'],
                    'final_success_rate': final['success_rate'],
                    'final_mean_score': final['mean_score']
                })

                print(f"{algo} seed={seed}: {final['success_rate']*100:.1f}% success, score={final['mean_score']:.2f}")

    # Write per-algorithm CSVs
    for algo in ALGORITHMS:
        algo_data = []
        for seed in SEEDS:
            key = f"{algo}_seed{seed}"
            if key in all_results:
                for row in all_results[key]:
                    algo_data.append({
                        'step': row['step'],
                        'seed': seed,
                        'success_rate': row['success_rate'],
                        'mean_score': row['mean_score']
                    })

        if algo_data:
            csv_path = OUTPUT_DIR / f"{algo}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['step', 'seed', 'success_rate', 'mean_score'])
                writer.writeheader()
                writer.writerows(algo_data)
            print(f"Wrote {csv_path}")

    # Write summary CSV
    summary_path = OUTPUT_DIR / "summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['algorithm', 'seed', 'final_step', 'final_success_rate', 'final_mean_score'])
        writer.writeheader()
        writer.writerows(final_results)
    print(f"Wrote {summary_path}")

    # Calculate and print mean results per algorithm
    print("\n=== Mean Results by Algorithm ===")
    algo_means = defaultdict(list)
    for r in final_results:
        algo_means[r['algorithm']].append(r['final_success_rate'])

    for algo in ALGORITHMS:
        if algo in algo_means:
            rates = algo_means[algo]
            mean_rate = sum(rates) / len(rates)
            print(f"{algo}: {mean_rate*100:.1f}% mean success ({len(rates)} seeds)")

if __name__ == "__main__":
    main()
