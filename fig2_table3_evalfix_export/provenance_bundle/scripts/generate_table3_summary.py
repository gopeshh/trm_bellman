#!/usr/bin/env python3
"""
Generate Table 3 summary from training logs.

Parses final eval_success_rate from each log file and generates
a markdown summary table.

Usage:
    python generate_table3_summary.py --results-dir results/table3_baselines_rerun_evalfix_2026_01_22
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics


def parse_final_success_rate(log_path: Path) -> Optional[float]:
    """
    Parse a training log file and extract the final eval_success_rate.

    Returns:
        Final success rate or None if not found
    """
    pattern = r"\[step (\d+)\] eval_success_rate=(\d+\.\d+)"
    last_rate = None

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                last_rate = float(match.group(2))

    return last_rate


def get_method_display_name(method_key: str) -> str:
    """Get display name for method."""
    names = {
        "persistent_nc": "UPI-TRM (Persistent-z, NC)",
        "episodic_nc": "UPI-TRM (Episodic-z, NC)",
        "episodic_c_clean": "UPI-TRM (Episodic-z, C)",
        "ppo": "PPO",
        "a2c": "A2C",
        "dqn": "DQN",
        "dqn_nstep5": "DQN (n=5)",
    }
    return names.get(method_key, method_key)


def main():
    parser = argparse.ArgumentParser(description="Generate Table 3 summary from training logs")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing training logs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: <results-dir>/table3_summary.md)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "table3_summary.md"

    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    # Method order (UPI-TRM variants first, then baselines)
    methods = [
        "persistent_nc",
        "episodic_nc",
        "episodic_c_clean",
        "ppo",
        "a2c",
        "dqn",
        "dqn_nstep5",
    ]
    seeds = [42, 123, 456]

    # Collect results
    results: Dict[str, Dict[int, Optional[float]]] = {}
    for method in methods:
        results[method] = {}
        for seed in seeds:
            log_file = results_dir / f"{method}_s{seed}.log"
            if log_file.exists():
                rate = parse_final_success_rate(log_file)
                results[method][seed] = rate
            else:
                results[method][seed] = None

    # Generate summary
    lines = []
    lines.append("# Table 3: Final Success Rates (Trivial 4×4 Sudoku, 1-4 empties)")
    lines.append("")
    lines.append(f"**Source:** `{results_dir}`")
    lines.append(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |")
    lines.append("|--------|---------|----------|----------|------------|")

    for method in methods:
        display_name = get_method_display_name(method)
        seed_values = []
        row_parts = [display_name]

        for seed in seeds:
            rate = results[method].get(seed)
            if rate is not None:
                row_parts.append(f"{rate:.3f}")
                seed_values.append(rate)
            else:
                row_parts.append("N/A")

        if len(seed_values) >= 2:
            mean = statistics.mean(seed_values)
            std = statistics.stdev(seed_values)
            row_parts.append(f"**{mean:.3f}** ± {std:.3f}")
        elif len(seed_values) == 1:
            row_parts.append(f"**{seed_values[0]:.3f}**")
        else:
            row_parts.append("N/A")

        lines.append("| " + " | ".join(row_parts) + " |")

    lines.append("")
    lines.append("## Per-Seed Details")
    lines.append("")

    for method in methods:
        display_name = get_method_display_name(method)
        lines.append(f"### {display_name}")
        for seed in seeds:
            rate = results[method].get(seed)
            log_file = results_dir / f"{method}_s{seed}.log"
            status = "✓" if log_file.exists() else "✗"
            rate_str = f"{rate:.3f}" if rate is not None else "N/A"
            lines.append(f"- Seed {seed}: {rate_str} {status}")
        lines.append("")

    # Write output
    output_content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output_content)

    print(output_content)
    print(f"\n--- Saved to: {output_path} ---")


if __name__ == "__main__":
    main()
