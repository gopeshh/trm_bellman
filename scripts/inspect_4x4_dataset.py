#!/usr/bin/env python3
"""
Inspect 4x4 Sudoku dataset to verify difficulty (count empties per puzzle).

Empties are represented by token==1 (the empty/mask token).
For 4x4 "ultra-easy", we expect 1-4 empties (12-15 clues).
"""

import numpy as np
from pathlib import Path
from collections import Counter

def count_empties(inputs: np.ndarray, empty_token: int = 1) -> np.ndarray:
    """Count empty cells per puzzle. inputs shape: (N, 16) for flattened 4x4."""
    return np.sum(inputs == empty_token, axis=1)

def analyze_dataset(data_dir: Path, split: str = "train"):
    """Analyze a dataset split."""
    inputs_path = data_dir / split / "all__inputs.npy"
    labels_path = data_dir / split / "all__labels.npy"

    if not inputs_path.exists():
        print(f"  {split}: NOT FOUND")
        return None

    inputs = np.load(inputs_path)
    labels = np.load(labels_path) if labels_path.exists() else None

    print(f"\n=== {split.upper()} ===")
    print(f"  Shape: {inputs.shape}")
    print(f"  Dtype: {inputs.dtype}")

    # Count empties
    empties = count_empties(inputs)
    filled = 16 - empties  # 4x4 = 16 cells

    print(f"\n  Empties per puzzle:")
    print(f"    Mean: {empties.mean():.2f}")
    print(f"    Median: {np.median(empties):.2f}")
    print(f"    Min: {empties.min()}")
    print(f"    Max: {empties.max()}")
    print(f"    Std: {empties.std():.2f}")

    print(f"\n  Filled cells (clues) per puzzle:")
    print(f"    Mean: {filled.mean():.2f}")
    print(f"    Min: {filled.min()}")
    print(f"    Max: {filled.max()}")

    # Histogram
    print(f"\n  Empties histogram:")
    counts = Counter(empties)
    for k in sorted(counts.keys()):
        pct = 100 * counts[k] / len(empties)
        bar = "#" * int(pct / 2)
        print(f"    {k:2d} empties: {counts[k]:4d} ({pct:5.1f}%) {bar}")

    # Check expected range
    ultra_easy_expected = (1, 4)
    in_range = np.sum((empties >= ultra_easy_expected[0]) & (empties <= ultra_easy_expected[1]))
    print(f"\n  Puzzles with 1-4 empties (ultra-easy): {in_range}/{len(empties)} ({100*in_range/len(empties):.1f}%)")

    # Sample a few puzzles
    print(f"\n  Sample puzzles (first 3):")
    for i in range(min(3, len(inputs))):
        grid = inputs[i].reshape(4, 4)
        empty_count = np.sum(grid == 1)
        print(f"    Puzzle {i}: {empty_count} empties")
        print(f"      {grid[0]}")
        print(f"      {grid[1]}")
        print(f"      {grid[2]}")
        print(f"      {grid[3]}")

    return {
        "n_puzzles": len(inputs),
        "mean_empties": empties.mean(),
        "median_empties": np.median(empties),
        "min_empties": empties.min(),
        "max_empties": empties.max(),
        "histogram": dict(counts),
    }

def main():
    datasets = [
        "data/sudoku-4x4-ultra-easy",
        "data/sudoku-4x4",
    ]

    print("=" * 60)
    print("4x4 SUDOKU DATASET INSPECTION")
    print("=" * 60)

    for data_path in datasets:
        data_dir = Path(data_path)
        if not data_dir.exists():
            print(f"\n{data_path}: NOT FOUND")
            continue

        print(f"\n{'=' * 60}")
        print(f"DATASET: {data_path}")
        print("=" * 60)

        for split in ["train", "test"]:
            analyze_dataset(data_dir, split)

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("""
If mean empties >> 4, the dataset is NOT "ultra-easy".
Expected for ultra-easy: 1-4 empties (12-15 clues).

The eval logs showed initial_score ≈ 8.84, suggesting:
  - initial filled cells ≈ 8.84 (feasibility score = filled)
  - empties ≈ 16 - 8.84 ≈ 7.16 (NOT ultra-easy!)

This would explain 0% success: with ~7 empties and strict
feasibility penalties, the agent gets stuck in bad states.

Fix: Generate true ultra-easy dataset with 1-4 empties.
""")

if __name__ == "__main__":
    main()
