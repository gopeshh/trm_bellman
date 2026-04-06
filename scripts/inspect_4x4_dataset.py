#!/usr/bin/env python3
"""
Inspect 4x4 Sudoku datasets and verify their empties ranges.

Empties are represented by token==1 (the empty/mask token).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = [
    "buiksat_trm/data/sudoku-4x4-trivial",
    "buiksat_trm/data/sudoku-4x4-ultra-easy",
    "buiksat_trm/data/sudoku-4x4-easy_6to8empties",
]


def expected_range_for_path(data_dir: Path) -> tuple[int, int, str] | None:
    name = data_dir.name
    if name == "sudoku-4x4-trivial":
        return (1, 4, "true ultra-easy")
    if name in {"sudoku-4x4-ultra-easy", "sudoku-4x4-easy_6to8empties"}:
        label = (
            "legacy compatibility split"
            if name == "sudoku-4x4-ultra-easy"
            else "paper hard split"
        )
        return (6, 8, label)
    return None


def count_empties(inputs: np.ndarray, empty_token: int = 1) -> np.ndarray:
    """Count empty cells per puzzle. inputs shape: (N, 16) for flattened 4x4."""
    return np.sum(inputs == empty_token, axis=1)


def analyze_dataset(data_dir: Path, split: str = "train") -> dict[str, object] | None:
    """Analyze a dataset split."""
    inputs_path = data_dir / split / "all__inputs.npy"

    if not inputs_path.exists():
        print(f"  {split}: NOT FOUND")
        return None

    inputs = np.load(inputs_path)

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

    expected = expected_range_for_path(data_dir)
    if expected is not None:
        expected_min, expected_max, description = expected
        in_range = np.sum((empties >= expected_min) & (empties <= expected_max))
        pct = 100 * in_range / len(empties)
        expected_range_ok = in_range == len(empties)
        status = "PASS" if expected_range_ok else "FAIL"
        print(f"\n  Expected range ({description}): {expected_min}-{expected_max} empties")
        print(f"    In range: {in_range}/{len(empties)} ({pct:.1f}%) -> {status}")
    else:
        in_range = None
        expected_range_ok = None

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
        "in_range": in_range,
        "expected_range_ok": expected_range_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect 4x4 Sudoku dataset empties ranges")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    print("=" * 60)
    print("4x4 SUDOKU DATASET INSPECTION")
    print("=" * 60)

    had_failure = False

    for data_path in args.datasets:
        data_dir = Path(data_path)
        if not data_dir.exists():
            print(f"\n{data_path}: NOT FOUND")
            had_failure = True
            continue

        print(f"\n{'=' * 60}")
        print(f"DATASET: {data_path}")
        print("=" * 60)

        for split in ["train", "test"]:
            result = analyze_dataset(data_dir, split)
            if result is None:
                had_failure = True
                continue
            if result["expected_range_ok"] is False:
                had_failure = True

    print("\n" + "=" * 60)
    print("REFERENCE")
    print("=" * 60)
    print("""
Expected dataset identities:
  - sudoku-4x4-trivial: true ultra-easy, 1-4 empties
  - sudoku-4x4-ultra-easy: legacy compatibility path, 6-8 empties
  - sudoku-4x4-easy_6to8empties: paper hard split, 6-8 empties

The old ultra-easy name is historical baggage: the true easy curriculum
dataset is sudoku-4x4-trivial.
""")

    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
