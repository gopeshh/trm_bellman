#!/usr/bin/env python3
"""
Generate TRUE ultra-easy 4x4 Sudoku puzzles with 12-15 clues (1-4 empties).

This is for debugging and validating the RL setup before scaling to harder puzzles.
"""

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np


def generate_solved_4x4() -> np.ndarray:
    """Generate a valid solved 4x4 Sudoku grid."""
    grid = np.zeros((4, 4), dtype=np.int32)

    def is_valid(grid, row, col, num):
        if num in grid[row]:
            return False
        if num in grid[:, col]:
            return False
        box_row, box_col = 2 * (row // 2), 2 * (col // 2)
        if num in grid[box_row:box_row+2, box_col:box_col+2]:
            return False
        return True

    def solve(grid, pos=0):
        if pos == 16:
            return True
        row, col = pos // 4, pos % 4
        if grid[row, col] != 0:
            return solve(grid, pos + 1)

        nums = list(range(1, 5))
        random.shuffle(nums)
        for num in nums:
            if is_valid(grid, row, col, num):
                grid[row, col] = num
                if solve(grid, pos + 1):
                    return True
                grid[row, col] = 0
        return False

    solve(grid)
    return grid


def create_puzzle(solution: np.ndarray, num_clues: int) -> np.ndarray:
    """Remove cells from solution to create a puzzle with given number of clues."""
    puzzle = solution.copy()
    cells = list(range(16))
    random.shuffle(cells)

    cells_to_remove = 16 - num_clues
    for i in range(cells_to_remove):
        row, col = cells[i] // 4, cells[i] % 4
        puzzle[row, col] = 0

    return puzzle


def generate_ultra_easy_puzzles(
    num_puzzles: int,
    min_clues: int = 12,
    max_clues: int = 15,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate puzzle-solution pairs with 12-15 clues (1-4 empties)."""
    random.seed(seed)
    np.random.seed(seed)

    puzzles = []
    for _ in range(num_puzzles):
        solution = generate_solved_4x4()
        num_clues = random.randint(min_clues, max_clues)
        puzzle = create_puzzle(solution, num_clues)
        puzzles.append((puzzle, solution))

    return puzzles


def main():
    parser = argparse.ArgumentParser(description="Generate TRUE ultra-easy 4x4 Sudoku")
    parser.add_argument("--output-dir", type=str, default="data/sudoku-4x4-trivial")
    parser.add_argument("--num-puzzles", type=int, default=500)
    parser.add_argument("--min-clues", type=int, default=12, help="Minimum clues (default 12 = 4 empties)")
    parser.add_argument("--max-clues", type=int, default=15, help="Maximum clues (default 15 = 1 empty)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    empties_min = 16 - args.max_clues
    empties_max = 16 - args.min_clues
    print(f"Generating TRUE ultra-easy 4x4 Sudoku puzzles...")
    print(f"  Clues: {args.min_clues}-{args.max_clues}")
    print(f"  Empties: {empties_min}-{empties_max}")
    print(f"  Puzzles: {args.num_puzzles}")

    puzzles = generate_ultra_easy_puzzles(
        args.num_puzzles,
        min_clues=args.min_clues,
        max_clues=args.max_clues,
        seed=args.seed
    )

    # Convert to arrays with encoding: 0 -> 1 (empty), 1-4 -> 2-5 (digits)
    inputs_list = []
    labels_list = []
    for puzzle, solution in puzzles:
        puzzle_flat = puzzle.flatten()
        solution_flat = solution.flatten()
        inputs_encoded = np.where(puzzle_flat == 0, 1, puzzle_flat + 1)
        labels_encoded = solution_flat + 1
        inputs_list.append(inputs_encoded)
        labels_list.append(labels_encoded)

    inputs_arr = np.array(inputs_list, dtype=np.int32)
    labels_arr = np.array(labels_list, dtype=np.int32)

    # Verify empties distribution
    empties = np.sum(inputs_arr == 1, axis=1)
    print(f"\n  Empties distribution:")
    print(f"    Mean: {empties.mean():.2f}")
    print(f"    Min: {empties.min()}, Max: {empties.max()}")
    from collections import Counter
    for k, v in sorted(Counter(empties).items()):
        print(f"    {k} empties: {v} puzzles")

    # Split into train/test (90/10)
    n = len(inputs_arr)
    split_idx = int(0.9 * n)

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, start_idx, end_idx in [("train", 0, split_idx), ("test", split_idx, n)]:
        split_inputs = inputs_arr[start_idx:end_idx]
        split_labels = labels_arr[start_idx:end_idx]
        split_n = len(split_inputs)

        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "all__inputs.npy"), split_inputs)
        np.save(os.path.join(split_dir, "all__labels.npy"), split_labels)
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.arange(split_n, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.arange(split_n + 1, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.arange(split_n + 1, dtype=np.int32))

        metadata = {
            "seq_len": 16,
            "vocab_size": 6,
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": split_n,
            "total_groups": split_n,
            "mean_puzzle_examples": 1,
            "total_puzzles": split_n,
            "sets": ["all"]
        }
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)

        print(f"  {split_name}: {split_n} puzzles")

    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"\nDataset saved to {args.output_dir}")
    print(f"  This is TRUE ultra-easy: 1-4 empties (12-15 clues)")


if __name__ == "__main__":
    main()
