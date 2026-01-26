#!/usr/bin/env python3
"""
Generate 9x9 Sudoku puzzles with train/val/test splits.

9x9 Sudoku uses digits 1-9 in a 9x9 grid with nine 3x3 boxes.

Encoding scheme:
  - Token 0: PAD (invalid)
  - Token 1: Empty cell
  - Tokens 2-10: Digits 1-9

Usage:
    python scripts/gen_sudoku9x9.py --output-dir data/sudoku-9x9 \
        --num-train 10000 --num-val 1000 --num-test 1000 --seed 42
"""

import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np


def generate_solved_9x9() -> np.ndarray:
    """Generate a valid solved 9x9 Sudoku grid using backtracking with randomization."""
    grid = np.zeros((9, 9), dtype=np.int32)

    def is_valid(grid: np.ndarray, row: int, col: int, num: int) -> bool:
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in grid[:, col]:
            return False
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True

    def solve(grid: np.ndarray, pos: int = 0) -> bool:
        if pos == 81:
            return True
        row, col = pos // 9, pos % 9
        if grid[row, col] != 0:
            return solve(grid, pos + 1)

        nums = list(range(1, 10))
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


def count_solutions(grid: np.ndarray, limit: int = 2) -> int:
    """Count solutions up to limit (for checking uniqueness)."""
    grid = grid.copy()
    count = [0]

    def is_valid(grid: np.ndarray, row: int, col: int, num: int) -> bool:
        if num in grid[row]:
            return False
        if num in grid[:, col]:
            return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True

    def solve(pos: int = 0) -> bool:
        if count[0] >= limit:
            return True
        if pos == 81:
            count[0] += 1
            return count[0] >= limit
        row, col = pos // 9, pos % 9
        if grid[row, col] != 0:
            return solve(pos + 1)

        for num in range(1, 10):
            if is_valid(grid, row, col, num):
                grid[row, col] = num
                if solve(pos + 1):
                    return True
                grid[row, col] = 0
        return False

    solve()
    return count[0]


def create_puzzle_with_unique_solution(
    solution: np.ndarray,
    num_clues: int,
    max_attempts: int = 100
) -> Optional[np.ndarray]:
    """
    Remove cells from solution to create a puzzle with given number of clues.
    Ensures unique solution by checking after each removal.
    """
    for _ in range(max_attempts):
        puzzle = solution.copy()
        cells = list(range(81))
        random.shuffle(cells)

        removed = 0
        target_removals = 81 - num_clues

        for cell_idx in cells:
            if removed >= target_removals:
                break
            row, col = cell_idx // 9, cell_idx % 9
            old_val = puzzle[row, col]
            puzzle[row, col] = 0

            # Check if still has unique solution
            if count_solutions(puzzle, limit=2) == 1:
                removed += 1
            else:
                # Restore if not unique
                puzzle[row, col] = old_val

        if removed >= target_removals:
            return puzzle

    return None


def create_puzzle_fast(solution: np.ndarray, num_clues: int) -> np.ndarray:
    """
    Fast puzzle creation without uniqueness checking.
    Use for generating larger datasets where some puzzles may have multiple solutions.
    """
    puzzle = solution.copy()
    cells = list(range(81))
    random.shuffle(cells)

    cells_to_remove = 81 - num_clues
    for i in range(cells_to_remove):
        row, col = cells[i] // 9, cells[i] % 9
        puzzle[row, col] = 0

    return puzzle


def generate_puzzles(
    num_puzzles: int,
    min_clues: int = 17,
    max_clues: int = 35,
    seed: int = 42,
    ensure_unique: bool = False,
    verbose: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate puzzle-solution pairs."""
    random.seed(seed)
    np.random.seed(seed)

    puzzles = []
    attempts = 0
    max_total_attempts = num_puzzles * 10

    while len(puzzles) < num_puzzles and attempts < max_total_attempts:
        attempts += 1
        solution = generate_solved_9x9()
        num_clues = random.randint(min_clues, max_clues)

        if ensure_unique:
            puzzle = create_puzzle_with_unique_solution(solution, num_clues)
            if puzzle is None:
                continue
        else:
            puzzle = create_puzzle_fast(solution, num_clues)

        puzzles.append((puzzle, solution))

        if verbose and len(puzzles) % 100 == 0:
            print(f"  Generated {len(puzzles)}/{num_puzzles} puzzles...")

    return puzzles


def save_split(
    puzzles: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    split_name: str
) -> None:
    """Save a split of puzzles to disk."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    inputs_list = []
    labels_list = []

    for puzzle, solution in puzzles:
        # Flatten and encode
        puzzle_flat = puzzle.flatten()
        solution_flat = solution.flatten()

        # Encode: empty (0) -> 1, digits (1-9) -> 2-10
        inputs_encoded = np.where(puzzle_flat == 0, 1, puzzle_flat + 1)
        labels_encoded = solution_flat + 1  # Digits 1-9 -> 2-10

        inputs_list.append(inputs_encoded)
        labels_list.append(labels_encoded)

    inputs_arr = np.array(inputs_list, dtype=np.int32)
    labels_arr = np.array(labels_list, dtype=np.int32)
    n = len(inputs_arr)

    # Save arrays
    np.save(os.path.join(split_dir, "all__inputs.npy"), inputs_arr)
    np.save(os.path.join(split_dir, "all__labels.npy"), labels_arr)
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"),
            np.arange(n, dtype=np.int32))
    np.save(os.path.join(split_dir, "all__puzzle_indices.npy"),
            np.arange(n + 1, dtype=np.int32))
    np.save(os.path.join(split_dir, "all__group_indices.npy"),
            np.arange(n + 1, dtype=np.int32))

    # Save metadata
    metadata = {
        "seq_len": 81,           # 9x9 = 81 cells
        "vocab_size": 11,        # PAD(0) + empty(1) + digits(2-10)
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": n,
        "total_groups": n,
        "mean_puzzle_examples": 1,
        "total_puzzles": n,
        "sets": ["all"]
    }
    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {split_name}: {n} puzzles saved to {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 9x9 Sudoku dataset with train/val/test splits"
    )
    parser.add_argument("--output-dir", type=str, default="data/sudoku-9x9",
                        help="Output directory for dataset")
    parser.add_argument("--num-train", type=int, default=10000,
                        help="Number of training puzzles")
    parser.add_argument("--num-val", type=int, default=1000,
                        help="Number of validation puzzles")
    parser.add_argument("--num-test", type=int, default=1000,
                        help="Number of test puzzles")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--ensure-unique", action="store_true",
                        help="Ensure each puzzle has a unique solution (slower)")

    # Difficulty settings (by number of clues)
    parser.add_argument("--easy-ratio", type=float, default=0.3,
                        help="Ratio of easy puzzles (30-35 clues)")
    parser.add_argument("--medium-ratio", type=float, default=0.4,
                        help="Ratio of medium puzzles (24-29 clues)")
    parser.add_argument("--hard-ratio", type=float, default=0.3,
                        help="Ratio of hard puzzles (17-23 clues)")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.easy_ratio + args.medium_ratio + args.hard_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: ratios sum to {total_ratio}, normalizing...")
        args.easy_ratio /= total_ratio
        args.medium_ratio /= total_ratio
        args.hard_ratio /= total_ratio

    print(f"Generating 9x9 Sudoku dataset...")
    print(f"  Train: {args.num_train}, Val: {args.num_val}, Test: {args.num_test}")
    print(f"  Difficulty distribution:")
    print(f"    Easy (30-35 clues): {args.easy_ratio*100:.0f}%")
    print(f"    Medium (24-29 clues): {args.medium_ratio*100:.0f}%")
    print(f"    Hard (17-23 clues): {args.hard_ratio*100:.0f}%")
    print(f"  Ensure unique solutions: {args.ensure_unique}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Difficulty configurations: (name, min_clues, max_clues)
    difficulty_configs = [
        ("easy", 30, 35),
        ("medium", 24, 29),
        ("hard", 17, 23),
    ]
    ratios = [args.easy_ratio, args.medium_ratio, args.hard_ratio]

    # Generate each split
    for split_name, num_total in [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test)
    ]:
        print(f"Generating {split_name} split ({num_total} puzzles)...")

        # Use different seed offsets for each split to ensure independence
        seed_offset = {"train": 0, "val": 10000, "test": 20000}[split_name]

        all_puzzles = []
        for (name, min_clues, max_clues), ratio in zip(difficulty_configs, ratios):
            count = int(num_total * ratio)
            if count == 0:
                continue

            print(f"  Generating {count} {name} puzzles ({min_clues}-{max_clues} clues)...")
            puzzles = generate_puzzles(
                count,
                min_clues=min_clues,
                max_clues=max_clues,
                seed=args.seed + seed_offset + hash(name) % 1000,
                ensure_unique=args.ensure_unique,
                verbose=True
            )
            all_puzzles.extend(puzzles)

        # Shuffle with split-specific seed
        random.seed(args.seed + seed_offset)
        random.shuffle(all_puzzles)

        # Save split
        save_split(all_puzzles, args.output_dir, split_name)

    # Save identifiers file (for compatibility)
    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    # Save dataset info
    dataset_info = {
        "grid_size": 9,
        "seq_len": 81,
        "vocab_size": 11,
        "action_space": 81 * 10 + 1,  # 81 cells * 10 values (0-9) + noop
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "seed": args.seed,
        "ensure_unique": args.ensure_unique,
        "difficulty_ratios": {
            "easy": args.easy_ratio,
            "medium": args.medium_ratio,
            "hard": args.hard_ratio
        }
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print()
    print(f"9x9 Sudoku dataset saved to {args.output_dir}")
    print(f"  seq_len: 81 (9x9 grid)")
    print(f"  vocab_size: 11 (PAD + empty + digits 1-9)")
    print(f"  action_space: 81 * 10 + 1 = 811 actions")
    print()
    print("To use with training:")
    print(f"  python upi_trm_train.py --config configs/sudoku9x9/upi_trm_9x9.yaml")


if __name__ == "__main__":
    main()
