#!/usr/bin/env python3
"""
Generate 4x4 Sudoku puzzles for curriculum learning.

4x4 Sudoku uses digits 1-4 in a 4x4 grid with four 2x2 boxes.
Much simpler than 9x9 - great for initial RL training!

Encoding (same scheme as 9x9):
  - Token 0: PAD (invalid)
  - Token 1: Empty cell
  - Tokens 2-5: Digits 1-4

Usage:
    python dataset/build_4x4_sudoku.py --output-dir data/sudoku-4x4 --num-puzzles 1000
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
        # Check row
        if num in grid[row]:
            return False
        # Check column
        if num in grid[:, col]:
            return False
        # Check 2x2 box
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
        puzzle[row, col] = 0  # 0 = empty
    
    return puzzle


def generate_puzzles(
    num_puzzles: int,
    min_clues: int = 4,
    max_clues: int = 10,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate puzzle-solution pairs."""
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
    parser = argparse.ArgumentParser(description="Generate 4x4 Sudoku dataset")
    parser.add_argument("--output-dir", type=str, default="data/sudoku-4x4")
    parser.add_argument("--num-puzzles", type=int, default=1000)
    parser.add_argument("--num-easy", type=int, default=500, help="Puzzles with 8-10 clues")
    parser.add_argument("--num-medium", type=int, default=300, help="Puzzles with 6-7 clues")
    parser.add_argument("--num-hard", type=int, default=200, help="Puzzles with 4-5 clues")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print(f"Generating 4x4 Sudoku puzzles...")
    print(f"  Easy (8-10 clues): {args.num_easy}")
    print(f"  Medium (6-7 clues): {args.num_medium}")
    print(f"  Hard (4-5 clues): {args.num_hard}")
    
    all_puzzles = []
    
    # Generate puzzles by difficulty
    configs = [
        ("easy", args.num_easy, 8, 10),
        ("medium", args.num_medium, 6, 7),
        ("hard", args.num_hard, 4, 5),
    ]
    
    for name, count, min_clues, max_clues in configs:
        puzzles = generate_puzzles(count, min_clues, max_clues, seed=args.seed)
        all_puzzles.extend(puzzles)
        print(f"  Generated {count} {name} puzzles")
    
    # Shuffle
    random.seed(args.seed)
    random.shuffle(all_puzzles)
    
    # Convert to arrays with encoding: 0 -> 1 (empty), 1-4 -> 2-5 (digits)
    inputs_list = []
    labels_list = []
    for puzzle, solution in all_puzzles:
        # Flatten and encode
        puzzle_flat = puzzle.flatten()
        solution_flat = solution.flatten()
        
        # Encode: empty (0) -> 1, digits (1-4) -> 2-5
        inputs_encoded = np.where(puzzle_flat == 0, 1, puzzle_flat + 1)
        labels_encoded = solution_flat + 1  # Digits 1-4 -> 2-5
        
        inputs_list.append(inputs_encoded)
        labels_list.append(labels_encoded)
    
    inputs_arr = np.array(inputs_list, dtype=np.int32)
    labels_arr = np.array(labels_list, dtype=np.int32)
    
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
        
        # Save arrays
        np.save(os.path.join(split_dir, "all__inputs.npy"), split_inputs)
        np.save(os.path.join(split_dir, "all__labels.npy"), split_labels)
        # Each puzzle gets a unique identifier for puzzle embeddings
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.arange(split_n, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.arange(split_n + 1, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.arange(split_n + 1, dtype=np.int32))
        
        # Save metadata
        metadata = {
            "seq_len": 16,          # 4x4 = 16 cells
            "vocab_size": 6,        # PAD(0) + empty(1) + digits(2-5)
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": split_n,  # Unique ID per puzzle
            "total_groups": split_n,
            "mean_puzzle_examples": 1,
            "total_puzzles": split_n,
            "sets": ["all"]
        }
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)
        
        print(f"  {split_name}: {split_n} puzzles")
    
    # Save identifiers
    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"\n4x4 Sudoku dataset saved to {args.output_dir}")
    print(f"  seq_len: 16 (4x4 grid)")
    print(f"  vocab_size: 6 (PAD + empty + digits 1-4)")
    print(f"  action_space: 16 * 5 + 1 = 81 actions")


if __name__ == "__main__":
    main()

