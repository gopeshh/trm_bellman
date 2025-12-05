"""
Build an easier Sudoku dataset for curriculum learning.
Generates Sudoku puzzles with varying difficulty levels based on number of given clues.

Easy puzzles have more clues (35-45 given), making them solvable with simpler strategies.
Medium puzzles have 27-35 clues, and hard puzzles have fewer.
"""
import os
import json
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


def is_valid_sudoku(board: np.ndarray) -> bool:
    """Check if a Sudoku solution is valid."""
    for i in range(9):
        row = board[i, :]
        col = board[:, i]
        if len(set(row)) != 9 or len(set(col)) != 9:
            return False
        if not all(1 <= x <= 9 for x in row) or not all(1 <= x <= 9 for x in col):
            return False
    
    for box_row in range(3):
        for box_col in range(3):
            box = board[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            if len(set(box)) != 9 or not all(1 <= x <= 9 for x in box):
                return False
    return True


def generate_filled_board() -> np.ndarray:
    """Generate a complete valid Sudoku board using backtracking."""
    board = np.zeros((9, 9), dtype=np.int32)
    
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False
        # Check column
        if num in board[:, col]:
            return False
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True
    
    def solve(board):
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    nums = list(range(1, 10))
                    np.random.shuffle(nums)
                    for num in nums:
                        if is_valid(board, row, col, num):
                            board[row, col] = num
                            if solve(board):
                                return True
                            board[row, col] = 0
                    return False
        return True
    
    solve(board)
    return board


def create_puzzle(solution: np.ndarray, num_clues: int) -> np.ndarray:
    """Create a puzzle from a solution by removing cells."""
    puzzle = solution.copy()
    cells_to_remove = 81 - num_clues
    
    # Get all cell positions and shuffle
    positions = [(i, j) for i in range(9) for j in range(9)]
    np.random.shuffle(positions)
    
    removed = 0
    for row, col in positions:
        if removed >= cells_to_remove:
            break
        puzzle[row, col] = 0
        removed += 1
    
    return puzzle


def build_easy_dataset(
    output_dir: str,
    num_easy: int = 500,
    num_medium: int = 300,
    num_hard: int = 200,
    num_aug: int = 5,
    seed: int = 42,
):
    """
    Build a curriculum dataset with puzzles of varying difficulty.
    
    Args:
        output_dir: Where to save the dataset
        num_easy: Number of easy puzzles (40-50 clues)
        num_medium: Number of medium puzzles (32-40 clues)
        num_hard: Number of hard puzzles (25-32 clues)
        num_aug: Number of augmentations per puzzle
        seed: Random seed
    """
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    difficulties = [
        ("easy", num_easy, 40, 50),      # Easy: 40-50 clues (31-41 empty)
        ("medium", num_medium, 32, 40),  # Medium: 32-40 clues (41-49 empty)
        ("hard", num_hard, 25, 32),      # Hard: 25-32 clues (49-56 empty)
    ]
    
    all_inputs = []
    all_labels = []
    all_difficulties = []
    
    print("Generating puzzles...")
    for diff_name, count, min_clues, max_clues in difficulties:
        print(f"  Generating {count} {diff_name} puzzles ({min_clues}-{max_clues} clues)...")
        for _ in tqdm(range(count), desc=diff_name):
            solution = generate_filled_board()
            if not is_valid_sudoku(solution):
                continue
            
            num_clues = np.random.randint(min_clues, max_clues + 1)
            puzzle = create_puzzle(solution, num_clues)
            
            # Store original and augmented versions
            for aug_idx in range(1 + num_aug):
                if aug_idx == 0:
                    inp, sol = puzzle, solution
                else:
                    inp, sol = shuffle_sudoku(puzzle, solution)
                
                all_inputs.append(inp)
                all_labels.append(sol)
                all_difficulties.append(diff_name)
    
    # Convert to numpy arrays (add 1 to match original encoding: 0=empty becomes 1)
    inputs_arr = np.array(all_inputs, dtype=np.int32).reshape(len(all_inputs), -1) + 1
    labels_arr = np.array(all_labels, dtype=np.int32).reshape(len(all_labels), -1) + 1
    
    # Shuffle the dataset but keep track of original order for stratified splits
    indices = np.arange(len(inputs_arr))
    np.random.shuffle(indices)
    inputs_arr = inputs_arr[indices]
    labels_arr = labels_arr[indices]
    
    # Split into train/test (90/10)
    split_idx = int(0.9 * len(inputs_arr))
    
    for split_name, start_idx, end_idx in [("train", 0, split_idx), ("test", split_idx, len(inputs_arr))]:
        split_inputs = inputs_arr[start_idx:end_idx]
        split_labels = labels_arr[start_idx:end_idx]
        n = len(split_inputs)
        
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(split_dir, "all__inputs.npy"), split_inputs)
        np.save(os.path.join(split_dir, "all__labels.npy"), split_labels)
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.zeros(n, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.arange(n + 1, dtype=np.int32))
        
        # Save metadata
        metadata = {
            "seq_len": 81,
            "vocab_size": 11,
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": 1,
            "total_groups": n,
            "mean_puzzle_examples": 1,
            "total_puzzles": n,
            "sets": ["all"]
        }
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)
        
        print(f"  {split_name}: {n} puzzles")
    
    # Save identifiers
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"Total puzzles: {len(inputs_arr)} ({split_idx} train, {len(inputs_arr) - split_idx} test)")


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random valid transformations to a Sudoku puzzle."""
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.concatenate([[0], np.random.permutation(np.arange(1, 10))])
    
    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        result = x.reshape(9, 9).copy()
        if transpose_flag:
            result = result.T
        # Apply row/col permutation
        result = result[row_perm][:, col_perm]
        # Apply digit mapping
        return digit_map[result]

    return apply_transformation(board), apply_transformation(solution)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sudoku-easy-curriculum")
    parser.add_argument("--num-easy", type=int, default=500)
    parser.add_argument("--num-medium", type=int, default=300)
    parser.add_argument("--num-hard", type=int, default=200)
    parser.add_argument("--num-aug", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    build_easy_dataset(
        output_dir=args.output_dir,
        num_easy=args.num_easy,
        num_medium=args.num_medium,
        num_hard=args.num_hard,
        num_aug=args.num_aug,
        seed=args.seed,
    )

