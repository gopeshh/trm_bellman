"""
Shared Sudoku utilities for RL training and evaluation.

This module provides:
- Violation counting for 4x4 and 9x9 grids
- Filled cell counting
- Zero-candidate cell detection (feasibility)
- Solved state checking

These utilities are used by checkers in upi_trm_train.py and evaluators.
"""

import torch
from typing import Tuple, Set


def _grid_to_digits(grid: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Convert grid from token space to digit space.

    Token encoding:
    - Token 1 = empty (becomes 0)
    - Token 2..N+1 = digits 1..N

    Args:
        grid: Grid tensor in token space
        grid_size: 4 for 4x4, 9 for 9x9

    Returns:
        Grid in digit space (0 = empty, 1..N = digits)
    """
    digits = grid.clone()
    digits[digits == 1] = 0  # Empty cells
    digits = digits - 1  # Shift: token 2 -> digit 1, etc.
    digits = torch.clamp(digits, min=0)  # Ensure non-negative
    return digits


def count_sudoku_violations_4x4(grid: torch.Tensor) -> int:
    """
    Count constraint violations in a 4x4 Sudoku grid.

    A 4x4 Sudoku has:
    - 4 rows (each should have unique digits 1-4)
    - 4 columns (each should have unique digits 1-4)
    - 4 boxes of 2x2 (each should have unique digits 1-4)

    Violations are counted as the number of duplicate entries in each constraint.

    Args:
        grid: [16] or [4, 4] tensor with values in token space (1=empty, 2-5 = digits 1-4)

    Returns:
        Total number of constraint violations (0 = no violations)
    """
    grid = grid.reshape(4, 4)
    violations = 0

    digits = _grid_to_digits(grid, 4)

    # Check rows
    for r in range(4):
        row = digits[r, :]
        filled = row[row > 0]
        if len(filled) > 0:
            violations += len(filled) - len(torch.unique(filled))

    # Check columns
    for c in range(4):
        col = digits[:, c]
        filled = col[col > 0]
        if len(filled) > 0:
            violations += len(filled) - len(torch.unique(filled))

    # Check 2x2 boxes
    for box_r in range(2):
        for box_c in range(2):
            box = digits[box_r*2:(box_r+1)*2, box_c*2:(box_c+1)*2].reshape(-1)
            filled = box[box > 0]
            if len(filled) > 0:
                violations += len(filled) - len(torch.unique(filled))

    return violations


def count_sudoku_violations_9x9(grid: torch.Tensor) -> int:
    """
    Count constraint violations in a 9x9 Sudoku grid (vectorized).

    A 9x9 Sudoku has:
    - 9 rows (each should have unique digits 1-9)
    - 9 columns (each should have unique digits 1-9)
    - 9 boxes of 3x3 (each should have unique digits 1-9)

    Violations are counted as the number of duplicate entries in each constraint.

    Args:
        grid: [81] or [9, 9] tensor with values in token space (1=empty, 2-10 = digits 1-9)

    Returns:
        Total number of constraint violations (0 = no violations)
    """
    grid = grid.reshape(9, 9)
    digits = _grid_to_digits(grid, 9)

    # Create one-hot encoding for digits 1-9 (ignore 0 = empty)
    # Shape: [9, 9, 9] where last dim is one-hot for digits 1-9
    digits_flat = digits.reshape(-1)  # [81]
    one_hot = torch.zeros(81, 9, dtype=torch.int32, device=digits.device)
    filled_mask = digits_flat > 0
    one_hot[filled_mask, digits_flat[filled_mask].long() - 1] = 1
    one_hot = one_hot.reshape(9, 9, 9)  # [row, col, digit]

    violations = 0

    # Row violations: sum across columns for each row
    # Shape: [9, 9] -> counts per (row, digit)
    row_counts = one_hot.sum(dim=1)  # [9, 9]
    violations += int((row_counts - 1).clamp(min=0).sum().item())

    # Column violations: sum across rows for each column
    col_counts = one_hot.sum(dim=0)  # [9, 9]
    violations += int((col_counts - 1).clamp(min=0).sum().item())

    # Box violations: reshape to access 3x3 boxes
    # Reshape to [3, 3, 3, 3, 9] = [box_row, inner_row, box_col, inner_col, digit]
    box_view = one_hot.reshape(3, 3, 3, 3, 9)
    # Sum across inner_row and inner_col (dims 1 and 3)
    box_counts = box_view.sum(dim=(1, 3))  # [3, 3, 9]
    violations += int((box_counts - 1).clamp(min=0).sum().item())

    return violations


def sudoku_filled_cells(grid: torch.Tensor, empty_token: int = 1) -> int:
    """
    Count the number of filled (non-empty) cells in a Sudoku grid.

    Args:
        grid: Grid tensor in token space
        empty_token: Token value representing empty cell (default: 1)

    Returns:
        Number of filled cells
    """
    return int((grid != empty_token).sum().item())


def _get_candidates_4x4(digits: torch.Tensor, row: int, col: int) -> Set[int]:
    """
    Get candidate digits for a cell in a 4x4 Sudoku.

    Args:
        digits: 4x4 grid in digit space (0 = empty, 1-4 = digits)
        row: Row index (0-3)
        col: Column index (0-3)

    Returns:
        Set of valid candidate digits (1-4)
    """
    candidates = set(range(1, 5))  # Start with all digits 1-4

    # Remove digits in same row
    for c in range(4):
        d = int(digits[row, c].item())
        if d > 0:
            candidates.discard(d)

    # Remove digits in same column
    for r in range(4):
        d = int(digits[r, col].item())
        if d > 0:
            candidates.discard(d)

    # Remove digits in same 2x2 box
    box_r, box_c = (row // 2) * 2, (col // 2) * 2
    for r in range(box_r, box_r + 2):
        for c in range(box_c, box_c + 2):
            d = int(digits[r, c].item())
            if d > 0:
                candidates.discard(d)

    return candidates


def _get_candidates_9x9(digits: torch.Tensor, row: int, col: int) -> Set[int]:
    """
    Get candidate digits for a cell in a 9x9 Sudoku.

    Args:
        digits: 9x9 grid in digit space (0 = empty, 1-9 = digits)
        row: Row index (0-8)
        col: Column index (0-8)

    Returns:
        Set of valid candidate digits (1-9)
    """
    candidates = set(range(1, 10))  # Start with all digits 1-9

    # Remove digits in same row
    for c in range(9):
        d = int(digits[row, c].item())
        if d > 0:
            candidates.discard(d)

    # Remove digits in same column
    for r in range(9):
        d = int(digits[r, col].item())
        if d > 0:
            candidates.discard(d)

    # Remove digits in same 3x3 box
    box_r, box_c = (row // 3) * 3, (col // 3) * 3
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            d = int(digits[r, c].item())
            if d > 0:
                candidates.discard(d)

    return candidates


def sudoku_zero_candidate_cells(grid: torch.Tensor, grid_size: int = 4) -> int:
    """
    Count empty cells that have zero legal candidates (dead-end cells).

    A cell has zero candidates when:
    - It is empty
    - All possible digits (1-4 for 4x4, 1-9 for 9x9) are already present
      in the cell's row, column, or box

    These cells indicate an impossible state that cannot be completed.

    Args:
        grid: Grid tensor in token space
        grid_size: 4 for 4x4, 9 for 9x9

    Returns:
        Number of empty cells with zero candidates
    """
    if grid_size == 4:
        grid = grid.reshape(4, 4)
        digits = _grid_to_digits(grid, 4)
        zero_cand_count = 0

        for r in range(4):
            for c in range(4):
                if digits[r, c] == 0:  # Empty cell
                    candidates = _get_candidates_4x4(digits, r, c)
                    if len(candidates) == 0:
                        zero_cand_count += 1

        return zero_cand_count

    elif grid_size == 9:
        # Vectorized implementation for 9x9
        grid = grid.reshape(9, 9)
        digits = _grid_to_digits(grid, 9)
        device = digits.device

        # Build one-hot presence masks for digits 1-9
        # Shape: [9, 9, 9] where [r, c, d] = 1 if digit d+1 is at (r, c)
        digits_flat = digits.reshape(-1)  # [81]
        one_hot = torch.zeros(81, 9, dtype=torch.bool, device=device)
        filled_mask = digits_flat > 0
        one_hot[filled_mask, digits_flat[filled_mask].long() - 1] = True
        one_hot = one_hot.reshape(9, 9, 9)  # [row, col, digit]

        # Row blocked: for each row, which digits are present?
        # Shape: [9, 9] -> [row, digit]
        row_blocked = one_hot.any(dim=1)  # [9, 9]

        # Column blocked: for each column, which digits are present?
        # Shape: [9, 9] -> [col, digit]
        col_blocked = one_hot.any(dim=0)  # [9, 9]

        # Box blocked: for each box, which digits are present?
        # Reshape to [3, 3, 3, 3, 9] = [box_row, inner_row, box_col, inner_col, digit]
        box_view = one_hot.reshape(3, 3, 3, 3, 9)
        # Any across inner dims -> [3, 3, 9] = [box_row, box_col, digit]
        box_blocked = box_view.any(dim=(1, 3))  # [3, 3, 9]

        # For each cell, compute which digits are blocked (union of row, col, box)
        # We need to expand these to [9, 9, 9] = [row, col, digit]
        # row_blocked[r, d] -> expand to [r, :, d]
        row_expand = row_blocked.unsqueeze(1).expand(9, 9, 9)  # [9, 9, 9]
        # col_blocked[c, d] -> expand to [:, c, d]
        col_expand = col_blocked.unsqueeze(0).expand(9, 9, 9)  # [9, 9, 9]

        # box_blocked[br, bc, d] needs to map to cells in that box
        # Cell (r, c) is in box (r//3, c//3)
        # Create box index tensors
        box_row_idx = torch.arange(9, device=device) // 3  # [0,0,0,1,1,1,2,2,2]
        box_col_idx = torch.arange(9, device=device) // 3
        # For each cell position, lookup its box's blocked digits
        # box_expand[r, c, d] = box_blocked[r//3, c//3, d]
        box_expand = box_blocked[box_row_idx][:, box_col_idx]  # [9, 9, 9]

        # Union of all blocked digits per cell
        all_blocked = row_expand | col_expand | box_expand  # [9, 9, 9]

        # A cell has zero candidates if:
        # 1. It's empty (digits[r, c] == 0)
        # 2. All 9 digits are blocked
        all_digits_blocked = all_blocked.all(dim=2)  # [9, 9]
        empty_cells = (digits == 0)  # [9, 9]
        zero_cand_cells = empty_cells & all_digits_blocked

        return int(zero_cand_cells.sum().item())

    else:
        raise ValueError(f"Unsupported grid size: {grid_size}. Use 4 or 9.")


def sudoku_is_solved(grid: torch.Tensor, empty_token: int = 1) -> bool:
    """
    Check if a Sudoku grid is completely and correctly solved.

    A grid is solved if:
    - All cells are filled (no empty cells)
    - There are zero constraint violations

    This is a solution-independent success criterion that doesn't require
    knowing the ground truth solution.

    Args:
        grid: Grid tensor in token space
        empty_token: Token value representing empty cell (default: 1)

    Returns:
        True if grid is solved, False otherwise
    """
    total_cells = grid.numel()

    # Check all cells are filled
    filled = sudoku_filled_cells(grid, empty_token)
    if filled != total_cells:
        return False

    # Check no violations
    if total_cells == 16:
        violations = count_sudoku_violations_4x4(grid)
    elif total_cells == 81:
        violations = count_sudoku_violations_9x9(grid)
    else:
        # Unknown grid size - cannot verify
        return False

    return violations == 0


def sudoku_get_stats(grid: torch.Tensor, empty_token: int = 1) -> Tuple[int, int, int, int]:
    """
    Get all relevant statistics for a Sudoku grid.

    Args:
        grid: Grid tensor in token space
        empty_token: Token value representing empty cell (default: 1)

    Returns:
        Tuple of (total_cells, filled, violations, zero_cand_cells)
    """
    total_cells = grid.numel()
    filled = sudoku_filled_cells(grid, empty_token)

    if total_cells == 16:
        violations = count_sudoku_violations_4x4(grid)
        zero_cand = sudoku_zero_candidate_cells(grid, grid_size=4)
    elif total_cells == 81:
        violations = count_sudoku_violations_9x9(grid)
        zero_cand = sudoku_zero_candidate_cells(grid, grid_size=9)
    else:
        violations = 0
        zero_cand = 0

    return total_cells, filled, violations, zero_cand
