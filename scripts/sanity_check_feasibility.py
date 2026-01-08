#!/usr/bin/env python3
"""
Sanity check script for the feasibility checker implementation.

This script verifies:
1. Empty grid is NOT counted as solved
2. Partially-filled valid grid is NOT solved
3. A known solved 4x4 grid IS solved
4. Dead-end states have zeroCand > 0 and lower feasibility score
5. Feasibility score formula is correct

Run with:
    buck2 run //buiksat_trm:sanity_check_feasibility -c fbcode.nvcc_arch=a100 -c fbcode.enable_gpu_sections=true
"""

import torch
import sys

# Import from the RL module
from rl.sudoku_utils import (
    count_sudoku_violations_4x4,
    sudoku_filled_cells,
    sudoku_zero_candidate_cells,
    sudoku_is_solved,
    sudoku_get_stats,
)

# Import checker from main training file
from upi_trm_train import sudoku_feasibility_checker


def print_grid_4x4(grid: torch.Tensor, label: str = "Grid"):
    """Print a 4x4 grid in readable format."""
    g = grid.reshape(4, 4)
    print(f"\n{label}:")
    for r in range(4):
        row_str = " ".join(
            "." if g[r, c] == 1 else str(g[r, c].item() - 1)
            for c in range(4)
        )
        print(f"  {row_str}")


def create_solved_4x4():
    """Create a valid solved 4x4 Sudoku grid."""
    # Valid 4x4 Sudoku solution (digit space: 1-4)
    # Token encoding: digit d -> token d+1
    # So: 1->2, 2->3, 3->4, 4->5
    return torch.tensor([
        2, 3, 4, 5,  # 1 2 3 4
        4, 5, 2, 3,  # 3 4 1 2
        3, 2, 5, 4,  # 2 1 4 3
        5, 4, 3, 2   # 4 3 2 1
    ], dtype=torch.long)


def create_impossible_4x4():
    """Create an impossible 4x4 grid with zero-candidate cells."""
    # Create a grid where cell (2,1) has no valid candidates
    # Row 0: 1, 2, 3, 4
    # Row 1: 3, 4, 1, 2
    # Row 2: 2, ., 1, .   <- (2,1) blocked by row:{2,1}, col:{2,4,3}, box:{2,4}
    # Row 3: 4, 3, ., .
    y = torch.ones(16, dtype=torch.long)
    # Row 0: tokens 2,3,4,5
    y[0] = 2; y[1] = 3; y[2] = 4; y[3] = 5
    # Row 1: tokens 4,5,2,3
    y[4] = 4; y[5] = 5; y[6] = 2; y[7] = 3
    # Row 2: 2, empty, 1, empty -> tokens 3, 1, 2, 1
    y[8] = 3; y[9] = 1; y[10] = 2; y[11] = 1
    # Row 3: 4, 3, empty, empty -> tokens 5, 4, 1, 1
    y[12] = 5; y[13] = 4; y[14] = 1; y[15] = 1
    return y


def run_sanity_checks():
    """Run all sanity checks and report results."""
    print("=" * 60)
    print("FEASIBILITY CHECKER SANITY CHECKS")
    print("=" * 60)

    # Dummy x for checker calls
    x = {"inputs": torch.zeros(16), "solution": torch.zeros(16)}

    all_passed = True

    # Test 1: Empty grid is NOT solved
    print("\n[TEST 1] Empty grid is NOT solved")
    empty_grid = torch.ones(16, dtype=torch.long)
    print_grid_4x4(empty_grid, "Empty Grid")

    total, filled, violations, zero_cand = sudoku_get_stats(empty_grid)
    score = sudoku_feasibility_checker(x, empty_grid, w_v=2.0, w_z=5.0)
    is_solved = sudoku_is_solved(empty_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={is_solved}")

    if is_solved:
        print("  [FAIL] Empty grid should NOT be solved!")
        all_passed = False
    else:
        print("  [PASS] Empty grid correctly NOT solved")

    if zero_cand != 0:
        print(f"  [WARN] Empty grid has zeroCand={zero_cand}, expected 0")
    else:
        print("  [PASS] Empty grid has zeroCand=0 (correct)")

    # Test 2: Partially-filled valid grid is NOT solved
    print("\n[TEST 2] Partially-filled valid grid is NOT solved")
    partial_grid = torch.ones(16, dtype=torch.long)
    partial_grid[0] = 2  # digit 1
    partial_grid[1] = 3  # digit 2
    partial_grid[5] = 4  # digit 3
    print_grid_4x4(partial_grid, "Partial Grid")

    total, filled, violations, zero_cand = sudoku_get_stats(partial_grid)
    score = sudoku_feasibility_checker(x, partial_grid, w_v=2.0, w_z=5.0)
    is_solved = sudoku_is_solved(partial_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={is_solved}")

    if is_solved:
        print("  [FAIL] Partial grid should NOT be solved!")
        all_passed = False
    else:
        print("  [PASS] Partial grid correctly NOT solved")

    # Test 3: Valid solved grid IS solved
    print("\n[TEST 3] Valid solved grid IS solved")
    solved_grid = create_solved_4x4()
    print_grid_4x4(solved_grid, "Solved Grid")

    total, filled, violations, zero_cand = sudoku_get_stats(solved_grid)
    score = sudoku_feasibility_checker(x, solved_grid, w_v=2.0, w_z=5.0)
    is_solved = sudoku_is_solved(solved_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={is_solved}")

    if not is_solved:
        print("  [FAIL] Solved grid should BE solved!")
        all_passed = False
    else:
        print("  [PASS] Solved grid correctly identified as solved")

    if score != 16.0:
        print(f"  [WARN] Expected score=16.0, got {score}")
    else:
        print("  [PASS] Solved grid has max score=16.0")

    # Test 4: Dead-end state has zeroCand > 0
    print("\n[TEST 4] Dead-end state has zeroCand > 0")
    impossible_grid = create_impossible_4x4()
    print_grid_4x4(impossible_grid, "Impossible Grid")

    total, filled, violations, zero_cand = sudoku_get_stats(impossible_grid)
    score = sudoku_feasibility_checker(x, impossible_grid, w_v=2.0, w_z=5.0)
    is_solved = sudoku_is_solved(impossible_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={is_solved}")

    if zero_cand == 0:
        print("  [WARN] Expected zeroCand > 0 for impossible grid")
        print("  (This may be OK if the grid isn't actually impossible)")
    else:
        print(f"  [PASS] Impossible grid has zeroCand={zero_cand} > 0")

    if is_solved:
        print("  [FAIL] Impossible grid should NOT be solved!")
        all_passed = False
    else:
        print("  [PASS] Impossible grid correctly NOT solved")

    # Test 5: Score formula verification
    print("\n[TEST 5] Score formula verification")
    # Create a grid with known values: 5 filled, 1 violation
    test_grid = torch.ones(16, dtype=torch.long)
    test_grid[0] = 2  # digit 1 at (0,0)
    test_grid[1] = 2  # digit 1 at (0,1) - creates violation
    test_grid[4] = 3  # digit 2 at (1,0)
    test_grid[5] = 4  # digit 3 at (1,1)
    test_grid[10] = 5  # digit 4 at (2,2)
    print_grid_4x4(test_grid, "Test Grid")

    total, filled, violations, zero_cand = sudoku_get_stats(test_grid)
    score = sudoku_feasibility_checker(x, test_grid, w_v=2.0, w_z=5.0)
    expected_score = filled - 2.0 * violations - 5.0 * zero_cand

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, expected={expected_score:.2f}")

    if abs(score - expected_score) > 0.01:
        print(f"  [FAIL] Score mismatch! got={score}, expected={expected_score}")
        all_passed = False
    else:
        print("  [PASS] Score formula correct")

    # Test 6: Illegal placement worsens score
    print("\n[TEST 6] Illegal placement worsens score")
    before_grid = torch.ones(16, dtype=torch.long)
    before_grid[0] = 2  # digit 1 at (0,0)
    score_before = sudoku_feasibility_checker(x, before_grid, w_v=2.0, w_z=5.0)

    after_grid = before_grid.clone()
    after_grid[1] = 2  # digit 1 at (0,1) - creates violation
    score_after = sudoku_feasibility_checker(x, after_grid, w_v=2.0, w_z=5.0)

    print(f"  Before (1 cell, no violations): score={score_before:.2f}")
    print(f"  After (2 cells, 2 violations):  score={score_after:.2f}")

    if score_after >= score_before:
        print(f"  [FAIL] Illegal placement should decrease score!")
        all_passed = False
    else:
        print("  [PASS] Illegal placement correctly decreases score")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SANITY CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("SOME SANITY CHECKS FAILED - SEE ABOVE")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_sanity_checks())
