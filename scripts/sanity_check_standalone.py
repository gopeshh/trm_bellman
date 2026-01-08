#!/usr/bin/env python3
"""
Standalone sanity check script for feasibility checker - no external deps.
Just verifies the core logic using inline implementations.
"""

import sys

# ============= INLINE IMPLEMENTATIONS (copied from rl/sudoku_utils.py) =============

def _grid_to_digits_4x4(grid):
    """Convert grid from token space to digit space for 4x4."""
    digits = [0] * 16
    for i in range(16):
        v = grid[i]
        if v == 1:
            digits[i] = 0  # empty
        else:
            digits[i] = max(0, v - 1)
    return digits


def count_violations_4x4(grid):
    """Count violations in 4x4 grid."""
    g = [grid[i] for i in range(16)]
    digits = [[_grid_to_digits_4x4(g)[r*4+c] for c in range(4)] for r in range(4)]
    violations = 0

    # Check rows
    for r in range(4):
        row = [digits[r][c] for c in range(4) if digits[r][c] > 0]
        violations += len(row) - len(set(row))

    # Check columns
    for c in range(4):
        col = [digits[r][c] for r in range(4) if digits[r][c] > 0]
        violations += len(col) - len(set(col))

    # Check 2x2 boxes
    for box_r in range(2):
        for box_c in range(2):
            box = []
            for r in range(box_r*2, box_r*2 + 2):
                for c in range(box_c*2, box_c*2 + 2):
                    if digits[r][c] > 0:
                        box.append(digits[r][c])
            violations += len(box) - len(set(box))

    return violations


def count_filled(grid, empty_token=1):
    """Count filled cells."""
    return sum(1 for v in grid if v != empty_token)


def get_candidates_4x4(digits, row, col):
    """Get candidate digits for a cell."""
    candidates = set(range(1, 5))

    # Row
    for c in range(4):
        if digits[row][c] > 0:
            candidates.discard(digits[row][c])

    # Column
    for r in range(4):
        if digits[r][col] > 0:
            candidates.discard(digits[r][col])

    # Box
    box_r, box_c = (row // 2) * 2, (col // 2) * 2
    for r in range(box_r, box_r + 2):
        for c in range(box_c, box_c + 2):
            if digits[r][c] > 0:
                candidates.discard(digits[r][c])

    return candidates


def count_zero_cand(grid):
    """Count cells with zero candidates."""
    g = [grid[i] for i in range(16)]
    digits = [[_grid_to_digits_4x4(g)[r*4+c] for c in range(4)] for r in range(4)]

    zero_count = 0
    for r in range(4):
        for c in range(4):
            if digits[r][c] == 0:  # Empty cell
                candidates = get_candidates_4x4(digits, r, c)
                if len(candidates) == 0:
                    zero_count += 1

    return zero_count


def is_solved_4x4(grid):
    """Check if grid is solved (all filled, no violations)."""
    filled = count_filled(grid)
    if filled != 16:
        return False
    violations = count_violations_4x4(grid)
    return violations == 0


def feasibility_score(grid, w_v=2.0, w_z=5.0):
    """Compute feasibility score: filled - w_v*violations - w_z*zeroCand."""
    filled = count_filled(grid)
    violations = count_violations_4x4(grid)
    zero_cand = count_zero_cand(grid)
    return filled - w_v * violations - w_z * zero_cand


def print_grid(grid, label="Grid"):
    """Print grid nicely."""
    print(f"\n{label}:")
    for r in range(4):
        row_str = " ".join(
            "." if grid[r*4+c] == 1 else str(grid[r*4+c] - 1)
            for c in range(4)
        )
        print(f"  {row_str}")


# ============= SANITY CHECKS =============

def main():
    print("=" * 60)
    print("FEASIBILITY CHECKER SANITY CHECKS (Standalone)")
    print("=" * 60)

    all_passed = True

    # Test 1: Empty grid is NOT solved
    print("\n[TEST 1] Empty grid is NOT solved")
    empty_grid = [1] * 16
    print_grid(empty_grid, "Empty Grid")

    filled = count_filled(empty_grid)
    violations = count_violations_4x4(empty_grid)
    zero_cand = count_zero_cand(empty_grid)
    score = feasibility_score(empty_grid)
    solved = is_solved_4x4(empty_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={solved}")

    if solved:
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
    partial_grid = [1] * 16
    partial_grid[0] = 2  # digit 1
    partial_grid[1] = 3  # digit 2
    partial_grid[5] = 4  # digit 3
    print_grid(partial_grid, "Partial Grid")

    filled = count_filled(partial_grid)
    violations = count_violations_4x4(partial_grid)
    zero_cand = count_zero_cand(partial_grid)
    score = feasibility_score(partial_grid)
    solved = is_solved_4x4(partial_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={solved}")

    if solved:
        print("  [FAIL] Partial grid should NOT be solved!")
        all_passed = False
    else:
        print("  [PASS] Partial grid correctly NOT solved")

    # Test 3: Valid solved grid IS solved
    print("\n[TEST 3] Valid solved grid IS solved")
    # Valid 4x4 Sudoku: token encoding digit d -> token d+1
    solved_grid = [
        2, 3, 4, 5,  # 1 2 3 4
        4, 5, 2, 3,  # 3 4 1 2
        3, 2, 5, 4,  # 2 1 4 3
        5, 4, 3, 2   # 4 3 2 1
    ]
    print_grid(solved_grid, "Solved Grid")

    filled = count_filled(solved_grid)
    violations = count_violations_4x4(solved_grid)
    zero_cand = count_zero_cand(solved_grid)
    score = feasibility_score(solved_grid)
    solved = is_solved_4x4(solved_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={solved}")

    if not solved:
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
    # Create grid where cell (2,1) has no valid candidates
    # Row 0: 1,2,3,4 -> tokens 2,3,4,5
    # Row 1: 3,4,1,2 -> tokens 4,5,2,3
    # Row 2: 2,.,1,. -> (2,1) is blocked by row:{2,1}, col:{2,4,3}
    # Row 3: 4,3,.,. -> tokens 5,4,1,1
    impossible_grid = [
        2, 3, 4, 5,  # Row 0: 1,2,3,4
        4, 5, 2, 3,  # Row 1: 3,4,1,2
        3, 1, 2, 1,  # Row 2: 2,.,1,.
        5, 4, 1, 1   # Row 3: 4,3,.,.
    ]
    print_grid(impossible_grid, "Impossible Grid")

    filled = count_filled(impossible_grid)
    violations = count_violations_4x4(impossible_grid)
    zero_cand = count_zero_cand(impossible_grid)
    score = feasibility_score(impossible_grid)
    solved = is_solved_4x4(impossible_grid)

    print(f"  filled={filled}, violations={violations}, zeroCand={zero_cand}")
    print(f"  score={score:.2f}, is_solved={solved}")

    if zero_cand == 0:
        print("  [WARN] Expected zeroCand > 0 for impossible grid")
        print("  (Checking if grid is actually impossible...)")
        # Debug: check candidates for empty cells
        g = [impossible_grid[i] for i in range(16)]
        digits = [[_grid_to_digits_4x4(g)[r*4+c] for c in range(4)] for r in range(4)]
        for r in range(4):
            for c in range(4):
                if digits[r][c] == 0:
                    cands = get_candidates_4x4(digits, r, c)
                    print(f"    Cell ({r},{c}): candidates = {cands}")
    else:
        print(f"  [PASS] Impossible grid has zeroCand={zero_cand} > 0")

    if solved:
        print("  [FAIL] Impossible grid should NOT be solved!")
        all_passed = False
    else:
        print("  [PASS] Impossible grid correctly NOT solved")

    # Test 5: Score formula verification
    print("\n[TEST 5] Score formula verification")
    # Grid with known values: 5 filled, some violations
    test_grid = [1] * 16
    test_grid[0] = 2  # digit 1 at (0,0)
    test_grid[1] = 2  # digit 1 at (0,1) - creates violation
    test_grid[4] = 3  # digit 2 at (1,0)
    test_grid[5] = 4  # digit 3 at (1,1)
    test_grid[10] = 5  # digit 4 at (2,2)
    print_grid(test_grid, "Test Grid")

    filled = count_filled(test_grid)
    violations = count_violations_4x4(test_grid)
    zero_cand = count_zero_cand(test_grid)
    score = feasibility_score(test_grid, w_v=2.0, w_z=5.0)
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
    before_grid = [1] * 16
    before_grid[0] = 2  # digit 1 at (0,0)
    score_before = feasibility_score(before_grid, w_v=2.0, w_z=5.0)

    after_grid = before_grid.copy()
    after_grid[1] = 2  # digit 1 at (0,1) - creates violation
    score_after = feasibility_score(after_grid, w_v=2.0, w_z=5.0)

    print(f"  Before (1 cell, no violations): score={score_before:.2f}")
    print(f"  After (2 cells, with violations): score={score_after:.2f}")

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
    sys.exit(main())
