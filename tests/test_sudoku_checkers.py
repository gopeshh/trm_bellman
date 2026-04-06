
import unittest
import torch
import yaml
from pathlib import Path
import os

# Import from the shared utility module
from rl.sudoku_utils import (
    count_sudoku_violations_4x4,
    count_sudoku_violations_9x9,
    sudoku_filled_cells,
    sudoku_zero_candidate_cells,
    sudoku_is_solved,
    sudoku_get_stats,
)

# Import the checker functions
from rl.sudoku_checkers import (
    sudoku_progress_checker,
    sudoku_constraint_checker,
    sudoku_feasibility_checker,
)


class TestSudokuUtils(unittest.TestCase):
    """Tests for the shared Sudoku utilities in rl/sudoku_utils.py"""

    def test_count_violations_empty_grid(self):
        """Empty grid should have 0 violations."""
        y = torch.ones(16, dtype=torch.long)  # All empty tokens
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)

    def test_count_violations_solved_grid(self):
        """Solved grid should have 0 violations."""
        # Valid 4x4 Sudoku solution
        # 1 2 3 4 -> tokens 2 3 4 5
        # 3 4 1 2 -> tokens 4 5 2 3
        # 2 1 4 3 -> tokens 3 2 5 4
        # 4 3 2 1 -> tokens 5 4 3 2
        y = torch.tensor([
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2
        ], dtype=torch.long)
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)

    def test_count_violations_with_duplicates(self):
        """Grid with duplicates should have violations."""
        # 1 1 . . (duplicate in row and box)
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2  # Digit 1
        y[1] = 2  # Digit 1 (duplicate)
        violations = count_sudoku_violations_4x4(y)
        # Row 0: 1 duplicate, Box 0: 1 duplicate = 2 violations
        self.assertEqual(violations, 2)

    def test_filled_cells_empty(self):
        """Empty grid should have 0 filled cells."""
        y = torch.ones(16, dtype=torch.long)
        filled = sudoku_filled_cells(y, empty_token=1)
        self.assertEqual(filled, 0)

    def test_filled_cells_partial(self):
        """Partially filled grid should count correctly."""
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2
        y[1] = 3
        y[2] = 4
        filled = sudoku_filled_cells(y, empty_token=1)
        self.assertEqual(filled, 3)

    def test_filled_cells_full(self):
        """Fully filled grid should have all cells counted."""
        y = torch.tensor([
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2
        ], dtype=torch.long)
        filled = sudoku_filled_cells(y, empty_token=1)
        self.assertEqual(filled, 16)


class TestZeroCandidateCells(unittest.TestCase):
    """Tests for zero-candidate cell detection."""

    def test_empty_grid_no_zero_cand(self):
        """Empty grid should have no zero-candidate cells (all candidates available)."""
        y = torch.ones(16, dtype=torch.long)  # All empty
        zero_cand = sudoku_zero_candidate_cells(y, grid_size=4)
        self.assertEqual(zero_cand, 0)

    def test_impossible_state_has_zero_cand(self):
        """An impossible state should have zero-candidate cells."""
        # Create a 4x4 grid where first row has all 4 digits
        # and first column has 3 digits, leaving cell (1,0) impossible
        # Row 0: 1, 2, 3, 4
        # Row 1: empty, ?, ?, ?
        # But column 0 also has digits blocking...

        # Actually, let's create a clearer impossible case:
        # Fill row 0 with 1,2,3,4 and column 0 with 1,2,3,4
        # This makes cell (1,0) impossible since all candidates are taken

        # Token encoding: 1=empty, 2=1, 3=2, 4=3, 5=4
        y = torch.ones(16, dtype=torch.long)
        # Row 0: 1, 2, 3, 4 -> tokens 2, 3, 4, 5
        y[0] = 2  # (0,0) = 1
        y[1] = 3  # (0,1) = 2
        y[2] = 4  # (0,2) = 3
        y[3] = 5  # (0,3) = 4

        # Col 0: 1, 2, 3, 4 -> rest of column
        # But (0,0) already = 1, so we fill (1,0), (2,0), (3,0)
        y[4] = 3  # (1,0) = 2
        y[8] = 4  # (2,0) = 3
        y[12] = 5  # (3,0) = 4

        # Now box 0 (top-left 2x2) has: (0,0)=1, (0,1)=2, (1,0)=2, (1,1)=empty
        # Cell (1,1) candidates: must avoid row 1's {2}, col 1's {2}, box 0's {1,2,2}
        # Available in box 0: needs 3 and 4
        # Row 1: {2}, Col 1: {2} -> (1,1) can be 3 or 4 if not blocked
        # This is actually not an impossible state

        # Let's make a truly impossible state by filling more
        # Actually the simplest is: fill all of row 0, col 0, and box 0
        # leaving one cell that has no valid candidates

        # For 4x4, a simpler approach:
        # Row 0: 1,2,3,4; Row 1: 3,4,1,2; but then fill col 0 with duplicates
        # Or just put all 4 different digits in a way that blocks an empty cell

        # Clearest impossible case:
        # (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4 - this fills box 0 completely
        # Then row 0 also has (0,2)=3, (0,3)=4 to complete row
        # And column 0 has (2,0)=2, (3,0)=4
        # This creates violations but that's fine for testing zero_cand

        # Actually, let me think differently. An impossible state is when
        # an empty cell has all 4 candidates blocked by row/col/box

        # Grid:
        # 1 2 | 3 4
        # 3 4 | 1 2
        # ----+----
        # 2 . | . .  <- (2,1) has candidates from row 2: {2 taken}, col 1: {2,4 taken}
        # 4 . | . .     box 2: {2,4 taken}. So (2,1) can be 1 or 3

        # To make (2,1) impossible: need 1 and 3 in row 2, col 1, or box 2
        # Add (2,2)=1 and (3,1)=3
        y = torch.ones(16, dtype=torch.long)
        # Row 0: 1, 2, 3, 4
        y[0] = 2  # 1
        y[1] = 3  # 2
        y[2] = 4  # 3
        y[3] = 5  # 4
        # Row 1: 3, 4, 1, 2
        y[4] = 4  # 3
        y[5] = 5  # 4
        y[6] = 2  # 1
        y[7] = 3  # 2
        # Row 2: 2, empty, 1, empty
        y[8] = 3   # 2
        y[9] = 1   # empty - this should become impossible
        y[10] = 2  # 1
        y[11] = 1  # empty
        # Row 3: 4, 3, empty, empty
        y[12] = 5  # 4
        y[13] = 4  # 3
        y[14] = 1  # empty
        y[15] = 1  # empty

        # Now check (2,1) - it's empty (y[9]=1)
        # Row 2 has: 2, 1 -> candidates blocked: {1, 2}
        # Col 1 has: 2, 4, 3 -> candidates blocked: {2, 3, 4}
        # Combined: {1, 2, 3, 4} all blocked! -> zero candidates

        zero_cand = sudoku_zero_candidate_cells(y, grid_size=4)
        self.assertGreater(zero_cand, 0, "Expected at least one zero-candidate cell")


class TestSudokuIsSolved(unittest.TestCase):
    """Tests for the sudoku_is_solved function."""

    def test_empty_is_not_solved(self):
        """Empty grid is not solved."""
        y = torch.ones(16, dtype=torch.long)
        self.assertFalse(sudoku_is_solved(y))

    def test_partial_is_not_solved(self):
        """Partially filled grid is not solved."""
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2
        y[1] = 3
        self.assertFalse(sudoku_is_solved(y))

    def test_full_with_violations_is_not_solved(self):
        """Full grid with violations is not solved."""
        # All cells filled with same digit -> many violations
        y = torch.full((16,), 2, dtype=torch.long)  # All 1s
        self.assertFalse(sudoku_is_solved(y))

    def test_valid_solved_grid(self):
        """Valid solved grid should return True."""
        y = torch.tensor([
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2
        ], dtype=torch.long)
        self.assertTrue(sudoku_is_solved(y))


class TestSudokuCheckers(unittest.TestCase):
    """Tests for the checker functions in upi_trm_train.py"""

    def setUp(self):
        self.x = {"inputs": torch.zeros(16)}

    def test_progress_checker_empty_grid(self):
        """Test score for empty grid."""
        y = torch.ones(16, dtype=torch.long)
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)
        score = sudoku_progress_checker(self.x, y, violation_penalty=2.0)
        self.assertEqual(score, 0.0)

    def test_progress_checker_partial_valid(self):
        """Test score for partially filled valid grid."""
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2  # Digit 1
        y[1] = 3  # Digit 2
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)
        score = sudoku_progress_checker(self.x, y)
        self.assertEqual(score, 2.0)

    def test_progress_checker_partial_invalid(self):
        """Test score for partially filled grid with violations."""
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2  # Digit 1
        y[1] = 2  # Digit 1 (Duplicate)
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 2)
        score = sudoku_progress_checker(self.x, y, violation_penalty=2.0)
        self.assertEqual(score, 2.0 - 2 * 2.0)  # -2.0

    def test_progress_checker_solved(self):
        """Test score for fully solved grid."""
        y = torch.tensor([
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2
        ], dtype=torch.long)
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)
        score = sudoku_progress_checker(self.x, y)
        self.assertEqual(score, 16.0)

    def test_constraint_checker_empty(self):
        """Verify constraint checker flaw: empty grid scores perfect."""
        y = torch.ones(16, dtype=torch.long)
        score = sudoku_constraint_checker(self.x, y)
        self.assertEqual(score, 10.0)


class TestFeasibilityChecker(unittest.TestCase):
    """Tests for the feasibility checker."""

    def setUp(self):
        self.x = {"inputs": torch.zeros(16), "solution": torch.zeros(16)}

    def test_empty_grid_score(self):
        """Empty grid: filled=0, violations=0, zeroCand=0."""
        y = torch.ones(16, dtype=torch.long)
        score = sudoku_feasibility_checker(self.x, y, w_v=2.0, w_z=5.0)
        # score = 0 - 2*0 - 5*0 = 0
        self.assertEqual(score, 0.0)

    def test_valid_partial_score(self):
        """Valid partial grid should have positive score."""
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2  # Digit 1
        y[1] = 3  # Digit 2
        score = sudoku_feasibility_checker(self.x, y, w_v=2.0, w_z=5.0)
        # filled=2, violations=0, zeroCand=0
        # score = 2 - 0 - 0 = 2
        self.assertEqual(score, 2.0)

    def test_invalid_partial_score(self):
        """Invalid partial grid should be penalized."""
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2  # Digit 1
        y[1] = 2  # Digit 1 (duplicate)
        score = sudoku_feasibility_checker(self.x, y, w_v=2.0, w_z=5.0)
        # filled=2, violations=2 (row + box), zeroCand=?
        # score = 2 - 2*2 - 5*zeroCand
        violations = count_sudoku_violations_4x4(y)
        zero_cand = sudoku_zero_candidate_cells(y, grid_size=4)
        expected = 2 - 2.0 * violations - 5.0 * zero_cand
        self.assertEqual(score, expected)

    def test_solved_grid_score(self):
        """Solved grid should have maximum score."""
        y = torch.tensor([
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2
        ], dtype=torch.long)
        score = sudoku_feasibility_checker(self.x, y, w_v=2.0, w_z=5.0)
        # filled=16, violations=0, zeroCand=0
        # score = 16 - 0 - 0 = 16
        self.assertEqual(score, 16.0)

    def test_legal_placement_improves_score(self):
        """A legal placement should not decrease score."""
        y_before = torch.ones(16, dtype=torch.long)
        y_before[0] = 2  # Digit 1

        y_after = y_before.clone()
        y_after[1] = 3  # Digit 2 (valid placement)

        score_before = sudoku_feasibility_checker(self.x, y_before)
        score_after = sudoku_feasibility_checker(self.x, y_after)

        self.assertGreater(score_after, score_before)

    def test_illegal_placement_worsens_score(self):
        """An illegal placement (violation) should decrease score or keep it low."""
        y_before = torch.ones(16, dtype=torch.long)
        y_before[0] = 2  # Digit 1

        y_after = y_before.clone()
        y_after[1] = 2  # Digit 1 (duplicate - violation)

        score_before = sudoku_feasibility_checker(self.x, y_before)
        score_after = sudoku_feasibility_checker(self.x, y_after)

        # The illegal placement adds a violation penalty
        # Even though filled cells increase by 1, penalty should offset
        self.assertLess(score_after, score_before)


class TestVectorized9x9Violations(unittest.TestCase):
    """Tests for vectorized 9x9 violation counting."""

    def test_empty_grid_no_violations(self):
        """Empty 9x9 grid should have 0 violations."""
        y = torch.ones(81, dtype=torch.long)  # All empty tokens
        violations = count_sudoku_violations_9x9(y)
        self.assertEqual(violations, 0)

    def test_solved_grid_no_violations(self):
        """Valid solved 9x9 grid should have 0 violations."""
        # A valid 9x9 Sudoku solution in token space (token = digit + 1)
        # Standard Sudoku solution:
        solution = [
            5, 3, 4, 6, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 3, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        ]
        # Convert to token space: token = digit + 1
        y = torch.tensor([d + 1 for d in solution], dtype=torch.long)
        violations = count_sudoku_violations_9x9(y)
        self.assertEqual(violations, 0)

    def test_row_duplicate_violation(self):
        """Duplicate in row should be counted."""
        y = torch.ones(81, dtype=torch.long)
        # Put two 1s in row 0 (tokens 2, 2)
        y[0] = 2  # (0,0) = digit 1
        y[1] = 2  # (0,1) = digit 1 (duplicate)
        violations = count_sudoku_violations_9x9(y)
        # Row violation: 1, Box violation: 1 (both in box 0)
        self.assertEqual(violations, 2)

    def test_column_duplicate_violation(self):
        """Duplicate in column should be counted."""
        y = torch.ones(81, dtype=torch.long)
        # Put two 1s in column 0, different boxes
        y[0] = 2   # (0,0) = digit 1
        y[27] = 2  # (3,0) = digit 1 (same column, different box)
        violations = count_sudoku_violations_9x9(y)
        # Column violation: 1
        self.assertEqual(violations, 1)

    def test_box_duplicate_violation(self):
        """Duplicate in box (not row/col) should be counted."""
        y = torch.ones(81, dtype=torch.long)
        # Put two 1s in box 0, different rows and columns
        y[0] = 2   # (0,0) = digit 1
        y[10] = 2  # (1,1) = digit 1 (same box, different row/col)
        violations = count_sudoku_violations_9x9(y)
        # Box violation only: 1
        self.assertEqual(violations, 1)

    def test_multiple_duplicates(self):
        """Multiple duplicates should sum correctly."""
        y = torch.ones(81, dtype=torch.long)
        # Three 1s in row 0 (positions 0, 1, 2) - all in box 0 too
        y[0] = 2
        y[1] = 2
        y[2] = 2
        violations = count_sudoku_violations_9x9(y)
        # Row 0: 3 cells with digit 1 -> 2 violations
        # Box 0: 3 cells with digit 1 -> 2 violations
        # Total: 4 violations
        self.assertEqual(violations, 4)


class TestVectorized9x9ZeroCandidates(unittest.TestCase):
    """Tests for vectorized 9x9 zero-candidate detection."""

    def test_empty_grid_no_zero_cand(self):
        """Empty 9x9 grid should have no zero-candidate cells."""
        y = torch.ones(81, dtype=torch.long)
        zero_cand = sudoku_zero_candidate_cells(y, grid_size=9)
        self.assertEqual(zero_cand, 0)

    def test_solved_grid_no_zero_cand(self):
        """Solved grid has no empty cells, so no zero-candidate cells."""
        solution = [
            5, 3, 4, 6, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 3, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        ]
        y = torch.tensor([d + 1 for d in solution], dtype=torch.long)
        zero_cand = sudoku_zero_candidate_cells(y, grid_size=9)
        self.assertEqual(zero_cand, 0)

    def test_impossible_state_has_zero_cand(self):
        """An impossible 9x9 state should have zero-candidate cells."""
        # Create a state where cell (0,8) has all 9 digits blocked
        # Fill row 0 with digits 1-8 (leaving position 8 empty)
        # Then put digit 9 somewhere that blocks (0,8)
        y = torch.ones(81, dtype=torch.long)

        # Row 0, positions 0-7: digits 1-8 (tokens 2-9)
        for i in range(8):
            y[i] = i + 2  # digit 1-8

        # Position 8 (0,8) is empty - needs digit 9
        # But let's put digit 9 in column 8 or box 2 to block it
        # Box 2 = top-right 3x3 = positions (0,6), (0,7), (0,8), (1,6)...
        # Column 8 = positions 8, 17, 26, 35, ...
        y[17] = 10  # (1,8) = digit 9

        # Now cell (0,8) is blocked:
        # Row 0 has 1-8, column 8 has 9, so all digits 1-9 are blocked
        zero_cand = sudoku_zero_candidate_cells(y, grid_size=9)
        self.assertGreater(zero_cand, 0, "Expected at least one zero-candidate cell")

    def test_partial_grid_with_candidates(self):
        """Partial grid where empty cells still have candidates."""
        y = torch.ones(81, dtype=torch.long)
        # Fill first row partially
        y[0] = 2  # digit 1
        y[1] = 3  # digit 2
        y[2] = 4  # digit 3

        # All other empty cells should still have candidates
        zero_cand = sudoku_zero_candidate_cells(y, grid_size=9)
        self.assertEqual(zero_cand, 0)


if __name__ == '__main__':
    unittest.main()
