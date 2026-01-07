
import unittest
import torch
import yaml
from pathlib import Path
import os

# Import the checker function directly
from upi_trm_train import sudoku_progress_checker, sudoku_constraint_checker, count_sudoku_violations_4x4

class TestSudokuCheckers(unittest.TestCase):
    def setUp(self):
        # Create standard inputs
        self.x = {"inputs": torch.zeros(16)} # Dummy inputs
        
    def test_progress_checker_empty_grid(self):
        """Test score for empty grid (should satisfy constraints but score low)."""
        # Grid of all 1s (empty token)
        y = torch.ones(16, dtype=torch.long)
        
        # Violations should be 0
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)
        
        # Filled cells = 0
        score = sudoku_progress_checker(self.x, y, violation_penalty=2.0)
        self.assertEqual(score, 0.0)

    def test_progress_checker_partial_valid(self):
        """Test score for partially filled valid grid."""
        # 1 2 . .
        # . . . .
        # . . . .
        # . . . .
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2 # Digit 1
        y[1] = 3 # Digit 2
        
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)
        
        # Score = filled_cells = 2
        score = sudoku_progress_checker(self.x, y)
        self.assertEqual(score, 2.0)

    def test_progress_checker_partial_invalid(self):
        """Test score for partially filled grid with violations."""
        # 1 1 . . (Duplicate 1s in row)
        y = torch.ones(16, dtype=torch.long)
        y[0] = 2 # Digit 1
        y[1] = 2 # Digit 1 (Duplicate)
        
        # Should have 1 violation (row) + 1 violation (box) = 2 violations?
        # Let's check implementation of count_sudoku_violations_4x4
        # It sums violations across rows, cols, boxes.
        # Row 0: [1, 1] -> 1 duplicate.
        # Box 0: [1, 1] -> 1 duplicate.
        # Total violations = 2.
        
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 2)
        
        # Score = filled_cells (2) - violations (2) * penalty (2.0) = -2.0
        score = sudoku_progress_checker(self.x, y, violation_penalty=2.0)
        self.assertEqual(score, 2.0 - 2 * 2.0) # -2.0

    def test_progress_checker_solved(self):
        """Test score for fully solved grid."""
        # Valid 4x4 grid
        # 1 2 3 4
        # 3 4 1 2
        # 2 1 4 3
        # 4 3 2 1
        # Tokens: 1->2, 2->3, 3->4, 4->5
        y = torch.tensor([
            2, 3, 4, 5,
            4, 5, 2, 3,
            3, 2, 5, 4,
            5, 4, 3, 2
        ], dtype=torch.long)
        
        violations = count_sudoku_violations_4x4(y)
        self.assertEqual(violations, 0)
        
        score = sudoku_progress_checker(self.x, y)
        self.assertEqual(score, 16.0) # Max score

    def test_constraint_checker_empty(self):
        """Verify constraint checker flaw: empty grid scores perfect."""
        y = torch.ones(16, dtype=torch.long)
        score = sudoku_constraint_checker(self.x, y)
        # Should be 10.0 because 0 violations
        self.assertEqual(score, 10.0)

if __name__ == '__main__':
    unittest.main()

