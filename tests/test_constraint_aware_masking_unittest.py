"""
Tests for constraint-aware action masking in Sudoku - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import unittest
import torch

from rl.task_config import DummyTaskConfig, SudokuTaskConfig, get_task_config


class TestConstraintAwareMasking(unittest.TestCase):
    """Tests for the constraint-aware action masking in SudokuTaskConfig."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SudokuTaskConfig()
        self.vocab_size_4x4 = 6   # 0=PAD, 1=empty, 2-5=digits 1-4
        self.vocab_size_9x9 = 11  # 0=PAD, 1=empty, 2-10=digits 1-9

    def test_given_cells_masked_4x4(self):
        """Test that given cells (clues) cannot be edited in 4x4."""
        inputs = torch.tensor([
            2, 1, 1, 3,  # Row 0: digit 1, empty, empty, digit 2
            1, 1, 1, 1,  # Row 1: all empty
            1, 1, 1, 1,  # Row 2: all empty
            4, 1, 1, 5,  # Row 3: digit 3, empty, empty, digit 4
        ])
        stop_action_id = 16 * self.vocab_size_4x4  # 96

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        # Check that given cells (positions 0, 3, 12, 15) are fully masked
        given_positions = [0, 3, 12, 15]
        for pos in given_positions:
            start = pos * self.vocab_size_4x4
            end = start + self.vocab_size_4x4
            self.assertFalse(
                mask[start:end].any().item(),
                f"Given cell at pos {pos} should be masked"
            )

        # Check that empty cells have some valid actions
        empty_positions = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
        for pos in empty_positions:
            start = pos * self.vocab_size_4x4
            end = start + self.vocab_size_4x4
            self.assertTrue(
                mask[start+2:end].any().item(),
                f"Empty cell at pos {pos} should have valid actions"
            )

    def test_pad_and_empty_tokens_masked(self):
        """Test that PAD (0) and empty (1) tokens are masked for all positions."""
        inputs = torch.tensor([1] * 16)  # All empty 4x4
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        # Check that token 0 and 1 are masked for all positions
        for pos in range(16):
            for tok in [0, 1]:
                action_idx = pos * self.vocab_size_4x4 + tok
                self.assertFalse(
                    mask[action_idx].item(),
                    f"Token {tok} at pos {pos} should be masked"
                )

    def test_row_constraint_masking(self):
        """Test that digits already in the same row are masked."""
        inputs = torch.tensor([
            2, 1, 1, 1,  # Row 0: digit 1 at col 0, rest empty
            1, 1, 1, 1,  # Rows 1-3: all empty
            1, 1, 1, 1,
            1, 1, 1, 1,
        ])
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        # Token 2 (digit 1) should be masked for all other cells in row 0
        for pos in [1, 2, 3]:  # Other cells in row 0
            action_idx = pos * self.vocab_size_4x4 + 2  # token 2 = digit 1
            self.assertFalse(
                mask[action_idx].item(),
                f"Token 2 should be masked at pos {pos} (same row)"
            )

    def test_column_constraint_masking(self):
        """Test that digits already in the same column are masked."""
        inputs = torch.tensor([
            4, 1, 1, 1,  # Digit 3 at (0,0)
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ])
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        # Token 4 (digit 3) should be masked for all other cells in column 0
        for pos in [4, 8, 12]:  # Other cells in column 0
            action_idx = pos * self.vocab_size_4x4 + 4  # token 4 = digit 3
            self.assertFalse(
                mask[action_idx].item(),
                f"Token 4 should be masked at pos {pos} (same column)"
            )

    def test_box_constraint_masking_4x4(self):
        """Test that digits already in the same 2x2 box are masked."""
        inputs = torch.tensor([
            3, 1, 1, 1,  # Digit 2 at (0,0)
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ])
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        # Token 3 (digit 2) should be masked for all other cells in the 2x2 box
        # Box (0,0) contains positions: 0, 1, 4, 5
        for pos in [1, 4, 5]:  # Other cells in same box
            action_idx = pos * self.vocab_size_4x4 + 3  # token 3 = digit 2
            self.assertFalse(
                mask[action_idx].item(),
                f"Token 3 should be masked at pos {pos} (same box)"
            )

    def test_constraint_masking_9x9(self):
        """Test constraint masking on 9x9 grid."""
        inputs = torch.ones(81, dtype=torch.long)  # All empty
        inputs[0] = 5  # Digit 4 at position (0,0)

        stop_action_id = 81 * self.vocab_size_9x9  # 891

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_9x9, stop_action_id
        )

        # Token 5 (digit 4) should be masked in row 0
        for col in range(1, 9):
            pos = col
            action_idx = pos * self.vocab_size_9x9 + 5
            self.assertFalse(
                mask[action_idx].item(),
                f"Token 5 should be masked at row 0, col {col}"
            )

        # Token 5 should be masked in column 0
        for row in range(1, 9):
            pos = row * 9
            action_idx = pos * self.vocab_size_9x9 + 5
            self.assertFalse(
                mask[action_idx].item(),
                f"Token 5 should be masked at row {row}, col 0"
            )

    def test_current_state_updates_mask(self):
        """Test that passing current_state updates the mask correctly."""
        inputs = torch.ones(16, dtype=torch.long)
        current_state = torch.ones(16, dtype=torch.long)
        current_state[0] = 2  # Digit 1 at (0,0)

        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id,
            current_state=current_state
        )

        # Token 2 should be masked in row 0 positions 1, 2, 3
        for pos in [1, 2, 3]:
            action_idx = pos * self.vocab_size_4x4 + 2
            self.assertFalse(
                mask[action_idx].item(),
                f"Token 2 should be masked at pos {pos}"
            )

    def test_batch_mask_uses_current_state(self):
        """Test batch mask honors current_state constraints."""
        inputs = torch.ones(16, dtype=torch.long)
        current_state = inputs.clone()
        current_state[0] = 2  # digit 1 at (0,0)
        stop_action_id = 16 * self.vocab_size_4x4

        batch_mask = self.config.compute_batch_action_mask(
            inputs.unsqueeze(0),
            self.vocab_size_4x4,
            stop_action_id,
            current_state=current_state.unsqueeze(0),
        )

        # Token 2 should be masked in row 0 positions 1,2,3
        for pos in [1, 2, 3]:
            action_idx = pos * self.vocab_size_4x4 + 2
            self.assertFalse(
                batch_mask[0, action_idx].item(),
                f"Token 2 should be masked at pos {pos}"
            )

    def test_stop_action_always_valid(self):
        """Test that STOP action is always valid."""
        inputs = torch.ones(16, dtype=torch.long)
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        self.assertTrue(mask[stop_action_id].item(), "STOP action should be valid")

    def test_no_nan_in_mask(self):
        """Test that mask contains no NaN values."""
        inputs = torch.randint(1, 6, (16,))
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        self.assertFalse(
            torch.isnan(mask.float()).any().item(),
            "Mask should not contain NaN"
        )

    def test_mask_is_deterministic(self):
        """Test that mask computation is deterministic."""
        inputs = torch.tensor([2, 1, 3, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 1, 1, 2])
        stop_action_id = 16 * self.vocab_size_4x4

        mask1 = self.config.compute_action_mask(inputs, self.vocab_size_4x4, stop_action_id)
        mask2 = self.config.compute_action_mask(inputs, self.vocab_size_4x4, stop_action_id)

        self.assertTrue(torch.equal(mask1, mask2), "Mask should be deterministic")


class TestGetTaskConfig(unittest.TestCase):
    """Tests for the task config factory function."""

    def test_get_sudoku_config(self):
        """Test getting Sudoku task config."""
        config = get_task_config("sudoku")
        self.assertIsInstance(config, SudokuTaskConfig)
        self.assertEqual(config.name, "sudoku")

    def test_constraint_masking_enabled(self):
        """Test that constraint masking is enabled in Sudoku config."""
        config = get_task_config("sudoku")
        inputs = torch.tensor([2, 1, 1, 1] + [1] * 12)  # Digit 1 at (0,0)
        mask = config.compute_action_mask(inputs, 6, 96)
        # Token 2 should be masked in same row
        self.assertFalse(
            mask[1 * 6 + 2].item(),
            "Constraint masking should be active"
        )


class TestTaskConfigSignatures(unittest.TestCase):
    """Tests for TaskConfig compatibility with current_state."""

    def test_dummy_accepts_current_state(self):
        inputs = torch.ones(16, dtype=torch.long)
        stop_action_id = 16 * 6
        config = DummyTaskConfig()

        mask_default = config.compute_action_mask(inputs, 6, stop_action_id)
        mask_with_state = config.compute_action_mask(
            inputs, 6, stop_action_id, current_state=inputs
        )

        self.assertTrue(torch.equal(mask_default, mask_with_state))


if __name__ == "__main__":
    unittest.main()
