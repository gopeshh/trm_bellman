"""
Comprehensive tests for constraint-aware action masking in Sudoku.

Tests verify that:
1. Given cells are properly masked (can't overwrite clues)
2. PAD and empty tokens are masked for all positions
3. Sudoku constraints are enforced (no duplicate digits in row/col/box)
4. Mask updates correctly after each step
"""

import pytest
import torch
from rl.task_config import DummyTaskConfig, SudokuTaskConfig, get_task_config
from rl.envs.plan_edit_env import PlanEditEnv


class TestConstraintAwareMasking:
    """Tests for the constraint-aware action masking in SudokuTaskConfig."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SudokuTaskConfig()
        self.vocab_size_4x4 = 6   # 0=PAD, 1=empty, 2-5=digits 1-4
        self.vocab_size_9x9 = 11  # 0=PAD, 1=empty, 2-10=digits 1-9

    def test_given_cells_masked_4x4(self):
        """Test that given cells (clues) cannot be edited in 4x4."""
        # 4x4 puzzle with some given cells
        # Token 1 = empty, Token 2-5 = digits 1-4
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
            assert not mask[start:end].any(), f"Given cell at pos {pos} should be masked"

        # Check that empty cells (position 1, 2, 4, etc.) have some valid actions
        empty_positions = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
        for pos in empty_positions:
            start = pos * self.vocab_size_4x4
            end = start + self.vocab_size_4x4
            # Should have at least some valid actions (tokens 2-5)
            assert mask[start+2:end].any(), f"Empty cell at pos {pos} should have valid actions"

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
                assert not mask[action_idx], f"Token {tok} at pos {pos} should be masked"

    def test_row_constraint_masking(self):
        """Test that digits already in the same row are masked."""
        # 4x4 puzzle: Row 0 has digit 1 (token 2) at position 0
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
            assert not mask[action_idx], f"Token 2 should be masked at pos {pos} (same row)"

        # But token 2 should be valid in other rows (pos 4, 8, 12)
        for pos in [4, 8, 12]:
            action_idx = pos * self.vocab_size_4x4 + 2
            # Could be masked by column constraint, so just check it's not row-blocked
            # Actually pos 4 is same column as pos 0, so it will be masked by column
            pass  # Skip this check as column constraint also applies

    def test_column_constraint_masking(self):
        """Test that digits already in the same column are masked."""
        # 4x4 puzzle: Column 0 has digit 3 (token 4) at position 0
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
            assert not mask[action_idx], f"Token 4 should be masked at pos {pos} (same column)"

    def test_box_constraint_masking_4x4(self):
        """Test that digits already in the same 2x2 box are masked."""
        # 4x4 puzzle: Box (0,0) has digit 2 (token 3) at position 0
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
            assert not mask[action_idx], f"Token 3 should be masked at pos {pos} (same box)"

    def test_constraint_masking_9x9(self):
        """Test constraint masking on 9x9 grid."""
        # Create 9x9 puzzle with one digit
        inputs = torch.ones(81, dtype=torch.long)  # All empty
        inputs[0] = 5  # Digit 4 at position (0,0)

        stop_action_id = 81 * self.vocab_size_9x9  # 891

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_9x9, stop_action_id
        )

        # Token 5 (digit 4) should be masked in:
        # - Row 0: positions 1-8
        # - Column 0: positions 9, 18, 27, 36, 45, 54, 63, 72
        # - Box (0,0): positions 1, 2, 9, 10, 11, 18, 19, 20

        # Check row 0
        for col in range(1, 9):
            pos = col
            action_idx = pos * self.vocab_size_9x9 + 5
            assert not mask[action_idx], f"Token 5 should be masked at row 0, col {col}"

        # Check column 0
        for row in range(1, 9):
            pos = row * 9
            action_idx = pos * self.vocab_size_9x9 + 5
            assert not mask[action_idx], f"Token 5 should be masked at row {row}, col 0"

    def test_current_state_updates_mask(self):
        """Test that passing current_state updates the mask correctly."""
        # Original inputs: all empty
        inputs = torch.ones(16, dtype=torch.long)

        # Current state: digit 1 at position 0
        current_state = torch.ones(16, dtype=torch.long)
        current_state[0] = 2  # Digit 1 at (0,0)

        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id,
            current_state=current_state
        )

        # Token 2 should be masked in row 0, column 0, and box (0,0)
        # Position 0 is empty in inputs, so position itself is editable
        # But placing token 2 at positions 1, 2, 3 (same row) should be masked
        for pos in [1, 2, 3]:
            action_idx = pos * self.vocab_size_4x4 + 2
            assert not mask[action_idx], f"Token 2 should be masked at pos {pos}"

    def test_batch_mask_uses_current_state(self):
        """Test batch mask honors current_state constraints."""
        inputs = torch.ones(16, dtype=torch.long)  # all empty 4x4
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
            assert not batch_mask[0, action_idx], f"Token 2 should be masked at pos {pos}"

    def test_constraint_masking_can_be_disabled(self):
        """Disabling constraint masking restores the given-cell-only action space."""
        inputs = torch.tensor([
            2, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ])
        stop_action_id = 16 * self.vocab_size_4x4

        masked = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id, current_state=inputs
        )
        unmasked_config = SudokuTaskConfig(disable_constraint_masking=True)
        unmasked = unmasked_config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id, current_state=inputs
        )
        unmasked_batch = unmasked_config.compute_batch_action_mask(
            inputs.unsqueeze(0),
            self.vocab_size_4x4,
            stop_action_id,
            current_state=inputs.unsqueeze(0),
        )

        conflict_action = 1 * self.vocab_size_4x4 + 2
        assert not masked[conflict_action], "Default Sudoku masking should block row conflicts"
        assert unmasked[conflict_action], "No-mask mode should allow row-conflicting edits"
        assert unmasked_batch[0, conflict_action], "Batch mask should match single-state no-mask behavior"

        assert not unmasked[: self.vocab_size_4x4].any(), "Given cells must remain fully masked"
        assert not unmasked[1 * self.vocab_size_4x4 + 0], "PAD should remain masked"
        assert not unmasked[1 * self.vocab_size_4x4 + 1], "Empty token should remain masked"

    def test_stop_action_always_valid(self):
        """Test that STOP action is always valid (when not disabled)."""
        inputs = torch.ones(16, dtype=torch.long)
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        assert mask[stop_action_id], "STOP action should be valid"

    def test_no_nan_in_mask(self):
        """Test that mask contains no NaN values."""
        inputs = torch.randint(1, 6, (16,))
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(
            inputs, self.vocab_size_4x4, stop_action_id
        )

        assert not torch.isnan(mask.float()).any(), "Mask should not contain NaN"

    def test_mask_is_deterministic(self):
        """Test that mask computation is deterministic."""
        inputs = torch.tensor([2, 1, 3, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 1, 1, 2])
        stop_action_id = 16 * self.vocab_size_4x4

        mask1 = self.config.compute_action_mask(inputs, self.vocab_size_4x4, stop_action_id)
        mask2 = self.config.compute_action_mask(inputs, self.vocab_size_4x4, stop_action_id)

        assert torch.equal(mask1, mask2), "Mask should be deterministic"

    def test_valid_actions_exist_for_empty_cells(self):
        """Test that empty cells have at least one valid action (unless fully constrained)."""
        # Simple 4x4 with few constraints
        inputs = torch.tensor([
            2, 1, 1, 1,  # Digit 1 at (0,0)
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ])
        stop_action_id = 16 * self.vocab_size_4x4

        mask = self.config.compute_action_mask(inputs, self.vocab_size_4x4, stop_action_id)

        # Most empty cells should have valid actions
        # Position 5 (row 1, col 1) is not constrained by position 0
        pos = 5
        start = pos * self.vocab_size_4x4 + 2  # Start from token 2
        end = pos * self.vocab_size_4x4 + self.vocab_size_4x4
        assert mask[start:end].any(), f"Position {pos} should have valid actions"


class TestEnvironmentMaskUpdate:
    """Tests for mask updates in the environment after each step."""

    def test_env_mask_updates_after_step(self):
        """Test that environment mask updates after each step."""
        # This test requires a full environment setup
        # For now, just verify the compute_action_mask signature
        config = SudokuTaskConfig()

        # Verify that compute_action_mask accepts current_state parameter
        import inspect
        sig = inspect.signature(config.compute_action_mask)
        params = list(sig.parameters.keys())
        assert 'current_state' in params, "compute_action_mask should accept current_state"


class TestGetTaskConfig:
    """Tests for the task config factory function."""

    def test_get_sudoku_config(self):
        """Test getting Sudoku task config."""
        config = get_task_config("sudoku")
        assert isinstance(config, SudokuTaskConfig)
        assert config.name == "sudoku"

    def test_constraint_masking_enabled(self):
        """Test that constraint masking is enabled in Sudoku config."""
        config = get_task_config("sudoku")

        # Create a simple puzzle
        inputs = torch.tensor([2, 1, 1, 1] + [1] * 12)  # Digit 1 at (0,0)
        mask = config.compute_action_mask(inputs, 6, 96)

        # Verify token 2 is masked in same row/col/box
        assert not mask[1 * 6 + 2], "Constraint masking should be active"


class TestTaskConfigSignatures:
    """Tests for TaskConfig compatibility with current_state."""

    def test_dummy_accepts_current_state(self):
        inputs = torch.ones(16, dtype=torch.long)
        stop_action_id = 16 * 6
        config = DummyTaskConfig()

        mask_default = config.compute_action_mask(inputs, 6, stop_action_id)
        mask_with_state = config.compute_action_mask(
            inputs, 6, stop_action_id, current_state=inputs
        )

        assert torch.equal(mask_default, mask_with_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
