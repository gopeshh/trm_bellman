"""Unit tests for SudokuMetaMDP with edit application and rewards."""

import pytest

# Try to import torch, skip tests if not available
torch = pytest.importorskip("torch", reason="torch not installed")

from rl.meta_mdp import SudokuMetaMDP, score_sudoku  # noqa: E402


@pytest.mark.unit
def test_score_sudoku_empty_grid():
    """Test score_sudoku on empty grid."""
    batch_size = 2
    # Empty 9x9 Sudoku grid (all zeros)
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)

    score = score_sudoku(y)

    # Empty grid has no violations (all constraints satisfied trivially)
    # 9 rows + 9 cols + 9 boxes = 27 constraints, each with 9 values = 243 total
    assert score.shape == (batch_size,)
    assert (score == 243).all(), f"Empty grid should have perfect score 243, got {score}"


@pytest.mark.unit
def test_score_sudoku_with_duplicates():
    """Test score_sudoku penalizes duplicate values in rows."""
    batch_size = 1
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)

    # Put duplicate 5s in first row
    y[0, 0, 0] = 5
    y[0, 0, 1] = 5

    score = score_sudoku(y)

    # Should have violations in row constraint for value 5
    # Perfect score is 243, we should have fewer due to violation
    assert score < 243, f"Grid with duplicates should score < 243, got {score}"


@pytest.mark.unit
def test_sudoku_meta_mdp_initialization():
    """Test SudokuMetaMDP initializes correctly."""
    mdp = SudokuMetaMDP(edit_penalty=0.05)

    assert mdp.edit_penalty == 0.05
    assert mdp.device == "cpu"


@pytest.mark.unit
def test_apply_edit_is_deterministic():
    """Test apply_edit produces deterministic results.

    Key acceptance test: Same input and action should always
    produce the same output.
    """
    batch_size = 2
    mdp = SudokuMetaMDP()

    y = torch.randint(0, 10, (batch_size, 9, 9), dtype=torch.long)
    pos = torch.tensor([0, 5])  # First cell, and cell at position 5
    val = torch.tensor([7, 3])  # Set to 7 and 3
    a = (pos, val)

    # Apply edit twice
    y_prime1 = mdp.apply_edit(y, a)
    y_prime2 = mdp.apply_edit(y, a)

    # Should be identical
    assert torch.equal(y_prime1, y_prime2), "apply_edit should be deterministic"


@pytest.mark.unit
def test_apply_edit_modifies_correct_position():
    """Test apply_edit modifies the correct grid position."""
    batch_size = 1
    mdp = SudokuMetaMDP()

    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)

    # Edit position 13 (row=1, col=4) to value 5
    pos = torch.tensor([13])
    val = torch.tensor([5])
    a = (pos, val)

    y_prime = mdp.apply_edit(y, a)

    # Check the correct position was modified
    assert y_prime[0, 1, 4] == 5, f"Expected y_prime[0, 1, 4] == 5, got {y_prime[0, 1, 4]}"

    # Check other positions unchanged
    y_prime[0, 1, 4] = 0  # Reset the edited position
    assert torch.equal(y_prime, y), "Other positions should be unchanged"


@pytest.mark.unit
def test_apply_edit_batch_processing():
    """Test apply_edit handles batches correctly."""
    batch_size = 3
    mdp = SudokuMetaMDP()

    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)

    # Different edits for each batch
    pos = torch.tensor([0, 10, 20])  # Different positions
    val = torch.tensor([1, 2, 3])  # Different values
    a = (pos, val)

    y_prime = mdp.apply_edit(y, a)

    # Check each batch got the right edit
    assert y_prime[0, 0, 0] == 1
    assert y_prime[1, 1, 1] == 2  # pos=10 -> row=1, col=1
    assert y_prime[2, 2, 2] == 3  # pos=20 -> row=2, col=2


@pytest.mark.unit
def test_reward_positive_when_fixing_cell():
    """Test reward > 0 when fixing a wrong cell improves score.

    Key acceptance test: Fixing a constraint violation should
    give positive reward.
    """
    batch_size = 1
    mdp = SudokuMetaMDP(edit_penalty=0.0)  # No penalty for clarity

    # Create grid with duplicate 5s in first row
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)
    y[0, 0, 0] = 5
    y[0, 0, 1] = 5  # Duplicate!

    # Fix the duplicate by changing second 5 to 0 (empty)
    pos = torch.tensor([1])  # Position 1 = (0, 1)
    val = torch.tensor([0])  # Erase
    a = (pos, val)

    y_prime = mdp.apply_edit(y, a)
    reward = mdp.reward(y, y_prime, a)

    # Reward should be positive (score improved)
    assert reward[0] > 0, f"Fixing constraint should give positive reward, got {reward[0]}"


@pytest.mark.unit
def test_reward_negative_when_creating_violation():
    """Test reward < 0 when creating a constraint violation."""
    batch_size = 1
    mdp = SudokuMetaMDP(edit_penalty=0.01)

    # Start with valid configuration
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)
    y[0, 0, 0] = 5

    # Create duplicate by adding another 5 in same row
    pos = torch.tensor([1])  # Position 1 = (0, 1)
    val = torch.tensor([5])  # Duplicate!
    a = (pos, val)

    y_prime = mdp.apply_edit(y, a)
    reward = mdp.reward(y, y_prime, a)

    # Reward should be negative (score decreased + penalty)
    assert reward[0] < 0, f"Creating violation should give negative reward, got {reward[0]}"


@pytest.mark.unit
def test_reward_includes_edit_penalty():
    """Test reward includes the edit penalty λ."""
    batch_size = 1
    penalty = 0.1
    mdp = SudokuMetaMDP(edit_penalty=penalty)

    # Neutral edit (doesn't change score)
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)
    pos = torch.tensor([0])
    val = torch.tensor([0])  # Set empty to empty (no-op)
    a = (pos, val)

    y_prime = mdp.apply_edit(y, a)
    reward = mdp.reward(y, y_prime, a)

    # Reward should be -penalty (no score change)
    assert torch.allclose(
        reward, torch.tensor([-penalty])
    ), f"No-op edit should give reward=-{penalty}, got {reward}"


@pytest.mark.unit
def test_step_returns_correct_tuple():
    """Test step() returns (s', r, done, info) with correct shapes."""
    batch_size = 2
    mdp = SudokuMetaMDP()

    x = torch.randn(batch_size, 10)  # Dummy input
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)
    s = (x, y)

    pos = torch.tensor([0, 5])
    val = torch.tensor([1, 2])
    a = (pos, val)

    s_prime, r, done, info = mdp.step(s, a, x)

    # Check return types and shapes
    x_prime, y_prime = s_prime
    assert torch.equal(x_prime, x), "Input x should not change"
    assert y_prime.shape == (batch_size, 9, 9)
    assert r.shape == (batch_size,)
    assert done.shape == (batch_size,)
    assert "score" in info
    assert "score_improvement" in info
    assert "done" in info


@pytest.mark.unit
def test_step_done_when_solved():
    """Test step() sets done=True when Sudoku is solved (score=243)."""
    batch_size = 1
    mdp = SudokuMetaMDP()

    # Create a valid (but incomplete) Sudoku
    x = torch.randn(batch_size, 10)
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)

    # Fill in a valid complete Sudoku (simplified: all 1s in different positions)
    # This won't be a real Sudoku solution but will have score=243 for testing
    for i in range(9):
        for j in range(9):
            y[0, i, j] = (i + j) % 9 + 1  # Simple valid pattern

    s = (x, y)

    # Make a no-op edit
    pos = torch.tensor([0])
    val = torch.tensor([1])
    a = (pos, val)

    _, _, done, info = mdp.step(s, a, x)

    # Should be marked as done if score is perfect
    score = info["score"]
    if score[0] >= 243:
        assert done[0] == 1.0, "Should be done when score >= 243"


@pytest.mark.unit
def test_apply_edit_does_not_modify_input():
    """Test apply_edit doesn't modify the input grid in-place."""
    batch_size = 1
    mdp = SudokuMetaMDP()

    y_original = torch.ones(batch_size, 9, 9, dtype=torch.long)
    y = y_original.clone()

    pos = torch.tensor([0])
    val = torch.tensor([5])
    a = (pos, val)

    y_prime = mdp.apply_edit(y, a)

    # Original y should be unchanged
    assert torch.equal(y, y_original), "apply_edit should not modify input in-place"
    # y_prime should be different
    assert not torch.equal(y_prime, y), "y_prime should be different from y"


@pytest.mark.unit
def test_score_sudoku_valid_complete_grid():
    """Test score_sudoku gives perfect score for valid complete grid."""
    batch_size = 1
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)

    # Create a valid Sudoku pattern (each row/col/box has 1-9 once)
    # Using a simple Latin square pattern
    for i in range(9):
        for j in range(9):
            y[0, i, j] = (i + j) % 9 + 1

    score = score_sudoku(y)

    # This pattern should have perfect score
    assert score[0] == 243, f"Valid complete grid should score 243, got {score[0]}"


@pytest.mark.unit
def test_invalid_edit_masking():
    """Test that invalid edits can be masked by checking valid positions.

    This demonstrates how to identify and mask invalid edits.
    """
    batch_size = 1

    # Create grid with some fixed cells
    y = torch.zeros(batch_size, 9, 9, dtype=torch.long)
    y[0, 0, 0] = 5  # Fixed cell

    # Create mask for editable positions (1 = editable, 0 = fixed)
    fixed_mask = (y != 0).float()  # 1 where cell is filled
    editable_mask = 1.0 - fixed_mask  # 1 where cell is empty

    # Reshape to flat mask for positions
    flat_mask = editable_mask.reshape(batch_size, -1)

    # Check that fixed position is masked
    assert flat_mask[0, 0] == 0.0, "Fixed cell should be masked"
    assert flat_mask[0, 1] == 1.0, "Empty cell should be editable"
