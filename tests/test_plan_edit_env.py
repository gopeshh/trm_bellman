import torch
import pytest

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.task_config import SudokuTaskConfig


class DummyDataset:
    def __init__(self):
        # Single 1D "puzzle": x is a tensor [3], y is initialized to ones (empty cells)
        # In Sudoku encoding: 0 = PAD, 1 = empty cell, 2+ = digits
        self.data = [
            {"inputs": torch.tensor([1, 2, 3]), "puzzle_identifiers": torch.tensor([0])}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dummy_checker(x, y) -> float:
    # Reward is negative L1 distance between y and inputs
    inputs = x["inputs"]
    return float(-(inputs - y).abs().sum().item())


class SingleTokenDataset:
    def __init__(self):
        self.data = [
            {
                "inputs": torch.tensor([2]),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": torch.tensor([0]),
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def solved_checker(x, y) -> float:
    return 1.0 if torch.equal(y, x["inputs"]) else -1.0


def test_zero_edit_budget_is_rejected():
    with pytest.raises(ValueError, match="max_edits must be at least 1"):
        PlanEditEnv(
            DummyDataset(),
            dummy_checker,
            PlanEditEnvConfig(max_edits=0, gamma=0.99, vocab_size=4),
        )


@pytest.mark.parametrize("C_max", [-1.0, float("inf")])
def test_invalid_absorbing_boundary_is_rejected(C_max):
    with pytest.raises(ValueError, match="C_max must be finite and nonnegative"):
        PlanEditEnv(
            DummyDataset(),
            dummy_checker,
            PlanEditEnvConfig(
                max_edits=1,
                gamma=0.99,
                vocab_size=4,
                C_max=C_max,
            ),
        )


def test_plan_edit_env_step_and_stop():
    """
    Test that:
    1. Edit actions modify the plan correctly
    2. STOP action does NOT terminate (prevents STOP collapse)
    3. Episode terminates when max_edits is reached
    4. After termination, further steps are not allowed
    
    Note: STOP was intentionally changed to NOT terminate the episode
    to prevent the "STOP collapse" issue where the policy learns to
    always stop immediately. The agent should use all available edits.
    """
    dataset = DummyDataset()
    cfg = PlanEditEnvConfig(max_edits=3, gamma=0.99, reward_shaping=True, vocab_size=4)
    env = PlanEditEnv(dataset, dummy_checker, cfg)
    seq_len = dataset.data[0]["inputs"].numel()
    stop_id = seq_len * cfg.vocab_size
    env.set_stop_action_id(stop_id=stop_id)

    x, y = env.reset()
    assert env.step_count == 0
    assert env.done is False

    # Take a non-stop action (edit position 2 -> token 0)
    # The plan starts as inputs.clone() = [1, 2, 3], so we edit position 2 from 3 to 0
    edit_action = 2 * cfg.vocab_size + 0
    (x1, y1), r1, done1, _ = env.step(action=edit_action)
    assert env.step_count == 1
    assert done1 is False
    # Plan is initialized from inputs [1, 2, 3], position 2 edited to 0
    assert torch.equal(y1, torch.tensor([1, 2, 0]))

    # Take STOP action - this now does NOT terminate (to prevent STOP collapse)
    # Instead, it counts as a wasted step with a small penalty
    (x2, y2), r2, done2, info2 = env.step(action=stop_id)
    assert done2 is False, "STOP should not terminate (prevents STOP collapse)"
    assert env.done is False
    assert info2.get("terminated_by_stop") is True, "Should flag that STOP was chosen"
    assert torch.equal(y2, y1), "Plan should be unchanged after STOP"
    
    # Episode terminates when max_edits (3) is reached
    (x3, y3), r3, done3, _ = env.step(action=0)  # Step 3 -> terminates
    assert done3 is True, "Should terminate at max_edits"
    assert env.done is True

    # After termination, further steps should not be allowed
    try:
        env.step(action=0)
        assert False, "Expected an assertion when stepping after episode is done"
    except AssertionError:
        pass


def test_plan_edit_env_terminates_when_solved_threshold_met():
    dataset = SingleTokenDataset()
    cfg = PlanEditEnvConfig(
        max_edits=5,
        gamma=0.5,
        reward_shaping=True,
        vocab_size=3,
        solved_threshold=0.5,
    )
    env = PlanEditEnv(dataset, solved_checker, cfg)
    seq_len = dataset.data[0]["inputs"].numel()
    env.set_stop_action_id(stop_id=seq_len * cfg.vocab_size)

    x, y = env.reset()
    edit_action = 2  # set token at position 0 to value 2
    (_, y_next), reward, done, _ = env.step(action=edit_action)

    assert done is True
    assert env.done is True
    assert torch.equal(y_next, x["inputs"])

    phi_old = solved_checker(x, y)
    # The episodic implementation folds the absorbing-state tail into the
    # terminal transition, giving r_0 - Phi(s).
    expected_reward = -phi_old
    assert abs(reward - expected_reward) < 1e-6, f"Expected {expected_reward}, got {reward}"


def test_remaining_edit_clock_is_part_of_returned_state():
    dataset = DummyDataset()
    cfg = PlanEditEnvConfig(max_edits=2, gamma=0.99, vocab_size=4)
    env = PlanEditEnv(dataset, dummy_checker, cfg)
    env.set_stop_action_id(dataset.data[0]["inputs"].numel() * cfg.vocab_size)

    x0, _ = env.reset(idx=0)
    assert x0["remaining_edits"].item() == 2
    (x1, _), _, done, _ = env.step(0)
    assert not done
    assert x1["remaining_edits"].item() == 1
    assert x0["remaining_edits"].item() == 2


def test_plan_edit_env_threshold_works_without_reward_shaping():
    dataset = SingleTokenDataset()
    cfg = PlanEditEnvConfig(
        max_edits=2,
        gamma=0.9,
        reward_shaping=False,
        vocab_size=3,
        solved_threshold=0.5,
        C_max=2.0,
    )
    env = PlanEditEnv(dataset, solved_checker, cfg)
    seq_len = dataset.data[0]["inputs"].numel()
    env.set_stop_action_id(stop_id=seq_len * cfg.vocab_size)

    env.reset()
    (_, _), reward, done, _ = env.step(action=2)

    assert done is True
    assert abs(reward - (1.0 - 0.9 * 2.0)) < 1e-6


def test_terminal_stop_folds_sparse_absorbing_boundary_once():
    dataset = SingleTokenDataset()
    cfg = PlanEditEnvConfig(
        max_edits=2,
        gamma=0.5,
        reward_shaping=False,
        vocab_size=3,
        stop_action_mode="terminal",
        fail_terminal_reward=-3.0,
        C_max=2.0,
    )
    env = PlanEditEnv(dataset, solved_checker, cfg)
    stop_id = dataset.data[0]["inputs"].numel() * cfg.vocab_size
    env.set_stop_action_id(stop_id)
    env.reset()

    (_, _), reward, done, info = env.step(stop_id)

    assert done is True
    assert info["done_reason"] == "stop"
    assert info["terminated_by_stop"] is True
    assert abs(reward - (-1.0 - 3.0 - 0.5 * 2.0)) < 1e-6


def test_action_masking():
    """
    Test that action masking correctly identifies 'given' cells (clues)
    and prevents them from being edited.
    """
    # Create dataset with 1 empty cell (value=1) and 1 clue cell (value=2)
    # 0=PAD, 1=Empty, 2=Value
    inputs = torch.tensor([1, 2], dtype=torch.long) 
    
    class MockDataset:
        def __init__(self, data):
            self.data = [{"inputs": data, "puzzle_identifiers": torch.tensor([0])}]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    dataset = MockDataset(inputs)
    
    # vocab_size=3 (0, 1, 2)
    cfg = PlanEditEnvConfig(max_edits=10, gamma=0.99, vocab_size=3, stop_action_mode="noop")
    env = PlanEditEnv(dataset, dummy_checker, cfg)
    
    # 2 positions * 3 tokens = 6 edit actions. STOP is action 6.
    stop_id = 6
    env.set_stop_action_id(stop_id)
    
    env.reset(0)
    mask = env.get_action_mask()
    
    # Position 0 (Value=1, Empty):
    # Should allow editing (setting to tokens).
    # Tokens 0 and 1 are masked out by default in _compute_action_mask (range(min(2, vocab_size)))
    # So for pos 0:
    # Action 0 (pos=0, tok=0): False (PAD)
    # Action 1 (pos=0, tok=1): False (Empty)
    # Action 2 (pos=0, tok=2): True (Value)
    
    # Position 1 (Value=2, Clue):
    # Should be COMPLETELY masked.
    # Action 3 (pos=1, tok=0): False
    # Action 4 (pos=1, tok=1): False
    # Action 5 (pos=1, tok=2): False
    
    print(f"Mask: {mask}")
    
    # Verify Pos 0
    assert mask[0].item() is False, "Pos 0, Tok 0 (PAD) should be masked"
    assert mask[1].item() is False, "Pos 0, Tok 1 (Empty) should be masked"
    assert mask[2].item() is True,  "Pos 0, Tok 2 (Value) should be allowed"
    
    # Verify Pos 1 (Clue)
    assert mask[3].item() is False, "Pos 1 (Clue) should be masked"
    assert mask[4].item() is False, "Pos 1 (Clue) should be masked"
    assert mask[5].item() is False, "Pos 1 (Clue) should be masked"
    
    # Verify STOP
    assert mask[6].item() is True, "STOP should be allowed"


def test_action_masking_comprehensive():
    """
    Comprehensive test for action masking logic across a batch.
    Verifies that for every cell with value > 1 (clue), ALL corresponding edit actions are masked.
    This corresponds to the fix for 'Action masking bugs (agent edits clues)' in IMPLEMENTATION_GUIDELINE.md.
    """
    vocab_size = 5 # 0, 1, 2, 3, 4
    seq_len = 4
    # inputs: [2, 1, 3, 1] -> Clue, Empty, Clue, Empty
    # Clues are at pos 0 (val 2) and pos 2 (val 3).
    inputs = torch.tensor([[2, 1, 3, 1], [1, 4, 1, 2]], dtype=torch.long)
    batch_size = 2
    
    stop_id = seq_len * vocab_size
    
    mask = PlanEditEnv.compute_batch_action_mask(
        inputs, vocab_size, stop_id, stop_mode="noop"
    )
    
    # Check shape: [B, num_actions] = [2, 4*5 + 1] = [2, 21]
    assert mask.shape == (2, 21)
    
    # Check Batch 0: [2, 1, 3, 1]
    # Pos 0 (Clue): Actions 0-4 should be False
    assert not mask[0, 0:5].any(), "Batch 0 Pos 0 is clue, should be fully masked"
    # Pos 1 (Empty): Actions 5-9. 
    # Tokens 0, 1 are masked (invalid). Tokens 2,3,4 allowed.
    assert not mask[0, 5].item() # Tok 0
    assert not mask[0, 6].item() # Tok 1
    assert mask[0, 7].item()     # Tok 2
    # Pos 2 (Clue): Actions 10-14 masked
    assert not mask[0, 10:15].any(), "Batch 0 Pos 2 is clue, should be fully masked"
    # Pos 3 (Empty): Actions 15-19. Tok 2,3,4 allowed.
    assert mask[0, 17].item()
    
    # Check Batch 1: [1, 4, 1, 2]
    # Pos 0 (Empty): Allowed
    assert mask[1, 2].item()
    # Pos 1 (Clue): Masked
    assert not mask[1, 5:10].any(), "Batch 1 Pos 1 is clue, should be fully masked"
    # Pos 2 (Empty): Allowed
    assert mask[1, 12].item()
    # Pos 3 (Clue): Masked
    assert not mask[1, 15:20].any(), "Batch 1 Pos 3 is clue, should be fully masked"
    
    print("Comprehensive masking test passed!")


def test_sudoku_no_mask_flag_restores_given_cell_only_actions():
    inputs = torch.tensor([
        2, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    ], dtype=torch.long)

    class SudokuDataset:
        def __init__(self, data):
            self.data = [{"inputs": data, "puzzle_identifiers": torch.tensor([0])}]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = SudokuDataset(inputs)
    stop_id = 16 * 6
    conflict_action = 1 * 6 + 2

    masked_cfg = PlanEditEnvConfig(
        max_edits=4,
        gamma=0.99,
        vocab_size=6,
        task_type="sudoku",
        stop_action_mode="disabled",
    )
    masked_env = PlanEditEnv(
        dataset,
        dummy_checker,
        masked_cfg,
        task_config=SudokuTaskConfig(),
    )
    masked_env.set_stop_action_id(stop_id)
    masked_env.reset(0)
    masked_mask = masked_env.get_action_mask()

    unmasked_cfg = PlanEditEnvConfig(
        max_edits=4,
        gamma=0.99,
        vocab_size=6,
        task_type="sudoku",
        stop_action_mode="disabled",
        disable_constraint_masking=True,
    )
    unmasked_env = PlanEditEnv(
        dataset,
        dummy_checker,
        unmasked_cfg,
        task_config=SudokuTaskConfig(disable_constraint_masking=True),
    )
    unmasked_env.set_stop_action_id(stop_id)
    unmasked_env.reset(0)
    unmasked_mask = unmasked_env.get_action_mask()

    assert masked_env._use_incremental_masking is True
    assert unmasked_env._use_incremental_masking is False
    assert not masked_mask[conflict_action]
    assert unmasked_mask[conflict_action]
    assert not unmasked_mask[:6].any()


def test_sudoku_no_mask_behavior_persists_after_step():
    inputs = torch.tensor([
        2, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    ], dtype=torch.long)

    class SudokuDataset:
        def __init__(self, data):
            self.data = [{"inputs": data, "puzzle_identifiers": torch.tensor([0])}]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = SudokuDataset(inputs)
    cfg = PlanEditEnvConfig(
        max_edits=4,
        gamma=0.99,
        vocab_size=6,
        task_type="sudoku",
        stop_action_mode="disabled",
        disable_constraint_masking=True,
    )
    env = PlanEditEnv(
        dataset,
        dummy_checker,
        cfg,
        task_config=SudokuTaskConfig(disable_constraint_masking=True),
    )
    stop_id = 16 * 6
    env.set_stop_action_id(stop_id)
    env.reset(0)

    initial_mask = env.get_action_mask().clone()
    first_edit = 1 * 6 + 2
    (_, _), _, done, _ = env.step(first_edit)
    assert not done

    updated_mask = env.get_action_mask()
    second_conflict_action = 2 * 6 + 2

    assert env._use_incremental_masking is False
    assert updated_mask[second_conflict_action]
    assert not updated_mask[:6].any()
    assert torch.equal(initial_mask, updated_mask)


def test_sudoku_terminates_when_solved_via_sudoku_is_solved():
    """
    Test that a Sudoku episode terminates immediately when the grid becomes solved,
    using the solution-independent sudoku_is_solved() criterion.

    This test verifies Fix #1 for feasibility checker:
    - When solved_threshold is None, episodes still terminate via sudoku_is_solved()
    - Prevents "solved then unsolved" scenarios that could inflate success metrics
    """
    # Create a 4x4 Sudoku dataset where the grid is almost solved
    # The plan starts with one empty cell (position 0), rest are solved
    # After one edit, the grid becomes solved

    # Valid 4x4 Sudoku solution (token encoding: digit d -> token d+1)
    # 1 2 3 4 -> tokens 2 3 4 5
    # 3 4 1 2 -> tokens 4 5 2 3
    # 2 1 4 3 -> tokens 3 2 5 4
    # 4 3 2 1 -> tokens 5 4 3 2
    solved_grid = torch.tensor([
        2, 3, 4, 5,
        4, 5, 2, 3,
        3, 2, 5, 4,
        5, 4, 3, 2
    ], dtype=torch.long)

    # Create initial plan with one empty cell at position 0
    initial_plan = solved_grid.clone()
    initial_plan[0] = 1  # Empty cell (token 1)

    class SudokuDataset:
        def __init__(self):
            # inputs has clue at position 0 as empty (1) to allow editing
            # All other positions are clues (>1)
            self.data = [{
                "inputs": initial_plan.clone(),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": initial_plan.clone(),
            }]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    def feasibility_checker(x, y):
        """Simple checker that returns filled count (for testing)."""
        return float((y != 1).sum().item())

    dataset = SudokuDataset()

    # Create config with solved_threshold=None (like feasibility configs)
    # IMPORTANT: Set task_type="sudoku" to enable solution-independent termination
    cfg = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=6,  # tokens 0-5 for 4x4 Sudoku
        solved_threshold=None,  # No threshold-based termination
        stop_action_mode="disabled",  # STOP disabled
        task_type="sudoku",  # Required for sudoku_is_solved() termination
    )

    env = PlanEditEnv(dataset, feasibility_checker, cfg)
    seq_len = 16
    stop_id = seq_len * cfg.vocab_size
    env.set_stop_action_id(stop_id)

    x, y = env.reset()

    # Verify initial state
    assert not env.done
    assert y[0].item() == 1, "Position 0 should be empty initially"

    # Take an edit action to set position 0 to digit 1 (token 2)
    # This completes the solved grid
    edit_action = 0 * cfg.vocab_size + 2  # position 0, token 2 (digit 1)
    (x_next, y_next), reward, done, info = env.step(edit_action)

    # The grid should now be solved and episode should terminate
    assert done is True, "Episode should terminate when grid is solved"
    assert info["done_reason"] == "solved", f"done_reason should be 'solved', got {info['done_reason']}"
    assert info["solved"] is True, "info['solved'] should be True"
    assert info["terminated_by_solved"] is True, "info['terminated_by_solved'] should be True"

    # Verify the grid is actually solved
    assert y_next[0].item() == 2, "Position 0 should now be digit 1 (token 2)"
    assert torch.equal(y_next, solved_grid), "Grid should match the solved grid"

    print("Sudoku solution-independent termination test passed!")


def test_sudoku_does_not_terminate_when_not_solved():
    """
    Test that a Sudoku episode does NOT terminate when the grid is not fully solved,
    even if it's partially filled.

    This ensures sudoku_is_solved() correctly requires:
    - All cells filled (no empty cells)
    - Zero violations
    """
    # Create a 4x4 grid that is NOT solved (has empty cells)
    partial_grid = torch.tensor([
        2, 3, 4, 5,  # Row 0: filled
        4, 5, 2, 3,  # Row 1: filled
        3, 2, 1, 4,  # Row 2: one empty cell at position 10
        5, 4, 3, 2   # Row 3: filled
    ], dtype=torch.long)

    class PartialSudokuDataset:
        def __init__(self):
            self.data = [{
                "inputs": partial_grid.clone(),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": partial_grid.clone(),
            }]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    def feasibility_checker(x, y):
        return float((y != 1).sum().item())

    dataset = PartialSudokuDataset()
    cfg = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=6,
        solved_threshold=None,
        stop_action_mode="disabled",
        task_type="sudoku",  # Even with task_type="sudoku", should not terminate if not fully solved
    )

    env = PlanEditEnv(dataset, feasibility_checker, cfg)
    env.set_stop_action_id(16 * cfg.vocab_size)

    x, y = env.reset()

    # Take a no-op edit (edit position 0 to same value)
    edit_action = 0 * cfg.vocab_size + 2  # position 0, token 2 (same as current)
    (x_next, y_next), reward, done, info = env.step(edit_action)

    # Episode should NOT terminate (grid still has empty cell)
    assert done is False, "Episode should not terminate when grid is not solved"
    assert info["solved"] is False, "info['solved'] should be False"

    print("Sudoku non-termination test passed!")


def test_non_sudoku_task_does_not_trigger_sudoku_termination():
    """
    Test that a non-Sudoku task (task_type != "sudoku") does NOT trigger
    early termination via sudoku_is_solved(), even if the plan has 16 or 81 elements
    and happens to satisfy Sudoku constraints.

    This verifies the fix that guards sudoku_is_solved() termination with task_type check.
    """
    # Use the same solved 4x4 Sudoku grid
    solved_grid = torch.tensor([
        2, 3, 4, 5,
        4, 5, 2, 3,
        3, 2, 5, 4,
        5, 4, 3, 2
    ], dtype=torch.long)

    # Create initial plan with one empty cell at position 0
    initial_plan = solved_grid.clone()
    initial_plan[0] = 1  # Empty cell (token 1)

    class NonSudokuDataset:
        def __init__(self):
            self.data = [{
                "inputs": initial_plan.clone(),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": initial_plan.clone(),
            }]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    def dummy_checker(x, y):
        return float((y != 1).sum().item())

    dataset = NonSudokuDataset()

    # Create config with task_type="dummy" (NOT "sudoku")
    cfg = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=6,
        solved_threshold=None,  # No threshold-based termination
        stop_action_mode="disabled",
        task_type="dummy",  # NOT "sudoku" - should NOT trigger sudoku_is_solved()
    )

    env = PlanEditEnv(dataset, dummy_checker, cfg)
    seq_len = 16
    stop_id = seq_len * cfg.vocab_size
    env.set_stop_action_id(stop_id)

    x, y = env.reset()

    # Take the same edit action that would complete the grid
    edit_action = 0 * cfg.vocab_size + 2  # position 0, token 2 (digit 1)
    (x_next, y_next), reward, done, info = env.step(edit_action)

    # The grid IS solved, but task_type="dummy" so sudoku_is_solved() should NOT trigger
    assert done is False, "Episode should NOT terminate for non-Sudoku task_type"
    assert info["solved"] is False, "info['solved'] should be False for non-Sudoku task"
    assert info["terminated_by_solved"] is False, "terminated_by_solved should be False"
    assert info["done_reason"] is None, "done_reason should be None (episode continues)"

    # Verify the grid matches the solved Sudoku (it IS solved, just not triggering termination)
    assert torch.equal(y_next, solved_grid), "Grid should match solved grid"

    print("Non-Sudoku task_type termination guard test passed!")


def test_task_type_none_does_not_trigger_sudoku_termination():
    """
    Test that task_type=None (default) does NOT trigger sudoku_is_solved() termination.
    This is the safe default behavior.
    """
    solved_grid = torch.tensor([
        2, 3, 4, 5,
        4, 5, 2, 3,
        3, 2, 5, 4,
        5, 4, 3, 2
    ], dtype=torch.long)

    initial_plan = solved_grid.clone()
    initial_plan[0] = 1

    class DefaultTaskDataset:
        def __init__(self):
            self.data = [{
                "inputs": initial_plan.clone(),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": initial_plan.clone(),
            }]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    def dummy_checker(x, y):
        return float((y != 1).sum().item())

    dataset = DefaultTaskDataset()

    # Create config with task_type=None (default)
    cfg = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=6,
        solved_threshold=None,
        stop_action_mode="disabled",
        # task_type defaults to None - should NOT trigger sudoku_is_solved()
    )

    env = PlanEditEnv(dataset, dummy_checker, cfg)
    env.set_stop_action_id(16 * cfg.vocab_size)

    x, y = env.reset()

    edit_action = 0 * cfg.vocab_size + 2
    (x_next, y_next), reward, done, info = env.step(edit_action)

    # task_type=None should NOT trigger sudoku_is_solved()
    assert done is False, "Episode should NOT terminate when task_type is None"
    assert info["solved"] is False, "info['solved'] should be False when task_type is None"

    print("task_type=None termination guard test passed!")
