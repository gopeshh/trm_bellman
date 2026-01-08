#!/usr/bin/env python3
"""
Quick sanity check for sudoku_is_solved termination.
"""

import torch
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_utils import sudoku_is_solved


def test_solved_termination():
    """Test that solved Sudoku terminates immediately."""
    print("=" * 60)
    print("Testing sudoku_is_solved termination in PlanEditEnv")
    print("=" * 60)

    # Valid 4x4 Sudoku solution
    solved_grid = torch.tensor([
        2, 3, 4, 5,
        4, 5, 2, 3,
        3, 2, 5, 4,
        5, 4, 3, 2
    ], dtype=torch.long)

    # Initial plan with one empty cell
    initial_plan = solved_grid.clone()
    initial_plan[0] = 1  # Empty cell

    class SudokuDataset:
        def __init__(self):
            self.data = [{
                "inputs": initial_plan.clone(),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": initial_plan.clone(),
            }]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    def checker(x, y):
        return float((y != 1).sum().item())

    dataset = SudokuDataset()
    cfg = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=6,
        solved_threshold=None,  # No threshold-based termination
        stop_action_mode="disabled",
    )

    env = PlanEditEnv(dataset, checker, cfg)
    env.set_stop_action_id(16 * 6)

    x, y = env.reset()
    print(f"Initial plan[0] = {y[0].item()} (should be 1 = empty)")
    print(f"sudoku_is_solved(initial) = {sudoku_is_solved(y)}")

    # Take action to complete the grid
    edit_action = 0 * 6 + 2  # position 0, token 2 (digit 1)
    (x_next, y_next), reward, done, info = env.step(edit_action)

    print(f"\nAfter edit:")
    print(f"  y_next[0] = {y_next[0].item()} (should be 2)")
    print(f"  sudoku_is_solved(y_next) = {sudoku_is_solved(y_next)}")
    print(f"  done = {done}")
    print(f"  done_reason = {info['done_reason']}")
    print(f"  solved = {info['solved']}")

    assert done is True, "Episode should terminate when grid is solved"
    assert info["done_reason"] == "solved", f"done_reason should be 'solved', got {info['done_reason']}"
    assert info["solved"] is True

    print("\n✅ PASS: Sudoku episode terminates when grid becomes solved!")
    return True


def test_non_solved_continues():
    """Test that non-solved Sudoku does not terminate."""
    print("\n" + "=" * 60)
    print("Testing that non-solved Sudoku does NOT terminate")
    print("=" * 60)

    # Partial grid (not solved)
    partial_grid = torch.tensor([
        2, 3, 4, 5,
        4, 5, 2, 3,
        3, 2, 1, 4,  # Empty cell at position 10
        5, 4, 3, 2
    ], dtype=torch.long)

    class PartialDataset:
        def __init__(self):
            self.data = [{
                "inputs": partial_grid.clone(),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": partial_grid.clone(),
            }]
        def __len__(self): return 1
        def __getitem__(self, idx): return self.data[idx]

    def checker(x, y):
        return float((y != 1).sum().item())

    dataset = PartialDataset()
    cfg = PlanEditEnvConfig(
        max_edits=16,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=6,
        solved_threshold=None,
        stop_action_mode="disabled",
    )

    env = PlanEditEnv(dataset, checker, cfg)
    env.set_stop_action_id(16 * 6)

    x, y = env.reset()
    print(f"Initial grid has empty cell: {(y == 1).sum().item()} empty cells")
    print(f"sudoku_is_solved(initial) = {sudoku_is_solved(y)}")

    # Take a no-op edit
    edit_action = 0 * 6 + 2
    (x_next, y_next), reward, done, info = env.step(edit_action)

    print(f"\nAfter edit:")
    print(f"  sudoku_is_solved(y_next) = {sudoku_is_solved(y_next)}")
    print(f"  done = {done}")
    print(f"  solved = {info['solved']}")

    assert done is False, "Episode should NOT terminate when grid is not solved"
    assert info["solved"] is False

    print("\n✅ PASS: Non-solved Sudoku correctly continues!")
    return True


if __name__ == "__main__":
    test_solved_termination()
    test_non_solved_continues()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
