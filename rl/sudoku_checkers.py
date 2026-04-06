"""Shared checker functions for tokenized Sudoku training and evaluation."""

from __future__ import annotations

from typing import Any, Callable

import torch

from rl.sudoku_utils import (
    count_sudoku_violations_4x4,
    count_sudoku_violations_9x9,
    sudoku_filled_cells,
    sudoku_zero_candidate_cells,
)


def _to_plan_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def dummy_checker(x: dict[str, Any], y: Any) -> float:
    """Fallback checker based on negative L1 distance from the input state."""
    target = x["inputs"]
    plan = _to_plan_tensor(y)
    target = target.to(torch.float32)
    plan = plan.to(torch.float32)
    return float(-(target - plan).abs().mean().item())


def sudoku_solution_checker(x: dict[str, Any], y: Any, scale_factor: float = 10.0) -> float:
    """Return a bounded score based on exact token matches with the known solution."""
    solution = x.get("solution")
    if solution is None:
        return dummy_checker(x, y)

    plan = _to_plan_tensor(y).to(torch.long)
    solution_tensor = _to_plan_tensor(solution).to(torch.long)
    if plan.shape != solution_tensor.shape:
        solution_tensor = solution_tensor.view_as(plan)
    matches = (plan == solution_tensor).to(torch.float32)
    return float(matches.mean().item() * scale_factor)


def sudoku_constraint_checker(
    x: dict[str, Any],
    y: Any,
    max_violations_4x4: int = 24,
    max_violations_9x9: int = 162,
) -> float:
    """Constraint-based score normalized to the `[0, 10]` range."""
    plan = _to_plan_tensor(y).to(torch.long)
    total_cells = plan.numel()

    if total_cells == 16:
        violations = count_sudoku_violations_4x4(plan)
        max_violations = max_violations_4x4
    elif total_cells == 81:
        violations = count_sudoku_violations_9x9(plan)
        max_violations = max_violations_9x9
    else:
        return sudoku_solution_checker(x, y)

    score = 10.0 * (1.0 - min(violations, max_violations) / max_violations)
    return float(score)


def sudoku_progress_checker(x: dict[str, Any], y: Any, violation_penalty: float = 2.0) -> float:
    """Reward filled cells while penalizing constraint violations."""
    plan = _to_plan_tensor(y).to(torch.long)
    total_cells = plan.numel()

    if total_cells == 16:
        filled_cells = (plan != 1).sum().item()
        violations = count_sudoku_violations_4x4(plan)
    elif total_cells == 81:
        filled_cells = (plan != 1).sum().item()
        violations = count_sudoku_violations_9x9(plan)
    else:
        return sudoku_solution_checker(x, y)

    if violations == 0:
        return float(filled_cells)
    return float(filled_cells - violations * violation_penalty)


def sudoku_feasibility_checker(x: dict[str, Any], y: Any, w_v: float = 2.0, w_z: float = 5.0) -> float:
    """Reward fill progress while penalizing violations and dead-end cells."""
    plan = _to_plan_tensor(y).to(torch.long)
    total_cells = plan.numel()

    if total_cells == 16:
        filled = sudoku_filled_cells(plan, empty_token=1)
        violations = count_sudoku_violations_4x4(plan)
        zero_cand = sudoku_zero_candidate_cells(plan, grid_size=4)
    elif total_cells == 81:
        filled = sudoku_filled_cells(plan, empty_token=1)
        violations = count_sudoku_violations_9x9(plan)
        zero_cand = sudoku_zero_candidate_cells(plan, grid_size=9)
    else:
        return sudoku_solution_checker(x, y)

    return float(filled - w_v * violations - w_z * zero_cand)


def make_sudoku_feasibility_checker(w_v: float = 2.0, w_z: float = 5.0) -> Callable[[dict[str, Any], Any], float]:
    """Build a configured feasibility checker with fixed weighting parameters."""

    def checker(x: dict[str, Any], y: Any) -> float:
        return sudoku_feasibility_checker(x, y, w_v=w_v, w_z=w_z)

    checker.__name__ = "sudoku_feasibility_checker"
    return checker


sudoku_checker = sudoku_solution_checker

