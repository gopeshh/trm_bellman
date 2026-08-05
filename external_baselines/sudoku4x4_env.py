from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .npy_reader import load_npy

try:  # Optional: only needed when an actual Gym/SB3 backend is installed.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only when packages are absent
    np = None

if TYPE_CHECKING:
    # The Buck/SB3 build uses Gym. Runtime fallback remains available for
    # standalone installs that provide only Gymnasium.
    import gym as gym_api
    from gym import spaces

    GYM_BACKEND = "gym"
else:
    try:  # SB3 in Buck is pinned to a Gym-era release, so prefer gym when available.
        import gym as gym_api
        from gym import spaces

        GYM_BACKEND = "gym"
    except ImportError:  # pragma: no cover - exercised only when gym is absent
        try:
            import gymnasium as gym_api
            from gymnasium import spaces

            GYM_BACKEND = "gymnasium"
        except ImportError:  # pragma: no cover - exercised only when packages are absent
            gym_api = None
            spaces = None
            GYM_BACKEND = None


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sudoku-4x4-easy_6to8empties"


def _token_to_digit(token: int) -> int:
    if token <= 1:
        return 0
    return token - 1


def count_sudoku_violations_4x4(plan: list[int]) -> int:
    grid = [_token_to_digit(token) for token in plan]
    violations = 0

    for row in range(4):
        digits = [grid[row * 4 + col] for col in range(4) if grid[row * 4 + col] > 0]
        violations += len(digits) - len(set(digits))

    for col in range(4):
        digits = [grid[row * 4 + col] for row in range(4) if grid[row * 4 + col] > 0]
        violations += len(digits) - len(set(digits))

    for box_row in range(2):
        for box_col in range(2):
            digits = []
            for row in range(box_row * 2, box_row * 2 + 2):
                for col in range(box_col * 2, box_col * 2 + 2):
                    value = grid[row * 4 + col]
                    if value > 0:
                        digits.append(value)
            violations += len(digits) - len(set(digits))

    return violations


def sudoku_zero_candidate_cells_4x4(plan: list[int]) -> int:
    grid = [_token_to_digit(token) for token in plan]
    zero_candidate = 0

    for row in range(4):
        for col in range(4):
            if grid[row * 4 + col] != 0:
                continue

            candidates = {1, 2, 3, 4}
            for other_col in range(4):
                candidates.discard(grid[row * 4 + other_col])
            for other_row in range(4):
                candidates.discard(grid[other_row * 4 + col])

            box_row = (row // 2) * 2
            box_col = (col // 2) * 2
            for other_row in range(box_row, box_row + 2):
                for other_col in range(box_col, box_col + 2):
                    candidates.discard(grid[other_row * 4 + other_col])

            candidates.discard(0)
            if not candidates:
                zero_candidate += 1

    return zero_candidate


def sudoku_is_solved_4x4(plan: list[int]) -> bool:
    if any(token <= 1 for token in plan):
        return False

    grid = [_token_to_digit(token) for token in plan]
    target = {1, 2, 3, 4}

    for row in range(4):
        if {grid[row * 4 + col] for col in range(4)} != target:
            return False
    for col in range(4):
        if {grid[row * 4 + col] for row in range(4)} != target:
            return False
    for box_row in range(2):
        for box_col in range(2):
            digits = {
                grid[row * 4 + col]
                for row in range(box_row * 2, box_row * 2 + 2)
                for col in range(box_col * 2, box_col * 2 + 2)
            }
            if digits != target:
                return False
    return True


class Sudoku4x4ExternalEnv:
    """
    Minimal hard-4x4 no-mask environment for external baseline smoke tests.

    Observation semantics intentionally mirror the internal setup:
    - `inputs`: original puzzle tokens
    - `plan`: current editable plan
    - `action_mask`: same flat action layout as the internal environment

    The smoke harness keeps the implementation dependency-light so the wrapper
    can be validated even before Gym/SB3 packages are installed.
    """

    grid_size = 4
    seq_len = 16
    vocab_size = 6
    stop_action_id = seq_len * vocab_size
    action_space_n = stop_action_id + 1

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        split: str = "train",
        max_edits: int = 16,
        gamma: float = 0.99,
        protocol: str = "no-mask",
        stop_action_mode: str = "disabled",
        stop_action_penalty: float = -0.1,
        fail_terminal_reward: float = -16.0,
        solve_terminal_reward: float = 1.0,
        feasibility_violation_weight: float = 2.0,
        feasibility_zerocand_weight: float = 5.0,
    ):
        if protocol != "no-mask":
            raise ValueError(f"Unsupported protocol {protocol!r}; only 'no-mask' is implemented")
        if stop_action_mode != "disabled":
            raise ValueError(
                f"Unsupported stop_action_mode {stop_action_mode!r}; expected 'disabled'"
            )
        if max_edits < 1:
            raise ValueError("max_edits must be at least 1")

        dataset_dir = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir()
        split_dir = dataset_dir / split
        metadata_path = split_dir / "dataset.json"
        inputs_path = split_dir / "all__inputs.npy"
        labels_path = split_dir / "all__labels.npy"

        metadata = json.loads(metadata_path.read_text())
        self.dataset_dir = dataset_dir
        self.split = split
        self.protocol = protocol
        self.max_edits = max_edits
        self.gamma = gamma
        self.stop_action_mode = stop_action_mode
        self.stop_action_penalty = stop_action_penalty
        self.fail_terminal_reward = fail_terminal_reward
        self.solve_terminal_reward = solve_terminal_reward
        self.feasibility_violation_weight = feasibility_violation_weight
        self.feasibility_zerocand_weight = feasibility_zerocand_weight
        self.metadata = metadata
        self.inputs_data = load_npy(inputs_path)
        self.labels_data = load_npy(labels_path)

        if metadata["seq_len"] != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {metadata['seq_len']}")
        if metadata["vocab_size"] != self.vocab_size:
            raise ValueError(
                f"Expected vocab_size {self.vocab_size}, got {metadata['vocab_size']}"
            )

        self._rng = random.Random(0)
        self._episode_return = 0.0
        self._last_puzzle_index = 0
        self._steps = 0
        self.inputs: list[int] = []
        self.solution: list[int] = []
        self.plan: list[int] = []

    @property
    def observation_spec(self) -> dict[str, Any]:
        return {
            "inputs": {"shape": (self.seq_len,), "dtype": "int", "semantics": "original puzzle"},
            "plan": {"shape": (self.seq_len,), "dtype": "int", "semantics": "current editable plan"},
            "remaining_edits": {
                "shape": (1,),
                "dtype": "int",
                "semantics": "remaining finite-horizon edit budget",
            },
            "action_mask": {
                "shape": (self.action_space_n,),
                "dtype": "bool",
                "semantics": "flat edit mask with STOP in the final slot",
            },
        }

    def _checker(self, plan: list[int]) -> float:
        filled = sum(1 for token in plan if token != 1)
        violations = count_sudoku_violations_4x4(plan)
        zero_candidate = sudoku_zero_candidate_cells_4x4(plan)
        return float(
            filled
            - self.feasibility_violation_weight * violations
            - self.feasibility_zerocand_weight * zero_candidate
        )

    def get_action_mask(self) -> list[bool]:
        mask = [True] * self.action_space_n
        for position, token in enumerate(self.inputs):
            start = position * self.vocab_size
            if token > 1:
                for action in range(start, start + self.vocab_size):
                    mask[action] = False
            mask[start] = False
            mask[start + 1] = False
        mask[self.stop_action_id] = False
        return mask

    def _make_observation(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "plan": list(self.plan),
            "remaining_edits": [max(self.max_edits - self._steps, 0)],
            "action_mask": self.get_action_mask(),
        }

    def reset(
        self, seed: Optional[int] = None, puzzle_index: Optional[int] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        if puzzle_index is None:
            puzzle_index = self._rng.randrange(len(self.inputs_data))

        self._last_puzzle_index = puzzle_index
        self.inputs = list(self.inputs_data[puzzle_index])
        self.solution = list(self.labels_data[puzzle_index])
        self.plan = list(self.inputs)
        self._steps = 0
        self._episode_return = 0.0

        info = {
            "puzzle_index": puzzle_index,
            "protocol": self.protocol,
            "action_space_n": self.action_space_n,
            "observation_spec": self.observation_spec,
        }
        return self._make_observation(), info

    def _apply_action(self, action: int) -> tuple[list[int], bool, bool]:
        mask = self.get_action_mask()

        if action == self.stop_action_id:
            # In the internal env, disabled STOP is masked out but still behaves
            # like a no-op with penalty rather than an invalid action.
            return list(self.plan), True, False
        invalid_action = action < 0 or action >= self.action_space_n or not mask[action]
        if invalid_action:
            return list(self.plan), False, True

        position = action // self.vocab_size
        token = action % self.vocab_size
        next_plan = list(self.plan)
        next_plan[position] = token
        return next_plan, False, False

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        phi_old = self._checker(self.plan)
        self._steps += 1

        next_plan, selected_stop, invalid_action = self._apply_action(int(action))
        stop_penalty = self.stop_action_penalty if selected_stop else 0.0

        self.plan = next_plan
        phi_new = self._checker(self.plan)
        solved = sudoku_is_solved_4x4(self.plan)
        reached_budget = self._steps >= self.max_edits

        # The edit budget is part of the MDP state, so budget exhaustion is a
        # true terminal state. It is not an external time-limit truncation.
        terminated = solved or reached_budget
        truncated = False
        terminal_bonus = 0.0
        if solved:
            terminal_bonus = self.solve_terminal_reward
        elif reached_budget:
            terminal_bonus = self.fail_terminal_reward

        if terminated:
            # Match PlanEditEnv: fold the absorbing-state tail into the
            # terminal transition instead of bootstrapping after episode end.
            reward = terminal_bonus - phi_old + stop_penalty
        else:
            reward = terminal_bonus + self.gamma * phi_new - phi_old + stop_penalty
        self._episode_return += reward

        info = {
            "puzzle_index": self._last_puzzle_index,
            "steps": self._steps,
            "reward_components": {
                "phi_old": phi_old,
                "phi_new": phi_new,
                "terminal_bonus": terminal_bonus,
                "stop_penalty": stop_penalty,
            },
            "solved": solved,
            "invalid_action": invalid_action,
            "done_reason": "solved" if solved else ("budget" if reached_budget else None),
            "episode_return": self._episode_return,
        }
        return self._make_observation(), reward, terminated, truncated, info


if gym_api is not None and spaces is not None and np is not None:  # pragma: no branch

    class GymSudoku4x4Env(gym_api.Env):
        metadata = {"render.modes": []} if GYM_BACKEND == "gym" else {"render_modes": []}

        def __init__(self, **kwargs: Any):
            super().__init__()
            self.core = Sudoku4x4ExternalEnv(**kwargs)
            self.action_space = spaces.Discrete(self.core.action_space_n)
            self.observation_space = spaces.Dict(
                {
                    "inputs": spaces.MultiDiscrete([self.core.vocab_size] * self.core.seq_len),
                    "plan": spaces.MultiDiscrete([self.core.vocab_size] * self.core.seq_len),
                    "remaining_edits": spaces.MultiDiscrete([self.core.max_edits + 1]),
                    "action_mask": spaces.MultiBinary(self.core.action_space_n),
                }
            )

        def _convert_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
            return {
                "inputs": np.asarray(obs["inputs"], dtype=np.int64),
                "plan": np.asarray(obs["plan"], dtype=np.int64),
                "remaining_edits": np.asarray(obs["remaining_edits"], dtype=np.int64),
                "action_mask": np.asarray(obs["action_mask"], dtype=np.int8),
            }

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
            puzzle_index = None if options is None else options.get("puzzle_index")
            obs, info = self.core.reset(seed=seed, puzzle_index=puzzle_index)
            if GYM_BACKEND == "gym":
                return self._convert_obs(obs)
            return self._convert_obs(obs), info

        def step(self, action: int):
            obs, reward, terminated, truncated, info = self.core.step(int(action))
            if GYM_BACKEND == "gym":
                done = bool(terminated or truncated)
                return self._convert_obs(obs), float(reward), done, info
            return self._convert_obs(obs), float(reward), terminated, truncated, info

        def seed(self, seed: Optional[int] = None):
            if seed is not None:
                self.core._rng.seed(seed)
            return [seed]

        def action_masks(self):
            return np.asarray(self.core.get_action_mask(), dtype=np.bool_)

else:
    GymSudoku4x4Env = None
