"""Meta-MDP environment for recursive reasoning with edit actions.

This module defines the MDP where:
- States are (x, y) pairs: input and current internal representation
- Actions are edits to the internal state
- Rewards are based on task progress
- Transitions follow model dynamics with applied edits
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch


def score_sudoku(y: torch.Tensor) -> torch.Tensor:
    """Score a Sudoku grid based on constraint violations.

    Counts the number of valid constraints satisfied. A perfect Sudoku
    has all constraints satisfied.

    Args:
        y: Sudoku grid (B, 9, 9) with values 0-9 (0 = empty)

    Returns:
        Score tensor (B,) - higher is better, max score = 27 per puzzle
    """
    batch_size = y.shape[0]
    score = torch.zeros(batch_size, device=y.device)

    # Check each row (9 constraints)
    for i in range(9):
        row = y[:, i, :]
        # Valid row has no duplicate non-zero values
        for val in range(1, 10):
            mask = row == val
            # Count rows where this value appears at most once
            count = mask.sum(dim=1)
            score += (count <= 1).float()

    # Check each column (9 constraints)
    for j in range(9):
        col = y[:, :, j]
        for val in range(1, 10):
            mask = col == val
            count = mask.sum(dim=1)
            score += (count <= 1).float()

    # Check each 3x3 box (9 constraints)
    for box_i in range(3):
        for box_j in range(3):
            box = y[:, box_i * 3 : (box_i + 1) * 3, box_j * 3 : (box_j + 1) * 3]
            box_flat = box.reshape(batch_size, -1)
            for val in range(1, 10):
                mask = box_flat == val
                count = mask.sum(dim=1)
                score += (count <= 1).float()

    return score


class SudokuMetaMDP:
    """Meta-MDP for Sudoku puzzle solving with plan edits.

    Applies edits to Sudoku grids, recomputes internal states, and
    provides improvement-based rewards.

    Attributes:
        edit_penalty: Penalty λ for making an edit (encourages fewer edits)
        device: Device for tensor operations
    """

    def __init__(self, edit_penalty: float = 0.01, device: str = "cpu"):
        """Initialize Sudoku Meta-MDP.

        Args:
            edit_penalty: Penalty λ applied per edit (default: 0.01)
            device: Device for tensor operations
        """
        self.edit_penalty = edit_penalty
        self.device = device

    def apply_edit(
        self, y: torch.Tensor, a: Tuple[torch.Tensor, torch.Tensor], rules: Optional[str] = "sudoku"
    ) -> torch.Tensor:
        """Apply edit action to Sudoku grid.

        Args:
            y: Current grid (B, 9, 9) with values 0-9
            a: Action tuple (pos, val) where:
                - pos: Position indices (B,) in [0, 81)
                - val: Value indices (B,) in [0, 9] (0=erase, 1-9=digits)
            rules: Task rules (default: "sudoku")

        Returns:
            Updated grid y' (B, 9, 9) after applying edits
        """
        pos, val = a
        batch_size = y.shape[0]

        # Clone to avoid in-place modification
        y_prime = y.clone()

        # Convert flat position to (row, col)
        row = pos // 9
        col = pos % 9

        # Apply edits batch-wise
        for b in range(batch_size):
            y_prime[b, row[b], col[b]] = val[b]

        return y_prime

    def reward(
        self,
        y: torch.Tensor,
        y_prime: torch.Tensor,
        a: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute improvement reward for Sudoku.

        Reward is the improvement in score minus edit penalty:
            r = score(y') - score(y) - λ

        Args:
            y: Current grid (B, 9, 9)
            y_prime: Next grid after edit (B, 9, 9)
            a: Action (unused, kept for API consistency)

        Returns:
            Reward tensor (B,)
        """
        score_before = score_sudoku(y)
        score_after = score_sudoku(y_prime)
        improvement = score_after - score_before
        return improvement - self.edit_penalty

    def step(
        self,
        s: Tuple[torch.Tensor, torch.Tensor],
        a: Tuple[torch.Tensor, torch.Tensor],
        x: torch.Tensor,
        f_theta: Optional[Callable] = None,
        n: int = 1,
        rules: str = "sudoku",
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Take one step in the meta-MDP.

        Args:
            s: Current state (x, y) where:
                - x: Input (B, ...) - not modified
                - y: Current grid (B, 9, 9)
            a: Action (pos, val)
            x: Input tensor (B, ...) - same as s[0]
            f_theta: Optional model for recomputing z_n (if None, z_n not recomputed)
            n: Number of reasoning steps for recomputation
            rules: Task rules (default: "sudoku")

        Returns:
            Tuple of (s', reward, done, info) where:
                - s': Next state (x, y')
                - reward: Reward tensor (B,)
                - done: Terminal flags (B,)
                - info: Dict with metrics
        """
        x_cur, y = s

        # Apply edit to get y'
        y_prime = self.apply_edit(y, a, rules)

        # Compute reward
        r = self.reward(y, y_prime, a)

        # Next state is (x, y') - x doesn't change
        s_prime = (x_cur, y_prime)

        # Check if done (perfect score = 27 * 9 = 243 constraints)
        score = score_sudoku(y_prime)
        done = (score >= 243).float()

        # Info dict
        info = {
            "score": score,
            "score_improvement": score - score_sudoku(y),
            "done": done,
        }

        return s_prime, r, done, info


# Keep the original MetaMDP class for backward compatibility
class MetaMDP:
    """Meta-MDP for training models to edit their internal states.

    The meta-MDP treats the model's internal reasoning process as a sequential
    decision problem where the model learns to iteratively refine its internal
    representations to solve tasks.

    Attributes:
        state_dim: Dimension of internal state representations
        action_dim: Dimension of edit action space
        device: Device for tensor operations
    """

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        """Initialize the Meta-MDP environment.

        Args:
            state_dim: Dimension of internal state representations
            action_dim: Dimension of edit action space
            device: Device for tensor operations
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def reset(self, x: Any) -> Tuple[Any, Any]:
        """Reset environment with new input.

        Args:
            x: Input tensor

        Returns:
            Tuple of (input, initial_state)
        """
        pass

    def apply_edit(self, state: Any, edit: Any) -> Any:
        """Apply edit action to current state.

        Args:
            state: Current internal state (B, state_dim)
            edit: Edit action to apply (B, action_dim)

        Returns:
            Updated state after applying edit
        """
        pass

    def reward(self, state: Any, next_state: Any, done: bool) -> Any:
        """Compute reward for transition.

        Args:
            state: Current state
            next_state: Next state after edit
            done: Whether episode is complete

        Returns:
            Reward tensor (B,)
        """
        pass

    def step(self, state: Any, edit: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """Take one step in the meta-MDP.

        Args:
            state: Current state (B, state_dim)
            edit: Edit action (B, action_dim)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass

    def is_terminal(self, state: Any, step: int) -> Any:
        """Check if state is terminal.

        Args:
            state: Current state
            step: Current step number

        Returns:
            Boolean tensor indicating terminal states
        """
        pass


def create_meta_mdp(config: Dict[str, Any]) -> MetaMDP:
    """Factory function to create MetaMDP from configuration.

    Args:
        config: Configuration dictionary with MDP parameters

    Returns:
        Initialized MetaMDP instance
    """
    task = config.get("task", "sudoku")
    if task == "sudoku":
        return SudokuMetaMDP(
            edit_penalty=config.get("edit_penalty", 0.01),
            device=config.get("device", "cpu"),
        )
    else:
        # Fallback to generic MetaMDP
        return MetaMDP(
            state_dim=config.get("state_dim", 256),
            action_dim=config.get("action_dim", 128),
            device=config.get("device", "cpu"),
        )
