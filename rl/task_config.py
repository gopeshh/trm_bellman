"""
Task-specific configuration for UPI-TRM.

This module provides abstractions for task-specific behavior such as:
- Action masking (which cells can be edited)
- Checker functions (how to score solutions)
- Termination conditions (when is the puzzle solved)
- Reward scaling and shaping

This decouples task-specific logic from the core RL infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import torch

from rl.sudoku_checkers import dummy_checker, sudoku_solution_checker


class TaskConfig(ABC):
    """
    Abstract base class for task-specific configuration.
    
    Subclasses implement task-specific logic for:
    - Action masking: Which actions are valid for a given state
    - Checker: Scoring function for solutions
    - Termination: When is a puzzle considered solved
    - Cell identification: Which cells are "given" vs editable
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Task identifier (e.g., 'sudoku', 'arc', 'maze')."""
        pass
    
    @abstractmethod
    def is_given_cell(self, cell_value: int) -> bool:
        """
        Check if a cell value represents a "given" (non-editable) clue.
        
        Args:
            cell_value: Token value at a cell position
            
        Returns:
            True if this cell should not be edited
        """
        pass
    
    @abstractmethod
    def checker(self, x: Dict[str, Any], y: Any) -> float:
        """
        Score a candidate solution y for instance x.
        
        Args:
            x: Instance dict with "inputs", "puzzle_identifiers", optionally "solution"
            y: Candidate plan/solution tensor
            
        Returns:
            Score (higher is better). Typically 0-10 for Sudoku.
        """
        pass
    
    @abstractmethod
    def is_solved(self, score: float) -> bool:
        """
        Check if a score indicates the puzzle is solved.
        
        Args:
            score: Score from checker()
            
        Returns:
            True if puzzle is fully solved
        """
        pass
    
    def compute_action_mask(
        self,
        inputs: torch.Tensor,
        vocab_size: int,
        stop_action_id: int,
        current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute action mask for a single instance.
        
        Default implementation masks edit actions for "given" cells.
        
        Args:
            inputs: [seq_len] input tokens
            vocab_size: Number of tokens per position
            stop_action_id: Index of STOP action
            current_state: Optional current plan/state (unused by default)
            
        Returns:
            [num_actions] boolean mask (True = valid action)
        """
        flat_inputs = inputs.reshape(-1)
        num_positions = flat_inputs.numel()
        num_actions = stop_action_id + 1
        
        # Start with all actions valid
        mask = torch.ones(num_actions, dtype=torch.bool, device=inputs.device)
        
        # Mask out all tokens for "given" positions
        for pos in range(num_positions):
            cell_value = int(flat_inputs[pos].item())
            if self.is_given_cell(cell_value):
                # Block all edit actions for this position
                start_action = pos * vocab_size
                end_action = start_action + vocab_size
                if end_action <= num_actions:
                    mask[start_action:end_action] = False
        
        # STOP action is always valid
        if stop_action_id < num_actions:
            mask[stop_action_id] = True
        
        return mask
    
    def compute_batch_action_mask(
        self,
        inputs: torch.Tensor,
        vocab_size: int,
        stop_action_id: int,
        current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute action masks for a batch of instances.
        
        Args:
            inputs: [B, ...] batch of input tokens
            vocab_size: Number of tokens per position
            stop_action_id: Index of STOP action
            current_state: Optional current plan/state (unused by default)
            
        Returns:
            [B, num_actions] boolean mask
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        batch_size = inputs.shape[0]
        flat_inputs = inputs.reshape(batch_size, -1)
        num_positions = flat_inputs.shape[1]
        num_actions = stop_action_id + 1
        
        mask = torch.ones((batch_size, num_actions), dtype=torch.bool, device=inputs.device)
        
        # Identify given cells using vectorized operations
        given_mask = torch.zeros_like(flat_inputs, dtype=torch.bool)
        for pos in range(num_positions):
            for b in range(batch_size):
                given_mask[b, pos] = self.is_given_cell(int(flat_inputs[b, pos].item()))
        
        # Expand given_mask to cover all vocab_size actions per position
        edits_mask = ~given_mask.repeat_interleave(vocab_size, dim=1)
        
        # Apply to edit portion of action space
        num_edit_actions = edits_mask.shape[1]
        limit = min(num_actions, num_edit_actions)
        mask[:, :limit] = edits_mask[:, :limit]
        
        # Ensure STOP is always valid
        if stop_action_id < num_actions:
            mask[:, stop_action_id] = True

        return mask


class SudokuConstraintTracker:
    """
    Incremental constraint tracker for Sudoku puzzle environments.

    Maintains per-row/col/box digit COUNTS (not just presence flags) to correctly
    handle duplicate digits. When a digit count goes from 1->0, that digit becomes
    available again in that constraint group.

    This enables O(1) mask updates after each edit instead of O(P*V) recomputation.

    Usage:
        tracker = SudokuConstraintTracker(grid_size=9)
        tracker.initialize(current_state)

        # After an edit at position p, old_digit -> new_digit:
        tracker.update(position, old_digit, new_digit)

        # Get valid digits for a position:
        valid = tracker.get_valid_digits(position)

        # Or patch an existing mask:
        tracker.patch_mask(mask, affected_positions, vocab_size)
    """

    def __init__(self, grid_size: int, device: torch.device = None):
        """
        Initialize the constraint tracker.

        Args:
            grid_size: Size of the Sudoku grid (4 for 4x4, 9 for 9x9)
            device: Torch device for tensor storage
        """
        self.grid_size = grid_size
        self.box_size = int(grid_size ** 0.5)
        self.num_positions = grid_size * grid_size
        self.num_digits = grid_size  # Digits 1-4 for 4x4, 1-9 for 9x9
        self.device = device if device is not None else torch.device('cpu')

        # Digit counts: track how many times each digit appears in each constraint group
        # Using int16 to save memory while supporting counts > 1
        self.row_counts = torch.zeros(
            grid_size, self.num_digits, dtype=torch.int16, device=self.device
        )
        self.col_counts = torch.zeros(
            grid_size, self.num_digits, dtype=torch.int16, device=self.device
        )
        self.box_counts = torch.zeros(
            grid_size, self.num_digits, dtype=torch.int16, device=self.device
        )

        # Precompute position -> (row, col, box) mapping
        self._pos_to_row = torch.zeros(self.num_positions, dtype=torch.long, device=self.device)
        self._pos_to_col = torch.zeros(self.num_positions, dtype=torch.long, device=self.device)
        self._pos_to_box = torch.zeros(self.num_positions, dtype=torch.long, device=self.device)

        for pos in range(self.num_positions):
            row = pos // grid_size
            col = pos % grid_size
            box_row = row // self.box_size
            box_col = col // self.box_size
            box_idx = box_row * self.box_size + box_col

            self._pos_to_row[pos] = row
            self._pos_to_col[pos] = col
            self._pos_to_box[pos] = box_idx

        # Precompute which positions share constraints with each position
        # For patching the mask after an edit
        self._affected_positions: List[List[int]] = []
        for pos in range(self.num_positions):
            row = pos // grid_size
            col = pos % grid_size
            box_row = (row // self.box_size) * self.box_size
            box_col = (col // self.box_size) * self.box_size

            affected = set()
            # Same row
            for c in range(grid_size):
                affected.add(row * grid_size + c)
            # Same column
            for r in range(grid_size):
                affected.add(r * grid_size + col)
            # Same box
            for dr in range(self.box_size):
                for dc in range(self.box_size):
                    affected.add((box_row + dr) * grid_size + (box_col + dc))

            self._affected_positions.append(sorted(affected))

        # Cache for the full mask (will be set by initialize or compute_full_mask)
        self._cached_mask: Optional[torch.Tensor] = None
        self._mask_valid = False

        # Store given cells mask (from original inputs, immutable)
        self._given_mask: Optional[torch.Tensor] = None

    def initialize(
        self,
        current_state: torch.Tensor,
        given_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize constraint counts from a board state.

        Args:
            current_state: Flat tensor of board values [num_positions]
                          Values: 1=empty, 2-10=digits 1-9
            given_mask: Optional boolean mask [num_positions] where True = given cell
        """
        state = current_state.reshape(-1)
        if state.device != self.device:
            state = state.to(self.device)

        # Reset counts
        self.row_counts.zero_()
        self.col_counts.zero_()
        self.box_counts.zero_()

        # Count digits in each constraint group
        for pos in range(self.num_positions):
            val = state[pos].item()
            if val >= 2:  # Valid digit (tokens 2-10 = digits 1-9)
                digit_idx = val - 2  # Convert token to 0-indexed digit
                if digit_idx < self.num_digits:
                    row = self._pos_to_row[pos].item()
                    col = self._pos_to_col[pos].item()
                    box = self._pos_to_box[pos].item()

                    self.row_counts[row, digit_idx] += 1
                    self.col_counts[col, digit_idx] += 1
                    self.box_counts[box, digit_idx] += 1

        # Store given mask
        if given_mask is not None:
            self._given_mask = given_mask.to(self.device) if given_mask.device != self.device else given_mask
        else:
            self._given_mask = None

        # Invalidate cached mask
        self._mask_valid = False
        self._cached_mask = None

    def update(self, position: int, old_digit: int, new_digit: int) -> List[int]:
        """
        Update counts after an edit at the given position.

        Args:
            position: Flat position index (0 to num_positions-1)
            old_digit: Previous token value at this position (1=empty, 2-10=digits)
            new_digit: New token value at this position

        Returns:
            List of positions whose valid digits may have changed
        """
        if old_digit == new_digit:
            return []

        row = self._pos_to_row[position].item()
        col = self._pos_to_col[position].item()
        box = self._pos_to_box[position].item()

        # Decrement count for old digit
        if old_digit >= 2:
            digit_idx = old_digit - 2
            if digit_idx < self.num_digits:
                self.row_counts[row, digit_idx] -= 1
                self.col_counts[col, digit_idx] -= 1
                self.box_counts[box, digit_idx] -= 1

        # Increment count for new digit
        if new_digit >= 2:
            digit_idx = new_digit - 2
            if digit_idx < self.num_digits:
                self.row_counts[row, digit_idx] += 1
                self.col_counts[col, digit_idx] += 1
                self.box_counts[box, digit_idx] += 1

        # Invalidate mask and return affected positions
        self._mask_valid = False
        return self._affected_positions[position]

    def get_valid_digits(self, position: int) -> torch.Tensor:
        """
        Get valid digits for a position based on current constraints.

        Args:
            position: Flat position index

        Returns:
            Boolean tensor [num_digits] where True = digit is valid
        """
        row = self._pos_to_row[position].item()
        col = self._pos_to_col[position].item()
        box = self._pos_to_box[position].item()

        # Digit is valid if count == 0 in row AND col AND box
        row_available = self.row_counts[row] == 0
        col_available = self.col_counts[col] == 0
        box_available = self.box_counts[box] == 0

        return row_available & col_available & box_available

    def compute_full_mask(
        self,
        vocab_size: int,
        stop_action_id: int,
    ) -> torch.Tensor:
        """
        Compute the full action mask from current constraint state.

        This is used when the cached mask is invalid or for initial computation.

        Args:
            vocab_size: Number of tokens (6 for 4x4, 11 for 9x9)
            stop_action_id: Index of STOP action

        Returns:
            Boolean mask [num_actions] where True = valid action
        """
        num_actions = stop_action_id + 1
        mask = torch.ones(num_actions, dtype=torch.bool, device=self.device)

        for pos in range(self.num_positions):
            start = pos * vocab_size
            end = start + vocab_size

            if end > num_actions:
                break

            # Check if this is a given cell
            if self._given_mask is not None and self._given_mask[pos]:
                mask[start:end] = False
                continue

            # Mask PAD (token 0) and empty (token 1)
            mask[start:start+2] = False

            # Mask invalid digits based on constraints
            valid_digits = self.get_valid_digits(pos)
            # valid_digits is [num_digits], maps to tokens [2, 2+num_digits)
            for d in range(self.num_digits):
                if d + 2 < vocab_size:
                    mask[start + d + 2] = valid_digits[d]

        # STOP is always valid
        mask[stop_action_id] = True

        self._cached_mask = mask
        self._mask_valid = True

        return mask

    def get_mask(
        self,
        vocab_size: int,
        stop_action_id: int,
    ) -> torch.Tensor:
        """
        Get action mask, using cache if valid.

        Args:
            vocab_size: Number of tokens
            stop_action_id: Index of STOP action

        Returns:
            Boolean mask [num_actions]
        """
        if self._mask_valid and self._cached_mask is not None:
            return self._cached_mask

        return self.compute_full_mask(vocab_size, stop_action_id)

    def patch_mask(
        self,
        affected_positions: List[int],
        vocab_size: int,
    ) -> None:
        """
        Patch the cached mask for affected positions only.

        This is O(affected_positions) instead of O(num_positions).
        For a single edit, affected_positions is typically ~20 for 9x9.

        Args:
            affected_positions: List of position indices to update
            vocab_size: Number of tokens
        """
        if self._cached_mask is None:
            return

        for pos in affected_positions:
            start = pos * vocab_size
            end = start + vocab_size

            if end > len(self._cached_mask):
                continue

            # Check if this is a given cell (skip patching)
            if self._given_mask is not None and self._given_mask[pos]:
                # Given cells stay masked - no change needed
                continue

            # Keep PAD and empty masked
            self._cached_mask[start:start+2] = False

            # Update digit validity
            valid_digits = self.get_valid_digits(pos)
            for d in range(self.num_digits):
                if d + 2 < vocab_size:
                    self._cached_mask[start + d + 2] = valid_digits[d]

        self._mask_valid = True

    def update_and_patch(
        self,
        position: int,
        old_digit: int,
        new_digit: int,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Combined update and patch operation for efficiency.

        Args:
            position: Position of the edit
            old_digit: Old token value
            new_digit: New token value
            vocab_size: Number of tokens

        Returns:
            The updated cached mask
        """
        affected = self.update(position, old_digit, new_digit)

        if self._cached_mask is not None and affected:
            self.patch_mask(affected, vocab_size)

        return self._cached_mask


@dataclass
class SudokuTaskConfig(TaskConfig):
    """
    Task configuration for Sudoku puzzles.
    
    Sudoku encoding:
    - Token 1 = empty cell (editable)
    - Tokens 2-10 = digits 1-9 (given clues if in original input)
    
    Attributes:
        solved_score: Score that indicates a solved puzzle (default: 10.0)
        scale_factor: Multiplier for checker scores (default: 10.0)
    """
    
    solved_score: float = 10.0
    scale_factor: float = 10.0
    
    @property
    def name(self) -> str:
        return "sudoku"
    
    def is_given_cell(self, cell_value: int) -> bool:
        """
        In Sudoku encoding:
        - 1 = empty cell (editable)
        - > 1 = given clue (not editable)
        """
        return cell_value > 1
    
    def checker(self, x: Dict[str, Any], y: Any) -> float:
        """
        Score based on fraction of cells matching the solution.
        Returns score in [0, scale_factor] range.
        """
        return sudoku_solution_checker(x, y, scale_factor=self.scale_factor)
    
    def _fallback_checker(self, x: Dict[str, Any], y: Any) -> float:
        """Fallback checker when no solution is available."""
        return dummy_checker(x, y)
    
    def _to_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value)
    
    def is_solved(self, score: float) -> bool:
        """Puzzle is solved when score reaches solved_score."""
        return score >= self.solved_score - 1e-6

    def compute_action_mask(
        self,
        inputs: torch.Tensor,
        vocab_size: int,
        stop_action_id: int,
        current_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute action mask for Sudoku with constraint-aware masking.

        Delegates to compute_batch_action_mask with batch_size=1 for efficiency.
        The batch implementation is fully vectorized and GPU-friendly.

        Masks out:
        1. All tokens for "given" cells (clues from original inputs)
        2. Token 0 (PAD) and Token 1 (empty) for all positions
        3. Digits that violate Sudoku constraints (same digit in row/col/box)

        Args:
            inputs: Original puzzle state (to identify given cells) - 1D or 2D
            vocab_size: Number of tokens (11 for 9x9: 0=PAD, 1=empty, 2-10=digits)
            stop_action_id: Index of STOP action
            current_state: Current board state for constraint checking (optional)
        """
        # Handle input dimensionality
        is_1d = inputs.dim() == 1
        batch_inputs = inputs.unsqueeze(0) if is_1d else inputs

        # Handle current_state dimensionality
        if current_state is not None:
            batch_state = current_state.unsqueeze(0) if current_state.dim() == 1 else current_state
        else:
            batch_state = None

        # Delegate to batch implementation
        batch_mask = self.compute_batch_action_mask(
            batch_inputs,
            vocab_size,
            stop_action_id,
            batch_state,
        )

        # Return single-instance mask if input was 1D, otherwise return batch mask
        if is_1d:
            return batch_mask.squeeze(0)
        return batch_mask

    # Class-level cache for constraint indices (shared across instances)
    _constraint_cache: ClassVar[Dict] = {}

    def _get_constraint_indices(
        self, grid_size: int, box_size: int, device: torch.device
    ) -> tuple:
        """
        Get or build constraint index tensors (cached per grid size).

        Returns:
            constraint_tensor: [P, C] - indices of constraint cells for each position
            constraint_valid: [P, C] - validity mask for constraint indices
        """
        cache_key = (grid_size, box_size, str(device))
        if cache_key in SudokuTaskConfig._constraint_cache:
            cached = SudokuTaskConfig._constraint_cache[cache_key]
            # Move to correct device if needed
            if cached[0].device != device:
                return cached[0].to(device), cached[1].to(device)
            return cached

        num_positions = grid_size * grid_size

        # Build constraint indices using vectorized operations where possible
        # For each position, we need indices of all cells in same row, col, and box
        positions = torch.arange(num_positions, device=device)
        rows = positions // grid_size
        cols = positions % grid_size

        # Row constraints: for position p in row r, all positions r*G + [0..G-1]
        # Col constraints: for position p in col c, all positions [0..G-1]*G + c
        # Box constraints: for position p in box (br, bc), positions in that box

        # Pre-allocate constraint tensor
        # Max constraints per cell: G (row) + G (col) + box_size^2 (box) = 2G + B^2
        # But there are overlaps, actual unique is about G + G + B^2 - 2 for corner
        # Use safe upper bound
        max_constraints = grid_size + grid_size + box_size * box_size

        constraint_tensor = torch.zeros(
            num_positions, max_constraints, dtype=torch.long, device=device
        )
        constraint_valid = torch.zeros(
            num_positions, max_constraints, dtype=torch.bool, device=device
        )

        # Build constraints vectorized per constraint type, then combine
        # This is one-time cost, so acceptable to use some Python for clarity
        for pos in range(num_positions):
            r, c = pos // grid_size, pos % grid_size
            br = (r // box_size) * box_size
            bc = (c // box_size) * box_size

            cells = set()
            # Row cells
            for i in range(grid_size):
                cells.add(r * grid_size + i)
            # Col cells
            for i in range(grid_size):
                cells.add(i * grid_size + c)
            # Box cells
            for dr in range(box_size):
                for dc in range(box_size):
                    cells.add((br + dr) * grid_size + (bc + dc))

            cells_list = sorted(cells)
            constraint_tensor[pos, : len(cells_list)] = torch.tensor(
                cells_list, dtype=torch.long, device=device
            )
            constraint_valid[pos, : len(cells_list)] = True

        SudokuTaskConfig._constraint_cache[cache_key] = (constraint_tensor, constraint_valid)
        return constraint_tensor, constraint_valid

    def compute_batch_action_mask(
        self,
        inputs: torch.Tensor,
        vocab_size: int,
        stop_action_id: int,
        current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute action masks for a batch, honoring Sudoku constraints.

        Fully vectorized implementation with no Python loops in the hot path.
        Uses [B, P, V] tensor space and flattens to [B, num_actions] at the end.

        Constraint indices are cached per grid size for efficiency.

        Args:
            inputs: [B, ...] original puzzle tokens (identifies given cells)
            vocab_size: Number of tokens per position (6 for 4x4, 11 for 9x9)
            stop_action_id: Index of STOP action
            current_state: Optional [B, ...] current board state for constraint checking

        Returns:
            [B, num_actions] boolean mask where True = action allowed
        """
        # Ensure batch dimension
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        if current_state is not None and current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)

        flat_inputs = inputs.reshape(batch_size, -1)
        num_positions = flat_inputs.shape[1]
        num_actions = stop_action_id + 1
        device = inputs.device

        # Determine grid size (4 for 4x4, 9 for 9x9)
        grid_size = int(num_positions ** 0.5)
        box_size = int(grid_size ** 0.5)

        # Use current state for constraint checking, fall back to inputs
        state = (
            current_state.reshape(batch_size, -1)
            if current_state is not None
            else flat_inputs
        )

        # === Build [B, P, V] mask where True = action allowed ===
        pos_mask = torch.ones(
            batch_size, num_positions, vocab_size, dtype=torch.bool, device=device
        )

        # --- 1. Mask PAD (token 0) and empty (token 1) for all positions ---
        # These tokens should never be placed
        if vocab_size >= 2:
            pos_mask[:, :, :2] = False

        # --- 2. Mask constraint violations (digits already in row/col/box) ---
        # Get cached constraint indices
        constraint_tensor, constraint_valid = self._get_constraint_indices(
            grid_size, box_size, device
        )
        # constraint_tensor: [P, C], constraint_valid: [P, C]

        # Gather values at constraint positions: [B, P, C]
        constraint_values = state[:, constraint_tensor]

        # Check which digits (2 to vocab_size-1) appear in constraints
        # digits: [D] where D = vocab_size - 2
        num_digits = vocab_size - 2
        if num_digits > 0:
            digits = torch.arange(2, vocab_size, device=device)  # [D]

            # Compare constraint values with each digit: [B, P, C, 1] vs [1, 1, 1, D]
            # Result: [B, P, C, D]
            digit_match = constraint_values.unsqueeze(3) == digits.view(1, 1, 1, -1)

            # Mask out invalid constraint positions (padding in constraint_tensor)
            digit_match = digit_match & constraint_valid.view(1, num_positions, -1, 1)

            # digit_present[b, p, d] = True if digit d+2 appears in constraints for position p
            digit_present = digit_match.any(dim=2)  # [B, P, D]

            # Mask these digits: pos_mask[:, :, 2:] should be False where digit_present is True
            pos_mask[:, :, 2:vocab_size] = pos_mask[:, :, 2:vocab_size] & ~digit_present

        # --- 3. Mask all tokens for given cells ---
        # given_mask: [B, P] - True where cell has a given value (> 1)
        given_mask = flat_inputs > 1  # [B, P]

        # Expand to [B, P, V] and apply: mask all tokens for given positions
        pos_mask = pos_mask & ~given_mask.unsqueeze(2)

        # === Flatten [B, P, V] to [B, num_actions] ===
        # Action index = pos * vocab_size + token
        # Total edit actions = P * V, but we only use first (num_actions - 1)
        pos_mask_flat = pos_mask.reshape(batch_size, -1)  # [B, P * V]

        # Create output mask with STOP action
        mask = torch.ones(batch_size, num_actions, dtype=torch.bool, device=device)

        # Copy edit action mask (truncate if P*V > num_actions-1)
        num_edit_actions = num_positions * vocab_size
        actual_edit_actions = min(num_edit_actions, num_actions - 1)
        mask[:, :actual_edit_actions] = pos_mask_flat[:, :actual_edit_actions]

        # STOP action is always valid
        mask[:, stop_action_id] = True

        return mask



@dataclass
class DummyTaskConfig(TaskConfig):
    """
    Task configuration for dummy/synthetic puzzles (smoke testing).
    
    All cells are editable and the "solution" is the input itself.
    """
    
    @property
    def name(self) -> str:
        return "dummy"
    
    def is_given_cell(self, cell_value: int) -> bool:
        """In dummy task, all cells are editable."""
        return False
    
    def checker(self, x: Dict[str, Any], y: Any) -> float:
        """
        Score based on negative L1 distance between plan and inputs.
        """
        return dummy_checker(x, y)
    
    def is_solved(self, score: float) -> bool:
        """Dummy task is "solved" when plan exactly matches inputs (score = 0)."""
        return abs(score) < 1e-6


@dataclass  
class ARCTaskConfig(TaskConfig):
    """
    Task configuration for ARC-AGI puzzles.
    
    ARC encoding varies by representation, but generally:
    - Background color (0) may or may not be editable
    - All other colors are part of the solution
    """
    
    background_value: int = 0
    background_editable: bool = True
    
    @property
    def name(self) -> str:
        return "arc"
    
    def is_given_cell(self, cell_value: int) -> bool:
        """
        In ARC, this depends on the specific representation.
        Default: background is editable, nothing else is "given".
        """
        if self.background_editable:
            return False
        return cell_value == self.background_value
    
    def checker(self, x: Dict[str, Any], y: Any) -> float:
        """
        Score based on exact pixel match with solution.
        """
        solution = x.get("solution")
        if solution is None:
            return 0.0
        
        if torch.is_tensor(y):
            plan = y
        else:
            plan = torch.as_tensor(y)
        
        if torch.is_tensor(solution):
            sol = solution
        else:
            sol = torch.as_tensor(solution)
        
        if plan.shape != sol.shape:
            sol = sol.view_as(plan)
        
        matches = (plan == sol).float()
        return float(matches.mean().item() * 100.0)  # 0-100 scale
    
    def is_solved(self, score: float) -> bool:
        """ARC is solved when all pixels match (score = 100)."""
        return score >= 100.0 - 1e-6


def get_task_config(task_name: str, **kwargs) -> TaskConfig:
    """
    Factory function to get task configuration by name.
    
    Args:
        task_name: One of "sudoku", "arc", "dummy"
        **kwargs: Task-specific configuration options
        
    Returns:
        TaskConfig instance
        
    Raises:
        ValueError: If task_name is not recognized
    """
    task_name = task_name.lower()
    
    if task_name == "sudoku":
        return SudokuTaskConfig(**kwargs)
    elif task_name == "arc":
        return ARCTaskConfig(**kwargs)
    elif task_name == "dummy":
        return DummyTaskConfig(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: sudoku, arc, dummy")
