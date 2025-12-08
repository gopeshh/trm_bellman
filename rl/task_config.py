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
from typing import Any, Callable, Dict, Optional

import torch


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
    ) -> torch.Tensor:
        """
        Compute action mask for a single instance.
        
        Default implementation masks edit actions for "given" cells.
        
        Args:
            inputs: [seq_len] input tokens
            vocab_size: Number of tokens per position
            stop_action_id: Index of STOP action
            
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
    ) -> torch.Tensor:
        """
        Compute action masks for a batch of instances.
        
        Args:
            inputs: [B, ...] batch of input tokens
            vocab_size: Number of tokens per position
            stop_action_id: Index of STOP action
            
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
        solution = x.get("solution")
        if solution is None:
            # Fallback: use negative L1 distance from inputs
            return self._fallback_checker(x, y)
        
        plan = self._to_tensor(y).to(torch.long)
        solution_tensor = self._to_tensor(solution).to(torch.long)
        
        if plan.shape != solution_tensor.shape:
            solution_tensor = solution_tensor.view_as(plan)
        
        matches = (plan == solution_tensor).to(torch.float32)
        return float(matches.mean().item() * self.scale_factor)
    
    def _fallback_checker(self, x: Dict[str, Any], y: Any) -> float:
        """Fallback checker when no solution is available."""
        target = x["inputs"]
        plan = self._to_tensor(y).to(torch.float32)
        target = target.to(torch.float32)
        return float(-(target - plan).abs().mean().item())
    
    def _to_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value)
    
    def is_solved(self, score: float) -> bool:
        """Puzzle is solved when score reaches solved_score."""
        return score >= self.solved_score - 1e-6


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
        target = x["inputs"]
        if torch.is_tensor(y):
            plan = y
        else:
            plan = torch.as_tensor(y)
        target = target.to(torch.float32)
        plan = plan.to(torch.float32)
        return float(-(target - plan).abs().mean().item())
    
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

