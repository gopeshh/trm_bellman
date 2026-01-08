from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

import torch

from rl.sudoku_utils import sudoku_is_solved

if TYPE_CHECKING:
    from rl.task_config import TaskConfig


@dataclass
class PlanEditEnvConfig:
    """
    Configuration for the plan-space edit environment.

    Attributes:
        max_edits: Maximum number of edit steps per episode
        gamma: Discount factor (used for potential-based reward shaping)
        reward_shaping: If True, use potential-based reward shaping
        task_type: Task identifier (e.g., "sudoku", "arc", "maze")
        vocab_size: Number of discrete tokens per cell
        solved_threshold: Checker score that triggers episode termination
        stop_action_mode: How STOP action behaves ("terminal", "noop", "disabled")
        stop_action_penalty: Penalty for choosing STOP in "noop" mode

        UNDO ACTION (IMPLEMENTATION_GUIDELINE Phase 5.4):
        enable_undo: If True, add UNDO action that reverts to previous plan state.
            Expands action space: num_actions = seq_len * vocab_size + 2 (STOP + UNDO).
            UNDO is valid when history > 1 (at least one edit has been made).
            UNDO pops the last plan from history and returns the previous state.

        RUSH-TO-FAIL MITIGATION (Paper Remark 2.6):
        fail_terminal_reward: Terminal reward for failing (not solving) the puzzle.
            Set to a sufficiently negative value (e.g., -C_max) to prevent the
            agent from "rushing to fail" under potential-based shaping.

            Theory: With Φ(s_abs) = C_max and shaping r = γΦ(s') - Φ(s),
            the condition r_term_fail ≤ -γC_max ensures that failing from
            any state yields non-positive total reward.

            Default is 0.0 (no extra penalty), which may allow rush-to-fail
            in some edge cases. Set to -C_max for theory alignment.

        solve_terminal_reward: Terminal reward bonus for solving the puzzle.
            Default is 0.0 (rely on shaping). Can be positive for extra incentive.
    """
    max_edits: int
    gamma: float
    reward_shaping: bool = True
    task_type: Optional[str] = None
    vocab_size: Optional[int] = None
    solved_threshold: Optional[float] = None
    # STOP action behavior (inherited from RLConfig)
    stop_action_mode: str = "noop"  # "terminal", "noop", "disabled"
    stop_action_penalty: float = -0.1
    # UNDO action support (Phase 5.4)
    enable_undo: bool = False
    # Terminal rewards (Paper Remark 2.6: rush-to-fail mitigation)
    fail_terminal_reward: float = 0.0   # Set to -C_max for theory alignment
    solve_terminal_reward: float = 0.0  # Optional bonus for solving


class PlanEditEnv:
    """
    Simple plan-space meta–MDP:
      - State: (x, y), where x is an instance and y is a plan / candidate solution.
      - Actions: edit actions over y plus special STOP action (and optional UNDO).
      - Reward: derived from a checker c(x, y) with potential-based shaping.

    UNDO ACTION (Phase 5.4):
      When config.enable_undo=True:
      - Action space expands: num_actions = seq_len * vocab_size + 2 (STOP + UNDO)
      - UNDO reverts to the previous plan state (pops from edit history)
      - UNDO is masked out when history has only 1 entry (nothing to undo)
      - This allows the agent to explore without permanently committing to edits

    STOP Action Modes:
      - "terminal": STOP ends the episode immediately (standard RL)
      - "noop": STOP is treated as no-op, episode continues (prevents STOP collapse)
      - "disabled": STOP action is masked out and cannot be selected

    STOP + REWARD SHAPING INTERACTION (Issue 7):
        In stop_action_mode="noop" with reward_shaping=True, STOP yields:
            r = stop_action_penalty + (γ * Φ(y) - Φ(y))
              = stop_action_penalty + (γ - 1) * Φ(y)

        Since γ < 1, the term (γ - 1) * Φ(y) is NEGATIVE when Φ(y) > 0.
        Near high-scoring states (large Φ), STOP is strongly penalized.
        This is intentional: it discourages stopping when progress is possible.

    DEVIATION FROM PAPER (Issue 6 - Absorbing State):
        The ICML paper defines an explicit absorbing state s_abs with potential
        Φ(s_abs) = C_max. Terminal transitions get:
            r = r_term(x, y) + γ * C_max - c(x, y)

        This implementation does NOT model an explicit s_abs node. Instead:
        - Episodes halt directly at the final plan (x, y_final)
        - phi_new = checker(x, y_final) as usual (no separate C_max)
        - Terminal rewards are controlled by fail_terminal_reward and
          solve_terminal_reward in PlanEditEnvConfig

        This is behaviorally equivalent for most experiments, but readers
        comparing code to paper notation should note the difference.
    """

    def __init__(
        self,
        dataset: Any,
        checker: Callable[[Any, Any], float],
        config: PlanEditEnvConfig,
        task_config: Optional["TaskConfig"] = None,
    ):
        self.dataset = dataset
        self.checker = checker
        self.config = config
        self.task_config = task_config  # Optional task-specific config

        self.vocab_size: Optional[int] = config.vocab_size
        if self.vocab_size is None:
            self.vocab_size = self._infer_vocab_size()

        self.step_count: int = 0
        self.x: Any = None
        self.y: Any = None
        self.done: bool = False

        # Store original inputs to identify "given" cells that shouldn't be edited
        self._original_inputs: Optional[torch.Tensor] = None
        self._action_mask: Optional[torch.Tensor] = None
        self._stop_penalty: float = 0.0  # Penalty for early stopping

        # Action space: caller is responsible for interpreting action indices.
        # We require STOP to be the last action index by convention.
        self.stop_action_id: Optional[int] = None

        # UNDO action support (Phase 5.4)
        self._enable_undo = getattr(config, "enable_undo", False)
        self.undo_action_id: Optional[int] = None
        self._edit_history: List[torch.Tensor] = []  # Stack of plan states for UNDO

        # STOP action behavior
        self._stop_mode = getattr(config, "stop_action_mode", "noop")
        self._stop_penalty_value = getattr(config, "stop_action_penalty", -0.1)

    def set_stop_action_id(self, stop_id: int) -> None:
        """
        Configure which discrete action index corresponds to STOP.

        If UNDO is enabled, UNDO action is assumed to be at stop_id + 1.
        """
        self.stop_action_id = stop_id

        if self._enable_undo:
            self.undo_action_id = stop_id + 1

    def set_undo_action_id(self, undo_id: int) -> None:
        """
        Explicitly configure which action index corresponds to UNDO.

        Only needed if using non-standard action layout.
        """
        self.undo_action_id = undo_id

    def reset(self, idx: Optional[int] = None) -> Tuple[Any, Any]:
        """
        Sample an instance x and an initial plan y_0 from the dataset.
        - If idx is provided, use that index; otherwise, sample uniformly.
        - y_0 can be dataset-provided or a trivial initialization.
        """

        assert hasattr(self.dataset, "__len__"), "Dataset must implement __len__"
        if idx is None:
            idx = torch.randint(low=0, high=len(self.dataset), size=(1,)).item()
        sample = self.dataset[idx]

        # Expect the dataset sample to provide x and an initial plan y_0.
        # Typical format: {"inputs": ..., "puzzle_identifiers": ..., "initial_plan": ...}
        # For now, use "inputs" as x and "initial_plan" if present, else a zero plan.
        if isinstance(sample, dict):
            self.x = self._standardize_state(sample)
            if "initial_plan" in sample:
                self.y = self._standardize_plan(sample["initial_plan"])
            else:
                # Default: initialize plan from inputs (copy clues).
                # If we used ones_like(inputs), we'd start with a blank grid (all empty),
                # forcing the agent to memorize/copy clues from x to y.
                # Copying inputs ensures we start with the clues pre-filled.
                inputs = self.x["inputs"]
                self.y = inputs.clone()
        else:
            # Fallback: treat sample as x and create a trivial plan with empty cells
            self.x = sample
            self.y = torch.ones_like(sample)

        self.step_count = 0
        self.done = False

        # Initialize edit history for UNDO support
        if self._enable_undo:
            # Store initial plan as first entry (cannot undo past this)
            self._edit_history = [self.y.clone() if torch.is_tensor(self.y) else torch.as_tensor(self.y)]
        else:
            self._edit_history = []

        # Compute action mask to protect "given" cells (non-zero in original inputs)
        self._compute_action_mask()

        return self.x, self.y
    
    def _compute_action_mask(self) -> None:
        """
        Compute action mask to prevent editing "given" cells and invalid tokens.
        
        If a TaskConfig is provided, it will be used to determine which cells
        are "given" (non-editable). Otherwise, falls back to Sudoku-style logic
        where cells with value > 1 are considered given.
        
        Additionally, tokens 0 (PAD) and 1 (empty) are always masked out because
        setting a cell to empty is never useful for solving Sudoku.
        
        Action space: [pos * vocab_size + tok for all pos, tok] + [STOP]
        Mask is True for valid actions, False for invalid.
        """
        if self.vocab_size is None or self.stop_action_id is None:
            self._action_mask = None
            self._original_inputs = None
            return
            
        # Get original inputs
        if isinstance(self.x, dict):
            inputs = self.x.get("inputs")
        else:
            inputs = self.x
            
        if inputs is None:
            self._action_mask = None
            self._original_inputs = None
            return
            
        if not torch.is_tensor(inputs):
            inputs = torch.as_tensor(inputs)
        
        self._original_inputs = inputs.clone()
        
        # Use TaskConfig if available, otherwise fall back to default behavior
        if self.task_config is not None:
            mask = self.task_config.compute_action_mask(
                inputs, self.vocab_size, self.stop_action_id
            )
        else:
            # Default Sudoku-style logic
            flat_inputs = inputs.reshape(-1)
            num_positions = flat_inputs.numel()
            num_actions = self.stop_action_id + 1
            
            # Create mask on the same device as inputs to avoid device mismatch
            mask = torch.ones(num_actions, dtype=torch.bool, device=inputs.device)
            
            # Mask out all tokens for "given" positions
            # In Sudoku encoding: 1 = empty cell, values > 1 = given clues
            for pos in range(num_positions):
                cell_value = flat_inputs[pos].item()
                if cell_value > 1:  # This is a given cell (not empty)
                    start_action = pos * self.vocab_size
                    end_action = start_action + self.vocab_size
                    mask[start_action:end_action] = False
            
            # === Mask out invalid token values (token 0 = PAD, token 1 = empty) ===
            # For Sudoku, only tokens 2-5 (digits 1-4 for 4x4, 2-10 for 9x9) are valid.
            # Setting a cell to PAD or empty is never useful for solving.
            for tok in range(min(2, self.vocab_size)):  # tok=0 (PAD), tok=1 (empty)
                for pos in range(num_positions):
                    action_idx = pos * self.vocab_size + tok
                    if action_idx < num_actions - 1:  # Don't touch STOP action
                        mask[action_idx] = False
            
            # STOP action is always valid (will be handled below)
            mask[self.stop_action_id] = True
        
        # Handle STOP action based on mode
        if self._stop_mode == "disabled":
            mask[self.stop_action_id] = False
        else:
            mask[self.stop_action_id] = True

        # Handle UNDO action if enabled
        if self._enable_undo and self.undo_action_id is not None:
            # Expand mask to include UNDO action if needed
            if len(mask) <= self.undo_action_id:
                # Expand mask to include UNDO
                new_mask = torch.ones(self.undo_action_id + 1, dtype=torch.bool, device=mask.device)
                new_mask[:len(mask)] = mask
                mask = new_mask

            # UNDO is valid only when there's something to undo (history > 1)
            # At reset, history has exactly 1 entry (initial state), so UNDO is invalid
            can_undo = len(self._edit_history) > 1
            mask[self.undo_action_id] = can_undo

        self._action_mask = mask
    
    def get_action_mask(self) -> Optional[torch.Tensor]:
        """
        Return the current action mask. True = valid action, False = invalid.
        Returns None if mask hasn't been computed (e.g., vocab_size not set).
        """
        return self._action_mask

    @staticmethod
    def compute_batch_action_mask(
        inputs: torch.Tensor,
        vocab_size: int,
        stop_action_id: int,
        stop_mode: str = "noop",
        exclude_empty_token: bool = True,
    ) -> torch.Tensor:
        """
        Compute action masks for a batch of inputs preventing edits to 'given' cells.
        
        Args:
            inputs: [B, ...] tensor of tokens.
            vocab_size: number of tokens per position.
            stop_action_id: index of the STOP action.
            stop_mode: One of "terminal", "noop", or "disabled". When "disabled",
                STOP action is masked out and not available to the policy.
            exclude_empty_token: If True (default), mask out setting cells to token 0 (PAD)
                and token 1 (empty). For Sudoku, only digits (tokens 2-5) are valid.
                
        Returns:
            [B, num_actions] boolean mask (True=valid).
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        batch_size = inputs.shape[0]
        # Flatten inputs to [B, P]
        flat_inputs = inputs.reshape(batch_size, -1)
        num_positions = flat_inputs.shape[1]
        
        # Assume action space size is at least stop_action_id + 1
        # and edit actions cover [0, num_positions * vocab_size)
        # Note: In upi_trm_train.py, rl_num_actions = seq_len * vocab_size + 1
        # and stop_id = rl_num_actions - 1.
        
        num_actions = stop_action_id + 1
        mask = torch.ones((batch_size, num_actions), dtype=torch.bool, device=inputs.device)
        
        # Identify given cells: value > 1 (Sudoku specific, but consistent with class logic)
        given_mask = (flat_inputs > 1)  # [B, P]
        
        # Edit actions are contiguous blocks of size vocab_size for each position
        # We expand given_mask to [B, P * V]
        # repeat_interleave is efficient for this
        edits_mask = ~given_mask.repeat_interleave(vocab_size, dim=1)
        
        # Determine how many edit actions fit in num_actions
        # Usually num_edit_actions = P * V
        num_edit_actions = edits_mask.shape[1]
        
        # If stop_action_id is the last one, num_actions = num_edit_actions + 1
        # We just fill the edit part.
        limit = min(num_actions, num_edit_actions)
        mask[:, :limit] = edits_mask[:, :limit]
        
        # === Mask out invalid token values (token 0 = PAD, token 1 = empty) ===
        # For Sudoku, only tokens 2-5 (digits 1-4) are valid cell values.
        # Setting a cell to PAD or empty is never useful for solving.
        if exclude_empty_token and vocab_size > 2:
            # For each position, mask out token 0 and token 1
            # Action index for (pos, tok) = pos * vocab_size + tok
            for tok in range(min(2, vocab_size)):  # tok=0 (PAD), tok=1 (empty)
                for pos in range(num_positions):
                    action_idx = pos * vocab_size + tok
                    if action_idx < limit:
                        mask[:, action_idx] = False
        
        # Handle STOP action based on mode
        if stop_action_id < num_actions:
            if stop_mode == "disabled":
                # STOP completely disabled: keep masked out
                mask[:, stop_action_id] = False
            else:
                # "terminal" or "noop": STOP is a valid action
                mask[:, stop_action_id] = True
            
        return mask

    def _standardize_state(self, sample: Any) -> Any:
        """
        Ensure environment state tensors have at least 1 dimension to satisfy batch heuristics.
        """
        if not isinstance(sample, dict):
            return sample
        state = dict(sample)
        for key in ("inputs", "puzzle_identifiers"):
            tensor = state.get(key)
            if tensor is None:
                continue
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            state[key] = tensor
        return state

    def _standardize_plan(self, plan: Any) -> torch.Tensor:
        if torch.is_tensor(plan):
            tensor = plan
        else:
            tensor = torch.as_tensor(plan)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _infer_vocab_size(self) -> Optional[int]:
        if not hasattr(self.dataset, "__len__") or len(self.dataset) == 0:
            return None
        sample = self.dataset[0]
        tokens = sample
        if isinstance(sample, dict):
            tokens = sample.get("inputs", sample)
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        if tokens.numel() == 0:
            return None
        max_token = int(torch.max(tokens).item())
        return max_token + 1

    def compute_transition_reward(
        self,
        phi_old: float,
        phi_new: float,
        is_stop_action: bool,
        is_terminal: bool,
        is_solved: bool,
    ) -> float:
        """
        Compute the reward for a transition given potential values.
        
        This is the canonical reward computation shared by:
        - step() for actual environment transitions
        - compute_exact_baseline_summation() for theory-exact Q estimation
        
        Implements Paper Eq. 4: r = r_0 + γ·Φ(s') - Φ(s) when reward_shaping=True.
        
        RUSH-TO-FAIL MITIGATION (Paper Remark 2.6):
            Terminal transitions receive r_0 based on outcome:
            - Solved: r_0 = solve_terminal_reward (default 0)
            - Failed: r_0 = fail_terminal_reward (set to -C_max for theory)
            
            The condition r_term_fail ≤ -γC_max guarantees that failing from
            any state yields non-positive total reward, preventing "rush to fail".
        
        Args:
            phi_old: Checker score Φ(s) = c(x, y_old)
            phi_new: Checker score Φ(s') = c(x, y_new)
            is_stop_action: Whether the action was STOP
            is_terminal: Whether this transition ends the episode
            is_solved: Whether the puzzle was solved (phi_new >= solved_threshold)
            
        Returns:
            Reward value matching the environment's reward semantics.
            
        Notes:
            - In stop_action_mode="noop" with reward_shaping=True, STOP yields:
              r = stop_action_penalty + (gamma * phi(y) - phi(y))
                = stop_action_penalty + (gamma - 1) * phi(y)
              so it is strongly penalized near high-scoring states.
        """
        gamma = self.config.gamma
        
        # Compute STOP penalty (only applies in noop/disabled modes)
        stop_penalty = 0.0
        if is_stop_action and self._stop_mode != "terminal":
            stop_penalty = self._stop_penalty_value
        
        # Compute terminal reward r_0 (Paper Remark 2.6)
        r_0 = 0.0
        if is_terminal:
            if is_solved:
                r_0 = getattr(self.config, "solve_terminal_reward", 0.0)
            else:
                # Failed termination (budget exhausted, STOP with terminal mode, etc.)
                # Set to negative value to prevent "rush to fail"
                r_0 = getattr(self.config, "fail_terminal_reward", 0.0)
        
        if self.config.reward_shaping:
            # Potential-based shaping: r = r_0 + γ·Φ(s') - Φ(s)
            r = r_0 + gamma * phi_new - phi_old + stop_penalty
        else:
            # Sparse reward: only terminal states get checker score + terminal bonus
            if is_terminal:
                r = phi_new + r_0 + stop_penalty
            else:
                r = stop_penalty  # Only STOP penalty if any
        
        return r
    
    def is_stop_terminal(self) -> bool:
        """
        Returns True if STOP action terminates the episode.
        
        Used by compute_exact_baseline_summation to determine whether to
        bootstrap from V(s') or treat STOP as absorbing.
        """
        return self._stop_mode == "terminal"

    def apply_edit(self, y: Any, action: int, x: Any) -> Any:
        """
        Apply an edit action to the plan y.

        Action decoding:
          - STOP action id (set via set_stop_action_id) terminates the episode.
          - UNDO action id (if enabled) reverts to previous plan state.
          - Other actions represent (position, token) edits using a flattened index.
        """

        if action == self.stop_action_id:
            return y

        # Handle UNDO action
        if self._enable_undo and action == self.undo_action_id:
            return self._apply_undo(y)

        vocab_size = self.vocab_size
        if vocab_size is None:
            raise RuntimeError(
                "PlanEditEnv requires `vocab_size` in the config (or inferable from dataset)."
            )

        if torch.is_tensor(y):
            plan_tensor = y.clone()
        else:
            plan_tensor = torch.as_tensor(y)

        flat = plan_tensor.reshape(-1)
        num_positions = flat.numel()
        max_edit_action = num_positions * vocab_size

        if action < 0 or action >= max_edit_action:
            # Invalid edit: no-op; return the original plan.
            return plan_tensor

        pos = action // vocab_size
        tok = action % vocab_size
        if pos >= num_positions:
            return plan_tensor

        # Safety check: prevent editing "given" cells (clues)
        # This acts as a second line of defense behind action masking
        if torch.is_tensor(x):
            inputs = x
        elif isinstance(x, dict):
            inputs = x.get("inputs")
        else:
            inputs = None
        
        if inputs is not None:
            if torch.is_tensor(inputs):
                flat_inputs = inputs.view(-1)
            else:
                flat_inputs = torch.as_tensor(inputs, device=plan_tensor.device).view(-1)
            
            # Check if pos corresponds to a clue (value > 1)
            # Use safe indexing
            if pos < flat_inputs.numel():
                if flat_inputs[pos].item() > 1:
                    # Attempted to edit a clue! Return original plan (no-op)
                    return plan_tensor

        new_flat = flat.clone()
        new_flat[pos] = torch.as_tensor(tok, dtype=new_flat.dtype, device=new_flat.device)

        return new_flat.view_as(plan_tensor)

    def _apply_undo(self, y: Any) -> Any:
        """
        Apply UNDO action by reverting to the previous plan state.

        UNDO pops the last entry from the edit history and returns the previous state.
        If history has only 1 entry (initial state), returns the current plan (no-op).

        Args:
            y: Current plan (for fallback if can't undo)

        Returns:
            Previous plan state from history, or current plan if can't undo.
        """
        if len(self._edit_history) > 1:
            # Pop the current state and return the previous one
            self._edit_history.pop()
            return self._edit_history[-1].clone()
        else:
            # Can't undo past initial state - return current plan
            if torch.is_tensor(y):
                return y.clone()
            return torch.as_tensor(y)

    def step(self, action: int):
        """
        Take a discrete action in the plan-space MDP.

        STOP action behavior depends on config.stop_action_mode:
        - "terminal": STOP ends the episode immediately
        - "noop": STOP is a no-op (plan unchanged), episode continues
        - "disabled": STOP should have been masked, but if received, treated as noop

        UNDO action (if enabled):
        - Reverts to the previous plan state from history
        - Valid only when there's something to undo (history > 1)
        - No termination, episode continues

        Non-STOP/UNDO actions apply edits until max_edits or checker termination.
        """

        assert self.stop_action_id is not None, "stop_action_id must be set before calling step()"
        assert not self.done, "Cannot call step() on a finished episode. Call reset() first."

        done_reason: Optional[str] = None
        terminated_by_stop = False
        terminated_by_budget = False
        terminated_by_solved = False
        is_undo_action = False

        self.step_count += 1

        if action == self.stop_action_id:
            terminated_by_stop = True

            if self._stop_mode == "terminal":
                # Standard RL: STOP terminates the episode
                y_next = self.y
                done = True
                done_reason = "stop"
                self._stop_penalty = 0.0
            else:
                # "noop" or "disabled": STOP is a no-op, episode continues
                # This prevents STOP collapse where policy learns to always stop
                y_next = self.y
                done = False
                done_reason = None
                self._stop_penalty = self._stop_penalty_value
        elif self._enable_undo and action == self.undo_action_id:
            # UNDO action: revert to previous plan state
            is_undo_action = True
            self._stop_penalty = 0.0
            y_next = self.apply_edit(self.y, action, self.x)
            done = False
        else:
            # Non-stop, non-undo action: apply edit
            self._stop_penalty = 0.0
            y_next = self.apply_edit(self.y, action, self.x)
            done = False

            # Push new state to history for UNDO support
            if self._enable_undo:
                self._edit_history.append(y_next.clone() if torch.is_tensor(y_next) else torch.as_tensor(y_next))

        pending_budget_termination = False
        if not done and self.step_count >= self.config.max_edits:
            done = True
            pending_budget_termination = True

        phi_old: Optional[float] = None
        phi_new: Optional[float] = None

        if self.config.reward_shaping:
            phi_old = float(self.checker(self.x, self.y))

        needs_phi_new = (
            done or self.config.reward_shaping or self.config.solved_threshold is not None
        )
        if needs_phi_new:
            phi_new = float(self.checker(self.x, y_next))

        solved = (
            self.config.solved_threshold is not None
            and phi_new is not None
            and phi_new >= self.config.solved_threshold
        )
        if solved:
            done = True
            if done_reason is None:
                done_reason = "solved"
            terminated_by_solved = True

        # Solution-independent termination for Sudoku tasks (Fix for feasibility checker)
        # When solved_threshold is None, we use sudoku_is_solved() to detect completion.
        # This ensures episodes terminate immediately when the puzzle is solved,
        # preventing "solved then unsolved" scenarios that inflate success metrics.
        #
        # IMPORTANT: Only apply this check when task_type == "sudoku" to prevent
        # accidental termination for non-Sudoku tasks that happen to use 16/81-length plans.
        if not terminated_by_solved:
            is_sudoku_task = (self.config.task_type == "sudoku")
            plan_tensor = y_next if torch.is_tensor(y_next) else None
            if is_sudoku_task and plan_tensor is not None and plan_tensor.numel() in (16, 81):
                # This is a Sudoku task (4x4 or 9x9)
                if sudoku_is_solved(plan_tensor):
                    done = True
                    if done_reason is None:
                        done_reason = "solved"
                    terminated_by_solved = True

        if done_reason is None and pending_budget_termination:
            done_reason = "budget"
            terminated_by_budget = True

        # Compute reward using shared helper (ensures consistency with exact baseline)
        # See compute_transition_reward() for the full reward semantics.
        if self.config.reward_shaping:
            assert phi_old is not None and phi_new is not None
            r = self.compute_transition_reward(
                phi_old=phi_old,
                phi_new=phi_new,
                is_stop_action=terminated_by_stop,
                is_terminal=done,
                is_solved=terminated_by_solved,
            )
        else:
            # No shaping: only reward on terminal step
            r = self.compute_transition_reward(
                phi_old=phi_old if phi_old is not None else 0.0,
                phi_new=phi_new if phi_new is not None else 0.0,
                is_stop_action=terminated_by_stop,
                is_terminal=done,
                is_solved=terminated_by_solved,
            )

        # #region agent log
        if r > 0.0 or done:
            try:
                import json
                with open('.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "location": "rl/envs/plan_edit_env.py:step",
                        "message": "Step Info",
                        "data": {
                            "step_count": self.step_count,
                            "reward": r,
                            "done": done,
                            "done_reason": done_reason,
                            "solved": terminated_by_solved,
                            "phi_old": phi_old,
                            "phi_new": phi_new,
                            "action": action,
                            "is_stop": action == self.stop_action_id
                        },
                        "timestamp": 0,
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C"
                    }
                    f.write(json.dumps(log_entry) + "\n")
            except Exception: pass
        # #endregion

        info = {
            "done_reason": done_reason,
            "solved": terminated_by_solved,  # Explicit solved flag for clarity
            "terminated_by_stop": terminated_by_stop,
            "terminated_by_budget": terminated_by_budget,
            "terminated_by_solved": terminated_by_solved,
            "is_undo_action": is_undo_action,
            "phi_old": phi_old,
            "phi_new": phi_new,
        }

        self.y = y_next
        self.done = done

        # Update action mask after state change (important for UNDO validity)
        if self._enable_undo and not done:
            self._compute_action_mask()

        return (self.x, self.y), r, done, info

