from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch


@dataclass
class PlanEditEnvConfig:
    max_edits: int
    gamma: float
    reward_shaping: bool = True
    # Optional task type for future specialization (e.g., "sudoku", "arc", "maze")
    task_type: Optional[str] = None
    vocab_size: Optional[int] = None  # number of discrete plan tokens


class PlanEditEnv:
    """
    Simple plan-space meta–MDP:
      - State: (x, y), where x is an instance and y is a plan / candidate solution.
      - Actions: edit actions over y plus a special STOP action.
      - Reward: derived from a checker c(x, y) with potential-based shaping.
    """

    def __init__(
        self,
        dataset: Any,
        checker: Callable[[Any, Any], float],
        config: PlanEditEnvConfig,
    ):
        self.dataset = dataset
        self.checker = checker
        self.config = config

        self.vocab_size: Optional[int] = config.vocab_size
        if self.vocab_size is None:
            self.vocab_size = self._infer_vocab_size()

        self.step_count: int = 0
        self.x: Any = None
        self.y: Any = None
        self.done: bool = False

        # Action space: caller is responsible for interpreting action indices.
        # We require STOP to be the last action index by convention.
        self.stop_action_id: Optional[int] = None

    def set_stop_action_id(self, stop_id: int) -> None:
        """
        Configure which discrete action index corresponds to STOP.
        """

        self.stop_action_id = stop_id

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
            self.x = sample
            if "initial_plan" in sample:
                self.y = sample["initial_plan"]
            else:
                # Default: trivial zero plan with same shape as inputs
                self.y = torch.zeros_like(sample["inputs"])
        else:
            # Fallback: treat sample as x and create a trivial zero plan
            self.x = sample
            self.y = torch.zeros_like(sample)

        self.step_count = 0
        self.done = False
        return self.x, self.y

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

    def apply_edit(self, y: Any, action: int, x: Any) -> Any:
        """
        Apply an edit action to the plan y.

        Action decoding:
          - STOP action id (set via set_stop_action_id) terminates the episode.
          - Other actions represent (position, token) edits using a flattened index.
        """

        if action == self.stop_action_id:
            return y

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
            # Invalid edit: no-op.
            return plan_tensor

        pos = action // vocab_size
        tok = action % vocab_size
        if pos >= num_positions:
            return plan_tensor

        new_flat = flat.clone()
        new_flat[pos] = torch.as_tensor(tok, dtype=new_flat.dtype, device=new_flat.device)

        return new_flat.view_as(plan_tensor)

    def step(self, action: int):
        """
        Take a discrete action in the plan-space MDP.
        - action == stop_action_id: terminate without changing y.
        - otherwise: apply edit and continue until max_edits or checker termination.
        """

        assert self.stop_action_id is not None, "stop_action_id must be set before calling step()"
        assert not self.done, "Cannot call step() on a finished episode. Call reset() first."

        self.step_count += 1
        if action == self.stop_action_id:
            # terminal, no edit
            y_next = self.y
            done = True
        else:
            y_next = self.apply_edit(self.y, action, self.x)
            done = self.step_count >= self.config.max_edits

        # Base reward from checker: only on terminal step
        r_base = 0.0
        if done:
            r_base = float(self.checker(self.x, y_next))

        # Potential-based shaping
        if self.config.reward_shaping:
            phi_old = float(self.checker(self.x, self.y))
            phi_new = float(self.checker(self.x, y_next))
            r = r_base + self.config.gamma * phi_new - phi_old
        else:
            r = r_base

        self.y = y_next
        self.done = done
        return (self.x, self.y), r, done, {}

