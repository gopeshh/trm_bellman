"""
Shared utilities for preparing batched states and plans for the TRM RL stack.

These helpers normalize environment states into the batch dict format expected by
TinyRecursiveReasoningModel_ACTV1 and are used by both UPITrmTrainer and the
evaluation helpers in evaluators/rl_plan_evaluator.py.
"""

from typing import Any, Dict, Sequence

import torch


def state_is_batched(x: Dict[str, torch.Tensor]) -> bool:
    """
    Heuristic to detect whether the env already produced a batch of states.
    PlanEditEnv currently returns single instances, but custom envs may batch.
    """
    if not isinstance(x, dict):
        return False
    inputs = x.get("inputs")
    puzzle_ids = x.get("puzzle_identifiers")
    if not isinstance(inputs, torch.Tensor) or not isinstance(
        puzzle_ids, torch.Tensor
    ):
        return False
    if inputs.ndim not in (1, 2) or puzzle_ids.ndim not in (1, 2):
        raise ValueError(
            "Expected `inputs` and `puzzle_identifiers` to be 1D or 2D tensors, "
            f"got shapes {tuple(inputs.shape)} and {tuple(puzzle_ids.shape)}."
        )
    if inputs.ndim == 0:
        return False
    if puzzle_ids.ndim == 0:
        return False
    return inputs.shape[0] == puzzle_ids.shape[0]


def prepare_batch_x(
    x: Dict[str, Any],
    device: torch.device,
    batched: bool,
) -> Dict[str, torch.Tensor]:
    """
    Normalize env state dictionaries into the TRM batch dict format.
    Ensures we always return tensors shaped as:
      inputs  -> [B, seq_len]
      puzzle_identifiers -> [B]  (1D, one identifier per batch element)
    """
    if not isinstance(x, dict):
        raise TypeError("Expected environment state x to be a dict with tensor entries.")
    batch: Dict[str, torch.Tensor] = {}
    for key in ("inputs", "puzzle_identifiers"):
        if key not in x:
            raise KeyError(f"Expected key `{key}` in environment state for TRM inputs.")
        tensor = x[key]
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        
        if key == "puzzle_identifiers":
            # puzzle_identifiers must be 1D [B] - one identifier per sample
            # CastedSparseEmbedding indexes directly with this tensor
            if tensor.ndim == 0:
                # Scalar -> [1]
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 1:
                # Already 1D - if unbatched, ensure we have exactly one identifier
                if not batched:
                    tensor = tensor[:1]  # Take first identifier for single sample
            elif tensor.ndim == 2:
                # [B, N] -> [B] by taking first identifier per sample
                tensor = tensor[:, 0]
            else:
                raise ValueError(f"puzzle_identifiers has unexpected ndim={tensor.ndim}, shape={tensor.shape}")
        elif key == "inputs":
            if not batched:
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        
        tensor = tensor.to(device).to(torch.long)
        batch[key] = tensor

    # The edit clock is part of the MDP state. The current TRM may choose to
    # ignore it, but retaining it keeps transition and diagnostic code Markov.
    if "remaining_edits" in x:
        remaining = x["remaining_edits"]
        if not torch.is_tensor(remaining):
            remaining = torch.as_tensor(remaining)
        if remaining.ndim == 0:
            remaining = remaining.unsqueeze(0)
        elif not batched:
            remaining = remaining.reshape(-1)[:1]
        batch["remaining_edits"] = remaining.to(device=device, dtype=torch.long)
    return batch


def stack_batch_states(
    states: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Stack replay or rollout states without dropping the edit clock.

    ``remaining_edits`` is transition-relevant state. A mixed batch where only
    some records contain it is malformed, so fail instead of silently training
    on a projection of the state.
    """

    if not states:
        raise ValueError("Cannot stack an empty state batch.")

    required = ("inputs", "puzzle_identifiers")
    for index, state in enumerate(states):
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(
                f"State {index} is missing required fields: {missing}"
            )

    inputs = torch.stack([state["inputs"] for state in states], dim=0).to(device)
    puzzle_ids = torch.stack(
        [normalize_puzzle_id(state["puzzle_identifiers"]) for state in states],
        dim=0,
    ).to(device)
    if puzzle_ids.dim() > 1:
        puzzle_ids = puzzle_ids.squeeze(-1)

    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_ids,
    }

    clock_presence = ["remaining_edits" in state for state in states]
    if any(clock_presence) and not all(clock_presence):
        raise ValueError(
            "Replay/rollout batch mixes states with and without remaining_edits."
        )
    if all(clock_presence):
        clocks = []
        for index, state in enumerate(states):
            clock = state["remaining_edits"]
            if clock.numel() != 1:
                raise ValueError(
                    "remaining_edits must be scalar per state; "
                    f"state {index} has shape {tuple(clock.shape)}."
                )
            clocks.append(clock.reshape(()))
        batch["remaining_edits"] = torch.stack(clocks, dim=0).to(
            device=device,
            dtype=torch.long,
        )

    return batch


def normalize_puzzle_id(pid: torch.Tensor) -> torch.Tensor:
    """
    Normalize a puzzle identifier tensor to a scalar tensor.
    
    Handles various input shapes:
    - Scalar (0D) -> returns as-is
    - 1D with single element -> squeezes to scalar
    - 1D with multiple elements -> takes first element
    - 2D -> squeezes and takes first element
    
    Args:
        pid: Puzzle identifier tensor of any shape
        
    Returns:
        Scalar tensor containing the puzzle identifier
    """
    if pid.dim() > 0:
        pid = pid.squeeze()
    if pid.dim() == 0:
        return pid
    # Multi-element tensor: take first element
    return pid[0] if pid.numel() > 1 else pid


def prepare_plan(
    y: Any,
    device: torch.device,
    batched: bool,
) -> torch.Tensor:
    """
    Convert plan objects to tensors shaped [B, seq_len].
    Currently treats plans as simple tensors mirroring the input tokens.
    """
    if isinstance(y, dict):
        plan = y.get("inputs")
        if plan is None:
            raise KeyError("Dictionary plan must contain an `inputs` tensor.")
    else:
        plan = y
    if not torch.is_tensor(plan):
        plan = torch.as_tensor(plan)
    if not batched:
        plan = plan.unsqueeze(0)
    return plan.to(device).to(torch.long)
