"""Shared dataset bootstrap and checker-resolution helpers for training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from rl.sudoku_checkers import (
    dummy_checker,
    make_sudoku_feasibility_checker,
    sudoku_checker,
    sudoku_constraint_checker,
    sudoku_progress_checker,
)


class DummyPuzzleDataset:
    """
    Tiny in-memory dataset suitable for smoke tests of the RL loop.
    """

    def __init__(self, num_instances: int = 32, seq_len: int = 16, vocab_size: int = 32):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_instances

        self.samples: List[dict] = []
        for idx in range(num_instances):
            inputs = torch.randint(low=0, high=vocab_size, size=(seq_len,), dtype=torch.long)
            puzzle_identifier = torch.tensor(idx, dtype=torch.long)
            sample = {
                "inputs": inputs,
                "puzzle_identifiers": puzzle_identifier,
                "initial_plan": torch.zeros_like(inputs),
                "solution": inputs.clone(),
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class OfflinePuzzleDataset:
    """
    Wrap a finite list of samples gathered from PuzzleDataset.
    """

    def __init__(self, samples: List[dict], seq_len: int, vocab_size: int, num_identifiers: int):
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_identifiers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_dataset_from_paths(
    dataset_paths: Optional[List[str]],
    pool_size: int,
) -> Tuple[object, int, int, int]:
    """
    Attempt to load supervised samples for RL bootstrap, then fall back to dummy data.
    """

    if dataset_paths:
        try:
            ds_cfg = PuzzleDatasetConfig(
                seed=0,
                dataset_paths=dataset_paths,
                global_batch_size=pool_size,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=0,
                num_replicas=1,
            )
            iterable = PuzzleDataset(ds_cfg, split="train")
            samples: List[dict] = []
            for _set_name, batch, _ in iterable:
                batch_inputs = batch["inputs"]
                batch_ids = batch["puzzle_identifiers"]
                batch_labels = batch.get("labels")
                batch_size = batch_inputs.shape[0]
                for i in range(batch_size):
                    inputs = batch_inputs[i].clone()
                    puzzle_id = batch_ids[i].clone()
                    solution = batch_labels[i].clone() if batch_labels is not None else None
                    initial_plan = inputs.clone()
                    samples.append(
                        {
                            "inputs": inputs,
                            "puzzle_identifiers": puzzle_id,
                            "initial_plan": initial_plan,
                            **({"solution": solution} if solution is not None else {}),
                        }
                    )
                    if len(samples) >= pool_size:
                        break
                if len(samples) >= pool_size:
                    break
            if samples:
                num_identifiers = int(
                    torch.stack([sample["puzzle_identifiers"] for sample in samples]).max().item() + 1
                )
                dataset = OfflinePuzzleDataset(
                    samples=samples,
                    seq_len=samples[0]["inputs"].shape[-1],
                    vocab_size=iterable.metadata.vocab_size,
                    num_identifiers=max(num_identifiers, iterable.metadata.num_puzzle_identifiers),
                )
                return dataset, dataset.seq_len, dataset.vocab_size, dataset.num_identifiers
        except Exception as exc:  # pragma: no cover - bootstrap stays permissive by design
            message = (
                "[upi_trm_train] Falling back to dummy dataset after supervised bootstrap "
                f"failed ({type(exc).__name__}: {exc})"
            )
            warnings.warn(message, RuntimeWarning)
            print(message)

    dummy = DummyPuzzleDataset()
    return dummy, dummy.seq_len, dummy.vocab_size, dummy.num_identifiers


@dataclass(frozen=True)
class ResolvedChecker:
    checker_fn: Callable[[Any, Any], float]
    checker_kind: str
    info_lines: Tuple[str, ...] = ()


def resolve_checker_from_dataset(rl_cfg: Any, dataset: Any, seq_len: int) -> ResolvedChecker:
    """Resolve the checker function and related log lines for the current dataset."""
    use_feasibility_checker = getattr(rl_cfg, "use_feasibility_checker", False)
    use_progress_checker = getattr(rl_cfg, "use_progress_checker", False)
    use_constraint_checker = getattr(rl_cfg, "use_constraint_checker", False)

    w_v = getattr(rl_cfg, "feasibility_violation_weight", 2.0)
    w_z = getattr(rl_cfg, "feasibility_zerocand_weight", 5.0)

    if len(dataset) == 0:
        return ResolvedChecker(dummy_checker, "dummy")

    sample = dataset[0]
    if not (isinstance(sample, dict) and "solution" in sample):
        return ResolvedChecker(dummy_checker, "dummy")

    if use_feasibility_checker and seq_len in (16, 81):
        grid_size = "4x4" if seq_len == 16 else "9x9"
        return ResolvedChecker(
            checker_fn=make_sudoku_feasibility_checker(w_v=w_v, w_z=w_z),
            checker_kind="feasibility",
            info_lines=(
                f"[INFO] Using feasibility-aware Sudoku checker for {grid_size}",
                f"       score = filled - {w_v}*violations - {w_z}*zeroCand",
                f"       Max score: {seq_len} (all filled, no violations)",
            ),
        )

    if use_progress_checker and seq_len in (16, 81):
        grid_size = "4x4" if seq_len == 16 else "9x9"
        return ResolvedChecker(
            checker_fn=sudoku_progress_checker,
            checker_kind="progress",
            info_lines=(
                f"[INFO] Using progress-based Sudoku checker for {grid_size} (score = filled_cells, range 0-{seq_len})",
            ),
        )

    if use_constraint_checker and seq_len in (16, 81):
        grid_size = "4x4" if seq_len == 16 else "9x9"
        return ResolvedChecker(
            checker_fn=sudoku_constraint_checker,
            checker_kind="constraint",
            info_lines=(
                f"[INFO] Using constraint-based Sudoku checker for {grid_size} for dense intermediate rewards",
            ),
        )

    return ResolvedChecker(sudoku_checker, "solution")
