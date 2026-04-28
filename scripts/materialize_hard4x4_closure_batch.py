#!/usr/bin/env python3
"""
Materialize the frozen hard-4x4 closure batch and fixed finite-difference directions.

This script intentionally uses a dedicated local RNG path so the held-out
closure batch does not share mutable state with any training data loader.
It samples from the requested dataset split of the canonical 6--8-empty
no-mask distribution and writes compact `.npz` artifacts into the paper repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        required=True,
        help="Canonical dataset roots for the requested split.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split to materialize from (default: test).",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        required=True,
        help="Where to write the frozen closure batch (.npz).",
    )
    parser.add_argument(
        "--directions-npz",
        type=Path,
        required=True,
        help="Where to write the fixed L_V / latent perturbation directions (.npz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Dedicated RNG seed for the closure batch and direction draws.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=256,
        help="Number of held-out puzzle instances to materialize.",
    )
    parser.add_argument(
        "--direction-count",
        type=int,
        default=16,
        help="Number of unit directions to freeze for finite differences.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="TRM hidden size used to determine the flattened latent dimension.",
    )
    return parser.parse_args()


def collect_split_samples(dataset_paths: List[str], *, split: str) -> Dict[str, np.ndarray]:
    ds_cfg = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=dataset_paths,
        global_batch_size=256,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(ds_cfg, split=split)

    inputs: List[np.ndarray] = []
    puzzle_ids: List[np.ndarray] = []
    plans: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    source_sets: List[str] = []

    for set_name, batch, _ in dataset:
        batch_inputs = batch["inputs"].cpu().numpy()
        batch_ids = batch["puzzle_identifiers"].cpu().numpy()
        batch_labels = batch.get("labels")
        batch_labels_np = batch_labels.cpu().numpy() if batch_labels is not None else None
        for idx in range(batch_inputs.shape[0]):
            if int(batch_ids[idx]) == int(dataset.metadata.blank_identifier_id):
                continue
            inputs.append(batch_inputs[idx].astype(np.int32, copy=True))
            puzzle_ids.append(np.asarray(batch_ids[idx]).astype(np.int32, copy=True))
            plans.append(batch_inputs[idx].astype(np.int32, copy=True))
            if batch_labels_np is not None:
                labels.append(batch_labels_np[idx].astype(np.int32, copy=True))
            source_sets.append(set_name)

    if not inputs:
        raise RuntimeError(
            f"No samples were loaded from split={split} for the requested dataset paths."
        )

    result = {
        "inputs": np.stack(inputs, axis=0),
        "puzzle_identifiers": np.stack(puzzle_ids, axis=0).reshape(-1),
        "initial_plan": np.stack(plans, axis=0),
        "source_set": np.asarray(source_sets),
    }
    if labels:
        result["labels"] = np.stack(labels, axis=0)
    return result


def sample_closure_batch(
    pool: Dict[str, np.ndarray],
    *,
    num_instances: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    total = int(pool["inputs"].shape[0])
    if num_instances > total:
        raise ValueError(f"Requested {num_instances} instances but test split only has {total}.")

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=num_instances, replace=False))
    sampled = {key: value[indices] for key, value in pool.items()}
    sampled["sample_indices"] = indices.astype(np.int32, copy=False)
    sampled["rng_seed"] = np.asarray([seed], dtype=np.int32)
    return sampled


def build_unit_directions(
    *,
    count: int,
    seq_len: int,
    hidden_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = seq_len * hidden_size
    directions = rng.standard_normal(size=(count, dim), dtype=np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return directions / norms


def main() -> int:
    args = parse_args()
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.directions_npz.parent.mkdir(parents=True, exist_ok=True)

    pool = collect_split_samples(args.dataset_paths, split=args.split)
    closure_batch = sample_closure_batch(
        pool,
        num_instances=int(args.num_instances),
        seed=int(args.seed),
    )

    seq_len = int(closure_batch["inputs"].shape[1])
    lv_directions = build_unit_directions(
        count=int(args.direction_count),
        seq_len=seq_len,
        hidden_size=int(args.hidden_size),
        seed=int(args.seed),
    )
    latent_directions = build_unit_directions(
        count=int(args.direction_count),
        seq_len=2 * seq_len,
        hidden_size=int(args.hidden_size),
        seed=int(args.seed) + 1,
    )

    batch_metadata = {
        "dataset_paths": list(args.dataset_paths),
        "split": str(args.split),
        "num_instances": int(args.num_instances),
        "rng_seed": int(args.seed),
        "seq_len": seq_len,
        "hidden_size": int(args.hidden_size),
        "direction_count": int(args.direction_count),
        "direction_eps": 1e-4,
    }

    np.savez_compressed(
        args.output_npz,
        **closure_batch,
        metadata_json=np.asarray(json.dumps(batch_metadata)),
    )
    np.savez_compressed(
        args.directions_npz,
        lv_directions=lv_directions.astype(np.float32, copy=False),
        latent_directions=latent_directions.astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps(batch_metadata)),
    )

    print(
        json.dumps(
            {
                "closure_batch": str(args.output_npz),
                "directions": str(args.directions_npz),
                "num_instances": int(args.num_instances),
                "seq_len": seq_len,
                "hidden_size": int(args.hidden_size),
                "direction_count": int(args.direction_count),
                "rng_seed": int(args.seed),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
