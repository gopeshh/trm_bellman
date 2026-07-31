"""Deterministic provenance helpers for ordered evaluation pools."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def sample_sha256(inputs: Any, solution: Any) -> str:
    payload = {
        "inputs": _to_list(inputs),
        "solution": _to_list(solution),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_sha256(inputs: Any) -> str:
    encoded = json.dumps(
        _to_list(inputs),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_pool_sha256(
    inputs: Sequence[Any],
    solutions: Sequence[Any],
    count: int,
) -> str:
    if count < 1:
        raise ValueError("Evaluation pool count must be positive.")
    if len(inputs) < count or len(solutions) < count:
        raise ValueError(
            f"Evaluation pool has fewer than {count} records: "
            f"inputs={len(inputs)}, solutions={len(solutions)}."
        )
    digest = hashlib.sha256()
    for index in range(count):
        digest.update(sample_sha256(inputs[index], solutions[index]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_sample_sha256s(dataset: Any) -> list[str]:
    fingerprints = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, dict) or "inputs" not in sample:
            raise TypeError("Dataset provenance requires dictionary samples with inputs.")
        fingerprints.append(
            sample_sha256(
                sample["inputs"],
                sample.get("solution", sample.get("labels")),
            )
        )
    return fingerprints


def dataset_input_sha256s(dataset: Any) -> list[str]:
    fingerprints = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, dict) or "inputs" not in sample:
            raise TypeError("Dataset provenance requires dictionary samples with inputs.")
        fingerprints.append(input_sha256(sample["inputs"]))
    return fingerprints


def sequence_input_sha256s(inputs: Sequence[Any]) -> list[str]:
    return [input_sha256(value) for value in inputs]


def dataset_pool_sha256(dataset: Any, count: int) -> str:
    if len(dataset) < count:
        raise ValueError(f"Dataset has {len(dataset)} samples; expected at least {count}.")
    inputs = []
    solutions = []
    for index in range(count):
        sample = dataset[index]
        if not isinstance(sample, dict) or "inputs" not in sample:
            raise TypeError("Dataset provenance requires dictionary samples with inputs.")
        inputs.append(sample["inputs"])
        solutions.append(sample.get("solution", sample.get("labels")))
    return ordered_pool_sha256(inputs, solutions, count)
