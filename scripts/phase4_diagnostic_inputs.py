#!/usr/bin/env python3
"""Exact input identity for Phase 4 finite checkpoint diagnostics."""

import hashlib
import io
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from utils.run_identity import canonical_json_sha256
from scripts.phase4_result_schema import (
    PHASE4_DIAGNOSTIC_DATASET_NAME,
    PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT,
    PHASE4_DIAGNOSTIC_SPLITS,
    PHASE4_DIAGNOSTIC_STATE_COUNT,
    PHASE4_LIPSCHITZ_PERTURBATION_SCHEME,
    PHASE4_LIPSCHITZ_PERTURBATION_SEED,
)


class Phase4DiagnosticInputError(RuntimeError):
    """Raised when the fixed diagnostic input population cannot be authenticated."""


def _stable_read(path: Path, label: str) -> Tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Phase4DiagnosticInputError(
            f"{label} is not a readable regular file."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Phase4DiagnosticInputError(f"{label} must be a regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise Phase4DiagnosticInputError(f"{label} changed while it was read.")
    if len(payload) != before.st_size:
        raise Phase4DiagnosticInputError(f"{label} size changed while it was read.")
    return payload, hashlib.sha256(payload).hexdigest()


def _load_integer_array(
    path: Path,
    *,
    label: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    payload, sha256 = _stable_read(path, label)
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise Phase4DiagnosticInputError(f"{label} is not a valid NPY array.") from exc
    if not isinstance(array, np.ndarray) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise Phase4DiagnosticInputError(f"{label} must contain an integer array.")
    identity = {
        "relative_path": path.as_posix(),
        "bytes": len(payload),
        "sha256": sha256,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    return array, identity


def load_phase4_diagnostic_states(
    data_root: str | Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load the exact ordered 100-state diagnostic population and its identity."""

    try:
        resolved_root = Path(data_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Phase4DiagnosticInputError(
            "Phase 4 diagnostic data root must exist."
        ) from exc
    if not resolved_root.is_dir():
        raise Phase4DiagnosticInputError(
            "Phase 4 diagnostic data root must be a directory."
        )

    states: List[Dict[str, Any]] = []
    split_identities: List[Dict[str, Any]] = []
    for split in PHASE4_DIAGNOSTIC_SPLITS:
        relative_inputs = Path(PHASE4_DIAGNOSTIC_DATASET_NAME) / split / "all__inputs.npy"
        relative_identifiers = (
            Path(PHASE4_DIAGNOSTIC_DATASET_NAME)
            / split
            / "all__puzzle_identifiers.npy"
        )
        inputs, inputs_identity = _load_integer_array(
            resolved_root / relative_inputs,
            label=f"Phase 4 {split} inputs",
        )
        identifiers, identifiers_identity = _load_integer_array(
            resolved_root / relative_identifiers,
            label=f"Phase 4 {split} puzzle identifiers",
        )
        inputs_identity["relative_path"] = relative_inputs.as_posix()
        identifiers_identity["relative_path"] = relative_identifiers.as_posix()
        if inputs.ndim != 2 or inputs.shape[1] != 16:
            raise Phase4DiagnosticInputError(
                f"Phase 4 {split} inputs must have shape [N, 16]."
            )
        if identifiers.ndim != 1 or identifiers.shape[0] != inputs.shape[0]:
            raise Phase4DiagnosticInputError(
                f"Phase 4 {split} identifiers must have shape [N]."
            )
        if inputs.shape[0] < PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT:
            raise Phase4DiagnosticInputError(
                f"Phase 4 {split} requires at least "
                f"{PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT} ordered records."
            )

        selected_inputs = inputs[:PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT]
        selected_identifiers = identifiers[:PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT]
        if selected_inputs.size == 0 or int(selected_inputs.min()) < 0 or int(
            selected_inputs.max()
        ) >= 32:
            raise Phase4DiagnosticInputError(
                f"Phase 4 {split} inputs must use token IDs in [0, 31]."
            )
        if selected_identifiers.size == 0 or int(selected_identifiers.min()) < 0 or int(
            selected_identifiers.max()
        ) >= 32:
            raise Phase4DiagnosticInputError(
                f"Phase 4 {split} identifiers must lie in [0, 31]."
            )

        split_states = [
            {
                "inputs": selected_inputs[index].astype(np.int64).tolist(),
                "puzzle_identifier": int(selected_identifiers[index]),
                "plan": selected_inputs[index].astype(np.int64).tolist(),
            }
            for index in range(PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT)
        ]
        states.extend(split_states)
        split_identities.append(
            {
                "name": split,
                "available_records": int(inputs.shape[0]),
                "selected_records": PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT,
                "inputs": inputs_identity,
                "puzzle_identifiers": identifiers_identity,
                "selected_records_sha256": canonical_json_sha256(split_states),
            }
        )

    identity = {
        "schema_version": 1,
        "dataset_name": PHASE4_DIAGNOSTIC_DATASET_NAME,
        "selection": "first_n_in_file_order",
        "records_per_split": PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT,
        "total_selected_records": len(states),
        "splits": split_identities,
        "ordered_states_sha256": canonical_json_sha256(states),
    }
    if len(states) != PHASE4_DIAGNOSTIC_STATE_COUNT:
        raise Phase4DiagnosticInputError(
            "Phase 4 diagnostic population has an unexpected state count."
        )
    return states, identity


def verify_phase4_diagnostic_inputs(
    summary: Mapping[str, Any],
    data_root: str | Path,
) -> int:
    """Recompute and compare the input and perturbation identity in a summary."""

    _, actual_identity = load_phase4_diagnostic_states(data_root)
    expected_identity = summary.get("diagnostic_dataset")
    if expected_identity != actual_identity:
        raise Phase4DiagnosticInputError(
            "Phase 4 diagnostic data identity does not match the publication record."
        )
    actual_sha256 = canonical_json_sha256(actual_identity)
    if summary.get("diagnostic_dataset_sha256") != actual_sha256:
        raise Phase4DiagnosticInputError(
            "Phase 4 diagnostic data digest does not match the publication record."
        )
    if summary.get("lipschitz_perturbation_seed") != (
        PHASE4_LIPSCHITZ_PERTURBATION_SEED
    ) or summary.get("lipschitz_perturbation_scheme") != (
        PHASE4_LIPSCHITZ_PERTURBATION_SCHEME
    ):
        raise Phase4DiagnosticInputError(
            "Phase 4 perturbation identity does not match the registered scheme."
        )
    return PHASE4_DIAGNOSTIC_STATE_COUNT
