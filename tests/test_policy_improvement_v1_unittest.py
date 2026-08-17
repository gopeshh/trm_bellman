#!/usr/bin/env fbpython
"""Focused tests for the registered learned policy-improvement infrastructure."""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

try:
    import numpy as np
except ImportError:  # The owned Buck test supplies NumPy; host fbpython may not.
    np = None
from scripts.policy_improvement_analysis import (
    _contrast_estimates,
    analyze_stage2_confirmatory as _analyze_stage2_confirmatory,
)
from scripts.policy_improvement_audit import (
    _canonical_sudoku_record_sha256,
    _count_sudoku_solutions,
    _input_sha256,
    _load_dataset_bindings,
    _sample_sha256,
    audit_result_set as _audit_result_set,
    derive_registered_selection,
)
from phase4_runtime_profile import POLICY_DATASET_BUILDER_PROFILE_PATHS
from scripts.policy_improvement_registry import (
    derive_seed,
    generate_registry,
    load_registered_base_configs,
    registry_sha256,
    validate_registry_document,
)
from scripts.policy_improvement_schema import (
    amendment_history_sha256,
    canonical_json_bytes,
    load_strict_json,
    policy_variants_for_method,
    primary_policy_variant_for_method,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    stage3_method_id,
    validate_amendment_history,
    validate_compute_freeze,
    validate_final_selection,
    validate_protocol,
    validate_result,
    validate_runtime_authorization,
    validate_screen_selection,
)
from scripts.policy_improvement_smoke_plan import render_smoke_plan
from scripts.policy_improvement_statistics import (
    holm_adjust,
    paired_seed_cluster_puzzle_bootstrap,
    paired_seed_permutation_test,
)
from scripts.policy_improvement_test_open import (
    _publish_record,
    expected_test_open_record,
)
from scripts.policy_improvement_test_open_cli import main as test_open_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs/policy_improvement_v1/protocol.json"
REGISTRY_PATH = REPOSITORY_ROOT / "configs/policy_improvement_v1/registry.json"
THEORY_AMENDMENT_PATH = (
    REPOSITORY_ROOT / "configs/policy_improvement_v1/amendments/theory_bridge_v1.json"
)
HEX64 = "a" * 64


def audit_result_set(*args: object, **kwargs: object) -> dict[str, Any]:
    kwargs.setdefault("project_root", REPOSITORY_ROOT)
    kwargs.setdefault("checkpoint_validator", _fixture_checkpoint_validator)
    kwargs.setdefault("_producer_source_authenticator", lambda _: None)
    return _audit_result_set(*args, **kwargs)


def analyze_stage2_confirmatory(*args: object, **kwargs: object) -> dict[str, Any]:
    kwargs.setdefault("project_root", REPOSITORY_ROOT)
    kwargs.setdefault("checkpoint_validator", _fixture_checkpoint_validator)
    kwargs.setdefault("_producer_source_authenticator", lambda _: None)
    return _analyze_stage2_confirmatory(*args, **kwargs)


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _unavailable(
    reason: str = "not_collected_by_registered_protocol",
) -> dict[str, object]:
    return {"status": "unavailable", "reason": reason}


def _evidence(phase: str, rows: int, marker: str) -> dict[str, object]:
    return {
        "phase": phase,
        "audit_report_sha256": marker * 64,
        "result_set_sha256": chr(ord(marker) + 1) * 64,
        "per_instance_set_sha256": chr(ord(marker) + 2) * 64,
        "expected_rows": rows,
        "complete_rows": rows,
        "failed_rows": 0,
    }


def _runtime_authorization(protocol: dict[str, object]) -> dict[str, object]:
    roles = []
    for index, role in enumerate(
        (
            "policy-improvement-training",
            "policy-improvement-evaluation",
            "policy-improvement-audit",
            "policy-improvement-analysis",
        ),
        start=1,
    ):
        roles.append(
            {
                "role": role,
                "source_git_commit": (
                    "b" * 40 if index == 1 else format(index + 7, "x") * 40
                ),
                "runtime_sha256": format(index, "x") * 64,
                "runtime_profile_sha256": format(index + 4, "x") * 64,
                "selected_source_manifest_sha256": format(index + 4, "x") * 64,
            }
        )
    return {
        "schema_name": "policy_improvement_runtime_authorization_v1",
        "schema_version": 1,
        "authorization_id": "runtime-authorization-fixture-v1",
        "created_at_utc": "2026-08-14T12:00:00Z",
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "producer_git_commit": "b" * 40,
        "producer_source_manifest_sha256": "c" * 64,
        "launcher_sha256": "d" * 64,
        "roles": roles,
    }


def _stage0_runtime_authorization(
    protocol: dict[str, object],
) -> dict[str, object]:
    authorization = copy.deepcopy(_runtime_authorization(protocol))
    training = authorization["roles"][0]
    training["runtime_profile_sha256"] = authorization[
        "producer_source_manifest_sha256"
    ]
    training["selected_source_manifest_sha256"] = training["runtime_profile_sha256"]
    authorization["roles"][1] = {
        **training,
        "role": "policy-improvement-evaluation",
    }
    return authorization


def _audit_execution(authorization: dict[str, object]) -> dict[str, object]:
    role = next(
        role
        for role in authorization["roles"]
        if role["role"] == "policy-improvement-audit"
    )
    return {
        "runtime_sha256": role["runtime_sha256"],
        "runtime_profile_sha256": role["runtime_profile_sha256"],
        "source_git_commit": role["source_git_commit"],
        "launcher_sha256": authorization["launcher_sha256"],
        "producer_git_commit": authorization["producer_git_commit"],
        "producer_source_manifest_sha256": authorization[
            "producer_source_manifest_sha256"
        ],
    }


def _theory_amendment(protocol: dict[str, object]) -> dict[str, object]:
    base = generate_registry(protocol)
    theory = copy.deepcopy(load_strict_json(THEORY_AMENDMENT_PATH))
    theory["protocol_id"] = protocol["protocol_id"]
    theory["protocol_sha256"] = hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()
    theory["source_registry_sha256"] = registry_sha256(base)
    return theory


def _amendment_history(
    protocol: dict[str, object],
    *,
    smoke_evidence: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    authorization = _runtime_authorization(protocol)
    theory = _theory_amendment(protocol)
    after_theory = generate_registry(protocol, [theory])
    compute = {
        "schema_name": "policy_improvement_compute_freeze_v1",
        "schema_version": 1,
        "amendment_id": "post-smoke-compute-freeze-v1",
        "created_at_utc": "2026-08-14T12:01:00Z",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "prior_amendment_history_sha256": amendment_history_sha256([theory]),
        "source_registry_sha256": registry_sha256(after_theory),
        "runtime_authorization_sha256": runtime_authorization_sha256(authorization),
        "test_data_opened": False,
        "evidence": smoke_evidence or _evidence("stage0_smoke", 4, "1"),
        "common_compute_targets": {
            "unit": "recurrent_map_applications",
            "pilot": 1000,
            "confirmatory": 2000,
            "ablation": 2000,
            "maximum_relative_mismatch": 0.05,
        },
    }
    after_compute = generate_registry(protocol, [theory, compute])
    screen = {
        "schema_name": "policy_improvement_screen_selection_v1",
        "schema_version": 1,
        "amendment_id": "post-screen-selection-v1",
        "created_at_utc": "2026-08-14T12:02:00Z",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": compute["protocol_sha256"],
        "prior_amendment_history_sha256": amendment_history_sha256([theory, compute]),
        "source_registry_sha256": registry_sha256(after_compute),
        "test_data_opened": False,
        "evidence": _evidence("stage1_screen", 48, "4"),
        "selected_exact": {
            "method_id": "fixed_base_exact_persistent",
            "n": 2,
            "K": 1,
        },
    }
    after_screen = generate_registry(protocol, [theory, compute, screen])
    selected = {
        "method_id": "fixed_base_exact_persistent",
        "n": 2,
        "K": 1,
        "alpha": 0.1,
    }
    variants = []
    from scripts.policy_improvement_schema import canonical_stage3_override

    for variant in (
        "batch_only_centering",
        "distilled_realization",
        "projection_identity",
        "contraction_enabled",
        "target_retention_0p9",
        "target_retention_0p999",
        "depth_lower_neighbor",
        "depth_upper_neighbor",
    ):
        override = canonical_stage3_override(variant, 2)
        variants.append(
            {
                "variant": variant,
                "method_id": stage3_method_id(variant, selected["method_id"]),
                "base_method_id": selected["method_id"],
                "n": int(override.get("inner_unroll_n", 2)),
                "K": 1,
                "alpha": 0.1,
                "override_payload": override,
            }
        )
    final = {
        "schema_name": "policy_improvement_final_selection_v1",
        "schema_version": 1,
        "amendment_id": "post-alpha-final-selection-v1",
        "created_at_utc": "2026-08-14T12:03:00Z",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": compute["protocol_sha256"],
        "prior_amendment_history_sha256": amendment_history_sha256(
            [theory, compute, screen]
        ),
        "source_registry_sha256": registry_sha256(after_screen),
        "test_data_opened": False,
        "evidence": _evidence("stage1_alpha", 9, "7"),
        "selected_exact": selected,
        "stage3_variants": variants,
    }
    return [theory, compute, screen, final], authorization


def _frozen_protocol() -> dict[str, object]:
    protocol = copy.deepcopy(validate_protocol(load_strict_json(PROTOCOL_PATH)))
    # A complete tiny fixture retains the registered validation and test pools,
    # but one training row keeps focused audit tests cheap.
    protocol["dataset"]["splits"]["train"]["count"] = 1
    protocol["dataset"]["manifest_sha256"] = _available("f" * 64)
    protocol["dataset"]["producer_source"] = {
        "git_commit": _available("b" * 40),
        "source_manifest_sha256": _available("1" * 64),
        "runtime_sha256": _available("2" * 64),
        "launcher_sha256": _available("3" * 64),
    }
    identities = {
        "train": ("c" * 64, "7" * 64),
        "validation": ("d" * 64, "8" * 64),
        "test": ("e" * 64, "9" * 64),
    }
    for split, (manifest, ordered) in identities.items():
        protocol["dataset"]["splits"][split]["manifest_sha256"] = _available(manifest)
        protocol["dataset"]["splits"][split]["ordered_record_sha256"] = _available(
            ordered
        )
    puzzle_hashes = [_fixture_record_sha256("validation", index) for index in range(8)]
    smoke_order = hashlib.sha256()
    for puzzle_hash in puzzle_hashes:
        smoke_order.update(puzzle_hash.encode("ascii"))
        smoke_order.update(b"\n")
    protocol["evaluation_populations"]["smoke"]["ordered_record_sha256"] = _available(
        smoke_order.hexdigest()
    )
    protocol["evaluation_populations"]["pilot"]["ordered_record_sha256"] = _available(
        "8" * 64
    )
    protocol["evaluation_populations"]["confirmatory"]["ordered_record_sha256"] = (
        _available("9" * 64)
    )
    return validate_protocol(protocol)


def _ordered_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


_FIXTURE_SOLUTION = (2, 3, 4, 5, 4, 5, 2, 3, 3, 2, 5, 4, 5, 4, 3, 2)
_FIXTURE_RECORDS: tuple[tuple[int, ...], ...] | None = None


def _fixture_sudoku_records(count: int) -> tuple[tuple[int, ...], ...]:
    """Generate deterministic, unique, symmetry-disjoint, solvable records."""

    global _FIXTURE_RECORDS
    if _FIXTURE_RECORDS is None or len(_FIXTURE_RECORDS) < count:
        values: list[tuple[int, ...]] = []
        symmetry_seen: set[str] = set()
        for blank_count in range(4, 13):
            for blank_indices in itertools.combinations(range(16), blank_count):
                puzzle = list(_FIXTURE_SOLUTION)
                for index in blank_indices:
                    puzzle[index] = 1
                encoded = tuple(puzzle)
                if _count_sudoku_solutions(encoded) != 1:
                    continue
                symmetry = _canonical_sudoku_record_sha256(encoded, _FIXTURE_SOLUTION)
                if symmetry in symmetry_seen:
                    continue
                symmetry_seen.add(symmetry)
                values.append(encoded)
                if len(values) == count:
                    break
            if len(values) == count:
                break
        if len(values) != count:
            raise AssertionError("Could not construct the tiny registered corpus.")
        _FIXTURE_RECORDS = tuple(values)
    return _FIXTURE_RECORDS[:count]


def _fixture_puzzle(split: str, index: int) -> tuple[int, ...]:
    offsets = {"train": 0, "validation": 1, "test": 257}
    return _fixture_sudoku_records(769)[offsets[split] + index]


def _fixture_record_sha256(split: str, index: int) -> str:
    return _sample_sha256(_fixture_puzzle(split, index), _FIXTURE_SOLUTION)


def _fixture_input_sha256(split: str, index: int) -> str:
    return _input_sha256(_fixture_puzzle(split, index))


def _npy_int32_bytes(shape: tuple[int, ...], values: list[int]) -> bytes:
    if math.prod(shape) != len(values):
        raise AssertionError("NPY fixture shape differs from its values.")
    header = repr({"descr": "<i4", "fortran_order": False, "shape": shape}).encode(
        "latin1"
    )
    padding = (-((10 + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    if len(header) > 65535:
        raise AssertionError("NPY fixture header exceeds version-1 limits.")
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + struct.pack(f"<{len(values)}i", *values)
    )


def _write_fixture_file(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _materialize_dataset_fixture(
    protocol: dict[str, object], parent: Path
) -> tuple[dict[str, object], Path]:
    """Write a complete tiny corpus using the registered materialized schema."""

    checked = copy.deepcopy(protocol)
    root = parent / str(checked["dataset"]["root"])
    owner = root.parent
    owner.mkdir(parents=True, mode=0o700)
    owner.chmod(0o700)
    root.mkdir(mode=0o700)
    manifests = root / "manifests"
    manifests.mkdir()
    top_splits: dict[str, object] = {}
    offset = 0
    total_count = sum(
        int(checked["dataset"]["splits"][name]["count"])
        for name in ("train", "validation", "test")
    )
    all_puzzles = _fixture_sudoku_records(total_count)
    for split in ("train", "validation", "test"):
        registration = checked["dataset"]["splits"][split]
        count = int(registration["count"])
        split_puzzles = all_puzzles[offset : offset + count]
        records = [
            _sample_sha256(puzzle, _FIXTURE_SOLUTION) for puzzle in split_puzzles
        ]
        inputs = [_input_sha256(puzzle) for puzzle in split_puzzles]
        symmetry = [
            _canonical_sudoku_record_sha256(puzzle, _FIXTURE_SOLUTION)
            for puzzle in split_puzzles
        ]
        split_root = root / split
        split_root.mkdir()
        materialized_records = [
            {
                "index": index,
                "record_sha256": records[index],
                "symmetry_canonical_sha256": symmetry[index],
            }
            for index in range(count)
        ]
        arrays = {
            "all__inputs.npy": _npy_int32_bytes(
                (count, 16), [value for puzzle in split_puzzles for value in puzzle]
            ),
            "all__labels.npy": _npy_int32_bytes(
                (count, 16), list(_FIXTURE_SOLUTION) * count
            ),
            "all__puzzle_identifiers.npy": _npy_int32_bytes(
                (count,), list(range(offset, offset + count))
            ),
            "all__puzzle_indices.npy": _npy_int32_bytes(
                (count + 1,), list(range(count + 1))
            ),
            "all__group_indices.npy": _npy_int32_bytes(
                (count + 1,), list(range(count + 1))
            ),
            "dataset.json": canonical_json_bytes(
                {
                    "blank_identifier_id": 0,
                    "ignore_label_id": 0,
                    "mean_puzzle_examples": 1,
                    "num_puzzle_identifiers": total_count,
                    "pad_id": 0,
                    "seq_len": 16,
                    "sets": ["all"],
                    "total_groups": count,
                    "total_puzzles": count,
                    "vocab_size": 6,
                }
            ),
            "records.json": canonical_json_bytes(materialized_records),
        }
        file_inventory = {
            f"{split}/{name}": _write_fixture_file(split_root / name, payload)
            for name, payload in sorted(arrays.items())
        }
        split_manifest = {
            "build_schema_version": checked["dataset"]["builder_schema_version"],
            "builder": "dataset.build_policy_improvement_4x4",
            "files": file_inventory,
            "generated_count": count,
            "generation_seed": registration["generation_seed"],
            "input_sha256s": inputs,
            "ordered_record_sha256": _ordered_digest(records),
            "ordered_symmetry_sha256": _ordered_digest(symmetry),
            "record_sha256s": records,
            "symmetry_canonical_sha256s": symmetry,
            "symmetry_canonicalization": checked["dataset"][
                "symmetry_canonicalization"
            ]["scheme"],
        }
        split_bytes = canonical_json_bytes(split_manifest)
        split_path = manifests / f"{split}.json"
        split_path.write_bytes(split_bytes)
        split_digest = hashlib.sha256(split_bytes).hexdigest()
        registration["manifest_sha256"] = _available(split_digest)
        registration["ordered_record_sha256"] = _available(
            split_manifest["ordered_record_sha256"]
        )
        top_splits[split] = {
            "count": count,
            "generation_seed": registration["generation_seed"],
            "manifest_sha256": split_digest,
            "ordered_record_sha256": split_manifest["ordered_record_sha256"],
            "ordered_symmetry_sha256": split_manifest["ordered_symmetry_sha256"],
            "puzzle_identifier_start": offset,
        }
        offset += count
    checked["evaluation_populations"]["smoke"]["ordered_record_sha256"] = _available(
        _ordered_digest(
            [_sample_sha256(puzzle, _FIXTURE_SOLUTION) for puzzle in all_puzzles[1:9]]
        )
    )
    checked["evaluation_populations"]["pilot"]["ordered_record_sha256"] = checked[
        "dataset"
    ]["splits"]["validation"]["ordered_record_sha256"]
    checked["evaluation_populations"]["confirmatory"]["ordered_record_sha256"] = (
        checked["dataset"]["splits"]["test"]["ordered_record_sha256"]
    )
    producer = {
        key: checked["dataset"]["producer_source"][key]["value"]
        for key in (
            "git_commit",
            "launcher_sha256",
            "runtime_sha256",
            "source_manifest_sha256",
        )
    }
    top = {
        "build_schema_version": checked["dataset"]["builder_schema_version"],
        "builder": "dataset.build_policy_improvement_4x4",
        "canonicalization": checked["dataset"]["symmetry_canonicalization"],
        "manifest_schema_version": checked["dataset"]["manifest_schema_version"],
        "producer_source": producer,
        "producer_source_paths": list(POLICY_DATASET_BUILDER_PROFILE_PATHS),
        "splits": top_splits,
    }
    build_config = {
        "build_schema_version": checked["dataset"]["builder_schema_version"],
        "builder": "dataset.build_policy_improvement_4x4",
        "domain": "sudoku_4x4",
        "producer_source": producer,
        "require_symmetry_disjoint_splits": True,
        "require_unique_solution": True,
        "seed": checked["dataset"]["splits"]["train"]["generation_seed"],
        "split_order": ["train", "validation", "test"],
        "splits": {
            name: {
                "count": checked["dataset"]["splits"][name]["count"],
                "seed": checked["dataset"]["splits"][name]["generation_seed"],
            }
            for name in ("train", "validation", "test")
        },
        "symmetry_canonicalization": checked["dataset"]["symmetry_canonicalization"][
            "scheme"
        ],
    }
    (root / "build_config.json").write_bytes(canonical_json_bytes(build_config))
    (root / "identifiers.json").write_bytes(canonical_json_bytes(["<blank>"]))
    top_bytes = canonical_json_bytes(top)
    (root / "MANIFEST.json").write_bytes(top_bytes)
    checked["dataset"]["manifest_sha256"] = _available(
        hashlib.sha256(top_bytes).hexdigest()
    )
    return validate_protocol(checked), root


def _valid_result() -> dict[str, object]:
    training = {
        "interactions_to_first_solve": _available(3),
        "value_loss": _available(0.2),
        "policy_loss": _available(-0.1),
        "wall_time_seconds": _available(2.0),
        "gpu_hours": _available(2.0 / 3600.0),
        "gpu_utilization_fraction": _available(0.5),
        "gpu_utilization_sample_count": _available(20),
        "gpu_utilization_sampling_interval_seconds": _available(0.1),
        "peak_allocated_memory_bytes": _available(1024),
        "peak_reserved_memory_bytes": _available(2048),
        "optimizer_steps": _available(2),
        "cells_processed": _available(512),
        "actions_processed": _available(128),
        "tokens_processed": _available(512),
        "policy_head_calls": _available(64),
        "recurrent_map_applications": _available(256),
        "value_head_calls": _available(64),
    }
    diagnostics = {
        name: _available(0.0)
        for name in (
            "L_preproj",
            "projection_active_rate",
            "absolute_depth_policy_agreement",
            "finite_depth_discrepancy",
            "fixed_target_residual",
            "propagated_target_lag",
            "exact_centering_defect",
            "candidate_current_tv",
            "mixture_realized_tv",
            "mixture_realized_kl",
            "fresh_initialization_discrepancy",
            "value_of_memory",
        )
    }
    run_id = "s0-fixed-base-exact-persistent-s1257297357"
    policy_evaluations = []
    for variant in ("base", "candidate", "exact_mixture"):
        policy_evaluations.append(
            {
                "evaluation_id": f"{run_id}.interaction_matched.{variant}",
                "policy_variant": variant,
                "evaluation_pool_sha256": _available(HEX64),
                "evaluation_artifact_sha256": _available(HEX64),
                "per_instance_artifact_sha256": _available(HEX64),
                "primary": {
                    "solve_rate": _available(0.875),
                    "solved_count": _available(7),
                    "denominator": _available(8),
                },
                "secondary": {
                    "discounted_return_mean": _available(0.25),
                    "terminal_reason_counts": _available(
                        {"budget": 1, "solved": 7, "stop": 0}
                    ),
                    "edits_to_solve_mean": _available(4.0),
                    "value_calibration": _available(0.1),
                },
            }
        )
    return {
        "schema_name": "policy_improvement_v1",
        "schema_version": 3,
        "protocol_id": "policy-improvement-v1-20260814",
        "protocol_sha256": HEX64,
        "amendment_history_sha256": HEX64,
        "run_id": run_id,
        "registry_row_sha256": HEX64,
        "phase": "stage0_smoke",
        "tier": "smoke",
        "seed": 1257297357,
        "evaluation_split": "validation",
        "method_id": "fixed_base_exact_persistent",
        "base_method_id": "fixed_base_exact_persistent",
        "n": 2,
        "K": 1,
        "alpha": 0.1,
        "ablation_variant": None,
        "applied_config_override": {"inner_unroll_n": 2},
        "primary_policy_variant": "exact_mixture",
        "evaluation_snapshots": [
            {
                "snapshot_kind": "interaction_matched",
                "status": "available",
                "unavailable_reason": None,
                "target": {
                    "unit": "environment_interactions",
                    "registered_quantity": _available(32),
                },
                "observed_environment_interactions": _available(32),
                "observed_recurrent_map_applications": _available(256),
                "accelerator_seconds_observed": _available(2.0),
                "checkpoint_sha256": _available(HEX64),
                "model_state_sha256": _available(HEX64),
                "checkpoint_lineage_sha256": _available("e" * 64),
                "policy_evaluations": policy_evaluations,
            },
            {
                "snapshot_kind": "compute_matched",
                "status": "unavailable",
                "unavailable_reason": "compute_snapshot_not_registered",
                "target": {
                    "unit": "recurrent_map_applications",
                    "registered_quantity": _unavailable(
                        "compute_budget_pending_smoke_measurement"
                    ),
                },
                "observed_environment_interactions": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "observed_recurrent_map_applications": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "accelerator_seconds_observed": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "checkpoint_sha256": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "model_state_sha256": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "checkpoint_lineage_sha256": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "policy_evaluations": [],
            },
        ],
        "status": "complete",
        "failure": None,
        "identities": {
            "producer_git_commit": "b" * 40,
            "git_clean": True,
            "runtime_authorization_sha256": HEX64,
            "training_source_git_commit": "b" * 40,
            "training_runtime_sha256": HEX64,
            "training_runtime_profile_sha256": HEX64,
            "training_selected_source_manifest_sha256": HEX64,
            "launcher_sha256": HEX64,
            "producer_manifest_sha256": HEX64,
            "method_config_sha256": "2c101307f77ef788d588928b694b0ef01de186c3f9f1ae6f561e52d5e8f433a9",
            "effective_config_sha256": HEX64,
            "dataset_manifest_sha256": HEX64,
            "train_ordered_records_sha256": HEX64,
            "evaluation_ordered_records_sha256": HEX64,
            "initialization_sha256": HEX64,
            "checkpoint_sha256": _available(HEX64),
            "model_state_sha256": _available(HEX64),
            "evaluation_runtime_sha256": _available(HEX64),
            "evaluation_source_git_commit": _available("c" * 40),
            "evaluation_runtime_profile_sha256": _available(HEX64),
            "evaluation_selected_source_manifest_sha256": _available(HEX64),
            "evaluation_pool_sha256": _available(HEX64),
            "test_open_sha256": _unavailable("test_data_not_opened"),
            "device": "cuda:0",
        },
        "metrics": {
            "training": training,
            "diagnostics": diagnostics,
        },
        "artifacts": {
            "checkpoint": _available(HEX64),
            "checkpoint_validation": _available(HEX64),
            "model_state_inventory": _available(HEX64),
            "run_manifest": _available(HEX64),
        },
    }


def _policy_variants(method_id: str) -> tuple[str, ...]:
    return policy_variants_for_method(method_id)


def _validator_execution_identity(
    result: dict[str, object], *, validator: str
) -> dict[str, object]:
    identities = result["identities"]
    assert isinstance(identities, dict)
    common = {
        "runtime_authorization_sha256": identities["runtime_authorization_sha256"],
        "launcher_sha256": identities["launcher_sha256"],
    }
    if validator == "policy_improvement_smoke_runtime":
        return {
            "role": "policy-improvement-training",
            "source_git_commit": identities["training_source_git_commit"],
            "runtime_sha256": identities["training_runtime_sha256"],
            "runtime_profile_sha256": identities["training_runtime_profile_sha256"],
            "selected_source_manifest_sha256": identities[
                "training_selected_source_manifest_sha256"
            ],
            **common,
        }
    assert validator == "policy_improvement_checkpoint_validator"

    def available(name: str) -> object:
        value = identities[name]
        assert isinstance(value, dict) and value.get("status") == "available"
        return value["value"]

    return {
        "role": "policy-improvement-evaluation",
        "source_git_commit": available("evaluation_source_git_commit"),
        "runtime_sha256": available("evaluation_runtime_sha256"),
        "runtime_profile_sha256": available("evaluation_runtime_profile_sha256"),
        "selected_source_manifest_sha256": available(
            "evaluation_selected_source_manifest_sha256"
        ),
        **common,
    }


def _fixture_checkpoint_validator(
    request: dict[str, object],
) -> dict[str, object]:
    """Test-only semantic validator for synthetic non-Torch fixture bytes."""

    checkpoint = Path(str(request["checkpoint_path"]))
    generation = checkpoint.parents[1]
    candidates = sorted(generation.rglob("checkpoint_validation.json"))
    receipt = None
    for candidate in candidates:
        value = load_strict_json(candidate)
        if value.get("checkpoint_sha256") == request["checkpoint_sha256"]:
            receipt = value
            break
    if receipt is None:
        raise PolicyImprovementSchemaError(
            "Synthetic fixture has no checkpoint validation record."
        )
    protocol = request["protocol"]
    row = request["registry_row"]
    assert isinstance(protocol, dict) and isinstance(row, dict)
    base_method = str(row["base_method_id"])
    method = next(item for item in protocol["methods"] if item["id"] == base_method)
    dataset_manifest = protocol["dataset"]["manifest_sha256"]
    assert dataset_manifest["status"] == "available"
    return {
        "schema_name": "policy_improvement_checkpoint_semantic_validation_v1",
        "schema_version": 1,
        "run_id": request["run_id"],
        "method_id": request["method_id"],
        "seed": request["seed"],
        "snapshot_kind": request["snapshot_kind"],
        "environment_interactions": request["environment_interactions"],
        "parent_checkpoint_sha256": request["parent_checkpoint_sha256"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "initialization_sha256": request["initialization_sha256"],
        "model_state_sha256": receipt["model_state_sha256"],
        "role_state_sha256s": receipt["role_state_sha256s"],
        "method_config_sha256": method["config_sha256"],
        "registered_effective_config_sha256": row["expected_effective_config_sha256"],
        "effective_config_sha256": hashlib.sha256(
            canonical_json_bytes({"request": request["run_id"], "fixture": True})
        ).hexdigest(),
        "dataset_manifest_sha256": dataset_manifest["value"],
        "dataset_provenance_sha256": hashlib.sha256(
            canonical_json_bytes({"dataset": dataset_manifest["value"]})
        ).hexdigest(),
        "run_identity_sha256": (
            hashlib.sha256(canonical_json_bytes({"run": request["run_id"]})).hexdigest()
            if base_method.startswith("fixed_base_exact")
            else None
        ),
        "training_call_delta": 0,
        "evaluation_call_delta": 0,
        "optimizer_step_delta": 0,
    }


def _materialize_result_generations(
    protocol: dict[str, object],
    registry: dict[str, object],
    results: list[dict[str, object]],
    documents: dict[str, object],
    evidence_root: Path,
) -> None:
    """Write byte-authenticated fixture generations for audit tests."""

    evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(evidence_root, 0o700)
    rows = {str(row["run_id"]): row for row in registry["rows"]}
    registry_digest = registry_sha256(registry)
    for result in results:
        row = rows[str(result["run_id"])]
        tier = str(row["tier"])
        budget_tier = "confirmatory" if tier == "ablation" else tier
        budget = int(protocol["budgets"][budget_tier]["environment_interactions"])
        generation = (
            evidence_root
            / "runs"
            / str(result["run_id"])
            / "segments"
            / f"env_{budget:09d}"
        )
        (generation / "checkpoints").mkdir(parents=True)
        parent_checkpoint_sha: str | None = None
        if tier == "smoke":
            parent_generation = generation.parent / "env_000000016"
            (parent_generation / "checkpoints").mkdir(parents=True)
            parent_checkpoint = parent_generation / "checkpoints" / "prepare.pt"
            parent_payload = f"fixture parent checkpoint:{result['run_id']}\n".encode(
                "ascii"
            )
            parent_checkpoint.write_bytes(parent_payload)
            parent_checkpoint_sha = hashlib.sha256(parent_payload).hexdigest()
            parent_roles = {
                "model": hashlib.sha256(
                    f"{result['run_id']}.parent.model".encode("ascii")
                ).hexdigest()
            }
            _write_json_fixture(
                parent_generation / "checkpoint_validation.json",
                {
                    "schema_name": "policy_improvement_checkpoint_validation_v1",
                    "schema_version": 1,
                    "validator": "policy_improvement_smoke_runtime",
                    "validator_execution_identity": _validator_execution_identity(
                        result, validator="policy_improvement_smoke_runtime"
                    ),
                    "run_id": result["run_id"],
                    "method_id": result["method_id"],
                    "snapshot_kind": "interaction_matched",
                    "environment_interactions": 16,
                    "checkpoint_sha256": parent_checkpoint_sha,
                    "parent_checkpoint_sha256": None,
                    "model_state_sha256": hashlib.sha256(
                        canonical_json_bytes(parent_roles)
                    ).hexdigest(),
                    "role_state_sha256s": parent_roles,
                    "strict_resume_validated": True,
                },
            )
            parent_files = {
                path.relative_to(parent_generation).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in parent_generation.rglob("*")
                if path.is_file()
            }
            _write_json_fixture(
                parent_generation / "MANIFEST.json",
                {
                    "schema_name": "policy_improvement_smoke_segment_v1",
                    "schema_version": 1,
                    "protocol_sha256": result["protocol_sha256"],
                    "registry_sha256": registry_digest,
                    "registry_row_sha256": result["registry_row_sha256"],
                    "run_id": result["run_id"],
                    "method_id": result["method_id"],
                    "segment": "prepare",
                    "environment_interactions": 16,
                    "parent_checkpoint_sha256": None,
                    "result_status": None,
                    "outputs": {
                        "checkpoint": {
                            "path": "checkpoints/prepare.pt",
                            **parent_files["checkpoints/prepare.pt"],
                        },
                        "files": parent_files,
                    },
                    "storage_bytes": sum(
                        int(item["bytes"]) for item in parent_files.values()
                    ),
                },
            )
        checkpoint_by_digest: dict[str, str] = {}
        snapshot_bindings: list[dict[str, object]] = []
        role_hashes_by_kind: dict[str, dict[str, str]] = {}
        for snapshot in result["evaluation_snapshots"]:
            if snapshot["status"] != "available":
                continue
            kind = str(snapshot["snapshot_kind"])
            checkpoint_path = generation / "checkpoints" / f"{kind}.pt"
            checkpoint_payload = (
                f"fixture checkpoint:{result['run_id']}:{kind}\n".encode("ascii")
            )
            checkpoint_path.write_bytes(checkpoint_payload)
            checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
            snapshot["checkpoint_sha256"] = _available(checkpoint_digest)
            role_hashes = {
                "model": hashlib.sha256(
                    f"{result['run_id']}:{kind}:model-role".encode("ascii")
                ).hexdigest()
            }
            snapshot_model_sha256 = hashlib.sha256(
                canonical_json_bytes(role_hashes)
            ).hexdigest()
            snapshot["model_state_sha256"] = _available(snapshot_model_sha256)
            role_hashes_by_kind[kind] = role_hashes
            validator = (
                "policy_improvement_smoke_runtime"
                if tier == "smoke"
                else "policy_improvement_checkpoint_validator"
            )
            validation_sha256 = _write_json_fixture(
                generation
                / "checkpoint_validations"
                / kind
                / "checkpoint_validation.json",
                {
                    "schema_name": "policy_improvement_checkpoint_validation_v1",
                    "schema_version": 1,
                    "validator": validator,
                    "validator_execution_identity": _validator_execution_identity(
                        result, validator=validator
                    ),
                    "run_id": result["run_id"],
                    "method_id": result["method_id"],
                    "snapshot_kind": kind,
                    "environment_interactions": snapshot[
                        "observed_environment_interactions"
                    ]["value"],
                    "checkpoint_sha256": checkpoint_digest,
                    "parent_checkpoint_sha256": (
                        parent_checkpoint_sha if tier == "smoke" else None
                    ),
                    "model_state_sha256": snapshot_model_sha256,
                    "role_state_sha256s": role_hashes,
                    "strict_resume_validated": True,
                },
            )
            snapshot_bindings.append(
                {
                    "snapshot_kind": kind,
                    "checkpoint_sha256": checkpoint_digest,
                    "model_state_sha256": snapshot_model_sha256,
                    "checkpoint_validation_sha256": validation_sha256,
                }
            )
            checkpoint_by_digest[checkpoint_digest] = checkpoint_path.relative_to(
                generation
            ).as_posix()
            if kind == "interaction_matched":
                result["identities"]["checkpoint_sha256"] = _available(
                    checkpoint_digest
                )
                result["artifacts"]["checkpoint"] = _available(checkpoint_digest)
                result["artifacts"]["checkpoint_validation"] = _available(
                    validation_sha256
                )
                result["identities"]["model_state_sha256"] = _available(
                    snapshot_model_sha256
                )
            for evaluation in snapshot["policy_evaluations"]:
                variant = str(evaluation["policy_variant"])
                evaluation_root = generation / "evaluations" / kind
                aggregate = {
                    "primary": evaluation["primary"],
                    "secondary": evaluation["secondary"],
                    "diagnostic_details": {},
                }
                _write_json_fixture(evaluation_root / f"{variant}.json", aggregate)
                per_digest = str(evaluation["per_instance_artifact_sha256"]["value"])
                _write_json_fixture(
                    evaluation_root / f"{variant}.per_instance.json",
                    documents[per_digest],
                )
        interaction_checkpoint = str(result["identities"]["checkpoint_sha256"]["value"])
        model_state_sha = str(result["identities"]["model_state_sha256"]["value"])
        role_hashes = role_hashes_by_kind["interaction_matched"]
        model_inventory_digest = _write_json_fixture(
            generation / "model_state_inventory.json",
            {
                "schema_name": "policy_improvement_model_state_inventory_v1",
                "run_id": result["run_id"],
                "method_id": result["method_id"],
                "model_state_sha256": model_state_sha,
                "role_state_sha256s": role_hashes,
                "snapshot_state_bindings": snapshot_bindings,
            },
        )
        result["artifacts"]["model_state_inventory"] = _available(
            model_inventory_digest
        )
        run_manifest_digest = _write_json_fixture(
            generation / "RUN_MANIFEST.json",
            {
                "run_id": result["run_id"],
                "method_id": result["method_id"],
                "environment_interactions": budget,
                "protocol_sha256": result["protocol_sha256"],
                "registry_row_sha256": result["registry_row_sha256"],
                "checkpoint_sha256": interaction_checkpoint,
                "model_state_sha256": model_state_sha,
                "model_state_sha256s": role_hashes,
                "model_state_inventory_sha256": model_inventory_digest,
            },
        )
        result["artifacts"]["run_manifest"] = _available(run_manifest_digest)
        if tier == "smoke":
            assert parent_checkpoint_sha is not None
            lineage_sha = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema_name": (
                            "policy_improvement_smoke_checkpoint_lineage_v1"
                        ),
                        "run_id": result["run_id"],
                        "initialization_sha256": result["identities"][
                            "initialization_sha256"
                        ],
                        "parent_checkpoint_sha256": parent_checkpoint_sha,
                        "checkpoint_sha256": interaction_checkpoint,
                    }
                )
            ).hexdigest()
            for snapshot in result["evaluation_snapshots"]:
                if snapshot["status"] == "available":
                    snapshot["checkpoint_lineage_sha256"] = _available(lineage_sha)
        _write_json_fixture(generation / "result.json", result)
        files = {
            path.relative_to(generation).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in generation.rglob("*")
            if path.is_file()
        }
        manifest = {
            "schema_name": (
                "policy_improvement_smoke_segment_v1"
                if tier == "smoke"
                else "policy_improvement_run_segment_v1"
            ),
            "schema_version": 1,
            "protocol_sha256": result["protocol_sha256"],
            "registry_sha256": registry_digest,
            "registry_row_sha256": result["registry_row_sha256"],
            "run_id": result["run_id"],
            "method_id": result["method_id"],
            "segment": "resume" if tier == "smoke" else "complete",
            "environment_interactions": budget,
            "parent_checkpoint_sha256": parent_checkpoint_sha,
            "result_status": "complete",
            "outputs": {
                "checkpoint": {
                    "path": checkpoint_by_digest[interaction_checkpoint],
                    **files[checkpoint_by_digest[interaction_checkpoint]],
                },
                "files": files,
            },
            "storage_bytes": sum(int(item["bytes"]) for item in files.values()),
        }
        _write_json_fixture(generation / "MANIFEST.json", manifest)


def _write_json_fixture(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _materialize_failed_attempt(result: dict[str, object], evidence_root: Path) -> None:
    attempt_id = "0" * 32
    is_smoke = result["tier"] == "smoke"
    segment_name = "resume" if is_smoke else "complete"
    attempt = (
        evidence_root
        / "runs"
        / str(result["run_id"])
        / "attempts"
        / segment_name
        / attempt_id
    )
    result_digest = _write_json_fixture(attempt / "result.json", result)
    result_path = attempt / "result.json"
    manifest = {
        "schema_name": (
            "policy_improvement_smoke_failed_attempt_v1"
            if is_smoke
            else "policy_improvement_run_failed_attempt_v1"
        ),
        "schema_version": 1,
        "attempt_id": attempt_id,
        "protocol_sha256": result["protocol_sha256"],
        "registry_row_sha256": result["registry_row_sha256"],
        "runtime_authorization_sha256": result["identities"][
            "runtime_authorization_sha256"
        ],
        "run_id": result["run_id"],
        "segment": segment_name,
        "failure_phase": result["failure"]["phase"],
        "result": {
            "path": "result.json",
            "bytes": result_path.stat().st_size,
            "sha256": result_digest,
        },
    }
    if not is_smoke:
        manifest.update(
            {
                "phase": result["phase"],
                "tier": result["tier"],
                "environment_interactions": 80000,
            }
        )
    _write_json_fixture(attempt / "MANIFEST.json", manifest)


def _auditable_smoke_results(
    protocol: dict[str, object], registry: dict[str, object], evidence_root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    results = []
    documents: dict[str, object] = {}
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    amendment_digest = hashlib.sha256(canonical_json_bytes([])).hexdigest()
    puzzle_hashes = [_fixture_record_sha256("validation", index) for index in range(8)]
    pool_digest = hashlib.sha256(canonical_json_bytes(puzzle_hashes)).hexdigest()
    ordered_pool = hashlib.sha256()
    for puzzle_hash in puzzle_hashes:
        ordered_pool.update(puzzle_hash.encode("ascii"))
        ordered_pool.update(b"\n")
    ordered_pool_digest = ordered_pool.hexdigest()
    rows = [row for row in registry["rows"] if row["phase"] == "stage0_smoke"]
    authorization = _runtime_authorization(protocol)
    training_role = authorization["roles"][0]
    evaluation_role = authorization["roles"][1]
    for row in rows:
        result = _valid_result()
        for field in (
            "run_id",
            "phase",
            "tier",
            "seed",
            "evaluation_split",
            "method_id",
            "base_method_id",
            "n",
            "K",
            "alpha",
            "ablation_variant",
        ):
            result[field] = row[field]
        method_id = str(row["method_id"])
        method_registration = next(
            method
            for method in protocol["methods"]
            if method["id"] == row["base_method_id"]
        )
        result["identities"]["method_config_sha256"] = method_registration[
            "config_sha256"
        ]
        result["primary_policy_variant"] = primary_policy_variant_for_method(method_id)
        result["protocol_sha256"] = protocol_digest
        result["amendment_history_sha256"] = amendment_digest
        result["registry_row_sha256"] = hashlib.sha256(
            canonical_json_bytes(row)
        ).hexdigest()
        result["applied_config_override"] = row["config_override"]
        result["identities"]["effective_config_sha256"] = row[
            "expected_effective_config_sha256"
        ]
        result["identities"]["runtime_authorization_sha256"] = (
            runtime_authorization_sha256(authorization)
        )
        result["identities"]["producer_git_commit"] = authorization[
            "producer_git_commit"
        ]
        result["identities"]["producer_manifest_sha256"] = authorization[
            "producer_source_manifest_sha256"
        ]
        result["identities"]["launcher_sha256"] = authorization["launcher_sha256"]
        result["identities"]["training_runtime_sha256"] = training_role[
            "runtime_sha256"
        ]
        result["identities"]["training_source_git_commit"] = training_role[
            "source_git_commit"
        ]
        result["identities"]["training_runtime_profile_sha256"] = training_role[
            "runtime_profile_sha256"
        ]
        result["identities"]["training_selected_source_manifest_sha256"] = (
            training_role["selected_source_manifest_sha256"]
        )
        result["identities"]["evaluation_runtime_sha256"] = _available(
            evaluation_role["runtime_sha256"]
        )
        result["identities"]["evaluation_source_git_commit"] = _available(
            evaluation_role["source_git_commit"]
        )
        result["identities"]["evaluation_runtime_profile_sha256"] = _available(
            evaluation_role["runtime_profile_sha256"]
        )
        result["identities"]["evaluation_selected_source_manifest_sha256"] = _available(
            evaluation_role["selected_source_manifest_sha256"]
        )
        checkpoint_digest = hashlib.sha256(
            f"{row['run_id']}.checkpoint".encode("ascii")
        ).hexdigest()
        model_digest = hashlib.sha256(
            f"{row['run_id']}.model".encode("ascii")
        ).hexdigest()
        result["evaluation_snapshots"][0]["checkpoint_sha256"] = _available(
            checkpoint_digest
        )
        result["evaluation_snapshots"][0]["model_state_sha256"] = _available(
            model_digest
        )
        result["identities"]["checkpoint_sha256"] = _available(checkpoint_digest)
        result["identities"]["model_state_sha256"] = _available(model_digest)
        result["artifacts"]["checkpoint"] = _available(checkpoint_digest)
        result["identities"]["evaluation_pool_sha256"] = _available(pool_digest)
        result["identities"]["dataset_manifest_sha256"] = protocol["dataset"][
            "manifest_sha256"
        ]["value"]
        result["identities"]["train_ordered_records_sha256"] = protocol["dataset"][
            "splits"
        ]["train"]["ordered_record_sha256"]["value"]
        result["identities"]["evaluation_ordered_records_sha256"] = protocol["dataset"][
            "splits"
        ][row["evaluation_split"]]["ordered_record_sha256"]["value"]
        result["identities"]["evaluation_ordered_records_sha256"] = ordered_pool_digest
        evaluations = []
        for variant in _policy_variants(method_id):
            evaluation_id = f"{row['run_id']}.interaction_matched.{variant}"
            primary = {
                "solve_rate": _available(0.875),
                "solved_count": _available(7),
                "denominator": _available(8),
            }
            secondary = {
                "discounted_return_mean": _available(0.25),
                "terminal_reason_counts": _available(
                    {"budget": 1, "solved": 7, "stop": 0}
                ),
                "edits_to_solve_mean": _available(4.0),
                "value_calibration": _available(0.1),
            }
            records = [
                {
                    "registered_index": index,
                    "registered_record_sha256": _fixture_record_sha256(
                        "validation", index
                    ),
                    "puzzle_id": f"validation-{index:06d}",
                    "puzzle_sha256": _fixture_input_sha256("validation", index),
                    "solved": index != 0,
                    "discounted_return": 0.25,
                    "terminal_reason": "budget" if index == 0 else "solved",
                    "edits_to_solve": None if index == 0 else 4,
                    "value_prediction": 0.1,
                    "realized_return": 0.2,
                }
                for index in range(8)
            ]
            document = {
                "schema_name": "policy_improvement_instances_v1",
                "schema_version": 2,
                "protocol_id": protocol["protocol_id"],
                "run_id": row["run_id"],
                "phase": row["phase"],
                "tier": row["tier"],
                "seed": row["seed"],
                "evaluation_split": row["evaluation_split"],
                "method_id": method_id,
                "snapshot_kind": "interaction_matched",
                "evaluation_id": evaluation_id,
                "policy_variant": variant,
                "evaluation_pool_sha256": pool_digest,
                "records": records,
            }
            per_instance_digest = hashlib.sha256(
                canonical_json_bytes(document)
            ).hexdigest()
            documents[per_instance_digest] = document
            evaluations.append(
                {
                    "evaluation_id": evaluation_id,
                    "policy_variant": variant,
                    "evaluation_pool_sha256": _available(pool_digest),
                    "evaluation_artifact_sha256": _available(
                        hashlib.sha256(
                            canonical_json_bytes(
                                {"primary": primary, "secondary": secondary}
                            )
                        ).hexdigest()
                    ),
                    "per_instance_artifact_sha256": _available(per_instance_digest),
                    "primary": primary,
                    "secondary": secondary,
                }
            )
        result["evaluation_snapshots"][0]["policy_evaluations"] = evaluations
        results.append(result)
    _materialize_result_generations(
        protocol, registry, results, documents, evidence_root
    )
    return results, documents


def _auditable_phase_results(
    protocol: dict[str, object],
    registry: dict[str, object],
    history: list[dict[str, object]],
    authorization: dict[str, object],
    phase: str,
    evidence_root: Path,
    test_open_sha256: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build schema-complete deterministic evidence for one non-smoke phase."""

    tier = "confirmatory" if phase == "stage2_confirmatory" else "pilot"
    population_tier = tier
    population = protocol["evaluation_populations"][population_tier]
    split = str(population["split"])
    count = int(population["count"])
    record_hashes = [_fixture_record_sha256(split, index) for index in range(count)]
    pool_digest = hashlib.sha256(canonical_json_bytes(record_hashes)).hexdigest()
    ordered_digest = _ordered_digest(record_hashes)
    training_role = authorization["roles"][0]
    evaluation_role = authorization["roles"][1]
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    history_digest = amendment_history_sha256(history)
    rows = [row for row in registry["rows"] if row["phase"] == phase]
    results: list[dict[str, object]] = []
    documents: dict[str, object] = {}

    def solved_for(row: dict[str, object], index: int) -> bool:
        method = str(row["method_id"])
        if phase == "stage1_screen" and method.startswith("fixed_base_exact_"):
            score = (
                7
                if method == "fixed_base_exact_persistent"
                and row["n"] == 2
                and row["K"] == 1
                else 5
            )
            return index % 10 < score
        if phase == "stage1_alpha":
            score = {0.05: 6, 0.1: 8, 0.2: 7}[float(row["alpha"])]
            return index % 10 < score
        if method == "fixed_base_exact_persistent":
            return index % 4 != 0
        if method == "fixed_base_exact_episodic":
            return index % 3 != 0
        return index % 2 == 0

    for row in rows:
        result = _valid_result()
        for field in (
            "run_id",
            "phase",
            "tier",
            "seed",
            "evaluation_split",
            "method_id",
            "base_method_id",
            "n",
            "K",
            "alpha",
            "ablation_variant",
        ):
            result[field] = row[field]
        method = str(row["method_id"])
        result["primary_policy_variant"] = primary_policy_variant_for_method(method)
        result["protocol_sha256"] = protocol_digest
        result["amendment_history_sha256"] = history_digest
        result["registry_row_sha256"] = hashlib.sha256(
            canonical_json_bytes(row)
        ).hexdigest()
        result["applied_config_override"] = row["config_override"]
        identities = result["identities"]
        identities["runtime_authorization_sha256"] = runtime_authorization_sha256(
            authorization
        )
        identities["producer_git_commit"] = authorization["producer_git_commit"]
        identities["producer_manifest_sha256"] = authorization[
            "producer_source_manifest_sha256"
        ]
        identities["launcher_sha256"] = authorization["launcher_sha256"]
        identities["training_source_git_commit"] = training_role["source_git_commit"]
        identities["training_runtime_sha256"] = training_role["runtime_sha256"]
        identities["training_runtime_profile_sha256"] = training_role[
            "runtime_profile_sha256"
        ]
        identities["training_selected_source_manifest_sha256"] = training_role[
            "selected_source_manifest_sha256"
        ]
        identities["evaluation_runtime_sha256"] = _available(
            evaluation_role["runtime_sha256"]
        )
        identities["evaluation_source_git_commit"] = _available(
            evaluation_role["source_git_commit"]
        )
        identities["evaluation_runtime_profile_sha256"] = _available(
            evaluation_role["runtime_profile_sha256"]
        )
        identities["evaluation_selected_source_manifest_sha256"] = _available(
            evaluation_role["selected_source_manifest_sha256"]
        )
        registration = next(
            item for item in protocol["methods"] if item["id"] == row["base_method_id"]
        )
        identities["method_config_sha256"] = registration["config_sha256"]
        identities["effective_config_sha256"] = row["expected_effective_config_sha256"]
        identities["dataset_manifest_sha256"] = protocol["dataset"]["manifest_sha256"][
            "value"
        ]
        identities["train_ordered_records_sha256"] = protocol["dataset"]["splits"][
            "train"
        ]["ordered_record_sha256"]["value"]
        identities["evaluation_ordered_records_sha256"] = ordered_digest
        identities["evaluation_pool_sha256"] = _available(pool_digest)
        identities["test_open_sha256"] = (
            _available(test_open_sha256)
            if str(row["evaluation_split"]) == "test" and test_open_sha256 is not None
            else _unavailable("test_data_not_opened")
        )
        lineage = hashlib.sha256(f"{row['run_id']}.lineage".encode("ascii")).hexdigest()
        interaction_checkpoint = hashlib.sha256(
            f"{row['run_id']}.interaction.checkpoint".encode("ascii")
        ).hexdigest()
        interaction_model = hashlib.sha256(
            f"{row['run_id']}.interaction.model".encode("ascii")
        ).hexdigest()
        identities["checkpoint_sha256"] = _available(interaction_checkpoint)
        identities["model_state_sha256"] = _available(interaction_model)
        result["artifacts"]["checkpoint"] = _available(interaction_checkpoint)
        budget = int(protocol["budgets"][tier]["environment_interactions"])
        compute_target = int(history[1]["common_compute_targets"][tier])
        snapshots = []
        for snapshot_kind in ("interaction_matched", "compute_matched"):
            interaction = snapshot_kind == "interaction_matched"
            checkpoint = (
                interaction_checkpoint
                if interaction
                else hashlib.sha256(
                    f"{row['run_id']}.compute.checkpoint".encode("ascii")
                ).hexdigest()
            )
            model = (
                interaction_model
                if interaction
                else hashlib.sha256(
                    f"{row['run_id']}.compute.model".encode("ascii")
                ).hexdigest()
            )
            evaluations = []
            for variant in _policy_variants(method):
                records = []
                for index, record_sha in enumerate(record_hashes):
                    solved = solved_for(row, index)
                    records.append(
                        {
                            "registered_index": index,
                            "registered_record_sha256": record_sha,
                            "puzzle_id": f"{split}-{index:06d}",
                            "puzzle_sha256": _fixture_input_sha256(split, index),
                            "solved": solved,
                            "discounted_return": 1.0 if solved else 0.0,
                            "terminal_reason": "solved" if solved else "budget",
                            "edits_to_solve": 4 if solved else None,
                            "value_prediction": 0.5,
                            "realized_return": 1.0 if solved else 0.0,
                        }
                    )
                solved_count = sum(1 for record in records if record["solved"])
                primary = {
                    "solve_rate": _available(solved_count / count),
                    "solved_count": _available(solved_count),
                    "denominator": _available(count),
                }
                secondary = {
                    "discounted_return_mean": _available(solved_count / count),
                    "terminal_reason_counts": _available(
                        {
                            "budget": count - solved_count,
                            "solved": solved_count,
                            "stop": 0,
                        }
                    ),
                    "edits_to_solve_mean": _available(4.0),
                    "value_calibration": _available(0.5),
                }
                evaluation_id = f"{row['run_id']}.{snapshot_kind}.{variant}"
                document = {
                    "schema_name": "policy_improvement_instances_v1",
                    "schema_version": 2,
                    "protocol_id": protocol["protocol_id"],
                    "run_id": row["run_id"],
                    "phase": phase,
                    "tier": row["tier"],
                    "seed": row["seed"],
                    "evaluation_split": split,
                    "method_id": method,
                    "snapshot_kind": snapshot_kind,
                    "evaluation_id": evaluation_id,
                    "policy_variant": variant,
                    "evaluation_pool_sha256": pool_digest,
                    "records": records,
                }
                document_digest = hashlib.sha256(
                    canonical_json_bytes(document)
                ).hexdigest()
                documents[document_digest] = document
                evaluations.append(
                    {
                        "evaluation_id": evaluation_id,
                        "policy_variant": variant,
                        "evaluation_pool_sha256": _available(pool_digest),
                        "evaluation_artifact_sha256": _available(
                            hashlib.sha256(
                                canonical_json_bytes(
                                    {"primary": primary, "secondary": secondary}
                                )
                            ).hexdigest()
                        ),
                        "per_instance_artifact_sha256": _available(document_digest),
                        "primary": primary,
                        "secondary": secondary,
                    }
                )
            snapshots.append(
                {
                    "snapshot_kind": snapshot_kind,
                    "status": "available",
                    "unavailable_reason": None,
                    "target": {
                        "unit": (
                            "environment_interactions"
                            if interaction
                            else "recurrent_map_applications"
                        ),
                        "registered_quantity": _available(
                            budget if interaction else compute_target
                        ),
                    },
                    "observed_environment_interactions": _available(budget),
                    "observed_recurrent_map_applications": _available(
                        compute_target + 1 if interaction else compute_target
                    ),
                    "accelerator_seconds_observed": _available(2.0),
                    "checkpoint_sha256": _available(checkpoint),
                    "model_state_sha256": _available(model),
                    "checkpoint_lineage_sha256": _available(lineage),
                    "policy_evaluations": evaluations,
                }
            )
        result["evaluation_snapshots"] = snapshots
        results.append(result)
    _materialize_result_generations(
        protocol, registry, results, documents, evidence_root
    )
    return results, documents


class ProtocolAndRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
        self.theory = _theory_amendment(self.protocol)
        self.registry = generate_registry(self.protocol)

    def test_protocol_and_deterministic_seed_derivation(self) -> None:
        self.assertEqual(
            self.protocol["dataset"]["root"],
            "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1",
        )
        namespace = "upi-trm-policy-improvement-v1"
        self.assertEqual(derive_seed(namespace, "smoke", 0), 1257297357)
        self.assertEqual(
            [derive_seed(namespace, "pilot", index) for index in range(3)],
            [784831257, 2087907586, 4056782312],
        )
        self.assertEqual(
            [derive_seed(namespace, "confirmatory", index) for index in range(8)],
            [
                2081976412,
                781025396,
                1148619853,
                913535362,
                2143253519,
                2279379379,
                402312153,
                2736405725,
            ],
        )

    def test_registry_counts_ids_and_hash_are_deterministic(self) -> None:
        self.assertEqual(len(self.registry["rows"]), 157)
        self.assertEqual(self.registry["counts"]["smoke_rows"], 4)
        self.assertEqual(self.registry["counts"]["full_rows"], 153)
        self.assertEqual(self.registry["counts"]["stage1_screen"], 48)
        self.assertEqual(self.registry["counts"]["stage1_alpha"], 9)
        self.assertEqual(self.registry["counts"]["stage2_confirmatory"], 32)
        self.assertEqual(self.registry["counts"]["stage3_ablation"], 64)
        run_ids = [row["run_id"] for row in self.registry["rows"]]
        self.assertEqual(len(run_ids), len(set(run_ids)))
        self.assertEqual(
            registry_sha256(self.registry),
            registry_sha256(generate_registry(copy.deepcopy(self.protocol))),
        )
        self.assertEqual(
            REGISTRY_PATH.read_bytes(),
            canonical_json_bytes(self.registry) + b"\n",
        )
        validate_registry_document(
            load_strict_json(REGISTRY_PATH),
            self.protocol,
            [],
        )

    def test_staged_freezes_materialize_only_the_next_registered_stage(self) -> None:
        history, authorization = _amendment_history(self.protocol)
        validate_runtime_authorization(authorization)
        validate_compute_freeze(history[1])
        validate_screen_selection(history[2])
        validate_final_selection(history[3])
        validate_amendment_history(history, protocol=self.protocol)
        after_compute = generate_registry(self.protocol, history[:2])
        self.assertTrue(
            all(
                row["row_kind"] == "concrete"
                for row in after_compute["rows"]
                if row["phase"] == "stage1_screen"
            )
        )
        self.assertTrue(
            all(
                row["row_kind"] == "selection_template"
                for row in after_compute["rows"]
                if row["phase"]
                in {
                    "stage1_alpha",
                    "stage2_confirmatory",
                    "stage3_ablation",
                }
            )
        )
        after_screen = generate_registry(self.protocol, history[:3])
        alpha_rows = [
            row for row in after_screen["rows"] if row["phase"] == "stage1_alpha"
        ]
        self.assertTrue(all(row["row_kind"] == "concrete" for row in alpha_rows))
        self.assertEqual(
            {row["method_id"] for row in alpha_rows},
            {"fixed_base_exact_persistent"},
        )
        self.assertEqual({row["n"] for row in alpha_rows}, {2})
        self.assertEqual({row["K"] for row in alpha_rows}, {1})
        self.assertTrue(all("-n2-k1-a" in row["run_id"] for row in alpha_rows))
        self.assertTrue(
            all(
                row["row_kind"] == "selection_template"
                for row in after_screen["rows"]
                if row["phase"] in {"stage2_confirmatory", "stage3_ablation"}
            )
        )
        base_configs = load_registered_base_configs(self.protocol, REPOSITORY_ROOT)
        materialized = generate_registry(
            self.protocol, history, base_configs=base_configs
        )
        self.assertTrue(
            all(row["row_kind"] == "concrete" for row in materialized["rows"])
        )
        stage3 = [
            row for row in materialized["rows"] if row["phase"] == "stage3_ablation"
        ]
        self.assertTrue(all(row["config_override"] for row in stage3))
        self.assertTrue(all(row["expected_effective_config_sha256"] for row in stage3))

        opened = copy.deepcopy(history[2])
        opened["test_data_opened"] = True
        wrong_source = copy.deepcopy(history[2])
        wrong_source["source_registry_sha256"] = "d" * 64
        wrong_prefix = copy.deepcopy(history)
        wrong_prefix[3]["prior_amendment_history_sha256"] = "e" * 64
        wrong_profile = copy.deepcopy(authorization)
        wrong_profile["roles"][0]["selected_source_manifest_sha256"] = "f" * 64
        wrong_training_source = copy.deepcopy(authorization)
        wrong_training_source["roles"][0]["source_git_commit"] = "f" * 40
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_screen_selection(opened)
        with self.assertRaises(PolicyImprovementSchemaError):
            generate_registry(self.protocol, [*history[:2], wrong_source])
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_amendment_history(wrong_prefix, protocol=self.protocol)
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_runtime_authorization(wrong_profile)
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_runtime_authorization(wrong_training_source)

    def test_compute_freeze_rejects_failed_stage0_rows(self) -> None:
        history, _ = _amendment_history(self.protocol)
        failed = copy.deepcopy(history[1])
        failed["evidence"]["complete_rows"] = 0
        failed["evidence"]["failed_rows"] = 4
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "four complete Stage-0 rows",
        ):
            validate_compute_freeze(failed)

    def test_extended_runtime_authorization_is_strict_and_complete(self) -> None:
        authorization = _runtime_authorization(self.protocol)
        authorization["schema_name"] = "policy_improvement_runtime_authorization_v2"
        authorization["schema_version"] = 2
        authorization["roles"].extend(
            [
                {
                    **authorization["roles"][0],
                    "role": "policy-improvement-full",
                },
                {
                    **authorization["roles"][1],
                    "role": "policy-improvement-theory-bridge",
                },
            ]
        )
        validate_runtime_authorization(authorization)

        missing = copy.deepcopy(authorization)
        missing["roles"].pop()
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_runtime_authorization(missing)

        mixed = copy.deepcopy(authorization)
        mixed["roles"][-1]["source_git_commit"] = "f" * 40
        mixed["roles"][-1]["role"] = "policy-improvement-full"
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_runtime_authorization(mixed)

        mismatched_full = copy.deepcopy(authorization)
        mismatched_full["roles"][4]["runtime_sha256"] = "e" * 64
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "full-runtime and training",
        ):
            validate_runtime_authorization(mismatched_full)

        mismatched_theory = copy.deepcopy(authorization)
        mismatched_theory["roles"][5]["runtime_profile_sha256"] = "e" * 64
        mismatched_theory["roles"][5]["selected_source_manifest_sha256"] = "e" * 64
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "theory-bridge and evaluation",
        ):
            validate_runtime_authorization(mismatched_theory)

    def test_stage3_override_or_base_config_drift_fails_closed(self) -> None:
        history, _ = _amendment_history(self.protocol)
        bad_override = copy.deepcopy(history)
        bad_override[3]["stage3_variants"][0]["override_payload"] = {
            "batch_centered_advantage": True
        }
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_amendment_history(bad_override, protocol=self.protocol)
        configs = load_registered_base_configs(self.protocol, REPOSITORY_ROOT)
        bad_configs = copy.deepcopy(configs)
        bad_configs["fixed_base_exact_persistent"]["K"] = 99
        with self.assertRaises(PolicyImprovementSchemaError):
            generate_registry(self.protocol, history, base_configs=bad_configs)

    def test_all_materialized_stage3_variants_construct_registered_trainers(
        self,
    ) -> None:
        import gc

        import torch

        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
        from rl.config import RLConfig
        from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
        from rl.sudoku_checkers import dummy_checker
        from rl.training_setup import DummyPuzzleDataset
        from rl.upi_trm_trainer import UPITrmTrainer

        history, _ = _amendment_history(self.protocol)
        base_configs = load_registered_base_configs(self.protocol, REPOSITORY_ROOT)
        materialized = generate_registry(
            self.protocol, history, base_configs=base_configs
        )
        stage3_rows = {
            str(row["ablation_variant"]): row
            for row in materialized["rows"]
            if row["phase"] == "stage3_ablation"
        }
        self.assertEqual(len(stage3_rows), 8)

        non_theorem = {
            "batch_only_centering": (
                "fixed_base_batch_only_centering",
                "fixed_base_ablation_batch_only_centering",
                "exact_mixture",
            ),
            "distilled_realization": (
                "fixed_base_distilled_realization",
                "fixed_base_ablation_distilled_realization",
                "realized_policy",
            ),
        }
        for variant, row in stage3_rows.items():
            with self.subTest(variant=variant):
                base_method_id = str(row["base_method_id"])
                effective = dict(base_configs[base_method_id])
                effective.update(row["config_override"])
                config = RLConfig(**effective)

                dataset = DummyPuzzleDataset(
                    num_instances=2,
                    seq_len=16,
                    vocab_size=5,
                )
                env_config = PlanEditEnvConfig(
                    max_edits=config.max_edits,
                    gamma=config.gamma,
                    reward_shaping=config.reward_shaping,
                    vocab_size=dataset.vocab_size,
                    stop_action_mode=config.stop_action_mode,
                )
                env = PlanEditEnv(dataset, dummy_checker, env_config)
                env.set_stop_action_id(dataset.seq_len * dataset.vocab_size)
                model = TinyRecursiveReasoningModel_ACTV1(
                    {
                        "batch_size": 2,
                        "seq_len": dataset.seq_len,
                        "puzzle_emb_ndim": 0,
                        "num_puzzle_identifiers": dataset.num_identifiers,
                        "vocab_size": dataset.vocab_size,
                        "H_cycles": 1,
                        "L_cycles": 1,
                        "H_layers": 0,
                        "L_layers": 1,
                        "hidden_size": 16,
                        "expansion": 2.0,
                        "num_heads": 2,
                        "pos_encodings": "rope",
                        "rms_norm_eps": 1e-5,
                        "rope_theta": 10000.0,
                        "halt_max_steps": 2,
                        "halt_exploration_prob": 0.0,
                        "forward_dtype": "float32",
                        "mlp_t": False,
                        "puzzle_emb_len": 0,
                        "no_ACT_continue": True,
                        "rl_enable_value_head": True,
                        "rl_enable_policy_head": True,
                        "rl_num_actions": (dataset.seq_len * dataset.vocab_size + 1),
                        "rl_enable_contraction": config.enable_contraction,
                        "rl_target_Lz": config.target_Lz,
                        "rl_target_Lv": config.target_Lv,
                        "rl_disable_value_head_norm": (config.disable_value_head_norm),
                        "rl_latent_projection_mode": (config.latent_projection_mode),
                        "rl_latent_ball_radius": config.latent_ball_radius,
                    }
                )
                trainer = UPITrmTrainer(
                    model=model,
                    env=env,
                    rl_cfg=config,
                    device=torch.device("cpu"),
                )

                self.assertEqual(row["base_method_id"], "fixed_base_exact_persistent")
                self.assertTrue(trainer._fixed_base_ownership)
                if variant in non_theorem:
                    method_id, protocol_id, primary_policy = non_theorem[variant]
                    self.assertEqual(row["method_id"], method_id)
                    self.assertEqual(config.training_protocol, protocol_id)
                    self.assertEqual(
                        primary_policy_variant_for_method(method_id), primary_policy
                    )
                    self.assertFalse(config.is_fixed_base_proposal_exact())
                    self.assertTrue(config.is_registered_fixed_base_ablation())
                    self.assertFalse(
                        config.validate_theory_alignment(warn=False)["theory_aligned"]
                    )
                    self.assertFalse(trainer.replay.theorem_facing)
                else:
                    self.assertEqual(row["method_id"], "fixed_base_exact_persistent")
                    self.assertEqual(config.training_protocol, "fixed_base_exact")
                    self.assertTrue(config.is_fixed_base_proposal_exact())
                    self.assertFalse(config.is_registered_fixed_base_ablation())
                    self.assertTrue(
                        config.validate_theory_alignment(warn=False)["theory_aligned"]
                    )
                    self.assertTrue(trainer.replay.theorem_facing)
                if variant == "contraction_enabled":
                    self.assertTrue(config.enable_contraction)
                    self.assertEqual(config.opnorm_clamp_interval, 0)
                    self.assertTrue(config.disable_value_head_norm)
                del trainer, model, env
                gc.collect()

    def test_method_tuples_and_row_contracts_are_exact(self) -> None:
        for method in self.protocol["methods"]:
            self.assertEqual(
                hashlib.sha256(
                    (REPOSITORY_ROOT / method["config_path"]).read_bytes()
                ).hexdigest(),
                method["config_sha256"],
            )
        bad_method = copy.deepcopy(self.protocol)
        bad_method["methods"][0]["latent_mode"] = "episodic"
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_protocol(bad_method)
        bad_row = copy.deepcopy(self.registry)
        bad_row["rows"][0]["tier"] = "pilot"
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_registry_document(bad_row, self.protocol)

    def test_split_isolation_templates_and_ppo_applicability(self) -> None:
        for row in self.registry["rows"]:
            if row["tier"] in {"smoke", "pilot"}:
                self.assertEqual(row["evaluation_split"], "validation")
                self.assertFalse(row["paper_evidence_eligible"])
            else:
                self.assertEqual(row["evaluation_split"], "test")
            if row["method_id"] == "matched_ppo":
                self.assertTrue(row["factor_applicability"]["n"])
                self.assertFalse(row["factor_applicability"]["K"])
                self.assertFalse(row["factor_applicability"]["alpha"])
        templates = [
            row
            for row in self.registry["rows"]
            if row["row_kind"] == "selection_template"
        ]
        self.assertEqual(len(templates), 105)
        self.assertTrue(all(row["selection_rule"] for row in templates))

    def test_protocol_rejects_extra_fields_and_seed_drift(self) -> None:
        extra = copy.deepcopy(self.protocol)
        extra["unregistered"] = True
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_protocol(extra)
        drift = copy.deepcopy(self.protocol)
        drift["seeds"]["pilot"][0] += 1
        with self.assertRaises(PolicyImprovementSchemaError):
            generate_registry(drift)
        unsafe_owner = copy.deepcopy(self.protocol)
        unsafe_owner["dataset"]["root"] = "data/policy-improvement-hard-4x4-v1"
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError, "private-owner child"
        ):
            validate_protocol(unsafe_owner)


class ResultSchemaTest(unittest.TestCase):
    def test_complete_result_is_strict_and_consistent(self) -> None:
        result = _valid_result()
        self.assertEqual(validate_result(result)["status"], "complete")
        self.assertEqual(
            canonical_json_bytes(result), canonical_json_bytes(copy.deepcopy(result))
        )

    def test_stage3_methods_bind_primary_policy_and_evaluation_inventory(self) -> None:
        def stage3_result(method_id: str, variant: str) -> dict[str, Any]:
            result = _valid_result()
            result["phase"] = "stage3_ablation"
            result["tier"] = "ablation"
            result["evaluation_split"] = "test"
            result["identities"]["test_open_sha256"] = _available("d" * 64)
            result["method_id"] = method_id
            result["base_method_id"] = "fixed_base_exact_persistent"
            result["ablation_variant"] = variant
            result["primary_policy_variant"] = primary_policy_variant_for_method(
                method_id
            )
            templates = {
                evaluation["policy_variant"]: evaluation
                for evaluation in result["evaluation_snapshots"][0][
                    "policy_evaluations"
                ]
            }

            def evaluations(snapshot_kind: str) -> list[dict[str, Any]]:
                rows = []
                for policy_variant in policy_variants_for_method(method_id):
                    template = copy.deepcopy(
                        templates.get(policy_variant, templates["exact_mixture"])
                    )
                    template["policy_variant"] = policy_variant
                    template["evaluation_id"] = (
                        f"{result['run_id']}.{snapshot_kind}.{policy_variant}"
                    )
                    rows.append(template)
                return rows

            interaction = result["evaluation_snapshots"][0]
            interaction["policy_evaluations"] = evaluations("interaction_matched")
            compute = copy.deepcopy(interaction)
            compute["snapshot_kind"] = "compute_matched"
            compute["target"] = {
                "unit": "recurrent_map_applications",
                "registered_quantity": _available(256),
            }
            compute["policy_evaluations"] = evaluations("compute_matched")
            result["evaluation_snapshots"] = [interaction, compute]
            return result

        cases = (
            (
                "fixed_base_batch_only_centering",
                "batch_only_centering",
                "exact_mixture",
                ("base", "candidate", "exact_mixture"),
            ),
            (
                "fixed_base_distilled_realization",
                "distilled_realization",
                "realized_policy",
                ("base", "candidate", "realized_policy"),
            ),
        )
        for method_id, variant, primary, inventory in cases:
            with self.subTest(method_id=method_id):
                result = stage3_result(method_id, variant)
                checked = validate_result(result)
                self.assertEqual(checked["primary_policy_variant"], primary)
                self.assertEqual(
                    tuple(
                        item["policy_variant"]
                        for item in checked["evaluation_snapshots"][0][
                            "policy_evaluations"
                        ]
                    ),
                    inventory,
                )
                wrong = copy.deepcopy(result)
                wrong["primary_policy_variant"] = (
                    "realized_policy" if primary == "exact_mixture" else "exact_mixture"
                )
                with self.assertRaises(PolicyImprovementSchemaError):
                    validate_result(wrong)

    def test_zero_solve_result_uses_unavailable_not_numeric_placeholder(self) -> None:
        result = _valid_result()
        for evaluation in result["evaluation_snapshots"][0]["policy_evaluations"]:
            evaluation["primary"] = {
                "solve_rate": _available(0.0),
                "solved_count": _available(0),
                "denominator": _available(8),
            }
            evaluation["secondary"]["edits_to_solve_mean"] = _unavailable(
                "not_applicable"
            )
            evaluation["secondary"]["value_calibration"] = _unavailable()
            evaluation["secondary"]["terminal_reason_counts"] = _available(
                {"budget": 8, "solved": 0, "stop": 0}
            )
        self.assertEqual(validate_result(result)["status"], "complete")

    def test_rejects_missing_extra_nonfinite_placeholder_and_budget_drift(self) -> None:
        mutations = []
        missing = _valid_result()
        missing.pop("artifacts")
        mutations.append(missing)
        extra = _valid_result()
        extra["extra"] = 1
        mutations.append(extra)
        nonfinite = _valid_result()
        nonfinite["metrics"]["training"]["value_loss"] = _available(math.nan)
        mutations.append(nonfinite)
        placeholder = _valid_result()
        placeholder["identities"]["device"] = "TBD"
        mutations.append(placeholder)
        drift = _valid_result()
        drift["evaluation_snapshots"][0]["observed_environment_interactions"] = (
            _available(31)
        )
        mutations.append(drift)
        bad_rate = _valid_result()
        bad_rate["evaluation_snapshots"][0]["policy_evaluations"][2]["primary"][
            "solve_rate"
        ] = _available(0.9)
        mutations.append(bad_rate)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(PolicyImprovementSchemaError):
                    validate_result(mutation)

    def test_failed_result_cannot_publish_primary_endpoint(self) -> None:
        result = _valid_result()
        result["status"] = "failed"
        result["failure"] = {
            "phase": "training",
            "error_class": "runtime_error",
            "message_sha256": HEX64,
        }
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_result(result)

    def test_policy_evaluation_inventory_and_ids_are_exact(self) -> None:
        missing_candidate = _valid_result()
        missing_candidate["evaluation_snapshots"][0]["policy_evaluations"].pop(1)
        duplicate_id = _valid_result()
        duplicate_id["evaluation_snapshots"][0]["policy_evaluations"][1][
            "evaluation_id"
        ] = duplicate_id["evaluation_snapshots"][0]["policy_evaluations"][0][
            "evaluation_id"
        ]
        wrong_primary = _valid_result()
        wrong_primary["primary_policy_variant"] = "realized_policy"
        for result in (missing_candidate, duplicate_id, wrong_primary):
            with self.assertRaises(PolicyImprovementSchemaError):
                validate_result(result)

    def test_compute_and_interaction_snapshots_are_distinct_and_consistent(
        self,
    ) -> None:
        duplicate_kind = _valid_result()
        duplicate_kind["evaluation_snapshots"][1][
            "snapshot_kind"
        ] = "interaction_matched"
        smoke_claims_prior_compute_budget = _valid_result()
        smoke_claims_prior_compute_budget["evaluation_snapshots"][1] = copy.deepcopy(
            smoke_claims_prior_compute_budget["evaluation_snapshots"][0]
        )
        smoke_claims_prior_compute_budget["evaluation_snapshots"][1][
            "snapshot_kind"
        ] = "compute_matched"
        smoke_claims_prior_compute_budget["evaluation_snapshots"][1]["target"] = {
            "unit": "recurrent_map_applications",
            "registered_quantity": _available(256),
        }
        for result in (duplicate_kind, smoke_claims_prior_compute_budget):
            with self.assertRaises(PolicyImprovementSchemaError):
                validate_result(result)

    def test_pilot_result_cannot_use_test_split(self) -> None:
        result = _valid_result()
        result["phase"] = "stage1_screen"
        result["tier"] = "pilot"
        result["evaluation_split"] = "test"
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_result(result)

    def test_complete_result_requires_bounded_core_instrumentation(self) -> None:
        unavailable = _valid_result()
        unavailable["metrics"]["training"]["optimizer_steps"] = _unavailable()
        bad_gpu = _valid_result()
        bad_gpu["metrics"]["training"]["gpu_utilization_fraction"] = _available(1.01)
        too_few_gpu_samples = _valid_result()
        too_few_gpu_samples["metrics"]["training"]["gpu_utilization_sample_count"] = (
            _available(1)
        )
        bad_memory = _valid_result()
        bad_memory["metrics"]["training"]["peak_reserved_memory_bytes"] = _available(
            512
        )
        zero_actions = _valid_result()
        zero_actions["metrics"]["training"]["actions_processed"] = _available(0)
        for result in (
            unavailable,
            bad_gpu,
            too_few_gpu_samples,
            bad_memory,
            zero_actions,
        ):
            with self.assertRaises(PolicyImprovementSchemaError):
                validate_result(result)

        cpu = _valid_result()
        cpu["identities"]["device"] = "cpu"
        for field in (
            "gpu_utilization_fraction",
            "gpu_utilization_sample_count",
            "gpu_utilization_sampling_interval_seconds",
        ):
            cpu["metrics"]["training"][field] = _unavailable("not_applicable")
        validate_result(cpu)

        false_cpu = _valid_result()
        false_cpu["identities"]["device"] = "cpu"
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_result(false_cpu)


class ResultAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.protocol, self.dataset_root = _materialize_dataset_fixture(
            _frozen_protocol(), Path(temporary.name)
        )
        self.registry = generate_registry(self.protocol)
        self.evidence_root = Path(temporary.name) / "evidence"
        self.authorization = _runtime_authorization(self.protocol)
        self.audit_execution = _audit_execution(self.authorization)
        self.results, self.documents = _auditable_smoke_results(
            self.protocol, self.registry, self.evidence_root
        )

    def test_audit_binds_protocol_registry_and_recomputes_aggregates(self) -> None:
        bindings = _load_dataset_bindings(self.protocol, self.dataset_root)
        self.assertEqual(len(bindings["train"]["record_sha256s"]), 1)
        self.assertEqual(len(bindings["validation"]["record_sha256s"]), 256)
        self.assertEqual(len(bindings["test"]["record_sha256s"]), 512)
        report = audit_result_set(
            self.protocol,
            self.registry,
            self.results,
            self.documents,
            phases=["stage0_smoke"],
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=self.audit_execution,
        )
        self.assertEqual(report["expected_rows"], 4)
        self.assertEqual(report["complete_rows"], 4)
        self.assertEqual(report["failed_rows"], 0)
        self.assertFalse(report["test_open_verified"])
        self.assertEqual(
            report["execution_source_git_commit"],
            self.authorization["roles"][2]["source_git_commit"],
        )

    def test_stage0_audit_never_opens_held_out_test_content(self) -> None:
        test_root = self.dataset_root / "test"
        opened: list[Path] = []
        real_open = os.open

        def tracking_open(path: object, flags: int, *args: object, **kwargs: object):
            candidate = Path(path)
            opened.append(candidate)
            if candidate == test_root or test_root in candidate.parents:
                raise AssertionError(f"held-out test path was opened: {candidate}")
            return real_open(path, flags, *args, **kwargs)

        with mock.patch(
            "scripts.policy_improvement_audit.os.open",
            side_effect=tracking_open,
        ):
            report = audit_result_set(
                self.protocol,
                self.registry,
                self.results,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )
        self.assertEqual(report["complete_rows"], 4)
        self.assertTrue(
            any(self.dataset_root / "train" in path.parents for path in opened)
        )
        self.assertTrue(
            any(self.dataset_root / "validation" in path.parents for path in opened)
        )
        self.assertFalse(any(test_root in path.parents for path in opened))

    def test_dataset_audit_still_verifies_train_and_validation_content(self) -> None:
        for split in ("train", "validation"):
            target = self.dataset_root / split / "all__inputs.npy"
            original = target.read_bytes()
            try:
                target.write_bytes(original + b"tampered")
                with self.subTest(split=split), self.assertRaises(
                    PolicyImprovementSchemaError
                ):
                    _load_dataset_bindings(self.protocol, self.dataset_root)
            finally:
                target.write_bytes(original)

        test_target = self.dataset_root / "test/all__inputs.npy"
        original_test = test_target.read_bytes()
        try:
            test_target.write_bytes(original_test + b"tampered")
            _load_dataset_bindings(self.protocol, self.dataset_root)
            with self.assertRaises(PolicyImprovementSchemaError):
                _load_dataset_bindings(
                    self.protocol,
                    self.dataset_root,
                    verify_test_content=True,
                )
        finally:
            test_target.write_bytes(original_test)

    def test_dataset_authentication_rejects_missing_tampered_extra_and_aliases(
        self,
    ) -> None:
        for case in ("missing", "tampered", "extra", "symlink", "hardlink"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                protocol, root = _materialize_dataset_fixture(
                    _frozen_protocol(), Path(directory)
                )
                target = root / "validation/all__inputs.npy"
                if case == "missing":
                    target.unlink()
                elif case == "tampered":
                    target.write_bytes(target.read_bytes() + b"tampered")
                elif case == "extra":
                    (root / "validation/extra.npy").write_bytes(b"unexpected")
                elif case == "symlink":
                    target.unlink()
                    target.symlink_to(root / "train/all__inputs.npy")
                else:
                    target.unlink()
                    os.link(root / "train/all__inputs.npy", target)
                with self.assertRaises(PolicyImprovementSchemaError):
                    _load_dataset_bindings(protocol, root)

    def test_audit_rejects_missing_rows_arbitrary_hashes_and_aggregate_drift(
        self,
    ) -> None:
        missing = copy.deepcopy(self.results[:-1])
        arbitrary_hash = copy.deepcopy(self.results)
        arbitrary_hash[0]["protocol_sha256"] = "f" * 64
        aggregate_drift = copy.deepcopy(self.results)
        aggregate_drift[0]["evaluation_snapshots"][0]["policy_evaluations"][0][
            "primary"
        ]["solve_rate"] = _available(0.75)
        for results in (missing, arbitrary_hash, aggregate_drift):
            with self.assertRaises(PolicyImprovementSchemaError):
                audit_result_set(
                    self.protocol,
                    self.registry,
                    results,
                    self.documents,
                    phases=["stage0_smoke"],
                    dataset_root=self.dataset_root,
                    evidence_root=self.evidence_root,
                    runtime_authorization=self.authorization,
                    audit_execution_identity=self.audit_execution,
                )

    def test_audit_binds_effective_config_and_dataset_record_bytes(self) -> None:
        bad_config = copy.deepcopy(self.results)
        bad_config[0]["identities"]["effective_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(PolicyImprovementSchemaError, "effective config"):
            audit_result_set(
                self.protocol,
                self.registry,
                bad_config,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )

        bad_documents = copy.deepcopy(self.documents)
        bad_results = copy.deepcopy(self.results)
        evaluation = bad_results[0]["evaluation_snapshots"][0]["policy_evaluations"][0]
        old_digest = evaluation["per_instance_artifact_sha256"]["value"]
        changed = bad_documents.pop(old_digest)
        changed["records"][0]["puzzle_id"] = "validation-forged"
        new_digest = hashlib.sha256(canonical_json_bytes(changed)).hexdigest()
        bad_documents[new_digest] = changed
        evaluation["per_instance_artifact_sha256"] = _available(new_digest)
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "authenticated split manifest|immutable result",
        ):
            audit_result_set(
                self.protocol,
                self.registry,
                bad_results,
                bad_documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )

        split_manifest = self.dataset_root / "manifests" / "validation.json"
        original = split_manifest.read_bytes()
        try:
            split_manifest.write_bytes(original + b"\n")
            with self.assertRaisesRegex(
                PolicyImprovementSchemaError, "registered SHA-256"
            ):
                audit_result_set(
                    self.protocol,
                    self.registry,
                    self.results,
                    self.documents,
                    phases=["stage0_smoke"],
                    dataset_root=self.dataset_root,
                    evidence_root=self.evidence_root,
                    runtime_authorization=self.authorization,
                    audit_execution_identity=self.audit_execution,
                )
        finally:
            split_manifest.write_bytes(original)

    def test_audit_rejects_unauthorized_runtime_and_population_denominator(
        self,
    ) -> None:
        unauthorized = copy.deepcopy(self.results)
        unauthorized[0]["identities"]["training_runtime_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError, "not externally authorized"
        ):
            audit_result_set(
                self.protocol,
                self.registry,
                unauthorized,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )
        unauthorized_source = copy.deepcopy(self.results)
        unauthorized_source[0]["identities"]["evaluation_source_git_commit"] = (
            _available("f" * 40)
        )
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError, "not authorized|immutable result"
        ):
            audit_result_set(
                self.protocol,
                self.registry,
                unauthorized_source,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )
        unauthorized_auditor = copy.deepcopy(self.audit_execution)
        unauthorized_auditor["source_git_commit"] = "f" * 40
        with self.assertRaisesRegex(PolicyImprovementSchemaError, "not authorized"):
            audit_result_set(
                self.protocol,
                self.registry,
                self.results,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=unauthorized_auditor,
            )

        shortened_results = copy.deepcopy(self.results)
        shortened_documents = copy.deepcopy(self.documents)
        evaluation = shortened_results[0]["evaluation_snapshots"][0][
            "policy_evaluations"
        ][0]
        old_digest = evaluation["per_instance_artifact_sha256"]["value"]
        shortened = copy.deepcopy(shortened_documents.pop(old_digest))
        shortened["records"].pop()
        puzzle_hashes = [
            record["registered_record_sha256"] for record in shortened["records"]
        ]
        shortened["evaluation_pool_sha256"] = hashlib.sha256(
            canonical_json_bytes(puzzle_hashes)
        ).hexdigest()
        new_digest = hashlib.sha256(canonical_json_bytes(shortened)).hexdigest()
        shortened_documents[new_digest] = shortened
        evaluation["per_instance_artifact_sha256"] = _available(new_digest)
        evaluation["evaluation_pool_sha256"] = _available(
            shortened["evaluation_pool_sha256"]
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            audit_result_set(
                self.protocol,
                self.registry,
                shortened_results,
                shortened_documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )

    def test_audit_rejects_unpaired_puzzle_order_and_test_open_on_validation(
        self,
    ) -> None:
        results = copy.deepcopy(self.results)
        documents = copy.deepcopy(self.documents)
        evaluation = results[1]["evaluation_snapshots"][0]["policy_evaluations"][0]
        old_digest = evaluation["per_instance_artifact_sha256"]["value"]
        changed = copy.deepcopy(documents.pop(old_digest))
        changed["records"] = list(reversed(changed["records"]))
        new_digest = hashlib.sha256(canonical_json_bytes(changed)).hexdigest()
        documents[new_digest] = changed
        evaluation["per_instance_artifact_sha256"] = _available(new_digest)
        with self.assertRaises(PolicyImprovementSchemaError):
            audit_result_set(
                self.protocol,
                self.registry,
                results,
                documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )
        with self.assertRaises(PolicyImprovementSchemaError):
            audit_result_set(
                self.protocol,
                self.registry,
                self.results,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
                test_open_record={},
            )

    def test_failed_cell_is_retained_and_counted(self) -> None:
        results = copy.deepcopy(self.results)
        failed = results[2]
        failed["status"] = "failed"
        failed["failure"] = {
            "phase": "evaluation",
            "error_class": "runtime_error",
            "message_sha256": "e" * 64,
        }
        for snapshot in failed["evaluation_snapshots"]:
            snapshot["status"] = "unavailable"
            snapshot["unavailable_reason"] = "run_failed_before_evaluation"
            snapshot["target"]["registered_quantity"] = _unavailable(
                "run_failed_before_evaluation"
            )
            for field in (
                "observed_environment_interactions",
                "observed_recurrent_map_applications",
                "accelerator_seconds_observed",
                "checkpoint_sha256",
                "model_state_sha256",
                "checkpoint_lineage_sha256",
            ):
                snapshot[field] = _unavailable("run_failed_before_evaluation")
            snapshot["policy_evaluations"] = []
        shutil.rmtree(self.evidence_root / "runs" / str(failed["run_id"]) / "segments")
        _materialize_failed_attempt(failed, self.evidence_root)
        report = audit_result_set(
            self.protocol,
            self.registry,
            results,
            self.documents,
            phases=["stage0_smoke"],
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=self.audit_execution,
        )
        self.assertEqual(report["complete_rows"], 3)
        self.assertEqual(report["failed_rows"], 1)

    def test_failed_and_complete_generations_are_mutually_exclusive(self) -> None:
        results = copy.deepcopy(self.results)
        failed = results[0]
        failed["status"] = "failed"
        failed["failure"] = {
            "phase": "evaluation",
            "error_class": "runtime_error",
            "message_sha256": "e" * 64,
        }
        for snapshot in failed["evaluation_snapshots"]:
            snapshot["status"] = "unavailable"
            snapshot["unavailable_reason"] = "run_failed_before_evaluation"
            snapshot["target"]["registered_quantity"] = _unavailable(
                "run_failed_before_evaluation"
            )
            for field in (
                "observed_environment_interactions",
                "observed_recurrent_map_applications",
                "accelerator_seconds_observed",
                "checkpoint_sha256",
                "model_state_sha256",
                "checkpoint_lineage_sha256",
            ):
                snapshot[field] = _unavailable("run_failed_before_evaluation")
            snapshot["policy_evaluations"] = []
        _materialize_failed_attempt(failed, self.evidence_root)
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "Failed run root entries differ",
        ):
            audit_result_set(
                self.protocol,
                self.registry,
                results,
                self.documents,
                phases=["stage0_smoke"],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                audit_execution_identity=self.audit_execution,
            )

    def test_confirmatory_audit_requires_frozen_selection_and_test_open(self) -> None:
        protocol = self.protocol
        history, authorization = _amendment_history(protocol)
        base_configs = load_registered_base_configs(protocol, REPOSITORY_ROOT)
        materialized = generate_registry(protocol, history, base_configs=base_configs)
        with self.assertRaisesRegex(PolicyImprovementSchemaError, "test-open record"):
            audit_result_set(
                protocol,
                materialized,
                [],
                {},
                phases=["stage2_confirmatory"],
                amendment_history=history,
                base_configs=base_configs,
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=authorization,
                audit_execution_identity=_audit_execution(authorization),
                amendment_evidence={},
                _verify_amendment_evidence=False,
            )

    def test_post_smoke_freeze_binds_independently_reaudited_results(self) -> None:
        smoke_report = audit_result_set(
            self.protocol,
            self.registry,
            self.results,
            self.documents,
            phases=["stage0_smoke"],
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=self.audit_execution,
        )
        smoke_evidence = {
            "phase": "stage0_smoke",
            "audit_report_sha256": hashlib.sha256(
                canonical_json_bytes(smoke_report)
            ).hexdigest(),
            "result_set_sha256": smoke_report["result_set_sha256"],
            "per_instance_set_sha256": smoke_report["per_instance_set_sha256"],
            "expected_rows": 4,
            "complete_rows": 4,
            "failed_rows": 0,
        }
        history, authorization = _amendment_history(
            self.protocol, smoke_evidence=smoke_evidence
        )
        after_compute = generate_registry(self.protocol, history[:2])
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError, "Result inventory differs"
        ):
            audit_result_set(
                self.protocol,
                after_compute,
                [],
                {},
                phases=["stage1_screen"],
                amendment_history=history[:2],
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=authorization,
                audit_execution_identity=_audit_execution(authorization),
                amendment_evidence={
                    "stage0_smoke": {
                        "results": self.results,
                        "per_instance_documents": self.documents,
                    }
                },
            )
        tampered = copy.deepcopy(history[:2])
        tampered[1]["evidence"]["result_set_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError, "independent reaudit"
        ):
            audit_result_set(
                self.protocol,
                generate_registry(self.protocol, tampered),
                [],
                {},
                phases=["stage1_screen"],
                amendment_history=tampered,
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=authorization,
                audit_execution_identity=_audit_execution(authorization),
                amendment_evidence={
                    "stage0_smoke": {
                        "results": self.results,
                        "per_instance_documents": self.documents,
                    }
                },
            )


class RegisteredStatisticsTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_paired_and_seed_clustered(self) -> None:
        pairs = {
            11: [(1.0, 0.0), (1.0, 0.0)],
            22: [(0.0, 1.0), (0.0, 1.0)],
        }
        first = paired_seed_cluster_puzzle_bootstrap(pairs, replicates=64, seed=7)
        second = paired_seed_cluster_puzzle_bootstrap(
            copy.deepcopy(pairs), replicates=64, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first["observed_difference"], 0.0)
        self.assertIn(-100.0, first["replicate_differences"])
        self.assertIn(100.0, first["replicate_differences"])

        identical = paired_seed_cluster_puzzle_bootstrap(
            {1: [(1.0, 1.0), (0.0, 0.0)]},
            replicates=20,
            seed=3,
        )
        self.assertEqual(set(identical["replicate_differences"]), {0.0})
        self.assertEqual(identical["interval_lower"], 0.0)
        self.assertEqual(identical["interval_upper"], 0.0)

    def test_exact_seed_permutation_matches_hand_oracles(self) -> None:
        self.assertEqual(
            paired_seed_permutation_test([1.0, 1.0])["two_sided_p_value"],
            0.5,
        )
        self.assertEqual(
            paired_seed_permutation_test([1.0, -1.0])["two_sided_p_value"],
            1.0,
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            paired_seed_permutation_test([])

    def test_holm_matches_hand_oracle_and_rejects_invalid_values(self) -> None:
        self.assertEqual(
            holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}),
            {"a": 0.03, "b": 0.06, "c": 0.06},
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            holm_adjust({"bad": math.nan})
        with self.assertRaises(PolicyImprovementSchemaError):
            paired_seed_cluster_puzzle_bootstrap(
                {1: [(1.0, math.nan)]}, replicates=10, seed=1
            )

    def test_relative_ratio_is_unavailable_for_zero_control_rate(self) -> None:
        estimates = _contrast_estimates(
            [
                {
                    "treatment_solve_rate": 0.25,
                    "control_solve_rate": 0.0,
                }
            ]
        )
        self.assertEqual(
            estimates["relative_ratio"],
            {"status": "unavailable", "reason": "zero_control_solve_rate"},
        )


class RegisteredAnalysisConsumerTest(unittest.TestCase):
    def setUp(self) -> None:
        shared = getattr(type(self), "_shared_fixture", None)
        if shared is not None:
            self.__dict__.update(shared)
            return
        temporary = tempfile.TemporaryDirectory()
        self.protocol, self.dataset_root = _materialize_dataset_fixture(
            _frozen_protocol(), Path(temporary.name)
        )
        self.evidence_root = Path(temporary.name) / "evidence"
        self.authorization = _runtime_authorization(self.protocol)
        self.base_configs = load_registered_base_configs(self.protocol, REPOSITORY_ROOT)
        base_registry = generate_registry(self.protocol, base_configs=self.base_configs)
        smoke_results, smoke_documents = _auditable_smoke_results(
            self.protocol, base_registry, self.evidence_root
        )
        smoke_report = audit_result_set(
            self.protocol,
            base_registry,
            smoke_results,
            smoke_documents,
            phases=["stage0_smoke"],
            base_configs=self.base_configs,
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=_audit_execution(self.authorization),
        )

        def evidence(phase: str, report: dict[str, object]) -> dict[str, object]:
            return {
                "phase": phase,
                "audit_report_sha256": hashlib.sha256(
                    canonical_json_bytes(report)
                ).hexdigest(),
                "result_set_sha256": report["result_set_sha256"],
                "per_instance_set_sha256": report["per_instance_set_sha256"],
                "expected_rows": report["expected_rows"],
                "complete_rows": report["complete_rows"],
                "failed_rows": report["failed_rows"],
            }

        template_history, _ = _amendment_history(
            self.protocol, smoke_evidence=evidence("stage0_smoke", smoke_report)
        )
        theory = template_history[0]
        compute = template_history[1]
        screen_registry = generate_registry(
            self.protocol, [theory, compute], base_configs=self.base_configs
        )
        screen_results, screen_documents = _auditable_phase_results(
            self.protocol,
            screen_registry,
            [theory, compute],
            self.authorization,
            "stage1_screen",
            self.evidence_root,
        )
        screen_report = audit_result_set(
            self.protocol,
            screen_registry,
            screen_results,
            screen_documents,
            phases=["stage1_screen"],
            amendment_history=[theory, compute],
            base_configs=self.base_configs,
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=_audit_execution(self.authorization),
            amendment_evidence={
                "stage0_smoke": {
                    "results": smoke_results,
                    "per_instance_documents": smoke_documents,
                }
            },
        )
        screen = copy.deepcopy(template_history[2])
        screen["evidence"] = evidence("stage1_screen", screen_report)
        screen["selected_exact"] = derive_registered_selection(
            "stage1_screen", screen_results
        )
        alpha_registry = generate_registry(
            self.protocol, [theory, compute, screen], base_configs=self.base_configs
        )
        alpha_results, alpha_documents = _auditable_phase_results(
            self.protocol,
            alpha_registry,
            [theory, compute, screen],
            self.authorization,
            "stage1_alpha",
            self.evidence_root,
        )
        prior_evidence = {
            "stage0_smoke": {
                "results": smoke_results,
                "per_instance_documents": smoke_documents,
            },
            "stage1_screen": {
                "results": screen_results,
                "per_instance_documents": screen_documents,
            },
        }
        alpha_report = audit_result_set(
            self.protocol,
            alpha_registry,
            alpha_results,
            alpha_documents,
            phases=["stage1_alpha"],
            amendment_history=[theory, compute, screen],
            base_configs=self.base_configs,
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=_audit_execution(self.authorization),
            amendment_evidence=prior_evidence,
        )
        final = copy.deepcopy(template_history[3])
        final["evidence"] = evidence("stage1_alpha", alpha_report)
        final["selected_exact"] = derive_registered_selection(
            "stage1_alpha", alpha_results
        )
        final["prior_amendment_history_sha256"] = amendment_history_sha256(
            [theory, compute, screen]
        )
        final["source_registry_sha256"] = registry_sha256(alpha_registry)
        self.history = [theory, compute, screen, final]
        self.screen_results = screen_results
        self.alpha_results = alpha_results
        self.registry = generate_registry(
            self.protocol, self.history, base_configs=self.base_configs
        )
        self.amendment_evidence = {
            **prior_evidence,
            "stage1_alpha": {
                "results": alpha_results,
                "per_instance_documents": alpha_documents,
            },
        }
        self.test_open_owner_root = self.evidence_root
        self.test_open_record = expected_test_open_record(
            protocol=self.protocol,
            registry=self.registry,
            amendment_history=self.history,
            runtime_authorization=self.authorization,
            opened_at_utc="2026-08-14T12:04:00Z",
            base_configs=self.base_configs,
        )
        self.test_open_identity = _publish_record(
            owner_root=self.test_open_owner_root,
            record=self.test_open_record,
        )
        self.results, self.documents = _auditable_phase_results(
            self.protocol,
            self.registry,
            self.history,
            self.authorization,
            "stage2_confirmatory",
            self.evidence_root,
            test_open_sha256=self.test_open_identity["sha256"],
        )
        self.audit_report = audit_result_set(
            self.protocol,
            self.registry,
            self.results,
            self.documents,
            phases=["stage2_confirmatory"],
            amendment_history=self.history,
            base_configs=self.base_configs,
            dataset_root=self.dataset_root,
            evidence_root=self.evidence_root,
            runtime_authorization=self.authorization,
            audit_execution_identity=_audit_execution(self.authorization),
            amendment_evidence=self.amendment_evidence,
            test_open_record=self.test_open_record,
            test_open_owner_root=self.test_open_owner_root,
            test_open_sha256=self.test_open_identity["sha256"],
        )
        type(self)._shared_fixture = {
            **self.__dict__,
            "_temporary_directory": temporary,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shared = getattr(cls, "_shared_fixture", None)
        if shared is not None:
            shared["_temporary_directory"].cleanup()

    def _analysis_execution(self) -> dict[str, object]:
        role = self.authorization["roles"][3]
        return {
            "runtime_sha256": role["runtime_sha256"],
            "runtime_profile_sha256": role["runtime_profile_sha256"],
            "source_git_commit": role["source_git_commit"],
            "launcher_sha256": self.authorization["launcher_sha256"],
            "producer_git_commit": self.authorization["producer_git_commit"],
            "producer_source_manifest_sha256": self.authorization[
                "producer_source_manifest_sha256"
            ],
        }

    def test_test_open_cli_rejects_counts_without_immutable_generations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_evidence = root / "evidence"
            empty_evidence.mkdir(mode=0o700)
            protocol_path = root / "protocol.json"
            registry_path = root / "registry.json"
            _write_json_fixture(protocol_path, self.protocol)
            _write_json_fixture(registry_path, self.registry)
            amendment_paths = []
            for index, amendment in enumerate(self.history):
                path = root / f"amendment_{index}.json"
                _write_json_fixture(path, amendment)
                amendment_paths.append(path)
            evidence_arguments = []
            for phase, evidence in self.amendment_evidence.items():
                path = root / f"{phase}_evidence.json"
                _write_json_fixture(path, evidence)
                evidence_arguments.extend(["--amendment-evidence", f"{phase}={path}"])
            audit_role = self.authorization["roles"][2]
            arguments = [
                "--protocol",
                str(protocol_path),
                "--registry",
                str(registry_path),
            ]
            for path in amendment_paths:
                arguments.extend(["--amendment", str(path)])
            arguments.extend(
                [
                    "--project-root",
                    str(REPOSITORY_ROOT),
                    "--dataset-root",
                    str(self.dataset_root),
                    "--evidence-root",
                    str(empty_evidence),
                    *evidence_arguments,
                    "--audit-runtime-sha256",
                    str(audit_role["runtime_sha256"]),
                    "--audit-runtime-profile-sha256",
                    str(audit_role["runtime_profile_sha256"]),
                    "--audit-source-git-commit",
                    str(audit_role["source_git_commit"]),
                    "--launcher-sha256",
                    str(self.authorization["launcher_sha256"]),
                    "--producer-git-commit",
                    str(self.authorization["producer_git_commit"]),
                    "--producer-source-manifest-sha256",
                    str(self.authorization["producer_source_manifest_sha256"]),
                    "--runtime-authorization-json",
                    canonical_json_bytes(self.authorization).decode("ascii"),
                    "--runtime-authorization-sha256",
                    runtime_authorization_sha256(self.authorization),
                ]
            )
            with self.assertRaises(PolicyImprovementSchemaError):
                test_open_main(
                    arguments,
                    checkpoint_validator=_fixture_checkpoint_validator,
                )
            self.assertFalse((empty_evidence / "TEST_OPEN.json").exists())

    @staticmethod
    def _bootstrap_fixture(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "schema_name": "paired_seed_cluster_puzzle_bootstrap_v1",
            "schema_version": 1,
            "seed_count": 8,
            "replicates": 10000,
            "prng_seed": 3472274560,
            "confidence_level": 0.95,
            "scale": 100.0,
            "observed_difference": 25.0,
            "interval_lower": 20.0,
            "interval_upper": 30.0,
            "replicate_differences": [20.0, 25.0, 30.0],
        }

    def _run(self, results: list[dict[str, object]]) -> dict[str, object]:
        with mock.patch(
            "scripts.policy_improvement_analysis.paired_seed_cluster_puzzle_bootstrap",
            side_effect=self._bootstrap_fixture,
        ):
            return analyze_stage2_confirmatory(
                self.protocol,
                self.registry,
                self.history,
                results,
                self.documents,
                self.audit_report,
                base_configs=self.base_configs,
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                analysis_execution_identity=self._analysis_execution(),
                amendment_evidence=self.amendment_evidence,
                test_open_record=self.test_open_record,
                test_open_owner_root=self.test_open_owner_root,
                test_open_sha256=self.test_open_identity["sha256"],
            )

    def test_analysis_accepts_only_complete_registered_primary_set(self) -> None:
        analysis = self._run(self.results)
        self.assertEqual(
            self.history[2]["selected_exact"],
            derive_registered_selection("stage1_screen", self.screen_results),
        )
        self.assertEqual(
            self.history[3]["selected_exact"],
            derive_registered_selection("stage1_alpha", self.alpha_results),
        )
        tied_screen = copy.deepcopy(self.screen_results)
        for result in tied_screen:
            if not str(result["method_id"]).startswith("fixed_base_exact_"):
                continue
            primary = next(
                evaluation["primary"]
                for evaluation in result["evaluation_snapshots"][0][
                    "policy_evaluations"
                ]
                if evaluation["policy_variant"] == result["primary_policy_variant"]
            )
            primary["solved_count"] = _available(128)
            primary["solve_rate"] = _available(0.5)
            secondary = next(
                evaluation["secondary"]
                for evaluation in result["evaluation_snapshots"][0][
                    "policy_evaluations"
                ]
                if evaluation["policy_variant"] == result["primary_policy_variant"]
            )
            secondary["terminal_reason_counts"] = _available(
                {"budget": 128, "solved": 128, "stop": 0}
            )
        self.assertEqual(
            derive_registered_selection("stage1_screen", tied_screen),
            {
                "method_id": "fixed_base_exact_persistent",
                "n": 2,
                "K": 1,
            },
        )
        self.assertEqual(analysis["seed_count"], 8)
        self.assertEqual(analysis["evaluation_records_per_seed"], 512)
        self.assertEqual(
            analysis["primary"]["contrast"],
            "fixed_base_exact_persistent-minus-matched_ppo",
        )
        self.assertEqual(len(analysis["prespecified_secondary"]), 2)
        self.assertEqual(analysis["failed_run_ids"], [])
        self.assertEqual(len(analysis["primary"]["per_seed"]), 8)
        self.assertEqual(
            analysis["primary"]["estimates"]["absolute_percentage_point_difference"],
            25.0,
        )
        self.assertEqual(
            analysis["primary"]["estimates"]["relative_ratio"],
            {"status": "available", "value": 1.5},
        )
        self.assertEqual(
            analysis["provenance"]["analysis_source_git_commit"],
            self.authorization["roles"][3]["source_git_commit"],
        )
        self.assertEqual(
            analysis["provenance"]["external_audit_source_git_commit"],
            self.authorization["roles"][2]["source_git_commit"],
        )

    def test_analysis_refuses_failure_wrong_phase_or_incomplete_seed_set(self) -> None:
        failed = copy.deepcopy(self.results)
        failed[0]["status"] = "failed"
        wrong_phase = copy.deepcopy(self.results)
        wrong_phase[0]["phase"] = "stage1_alpha"
        incomplete = copy.deepcopy(self.results[:-1])
        for mutation in (failed, wrong_phase, incomplete):
            with self.subTest(length=len(mutation)):
                with self.assertRaises(PolicyImprovementSchemaError):
                    self._run(mutation)

    def test_analysis_refuses_wrong_policy_variant_and_unauthorized_runtime(
        self,
    ) -> None:
        wrong_variant = copy.deepcopy(self.results)
        wrong_variant[0]["primary_policy_variant"] = "realized_policy"
        with self.assertRaises(PolicyImprovementSchemaError):
            self._run(wrong_variant)
        bad_execution = self._analysis_execution()
        bad_execution["source_git_commit"] = "f" * 40
        with self.assertRaises(PolicyImprovementSchemaError):
            analyze_stage2_confirmatory(
                self.protocol,
                self.registry,
                self.history,
                self.results,
                self.documents,
                self.audit_report,
                base_configs=self.base_configs,
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                analysis_execution_identity=bad_execution,
                amendment_evidence=self.amendment_evidence,
                test_open_record=self.test_open_record,
                test_open_owner_root=self.test_open_owner_root,
                test_open_sha256=self.test_open_identity["sha256"],
            )

    def test_analysis_refuses_unregistered_validation_selection(self) -> None:
        history = copy.deepcopy(self.history)
        history[2]["selected_exact"] = {
            "method_id": "fixed_base_exact_episodic",
            "n": 4,
            "K": 5,
        }
        with self.assertRaises(PolicyImprovementSchemaError):
            analyze_stage2_confirmatory(
                self.protocol,
                self.registry,
                history,
                self.results,
                self.documents,
                self.audit_report,
                base_configs=self.base_configs,
                dataset_root=self.dataset_root,
                evidence_root=self.evidence_root,
                runtime_authorization=self.authorization,
                analysis_execution_identity=self._analysis_execution(),
                amendment_evidence=self.amendment_evidence,
                test_open_record=self.test_open_record,
                test_open_owner_root=self.test_open_owner_root,
                test_open_sha256=self.test_open_identity["sha256"],
            )


@unittest.skipIf(np is None, "NumPy is supplied by the owned Buck test")
class DatasetSymmetryTest(unittest.TestCase):
    def setUp(self) -> None:
        assert np is not None
        from dataset.build_policy_improvement_4x4 import (
            canonical_sudoku_record_bytes,
            canonical_sudoku_record_sha256,
            materialized_split_manifest,
            record_sha256,
            verify_cross_split_symmetry_hashes,
        )

        self.canonical_sudoku_record_bytes = canonical_sudoku_record_bytes
        self.canonical_sudoku_record_sha256 = canonical_sudoku_record_sha256
        self.materialized_split_manifest = materialized_split_manifest
        self.record_sha256 = record_sha256
        self.verify_cross_split_symmetry_hashes = verify_cross_split_symmetry_hashes
        solution = np.array(
            [
                [1, 2, 3, 4],
                [3, 4, 1, 2],
                [2, 1, 4, 3],
                [4, 3, 2, 1],
            ],
            dtype=np.int32,
        )
        puzzle = solution.copy()
        puzzle[0, 0] = 0
        puzzle[1, 3] = 0
        puzzle[2, 2] = 0
        self.inputs = np.where(puzzle == 0, 1, puzzle + 1).astype(np.int32)
        self.labels = (solution + 1).astype(np.int32)

    def test_canonical_identity_is_invariant_to_registered_symmetries(self) -> None:
        base = self.canonical_sudoku_record_bytes(self.inputs, self.labels)
        rows = [1, 0, 2, 3]
        columns = [2, 3, 0, 1]
        transformed_inputs = self.inputs.T[np.ix_(rows, columns)].copy()
        transformed_labels = self.labels.T[np.ix_(rows, columns)].copy()
        digit_map = {2: 5, 3: 4, 4: 2, 5: 3}
        for array in (transformed_inputs, transformed_labels):
            original = array.copy()
            for source, target in digit_map.items():
                array[original == source] = target
        self.assertEqual(
            base,
            self.canonical_sudoku_record_bytes(transformed_inputs, transformed_labels),
        )

    def test_cross_split_symmetry_overlap_fails(self) -> None:
        from dataset.build_policy_improvement_4x4 import PolicyImprovementDatasetError

        digest = self.canonical_sudoku_record_sha256(self.inputs, self.labels)
        with self.assertRaises(PolicyImprovementDatasetError):
            self.verify_cross_split_symmetry_hashes(
                {"train": [digest], "validation": ["b" * 64], "test": [digest]}
            )
        self.verify_cross_split_symmetry_hashes(
            {"train": [digest], "validation": ["b" * 64], "test": ["c" * 64]}
        )

    def test_materialized_manifest_matches_training_loader_contract(self) -> None:
        from utils.dataset_provenance import (
            input_sha256,
            ordered_record_sha256,
            sample_sha256,
        )

        expected_files = {
            "all__group_indices.npy",
            "all__inputs.npy",
            "all__labels.npy",
            "all__puzzle_identifiers.npy",
            "all__puzzle_indices.npy",
            "dataset.json",
            "records.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            split_path = Path(temporary) / "train"
            split_path.mkdir()
            for name in expected_files:
                path = split_path / name
                if name.endswith(".npy"):
                    with path.open("wb") as handle:
                        np.save(handle, np.array([1], dtype=np.int32))
                else:
                    path.write_text("{}\n", encoding="ascii")
            inputs = self.inputs.reshape(1, 16)
            labels = self.labels.reshape(1, 16)
            record_digest = self.record_sha256(inputs[0], labels[0])
            symmetry_digest = self.canonical_sudoku_record_sha256(inputs[0], labels[0])
            manifest = self.materialized_split_manifest(
                split_name="train",
                split_path=split_path,
                generation_seed=26081401,
                inputs=inputs,
                records=[
                    {
                        "index": 0,
                        "record_sha256": record_digest,
                        "symmetry_canonical_sha256": symmetry_digest,
                    }
                ],
            )
        self.assertEqual(manifest["generated_count"], 1)
        self.assertEqual(manifest["record_sha256s"], [record_digest])
        self.assertEqual(manifest["input_sha256s"], [input_sha256(inputs[0])])
        self.assertEqual(
            record_digest,
            sample_sha256(inputs[0], labels[0]),
        )
        self.assertEqual(
            manifest["ordered_record_sha256"],
            ordered_record_sha256([record_digest]),
        )
        self.assertEqual(
            set(manifest["files"]),
            {f"train/{name}" for name in expected_files},
        )


class SmokePlanTest(unittest.TestCase):
    def test_renderer_emits_executable_authenticated_smoke_commands(self) -> None:
        protocol = _frozen_protocol()
        authorization = _stage0_runtime_authorization(protocol)
        plan = render_smoke_plan(
            protocol,
            protocol_path="/repo/configs/policy_improvement_v1/protocol.json",
            launcher_path="/artifacts/phase4_runtime_launcher",
            training_runtime_path="/artifacts/upi_trm_train.par",
            training_runtime_sha256=authorization["roles"][0]["runtime_sha256"],
            source_project_root=str(REPOSITORY_ROOT),
            expected_source_git_commit=authorization["producer_git_commit"],
            dataset_root=(
                "/evidence/data/policy-improvement-v1-owner/"
                "policy-improvement-hard-4x4-v1"
            ),
            train_manifest_sha256="c" * 64,
            validation_manifest_sha256="d" * 64,
            evidence_root="/evidence",
            runtime_authorization_value=authorization,
            runtime_authorization_path="/evidence/runtime-authorization.json",
            expected_runtime_authorization_sha256=runtime_authorization_sha256(
                authorization
            ),
        )
        self.assertTrue(plan["execution_authorized"])
        self.assertEqual(plan["blocked_by"], [])
        self.assertEqual(len(plan["rows"]), 4)
        for row in plan["rows"]:
            self.assertTrue(row["executable"])
            self.assertEqual(set(row["commands"]), {"prepare", "resume"})
            for command in row["commands"].values():
                self.assertEqual(command[0], "/artifacts/phase4_runtime_launcher")
                self.assertIn("policy-improvement-smoke", command)
                self.assertIn("--expected-runtime-sha256", command)
                self.assertIn("--policy-improvement-row-id", command)
                self.assertIn("--validation-manifest-sha256", command)
                self.assertNotIn("--confirmatory", command)
                self.assertNotIn("--resume-checkpoint", command)
                self.assertNotIn("test", command)

    def test_renderer_refuses_uncommitted_dataset_manifests(self) -> None:
        protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
        authorization = _stage0_runtime_authorization(protocol)
        with self.assertRaises(PolicyImprovementSchemaError):
            render_smoke_plan(
                protocol,
                protocol_path="/repo/configs/policy_improvement_v1/protocol.json",
                launcher_path="/artifacts/phase4_runtime_launcher",
                training_runtime_path="/artifacts/upi_trm_train.par",
                training_runtime_sha256=authorization["roles"][0]["runtime_sha256"],
                source_project_root=str(REPOSITORY_ROOT),
                expected_source_git_commit=authorization["producer_git_commit"],
                dataset_root=(
                    "/evidence/data/policy-improvement-v1-owner/"
                    "policy-improvement-hard-4x4-v1"
                ),
                train_manifest_sha256="c" * 64,
                validation_manifest_sha256="d" * 64,
                evidence_root="/evidence",
                runtime_authorization_value=authorization,
                runtime_authorization_path="/evidence/runtime-authorization.json",
                expected_runtime_authorization_sha256=runtime_authorization_sha256(
                    authorization
                ),
            )

    def test_renderer_refuses_runtime_role_drift(self) -> None:
        protocol = _frozen_protocol()
        for role_index, field, value in (
            (1, "runtime_sha256", "f" * 64),
            (1, "runtime_profile_sha256", "e" * 64),
            (0, "runtime_profile_sha256", "f" * 64),
        ):
            authorization = _stage0_runtime_authorization(protocol)
            authorization["roles"][role_index][field] = value
            if field == "runtime_profile_sha256":
                authorization["roles"][role_index][
                    "selected_source_manifest_sha256"
                ] = value
            with (
                self.subTest(role_index=role_index, field=field),
                self.assertRaisesRegex(
                    PolicyImprovementSchemaError,
                    "in-process evaluation role",
                ),
            ):
                render_smoke_plan(
                    protocol,
                    protocol_path=("/repo/configs/policy_improvement_v1/protocol.json"),
                    launcher_path="/artifacts/phase4_runtime_launcher",
                    training_runtime_path="/artifacts/upi_trm_train.par",
                    training_runtime_sha256=authorization["roles"][0]["runtime_sha256"],
                    source_project_root=str(REPOSITORY_ROOT),
                    expected_source_git_commit=authorization["producer_git_commit"],
                    dataset_root=(
                        "/evidence/data/policy-improvement-v1-owner/"
                        "policy-improvement-hard-4x4-v1"
                    ),
                    train_manifest_sha256="c" * 64,
                    validation_manifest_sha256="d" * 64,
                    evidence_root="/evidence",
                    runtime_authorization_value=authorization,
                    runtime_authorization_path=("/evidence/runtime-authorization.json"),
                    expected_runtime_authorization_sha256=(
                        runtime_authorization_sha256(authorization)
                    ),
                )


if __name__ == "__main__":
    unittest.main()
