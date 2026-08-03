#!/usr/bin/env python3
"""Run finite-batch diagnostics from one verified persistent checkpoint.

The checkpoint fixes every training-facing quantity.  The external diagnostic
spec only fixes held-out record selection, finite-reference depths, Monte Carlo
sampling, and batching.  Output is deterministic strict JSON and never
overwrites an existing artifact bundle.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from rl.persistent_diagnostic_checkpoint import (
    UNVERIFIABLE_STATUS,
    PersistentDiagnosticContext,
    file_sha256,
    load_persistent_diagnostic_context,
)
from rl.persistent_diagnostics import (
    AugmentedDiagnosticState,
    DeploymentSpec,
    PersistentDiagnosticConfig,
    PersistentDiagnosticOutput,
    PersistentStateCollection,
    collect_persistent_diagnostic_states,
    run_persistent_diagnostics,
)
from rl.evaluator import evaluate_plan_policy_with_scores
from rl.theory_diagnostics import FINITE_BATCH_SCOPE
from rl.theory_diagnostics import summarize_finite_batch_metric
from rl.upi_trm_trainer import UPITrmTrainer
from utils.lipschitz import compute_exact_baseline_summation


DIAGNOSTIC_SPEC_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_EXPECTED_SPEC_KEYS = {
    "schema_version",
    "record_selection",
    "reference_depth_offsets",
    "monte_carlo",
    "collection_seed",
    "max_retained_states",
    "chunk_size",
    "probability_tolerance",
    "ratio_denominator_tolerance",
    "initial_latent_sensitivity",
}


class PersistentDiagnosticArtifactError(RuntimeError):
    """Raised before an ambiguous or destructive artifact operation."""


def _strict_json_loads(raw: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PersistentDiagnosticArtifactError(
                    f"JSON object contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise PersistentDiagnosticArtifactError(
            f"{label} keys differ from schema; missing={missing}, unknown={unknown}."
        )


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PersistentDiagnosticArtifactError(
            f"{label} must be an integer at least {minimum}."
        )
    return value


def _require_finite_float(
    value: Any,
    *,
    label: str,
    minimum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PersistentDiagnosticArtifactError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise PersistentDiagnosticArtifactError(
            f"{label} must be finite and at least {minimum}."
        )
    return result


def load_diagnostic_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one exact schema-v1 diagnostic spec without accepting defaults."""

    spec_path = Path(path).expanduser().resolve()
    try:
        raw = spec_path.read_bytes()
        parsed = _strict_json_loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistentDiagnosticArtifactError(
            "Diagnostic spec must be readable UTF-8 JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise PersistentDiagnosticArtifactError("Diagnostic spec must be an object.")
    _require_exact_keys(parsed, _EXPECTED_SPEC_KEYS, label="diagnostic spec")
    if parsed["schema_version"] != DIAGNOSTIC_SPEC_SCHEMA_VERSION:
        raise PersistentDiagnosticArtifactError(
            "Unsupported diagnostic-spec schema version."
        )

    selection = parsed["record_selection"]
    if not isinstance(selection, dict):
        raise PersistentDiagnosticArtifactError("record_selection must be an object.")
    _require_exact_keys(
        selection,
        {"kind", "record_count"},
        label="record_selection",
    )
    if selection["kind"] != "ordered_prefix_without_replacement":
        raise PersistentDiagnosticArtifactError(
            "record_selection.kind must be ordered_prefix_without_replacement."
        )
    record_count = _require_int(
        selection["record_count"],
        label="record_selection.record_count",
        minimum=1,
    )

    offsets = parsed["reference_depth_offsets"]
    if not isinstance(offsets, list) or not offsets:
        raise PersistentDiagnosticArtifactError(
            "reference_depth_offsets must be a nonempty list."
        )
    normalized_offsets = [
        _require_int(offset, label="reference depth offset", minimum=1)
        for offset in offsets
    ]
    if normalized_offsets != sorted(set(normalized_offsets)):
        raise PersistentDiagnosticArtifactError(
            "reference_depth_offsets must be strictly increasing and unique."
        )

    monte_carlo = parsed["monte_carlo"]
    if not isinstance(monte_carlo, dict):
        raise PersistentDiagnosticArtifactError("monte_carlo must be an object.")
    _require_exact_keys(
        monte_carlo,
        {"repeats", "seed"},
        label="monte_carlo",
    )
    repeats = _require_int(
        monte_carlo["repeats"],
        label="monte_carlo.repeats",
        minimum=2,
    )
    mc_seed = _require_int(
        monte_carlo["seed"],
        label="monte_carlo.seed",
    )
    collection_seed = _require_int(
        parsed["collection_seed"],
        label="collection_seed",
    )
    max_retained_states = _require_int(
        parsed["max_retained_states"],
        label="max_retained_states",
        minimum=1,
    )
    chunk_size = _require_int(
        parsed["chunk_size"],
        label="chunk_size",
        minimum=1,
    )
    probability_tolerance = _require_finite_float(
        parsed["probability_tolerance"],
        label="probability_tolerance",
        minimum=0.0,
    )
    ratio_tolerance = _require_finite_float(
        parsed["ratio_denominator_tolerance"],
        label="ratio_denominator_tolerance",
        minimum=0.0,
    )
    initial_sensitivity = parsed["initial_latent_sensitivity"]
    if not isinstance(initial_sensitivity, dict):
        raise PersistentDiagnosticArtifactError(
            "initial_latent_sensitivity must be an object."
        )
    _require_exact_keys(
        initial_sensitivity,
        {"perturbation_l2_norm", "seed"},
        label="initial_latent_sensitivity",
    )
    perturbation_norm = _require_finite_float(
        initial_sensitivity["perturbation_l2_norm"],
        label="initial_latent_sensitivity.perturbation_l2_norm",
        minimum=0.0,
    )
    if perturbation_norm == 0.0:
        raise PersistentDiagnosticArtifactError(
            "The registered latent perturbation norm must be positive."
        )
    perturbation_seed = _require_int(
        initial_sensitivity["seed"],
        label="initial_latent_sensitivity.seed",
    )

    normalized = {
        "schema_version": DIAGNOSTIC_SPEC_SCHEMA_VERSION,
        "record_selection": {
            "kind": "ordered_prefix_without_replacement",
            "record_count": record_count,
        },
        "reference_depth_offsets": normalized_offsets,
        "monte_carlo": {"repeats": repeats, "seed": mc_seed},
        "collection_seed": collection_seed,
        "max_retained_states": max_retained_states,
        "chunk_size": chunk_size,
        "probability_tolerance": probability_tolerance,
        "ratio_denominator_tolerance": ratio_tolerance,
        "initial_latent_sensitivity": {
            "perturbation_l2_norm": perturbation_norm,
            "seed": perturbation_seed,
        },
    }
    return normalized, hashlib.sha256(raw).hexdigest()


def build_diagnostic_config(
    context: PersistentDiagnosticContext,
    spec: Mapping[str, Any],
    *,
    reference_depth_offset: Optional[int] = None,
) -> PersistentDiagnosticConfig:
    """Combine checkpoint-authoritative mechanics with registered diagnostics."""

    rl_config = context.checkpoint.rl_config
    evaluator_device = next(context.checkpoint.evaluator.parameters()).device
    if evaluator_device.type != "cpu":
        raise PersistentDiagnosticArtifactError(
            "Registered persistent diagnostic artifacts require CPU execution."
        )
    offsets = spec["reference_depth_offsets"]
    primary_offset = int(
        offsets[-1]
        if reference_depth_offset is None
        else reference_depth_offset
    )
    if primary_offset not in offsets:
        raise PersistentDiagnosticArtifactError(
            "Requested reference depth is not registered in the diagnostic spec."
        )
    return PersistentDiagnosticConfig(
        inner_unroll_n=int(rl_config.inner_unroll_n),
        reference_depth_m=int(rl_config.inner_unroll_n) + primary_offset,
        gamma=float(rl_config.gamma),
        k_horizon=int(rl_config.K),
        mc_repeats=int(spec["monte_carlo"]["repeats"]),
        mc_seed=int(spec["monte_carlo"]["seed"]),
        collection_seed=int(spec["collection_seed"]),
        mixture_alpha=float(rl_config.mixture_alpha),
        max_retained_states=int(spec["max_retained_states"]),
        chunk_size=int(spec["chunk_size"]),
        advantage_clip=(
            None
            if rl_config.advantage_clip is None
            else float(rl_config.advantage_clip)
        ),
        probability_tolerance=float(spec["probability_tolerance"]),
        ratio_denominator_tolerance=float(
            spec["ratio_denominator_tolerance"]
        ),
        initial_latent_perturbation_l2_norm=float(
            spec["initial_latent_sensitivity"]["perturbation_l2_norm"]
        ),
        initial_latent_perturbation_seed=int(
            spec["initial_latent_sensitivity"]["seed"]
        ),
    )


def _merge_reference_depth_runs(
    runs: Mapping[int, PersistentDiagnosticOutput],
) -> PersistentDiagnosticOutput:
    """Merge registered depths while rejecting depth-dependent common results."""

    if not runs:
        raise PersistentDiagnosticArtifactError("No reference-depth run was produced.")
    primary_offset = max(runs)
    primary = runs[primary_offset]
    state_common_keys = {
        "occurrence_id",
        "augmented_state_hash",
        "value_n",
        "exact_one_step_backup",
        "exact_one_step_signed_residual",
        "exact_one_step_absolute_residual",
        "centering_defect",
        "production_q_max_absolute_error",
        "production_baseline_absolute_error",
        "mc_k_step_operator_mean",
        "mc_k_step_operator_standard_error",
        "mc_k_step_signed_residual",
        "mc_k_step_absolute_residual",
        "input_latent_norm",
        "post_unroll_latent_norm",
        "carry_update_distance",
        "carry_recomputation_error",
        "candidate_expected_estimated_advantage",
        "exact_mixture_total_variation_to_deployed",
        "exact_mixture_kl_to_deployed",
        "deployed_kl_to_exact_mixture",
        "exact_mixture_support_mismatch",
        "initial_latent_sensitivity",
    }
    action_common_keys = {
        "occurrence_id",
        "augmented_state_hash",
        "action_index",
        "valid_action",
        "current_policy_probability",
        "candidate_policy_probability",
        "exact_mixture_probability",
        "deployed_policy_probability",
        "q_n",
        "production_q_n",
        "raw_advantage_n",
        "configured_estimator_advantage_n",
    }
    mc_common_keys = {
        "schema_version",
        "scope",
        "occurrence_id",
        "state_index",
        "repeat_index",
        "actions",
        "rewards",
        "steps_taken",
        "terminated",
        "bootstrapped",
        "k_step_return",
        "k_step_return_n",
    }
    for offset, output in sorted(runs.items()):
        if len(output.state_rows) != len(primary.state_rows):
            raise PersistentDiagnosticArtifactError(
                "Reference-depth runs retained different state counts."
            )
        if len(output.mc_rows) != len(primary.mc_rows):
            raise PersistentDiagnosticArtifactError(
                "Reference-depth runs produced different Monte Carlo row counts."
            )
        for left, right in zip(output.mc_rows, primary.mc_rows):
            for key in mc_common_keys:
                if left.get(key) != right.get(key):
                    raise PersistentDiagnosticArtifactError(
                        f"Common Monte Carlo field {key!r} changed with reference depth."
                    )
        if len(output.action_rows) != len(primary.action_rows):
            raise PersistentDiagnosticArtifactError(
                "Reference-depth runs produced different action-row counts."
            )
        reference_depth_m = int(output.summary["config"]["reference_depth_m"])
        expected_depth_rows = [
            row
            for row in primary.depth_rows
            if int(row["depth"]) <= reference_depth_m
        ]
        if output.depth_rows != expected_depth_rows:
            raise PersistentDiagnosticArtifactError(
                "A shorter reference-depth run is not the raw prefix of the "
                "primary depth rows."
            )
        for left, right in zip(output.state_rows, primary.state_rows):
            for key in state_common_keys.intersection(left, right):
                if left[key] != right[key]:
                    raise PersistentDiagnosticArtifactError(
                        f"Common state metric {key!r} changed with reference depth."
                    )
        for left, right in zip(output.action_rows, primary.action_rows):
            for key in action_common_keys:
                if left.get(key) != right.get(key):
                    raise PersistentDiagnosticArtifactError(
                        f"Common action metric {key!r} changed with reference depth."
                    )

    depth_summaries: dict[str, Any] = {}
    for offset, output in sorted(runs.items()):
        depth_summaries[str(offset)] = {
            "reference_depth_m": output.summary["config"]["reference_depth_m"],
            "finite_reference_depth_path": output.summary[
                "finite_reference_depth_path"
            ],
            "exact_one_step_reference_depth_residual": output.summary[
                "exact_one_step_reference_depth_residual"
            ],
            "monte_carlo_k_step_reference_depth_residual": output.summary[
                "monte_carlo_k_step_reference_depth_residual"
            ],
            "finite_depth_candidate_advantage_discrepancy": output.summary[
                "finite_depth_candidate_advantage_discrepancy"
            ],
            "remaining_budget_strata": output.summary["remaining_budget_strata"],
            "radial_projection_activation": output.summary[
                "radial_projection_activation"
            ],
        }
    primary.summary["registered_reference_depths"] = {
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "primary_offset": primary_offset,
        "offsets": depth_summaries,
    }

    for state_index, primary_row in enumerate(primary.state_rows):
        references: dict[str, Any] = {}
        for offset, output in sorted(runs.items()):
            row = output.state_rows[state_index]
            references[str(offset)] = {
                "reference_depth_m": output.summary["config"][
                    "reference_depth_m"
                ],
                "value_m": row["value_m"],
                "path_length_n_to_m": row["path_length_n_to_m"],
                "depth_value_discrepancy": row["depth_value_discrepancy"],
                "exact_one_step_reference_backup": row[
                    "exact_one_step_reference_backup"
                ],
                "exact_one_step_reference_signed_residual": row[
                    "exact_one_step_reference_signed_residual"
                ],
                "exact_one_step_reference_absolute_residual": row[
                    "exact_one_step_reference_absolute_residual"
                ],
                "mc_k_step_reference_operator_mean": row[
                    "mc_k_step_reference_operator_mean"
                ],
                "mc_k_step_reference_operator_standard_error": row[
                    "mc_k_step_reference_operator_standard_error"
                ],
                "mc_k_step_reference_signed_residual": row[
                    "mc_k_step_reference_signed_residual"
                ],
                "mc_k_step_reference_absolute_residual": row[
                    "mc_k_step_reference_absolute_residual"
                ],
                "finite_depth_candidate_advantage_discrepancy": row.get(
                    "finite_depth_candidate_advantage_discrepancy"
                ),
                "configured_estimator_candidate_advantage_discrepancy": row.get(
                    "configured_estimator_candidate_advantage_discrepancy"
                ),
                "projection_active_step_rate": row.get(
                    "projection_active_step_rate"
                ),
                "maximum_pre_projection_norm": row.get(
                    "maximum_pre_projection_norm"
                ),
            }
        primary_row["registered_reference_depths"] = references

    for mc_index, primary_row in enumerate(primary.mc_rows):
        references = {}
        for offset, output in sorted(runs.items()):
            row = output.mc_rows[mc_index]
            references[str(offset)] = {
                "reference_depth_m": output.summary["config"][
                    "reference_depth_m"
                ],
                "k_step_return_m": row["k_step_return_m"],
            }
        primary_row["registered_reference_depths"] = references

    for action_index, primary_row in enumerate(primary.action_rows):
        references = {}
        for offset, output in sorted(runs.items()):
            row = output.action_rows[action_index]
            references[str(offset)] = {
                "reference_depth_m": output.summary["config"][
                    "reference_depth_m"
                ],
                "q_m": row["q_m"],
                "raw_advantage_m": row["raw_advantage_m"],
                "configured_estimator_advantage_m": row[
                    "configured_estimator_advantage_m"
                ],
            }
        primary_row["registered_reference_depths"] = references
    return primary


def run_registered_diagnostics(
    context: PersistentDiagnosticContext,
    spec: Mapping[str, Any],
    *,
    diagnostic_spec_sha256: str,
    diagnostic_code_commit: str,
) -> tuple[PersistentDiagnosticOutput, PersistentStateCollection]:
    """Collect the registered held-out prefix and run the bounded kernel."""

    if not _COMMIT_RE.fullmatch(diagnostic_code_commit):
        raise PersistentDiagnosticArtifactError(
            "diagnostic_code_commit must be one full 40-character Git hash."
        )
    record_count = int(spec["record_selection"]["record_count"])
    eval_count = len(context.dataset.eval_dataset)
    if record_count > eval_count:
        raise PersistentDiagnosticArtifactError(
            "Registered record count exceeds the unique held-out pool; cycling is forbidden."
        )
    config = build_diagnostic_config(context, spec)
    collection = collect_persistent_diagnostic_states(
        current_policy=context.checkpoint.current_policy,
        env=context.dataset.environment,
        record_indices=list(range(record_count)),
        config=config,
    )
    provenance = {
        "checkpoint_sha256": context.checkpoint.checkpoint_sha256,
        "checkpoint_producer_code_commit": {
            "value": context.checkpoint.producer_code_commit,
            "status": "verified from artifact",
        },
        "checkpoint_lineage": {
            "parent_checkpoint_sha256": (
                context.checkpoint.parent_checkpoint_sha256
            ),
            "parent_checkpoint_step": context.checkpoint.parent_checkpoint_step,
            "parent_environment_interactions": (
                context.checkpoint.parent_environment_steps
            ),
            "status": "verified from artifact",
        },
        "dataset_manifest_sha256": context.dataset.manifest_sha256,
        "dataset_provenance_sha256": (
            context.checkpoint.dataset_provenance_sha256
        ),
        "eval_pool_ordered_sha256": context.dataset.eval_ordered_sha256,
        "diagnostic_spec_sha256": diagnostic_spec_sha256,
        "diagnostic_code_commit": {
            "value": diagnostic_code_commit.lower(),
            "status": UNVERIFIABLE_STATUS,
        },
    }
    deployment_runtime = SimpleNamespace(
        rl_cfg=context.checkpoint.rl_config,
        policy_model_old=context.checkpoint.current_policy,
        policy_model_candidate=context.checkpoint.candidate_policy,
    )
    production_mixed_policy_dist = getattr(
        UPITrmTrainer,
        "_mixed_policy_dist",
    )

    def production_deployment_dist(*args: Any, **kwargs: Any) -> Any:
        return production_mixed_policy_dist(
            deployment_runtime,
            *args,
            **kwargs,
        )

    runs: dict[int, PersistentDiagnosticOutput] = {}
    for offset in spec["reference_depth_offsets"]:
        run_config = build_diagnostic_config(
            context,
            spec,
            reference_depth_offset=int(offset),
        )
        runs[int(offset)] = run_persistent_diagnostics(
            evaluator=context.checkpoint.evaluator,
            current_policy=context.checkpoint.current_policy,
            candidate_policy=context.checkpoint.candidate_policy,
            deployment=DeploymentSpec(
                kind="policy_dist_callback",
                policy_dist_fn=production_deployment_dist,
                label="production_exact_probability_mixture_callback",
            ),
            env=context.dataset.environment,
            states=collection.states,
            config=run_config,
            endpoint_policy_pair_invariants_verified=(
                context.checkpoint.endpoint_policy_pair_invariants_verified
            ),
            provenance=provenance,
        )
    output = _merge_reference_depth_runs(runs)
    output.summary["state_collection"] = collection.metadata
    return output, collection


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("Artifact output contains NaN.")
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _normalize_json(value.tolist()),
        }
    if torch.is_tensor(value):
        cpu = value.detach().to(device="cpu").contiguous()
        return {
            "dtype": str(cpu.dtype).removeprefix("torch."),
            "shape": list(cpu.shape),
            "data": _normalize_json(cpu.tolist()),
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Artifact JSON objects require string keys.")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"Unsupported artifact value type {type(value).__name__}.")


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    normalized = _normalize_json(value)
    if pretty:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    return (encoded + "\n").encode("ascii")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row, pretty=False) for row in rows)


def _latent_payload(latent: Optional[Any]) -> Optional[dict[str, Any]]:
    if latent is None:
        return None
    return {"z_H": latent.z_H, "z_L": latent.z_L}


def augmented_state_rows(
    states: Sequence[AugmentedDiagnosticState],
) -> list[dict[str, Any]]:
    """Retain the actual augmented states used by the diagnostic kernel."""

    return [
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "scope": FINITE_BATCH_SCOPE,
            "state_index": state_index,
            "occurrence_id": state.occurrence_id,
            "augmented_state_hash": state.augmented_state_hash,
            "source_record_index": state.source_record_index,
            "episode_id": state.episode_id,
            "timestep": state.timestep,
            "environment_state": state.environment_state,
            "input_latent": _latent_payload(state.input_latent),
            "observed_post_unroll_latent": _latent_payload(
                state.observed_post_unroll_latent
            ),
            "observed_action": state.observed_action,
            "observed_reward": state.observed_reward,
            "observed_done": state.observed_done,
            "successor_environment_state": state.successor_environment_state,
            "next_input_latent": _latent_payload(state.next_input_latent),
            "next_input_remaining_edits": state.next_input_remaining_edits,
        }
        for state_index, state in enumerate(states)
    ]


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PersistentDiagnosticArtifactError(
            "Atomic no-replace directory publication is unavailable."
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise PersistentDiagnosticArtifactError(
            "Output directory appeared during publication; overwrite refused."
        )
    raise OSError(error, os.strerror(error), str(destination))


def _diagnostic_source_sha256s(
    context: PersistentDiagnosticContext,
) -> dict[str, str]:
    sources = {
        "artifact_runner": main,
        "checkpoint_loader": load_persistent_diagnostic_context,
        "diagnostic_kernel": run_persistent_diagnostics,
        "model_implementation": type(context.checkpoint.evaluator),
        "environment_transition": type(context.dataset.environment).step,
        "checker_implementation": context.dataset.checker,
        "heldout_dataset_implementation": type(context.dataset.eval_dataset),
        "production_deployment_distribution": UPITrmTrainer._mixed_policy_dist,
        "production_episode_evaluator": evaluate_plan_policy_with_scores,
        "production_exact_baseline": compute_exact_baseline_summation,
        "summary_helpers": summarize_finite_batch_metric,
    }
    result: dict[str, str] = {}
    for label, source_object in sources.items():
        source_path = inspect.getsourcefile(source_object)
        if source_path is None:
            raise PersistentDiagnosticArtifactError(
                f"Cannot locate packaged source for {label}."
            )
        result[label] = file_sha256(source_path)
    return result


def _repository_behavior_source_sha256s() -> dict[str, str]:
    """Hash an over-approximation of every repository-local behavior dependency."""

    runner_source = inspect.getsourcefile(main)
    if runner_source is None:
        raise PersistentDiagnosticArtifactError(
            "Cannot locate the diagnostic runner source tree."
        )
    source_root = Path(runner_source).resolve().parent.parent
    directory_names = ("dataset", "evaluators", "models", "rl", "scripts", "utils")
    relative_paths: set[Path] = set()
    for directory_name in directory_names:
        directory = source_root / directory_name
        if not directory.is_dir():
            raise PersistentDiagnosticArtifactError(
                f"Repository behavior source directory {directory_name!r} is missing."
            )
        relative_paths.update(
            path.relative_to(source_root)
            for path in directory.rglob("*.py")
            if path.is_file()
        )
    for file_name in ("BUCK", "puzzle_dataset.py", "upi_trm_train.py"):
        path = source_root / file_name
        if not path.is_file():
            raise PersistentDiagnosticArtifactError(
                f"Repository behavior source file {file_name!r} is missing."
            )
        relative_paths.add(Path(file_name))
    return {
        relative_path.as_posix(): file_sha256(source_root / relative_path)
        for relative_path in sorted(relative_paths, key=lambda path: path.as_posix())
    }


def _source_manifest_sha256(source_sha256s: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"upi-trm-repository-behavior-source-v1\0")
    for relative_path, source_sha256 in sorted(source_sha256s.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_artifact_payloads(payloads: Mapping[str, bytes]) -> None:
    for name, payload in payloads.items():
        if not payload:
            raise PersistentDiagnosticArtifactError(
                f"Artifact payload {name} is empty."
            )
        if name.endswith(".json"):
            parsed = _strict_json_loads(payload.decode("ascii"))
            if parsed.get("scope") != FINITE_BATCH_SCOPE:
                raise PersistentDiagnosticArtifactError(
                    f"Artifact payload {name} has the wrong scope."
                )
        elif name.endswith(".jsonl"):
            for line_number, line in enumerate(payload.splitlines(), start=1):
                parsed = _strict_json_loads(line.decode("ascii"))
                if parsed.get("scope") != FINITE_BATCH_SCOPE:
                    raise PersistentDiagnosticArtifactError(
                        f"Artifact row {name}:{line_number} has the wrong scope."
                    )

    manifest = _strict_json_loads(payloads["MANIFEST.json"].decode("ascii"))
    for name, expected in manifest["outputs"].items():
        actual = hashlib.sha256(payloads[name]).hexdigest()
        if actual != expected:
            raise PersistentDiagnosticArtifactError(
                f"Manifest hash mismatch for {name}."
            )
    checksum_lines = payloads["SHA256SUMS"].decode("ascii").splitlines()
    expected_lines = [
        f"{hashlib.sha256(payload).hexdigest()}  {name}"
        for name, payload in sorted(payloads.items())
        if name != "SHA256SUMS"
    ]
    if checksum_lines != expected_lines:
        raise PersistentDiagnosticArtifactError("SHA256SUMS is inconsistent.")

    combined = b"\n".join(payloads.values())
    forbidden = [
        b"/home/",
        b"/Users/",
        b"\\Users\\",
        b"/data/",
        b"/private/",
        b"file://",
        b"@",
    ]
    home = str(Path.home()).encode("utf-8")
    if home:
        forbidden.append(home)
    user = os.environ.get("USER", "").encode("utf-8")
    if user:
        forbidden.append(user)
    if any(token in combined for token in forbidden):
        raise PersistentDiagnosticArtifactError(
            "Identity-bearing path or user content survived artifact staging."
        )


def write_artifact_bundle(
    *,
    output_dir: str | Path,
    context: PersistentDiagnosticContext,
    spec: Mapping[str, Any],
    diagnostic_spec_sha256: str,
    diagnostic_code_commit: str,
    output: PersistentDiagnosticOutput,
    collection: PersistentStateCollection,
) -> dict[str, str]:
    """Write one deterministic, anonymous, non-overwriting artifact bundle."""

    loaded_source_sha256s = _diagnostic_source_sha256s(context)
    behavior_source_sha256s = _repository_behavior_source_sha256s()

    validation = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scope": FINITE_BATCH_SCOPE,
        "input_validation_status": "verified_for_diagnostic_inputs",
        "checkpoint": {
            "sha256": context.checkpoint.checkpoint_sha256,
            "schema_version": context.checkpoint.checkpoint_schema_version,
            "training_protocol": context.checkpoint.training_protocol,
            "checkpoint_step": context.checkpoint.checkpoint_step,
            "environment_interactions": context.checkpoint.environment_steps,
            "outer_steps": context.checkpoint.outer_steps,
            "optimizer_steps": {
                "value": context.checkpoint.value_optimizer_steps,
                "policy": context.checkpoint.policy_optimizer_steps,
                "distill": context.checkpoint.distill_optimizer_steps,
                "puzzle_embedding": context.checkpoint.puzzle_optimizer_steps,
            },
            "source_execution_device": context.checkpoint.source_execution_device,
            "replay_structural_inventory": {
                "transition_count": context.checkpoint.replay.transition_count,
                "episode_count": context.checkpoint.replay.episode_count,
                "terminal_transition_count": (
                    context.checkpoint.replay.terminal_transition_count
                ),
                "environment_reexecution_performed": False,
            },
            "recurrent_map_shared": context.checkpoint.recurrent_map_shared,
            "endpoint_policy_pair_invariants_verified": (
                context.checkpoint.endpoint_policy_pair_invariants_verified
            ),
            "producer_code_commit": {
                "value": context.checkpoint.producer_code_commit,
                "status": "verified from artifact",
            },
            "training_seed": {
                "value": context.checkpoint.training_seed,
                "status": "verified from artifact",
            },
            "run_id": {
                "value": context.checkpoint.run_id,
                "status": "verified from artifact",
            },
            "run_identity_sha256": context.checkpoint.run_identity_sha256,
            "effective_config_sha256": (
                context.checkpoint.effective_config_sha256
            ),
            "training_runtime_fingerprint_sha256": (
                context.checkpoint.runtime_fingerprint_sha256
            ),
            "initialization": {
                "kind": context.checkpoint.initialization_kind,
                "artifact_sha256": (
                    context.checkpoint.initialization_artifact_sha256
                ),
                "status": "verified from artifact",
            },
            "lineage": {
                "parent_checkpoint_sha256": (
                    context.checkpoint.parent_checkpoint_sha256
                ),
                "parent_checkpoint_step": (
                    context.checkpoint.parent_checkpoint_step
                ),
                "parent_environment_interactions": (
                    context.checkpoint.parent_environment_steps
                ),
                "status": "verified from artifact",
            },
            "optimizer_rng_and_live_trainer_restore": {
                "status": UNVERIFIABLE_STATUS,
            },
        },
        "dataset": {
            "manifest_sha256": context.dataset.manifest_sha256,
            "provenance_sha256": context.checkpoint.dataset_provenance_sha256,
            "train_pool_ordered_sha256": context.dataset.train_ordered_sha256,
            "eval_pool_ordered_sha256": context.dataset.eval_ordered_sha256,
            "train_unique_input_count": len(context.dataset.train_input_sha256s),
            "eval_unique_input_count": len(context.dataset.eval_input_sha256s),
            "train_eval_input_overlap_count": 0,
            "checker_kind": context.dataset.checker_kind,
            "selected_input_sha256s": list(
                context.dataset.eval_input_sha256s[
                    : int(spec["record_selection"]["record_count"])
                ]
            ),
        },
        "diagnostic": {
            "spec_sha256": diagnostic_spec_sha256,
            "code_commit": {
                "value": diagnostic_code_commit.lower(),
                "status": UNVERIFIABLE_STATUS,
            },
            "loaded_source_sha256s": loaded_source_sha256s,
            "repository_behavior_source_manifest": {
                "schema": "upi-trm-repository-behavior-source-v1",
                "aggregate_sha256": _source_manifest_sha256(
                    behavior_source_sha256s
                ),
                "files": behavior_source_sha256s,
            },
            "scope": FINITE_BATCH_SCOPE,
            "uniform_certificate": False,
            "deployment_kind": context.checkpoint.deployment_kind,
            "record_selection": spec["record_selection"],
            "retained_state_count": len(collection.states),
            "state_occurrence_ids_sha256": output.summary[
                "state_occurrence_ids_sha256"
            ],
            "runtime": {
                "device": str(next(context.checkpoint.evaluator.parameters()).device),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": torch.version.cuda,
                "deterministic_algorithms_enabled": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "torch_num_threads": torch.get_num_threads(),
            },
            "model_state_identities": {
                "evaluator": {
                    "full_state_sha256": (
                        context.checkpoint.evaluator_identity.full_state_sha256
                    ),
                    "recurrent_map_sha256": (
                        context.checkpoint.evaluator_identity.recurrent_map_sha256
                    ),
                    "non_edit_state_sha256": (
                        context.checkpoint.evaluator_identity.non_edit_state_sha256
                    ),
                },
                "current_policy": {
                    "full_state_sha256": (
                        context.checkpoint.current_policy_identity.full_state_sha256
                    ),
                    "recurrent_map_sha256": (
                        context.checkpoint.current_policy_identity.recurrent_map_sha256
                    ),
                    "non_edit_state_sha256": (
                        context.checkpoint.current_policy_identity.non_edit_state_sha256
                    ),
                },
                "candidate_policy": {
                    "full_state_sha256": (
                        context.checkpoint.candidate_policy_identity.full_state_sha256
                    ),
                    "recurrent_map_sha256": (
                        context.checkpoint.candidate_policy_identity.recurrent_map_sha256
                    ),
                    "non_edit_state_sha256": (
                        context.checkpoint.candidate_policy_identity.non_edit_state_sha256
                    ),
                },
            },
            "omitted_metrics": output.summary["omitted_metrics"],
        },
    }
    payloads: dict[str, bytes] = {
        "validation.json": _json_bytes(validation, pretty=True),
        "summary.json": _json_bytes(output.summary, pretty=True),
        "per_state.jsonl": _jsonl_bytes(output.state_rows),
        "per_action.jsonl": _jsonl_bytes(output.action_rows),
        "depth_rows.jsonl": _jsonl_bytes(output.depth_rows),
        "mc_rows.jsonl": _jsonl_bytes(output.mc_rows),
        "augmented_states.jsonl": _jsonl_bytes(
            augmented_state_rows(collection.states)
        ),
    }
    data_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scope": FINITE_BATCH_SCOPE,
        "artifact_kind": "persistent_checkpoint_finite_batch_diagnostics",
        "deterministic_encoding": "sorted-key ASCII strict JSON v1",
        "checkpoint_sha256": context.checkpoint.checkpoint_sha256,
        "checkpoint_producer_code_commit": {
            "value": context.checkpoint.producer_code_commit,
            "status": "verified from artifact",
        },
        "checkpoint_run_identity_sha256": (
            context.checkpoint.run_identity_sha256
        ),
        "checkpoint_effective_config_sha256": (
            context.checkpoint.effective_config_sha256
        ),
        "checkpoint_runtime_fingerprint_sha256": (
            context.checkpoint.runtime_fingerprint_sha256
        ),
        "checkpoint_lineage": {
            "parent_checkpoint_sha256": (
                context.checkpoint.parent_checkpoint_sha256
            ),
            "parent_checkpoint_step": context.checkpoint.parent_checkpoint_step,
            "parent_environment_interactions": (
                context.checkpoint.parent_environment_steps
            ),
            "status": "verified from artifact",
        },
        "dataset_manifest_sha256": context.dataset.manifest_sha256,
        "dataset_provenance_sha256": (
            context.checkpoint.dataset_provenance_sha256
        ),
        "eval_pool_ordered_sha256": context.dataset.eval_ordered_sha256,
        "diagnostic_spec_sha256": diagnostic_spec_sha256,
        "diagnostic_code_commit": {
            "value": diagnostic_code_commit.lower(),
            "status": UNVERIFIABLE_STATUS,
        },
        "diagnostic_loaded_source_sha256s": loaded_source_sha256s,
        "repository_behavior_source_aggregate_sha256": (
            _source_manifest_sha256(behavior_source_sha256s)
        ),
        "rl_config_sha256": context.checkpoint.rl_config_sha256,
        "model_config_sha256": context.checkpoint.model_config_sha256,
        "outputs": data_hashes,
    }
    manifest_payload = _json_bytes(manifest, pretty=True)
    payloads["MANIFEST.json"] = manifest_payload
    all_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }
    payloads["SHA256SUMS"] = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())
    ).encode("ascii")
    _validate_artifact_payloads(payloads)

    destination = Path(output_dir).expanduser().absolute()
    if destination.exists():
        raise PersistentDiagnosticArtifactError(
            "Output directory already exists; artifact overwrite is forbidden."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        raise PersistentDiagnosticArtifactError(
            "Artifact staging directory already exists."
        )
    staging.mkdir()
    try:
        for name, payload in sorted(payloads.items()):
            _atomic_write(staging / name, payload)
        on_disk = {
            path.name: path.read_bytes()
            for path in staging.iterdir()
            if path.is_file()
        }
        _validate_artifact_payloads(on_disk)
        _publish_directory_no_replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded diagnostics on one fixed-base persistent checkpoint."
    )
    parser.add_argument(
        "--trusted-local-checkpoint",
        action="store_true",
        required=True,
        help="Acknowledge that torch.load will unpickle this local checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument(
        "--dataset-paths",
        "--dataset-path",
        action="append",
        required=True,
        help="Materialized dataset root; repeat for each manifest source.",
    )
    parser.add_argument("--diagnostic-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--diagnostic-code-commit", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    destination = Path(args.output_dir).expanduser().absolute()
    if destination.exists():
        raise PersistentDiagnosticArtifactError(
            "Output directory already exists; artifact overwrite is forbidden."
        )
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    spec, spec_sha256 = load_diagnostic_spec(args.diagnostic_spec)
    context = load_persistent_diagnostic_context(
        args.checkpoint,
        args.dataset_manifest,
        args.dataset_paths,
        device=args.device,
    )
    output, collection = run_registered_diagnostics(
        context,
        spec,
        diagnostic_spec_sha256=spec_sha256,
        diagnostic_code_commit=args.diagnostic_code_commit,
    )
    write_artifact_bundle(
        output_dir=args.output_dir,
        context=context,
        spec=spec,
        diagnostic_spec_sha256=spec_sha256,
        diagnostic_code_commit=args.diagnostic_code_commit,
        output=output,
        collection=collection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
