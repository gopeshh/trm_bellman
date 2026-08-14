#!/usr/bin/env python3
"""Publication schema for newly generated Phase 4 evaluation summaries."""

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, FrozenSet, List


PHASE4_SCHEMA_VERSION = 4
PHASE4_CHECKPOINT_STEP = 5000
PHASE4_CHECKPOINT_SCHEMA_VERSION = 4
PHASE4_TRAINING_INVOCATION_SCHEMA_VERSION = 3
PHASE4_DIAGNOSTIC_DATASET_NAME = "sudoku-4x4-trivial"
PHASE4_DIAGNOSTIC_SPLITS = ("train", "test")
PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT = 50
PHASE4_DIAGNOSTIC_STATE_COUNT = 100
PHASE4_LIPSCHITZ_SAMPLE_COUNT = 150
PHASE4_STABILITY_SAMPLE_COUNT = PHASE4_DIAGNOSTIC_STATE_COUNT
PHASE4_PROJECTION_SAMPLE_COUNT = PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT
PHASE4_LIPSCHITZ_PERTURBATION_SEED = 20260813
PHASE4_LIPSCHITZ_PERTURBATION_SCHEME = (
    "private_cpu_torch_generator_joint_normal_v1"
)

PHASE4_CONDITION_SPECS = {
    "nc_nv": {
        "label": "C-OFF, V-OFF",
        "enable_contraction": False,
        "disable_value_head_norm": True,
    },
    "nc_yv": {
        "label": "C-OFF, V-ON",
        "enable_contraction": False,
        "disable_value_head_norm": False,
    },
    "yc_nv": {
        "label": "C-ON, V-OFF",
        "enable_contraction": True,
        "disable_value_head_norm": True,
    },
    "yc_yv": {
        "label": "C-ON, V-ON",
        "enable_contraction": True,
        "disable_value_head_norm": False,
    },
}
PHASE4_SEEDS = frozenset({41, 42, 43})
PHASE4_PROJECTION_MODE = "enabled"
PHASE4_PROJECTION_RADIUS = 10.0


def phase4_run_id(condition: str, seed: int) -> str:
    """Return the checkpoint-bound run identifier for one design cell."""

    return f"phase4_2x2_norm_ablation.{condition}.seed{seed}"


def phase4_checkpoint_relpath(condition: str, seed: int) -> str:
    """Return the required full-checkpoint path for one design cell."""

    return f"{condition}_s{seed}/rl_checkpoint_step_{PHASE4_CHECKPOINT_STEP}.pt"

PHASE4_METRIC_AVAILABILITY = {
    "success_trivial": {
        "status": "unavailable",
        "reason": "no_environment_rollout",
    },
    "success_hard": {
        "status": "unavailable",
        "reason": "no_environment_rollout",
    },
    "final_loss": {
        "status": "unavailable",
        "reason": "training_log_not_loaded",
    },
    "has_nan": {
        "status": "unavailable",
        "reason": "training_history_not_loaded",
    },
}

_TOP_LEVEL_FIELDS: FrozenSet[str] = frozenset(
    {
        "schema_version",
        "metric_availability",
        "experiment",
        "description",
        "generated_at",
        "evaluator_git_commit",
        "evaluator_source_manifest_sha256",
        "evaluator_runtime_artifact_sha256",
        "diagnostic_dataset",
        "diagnostic_dataset_sha256",
        "lipschitz_perturbation_seed",
        "lipschitz_perturbation_scheme",
        "conditions",
        "seeds",
        "all_results",
        "aggregates",
    }
)

_RUN_FIELDS: FrozenSet[str] = frozenset(
    {
        "condition",
        "seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "model_state_sha256",
        "checkpoint_step",
        "training_run_id",
        "config_sha256",
        "rl_config_sha256",
        "model_config_sha256",
        "dataset_provenance_sha256",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "training_runtime_artifact_sha256",
        "initialization_kind",
        "checkpoint_schema_version",
        "training_invocation_schema_version",
        "enable_contraction",
        "disable_value_head_norm",
        "latent_projection_mode",
        "latent_ball_radius",
        "L_preproj",
        "L_preproj_std",
        "var_V",
        "projection_active_rate",
        "argmax_agreement_n4",
        "argmax_agreement_n8",
        "delta_V_n4",
        "delta_V_n8",
        "lipschitz_sample_count",
        "stability_sample_count",
        "projection_sample_count",
    }
)

_AGGREGATE_FIELDS: FrozenSet[str] = frozenset(
    {
        "condition",
        "label",
        "enable_contraction",
        "disable_value_head_norm",
        "latent_projection_mode",
        "latent_ball_radius",
        "n_seeds",
        "L_preproj_mean",
        "L_preproj_std",
        "var_V_mean",
        "var_V_std",
        "projection_active_rate_mean",
        "argmax_n4_mean",
        "argmax_n4_std",
        "argmax_n8_mean",
        "argmax_n8_std",
        "delta_V_n4_mean",
        "delta_V_n8_mean",
    }
)

_RETIRED_RUN_FIELDS: FrozenSet[str] = frozenset(
    {"success_trivial", "success_hard", "final_loss", "has_nan"}
)
_RETIRED_AGGREGATE_FIELDS: FrozenSet[str] = frozenset(
    {
        "success_trivial_mean",
        "success_trivial_std",
        "success_hard_mean",
        "success_hard_std",
        "nan_count",
    }
)


class Phase4SummaryValidationError(ValueError):
    """Raised when a Phase 4 summary is not publishable schema version 4."""


def phase4_metric_availability() -> Dict[str, Dict[str, str]]:
    """Return fresh JSON-compatible metadata for metrics this evaluator omits."""
    return {
        metric: dict(metadata)
        for metric, metadata in PHASE4_METRIC_AVAILABILITY.items()
    }


def _require_exact_fields(
    value: Dict[str, Any], expected: FrozenSet[str], path: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise Phase4SummaryValidationError(
            f"{path} has missing fields {missing} and unexpected fields {unexpected}"
        )


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase4SummaryValidationError(f"{path} must be a non-empty string")
    return value


def _require_sha256(value: Any, path: str) -> str:
    result = _require_string(value, path)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise Phase4SummaryValidationError(
            f"{path} must be a 64-character lowercase SHA-256"
        )
    return result


def _require_git_commit(value: Any, path: str) -> str:
    result = _require_string(value, path)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise Phase4SummaryValidationError(
            f"{path} must be a 40-character lowercase Git commit"
        )
    return result


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise Phase4SummaryValidationError(f"{path} must be a boolean")
    return value


def _require_int(value: Any, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Phase4SummaryValidationError(f"{path} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise Phase4SummaryValidationError(f"{path} must be positive")
    return result


def _require_number(
    value: Any,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Phase4SummaryValidationError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Phase4SummaryValidationError(f"{path} must be finite")
    if result < minimum or result > maximum:
        raise Phase4SummaryValidationError(
            f"{path}={result} is outside [{minimum}, {maximum}]"
        )
    return result


def _validate_projection_fields(value: Dict[str, Any], path: str) -> None:
    mode = value["latent_projection_mode"]
    radius = value["latent_ball_radius"]
    if mode == "enabled":
        _require_number(radius, f"{path}.latent_ball_radius", minimum=0.0)
        if float(radius) == 0.0:
            raise Phase4SummaryValidationError(
                f"{path}.latent_ball_radius must be positive when projection is enabled"
            )
    elif mode == "disabled":
        if radius is not None:
            raise Phase4SummaryValidationError(
                f"{path}.latent_ball_radius must be null when projection is disabled"
            )
    else:
        raise Phase4SummaryValidationError(
            f"{path}.latent_projection_mode must be 'enabled' or 'disabled'"
        )


def _validate_diagnostic_dataset(value: Any) -> Dict[str, Any]:
    path = "diagnostic_dataset"
    if not isinstance(value, dict):
        raise Phase4SummaryValidationError(f"{path} must be an object")
    _require_exact_fields(
        value,
        frozenset(
            {
                "schema_version",
                "dataset_name",
                "selection",
                "records_per_split",
                "total_selected_records",
                "splits",
                "ordered_states_sha256",
            }
        ),
        path,
    )
    if _require_int(value["schema_version"], f"{path}.schema_version") != 1:
        raise Phase4SummaryValidationError(
            f"{path}.schema_version must be 1"
        )
    if value["dataset_name"] != PHASE4_DIAGNOSTIC_DATASET_NAME:
        raise Phase4SummaryValidationError(
            f"{path}.dataset_name must be {PHASE4_DIAGNOSTIC_DATASET_NAME!r}"
        )
    if value["selection"] != "first_n_in_file_order":
        raise Phase4SummaryValidationError(
            f"{path}.selection must be 'first_n_in_file_order'"
        )
    if _require_int(
        value["records_per_split"], f"{path}.records_per_split"
    ) != PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT:
        raise Phase4SummaryValidationError(
            f"{path}.records_per_split must be "
            f"{PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT}"
        )
    if _require_int(
        value["total_selected_records"], f"{path}.total_selected_records"
    ) != PHASE4_DIAGNOSTIC_STATE_COUNT:
        raise Phase4SummaryValidationError(
            f"{path}.total_selected_records must be {PHASE4_DIAGNOSTIC_STATE_COUNT}"
        )
    _require_sha256(value["ordered_states_sha256"], f"{path}.ordered_states_sha256")

    splits = value["splits"]
    if not isinstance(splits, list) or len(splits) != len(PHASE4_DIAGNOSTIC_SPLITS):
        raise Phase4SummaryValidationError(
            f"{path}.splits must contain the exact train and test entries"
        )
    file_fields = frozenset({"relative_path", "bytes", "sha256", "dtype", "shape"})
    split_fields = frozenset(
        {
            "name",
            "available_records",
            "selected_records",
            "inputs",
            "puzzle_identifiers",
            "selected_records_sha256",
        }
    )
    for index, expected_name in enumerate(PHASE4_DIAGNOSTIC_SPLITS):
        split = splits[index]
        split_path = f"{path}.splits[{index}]"
        if not isinstance(split, dict):
            raise Phase4SummaryValidationError(f"{split_path} must be an object")
        _require_exact_fields(split, split_fields, split_path)
        if split["name"] != expected_name:
            raise Phase4SummaryValidationError(
                f"{split_path}.name must be {expected_name!r}"
            )
        available_records = _require_int(
            split["available_records"],
            f"{split_path}.available_records",
            positive=True,
        )
        if available_records < PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT:
            raise Phase4SummaryValidationError(
                f"{split_path}.available_records is too small"
            )
        if _require_int(
            split["selected_records"], f"{split_path}.selected_records"
        ) != PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT:
            raise Phase4SummaryValidationError(
                f"{split_path}.selected_records must be "
                f"{PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT}"
            )
        _require_sha256(
            split["selected_records_sha256"],
            f"{split_path}.selected_records_sha256",
        )
        for field, expected_shape, expected_filename in (
            ("inputs", [available_records, 16], "all__inputs.npy"),
            (
                "puzzle_identifiers",
                [available_records],
                "all__puzzle_identifiers.npy",
            ),
        ):
            file_identity = split[field]
            file_path = f"{split_path}.{field}"
            if not isinstance(file_identity, dict):
                raise Phase4SummaryValidationError(f"{file_path} must be an object")
            _require_exact_fields(file_identity, file_fields, file_path)
            expected_relative_path = (
                f"{PHASE4_DIAGNOSTIC_DATASET_NAME}/{expected_name}/"
                f"{expected_filename}"
            )
            if file_identity["relative_path"] != expected_relative_path:
                raise Phase4SummaryValidationError(
                    f"{file_path}.relative_path must be {expected_relative_path!r}"
                )
            _require_int(file_identity["bytes"], f"{file_path}.bytes", positive=True)
            _require_sha256(file_identity["sha256"], f"{file_path}.sha256")
            _require_string(file_identity["dtype"], f"{file_path}.dtype")
            if file_identity["shape"] != expected_shape:
                raise Phase4SummaryValidationError(
                    f"{file_path}.shape must be {expected_shape}"
                )
    return value


def _validate_run(
    run: Any,
    index: int,
    conditions: FrozenSet[str],
    seeds: FrozenSet[int],
) -> tuple[str, int]:
    path = f"all_results[{index}]"
    if not isinstance(run, dict):
        raise Phase4SummaryValidationError(f"{path} must be an object")
    _require_exact_fields(run, _RUN_FIELDS, path)
    retired = sorted(_RETIRED_RUN_FIELDS.intersection(run))
    if retired:
        raise Phase4SummaryValidationError(f"{path} contains retired fields {retired}")

    condition = _require_string(run["condition"], f"{path}.condition")
    seed = _require_int(run["seed"], f"{path}.seed")
    if condition not in conditions:
        raise Phase4SummaryValidationError(
            f"{path}.condition={condition!r} is not declared in conditions"
        )
    if seed not in seeds:
        raise Phase4SummaryValidationError(
            f"{path}.seed={seed} is not declared in seeds"
        )
    checkpoint_path = _require_string(
        run["checkpoint_path"], f"{path}.checkpoint_path"
    )
    expected_checkpoint_path = phase4_checkpoint_relpath(condition, seed)
    if checkpoint_path != expected_checkpoint_path:
        raise Phase4SummaryValidationError(
            f"{path}.checkpoint_path must be {expected_checkpoint_path!r}"
        )
    for field in (
        "checkpoint_sha256",
        "model_state_sha256",
        "config_sha256",
        "rl_config_sha256",
        "model_config_sha256",
        "dataset_provenance_sha256",
        "producer_source_manifest_sha256",
        "training_runtime_artifact_sha256",
    ):
        _require_sha256(run[field], f"{path}.{field}")
    _require_git_commit(run["producer_git_commit"], f"{path}.producer_git_commit")
    if run["initialization_kind"] != "random":
        raise Phase4SummaryValidationError(
            f"{path}.initialization_kind must be 'random'"
        )
    checkpoint_step = _require_int(run["checkpoint_step"], f"{path}.checkpoint_step")
    if checkpoint_step != PHASE4_CHECKPOINT_STEP:
        raise Phase4SummaryValidationError(
            f"{path}.checkpoint_step must be {PHASE4_CHECKPOINT_STEP}"
        )
    expected_run_id = phase4_run_id(condition, seed)
    if _require_string(run["training_run_id"], f"{path}.training_run_id") != expected_run_id:
        raise Phase4SummaryValidationError(
            f"{path}.training_run_id must be {expected_run_id!r}"
        )
    checkpoint_schema_version = _require_int(
        run["checkpoint_schema_version"],
        f"{path}.checkpoint_schema_version",
    )
    if checkpoint_schema_version != PHASE4_CHECKPOINT_SCHEMA_VERSION:
        raise Phase4SummaryValidationError(
            f"{path}.checkpoint_schema_version must be "
            f"{PHASE4_CHECKPOINT_SCHEMA_VERSION}"
        )
    invocation_schema_version = _require_int(
        run["training_invocation_schema_version"],
        f"{path}.training_invocation_schema_version",
    )
    if invocation_schema_version != PHASE4_TRAINING_INVOCATION_SCHEMA_VERSION:
        raise Phase4SummaryValidationError(
            f"{path}.training_invocation_schema_version must be "
            f"{PHASE4_TRAINING_INVOCATION_SCHEMA_VERSION}"
        )
    _require_bool(run["enable_contraction"], f"{path}.enable_contraction")
    _require_bool(
        run["disable_value_head_norm"], f"{path}.disable_value_head_norm"
    )
    _validate_projection_fields(run, path)
    if (
        run["latent_projection_mode"] != PHASE4_PROJECTION_MODE
        or float(run["latent_ball_radius"]) != PHASE4_PROJECTION_RADIUS
    ):
        raise Phase4SummaryValidationError(
            f"{path} does not match the Phase 4 projection configuration"
        )

    expected = PHASE4_CONDITION_SPECS[condition]
    for field in ("enable_contraction", "disable_value_head_norm"):
        if run[field] != expected[field]:
            raise Phase4SummaryValidationError(
                f"{path}.{field} does not match condition {condition}"
            )

    for field in ("L_preproj", "L_preproj_std", "var_V", "delta_V_n4", "delta_V_n8"):
        _require_number(run[field], f"{path}.{field}")
    for field in (
        "projection_active_rate",
        "argmax_agreement_n4",
        "argmax_agreement_n8",
    ):
        _require_number(run[field], f"{path}.{field}", maximum=1.0)
    expected_sample_counts = {
        "lipschitz_sample_count": PHASE4_LIPSCHITZ_SAMPLE_COUNT,
        "stability_sample_count": PHASE4_STABILITY_SAMPLE_COUNT,
        "projection_sample_count": PHASE4_PROJECTION_SAMPLE_COUNT,
    }
    for field, expected_count in expected_sample_counts.items():
        observed_count = _require_int(
            run[field],
            f"{path}.{field}",
            positive=True,
        )
        if observed_count != expected_count:
            raise Phase4SummaryValidationError(
                f"{path}.{field} must be {expected_count}"
            )
    return condition, seed


def _validate_aggregate(
    aggregate: Any, index: int, conditions: FrozenSet[str]
) -> str:
    path = f"aggregates[{index}]"
    if not isinstance(aggregate, dict):
        raise Phase4SummaryValidationError(f"{path} must be an object")
    _require_exact_fields(aggregate, _AGGREGATE_FIELDS, path)
    retired = sorted(_RETIRED_AGGREGATE_FIELDS.intersection(aggregate))
    if retired:
        raise Phase4SummaryValidationError(f"{path} contains retired fields {retired}")

    condition = _require_string(aggregate["condition"], f"{path}.condition")
    if condition not in conditions:
        raise Phase4SummaryValidationError(
            f"{path}.condition={condition!r} is not declared in conditions"
        )
    _require_string(aggregate["label"], f"{path}.label")
    _require_bool(
        aggregate["enable_contraction"], f"{path}.enable_contraction"
    )
    _require_bool(
        aggregate["disable_value_head_norm"],
        f"{path}.disable_value_head_norm",
    )
    _require_int(aggregate["n_seeds"], f"{path}.n_seeds", positive=True)
    _validate_projection_fields(aggregate, path)

    for field in (
        "L_preproj_mean",
        "L_preproj_std",
        "var_V_mean",
        "var_V_std",
        "delta_V_n4_mean",
        "delta_V_n8_mean",
    ):
        _require_number(aggregate[field], f"{path}.{field}")
    for field in (
        "projection_active_rate_mean",
        "argmax_n4_mean",
        "argmax_n4_std",
        "argmax_n8_mean",
        "argmax_n8_std",
    ):
        _require_number(aggregate[field], f"{path}.{field}", maximum=1.0)
    return condition


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: List[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _require_close(actual: Any, expected: float, path: str) -> None:
    value = _require_number(actual, path)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise Phase4SummaryValidationError(
            f"{path}={value} does not match the value recomputed from all_results "
            f"({expected})"
        )


def _validate_aggregate_consistency(
    aggregate: Dict[str, Any], runs: List[Dict[str, Any]], index: int
) -> None:
    path = f"aggregates[{index}]"
    condition = aggregate["condition"]
    condition_runs = [run for run in runs if run["condition"] == condition]
    if len(condition_runs) != len(PHASE4_SEEDS):
        raise Phase4SummaryValidationError(
            f"{path} does not have exactly {len(PHASE4_SEEDS)} source runs"
        )

    expected = PHASE4_CONDITION_SPECS[condition]
    if aggregate["label"] != expected["label"]:
        raise Phase4SummaryValidationError(
            f"{path}.label does not match condition {condition}"
        )
    for field in ("enable_contraction", "disable_value_head_norm"):
        if aggregate[field] != expected[field]:
            raise Phase4SummaryValidationError(
                f"{path}.{field} does not match condition {condition}"
            )
    if aggregate["n_seeds"] != len(PHASE4_SEEDS):
        raise Phase4SummaryValidationError(
            f"{path}.n_seeds must be {len(PHASE4_SEEDS)}"
        )
    for field in ("latent_projection_mode", "latent_ball_radius"):
        if any(run[field] != aggregate[field] for run in condition_runs):
            raise Phase4SummaryValidationError(
                f"{path}.{field} does not match every source run"
            )

    mean_fields = {
        "L_preproj_mean": "L_preproj",
        "var_V_mean": "var_V",
        "projection_active_rate_mean": "projection_active_rate",
        "argmax_n4_mean": "argmax_agreement_n4",
        "argmax_n8_mean": "argmax_agreement_n8",
        "delta_V_n4_mean": "delta_V_n4",
        "delta_V_n8_mean": "delta_V_n8",
    }
    std_fields = {
        "L_preproj_std": "L_preproj",
        "var_V_std": "var_V",
        "argmax_n4_std": "argmax_agreement_n4",
        "argmax_n8_std": "argmax_agreement_n8",
    }
    for aggregate_field, run_field in mean_fields.items():
        values = [float(run[run_field]) for run in condition_runs]
        _require_close(aggregate[aggregate_field], _mean(values), f"{path}.{aggregate_field}")
    for aggregate_field, run_field in std_fields.items():
        values = [float(run[run_field]) for run in condition_runs]
        _require_close(
            aggregate[aggregate_field],
            _population_std(values),
            f"{path}.{aggregate_field}",
        )


def validate_phase4_summary(summary: Any) -> None:
    """Validate a newly generated, publication-eligible Phase 4 summary.

    Schema-less and all earlier-schema files remain historical artifacts. They
    are not migrated because a legacy zero cannot be distinguished from a
    measurement.
    """
    if not isinstance(summary, dict):
        raise Phase4SummaryValidationError("summary must be an object")
    _require_exact_fields(summary, _TOP_LEVEL_FIELDS, "summary")
    schema_version = _require_int(summary["schema_version"], "schema_version")
    if schema_version != PHASE4_SCHEMA_VERSION:
        raise Phase4SummaryValidationError(
            f"schema_version must be {PHASE4_SCHEMA_VERSION}; schema-less and "
            "legacy summaries are historical and non-publishable"
        )
    if summary["metric_availability"] != PHASE4_METRIC_AVAILABILITY:
        raise Phase4SummaryValidationError(
            "metric_availability must exactly identify the four unavailable metrics"
        )

    for field in (
        "experiment",
        "description",
        "generated_at",
    ):
        _require_string(summary[field], field)
    _require_git_commit(summary["evaluator_git_commit"], "evaluator_git_commit")
    _require_sha256(
        summary["evaluator_source_manifest_sha256"],
        "evaluator_source_manifest_sha256",
    )
    _require_sha256(
        summary["evaluator_runtime_artifact_sha256"],
        "evaluator_runtime_artifact_sha256",
    )
    diagnostic_dataset = _validate_diagnostic_dataset(summary["diagnostic_dataset"])
    diagnostic_dataset_sha256 = _require_sha256(
        summary["diagnostic_dataset_sha256"], "diagnostic_dataset_sha256"
    )
    encoded_diagnostic_dataset = json.dumps(
        diagnostic_dataset,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if hashlib.sha256(encoded_diagnostic_dataset).hexdigest() != (
        diagnostic_dataset_sha256
    ):
        raise Phase4SummaryValidationError(
            "diagnostic_dataset_sha256 does not match diagnostic_dataset"
        )
    if _require_int(
        summary["lipschitz_perturbation_seed"],
        "lipschitz_perturbation_seed",
    ) != PHASE4_LIPSCHITZ_PERTURBATION_SEED:
        raise Phase4SummaryValidationError(
            "lipschitz_perturbation_seed does not match the registered seed"
        )
    if summary["lipschitz_perturbation_scheme"] != (
        PHASE4_LIPSCHITZ_PERTURBATION_SCHEME
    ):
        raise Phase4SummaryValidationError(
            "lipschitz_perturbation_scheme does not match the registered scheme"
        )

    conditions_value = summary["conditions"]
    if not isinstance(conditions_value, list) or not conditions_value:
        raise Phase4SummaryValidationError("conditions must be a non-empty list")
    conditions_list: List[str] = [
        _require_string(value, f"conditions[{index}]")
        for index, value in enumerate(conditions_value)
    ]
    conditions = frozenset(conditions_list)
    if len(conditions) != len(conditions_list):
        raise Phase4SummaryValidationError("conditions must not contain duplicates")
    if conditions != frozenset(PHASE4_CONDITION_SPECS):
        raise Phase4SummaryValidationError(
            "conditions must contain the exact four Phase 4 conditions"
        )

    seeds_value = summary["seeds"]
    if not isinstance(seeds_value, list) or not seeds_value:
        raise Phase4SummaryValidationError("seeds must be a non-empty list")
    seeds_list: List[int] = [
        _require_int(value, f"seeds[{index}]")
        for index, value in enumerate(seeds_value)
    ]
    seeds = frozenset(seeds_list)
    if len(seeds) != len(seeds_list):
        raise Phase4SummaryValidationError("seeds must not contain duplicates")
    if seeds != PHASE4_SEEDS:
        raise Phase4SummaryValidationError(
            "seeds must contain exactly 41, 42, and 43"
        )

    all_results = summary["all_results"]
    if not isinstance(all_results, list) or not all_results:
        raise Phase4SummaryValidationError("all_results must be a non-empty list")
    run_keys = [
        _validate_run(run, index, conditions, seeds)
        for index, run in enumerate(all_results)
    ]
    if len(set(run_keys)) != len(run_keys):
        raise Phase4SummaryValidationError(
            "all_results contains duplicate condition/seed pairs"
        )
    expected_run_keys = {
        (condition, seed) for condition in conditions for seed in seeds
    }
    if set(run_keys) != expected_run_keys:
        raise Phase4SummaryValidationError(
            "all_results must contain the complete four-condition by three-seed design"
        )
    for digest_field in ("checkpoint_sha256", "model_state_sha256"):
        digests = [run[digest_field] for run in all_results]
        if len(set(digests)) != len(digests):
            raise Phase4SummaryValidationError(
                f"all_results must not reuse a {digest_field} across design cells"
            )
    for condition in conditions:
        for digest_field in (
            "config_sha256",
            "rl_config_sha256",
            "model_config_sha256",
        ):
            digests = {
                run[digest_field]
                for run in all_results
                if run["condition"] == condition
            }
            if len(digests) != 1:
                raise Phase4SummaryValidationError(
                    f"all_results for {condition} must share one {digest_field}"
                )
    producer_commits = {run["producer_git_commit"] for run in all_results}
    producer_manifests = {
        run["producer_source_manifest_sha256"] for run in all_results
    }
    if len(producer_commits) != 1 or len(producer_manifests) != 1:
        raise Phase4SummaryValidationError(
            "all_results must share one clean producer commit and source manifest"
        )
    training_runtime_digests = {
        run["training_runtime_artifact_sha256"] for run in all_results
    }
    if len(training_runtime_digests) != 1:
        raise Phase4SummaryValidationError(
            "all_results must share one training runtime artifact SHA-256"
        )
    for seed in seeds:
        dataset_digests = {
            run["dataset_provenance_sha256"]
            for run in all_results
            if run["seed"] == seed
        }
        if len(dataset_digests) != 1:
            raise Phase4SummaryValidationError(
                f"all_results for seed {seed} must share one dataset provenance"
            )
    run_counts: Dict[str, int] = {}
    for condition, _ in run_keys:
        run_counts[condition] = run_counts.get(condition, 0) + 1

    aggregates = summary["aggregates"]
    if not isinstance(aggregates, list) or not aggregates:
        raise Phase4SummaryValidationError("aggregates must be a non-empty list")
    aggregate_conditions = [
        _validate_aggregate(aggregate, index, conditions)
        for index, aggregate in enumerate(aggregates)
    ]
    if len(set(aggregate_conditions)) != len(aggregate_conditions):
        raise Phase4SummaryValidationError(
            "aggregates contains duplicate conditions"
        )
    if set(aggregate_conditions) != conditions:
        raise Phase4SummaryValidationError(
            "aggregates must contain exactly one entry for every condition"
        )
    for index, aggregate in enumerate(aggregates):
        _validate_aggregate_consistency(aggregate, all_results, index)
    if set(aggregate_conditions) != set(run_counts):
        raise Phase4SummaryValidationError(
            "aggregates must contain exactly the conditions present in all_results"
        )
    for index, aggregate in enumerate(aggregates):
        condition = aggregate["condition"]
        if int(aggregate["n_seeds"]) != run_counts[condition]:
            raise Phase4SummaryValidationError(
                f"aggregates[{index}].n_seeds does not match all_results"
            )


def write_phase4_summary(summary: Dict[str, Any], output_path: Path) -> None:
    """Validate and serialize a Phase 4 summary without non-finite JSON values."""
    validate_phase4_summary(summary)
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
