#!/usr/bin/env python3
"""Publication schema for newly generated Phase 4 evaluation summaries."""

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, FrozenSet, List


PHASE4_SCHEMA_VERSION = 2

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
        "git_sha",
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
        "enable_contraction",
        "disable_value_head_norm",
        "latent_projection_mode",
        "latent_ball_radius",
        "L_preproj",
        "L_preproj_std",
        "var_V",
        "projection_active_rate",
        "argmax_agreement_4x",
        "argmax_agreement_8x",
        "delta_V_4x",
        "delta_V_8x",
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
        "argmax_4x_mean",
        "argmax_4x_std",
        "argmax_8x_mean",
        "argmax_8x_std",
        "delta_V_4x_mean",
        "delta_V_8x_mean",
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
    """Raised when a Phase 4 summary is not publishable schema version 2."""


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
    _require_string(run["checkpoint_path"], f"{path}.checkpoint_path")
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

    for field in ("L_preproj", "L_preproj_std", "var_V", "delta_V_4x", "delta_V_8x"):
        _require_number(run[field], f"{path}.{field}")
    for field in (
        "projection_active_rate",
        "argmax_agreement_4x",
        "argmax_agreement_8x",
    ):
        _require_number(run[field], f"{path}.{field}", maximum=1.0)
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
        "delta_V_4x_mean",
        "delta_V_8x_mean",
    ):
        _require_number(aggregate[field], f"{path}.{field}")
    for field in (
        "projection_active_rate_mean",
        "argmax_4x_mean",
        "argmax_4x_std",
        "argmax_8x_mean",
        "argmax_8x_std",
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
        "argmax_4x_mean": "argmax_agreement_4x",
        "argmax_8x_mean": "argmax_agreement_8x",
        "delta_V_4x_mean": "delta_V_4x",
        "delta_V_8x_mean": "delta_V_8x",
    }
    std_fields = {
        "L_preproj_std": "L_preproj",
        "var_V_std": "var_V",
        "argmax_4x_std": "argmax_agreement_4x",
        "argmax_8x_std": "argmax_agreement_8x",
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

    Schema-less and schema-v1 files remain historical artifacts. They are not
    migrated because a legacy zero cannot be distinguished from a measurement.
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

    for field in ("experiment", "description", "generated_at", "git_sha"):
        _require_string(summary[field], field)

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
