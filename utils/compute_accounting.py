"""Shared compute accounting for confirmatory RL runs.

The model counters measure completed evaluator API calls and the tensor work
performed by those calls.  A "state evaluation" is one batch element passed
to a policy or value API.  A "latent state update" is one batch element passed
through one recurrent latent transition.  Logit and value counts are tensor
elements produced, so evaluating the same environment state with distinct
old, candidate, or target models counts each model's work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import resource
import sys
from typing import Any, Mapping, Sequence

import torch


COMPUTE_SNAPSHOT_SCHEMA_VERSION = 2

MODEL_COUNTER_FIELDS = (
    "latent_initialization_calls",
    "latent_states_initialized",
    "policy_api_calls",
    "policy_state_evaluations",
    "value_api_calls",
    "value_state_evaluations",
    "recurrent_latent_update_calls",
    "recurrent_latent_state_updates",
    "action_logits_evaluated",
    "state_values_evaluated",
    "action_values_evaluated",
)

MODEL_COUNTER_DEFINITIONS = {
    "latent_initialization_calls": "completed calls to model.init_latent",
    "latent_states_initialized": (
        "sum of leading batch sizes in completed model.init_latent calls"
    ),
    "policy_api_calls": "completed calls to model.policy_dist",
    "policy_state_evaluations": (
        "sum of leading batch sizes in completed model.policy_dist calls"
    ),
    "value_api_calls": "completed calls to model.used_value",
    "value_state_evaluations": (
        "sum of leading batch sizes in completed model.used_value calls"
    ),
    "recurrent_latent_update_calls": (
        "completed inner recurrent latent-transition invocations"
    ),
    "recurrent_latent_state_updates": (
        "sum of leading batch sizes in completed recurrent latent transitions"
    ),
    "action_logits_evaluated": (
        "number of categorical logit tensor elements returned by policy_dist"
    ),
    "state_values_evaluated": (
        "number of value tensor elements returned by used_value"
    ),
    "action_values_evaluated": (
        "number of Q-hat action values constructed by exact action enumeration; "
        "vectorized per-row values count even when later masked"
    ),
}

OPTIMIZER_STEP_FIELDS = (
    "combined",
    "value",
    "policy",
    "distillation",
    "puzzle_embedding",
)


def zero_model_counters() -> dict[str, int]:
    return {field: 0 for field in MODEL_COUNTER_FIELDS}


def validate_model_counters(value: object, *, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    if set(value) != set(MODEL_COUNTER_FIELDS):
        missing = sorted(set(MODEL_COUNTER_FIELDS) - set(value))
        extra = sorted(set(value) - set(MODEL_COUNTER_FIELDS))
        raise ValueError(f"{name} counter inventory mismatch: missing={missing}, extra={extra}")
    result: dict[str, int] = {}
    for field in MODEL_COUNTER_FIELDS:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name}.{field} must be a nonnegative integer")
        result[field] = item
    return result


def add_model_counters(*values: Mapping[str, int]) -> dict[str, int]:
    result = zero_model_counters()
    for index, value in enumerate(values):
        checked = validate_model_counters(dict(value), name=f"model_counters[{index}]")
        for field in MODEL_COUNTER_FIELDS:
            result[field] += checked[field]
    return result


def subtract_model_counters(
    minuend: Mapping[str, int], subtrahend: Mapping[str, int]
) -> dict[str, int]:
    left = validate_model_counters(dict(minuend), name="minuend")
    right = validate_model_counters(dict(subtrahend), name="subtrahend")
    result: dict[str, int] = {}
    for field in MODEL_COUNTER_FIELDS:
        difference = left[field] - right[field]
        if difference < 0:
            raise ValueError(f"model counter {field} decreased")
        result[field] = difference
    return result


@dataclass
class ModelComputeCounters:
    latent_initialization_calls: int = 0
    latent_states_initialized: int = 0
    policy_api_calls: int = 0
    policy_state_evaluations: int = 0
    value_api_calls: int = 0
    value_state_evaluations: int = 0
    recurrent_latent_update_calls: int = 0
    recurrent_latent_state_updates: int = 0
    action_logits_evaluated: int = 0
    state_values_evaluated: int = 0
    action_values_evaluated: int = 0

    def snapshot(self) -> dict[str, int]:
        return validate_model_counters(
            {field: int(getattr(self, field)) for field in MODEL_COUNTER_FIELDS},
            name="model_compute_counters",
        )

    def restore(self, value: object) -> None:
        checked = validate_model_counters(value, name="model_compute_counters")
        for field, item in checked.items():
            setattr(self, field, item)


def _group_unique_models(
    role_models: Mapping[str, object],
) -> list[tuple[list[str], object]]:
    groups: list[tuple[list[str], object]] = []
    by_identity: dict[int, int] = {}
    for role in sorted(role_models):
        model = role_models[role]
        identity = id(model)
        group_index = by_identity.get(identity)
        if group_index is None:
            by_identity[identity] = len(groups)
            groups.append(([role], model))
        else:
            groups[group_index][0].append(role)
    return groups


def capture_model_compute_state(
    role_models: Mapping[str, object],
) -> list[dict[str, object]]:
    """Capture each unique model once, retaining all aliases as role names."""
    state: list[dict[str, object]] = []
    for roles, model in _group_unique_models(role_models):
        snapshot_fn = getattr(model, "compute_counter_snapshot", None)
        if not callable(snapshot_fn):
            continue
        state.append(
            {
                "roles": sorted(roles),
                "counters": validate_model_counters(
                    snapshot_fn(), name=f"model[{','.join(roles)}]"
                ),
            }
        )
    return state


def validate_model_compute_state(
    role_models: Mapping[str, object], value: object
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("model_compute_state must be a list")
    expected = _group_unique_models(role_models)
    expected_by_roles = {tuple(sorted(roles)): model for roles, model in expected}
    supplied: dict[tuple[str, ...], dict[str, int]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"roles", "counters"}:
            raise ValueError(f"model_compute_state[{index}] has an invalid inventory")
        roles = item["roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role for role in roles)
            or roles != sorted(set(roles))
        ):
            raise ValueError(f"model_compute_state[{index}].roles is invalid")
        key = tuple(roles)
        if key in supplied:
            raise ValueError(f"duplicate model compute role group {key}")
        supplied[key] = validate_model_counters(
            item["counters"], name=f"model_compute_state[{','.join(roles)}]"
        )
    if set(supplied) != set(expected_by_roles):
        raise ValueError(
            "model compute role groups do not match checkpoint: "
            f"expected={sorted(expected_by_roles)}, supplied={sorted(supplied)}"
        )
    canonical: list[dict[str, object]] = []
    for roles in sorted(expected_by_roles):
        canonical.append(
            {"roles": list(roles), "counters": supplied[roles]}
        )
    return canonical


def restore_model_compute_state(
    role_models: Mapping[str, object], value: object
) -> None:
    canonical = validate_model_compute_state(role_models, value)
    expected_by_roles = {
        tuple(sorted(roles)): model
        for roles, model in _group_unique_models(role_models)
    }
    for item in canonical:
        roles_value = item["roles"]
        if not isinstance(roles_value, list):
            raise ValueError("model compute roles must be a list")
        roles = tuple(str(role) for role in roles_value)
        model = expected_by_roles[roles]
        restore_fn = getattr(model, "restore_compute_counters", None)
        if not callable(restore_fn):
            raise ValueError(f"model roles {roles} do not support compute counter restore")
        restore_fn(item["counters"])


def aggregate_model_compute(
    role_models: Mapping[str, object],
) -> tuple[dict[str, int], list[list[str]], list[str]]:
    aggregate = zero_model_counters()
    role_groups: list[list[str]] = []
    uninstrumented_roles: list[str] = []
    for roles, model in _group_unique_models(role_models):
        snapshot_fn = getattr(model, "compute_counter_snapshot", None)
        if not callable(snapshot_fn):
            uninstrumented_roles.extend(roles)
            continue
        checked = validate_model_counters(
            snapshot_fn(), name=f"model[{','.join(roles)}]"
        )
        aggregate = add_model_counters(aggregate, checked)
        role_groups.append(sorted(roles))
    return aggregate, role_groups, sorted(uninstrumented_roles)


def process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB. macOS and iOS report bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def current_cuda_memory_peaks(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nonnegative_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return result


def validate_compute_snapshot(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("compute_snapshot must be a dictionary")
    expected = {
        "compute_schema_version",
        "model_work",
        "progress",
        "wall_time_seconds",
        "peak_memory_bytes",
    }
    if set(value) != expected:
        raise ValueError("compute_snapshot has an invalid field inventory")
    schema_version = value["compute_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != COMPUTE_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported compute snapshot schema")

    model_work = value["model_work"]
    if not isinstance(model_work, dict) or set(model_work) != {
        "total",
        "training",
        "evaluation",
        "counter_definitions",
        "role_groups",
        "uninstrumented_roles",
    }:
        raise ValueError("compute_snapshot.model_work has an invalid inventory")
    if model_work["counter_definitions"] != MODEL_COUNTER_DEFINITIONS:
        raise ValueError("compute_snapshot.model_work counter definitions differ")
    total = validate_model_counters(model_work["total"], name="model_work.total")
    training = validate_model_counters(
        model_work["training"], name="model_work.training"
    )
    evaluation = validate_model_counters(
        model_work["evaluation"], name="model_work.evaluation"
    )
    if add_model_counters(training, evaluation) != total:
        raise ValueError("model_work.total must equal training plus evaluation")
    role_groups = model_work["role_groups"]
    if not isinstance(role_groups, list):
        raise ValueError("model_work.role_groups must be a list")
    grouped_roles: list[str] = []
    for index, roles in enumerate(role_groups):
        if (
            not isinstance(roles, list)
            or not roles
            or roles != sorted(set(roles))
            or any(not isinstance(role, str) or not role for role in roles)
        ):
            raise ValueError(f"model_work.role_groups[{index}] is invalid")
        grouped_roles.extend(roles)
    if len(grouped_roles) != len(set(grouped_roles)):
        raise ValueError("model_work.role_groups contains duplicate roles")
    uninstrumented_roles = model_work["uninstrumented_roles"]
    if (
        not isinstance(uninstrumented_roles, list)
        or uninstrumented_roles != sorted(set(uninstrumented_roles))
        or any(not isinstance(role, str) or not role for role in uninstrumented_roles)
    ):
        raise ValueError("model_work.uninstrumented_roles is invalid")
    if set(grouped_roles) & set(uninstrumented_roles):
        raise ValueError("instrumented and uninstrumented model roles overlap")

    progress = value["progress"]
    if not isinstance(progress, dict) or set(progress) != {
        "environment_interactions",
        "outer_updates",
        "optimizer_steps_total",
        "optimizer_steps_by_kind",
    }:
        raise ValueError("compute_snapshot.progress has an invalid inventory")
    _nonnegative_int(
        progress["environment_interactions"], name="progress.environment_interactions"
    )
    _nonnegative_int(progress["outer_updates"], name="progress.outer_updates")
    optimizer_total = _nonnegative_int(
        progress["optimizer_steps_total"], name="progress.optimizer_steps_total"
    )
    by_kind = progress["optimizer_steps_by_kind"]
    if not isinstance(by_kind, dict) or set(by_kind) != set(OPTIMIZER_STEP_FIELDS):
        raise ValueError("optimizer_steps_by_kind has an invalid inventory")
    checked_steps = {
        field: _nonnegative_int(by_kind[field], name=f"optimizer_steps_by_kind.{field}")
        for field in OPTIMIZER_STEP_FIELDS
    }
    if sum(checked_steps.values()) != optimizer_total:
        raise ValueError("optimizer_steps_total does not equal the by-kind sum")

    wall = value["wall_time_seconds"]
    if not isinstance(wall, dict) or set(wall) != {"training", "evaluation"}:
        raise ValueError("wall_time_seconds has an invalid inventory")
    _nonnegative_finite_float(wall["training"], name="wall_time_seconds.training")
    _nonnegative_finite_float(wall["evaluation"], name="wall_time_seconds.evaluation")

    memory = value["peak_memory_bytes"]
    if not isinstance(memory, dict) or set(memory) != {
        "cuda_allocated",
        "cuda_reserved",
        "process_rss",
    }:
        raise ValueError("peak_memory_bytes has an invalid inventory")
    for field in ("cuda_allocated", "cuda_reserved"):
        if memory[field] is not None:
            _nonnegative_int(memory[field], name=f"peak_memory_bytes.{field}")
    _nonnegative_int(memory["process_rss"], name="peak_memory_bytes.process_rss")

    # Return a shallowly normalized copy so callers cannot mistake validation
    # for preservation of custom mapping subclasses.
    return dict(value)
