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
import hashlib
import json
import math
import platform
import resource
import sys
from typing import Any, Callable, Mapping, Sequence

import torch


COMPUTE_SNAPSHOT_SCHEMA_VERSION = 2
AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_NAME = (
    "policy_improvement_compute_accounting_v2"
)
AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_VERSION = 1

AUTHENTICATED_COMPUTE_FIELDS = (
    "environment_interactions",
    "recurrent_latent_state_updates",
    "action_logit_evaluations",
    "action_value_evaluations",
    "value_head_calls",
    "optimizer_steps_by_type",
    "wall_clock_training_seconds",
    "evaluation_seconds",
    "checkpoint_seconds",
    "audit_seconds",
    "cuda_utilization_samples",
    "peak_allocated_device_bytes",
    "peak_reserved_device_bytes",
    "process_peak_rss_bytes",
    "execution_device_identity",
)

_TRAINING_UNAVAILABLE_AUDIT_REASON = "owned_by_authenticated_audit_role"
_AUDIT_UNAVAILABLE_TRAINING_REASON = "not_owned_by_authenticated_audit_role"
_NON_CUDA_REASON = "not_applicable_for_non_cuda_execution"

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
        raise ValueError(
            f"{name} counter inventory mismatch: missing={missing}, extra={extra}"
        )
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
        canonical.append({"roles": list(roles), "counters": supplied[roles]})
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
            raise ValueError(
                f"model roles {roles} do not support compute counter restore"
            )
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


def execution_device_identity(device: torch.device | str) -> dict[str, object]:
    """Return the physical execution-device identity bound into v2 evidence."""

    resolved = torch.device(device)
    if resolved.type == "cpu":
        return {
            "canonical": "cpu",
            "device_type": "cpu",
            "device_index": None,
            "hardware_name": platform.processor() or platform.machine(),
            "cuda_capability": None,
        }
    if resolved.type != "cuda":
        raise ValueError(f"Unsupported execution device type {resolved.type!r}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA execution identity requires an available CUDA runtime")
    index = resolved.index
    if index is None:
        index = int(torch.cuda.current_device())
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError("CUDA execution device index is unavailable")
    capability = torch.cuda.get_device_capability(index)
    return {
        "canonical": f"cuda:{index}",
        "device_type": "cuda",
        "device_index": index,
        "hardware_name": str(torch.cuda.get_device_name(index)),
        "cuda_capability": [int(capability[0]), int(capability[1])],
    }


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _validate_device_identity(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "canonical",
        "device_type",
        "device_index",
        "hardware_name",
        "cuda_capability",
    }:
        raise ValueError(f"{name} has an invalid inventory")
    device_type = value["device_type"]
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(f"{name}.device_type is unsupported")
    canonical = value["canonical"]
    hardware_name = value["hardware_name"]
    if not isinstance(hardware_name, str) or not hardware_name:
        raise ValueError(f"{name}.hardware_name must be a nonempty string")
    if device_type == "cpu":
        if (
            canonical != "cpu"
            or value["device_index"] is not None
            or value["cuda_capability"] is not None
        ):
            raise ValueError(f"{name} is not a canonical CPU identity")
    else:
        index = value["device_index"]
        capability = value["cuda_capability"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or canonical != f"cuda:{index}"
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in capability
            )
        ):
            raise ValueError(f"{name} is not a canonical CUDA identity")
    return dict(value)


def _validate_availability(
    value: object,
    *,
    name: str,
    validator: Callable[..., object],
) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("status") not in {
        "available",
        "unavailable",
    }:
        raise ValueError(f"{name} must be an availability object")
    if value["status"] == "unavailable":
        if set(value) != {"status", "reason"}:
            raise ValueError(f"{name} unavailable inventory is invalid")
        reason = value["reason"]
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{name}.reason must be a nonempty string")
        return {"status": "unavailable", "reason": reason}
    if set(value) != {"status", "value"}:
        raise ValueError(f"{name} available inventory is invalid")
    return {"status": "available", "value": validator(value["value"], name=name)}


def _availability_value(value: Mapping[str, object], *, name: str) -> object:
    if value.get("status") != "available":
        raise ValueError(f"{name} must be available")
    return value["value"]


def _validate_optimizer_steps(value: object, *, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(OPTIMIZER_STEP_FIELDS):
        raise ValueError(f"{name} has an invalid optimizer inventory")
    return {
        field: _nonnegative_int(value[field], name=f"{name}.{field}")
        for field in OPTIMIZER_STEP_FIELDS
    }


def _validate_cuda_samples(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "sampling_interval_seconds",
        "samples",
    }:
        raise ValueError(f"{name} has an invalid inventory")
    interval = _nonnegative_finite_float(
        value["sampling_interval_seconds"],
        name=f"{name}.sampling_interval_seconds",
    )
    samples = value["samples"]
    if interval <= 0.0 or not isinstance(samples, list) or len(samples) < 2:
        raise ValueError(f"{name} requires an interval and at least two samples")
    checked_samples = [
        _nonnegative_finite_float(sample, name=f"{name}.samples[{index}]")
        for index, sample in enumerate(samples)
    ]
    if any(sample > 1.0 for sample in checked_samples):
        raise ValueError(f"{name} samples must lie in [0, 1]")
    return {
        "sampling_interval_seconds": interval,
        "samples": checked_samples,
    }


def _validate_nonnegative_int_value(value: object, *, name: str) -> int:
    return _nonnegative_int(value, name=name)


def _validate_nonnegative_float_value(value: object, *, name: str) -> float:
    return _nonnegative_finite_float(value, name=name)


def build_training_compute_accounting(
    compute_snapshot: object,
    *,
    device: torch.device | str,
    checkpoint_seconds: float,
    cuda_utilization: Mapping[str, object],
) -> dict[str, object]:
    """Bind a raw trainer snapshot to the protocol-v2 compute contract.

    Audit time is deliberately unavailable here.  The training runtime cannot
    measure work performed later by the independently authenticated audit PAR.
    """

    snapshot = validate_compute_snapshot(compute_snapshot)
    identity = execution_device_identity(device)
    training = snapshot["model_work"]["training"]
    progress = snapshot["progress"]
    memory = snapshot["peak_memory_bytes"]
    wall = snapshot["wall_time_seconds"]
    if not isinstance(cuda_utilization, Mapping) or set(cuda_utilization) != {
        "device_type",
        "sampling_interval_seconds",
        "samples",
    }:
        raise ValueError("cuda_utilization has an invalid inventory")
    if cuda_utilization["device_type"] != identity["device_type"]:
        raise ValueError("CUDA utilization names another execution device type")
    if identity["device_type"] == "cuda":
        utilization = _available(
            _validate_cuda_samples(
                {
                    "sampling_interval_seconds": cuda_utilization[
                        "sampling_interval_seconds"
                    ],
                    "samples": cuda_utilization["samples"],
                },
                name="cuda_utilization_samples",
            )
        )
        allocated = _available(memory["cuda_allocated"])
        reserved = _available(memory["cuda_reserved"])
    else:
        if (
            cuda_utilization["sampling_interval_seconds"] is not None
            or cuda_utilization["samples"] != []
            or memory["cuda_allocated"] is not None
            or memory["cuda_reserved"] is not None
        ):
            raise ValueError("CPU compute accounting cannot claim CUDA measurements")
        utilization = _unavailable(_NON_CUDA_REASON)
        allocated = _unavailable(_NON_CUDA_REASON)
        reserved = _unavailable(_NON_CUDA_REASON)
    result = {
        "schema_name": AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_NAME,
        "schema_version": AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_VERSION,
        "owner_role": "training",
        "source_compute_snapshot_sha256": _canonical_json_sha256(snapshot),
        "environment_interactions": _available(progress["environment_interactions"]),
        "recurrent_latent_state_updates": _available(
            training["recurrent_latent_state_updates"]
        ),
        "action_logit_evaluations": _available(training["action_logits_evaluated"]),
        "action_value_evaluations": _available(training["action_values_evaluated"]),
        "value_head_calls": _available(training["value_api_calls"]),
        "optimizer_steps_by_type": _available(progress["optimizer_steps_by_kind"]),
        "wall_clock_training_seconds": _available(wall["training"]),
        "evaluation_seconds": _available(wall["evaluation"]),
        "checkpoint_seconds": _available(float(checkpoint_seconds)),
        "audit_seconds": _unavailable(_TRAINING_UNAVAILABLE_AUDIT_REASON),
        "cuda_utilization_samples": utilization,
        "peak_allocated_device_bytes": allocated,
        "peak_reserved_device_bytes": reserved,
        "process_peak_rss_bytes": _available(memory["process_rss"]),
        "execution_device_identity": _available(identity),
    }
    return validate_authenticated_compute_accounting(
        result,
        source_compute_snapshot=snapshot,
    )


def build_audit_compute_accounting(
    *,
    audit_seconds: float,
    process_rss_bytes: int,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Record only the measurements owned by the authenticated audit role."""

    unavailable = _unavailable(_AUDIT_UNAVAILABLE_TRAINING_REASON)
    result = {
        "schema_name": AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_NAME,
        "schema_version": AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_VERSION,
        "owner_role": "audit",
        "source_compute_snapshot_sha256": None,
        **{field: dict(unavailable) for field in AUTHENTICATED_COMPUTE_FIELDS},
    }
    result["audit_seconds"] = _available(float(audit_seconds))
    result["process_peak_rss_bytes"] = _available(process_rss_bytes)
    result["execution_device_identity"] = _available(execution_device_identity(device))
    return validate_authenticated_compute_accounting(result)


def validate_authenticated_compute_accounting(
    value: object,
    *,
    source_compute_snapshot: object | None = None,
) -> dict[str, object]:
    """Validate the exact protocol-v2 compute record and optional source binding."""

    expected = {
        "schema_name",
        "schema_version",
        "owner_role",
        "source_compute_snapshot_sha256",
        *AUTHENTICATED_COMPUTE_FIELDS,
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("authenticated compute accounting has an invalid inventory")
    if (
        value["schema_name"] != AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_NAME
        or value["schema_version"] != AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_VERSION
    ):
        raise ValueError("unsupported authenticated compute accounting schema")
    role = value["owner_role"]
    if role not in {"training", "audit"}:
        raise ValueError("authenticated compute owner role is unsupported")
    validators = {
        "environment_interactions": _validate_nonnegative_int_value,
        "recurrent_latent_state_updates": _validate_nonnegative_int_value,
        "action_logit_evaluations": _validate_nonnegative_int_value,
        "action_value_evaluations": _validate_nonnegative_int_value,
        "value_head_calls": _validate_nonnegative_int_value,
        "optimizer_steps_by_type": _validate_optimizer_steps,
        "wall_clock_training_seconds": _validate_nonnegative_float_value,
        "evaluation_seconds": _validate_nonnegative_float_value,
        "checkpoint_seconds": _validate_nonnegative_float_value,
        "audit_seconds": _validate_nonnegative_float_value,
        "cuda_utilization_samples": _validate_cuda_samples,
        "peak_allocated_device_bytes": _validate_nonnegative_int_value,
        "peak_reserved_device_bytes": _validate_nonnegative_int_value,
        "process_peak_rss_bytes": _validate_nonnegative_int_value,
        "execution_device_identity": _validate_device_identity,
    }
    checked = {
        field: _validate_availability(
            value[field],
            name=field,
            validator=validators[field],
        )
        for field in AUTHENTICATED_COMPUTE_FIELDS
    }
    if role == "audit":
        if value["source_compute_snapshot_sha256"] is not None:
            raise ValueError("audit compute accounting cannot bind a trainer snapshot")
        for field in AUTHENTICATED_COMPUTE_FIELDS:
            item = checked[field]
            if field in {
                "audit_seconds",
                "process_peak_rss_bytes",
                "execution_device_identity",
            }:
                if item["status"] != "available":
                    raise ValueError(f"audit compute accounting requires {field}")
            elif item != _unavailable(_AUDIT_UNAVAILABLE_TRAINING_REASON):
                raise ValueError(f"audit role cannot claim {field}")
        identity = _availability_value(
            checked["execution_device_identity"],
            name="execution_device_identity",
        )
        if not isinstance(identity, dict) or identity["device_type"] != "cpu":
            raise ValueError("audit compute accounting must name its CPU role device")
    else:
        digest = value["source_compute_snapshot_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("training compute accounting source digest is invalid")
        required = set(AUTHENTICATED_COMPUTE_FIELDS) - {"audit_seconds"}
        identity_value = _availability_value(
            checked["execution_device_identity"],
            name="execution_device_identity",
        )
        if not isinstance(identity_value, dict):
            raise ValueError("execution device identity is invalid")
        device_type = identity_value["device_type"]
        if device_type == "cpu":
            required -= {
                "cuda_utilization_samples",
                "peak_allocated_device_bytes",
                "peak_reserved_device_bytes",
            }
            for field in (
                "cuda_utilization_samples",
                "peak_allocated_device_bytes",
                "peak_reserved_device_bytes",
            ):
                if checked[field] != _unavailable(_NON_CUDA_REASON):
                    raise ValueError(f"CPU training cannot claim {field}")
        for field in required:
            if checked[field]["status"] != "available":
                raise ValueError(f"training compute accounting requires {field}")
        if checked["audit_seconds"] != _unavailable(_TRAINING_UNAVAILABLE_AUDIT_REASON):
            raise ValueError("training runtime cannot claim audit_seconds")
        if source_compute_snapshot is not None:
            source = validate_compute_snapshot(source_compute_snapshot)
            if _canonical_json_sha256(source) != digest:
                raise ValueError("training compute source snapshot digest differs")
            training = source["model_work"]["training"]
            progress = source["progress"]
            wall = source["wall_time_seconds"]
            memory = source["peak_memory_bytes"]
            expected_values = {
                "environment_interactions": progress["environment_interactions"],
                "recurrent_latent_state_updates": training[
                    "recurrent_latent_state_updates"
                ],
                "action_logit_evaluations": training["action_logits_evaluated"],
                "action_value_evaluations": training["action_values_evaluated"],
                "value_head_calls": training["value_api_calls"],
                "optimizer_steps_by_type": progress["optimizer_steps_by_kind"],
                "wall_clock_training_seconds": wall["training"],
                "evaluation_seconds": wall["evaluation"],
                "process_peak_rss_bytes": memory["process_rss"],
            }
            if device_type == "cuda":
                expected_values.update(
                    {
                        "peak_allocated_device_bytes": memory["cuda_allocated"],
                        "peak_reserved_device_bytes": memory["cuda_reserved"],
                    }
                )
            for field, expected_value in expected_values.items():
                if _availability_value(checked[field], name=field) != expected_value:
                    raise ValueError(f"{field} differs from source compute snapshot")
    return {
        "schema_name": value["schema_name"],
        "schema_version": value["schema_version"],
        "owner_role": role,
        "source_compute_snapshot_sha256": value["source_compute_snapshot_sha256"],
        **checked,
    }


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
