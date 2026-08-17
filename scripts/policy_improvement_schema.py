#!/usr/bin/env fbpython
"""Strict schemas for the registered learned policy-improvement study.

This module is deliberately independent from the Phase 4 reporting schema.
Phase 4 remains a finite diagnostic study.  Values accepted here describe a
different experiment family whose primary endpoint is held-out solve rate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_NAME = "policy_improvement_v1"
PROTOCOL_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 3
REGISTRY_ROW_SCHEMA_VERSION = 3
COMPUTE_FREEZE_SCHEMA_VERSION = 1
SCREEN_SELECTION_SCHEMA_VERSION = 1
FINAL_SELECTION_SCHEMA_VERSION = 1
RUNTIME_AUTHORIZATION_SCHEMA_VERSION = 1
RUNTIME_AUTHORIZATION_SCHEMA_VERSION_V2 = 2
METHOD_IDS = (
    "fixed_base_exact_persistent",
    "fixed_base_exact_episodic",
    "legacy_parameter_interpolation",
    "matched_ppo",
)
STAGE3_ABLATION_METHOD_IDS = (
    "fixed_base_batch_only_centering",
    "fixed_base_distilled_realization",
)
RESULT_METHOD_IDS = METHOD_IDS + STAGE3_ABLATION_METHOD_IDS
SCREEN_SELECTION_RULE_ID = "best_validation_exact_configuration_v1"
ALPHA_SELECTION_RULE_ID = "best_validation_alpha_v1"
SELECTION_ENDPOINT = "mean_seed_level_primary_interaction_matched_solve_rate"
SCREEN_SELECTION_METHOD_ORDER = (
    "fixed_base_exact_persistent",
    "fixed_base_exact_episodic",
)
METHOD_CONTRACTS: dict[str, dict[str, str]] = {
    "fixed_base_exact_persistent": {
        "config_path": "configs/policy_improvement_v1/fixed_base_exact_persistent.yaml",
        "config_sha256": "2c101307f77ef788d588928b694b0ef01de186c3f9f1ae6f561e52d5e8f433a9",
        "canonical_config_sha256": "a7321e16d00e4dbe6e7694a8e0e1cee3f114e35ad960ee7c65f8f8526e7b8e4d",
        "training_protocol": "fixed_base_exact",
        "latent_mode": "persistent",
        "centering": "exact_statewise",
        "deployment": "exact_probability_mixture",
        "primary_policy": "exact_mixture",
    },
    "fixed_base_exact_episodic": {
        "config_path": "configs/policy_improvement_v1/fixed_base_exact_episodic.yaml",
        "config_sha256": "2f16dbbba406ae67f2b62342483800f0ae4c3a0d61019e3616697e1256dac36b",
        "canonical_config_sha256": "6f78dfd43e5ec579a943f688a55223e105ff6c92401f7c402426f43c021aa5e1",
        "training_protocol": "fixed_base_exact",
        "latent_mode": "episodic",
        "centering": "exact_statewise",
        "deployment": "exact_probability_mixture",
        "primary_policy": "exact_mixture",
    },
    "legacy_parameter_interpolation": {
        "config_path": "configs/policy_improvement_v1/legacy_parameter_interpolation.yaml",
        "config_sha256": "2489a5d053379f140b54c1d23cffe68554afc0d9f80045527717cdb6ebd07e7c",
        "canonical_config_sha256": "3963cbd17496da6d3a52a45b05c94de63745cc769cd98fe2a57b9f14ddc68e3d",
        "training_protocol": "legacy",
        "latent_mode": "persistent",
        "centering": "batch_only",
        "deployment": "parameter_interpolation",
        "primary_policy": "realized_policy",
    },
    "matched_ppo": {
        "config_path": "configs/policy_improvement_v1/matched_ppo.yaml",
        "config_sha256": "1aadcd2340d5de3bb6a344c101ba7a89b33f53b5681354ee97dfb2d37a4cd2ba",
        "canonical_config_sha256": "396399c83f02693f51a8850549f17468a94df7a73df186a8a2ea5780818da72d",
        "training_protocol": "ppo",
        "latent_mode": "episodic",
        "centering": "ppo_gae",
        "deployment": "greedy_policy",
        "primary_policy": "realized_policy",
    },
}
PHASE_CONTRACTS: dict[str, tuple[str, str, bool]] = {
    "stage0_smoke": ("smoke", "validation", False),
    "stage1_screen": ("pilot", "validation", False),
    "stage1_alpha": ("pilot", "validation", False),
    "stage2_confirmatory": ("confirmatory", "test", True),
    "stage3_ablation": ("ablation", "test", True),
}
TIERS = ("smoke", "pilot", "confirmatory", "ablation")
EVALUATION_SPLITS = ("validation", "test")
RUN_STATUSES = ("complete", "failed")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_UNAVAILABLE_REASONS = {
    "compute_budget_pending_smoke_measurement",
    "manifest_not_materialized",
    "not_applicable",
    "run_failed_before_measurement",
    "run_failed_before_checkpoint",
    "run_failed_before_evaluation",
    "not_collected_by_registered_protocol",
    "compute_snapshot_not_registered",
    "test_data_not_opened",
    "initialization_not_materialized",
    "execution_device_not_materialized",
}
_PLACEHOLDER_STRINGS = {
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
    "unavailable",
}

AMENDMENT_SCHEMAS = (
    "policy_improvement_theory_bridge_amendment_v1",
    "policy_improvement_compute_freeze_v1",
    "policy_improvement_screen_selection_v1",
    "policy_improvement_final_selection_v1",
)
RUNTIME_ROLES = (
    "policy-improvement-training",
    "policy-improvement-evaluation",
    "policy-improvement-audit",
    "policy-improvement-analysis",
)
RUNTIME_ROLES_V2 = (
    *RUNTIME_ROLES,
    "policy-improvement-full",
    "policy-improvement-theory-bridge",
)
PHASE_AMENDMENT_PREFIX_LENGTH = {
    "stage0_smoke": 0,
    "stage1_screen": 2,
    "stage1_alpha": 3,
    "stage2_confirmatory": 4,
    "stage3_ablation": 4,
}


class PolicyImprovementSchemaError(ValueError):
    """Raised when a protocol, registry row, or result is not canonical."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyImprovementSchemaError(
                f"Duplicate JSON key {key!r} is not allowed."
            )
        result[key] = value
    return result


def load_strict_json(path: str | Path) -> object:
    """Load strict UTF-8 JSON while rejecting duplicate object keys."""

    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise PolicyImprovementSchemaError("Strict JSON could not be read.") from exc
    return load_strict_json_bytes(payload)


def load_strict_json_bytes(payload: bytes) -> object:
    """Parse strict JSON from bytes already authenticated by the caller."""

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PolicyImprovementSchemaError(
                    f"Non-finite JSON constant {value!r} is not allowed."
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementSchemaError("Strict JSON could not be read.") from exc


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic ASCII JSON after recursively checking finiteness."""

    def normalize(item: object, path: str) -> object:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PolicyImprovementSchemaError(
                    f"{path} contains a non-finite number."
                )
            return item
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise PolicyImprovementSchemaError(f"{path} contains a non-string key.")
            return {key: normalize(item[key], f"{path}.{key}") for key in sorted(item)}
        if isinstance(item, list):
            return [
                normalize(child, f"{path}[{index}]") for index, child in enumerate(item)
            ]
        raise PolicyImprovementSchemaError(
            f"{path} contains unsupported type {type(item).__name__}."
        )

    return json.dumps(
        normalize(value, "value"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementSchemaError(f"{path} must be an object.")
    return value


def _exact_fields(
    value: object,
    expected: set[str],
    *,
    path: str,
) -> Mapping[str, object]:
    result = _mapping(value, path=path)
    actual = set(result)
    if actual != expected:
        raise PolicyImprovementSchemaError(
            f"{path} field inventory differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )
    return result


def _string(
    value: object,
    *,
    path: str,
    choices: set[str] | None = None,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise PolicyImprovementSchemaError(f"{path} must be nonempty ASCII text.")
    if value.strip().lower() in _PLACEHOLDER_STRINGS and (
        choices is None or value not in choices
    ):
        raise PolicyImprovementSchemaError(f"{path} cannot be a placeholder.")
    if choices is not None and value not in choices:
        raise PolicyImprovementSchemaError(f"{path} must be one of {sorted(choices)}.")
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise PolicyImprovementSchemaError(f"{path} is not a canonical identifier.")
    return value


def _sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PolicyImprovementSchemaError(
            f"{path} must be 64 lowercase hexadecimal characters."
        )
    return value


def _git_commit(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise PolicyImprovementSchemaError(
            f"{path} must be 40 lowercase hexadecimal characters."
        )
    return value


def _integer(
    value: object,
    *,
    path: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyImprovementSchemaError(
            f"{path} must be an integer greater than or equal to {minimum}."
        )
    return value


def _number(
    value: object,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyImprovementSchemaError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyImprovementSchemaError(f"{path} must be finite.")
    if minimum is not None and result < minimum:
        raise PolicyImprovementSchemaError(f"{path} is below {minimum}.")
    if maximum is not None and result > maximum:
        raise PolicyImprovementSchemaError(f"{path} is above {maximum}.")
    return result


def _string_list(
    value: object,
    *,
    path: str,
    exact_length: int | None = None,
    unique: bool = True,
) -> list[str]:
    if not isinstance(value, list) or (
        exact_length is not None and len(value) != exact_length
    ):
        raise PolicyImprovementSchemaError(f"{path} has an invalid list length.")
    result = [
        _string(item, path=f"{path}[{index}]") for index, item in enumerate(value)
    ]
    if unique and len(set(result)) != len(result):
        raise PolicyImprovementSchemaError(f"{path} must not contain duplicates.")
    return result


def _integer_list(
    value: object,
    *,
    path: str,
    exact_length: int,
) -> list[int]:
    if not isinstance(value, list) or len(value) != exact_length:
        raise PolicyImprovementSchemaError(
            f"{path} must contain exactly {exact_length} entries."
        )
    result = [
        _integer(item, path=f"{path}[{index}]") for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise PolicyImprovementSchemaError(f"{path} contains duplicate seeds.")
    return result


def _validate_dataset(value: object) -> dict[str, Any]:
    dataset = _exact_fields(
        value,
        {
            "name",
            "root",
            "builder_schema_version",
            "manifest_schema_version",
            "manifest_sha256",
            "producer_source",
            "domain",
            "symmetry_canonicalization",
            "splits",
        },
        path="protocol.dataset",
    )
    _string(dataset["name"], path="protocol.dataset.name", identifier=True)
    root = _string(dataset["root"], path="protocol.dataset.root")
    registered_root = Path(
        "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1"
    )
    if Path(root) != registered_root:
        raise PolicyImprovementSchemaError(
            "protocol.dataset.root must name the registered private-owner child."
        )
    builder_version = _integer(
        dataset["builder_schema_version"],
        path="protocol.dataset.builder_schema_version",
        minimum=1,
    )
    manifest_version = _integer(
        dataset["manifest_schema_version"],
        path="protocol.dataset.manifest_schema_version",
        minimum=1,
    )
    if builder_version != 2 or manifest_version != 2:
        raise PolicyImprovementSchemaError(
            "The registered dataset requires builder and manifest schema 2."
        )
    manifest_identity = _validate_available_sha256(
        dataset["manifest_sha256"], path="protocol.dataset.manifest_sha256"
    )
    producer = _exact_fields(
        dataset["producer_source"],
        {
            "git_commit",
            "source_manifest_sha256",
            "runtime_sha256",
            "launcher_sha256",
        },
        path="protocol.dataset.producer_source",
    )
    producer_identities = {
        "git_commit": _validate_availability(
            producer["git_commit"],
            path="protocol.dataset.producer_source.git_commit",
            kind="git_commit",
        )
    }
    for field in (
        "source_manifest_sha256",
        "runtime_sha256",
        "launcher_sha256",
    ):
        producer_identities[field] = _validate_available_sha256(
            producer[field], path=f"protocol.dataset.producer_source.{field}"
        )
    if dataset["domain"] != "sudoku_4x4":
        raise PolicyImprovementSchemaError("Only the registered 4x4 domain is valid.")
    symmetry = _exact_fields(
        dataset["symmetry_canonicalization"],
        {"scheme", "group_order", "reject_cross_split_overlap"},
        path="protocol.dataset.symmetry_canonicalization",
    )
    if symmetry["scheme"] != "sudoku4x4_spatial_digit_lexicographic_v1":
        raise PolicyImprovementSchemaError("Unexpected symmetry scheme.")
    if symmetry["group_order"] != 3072:
        raise PolicyImprovementSchemaError("Sudoku symmetry group order must be 3072.")
    if symmetry["reject_cross_split_overlap"] is not True:
        raise PolicyImprovementSchemaError("Cross-split symmetry overlap must fail.")
    splits = _exact_fields(
        dataset["splits"],
        {"train", "validation", "test"},
        path="protocol.dataset.splits",
    )
    for name in ("train", "validation", "test"):
        split = _exact_fields(
            splits[name],
            {
                "count",
                "generation_seed",
                "manifest_sha256",
                "ordered_record_sha256",
            },
            path=f"protocol.dataset.splits.{name}",
        )
        _integer(
            split["count"], path=f"protocol.dataset.splits.{name}.count", minimum=1
        )
        _integer(
            split["generation_seed"],
            path=f"protocol.dataset.splits.{name}.generation_seed",
        )
        availability = _validate_available_sha256(
            split["manifest_sha256"],
            path=f"protocol.dataset.splits.{name}.manifest_sha256",
        )
        if (
            availability["status"] == "unavailable"
            and availability["reason"] != "manifest_not_materialized"
        ):
            raise PolicyImprovementSchemaError(
                "Pending dataset manifests must use the registered unavailable reason."
            )
        ordered_availability = _validate_available_sha256(
            split["ordered_record_sha256"],
            path=f"protocol.dataset.splits.{name}.ordered_record_sha256",
        )
        if (
            ordered_availability["status"] == "unavailable"
            and ordered_availability["reason"] != "manifest_not_materialized"
        ):
            raise PolicyImprovementSchemaError(
                "Pending ordered-record identities use the wrong reason."
            )
    all_identities = [manifest_identity, *producer_identities.values()]
    all_identities.extend(
        _validate_available_sha256(
            splits[name]["manifest_sha256"],
            path=f"protocol.dataset.splits.{name}.manifest_sha256",
        )
        for name in ("train", "validation", "test")
    )
    all_identities.extend(
        _validate_available_sha256(
            splits[name]["ordered_record_sha256"],
            path=f"protocol.dataset.splits.{name}.ordered_record_sha256",
        )
        for name in ("train", "validation", "test")
    )
    statuses = {str(identity["status"]) for identity in all_identities}
    if len(statuses) != 1:
        raise PolicyImprovementSchemaError(
            "Dataset manifest, producer, and split identities must freeze together."
        )
    if statuses == {"unavailable"} and any(
        identity.get("reason") != "manifest_not_materialized"
        for identity in all_identities
    ):
        raise PolicyImprovementSchemaError(
            "Pending dataset identities must use manifest_not_materialized."
        )
    return dict(dataset)


def _validate_methods(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(METHOD_IDS):
        raise PolicyImprovementSchemaError(
            "protocol.methods must contain four methods."
        )
    methods: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        method = _exact_fields(
            item,
            {
                "id",
                "config_path",
                "config_sha256",
                "canonical_config_sha256",
                "training_protocol",
                "latent_mode",
                "centering",
                "deployment",
                "primary_policy",
            },
            path=f"protocol.methods[{index}]",
        )
        method_id = _string(
            method["id"],
            path=f"protocol.methods[{index}].id",
            choices=set(METHOD_IDS),
        )
        config_path = _string(
            method["config_path"],
            path=f"protocol.methods[{index}].config_path",
        )
        if Path(config_path).is_absolute() or ".." in Path(config_path).parts:
            raise PolicyImprovementSchemaError("Method config paths must be relative.")
        _sha256(
            method["config_sha256"],
            path=f"protocol.methods[{index}].config_sha256",
        )
        _sha256(
            method["canonical_config_sha256"],
            path=f"protocol.methods[{index}].canonical_config_sha256",
        )
        expected = {"id": method_id, **METHOD_CONTRACTS[method_id]}
        if dict(method) != expected:
            raise PolicyImprovementSchemaError(
                f"protocol.methods[{index}] differs from the exact {method_id!r} "
                "method tuple."
            )
        methods.append(dict(method))
    if [method["id"] for method in methods] != list(METHOD_IDS):
        raise PolicyImprovementSchemaError(
            "Methods must appear in the canonical registered order."
        )
    return methods


def _validate_evaluation_populations(value: object) -> dict[str, Any]:
    populations = _exact_fields(
        value,
        {"smoke", "pilot", "confirmatory"},
        path="protocol.evaluation_populations",
    )
    contracts = {
        "smoke": ("validation", 8),
        "pilot": ("validation", 256),
        "confirmatory": ("test", 512),
    }
    statuses: set[str] = set()
    checked: dict[str, Any] = {}
    for tier, (expected_split, expected_count) in contracts.items():
        path = f"protocol.evaluation_populations.{tier}"
        population = _exact_fields(
            populations[tier],
            {"split", "count", "indices", "ordered_record_sha256"},
            path=path,
        )
        if population["split"] != expected_split:
            raise PolicyImprovementSchemaError(
                f"{path}.split differs from the registered held-out population."
            )
        if population["count"] != expected_count:
            raise PolicyImprovementSchemaError(
                f"{path}.count must equal {expected_count}."
            )
        indices = _exact_fields(
            population["indices"],
            {"scheme", "first", "last_inclusive"},
            path=f"{path}.indices",
        )
        if indices != {
            "scheme": "contiguous_registered_indices_v1",
            "first": 0,
            "last_inclusive": expected_count - 1,
        }:
            raise PolicyImprovementSchemaError(
                f"{path}.indices must register the exact ordered prefix."
            )
        identity = _validate_available_sha256(
            population["ordered_record_sha256"],
            path=f"{path}.ordered_record_sha256",
        )
        statuses.add(str(identity["status"]))
        if identity["status"] == "unavailable" and identity.get("reason") != (
            "manifest_not_materialized"
        ):
            raise PolicyImprovementSchemaError(
                f"{path} pending identity uses the wrong reason."
            )
        checked[tier] = dict(population)
    if len(statuses) != 1:
        raise PolicyImprovementSchemaError(
            "Evaluation-population identities must freeze together."
        )
    return checked


def _validate_available_sha256(value: object, *, path: str) -> dict[str, Any]:
    return _validate_availability(value, path=path, kind="sha256")


def _validate_availability(
    value: object,
    *,
    path: str,
    kind: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    availability = _mapping(value, path=path)
    status = availability.get("status")
    if status == "available":
        available = _exact_fields(
            availability,
            {"status", "value"},
            path=path,
        )
        raw = available["value"]
        if kind == "sha256":
            normalized: object = _sha256(raw, path=f"{path}.value")
        elif kind == "git_commit":
            normalized = _git_commit(raw, path=f"{path}.value")
        elif kind == "number":
            normalized = _number(
                raw,
                path=f"{path}.value",
                minimum=minimum,
                maximum=maximum,
            )
        elif kind == "integer":
            normalized = _integer(
                raw,
                path=f"{path}.value",
                minimum=int(minimum or 0),
            )
        elif kind == "mapping":
            normalized = dict(_mapping(raw, path=f"{path}.value"))
            canonical_json_bytes(normalized)
        elif kind == "string":
            normalized = _string(raw, path=f"{path}.value")
        elif kind == "terminal_reason_counts":
            counts = _exact_fields(
                raw,
                {"budget", "solved", "stop"},
                path=f"{path}.value",
            )
            normalized = {
                reason: _integer(counts[reason], path=f"{path}.value.{reason}")
                for reason in ("budget", "solved", "stop")
            }
        elif kind == "model_forward_counts":
            counts = _exact_fields(
                raw,
                {
                    "actions_processed",
                    "policy_head_calls",
                    "recurrent_map_applications",
                    "value_head_calls",
                },
                path=f"{path}.value",
            )
            normalized = {
                field: _integer(counts[field], path=f"{path}.value.{field}")
                for field in sorted(counts)
            }
        else:
            raise AssertionError(f"Unknown availability kind {kind!r}.")
        return {"status": "available", "value": normalized}
    if status == "unavailable":
        unavailable = _exact_fields(
            availability,
            {"status", "reason"},
            path=path,
        )
        reason = _string(
            unavailable["reason"],
            path=f"{path}.reason",
            choices=_UNAVAILABLE_REASONS,
        )
        return {"status": "unavailable", "reason": reason}
    raise PolicyImprovementSchemaError(
        f"{path}.status must be 'available' or 'unavailable'."
    )


def _timestamp(value: object, *, path: str) -> str:
    text = _string(value, path=path)
    if (
        re.fullmatch(
            r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            text,
        )
        is None
    ):
        raise PolicyImprovementSchemaError(
            f"{path} must be a second-resolution UTC timestamp."
        )
    return text


def _validate_audit_evidence(
    value: object, *, path: str, expected_phase: str, expected_rows: int
) -> dict[str, Any]:
    evidence = _exact_fields(
        value,
        {
            "phase",
            "audit_report_sha256",
            "result_set_sha256",
            "per_instance_set_sha256",
            "expected_rows",
            "complete_rows",
            "failed_rows",
        },
        path=path,
    )
    if evidence["phase"] != expected_phase:
        raise PolicyImprovementSchemaError(f"{path}.phase is not the prior stage.")
    for field in (
        "audit_report_sha256",
        "result_set_sha256",
        "per_instance_set_sha256",
    ):
        _sha256(evidence[field], path=f"{path}.{field}")
    if evidence["expected_rows"] != expected_rows:
        raise PolicyImprovementSchemaError(
            f"{path}.expected_rows must equal {expected_rows}."
        )
    complete = _integer(evidence["complete_rows"], path=f"{path}.complete_rows")
    failed = _integer(evidence["failed_rows"], path=f"{path}.failed_rows")
    if complete + failed != expected_rows:
        raise PolicyImprovementSchemaError(
            f"{path} must account for every registered prior-stage row."
        )
    return dict(evidence)


def amendment_history_sha256(history: Sequence[object]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(history))).hexdigest()


def runtime_authorization_sha256(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(validate_runtime_authorization(value))
    ).hexdigest()


def validate_runtime_authorization(value: object) -> dict[str, Any]:
    """Validate the external, non-self-referential runtime/source freeze."""

    authorization = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "authorization_id",
            "created_at_utc",
            "protocol_sha256",
            "producer_git_commit",
            "producer_source_manifest_sha256",
            "launcher_sha256",
            "roles",
        },
        path="runtime_authorization",
    )
    schema_name = authorization["schema_name"]
    schema_version = authorization["schema_version"]
    if (
        not isinstance(schema_name, str)
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
    ):
        raise PolicyImprovementSchemaError(
            "Unsupported runtime authorization version."
        )
    schema = (schema_name, schema_version)
    role_schemas = {
        (
            "policy_improvement_runtime_authorization_v1",
            RUNTIME_AUTHORIZATION_SCHEMA_VERSION,
        ): RUNTIME_ROLES,
        (
            "policy_improvement_runtime_authorization_v2",
            RUNTIME_AUTHORIZATION_SCHEMA_VERSION_V2,
        ): RUNTIME_ROLES_V2,
    }
    if schema not in role_schemas:
        raise PolicyImprovementSchemaError("Unsupported runtime authorization version.")
    expected_roles = role_schemas[schema]
    _string(
        authorization["authorization_id"],
        path="runtime_authorization.authorization_id",
        identifier=True,
    )
    _timestamp(
        authorization["created_at_utc"],
        path="runtime_authorization.created_at_utc",
    )
    _sha256(
        authorization["protocol_sha256"],
        path="runtime_authorization.protocol_sha256",
    )
    _git_commit(
        authorization["producer_git_commit"],
        path="runtime_authorization.producer_git_commit",
    )
    _sha256(
        authorization["producer_source_manifest_sha256"],
        path="runtime_authorization.producer_source_manifest_sha256",
    )
    _sha256(
        authorization["launcher_sha256"],
        path="runtime_authorization.launcher_sha256",
    )
    roles = authorization["roles"]
    if not isinstance(roles, list) or len(roles) != len(expected_roles):
        raise PolicyImprovementSchemaError(
            "Runtime authorization must contain every exact consumer role."
        )
    checked_roles: list[dict[str, Any]] = []
    for index, expected_role in enumerate(expected_roles):
        path = f"runtime_authorization.roles[{index}]"
        role = _exact_fields(
            roles[index],
            {
                "role",
                "source_git_commit",
                "runtime_sha256",
                "runtime_profile_sha256",
                "selected_source_manifest_sha256",
            },
            path=path,
        )
        if role["role"] != expected_role:
            raise PolicyImprovementSchemaError(
                "Runtime authorization roles must use canonical order."
            )
        _git_commit(role["source_git_commit"], path=f"{path}.source_git_commit")
        for field in (
            "runtime_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
        ):
            _sha256(role[field], path=f"{path}.{field}")
        if role["runtime_profile_sha256"] != role["selected_source_manifest_sha256"]:
            raise PolicyImprovementSchemaError(
                f"{path} runtime-profile digest must equal the authenticated "
                "canonical selected-source profile digest."
            )
        checked_roles.append(dict(role))
    if checked_roles[0]["source_git_commit"] != authorization["producer_git_commit"]:
        raise PolicyImprovementSchemaError(
            "The training role source commit must equal the authorized producer commit."
        )
    if len(checked_roles) == len(RUNTIME_ROLES_V2):
        for semantic_index, launcher_index, label in (
            (0, 4, "full-runtime and training"),
            (1, 5, "theory-bridge and evaluation"),
        ):
            if any(
                checked_roles[semantic_index][field]
                != checked_roles[launcher_index][field]
                for field in (
                    "source_git_commit",
                    "runtime_sha256",
                    "runtime_profile_sha256",
                    "selected_source_manifest_sha256",
                )
            ):
                raise PolicyImprovementSchemaError(
                    f"The {label} roles must bind the same sealed artifact."
                )
        if checked_roles[4]["source_git_commit"] != authorization[
            "producer_git_commit"
        ]:
            raise PolicyImprovementSchemaError(
                "The full-runtime role source commit must equal the authorized "
                "producer."
            )
    canonical_json_bytes(authorization)
    return {**dict(authorization), "roles": checked_roles}


def _validate_common_compute_targets(value: object, *, path: str) -> dict[str, Any]:
    targets = _exact_fields(
        value,
        {
            "unit",
            "pilot",
            "confirmatory",
            "ablation",
            "maximum_relative_mismatch",
        },
        path=path,
    )
    if targets["unit"] != "recurrent_map_applications":
        raise PolicyImprovementSchemaError(
            "The common compute target must use recurrent-map applications."
        )
    for tier in ("pilot", "confirmatory", "ablation"):
        _integer(targets[tier], path=f"{path}.{tier}", minimum=1)
    if (
        _number(
            targets["maximum_relative_mismatch"],
            path=f"{path}.maximum_relative_mismatch",
            minimum=0.0,
            maximum=1.0,
        )
        != 0.05
    ):
        raise PolicyImprovementSchemaError(
            "The registered compute mismatch tolerance must remain five percent."
        )
    return dict(targets)


def validate_compute_freeze(value: object) -> dict[str, Any]:
    freeze = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "amendment_id",
            "created_at_utc",
            "protocol_id",
            "protocol_sha256",
            "prior_amendment_history_sha256",
            "source_registry_sha256",
            "runtime_authorization_sha256",
            "test_data_opened",
            "evidence",
            "common_compute_targets",
        },
        path="compute_freeze",
    )
    if (
        freeze["schema_name"] != AMENDMENT_SCHEMAS[1]
        or freeze["schema_version"] != COMPUTE_FREEZE_SCHEMA_VERSION
    ):
        raise PolicyImprovementSchemaError("Unexpected compute-freeze schema.")
    _string(freeze["amendment_id"], path="compute_freeze.amendment_id", identifier=True)
    _timestamp(freeze["created_at_utc"], path="compute_freeze.created_at_utc")
    _string(freeze["protocol_id"], path="compute_freeze.protocol_id", identifier=True)
    for field in (
        "protocol_sha256",
        "prior_amendment_history_sha256",
        "source_registry_sha256",
        "runtime_authorization_sha256",
    ):
        _sha256(freeze[field], path=f"compute_freeze.{field}")
    if freeze["test_data_opened"] is not False:
        raise PolicyImprovementSchemaError("Compute freeze must precede test opening.")
    evidence = _validate_audit_evidence(
        freeze["evidence"],
        path="compute_freeze.evidence",
        expected_phase="stage0_smoke",
        expected_rows=4,
    )
    if evidence["complete_rows"] != 4 or evidence["failed_rows"] != 0:
        raise PolicyImprovementSchemaError(
            "Compute freeze requires four complete Stage-0 rows and no failures."
        )
    _validate_common_compute_targets(
        freeze["common_compute_targets"],
        path="compute_freeze.common_compute_targets",
    )
    canonical_json_bytes(freeze)
    return dict(freeze)


def _validate_selected_exact(
    value: object, *, path: str, include_alpha: bool
) -> dict[str, Any]:
    expected = {"method_id", "n", "K"} | ({"alpha"} if include_alpha else set())
    selected = _exact_fields(value, expected, path=path)
    _string(
        selected["method_id"],
        path=f"{path}.method_id",
        choices={"fixed_base_exact_persistent", "fixed_base_exact_episodic"},
    )
    if selected["n"] not in (2, 4) or selected["K"] not in (1, 5):
        raise PolicyImprovementSchemaError(
            f"{path} must select n/K from the registered screen."
        )
    if include_alpha and selected["alpha"] not in (0.05, 0.1, 0.2):
        raise PolicyImprovementSchemaError(
            f"{path}.alpha must come from the registered sensitivity grid."
        )
    return dict(selected)


def validate_screen_selection(value: object) -> dict[str, Any]:
    selection = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "amendment_id",
            "created_at_utc",
            "protocol_id",
            "protocol_sha256",
            "prior_amendment_history_sha256",
            "source_registry_sha256",
            "test_data_opened",
            "evidence",
            "selected_exact",
        },
        path="screen_selection",
    )
    if (
        selection["schema_name"] != AMENDMENT_SCHEMAS[2]
        or selection["schema_version"] != SCREEN_SELECTION_SCHEMA_VERSION
    ):
        raise PolicyImprovementSchemaError("Unexpected screen-selection schema.")
    _string(
        selection["amendment_id"], path="screen_selection.amendment_id", identifier=True
    )
    _timestamp(selection["created_at_utc"], path="screen_selection.created_at_utc")
    _string(
        selection["protocol_id"], path="screen_selection.protocol_id", identifier=True
    )
    for field in (
        "protocol_sha256",
        "prior_amendment_history_sha256",
        "source_registry_sha256",
    ):
        _sha256(selection[field], path=f"screen_selection.{field}")
    if selection["test_data_opened"] is not False:
        raise PolicyImprovementSchemaError(
            "Screen selection must precede test opening."
        )
    _validate_audit_evidence(
        selection["evidence"],
        path="screen_selection.evidence",
        expected_phase="stage1_screen",
        expected_rows=48,
    )
    _validate_selected_exact(
        selection["selected_exact"],
        path="screen_selection.selected_exact",
        include_alpha=False,
    )
    canonical_json_bytes(selection)
    return dict(selection)


_STAGE3_VARIANTS = (
    "batch_only_centering",
    "distilled_realization",
    "projection_identity",
    "contraction_enabled",
    "target_retention_0p9",
    "target_retention_0p999",
    "depth_lower_neighbor",
    "depth_upper_neighbor",
)


def stage3_method_id(variant: str, base_method_id: str) -> str:
    """Return the explicit method identity for one selected Stage 3 cell."""

    if base_method_id not in SCREEN_SELECTION_METHOD_ORDER:
        raise PolicyImprovementSchemaError(
            "Stage 3 must use the validation-selected exact base method."
        )
    if variant == "batch_only_centering":
        return STAGE3_ABLATION_METHOD_IDS[0]
    if variant == "distilled_realization":
        return STAGE3_ABLATION_METHOD_IDS[1]
    if variant in _STAGE3_VARIANTS:
        return base_method_id
    raise PolicyImprovementSchemaError("Unknown Stage 3 variant.")


def canonical_stage3_override(variant: str, selected_n: int) -> dict[str, Any]:
    overrides: dict[str, dict[str, Any]] = {
        "batch_only_centering": {
            "training_protocol": "fixed_base_ablation_batch_only_centering",
            "batch_centered_advantage": True,
            "exact_baseline_summation": False,
        },
        "distilled_realization": {
            "training_protocol": "fixed_base_ablation_distilled_realization",
            "theory_exact_mixture": False,
            "distill_mixture_policy": True,
        },
        "projection_identity": {
            "latent_projection_mode": "disabled",
            "latent_ball_radius": None,
        },
        "contraction_enabled": {
            "enable_contraction": True,
            "opnorm_clamp_interval": 0,
            "disable_value_head_norm": True,
        },
        "target_retention_0p9": {"target_ema_tau": 0.9},
        "target_retention_0p999": {"target_ema_tau": 0.999},
        "depth_lower_neighbor": {"inner_unroll_n": 1 if selected_n == 2 else 2},
        "depth_upper_neighbor": {"inner_unroll_n": 4 if selected_n == 2 else 8},
    }
    if variant not in overrides:
        raise PolicyImprovementSchemaError("Unknown Stage 3 variant.")
    return overrides[variant]


def effective_config_sha256(
    base_config: Mapping[str, object], override: Mapping[str, object]
) -> str:
    if any(key not in base_config for key in override):
        raise PolicyImprovementSchemaError(
            "Effective-config override contains an unknown base-config key."
        )
    effective = dict(base_config)
    effective.update(override)
    return hashlib.sha256(canonical_json_bytes(effective)).hexdigest()


def validate_final_selection(value: object) -> dict[str, Any]:
    selection = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "amendment_id",
            "created_at_utc",
            "protocol_id",
            "protocol_sha256",
            "prior_amendment_history_sha256",
            "source_registry_sha256",
            "test_data_opened",
            "evidence",
            "selected_exact",
            "stage3_variants",
        },
        path="final_selection",
    )
    if (
        selection["schema_name"] != AMENDMENT_SCHEMAS[3]
        or selection["schema_version"] != FINAL_SELECTION_SCHEMA_VERSION
    ):
        raise PolicyImprovementSchemaError("Unexpected final-selection schema.")
    _string(
        selection["amendment_id"], path="final_selection.amendment_id", identifier=True
    )
    _timestamp(selection["created_at_utc"], path="final_selection.created_at_utc")
    _string(
        selection["protocol_id"], path="final_selection.protocol_id", identifier=True
    )
    for field in (
        "protocol_sha256",
        "prior_amendment_history_sha256",
        "source_registry_sha256",
    ):
        _sha256(selection[field], path=f"final_selection.{field}")
    if selection["test_data_opened"] is not False:
        raise PolicyImprovementSchemaError("Final selection must precede test opening.")
    _validate_audit_evidence(
        selection["evidence"],
        path="final_selection.evidence",
        expected_phase="stage1_alpha",
        expected_rows=9,
    )
    selected = _validate_selected_exact(
        selection["selected_exact"],
        path="final_selection.selected_exact",
        include_alpha=True,
    )
    variants = selection["stage3_variants"]
    if not isinstance(variants, list) or len(variants) != len(_STAGE3_VARIANTS):
        raise PolicyImprovementSchemaError("Final Stage 3 variant inventory differs.")
    for index, expected_variant in enumerate(_STAGE3_VARIANTS):
        path = f"final_selection.stage3_variants[{index}]"
        item = _exact_fields(
            variants[index],
            {
                "variant",
                "method_id",
                "base_method_id",
                "n",
                "K",
                "alpha",
                "override_payload",
            },
            path=path,
        )
        expected_method_id = stage3_method_id(
            expected_variant, str(selected["method_id"])
        )
        if (
            item["variant"] != expected_variant
            or item["method_id"] != expected_method_id
            or item["base_method_id"] != selected["method_id"]
        ):
            raise PolicyImprovementSchemaError(
                "Stage 3 variants must use canonical order, their explicit method "
                "identity, and the selected exact base method."
            )
        expected_override = canonical_stage3_override(
            expected_variant, int(selected["n"])
        )
        if item["override_payload"] != expected_override:
            raise PolicyImprovementSchemaError(
                f"{path}.override_payload differs from the registered intervention."
            )
        expected_n = int(expected_override.get("inner_unroll_n", selected["n"]))
        if (
            item["n"] != expected_n
            or item["K"] != selected["K"]
            or item["alpha"] != selected["alpha"]
        ):
            raise PolicyImprovementSchemaError(
                f"{path} silently changes an unregistered factor."
            )
    canonical_json_bytes(selection)
    return dict(selection)


def validate_theory_design_amendment(value: object) -> dict[str, Any]:
    """Validate the pre-outcome theory design without creating an import cycle."""

    from scripts.policy_improvement_theory_schema import (
        TheoryBridgeSchemaError,
        validate_theory_amendment,
    )

    try:
        return validate_theory_amendment(value)
    except TheoryBridgeSchemaError as exc:
        raise PolicyImprovementSchemaError(str(exc)) from exc


def validate_selection_amendment(value: object) -> dict[str, Any]:
    """Compatibility name for the final, post-alpha selection document."""

    return validate_final_selection(value)


def validate_amendment_history(
    history: Sequence[object], *, protocol: Mapping[str, object] | None = None
) -> list[dict[str, Any]]:
    if len(history) > 4:
        raise PolicyImprovementSchemaError(
            "Amendment history is longer than registered."
        )
    validators = (
        validate_theory_design_amendment,
        validate_compute_freeze,
        validate_screen_selection,
        validate_final_selection,
    )
    checked: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        amendment = validators[index](raw)
        if amendment["schema_name"] != AMENDMENT_SCHEMAS[index]:
            raise PolicyImprovementSchemaError("Amendment schemas are out of order.")
        if amendment["prior_amendment_history_sha256"] != amendment_history_sha256(
            checked
        ):
            raise PolicyImprovementSchemaError(
                "Amendment does not bind the exact immutable history prefix."
            )
        if checked and amendment["protocol_id"] != checked[0]["protocol_id"]:
            raise PolicyImprovementSchemaError("Amendments name different protocols.")
        if protocol is not None:
            protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
            if (
                amendment["protocol_id"] != protocol["protocol_id"]
                or amendment["protocol_sha256"] != protocol_digest
            ):
                raise PolicyImprovementSchemaError(
                    "Amendment does not bind the exact protocol."
                )
        checked.append(amendment)
    if len(checked) >= 4:
        screen_selected = checked[2]["selected_exact"]
        final_selected = checked[3]["selected_exact"]
        for field in ("method_id", "n", "K"):
            if screen_selected[field] != final_selected[field]:
                raise PolicyImprovementSchemaError(
                    "Final selection changed a post-screen frozen factor."
                )
    return checked


def validate_protocol(value: object) -> dict[str, Any]:
    """Validate and return one strict protocol registration."""

    protocol = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "status",
            "created_date",
            "architecture",
            "output_root",
            "dataset",
            "evaluation_populations",
            "selection_rules",
            "methods",
            "seeds",
            "grid",
            "budgets",
            "statistics",
            "test_isolation",
            "amendments",
            "full_execution_gate",
        },
        path="protocol",
    )
    if protocol["schema_name"] != SCHEMA_NAME:
        raise PolicyImprovementSchemaError("Unexpected protocol schema name.")
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise PolicyImprovementSchemaError("Unsupported protocol schema version.")
    _string(protocol["protocol_id"], path="protocol.protocol_id", identifier=True)
    if protocol["status"] != "registered_not_authorized":
        raise PolicyImprovementSchemaError(
            "The committed protocol must remain registered_not_authorized."
        )
    if (
        not isinstance(protocol["created_date"], str)
        or re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", protocol["created_date"])
        is None
    ):
        raise PolicyImprovementSchemaError("created_date must be ISO YYYY-MM-DD.")

    architecture = _exact_fields(
        protocol["architecture"],
        {
            "backbone",
            "hidden_size",
            "h_cycles",
            "l_cycles",
            "l_layers",
            "puzzle_emb_ndim",
        },
        path="protocol.architecture",
    )
    if architecture != {
        "backbone": "trm",
        "hidden_size": 64,
        "h_cycles": 2,
        "l_cycles": 2,
        "l_layers": 1,
        "puzzle_emb_ndim": 0,
    }:
        raise PolicyImprovementSchemaError(
            "The registered architecture differs from the matched TRM backbone."
        )

    output_root = _exact_fields(
        protocol["output_root"],
        {"environment_variable", "relative_path"},
        path="protocol.output_root",
    )
    if output_root["environment_variable"] != "UPI_TRM_EVIDENCE_ROOT":
        raise PolicyImprovementSchemaError("Unexpected evidence-root variable.")
    relative_output = _string(
        output_root["relative_path"], path="protocol.output_root.relative_path"
    )
    if Path(relative_output).is_absolute() or ".." in Path(relative_output).parts:
        raise PolicyImprovementSchemaError("Output path must be safe and relative.")

    checked_dataset = _validate_dataset(protocol["dataset"])
    checked_populations = _validate_evaluation_populations(
        protocol["evaluation_populations"]
    )
    dataset_status = checked_dataset["manifest_sha256"]["status"]
    population_status = checked_populations["smoke"]["ordered_record_sha256"]["status"]
    if dataset_status != population_status:
        raise PolicyImprovementSchemaError(
            "Dataset and evaluation-population identities must freeze together."
        )
    if dataset_status == "available":
        for tier, split in (("pilot", "validation"), ("confirmatory", "test")):
            if (
                checked_populations[tier]["ordered_record_sha256"]["value"]
                != checked_dataset["splits"][split]["ordered_record_sha256"]["value"]
            ):
                raise PolicyImprovementSchemaError(
                    f"The full {tier} population must equal the frozen {split} split."
                )
    _validate_methods(protocol["methods"])

    selection_rules = _exact_fields(
        protocol["selection_rules"],
        {"screen", "alpha"},
        path="protocol.selection_rules",
    )
    screen_rule = _exact_fields(
        selection_rules["screen"],
        {
            "id",
            "endpoint",
            "eligible_methods",
            "maximize",
            "tie_break",
        },
        path="protocol.selection_rules.screen",
    )
    if screen_rule != {
        "id": SCREEN_SELECTION_RULE_ID,
        "endpoint": SELECTION_ENDPOINT,
        "eligible_methods": list(SCREEN_SELECTION_METHOD_ORDER),
        "maximize": True,
        "tie_break": ["method_order", "n_ascending", "K_ascending"],
    }:
        raise PolicyImprovementSchemaError(
            "The screen-selection rule differs from the registered deterministic rule."
        )
    alpha_rule = _exact_fields(
        selection_rules["alpha"],
        {"id", "endpoint", "maximize", "tie_break"},
        path="protocol.selection_rules.alpha",
    )
    if alpha_rule != {
        "id": ALPHA_SELECTION_RULE_ID,
        "endpoint": SELECTION_ENDPOINT,
        "maximize": True,
        "tie_break": ["alpha_ascending"],
    }:
        raise PolicyImprovementSchemaError(
            "The alpha-selection rule differs from the registered deterministic rule."
        )

    seeds = _exact_fields(
        protocol["seeds"],
        {"derivation", "namespace", "smoke", "pilot", "confirmatory"},
        path="protocol.seeds",
    )
    if seeds["derivation"] != "sha256_namespace_tier_index_u32be_v1":
        raise PolicyImprovementSchemaError("Unexpected seed derivation.")
    _string(seeds["namespace"], path="protocol.seeds.namespace", identifier=True)
    _integer_list(seeds["smoke"], path="protocol.seeds.smoke", exact_length=1)
    _integer_list(seeds["pilot"], path="protocol.seeds.pilot", exact_length=3)
    _integer_list(
        seeds["confirmatory"],
        path="protocol.seeds.confirmatory",
        exact_length=8,
    )

    grid = _exact_fields(
        protocol["grid"],
        {"stage0", "stage1_screen", "stage1_alpha", "stage2", "stage3"},
        path="protocol.grid",
    )
    stage0 = _exact_fields(
        grid["stage0"],
        {
            "method_ids",
            "n",
            "K",
            "alpha",
            "expected_rows",
            "ppo_rollout_environment_interactions",
            "upi_optimizer_batch_size",
            "upi_rollout_episodes_per_step",
        },
        path="protocol.grid.stage0",
    )
    if _string_list(
        stage0["method_ids"], path="protocol.grid.stage0.method_ids"
    ) != list(METHOD_IDS):
        raise PolicyImprovementSchemaError("Stage 0 must cover all methods once.")
    if stage0["expected_rows"] != 4:
        raise PolicyImprovementSchemaError("Stage 0 row count must be four.")
    _integer(stage0["n"], path="protocol.grid.stage0.n", minimum=1)
    _integer(stage0["K"], path="protocol.grid.stage0.K", minimum=1)
    _number(
        stage0["alpha"], path="protocol.grid.stage0.alpha", minimum=0.0, maximum=1.0
    )
    if stage0["ppo_rollout_environment_interactions"] != 16:
        raise PolicyImprovementSchemaError(
            "Stage 0 PPO rollout must match one 16-interaction segment."
        )
    if (
        stage0["upi_optimizer_batch_size"] != 1
        or stage0["upi_rollout_episodes_per_step"] != 1
    ):
        raise PolicyImprovementSchemaError(
            "Stage 0 must optimize every UPI rollout so both segments exercise updates."
        )

    screen = _exact_fields(
        grid["stage1_screen"],
        {"method_ids", "n_values", "K_values", "alpha", "expected_rows"},
        path="protocol.grid.stage1_screen",
    )
    if _string_list(
        screen["method_ids"], path="protocol.grid.stage1_screen.method_ids"
    ) != list(METHOD_IDS):
        raise PolicyImprovementSchemaError("Stage 1 screen method inventory differs.")
    if screen["n_values"] != [2, 4] or screen["K_values"] != [1, 5]:
        raise PolicyImprovementSchemaError("Stage 1 n/K grid differs.")
    _number(
        screen["alpha"],
        path="protocol.grid.stage1_screen.alpha",
        minimum=0.0,
        maximum=1.0,
    )
    if screen["expected_rows"] != 48:
        raise PolicyImprovementSchemaError("Stage 1 screen row count must be 48.")

    alpha = _exact_fields(
        grid["stage1_alpha"],
        {"selection_rule", "alpha_values", "expected_rows"},
        path="protocol.grid.stage1_alpha",
    )
    if alpha["selection_rule"] != SCREEN_SELECTION_RULE_ID:
        raise PolicyImprovementSchemaError(
            "Stage 1 alpha rows must use the registered screen-selection rule."
        )
    if alpha["alpha_values"] != [0.05, 0.1, 0.2] or alpha["expected_rows"] != 9:
        raise PolicyImprovementSchemaError("Stage 1 alpha grid differs.")

    stage2 = _exact_fields(
        grid["stage2"],
        {"selection_rule", "method_ids", "expected_rows"},
        path="protocol.grid.stage2",
    )
    if stage2["selection_rule"] != ALPHA_SELECTION_RULE_ID:
        raise PolicyImprovementSchemaError(
            "Stage 2 rows must use the registered alpha-selection rule."
        )
    if (
        _string_list(stage2["method_ids"], path="protocol.grid.stage2.method_ids")
        != list(METHOD_IDS)
        or stage2["expected_rows"] != 32
    ):
        raise PolicyImprovementSchemaError("Stage 2 design differs.")

    stage3 = _exact_fields(
        grid["stage3"],
        {"selection_rule", "training_variants", "reused_contrasts", "expected_rows"},
        path="protocol.grid.stage3",
    )
    _string(
        stage3["selection_rule"],
        path="protocol.grid.stage3.selection_rule",
        identifier=True,
    )
    variants = _string_list(
        stage3["training_variants"],
        path="protocol.grid.stage3.training_variants",
        exact_length=8,
    )
    if set(variants) != {
        "batch_only_centering",
        "distilled_realization",
        "projection_identity",
        "contraction_enabled",
        "target_retention_0p9",
        "target_retention_0p999",
        "depth_lower_neighbor",
        "depth_upper_neighbor",
    }:
        raise PolicyImprovementSchemaError("Stage 3 training variants differ.")
    reused = _string_list(
        stage3["reused_contrasts"],
        path="protocol.grid.stage3.reused_contrasts",
        exact_length=2,
    )
    if set(reused) != {
        "persistent_vs_episodic",
        "exact_mixture_vs_parameter_interpolation",
    }:
        raise PolicyImprovementSchemaError("Stage 3 reused contrasts differ.")
    if stage3["expected_rows"] != 64:
        raise PolicyImprovementSchemaError("Stage 3 row count must be 64.")

    budgets = _exact_fields(
        protocol["budgets"],
        {"smoke", "pilot", "confirmatory", "compute_matching"},
        path="protocol.budgets",
    )
    for tier in ("smoke", "pilot", "confirmatory"):
        budget = _exact_fields(
            budgets[tier],
            {
                "environment_interactions",
                "evaluation_records",
                "checkpoint_environment_interactions",
            },
            path=f"protocol.budgets.{tier}",
        )
        interactions = _integer(
            budget["environment_interactions"],
            path=f"protocol.budgets.{tier}.environment_interactions",
            minimum=1,
        )
        _integer(
            budget["evaluation_records"],
            path=f"protocol.budgets.{tier}.evaluation_records",
            minimum=1,
        )
        checkpoints = budget["checkpoint_environment_interactions"]
        if not isinstance(checkpoints, list) or not checkpoints:
            raise PolicyImprovementSchemaError(f"{tier} checkpoint schedule is empty.")
        checked = [
            _integer(
                item,
                path=f"protocol.budgets.{tier}.checkpoint_environment_interactions[{index}]",
                minimum=1,
            )
            for index, item in enumerate(checkpoints)
        ]
        if checked != sorted(set(checked)) or checked[-1] != interactions:
            raise PolicyImprovementSchemaError(
                f"{tier} checkpoint schedule must be unique, sorted, and end at the budget."
            )
    matching = _exact_fields(
        budgets["compute_matching"],
        {
            "interaction_matched",
            "compute_matched",
            "measured_smoke_required",
            "compute_unit",
            "maximum_relative_mismatch",
        },
        path="protocol.budgets.compute_matching",
    )
    if (
        matching["interaction_matched"] is not True
        or matching["compute_matched"] is not True
        or matching["measured_smoke_required"] is not True
        or matching["compute_unit"] != "recurrent_map_applications"
        or _number(
            matching["maximum_relative_mismatch"],
            path="protocol.budgets.compute_matching.maximum_relative_mismatch",
            minimum=0.0,
            maximum=1.0,
        )
        != 0.05
    ):
        raise PolicyImprovementSchemaError(
            "Both matching analyses and smoke measurement are required."
        )

    statistics = _exact_fields(
        protocol["statistics"],
        {
            "primary_contrast",
            "primary_endpoint",
            "prespecified_secondary_contrasts",
            "confidence_level",
            "bootstrap",
            "seed_level_permutation_test",
            "multiplicity",
        },
        path="protocol.statistics",
    )
    if (
        statistics["primary_contrast"]
        != "fixed_base_exact_persistent-minus-matched_ppo"
        or statistics["primary_endpoint"] != "final_test_solve_rate_percentage_points"
    ):
        raise PolicyImprovementSchemaError("Primary analysis changed.")
    if statistics["prespecified_secondary_contrasts"] != [
        "fixed_base_exact_episodic-minus-matched_ppo",
        "legacy_parameter_interpolation-minus-matched_ppo",
    ]:
        raise PolicyImprovementSchemaError("Pre-specified secondary contrasts changed.")
    if (
        _number(
            statistics["confidence_level"], path="protocol.statistics.confidence_level"
        )
        != 0.95
    ):
        raise PolicyImprovementSchemaError("Confidence level must be 0.95.")
    bootstrap = _exact_fields(
        statistics["bootstrap"],
        {"scheme", "replicates", "seed"},
        path="protocol.statistics.bootstrap",
    )
    if (
        bootstrap["scheme"] != "paired_seed_cluster_then_paired_puzzle_v1"
        or bootstrap["replicates"] != 10000
    ):
        raise PolicyImprovementSchemaError("Bootstrap procedure changed.")
    _integer(bootstrap["seed"], path="protocol.statistics.bootstrap.seed")
    if statistics["seed_level_permutation_test"] is not True:
        raise PolicyImprovementSchemaError("Seed-level permutation test is required.")
    multiplicity = _exact_fields(
        statistics["multiplicity"],
        {"primary", "prespecified_secondary", "other"},
        path="protocol.statistics.multiplicity",
    )
    if multiplicity != {
        "primary": "unadjusted",
        "prespecified_secondary": "holm",
        "other": "exploratory",
    }:
        raise PolicyImprovementSchemaError("Multiplicity policy changed.")

    isolation = _exact_fields(
        protocol["test_isolation"],
        {
            "pilot_split",
            "confirmatory_split",
            "test_open_count",
            "amendment_policy",
            "owner_root_binding",
        },
        path="protocol.test_isolation",
    )
    if isolation != {
        "pilot_split": "validation",
        "confirmatory_split": "test",
        "test_open_count": 1,
        "amendment_policy": "document_before_consistent_rerun_of_all_affected_cells",
        "owner_root_binding": "evidence_root",
    }:
        raise PolicyImprovementSchemaError("Test-isolation policy changed.")
    if protocol["amendments"] != []:
        raise PolicyImprovementSchemaError(
            "The base protocol is immutable and pre-selection. Use a strict "
            "policy_improvement_selection_amendment_v1 artifact instead."
        )
    gate = _exact_fields(
        protocol["full_execution_gate"],
        {"environment_variable", "required_value"},
        path="protocol.full_execution_gate",
    )
    if gate != {
        "environment_variable": "RUN_UPITRM_FULL_EXPERIMENTS",
        "required_value": "1",
    }:
        raise PolicyImprovementSchemaError("Full execution gate changed.")
    canonical_json_bytes(protocol)
    return dict(protocol)


def validate_registry_row(value: object) -> dict[str, Any]:
    """Validate one deterministic concrete row or selection template."""

    row = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "row_kind",
            "phase",
            "tier",
            "run_id",
            "method_id",
            "base_method_id",
            "seed",
            "evaluation_split",
            "n",
            "K",
            "alpha",
            "ablation_variant",
            "selection_rule",
            "factor_applicability",
            "paper_evidence_eligible",
            "config_override",
            "base_config_canonical_sha256",
            "expected_effective_config_sha256",
        },
        path="registry_row",
    )
    if (
        row["schema_name"] != SCHEMA_NAME
        or row["schema_version"] != REGISTRY_ROW_SCHEMA_VERSION
    ):
        raise PolicyImprovementSchemaError("Unsupported registry-row schema.")
    _string(row["protocol_id"], path="registry_row.protocol_id", identifier=True)
    kind = _string(
        row["row_kind"],
        path="registry_row.row_kind",
        choices={"concrete", "selection_template"},
    )
    phase = _string(
        row["phase"],
        path="registry_row.phase",
        choices=set(PHASE_CONTRACTS),
    )
    tier = _string(row["tier"], path="registry_row.tier", choices=set(TIERS))
    _string(row["run_id"], path="registry_row.run_id", identifier=True)
    method = row["method_id"]
    if method is not None:
        _string(method, path="registry_row.method_id", choices=set(RESULT_METHOD_IDS))
    base_method = row["base_method_id"]
    if base_method is not None:
        _string(
            base_method,
            path="registry_row.base_method_id",
            choices=set(METHOD_IDS),
        )
    _integer(row["seed"], path="registry_row.seed")
    split = _string(
        row["evaluation_split"],
        path="registry_row.evaluation_split",
        choices=set(EVALUATION_SPLITS),
    )
    expected_tier, expected_split, expected_evidence = PHASE_CONTRACTS[phase]
    if tier != expected_tier or split != expected_split:
        raise PolicyImprovementSchemaError(
            "Registry phase, tier, and evaluation split are inconsistent."
        )
    for field in ("n", "K"):
        if row[field] is not None:
            _integer(row[field], path=f"registry_row.{field}", minimum=1)
    if row["alpha"] is not None:
        _number(row["alpha"], path="registry_row.alpha", minimum=0.0, maximum=1.0)
    if row["ablation_variant"] is not None:
        _string(
            row["ablation_variant"],
            path="registry_row.ablation_variant",
            identifier=True,
        )
    if row["selection_rule"] is not None:
        _string(
            row["selection_rule"], path="registry_row.selection_rule", identifier=True
        )
    factors = _exact_fields(
        row["factor_applicability"],
        {"n", "K", "alpha", "ablation"},
        path="registry_row.factor_applicability",
    )
    if any(not isinstance(factors[field], bool) for field in factors):
        raise PolicyImprovementSchemaError(
            "Factor applicability values must be booleans."
        )
    if not isinstance(row["paper_evidence_eligible"], bool):
        raise PolicyImprovementSchemaError("paper_evidence_eligible must be boolean.")
    if row["paper_evidence_eligible"] is not expected_evidence:
        raise PolicyImprovementSchemaError(
            "paper_evidence_eligible differs from the phase contract."
        )
    if kind == "concrete" and (method is None or row["selection_rule"] is not None):
        raise PolicyImprovementSchemaError(
            "Concrete rows need a method and no selection rule."
        )
    if kind == "selection_template" and row["selection_rule"] is None:
        raise PolicyImprovementSchemaError("Selection templates need a rule.")
    if phase in {"stage0_smoke", "stage1_screen"} and kind != "concrete":
        raise PolicyImprovementSchemaError(
            "Stage 0 and the Stage 1 screen must be concrete."
        )
    if method is not None:
        expected_factors = {
            "n": True,
            "K": method != "matched_ppo",
            "alpha": method != "matched_ppo",
            "ablation": phase == "stage3_ablation",
        }
    else:
        expected_factors = {
            "n": True,
            "K": True,
            "alpha": True,
            "ablation": phase == "stage3_ablation",
        }
    if dict(factors) != expected_factors:
        raise PolicyImprovementSchemaError(
            "factor_applicability differs from the phase/method contract."
        )
    if phase == "stage3_ablation" and row["ablation_variant"] is None:
        raise PolicyImprovementSchemaError("Stage 3 rows require an ablation variant.")
    if phase != "stage3_ablation" and row["ablation_variant"] is not None:
        raise PolicyImprovementSchemaError(
            "Only Stage 3 rows may identify an ablation variant."
        )
    if kind == "selection_template" and base_method is not None:
        raise PolicyImprovementSchemaError(
            "Selection templates cannot name a concrete base method."
        )
    if kind == "concrete":
        if base_method is None:
            raise PolicyImprovementSchemaError(
                "Concrete rows require an explicit canonical base method."
            )
        if phase == "stage3_ablation":
            expected_method = stage3_method_id(
                str(row["ablation_variant"]), str(base_method)
            )
            if method != expected_method:
                raise PolicyImprovementSchemaError(
                    "Stage 3 method identity differs from its registered intervention."
                )
        elif method != base_method:
            raise PolicyImprovementSchemaError(
                "Only Stage 3 interventions may differ from their base method."
            )
    config_fields = (
        row["config_override"],
        row["base_config_canonical_sha256"],
        row["expected_effective_config_sha256"],
    )
    if kind == "concrete":
        override = _mapping(row["config_override"], path="registry_row.config_override")
        canonical_json_bytes(dict(override))
        if not override:
            raise PolicyImprovementSchemaError(
                "Every concrete row requires its nonempty exact config override."
            )
        _sha256(
            row["base_config_canonical_sha256"],
            path="registry_row.base_config_canonical_sha256",
        )
        _sha256(
            row["expected_effective_config_sha256"],
            path="registry_row.expected_effective_config_sha256",
        )
    elif config_fields != (None, None, None):
        raise PolicyImprovementSchemaError(
            "Selection templates cannot carry effective-config identities."
        )
    seed = int(row["seed"])
    run_id = str(row["run_id"])
    if phase in {"stage0_smoke", "stage1_screen"}:
        if any(row[field] is None for field in ("n", "K", "alpha")):
            raise PolicyImprovementSchemaError(
                "Concrete smoke/screen rows require n, K, and alpha."
            )
        expected_run_id = (
            f"s0-{method}-s{seed}"
            if phase == "stage0_smoke"
            else f"s1-{method}-n{row['n']}-k{row['K']}-s{seed}"
        )
    elif phase == "stage1_alpha":
        if kind == "selection_template":
            alpha_indices = {0.05: 0, 0.1: 1, 0.2: 2}
            if row["alpha"] not in alpha_indices or any(
                row[field] is not None
                for field in ("method_id", "base_method_id", "n", "K")
            ):
                raise PolicyImprovementSchemaError(
                    "Stage 1 alpha template factors differ."
                )
            expected_run_id = f"s1a-a{alpha_indices[float(row['alpha'])]}-s{seed}"
        else:
            if any(row[field] is None for field in ("method_id", "n", "K", "alpha")):
                raise PolicyImprovementSchemaError(
                    "Materialized alpha rows require all selected factors."
                )
            alpha_token = format(float(row["alpha"]), ".15g").replace(".", "p")
            expected_run_id = (
                f"s1a-{method}-n{row['n']}-k{row['K']}-a{alpha_token}-s{seed}"
            )
    elif phase == "stage2_confirmatory":
        if kind == "selection_template":
            if (
                method is None
                or base_method is not None
                or any(row[field] is not None for field in ("n", "K", "alpha"))
            ):
                raise PolicyImprovementSchemaError("Stage 2 template factors differ.")
            expected_run_id = f"s2-{method}-s{seed}"
        else:
            if any(row[field] is None for field in ("method_id", "n", "K", "alpha")):
                raise PolicyImprovementSchemaError(
                    "Materialized Stage 2 rows require all selected factors."
                )
            alpha_token = format(float(row["alpha"]), ".15g").replace(".", "p")
            expected_run_id = (
                f"s2-{method}-n{row['n']}-k{row['K']}-a{alpha_token}-s{seed}"
            )
    else:
        variant = str(row["ablation_variant"])
        if kind == "selection_template":
            if (
                method is not None
                or base_method is not None
                or any(row[field] is not None for field in ("n", "K", "alpha"))
            ):
                raise PolicyImprovementSchemaError("Stage 3 template factors differ.")
            expected_run_id = f"s3-{variant}-s{seed}"
        else:
            if any(row[field] is None for field in ("method_id", "n", "K", "alpha")):
                raise PolicyImprovementSchemaError(
                    "Materialized Stage 3 rows require all selected factors."
                )
            alpha_token = format(float(row["alpha"]), ".15g").replace(".", "p")
            expected_run_id = (
                f"s3-{variant}-{base_method}-n{row['n']}-k{row['K']}-"
                f"a{alpha_token}-s{seed}"
            )
    if run_id != expected_run_id:
        raise PolicyImprovementSchemaError(
            "Registry run ID does not encode its exact phase, method, factors, and seed."
        )
    canonical_json_bytes(row)
    return dict(row)


def _validate_result_identities(value: object, *, status: str) -> dict[str, Any]:
    identities = _exact_fields(
        value,
        {
            "producer_git_commit",
            "git_clean",
            "runtime_authorization_sha256",
            "training_source_git_commit",
            "training_runtime_sha256",
            "training_runtime_profile_sha256",
            "training_selected_source_manifest_sha256",
            "launcher_sha256",
            "producer_manifest_sha256",
            "method_config_sha256",
            "effective_config_sha256",
            "dataset_manifest_sha256",
            "train_ordered_records_sha256",
            "evaluation_ordered_records_sha256",
            "initialization_sha256",
            "checkpoint_sha256",
            "model_state_sha256",
            "evaluation_runtime_sha256",
            "evaluation_source_git_commit",
            "evaluation_runtime_profile_sha256",
            "evaluation_selected_source_manifest_sha256",
            "evaluation_pool_sha256",
            "test_open_sha256",
            "device",
        },
        path="result.identities",
    )
    _git_commit(
        identities["producer_git_commit"], path="result.identities.producer_git_commit"
    )
    _git_commit(
        identities["training_source_git_commit"],
        path="result.identities.training_source_git_commit",
    )
    if identities["training_source_git_commit"] != identities["producer_git_commit"]:
        raise PolicyImprovementSchemaError(
            "Result training source commit must equal its producer commit."
        )
    if identities["git_clean"] is not True:
        raise PolicyImprovementSchemaError("Result producer worktree must be clean.")
    for field in (
        "runtime_authorization_sha256",
        "training_runtime_sha256",
        "training_runtime_profile_sha256",
        "training_selected_source_manifest_sha256",
        "launcher_sha256",
        "producer_manifest_sha256",
        "method_config_sha256",
        "effective_config_sha256",
        "dataset_manifest_sha256",
        "train_ordered_records_sha256",
        "evaluation_ordered_records_sha256",
    ):
        _sha256(identities[field], path=f"result.identities.{field}")
    initialization = identities["initialization_sha256"]
    if status == "failed" and isinstance(initialization, Mapping):
        checked_initialization = _validate_available_sha256(
            initialization,
            path="result.identities.initialization_sha256",
        )
        if checked_initialization["status"] != "unavailable":
            raise PolicyImprovementSchemaError(
                "Failed initialization availability must be unavailable."
            )
    else:
        _sha256(initialization, path="result.identities.initialization_sha256")
    for field in (
        "checkpoint_sha256",
        "model_state_sha256",
        "evaluation_runtime_sha256",
        "evaluation_runtime_profile_sha256",
        "evaluation_selected_source_manifest_sha256",
        "evaluation_pool_sha256",
        "test_open_sha256",
    ):
        _validate_available_sha256(identities[field], path=f"result.identities.{field}")
    _validate_availability(
        identities["evaluation_source_git_commit"],
        path="result.identities.evaluation_source_git_commit",
        kind="git_commit",
    )
    device = identities["device"]
    if status == "failed" and isinstance(device, Mapping):
        checked_device = _validate_availability(
            device,
            path="result.identities.device",
            kind="string",
        )
        if checked_device["status"] != "unavailable":
            raise PolicyImprovementSchemaError(
                "Failed execution-device availability must be unavailable."
            )
    else:
        _string(device, path="result.identities.device")
    return dict(identities)


def policy_variants_for_method(method_id: str) -> tuple[str, ...]:
    """Return the canonical evaluation inventory for one result method."""

    if method_id in {
        "fixed_base_exact_persistent",
        "fixed_base_exact_episodic",
        "fixed_base_batch_only_centering",
    }:
        return ("base", "candidate", "exact_mixture")
    if method_id in {
        "legacy_parameter_interpolation",
        "fixed_base_distilled_realization",
    }:
        return ("base", "candidate", "realized_policy")
    if method_id == "matched_ppo":
        return ("realized_policy",)
    raise AssertionError(f"Unknown method {method_id!r}.")


def primary_policy_variant_for_method(method_id: str) -> str:
    """Return the policy variant that owns the registered primary endpoint."""

    variants = policy_variants_for_method(method_id)
    return "exact_mixture" if "exact_mixture" in variants else "realized_policy"


def _validate_policy_evaluations(
    value: object,
    *,
    status: str,
    method_id: str,
    run_id: str,
    evaluation_pool_sha256: Mapping[str, object],
) -> list[dict[str, Any]]:
    expected_variants = policy_variants_for_method(method_id)
    if not isinstance(value, list) or len(value) != len(expected_variants):
        raise PolicyImprovementSchemaError(
            "Policy-evaluation inventory differs from the registered method."
        )
    evaluations: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        path = f"result.policy_evaluations[{index}]"
        item = _exact_fields(
            raw,
            {
                "evaluation_id",
                "policy_variant",
                "evaluation_pool_sha256",
                "evaluation_artifact_sha256",
                "per_instance_artifact_sha256",
                "primary",
                "secondary",
            },
            path=path,
        )
        variant = _string(
            item["policy_variant"],
            path=f"{path}.policy_variant",
            choices={"base", "candidate", "exact_mixture", "realized_policy"},
        )
        if variant != expected_variants[index]:
            raise PolicyImprovementSchemaError(
                "Policy evaluations must use the registered canonical order."
            )
        evaluation_id = _string(
            item["evaluation_id"], path=f"{path}.evaluation_id", identifier=True
        )
        if evaluation_id != f"{run_id}.{variant}":
            raise PolicyImprovementSchemaError(
                "Policy evaluation ID must uniquely bind run and variant."
            )
        pool = _validate_available_sha256(
            item["evaluation_pool_sha256"],
            path=f"{path}.evaluation_pool_sha256",
        )
        evaluation_artifact = _validate_available_sha256(
            item["evaluation_artifact_sha256"],
            path=f"{path}.evaluation_artifact_sha256",
        )
        per_instance_artifact = _validate_available_sha256(
            item["per_instance_artifact_sha256"],
            path=f"{path}.per_instance_artifact_sha256",
        )
        primary = _exact_fields(
            item["primary"],
            {"solve_rate", "solved_count", "denominator"},
            path=f"{path}.primary",
        )
        solve_rate = _validate_availability(
            primary["solve_rate"],
            path=f"{path}.primary.solve_rate",
            kind="number",
            minimum=0.0,
            maximum=1.0,
        )
        solved = _validate_availability(
            primary["solved_count"],
            path=f"{path}.primary.solved_count",
            kind="integer",
        )
        denominator = _validate_availability(
            primary["denominator"],
            path=f"{path}.primary.denominator",
            kind="integer",
            minimum=1,
        )
        secondary = _exact_fields(
            item["secondary"],
            {
                "discounted_return_mean",
                "edits_to_solve_mean",
                "terminal_reason_counts",
                "value_calibration",
            },
            path=f"{path}.secondary",
        )
        discounted_return = _validate_availability(
            secondary["discounted_return_mean"],
            path=f"{path}.secondary.discounted_return_mean",
            kind="number",
        )
        edits_to_solve = _validate_availability(
            secondary["edits_to_solve_mean"],
            path=f"{path}.secondary.edits_to_solve_mean",
            kind="number",
            minimum=0.0,
        )
        terminal_counts = _validate_availability(
            secondary["terminal_reason_counts"],
            path=f"{path}.secondary.terminal_reason_counts",
            kind="terminal_reason_counts",
        )
        calibration = _validate_availability(
            secondary["value_calibration"],
            path=f"{path}.secondary.value_calibration",
            kind="number",
            minimum=0.0,
        )
        endpoint_values = (
            solve_rate,
            solved,
            denominator,
            discounted_return,
            terminal_counts,
            pool,
            evaluation_artifact,
            per_instance_artifact,
        )
        expected_status = "available" if status == "complete" else "unavailable"
        if any(endpoint["status"] != expected_status for endpoint in endpoint_values):
            raise PolicyImprovementSchemaError(
                "Policy evaluation availability is inconsistent with run status."
            )
        if status == "complete":
            solved_value = int(solved["value"])
            denominator_value = int(denominator["value"])
            if solved_value > denominator_value or not math.isclose(
                float(solve_rate["value"]),
                solved_value / denominator_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise PolicyImprovementSchemaError(
                    "Policy solve-rate fields are internally inconsistent."
                )
            counts = terminal_counts["value"]
            if (
                sum(int(counts[name]) for name in ("budget", "solved", "stop"))
                != denominator_value
            ):
                raise PolicyImprovementSchemaError(
                    "Terminal-reason counts do not match the denominator."
                )
            if int(counts["solved"]) != solved_value:
                raise PolicyImprovementSchemaError(
                    "Solved terminal count differs from solved_count."
                )
            if solved_value == 0:
                if edits_to_solve != {
                    "status": "unavailable",
                    "reason": "not_applicable",
                }:
                    raise PolicyImprovementSchemaError(
                        "Edits-to-solve must be unavailable when no puzzle was solved."
                    )
            elif edits_to_solve["status"] != "available":
                raise PolicyImprovementSchemaError(
                    "Solved evaluations require an edits-to-solve mean."
                )
            if calibration["status"] == "unavailable" and calibration != {
                "status": "unavailable",
                "reason": "not_collected_by_registered_protocol",
            }:
                raise PolicyImprovementSchemaError(
                    "Unavailable value calibration must identify the registered omission."
                )
            if (
                evaluation_pool_sha256["status"] != "available"
                or pool["value"] != evaluation_pool_sha256["value"]
            ):
                raise PolicyImprovementSchemaError(
                    "Policy evaluation uses a different held-out pool."
                )
        evaluations.append(dict(item))
    return evaluations


def _validate_evaluation_snapshots(
    value: object,
    *,
    status: str,
    tier: str,
    method_id: str,
    run_id: str,
    evaluation_pool_sha256: Mapping[str, object],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise PolicyImprovementSchemaError(
            "Every result must contain interaction- and compute-matched snapshots."
        )
    snapshots: list[dict[str, Any]] = []
    for index, expected_kind in enumerate(("interaction_matched", "compute_matched")):
        path = f"result.evaluation_snapshots[{index}]"
        snapshot = _exact_fields(
            value[index],
            {
                "snapshot_kind",
                "status",
                "unavailable_reason",
                "target",
                "observed_environment_interactions",
                "observed_recurrent_map_applications",
                "accelerator_seconds_observed",
                "checkpoint_sha256",
                "model_state_sha256",
                "checkpoint_lineage_sha256",
                "policy_evaluations",
            },
            path=path,
        )
        if snapshot["snapshot_kind"] != expected_kind:
            raise PolicyImprovementSchemaError(
                "Evaluation snapshots must use canonical interaction/compute order."
            )
        snapshot_status = _string(
            snapshot["status"],
            path=f"{path}.status",
            choices={"available", "unavailable"},
        )
        target = _exact_fields(
            snapshot["target"],
            {"unit", "registered_quantity"},
            path=f"{path}.target",
        )
        expected_unit = (
            "environment_interactions"
            if expected_kind == "interaction_matched"
            else "recurrent_map_applications"
        )
        if target["unit"] != expected_unit:
            raise PolicyImprovementSchemaError(
                f"{path}.target.unit differs from its matching basis."
            )
        registered = _validate_availability(
            target["registered_quantity"],
            path=f"{path}.target.registered_quantity",
            kind="integer",
            minimum=1,
        )
        observed_interactions = _validate_availability(
            snapshot["observed_environment_interactions"],
            path=f"{path}.observed_environment_interactions",
            kind="integer",
            minimum=0,
        )
        observed_compute = _validate_availability(
            snapshot["observed_recurrent_map_applications"],
            path=f"{path}.observed_recurrent_map_applications",
            kind="integer",
            minimum=0,
        )
        accelerator = _validate_availability(
            snapshot["accelerator_seconds_observed"],
            path=f"{path}.accelerator_seconds_observed",
            kind="number",
            minimum=0.0,
        )
        checkpoint = _validate_available_sha256(
            snapshot["checkpoint_sha256"], path=f"{path}.checkpoint_sha256"
        )
        model_state = _validate_available_sha256(
            snapshot["model_state_sha256"], path=f"{path}.model_state_sha256"
        )
        lineage = _validate_available_sha256(
            snapshot["checkpoint_lineage_sha256"],
            path=f"{path}.checkpoint_lineage_sha256",
        )
        if snapshot_status == "available":
            if snapshot["unavailable_reason"] is not None:
                raise PolicyImprovementSchemaError(
                    f"{path} cannot carry an unavailable reason when available."
                )
            if status != "complete":
                raise PolicyImprovementSchemaError(
                    "Failed runs cannot publish available evaluation snapshots."
                )
            if any(
                item["status"] != "available"
                for item in (
                    registered,
                    observed_interactions,
                    observed_compute,
                    accelerator,
                    checkpoint,
                    model_state,
                    lineage,
                )
            ):
                raise PolicyImprovementSchemaError(
                    f"{path} is available but one of its required measurements is not."
                )
            if expected_kind == "interaction_matched":
                if int(observed_interactions["value"]) != int(registered["value"]):
                    raise PolicyImprovementSchemaError(
                        "Interaction-matched snapshot missed its exact target."
                    )
            else:
                target_value = int(registered["value"])
                observed_value = int(observed_compute["value"])
                if abs(observed_value - target_value) / target_value > 0.05:
                    raise PolicyImprovementSchemaError(
                        "Compute-matched snapshot exceeds five percent tolerance."
                    )
            evaluations = _validate_policy_evaluations(
                snapshot["policy_evaluations"],
                status="complete",
                method_id=method_id,
                run_id=f"{run_id}.{expected_kind}",
                evaluation_pool_sha256=evaluation_pool_sha256,
            )
        else:
            reason = _string(
                snapshot["unavailable_reason"],
                path=f"{path}.unavailable_reason",
                choices={
                    "compute_snapshot_not_registered",
                    "run_failed_before_checkpoint",
                    "run_failed_before_evaluation",
                    "run_failed_before_measurement",
                },
            )
            if expected_kind == "interaction_matched" and status == "complete":
                raise PolicyImprovementSchemaError(
                    "Complete runs require an interaction-matched snapshot."
                )
            if (
                expected_kind == "compute_matched"
                and status == "complete"
                and tier != "smoke"
            ):
                raise PolicyImprovementSchemaError(
                    "Complete non-smoke runs require a compute-matched snapshot."
                )
            if (
                expected_kind == "compute_matched"
                and status == "complete"
                and reason != "compute_snapshot_not_registered"
            ):
                raise PolicyImprovementSchemaError(
                    "Smoke compute snapshots need the registered unavailable reason."
                )
            if snapshot["policy_evaluations"] != []:
                raise PolicyImprovementSchemaError(
                    "Unavailable snapshots cannot contain policy evaluations."
                )
            unavailable_values = (
                registered,
                observed_interactions,
                observed_compute,
                accelerator,
                checkpoint,
                model_state,
                lineage,
            )
            if any(item["status"] != "unavailable" for item in unavailable_values):
                raise PolicyImprovementSchemaError(
                    "Unavailable snapshots cannot publish measurements or artifacts."
                )
            evaluations = []
        snapshots.append({**dict(snapshot), "policy_evaluations": evaluations})
    return snapshots


def _validate_metrics(
    value: object,
    *,
    status: str,
    device: str,
) -> dict[str, Any]:
    metrics = _exact_fields(
        value,
        {"training", "diagnostics"},
        path="result.metrics",
    )
    training = _exact_fields(
        metrics["training"],
        {
            "interactions_to_first_solve",
            "value_loss",
            "policy_loss",
            "wall_time_seconds",
            "gpu_hours",
            "gpu_utilization_fraction",
            "gpu_utilization_sample_count",
            "gpu_utilization_sampling_interval_seconds",
            "peak_allocated_memory_bytes",
            "peak_reserved_memory_bytes",
            "optimizer_steps",
            "cells_processed",
            "actions_processed",
            "tokens_processed",
            "policy_head_calls",
            "recurrent_map_applications",
            "value_head_calls",
        },
        path="result.metrics.training",
    )
    for field in (
        "value_loss",
        "policy_loss",
        "wall_time_seconds",
        "gpu_hours",
    ):
        availability = _validate_availability(
            training[field],
            path=f"result.metrics.training.{field}",
            kind="number",
            minimum=(0.0 if field in {"wall_time_seconds", "gpu_hours"} else None),
        )
        if (
            status == "complete"
            and field in {"wall_time_seconds", "gpu_hours"}
            and availability["status"] != "available"
        ):
            raise PolicyImprovementSchemaError(
                f"Complete runs require result.metrics.training.{field}."
            )
    utilization = _validate_availability(
        training["gpu_utilization_fraction"],
        path="result.metrics.training.gpu_utilization_fraction",
        kind="number",
        minimum=0.0,
        maximum=1.0,
    )
    utilization_count = _validate_availability(
        training["gpu_utilization_sample_count"],
        path="result.metrics.training.gpu_utilization_sample_count",
        kind="integer",
        minimum=0,
    )
    utilization_interval = _validate_availability(
        training["gpu_utilization_sampling_interval_seconds"],
        path=("result.metrics.training.gpu_utilization_sampling_interval_seconds"),
        kind="number",
        minimum=0.0,
    )
    utilization_fields = (
        utilization,
        utilization_count,
        utilization_interval,
    )
    if status == "complete" and device.startswith("cuda"):
        if (
            any(item["status"] != "available" for item in utilization_fields)
            or int(utilization_count["value"]) < 2
            or float(utilization_interval["value"]) <= 0.0
        ):
            raise PolicyImprovementSchemaError(
                "Complete CUDA runs require at least two interval utilization "
                "samples and a positive sampling interval."
            )
    elif status == "complete":
        expected = {"status": "unavailable", "reason": "not_applicable"}
        if any(item != expected for item in utilization_fields):
            raise PolicyImprovementSchemaError(
                "Complete non-CUDA runs cannot claim GPU utilization."
            )
    _validate_availability(
        training["interactions_to_first_solve"],
        path="result.metrics.training.interactions_to_first_solve",
        kind="integer",
        minimum=0,
    )
    integer_fields = (
        "peak_allocated_memory_bytes",
        "peak_reserved_memory_bytes",
        "optimizer_steps",
        "cells_processed",
        "actions_processed",
        "tokens_processed",
        "policy_head_calls",
        "recurrent_map_applications",
        "value_head_calls",
    )
    checked_integers: dict[str, dict[str, Any]] = {}
    for field in integer_fields:
        availability = _validate_availability(
            training[field],
            path=f"result.metrics.training.{field}",
            kind="integer",
            minimum=0,
        )
        checked_integers[field] = availability
        if status == "complete" and availability["status"] != "available":
            raise PolicyImprovementSchemaError(
                f"Complete runs require result.metrics.training.{field}."
            )
    if status == "complete":
        positive = (
            "optimizer_steps",
            "cells_processed",
            "actions_processed",
            "tokens_processed",
            "policy_head_calls",
            "recurrent_map_applications",
            "value_head_calls",
        )
        if any(int(checked_integers[field]["value"]) <= 0 for field in positive):
            raise PolicyImprovementSchemaError(
                "Complete runs require positive core instrumentation counters."
            )
        allocated = int(checked_integers["peak_allocated_memory_bytes"]["value"])
        reserved = int(checked_integers["peak_reserved_memory_bytes"]["value"])
        if reserved < allocated:
            raise PolicyImprovementSchemaError(
                "Peak reserved memory cannot be smaller than peak allocated memory."
            )
    diagnostics = _exact_fields(
        metrics["diagnostics"],
        {
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
        },
        path="result.metrics.diagnostics",
    )
    for field in diagnostics:
        _validate_availability(
            diagnostics[field],
            path=f"result.metrics.diagnostics.{field}",
            kind="number",
            minimum=0.0,
            maximum=(
                1.0
                if field
                in {
                    "projection_active_rate",
                    "absolute_depth_policy_agreement",
                    "candidate_current_tv",
                    "mixture_realized_tv",
                }
                else None
            ),
        )
    return dict(metrics)


def validate_result(value: object) -> dict[str, Any]:
    """Validate one immutable run result without accepting placeholder data."""

    result = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "amendment_history_sha256",
            "run_id",
            "registry_row_sha256",
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
            "applied_config_override",
            "primary_policy_variant",
            "evaluation_snapshots",
            "status",
            "failure",
            "identities",
            "metrics",
            "artifacts",
        },
        path="result",
    )
    if (
        result["schema_name"] != SCHEMA_NAME
        or result["schema_version"] != RESULT_SCHEMA_VERSION
    ):
        raise PolicyImprovementSchemaError("Unsupported result schema.")
    _string(result["protocol_id"], path="result.protocol_id", identifier=True)
    for field in ("protocol_sha256", "amendment_history_sha256", "registry_row_sha256"):
        _sha256(result[field], path=f"result.{field}")
    _string(result["run_id"], path="result.run_id", identifier=True)
    phase = _string(result["phase"], path="result.phase", choices=set(PHASE_CONTRACTS))
    tier = _string(result["tier"], path="result.tier", choices=set(TIERS))
    _integer(result["seed"], path="result.seed")
    split = _string(
        result["evaluation_split"],
        path="result.evaluation_split",
        choices=set(EVALUATION_SPLITS),
    )
    expected_tier, expected_split, _ = PHASE_CONTRACTS[phase]
    if tier != expected_tier or split != expected_split:
        raise PolicyImprovementSchemaError(
            "Result phase, tier, and evaluation split are inconsistent."
        )
    method_id = _string(
        result["method_id"], path="result.method_id", choices=set(RESULT_METHOD_IDS)
    )
    base_method_id = _string(
        result["base_method_id"],
        path="result.base_method_id",
        choices=set(METHOD_IDS),
    )
    primary_variant = _string(
        result["primary_policy_variant"],
        path="result.primary_policy_variant",
        choices={"exact_mixture", "realized_policy"},
    )
    expected_primary = primary_policy_variant_for_method(method_id)
    if primary_variant != expected_primary:
        raise PolicyImprovementSchemaError(
            "Primary policy variant differs from the registered method."
        )
    for factor in ("n", "K"):
        _integer(result[factor], path=f"result.{factor}", minimum=1)
    _number(result["alpha"], path="result.alpha", minimum=0.0, maximum=1.0)
    applied_override = _mapping(
        result["applied_config_override"],
        path="result.applied_config_override",
    )
    if not applied_override:
        raise PolicyImprovementSchemaError(
            "Every result requires its exact applied config override."
        )
    canonical_json_bytes(dict(applied_override))
    if phase == "stage3_ablation":
        variant = _string(
            result["ablation_variant"],
            path="result.ablation_variant",
            identifier=True,
        )
        if base_method_id not in SCREEN_SELECTION_METHOD_ORDER:
            raise PolicyImprovementSchemaError(
                "Stage 3 results require the selected exact base method."
            )
        if method_id != stage3_method_id(variant, base_method_id):
            raise PolicyImprovementSchemaError(
                "Stage 3 result method differs from its registered intervention."
            )
    elif result["ablation_variant"] is not None:
        raise PolicyImprovementSchemaError(
            "Only Stage 3 results may identify an ablation variant."
        )
    elif base_method_id != method_id:
        raise PolicyImprovementSchemaError(
            "Only Stage 3 interventions may differ from their base method."
        )
    status = _string(result["status"], path="result.status", choices=set(RUN_STATUSES))
    if status == "complete":
        if result["failure"] is not None:
            raise PolicyImprovementSchemaError(
                "Complete runs cannot contain failure metadata."
            )
    else:
        failure = _exact_fields(
            result["failure"],
            {"phase", "error_class", "message_sha256"},
            path="result.failure",
        )
        _string(failure["phase"], path="result.failure.phase", identifier=True)
        _string(
            failure["error_class"], path="result.failure.error_class", identifier=True
        )
        _sha256(failure["message_sha256"], path="result.failure.message_sha256")
    identities = _validate_result_identities(result["identities"], status=status)
    test_open_identity = identities["test_open_sha256"]
    if (split == "test") != (test_open_identity["status"] == "available"):
        raise PolicyImprovementSchemaError(
            "Test results must bind TEST_OPEN.json; validation results must not."
        )
    if status == "complete":
        for field in (
            "checkpoint_sha256",
            "model_state_sha256",
            "evaluation_runtime_sha256",
            "evaluation_source_git_commit",
            "evaluation_runtime_profile_sha256",
            "evaluation_selected_source_manifest_sha256",
            "evaluation_pool_sha256",
        ):
            if identities[field]["status"] != "available":
                raise PolicyImprovementSchemaError(
                    f"Complete runs require result.identities.{field}."
                )
    snapshots = _validate_evaluation_snapshots(
        result["evaluation_snapshots"],
        status=status,
        tier=tier,
        method_id=method_id,
        run_id=str(result["run_id"]),
        evaluation_pool_sha256=identities["evaluation_pool_sha256"],
    )
    metrics = _validate_metrics(
        result["metrics"],
        status=status,
        device=str(identities["device"]),
    )
    first_solve = metrics["training"]["interactions_to_first_solve"]
    if status == "complete" and first_solve["status"] == "available":
        observed_interactions = snapshots[0]["observed_environment_interactions"]
        if int(first_solve["value"]) > int(observed_interactions["value"]):
            raise PolicyImprovementSchemaError(
                "Interactions to first solve cannot exceed the interaction snapshot."
            )
    artifacts = _exact_fields(
        result["artifacts"],
        {
            "checkpoint",
            "checkpoint_validation",
            "model_state_inventory",
            "run_manifest",
        },
        path="result.artifacts",
    )
    for field in artifacts:
        availability = _validate_available_sha256(
            artifacts[field], path=f"result.artifacts.{field}"
        )
        if status == "complete" and availability["status"] != "available":
            raise PolicyImprovementSchemaError(
                f"Complete runs require result.artifacts.{field}."
            )
    canonical_json_bytes(result)
    return dict(result)
