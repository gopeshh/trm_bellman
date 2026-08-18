#!/usr/bin/env fbpython
"""Strict schemas for the fresh policy-improvement v2 evidence cycle."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.policy_improvement_populations import (
    POPULATIONS_SCHEMA_NAME,
    POPULATIONS_SCHEMA_VERSION,
)


PROTOCOL_ID = "policy-improvement-v2-20260818"
PROTOCOL_SCHEMA_NAME = "policy_improvement_protocol_v2"
PROTOCOL_SCHEMA_VERSION = 2
REGISTRY_SCHEMA_NAME = "policy_improvement_registry_v2"
REGISTRY_SCHEMA_VERSION = 1
REGISTRY_ROW_SCHEMA_NAME = "policy_improvement_registry_row_v2"
REGISTRY_ROW_SCHEMA_VERSION = 1
RESULT_SCHEMA_NAME = "policy_improvement_result_v2"
RESULT_SCHEMA_VERSION = 1
METHOD_IDS = (
    "fixed_base_exact_persistent",
    "fixed_base_exact_episodic",
    "legacy_parameter_interpolation",
    "matched_ppo",
)
EXACT_METHOD_IDS = METHOD_IDS[:2]
PHASES = (
    "stage0_smoke",
    "stage1_screen",
    "stage1_alpha",
    "stage1_baseline_readiness",
    "stage2_confirmatory",
    "stage3_ablation",
)
IMMUTABLE_DATASET_V1: dict[str, object] = {
    "builder_schema_version": 2,
    "domain": "sudoku_4x4",
    "manifest_schema_version": 2,
    "manifest_sha256": {
        "status": "available",
        "value": "2572bb79faeec976dc83cb75b8520e59691a7c9dc3f8fe252554fc29bfe90ccd",
    },
    "name": "policy-improvement-hard-4x4-v1",
    "producer_source": {
        "git_commit": {
            "status": "available",
            "value": "7317d7011c31ac622c0723d22cf9f4bb0571bdf1",
        },
        "launcher_sha256": {
            "status": "available",
            "value": "8e73d60512934705f8a295fb67845f5a90b8c6f3006e1ea6ebcb373881c76d6a",
        },
        "runtime_sha256": {
            "status": "available",
            "value": "8ded72fa7dc774caad11e62ad10e62604f3393e483109a7c1c2d51d2bbd5ebe8",
        },
        "source_manifest_sha256": {
            "status": "available",
            "value": "5833f7cb1884651015d66901a31ed78cbf620826ee5035bed4192e5e5590f6b1",
        },
    },
    "root": "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1",
    "splits": {
        "test": {
            "count": 512,
            "generation_seed": 26081403,
            "manifest_sha256": {
                "status": "available",
                "value": "bf14acfc94e580bb3678102729f88432fda599610b2ca2578949f8c4cbc775bd",
            },
            "ordered_record_sha256": {
                "status": "available",
                "value": "94647c77419ef6ec150aa1dde6d07de1a3f9112c2fb768168404a520138ef6ac",
            },
        },
        "train": {
            "count": 1024,
            "generation_seed": 26081401,
            "manifest_sha256": {
                "status": "available",
                "value": "05146037857b1adb42520e80a0c2ab250053a517196c8b8ac95e002aa40c6f74",
            },
            "ordered_record_sha256": {
                "status": "available",
                "value": "73110263bb388e0f6e0976156d03f499b83541a58d07c39b8b634e94a98ad446",
            },
        },
        "validation": {
            "count": 256,
            "generation_seed": 26081402,
            "manifest_sha256": {
                "status": "available",
                "value": "a4b1bffb92f9c7c1ebe7baf9dcaaed83191f9c6249bec0887ed8ff7fb6b4a937",
            },
            "ordered_record_sha256": {
                "status": "available",
                "value": "9257ae46c71fa24afd8c0284af6087c5662faffd724b395037c9016ac4110380",
            },
        },
    },
    "symmetry_canonicalization": {
        "group_order": 3072,
        "reject_cross_split_overlap": True,
        "scheme": "sudoku4x4_spatial_digit_lexicographic_v1",
    },
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PolicyImprovementV2SchemaError(RuntimeError):
    """Raised when a v2 protocol, registry, or result is not canonical."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PolicyImprovementV2SchemaError("Document is not canonical JSON.") from exc
    if json.loads(payload.decode("ascii")) != value:
        raise PolicyImprovementV2SchemaError(
            "Document does not round-trip through canonical JSON."
        )
    return payload


def load_strict_json(path: str | Path) -> Any:
    def pairs(items: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PolicyImprovementV2SchemaError(f"Duplicate JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="ascii"),
            object_pairs_hook=pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementV2SchemaError(
            "Document is not strict ASCII JSON."
        ) from exc


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyImprovementV2SchemaError(f"{path} must be an object.")
    return dict(value)


def _exact_fields(value: object, expected: set[str], *, path: str) -> dict[str, Any]:
    result = _mapping(value, path=path)
    if set(result) != expected:
        raise PolicyImprovementV2SchemaError(
            f"{path} field inventory differs: {sorted(set(result) ^ expected)!r}."
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
        raise PolicyImprovementV2SchemaError(f"{path} must be nonempty ASCII.")
    if choices is not None and value not in choices:
        raise PolicyImprovementV2SchemaError(f"{path} is not registered.")
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise PolicyImprovementV2SchemaError(f"{path} is not an identifier.")
    return value


def _sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PolicyImprovementV2SchemaError(f"{path} is not a SHA-256 digest.")
    return value


def _integer(value: object, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyImprovementV2SchemaError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise PolicyImprovementV2SchemaError(f"{path} is below its minimum.")
    return value


def _number(
    value: object,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyImprovementV2SchemaError(f"{path} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise PolicyImprovementV2SchemaError(f"{path} must be finite.")
    if minimum is not None and number < minimum:
        raise PolicyImprovementV2SchemaError(f"{path} is below its minimum.")
    if maximum is not None and number > maximum:
        raise PolicyImprovementV2SchemaError(f"{path} is above its maximum.")
    return number


def _sequence(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise PolicyImprovementV2SchemaError(f"{path} must be an array.")
    return list(value)


def effective_config_sha256(
    base: Mapping[str, object], override: Mapping[str, object]
) -> str:
    result = dict(base)
    unknown = set(override) - set(result)
    if unknown:
        raise PolicyImprovementV2SchemaError(
            f"Config override contains unknown keys: {sorted(unknown)!r}."
        )
    result.update(override)
    return sha256_json(result)


def _validate_available(value: object, *, path: str, kind: str) -> dict[str, Any]:
    item = _mapping(value, path=path)
    if item.get("status") == "available":
        if set(item) != {"status", "value"}:
            raise PolicyImprovementV2SchemaError(f"{path} availability fields differ.")
        if kind == "sha256":
            _sha256(item["value"], path=f"{path}.value")
        elif kind == "commit":
            value_text = item["value"]
            if (
                not isinstance(value_text, str)
                or re.fullmatch(r"[0-9a-f]{40}", value_text) is None
            ):
                raise PolicyImprovementV2SchemaError(f"{path}.value is not a commit.")
        return item
    if set(item) != {"status", "reason"} or item.get("status") != "unavailable":
        raise PolicyImprovementV2SchemaError(f"{path} availability fields differ.")
    _string(item["reason"], path=f"{path}.reason", identifier=True)
    return item


def _validate_base_policy_artifact(value: object) -> dict[str, Any]:
    artifact = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "status",
            "reason",
            "required_identity_fields",
            "stage1_execution_allowed",
        },
        path="protocol.base_policy_artifact",
    )
    if (
        artifact["schema_name"] != "policy_improvement_base_policy_artifact_v2"
        or artifact["schema_version"] != 1
        or artifact["status"] != "unavailable"
        or artifact["reason"]
        != "no_authenticated_train_only_base_policy_artifact_supplied_for_v2"
        or artifact["stage1_execution_allowed"] is not False
    ):
        raise PolicyImprovementV2SchemaError("V2 base-policy blocker is not frozen.")
    required = _sequence(
        artifact["required_identity_fields"],
        path="protocol.base_policy_artifact.required_identity_fields",
    )
    if required != [
        "initialization_kind",
        "architecture_sha256",
        "model_state_sha256",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "training_data_sha256",
        "training_procedure_sha256",
        "checkpoint_sha256",
        "shared_across_persistent_and_episodic",
        "not_selected_by_validation_or_test",
    ]:
        raise PolicyImprovementV2SchemaError(
            "Base-policy required identity fields differ."
        )
    return artifact


def validate_v2_protocol(value: object) -> dict[str, Any]:
    protocol = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "created_date",
            "status",
            "amendments",
            "document_schemas",
            "architecture",
            "budgets",
            "compute_accounting",
            "dataset",
            "population_registry",
            "evaluation_populations",
            "full_execution_gate",
            "grid",
            "methods",
            "selection_rules",
            "output_root",
            "seeds",
            "statistics",
            "test_isolation",
            "base_policy_artifact",
            "base_policy_interpretations",
        },
        path="protocol",
    )
    if (
        protocol["schema_name"] != PROTOCOL_SCHEMA_NAME
        or protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["created_date"] != "2026-08-18"
        or protocol["status"] != "registered_stage0_only_base_policy_unavailable"
        or protocol["amendments"] != []
    ):
        raise PolicyImprovementV2SchemaError("Protocol identity differs.")

    schemas = _mapping(protocol["document_schemas"], path="protocol.document_schemas")
    expected_schemas = {
        "protocol": [PROTOCOL_SCHEMA_NAME, PROTOCOL_SCHEMA_VERSION],
        "populations": [POPULATIONS_SCHEMA_NAME, POPULATIONS_SCHEMA_VERSION],
        "registry": [REGISTRY_SCHEMA_NAME, REGISTRY_SCHEMA_VERSION],
        "registry_row": [REGISTRY_ROW_SCHEMA_NAME, REGISTRY_ROW_SCHEMA_VERSION],
        "result": [RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION],
        "theory_amendment": ["policy_improvement_theory_bridge_amendment_v2", 1],
        "runtime_authorization": ["policy_improvement_runtime_authorization_v2", 1],
        "launcher_request": ["policy_improvement_launcher_request_v2", 1],
        "audit": ["policy_improvement_audit_v2", 1],
        "analysis": ["policy_improvement_registered_analysis_v2", 1],
    }
    if schemas != expected_schemas:
        raise PolicyImprovementV2SchemaError("V2 document schema registry differs.")

    if protocol["architecture"] != {
        "backbone": "trm",
        "h_cycles": 2,
        "hidden_size": 64,
        "l_cycles": 2,
        "l_layers": 1,
        "puzzle_emb_ndim": 0,
    }:
        raise PolicyImprovementV2SchemaError("Architecture registration differs.")
    if protocol["budgets"] != {
        "smoke": {
            "environment_interactions": 32,
            "checkpoint_environment_interactions": [16, 32],
            "evaluation_records": 8,
        },
        "pilot": {
            "maximum_environment_interactions": 80000,
            "checkpoint_environment_interactions": [10000, 20000, 40000, 80000],
            "selection_records": 128,
            "bridge_records": 128,
        },
        "confirmatory": {
            "environment_interactions": 80000,
            "checkpoint_environment_interactions": [10000, 20000, 40000, 80000],
            "evaluation_records": 512,
        },
        "throughput_calibration": {
            "environment_interaction_caps": [256, 1024, 4096],
            "automatic_caps": [256, 1024],
            "user_authorization_required_for": 4096,
            "evaluation_rollouts": False,
            "split": "train",
        },
        "compute_matching": {
            "primary": "environment_interactions",
            "recurrent_map_matched_role": (
                "sensitivity_only_until_broader_measured_compute_budget_frozen"
            ),
            "maximum_relative_mismatch": 0.05,
            "measured_calibration_required": True,
        },
    }:
        raise PolicyImprovementV2SchemaError("Budget registration differs.")
    if protocol["compute_accounting"] != {
        "schema_name": "policy_improvement_compute_accounting_v2",
        "schema_version": 1,
        "fields": [
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
        ],
        "exact_action_enumeration_cost": "record_as_action_value_evaluations",
        "primary_analysis_budget": "environment_interactions",
        "recurrent_map_matched_label": "sensitivity_analysis",
    }:
        raise PolicyImprovementV2SchemaError("Compute-accounting registration differs.")

    population_registry = _exact_fields(
        protocol["population_registry"],
        {"path", "sha256", "schema_name", "schema_version"},
        path="protocol.population_registry",
    )
    if (
        population_registry["path"] != "configs/policy_improvement_v2/populations.json"
        or population_registry["schema_name"] != POPULATIONS_SCHEMA_NAME
        or population_registry["schema_version"] != POPULATIONS_SCHEMA_VERSION
    ):
        raise PolicyImprovementV2SchemaError("Population registry binding differs.")
    _sha256(population_registry["sha256"], path="protocol.population_registry.sha256")

    dataset = _exact_fields(
        protocol["dataset"],
        {
            "builder_schema_version",
            "manifest_schema_version",
            "domain",
            "name",
            "root",
            "manifest_sha256",
            "producer_source",
            "splits",
            "symmetry_canonicalization",
        },
        path="protocol.dataset",
    )
    if (
        dataset["builder_schema_version"] != 2
        or dataset["manifest_schema_version"] != 2
        or dataset["domain"] != "sudoku_4x4"
        or dataset["name"] != "policy-improvement-hard-4x4-v1"
        or dataset["root"]
        != "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1"
    ):
        raise PolicyImprovementV2SchemaError("Dataset registration differs.")
    _validate_available(
        dataset["manifest_sha256"],
        path="protocol.dataset.manifest_sha256",
        kind="sha256",
    )
    producer = _mapping(
        dataset["producer_source"], path="protocol.dataset.producer_source"
    )
    if set(producer) != {
        "git_commit",
        "launcher_sha256",
        "runtime_sha256",
        "source_manifest_sha256",
    }:
        raise PolicyImprovementV2SchemaError("Dataset producer inventory differs.")
    _validate_available(
        producer["git_commit"],
        path="protocol.dataset.producer_source.git_commit",
        kind="commit",
    )
    for field in ("launcher_sha256", "runtime_sha256", "source_manifest_sha256"):
        _validate_available(
            producer[field],
            path=f"protocol.dataset.producer_source.{field}",
            kind="sha256",
        )
    splits = _mapping(dataset["splits"], path="protocol.dataset.splits")
    if set(splits) != {"train", "validation", "test"}:
        raise PolicyImprovementV2SchemaError("Dataset split inventory differs.")
    for split, count, seed in (
        ("train", 1024, 26081401),
        ("validation", 256, 26081402),
        ("test", 512, 26081403),
    ):
        registration = _exact_fields(
            splits[split],
            {"count", "generation_seed", "manifest_sha256", "ordered_record_sha256"},
            path=f"protocol.dataset.splits.{split}",
        )
        if registration["count"] != count or registration["generation_seed"] != seed:
            raise PolicyImprovementV2SchemaError(f"Dataset split {split!r} differs.")
        _validate_available(
            registration["manifest_sha256"],
            path=f"protocol.dataset.splits.{split}.manifest_sha256",
            kind="sha256",
        )
        _validate_available(
            registration["ordered_record_sha256"],
            path=f"protocol.dataset.splits.{split}.ordered_record_sha256",
            kind="sha256",
        )
    if dataset != IMMUTABLE_DATASET_V1:
        raise PolicyImprovementV2SchemaError(
            "V2 must reuse the exact immutable v1 dataset registration."
        )

    populations = _mapping(
        protocol["evaluation_populations"],
        path="protocol.evaluation_populations",
    )
    if populations != {
        "stage0_smoke": {
            "population_id": "stage0_smoke",
            "split": "train",
            "count": 8,
            "scientific_selection": False,
            "paper_evidence_eligible": False,
        },
        "validation_select": {
            "population_id": "validation_select",
            "split": "validation",
            "count": 128,
            "selection_use": "stage1_configuration_and_alpha_only",
        },
        "validation_bridge": {
            "population_id": "validation_bridge",
            "split": "validation",
            "count": 128,
            "selection_use": "none",
        },
        "confirmatory_test": {
            "population_id": "confirmatory_test",
            "split": "test",
            "count": 512,
            "requires_authenticated_test_open": True,
        },
    }:
        raise PolicyImprovementV2SchemaError("Evaluation populations differ.")

    grid = _mapping(protocol["grid"], path="protocol.grid")
    if set(grid) != {
        "stage0",
        "stage1_screen",
        "stage1_alpha",
        "stage1_baseline_readiness",
        "stage2",
        "stage3",
    }:
        raise PolicyImprovementV2SchemaError("Protocol grid inventory differs.")
    stage0 = _mapping(grid["stage0"], path="protocol.grid.stage0")
    if stage0 != {
        "method_ids": list(METHOD_IDS),
        "expected_rows": 4,
        "n": 2,
        "K": 1,
        "alpha": 0.1,
        "prepare_environment_interactions": 16,
        "resume_environment_interactions": 32,
        "upi_optimizer_batch_size": 1,
        "upi_rollout_episodes_per_step": 1,
        "ppo_rollout_environment_interactions": 16,
        "evaluation_population": "stage0_smoke",
        "systems_only": True,
        "scientific_selection": False,
        "paper_evidence_eligible": False,
    }:
        raise PolicyImprovementV2SchemaError("Stage 0 grid differs.")
    screen = _mapping(grid["stage1_screen"], path="protocol.grid.stage1_screen")
    if screen != {
        "method_ids": list(EXACT_METHOD_IDS),
        "n_values": [2, 4],
        "K_values": [1, 5],
        "alpha": 0.1,
        "expected_rows": 24,
        "evaluation_population": "validation_select",
        "checkpoint_environment_interactions": [10000, 20000, 40000, 80000],
        "all_rows_required_through": 10000,
        "continuation_rule": "within_latent_mode_multifidelity_v2",
    }:
        raise PolicyImprovementV2SchemaError("Stage 1 screen grid differs.")
    if _mapping(grid["stage1_alpha"], path="protocol.grid.stage1_alpha") != {
        "alpha_values": [0.05, 0.1, 0.2],
        "expected_rows": 9,
        "evaluation_population": "validation_select",
        "selection_rule": "stage1_exact_configuration_selection_v2",
    }:
        raise PolicyImprovementV2SchemaError("Stage 1 alpha grid differs.")
    if _mapping(
        grid["stage1_baseline_readiness"],
        path="protocol.grid.stage1_baseline_readiness",
    ) != {
        "method_ids": ["legacy_parameter_interpolation", "matched_ppo"],
        "expected_rows": 6,
        "evaluation_population": "validation_select",
        "selection_role": "none",
        "materialization_rule": "after_exact_configuration_selection_v2",
    }:
        raise PolicyImprovementV2SchemaError("Baseline-readiness grid differs.")
    if _mapping(grid["stage2"], path="protocol.grid.stage2") != {
        "method_ids": list(METHOD_IDS),
        "expected_rows": 32,
        "evaluation_population": "confirmatory_test",
        "selection_rule": "best_validation_exact_and_alpha_v2",
    }:
        raise PolicyImprovementV2SchemaError("Stage 2 row count differs.")
    if _mapping(grid["stage3"], path="protocol.grid.stage3") != {
        "expected_rows": 64,
        "selection_rule": "frozen_best_exact_ablation_v2",
        "training_variants": [
            "batch_only_centering",
            "distilled_realization",
            "projection_identity",
            "contraction_enabled",
            "target_retention_0p9",
            "target_retention_0p999",
            "depth_lower_neighbor",
            "depth_upper_neighbor",
        ],
        "reused_contrasts": [
            "persistent_vs_episodic",
            "exact_mixture_vs_parameter_interpolation",
        ],
    }:
        raise PolicyImprovementV2SchemaError("Stage 3 row count differs.")

    methods = _sequence(protocol["methods"], path="protocol.methods")
    if len(methods) != 4 or [item.get("id") for item in methods] != list(METHOD_IDS):
        raise PolicyImprovementV2SchemaError("Method registration differs.")
    for index, method_value in enumerate(methods):
        method = _exact_fields(
            method_value,
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
        expected_path = f"configs/policy_improvement_v2/{method['id']}.yaml"
        if method["config_path"] != expected_path:
            raise PolicyImprovementV2SchemaError("Method config path differs.")
        _sha256(
            method["config_sha256"], path=f"protocol.methods[{index}].config_sha256"
        )
        _sha256(
            method["canonical_config_sha256"],
            path=f"protocol.methods[{index}].canonical_config_sha256",
        )
    method_semantics = {
        "fixed_base_exact_persistent": {
            "training_protocol": "fixed_base_exact",
            "latent_mode": "persistent",
            "centering": "exact_statewise",
            "deployment": "exact_probability_mixture",
            "primary_policy": "exact_mixture",
        },
        "fixed_base_exact_episodic": {
            "training_protocol": "fixed_base_exact",
            "latent_mode": "episodic",
            "centering": "exact_statewise",
            "deployment": "exact_probability_mixture",
            "primary_policy": "exact_mixture",
        },
        "legacy_parameter_interpolation": {
            "training_protocol": "legacy",
            "latent_mode": "persistent",
            "centering": "batch_only",
            "deployment": "parameter_interpolation",
            "primary_policy": "realized_policy",
        },
        "matched_ppo": {
            "training_protocol": "ppo",
            "latent_mode": "episodic",
            "centering": "ppo_gae",
            "deployment": "stochastic_deployed_policy",
            "primary_policy": "realized_policy",
        },
    }
    for method in methods:
        if {
            field: method[field]
            for field in (
                "training_protocol",
                "latent_mode",
                "centering",
                "deployment",
                "primary_policy",
            )
        } != method_semantics[method["id"]]:
            raise PolicyImprovementV2SchemaError(
                f"Method semantics for {method['id']!r} differ."
            )

    seeds = _mapping(protocol["seeds"], path="protocol.seeds")
    if seeds != {
        "namespace": "upi-trm-policy-improvement-v2",
        "derivation": "explicit_preoutcome_values_v2",
        "smoke": [1257297357],
        "pilot": [784831257, 2087907586, 4056782312],
        "confirmatory": [
            2081976412,
            781025396,
            1148619853,
            913535362,
            2143253519,
            2279379379,
            402312153,
            2736405725,
        ],
    }:
        raise PolicyImprovementV2SchemaError("Registered seed values differ.")
    _validate_base_policy_artifact(protocol["base_policy_artifact"])
    if protocol["base_policy_interpretations"] != {
        "preferred_practical_study": (
            "one_authenticated_competent_train_only_artifact_shared_across_"
            "persistent_and_episodic"
        ),
        "random_base_stress_test": (
            "one_proposal_stress_regime_from_random_policy_not_practical_safe_"
            "policy_improvement"
        ),
        "choice_status": "unresolved_user_decision_required",
    }:
        raise PolicyImprovementV2SchemaError("Base-policy interpretations differ.")

    if protocol["output_root"] != {
        "environment_variable": "UPI_TRM_EVIDENCE_ROOT",
        "relative_path": "policy_improvement_v2",
    }:
        raise PolicyImprovementV2SchemaError("Output namespace differs.")
    if protocol["full_execution_gate"] != {
        "environment_variable": "RUN_UPITRM_FULL_EXPERIMENTS",
        "required_value": "1",
        "stage0_exempt": True,
        "stage1_to_stage3_blocked_by_base_policy": True,
    }:
        raise PolicyImprovementV2SchemaError("Full-execution gate differs.")
    if protocol["selection_rules"] != {
        "screen_within_latent_mode": {
            "id": "within_latent_mode_multifidelity_v2",
            "checkpoint": 10000,
            "endpoint": (
                "mean_seed_level_primary_interaction_matched_solve_rate_on_"
                "validation_select"
            ),
            "maximize": True,
            "groups": ["persistent", "episodic"],
            "tie_break": ["n_ascending", "K_ascending"],
            "continuation_checkpoints": [20000, 40000, 80000],
        },
        "exact_configuration": {
            "id": "stage1_exact_configuration_selection_v2",
            "checkpoint": 80000,
            "endpoint": (
                "mean_seed_level_primary_interaction_matched_solve_rate_on_"
                "validation_select"
            ),
            "maximize": True,
            "tie_break": [
                "persistent_before_episodic",
                "n_ascending",
                "K_ascending",
            ],
        },
        "alpha": {
            "id": "best_validation_alpha_v2",
            "endpoint": (
                "mean_seed_level_primary_interaction_matched_solve_rate_on_"
                "validation_select"
            ),
            "maximize": True,
            "tie_break": ["alpha_ascending"],
        },
        "baseline_readiness": {
            "selection_role": "none",
            "scientific_selection": False,
        },
    }:
        raise PolicyImprovementV2SchemaError("Selection rules differ.")
    if protocol["statistics"] != {
        "confidence_level": 0.95,
        "bootstrap": {
            "replicates": 10000,
            "scheme": "paired_seed_cluster_then_paired_puzzle_v2",
            "seed": 3472274560,
        },
        "seed_level_permutation_test": True,
        "multiplicity": {
            "primary": "unadjusted",
            "prespecified_secondary": "holm",
            "other": "exploratory",
        },
        "primary_endpoint": "final_test_solve_rate_percentage_points",
        "primary_contrast": "fixed_base_exact_persistent-minus-matched_ppo",
    }:
        raise PolicyImprovementV2SchemaError("Statistics registration differs.")
    if protocol["test_isolation"] != {
        "stage0_split": "train",
        "throughput_split": "train",
        "pilot_split": "validation_select",
        "bridge_split": "validation_bridge",
        "confirmatory_split": "test",
        "test_open_count": 1,
        "test_open_required_before": ["stage2_confirmatory", "stage3_ablation"],
        "test_open_status": "not_created",
        "owner_writable_deletion_residual_debt": True,
    }:
        raise PolicyImprovementV2SchemaError("Test-isolation registration differs.")
    canonical_json_bytes(protocol)
    return protocol


def validate_v2_registry_row(value: object) -> dict[str, Any]:
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
            "evaluation_population",
            "evaluation_split",
            "n",
            "K",
            "alpha",
            "ablation_variant",
            "selection_rule",
            "continuation_rule",
            "checkpoint_environment_interactions",
            "factor_applicability",
            "scientific_selection",
            "paper_evidence_eligible",
            "config_override",
            "base_config_canonical_sha256",
            "expected_effective_config_sha256",
        },
        path="registry_row",
    )
    if (
        row["schema_name"] != REGISTRY_ROW_SCHEMA_NAME
        or row["schema_version"] != REGISTRY_ROW_SCHEMA_VERSION
        or row["protocol_id"] != PROTOCOL_ID
    ):
        raise PolicyImprovementV2SchemaError("Registry-row schema differs.")
    kind = _string(
        row["row_kind"],
        path="registry_row.row_kind",
        choices={"concrete", "selection_template"},
    )
    phase = _string(row["phase"], path="registry_row.phase", choices=set(PHASES))
    _string(row["tier"], path="registry_row.tier", identifier=True)
    _string(row["run_id"], path="registry_row.run_id", identifier=True)
    method = row["method_id"]
    if method is not None:
        _string(method, path="registry_row.method_id", choices=set(METHOD_IDS))
    base_method = row["base_method_id"]
    if base_method is not None:
        _string(
            base_method,
            path="registry_row.base_method_id",
            choices=set(METHOD_IDS),
        )
    _integer(row["seed"], path="registry_row.seed", minimum=0)
    split = _string(
        row["evaluation_split"],
        path="registry_row.evaluation_split",
        choices={"train", "validation", "test"},
    )
    population = _string(
        row["evaluation_population"],
        path="registry_row.evaluation_population",
        choices={
            "stage0_smoke",
            "validation_select",
            "confirmatory_test",
        },
    )
    expected_population = {
        "train": "stage0_smoke",
        "validation": "validation_select",
        "test": "confirmatory_test",
    }[split]
    if population != expected_population:
        raise PolicyImprovementV2SchemaError("Registry-row split binding differs.")
    for field in ("n", "K"):
        if row[field] is not None:
            _integer(row[field], path=f"registry_row.{field}", minimum=1)
    if row["alpha"] is not None:
        _number(
            row["alpha"],
            path="registry_row.alpha",
            minimum=0.0,
            maximum=1.0,
        )
    for field in ("ablation_variant", "selection_rule", "continuation_rule"):
        if row[field] is not None:
            _string(row[field], path=f"registry_row.{field}", identifier=True)
    checkpoints = _sequence(
        row["checkpoint_environment_interactions"],
        path="registry_row.checkpoint_environment_interactions",
    )
    if (
        not checkpoints
        or any(
            _integer(value, path="registry_row.checkpoint", minimum=1) != value
            for value in checkpoints
        )
        or checkpoints != sorted(set(checkpoints))
    ):
        raise PolicyImprovementV2SchemaError("Checkpoint schedule differs.")
    factors = _exact_fields(
        row["factor_applicability"],
        {"n", "K", "alpha", "ablation"},
        path="registry_row.factor_applicability",
    )
    if any(not isinstance(value, bool) for value in factors.values()):
        raise PolicyImprovementV2SchemaError("Factor applicability is not boolean.")
    if not isinstance(row["scientific_selection"], bool) or not isinstance(
        row["paper_evidence_eligible"], bool
    ):
        raise PolicyImprovementV2SchemaError("Registry-row evidence flags differ.")

    if kind == "concrete":
        if (
            method is None
            or row["n"] is None
            or row["K"] is None
            or row["alpha"] is None
        ):
            raise PolicyImprovementV2SchemaError("Concrete row factors are incomplete.")
        override = _mapping(row["config_override"], path="registry_row.config_override")
        _sha256(
            row["base_config_canonical_sha256"],
            path="registry_row.base_config_canonical_sha256",
        )
        _sha256(
            row["expected_effective_config_sha256"],
            path="registry_row.expected_effective_config_sha256",
        )
        if not override:
            raise PolicyImprovementV2SchemaError("Concrete config override is empty.")
    else:
        if any(
            row[field] is not None
            for field in (
                "config_override",
                "base_config_canonical_sha256",
                "expected_effective_config_sha256",
            )
        ):
            raise PolicyImprovementV2SchemaError(
                "Selection template carries concrete config identity."
            )
        if row["selection_rule"] is None:
            raise PolicyImprovementV2SchemaError(
                "Selection template lacks its materialization rule."
            )

    phase_contract = {
        "stage0_smoke": ("train", [16, 32], False, False),
        "stage1_screen": ("validation", [10000, 20000, 40000, 80000], True, False),
        "stage1_alpha": ("validation", [80000], True, False),
        "stage1_baseline_readiness": ("validation", [80000], False, False),
        "stage2_confirmatory": ("test", [80000], False, True),
        "stage3_ablation": ("test", [80000], False, True),
    }
    expected_split, expected_checkpoints, selection, paper = phase_contract[phase]
    if (
        split != expected_split
        or checkpoints != expected_checkpoints
        or row["scientific_selection"] is not selection
        or row["paper_evidence_eligible"] is not paper
    ):
        raise PolicyImprovementV2SchemaError("Registry-row phase contract differs.")
    if phase in {"stage0_smoke", "stage1_screen"} and kind != "concrete":
        raise PolicyImprovementV2SchemaError("Executable base row is not concrete.")
    canonical_json_bytes(row)
    return row


def validate_v2_result(value: object) -> dict[str, Any]:
    """Validate the strict v2 result envelope used by runtime-specific payloads."""

    result = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "protocol_schema_name",
            "protocol_schema_version",
            "protocol_sha256",
            "population_registry_schema_name",
            "population_registry_schema_version",
            "population_registry_sha256",
            "registry_schema_name",
            "registry_schema_version",
            "registry_sha256",
            "registry_row_schema_name",
            "registry_row_schema_version",
            "registry_row_sha256",
            "run_id",
            "phase",
            "method_id",
            "status",
            "evaluation_split",
            "evaluation_population_id",
            "evaluation_population_binding_sha256",
            "evaluation_population_ordered_record_sha256",
            "evaluation_population_ordered_input_sha256",
            "evaluation_record_count",
            "validation_data_opened",
            "test_data_opened",
            "scientific_selection",
            "paper_evidence_eligible",
            "payload",
        },
        path="result",
    )
    if (
        result["schema_name"] != RESULT_SCHEMA_NAME
        or result["schema_version"] != RESULT_SCHEMA_VERSION
        or result["protocol_id"] != PROTOCOL_ID
        or result["protocol_schema_name"] != PROTOCOL_SCHEMA_NAME
        or result["protocol_schema_version"] != PROTOCOL_SCHEMA_VERSION
        or result["population_registry_schema_name"] != POPULATIONS_SCHEMA_NAME
        or result["population_registry_schema_version"] != POPULATIONS_SCHEMA_VERSION
        or result["registry_schema_name"] != REGISTRY_SCHEMA_NAME
        or result["registry_schema_version"] != REGISTRY_SCHEMA_VERSION
        or result["registry_row_schema_name"] != REGISTRY_ROW_SCHEMA_NAME
        or result["registry_row_schema_version"] != REGISTRY_ROW_SCHEMA_VERSION
    ):
        raise PolicyImprovementV2SchemaError("Result schema differs.")
    for field in (
        "protocol_sha256",
        "population_registry_sha256",
        "registry_sha256",
        "registry_row_sha256",
        "evaluation_population_binding_sha256",
        "evaluation_population_ordered_record_sha256",
        "evaluation_population_ordered_input_sha256",
    ):
        _sha256(result[field], path=f"result.{field}")
    _string(result["run_id"], path="result.run_id", identifier=True)
    _string(result["phase"], path="result.phase", choices=set(PHASES))
    _string(result["method_id"], path="result.method_id", choices=set(METHOD_IDS))
    _string(
        result["status"],
        path="result.status",
        choices={"complete", "failed"},
    )
    _string(
        result["evaluation_population_id"],
        path="result.evaluation_population_id",
        choices={"stage0_smoke", "validation_select", "confirmatory_test"},
    )
    split = _string(
        result["evaluation_split"],
        path="result.evaluation_split",
        choices={"train", "validation", "test"},
    )
    if (
        result["evaluation_population_id"]
        != {
            "train": "stage0_smoke",
            "validation": "validation_select",
            "test": "confirmatory_test",
        }[split]
    ):
        raise PolicyImprovementV2SchemaError("Result population split differs.")
    _integer(
        result["evaluation_record_count"],
        path="result.evaluation_record_count",
        minimum=1,
    )
    for field in (
        "validation_data_opened",
        "test_data_opened",
        "scientific_selection",
        "paper_evidence_eligible",
    ):
        if not isinstance(result[field], bool):
            raise PolicyImprovementV2SchemaError(f"result.{field} must be boolean.")
    _mapping(result["payload"], path="result.payload")
    canonical_json_bytes(result)
    return result


def validate_checkpoint_schedule(values: object) -> list[int]:
    """Public helper for amendment and runtime schedule checks."""

    result = _sequence(values, path="checkpoint_schedule")
    checked = [
        _integer(value, path=f"checkpoint_schedule[{index}]", minimum=1)
        for index, value in enumerate(result)
    ]
    if checked != sorted(set(checked)):
        raise PolicyImprovementV2SchemaError(
            "Checkpoint schedule must be strictly increasing."
        )
    return checked


def sha256_fields(values: Sequence[str]) -> str:
    """Return an unambiguous digest for a fixed ordered identity tuple."""

    digest = hashlib.sha256()
    for index, value in enumerate(values):
        digest.update(_sha256(value, path=f"sha256_fields[{index}]").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
