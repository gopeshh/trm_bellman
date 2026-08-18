#!/usr/bin/env fbpython
"""Strict protocol-v2 schemas for the read-only theory bridge.

The v2 bridge is deliberately incompatible with the v1 request and result
schemas.  It binds an explicit, ordered record population and separates
algebraic centering roundoff from parity with the estimator used by training.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

from scripts.policy_improvement_schema import canonical_json_bytes


PROTOCOL_ID = "policy-improvement-v2-20260818"
THEORY_AMENDMENT_SCHEMA_NAME = "policy_improvement_theory_bridge_amendment_v2"
THEORY_REQUEST_SCHEMA_NAME = "policy_improvement_theory_bridge_request_v2"
THEORY_RESULT_SCHEMA_NAME = "policy_improvement_theory_bridge_result_v2"
THEORY_SCHEMA_VERSION = 1

EXACT_METHOD_IDS = (
    "fixed_base_exact_persistent",
    "fixed_base_exact_episodic",
)
REALIZED_METHOD_IDS = (
    "legacy_parameter_interpolation",
    "fixed_base_distilled_realization",
)
THEORY_METHOD_IDS = EXACT_METHOD_IDS + REALIZED_METHOD_IDS
THEORY_METRIC_IDS = (
    "D_nm",
    "bellman_residual_proxy",
    "B_nm",
    "policy_overlap_tau",
    "constructed_centering_roundoff",
    "training_estimator_centering_defect",
    "training_estimator_parity_max_abs_error",
    "E_n",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")


class TheoryBridgeV2SchemaError(ValueError):
    """Raised when a protocol-v2 theory document is not canonical."""


def theory_document_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TheoryBridgeV2SchemaError(f"{path} must be an object.")
    return value


def _fields(value: object, expected: set[str], *, path: str) -> Mapping[str, object]:
    item = _object(value, path=path)
    actual = set(item)
    if actual != expected:
        raise TheoryBridgeV2SchemaError(
            f"{path} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )
    return item


def _text(
    value: object,
    *,
    path: str,
    choices: set[str] | None = None,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise TheoryBridgeV2SchemaError(f"{path} must be nonempty ASCII text.")
    if choices is not None and value not in choices:
        raise TheoryBridgeV2SchemaError(f"{path} must be one of {sorted(choices)}.")
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise TheoryBridgeV2SchemaError(f"{path} is not a canonical identifier.")
    return value


def _sha256(value: object, *, path: str) -> str:
    text = _text(value, path=path)
    if _SHA256.fullmatch(text) is None:
        raise TheoryBridgeV2SchemaError(f"{path} must be a lowercase SHA-256 digest.")
    return text


def _git_commit(value: object, *, path: str) -> str:
    text = _text(value, path=path)
    if _GIT_COMMIT.fullmatch(text) is None:
        raise TheoryBridgeV2SchemaError(f"{path} must be a full lowercase Git commit.")
    return text


def _integer(value: object, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TheoryBridgeV2SchemaError(
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
        raise TheoryBridgeV2SchemaError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise TheoryBridgeV2SchemaError(f"{path} must be finite.")
    if minimum is not None and result < minimum:
        raise TheoryBridgeV2SchemaError(f"{path} is below its minimum.")
    if maximum is not None and result > maximum:
        raise TheoryBridgeV2SchemaError(f"{path} is above its maximum.")
    return result


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TheoryBridgeV2SchemaError(f"{path} must be boolean.")
    return value


def _summary(value: object, *, path: str) -> Mapping[str, object]:
    item = _fields(value, {"count", "mean", "minimum", "maximum"}, path=path)
    count = _integer(item["count"], path=f"{path}.count", minimum=1)
    minimum = _number(item["minimum"], path=f"{path}.minimum", minimum=0.0)
    maximum = _number(item["maximum"], path=f"{path}.maximum", minimum=0.0)
    mean = _number(item["mean"], path=f"{path}.mean", minimum=0.0)
    if minimum > maximum or not minimum <= mean <= maximum:
        raise TheoryBridgeV2SchemaError(f"{path} is not a valid nonnegative summary.")
    return {"count": count, "mean": mean, "minimum": minimum, "maximum": maximum}


def validate_identity_bundle(value: object) -> dict[str, Any]:
    identity = _fields(
        value,
        {
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
            "theory_amendment_sha256",
            "registry_row_sha256",
            "checkpoint",
            "model",
            "config",
            "producer_source",
            "training_runtime",
            "dataset_records",
            "evaluator_source",
            "evaluator_runtime",
        },
        path="identity",
    )
    if identity["protocol_id"] != PROTOCOL_ID:
        raise TheoryBridgeV2SchemaError("Theory identity names another protocol.")
    if (
        identity["protocol_schema_name"] != "policy_improvement_protocol_v2"
        or identity["protocol_schema_version"] != 2
        or identity["registry_schema_name"] != "policy_improvement_registry_v2"
        or identity["registry_schema_version"] != 1
        or identity["population_registry_schema_name"]
        != "policy_improvement_populations_v2"
        or identity["population_registry_schema_version"] != 1
        or identity["registry_row_schema_name"] != "policy_improvement_registry_row_v2"
        or identity["registry_row_schema_version"] != 1
    ):
        raise TheoryBridgeV2SchemaError("Theory identity schema bindings differ.")
    for field in (
        "protocol_sha256",
        "population_registry_sha256",
        "registry_sha256",
        "theory_amendment_sha256",
        "registry_row_sha256",
    ):
        _sha256(identity[field], path=f"identity.{field}")

    checkpoint = _fields(
        identity["checkpoint"],
        {"sha256", "size_bytes", "snapshot_kind", "environment_interactions"},
        path="identity.checkpoint",
    )
    _sha256(checkpoint["sha256"], path="identity.checkpoint.sha256")
    _integer(checkpoint["size_bytes"], path="identity.checkpoint.size_bytes", minimum=1)
    _text(
        checkpoint["snapshot_kind"],
        path="identity.checkpoint.snapshot_kind",
        choices={"smoke_prepare", "smoke_resume", "scheduled", "interaction_matched"},
    )
    _integer(
        checkpoint["environment_interactions"],
        path="identity.checkpoint.environment_interactions",
        minimum=1,
    )

    model = _fields(
        identity["model"],
        {
            "model_sha256",
            "model_config_sha256",
            "current_policy_sha256",
            "candidate_policy_sha256",
            "deployed_policy_sha256",
            "recurrent_transition_sha256",
        },
        path="identity.model",
    )
    for field in model:
        _sha256(model[field], path=f"identity.model.{field}")

    config = _fields(
        identity["config"],
        {"file_sha256", "base_canonical_sha256", "effective_config_sha256"},
        path="identity.config",
    )
    for field in config:
        _sha256(config[field], path=f"identity.config.{field}")

    producer = _fields(
        identity["producer_source"],
        {"git_commit", "source_manifest_sha256"},
        path="identity.producer_source",
    )
    _git_commit(producer["git_commit"], path="identity.producer_source.git_commit")
    _sha256(
        producer["source_manifest_sha256"],
        path="identity.producer_source.source_manifest_sha256",
    )

    training = _fields(
        identity["training_runtime"],
        {
            "role",
            "source_git_commit",
            "source_manifest_sha256",
            "runtime_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
            "runtime_authorization_sha256",
            "launcher_sha256",
        },
        path="identity.training_runtime",
    )
    _text(
        training["role"],
        path="identity.training_runtime.role",
        choices={"policy-improvement-full", "policy-improvement-smoke"},
    )
    _git_commit(
        training["source_git_commit"],
        path="identity.training_runtime.source_git_commit",
    )
    for field in training:
        if field not in {"role", "source_git_commit"}:
            _sha256(training[field], path=f"identity.training_runtime.{field}")
    if not (
        training["source_manifest_sha256"]
        == training["runtime_profile_sha256"]
        == training["selected_source_manifest_sha256"]
    ):
        raise TheoryBridgeV2SchemaError(
            "Training runtime source-profile identities differ."
        )
    if training["source_git_commit"] != producer["git_commit"]:
        raise TheoryBridgeV2SchemaError(
            "Training runtime and producer name different commits."
        )

    dataset = _fields(
        identity["dataset_records"],
        {
            "split",
            "population_id",
            "population_binding_sha256",
            "split_manifest_sha256",
            "ordered_record_sha256",
            "ordered_input_sha256",
            "selected_record_indices",
            "selected_record_indices_sha256",
            "selected_records",
            "selected_records_sha256",
            "selected_input_sha256s",
            "selected_input_sha256s_sha256",
            "record_count",
        },
        path="identity.dataset_records",
    )
    _text(
        dataset["split"],
        path="identity.dataset_records.split",
        choices={"train", "validation"},
    )
    _text(
        dataset["population_id"],
        path="identity.dataset_records.population_id",
        identifier=True,
    )
    for field in (
        "split_manifest_sha256",
        "population_binding_sha256",
        "ordered_record_sha256",
        "ordered_input_sha256",
        "selected_record_indices_sha256",
        "selected_records_sha256",
        "selected_input_sha256s_sha256",
    ):
        _sha256(dataset[field], path=f"identity.dataset_records.{field}")
    record_count = _integer(
        dataset["record_count"], path="identity.dataset_records.record_count", minimum=1
    )
    indices_value = dataset["selected_record_indices"]
    if not isinstance(indices_value, list) or len(indices_value) != record_count:
        raise TheoryBridgeV2SchemaError("Selected record indices are incomplete.")
    indices = [
        _integer(
            item, path=f"identity.dataset_records.selected_record_indices[{index}]"
        )
        for index, item in enumerate(indices_value)
    ]
    if len(set(indices)) != len(indices):
        raise TheoryBridgeV2SchemaError("Selected record indices must be unique.")
    records_value = dataset["selected_records"]
    if not isinstance(records_value, list) or len(records_value) != record_count:
        raise TheoryBridgeV2SchemaError("Selected record identities are incomplete.")
    records: list[dict[str, object]] = []
    for offset, raw in enumerate(records_value):
        record = _fields(
            raw,
            {"record_index", "dataset_record_sha256"},
            path=f"identity.dataset_records.selected_records[{offset}]",
        )
        record_index = _integer(
            record["record_index"],
            path=f"identity.dataset_records.selected_records[{offset}].record_index",
        )
        record_sha256 = _sha256(
            record["dataset_record_sha256"],
            path=(
                f"identity.dataset_records.selected_records[{offset}]"
                ".dataset_record_sha256"
            ),
        )
        if record_index != indices[offset]:
            raise TheoryBridgeV2SchemaError("Selected records and index order differ.")
        records.append(
            {"record_index": record_index, "dataset_record_sha256": record_sha256}
        )
    if (
        theory_document_sha256(indices) != dataset["selected_record_indices_sha256"]
        or theory_document_sha256(records) != dataset["selected_records_sha256"]
    ):
        raise TheoryBridgeV2SchemaError(
            "Selected record digests do not match their payloads."
        )
    input_sha256s = dataset["selected_input_sha256s"]
    if not isinstance(input_sha256s, list) or len(input_sha256s) != record_count:
        raise TheoryBridgeV2SchemaError("Selected input identities are incomplete.")
    for index, digest in enumerate(input_sha256s):
        _sha256(
            digest, path=f"identity.dataset_records.selected_input_sha256s[{index}]"
        )
    if (
        theory_document_sha256(input_sha256s)
        != dataset["selected_input_sha256s_sha256"]
    ):
        raise TheoryBridgeV2SchemaError(
            "Selected input digest does not match its payload."
        )

    evaluator_source = _fields(
        identity["evaluator_source"],
        {"git_commit", "source_manifest_sha256"},
        path="identity.evaluator_source",
    )
    _git_commit(
        evaluator_source["git_commit"], path="identity.evaluator_source.git_commit"
    )
    _sha256(
        evaluator_source["source_manifest_sha256"],
        path="identity.evaluator_source.source_manifest_sha256",
    )
    evaluator_runtime = _fields(
        identity["evaluator_runtime"],
        {
            "runtime_sha256",
            "runtime_profile_sha256",
            "runtime_authorization_sha256",
            "launcher_sha256",
        },
        path="identity.evaluator_runtime",
    )
    for field in evaluator_runtime:
        _sha256(evaluator_runtime[field], path=f"identity.evaluator_runtime.{field}")
    return dict(identity)


def _validate_estimator(
    value: object, *, path: str, horizon: int
) -> Mapping[str, object]:
    expected_fields = (
        {"horizon", "kind", "rollout_count", "uncertainty"}
        if horizon == 1
        else {
            "horizon",
            "kind",
            "rollout_count",
            "base_seed",
            "seed_derivation",
            "uncertainty",
        }
    )
    estimator = _fields(value, expected_fields, path=path)
    rollout_count = _integer(estimator["rollout_count"], path=f"{path}.rollout_count")
    if horizon == 1:
        if dict(estimator) != {
            "horizon": 1,
            "kind": "exact_masked_action_sum_v2",
            "rollout_count": 0,
            "uncertainty": "exact_zero_monte_carlo_standard_error",
        }:
            raise TheoryBridgeV2SchemaError(
                "K=1 requires exact masked action summation."
            )
    elif (
        estimator["horizon"] != 5
        or estimator["kind"] != "common_random_number_monte_carlo_v2"
        or rollout_count < 2
        or isinstance(estimator["base_seed"], bool)
        or not isinstance(estimator["base_seed"], int)
        or estimator["base_seed"] < 0
        or estimator["seed_derivation"]
        != "sha256_protocol_checkpoint_record_state_repeat_v2"
        or estimator["uncertainty"] != "sample_standard_error_of_operator_mean"
    ):
        raise TheoryBridgeV2SchemaError("K=5 requires registered CRN Monte Carlo.")
    return estimator


def _validate_return_estimator(value: object, *, path: str) -> Mapping[str, object]:
    estimator = _fields(
        value,
        {
            "rollout_count",
            "base_seed",
            "seed_derivation",
            "maximum_environment_steps",
            "terminal_requirement",
            "common_random_numbers",
            "independent_from_bellman_estimator",
            "policy",
            "reported_uncertainty",
        },
        path=path,
    )
    if (
        _integer(estimator["rollout_count"], path=f"{path}.rollout_count", minimum=2)
        < 2
        or isinstance(estimator["base_seed"], bool)
        or not isinstance(estimator["base_seed"], int)
        or estimator["base_seed"] < 0
        or estimator["seed_derivation"]
        != "sha256_protocol_checkpoint_record_state_repeat_v2"
        or _integer(
            estimator["maximum_environment_steps"],
            path=f"{path}.maximum_environment_steps",
            minimum=1,
        )
        < 1
        or estimator["terminal_requirement"] != "must_terminate_without_bootstrap"
        or estimator["common_random_numbers"] is not True
        or estimator["independent_from_bellman_estimator"] is not True
        or estimator["policy"] != "current_policy"
        or estimator["reported_uncertainty"] != "sample_standard_error"
    ):
        raise TheoryBridgeV2SchemaError("Current-policy return estimator is invalid.")
    return estimator


def validate_theory_request(value: object) -> dict[str, Any]:
    request = _fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "evaluation_id",
            "evaluation_population",
            "run_id",
            "method_id",
            "latent_mode",
            "checkpoint_environment_interactions",
            "n",
            "reference_depth_m",
            "reference_role",
            "bellman_horizon",
            "gamma",
            "alpha",
            "bellman_estimator",
            "return_estimator",
            "advantage_clipping",
            "constructed_centering_tolerance",
            "centering_parity_tolerance",
            "deployment_identity_tolerance",
            "scientific_selection",
            "paper_evidence_eligible",
            "identity",
            "test_data_opened",
        },
        path="request",
    )
    if (
        request["schema_name"] != THEORY_REQUEST_SCHEMA_NAME
        or request["schema_version"] != THEORY_SCHEMA_VERSION
        or request["protocol_id"] != PROTOCOL_ID
    ):
        raise TheoryBridgeV2SchemaError(
            "Theory request schema or protocol is unsupported."
        )
    _text(request["evaluation_id"], path="request.evaluation_id", identifier=True)
    population = _text(
        request["evaluation_population"],
        path="request.evaluation_population",
        choices={"stage0_smoke", "validation_bridge"},
    )
    _text(request["run_id"], path="request.run_id", identifier=True)
    method = _text(
        request["method_id"], path="request.method_id", choices=set(THEORY_METHOD_IDS)
    )
    mode = _text(
        request["latent_mode"],
        path="request.latent_mode",
        choices={"persistent", "episodic"},
    )
    expected_mode = (
        "episodic" if method == "fixed_base_exact_episodic" else "persistent"
    )
    if mode != expected_mode:
        raise TheoryBridgeV2SchemaError("Method and latent mode disagree.")
    checkpoint = _integer(
        request["checkpoint_environment_interactions"],
        path="request.checkpoint_environment_interactions",
        minimum=1,
    )
    n = _integer(request["n"], path="request.n", minimum=1)
    reference = _integer(
        request["reference_depth_m"], path="request.reference_depth_m", minimum=2
    )
    role = _text(
        request["reference_role"],
        path="request.reference_role",
        choices={"primary", "exploratory"},
    )
    if (
        reference <= n
        or reference not in {8, 16}
        or (reference == 8) != (role == "primary")
    ):
        raise TheoryBridgeV2SchemaError("Reference depth and role are invalid.")
    horizon = _integer(
        request["bellman_horizon"], path="request.bellman_horizon", minimum=1
    )
    if horizon not in {1, 5}:
        raise TheoryBridgeV2SchemaError("Bellman horizon must be 1 or 5.")
    gamma = _number(request["gamma"], path="request.gamma", minimum=0.0, maximum=1.0)
    if gamma >= 1.0:
        raise TheoryBridgeV2SchemaError("Gamma must be strictly below one.")
    _number(request["alpha"], path="request.alpha", minimum=0.0, maximum=1.0)
    _validate_estimator(
        request["bellman_estimator"], path="request.bellman_estimator", horizon=horizon
    )
    _validate_return_estimator(
        request["return_estimator"], path="request.return_estimator"
    )
    clipping = _fields(
        request["advantage_clipping"],
        {"kind", "clip_value"},
        path="request.advantage_clipping",
    )
    if clipping["kind"] == "none":
        if clipping["clip_value"] is not None:
            raise TheoryBridgeV2SchemaError(
                "Unclipped advantages cannot name a clip value."
            )
    elif clipping["kind"] == "clip_then_exact_recenter":
        _number(
            clipping["clip_value"],
            path="request.advantage_clipping.clip_value",
            minimum=0.0,
        )
        if float(clipping["clip_value"]) <= 0.0:
            raise TheoryBridgeV2SchemaError("Advantage clip value must be positive.")
    else:
        raise TheoryBridgeV2SchemaError("Advantage clipping contract is unsupported.")
    _number(
        request["constructed_centering_tolerance"],
        path="request.constructed_centering_tolerance",
        minimum=0.0,
    )
    _number(
        request["centering_parity_tolerance"],
        path="request.centering_parity_tolerance",
        minimum=0.0,
    )
    _number(
        request["deployment_identity_tolerance"],
        path="request.deployment_identity_tolerance",
        minimum=0.0,
    )
    if _boolean(request["scientific_selection"], path="request.scientific_selection"):
        raise TheoryBridgeV2SchemaError(
            "Theory bridge cannot perform scientific selection."
        )
    paper_eligible = _boolean(
        request["paper_evidence_eligible"], path="request.paper_evidence_eligible"
    )
    if population == "stage0_smoke" and paper_eligible:
        raise TheoryBridgeV2SchemaError("Stage 0 bridge smoke is not paper evidence.")
    if population == "validation_bridge" and not paper_eligible:
        raise TheoryBridgeV2SchemaError(
            "Registered validation-bridge evaluations are paper-evidence eligible."
        )
    identity = validate_identity_bundle(request["identity"])
    dataset = identity["dataset_records"]
    expected_split = "train" if population == "stage0_smoke" else "validation"
    if dataset["population_id"] != population or dataset["split"] != expected_split:
        raise TheoryBridgeV2SchemaError(
            "Theory population and dataset identity disagree."
        )
    checkpoint_identity = identity["checkpoint"]
    if checkpoint_identity["environment_interactions"] != checkpoint:
        raise TheoryBridgeV2SchemaError(
            "Request checkpoint progress differs from identity."
        )
    if request["test_data_opened"] is not False:
        raise TheoryBridgeV2SchemaError("Theory bridge cannot open test data.")
    return dict(request)


def validate_theory_amendment(value: object) -> dict[str, Any]:
    amendment = _fields(
        value,
        {
            "schema_name",
            "schema_version",
            "amendment_id",
            "created_at_utc",
            "protocol_id",
            "protocol_schema_name",
            "protocol_schema_version",
            "protocol_sha256",
            "population_registry_sha256",
            "source_registry_schema_name",
            "source_registry_schema_version",
            "prior_amendment_history_sha256",
            "source_registry_sha256",
            "test_data_opened",
            "outcome_evidence_inspected",
            "result_schema",
            "evaluator_contract",
            "metrics",
            "reference_depths",
            "bellman_estimators",
            "predictive_return_estimator",
            "centering_contract",
            "deployment_contract",
            "checkpoint_schedule",
            "multi_fidelity_exact_method_screen",
            "analysis",
            "smoke_policy",
        },
        path="amendment",
    )
    if (
        amendment["schema_name"] != THEORY_AMENDMENT_SCHEMA_NAME
        or amendment["schema_version"] != THEORY_SCHEMA_VERSION
        or amendment["protocol_id"] != PROTOCOL_ID
    ):
        raise TheoryBridgeV2SchemaError(
            "Theory amendment schema or protocol is unsupported."
        )
    _text(amendment["amendment_id"], path="amendment.amendment_id", identifier=True)
    _text(amendment["created_at_utc"], path="amendment.created_at_utc")
    if (
        amendment["protocol_schema_name"] != "policy_improvement_protocol_v2"
        or amendment["protocol_schema_version"] != 2
        or amendment["source_registry_schema_name"] != "policy_improvement_registry_v2"
        or amendment["source_registry_schema_version"] != 1
    ):
        raise TheoryBridgeV2SchemaError("Theory amendment schema bindings differ.")
    for field in (
        "protocol_sha256",
        "population_registry_sha256",
        "prior_amendment_history_sha256",
        "source_registry_sha256",
    ):
        _sha256(amendment[field], path=f"amendment.{field}")
    if (
        amendment["test_data_opened"] is not False
        or amendment["outcome_evidence_inspected"] is not False
    ):
        raise TheoryBridgeV2SchemaError(
            "Theory amendment must predate output and preserve test isolation."
        )
    result_schema = _fields(
        amendment["result_schema"],
        {"schema_name", "schema_version"},
        path="amendment.result_schema",
    )
    if dict(result_schema) != {
        "schema_name": THEORY_RESULT_SCHEMA_NAME,
        "schema_version": THEORY_SCHEMA_VERSION,
    }:
        raise TheoryBridgeV2SchemaError("Amendment registers the wrong result schema.")
    contract = _fields(
        amendment["evaluator_contract"],
        {
            "checkpoint_access",
            "launcher_purpose",
            "persistent_reference_semantics",
            "population_id",
            "runtime_role",
            "selection_use",
            "source_tree_fallback",
            "training_state_mutation",
        },
        path="amendment.evaluator_contract",
    )
    if dict(contract) != {
        "checkpoint_access": "authenticated_write_sealed_read_only_descriptor",
        "launcher_purpose": "policy-improvement-theory-bridge",
        "persistent_reference_semantics": (
            "m_changes_endpoint_evaluator_only_deployed_F_n_policy_transition_"
            "and_carried_successor_latent_fixed"
        ),
        "population_id": "validation_bridge",
        "runtime_role": "policy-improvement-theory-bridge",
        "selection_use": "none",
        "source_tree_fallback": False,
        "training_state_mutation": "forbidden",
    }:
        raise TheoryBridgeV2SchemaError("Evaluator contract differs from v2.")
    metrics = _object(amendment["metrics"], path="amendment.metrics")
    expected_metric_names = {
        "D_nm",
        "bellman_residual_proxy",
        "B_nm",
        "policy_overlap_tau",
        "constructed_centering_roundoff",
        "training_estimator_centering_defect",
        "training_estimator_parity_max_abs_error",
        "exact_mixture_deployment_identity_tv",
        "deployment_discrepancy_delta_dep",
        "E_n",
    }
    if set(metrics) != expected_metric_names or any(
        not isinstance(description, str) or not description
        for description in metrics.values()
    ):
        raise TheoryBridgeV2SchemaError(
            "Amendment metric registration differs from v2."
        )
    depths = _fields(
        amendment["reference_depths"],
        {
            "primary_m",
            "optional_exploratory_m",
            "m16_checkpoint_environment_interactions",
            "selection_use",
        },
        path="amendment.reference_depths",
    )
    if (
        depths["primary_m"] != 8
        or depths["optional_exploratory_m"] != [16]
        or depths["m16_checkpoint_environment_interactions"] != [80000]
        or depths["selection_use"] != "none"
    ):
        raise TheoryBridgeV2SchemaError("Reference-depth registration differs.")
    estimators = _fields(
        amendment["bellman_estimators"],
        {"K1", "K5"},
        path="amendment.bellman_estimators",
    )
    _validate_estimator(
        estimators["K1"], path="amendment.bellman_estimators.K1", horizon=1
    )
    _validate_estimator(
        estimators["K5"], path="amendment.bellman_estimators.K5", horizon=5
    )
    _validate_return_estimator(
        amendment["predictive_return_estimator"],
        path="amendment.predictive_return_estimator",
    )
    centering = _fields(
        amendment["centering_contract"],
        {
            "constructed_centering_roundoff_absolute_tolerance",
            "miscentered_estimator_action",
            "trainer_reconstruction",
            "training_estimator_parity_absolute_tolerance",
        },
        path="amendment.centering_contract",
    )
    if centering["miscentered_estimator_action"] != "reject" or centering[
        "trainer_reconstruction"
    ] != (
        "exact_frozen_checkpoint_advantage_tensor_with_registered_action_mask_"
        "clipping_and_recentering"
    ):
        raise TheoryBridgeV2SchemaError("Centering contract differs from v2.")
    for field in (
        "constructed_centering_roundoff_absolute_tolerance",
        "training_estimator_parity_absolute_tolerance",
    ):
        tolerance = _number(
            centering[field], path=f"amendment.centering_contract.{field}", minimum=0.0
        )
        if tolerance <= 0.0:
            raise TheoryBridgeV2SchemaError("Centering tolerance must be positive.")
    deployment = _fields(
        amendment["deployment_contract"],
        {
            "exact_method_role",
            "exact_mixture_identity_tv_absolute_tolerance",
            "nonexact_policy_role",
        },
        path="amendment.deployment_contract",
    )
    if (
        deployment["exact_method_role"] != "identity_check_expected_zero"
        or deployment["nonexact_policy_role"]
        != "deployment_discrepancy_nonselection_diagnostic"
    ):
        raise TheoryBridgeV2SchemaError("Deployment diagnostic registration differs.")
    deployment_tolerance = _number(
        deployment["exact_mixture_identity_tv_absolute_tolerance"],
        path=(
            "amendment.deployment_contract.exact_mixture_identity_tv_absolute_tolerance"
        ),
        minimum=0.0,
    )
    if deployment_tolerance <= 0.0:
        raise TheoryBridgeV2SchemaError(
            "Deployment identity tolerance must be positive."
        )
    schedule = _fields(
        amendment["checkpoint_schedule"],
        {
            "environment_interactions",
            "optional_exploratory_reference_depths_at_final",
            "primary_reference_depths",
        },
        path="amendment.checkpoint_schedule",
    )
    if dict(schedule) != {
        "environment_interactions": [10000, 20000, 40000, 80000],
        "optional_exploratory_reference_depths_at_final": [16],
        "primary_reference_depths": [8, 8, 8, 8],
    }:
        raise TheoryBridgeV2SchemaError("Theory checkpoint schedule differs.")
    screen = _fields(
        amendment["multi_fidelity_exact_method_screen"],
        {
            "all_rows_required_through_environment_interactions",
            "continuation_checkpoints",
            "final_exact_selection_checkpoint",
            "final_tie_break",
            "methods",
            "population_id",
            "theory_bridge_selection_use",
            "within_latent_mode_selection_checkpoint",
            "within_latent_mode_tie_break",
        },
        path="amendment.multi_fidelity_exact_method_screen",
    )
    if dict(screen) != {
        "all_rows_required_through_environment_interactions": 10000,
        "continuation_checkpoints": [20000, 40000, 80000],
        "final_exact_selection_checkpoint": 80000,
        "final_tie_break": [
            "persistent_before_episodic",
            "n_ascending",
            "K_ascending",
        ],
        "methods": list(EXACT_METHOD_IDS),
        "population_id": "validation_select",
        "theory_bridge_selection_use": "none",
        "within_latent_mode_selection_checkpoint": 10000,
        "within_latent_mode_tie_break": ["n_ascending", "K_ascending"],
    }:
        raise TheoryBridgeV2SchemaError("Exact-method screen registration differs.")
    analysis = _fields(
        amendment["analysis"],
        {
            "B_nm_calibration_bins",
            "alpha_values",
            "claims",
            "deployment_discrepancy_policies",
            "finite_population_only",
            "flexible_model_fitting",
            "population_id",
            "post_outcome_threshold_tuning",
            "registered_targets",
            "scientific_selection",
        },
        path="amendment.analysis",
    )
    if dict(analysis) != {
        "B_nm_calibration_bins": {
            "count": 4,
            "kind": "fixed_rank_quantile_bins_v1",
            "ties": "stable_record_index_order",
        },
        "alpha_values": [0.05, 0.1, 0.2],
        "claims": "no_uniform_certificate_and_no_claim_from_smoke_values",
        "deployment_discrepancy_policies": [
            "legacy_parameter_interpolation",
            "registered_distilled_realization_when_available",
        ],
        "finite_population_only": True,
        "flexible_model_fitting": False,
        "population_id": "validation_bridge",
        "post_outcome_threshold_tuning": False,
        "registered_targets": [
            "spearman_B_nm_vs_E_n",
            "spearman_D_nm_vs_E_n",
            "spearman_residual_penalty_vs_E_n",
            "fixed_B_nm_calibration_bins_with_paired_mean_E_n",
            "cluster_uncertainty_by_training_seed",
            (
                "alpha_penalty_proxy_vs_negative_paired_current_to_mixture_"
                "return_probability"
            ),
        ],
        "scientific_selection": False,
    }:
        raise TheoryBridgeV2SchemaError(
            "Predictive bridge analysis registration differs."
        )
    smoke = _fields(
        amendment["smoke_policy"],
        {
            "allowed_population_ids",
            "allowed_splits",
            "paper_evidence_eligible",
            "scientific_selection",
            "test_records",
            "validation_records",
        },
        path="amendment.smoke_policy",
    )
    if dict(smoke) != {
        "allowed_population_ids": ["stage0_smoke"],
        "allowed_splits": ["train"],
        "paper_evidence_eligible": False,
        "scientific_selection": False,
        "test_records": False,
        "validation_records": False,
    }:
        raise TheoryBridgeV2SchemaError("Theory smoke policy differs.")
    return dict(amendment)


def validate_request_against_amendment(
    request_value: object,
    amendment_value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = validate_theory_request(request_value)
    amendment = validate_theory_amendment(amendment_value)
    identity = request["identity"]
    if (
        identity["protocol_schema_name"] != amendment["protocol_schema_name"]
        or identity["protocol_schema_version"] != amendment["protocol_schema_version"]
        or identity["protocol_id"] != amendment["protocol_id"]
        or identity["protocol_sha256"] != amendment["protocol_sha256"]
        or identity["population_registry_sha256"]
        != amendment["population_registry_sha256"]
        or identity["registry_schema_name"] != amendment["source_registry_schema_name"]
        or identity["registry_schema_version"]
        != amendment["source_registry_schema_version"]
        or identity["registry_sha256"] != amendment["source_registry_sha256"]
        or identity["theory_amendment_sha256"] != theory_document_sha256(amendment)
    ):
        raise TheoryBridgeV2SchemaError("Request and amendment identities differ.")
    horizon_key = f"K{request['bellman_horizon']}"
    if request["bellman_estimator"] != amendment["bellman_estimators"][horizon_key]:
        raise TheoryBridgeV2SchemaError(
            "Request changes the registered Bellman estimator."
        )
    if request["return_estimator"] != amendment["predictive_return_estimator"]:
        raise TheoryBridgeV2SchemaError(
            "Request changes the registered return estimator."
        )
    if (
        float(request["constructed_centering_tolerance"])
        != float(
            amendment["centering_contract"][
                "constructed_centering_roundoff_absolute_tolerance"
            ]
        )
        or float(request["centering_parity_tolerance"])
        != float(
            amendment["centering_contract"][
                "training_estimator_parity_absolute_tolerance"
            ]
        )
        or float(request["deployment_identity_tolerance"])
        != float(
            amendment["deployment_contract"][
                "exact_mixture_identity_tv_absolute_tolerance"
            ]
        )
    ):
        raise TheoryBridgeV2SchemaError("Request changes a registered tolerance.")
    dataset = identity["dataset_records"]
    expected_population = (
        ("train", 8)
        if request["evaluation_population"] == "stage0_smoke"
        else ("validation", 128)
    )
    if (
        dataset["population_id"] != request["evaluation_population"]
        or (dataset["split"], dataset["record_count"]) != expected_population
    ):
        raise TheoryBridgeV2SchemaError(
            "Request changes the registered population role."
        )
    checkpoint = int(request["checkpoint_environment_interactions"])
    snapshot_kind = identity["checkpoint"]["snapshot_kind"]
    if request["evaluation_population"] == "stage0_smoke":
        if checkpoint != 32 or snapshot_kind != "smoke_resume":
            raise TheoryBridgeV2SchemaError(
                "Stage 0 bridge smoke requires the authenticated 32-interaction resume."
            )
    elif checkpoint not in amendment["checkpoint_schedule"]["environment_interactions"]:
        raise TheoryBridgeV2SchemaError(
            "Validation bridge checkpoint is outside the registered schedule."
        )
    if (
        request["reference_depth_m"] == 16
        and checkpoint
        not in amendment["reference_depths"]["m16_checkpoint_environment_interactions"]
    ):
        raise TheoryBridgeV2SchemaError(
            "m=16 is restricted to the final registered checkpoint."
        )
    return request, amendment


def validate_theory_result(value: object, *, request: object) -> dict[str, Any]:
    checked_request = validate_theory_request(request)
    result = _fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "status",
            "scope",
            "uniform_certificate",
            "evaluation_id",
            "evaluation_population",
            "run_id",
            "method_id",
            "checkpoint_environment_interactions",
            "n",
            "reference_depth_m",
            "reference_role",
            "bellman_horizon",
            "gamma",
            "alpha",
            "request_sha256",
            "identity",
            "state_count",
            "states",
            "metrics",
            "deployment_diagnostic",
            "monte_carlo_uncertainty",
            "persistent_semantics",
            "read_only_verification",
            "scientific_selection",
            "paper_evidence_eligible",
            "test_data_opened",
        },
        path="result",
    )
    if (
        result["schema_name"] != THEORY_RESULT_SCHEMA_NAME
        or result["schema_version"] != THEORY_SCHEMA_VERSION
        or result["protocol_id"] != PROTOCOL_ID
        or result["status"] != "complete"
        or result["scope"] != "finite_population_diagnostic_only"
        or result["uniform_certificate"] is not False
        or result["scientific_selection"] is not False
        or result["test_data_opened"] is not False
    ):
        raise TheoryBridgeV2SchemaError("Theory result header is invalid.")
    for field in (
        "evaluation_id",
        "evaluation_population",
        "run_id",
        "method_id",
        "checkpoint_environment_interactions",
        "n",
        "reference_depth_m",
        "reference_role",
        "bellman_horizon",
        "gamma",
        "alpha",
        "paper_evidence_eligible",
    ):
        if result[field] != checked_request[field]:
            raise TheoryBridgeV2SchemaError("Theory result differs from its request.")
    if result["request_sha256"] != theory_document_sha256(checked_request):
        raise TheoryBridgeV2SchemaError("Theory result names another request.")
    identity = validate_identity_bundle(result["identity"])
    if canonical_json_bytes(identity) != canonical_json_bytes(
        checked_request["identity"]
    ):
        raise TheoryBridgeV2SchemaError("Theory result mixes identities.")
    state_count = _integer(result["state_count"], path="result.state_count", minimum=1)
    rows = result["states"]
    if not isinstance(rows, list) or len(rows) != state_count:
        raise TheoryBridgeV2SchemaError("Theory result state inventory is incomplete.")
    expected_indices = identity["dataset_records"]["selected_record_indices"]
    expected_records = identity["dataset_records"]["selected_records"]
    seen_ids: set[str] = set()
    checked_rows: list[Mapping[str, object]] = []
    for offset, raw in enumerate(rows):
        row = _fields(
            raw,
            {
                "state_id",
                "record_index",
                "dataset_record_sha256",
                "registered_state_sha256",
                "value_n",
                "value_m",
                "D_nm",
                "bellman_operator_mean_m",
                "bellman_operator_standard_error_m",
                "bellman_residual_proxy",
                "B_nm",
                "B_nm_standard_error",
                "policy_overlap_tau",
                "constructed_centering_roundoff",
                "constructed_centering_tolerance",
                "training_estimator_centering_defect",
                "training_estimator_parity_max_abs_error",
                "training_estimator_parity_tolerance",
                "V_hat_pi",
                "V_hat_pi_standard_error",
                "E_n",
                "exact_mixture_deployment_identity_tv",
                "deployment_discrepancy_delta_dep",
            },
            path=f"result.states[{offset}]",
        )
        state_id = _text(row["state_id"], path=f"result.states[{offset}].state_id")
        if state_id in seen_ids:
            raise TheoryBridgeV2SchemaError("Theory result state IDs must be unique.")
        seen_ids.add(state_id)
        if (
            row["record_index"] != expected_indices[offset]
            or row["dataset_record_sha256"]
            != expected_records[offset]["dataset_record_sha256"]
        ):
            raise TheoryBridgeV2SchemaError(
                "Theory result reordered its registered population."
            )
        _sha256(
            row["registered_state_sha256"],
            path=f"result.states[{offset}].registered_state_sha256",
        )
        for field in ("value_n", "value_m", "bellman_operator_mean_m", "V_hat_pi"):
            _number(row[field], path=f"result.states[{offset}].{field}")
        for field in (
            "D_nm",
            "bellman_operator_standard_error_m",
            "bellman_residual_proxy",
            "B_nm",
            "B_nm_standard_error",
            "policy_overlap_tau",
            "constructed_centering_roundoff",
            "constructed_centering_tolerance",
            "training_estimator_centering_defect",
            "training_estimator_parity_max_abs_error",
            "training_estimator_parity_tolerance",
            "V_hat_pi_standard_error",
            "E_n",
        ):
            _number(row[field], path=f"result.states[{offset}].{field}", minimum=0.0)
        exact_tv = row["exact_mixture_deployment_identity_tv"]
        realized_tv = row["deployment_discrepancy_delta_dep"]
        if checked_request["method_id"] in EXACT_METHOD_IDS:
            _number(
                exact_tv,
                path=f"result.states[{offset}].exact_mixture_deployment_identity_tv",
                minimum=0.0,
                maximum=1.0,
            )
            if realized_tv is not None:
                raise TheoryBridgeV2SchemaError(
                    "Exact method cannot report realized-policy discrepancy."
                )
        else:
            _number(
                realized_tv,
                path=f"result.states[{offset}].deployment_discrepancy_delta_dep",
                minimum=0.0,
                maximum=1.0,
            )
            if exact_tv is not None:
                raise TheoryBridgeV2SchemaError(
                    "Realized method cannot claim exact-mixture identity."
                )
        if float(row["training_estimator_parity_max_abs_error"]) > float(
            row["training_estimator_parity_tolerance"]
        ):
            raise TheoryBridgeV2SchemaError(
                "Trainer/bridge advantage parity exceeds tolerance."
            )
        if float(row["constructed_centering_roundoff"]) > float(
            row["constructed_centering_tolerance"]
        ):
            raise TheoryBridgeV2SchemaError(
                "Constructed centering roundoff exceeds tolerance."
            )
        gamma = float(checked_request["gamma"])
        horizon = int(checked_request["bellman_horizon"])
        expected_d_nm = abs(float(row["value_n"]) - float(row["value_m"]))
        expected_residual = abs(
            float(row["value_m"]) - float(row["bellman_operator_mean_m"])
        )
        expected_b_nm = expected_d_nm + expected_residual / (1.0 - gamma**horizon)
        expected_b_se = float(row["bellman_operator_standard_error_m"]) / (
            1.0 - gamma**horizon
        )
        expected_e_n = abs(float(row["value_n"]) - float(row["V_hat_pi"]))
        for field, expected in (
            ("D_nm", expected_d_nm),
            ("bellman_residual_proxy", expected_residual),
            ("B_nm", expected_b_nm),
            ("B_nm_standard_error", expected_b_se),
            ("E_n", expected_e_n),
        ):
            if not math.isclose(
                float(row[field]), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise TheoryBridgeV2SchemaError(
                    f"result.states[{offset}].{field} differs from its definition."
                )
        checked_rows.append(row)
    metrics = _fields(result["metrics"], set(THEORY_METRIC_IDS), path="result.metrics")
    for metric in THEORY_METRIC_IDS:
        observed = _summary(metrics[metric], path=f"result.metrics.{metric}")
        values = [float(row[metric]) for row in checked_rows]
        expected = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        if observed["count"] != expected["count"] or any(
            not math.isclose(
                float(observed[field]),
                float(expected[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field in ("mean", "minimum", "maximum")
        ):
            raise TheoryBridgeV2SchemaError(f"Metric {metric} differs from state rows.")
    deployment = _fields(
        result["deployment_diagnostic"],
        {"kind", "metric", "summary"},
        path="result.deployment_diagnostic",
    )
    expected_kind = (
        "exact_mixture_identity"
        if checked_request["method_id"] in EXACT_METHOD_IDS
        else "realized_policy_discrepancy"
    )
    expected_metric = (
        "exact_mixture_deployment_identity_tv"
        if checked_request["method_id"] in EXACT_METHOD_IDS
        else "deployment_discrepancy_delta_dep"
    )
    if deployment["kind"] != expected_kind or deployment["metric"] != expected_metric:
        raise TheoryBridgeV2SchemaError("Deployment diagnostic has the wrong role.")
    deployment_summary = _summary(
        deployment["summary"], path="result.deployment_diagnostic.summary"
    )
    deployment_values = [float(row[expected_metric]) for row in checked_rows]
    expected_deployment_summary = {
        "count": len(deployment_values),
        "mean": sum(deployment_values) / len(deployment_values),
        "minimum": min(deployment_values),
        "maximum": max(deployment_values),
    }
    if deployment_summary["count"] != expected_deployment_summary["count"] or any(
        not math.isclose(
            float(deployment_summary[field]),
            float(expected_deployment_summary[field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field in ("mean", "minimum", "maximum")
    ):
        raise TheoryBridgeV2SchemaError("Deployment summary differs from state rows.")
    persistent = _fields(
        result["persistent_semantics"],
        {
            "applicable",
            "endpoint_depth_only",
            "carried_successor_latent_independent_of_m",
            "trajectory_identity_independent_of_m",
            "action_probabilities_independent_of_m",
            "recurrent_transition_independent_of_m",
        },
        path="result.persistent_semantics",
    )
    if persistent["applicable"] is not (checked_request["latent_mode"] == "persistent"):
        raise TheoryBridgeV2SchemaError("Persistent applicability is wrong.")
    if any(
        persistent[field] is not True for field in persistent if field != "applicable"
    ):
        raise TheoryBridgeV2SchemaError(
            "Persistent endpoint-only invariants were not proved."
        )
    readonly = _fields(
        result["read_only_verification"],
        {
            "checkpoint_unchanged",
            "model_state_unchanged",
            "mutable_training_state_unchanged",
            "recurrent_transition_unchanged",
            "mutable_state_sha256s_before",
            "mutable_state_sha256s_after",
        },
        path="result.read_only_verification",
    )
    if (
        any(
            readonly[field] is not True
            for field in (
                "checkpoint_unchanged",
                "model_state_unchanged",
                "mutable_training_state_unchanged",
                "recurrent_transition_unchanged",
            )
        )
        or readonly["mutable_state_sha256s_before"]
        != readonly["mutable_state_sha256s_after"]
    ):
        raise TheoryBridgeV2SchemaError("Theory evaluator did not remain read-only.")
    for side in ("mutable_state_sha256s_before", "mutable_state_sha256s_after"):
        inventory = _object(
            readonly[side], path=f"result.read_only_verification.{side}"
        )
        if not inventory:
            raise TheoryBridgeV2SchemaError("Mutable-state inventory cannot be empty.")
        for name, digest in inventory.items():
            _text(name, path=f"result.read_only_verification.{side}.name")
            _sha256(digest, path=f"result.read_only_verification.{side}.{name}")
    uncertainty = _fields(
        result["monte_carlo_uncertainty"],
        {
            "bellman_operator_standard_error",
            "B_nm_standard_error",
            "V_hat_pi_standard_error",
        },
        path="result.monte_carlo_uncertainty",
    )
    for field in uncertainty:
        _summary(uncertainty[field], path=f"result.monte_carlo_uncertainty.{field}")
    return dict(result)
