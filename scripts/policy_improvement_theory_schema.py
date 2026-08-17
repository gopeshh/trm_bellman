#!/usr/bin/env fbpython
"""Strict schemas for the read-only policy-improvement theory bridge.

The bridge reports finite-batch proxies.  It never upgrades a retained sample
maximum into a uniform certificate, and it never participates in method or
hyperparameter selection.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.policy_improvement_schema import canonical_json_bytes


THEORY_AMENDMENT_SCHEMA_NAME = "policy_improvement_theory_bridge_amendment_v1"
THEORY_REQUEST_SCHEMA_NAME = "policy_improvement_theory_bridge_request_v1"
THEORY_RESULT_SCHEMA_NAME = "policy_improvement_theory_bridge_result_v1"
THEORY_SCHEMA_VERSION = 1
THEORY_METHOD_IDS = (
    "fixed_base_exact_persistent",
    "fixed_base_exact_episodic",
)
THEORY_METRIC_IDS = (
    "D_nm",
    "bellman_residual_proxy",
    "B_nm",
    "policy_overlap_tau",
    "centering_defect",
    "deployment_tv",
)
_THEORY_FINAL_CHECKPOINT_ENVIRONMENT_INTERACTIONS = 80000
_THEORY_SNAPSHOT_KINDS = {"scheduled", "interaction_matched"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")


class TheoryBridgeSchemaError(ValueError):
    """Raised when a theory-bridge document is not canonical."""


def _expected_checkpoint_snapshot_kind(
    environment_interactions: int,
    *,
    final_environment_interactions: int = (
        _THEORY_FINAL_CHECKPOINT_ENVIRONMENT_INTERACTIONS
    ),
) -> str:
    return (
        "interaction_matched"
        if environment_interactions == final_environment_interactions
        else "scheduled"
    )


def theory_document_sha256(value: object) -> str:
    """Hash one already validated theory-bridge document canonically."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TheoryBridgeSchemaError(f"{path} must be an object.")
    return value


def _fields(value: object, expected: set[str], *, path: str) -> Mapping[str, object]:
    item = _object(value, path=path)
    actual = set(item)
    if actual != expected:
        raise TheoryBridgeSchemaError(
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
        raise TheoryBridgeSchemaError(f"{path} must be nonempty ASCII text.")
    if choices is not None and value not in choices:
        raise TheoryBridgeSchemaError(f"{path} must be one of {sorted(choices)}.")
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise TheoryBridgeSchemaError(f"{path} is not a canonical identifier.")
    return value


def _sha256(value: object, *, path: str) -> str:
    text = _text(value, path=path)
    if _SHA256.fullmatch(text) is None:
        raise TheoryBridgeSchemaError(f"{path} must be a lowercase SHA-256 digest.")
    return text


def _git_commit(value: object, *, path: str) -> str:
    text = _text(value, path=path)
    if _GIT_COMMIT.fullmatch(text) is None:
        raise TheoryBridgeSchemaError(f"{path} must be a full lowercase Git commit.")
    return text


def _integer(value: object, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TheoryBridgeSchemaError(
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
        raise TheoryBridgeSchemaError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise TheoryBridgeSchemaError(f"{path} must be finite.")
    if minimum is not None and result < minimum:
        raise TheoryBridgeSchemaError(f"{path} is below its minimum.")
    if maximum is not None and result > maximum:
        raise TheoryBridgeSchemaError(f"{path} is above its maximum.")
    return result


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TheoryBridgeSchemaError(f"{path} must be boolean.")
    return value


def _string_list(
    value: object,
    *,
    path: str,
    expected: Sequence[str] | None = None,
) -> list[str]:
    if not isinstance(value, list):
        raise TheoryBridgeSchemaError(f"{path} must be a list.")
    checked = [_text(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if expected is not None and checked != list(expected):
        raise TheoryBridgeSchemaError(f"{path} differs from the registered order.")
    return checked


def validate_identity_bundle(value: object) -> dict[str, Any]:
    """Validate all identities that one result must bind."""

    identity = _fields(
        value,
        {
            "protocol_sha256",
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
    for field in (
        "protocol_sha256",
        "theory_amendment_sha256",
        "registry_row_sha256",
    ):
        _sha256(identity[field], path=f"identity.{field}")

    checkpoint = _fields(
        identity["checkpoint"],
        {
            "sha256",
            "size_bytes",
            "snapshot_kind",
            "environment_interactions",
        },
        path="identity.checkpoint",
    )
    _sha256(checkpoint["sha256"], path="identity.checkpoint.sha256")
    _integer(checkpoint["size_bytes"], path="identity.checkpoint.size_bytes", minimum=1)
    _text(
        checkpoint["snapshot_kind"],
        path="identity.checkpoint.snapshot_kind",
        choices=_THEORY_SNAPSHOT_KINDS,
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
        {
            "file_sha256",
            "base_canonical_sha256",
            "effective_config_sha256",
        },
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
    if training["role"] != "policy-improvement-full":
        raise TheoryBridgeSchemaError("Training runtime has the wrong launcher role.")
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
        raise TheoryBridgeSchemaError(
            "Training runtime source-profile identities differ."
        )
    if training["source_git_commit"] != producer["git_commit"]:
        raise TheoryBridgeSchemaError(
            "Training runtime and producer source name different commits."
        )

    dataset = _fields(
        identity["dataset_records"],
        {
            "split",
            "split_manifest_sha256",
            "ordered_record_sha256",
            "selected_record_indices_sha256",
            "selected_records_sha256",
            "record_count",
        },
        path="identity.dataset_records",
    )
    _text(
        dataset["split"],
        path="identity.dataset_records.split",
        choices={"train", "validation"},
    )
    for field in (
        "split_manifest_sha256",
        "ordered_record_sha256",
        "selected_record_indices_sha256",
        "selected_records_sha256",
    ):
        _sha256(dataset[field], path=f"identity.dataset_records.{field}")
    _integer(
        dataset["record_count"], path="identity.dataset_records.record_count", minimum=1
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


def validate_theory_request(value: object) -> dict[str, Any]:
    """Validate one exact registered theory evaluation request."""

    request = _fields(
        value,
        {
            "schema_name",
            "schema_version",
            "evaluation_id",
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
            "identity",
            "test_data_opened",
        },
        path="request",
    )
    if (
        request["schema_name"] != THEORY_REQUEST_SCHEMA_NAME
        or request["schema_version"] != THEORY_SCHEMA_VERSION
    ):
        raise TheoryBridgeSchemaError("Theory request schema is unsupported.")
    _text(request["evaluation_id"], path="request.evaluation_id", identifier=True)
    _text(request["run_id"], path="request.run_id", identifier=True)
    method = _text(
        request["method_id"],
        path="request.method_id",
        choices=set(THEORY_METHOD_IDS),
    )
    latent_mode = _text(
        request["latent_mode"],
        path="request.latent_mode",
        choices={"persistent", "episodic"},
    )
    expected_mode = (
        "persistent" if method == "fixed_base_exact_persistent" else "episodic"
    )
    if latent_mode != expected_mode:
        raise TheoryBridgeSchemaError("Method and latent mode disagree.")
    checkpoint_environment_interactions = _integer(
        request["checkpoint_environment_interactions"],
        path="request.checkpoint_environment_interactions",
        minimum=1,
    )
    n = _integer(request["n"], path="request.n", minimum=1)
    reference = _integer(
        request["reference_depth_m"], path="request.reference_depth_m", minimum=2
    )
    if reference <= n or reference not in (8, 16):
        raise TheoryBridgeSchemaError(
            "Reference depth must be registered and exceed n."
        )
    role = _text(
        request["reference_role"],
        path="request.reference_role",
        choices={"primary", "exploratory"},
    )
    if (reference == 8) != (role == "primary"):
        raise TheoryBridgeSchemaError("Reference depth and analysis role disagree.")
    horizon = _integer(
        request["bellman_horizon"], path="request.bellman_horizon", minimum=1
    )
    if horizon not in (1, 5):
        raise TheoryBridgeSchemaError("Bellman horizon must be 1 or 5.")
    _number(request["gamma"], path="request.gamma", minimum=0.0, maximum=1.0)
    if float(request["gamma"]) >= 1.0:
        raise TheoryBridgeSchemaError("Gamma must be strictly below one.")
    _number(request["alpha"], path="request.alpha", minimum=0.0, maximum=1.0)
    estimator = _fields(
        request["bellman_estimator"],
        {"kind", "rollout_count", "base_seed", "seed_derivation"},
        path="request.bellman_estimator",
    )
    rollout_count = _integer(
        estimator["rollout_count"], path="request.bellman_estimator.rollout_count"
    )
    if horizon == 1:
        if (
            estimator["kind"] != "exact_masked_action_sum_v1"
            or rollout_count != 0
            or estimator["base_seed"] is not None
            or estimator["seed_derivation"] != "not_applicable"
        ):
            raise TheoryBridgeSchemaError("K=1 requires exact masked action summation.")
    else:
        if (
            estimator["kind"] != "common_random_number_monte_carlo_v1"
            or rollout_count < 2
            or isinstance(estimator["base_seed"], bool)
            or not isinstance(estimator["base_seed"], int)
            or estimator["base_seed"] < 0
            or estimator["seed_derivation"] != "sha256_base_record_state_repeat_v1"
        ):
            raise TheoryBridgeSchemaError("K=5 requires registered CRN Monte Carlo.")
    identity = validate_identity_bundle(request["identity"])
    checkpoint = identity["checkpoint"]
    if (
        checkpoint["snapshot_kind"]
        != _expected_checkpoint_snapshot_kind(checkpoint_environment_interactions)
        or checkpoint["environment_interactions"] != checkpoint_environment_interactions
    ):
        raise TheoryBridgeSchemaError(
            "Request checkpoint progress differs from its embedded identity."
        )
    if _boolean(request["test_data_opened"], path="request.test_data_opened"):
        raise TheoryBridgeSchemaError("Theory bridge cannot open test data.")
    return dict(request)


def validate_theory_amendment(
    value: object,
    *,
    protocol: Mapping[str, object] | None = None,
    registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the pre-outcome theory-bridge protocol amendment."""

    amendment = _fields(
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
            "outcome_evidence_inspected",
            "result_schema",
            "evaluator_contract",
            "metrics",
            "reference_depths",
            "bellman_estimators",
            "multi_fidelity_exact_method_screen",
            "analysis",
            "smoke_policy",
        },
        path="amendment",
    )
    if (
        amendment["schema_name"] != THEORY_AMENDMENT_SCHEMA_NAME
        or amendment["schema_version"] != THEORY_SCHEMA_VERSION
    ):
        raise TheoryBridgeSchemaError("Theory amendment schema is unsupported.")
    _text(amendment["amendment_id"], path="amendment.amendment_id", identifier=True)
    _text(amendment["created_at_utc"], path="amendment.created_at_utc")
    _text(amendment["protocol_id"], path="amendment.protocol_id", identifier=True)
    _sha256(amendment["protocol_sha256"], path="amendment.protocol_sha256")
    _sha256(
        amendment["prior_amendment_history_sha256"],
        path="amendment.prior_amendment_history_sha256",
    )
    if amendment["prior_amendment_history_sha256"] != hashlib.sha256(b"[]").hexdigest():
        raise TheoryBridgeSchemaError("Theory design must be the first amendment.")
    _sha256(
        amendment["source_registry_sha256"], path="amendment.source_registry_sha256"
    )
    if _boolean(amendment["test_data_opened"], path="amendment.test_data_opened"):
        raise TheoryBridgeSchemaError("Theory amendment must preserve test isolation.")
    if _boolean(
        amendment["outcome_evidence_inspected"],
        path="amendment.outcome_evidence_inspected",
    ):
        raise TheoryBridgeSchemaError(
            "Theory amendment must predate outcome inspection."
        )

    result_schema = _fields(
        amendment["result_schema"],
        {"schema_name", "schema_version"},
        path="amendment.result_schema",
    )
    if (
        result_schema["schema_name"] != THEORY_RESULT_SCHEMA_NAME
        or result_schema["schema_version"] != THEORY_SCHEMA_VERSION
    ):
        raise TheoryBridgeSchemaError("Amendment registers the wrong result schema.")

    contract = _fields(
        amendment["evaluator_contract"],
        {
            "launcher_purpose",
            "runtime_role",
            "source_tree_fallback",
            "checkpoint_access",
            "training_state_mutation",
            "persistent_reference_semantics",
        },
        path="amendment.evaluator_contract",
    )
    expected_contract = {
        "launcher_purpose": "policy-improvement-theory-bridge",
        "runtime_role": "policy-improvement-theory-bridge",
        "source_tree_fallback": False,
        "checkpoint_access": "read_only_stable_regular_file",
        "training_state_mutation": "forbidden",
        "persistent_reference_semantics": (
            "m_changes_endpoint_evaluator_only_F_n_and_augmented_transition_fixed"
        ),
    }
    if dict(contract) != expected_contract:
        raise TheoryBridgeSchemaError("Evaluator contract differs from registration.")

    _string_list(
        amendment["metrics"], path="amendment.metrics", expected=THEORY_METRIC_IDS
    )
    depths = _fields(
        amendment["reference_depths"],
        {"primary_m", "optional_exploratory_m", "selection_use"},
        path="amendment.reference_depths",
    )
    if (
        depths["primary_m"] != 8
        or depths["optional_exploratory_m"] != [16]
        or depths["selection_use"] != "none"
    ):
        raise TheoryBridgeSchemaError("Reference-depth registration differs.")

    estimators = _fields(
        amendment["bellman_estimators"],
        {"K1", "K5"},
        path="amendment.bellman_estimators",
    )
    k1 = _fields(
        estimators["K1"],
        {"horizon", "kind", "rollout_count", "uncertainty"},
        path="amendment.bellman_estimators.K1",
    )
    if dict(k1) != {
        "horizon": 1,
        "kind": "exact_masked_action_sum_v1",
        "rollout_count": 0,
        "uncertainty": "exact_zero_monte_carlo_standard_error",
    }:
        raise TheoryBridgeSchemaError("K=1 estimator registration differs.")
    k5 = _fields(
        estimators["K5"],
        {
            "horizon",
            "kind",
            "rollout_count",
            "base_seed",
            "seed_derivation",
            "uncertainty",
        },
        path="amendment.bellman_estimators.K5",
    )
    if (
        k5["horizon"] != 5
        or k5["kind"] != "common_random_number_monte_carlo_v1"
        or k5["rollout_count"] != 64
        or k5["base_seed"] != 26081601
        or k5["seed_derivation"] != "sha256_base_record_state_repeat_v1"
        or k5["uncertainty"] != "sample_standard_error_of_operator_mean"
    ):
        raise TheoryBridgeSchemaError("K=5 estimator registration differs.")

    screen = _fields(
        amendment["multi_fidelity_exact_method_screen"],
        {"methods", "split", "record_selector", "fidelities", "selection_use"},
        path="amendment.multi_fidelity_exact_method_screen",
    )
    _string_list(
        screen["methods"],
        path="amendment.multi_fidelity_exact_method_screen.methods",
        expected=THEORY_METHOD_IDS,
    )
    if (
        screen["split"] != "validation"
        or screen["record_selector"] != "registered_contiguous_prefix_v1"
        or screen["selection_use"] != "none"
    ):
        raise TheoryBridgeSchemaError("Exact-method screen registration differs.")
    if not isinstance(screen["fidelities"], list) or len(screen["fidelities"]) != 4:
        raise TheoryBridgeSchemaError("Exactly four screen fidelities are required.")
    expected_fidelities = (
        (10000, 64, [8]),
        (20000, 64, [8]),
        (40000, 128, [8]),
        (80000, 256, [8, 16]),
    )
    for index, (raw, expected) in enumerate(
        zip(screen["fidelities"], expected_fidelities)
    ):
        fidelity = _fields(
            raw,
            {"checkpoint_environment_interactions", "record_count", "reference_depths"},
            path=f"amendment.multi_fidelity_exact_method_screen.fidelities[{index}]",
        )
        observed = (
            fidelity["checkpoint_environment_interactions"],
            fidelity["record_count"],
            fidelity["reference_depths"],
        )
        if observed != expected:
            raise TheoryBridgeSchemaError("Screen fidelity differs from registration.")

    analysis = _fields(
        amendment["analysis"],
        {"scope", "aggregation", "mc_reporting", "scientific_selection", "claims"},
        path="amendment.analysis",
    )
    if dict(analysis) != {
        "scope": "finite_batch_proxy_only",
        "aggregation": "per_state_rows_plus_mean_and_maximum",
        "mc_reporting": "operator_mean_standard_error_and_B_nm_standard_error",
        "scientific_selection": False,
        "claims": "none_from_smoke_or_theory_bridge_values",
    }:
        raise TheoryBridgeSchemaError("Theory analysis registration differs.")

    smoke = _fields(
        amendment["smoke_policy"],
        {"allowed_sources", "test_records", "scientific_selection"},
        path="amendment.smoke_policy",
    )
    if dict(smoke) != {
        "allowed_sources": ["synthetic", "train"],
        "test_records": False,
        "scientific_selection": False,
    }:
        raise TheoryBridgeSchemaError("Smoke policy differs from registration.")

    if protocol is not None:
        protocol_digest = theory_document_sha256(protocol)
        if (
            amendment["protocol_id"] != protocol.get("protocol_id")
            or amendment["protocol_sha256"] != protocol_digest
        ):
            raise TheoryBridgeSchemaError(
                "Theory amendment does not bind the protocol."
            )
    if (
        registry_sha256 is not None
        and amendment["source_registry_sha256"] != registry_sha256
    ):
        raise TheoryBridgeSchemaError(
            "Theory amendment does not bind the source registry."
        )
    return dict(amendment)


def validate_request_against_amendment(
    request_value: object,
    amendment_value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one request to the exact pre-outcome amendment."""

    request = validate_theory_request(request_value)
    amendment = validate_theory_amendment(amendment_value)
    if request["identity"]["theory_amendment_sha256"] != theory_document_sha256(
        amendment
    ):
        raise TheoryBridgeSchemaError("Request names a different theory amendment.")
    if request["identity"]["protocol_sha256"] != amendment["protocol_sha256"]:
        raise TheoryBridgeSchemaError("Request and amendment name different protocols.")
    estimator_key = f"K{request['bellman_horizon']}"
    registered_estimator = amendment["bellman_estimators"][estimator_key]
    supplied_estimator = request["bellman_estimator"]
    if request["bellman_horizon"] == 1:
        matches_estimator = (
            supplied_estimator["kind"] == registered_estimator["kind"]
            and supplied_estimator["rollout_count"] == 0
            and supplied_estimator["base_seed"] is None
        )
    else:
        matches_estimator = (
            supplied_estimator["kind"] == registered_estimator["kind"]
            and supplied_estimator["rollout_count"]
            == registered_estimator["rollout_count"]
            and supplied_estimator["base_seed"] == registered_estimator["base_seed"]
            and supplied_estimator["seed_derivation"]
            == registered_estimator["seed_derivation"]
        )
    if not matches_estimator:
        raise TheoryBridgeSchemaError(
            "Request changes the registered Bellman estimator."
        )
    matching_fidelity = None
    for fidelity in amendment["multi_fidelity_exact_method_screen"]["fidelities"]:
        if (
            fidelity["checkpoint_environment_interactions"]
            == request["checkpoint_environment_interactions"]
        ):
            matching_fidelity = fidelity
            break
    if matching_fidelity is None:
        raise TheoryBridgeSchemaError("Request checkpoint is not registered.")
    registered_fidelities = amendment["multi_fidelity_exact_method_screen"][
        "fidelities"
    ]
    final_checkpoint = max(
        int(fidelity["checkpoint_environment_interactions"])
        for fidelity in registered_fidelities
    )
    checkpoint_identity = request["identity"]["checkpoint"]
    if checkpoint_identity["snapshot_kind"] != _expected_checkpoint_snapshot_kind(
        int(request["checkpoint_environment_interactions"]),
        final_environment_interactions=final_checkpoint,
    ):
        raise TheoryBridgeSchemaError(
            "Request checkpoint snapshot kind differs from the registered schedule."
        )
    dataset = request["identity"]["dataset_records"]
    if (
        dataset["split"] != amendment["multi_fidelity_exact_method_screen"]["split"]
        or dataset["record_count"] != matching_fidelity["record_count"]
        or request["reference_depth_m"] not in matching_fidelity["reference_depths"]
    ):
        raise TheoryBridgeSchemaError("Request changes the registered screen fidelity.")
    return request, amendment


def _validate_summary(value: object, *, path: str) -> dict[str, object]:
    summary = _fields(value, {"count", "mean", "minimum", "maximum"}, path=path)
    count = _integer(summary["count"], path=f"{path}.count", minimum=1)
    minimum = _number(summary["minimum"], path=f"{path}.minimum", minimum=0.0)
    maximum = _number(summary["maximum"], path=f"{path}.maximum", minimum=0.0)
    mean = _number(summary["mean"], path=f"{path}.mean", minimum=0.0)
    if minimum > maximum or not minimum <= mean <= maximum:
        raise TheoryBridgeSchemaError(f"{path} ordering is invalid.")
    return {"count": count, "mean": mean, "minimum": minimum, "maximum": maximum}


def _require_summary_matches(
    summary: Mapping[str, object],
    values: Sequence[float],
    *,
    path: str,
) -> None:
    expected = {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }
    if summary["count"] != expected["count"] or any(
        not math.isclose(
            float(summary[field]),
            float(expected[field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field in ("mean", "minimum", "maximum")
    ):
        raise TheoryBridgeSchemaError(f"{path} differs from its state rows.")


def validate_theory_result(
    value: object,
    *,
    request: object,
) -> dict[str, Any]:
    """Validate one complete read-only finite-batch theory result."""

    checked_request = validate_theory_request(request)

    result = _fields(
        value,
        {
            "schema_name",
            "schema_version",
            "status",
            "scope",
            "uniform_certificate",
            "evaluation_id",
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
            "monte_carlo_uncertainty",
            "persistent_semantics",
            "read_only_verification",
            "test_data_opened",
        },
        path="result",
    )
    if (
        result["schema_name"] != THEORY_RESULT_SCHEMA_NAME
        or result["schema_version"] != THEORY_SCHEMA_VERSION
        or result["status"] != "complete"
        or result["scope"] != "finite_batch_proxy_only"
        or result["uniform_certificate"] is not False
        or result["test_data_opened"] is not False
    ):
        raise TheoryBridgeSchemaError("Theory result header is invalid.")
    _text(result["evaluation_id"], path="result.evaluation_id", identifier=True)
    _text(result["run_id"], path="result.run_id", identifier=True)
    _text(result["method_id"], path="result.method_id", choices=set(THEORY_METHOD_IDS))
    _integer(
        result["checkpoint_environment_interactions"],
        path="result.checkpoint_environment_interactions",
        minimum=1,
    )
    n = _integer(result["n"], path="result.n", minimum=1)
    m = _integer(
        result["reference_depth_m"], path="result.reference_depth_m", minimum=2
    )
    if m <= n or m not in (8, 16):
        raise TheoryBridgeSchemaError("Result reference depth is invalid.")
    _text(
        result["reference_role"],
        path="result.reference_role",
        choices={"primary", "exploratory"},
    )
    horizon = _integer(
        result["bellman_horizon"], path="result.bellman_horizon", minimum=1
    )
    if horizon not in (1, 5):
        raise TheoryBridgeSchemaError("Result Bellman horizon is invalid.")
    gamma = _number(result["gamma"], path="result.gamma", minimum=0.0, maximum=1.0)
    if gamma >= 1.0:
        raise TheoryBridgeSchemaError("Result gamma must be strictly below one.")
    _number(result["alpha"], path="result.alpha", minimum=0.0, maximum=1.0)
    _sha256(result["request_sha256"], path="result.request_sha256")
    identity = validate_identity_bundle(result["identity"])
    repeated_fields = (
        "evaluation_id",
        "run_id",
        "method_id",
        "checkpoint_environment_interactions",
        "n",
        "reference_depth_m",
        "reference_role",
        "bellman_horizon",
        "gamma",
        "alpha",
    )
    if (
        any(result[field] != checked_request[field] for field in repeated_fields)
        or result["request_sha256"] != theory_document_sha256(checked_request)
        or canonical_json_bytes(identity)
        != canonical_json_bytes(checked_request["identity"])
    ):
        raise TheoryBridgeSchemaError("Theory result differs from its exact request.")
    state_count = _integer(result["state_count"], path="result.state_count", minimum=1)
    if identity["dataset_records"]["record_count"] != state_count:
        raise TheoryBridgeSchemaError(
            "Result state count differs from dataset identity."
        )
    if not isinstance(result["states"], list) or len(result["states"]) != state_count:
        raise TheoryBridgeSchemaError("Result state rows are incomplete.")
    state_ids: set[str] = set()
    checked_rows: list[Mapping[str, object]] = []
    for index, raw in enumerate(result["states"]):
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
                "centering_defect",
                "deployment_tv",
            },
            path=f"result.states[{index}]",
        )
        state_id = _text(
            row["state_id"], path=f"result.states[{index}].state_id", identifier=True
        )
        if state_id in state_ids:
            raise TheoryBridgeSchemaError("Result state IDs must be unique.")
        state_ids.add(state_id)
        _integer(row["record_index"], path=f"result.states[{index}].record_index")
        _sha256(
            row["dataset_record_sha256"],
            path=f"result.states[{index}].dataset_record_sha256",
        )
        _sha256(
            row["registered_state_sha256"],
            path=f"result.states[{index}].registered_state_sha256",
        )
        _number(row["value_n"], path=f"result.states[{index}].value_n")
        _number(row["value_m"], path=f"result.states[{index}].value_m")
        for field in (
            "D_nm",
            "bellman_operator_standard_error_m",
            "bellman_residual_proxy",
            "B_nm",
            "B_nm_standard_error",
            "policy_overlap_tau",
            "centering_defect",
            "deployment_tv",
        ):
            maximum = 1.0 if field in {"policy_overlap_tau", "deployment_tv"} else None
            _number(
                row[field],
                path=f"result.states[{index}].{field}",
                minimum=0.0,
                maximum=maximum,
            )
        _number(
            row["bellman_operator_mean_m"],
            path=f"result.states[{index}].bellman_operator_mean_m",
        )
        expected_d_nm = abs(float(row["value_n"]) - float(row["value_m"]))
        expected_residual = abs(
            float(row["value_m"]) - float(row["bellman_operator_mean_m"])
        )
        denominator = 1.0 - gamma**horizon
        expected_b_nm = expected_d_nm + expected_residual / denominator
        expected_b_se = float(row["bellman_operator_standard_error_m"]) / denominator
        for field, expected in (
            ("D_nm", expected_d_nm),
            ("bellman_residual_proxy", expected_residual),
            ("B_nm", expected_b_nm),
            ("B_nm_standard_error", expected_b_se),
        ):
            if not math.isclose(
                float(row[field]), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise TheoryBridgeSchemaError(
                    f"result.states[{index}].{field} differs from its definition."
                )
        if horizon == 1 and (
            row["bellman_operator_standard_error_m"] != 0.0
            or row["B_nm_standard_error"] != 0.0
        ):
            raise TheoryBridgeSchemaError(
                "Exact K=1 rows cannot report MC uncertainty."
            )
        checked_rows.append(row)

    dataset = identity["dataset_records"]
    selected_indices = [int(row["record_index"]) for row in checked_rows]
    selected_records = [
        {
            "record_index": int(row["record_index"]),
            "dataset_record_sha256": row["dataset_record_sha256"],
        }
        for row in checked_rows
    ]
    if (
        hashlib.sha256(canonical_json_bytes(selected_indices)).hexdigest()
        != dataset["selected_record_indices_sha256"]
        or hashlib.sha256(canonical_json_bytes(selected_records)).hexdigest()
        != dataset["selected_records_sha256"]
    ):
        raise TheoryBridgeSchemaError(
            "Result state rows differ from the selected dataset identity."
        )

    metrics = _fields(result["metrics"], set(THEORY_METRIC_IDS), path="result.metrics")
    for metric in THEORY_METRIC_IDS:
        summary = _validate_summary(metrics[metric], path=f"result.metrics.{metric}")
        if summary["count"] != state_count:
            raise TheoryBridgeSchemaError(
                "Metric count differs from result state count."
            )
        _require_summary_matches(
            summary,
            [float(row[metric]) for row in checked_rows],
            path=f"result.metrics.{metric}",
        )

    uncertainty = _fields(
        result["monte_carlo_uncertainty"],
        {"kind", "operator_standard_error", "B_nm_standard_error"},
        path="result.monte_carlo_uncertainty",
    )
    expected_uncertainty_kind = (
        "exact_zero_monte_carlo_standard_error"
        if horizon == 1
        else "sample_standard_error_of_operator_mean"
    )
    if uncertainty["kind"] != expected_uncertainty_kind:
        raise TheoryBridgeSchemaError("Result uses the wrong uncertainty estimator.")
    for field in ("operator_standard_error", "B_nm_standard_error"):
        summary = _validate_summary(
            uncertainty[field],
            path=f"result.monte_carlo_uncertainty.{field}",
        )
        if summary["count"] != state_count:
            raise TheoryBridgeSchemaError("Uncertainty count differs from state count.")
        if horizon == 1 and summary["maximum"] != 0.0:
            raise TheoryBridgeSchemaError("Exact K=1 uncertainty must be zero.")
        row_field = (
            "bellman_operator_standard_error_m"
            if field == "operator_standard_error"
            else "B_nm_standard_error"
        )
        _require_summary_matches(
            summary,
            [float(row[row_field]) for row in checked_rows],
            path=f"result.monte_carlo_uncertainty.{field}",
        )

    persistent = _fields(
        result["persistent_semantics"],
        {"applicable", "transition_sha256", "endpoint_depth_only", "F_n_unchanged"},
        path="result.persistent_semantics",
    )
    _boolean(persistent["applicable"], path="result.persistent_semantics.applicable")
    _sha256(
        persistent["transition_sha256"],
        path="result.persistent_semantics.transition_sha256",
    )
    if (
        persistent["applicable"] is not (checked_request["latent_mode"] == "persistent")
        or persistent["transition_sha256"]
        != identity["model"]["recurrent_transition_sha256"]
        or persistent["endpoint_depth_only"] is not True
        or persistent["F_n_unchanged"] is not True
    ):
        raise TheoryBridgeSchemaError(
            "Persistent endpoint-only contract was not verified."
        )

    readonly = _fields(
        result["read_only_verification"],
        {
            "checkpoint_sha256_before",
            "checkpoint_sha256_after",
            "model_state_sha256_before",
            "model_state_sha256_after",
            "training_state_sha256_before",
            "training_state_sha256_after",
            "checkpoint_snapshot_kind_before",
            "checkpoint_snapshot_kind_after",
            "checkpoint_environment_interactions_before",
            "checkpoint_environment_interactions_after",
            "checkpoint_unchanged",
            "model_state_unchanged",
            "training_state_unchanged",
        },
        path="result.read_only_verification",
    )
    for field in (
        "checkpoint_sha256_before",
        "checkpoint_sha256_after",
        "model_state_sha256_before",
        "model_state_sha256_after",
        "training_state_sha256_before",
        "training_state_sha256_after",
    ):
        _sha256(readonly[field], path=f"result.read_only_verification.{field}")
    for field in (
        "checkpoint_snapshot_kind_before",
        "checkpoint_snapshot_kind_after",
    ):
        _text(
            readonly[field],
            path=f"result.read_only_verification.{field}",
            choices=_THEORY_SNAPSHOT_KINDS,
        )
    for field in (
        "checkpoint_environment_interactions_before",
        "checkpoint_environment_interactions_after",
    ):
        _integer(
            readonly[field],
            path=f"result.read_only_verification.{field}",
            minimum=1,
        )
    if any(
        readonly[field] is not True
        for field in (
            "checkpoint_unchanged",
            "model_state_unchanged",
            "training_state_unchanged",
        )
    ):
        raise TheoryBridgeSchemaError("Theory evaluator did not remain read-only.")
    checkpoint_identity = identity["checkpoint"]
    if (
        readonly["checkpoint_sha256_before"] != checkpoint_identity["sha256"]
        or readonly["checkpoint_sha256_after"] != checkpoint_identity["sha256"]
        or readonly["model_state_sha256_before"] != identity["model"]["model_sha256"]
        or readonly["model_state_sha256_after"] != identity["model"]["model_sha256"]
        or readonly["training_state_sha256_before"]
        != readonly["training_state_sha256_after"]
        or readonly["checkpoint_snapshot_kind_before"]
        != checkpoint_identity["snapshot_kind"]
        or readonly["checkpoint_snapshot_kind_after"]
        != checkpoint_identity["snapshot_kind"]
        or readonly["checkpoint_environment_interactions_before"]
        != checkpoint_identity["environment_interactions"]
        or readonly["checkpoint_environment_interactions_after"]
        != checkpoint_identity["environment_interactions"]
    ):
        raise TheoryBridgeSchemaError(
            "Read-only verification differs from the request identity."
        )
    return dict(result)
