#!/usr/bin/env fbpython
"""Registered primary analysis for audited Stage 2 policy-improvement evidence.

This consumer is evidence-eligible only through the authenticated analysis
launcher role.  The Python API still checks every supplied runtime identity;
it cannot establish pre-import authentication by itself.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.policy_improvement_audit import (
    _load_historical_runtime_authorizations,
    audit_result_set,
)
from scripts.policy_improvement_registry import load_registered_base_configs
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
    load_strict_json,
    load_strict_json_bytes,
    runtime_authorization_sha256,
    validate_amendment_history,
    validate_protocol,
    validate_runtime_authorization,
)
from scripts.policy_improvement_statistics import (
    holm_adjust,
    paired_seed_cluster_puzzle_bootstrap,
    paired_seed_permutation_test,
)


ANALYSIS_SCHEMA_VERSION = 1


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementSchemaError(f"{path} must be an object.")
    return value


def _sha256(value: object, *, path: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyImprovementSchemaError(
            f"{path} must be {length} lowercase hexadecimal characters."
        )
    return value


def _available(value: object, *, path: str) -> object:
    item = _mapping(value, path=path)
    if set(item) != {"status", "value"} or item["status"] != "available":
        raise PolicyImprovementSchemaError(f"{path} must be available.")
    return item["value"]


def _role(authorization: Mapping[str, object], role_name: str) -> Mapping[str, object]:
    for role in authorization["roles"]:
        if role["role"] == role_name:
            return role
    raise AssertionError(f"Validated authorization omitted {role_name!r}.")


def _validate_analysis_execution(
    value: object, authorization: Mapping[str, object]
) -> dict[str, str]:
    identity = _mapping(value, path="analysis_execution_identity")
    expected_fields = {
        "runtime_sha256",
        "runtime_profile_sha256",
        "source_git_commit",
        "launcher_sha256",
        "producer_git_commit",
        "producer_source_manifest_sha256",
    }
    if set(identity) != expected_fields:
        raise PolicyImprovementSchemaError("Analysis execution-identity fields differ.")
    for field in expected_fields - {"producer_git_commit", "source_git_commit"}:
        _sha256(identity[field], path=f"analysis_execution_identity.{field}")
    _sha256(
        identity["source_git_commit"],
        path="analysis_execution_identity.source_git_commit",
        length=40,
    )
    _sha256(
        identity["producer_git_commit"],
        path="analysis_execution_identity.producer_git_commit",
        length=40,
    )
    role = _role(authorization, "policy-improvement-analysis")
    expected = {
        "runtime_sha256": role["runtime_sha256"],
        "runtime_profile_sha256": role["runtime_profile_sha256"],
        "source_git_commit": role["source_git_commit"],
        "launcher_sha256": authorization["launcher_sha256"],
        "producer_git_commit": authorization["producer_git_commit"],
        "producer_source_manifest_sha256": authorization[
            "producer_source_manifest_sha256"
        ],
    }
    if dict(identity) != expected:
        raise PolicyImprovementSchemaError(
            "Analysis execution identity is not externally authorized."
        )
    return {field: str(identity[field]) for field in sorted(identity)}


def _authorized_audit_execution(authorization: Mapping[str, object]) -> dict[str, str]:
    role = _role(authorization, "policy-improvement-audit")
    return {
        "runtime_sha256": str(role["runtime_sha256"]),
        "runtime_profile_sha256": str(role["runtime_profile_sha256"]),
        "source_git_commit": str(role["source_git_commit"]),
        "launcher_sha256": str(authorization["launcher_sha256"]),
        "producer_git_commit": str(authorization["producer_git_commit"]),
        "producer_source_manifest_sha256": str(
            authorization["producer_source_manifest_sha256"]
        ),
    }


def _document_for_primary(
    result: Mapping[str, object],
    per_instance_documents: Mapping[str, object],
) -> Mapping[str, object]:
    snapshots = result["evaluation_snapshots"]
    snapshot = snapshots[0]
    if (
        snapshot["snapshot_kind"] != "interaction_matched"
        or snapshot["status"] != "available"
    ):
        raise PolicyImprovementSchemaError(
            "Primary analysis requires the interaction-matched snapshot."
        )
    primary_variant = result["primary_policy_variant"]
    evaluations = [
        evaluation
        for evaluation in snapshot["policy_evaluations"]
        if evaluation["policy_variant"] == primary_variant
    ]
    if len(evaluations) != 1:
        raise PolicyImprovementSchemaError(
            "Primary policy evaluation is missing or duplicated."
        )
    digest = str(
        _available(
            evaluations[0]["per_instance_artifact_sha256"],
            path="analysis.per_instance_artifact_sha256",
        )
    )
    if digest not in per_instance_documents:
        raise PolicyImprovementSchemaError("Primary per-instance evidence is missing.")
    return _mapping(
        per_instance_documents[digest], path="analysis.per_instance_document"
    )


def _pairs_for_contrast(
    results_by_method_seed: Mapping[tuple[str, int], Mapping[str, object]],
    per_instance_documents: Mapping[str, object],
    *,
    treatment_method: str,
    control_method: str,
    seeds: Sequence[int],
) -> tuple[
    dict[int, list[tuple[float, float]]],
    list[float],
    list[dict[str, object]],
]:
    pairs: dict[int, list[tuple[float, float]]] = {}
    seed_differences: list[float] = []
    per_seed: list[dict[str, object]] = []
    for seed in seeds:
        treatment = _document_for_primary(
            results_by_method_seed[(treatment_method, seed)],
            per_instance_documents,
        )
        control = _document_for_primary(
            results_by_method_seed[(control_method, seed)],
            per_instance_documents,
        )
        treatment_records = treatment["records"]
        control_records = control["records"]
        if len(treatment_records) != 512 or len(control_records) != 512:
            raise PolicyImprovementSchemaError(
                "Confirmatory analysis requires exactly 512 registered test records."
            )
        seed_pairs: list[tuple[float, float]] = []
        for index, (treatment_record, control_record) in enumerate(
            zip(treatment_records, control_records)
        ):
            expected_identity = (
                index,
                treatment_record["puzzle_id"],
                treatment_record["registered_record_sha256"],
            )
            actual_identity = (
                control_record["registered_index"],
                control_record["puzzle_id"],
                control_record["registered_record_sha256"],
            )
            if treatment_record["registered_index"] != index or actual_identity != (
                index,
                expected_identity[1],
                expected_identity[2],
            ):
                raise PolicyImprovementSchemaError(
                    "Primary methods do not share the exact ordered test population."
                )
            seed_pairs.append(
                (
                    1.0 if treatment_record["solved"] else 0.0,
                    1.0 if control_record["solved"] else 0.0,
                )
            )
        pairs[seed] = seed_pairs
        seed_differences.append(
            sum(treatment - control for treatment, control in seed_pairs)
            / len(seed_pairs)
        )
        treatment_solved = sum(int(treatment) for treatment, _ in seed_pairs)
        control_solved = sum(int(control) for _, control in seed_pairs)
        per_seed.append(
            {
                "seed": seed,
                "denominator": len(seed_pairs),
                "treatment_solved_count": treatment_solved,
                "control_solved_count": control_solved,
                "treatment_solve_rate": treatment_solved / len(seed_pairs),
                "control_solve_rate": control_solved / len(seed_pairs),
                "percentage_point_difference": (
                    (treatment_solved - control_solved) * 100.0 / len(seed_pairs)
                ),
            }
        )
    return pairs, seed_differences, per_seed


def _contrast_estimates(per_seed: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not per_seed:
        raise PolicyImprovementSchemaError("Contrast has no registered seeds.")
    treatment = sum(float(item["treatment_solve_rate"]) for item in per_seed) / len(
        per_seed
    )
    control = sum(float(item["control_solve_rate"]) for item in per_seed) / len(
        per_seed
    )
    relative: dict[str, object]
    if control == 0.0:
        relative = {
            "status": "unavailable",
            "reason": "zero_control_solve_rate",
        }
    else:
        relative = {"status": "available", "value": treatment / control}
    return {
        "treatment_solve_rate": treatment,
        "control_solve_rate": control,
        "absolute_percentage_point_difference": (treatment - control) * 100.0,
        "relative_ratio": relative,
    }


def _summarize_bootstrap(value: Mapping[str, object]) -> dict[str, object]:
    replicates = value["replicate_differences"]
    return {
        key: value[key]
        for key in (
            "schema_name",
            "schema_version",
            "seed_count",
            "replicates",
            "prng_seed",
            "confidence_level",
            "scale",
            "observed_difference",
            "interval_lower",
            "interval_upper",
        )
    } | {
        "replicate_distribution_sha256": hashlib.sha256(
            canonical_json_bytes(replicates)
        ).hexdigest()
    }


def analyze_stage2_confirmatory(
    protocol_value: object,
    registry_value: object,
    amendment_history: Sequence[object],
    results: Sequence[object],
    per_instance_documents: Mapping[str, object],
    audit_report_value: object,
    *,
    base_configs: Mapping[str, Mapping[str, object]],
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    runtime_authorization: object,
    historical_runtime_authorizations: Mapping[str, Mapping[str, object]] | None = None,
    analysis_execution_identity: object,
    amendment_evidence: Mapping[str, Mapping[str, object]],
    test_open_record: object,
    test_open_owner_root: str | Path,
    test_open_sha256: str,
    checkpoint_validator: Callable[[Mapping[str, object]], Mapping[str, object]],
    _producer_source_authenticator: (
        Callable[[Mapping[str, object]], None] | None
    ) = None,
) -> dict[str, Any]:
    """Reaudit and analyze the one registered interaction-matched primary set."""

    protocol = validate_protocol(protocol_value)
    history = validate_amendment_history(amendment_history, protocol=protocol)
    if len(history) != 3:
        raise PolicyImprovementSchemaError(
            "Confirmatory analysis requires all three validation-only freezes."
        )
    authorization = validate_runtime_authorization(runtime_authorization)
    execution = _validate_analysis_execution(analysis_execution_identity, authorization)
    semantic_revalidation = audit_result_set(
        protocol,
        registry_value,
        results,
        per_instance_documents,
        phases=["stage2_confirmatory"],
        amendment_history=history,
        base_configs=base_configs,
        project_root=project_root,
        dataset_root=dataset_root,
        evidence_root=evidence_root,
        runtime_authorization=authorization,
        historical_runtime_authorizations=historical_runtime_authorizations,
        audit_execution_identity=execution,
        checkpoint_validator=checkpoint_validator,
        execution_role_name="policy-improvement-analysis",
        amendment_evidence=amendment_evidence,
        test_open_record=test_open_record,
        test_open_owner_root=test_open_owner_root,
        test_open_sha256=test_open_sha256,
        _producer_source_authenticator=_producer_source_authenticator,
    )
    audit_execution = _authorized_audit_execution(authorization)
    expected_external_audit = {
        **semantic_revalidation,
        "execution_role": "policy-improvement-audit",
        "execution_runtime_sha256": audit_execution["runtime_sha256"],
        "execution_source_git_commit": audit_execution["source_git_commit"],
        "execution_runtime_profile_sha256": audit_execution["runtime_profile_sha256"],
    }
    if canonical_json_bytes(audit_report_value) != canonical_json_bytes(
        expected_external_audit
    ):
        raise PolicyImprovementSchemaError(
            "Analysis input audit report differs from independent reaudit."
        )
    checked_results = [
        _mapping(result, path=f"analysis.results[{index}]")
        for index, result in enumerate(results)
    ]
    failed_ids = sorted(
        str(result["run_id"])
        for result in checked_results
        if result["status"] == "failed"
    )
    if failed_ids:
        raise PolicyImprovementSchemaError(
            "Primary analysis cannot silently drop failed registered runs: "
            + ",".join(failed_ids)
        )
    seeds = [int(seed) for seed in protocol["seeds"]["confirmatory"]]
    by_method_seed = {
        (str(result["method_id"]), int(result["seed"])): result
        for result in checked_results
    }
    expected_keys = {
        (method, seed)
        for method in (
            "fixed_base_exact_persistent",
            "fixed_base_exact_episodic",
            "legacy_parameter_interpolation",
            "matched_ppo",
        )
        for seed in seeds
    }
    if set(by_method_seed) != expected_keys:
        raise PolicyImprovementSchemaError(
            "Analysis requires all four methods and all eight registered seeds."
        )
    if any(
        result["phase"] != "stage2_confirmatory" or result["evaluation_split"] != "test"
        for result in checked_results
    ):
        raise PolicyImprovementSchemaError(
            "Analysis refuses wrong-phase or non-test results."
        )

    statistics = protocol["statistics"]
    bootstrap_registration = statistics["bootstrap"]
    primary_pairs, primary_seed_differences, primary_per_seed = _pairs_for_contrast(
        by_method_seed,
        per_instance_documents,
        treatment_method="fixed_base_exact_persistent",
        control_method="matched_ppo",
        seeds=seeds,
    )
    primary_bootstrap = paired_seed_cluster_puzzle_bootstrap(
        primary_pairs,
        replicates=int(bootstrap_registration["replicates"]),
        seed=int(bootstrap_registration["seed"]),
        confidence_level=float(statistics["confidence_level"]),
    )
    primary_permutation = paired_seed_permutation_test(primary_seed_differences)

    secondary_outputs: list[dict[str, Any]] = []
    secondary_p_values: dict[str, float] = {}
    secondary_methods = (
        "fixed_base_exact_episodic",
        "legacy_parameter_interpolation",
    )
    for contrast, treatment_method in zip(
        statistics["prespecified_secondary_contrasts"], secondary_methods
    ):
        pairs, seed_differences, per_seed = _pairs_for_contrast(
            by_method_seed,
            per_instance_documents,
            treatment_method=treatment_method,
            control_method="matched_ppo",
            seeds=seeds,
        )
        bootstrap = paired_seed_cluster_puzzle_bootstrap(
            pairs,
            replicates=int(bootstrap_registration["replicates"]),
            seed=int(bootstrap_registration["seed"]),
            confidence_level=float(statistics["confidence_level"]),
        )
        permutation = paired_seed_permutation_test(seed_differences)
        secondary_p_values[str(contrast)] = float(permutation["two_sided_p_value"])
        secondary_outputs.append(
            {
                "contrast": contrast,
                "estimates": _contrast_estimates(per_seed),
                "per_seed": per_seed,
                "bootstrap": _summarize_bootstrap(bootstrap),
                "permutation": permutation,
            }
        )
    adjusted = holm_adjust(secondary_p_values)
    for output in secondary_outputs:
        output["holm_adjusted_p_value"] = adjusted[str(output["contrast"])]

    test_open_digest = hashlib.sha256(
        canonical_json_bytes(test_open_record)
    ).hexdigest()
    analysis: dict[str, Any] = {
        "schema_name": "policy_improvement_registered_analysis_v1",
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "phase": "stage2_confirmatory",
        "snapshot_kind": "interaction_matched",
        "evaluation_split": "test",
        "seed_count": 8,
        "evaluation_records_per_seed": 512,
        "failed_run_ids": [],
        "primary": {
            "contrast": statistics["primary_contrast"],
            "treatment_policy_variant": "exact_mixture",
            "control_policy_variant": "realized_policy",
            "estimates": _contrast_estimates(primary_per_seed),
            "per_seed": primary_per_seed,
            "bootstrap": _summarize_bootstrap(primary_bootstrap),
            "permutation": primary_permutation,
        },
        "prespecified_secondary": secondary_outputs,
        "provenance": {
            "protocol_sha256": semantic_revalidation["protocol_sha256"],
            "registry_sha256": semantic_revalidation["registry_sha256"],
            "amendment_history_sha256": semantic_revalidation[
                "amendment_history_sha256"
            ],
            "runtime_authorization_sha256": runtime_authorization_sha256(authorization),
            "external_audit_report_sha256": hashlib.sha256(
                canonical_json_bytes(expected_external_audit)
            ).hexdigest(),
            "semantic_revalidation_report_sha256": hashlib.sha256(
                canonical_json_bytes(semantic_revalidation)
            ).hexdigest(),
            "external_audit_runtime_sha256": audit_execution["runtime_sha256"],
            "external_audit_source_git_commit": audit_execution["source_git_commit"],
            "external_audit_runtime_profile_sha256": audit_execution[
                "runtime_profile_sha256"
            ],
            "result_set_sha256": semantic_revalidation["result_set_sha256"],
            "per_instance_set_sha256": semantic_revalidation["per_instance_set_sha256"],
            "test_open_record_sha256": test_open_digest,
            "analysis_runtime_sha256": execution["runtime_sha256"],
            "analysis_source_git_commit": execution["source_git_commit"],
            "analysis_runtime_profile_sha256": execution["runtime_profile_sha256"],
        },
        "authentication_requirement": (
            "Must be executed through the pre-import authenticated "
            "policy-improvement-analysis launcher role."
        ),
    }
    canonical_json_bytes(analysis)
    return analysis


def _load_per_instance(paths: Sequence[str]) -> dict[str, object]:
    documents: dict[str, object] = {}
    for path in paths:
        value = load_strict_json(path)
        digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        if digest in documents:
            raise PolicyImprovementSchemaError(
                "Duplicate per-instance evidence document."
            )
        documents[digest] = value
    return documents


def main(
    argv: Sequence[str] | None = None,
    *,
    checkpoint_validator: (
        Callable[[Mapping[str, object]], Mapping[str, object]] | None
    ) = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--amendment", action="append", required=True)
    parser.add_argument("--amendment-evidence", action="append", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--runtime-authorization-json", required=True)
    parser.add_argument("--runtime-authorization-sha256", required=True)
    parser.add_argument(
        "--historical-runtime-authorization",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    parser.add_argument("--analysis-runtime-sha256", required=True)
    parser.add_argument("--analysis-runtime-profile-sha256", required=True)
    parser.add_argument("--analysis-source-git-commit", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--producer-source-manifest-sha256", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--test-open-record", required=True)
    parser.add_argument("--test-open-owner-root", required=True)
    parser.add_argument("--test-open-sha256", required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--per-instance", action="append", required=True)
    arguments = parser.parse_args(argv)
    if checkpoint_validator is None:
        raise PolicyImprovementSchemaError(
            "Analysis execution requires the sealed checkpoint validator."
        )
    protocol = load_strict_json(arguments.protocol)
    evidence: dict[str, Mapping[str, object]] = {}
    for specification in arguments.amendment_evidence:
        if specification.count("=") != 1:
            raise PolicyImprovementSchemaError(
                "--amendment-evidence must use PHASE=JSON_PATH."
            )
        phase, path = specification.split("=", 1)
        if phase in evidence:
            raise PolicyImprovementSchemaError("Duplicate amendment-evidence phase.")
        evidence[phase] = _mapping(
            load_strict_json(path), path=f"amendment_evidence.{phase}"
        )
    try:
        authorization_bytes = arguments.runtime_authorization_json.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PolicyImprovementSchemaError(
            "Protected runtime authorization is not canonical ASCII JSON."
        ) from exc
    authorization = load_strict_json_bytes(authorization_bytes)
    if (
        hashlib.sha256(authorization_bytes).hexdigest()
        != arguments.runtime_authorization_sha256
        or runtime_authorization_sha256(authorization)
        != arguments.runtime_authorization_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Protected runtime authorization digest differs."
        )
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    historical_authorizations = _load_historical_runtime_authorizations(
        arguments.historical_runtime_authorization,
        protocol_sha256=protocol_digest,
    )
    analysis = analyze_stage2_confirmatory(
        protocol,
        load_strict_json(arguments.registry),
        [load_strict_json(path) for path in arguments.amendment],
        [load_strict_json(path) for path in arguments.result],
        _load_per_instance(arguments.per_instance),
        load_strict_json(arguments.audit_report),
        base_configs=load_registered_base_configs(protocol, arguments.project_root),
        project_root=arguments.project_root,
        dataset_root=arguments.dataset_root,
        evidence_root=arguments.evidence_root,
        runtime_authorization=authorization,
        historical_runtime_authorizations=historical_authorizations,
        analysis_execution_identity={
            "runtime_sha256": arguments.analysis_runtime_sha256,
            "runtime_profile_sha256": arguments.analysis_runtime_profile_sha256,
            "source_git_commit": arguments.analysis_source_git_commit,
            "launcher_sha256": arguments.launcher_sha256,
            "producer_git_commit": arguments.producer_git_commit,
            "producer_source_manifest_sha256": (
                arguments.producer_source_manifest_sha256
            ),
        },
        amendment_evidence=evidence,
        test_open_record=load_strict_json(arguments.test_open_record),
        test_open_owner_root=arguments.test_open_owner_root,
        test_open_sha256=arguments.test_open_sha256,
        checkpoint_validator=checkpoint_validator,
    )
    print(canonical_json_bytes(analysis).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
