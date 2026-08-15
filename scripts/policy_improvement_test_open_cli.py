#!/usr/bin/env fbpython
"""Authenticated one-time opening of the registered held-out test population."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Mapping, Sequence

from scripts.policy_improvement_registry import (
    generate_registry,
    load_registered_base_configs,
    registry_sha256,
    validate_registry_document,
)
from scripts.policy_improvement_audit import (
    _load_historical_runtime_authorizations,
    audit_result_set,
    derive_registered_selection,
)
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
    amendment_history_sha256,
    load_strict_json,
    load_strict_json_bytes,
    runtime_authorization_sha256,
    validate_amendment_history,
    validate_protocol,
    validate_runtime_authorization,
)
from scripts.policy_improvement_test_open import publish_test_open


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementSchemaError(f"{path} must be an object.")
    return value


def _protected_authorization(value: str, expected_sha256: str) -> dict[str, object]:
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PolicyImprovementSchemaError(
            "Protected runtime authorization is not canonical ASCII JSON."
        ) from exc
    authorization = validate_runtime_authorization(load_strict_json_bytes(payload))
    if (
        hashlib.sha256(payload).hexdigest() != expected_sha256
        or runtime_authorization_sha256(authorization) != expected_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Protected runtime authorization digest differs."
        )
    return authorization


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
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--amendment-evidence", action="append", required=True)
    parser.add_argument("--audit-runtime-sha256", required=True)
    parser.add_argument("--audit-runtime-profile-sha256", required=True)
    parser.add_argument("--audit-source-git-commit", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--producer-source-manifest-sha256", required=True)
    parser.add_argument("--runtime-authorization-json", required=True)
    parser.add_argument("--runtime-authorization-sha256", required=True)
    parser.add_argument(
        "--historical-runtime-authorization",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    arguments = parser.parse_args(argv)
    if checkpoint_validator is None:
        raise PolicyImprovementSchemaError(
            "Test opening requires the sealed checkpoint validator."
        )

    protocol = validate_protocol(load_strict_json(arguments.protocol))
    history = validate_amendment_history(
        [load_strict_json(path) for path in arguments.amendment],
        protocol=protocol,
    )
    if len(history) != 3:
        raise PolicyImprovementSchemaError(
            "Test opening requires the complete frozen selection history."
        )
    base_configs = load_registered_base_configs(protocol, arguments.project_root)
    registry = validate_registry_document(
        load_strict_json(arguments.registry),
        protocol,
        history,
        base_configs=base_configs,
    )
    authorization = _protected_authorization(
        arguments.runtime_authorization_json,
        arguments.runtime_authorization_sha256,
    )
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    if authorization["protocol_sha256"] != protocol_sha256:
        raise PolicyImprovementSchemaError(
            "Runtime authorization names another protocol."
        )
    historical_authorizations = _load_historical_runtime_authorizations(
        arguments.historical_runtime_authorization,
        protocol_sha256=protocol_sha256,
    )
    audit_role = next(
        (
            _mapping(role, path="runtime_authorization.audit_role")
            for role in authorization["roles"]
            if role["role"] == "policy-improvement-audit"
        ),
        None,
    )
    if audit_role is None:
        raise PolicyImprovementSchemaError("Runtime authorization omits audit role.")
    expected_execution = {
        "runtime_sha256": arguments.audit_runtime_sha256,
        "runtime_profile_sha256": arguments.audit_runtime_profile_sha256,
        "source_git_commit": arguments.audit_source_git_commit,
    }
    if any(audit_role[field] != value for field, value in expected_execution.items()):
        raise PolicyImprovementSchemaError(
            "Test opening is not running through the authorized audit artifact."
        )
    if (
        authorization["launcher_sha256"] != arguments.launcher_sha256
        or authorization["producer_git_commit"] != arguments.producer_git_commit
        or authorization["producer_source_manifest_sha256"]
        != arguments.producer_source_manifest_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Test-opening producer or launcher identity differs."
        )
    evidence_by_phase: dict[str, Mapping[str, object]] = {}
    for specification in arguments.amendment_evidence:
        if specification.count("=") != 1:
            raise PolicyImprovementSchemaError(
                "--amendment-evidence must use PHASE=JSON_PATH."
            )
        phase, path = specification.split("=", 1)
        if phase in evidence_by_phase:
            raise PolicyImprovementSchemaError(
                "Duplicate test-opening amendment evidence."
            )
        evidence_by_phase[phase] = _mapping(
            load_strict_json(path), path=f"amendment_evidence.{phase}"
        )
    if set(evidence_by_phase) != {
        "stage0_smoke",
        "stage1_screen",
        "stage1_alpha",
    }:
        raise PolicyImprovementSchemaError(
            "Test opening requires all three registered pre-test evidence sets."
        )
    alpha_evidence = _mapping(
        evidence_by_phase["stage1_alpha"], path="amendment_evidence.stage1_alpha"
    )
    if set(alpha_evidence) != {"results", "per_instance_documents"}:
        raise PolicyImprovementSchemaError(
            "Alpha amendment evidence has the wrong inventory."
        )
    alpha_results = alpha_evidence["results"]
    if not isinstance(alpha_results, Sequence) or isinstance(
        alpha_results, (str, bytes)
    ):
        raise PolicyImprovementSchemaError("Alpha evidence results are invalid.")
    alpha_documents = _mapping(
        alpha_evidence["per_instance_documents"],
        path="amendment_evidence.stage1_alpha.per_instance_documents",
    )
    alpha_registry = generate_registry(
        protocol,
        history[:2],
        base_configs=base_configs,
    )
    prior_evidence = {
        phase: evidence_by_phase[phase] for phase in ("stage0_smoke", "stage1_screen")
    }
    alpha_report = audit_result_set(
        protocol,
        alpha_registry,
        list(alpha_results),
        alpha_documents,
        phases=["stage1_alpha"],
        amendment_history=history[:2],
        base_configs=base_configs,
        project_root=arguments.project_root,
        dataset_root=arguments.dataset_root,
        evidence_root=arguments.evidence_root,
        runtime_authorization=authorization,
        historical_runtime_authorizations=historical_authorizations,
        audit_execution_identity={
            **expected_execution,
            "launcher_sha256": arguments.launcher_sha256,
            "producer_git_commit": arguments.producer_git_commit,
            "producer_source_manifest_sha256": (
                arguments.producer_source_manifest_sha256
            ),
        },
        checkpoint_validator=checkpoint_validator,
        amendment_evidence=prior_evidence,
    )
    expected_alpha_evidence = {
        "phase": "stage1_alpha",
        "audit_report_sha256": hashlib.sha256(
            canonical_json_bytes(alpha_report)
        ).hexdigest(),
        "result_set_sha256": alpha_report["result_set_sha256"],
        "per_instance_set_sha256": alpha_report["per_instance_set_sha256"],
        "expected_rows": alpha_report["expected_rows"],
        "complete_rows": alpha_report["complete_rows"],
        "failed_rows": alpha_report["failed_rows"],
    }
    if history[2]["evidence"] != expected_alpha_evidence:
        raise PolicyImprovementSchemaError(
            "Final selection does not bind the independently re-audited alpha grid."
        )
    if history[2]["selected_exact"] != derive_registered_selection(
        "stage1_alpha", list(alpha_results)
    ):
        raise PolicyImprovementSchemaError(
            "Final selection differs from the registered alpha rule."
        )
    if history[2]["source_registry_sha256"] != registry_sha256(
        alpha_registry
    ) or history[2]["prior_amendment_history_sha256"] != amendment_history_sha256(
        history[:2]
    ):
        raise PolicyImprovementSchemaError(
            "Final selection does not bind the audited alpha registry."
        )
    identity = publish_test_open(
        owner_root=arguments.evidence_root,
        protocol=protocol,
        registry=registry,
        amendment_history=history,
        runtime_authorization=authorization,
        base_configs=base_configs,
    )
    print(canonical_json_bytes(identity).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
