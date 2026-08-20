#!/usr/bin/env fbpython
"""Render the four registered authenticated Stage 0 two-process smoke rows.

The commands use the dedicated policy-improvement launcher purpose. They never
enter confirmatory mode and never accept a caller-selected resume checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.policy_improvement_registry import (
    generate_registry,
    load_registered_base_configs,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    validate_protocol,
    validate_runtime_authorization,
)


PLAN_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _absolute_path(value: str, *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise PolicyImprovementSchemaError(f"{name} must be an absolute path.")
    return path


def _sha256(value: str, *, name: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise PolicyImprovementSchemaError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def render_smoke_plan(
    protocol_value: object,
    *,
    protocol_path: str,
    launcher_path: str,
    training_runtime_path: str,
    training_runtime_sha256: str,
    source_project_root: str,
    expected_source_git_commit: str,
    dataset_root: str,
    train_manifest_sha256: str,
    validation_manifest_sha256: str | None,
    evidence_root: str,
    runtime_authorization_value: object,
    runtime_authorization_path: str,
    expected_runtime_authorization_sha256: str,
) -> dict[str, Any]:
    """Build four two-segment command plans without spawning a process."""

    protocol = validate_protocol(protocol_value)
    is_v2 = protocol.get("schema_name") == "policy_improvement_protocol_v2"
    protocol_file = _absolute_path(protocol_path, name="protocol_path")
    launcher = _absolute_path(launcher_path, name="launcher_path")
    runtime = _absolute_path(training_runtime_path, name="training_runtime_path")
    source_root = _absolute_path(source_project_root, name="source_project_root")
    populations_document: object | None = None
    if is_v2:
        from scripts.policy_improvement_populations import load_registered_populations

        populations_document = load_registered_populations(protocol, source_root)
    registry = generate_registry(
        protocol,
        base_configs=load_registered_base_configs(protocol, source_root),
        populations_value=populations_document,
    )
    dataset = _absolute_path(dataset_root, name="dataset_root")
    evidence = _absolute_path(evidence_root, name="evidence_root")
    authorization_path = _absolute_path(
        runtime_authorization_path, name="runtime_authorization_path"
    )
    authorization = validate_runtime_authorization(runtime_authorization_value)
    if (
        is_v2
        and [
            authorization.get("schema_name"),
            authorization.get("schema_version"),
        ]
        != protocol["document_schemas"]["runtime_authorization"]
    ):
        raise PolicyImprovementSchemaError(
            "Runtime authorization schema differs from protocol v2."
        )
    authorization_digest = _sha256(
        expected_runtime_authorization_sha256,
        name="expected_runtime_authorization_sha256",
    )
    if runtime_authorization_sha256(authorization) != authorization_digest:
        raise PolicyImprovementSchemaError(
            "Runtime authorization digest differs from its canonical document."
        )
    runtime_digest = _sha256(training_runtime_sha256, name="training_runtime_sha256")
    train_digest = _sha256(train_manifest_sha256, name="train_manifest_sha256")
    if is_v2:
        if validation_manifest_sha256 is not None:
            raise PolicyImprovementSchemaError(
                "Protocol v2 Stage 0 must not accept a validation manifest argument."
            )
        validation_digest = None
    else:
        if validation_manifest_sha256 is None:
            raise PolicyImprovementSchemaError(
                "Protocol v1 Stage 0 requires the validation manifest digest."
            )
        validation_digest = _sha256(
            validation_manifest_sha256, name="validation_manifest_sha256"
        )
    if _GIT_COMMIT.fullmatch(expected_source_git_commit) is None:
        raise PolicyImprovementSchemaError(
            "expected_source_git_commit must be 40 lowercase hexadecimal characters."
        )
    training_role = next(
        role
        for role in authorization["roles"]
        if role["role"] == "policy-improvement-training"
    )
    evaluation_role = next(
        role
        for role in authorization["roles"]
        if role["role"] == "policy-improvement-evaluation"
    )
    expected_training_profile = authorization["producer_source_manifest_sha256"]
    if (
        authorization["producer_git_commit"] != expected_source_git_commit
        or training_role["source_git_commit"] != expected_source_git_commit
        or training_role["runtime_sha256"] != runtime_digest
        or training_role["runtime_profile_sha256"] != expected_training_profile
        or evaluation_role["source_git_commit"] != training_role["source_git_commit"]
        or evaluation_role["runtime_sha256"] != training_role["runtime_sha256"]
        or evaluation_role["runtime_profile_sha256"]
        != training_role["runtime_profile_sha256"]
        or evaluation_role["selected_source_manifest_sha256"]
        != training_role["selected_source_manifest_sha256"]
        or authorization["protocol_sha256"]
        != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    ):
        raise PolicyImprovementSchemaError(
            "Smoke command inputs or its in-process evaluation role differ from "
            "the external runtime authorization."
        )

    smoke_budget = protocol["budgets"]["smoke"]
    first_budget, final_budget = smoke_budget["checkpoint_environment_interactions"]
    if [first_budget, final_budget] != [16, 32]:
        raise PolicyImprovementSchemaError(
            "The smoke renderer supports only the registered 16/32 split."
        )
    supplied_manifests = [("train", train_digest)]
    if validation_digest is not None:
        supplied_manifests.append(("validation", validation_digest))
    for split, supplied in supplied_manifests:
        registration = protocol["dataset"]["splits"][split]["manifest_sha256"]
        if registration != {"status": "available", "value": supplied}:
            raise PolicyImprovementSchemaError(
                f"The {split} manifest must be committed before rendering smoke commands."
            )
    test_registration = protocol["dataset"]["splits"]["test"]["manifest_sha256"]
    if (
        not isinstance(test_registration, dict)
        or test_registration.get("status") != "available"
    ):
        raise PolicyImprovementSchemaError(
            "The test manifest must be committed before any Stage 0 training."
        )
    rows: list[dict[str, Any]] = []
    smoke_rows = [row for row in registry["rows"] if row["phase"] == "stage0_smoke"]
    for row in smoke_rows:
        run_id = str(row["run_id"])
        method_id = str(row["method_id"])
        common_child = [
            "--policy-improvement-protocol",
            str(protocol_file),
            "--policy-improvement-row-id",
            run_id,
            "--dataset-root",
            str(dataset),
            "--train-manifest-sha256",
            train_digest,
            "--evidence-root",
            str(evidence),
        ]
        if validation_digest is not None:
            common_child.extend(["--validation-manifest-sha256", validation_digest])
        launcher_prefix = [
            str(launcher),
            "--purpose",
            "policy-improvement-smoke",
            "--runtime-archive",
            str(runtime),
            "--expected-runtime-sha256",
            runtime_digest,
            "--source-project-root",
            str(source_root),
            "--expected-source-git-commit",
            expected_source_git_commit,
            "--runtime-authorization",
            str(authorization_path),
            "--expected-runtime-authorization-sha256",
            authorization_digest,
            "--",
        ]
        prepare = [
            *launcher_prefix,
            *common_child,
            "--policy-improvement-smoke-segment",
            "prepare",
        ]
        resume = [
            *launcher_prefix,
            *common_child,
            "--policy-improvement-smoke-segment",
            "resume",
        ]
        rows.append(
            {
                "run_id": run_id,
                "method_id": method_id,
                "seed": row["seed"],
                "evaluation_split": row["evaluation_split"],
                **(
                    {"evaluation_population_id": row["evaluation_population"]}
                    if is_v2
                    else {}
                ),
                "executable": True,
                "prepare_generation": "env_000000016",
                "resume_generation": "env_000000032",
                "commands": {"prepare": prepare, "resume": resume},
            }
        )
    document: dict[str, Any] = {
        "schema_name": (
            "policy_improvement_smoke_plan_v2"
            if is_v2
            else "policy_improvement_smoke_plan_v1"
        ),
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "execution_authorized": True,
        "runtime_authorization_sha256": authorization_digest,
        "blocked_by": [],
        "rows": rows,
    }
    canonical_json_bytes(document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--training-runtime", required=True)
    parser.add_argument("--training-runtime-sha256", required=True)
    parser.add_argument("--source-project-root", required=True)
    parser.add_argument("--expected-source-git-commit", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--validation-manifest-sha256")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--runtime-authorization", required=True)
    parser.add_argument("--expected-runtime-authorization-sha256", required=True)
    arguments = parser.parse_args(argv)
    plan = render_smoke_plan(
        load_strict_json(arguments.protocol),
        protocol_path=arguments.protocol,
        launcher_path=arguments.launcher,
        training_runtime_path=arguments.training_runtime,
        training_runtime_sha256=arguments.training_runtime_sha256,
        source_project_root=arguments.source_project_root,
        expected_source_git_commit=arguments.expected_source_git_commit,
        dataset_root=arguments.dataset_root,
        train_manifest_sha256=arguments.train_manifest_sha256,
        validation_manifest_sha256=arguments.validation_manifest_sha256,
        evidence_root=arguments.evidence_root,
        runtime_authorization_value=load_strict_json(arguments.runtime_authorization),
        runtime_authorization_path=arguments.runtime_authorization,
        expected_runtime_authorization_sha256=(
            arguments.expected_runtime_authorization_sha256
        ),
    )
    print(canonical_json_bytes(plan).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
