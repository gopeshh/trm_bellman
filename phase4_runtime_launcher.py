#!/usr/bin/env fbpython
"""Authenticate and execute one Phase 4 publication PAR before imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from confirmatory_runtime_launcher import (
    ConfirmatoryRuntimeError,
    launch_verified_runtime,
    validate_archive_layout,
    validate_confirmatory_archive_sources,
    validate_runtime_archive,
)
from phase4_runtime_profile import (
    assert_phase4_archive_matches_profile,
    authorize_phase4_source_profile,
    authorize_phase4_training_source,
    PHASE4_AUDIT_SOURCE_PROFILE,
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    PHASE4_FIGURE_SOURCE_PROFILE,
    Phase4RuntimeProfileError,
    POLICY_DATASET_BUILDER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
    PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH,
)


PHASE4_RUNTIME_ROLE_ENV = "UPI_TRM_PHASE4_RUNTIME_ROLE"
PHASE4_SOURCE_COMMIT_ENV = "UPI_TRM_PHASE4_SOURCE_COMMIT"
PHASE4_SOURCE_MANIFEST_SHA256_ENV = "UPI_TRM_PHASE4_SOURCE_MANIFEST_SHA256"
POLICY_SMOKE_LAUNCHER_SHA256_ENV = "UPI_TRM_POLICY_SMOKE_LAUNCHER_SHA256"
POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV = "UPI_TRM_POLICY_SMOKE_RUNTIME_AUTHORIZATION"
POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV = (
    "UPI_TRM_POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256"
)
POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV = "UPI_TRM_POLICY_SMOKE_RUNTIME_PROFILE_SHA256"
POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV = (
    "UPI_TRM_POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256"
)
POLICY_PROTOCOL_SHA256_ENV = "UPI_TRM_POLICY_PROTOCOL_SHA256"
POLICY_SMOKE_PURPOSE = "policy-improvement-smoke"
POLICY_DATASET_BUILDER_LAUNCHER_SHA256_ENV = (
    "UPI_TRM_POLICY_DATASET_BUILDER_LAUNCHER_SHA256"
)
POLICY_DATASET_BUILDER_PURPOSE = "policy-dataset-builder"
POLICY_IMPROVEMENT_AUDIT_PURPOSE = "policy-improvement-audit"
POLICY_IMPROVEMENT_ANALYSIS_PURPOSE = "policy-improvement-analysis"
POLICY_IMPROVEMENT_FULL_PURPOSE = "policy-improvement-full"
POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE = "policy-improvement-throughput"
POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE = "policy-improvement-theory-bridge"
POLICY_IMPROVEMENT_FULL_LAUNCHER_SHA256_ENV = "UPI_TRM_POLICY_FULL_LAUNCHER_SHA256"
POLICY_IMPROVEMENT_THEORY_BRIDGE_LAUNCHER_SHA256_ENV = (
    "UPI_TRM_POLICY_THEORY_BRIDGE_LAUNCHER_SHA256"
)
POLICY_CONSUMER_LAUNCHER_SHA256_ENV = "UPI_TRM_POLICY_CONSUMER_LAUNCHER_SHA256"
POLICY_PRODUCER_SOURCE_MANIFEST_SHA256_ENV = (
    "UPI_TRM_POLICY_PRODUCER_SOURCE_MANIFEST_SHA256"
)
POLICY_PRODUCER_GIT_COMMIT_ENV = "UPI_TRM_POLICY_PRODUCER_GIT_COMMIT"
POLICY_CONSUMER_RUNTIME_AUTHORIZATION_ENV = (
    "UPI_TRM_POLICY_CONSUMER_RUNTIME_AUTHORIZATION"
)
POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256_ENV = (
    "UPI_TRM_POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256"
)

_PURPOSE_TO_PROFILE = {
    "phase4-evaluator": PHASE4_EVALUATOR_SOURCE_PROFILE,
    "phase4-audit": PHASE4_AUDIT_SOURCE_PROFILE,
    "phase4-figure": PHASE4_FIGURE_SOURCE_PROFILE,
    POLICY_DATASET_BUILDER_PURPOSE: POLICY_DATASET_BUILDER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_PURPOSE: (POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE),
    POLICY_IMPROVEMENT_ANALYSIS_PURPOSE: (POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE),
    POLICY_IMPROVEMENT_FULL_PURPOSE: POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE: (POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE),
    POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE: (
        POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE
    ),
}
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_POLICY_RUNTIME_ROLES = (
    "policy-improvement-training",
    "policy-improvement-evaluation",
    "policy-improvement-audit",
    "policy-improvement-analysis",
)
_POLICY_RUNTIME_ROLES_V2 = (
    *_POLICY_RUNTIME_ROLES,
    POLICY_IMPROVEMENT_FULL_PURPOSE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
)
_POLICY_V1_PROTOCOL_RELATIVE_PATH = "configs/policy_improvement_v1/protocol.json"
_POLICY_V1_REGISTRY_RELATIVE_PATH = "configs/policy_improvement_v1/registry.json"
_POLICY_V1_THEORY_AMENDMENT_RELATIVE_PATH = (
    "configs/policy_improvement_v1/amendments/theory_bridge_v1.json"
)
_POLICY_V2_PROTOCOL_RELATIVE_PATH = "configs/policy_improvement_v2/protocol.json"
_POLICY_V2_REGISTRY_RELATIVE_PATH = "configs/policy_improvement_v2/registry.json"
_POLICY_V2_THEORY_AMENDMENT_RELATIVE_PATH = (
    "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
)


def _policy_authorization_roles(value: Mapping[str, object]) -> tuple[str, ...]:
    schema = (value.get("schema_name"), value.get("schema_version"))
    if schema == ("policy_improvement_runtime_authorization_v1", 1):
        return _POLICY_RUNTIME_ROLES
    if schema == ("policy_improvement_runtime_authorization_v2", 2):
        return _POLICY_RUNTIME_ROLES_V2
    if schema == ("policy_improvement_runtime_authorization_v3", 3):
        return _POLICY_RUNTIME_ROLES_V2
    raise ConfirmatoryRuntimeError("Unsupported runtime authorization schema.")


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ConfirmatoryRuntimeError(
                "Runtime authorization contains a duplicate JSON key."
            )
        value[key] = item
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization is not canonical JSON."
        ) from exc


def _canonical_policy_document(
    project_root: str,
    relative_path: str,
) -> tuple[dict[str, object], str]:
    path = Path(project_root) / relative_path
    payload = _stable_regular_file(path)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ConfirmatoryRuntimeError(f"Policy protocol contains {constant!r}.")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmatoryRuntimeError(
            "Policy protocol is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise ConfirmatoryRuntimeError("Policy registration must be one JSON object.")
    return value, hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_policy_protocol_sha256(
    project_root: str,
    *,
    v2: bool = False,
) -> str:
    _, digest = _canonical_policy_document(
        project_root,
        (
            _POLICY_V2_PROTOCOL_RELATIVE_PATH
            if v2
            else _POLICY_V1_PROTOCOL_RELATIVE_PATH
        ),
    )
    return digest


def _runtime_authorization_uses_v2_protocol(path_value: str) -> bool:
    payload = _stable_regular_file(Path(path_value))
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ConfirmatoryRuntimeError(
                    f"Runtime authorization contains {constant!r}."
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise ConfirmatoryRuntimeError("Runtime authorization must be one object.")
    schema = (value.get("schema_name"), value.get("schema_version"))
    if schema in {
        ("policy_improvement_runtime_authorization_v1", 1),
        ("policy_improvement_runtime_authorization_v2", 2),
    }:
        return False
    if schema == ("policy_improvement_runtime_authorization_v3", 3):
        return True
    raise ConfirmatoryRuntimeError("Unsupported runtime authorization schema.")


def _validate_v3_policy_registration(
    value: Mapping[str, object],
    *,
    project_root: str,
    protocol_sha256: str,
) -> None:
    protocol, registered_protocol_sha256 = _canonical_policy_document(
        project_root,
        _POLICY_V2_PROTOCOL_RELATIVE_PATH,
    )
    registry, registry_sha256 = _canonical_policy_document(
        project_root,
        _POLICY_V2_REGISTRY_RELATIVE_PATH,
    )
    amendment, amendment_sha256 = _canonical_policy_document(
        project_root,
        _POLICY_V2_THEORY_AMENDMENT_RELATIVE_PATH,
    )
    protocol_identity = value.get("protocol")
    registry_identity = value.get("registry")
    amendments = value.get("amendments")
    if (
        registered_protocol_sha256 != protocol_sha256
        or not isinstance(protocol_identity, dict)
        or set(protocol_identity)
        != {"schema_name", "schema_version", "protocol_id", "sha256"}
        or protocol_identity
        != {
            "schema_name": protocol.get("schema_name"),
            "schema_version": protocol.get("schema_version"),
            "protocol_id": protocol.get("protocol_id"),
            "sha256": registered_protocol_sha256,
        }
        or value.get("protocol_sha256") != registered_protocol_sha256
        or not isinstance(registry_identity, dict)
        or set(registry_identity) != {"schema_name", "schema_version", "sha256"}
        or registry_identity
        != {
            "schema_name": registry.get("schema_name"),
            "schema_version": registry.get("registry_schema_version"),
            "sha256": registry_sha256,
        }
        or not isinstance(amendments, list)
        or amendments
        != [
            {
                "schema_name": amendment.get("schema_name"),
                "schema_version": amendment.get("schema_version"),
                "amendment_id": amendment.get("amendment_id"),
                "sha256": amendment_sha256,
            }
        ]
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime authorization v3 does not bind the exact v2 protocol, "
            "registry, and amendment."
        )


def _stable_regular_file(path: Path) -> bytes:
    if not path.is_absolute() or ".." in path.parts:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization path must be absolute and canonical."
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization cannot be opened safely."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ConfirmatoryRuntimeError(
                "Runtime authorization must be a singly linked regular file."
            )
        chunks: list[bytes] = []
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ConfirmatoryRuntimeError(
                "Runtime authorization changed while it was authenticated."
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_policy_runtime_authorization(
    path_value: str,
    expected_sha256: str,
    *,
    runtime_sha256: str,
    source_git_commit: str,
    source_manifest_sha256: str,
    launcher_sha256: str,
    protocol_sha256: str,
    policy_project_root: str | None = None,
) -> tuple[str, str, str]:
    """Authenticate the external Stage 0 freeze before behavior imports."""

    if _LOWER_SHA256.fullmatch(expected_sha256) is None:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization SHA-256 must be lowercase hexadecimal."
        )
    payload = _stable_regular_file(Path(path_value))
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ConfirmatoryRuntimeError(
                    f"Runtime authorization contains {constant!r}."
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise ConfirmatoryRuntimeError("Runtime authorization must be one object.")
    common_fields = {
        "schema_name",
        "schema_version",
        "authorization_id",
        "created_at_utc",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "launcher_sha256",
        "roles",
    }
    is_v3 = (value.get("schema_name"), value.get("schema_version")) == (
        "policy_improvement_runtime_authorization_v3",
        3,
    )
    expected_fields = common_fields | (
        {"protocol_sha256", "protocol", "registry", "amendments"}
        if is_v3
        else {"protocol_sha256"}
    )
    if set(value) != expected_fields:
        raise ConfirmatoryRuntimeError("Runtime authorization field inventory differs.")
    role_names = _policy_authorization_roles(value)
    if is_v3:
        if policy_project_root is None:
            raise ConfirmatoryRuntimeError(
                "Runtime authorization v3 lacks its canonical project root."
            )
        _validate_v3_policy_registration(
            value,
            project_root=policy_project_root,
            protocol_sha256=protocol_sha256,
        )
    if (
        not isinstance(value["authorization_id"], str)
        or not value["authorization_id"]
        or not isinstance(value["created_at_utc"], str)
        or re.fullmatch(
            r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            value["created_at_utc"],
        )
        is None
        or not isinstance(value["protocol_sha256"], str)
        or _LOWER_SHA256.fullmatch(value["protocol_sha256"]) is None
        or value["protocol_sha256"] != protocol_sha256
        or value["producer_git_commit"] != source_git_commit
        or value["producer_source_manifest_sha256"] != source_manifest_sha256
        or value["launcher_sha256"] != launcher_sha256
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime authorization does not authorize this producer, protocol, "
            "or launcher."
        )
    roles = value["roles"]
    if not isinstance(roles, list) or len(roles) != len(role_names):
        raise ConfirmatoryRuntimeError("Runtime authorization role inventory differs.")
    role_fields = {
        "role",
        "source_git_commit",
        "runtime_sha256",
        "runtime_profile_sha256",
        "selected_source_manifest_sha256",
    }
    for index, role_name in enumerate(role_names):
        role = roles[index]
        if (
            not isinstance(role, dict)
            or set(role) != role_fields
            or role["role"] != role_name
            or not isinstance(role["source_git_commit"], str)
            or _LOWER_COMMIT.fullmatch(role["source_git_commit"]) is None
            or any(
                not isinstance(role[field], str)
                or _LOWER_SHA256.fullmatch(role[field]) is None
                for field in (
                    "runtime_sha256",
                    "runtime_profile_sha256",
                    "selected_source_manifest_sha256",
                )
            )
            or role["runtime_profile_sha256"] != role["selected_source_manifest_sha256"]
        ):
            raise ConfirmatoryRuntimeError(
                "Runtime authorization contains an invalid role."
            )
    if (value["schema_name"], value["schema_version"]) == (
        "policy_improvement_runtime_authorization_v2",
        2,
    ):
        checked_roles = {
            role_name: roles[index] for index, role_name in enumerate(role_names)
        }
        for semantic_role, launcher_role in (
            ("policy-improvement-training", POLICY_IMPROVEMENT_FULL_PURPOSE),
            (
                "policy-improvement-evaluation",
                POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
            ),
        ):
            if any(
                checked_roles[semantic_role][field]
                != checked_roles[launcher_role][field]
                for field in (
                    "source_git_commit",
                    "runtime_sha256",
                    "runtime_profile_sha256",
                    "selected_source_manifest_sha256",
                )
            ):
                raise ConfirmatoryRuntimeError(
                    "Runtime authorization semantic and launcher roles differ."
                )
        if (
            checked_roles[POLICY_IMPROVEMENT_FULL_PURPOSE]["source_git_commit"]
            != value["producer_git_commit"]
        ):
            raise ConfirmatoryRuntimeError(
                "Runtime authorization full source differs from the producer."
            )
    training_role = roles[0]
    evaluation_role = roles[1]
    assert isinstance(training_role, dict)
    assert isinstance(evaluation_role, dict)
    if (
        training_role["source_git_commit"] != source_git_commit
        or training_role["runtime_sha256"] != runtime_sha256
        or training_role["runtime_profile_sha256"] != source_manifest_sha256
        or evaluation_role["source_git_commit"] != source_git_commit
        or evaluation_role["runtime_sha256"] != runtime_sha256
        or evaluation_role["runtime_profile_sha256"]
        != training_role["runtime_profile_sha256"]
        or evaluation_role["selected_source_manifest_sha256"]
        != training_role["selected_source_manifest_sha256"]
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime authorization does not authorize this Stage 0 training and "
            "evaluation artifact."
        )
    canonical = _canonical_json_bytes(value)
    if hashlib.sha256(canonical).hexdigest() != expected_sha256:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization digest differs from its canonical document."
        )
    return (
        canonical.decode("ascii"),
        str(training_role["runtime_profile_sha256"]),
        str(training_role["selected_source_manifest_sha256"]),
    )


def _load_policy_consumer_runtime_authorization(
    path_value: str,
    expected_sha256: str,
    *,
    purpose: str,
    runtime_sha256: str,
    source_git_commit: str,
    source_manifest_sha256: str,
    launcher_sha256: str,
    producer_git_commit: str,
    producer_source_manifest_sha256: str,
    protocol_sha256: str,
    policy_project_root: str | None = None,
) -> str:
    """Authenticate the externally frozen authorization for an evidence consumer."""

    if _LOWER_SHA256.fullmatch(expected_sha256) is None:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization SHA-256 must be lowercase hexadecimal."
        )
    payload = _stable_regular_file(Path(path_value))
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ConfirmatoryRuntimeError(
                    f"Runtime authorization contains {constant!r}."
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise ConfirmatoryRuntimeError("Runtime authorization must be one object.")
    common_fields = {
        "schema_name",
        "schema_version",
        "authorization_id",
        "created_at_utc",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "launcher_sha256",
        "roles",
    }
    is_v3 = (value.get("schema_name"), value.get("schema_version")) == (
        "policy_improvement_runtime_authorization_v3",
        3,
    )
    expected_fields = common_fields | (
        {"protocol_sha256", "protocol", "registry", "amendments"}
        if is_v3
        else {"protocol_sha256"}
    )
    if set(value) != expected_fields:
        raise ConfirmatoryRuntimeError("Runtime authorization field inventory differs.")
    role_names = _policy_authorization_roles(value)
    if (
        purpose
        in {
            POLICY_IMPROVEMENT_FULL_PURPOSE,
            POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE,
            POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
        }
        and role_names != _POLICY_RUNTIME_ROLES_V2
    ):
        raise ConfirmatoryRuntimeError(
            "Full and theory-bridge execution require runtime authorization v2 "
            "or v3."
        )
    if is_v3:
        if policy_project_root is None:
            raise ConfirmatoryRuntimeError(
                "Runtime authorization v3 lacks its canonical project root."
            )
        _validate_v3_policy_registration(
            value,
            project_root=policy_project_root,
            protocol_sha256=protocol_sha256,
        )
    if (
        not isinstance(value["authorization_id"], str)
        or not value["authorization_id"]
        or not isinstance(value["created_at_utc"], str)
        or re.fullmatch(
            r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            value["created_at_utc"],
        )
        is None
        or not isinstance(value["protocol_sha256"], str)
        or _LOWER_SHA256.fullmatch(value["protocol_sha256"]) is None
        or value["protocol_sha256"] != protocol_sha256
        or value["producer_git_commit"] != producer_git_commit
        or value["producer_source_manifest_sha256"] != producer_source_manifest_sha256
        or value["launcher_sha256"] != launcher_sha256
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime authorization does not authorize this producer, protocol, "
            "or launcher."
        )
    roles = value["roles"]
    if not isinstance(roles, list) or len(roles) != len(role_names):
        raise ConfirmatoryRuntimeError("Runtime authorization role inventory differs.")
    role_fields = {
        "role",
        "source_git_commit",
        "runtime_sha256",
        "runtime_profile_sha256",
        "selected_source_manifest_sha256",
    }
    checked_roles: dict[str, dict[str, object]] = {}
    for index, role_name in enumerate(role_names):
        role = roles[index]
        if (
            not isinstance(role, dict)
            or set(role) != role_fields
            or role["role"] != role_name
            or not isinstance(role["source_git_commit"], str)
            or _LOWER_COMMIT.fullmatch(role["source_git_commit"]) is None
            or any(
                not isinstance(role[field], str)
                or _LOWER_SHA256.fullmatch(role[field]) is None
                for field in (
                    "runtime_sha256",
                    "runtime_profile_sha256",
                    "selected_source_manifest_sha256",
                )
            )
            or role["runtime_profile_sha256"] != role["selected_source_manifest_sha256"]
        ):
            raise ConfirmatoryRuntimeError(
                "Runtime authorization contains an invalid role."
            )
        checked_roles[role_name] = role
    if checked_roles["policy-improvement-training"]["source_git_commit"] != (
        producer_git_commit
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime authorization training source differs from the producer."
        )
    expected_role = {
        POLICY_IMPROVEMENT_AUDIT_PURPOSE: "policy-improvement-audit",
        POLICY_IMPROVEMENT_ANALYSIS_PURPOSE: "policy-improvement-analysis",
        POLICY_IMPROVEMENT_FULL_PURPOSE: POLICY_IMPROVEMENT_FULL_PURPOSE,
        POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE: POLICY_IMPROVEMENT_FULL_PURPOSE,
        POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE: (
            POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE
        ),
    }.get(purpose)
    if expected_role is None:
        raise ConfirmatoryRuntimeError("Unsupported policy consumer purpose.")
    active = checked_roles[expected_role]
    if (
        active["runtime_sha256"] != runtime_sha256
        or active["source_git_commit"] != source_git_commit
        or active["runtime_profile_sha256"] != source_manifest_sha256
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime authorization does not authorize this consumer artifact."
        )
    if (value["schema_name"], value["schema_version"]) == (
        "policy_improvement_runtime_authorization_v2",
        2,
    ):
        for semantic_role, launcher_role in (
            ("policy-improvement-training", POLICY_IMPROVEMENT_FULL_PURPOSE),
            (
                "policy-improvement-evaluation",
                POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
            ),
        ):
            if any(
                checked_roles[semantic_role][field]
                != checked_roles[launcher_role][field]
                for field in (
                    "source_git_commit",
                    "runtime_sha256",
                    "runtime_profile_sha256",
                    "selected_source_manifest_sha256",
                )
            ):
                raise ConfirmatoryRuntimeError(
                    "Runtime authorization semantic and launcher roles differ."
                )
        if (
            checked_roles[POLICY_IMPROVEMENT_FULL_PURPOSE]["source_git_commit"]
            != producer_git_commit
        ):
            raise ConfirmatoryRuntimeError(
                "Runtime authorization full source differs from the producer."
            )
    canonical = _canonical_json_bytes(value)
    if hashlib.sha256(canonical).hexdigest() != expected_sha256:
        raise ConfirmatoryRuntimeError(
            "Runtime authorization digest differs from its canonical document."
        )
    return canonical.decode("ascii")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and execute a Phase 4 publication PAR."
    )
    parser.add_argument(
        "--purpose",
        required=True,
        choices=["phase4-training", POLICY_SMOKE_PURPOSE, *_PURPOSE_TO_PROFILE],
    )
    parser.add_argument("--runtime-archive", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--source-project-root", required=True)
    parser.add_argument("--expected-source-git-commit", required=True)
    parser.add_argument("--producer-source-project-root")
    parser.add_argument("--expected-producer-git-commit")
    parser.add_argument("--runtime-authorization")
    parser.add_argument("--expected-runtime-authorization-sha256")
    parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _training_archive_validator(
    expected_manifest_bytes: bytes,
):
    def validate(archive: ZipFile) -> None:
        try:
            validate_confirmatory_archive_sources(archive)
            if archive.read(PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH) != (
                expected_manifest_bytes
            ):
                raise ConfirmatoryRuntimeError(
                    "Training runtime manifest differs from the authorized checkout."
                )
        except (BadZipFile, KeyError, OSError, RuntimeError) as exc:
            if isinstance(exc, ConfirmatoryRuntimeError):
                raise
            raise ConfirmatoryRuntimeError(
                "Training runtime source profile cannot be authenticated."
            ) from exc

    return validate


def _consumer_archive_validator(authorized_profile):
    def validate(archive: ZipFile) -> None:
        try:
            validate_archive_layout(archive)
            assert_phase4_archive_matches_profile(archive, authorized_profile)
        except (BadZipFile, OSError, RuntimeError) as exc:
            if isinstance(exc, ConfirmatoryRuntimeError):
                raise
            raise ConfirmatoryRuntimeError(
                "Phase 4 runtime source profile cannot be authenticated."
            ) from exc

    return validate


def _normalize_child_args(
    purpose: str,
    source_project_root: str,
    runtime_args: Sequence[str],
    *,
    policy_project_root: str | None = None,
    runtime_authorization_sha256: str | None = None,
    policy_protocol_v2: bool = False,
) -> list[str]:
    arguments = list(runtime_args)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]

    def contains_option(name: str) -> bool:
        return any(
            argument == name or argument.startswith(f"{name}=")
            for argument in arguments
        )

    if contains_option("--confirmatory"):
        raise ConfirmatoryRuntimeError(
            "Phase 4 publication runtimes cannot use confirmatory mode."
        )
    if purpose in {
        POLICY_IMPROVEMENT_AUDIT_PURPOSE,
        POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
    }:
        protected = {
            "--policy-improvement-consumer-entrypoint",
            "--audit-runtime-sha256",
            "--audit-runtime-profile-sha256",
            "--audit-source-git-commit",
            "--analysis-runtime-sha256",
            "--analysis-runtime-profile-sha256",
            "--analysis-source-git-commit",
            "--launcher-sha256",
            "--producer-git-commit",
            "--producer-source-manifest-sha256",
            "--runtime-authorization",
            "--runtime-authorization-json",
            "--runtime-authorization-sha256",
            "--source-project-root",
            "--expected-source-git-commit",
            "--producer-source-project-root",
            "--expected-producer-git-commit",
            "--runtime-archive",
            "--expected-runtime-sha256",
        }
        supplied_option_names = {
            argument.partition("=")[0]
            for argument in arguments
            if argument.startswith("--")
        }
        if any(
            protected_option.startswith(supplied_name)
            for supplied_name in supplied_option_names
            for protected_option in protected
        ):
            raise ConfirmatoryRuntimeError(
                "Policy consumer arguments contain a launcher-owned option."
            )
        if arguments == ["--help"]:
            return ["--policy-improvement-consumer-entrypoint", "--help"]
        return ["--policy-improvement-consumer-entrypoint", *arguments]
    if purpose == POLICY_DATASET_BUILDER_PURPOSE:
        if arguments == ["--help"]:
            return ["--policy-dataset-builder-entrypoint", "--help"]
        if len(arguments) != 5 or arguments[0] not in {"build", "verify"}:
            raise ConfirmatoryRuntimeError(
                "Dataset-builder arguments must select exactly one build or "
                "verify operation."
            )
        expected_target_option = (
            "--output-root" if arguments[0] == "build" else "--root"
        )
        if (
            arguments[1] != "--owner-root"
            or arguments[3] != expected_target_option
            or not arguments[2]
            or not arguments[4]
            or arguments[2].startswith("--")
            or arguments[4].startswith("--")
            or not Path(arguments[2]).is_absolute()
            or not Path(arguments[4]).is_absolute()
        ):
            raise ConfirmatoryRuntimeError(
                "Dataset-builder arguments contain a protected or unsupported "
                "child option."
            )
        return ["--policy-dataset-builder-entrypoint", *arguments]
    if purpose == POLICY_SMOKE_PURPOSE:
        allowed_value_options = {
            "--dataset-root",
            "--evidence-root",
            "--policy-improvement-protocol",
            "--policy-improvement-row-id",
            "--policy-improvement-smoke-segment",
            "--train-manifest-sha256",
        }
        if not policy_protocol_v2:
            allowed_value_options.add("--validation-manifest-sha256")
        if arguments == ["--help"]:
            return [
                "--policy-improvement-smoke-entrypoint",
                "--source-project-root",
                source_project_root,
                "--help",
            ]
        index = 0
        seen: set[str] = set()
        while index < len(arguments):
            argument = arguments[index]
            matching = next(
                (
                    option
                    for option in allowed_value_options
                    if argument.startswith(f"{option}=")
                ),
                None,
            )
            if matching is not None:
                if matching in seen or argument == f"{matching}=":
                    raise ConfirmatoryRuntimeError(
                        "Policy-improvement smoke arguments contain a duplicate "
                        "or empty child option."
                    )
                seen.add(matching)
                index += 1
                continue
            if argument not in allowed_value_options or index + 1 >= len(arguments):
                raise ConfirmatoryRuntimeError(
                    "Policy-improvement smoke arguments contain a protected or "
                    "unsupported child option."
                )
            if argument in seen or arguments[index + 1].startswith("--"):
                raise ConfirmatoryRuntimeError(
                    "Policy-improvement smoke arguments contain a duplicate or "
                    "missing child value."
                )
            seen.add(argument)
            index += 2
        if seen != allowed_value_options:
            raise ConfirmatoryRuntimeError(
                "Policy-improvement smoke arguments omit a required child option."
            )
        return [
            "--policy-improvement-smoke-entrypoint",
            "--source-project-root",
            source_project_root,
            *arguments,
        ]
    if purpose == POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE:
        if not policy_protocol_v2:
            raise ConfirmatoryRuntimeError(
                "Throughput calibration accepts only protocol v2 authorization."
            )
        if (
            not isinstance(runtime_authorization_sha256, str)
            or _LOWER_SHA256.fullmatch(runtime_authorization_sha256) is None
        ):
            raise ConfirmatoryRuntimeError(
                "Throughput calibration lacks its authorization digest."
            )
        project_root = policy_project_root or source_project_root
        if arguments == ["--help"]:
            return [
                "--policy-improvement-throughput-entrypoint",
                "--project-root",
                project_root,
                "--protocol",
                str(Path(project_root) / _POLICY_V2_PROTOCOL_RELATIVE_PATH),
                "--registry",
                str(Path(project_root) / _POLICY_V2_REGISTRY_RELATIVE_PATH),
                "--runtime-authorization-sha256",
                runtime_authorization_sha256,
                "--help",
            ]
        throughput_values: dict[str, str] = {}
        authorize_4096 = False
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            name, separator, inline_value = argument.partition("=")
            if name == "--authorize-4096-tier":
                if separator or authorize_4096:
                    raise ConfirmatoryRuntimeError(
                        "Throughput 4096 authorization is duplicated or valued."
                    )
                authorize_4096 = True
                index += 1
                continue
            if name not in {
                "--dataset-root",
                "--maximum-total-predicted-4096-seconds",
            }:
                raise ConfirmatoryRuntimeError(
                    "Throughput arguments contain a protected or unsupported option."
                )
            if name in throughput_values:
                raise ConfirmatoryRuntimeError(
                    "Throughput arguments repeat a singleton option."
                )
            if separator:
                value = inline_value
                index += 1
            else:
                if index + 1 >= len(arguments):
                    raise ConfirmatoryRuntimeError(
                        "Throughput option is missing its value."
                    )
                value = arguments[index + 1]
                index += 2
            if not value or value.startswith("--"):
                raise ConfirmatoryRuntimeError(
                    "Throughput option has an empty or missing value."
                )
            throughput_values[name] = value
        dataset_root = throughput_values.get("--dataset-root")
        if dataset_root is None:
            raise ConfirmatoryRuntimeError(
                "Throughput calibration requires dataset root."
            )
        dataset_path = Path(dataset_root)
        if not dataset_path.is_absolute() or ".." in dataset_path.parts:
            raise ConfirmatoryRuntimeError(
                "Throughput dataset root must be absolute and canonical."
            )
        ceiling = throughput_values.get("--maximum-total-predicted-4096-seconds")
        if authorize_4096 != (ceiling is not None):
            raise ConfirmatoryRuntimeError(
                "Throughput 4096 execution requires authorization and a ceiling."
            )
        return [
            "--policy-improvement-throughput-entrypoint",
            "--project-root",
            project_root,
            "--protocol",
            str(Path(project_root) / _POLICY_V2_PROTOCOL_RELATIVE_PATH),
            "--registry",
            str(Path(project_root) / _POLICY_V2_REGISTRY_RELATIVE_PATH),
            "--dataset-root",
            dataset_root,
            "--runtime-authorization-sha256",
            runtime_authorization_sha256,
            *(
                [
                    "--authorize-4096-tier",
                    "--maximum-total-predicted-4096-seconds",
                    ceiling,
                ]
                if authorize_4096 and ceiling is not None
                else []
            ),
        ]
    if purpose in {
        POLICY_IMPROVEMENT_FULL_PURPOSE,
        POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
    }:
        project_root = policy_project_root or source_project_root
        protocol_path = str(
            Path(project_root)
            / (
                _POLICY_V2_PROTOCOL_RELATIVE_PATH
                if policy_protocol_v2
                else _POLICY_V1_PROTOCOL_RELATIVE_PATH
            )
        )
        registry_path = str(
            Path(project_root)
            / (
                _POLICY_V2_REGISTRY_RELATIVE_PATH
                if policy_protocol_v2
                else _POLICY_V1_REGISTRY_RELATIVE_PATH
            )
        )
        theory_amendment_path = str(
            Path(project_root)
            / (
                _POLICY_V2_THEORY_AMENDMENT_RELATIVE_PATH
                if policy_protocol_v2
                else _POLICY_V1_THEORY_AMENDMENT_RELATIVE_PATH
            )
        )
        if purpose == POLICY_IMPROVEMENT_FULL_PURPOSE and (
            not isinstance(runtime_authorization_sha256, str)
            or _LOWER_SHA256.fullmatch(runtime_authorization_sha256) is None
        ):
            raise ConfirmatoryRuntimeError(
                "Full runtime lacks its launcher-owned authorization digest."
            )
        if arguments == ["--help"]:
            if purpose == POLICY_IMPROVEMENT_FULL_PURPOSE:
                assert isinstance(runtime_authorization_sha256, str)
                return [
                    "--project-root",
                    project_root,
                    "--protocol",
                    protocol_path,
                    "--registry",
                    registry_path,
                    "--amendment",
                    theory_amendment_path,
                    "--runtime-authorization-sha256",
                    runtime_authorization_sha256,
                    "--help",
                ]
            return [
                "--policy-improvement-theory-bridge-entrypoint",
                "--theory-amendment",
                theory_amendment_path,
                "--amendment",
                theory_amendment_path,
                "--project-root",
                project_root,
                "--protocol",
                protocol_path,
                "--registry",
                registry_path,
                "--help",
            ]
        value_options = {
            "--evidence-root",
            "--dataset-root",
            "--row-id",
            "--amendment",
        }
        if purpose == POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE:
            value_options.update({"--request", "--checkpoint"})
        flag_options = (
            {"--print-contract"}
            if purpose == POLICY_IMPROVEMENT_FULL_PURPOSE
            else set()
        )
        required_options = {
            "--evidence-root",
            "--dataset-root",
            "--row-id",
        }
        if purpose == POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE:
            required_options.update({"--request", "--checkpoint"})
        values: dict[str, list[str]] = {option: [] for option in value_options}
        flags: set[str] = set()
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            name, separator, inline_value = argument.partition("=")
            if name in flag_options:
                if separator or name in flags:
                    raise ConfirmatoryRuntimeError(
                        "Policy runtime arguments contain a duplicate or valued flag."
                    )
                flags.add(name)
                index += 1
                continue
            if name not in value_options:
                raise ConfirmatoryRuntimeError(
                    "Policy runtime arguments contain a protected or unsupported "
                    "child option."
                )
            if separator:
                value = inline_value
                index += 1
            else:
                if index + 1 >= len(arguments):
                    raise ConfirmatoryRuntimeError(
                        "Policy runtime child option is missing its value."
                    )
                value = arguments[index + 1]
                index += 2
            if not value or value.startswith("--"):
                raise ConfirmatoryRuntimeError(
                    "Policy runtime child option has an empty or missing value."
                )
            if name != "--amendment" and values[name]:
                raise ConfirmatoryRuntimeError(
                    "Policy runtime arguments repeat a singleton child option."
                )
            if name != "--row-id":
                supplied_path = Path(value)
                if not supplied_path.is_absolute() or ".." in supplied_path.parts:
                    raise ConfirmatoryRuntimeError(
                        "Policy runtime data and amendment paths must be absolute."
                    )
            values[name].append(value)
        if any(not values[option] for option in required_options):
            raise ConfirmatoryRuntimeError(
                "Policy runtime arguments omit a required safe child option."
            )
        if theory_amendment_path in values["--amendment"]:
            raise ConfirmatoryRuntimeError(
                "The launcher-owned theory amendment cannot be supplied again."
            )
        additional_amendments = [
            item
            for amendment in values["--amendment"]
            for item in ("--amendment", amendment)
        ]
        common = [
            "--project-root",
            project_root,
            "--protocol",
            protocol_path,
            "--registry",
            registry_path,
            "--amendment",
            theory_amendment_path,
            *additional_amendments,
            "--evidence-root",
            values["--evidence-root"][0],
            "--dataset-root",
            values["--dataset-root"][0],
            "--row-id",
            values["--row-id"][0],
        ]
        if purpose == POLICY_IMPROVEMENT_FULL_PURPOSE:
            assert runtime_authorization_sha256 is not None
            return [
                *common,
                "--runtime-authorization-sha256",
                runtime_authorization_sha256,
                *(["--print-contract"] if "--print-contract" in flags else []),
            ]
        return [
            "--policy-improvement-theory-bridge-entrypoint",
            "--request",
            values["--request"][0],
            "--theory-amendment",
            theory_amendment_path,
            "--amendment",
            theory_amendment_path,
            *additional_amendments,
            "--checkpoint",
            values["--checkpoint"][0],
            "--project-root",
            project_root,
            "--protocol",
            protocol_path,
            "--registry",
            registry_path,
            "--evidence-root",
            values["--evidence-root"][0],
            "--dataset-root",
            values["--dataset-root"][0],
            "--row-id",
            values["--row-id"][0],
        ]
    if purpose != "phase4-training":
        if contains_option("--phase4-publication"):
            raise ConfirmatoryRuntimeError(
                "Phase 4 consumer arguments contain a training-only flag."
            )
        return arguments
    if contains_option("--phase4-publication") or contains_option(
        "--producer-repo-root"
    ):
        raise ConfirmatoryRuntimeError(
            "Phase 4 training launcher owns its publication and producer flags."
        )
    return [
        "--phase4-publication",
        "--producer-repo-root",
        source_project_root,
        *arguments,
    ]


def _launcher_artifact_sha256() -> str:
    path = Path(sys.argv[0])
    try:
        requested = path.lstat()
        resolved = path.resolve(strict=True)
        status = resolved.lstat()
    except OSError as exc:
        raise ConfirmatoryRuntimeError(
            "Phase 4 launcher artifact cannot be inspected."
        ) from exc
    if stat.S_ISLNK(requested.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ConfirmatoryRuntimeError(
            "Phase 4 launcher artifact must be a regular non-symlink file."
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ConfirmatoryRuntimeError(
            "Phase 4 launcher artifact cannot be hashed."
        ) from exc
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        source_root = str(Path(arguments.source_project_root).resolve(strict=True))
        separate_producer_consumer = arguments.purpose in {
            POLICY_IMPROVEMENT_AUDIT_PURPOSE,
            POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
            POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
        }
        producer_options_present = (
            arguments.producer_source_project_root is not None,
            arguments.expected_producer_git_commit is not None,
        )
        if separate_producer_consumer != all(producer_options_present) or (
            not separate_producer_consumer and any(producer_options_present)
        ):
            raise ConfirmatoryRuntimeError(
                "Policy consumers require a separate complete producer-source "
                "authorization; other roles cannot accept one."
            )
        authorization_options_present = (
            arguments.runtime_authorization is not None,
            arguments.expected_runtime_authorization_sha256 is not None,
        )
        requires_runtime_authorization = arguments.purpose in {
            POLICY_SMOKE_PURPOSE,
            POLICY_IMPROVEMENT_AUDIT_PURPOSE,
            POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
            POLICY_IMPROVEMENT_FULL_PURPOSE,
            POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE,
            POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
        }
        if requires_runtime_authorization != all(authorization_options_present):
            raise ConfirmatoryRuntimeError(
                "Policy-improvement smoke and evidence consumers require one "
                "complete externally digested runtime authorization."
            )
        policy_protocol_v2 = (
            _runtime_authorization_uses_v2_protocol(arguments.runtime_authorization)
            if requires_runtime_authorization
            and arguments.runtime_authorization is not None
            else False
        )
        producer_source_root = (
            str(Path(arguments.producer_source_project_root).resolve(strict=True))
            if separate_producer_consumer
            else None
        )
        if separate_producer_consumer and producer_source_root == source_root:
            raise ConfirmatoryRuntimeError(
                "Policy evaluator and producer checkouts must be distinct paths."
            )
        child_args = _normalize_child_args(
            arguments.purpose,
            source_root,
            arguments.runtime_args,
            policy_project_root=(
                producer_source_root
                if arguments.purpose == POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE
                else source_root
            ),
            runtime_authorization_sha256=(
                arguments.expected_runtime_authorization_sha256
            ),
            policy_protocol_v2=policy_protocol_v2,
        )
        producer_authorized = None
        if arguments.purpose in {"phase4-training", POLICY_SMOKE_PURPOSE}:
            authorized = authorize_phase4_training_source(
                source_root,
                arguments.expected_source_git_commit,
            )
            role = (
                "training"
                if arguments.purpose == "phase4-training"
                else POLICY_SMOKE_PURPOSE
            )
            archive_validator = _training_archive_validator(authorized.manifest_bytes)
        else:
            profile = _PURPOSE_TO_PROFILE[arguments.purpose]
            authorized = authorize_phase4_source_profile(
                source_root,
                arguments.expected_source_git_commit,
                profile,
            )
            if separate_producer_consumer:
                assert producer_source_root is not None
                assert arguments.expected_producer_git_commit is not None
                producer_authorized = authorize_phase4_training_source(
                    producer_source_root,
                    arguments.expected_producer_git_commit,
                )
            else:
                producer_authorized = (
                    authorize_phase4_training_source(
                        source_root,
                        arguments.expected_source_git_commit,
                    )
                    if arguments.purpose
                    in {
                        POLICY_IMPROVEMENT_FULL_PURPOSE,
                        POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE,
                    }
                    else None
                )
            role = profile
            archive_validator = _consumer_archive_validator(authorized)
        policy_project_root = (
            producer_source_root if separate_producer_consumer else source_root
        )
        if policy_project_root is None:
            raise ConfirmatoryRuntimeError("Policy project root is unavailable.")
        policy_protocol_sha256 = (
            (
                _canonical_policy_protocol_sha256(policy_project_root, v2=True)
                if policy_protocol_v2
                else _canonical_policy_protocol_sha256(policy_project_root)
            )
            if requires_runtime_authorization
            else None
        )
        runtime = validate_runtime_archive(
            arguments.runtime_archive,
            arguments.expected_runtime_sha256,
            archive_validator=archive_validator,
        )
        attestation_environment = {
            PHASE4_RUNTIME_ROLE_ENV: role,
            PHASE4_SOURCE_COMMIT_ENV: authorized.git_commit,
            PHASE4_SOURCE_MANIFEST_SHA256_ENV: (authorized.source_manifest_sha256),
        }
        if arguments.purpose == POLICY_SMOKE_PURPOSE:
            launcher_sha256 = _launcher_artifact_sha256()
            assert arguments.runtime_authorization is not None
            assert arguments.expected_runtime_authorization_sha256 is not None
            assert policy_protocol_sha256 is not None
            (
                authorization_json,
                runtime_profile_sha256,
                selected_source_manifest_sha256,
            ) = _load_policy_runtime_authorization(
                arguments.runtime_authorization,
                arguments.expected_runtime_authorization_sha256,
                runtime_sha256=arguments.expected_runtime_sha256,
                source_git_commit=authorized.git_commit,
                source_manifest_sha256=authorized.source_manifest_sha256,
                launcher_sha256=launcher_sha256,
                protocol_sha256=policy_protocol_sha256,
                policy_project_root=source_root,
            )
            attestation_environment[POLICY_SMOKE_LAUNCHER_SHA256_ENV] = launcher_sha256
            attestation_environment[POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV] = (
                authorization_json
            )
            attestation_environment[POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV] = (
                arguments.expected_runtime_authorization_sha256
            )
            attestation_environment[POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV] = (
                runtime_profile_sha256
            )
            attestation_environment[
                POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV
            ] = selected_source_manifest_sha256
            attestation_environment[POLICY_PROTOCOL_SHA256_ENV] = policy_protocol_sha256
        if arguments.purpose in {
            POLICY_IMPROVEMENT_FULL_PURPOSE,
            POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE,
            POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
        }:
            assert producer_authorized is not None
            assert arguments.runtime_authorization is not None
            assert arguments.expected_runtime_authorization_sha256 is not None
            assert policy_protocol_sha256 is not None
            policy_launcher_sha256 = _launcher_artifact_sha256()
            authorization_json = _load_policy_consumer_runtime_authorization(
                arguments.runtime_authorization,
                arguments.expected_runtime_authorization_sha256,
                purpose=arguments.purpose,
                runtime_sha256=arguments.expected_runtime_sha256,
                source_git_commit=authorized.git_commit,
                source_manifest_sha256=authorized.source_manifest_sha256,
                launcher_sha256=policy_launcher_sha256,
                producer_git_commit=producer_authorized.git_commit,
                producer_source_manifest_sha256=(
                    producer_authorized.source_manifest_sha256
                ),
                protocol_sha256=policy_protocol_sha256,
                policy_project_root=(
                    producer_source_root
                    if producer_source_root is not None
                    else source_root
                ),
            )
            attestation_environment[POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV] = (
                authorization_json
            )
            attestation_environment[POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV] = (
                arguments.expected_runtime_authorization_sha256
            )
            attestation_environment[POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV] = (
                authorized.source_manifest_sha256
            )
            attestation_environment[
                POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV
            ] = authorized.source_manifest_sha256
            attestation_environment[POLICY_PROTOCOL_SHA256_ENV] = policy_protocol_sha256
            if arguments.purpose in {
                POLICY_IMPROVEMENT_FULL_PURPOSE,
                POLICY_IMPROVEMENT_THROUGHPUT_PURPOSE,
            }:
                attestation_environment[POLICY_IMPROVEMENT_FULL_LAUNCHER_SHA256_ENV] = (
                    policy_launcher_sha256
                )
            else:
                attestation_environment[
                    POLICY_IMPROVEMENT_THEORY_BRIDGE_LAUNCHER_SHA256_ENV
                ] = policy_launcher_sha256
        if arguments.purpose == POLICY_DATASET_BUILDER_PURPOSE:
            attestation_environment[POLICY_DATASET_BUILDER_LAUNCHER_SHA256_ENV] = (
                _launcher_artifact_sha256()
            )
        if arguments.purpose in {
            POLICY_IMPROVEMENT_AUDIT_PURPOSE,
            POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
        }:
            assert producer_authorized is not None
            assert arguments.runtime_authorization is not None
            assert arguments.expected_runtime_authorization_sha256 is not None
            assert policy_protocol_sha256 is not None
            consumer_launcher_sha256 = _launcher_artifact_sha256()
            authorization_json = _load_policy_consumer_runtime_authorization(
                arguments.runtime_authorization,
                arguments.expected_runtime_authorization_sha256,
                purpose=arguments.purpose,
                runtime_sha256=arguments.expected_runtime_sha256,
                source_git_commit=authorized.git_commit,
                source_manifest_sha256=authorized.source_manifest_sha256,
                launcher_sha256=consumer_launcher_sha256,
                producer_git_commit=producer_authorized.git_commit,
                producer_source_manifest_sha256=(
                    producer_authorized.source_manifest_sha256
                ),
                protocol_sha256=policy_protocol_sha256,
                policy_project_root=(
                    producer_source_root
                    if producer_source_root is not None
                    else source_root
                ),
            )
            attestation_environment[POLICY_CONSUMER_LAUNCHER_SHA256_ENV] = (
                consumer_launcher_sha256
            )
            attestation_environment[POLICY_PRODUCER_SOURCE_MANIFEST_SHA256_ENV] = (
                producer_authorized.source_manifest_sha256
            )
            attestation_environment[POLICY_PRODUCER_GIT_COMMIT_ENV] = (
                producer_authorized.git_commit
            )
            attestation_environment[POLICY_CONSUMER_RUNTIME_AUTHORIZATION_ENV] = (
                authorization_json
            )
            attestation_environment[
                POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256_ENV
            ] = arguments.expected_runtime_authorization_sha256
        return_code = launch_verified_runtime(
            runtime,
            child_args,
            required_argument=None,
            attestation_environment=attestation_environment,
        )
    except (
        ConfirmatoryRuntimeError,
        OSError,
        Phase4RuntimeProfileError,
    ) as exc:
        print(f"Phase 4 runtime rejected: {exc}", file=sys.stderr)
        return 2
    return 128 - return_code if return_code < 0 else return_code


if __name__ == "__main__":
    raise SystemExit(main())
