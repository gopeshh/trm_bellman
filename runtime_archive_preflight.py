#!/usr/bin/env fbpython
"""Standard-library pre-import verification for sealed UPI-TRM runtimes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import zipfile
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path


VERIFIED_RUNTIME_PATH_ENV = "UPI_TRM_VERIFIED_RUNTIME_PATH"
VERIFIED_RUNTIME_SHA256_ENV = "UPI_TRM_VERIFIED_RUNTIME_SHA256"
VERIFIED_RUNTIME_FD_ENV = "UPI_TRM_VERIFIED_RUNTIME_FD"
PRIVATE_UNPACK_BASE_ENV = "UPI_TRM_PRIVATE_UNPACK_BASE"
PRIVATE_UNPACK_FD_ENV = "UPI_TRM_PRIVATE_UNPACK_FD"
PHASE4_RUNTIME_ROLE_ENV = "UPI_TRM_PHASE4_RUNTIME_ROLE"
PHASE4_SOURCE_COMMIT_ENV = "UPI_TRM_PHASE4_SOURCE_COMMIT"
PHASE4_SOURCE_MANIFEST_SHA256_ENV = "UPI_TRM_PHASE4_SOURCE_MANIFEST_SHA256"
POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV = "UPI_TRM_POLICY_SMOKE_RUNTIME_AUTHORIZATION"
POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV = (
    "UPI_TRM_POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256"
)
POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV = "UPI_TRM_POLICY_SMOKE_RUNTIME_PROFILE_SHA256"
POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV = (
    "UPI_TRM_POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256"
)
POLICY_PROTOCOL_SHA256_ENV = "UPI_TRM_POLICY_PROTOCOL_SHA256"
POLICY_SMOKE_ROLE = "policy-improvement-smoke"
POLICY_FULL_ROLE = "policy-improvement-full"
POLICY_THEORY_BRIDGE_ROLE = "policy-improvement-theory-bridge"
POLICY_AUTHORIZED_ROLES = frozenset(
    {POLICY_SMOKE_ROLE, POLICY_FULL_ROLE, POLICY_THEORY_BRIDGE_ROLE}
)
_POLICY_RUNTIME_ROLES_V1 = (
    "policy-improvement-training",
    "policy-improvement-evaluation",
    "policy-improvement-audit",
    "policy-improvement-analysis",
)
_POLICY_RUNTIME_ROLES_V2 = (
    *_POLICY_RUNTIME_ROLES_V1,
    POLICY_FULL_ROLE,
    POLICY_THEORY_BRIDGE_ROLE,
)

_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MEMFD_SEALING_AVAILABLE = all(
    hasattr(fcntl, name)
    for name in (
        "F_GET_SEALS",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_WRITE",
    )
)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 0)
_REQUIRED_MEMFD_SEALS = sum(
    getattr(fcntl, name, 0)
    for name in (
        "F_SEAL_WRITE",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
    )
)


@dataclass(frozen=True)
class RuntimePreflight:
    """Authenticated runtime identity available before behavior imports."""

    runtime_sha256: str
    private_unpack_descriptor: int
    runtime_descriptor: int
    phase4_role: str | None
    source_git_commit: str | None
    source_manifest_sha256: str | None
    policy_runtime_authorization_json: str | None = None
    policy_runtime_authorization_sha256: str | None = None
    policy_runtime_profile_sha256: str | None = None
    policy_selected_source_manifest_sha256: str | None = None
    policy_protocol_sha256: str | None = None


def _validate_policy_runtime_authorization(
    value: object,
    *,
    phase4_role: str,
    runtime_sha256: str,
    source_git_commit: str,
    source_manifest_sha256: str,
    protocol_sha256: str,
) -> None:
    """Validate semantic role binding before any behavior import."""

    if not isinstance(value, dict):
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    schema = (value.get("schema_name"), value.get("schema_version"))
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
    expected_fields = common_fields | (
        {"protocol_sha256", "protocol", "registry", "amendments"}
        if schema == ("policy_improvement_runtime_authorization_v3", 3)
        else {"protocol_sha256"}
    )
    if set(value) != expected_fields:
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    if schema == ("policy_improvement_runtime_authorization_v1", 1):
        role_names = _POLICY_RUNTIME_ROLES_V1
    elif schema == ("policy_improvement_runtime_authorization_v2", 2):
        role_names = _POLICY_RUNTIME_ROLES_V2
    elif schema == ("policy_improvement_runtime_authorization_v3", 3):
        role_names = _POLICY_RUNTIME_ROLES_V2
    else:
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    if phase4_role in {POLICY_FULL_ROLE, POLICY_THEORY_BRIDGE_ROLE} and (
        role_names != _POLICY_RUNTIME_ROLES_V2
    ):
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
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
        or not isinstance(value["producer_git_commit"], str)
        or _LOWER_COMMIT.fullmatch(value["producer_git_commit"]) is None
        or not isinstance(value["producer_source_manifest_sha256"], str)
        or _LOWER_SHA256.fullmatch(value["producer_source_manifest_sha256"]) is None
        or not isinstance(value["launcher_sha256"], str)
        or _LOWER_SHA256.fullmatch(value["launcher_sha256"]) is None
    ):
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    if schema == ("policy_improvement_runtime_authorization_v3", 3):
        protocol = value["protocol"]
        registry = value["registry"]
        amendments = value["amendments"]
        if (
            not isinstance(protocol, dict)
            or protocol
            != {
                "schema_name": "policy_improvement_protocol_v2",
                "schema_version": 2,
                "protocol_id": "policy-improvement-v2-20260818",
                "sha256": protocol_sha256,
            }
            or not isinstance(registry, dict)
            or set(registry) != {"schema_name", "schema_version", "sha256"}
            or registry.get("schema_name") != "policy_improvement_registry_v2"
            or registry.get("schema_version") != 1
            or not isinstance(registry.get("sha256"), str)
            or _LOWER_SHA256.fullmatch(str(registry.get("sha256"))) is None
            or not isinstance(amendments, list)
            or len(amendments) != 1
            or not isinstance(amendments[0], dict)
            or set(amendments[0])
            != {"schema_name", "schema_version", "amendment_id", "sha256"}
            or amendments[0].get("schema_name")
            != "policy_improvement_theory_bridge_amendment_v2"
            or amendments[0].get("schema_version") != 1
            or amendments[0].get("amendment_id") != "pre-stage1-theory-bridge-v2"
            or not isinstance(amendments[0].get("sha256"), str)
            or _LOWER_SHA256.fullmatch(str(amendments[0].get("sha256"))) is None
        ):
            raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    roles = value["roles"]
    if not isinstance(roles, list) or len(roles) != len(role_names):
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    role_fields = {
        "role",
        "source_git_commit",
        "runtime_sha256",
        "runtime_profile_sha256",
        "selected_source_manifest_sha256",
    }
    checked: dict[str, dict[str, object]] = {}
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
            raise RuntimeError("Policy-improvement runtime authorization is invalid.")
        checked[role_name] = role
    if (
        checked["policy-improvement-training"]["source_git_commit"]
        != value["producer_git_commit"]
    ):
        raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    if schema == ("policy_improvement_runtime_authorization_v2", 2):
        for semantic_role, launcher_role in (
            ("policy-improvement-training", POLICY_FULL_ROLE),
            ("policy-improvement-evaluation", POLICY_THEORY_BRIDGE_ROLE),
        ):
            if any(
                checked[semantic_role][field] != checked[launcher_role][field]
                for field in (
                    "source_git_commit",
                    "runtime_sha256",
                    "runtime_profile_sha256",
                    "selected_source_manifest_sha256",
                )
            ):
                raise RuntimeError(
                    "Policy-improvement runtime authorization is invalid."
                )
        if (
            checked[POLICY_FULL_ROLE]["source_git_commit"]
            != value["producer_git_commit"]
        ):
            raise RuntimeError("Policy-improvement runtime authorization is invalid.")
    active_roles = (
        ("policy-improvement-training", "policy-improvement-evaluation")
        if phase4_role == POLICY_SMOKE_ROLE
        else (phase4_role,)
    )
    for active_role in active_roles:
        role = checked.get(active_role)
        if role is None or any(
            role[field] != expected
            for field, expected in (
                ("source_git_commit", source_git_commit),
                ("runtime_sha256", runtime_sha256),
                ("runtime_profile_sha256", source_manifest_sha256),
                ("selected_source_manifest_sha256", source_manifest_sha256),
            )
        ):
            raise RuntimeError("Policy-improvement runtime authorization is invalid.")


def _hash_runtime(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeError("Verified runtime artifact cannot be read.") from exc
    return digest.hexdigest()


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def preflight_runtime(
    *,
    module_file: str,
    expected_module_name: str,
    environ: MutableMapping[str, str],
    attestation_required: bool,
    allowed_phase4_roles: frozenset[str] | None = None,
) -> RuntimePreflight | None:
    """Verify one inherited sealed runtime before behavior modules are imported."""

    attested_path_raw = environ.pop(VERIFIED_RUNTIME_PATH_ENV, None)
    attested_sha256 = environ.pop(VERIFIED_RUNTIME_SHA256_ENV, None)
    attested_fd_raw = environ.pop(VERIFIED_RUNTIME_FD_ENV, None)
    private_unpack_raw = environ.pop(PRIVATE_UNPACK_BASE_ENV, None)
    private_unpack_fd_raw = environ.pop(PRIVATE_UNPACK_FD_ENV, None)
    phase4_role = environ.pop(PHASE4_RUNTIME_ROLE_ENV, None)
    source_git_commit = environ.pop(PHASE4_SOURCE_COMMIT_ENV, None)
    source_manifest_sha256 = environ.pop(
        PHASE4_SOURCE_MANIFEST_SHA256_ENV,
        None,
    )
    policy_runtime_authorization_json = environ.pop(
        POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV,
        None,
    )
    policy_runtime_authorization_sha256 = environ.pop(
        POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV,
        None,
    )
    policy_runtime_profile_sha256 = environ.pop(
        POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV,
        None,
    )
    policy_selected_source_manifest_sha256 = environ.pop(
        POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV,
        None,
    )
    policy_protocol_sha256 = environ.pop(POLICY_PROTOCOL_SHA256_ENV, None)
    attestation_values = (
        attested_path_raw,
        attested_sha256,
        attested_fd_raw,
        private_unpack_raw,
        private_unpack_fd_raw,
        phase4_role,
        source_git_commit,
        source_manifest_sha256,
        policy_runtime_authorization_json,
        policy_runtime_authorization_sha256,
        policy_runtime_profile_sha256,
        policy_selected_source_manifest_sha256,
        policy_protocol_sha256,
    )
    if not attestation_required:
        if any(value is not None for value in attestation_values):
            raise RuntimeError(
                "Packaged-runtime attestation is valid only for an "
                "authenticated evidence mode."
            )
        return None

    if any(
        value is None
        for value in (
            attested_path_raw,
            attested_sha256,
            attested_fd_raw,
            private_unpack_raw,
            private_unpack_fd_raw,
        )
    ):
        raise RuntimeError(
            "Authenticated execution requires the verified packaged-runtime launcher."
        )
    assert attested_sha256 is not None
    assert attested_fd_raw is not None
    assert private_unpack_fd_raw is not None
    assert attested_path_raw is not None
    assert private_unpack_raw is not None
    if not _LOWER_SHA256.fullmatch(attested_sha256):
        raise RuntimeError(
            "Verified runtime SHA-256 must be 64 lowercase hexadecimal characters."
        )
    if allowed_phase4_roles is None:
        if any(
            value is not None
            for value in (
                phase4_role,
                source_git_commit,
                source_manifest_sha256,
            )
        ):
            raise RuntimeError(
                "Confirmatory execution cannot consume Phase 4 attestation."
            )
    else:
        if (
            phase4_role not in allowed_phase4_roles
            or source_git_commit is None
            or not _LOWER_COMMIT.fullmatch(source_git_commit)
            or source_manifest_sha256 is None
            or not _LOWER_SHA256.fullmatch(source_manifest_sha256)
        ):
            raise RuntimeError("Phase 4 runtime attestation is invalid.")
        policy_values = (
            policy_runtime_authorization_json,
            policy_runtime_authorization_sha256,
            policy_runtime_profile_sha256,
            policy_selected_source_manifest_sha256,
            policy_protocol_sha256,
        )
        if phase4_role in POLICY_AUTHORIZED_ROLES:
            if (
                any(value is None for value in policy_values)
                or not isinstance(policy_runtime_authorization_json, str)
                or not isinstance(policy_runtime_authorization_sha256, str)
                or not _LOWER_SHA256.fullmatch(policy_runtime_authorization_sha256)
                or not isinstance(policy_runtime_profile_sha256, str)
                or not _LOWER_SHA256.fullmatch(policy_runtime_profile_sha256)
                or not isinstance(policy_selected_source_manifest_sha256, str)
                or not _LOWER_SHA256.fullmatch(policy_selected_source_manifest_sha256)
                or not isinstance(policy_protocol_sha256, str)
                or not _LOWER_SHA256.fullmatch(policy_protocol_sha256)
                or policy_runtime_profile_sha256
                != policy_selected_source_manifest_sha256
            ):
                raise RuntimeError(
                    "Policy-improvement runtime authorization is invalid."
                )
            try:
                authorization_bytes = policy_runtime_authorization_json.encode("ascii")
                decoded_authorization = json.loads(policy_runtime_authorization_json)
                canonical_authorization = json.dumps(
                    decoded_authorization,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            except (
                UnicodeEncodeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    "Policy-improvement runtime authorization is invalid."
                ) from exc
            if (
                canonical_authorization != authorization_bytes
                or hashlib.sha256(authorization_bytes).hexdigest()
                != policy_runtime_authorization_sha256
            ):
                raise RuntimeError(
                    "Policy-improvement runtime authorization changed in transit."
                )
            assert phase4_role is not None
            assert source_git_commit is not None
            assert source_manifest_sha256 is not None
            _validate_policy_runtime_authorization(
                decoded_authorization,
                phase4_role=phase4_role,
                runtime_sha256=attested_sha256,
                source_git_commit=source_git_commit,
                source_manifest_sha256=source_manifest_sha256,
                protocol_sha256=policy_protocol_sha256,
            )
        elif any(value is not None for value in policy_values):
            raise RuntimeError(
                "Policy-improvement authorization is valid only for an "
                "authenticated policy runtime role."
            )
    if not _MEMFD_SEALING_AVAILABLE:
        raise RuntimeError("This host cannot inspect a sealed runtime.")
    if not re.fullmatch(r"[0-9]+", attested_fd_raw):
        raise RuntimeError("Verified runtime descriptor must be a decimal integer.")
    runtime_descriptor = int(attested_fd_raw)
    if runtime_descriptor < 3:
        raise RuntimeError("Verified runtime descriptor is reserved or invalid.")
    if not re.fullmatch(r"[0-9]+", private_unpack_fd_raw):
        raise RuntimeError("Private unpack descriptor must be a decimal integer.")
    private_unpack_descriptor = int(private_unpack_fd_raw)
    if private_unpack_descriptor < 3 or private_unpack_descriptor == runtime_descriptor:
        raise RuntimeError("Private unpack descriptor is reserved or invalid.")

    descriptor_path = Path(f"/proc/self/fd/{runtime_descriptor}")
    if attested_path_raw != str(descriptor_path) or environ.get(
        "FB_PAR_FILENAME"
    ) != str(descriptor_path):
        raise RuntimeError("Runtime attestation has an invalid path.")
    module_match = re.fullmatch(
        rf"/proc/self/fd/([0-9]+)/{re.escape(expected_module_name)}",
        Path(module_file).as_posix(),
    )
    if module_match is None:
        raise RuntimeError("Entrypoint was not imported from the attested PAR.")
    module_descriptor = int(module_match.group(1))
    if module_descriptor < 3:
        raise RuntimeError("Entrypoint module descriptor is reserved or invalid.")

    try:
        runtime_before = os.fstat(runtime_descriptor)
        module_before = os.fstat(module_descriptor)
        runtime_seals = fcntl.fcntl(runtime_descriptor, _F_GET_SEALS)
        module_seals = fcntl.fcntl(module_descriptor, _F_GET_SEALS)
    except OSError as exc:
        raise RuntimeError("Runtime descriptor cannot be inspected.") from exc
    if (
        not stat.S_ISREG(runtime_before.st_mode)
        or not stat.S_ISREG(module_before.st_mode)
        or not os.get_inheritable(runtime_descriptor)
        or runtime_seals & _REQUIRED_MEMFD_SEALS != _REQUIRED_MEMFD_SEALS
        or module_seals & _REQUIRED_MEMFD_SEALS != _REQUIRED_MEMFD_SEALS
        or (
            runtime_before.st_dev,
            runtime_before.st_ino,
            runtime_before.st_size,
        )
        != (
            module_before.st_dev,
            module_before.st_ino,
            module_before.st_size,
        )
        or not zipfile.is_zipfile(descriptor_path)
    ):
        raise RuntimeError(
            "Authenticated execution requires the immutable attested PAR artifact."
        )

    private_unpack_path = Path(f"/proc/self/fd/{private_unpack_descriptor}")
    if private_unpack_raw != str(private_unpack_path) or environ.pop(
        "FB_PAR_UNPACK_BASEDIR", None
    ) != str(private_unpack_path):
        raise RuntimeError("Private unpack attestation has an invalid path.")
    try:
        unpack_before = os.fstat(private_unpack_descriptor)
    except OSError as exc:
        raise RuntimeError("Private unpack descriptor cannot be inspected.") from exc
    actual_sha256 = _hash_runtime(descriptor_path)
    try:
        runtime_after = os.fstat(runtime_descriptor)
        module_after = os.fstat(module_descriptor)
        unpack_after = os.fstat(private_unpack_descriptor)
    except OSError as exc:
        raise RuntimeError(
            "Runtime or private unpack directory changed during preflight."
        ) from exc
    if (
        len(
            {
                _identity(runtime_before),
                _identity(runtime_after),
                _identity(module_before),
                _identity(module_after),
            }
        )
        != 1
        or actual_sha256 != attested_sha256
    ):
        raise RuntimeError("Runtime artifact differs from its launcher attestation.")
    if (
        (
            unpack_before.st_dev,
            unpack_before.st_ino,
            stat.S_IFMT(unpack_before.st_mode),
        )
        != (
            unpack_after.st_dev,
            unpack_after.st_ino,
            stat.S_IFMT(unpack_after.st_mode),
        )
        or not stat.S_ISDIR(unpack_after.st_mode)
        or unpack_after.st_uid != os.geteuid()
        or stat.S_IMODE(unpack_after.st_mode) != 0o700
        or not os.get_inheritable(private_unpack_descriptor)
    ):
        raise RuntimeError("Private unpack directory has an invalid identity or mode.")
    try:
        os.set_inheritable(runtime_descriptor, False)
        if module_descriptor != runtime_descriptor:
            os.set_inheritable(module_descriptor, False)
        os.set_inheritable(private_unpack_descriptor, False)
    except OSError as exc:
        raise RuntimeError("Runtime descriptor cannot be isolated.") from exc
    return RuntimePreflight(
        runtime_sha256=actual_sha256,
        private_unpack_descriptor=private_unpack_descriptor,
        runtime_descriptor=runtime_descriptor,
        phase4_role=phase4_role,
        source_git_commit=source_git_commit,
        source_manifest_sha256=source_manifest_sha256,
        policy_runtime_authorization_json=(policy_runtime_authorization_json),
        policy_runtime_authorization_sha256=(policy_runtime_authorization_sha256),
        policy_runtime_profile_sha256=policy_runtime_profile_sha256,
        policy_selected_source_manifest_sha256=(policy_selected_source_manifest_sha256),
        policy_protocol_sha256=policy_protocol_sha256,
    )
