#!/usr/bin/env fbpython
"""Generate one fail-closed policy-improvement runtime authorization v3."""

# pyre-strict

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

from confirmatory_runtime_launcher import (
    ConfirmatoryRuntimeError,
    SOURCE_MANIFEST_RELATIVE_PATH,
    validate_archive_layout,
    validate_confirmatory_archive_sources,
)
from phase4_runtime_profile import (
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
    AuthorizedPhase4Profile,
    AuthorizedTrainingSource,
    Phase4RuntimeProfileError,
    assert_phase4_archive_matches_profile,
    authorize_phase4_source_profile,
    authorize_phase4_training_source,
)
from scripts.policy_improvement_populations import validate_v2_populations
from scripts.policy_improvement_schema import (
    RUNTIME_ROLES_V2,
    canonical_json_bytes,
    validate_runtime_authorization,
)
from scripts.policy_improvement_v2_registry import (
    load_v2_base_configs,
    validate_v2_registry_document,
)
from scripts.policy_improvement_v2_schema import (
    PolicyImprovementV2SchemaError,
    load_strict_json,
    sha256_json,
    validate_v2_protocol,
)


_PROTOCOL_RELATIVE_PATH = "configs/policy_improvement_v2/protocol.json"
_POPULATIONS_RELATIVE_PATH = "configs/policy_improvement_v2/populations.json"
_REGISTRY_RELATIVE_PATH = "configs/policy_improvement_v2/registry.json"
_THEORY_AMENDMENT_RELATIVE_PATH = (
    "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
)
_PROFILE_NAMES = (
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
)
_RUNTIME_NAMES = ("training", "full", "theory", "audit", "analysis")
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_READ_SIZE = 1024 * 1024


class RuntimeAuthorizationGenerationError(RuntimeError):
    """Raised when authorization inputs cannot be bound without ambiguity."""


@dataclass(frozen=True)
class SourceAuthorization:
    training: AuthorizedTrainingSource
    profiles: Mapping[str, AuthorizedPhase4Profile]


@dataclass(frozen=True)
class PolicyRegistration:
    protocol: dict[str, Any]
    protocol_sha256: str
    registry: dict[str, Any]
    registry_sha256: str
    amendment: dict[str, Any]
    amendment_sha256: str


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class ArtifactAuthorization:
    launcher: ArtifactIdentity
    runtimes: Mapping[str, ArtifactIdentity]


@dataclass(frozen=True)
class PublishedAuthorization:
    path: Path
    sha256: str
    document: dict[str, Any]


def _run_git(root: Path, *arguments: str) -> str:
    environment = {
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise RuntimeAuthorizationGenerationError(
            "Source Git identity cannot be authenticated."
        ) from exc
    return completed.stdout.strip()


def _canonical_existing_directory(path_value: str, *, label: str) -> Path:
    if not os.path.isabs(path_value) or os.path.realpath(path_value) != path_value:
        raise RuntimeAuthorizationGenerationError(
            f"{label} must be an absolute canonical directory path."
        )
    path = Path(path_value)
    try:
        status = path.lstat()
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(f"{label} does not exist.") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise RuntimeAuthorizationGenerationError(f"{label} must be a directory.")
    return path


def require_clean_exact_git_commit(
    project_root: str,
    expected_git_commit: str,
) -> Path:
    """Require one canonical Git root at the exact clean commit."""

    if _LOWER_COMMIT.fullmatch(expected_git_commit) is None:
        raise RuntimeAuthorizationGenerationError(
            "Expected Git commit must be 40 lowercase hexadecimal characters."
        )
    root = _canonical_existing_directory(project_root, label="Source project root")
    top_level = _run_git(root, "rev-parse", "--show-toplevel")
    if os.path.realpath(top_level) != str(root):
        raise RuntimeAuthorizationGenerationError(
            "Source project root must be the Git top level."
        )
    if _run_git(root, "rev-parse", "--verify", "HEAD^{commit}") != expected_git_commit:
        raise RuntimeAuthorizationGenerationError(
            "Source checkout differs from the expected commit."
        )
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeAuthorizationGenerationError("Source checkout is not clean.")
    return root


def _load_canonical_document(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    try:
        value = load_strict_json(path)
    except PolicyImprovementV2SchemaError as exc:
        raise RuntimeAuthorizationGenerationError(
            f"Registered document {relative_path!r} is not strict JSON."
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeAuthorizationGenerationError(
            f"Registered document {relative_path!r} must be one JSON object."
        )
    # Source profiles bind the checked-in spelling. Authorization identities use
    # the canonical parsed bytes, matching the pre-import launcher contract.
    canonical_json_bytes(value)
    return dict(value)


def _validate_theory_amendment(
    amendment: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    populations_sha256: str,
    registry: Mapping[str, Any],
    registry_sha256: str,
) -> None:
    expected_fields = {
        "amendment_id",
        "analysis",
        "bellman_estimators",
        "centering_contract",
        "checkpoint_schedule",
        "created_at_utc",
        "deployment_contract",
        "evaluator_contract",
        "metrics",
        "multi_fidelity_exact_method_screen",
        "outcome_evidence_inspected",
        "population_registry_sha256",
        "predictive_return_estimator",
        "prior_amendment_history_sha256",
        "protocol_id",
        "protocol_schema_name",
        "protocol_schema_version",
        "protocol_sha256",
        "reference_depths",
        "result_schema",
        "schema_name",
        "schema_version",
        "smoke_policy",
        "source_registry_schema_name",
        "source_registry_schema_version",
        "source_registry_sha256",
        "test_data_opened",
    }
    if set(amendment) != expected_fields:
        raise RuntimeAuthorizationGenerationError(
            "Theory amendment field inventory differs."
        )
    required = {
        "schema_name": "policy_improvement_theory_bridge_amendment_v2",
        "schema_version": 1,
        "amendment_id": "pre-stage1-theory-bridge-v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_schema_name": protocol["schema_name"],
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": protocol_sha256,
        "population_registry_sha256": populations_sha256,
        "source_registry_schema_name": registry["schema_name"],
        "source_registry_schema_version": registry["registry_schema_version"],
        "source_registry_sha256": registry_sha256,
        "prior_amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
        "test_data_opened": False,
        "outcome_evidence_inspected": False,
    }
    if any(amendment.get(field) != expected for field, expected in required.items()):
        raise RuntimeAuthorizationGenerationError(
            "Theory amendment identity or pre-outcome attestations differ."
        )


def _load_policy_registration(root: Path) -> PolicyRegistration:
    protocol = validate_v2_protocol(
        _load_canonical_document(root, _PROTOCOL_RELATIVE_PATH)
    )
    populations = validate_v2_populations(
        _load_canonical_document(root, _POPULATIONS_RELATIVE_PATH)
    )
    population_sha256 = sha256_json(populations)
    if protocol["population_registry"]["sha256"] != population_sha256:
        raise RuntimeAuthorizationGenerationError(
            "Protocol population-registry digest differs from canonical bytes."
        )
    configs = load_v2_base_configs(protocol, root)
    registry = validate_v2_registry_document(
        _load_canonical_document(root, _REGISTRY_RELATIVE_PATH),
        protocol,
        populations,
        base_configs=configs,
    )
    protocol_sha256 = sha256_json(protocol)
    registry_sha256 = sha256_json(registry)
    amendment = _load_canonical_document(root, _THEORY_AMENDMENT_RELATIVE_PATH)
    _validate_theory_amendment(
        amendment,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        populations_sha256=population_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
    )
    return PolicyRegistration(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
        amendment=amendment,
        amendment_sha256=sha256_json(amendment),
    )


def _authenticate_sources(root: Path, expected_git_commit: str) -> SourceAuthorization:
    training = authorize_phase4_training_source(root, expected_git_commit)
    profiles = {
        profile: authorize_phase4_source_profile(root, expected_git_commit, profile)
        for profile in _PROFILE_NAMES
    }
    return SourceAuthorization(training=training, profiles=profiles)


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode))


def _hash_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(_READ_SIZE), b""):
        digest.update(block)
    return digest.hexdigest()


def _authenticate_artifact(
    path_value: str,
    *,
    label: str,
    archive_validator: Callable[[ZipFile], None] | None = None,
) -> ArtifactIdentity:
    if not os.path.isabs(path_value) or os.path.realpath(path_value) != path_value:
        raise RuntimeAuthorizationGenerationError(
            f"{label} path must be absolute, canonical, and non-symlinked."
        )
    path = Path(path_value)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(f"{label} does not exist.") from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise RuntimeAuthorizationGenerationError(f"{label} must be a regular file.")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(f"{label} cannot be opened.") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            descriptor_before = os.fstat(descriptor)
            if (
                _file_identity(descriptor_before) != _file_identity(path_before)
                or descriptor_before.st_size <= 0
            ):
                raise RuntimeAuthorizationGenerationError(
                    f"{label} changed before authentication."
                )
            digest = _hash_open_file(handle)
            if archive_validator is not None:
                handle.seek(0)
                try:
                    with ZipFile(handle, "r") as archive:
                        archive_validator(archive)
                except BadZipFile as exc:
                    raise RuntimeAuthorizationGenerationError(
                        f"{label} is not a valid ZIP-based PAR."
                    ) from exc
            descriptor_after = os.fstat(descriptor)
        path_after = path.lstat()
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            f"{label} could not be authenticated."
        ) from exc
    finally:
        os.close(descriptor)
    identities = {
        _file_identity(path_before),
        _file_identity(descriptor_before),
        _file_identity(descriptor_after),
        _file_identity(path_after),
    }
    if len(identities) != 1:
        raise RuntimeAuthorizationGenerationError(
            f"{label} changed during authentication."
        )
    return ArtifactIdentity(
        path=path,
        sha256=digest,
        device=descriptor_before.st_dev,
        inode=descriptor_before.st_ino,
    )


def _training_archive_validator(
    expected_manifest_bytes: bytes,
) -> Callable[[ZipFile], None]:
    def validate(archive: ZipFile) -> None:
        validate_archive_layout(archive)
        validate_confirmatory_archive_sources(archive)
        try:
            embedded_manifest = archive.read(SOURCE_MANIFEST_RELATIVE_PATH)
        except (BadZipFile, KeyError, OSError, RuntimeError) as exc:
            raise RuntimeAuthorizationGenerationError(
                "Training PAR lacks its exact producer manifest."
            ) from exc
        if embedded_manifest != expected_manifest_bytes:
            raise RuntimeAuthorizationGenerationError(
                "Training PAR producer manifest differs from the clean checkout."
            )

    return validate


def _profile_archive_validator(
    authorized: AuthorizedPhase4Profile,
) -> Callable[[ZipFile], None]:
    def validate(archive: ZipFile) -> None:
        validate_archive_layout(archive)
        assert_phase4_archive_matches_profile(archive, authorized)

    return validate


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR manifest contains a duplicate JSON key."
            )
        value[key] = item
    return value


def _launcher_archive_validator(
    authorized: AuthorizedPhase4Profile,
) -> Callable[[ZipFile], None]:
    validate_profile = _profile_archive_validator(authorized)

    def validate(archive: ZipFile) -> None:
        try:
            validate_profile(archive)
        except (ConfirmatoryRuntimeError, Phase4RuntimeProfileError) as exc:
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR source profile differs from the clean checkout."
            ) from exc
        try:
            raw_manifest = archive.read("__manifest__.json")
            manifest = json.loads(
                raw_manifest.decode("utf-8"),
                object_pairs_hook=_strict_json_object,
            )
        except (
            BadZipFile,
            KeyError,
            OSError,
            RuntimeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            if isinstance(exc, RuntimeAuthorizationGenerationError):
                raise
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR lacks a valid executable manifest."
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR executable manifest must be one JSON object."
            )
        fbmake = manifest.get("fbmake")
        if (
            not isinstance(fbmake, dict)
            or fbmake.get("main_module") != "phase4_runtime_launcher"
            or "main_function" not in fbmake
            or fbmake["main_function"] is not None
            or fbmake.get("build_rule_type") != "python_binary"
            or type(fbmake.get("rule_type_is_unit_test")) is not int
            or fbmake["rule_type_is_unit_test"] != 0
        ):
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR does not execute the authenticated launcher module."
            )

    return validate


def _authenticate_artifacts(
    *,
    launcher_path: str,
    runtime_paths: Mapping[str, str],
    sources: SourceAuthorization,
) -> ArtifactAuthorization:
    if set(runtime_paths) != set(_RUNTIME_NAMES):
        raise RuntimeAuthorizationGenerationError(
            "Exactly five named runtime PAR paths are required."
        )
    launcher = _authenticate_artifact(
        launcher_path,
        label="Launcher",
        archive_validator=_launcher_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE]
        ),
    )
    validators = {
        "training": _training_archive_validator(sources.training.manifest_bytes),
        "full": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE]
        ),
        "theory": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE]
        ),
        "audit": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE]
        ),
        "analysis": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE]
        ),
    }
    runtimes = {
        name: _authenticate_artifact(
            runtime_paths[name],
            label=f"{name} runtime PAR",
            archive_validator=validators[name],
        )
        for name in _RUNTIME_NAMES
    }
    identities = {(launcher.device, launcher.inode)}
    identities.update(
        (artifact.device, artifact.inode) for artifact in runtimes.values()
    )
    if len(identities) != len(_RUNTIME_NAMES) + 1:
        raise RuntimeAuthorizationGenerationError(
            "Launcher and the five runtime PAR paths must identify distinct files."
        )
    return ArtifactAuthorization(launcher=launcher, runtimes=runtimes)


def build_runtime_authorization(
    *,
    authorization_id: str,
    created_at_utc: str,
    expected_git_commit: str,
    registration: PolicyRegistration,
    sources: SourceAuthorization,
    artifacts: ArtifactAuthorization,
) -> dict[str, Any]:
    if (
        registration.protocol_sha256 != sha256_json(registration.protocol)
        or registration.registry_sha256 != sha256_json(registration.registry)
        or registration.amendment_sha256 != sha256_json(registration.amendment)
    ):
        raise RuntimeAuthorizationGenerationError(
            "Canonical registration digests differ from their documents."
        )
    _validate_theory_amendment(
        registration.amendment,
        protocol=registration.protocol,
        protocol_sha256=registration.protocol_sha256,
        populations_sha256=registration.protocol["population_registry"]["sha256"],
        registry=registration.registry,
        registry_sha256=registration.registry_sha256,
    )
    if (
        sources.training.git_commit != expected_git_commit
        or set(sources.profiles) != set(_PROFILE_NAMES)
        or any(
            profile.git_commit != expected_git_commit
            for profile in sources.profiles.values()
        )
    ):
        raise RuntimeAuthorizationGenerationError(
            "Source authorizations do not bind the expected exact commit."
        )
    if set(artifacts.runtimes) != set(_RUNTIME_NAMES):
        raise RuntimeAuthorizationGenerationError(
            "Artifact authorization lacks one of the five runtime PARs."
        )
    artifact_identities = {(artifacts.launcher.device, artifacts.launcher.inode)}
    artifact_identities.update(
        (artifact.device, artifact.inode) for artifact in artifacts.runtimes.values()
    )
    if len(artifact_identities) != len(_RUNTIME_NAMES) + 1:
        raise RuntimeAuthorizationGenerationError(
            "Artifact authorization contains aliased files."
        )
    profile_digests = {
        "training": sources.training.source_manifest_sha256,
        "evaluation": sources.training.source_manifest_sha256,
        "audit": sources.profiles[
            POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE
        ].source_manifest_sha256,
        "analysis": sources.profiles[
            POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE
        ].source_manifest_sha256,
        "full": sources.profiles[
            POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE
        ].source_manifest_sha256,
        "theory": sources.profiles[
            POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE
        ].source_manifest_sha256,
    }
    runtime_digests = {
        "training": artifacts.runtimes["training"].sha256,
        "evaluation": artifacts.runtimes["training"].sha256,
        "audit": artifacts.runtimes["audit"].sha256,
        "analysis": artifacts.runtimes["analysis"].sha256,
        "full": artifacts.runtimes["full"].sha256,
        "theory": artifacts.runtimes["theory"].sha256,
    }
    role_keys = ("training", "evaluation", "audit", "analysis", "full", "theory")
    document = {
        "schema_name": "policy_improvement_runtime_authorization_v3",
        "schema_version": 3,
        "authorization_id": authorization_id,
        "created_at_utc": created_at_utc,
        "protocol_sha256": registration.protocol_sha256,
        "protocol": {
            "schema_name": registration.protocol["schema_name"],
            "schema_version": registration.protocol["schema_version"],
            "protocol_id": registration.protocol["protocol_id"],
            "sha256": registration.protocol_sha256,
        },
        "registry": {
            "schema_name": registration.registry["schema_name"],
            "schema_version": registration.registry["registry_schema_version"],
            "sha256": registration.registry_sha256,
        },
        "amendments": [
            {
                "schema_name": registration.amendment["schema_name"],
                "schema_version": registration.amendment["schema_version"],
                "amendment_id": registration.amendment["amendment_id"],
                "sha256": registration.amendment_sha256,
            }
        ],
        "producer_git_commit": expected_git_commit,
        "producer_source_manifest_sha256": sources.training.source_manifest_sha256,
        "launcher_sha256": artifacts.launcher.sha256,
        "roles": [
            {
                "role": RUNTIME_ROLES_V2[index],
                "source_git_commit": expected_git_commit,
                "runtime_sha256": runtime_digests[key],
                "runtime_profile_sha256": profile_digests[key],
                "selected_source_manifest_sha256": profile_digests[key],
            }
            for index, key in enumerate(role_keys)
        ],
    }
    validated = validate_runtime_authorization(document)
    if canonical_json_bytes(validated) != canonical_json_bytes(document):
        raise RuntimeAuthorizationGenerationError(
            "Runtime authorization changed during self-validation."
        )
    return validated


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeAuthorizationGenerationError(
                "Runtime authorization publication made no progress."
            )
        offset += written


def _publish_new_private_file(
    *,
    owner_directory: str,
    output_name: str,
    project_root: Path,
    payload: bytes,
) -> Path:
    owner = _canonical_existing_directory(
        owner_directory,
        label="Authorization owner directory",
    )
    try:
        inside_project = os.path.commonpath((str(project_root), str(owner))) == str(
            project_root
        )
    except ValueError:
        inside_project = False
    if inside_project:
        raise RuntimeAuthorizationGenerationError(
            "Runtime authorization cannot be published inside the source repository."
        )
    if (
        not output_name
        or output_name in {".", ".."}
        or Path(output_name).name != output_name
        or not output_name.isascii()
    ):
        raise RuntimeAuthorizationGenerationError(
            "Authorization output name must be one ASCII filename."
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(owner, flags)
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Authorization owner directory cannot be opened safely."
        ) from exc
    temporary_name = f".{output_name}.{secrets.token_hex(16)}.tmp"
    temporary_descriptor = -1
    published = False
    complete = False
    temporary_identity: tuple[int, int] | None = None
    try:
        directory_status = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_status.st_mode)
            or directory_status.st_uid != os.geteuid()
            or stat.S_IMODE(directory_status.st_mode) != 0o700
        ):
            raise RuntimeAuthorizationGenerationError(
                "Authorization owner directory must be owner-only mode 0700."
            )
        try:
            os.stat(output_name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeAuthorizationGenerationError(
                "Authorization output already exists."
            )
        temporary_descriptor = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        os.fchmod(temporary_descriptor, 0o600)
        _write_all(temporary_descriptor, payload)
        os.fsync(temporary_descriptor)
        status = os.fstat(temporary_descriptor)
        temporary_identity = (status.st_dev, status.st_ino)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_size != len(payload)
        ):
            raise RuntimeAuthorizationGenerationError(
                "Authorization temporary file has an invalid identity."
            )
        os.lseek(temporary_descriptor, 0, os.SEEK_SET)
        if (
            b"".join(iter(lambda: os.read(temporary_descriptor, _READ_SIZE), b""))
            != payload
        ):
            raise RuntimeAuthorizationGenerationError(
                "Authorization temporary file differs from canonical bytes."
            )
        os.link(
            temporary_name,
            output_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        published = True
        output_status = os.stat(
            output_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            (output_status.st_dev, output_status.st_ino) != temporary_identity
            or not stat.S_ISREG(output_status.st_mode)
            or output_status.st_uid != os.geteuid()
            or stat.S_IMODE(output_status.st_mode) != 0o600
            or output_status.st_nlink != 2
        ):
            raise RuntimeAuthorizationGenerationError(
                "Published authorization has an invalid identity."
            )
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        temporary_name = ""
        os.fsync(directory_descriptor)
        final_status = os.stat(
            output_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        current_directory_status = owner.lstat()
        if (
            (final_status.st_dev, final_status.st_ino) != temporary_identity
            or final_status.st_nlink != 1
            or stat.S_IMODE(final_status.st_mode) != 0o600
            or _directory_identity(current_directory_status)
            != _directory_identity(directory_status)
            or current_directory_status.st_uid != os.geteuid()
            or stat.S_IMODE(current_directory_status.st_mode) != 0o700
        ):
            raise RuntimeAuthorizationGenerationError(
                "Final authorization or owner-directory identity changed."
            )
        complete = True
    except FileExistsError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Authorization output already exists."
        ) from exc
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Runtime authorization could not be published atomically."
        ) from exc
    finally:
        if published and not complete and temporary_identity is not None:
            try:
                status = os.stat(
                    output_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (status.st_dev, status.st_ino) == temporary_identity:
                    os.unlink(output_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        os.close(directory_descriptor)
    return owner / output_name


def generate_runtime_authorization(
    *,
    project_root: str,
    expected_git_commit: str,
    authorization_id: str,
    created_at_utc: str,
    launcher_path: str,
    runtime_paths: Mapping[str, str],
    owner_directory: str,
    output_name: str,
) -> PublishedAuthorization:
    root = require_clean_exact_git_commit(project_root, expected_git_commit)
    registration_before = _load_policy_registration(root)
    sources_before = _authenticate_sources(root, expected_git_commit)
    artifacts = _authenticate_artifacts(
        launcher_path=launcher_path,
        runtime_paths=runtime_paths,
        sources=sources_before,
    )
    registration_after = _load_policy_registration(root)
    sources_after = _authenticate_sources(root, expected_git_commit)
    require_clean_exact_git_commit(project_root, expected_git_commit)
    if registration_after != registration_before or sources_after != sources_before:
        raise RuntimeAuthorizationGenerationError(
            "Source or registered protocol changed during authorization."
        )
    document = build_runtime_authorization(
        authorization_id=authorization_id,
        created_at_utc=created_at_utc,
        expected_git_commit=expected_git_commit,
        registration=registration_before,
        sources=sources_before,
        artifacts=artifacts,
    )
    payload = canonical_json_bytes(document)
    output_path = _publish_new_private_file(
        owner_directory=owner_directory,
        output_name=output_name,
        project_root=root,
        payload=payload,
    )
    return PublishedAuthorization(
        path=output_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        document=document,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--training-runtime", required=True)
    parser.add_argument("--full-runtime", required=True)
    parser.add_argument("--theory-runtime", required=True)
    parser.add_argument("--audit-runtime", required=True)
    parser.add_argument("--analysis-runtime", required=True)
    parser.add_argument("--owner-directory", required=True)
    parser.add_argument("--output-name", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    published = generate_runtime_authorization(
        project_root=arguments.project_root,
        expected_git_commit=arguments.expected_git_commit,
        authorization_id=arguments.authorization_id,
        created_at_utc=arguments.created_at_utc,
        launcher_path=arguments.launcher,
        runtime_paths={
            "training": arguments.training_runtime,
            "full": arguments.full_runtime,
            "theory": arguments.theory_runtime,
            "audit": arguments.audit_runtime,
            "analysis": arguments.analysis_runtime,
        },
        owner_directory=arguments.owner_directory,
        output_name=arguments.output_name,
    )
    summary = canonical_json_bytes(
        {
            "path": str(published.path),
            "runtime_authorization_sha256": published.sha256,
        }
    )
    sys.stdout.buffer.write(summary + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
