#!/usr/bin/env fbpython
"""Authenticate immutable policy-improvement run generations.

The result schema validates claims.  This module binds those claims to the
regular files in one immutable run generation.  It deliberately uses only the
standard library so the audit entrypoint can authenticate evidence without
loading a checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from policy_improvement_sealed_evidence import (
    SealedCheckpoint,
    SealedCheckpointError,
    seal_generation_checkpoint,
)
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
    load_strict_json_bytes,
    runtime_authorization_sha256,
    validate_runtime_authorization,
    validate_result,
)


_COMPLETE_SEGMENT_SCHEMAS = {
    "policy_improvement_smoke_segment_v1",
    "policy_improvement_run_segment_v1",
}
_SEMANTIC_VALIDATION_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "method_id",
    "seed",
    "snapshot_kind",
    "environment_interactions",
    "parent_checkpoint_sha256",
    "checkpoint_sha256",
    "initialization_sha256",
    "model_state_sha256",
    "role_state_sha256s",
    "method_config_sha256",
    "registered_effective_config_sha256",
    "effective_config_sha256",
    "dataset_manifest_sha256",
    "dataset_provenance_sha256",
    "run_identity_sha256",
    "training_call_delta",
    "evaluation_call_delta",
    "optimizer_step_delta",
}


def _object(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementSchemaError(f"{path} must be an object.")
    return value


def _fields(value: object, expected: set[str], *, path: str) -> Mapping[str, object]:
    item = _object(value, path=path)
    if set(item) != expected:
        raise PolicyImprovementSchemaError(
            f"{path} fields differ: missing={sorted(expected - set(item))}, "
            f"extra={sorted(set(item) - expected)}."
        )
    return item


def _sha256(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyImprovementSchemaError(
            f"{path} must be a lowercase SHA-256 digest."
        )
    return value


def _available(value: object, *, path: str) -> object:
    availability = _fields(value, {"status", "value"}, path=path)
    if availability["status"] != "available":
        raise PolicyImprovementSchemaError(f"{path} must be available.")
    return availability["value"]


@dataclass(frozen=True)
class _PendingCheckpointValidation:
    """One authenticated checkpoint awaiting sealing and semantic validation.

    Nothing here is deserialized.  The record only carries the identity this
    module proved against an immutable generation manifest, so that sealing and
    the semantic validator can run strictly after every authentication step.
    """

    generation: Path
    generation_relative_path: str
    checkpoint_sha256: str
    size_bytes: int
    validation: Mapping[str, object]
    environment_interactions: int
    parent_checkpoint_sha256: str | None


def _validate_checkpoint_validator_identity(
    value: object,
    *,
    validator: object,
    result: Mapping[str, object],
    path: str,
) -> None:
    identity = _fields(
        value,
        {
            "role",
            "source_git_commit",
            "runtime_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
            "runtime_authorization_sha256",
            "launcher_sha256",
        },
        path=path,
    )
    result_identities = _object(result["identities"], path="result.identities")
    if validator in {
        "policy_improvement_smoke_runtime",
        "policy_improvement_full_runtime",
    }:
        expected = {
            "role": "policy-improvement-training",
            "source_git_commit": result_identities["training_source_git_commit"],
            "runtime_sha256": result_identities["training_runtime_sha256"],
            "runtime_profile_sha256": result_identities[
                "training_runtime_profile_sha256"
            ],
            "selected_source_manifest_sha256": result_identities[
                "training_selected_source_manifest_sha256"
            ],
            "runtime_authorization_sha256": result_identities[
                "runtime_authorization_sha256"
            ],
            "launcher_sha256": result_identities["launcher_sha256"],
        }
    elif validator == "policy_improvement_checkpoint_validator":
        expected = {
            "role": "policy-improvement-evaluation",
            "source_git_commit": _available(
                result_identities["evaluation_source_git_commit"],
                path="result.identities.evaluation_source_git_commit",
            ),
            "runtime_sha256": _available(
                result_identities["evaluation_runtime_sha256"],
                path="result.identities.evaluation_runtime_sha256",
            ),
            "runtime_profile_sha256": _available(
                result_identities["evaluation_runtime_profile_sha256"],
                path="result.identities.evaluation_runtime_profile_sha256",
            ),
            "selected_source_manifest_sha256": _available(
                result_identities["evaluation_selected_source_manifest_sha256"],
                path="result.identities.evaluation_selected_source_manifest_sha256",
            ),
            "runtime_authorization_sha256": result_identities[
                "runtime_authorization_sha256"
            ],
            "launcher_sha256": result_identities["launcher_sha256"],
        }
    else:
        raise PolicyImprovementSchemaError("Unknown strict checkpoint validator.")
    if dict(identity) != expected:
        raise PolicyImprovementSchemaError(
            "Checkpoint validation is not bound to the authorized runtime."
        )


def _semantic_checkpoint_validation(
    *,
    checkpoint_validator: Callable[[Mapping[str, object]], Mapping[str, object]],
    sealed_checkpoint: SealedCheckpoint,
    checkpoint_sha256: str,
    validation: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_sha256: str,
    registry_sha256: str,
    registry_row: Mapping[str, object],
    registry_row_sha256: str,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: Path,
    runtime_authorization: Mapping[str, object],
    result: Mapping[str, object],
    environment_interactions: int,
    parent_checkpoint_sha256: str | None,
) -> dict[str, object]:
    """Run the sealed validator and compare computed semantics with claims.

    The validator receives a write-sealed descriptor, never a pathname.  It
    therefore cannot deserialize any bytes other than the ones this module
    authenticated against the immutable generation manifest.
    """

    if sealed_checkpoint.sha256 != checkpoint_sha256:
        raise PolicyImprovementSchemaError(
            "Sealed checkpoint descriptor names another authenticated checkpoint."
        )
    request = {
        "schema_name": "policy_improvement_checkpoint_validation_request_v2",
        "schema_version": 2,
        "checkpoint": sealed_checkpoint.as_request_field(),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": dict(protocol),
        "protocol_sha256": protocol_sha256,
        "registry_row": dict(registry_row),
        "registry_row_sha256": registry_row_sha256,
        "project_root": str(Path(project_root).resolve(strict=True)),
        "dataset_root": str(Path(dataset_root).resolve(strict=True)),
        "evidence_root": str(evidence_root),
        "runtime_authorization": dict(runtime_authorization),
        "run_id": result["run_id"],
        "method_id": result["method_id"],
        "seed": result["seed"],
        "snapshot_kind": validation["snapshot_kind"],
        "environment_interactions": environment_interactions,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "initialization_sha256": result["identities"]["initialization_sha256"],
    }
    if result.get("tier") != "smoke":
        matching_snapshots = [
            snapshot
            for snapshot in result["evaluation_snapshots"]
            if snapshot["snapshot_kind"] == validation["snapshot_kind"]
        ]
        compute_snapshots = [
            snapshot
            for snapshot in result["evaluation_snapshots"]
            if snapshot["snapshot_kind"] == "compute_matched"
        ]
        if len(matching_snapshots) != 1 or len(compute_snapshots) != 1:
            raise PolicyImprovementSchemaError(
                "Full checkpoint validation cannot resolve registered snapshots."
            )
        test_open_identity = _object(
            result["identities"]["test_open_sha256"],
            path="result.identities.test_open_sha256",
        )
        test_open_sha256 = (
            _sha256(test_open_identity.get("value"), path="test_open_sha256")
            if test_open_identity.get("status") == "available"
            else None
        )
        request.update(
            {
                "registry_sha256": registry_sha256,
                "amendment_history_sha256": result["amendment_history_sha256"],
                "runtime_authorization_sha256": result["identities"][
                    "runtime_authorization_sha256"
                ],
                "recurrent_map_applications": _available(
                    matching_snapshots[0]["observed_recurrent_map_applications"],
                    path="snapshot.observed_recurrent_map_applications",
                ),
                "compute_target_recurrent_map_applications": _available(
                    compute_snapshots[0]["target"]["registered_quantity"],
                    path="compute.target.registered_quantity",
                ),
                "test_open_sha256": test_open_sha256,
            }
        )
    computed = _fields(
        checkpoint_validator(request),
        _SEMANTIC_VALIDATION_FIELDS,
        path="checkpoint_semantic_validation",
    )
    nullable_sha = computed["run_identity_sha256"]
    if nullable_sha is not None:
        _sha256(nullable_sha, path="checkpoint_semantic_validation.run_identity")
    for field in (
        "checkpoint_sha256",
        "initialization_sha256",
        "model_state_sha256",
        "method_config_sha256",
        "registered_effective_config_sha256",
        "effective_config_sha256",
        "dataset_manifest_sha256",
        "dataset_provenance_sha256",
    ):
        _sha256(computed[field], path=f"checkpoint_semantic_validation.{field}")
    role_hashes = _object(
        computed["role_state_sha256s"],
        path="checkpoint_semantic_validation.role_state_sha256s",
    )
    if not role_hashes or any(
        not isinstance(role, str)
        or not role
        or _sha256(digest, path=f"checkpoint role {role}") != digest
        for role, digest in role_hashes.items()
    ):
        raise PolicyImprovementSchemaError(
            "Semantic checkpoint role-state inventory is invalid."
        )
    if (
        computed["schema_name"]
        != "policy_improvement_checkpoint_semantic_validation_v1"
        or computed["schema_version"] != 1
        or computed["run_id"] != result["run_id"]
        or computed["method_id"] != result["method_id"]
        or computed["seed"] != result["seed"]
        or computed["snapshot_kind"] != validation["snapshot_kind"]
        or computed["environment_interactions"] != environment_interactions
        or computed["parent_checkpoint_sha256"] != parent_checkpoint_sha256
        or computed["checkpoint_sha256"] != checkpoint_sha256
        or computed["initialization_sha256"]
        != result["identities"]["initialization_sha256"]
        or computed["model_state_sha256"] != validation["model_state_sha256"]
        or dict(role_hashes) != validation["role_state_sha256s"]
        or hashlib.sha256(canonical_json_bytes(role_hashes)).hexdigest()
        != computed["model_state_sha256"]
        or computed["method_config_sha256"]
        != result["identities"]["method_config_sha256"]
        or computed["registered_effective_config_sha256"]
        != result["identities"]["effective_config_sha256"]
        or computed["dataset_manifest_sha256"]
        != result["identities"]["dataset_manifest_sha256"]
        or any(
            computed[field] != 0
            for field in (
                "training_call_delta",
                "evaluation_call_delta",
                "optimizer_step_delta",
            )
        )
    ):
        raise PolicyImprovementSchemaError(
            "Sealed semantic checkpoint validation differs from frozen evidence."
        )
    return dict(computed)


def _authenticate_test_split_state(
    result: Mapping[str, object],
    *,
    authenticated_test_open_sha256: str | None,
) -> None:
    """Require the result's split and TEST_OPEN state to agree, fail-closed."""

    identities = _object(result["identities"], path="result.identities")
    declared = _object(
        identities["test_open_sha256"],
        path="result.identities.test_open_sha256",
    )
    split = result["evaluation_split"]
    if split == "test":
        if authenticated_test_open_sha256 is None:
            raise PolicyImprovementSchemaError(
                "Test-split evidence requires an independently authenticated "
                "TEST_OPEN digest."
            )
        expected = _sha256(
            authenticated_test_open_sha256,
            path="authenticated_test_open_sha256",
        )
        if _available(declared, path="result.identities.test_open_sha256") != expected:
            raise PolicyImprovementSchemaError(
                "Test-split result does not bind the authenticated test opening."
            )
        return
    if split != "validation":
        raise PolicyImprovementSchemaError(
            "Complete evidence must be a validation or test evaluation."
        )
    if authenticated_test_open_sha256 is not None:
        raise PolicyImprovementSchemaError(
            "Validation-split evidence cannot carry an authenticated TEST_OPEN."
        )
    if declared.get("status") != "unavailable":
        raise PolicyImprovementSchemaError(
            "Validation-split evidence cannot claim an opened test population."
        )


def _sealed_semantic_checkpoint_validations(
    pending: Sequence[_PendingCheckpointValidation],
    *,
    checkpoint_validator: Callable[[Mapping[str, object]], Mapping[str, object]],
    protocol: Mapping[str, object],
    protocol_sha256: str,
    registry_sha256: str,
    registry_row: Mapping[str, object],
    registry_row_sha256: str,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: Path,
    runtime_authorization: Mapping[str, object],
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    """Seal every authenticated checkpoint, then validate it semantically.

    Sealing happens strictly after evidence authentication and strictly before
    the first loader call, so the bytes a loader deserializes cannot differ from
    the bytes this module hashed.  Descriptors are closed on both paths; the
    caller never receives one.
    """

    sealed: list[SealedCheckpoint] = []
    try:
        for item in pending:
            try:
                sealed.append(
                    seal_generation_checkpoint(
                        item.generation,
                        item.generation_relative_path,
                        expected_sha256=item.checkpoint_sha256,
                        expected_size_bytes=item.size_bytes,
                    )
                )
            except SealedCheckpointError as exc:
                raise PolicyImprovementSchemaError(
                    "Authenticated checkpoint bytes could not be sealed."
                ) from exc
        return [
            _semantic_checkpoint_validation(
                checkpoint_validator=checkpoint_validator,
                sealed_checkpoint=descriptor,
                checkpoint_sha256=item.checkpoint_sha256,
                validation=item.validation,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                registry_sha256=registry_sha256,
                registry_row=registry_row,
                registry_row_sha256=registry_row_sha256,
                project_root=project_root,
                dataset_root=dataset_root,
                evidence_root=evidence_root,
                runtime_authorization=runtime_authorization,
                result=result,
                environment_interactions=item.environment_interactions,
                parent_checkpoint_sha256=item.parent_checkpoint_sha256,
            )
            for item, descriptor in zip(pending, sealed)
        ]
    finally:
        for descriptor in sealed:
            descriptor.close()


def _canonical_relative(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PolicyImprovementSchemaError(f"{path} is not a canonical relative path.")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise PolicyImprovementSchemaError(f"{path} is not a canonical relative path.")
    return value


def _stable_regular_file(path: Path) -> tuple[bytes, dict[str, object]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before_path.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or before_path.st_nlink != 1
                or opened.st_nlink != 1
                or (before_path.st_dev, before_path.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise PolicyImprovementSchemaError(
                    f"Evidence file {path} is a symlink, alias, or wrong type."
                )
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
                digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            f"Evidence file {path} could not be authenticated."
        ) from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(opened) != identity(after) or identity(after) != identity(after_path):
        raise PolicyImprovementSchemaError(
            f"Evidence file {path} changed while it was read."
        )
    payload = b"".join(chunks)
    return payload, {"bytes": len(payload), "sha256": digest.hexdigest()}


def _private_owner_root(value: str | Path) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise PolicyImprovementSchemaError("Evidence root must be absolute.")
    try:
        status = supplied.lstat()
        root = supplied.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementSchemaError("Evidence root is unavailable.") from exc
    if (
        supplied != root
        or stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise PolicyImprovementSchemaError(
            "Evidence root must be a canonical private owner directory."
        )
    return root


def _require_exact_directory_entries(
    directory: Path,
    expected: set[str],
    *,
    label: str,
) -> None:
    """Reject aliases and every unregistered sibling at an evidence boundary."""

    try:
        status = directory.lstat()
        resolved = directory.resolve(strict=True)
        entries = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise PolicyImprovementSchemaError(f"{label} is unavailable.") from exc
    if (
        directory != resolved
        or stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSchemaError(f"{label} is an alias or wrong type.")
    if entries != expected:
        raise PolicyImprovementSchemaError(
            f"{label} entries differ: expected={sorted(expected)}, "
            f"actual={sorted(entries)}."
        )


def _inventory(directory: Path) -> tuple[dict[str, dict[str, object]], set[str]]:
    files: dict[str, dict[str, object]] = {}
    directories: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for candidate in sorted(directory.rglob("*")):
        relative = candidate.relative_to(directory).as_posix()
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PolicyImprovementSchemaError(
                "Evidence generations cannot contain symlinks."
            )
        inode = (info.st_dev, info.st_ino)
        if inode in seen_inodes:
            raise PolicyImprovementSchemaError(
                "Evidence generation entries alias one inode."
            )
        seen_inodes.add(inode)
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PolicyImprovementSchemaError(
                "Evidence generation entries must be single-link regular files."
            )
        _, identity = _stable_regular_file(candidate)
        files[relative] = identity
    return files, directories


def _load_ascii_json(path: Path) -> object:
    payload, _ = _stable_regular_file(path)
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PolicyImprovementSchemaError(
            f"Evidence JSON {path} is not ASCII."
        ) from exc
    return load_strict_json_bytes(payload)


def _prior_failed_attempt_commitments(
    value: object,
    *,
    path: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise PolicyImprovementSchemaError(f"{path} must be a list.")
    commitments: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        item = _fields(
            raw,
            {
                "segment",
                "attempt_id",
                "generation_manifest_sha256",
                "result_sha256",
            },
            path=f"{path}[{index}]",
        )
        segment = item["segment"]
        attempt_id = item["attempt_id"]
        if (
            segment not in {"prepare", "resume"}
            or not isinstance(attempt_id, str)
            or len(attempt_id) != 32
            or any(character not in "0123456789abcdef" for character in attempt_id)
        ):
            raise PolicyImprovementSchemaError(
                f"{path}[{index}] names an invalid attempt."
            )
        commitments.append(
            {
                "segment": segment,
                "attempt_id": attempt_id,
                "generation_manifest_sha256": _sha256(
                    item["generation_manifest_sha256"],
                    path=f"{path}[{index}].generation_manifest_sha256",
                ),
                "result_sha256": _sha256(
                    item["result_sha256"],
                    path=f"{path}[{index}].result_sha256",
                ),
            }
        )
    if commitments != sorted(
        commitments,
        key=lambda item: (str(item["segment"]), str(item["attempt_id"])),
    ) or len({(item["segment"], item["attempt_id"]) for item in commitments}) != len(
        commitments
    ):
        raise PolicyImprovementSchemaError(
            f"{path} must be unique and canonically ordered."
        )
    return commitments


def _historical_attempt_commitments(
    identities: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "segment": identity["segment"],
            "attempt_id": identity["attempt_id"],
            "generation_manifest_sha256": identity["generation_manifest_sha256"],
            "result_sha256": identity["result_sha256"],
        }
        for identity in identities
    ]


def _authenticate_historical_failed_attempts(
    *,
    root: Path,
    run_root: Path,
    complete_result: Mapping[str, object],
    protocol_sha256: str,
    registry_row_sha256: str,
    expected_environment_interactions: int,
    runtime_authorizations: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Authenticate every immutable failure retained before a successful retry."""

    attempts = run_root / "attempts"
    try:
        attempts.relative_to(root)
        status = attempts.lstat()
        resolved = attempts.resolve(strict=True)
        segment_entries = sorted(attempts.iterdir(), key=lambda item: item.name)
    except (OSError, ValueError) as exc:
        raise PolicyImprovementSchemaError(
            "Historical failed-attempt root is unavailable."
        ) from exc
    if (
        attempts != resolved
        or stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or not segment_entries
    ):
        raise PolicyImprovementSchemaError(
            "Historical failed-attempt root is invalid or empty."
        )
    is_smoke = complete_result.get("tier") == "smoke"
    allowed_segments = {"prepare", "resume"} if is_smoke else {"complete"}
    identities: list[dict[str, object]] = []
    for segment in segment_entries:
        try:
            segment_status = segment.lstat()
            resolved_segment = segment.resolve(strict=True)
            attempt_entries = sorted(segment.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise PolicyImprovementSchemaError(
                "Historical failed-attempt segment is unavailable."
            ) from exc
        if (
            segment.name not in allowed_segments
            or segment != resolved_segment
            or stat.S_ISLNK(segment_status.st_mode)
            or not stat.S_ISDIR(segment_status.st_mode)
            or segment_status.st_uid != os.geteuid()
            or not attempt_entries
        ):
            raise PolicyImprovementSchemaError(
                "Historical failed-attempt segment is invalid or empty."
            )
        for attempt in attempt_entries:
            attempt_id = attempt.name
            try:
                attempt_status = attempt.lstat()
                resolved_attempt = attempt.resolve(strict=True)
            except OSError as exc:
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt generation is unavailable."
                ) from exc
            if (
                len(attempt_id) != 32
                or any(character not in "0123456789abcdef" for character in attempt_id)
                or attempt != resolved_attempt
                or stat.S_ISLNK(attempt_status.st_mode)
                or not stat.S_ISDIR(attempt_status.st_mode)
                or attempt_status.st_uid != os.geteuid()
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt generation is invalid."
                )
            raw_result = validate_result(_load_ascii_json(attempt / "result.json"))
            for field in (
                "protocol_id",
                "run_id",
                "registry_row_sha256",
                "phase",
                "tier",
                "seed",
                "method_id",
                "base_method_id",
                "evaluation_split",
                "n",
                "K",
                "alpha",
                "ablation_variant",
                "primary_policy_variant",
                "applied_config_override",
                "amendment_history_sha256",
            ):
                if raw_result[field] != complete_result[field]:
                    raise PolicyImprovementSchemaError(
                        f"Historical failed attempt {field} differs from its retry."
                    )
            if (
                raw_result["status"] != "failed"
                or raw_result["protocol_sha256"] != protocol_sha256
                or raw_result["registry_row_sha256"] != registry_row_sha256
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt result identity differs."
                )
            raw_identities = _object(
                raw_result["identities"],
                path="historical_failed_attempt.identities",
            )
            complete_identities = _object(
                complete_result["identities"],
                path="complete_result.identities",
            )
            for field in (
                "git_clean",
                "method_config_sha256",
                "effective_config_sha256",
                "dataset_manifest_sha256",
                "train_ordered_records_sha256",
                "evaluation_ordered_records_sha256",
                "test_open_sha256",
            ):
                if raw_identities[field] != complete_identities[field]:
                    raise PolicyImprovementSchemaError(
                        "Historical failed-attempt registered identity "
                        f"{field} differs from its retry."
                    )
            failed_initialization = raw_identities["initialization_sha256"]
            if (
                not (
                    isinstance(failed_initialization, Mapping)
                    and failed_initialization
                    == {
                        "status": "unavailable",
                        "reason": "initialization_not_materialized",
                    }
                )
                and failed_initialization
                != complete_identities["initialization_sha256"]
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt initialization differs from its retry."
                )
            failed_device = raw_identities["device"]
            if (
                not (
                    isinstance(failed_device, Mapping)
                    and failed_device
                    == {
                        "status": "unavailable",
                        "reason": "execution_device_not_materialized",
                    }
                )
                and failed_device != complete_identities["device"]
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt device differs from its retry."
                )
            manifest_fields = {
                "schema_name",
                "schema_version",
                "attempt_id",
                "protocol_sha256",
                "registry_row_sha256",
                "runtime_authorization_sha256",
                "run_id",
                "segment",
                "failure_phase",
                "result",
            }
            if not is_smoke:
                manifest_fields.update({"phase", "tier", "environment_interactions"})
            manifest = _fields(
                _load_ascii_json(attempt / "MANIFEST.json"),
                manifest_fields,
                path="historical_failed_attempt.manifest",
            )
            _, result_identity = _stable_regular_file(attempt / "result.json")
            authorization_digest = _sha256(
                raw_identities["runtime_authorization_sha256"],
                path="historical_failed_attempt.runtime_authorization_sha256",
            )
            if authorization_digest not in runtime_authorizations:
                raise PolicyImprovementSchemaError(
                    "Historical failed attempt lacks its frozen runtime authorization."
                )
            authorization = validate_runtime_authorization(
                runtime_authorizations[authorization_digest]
            )
            if (
                runtime_authorization_sha256(authorization) != authorization_digest
                or authorization["protocol_sha256"] != protocol_sha256
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt runtime authorization differs."
                )
            training_role = authorization["roles"][0]
            expected_runtime_bindings = {
                "producer_git_commit": authorization["producer_git_commit"],
                "producer_manifest_sha256": authorization[
                    "producer_source_manifest_sha256"
                ],
                "launcher_sha256": authorization["launcher_sha256"],
                "training_source_git_commit": training_role["source_git_commit"],
                "training_runtime_sha256": training_role["runtime_sha256"],
                "training_runtime_profile_sha256": training_role[
                    "runtime_profile_sha256"
                ],
                "training_selected_source_manifest_sha256": training_role[
                    "selected_source_manifest_sha256"
                ],
            }
            for field, expected in expected_runtime_bindings.items():
                if raw_identities[field] != expected:
                    raise PolicyImprovementSchemaError(
                        f"Historical failed-attempt identity {field} is not authorized."
                    )
            expected_schema = (
                "policy_improvement_smoke_failed_attempt_v1"
                if is_smoke
                else "policy_improvement_run_failed_attempt_v1"
            )
            if (
                manifest["schema_name"] != expected_schema
                or manifest["schema_version"] != 1
                or manifest["attempt_id"] != attempt_id
                or manifest["protocol_sha256"] != protocol_sha256
                or manifest["registry_row_sha256"] != registry_row_sha256
                or manifest["runtime_authorization_sha256"] != authorization_digest
                or manifest["run_id"] != raw_result["run_id"]
                or manifest["segment"] != segment.name
                or manifest["failure_phase"] != raw_result["failure"]["phase"]
                or manifest["result"]
                != {
                    "path": "result.json",
                    "bytes": result_identity["bytes"],
                    "sha256": result_identity["sha256"],
                }
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt manifest identity differs."
                )
            if not is_smoke and (
                manifest["phase"] != raw_result["phase"]
                or manifest["tier"] != raw_result["tier"]
                or manifest["environment_interactions"]
                != expected_environment_interactions
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed attempt has the wrong phase or budget."
                )
            actual_files, actual_directories = _inventory(attempt)
            if (
                set(actual_files) != {"MANIFEST.json", "result.json"}
                or actual_directories
            ):
                raise PolicyImprovementSchemaError(
                    "Historical failed-attempt inventory differs."
                )
            _, manifest_identity = _stable_regular_file(attempt / "MANIFEST.json")
            identities.append(
                {
                    "attempt_id": attempt_id,
                    "segment": segment.name,
                    "failure_phase": raw_result["failure"]["phase"],
                    "runtime_authorization_sha256": authorization_digest,
                    "generation_manifest_sha256": manifest_identity["sha256"],
                    "result_sha256": result_identity["sha256"],
                }
            )
    return identities


def _authenticate_smoke_parent_segment(
    *,
    root: Path,
    run_id: str,
    result: Mapping[str, object],
    protocol_sha256: str,
    registry_sha256: str,
    registry_row_sha256: str,
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    """Authenticate the prepare checkpoint resumed by a final smoke segment."""

    method_id = result["method_id"]
    parent = root / "runs" / run_id / "segments" / "env_000000016"
    try:
        parent_status = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            f"Smoke result {run_id} has no authenticated prepare segment."
        ) from exc
    if (
        parent != resolved_parent
        or stat.S_ISLNK(parent_status.st_mode)
        or not stat.S_ISDIR(parent_status.st_mode)
        or parent_status.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSchemaError("Smoke prepare segment is an alias.")
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise PolicyImprovementSchemaError(
            "Smoke prepare segment escaped the evidence root."
        ) from exc
    raw_manifest = _object(
        _load_ascii_json(parent / "MANIFEST.json"),
        path=f"generation[{run_id}].parent_manifest",
    )
    parent_schema_version = raw_manifest.get("schema_version")
    parent_manifest_fields = {
        "schema_name",
        "schema_version",
        "protocol_sha256",
        "registry_sha256",
        "registry_row_sha256",
        "run_id",
        "method_id",
        "segment",
        "environment_interactions",
        "parent_checkpoint_sha256",
        "result_status",
        "outputs",
        "storage_bytes",
    }
    if parent_schema_version == 2:
        parent_manifest_fields.add("prior_failed_attempts")
    manifest = _fields(
        raw_manifest,
        parent_manifest_fields,
        path=f"generation[{run_id}].parent_manifest",
    )
    if (
        manifest["schema_name"] != "policy_improvement_smoke_segment_v1"
        or parent_schema_version not in {1, 2}
        or manifest["protocol_sha256"] != protocol_sha256
        or manifest["registry_sha256"] != registry_sha256
        or manifest["registry_row_sha256"] != registry_row_sha256
        or manifest["run_id"] != run_id
        or manifest["method_id"] != method_id
        or manifest["segment"] != "prepare"
        or manifest["environment_interactions"] != 16
        or manifest["parent_checkpoint_sha256"] is not None
        or manifest["result_status"] is not None
    ):
        raise PolicyImprovementSchemaError("Smoke prepare segment identity differs.")
    parent_prior_failed_attempts = (
        _prior_failed_attempt_commitments(
            manifest["prior_failed_attempts"],
            path="parent.prior_failed_attempts",
        )
        if parent_schema_version == 2
        else []
    )
    outputs = _fields(
        manifest["outputs"], {"checkpoint", "files"}, path="parent.outputs"
    )
    registered_files = _object(outputs["files"], path="parent.outputs.files")
    canonical_files: dict[str, dict[str, object]] = {}
    expected_directories: set[str] = set()
    for raw_name, raw_identity in registered_files.items():
        name = _canonical_relative(raw_name, path="parent output path")
        if name == "MANIFEST.json":
            raise PolicyImprovementSchemaError("Parent manifest cannot hash itself.")
        identity = _fields(
            raw_identity, {"bytes", "sha256"}, path=f"parent.output.{name}"
        )
        size = identity["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PolicyImprovementSchemaError("Parent output byte count is invalid.")
        canonical_files[name] = {
            "bytes": size,
            "sha256": _sha256(identity["sha256"], path=f"parent.output.{name}.sha256"),
        }
        parts = PurePosixPath(name).parts
        expected_directories.update(
            PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    actual_files, actual_directories = _inventory(parent)
    if actual_files.pop("MANIFEST.json", None) is None:
        raise PolicyImprovementSchemaError("Parent manifest is missing.")
    if actual_files != canonical_files or actual_directories != expected_directories:
        raise PolicyImprovementSchemaError("Smoke prepare inventory differs.")
    if manifest["storage_bytes"] != sum(
        int(identity["bytes"]) for identity in canonical_files.values()
    ):
        raise PolicyImprovementSchemaError("Smoke prepare storage accounting differs.")
    checkpoint = _fields(
        outputs["checkpoint"], {"path", "bytes", "sha256"}, path="parent.checkpoint"
    )
    checkpoint_name = _canonical_relative(
        checkpoint["path"], path="parent.checkpoint.path"
    )
    if (
        checkpoint_name not in canonical_files
        or dict(checkpoint)
        != {"path": checkpoint_name, **canonical_files[checkpoint_name]}
        or checkpoint["sha256"] != expected_checkpoint_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Final smoke segment does not resume its immutable prepare checkpoint."
        )
    parent_validation_name = "checkpoint_validation.json"
    if parent_validation_name not in canonical_files:
        raise PolicyImprovementSchemaError(
            "Smoke prepare segment lacks strict checkpoint validation."
        )
    parent_validation = _fields(
        _load_ascii_json(parent / parent_validation_name),
        {
            "schema_name",
            "schema_version",
            "validator",
            "validator_execution_identity",
            "run_id",
            "method_id",
            "snapshot_kind",
            "environment_interactions",
            "checkpoint_sha256",
            "parent_checkpoint_sha256",
            "model_state_sha256",
            "role_state_sha256s",
            "strict_resume_validated",
        },
        path="parent.checkpoint_validation",
    )
    if (
        parent_validation["schema_name"]
        != "policy_improvement_checkpoint_validation_v1"
        or parent_validation["schema_version"] != 1
        or parent_validation["validator"] != "policy_improvement_smoke_runtime"
        or parent_validation["run_id"] != run_id
        or parent_validation["method_id"] != method_id
        or parent_validation["snapshot_kind"] != "interaction_matched"
        or parent_validation["environment_interactions"] != 16
        or parent_validation["checkpoint_sha256"] != expected_checkpoint_sha256
        or parent_validation["parent_checkpoint_sha256"] is not None
        or hashlib.sha256(
            canonical_json_bytes(parent_validation["role_state_sha256s"])
        ).hexdigest()
        != parent_validation["model_state_sha256"]
        or parent_validation["strict_resume_validated"] is not True
    ):
        raise PolicyImprovementSchemaError(
            "Smoke prepare strict checkpoint validation differs."
        )
    _validate_checkpoint_validator_identity(
        parent_validation["validator_execution_identity"],
        validator=parent_validation["validator"],
        result=result,
        path="parent.checkpoint_validation.validator_execution_identity",
    )
    _, manifest_identity = _stable_regular_file(parent / "MANIFEST.json")
    return {
        "generation_manifest_sha256": manifest_identity["sha256"],
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_generation": parent,
        "checkpoint_relative_path": checkpoint_name,
        "checkpoint_bytes": int(canonical_files[checkpoint_name]["bytes"]),
        "checkpoint_validation": dict(parent_validation),
        "prior_failed_attempts": parent_prior_failed_attempts,
    }


def authenticate_complete_generation(
    *,
    evidence_root: str | Path,
    result_path: str | Path,
    result: Mapping[str, object],
    protocol_sha256: str,
    registry_sha256: str,
    registry_row_sha256: str,
    expected_environment_interactions: int,
    per_instance_documents: Mapping[str, object],
    checkpoint_validator: Callable[[Mapping[str, object]], Mapping[str, object]],
    protocol: Mapping[str, object],
    registry_row: Mapping[str, object],
    project_root: str | Path,
    dataset_root: str | Path,
    runtime_authorization: Mapping[str, object],
    amendment_history_sha256: str,
    authenticated_test_open_sha256: str | None,
    historical_runtime_authorizations: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Reopen one complete immutable generation and bind every reported artifact.

    ``amendment_history_sha256`` and ``authenticated_test_open_sha256`` are the
    digests the caller authenticated independently.  Requiring them here makes
    amendment history and test-split state part of the same fail-closed gate
    that precedes checkpoint sealing, instead of relying on call ordering.
    """

    root = _private_owner_root(evidence_root)
    _sha256(amendment_history_sha256, path="amendment_history_sha256")
    if result["amendment_history_sha256"] != amendment_history_sha256:
        raise PolicyImprovementSchemaError(
            "Complete result names another authenticated amendment history."
        )
    _authenticate_test_split_state(
        result,
        authenticated_test_open_sha256=authenticated_test_open_sha256,
    )
    current_authorization = validate_runtime_authorization(runtime_authorization)
    current_authorization_digest = runtime_authorization_sha256(current_authorization)
    result_identities = _object(result["identities"], path="result.identities")
    training_role = current_authorization["roles"][0]
    current_runtime_bindings = {
        "runtime_authorization_sha256": current_authorization_digest,
        "producer_git_commit": current_authorization["producer_git_commit"],
        "producer_manifest_sha256": current_authorization[
            "producer_source_manifest_sha256"
        ],
        "launcher_sha256": current_authorization["launcher_sha256"],
        "training_source_git_commit": training_role["source_git_commit"],
        "training_runtime_sha256": training_role["runtime_sha256"],
        "training_runtime_profile_sha256": training_role["runtime_profile_sha256"],
        "training_selected_source_manifest_sha256": training_role[
            "selected_source_manifest_sha256"
        ],
    }
    if current_authorization["protocol_sha256"] != protocol_sha256 or any(
        result_identities[field] != expected
        for field, expected in current_runtime_bindings.items()
    ):
        raise PolicyImprovementSchemaError(
            "Complete result is not bound to its supplied runtime authorization."
        )
    runtime_authorizations: dict[str, Mapping[str, object]] = {
        current_authorization_digest: current_authorization,
    }
    for supplied_digest, supplied_authorization in (
        historical_runtime_authorizations or {}
    ).items():
        digest = _sha256(
            supplied_digest,
            path="historical_runtime_authorizations.key",
        )
        authorization = validate_runtime_authorization(supplied_authorization)
        if runtime_authorization_sha256(authorization) != digest:
            raise PolicyImprovementSchemaError(
                "Historical runtime-authorization key differs from its document."
            )
        existing = runtime_authorizations.get(digest)
        if existing is not None and canonical_json_bytes(
            existing
        ) != canonical_json_bytes(authorization):
            raise PolicyImprovementSchemaError(
                "Runtime-authorization digest resolves to different documents."
            )
        runtime_authorizations[digest] = authorization
    run_id = str(result["run_id"])
    generation = (
        root
        / "runs"
        / run_id
        / "segments"
        / f"env_{expected_environment_interactions:09d}"
    )
    expected_result = generation / "result.json"
    supplied_result = Path(result_path)
    if not supplied_result.is_absolute() or supplied_result != expected_result:
        raise PolicyImprovementSchemaError(
            f"Result {run_id} is not at its registered immutable generation path."
        )
    try:
        generation_status = generation.lstat()
        resolved_generation = generation.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            f"Result generation for {run_id} is unavailable."
        ) from exc
    if (
        generation != resolved_generation
        or stat.S_ISLNK(generation_status.st_mode)
        or not stat.S_ISDIR(generation_status.st_mode)
        or generation_status.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSchemaError(
            f"Result generation for {run_id} is an alias or wrong type."
        )
    try:
        generation.relative_to(root)
    except ValueError as exc:
        raise PolicyImprovementSchemaError(
            "Result generation escaped the evidence root."
        ) from exc

    is_smoke = result.get("tier") == "smoke"
    raw_manifest = _object(
        _load_ascii_json(generation / "MANIFEST.json"),
        path=f"generation[{run_id}].manifest",
    )
    segment_schema_version = raw_manifest.get("schema_version")
    manifest_fields = {
        "schema_name",
        "schema_version",
        "protocol_sha256",
        "registry_sha256",
        "registry_row_sha256",
        "run_id",
        "method_id",
        "segment",
        "environment_interactions",
        "parent_checkpoint_sha256",
        "result_status",
        "outputs",
        "storage_bytes",
    }
    if segment_schema_version == 2:
        manifest_fields.add("prior_failed_attempts")
    manifest = _fields(
        raw_manifest,
        manifest_fields,
        path=f"generation[{run_id}].manifest",
    )
    if (
        manifest["schema_name"] not in _COMPLETE_SEGMENT_SCHEMAS
        or segment_schema_version not in {1, 2}
        or manifest["protocol_sha256"] != protocol_sha256
        or manifest["registry_sha256"] != registry_sha256
        or manifest["registry_row_sha256"] != registry_row_sha256
        or manifest["run_id"] != run_id
        or manifest["method_id"] != result["method_id"]
        or manifest["environment_interactions"] != expected_environment_interactions
        or manifest["result_status"] != "complete"
    ):
        raise PolicyImprovementSchemaError(
            f"Result generation manifest for {run_id} has the wrong identity."
        )
    if is_smoke != (manifest["schema_name"] == "policy_improvement_smoke_segment_v1"):
        raise PolicyImprovementSchemaError(
            "Result tier and immutable segment schema differ."
        )
    run_root = root / "runs" / run_id
    attempts_path = run_root / "attempts"
    try:
        attempts_path.lstat()
    except FileNotFoundError:
        has_historical_attempts = False
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            "Run attempt inventory cannot be authenticated."
        ) from exc
    else:
        has_historical_attempts = True
    _require_exact_directory_entries(
        run_root,
        {"segments", "attempts"} if has_historical_attempts else {"segments"},
        label="Complete run root",
    )
    historical_attempts = (
        _authenticate_historical_failed_attempts(
            root=root,
            run_root=run_root,
            complete_result=result,
            protocol_sha256=protocol_sha256,
            registry_row_sha256=registry_row_sha256,
            expected_environment_interactions=expected_environment_interactions,
            runtime_authorizations=runtime_authorizations,
        )
        if has_historical_attempts
        else []
    )
    committed_attempts = (
        _prior_failed_attempt_commitments(
            manifest["prior_failed_attempts"],
            path="generation.prior_failed_attempts",
        )
        if segment_schema_version == 2
        else []
    )
    observed_attempts = _historical_attempt_commitments(historical_attempts)
    if committed_attempts != observed_attempts:
        raise PolicyImprovementSchemaError(
            "Complete generation does not bind its exact prior failed-attempt set."
        )
    segments_path = run_root / "segments"
    expected_segments = (
        {"env_000000016", "env_000000032"}
        if is_smoke
        else {f"env_{expected_environment_interactions:09d}"}
    )
    _require_exact_directory_entries(
        segments_path,
        expected_segments,
        label="Complete segment inventory",
    )
    parent_identity: dict[str, object] | None = None
    if is_smoke:
        parent_checkpoint_sha256 = _sha256(
            manifest["parent_checkpoint_sha256"],
            path="generation.parent_checkpoint_sha256",
        )
        if manifest["segment"] != "resume":
            raise PolicyImprovementSchemaError(
                "Final smoke generation must be the resume segment."
            )
        parent_identity = _authenticate_smoke_parent_segment(
            root=root,
            run_id=run_id,
            result=result,
            protocol_sha256=protocol_sha256,
            registry_sha256=registry_sha256,
            registry_row_sha256=registry_row_sha256,
            expected_checkpoint_sha256=parent_checkpoint_sha256,
        )
        parent_attempts = parent_identity["prior_failed_attempts"]
        if not isinstance(parent_attempts, list) or any(
            attempt not in committed_attempts for attempt in parent_attempts
        ):
            raise PolicyImprovementSchemaError(
                "Final smoke generation omits a failure bound by its parent."
            )
    elif (
        manifest["segment"] != "complete"
        or manifest["parent_checkpoint_sha256"] is not None
    ):
        raise PolicyImprovementSchemaError(
            "Non-smoke result must use one complete, parentless generation."
        )
    outputs = _fields(
        manifest["outputs"], {"checkpoint", "files"}, path="generation.outputs"
    )
    registered_files = _object(outputs["files"], path="generation.outputs.files")
    canonical_files: dict[str, dict[str, object]] = {}
    expected_directories: set[str] = set()
    for raw_name, raw_identity in registered_files.items():
        name = _canonical_relative(raw_name, path="generation output path")
        if name == "MANIFEST.json":
            raise PolicyImprovementSchemaError(
                "Generation manifest cannot hash itself."
            )
        identity = _fields(raw_identity, {"bytes", "sha256"}, path=f"output.{name}")
        size = identity["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PolicyImprovementSchemaError(f"output.{name}.bytes is invalid.")
        canonical_files[name] = {
            "bytes": size,
            "sha256": _sha256(identity["sha256"], path=f"output.{name}.sha256"),
        }
        parts = PurePosixPath(name).parts
        expected_directories.update(
            PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    actual_files, actual_directories = _inventory(generation)
    manifest_identity = actual_files.pop("MANIFEST.json", None)
    if manifest_identity is None or actual_files != canonical_files:
        raise PolicyImprovementSchemaError(
            f"Result generation files for {run_id} differ from the manifest."
        )
    if actual_directories != expected_directories:
        raise PolicyImprovementSchemaError(
            f"Result generation directories for {run_id} differ from the manifest."
        )
    if manifest["storage_bytes"] != sum(
        int(identity["bytes"]) for identity in canonical_files.values()
    ):
        raise PolicyImprovementSchemaError("Generation storage accounting differs.")

    checkpoint = _fields(
        outputs["checkpoint"], {"path", "bytes", "sha256"}, path="checkpoint"
    )
    checkpoint_name = _canonical_relative(checkpoint["path"], path="checkpoint.path")
    if dict(checkpoint) != {
        "path": checkpoint_name,
        **canonical_files[checkpoint_name],
    }:
        raise PolicyImprovementSchemaError(
            "Checkpoint identity differs from generation inventory."
        )
    checkpoint_sha256 = _sha256(checkpoint["sha256"], path="checkpoint.sha256")
    if (
        _available(result["artifacts"]["checkpoint"], path="artifacts.checkpoint")
        != checkpoint_sha256
        or _available(
            result["identities"]["checkpoint_sha256"], path="identities.checkpoint"
        )
        != checkpoint_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Result checkpoint does not bind checkpoint bytes."
        )
    if parent_identity is not None:
        available_snapshots = [
            snapshot
            for snapshot in result["evaluation_snapshots"]
            if snapshot["status"] == "available"
        ]
        expected_lineage_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_name": "policy_improvement_smoke_checkpoint_lineage_v1",
                    "run_id": run_id,
                    "initialization_sha256": result["identities"][
                        "initialization_sha256"
                    ],
                    "parent_checkpoint_sha256": parent_identity["checkpoint_sha256"],
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
        ).hexdigest()
        if any(
            _available(
                snapshot["checkpoint_lineage_sha256"],
                path="snapshot.checkpoint_lineage_sha256",
            )
            != expected_lineage_sha256
            for snapshot in available_snapshots
        ):
            raise PolicyImprovementSchemaError(
                "Smoke checkpoint lineage differs from its immutable segments."
            )
    checkpoint_files = {
        name: identity
        for name, identity in canonical_files.items()
        if name.startswith("checkpoints/")
    }
    snapshot_checkpoint_files: dict[str, tuple[str, int]] = {}
    for snapshot in result["evaluation_snapshots"]:
        if snapshot["status"] != "available":
            continue
        snapshot_checkpoint = _sha256(
            _available(
                snapshot["checkpoint_sha256"],
                path="snapshot.checkpoint_sha256",
            ),
            path="snapshot.checkpoint_sha256.value",
        )
        matching_files = [
            name
            for name, identity in checkpoint_files.items()
            if identity["sha256"] == snapshot_checkpoint
        ]
        if len(matching_files) != 1:
            raise PolicyImprovementSchemaError(
                "Every available snapshot must bind one immutable checkpoint file."
            )
        snapshot_checkpoint_files[str(snapshot["snapshot_kind"])] = (
            matching_files[0],
            int(checkpoint_files[matching_files[0]]["bytes"]),
        )

    stored_result = _load_ascii_json(expected_result)
    if canonical_json_bytes(stored_result) != canonical_json_bytes(result):
        raise PolicyImprovementSchemaError(
            "Supplied result differs from immutable result.json."
        )
    run_manifest_path = generation / "RUN_MANIFEST.json"
    _, run_manifest_identity = _stable_regular_file(run_manifest_path)
    if (
        _available(result["artifacts"]["run_manifest"], path="artifacts.run_manifest")
        != run_manifest_identity["sha256"]
    ):
        raise PolicyImprovementSchemaError(
            "Result run-manifest digest differs from bytes."
        )
    run_manifest = _object(_load_ascii_json(run_manifest_path), path="run_manifest")
    expected_run_fields = {
        "run_id": run_id,
        "method_id": result["method_id"],
        "environment_interactions": expected_environment_interactions,
        "protocol_sha256": protocol_sha256,
        "registry_row_sha256": registry_row_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": _available(
            result["identities"]["model_state_sha256"], path="identities.model_state"
        ),
    }
    for field, expected in expected_run_fields.items():
        if run_manifest.get(field) != expected:
            raise PolicyImprovementSchemaError(
                f"Run manifest field {field} differs from the audited result."
            )
    model_inventory_path = generation / "model_state_inventory.json"
    _, model_inventory_identity = _stable_regular_file(model_inventory_path)
    model_inventory_sha256 = _sha256(
        run_manifest.get("model_state_inventory_sha256"),
        path="run_manifest.model_state_inventory_sha256",
    )
    if (
        model_inventory_identity["sha256"] != model_inventory_sha256
        or _available(
            result["artifacts"]["model_state_inventory"],
            path="artifacts.model_state_inventory",
        )
        != model_inventory_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Model-state inventory digest differs from immutable bytes."
        )
    model_inventory = _fields(
        _load_ascii_json(model_inventory_path),
        {
            "schema_name",
            "run_id",
            "method_id",
            "model_state_sha256",
            "role_state_sha256s",
            "snapshot_state_bindings",
        },
        path="model_state_inventory",
    )
    if (
        model_inventory["schema_name"] != "policy_improvement_model_state_inventory_v1"
        or model_inventory["run_id"] != run_id
        or model_inventory["method_id"] != result["method_id"]
        or model_inventory["model_state_sha256"]
        != expected_run_fields["model_state_sha256"]
        or run_manifest.get("model_state_sha256s")
        != model_inventory["role_state_sha256s"]
        or hashlib.sha256(
            canonical_json_bytes(model_inventory["role_state_sha256s"])
        ).hexdigest()
        != model_inventory["model_state_sha256"]
    ):
        raise PolicyImprovementSchemaError(
            "Model-state inventory differs from the run manifest and result."
        )
    raw_snapshot_bindings = model_inventory["snapshot_state_bindings"]
    if not isinstance(raw_snapshot_bindings, list):
        raise PolicyImprovementSchemaError(
            "Model-state snapshot bindings must be a list."
        )
    snapshot_bindings: dict[str, Mapping[str, object]] = {}
    for index, raw_binding in enumerate(raw_snapshot_bindings):
        binding = _fields(
            raw_binding,
            {
                "snapshot_kind",
                "checkpoint_sha256",
                "model_state_sha256",
                "checkpoint_validation_sha256",
            },
            path=f"model_state_inventory.snapshot_state_bindings[{index}]",
        )
        kind = binding["snapshot_kind"]
        if not isinstance(kind, str) or kind in snapshot_bindings:
            raise PolicyImprovementSchemaError(
                "Model-state snapshot binding kinds are invalid or duplicated."
            )
        for field in (
            "checkpoint_sha256",
            "model_state_sha256",
            "checkpoint_validation_sha256",
        ):
            _sha256(binding[field], path=f"snapshot_binding.{kind}.{field}")
        snapshot_bindings[kind] = binding

    available_snapshot_kinds = {
        str(snapshot["snapshot_kind"])
        for snapshot in result["evaluation_snapshots"]
        if snapshot["status"] == "available"
    }
    if set(snapshot_bindings) != available_snapshot_kinds:
        raise PolicyImprovementSchemaError(
            "Model-state bindings differ from available evaluation snapshots."
        )
    validation_digests: set[str] = set()
    pending_validations: list[_PendingCheckpointValidation] = []
    if parent_identity is not None:
        parent_validation = _object(
            parent_identity["checkpoint_validation"],
            path="parent.checkpoint_validation",
        )
        parent_generation = parent_identity["checkpoint_generation"]
        assert isinstance(parent_generation, Path)
        pending_validations.append(
            _PendingCheckpointValidation(
                generation=parent_generation,
                generation_relative_path=str(
                    parent_identity["checkpoint_relative_path"]
                ),
                checkpoint_sha256=str(parent_identity["checkpoint_sha256"]),
                size_bytes=int(parent_identity["checkpoint_bytes"]),
                validation=parent_validation,
                environment_interactions=16,
                parent_checkpoint_sha256=None,
            )
        )
    for snapshot in result["evaluation_snapshots"]:
        if snapshot["status"] != "available":
            continue
        kind = str(snapshot["snapshot_kind"])
        binding = snapshot_bindings[kind]
        snapshot_checkpoint_sha256 = _sha256(
            _available(
                snapshot["checkpoint_sha256"], path="snapshot.checkpoint_sha256"
            ),
            path="snapshot.checkpoint_sha256.value",
        )
        snapshot_model_sha256 = _sha256(
            _available(
                snapshot["model_state_sha256"], path="snapshot.model_state_sha256"
            ),
            path="snapshot.model_state_sha256.value",
        )
        validation_sha256 = str(binding["checkpoint_validation_sha256"])
        if (
            binding["checkpoint_sha256"] != snapshot_checkpoint_sha256
            or binding["model_state_sha256"] != snapshot_model_sha256
            or validation_sha256 in validation_digests
        ):
            raise PolicyImprovementSchemaError(
                "Snapshot checkpoint/model binding differs or aliases another validation."
            )
        validation_digests.add(validation_sha256)
        validation_paths = [
            generation / name
            for name, identity in canonical_files.items()
            if identity["sha256"] == validation_sha256
            and name.endswith("checkpoint_validation.json")
        ]
        if len(validation_paths) != 1:
            raise PolicyImprovementSchemaError(
                "Snapshot strict checkpoint validation artifact is missing or ambiguous."
            )
        validation = _fields(
            _load_ascii_json(validation_paths[0]),
            {
                "schema_name",
                "schema_version",
                "validator",
                "validator_execution_identity",
                "run_id",
                "method_id",
                "snapshot_kind",
                "environment_interactions",
                "checkpoint_sha256",
                "parent_checkpoint_sha256",
                "model_state_sha256",
                "role_state_sha256s",
                "strict_resume_validated",
            },
            path=f"checkpoint_validation.{kind}",
        )
        observed_interactions = _available(
            snapshot["observed_environment_interactions"],
            path="snapshot.observed_environment_interactions",
        )
        validation_parent = validation["parent_checkpoint_sha256"]
        if validation_parent is not None:
            _sha256(
                validation_parent,
                path=f"checkpoint_validation.{kind}.parent_checkpoint_sha256",
            )
        if (
            validation["schema_name"] != "policy_improvement_checkpoint_validation_v1"
            or validation["schema_version"] != 1
            or validation["validator"]
            not in {
                "policy_improvement_smoke_runtime",
                "policy_improvement_full_runtime",
                "policy_improvement_checkpoint_validator",
            }
            or validation["run_id"] != run_id
            or validation["method_id"] != result["method_id"]
            or validation["snapshot_kind"] != kind
            or validation["environment_interactions"] != observed_interactions
            or validation["checkpoint_sha256"] != snapshot_checkpoint_sha256
            or validation["model_state_sha256"] != snapshot_model_sha256
            or (
                is_smoke
                and validation["parent_checkpoint_sha256"]
                != (
                    manifest["parent_checkpoint_sha256"]
                    if kind == "interaction_matched"
                    else None
                )
            )
            or hashlib.sha256(
                canonical_json_bytes(validation["role_state_sha256s"])
            ).hexdigest()
            != validation["model_state_sha256"]
            or validation["strict_resume_validated"] is not True
        ):
            raise PolicyImprovementSchemaError(
                "Strict checkpoint validation differs from its result snapshot."
            )
        _validate_checkpoint_validator_identity(
            validation["validator_execution_identity"],
            validator=validation["validator"],
            result=result,
            path=f"checkpoint_validation.{kind}.validator_execution_identity",
        )
        snapshot_relative_path, snapshot_size_bytes = snapshot_checkpoint_files[kind]
        pending_validations.append(
            _PendingCheckpointValidation(
                generation=generation,
                generation_relative_path=snapshot_relative_path,
                checkpoint_sha256=snapshot_checkpoint_sha256,
                size_bytes=snapshot_size_bytes,
                validation=validation,
                environment_interactions=int(observed_interactions),
                parent_checkpoint_sha256=(
                    str(validation["parent_checkpoint_sha256"])
                    if validation["parent_checkpoint_sha256"] is not None
                    else None
                ),
            )
        )
    primary_validation_sha256 = _available(
        result["artifacts"]["checkpoint_validation"],
        path="artifacts.checkpoint_validation",
    )
    interaction_binding = snapshot_bindings.get("interaction_matched")
    if (
        interaction_binding is None
        or interaction_binding["checkpoint_validation_sha256"]
        != primary_validation_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Top-level checkpoint validation does not bind interaction evidence."
        )

    for snapshot in result["evaluation_snapshots"]:
        if snapshot["status"] != "available":
            continue
        for evaluation in snapshot["policy_evaluations"]:
            variant = str(evaluation["policy_variant"])
            relative_candidates = (
                f"evaluations/{variant}.json",
                (f"evaluations/{snapshot['snapshot_kind']}/" f"{variant}.json"),
            )
            aggregate_paths = [
                generation / name
                for name in relative_candidates
                if name in canonical_files
            ]
            if len(aggregate_paths) != 1:
                raise PolicyImprovementSchemaError(
                    f"Evaluation artifact path for {run_id}/{variant} is ambiguous or missing."
                )
            aggregate = _object(
                _load_ascii_json(aggregate_paths[0]), path="evaluation_artifact"
            )
            if (
                aggregate.get("primary") != evaluation["primary"]
                or aggregate.get("secondary") != evaluation["secondary"]
            ):
                raise PolicyImprovementSchemaError(
                    "Evaluation artifact differs from reported primary/secondary metrics."
                )
            semantic_aggregate = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "primary": evaluation["primary"],
                        "secondary": evaluation["secondary"],
                    }
                )
            ).hexdigest()
            if (
                _available(
                    evaluation["evaluation_artifact_sha256"],
                    path="evaluation.evaluation_artifact_sha256",
                )
                != semantic_aggregate
            ):
                raise PolicyImprovementSchemaError(
                    "Evaluation semantic digest differs."
                )
            per_instance_path = aggregate_paths[0].with_name(
                f"{variant}.per_instance.json"
            )
            per_instance = _load_ascii_json(per_instance_path)
            per_instance_digest = hashlib.sha256(
                canonical_json_bytes(per_instance)
            ).hexdigest()
            expected_digest = _available(
                evaluation["per_instance_artifact_sha256"],
                path="evaluation.per_instance_artifact_sha256",
            )
            if (
                per_instance_digest != expected_digest
                or expected_digest not in per_instance_documents
                or canonical_json_bytes(per_instance_documents[expected_digest])
                != canonical_json_bytes(per_instance)
            ):
                raise PolicyImprovementSchemaError(
                    "Per-instance artifact differs from its immutable generation."
                )
    # Every identity a checkpoint could be judged against is now authenticated:
    # runtime authorization and producer identity, protocol/registry/amendment
    # digests, the complete immutable generation and its outer manifest, the
    # result and RUN_MANIFEST, the model-state inventory and per-snapshot
    # checkpoint validations, snapshot kind/progress/lineage, and test-split
    # state.  Only now may checkpoint bytes be sealed and deserialized.
    presealed_files, presealed_directories = _inventory(generation)
    if (
        presealed_files.pop("MANIFEST.json", None) is None
        or presealed_files != canonical_files
        or presealed_directories != expected_directories
    ):
        raise PolicyImprovementSchemaError(
            "Result generation changed before checkpoint sealing."
        )
    semantic_validations = _sealed_semantic_checkpoint_validations(
        pending_validations,
        checkpoint_validator=checkpoint_validator,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        registry_row=registry_row,
        registry_row_sha256=registry_row_sha256,
        project_root=project_root,
        dataset_root=dataset_root,
        evidence_root=root,
        runtime_authorization=runtime_authorization,
        result=result,
    )
    final_files, final_directories = _inventory(generation)
    manifest_file_identity = final_files.pop("MANIFEST.json", None)
    if (
        manifest_file_identity is None
        or final_files != canonical_files
        or final_directories != expected_directories
    ):
        raise PolicyImprovementSchemaError(
            "Result generation changed during semantic checkpoint validation."
        )
    if parent_identity is not None:
        parent_after = _authenticate_smoke_parent_segment(
            root=root,
            run_id=run_id,
            result=result,
            protocol_sha256=protocol_sha256,
            registry_sha256=registry_sha256,
            registry_row_sha256=registry_row_sha256,
            expected_checkpoint_sha256=str(parent_identity["checkpoint_sha256"]),
        )
        if (
            parent_after["generation_manifest_sha256"]
            != parent_identity["generation_manifest_sha256"]
            or parent_after["checkpoint_generation"]
            != parent_identity["checkpoint_generation"]
            or parent_after["checkpoint_relative_path"]
            != parent_identity["checkpoint_relative_path"]
            or parent_after["checkpoint_bytes"] != parent_identity["checkpoint_bytes"]
            or canonical_json_bytes(parent_after["checkpoint_validation"])
            != canonical_json_bytes(parent_identity["checkpoint_validation"])
            or canonical_json_bytes(parent_after["prior_failed_attempts"])
            != canonical_json_bytes(parent_identity["prior_failed_attempts"])
        ):
            raise PolicyImprovementSchemaError(
                "Smoke prepare segment changed during semantic validation."
            )
    _require_exact_directory_entries(
        run_root,
        {"segments", "attempts"} if has_historical_attempts else {"segments"},
        label="Complete run root",
    )
    historical_attempts_after = (
        _authenticate_historical_failed_attempts(
            root=root,
            run_root=run_root,
            complete_result=result,
            protocol_sha256=protocol_sha256,
            registry_row_sha256=registry_row_sha256,
            expected_environment_interactions=expected_environment_interactions,
            runtime_authorizations=runtime_authorizations,
        )
        if has_historical_attempts
        else []
    )
    if canonical_json_bytes(historical_attempts_after) != canonical_json_bytes(
        historical_attempts
    ):
        raise PolicyImprovementSchemaError(
            "Historical failed-attempt inventory changed during validation."
        )
    return {
        "generation_path": str(generation),
        "generation_manifest_sha256": manifest_file_identity["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "run_manifest_sha256": run_manifest_identity["sha256"],
        "model_state_inventory_sha256": model_inventory_sha256,
        "semantic_validations": semantic_validations,
        "parent_generation_manifest_sha256": (
            None
            if parent_identity is None
            else parent_identity["generation_manifest_sha256"]
        ),
        "historical_failed_attempts": historical_attempts,
    }


def authenticate_failed_attempt(
    *,
    evidence_root: str | Path,
    result: Mapping[str, object],
    protocol_sha256: str,
    registry_row_sha256: str,
    runtime_authorization_sha256: str,
    expected_environment_interactions: int,
) -> dict[str, object]:
    """Find and authenticate the one immutable failed attempt for a result."""

    root = _private_owner_root(evidence_root)
    run_id = str(result["run_id"])
    run_root = root / "runs" / run_id
    _require_exact_directory_entries(
        run_root,
        {"attempts"},
        label="Failed run root",
    )
    attempts = run_root / "attempts"
    try:
        attempts_status = attempts.lstat()
        resolved_attempts = attempts.resolve(strict=True)
        if (
            attempts != resolved_attempts
            or stat.S_ISLNK(attempts_status.st_mode)
            or not stat.S_ISDIR(attempts_status.st_mode)
            or attempts_status.st_uid != os.geteuid()
        ):
            raise PolicyImprovementSchemaError("Failed-attempt root is invalid.")
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            f"Failed result {run_id} has no immutable attempt evidence."
        ) from exc
    try:
        attempts.relative_to(root)
    except ValueError as exc:
        raise PolicyImprovementSchemaError(
            "Failed-attempt root escaped the evidence root."
        ) from exc
    attempt_directories: list[Path] = []
    try:
        segment_entries = list(attempts.iterdir())
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            "Failed-attempt root is unreadable."
        ) from exc
    is_smoke = result.get("tier") == "smoke"
    allowed_segments = {"prepare", "resume"} if is_smoke else {"complete"}
    if (is_smoke and len(segment_entries) != 1) or not segment_entries:
        raise PolicyImprovementSchemaError(
            "Failed result has an invalid failed-segment inventory."
        )
    for segment in segment_entries:
        try:
            segment_status = segment.lstat()
            resolved_segment = segment.resolve(strict=True)
        except OSError as exc:
            raise PolicyImprovementSchemaError(
                "Failed-attempt segment is unavailable."
            ) from exc
        if (
            segment.name not in allowed_segments
            or segment != resolved_segment
            or stat.S_ISLNK(segment_status.st_mode)
            or not stat.S_ISDIR(segment_status.st_mode)
            or segment_status.st_uid != os.geteuid()
        ):
            raise PolicyImprovementSchemaError("Failed-attempt segment is invalid.")
        for attempt in segment.iterdir():
            try:
                attempt_status = attempt.lstat()
                resolved_attempt = attempt.resolve(strict=True)
            except OSError as exc:
                raise PolicyImprovementSchemaError(
                    "Failed-attempt generation is unavailable."
                ) from exc
            if (
                attempt != resolved_attempt
                or stat.S_ISLNK(attempt_status.st_mode)
                or not stat.S_ISDIR(attempt_status.st_mode)
                or attempt_status.st_uid != os.geteuid()
            ):
                raise PolicyImprovementSchemaError(
                    "Failed-attempt generation is invalid."
                )
            attempt_directories.append(attempt)
    if (is_smoke and len(attempt_directories) != 1) or not attempt_directories:
        raise PolicyImprovementSchemaError(
            f"Failed result {run_id} has an invalid immutable-attempt inventory."
        )
    matches: list[tuple[Path, Mapping[str, object], dict[str, object]]] = []
    for generation in attempt_directories:
        candidate = generation / "result.json"
        try:
            loaded = validate_result(_load_ascii_json(candidate))
        except PolicyImprovementSchemaError:
            raise PolicyImprovementSchemaError(
                "Failed-attempt result is missing or invalid."
            )
        matches_supplied_result = canonical_json_bytes(loaded) == canonical_json_bytes(
            result
        )
        if (
            loaded["status"] != "failed"
            or loaded["run_id"] != run_id
            or loaded["protocol_sha256"] != protocol_sha256
            or loaded["registry_row_sha256"] != registry_row_sha256
            or loaded["identities"]["runtime_authorization_sha256"]
            != runtime_authorization_sha256
        ):
            raise PolicyImprovementSchemaError(
                "Failed-attempt result identity differs from its run."
            )
        attempt_id = generation.name
        if len(attempt_id) != 32 or any(
            character not in "0123456789abcdef" for character in attempt_id
        ):
            raise PolicyImprovementSchemaError("Failed-attempt ID is invalid.")
        manifest_fields = {
            "schema_name",
            "schema_version",
            "attempt_id",
            "protocol_sha256",
            "registry_row_sha256",
            "runtime_authorization_sha256",
            "run_id",
            "segment",
            "failure_phase",
            "result",
        }
        if not is_smoke:
            manifest_fields.update({"phase", "tier", "environment_interactions"})
        manifest = _fields(
            _load_ascii_json(generation / "MANIFEST.json"),
            manifest_fields,
            path="failed_attempt.manifest",
        )
        _, result_identity = _stable_regular_file(candidate)
        expected_schema = (
            "policy_improvement_smoke_failed_attempt_v1"
            if is_smoke
            else "policy_improvement_run_failed_attempt_v1"
        )
        if (
            manifest["schema_name"] != expected_schema
            or manifest["schema_version"] != 1
            or manifest["attempt_id"] != attempt_id
            or manifest["protocol_sha256"] != protocol_sha256
            or manifest["registry_row_sha256"] != registry_row_sha256
            or manifest["runtime_authorization_sha256"] != runtime_authorization_sha256
            or manifest["run_id"] != run_id
            or manifest["segment"] != generation.parent.name
            or manifest["failure_phase"] != loaded["failure"]["phase"]
            or manifest["result"]
            != {
                "path": "result.json",
                "bytes": result_identity["bytes"],
                "sha256": result_identity["sha256"],
            }
        ):
            raise PolicyImprovementSchemaError("Failed-attempt identity differs.")
        if not is_smoke and (
            manifest["phase"] != loaded["phase"]
            or manifest["tier"] != loaded["tier"]
            or manifest["environment_interactions"] != expected_environment_interactions
        ):
            raise PolicyImprovementSchemaError(
                "Failed run attempt phase, tier, or budget differs."
            )
        actual_files, actual_directories = _inventory(generation)
        if set(actual_files) != {"MANIFEST.json", "result.json"} or actual_directories:
            raise PolicyImprovementSchemaError("Failed-attempt inventory differs.")
        if matches_supplied_result:
            matches.append((generation, manifest, result_identity))
    if not matches or (is_smoke and len(matches) != 1):
        raise PolicyImprovementSchemaError(
            f"Failed result {run_id} must resolve to exactly one immutable attempt."
        )
    generation, _, result_identity = max(matches, key=lambda item: item[0].name)
    _, manifest_identity = _stable_regular_file(generation / "MANIFEST.json")
    return {
        "generation_path": str(generation),
        "generation_manifest_sha256": manifest_identity["sha256"],
        "result_sha256": result_identity["sha256"],
    }
