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
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
    load_strict_json_bytes,
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
    if validator == "policy_improvement_smoke_runtime":
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
    checkpoint_path: str,
    checkpoint_sha256: str,
    validation: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_sha256: str,
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
    """Run the sealed validator and compare computed semantics with claims."""

    request = {
        "schema_name": "policy_improvement_checkpoint_validation_request_v1",
        "schema_version": 1,
        "checkpoint_path": checkpoint_path,
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
    manifest = _fields(
        _load_ascii_json(parent / "MANIFEST.json"),
        {
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
        },
        path=f"generation[{run_id}].parent_manifest",
    )
    if (
        manifest["schema_name"] != "policy_improvement_smoke_segment_v1"
        or manifest["schema_version"] != 1
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
        "checkpoint_path": str(parent / checkpoint_name),
        "checkpoint_validation": dict(parent_validation),
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
) -> dict[str, object]:
    """Reopen one complete immutable generation and bind every reported artifact."""

    root = _private_owner_root(evidence_root)
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

    manifest = _fields(
        _load_ascii_json(generation / "MANIFEST.json"),
        {
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
        },
        path=f"generation[{run_id}].manifest",
    )
    if (
        manifest["schema_name"] not in _COMPLETE_SEGMENT_SCHEMAS
        or manifest["schema_version"] != 1
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
    is_smoke = result.get("tier") == "smoke"
    if is_smoke != (manifest["schema_name"] == "policy_improvement_smoke_segment_v1"):
        raise PolicyImprovementSchemaError(
            "Result tier and immutable segment schema differ."
        )
    run_root = root / "runs" / run_id
    _require_exact_directory_entries(
        run_root,
        {"segments"},
        label="Complete run root",
    )
    attempts_path = run_root / "attempts"
    try:
        attempts_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            "Run attempt inventory cannot be authenticated."
        ) from exc
    else:
        raise PolicyImprovementSchemaError(
            "A complete result cannot hide an earlier failed attempt."
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
    snapshot_checkpoint_paths: dict[str, str] = {}
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
        snapshot_checkpoint_paths[str(snapshot["snapshot_kind"])] = str(
            generation / matching_files[0]
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
    semantic_validations: list[dict[str, object]] = []
    if parent_identity is not None:
        parent_validation = _object(
            parent_identity["checkpoint_validation"],
            path="parent.checkpoint_validation",
        )
        semantic_validations.append(
            _semantic_checkpoint_validation(
                checkpoint_validator=checkpoint_validator,
                checkpoint_path=str(parent_identity["checkpoint_path"]),
                checkpoint_sha256=str(parent_identity["checkpoint_sha256"]),
                validation=parent_validation,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                registry_row=registry_row,
                registry_row_sha256=registry_row_sha256,
                project_root=project_root,
                dataset_root=dataset_root,
                evidence_root=root,
                runtime_authorization=runtime_authorization,
                result=result,
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
        if (
            validation["schema_name"] != "policy_improvement_checkpoint_validation_v1"
            or validation["schema_version"] != 1
            or validation["validator"]
            not in {
                "policy_improvement_smoke_runtime",
                "policy_improvement_checkpoint_validator",
            }
            or validation["run_id"] != run_id
            or validation["method_id"] != result["method_id"]
            or validation["snapshot_kind"] != kind
            or validation["environment_interactions"] != observed_interactions
            or validation["checkpoint_sha256"] != snapshot_checkpoint_sha256
            or validation["model_state_sha256"] != snapshot_model_sha256
            or validation["parent_checkpoint_sha256"]
            != (
                manifest["parent_checkpoint_sha256"]
                if kind == "interaction_matched"
                else None
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
        semantic_validations.append(
            _semantic_checkpoint_validation(
                checkpoint_validator=checkpoint_validator,
                checkpoint_path=snapshot_checkpoint_paths[kind],
                checkpoint_sha256=snapshot_checkpoint_sha256,
                validation=validation,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                registry_row=registry_row,
                registry_row_sha256=registry_row_sha256,
                project_root=project_root,
                dataset_root=dataset_root,
                evidence_root=root,
                runtime_authorization=runtime_authorization,
                result=result,
                environment_interactions=int(observed_interactions),
                parent_checkpoint_sha256=(
                    str(manifest["parent_checkpoint_sha256"])
                    if kind == "interaction_matched"
                    and manifest["parent_checkpoint_sha256"] is not None
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
            or parent_after["checkpoint_path"] != parent_identity["checkpoint_path"]
            or canonical_json_bytes(parent_after["checkpoint_validation"])
            != canonical_json_bytes(parent_identity["checkpoint_validation"])
        ):
            raise PolicyImprovementSchemaError(
                "Smoke prepare segment changed during semantic validation."
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
    if len(segment_entries) != 1:
        raise PolicyImprovementSchemaError(
            "Failed result must contain exactly one failed segment."
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
    if len(attempt_directories) != 1:
        raise PolicyImprovementSchemaError(
            f"Failed result {run_id} must have exactly one immutable attempt."
        )
    matches: list[tuple[Path, Mapping[str, object], dict[str, object]]] = []
    for generation in attempt_directories:
        candidate = generation / "result.json"
        try:
            loaded = _load_ascii_json(candidate)
        except PolicyImprovementSchemaError:
            raise PolicyImprovementSchemaError(
                "Failed-attempt result is missing or invalid."
            )
        if canonical_json_bytes(loaded) != canonical_json_bytes(result):
            raise PolicyImprovementSchemaError(
                "Failed-attempt result differs from the audited result."
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
            or manifest["failure_phase"] != result["failure"]["phase"]
            or manifest["result"]
            != {
                "path": "result.json",
                "bytes": result_identity["bytes"],
                "sha256": result_identity["sha256"],
            }
        ):
            raise PolicyImprovementSchemaError("Failed-attempt identity differs.")
        if not is_smoke and (
            manifest["phase"] != result["phase"]
            or manifest["tier"] != result["tier"]
            or manifest["environment_interactions"] != expected_environment_interactions
        ):
            raise PolicyImprovementSchemaError(
                "Failed run attempt phase, tier, or budget differs."
            )
        actual_files, actual_directories = _inventory(generation)
        if set(actual_files) != {"MANIFEST.json", "result.json"} or actual_directories:
            raise PolicyImprovementSchemaError("Failed-attempt inventory differs.")
        matches.append((generation, manifest, result_identity))
    if len(matches) != 1:
        raise PolicyImprovementSchemaError(
            f"Failed result {run_id} must resolve to exactly one immutable attempt."
        )
    generation, _, result_identity = matches[0]
    _, manifest_identity = _stable_regular_file(generation / "MANIFEST.json")
    return {
        "generation_path": str(generation),
        "generation_manifest_sha256": manifest_identity["sha256"],
        "result_sha256": result_identity["sha256"],
    }
