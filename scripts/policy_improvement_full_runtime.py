#!/usr/bin/env fbpython
"""Fail-closed runtime core for registered Stage 1, 2, and 3 runs.

This module deliberately does not select a trainer from a command-line string.
The authenticated packaged entrypoint must inject a sealed ``FullRunBackend``.
Until that entrypoint and its Buck target exist, the CLI prints the exact run
contract and exits with status 2.  This prevents a source-tree invocation from
being mistaken for publication evidence.

The runtime core owns registration checks, tier isolation, schedules, and the
immutable run-directory transaction.  A backend owns model construction,
training, strict checkpoint resume, evaluation, and creation of a schema-valid
artifact package inside the private staging directory.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from scripts.policy_improvement_registry import (
    load_registered_base_configs,
    registry_sha256,
    validate_registry_document,
)
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    amendment_history_sha256,
    canonical_json_bytes,
    validate_amendment_history,
    validate_protocol,
    validate_result,
)


FULL_RUNTIME_SCHEMA_VERSION: int = 1
FULL_SEGMENT_SCHEMA_VERSION: int = 1
FULL_EXECUTION_ENV: str = "RUN_UPITRM_FULL_EXPERIMENTS"
FULL_EXECUTION_VALUE: str = "1"
_AT_FDCWD: int = -100
_RENAME_NOREPLACE: int = 1
_PRIMARY_METHODS: frozenset[str] = frozenset(
    {
        "fixed_base_exact_persistent",
        "fixed_base_exact_episodic",
        "legacy_parameter_interpolation",
        "matched_ppo",
    }
)
_PHASE_HISTORY_LENGTH: Mapping[str, int] = {
    "stage1_screen": 1,
    "stage1_alpha": 2,
    "stage2_confirmatory": 3,
    "stage3_ablation": 3,
}
_TEST_PHASES: frozenset[str] = frozenset({"stage2_confirmatory", "stage3_ablation"})


class FullRuntimeError(RuntimeError):
    """Raised when a full learned run is not authorized or not publishable."""


@dataclass(frozen=True)
class RegisteredFullRun:
    """One exact concrete registry row and its frozen execution schedule."""

    project_root: Path
    protocol_path: Path
    registry_path: Path
    evidence_root: Path
    protocol: dict[str, Any]
    registry: dict[str, Any]
    amendment_history: tuple[dict[str, Any], ...]
    row: dict[str, Any]
    protocol_sha256: str
    registry_sha256: str
    amendment_history_sha256: str
    registry_row_sha256: str
    runtime_authorization_sha256: str
    interaction_checkpoints: tuple[int, ...]
    final_environment_interactions: int
    compute_target_recurrent_map_applications: int
    evaluation_records: int
    test_open_sha256: str | None


@dataclass(frozen=True)
class BackendRequest:
    """Exact contract passed to the sealed model-training backend."""

    run: RegisteredFullRun
    staging_generation: Path
    interaction_checkpoints: tuple[int, ...]
    compute_target_recurrent_map_applications: int
    compute_maximum_relative_mismatch: float
    require_strict_checkpoint_resume: bool
    require_interaction_and_compute_snapshots: bool


@dataclass(frozen=True)
class BackendPackage:
    """Backend-written package metadata returned to the transaction owner."""

    result: Mapping[str, object]
    primary_checkpoint_relative_path: str


class FullRunBackend(Protocol):
    """Trusted backend linked by the authenticated packaged entrypoint."""

    def execute(self, request: BackendRequest) -> BackendPackage:
        """Train, strictly resume, evaluate, and write the private package."""

        ...


class FullRunFailure(RuntimeError):
    """A backend failure carrying its schema-valid failed result."""

    def __init__(self, phase: str, result: Mapping[str, object]) -> None:
        super().__init__(f"Full policy-improvement run failed during {phase}.")
        self.phase = phase
        self.result = result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _stable_regular_file(path: Path) -> tuple[bytes, dict[str, object]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (
                stat.S_ISLNK(before_path.st_mode)
                or not stat.S_ISREG(before_path.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before_path.st_nlink != 1
                or (before_path.st_dev, before_path.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise FullRuntimeError(
                    f"Runtime input {path} is a symlink, hard-link alias, or wrong type."
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
        raise FullRuntimeError(
            f"Runtime input {path} cannot be authenticated."
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
    if identity(before) != identity(after) or identity(after) != identity(after_path):
        raise FullRuntimeError(f"Runtime input {path} changed while it was read.")
    payload = b"".join(chunks)
    return payload, {"bytes": len(payload), "sha256": digest.hexdigest()}


def _load_authenticated_json(path: Path) -> tuple[object, str]:
    payload, identity = _stable_regular_file(path)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FullRuntimeError(f"Non-finite JSON constant {token!r} is forbidden.")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullRuntimeError(f"Runtime JSON {path} is invalid.") from exc
    return value, str(identity["sha256"])


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FullRuntimeError(f"Duplicate JSON key {key!r} is forbidden.")
        result[key] = value
    return result


def _absolute_canonical_file(value: str | Path, *, name: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise FullRuntimeError(f"{name} must be an absolute canonical path.")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise FullRuntimeError(f"{name} is unavailable.") from exc
    if resolved != supplied:
        raise FullRuntimeError(f"{name} must not traverse a symlink or alias.")
    _stable_regular_file(resolved)
    return resolved


def _private_owner_root(value: str | Path, *, project_root: Path) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise FullRuntimeError("Evidence root must be an absolute canonical path.")
    try:
        info = supplied.lstat()
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise FullRuntimeError("Evidence root is unavailable.") from exc
    if (
        supplied != resolved
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or resolved == project_root
        or resolved in project_root.parents
        or project_root in resolved.parents
    ):
        raise FullRuntimeError(
            "Evidence root must be one private owner directory outside the source tree."
        )
    return resolved


def _load_test_open(
    *,
    evidence_root: Path,
    protocol_sha256: str,
    registry_digest: str,
    history_digest: str,
    runtime_authorization_digest: str,
    phase: str,
) -> str:
    path = evidence_root / "TEST_OPEN.json"
    value, digest = _load_authenticated_json(path)
    if not isinstance(value, Mapping):
        raise FullRuntimeError("TEST_OPEN.json must contain one JSON object.")
    required = {
        "schema_name",
        "schema_version",
        "record_id",
        "state",
        "protocol_sha256",
        "registry_sha256",
        "amendment_history_sha256",
        "runtime_authorization_sha256",
        "test_manifest_sha256",
        "prior_open_record_sha256",
        "open_ordinal",
        "authorized_phases",
        "final_selection_created_at_utc",
        "opened_at_utc",
    }
    if (
        set(value) != required
        or value.get("schema_name") != "policy_improvement_test_open_v1"
        or value.get("schema_version") != 3
        or value.get("state") != "opened_immutable"
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("registry_sha256") != registry_digest
        or value.get("amendment_history_sha256") != history_digest
        or value.get("runtime_authorization_sha256") != runtime_authorization_digest
        or value.get("open_ordinal") != 1
        or phase not in value.get("authorized_phases", [])
    ):
        raise FullRuntimeError(
            "TEST_OPEN.json does not authorize this exact protocol, registry, and phase."
        )
    return digest


def load_registered_full_run(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    registry_path: str | Path,
    amendment_paths: Sequence[str | Path],
    evidence_root: str | Path,
    row_id: str,
    runtime_authorization_sha256: str,
    environment: Mapping[str, str] | None = None,
) -> RegisteredFullRun:
    """Authenticate one concrete non-smoke row and freeze its exact schedules."""

    env = os.environ if environment is None else environment
    if env.get(FULL_EXECUTION_ENV) != FULL_EXECUTION_VALUE:
        raise FullRuntimeError(
            f"{FULL_EXECUTION_ENV}=1 is required for Stage 1, 2, and 3 execution."
        )
    if len(runtime_authorization_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in runtime_authorization_sha256
    ):
        raise FullRuntimeError(
            "Runtime authorization must be a lowercase SHA-256 digest."
        )
    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise FullRuntimeError("Project root is not a directory.")
    protocol_file = _absolute_canonical_file(protocol_path, name="protocol path")
    registry_file = _absolute_canonical_file(registry_path, name="registry path")
    if protocol_file != project / "configs/policy_improvement_v1/protocol.json":
        raise FullRuntimeError("Only the canonical committed protocol is accepted.")
    if registry_file != project / "configs/policy_improvement_v1/registry.json":
        raise FullRuntimeError("Only the canonical committed registry is accepted.")
    raw_protocol, _ = _load_authenticated_json(protocol_file)
    try:
        protocol = validate_protocol(raw_protocol)
    except PolicyImprovementSchemaError as exc:
        raise FullRuntimeError("Protocol validation failed.") from exc
    expected_gate = protocol["full_execution_gate"]
    if expected_gate != {
        "environment_variable": FULL_EXECUTION_ENV,
        "required_value": FULL_EXECUTION_VALUE,
    }:
        raise FullRuntimeError("Protocol full-execution gate differs from the runtime.")

    history: list[dict[str, Any]] = []
    for index, raw_path in enumerate(amendment_paths):
        path = _absolute_canonical_file(raw_path, name=f"amendment {index}")
        raw_amendment, _ = _load_authenticated_json(path)
        if not isinstance(raw_amendment, Mapping):
            raise FullRuntimeError("Every amendment must be one JSON object.")
        history.append(dict(raw_amendment))
    try:
        history = validate_amendment_history(history, protocol=protocol)
        base_configs = load_registered_base_configs(protocol, project)
        raw_registry, _ = _load_authenticated_json(registry_file)
        registry = validate_registry_document(
            raw_registry,
            protocol,
            history,
            base_configs=base_configs,
        )
    except PolicyImprovementSchemaError as exc:
        raise FullRuntimeError(
            "Registry or amendment history differs from deterministic regeneration."
        ) from exc
    rows = [row for row in registry["rows"] if row["run_id"] == row_id]
    if len(rows) != 1:
        raise FullRuntimeError("Requested run ID does not resolve to exactly one row.")
    row = rows[0]
    phase = str(row["phase"])
    if phase not in _PHASE_HISTORY_LENGTH:
        raise FullRuntimeError("The full runtime rejects smoke and unknown phases.")
    if row["row_kind"] != "concrete":
        raise FullRuntimeError(
            "Selection-dependent rows cannot run before their registered amendment."
        )
    if row["base_method_id"] not in _PRIMARY_METHODS or (
        phase != "stage3_ablation" and row["method_id"] not in _PRIMARY_METHODS
    ):
        raise FullRuntimeError(
            "Registered row does not use one of the four pre-registered methods."
        )
    expected_history_length = _PHASE_HISTORY_LENGTH[phase]
    if len(history) != expected_history_length:
        raise FullRuntimeError(
            f"{phase} requires exactly {expected_history_length} registered amendments."
        )
    tier = str(row["tier"])
    split = str(row["evaluation_split"])
    if (tier in {"pilot"}) != (split == "validation"):
        raise FullRuntimeError("Pilot execution is restricted to validation data.")
    if (phase in _TEST_PHASES) != (split == "test"):
        raise FullRuntimeError("Only frozen Stage 2/3 rows may open test data.")

    protocol_digest = _canonical_digest(protocol)
    registry_digest = registry_sha256(registry)
    history_digest = amendment_history_sha256(history)
    owner = _private_owner_root(evidence_root, project_root=project)
    test_open_digest = (
        _load_test_open(
            evidence_root=owner,
            protocol_sha256=protocol_digest,
            registry_digest=registry_digest,
            history_digest=history_digest,
            runtime_authorization_digest=runtime_authorization_sha256,
            phase=phase,
        )
        if split == "test"
        else None
    )

    budget_tier = "confirmatory" if tier in {"confirmatory", "ablation"} else tier
    budget = protocol["budgets"][budget_tier]
    checkpoints = tuple(
        int(value) for value in budget["checkpoint_environment_interactions"]
    )
    final_interactions = int(budget["environment_interactions"])
    if not checkpoints or checkpoints[-1] != final_interactions:
        raise FullRuntimeError(
            "Interaction checkpoint schedule does not end at its budget."
        )
    compute_tier = "ablation" if tier == "ablation" else tier
    compute_targets = history[0]["common_compute_targets"]
    compute_target = int(compute_targets[compute_tier])
    if compute_targets["unit"] != "recurrent_map_applications":
        raise FullRuntimeError("Compute schedule uses the wrong unit.")

    return RegisteredFullRun(
        project_root=project,
        protocol_path=protocol_file,
        registry_path=registry_file,
        evidence_root=owner,
        protocol=protocol,
        registry=registry,
        amendment_history=tuple(history),
        row=dict(row),
        protocol_sha256=protocol_digest,
        registry_sha256=registry_digest,
        amendment_history_sha256=history_digest,
        registry_row_sha256=_canonical_digest(row),
        runtime_authorization_sha256=runtime_authorization_sha256,
        interaction_checkpoints=checkpoints,
        final_environment_interactions=final_interactions,
        compute_target_recurrent_map_applications=compute_target,
        evaluation_records=int(budget["evaluation_records"]),
        test_open_sha256=test_open_digest,
    )


def _canonical_relative(value: str, *, name: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise FullRuntimeError(f"{name} is not a canonical relative path.")
    return value


def _inventory(root: Path) -> tuple[dict[str, dict[str, object]], set[str]]:
    files: dict[str, dict[str, object]] = {}
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise FullRuntimeError("A backend package contains a symlink.")
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FullRuntimeError(
                "Backend package entries must be singly linked regular files."
            )
        _, identity = _stable_regular_file(path)
        files[relative] = identity
    return files, directories


def _write_exclusive_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise FullRuntimeError("Atomic JSON staging made no write progress.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _sha256_bytes(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(root: Path, relative: PurePosixPath) -> Path:
    cursor = root
    for component in relative.parts:
        if component in {"", ".", ".."}:
            raise FullRuntimeError("Publication directory is not canonical.")
        parent = cursor
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            cursor.mkdir(mode=0o700)
            _fsync_directory(parent)
            info = cursor.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise FullRuntimeError(
                "Publication path contains an unsafe directory or symlink."
            )
    return cursor


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise FullRuntimeError("renameat2 is required for immutable publication.")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FullRuntimeError("This immutable run already exists.")
        raise FullRuntimeError(f"Atomic run publication failed: {os.strerror(error)}.")


def _validate_result_contract(
    run: RegisteredFullRun,
    result: Mapping[str, object],
    *,
    result_validator: Callable[[object], Mapping[str, object]],
) -> dict[str, object]:
    try:
        checked = dict(result_validator(result))
    except (PolicyImprovementSchemaError, ValueError, TypeError) as exc:
        raise FullRuntimeError(
            "Backend result failed policy_improvement_v1 schema."
        ) from exc
    row = run.row
    for field in (
        "run_id",
        "phase",
        "tier",
        "seed",
        "evaluation_split",
        "method_id",
        "base_method_id",
        "n",
        "K",
        "alpha",
        "ablation_variant",
    ):
        if checked.get(field) != row[field]:
            raise FullRuntimeError(f"Backend result changed registered field {field}.")
    if (
        checked.get("status") != "complete"
        or checked.get("protocol_sha256") != run.protocol_sha256
        or checked.get("registry_row_sha256") != run.registry_row_sha256
        or checked.get("amendment_history_sha256") != run.amendment_history_sha256
        or checked.get("applied_config_override") != row["config_override"]
    ):
        raise FullRuntimeError("Backend result is not the complete registered run.")
    snapshots = checked.get("evaluation_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 2:
        raise FullRuntimeError("Backend omitted one registered evaluation snapshot.")
    by_kind = {
        str(snapshot.get("snapshot_kind")): snapshot
        for snapshot in snapshots
        if isinstance(snapshot, Mapping)
    }
    if set(by_kind) != {"interaction_matched", "compute_matched"}:
        raise FullRuntimeError("Backend snapshot inventory differs.")
    interaction = by_kind["interaction_matched"]
    compute = by_kind["compute_matched"]
    expected_interactions = run.final_environment_interactions
    if (
        interaction.get("status") != "available"
        or interaction.get("target")
        != {
            "unit": "environment_interactions",
            "registered_quantity": {
                "status": "available",
                "value": expected_interactions,
            },
        }
        or interaction.get("observed_environment_interactions")
        != {"status": "available", "value": expected_interactions}
    ):
        raise FullRuntimeError("Interaction-matched snapshot missed its exact budget.")
    target = run.compute_target_recurrent_map_applications
    observed = compute.get("observed_recurrent_map_applications")
    if (
        compute.get("status") != "available"
        or compute.get("target")
        != {
            "unit": "recurrent_map_applications",
            "registered_quantity": {"status": "available", "value": target},
        }
        or not isinstance(observed, Mapping)
        or observed.get("status") != "available"
        or isinstance(observed.get("value"), bool)
        or not isinstance(observed.get("value"), int)
        or abs(int(observed["value"]) - target) / target > 0.05
    ):
        raise FullRuntimeError(
            "Compute-matched snapshot missed its registered tolerance."
        )
    identities = checked.get("identities")
    if not isinstance(identities, Mapping):
        raise FullRuntimeError("Backend result omitted identities.")
    if (
        identities.get("runtime_authorization_sha256")
        != run.runtime_authorization_sha256
    ):
        raise FullRuntimeError("Backend result names another runtime authorization.")
    test_open = identities.get("test_open_sha256")
    expected_test_open = (
        {"status": "available", "value": run.test_open_sha256}
        if run.test_open_sha256 is not None
        else {"status": "unavailable", "reason": "test_data_not_opened"}
    )
    if test_open != expected_test_open:
        raise FullRuntimeError("Backend result violates validation/test isolation.")
    return checked


def _finish_complete_generation(
    run: RegisteredFullRun,
    generation: Path,
    package: BackendPackage,
    *,
    result_validator: Callable[[object], Mapping[str, object]],
) -> None:
    result = _validate_result_contract(
        run, package.result, result_validator=result_validator
    )
    result_path = generation / "result.json"
    expected_result_bytes = canonical_json_bytes(result) + b"\n"
    observed_result, _ = _stable_regular_file(result_path)
    if observed_result != expected_result_bytes:
        raise FullRuntimeError("Backend result.json differs from its returned result.")
    primary_name = _canonical_relative(
        package.primary_checkpoint_relative_path,
        name="primary checkpoint path",
    )
    if primary_name == "MANIFEST.json":
        raise FullRuntimeError("Primary checkpoint cannot alias the manifest.")
    files, _ = _inventory(generation)
    if "MANIFEST.json" in files:
        raise FullRuntimeError("Backend must not create the generation manifest.")
    required = {"result.json", "RUN_MANIFEST.json", "model_state_inventory.json"}
    if not required.issubset(files) or primary_name not in files:
        raise FullRuntimeError(
            "Backend package omitted a required publication artifact."
        )
    identities = result["identities"]
    artifacts = result["artifacts"]
    expected_checkpoint = identities["checkpoint_sha256"]
    if (
        expected_checkpoint.get("status") != "available"
        or expected_checkpoint.get("value") != files[primary_name]["sha256"]
        or artifacts["checkpoint"] != expected_checkpoint
    ):
        raise FullRuntimeError("Primary checkpoint bytes differ from result identity.")
    validation_files = [
        name for name in files if name.endswith("checkpoint_validation.json")
    ]
    if len(validation_files) != 2:
        raise FullRuntimeError(
            "Full runs require strict validation for both registered snapshots."
        )
    for name in validation_files:
        raw, _ = _load_authenticated_json(generation / name)
        if (
            not isinstance(raw, Mapping)
            or raw.get("strict_resume_validated") is not True
        ):
            raise FullRuntimeError(
                "A snapshot lacks strict checkpoint-resume validation."
            )

    manifest = {
        "schema_name": "policy_improvement_run_segment_v1",
        "schema_version": FULL_SEGMENT_SCHEMA_VERSION,
        "protocol_sha256": run.protocol_sha256,
        "registry_sha256": run.registry_sha256,
        "registry_row_sha256": run.registry_row_sha256,
        "run_id": run.row["run_id"],
        "method_id": run.row["method_id"],
        "segment": "complete",
        "environment_interactions": run.final_environment_interactions,
        "parent_checkpoint_sha256": None,
        "result_status": "complete",
        "outputs": {
            "checkpoint": {"path": primary_name, **files[primary_name]},
            "files": files,
        },
        "storage_bytes": sum(int(identity["bytes"]) for identity in files.values()),
    }
    _write_exclusive_json(generation / "MANIFEST.json", manifest)
    for path in generation.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted(
        (path for path in generation.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(generation)


def _finish_failed_attempt(
    run: RegisteredFullRun,
    staging_run: Path,
    failure: FullRunFailure,
    *,
    result_validator: Callable[[object], Mapping[str, object]],
) -> None:
    try:
        result = dict(result_validator(failure.result))
    except (PolicyImprovementSchemaError, ValueError, TypeError) as exc:
        raise FullRuntimeError(
            "Backend failure did not provide a schema-valid result."
        ) from exc
    if (
        result.get("status") != "failed"
        or result.get("run_id") != run.row["run_id"]
        or result.get("protocol_sha256") != run.protocol_sha256
        or result.get("registry_row_sha256") != run.registry_row_sha256
        or result.get("failure", {}).get("phase") != failure.phase
        or result.get("identities", {}).get("runtime_authorization_sha256")
        != run.runtime_authorization_sha256
    ):
        raise FullRuntimeError(
            "Backend failure result differs from the registered run."
        )
    attempt_id = uuid.uuid4().hex
    attempt = staging_run / "attempts" / "complete" / attempt_id
    attempt.mkdir(parents=True, mode=0o700)
    result_sha256 = _write_exclusive_json(attempt / "result.json", result)
    result_bytes = (attempt / "result.json").stat().st_size
    manifest = {
        "schema_name": "policy_improvement_run_failed_attempt_v1",
        "schema_version": 1,
        "attempt_id": attempt_id,
        "protocol_sha256": run.protocol_sha256,
        "registry_row_sha256": run.registry_row_sha256,
        "runtime_authorization_sha256": result["identities"][
            "runtime_authorization_sha256"
        ],
        "run_id": run.row["run_id"],
        "segment": "complete",
        "failure_phase": failure.phase,
        "result": {
            "path": "result.json",
            "bytes": result_bytes,
            "sha256": result_sha256,
        },
        "phase": run.row["phase"],
        "tier": run.row["tier"],
        "environment_interactions": run.final_environment_interactions,
    }
    _write_exclusive_json(attempt / "MANIFEST.json", manifest)
    _fsync_directory(attempt)
    _fsync_directory(attempt.parent)
    _fsync_directory(attempt.parent.parent)
    _fsync_directory(staging_run)


def execute_registered_run(
    run: RegisteredFullRun,
    *,
    backend: FullRunBackend | None,
    result_validator: Callable[[object], Mapping[str, object]] = validate_result,
) -> Path:
    """Execute and atomically publish one complete run or one failed attempt."""

    if backend is None:
        raise FullRuntimeError(
            "No sealed full-training backend is linked. Source-tree execution is forbidden."
        )
    output_relative = PurePosixPath(str(run.protocol["output_root"]["relative_path"]))
    if any(part in {"", ".", ".."} for part in output_relative.parts):
        raise FullRuntimeError("Protocol output root is not canonical.")
    runs = _ensure_private_directory(
        run.evidence_root,
        PurePosixPath(*output_relative.parts, "runs"),
    )
    runs_identity = (runs.stat().st_dev, runs.stat().st_ino)
    final_run = runs / str(run.row["run_id"])
    staging_run = Path(tempfile.mkdtemp(prefix=".full-run-stage.", dir=runs))
    staging_identity = (staging_run.stat().st_dev, staging_run.stat().st_ino)
    published = False
    try:
        generation = (
            staging_run / "segments" / f"env_{run.final_environment_interactions:09d}"
        )
        generation.mkdir(parents=True, mode=0o700)
        request = BackendRequest(
            run=run,
            staging_generation=generation,
            interaction_checkpoints=run.interaction_checkpoints,
            compute_target_recurrent_map_applications=(
                run.compute_target_recurrent_map_applications
            ),
            compute_maximum_relative_mismatch=0.05,
            require_strict_checkpoint_resume=True,
            require_interaction_and_compute_snapshots=True,
        )
        try:
            package = backend.execute(request)
            _finish_complete_generation(
                run,
                generation,
                package,
                result_validator=result_validator,
            )
        except FullRunFailure as failure:
            shutil.rmtree(staging_run / "segments")
            _finish_failed_attempt(
                run,
                staging_run,
                failure,
                result_validator=result_validator,
            )
        _fsync_directory(staging_run)
        current_runs = runs.lstat()
        if (
            stat.S_ISLNK(current_runs.st_mode)
            or not stat.S_ISDIR(current_runs.st_mode)
            or (current_runs.st_dev, current_runs.st_ino) != runs_identity
        ):
            raise FullRuntimeError("Publication root changed before atomic rename.")
        _rename_noreplace(staging_run, final_run)
        published = True
        _fsync_directory(runs)
        published_info = final_run.lstat()
        if (
            stat.S_ISLNK(published_info.st_mode)
            or not stat.S_ISDIR(published_info.st_mode)
            or (published_info.st_dev, published_info.st_ino) != staging_identity
        ):
            raise FullRuntimeError("Published run directory has the wrong identity.")
        return final_run
    finally:
        if not published:
            try:
                current = staging_run.lstat()
                if (current.st_dev, current.st_ino) == staging_identity:
                    shutil.rmtree(staging_run)
                    _fsync_directory(runs)
            except OSError:
                pass


def execution_contract(run: RegisteredFullRun) -> dict[str, object]:
    """Return the exact sealed-entrypoint contract without starting training."""

    return {
        "schema_name": "policy_improvement_full_runtime_contract_v1",
        "schema_version": FULL_RUNTIME_SCHEMA_VERSION,
        "execution_ready": False,
        "blocked_by": [
            "phase4 launcher lacks a policy-improvement-full purpose",
            "packaged entrypoint does not inject a sealed FullRunBackend",
            "BUCK has no full learned training/evaluation binary target",
        ],
        "run_id": run.row["run_id"],
        "method_id": run.row["method_id"],
        "phase": run.row["phase"],
        "tier": run.row["tier"],
        "evaluation_split": run.row["evaluation_split"],
        "protocol_sha256": run.protocol_sha256,
        "registry_sha256": run.registry_sha256,
        "registry_row_sha256": run.registry_row_sha256,
        "amendment_history_sha256": run.amendment_history_sha256,
        "runtime_authorization_sha256": run.runtime_authorization_sha256,
        "interaction_checkpoints": list(run.interaction_checkpoints),
        "compute_target": {
            "unit": "recurrent_map_applications",
            "value": run.compute_target_recurrent_map_applications,
            "maximum_relative_mismatch": 0.05,
        },
        "evaluation_records": run.evaluation_records,
        "test_open_sha256": run.test_open_sha256,
        "required_entrypoint_call": {
            "callable": "execute_registered_run",
            "backend": "sealed packaged FullRunBackend",
            "source_tree_fallback": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--amendment", action="append", default=[])
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--runtime-authorization-sha256", required=True)
    parser.add_argument("--print-contract", action="store_true")
    arguments = parser.parse_args(argv)
    run = load_registered_full_run(
        project_root=arguments.project_root,
        protocol_path=arguments.protocol,
        registry_path=arguments.registry,
        amendment_paths=arguments.amendment,
        evidence_root=arguments.evidence_root,
        row_id=arguments.row_id,
        runtime_authorization_sha256=arguments.runtime_authorization_sha256,
    )
    contract = execution_contract(run)
    print(canonical_json_bytes(contract).decode("ascii"))
    if not arguments.print_contract:
        print(
            "Full execution is blocked until the authenticated launcher injects "
            "the sealed trainer backend.",
            file=os.sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
