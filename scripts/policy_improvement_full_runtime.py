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
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from policy_improvement_sealed_evidence import (
    seal_authenticated_checkpoint,
    SealedCheckpointError,
)
from scripts.policy_improvement_registry import (
    generate_registry,
    load_registered_base_configs,
    registry_sha256,
    validate_registry_document,
)
from scripts.policy_improvement_schema import (
    amendment_history_sha256,
    canonical_json_bytes,
    PHASE_AMENDMENT_PREFIX_LENGTH,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    validate_amendment_history,
    validate_protocol,
    validate_result,
    validate_runtime_authorization,
    validated_result_payload,
)
from scripts.policy_improvement_v2_schema import (
    bind_v2_result_to_registration,
    PolicyImprovementV2SchemaError,
    validate_v2_amendment_history,
)


FULL_RUNTIME_SCHEMA_VERSION: int = 1
FULL_SEGMENT_SCHEMA_VERSION: int = 2
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
    dataset_root: Path | None = None
    population_document: dict[str, Any] | None = None


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


@dataclass(frozen=True)
class AuthenticatedFullCheckpoint:
    """One checkpoint resolved from a complete immutable full-run package."""

    path: Path
    sha256: str
    size_bytes: int
    snapshot_kind: str
    environment_interactions: int
    model_state_sha256: str
    role_state_sha256s: dict[str, str]
    parent_checkpoint_sha256: str | None
    generation_manifest_sha256: str
    run_manifest_sha256: str
    validation_sha256: str
    sealed_descriptor: int = -1
    theory_model_identity: dict[str, str] | None = None


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


class PublishedFullRunFailure(FullRuntimeError):
    """A schema-valid failed attempt that was durably published."""

    def __init__(self, published_run: Path, failure: FullRunFailure) -> None:
        super().__init__(
            f"Full policy-improvement run published a failed {failure.phase} attempt."
        )
        self.published_run = published_run
        self.phase = failure.phase
        self.result = failure.result


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
    dataset_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    require_stage1_selection: bool = False,
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
    canonical_protocols = {
        project
        / "configs/policy_improvement_v1/protocol.json": (
            project / "configs/policy_improvement_v1/registry.json"
        ),
        project
        / "configs/policy_improvement_v2/protocol.json": (
            project / "configs/policy_improvement_v2/registry.json"
        ),
    }
    if protocol_file not in canonical_protocols:
        raise FullRuntimeError("Only a canonical committed protocol is accepted.")
    if registry_file != canonical_protocols[protocol_file]:
        raise FullRuntimeError(
            "Protocol and registry paths do not use one canonical namespace."
        )
    raw_protocol, _ = _load_authenticated_json(protocol_file)
    try:
        protocol = validate_protocol(raw_protocol)
    except (
        PolicyImprovementSchemaError,
        PolicyImprovementV2SchemaError,
        ValueError,
    ) as exc:
        raise FullRuntimeError("Protocol validation failed.") from exc
    is_v2 = protocol.get("schema_name") == "policy_improvement_protocol_v2"
    expected_gate = protocol["full_execution_gate"]
    required_gate = {
        "environment_variable": FULL_EXECUTION_ENV,
        "required_value": FULL_EXECUTION_VALUE,
    }
    if is_v2:
        required_gate.update(
            {
                "stage0_exempt": True,
                "stage1_to_stage3_blocked_by_base_policy": True,
            }
        )
    if expected_gate != required_gate:
        raise FullRuntimeError("Protocol full-execution gate differs from the runtime.")
    history: list[dict[str, Any]] = []
    for index, raw_path in enumerate(amendment_paths):
        path = _absolute_canonical_file(raw_path, name=f"amendment {index}")
        raw_amendment, _ = _load_authenticated_json(path)
        if not isinstance(raw_amendment, Mapping):
            raise FullRuntimeError("Every amendment must be one JSON object.")
        history.append(dict(raw_amendment))
    try:
        base_configs = load_registered_base_configs(protocol, project)
        populations_document: object | None = None
        if is_v2:
            from scripts.policy_improvement_populations import (
                load_registered_populations,
            )

            populations_document = load_registered_populations(protocol, project)
        raw_registry, _ = _load_authenticated_json(registry_file)
        validate_registry_document(
            raw_registry,
            protocol,
            [],
            base_configs=base_configs,
            populations_value=populations_document,
        )
        registry = generate_registry(
            protocol,
            [] if is_v2 else history,
            base_configs=base_configs,
            populations_value=populations_document,
        )
        if is_v2:
            if len(history) not in {3, 4}:
                raise FullRuntimeError(
                    "Protocol v2 Stage 1 requires theory, base-policy, and compute "
                    "amendments, followed by at most one V_select configuration "
                    "selection."
                )
            assert isinstance(populations_document, Mapping)
            history = validate_v2_amendment_history(
                history,
                protocol=protocol,
                registry=registry,
                populations=populations_document,
            )
            registry = generate_registry(
                protocol,
                history,
                base_configs=base_configs,
                populations_value=populations_document,
            )
            base_policy = history[1]
            compute_freeze = history[2]
            selection = history[3] if len(history) == 4 else None
            if (
                base_policy["runtime_authorization_sha256"]
                != runtime_authorization_sha256
                or compute_freeze["runtime_authorization_sha256"]
                != runtime_authorization_sha256
                or (
                    selection is not None
                    and selection["runtime_authorization_sha256"]
                    != runtime_authorization_sha256
                )
            ):
                raise FullRuntimeError(
                    "Protocol v2 amendments bind another runtime authorization."
                )
            if require_stage1_selection and selection is None:
                raise FullRuntimeError(
                    "Validation-bridge evaluation requires an authenticated "
                    "V_select configuration selection."
                )
            if not require_stage1_selection:
                raise FullRuntimeError(
                    "Protocol v2 learned execution remains blocked until train-only "
                    "base-artifact restore and cross-generation continuation are "
                    "implemented."
                )
        else:
            history = validate_amendment_history(history, protocol=protocol)
    except (
        PolicyImprovementSchemaError,
        PolicyImprovementV2SchemaError,
        ValueError,
    ) as exc:
        raise FullRuntimeError(
            "Registry or amendment history differs from deterministic regeneration."
        ) from exc
    rows = [row for row in registry["rows"] if row["run_id"] == row_id]
    if len(rows) != 1:
        raise FullRuntimeError("Requested run ID does not resolve to exactly one row.")
    row = rows[0]
    phase = str(row["phase"])
    if is_v2:
        if phase != "stage1_screen":
            raise FullRuntimeError(
                "Protocol v2 full execution requires future selection amendments "
                "outside the Stage 1 screen."
            )
    elif phase not in PHASE_AMENDMENT_PREFIX_LENGTH or phase == "stage0_smoke":
        raise FullRuntimeError("The full runtime rejects smoke and unknown phases.")
    if row["row_kind"] != "concrete":
        raise FullRuntimeError(
            "Selection-dependent rows cannot run before their registered amendment."
        )
    if is_v2 and len(history) == 4:
        selected = history[3]["selected_configurations"]
        matching = [
            item for item in selected if item["method_id"] == row.get("method_id")
        ]
        if (
            len(matching) != 1
            or matching[0]["n"] != row.get("n")
            or matching[0]["K"] != row.get("K")
        ):
            raise FullRuntimeError("Protocol v2 row was not selected from V_select.")
    if row["base_method_id"] not in _PRIMARY_METHODS or (
        phase != "stage3_ablation" and row["method_id"] not in _PRIMARY_METHODS
    ):
        raise FullRuntimeError(
            "Registered row does not use one of the four pre-registered methods."
        )
    expected_history_length = (
        len(history) if is_v2 else PHASE_AMENDMENT_PREFIX_LENGTH[phase]
    )
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
    materialized_dataset = None
    if dataset_root is not None:
        supplied_dataset = Path(dataset_root)
        if not supplied_dataset.is_absolute() or ".." in supplied_dataset.parts:
            raise FullRuntimeError(
                "Dataset root must be an absolute canonical directory."
            )
        try:
            materialized_dataset = supplied_dataset.resolve(strict=True)
            dataset_status = supplied_dataset.lstat()
        except OSError as exc:
            raise FullRuntimeError("Dataset root is unavailable.") from exc
        if (
            materialized_dataset != supplied_dataset
            or stat.S_ISLNK(dataset_status.st_mode)
            or not stat.S_ISDIR(dataset_status.st_mode)
        ):
            raise FullRuntimeError(
                "Dataset root must be an absolute canonical directory."
            )
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
        int(value)
        for value in (
            row["checkpoint_environment_interactions"]
            if is_v2
            else budget["checkpoint_environment_interactions"]
        )
    )
    if is_v2 and len(history) == 3:
        checkpoints = checkpoints[:1]
    final_interactions = int(
        checkpoints[-1] if is_v2 else budget["environment_interactions"]
    )
    if not checkpoints or checkpoints[-1] != final_interactions:
        raise FullRuntimeError(
            "Interaction checkpoint schedule does not end at its budget."
        )
    compute_tier = "ablation" if tier == "ablation" else tier
    compute_freezes = [
        amendment
        for amendment in history
        if amendment.get("schema_name")
        in {
            "policy_improvement_compute_freeze_v1",
            "policy_improvement_compute_freeze_v2",
        }
        or ("schema_name" not in amendment and "common_compute_targets" in amendment)
    ]
    if len(compute_freezes) != 1:
        raise FullRuntimeError(
            "Amendment history must contain exactly one compute freeze."
        )
    compute_targets = compute_freezes[0]["common_compute_targets"]
    compute_target = int(compute_targets[compute_tier])
    if compute_targets["unit"] != "recurrent_map_applications":
        raise FullRuntimeError("Compute schedule uses the wrong unit.")

    if is_v2 and not isinstance(populations_document, Mapping):
        raise FullRuntimeError("Protocol v2 population registration is unavailable.")
    if is_v2:
        populations = populations_document.get("populations")
        population_id = row.get("evaluation_population")
        population = (
            populations.get(population_id)
            if isinstance(populations, Mapping) and isinstance(population_id, str)
            else None
        )
        if (
            not isinstance(population, Mapping)
            or population.get("split") != split
            or isinstance(population.get("count"), bool)
            or not isinstance(population.get("count"), int)
        ):
            raise FullRuntimeError(
                "Protocol v2 row lacks its exact registered evaluation population."
            )
        evaluation_records = int(population["count"])
    else:
        evaluation_records = int(budget["evaluation_records"])
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
        evaluation_records=evaluation_records,
        test_open_sha256=test_open_digest,
        dataset_root=materialized_dataset,
        population_document=(
            dict(populations_document)
            if isinstance(populations_document, Mapping)
            else None
        ),
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


def _acquire_run_publication_lock(runs: Path, run_id: str) -> int:
    locks = _ensure_private_directory(runs, PurePosixPath(".locks"))
    lock_path = locks / f"{run_id}.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        opened = os.fstat(descriptor)
        visible = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or opened.st_nlink != 1
            or visible.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
            or visible.st_uid != os.geteuid()
            or stat.S_IMODE(visible.st_mode) != 0o600
        ):
            raise FullRuntimeError("Run publication lock is unsafe.")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


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


def _validated_result_for_run(
    run: RegisteredFullRun,
    result: Mapping[str, object],
    *,
    result_validator: Callable[[object], Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        checked_document = dict(result_validator(result))
    except (
        PolicyImprovementSchemaError,
        PolicyImprovementV2SchemaError,
        ValueError,
        TypeError,
    ) as exc:
        raise FullRuntimeError(
            "Backend result failed its registered result schema."
        ) from exc
    is_v2 = run.protocol.get("schema_name") == "policy_improvement_protocol_v2"
    is_v2_document = (
        checked_document.get("schema_name") == "policy_improvement_result_v2"
    )
    if is_v2 != is_v2_document:
        raise FullRuntimeError(
            "Backend result schema differs from its registered protocol namespace."
        )
    if is_v2:
        if run.population_document is None:
            raise FullRuntimeError(
                "Protocol v2 result lacks its authenticated population registration."
            )
        try:
            bind_v2_result_to_registration(
                checked_document,
                run.row,
                run.protocol,
                run.registry,
                run.population_document,
            )
        except (PolicyImprovementV2SchemaError, KeyError, TypeError) as exc:
            raise FullRuntimeError(
                "Protocol v2 result differs from its authenticated registration."
            ) from exc
    try:
        document, checked = validated_result_payload(checked_document)
    except (
        PolicyImprovementSchemaError,
        PolicyImprovementV2SchemaError,
        ValueError,
        TypeError,
    ) as exc:
        raise FullRuntimeError(
            "Backend result payload failed its registered runtime schema."
        ) from exc
    if is_v2 and checked.get("schema_name") != (
        "policy_improvement_full_result_payload_v2"
    ):
        raise FullRuntimeError(
            "The full runtime requires the non-smoke protocol v2 payload schema."
        )
    return dict(document), dict(checked)


def _validate_result_contract(
    run: RegisteredFullRun,
    result: Mapping[str, object],
    *,
    result_validator: Callable[[object], Mapping[str, object]],
) -> dict[str, object]:
    document, checked = _validated_result_for_run(
        run,
        result,
        result_validator=result_validator,
    )
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
    return document


def _available_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"status", "value"}
        or value.get("status") != "available"
    ):
        raise FullRuntimeError(f"{name} must be an available digest.")
    digest = value.get("value")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise FullRuntimeError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _validated_theory_model_identity(
    value: object,
    *,
    required: bool,
    name: str,
) -> dict[str, str] | None:
    if value is None:
        if required:
            raise FullRuntimeError(f"{name} is required for this method.")
        return None
    fields = {
        "model_sha256",
        "model_config_sha256",
        "current_policy_sha256",
        "candidate_policy_sha256",
        "deployed_policy_sha256",
        "recurrent_transition_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FullRuntimeError(f"{name} field inventory differs.")
    checked: dict[str, str] = {}
    for field in sorted(fields):
        digest = value[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FullRuntimeError(f"{name}.{field} is not a SHA-256 digest.")
        checked[field] = digest
    return checked


def _authenticated_publication_directory(path: Path, *, name: str) -> tuple[int, int]:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FullRuntimeError(f"{name} is unavailable.") from exc
    if (
        resolved != path
        or stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & 0o022
    ):
        raise FullRuntimeError(f"{name} is an unsafe alias or directory.")
    return status.st_dev, status.st_ino


def _seal_authenticated_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> int:
    """Copy exact checkpoint bytes into a write-sealed anonymous file.

    The sealing primitive is shared with the evidence-audit boundary so both
    trust boundaries cannot drift apart.
    """

    try:
        return seal_authenticated_checkpoint(
            path,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
    except SealedCheckpointError as exc:
        raise FullRuntimeError(str(exc)) from exc


def _authorized_runtime_role(
    authorization: Mapping[str, object], role_name: str
) -> Mapping[str, object]:
    roles = authorization.get("roles")
    if not isinstance(roles, list):
        raise FullRuntimeError("Runtime authorization has no role inventory.")
    matches = [
        role
        for role in roles
        if isinstance(role, Mapping) and role.get("role") == role_name
    ]
    if len(matches) != 1:
        raise FullRuntimeError(
            f"Runtime authorization does not bind exactly one {role_name} role."
        )
    return matches[0]


def _full_producer_runtime_role(
    authorization: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    """Return the artifact that actually produced a non-smoke full result."""

    role_name = (
        "policy-improvement-full"
        if authorization.get("schema_name")
        == "policy_improvement_runtime_authorization_v3"
        else "policy-improvement-training"
    )
    return role_name, _authorized_runtime_role(authorization, role_name)


def resolve_authenticated_full_checkpoint(
    run: RegisteredFullRun,
    *,
    checkpoint_environment_interactions: int,
    runtime_authorization: Mapping[str, object],
    result_validator: Callable[[object], Mapping[str, object]] = validate_result,
) -> AuthenticatedFullCheckpoint:
    """Resolve a registered checkpoint without deserializing checkpoint bytes."""

    if (
        isinstance(checkpoint_environment_interactions, bool)
        or not isinstance(checkpoint_environment_interactions, int)
        or checkpoint_environment_interactions not in run.interaction_checkpoints
        or not run.interaction_checkpoints
        or run.interaction_checkpoints[-1] != run.final_environment_interactions
    ):
        raise FullRuntimeError(
            "Theory checkpoint progress differs from the registered full-run schedule."
        )
    try:
        authorization = validate_runtime_authorization(runtime_authorization)
        authorization_digest = runtime_authorization_sha256(authorization)
    except PolicyImprovementSchemaError as exc:
        raise FullRuntimeError("Theory runtime authorization is invalid.") from exc
    expected_authorization_schema = (
        "policy_improvement_runtime_authorization_v3"
        if run.protocol.get("schema_name") == "policy_improvement_protocol_v2"
        else "policy_improvement_runtime_authorization_v2"
    )
    if (
        authorization.get("schema_name") != expected_authorization_schema
        or authorization.get("protocol_sha256") != run.protocol_sha256
        or authorization_digest != run.runtime_authorization_sha256
    ):
        raise FullRuntimeError(
            "Theory runtime authorization differs from the registered full run."
        )
    producer_role_name, producer_role = _full_producer_runtime_role(authorization)

    owner = _private_owner_root(run.evidence_root, project_root=run.project_root)
    output_relative = _canonical_relative(
        str(run.protocol["output_root"]["relative_path"]),
        name="protocol output root",
    )
    publication_root = owner / PurePosixPath(output_relative)
    _authenticated_publication_directory(
        publication_root,
        name="Policy-improvement publication root",
    )
    run_root = publication_root / "runs" / str(run.row["run_id"])
    _authenticated_publication_directory(run_root, name="Published full-run root")
    try:
        run_entries = {entry.name for entry in run_root.iterdir()}
    except OSError as exc:
        raise FullRuntimeError("Published full-run root cannot be enumerated.") from exc
    expected_run_entries = {"segments"}
    if (run_root / "attempts").exists():
        expected_run_entries.add("attempts")
    if run_entries != expected_run_entries:
        raise FullRuntimeError("Published full-run root contains unregistered entries.")
    prior_failed_attempts = _prior_failed_attempts(run_root)
    segments = run_root / "segments"
    _authenticated_publication_directory(segments, name="Published segment root")
    expected_segment_name = f"env_{run.final_environment_interactions:09d}"
    try:
        segment_entries = {entry.name for entry in segments.iterdir()}
    except OSError as exc:
        raise FullRuntimeError("Published segment root cannot be enumerated.") from exc
    if segment_entries != {expected_segment_name}:
        raise FullRuntimeError("Published segment inventory differs from the full run.")
    generation = segments / expected_segment_name
    generation_identity = _authenticated_publication_directory(
        generation,
        name="Published full-run generation",
    )

    raw_result, _ = _load_authenticated_json(generation / "result.json")
    if not isinstance(raw_result, Mapping):
        raise FullRuntimeError("Published full-run result must be one JSON object.")
    result_document = _validate_result_contract(
        run,
        raw_result,
        result_validator=result_validator,
    )
    _, result = validated_result_payload(result_document)
    result_payload, result_file_identity = _stable_regular_file(
        generation / "result.json"
    )
    if result_payload != canonical_json_bytes(result_document) + b"\n":
        raise FullRuntimeError("Published full-run result is not canonical JSON.")
    if result.get("evaluation_split") != "validation":
        raise FullRuntimeError("Theory evaluation accepts validation-only full runs.")
    result_identities = result.get("identities")
    if not isinstance(result_identities, Mapping):
        raise FullRuntimeError("Published full-run result omits runtime identities.")
    expected_runtime_identities = {
        "producer_git_commit": authorization["producer_git_commit"],
        "producer_manifest_sha256": authorization["producer_source_manifest_sha256"],
        "runtime_authorization_sha256": authorization_digest,
        "training_source_git_commit": producer_role["source_git_commit"],
        "training_runtime_sha256": producer_role["runtime_sha256"],
        "training_runtime_profile_sha256": producer_role["runtime_profile_sha256"],
        "training_selected_source_manifest_sha256": producer_role[
            "selected_source_manifest_sha256"
        ],
        "launcher_sha256": authorization["launcher_sha256"],
    }
    if any(
        result_identities.get(field) != expected
        for field, expected in expected_runtime_identities.items()
    ):
        raise FullRuntimeError(
            "Published full-run result differs from the authenticated training runtime."
        )

    raw_manifest, generation_manifest_sha256 = _load_authenticated_json(
        generation / "MANIFEST.json"
    )
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
        "prior_failed_attempts",
        "outputs",
        "storage_bytes",
    }
    if not isinstance(raw_manifest, Mapping) or set(raw_manifest) != manifest_fields:
        raise FullRuntimeError("Published generation manifest fields differ.")
    manifest = dict(raw_manifest)
    if (
        manifest["schema_name"] != "policy_improvement_run_segment_v1"
        or manifest["schema_version"] != FULL_SEGMENT_SCHEMA_VERSION
        or manifest["protocol_sha256"] != run.protocol_sha256
        or manifest["registry_sha256"] != run.registry_sha256
        or manifest["registry_row_sha256"] != run.registry_row_sha256
        or manifest["run_id"] != run.row["run_id"]
        or manifest["method_id"] != run.row["method_id"]
        or manifest["segment"] != "complete"
        or manifest["environment_interactions"] != run.final_environment_interactions
        or manifest["parent_checkpoint_sha256"] is not None
        or manifest["result_status"] != "complete"
        or manifest["prior_failed_attempts"] != prior_failed_attempts
    ):
        raise FullRuntimeError("Published generation manifest identity differs.")
    outputs = manifest["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != {"checkpoint", "files"}:
        raise FullRuntimeError("Published generation output inventory differs.")
    registered_files = outputs["files"]
    if not isinstance(registered_files, Mapping):
        raise FullRuntimeError("Published generation file inventory is invalid.")
    canonical_files: dict[str, dict[str, object]] = {}
    expected_directories: set[str] = set()
    for raw_name, raw_identity in registered_files.items():
        name = _canonical_relative(str(raw_name), name="generation output path")
        if name == "MANIFEST.json" or not isinstance(raw_identity, Mapping):
            raise FullRuntimeError("Published generation file identity is invalid.")
        if set(raw_identity) != {"bytes", "sha256"}:
            raise FullRuntimeError("Published generation file identity fields differ.")
        size = raw_identity["bytes"]
        digest = raw_identity["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FullRuntimeError("Published generation file identity is invalid.")
        canonical_files[name] = {"bytes": size, "sha256": digest}
        parts = PurePosixPath(name).parts
        expected_directories.update(
            PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    actual_files, actual_directories = _inventory(generation)
    manifest_identity = actual_files.pop("MANIFEST.json", None)
    if (
        manifest_identity is None
        or manifest_identity["sha256"] != generation_manifest_sha256
        or actual_files != canonical_files
        or actual_directories != expected_directories
        or manifest["storage_bytes"]
        != sum(int(identity["bytes"]) for identity in canonical_files.values())
    ):
        raise FullRuntimeError("Published generation differs from its manifest.")
    if canonical_files.get("result.json") != result_file_identity:
        raise FullRuntimeError("Published result differs from its generation manifest.")

    primary = outputs["checkpoint"]
    if not isinstance(primary, Mapping) or set(primary) != {"path", "bytes", "sha256"}:
        raise FullRuntimeError("Published primary checkpoint identity differs.")
    primary_name = _canonical_relative(
        str(primary["path"]), name="primary checkpoint path"
    )
    if dict(primary) != {"path": primary_name, **canonical_files.get(primary_name, {})}:
        raise FullRuntimeError("Published primary checkpoint is not inventoried.")

    raw_run_manifest, run_manifest_sha256 = _load_authenticated_json(
        generation / "RUN_MANIFEST.json"
    )
    run_manifest_fields = {
        "schema_name",
        "schema_version",
        "run_id",
        "method_id",
        "environment_interactions",
        "protocol_sha256",
        "registry_row_sha256",
        "amendment_history_sha256",
        "runtime_authorization_sha256",
        "runtime_sha256",
        "source_git_commit",
        "source_manifest_sha256",
        "dataset_manifest_sha256",
        "checkpoint_sha256",
        "model_state_sha256",
        "model_state_sha256s",
        "model_state_inventory_sha256",
        "checkpoint_schedule",
        "snapshots",
    }
    if (
        not isinstance(raw_run_manifest, Mapping)
        or set(raw_run_manifest) != run_manifest_fields
    ):
        raise FullRuntimeError("Published run-manifest fields differ.")
    run_manifest = dict(raw_run_manifest)
    if (
        run_manifest["schema_name"] != "policy_improvement_full_run_manifest_v1"
        or run_manifest["schema_version"] != 1
        or run_manifest["run_id"] != run.row["run_id"]
        or run_manifest["method_id"] != run.row["method_id"]
        or run_manifest["environment_interactions"]
        != run.final_environment_interactions
        or run_manifest["protocol_sha256"] != run.protocol_sha256
        or run_manifest["registry_row_sha256"] != run.registry_row_sha256
        or run_manifest["amendment_history_sha256"] != run.amendment_history_sha256
        or run_manifest["runtime_authorization_sha256"] != authorization_digest
        or run_manifest["runtime_sha256"] != producer_role["runtime_sha256"]
        or run_manifest["source_git_commit"] != producer_role["source_git_commit"]
        or run_manifest["source_manifest_sha256"]
        != producer_role["selected_source_manifest_sha256"]
        or run_manifest["dataset_manifest_sha256"]
        != result_identities.get("dataset_manifest_sha256")
    ):
        raise FullRuntimeError(
            "Published run manifest differs from the registered run and runtime."
        )
    result_artifacts = result.get("artifacts")
    if not isinstance(result_artifacts, Mapping):
        raise FullRuntimeError("Published result omits artifact identities.")
    if (
        canonical_files.get("RUN_MANIFEST.json", {}).get("sha256")
        != run_manifest_sha256
        or _available_digest(
            result_artifacts.get("run_manifest"), name="result run manifest"
        )
        != run_manifest_sha256
    ):
        raise FullRuntimeError("Published run manifest digest differs.")

    model_inventory_sha256 = run_manifest["model_state_inventory_sha256"]
    if (
        not isinstance(model_inventory_sha256, str)
        or canonical_files.get("model_state_inventory.json", {}).get("sha256")
        != model_inventory_sha256
        or _available_digest(
            result_artifacts.get("model_state_inventory"),
            name="result model-state inventory",
        )
        != model_inventory_sha256
    ):
        raise FullRuntimeError("Published model-state inventory digest differs.")
    raw_model_inventory, _ = _load_authenticated_json(
        generation / "model_state_inventory.json"
    )
    if not isinstance(raw_model_inventory, Mapping):
        raise FullRuntimeError("Published model-state inventory is invalid.")
    model_inventory = dict(raw_model_inventory)
    is_v2_run = run.protocol.get("schema_name") == "policy_improvement_protocol_v2"
    theory_identity_required = is_v2_run and run.row.get("method_id") in {
        "fixed_base_exact_persistent",
        "fixed_base_exact_episodic",
        "legacy_parameter_interpolation",
        "fixed_base_distilled_realization",
    }
    model_inventory_fields = {
        "schema_name",
        "run_id",
        "method_id",
        "model_state_sha256",
        "role_state_sha256s",
        "snapshot_state_bindings",
    }
    if is_v2_run:
        model_inventory_fields.add("theory_model_identity")
    if set(model_inventory) != model_inventory_fields or (
        model_inventory["schema_name"]
        != (
            "policy_improvement_model_state_inventory_v2"
            if is_v2_run
            else "policy_improvement_model_state_inventory_v1"
        )
        or model_inventory["run_id"] != run.row["run_id"]
        or model_inventory["method_id"] != run.row["method_id"]
        or model_inventory["model_state_sha256"] != run_manifest["model_state_sha256"]
        or model_inventory["role_state_sha256s"] != run_manifest["model_state_sha256s"]
    ):
        raise FullRuntimeError("Published model-state inventory identity differs.")
    inventory_theory_identity = _validated_theory_model_identity(
        model_inventory.get("theory_model_identity"),
        required=theory_identity_required,
        name="Published model-state theory identity",
    )

    expected_execution_identity = {
        "role": producer_role_name,
        "source_git_commit": producer_role["source_git_commit"],
        "runtime_sha256": producer_role["runtime_sha256"],
        "runtime_profile_sha256": producer_role["runtime_profile_sha256"],
        "selected_source_manifest_sha256": producer_role[
            "selected_source_manifest_sha256"
        ],
        "runtime_authorization_sha256": authorization_digest,
        "launcher_sha256": authorization["launcher_sha256"],
    }

    def validated_checkpoint(
        *,
        checkpoint_name: str,
        checkpoint_sha256: object,
        validation_name: str,
        validation_sha256: object,
        snapshot_kind: str,
        environment_interactions: int,
    ) -> AuthenticatedFullCheckpoint:
        canonical_checkpoint = _canonical_relative(
            checkpoint_name, name=f"{snapshot_kind} checkpoint path"
        )
        checkpoint_identity = canonical_files.get(canonical_checkpoint)
        if (
            checkpoint_identity is None
            or checkpoint_identity.get("sha256") != checkpoint_sha256
            or Path(canonical_checkpoint).name.startswith("model_step_")
        ):
            raise FullRuntimeError(
                f"Published {snapshot_kind} checkpoint identity differs."
            )
        canonical_validation = _canonical_relative(
            validation_name, name=f"{snapshot_kind} validation path"
        )
        validation_identity = canonical_files.get(canonical_validation)
        if (
            validation_identity is None
            or validation_identity.get("sha256") != validation_sha256
        ):
            raise FullRuntimeError(
                f"Published {snapshot_kind} validation identity differs."
            )
        raw_validation, observed_validation_sha256 = _load_authenticated_json(
            generation / canonical_validation
        )
        validation_fields = {
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
        }
        if is_v2_run:
            validation_fields.add("theory_model_identity")
        if (
            not isinstance(raw_validation, Mapping)
            or set(raw_validation) != validation_fields
        ):
            raise FullRuntimeError(
                f"Published {snapshot_kind} validation fields differ."
            )
        validation = dict(raw_validation)
        role_hashes = validation["role_state_sha256s"]
        model_state_sha256 = validation["model_state_sha256"]
        if (
            observed_validation_sha256 != validation_sha256
            or validation["schema_name"]
            != "policy_improvement_checkpoint_validation_v1"
            or validation["schema_version"] != (2 if is_v2_run else 1)
            or validation["validator"] != "policy_improvement_full_runtime"
            or validation["validator_execution_identity"] != expected_execution_identity
            or validation["run_id"] != run.row["run_id"]
            or validation["method_id"] != run.row["method_id"]
            or validation["snapshot_kind"] != snapshot_kind
            or validation["environment_interactions"] != environment_interactions
            or validation["checkpoint_sha256"] != checkpoint_sha256
            or not isinstance(role_hashes, Mapping)
            or not role_hashes
            or not isinstance(model_state_sha256, str)
            or hashlib.sha256(canonical_json_bytes(role_hashes)).hexdigest()
            != model_state_sha256
            or validation["strict_resume_validated"] is not True
        ):
            raise FullRuntimeError(
                f"Published {snapshot_kind} validation differs from its checkpoint."
            )
        validation_theory_identity = _validated_theory_model_identity(
            validation.get("theory_model_identity"),
            required=theory_identity_required,
            name=f"Published {snapshot_kind} theory identity",
        )
        if validation_theory_identity is not None:
            role_names = {
                "fixed_base_exact_persistent": (
                    "policy_model_old",
                    "policy_model_candidate",
                ),
                "fixed_base_exact_episodic": (
                    "policy_model_old",
                    "policy_model_candidate",
                ),
                "legacy_parameter_interpolation": (
                    "preinterpolation_policy_base",
                    "preinterpolation_policy_candidate",
                ),
                "fixed_base_distilled_realization": ("base", "candidate"),
            }.get(str(run.row.get("method_id")))
            if (
                validation_theory_identity["model_sha256"] != model_state_sha256
                or role_names is None
                or validation_theory_identity["current_policy_sha256"]
                != role_hashes.get(role_names[0])
                or validation_theory_identity["candidate_policy_sha256"]
                != role_hashes.get(role_names[1])
            ):
                raise FullRuntimeError(
                    f"Published {snapshot_kind} theory identity differs from its "
                    "model roles."
                )
        parent = validation["parent_checkpoint_sha256"]
        if parent is not None and (
            not isinstance(parent, str)
            or len(parent) != 64
            or any(character not in "0123456789abcdef" for character in parent)
        ):
            raise FullRuntimeError(
                f"Published {snapshot_kind} checkpoint parent is invalid."
            )
        return AuthenticatedFullCheckpoint(
            path=generation / canonical_checkpoint,
            sha256=str(checkpoint_sha256),
            size_bytes=int(checkpoint_identity["bytes"]),
            snapshot_kind=snapshot_kind,
            environment_interactions=environment_interactions,
            model_state_sha256=model_state_sha256,
            role_state_sha256s={
                str(name): str(digest) for name, digest in role_hashes.items()
            },
            theory_model_identity=validation_theory_identity,
            parent_checkpoint_sha256=parent,
            generation_manifest_sha256=generation_manifest_sha256,
            run_manifest_sha256=run_manifest_sha256,
            validation_sha256=str(validation_sha256),
        )

    schedule = run_manifest["checkpoint_schedule"]
    expected_schedule = run.interaction_checkpoints[:-1]
    if not isinstance(schedule, list) or len(schedule) != len(expected_schedule):
        raise FullRuntimeError("Published checkpoint schedule length differs.")
    checkpoints: list[AuthenticatedFullCheckpoint] = []
    for index, expected_interactions in enumerate(expected_schedule):
        entry = schedule[index]
        if not isinstance(entry, Mapping) or set(entry) != {
            "environment_interactions",
            "checkpoint_sha256",
            "resume_validation_sha256",
        }:
            raise FullRuntimeError("Published checkpoint schedule fields differ.")
        if entry["environment_interactions"] != expected_interactions:
            raise FullRuntimeError("Published checkpoint schedule order differs.")
        directory = f"checkpoints/scheduled/env_{expected_interactions:09d}/"
        matching_names = [
            name
            for name, identity in canonical_files.items()
            if name.startswith(directory)
            and identity["sha256"] == entry["checkpoint_sha256"]
            and not Path(name).name.startswith("model_step_")
        ]
        if len(matching_names) != 1:
            raise FullRuntimeError(
                "Published scheduled checkpoint is missing or ambiguous."
            )
        checkpoints.append(
            validated_checkpoint(
                checkpoint_name=matching_names[0],
                checkpoint_sha256=entry["checkpoint_sha256"],
                validation_name=(
                    f"validations/scheduled/env_{expected_interactions:09d}/"
                    "resume_validation.json"
                ),
                validation_sha256=entry["resume_validation_sha256"],
                snapshot_kind="scheduled",
                environment_interactions=expected_interactions,
            )
        )

    raw_snapshots = run_manifest["snapshots"]
    if not isinstance(raw_snapshots, list) or len(raw_snapshots) != 2:
        raise FullRuntimeError("Published snapshot inventory differs.")
    snapshot_kinds = ("interaction_matched", "compute_matched")
    snapshots: dict[str, AuthenticatedFullCheckpoint] = {}
    snapshot_recurrent_work: dict[str, int] = {}
    for index, expected_kind in enumerate(snapshot_kinds):
        snapshot = raw_snapshots[index]
        if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "snapshot_kind",
            "checkpoint_path",
            "checkpoint_sha256",
            "environment_interactions",
            "recurrent_map_applications",
        }:
            raise FullRuntimeError("Published snapshot fields differ.")
        interactions = snapshot["environment_interactions"]
        recurrent = snapshot["recurrent_map_applications"]
        if (
            snapshot["snapshot_kind"] != expected_kind
            or isinstance(interactions, bool)
            or not isinstance(interactions, int)
            or interactions <= 0
            or interactions > run.final_environment_interactions
            or isinstance(recurrent, bool)
            or not isinstance(recurrent, int)
            or recurrent < 0
        ):
            raise FullRuntimeError("Published snapshot schedule differs.")
        expected_prefix = f"checkpoints/{expected_kind}/"
        checkpoint_name = _canonical_relative(
            str(snapshot["checkpoint_path"]), name=f"{expected_kind} checkpoint path"
        )
        if not checkpoint_name.startswith(expected_prefix):
            raise FullRuntimeError("Published snapshot checkpoint path differs.")
        snapshots[expected_kind] = validated_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_sha256=snapshot["checkpoint_sha256"],
            validation_name=f"validations/{expected_kind}/checkpoint_validation.json",
            validation_sha256=canonical_files.get(
                f"validations/{expected_kind}/checkpoint_validation.json", {}
            ).get("sha256"),
            snapshot_kind=expected_kind,
            environment_interactions=interactions,
        )
        snapshot_recurrent_work[expected_kind] = recurrent
    interaction = snapshots["interaction_matched"]
    if (
        interaction.environment_interactions != run.final_environment_interactions
        or run_manifest["checkpoint_sha256"] != interaction.sha256
        or primary_name != interaction.path.relative_to(generation).as_posix()
        or primary["sha256"] != interaction.sha256
        or _available_digest(
            result_artifacts.get("checkpoint"), name="result checkpoint"
        )
        != interaction.sha256
        or _available_digest(
            result_identities.get("checkpoint_sha256"),
            name="result checkpoint identity",
        )
        != interaction.sha256
    ):
        raise FullRuntimeError("Published interaction checkpoint binding differs.")

    raw_bindings = model_inventory["snapshot_state_bindings"]
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 2:
        raise FullRuntimeError("Published model-state snapshot bindings differ.")
    bindings: dict[str, Mapping[str, object]] = {}
    for raw_binding in raw_bindings:
        binding_fields = {
            "snapshot_kind",
            "checkpoint_sha256",
            "model_state_sha256",
            "checkpoint_validation_sha256",
        }
        if is_v2_run:
            binding_fields.add("theory_model_identity")
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != binding_fields:
            raise FullRuntimeError("Published model-state binding fields differ.")
        kind = raw_binding["snapshot_kind"]
        if kind not in snapshots or kind in bindings:
            raise FullRuntimeError("Published model-state binding kinds differ.")
        bindings[str(kind)] = raw_binding
    for kind, checkpoint in snapshots.items():
        binding = bindings.get(kind)
        if binding is None or dict(binding) != {
            "snapshot_kind": kind,
            "checkpoint_sha256": checkpoint.sha256,
            "model_state_sha256": checkpoint.model_state_sha256,
            "checkpoint_validation_sha256": checkpoint.validation_sha256,
            **(
                {"theory_model_identity": checkpoint.theory_model_identity}
                if is_v2_run
                else {}
            ),
        }:
            raise FullRuntimeError("Published model-state binding differs.")

    if (
        run_manifest["model_state_sha256"] != interaction.model_state_sha256
        or run_manifest["model_state_sha256s"] != interaction.role_state_sha256s
        or _available_digest(
            result_artifacts.get("checkpoint_validation"),
            name="result checkpoint validation",
        )
        != interaction.validation_sha256
        or _available_digest(
            result_identities.get("model_state_sha256"),
            name="result model-state identity",
        )
        != interaction.model_state_sha256
    ):
        raise FullRuntimeError(
            "Published result does not bind its primary validation and model state."
        )
    if is_v2_run and interaction.theory_model_identity != inventory_theory_identity:
        raise FullRuntimeError(
            "Published model-state inventory and interaction checkpoint theory "
            "identities differ."
        )
    result_snapshots = result.get("evaluation_snapshots")
    if not isinstance(result_snapshots, list) or len(result_snapshots) != 2:
        raise FullRuntimeError("Published result snapshot inventory differs.")
    result_snapshots_by_kind = {
        snapshot.get("snapshot_kind"): snapshot
        for snapshot in result_snapshots
        if isinstance(snapshot, Mapping)
    }
    if set(result_snapshots_by_kind) != set(snapshots):
        raise FullRuntimeError("Published result snapshot kinds differ.")
    for kind, checkpoint in snapshots.items():
        snapshot = result_snapshots_by_kind[kind]
        expected_lineage_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_name": "policy_improvement_full_checkpoint_lineage_v1",
                    "run_id": run.row["run_id"],
                    "snapshot_kind": kind,
                    "parent_checkpoint_sha256": (checkpoint.parent_checkpoint_sha256),
                    "checkpoint_sha256": checkpoint.sha256,
                    "environment_interactions": (checkpoint.environment_interactions),
                    "recurrent_map_applications": snapshot_recurrent_work[kind],
                }
            )
        ).hexdigest()
        if (
            _available_digest(
                snapshot.get("checkpoint_sha256"),
                name=f"result {kind} checkpoint",
            )
            != checkpoint.sha256
            or _available_digest(
                snapshot.get("model_state_sha256"),
                name=f"result {kind} model state",
            )
            != checkpoint.model_state_sha256
            or snapshot.get("observed_environment_interactions")
            != {
                "status": "available",
                "value": checkpoint.environment_interactions,
            }
            or snapshot.get("observed_recurrent_map_applications")
            != {
                "status": "available",
                "value": snapshot_recurrent_work[kind],
            }
            or _available_digest(
                snapshot.get("checkpoint_lineage_sha256"),
                name=f"result {kind} checkpoint lineage",
            )
            != expected_lineage_sha256
        ):
            raise FullRuntimeError(
                f"Published result {kind} snapshot differs from its checkpoint."
            )

    lineage = [*checkpoints, snapshots["compute_matched"], interaction]
    lineage.sort(
        key=lambda item: (
            item.environment_interactions,
            {"compute_matched": 0, "scheduled": 1, "interaction_matched": 2}[
                item.snapshot_kind
            ],
        )
    )
    if len({item.sha256 for item in lineage}) != len(lineage):
        raise FullRuntimeError("Published checkpoint lineage aliases one checkpoint.")
    parent: str | None = None
    for checkpoint in lineage:
        if checkpoint.parent_checkpoint_sha256 != parent:
            raise FullRuntimeError("Published checkpoint parent lineage differs.")
        parent = checkpoint.sha256

    selected = (
        interaction
        if checkpoint_environment_interactions == run.final_environment_interactions
        else next(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.environment_interactions
            == checkpoint_environment_interactions
        )
    )
    files_after, directories_after = _inventory(generation)
    current_generation = generation.lstat()
    if (
        files_after != {**actual_files, "MANIFEST.json": manifest_identity}
        or directories_after != actual_directories
        or (current_generation.st_dev, current_generation.st_ino) != generation_identity
    ):
        raise FullRuntimeError(
            "Published full-run generation changed during checkpoint resolution."
        )
    sealed_descriptor = _seal_authenticated_checkpoint(
        selected.path,
        expected_sha256=selected.sha256,
        expected_size_bytes=selected.size_bytes,
    )
    return replace(selected, sealed_descriptor=sealed_descriptor)


def _finish_complete_generation(
    run: RegisteredFullRun,
    generation: Path,
    package: BackendPackage,
    *,
    prior_failed_attempts: Sequence[Mapping[str, object]],
    result_validator: Callable[[object], Mapping[str, object]],
) -> None:
    result_document = _validate_result_contract(
        run, package.result, result_validator=result_validator
    )
    _, result = validated_result_payload(result_document)
    result_path = generation / "result.json"
    expected_result_bytes = canonical_json_bytes(result_document) + b"\n"
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
    validation_files = {
        "validations/interaction_matched/checkpoint_validation.json",
        "validations/compute_matched/checkpoint_validation.json",
    }
    if not validation_files.issubset(files):
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
        "prior_failed_attempts": [dict(item) for item in prior_failed_attempts],
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
) -> Path:
    result_document, result = _validated_result_for_run(
        run,
        failure.result,
        result_validator=result_validator,
    )
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
    attempt_id = f"{time.time_ns():032x}"
    attempt = staging_run / "attempts" / "complete" / attempt_id
    attempt.mkdir(parents=True, mode=0o700)
    result_sha256 = _write_exclusive_json(attempt / "result.json", result_document)
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
    return attempt


def _prior_failed_attempts(run_root: Path) -> list[dict[str, object]]:
    attempts_root = run_root / "attempts"
    try:
        attempts_status = attempts_root.lstat()
    except FileNotFoundError:
        return []
    if stat.S_ISLNK(attempts_status.st_mode) or not stat.S_ISDIR(
        attempts_status.st_mode
    ):
        raise FullRuntimeError("Failed-attempt root is unsafe.")
    complete_root = attempts_root / "complete"
    try:
        complete_status = complete_root.lstat()
    except FileNotFoundError as exc:
        raise FullRuntimeError(
            "Failed-attempt segment inventory is incomplete."
        ) from exc
    if stat.S_ISLNK(complete_status.st_mode) or not stat.S_ISDIR(
        complete_status.st_mode
    ):
        raise FullRuntimeError("Failed-attempt segment is unsafe.")
    commitments: list[dict[str, object]] = []
    for attempt in sorted(complete_root.iterdir(), key=lambda path: path.name):
        if len(attempt.name) != 32 or any(
            character not in "0123456789abcdef" for character in attempt.name
        ):
            raise FullRuntimeError("Failed-attempt ID is invalid.")
        manifest_value, manifest_sha256 = _load_authenticated_json(
            attempt / "MANIFEST.json"
        )
        if not isinstance(manifest_value, Mapping):
            raise FullRuntimeError("Failed-attempt manifest is invalid.")
        result_identity = manifest_value.get("result")
        if (
            manifest_value.get("schema_name")
            != "policy_improvement_run_failed_attempt_v1"
            or manifest_value.get("attempt_id") != attempt.name
            or manifest_value.get("segment") != "complete"
            or not isinstance(result_identity, Mapping)
            or result_identity.get("path") != "result.json"
        ):
            raise FullRuntimeError("Failed-attempt manifest identity differs.")
        _, result_file_identity = _stable_regular_file(attempt / "result.json")
        if result_identity.get("sha256") != result_file_identity["sha256"]:
            raise FullRuntimeError("Failed-attempt result digest differs.")
        commitments.append(
            {
                "segment": "complete",
                "attempt_id": attempt.name,
                "generation_manifest_sha256": manifest_sha256,
                "result_sha256": result_file_identity["sha256"],
            }
        )
    if not commitments:
        raise FullRuntimeError("Failed-attempt root is empty.")
    return commitments


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
    final_run = _ensure_private_directory(runs, PurePosixPath(str(run.row["run_id"])))
    final_run_identity = (final_run.stat().st_dev, final_run.stat().st_ino)
    final_segment = (
        final_run / "segments" / f"env_{run.final_environment_interactions:09d}"
    )
    staging_run = Path(tempfile.mkdtemp(prefix=".full-run-stage.", dir=runs))
    staging_identity = (staging_run.stat().st_dev, staging_run.stat().st_ino)
    publication_lock = _acquire_run_publication_lock(runs, str(run.row["run_id"]))
    try:
        if final_segment.exists():
            raise FullRuntimeError("This immutable full run is already complete.")
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
            prior_failed_attempts = _prior_failed_attempts(final_run)
            _finish_complete_generation(
                run,
                generation,
                package,
                prior_failed_attempts=prior_failed_attempts,
                result_validator=result_validator,
            )
        except FullRunFailure as failure:
            shutil.rmtree(staging_run / "segments")
            attempt = _finish_failed_attempt(
                run,
                staging_run,
                failure,
                result_validator=result_validator,
            )
            if final_segment.exists():
                raise FullRuntimeError(
                    "A failed attempt cannot be appended after run completion."
                )
            attempts = _ensure_private_directory(
                final_run,
                PurePosixPath("attempts", "complete"),
            )
            published_attempt = attempts / attempt.name
            _rename_noreplace(attempt, published_attempt)
            _fsync_directory(attempts)
            _fsync_directory(attempts.parent)
            _fsync_directory(final_run)
            raise PublishedFullRunFailure(final_run, failure)

        if _prior_failed_attempts(final_run) != prior_failed_attempts:
            raise FullRuntimeError(
                "Failed-attempt inventory changed before complete publication."
            )
        current_runs = runs.lstat()
        current_run = final_run.lstat()
        if (
            stat.S_ISLNK(current_runs.st_mode)
            or not stat.S_ISDIR(current_runs.st_mode)
            or (current_runs.st_dev, current_runs.st_ino) != runs_identity
            or stat.S_ISLNK(current_run.st_mode)
            or not stat.S_ISDIR(current_run.st_mode)
            or (current_run.st_dev, current_run.st_ino) != final_run_identity
        ):
            raise FullRuntimeError("Publication root changed before atomic rename.")
        segments = _ensure_private_directory(final_run, PurePosixPath("segments"))
        _rename_noreplace(generation, final_segment)
        _fsync_directory(segments)
        _fsync_directory(final_run)
        return final_run
    finally:
        try:
            current = staging_run.lstat()
            if (current.st_dev, current.st_ino) == staging_identity:
                shutil.rmtree(staging_run)
                _fsync_directory(runs)
        except OSError:
            pass
        try:
            current_run = final_run.lstat()
            if (
                current_run.st_dev,
                current_run.st_ino,
            ) == final_run_identity and not any(final_run.iterdir()):
                final_run.rmdir()
                _fsync_directory(runs)
        except OSError:
            pass
        fcntl.flock(publication_lock, fcntl.LOCK_UN)
        os.close(publication_lock)


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
        "dataset_root": str(run.dataset_root) if run.dataset_root is not None else None,
        "required_entrypoint_call": {
            "callable": "execute_registered_run",
            "backend": "sealed packaged FullRunBackend",
            "source_tree_fallback": False,
        },
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: FullRunBackend | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--amendment", action="append", default=[])
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--dataset-root", required=True)
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
        dataset_root=arguments.dataset_root,
    )
    if arguments.print_contract:
        contract = execution_contract(run)
        if backend is not None:
            contract["execution_ready"] = True
            contract["blocked_by"] = []
        print(canonical_json_bytes(contract).decode("ascii"))
        return 0 if backend is not None else 2
    if backend is None:
        print(canonical_json_bytes(execution_contract(run)).decode("ascii"))
        print(
            "Full execution requires the authenticated packaged entrypoint.",
            file=os.sys.stderr,
        )
        return 2
    try:
        final = execute_registered_run(run, backend=backend)
    except PublishedFullRunFailure as failure:
        completion = {
            "schema_name": "policy_improvement_full_runtime_completion_v1",
            "schema_version": 1,
            "status": "failed",
            "run_id": run.row["run_id"],
            "published_run": str(failure.published_run),
            "failure_phase": failure.phase,
            "runtime_authorization_sha256": run.runtime_authorization_sha256,
        }
        print(canonical_json_bytes(completion).decode("ascii"))
        return 1
    completion = {
        "schema_name": "policy_improvement_full_runtime_completion_v1",
        "schema_version": 1,
        "status": "complete",
        "run_id": run.row["run_id"],
        "published_run": str(final),
        "runtime_authorization_sha256": run.runtime_authorization_sha256,
    }
    print(canonical_json_bytes(completion).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
