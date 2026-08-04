"""Fail-closed identity for reproducible training runs and exact resumes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RUN_IDENTITY_SCHEMA_VERSION = 1
CHECKPOINT_LINEAGE_SCHEMA_VERSION = 1
_RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$"
)
_LOWER_HEX_40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INITIALIZATION_KINDS = {"random", "weights_checkpoint"}
_MAX_TRAINING_SEED = 2**32 - 1


class RunIdentityError(RuntimeError):
    """Raised when a run identity is incomplete, ambiguous, or mismatched."""


def _canonical_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunIdentityError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise RunIdentityError(f"{path} has a non-string key.")
        return {
            key: _canonical_json_value(value[key], path=f"{path}.{key}")
            for key in sorted(keys)
        }
    if isinstance(value, list):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RunIdentityError(
        f"{path} contains unsupported JSON value type {type(value).__name__}."
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict JSON deterministically for content-addressed metadata."""

    canonical = _canonical_json_value(value, path="value")
    return json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without retaining its filesystem path in run metadata."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RunIdentityError("Initialization artifact cannot be read.") from exc
    return digest.hexdigest()


def _run_git(lookup_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(lookup_root),
                *arguments,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunIdentityError("Git source identity could not be inspected.") from exc
    if completed.returncode != 0:
        raise RunIdentityError("Git source identity could not be inspected.")
    return completed.stdout.strip()


def discover_clean_git_source(lookup_root: str | Path) -> dict[str, object]:
    """Return a path-free commit identity, rejecting dirty or unstable trees."""

    root = Path(lookup_root).expanduser()
    if not root.exists():
        raise RunIdentityError("Git lookup root does not exist.")
    if _run_git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise RunIdentityError("Git lookup root is not inside a work tree.")

    commit_before = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    if not _LOWER_HEX_40_PATTERN.fullmatch(commit_before):
        raise RunIdentityError("Producer Git commit must be 40 lowercase hex characters.")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    commit_after = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    if commit_after != commit_before:
        raise RunIdentityError("Producer Git HEAD changed during identity discovery.")
    if status:
        raise RunIdentityError("Producer Git work tree is not clean.")
    return {"git_commit": commit_before, "git_clean": True}


def assert_git_files_match_head(
    lookup_root: str | Path,
    relative_paths: list[str],
) -> None:
    """Require tracked worktree bytes and index flags to match the named HEAD."""

    root = Path(lookup_root).expanduser().resolve()
    top_level = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise RunIdentityError("Git lookup root must be the repository top level.")
    if not relative_paths:
        raise RunIdentityError("At least one producer source must be verified.")
    for relative_path in relative_paths:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise RunIdentityError("Producer source paths must be repository-relative.")
        index_entry = _run_git(root, "ls-files", "-v", "--", relative_path)
        if not index_entry or index_entry[0] != "H":
            raise RunIdentityError(
                f"Producer source {relative_path!r} is untracked or has unsafe index flags."
            )
        worktree_blob = _run_git(root, "hash-object", "--", relative_path)
        head_blob = _run_git(root, "rev-parse", f"HEAD:{relative_path}")
        if worktree_blob != head_blob:
            raise RunIdentityError(
                f"Producer source {relative_path!r} does not match the recorded commit."
            )


def _require_exact_fields(
    value: object,
    *,
    expected: set[str],
    path: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise RunIdentityError(f"{path} must be a string-keyed mapping.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RunIdentityError(
            f"{path} has missing fields {missing} and unknown fields {unknown}."
        )
    return value


def _require_sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not _LOWER_HEX_64_PATTERN.fullmatch(value):
        raise RunIdentityError(f"{path} must be 64 lowercase hex characters.")
    return value


def _validate_run_id(value: object) -> str:
    if not isinstance(value, str) or not _RUN_ID_PATTERN.fullmatch(value):
        raise RunIdentityError(
            "run_id must be 1-128 ASCII letters, digits, '.', '_', or '-', "
            "with an alphanumeric first and last character."
        )
    return value


def validate_run_id(value: object) -> str:
    """Validate a path-safe run identifier without constructing a full identity."""

    return _validate_run_id(value)


def _validate_training_seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_TRAINING_SEED
    ):
        raise RunIdentityError(
            f"training_seed must be an integer in [0, {_MAX_TRAINING_SEED}]."
        )
    return value


def _require_nonnegative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunIdentityError(f"{path} must be a non-negative integer.")
    return value


def _require_optional_nonnegative_int(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, path=path)


def _require_finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunIdentityError(f"{path} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise RunIdentityError(f"{path} must be a finite number.")
    return converted


def validate_upi_effective_config(value: object) -> dict[str, Any]:
    """Validate the exact path-free configuration schema used by UPI runs."""

    canonical_value = _canonical_json_value(value, path="effective_config")
    if not isinstance(canonical_value, Mapping):
        raise RunIdentityError("effective_config must be a string-keyed mapping.")
    schema_version = canonical_value.get("effective_config_schema_version")
    expected_fields = {
            "effective_config_schema_version",
            "algorithm",
            "training_protocol",
            "backbone",
            "rl_config",
            "model_config",
            "execution_device",
            "runtime_fingerprint_sha256",
            "dataset",
            "budget",
            "schedule",
            "evaluation",
            "puzzle_embedding_optimizer",
            "imitation",
            "external_logging",
            "debug_checks",
            "config_source_sha256s",
    }
    if schema_version in {2, 3}:
        expected_fields.update(
            {
                "registration",
                "dataset_provenance_sha256",
                "initialization",
            }
        )
    elif schema_version != 1:
        raise RunIdentityError("Unsupported effective configuration schema.")
    config = _require_exact_fields(
        canonical_value,
        expected=expected_fields,
        path="effective_config",
    )
    if config["algorithm"] != "upi_trm":
        raise RunIdentityError("effective_config.algorithm must be 'upi_trm'.")
    if config["training_protocol"] != "fixed_base_exact":
        raise RunIdentityError(
            "effective_config.training_protocol must be 'fixed_base_exact'."
        )
    if config["backbone"] != "trm":
        raise RunIdentityError("Schema-v5 UPI checkpoints require the TRM backbone.")
    for name in ("rl_config", "model_config"):
        nested = config[name]
        if not isinstance(nested, Mapping) or not nested:
            raise RunIdentityError(f"effective_config.{name} must be nonempty.")
    execution_device = config["execution_device"]
    if not isinstance(execution_device, str) or not re.fullmatch(
        r"cpu|cuda:[0-9]+", execution_device
    ):
        raise RunIdentityError(
            "effective_config.execution_device must be 'cpu' or indexed CUDA."
        )
    _require_sha256(
        config["runtime_fingerprint_sha256"],
        path="effective_config.runtime_fingerprint_sha256",
    )
    if schema_version in {2, 3}:
        registration_fields = {
            "cell",
            "tier",
            "run_id",
            "training_seed",
            "registry_sha256",
        }
        if schema_version == 3:
            registration_fields.add("attempt_index")
        registration = _require_exact_fields(
            config["registration"],
            expected=registration_fields,
            path="effective_config.registration",
        )
        _validate_run_id(registration["cell"])
        if registration["tier"] not in {"confirmatory", "debug"}:
            raise RunIdentityError(
                "effective_config.registration.tier is unsupported."
            )
        _validate_run_id(registration["run_id"])
        _validate_training_seed(registration["training_seed"])
        if schema_version == 3:
            _require_nonnegative_int(
                registration["attempt_index"],
                path="effective_config.registration.attempt_index",
            )
        _require_sha256(
            registration["registry_sha256"],
            path="effective_config.registration.registry_sha256",
        )
        _require_sha256(
            config["dataset_provenance_sha256"],
            path="effective_config.dataset_provenance_sha256",
        )
        initialization = _require_exact_fields(
            config["initialization"],
            expected={"kind", "artifact_sha256"},
            path="effective_config.initialization",
        )
        kind = initialization["kind"]
        if kind not in _INITIALIZATION_KINDS:
            raise RunIdentityError(
                "effective_config.initialization.kind is unsupported."
            )
        if kind == "random":
            if initialization["artifact_sha256"] is not None:
                raise RunIdentityError(
                    "Random effective initialization must not name an artifact."
                )
        else:
            _require_sha256(
                initialization["artifact_sha256"],
                path="effective_config.initialization.artifact_sha256",
            )

    dataset = _require_exact_fields(
        config["dataset"],
        expected={
            "train_split",
            "eval_split",
            "train_record_count",
            "eval_record_count",
        },
        path="effective_config.dataset",
    )
    for split in ("train_split", "eval_split"):
        if not isinstance(dataset[split], str) or not dataset[split]:
            raise RunIdentityError(f"effective_config.dataset.{split} must be nonempty.")
    _require_nonnegative_int(
        dataset["train_record_count"],
        path="effective_config.dataset.train_record_count",
    )
    _require_nonnegative_int(
        dataset["eval_record_count"],
        path="effective_config.dataset.eval_record_count",
    )

    budget = _require_exact_fields(
        config["budget"],
        expected={"outer_train_steps", "environment_interactions"},
        path="effective_config.budget",
    )
    _require_nonnegative_int(
        budget["outer_train_steps"],
        path="effective_config.budget.outer_train_steps",
    )
    _require_optional_nonnegative_int(
        budget["environment_interactions"],
        path="effective_config.budget.environment_interactions",
    )

    schedule = _require_exact_fields(
        config["schedule"],
        expected={
            "log_outer_interval",
            "eval_outer_interval",
            "save_outer_interval",
            "log_environment_interval",
            "eval_environment_interval",
            "save_environment_interval",
        },
        path="effective_config.schedule",
    )
    for field in (
        "log_outer_interval",
        "eval_outer_interval",
        "save_outer_interval",
        "log_environment_interval",
        "eval_environment_interval",
        "save_environment_interval",
    ):
        _require_optional_nonnegative_int(
            schedule[field],
            path=f"effective_config.schedule.{field}",
        )

    evaluation = _require_exact_fields(
        config["evaluation"],
        expected={"episode_count", "seed", "pool_size"},
        path="effective_config.evaluation",
    )
    _require_nonnegative_int(
        evaluation["episode_count"],
        path="effective_config.evaluation.episode_count",
    )
    _require_nonnegative_int(
        evaluation["seed"],
        path="effective_config.evaluation.seed",
    )
    _require_nonnegative_int(
        evaluation["pool_size"],
        path="effective_config.evaluation.pool_size",
    )

    puzzle_optimizer = _require_exact_fields(
        config["puzzle_embedding_optimizer"],
        expected={"learning_rate", "weight_decay"},
        path="effective_config.puzzle_embedding_optimizer",
    )
    for field in ("learning_rate", "weight_decay"):
        if _require_finite_number(
            puzzle_optimizer[field],
            path=f"effective_config.puzzle_embedding_optimizer.{field}",
        ) < 0:
            raise RunIdentityError(
                f"effective_config.puzzle_embedding_optimizer.{field} must be non-negative."
            )

    imitation = _require_exact_fields(
        config["imitation"],
        expected={"enabled", "epochs"},
        path="effective_config.imitation",
    )
    if not isinstance(imitation["enabled"], bool):
        raise RunIdentityError("effective_config.imitation.enabled must be boolean.")
    _require_nonnegative_int(
        imitation["epochs"],
        path="effective_config.imitation.epochs",
    )
    if config["external_logging"] != "disabled":
        raise RunIdentityError("effective_config.external_logging must be 'disabled'.")
    if not isinstance(config["debug_checks"], bool):
        raise RunIdentityError("effective_config.debug_checks must be boolean.")
    source_hashes = config["config_source_sha256s"]
    if not isinstance(source_hashes, list):
        raise RunIdentityError("effective_config.config_source_sha256s must be a list.")
    for index, digest in enumerate(source_hashes):
        _require_sha256(
            digest,
            path=f"effective_config.config_source_sha256s[{index}]",
        )
    return dict(config)


def validate_run_identity(identity: object) -> dict[str, Any]:
    """Validate and canonicalize a schema-1 run identity."""

    top = _require_exact_fields(
        identity,
        expected={
            "run_identity_schema_version",
            "run_id",
            "training_seed",
            "producer",
            "effective_config",
            "effective_config_sha256",
            "dataset_provenance_sha256",
            "initialization",
        },
        path="run_identity",
    )
    if top["run_identity_schema_version"] != RUN_IDENTITY_SCHEMA_VERSION:
        raise RunIdentityError(
            "Unsupported run identity schema version "
            f"{top['run_identity_schema_version']!r}."
        )
    run_id = _validate_run_id(top["run_id"])
    training_seed = _validate_training_seed(top["training_seed"])

    producer = _require_exact_fields(
        top["producer"],
        expected={"git_commit", "git_clean"},
        path="run_identity.producer",
    )
    git_commit = producer["git_commit"]
    if not isinstance(git_commit, str) or not _LOWER_HEX_40_PATTERN.fullmatch(
        git_commit
    ):
        raise RunIdentityError(
            "run_identity.producer.git_commit must be 40 lowercase hex characters."
        )
    if producer["git_clean"] is not True:
        raise RunIdentityError("run_identity.producer.git_clean must be true.")

    effective_config = validate_upi_effective_config(top["effective_config"])
    effective_config_sha256 = _require_sha256(
        top["effective_config_sha256"],
        path="run_identity.effective_config_sha256",
    )
    if canonical_json_sha256(effective_config) != effective_config_sha256:
        raise RunIdentityError(
            "run_identity.effective_config_sha256 does not match effective_config."
        )
    dataset_provenance_sha256 = _require_sha256(
        top["dataset_provenance_sha256"],
        path="run_identity.dataset_provenance_sha256",
    )

    initialization = _require_exact_fields(
        top["initialization"],
        expected={"kind", "artifact_sha256"},
        path="run_identity.initialization",
    )
    initialization_kind = initialization["kind"]
    if initialization_kind not in _INITIALIZATION_KINDS:
        raise RunIdentityError(
            "run_identity.initialization.kind must be 'random' or 'weights_checkpoint'."
        )
    artifact_sha256 = initialization["artifact_sha256"]
    if initialization_kind == "random":
        if artifact_sha256 is not None:
            raise RunIdentityError(
                "Random initialization must not name an initialization artifact."
            )
    else:
        artifact_sha256 = _require_sha256(
            artifact_sha256,
            path="run_identity.initialization.artifact_sha256",
        )

    if effective_config["effective_config_schema_version"] in {2, 3}:
        registration = effective_config["registration"]
        if not isinstance(registration, Mapping):
            raise RunIdentityError("effective_config.registration is invalid.")
        if registration["run_id"] != run_id:
            raise RunIdentityError(
                "Effective configuration run_id differs from run identity."
            )
        if registration["training_seed"] != training_seed:
            raise RunIdentityError(
                "Effective configuration seed differs from run identity."
            )
        if (
            effective_config["dataset_provenance_sha256"]
            != dataset_provenance_sha256
        ):
            raise RunIdentityError(
                "Effective configuration dataset identity differs from run identity."
            )
        if effective_config["initialization"] != {
            "kind": initialization_kind,
            "artifact_sha256": artifact_sha256,
        }:
            raise RunIdentityError(
                "Effective configuration initialization differs from run identity."
            )

    canonical: dict[str, Any] = {
        "run_identity_schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "run_id": run_id,
        "training_seed": training_seed,
        "producer": {
            "git_commit": git_commit,
            "git_clean": True,
        },
        "effective_config": effective_config,
        "effective_config_sha256": effective_config_sha256,
        "dataset_provenance_sha256": dataset_provenance_sha256,
        "initialization": {
            "kind": initialization_kind,
            "artifact_sha256": artifact_sha256,
        },
    }
    # Exercise the strict encoder here so callers never receive an identity that
    # fails only when a checkpoint is written.
    canonical_json_bytes(canonical)
    return canonical


def build_run_identity(
    *,
    run_id: str,
    training_seed: int,
    git_lookup_root: str | Path,
    effective_config: Mapping[str, object],
    dataset_provenance: Mapping[str, object],
    initialization_kind: str,
    initialization_artifact_sha256: str | None,
) -> dict[str, Any]:
    """Build a validated identity from effective inputs and a clean Git tree."""

    canonical_config = validate_upi_effective_config(effective_config)
    identity = {
        "run_identity_schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "run_id": run_id,
        "training_seed": training_seed,
        "producer": discover_clean_git_source(git_lookup_root),
        "effective_config": canonical_config,
        "effective_config_sha256": canonical_json_sha256(canonical_config),
        "dataset_provenance_sha256": canonical_json_sha256(dataset_provenance),
        "initialization": {
            "kind": initialization_kind,
            "artifact_sha256": initialization_artifact_sha256,
        },
    }
    return validate_run_identity(identity)


def run_identity_sha256(identity: object) -> str:
    """Hash a fully validated run identity."""

    return canonical_json_sha256(validate_run_identity(identity))


def build_checkpoint_lineage(
    *,
    parent_checkpoint_sha256: str | None,
    parent_checkpoint_step: int | None,
    parent_environment_steps: int | None,
) -> dict[str, Any]:
    """Build the immutable parent link stored in a schema-v5 checkpoint."""

    return validate_checkpoint_lineage(
        {
            "checkpoint_lineage_schema_version": CHECKPOINT_LINEAGE_SCHEMA_VERSION,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_checkpoint_step": parent_checkpoint_step,
            "parent_environment_steps": parent_environment_steps,
        }
    )


def validate_checkpoint_lineage(value: object) -> dict[str, Any]:
    """Validate a root lineage or one complete content-addressed parent link."""

    lineage = _require_exact_fields(
        value,
        expected={
            "checkpoint_lineage_schema_version",
            "parent_checkpoint_sha256",
            "parent_checkpoint_step",
            "parent_environment_steps",
        },
        path="checkpoint_lineage",
    )
    if (
        lineage["checkpoint_lineage_schema_version"]
        != CHECKPOINT_LINEAGE_SCHEMA_VERSION
    ):
        raise RunIdentityError("Unsupported checkpoint lineage schema version.")
    parent_hash = lineage["parent_checkpoint_sha256"]
    parent_step = lineage["parent_checkpoint_step"]
    parent_environment_steps = lineage["parent_environment_steps"]
    root_fields = (
        parent_hash is None,
        parent_step is None,
        parent_environment_steps is None,
    )
    if any(root_fields) and not all(root_fields):
        raise RunIdentityError(
            "Checkpoint lineage parent hash, step, and interactions must be all null or all set."
        )
    if all(root_fields):
        return {
            "checkpoint_lineage_schema_version": CHECKPOINT_LINEAGE_SCHEMA_VERSION,
            "parent_checkpoint_sha256": None,
            "parent_checkpoint_step": None,
            "parent_environment_steps": None,
        }
    normalized_hash = _require_sha256(
        parent_hash,
        path="checkpoint_lineage.parent_checkpoint_sha256",
    )
    normalized_step = _require_nonnegative_int(
        parent_step,
        path="checkpoint_lineage.parent_checkpoint_step",
    )
    normalized_environment_steps = _require_nonnegative_int(
        parent_environment_steps,
        path="checkpoint_lineage.parent_environment_steps",
    )
    return {
        "checkpoint_lineage_schema_version": CHECKPOINT_LINEAGE_SCHEMA_VERSION,
        "parent_checkpoint_sha256": normalized_hash,
        "parent_checkpoint_step": normalized_step,
        "parent_environment_steps": normalized_environment_steps,
    }


def assert_matching_run_identity(
    saved: object,
    expected: object,
) -> dict[str, Any]:
    """Return the saved identity if it exactly matches the expected identity."""

    canonical_saved = validate_run_identity(saved)
    canonical_expected = validate_run_identity(expected)
    if canonical_json_bytes(canonical_saved) != canonical_json_bytes(canonical_expected):
        differing_fields = sorted(
            key
            for key in canonical_saved
            if canonical_saved[key] != canonical_expected[key]
        )
        raise RunIdentityError(
            "Run identity mismatch in fields: " + ", ".join(differing_fields) + "."
        )
    return canonical_saved
