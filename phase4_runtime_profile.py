#!/usr/bin/env fbpython
"""Standard-library source profiles for authenticated Phase 4 runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


PHASE4_EVALUATOR_SOURCE_PROFILE = "evaluator"
PHASE4_AUDIT_SOURCE_PROFILE = "audit"
PHASE4_FIGURE_SOURCE_PROFILE = "figure"
POLICY_DATASET_BUILDER_SOURCE_PROFILE = "policy-dataset-builder"
POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE = "policy-improvement-launcher"
POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE = "policy-improvement-audit"
POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE = "policy-improvement-analysis"
POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE = "policy-improvement-full"
POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE = "policy-improvement-theory-bridge"
PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION = 1
# Producer inventory versions, duplicated from utils.source_identity because
# this module must stay standard-library only: it is imported before runtime
# attestation, and :utils pulls Torch.  A consistency test pins the two
# definitions together so they cannot drift.
PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION = 3
_PRODUCER_ROOT_SOURCES_BY_VERSION: dict[int, tuple[str, ...]] = {
    1: (
        "confirmatory_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_smoke_checkpoint.py",
        "policy_improvement_smoke_runtime.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "upi_trm_train.py",
    ),
    2: (
        "confirmatory_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_checkpoint_allowlist.py",
        "policy_improvement_smoke_checkpoint.py",
        "policy_improvement_smoke_runtime.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "upi_trm_train.py",
    ),
    3: (
        "confirmatory_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_checkpoint_allowlist.py",
        "policy_improvement_smoke_checkpoint.py",
        "policy_improvement_smoke_runtime.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "upi_trm_train.py",
    ),
}
_PRODUCER_ADDITIONAL_SOURCES_BY_VERSION: dict[int, tuple[str, ...]] = {
    1: (
        "scripts/policy_improvement_registry.py",
        "scripts/policy_improvement_schema.py",
    ),
    2: (
        "scripts/policy_improvement_registry.py",
        "scripts/policy_improvement_schema.py",
    ),
    3: (
        "scripts/policy_improvement_registry.py",
        "scripts/policy_improvement_schema.py",
        "scripts/policy_improvement_populations.py",
        "scripts/policy_improvement_v2_registry.py",
        "scripts/policy_improvement_v2_schema.py",
    ),
}
_PRODUCER_CONFIG_DIRECTORIES_BY_VERSION: dict[int, tuple[str, ...]] = {
    1: ("configs/iclr_confirmatory", "configs/policy_improvement_v1"),
    2: ("configs/iclr_confirmatory", "configs/policy_improvement_v1"),
    3: (
        "configs/iclr_confirmatory",
        "configs/policy_improvement_v1",
        "configs/policy_improvement_v2",
    ),
}
PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH = (
    "configs/iclr_confirmatory/producer_source_manifest.json"
)

PHASE4_SOURCE_DIRECTORIES = ("dataset", "models", "rl", "utils")
PHASE4_ROOT_SOURCES = (
    "phase4_runtime_entrypoint.py",
    "phase4_runtime_profile.py",
    "puzzle_dataset.py",
    "runtime_archive_preflight.py",
)
PHASE4_SHARED_SOURCES = (
    "scripts/phase4_checkpoint.py",
    "scripts/phase4_diagnostic_inputs.py",
    "scripts/phase4_result_schema.py",
    "scripts/phase4_source.py",
)
PHASE4_PROFILE_ENTRYPOINTS = {
    PHASE4_EVALUATOR_SOURCE_PROFILE: ("scripts/eval_phase4_2x2_norm_ablation.py",),
    PHASE4_AUDIT_SOURCE_PROFILE: ("scripts/audit_phase4_paper_ready.py",),
    PHASE4_FIGURE_SOURCE_PROFILE: (
        "scripts/phase4_figure_publication.py",
        "scripts/make_paper_figures_phase4.py",
    ),
    POLICY_DATASET_BUILDER_SOURCE_PROFILE: ("policy_dataset_builder_entrypoint.py",),
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE: (
        "phase4_runtime_launcher.py",
    ),
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE: (
        "policy_improvement_consumer_entrypoint.py",
        "scripts/policy_improvement_audit.py",
    ),
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE: (
        "policy_improvement_consumer_entrypoint.py",
        "scripts/policy_improvement_analysis.py",
    ),
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE: ("policy_improvement_full_entrypoint.py",),
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE: (
        "policy_improvement_theory_bridge_entrypoint.py",
    ),
}
POLICY_DATASET_BUILDER_PROFILE_PATHS = (
    "configs/policy_improvement_v1/fixed_base_exact_episodic.yaml",
    "configs/policy_improvement_v1/fixed_base_exact_persistent.yaml",
    "configs/policy_improvement_v1/legacy_parameter_interpolation.yaml",
    "configs/policy_improvement_v1/matched_ppo.yaml",
    "configs/policy_improvement_v1/protocol.json",
    "configs/policy_improvement_v1/registry.json",
    "dataset/__init__.py",
    "dataset/build_4x4_sudoku.py",
    "dataset/build_iclr_confirmatory_4x4.py",
    "dataset/build_policy_improvement_4x4.py",
    "phase4_runtime_profile.py",
    "policy_dataset_builder_entrypoint.py",
    "runtime_archive_preflight.py",
    "utils/__init__.py",
    "utils/dataset_provenance.py",
    "utils/run_identity.py",
)
POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS = (
    PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH,
    "confirmatory_runtime_launcher.py",
    "phase4_runtime_launcher.py",
    "phase4_runtime_profile.py",
)
_POLICY_IMPROVEMENT_CONFIG_PATHS = (
    "configs/policy_improvement_v1/amendments/theory_bridge_v1.json",
    "configs/policy_improvement_v1/fixed_base_exact_episodic.yaml",
    "configs/policy_improvement_v1/fixed_base_exact_persistent.yaml",
    "configs/policy_improvement_v1/legacy_parameter_interpolation.yaml",
    "configs/policy_improvement_v1/matched_ppo.yaml",
    "configs/policy_improvement_v1/protocol.json",
    "configs/policy_improvement_v1/registry.json",
    "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
    "configs/policy_improvement_v2/fixed_base_exact_episodic.yaml",
    "configs/policy_improvement_v2/fixed_base_exact_persistent.yaml",
    "configs/policy_improvement_v2/legacy_parameter_interpolation.yaml",
    "configs/policy_improvement_v2/matched_ppo.yaml",
    "configs/policy_improvement_v2/populations.json",
    "configs/policy_improvement_v2/protocol.json",
    "configs/policy_improvement_v2/registry.json",
)
_POLICY_IMPROVEMENT_V2_SOURCE_PATHS = (
    "scripts/policy_improvement_populations.py",
    "scripts/policy_improvement_v2_registry.py",
    "scripts/policy_improvement_v2_schema.py",
)
POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS = (
    *_POLICY_IMPROVEMENT_CONFIG_PATHS,
    *_POLICY_IMPROVEMENT_V2_SOURCE_PATHS,
    "policy_improvement_checkpoint_validator.py",
    "policy_improvement_consumer_entrypoint.py",
    "policy_improvement_full_backend.py",
    "policy_improvement_non_smoke_checkpoint.py",
    "policy_improvement_sealed_evidence.py",
    "runtime_archive_preflight.py",
    "scripts/policy_improvement_audit.py",
    "scripts/policy_improvement_evidence.py",
    "scripts/policy_improvement_full_runtime.py",
    "scripts/policy_improvement_throughput.py",
    "scripts/policy_improvement_registry.py",
    "scripts/policy_improvement_schema.py",
    "scripts/policy_improvement_test_open.py",
    "scripts/policy_improvement_test_open_cli.py",
    "scripts/policy_improvement_theory_schema.py",
    "scripts/policy_improvement_theory_schema_v2.py",
)
POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS = (
    *_POLICY_IMPROVEMENT_CONFIG_PATHS,
    *_POLICY_IMPROVEMENT_V2_SOURCE_PATHS,
    "policy_improvement_checkpoint_validator.py",
    "policy_improvement_consumer_entrypoint.py",
    "policy_improvement_full_backend.py",
    "policy_improvement_non_smoke_checkpoint.py",
    "policy_improvement_sealed_evidence.py",
    "runtime_archive_preflight.py",
    "scripts/policy_improvement_analysis.py",
    "scripts/policy_improvement_audit.py",
    "scripts/policy_improvement_evidence.py",
    "scripts/policy_improvement_full_runtime.py",
    "scripts/policy_improvement_throughput.py",
    "scripts/policy_improvement_registry.py",
    "scripts/policy_improvement_schema.py",
    "scripts/policy_improvement_statistics.py",
    "scripts/policy_improvement_test_open.py",
    "scripts/policy_improvement_theory_schema.py",
    "scripts/policy_improvement_theory_schema_v2.py",
)
_POLICY_IMPROVEMENT_FULL_COMMON_PROFILE_PATHS = (
    *_POLICY_IMPROVEMENT_CONFIG_PATHS,
    *_POLICY_IMPROVEMENT_V2_SOURCE_PATHS,
    "policy_improvement_full_backend.py",
    "policy_improvement_non_smoke_checkpoint.py",
    "policy_improvement_sealed_evidence.py",
    "runtime_archive_preflight.py",
    "scripts/policy_improvement_full_runtime.py",
    "scripts/policy_improvement_throughput.py",
    "scripts/policy_improvement_registry.py",
    "scripts/policy_improvement_schema.py",
    "scripts/policy_improvement_theory_schema.py",
)
POLICY_IMPROVEMENT_FULL_PROFILE_PATHS = (
    *_POLICY_IMPROVEMENT_FULL_COMMON_PROFILE_PATHS,
    "policy_improvement_full_entrypoint.py",
)
POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS = (
    *_POLICY_IMPROVEMENT_FULL_COMMON_PROFILE_PATHS,
    "policy_improvement_checkpoint_validator.py",
    "policy_improvement_theory_bridge_entrypoint.py",
    "scripts/policy_improvement_evidence.py",
    "scripts/policy_improvement_populations.py",
    "scripts/policy_improvement_theory_backend.py",
    "scripts/policy_improvement_theory_backend_v2.py",
    "scripts/policy_improvement_theory_bridge.py",
    "scripts/policy_improvement_theory_bridge_v2.py",
    "scripts/policy_improvement_theory_schema_v2.py",
    "scripts/policy_improvement_v2_schema.py",
)
_EXACT_PROFILE_PATHS = {
    POLICY_DATASET_BUILDER_SOURCE_PROFILE: POLICY_DATASET_BUILDER_PROFILE_PATHS,
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE: (
        POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS
    ),
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE: (POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS),
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE: (
        POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS
    ),
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE: POLICY_IMPROVEMENT_FULL_PROFILE_PATHS,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE: (
        POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS
    ),
}
_PROFILES_WITH_TRAINING_SOURCE = {
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
}
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_READ_SIZE = 1024 * 1024


class Phase4RuntimeProfileError(RuntimeError):
    """Raised when a Phase 4 source profile cannot be authenticated."""


def producer_root_sources(schema_version: int) -> tuple[str, ...]:
    """Root sources a producer of the given manifest schema version must have."""

    try:
        return _PRODUCER_ROOT_SOURCES_BY_VERSION[schema_version]
    except KeyError:
        raise Phase4RuntimeProfileError(
            f"Unsupported producer source manifest schema {schema_version!r}."
        ) from None


def _producer_additional_sources(schema_version: int) -> tuple[str, ...]:
    try:
        return _PRODUCER_ADDITIONAL_SOURCES_BY_VERSION[schema_version]
    except KeyError:
        raise Phase4RuntimeProfileError(
            f"Unsupported producer source manifest schema {schema_version!r}."
        ) from None


def _producer_config_directories(schema_version: int) -> tuple[str, ...]:
    try:
        return _PRODUCER_CONFIG_DIRECTORIES_BY_VERSION[schema_version]
    except KeyError:
        raise Phase4RuntimeProfileError(
            f"Unsupported producer source manifest schema {schema_version!r}."
        ) from None


def _is_registered_producer_config(
    relative_path: str,
    relative_directory: str,
) -> bool:
    path = PurePosixPath(relative_path)
    parent = PurePosixPath(relative_directory)
    if path.suffix not in {".json", ".yaml"}:
        return False
    if path.parent == parent:
        return True
    return (
        relative_directory == "configs/policy_improvement_v2"
        and path.parent == parent / "amendments"
    )


def _producer_inventory_schema_version(root: Path) -> int:
    """Highest producer inventory version satisfied by one checkout tree."""

    best = 0
    for schema_version in sorted(_PRODUCER_ROOT_SOURCES_BY_VERSION):
        required = (
            *producer_root_sources(schema_version),
            *_producer_additional_sources(schema_version),
        )
        if all((root / path).is_file() for path in required):
            best = schema_version
    if not best:
        missing = sorted(
            path for path in producer_root_sources(1) if not (root / path).is_file()
        )
        raise Phase4RuntimeProfileError(
            f"Producer repository is missing sources {missing!r}."
        )
    return best


@dataclass(frozen=True)
class AuthorizedPhase4Profile:
    """Stable checkout identity and exact profile manifest."""

    git_commit: str
    source_manifest_sha256: str
    profile: str
    sources: dict[str, str]


@dataclass(frozen=True)
class AuthorizedTrainingSource:
    """Stable producer checkout and its exact checked-in source manifest."""

    git_commit: str
    source_manifest_sha256: str
    manifest_bytes: bytes


def _run_git(root: Path, *arguments: str) -> str:
    environment = {
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
    }
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "--no-optional-locks",
                "-C",
                str(root),
                *arguments,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Phase4RuntimeProfileError(
            "Phase 4 source Git identity cannot be inspected."
        ) from exc
    if completed.returncode != 0:
        raise Phase4RuntimeProfileError(
            "Phase 4 source Git identity cannot be inspected."
        )
    return completed.stdout.strip()


def _entrypoints(profile: str) -> tuple[str, ...]:
    try:
        return PHASE4_PROFILE_ENTRYPOINTS[profile]
    except KeyError as exc:
        raise Phase4RuntimeProfileError(
            f"Unsupported Phase 4 source profile {profile!r}."
        ) from exc


def _is_directory_source(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return (
        bool(path.parts)
        and path.parts[0] in PHASE4_SOURCE_DIRECTORIES
        and path.suffix == ".py"
        and "__pycache__" not in path.parts
    )


def phase4_profile_relative_paths(root: Path, profile: str) -> list[str]:
    if profile in _EXACT_PROFILE_PATHS:
        paths = set(_EXACT_PROFILE_PATHS[profile])
        if profile in _PROFILES_WITH_TRAINING_SOURCE:
            paths.update(_training_source_relative_paths(root))
        for relative_path in paths:
            if not (root / relative_path).is_file():
                raise Phase4RuntimeProfileError(
                    f"Exact-profile source {relative_path!r} is missing."
                )
        return sorted(paths)
    paths = {
        *PHASE4_ROOT_SOURCES,
        *PHASE4_SHARED_SOURCES,
        *_entrypoints(profile),
    }
    for directory_name in PHASE4_SOURCE_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            raise Phase4RuntimeProfileError(
                f"Phase 4 source directory {directory_name!r} is missing."
            )
        paths.update(
            str(path.relative_to(root))
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    for relative_path in paths:
        if not (root / relative_path).is_file():
            raise Phase4RuntimeProfileError(
                f"Phase 4 source {relative_path!r} is missing."
            )
    return sorted(paths)


def _git_profile_paths(root: Path, profile: str) -> list[str]:
    if profile in _EXACT_PROFILE_PATHS:
        requested_paths = set(_EXACT_PROFILE_PATHS[profile])
        if profile in _PROFILES_WITH_TRAINING_SOURCE:
            requested_paths.update(_training_source_relative_paths(root))
        output = _run_git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            *sorted(requested_paths),
        )
        paths = output.splitlines() if output else []
        if len(paths) != len(set(paths)):
            raise Phase4RuntimeProfileError(
                "Exact-profile Git source inventory contains duplicates."
            )
        return sorted(paths)
    pathspecs = [
        *PHASE4_SOURCE_DIRECTORIES,
        *PHASE4_ROOT_SOURCES,
        *PHASE4_SHARED_SOURCES,
        *_entrypoints(profile),
    ]
    output = _run_git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "HEAD",
        "--",
        *pathspecs,
    )
    paths = output.splitlines() if output else []
    selected = {
        relative_path
        for relative_path in paths
        if relative_path in PHASE4_ROOT_SOURCES
        or relative_path in PHASE4_SHARED_SOURCES
        or relative_path in _entrypoints(profile)
        or _is_directory_source(relative_path)
    }
    if len(paths) != len(set(paths)):
        raise Phase4RuntimeProfileError(
            "Phase 4 Git source inventory contains duplicates."
        )
    return sorted(selected)


def _assert_worktree_matches_head(root: Path, relative_paths: list[str]) -> None:
    for relative_path in relative_paths:
        index_entry = _run_git(root, "ls-files", "-v", "--", relative_path)
        if not index_entry or index_entry[0] != "H":
            raise Phase4RuntimeProfileError(
                f"Phase 4 source {relative_path!r} has unsafe index state."
            )
        if _run_git(root, "hash-object", "--", relative_path) != _run_git(
            root,
            "rev-parse",
            f"HEAD:{relative_path}",
        ):
            raise Phase4RuntimeProfileError(
                f"Phase 4 source {relative_path!r} differs from Git HEAD."
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_READ_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise Phase4RuntimeProfileError("Phase 4 source cannot be hashed.") from exc
    return digest.hexdigest()


def _manifest_sha256(profile: str, sources: dict[str, str]) -> str:
    value = {
        "source_manifest_schema_version": (PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION),
        "profile": profile,
        "sources": sources,
    }
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Phase4RuntimeProfileError(
                "Producer source manifest contains a duplicate JSON key."
            )
        result[key] = value
    return result


def _training_source_relative_paths(
    root: Path,
    *,
    schema_version: int = PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION,
) -> list[str]:
    """Training sources for one producer inventory version.

    Defaults to the current version so an executing runtime must carry every
    current source.  A historical producer checkout is enumerated by passing
    its authenticated manifest's version.
    """

    root_sources = producer_root_sources(schema_version)
    additional_sources = _producer_additional_sources(schema_version)
    directories = ("dataset", "evaluators", "models", "rl", "utils")
    paths = {*root_sources, *additional_sources}
    for directory_name in directories:
        directory = root / directory_name
        if not directory.is_dir():
            raise Phase4RuntimeProfileError(
                f"Producer source directory {directory_name!r} is missing."
            )
        paths.update(
            str(path.relative_to(root))
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    for config_relative in _producer_config_directories(schema_version):
        config_directory = root / config_relative
        if not config_directory.is_dir():
            raise Phase4RuntimeProfileError(
                "Producer registered configuration directory is missing."
            )
        candidates = (
            config_directory.rglob("*")
            if config_relative == "configs/policy_improvement_v2"
            else config_directory.iterdir()
        )
        paths.update(
            str(path.relative_to(root))
            for path in candidates
            if path.is_file()
            and _is_registered_producer_config(
                str(path.relative_to(root)), config_relative
            )
            and str(path.relative_to(root)) != PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH
        )
    for relative_path in (*root_sources, *additional_sources):
        if not (root / relative_path).is_file():
            raise Phase4RuntimeProfileError(
                f"Producer source {relative_path!r} is missing."
            )
    return sorted(paths)


def authorize_phase4_training_source(
    project_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> AuthorizedTrainingSource:
    """Authorize the producer manifest against one clean external checkout."""

    if not _LOWER_COMMIT.fullmatch(expected_git_commit):
        raise Phase4RuntimeProfileError(
            "Expected producer commit must be 40 lowercase hex characters."
        )
    requested_root = Path(project_root)
    if not requested_root.is_absolute():
        raise Phase4RuntimeProfileError(
            "Producer source root must be an absolute path."
        )
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise Phase4RuntimeProfileError("Producer source root does not exist.") from exc
    if (
        not root.is_dir()
        or Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve() != root
    ):
        raise Phase4RuntimeProfileError(
            "Producer source root must be the Git repository top level."
        )
    commit_before = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    status_before = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if commit_before != expected_git_commit or status_before:
        raise Phase4RuntimeProfileError(
            "Producer checkout does not match the authorized clean commit."
        )

    manifest_path = root / PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH
    try:
        manifest_bytes_before = manifest_path.read_bytes()
        manifest = json.loads(
            manifest_bytes_before.decode("ascii"),
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise Phase4RuntimeProfileError(
            "Producer source manifest is not strict ASCII JSON."
        ) from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "source_manifest_schema_version",
        "sources",
    }:
        raise Phase4RuntimeProfileError(
            "Producer source manifest has an invalid inventory."
        )
    schema_version = manifest["source_manifest_schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise Phase4RuntimeProfileError(
            "Producer source manifest has an invalid schema version."
        )
    try:
        producer_root_sources(schema_version)
    except Phase4RuntimeProfileError as exc:
        raise Phase4RuntimeProfileError(
            "Producer source manifest has an unsupported schema version."
        ) from exc
    derived_schema_version = _producer_inventory_schema_version(root)
    if schema_version != derived_schema_version:
        raise Phase4RuntimeProfileError(
            f"Producer source manifest declares schema {schema_version!r} but "
            f"its checkout satisfies schema {derived_schema_version!r}."
        )
    if not isinstance(manifest["sources"], dict):
        raise Phase4RuntimeProfileError(
            "Producer source manifest has an invalid inventory."
        )
    expected_paths = _training_source_relative_paths(
        root,
        schema_version=schema_version,
    )
    sources = manifest["sources"]
    if set(sources) != set(expected_paths):
        raise Phase4RuntimeProfileError(
            "Producer source manifest inventory differs from the checkout."
        )
    if any(
        not isinstance(relative_path, str)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for relative_path, digest in sources.items()
    ):
        raise Phase4RuntimeProfileError(
            "Producer source manifest contains an invalid entry."
        )
    tracked_paths = [
        *expected_paths,
        PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH,
        *(
            f"configs/phase4_2x2_norm_ablation/{condition}.yaml"
            for condition in ("nc_nv", "nc_yv", "yc_nv", "yc_yv")
        ),
    ]
    _assert_worktree_matches_head(root, sorted(tracked_paths))
    actual_sources_before = {
        relative_path: _sha256_file(root / relative_path)
        for relative_path in expected_paths
    }
    if actual_sources_before != sources:
        raise Phase4RuntimeProfileError(
            "Producer source manifest differs from the checkout bytes."
        )
    _assert_worktree_matches_head(root, sorted(tracked_paths))
    manifest_bytes_after = manifest_path.read_bytes()
    actual_sources_after = {
        relative_path: _sha256_file(root / relative_path)
        for relative_path in expected_paths
    }
    commit_after = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    status_after = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if (
        manifest_bytes_after != manifest_bytes_before
        or actual_sources_after != actual_sources_before
        or commit_after != commit_before
        or status_after
    ):
        raise Phase4RuntimeProfileError("Producer source changed during authorization.")
    return AuthorizedTrainingSource(
        git_commit=expected_git_commit,
        source_manifest_sha256=hashlib.sha256(manifest_bytes_before).hexdigest(),
        manifest_bytes=manifest_bytes_before,
    )


def authorize_phase4_source_profile(
    project_root: str | os.PathLike[str],
    expected_git_commit: str,
    profile: str,
) -> AuthorizedPhase4Profile:
    """Authorize one exact clean checkout before opening the runtime PAR."""

    if not _LOWER_COMMIT.fullmatch(expected_git_commit):
        raise Phase4RuntimeProfileError(
            "Expected Phase 4 source commit must be 40 lowercase hex characters."
        )
    requested_root = Path(project_root)
    if not requested_root.is_absolute():
        raise Phase4RuntimeProfileError("Phase 4 source root must be an absolute path.")
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise Phase4RuntimeProfileError("Phase 4 source root does not exist.") from exc
    if (
        not root.is_dir()
        or Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve() != root
    ):
        raise Phase4RuntimeProfileError(
            "Phase 4 source root must be the Git repository top level."
        )

    commit_before = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    status_before = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if commit_before != expected_git_commit or status_before:
        raise Phase4RuntimeProfileError(
            "Phase 4 source checkout does not match the authorized clean commit."
        )
    filesystem_paths = phase4_profile_relative_paths(root, profile)
    git_paths = _git_profile_paths(root, profile)
    if filesystem_paths != git_paths:
        raise Phase4RuntimeProfileError(
            "Phase 4 source inventory differs from Git HEAD."
        )
    _assert_worktree_matches_head(root, filesystem_paths)
    sources_before = {
        relative_path: _sha256_file(root / relative_path)
        for relative_path in filesystem_paths
    }
    _assert_worktree_matches_head(root, filesystem_paths)
    sources_after = {
        relative_path: _sha256_file(root / relative_path)
        for relative_path in filesystem_paths
    }
    commit_after = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    status_after = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if sources_after != sources_before or commit_after != commit_before or status_after:
        raise Phase4RuntimeProfileError(
            "Phase 4 source checkout changed during authorization."
        )
    return AuthorizedPhase4Profile(
        git_commit=expected_git_commit,
        source_manifest_sha256=_manifest_sha256(profile, sources_before),
        profile=profile,
        sources=sources_before,
    )


def _is_selected_bytecode(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if not path.parts or path.suffix not in {".pyc", ".pyo"}:
        return False
    if path.parts[0] in PHASE4_SOURCE_DIRECTORIES:
        return True
    selected_stems = {
        PurePosixPath(relative_path).stem
        for relative_path in (
            *PHASE4_ROOT_SOURCES,
            *PHASE4_SHARED_SOURCES,
            *(
                entrypoint
                for entrypoints in PHASE4_PROFILE_ENTRYPOINTS.values()
                for entrypoint in entrypoints
            ),
            *(
                relative_path
                for paths in _EXACT_PROFILE_PATHS.values()
                for relative_path in paths
            ),
        )
    }
    return any(
        path.name == f"{stem}.pyc"
        or path.name == f"{stem}.pyo"
        or path.name.startswith(f"{stem}.")
        for stem in selected_stems
    )


def _is_dataset_builder_selected(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if relative_path in POLICY_DATASET_BUILDER_PROFILE_PATHS:
        return True
    if (
        len(path.parts) >= 2
        and path.parts[0] in {"dataset", "utils"}
        and path.suffix == ".py"
    ):
        return True
    if (
        len(path.parts) == 3
        and path.parts[:2] == ("configs", "policy_improvement_v1")
        and path.suffix in {".json", ".yaml"}
    ):
        return True
    return path.name in {
        "phase4_runtime_profile.py",
        "policy_dataset_builder_entrypoint.py",
        "runtime_archive_preflight.py",
    }


def _is_policy_consumer_selected(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if (
        len(path.parts) == 2
        and path.parts[0] == "scripts"
        and path.name.startswith("policy_improvement_")
        and path.suffix == ".py"
    ):
        return True
    if any(
        _is_registered_producer_config(relative_path, directory)
        for directory in (
            "configs/policy_improvement_v1",
            "configs/policy_improvement_v2",
        )
    ):
        return True
    return relative_path in {
        "confirmatory_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_checkpoint_validator.py",
        "policy_improvement_consumer_entrypoint.py",
        "policy_improvement_full_backend.py",
        "policy_improvement_checkpoint_allowlist.py",
        "policy_improvement_full_entrypoint.py",
        "policy_improvement_non_smoke_checkpoint.py",
        "policy_improvement_sealed_evidence.py",
        "policy_improvement_smoke_checkpoint.py",
        "policy_improvement_smoke_runtime.py",
        "policy_improvement_theory_bridge_entrypoint.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "upi_trm_train.py",
    }


def _is_training_source_selected(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if (
        len(path.parts) >= 2
        and path.parts[0] in {"dataset", "evaluators", "models", "rl", "utils"}
        and path.suffix == ".py"
    ):
        return True
    if any(
        _is_registered_producer_config(relative_path, directory)
        for directory in _producer_config_directories(
            PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION
        )
    ):
        return relative_path != PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH
    return relative_path in {
        "confirmatory_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_checkpoint_allowlist.py",
        "policy_improvement_checkpoint_validator.py",
        "policy_improvement_smoke_checkpoint.py",
        "policy_improvement_smoke_runtime.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "scripts/policy_improvement_registry.py",
        "scripts/policy_improvement_schema.py",
        "scripts/policy_improvement_populations.py",
        "scripts/policy_improvement_v2_registry.py",
        "scripts/policy_improvement_v2_schema.py",
        "upi_trm_train.py",
    }


def _hash_archive_member(archive: ZipFile, relative_path: str) -> str:
    digest = hashlib.sha256()
    try:
        with archive.open(relative_path, "r") as handle:
            for block in iter(lambda: handle.read(_READ_SIZE), b""):
                digest.update(block)
    except (BadZipFile, KeyError, OSError, RuntimeError) as exc:
        raise Phase4RuntimeProfileError(
            "Phase 4 runtime source cannot be hashed."
        ) from exc
    return digest.hexdigest()


def assert_phase4_archive_matches_profile(
    archive: ZipFile,
    authorized: AuthorizedPhase4Profile,
) -> None:
    """Require the executing PAR to contain the authorized profile bytes."""

    infos = archive.infolist()
    names = [info.filename for info in infos if not info.is_dir()]
    if any(_is_selected_bytecode(name) for name in names):
        raise Phase4RuntimeProfileError(
            "Phase 4 runtime contains selected-source bytecode."
        )
    expected = set(authorized.sources)
    if authorized.profile == POLICY_DATASET_BUILDER_SOURCE_PROFILE:
        selected = {name for name in names if _is_dataset_builder_selected(name)}
    elif authorized.profile in _PROFILES_WITH_TRAINING_SOURCE:
        selected = {
            name
            for name in names
            if name in expected
            or _is_policy_consumer_selected(name)
            or _is_training_source_selected(name)
        }
    else:
        selected = {
            name for name in names if name in expected or _is_directory_source(name)
        }
    if selected != expected:
        raise Phase4RuntimeProfileError(
            "Phase 4 runtime source inventory differs from the authorized profile."
        )
    actual = {
        relative_path: _hash_archive_member(archive, relative_path)
        for relative_path in sorted(expected)
    }
    if actual != authorized.sources:
        raise Phase4RuntimeProfileError(
            "Phase 4 runtime source differs from the authorized checkout."
        )
