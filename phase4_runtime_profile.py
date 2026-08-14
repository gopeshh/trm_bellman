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
PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION = 1
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
    PHASE4_EVALUATOR_SOURCE_PROFILE: (
        "scripts/eval_phase4_2x2_norm_ablation.py",
    ),
    PHASE4_AUDIT_SOURCE_PROFILE: (
        "scripts/audit_phase4_paper_ready.py",
    ),
    PHASE4_FIGURE_SOURCE_PROFILE: (
        "scripts/phase4_figure_publication.py",
        "scripts/make_paper_figures_phase4.py",
    ),
}
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_READ_SIZE = 1024 * 1024


class Phase4RuntimeProfileError(RuntimeError):
    """Raised when a Phase 4 source profile cannot be authenticated."""


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
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
    }
    try:
        completed = subprocess.run(
            [
                "git",
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
        raise Phase4RuntimeProfileError(
            "Phase 4 source cannot be hashed."
        ) from exc
    return digest.hexdigest()


def _manifest_sha256(profile: str, sources: dict[str, str]) -> str:
    value = {
        "source_manifest_schema_version": (
            PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION
        ),
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


def _training_source_relative_paths(root: Path) -> list[str]:
    root_sources = (
        "confirmatory_runtime_launcher.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "upi_trm_train.py",
    )
    directories = ("dataset", "evaluators", "models", "rl", "utils")
    paths = set(root_sources)
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
    config_directory = root / "configs" / "iclr_confirmatory"
    if not config_directory.is_dir():
        raise Phase4RuntimeProfileError(
            "Producer confirmatory configuration directory is missing."
        )
    paths.update(
        str(path.relative_to(root))
        for path in config_directory.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".yaml"}
        and str(path.relative_to(root))
        != PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH
    )
    for relative_path in root_sources:
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
        raise Phase4RuntimeProfileError(
            "Producer source root does not exist."
        ) from exc
    if not root.is_dir() or Path(
        _run_git(root, "rev-parse", "--show-toplevel")
    ).resolve() != root:
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
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"source_manifest_schema_version", "sources"}
        or manifest["source_manifest_schema_version"] != 1
        or not isinstance(manifest["sources"], dict)
    ):
        raise Phase4RuntimeProfileError(
            "Producer source manifest has an invalid inventory."
        )
    expected_paths = _training_source_relative_paths(root)
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
        raise Phase4RuntimeProfileError(
            "Producer source changed during authorization."
        )
    return AuthorizedTrainingSource(
        git_commit=expected_git_commit,
        source_manifest_sha256=hashlib.sha256(
            manifest_bytes_before
        ).hexdigest(),
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
        raise Phase4RuntimeProfileError(
            "Phase 4 source root must be an absolute path."
        )
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise Phase4RuntimeProfileError(
            "Phase 4 source root does not exist."
        ) from exc
    if not root.is_dir() or Path(
        _run_git(root, "rev-parse", "--show-toplevel")
    ).resolve() != root:
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
    if (
        sources_after != sources_before
        or commit_after != commit_before
        or status_after
    ):
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
        )
    }
    return any(
        path.name == f"{stem}.pyc"
        or path.name == f"{stem}.pyo"
        or path.name.startswith(f"{stem}.")
        for stem in selected_stems
    )


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
    selected = {
        name
        for name in names
        if name in expected or _is_directory_source(name)
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
