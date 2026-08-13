#!/usr/bin/env python3
"""Source-checkout and runtime-byte binding for Buck-owned Phase 4 tools."""

import hashlib
import json
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Tuple
from zipfile import BadZipFile, ZipFile, is_zipfile

from utils.run_identity import (
    RunIdentityError,
    assert_git_files_match_head,
    git_files_at_head,
)


PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION = 1
PHASE4_EVALUATOR_SOURCE_PROFILE = "evaluator"
PHASE4_AUDIT_SOURCE_PROFILE = "audit"
PHASE4_FIGURE_SOURCE_PROFILE = "figure"

_SOURCE_DIRECTORIES = ("models", "rl", "utils")
_SHARED_SOURCES = (
    "scripts/phase4_checkpoint.py",
    "scripts/phase4_diagnostic_inputs.py",
    "scripts/phase4_result_schema.py",
    "scripts/phase4_source.py",
)
_PROFILE_ENTRYPOINTS = {
    PHASE4_EVALUATOR_SOURCE_PROFILE: (
        "scripts/eval_phase4_2x2_norm_ablation.py",
    ),
    PHASE4_AUDIT_SOURCE_PROFILE: (
        "scripts/audit_phase4_paper_ready.py",
    ),
    PHASE4_FIGURE_SOURCE_PROFILE: (
        "scripts/make_paper_figures_phase4.py",
    ),
}


class Phase4SourceError(RuntimeError):
    """Raised when a Phase 4 tool cannot bind its source checkout."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_manifest_sha256(manifest: Dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def phase4_source_relative_paths(
    project_root: Path,
    profile: str,
) -> list[str]:
    try:
        entrypoints = _PROFILE_ENTRYPOINTS[profile]
    except KeyError as exc:
        raise Phase4SourceError(
            f"Unsupported Phase 4 source profile {profile!r}."
        ) from exc

    paths = set(_SHARED_SOURCES)
    paths.update(entrypoints)
    for directory_name in _SOURCE_DIRECTORIES:
        directory = project_root / directory_name
        if not directory.is_dir():
            raise Phase4SourceError(
                f"Phase 4 project source directory {directory_name!r} is missing."
            )
        paths.update(
            str(path.relative_to(project_root))
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    for relative_path in paths:
        if not (project_root / relative_path).is_file():
            raise Phase4SourceError(
                f"Phase 4 project source {relative_path!r} is missing."
            )
    return sorted(paths)


def _assert_project_inventory_matches_head(
    project_root: Path,
    profile: str,
    current_paths: list[str],
) -> None:
    entrypoints = _PROFILE_ENTRYPOINTS[profile]
    pathspecs = [
        *_SOURCE_DIRECTORIES,
        *_SHARED_SOURCES,
        *entrypoints,
    ]
    try:
        tracked_paths = git_files_at_head(project_root, pathspecs)
        head_paths = {
            relative_path
            for relative_path in tracked_paths
            if relative_path in _SHARED_SOURCES
            or relative_path in entrypoints
            or _is_selected_python_source(relative_path)
        }
        if head_paths != set(current_paths):
            raise Phase4SourceError(
                "Phase 4 project source inventory differs from Git HEAD."
            )
        assert_git_files_match_head(project_root, current_paths)
    except RunIdentityError as exc:
        raise Phase4SourceError(
            "Phase 4 project sources do not match Git HEAD."
        ) from exc


def build_phase4_source_manifest(
    project_root: str | Path,
    profile: str,
) -> Dict[str, Any]:
    """Hash the exact project sources required by one Phase 4 executable."""

    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Phase4SourceError("Phase 4 project root does not exist.") from exc
    if not root.is_dir():
        raise Phase4SourceError("Phase 4 project root must be a directory.")

    sources: Dict[str, str] = {}
    for relative_path in phase4_source_relative_paths(root, profile):
        try:
            sources[relative_path] = _sha256_bytes(
                (root / relative_path).read_bytes()
            )
        except OSError as exc:
            raise Phase4SourceError(
                f"Phase 4 project source {relative_path!r} cannot be hashed."
            ) from exc
    return {
        "source_manifest_schema_version": (
            PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION
        ),
        "profile": profile,
        "sources": sources,
    }


def _verified_phase4_source_manifest(
    project_root: str | Path,
    profile: str,
) -> Dict[str, Any]:
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Phase4SourceError("Phase 4 project root does not exist.") from exc
    paths_before = phase4_source_relative_paths(root, profile)
    _assert_project_inventory_matches_head(root, profile, paths_before)
    manifest = build_phase4_source_manifest(root, profile)
    paths_after = phase4_source_relative_paths(root, profile)
    _assert_project_inventory_matches_head(root, profile, paths_after)
    if paths_after != paths_before:
        raise Phase4SourceError(
            "Phase 4 project source inventory changed while hashing."
        )
    return manifest


def phase4_source_manifest_sha256(
    project_root: str | Path,
    profile: str,
) -> str:
    """Return the canonical digest for one project-source profile."""

    return _canonical_manifest_sha256(
        _verified_phase4_source_manifest(project_root, profile)
    )


def phase4_evaluator_source_manifest_sha256(
    project_root: str | Path,
) -> str:
    """Return the source digest a schema-v3 evaluator must record."""

    return phase4_source_manifest_sha256(
        project_root,
        PHASE4_EVALUATOR_SOURCE_PROFILE,
    )


def _is_selected_python_source(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix == ".py"
        and bool(path.parts)
        and path.parts[0] in _SOURCE_DIRECTORIES
    )


def _is_selected_bytecode(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.suffix not in {".pyc", ".pyo"}
    ):
        return False
    if path.parts[0] in _SOURCE_DIRECTORIES:
        return True
    selected_script_stems = {
        PurePosixPath(relative_path).stem
        for relative_path in (
            *_SHARED_SOURCES,
            *(
                entrypoint
                for entrypoints in _PROFILE_ENTRYPOINTS.values()
                for entrypoint in entrypoints
            ),
        )
    }
    return path.parts[0] == "scripts" and any(
        path.name == f"{stem}.pyc"
        or path.name == f"{stem}.pyo"
        or path.name.startswith(f"{stem}.")
        for stem in selected_script_stems
    )


def _runtime_archive_from_module_path() -> Path | None:
    candidates = [Path(sys.argv[0]).expanduser().absolute()]
    module_path = Path(__file__).expanduser().absolute()
    candidates.extend(module_path.parents)
    for candidate in candidates:
        try:
            if candidate.is_file() and is_zipfile(candidate):
                return candidate
        except OSError:
            continue
    return None


def _runtime_filesystem_root() -> Path:
    module_path = Path(__file__).expanduser().absolute()
    if module_path.suffix != ".py" or module_path.name != "phase4_source.py":
        raise Phase4SourceError(
            "Phase 4 runtime must expose source, not bytecode, for phase4_source."
        )
    root = module_path.parent.parent
    if not root.is_dir():
        raise Phase4SourceError("Phase 4 runtime source root is unavailable.")
    return root


def _archive_runtime_manifest(
    archive_path: Path,
    expected_paths: Iterable[str],
    profile: str,
) -> Dict[str, Any]:
    expected = set(expected_paths)
    try:
        with ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if any(info.orig_filename != info.filename for info in infos):
                raise Phase4SourceError(
                    "Phase 4 runtime archive raw and effective names differ."
                )
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                raise Phase4SourceError(
                    "Phase 4 runtime archive contains duplicate members."
                )
            if any(_is_selected_bytecode(name) for name in names):
                raise Phase4SourceError(
                    "Phase 4 runtime archive contains selected-source bytecode."
                )
            selected_names = {
                name for name in names if _is_selected_python_source(name)
            }
            selected_names.update(expected.intersection(names))
            if selected_names != expected:
                raise Phase4SourceError(
                    "Phase 4 runtime source inventory differs from the project."
                )
            sources = {
                relative_path: _sha256_bytes(archive.read(relative_path))
                for relative_path in sorted(expected)
            }
    except Phase4SourceError:
        raise
    except (BadZipFile, KeyError, OSError, RuntimeError) as exc:
        raise Phase4SourceError(
            "Phase 4 runtime archive sources cannot be hashed."
        ) from exc
    return {
        "source_manifest_schema_version": (
            PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION
        ),
        "profile": profile,
        "sources": sources,
    }


def _filesystem_runtime_manifest(
    runtime_root: Path,
    expected_paths: Iterable[str],
    profile: str,
) -> Dict[str, Any]:
    expected = set(expected_paths)
    selected: set[str] = set()
    for directory_name in _SOURCE_DIRECTORIES:
        directory = runtime_root / directory_name
        if not directory.is_dir():
            raise Phase4SourceError(
                f"Phase 4 runtime directory {directory_name!r} is missing."
            )
        selected.update(
            str(path.relative_to(runtime_root))
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    selected.update(
        relative_path
        for relative_path in expected
        if relative_path.startswith("scripts/")
        and (runtime_root / relative_path).is_file()
    )
    if selected != expected:
        raise Phase4SourceError(
            "Phase 4 runtime source inventory differs from the project."
        )

    sources: Dict[str, str] = {}
    for relative_path in sorted(expected):
        runtime_path = runtime_root / relative_path
        if not runtime_path.is_file() or runtime_path.suffix != ".py":
            raise Phase4SourceError(
                f"Phase 4 runtime source {relative_path!r} is unavailable."
            )
        try:
            sources[relative_path] = _sha256_bytes(runtime_path.read_bytes())
        except OSError as exc:
            raise Phase4SourceError(
                f"Phase 4 runtime source {relative_path!r} cannot be hashed."
            ) from exc
    return {
        "source_manifest_schema_version": (
            PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION
        ),
        "profile": profile,
        "sources": sources,
    }


def verify_phase4_runtime_sources(
    project_root: str | Path,
    profile: str,
    *,
    runtime_location: str | Path | None = None,
) -> str:
    """Require the executing source bytes to match the named clean checkout.

    ``runtime_location`` exists for deterministic unit tests. Production callers
    do not expose it through their command lines and always inspect this module's
    actual runtime container.
    """

    project_manifest = _verified_phase4_source_manifest(project_root, profile)
    expected_paths = project_manifest["sources"]
    if runtime_location is None:
        archive_path = _runtime_archive_from_module_path()
        if archive_path is not None:
            runtime_manifest = _archive_runtime_manifest(
                archive_path,
                expected_paths,
                profile,
            )
        else:
            runtime_manifest = _filesystem_runtime_manifest(
                _runtime_filesystem_root(),
                expected_paths,
                profile,
            )
    else:
        location = Path(runtime_location).expanduser().absolute()
        try:
            archive = location.is_file() and is_zipfile(location)
        except OSError as exc:
            raise Phase4SourceError(
                "Phase 4 test runtime location is unavailable."
            ) from exc
        if archive:
            runtime_manifest = _archive_runtime_manifest(
                location,
                expected_paths,
                profile,
            )
        elif location.is_dir():
            runtime_manifest = _filesystem_runtime_manifest(
                location,
                expected_paths,
                profile,
            )
        else:
            raise Phase4SourceError(
                "Phase 4 runtime location is neither a directory nor a ZIP."
            )
    if runtime_manifest != project_manifest:
        raise Phase4SourceError(
            "Phase 4 runtime source bytes differ from the claimed checkout."
        )
    final_project_manifest = _verified_phase4_source_manifest(
        project_root,
        profile,
    )
    if final_project_manifest != project_manifest:
        raise Phase4SourceError(
            "Phase 4 project source bytes changed during runtime verification."
        )
    return _canonical_manifest_sha256(project_manifest)


def resolve_phase4_source_roots(
    project_root: str | Path,
    fbcode_root: str | Path,
) -> Tuple[Path, Path]:
    """Require an explicit implementation checkout and matching Buck cell link."""

    try:
        project = Path(project_root).expanduser().resolve(strict=True)
        fbcode = Path(fbcode_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Phase4SourceError(
            "Phase 4 source and fbcode roots must exist."
        ) from exc
    if not project.is_dir() or not fbcode.is_dir():
        raise Phase4SourceError(
            "Phase 4 source and fbcode roots must be directories."
        )
    required_project_paths = (
        project / ".git",
        project / "upi_trm_train.py",
        project / "configs" / "phase4_2x2_norm_ablation",
    )
    if not all(path.exists() for path in required_project_paths):
        raise Phase4SourceError(
            "Phase 4 project root is not the implementation checkout."
        )
    cell_link = fbcode / "buiksat_trm"
    try:
        linked_project = cell_link.resolve(strict=True)
    except OSError as exc:
        raise Phase4SourceError(
            "fbcode//buiksat_trm is not materialized."
        ) from exc
    if linked_project != project:
        raise Phase4SourceError(
            "fbcode//buiksat_trm does not resolve to the project root."
        )
    return project, fbcode
