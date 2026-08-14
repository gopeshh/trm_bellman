#!/usr/bin/env python3
"""Source-checkout and runtime-byte binding for Buck-owned Phase 4 tools."""

import hashlib
import json
import re
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Tuple
from zipfile import BadZipFile, ZipFile, is_zipfile

from phase4_runtime_profile import (
    PHASE4_AUDIT_SOURCE_PROFILE,
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    PHASE4_FIGURE_SOURCE_PROFILE,
    PHASE4_PROFILE_ENTRYPOINTS,
    PHASE4_ROOT_SOURCES,
    PHASE4_SHARED_SOURCES,
    PHASE4_SOURCE_DIRECTORIES,
    PHASE4_SOURCE_MANIFEST_SCHEMA_VERSION,
    phase4_profile_relative_paths,
)
from utils.run_identity import (
    RunIdentityError,
    assert_git_files_match_head,
    discover_clean_git_source,
    git_files_at_head,
)
from utils.source_identity import (
    SOURCE_MANIFEST_RELATIVE_PATH,
    SourceIdentityError,
    behavior_source_relative_paths,
    build_producer_source_manifest,
    validate_producer_source_manifest,
)


_PHASE4_CONDITION_CONFIGS = tuple(
    f"configs/phase4_2x2_norm_ablation/{condition}.yaml"
    for condition in ("nc_nv", "nc_yv", "yc_nv", "yc_yv")
)

class Phase4SourceError(RuntimeError):
    """Raised when a Phase 4 tool cannot bind its source checkout."""


def require_phase4_runtime_attestation(
    value: Mapping[str, Any] | None,
    profile: str,
) -> Dict[str, str]:
    """Validate the sealed-launcher identity passed before behavior imports."""

    if value is None or set(value) != {
        "runtime_sha256",
        "role",
        "source_git_commit",
        "source_manifest_sha256",
    }:
        raise Phase4SourceError(
            "Phase 4 publication requires the authenticated runtime launcher."
        )
    if value["role"] != profile:
        raise Phase4SourceError(
            "Phase 4 runtime role differs from the selected publication tool."
        )
    runtime_sha256 = value["runtime_sha256"]
    source_git_commit = value["source_git_commit"]
    source_manifest_sha256 = value["source_manifest_sha256"]
    if (
        not isinstance(runtime_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None
        or not isinstance(source_git_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_git_commit) is None
        or not isinstance(source_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None
    ):
        raise Phase4SourceError("Phase 4 runtime attestation is malformed.")
    return {
        "runtime_sha256": runtime_sha256,
        "role": profile,
        "source_git_commit": source_git_commit,
        "source_manifest_sha256": source_manifest_sha256,
    }


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


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        value[key] = item
    return value


def _require_lowercase_git_commit(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Phase4SourceError(
            "Expected Phase 4 producer commit must be 40 lowercase hex "
            "characters."
        )
    return value


def verify_phase4_producer_source(
    project_root: str | Path,
    expected_git_commit: str,
) -> Dict[str, Any]:
    """Bind a checkpoint producer claim to one explicit clean Git checkout."""

    expected_commit = _require_lowercase_git_commit(expected_git_commit)
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Phase4SourceError(
            "Phase 4 producer project root does not exist."
        ) from exc
    if not root.is_dir():
        raise Phase4SourceError(
            "Phase 4 producer project root must be a directory."
        )

    expected_git_identity = {
        "git_commit": expected_commit,
        "git_clean": True,
    }
    manifest_path = root / SOURCE_MANIFEST_RELATIVE_PATH
    try:
        git_identity_before = discover_clean_git_source(root)
        if git_identity_before != expected_git_identity:
            raise Phase4SourceError(
                "Phase 4 producer checkout does not match the authorized commit."
            )

        behavior_sources_before = behavior_source_relative_paths(root)
        tracked_paths = sorted(
            {
                *behavior_sources_before,
                SOURCE_MANIFEST_RELATIVE_PATH,
                *_PHASE4_CONDITION_CONFIGS,
            }
        )
        assert_git_files_match_head(root, tracked_paths)

        manifest_bytes_before = manifest_path.read_bytes()
        manifest_value = json.loads(
            manifest_bytes_before.decode("ascii"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
        recorded_manifest = validate_producer_source_manifest(manifest_value)
        generated_manifest_before = build_producer_source_manifest(root)
        if recorded_manifest != generated_manifest_before:
            raise Phase4SourceError(
                "Phase 4 producer manifest differs from the producer source tree."
            )

        behavior_sources_after = behavior_source_relative_paths(root)
        if behavior_sources_after != behavior_sources_before:
            raise Phase4SourceError(
                "Phase 4 producer source inventory changed during verification."
            )
        assert_git_files_match_head(root, tracked_paths)
        manifest_bytes_after = manifest_path.read_bytes()
        generated_manifest_after = build_producer_source_manifest(root)
        if (
            manifest_bytes_after != manifest_bytes_before
            or generated_manifest_after != generated_manifest_before
        ):
            raise Phase4SourceError(
                "Phase 4 producer source changed during verification."
            )

        git_identity_after = discover_clean_git_source(root)
        if (
            git_identity_after != expected_git_identity
            or git_identity_after != git_identity_before
        ):
            raise Phase4SourceError(
                "Phase 4 producer Git identity changed during verification."
            )
    except Phase4SourceError:
        raise
    except (
        json.JSONDecodeError,
        OSError,
        RunIdentityError,
        SourceIdentityError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise Phase4SourceError(
            "Phase 4 producer source identity cannot be verified."
        ) from exc

    return {
        "git_commit": expected_commit,
        "git_clean": True,
        "source_manifest_sha256": _sha256_bytes(manifest_bytes_before),
    }


def resolve_phase4_path(
    path_value: str | Path,
    relative_to: str | Path,
) -> Path:
    """Resolve a Phase 4 CLI path against its declared ownership root."""

    requested = Path(path_value).expanduser()
    if requested.is_absolute():
        return requested.resolve()
    return (Path(relative_to).expanduser() / requested).resolve()


def phase4_source_relative_paths(
    project_root: Path,
    profile: str,
) -> list[str]:
    try:
        return phase4_profile_relative_paths(project_root, profile)
    except RuntimeError as exc:
        raise Phase4SourceError(str(exc)) from exc


def _assert_project_inventory_matches_head(
    project_root: Path,
    profile: str,
    current_paths: list[str],
) -> None:
    entrypoints = PHASE4_PROFILE_ENTRYPOINTS[profile]
    pathspecs = [
        *PHASE4_SOURCE_DIRECTORIES,
        *PHASE4_ROOT_SOURCES,
        *PHASE4_SHARED_SOURCES,
        *entrypoints,
    ]
    try:
        tracked_paths = git_files_at_head(project_root, pathspecs)
        head_paths = {
            relative_path
            for relative_path in tracked_paths
            if relative_path in PHASE4_SHARED_SOURCES
            or relative_path in PHASE4_ROOT_SOURCES
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
    """Return the source digest a schema-v4 evaluator must record."""

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
        and path.parts[0] in PHASE4_SOURCE_DIRECTORIES
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
    if path.parts[0] in PHASE4_SOURCE_DIRECTORIES:
        return True
    selected_script_stems = {
        PurePosixPath(relative_path).stem
        for relative_path in (
            *PHASE4_SHARED_SOURCES,
            "phase4_runtime_entrypoint.py",
            "phase4_runtime_profile.py",
            "runtime_archive_preflight.py",
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
    for directory_name in PHASE4_SOURCE_DIRECTORIES:
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
        if not _is_selected_python_source(relative_path)
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
