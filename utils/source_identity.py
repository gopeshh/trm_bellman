"""Deterministic source manifest for confirmatory producer verification."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile


SOURCE_MANIFEST_SCHEMA_VERSION = 1
SOURCE_MANIFEST_RELATIVE_PATH = (
    "configs/iclr_confirmatory/producer_source_manifest.json"
)
_ROOT_SOURCES = ("upi_trm_train.py", "puzzle_dataset.py")
_SOURCE_DIRECTORIES = ("dataset", "evaluators", "models", "rl", "utils")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceIdentityError(RuntimeError):
    """Raised when producer source bytes cannot be bound to the runtime."""


def behavior_source_relative_paths(root: str | Path) -> list[str]:
    source_root = Path(root).expanduser().resolve()
    paths: set[str] = set()
    for relative_path in _ROOT_SOURCES:
        if not (source_root / relative_path).is_file():
            raise SourceIdentityError(
                f"Producer repository is missing source {relative_path!r}."
            )
        paths.add(relative_path)
    for relative_directory in _SOURCE_DIRECTORIES:
        directory = source_root / relative_directory
        if not directory.is_dir():
            raise SourceIdentityError(
                f"Producer repository is missing directory {relative_directory!r}."
            )
        paths.update(
            str(path.relative_to(source_root))
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    config_directory = source_root / "configs" / "iclr_confirmatory"
    if not config_directory.is_dir():
        raise SourceIdentityError(
            "Producer repository is missing confirmatory configuration sources."
        )
    paths.update(
        str(path.relative_to(source_root))
        for path in config_directory.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".yaml"}
        and str(path.relative_to(source_root)) != SOURCE_MANIFEST_RELATIVE_PATH
    )
    return sorted(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceIdentityError("Producer source cannot be hashed.") from exc
    return digest.hexdigest()


def build_producer_source_manifest(root: str | Path) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    return {
        "source_manifest_schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "sources": {
            relative_path: _sha256(source_root / relative_path)
            for relative_path in behavior_source_relative_paths(source_root)
        },
    }


def validate_producer_source_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "source_manifest_schema_version",
        "sources",
    }:
        raise SourceIdentityError("Producer source manifest has an invalid inventory.")
    schema_version = value["source_manifest_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise SourceIdentityError("Unsupported producer source manifest schema.")
    raw_sources = value["sources"]
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise SourceIdentityError("Producer source manifest has no sources.")
    sources: dict[str, str] = {}
    for relative_path, digest in raw_sources.items():
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise SourceIdentityError("Producer source manifest entry is invalid.")
        sources[relative_path] = digest
    return {
        "source_manifest_schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "sources": {name: sources[name] for name in sorted(sources)},
    }


def _is_behavior_python_archive_member(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "__pycache__" in path.parts
        or path.suffix != ".py"
    ):
        return False
    if len(path.parts) == 1:
        return relative_path in _ROOT_SOURCES
    return path.parts[0] in _SOURCE_DIRECTORIES


def assert_runtime_archive_sources_match_manifest(
    archive: ZipFile,
    manifest: object,
) -> None:
    """Bind behavior-source members in a standalone PAR to its manifest."""

    validated = validate_producer_source_manifest(manifest)
    expected = {
        relative_path: digest
        for relative_path, digest in validated["sources"].items()
        if relative_path.endswith(".py")
    }
    runtime_names = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and _is_behavior_python_archive_member(info.filename)
    ]
    if len(runtime_names) != len(set(runtime_names)):
        raise SourceIdentityError(
            "Runtime archive contains duplicate behavior-source members."
        )
    if set(runtime_names) != set(expected):
        raise SourceIdentityError(
            "Runtime archive behavior-source inventory differs from the "
            "embedded manifest."
        )

    for relative_path, expected_digest in expected.items():
        digest = hashlib.sha256()
        try:
            with archive.open(relative_path, "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except (BadZipFile, OSError, RuntimeError) as exc:
            raise SourceIdentityError(
                "Runtime archive source cannot be hashed."
            ) from exc
        if digest.hexdigest() != expected_digest:
            raise SourceIdentityError(
                "Runtime archive source bytes differ from the embedded manifest."
            )
