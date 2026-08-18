#!/usr/bin/env fbpython
"""Verify and execute the frozen runtime used for confirmatory runs.

This module intentionally imports only the Python standard library.  It must
finish authenticating the PAR before any training or model module is loaded.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable
from zipfile import BadZipFile, ZipFile


SOURCE_MANIFEST_RELATIVE_PATH = (
    "configs/iclr_confirmatory/producer_source_manifest.json"
)
# Accept every producer inventory version, because the launcher may verify a
# historical producer runtime whose manifest predates a later root source.
SOURCE_MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_SOURCE_MANIFEST_SCHEMA_VERSIONS = (1, 2)
VERIFIED_RUNTIME_PATH_ENV = "UPI_TRM_VERIFIED_RUNTIME_PATH"
VERIFIED_RUNTIME_SHA256_ENV = "UPI_TRM_VERIFIED_RUNTIME_SHA256"
VERIFIED_RUNTIME_FD_ENV = "UPI_TRM_VERIFIED_RUNTIME_FD"
PRIVATE_UNPACK_BASE_ENV = "UPI_TRM_PRIVATE_UNPACK_BASE"
PRIVATE_UNPACK_FD_ENV = "UPI_TRM_PRIVATE_UNPACK_FD"
PAR_FILENAME_ENV = "FB_PAR_FILENAME"

_ROOT_SOURCES_BY_VERSION: dict[int, tuple[str, ...]] = {
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
}
_ROOT_SOURCES = (
    "confirmatory_runtime_launcher.py",
    "phase4_runtime_profile.py",
    "policy_improvement_checkpoint_allowlist.py",
    "policy_improvement_smoke_checkpoint.py",
    "policy_improvement_smoke_runtime.py",
    "puzzle_dataset.py",
    "runtime_archive_preflight.py",
    "upi_trm_train.py",
)
_ADDITIONAL_SOURCES = (
    "scripts/policy_improvement_registry.py",
    "scripts/policy_improvement_schema.py",
)
_SOURCE_DIRECTORIES = ("dataset", "evaluators", "models", "rl", "utils")
_CONFIG_DIRECTORIES = (
    PurePosixPath("configs/iclr_confirmatory"),
    PurePosixPath("configs/policy_improvement_v1"),
)
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_SIZE = 1024 * 1024
_MEMFD_SEALING_AVAILABLE = all(
    hasattr(fcntl, name)
    for name in (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_WRITE",
    )
)
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 0)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 0)
_REQUIRED_MEMFD_SEALS = sum(
    getattr(fcntl, name, 0)
    for name in (
        "F_SEAL_WRITE",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
    )
)
_UNSAFE_CHILD_ENV_NAMES = {
    "BASHOPTS",
    "BASH_ENV",
    "CDPATH",
    "ENV",
    "GCONV_PATH",
    "LOCPATH",
    "MV_ARGS",
    "MV_CMD",
    "PYTHONHOME",
    "PYTHONPATH",
    "SHELLOPTS",
}


class ConfirmatoryRuntimeError(RuntimeError):
    """Raised when a runtime artifact cannot be authenticated or executed."""


@dataclass(frozen=True)
class VerifiedRuntime:
    """Identity passed from validation to the final exec boundary."""

    path: Path
    sha256: str
    descriptor: int


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfirmatoryRuntimeError(
                "Runtime source manifest contains a duplicate JSON key."
            )
        result[key] = value
    return result


def _is_safe_relative_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return (
        bool(relative_path)
        and bool(path.parts)
        and "\\" not in relative_path
        and not path.is_absolute()
        and relative_path == path.as_posix()
        and "." not in path.parts
        and ".." not in path.parts
    )


def _canonical_archive_member_path(member_name: str) -> str | None:
    """Return one canonical extraction path, rejecting aliases and roots."""

    is_directory = member_name.endswith("/")
    relative_path = member_name[:-1] if is_directory else member_name
    if not _is_safe_relative_path(relative_path):
        return None
    canonical_path = PurePosixPath(relative_path).as_posix()
    canonical_member = f"{canonical_path}/" if is_directory else canonical_path
    if member_name != canonical_member:
        return None
    return canonical_path


def _is_behavior_python_source(relative_path: str) -> bool:
    if not _is_safe_relative_path(relative_path):
        return False
    path = PurePosixPath(relative_path)
    if "__pycache__" in path.parts or path.suffix != ".py":
        return False
    if len(path.parts) == 1:
        return relative_path in _ROOT_SOURCES
    if relative_path in _ADDITIONAL_SOURCES:
        return True
    return path.parts[0] in _SOURCE_DIRECTORIES


def _is_behavior_bytecode(relative_path: str) -> bool:
    if not _is_safe_relative_path(relative_path):
        return False
    path = PurePosixPath(relative_path)
    if path.suffix not in {".pyc", ".pyo"}:
        return False
    if path.parts[0] in _SOURCE_DIRECTORIES:
        return True
    root_stems = tuple(PurePosixPath(source).stem for source in _ROOT_SOURCES)
    if len(path.parts) == 1:
        return path.stem in root_stems
    if path.parent == PurePosixPath("scripts"):
        return any(
            path.stem == PurePosixPath(source).stem
            for source in _ADDITIONAL_SOURCES
        )
    if path.parts[:2] == ("scripts", "__pycache__"):
        return any(
            path.name.startswith(f"{PurePosixPath(source).stem}.")
            for source in _ADDITIONAL_SOURCES
        )
    return path.parts[0] == "__pycache__" and any(
        path.name.startswith(f"{stem}.") for stem in root_stems
    )


def _is_confirmatory_config_source(relative_path: str) -> bool:
    if not _is_safe_relative_path(relative_path):
        return False
    path = PurePosixPath(relative_path)
    return (
        path.parent in _CONFIG_DIRECTORIES
        and path.suffix in {".json", ".yaml"}
        and relative_path != SOURCE_MANIFEST_RELATIVE_PATH
    )


def _validate_manifest(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "source_manifest_schema_version",
        "sources",
    }:
        raise ConfirmatoryRuntimeError(
            "Runtime source manifest has an invalid top-level inventory."
        )
    schema_version = value["source_manifest_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_SOURCE_MANIFEST_SCHEMA_VERSIONS
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime source manifest has an unsupported schema version."
        )
    raw_sources = value["sources"]
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ConfirmatoryRuntimeError("Runtime source manifest has no sources.")

    sources: dict[str, str] = {}
    for relative_path, digest in raw_sources.items():
        if (
            not isinstance(relative_path, str)
            or not isinstance(digest, str)
            or not _LOWER_SHA256.fullmatch(digest)
            or not (
                _is_behavior_python_source(relative_path)
                or _is_confirmatory_config_source(relative_path)
            )
        ):
            raise ConfirmatoryRuntimeError(
                "Runtime source manifest contains an invalid source entry."
            )
        sources[relative_path] = digest

    # The manifest's own version selects the root sources its producer had.
    required_roots = _ROOT_SOURCES_BY_VERSION.get(schema_version, _ROOT_SOURCES)
    missing_roots = set((*required_roots, *_ADDITIONAL_SOURCES)).difference(sources)
    missing_directories = {
        directory
        for directory in _SOURCE_DIRECTORIES
        if not any(
            PurePosixPath(relative_path).parts[0] == directory
            for relative_path in sources
            if relative_path.endswith(".py")
        )
    }
    if missing_roots or missing_directories:
        raise ConfirmatoryRuntimeError(
            "Runtime source manifest omits a required behavior-source root."
        )
    missing_config_directories = {
        directory
        for directory in _CONFIG_DIRECTORIES
        if not any(
            PurePosixPath(path).parent == directory
            and _is_confirmatory_config_source(path)
            for path in sources
        )
    }
    if missing_config_directories:
        raise ConfirmatoryRuntimeError(
            "Runtime source manifest omits registered configuration sources."
        )
    return {name: sources[name] for name in sorted(sources)}


def _load_embedded_manifest(archive: ZipFile) -> dict[str, str]:
    try:
        raw_manifest = archive.read(SOURCE_MANIFEST_RELATIVE_PATH)
    except (BadZipFile, KeyError, OSError, RuntimeError, zlib.error) as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime archive is missing a readable source manifest."
        ) from exc
    try:
        value = json.loads(
            raw_manifest.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (ConfirmatoryRuntimeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        if isinstance(exc, ConfirmatoryRuntimeError):
            raise
        raise ConfirmatoryRuntimeError(
            "Runtime source manifest is not strict UTF-8 JSON."
        ) from exc
    return _validate_manifest(value)


def _sha256_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    try:
        handle.seek(0)
        for block in iter(lambda: handle.read(_READ_SIZE), b""):
            digest.update(block)
        handle.seek(0)
    except OSError as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime artifact cannot be read completely."
        ) from exc
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _assert_runtime_unchanged(
    path: Path,
    path_before: os.stat_result,
    descriptor_before: os.stat_result,
    descriptor_after: os.stat_result,
) -> None:
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ConfirmatoryRuntimeError(
            "Runtime artifact changed during validation."
        ) from exc
    identities = {
        _stat_identity(path_before),
        _stat_identity(descriptor_before),
        _stat_identity(descriptor_after),
        _stat_identity(path_after),
    }
    if len(identities) != 1 or not stat.S_ISREG(path_after.st_mode):
        raise ConfirmatoryRuntimeError(
            "Runtime artifact identity, size, or timestamps changed during validation."
        )


def _sealed_runtime_copy(source_descriptor: int, expected_sha256: str) -> int:
    """Copy verified bytes into an anonymous file that cannot be mutated."""

    if not _MEMFD_SEALING_AVAILABLE or not hasattr(os, "memfd_create"):
        raise ConfirmatoryRuntimeError(
            "This host cannot create a sealable confirmatory runtime."
        )
    try:
        sealed_descriptor = os.memfd_create(
            "upi_trm_confirmatory_runtime",
            os.MFD_ALLOW_SEALING,
        )
    except (AttributeError, OSError) as exc:
        raise ConfirmatoryRuntimeError(
            "This host cannot create a sealable confirmatory runtime."
        ) from exc
    digest = hashlib.sha256()
    try:
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(source_descriptor, _READ_SIZE)
            if not block:
                break
            digest.update(block)
            remaining = memoryview(block)
            while remaining:
                written = os.write(sealed_descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while sealing runtime")
                remaining = remaining[written:]
        if digest.hexdigest() != expected_sha256:
            raise ConfirmatoryRuntimeError(
                "Runtime artifact changed while creating the sealed copy."
            )
        os.fchmod(sealed_descriptor, 0o500)
        fcntl.fcntl(
            sealed_descriptor,
            _F_ADD_SEALS,
            _REQUIRED_MEMFD_SEALS,
        )
        actual_seals = fcntl.fcntl(sealed_descriptor, _F_GET_SEALS)
        if actual_seals & _REQUIRED_MEMFD_SEALS != _REQUIRED_MEMFD_SEALS:
            raise ConfirmatoryRuntimeError(
                "Confirmatory runtime memfd does not have all required seals."
            )
        os.lseek(sealed_descriptor, 0, os.SEEK_SET)
        os.set_inheritable(sealed_descriptor, True)
    except ConfirmatoryRuntimeError:
        os.close(sealed_descriptor)
        raise
    except OSError as exc:
        os.close(sealed_descriptor)
        raise ConfirmatoryRuntimeError(
            "Verified runtime cannot be copied into a sealed memfd."
        ) from exc
    return sealed_descriptor


def _validate_archive_layout(archive: ZipFile) -> None:
    """Reject unsafe or ambiguous members before profile-specific checks."""

    infos = archive.infolist()
    if any(info.orig_filename != info.filename for info in infos):
        raise ConfirmatoryRuntimeError(
            "Runtime archive raw and effective member names differ."
        )
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ConfirmatoryRuntimeError(
            "Runtime archive contains duplicate member names."
        )
    canonical_members: dict[str, bool] = {}
    for info in infos:
        canonical_path = _canonical_archive_member_path(info.filename)
        if canonical_path is None:
            raise ConfirmatoryRuntimeError(
                "Runtime archive contains an unsafe member path."
            )
        if canonical_path in canonical_members:
            raise ConfirmatoryRuntimeError(
                "Runtime archive contains canonical member-path aliases."
            )
        canonical_members[canonical_path] = info.is_dir()
    for canonical_path in canonical_members:
        parts = PurePosixPath(canonical_path).parts
        for prefix_length in range(1, len(parts)):
            prefix = PurePosixPath(*parts[:prefix_length]).as_posix()
            if prefix in canonical_members and not canonical_members[prefix]:
                raise ConfirmatoryRuntimeError(
                    "Runtime archive contains a file/directory path collision."
                )


def _validate_archive_sources(archive: ZipFile) -> None:
    _validate_archive_layout(archive)
    infos = archive.infolist()

    sources = _load_embedded_manifest(archive)
    expected = dict(sources)
    runtime_infos = {
        info.filename: info
        for info in infos
        if not info.is_dir()
        and (
            _is_behavior_python_source(info.filename)
            or _is_confirmatory_config_source(info.filename)
        )
    }
    if any(
        _is_behavior_bytecode(info.filename)
        for info in infos
        if not info.is_dir()
    ):
        raise ConfirmatoryRuntimeError(
            "Runtime archive contains unverified behavior bytecode."
        )
    if set(runtime_infos) != set(expected):
        raise ConfirmatoryRuntimeError(
            "Runtime source inventory differs from the embedded manifest."
        )

    for relative_path, expected_digest in expected.items():
        digest = hashlib.sha256()
        try:
            with archive.open(runtime_infos[relative_path], "r") as source:
                for block in iter(lambda: source.read(_READ_SIZE), b""):
                    digest.update(block)
        except (BadZipFile, OSError, RuntimeError, zlib.error) as exc:
            raise ConfirmatoryRuntimeError(
                "Runtime source cannot be read completely."
            ) from exc
        if digest.hexdigest() != expected_digest:
            raise ConfirmatoryRuntimeError(
                "Runtime source differs from the embedded manifest."
            )


def validate_archive_layout(archive: ZipFile) -> None:
    """Public standard-library archive-layout gate for trusted launchers."""

    _validate_archive_layout(archive)


def validate_confirmatory_archive_sources(archive: ZipFile) -> None:
    """Public source-profile gate used by confirmatory and Phase 4 training."""

    _validate_archive_sources(archive)


def validate_runtime_archive(
    runtime_archive: str | os.PathLike[str],
    expected_sha256: str,
    *,
    archive_validator: Callable[[ZipFile], None] | None = None,
) -> VerifiedRuntime:
    """Authenticate one frozen standalone PAR without importing project code."""

    if not _LOWER_SHA256.fullmatch(expected_sha256):
        raise ConfirmatoryRuntimeError(
            "Expected runtime SHA-256 must be 64 lowercase hexadecimal characters."
        )
    requested_path = Path(runtime_archive)
    if not requested_path.is_absolute():
        raise ConfirmatoryRuntimeError("Runtime artifact path must be absolute.")
    try:
        requested_status = requested_path.lstat()
        path = requested_path.resolve(strict=True)
        path_before = path.lstat()
    except OSError as exc:
        raise ConfirmatoryRuntimeError("Runtime artifact does not exist.") from exc
    if stat.S_ISLNK(requested_status.st_mode):
        raise ConfirmatoryRuntimeError(
            "Runtime artifact path must not be a symlink."
        )
    if not stat.S_ISREG(path_before.st_mode):
        raise ConfirmatoryRuntimeError("Runtime artifact must be a regular file.")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfirmatoryRuntimeError("Runtime artifact cannot be opened.") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            descriptor_before = os.fstat(handle.fileno())
            actual_sha256 = _sha256_open_file(handle)
            if actual_sha256 != expected_sha256:
                raise ConfirmatoryRuntimeError(
                    "Runtime artifact differs from the expected SHA-256."
                )
            try:
                with ZipFile(handle, "r") as archive:
                    (archive_validator or _validate_archive_sources)(archive)
            except BadZipFile as exc:
                raise ConfirmatoryRuntimeError(
                    "Runtime artifact is not a valid ZIP-based PAR."
                ) from exc
            descriptor_after = os.fstat(handle.fileno())
    except ConfirmatoryRuntimeError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise ConfirmatoryRuntimeError(
            "Runtime artifact validation failed."
        ) from exc

    try:
        _assert_runtime_unchanged(
            path,
            path_before,
            descriptor_before,
            descriptor_after,
        )
    except ConfirmatoryRuntimeError:
        os.close(descriptor)
        raise
    try:
        sealed_descriptor = _sealed_runtime_copy(descriptor, actual_sha256)
    except ConfirmatoryRuntimeError:
        os.close(descriptor)
        raise
    os.close(descriptor)
    return VerifiedRuntime(
        path=path,
        sha256=actual_sha256,
        descriptor=sealed_descriptor,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
    )


def _open_private_unpack_directory(
    path: Path,
    created_status: os.stat_result,
) -> int:
    """Bind an empty unpack directory to one inherited descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfirmatoryRuntimeError(
            "Private confirmatory unpack directory cannot be opened."
        ) from exc
    try:
        opened_status = os.fstat(descriptor)
        current_status = path.lstat()
        if (
            len(
                {
                    _directory_identity(created_status),
                    _directory_identity(opened_status),
                    _directory_identity(current_status),
                }
            )
            != 1
            or not stat.S_ISDIR(opened_status.st_mode)
            or stat.S_ISLNK(current_status.st_mode)
            or opened_status.st_uid != os.geteuid()
            or stat.S_IMODE(opened_status.st_mode) != 0o700
            or os.listdir(descriptor)
        ):
            raise ConfirmatoryRuntimeError(
                "Private confirmatory unpack directory has an invalid identity."
            )
        os.set_inheritable(descriptor, True)
    except ConfirmatoryRuntimeError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise ConfirmatoryRuntimeError(
            "Private confirmatory unpack directory cannot be attested."
        ) from exc
    return descriptor


def _remove_private_unpack_directory(path: Path, descriptor: int) -> None:
    """Best-effort cleanup while the trusted namespace still names the object."""

    try:
        path_status = path.lstat()
        descriptor_status = os.fstat(descriptor)
    except OSError:
        return
    if (
        _directory_identity(path_status) == _directory_identity(descriptor_status)
        and stat.S_ISDIR(path_status.st_mode)
        and not stat.S_ISLNK(path_status.st_mode)
        and path_status.st_uid == os.geteuid()
        and stat.S_IMODE(path_status.st_mode) == 0o700
    ):
        shutil.rmtree(path, ignore_errors=True)


def launch_verified_runtime(
    runtime: VerifiedRuntime,
    runtime_args: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
    required_argument: str | None = "--confirmatory",
    attestation_environment: Mapping[str, str] | None = None,
) -> int:
    """Run the exact verified PAR as a supervised child, without a shell."""

    arguments = list(runtime_args)
    if required_argument is not None and required_argument not in arguments:
        try:
            os.close(runtime.descriptor)
        except OSError:
            pass
        raise ConfirmatoryRuntimeError(
            f"Verified runtime must be launched with {required_argument}."
        )
    if not _MEMFD_SEALING_AVAILABLE:
        try:
            os.close(runtime.descriptor)
        except OSError:
            pass
        raise ConfirmatoryRuntimeError(
            "This host cannot inspect a sealed confirmatory runtime."
        )
    try:
        descriptor_status = os.fstat(runtime.descriptor)
        actual_seals = fcntl.fcntl(runtime.descriptor, _F_GET_SEALS)
    except OSError as exc:
        try:
            os.close(runtime.descriptor)
        except OSError:
            pass
        raise ConfirmatoryRuntimeError(
            "Verified runtime descriptor is no longer available."
        ) from exc
    if (
        not stat.S_ISREG(descriptor_status.st_mode)
        or actual_seals & _REQUIRED_MEMFD_SEALS != _REQUIRED_MEMFD_SEALS
    ):
        os.close(runtime.descriptor)
        raise ConfirmatoryRuntimeError(
            "Verified runtime descriptor is not an immutable sealed file."
        )
    try:
        os.set_inheritable(runtime.descriptor, True)
    except OSError as exc:
        try:
            os.close(runtime.descriptor)
        except OSError:
            pass
        raise ConfirmatoryRuntimeError(
            "Verified runtime descriptor cannot be inherited."
        ) from exc

    child_environment = dict(os.environ if environ is None else environ)
    for name in list(child_environment):
        if (
            name in _UNSAFE_CHILD_ENV_NAMES
            or name.startswith("BASH_FUNC_")
            or name.startswith("FB_PAR_")
            or name.startswith("GIT_")
            or name.startswith("LD_")
            or name.startswith("PAR_")
            or (name.startswith("PYTHON") and name != "PYTHONHASHSEED")
        ):
            child_environment.pop(name)
    child_environment["PATH"] = "/usr/bin:/bin"
    child_environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    child_environment[VERIFIED_RUNTIME_SHA256_ENV] = runtime.sha256
    child_environment[VERIFIED_RUNTIME_FD_ENV] = str(runtime.descriptor)
    exec_path = f"/proc/self/fd/{runtime.descriptor}"
    child_environment[VERIFIED_RUNTIME_PATH_ENV] = exec_path
    child_environment[PAR_FILENAME_ENV] = exec_path
    private_unpack_base: Path | None = None
    private_unpack_descriptor: int | None = None
    try:
        # The host namespace is trusted until this descriptor is acquired.
        # Execution and extraction use only the descriptor-backed path afterward.
        private_unpack_base = Path(
            tempfile.mkdtemp(prefix="upi_trm_confirmatory_unpack.")
        )
        if not private_unpack_base.is_absolute():
            raise ConfirmatoryRuntimeError(
                "Private confirmatory unpack directory must be absolute."
            )
        created_status = private_unpack_base.lstat()
        private_unpack_descriptor = _open_private_unpack_directory(
            private_unpack_base,
            created_status,
        )
    except (ConfirmatoryRuntimeError, OSError) as exc:
        try:
            os.close(runtime.descriptor)
        except OSError:
            pass
        if private_unpack_base is not None:
            shutil.rmtree(private_unpack_base, ignore_errors=True)
        if isinstance(exc, ConfirmatoryRuntimeError):
            raise
        raise ConfirmatoryRuntimeError(
            "Private confirmatory unpack directory cannot be created."
        ) from exc
    assert private_unpack_base is not None
    assert private_unpack_descriptor is not None
    private_unpack_path = f"/proc/self/fd/{private_unpack_descriptor}"
    child_environment["FB_PAR_UNPACK_BASEDIR"] = private_unpack_path
    child_environment[PRIVATE_UNPACK_BASE_ENV] = private_unpack_path
    child_environment[PRIVATE_UNPACK_FD_ENV] = str(private_unpack_descriptor)
    if attestation_environment is not None:
        reserved_names = {
            VERIFIED_RUNTIME_PATH_ENV,
            VERIFIED_RUNTIME_SHA256_ENV,
            VERIFIED_RUNTIME_FD_ENV,
            PRIVATE_UNPACK_BASE_ENV,
            PRIVATE_UNPACK_FD_ENV,
            PAR_FILENAME_ENV,
            "FB_PAR_UNPACK_BASEDIR",
        }
        if any(
            name in reserved_names or name.startswith("GIT_")
            for name in attestation_environment
        ):
            os.close(runtime.descriptor)
            _remove_private_unpack_directory(
                private_unpack_base,
                private_unpack_descriptor,
            )
            os.close(private_unpack_descriptor)
            raise ConfirmatoryRuntimeError(
                "Additional runtime attestation attempts to replace a reserved field."
            )
        child_environment.update(attestation_environment)
    argv = [exec_path, *arguments]
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            argv,
            executable=exec_path,
            env=child_environment,
            close_fds=True,
            pass_fds=(runtime.descriptor, private_unpack_descriptor),
        )
    except (OSError, ValueError) as exc:
        os.close(runtime.descriptor)
        _remove_private_unpack_directory(
            private_unpack_base,
            private_unpack_descriptor,
        )
        os.close(private_unpack_descriptor)
        raise ConfirmatoryRuntimeError(
            "Verified runtime could not be executed."
        ) from exc
    os.close(runtime.descriptor)
    try:
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        _remove_private_unpack_directory(
            private_unpack_base,
            private_unpack_descriptor,
        )
        os.close(private_unpack_descriptor)
    return return_code


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and execute a confirmatory UPI-TRM PAR.",
    )
    parser.add_argument("--runtime-archive", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    runtime_args = list(arguments.runtime_args)
    if runtime_args[:1] == ["--"]:
        runtime_args = runtime_args[1:]
    try:
        runtime = validate_runtime_archive(
            arguments.runtime_archive,
            arguments.expected_runtime_sha256,
        )
        return_code = launch_verified_runtime(runtime, runtime_args)
    except ConfirmatoryRuntimeError as exc:
        print(f"confirmatory runtime rejected: {exc}", file=sys.stderr)
        return 2
    return 128 - return_code if return_code < 0 else return_code


if __name__ == "__main__":
    raise SystemExit(main())
