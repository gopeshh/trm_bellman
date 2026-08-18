#!/usr/bin/env fbpython
"""Write-sealed checkpoint descriptors for authenticated evidence packages.

Checkpoint payloads are historically pickled ``torch.save`` documents, so
deserializing one executes arbitrary code.  A pathname is not a safe way to
name those bytes: the authenticated inventory and the loader would resolve the
name twice, and anything with write access to the generation can swap the file
between the two resolutions.

This module is the single place that converts *already authenticated* bytes
into a name a loader may safely consume.  It copies the exact bytes into an
anonymous file, verifies that the source did not change while the copy was
made, applies the full write-seal mask, and hands back a descriptor.  A sealed
descriptor cannot be regrown, shrunk, rewritten, or unsealed, so the bytes a
loader reads are provably the bytes that were authenticated.

The module deliberately uses only the standard library.  Every consumer that
authenticates evidence must be able to import it before Torch, the training
module, or any checkpoint loader is imported.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


SEALED_CHECKPOINT_SCHEMA_NAME = "policy_improvement_sealed_checkpoint_v1"
SEALED_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_name",
        "descriptor",
        "sha256",
        "size_bytes",
        "generation_relative_path",
    }
)
_READ_SIZE = 1024 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")
_REQUIRED_OS_NAMES = ("memfd_create", "MFD_ALLOW_SEALING", "MFD_CLOEXEC")
_REQUIRED_FCNTL_NAMES = (
    "F_ADD_SEALS",
    "F_GET_SEALS",
    "F_SEAL_GROW",
    "F_SEAL_SEAL",
    "F_SEAL_SHRINK",
    "F_SEAL_WRITE",
)


class SealedCheckpointError(RuntimeError):
    """Raised when checkpoint bytes cannot be sealed or a seal is not intact."""


def _require_sealing_support() -> int:
    if any(not hasattr(os, name) for name in _REQUIRED_OS_NAMES) or any(
        not hasattr(fcntl, name) for name in _REQUIRED_FCNTL_NAMES
    ):
        raise SealedCheckpointError("Sealed checkpoint descriptors are unavailable.")
    return (
        fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SEAL
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_WRITE
    )


def _lower_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise SealedCheckpointError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _exact_size(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SealedCheckpointError(f"{name} must be a non-negative integer.")
    return value


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def canonical_generation_relative_path(value: object, *, name: str) -> str:
    """Reject anything that is not one canonical relative POSIX path."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise SealedCheckpointError(f"{name} is not a canonical relative path.")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise SealedCheckpointError(f"{name} is not a canonical relative path.")
    return value


def seal_authenticated_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> int:
    """Copy exact checkpoint bytes into a write-sealed anonymous file.

    The caller must already have authenticated ``expected_sha256`` and
    ``expected_size_bytes`` against an immutable manifest.  This function only
    proves that the named file still holds exactly those bytes and that it did
    not change while the sealed copy was taken.
    """

    seals = _require_sealing_support()
    expected_digest = _lower_sha256(expected_sha256, name="Sealed checkpoint digest")
    expected_size = _exact_size(expected_size_bytes, name="Sealed checkpoint size")
    source_descriptor = -1
    sealed_descriptor = -1
    try:
        before_path = path.lstat()
        source_descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(source_descriptor)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before_path.st_nlink != 1
            or before.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise SealedCheckpointError("Checkpoint snapshot source is unsafe.")
        sealed_descriptor = os.memfd_create(
            "upi-trm-authenticated-checkpoint",
            os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(source_descriptor, _READ_SIZE, offset)
            if not block:
                break
            digest.update(block)
            remaining = memoryview(block)
            while remaining:
                written = os.write(sealed_descriptor, remaining)
                if written <= 0:
                    raise SealedCheckpointError(
                        "Sealed checkpoint snapshot made no write progress."
                    )
                remaining = remaining[written:]
            offset += len(block)
        after = os.fstat(source_descriptor)
        after_path = path.lstat()
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(after_path)
            or offset != expected_size
            or digest.hexdigest() != expected_digest
        ):
            raise SealedCheckpointError(
                "Checkpoint changed while its sealed snapshot was created."
            )
        os.fchmod(sealed_descriptor, 0o400)
        fcntl.fcntl(sealed_descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(sealed_descriptor, fcntl.F_GET_SEALS) & seals != seals:
            raise SealedCheckpointError(
                "Checkpoint snapshot could not be write-sealed."
            )
        os.lseek(sealed_descriptor, 0, os.SEEK_SET)
        result = sealed_descriptor
        sealed_descriptor = -1
        return result
    except OSError as exc:
        raise SealedCheckpointError("Checkpoint snapshot could not be sealed.") from exc
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if sealed_descriptor >= 0:
            os.close(sealed_descriptor)


def authenticate_sealed_descriptor(
    descriptor: int,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[str, int]:
    """Require one stable, fully write-sealed descriptor and hash its bytes."""

    seals = _require_sealing_support()
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
    ):
        raise SealedCheckpointError("Sealed checkpoint descriptor is invalid.")
    expected_digest = (
        None
        if expected_sha256 is None
        else _lower_sha256(expected_sha256, name="Sealed checkpoint digest")
    )
    expected_size = (
        None
        if expected_size_bytes is None
        else _exact_size(expected_size_bytes, name="Sealed checkpoint size")
    )
    try:
        info = os.fstat(descriptor)
        observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            block = os.pread(descriptor, min(_READ_SIZE, info.st_size - offset), offset)
            if not block:
                raise SealedCheckpointError(
                    "Sealed checkpoint descriptor ended before its registered size."
                )
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise SealedCheckpointError(
            "Sealed checkpoint descriptor cannot be authenticated."
        ) from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or observed_seals & seals != seals
        or (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise SealedCheckpointError(
            "Sealed checkpoint descriptor is not one stable write-sealed file."
        )
    observed_digest = digest.hexdigest()
    if (expected_digest is not None and observed_digest != expected_digest) or (
        expected_size is not None and offset != expected_size
    ):
        raise SealedCheckpointError(
            "Sealed checkpoint descriptor differs from its authenticated identity."
        )
    return observed_digest, offset


@dataclass(frozen=True)
class SealedCheckpoint:
    """One authenticated checkpoint named only by a write-sealed descriptor."""

    descriptor: int
    sha256: str
    size_bytes: int
    generation_relative_path: str

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)

    def as_request_field(self) -> dict[str, object]:
        """Render the descriptor bundle carried by a validation request."""

        return {
            "schema_name": SEALED_CHECKPOINT_SCHEMA_NAME,
            "descriptor": self.descriptor,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "generation_relative_path": self.generation_relative_path,
        }


def seal_generation_checkpoint(
    generation: Path,
    generation_relative_path: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> SealedCheckpoint:
    """Seal one inventoried file of an authenticated immutable generation."""

    relative = canonical_generation_relative_path(
        generation_relative_path,
        name="Sealed checkpoint generation path",
    )
    descriptor = seal_authenticated_checkpoint(
        generation / relative,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    return SealedCheckpoint(
        descriptor=descriptor,
        sha256=_lower_sha256(expected_sha256, name="Sealed checkpoint digest"),
        size_bytes=_exact_size(expected_size_bytes, name="Sealed checkpoint size"),
        generation_relative_path=relative,
    )


def authenticate_sealed_checkpoint_field(
    value: object,
    *,
    expected_sha256: str,
    name: str = "checkpoint",
) -> SealedCheckpoint:
    """Consume the descriptor bundle of a validation request, fail-closed.

    The returned object never exposes a pathname.  Callers must load through
    ``SealedCheckpoint.descriptor`` so that the bytes they deserialize are the
    bytes the producer of the request authenticated.
    """

    if not isinstance(value, Mapping) or set(value) != SEALED_CHECKPOINT_FIELDS:
        raise SealedCheckpointError(f"{name} sealed-descriptor fields differ.")
    if value["schema_name"] != SEALED_CHECKPOINT_SCHEMA_NAME:
        raise SealedCheckpointError(f"{name} sealed-descriptor schema is unsupported.")
    descriptor = value["descriptor"]
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
    ):
        raise SealedCheckpointError(f"{name} sealed descriptor is invalid.")
    declared_digest = _lower_sha256(value["sha256"], name=f"{name}.sha256")
    declared_size = _exact_size(value["size_bytes"], name=f"{name}.size_bytes")
    if declared_digest != _lower_sha256(expected_sha256, name=f"{name}.expected"):
        raise SealedCheckpointError(
            f"{name} sealed descriptor names another checkpoint digest."
        )
    relative = canonical_generation_relative_path(
        value["generation_relative_path"],
        name=f"{name}.generation_relative_path",
    )
    observed_digest, observed_size = authenticate_sealed_descriptor(
        descriptor,
        expected_sha256=declared_digest,
        expected_size_bytes=declared_size,
    )
    return SealedCheckpoint(
        descriptor=descriptor,
        sha256=observed_digest,
        size_bytes=observed_size,
        generation_relative_path=relative,
    )


def load_sealed_checkpoint_validator() -> (
    Callable[[Mapping[str, object]], Mapping[str, object]]
):
    """Return a validator that imports Torch only when it is first called.

    The audit entrypoint must authenticate and seal evidence with the standard
    library alone.  Importing ``policy_improvement_checkpoint_validator`` at
    entrypoint import time would execute Torch, the training module, and every
    checkpoint loader before one byte of evidence had been authenticated.  This
    factory defers that import to the first sealed validation request.
    """

    def validate(request: Mapping[str, object]) -> Mapping[str, object]:
        module = importlib.import_module("policy_improvement_checkpoint_validator")
        return module.validate_checkpoint(request)

    return validate
