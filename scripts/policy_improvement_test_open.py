#!/usr/bin/env fbpython
"""Publish and authenticate the protocol's one immutable test-data opening."""

from __future__ import annotations

import ctypes
import datetime as dt
import errno
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    amendment_history_sha256,
    canonical_json_bytes,
    runtime_authorization_sha256,
    validate_amendment_history,
    validate_protocol,
    validate_runtime_authorization,
)
from scripts.policy_improvement_registry import (
    registry_sha256,
    validate_registry_document,
)


TEST_OPEN_SCHEMA_VERSION = 3
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


def _timestamp(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise PolicyImprovementSchemaError(
            f"{path} must be a second-resolution UTC timestamp."
        )
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PolicyImprovementSchemaError(f"{path} is not a real UTC time.") from exc
    return value


def _sha256(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolicyImprovementSchemaError(f"{path} must be a lowercase SHA-256.")
    return value


def _private_owner_root(value: str | Path) -> tuple[Path, int, tuple[int, int]]:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise PolicyImprovementSchemaError("Test-open owner root must be absolute.")
    try:
        status = supplied.lstat()
        root = supplied.resolve(strict=True)
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            "Test-open owner root is unavailable."
        ) from exc
    if (
        supplied != root
        or not stat.S_ISDIR(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
        or (status.st_dev, status.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise PolicyImprovementSchemaError(
            "Test-open owner root must be one canonical private owner directory."
        )
    return root, descriptor, (opened.st_dev, opened.st_ino)


def _revalidate_owner(root: Path, descriptor: int, identity: tuple[int, int]) -> None:
    try:
        path_status = root.lstat()
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise PolicyImprovementSchemaError("Test-open owner root changed.") from exc
    if (
        not stat.S_ISDIR(path_status.st_mode)
        or stat.S_ISLNK(path_status.st_mode)
        or (path_status.st_dev, path_status.st_ino) != identity
        or (opened.st_dev, opened.st_ino) != identity
        or stat.S_IMODE(path_status.st_mode) != 0o700
        or path_status.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSchemaError("Test-open owner root changed.")


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise PolicyImprovementSchemaError("renameat2 is required for test opening.")
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
            raise PolicyImprovementSchemaError(
                "The registered test population was already opened."
            )
        raise PolicyImprovementSchemaError(
            f"Atomic test-open publication failed: {os.strerror(error)}."
        )


def _stable_file(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise PolicyImprovementSchemaError(
                    "TEST_OPEN.json is a symlink, alias, or wrong type."
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
        raise PolicyImprovementSchemaError("TEST_OPEN.json is unavailable.") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(opened) != identity(after) or identity(after) != identity(after_path):
        raise PolicyImprovementSchemaError("TEST_OPEN.json changed while hashing.")
    return b"".join(chunks), digest.hexdigest()


def expected_test_open_record(
    *,
    protocol: Mapping[str, object],
    registry: Mapping[str, object],
    amendment_history: Sequence[Mapping[str, object]],
    runtime_authorization: Mapping[str, object],
    opened_at_utc: str,
    base_configs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    checked_protocol = validate_protocol(protocol)
    checked_history = validate_amendment_history(
        amendment_history, protocol=checked_protocol
    )
    if len(checked_history) != 3:
        raise PolicyImprovementSchemaError(
            "Test data cannot open before the final registered selection."
        )
    checked_registry = validate_registry_document(
        registry,
        checked_protocol,
        checked_history,
        base_configs=base_configs,
    )
    checked_authorization = validate_runtime_authorization(runtime_authorization)
    opened = _timestamp(opened_at_utc, path="test_open.opened_at_utc")
    final_selection_time = _timestamp(
        checked_history[-1]["created_at_utc"],
        path="final_selection.created_at_utc",
    )
    if opened <= final_selection_time:
        raise PolicyImprovementSchemaError(
            "Test opening must occur after the final registered selection."
        )
    test_manifest = checked_protocol["dataset"]["splits"]["test"]["manifest_sha256"]
    if test_manifest.get("status") != "available":
        raise PolicyImprovementSchemaError(
            "Test data cannot open before its manifest is frozen."
        )
    protocol_digest = hashlib.sha256(canonical_json_bytes(checked_protocol)).hexdigest()
    if checked_authorization["protocol_sha256"] != protocol_digest:
        raise PolicyImprovementSchemaError(
            "Test opening runtime authorization names another protocol."
        )
    return {
        "schema_name": "policy_improvement_test_open_v1",
        "schema_version": TEST_OPEN_SCHEMA_VERSION,
        "record_id": "confirmatory-test-open-v1",
        "state": "opened_immutable",
        "protocol_sha256": protocol_digest,
        "registry_sha256": registry_sha256(checked_registry),
        "amendment_history_sha256": amendment_history_sha256(checked_history),
        "runtime_authorization_sha256": runtime_authorization_sha256(
            checked_authorization
        ),
        "test_manifest_sha256": str(test_manifest["value"]),
        "prior_open_record_sha256": hashlib.sha256(
            canonical_json_bytes([])
        ).hexdigest(),
        "open_ordinal": 1,
        "authorized_phases": ["stage2_confirmatory", "stage3_ablation"],
        "final_selection_created_at_utc": final_selection_time,
        "opened_at_utc": opened,
    }


def _publish_record(
    *,
    owner_root: str | Path,
    record: Mapping[str, object],
) -> dict[str, str]:
    """Publish TEST_OPEN.json exactly once and return its immutable identity."""

    root, root_descriptor, root_identity = _private_owner_root(owner_root)
    final = root / "TEST_OPEN.json"
    payload = canonical_json_bytes(record) + b"\n"
    staging: Path | None = None
    staging_identity: tuple[int, int] | None = None
    published = False
    try:
        descriptor, name = tempfile.mkstemp(prefix=".test-open-stage.", dir=root)
        staging = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, payload)
            os.fsync(descriptor)
            info = os.fstat(descriptor)
            staging_identity = (info.st_dev, info.st_ino)
        finally:
            os.close(descriptor)
        _revalidate_owner(root, root_descriptor, root_identity)
        _rename_noreplace(staging, final)
        published = True
        os.fsync(root_descriptor)
        _revalidate_owner(root, root_descriptor, root_identity)
        final_payload, digest = _stable_file(final)
        if final_payload != payload:
            raise PolicyImprovementSchemaError(
                "Published TEST_OPEN.json differs from staged bytes."
            )
        return {"path": str(final), "sha256": digest}
    finally:
        if not published and staging is not None and staging_identity is not None:
            try:
                current = staging.lstat()
                if (current.st_dev, current.st_ino) == staging_identity:
                    staging.unlink()
                    os.fsync(root_descriptor)
            except OSError:
                pass
        os.close(root_descriptor)


def publish_test_open(
    *,
    owner_root: str | Path,
    protocol: Mapping[str, object],
    registry: Mapping[str, object],
    amendment_history: Sequence[Mapping[str, object]],
    runtime_authorization: Mapping[str, object],
    base_configs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, str]:
    """Validate final selection and atomically publish the only test opening."""

    opened_at_utc = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    record = expected_test_open_record(
        protocol=protocol,
        registry=registry,
        amendment_history=amendment_history,
        runtime_authorization=runtime_authorization,
        opened_at_utc=opened_at_utc,
        base_configs=base_configs,
    )
    return _publish_record(owner_root=owner_root, record=record)


def _authenticate_record(
    *,
    owner_root: str | Path,
    expected_record: Mapping[str, object],
    expected_sha256: str,
) -> dict[str, Any]:
    root, descriptor, identity = _private_owner_root(owner_root)
    try:
        _revalidate_owner(root, descriptor, identity)
        payload, observed = _stable_file(root / "TEST_OPEN.json")
        if observed != _sha256(expected_sha256, path="test_open_sha256"):
            raise PolicyImprovementSchemaError("TEST_OPEN.json digest differs.")
        expected_payload = canonical_json_bytes(expected_record) + b"\n"
        if payload != expected_payload:
            raise PolicyImprovementSchemaError("TEST_OPEN.json content differs.")
        _revalidate_owner(root, descriptor, identity)
        return {"path": str(root / "TEST_OPEN.json"), "sha256": observed}
    finally:
        os.close(descriptor)


def authenticate_test_open(
    *,
    owner_root: str | Path,
    protocol: Mapping[str, object],
    registry: Mapping[str, object],
    amendment_history: Sequence[Mapping[str, object]],
    runtime_authorization: Mapping[str, object],
    opened_at_utc: str,
    expected_sha256: str,
    base_configs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    expected_record = expected_test_open_record(
        protocol=protocol,
        registry=registry,
        amendment_history=amendment_history,
        runtime_authorization=runtime_authorization,
        opened_at_utc=opened_at_utc,
        base_configs=base_configs,
    )
    identity = _authenticate_record(
        owner_root=owner_root,
        expected_record=expected_record,
        expected_sha256=expected_sha256,
    )
    return {**identity, "record": expected_record}
