#!/usr/bin/env fbpython
"""Fail-closed audit for registered policy-improvement results.

The schema validator checks one result in isolation.  This module binds each
result to the exact protocol and registry row, accounts for failed cells, and
recomputes every reported endpoint from the immutable per-instance records.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import stat
import struct
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.policy_improvement_evidence import (
    authenticate_complete_generation,
    authenticate_failed_attempt,
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
    load_strict_json,
    load_strict_json_bytes,
    PHASE_AMENDMENT_PREFIX_LENGTH,
    PHASE_CONTRACTS,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    SCREEN_SELECTION_METHOD_ORDER,
    validate_amendment_history,
    validate_protocol,
    validate_result,
    validate_runtime_authorization,
    validated_result_payload,
)
from scripts.policy_improvement_test_open import authenticate_test_open
from utils.source_identity import (
    behavior_source_relative_paths_from_inventory,
    SOURCE_MANIFEST_RELATIVE_PATH,
    SourceIdentityError,
    validate_producer_source_manifest,
)


AUDIT_SCHEMA_VERSION = 5
PER_INSTANCE_SCHEMA_VERSION = 2
_DATASET_TOP_LEVEL_NAMES = {
    "MANIFEST.json",
    "build_config.json",
    "identifiers.json",
    "manifests",
    "test",
    "train",
    "validation",
}
_DATASET_SPLIT_FILE_NAMES = {
    "all__group_indices.npy",
    "all__inputs.npy",
    "all__labels.npy",
    "all__puzzle_identifiers.npy",
    "all__puzzle_indices.npy",
    "dataset.json",
    "records.json",
}
_DATASET_PRODUCER_SOURCE_PATHS = (
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
_DATASET_CANONICALIZATION = {
    "group_order": 3072,
    "reject_cross_split_overlap": True,
    "scheme": "sudoku4x4_spatial_digit_lexicographic_v1",
}


def _object(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementSchemaError(f"{path} must be an object.")
    return value


def _fields(value: object, expected: set[str], *, path: str) -> Mapping[str, object]:
    item = _object(value, path=path)
    actual = set(item)
    if actual != expected:
        raise PolicyImprovementSchemaError(
            f"{path} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )
    return item


def _ascii(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise PolicyImprovementSchemaError(f"{path} must be nonempty ASCII text.")
    return value


def _hex(value: object, *, path: str, length: int) -> str:
    text = _ascii(value, path=path)
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise PolicyImprovementSchemaError(
            f"{path} must be {length} lowercase hexadecimal characters."
        )
    return text


def _integer(value: object, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyImprovementSchemaError(
            f"{path} must be an integer greater than or equal to {minimum}."
        )
    return value


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyImprovementSchemaError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyImprovementSchemaError(f"{path} must be finite.")
    return result


def _git_object_bytes(
    project_root: Path,
    arguments: Sequence[str],
    *,
    label: str,
) -> bytes:
    """Read one exact Git object without a shell or working-tree filters."""

    try:
        git_environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        git_environment.update(
            {
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        completed = subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=project_root,
            check=False,
            env=git_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyImprovementSchemaError(
            f"Historical producer {label} could not be resolved."
        ) from exc
    if completed.returncode != 0:
        raise PolicyImprovementSchemaError(
            f"Historical producer {label} could not be resolved."
        )
    return completed.stdout


def _authenticate_authorization_producer_source(
    authorization: Mapping[str, object],
    *,
    project_root: str | Path,
) -> None:
    """Resolve one authorized producer commit and rehash its complete manifest."""

    root = Path(project_root)
    try:
        root_status = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            "Historical producer project root is unavailable."
        ) from exc
    if (
        root != resolved_root
        or stat.S_ISLNK(root_status.st_mode)
        or not stat.S_ISDIR(root_status.st_mode)
        or root_status.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSchemaError(
            "Historical producer project root is aliased or unowned."
        )
    commit = str(authorization["producer_git_commit"])
    resolved_commit = _git_object_bytes(
        root,
        ["rev-parse", "--verify", f"{commit}^{{commit}}"],
        label="commit",
    ).strip()
    if resolved_commit != commit.encode("ascii"):
        raise PolicyImprovementSchemaError(
            "Historical producer commit does not resolve exactly."
        )
    manifest_bytes = _git_object_bytes(
        root,
        ["cat-file", "blob", f"{commit}:{SOURCE_MANIFEST_RELATIVE_PATH}"],
        label="manifest",
    )
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != authorization["producer_source_manifest_sha256"]
    ):
        raise PolicyImprovementSchemaError(
            "Historical producer manifest digest differs from its commit."
        )
    try:
        manifest = validate_producer_source_manifest(
            load_strict_json_bytes(manifest_bytes)
        )
    except (PolicyImprovementSchemaError, SourceIdentityError) as exc:
        raise PolicyImprovementSchemaError(
            "Historical producer manifest is invalid."
        ) from exc
    tree_output = _git_object_bytes(
        root,
        ["ls-tree", "-r", "--name-only", "-z", commit],
        label="tree inventory",
    )
    try:
        tree_paths = [item.decode("utf-8") for item in tree_output.split(b"\0") if item]
        # The manifest's own schema version selects the inventory this
        # producer commit must satisfy, and the selector rejects a version
        # lower than the tree actually satisfies, so a historical manifest
        # cannot be replayed against a newer tree to drop sources.
        expected_sources = behavior_source_relative_paths_from_inventory(
            tree_paths,
            schema_version=int(manifest["source_manifest_schema_version"]),
        )
    except (UnicodeDecodeError, SourceIdentityError) as exc:
        raise PolicyImprovementSchemaError(
            "Historical producer tree inventory is invalid."
        ) from exc
    sources = manifest["sources"]
    if list(sources) != expected_sources:
        raise PolicyImprovementSchemaError(
            "Historical producer manifest inventory differs from its commit."
        )
    for relative_path, expected_sha256 in sources.items():
        source_bytes = _git_object_bytes(
            root,
            ["cat-file", "blob", f"{commit}:{relative_path}"],
            label=f"source {relative_path}",
        )
        if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
            raise PolicyImprovementSchemaError(
                "Historical producer source digest differs from its manifest."
            )


def _validated_runtime_authorization_map(
    *,
    current_authorization: Mapping[str, object],
    historical_runtime_authorizations: Mapping[str, Mapping[str, object]] | None,
    protocol_sha256: str,
    producer_source_authenticator: Callable[[Mapping[str, object]], None],
) -> dict[str, Mapping[str, object]]:
    """Validate every authorization used by a retained failed attempt."""

    current = validate_runtime_authorization(current_authorization)
    producer_source_authenticator(current)
    current_digest = runtime_authorization_sha256(current)
    checked: dict[str, Mapping[str, object]] = {current_digest: current}
    for supplied_digest, supplied_document in (
        historical_runtime_authorizations or {}
    ).items():
        digest = _hex(
            supplied_digest,
            path="historical_runtime_authorizations.key",
            length=64,
        )
        document = validate_runtime_authorization(supplied_document)
        producer_source_authenticator(document)
        if runtime_authorization_sha256(document) != digest:
            raise PolicyImprovementSchemaError(
                "Historical runtime authorization digest differs from its document."
            )
        if document["protocol_sha256"] != protocol_sha256:
            raise PolicyImprovementSchemaError(
                "Historical runtime authorization names a different protocol."
            )
        prior = checked.get(digest)
        if prior is not None and canonical_json_bytes(prior) != canonical_json_bytes(
            document
        ):
            raise PolicyImprovementSchemaError(
                "One runtime authorization digest names different documents."
            )
        checked[digest] = document
    return checked


def _load_historical_runtime_authorizations(
    specifications: Sequence[str],
    *,
    protocol_sha256: str,
) -> dict[str, Mapping[str, object]]:
    """Load repeatable PATH=SHA256 authorization arguments without aliases."""

    loaded: dict[str, Mapping[str, object]] = {}
    for specification in specifications:
        if specification.count("=") != 1:
            raise PolicyImprovementSchemaError(
                "--historical-runtime-authorization must use PATH=SHA256."
            )
        path_text, expected_digest = specification.rsplit("=", 1)
        digest = _hex(
            expected_digest,
            path="historical_runtime_authorization.sha256",
            length=64,
        )
        path = Path(path_text)
        if not path.is_absolute():
            raise PolicyImprovementSchemaError(
                "Historical runtime authorization path must be absolute."
            )
        payload, _ = _stable_regular_file(path)
        try:
            payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PolicyImprovementSchemaError(
                "Historical runtime authorization is not ASCII JSON."
            ) from exc
        document = validate_runtime_authorization(load_strict_json_bytes(payload))
        if runtime_authorization_sha256(document) != digest:
            raise PolicyImprovementSchemaError(
                "Historical runtime authorization digest differs."
            )
        if document["protocol_sha256"] != protocol_sha256:
            raise PolicyImprovementSchemaError(
                "Historical runtime authorization names a different protocol."
            )
        if digest in loaded:
            raise PolicyImprovementSchemaError(
                "Duplicate historical runtime authorization digest."
            )
        loaded[digest] = document
    return loaded


def _historical_failed_attempt_manifest_sha256s(
    generation_identity: Mapping[str, object],
) -> list[str]:
    """Extract the exact retained-failure manifest inventory from one run."""

    raw_attempts = generation_identity.get("historical_failed_attempts")
    if not isinstance(raw_attempts, list):
        raise PolicyImprovementSchemaError(
            "Complete evidence lacks its historical failed-attempt inventory."
        )
    manifests: list[str] = []
    for index, raw_attempt in enumerate(raw_attempts):
        attempt = _fields(
            raw_attempt,
            {
                "attempt_id",
                "segment",
                "failure_phase",
                "runtime_authorization_sha256",
                "generation_manifest_sha256",
                "result_sha256",
            },
            path=f"historical_failed_attempts[{index}]",
        )
        _hex(
            attempt["attempt_id"],
            path=f"historical_failed_attempts[{index}].attempt_id",
            length=32,
        )
        _ascii(
            attempt["segment"],
            path=f"historical_failed_attempts[{index}].segment",
        )
        _ascii(
            attempt["failure_phase"],
            path=f"historical_failed_attempts[{index}].failure_phase",
        )
        _hex(
            attempt["runtime_authorization_sha256"],
            path=f"historical_failed_attempts[{index}].runtime_authorization_sha256",
            length=64,
        )
        manifest = _hex(
            attempt["generation_manifest_sha256"],
            path=f"historical_failed_attempts[{index}].generation_manifest_sha256",
            length=64,
        )
        _hex(
            attempt["result_sha256"],
            path=f"historical_failed_attempts[{index}].result_sha256",
            length=64,
        )
        manifests.append(manifest)
    if len(manifests) != len(set(manifests)):
        raise PolicyImprovementSchemaError(
            "Historical failed-attempt manifest digests must be unique."
        )
    return manifests


def validate_per_instance_document(value: object) -> dict[str, Any]:
    """Validate ordered puzzle-level evidence used to recompute aggregates."""

    if not isinstance(value, Mapping):
        raise PolicyImprovementSchemaError("Per-instance document must be an object.")
    is_v2 = value.get("schema_name") == "policy_improvement_instances_v2"
    top_fields = {
        "schema_name",
        "schema_version",
        "protocol_id",
        "run_id",
        "phase",
        "tier",
        "seed",
        "evaluation_split",
        "method_id",
        "snapshot_kind",
        "evaluation_id",
        "policy_variant",
        "evaluation_pool_sha256",
        "records",
    }
    if is_v2:
        top_fields.update(
            {
                "evaluation_population_id",
                "evaluation_population_binding_sha256",
            }
        )
    document = _fields(
        value,
        top_fields,
        path="per_instance",
    )
    if document["schema_name"] not in {
        "policy_improvement_instances_v1",
        "policy_improvement_instances_v2",
    }:
        raise PolicyImprovementSchemaError("Unexpected per-instance schema.")
    expected_schema_version = 1 if is_v2 else PER_INSTANCE_SCHEMA_VERSION
    if document["schema_version"] != expected_schema_version:
        raise PolicyImprovementSchemaError("Unsupported per-instance schema version.")
    if is_v2:
        _ascii(
            document["evaluation_population_id"],
            path="per_instance.evaluation_population_id",
        )
        _hex(
            document["evaluation_population_binding_sha256"],
            path="per_instance.evaluation_population_binding_sha256",
            length=64,
        )
    for field in (
        "protocol_id",
        "run_id",
        "phase",
        "tier",
        "evaluation_split",
        "method_id",
        "snapshot_kind",
        "evaluation_id",
        "policy_variant",
    ):
        _ascii(document[field], path=f"per_instance.{field}")
    _integer(document["seed"], path="per_instance.seed")
    _hex(
        document["evaluation_pool_sha256"],
        path="per_instance.evaluation_pool_sha256",
        length=64,
    )
    raw_records = document["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise PolicyImprovementSchemaError(
            "per_instance.records must be a nonempty ordered list."
        )
    records: list[dict[str, Any]] = []
    puzzle_ids: list[str] = []
    for index, raw in enumerate(raw_records):
        path = f"per_instance.records[{index}]"
        record_fields = {
            "registered_index",
            "registered_record_sha256",
            "puzzle_id",
            "puzzle_sha256",
            "solved",
            "discounted_return",
            "terminal_reason",
            "edits_to_solve",
            "value_prediction",
            "realized_return",
        }
        if is_v2:
            record_fields.update(
                {
                    "population_position",
                    "original_dataset_index",
                    "registered_input_sha256",
                }
            )
        record = _fields(
            raw,
            record_fields,
            path=path,
        )
        registered_index = _integer(
            record["registered_index"], path=f"{path}.registered_index"
        )
        if registered_index != index:
            raise PolicyImprovementSchemaError(
                f"{path}.registered_index differs from the frozen ordered population."
            )
        if is_v2:
            if (
                _integer(
                    record["population_position"],
                    path=f"{path}.population_position",
                )
                != index
            ):
                raise PolicyImprovementSchemaError(
                    f"{path}.population_position differs from its ordered position."
                )
            _integer(
                record["original_dataset_index"],
                path=f"{path}.original_dataset_index",
                minimum=0,
            )
        _hex(
            record["registered_record_sha256"],
            path=f"{path}.registered_record_sha256",
            length=64,
        )
        puzzle_id = _ascii(record["puzzle_id"], path=f"{path}.puzzle_id")
        puzzle_ids.append(puzzle_id)
        _hex(record["puzzle_sha256"], path=f"{path}.puzzle_sha256", length=64)
        if is_v2:
            registered_input = _hex(
                record["registered_input_sha256"],
                path=f"{path}.registered_input_sha256",
                length=64,
            )
            if registered_input != record["puzzle_sha256"]:
                raise PolicyImprovementSchemaError(f"{path} input identities differ.")
        if not isinstance(record["solved"], bool):
            raise PolicyImprovementSchemaError(f"{path}.solved must be boolean.")
        _number(record["discounted_return"], path=f"{path}.discounted_return")
        terminal = _ascii(record["terminal_reason"], path=f"{path}.terminal_reason")
        if terminal not in {"budget", "solved", "stop"}:
            raise PolicyImprovementSchemaError(
                f"{path}.terminal_reason is not registered."
            )
        if (terminal == "solved") is not record["solved"]:
            raise PolicyImprovementSchemaError(
                f"{path} solved flag differs from terminal reason."
            )
        edits = record["edits_to_solve"]
        if record["solved"]:
            _integer(edits, path=f"{path}.edits_to_solve", minimum=0)
        elif edits is not None:
            raise PolicyImprovementSchemaError(
                f"{path}.edits_to_solve must be null when unsolved."
            )
        _number(record["value_prediction"], path=f"{path}.value_prediction")
        _number(record["realized_return"], path=f"{path}.realized_return")
        records.append(dict(record))
    if len(puzzle_ids) != len(set(puzzle_ids)):
        raise PolicyImprovementSchemaError(
            "Per-instance puzzle identifiers must be unique."
        )
    computed_pool = hashlib.sha256(
        canonical_json_bytes(
            [str(record["registered_record_sha256"]) for record in records]
        )
    ).hexdigest()
    if document["evaluation_pool_sha256"] != computed_pool:
        raise PolicyImprovementSchemaError(
            "Evaluation-pool digest does not bind the ordered puzzle records."
        )
    canonical_json_bytes(document)
    return {**dict(document), "records": records}


def _available_value(value: object, *, path: str) -> object:
    item = _fields(value, {"status", "value"}, path=path)
    if item["status"] != "available":
        raise PolicyImprovementSchemaError(f"{path} must be available.")
    return item["value"]


def _assert_close(actual: float, expected: float, *, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise PolicyImprovementSchemaError(
            f"{path} differs from per-instance recomputation: "
            f"reported={actual!r}, recomputed={expected!r}."
        )


def _ordered_record_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        _hex(value, path="ordered_record_digest", length=64)
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_regular_file(
    path: Path,
    *,
    seen_inodes: set[tuple[int, int]] | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Read one stable, singly linked, non-symlink dataset file."""

    try:
        before = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise PolicyImprovementSchemaError(
                f"Dataset evidence {path} is aliased or not a regular file."
            )
        inode = (before.st_dev, before.st_ino)
        if seen_inodes is not None:
            if inode in seen_inodes:
                raise PolicyImprovementSchemaError(
                    "Dataset evidence files alias one inode."
                )
            seen_inodes.add(inode)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise PolicyImprovementSchemaError(
                    f"Dataset evidence {path} changed before hashing."
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
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            f"Dataset evidence {path} could not be authenticated."
        ) from exc
    if _file_identity(opened) != _file_identity(after):
        raise PolicyImprovementSchemaError(
            f"Dataset evidence {path} changed while hashing."
        )
    return b"".join(chunks), {
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _regular_file_sha256(path: Path) -> str:
    return str(_stable_regular_file(path)[1]["sha256"])


def _strict_json_payload(payload: bytes, *, path: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PolicyImprovementSchemaError(
                    f"Duplicate JSON key {key!r} in {path}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PolicyImprovementSchemaError(
                    f"Dataset JSON {path} contains {constant!r}."
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementSchemaError(f"Dataset JSON {path} is invalid.") from exc


def _npy_int32(payload: bytes, *, path: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Decode the exact uncompressed C-order int32 NPY form emitted by the builder."""

    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise PolicyImprovementSchemaError(f"Dataset array {path} is not NPY.")
    major, minor = payload[6], payload[7]
    if (major, minor) == (1, 0):
        header_size = struct.unpack_from("<H", payload, 8)[0]
        header_start = 10
    elif (major, minor) in {(2, 0), (3, 0)}:
        if len(payload) < 12:
            raise PolicyImprovementSchemaError(f"Dataset array {path} is truncated.")
        header_size = struct.unpack_from("<I", payload, 8)[0]
        header_start = 12
    else:
        raise PolicyImprovementSchemaError(
            f"Dataset array {path} uses an unsupported NPY version."
        )
    header_end = header_start + header_size
    if header_end > len(payload):
        raise PolicyImprovementSchemaError(f"Dataset array {path} is truncated.")
    try:
        header = ast.literal_eval(payload[header_start:header_end].decode("latin1"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise PolicyImprovementSchemaError(
            f"Dataset array {path} has an invalid NPY header."
        ) from exc
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["descr"] not in {"<i4", "=i4"}
        or header["fortran_order"] is not False
        or not isinstance(header["shape"], tuple)
        or not header["shape"]
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in header["shape"]
        )
    ):
        raise PolicyImprovementSchemaError(
            f"Dataset array {path} is not a C-order int32 array."
        )
    shape = tuple(int(dimension) for dimension in header["shape"])
    count = math.prod(shape)
    raw = payload[header_end:]
    if len(raw) != count * 4:
        raise PolicyImprovementSchemaError(
            f"Dataset array {path} payload length differs from its shape."
        )
    values = tuple(item[0] for item in struct.iter_unpack("<i", raw))
    return shape, values


def _axis_permutations() -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    for bands in itertools.permutations((0, 1)):
        for first in itertools.permutations((0, 1)):
            for second in itertools.permutations((0, 1)):
                within = (first, second)
                values.append(
                    tuple(2 * band + row for band in bands for row in within[band])
                )
    if len(values) != 8 or len(set(values)) != 8:
        raise AssertionError("4x4 Sudoku axis permutation inventory differs.")
    return tuple(values)


def _spatial_index_permutations() -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    axes = _axis_permutations()
    for transpose in (False, True):
        for rows in axes:
            for columns in axes:
                result.append(
                    tuple(
                        (column * 4 + row) if transpose else (row * 4 + column)
                        for row in rows
                        for column in columns
                    )
                )
    if len(result) != 128 or len(set(result)) != 128:
        raise AssertionError("4x4 Sudoku spatial symmetry inventory differs.")
    return tuple(result)


_SPATIAL_INDEX_PERMUTATIONS = _spatial_index_permutations()


def _validate_sudoku_record(
    inputs: Sequence[int], labels: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    puzzle = tuple(inputs)
    solution = tuple(labels)
    if len(puzzle) != 16 or len(solution) != 16:
        raise PolicyImprovementSchemaError("Sudoku records must contain 16 cells.")
    if any(value < 1 or value > 5 for value in puzzle) or any(
        value < 2 or value > 5 for value in solution
    ):
        raise PolicyImprovementSchemaError(
            "Encoded Sudoku tokens are outside their domains."
        )
    if any(
        value != 1 and value != solution[index] for index, value in enumerate(puzzle)
    ):
        raise PolicyImprovementSchemaError(
            "Dataset puzzle givens disagree with their solution."
        )
    expected = {2, 3, 4, 5}
    if any(
        {solution[4 * axis + column] for column in range(4)} != expected
        or {solution[4 * row + axis] for row in range(4)} != expected
        for axis in range(4)
    ) or any(
        {
            solution[4 * row + column]
            for row in range(box_row, box_row + 2)
            for column in range(box_column, box_column + 2)
        }
        != expected
        for box_row in (0, 2)
        for box_column in (0, 2)
    ):
        raise PolicyImprovementSchemaError(
            "Dataset label is not a valid Sudoku solution."
        )
    return puzzle, solution


def _count_sudoku_solutions(encoded_puzzle: Sequence[int], *, limit: int = 2) -> int:
    return _count_sudoku_solutions_cached(tuple(encoded_puzzle), limit)


@lru_cache(maxsize=None)
def _count_sudoku_solutions_cached(encoded_puzzle: tuple[int, ...], limit: int) -> int:
    grid = [0 if value == 1 else value - 1 for value in encoded_puzzle]
    total = 0

    def candidates(index: int) -> list[int]:
        row, column = divmod(index, 4)
        used = {grid[4 * row + item] for item in range(4)}
        used.update(grid[4 * item + column] for item in range(4))
        box_row, box_column = 2 * (row // 2), 2 * (column // 2)
        used.update(
            grid[4 * box_r + box_c]
            for box_r in range(box_row, box_row + 2)
            for box_c in range(box_column, box_column + 2)
        )
        return [value for value in range(1, 5) if value not in used]

    def search() -> None:
        nonlocal total
        if total >= limit:
            return
        best_index: int | None = None
        best_values: list[int] | None = None
        for index, value in enumerate(grid):
            if value != 0:
                continue
            values = candidates(index)
            if not values:
                return
            if best_values is None or len(values) < len(best_values):
                best_index, best_values = index, values
        if best_index is None or best_values is None:
            total += 1
            return
        for value in best_values:
            grid[best_index] = value
            search()
            grid[best_index] = 0
            if total >= limit:
                return

    search()
    return total


def _canonical_sudoku_record_sha256(
    inputs: Sequence[int], labels: Sequence[int]
) -> str:
    return _canonical_sudoku_record_sha256_cached(tuple(inputs), tuple(labels))


@lru_cache(maxsize=None)
def _canonical_sudoku_record_sha256_cached(
    inputs: tuple[int, ...], labels: tuple[int, ...]
) -> str:
    puzzle, solution = _validate_sudoku_record(inputs, labels)
    best: bytes | None = None
    for indices in _SPATIAL_INDEX_PERMUTATIONS:
        mapping: dict[int, int] = {}
        next_token = 2
        candidate = bytearray()
        for source in (puzzle, solution):
            for index in indices:
                token = source[index]
                if token == 1:
                    candidate.append(1)
                    continue
                if token not in mapping:
                    mapping[token] = next_token
                    next_token += 1
                candidate.append(mapping[token])
        if next_token != 6:
            raise PolicyImprovementSchemaError(
                "A solved record must contain all four digits."
            )
        encoded = bytes(candidate)
        if best is None or encoded < best:
            best = encoded
    assert best is not None
    return hashlib.sha256(best).hexdigest()


def _sample_sha256(inputs: Sequence[int], labels: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"inputs": list(inputs), "solution": list(labels)})
    ).hexdigest()


def _input_sha256(inputs: Sequence[int]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(inputs))).hexdigest()


def _load_dataset_bindings(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    verify_test_content: bool = False,
    verify_content_splits: Collection[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Authenticate registered metadata and the splits authorized for opening."""

    known_splits = {"train", "validation", "test"}
    if verify_content_splits is None:
        content_splits = (
            known_splits if verify_test_content else {"train", "validation"}
        )
    else:
        content_splits = set(verify_content_splits)
        if (
            not content_splits
            or not content_splits <= known_splits
            or any(not isinstance(name, str) for name in verify_content_splits)
        ):
            raise PolicyImprovementSchemaError(
                "Dataset content-split authorization is empty or unsupported."
            )
        if verify_test_content:
            raise PolicyImprovementSchemaError(
                "Use either verify_test_content or verify_content_splits, not both."
            )

    supplied = Path(dataset_root)
    if not supplied.is_absolute():
        raise PolicyImprovementSchemaError("Dataset root must be absolute.")
    try:
        status = supplied.lstat()
        root = supplied.resolve(strict=True)
        owner_status = supplied.parent.lstat()
        owner = supplied.parent.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementSchemaError("Dataset root is unavailable.") from exc
    registered_parts = Path(str(protocol["dataset"]["root"])).parts
    if (
        not stat.S_ISDIR(status.st_mode)
        or supplied != root
        or tuple(supplied.parts[-len(registered_parts) :]) != registered_parts
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
        or not stat.S_ISDIR(owner_status.st_mode)
        or owner != supplied.parent
        or owner_status.st_uid != os.geteuid()
        or stat.S_IMODE(owner_status.st_mode) != 0o700
    ):
        raise PolicyImprovementSchemaError(
            "Dataset root or private owner has the wrong identity, mode, or registered path."
        )
    try:
        if {item.name for item in root.iterdir()} != _DATASET_TOP_LEVEL_NAMES:
            raise PolicyImprovementSchemaError(
                "Dataset root file inventory differs from the registered builder."
            )
        for name in ("manifests", "train", "validation", "test"):
            child = root / name
            child_status = child.lstat()
            if (
                not stat.S_ISDIR(child_status.st_mode)
                or child.resolve(strict=True) != child
            ):
                raise PolicyImprovementSchemaError(
                    f"Dataset directory {name!r} is aliased or has the wrong type."
                )
    except OSError as exc:
        raise PolicyImprovementSchemaError(
            "Dataset root inventory could not be authenticated."
        ) from exc

    seen_inodes: set[tuple[int, int]] = set()
    manifest_path = root / "MANIFEST.json"
    registered_manifest = str(
        _available_value(
            protocol["dataset"]["manifest_sha256"],
            path="protocol.dataset.manifest_sha256",
        )
    )
    manifest_payload, manifest_identity = _stable_regular_file(
        manifest_path, seen_inodes=seen_inodes
    )
    if manifest_identity["sha256"] != registered_manifest:
        raise PolicyImprovementSchemaError(
            "Dataset MANIFEST.json differs from the registered SHA-256."
        )
    top = _fields(
        _strict_json_payload(manifest_payload, path="MANIFEST.json"),
        {
            "build_schema_version",
            "builder",
            "canonicalization",
            "manifest_schema_version",
            "producer_source",
            "producer_source_paths",
            "splits",
        },
        path="dataset_manifest",
    )
    dataset = protocol["dataset"]
    if (
        top["build_schema_version"] != dataset["builder_schema_version"]
        or top["manifest_schema_version"] != dataset["manifest_schema_version"]
        or top["builder"] != "dataset.build_policy_improvement_4x4"
        or top["canonicalization"] != _DATASET_CANONICALIZATION
        or top["canonicalization"] != dataset["symmetry_canonicalization"]
        or top["producer_source_paths"] != list(_DATASET_PRODUCER_SOURCE_PATHS)
    ):
        raise PolicyImprovementSchemaError("Dataset manifest schema differs.")
    expected_producer = {
        key: _available_value(
            dataset["producer_source"][key],
            path=f"protocol.dataset.producer_source.{key}",
        )
        for key in (
            "git_commit",
            "launcher_sha256",
            "runtime_sha256",
            "source_manifest_sha256",
        )
    }
    if top["producer_source"] != expected_producer:
        raise PolicyImprovementSchemaError(
            "Dataset producer identity differs from the registered protocol."
        )
    identifiers_payload, _ = _stable_regular_file(
        root / "identifiers.json", seen_inodes=seen_inodes
    )
    if _strict_json_payload(identifiers_payload, path="identifiers.json") != [
        "<blank>"
    ]:
        raise PolicyImprovementSchemaError("Dataset identifier vocabulary differs.")
    expected_build_config = {
        "build_schema_version": dataset["builder_schema_version"],
        "builder": "dataset.build_policy_improvement_4x4",
        "domain": "sudoku_4x4",
        "producer_source": expected_producer,
        "require_symmetry_disjoint_splits": True,
        "require_unique_solution": True,
        "seed": dataset["splits"]["train"]["generation_seed"],
        "split_order": ["train", "validation", "test"],
        "splits": {
            name: {
                "count": dataset["splits"][name]["count"],
                "seed": dataset["splits"][name]["generation_seed"],
            }
            for name in ("train", "validation", "test")
        },
        "symmetry_canonicalization": str(
            dataset["symmetry_canonicalization"]["scheme"]
        ),
    }
    build_payload, _ = _stable_regular_file(
        root / "build_config.json", seen_inodes=seen_inodes
    )
    if (
        _strict_json_payload(build_payload, path="build_config.json")
        != expected_build_config
    ):
        raise PolicyImprovementSchemaError("Dataset build config differs.")

    manifests_root = root / "manifests"
    if {item.name for item in manifests_root.iterdir()} != {
        "train.json",
        "validation.json",
        "test.json",
    }:
        raise PolicyImprovementSchemaError("Dataset split-manifest inventory differs.")
    top_splits = _object(top["splits"], path="dataset_manifest.splits")
    if set(top_splits) != {"train", "validation", "test"}:
        raise PolicyImprovementSchemaError("Dataset split inventory differs.")
    bindings: dict[str, dict[str, Any]] = {}
    total_count = sum(
        int(dataset["splits"][name]["count"])
        for name in ("train", "validation", "test")
    )
    identifier_offset = 0
    seen_symmetry: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        registration = dataset["splits"][split]
        split_manifest_path = manifests_root / f"{split}.json"
        split_payload, split_identity = _stable_regular_file(
            split_manifest_path, seen_inodes=seen_inodes
        )
        manifest_digest = str(split_identity["sha256"])
        registered_split_digest = str(
            _available_value(
                registration["manifest_sha256"],
                path=f"protocol.dataset.splits.{split}.manifest_sha256",
            )
        )
        top_split = _fields(
            top_splits[split],
            {
                "count",
                "generation_seed",
                "manifest_sha256",
                "ordered_record_sha256",
                "ordered_symmetry_sha256",
                "puzzle_identifier_start",
            },
            path=f"dataset_manifest.splits.{split}",
        )
        if (
            manifest_digest != registered_split_digest
            or top_split["manifest_sha256"] != registered_split_digest
        ):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} manifest differs from its registered SHA-256."
            )
        split_manifest = _fields(
            _strict_json_payload(split_payload, path=f"manifests/{split}.json"),
            {
                "build_schema_version",
                "builder",
                "files",
                "generated_count",
                "generation_seed",
                "input_sha256s",
                "ordered_record_sha256",
                "ordered_symmetry_sha256",
                "record_sha256s",
                "symmetry_canonical_sha256s",
                "symmetry_canonicalization",
            },
            path=f"dataset_split_manifest.{split}",
        )
        count = int(registration["count"])
        record_hashes = split_manifest["record_sha256s"]
        input_hashes = split_manifest["input_sha256s"]
        symmetry_hashes = split_manifest["symmetry_canonical_sha256s"]
        if (
            split_manifest["generated_count"] != count
            or split_manifest["generation_seed"] != registration["generation_seed"]
            or split_manifest["build_schema_version"]
            != dataset["builder_schema_version"]
            or split_manifest["builder"] != "dataset.build_policy_improvement_4x4"
            or split_manifest["symmetry_canonicalization"]
            != dataset["symmetry_canonicalization"]["scheme"]
            or top_split["count"] != count
            or top_split["generation_seed"] != registration["generation_seed"]
            or top_split["puzzle_identifier_start"] != identifier_offset
            or not isinstance(record_hashes, list)
            or not isinstance(input_hashes, list)
            or not isinstance(symmetry_hashes, list)
            or len(record_hashes) != count
            or len(input_hashes) != count
            or len(symmetry_hashes) != count
        ):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} count or generation seed differs."
            )
        checked_records = [
            _hex(value, path=f"dataset.{split}.record_sha256s", length=64)
            for value in record_hashes
        ]
        checked_inputs = [
            _hex(value, path=f"dataset.{split}.input_sha256s", length=64)
            for value in input_hashes
        ]
        checked_symmetry = [
            _hex(value, path=f"dataset.{split}.symmetry_sha256s", length=64)
            for value in symmetry_hashes
        ]
        if len(checked_symmetry) != len(set(checked_symmetry)):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} contains a symmetry-equivalent duplicate."
            )
        for digest in checked_symmetry:
            prior = seen_symmetry.setdefault(digest, split)
            if prior != split:
                raise PolicyImprovementSchemaError(
                    f"Dataset splits {prior!r} and {split!r} overlap under symmetry."
                )
        ordered = _ordered_record_digest(checked_records)
        ordered_symmetry = _ordered_record_digest(checked_symmetry)
        registered_ordered = str(
            _available_value(
                registration["ordered_record_sha256"],
                path=f"protocol.dataset.splits.{split}.ordered_record_sha256",
            )
        )
        if (
            ordered != registered_ordered
            or split_manifest["ordered_record_sha256"] != ordered
            or top_split["ordered_record_sha256"] != ordered
            or split_manifest["ordered_symmetry_sha256"] != ordered_symmetry
            or top_split["ordered_symmetry_sha256"] != ordered_symmetry
        ):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} ordered-record identity differs."
            )

        expected_relative_names = {
            f"{split}/{name}" for name in _DATASET_SPLIT_FILE_NAMES
        }
        files = _object(
            split_manifest["files"], path=f"dataset_split_manifest.{split}.files"
        )
        if set(files) != expected_relative_names:
            raise PolicyImprovementSchemaError(
                f"Dataset {split} registered file inventory differs."
            )
        for relative in sorted(expected_relative_names):
            entry = _fields(
                files[relative],
                {"bytes", "sha256"},
                path=f"dataset_split_manifest.{split}.files.{relative}",
            )
            if (
                isinstance(entry["bytes"], bool)
                or not isinstance(entry["bytes"], int)
                or entry["bytes"] < 0
            ):
                raise PolicyImprovementSchemaError(
                    f"Dataset file {relative!r} has invalid registered size."
                )
            _hex(entry["sha256"], path=f"dataset_file.{relative}", length=64)
        bindings[split] = {
            "record_sha256s": tuple(checked_records),
            "input_sha256s": tuple(checked_inputs),
            "puzzle_ids": tuple(f"{split}-{index:06d}" for index in range(count)),
        }
        if split not in content_splits:
            identifier_offset += count
            continue

        split_root = root / split
        if {item.name for item in split_root.iterdir()} != _DATASET_SPLIT_FILE_NAMES:
            raise PolicyImprovementSchemaError(
                f"Dataset {split} materialized file inventory differs."
            )
        materialized: dict[str, bytes] = {}
        for name in sorted(_DATASET_SPLIT_FILE_NAMES):
            relative = f"{split}/{name}"
            payload, identity = _stable_regular_file(
                split_root / name, seen_inodes=seen_inodes
            )
            entry = _fields(
                files[relative],
                {"bytes", "sha256"},
                path=f"dataset_split_manifest.{split}.files.{relative}",
            )
            if (
                isinstance(entry["bytes"], bool)
                or not isinstance(entry["bytes"], int)
                or entry["bytes"] < 0
                or _hex(entry["sha256"], path=f"dataset_file.{relative}", length=64)
                != identity["sha256"]
                or entry["bytes"] != identity["bytes"]
            ):
                raise PolicyImprovementSchemaError(
                    f"Dataset file {relative!r} differs from its manifest."
                )
            materialized[name] = payload

        input_shape, input_values = _npy_int32(
            materialized["all__inputs.npy"], path=f"{split}/all__inputs.npy"
        )
        label_shape, label_values = _npy_int32(
            materialized["all__labels.npy"], path=f"{split}/all__labels.npy"
        )
        if input_shape != (count, 16) or label_shape != (count, 16):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} input/label array shape differs."
            )
        identifier_shape, identifiers = _npy_int32(
            materialized["all__puzzle_identifiers.npy"],
            path=f"{split}/all__puzzle_identifiers.npy",
        )
        if identifier_shape != (count,) or identifiers != tuple(
            range(identifier_offset, identifier_offset + count)
        ):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} puzzle identifiers are not registered and disjoint."
            )
        expected_indices = tuple(range(count + 1))
        for name in ("all__puzzle_indices.npy", "all__group_indices.npy"):
            shape, values = _npy_int32(materialized[name], path=f"{split}/{name}")
            if shape != (count + 1,) or values != expected_indices:
                raise PolicyImprovementSchemaError(
                    f"Dataset {split} puzzle/group indices differ."
                )
        expected_metadata = {
            "blank_identifier_id": 0,
            "ignore_label_id": 0,
            "mean_puzzle_examples": 1,
            "num_puzzle_identifiers": total_count,
            "pad_id": 0,
            "seq_len": 16,
            "sets": ["all"],
            "total_groups": count,
            "total_puzzles": count,
            "vocab_size": 6,
        }
        if (
            _strict_json_payload(
                materialized["dataset.json"], path=f"{split}/dataset.json"
            )
            != expected_metadata
        ):
            raise PolicyImprovementSchemaError(f"Dataset {split} metadata differs.")
        records = _strict_json_payload(
            materialized["records.json"], path=f"{split}/records.json"
        )
        if not isinstance(records, list) or len(records) != count:
            raise PolicyImprovementSchemaError(
                f"Dataset {split} record inventory differs."
            )
        computed_records: list[str] = []
        computed_inputs: list[str] = []
        computed_symmetry: list[str] = []
        for index in range(count):
            inputs = input_values[16 * index : 16 * (index + 1)]
            labels = label_values[16 * index : 16 * (index + 1)]
            puzzle, solution = _validate_sudoku_record(inputs, labels)
            if _count_sudoku_solutions(puzzle) != 1:
                raise PolicyImprovementSchemaError(
                    f"Dataset {split} record {index} is not uniquely solvable."
                )
            record_digest = _sample_sha256(puzzle, solution)
            input_digest = _input_sha256(puzzle)
            symmetry_digest = _canonical_sudoku_record_sha256(puzzle, solution)
            expected_record = {
                "index": index,
                "record_sha256": record_digest,
                "symmetry_canonical_sha256": symmetry_digest,
            }
            if records[index] != expected_record:
                raise PolicyImprovementSchemaError(
                    f"Dataset {split} record {index} identity differs from its arrays."
                )
            computed_records.append(record_digest)
            computed_inputs.append(input_digest)
            computed_symmetry.append(symmetry_digest)
        if (
            computed_records != checked_records
            or computed_inputs != checked_inputs
            or computed_symmetry != checked_symmetry
        ):
            raise PolicyImprovementSchemaError(
                f"Dataset {split} manifest identities differ from its arrays."
            )
        identifier_offset += count
    return bindings


def _audit_evaluation(
    result: Mapping[str, object],
    snapshot: Mapping[str, object],
    evaluation: Mapping[str, object],
    per_instance_documents: Mapping[str, object],
    *,
    registered_population_count: int,
    registered_ordered_record_sha256: str,
    registered_record_sha256s: Sequence[str],
    registered_input_sha256s: Sequence[str],
    registered_puzzle_ids: Sequence[str],
    registered_original_indices: Sequence[int] | None = None,
    consumed_per_instance_sha256s: set[str],
) -> tuple[str, ...]:
    digest = str(
        _available_value(
            evaluation["per_instance_artifact_sha256"],
            path="evaluation.per_instance_artifact_sha256",
        )
    )
    if digest not in per_instance_documents:
        raise PolicyImprovementSchemaError(f"Missing per-instance artifact {digest}.")
    consumed_per_instance_sha256s.add(digest)
    document = validate_per_instance_document(per_instance_documents[digest])
    if hashlib.sha256(canonical_json_bytes(document)).hexdigest() != digest:
        raise PolicyImprovementSchemaError(
            "Per-instance artifact digest does not match its canonical bytes."
        )
    expected_identity = {
        "protocol_id": result["protocol_id"],
        "run_id": result["run_id"],
        "phase": result["phase"],
        "tier": result["tier"],
        "seed": result["seed"],
        "evaluation_split": result["evaluation_split"],
        "method_id": result["method_id"],
        "snapshot_kind": snapshot["snapshot_kind"],
        "evaluation_id": evaluation["evaluation_id"],
        "policy_variant": evaluation["policy_variant"],
        "evaluation_pool_sha256": _available_value(
            evaluation["evaluation_pool_sha256"],
            path="evaluation.evaluation_pool_sha256",
        ),
    }
    for field, expected in expected_identity.items():
        if document[field] != expected:
            raise PolicyImprovementSchemaError(
                f"Per-instance {field} differs from its result."
            )
    if document["schema_name"] == "policy_improvement_instances_v2":
        if (
            document["evaluation_population_id"] != result["evaluation_population_id"]
            or document["evaluation_population_binding_sha256"]
            != result["evaluation_population_binding_sha256"]
        ):
            raise PolicyImprovementSchemaError(
                "Per-instance population identity differs from its result."
            )
    records = document["records"]
    if len(records) != registered_population_count:
        raise PolicyImprovementSchemaError(
            "Per-instance denominator differs from the registered population."
        )
    if [int(record["registered_index"]) for record in records] != list(
        range(registered_population_count)
    ):
        raise PolicyImprovementSchemaError(
            "Per-instance rows are not the exact registered ordered population."
        )
    for index, record in enumerate(records):
        if (
            record["registered_record_sha256"] != registered_record_sha256s[index]
            or record["puzzle_sha256"] != registered_input_sha256s[index]
            or record["puzzle_id"] != registered_puzzle_ids[index]
        ):
            raise PolicyImprovementSchemaError(
                "Per-instance record, input, or canonical puzzle identity differs "
                "from the authenticated split manifest."
            )
        if registered_original_indices is not None and (
            record["population_position"] != index
            or record["original_dataset_index"] != registered_original_indices[index]
            or record["registered_input_sha256"] != registered_input_sha256s[index]
        ):
            raise PolicyImprovementSchemaError(
                "Per-instance v2 population position or source index differs."
            )
    ordered_puzzle_digest = _ordered_record_digest(
        [str(record["registered_record_sha256"]) for record in records]
    )
    if (
        result["identities"]["evaluation_ordered_records_sha256"]
        != ordered_puzzle_digest
        or ordered_puzzle_digest != registered_ordered_record_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Result evaluation order differs from the per-instance puzzle order."
        )
    denominator = len(records)
    solved = sum(1 for record in records if record["solved"])
    counts = {
        reason: sum(1 for record in records if record["terminal_reason"] == reason)
        for reason in ("budget", "solved", "stop")
    }
    returns = [float(record["discounted_return"]) for record in records]
    solved_edits = [
        int(record["edits_to_solve"]) for record in records if record["solved"]
    ]
    calibration = (
        sum(
            abs(float(record["value_prediction"]) - float(record["realized_return"]))
            for record in records
        )
        / denominator
    )
    primary = _object(evaluation["primary"], path="evaluation.primary")
    secondary = _object(evaluation["secondary"], path="evaluation.secondary")
    reported_denominator = int(
        _available_value(primary["denominator"], path="evaluation.primary.denominator")
    )
    reported_solved = int(
        _available_value(
            primary["solved_count"], path="evaluation.primary.solved_count"
        )
    )
    if reported_denominator != denominator or reported_solved != solved:
        raise PolicyImprovementSchemaError(
            "Per-instance counts differ from the reported primary endpoint."
        )
    _assert_close(
        float(
            _available_value(
                primary["solve_rate"], path="evaluation.primary.solve_rate"
            )
        ),
        solved / denominator,
        path="evaluation.primary.solve_rate",
    )
    reported_counts = _available_value(
        secondary["terminal_reason_counts"],
        path="evaluation.secondary.terminal_reason_counts",
    )
    if reported_counts != counts:
        raise PolicyImprovementSchemaError(
            "Terminal-reason aggregate differs from per-instance records."
        )
    _assert_close(
        float(
            _available_value(
                secondary["discounted_return_mean"],
                path="evaluation.secondary.discounted_return_mean",
            )
        ),
        sum(returns) / denominator,
        path="evaluation.secondary.discounted_return_mean",
    )
    if solved_edits:
        _assert_close(
            float(
                _available_value(
                    secondary["edits_to_solve_mean"],
                    path="evaluation.secondary.edits_to_solve_mean",
                )
            ),
            sum(solved_edits) / len(solved_edits),
            path="evaluation.secondary.edits_to_solve_mean",
        )
    _assert_close(
        float(
            _available_value(
                secondary["value_calibration"],
                path="evaluation.secondary.value_calibration",
            )
        ),
        calibration,
        path="evaluation.secondary.value_calibration",
    )
    aggregate_digest = hashlib.sha256(
        canonical_json_bytes(
            {"primary": evaluation["primary"], "secondary": evaluation["secondary"]}
        )
    ).hexdigest()
    if (
        _available_value(
            evaluation["evaluation_artifact_sha256"],
            path="evaluation.evaluation_artifact_sha256",
        )
        != aggregate_digest
    ):
        raise PolicyImprovementSchemaError(
            "Evaluation artifact digest does not bind the reported aggregate."
        )
    return tuple(
        f"{record['registered_index']}:{record['registered_record_sha256']}"
        for record in records
    )


def _authorized_role(
    authorization: Mapping[str, object], role_name: str
) -> Mapping[str, object]:
    for role in authorization["roles"]:
        if role["role"] == role_name:
            return role
    raise AssertionError(f"Validated authorization omitted {role_name!r}.")


def _validate_execution_identity(
    value: object,
    authorization: Mapping[str, object],
    *,
    role_name: str,
) -> dict[str, str]:
    identity = _fields(
        value,
        {
            "runtime_sha256",
            "runtime_profile_sha256",
            "source_git_commit",
            "launcher_sha256",
            "producer_git_commit",
            "producer_source_manifest_sha256",
        },
        path="execution_identity",
    )
    for field in (
        "runtime_sha256",
        "runtime_profile_sha256",
        "launcher_sha256",
        "producer_source_manifest_sha256",
    ):
        _hex(identity[field], path=f"execution_identity.{field}", length=64)
    _hex(
        identity["source_git_commit"],
        path="execution_identity.source_git_commit",
        length=40,
    )
    _hex(
        identity["producer_git_commit"],
        path="execution_identity.producer_git_commit",
        length=40,
    )
    role = _authorized_role(authorization, role_name)
    expected = {
        "runtime_sha256": role["runtime_sha256"],
        "runtime_profile_sha256": role["runtime_profile_sha256"],
        "source_git_commit": role["source_git_commit"],
        "launcher_sha256": authorization["launcher_sha256"],
        "producer_git_commit": authorization["producer_git_commit"],
        "producer_source_manifest_sha256": authorization[
            "producer_source_manifest_sha256"
        ],
    }
    if dict(identity) != expected:
        raise PolicyImprovementSchemaError(
            f"Execution identity is not authorized for {role_name!r}."
        )
    return {field: str(identity[field]) for field in sorted(identity)}


def _canonical_result_set_sha256(results: Sequence[Mapping[str, object]]) -> str:
    ordered = sorted(results, key=lambda result: str(result["run_id"]))
    return hashlib.sha256(canonical_json_bytes(ordered)).hexdigest()


def _canonical_per_instance_set_sha256(
    documents: Mapping[str, object],
) -> str:
    inventory = [
        {"sha256": digest, "document": documents[digest]}
        for digest in sorted(documents)
    ]
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def _primary_interaction_rate(result: Mapping[str, object]) -> Fraction:
    if result["status"] != "complete":
        raise PolicyImprovementSchemaError(
            "Registered selection requires every eligible validation run to complete."
        )
    snapshots = [
        snapshot
        for snapshot in result["evaluation_snapshots"]
        if snapshot["snapshot_kind"] == "interaction_matched"
    ]
    if len(snapshots) != 1 or snapshots[0]["status"] != "available":
        raise PolicyImprovementSchemaError(
            "Registered selection requires one interaction-matched snapshot."
        )
    primary_variant = result["primary_policy_variant"]
    evaluations = [
        evaluation
        for evaluation in snapshots[0]["policy_evaluations"]
        if evaluation["policy_variant"] == primary_variant
    ]
    if len(evaluations) != 1:
        raise PolicyImprovementSchemaError(
            "Registered selection requires one primary-policy evaluation."
        )
    primary = evaluations[0]["primary"]
    solved = _integer(
        _available_value(primary["solved_count"], path="selection.solved_count"),
        path="selection.solved_count",
    )
    denominator = _integer(
        _available_value(primary["denominator"], path="selection.denominator"),
        path="selection.denominator",
        minimum=1,
    )
    return Fraction(solved, denominator)


def derive_registered_selection(
    phase: str, results: Sequence[object]
) -> dict[str, object]:
    """Derive one deterministic validation selection from audited result rows."""

    checked = [validated_result_payload(result)[1] for result in results]
    if phase == "stage1_screen":
        eligible = [
            result
            for result in checked
            if result["method_id"] in SCREEN_SELECTION_METHOD_ORDER
        ]
        grouped: dict[tuple[str, int, int], list[Fraction]] = {}
        for result in eligible:
            key = (
                str(result["method_id"]),
                int(result["n"]),
                int(result["K"]),
            )
            grouped.setdefault(key, []).append(_primary_interaction_rate(result))
        if not grouped or any(len(values) != 3 for values in grouped.values()):
            raise PolicyImprovementSchemaError(
                "Screen selection requires three paired seeds for every exact tuple."
            )
        means = {
            key: sum(values, Fraction(0, 1)) / len(values)
            for key, values in grouped.items()
        }
        best_score = max(means.values())
        method_rank = {
            method: index for index, method in enumerate(SCREEN_SELECTION_METHOD_ORDER)
        }
        method, n, K = min(
            (key for key, score in means.items() if score == best_score),
            key=lambda key: (method_rank[key[0]], key[1], key[2]),
        )
        return {"method_id": method, "n": n, "K": K}
    if phase == "stage1_alpha":
        grouped_alpha: dict[Fraction, list[Fraction]] = {}
        common: set[tuple[str, int, int]] = set()
        for result in checked:
            alpha = Fraction(str(result["alpha"]))
            grouped_alpha.setdefault(alpha, []).append(
                _primary_interaction_rate(result)
            )
            common.add((str(result["method_id"]), int(result["n"]), int(result["K"])))
        if (
            len(common) != 1
            or set(grouped_alpha) != {Fraction(1, 20), Fraction(1, 10), Fraction(1, 5)}
            or any(len(values) != 3 for values in grouped_alpha.values())
        ):
            raise PolicyImprovementSchemaError(
                "Alpha selection requires the registered three alphas and three seeds."
            )
        means = {
            alpha: sum(values, Fraction(0, 1)) / len(values)
            for alpha, values in grouped_alpha.items()
        }
        best_score = max(means.values())
        alpha = min(value for value, score in means.items() if score == best_score)
        method, n, K = next(iter(common))
        return {
            "method_id": method,
            "n": n,
            "K": K,
            "alpha": float(alpha),
        }
    raise PolicyImprovementSchemaError(
        "Only the registered validation selection phases can be derived."
    )


def audit_result_set(
    protocol_value: object,
    registry_value: object,
    results: Sequence[object],
    per_instance_documents: Mapping[str, object],
    *,
    phases: Sequence[str],
    amendment_history: Sequence[object] = (),
    base_configs: Mapping[str, Mapping[str, object]] | None = None,
    project_root: str | Path,
    dataset_root: str | Path,
    evidence_root: str | Path,
    runtime_authorization: object,
    historical_runtime_authorizations: Mapping[str, Mapping[str, object]] | None = None,
    audit_execution_identity: object,
    checkpoint_validator: Callable[[Mapping[str, object]], Mapping[str, object]],
    execution_role_name: str = "policy-improvement-audit",
    amendment_evidence: Mapping[str, Mapping[str, object]] | None = None,
    test_open_record: object | None = None,
    test_open_owner_root: str | Path | None = None,
    test_open_sha256: str | None = None,
    _verify_amendment_evidence: bool = True,
    _dataset_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    _producer_source_authenticator: (
        Callable[[Mapping[str, object]], None] | None
    ) = None,
) -> dict[str, Any]:
    """Audit an exact phase inventory, including failures and paired records."""

    protocol = validate_protocol(protocol_value)
    history = validate_amendment_history(amendment_history, protocol=protocol)
    authorization = validate_runtime_authorization(runtime_authorization)
    if (
        protocol.get("schema_name") == "policy_improvement_protocol_v2"
        and [
            authorization.get("schema_name"),
            authorization.get("schema_version"),
        ]
        != protocol["document_schemas"]["runtime_authorization"]
    ):
        raise PolicyImprovementSchemaError(
            "Runtime authorization schema differs from the protocol-v2 registration."
        )
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    if authorization["protocol_sha256"] != protocol_digest:
        raise PolicyImprovementSchemaError(
            "Runtime authorization names a different protocol."
        )
    authorization_digest = runtime_authorization_sha256(authorization)
    producer_source_authenticator = (
        _producer_source_authenticator
        if _producer_source_authenticator is not None
        else lambda item: _authenticate_authorization_producer_source(
            item,
            project_root=project_root,
        )
    )
    runtime_authorizations = _validated_runtime_authorization_map(
        current_authorization=authorization,
        historical_runtime_authorizations=historical_runtime_authorizations,
        protocol_sha256=protocol_digest,
        producer_source_authenticator=producer_source_authenticator,
    )
    if (
        len(history) >= 2
        and history[1]["runtime_authorization_sha256"] != authorization_digest
    ):
        raise PolicyImprovementSchemaError(
            "Compute freeze binds a different runtime/source authorization."
        )
    checked_execution_identity = _validate_execution_identity(
        audit_execution_identity,
        authorization,
        role_name=execution_role_name,
    )
    if execution_role_name not in {
        "policy-improvement-audit",
        "policy-improvement-analysis",
    }:
        raise PolicyImprovementSchemaError("Unsupported evidence-audit execution role.")
    registered_population_document: Mapping[str, Any] | None = None
    registered_populations: Mapping[str, Any] | None = None
    if protocol.get("schema_name") == "policy_improvement_protocol_v2":
        from scripts.policy_improvement_populations import load_registered_populations

        registered_population_document = load_registered_populations(
            protocol,
            project_root,
        )
        registered_populations = registered_population_document["populations"]
    registry = validate_registry_document(
        registry_value,
        protocol,
        history,
        base_configs=base_configs,
        populations_value=registered_population_document,
    )
    if (
        not phases
        or len(phases) != len(set(phases))
        or any(phase not in PHASE_CONTRACTS for phase in phases)
    ):
        raise PolicyImprovementSchemaError("Audit phases are invalid or duplicated.")
    required_prefixes = {PHASE_AMENDMENT_PREFIX_LENGTH[phase] for phase in phases}
    if required_prefixes != {len(history)}:
        raise PolicyImprovementSchemaError(
            "Audit phases do not use their exact registered amendment-history prefix."
        )
    expected_rows = [row for row in registry["rows"] if row["phase"] in set(phases)]
    if any(row["row_kind"] != "concrete" for row in expected_rows):
        raise PolicyImprovementSchemaError(
            "Selection-dependent rows cannot be audited before materialization."
        )
    registry_digest = registry_sha256(registry)
    history_digest = amendment_history_sha256(history)

    uses_test = any(PHASE_CONTRACTS[phase][1] == "test" for phase in phases)
    if not uses_test and any(
        value is not None
        for value in (test_open_record, test_open_owner_root, test_open_sha256)
    ):
        raise PolicyImprovementSchemaError(
            "Validation-only audits must not carry a test-open record."
        )

    pre_test_content_splits = {
        "train",
        *(
            ("validation",)
            if uses_test
            else tuple(
                str(row["evaluation_split"])
                for row in expected_rows
                if str(row["evaluation_split"]) != "test"
            )
        ),
    }
    dataset_bindings = (
        _load_dataset_bindings(
            protocol,
            dataset_root,
            verify_content_splits=pre_test_content_splits,
        )
        if _dataset_bindings is None
        else _dataset_bindings
    )
    if _verify_amendment_evidence and len(history) > 1:
        evidence_amendments = (
            (1, "stage0_smoke", 0),
            (2, "stage1_screen", 2),
            (3, "stage1_alpha", 3),
        )
        if amendment_evidence is None:
            raise PolicyImprovementSchemaError(
                "Amendment history requires prior result evidence for recomputation."
            )
        for amendment_index, phase, prior_prefix_length in evidence_amendments:
            if amendment_index >= len(history):
                break
            if phase not in amendment_evidence:
                raise PolicyImprovementSchemaError(
                    f"Missing recomputation evidence for {phase}."
                )
            supplied = _fields(
                amendment_evidence[phase],
                {"results", "per_instance_documents"},
                path=f"amendment_evidence.{phase}",
            )
            if not isinstance(supplied["results"], Sequence) or isinstance(
                supplied["results"], (str, bytes)
            ):
                raise PolicyImprovementSchemaError(
                    f"amendment_evidence.{phase}.results must be a sequence."
                )
            supplied_documents = _object(
                supplied["per_instance_documents"],
                path=f"amendment_evidence.{phase}.per_instance_documents",
            )
            prior_history = history[:prior_prefix_length]
            prior_registry = generate_registry(
                protocol,
                prior_history,
                base_configs=base_configs,
                populations_value=registered_population_document,
            )
            prior_report = audit_result_set(
                protocol,
                prior_registry,
                list(supplied["results"]),
                supplied_documents,
                phases=[phase],
                amendment_history=prior_history,
                base_configs=base_configs,
                project_root=project_root,
                dataset_root=dataset_root,
                evidence_root=evidence_root,
                runtime_authorization=authorization,
                historical_runtime_authorizations=runtime_authorizations,
                audit_execution_identity=checked_execution_identity,
                checkpoint_validator=checkpoint_validator,
                execution_role_name=execution_role_name,
                test_open_record=None,
                test_open_owner_root=None,
                test_open_sha256=None,
                _verify_amendment_evidence=False,
                _dataset_bindings=dataset_bindings,
                _producer_source_authenticator=producer_source_authenticator,
            )
            evidence_report = dict(prior_report)
            if execution_role_name != "policy-improvement-audit":
                audit_role = _authorized_role(authorization, "policy-improvement-audit")
                evidence_report.update(
                    {
                        "execution_role": "policy-improvement-audit",
                        "execution_runtime_sha256": audit_role["runtime_sha256"],
                        "execution_source_git_commit": audit_role["source_git_commit"],
                        "execution_runtime_profile_sha256": audit_role[
                            "runtime_profile_sha256"
                        ],
                    }
                )
            evidence = history[amendment_index]["evidence"]
            expected_evidence = {
                "phase": phase,
                "audit_report_sha256": hashlib.sha256(
                    canonical_json_bytes(evidence_report)
                ).hexdigest(),
                "result_set_sha256": prior_report["result_set_sha256"],
                "per_instance_set_sha256": prior_report["per_instance_set_sha256"],
                "expected_rows": prior_report["expected_rows"],
                "complete_rows": prior_report["complete_rows"],
                "failed_rows": prior_report["failed_rows"],
            }
            if evidence != expected_evidence:
                raise PolicyImprovementSchemaError(
                    f"{phase} amendment evidence differs from independent reaudit."
                )
            if phase in {"stage1_screen", "stage1_alpha"}:
                derived = derive_registered_selection(phase, list(supplied["results"]))
                if history[amendment_index]["selected_exact"] != derived:
                    raise PolicyImprovementSchemaError(
                        f"{phase} frozen selection differs from the registered "
                        "endpoint and tie-break rule."
                    )
    if uses_test:
        if _dataset_bindings is not None:
            raise PolicyImprovementSchemaError(
                "Test audits must independently authenticate materialized test data."
            )
        if len(history) != 4 or test_open_record is None:
            raise PolicyImprovementSchemaError(
                "Test results require frozen selection and a test-open record."
            )
        test_registration = protocol["dataset"]["splits"]["test"]["manifest_sha256"]
        if test_registration.get("status") != "available":
            raise PolicyImprovementSchemaError(
                "Test data cannot be opened before its manifest is frozen."
            )
        if test_open_owner_root is None or test_open_sha256 is None:
            raise PolicyImprovementSchemaError(
                "Test results require the immutable TEST_OPEN.json owner and digest."
            )
        if Path(test_open_owner_root) != Path(evidence_root):
            raise PolicyImprovementSchemaError(
                "TEST_OPEN.json must use the registered evidence owner root."
            )
        if not isinstance(test_open_record, Mapping):
            raise PolicyImprovementSchemaError("Test-open record must be an object.")
        authenticated_test_open = authenticate_test_open(
            owner_root=evidence_root,
            protocol=protocol,
            registry=registry,
            amendment_history=history,
            runtime_authorization=authorization,
            opened_at_utc=str(test_open_record.get("opened_at_utc")),
            expected_sha256=test_open_sha256,
            base_configs=base_configs,
        )
        if canonical_json_bytes(test_open_record) != canonical_json_bytes(
            authenticated_test_open["record"]
        ):
            raise PolicyImprovementSchemaError(
                "Supplied test-open record differs from immutable TEST_OPEN.json."
            )
        dataset_bindings = _load_dataset_bindings(
            protocol,
            dataset_root,
            verify_test_content=True,
        )
    checked_result_documents = [validate_result(result) for result in results]
    by_run_id = {str(result["run_id"]): result for result in checked_result_documents}
    if len(by_run_id) != len(checked_result_documents):
        raise PolicyImprovementSchemaError("Result run IDs must be unique.")
    expected_ids = {str(row["run_id"]) for row in expected_rows}
    if set(by_run_id) != expected_ids:
        raise PolicyImprovementSchemaError(
            f"Result inventory differs: missing={sorted(expected_ids - set(by_run_id))}, "
            f"extra={sorted(set(by_run_id) - expected_ids)}."
        )

    rows_by_id = {str(row["run_id"]): row for row in expected_rows}
    paired_puzzle_orders: dict[tuple[int, str], tuple[str, ...]] = {}
    paired_initializations: dict[tuple[str, int], str] = {}
    checkpoint_owners: dict[str, str] = {}
    complete_count = 0
    failed_count = 0
    generation_manifest_sha256s: list[str] = []
    historical_failed_attempt_manifest_sha256s: list[str] = []
    semantic_validation_sha256s: list[str] = []
    consumed_per_instance_sha256s: set[str] = set()
    for run_id, result_document in by_run_id.items():
        row = rows_by_id[run_id]
        if result_document.get("schema_name") == "policy_improvement_result_v2":
            from scripts.policy_improvement_v2_schema import (
                bind_v2_result_to_registration,
            )

            if registered_population_document is None:
                raise PolicyImprovementSchemaError(
                    "Protocol v2 result lacks its authenticated population document."
                )
            bind_v2_result_to_registration(
                result_document,
                row,
                protocol,
                registry,
                registered_population_document,
            )
        _, result = validated_result_payload(result_document)
        row_digest = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        exact_bindings = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_digest,
            "amendment_history_sha256": history_digest,
            "registry_row_sha256": row_digest,
            "phase": row["phase"],
            "tier": row["tier"],
            "seed": row["seed"],
            "evaluation_split": row["evaluation_split"],
            "method_id": row["method_id"],
            "base_method_id": row["base_method_id"],
            "n": row["n"],
            "K": row["K"],
            "alpha": row["alpha"],
            "ablation_variant": row["ablation_variant"],
            "applied_config_override": (
                row["config_override"] if row["config_override"] is not None else {}
            ),
        }
        for field, expected in exact_bindings.items():
            if result[field] != expected:
                raise PolicyImprovementSchemaError(
                    f"Result {run_id} {field} differs from its registry row."
                )
        if (
            uses_test
            and _available_value(
                result["identities"]["test_open_sha256"],
                path="result.identities.test_open_sha256",
            )
            != test_open_sha256
        ):
            raise PolicyImprovementSchemaError(
                f"Result {run_id} does not bind the authenticated test opening."
            )
        method_registration = next(
            method
            for method in protocol["methods"]
            if method["id"] == result["base_method_id"]
        )
        if (
            result["identities"]["method_config_sha256"]
            != method_registration["config_sha256"]
        ):
            raise PolicyImprovementSchemaError(
                "Result method config digest differs from the protocol tuple."
            )
        result_authorization_digest = _hex(
            result["identities"]["runtime_authorization_sha256"],
            path=f"result[{run_id}].identities.runtime_authorization_sha256",
            length=64,
        )
        result_authorization = runtime_authorizations.get(result_authorization_digest)
        if result_authorization is None:
            raise PolicyImprovementSchemaError(
                f"Result {run_id} lacks its frozen runtime authorization."
            )
        training_role = _authorized_role(
            result_authorization, "policy-improvement-training"
        )
        expected_runtime_bindings = {
            "runtime_authorization_sha256": result_authorization_digest,
            "producer_git_commit": result_authorization["producer_git_commit"],
            "producer_manifest_sha256": result_authorization[
                "producer_source_manifest_sha256"
            ],
            "launcher_sha256": result_authorization["launcher_sha256"],
            "training_source_git_commit": training_role["source_git_commit"],
            "training_runtime_sha256": training_role["runtime_sha256"],
            "training_runtime_profile_sha256": training_role["runtime_profile_sha256"],
            "training_selected_source_manifest_sha256": training_role[
                "selected_source_manifest_sha256"
            ],
        }
        for field, expected in expected_runtime_bindings.items():
            if result["identities"][field] != expected:
                raise PolicyImprovementSchemaError(
                    f"Result {run_id} identity {field} is not externally authorized."
                )
        if (
            result["identities"]["effective_config_sha256"]
            != row["expected_effective_config_sha256"]
        ):
            raise PolicyImprovementSchemaError(
                "Result effective config differs from registry recomputation."
            )
        dataset = protocol["dataset"]
        dataset_manifest = dataset["manifest_sha256"]
        train_order = dataset["splits"]["train"]["ordered_record_sha256"]
        if any(
            identity["status"] != "available"
            for identity in (dataset_manifest, train_order)
        ):
            raise PolicyImprovementSchemaError(
                "Result audit requires frozen dataset and ordered-record identities."
            )
        if (
            result["identities"]["dataset_manifest_sha256"] != dataset_manifest["value"]
            or result["identities"]["train_ordered_records_sha256"]
            != train_order["value"]
        ):
            raise PolicyImprovementSchemaError(
                "Result dataset identities differ from the frozen protocol."
            )
        tier = str(row["tier"])
        interaction_tier = "confirmatory" if tier == "ablation" else tier
        expected_interactions = int(
            protocol["budgets"][interaction_tier]["environment_interactions"]
        )
        if result["status"] == "failed":
            failed_identity = authenticate_failed_attempt(
                evidence_root=evidence_root,
                result=result_document,
                protocol_sha256=protocol_digest,
                registry_row_sha256=row_digest,
                runtime_authorization_sha256=result_authorization_digest,
                expected_environment_interactions=expected_interactions,
            )
            generation_manifest_sha256s.append(
                str(failed_identity["generation_manifest_sha256"])
            )
            failed_count += 1
            continue
        generation_identity = authenticate_complete_generation(
            evidence_root=evidence_root,
            result_path=(
                Path(evidence_root)
                / "runs"
                / run_id
                / "segments"
                / f"env_{expected_interactions:09d}"
                / "result.json"
            ),
            result=result_document,
            protocol_sha256=protocol_digest,
            registry_sha256=registry_digest,
            registry_row_sha256=row_digest,
            expected_environment_interactions=expected_interactions,
            per_instance_documents=per_instance_documents,
            checkpoint_validator=checkpoint_validator,
            protocol=protocol,
            registry_row=row,
            project_root=project_root,
            dataset_root=dataset_root,
            runtime_authorization=result_authorization,
            amendment_history_sha256=history_digest,
            authenticated_test_open_sha256=(
                test_open_sha256 if row["evaluation_split"] == "test" else None
            ),
            historical_runtime_authorizations=runtime_authorizations,
        )
        generation_manifest_sha256s.append(
            str(generation_identity["generation_manifest_sha256"])
        )
        semantic_validations = generation_identity["semantic_validations"]
        if not isinstance(semantic_validations, list) or not semantic_validations:
            raise PolicyImprovementSchemaError(
                "Complete evidence lacks sealed semantic checkpoint validation."
            )
        semantic_validation_sha256s.extend(
            hashlib.sha256(canonical_json_bytes(item)).hexdigest()
            for item in semantic_validations
        )
        parent_generation_manifest_sha256 = generation_identity.get(
            "parent_generation_manifest_sha256"
        )
        if parent_generation_manifest_sha256 is not None:
            generation_manifest_sha256s.append(str(parent_generation_manifest_sha256))
        historical_manifests = _historical_failed_attempt_manifest_sha256s(
            generation_identity
        )
        generation_manifest_sha256s.extend(historical_manifests)
        historical_failed_attempt_manifest_sha256s.extend(historical_manifests)
        complete_count += 1
        observed_evaluation_identity = {
            field: _available_value(
                result["identities"][field], path=f"result.identities.{field}"
            )
            for field in (
                "evaluation_runtime_sha256",
                "evaluation_source_git_commit",
                "evaluation_runtime_profile_sha256",
                "evaluation_selected_source_manifest_sha256",
            )
        }
        allowed_evaluation_roles = [
            _authorized_role(result_authorization, "policy-improvement-evaluation")
        ]
        if result["tier"] != "smoke":
            allowed_evaluation_roles.append(
                _authorized_role(result_authorization, "policy-improvement-training")
            )
        authorized_evaluation_identities = [
            {
                "evaluation_runtime_sha256": role["runtime_sha256"],
                "evaluation_source_git_commit": role["source_git_commit"],
                "evaluation_runtime_profile_sha256": role["runtime_profile_sha256"],
                "evaluation_selected_source_manifest_sha256": role[
                    "selected_source_manifest_sha256"
                ],
            }
            for role in allowed_evaluation_roles
        ]
        if observed_evaluation_identity not in authorized_evaluation_identities:
            raise PolicyImprovementSchemaError(
                "Result evaluation identity is not authorized."
            )
        initialization = str(result["identities"]["initialization_sha256"])
        initialization_key = (str(result["phase"]), int(result["seed"]))
        prior_initialization = paired_initializations.setdefault(
            initialization_key, initialization
        )
        if prior_initialization != initialization:
            raise PolicyImprovementSchemaError(
                "Paired methods do not share the registered initialization identity."
            )
        registered_original_indices: tuple[int, ...] | None = None
        if registered_populations is not None:
            population_id = str(row["evaluation_population"])
            population = registered_populations.get(population_id)
            if not isinstance(population, Mapping):
                raise PolicyImprovementSchemaError(
                    "Result row names an unregistered v2 evaluation population."
                )
            split_name = str(population["split"])
            split_bindings = dataset_bindings[split_name]
            indices = tuple(int(index) for index in population["indices"])
            registered_original_indices = indices
            registered_records = tuple(
                split_bindings["record_sha256s"][index] for index in indices
            )
            registered_inputs = tuple(
                split_bindings["input_sha256s"][index] for index in indices
            )
            registered_puzzle_ids = tuple(
                f"{split_name}-{index:06d}" for index in indices
            )
            population_identity_value = str(population["ordered_record_sha256"])
        else:
            population_tier = (
                "confirmatory" if tier in {"confirmatory", "ablation"} else tier
            )
            population = protocol["evaluation_populations"][population_tier]
            split_name = str(population["split"])
            split_bindings = dataset_bindings[split_name]
            population_count = int(population["count"])
            registered_records = split_bindings["record_sha256s"][:population_count]
            registered_inputs = split_bindings["input_sha256s"][:population_count]
            registered_puzzle_ids = split_bindings["puzzle_ids"][:population_count]
            population_identity = population["ordered_record_sha256"]
            if population_identity["status"] != "available":
                raise PolicyImprovementSchemaError(
                    "Result audit requires a frozen exact evaluation population."
                )
            population_identity_value = str(population_identity["value"])
        if population_identity_value != _ordered_record_digest(registered_records):
            raise PolicyImprovementSchemaError(
                "Registered evaluation population differs from authenticated split bytes."
            )
        if (
            result["identities"]["evaluation_ordered_records_sha256"]
            != population_identity_value
        ):
            raise PolicyImprovementSchemaError(
                "Result evaluation population differs from the registered tier."
            )
        snapshots = result["evaluation_snapshots"]
        interaction = snapshots[0]
        interaction_checkpoint = _available_value(
            interaction["checkpoint_sha256"],
            path="interaction_snapshot.checkpoint_sha256",
        )
        interaction_model = _available_value(
            interaction["model_state_sha256"],
            path="interaction_snapshot.model_state_sha256",
        )
        if (
            _available_value(
                result["identities"]["checkpoint_sha256"],
                path="result.identities.checkpoint_sha256",
            )
            != interaction_checkpoint
            or _available_value(
                result["identities"]["model_state_sha256"],
                path="result.identities.model_state_sha256",
            )
            != interaction_model
            or _available_value(
                result["artifacts"]["checkpoint"],
                path="result.artifacts.checkpoint",
            )
            != interaction_checkpoint
        ):
            raise PolicyImprovementSchemaError(
                "Interaction snapshot checkpoint/model identities are inconsistent."
            )
        target_value = _available_value(
            interaction["target"]["registered_quantity"],
            path="interaction_snapshot.target.registered_quantity",
        )
        if target_value != expected_interactions:
            raise PolicyImprovementSchemaError(
                "Interaction snapshot target differs from the registered tier budget."
            )
        compute = snapshots[1]
        if tier == "smoke":
            if compute["status"] != "unavailable":
                raise PolicyImprovementSchemaError(
                    "Smoke cannot claim a pre-frozen compute target."
                )
        else:
            if not history:
                raise PolicyImprovementSchemaError(
                    "Non-smoke compute snapshots require the post-smoke freeze."
                )
            targets = history[1]["common_compute_targets"]
            if (
                compute["status"] != "available"
                or _available_value(
                    compute["target"]["registered_quantity"],
                    path="compute_snapshot.target.registered_quantity",
                )
                != targets[tier]
            ):
                raise PolicyImprovementSchemaError(
                    "Compute snapshot does not use the common cross-method target."
                )
        available_snapshots = [
            snapshot for snapshot in snapshots if snapshot["status"] == "available"
        ]
        available_lineages = {
            str(
                _available_value(
                    snapshot["checkpoint_lineage_sha256"],
                    path="snapshot.checkpoint_lineage_sha256",
                )
            )
            for snapshot in available_snapshots
        }
        if len(available_lineages) != 1:
            raise PolicyImprovementSchemaError(
                "A run's available snapshots must share one checkpoint lineage."
            )
        local_checkpoints: dict[str, Mapping[str, object]] = {}
        for snapshot in snapshots:
            if snapshot["status"] != "available":
                continue
            observed_environment = int(
                _available_value(
                    snapshot["observed_environment_interactions"],
                    path="snapshot.observed_environment_interactions",
                )
            )
            if observed_environment > expected_interactions:
                raise PolicyImprovementSchemaError(
                    "Snapshot environment interactions exceed the tier cap."
                )
            checkpoint = str(
                _available_value(
                    snapshot["checkpoint_sha256"],
                    path="snapshot.checkpoint_sha256",
                )
            )
            owner = checkpoint_owners.setdefault(checkpoint, run_id)
            if owner != run_id:
                raise PolicyImprovementSchemaError(
                    "A checkpoint artifact was reused by multiple registered runs."
                )
            if checkpoint in local_checkpoints:
                prior_snapshot = local_checkpoints[checkpoint]
                prior_model = _available_value(
                    prior_snapshot["model_state_sha256"],
                    path="prior_snapshot.model_state_sha256",
                )
                current_model = _available_value(
                    snapshot["model_state_sha256"],
                    path="snapshot.model_state_sha256",
                )
                interaction_compute = _available_value(
                    interaction["observed_recurrent_map_applications"],
                    path="interaction_snapshot.observed_recurrent_map_applications",
                )
                compute_target = _available_value(
                    compute["target"]["registered_quantity"],
                    path="compute_snapshot.target.registered_quantity",
                )
                if not (
                    prior_model == current_model
                    and observed_environment == expected_interactions
                    and int(interaction_compute) == int(compute_target)
                    and _available_value(
                        compute["observed_environment_interactions"],
                        path="compute_snapshot.observed_environment_interactions",
                    )
                    == expected_interactions
                    and _available_value(
                        compute["observed_recurrent_map_applications"],
                        path="compute_snapshot.observed_recurrent_map_applications",
                    )
                    == compute_target
                ):
                    raise PolicyImprovementSchemaError(
                        "A checkpoint may serve both snapshots only when both frozen "
                        "stopping conditions coincide on the same lineage."
                    )
            else:
                local_checkpoints[checkpoint] = snapshot
            primary_variant = str(result["primary_policy_variant"])
            for evaluation in snapshot["policy_evaluations"]:
                order = _audit_evaluation(
                    result,
                    snapshot,
                    evaluation,
                    per_instance_documents,
                    registered_population_count=len(registered_records),
                    registered_ordered_record_sha256=population_identity_value,
                    registered_record_sha256s=registered_records,
                    registered_input_sha256s=registered_inputs,
                    registered_puzzle_ids=registered_puzzle_ids,
                    registered_original_indices=registered_original_indices,
                    consumed_per_instance_sha256s=consumed_per_instance_sha256s,
                )
                if evaluation["policy_variant"] == primary_variant:
                    key = (int(result["seed"]), str(snapshot["snapshot_kind"]))
                    prior = paired_puzzle_orders.setdefault(key, order)
                    if prior != order:
                        raise PolicyImprovementSchemaError(
                            "Paired methods use different puzzle identities or order."
                        )
    supplied_per_instance_sha256s = set(per_instance_documents)
    if consumed_per_instance_sha256s != supplied_per_instance_sha256s:
        raise PolicyImprovementSchemaError(
            "Per-instance artifact inventory differs: "
            f"unreferenced={sorted(supplied_per_instance_sha256s - consumed_per_instance_sha256s)}, "
            f"missing={sorted(consumed_per_instance_sha256s - supplied_per_instance_sha256s)}."
        )
    is_v2_report = registered_population_document is not None
    report = {
        "schema_name": (
            str(protocol["document_schemas"]["audit"][0])
            if is_v2_report
            else "policy_improvement_audit_v5"
        ),
        "schema_version": (
            int(protocol["document_schemas"]["audit"][1])
            if is_v2_report
            else AUDIT_SCHEMA_VERSION
        ),
        "protocol_sha256": protocol_digest,
        "registry_sha256": registry_digest,
        "amendment_history_sha256": history_digest,
        "runtime_authorization_sha256": authorization_digest,
        "execution_role": execution_role_name,
        "execution_runtime_sha256": checked_execution_identity["runtime_sha256"],
        "execution_source_git_commit": checked_execution_identity["source_git_commit"],
        "execution_runtime_profile_sha256": checked_execution_identity[
            "runtime_profile_sha256"
        ],
        "phases": list(phases),
        "expected_rows": len(expected_rows),
        "complete_rows": complete_count,
        "failed_rows": failed_count,
        "test_open_verified": uses_test,
        "test_open_sha256": test_open_sha256 if uses_test else None,
        "per_instance_artifact_count": len(per_instance_documents),
        "result_set_sha256": _canonical_result_set_sha256(checked_result_documents),
        "per_instance_set_sha256": _canonical_per_instance_set_sha256(
            per_instance_documents
        ),
        "generation_manifest_set_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(generation_manifest_sha256s))
        ).hexdigest(),
        "historical_failed_attempt_count": len(
            historical_failed_attempt_manifest_sha256s
        ),
        "historical_failed_attempt_manifest_set_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(historical_failed_attempt_manifest_sha256s))
        ).hexdigest(),
        "semantic_checkpoint_validation_count": len(semantic_validation_sha256s),
        "semantic_checkpoint_validation_set_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(semantic_validation_sha256s))
        ).hexdigest(),
    }
    if is_v2_report:
        assert registered_population_document is not None
        population_ids = sorted(
            {str(row["evaluation_population"]) for row in expected_rows}
        )
        populations = registered_population_document["populations"]
        report.update(
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_schema_name": protocol["schema_name"],
                "protocol_schema_version": protocol["schema_version"],
                "registry_schema_name": registry["schema_name"],
                "registry_schema_version": registry["registry_schema_version"],
                "population_registry_schema_name": registered_population_document[
                    "schema_name"
                ],
                "population_registry_schema_version": registered_population_document[
                    "schema_version"
                ],
                "population_registry_sha256": hashlib.sha256(
                    canonical_json_bytes(registered_population_document)
                ).hexdigest(),
                "runtime_authorization_schema_name": authorization["schema_name"],
                "runtime_authorization_schema_version": authorization["schema_version"],
                "evaluation_populations": [
                    {
                        "population_id": population_id,
                        "split": populations[population_id]["split"],
                        "count": populations[population_id]["count"],
                        "binding_sha256": populations[population_id]["binding_sha256"],
                        "ordered_record_sha256": populations[population_id][
                            "ordered_record_sha256"
                        ],
                        "ordered_input_sha256": populations[population_id][
                            "ordered_input_sha256"
                        ],
                    }
                    for population_id in population_ids
                ],
                "validation_data_opened": any(
                    str(row["evaluation_split"]) == "validation"
                    for row in expected_rows
                ),
                "test_data_opened": uses_test,
            }
        )
    canonical_json_bytes(report)
    return report


def main(
    argv: Sequence[str] | None = None,
    *,
    checkpoint_validator: (
        Callable[[Mapping[str, object]], Mapping[str, object]] | None
    ) = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--amendment", action="append", default=[])
    parser.add_argument("--amendment-evidence", action="append", default=[])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--runtime-authorization-json", required=True)
    parser.add_argument("--runtime-authorization-sha256", required=True)
    parser.add_argument(
        "--historical-runtime-authorization",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    parser.add_argument("--audit-runtime-sha256", required=True)
    parser.add_argument("--audit-runtime-profile-sha256", required=True)
    parser.add_argument("--audit-source-git-commit", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--producer-source-manifest-sha256", required=True)
    parser.add_argument("--test-open-record")
    parser.add_argument("--test-open-owner-root")
    parser.add_argument("--test-open-sha256")
    parser.add_argument("--phase", action="append", required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--per-instance", action="append", default=[])
    arguments = parser.parse_args(argv)
    if checkpoint_validator is None:
        raise PolicyImprovementSchemaError(
            "Audit execution requires the sealed checkpoint validator."
        )
    per_instance: dict[str, object] = {}
    for path_text in arguments.per_instance:
        value = load_strict_json(path_text)
        digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        if digest in per_instance:
            raise PolicyImprovementSchemaError(
                "Duplicate per-instance artifact digest."
            )
        per_instance[digest] = value
    protocol = load_strict_json(arguments.protocol)
    amendments = [load_strict_json(path) for path in arguments.amendment]
    prior_evidence: dict[str, Mapping[str, object]] = {}
    for specification in arguments.amendment_evidence:
        if specification.count("=") != 1:
            raise PolicyImprovementSchemaError(
                "--amendment-evidence must use PHASE=JSON_PATH."
            )
        phase, path = specification.split("=", 1)
        if phase in prior_evidence:
            raise PolicyImprovementSchemaError("Duplicate amendment-evidence phase.")
        prior_evidence[phase] = _object(
            load_strict_json(path), path=f"amendment_evidence.{phase}"
        )
    try:
        authorization_bytes = arguments.runtime_authorization_json.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PolicyImprovementSchemaError(
            "Protected runtime authorization is not canonical ASCII JSON."
        ) from exc
    authorization = load_strict_json_bytes(authorization_bytes)
    if (
        hashlib.sha256(authorization_bytes).hexdigest()
        != arguments.runtime_authorization_sha256
        or runtime_authorization_sha256(authorization)
        != arguments.runtime_authorization_sha256
    ):
        raise PolicyImprovementSchemaError(
            "Protected runtime authorization digest differs."
        )
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    historical_authorizations = _load_historical_runtime_authorizations(
        arguments.historical_runtime_authorization,
        protocol_sha256=protocol_digest,
    )
    report = audit_result_set(
        protocol,
        load_strict_json(arguments.registry),
        [load_strict_json(path) for path in arguments.result],
        per_instance,
        phases=arguments.phase,
        amendment_history=amendments,
        base_configs=load_registered_base_configs(protocol, arguments.project_root),
        project_root=arguments.project_root,
        dataset_root=arguments.dataset_root,
        evidence_root=arguments.evidence_root,
        runtime_authorization=authorization,
        historical_runtime_authorizations=historical_authorizations,
        audit_execution_identity={
            "runtime_sha256": arguments.audit_runtime_sha256,
            "runtime_profile_sha256": arguments.audit_runtime_profile_sha256,
            "source_git_commit": arguments.audit_source_git_commit,
            "launcher_sha256": arguments.launcher_sha256,
            "producer_git_commit": arguments.producer_git_commit,
            "producer_source_manifest_sha256": (
                arguments.producer_source_manifest_sha256
            ),
        },
        checkpoint_validator=checkpoint_validator,
        amendment_evidence=prior_evidence,
        test_open_record=(
            load_strict_json(arguments.test_open_record)
            if arguments.test_open_record is not None
            else None
        ),
        test_open_owner_root=arguments.test_open_owner_root,
        test_open_sha256=arguments.test_open_sha256,
    )
    print(canonical_json_bytes(report).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
