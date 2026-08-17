#!/usr/bin/env fbpython
"""Build and verify the symmetry-disjoint policy-improvement 4x4 corpus."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import itertools
import json
import os
import random
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dataset.build_4x4_sudoku import create_puzzle, generate_solved_4x4
from dataset.build_iclr_confirmatory_4x4 import count_solutions
from phase4_runtime_profile import POLICY_DATASET_BUILDER_PROFILE_PATHS
from utils.dataset_provenance import (
    input_sha256,
    ordered_record_sha256,
    sample_sha256,
)


BUILD_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 2
CANONICALIZATION_SCHEME = "sudoku4x4_spatial_digit_lexicographic_v1"
SYMMETRY_GROUP_ORDER = 3072
DEFAULT_SPLITS = (
    ("train", 1024, 26081401),
    ("validation", 256, 26081402),
    ("test", 512, 26081403),
)
PRODUCER_SOURCE_PATHS = POLICY_DATASET_BUILDER_PROFILE_PATHS
PRODUCER_ATTESTATION_FIELDS = frozenset(
    {
        "git_commit",
        "launcher_sha256",
        "runtime_sha256",
        "source_manifest_sha256",
    }
)
_TOP_LEVEL_NAMES = {
    "MANIFEST.json",
    "build_config.json",
    "identifiers.json",
    "manifests",
    "test",
    "train",
    "validation",
}
_SPLIT_FILE_NAMES = {
    "all__group_indices.npy",
    "all__inputs.npy",
    "all__labels.npy",
    "all__puzzle_identifiers.npy",
    "all__puzzle_indices.npy",
    "dataset.json",
    "records.json",
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class PolicyImprovementDatasetError(RuntimeError):
    pass


def _producer_attestation(value: Mapping[str, object]) -> dict[str, str]:
    """Return one exact launcher-authenticated producer identity."""

    if not isinstance(value, Mapping) or set(value) != PRODUCER_ATTESTATION_FIELDS:
        raise PolicyImprovementDatasetError(
            "Dataset producer attestation has an invalid inventory."
        )
    result: dict[str, str] = {}
    for name in sorted(PRODUCER_ATTESTATION_FIELDS):
        field = value[name]
        pattern = r"[0-9a-f]{40}" if name == "git_commit" else r"[0-9a-f]{64}"
        if not isinstance(field, str) or re.fullmatch(pattern, field) is None:
            raise PolicyImprovementDatasetError(
                f"Dataset producer attestation field {name!r} is invalid."
            )
        result[name] = field
    return result


def _row_axis_permutations() -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    for bands in itertools.permutations((0, 1)):
        for first in itertools.permutations((0, 1)):
            for second in itertools.permutations((0, 1)):
                within = (first, second)
                values.append(
                    tuple(2 * band + row for band in bands for row in within[band])
                )
    if len(values) != 8 or len(set(values)) != 8:
        raise AssertionError("4x4 axis symmetry enumeration is invalid.")
    return tuple(values)


_AXIS_PERMUTATIONS = _row_axis_permutations()


def _validate_record_arrays(
    inputs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    puzzle = np.asarray(inputs)
    solution = np.asarray(labels)
    if puzzle.shape not in {(16,), (4, 4)} or solution.shape not in {(16,), (4, 4)}:
        raise PolicyImprovementDatasetError("Sudoku records must contain 16 cells.")
    puzzle = puzzle.reshape(4, 4).astype(np.uint8, copy=False)
    solution = solution.reshape(4, 4).astype(np.uint8, copy=False)
    if np.any((puzzle < 1) | (puzzle > 5)) or np.any((solution < 2) | (solution > 5)):
        raise PolicyImprovementDatasetError(
            "Encoded Sudoku tokens are outside their domains."
        )
    if np.any((puzzle != 1) & (puzzle != solution)):
        raise PolicyImprovementDatasetError("Puzzle givens disagree with the solution.")
    return puzzle, solution


def _is_valid_solution(solution: np.ndarray) -> bool:
    expected = {2, 3, 4, 5}
    return all(
        set(int(value) for value in solution[index]) == expected
        and set(int(value) for value in solution[:, index]) == expected
        for index in range(4)
    ) and all(
        set(
            int(value)
            for value in solution[
                box_row : box_row + 2, box_column : box_column + 2
            ].flat
        )
        == expected
        for box_row in (0, 2)
        for box_column in (0, 2)
    )


def _canonicalize_digits(sequence: Sequence[int]) -> bytes:
    mapping: dict[int, int] = {}
    next_token = 2
    result = bytearray()
    for token in sequence:
        if token == 1:
            result.append(1)
            continue
        if token not in mapping:
            mapping[token] = next_token
            next_token += 1
        result.append(mapping[token])
    if next_token != 6:
        raise PolicyImprovementDatasetError(
            "A solved record must contain all four digits."
        )
    return bytes(result)


def canonical_sudoku_record_bytes(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    """Canonicalize one record under 128 spatial and 24 digit symmetries."""

    puzzle, solution = _validate_record_arrays(inputs, labels)
    candidates: list[bytes] = []
    for transpose in (False, True):
        base_puzzle = puzzle.T if transpose else puzzle
        base_solution = solution.T if transpose else solution
        for rows in _AXIS_PERMUTATIONS:
            for columns in _AXIS_PERMUTATIONS:
                transformed_puzzle = base_puzzle[np.ix_(rows, columns)]
                transformed_solution = base_solution[np.ix_(rows, columns)]
                sequence = [
                    *(int(value) for value in transformed_puzzle.flat),
                    *(int(value) for value in transformed_solution.flat),
                ]
                candidates.append(_canonicalize_digits(sequence))
    if len(candidates) != 128:
        raise AssertionError("Spatial Sudoku symmetry enumeration is incomplete.")
    return min(candidates)


def canonical_sudoku_record_sha256(inputs: np.ndarray, labels: np.ndarray) -> str:
    return hashlib.sha256(canonical_sudoku_record_bytes(inputs, labels)).hexdigest()


def record_sha256(inputs: np.ndarray, labels: np.ndarray) -> str:
    puzzle, solution = _validate_record_arrays(inputs, labels)
    return sample_sha256(
        puzzle.reshape(-1).astype(np.int32),
        solution.reshape(-1).astype(np.int32),
    )


def ordered_sha256(values: Sequence[str]) -> str:
    try:
        return ordered_record_sha256(values)
    except ValueError as exc:
        raise PolicyImprovementDatasetError(
            "Ordered digest input is not a nonempty SHA-256 list."
        ) from exc


def verify_cross_split_symmetry_hashes(
    values: Mapping[str, Sequence[str]],
) -> None:
    """Reject both within-split and cross-split symmetry-equivalent records."""

    seen: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in values:
            raise PolicyImprovementDatasetError(f"Missing split {split_name!r}.")
        split_values = list(values[split_name])
        if len(split_values) != len(set(split_values)):
            raise PolicyImprovementDatasetError(
                f"Split {split_name!r} contains a symmetry-equivalent duplicate."
            )
        for value in split_values:
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise PolicyImprovementDatasetError("Symmetry digest is malformed.")
            prior = seen.get(value)
            if prior is not None:
                raise PolicyImprovementDatasetError(
                    f"Splits {prior!r} and {split_name!r} overlap under Sudoku symmetry."
                )
            seen[value] = split_name


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _strict_json(path: Path) -> Any:
    def pairs(items: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PolicyImprovementDatasetError(f"Duplicate JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="ascii"), object_pairs_hook=pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementDatasetError("Dataset JSON is invalid.") from exc


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _regular_file_sha256(path: Path) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PolicyImprovementDatasetError(
            f"{path.name!r} is not a singly linked regular file."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
    )


def _validate_private_owner_root(
    owner_root: str | Path,
) -> tuple[Path, int, tuple[int, int, int]]:
    requested = Path(owner_root)
    if not requested.is_absolute() or requested != Path(os.path.normpath(requested)):
        raise PolicyImprovementDatasetError(
            "Dataset owner root must be an absolute canonical path."
        )
    try:
        requested_status = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementDatasetError(
            "Dataset owner root does not exist."
        ) from exc
    if resolved != requested or stat.S_ISLNK(requested_status.st_mode):
        raise PolicyImprovementDatasetError(
            "Dataset owner root cannot contain a path alias or symlink."
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(resolved, _DIRECTORY_OPEN_FLAGS)
        opened_status = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise PolicyImprovementDatasetError(
            "Dataset owner root cannot be opened safely."
        ) from exc
    assert descriptor is not None
    identity = _directory_identity(requested_status)
    if (
        identity != _directory_identity(opened_status)
        or not stat.S_ISDIR(opened_status.st_mode)
        or opened_status.st_uid != os.geteuid()
        or stat.S_IMODE(opened_status.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PolicyImprovementDatasetError(
            "Dataset owner root must be an owned mode-0700 directory."
        )
    return resolved, descriptor, identity


def _revalidate_owner_root(
    owner: Path,
    descriptor: int,
    expected_identity: tuple[int, int, int],
) -> None:
    try:
        path_status = owner.lstat()
        opened_status = os.fstat(descriptor)
    except OSError as exc:
        raise PolicyImprovementDatasetError(
            "Dataset owner root changed during publication."
        ) from exc
    if (
        _directory_identity(path_status) != expected_identity
        or _directory_identity(opened_status) != expected_identity
        or path_status.st_uid != os.geteuid()
        or opened_status.st_uid != os.geteuid()
        or stat.S_IMODE(path_status.st_mode) != 0o700
        or stat.S_IMODE(opened_status.st_mode) != 0o700
    ):
        raise PolicyImprovementDatasetError(
            "Dataset owner root changed during publication."
        )


def _validate_output_path(output_root: str | Path, owner: Path) -> Path:
    output = Path(output_root)
    if not output.is_absolute() or output != Path(os.path.normpath(output)):
        raise PolicyImprovementDatasetError(
            "Dataset output must be an absolute canonical path."
        )
    if output.parent != owner or output.name in {"", ".", ".."}:
        raise PolicyImprovementDatasetError(
            "Dataset output must be one direct child of its owner root."
        )
    return output


def _open_and_validate_stage(
    owner_descriptor: int,
    name: str,
    expected_identity: tuple[int, int, int],
) -> int:
    descriptor: int | None = None
    try:
        path_status = os.stat(
            name,
            dir_fd=owner_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=owner_descriptor,
        )
        opened_status = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise PolicyImprovementDatasetError(
            "Dataset staging directory changed during publication."
        ) from exc
    assert descriptor is not None
    if (
        _directory_identity(path_status) != expected_identity
        or _directory_identity(opened_status) != expected_identity
        or path_status.st_uid != os.geteuid()
        or opened_status.st_uid != os.geteuid()
        or stat.S_IMODE(path_status.st_mode) != 0o700
        or stat.S_IMODE(opened_status.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PolicyImprovementDatasetError(
            "Dataset staging directory changed during publication."
        )
    return descriptor


def _remove_directory_contents(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(status.st_mode):
            child = os.open(
                name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            try:
                if _directory_identity(os.fstat(child)) != _directory_identity(status):
                    raise PolicyImprovementDatasetError(
                        "Dataset staging child changed during cleanup."
                    )
                _remove_directory_contents(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _cleanup_exact_staging_directory(
    owner_descriptor: int,
    stage_name: str,
    expected_identity: tuple[int, int, int],
) -> bool:
    try:
        descriptor = _open_and_validate_stage(
            owner_descriptor,
            stage_name,
            expected_identity,
        )
    except PolicyImprovementDatasetError:
        return False
    try:
        _remove_directory_contents(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(
            stage_name,
            dir_fd=owner_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return True
    if _directory_identity(current) != expected_identity:
        return False
    os.rmdir(stage_name, dir_fd=owner_descriptor)
    return True


def _rename_directory_no_replace(
    source_directory_descriptor: int,
    source_name: str,
    destination_directory_descriptor: int,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise PolicyImprovementDatasetError(
            "This host lacks immutable dataset publication support."
        )
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
            source_directory_descriptor,
            os.fsencode(source_name),
            destination_directory_descriptor,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise PolicyImprovementDatasetError("Dataset output already exists.")
    raise OSError(error, os.strerror(error), destination_name)


def _publish_directory_no_replace(stage: Path, output: Path) -> None:
    """Compatibility wrapper used by direct no-replace regression tests."""

    _rename_directory_no_replace(
        _AT_FDCWD,
        os.fspath(stage),
        _AT_FDCWD,
        os.fspath(output),
    )


def _generate_split(
    *, count: int, seed: int, seen_records: set[str], seen_symmetries: set[str]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = random.Random(seed)
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    attempts = 0
    while len(inputs) < count:
        attempts += 1
        if attempts > count * 20000:
            raise PolicyImprovementDatasetError(
                "Could not construct the registered symmetry-disjoint population."
            )
        solution = generate_solved_4x4(rng)
        puzzle = create_puzzle(solution, rng.randint(8, 10), rng)
        if count_solutions(puzzle, limit=2) != 1:
            continue
        encoded_inputs = np.where(
            puzzle.reshape(16) == 0, 1, puzzle.reshape(16) + 1
        ).astype(np.int32)
        encoded_labels = (solution.reshape(16) + 1).astype(np.int32)
        record_digest = record_sha256(encoded_inputs, encoded_labels)
        symmetry_digest = canonical_sudoku_record_sha256(encoded_inputs, encoded_labels)
        if record_digest in seen_records or symmetry_digest in seen_symmetries:
            continue
        seen_records.add(record_digest)
        seen_symmetries.add(symmetry_digest)
        index = len(inputs)
        inputs.append(encoded_inputs)
        labels.append(encoded_labels)
        records.append(
            {
                "index": index,
                "record_sha256": record_digest,
                "symmetry_canonical_sha256": symmetry_digest,
            }
        )
    return np.stack(inputs), np.stack(labels), records


def materialized_split_manifest(
    *,
    split_name: str,
    split_path: Path,
    generation_seed: int,
    inputs: np.ndarray,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the exact manifest shape consumed by ``upi_trm_train``."""

    if split_name not in {"train", "validation", "test"}:
        raise PolicyImprovementDatasetError("Split name is not registered.")
    if len(inputs) != len(records) or len(inputs) < 1:
        raise PolicyImprovementDatasetError(
            "Split inputs and records differ in length."
        )
    record_hashes = [str(item["record_sha256"]) for item in records]
    symmetry_hashes = [str(item["symmetry_canonical_sha256"]) for item in records]
    input_hashes = [input_sha256(value) for value in inputs]
    file_inventory = {
        f"{split_name}/{name}": {
            "bytes": (split_path / name).stat().st_size,
            "sha256": _regular_file_sha256(split_path / name),
        }
        for name in sorted(_SPLIT_FILE_NAMES)
    }
    return {
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "builder": "dataset.build_policy_improvement_4x4",
        "files": file_inventory,
        "generated_count": len(inputs),
        "generation_seed": generation_seed,
        "input_sha256s": input_hashes,
        "ordered_record_sha256": ordered_sha256(record_hashes),
        "ordered_symmetry_sha256": ordered_sha256(symmetry_hashes),
        "record_sha256s": record_hashes,
        "symmetry_canonical_sha256s": symmetry_hashes,
        "symmetry_canonicalization": CANONICALIZATION_SCHEME,
    }


def build_dataset(
    output_root: str | Path,
    *,
    owner_root: str | Path,
    producer_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Build an immutable corpus from a pre-import authenticated runtime."""

    producer = _producer_attestation(producer_attestation)
    owner, owner_descriptor, owner_identity = _validate_private_owner_root(owner_root)
    output = _validate_output_path(output_root, owner)
    stage_identity: tuple[int, int, int] | None = None
    staging: Path | None = None
    published = False
    try:
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        try:
            existing = os.stat(
                output.name,
                dir_fd=owner_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISDIR(existing.st_mode):
                raise PolicyImprovementDatasetError(
                    "Dataset output already exists with an invalid type."
                )
            result = verify_dataset(
                output,
                owner_root=owner,
                expected_producer=producer,
            )
            _revalidate_owner_root(owner, owner_descriptor, owner_identity)
            _fsync_directory_descriptor(owner_descriptor)
            _revalidate_owner_root(owner, owner_descriptor, owner_identity)
            return result

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=owner,
            )
        )
        stage_status = staging.lstat()
        stage_identity = _directory_identity(stage_status)
        if (
            not stat.S_ISDIR(stage_status.st_mode)
            or stage_status.st_uid != os.geteuid()
            or stat.S_IMODE(stage_status.st_mode) != 0o700
            or stage_status.st_dev != owner_identity[0]
        ):
            raise PolicyImprovementDatasetError(
                "Dataset staging directory is not private or on the owner filesystem."
            )
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        _fsync_directory_descriptor(owner_descriptor)
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)

        _write_bytes(staging / "identifiers.json", _json_bytes(["<blank>"]))
        manifests_path = staging / "manifests"
        manifests_path.mkdir(mode=0o700)
        build_config = {
            "build_schema_version": BUILD_SCHEMA_VERSION,
            "builder": "dataset.build_policy_improvement_4x4",
            "domain": "sudoku_4x4",
            "producer_source": producer,
            "require_symmetry_disjoint_splits": True,
            "require_unique_solution": True,
            "seed": DEFAULT_SPLITS[0][2],
            "split_order": [name for name, _, _ in DEFAULT_SPLITS],
            "splits": {
                name: {"count": count, "seed": seed}
                for name, count, seed in DEFAULT_SPLITS
            },
            "symmetry_canonicalization": CANONICALIZATION_SCHEME,
        }
        _write_bytes(staging / "build_config.json", _json_bytes(build_config))
        seen_records: set[str] = set()
        seen_symmetries: set[str] = set()
        split_manifests: dict[str, object] = {}
        symmetry_by_split: dict[str, list[str]] = {}
        identifier_offset = 0
        total_count = sum(count for _, count, _ in DEFAULT_SPLITS)
        for split_name, count, seed in DEFAULT_SPLITS:
            split_path = staging / split_name
            split_path.mkdir(mode=0o700)
            inputs, labels, records = _generate_split(
                count=count,
                seed=seed,
                seen_records=seen_records,
                seen_symmetries=seen_symmetries,
            )
            arrays = {
                "all__inputs.npy": inputs,
                "all__labels.npy": labels,
                "all__puzzle_identifiers.npy": np.arange(
                    identifier_offset, identifier_offset + count, dtype=np.int32
                ),
                "all__puzzle_indices.npy": np.arange(count + 1, dtype=np.int32),
                "all__group_indices.npy": np.arange(count + 1, dtype=np.int32),
            }
            for name, value in arrays.items():
                _write_npy(split_path / name, value)
            metadata = {
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
            _write_bytes(split_path / "dataset.json", _json_bytes(metadata))
            _write_bytes(split_path / "records.json", _json_bytes(records))
            _fsync_directory(split_path)
            record_hashes = [str(item["record_sha256"]) for item in records]
            symmetry_hashes = [
                str(item["symmetry_canonical_sha256"]) for item in records
            ]
            symmetry_by_split[split_name] = symmetry_hashes
            materialized_manifest = materialized_split_manifest(
                split_name=split_name,
                split_path=split_path,
                generation_seed=seed,
                inputs=inputs,
                records=records,
            )
            materialized_manifest_path = manifests_path / f"{split_name}.json"
            _write_bytes(materialized_manifest_path, _json_bytes(materialized_manifest))
            split_manifests[split_name] = {
                "count": count,
                "generation_seed": seed,
                "manifest_sha256": _regular_file_sha256(materialized_manifest_path),
                "ordered_record_sha256": ordered_sha256(record_hashes),
                "ordered_symmetry_sha256": ordered_sha256(symmetry_hashes),
                "puzzle_identifier_start": identifier_offset,
            }
            identifier_offset += count
        verify_cross_split_symmetry_hashes(symmetry_by_split)
        _fsync_directory(manifests_path)
        manifest: dict[str, object] = {
            "build_schema_version": BUILD_SCHEMA_VERSION,
            "builder": "dataset.build_policy_improvement_4x4",
            "canonicalization": {
                "group_order": SYMMETRY_GROUP_ORDER,
                "reject_cross_split_overlap": True,
                "scheme": CANONICALIZATION_SCHEME,
            },
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "producer_source": producer,
            "producer_source_paths": list(PRODUCER_SOURCE_PATHS),
            "splits": split_manifests,
        }
        _write_bytes(staging / "MANIFEST.json", _json_bytes(manifest))
        _fsync_directory(staging)
        staged_verification = verify_dataset(
            staging,
            owner_root=owner,
            expected_producer=producer,
        )
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        stage_descriptor = _open_and_validate_stage(
            owner_descriptor,
            staging.name,
            stage_identity,
        )
        try:
            try:
                os.stat(
                    output.name,
                    dir_fd=owner_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise PolicyImprovementDatasetError(
                    "Dataset output appeared during build."
                )
            _revalidate_owner_root(owner, owner_descriptor, owner_identity)
            _rename_directory_no_replace(
                owner_descriptor,
                staging.name,
                owner_descriptor,
                output.name,
            )
            published = True
            try:
                published_status = os.stat(
                    output.name,
                    dir_fd=owner_descriptor,
                    follow_symlinks=False,
                )
                opened_status = os.fstat(stage_descriptor)
            except OSError as exc:
                raise PolicyImprovementDatasetError(
                    "Published dataset cannot be inspected."
                ) from exc
            if (
                _directory_identity(published_status) != stage_identity
                or _directory_identity(opened_status) != stage_identity
            ):
                raise PolicyImprovementDatasetError(
                    "Published dataset differs from the staged directory."
                )
        finally:
            os.close(stage_descriptor)
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        published_verification = verify_dataset(
            output,
            owner_root=owner,
            expected_producer=producer,
        )
        if published_verification != staged_verification:
            raise PolicyImprovementDatasetError(
                "Published dataset identity differs from the staged identity."
            )
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        _fsync_directory_descriptor(owner_descriptor)
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        return published_verification
    except BaseException:
        if not published and staging is not None and stage_identity is not None:
            try:
                removed = _cleanup_exact_staging_directory(
                    owner_descriptor,
                    staging.name,
                    stage_identity,
                )
            except (OSError, PolicyImprovementDatasetError):
                removed = False
            if removed:
                try:
                    _fsync_directory_descriptor(owner_descriptor)
                except OSError:
                    pass
        raise
    finally:
        os.close(owner_descriptor)


def _load_npy_regular(path: Path) -> np.ndarray:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise PolicyImprovementDatasetError("Dataset array is not a regular file.")
    with path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def verify_dataset(
    root: str | Path,
    *,
    owner_root: str | Path,
    expected_producer: Mapping[str, object],
    verify_test_content: bool = True,
) -> dict[str, object]:
    """Recompute corpus identity and require an external producer identity.

    Stage-0 callers set ``verify_test_content`` to false. In that mode the
    authenticated split manifest supplies the registered test metadata, while
    no path below the held-out test directory is opened or deserialized.
    """

    producer = _producer_attestation(expected_producer)
    owner, owner_descriptor, owner_identity = _validate_private_owner_root(owner_root)
    path = _validate_output_path(root, owner)
    try:
        _revalidate_owner_root(owner, owner_descriptor, owner_identity)
        path_status = os.stat(
            path.name,
            dir_fd=owner_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            path.name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=owner_descriptor,
        )
        try:
            opened_status = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        os.close(owner_descriptor)
        raise PolicyImprovementDatasetError(
            "Dataset root cannot be opened safely."
        ) from exc
    if not stat.S_ISDIR(path_status.st_mode) or _directory_identity(
        path_status
    ) != _directory_identity(opened_status):
        os.close(owner_descriptor)
        raise PolicyImprovementDatasetError("Dataset root is not a real directory.")
    os.close(owner_descriptor)
    if {item.name for item in path.iterdir()} != _TOP_LEVEL_NAMES:
        raise PolicyImprovementDatasetError("Dataset root inventory differs.")
    for name in sorted(_TOP_LEVEL_NAMES):
        info = (path / name).lstat()
        if name in {"manifests", "test", "train", "validation"}:
            valid = stat.S_ISDIR(info.st_mode)
        else:
            valid = stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        if not valid:
            raise PolicyImprovementDatasetError(
                "Dataset root contains a symlink, alias, or wrong file type."
            )
    if _strict_json(path / "identifiers.json") != ["<blank>"]:
        raise PolicyImprovementDatasetError("Dataset identifiers differ.")
    expected_build_config = {
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "builder": "dataset.build_policy_improvement_4x4",
        "domain": "sudoku_4x4",
        "producer_source": producer,
        "require_symmetry_disjoint_splits": True,
        "require_unique_solution": True,
        "seed": DEFAULT_SPLITS[0][2],
        "split_order": [name for name, _, _ in DEFAULT_SPLITS],
        "splits": {
            name: {"count": count, "seed": seed} for name, count, seed in DEFAULT_SPLITS
        },
        "symmetry_canonicalization": CANONICALIZATION_SCHEME,
    }
    build_config = _strict_json(path / "build_config.json")
    if not isinstance(build_config, Mapping):
        raise PolicyImprovementDatasetError("Dataset build config is malformed.")
    if dict(build_config) != expected_build_config:
        raise PolicyImprovementDatasetError("Dataset build config differs.")
    manifest = _strict_json(path / "MANIFEST.json")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "build_schema_version",
        "builder",
        "canonicalization",
        "manifest_schema_version",
        "producer_source",
        "producer_source_paths",
        "splits",
    }:
        raise PolicyImprovementDatasetError("Dataset manifest inventory differs.")
    if (
        manifest["build_schema_version"] != BUILD_SCHEMA_VERSION
        or manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["builder"] != "dataset.build_policy_improvement_4x4"
        or manifest["producer_source"] != producer
        or manifest["producer_source_paths"] != list(PRODUCER_SOURCE_PATHS)
    ):
        raise PolicyImprovementDatasetError("Dataset manifest schema differs.")
    if manifest["canonicalization"] != {
        "group_order": 3072,
        "reject_cross_split_overlap": True,
        "scheme": CANONICALIZATION_SCHEME,
    }:
        raise PolicyImprovementDatasetError("Canonicalization contract differs.")
    symmetry_by_split: dict[str, list[str]] = {}
    manifests_path = path / "manifests"
    if manifests_path.is_symlink() or not manifests_path.is_dir():
        raise PolicyImprovementDatasetError("Split manifest directory is invalid.")
    if {item.name for item in manifests_path.iterdir()} != {
        "test.json",
        "train.json",
        "validation.json",
    }:
        raise PolicyImprovementDatasetError("Split manifest inventory differs.")
    identifier_offset = 0
    total_count = sum(count for _, count, _ in DEFAULT_SPLITS)
    for split_name, expected_count, expected_seed in DEFAULT_SPLITS:
        split_registration = manifest["splits"][split_name]
        materialized_manifest_path = manifests_path / f"{split_name}.json"
        if split_registration["manifest_sha256"] != _regular_file_sha256(
            materialized_manifest_path
        ):
            raise PolicyImprovementDatasetError("Split manifest hash differs.")
        split_manifest = _strict_json(materialized_manifest_path)
        if not isinstance(split_manifest, Mapping) or set(split_manifest) != {
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
        }:
            raise PolicyImprovementDatasetError("Materialized split manifest differs.")
        if (
            split_registration["count"] != expected_count
            or split_registration["generation_seed"] != expected_seed
            or split_manifest["generated_count"] != expected_count
            or split_manifest["generation_seed"] != expected_seed
            or split_manifest["build_schema_version"] != BUILD_SCHEMA_VERSION
            or split_manifest["builder"] != "dataset.build_policy_improvement_4x4"
            or split_manifest["symmetry_canonicalization"] != CANONICALIZATION_SCHEME
        ):
            raise PolicyImprovementDatasetError("Split registration differs.")
        expected_relative_names = {f"{split_name}/{name}" for name in _SPLIT_FILE_NAMES}
        files = split_manifest["files"]
        record_hashes = split_manifest["record_sha256s"]
        input_hashes = split_manifest["input_sha256s"]
        symmetry_hashes = split_manifest["symmetry_canonical_sha256s"]
        if (
            not isinstance(files, Mapping)
            or set(files) != expected_relative_names
            or not isinstance(record_hashes, list)
            or not isinstance(input_hashes, list)
            or not isinstance(symmetry_hashes, list)
            or len(record_hashes) != expected_count
            or len(input_hashes) != expected_count
            or len(symmetry_hashes) != expected_count
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for values in (record_hashes, input_hashes, symmetry_hashes)
                for digest in values
            )
        ):
            raise PolicyImprovementDatasetError("Split file inventory differs.")
        for relative_name in sorted(expected_relative_names):
            entry = files[relative_name]
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"bytes", "sha256"}
                or isinstance(entry["bytes"], bool)
                or not isinstance(entry["bytes"], int)
                or entry["bytes"] < 0
                or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef" for character in entry["sha256"]
                )
            ):
                raise PolicyImprovementDatasetError("Split file identity is malformed.")
        if (
            split_manifest["ordered_record_sha256"] != ordered_sha256(record_hashes)
            or split_registration["ordered_record_sha256"]
            != ordered_sha256(record_hashes)
            or split_manifest["ordered_symmetry_sha256"]
            != ordered_sha256(symmetry_hashes)
            or split_registration["ordered_symmetry_sha256"]
            != ordered_sha256(symmetry_hashes)
        ):
            raise PolicyImprovementDatasetError("Ordered split identity differs.")
        symmetry_by_split[split_name] = symmetry_hashes
        if split_name == "test" and not verify_test_content:
            identifier_offset += expected_count
            continue

        split_path = path / split_name
        if split_path.is_symlink() or not split_path.is_dir():
            raise PolicyImprovementDatasetError("Split path is not a real directory.")
        if {item.name for item in split_path.iterdir()} != _SPLIT_FILE_NAMES:
            raise PolicyImprovementDatasetError("Split file inventory differs.")
        for relative_name in sorted(expected_relative_names):
            source_path = path / relative_name
            entry = files[relative_name]
            if entry != {
                "bytes": source_path.stat().st_size,
                "sha256": _regular_file_sha256(source_path),
            }:
                raise PolicyImprovementDatasetError("Split file hash differs.")
        inputs = _load_npy_regular(split_path / "all__inputs.npy")
        labels = _load_npy_regular(split_path / "all__labels.npy")
        if inputs.shape != (expected_count, 16) or labels.shape != (expected_count, 16):
            raise PolicyImprovementDatasetError("Split array shape differs.")
        puzzle_identifiers = _load_npy_regular(
            split_path / "all__puzzle_identifiers.npy"
        )
        expected_identifiers = np.arange(
            identifier_offset, identifier_offset + expected_count, dtype=np.int32
        )
        if not np.array_equal(puzzle_identifiers, expected_identifiers):
            raise PolicyImprovementDatasetError("Puzzle identifiers are not disjoint.")
        expected_indices = np.arange(expected_count + 1, dtype=np.int32)
        for name in ("all__puzzle_indices.npy", "all__group_indices.npy"):
            if not np.array_equal(
                _load_npy_regular(split_path / name), expected_indices
            ):
                raise PolicyImprovementDatasetError("Puzzle/group indices differ.")
        expected_metadata = {
            "blank_identifier_id": 0,
            "ignore_label_id": 0,
            "mean_puzzle_examples": 1,
            "num_puzzle_identifiers": total_count,
            "pad_id": 0,
            "seq_len": 16,
            "sets": ["all"],
            "total_groups": expected_count,
            "total_puzzles": expected_count,
            "vocab_size": 6,
        }
        if _strict_json(split_path / "dataset.json") != expected_metadata:
            raise PolicyImprovementDatasetError("Dataset metadata differs.")
        records = _strict_json(split_path / "records.json")
        if not isinstance(records, list) or len(records) != expected_count:
            raise PolicyImprovementDatasetError("Record manifest length differs.")
        computed_record_hashes: list[str] = []
        computed_input_hashes: list[str] = []
        computed_symmetry_hashes: list[str] = []
        for index, record in enumerate(records):
            puzzle, solution = _validate_record_arrays(inputs[index], labels[index])
            if not _is_valid_solution(solution):
                raise PolicyImprovementDatasetError(
                    "Label is not a valid Sudoku solution."
                )
            decoded_puzzle = np.where(puzzle == 1, 0, puzzle - 1)
            if count_solutions(decoded_puzzle, limit=2) != 1:
                raise PolicyImprovementDatasetError("Puzzle is not uniquely solvable.")
            expected_record = record_sha256(inputs[index], labels[index])
            expected_symmetry = canonical_sudoku_record_sha256(
                inputs[index], labels[index]
            )
            if record != {
                "index": index,
                "record_sha256": expected_record,
                "symmetry_canonical_sha256": expected_symmetry,
            }:
                raise PolicyImprovementDatasetError("Record identity differs.")
            computed_record_hashes.append(expected_record)
            computed_input_hashes.append(input_sha256(inputs[index]))
            computed_symmetry_hashes.append(expected_symmetry)
        if (
            record_hashes != computed_record_hashes
            or input_hashes != computed_input_hashes
        ):
            raise PolicyImprovementDatasetError("Ordered record identity differs.")
        if symmetry_hashes != computed_symmetry_hashes:
            raise PolicyImprovementDatasetError("Ordered symmetry identity differs.")
        identifier_offset += expected_count
    verify_cross_split_symmetry_hashes(symmetry_by_split)
    return {
        "manifest_sha256": _regular_file_sha256(path / "MANIFEST.json"),
        "producer_source": producer,
        "split_manifest_sha256": {
            split_name: _regular_file_sha256(path / "manifests" / f"{split_name}.json")
            for split_name, _, _ in DEFAULT_SPLITS
        },
        "split_ordered_record_sha256": {
            split_name: str(manifest["splits"][split_name]["ordered_record_sha256"])
            for split_name, _, _ in DEFAULT_SPLITS
        },
        "total_records": sum(count for _, count, _ in DEFAULT_SPLITS),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_attestation: Mapping[str, object] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dataset-builder-entrypoint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--owner-root", required=True)
    build.add_argument("--output-root", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--owner-root", required=True)
    verify.add_argument("--root", required=True)
    arguments = parser.parse_args(argv)
    if not arguments.policy_dataset_builder_entrypoint or runtime_attestation is None:
        raise PolicyImprovementDatasetError(
            "Dataset build and verification require the authenticated launcher."
        )
    producer = _producer_attestation(runtime_attestation)
    if arguments.command == "build":
        result = build_dataset(
            arguments.output_root,
            owner_root=arguments.owner_root,
            producer_attestation=producer,
        )
    else:
        result = verify_dataset(
            arguments.root,
            owner_root=arguments.owner_root,
            expected_producer=producer,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
