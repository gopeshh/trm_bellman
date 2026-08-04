#!/usr/bin/env python3
"""Build the immutable hard-4x4 Sudoku corpus registered for ICLR runs."""

from __future__ import annotations

import argparse
import ctypes
import errno
import functools
import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import dataset.build_4x4_sudoku as build_4x4_sudoku_module
import utils.dataset_provenance as dataset_provenance_module
import utils.run_identity as run_identity_module
from dataset.build_4x4_sudoku import create_puzzle, generate_solved_4x4
from utils.dataset_provenance import (
    DatasetProvenanceError,
    input_sha256,
    ordered_record_sha256,
    sample_sha256,
)
from utils.run_identity import (
    RunIdentityError,
    assert_git_files_match_head,
    discover_clean_git_source,
)


BUILD_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
BUILDER_NAME = "dataset.build_iclr_confirmatory_4x4"
PRODUCER_SOURCE_PATHS = (
    "dataset/build_4x4_sudoku.py",
    "dataset/build_iclr_confirmatory_4x4.py",
    "utils/dataset_provenance.py",
    "utils/run_identity.py",
)
DEFAULT_SPLITS = (
    ("train", 1024, 26080301),
    ("validation", 256, 26080302),
    ("test", 512, 26080303),
)


class DatasetBuildError(RuntimeError):
    """Raised when generation or verification violates the registered contract."""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    count: int
    seed: int


@dataclass(frozen=True)
class BuildSpec:
    splits: tuple[SplitSpec, ...]
    producer_commit: str
    min_empty_cells: int = 6
    max_empty_cells: int = 8

    def __post_init__(self) -> None:
        names = [split.name for split in self.splits]
        if not self.splits or len(set(names)) != len(names):
            raise ValueError("Split names must be nonempty and unique.")
        if any(not split.name or split.count < 1 for split in self.splits):
            raise ValueError("Every split needs a name and positive count.")
        if not (0 <= self.min_empty_cells <= self.max_empty_cells < 16):
            raise ValueError("Empty-cell bounds must satisfy 0 <= min <= max < 16.")
        if len(self.producer_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.producer_commit
        ):
            raise ValueError("Producer commit must be 40 lowercase hexadecimal digits.")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_sha256(inputs: np.ndarray) -> str:
    return input_sha256(inputs)


def _record_sha256(inputs: np.ndarray, solution: np.ndarray) -> str:
    return sample_sha256(inputs, solution)


def _ordered_sha256(values: Sequence[str]) -> str:
    try:
        return ordered_record_sha256(values)
    except (ValueError, DatasetProvenanceError) as exc:
        raise DatasetBuildError("Ordered manifest hash input is invalid.") from exc


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _json_bytes(value))


def _load_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DatasetBuildError(
                    f"JSON file {path.name!r} contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(f"Invalid JSON file {path}.") from exc


def _candidate_values(grid: np.ndarray, row: int, col: int) -> list[int]:
    used = set(int(value) for value in grid[row] if value)
    used.update(int(value) for value in grid[:, col] if value)
    box_row = 2 * (row // 2)
    box_col = 2 * (col // 2)
    used.update(
        int(value)
        for value in grid[box_row : box_row + 2, box_col : box_col + 2].flat
        if value
    )
    return [value for value in range(1, 5) if value not in used]


def count_solutions(puzzle: np.ndarray, limit: int = 2) -> int:
    """Count solutions up to ``limit`` without touching process-global RNG state."""

    if limit < 1:
        raise ValueError("Solution-count limit must be positive.")
    grid = np.asarray(puzzle, dtype=np.int32).copy()
    if grid.shape != (4, 4) or np.any((grid < 0) | (grid > 4)):
        return 0

    for index in range(4):
        row = [int(value) for value in grid[index] if value]
        col = [int(value) for value in grid[:, index] if value]
        if len(row) != len(set(row)) or len(col) != len(set(col)):
            return 0
    for box_row in (0, 2):
        for box_col in (0, 2):
            box = [
                int(value)
                for value in grid[box_row : box_row + 2, box_col : box_col + 2].flat
                if value
            ]
            if len(box) != len(set(box)):
                return 0

    total = 0

    def search() -> None:
        nonlocal total
        if total >= limit:
            return
        best: tuple[int, int, list[int]] | None = None
        for row in range(4):
            for col in range(4):
                if grid[row, col] != 0:
                    continue
                candidates = _candidate_values(grid, row, col)
                if not candidates:
                    return
                if best is None or len(candidates) < len(best[2]):
                    best = (row, col, candidates)
        if best is None:
            total += 1
            return
        row, col, candidates = best
        for value in candidates:
            grid[row, col] = value
            search()
            grid[row, col] = 0
            if total >= limit:
                return

    search()
    return total


def _is_valid_solution(solution: np.ndarray) -> bool:
    solution = np.asarray(solution)
    expected = {1, 2, 3, 4}
    if solution.shape != (4, 4):
        return False
    for index in range(4):
        if set(int(value) for value in solution[index]) != expected:
            return False
        if set(int(value) for value in solution[:, index]) != expected:
            return False
    for box_row in (0, 2):
        for box_col in (0, 2):
            if (
                set(
                    int(value)
                    for value in solution[
                        box_row : box_row + 2,
                        box_col : box_col + 2,
                    ].flat
                )
                != expected
            ):
                return False
    return True


@functools.lru_cache(maxsize=1)
def _valid_solution_grid_keys() -> frozenset[tuple[int, ...]]:
    """Enumerate every valid 4x4 completion deterministically."""

    grid = np.zeros((4, 4), dtype=np.int32)
    solutions: set[tuple[int, ...]] = set()

    def search(position: int) -> None:
        if position == grid.size:
            solutions.add(tuple(int(value) for value in grid.flat))
            return
        row, col = divmod(position, 4)
        for value in _candidate_values(grid, row, col):
            grid[row, col] = value
            search(position + 1)
        grid[row, col] = 0

    search(0)
    return frozenset(solutions)


def audit_solution_grid_coverage(
    encoded_solutions_by_split: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Measure completion-grid coverage and reuse without inspecting givens."""

    if not encoded_solutions_by_split:
        raise DatasetBuildError("Solution-grid audit requires at least one split.")

    valid_solutions = _valid_solution_grid_keys()
    solution_keys: dict[str, list[tuple[int, ...]]] = {}
    for split_name, raw_solutions in encoded_solutions_by_split.items():
        solutions = np.asarray(raw_solutions)
        if (
            not isinstance(split_name, str)
            or not split_name
            or solutions.ndim != 2
            or solutions.shape[1] != 16
            or not np.issubdtype(solutions.dtype, np.integer)
        ):
            raise DatasetBuildError("Solution-grid audit input is malformed.")
        keys = [
            tuple(int(value) - 1 for value in encoded_solution)
            for encoded_solution in solutions
        ]
        if any(key not in valid_solutions for key in keys):
            raise DatasetBuildError(
                f"Solution-grid audit found an invalid completion in {split_name}."
            )
        solution_keys[split_name] = keys

    unique_by_split = {
        split_name: set(keys) for split_name, keys in solution_keys.items()
    }
    split_names = list(solution_keys)
    pairwise_overlap_counts = {
        f"{left}/{right}": len(unique_by_split[left] & unique_by_split[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    result: dict[str, object] = {
        "all_valid_completion_count": len(valid_solutions),
        "corpus_unique_completion_count": len(
            set().union(*unique_by_split.values())
        ),
        "pairwise_distinct_completion_overlap_counts": pairwise_overlap_counts,
        "split_unique_completion_counts": {
            split_name: len(unique_by_split[split_name])
            for split_name in split_names
        },
    }
    if "train" in solution_keys and "test" in solution_keys:
        result.update(
            {
                "test_record_count": len(solution_keys["test"]),
                "test_records_sharing_train_completion": sum(
                    key in unique_by_split["train"]
                    for key in solution_keys["test"]
                ),
            }
        )
    return result


def _encode_record(
    puzzle: np.ndarray,
    solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.where(puzzle.reshape(-1) == 0, 1, puzzle.reshape(-1) + 1)
    labels = solution.reshape(-1) + 1
    return inputs.astype(np.int32), labels.astype(np.int32)


def _generate_split(
    split: SplitSpec,
    *,
    min_empty_cells: int,
    max_empty_cells: int,
    seen_inputs: set[str],
    seen_records: set[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    rng = random.Random(split.seed)
    records: list[tuple[np.ndarray, np.ndarray]] = []
    stats = {
        "attempts": 0,
        "rejected_nonunique_solution": 0,
        "rejected_duplicate_input": 0,
        "rejected_duplicate_record": 0,
    }
    max_attempts = max(10_000, split.count * 1_000)
    while len(records) < split.count:
        stats["attempts"] += 1
        if stats["attempts"] > max_attempts:
            raise DatasetBuildError(
                f"Unable to generate {split.count} unique records for {split.name!r}."
            )
        solution = generate_solved_4x4(rng)
        empty_cells = rng.randint(min_empty_cells, max_empty_cells)
        puzzle = create_puzzle(solution, 16 - empty_cells, rng)
        if count_solutions(puzzle, limit=2) != 1:
            stats["rejected_nonunique_solution"] += 1
            continue
        inputs, labels = _encode_record(puzzle, solution)
        input_digest = _input_sha256(inputs)
        record_digest = _record_sha256(inputs, labels)
        if input_digest in seen_inputs:
            stats["rejected_duplicate_input"] += 1
            continue
        if record_digest in seen_records:
            stats["rejected_duplicate_record"] += 1
            continue
        seen_inputs.add(input_digest)
        seen_records.add(record_digest)
        records.append((inputs, labels))
    return records, stats


def _metadata(count: int) -> dict[str, object]:
    return {
        "blank_identifier_id": 0,
        "ignore_label_id": 0,
        "mean_puzzle_examples": 1,
        "num_puzzle_identifiers": count,
        "pad_id": 0,
        "seq_len": 16,
        "sets": ["all"],
        "total_groups": count,
        "total_puzzles": count,
        "vocab_size": 6,
    }


def _write_split(
    root: Path,
    split: SplitSpec,
    records: Sequence[tuple[np.ndarray, np.ndarray]],
    generation_stats: dict[str, int],
    *,
    min_empty_cells: int,
    max_empty_cells: int,
) -> tuple[dict[str, object], str]:
    split_dir = root / split.name
    split_dir.mkdir(parents=True, exist_ok=False)
    inputs = np.stack([record[0] for record in records]).astype(np.int32)
    labels = np.stack([record[1] for record in records]).astype(np.int32)
    arrays = {
        "all__group_indices.npy": np.arange(split.count + 1, dtype=np.int32),
        "all__inputs.npy": inputs,
        "all__labels.npy": labels,
        "all__puzzle_identifiers.npy": np.arange(split.count, dtype=np.int32),
        "all__puzzle_indices.npy": np.arange(split.count + 1, dtype=np.int32),
    }
    for filename, array in arrays.items():
        with (split_dir / filename).open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    _write_json(split_dir / "dataset.json", _metadata(split.count))

    input_hashes = [_input_sha256(inputs[index]) for index in range(split.count)]
    record_hashes = [
        _record_sha256(inputs[index], labels[index]) for index in range(split.count)
    ]
    files: dict[str, object] = {}
    for path in sorted(split_dir.iterdir()):
        if path.is_file():
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
    manifest: dict[str, object] = {
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "builder": BUILDER_NAME,
        "empty_cells": {
            "max": max_empty_cells,
            "min": min_empty_cells,
        },
        "files": files,
        "generated_count": split.count,
        "generation": generation_stats,
        "input_sha256s": input_hashes,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "ordered_input_sha256": _ordered_sha256(input_hashes),
        "ordered_record_sha256": _ordered_sha256(record_hashes),
        "record_sha256s": record_hashes,
        "require_unique_solution": True,
        "seed": split.seed,
        "split": split.name,
    }
    manifest_path = root / "manifests" / f"{split.name}.json"
    encoded = _json_bytes(manifest)
    _write_bytes(manifest_path, encoded)
    return manifest, _sha256_bytes(encoded)


def _build_config(spec: BuildSpec) -> dict[str, object]:
    return {
        "build_schema_version": BUILD_SCHEMA_VERSION,
        "builder": BUILDER_NAME,
        "domain": "sudoku_4x4",
        "empty_cells": {
            "max": spec.max_empty_cells,
            "min": spec.min_empty_cells,
        },
        "require_unique_inputs": True,
        "require_unique_records": True,
        "require_unique_solution": True,
        "producer_git_commit": spec.producer_commit,
        "runtime": {
            "numpy_version": np.__version__,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "seed": spec.splits[0].seed,
        "split_order": [split.name for split in spec.splits],
        "splits": {
            split.name: {"count": split.count, "seed": split.seed}
            for split in spec.splits
        },
    }


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            lines.append(f"{_file_sha256(path)}  {path.relative_to(root)}\n")
    _write_bytes(root / "CHECKSUMS.sha256", "".join(lines).encode("ascii"))


def _verify_producer_source_matches_runtime(producer_repo_root: str | Path) -> None:
    producer_root = Path(producer_repo_root).expanduser().resolve()
    assert_git_files_match_head(producer_root, list(PRODUCER_SOURCE_PATHS))
    runtime_sources = {
        "dataset/build_4x4_sudoku.py": Path(
            build_4x4_sudoku_module.__file__
        ).resolve(),
        "dataset/build_iclr_confirmatory_4x4.py": Path(__file__).resolve(),
        "utils/run_identity.py": Path(run_identity_module.__file__).resolve(),
        "utils/dataset_provenance.py": Path(
            dataset_provenance_module.__file__
        ).resolve(),
    }
    for relative_path, runtime_path in runtime_sources.items():
        producer_path = producer_root / relative_path
        if not runtime_path.is_file() or not producer_path.is_file():
            raise DatasetBuildError(
                f"Producer/runtime source {relative_path!r} is missing."
            )
        if producer_path.read_bytes() != runtime_path.read_bytes():
            raise DatasetBuildError(
                f"Producer source {relative_path!r} differs from the running binary."
            )


def _publish_directory_no_replace(stage: Path, output: Path) -> None:
    """Atomically publish a directory without replacing any existing path."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise DatasetBuildError(
            "This host has no renameat2 no-replace primitive for dataset publication."
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(stage),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise DatasetBuildError(
            f"Refusing to replace output that appeared during generation: {output}."
        )
    raise DatasetBuildError(
        f"Atomic dataset publication failed: {os.strerror(error_number)}."
    )


def build_dataset(output_dir: str | Path, spec: BuildSpec) -> dict[str, object]:
    """Generate, verify, and atomically publish one immutable dataset tree."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise DatasetBuildError(f"Refusing to overwrite existing output {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    try:
        config = _build_config(spec)
        config_bytes = _json_bytes(config)
        _write_bytes(stage / "build_config.json", config_bytes)
        _write_json(stage / "identifiers.json", ["<blank>"])

        seen_inputs: set[str] = set()
        seen_records: set[str] = set()
        split_manifest_hashes: dict[str, str] = {}
        for split in spec.splits:
            records, stats = _generate_split(
                split,
                min_empty_cells=spec.min_empty_cells,
                max_empty_cells=spec.max_empty_cells,
                seen_inputs=seen_inputs,
                seen_records=seen_records,
            )
            _, manifest_sha256 = _write_split(
                stage,
                split,
                records,
                stats,
                min_empty_cells=spec.min_empty_cells,
                max_empty_cells=spec.max_empty_cells,
            )
            split_manifest_hashes[split.name] = manifest_sha256

        dataset_manifest = {
            "build_config_sha256": _sha256_bytes(config_bytes),
            "build_schema_version": BUILD_SCHEMA_VERSION,
            "builder": BUILDER_NAME,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "pairwise_input_overlap_count": 0,
            "pairwise_record_overlap_count": 0,
            "split_manifests": {
                split.name: {
                    "path": f"manifests/{split.name}.json",
                    "sha256": split_manifest_hashes[split.name],
                }
                for split in spec.splits
            },
        }
        _write_json(stage / "corpus_manifest.json", dataset_manifest)
        _write_checksums(stage)
        verified = verify_dataset(stage)
        _publish_directory_no_replace(stage, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return verified
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _verify_file_manifest(root: Path, manifest: dict[str, object]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise DatasetBuildError("Split manifest has no file inventory.")
    for relative, raw_entry in files.items():
        if not isinstance(relative, str) or not isinstance(raw_entry, dict):
            raise DatasetBuildError("Split file inventory is malformed.")
        path = root / relative
        if not path.is_file():
            raise DatasetBuildError(f"Manifest file is absent: {relative}.")
        if raw_entry.get("bytes") != path.stat().st_size:
            raise DatasetBuildError(f"Manifest byte count differs for {relative}.")
        if raw_entry.get("sha256") != _file_sha256(path):
            raise DatasetBuildError(f"Manifest SHA-256 differs for {relative}.")


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "CHECKSUMS.sha256"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or relative in expected:
            raise DatasetBuildError("CHECKSUMS.sha256 is malformed.")
        expected[relative] = digest
    actual = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if expected != actual:
        raise DatasetBuildError("CHECKSUMS.sha256 does not match the dataset tree.")


def verify_dataset(root_dir: str | Path) -> dict[str, object]:
    """Verify file hashes, record semantics, order, and cross-split disjointness."""

    root = Path(root_dir).expanduser().resolve()
    config_path = root / "build_config.json"
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise DatasetBuildError("Build configuration must be a JSON object.")
    if config.get("builder") != BUILDER_NAME:
        raise DatasetBuildError("Unexpected dataset builder name.")
    if config.get("build_schema_version") != BUILD_SCHEMA_VERSION:
        raise DatasetBuildError("Unexpected dataset build schema.")
    if (
        config.get("domain") != "sudoku_4x4"
        or config.get("require_unique_inputs") is not True
        or config.get("require_unique_records") is not True
        or config.get("require_unique_solution") is not True
    ):
        raise DatasetBuildError("Build configuration weakens the dataset contract.")
    producer_commit = config.get("producer_git_commit")
    if not isinstance(producer_commit, str) or len(producer_commit) != 40 or any(
        character not in "0123456789abcdef" for character in producer_commit
    ):
        raise DatasetBuildError("Dataset producer commit is invalid.")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "numpy_version",
        "python_implementation",
        "python_version",
    } or not all(isinstance(value, str) and value for value in runtime.values()):
        raise DatasetBuildError("Dataset build runtime metadata is invalid.")
    split_order = config.get("split_order")
    splits = config.get("splits")
    if (
        not isinstance(split_order, list)
        or not split_order
        or not all(isinstance(name, str) and name for name in split_order)
        or not isinstance(splits, dict)
    ):
        raise DatasetBuildError("Build configuration has no split contract.")
    primary_split = splits.get(split_order[0])
    if not isinstance(primary_split, dict) or config.get("seed") != primary_split.get(
        "seed"
    ):
        raise DatasetBuildError("Primary generation seed does not match train split.")
    empty_cells = config.get("empty_cells")
    if not isinstance(empty_cells, dict):
        raise DatasetBuildError("Build configuration has no difficulty range.")
    min_empty = empty_cells.get("min")
    max_empty = empty_cells.get("max")
    if not isinstance(min_empty, int) or not isinstance(max_empty, int):
        raise DatasetBuildError("Build configuration difficulty is invalid.")

    dataset_manifest = _load_json(root / "corpus_manifest.json")
    if not isinstance(dataset_manifest, dict):
        raise DatasetBuildError("Dataset manifest must be a JSON object.")
    if (
        dataset_manifest.get("builder") != BUILDER_NAME
        or dataset_manifest.get("build_schema_version") != BUILD_SCHEMA_VERSION
        or dataset_manifest.get("manifest_schema_version")
        != MANIFEST_SCHEMA_VERSION
    ):
        raise DatasetBuildError("Dataset manifest contract differs.")
    if dataset_manifest.get("build_config_sha256") != _file_sha256(config_path):
        raise DatasetBuildError("Dataset manifest build-config hash differs.")

    raw_manifest_refs = dataset_manifest.get("split_manifests")
    if not isinstance(raw_manifest_refs, dict):
        raise DatasetBuildError("Dataset manifest has no split-manifest map.")

    all_input_hashes: set[str] = set()
    all_record_hashes: set[str] = set()
    encoded_solutions_by_split: dict[str, np.ndarray] = {}
    verified_split_hashes: dict[str, str] = {}
    for split_name in split_order:
        if not isinstance(split_name, str) or split_name not in splits:
            raise DatasetBuildError("Split order and split configuration disagree.")
        split_config = splits[split_name]
        if not isinstance(split_config, dict):
            raise DatasetBuildError("Split configuration is malformed.")
        count = split_config.get("count")
        seed = split_config.get("seed")
        if not isinstance(count, int) or count < 1 or not isinstance(seed, int):
            raise DatasetBuildError("Split count or seed is invalid.")

        manifest_path = root / "manifests" / f"{split_name}.json"
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise DatasetBuildError("Split manifest must be a JSON object.")
        manifest_sha256 = _file_sha256(manifest_path)
        manifest_ref = raw_manifest_refs.get(split_name)
        if not isinstance(manifest_ref, dict) or manifest_ref != {
            "path": f"manifests/{split_name}.json",
            "sha256": manifest_sha256,
        }:
            raise DatasetBuildError("Dataset and split manifests disagree.")
        if (
            manifest.get("builder") != BUILDER_NAME
            or manifest.get("build_schema_version") != BUILD_SCHEMA_VERSION
            or manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("split") != split_name
            or manifest.get("seed") != seed
            or manifest.get("generated_count") != count
            or manifest.get("require_unique_solution") is not True
        ):
            raise DatasetBuildError(f"Split manifest contract differs for {split_name}.")
        if manifest.get("empty_cells") != empty_cells:
            raise DatasetBuildError(f"Split difficulty differs for {split_name}.")
        _verify_file_manifest(root, manifest)

        split_dir = root / split_name
        inputs = np.load(split_dir / "all__inputs.npy", allow_pickle=False)
        labels = np.load(split_dir / "all__labels.npy", allow_pickle=False)
        identifiers = np.load(
            split_dir / "all__puzzle_identifiers.npy", allow_pickle=False
        )
        puzzle_indices = np.load(
            split_dir / "all__puzzle_indices.npy", allow_pickle=False
        )
        group_indices = np.load(
            split_dir / "all__group_indices.npy", allow_pickle=False
        )
        if inputs.dtype != np.int32 or labels.dtype != np.int32:
            raise DatasetBuildError(f"Split {split_name} has an unexpected dtype.")
        if inputs.shape != (count, 16) or labels.shape != (count, 16):
            raise DatasetBuildError(f"Split {split_name} has an unexpected shape.")
        if not np.array_equal(identifiers, np.arange(count, dtype=np.int32)):
            raise DatasetBuildError(f"Split {split_name} identifiers are not canonical.")
        expected_indices = np.arange(count + 1, dtype=np.int32)
        if not np.array_equal(puzzle_indices, expected_indices) or not np.array_equal(
            group_indices, expected_indices
        ):
            raise DatasetBuildError(f"Split {split_name} indices are not canonical.")
        if _load_json(split_dir / "dataset.json") != _metadata(count):
            raise DatasetBuildError(f"Split {split_name} metadata differs.")

        input_hashes: list[str] = []
        record_hashes: list[str] = []
        for index in range(count):
            encoded_input = inputs[index]
            encoded_solution = labels[index]
            if np.any((encoded_input < 1) | (encoded_input > 5)) or np.any(
                (encoded_solution < 2) | (encoded_solution > 5)
            ):
                raise DatasetBuildError(f"Split {split_name} has invalid tokens.")
            puzzle = np.where(
                encoded_input.reshape(4, 4) == 1,
                0,
                encoded_input.reshape(4, 4) - 1,
            ).astype(np.int32)
            solution = (encoded_solution.reshape(4, 4) - 1).astype(np.int32)
            empty_count = int(np.count_nonzero(puzzle == 0))
            if not (min_empty <= empty_count <= max_empty):
                raise DatasetBuildError(f"Split {split_name} has wrong difficulty.")
            if not _is_valid_solution(solution):
                raise DatasetBuildError(f"Split {split_name} has an invalid solution.")
            if np.any((puzzle != 0) & (puzzle != solution)):
                raise DatasetBuildError(f"Split {split_name} clues disagree with solution.")
            if count_solutions(puzzle, limit=2) != 1:
                raise DatasetBuildError(f"Split {split_name} puzzle is not unique.")
            input_hashes.append(_input_sha256(encoded_input))
            record_hashes.append(_record_sha256(encoded_input, encoded_solution))

        if (
            manifest.get("input_sha256s") != input_hashes
            or manifest.get("record_sha256s") != record_hashes
            or manifest.get("ordered_input_sha256") != _ordered_sha256(input_hashes)
            or manifest.get("ordered_record_sha256")
            != _ordered_sha256(record_hashes)
        ):
            raise DatasetBuildError(f"Split {split_name} ordered hashes differ.")
        if len(set(input_hashes)) != count or len(set(record_hashes)) != count:
            raise DatasetBuildError(f"Split {split_name} contains duplicates.")
        if all_input_hashes.intersection(input_hashes):
            raise DatasetBuildError(f"Split {split_name} overlaps an earlier input set.")
        if all_record_hashes.intersection(record_hashes):
            raise DatasetBuildError(f"Split {split_name} overlaps an earlier record set.")
        all_input_hashes.update(input_hashes)
        all_record_hashes.update(record_hashes)
        encoded_solutions_by_split[split_name] = labels
        verified_split_hashes[split_name] = manifest_sha256

    if set(split_order) != set(splits):
        raise DatasetBuildError("Unordered split configuration is present.")
    if set(raw_manifest_refs) != set(split_order):
        raise DatasetBuildError("Dataset manifest contains an unknown split.")
    if dataset_manifest.get("pairwise_input_overlap_count") != 0 or dataset_manifest.get(
        "pairwise_record_overlap_count"
    ) != 0:
        raise DatasetBuildError("Dataset manifest reports nonzero overlap.")
    _verify_checksums(root)
    return {
        "build_config_sha256": _file_sha256(config_path),
        "corpus_manifest_sha256": _file_sha256(root / "corpus_manifest.json"),
        "solution_grid_audit": audit_solution_grid_coverage(
            encoded_solutions_by_split
        ),
        **{
            f"{name}_manifest_sha256": digest
            for name, digest in verified_split_hashes.items()
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=26080301)
    parser.add_argument("--validation-seed", type=int, default=26080302)
    parser.add_argument("--test-seed", type=int, default=26080303)
    parser.add_argument("--min-empty-cells", type=int, default=6)
    parser.add_argument("--max-empty-cells", type=int, default=8)
    parser.add_argument(
        "--producer-repo-root",
        help="Clean Git repository whose committed builder creates the dataset.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_only:
        result = verify_dataset(args.output_dir)
    else:
        if args.producer_repo_root is None:
            raise DatasetBuildError(
                "Building requires an explicit --producer-repo-root."
            )
        try:
            producer = discover_clean_git_source(args.producer_repo_root)
            _verify_producer_source_matches_runtime(args.producer_repo_root)
        except RunIdentityError as exc:
            raise DatasetBuildError(
                "Dataset producer source is not a clean committed tree."
            ) from exc
        spec = BuildSpec(
            splits=(
                SplitSpec("train", args.train_count, args.train_seed),
                SplitSpec("validation", args.validation_count, args.validation_seed),
                SplitSpec("test", args.test_count, args.test_seed),
            ),
            producer_commit=str(producer["git_commit"]),
            min_empty_cells=args.min_empty_cells,
            max_empty_cells=args.max_empty_cells,
        )
        result = build_dataset(args.output_dir, spec)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
