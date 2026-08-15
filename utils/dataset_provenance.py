"""Deterministic dataset identity for evaluation and exact checkpoint resume."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict


DATASET_PROVENANCE_SCHEMA_VERSION = 1


class DatasetProvenanceError(RuntimeError):
    """Raised when checkpoint dataset identity is incomplete or mismatched."""


class DatasetSourceBuildMetadata(TypedDict):
    source_name: str
    builder_name: str
    builder_version: str | int
    generation_seed: int | None
    build_config_sha256: str | None
    split_generation_seeds: NotRequired[dict[str, int]]
    producer_git_commit: NotRequired[str]
    producer_launcher_sha256: NotRequired[str]
    producer_runtime_sha256: NotRequired[str]
    producer_source_manifest_sha256: NotRequired[str]


def dataset_source_build_metadata(
    dataset_paths: Sequence[str],
) -> list[DatasetSourceBuildMetadata]:
    """Read anonymous generator identity from each materialized dataset root."""

    sources: list[DatasetSourceBuildMetadata] = []
    for raw_path in dataset_paths:
        root = Path(raw_path)
        config_path = root / "build_config.json"
        source: DatasetSourceBuildMetadata = {
            "source_name": root.name,
            "builder_name": "unrecorded",
            "builder_version": "unrecorded",
            "generation_seed": None,
            "build_config_sha256": None,
        }
        if config_path.is_file():
            encoded = config_path.read_bytes()
            try:
                parsed = json.loads(encoded.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetProvenanceError(
                    f"Invalid dataset build config for source {root.name!r}."
                ) from exc
            config = _require_mapping(
                parsed,
                path=f"dataset_source[{root.name}].build_config",
            )
            builder_name = config.get("builder")
            builder_version = config.get("build_schema_version")
            generation_seed = config.get("seed")
            raw_splits = config.get("splits")
            split_generation_seeds: dict[str, int] | None = None
            if raw_splits is not None:
                if not isinstance(raw_splits, Mapping) or not all(
                    isinstance(name, str) and isinstance(value, Mapping)
                    for name, value in raw_splits.items()
                ):
                    raise DatasetProvenanceError(
                        f"Dataset source {root.name!r} has invalid split metadata."
                    )
                split_generation_seeds = {}
                for name, value in raw_splits.items():
                    split_seed = value.get("seed")
                    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
                        raise DatasetProvenanceError(
                            f"Dataset source {root.name!r} has invalid seed for "
                            f"split {name!r}."
                        )
                    split_generation_seeds[name] = split_seed
            producer_git_commit = config.get("producer_git_commit")
            producer_launcher_sha256: str | None = None
            producer_runtime_sha256: str | None = None
            producer_source_manifest_sha256: str | None = None
            producer_source = config.get("producer_source")
            if producer_source is not None:
                if (
                    not isinstance(producer_source, Mapping)
                    or set(producer_source)
                    != {
                        "git_commit",
                        "launcher_sha256",
                        "runtime_sha256",
                        "source_manifest_sha256",
                    }
                ):
                    raise DatasetProvenanceError(
                        f"Dataset source {root.name!r} has invalid producer identity."
                    )
                producer_git_commit = producer_source["git_commit"]
                producer_launcher_sha256 = producer_source["launcher_sha256"]
                producer_runtime_sha256 = producer_source["runtime_sha256"]
                producer_source_manifest_sha256 = producer_source[
                    "source_manifest_sha256"
                ]
            if producer_git_commit is not None and (
                not isinstance(producer_git_commit, str)
                or len(producer_git_commit) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in producer_git_commit
                )
            ):
                raise DatasetProvenanceError(
                    f"Dataset source {root.name!r} has invalid producer commit."
                )
            for label, digest in (
                ("launcher", producer_launcher_sha256),
                ("runtime", producer_runtime_sha256),
                ("source manifest", producer_source_manifest_sha256),
            ):
                if digest is not None and (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest
                    )
                ):
                    raise DatasetProvenanceError(
                        f"Dataset source {root.name!r} has invalid producer {label}."
                    )
            if not isinstance(builder_name, str) or not builder_name:
                raise DatasetProvenanceError(
                    f"Dataset source {root.name!r} has no recorded builder."
                )
            if (
                isinstance(builder_version, bool)
                or not isinstance(builder_version, (str, int))
                or builder_version == ""
            ):
                raise DatasetProvenanceError(
                    f"Dataset source {root.name!r} has no valid builder version."
                )
            if generation_seed is not None and (
                isinstance(generation_seed, bool)
                or not isinstance(generation_seed, int)
            ):
                raise DatasetProvenanceError(
                    f"Dataset source {root.name!r} has no valid generation seed."
                )
            source.update(
                {
                    "builder_name": builder_name,
                    "builder_version": builder_version,
                    "generation_seed": generation_seed,
                    "build_config_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
            if split_generation_seeds is not None:
                source["split_generation_seeds"] = split_generation_seeds
            if producer_git_commit is not None:
                source["producer_git_commit"] = producer_git_commit
            if producer_launcher_sha256 is not None:
                source["producer_launcher_sha256"] = producer_launcher_sha256
            if producer_runtime_sha256 is not None:
                source["producer_runtime_sha256"] = producer_runtime_sha256
            if producer_source_manifest_sha256 is not None:
                source["producer_source_manifest_sha256"] = (
                    producer_source_manifest_sha256
                )
        sources.append(source)
    if not sources:
        raise ValueError("At least one dataset source is required.")
    return sources


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def sample_sha256(inputs: Any, solution: Any) -> str:
    payload = {
        "inputs": _to_list(inputs),
        "solution": _to_list(solution),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_sha256(inputs: Any) -> str:
    encoded = json.dumps(
        _to_list(inputs),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_pool_sha256(
    inputs: Sequence[Any],
    solutions: Sequence[Any],
    count: int,
) -> str:
    if count < 1:
        raise ValueError("Evaluation pool count must be positive.")
    if len(inputs) < count or len(solutions) < count:
        raise ValueError(
            f"Evaluation pool has fewer than {count} records: "
            f"inputs={len(inputs)}, solutions={len(solutions)}."
        )
    digest = hashlib.sha256()
    for index in range(count):
        digest.update(sample_sha256(inputs[index], solutions[index]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def ordered_record_sha256(record_sha256s: Sequence[str]) -> str:
    """Hash an ordered fingerprint list with an unambiguous record delimiter."""

    if not record_sha256s:
        raise ValueError("An ordered record set must be nonempty.")
    digest = hashlib.sha256()
    for fingerprint in record_sha256s:
        _validate_sha256(fingerprint, label="record fingerprint")
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_sample_sha256s(dataset: Any, count: int | None = None) -> list[str]:
    if count is None:
        count = len(dataset)
    if count < 1 or len(dataset) < count:
        raise ValueError(
            f"Dataset has {len(dataset)} samples; requested provenance count={count}."
        )
    fingerprints = []
    for index in range(count):
        sample = dataset[index]
        if not isinstance(sample, dict) or "inputs" not in sample:
            raise TypeError("Dataset provenance requires dictionary samples with inputs.")
        fingerprints.append(
            sample_sha256(
                sample["inputs"],
                sample.get("solution", sample.get("labels")),
            )
        )
    return fingerprints


def dataset_input_sha256s(dataset: Any) -> list[str]:
    fingerprints = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, dict) or "inputs" not in sample:
            raise TypeError("Dataset provenance requires dictionary samples with inputs.")
        fingerprints.append(input_sha256(sample["inputs"]))
    return fingerprints


def dataset_puzzle_identifier_sha256s(dataset: Any) -> list[str]:
    fingerprints = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, dict) or "puzzle_identifiers" not in sample:
            raise TypeError(
                "Dataset provenance requires puzzle_identifiers on every sample."
            )
        encoded = json.dumps(
            _to_list(sample["puzzle_identifiers"]),
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprints.append(hashlib.sha256(encoded).hexdigest())
    return fingerprints


def sequence_input_sha256s(inputs: Sequence[Any]) -> list[str]:
    return [input_sha256(value) for value in inputs]


def dataset_pool_sha256(dataset: Any, count: int) -> str:
    return ordered_record_sha256(dataset_sample_sha256s(dataset, count=count))


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DatasetProvenanceError(f"{label} must be a 64-character SHA-256.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DatasetProvenanceError(f"{label} is not hexadecimal.") from exc
    return value.lower()


def _canonical_json_value(value: object, *, path: str) -> object:
    """Return a JSON-safe, order-stable value or reject ambiguous state."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetProvenanceError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise DatasetProvenanceError(f"{path} has a non-string key.")
        for key in sorted(keys):
            canonical[key] = _canonical_json_value(
                value[key],
                path=f"{path}.{key}",
            )
        return canonical
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise DatasetProvenanceError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _record_manifest(record_sha256s: Sequence[str]) -> dict[str, object]:
    fingerprints = [
        _validate_sha256(value, label="record fingerprint")
        for value in record_sha256s
    ]
    return {
        "count": len(fingerprints),
        "ordered_sha256": ordered_record_sha256(fingerprints),
        "record_sha256s": fingerprints,
    }


def build_dataset_provenance(
    *,
    builder_name: str,
    builder_version: str | int,
    generation_seed: int | None,
    train_record_sha256s: Sequence[str],
    eval_record_sha256s: Sequence[str],
    train_split: str,
    eval_split: str,
    environment_config: Mapping[str, object],
    action_mask_config: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build the complete versioned identity required for exact resume."""

    payload: dict[str, Any] = {
        "provenance_schema_version": DATASET_PROVENANCE_SCHEMA_VERSION,
        "dataset_builder": {
            "name": builder_name,
            "version": builder_version,
        },
        "generation_seed": generation_seed,
        "splits": {
            "train": train_split,
            "eval": eval_split,
        },
        "ordered_records": {
            "train": _record_manifest(train_record_sha256s),
            "eval": _record_manifest(eval_record_sha256s),
        },
        "environment_config": dict(environment_config),
        "action_mask_config": dict(action_mask_config),
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    return validate_dataset_provenance(payload)


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetProvenanceError(f"{path} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise DatasetProvenanceError(f"{path} must use string keys.")
    return value


def _validate_record_manifest(value: object, *, path: str) -> None:
    manifest = _require_mapping(value, path=path)
    count = manifest.get("count")
    fingerprints = manifest.get("record_sha256s")
    digest = manifest.get("ordered_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise DatasetProvenanceError(f"{path}.count must be a positive integer.")
    if not isinstance(fingerprints, list) or len(fingerprints) != count:
        raise DatasetProvenanceError(
            f"{path}.record_sha256s length must equal count={count}."
        )
    normalized = [
        _validate_sha256(fingerprint, label=f"{path}.record_sha256s[{index}]")
        for index, fingerprint in enumerate(fingerprints)
    ]
    expected_digest = ordered_record_sha256(normalized)
    if _validate_sha256(digest, label=f"{path}.ordered_sha256") != expected_digest:
        raise DatasetProvenanceError(
            f"{path}.ordered_sha256 does not match its ordered record list."
        )


def validate_dataset_provenance(
    provenance: Mapping[str, object],
) -> dict[str, Any]:
    """Validate and canonicalize an exact-resume provenance payload."""

    canonical_value = _canonical_json_value(provenance, path="dataset_provenance")
    canonical = dict(
        _require_mapping(canonical_value, path="dataset_provenance")
    )
    schema_version = canonical.get("provenance_schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != DATASET_PROVENANCE_SCHEMA_VERSION
    ):
        raise DatasetProvenanceError(
            "Unsupported dataset provenance schema version "
            f"{schema_version!r}; expected {DATASET_PROVENANCE_SCHEMA_VERSION}."
        )

    builder = _require_mapping(
        canonical.get("dataset_builder"),
        path="dataset_provenance.dataset_builder",
    )
    if not isinstance(builder.get("name"), str) or not builder["name"]:
        raise DatasetProvenanceError("dataset_builder.name must be nonempty.")
    version = builder.get("version")
    if isinstance(version, bool) or not isinstance(version, (str, int)) or version == "":
        raise DatasetProvenanceError(
            "dataset_builder.version must be a nonempty string or integer."
        )

    seed = canonical.get("generation_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DatasetProvenanceError("generation_seed must be an integer or null.")

    splits = _require_mapping(
        canonical.get("splits"),
        path="dataset_provenance.splits",
    )
    for split in ("train", "eval"):
        if not isinstance(splits.get(split), str) or not splits[split]:
            raise DatasetProvenanceError(f"splits.{split} must be nonempty.")

    records = _require_mapping(
        canonical.get("ordered_records"),
        path="dataset_provenance.ordered_records",
    )
    for split in ("train", "eval"):
        _validate_record_manifest(
            records.get(split),
            path=f"dataset_provenance.ordered_records.{split}",
        )

    for config_name in ("environment_config", "action_mask_config"):
        config = _require_mapping(
            canonical.get(config_name),
            path=f"dataset_provenance.{config_name}",
        )
        if not config:
            raise DatasetProvenanceError(f"{config_name} must be nonempty.")

    metadata = canonical.get("metadata")
    if metadata is not None:
        _require_mapping(metadata, path="dataset_provenance.metadata")
    return canonical


def _collect_mismatches(
    saved: object,
    expected: object,
    *,
    path: str,
    mismatches: list[str],
    limit: int,
) -> None:
    if len(mismatches) >= limit:
        return
    if isinstance(saved, Mapping) and isinstance(expected, Mapping):
        for key in sorted(set(saved) | set(expected)):
            if key not in saved:
                mismatches.append(f"{path}.{key}: missing from checkpoint")
            elif key not in expected:
                mismatches.append(f"{path}.{key}: absent from current run")
            else:
                _collect_mismatches(
                    saved[key],
                    expected[key],
                    path=f"{path}.{key}",
                    mismatches=mismatches,
                    limit=limit,
                )
            if len(mismatches) >= limit:
                return
        return
    if isinstance(saved, list) and isinstance(expected, list):
        if len(saved) != len(expected):
            mismatches.append(
                f"{path}.length: checkpoint={len(saved)!r}, current={len(expected)!r}"
            )
            return
        for index, (saved_item, expected_item) in enumerate(zip(saved, expected)):
            _collect_mismatches(
                saved_item,
                expected_item,
                path=f"{path}[{index}]",
                mismatches=mismatches,
                limit=limit,
            )
            if len(mismatches) >= limit:
                return
        return
    if saved != expected:
        mismatches.append(f"{path}: checkpoint={saved!r}, current={expected!r}")


def assert_matching_dataset_provenance(
    saved: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Fail with bounded field-level differences before checkpoint mutation."""

    try:
        saved_canonical = validate_dataset_provenance(saved)
    except DatasetProvenanceError as exc:
        raise DatasetProvenanceError(
            f"Saved dataset provenance is invalid: {exc}"
        ) from exc
    try:
        expected_canonical = validate_dataset_provenance(expected)
    except DatasetProvenanceError as exc:
        raise DatasetProvenanceError(
            f"Current dataset provenance is invalid: {exc}"
        ) from exc

    mismatches: list[str] = []
    _collect_mismatches(
        saved_canonical,
        expected_canonical,
        path="dataset_provenance",
        mismatches=mismatches,
        limit=8,
    )
    if mismatches:
        raise DatasetProvenanceError(
            "Dataset provenance mismatch; state was not restored: "
            + "; ".join(mismatches)
        )
