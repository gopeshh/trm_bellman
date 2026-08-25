#!/usr/bin/env fbpython
"""Materialize and validate the pre-outcome v2 evaluation populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL_ID = "policy-improvement-v2-20260818"
POPULATIONS_SCHEMA_NAME = "policy_improvement_populations_v2"
POPULATIONS_SCHEMA_VERSION = 1
DATASET_NAME = "policy-improvement-hard-4x4-v1"
STAGE0_SELECTION_NAMESPACE = "upi-trm-policy-improvement-v2-stage0-smoke:"
VALIDATION_PARTITION_NAMESPACE = "upi-trm-policy-improvement-v2-validation-partition:"
SELECTION_ALGORITHM = "sha256_namespace_record_sha256_lexicographic_v1"
# The Stage 0 training population is the deterministic complement of the frozen
# Stage 0 evaluation population inside the same registered train split. It is
# ordered by original train index, not by selection score, because it is the
# pool a Stage 0 run trains on rather than a score-ranked selection.
COMPLEMENT_SELECTION_ALGORITHM = (
    "sha256_namespace_record_sha256_lexicographic_complement_v1"
)
SELECTION_TIE_BREAK = "original_index_ascending"
STAGE0_EVALUATION_POPULATION_ID = "stage0_smoke"
STAGE0_TRAINING_POPULATION_ID = "train_minus_stage0_smoke"
TRAIN_SPLIT_RECORD_COUNT = 1024
STAGE0_EVALUATION_RECORD_COUNT = 8
STAGE0_TRAINING_RECORD_COUNT = (
    TRAIN_SPLIT_RECORD_COUNT - STAGE0_EVALUATION_RECORD_COUNT
)
REGISTERED_POPULATION_IDS = (
    STAGE0_EVALUATION_POPULATION_ID,
    STAGE0_TRAINING_POPULATION_ID,
    "validation_select",
    "validation_bridge",
)
_SCORE_ORDERED = "score"
_INDEX_ORDERED = "index"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicyImprovementPopulationError(RuntimeError):
    """Raised when a registered population is incomplete or inconsistent."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON encoding used by v2 identity documents."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PolicyImprovementPopulationError(
            "Population document is not canonical JSON."
        ) from exc
    if json.loads(encoded.decode("ascii")) != value:
        raise PolicyImprovementPopulationError(
            "Population document does not round-trip through canonical JSON."
        )
    return encoded


def load_strict_json(path: str | Path) -> Any:
    """Load ASCII JSON while rejecting duplicate object keys."""

    def pairs(items: Iterable[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise PolicyImprovementPopulationError(f"Duplicate JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="ascii"),
            object_pairs_hook=pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementPopulationError(
            "Population input is not strict ASCII JSON."
        ) from exc


def _sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PolicyImprovementPopulationError(f"{path} is not a SHA-256 digest.")
    return value


def ordered_sha256(values: Sequence[str]) -> str:
    """Hash an ordered list with the dataset manifest's newline delimiter."""

    if not values:
        raise PolicyImprovementPopulationError("Ordered identity list is empty.")
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        digest.update(_sha256(value, path=f"ordered[{index}]").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def derive_population_score(namespace: str, record_sha256: str) -> str:
    """Derive one outcome-independent population ordering score."""

    if not isinstance(namespace, str) or not namespace.isascii() or not namespace:
        raise PolicyImprovementPopulationError("Selection namespace is invalid.")
    record = _sha256(record_sha256, path="record_sha256")
    return hashlib.sha256((namespace + record).encode("ascii")).hexdigest()


def population_binding_sha256(population: Mapping[str, object]) -> str:
    """Bind every population field except the digest itself."""

    material = dict(population)
    material.pop("binding_sha256", None)
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _population(
    *,
    population_id: str,
    split: str,
    namespace: str,
    indices: Sequence[int],
    record_sha256s: Sequence[str],
    input_sha256s: Sequence[str],
    algorithm: str = SELECTION_ALGORITHM,
) -> dict[str, object]:
    scores = [derive_population_score(namespace, value) for value in record_sha256s]
    result: dict[str, object] = {
        "population_id": population_id,
        "split": split,
        "count": len(indices),
        "selection_namespace": namespace,
        "selection_algorithm": algorithm,
        "tie_break": SELECTION_TIE_BREAK,
        "indices": list(indices),
        "record_sha256s": list(record_sha256s),
        "input_sha256s": list(input_sha256s),
        "selection_scores": scores,
        "ordered_record_sha256": ordered_sha256(record_sha256s),
        "ordered_input_sha256": ordered_sha256(input_sha256s),
    }
    result["binding_sha256"] = population_binding_sha256(result)
    return result


def materialize_v2_populations(
    *,
    train_record_sha256s: Sequence[str],
    train_input_sha256s: Sequence[str],
    validation_record_sha256s: Sequence[str],
    validation_input_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Freeze Stage 0 and the disjoint validation populations before outcomes."""

    train_records = list(train_record_sha256s)
    train_inputs = list(train_input_sha256s)
    validation_records = list(validation_record_sha256s)
    validation_inputs = list(validation_input_sha256s)
    if len(train_records) != 1024 or len(train_inputs) != 1024:
        raise PolicyImprovementPopulationError(
            "V2 population materialization requires 1024 train identities."
        )
    if len(validation_records) != 256 or len(validation_inputs) != 256:
        raise PolicyImprovementPopulationError(
            "V2 population materialization requires 256 validation identities."
        )
    for label, values in (
        ("train records", train_records),
        ("train inputs", train_inputs),
        ("validation records", validation_records),
        ("validation inputs", validation_inputs),
    ):
        for index, value in enumerate(values):
            _sha256(value, path=f"{label}[{index}]")
    if len(set(train_records)) != len(train_records):
        raise PolicyImprovementPopulationError(
            "Train record identities are not unique."
        )
    if len(set(validation_records)) != len(validation_records):
        raise PolicyImprovementPopulationError(
            "Validation record identities are not unique."
        )

    stage0_indices = sorted(
        range(len(train_records)),
        key=lambda index: (
            derive_population_score(STAGE0_SELECTION_NAMESPACE, train_records[index]),
            index,
        ),
    )[:STAGE0_EVALUATION_RECORD_COUNT]
    # Everything the Stage 0 evaluation population did not take, in original
    # train index order. Stage 0 trains on exactly this complement so the
    # schema-v5 train/evaluation disjointness invariant holds without a
    # Stage 0 special case.
    stage0_training_indices = [
        index
        for index in range(len(train_records))
        if index not in set(stage0_indices)
    ]
    validation_order = sorted(
        range(len(validation_records)),
        key=lambda index: (
            derive_population_score(
                VALIDATION_PARTITION_NAMESPACE, validation_records[index]
            ),
            index,
        ),
    )
    populations = {
        "stage0_smoke": _population(
            population_id="stage0_smoke",
            split="train",
            namespace=STAGE0_SELECTION_NAMESPACE,
            indices=stage0_indices,
            record_sha256s=[train_records[index] for index in stage0_indices],
            input_sha256s=[train_inputs[index] for index in stage0_indices],
        ),
        STAGE0_TRAINING_POPULATION_ID: _population(
            population_id=STAGE0_TRAINING_POPULATION_ID,
            split="train",
            namespace=STAGE0_SELECTION_NAMESPACE,
            indices=stage0_training_indices,
            record_sha256s=[
                train_records[index] for index in stage0_training_indices
            ],
            input_sha256s=[train_inputs[index] for index in stage0_training_indices],
            algorithm=COMPLEMENT_SELECTION_ALGORITHM,
        ),
        "validation_select": _population(
            population_id="validation_select",
            split="validation",
            namespace=VALIDATION_PARTITION_NAMESPACE,
            indices=validation_order[:128],
            record_sha256s=[
                validation_records[index] for index in validation_order[:128]
            ],
            input_sha256s=[
                validation_inputs[index] for index in validation_order[:128]
            ],
        ),
        "validation_bridge": _population(
            population_id="validation_bridge",
            split="validation",
            namespace=VALIDATION_PARTITION_NAMESPACE,
            indices=validation_order[128:],
            record_sha256s=[
                validation_records[index] for index in validation_order[128:]
            ],
            input_sha256s=[
                validation_inputs[index] for index in validation_order[128:]
            ],
        ),
    }
    document: dict[str, Any] = {
        "schema_name": POPULATIONS_SCHEMA_NAME,
        "schema_version": POPULATIONS_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_builder_schema_version": 2,
        "dataset_name": DATASET_NAME,
        "split_ordered_record_sha256": {
            "train": ordered_sha256(train_records),
            "validation": ordered_sha256(validation_records),
        },
        "populations": populations,
    }
    return validate_v2_populations(document)


def _validate_population(
    value: object,
    *,
    population_id: str,
    expected_split: str,
    expected_count: int,
    expected_namespace: str,
    expected_algorithm: str = SELECTION_ALGORITHM,
    ordering: str = _SCORE_ORDERED,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} must be an object."
        )
    expected_fields = {
        "population_id",
        "split",
        "count",
        "selection_namespace",
        "selection_algorithm",
        "tie_break",
        "indices",
        "record_sha256s",
        "input_sha256s",
        "selection_scores",
        "ordered_record_sha256",
        "ordered_input_sha256",
        "binding_sha256",
    }
    if set(value) != expected_fields:
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} has an invalid field inventory."
        )
    result = dict(value)
    if (
        result["population_id"] != population_id
        or result["split"] != expected_split
        or result["count"] != expected_count
        or result["selection_namespace"] != expected_namespace
        or result["selection_algorithm"] != expected_algorithm
        or result["tie_break"] != SELECTION_TIE_BREAK
    ):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} registration differs."
        )
    indices = result["indices"]
    records = result["record_sha256s"]
    inputs = result["input_sha256s"]
    scores = result["selection_scores"]
    if not all(isinstance(item, list) for item in (indices, records, inputs, scores)):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} identity lists are malformed."
        )
    if not all(
        len(item) == expected_count for item in (indices, records, inputs, scores)
    ):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} identity list length differs."
        )
    if (
        any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or len(set(indices)) != expected_count
    ):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} indices are invalid."
        )
    checked_records = [
        _sha256(value, path=f"{population_id}.record_sha256s[{index}]")
        for index, value in enumerate(records)
    ]
    checked_inputs = [
        _sha256(value, path=f"{population_id}.input_sha256s[{index}]")
        for index, value in enumerate(inputs)
    ]
    expected_scores = [
        derive_population_score(expected_namespace, value) for value in checked_records
    ]
    if scores != expected_scores:
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} selection scores differ."
        )
    if ordering == _SCORE_ORDERED:
        if list(zip(scores, indices)) != sorted(zip(scores, indices)):
            raise PolicyImprovementPopulationError(
                f"Population {population_id!r} is not in registered score order."
            )
    elif ordering == _INDEX_ORDERED:
        if list(indices) != sorted(indices):
            raise PolicyImprovementPopulationError(
                f"Population {population_id!r} is not in registered index order."
            )
    else:  # pragma: no cover - guarded by the callers below.
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} has an unsupported ordering."
        )
    if result["ordered_record_sha256"] != ordered_sha256(checked_records):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} ordered record identity differs."
        )
    if result["ordered_input_sha256"] != ordered_sha256(checked_inputs):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} ordered input identity differs."
        )
    if result["binding_sha256"] != population_binding_sha256(result):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} binding differs."
        )
    canonical_json_bytes(result)
    return result


def validate_v2_populations(
    value: object,
    *,
    train_record_sha256s: Sequence[str] | None = None,
    train_input_sha256s: Sequence[str] | None = None,
    validation_record_sha256s: Sequence[str] | None = None,
    validation_input_sha256s: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the frozen document, optionally against authenticated manifests."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_name",
        "schema_version",
        "protocol_id",
        "dataset_builder_schema_version",
        "dataset_name",
        "split_ordered_record_sha256",
        "populations",
    }:
        raise PolicyImprovementPopulationError(
            "Population document has an invalid field inventory."
        )
    result = dict(value)
    if (
        result["schema_name"] != POPULATIONS_SCHEMA_NAME
        or result["schema_version"] != POPULATIONS_SCHEMA_VERSION
        or result["protocol_id"] != PROTOCOL_ID
        or result["dataset_builder_schema_version"] != 2
        or result["dataset_name"] != DATASET_NAME
    ):
        raise PolicyImprovementPopulationError("Population document schema differs.")
    split_identity = result["split_ordered_record_sha256"]
    if not isinstance(split_identity, Mapping) or set(split_identity) != {
        "train",
        "validation",
    }:
        raise PolicyImprovementPopulationError("Split identity inventory differs.")
    for split in ("train", "validation"):
        _sha256(split_identity[split], path=f"split_ordered_record_sha256.{split}")
    raw_populations = result["populations"]
    if not isinstance(raw_populations, Mapping) or set(raw_populations) != set(
        REGISTERED_POPULATION_IDS
    ):
        raise PolicyImprovementPopulationError("Population inventory differs.")
    populations = {
        "stage0_smoke": _validate_population(
            raw_populations["stage0_smoke"],
            population_id="stage0_smoke",
            expected_split="train",
            expected_count=STAGE0_EVALUATION_RECORD_COUNT,
            expected_namespace=STAGE0_SELECTION_NAMESPACE,
        ),
        STAGE0_TRAINING_POPULATION_ID: _validate_population(
            raw_populations[STAGE0_TRAINING_POPULATION_ID],
            population_id=STAGE0_TRAINING_POPULATION_ID,
            expected_split="train",
            expected_count=STAGE0_TRAINING_RECORD_COUNT,
            expected_namespace=STAGE0_SELECTION_NAMESPACE,
            expected_algorithm=COMPLEMENT_SELECTION_ALGORITHM,
            ordering=_INDEX_ORDERED,
        ),
        "validation_select": _validate_population(
            raw_populations["validation_select"],
            population_id="validation_select",
            expected_split="validation",
            expected_count=128,
            expected_namespace=VALIDATION_PARTITION_NAMESPACE,
        ),
        "validation_bridge": _validate_population(
            raw_populations["validation_bridge"],
            population_id="validation_bridge",
            expected_split="validation",
            expected_count=128,
            expected_namespace=VALIDATION_PARTITION_NAMESPACE,
        ),
    }
    # The Stage 0 evaluation population and its registered training complement
    # must partition the registered train split exactly, must not overlap, and
    # the complement must be strictly the records the selection did not take.
    stage0 = populations["stage0_smoke"]
    training = populations[STAGE0_TRAINING_POPULATION_ID]
    stage0_indices = list(stage0["indices"])
    training_indices = list(training["indices"])
    if set(stage0_indices) & set(training_indices):
        raise PolicyImprovementPopulationError(
            "Stage 0 evaluation and training populations overlap."
        )
    combined_train = stage0_indices + training_indices
    if len(combined_train) != TRAIN_SPLIT_RECORD_COUNT or set(combined_train) != set(
        range(TRAIN_SPLIT_RECORD_COUNT)
    ):
        raise PolicyImprovementPopulationError(
            "Stage 0 populations do not partition the registered train split."
        )
    stage0_pairs = list(zip(stage0["selection_scores"], stage0_indices))
    training_pairs = list(zip(training["selection_scores"], training_indices))
    if max(stage0_pairs) >= min(training_pairs):
        raise PolicyImprovementPopulationError(
            "Stage 0 training complement is not the unselected remainder."
        )
    train_by_index = dict(
        zip(
            combined_train,
            list(stage0["record_sha256s"]) + list(training["record_sha256s"]),
        )
    )
    if split_identity["train"] != ordered_sha256(
        [train_by_index[index] for index in range(TRAIN_SPLIT_RECORD_COUNT)]
    ):
        raise PolicyImprovementPopulationError(
            "Stage 0 populations do not reassemble the registered train split."
        )

    select = populations["validation_select"]
    bridge = populations["validation_bridge"]
    combined_indices = list(select["indices"]) + list(bridge["indices"])
    if set(combined_indices) != set(range(256)) or len(combined_indices) != 256:
        raise PolicyImprovementPopulationError(
            "Validation populations overlap or omit registered indices."
        )
    combined_pairs = list(
        zip(
            list(select["selection_scores"]) + list(bridge["selection_scores"]),
            combined_indices,
        )
    )
    if combined_pairs != sorted(combined_pairs):
        raise PolicyImprovementPopulationError(
            "Validation partition boundary or ordering differs."
        )
    by_index = {
        index: record
        for index, record in zip(
            combined_indices,
            list(select["record_sha256s"]) + list(bridge["record_sha256s"]),
        )
    }
    if split_identity["validation"] != ordered_sha256(
        [by_index[index] for index in range(256)]
    ):
        raise PolicyImprovementPopulationError(
            "Validation split identity differs from its full partition."
        )
    result["populations"] = populations
    canonical_json_bytes(result)

    supplied = (
        train_record_sha256s,
        train_input_sha256s,
        validation_record_sha256s,
        validation_input_sha256s,
    )
    if any(item is not None for item in supplied):
        if not all(item is not None for item in supplied):
            raise PolicyImprovementPopulationError(
                "Authenticated manifest identity lists must be supplied together."
            )
        expected = materialize_v2_populations(
            train_record_sha256s=train_record_sha256s or (),
            train_input_sha256s=train_input_sha256s or (),
            validation_record_sha256s=validation_record_sha256s or (),
            validation_input_sha256s=validation_input_sha256s or (),
        )
        if canonical_json_bytes(result) != canonical_json_bytes(expected):
            raise PolicyImprovementPopulationError(
                "Population document differs from authenticated split manifests."
            )
    return result


def load_registered_populations(
    protocol: Mapping[str, object], project_root: str | Path
) -> dict[str, Any]:
    """Load the one population document bound by a v2 protocol."""

    registration = protocol.get("population_registry")
    if not isinstance(registration, Mapping) or set(registration) != {
        "path",
        "sha256",
        "schema_name",
        "schema_version",
    }:
        raise PolicyImprovementPopulationError(
            "Protocol population registry binding is invalid."
        )
    if (
        registration["schema_name"] != POPULATIONS_SCHEMA_NAME
        or registration["schema_version"] != POPULATIONS_SCHEMA_VERSION
    ):
        raise PolicyImprovementPopulationError(
            "Protocol population schema binding differs."
        )
    expected_digest = _sha256(
        registration["sha256"], path="protocol.population_registry.sha256"
    )
    root = Path(project_root).resolve(strict=True)
    path = (root / str(registration["path"])).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PolicyImprovementPopulationError(
            "Population registry path escaped the project root."
        ) from exc
    document = validate_v2_populations(load_strict_json(path))
    if hashlib.sha256(canonical_json_bytes(document)).hexdigest() != expected_digest:
        raise PolicyImprovementPopulationError(
            "Population registry bytes differ from the protocol binding."
        )
    return document


def population_for_id(
    document: Mapping[str, object], population_id: str
) -> dict[str, Any]:
    """Return one validated named population without accepting caller aliases."""

    checked = validate_v2_populations(document)
    if population_id not in set(REGISTERED_POPULATION_IDS):
        raise PolicyImprovementPopulationError(
            f"Population {population_id!r} is not registered."
        )
    return dict(checked["populations"][population_id])


def _manifest_identities(
    value: object, *, split: str, expected_count: int
) -> tuple[list[str], list[str]]:
    if not isinstance(value, Mapping):
        raise PolicyImprovementPopulationError(f"{split} manifest is not an object.")
    records = value.get("record_sha256s")
    inputs = value.get("input_sha256s")
    if (
        value.get("generated_count") != expected_count
        or not isinstance(records, list)
        or not isinstance(inputs, list)
        or len(records) != expected_count
        or len(inputs) != expected_count
    ):
        raise PolicyImprovementPopulationError(
            f"{split} manifest identity inventory differs."
        )
    return list(records), list(inputs)


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    train_records, train_inputs = _manifest_identities(
        load_strict_json(arguments.train_manifest),
        split="train",
        expected_count=1024,
    )
    validation_records, validation_inputs = _manifest_identities(
        load_strict_json(arguments.validation_manifest),
        split="validation",
        expected_count=256,
    )
    document = materialize_v2_populations(
        train_record_sha256s=train_records,
        train_input_sha256s=train_inputs,
        validation_record_sha256s=validation_records,
        validation_input_sha256s=validation_inputs,
    )
    payload = canonical_json_bytes(document)
    if arguments.output is None:
        print(payload.decode("ascii"))
    else:
        _write_new(Path(arguments.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
