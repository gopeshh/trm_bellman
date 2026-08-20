#!/usr/bin/env fbpython
"""Authenticated, training-only policy-improvement throughput calibration."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Protocol

from scripts.policy_improvement_populations import load_registered_populations
from scripts.policy_improvement_v2_registry import (
    load_v2_base_configs,
    validate_v2_registry_document,
)
from scripts.policy_improvement_v2_schema import (
    METHOD_IDS,
    PROTOCOL_ID,
    canonical_json_bytes,
    load_strict_json,
    validate_v2_protocol,
)
from utils.compute_accounting import (
    validate_authenticated_compute_accounting,
    validate_compute_snapshot,
)


THROUGHPUT_RESULT_SCHEMA_NAME = "policy_improvement_training_throughput_calibration_v2"
THROUGHPUT_RESULT_SCHEMA_VERSION = 1
THROUGHPUT_SAMPLE_SCHEMA_NAME = "policy_improvement_training_throughput_sample_v2"
THROUGHPUT_SAMPLE_SCHEMA_VERSION = 1
ENGINEERING_SEED_NAMESPACE = "upi-trm-policy-improvement-v2-throughput-engineering-seed"
ENGINEERING_SEED_NAMESPACE_SHA256 = (
    "2ff9a6c0f4213abd8e8591bf3fe2d8061e54d2c0664e025560135631d7113f03"
)
ENGINEERING_SEED = 804890304
AUTOMATIC_CAPS = (256, 1024)
OPTIONAL_CAP = 4096
CALIBRATION_PPO_ROLLOUT_INTERACTIONS = 16
_LOWER_HEX = frozenset("0123456789abcdef")


class ThroughputCalibrationError(RuntimeError):
    """Raised when a mechanics-only calibration request is not fail-closed."""


@dataclass(frozen=True)
class ThroughputRegistration:
    project_root: Path
    dataset_root: Path
    protocol: dict[str, Any]
    registry: dict[str, Any]
    method_configs: dict[str, Mapping[str, object]]
    protocol_sha256: str
    registry_sha256: str
    population_registry_sha256: str
    runtime_authorization_sha256: str
    dataset_manifest_sha256: str
    train_manifest_sha256: str
    train_ordered_record_sha256: str


@dataclass(frozen=True)
class ThroughputSampleRequest:
    registration: ThroughputRegistration
    method_id: str
    environment_interactions: int
    engineering_seed: int = ENGINEERING_SEED


class ThroughputBackend(Protocol):
    """Backend injected only by the authenticated packaged full runtime."""

    def execute_throughput_sample(
        self, request: ThroughputSampleRequest
    ) -> Mapping[str, object]: ...


class TrainingSplitLoader(Protocol):
    def __call__(self, root: Path, split: str, count: int) -> object: ...


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ThroughputCalibrationError(
            f"{name} must be 64 lowercase hexadecimal characters."
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ThroughputCalibrationError(f"{name} must be a nonnegative integer.")
    return value


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThroughputCalibrationError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ThroughputCalibrationError(f"{name} is outside its allowed range.")
    return result


def _stable_regular_file_bytes(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ThroughputCalibrationError(f"{name} cannot be opened safely.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ThroughputCalibrationError(
                f"{name} must be a singly linked regular file."
            )
        chunks: list[bytes] = []
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ThroughputCalibrationError(f"{name} changed while it was read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_directory(path: str | Path, *, name: str) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise ThroughputCalibrationError(f"{name} must be an absolute canonical path.")
    try:
        requested = supplied.lstat()
        resolved = supplied.resolve(strict=True)
        status = resolved.lstat()
    except OSError as exc:
        raise ThroughputCalibrationError(f"{name} is unavailable.") from exc
    if (
        resolved != supplied
        or stat.S_ISLNK(requested.st_mode)
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise ThroughputCalibrationError(
            f"{name} must be a canonical non-symlink directory."
        )
    return resolved


def _canonical_registered_path(
    project_root: Path,
    supplied: str | Path,
    relative: str,
    *,
    name: str,
) -> Path:
    expected = project_root / relative
    candidate = Path(supplied)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ThroughputCalibrationError(f"{name} must be absolute and canonical.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ThroughputCalibrationError(f"{name} is unavailable.") from exc
    if resolved != candidate or resolved != expected:
        raise ThroughputCalibrationError(f"{name} is not the committed v2 document.")
    _stable_regular_file_bytes(resolved, name=name)
    return resolved


def _validate_engineering_seed() -> None:
    digest = hashlib.sha256(ENGINEERING_SEED_NAMESPACE.encode("ascii")).digest()
    if (
        hashlib.sha256(ENGINEERING_SEED_NAMESPACE.encode("ascii")).hexdigest()
        != ENGINEERING_SEED_NAMESPACE_SHA256
        or int.from_bytes(digest[:4], "big") & 0x7FFFFFFF != ENGINEERING_SEED
    ):
        raise ThroughputCalibrationError("Engineering seed derivation is inconsistent.")


def load_throughput_registration(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    registry_path: str | Path,
    dataset_root: str | Path,
    runtime_authorization_sha256: str,
) -> ThroughputRegistration:
    """Authenticate the v2 mechanics-only registration without opening a split."""

    _validate_engineering_seed()
    project = _regular_directory(project_root, name="project root")
    protocol_file = _canonical_registered_path(
        project,
        protocol_path,
        "configs/policy_improvement_v2/protocol.json",
        name="protocol",
    )
    registry_file = _canonical_registered_path(
        project,
        registry_path,
        "configs/policy_improvement_v2/registry.json",
        name="registry",
    )
    try:
        protocol = validate_v2_protocol(load_strict_json(protocol_file))
        populations = load_registered_populations(protocol, project)
        method_configs = load_v2_base_configs(protocol, project)
        registry = validate_v2_registry_document(
            load_strict_json(registry_file),
            protocol,
            populations,
            base_configs=method_configs,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ThroughputCalibrationError(
            "Protocol-v2 throughput registration is invalid."
        ) from exc
    calibration = protocol["budgets"]["throughput_calibration"]
    if calibration != {
        "automatic_caps": list(AUTOMATIC_CAPS),
        "environment_interaction_caps": [*AUTOMATIC_CAPS, OPTIONAL_CAP],
        "evaluation_rollouts": False,
        "split": "train",
        "user_authorization_required_for": OPTIONAL_CAP,
    }:
        raise ThroughputCalibrationError(
            "Protocol-v2 throughput schedule differs from the runtime."
        )
    isolation = protocol["test_isolation"]
    if (
        isolation.get("throughput_split") != "train"
        or isolation.get("test_open_status") != "not_created"
    ):
        raise ThroughputCalibrationError(
            "Protocol-v2 throughput isolation is not train-only."
        )
    if tuple(method_configs) != METHOD_IDS:
        raise ThroughputCalibrationError(
            "Throughput calibration must contain the four registered methods."
        )

    dataset = _regular_directory(dataset_root, name="dataset root")
    dataset_manifest = _stable_regular_file_bytes(
        dataset / "MANIFEST.json",
        name="dataset manifest",
    )
    train_manifest = _stable_regular_file_bytes(
        dataset / "manifests" / "train.json",
        name="train manifest",
    )
    dataset_manifest_sha256 = hashlib.sha256(dataset_manifest).hexdigest()
    train_manifest_sha256 = hashlib.sha256(train_manifest).hexdigest()
    expected_dataset_manifest = protocol["dataset"]["manifest_sha256"]
    expected_train_manifest = protocol["dataset"]["splits"]["train"]["manifest_sha256"]
    if expected_dataset_manifest != {
        "status": "available",
        "value": dataset_manifest_sha256,
    } or expected_train_manifest != {
        "status": "available",
        "value": train_manifest_sha256,
    }:
        raise ThroughputCalibrationError(
            "Dataset or train manifest differs from protocol v2."
        )
    train_order = protocol["dataset"]["splits"]["train"]["ordered_record_sha256"]
    if not isinstance(train_order, Mapping) or train_order.get("status") != "available":
        raise ThroughputCalibrationError("Train record identity is unavailable.")
    return ThroughputRegistration(
        project_root=project,
        dataset_root=dataset,
        protocol=protocol,
        registry=registry,
        method_configs=dict(method_configs),
        protocol_sha256=hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        registry_sha256=hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
        population_registry_sha256=_sha256(
            protocol["population_registry"]["sha256"],
            name="population registry SHA-256",
        ),
        runtime_authorization_sha256=_sha256(
            runtime_authorization_sha256,
            name="runtime authorization SHA-256",
        ),
        dataset_manifest_sha256=dataset_manifest_sha256,
        train_manifest_sha256=train_manifest_sha256,
        train_ordered_record_sha256=_sha256(
            train_order.get("value"),
            name="train ordered-record SHA-256",
        ),
    )


def load_training_only_dataset(
    registration: ThroughputRegistration,
    loader: TrainingSplitLoader,
) -> object:
    """Call a dataset loader exactly once with the registered train split."""

    return loader(
        registration.dataset_root,
        "train",
        int(registration.protocol["dataset"]["splits"]["train"]["count"]),
    )


def _validate_runtime_identity(
    value: object,
    *,
    runtime_authorization_sha256: str,
) -> dict[str, str]:
    fields = {
        "role",
        "runtime_sha256",
        "source_git_commit",
        "source_manifest_sha256",
        "producer_source_manifest_sha256",
        "runtime_profile_sha256",
        "selected_source_manifest_sha256",
        "runtime_authorization_sha256",
        "launcher_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ThroughputCalibrationError("Runtime identity inventory differs.")
    result = {name: str(value[name]) for name in fields}
    if result["role"] != "policy-improvement-full":
        raise ThroughputCalibrationError(
            "Throughput runtime does not use the full role."
        )
    for name in fields - {"role", "source_git_commit"}:
        _sha256(result[name], name=name)
    if (
        len(result["source_git_commit"]) != 40
        or any(character not in _LOWER_HEX for character in result["source_git_commit"])
        or result["runtime_authorization_sha256"] != runtime_authorization_sha256
        or result["runtime_profile_sha256"] != result["source_manifest_sha256"]
        or result["selected_source_manifest_sha256"] != result["source_manifest_sha256"]
    ):
        raise ThroughputCalibrationError("Runtime identity is not self-consistent.")
    return result


def validate_throughput_sample(
    value: object,
    *,
    request: ThroughputSampleRequest,
) -> dict[str, object]:
    fields = {
        "schema_name",
        "schema_version",
        "method_id",
        "environment_interaction_cap",
        "engineering_seed",
        "training_split",
        "started_environment_interactions",
        "completed_environment_interactions",
        "session_setup_seconds",
        "elapsed_seconds",
        "recurrent_map_applications",
        "method_config_sha256",
        "effective_config_sha256",
        "dataset_manifest_sha256",
        "train_manifest_sha256",
        "train_ordered_record_sha256",
        "runtime_identity",
        "compute_snapshot",
        "compute_accounting",
        "evaluation_rollouts",
        "validation_data_opened",
        "test_data_opened",
        "test_open_bound",
        "scientific_selection",
        "paper_evidence_eligible",
        "performance_metrics_collected",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ThroughputCalibrationError("Throughput sample inventory differs.")
    sample = dict(value)
    registration = request.registration
    if (
        sample["schema_name"] != THROUGHPUT_SAMPLE_SCHEMA_NAME
        or sample["schema_version"] != THROUGHPUT_SAMPLE_SCHEMA_VERSION
        or sample["method_id"] != request.method_id
        or sample["environment_interaction_cap"] != request.environment_interactions
        or sample["engineering_seed"] != request.engineering_seed
        or sample["training_split"] != "train"
        or sample["started_environment_interactions"] != 0
        or sample["completed_environment_interactions"]
        != request.environment_interactions
    ):
        raise ThroughputCalibrationError("Throughput sample registration differs.")
    for field in (
        "evaluation_rollouts",
        "validation_data_opened",
        "test_data_opened",
        "test_open_bound",
        "scientific_selection",
        "paper_evidence_eligible",
        "performance_metrics_collected",
    ):
        if sample[field] is not False:
            raise ThroughputCalibrationError(
                f"Throughput sample must declare {field}=false."
            )
    _finite_float(sample["session_setup_seconds"], name="session setup", minimum=0.0)
    _finite_float(sample["elapsed_seconds"], name="elapsed seconds", minimum=0.0)
    _nonnegative_int(
        sample["recurrent_map_applications"],
        name="recurrent map applications",
    )
    method = next(
        item
        for item in registration.protocol["methods"]
        if item["id"] == request.method_id
    )
    expected_digests = {
        "method_config_sha256": method["config_sha256"],
        "dataset_manifest_sha256": registration.dataset_manifest_sha256,
        "train_manifest_sha256": registration.train_manifest_sha256,
        "train_ordered_record_sha256": registration.train_ordered_record_sha256,
    }
    for field, expected in expected_digests.items():
        if _sha256(sample[field], name=field) != expected:
            raise ThroughputCalibrationError(f"Throughput sample changed {field}.")
    _sha256(sample["effective_config_sha256"], name="effective config SHA-256")
    runtime = _validate_runtime_identity(
        sample["runtime_identity"],
        runtime_authorization_sha256=registration.runtime_authorization_sha256,
    )
    snapshot = validate_compute_snapshot(sample["compute_snapshot"])
    accounting = validate_authenticated_compute_accounting(
        sample["compute_accounting"],
        source_compute_snapshot=snapshot,
    )
    if (
        accounting["environment_interactions"]
        != {"status": "available", "value": request.environment_interactions}
        or accounting["evaluation_seconds"] != {"status": "available", "value": 0.0}
        or snapshot["wall_time_seconds"]["evaluation"] != 0.0
        or any(
            snapshot["model_work"]["evaluation"][name] != 0
            for name in snapshot["model_work"]["evaluation"]
        )
    ):
        raise ThroughputCalibrationError(
            "Throughput sample contains evaluation work or the wrong cap."
        )
    sample["runtime_identity"] = runtime
    sample["compute_snapshot"] = snapshot
    sample["compute_accounting"] = accounting
    return sample


def fit_engineering_model(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Fit only t(N)=a+bN by ordinary least squares."""

    if len(samples) < 2:
        raise ThroughputCalibrationError("Engineering fit requires at least two caps.")
    points = sorted(
        (
            _nonnegative_int(
                sample["environment_interaction_cap"],
                name="fit environment cap",
            ),
            _finite_float(sample["elapsed_seconds"], name="fit elapsed", minimum=0.0),
        )
        for sample in samples
    )
    caps = [point[0] for point in points]
    if len(caps) != len(set(caps)):
        raise ThroughputCalibrationError("Engineering fit caps are not unique.")
    mean_n = sum(caps) / len(caps)
    mean_t = sum(point[1] for point in points) / len(points)
    denominator = sum((cap - mean_n) ** 2 for cap in caps)
    if denominator <= 0.0:
        raise ThroughputCalibrationError("Engineering fit has no cap variation.")
    slope = (
        sum((cap - mean_n) * (elapsed - mean_t) for cap, elapsed in points)
        / denominator
    )
    intercept = mean_t - slope * mean_n
    residual = sum(
        (elapsed - (intercept + slope * cap)) ** 2 for cap, elapsed in points
    )
    predicted_4096 = intercept + slope * OPTIONAL_CAP
    if not all(
        math.isfinite(value) for value in (slope, intercept, residual, predicted_4096)
    ):
        raise ThroughputCalibrationError("Engineering fit produced a nonfinite value.")
    return {
        "schema_name": "policy_improvement_throughput_linear_model_v2",
        "schema_version": 1,
        "equation": "t(N)=a+bN",
        "fit_caps": caps,
        "startup_seconds_a": intercept,
        "steady_state_seconds_per_interaction_b": slope,
        "sum_squared_error": residual,
        "predicted_4096_seconds": predicted_4096,
    }


def execute_calibration(
    registration: ThroughputRegistration,
    *,
    backend: ThroughputBackend,
    authorize_4096_tier: bool,
    maximum_total_predicted_4096_seconds: float | None,
) -> dict[str, object]:
    if authorize_4096_tier != (maximum_total_predicted_4096_seconds is not None):
        raise ThroughputCalibrationError(
            "The 4096 tier requires both explicit authorization and a prediction ceiling."
        )
    if maximum_total_predicted_4096_seconds is not None:
        maximum_total_predicted_4096_seconds = _finite_float(
            maximum_total_predicted_4096_seconds,
            name="maximum total predicted 4096 seconds",
            minimum=0.0,
        )
        if maximum_total_predicted_4096_seconds <= 0.0:
            raise ThroughputCalibrationError(
                "The 4096 prediction ceiling must be positive."
            )

    samples: list[dict[str, object]] = []
    for cap in AUTOMATIC_CAPS:
        for method_id in METHOD_IDS:
            request = ThroughputSampleRequest(
                registration=registration,
                method_id=method_id,
                environment_interactions=cap,
            )
            with redirect_stdout(sys.stderr):
                raw = backend.execute_throughput_sample(request)
            samples.append(validate_throughput_sample(raw, request=request))

    automatic_models = {
        method_id: fit_engineering_model(
            [sample for sample in samples if sample["method_id"] == method_id]
        )
        for method_id in METHOD_IDS
    }
    predicted_total = sum(
        _finite_float(
            model["predicted_4096_seconds"],
            name="predicted 4096 seconds",
        )
        for model in automatic_models.values()
    )
    if not math.isfinite(predicted_total):
        raise ThroughputCalibrationError("The total 4096 prediction is nonfinite.")

    optional_executed = False
    if authorize_4096_tier:
        assert maximum_total_predicted_4096_seconds is not None
        for method_id in METHOD_IDS:
            model = automatic_models[method_id]
            method_samples = [
                sample for sample in samples if sample["method_id"] == method_id
            ]
            measured_1024 = next(
                _finite_float(
                    sample["elapsed_seconds"],
                    name="measured 1024 seconds",
                    minimum=0.0,
                )
                for sample in method_samples
                if sample["environment_interaction_cap"] == 1024
            )
            if (
                _finite_float(
                    model["steady_state_seconds_per_interaction_b"],
                    name="steady-state seconds per interaction",
                )
                < 0.0
                or _finite_float(
                    model["predicted_4096_seconds"],
                    name="predicted 4096 seconds",
                )
                < measured_1024
            ):
                raise ThroughputCalibrationError(
                    "The automatic tiers do not support a fail-closed 4096 prediction."
                )
        if predicted_total > maximum_total_predicted_4096_seconds:
            raise ThroughputCalibrationError(
                "The predicted 4096 runtime exceeds the authorized resource ceiling."
            )
        for method_id in METHOD_IDS:
            request = ThroughputSampleRequest(
                registration=registration,
                method_id=method_id,
                environment_interactions=OPTIONAL_CAP,
            )
            with redirect_stdout(sys.stderr):
                raw = backend.execute_throughput_sample(request)
            samples.append(validate_throughput_sample(raw, request=request))
        optional_executed = True

    final_models = {
        method_id: fit_engineering_model(
            [sample for sample in samples if sample["method_id"] == method_id]
        )
        for method_id in METHOD_IDS
    }
    runtime_identities = {
        canonical_json_bytes(sample["runtime_identity"]) for sample in samples
    }
    if len(runtime_identities) != 1:
        raise ThroughputCalibrationError("Calibration samples mix runtime identities.")
    runtime_value = samples[0]["runtime_identity"]
    if not isinstance(runtime_value, Mapping):
        raise ThroughputCalibrationError("Calibration runtime identity is invalid.")
    runtime_identity = dict(runtime_value)
    result: dict[str, object] = {
        "schema_name": THROUGHPUT_RESULT_SCHEMA_NAME,
        "schema_version": THROUGHPUT_RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": registration.protocol_sha256,
        "population_registry_sha256": registration.population_registry_sha256,
        "registry_sha256": registration.registry_sha256,
        "runtime_authorization_sha256": (registration.runtime_authorization_sha256),
        "runtime_identity": runtime_identity,
        "dataset_identity": {
            "dataset_manifest_sha256": registration.dataset_manifest_sha256,
            "train_manifest_sha256": registration.train_manifest_sha256,
            "train_ordered_record_sha256": (registration.train_ordered_record_sha256),
        },
        "engineering_seed": {
            "namespace": ENGINEERING_SEED_NAMESPACE,
            "namespace_sha256": ENGINEERING_SEED_NAMESPACE_SHA256,
            "derivation": "sha256_first_u32_masked_to_31_bits",
            "value": ENGINEERING_SEED,
        },
        "method_ids": list(METHOD_IDS),
        "schedule": {
            "automatic_caps": list(AUTOMATIC_CAPS),
            "optional_cap": OPTIONAL_CAP,
            "optional_cap_authorized": authorize_4096_tier,
            "optional_cap_executed": optional_executed,
            "maximum_total_predicted_4096_seconds": (
                maximum_total_predicted_4096_seconds
            ),
            "pre_execution_predicted_total_4096_seconds": predicted_total,
        },
        "samples": samples,
        "automatic_tier_models": automatic_models,
        "final_models": final_models,
        "training_split": "train",
        "evaluation_rollouts": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "test_open_bound": False,
        "scientific_selection": False,
        "paper_evidence_eligible": False,
        "performance_metrics_collected": False,
    }
    canonical_json_bytes(result)
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: ThroughputBackend | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--runtime-authorization-sha256", required=True)
    parser.add_argument("--authorize-4096-tier", action="store_true")
    parser.add_argument("--maximum-total-predicted-4096-seconds", type=float)
    arguments = parser.parse_args(argv)
    if backend is None:
        print(
            "Throughput calibration requires the authenticated packaged runtime.",
            file=sys.stderr,
        )
        return 2
    try:
        registration = load_throughput_registration(
            project_root=arguments.project_root,
            protocol_path=arguments.protocol,
            registry_path=arguments.registry,
            dataset_root=arguments.dataset_root,
            runtime_authorization_sha256=(arguments.runtime_authorization_sha256),
        )
        result = execute_calibration(
            registration,
            backend=backend,
            authorize_4096_tier=arguments.authorize_4096_tier,
            maximum_total_predicted_4096_seconds=(
                arguments.maximum_total_predicted_4096_seconds
            ),
        )
    except (ThroughputCalibrationError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
