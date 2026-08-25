#!/usr/bin/env fbpython
"""Authenticated Stage 0 runner for the registered policy-improvement study.

This module is reached only through the hidden training-PAR entrypoint owned by
``phase4_runtime_launcher``.  It runs one registered row and one immutable
16-interaction segment per process.  Full pilot or confirmatory rows are not
accepted here.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from dataset.build_policy_improvement_4x4 import verify_dataset
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from phase4_runtime_profile import authorize_phase4_training_source
from policy_improvement_smoke_checkpoint import (
    build_ppo_smoke_checkpoint,
    load_stable_checkpoint,
    publish_checkpoint,
    validate_ppo_smoke_checkpoint,
)
from rl.batch_utils import prepare_batch_x, prepare_plan
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.evaluator import evaluate_plan_policy_with_scores
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from rl.upi_trm_trainer import UPITrmTrainer
from scripts.policy_improvement_registry import (
    generate_registry,
    load_registered_base_configs,
)
from scripts.policy_improvement_schema import (
    amendment_history_sha256,
    canonical_json_bytes,
    load_strict_json,
    policy_variants_for_method,
    PolicyImprovementSchemaError,
    primary_policy_variant_for_method,
    RESULT_SCHEMA_VERSION,
    runtime_authorization_sha256,
    SCHEMA_NAME,
    validate_protocol,
    validate_result,
    validate_runtime_authorization,
    validated_result_payload,
)
from utils.compute_accounting import (
    add_model_counters,
    aggregate_model_compute,
    build_training_compute_accounting,
    subtract_model_counters,
    zero_model_counters,
)
from utils.dataset_provenance import (
    build_dataset_provenance,
    dataset_input_sha256s,
    dataset_pool_sha256,
    dataset_puzzle_identifier_sha256s,
    dataset_sample_sha256s,
    dataset_source_build_metadata,
    input_sha256,
    ordered_record_sha256,
)
from utils.run_identity import (
    build_checkpoint_lineage,
    build_run_identity,
    canonical_json_sha256,
    discover_clean_git_source,
    file_sha256,
)


SMOKE_SEGMENT_SCHEMA_VERSION = 2
_POLICY_SMOKE_ROLE = "policy-improvement-smoke"
_LAUNCHER_SHA256_ENV = "UPI_TRM_POLICY_SMOKE_LAUNCHER_SHA256"
_LOWER_SHA256 = frozenset("0123456789abcdef")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class PolicyImprovementSmokeError(RuntimeError):
    """Raised when a Stage 0 row cannot be authenticated or published."""


class _SmokeExecutionError(PolicyImprovementSmokeError):
    """Attach one canonical failure phase to an authenticated run failure."""

    def __init__(self, phase: str, cause: BaseException) -> None:
        super().__init__(f"Stage 0 {phase} failed: {cause}")
        self.phase = phase
        self.cause = cause


@dataclass(frozen=True)
class SmokeContext:
    protocol: dict[str, Any]
    protocol_sha256: str
    registry: dict[str, Any]
    registry_sha256: str
    row: dict[str, Any]
    registry_row_sha256: str
    source_root: Path
    dataset_root: Path
    dataset_manifest_sha256: str
    dataset_producer_source: dict[str, str]
    evidence_root: Path
    run_root: Path
    segment_budget: int
    segment_name: str
    runtime_sha256: str
    runtime_authorization_sha256: str
    runtime_profile_sha256: str
    selected_source_manifest_sha256: str
    launcher_sha256: str
    producer_commit: str
    producer_manifest_sha256: str


@dataclass
class SmokeSession:
    model: Any
    trainer: Any
    rl_config: Any
    env_config: PlanEditEnvConfig
    train_dataset: Any
    evaluation_dataset: Any
    checker: Any
    task_config: Any
    dataset_provenance: dict[str, Any]
    effective_config: dict[str, Any]
    effective_config_sha256: str
    initialization_sha256: str
    device: torch.device
    config_path: Path
    method_config_sha256: str
    run_identity: dict[str, Any] | None
    evidence_identity: dict[str, Any]
    evaluation_population: dict[str, Any] | None = None
    training_population: dict[str, Any] | None = None
    evaluation_started: bool = False


@dataclass(frozen=True)
class GpuUtilizationSummary:
    """Training-interval-only device utilization observations."""

    device_type: str
    sampling_interval_seconds: float | None
    samples: tuple[float, ...]


def _optional_trainer_config_dict(trainer: Any, module: Any) -> dict[str, Any] | None:
    """Match production evidence semantics for baseline-only trainer config."""

    config = getattr(trainer, "config", None)
    return module._config_dict(config) if config is not None else None


def _bind_registered_checker(trainer: Any, checker: Any) -> None:
    """Install the exact-baseline checker on UPI trainers as production does."""

    if not isinstance(trainer, UPITrmTrainer):
        return
    trainer.set_checker_fn(checker)
    if trainer._checker_fn is not checker:
        raise PolicyImprovementSmokeError(
            "UPI trainer did not retain the registered checker."
        )


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise PolicyImprovementSmokeError(
            f"{name} must be 64 lowercase hexadecimal characters."
        )
    return value


def _available_hex(
    value: object,
    *,
    name: str,
    length: int,
) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"status", "value"}
        or value.get("status") != "available"
    ):
        raise PolicyImprovementSmokeError(
            f"{name} must be frozen before Stage 0 execution."
        )
    encoded = value.get("value")
    if (
        not isinstance(encoded, str)
        or len(encoded) != length
        or any(character not in _LOWER_SHA256 for character in encoded)
    ):
        raise PolicyImprovementSmokeError(f"{name} has an invalid digest.")
    return encoded


def _registered_hex(value: object, *, name: str, length: int) -> str:
    """Validate one bare registered digest.

    Protocol dataset registrations wrap their digests in a frozen
    ``{"status", "value"}`` envelope, which ``_available_hex`` unwraps.
    Population documents store the digest directly, so they need this
    variant. Both enforce the same lowercase-hex shape.
    """

    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise PolicyImprovementSmokeError(f"{name} has an invalid digest.")
    return value


def _absolute_path(value: str, *, name: str, must_exist: bool) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise PolicyImprovementSmokeError(f"{name} must be an absolute path.")
    if must_exist:
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise PolicyImprovementSmokeError(f"{name} does not exist.") from exc
    return path


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one authenticated policy-improvement Stage 0 segment.",
        allow_abbrev=False,
    )
    parser.add_argument("--policy-improvement-smoke-entrypoint", action="store_true")
    parser.add_argument("--policy-improvement-protocol", required=True)
    parser.add_argument("--policy-improvement-row-id", required=True)
    parser.add_argument(
        "--policy-improvement-smoke-segment",
        required=True,
        choices=("prepare", "resume"),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--validation-manifest-sha256")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--source-project-root", required=True)
    arguments = parser.parse_args(list(argv))
    if not arguments.policy_improvement_smoke_entrypoint:
        raise PolicyImprovementSmokeError(
            "The Stage 0 runtime requires its launcher-owned hidden entrypoint."
        )
    return arguments


def _safe_owner_root(path: Path, *, source_root: Path) -> Path:
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            try:
                cursor.mkdir(mode=0o700)
            except OSError as exc:
                raise PolicyImprovementSmokeError(
                    "The evidence owner root cannot be created safely."
                ) from exc
            _fsync_dir(cursor.parent)
            info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PolicyImprovementSmokeError(
                "The evidence owner path contains a non-directory or symlink."
            )
    root = path.resolve(strict=True)
    if (
        root == source_root
        or source_root in root.parents
        or root in source_root.parents
    ):
        raise PolicyImprovementSmokeError(
            "Policy-improvement evidence must live outside the source repository."
        )
    if root.stat().st_uid != os.geteuid():
        raise PolicyImprovementSmokeError("The evidence owner root has the wrong UID.")
    if stat.S_IMODE(root.stat().st_mode) & 0o022:
        raise PolicyImprovementSmokeError(
            "The evidence owner root must not be group- or world-writable."
        )
    return root


def _canonical_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyImprovementSmokeError(f"{name} must be a nonempty path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or "\\" in value
        or any(component in {"", ".", ".."} for component in path.parts)
    ):
        raise PolicyImprovementSmokeError(f"{name} is not canonical.")
    return value


def _ensure_owned_directory(root: Path, destination: Path) -> Path:
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise PolicyImprovementSmokeError(
            "Stage 0 output path escapes its evidence owner root."
        ) from exc
    cursor = root
    for component in relative.parts:
        if component in {"", ".", ".."}:
            raise PolicyImprovementSmokeError("Stage 0 output path is not canonical.")
        parent = cursor
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            try:
                cursor.mkdir(mode=0o700)
            except OSError as exc:
                raise PolicyImprovementSmokeError(
                    "Stage 0 output directory cannot be created safely."
                ) from exc
            _fsync_dir(parent)
            info = cursor.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise PolicyImprovementSmokeError(
                "Stage 0 output directory has unsafe ownership or permissions."
            )
    return cursor


def _load_context(
    arguments: argparse.Namespace,
    *,
    runtime_preflight: Any,
) -> SmokeContext:
    if (
        runtime_preflight is None
        or runtime_preflight.phase4_role != _POLICY_SMOKE_ROLE
        or runtime_preflight.source_git_commit is None
        or runtime_preflight.source_manifest_sha256 is None
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 requires policy-improvement-smoke runtime attestation."
        )
    launcher_sha256 = _sha256(
        os.environ.pop(_LAUNCHER_SHA256_ENV, None), name="launcher SHA-256"
    )
    authorization_json = getattr(
        runtime_preflight, "policy_runtime_authorization_json", None
    )
    authorization_digest = _sha256(
        getattr(
            runtime_preflight,
            "policy_runtime_authorization_sha256",
            None,
        ),
        name="runtime authorization SHA-256",
    )
    runtime_profile_sha256 = _sha256(
        getattr(runtime_preflight, "policy_runtime_profile_sha256", None),
        name="training runtime profile SHA-256",
    )
    selected_source_manifest_sha256 = _sha256(
        getattr(
            runtime_preflight,
            "policy_selected_source_manifest_sha256",
            None,
        ),
        name="training selected-source manifest SHA-256",
    )
    if not isinstance(authorization_json, str):
        raise PolicyImprovementSmokeError("Stage 0 runtime authorization is missing.")
    try:
        authorization = validate_runtime_authorization(json.loads(authorization_json))
    except (
        json.JSONDecodeError,
        PolicyImprovementSchemaError,
    ) as exc:
        raise PolicyImprovementSmokeError(
            "Stage 0 runtime authorization is invalid."
        ) from exc
    if runtime_authorization_sha256(authorization) != authorization_digest:
        raise PolicyImprovementSmokeError(
            "Stage 0 runtime authorization digest differs."
        )
    training_role = next(
        role
        for role in authorization["roles"]
        if role["role"] == "policy-improvement-training"
    )
    evaluation_role = next(
        role
        for role in authorization["roles"]
        if role["role"] == "policy-improvement-evaluation"
    )
    expected_role = {
        "source_git_commit": runtime_preflight.source_git_commit,
        "runtime_sha256": runtime_preflight.runtime_sha256,
        "runtime_profile_sha256": runtime_profile_sha256,
        "selected_source_manifest_sha256": (selected_source_manifest_sha256),
    }
    if (
        any(
            training_role[field] != expected
            for field, expected in expected_role.items()
        )
        or any(
            evaluation_role[field] != expected
            for field, expected in expected_role.items()
        )
        or authorization["launcher_sha256"] != launcher_sha256
        or authorization["producer_git_commit"] != runtime_preflight.source_git_commit
        or authorization["producer_source_manifest_sha256"]
        != runtime_preflight.source_manifest_sha256
        or runtime_profile_sha256 != runtime_preflight.source_manifest_sha256
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 runtime authorization differs from the sealed execution."
        )
    source_root = _absolute_path(
        arguments.source_project_root, name="source project root", must_exist=True
    )
    if not source_root.is_dir():
        raise PolicyImprovementSmokeError("Source project root is not a directory.")
    producer = discover_clean_git_source(source_root)
    if producer != {
        "git_commit": runtime_preflight.source_git_commit,
        "git_clean": True,
    }:
        raise PolicyImprovementSmokeError(
            "Producer checkout differs from the launcher authorization."
        )

    protocol_path = _absolute_path(
        arguments.policy_improvement_protocol,
        name="policy-improvement protocol",
        must_exist=True,
    )
    expected_protocol = source_root / (
        "configs/policy_improvement_v2/protocol.json"
        if authorization.get("schema_name")
        == "policy_improvement_runtime_authorization_v3"
        else "configs/policy_improvement_v1/protocol.json"
    )
    if protocol_path != expected_protocol.resolve(strict=True):
        raise PolicyImprovementSmokeError(
            "Stage 0 accepts only the committed canonical protocol path."
        )
    protocol = validate_protocol(load_strict_json(protocol_path))
    if (
        protocol.get("schema_name") == "policy_improvement_protocol_v2"
        and [
            authorization.get("schema_name"),
            authorization.get("schema_version"),
        ]
        != protocol["document_schemas"]["runtime_authorization"]
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 authorization schema differs from protocol v2."
        )
    if (
        authorization["protocol_sha256"]
        != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 runtime authorization names a different protocol."
        )
    smoke_budget = protocol["budgets"]["smoke"]
    if smoke_budget["environment_interactions"] != 32 or smoke_budget[
        "checkpoint_environment_interactions"
    ] != [16, 32]:
        raise PolicyImprovementSmokeError(
            "Stage 0 runtime supports only the registered 16/32 interaction schedule."
        )
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    base_configs = load_registered_base_configs(protocol, source_root)
    is_v2 = protocol.get("schema_name") == "policy_improvement_protocol_v2"
    populations_document: object | None = None
    if is_v2:
        from scripts.policy_improvement_populations import load_registered_populations

        populations_document = load_registered_populations(protocol, source_root)
    registry = generate_registry(
        protocol,
        base_configs=base_configs,
        populations_value=populations_document,
    )
    registry_sha256 = hashlib.sha256(canonical_json_bytes(registry)).hexdigest()
    if (
        authorization.get("schema_name")
        == "policy_improvement_runtime_authorization_v3"
    ):
        theory_amendment = load_strict_json(
            source_root
            / "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
        )
        expected_authorization_bindings = {
            "protocol": {
                "schema_name": protocol["schema_name"],
                "schema_version": protocol["schema_version"],
                "protocol_id": protocol["protocol_id"],
                "sha256": protocol_sha256,
            },
            "registry": {
                "schema_name": registry["schema_name"],
                "schema_version": registry["registry_schema_version"],
                "sha256": registry_sha256,
            },
            "amendments": [
                {
                    "schema_name": theory_amendment["schema_name"],
                    "schema_version": theory_amendment["schema_version"],
                    "amendment_id": theory_amendment["amendment_id"],
                    "sha256": hashlib.sha256(
                        canonical_json_bytes(theory_amendment)
                    ).hexdigest(),
                }
            ],
        }
        for field, expected in expected_authorization_bindings.items():
            if authorization.get(field) != expected:
                raise PolicyImprovementSmokeError(
                    f"Stage 0 authorization {field} differs from canonical v2."
                )
    rows = [
        row
        for row in registry["rows"]
        if row["run_id"] == arguments.policy_improvement_row_id
    ]
    if len(rows) != 1 or rows[0]["phase"] != "stage0_smoke":
        raise PolicyImprovementSmokeError(
            "Requested row is not one concrete registered Stage 0 row."
        )
    row = rows[0]
    row_sha256 = hashlib.sha256(canonical_json_bytes(row)).hexdigest()

    dataset_root = _absolute_path(
        arguments.dataset_root, name="dataset root", must_exist=True
    )
    expected_dataset = source_root / str(protocol["dataset"]["root"])
    if dataset_root != expected_dataset.resolve(strict=True):
        raise PolicyImprovementSmokeError(
            "Dataset root differs from the registered protocol."
        )
    dataset_registration = protocol["dataset"]
    registered_dataset_manifest = _available_hex(
        dataset_registration["manifest_sha256"],
        name="registered dataset manifest",
        length=64,
    )
    registered_dataset_producer = {
        field: _available_hex(
            dataset_registration["producer_source"][field],
            name=f"registered dataset producer {field}",
            length=40 if field == "git_commit" else 64,
        )
        for field in (
            "git_commit",
            "launcher_sha256",
            "runtime_sha256",
            "source_manifest_sha256",
        )
    }
    verified_dataset = verify_dataset(
        dataset_root,
        owner_root=dataset_root.parent,
        expected_producer=registered_dataset_producer,
        verify_content_splits=({"train"} if is_v2 else {"train", "validation"}),
    )
    if verified_dataset.get("manifest_sha256") != registered_dataset_manifest:
        raise PolicyImprovementSmokeError(
            "Materialized dataset manifest differs from the protocol."
        )
    splits = verified_dataset.get("split_manifest_sha256")
    if not isinstance(splits, Mapping):
        raise PolicyImprovementSmokeError(
            "Dataset verification omitted split identities."
        )
    ordered_records = verified_dataset.get("split_ordered_record_sha256")
    if not isinstance(ordered_records, Mapping):
        raise PolicyImprovementSmokeError(
            "Dataset verification omitted ordered-record identities."
        )
    expected_manifest_hashes = {
        "train": _sha256(arguments.train_manifest_sha256, name="train manifest")
    }
    if is_v2:
        if arguments.validation_manifest_sha256 is not None:
            raise PolicyImprovementSmokeError(
                "Protocol v2 Stage 0 must not accept a validation manifest argument."
            )
    else:
        expected_manifest_hashes["validation"] = _sha256(
            arguments.validation_manifest_sha256, name="validation manifest"
        )
    registered_manifest_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        registration = dataset_registration["splits"][split]["manifest_sha256"]
        registered_manifest_hashes[split] = _available_hex(
            registration,
            name=f"registered {split} manifest",
            length=64,
        )
        if splits.get(split) != registered_manifest_hashes[split]:
            raise PolicyImprovementSmokeError(
                f"Materialized {split} manifest differs from the protocol."
            )
        registered_ordered_records = _available_hex(
            dataset_registration["splits"][split]["ordered_record_sha256"],
            name=f"registered {split} ordered records",
            length=64,
        )
        if ordered_records.get(split) != registered_ordered_records:
            raise PolicyImprovementSmokeError(
                f"Materialized {split} record order differs from the protocol."
            )
    for split, expected in expected_manifest_hashes.items():
        if expected != registered_manifest_hashes[split]:
            raise PolicyImprovementSmokeError(
                f"Launcher {split} manifest differs from the protocol."
            )

    evidence_owner = _safe_owner_root(
        _absolute_path(arguments.evidence_root, name="evidence root", must_exist=False),
        source_root=source_root,
    )
    output_relative = _canonical_relative_path(
        protocol["output_root"]["relative_path"], name="protocol output root"
    )
    output_root = evidence_owner / output_relative
    run_root = output_root / "runs" / str(row["run_id"])
    segment = arguments.policy_improvement_smoke_segment
    segment_budget = 16 if segment == "prepare" else 32
    return SmokeContext(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
        row=row,
        registry_row_sha256=row_sha256,
        source_root=source_root,
        dataset_root=dataset_root,
        dataset_manifest_sha256=registered_dataset_manifest,
        dataset_producer_source=registered_dataset_producer,
        evidence_root=evidence_owner,
        run_root=run_root,
        segment_budget=segment_budget,
        segment_name=segment,
        runtime_sha256=_sha256(
            runtime_preflight.runtime_sha256, name="training runtime SHA-256"
        ),
        runtime_authorization_sha256=authorization_digest,
        runtime_profile_sha256=runtime_profile_sha256,
        selected_source_manifest_sha256=(selected_source_manifest_sha256),
        launcher_sha256=launcher_sha256,
        producer_commit=str(runtime_preflight.source_git_commit),
        producer_manifest_sha256=_sha256(
            runtime_preflight.source_manifest_sha256,
            name="producer manifest SHA-256",
        ),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PolicyImprovementSmokeError(
            "Registered method config is invalid."
        ) from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementSmokeError("Registered method config is not a mapping.")
    return value


def _task_config(
    rl_config: Any, dataset: Any, seq_len: int, module: Any
) -> tuple[Any, Any, str]:
    resolution = module.resolve_checker_from_dataset(
        rl_cfg=rl_config, dataset=dataset, seq_len=seq_len
    )
    task = None
    try:
        from rl.task_config import get_task_config

        if resolution.checker_kind in {
            "solution",
            "constraint",
            "progress",
            "feasibility",
        }:
            task = get_task_config(
                "sudoku",
                disable_constraint_masking=rl_config.disable_constraint_masking,
            )
    except ImportError:
        task = None
    return resolution.checker_fn, task, resolution.checker_kind


def _build_session(context: SmokeContext, module: Any) -> SmokeSession:
    module.set_global_seed(int(context.row["seed"]))
    method_id = str(context.row["base_method_id"])
    method = next(
        item for item in context.protocol["methods"] if item["id"] == method_id
    )
    config_path = (context.source_root / str(method["config_path"])).resolve(
        strict=True
    )
    method_config_sha256 = _sha256(
        method["config_sha256"],
        name="registered method config SHA-256",
    )
    if file_sha256(config_path) != method_config_sha256:
        raise PolicyImprovementSmokeError(
            "Registered method configuration differs from its protocol digest."
        )
    config_layer = _read_yaml(config_path)
    base = module.RLConfig(
        batch_size=32,
        num_train_steps=200,
        rollout_episodes_per_step=1,
        max_edits=8,
        log_interval=10,
        eval_interval=50,
        eval_num_episodes=50,
        eval_seed=1729,
        use_tqdm=False,
        debug_checks=False,
    )
    merged = module._config_dict(base)
    merged = module.merge_rl_config_layer(merged, config_layer)
    stage0 = context.protocol["grid"]["stage0"]
    registered_override = dict(context.row["config_override"])
    registered_layer = dict(config_layer)
    registered_layer.update(registered_override)
    registered_effective_sha256 = canonical_json_sha256(registered_layer)
    if registered_effective_sha256 != context.row["expected_effective_config_sha256"]:
        raise PolicyImprovementSmokeError(
            "Applied method layer differs from the registered effective config."
        )
    merged.update(registered_override)
    merged.update(
        {
            "num_train_steps": 32,
            "eval_num_episodes": int(
                context.protocol["budgets"]["smoke"]["evaluation_records"]
            ),
            "log_interval": 1,
            "eval_interval": 32,
            "use_tqdm": False,
            "debug_checks": False,
        }
    )
    if method_id != "matched_ppo":
        merged["batch_size"] = int(stage0["upi_optimizer_batch_size"])
        merged["rollout_episodes_per_step"] = int(
            stage0["upi_rollout_episodes_per_step"]
        )
    rl_config = module.RLConfig(**merged)

    is_v2 = context.protocol.get("schema_name") == "policy_improvement_protocol_v2"
    train_dataset, seq_len, vocab_size, train_identifiers = (
        module.build_dataset_from_paths(
            dataset_paths=[str(context.dataset_root)],
            pool_size=int(context.protocol["dataset"]["splits"]["train"]["count"]),
            split="train",
        )
    )
    stage0_population: Mapping[str, Any] | None = None
    if is_v2:
        from scripts.policy_improvement_populations import (
            load_registered_populations,
            STAGE0_TRAINING_POPULATION_ID,
        )

        population_document = load_registered_populations(
            context.protocol,
            context.source_root,
        )
        stage0_population = population_document["populations"]["stage0_smoke"]
        if context.row.get("evaluation_population") != stage0_population.get(
            "population_id"
        ):
            raise PolicyImprovementSmokeError(
                "Stage 0 row differs from the registered train-only population."
            )
        stage0_training_population = population_document["populations"][
            STAGE0_TRAINING_POPULATION_ID
        ]
        if context.row.get("training_population") != stage0_training_population.get(
            "population_id"
        ):
            raise PolicyImprovementSmokeError(
                "Stage 0 row differs from the registered training population."
            )
        evaluation_dataset = module.select_materialized_dataset_records(
            train_dataset,
            list(stage0_population["indices"]),
            expected_record_sha256s=list(stage0_population["record_sha256s"]),
            expected_input_sha256s=list(stage0_population["input_sha256s"]),
        )
        # Stage 0 trains on the registered complement of its evaluation
        # population, so the schema-v5 train/evaluation disjointness invariant
        # holds without a Stage 0 exemption. The full split above stays bound
        # for manifest authentication only.
        training_dataset = module.select_materialized_dataset_records(
            train_dataset,
            list(stage0_training_population["indices"]),
            expected_record_sha256s=list(
                stage0_training_population["record_sha256s"]
            ),
            expected_input_sha256s=list(stage0_training_population["input_sha256s"]),
        )
        if set(dataset_input_sha256s(training_dataset)).intersection(
            dataset_input_sha256s(evaluation_dataset)
        ):
            raise PolicyImprovementSmokeError(
                "Stage 0 training and evaluation inputs overlap."
            )
        eval_seq_len = seq_len
        eval_vocab_size = vocab_size
        eval_identifiers = 0
        num_identifiers = train_identifiers
    else:
        stage0_training_population = None
        training_dataset = train_dataset
        evaluation_dataset, eval_seq_len, eval_vocab_size, eval_identifiers = (
            module.build_dataset_from_paths(
                dataset_paths=[str(context.dataset_root)],
                pool_size=int(
                    context.protocol["dataset"]["splits"]["validation"]["count"]
                ),
                split="validation",
            )
        )
        if set(dataset_input_sha256s(train_dataset)).intersection(
            dataset_input_sha256s(evaluation_dataset)
        ):
            raise PolicyImprovementSmokeError("Train and validation inputs overlap.")
        module.offset_puzzle_identifiers(evaluation_dataset, train_identifiers)
        num_identifiers = train_identifiers + eval_identifiers
    if (eval_seq_len, eval_vocab_size) != (seq_len, vocab_size):
        raise PolicyImprovementSmokeError("Train and evaluation shapes differ.")

    train_manifest = file_sha256(context.dataset_root / "manifests/train.json")
    module._validate_materialized_split_manifest(
        dataset_root=context.dataset_root,
        split="train",
        registered_sha256=train_manifest,
        dataset=train_dataset,
    )
    if not is_v2:
        validation_manifest = file_sha256(
            context.dataset_root / "manifests/validation.json"
        )
        module._validate_materialized_split_manifest(
            dataset_root=context.dataset_root,
            split="validation",
            registered_sha256=validation_manifest,
            dataset=evaluation_dataset,
        )

    checker, task, checker_kind = _task_config(
        rl_config, train_dataset, seq_len, module
    )
    env_config = PlanEditEnvConfig(
        max_edits=rl_config.max_edits,
        gamma=rl_config.gamma,
        reward_shaping=rl_config.reward_shaping,
        vocab_size=vocab_size,
        solved_threshold=(
            rl_config.solved_threshold
            if checker_kind in {"solution", "constraint", "progress", "feasibility"}
            else None
        ),
        task_type=rl_config.task_name,
        stop_action_mode=rl_config.stop_action_mode,
        stop_action_penalty=rl_config.stop_action_penalty,
        fail_terminal_reward=rl_config.fail_terminal_reward,
        solve_terminal_reward=rl_config.solve_terminal_reward,
        C_max=rl_config.C_max,
        disable_constraint_masking=rl_config.disable_constraint_masking,
    )
    env = PlanEditEnv(
        dataset=training_dataset, checker=checker, config=env_config, task_config=task
    )
    action_count = seq_len * vocab_size + 1
    env.set_stop_action_id(action_count - 1)

    evaluation_count = int(context.protocol["budgets"]["smoke"]["evaluation_records"])
    sources = dataset_source_build_metadata([str(context.dataset_root)])
    source = sources[0]
    metadata = {
        "source_build_metadata": sources,
        "train_pool_sha256": dataset_pool_sha256(
            training_dataset, len(training_dataset)
        ),
        "eval_pool_sha256": dataset_pool_sha256(evaluation_dataset, evaluation_count),
        "train_puzzle_identifier_ordered_sha256": ordered_record_sha256(
            dataset_puzzle_identifier_sha256s(training_dataset)
        ),
        "eval_puzzle_identifier_ordered_sha256": ordered_record_sha256(
            dataset_puzzle_identifier_sha256s(evaluation_dataset)[:evaluation_count]
        ),
        "dataset_source_names": [context.dataset_root.name],
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_identifiers": num_identifiers,
        "eval_puzzle_id_offset": 0 if is_v2 else train_identifiers,
        "materialization_seed": 0,
    }
    if stage0_population is not None:
        assert stage0_training_population is not None
        metadata.update(
            {
                "evaluation_population_id": stage0_population["population_id"],
                "evaluation_population_binding_sha256": stage0_population[
                    "binding_sha256"
                ],
                "evaluation_original_dataset_indices": list(
                    stage0_population["indices"]
                ),
                "training_population_id": stage0_training_population[
                    "population_id"
                ],
                "training_population_binding_sha256": stage0_training_population[
                    "binding_sha256"
                ],
                "training_population_ordered_record_sha256": (
                    stage0_training_population["ordered_record_sha256"]
                ),
                "training_population_count": stage0_training_population["count"],
            }
        )
    dataset_provenance = build_dataset_provenance(
        builder_name=str(source["builder_name"]),
        builder_version=source["builder_version"],
        generation_seed=source["generation_seed"],
        train_record_sha256s=dataset_sample_sha256s(training_dataset),
        eval_record_sha256s=dataset_sample_sha256s(
            evaluation_dataset, count=evaluation_count
        ),
        train_split="train",
        eval_split=str(context.row["evaluation_split"]),
        environment_config=dict(vars(env_config)),
        action_mask_config={
            "task_config_class": type(task).__name__ if task is not None else None,
            "task_config_name": getattr(task, "name", None),
            "disable_constraint_masking": env_config.disable_constraint_masking,
            "stop_action_mode": env._stop_mode,
            "stop_action_id": env.stop_action_id,
            "enable_undo": env._enable_undo,
            "undo_action_id": env.undo_action_id,
            "vocab_size": env.vocab_size,
            "num_actions": action_count,
            "masked_token_ids": [0, 1],
        },
        metadata=metadata,
    )

    architecture = context.protocol["architecture"]
    model_config = {
        "batch_size": rl_config.batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "num_puzzle_identifiers": max(num_identifiers, rl_config.batch_size),
        "vocab_size": vocab_size,
        "H_cycles": int(architecture["h_cycles"]),
        "L_cycles": int(architecture["l_cycles"]),
        "H_layers": 0,
        "L_layers": int(architecture["l_layers"]),
        "hidden_size": int(architecture["hidden_size"]),
        "expansion": 2.0,
        "num_heads": max(4, int(architecture["hidden_size"]) // 16),
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "no_ACT_continue": True,
        "rl_enable_value_head": True,
        "rl_enable_contraction": rl_config.enable_contraction,
        "rl_target_Lz": rl_config.target_Lz,
        "rl_target_Lv": rl_config.target_Lv,
        "rl_disable_value_head_norm": rl_config.disable_value_head_norm,
        "rl_enable_policy_head": True,
        "rl_num_actions": action_count,
        "rl_latent_projection_mode": rl_config.latent_projection_mode,
        "rl_latent_ball_radius": rl_config.latent_ball_radius,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRecursiveReasoningModel_ACTV1(model_config)
    initialization_sha256 = state_dict_sha256(model.state_dict())
    baseline = module.select_baseline_from_configs(None, [str(config_path)])
    if method_id == "matched_ppo":
        original_lookup = baseline.get_yaml_key

        def smoke_lookup(name: str, default: object) -> object:
            if name == "ppo_num_steps":
                return int(stage0["ppo_rollout_environment_interactions"])
            return original_lookup(name, default)

        baseline = module.BaselineSelection(
            baseline.selected_baseline,
            baseline.yaml_algorithm,
            smoke_lookup,
        )
    trainer = module.build_trainer(
        model=model,
        env=env,
        rl_cfg=rl_config,
        device=device,
        baseline_selection=baseline,
        cli_baseline=None,
        verbose=False,
    )
    if method_id == "matched_ppo":
        if type(trainer).__name__ != "PPOTrainer" or trainer.config.num_steps != 16:
            raise PolicyImprovementSmokeError("Registered PPO smoke trainer differs.")
    elif not isinstance(trainer, UPITrmTrainer):
        raise PolicyImprovementSmokeError("Registered UPI smoke trainer differs.")
    _bind_registered_checker(trainer, checker)
    effective_config = {
        "schema_name": "policy_improvement_smoke_effective_config_v1",
        "protocol_sha256": context.protocol_sha256,
        "registry_row_sha256": context.registry_row_sha256,
        "method_id": method_id,
        "run_id": context.row["run_id"],
        "training_seed": context.row["seed"],
        "runtime_artifact_sha256": context.runtime_sha256,
        "launcher_sha256": context.launcher_sha256,
        "producer_manifest_sha256": context.producer_manifest_sha256,
        "rl_config": module._config_dict(rl_config),
        "trainer_config": _optional_trainer_config_dict(trainer, module),
        "model_config": module._config_dict(model.config),
        "dataset_provenance_sha256": canonical_json_sha256(dataset_provenance),
        "config_source_sha256": file_sha256(config_path),
        "execution_device": module._canonical_device(device),
        "environment_interactions": 32,
        "checkpoint_environment_interactions": [16, 32],
        "evaluation_records": evaluation_count,
    }
    effective_sha256 = canonical_json_sha256(effective_config)
    evidence_identity = {
        "schema_version": 2,
        "run_id": context.row["run_id"],
        "algorithm": ("trm_ppo" if method_id == "matched_ppo" else "upi_trm"),
        "training_seed": context.row["seed"],
        "producer_git_commit": context.producer_commit,
        "effective_config_sha256": effective_sha256,
        "effective_config": effective_config,
        "dataset_provenance_sha256": canonical_json_sha256(dataset_provenance),
        "runtime_artifact_sha256": context.runtime_sha256,
    }

    run_identity = None
    if str(rl_config.training_protocol) == "fixed_base_exact":
        fixed_args = SimpleNamespace(
            config=[str(config_path)],
            confirmatory_cell=method_id,
            confirmatory_tier="debug",
            run_id=context.row["run_id"],
            seed=context.row["seed"],
            backbone="trm",
            train_split="train",
            eval_split=str(context.row["evaluation_split"]),
            env_step_budget=32,
            save_interval=0,
            log_env_interval=16,
            eval_env_interval=32,
            save_env_interval=16,
            puzzle_emb_lr=0.0,
            puzzle_emb_weight_decay=0.0,
            imitation_pretrain=False,
            imitation_epochs=0,
            debug_checks=False,
        )
        exact_effective = module._fixed_base_effective_config(
            args=fixed_args,
            rl_config=module._config_dict(rl_config),
            model_config=module._config_dict(model.config),
            execution_device=module._canonical_device(device),
            train_record_count=len(training_dataset),
            eval_record_count=evaluation_count,
            dataset_provenance=dataset_provenance,
            initialization_kind="random",
            initialization_artifact_sha256=None,
            registered_assignment={
                "attempt_index": 0,
                "registry_sha256": context.registry_sha256,
            },
            runtime_artifact_sha256=context.runtime_sha256,
        )
        run_identity = build_run_identity(
            run_id=str(context.row["run_id"]),
            training_seed=int(context.row["seed"]),
            git_lookup_root=context.source_root,
            effective_config=exact_effective,
            dataset_provenance=dataset_provenance,
            initialization_kind="random",
            initialization_artifact_sha256=None,
        )

    return SmokeSession(
        model=model,
        trainer=trainer,
        rl_config=rl_config,
        env_config=env_config,
        train_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset,
        checker=checker,
        task_config=task,
        dataset_provenance=dataset_provenance,
        effective_config=effective_config,
        effective_config_sha256=effective_sha256,
        initialization_sha256=initialization_sha256,
        device=device,
        config_path=config_path,
        method_config_sha256=method_config_sha256,
        run_identity=run_identity,
        evidence_identity=evidence_identity,
        evaluation_population=(
            dict(stage0_population) if stage0_population is not None else None
        ),
        training_population=(
            dict(stage0_training_population)
            if stage0_training_population is not None
            else None
        ),
    )


def build_stage0_validation_session(
    context: SmokeContext,
    training_module: Any,
) -> SmokeSession:
    """Reconstruct a disposable registered Stage-0 session without running it."""

    if (
        context.row.get("phase") != "stage0_smoke"
        or context.row.get("tier") != "smoke"
        or context.segment_budget not in {16, 32}
        or context.segment_name not in {"prepare", "resume"}
    ):
        raise PolicyImprovementSmokeError(
            "Checkpoint validation accepts only a registered Stage-0 session."
        )
    return _build_session(context, training_module)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyImprovementSmokeError(
                "Stage 0 segment manifest contains a duplicate JSON key."
            )
        result[key] = value
    return result


def _stable_regular_file(path: Path) -> tuple[bytes, dict[str, object]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PolicyImprovementSmokeError(
            f"Stage 0 output {path.name!r} cannot be opened safely."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PolicyImprovementSmokeError(
                "Stage 0 outputs must be singly linked regular files."
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            chunks.append(block)
            digest.update(block)
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
            raise PolicyImprovementSmokeError(
                "Stage 0 output changed while it was authenticated."
            )
        return b"".join(chunks), {
            "bytes": before.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _segment_inventory(path: Path) -> tuple[dict[str, dict[str, object]], set[str]]:
    files: dict[str, dict[str, object]] = {}
    directories: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        _canonical_relative_path(relative, name="segment member")
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PolicyImprovementSmokeError("Stage 0 output contains a symlink.")
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PolicyImprovementSmokeError(
                "Stage 0 output is not a singly linked regular file."
            )
        inode = (info.st_dev, info.st_ino)
        if inode in seen_inodes:
            raise PolicyImprovementSmokeError("Stage 0 outputs alias one inode.")
        seen_inodes.add(inode)
        _, identity = _stable_regular_file(candidate)
        files[relative] = identity
    return files, directories


def _validate_segment_generation(
    context: SmokeContext,
    path: Path,
    *,
    expected_budget: int,
) -> tuple[Path, dict[str, Any]]:
    try:
        directory_before = path.lstat()
    except OSError as exc:
        raise PolicyImprovementSmokeError(
            "Resume requires a complete immutable Stage 0 segment."
        ) from exc
    if (
        stat.S_ISLNK(directory_before.st_mode)
        or not stat.S_ISDIR(directory_before.st_mode)
        or directory_before.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSmokeError("Stage 0 segment directory is invalid.")
    manifest_bytes, _ = _stable_regular_file(path / "MANIFEST.json")
    try:
        manifest = json.loads(
            manifest_bytes.decode("ascii"), object_pairs_hook=_strict_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementSmokeError(
            "Stage 0 segment manifest is invalid."
        ) from exc
    expected_manifest_fields = {
        "schema_name",
        "schema_version",
        "protocol_sha256",
        "registry_sha256",
        "registry_row_sha256",
        "run_id",
        "method_id",
        "segment",
        "environment_interactions",
        "parent_checkpoint_sha256",
        "result_status",
        "prior_failed_attempts",
        "outputs",
        "storage_bytes",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise PolicyImprovementSmokeError(
            "Stage 0 segment manifest field inventory differs."
        )
    expected_segment = "prepare" if expected_budget == 16 else "resume"
    if (
        manifest["schema_name"] != "policy_improvement_smoke_segment_v1"
        or manifest["schema_version"] != SMOKE_SEGMENT_SCHEMA_VERSION
        or manifest["protocol_sha256"] != context.protocol_sha256
        or manifest["registry_sha256"] != context.registry_sha256
        or manifest["registry_row_sha256"] != context.registry_row_sha256
        or manifest["run_id"] != context.row["run_id"]
        or manifest["method_id"] != context.row["method_id"]
        or manifest["segment"] != expected_segment
        or manifest["environment_interactions"] != expected_budget
    ):
        raise PolicyImprovementSmokeError("Stage 0 segment identity differs.")
    if expected_budget == 16 and (
        manifest["parent_checkpoint_sha256"] is not None
        or manifest["result_status"] is not None
    ):
        raise PolicyImprovementSmokeError("Prepare segment lineage is invalid.")
    if expected_budget == 32:
        _sha256(manifest["parent_checkpoint_sha256"], name="segment parent checkpoint")
        if manifest["result_status"] != "complete":
            raise PolicyImprovementSmokeError("Resume segment result is incomplete.")
    prior_failed_attempts = manifest["prior_failed_attempts"]
    current_failed_attempts = _prior_failed_attempt_inventory(context)
    exact_current_segment = expected_budget == context.segment_budget
    if not isinstance(prior_failed_attempts, list) or (
        prior_failed_attempts != current_failed_attempts
        if exact_current_segment
        else any(
            attempt not in current_failed_attempts for attempt in prior_failed_attempts
        )
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 segment does not bind the prior failed-attempt inventory."
        )
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"checkpoint", "files"}:
        raise PolicyImprovementSmokeError("Stage 0 output inventory is invalid.")
    files = outputs["files"]
    if not isinstance(files, dict) or not files:
        raise PolicyImprovementSmokeError("Stage 0 segment has no registered outputs.")
    canonical_files: dict[str, dict[str, object]] = {}
    expected_directories: set[str] = set()
    for raw_name, raw_entry in files.items():
        name = _canonical_relative_path(raw_name, name="segment output")
        if name == "MANIFEST.json":
            raise PolicyImprovementSmokeError(
                "Stage 0 manifest cannot include its own digest."
            )
        if (
            not isinstance(raw_entry, dict)
            or set(raw_entry) != {"bytes", "sha256"}
            or isinstance(raw_entry["bytes"], bool)
            or not isinstance(raw_entry["bytes"], int)
            or raw_entry["bytes"] < 0
        ):
            raise PolicyImprovementSmokeError("Stage 0 output identity is invalid.")
        canonical_files[name] = {
            "bytes": raw_entry["bytes"],
            "sha256": _sha256(raw_entry["sha256"], name=f"output {name}"),
        }
        parts = PurePosixPath(name).parts
        expected_directories.update(
            PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    actual_files, actual_directories = _segment_inventory(path)
    manifest_identity = actual_files.pop("MANIFEST.json", None)
    if manifest_identity is None or actual_files != canonical_files:
        raise PolicyImprovementSmokeError(
            "Stage 0 output bytes differ from the segment manifest."
        )
    if actual_directories != expected_directories:
        raise PolicyImprovementSmokeError(
            "Stage 0 output directory inventory differs from the manifest."
        )
    checkpoint = outputs["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise PolicyImprovementSmokeError("Stage 0 checkpoint identity is invalid.")
    checkpoint_name = _canonical_relative_path(
        checkpoint["path"], name="segment checkpoint"
    )
    if checkpoint != {
        "path": checkpoint_name,
        **canonical_files.get(checkpoint_name, {}),
    }:
        raise PolicyImprovementSmokeError(
            "Stage 0 checkpoint differs from the registered file identity."
        )
    if (
        isinstance(manifest["storage_bytes"], bool)
        or not isinstance(manifest["storage_bytes"], int)
        or manifest["storage_bytes"]
        != sum(int(item["bytes"]) for item in canonical_files.values())
    ):
        raise PolicyImprovementSmokeError("Stage 0 storage accounting differs.")
    directory_after = path.lstat()
    if (
        directory_before.st_dev,
        directory_before.st_ino,
        directory_before.st_mtime_ns,
        directory_before.st_ctime_ns,
    ) != (
        directory_after.st_dev,
        directory_after.st_ino,
        directory_after.st_mtime_ns,
        directory_after.st_ctime_ns,
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 segment changed while it was authenticated."
        )
    return path / checkpoint_name, manifest


def _parent_segment(context: SmokeContext) -> tuple[Path, dict[str, Any]]:
    return _validate_segment_generation(
        context,
        context.run_root / "segments/env_000000016",
        expected_budget=16,
    )


def _parent_gpu_utilization_summary(
    context: SmokeContext,
    *,
    expected_device_type: str,
) -> GpuUtilizationSummary:
    path = context.run_root / "segments/env_000000016"
    _validate_segment_generation(context, path, expected_budget=16)
    payload, _ = _stable_regular_file(path / "training_gpu_utilization.json")
    try:
        value = json.loads(
            payload.decode("ascii"), object_pairs_hook=_strict_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementSmokeError(
            "Parent GPU utilization evidence is invalid."
        ) from exc
    return _validate_gpu_utilization_summary(
        value,
        expected_device_type=expected_device_type,
    )


def _checkpoint_timing_document(seconds: float) -> dict[str, object]:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise PolicyImprovementSmokeError(
            "Checkpoint wall time must be a nonnegative finite number."
        )
    return {
        "schema_name": "policy_improvement_checkpoint_timing_v2",
        "schema_version": 1,
        "checkpoint_seconds": seconds,
    }


def _parent_checkpoint_seconds(context: SmokeContext) -> float:
    path = context.run_root / "segments/env_000000016"
    _validate_segment_generation(context, path, expected_budget=16)
    payload, _ = _stable_regular_file(path / "checkpoint_timing.json")
    try:
        value = json.loads(
            payload.decode("ascii"), object_pairs_hook=_strict_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementSmokeError(
            "Parent checkpoint timing evidence is invalid."
        ) from exc
    if not isinstance(value, Mapping) or set(value) != {
        "schema_name",
        "schema_version",
        "checkpoint_seconds",
    }:
        raise PolicyImprovementSmokeError("Parent checkpoint timing inventory differs.")
    seconds = value["checkpoint_seconds"]
    if (
        value["schema_name"] != "policy_improvement_checkpoint_timing_v2"
        or value["schema_version"] != 1
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
    ):
        raise PolicyImprovementSmokeError(
            "Parent checkpoint timing evidence is invalid."
        )
    return float(seconds)


def _assert_restored_state(
    module: Any, raw: Mapping[str, object], field: str, *, label: str
) -> None:
    value = raw.get(field)
    if not isinstance(value, Mapping) or state_dict_sha256(
        module.state_dict()
    ) != state_dict_sha256(value):
        raise PolicyImprovementSmokeError(
            f"Restored {label} differs from the parent checkpoint."
        )


def _restore_parent(
    context: SmokeContext,
    session: SmokeSession,
    module: Any,
) -> str | None:
    if context.segment_name == "prepare":
        return None
    checkpoint, manifest = _parent_segment(context)
    checkpoint_sha256 = manifest["outputs"]["checkpoint"]["sha256"]
    if str(context.row["method_id"]) == "matched_ppo":
        payload, observed = load_stable_checkpoint(
            checkpoint, expected_sha256=checkpoint_sha256
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("parent_checkpoint_sha256") is not None
        ):
            raise PolicyImprovementSmokeError(
                "The prepare PPO checkpoint has unexpected parent lineage."
            )
        probe = _build_session(context, module)
        validate_ppo_smoke_checkpoint(
            payload,
            probe.trainer,
            expected_identity=session.effective_config,
            validate_only=True,
        )
        validate_ppo_smoke_checkpoint(
            payload,
            session.trainer,
            expected_identity=session.effective_config,
            validate_only=False,
        )
        if observed != checkpoint_sha256:
            raise PolicyImprovementSmokeError("PPO parent checkpoint digest changed.")
    else:
        raw, observed = module._load_checkpoint_payload(
            str(checkpoint), expected_sha256=checkpoint_sha256
        )
        if not isinstance(raw, dict):
            raise PolicyImprovementSmokeError("UPI parent checkpoint is invalid.")
        validate_policy_improvement_smoke_identity(
            raw.get("policy_improvement_smoke_identity"),
            context,
            environment_interactions=16,
            parent_checkpoint_sha256=None,
        )
        if raw.get("evidence_identity") != session.evidence_identity:
            raise PolicyImprovementSmokeError(
                "UPI parent checkpoint evidence identity differs."
            )
        expected_run_identity = session.run_identity
        if (
            expected_run_identity is not None
            and raw.get("run_identity") != expected_run_identity
        ):
            raise PolicyImprovementSmokeError("Fixed-base parent run identity differs.")
        module.resume_from_checkpoint(
            str(checkpoint),
            session.model,
            session.trainer,
            str(session.device),
            None,
            expected_dataset_provenance=session.dataset_provenance,
            expected_run_identity=expected_run_identity,
            expected_checkpoint_sha256=checkpoint_sha256,
            allow_legacy_warm_start=False,
        )
        if observed != checkpoint_sha256:
            raise PolicyImprovementSmokeError("UPI parent checkpoint digest changed.")
        trainer = session.trainer
        _assert_restored_state(
            session.model, raw, "model_state_dict", label="UPI model"
        )
        _assert_restored_state(
            trainer.policy_model_old,
            raw,
            "policy_model_old_state_dict",
            label="UPI deployed policy",
        )
        _assert_restored_state(
            trainer.policy_model_candidate,
            raw,
            "policy_model_candidate_state_dict",
            label="UPI candidate policy",
        )
        _assert_restored_state(
            trainer.target_model,
            raw,
            "target_model_state_dict",
            label="UPI target model",
        )
        if str(context.row["method_id"]) == "legacy_parameter_interpolation":
            if trainer.policy_model_old is not session.model:
                raise PolicyImprovementSmokeError(
                    "Legacy deployed policy lost its registered model alias."
                )
            for attribute, field in (
                (
                    "preinterpolation_policy_base",
                    "preinterpolation_policy_base_state_dict",
                ),
                (
                    "preinterpolation_policy_candidate",
                    "preinterpolation_policy_candidate_state_dict",
                ),
            ):
                snapshot = getattr(trainer, attribute, None)
                if snapshot is None:
                    raise PolicyImprovementSmokeError(
                        "Legacy pre-interpolation snapshot is missing after restore."
                    )
                _assert_restored_state(snapshot, raw, field, label=attribute)
            if raw.get("preinterpolation_pair_generation") != getattr(
                trainer, "_preinterpolation_pair_generation", None
            ):
                raise PolicyImprovementSmokeError(
                    "Legacy pre-interpolation generation differs after restore."
                )
    if session.trainer.get_env_step_count() != 16:
        raise PolicyImprovementSmokeError("Restored parent is not at interaction 16.")
    return checkpoint_sha256


def _train_to_budget(session: SmokeSession, budget: int) -> dict[str, float]:
    latest: dict[str, float] = {}
    while session.trainer.get_env_step_count() < budget:
        before = session.trainer.get_env_step_count()
        remaining = budget - before
        if type(session.trainer).__name__ == "PPOTrainer":
            if remaining < session.trainer.config.num_steps:
                raise PolicyImprovementSmokeError(
                    "PPO segment cannot end with a partial rollout."
                )
            cap = session.trainer.config.num_steps
        else:
            cap = remaining
        raw = session.trainer.train_step(max_env_steps_to_collect=cap)
        latest = {}
        for name, value in raw.items():
            if not isinstance(value, (int, float)):
                continue
            measured = float(value)
            if not math.isfinite(measured):
                raise PolicyImprovementSmokeError(
                    f"Stage 0 trainer emitted nonfinite metric {name!r}."
                )
            latest[name] = measured
        after = session.trainer.get_env_step_count()
        if after <= before or after > budget:
            raise PolicyImprovementSmokeError(
                "Stage 0 trainer violated its exact interaction cap."
            )
    return latest


class _GpuTrainingSampler:
    """Poll CUDA utilization only while the registered trainer is running."""

    def __init__(
        self,
        device: torch.device,
        *,
        interval_seconds: float = 0.1,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise PolicyImprovementSmokeError(
                "GPU utilization sampling interval must be positive."
            )
        self._device = device
        self._interval_seconds = interval_seconds
        self._samples: list[float] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        utilization = getattr(torch.cuda, "utilization", None)
        if not callable(utilization):
            raise PolicyImprovementSmokeError(
                "CUDA utilization sampling is unavailable."
            )
        percentage = float(utilization(self._device))
        fraction = percentage / 100.0
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise PolicyImprovementSmokeError(
                "GPU utilization sample is outside [0, 1]."
            )
        self._samples.append(fraction)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._sample()
            except BaseException as exc:
                self._error = exc
                self._stop.set()
                return

    def start(self) -> None:
        if self._device.type != "cuda":
            return
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="policy-smoke-gpu-utilization",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> GpuUtilizationSummary:
        if self._device.type != "cuda":
            return GpuUtilizationSummary(
                device_type=self._device.type,
                sampling_interval_seconds=None,
                samples=(),
            )
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise PolicyImprovementSmokeError(
                "GPU utilization interval sampling failed."
            ) from self._error
        self._sample()
        if len(self._samples) < 2:
            raise PolicyImprovementSmokeError(
                "GPU utilization interval sampling produced fewer than two samples."
            )
        return GpuUtilizationSummary(
            device_type="cuda",
            sampling_interval_seconds=self._interval_seconds,
            samples=tuple(self._samples),
        )


def _validate_gpu_utilization_summary(
    value: object,
    *,
    expected_device_type: str,
) -> GpuUtilizationSummary:
    if not isinstance(value, Mapping) or set(value) != {
        "device_type",
        "sampling_interval_seconds",
        "samples",
    }:
        raise PolicyImprovementSmokeError(
            "GPU utilization evidence has an invalid inventory."
        )
    if value["device_type"] != expected_device_type:
        raise PolicyImprovementSmokeError(
            "GPU utilization evidence names a different device type."
        )
    raw_samples = value["samples"]
    if not isinstance(raw_samples, list):
        raise PolicyImprovementSmokeError(
            "GPU utilization evidence samples are invalid."
        )
    samples = tuple(float(sample) for sample in raw_samples)
    if any(not math.isfinite(sample) or not 0.0 <= sample <= 1.0 for sample in samples):
        raise PolicyImprovementSmokeError(
            "GPU utilization evidence contains an invalid sample."
        )
    interval = value["sampling_interval_seconds"]
    if expected_device_type == "cuda":
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(float(interval))
            or float(interval) <= 0.0
            or len(samples) < 2
        ):
            raise PolicyImprovementSmokeError(
                "CUDA utilization evidence is incomplete."
            )
        canonical_interval: float | None = float(interval)
    else:
        if interval is not None or samples:
            raise PolicyImprovementSmokeError(
                "Non-CUDA execution cannot claim GPU utilization samples."
            )
        canonical_interval = None
    return GpuUtilizationSummary(
        device_type=expected_device_type,
        sampling_interval_seconds=canonical_interval,
        samples=samples,
    )


def _gpu_utilization_document(
    summary: GpuUtilizationSummary,
) -> dict[str, object]:
    return {
        "device_type": summary.device_type,
        "sampling_interval_seconds": summary.sampling_interval_seconds,
        "samples": list(summary.samples),
    }


def _combine_gpu_utilization_summaries(
    first: GpuUtilizationSummary,
    second: GpuUtilizationSummary,
) -> GpuUtilizationSummary:
    if first.device_type != second.device_type:
        raise PolicyImprovementSmokeError(
            "GPU utilization segment device types differ."
        )
    if first.device_type != "cuda":
        return _validate_gpu_utilization_summary(
            _gpu_utilization_document(second),
            expected_device_type=second.device_type,
        )
    if first.sampling_interval_seconds != second.sampling_interval_seconds:
        raise PolicyImprovementSmokeError(
            "GPU utilization segment sampling intervals differ."
        )
    return GpuUtilizationSummary(
        device_type="cuda",
        sampling_interval_seconds=first.sampling_interval_seconds,
        samples=(*first.samples, *second.samples),
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    _write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _policy_improvement_smoke_identity(
    context: SmokeContext,
    *,
    environment_interactions: int,
    parent_checkpoint_sha256: str | None,
) -> dict[str, object]:
    if environment_interactions not in {16, 32}:
        raise PolicyImprovementSmokeError(
            "Stage 0 checkpoint identity has an unsupported interaction count."
        )
    if environment_interactions == 16:
        if parent_checkpoint_sha256 is not None:
            raise PolicyImprovementSmokeError(
                "Stage 0 prepare checkpoint must not name a parent."
            )
    else:
        _sha256(
            parent_checkpoint_sha256,
            name="Stage 0 resume parent checkpoint SHA-256",
        )
    return {
        "schema_version": 1,
        "run_id": context.row["run_id"],
        "method_id": context.row["method_id"],
        "seed": context.row["seed"],
        "environment_interactions": environment_interactions,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }


def validate_policy_improvement_smoke_identity(
    value: object,
    context: SmokeContext,
    *,
    environment_interactions: int,
    parent_checkpoint_sha256: str | None,
) -> dict[str, object]:
    """Validate the checkpoint-native Stage-0 lineage and row identity."""

    expected = _policy_improvement_smoke_identity(
        context,
        environment_interactions=environment_interactions,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise PolicyImprovementSmokeError(
            "Checkpoint Stage 0 identity differs from its registered row or lineage."
        )
    return expected


def _rewrite_checkpoint_with_smoke_identity(
    checkpoint: Path,
    payload: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    """Durably rewrite one still-private checkpoint with its smoke lineage."""

    try:
        checkpoint_info = checkpoint.lstat()
        parent_info = checkpoint.parent.lstat()
    except OSError as exc:
        raise PolicyImprovementSmokeError(
            "Stage 0 checkpoint cannot be rebound to its semantic identity."
        ) from exc
    if (
        stat.S_ISLNK(checkpoint_info.st_mode)
        or not stat.S_ISREG(checkpoint_info.st_mode)
        or checkpoint_info.st_nlink != 1
        or stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 checkpoint identity requires a private regular file."
        )
    if "policy_improvement_smoke_identity" in payload:
        raise PolicyImprovementSmokeError(
            "Stage 0 checkpoint already contains a smoke identity."
        )
    rebound = dict(payload)
    rebound["policy_improvement_smoke_identity"] = dict(identity)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint.name}.smoke-identity.",
        suffix=".tmp",
        dir=str(checkpoint.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(rebound, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)
        _fsync_dir(checkpoint.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PolicyImprovementSmokeError(
            "This host lacks generation-safe renameat2 support."
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
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise PolicyImprovementSmokeError("Stage 0 segment already exists.")
        raise OSError(error, os.strerror(error), str(destination))


def _checkpoint_output(
    context: SmokeContext,
    session: SmokeSession,
    module: Any,
    staging: Path,
    parent_sha256: str | None,
) -> tuple[Path, str, dict[str, Any]]:
    checkpoints = staging / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    before_model_sha256, before_role_sha256s = _model_state_identity(session)
    if str(context.row["method_id"]) == "matched_ppo":
        payload = build_ppo_smoke_checkpoint(
            session.trainer,
            identity=session.effective_config,
            parent_checkpoint_sha256=parent_sha256,
        )
        checkpoint = checkpoints / f"ppo_checkpoint_env_{context.segment_budget:09d}.pt"
        digest = publish_checkpoint(payload, checkpoint)
        loaded, observed = load_stable_checkpoint(
            checkpoint,
            expected_sha256=digest,
        )
        validate_ppo_smoke_checkpoint(
            loaded,
            session.trainer,
            expected_identity=session.effective_config,
            validate_only=True,
        )
        loaded_role_sha256s = {
            "model": state_dict_sha256(loaded["model_state_dict"]),
        }
        progress = loaded["progress"]["environment_interactions"]
        embedded_parent = loaded["parent_checkpoint_sha256"]
    else:
        lineage = None
        if session.run_identity is not None:
            lineage = build_checkpoint_lineage(
                parent_checkpoint_sha256=parent_sha256,
                parent_checkpoint_step=16 if parent_sha256 is not None else None,
                parent_environment_steps=16 if parent_sha256 is not None else None,
            )
        checkpoint_name = module.save_checkpoint(
            session.model,
            session.trainer,
            context.segment_budget,
            str(checkpoints),
            None,
            session.rl_config,
            session.dataset_provenance,
            session.run_identity,
            lineage,
            session.evidence_identity,
            training_seed=int(context.row["seed"]),
            training_run_id=str(context.row["run_id"]),
            config_source_paths=[str(session.config_path)],
        )
        checkpoint = Path(checkpoint_name)
        raw_checkpoint, _ = module._load_checkpoint_payload(str(checkpoint))
        if not isinstance(raw_checkpoint, Mapping):
            raise PolicyImprovementSmokeError(
                "Published UPI checkpoint payload is invalid."
            )
        smoke_identity = _policy_improvement_smoke_identity(
            context,
            environment_interactions=context.segment_budget,
            parent_checkpoint_sha256=parent_sha256,
        )
        _rewrite_checkpoint_with_smoke_identity(
            checkpoint,
            raw_checkpoint,
            smoke_identity,
        )
        digest = file_sha256(checkpoint)
        module.resume_from_checkpoint(
            str(checkpoint),
            session.model,
            session.trainer,
            str(session.device),
            expected_dataset_provenance=session.dataset_provenance,
            expected_run_identity=session.run_identity,
            expected_checkpoint_sha256=(
                digest if session.run_identity is not None else None
            ),
        )
        loaded, observed = module._load_checkpoint_payload(
            str(checkpoint), expected_sha256=digest
        )
        if not isinstance(loaded, Mapping):
            raise PolicyImprovementSmokeError(
                "Published UPI checkpoint payload is invalid after rebinding."
            )
        validate_policy_improvement_smoke_identity(
            loaded.get("policy_improvement_smoke_identity"),
            context,
            environment_interactions=context.segment_budget,
            parent_checkpoint_sha256=parent_sha256,
        )
        role_fields = {
            "model": "model_state_dict",
            "policy_model_old": "policy_model_old_state_dict",
            "policy_model_candidate": "policy_model_candidate_state_dict",
            "target_model": "target_model_state_dict",
            "preinterpolation_policy_base": ("preinterpolation_policy_base_state_dict"),
            "preinterpolation_policy_candidate": (
                "preinterpolation_policy_candidate_state_dict"
            ),
        }
        loaded_role_sha256s = {
            role: state_dict_sha256(loaded[field])
            for role, field in role_fields.items()
            if field in loaded
        }
        progress = loaded["progress"]["env_steps"]
        embedded_lineage = loaded.get("checkpoint_lineage")
        embedded_parent = smoke_identity["parent_checkpoint_sha256"]
        if (
            isinstance(embedded_lineage, Mapping)
            and embedded_lineage.get("parent_checkpoint_sha256") != embedded_parent
        ):
            raise PolicyImprovementSmokeError(
                "Fixed-base checkpoint lineage differs from its Stage 0 identity."
            )

    after_model_sha256, after_role_sha256s = _model_state_identity(session)
    if (
        observed != digest
        or isinstance(progress, bool)
        or not isinstance(progress, int)
        or progress != context.segment_budget
        or embedded_parent != parent_sha256
        or before_model_sha256 != after_model_sha256
        or before_role_sha256s != after_role_sha256s
        or loaded_role_sha256s != after_role_sha256s
        or canonical_json_sha256(loaded_role_sha256s) != after_model_sha256
    ):
        raise PolicyImprovementSmokeError(
            "Published checkpoint failed strict byte, progress, lineage, or model-state validation."
        )
    return (
        checkpoint,
        digest,
        {
            "schema_name": "policy_improvement_checkpoint_validation_v1",
            "schema_version": 1,
            "validator": "policy_improvement_smoke_runtime",
            "validator_execution_identity": {
                "role": "policy-improvement-training",
                "source_git_commit": context.producer_commit,
                "runtime_sha256": context.runtime_sha256,
                "runtime_profile_sha256": context.runtime_profile_sha256,
                "selected_source_manifest_sha256": (
                    context.selected_source_manifest_sha256
                ),
                "runtime_authorization_sha256": (context.runtime_authorization_sha256),
                "launcher_sha256": context.launcher_sha256,
            },
            "run_id": context.row["run_id"],
            "method_id": context.row["method_id"],
            "snapshot_kind": "interaction_matched",
            "environment_interactions": context.segment_budget,
            "checkpoint_sha256": digest,
            "parent_checkpoint_sha256": parent_sha256,
            "model_state_sha256": after_model_sha256,
            "role_state_sha256s": after_role_sha256s,
            "strict_resume_validated": True,
        },
    )


def _record_evaluation_accounting(
    trainer: Any, callback: Callable[[], tuple[float, float, dict[str, Any]]]
) -> tuple[float, float, dict[str, Any]]:
    before, _, _ = aggregate_model_compute(trainer._model_roles())
    started = time.perf_counter()
    try:
        return callback()
    finally:
        trainer._evaluation_wall_time_seconds += time.perf_counter() - started
        after, _, _ = aggregate_model_compute(trainer._model_roles())
        trainer._evaluation_model_work = add_model_counters(
            trainer._evaluation_model_work,
            subtract_model_counters(after, before),
        )


def _variant_specs(
    context: SmokeContext, session: SmokeSession
) -> list[tuple[str, Any, Callable[..., Any] | None, bool, tuple[Any, ...]]]:
    method_id = str(context.row["method_id"])
    trainer = session.trainer
    if "exact_mixture" in policy_variants_for_method(method_id):
        checks = {
            "calls": 0,
            "maximum_probability_error": 0.0,
            "proof_model_work": zero_model_counters(),
            "proof_wall_time_seconds": 0.0,
        }

        def checked_mixture(
            x_batch: Any,
            y_batch: Any,
            n: int,
            action_mask: Any = None,
            z: Any = None,
        ) -> Any:
            mixed, latent = trainer._mixed_policy_dist(
                x_batch,
                y_batch,
                n=n,
                action_mask=action_mask,
                z=z,
            )
            proof_before, _, _ = aggregate_model_compute(trainer._model_roles())
            proof_started = time.perf_counter()
            try:
                base, base_latent = trainer.policy_model_old.policy_dist(
                    x_batch,
                    y_batch,
                    n=n,
                    action_mask=action_mask,
                    z=z,
                )
                if z is None:
                    candidate, _ = trainer.policy_model_candidate.policy_dist(
                        x_batch,
                        y_batch,
                        n=n,
                        action_mask=action_mask,
                        z=None,
                    )
                else:
                    candidate, _ = trainer.policy_model_candidate.policy_dist(
                        x_batch,
                        y_batch,
                        n=0,
                        action_mask=action_mask,
                        z=base_latent,
                    )
            finally:
                checks["proof_wall_time_seconds"] = float(
                    checks["proof_wall_time_seconds"]
                ) + (time.perf_counter() - proof_started)
                proof_after, _, _ = aggregate_model_compute(trainer._model_roles())
                checks["proof_model_work"] = add_model_counters(
                    checks["proof_model_work"],
                    subtract_model_counters(proof_after, proof_before),
                )
            expected = (
                1.0 - float(session.rl_config.mixture_alpha)
            ) * base.probs + float(session.rl_config.mixture_alpha) * candidate.probs
            epsilon = float(session.rl_config.policy_epsilon)
            if epsilon > 0.0:
                if action_mask is None:
                    uniform = torch.full_like(expected, 1.0 / expected.shape[-1])
                else:
                    mask = action_mask
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(0).expand(expected.shape[0], -1)
                    uniform = mask.float() / mask.float().sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1)
                expected = (1.0 - epsilon) * expected + epsilon * uniform
            error = float((mixed.probs - expected).abs().max().item())
            checks["calls"] += 1
            checks["maximum_probability_error"] = max(
                checks["maximum_probability_error"], error
            )
            if error > 1e-7:
                raise PolicyImprovementSmokeError(
                    "Production exact mixture differs from pointwise probability mixing."
                )
            return mixed, latent

        setattr(checked_mixture, "_smoke_checks", checks)
        return [
            ("base", trainer.policy_model_old, None, False, ()),
            ("candidate", trainer.policy_model_candidate, None, False, ()),
            (
                "exact_mixture",
                trainer.policy_model_old,
                checked_mixture,
                False,
                (trainer.policy_model_candidate,),
            ),
        ]
    if method_id == "legacy_parameter_interpolation":
        if (
            trainer.preinterpolation_policy_base is None
            or trainer.preinterpolation_policy_candidate is None
        ):
            raise PolicyImprovementSmokeError(
                "Legacy smoke did not retain its pre-interpolation policy pair."
            )
        return [
            ("base", trainer.preinterpolation_policy_base, None, False, ()),
            ("candidate", trainer.preinterpolation_policy_candidate, None, False, ()),
            ("realized_policy", trainer.policy_model_old, None, False, ()),
        ]
    return [("realized_policy", session.model, None, True, ())]


def _evaluation_payload(
    context: SmokeContext,
    session: SmokeSession,
    variant: str,
    model: Any,
    policy_dist_fn: Callable[..., Any] | None,
    greedy: bool,
    additional_models: tuple[Any, ...],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
]:
    count = int(context.protocol["budgets"]["smoke"]["evaluation_records"])
    seed = int(session.rl_config.eval_seed)

    def evaluate() -> tuple[float, float, dict[str, Any]]:
        session.evaluation_started = True
        return evaluate_plan_policy_with_scores(
            model=model,
            dataset=session.evaluation_dataset,
            checker=session.checker,
            env_cfg=session.env_config,
            task_config=session.task_config,
            num_episodes=count,
            inner_unroll_n=session.rl_config.inner_unroll_n,
            episodic_latent=session.rl_config.episodic_latent,
            greedy=greedy,
            policy_dist_fn=policy_dist_fn,
            evaluation_seed=seed,
            collect_per_instance=True,
            additional_models=additional_models,
            record_local_seeding=True,
        )

    rollout_before = session.trainer.compute_accounting_snapshot()["model_work"][
        "evaluation"
    ]
    rollout_started = time.perf_counter()
    mean_score, solve_rate, details = _record_evaluation_accounting(
        session.trainer, evaluate
    )
    rollout_wall = time.perf_counter() - rollout_started
    rollout_after = session.trainer.compute_accounting_snapshot()["model_work"][
        "evaluation"
    ]
    rollout_work = subtract_model_counters(rollout_after, rollout_before)
    per_instance = details.pop("per_instance")
    if not isinstance(per_instance, list) or len(per_instance) != count:
        raise PolicyImprovementSmokeError("Held-out evaluation inventory differs.")
    solved = int(details["solved_count"])
    terminal = {"budget": 0, "solved": 0, "stop": 0}
    discounted_returns: list[float] = []
    solved_steps: list[int] = []
    for record in per_instance:
        reason = record["termination_reason"]
        if reason not in terminal:
            raise PolicyImprovementSmokeError(
                "Held-out rollout has an unregistered terminal reason."
            )
        terminal[reason] += 1
        discounted_returns.append(
            sum(
                (float(session.rl_config.gamma) ** index) * float(reward)
                for index, reward in enumerate(record["rewards"])
            )
        )
        if record["success"]:
            solved_steps.append(int(record["environment_interactions"]))

    def predict_initial_values() -> tuple[float, float, dict[str, Any]]:
        predictions: list[float] = []
        training = bool(model.training)
        model.eval()
        try:
            with torch.no_grad():
                for index in range(count):
                    sample = session.evaluation_dataset[index]
                    if not isinstance(sample, Mapping):
                        raise PolicyImprovementSmokeError(
                            "Held-out value calibration requires mapping samples."
                        )
                    batched_x = prepare_batch_x(
                        sample,
                        device=session.device,
                        batched=False,
                    )
                    plan_value = sample.get("initial_plan")
                    if plan_value is None:
                        raise PolicyImprovementSmokeError(
                            "Held-out value calibration requires an initial plan."
                        )
                    plan = prepare_plan(
                        plan_value,
                        device=session.device,
                        batched=False,
                    )
                    value, _ = model.used_value(
                        batched_x,
                        plan,
                        n=session.rl_config.inner_unroll_n,
                        z=None,
                    )
                    scalar = float(value.reshape(-1)[0].item())
                    if not math.isfinite(scalar):
                        raise PolicyImprovementSmokeError(
                            "Held-out value prediction is nonfinite."
                        )
                    predictions.append(scalar)
        finally:
            model.train(training)
        return 0.0, 0.0, {"predictions": predictions}

    calibration_before = session.trainer.compute_accounting_snapshot()["model_work"][
        "evaluation"
    ]
    calibration_started = time.perf_counter()
    _, _, prediction_details = _record_evaluation_accounting(
        session.trainer,
        predict_initial_values,
    )
    calibration_wall = time.perf_counter() - calibration_started
    calibration_after = session.trainer.compute_accounting_snapshot()["model_work"][
        "evaluation"
    ]
    calibration_work = subtract_model_counters(
        calibration_after,
        calibration_before,
    )
    predictions = prediction_details["predictions"]
    if not isinstance(predictions, list) or len(predictions) != count:
        raise PolicyImprovementSmokeError(
            "Held-out value prediction inventory differs."
        )
    calibration = (
        sum(
            abs(float(prediction) - realized)
            for prediction, realized in zip(predictions, discounted_returns)
        )
        / count
    )
    aggregate = {
        "variant": variant,
        "mean_checker_score": float(mean_score),
        "solve_rate": float(solve_rate),
        "solved_count": solved,
        "denominator": count,
        "discounted_return_mean": sum(discounted_returns) / count,
        "terminal_reason_counts": terminal,
        "value_calibration": calibration,
        "details": details,
    }
    exact_proof = None
    if policy_dist_fn is not None:
        proof = getattr(policy_dist_fn, "_smoke_checks", None)
        if not isinstance(proof, dict) or proof.get("calls", 0) < 1:
            raise PolicyImprovementSmokeError(
                "Exact-mixture evaluation did not exercise probability mixing."
            )
        exact_proof = {
            "deployment": "exact_probability_mixture",
            "sampled": True,
            "policy_distribution_calls": int(proof["calls"]),
            "maximum_probability_error": float(proof["maximum_probability_error"]),
            "model_work": proof["proof_model_work"],
            "wall_time_seconds": float(proof["proof_wall_time_seconds"]),
        }
    aggregate["edits_to_solve_mean"] = (
        sum(solved_steps) / len(solved_steps) if solved_steps else None
    )
    evidence_records: list[dict[str, Any]] = []
    registered_records = dataset_sample_sha256s(
        session.evaluation_dataset,
        count=count,
    )
    for index, record in enumerate(per_instance):
        success = bool(record["success"])
        terminal_reason = str(record["termination_reason"])
        if (terminal_reason == "solved") != success:
            raise PolicyImprovementSmokeError(
                "Held-out solved flag differs from its terminal reason."
            )
        sample = session.evaluation_dataset[index]
        original_dataset_index = sample.get("original_dataset_index")
        if session.evaluation_population is not None and (
            isinstance(original_dataset_index, bool)
            or not isinstance(original_dataset_index, int)
            or original_dataset_index != session.evaluation_population["indices"][index]
        ):
            raise PolicyImprovementSmokeError(
                "Stage 0 evaluation record lost its registered train index."
            )
        input_digest = input_sha256(sample["inputs"])
        evidence_record = {
            "registered_index": index,
            "registered_record_sha256": registered_records[index],
            "puzzle_id": f"validation-{index:06d}",
            "puzzle_sha256": input_digest,
            "solved": success,
            "discounted_return": discounted_returns[index],
            "terminal_reason": terminal_reason,
            "edits_to_solve": (
                int(record["environment_interactions"]) if success else None
            ),
            "value_prediction": float(predictions[index]),
            "realized_return": discounted_returns[index],
        }
        if session.evaluation_population is not None:
            evidence_record.update(
                {
                    "population_position": index,
                    "original_dataset_index": original_dataset_index,
                    "registered_input_sha256": input_digest,
                    "puzzle_id": f"train-{original_dataset_index:06d}",
                }
            )
        evidence_records.append(evidence_record)
    evidence = {
        "schema_name": (
            "policy_improvement_instances_v2"
            if session.evaluation_population is not None
            else "policy_improvement_instances_v1"
        ),
        "schema_version": (1 if session.evaluation_population is not None else 2),
        "protocol_id": context.protocol["protocol_id"],
        "run_id": context.row["run_id"],
        "phase": context.row["phase"],
        "tier": context.row["tier"],
        "seed": context.row["seed"],
        "evaluation_split": context.row["evaluation_split"],
        "method_id": context.row["method_id"],
        "snapshot_kind": "interaction_matched",
        "evaluation_id": (f"{context.row['run_id']}.interaction_matched.{variant}"),
        "policy_variant": variant,
        "evaluation_pool_sha256": canonical_json_sha256(
            dataset_sample_sha256s(session.evaluation_dataset, count=count)
        ),
        "records": evidence_records,
    }
    if session.evaluation_population is not None:
        evidence.update(
            {
                "evaluation_population_id": session.evaluation_population[
                    "population_id"
                ],
                "evaluation_population_binding_sha256": (
                    session.evaluation_population["binding_sha256"]
                ),
            }
        )
    proof_work = (
        exact_proof["model_work"] if exact_proof is not None else zero_model_counters()
    )
    standard_policy_work = subtract_model_counters(rollout_work, proof_work)
    proof_wall = (
        float(exact_proof["wall_time_seconds"]) if exact_proof is not None else 0.0
    )
    standard_policy_wall = rollout_wall - proof_wall
    if standard_policy_wall < 0.0:
        raise PolicyImprovementSmokeError(
            "Exact-mixture proof wall time exceeds its evaluation interval."
        )
    return (
        aggregate,
        evidence,
        exact_proof,
        {
            "standard_policy_evaluation_model_work": standard_policy_work,
            "exact_mixture_proof_model_work": proof_work,
            "value_calibration_model_work": calibration_work,
            "standard_policy_evaluation_wall_time_seconds": (standard_policy_wall),
            "exact_mixture_proof_wall_time_seconds": proof_wall,
            "value_calibration_wall_time_seconds": calibration_wall,
        },
    )


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _unavailable(
    reason: str = "not_collected_by_registered_protocol",
) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _registered_result_document(
    context: SmokeContext,
    session: SmokeSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Wrap one Stage 0 runtime payload in its exact protocol-v2 envelope."""

    if context.protocol.get("schema_name") != "policy_improvement_protocol_v2":
        validate_result(payload)
        return payload
    population = session.evaluation_population
    if population is None:
        raise PolicyImprovementSmokeError(
            "Protocol v2 result lacks its registered evaluation population."
        )
    payload = dict(payload)
    payload["schema_name"] = "policy_improvement_stage0_result_payload_v2"
    payload["schema_version"] = 1
    payload["evaluation_population_id"] = population["population_id"]
    payload["evaluation_population_binding_sha256"] = population["binding_sha256"]
    population_registration = context.protocol["population_registry"]
    document = {
        "schema_name": "policy_improvement_result_v2",
        "schema_version": 1,
        "protocol_id": context.protocol["protocol_id"],
        "protocol_schema_name": context.protocol["schema_name"],
        "protocol_schema_version": context.protocol["schema_version"],
        "protocol_sha256": context.protocol_sha256,
        "population_registry_schema_name": population_registration["schema_name"],
        "population_registry_schema_version": population_registration["schema_version"],
        "population_registry_sha256": population_registration["sha256"],
        "registry_schema_name": context.registry["schema_name"],
        "registry_schema_version": context.registry["registry_schema_version"],
        "registry_sha256": context.registry_sha256,
        "registry_row_schema_name": context.row["schema_name"],
        "registry_row_schema_version": context.row["schema_version"],
        "registry_row_sha256": context.registry_row_sha256,
        "run_id": context.row["run_id"],
        "phase": context.row["phase"],
        "method_id": context.row["method_id"],
        "status": payload["status"],
        "evaluation_split": context.row["evaluation_split"],
        "evaluation_population_id": population["population_id"],
        "evaluation_population_binding_sha256": population["binding_sha256"],
        "evaluation_population_ordered_record_sha256": population[
            "ordered_record_sha256"
        ],
        "evaluation_population_ordered_input_sha256": population[
            "ordered_input_sha256"
        ],
        "evaluation_record_count": population["count"],
        "validation_data_opened": False,
        "test_data_opened": False,
        "scientific_selection": context.row["scientific_selection"],
        "paper_evidence_eligible": context.row["paper_evidence_eligible"],
        "payload": payload,
    }
    validate_result(document)
    from scripts.policy_improvement_populations import load_registered_populations
    from scripts.policy_improvement_v2_schema import bind_v2_result_to_registration

    bind_v2_result_to_registration(
        document,
        context.row,
        context.protocol,
        context.registry,
        load_registered_populations(context.protocol, context.source_root),
    )
    return document


def _model_state_identity(session: SmokeSession) -> tuple[str, dict[str, str]]:
    trainer = session.trainer
    if type(trainer).__name__ == "PPOTrainer":
        roles = {"model": session.model}
    else:
        roles = {
            "model": session.model,
            "policy_model_old": trainer.policy_model_old,
            "policy_model_candidate": trainer.policy_model_candidate,
            "target_model": trainer.target_model,
        }
        if trainer.preinterpolation_policy_base is not None:
            roles["preinterpolation_policy_base"] = trainer.preinterpolation_policy_base
        if trainer.preinterpolation_policy_candidate is not None:
            roles["preinterpolation_policy_candidate"] = (
                trainer.preinterpolation_policy_candidate
            )
    hashes = {
        name: state_dict_sha256(model.state_dict())
        for name, model in sorted(roles.items())
    }
    return canonical_json_sha256(hashes), hashes


def stage0_model_state_identity(
    session: SmokeSession,
) -> tuple[str, dict[str, str]]:
    """Return the canonical behavior-bearing model roles for audit validation."""

    return _model_state_identity(session)


def stage0_theory_model_identity(
    session: SmokeSession,
    *,
    method_id: str,
    alpha: float,
) -> dict[str, str]:
    """Return the exact model identity consumed by the Stage 0 theory bridge."""

    if method_id not in {
        "fixed_base_exact_persistent",
        "fixed_base_exact_episodic",
    }:
        raise PolicyImprovementSmokeError(
            "Stage 0 theory identity is defined only for exact methods."
        )
    trainer = session.trainer
    current = getattr(trainer, "policy_model_old", None)
    candidate = getattr(trainer, "policy_model_candidate", None)
    if current is None or candidate is None:
        raise PolicyImprovementSmokeError(
            "Exact Stage 0 trainer lacks its fixed current/candidate policies."
        )
    model_sha256, role_sha256s = _model_state_identity(session)
    current_sha256 = state_dict_sha256(current.state_dict())
    candidate_sha256 = state_dict_sha256(candidate.state_dict())
    if (
        role_sha256s.get("policy_model_old") != current_sha256
        or role_sha256s.get("policy_model_candidate") != candidate_sha256
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 theory policy identities differ from the model inventory."
        )
    recurrent_state = {
        name: value
        for name, value in current.state_dict().items()
        if not name.startswith("edit_policy.")
        and not name.startswith("value_head.")
    }
    recurrent_sha256 = state_dict_sha256(recurrent_state)
    model_config = session.effective_config.get("model_config")
    if not isinstance(model_config, Mapping):
        raise PolicyImprovementSmokeError(
            "Stage 0 effective configuration lacks its model configuration."
        )
    deployed_sha256 = canonical_json_sha256(
        {
            "kind": "exact_probability_mixture",
            "current_policy_sha256": current_sha256,
            "candidate_policy_sha256": candidate_sha256,
            "alpha": float(alpha),
            "recurrent_transition_sha256": recurrent_sha256,
        }
    )
    return {
        "model_sha256": model_sha256,
        "model_config_sha256": canonical_json_sha256(dict(model_config)),
        "current_policy_sha256": current_sha256,
        "candidate_policy_sha256": candidate_sha256,
        "deployed_policy_sha256": deployed_sha256,
        "recurrent_transition_sha256": recurrent_sha256,
    }


def _build_final_result(
    context: SmokeContext,
    session: SmokeSession,
    staging: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_validation: Mapping[str, object],
    checkpoint_validation_sha256: str,
    parent_checkpoint_sha256: str,
    latest_metrics: Mapping[str, float],
    gpu_utilization: GpuUtilizationSummary,
    checkpoint_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training_compute = session.trainer.compute_accounting_snapshot()
    evaluations_dir = staging / "evaluations"
    evaluations_dir.mkdir(mode=0o700)
    policy_rows: list[dict[str, Any]] = []
    exact_proofs: list[dict[str, Any]] = []
    standard_policy_evaluation_work = zero_model_counters()
    exact_mixture_proof_work = zero_model_counters()
    value_calibration_work = zero_model_counters()
    standard_policy_evaluation_wall = 0.0
    exact_mixture_proof_wall = 0.0
    value_calibration_wall = 0.0
    evaluation_pool_sha256 = canonical_json_sha256(
        dataset_sample_sha256s(
            session.evaluation_dataset,
            count=int(context.protocol["budgets"]["smoke"]["evaluation_records"]),
        )
    )
    for variant, model, callback, greedy, additional in _variant_specs(
        context, session
    ):
        aggregate, per_instance, proof, evaluation_accounting = _evaluation_payload(
            context,
            session,
            variant,
            model,
            callback,
            greedy,
            additional,
        )
        aggregate_path = evaluations_dir / f"{variant}.json"
        per_instance_path = evaluations_dir / f"{variant}.per_instance.json"
        primary = {
            "solve_rate": _available(aggregate["solve_rate"]),
            "solved_count": _available(aggregate["solved_count"]),
            "denominator": _available(aggregate["denominator"]),
        }
        edits = aggregate["edits_to_solve_mean"]
        secondary = {
            "discounted_return_mean": _available(aggregate["discounted_return_mean"]),
            "edits_to_solve_mean": (
                _available(edits)
                if edits is not None
                else _unavailable("not_applicable")
            ),
            "terminal_reason_counts": _available(aggregate["terminal_reason_counts"]),
            "value_calibration": _available(aggregate["value_calibration"]),
        }
        aggregate_evidence = {"primary": primary, "secondary": secondary}
        _write_json(
            aggregate_path,
            {**aggregate_evidence, "diagnostic_details": aggregate},
        )
        _write_json(per_instance_path, per_instance)
        aggregate_sha = canonical_json_sha256(aggregate_evidence)
        per_instance_sha = canonical_json_sha256(per_instance)
        if proof is not None:
            exact_proofs.append(proof)
        standard_policy_evaluation_work = add_model_counters(
            standard_policy_evaluation_work,
            evaluation_accounting["standard_policy_evaluation_model_work"],
        )
        exact_mixture_proof_work = add_model_counters(
            exact_mixture_proof_work,
            evaluation_accounting["exact_mixture_proof_model_work"],
        )
        value_calibration_work = add_model_counters(
            value_calibration_work,
            evaluation_accounting["value_calibration_model_work"],
        )
        standard_policy_evaluation_wall += float(
            evaluation_accounting["standard_policy_evaluation_wall_time_seconds"]
        )
        exact_mixture_proof_wall += float(
            evaluation_accounting["exact_mixture_proof_wall_time_seconds"]
        )
        value_calibration_wall += float(
            evaluation_accounting["value_calibration_wall_time_seconds"]
        )
        policy_rows.append(
            {
                "evaluation_id": (
                    f"{context.row['run_id']}.interaction_matched.{variant}"
                ),
                "policy_variant": variant,
                "evaluation_pool_sha256": _available(evaluation_pool_sha256),
                "evaluation_artifact_sha256": _available(aggregate_sha),
                "per_instance_artifact_sha256": _available(per_instance_sha),
                "primary": primary,
                "secondary": secondary,
            }
        )
    if (
        "exact_mixture" in policy_variants_for_method(str(context.row["method_id"]))
        and len(exact_proofs) != 1
    ):
        raise PolicyImprovementSmokeError("Exact-mixture proof inventory differs.")

    compute = session.trainer.compute_accounting_snapshot()
    compute_path = staging / "compute_snapshot.json"
    compute_sha256 = _write_json(compute_path, compute)
    compute_accounting_sha256: str | None = None
    if context.protocol.get("schema_name") == "policy_improvement_protocol_v2":
        compute_accounting = build_training_compute_accounting(
            compute,
            device=session.device,
            checkpoint_seconds=checkpoint_seconds,
            cuda_utilization=_gpu_utilization_document(gpu_utilization),
        )
        compute_accounting_sha256 = _write_json(
            staging / "compute_accounting.json",
            compute_accounting,
        )
    training_work = training_compute["model_work"]["training"]
    evaluation_work = compute["model_work"]["evaluation"]
    if (
        add_model_counters(
            standard_policy_evaluation_work,
            exact_mixture_proof_work,
            value_calibration_work,
        )
        != evaluation_work
    ):
        raise PolicyImprovementSmokeError(
            "Stage 0 evaluation and proof accounting does not reconcile."
        )
    recurrent_work = int(training_work["recurrent_latent_state_updates"])
    optimizer_steps = int(training_compute["progress"]["optimizer_steps_total"])
    training_wall = float(training_compute["wall_time_seconds"]["training"])
    evaluation_wall = float(compute["wall_time_seconds"]["evaluation"])
    memory = training_compute["peak_memory_bytes"]
    peak_allocated = int(memory["cuda_allocated"] or 0)
    peak_reserved = int(memory["cuda_reserved"] or 0)
    peak = max(
        int(memory["process_rss"]),
        peak_allocated,
        peak_reserved,
    )
    seq_len = int(session.model.config.seq_len)
    cells_processed = recurrent_work * seq_len
    actions_processed = int(training_work["action_logits_evaluated"]) + int(
        training_work["action_values_evaluated"]
    )
    tokens_processed = cells_processed
    checked_gpu = _validate_gpu_utilization_summary(
        _gpu_utilization_document(gpu_utilization),
        expected_device_type=session.device.type,
    )
    if session.device.type == "cuda":
        gpu_fraction = _available(sum(checked_gpu.samples) / len(checked_gpu.samples))
        gpu_sample_count = _available(len(checked_gpu.samples))
        gpu_sampling_interval = _available(checked_gpu.sampling_interval_seconds)
    else:
        gpu_fraction = _unavailable("not_applicable")
        gpu_sample_count = _unavailable("not_applicable")
        gpu_sampling_interval = _unavailable("not_applicable")
    accelerator_seconds = training_wall if session.device.type == "cuda" else 0.0
    root_manifest_sha = file_sha256(context.dataset_root / "MANIFEST.json")
    if root_manifest_sha != context.dataset_manifest_sha256:
        raise PolicyImprovementSmokeError(
            "Dataset top-level manifest changed before result construction."
        )
    training_records = dataset_sample_sha256s(session.train_dataset)
    evaluation_count = int(context.protocol["budgets"]["smoke"]["evaluation_records"])
    evaluation_records = dataset_sample_sha256s(
        session.evaluation_dataset, count=evaluation_count
    )
    train_ordered_records_sha256 = ordered_record_sha256(training_records)
    evaluation_ordered_records_sha256 = ordered_record_sha256(evaluation_records)
    evaluation_population_ordered_records_sha256 = (
        evaluation_ordered_records_sha256
        if session.evaluation_population is not None
        else ordered_record_sha256(dataset_sample_sha256s(session.evaluation_dataset))
    )
    if session.training_population is not None:
        registered_train_order = _registered_hex(
            session.training_population["ordered_record_sha256"],
            name="registered training population ordered records",
            length=64,
        )
    else:
        registered_train_order = _available_hex(
            context.protocol["dataset"]["splits"]["train"]["ordered_record_sha256"],
            name="registered train ordered records",
            length=64,
        )
    if session.evaluation_population is not None:
        registered_evaluation_order = _registered_hex(
            session.evaluation_population["ordered_record_sha256"],
            name="registered evaluation population ordered records",
            length=64,
        )
    else:
        registered_evaluation_order = _available_hex(
            context.protocol["dataset"]["splits"][str(context.row["evaluation_split"])][
                "ordered_record_sha256"
            ],
            name="registered evaluation ordered records",
            length=64,
        )
    if (
        train_ordered_records_sha256 != registered_train_order
        or evaluation_population_ordered_records_sha256 != registered_evaluation_order
    ):
        raise PolicyImprovementSmokeError(
            "Loaded dataset record order differs from the registered protocol."
        )
    model_sha256, model_state_sha256s = _model_state_identity(session)
    model_state_inventory_sha256 = _write_json(
        staging / "model_state_inventory.json",
        {
            "schema_name": "policy_improvement_model_state_inventory_v1",
            "run_id": context.row["run_id"],
            "method_id": context.row["method_id"],
            "model_state_sha256": model_sha256,
            "role_state_sha256s": model_state_sha256s,
            "snapshot_state_bindings": [
                {
                    "snapshot_kind": "interaction_matched",
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_state_sha256": model_sha256,
                    "checkpoint_validation_sha256": (checkpoint_validation_sha256),
                }
            ],
        },
    )
    if (
        checkpoint_validation["model_state_sha256"] != model_sha256
        or checkpoint_validation["role_state_sha256s"] != model_state_sha256s
    ):
        raise PolicyImprovementSmokeError(
            "Post-evaluation model state differs from the strictly validated checkpoint."
        )
    amendment_sha256 = amendment_history_sha256([])
    checkpoint_lineage_sha256 = canonical_json_sha256(
        {
            "schema_name": "policy_improvement_smoke_checkpoint_lineage_v1",
            "run_id": context.row["run_id"],
            "initialization_sha256": session.initialization_sha256,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    run_manifest = {
        "schema_name": "policy_improvement_smoke_run_manifest_v1",
        "run_id": context.row["run_id"],
        "method_id": context.row["method_id"],
        "environment_interactions": 32,
        "protocol_sha256": context.protocol_sha256,
        "registry_row_sha256": context.registry_row_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_sha256,
        "model_state_sha256s": model_state_sha256s,
        "model_state_inventory_sha256": model_state_inventory_sha256,
        "compute_snapshot_sha256": compute_sha256,
        "exact_probability_mixture_proofs": exact_proofs,
        "identities": {
            "producer_git_commit": context.producer_commit,
            "training_source_git_commit": context.producer_commit,
            "training_runtime_sha256": context.runtime_sha256,
            "training_runtime_profile_sha256": (context.runtime_profile_sha256),
            "training_selected_source_manifest_sha256": (
                context.selected_source_manifest_sha256
            ),
            "runtime_authorization_sha256": (context.runtime_authorization_sha256),
            "launcher_sha256": context.launcher_sha256,
            "producer_manifest_sha256": context.producer_manifest_sha256,
            "method_config_sha256": session.method_config_sha256,
            "effective_config_sha256": session.effective_config_sha256,
            "dataset_manifest_sha256": root_manifest_sha,
            "dataset_producer_source": context.dataset_producer_source,
            "train_ordered_records_sha256": train_ordered_records_sha256,
            "evaluation_ordered_records_sha256": (evaluation_ordered_records_sha256),
            "evaluation_population_ordered_records_sha256": (
                evaluation_population_ordered_records_sha256
            ),
            "initialization_sha256": session.initialization_sha256,
            "evaluation_pool_sha256": evaluation_pool_sha256,
        },
        "measurement": {
            "training_wall_time_seconds": training_wall,
            "evaluation_wall_time_seconds": evaluation_wall,
            "peak_memory_bytes": peak,
            "training_model_work": training_work,
            "evaluation_model_work": evaluation_work,
            "standard_policy_evaluation_model_work": (standard_policy_evaluation_work),
            "exact_mixture_proof_model_work": exact_mixture_proof_work,
            "value_calibration_model_work": value_calibration_work,
            "standard_policy_evaluation_wall_time_seconds": (
                standard_policy_evaluation_wall
            ),
            "exact_mixture_proof_wall_time_seconds": (exact_mixture_proof_wall),
            "value_calibration_wall_time_seconds": value_calibration_wall,
            "training_recurrent_map_applications": recurrent_work,
            "gpu_utilization": _gpu_utilization_document(checked_gpu),
        },
    }
    if compute_accounting_sha256 is not None:
        run_manifest["compute_accounting_sha256"] = compute_accounting_sha256
    run_manifest_path = staging / "RUN_MANIFEST.json"
    run_manifest_sha256 = _write_json(run_manifest_path, run_manifest)
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": context.protocol["protocol_id"],
        "protocol_sha256": context.protocol_sha256,
        "amendment_history_sha256": amendment_sha256,
        "run_id": context.row["run_id"],
        "registry_row_sha256": context.registry_row_sha256,
        "phase": context.row["phase"],
        "tier": context.row["tier"],
        "seed": context.row["seed"],
        "evaluation_split": context.row["evaluation_split"],
        "method_id": context.row["method_id"],
        "base_method_id": context.row["base_method_id"],
        "n": context.row["n"],
        "K": context.row["K"],
        "alpha": context.row["alpha"],
        "ablation_variant": context.row["ablation_variant"],
        "applied_config_override": dict(context.row["config_override"]),
        "primary_policy_variant": primary_policy_variant_for_method(
            str(context.row["method_id"])
        ),
        "evaluation_snapshots": [
            {
                "snapshot_kind": "interaction_matched",
                "status": "available",
                "unavailable_reason": None,
                "target": {
                    "unit": "environment_interactions",
                    "registered_quantity": _available(32),
                },
                "observed_environment_interactions": _available(32),
                "observed_recurrent_map_applications": _available(recurrent_work),
                "accelerator_seconds_observed": _available(accelerator_seconds),
                "checkpoint_sha256": _available(checkpoint_sha256),
                "model_state_sha256": _available(model_sha256),
                "checkpoint_lineage_sha256": _available(checkpoint_lineage_sha256),
                "policy_evaluations": policy_rows,
            },
            {
                "snapshot_kind": "compute_matched",
                "status": "unavailable",
                "unavailable_reason": "compute_snapshot_not_registered",
                "target": {
                    "unit": "recurrent_map_applications",
                    "registered_quantity": _unavailable(
                        "compute_budget_pending_smoke_measurement"
                    ),
                },
                "observed_environment_interactions": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "observed_recurrent_map_applications": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "accelerator_seconds_observed": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "checkpoint_sha256": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "model_state_sha256": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "checkpoint_lineage_sha256": _unavailable(
                    "compute_budget_pending_smoke_measurement"
                ),
                "policy_evaluations": [],
            },
        ],
        "status": "complete",
        "failure": None,
        "identities": {
            "producer_git_commit": context.producer_commit,
            "git_clean": True,
            "runtime_authorization_sha256": (context.runtime_authorization_sha256),
            "training_source_git_commit": context.producer_commit,
            "training_runtime_sha256": context.runtime_sha256,
            "training_runtime_profile_sha256": (context.runtime_profile_sha256),
            "training_selected_source_manifest_sha256": (
                context.selected_source_manifest_sha256
            ),
            "launcher_sha256": context.launcher_sha256,
            "producer_manifest_sha256": context.producer_manifest_sha256,
            "method_config_sha256": session.method_config_sha256,
            "effective_config_sha256": context.row["expected_effective_config_sha256"],
            "dataset_manifest_sha256": root_manifest_sha,
            "train_ordered_records_sha256": train_ordered_records_sha256,
            "evaluation_ordered_records_sha256": (evaluation_ordered_records_sha256),
            "initialization_sha256": session.initialization_sha256,
            "checkpoint_sha256": _available(checkpoint_sha256),
            "model_state_sha256": _available(model_sha256),
            "evaluation_runtime_sha256": _available(context.runtime_sha256),
            "evaluation_source_git_commit": _available(context.producer_commit),
            "evaluation_runtime_profile_sha256": _available(
                context.runtime_profile_sha256
            ),
            "evaluation_selected_source_manifest_sha256": _available(
                context.selected_source_manifest_sha256
            ),
            "evaluation_pool_sha256": _available(evaluation_pool_sha256),
            "test_open_sha256": _unavailable("test_data_not_opened"),
            "device": str(session.device),
        },
        "metrics": {
            "training": {
                "interactions_to_first_solve": _unavailable(),
                "value_loss": (
                    _available(latest_metrics["loss_value"])
                    if "loss_value" in latest_metrics
                    else _unavailable()
                ),
                "policy_loss": (
                    _available(latest_metrics["loss_policy"])
                    if "loss_policy" in latest_metrics
                    else _unavailable()
                ),
                "wall_time_seconds": _available(training_wall),
                "gpu_hours": _available(
                    training_wall / 3600.0 if session.device.type == "cuda" else 0.0
                ),
                "gpu_utilization_fraction": gpu_fraction,
                "gpu_utilization_sample_count": gpu_sample_count,
                "gpu_utilization_sampling_interval_seconds": (gpu_sampling_interval),
                "peak_allocated_memory_bytes": _available(peak_allocated),
                "peak_reserved_memory_bytes": _available(peak_reserved),
                "optimizer_steps": _available(optimizer_steps),
                "cells_processed": _available(cells_processed),
                "actions_processed": _available(actions_processed),
                "tokens_processed": _available(tokens_processed),
                "policy_head_calls": _available(int(training_work["policy_api_calls"])),
                "recurrent_map_applications": _available(recurrent_work),
                "value_head_calls": _available(int(training_work["value_api_calls"])),
            },
            "diagnostics": {
                name: _unavailable()
                for name in (
                    "L_preproj",
                    "projection_active_rate",
                    "absolute_depth_policy_agreement",
                    "finite_depth_discrepancy",
                    "fixed_target_residual",
                    "propagated_target_lag",
                    "exact_centering_defect",
                    "candidate_current_tv",
                    "mixture_realized_tv",
                    "mixture_realized_kl",
                    "fresh_initialization_discrepancy",
                    "value_of_memory",
                )
            },
        },
        "artifacts": {
            "checkpoint": _available(checkpoint_sha256),
            "checkpoint_validation": _available(checkpoint_validation_sha256),
            "model_state_inventory": _available(model_state_inventory_sha256),
            "run_manifest": _available(run_manifest_sha256),
        },
    }
    result = _registered_result_document(context, session, result)
    _write_json(staging / "result.json", result)
    return result, run_manifest


def _rehash_runtime(runtime_preflight: Any, expected: str) -> None:
    path = Path(f"/proc/self/fd/{runtime_preflight.runtime_descriptor}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise PolicyImprovementSmokeError("Training runtime changed during Stage 0.")


def _revalidate_external_inputs(
    context: SmokeContext,
    *,
    runtime_preflight: Any,
) -> None:
    if discover_clean_git_source(context.source_root) != {
        "git_commit": context.producer_commit,
        "git_clean": True,
    }:
        raise PolicyImprovementSmokeError("Producer source changed during Stage 0.")
    authorized = authorize_phase4_training_source(
        context.source_root, context.producer_commit
    )
    if authorized.source_manifest_sha256 != context.producer_manifest_sha256:
        raise PolicyImprovementSmokeError(
            "Producer source manifest changed during Stage 0."
        )
    protocol = validate_protocol(
        load_strict_json(
            context.source_root
            / (
                "configs/policy_improvement_v2/protocol.json"
                if context.protocol.get("schema_name")
                == "policy_improvement_protocol_v2"
                else "configs/policy_improvement_v1/protocol.json"
            )
        )
    )
    if (
        hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        != context.protocol_sha256
    ):
        raise PolicyImprovementSmokeError("Protocol changed during Stage 0.")
    registered_dataset_manifest = _available_hex(
        protocol["dataset"]["manifest_sha256"],
        name="registered dataset manifest",
        length=64,
    )
    registered_dataset_producer = {
        field: _available_hex(
            protocol["dataset"]["producer_source"][field],
            name=f"registered dataset producer {field}",
            length=40 if field == "git_commit" else 64,
        )
        for field in (
            "git_commit",
            "launcher_sha256",
            "runtime_sha256",
            "source_manifest_sha256",
        )
    }
    if (
        registered_dataset_manifest != context.dataset_manifest_sha256
        or registered_dataset_producer != context.dataset_producer_source
    ):
        raise PolicyImprovementSmokeError(
            "Registered dataset producer identity changed during Stage 0."
        )
    verified = verify_dataset(
        context.dataset_root,
        owner_root=context.dataset_root.parent,
        expected_producer=context.dataset_producer_source,
        verify_content_splits=(
            {"train"}
            if context.protocol.get("schema_name") == "policy_improvement_protocol_v2"
            else {"train", "validation"}
        ),
    )
    if not isinstance(verified, Mapping):
        raise PolicyImprovementSmokeError("Dataset revalidation failed.")
    splits = verified.get("split_manifest_sha256")
    if not isinstance(splits, Mapping):
        raise PolicyImprovementSmokeError("Dataset split identities are missing.")
    ordered_records = verified.get("split_ordered_record_sha256")
    if not isinstance(ordered_records, Mapping):
        raise PolicyImprovementSmokeError(
            "Dataset ordered-record identities are missing."
        )
    if verified.get("manifest_sha256") != context.dataset_manifest_sha256:
        raise PolicyImprovementSmokeError(
            "Dataset top-level manifest changed during Stage 0."
        )
    for split in ("train", "validation", "test"):
        registration = protocol["dataset"]["splits"][split]
        registered = registration["manifest_sha256"]
        registered_order = registration["ordered_record_sha256"]
        if (
            not isinstance(registered, Mapping)
            or registered.get("status") != "available"
            or splits.get(split) != registered.get("value")
            or not isinstance(registered_order, Mapping)
            or registered_order.get("status") != "available"
            or ordered_records.get(split) != registered_order.get("value")
        ):
            raise PolicyImprovementSmokeError(
                "Dataset identity changed during Stage 0."
            )
    _rehash_runtime(runtime_preflight, context.runtime_sha256)


def _inventory(staging: Path) -> dict[str, dict[str, object]]:
    result, _ = _segment_inventory(staging)
    return result


def _prior_failed_attempt_inventory(
    context: SmokeContext,
) -> list[dict[str, object]]:
    """Snapshot every failed attempt that exists before segment publication."""

    attempts_root = context.run_root / "attempts"
    try:
        root_status = attempts_root.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise PolicyImprovementSmokeError(
            "Prior failed-attempt inventory is unavailable."
        ) from exc
    if (
        attempts_root.resolve(strict=True) != attempts_root
        or stat.S_ISLNK(root_status.st_mode)
        or not stat.S_ISDIR(root_status.st_mode)
        or root_status.st_uid != os.geteuid()
    ):
        raise PolicyImprovementSmokeError(
            "Prior failed-attempt inventory is aliased or unowned."
        )
    inventory: list[dict[str, object]] = []
    segment_entries = sorted(attempts_root.iterdir(), key=lambda item: item.name)
    if not segment_entries:
        raise PolicyImprovementSmokeError("Prior failed-attempt root is empty.")
    for segment in segment_entries:
        segment_status = segment.lstat()
        attempt_entries = sorted(segment.iterdir(), key=lambda item: item.name)
        if (
            segment.name not in {"prepare", "resume"}
            or segment.resolve(strict=True) != segment
            or stat.S_ISLNK(segment_status.st_mode)
            or not stat.S_ISDIR(segment_status.st_mode)
            or segment_status.st_uid != os.geteuid()
            or not attempt_entries
        ):
            raise PolicyImprovementSmokeError(
                "Prior failed-attempt segment is invalid or empty."
            )
        for attempt in attempt_entries:
            attempt_status = attempt.lstat()
            if (
                len(attempt.name) != 32
                or any(character not in _LOWER_SHA256 for character in attempt.name)
                or attempt.resolve(strict=True) != attempt
                or stat.S_ISLNK(attempt_status.st_mode)
                or not stat.S_ISDIR(attempt_status.st_mode)
                or attempt_status.st_uid != os.geteuid()
            ):
                raise PolicyImprovementSmokeError(
                    "Prior failed-attempt generation is invalid."
                )
            files, directories = _segment_inventory(attempt)
            if set(files) != {"MANIFEST.json", "result.json"} or directories:
                raise PolicyImprovementSmokeError(
                    "Prior failed-attempt generation inventory differs."
                )
            inventory.append(
                {
                    "segment": segment.name,
                    "attempt_id": attempt.name,
                    "generation_manifest_sha256": files["MANIFEST.json"]["sha256"],
                    "result_sha256": files["result.json"]["sha256"],
                }
            )
    return inventory


def _publish_segment(
    context: SmokeContext,
    session: SmokeSession,
    module: Any,
    *,
    runtime_preflight: Any,
    parent_sha256: str | None,
    latest_metrics: Mapping[str, float],
    gpu_utilization: GpuUtilizationSummary,
) -> Path:
    segments = _ensure_owned_directory(
        context.evidence_root, context.run_root / "segments"
    )
    staging = Path(tempfile.mkdtemp(prefix=".segment-stage.", dir=str(segments)))
    staging_identity = (staging.stat().st_dev, staging.stat().st_ino)
    final = segments / f"env_{context.segment_budget:09d}"
    try:
        _write_json(
            staging / "training_gpu_utilization.json",
            _gpu_utilization_document(gpu_utilization),
        )
        try:
            checkpoint_started = time.perf_counter()
            checkpoint_path, checkpoint_sha, checkpoint_validation = _checkpoint_output(
                context, session, module, staging, parent_sha256
            )
            checkpoint_validation_sha256 = _write_json(
                staging / "checkpoint_validation.json",
                checkpoint_validation,
            )
            segment_checkpoint_seconds = time.perf_counter() - checkpoint_started
        except Exception as exc:
            raise _SmokeExecutionError("checkpoint", exc) from exc
        checkpoint_seconds = segment_checkpoint_seconds
        if context.protocol.get("schema_name") == "policy_improvement_protocol_v2":
            _write_json(
                staging / "checkpoint_timing.json",
                _checkpoint_timing_document(segment_checkpoint_seconds),
            )
            if context.segment_name == "resume":
                checkpoint_seconds += _parent_checkpoint_seconds(context)
        result = None
        if context.segment_budget == 32:
            if parent_sha256 is None:
                raise PolicyImprovementSmokeError(
                    "Final Stage 0 result requires its parent checkpoint."
                )
            try:
                result, _ = _build_final_result(
                    context,
                    session,
                    staging,
                    checkpoint_path,
                    checkpoint_sha,
                    checkpoint_validation,
                    checkpoint_validation_sha256,
                    parent_sha256,
                    latest_metrics,
                    gpu_utilization,
                    checkpoint_seconds,
                )
            except Exception as exc:
                raise _SmokeExecutionError("evaluation", exc) from exc
        outputs = _inventory(staging)
        checkpoint_relative = checkpoint_path.relative_to(staging).as_posix()
        prior_failed_attempts = _prior_failed_attempt_inventory(context)
        manifest = {
            "schema_name": "policy_improvement_smoke_segment_v1",
            "schema_version": SMOKE_SEGMENT_SCHEMA_VERSION,
            "protocol_sha256": context.protocol_sha256,
            "registry_sha256": context.registry_sha256,
            "registry_row_sha256": context.registry_row_sha256,
            "run_id": context.row["run_id"],
            "method_id": context.row["method_id"],
            "segment": context.segment_name,
            "environment_interactions": context.segment_budget,
            "parent_checkpoint_sha256": parent_sha256,
            "result_status": None if result is None else result["status"],
            "prior_failed_attempts": prior_failed_attempts,
            "outputs": {
                "checkpoint": {
                    "path": checkpoint_relative,
                    "bytes": outputs[checkpoint_relative]["bytes"],
                    "sha256": checkpoint_sha,
                },
                "files": outputs,
            },
            "storage_bytes": sum(int(item["bytes"]) for item in outputs.values()),
        }
        _write_json(staging / "MANIFEST.json", manifest)
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_dir(path)
        _fsync_dir(staging)
        _validate_segment_generation(
            context,
            staging,
            expected_budget=context.segment_budget,
        )
        if _prior_failed_attempt_inventory(context) != prior_failed_attempts:
            raise PolicyImprovementSmokeError(
                "Prior failed-attempt inventory changed before publication."
            )
        _revalidate_external_inputs(context, runtime_preflight=runtime_preflight)
        try:
            final.lstat()
        except FileNotFoundError:
            pass
        else:
            raise PolicyImprovementSmokeError("Stage 0 segment already exists.")
        _rename_noreplace(staging, final)
        _fsync_dir(segments)
        final_identity = final.lstat()
        if (
            not stat.S_ISDIR(final_identity.st_mode)
            or stat.S_ISLNK(final_identity.st_mode)
            or (final_identity.st_dev, final_identity.st_ino) != staging_identity
        ):
            raise PolicyImprovementSmokeError(
                "Published Stage 0 segment has the wrong identity."
            )
        _validate_segment_generation(
            context,
            final,
            expected_budget=context.segment_budget,
        )
        return final
    except BaseException:
        try:
            if (
                staging.exists()
                and (
                    staging.stat().st_dev,
                    staging.stat().st_ino,
                )
                == staging_identity
            ):
                shutil.rmtree(staging)
        except OSError:
            pass
        raise


def _failed_result(
    context: SmokeContext,
    session: SmokeSession,
    *,
    phase: str,
    error: BaseException,
) -> dict[str, Any]:
    reason = (
        "run_failed_before_evaluation"
        if phase in {"evaluation", "publication"}
        else "run_failed_before_checkpoint"
    )
    unavailable = _unavailable(reason)
    measurement_unavailable = _unavailable("run_failed_before_measurement")
    evaluation_count = int(context.protocol["budgets"]["smoke"]["evaluation_records"])
    evaluation_records = dataset_sample_sha256s(
        session.evaluation_dataset,
        count=evaluation_count,
    )
    evaluation_pool_sha256 = canonical_json_sha256(evaluation_records)
    if session.evaluation_started:
        evaluation_runtime_sha256 = _available(context.runtime_sha256)
        evaluation_source_git_commit = _available(context.producer_commit)
        evaluation_runtime_profile_sha256 = _available(context.runtime_profile_sha256)
        evaluation_selected_source_manifest_sha256 = _available(
            context.selected_source_manifest_sha256
        )
        available_evaluation_pool_sha256 = _available(evaluation_pool_sha256)
    else:
        evaluation_runtime_sha256 = _unavailable("run_failed_before_evaluation")
        evaluation_source_git_commit = _unavailable("run_failed_before_evaluation")
        evaluation_runtime_profile_sha256 = _unavailable("run_failed_before_evaluation")
        evaluation_selected_source_manifest_sha256 = _unavailable(
            "run_failed_before_evaluation"
        )
        available_evaluation_pool_sha256 = _unavailable("run_failed_before_evaluation")
    identities = {
        "producer_git_commit": context.producer_commit,
        "git_clean": True,
        "runtime_authorization_sha256": (context.runtime_authorization_sha256),
        "training_source_git_commit": context.producer_commit,
        "training_runtime_sha256": context.runtime_sha256,
        "training_runtime_profile_sha256": context.runtime_profile_sha256,
        "training_selected_source_manifest_sha256": (
            context.selected_source_manifest_sha256
        ),
        "launcher_sha256": context.launcher_sha256,
        "producer_manifest_sha256": context.producer_manifest_sha256,
        "method_config_sha256": session.method_config_sha256,
        "effective_config_sha256": context.row["expected_effective_config_sha256"],
        "dataset_manifest_sha256": context.dataset_manifest_sha256,
        "train_ordered_records_sha256": ordered_record_sha256(
            dataset_sample_sha256s(session.train_dataset)
        ),
        "evaluation_ordered_records_sha256": ordered_record_sha256(evaluation_records),
        "initialization_sha256": session.initialization_sha256,
        "checkpoint_sha256": _unavailable(reason),
        "model_state_sha256": _unavailable(reason),
        "evaluation_runtime_sha256": evaluation_runtime_sha256,
        "evaluation_source_git_commit": evaluation_source_git_commit,
        "evaluation_runtime_profile_sha256": evaluation_runtime_profile_sha256,
        "evaluation_selected_source_manifest_sha256": (
            evaluation_selected_source_manifest_sha256
        ),
        "evaluation_pool_sha256": available_evaluation_pool_sha256,
        "test_open_sha256": _unavailable("test_data_not_opened"),
        "device": str(session.device),
    }
    snapshots = []
    for kind, unit in (
        ("interaction_matched", "environment_interactions"),
        ("compute_matched", "recurrent_map_applications"),
    ):
        snapshots.append(
            {
                "snapshot_kind": kind,
                "status": "unavailable",
                "unavailable_reason": reason,
                "target": {
                    "unit": unit,
                    "registered_quantity": dict(unavailable),
                },
                "observed_environment_interactions": dict(unavailable),
                "observed_recurrent_map_applications": dict(unavailable),
                "accelerator_seconds_observed": dict(unavailable),
                "checkpoint_sha256": dict(unavailable),
                "model_state_sha256": dict(unavailable),
                "checkpoint_lineage_sha256": dict(unavailable),
                "policy_evaluations": [],
            }
        )
    training_fields = (
        "interactions_to_first_solve",
        "value_loss",
        "policy_loss",
        "wall_time_seconds",
        "gpu_hours",
        "gpu_utilization_fraction",
        "gpu_utilization_sample_count",
        "gpu_utilization_sampling_interval_seconds",
        "peak_allocated_memory_bytes",
        "peak_reserved_memory_bytes",
        "optimizer_steps",
        "cells_processed",
        "actions_processed",
        "tokens_processed",
        "policy_head_calls",
        "recurrent_map_applications",
        "value_head_calls",
    )
    diagnostics = (
        "L_preproj",
        "projection_active_rate",
        "absolute_depth_policy_agreement",
        "finite_depth_discrepancy",
        "fixed_target_residual",
        "propagated_target_lag",
        "exact_centering_defect",
        "candidate_current_tv",
        "mixture_realized_tv",
        "mixture_realized_kl",
        "fresh_initialization_discrepancy",
        "value_of_memory",
    )
    original_error = error.cause if isinstance(error, _SmokeExecutionError) else error
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": context.protocol["protocol_id"],
        "protocol_sha256": context.protocol_sha256,
        "amendment_history_sha256": amendment_history_sha256([]),
        "run_id": context.row["run_id"],
        "registry_row_sha256": context.registry_row_sha256,
        "phase": context.row["phase"],
        "tier": context.row["tier"],
        "seed": context.row["seed"],
        "evaluation_split": context.row["evaluation_split"],
        "method_id": context.row["method_id"],
        "base_method_id": context.row["base_method_id"],
        "n": context.row["n"],
        "K": context.row["K"],
        "alpha": context.row["alpha"],
        "ablation_variant": context.row["ablation_variant"],
        "applied_config_override": dict(context.row["config_override"]),
        "primary_policy_variant": primary_policy_variant_for_method(
            str(context.row["method_id"])
        ),
        "evaluation_snapshots": snapshots,
        "status": "failed",
        "failure": {
            "phase": phase,
            "error_class": type(original_error).__name__.lower(),
            "message_sha256": hashlib.sha256(
                str(original_error).encode("utf-8")
            ).hexdigest(),
        },
        "identities": identities,
        "metrics": {
            "training": {
                name: dict(measurement_unavailable) for name in training_fields
            },
            "diagnostics": {
                name: dict(measurement_unavailable) for name in diagnostics
            },
        },
        "artifacts": {
            "checkpoint": _unavailable(reason),
            "checkpoint_validation": _unavailable(reason),
            "model_state_inventory": _unavailable(reason),
            "run_manifest": _unavailable(reason),
        },
    }
    return _registered_result_document(context, session, result)


def _validate_failure_attempt(
    context: SmokeContext,
    path: Path,
    *,
    expected_attempt_id: str,
) -> None:
    result_bytes, result_identity = _stable_regular_file(path / "result.json")
    manifest_bytes, _ = _stable_regular_file(path / "MANIFEST.json")
    try:
        result = json.loads(
            result_bytes.decode("ascii"),
            object_pairs_hook=_strict_json_object,
        )
        manifest = json.loads(
            manifest_bytes.decode("ascii"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyImprovementSmokeError(
            "Failed Stage 0 attempt contains invalid JSON."
        ) from exc
    _, result_payload = validated_result_payload(result)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_name",
            "schema_version",
            "attempt_id",
            "protocol_sha256",
            "registry_row_sha256",
            "runtime_authorization_sha256",
            "run_id",
            "segment",
            "failure_phase",
            "result",
        }
        or manifest["schema_name"] != "policy_improvement_smoke_failed_attempt_v1"
        or manifest["schema_version"] != 1
        or manifest["attempt_id"] != expected_attempt_id
        or not isinstance(manifest["attempt_id"], str)
        or len(manifest["attempt_id"]) != 32
        or any(character not in _LOWER_SHA256 for character in manifest["attempt_id"])
        or manifest["protocol_sha256"] != context.protocol_sha256
        or manifest["registry_row_sha256"] != context.registry_row_sha256
        or manifest["runtime_authorization_sha256"]
        != context.runtime_authorization_sha256
        or manifest["run_id"] != context.row["run_id"]
        or manifest["segment"] != context.segment_name
        or manifest["failure_phase"] != result_payload["failure"]["phase"]
        or manifest["result"]
        != {
            "path": "result.json",
            "bytes": result_identity["bytes"],
            "sha256": result_identity["sha256"],
        }
    ):
        raise PolicyImprovementSmokeError(
            "Failed Stage 0 attempt manifest differs from its result."
        )
    actual_files, actual_directories = _segment_inventory(path)
    if set(actual_files) != {"MANIFEST.json", "result.json"} or actual_directories:
        raise PolicyImprovementSmokeError(
            "Failed Stage 0 attempt output inventory differs."
        )


def _publish_failure_attempt(
    context: SmokeContext,
    session: SmokeSession,
    *,
    runtime_preflight: Any,
    phase: str,
    error: BaseException,
) -> Path:
    attempts = _ensure_owned_directory(
        context.evidence_root,
        context.run_root / "attempts" / context.segment_name,
    )
    staging = Path(tempfile.mkdtemp(prefix=".failed-attempt-stage.", dir=str(attempts)))
    staging_identity = (staging.stat().st_dev, staging.stat().st_ino)
    attempt_id = uuid.uuid4().hex
    final = attempts / attempt_id
    try:
        result = _failed_result(
            context,
            session,
            phase=phase,
            error=error,
        )
        result_path = staging / "result.json"
        _write_json(result_path, result)
        _, result_identity = _stable_regular_file(result_path)
        manifest = {
            "schema_name": "policy_improvement_smoke_failed_attempt_v1",
            "schema_version": 1,
            "attempt_id": attempt_id,
            "protocol_sha256": context.protocol_sha256,
            "registry_row_sha256": context.registry_row_sha256,
            "runtime_authorization_sha256": (context.runtime_authorization_sha256),
            "run_id": context.row["run_id"],
            "segment": context.segment_name,
            "failure_phase": phase,
            "result": {
                "path": "result.json",
                "bytes": result_identity["bytes"],
                "sha256": result_identity["sha256"],
            },
        }
        _write_json(staging / "MANIFEST.json", manifest)
        _fsync_dir(staging)
        _validate_failure_attempt(
            context,
            staging,
            expected_attempt_id=attempt_id,
        )
        _revalidate_external_inputs(
            context,
            runtime_preflight=runtime_preflight,
        )
        _rename_noreplace(staging, final)
        _fsync_dir(attempts)
        published = final.lstat()
        if (
            not stat.S_ISDIR(published.st_mode)
            or stat.S_ISLNK(published.st_mode)
            or (published.st_dev, published.st_ino) != staging_identity
        ):
            raise PolicyImprovementSmokeError(
                "Published failed attempt has the wrong identity."
            )
        _validate_failure_attempt(
            context,
            final,
            expected_attempt_id=attempt_id,
        )
        return final
    except BaseException:
        try:
            if (
                staging.exists()
                and (
                    staging.stat().st_dev,
                    staging.stat().st_ino,
                )
                == staging_identity
            ):
                shutil.rmtree(staging)
        except OSError:
            pass
        raise


def main(
    argv: Sequence[str],
    *,
    runtime_preflight: Any,
    training_module: Any,
) -> int:
    """Run one registered segment. The launcher must invoke this directly."""

    arguments = _parse_args(argv)
    context = _load_context(arguments, runtime_preflight=runtime_preflight)
    session = _build_session(context, training_module)
    phase = "resume" if context.segment_name == "resume" else "training"
    try:
        try:
            parent_sha256 = _restore_parent(
                context,
                session,
                training_module,
            )
            parent_gpu = (
                _parent_gpu_utilization_summary(
                    context,
                    expected_device_type=session.device.type,
                )
                if context.segment_name == "resume"
                else None
            )
        except Exception as exc:
            raise _SmokeExecutionError("resume", exc) from exc

        sampler = _GpuTrainingSampler(session.device)
        try:
            sampler.start()
        except Exception as exc:
            raise _SmokeExecutionError("measurement", exc) from exc
        phase = "training"
        training_error: Exception | None = None
        try:
            latest = _train_to_budget(session, context.segment_budget)
        except Exception as exc:
            training_error = exc
            latest = {}
        try:
            segment_gpu = sampler.stop()
        except Exception as exc:
            raise _SmokeExecutionError("measurement", exc) from exc
        if training_error is not None:
            raise _SmokeExecutionError("training", training_error) from training_error
        gpu_utilization = (
            _combine_gpu_utilization_summaries(parent_gpu, segment_gpu)
            if parent_gpu is not None
            else segment_gpu
        )
        if (
            context.segment_name == "prepare"
            and session.trainer.get_env_step_count() != 16
        ):
            raise PolicyImprovementSmokeError(
                "Prepare segment did not end at interaction 16."
            )
        if (
            context.segment_name == "resume"
            and session.trainer.get_env_step_count() != 32
        ):
            raise PolicyImprovementSmokeError(
                "Resume segment did not end at interaction 32."
            )
        phase = "publication"
        destination = _publish_segment(
            context,
            session,
            training_module,
            runtime_preflight=runtime_preflight,
            parent_sha256=parent_sha256,
            latest_metrics=latest,
            gpu_utilization=gpu_utilization,
        )
    except Exception as exc:
        failure_phase = exc.phase if isinstance(exc, _SmokeExecutionError) else phase
        try:
            failure_destination = _publish_failure_attempt(
                context,
                session,
                runtime_preflight=runtime_preflight,
                phase=failure_phase,
                error=exc,
            )
        except Exception as publication_error:
            print(
                "Policy-improvement smoke failed without publishable evidence: "
                f"{exc}; failure publication rejected: {publication_error}",
                file=sys.stderr,
            )
            return 3
        print(
            canonical_json_bytes(
                {
                    "status": "failed",
                    "run_id": context.row["run_id"],
                    "segment": context.segment_name,
                    "failure_phase": failure_phase,
                    "attempt": failure_destination.name,
                }
            ).decode("ascii"),
            file=sys.stderr,
        )
        return 1
    print(
        canonical_json_bytes(
            {
                "status": "complete",
                "run_id": context.row["run_id"],
                "segment": context.segment_name,
                "environment_interactions": context.segment_budget,
                "generation": destination.name,
            }
        ).decode("ascii")
    )
    return 0
