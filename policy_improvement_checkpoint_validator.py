#!/usr/bin/env fbpython
"""Independent semantic validation for registered Stage-0 checkpoints."""

from __future__ import annotations

import hashlib
import importlib
import os
import random
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from dataset.build_policy_improvement_4x4 import verify_dataset
from policy_improvement_smoke_checkpoint import (
    load_stable_checkpoint,
    validate_ppo_smoke_checkpoint,
)
from policy_improvement_smoke_runtime import (
    PolicyImprovementSmokeError,
    SmokeContext,
    build_stage0_validation_session,
    stage0_model_state_identity,
    validate_policy_improvement_smoke_identity,
)
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from scripts.policy_improvement_registry import (
    generate_registry,
    load_registered_base_configs,
)
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
    validate_protocol,
    validate_registry_row,
    validate_runtime_authorization,
)
from utils.run_identity import (
    canonical_json_sha256,
    discover_clean_git_source,
)


_REQUEST_FIELDS = {
    "schema_name",
    "schema_version",
    "checkpoint_path",
    "checkpoint_sha256",
    "protocol",
    "protocol_sha256",
    "registry_row",
    "registry_row_sha256",
    "project_root",
    "dataset_root",
    "evidence_root",
    "runtime_authorization",
    "run_id",
    "method_id",
    "seed",
    "snapshot_kind",
    "environment_interactions",
    "parent_checkpoint_sha256",
    "initialization_sha256",
}
_RESULT_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "method_id",
    "seed",
    "snapshot_kind",
    "environment_interactions",
    "parent_checkpoint_sha256",
    "checkpoint_sha256",
    "initialization_sha256",
    "model_state_sha256",
    "role_state_sha256s",
    "method_config_sha256",
    "registered_effective_config_sha256",
    "effective_config_sha256",
    "dataset_manifest_sha256",
    "dataset_provenance_sha256",
    "run_identity_sha256",
    "training_call_delta",
    "evaluation_call_delta",
    "optimizer_step_delta",
}
_LOWER_HEX = frozenset("0123456789abcdef")


class PolicyImprovementCheckpointValidationError(RuntimeError):
    """Raised when checkpoint bytes do not implement their registered row."""


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PolicyImprovementCheckpointValidationError(f"{name} must be an object.")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise PolicyImprovementCheckpointValidationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _available_sha256(value: object, *, name: str) -> str:
    item = _object(value, name=name)
    if set(item) != {"status", "value"} or item["status"] != "available":
        raise PolicyImprovementCheckpointValidationError(
            f"{name} must be frozen and available."
        )
    return _sha256(item["value"], name=f"{name}.value")


def _absolute_path(
    value: object,
    *,
    name: str,
    kind: str,
) -> Path:
    if not isinstance(value, str) or not value or not value.isascii():
        raise PolicyImprovementCheckpointValidationError(
            f"{name} must be a nonempty ASCII path."
        )
    path = Path(value)
    if not path.is_absolute():
        raise PolicyImprovementCheckpointValidationError(f"{name} must be absolute.")
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementCheckpointValidationError(
            f"{name} is unavailable."
        ) from exc
    if resolved != path or stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid():
        raise PolicyImprovementCheckpointValidationError(
            f"{name} must be an owned canonical path without aliases."
        )
    if kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise PolicyImprovementCheckpointValidationError(f"{name} must be a directory.")
    if kind == "file" and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise PolicyImprovementCheckpointValidationError(
            f"{name} must be a singly linked regular file."
        )
    return path


def _runtime_role(
    authorization: Mapping[str, object], role_name: str
) -> Mapping[str, object]:
    roles = authorization["roles"]
    assert isinstance(roles, list)
    matches = [
        role
        for role in roles
        if isinstance(role, Mapping) and role.get("role") == role_name
    ]
    if len(matches) != 1:
        raise PolicyImprovementCheckpointValidationError(
            f"Runtime authorization lacks exact role {role_name!r}."
        )
    return matches[0]


def _validate_request(value: Mapping[str, object]) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
    Path,
]:
    if set(value) != _REQUEST_FIELDS:
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint validation request field inventory differs."
        )
    if (
        value["schema_name"] != "policy_improvement_checkpoint_validation_request_v1"
        or value["schema_version"] != 1
    ):
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint validation request schema is unsupported."
        )
    try:
        protocol = validate_protocol(value["protocol"])
        row = validate_registry_row(value["registry_row"])
        authorization = validate_runtime_authorization(value["runtime_authorization"])
    except PolicyImprovementSchemaError as exc:
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint validation request contains invalid registered evidence."
        ) from exc
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    row_sha256 = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
    if (
        _sha256(value["protocol_sha256"], name="protocol_sha256") != protocol_sha256
        or _sha256(value["registry_row_sha256"], name="registry_row_sha256")
        != row_sha256
        or authorization["protocol_sha256"] != protocol_sha256
    ):
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint validation request digest bindings differ."
        )
    run_id = value["run_id"]
    method_id = value["method_id"]
    seed = value["seed"]
    environment_interactions = value["environment_interactions"]
    if (
        row["row_kind"] != "concrete"
        or row["phase"] != "stage0_smoke"
        or row["tier"] != "smoke"
        or row["evaluation_split"] != "validation"
        or row["run_id"] != run_id
        or row["method_id"] != method_id
        or row["base_method_id"] != method_id
        or row["seed"] != seed
        or value["snapshot_kind"] != "interaction_matched"
        or isinstance(environment_interactions, bool)
        or environment_interactions not in {16, 32}
    ):
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint validation supports only a concrete registered "
            "Stage-0 snapshot."
        )
    registered_budgets = protocol["budgets"]["smoke"]
    if registered_budgets["environment_interactions"] != 32 or registered_budgets[
        "checkpoint_environment_interactions"
    ] != [16, 32]:
        raise PolicyImprovementCheckpointValidationError(
            "Registered Stage-0 checkpoint schedule is unsupported."
        )
    parent = value["parent_checkpoint_sha256"]
    if environment_interactions == 16:
        if parent is not None:
            raise PolicyImprovementCheckpointValidationError(
                "Stage-0 prepare validation must not name a parent."
            )
    else:
        _sha256(parent, name="parent_checkpoint_sha256")
    _sha256(value["checkpoint_sha256"], name="checkpoint_sha256")
    _sha256(value["initialization_sha256"], name="initialization_sha256")

    project_root = _absolute_path(
        value["project_root"], name="project_root", kind="directory"
    )
    dataset_root = _absolute_path(
        value["dataset_root"], name="dataset_root", kind="directory"
    )
    evidence_root = _absolute_path(
        value["evidence_root"], name="evidence_root", kind="directory"
    )
    checkpoint_path = _absolute_path(
        value["checkpoint_path"], name="checkpoint_path", kind="file"
    )
    try:
        checkpoint_path.relative_to(evidence_root)
    except ValueError as exc:
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint path escaped the evidence root."
        ) from exc
    expected_checkpoint_parent = (
        evidence_root
        / "runs"
        / str(run_id)
        / "segments"
        / f"env_{environment_interactions:09d}"
        / "checkpoints"
    )
    if checkpoint_path.parent != expected_checkpoint_parent:
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint path differs from its immutable Stage-0 generation."
        )
    expected_dataset = project_root / str(protocol["dataset"]["root"])
    try:
        resolved_expected_dataset = expected_dataset.resolve(strict=True)
    except OSError as exc:
        raise PolicyImprovementCheckpointValidationError(
            "Registered dataset root is unavailable."
        ) from exc
    if dataset_root != resolved_expected_dataset:
        raise PolicyImprovementCheckpointValidationError(
            "Dataset root differs from the registered protocol."
        )
    return (
        protocol,
        row,
        authorization,
        project_root,
        dataset_root,
        evidence_root,
        checkpoint_path,
    )


def _build_context(
    request: Mapping[str, object],
    protocol: dict[str, Any],
    row: dict[str, Any],
    authorization: dict[str, Any],
    project_root: Path,
    dataset_root: Path,
    evidence_root: Path,
) -> SmokeContext:
    source_identity = discover_clean_git_source(project_root)
    training_role = _runtime_role(authorization, "policy-improvement-training")
    evaluation_role = _runtime_role(authorization, "policy-improvement-evaluation")
    if source_identity != {
        "git_commit": authorization["producer_git_commit"],
        "git_clean": True,
    } or any(
        training_role[field] != evaluation_role[field]
        for field in (
            "source_git_commit",
            "runtime_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
        )
    ):
        raise PolicyImprovementCheckpointValidationError(
            "Stage-0 source checkout or in-process evaluation runtime differs."
        )
    if (
        training_role["source_git_commit"] != source_identity["git_commit"]
        or training_role["runtime_profile_sha256"]
        != authorization["producer_source_manifest_sha256"]
    ):
        raise PolicyImprovementCheckpointValidationError(
            "Stage-0 originating runtime is not authorized for this producer."
        )

    registry = generate_registry(
        protocol,
        base_configs=load_registered_base_configs(protocol, project_root),
    )
    matching_rows = [
        item for item in registry["rows"] if item["run_id"] == row["run_id"]
    ]
    if len(matching_rows) != 1 or matching_rows[0] != row:
        raise PolicyImprovementCheckpointValidationError(
            "Requested Stage-0 row differs from deterministic registry generation."
        )

    dataset_registration = protocol["dataset"]
    dataset_manifest_sha256 = _available_sha256(
        dataset_registration["manifest_sha256"],
        name="dataset.manifest_sha256",
    )
    producer_source = {
        field: _available_sha256(
            dataset_registration["producer_source"][field],
            name=f"dataset.producer_source.{field}",
        )
        for field in (
            "launcher_sha256",
            "runtime_sha256",
            "source_manifest_sha256",
        )
    }
    producer_commit = _object(
        dataset_registration["producer_source"]["git_commit"],
        name="dataset.producer_source.git_commit",
    )
    if (
        set(producer_commit) != {"status", "value"}
        or producer_commit["status"] != "available"
        or not isinstance(producer_commit["value"], str)
        or len(producer_commit["value"]) != 40
        or any(character not in _LOWER_HEX for character in producer_commit["value"])
    ):
        raise PolicyImprovementCheckpointValidationError(
            "dataset.producer_source.git_commit must be frozen and available."
        )
    registered_dataset_producer = {
        "git_commit": producer_commit["value"],
        **producer_source,
    }
    verified_dataset = verify_dataset(
        dataset_root,
        owner_root=dataset_root.parent,
        expected_producer=registered_dataset_producer,
    )
    if verified_dataset.get("manifest_sha256") != dataset_manifest_sha256:
        raise PolicyImprovementCheckpointValidationError(
            "Materialized dataset differs from the frozen root manifest."
        )
    split_manifests = _object(
        verified_dataset.get("split_manifest_sha256"),
        name="verified_dataset.split_manifest_sha256",
    )
    split_orders = _object(
        verified_dataset.get("split_ordered_record_sha256"),
        name="verified_dataset.split_ordered_record_sha256",
    )
    for split in ("train", "validation", "test"):
        if split_manifests.get(split) != _available_sha256(
            dataset_registration["splits"][split]["manifest_sha256"],
            name=f"dataset.splits.{split}.manifest_sha256",
        ) or split_orders.get(split) != _available_sha256(
            dataset_registration["splits"][split]["ordered_record_sha256"],
            name=f"dataset.splits.{split}.ordered_record_sha256",
        ):
            raise PolicyImprovementCheckpointValidationError(
                f"Materialized {split} identity differs from the protocol."
            )

    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    registry_sha256 = hashlib.sha256(canonical_json_bytes(registry)).hexdigest()
    row_sha256 = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
    interactions = int(request["environment_interactions"])
    return SmokeContext(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
        row=row,
        registry_row_sha256=row_sha256,
        source_root=project_root,
        dataset_root=dataset_root,
        dataset_manifest_sha256=dataset_manifest_sha256,
        dataset_producer_source=registered_dataset_producer,
        evidence_root=evidence_root,
        run_root=evidence_root / "runs" / str(row["run_id"]),
        segment_budget=interactions,
        segment_name="prepare" if interactions == 16 else "resume",
        runtime_sha256=str(training_role["runtime_sha256"]),
        runtime_authorization_sha256=canonical_json_sha256(authorization),
        runtime_profile_sha256=str(training_role["runtime_profile_sha256"]),
        selected_source_manifest_sha256=str(
            training_role["selected_source_manifest_sha256"]
        ),
        launcher_sha256=str(authorization["launcher_sha256"]),
        producer_commit=str(authorization["producer_git_commit"]),
        producer_manifest_sha256=str(authorization["producer_source_manifest_sha256"]),
    )


@contextmanager
def _forbid_training_and_optimizer_steps(trainer: Any) -> Iterator[dict[str, int]]:
    counters = {"training": 0, "optimizer": 0}
    restorations: list[tuple[object, str, object]] = []

    train_step = getattr(trainer, "train_step", None)
    if callable(train_step):

        def forbidden_train_step(*args: object, **kwargs: object) -> object:
            del args, kwargs
            counters["training"] += 1
            raise PolicyImprovementCheckpointValidationError(
                "Checkpoint validation attempted to train."
            )

        restorations.append((trainer, "train_step", train_step))
        setattr(trainer, "train_step", forbidden_train_step)

    optimizer_classes: dict[type[object], object] = {}
    for name in (
        "optimizer",
        "value_opt",
        "policy_opt",
        "old_policy_distill_opt",
        "puzzle_emb_optimizer",
    ):
        optimizer = getattr(trainer, name, None)
        if optimizer is None:
            continue
        optimizer_class = type(optimizer)
        optimizer_classes.setdefault(optimizer_class, optimizer_class.step)
    for optimizer_class, original_step in optimizer_classes.items():

        def forbidden_optimizer_step(
            optimizer_self: object,
            *args: object,
            _original: object = original_step,
            **kwargs: object,
        ) -> object:
            del optimizer_self, args, kwargs, _original
            counters["optimizer"] += 1
            raise PolicyImprovementCheckpointValidationError(
                "Checkpoint validation attempted an optimizer step."
            )

        restorations.append((optimizer_class, "step", original_step))
        setattr(optimizer_class, "step", forbidden_optimizer_step)
    try:
        yield counters
    finally:
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)


def _validate_ppo(
    request: Mapping[str, object],
    checkpoint_path: Path,
    session: Any,
) -> tuple[str, str | None, int, str, dict[str, str]]:
    expected_sha256 = str(request["checkpoint_sha256"])
    payload, observed_sha256 = load_stable_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
    )
    if not isinstance(payload, Mapping):
        raise PolicyImprovementCheckpointValidationError(
            "PPO checkpoint payload is not an object."
        )
    validate_ppo_smoke_checkpoint(
        payload,
        session.trainer,
        expected_identity=session.effective_config,
        validate_only=True,
    )
    progress = _object(payload.get("progress"), name="ppo.progress")
    interactions = progress.get("environment_interactions")
    parent = payload.get("parent_checkpoint_sha256")
    role_hashes = {
        "model": state_dict_sha256(payload["model_state_dict"]),
    }
    return (
        observed_sha256,
        parent if isinstance(parent, str) else None,
        (
            interactions
            if isinstance(interactions, int) and not isinstance(interactions, bool)
            else -1
        ),
        canonical_json_sha256(role_hashes),
        role_hashes,
    )


def _validate_upi(
    request: Mapping[str, object],
    checkpoint_path: Path,
    context: SmokeContext,
    session: Any,
    module: Any,
) -> tuple[str, str | None, int, str, dict[str, str]]:
    expected_sha256 = str(request["checkpoint_sha256"])
    raw, observed_sha256 = module._load_checkpoint_payload(
        str(checkpoint_path),
        expected_sha256=expected_sha256,
    )
    if not isinstance(raw, Mapping):
        raise PolicyImprovementCheckpointValidationError(
            "UPI checkpoint payload is not an object."
        )
    interactions = int(request["environment_interactions"])
    parent = request["parent_checkpoint_sha256"]
    smoke_identity = validate_policy_improvement_smoke_identity(
        raw.get("policy_improvement_smoke_identity"),
        context,
        environment_interactions=interactions,
        parent_checkpoint_sha256=parent if isinstance(parent, str) else None,
    )
    if raw.get("evidence_identity") != session.evidence_identity:
        raise PolicyImprovementCheckpointValidationError(
            "UPI checkpoint evidence identity differs from the reconstructed row."
        )
    invocation = _object(raw.get("training_invocation"), name="upi.training_invocation")
    if (
        invocation.get("training_seed") != request["seed"]
        or invocation.get("run_id") != request["run_id"]
    ):
        raise PolicyImprovementCheckpointValidationError(
            "UPI checkpoint invocation differs from the registered row."
        )
    module.validate_checkpoint_state_for_audit(
        str(checkpoint_path),
        session.model,
        session.trainer,
        str(session.device),
        expected_dataset_provenance=session.dataset_provenance,
        expected_run_identity=session.run_identity,
        expected_checkpoint_sha256=expected_sha256,
        authorized_originating_runtime_sha256=context.runtime_sha256,
    )
    model_sha256, role_hashes = stage0_model_state_identity(session)
    role_fields = {
        "model": "model_state_dict",
        "policy_model_old": "policy_model_old_state_dict",
        "policy_model_candidate": "policy_model_candidate_state_dict",
        "target_model": "target_model_state_dict",
        "preinterpolation_policy_base": "preinterpolation_policy_base_state_dict",
        "preinterpolation_policy_candidate": (
            "preinterpolation_policy_candidate_state_dict"
        ),
    }
    loaded_role_hashes = {
        role: state_dict_sha256(raw[field])
        for role, field in role_fields.items()
        if field in raw
    }
    if loaded_role_hashes != role_hashes:
        raise PolicyImprovementCheckpointValidationError(
            "Restored UPI model roles differ from checkpoint state dictionaries."
        )
    progress = _object(raw.get("progress"), name="upi.progress")
    observed_interactions = progress.get("env_steps")
    lineage = raw.get("checkpoint_lineage")
    if (
        isinstance(lineage, Mapping)
        and lineage.get("parent_checkpoint_sha256")
        != smoke_identity["parent_checkpoint_sha256"]
    ):
        raise PolicyImprovementCheckpointValidationError(
            "UPI fixed-base lineage differs from its Stage-0 identity."
        )
    return (
        observed_sha256,
        (
            smoke_identity["parent_checkpoint_sha256"]
            if isinstance(smoke_identity["parent_checkpoint_sha256"], str)
            else None
        ),
        (
            observed_interactions
            if isinstance(observed_interactions, int)
            and not isinstance(observed_interactions, bool)
            else -1
        ),
        model_sha256,
        role_hashes,
    )


def validate_checkpoint(request: Mapping[str, object]) -> dict[str, object]:
    """Strictly reconstruct and reload one immutable registered smoke checkpoint."""

    if not isinstance(request, Mapping):
        raise PolicyImprovementCheckpointValidationError(
            "Checkpoint validation request must be an object."
        )
    try:
        (
            protocol,
            row,
            authorization,
            project_root,
            dataset_root,
            evidence_root,
            checkpoint_path,
        ) = _validate_request(request)
        context = _build_context(
            request,
            protocol,
            row,
            authorization,
            project_root,
            dataset_root,
            evidence_root,
        )
        training_module = importlib.import_module("upi_trm_train")
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            session = build_stage0_validation_session(context, training_module)
            if session.initialization_sha256 != request["initialization_sha256"]:
                raise PolicyImprovementCheckpointValidationError(
                    "Reconstructed initialization differs from the result."
                )
            method_registration = next(
                method
                for method in protocol["methods"]
                if method["id"] == request["method_id"]
            )
            if session.method_config_sha256 != method_registration["config_sha256"]:
                raise PolicyImprovementCheckpointValidationError(
                    "Reconstructed method file differs from the protocol registration."
                )
            with _forbid_training_and_optimizer_steps(session.trainer) as calls:
                if request["method_id"] == "matched_ppo":
                    semantic = _validate_ppo(request, checkpoint_path, session)
                else:
                    semantic = _validate_upi(
                        request,
                        checkpoint_path,
                        context,
                        session,
                        training_module,
                    )
            observed_sha256, parent, interactions, model_sha256, role_hashes = semantic
            if (
                observed_sha256 != request["checkpoint_sha256"]
                or parent != request["parent_checkpoint_sha256"]
                or interactions != request["environment_interactions"]
                or calls != {"training": 0, "optimizer": 0}
                or session.evaluation_started
            ):
                raise PolicyImprovementCheckpointValidationError(
                    "Checkpoint semantic progress, lineage, or execution "
                    "boundary differs."
                )
            result: dict[str, object] = {
                "schema_name": "policy_improvement_checkpoint_semantic_validation_v1",
                "schema_version": 1,
                "run_id": request["run_id"],
                "method_id": request["method_id"],
                "seed": request["seed"],
                "snapshot_kind": request["snapshot_kind"],
                "environment_interactions": interactions,
                "parent_checkpoint_sha256": parent,
                "checkpoint_sha256": observed_sha256,
                "initialization_sha256": session.initialization_sha256,
                "model_state_sha256": model_sha256,
                "role_state_sha256s": role_hashes,
                "method_config_sha256": session.method_config_sha256,
                "registered_effective_config_sha256": row[
                    "expected_effective_config_sha256"
                ],
                "effective_config_sha256": session.effective_config_sha256,
                "dataset_manifest_sha256": context.dataset_manifest_sha256,
                "dataset_provenance_sha256": canonical_json_sha256(
                    session.dataset_provenance
                ),
                "run_identity_sha256": (
                    canonical_json_sha256(session.run_identity)
                    if session.run_identity is not None
                    else None
                ),
                "training_call_delta": calls["training"],
                "evaluation_call_delta": int(session.evaluation_started),
                "optimizer_step_delta": calls["optimizer"],
            }
            if set(result) != _RESULT_FIELDS:
                raise AssertionError("Semantic checkpoint result inventory drifted.")
            canonical_json_bytes(result)
            return result
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
    except PolicyImprovementCheckpointValidationError:
        raise
    except Exception as exc:
        if isinstance(exc, PolicyImprovementSmokeError):
            message = str(exc)
        else:
            message = "checkpoint semantic validation failed"
        raise PolicyImprovementCheckpointValidationError(message) from exc
