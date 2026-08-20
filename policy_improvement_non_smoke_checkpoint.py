#!/usr/bin/env fbpython
"""Strict continuation checkpoints for registered non-smoke runs.

This module adds the experiment identity that the generic trainer checkpoint
does not know about.  Validation always restores into a disposable session.
The live training session and process RNG are fingerprinted before and after
that restore so checkpoint validation cannot advance the learned run.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from policy_improvement_smoke_checkpoint import (
    build_ppo_smoke_checkpoint,
    load_stable_checkpoint,
    publish_checkpoint,
    validate_ppo_smoke_checkpoint,
)
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from utils.run_identity import build_checkpoint_lineage, canonical_json_sha256


FULL_CHECKPOINT_IDENTITY_SCHEMA_VERSION: int = 1
_SNAPSHOT_KINDS: frozenset[str] = frozenset(
    {"interaction_matched", "compute_matched", "scheduled"}
)


class FullCheckpointError(RuntimeError):
    """Raised when a non-smoke checkpoint is not an exact continuation."""


@dataclass(frozen=True)
class FullCheckpointArtifact:
    """One immutable checkpoint and its semantic validation document."""

    path: Path
    sha256: str
    model_state_sha256: str
    role_state_sha256s: dict[str, str]
    theory_model_identity: dict[str, str] | None
    validation: dict[str, object]


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FullCheckpointError(
            f"{name} must be 64 lowercase hexadecimal characters."
        )
    return value


def validate_full_checkpoint_identity(value: object) -> dict[str, object]:
    """Validate the complete row, runtime, dataset, and snapshot binding."""

    fields = {
        "schema_name",
        "schema_version",
        "run_id",
        "method_id",
        "protocol_sha256",
        "registry_row_sha256",
        "amendment_history_sha256",
        "runtime_authorization_sha256",
        "training_runtime_sha256",
        "training_source_git_commit",
        "training_source_manifest_sha256",
        "launcher_sha256",
        "dataset_manifest_sha256",
        "dataset_provenance_sha256",
        "effective_config_sha256",
        "snapshot_kind",
        "environment_interactions",
        "recurrent_map_applications",
        "parent_checkpoint_sha256",
        "test_open_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FullCheckpointError("Full checkpoint identity field inventory differs.")
    identity = dict(value)
    if (
        identity["schema_name"] != "policy_improvement_full_checkpoint_identity_v1"
        or identity["schema_version"] != FULL_CHECKPOINT_IDENTITY_SCHEMA_VERSION
    ):
        raise FullCheckpointError("Full checkpoint identity schema is unsupported.")
    for field in ("run_id", "method_id"):
        item = identity[field]
        if not isinstance(item, str) or not item:
            raise FullCheckpointError(f"Full checkpoint {field} is invalid.")
    for field in (
        "protocol_sha256",
        "registry_row_sha256",
        "amendment_history_sha256",
        "runtime_authorization_sha256",
        "training_runtime_sha256",
        "training_source_manifest_sha256",
        "launcher_sha256",
        "dataset_manifest_sha256",
        "dataset_provenance_sha256",
        "effective_config_sha256",
    ):
        _sha256(identity[field], name=f"Full checkpoint {field}")
    commit = identity["training_source_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise FullCheckpointError("Full checkpoint source commit is invalid.")
    snapshot_kind = identity["snapshot_kind"]
    if snapshot_kind not in _SNAPSHOT_KINDS:
        raise FullCheckpointError("Full checkpoint snapshot kind is invalid.")
    for field in ("environment_interactions", "recurrent_map_applications"):
        item = identity[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise FullCheckpointError(f"Full checkpoint {field} is invalid.")
    for field in ("parent_checkpoint_sha256", "test_open_sha256"):
        item = identity[field]
        if item is not None:
            _sha256(item, name=f"Full checkpoint {field}")
    canonical_json_sha256(identity)
    return identity


def _stable_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise FullCheckpointError(
                    "Checkpoint must be a singly linked regular file."
                )
            digest = hashlib.sha256()
            for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise FullCheckpointError("Checkpoint cannot be hashed safely.") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise FullCheckpointError("Checkpoint changed while it was hashed.")
    return digest.hexdigest()


def _clone_state_dict(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, item in value.items():
        if torch.is_tensor(item):
            result[name] = item.detach().cpu().clone()
        else:
            result[name] = item
    return result


def _assert_finite_state_dict(value: object, *, role: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise FullCheckpointError(f"Checkpoint role {role!r} has no state dictionary.")
    for name, item in value.items():
        if not isinstance(name, str) or not torch.is_tensor(item):
            raise FullCheckpointError(
                f"Checkpoint role {role!r} contains a non-tensor state entry."
            )
        if (item.is_floating_point() or item.is_complex()) and not bool(
            torch.isfinite(item).all().item()
        ):
            raise FullCheckpointError(
                f"Checkpoint role {role!r} contains nonfinite parameters."
            )


def _session_role_modules(session: Any) -> dict[str, Any]:
    trainer = session.trainer
    if type(trainer).__name__ == "PPOTrainer":
        return {"model": session.model}
    roles = {
        "model": session.model,
        "policy_model_old": trainer.policy_model_old,
        "policy_model_candidate": trainer.policy_model_candidate,
        "target_model": trainer.target_model,
    }
    for name in (
        "preinterpolation_policy_base",
        "preinterpolation_policy_candidate",
    ):
        module = getattr(trainer, name, None)
        if module is not None:
            roles[name] = module
    return roles


def session_model_state_identity(
    session: Any,
    *,
    evaluation_state_dicts: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[str, dict[str, str]]:
    """Hash every behavior-bearing model role in one live or restored session."""

    hashes: dict[str, str] = {}
    for name, module in sorted(_session_role_modules(session).items()):
        state = module.state_dict()
        _assert_finite_state_dict(state, role=name)
        hashes[name] = state_dict_sha256(state)
    if evaluation_state_dicts is not None:
        for name, state in sorted(evaluation_state_dicts.items()):
            _assert_finite_state_dict(state, role=name)
            hashes[name] = state_dict_sha256(state)
    return canonical_json_sha256(hashes), hashes


def session_theory_model_identity(
    session: Any,
    *,
    model_config_sha256: str,
    evaluation_state_dicts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, str] | None:
    """Derive the theorem-facing model identity from one restored session."""

    _sha256(model_config_sha256, name="Theory model config")
    method = str(session.run.row["method_id"])
    trainer = session.trainer
    if method in {
        "fixed_base_exact_persistent",
        "fixed_base_exact_episodic",
    }:
        current_state = trainer.policy_model_old.state_dict()
        candidate_state = trainer.policy_model_candidate.state_dict()
        deployed_state = current_state
        deployed_policy_sha256 = canonical_json_sha256(
            {
                "kind": "exact_probability_mixture",
                "current_policy_sha256": state_dict_sha256(current_state),
                "candidate_policy_sha256": state_dict_sha256(candidate_state),
                "alpha": float(session.run.row["alpha"]),
                "recurrent_transition_sha256": state_dict_sha256(
                    {
                        name: value
                        for name, value in deployed_state.items()
                        if not name.startswith("edit_policy.")
                        and not name.startswith("value_head.")
                    }
                ),
            }
        )
    elif method == "legacy_parameter_interpolation":
        current_model = getattr(trainer, "preinterpolation_policy_base", None)
        candidate_model = getattr(trainer, "preinterpolation_policy_candidate", None)
        if current_model is None or candidate_model is None:
            raise FullCheckpointError(
                "Legacy checkpoint lacks its pre-interpolation policy pair."
            )
        current_state = current_model.state_dict()
        candidate_state = candidate_model.state_dict()
        deployed_state = trainer.policy_model_old.state_dict()
        deployed_policy_sha256 = state_dict_sha256(deployed_state)
    elif method == "fixed_base_distilled_realization":
        states = dict(evaluation_state_dicts or {})
        if set(states) != {"base", "candidate"}:
            raise FullCheckpointError(
                "Distilled checkpoint lacks its frozen base/candidate pair."
            )
        current_state = states["base"]
        candidate_state = states["candidate"]
        deployed_state = trainer.policy_model_old.state_dict()
        deployed_policy_sha256 = state_dict_sha256(deployed_state)
    else:
        return None

    def recurrent_state(state: Mapping[str, object]) -> dict[str, object]:
        return {
            name: value
            for name, value in state.items()
            if not name.startswith("edit_policy.")
            and not name.startswith("value_head.")
        }

    recurrent_sha256s = {
        state_dict_sha256(recurrent_state(state))
        for state in (current_state, candidate_state, deployed_state)
    }
    if len(recurrent_sha256s) != 1:
        raise FullCheckpointError(
            "Theory policy roles do not share one frozen recurrent map."
        )
    model_sha256, _ = session_model_state_identity(
        session,
        evaluation_state_dicts=evaluation_state_dicts,
    )
    return {
        "model_sha256": model_sha256,
        "model_config_sha256": model_config_sha256,
        "current_policy_sha256": state_dict_sha256(current_state),
        "candidate_policy_sha256": state_dict_sha256(candidate_state),
        "deployed_policy_sha256": deployed_policy_sha256,
        "recurrent_transition_sha256": next(iter(recurrent_sha256s)),
    }


def _rewrite_upi_checkpoint(
    path: Path,
    payload: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    evaluation_state_dicts: Mapping[str, Mapping[str, object]],
) -> None:
    rebound = dict(payload)
    rebound["policy_improvement_full_identity"] = dict(identity)
    rebound["policy_improvement_full_identity_sha256"] = canonical_json_sha256(
        dict(identity)
    )
    rebound["policy_improvement_full_evaluation_state_dicts"] = {
        name: _clone_state_dict(state)
        for name, state in sorted(evaluation_state_dicts.items())
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(rebound, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _validate_embedded_identity(
    payload: object,
    *,
    expected_identity: Mapping[str, object],
    ppo: bool,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        raise FullCheckpointError("Checkpoint payload must be one mapping.")
    if ppo:
        observed = payload.get("identity")
        evaluation_states: object = {}
    else:
        observed = payload.get("policy_improvement_full_identity")
        if payload.get("policy_improvement_full_identity_sha256") != (
            canonical_json_sha256(dict(expected_identity))
        ):
            raise FullCheckpointError("Checkpoint identity digest differs.")
        evaluation_states = payload.get(
            "policy_improvement_full_evaluation_state_dicts"
        )
    checked = validate_full_checkpoint_identity(observed)
    if checked != dict(expected_identity):
        raise FullCheckpointError("Checkpoint identity differs from the active row.")
    if not isinstance(evaluation_states, Mapping):
        raise FullCheckpointError("Checkpoint evaluation-state inventory is invalid.")
    canonical_states: dict[str, Mapping[str, object]] = {}
    for name, state in evaluation_states.items():
        if not isinstance(name, str) or not isinstance(state, Mapping):
            raise FullCheckpointError(
                "Checkpoint evaluation-state inventory is invalid."
            )
        _assert_finite_state_dict(state, role=name)
        canonical_states[name] = state
    return canonical_states


def publish_and_validate_full_checkpoint(
    *,
    training_module: Any,
    session: Any,
    fresh_session_factory: Callable[[], Any],
    checkpoint_directory: Path,
    identity: Mapping[str, object],
    validator_execution_identity: Mapping[str, object],
    evaluation_state_dicts: Mapping[str, Mapping[str, object]] | None = None,
    model_config_sha256: str,
) -> FullCheckpointArtifact:
    """Publish one checkpoint and prove an exact restore on a fresh session."""

    checked_identity = validate_full_checkpoint_identity(identity)
    if checked_identity["run_id"] != session.run.row["run_id"]:
        raise FullCheckpointError("Checkpoint row differs from the live session.")
    checkpoint_directory.mkdir(parents=True, mode=0o700)
    if any(checkpoint_directory.iterdir()):
        raise FullCheckpointError("Checkpoint directory must start empty.")
    evaluation_states = dict(evaluation_state_dicts or {})
    before_model, before_roles = session_model_state_identity(
        session, evaluation_state_dicts=evaluation_states
    )
    before_theory = (
        session_theory_model_identity(
            session,
            model_config_sha256=model_config_sha256,
            evaluation_state_dicts=evaluation_states,
        )
        if session.run.protocol.get("schema_name") == "policy_improvement_protocol_v2"
        else None
    )
    rng_state = training_module._capture_rng_state()
    ppo = type(session.trainer).__name__ == "PPOTrainer"
    try:
        if ppo:
            if evaluation_states:
                raise FullCheckpointError(
                    "PPO checkpoints cannot contain auxiliary evaluation models."
                )
            payload = build_ppo_smoke_checkpoint(
                session.trainer,
                identity=checked_identity,
                parent_checkpoint_sha256=checked_identity["parent_checkpoint_sha256"],
            )
            checkpoint = checkpoint_directory / "ppo_exact_continuation.pt"
            checkpoint_sha256 = publish_checkpoint(payload, checkpoint)
        else:
            lineage = None
            if session.run_identity is not None:
                parent = checked_identity["parent_checkpoint_sha256"]
                lineage = build_checkpoint_lineage(
                    parent_checkpoint_sha256=parent,
                    parent_checkpoint_step=(
                        session.parent_environment_interactions
                        if parent is not None
                        else None
                    ),
                    parent_environment_steps=(
                        session.parent_environment_interactions
                        if parent is not None
                        else None
                    ),
                )
            checkpoint_name = training_module.save_checkpoint(
                session.model,
                session.trainer,
                int(checked_identity["environment_interactions"]),
                str(checkpoint_directory),
                None,
                session.rl_config,
                session.dataset_provenance,
                session.run_identity,
                lineage,
                session.evidence_identity,
                training_seed=int(session.run.row["seed"]),
                training_run_id=str(session.run.row["run_id"]),
                config_source_paths=[str(session.config_path)],
            )
            checkpoint = Path(checkpoint_name)
            raw, _ = training_module._load_checkpoint_payload(str(checkpoint))
            if not isinstance(raw, Mapping):
                raise FullCheckpointError("UPI checkpoint payload is invalid.")
            _rewrite_upi_checkpoint(
                checkpoint,
                raw,
                identity=checked_identity,
                evaluation_state_dicts=evaluation_states,
            )
            checkpoint_sha256 = _stable_sha256(checkpoint)

        if _stable_sha256(checkpoint) != checkpoint_sha256:
            raise FullCheckpointError("Checkpoint changed after publication.")
        fresh = fresh_session_factory()
        if ppo:
            loaded, observed = load_stable_checkpoint(
                checkpoint, expected_sha256=checkpoint_sha256
            )
            validate_ppo_smoke_checkpoint(
                loaded,
                fresh.trainer,
                expected_identity=checked_identity,
                validate_only=False,
            )
        else:
            loaded, observed = training_module._load_checkpoint_payload(
                str(checkpoint), expected_sha256=checkpoint_sha256
            )
            training_module.resume_from_checkpoint(
                str(checkpoint),
                fresh.model,
                fresh.trainer,
                str(fresh.device),
                expected_dataset_provenance=fresh.dataset_provenance,
                expected_run_identity=fresh.run_identity,
                expected_checkpoint_sha256=checkpoint_sha256,
                allow_legacy_warm_start=False,
            )
        if observed != checkpoint_sha256:
            raise FullCheckpointError("Checkpoint digest changed during restore.")
        restored_states = _validate_embedded_identity(
            loaded,
            expected_identity=checked_identity,
            ppo=ppo,
        )
        restored_model, restored_roles = session_model_state_identity(
            fresh,
            evaluation_state_dicts=restored_states,
        )
        restored_theory = (
            session_theory_model_identity(
                fresh,
                model_config_sha256=model_config_sha256,
                evaluation_state_dicts=restored_states,
            )
            if session.run.protocol.get("schema_name")
            == "policy_improvement_protocol_v2"
            else None
        )
        if (
            fresh.trainer.get_env_step_count()
            != checked_identity["environment_interactions"]
            or restored_model != before_model
            or restored_roles != before_roles
            or restored_theory != before_theory
        ):
            raise FullCheckpointError(
                "Checkpoint restore changed progress or behavior-bearing model state."
            )
    finally:
        training_module._restore_rng_state(rng_state)
    after_model, after_roles = session_model_state_identity(
        session, evaluation_state_dicts=evaluation_states
    )
    if after_model != before_model or after_roles != before_roles:
        raise FullCheckpointError(
            "Checkpoint validation mutated the live training state."
        )
    validator_fields = {
        "role",
        "source_git_commit",
        "runtime_sha256",
        "runtime_profile_sha256",
        "selected_source_manifest_sha256",
        "runtime_authorization_sha256",
        "launcher_sha256",
    }
    if set(validator_execution_identity) != validator_fields:
        raise FullCheckpointError(
            "Full checkpoint validator execution identity inventory differs."
        )
    validation = {
        "schema_name": "policy_improvement_checkpoint_validation_v1",
        "schema_version": 1,
        "validator": "policy_improvement_full_runtime",
        "validator_execution_identity": dict(validator_execution_identity),
        "run_id": checked_identity["run_id"],
        "method_id": checked_identity["method_id"],
        "snapshot_kind": checked_identity["snapshot_kind"],
        "environment_interactions": checked_identity["environment_interactions"],
        "checkpoint_sha256": checkpoint_sha256,
        "parent_checkpoint_sha256": checked_identity["parent_checkpoint_sha256"],
        "model_state_sha256": before_model,
        "role_state_sha256s": before_roles,
        "strict_resume_validated": True,
    }
    if session.run.protocol.get("schema_name") == "policy_improvement_protocol_v2":
        validation["schema_version"] = 2
        validation["theory_model_identity"] = before_theory
    return FullCheckpointArtifact(
        path=checkpoint,
        sha256=checkpoint_sha256,
        model_state_sha256=before_model,
        role_state_sha256s=before_roles,
        theory_model_identity=before_theory,
        validation=validation,
    )


def evaluate_without_mutation(
    *,
    checkpoint_path: Path,
    live_state_fingerprint: Callable[[], str],
    evaluator: Callable[[], Any],
) -> Any:
    """Run an evaluator while proving checkpoint and live state immutability."""

    checkpoint_before = _stable_sha256(checkpoint_path)
    live_before = live_state_fingerprint()
    result: Any = None
    evaluator_error: BaseException | None = None
    try:
        result = evaluator()
    except BaseException as exc:
        evaluator_error = exc
    mutation_error: BaseException | None = None
    try:
        if _stable_sha256(checkpoint_path) != checkpoint_before:
            raise FullCheckpointError("Evaluation mutated its checkpoint.")
        if live_state_fingerprint() != live_before:
            raise FullCheckpointError("Evaluation mutated the live training state.")
    except BaseException as exc:
        mutation_error = exc
    if mutation_error is not None:
        if evaluator_error is not None:
            raise mutation_error from evaluator_error
        raise mutation_error
    if evaluator_error is not None:
        raise evaluator_error
    return result
