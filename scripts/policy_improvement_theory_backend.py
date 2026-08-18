#!/usr/bin/env fbpython
"""Concrete sealed adapter from full-run checkpoints to the theory bridge.

Imports of the training module and full backend are intentionally lazy.  The
authenticated pre-import entrypoint must complete runtime authorization before
this factory can construct a disposable, strictly resumed evaluation session.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast, Protocol

from scripts.policy_improvement_full_runtime import (
    FULL_EXECUTION_ENV,
    FULL_EXECUTION_VALUE,
    FullRuntimeError,
    load_registered_full_run,
    resolve_authenticated_full_checkpoint,
)
from scripts.policy_improvement_registry import load_registered_base_configs
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    validate_runtime_authorization,
)
from scripts.policy_improvement_theory_bridge import (
    ReadOnlySnapshot,
    TheoryBackendInputs,
    TheoryBridgeBackend,
    TheoryBridgeError,
    TheoryOutcome,
    TheoryRollout,
    TheoryState,
)


class _FullTheorySession(Protocol):
    def identity_bundle(self) -> Mapping[str, object]: ...

    def read_only_snapshot(self) -> object: ...

    def registered_states(self) -> Sequence[object]: ...

    def endpoint_value(self, state_id: str, depth: int) -> float: ...

    def exact_action_outcomes(self, state_id: str, action_index: int) -> object: ...

    def sample_rollout(self, state_id: str, horizon: int, seed: int) -> object: ...


class _SealedTheoryBackend(TheoryBridgeBackend):
    def __init__(self, session: _FullTheorySession) -> None:
        self._session = session

    def identity_bundle(self) -> Mapping[str, object]:
        value = self._session.identity_bundle()
        if not isinstance(value, Mapping):
            raise TheoryBridgeError("Full backend returned an invalid identity bundle.")
        return copy.deepcopy(dict(value))

    def read_only_snapshot(self) -> ReadOnlySnapshot:
        value = self._session.read_only_snapshot()
        if isinstance(value, ReadOnlySnapshot):
            return value
        required = (
            "checkpoint_sha256",
            "model_state_sha256",
            "training_state_sha256",
            "recurrent_transition_sha256",
            "snapshot_kind",
            "environment_interactions",
        )
        if any(not hasattr(value, field) for field in required):
            raise TheoryBridgeError(
                "Full backend returned an invalid read-only snapshot."
            )
        return ReadOnlySnapshot(
            checkpoint_sha256=str(value.checkpoint_sha256),
            model_state_sha256=str(value.model_state_sha256),
            training_state_sha256=str(value.training_state_sha256),
            recurrent_transition_sha256=str(value.recurrent_transition_sha256),
            snapshot_kind=str(value.snapshot_kind),
            environment_interactions=int(value.environment_interactions),
        )

    def registered_states(self) -> Sequence[TheoryState]:
        raw_states = self._session.registered_states()
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
            raise TheoryBridgeError("Full backend returned an invalid state sequence.")
        states: list[TheoryState] = []
        for value in raw_states:
            if isinstance(value, TheoryState):
                states.append(value)
                continue
            required = (
                "state_id",
                "record_index",
                "dataset_record_sha256",
                "registered_state_sha256",
                "action_mask",
                "current_probabilities",
                "candidate_probabilities",
                "deployed_probabilities",
            )
            if any(not hasattr(value, field) for field in required):
                raise TheoryBridgeError(
                    "Full backend returned an invalid theory state."
                )
            states.append(
                TheoryState(
                    state_id=str(value.state_id),
                    record_index=int(value.record_index),
                    dataset_record_sha256=str(value.dataset_record_sha256),
                    registered_state_sha256=str(value.registered_state_sha256),
                    action_mask=tuple(bool(item) for item in value.action_mask),
                    current_probabilities=tuple(
                        float(item) for item in value.current_probabilities
                    ),
                    candidate_probabilities=tuple(
                        float(item) for item in value.candidate_probabilities
                    ),
                    deployed_probabilities=tuple(
                        float(item) for item in value.deployed_probabilities
                    ),
                )
            )
        return tuple(states)

    def endpoint_value(self, state_id: str, depth: int) -> float:
        return float(self._session.endpoint_value(state_id, depth))

    def exact_action_outcomes(
        self,
        state_id: str,
        action_index: int,
    ) -> Sequence[TheoryOutcome]:
        value = self._session.exact_action_outcomes(state_id, action_index)
        raw_outcomes = (
            value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else (value,)
        )
        outcomes: list[TheoryOutcome] = []
        for raw in raw_outcomes:
            if isinstance(raw, TheoryOutcome):
                outcomes.append(raw)
                continue
            required = ("probability", "reward", "terminal", "next_state_id")
            if any(not hasattr(raw, field) for field in required):
                raise TheoryBridgeError(
                    "Full backend returned an invalid exact outcome."
                )
            outcomes.append(
                TheoryOutcome(
                    probability=float(raw.probability),
                    reward=float(raw.reward),
                    terminal=bool(raw.terminal),
                    next_state_id=(
                        None
                        if bool(raw.terminal) or raw.next_state_id is None
                        else str(raw.next_state_id)
                    ),
                )
            )
        return tuple(outcomes)

    def sample_rollout(
        self,
        state_id: str,
        horizon: int,
        seed: int,
    ) -> TheoryRollout:
        value = self._session.sample_rollout(state_id, horizon, seed)
        if isinstance(value, TheoryRollout):
            return value
        required = (
            "rewards",
            "terminal",
            "bootstrap_state_id",
            "trajectory_sha256",
        )
        if any(not hasattr(value, field) for field in required):
            raise TheoryBridgeError("Full backend returned an invalid CRN rollout.")
        return TheoryRollout(
            rewards=tuple(float(item) for item in value.rewards),
            terminal=bool(value.terminal),
            bootstrap_state_id=(
                None
                if value.bootstrap_state_id is None
                else str(value.bootstrap_state_id)
            ),
            trajectory_sha256=str(value.trajectory_sha256),
        )


def _registered_checkpoint_snapshot_kind(
    registered_run: object,
    environment_interactions: object,
) -> str:
    checkpoints = getattr(registered_run, "interaction_checkpoints", None)
    final_interactions = getattr(
        registered_run,
        "final_environment_interactions",
        None,
    )
    if (
        not isinstance(checkpoints, tuple)
        or not checkpoints
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in checkpoints
        )
        or isinstance(final_interactions, bool)
        or not isinstance(final_interactions, int)
        or final_interactions <= 0
        or checkpoints[-1] != final_interactions
        or isinstance(environment_interactions, bool)
        or not isinstance(environment_interactions, int)
        or environment_interactions not in checkpoints
    ):
        raise TheoryBridgeError(
            "Theory checkpoint progress differs from the registered run schedule."
        )
    return (
        "interaction_matched"
        if environment_interactions == final_interactions
        else "scheduled"
    )


def create_theory_bridge_backend(
    request: Mapping[str, object],
    checkpoint_path: Path,
    inputs: TheoryBackendInputs,
) -> TheoryBridgeBackend:
    """Strictly reconstruct one disposable full-run theory session."""

    identity = request.get("identity")
    if not isinstance(identity, Mapping):
        raise TheoryBridgeError("Theory request is missing its identity bundle.")
    checkpoint_identity = identity.get("checkpoint")
    training_runtime = identity.get("training_runtime")
    dataset_identity = identity.get("dataset_records")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint_identity, training_runtime, dataset_identity)
    ):
        raise TheoryBridgeError("Theory request identity bundle is incomplete.")
    assert isinstance(checkpoint_identity, Mapping)
    assert isinstance(training_runtime, Mapping)
    assert isinstance(dataset_identity, Mapping)

    try:
        runtime_authorization = validate_runtime_authorization(
            inputs.runtime_authorization
        )
        runtime_authorization_digest = runtime_authorization_sha256(
            runtime_authorization
        )
    except PolicyImprovementSchemaError as exc:
        raise TheoryBridgeError(
            "Evaluator runtime authorization cannot be authenticated."
        ) from exc
    roles = runtime_authorization["roles"]
    if not isinstance(roles, list) or len(roles) != 6:
        raise TheoryBridgeError(
            "Evaluator authorization does not contain the full/theory roles."
        )
    full_role = roles[4]
    if not isinstance(full_role, Mapping):
        raise TheoryBridgeError("Evaluator authorization has no full-runtime role.")
    expected_training_runtime = {
        "role": "policy-improvement-full",
        "source_git_commit": full_role["source_git_commit"],
        "source_manifest_sha256": full_role["selected_source_manifest_sha256"],
        "runtime_sha256": full_role["runtime_sha256"],
        "runtime_profile_sha256": full_role["runtime_profile_sha256"],
        "selected_source_manifest_sha256": full_role["selected_source_manifest_sha256"],
        "runtime_authorization_sha256": runtime_authorization_digest,
        "launcher_sha256": runtime_authorization["launcher_sha256"],
    }
    if dict(training_runtime) != expected_training_runtime:
        raise TheoryBridgeError(
            "Theory request training runtime differs from evaluator authorization."
        )
    producer_source = identity.get("producer_source")
    if not isinstance(producer_source, Mapping) or dict(producer_source) != {
        "git_commit": runtime_authorization["producer_git_commit"],
        "source_manifest_sha256": runtime_authorization[
            "producer_source_manifest_sha256"
        ],
    }:
        raise TheoryBridgeError(
            "Theory request producer source differs from evaluator authorization."
        )

    try:
        registered_run = load_registered_full_run(
            project_root=inputs.project_root,
            protocol_path=inputs.protocol_path,
            registry_path=inputs.registry_path,
            amendment_paths=inputs.amendment_paths,
            evidence_root=inputs.evidence_root,
            dataset_root=inputs.dataset_root,
            row_id=inputs.row_id,
            runtime_authorization_sha256=runtime_authorization_digest,
            environment={FULL_EXECUTION_ENV: FULL_EXECUTION_VALUE},
        )
    except FullRuntimeError as exc:
        raise TheoryBridgeError(
            "Registered theory run cannot be authenticated."
        ) from exc
    row = getattr(registered_run, "row", None)
    if (
        not isinstance(row, Mapping)
        or any(
            (
                row.get(field) != request.get(field)
                for field in ("run_id", "method_id", "n", "alpha")
            )
        )
        or row.get("K") != request.get("bellman_horizon")
    ):
        raise TheoryBridgeError("Loaded registered run differs from theory request.")
    expected_snapshot_kind = _registered_checkpoint_snapshot_kind(
        registered_run,
        checkpoint_identity.get("environment_interactions"),
    )
    if checkpoint_identity.get("snapshot_kind") != expected_snapshot_kind:
        raise TheoryBridgeError(
            "Theory checkpoint snapshot kind differs from the registered run schedule."
        )
    try:
        resolved_checkpoint = resolve_authenticated_full_checkpoint(
            registered_run,
            checkpoint_environment_interactions=int(
                checkpoint_identity["environment_interactions"]
            ),
            runtime_authorization=runtime_authorization,
        )
    except (FullRuntimeError, ValueError, TypeError) as exc:
        raise TheoryBridgeError(
            "Theory checkpoint is not bound to a complete authenticated full run."
        ) from exc
    try:
        if checkpoint_path != resolved_checkpoint.path or dict(checkpoint_identity) != {
            "sha256": resolved_checkpoint.sha256,
            "size_bytes": resolved_checkpoint.size_bytes,
            "snapshot_kind": resolved_checkpoint.snapshot_kind,
            "environment_interactions": (resolved_checkpoint.environment_interactions),
        }:
            raise TheoryBridgeError(
                "Theory checkpoint path or identity differs from published evidence."
            )
        config_identity = identity.get("config")
        amendment_history = getattr(registered_run, "amendment_history", None)
        protocol = getattr(registered_run, "protocol", None)
        if (
            not isinstance(config_identity, Mapping)
            or not isinstance(amendment_history, tuple)
            or not isinstance(protocol, Mapping)
        ):
            raise TheoryBridgeError(
                "Registered run omits config or amendment identity."
            )
        method_registrations = protocol.get("methods")
        if not isinstance(method_registrations, list):
            raise TheoryBridgeError("Registered protocol omits method registrations.")
        matching_methods = [
            method
            for method in method_registrations
            if isinstance(method, Mapping)
            and method.get("id") == row.get("base_method_id")
        ]
        if len(matching_methods) != 1:
            raise TheoryBridgeError(
                "Registered method config does not resolve exactly."
            )
        method_registration = matching_methods[0]
        project_root = getattr(registered_run, "project_root", inputs.project_root)
        try:
            base_configs = load_registered_base_configs(protocol, project_root)
        except Exception as exc:
            raise TheoryBridgeError(
                "Registered method config cannot be authenticated."
            ) from exc
        base_method_id = row.get("base_method_id")
        base_config = base_configs.get(base_method_id)
        override = row.get("config_override")
        if not isinstance(base_config, Mapping) or not isinstance(override, Mapping):
            raise TheoryBridgeError("Registered effective config is incomplete.")
        effective_config = dict(base_config)
        effective_config.update(override)
        configured_gamma = effective_config.get("gamma")
        if (
            isinstance(configured_gamma, bool)
            or not isinstance(configured_gamma, (int, float))
            or float(configured_gamma) != float(request.get("gamma", -1.0))
        ):
            raise TheoryBridgeError(
                "Theory gamma differs from the authenticated effective config."
            )
        theory_amendment_sha256 = (
            hashlib.sha256(canonical_json_bytes(amendment_history[0])).hexdigest()
            if amendment_history
            else None
        )
        if (
            identity.get("protocol_sha256")
            != getattr(registered_run, "protocol_sha256", None)
            or identity.get("registry_row_sha256")
            != getattr(registered_run, "registry_row_sha256", None)
            or identity.get("theory_amendment_sha256") != theory_amendment_sha256
            or config_identity.get("file_sha256")
            != method_registration.get("config_sha256")
            or config_identity.get("base_canonical_sha256")
            != row.get("base_config_canonical_sha256")
            or config_identity.get("effective_config_sha256")
            != row.get("expected_effective_config_sha256")
            or dataset_identity.get("split") != row.get("evaluation_split")
        ):
            raise TheoryBridgeError("Registered run and theory identity bundle differ.")
        record_count = int(dataset_identity["record_count"])
        session_request = {
            "identity": copy.deepcopy(dict(identity)),
            "registered_run": registered_run,
            "registered_state_indices": tuple(range(record_count)),
        }
        full_backend = importlib.import_module("policy_improvement_full_backend")
        open_session = getattr(full_backend, "open_theory_bridge_session", None)
        if not callable(open_session):
            raise TheoryBridgeError(
                "Full backend does not expose the sealed theory API."
            )
        training_module = importlib.import_module("upi_trm_train")
        session = open_session(
            session_request,
            resolved_checkpoint.path,
            expected_checkpoint_sha256=resolved_checkpoint.sha256,
            sealed_checkpoint_descriptor=resolved_checkpoint.sealed_descriptor,
            authenticated_model_state_sha256=(resolved_checkpoint.model_state_sha256),
            authenticated_role_state_sha256s=(resolved_checkpoint.role_state_sha256s),
            authenticated_validation_sha256=(resolved_checkpoint.validation_sha256),
            runtime_identity=copy.deepcopy(dict(training_runtime)),
            training_module=training_module,
        )
        return _SealedTheoryBackend(cast(_FullTheorySession, session))
    finally:
        os.close(resolved_checkpoint.sealed_descriptor)
