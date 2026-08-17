#!/usr/bin/env fbpython
"""Read-only finite-batch evaluator for the policy-improvement theory bridge.

The authenticated entrypoint injects a sealed backend.  This module owns the
registered numerical estimators, identity checks, stable checkpoint hashing,
and read-only verification.  The backend may expose model and environment
operations, but it cannot choose metric definitions or Monte Carlo seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Protocol

from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
)
from scripts.policy_improvement_theory_schema import (
    THEORY_METRIC_IDS,
    THEORY_RESULT_SCHEMA_NAME,
    THEORY_SCHEMA_VERSION,
    TheoryBridgeSchemaError,
    theory_document_sha256,
    validate_identity_bundle,
    validate_request_against_amendment,
    validate_theory_result,
)


_PROBABILITY_TOLERANCE = 1e-10


class TheoryBridgeError(RuntimeError):
    """Raised when theory evaluation cannot preserve its registered contract."""


@dataclass(frozen=True)
class TheoryState:
    """One registered evaluator state and its three action distributions."""

    state_id: str
    record_index: int
    dataset_record_sha256: str
    registered_state_sha256: str
    action_mask: tuple[bool, ...]
    current_probabilities: tuple[float, ...]
    candidate_probabilities: tuple[float, ...]
    deployed_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TheoryOutcome:
    """One exact transition outcome for a fixed state and action."""

    probability: float
    reward: float
    terminal: bool
    next_state_id: str | None


@dataclass(frozen=True)
class TheoryRollout:
    """One CRN rollout prefix, independent of endpoint evaluation depth."""

    rewards: tuple[float, ...]
    terminal: bool
    bootstrap_state_id: str | None
    trajectory_sha256: str


@dataclass(frozen=True)
class ReadOnlySnapshot:
    """Content identities sampled before and after evaluation."""

    checkpoint_sha256: str
    model_state_sha256: str
    training_state_sha256: str
    recurrent_transition_sha256: str
    snapshot_kind: str
    environment_interactions: int


class TheoryBridgeBackend(Protocol):
    """Sealed model/environment adapter injected by the packaged entrypoint."""

    def identity_bundle(self) -> Mapping[str, object]:
        """Return identities independently reconstructed from the checkpoint."""

        ...

    def read_only_snapshot(self) -> ReadOnlySnapshot:
        """Hash checkpoint-visible model, training, and transition state."""

        ...

    def registered_states(self) -> Sequence[TheoryState]:
        """Return only the pre-registered train or validation records."""

        ...

    def endpoint_value(self, state_id: str, depth: int) -> float:
        """Evaluate U_depth without changing the deployed transition."""

        ...

    def exact_action_outcomes(
        self,
        state_id: str,
        action_index: int,
    ) -> Sequence[TheoryOutcome]:
        """Enumerate the exact masked one-step transition distribution."""

        ...

    def sample_rollout(
        self,
        state_id: str,
        horizon: int,
        seed: int,
    ) -> TheoryRollout:
        """Sample one policy rollout prefix using the core-provided CRN seed."""

        ...


@dataclass(frozen=True)
class TheoryBackendInputs:
    """Authenticated paths needed to reconstruct the registered full run."""

    project_root: Path
    protocol_path: Path
    registry_path: Path
    amendment_paths: tuple[Path, ...]
    evidence_root: Path
    dataset_root: Path
    row_id: str
    runtime_authorization: Mapping[str, object]


TheoryBackendFactory = Callable[
    [Mapping[str, object], Path, TheoryBackendInputs],
    TheoryBridgeBackend,
]


@dataclass(frozen=True)
class _StableFile:
    sha256: str
    size_bytes: int
    identity: tuple[int, int, int, int, int, int, int]


def _stable_regular_file(
    path: Path, *, retain_payload: bool
) -> tuple[_StableFile, bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        before_path = path.lstat()
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before_path.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise TheoryBridgeError(
                f"Theory input {path} is a symlink, hard-link alias, or wrong type."
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if retain_payload:
                chunks.append(block)
        after = os.fstat(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise TheoryBridgeError(
            f"Theory input {path} cannot be authenticated."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before_path) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(after_path)
    ):
        raise TheoryBridgeError(f"Theory input {path} changed while it was read.")
    stable = _StableFile(
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
        identity=identity(before),
    )
    return stable, b"".join(chunks) if retain_payload else None


def _absolute_canonical_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise TheoryBridgeError(f"{label} must be an absolute canonical path.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TheoryBridgeError(f"{label} is unavailable.") from exc
    if resolved != path:
        raise TheoryBridgeError(f"{label} must not traverse a symlink or alias.")
    return resolved


def _absolute_canonical_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise TheoryBridgeError(f"{label} must be an absolute canonical path.")
    try:
        resolved = path.resolve(strict=True)
        status = path.lstat()
    except OSError as exc:
        raise TheoryBridgeError(f"{label} is unavailable.") from exc
    if (
        resolved != path
        or stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise TheoryBridgeError(f"{label} must be one unaliased directory.")
    return resolved


def _load_stable_json(path: Path) -> object:
    _, payload = _stable_regular_file(path, retain_payload=True)
    assert payload is not None
    try:
        return load_strict_json_bytes(payload)
    except Exception as exc:
        raise TheoryBridgeError(f"Theory JSON {path} is invalid.") from exc


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TheoryBridgeError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise TheoryBridgeError(f"{label} must be finite.")
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TheoryBridgeError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _probabilities(
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    label: str,
) -> tuple[float, ...]:
    if len(values) != len(mask) or not values:
        raise TheoryBridgeError(f"{label} has the wrong action inventory.")
    checked = tuple(
        _finite(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    )
    if any(value < 0.0 for value in checked):
        raise TheoryBridgeError(f"{label} contains a negative probability.")
    for index, (valid, probability) in enumerate(zip(mask, checked)):
        if not valid and abs(probability) > _PROBABILITY_TOLERANCE:
            raise TheoryBridgeError(
                f"{label}[{index}] assigns mass to a masked action."
            )
    valid_mass = sum(probability for valid, probability in zip(mask, checked) if valid)
    if not math.isclose(valid_mass, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise TheoryBridgeError(f"{label} does not sum to one on valid actions.")
    return tuple(
        probability / valid_mass if valid else 0.0
        for valid, probability in zip(mask, checked)
    )


def _state_identity(state: TheoryState) -> str:
    payload = {
        "state_id": state.state_id,
        "record_index": state.record_index,
        "dataset_record_sha256": state.dataset_record_sha256,
        "action_mask": list(state.action_mask),
        "current_probabilities": list(state.current_probabilities),
        "candidate_probabilities": list(state.candidate_probabilities),
        "deployed_probabilities": list(state.deployed_probabilities),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_states(
    states: Sequence[TheoryState],
    identity: Mapping[str, object],
) -> list[TheoryState]:
    checked = list(states)
    dataset = identity["dataset_records"]
    assert isinstance(dataset, Mapping)
    expected_count = int(dataset["record_count"])
    if len(checked) != expected_count:
        raise TheoryBridgeError("Backend returned the wrong registered-state count.")
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    record_pairs: list[dict[str, object]] = []
    for offset, state in enumerate(checked):
        if not isinstance(state, TheoryState):
            raise TheoryBridgeError("Backend returned a non-TheoryState value.")
        if not state.state_id or not state.state_id.isascii():
            raise TheoryBridgeError("Theory state ID must be nonempty ASCII text.")
        if state.state_id in seen_ids or state.record_index in seen_indices:
            raise TheoryBridgeError(
                "Theory state IDs and record indices must be unique."
            )
        seen_ids.add(state.state_id)
        seen_indices.add(state.record_index)
        if state.record_index < 0:
            raise TheoryBridgeError("Theory record indices must be nonnegative.")
        _sha256(state.dataset_record_sha256, label="dataset record SHA-256")
        _sha256(state.registered_state_sha256, label="registered state SHA-256")
        if not state.action_mask or not any(state.action_mask):
            raise TheoryBridgeError("Every theory state needs one valid action.")
        if state.registered_state_sha256 != _state_identity(state):
            raise TheoryBridgeError("Theory state identity differs from its payload.")
        _probabilities(
            state.current_probabilities,
            state.action_mask,
            label=f"state[{offset}].current_probabilities",
        )
        _probabilities(
            state.candidate_probabilities,
            state.action_mask,
            label=f"state[{offset}].candidate_probabilities",
        )
        _probabilities(
            state.deployed_probabilities,
            state.action_mask,
            label=f"state[{offset}].deployed_probabilities",
        )
        record_pairs.append(
            {
                "record_index": state.record_index,
                "dataset_record_sha256": state.dataset_record_sha256,
            }
        )
    indices_digest = hashlib.sha256(
        canonical_json_bytes([state.record_index for state in checked])
    ).hexdigest()
    records_digest = hashlib.sha256(canonical_json_bytes(record_pairs)).hexdigest()
    if (
        indices_digest != dataset["selected_record_indices_sha256"]
        or records_digest != dataset["selected_records_sha256"]
    ):
        raise TheoryBridgeError(
            "Backend returned mixed or reordered dataset identities."
        )
    return checked


def _validate_outcomes(
    outcomes: Sequence[TheoryOutcome],
    *,
    state_id: str,
    action_index: int,
) -> list[TheoryOutcome]:
    checked = list(outcomes)
    if not checked:
        raise TheoryBridgeError("Exact action enumeration returned no outcomes.")
    probability_sum = 0.0
    for offset, outcome in enumerate(checked):
        if not isinstance(outcome, TheoryOutcome):
            raise TheoryBridgeError("Exact action enumeration returned a wrong type.")
        probability = _finite(
            outcome.probability,
            label=f"outcomes[{state_id},{action_index},{offset}].probability",
        )
        if probability < 0.0:
            raise TheoryBridgeError("Outcome probabilities must be nonnegative.")
        probability_sum += probability
        _finite(
            outcome.reward,
            label=f"outcomes[{state_id},{action_index},{offset}].reward",
        )
        if outcome.terminal != (outcome.next_state_id is None):
            raise TheoryBridgeError(
                "Terminal outcomes must omit, and nonterminal outcomes must name, a successor."
            )
        if outcome.next_state_id is not None and (
            not outcome.next_state_id or not outcome.next_state_id.isascii()
        ):
            raise TheoryBridgeError("Successor state IDs must be nonempty ASCII text.")
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise TheoryBridgeError("Exact outcome probabilities do not sum to one.")
    return checked


def _exact_action_values(
    *,
    backend: TheoryBridgeBackend,
    state: TheoryState,
    n: int,
    m: int,
    gamma: float,
) -> tuple[list[float], list[float]]:
    q_n = [0.0] * len(state.action_mask)
    q_m = [0.0] * len(state.action_mask)
    for action_index, valid in enumerate(state.action_mask):
        if not valid:
            continue
        outcomes = _validate_outcomes(
            backend.exact_action_outcomes(state.state_id, action_index),
            state_id=state.state_id,
            action_index=action_index,
        )
        for outcome in outcomes:
            n_value = 0.0
            m_value = 0.0
            if not outcome.terminal:
                assert outcome.next_state_id is not None
                n_value = _finite(
                    backend.endpoint_value(outcome.next_state_id, n),
                    label="successor U_n",
                )
                m_value = _finite(
                    backend.endpoint_value(outcome.next_state_id, m),
                    label="successor U_m",
                )
            q_n[action_index] += outcome.probability * (
                outcome.reward + gamma * n_value
            )
            q_m[action_index] += outcome.probability * (
                outcome.reward + gamma * m_value
            )
    return q_n, q_m


def _crn_seed(
    *,
    base_seed: int,
    record_index: int,
    state_id: str,
    registered_state_sha256: str,
    repeat_index: int,
) -> int:
    payload = canonical_json_bytes(
        [
            base_seed,
            record_index,
            state_id,
            registered_state_sha256,
            repeat_index,
        ]
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rollout_return(
    *,
    rollout: TheoryRollout,
    backend: TheoryBridgeBackend,
    depth: int,
    horizon: int,
    gamma: float,
) -> float:
    if not isinstance(rollout, TheoryRollout):
        raise TheoryBridgeError("Backend returned a non-TheoryRollout value.")
    _sha256(rollout.trajectory_sha256, label="rollout trajectory SHA-256")
    if not 1 <= len(rollout.rewards) <= horizon:
        raise TheoryBridgeError("Rollout length is outside the registered horizon.")
    if rollout.terminal:
        if rollout.bootstrap_state_id is not None:
            raise TheoryBridgeError("Terminal rollout cannot carry a bootstrap state.")
    elif len(rollout.rewards) != horizon or not rollout.bootstrap_state_id:
        raise TheoryBridgeError(
            "Nonterminal rollout must reach the horizon and name its bootstrap state."
        )
    discounted = 0.0
    discount = 1.0
    for index, reward in enumerate(rollout.rewards):
        discounted += discount * _finite(reward, label=f"rollout.rewards[{index}]")
        discount *= gamma
    if not rollout.terminal:
        assert rollout.bootstrap_state_id is not None
        discounted += discount * _finite(
            backend.endpoint_value(rollout.bootstrap_state_id, depth),
            label="rollout bootstrap endpoint value",
        )
    return discounted


def _summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        raise TheoryBridgeError("Cannot summarize an empty theory metric.")
    checked = [_finite(value, label="metric value") for value in values]
    if any(value < 0.0 for value in checked):
        raise TheoryBridgeError("Theory summaries require nonnegative metrics.")
    return {
        "count": len(checked),
        "mean": fmean(checked),
        "minimum": min(checked),
        "maximum": max(checked),
    }


def _snapshot(snapshot: object, *, label: str) -> ReadOnlySnapshot:
    if not isinstance(snapshot, ReadOnlySnapshot):
        raise TheoryBridgeError(f"{label} is not a ReadOnlySnapshot.")
    _sha256(snapshot.checkpoint_sha256, label=f"{label}.checkpoint_sha256")
    _sha256(snapshot.model_state_sha256, label=f"{label}.model_state_sha256")
    _sha256(snapshot.training_state_sha256, label=f"{label}.training_state_sha256")
    _sha256(
        snapshot.recurrent_transition_sha256,
        label=f"{label}.recurrent_transition_sha256",
    )
    if snapshot.snapshot_kind not in {"scheduled", "interaction_matched"}:
        raise TheoryBridgeError(
            f"{label}.snapshot_kind must be scheduled or interaction_matched."
        )
    if (
        isinstance(snapshot.environment_interactions, bool)
        or not isinstance(snapshot.environment_interactions, int)
        or snapshot.environment_interactions <= 0
    ):
        raise TheoryBridgeError(f"{label}.environment_interactions must be positive.")
    return snapshot


def evaluate_theory_bridge(
    *,
    request_document: object,
    amendment_document: object,
    checkpoint_path: str | Path,
    backend: TheoryBridgeBackend,
) -> dict[str, Any]:
    """Evaluate one registered checkpoint without mutating model or training state."""

    try:
        request, _ = validate_request_against_amendment(
            request_document,
            amendment_document,
        )
    except TheoryBridgeSchemaError as exc:
        raise TheoryBridgeError(str(exc)) from exc
    identity = request["identity"]
    assert isinstance(identity, Mapping)
    try:
        backend_identity = validate_identity_bundle(backend.identity_bundle())
    except TheoryBridgeSchemaError as exc:
        raise TheoryBridgeError("Backend identity bundle is invalid.") from exc
    if canonical_json_bytes(backend_identity) != canonical_json_bytes(identity):
        raise TheoryBridgeError("Backend and request identities are missing or mixed.")

    checkpoint = _absolute_canonical_file(checkpoint_path, label="checkpoint")
    checkpoint_before, _ = _stable_regular_file(checkpoint, retain_payload=False)
    expected_checkpoint = identity["checkpoint"]
    assert isinstance(expected_checkpoint, Mapping)
    if (
        checkpoint_before.sha256 != expected_checkpoint["sha256"]
        or checkpoint_before.size_bytes != expected_checkpoint["size_bytes"]
    ):
        raise TheoryBridgeError("Checkpoint file differs from the registered identity.")

    before = _snapshot(backend.read_only_snapshot(), label="before snapshot")
    model_identity = identity["model"]
    assert isinstance(model_identity, Mapping)
    if (
        before.checkpoint_sha256 != checkpoint_before.sha256
        or before.model_state_sha256 != model_identity["model_sha256"]
        or before.recurrent_transition_sha256
        != model_identity["recurrent_transition_sha256"]
        or before.snapshot_kind != expected_checkpoint["snapshot_kind"]
        or before.environment_interactions
        != expected_checkpoint["environment_interactions"]
        or before.environment_interactions
        != request["checkpoint_environment_interactions"]
    ):
        raise TheoryBridgeError(
            "Backend snapshot differs from checkpoint/model identity."
        )

    states = _validate_states(backend.registered_states(), identity)
    n = int(request["n"])
    m = int(request["reference_depth_m"])
    horizon = int(request["bellman_horizon"])
    gamma = float(request["gamma"])
    alpha = float(request["alpha"])
    estimator = request["bellman_estimator"]
    assert isinstance(estimator, Mapping)
    state_rows: list[dict[str, object]] = []
    uncertainty_operator: list[float] = []
    uncertainty_b: list[float] = []

    for state in states:
        current = _probabilities(
            state.current_probabilities,
            state.action_mask,
            label=f"{state.state_id}.current",
        )
        candidate = _probabilities(
            state.candidate_probabilities,
            state.action_mask,
            label=f"{state.state_id}.candidate",
        )
        deployed = _probabilities(
            state.deployed_probabilities,
            state.action_mask,
            label=f"{state.state_id}.deployed",
        )
        value_n = _finite(
            backend.endpoint_value(state.state_id, n),
            label=f"{state.state_id}.value_n",
        )
        value_m = _finite(
            backend.endpoint_value(state.state_id, m),
            label=f"{state.state_id}.value_m",
        )
        d_nm = abs(value_n - value_m)
        q_n, q_m = _exact_action_values(
            backend=backend,
            state=state,
            n=n,
            m=m,
            gamma=gamma,
        )
        exact_operator_n = sum(
            probability * q_n[action]
            for action, probability in enumerate(current)
            if state.action_mask[action]
        )
        centering_defect = abs(
            sum(
                probability * (q_n[action] - exact_operator_n)
                for action, probability in enumerate(current)
                if state.action_mask[action]
            )
        )

        if horizon == 1:
            operator_m = sum(
                probability * q_m[action]
                for action, probability in enumerate(current)
                if state.action_mask[action]
            )
            operator_se = 0.0
        else:
            rollout_returns_m: list[float] = []
            rollout_count = int(estimator["rollout_count"])
            base_seed = int(estimator["base_seed"])
            for repeat_index in range(rollout_count):
                seed = _crn_seed(
                    base_seed=base_seed,
                    record_index=state.record_index,
                    state_id=state.state_id,
                    registered_state_sha256=state.registered_state_sha256,
                    repeat_index=repeat_index,
                )
                rollout = backend.sample_rollout(state.state_id, horizon, seed)
                # The path is sampled once.  Only this endpoint call depends on m.
                rollout_returns_m.append(
                    _rollout_return(
                        rollout=rollout,
                        backend=backend,
                        depth=m,
                        horizon=horizon,
                        gamma=gamma,
                    )
                )
            operator_m = fmean(rollout_returns_m)
            operator_se = stdev(rollout_returns_m) / math.sqrt(rollout_count)

        residual = abs(value_m - operator_m)
        denominator = 1.0 - gamma**horizon
        b_nm = d_nm + residual / denominator
        b_se = operator_se / denominator
        tau = 0.5 * sum(
            abs(candidate_probability - current_probability)
            for candidate_probability, current_probability in zip(candidate, current)
        )
        exact_mixture = tuple(
            (1.0 - alpha) * current_probability + alpha * candidate_probability
            for current_probability, candidate_probability in zip(current, candidate)
        )
        deployment_tv = 0.5 * sum(
            abs(mixture_probability - deployed_probability)
            for mixture_probability, deployed_probability in zip(
                exact_mixture, deployed
            )
        )
        state_rows.append(
            {
                "state_id": state.state_id,
                "record_index": state.record_index,
                "dataset_record_sha256": state.dataset_record_sha256,
                "registered_state_sha256": state.registered_state_sha256,
                "value_n": value_n,
                "value_m": value_m,
                "D_nm": d_nm,
                "bellman_operator_mean_m": operator_m,
                "bellman_operator_standard_error_m": operator_se,
                "bellman_residual_proxy": residual,
                "B_nm": b_nm,
                "B_nm_standard_error": b_se,
                "policy_overlap_tau": tau,
                "centering_defect": centering_defect,
                "deployment_tv": deployment_tv,
            }
        )
        uncertainty_operator.append(operator_se)
        uncertainty_b.append(b_se)

    after = _snapshot(backend.read_only_snapshot(), label="after snapshot")
    checkpoint_after, _ = _stable_regular_file(checkpoint, retain_payload=False)
    if checkpoint_after.identity != checkpoint_before.identity:
        raise TheoryBridgeError("Checkpoint file metadata changed during evaluation.")
    checkpoint_unchanged = checkpoint_after.sha256 == checkpoint_before.sha256
    model_unchanged = after.model_state_sha256 == before.model_state_sha256
    training_unchanged = after.training_state_sha256 == before.training_state_sha256
    transition_unchanged = (
        after.recurrent_transition_sha256 == before.recurrent_transition_sha256
    )
    if not (
        checkpoint_unchanged
        and model_unchanged
        and training_unchanged
        and transition_unchanged
        and after.checkpoint_sha256 == before.checkpoint_sha256
        and after.snapshot_kind == before.snapshot_kind
        and after.environment_interactions == before.environment_interactions
    ):
        raise TheoryBridgeError(
            "Theory evaluation mutated checkpoint or training state."
        )

    metrics = {
        metric: _summary([float(row[metric]) for row in state_rows])
        for metric in THEORY_METRIC_IDS
    }
    result: dict[str, Any] = {
        "schema_name": THEORY_RESULT_SCHEMA_NAME,
        "schema_version": THEORY_SCHEMA_VERSION,
        "status": "complete",
        "scope": "finite_batch_proxy_only",
        "uniform_certificate": False,
        "evaluation_id": request["evaluation_id"],
        "run_id": request["run_id"],
        "method_id": request["method_id"],
        "checkpoint_environment_interactions": request[
            "checkpoint_environment_interactions"
        ],
        "n": n,
        "reference_depth_m": m,
        "reference_role": request["reference_role"],
        "bellman_horizon": horizon,
        "gamma": gamma,
        "alpha": alpha,
        "request_sha256": theory_document_sha256(request),
        "identity": dict(identity),
        "state_count": len(state_rows),
        "states": state_rows,
        "metrics": metrics,
        "monte_carlo_uncertainty": {
            "kind": (
                "exact_zero_monte_carlo_standard_error"
                if horizon == 1
                else "sample_standard_error_of_operator_mean"
            ),
            "operator_standard_error": _summary(uncertainty_operator),
            "B_nm_standard_error": _summary(uncertainty_b),
        },
        "persistent_semantics": {
            "applicable": request["latent_mode"] == "persistent",
            "transition_sha256": before.recurrent_transition_sha256,
            "endpoint_depth_only": True,
            "F_n_unchanged": transition_unchanged,
        },
        "read_only_verification": {
            "checkpoint_sha256_before": checkpoint_before.sha256,
            "checkpoint_sha256_after": checkpoint_after.sha256,
            "model_state_sha256_before": before.model_state_sha256,
            "model_state_sha256_after": after.model_state_sha256,
            "training_state_sha256_before": before.training_state_sha256,
            "training_state_sha256_after": after.training_state_sha256,
            "checkpoint_snapshot_kind_before": before.snapshot_kind,
            "checkpoint_snapshot_kind_after": after.snapshot_kind,
            "checkpoint_environment_interactions_before": (
                before.environment_interactions
            ),
            "checkpoint_environment_interactions_after": (
                after.environment_interactions
            ),
            "checkpoint_unchanged": checkpoint_unchanged,
            "model_state_unchanged": model_unchanged,
            "training_state_unchanged": training_unchanged,
        },
        "test_data_opened": False,
    }
    try:
        return validate_theory_result(result, request=request)
    except TheoryBridgeSchemaError as exc:
        raise TheoryBridgeError(
            "Computed theory result failed its strict schema."
        ) from exc


def _attestation_matches_request(
    attestation: Mapping[str, str],
    request: Mapping[str, object],
) -> None:
    expected_fields = {
        "source_git_commit",
        "source_manifest_sha256",
        "runtime_sha256",
        "runtime_profile_sha256",
        "runtime_authorization_sha256",
        "launcher_sha256",
    }
    if set(attestation) != expected_fields:
        raise TheoryBridgeError("Evaluator attestation field inventory differs.")
    identity = request["identity"]
    assert isinstance(identity, Mapping)
    source = identity["evaluator_source"]
    runtime = identity["evaluator_runtime"]
    assert isinstance(source, Mapping) and isinstance(runtime, Mapping)
    expected = {
        "source_git_commit": source["git_commit"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "runtime_profile_sha256": runtime["runtime_profile_sha256"],
        "runtime_authorization_sha256": runtime["runtime_authorization_sha256"],
        "launcher_sha256": runtime["launcher_sha256"],
    }
    if dict(attestation) != expected:
        raise TheoryBridgeError("Evaluator runtime differs from request identity.")


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: TheoryBackendFactory | None = None,
    evaluator_attestation: Mapping[str, str] | None = None,
    runtime_authorization: Mapping[str, object] | None = None,
) -> int:
    """Run only through the authenticated pre-import entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-improvement-theory-bridge-entrypoint", action="store_true"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--theory-amendment", required=True)
    parser.add_argument("--amendment", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--row-id", required=True)
    arguments = parser.parse_args(argv)
    if (
        not arguments.policy_improvement_theory_bridge_entrypoint
        or backend_factory is None
        or evaluator_attestation is None
        or runtime_authorization is None
    ):
        raise TheoryBridgeError(
            "Theory evaluation requires its authenticated packaged entrypoint."
        )
    request_path = _absolute_canonical_file(arguments.request, label="request")
    theory_amendment_path = _absolute_canonical_file(
        arguments.theory_amendment,
        label="theory amendment",
    )
    amendment_paths = tuple(
        _absolute_canonical_file(path, label=f"amendment {index}")
        for index, path in enumerate(arguments.amendment)
    )
    if not amendment_paths or amendment_paths[0] != theory_amendment_path:
        raise TheoryBridgeError(
            "The complete amendment history must begin with the theory design."
        )
    checkpoint_path = _absolute_canonical_file(arguments.checkpoint, label="checkpoint")
    backend_inputs = TheoryBackendInputs(
        project_root=_absolute_canonical_directory(
            arguments.project_root,
            label="project root",
        ),
        protocol_path=_absolute_canonical_file(arguments.protocol, label="protocol"),
        registry_path=_absolute_canonical_file(arguments.registry, label="registry"),
        amendment_paths=amendment_paths,
        evidence_root=_absolute_canonical_directory(
            arguments.evidence_root,
            label="evidence root",
        ),
        dataset_root=_absolute_canonical_directory(
            arguments.dataset_root,
            label="dataset root",
        ),
        row_id=arguments.row_id,
        runtime_authorization=copy.deepcopy(dict(runtime_authorization)),
    )
    request_document = _load_stable_json(request_path)
    amendment_document = _load_stable_json(theory_amendment_path)
    try:
        request, _ = validate_request_against_amendment(
            request_document,
            amendment_document,
        )
    except TheoryBridgeSchemaError as exc:
        raise TheoryBridgeError(str(exc)) from exc
    if request["run_id"] != backend_inputs.row_id:
        raise TheoryBridgeError("Theory request and registered row ID differ.")
    _attestation_matches_request(evaluator_attestation, request)
    backend_stdout = io.StringIO()
    with redirect_stdout(backend_stdout):
        backend = backend_factory(request, checkpoint_path, backend_inputs)
        result = evaluate_theory_bridge(
            request_document=request,
            amendment_document=amendment_document,
            checkpoint_path=checkpoint_path,
            backend=backend,
        )
    if backend_stdout.getvalue():
        raise TheoryBridgeError("Theory backend wrote unauthenticated stdout.")
    payload = canonical_json_bytes(result) + b"\n"
    offset = 0
    while offset < len(payload):
        written = os.write(1, payload[offset:])
        if written <= 0:
            raise TheoryBridgeError(
                "Canonical theory result stdout could not be written."
            )
        offset += written
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
