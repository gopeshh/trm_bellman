#!/usr/bin/env fbpython
"""Protocol-v2 read-only finite-population theory-bridge evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import math
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Protocol

from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
)
from scripts.policy_improvement_theory_schema_v2 import (
    EXACT_METHOD_IDS,
    PAIRED_RETURN_ALPHA_VALUES,
    theory_document_sha256,
    THEORY_METRIC_IDS,
    THEORY_RESULT_SCHEMA_NAME,
    THEORY_SCHEMA_VERSION,
    TheoryBridgeV2SchemaError,
    validate_identity_bundle,
    validate_request_against_amendment,
    validate_theory_result,
)


_PROBABILITY_TOLERANCE = 1e-10


class TheoryBridgeV2Error(RuntimeError):
    """Raised when v2 evaluation cannot preserve its registered contract."""


@dataclass(frozen=True)
class TheoryStateV2:
    state_id: str
    record_index: int
    dataset_record_sha256: str
    registered_state_sha256: str
    action_mask: tuple[bool, ...]
    current_probabilities: tuple[float, ...]
    candidate_probabilities: tuple[float, ...]
    deployed_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TheoryOutcomeV2:
    probability: float
    reward: float
    terminal: bool
    next_state_id: str | None


@dataclass(frozen=True)
class BellmanRolloutV2:
    rewards: tuple[float, ...]
    terminal: bool
    bootstrap_state_id: str | None
    trajectory_sha256: str


@dataclass(frozen=True)
class CurrentPolicyReturnV2:
    discounted_return: float
    environment_steps: int
    terminal: bool
    trajectory_sha256: str


@dataclass(frozen=True)
class PairedPolicyReturnV2:
    """One common-random-number current-policy/exact-mixture comparison."""

    current_policy: CurrentPolicyReturnV2
    exact_mixture: CurrentPolicyReturnV2
    common_random_numbers_sha256: str


@dataclass(frozen=True)
class TrainingAdvantageEstimatorV2:
    action_mask: tuple[bool, ...]
    current_probabilities: tuple[float, ...]
    advantages: tuple[float, ...]
    clipping_kind: str
    clip_value: float | None


@dataclass(frozen=True)
class PersistentEndpointWitnessV2:
    requested_endpoint_depth: int
    deployed_transition_depth: int
    carried_successor_latent_sha256: str
    trajectory_sha256: str
    action_probabilities_sha256: str
    recurrent_transition_sha256: str


@dataclass(frozen=True)
class ReadOnlySnapshotV2:
    checkpoint_sha256: str
    model_state_sha256: str
    mutable_state_sha256s: Mapping[str, str]
    recurrent_transition_sha256: str
    snapshot_kind: str
    environment_interactions: int


@dataclass(frozen=True)
class TheoryBackendInputsV2:
    project_root: Path
    protocol_path: Path
    registry_path: Path
    amendment_paths: tuple[Path, ...]
    evidence_root: Path
    dataset_root: Path
    row_id: str
    runtime_authorization: Mapping[str, object]


class TheoryBridgeV2Backend(Protocol):
    def begin_read_only_evaluation(self) -> None: ...

    def end_read_only_evaluation(self) -> None: ...

    def identity_bundle(self) -> Mapping[str, object]: ...

    def read_only_snapshot(self) -> ReadOnlySnapshotV2: ...

    def registered_states(self) -> Sequence[TheoryStateV2]: ...

    def normalization_diagnostic(self) -> Mapping[str, object]: ...

    def endpoint_value(self, state_id: str, depth: int) -> float: ...

    def exact_action_outcomes(
        self,
        state_id: str,
        action_index: int,
    ) -> Sequence[TheoryOutcomeV2]: ...

    def sample_bellman_rollout(
        self,
        state_id: str,
        horizon: int,
        seed: int,
    ) -> BellmanRolloutV2: ...

    def training_advantage_estimator(
        self,
        state_id: str,
    ) -> TrainingAdvantageEstimatorV2: ...

    def persistent_endpoint_witness(
        self,
        state_id: str,
        endpoint_depth: int,
    ) -> PersistentEndpointWitnessV2: ...

    def sample_current_policy_return(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
    ) -> CurrentPolicyReturnV2: ...

    def sample_paired_policy_returns(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
        alpha: float,
    ) -> PairedPolicyReturnV2: ...

    def close(self) -> None: ...


TheoryBackendFactoryV2 = Callable[
    [Mapping[str, object], Path, TheoryBackendInputsV2],
    TheoryBridgeV2Backend,
]


@dataclass(frozen=True)
class _StableFile:
    sha256: str
    size_bytes: int
    identity: tuple[int, int, int, int, int, int, int]


def _stable_regular_file(path: Path) -> _StableFile:
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
            raise TheoryBridgeV2Error(
                "Theory checkpoint is a link alias or has the wrong type."
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise TheoryBridgeV2Error("Theory checkpoint cannot be authenticated.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
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
        raise TheoryBridgeV2Error("Theory checkpoint changed while it was read.")
    return _StableFile(
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
        identity=identity(before),
    )


def _canonical_file(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise TheoryBridgeV2Error("Checkpoint path must be absolute and canonical.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TheoryBridgeV2Error("Checkpoint path is unavailable.") from exc
    if resolved != path:
        raise TheoryBridgeV2Error("Checkpoint path must not traverse an alias.")
    return resolved


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TheoryBridgeV2Error(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise TheoryBridgeV2Error(f"{label} must be finite.")
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TheoryBridgeV2Error(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _probabilities(
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    label: str,
) -> tuple[float, ...]:
    if not values or len(values) != len(mask):
        raise TheoryBridgeV2Error(f"{label} has the wrong action inventory.")
    checked = tuple(
        _finite(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    )
    if any(value < 0.0 for value in checked):
        raise TheoryBridgeV2Error(f"{label} contains a negative probability.")
    for index, (valid, probability) in enumerate(zip(mask, checked)):
        if not valid and abs(probability) > _PROBABILITY_TOLERANCE:
            raise TheoryBridgeV2Error(
                f"{label}[{index}] assigns mass to a masked action."
            )
    valid_mass = sum(value for valid, value in zip(mask, checked) if valid)
    if not math.isclose(valid_mass, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise TheoryBridgeV2Error(f"{label} does not sum to one on valid actions.")
    return tuple(
        value / valid_mass if valid else 0.0 for valid, value in zip(mask, checked)
    )


def state_identity(state: TheoryStateV2) -> str:
    return theory_document_sha256(
        {
            "state_id": state.state_id,
            "record_index": state.record_index,
            "dataset_record_sha256": state.dataset_record_sha256,
            "action_mask": list(state.action_mask),
            "current_probabilities": list(state.current_probabilities),
            "candidate_probabilities": list(state.candidate_probabilities),
            "deployed_probabilities": list(state.deployed_probabilities),
        }
    )


def _validate_states(
    values: Sequence[TheoryStateV2],
    identity: Mapping[str, object],
) -> list[TheoryStateV2]:
    states = list(values)
    dataset = identity["dataset_records"]
    assert isinstance(dataset, Mapping)
    expected_indices = dataset["selected_record_indices"]
    expected_records = dataset["selected_records"]
    assert isinstance(expected_indices, list) and isinstance(expected_records, list)
    if len(states) != len(expected_indices):
        raise TheoryBridgeV2Error("Backend returned the wrong registered-state count.")
    seen_ids: set[str] = set()
    for offset, state in enumerate(states):
        if not isinstance(state, TheoryStateV2):
            raise TheoryBridgeV2Error("Backend returned a non-v2 theory state.")
        if state.state_id in seen_ids:
            raise TheoryBridgeV2Error("Backend returned duplicate state IDs.")
        seen_ids.add(state.state_id)
        if (
            state.record_index != expected_indices[offset]
            or state.dataset_record_sha256
            != expected_records[offset]["dataset_record_sha256"]
        ):
            raise TheoryBridgeV2Error("Backend mixed or reordered record identities.")
        _sha256(state.dataset_record_sha256, label="dataset record SHA-256")
        if state.registered_state_sha256 != state_identity(state):
            raise TheoryBridgeV2Error(
                "Backend state identity differs from its payload."
            )
        if not state.action_mask or not any(state.action_mask):
            raise TheoryBridgeV2Error("Every theory state needs one valid action.")
        _probabilities(
            state.current_probabilities, state.action_mask, label="current policy"
        )
        _probabilities(
            state.candidate_probabilities, state.action_mask, label="candidate policy"
        )
        _probabilities(
            state.deployed_probabilities, state.action_mask, label="deployed policy"
        )
    return states


def _validate_outcomes(
    values: Sequence[TheoryOutcomeV2],
    *,
    state_id: str,
    action_index: int,
) -> list[TheoryOutcomeV2]:
    outcomes = list(values)
    if not outcomes:
        raise TheoryBridgeV2Error("Exact action enumeration returned no outcomes.")
    probability_sum = 0.0
    for offset, outcome in enumerate(outcomes):
        if not isinstance(outcome, TheoryOutcomeV2):
            raise TheoryBridgeV2Error("Exact action enumeration returned a wrong type.")
        probability = _finite(
            outcome.probability,
            label=f"outcome[{state_id},{action_index},{offset}].probability",
        )
        if probability < 0.0:
            raise TheoryBridgeV2Error("Outcome probability must be nonnegative.")
        probability_sum += probability
        _finite(outcome.reward, label="outcome reward")
        if outcome.terminal != (outcome.next_state_id is None):
            raise TheoryBridgeV2Error("Terminal and successor identities disagree.")
    if not math.isclose(
        probability_sum, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE
    ):
        raise TheoryBridgeV2Error("Exact outcome probabilities do not sum to one.")
    return outcomes


def _exact_action_values(
    *,
    backend: TheoryBridgeV2Backend,
    state: TheoryStateV2,
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
            next_n = 0.0
            next_m = 0.0
            if not outcome.terminal:
                assert outcome.next_state_id is not None
                next_n = _finite(
                    backend.endpoint_value(outcome.next_state_id, n),
                    label="successor U_n",
                )
                next_m = _finite(
                    backend.endpoint_value(outcome.next_state_id, m),
                    label="successor U_m",
                )
            q_n[action_index] += outcome.probability * (outcome.reward + gamma * next_n)
            q_m[action_index] += outcome.probability * (outcome.reward + gamma * next_m)
    return q_n, q_m


def _seed(
    *,
    namespace: str,
    protocol_id: str,
    protocol_sha256: str,
    checkpoint_sha256: str,
    base_seed: int,
    state: TheoryStateV2,
    repeat_index: int,
) -> int:
    payload = canonical_json_bytes(
        [
            namespace,
            protocol_id,
            protocol_sha256,
            checkpoint_sha256,
            base_seed,
            state.record_index,
            state.dataset_record_sha256,
            state.state_id,
            repeat_index,
        ]
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bellman_rollout_return(
    *,
    rollout: BellmanRolloutV2,
    backend: TheoryBridgeV2Backend,
    depth: int,
    horizon: int,
    gamma: float,
) -> float:
    if not isinstance(rollout, BellmanRolloutV2):
        raise TheoryBridgeV2Error("Backend returned an invalid Bellman rollout.")
    _sha256(rollout.trajectory_sha256, label="Bellman trajectory SHA-256")
    if not 1 <= len(rollout.rewards) <= horizon:
        raise TheoryBridgeV2Error("Bellman rollout length is outside its horizon.")
    if rollout.terminal:
        if rollout.bootstrap_state_id is not None:
            raise TheoryBridgeV2Error("Terminal Bellman rollout cannot bootstrap.")
    elif len(rollout.rewards) != horizon or not rollout.bootstrap_state_id:
        raise TheoryBridgeV2Error("Nonterminal Bellman rollout must reach its horizon.")
    result = 0.0
    discount = 1.0
    for reward in rollout.rewards:
        result += discount * _finite(reward, label="Bellman reward")
        discount *= gamma
    if not rollout.terminal:
        assert rollout.bootstrap_state_id is not None
        result += discount * _finite(
            backend.endpoint_value(rollout.bootstrap_state_id, depth),
            label="Bellman bootstrap endpoint",
        )
    return result


def _summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        raise TheoryBridgeV2Error("Cannot summarize an empty metric.")
    checked = [_finite(value, label="metric") for value in values]
    if any(value < 0.0 for value in checked):
        raise TheoryBridgeV2Error("Theory metric summary must be nonnegative.")
    return {
        "count": len(checked),
        "mean": fmean(checked),
        "minimum": min(checked),
        "maximum": max(checked),
    }


def _normalization_diagnostic(backend: TheoryBridgeV2Backend) -> dict[str, object]:
    """Validate and copy the backend's renormalization diagnostic.

    Systems measurement only. It records how far the exported float32 vectors
    were from a normalized law and how much the binary64 renormalization moved
    them. It is not a theory quantity.
    """

    supplied = backend.normalization_diagnostic()
    if not isinstance(supplied, Mapping):
        raise TheoryBridgeV2Error("Backend normalization diagnostic is not an object.")
    expected = {
        "kind",
        "float32_mass_envelope",
        "normalized_state_count",
        "maximum_valid_mass_error",
        "maximum_normalization_correction",
        "masked_entries_zeroed_before_validation",
        "deployed_reconstructed_from_mixture",
    }
    if set(supplied) != expected:
        raise TheoryBridgeV2Error(
            "Backend normalization diagnostic inventory differs."
        )
    envelope = _finite(supplied["float32_mass_envelope"], label="mass envelope")
    mass_error = _finite(
        supplied["maximum_valid_mass_error"], label="maximum valid mass error"
    )
    correction = _finite(
        supplied["maximum_normalization_correction"],
        label="maximum normalization correction",
    )
    count = supplied["normalized_state_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or mass_error < 0.0
        or correction < 0.0
        or envelope <= 0.0
    ):
        raise TheoryBridgeV2Error("Backend normalization diagnostic is invalid.")
    if mass_error > envelope:
        raise TheoryBridgeV2Error(
            "Backend renormalized a distribution outside the float32 envelope."
        )
    # The backend must not have hidden masked leakage or rebuilt the deployed
    # law from the mixture; either would make this evaluator's checks vacuous.
    if (
        supplied["masked_entries_zeroed_before_validation"] is not False
        or supplied["deployed_reconstructed_from_mixture"] is not False
    ):
        raise TheoryBridgeV2Error(
            "Backend normalization removed evidence this evaluator must check."
        )
    return {
        "kind": str(supplied["kind"]),
        "float32_mass_envelope": envelope,
        "normalized_state_count": count,
        "maximum_valid_mass_error": mass_error,
        "maximum_normalization_correction": correction,
        "masked_entries_zeroed_before_validation": False,
        "deployed_reconstructed_from_mixture": False,
    }


def _validated_policy_return(
    value: object,
    *,
    label: str,
    maximum_environment_steps: int,
) -> CurrentPolicyReturnV2:
    if not isinstance(value, CurrentPolicyReturnV2):
        raise TheoryBridgeV2Error(f"Backend returned an invalid {label} return.")
    _sha256(value.trajectory_sha256, label=f"{label} trajectory SHA-256")
    if (
        value.terminal is not True
        or value.environment_steps <= 0
        or value.environment_steps > maximum_environment_steps
    ):
        raise TheoryBridgeV2Error(f"{label} return did not terminate as registered.")
    _finite(value.discounted_return, label=f"{label} discounted return")
    return value


def _paired_return_diagnostic(
    *,
    backend: TheoryBridgeV2Backend,
    request: Mapping[str, object],
    identity: Mapping[str, object],
    checkpoint_identity: Mapping[str, object],
    state: TheoryStateV2,
    estimator: Mapping[str, object],
    gamma: float,
) -> tuple[dict[str, object], list[float]]:
    """Evaluate registered alphas with paired common random numbers."""

    rollout_count = int(estimator["rollout_count"])
    maximum_environment_steps = int(estimator["maximum_environment_steps"])
    reference_current: list[CurrentPolicyReturnV2] | None = None
    alpha_results: list[dict[str, object]] = []
    for alpha in PAIRED_RETURN_ALPHA_VALUES:
        current_returns: list[CurrentPolicyReturnV2] = []
        mixture_returns: list[CurrentPolicyReturnV2] = []
        randomness_sha256s: list[str] = []
        for repeat_index in range(rollout_count):
            seed = _seed(
                namespace="upi-trm-policy-improvement-v2-paired-return-crn",
                protocol_id=str(request["protocol_id"]),
                protocol_sha256=str(identity["protocol_sha256"]),
                checkpoint_sha256=str(checkpoint_identity["sha256"]),
                base_seed=int(estimator["base_seed"]),
                state=state,
                repeat_index=repeat_index,
            )
            paired = backend.sample_paired_policy_returns(
                state.state_id,
                seed,
                maximum_environment_steps,
                gamma,
                alpha,
            )
            if not isinstance(paired, PairedPolicyReturnV2):
                raise TheoryBridgeV2Error(
                    "Backend returned an invalid paired policy return."
                )
            current_returns.append(
                _validated_policy_return(
                    paired.current_policy,
                    label="paired current-policy",
                    maximum_environment_steps=maximum_environment_steps,
                )
            )
            mixture_returns.append(
                _validated_policy_return(
                    paired.exact_mixture,
                    label="paired exact-mixture",
                    maximum_environment_steps=maximum_environment_steps,
                )
            )
            randomness_sha256s.append(
                _sha256(
                    paired.common_random_numbers_sha256,
                    label="paired common-random-numbers SHA-256",
                )
            )
        if reference_current is None:
            reference_current = current_returns
        elif current_returns != reference_current:
            raise TheoryBridgeV2Error(
                "Current-policy CRN returns changed across registered alpha values."
            )
        differences = [
            mixture.discounted_return - current.discounted_return
            for current, mixture in zip(current_returns, mixture_returns)
        ]
        alpha_results.append(
            {
                "alpha": alpha,
                "rollout_count": rollout_count,
                "current_policy_return_mean": fmean(
                    [value.discounted_return for value in current_returns]
                ),
                "exact_mixture_return_mean": fmean(
                    [value.discounted_return for value in mixture_returns]
                ),
                "paired_difference_mean": fmean(differences),
                "paired_difference_standard_error": (
                    stdev(differences) / math.sqrt(len(differences))
                ),
                "negative_paired_difference_count": sum(
                    difference < 0.0 for difference in differences
                ),
                "negative_paired_difference_probability": sum(
                    difference < 0.0 for difference in differences
                )
                / len(differences),
                "common_random_numbers_sha256": theory_document_sha256(
                    randomness_sha256s
                ),
            }
        )
    assert reference_current is not None
    return (
        {
            "status": "available",
            "kind": "paired_crn_current_vs_exact_mixture_v2",
            "difference_definition": "exact_mixture_minus_current",
            "alpha_results": alpha_results,
        },
        [value.discounted_return for value in reference_current],
    )


def _snapshot(value: object, *, label: str) -> ReadOnlySnapshotV2:
    if not isinstance(value, ReadOnlySnapshotV2):
        raise TheoryBridgeV2Error(f"{label} is not a v2 read-only snapshot.")
    for field in (
        "checkpoint_sha256",
        "model_state_sha256",
        "recurrent_transition_sha256",
    ):
        _sha256(getattr(value, field), label=f"{label}.{field}")
    if not value.mutable_state_sha256s:
        raise TheoryBridgeV2Error(f"{label} has no mutable-state inventory.")
    required = {
        "optimizer_states",
        "scheduler_states",
        "rng_states",
        "replay_state",
        "collector_state",
        "environment_state",
        "persistent_latent_state",
        "exact_centering_state",
    }
    if set(value.mutable_state_sha256s) != required:
        raise TheoryBridgeV2Error(f"{label} mutable-state inventory differs.")
    for name, digest in value.mutable_state_sha256s.items():
        _sha256(digest, label=f"{label}.mutable_state_sha256s.{name}")
    if value.snapshot_kind not in {
        "smoke_prepare",
        "smoke_resume",
        "scheduled",
        "interaction_matched",
    }:
        raise TheoryBridgeV2Error(f"{label} snapshot kind is invalid.")
    if value.environment_interactions <= 0:
        raise TheoryBridgeV2Error(f"{label} progress must be positive.")
    return value


def _centered_advantages(
    *,
    q_values: Sequence[float],
    probabilities: Sequence[float],
    mask: Sequence[bool],
    clipping_kind: str,
    clip_value: float | None,
) -> tuple[float, ...]:
    baseline = sum(
        probability * q_value
        for q_value, probability, valid in zip(q_values, probabilities, mask)
        if valid
    )
    raw = tuple(
        q_value - baseline if valid else 0.0 for q_value, valid in zip(q_values, mask)
    )
    if clipping_kind == "none":
        if clip_value is not None:
            raise TheoryBridgeV2Error("Unclipped trainer estimator named a clip value.")
        return raw
    if (
        clipping_kind != "clip_then_exact_recenter"
        or clip_value is None
        or clip_value <= 0.0
    ):
        raise TheoryBridgeV2Error("Trainer clipping contract is invalid.")
    clipped = tuple(
        max(-clip_value, min(clip_value, value)) if valid else 0.0
        for value, valid in zip(raw, mask)
    )
    clipped_mean = sum(
        probability * value
        for probability, value, valid in zip(probabilities, clipped, mask)
        if valid
    )
    return tuple(
        value - clipped_mean if valid else 0.0 for value, valid in zip(clipped, mask)
    )


def _validate_trainer_estimator(
    value: object,
    *,
    state: TheoryStateV2,
    current: Sequence[float],
    request: Mapping[str, object],
) -> tuple[TrainingAdvantageEstimatorV2, tuple[float, ...]]:
    if not isinstance(value, TrainingAdvantageEstimatorV2):
        raise TheoryBridgeV2Error("Backend returned an invalid trainer estimator.")
    if value.action_mask != state.action_mask:
        raise TheoryBridgeV2Error("Trainer and bridge action masks differ.")
    trainer_probabilities = _probabilities(
        value.current_probabilities,
        value.action_mask,
        label="trainer current policy",
    )
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE)
        for left, right in zip(trainer_probabilities, current)
    ):
        raise TheoryBridgeV2Error("Trainer and bridge current policies differ.")
    clipping = request["advantage_clipping"]
    assert isinstance(clipping, Mapping)
    if (
        value.clipping_kind != clipping["kind"]
        or value.clip_value != clipping["clip_value"]
        or len(value.advantages) != len(value.action_mask)
    ):
        raise TheoryBridgeV2Error(
            "Trainer estimator changes the registered clipping contract."
        )
    advantages = tuple(
        _finite(item, label=f"trainer advantage[{index}]")
        for index, item in enumerate(value.advantages)
    )
    if any(
        not valid and item != 0.0 for valid, item in zip(value.action_mask, advantages)
    ):
        raise TheoryBridgeV2Error("Trainer estimator assigns a masked advantage.")
    return value, advantages


def _persistent_invariants(
    *,
    backend: TheoryBridgeV2Backend,
    state: TheoryStateV2,
    n: int,
    m: int,
    recurrent_transition_sha256: str,
) -> None:
    deployed = backend.persistent_endpoint_witness(state.state_id, n)
    reference = backend.persistent_endpoint_witness(state.state_id, m)
    if not isinstance(deployed, PersistentEndpointWitnessV2) or not isinstance(
        reference, PersistentEndpointWitnessV2
    ):
        raise TheoryBridgeV2Error("Backend returned an invalid persistent witness.")
    for witness, depth in ((deployed, n), (reference, m)):
        if (
            witness.requested_endpoint_depth != depth
            or witness.deployed_transition_depth != n
            or witness.recurrent_transition_sha256 != recurrent_transition_sha256
        ):
            raise TheoryBridgeV2Error("Persistent witness changes deployed F_n.")
        for field in (
            "carried_successor_latent_sha256",
            "trajectory_sha256",
            "action_probabilities_sha256",
            "recurrent_transition_sha256",
        ):
            _sha256(getattr(witness, field), label=f"persistent witness {field}")
    invariant_fields = (
        "deployed_transition_depth",
        "carried_successor_latent_sha256",
        "trajectory_sha256",
        "action_probabilities_sha256",
        "recurrent_transition_sha256",
    )
    if any(
        getattr(deployed, field) != getattr(reference, field)
        for field in invariant_fields
    ):
        raise TheoryBridgeV2Error(
            "Persistent endpoint depth changed the deployed transition or carry."
        )


def _evaluate(
    *,
    request: Mapping[str, object],
    checkpoint: Path,
    backend: TheoryBridgeV2Backend,
) -> dict[str, Any]:
    identity = request["identity"]
    assert isinstance(identity, Mapping)
    try:
        backend_identity = validate_identity_bundle(backend.identity_bundle())
    except TheoryBridgeV2SchemaError as exc:
        raise TheoryBridgeV2Error("Backend identity bundle is invalid.") from exc
    if canonical_json_bytes(backend_identity) != canonical_json_bytes(identity):
        raise TheoryBridgeV2Error(
            "Backend and request identities are missing or mixed."
        )

    checkpoint_before = _stable_regular_file(checkpoint)
    checkpoint_identity = identity["checkpoint"]
    assert isinstance(checkpoint_identity, Mapping)
    if (
        checkpoint_before.sha256 != checkpoint_identity["sha256"]
        or checkpoint_before.size_bytes != checkpoint_identity["size_bytes"]
    ):
        raise TheoryBridgeV2Error("Checkpoint bytes differ from their identity.")
    before = _snapshot(backend.read_only_snapshot(), label="before snapshot")
    model_identity = identity["model"]
    assert isinstance(model_identity, Mapping)
    if (
        before.checkpoint_sha256 != checkpoint_before.sha256
        or before.model_state_sha256 != model_identity["model_sha256"]
        or before.recurrent_transition_sha256
        != model_identity["recurrent_transition_sha256"]
        or before.snapshot_kind != checkpoint_identity["snapshot_kind"]
        or before.environment_interactions
        != checkpoint_identity["environment_interactions"]
    ):
        raise TheoryBridgeV2Error(
            "Backend snapshot differs from authenticated identity."
        )

    states = _validate_states(backend.registered_states(), identity)
    n = int(request["n"])
    m = int(request["reference_depth_m"])
    horizon = int(request["bellman_horizon"])
    gamma = float(request["gamma"])
    alpha = float(request["alpha"])
    bellman_estimator = request["bellman_estimator"]
    return_estimator = request["return_estimator"]
    clipping = request["advantage_clipping"]
    assert isinstance(bellman_estimator, Mapping)
    assert isinstance(return_estimator, Mapping)
    assert isinstance(clipping, Mapping)
    parity_tolerance = float(request["centering_parity_tolerance"])
    constructed_tolerance = float(request["constructed_centering_tolerance"])
    deployment_tolerance = float(request["deployment_identity_tolerance"])

    rows: list[dict[str, object]] = []
    operator_uncertainty: list[float] = []
    b_uncertainty: list[float] = []
    return_uncertainty: list[float] = []
    deployment_values: list[float] = []

    for state in states:
        current = _probabilities(
            state.current_probabilities, state.action_mask, label="current policy"
        )
        candidate = _probabilities(
            state.candidate_probabilities, state.action_mask, label="candidate policy"
        )
        deployed = _probabilities(
            state.deployed_probabilities, state.action_mask, label="deployed policy"
        )
        if request["latent_mode"] == "persistent":
            _persistent_invariants(
                backend=backend,
                state=state,
                n=n,
                m=m,
                recurrent_transition_sha256=before.recurrent_transition_sha256,
            )
        value_n = _finite(backend.endpoint_value(state.state_id, n), label="U_n")
        value_m = _finite(backend.endpoint_value(state.state_id, m), label="U_m")
        q_n, q_m = _exact_action_values(
            backend=backend,
            state=state,
            n=n,
            m=m,
            gamma=gamma,
        )
        bridge_advantages = _centered_advantages(
            q_values=q_n,
            probabilities=current,
            mask=state.action_mask,
            clipping_kind=str(clipping["kind"]),
            clip_value=(
                None
                if clipping["clip_value"] is None
                else float(clipping["clip_value"])
            ),
        )
        _, trainer_advantages = _validate_trainer_estimator(
            backend.training_advantage_estimator(state.state_id),
            state=state,
            current=current,
            request=request,
        )
        constructed_roundoff = abs(
            sum(
                probability * advantage
                for probability, advantage, valid in zip(
                    current, bridge_advantages, state.action_mask
                )
                if valid
            )
        )
        trainer_defect = abs(
            sum(
                probability * advantage
                for probability, advantage, valid in zip(
                    current, trainer_advantages, state.action_mask
                )
                if valid
            )
        )
        parity_error = max(
            abs(left - right)
            for left, right, valid in zip(
                bridge_advantages, trainer_advantages, state.action_mask
            )
            if valid
        )
        if parity_error > parity_tolerance:
            raise TheoryBridgeV2Error(
                "Trainer and bridge exact-advantage tensors differ beyond tolerance."
            )
        if constructed_roundoff > constructed_tolerance:
            raise TheoryBridgeV2Error(
                "Independently constructed centering roundoff exceeds tolerance."
            )
        if trainer_defect > parity_tolerance:
            raise TheoryBridgeV2Error(
                "Trainer exact-advantage centering defect exceeds tolerance."
            )

        if horizon == 1:
            operator_m = sum(
                probability * q_value
                for probability, q_value, valid in zip(current, q_m, state.action_mask)
                if valid
            )
            operator_se = 0.0
        else:
            returns: list[float] = []
            for repeat_index in range(int(bellman_estimator["rollout_count"])):
                seed = _seed(
                    namespace="upi-trm-policy-improvement-v2-bellman-crn",
                    protocol_id=str(request["protocol_id"]),
                    protocol_sha256=str(identity["protocol_sha256"]),
                    checkpoint_sha256=str(checkpoint_identity["sha256"]),
                    base_seed=int(bellman_estimator["base_seed"]),
                    state=state,
                    repeat_index=repeat_index,
                )
                returns.append(
                    _bellman_rollout_return(
                        rollout=backend.sample_bellman_rollout(
                            state.state_id, horizon, seed
                        ),
                        backend=backend,
                        depth=m,
                        horizon=horizon,
                        gamma=gamma,
                    )
                )
            operator_m = fmean(returns)
            operator_se = stdev(returns) / math.sqrt(len(returns))

        if (
            request["evaluation_population"] == "validation_bridge"
            and request["method_id"] in EXACT_METHOD_IDS
        ):
            paired_return_diagnostic, return_samples = _paired_return_diagnostic(
                backend=backend,
                request=request,
                identity=identity,
                checkpoint_identity=checkpoint_identity,
                state=state,
                estimator=return_estimator,
                gamma=gamma,
            )
        else:
            return_samples = []
            for repeat_index in range(int(return_estimator["rollout_count"])):
                seed = _seed(
                    namespace="upi-trm-policy-improvement-v2-current-return-crn",
                    protocol_id=str(request["protocol_id"]),
                    protocol_sha256=str(identity["protocol_sha256"]),
                    checkpoint_sha256=str(checkpoint_identity["sha256"]),
                    base_seed=int(return_estimator["base_seed"]),
                    state=state,
                    repeat_index=repeat_index,
                )
                sampled = _validated_policy_return(
                    backend.sample_current_policy_return(
                        state.state_id,
                        seed,
                        int(return_estimator["maximum_environment_steps"]),
                        gamma,
                    ),
                    label="current-policy",
                    maximum_environment_steps=int(
                        return_estimator["maximum_environment_steps"]
                    ),
                )
                return_samples.append(sampled.discounted_return)
            paired_return_diagnostic = {
                "status": "unavailable",
                "reason": (
                    "stage0_smoke_not_scientific_calibration"
                    if request["evaluation_population"] == "stage0_smoke"
                    else "not_an_exact_probability_mixture_method"
                ),
            }
        value_hat = fmean(return_samples)
        value_hat_se = stdev(return_samples) / math.sqrt(len(return_samples))

        d_nm = abs(value_n - value_m)
        residual = abs(value_m - operator_m)
        denominator = 1.0 - gamma**horizon
        b_nm = d_nm + residual / denominator
        b_se = operator_se / denominator
        tau = 0.5 * sum(abs(left - right) for left, right in zip(candidate, current))
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
        if request["method_id"] in EXACT_METHOD_IDS:
            if deployment_tv > deployment_tolerance:
                raise TheoryBridgeV2Error("Exact pointwise mixture identity failed.")
            exact_identity_tv: float | None = deployment_tv
            realized_delta: float | None = None
        else:
            exact_identity_tv = None
            realized_delta = deployment_tv
        deployment_values.append(deployment_tv)
        rows.append(
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
                "constructed_centering_roundoff": constructed_roundoff,
                "constructed_centering_tolerance": constructed_tolerance,
                "training_estimator_centering_defect": trainer_defect,
                "training_estimator_parity_max_abs_error": parity_error,
                "training_estimator_parity_tolerance": parity_tolerance,
                "V_hat_pi": value_hat,
                "V_hat_pi_standard_error": value_hat_se,
                "E_n": abs(value_n - value_hat),
                "paired_current_to_exact_mixture_return": (paired_return_diagnostic),
                "exact_mixture_deployment_identity_tv": exact_identity_tv,
                "deployment_discrepancy_delta_dep": realized_delta,
            }
        )
        operator_uncertainty.append(operator_se)
        b_uncertainty.append(b_se)
        return_uncertainty.append(value_hat_se)

    after = _snapshot(backend.read_only_snapshot(), label="after snapshot")
    checkpoint_after = _stable_regular_file(checkpoint)
    checkpoint_unchanged = (
        checkpoint_after.identity == checkpoint_before.identity
        and checkpoint_after.sha256 == checkpoint_before.sha256
    )
    model_unchanged = after.model_state_sha256 == before.model_state_sha256
    mutable_unchanged = dict(after.mutable_state_sha256s) == dict(
        before.mutable_state_sha256s
    )
    transition_unchanged = (
        after.recurrent_transition_sha256 == before.recurrent_transition_sha256
    )
    if not (
        checkpoint_unchanged
        and model_unchanged
        and mutable_unchanged
        and transition_unchanged
        and after.checkpoint_sha256 == before.checkpoint_sha256
        and after.snapshot_kind == before.snapshot_kind
        and after.environment_interactions == before.environment_interactions
    ):
        raise TheoryBridgeV2Error(
            "Theory evaluation mutated checkpoint or training state."
        )

    metrics = {
        metric: _summary([float(row[metric]) for row in rows])
        for metric in THEORY_METRIC_IDS
    }
    deployment_kind = (
        "exact_mixture_identity"
        if request["method_id"] in EXACT_METHOD_IDS
        else "realized_policy_discrepancy"
    )
    deployment_metric = (
        "exact_mixture_deployment_identity_tv"
        if request["method_id"] in EXACT_METHOD_IDS
        else "deployment_discrepancy_delta_dep"
    )
    result: dict[str, Any] = {
        "schema_name": THEORY_RESULT_SCHEMA_NAME,
        "schema_version": THEORY_SCHEMA_VERSION,
        "protocol_id": request["protocol_id"],
        "status": "complete",
        "scope": "finite_population_diagnostic_only",
        "uniform_certificate": False,
        "evaluation_id": request["evaluation_id"],
        "evaluation_population": request["evaluation_population"],
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
        "state_count": len(rows),
        "states": rows,
        "metrics": metrics,
        "deployment_diagnostic": {
            "kind": deployment_kind,
            "metric": deployment_metric,
            "summary": _summary(deployment_values),
        },
        # Systems-only record of the exported-vector renormalization the
        # backend applies before this evaluator validates any probability.
        "normalization_diagnostic": _normalization_diagnostic(backend),
        "monte_carlo_uncertainty": {
            "bellman_operator_standard_error": _summary(operator_uncertainty),
            "B_nm_standard_error": _summary(b_uncertainty),
            "V_hat_pi_standard_error": _summary(return_uncertainty),
        },
        "persistent_semantics": {
            "applicable": request["latent_mode"] == "persistent",
            "endpoint_depth_only": True,
            "carried_successor_latent_independent_of_m": True,
            "trajectory_identity_independent_of_m": True,
            "action_probabilities_independent_of_m": True,
            "recurrent_transition_independent_of_m": True,
        },
        "read_only_verification": {
            "checkpoint_unchanged": checkpoint_unchanged,
            "model_state_unchanged": model_unchanged,
            "mutable_training_state_unchanged": mutable_unchanged,
            "recurrent_transition_unchanged": transition_unchanged,
            "mutable_state_sha256s_before": dict(before.mutable_state_sha256s),
            "mutable_state_sha256s_after": dict(after.mutable_state_sha256s),
        },
        "scientific_selection": False,
        "paper_evidence_eligible": request["paper_evidence_eligible"],
        "test_data_opened": False,
    }
    try:
        return validate_theory_result(result, request=request)
    except TheoryBridgeV2SchemaError as exc:
        raise TheoryBridgeV2Error(
            "Computed theory result failed its strict schema."
        ) from exc


def evaluate_theory_bridge_v2(
    *,
    request_document: object,
    amendment_document: object,
    checkpoint_path: str | Path,
    backend: TheoryBridgeV2Backend,
) -> dict[str, Any]:
    """Evaluate one v2 checkpoint and close its sealed backend on every path."""

    transaction_started = False
    primary_error: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        try:
            request, _ = validate_request_against_amendment(
                request_document,
                amendment_document,
            )
        except TheoryBridgeV2SchemaError as exc:
            raise TheoryBridgeV2Error(str(exc)) from exc
        backend.begin_read_only_evaluation()
        transaction_started = True
        result = _evaluate(
            request=request,
            checkpoint=_canonical_file(checkpoint_path),
            backend=backend,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        transaction_error: BaseException | None = None
        if transaction_started:
            try:
                backend.end_read_only_evaluation()
            except BaseException as exc:
                transaction_error = exc
        close_error: BaseException | None = None
        try:
            backend.close()
        except BaseException as exc:
            close_error = exc
        if transaction_error is not None:
            if primary_error is not None:
                raise transaction_error from primary_error
            raise transaction_error
        if close_error is not None:
            if primary_error is not None:
                raise close_error from primary_error
            raise close_error
    if primary_error is not None:
        raise primary_error
    if result is None:
        raise TheoryBridgeV2Error("Theory evaluation produced no result.")
    return result


def _absolute_file(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise TheoryBridgeV2Error(f"{label} must be an absolute canonical file.")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise TheoryBridgeV2Error(f"{label} is unavailable.") from exc
    if (
        resolved != path
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise TheoryBridgeV2Error(f"{label} must be a singly linked regular file.")
    return path


def _absolute_directory(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise TheoryBridgeV2Error(f"{label} must be an absolute canonical directory.")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise TheoryBridgeV2Error(f"{label} is unavailable.") from exc
    if resolved != path or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TheoryBridgeV2Error(f"{label} must be a canonical directory.")
    return path


def _json_file(path: Path, *, label: str) -> object:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise TheoryBridgeV2Error(f"{label} cannot be read.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
    ):
        raise TheoryBridgeV2Error(f"{label} changed while it was read.")
    try:
        return load_strict_json_bytes(b"".join(chunks))
    except PolicyImprovementSchemaError as exc:
        raise TheoryBridgeV2Error(f"{label} is not strict JSON.") from exc


def _attestation_matches_request(
    attestation: Mapping[str, str],
    request: Mapping[str, object],
) -> None:
    if set(attestation) != {
        "source_git_commit",
        "source_manifest_sha256",
        "runtime_sha256",
        "runtime_profile_sha256",
        "runtime_authorization_sha256",
        "launcher_sha256",
    }:
        raise TheoryBridgeV2Error("Evaluator attestation field inventory differs.")
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
        raise TheoryBridgeV2Error("Evaluator runtime differs from request identity.")


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: TheoryBackendFactoryV2 | None = None,
    evaluator_attestation: Mapping[str, str] | None = None,
    runtime_authorization: Mapping[str, object] | None = None,
) -> int:
    """Run protocol v2 only through the authenticated packaged entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
        raise TheoryBridgeV2Error(
            "Theory v2 requires its authenticated packaged entrypoint."
        )
    request_path = _absolute_file(arguments.request, label="request")
    theory_amendment_path = _absolute_file(
        arguments.theory_amendment,
        label="theory amendment",
    )
    amendment_paths = tuple(
        _absolute_file(value, label=f"amendment {index}")
        for index, value in enumerate(arguments.amendment)
    )
    if not amendment_paths or amendment_paths[0] != theory_amendment_path:
        raise TheoryBridgeV2Error(
            "The amendment history must begin with the v2 theory design."
        )
    checkpoint_path = _absolute_file(arguments.checkpoint, label="checkpoint")
    inputs = TheoryBackendInputsV2(
        project_root=_absolute_directory(arguments.project_root, label="project root"),
        protocol_path=_absolute_file(arguments.protocol, label="protocol"),
        registry_path=_absolute_file(arguments.registry, label="registry"),
        amendment_paths=amendment_paths,
        evidence_root=_absolute_directory(
            arguments.evidence_root,
            label="evidence root",
        ),
        dataset_root=_absolute_directory(
            arguments.dataset_root,
            label="dataset root",
        ),
        row_id=arguments.row_id,
        runtime_authorization=copy.deepcopy(dict(runtime_authorization)),
    )
    request_document = _json_file(request_path, label="request")
    amendment_document = _json_file(theory_amendment_path, label="theory amendment")
    try:
        request, _ = validate_request_against_amendment(
            request_document,
            amendment_document,
        )
    except TheoryBridgeV2SchemaError as exc:
        raise TheoryBridgeV2Error(str(exc)) from exc
    if request["run_id"] != inputs.row_id:
        raise TheoryBridgeV2Error("Theory request and registered row ID differ.")
    _attestation_matches_request(evaluator_attestation, request)
    backend_stdout = io.StringIO()
    with redirect_stdout(backend_stdout):
        backend = backend_factory(request, checkpoint_path, inputs)
        result = evaluate_theory_bridge_v2(
            request_document=request,
            amendment_document=amendment_document,
            checkpoint_path=checkpoint_path,
            backend=backend,
        )
    if backend_stdout.getvalue():
        raise TheoryBridgeV2Error("Theory backend wrote unauthenticated stdout.")
    payload = canonical_json_bytes(result) + b"\n"
    offset = 0
    while offset < len(payload):
        written = os.write(1, payload[offset:])
        if written <= 0:
            raise TheoryBridgeV2Error("Canonical theory stdout made no progress.")
        offset += written
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
