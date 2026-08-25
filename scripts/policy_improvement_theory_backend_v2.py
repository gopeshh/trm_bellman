#!/usr/bin/env fbpython
"""Adapter from the sealed full-run session to the protocol-v2 theory core."""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, TYPE_CHECKING

from policy_improvement_sealed_evidence import (
    load_sealed_checkpoint_validator,
    SealedCheckpoint,
)
from scripts.policy_improvement_full_runtime import (
    FULL_EXECUTION_ENV,
    FULL_EXECUTION_VALUE,
    FullRuntimeError,
    load_registered_full_run,
    resolve_authenticated_full_checkpoint,
)
from scripts.policy_improvement_evidence import authenticate_complete_generation
from scripts.policy_improvement_populations import load_registered_populations
from scripts.policy_improvement_registry import (
    generate_registry,
    load_registered_base_configs,
    validate_registry_document,
)
from scripts.policy_improvement_schema import (
    amendment_history_sha256,
    canonical_json_bytes,
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    validate_protocol,
    validate_runtime_authorization,
    validated_result_payload,
)
from scripts.policy_improvement_theory_bridge_v2 import (
    BellmanRolloutV2,
    CurrentPolicyReturnV2,
    PairedPolicyReturnV2,
    PersistentEndpointWitnessV2,
    ReadOnlySnapshotV2,
    TheoryBackendInputsV2,
    TheoryBridgeV2Error,
    TheoryOutcomeV2,
    TheoryStateV2,
    TrainingAdvantageEstimatorV2,
)
from scripts.policy_improvement_theory_schema_v2 import (
    build_stage0_theory_request,
    build_validation_theory_request,
    theory_document_sha256,
    validate_theory_request,
)
from scripts.policy_improvement_v2_schema import (
    bind_v2_result_to_registration,
    PolicyImprovementV2SchemaError,
    validate_v2_result as validate_full_v2_result,
)
from utils.run_identity import discover_clean_git_source


if TYPE_CHECKING:
    from policy_improvement_smoke_runtime import SmokeContext


class FullTheorySessionV2(Protocol):
    def begin_read_only_evaluation(self) -> None: ...

    def end_read_only_evaluation(self) -> None: ...

    def identity_bundle(self) -> Mapping[str, object]: ...

    def read_only_snapshot_v2(self) -> object: ...

    def registered_states(self) -> Sequence[object]: ...

    def normalization_diagnostic(self) -> Mapping[str, object]: ...

    def endpoint_value(self, state_id: str, depth: int) -> float: ...

    def exact_action_outcomes(self, state_id: str, action_index: int) -> object: ...

    def sample_rollout(self, state_id: str, horizon: int, seed: int) -> object: ...

    def training_advantage_estimator(self, state_id: str) -> object: ...

    def persistent_endpoint_witness(
        self, state_id: str, endpoint_depth: int
    ) -> object: ...

    def sample_current_policy_return(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
    ) -> object: ...

    def sample_paired_policy_returns(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
        alpha: float,
    ) -> object: ...

    def close(self) -> None: ...


def _attributes(value: object, fields: tuple[str, ...], *, label: str) -> None:
    if any(not hasattr(value, field) for field in fields):
        raise TheoryBridgeV2Error(f"Full backend returned an invalid {label}.")


class SealedTheoryBackendV2:
    """Convert full-backend records without exposing trainer objects to the core."""

    def __init__(self, session: FullTheorySessionV2) -> None:
        self._session = session
        self._closed = False

    def identity_bundle(self) -> Mapping[str, object]:
        value = self._session.identity_bundle()
        if not isinstance(value, Mapping):
            raise TheoryBridgeV2Error(
                "Full backend returned an invalid identity bundle."
            )
        return copy.deepcopy(dict(value))

    def begin_read_only_evaluation(self) -> None:
        self._session.begin_read_only_evaluation()

    def end_read_only_evaluation(self) -> None:
        self._session.end_read_only_evaluation()

    def normalization_diagnostic(self) -> Mapping[str, object]:
        return self._session.normalization_diagnostic()

    def read_only_snapshot(self) -> ReadOnlySnapshotV2:
        value = self._session.read_only_snapshot_v2()
        _attributes(
            value,
            (
                "checkpoint_sha256",
                "model_state_sha256",
                "mutable_state_sha256s",
                "recurrent_transition_sha256",
                "snapshot_kind",
                "environment_interactions",
            ),
            label="v2 read-only snapshot",
        )
        mutable = value.mutable_state_sha256s
        if not isinstance(mutable, Mapping):
            raise TheoryBridgeV2Error(
                "Full backend returned an invalid mutable inventory."
            )
        return ReadOnlySnapshotV2(
            checkpoint_sha256=str(value.checkpoint_sha256),
            model_state_sha256=str(value.model_state_sha256),
            mutable_state_sha256s={
                str(name): str(digest) for name, digest in mutable.items()
            },
            recurrent_transition_sha256=str(value.recurrent_transition_sha256),
            snapshot_kind=str(value.snapshot_kind),
            environment_interactions=int(value.environment_interactions),
        )

    def registered_states(self) -> tuple[TheoryStateV2, ...]:
        raw_states = self._session.registered_states()
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
            raise TheoryBridgeV2Error(
                "Full backend returned an invalid state sequence."
            )
        result: list[TheoryStateV2] = []
        for value in raw_states:
            _attributes(
                value,
                (
                    "state_id",
                    "record_index",
                    "dataset_record_sha256",
                    "registered_state_sha256",
                    "action_mask",
                    "current_probabilities",
                    "candidate_probabilities",
                    "deployed_probabilities",
                ),
                label="v2 theory state",
            )
            result.append(
                TheoryStateV2(
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
        return tuple(result)

    def endpoint_value(self, state_id: str, depth: int) -> float:
        return float(self._session.endpoint_value(state_id, depth))

    def exact_action_outcomes(
        self, state_id: str, action_index: int
    ) -> tuple[TheoryOutcomeV2, ...]:
        value = self._session.exact_action_outcomes(state_id, action_index)
        raw = (
            value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else (value,)
        )
        result: list[TheoryOutcomeV2] = []
        for item in raw:
            _attributes(
                item,
                ("probability", "reward", "terminal", "next_state_id"),
                label="v2 exact outcome",
            )
            result.append(
                TheoryOutcomeV2(
                    probability=float(item.probability),
                    reward=float(item.reward),
                    terminal=bool(item.terminal),
                    next_state_id=(
                        None if item.next_state_id is None else str(item.next_state_id)
                    ),
                )
            )
        return tuple(result)

    def sample_bellman_rollout(
        self, state_id: str, horizon: int, seed: int
    ) -> BellmanRolloutV2:
        value = self._session.sample_rollout(state_id, horizon, seed)
        _attributes(
            value,
            ("rewards", "terminal", "bootstrap_state_id", "trajectory_sha256"),
            label="v2 Bellman rollout",
        )
        return BellmanRolloutV2(
            rewards=tuple(float(item) for item in value.rewards),
            terminal=bool(value.terminal),
            bootstrap_state_id=(
                None
                if value.bootstrap_state_id is None
                else str(value.bootstrap_state_id)
            ),
            trajectory_sha256=str(value.trajectory_sha256),
        )

    def training_advantage_estimator(
        self, state_id: str
    ) -> TrainingAdvantageEstimatorV2:
        value = self._session.training_advantage_estimator(state_id)
        _attributes(
            value,
            (
                "action_mask",
                "current_probabilities",
                "advantages",
                "clipping_kind",
                "clip_value",
            ),
            label="v2 training estimator",
        )
        return TrainingAdvantageEstimatorV2(
            action_mask=tuple(bool(item) for item in value.action_mask),
            current_probabilities=tuple(
                float(item) for item in value.current_probabilities
            ),
            advantages=tuple(float(item) for item in value.advantages),
            clipping_kind=str(value.clipping_kind),
            clip_value=(None if value.clip_value is None else float(value.clip_value)),
        )

    def persistent_endpoint_witness(
        self, state_id: str, endpoint_depth: int
    ) -> PersistentEndpointWitnessV2:
        value = self._session.persistent_endpoint_witness(state_id, endpoint_depth)
        _attributes(
            value,
            (
                "requested_endpoint_depth",
                "deployed_transition_depth",
                "carried_successor_latent_sha256",
                "trajectory_sha256",
                "action_probabilities_sha256",
                "recurrent_transition_sha256",
            ),
            label="v2 persistent witness",
        )
        return PersistentEndpointWitnessV2(
            requested_endpoint_depth=int(value.requested_endpoint_depth),
            deployed_transition_depth=int(value.deployed_transition_depth),
            carried_successor_latent_sha256=str(value.carried_successor_latent_sha256),
            trajectory_sha256=str(value.trajectory_sha256),
            action_probabilities_sha256=str(value.action_probabilities_sha256),
            recurrent_transition_sha256=str(value.recurrent_transition_sha256),
        )

    def sample_current_policy_return(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
    ) -> CurrentPolicyReturnV2:
        value = self._session.sample_current_policy_return(
            state_id,
            seed,
            maximum_environment_steps,
            gamma,
        )
        _attributes(
            value,
            ("discounted_return", "environment_steps", "terminal", "trajectory_sha256"),
            label="v2 current-policy return",
        )
        return CurrentPolicyReturnV2(
            discounted_return=float(value.discounted_return),
            environment_steps=int(value.environment_steps),
            terminal=bool(value.terminal),
            trajectory_sha256=str(value.trajectory_sha256),
        )

    def sample_paired_policy_returns(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
        alpha: float,
    ) -> PairedPolicyReturnV2:
        value = self._session.sample_paired_policy_returns(
            state_id,
            seed,
            maximum_environment_steps,
            gamma,
            alpha,
        )
        _attributes(
            value,
            (
                "current_policy",
                "exact_mixture",
                "common_random_numbers_sha256",
            ),
            label="v2 paired policy return",
        )

        def convert(raw: object, *, label: str) -> CurrentPolicyReturnV2:
            _attributes(
                raw,
                (
                    "discounted_return",
                    "environment_steps",
                    "terminal",
                    "trajectory_sha256",
                ),
                label=label,
            )
            return CurrentPolicyReturnV2(
                discounted_return=float(raw.discounted_return),
                environment_steps=int(raw.environment_steps),
                terminal=bool(raw.terminal),
                trajectory_sha256=str(raw.trajectory_sha256),
            )

        return PairedPolicyReturnV2(
            current_policy=convert(
                value.current_policy,
                label="v2 paired current-policy return",
            ),
            exact_mixture=convert(
                value.exact_mixture,
                label="v2 paired exact-mixture return",
            ),
            common_random_numbers_sha256=str(value.common_random_numbers_sha256),
        )

    def close(self) -> None:
        if not self._closed:
            self._session.close()
            self._closed = True


@dataclass(frozen=True)
class _AuthenticatedStage0Checkpoint:
    path: Path
    sealed: SealedCheckpoint
    parent_checkpoint_sha256: str
    model_state_sha256: str
    role_state_sha256s: Mapping[str, object]


def _available(value: object, *, label: str, length: int = 64) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"status", "value"}
        or value.get("status") != "available"
    ):
        raise TheoryBridgeV2Error(f"{label} is not frozen.")
    payload = value.get("value")
    if (
        not isinstance(payload, str)
        or len(payload) != length
        or any(character not in "0123456789abcdef" for character in payload)
    ):
        raise TheoryBridgeV2Error(f"{label} has an invalid digest.")
    return payload


def _private_owner_root(path: Path, *, project_root: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise TheoryBridgeV2Error("Evidence root must be an absolute canonical path.")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise TheoryBridgeV2Error("Evidence root is unavailable.") from exc
    if (
        resolved != path
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o022
        or path == project_root
        or path in project_root.parents
        or project_root in path.parents
    ):
        raise TheoryBridgeV2Error("Evidence root ownership or placement is unsafe.")
    return path


def _canonical_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TheoryBridgeV2Error(f"{label} is not a canonical relative path.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise TheoryBridgeV2Error(f"{label} is not a canonical relative path.")
    return relative


def _load_stable_json(path: Path, *, label: str) -> object:
    descriptor = -1
    try:
        before_path = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before_path.st_nlink != 1
            or before.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise TheoryBridgeV2Error(f"{label} is not a private regular file.")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise TheoryBridgeV2Error(f"{label} cannot be read safely.") from exc
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
        identity(before_path) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(after_path)
    ):
        raise TheoryBridgeV2Error(f"{label} changed while it was read.")
    try:
        return load_strict_json_bytes(b"".join(chunks))
    except PolicyImprovementSchemaError as exc:
        raise TheoryBridgeV2Error(f"{label} is not strict JSON.") from exc


def _stage0_context(
    *,
    request: Mapping[str, object],
    inputs: TheoryBackendInputsV2,
    authorization: Mapping[str, object],
) -> tuple[SmokeContext, Mapping[str, object], Mapping[str, object]]:
    project = inputs.project_root
    if discover_clean_git_source(project) != {
        "git_commit": authorization["producer_git_commit"],
        "git_clean": True,
    }:
        raise TheoryBridgeV2Error(
            "Stage 0 theory source differs from its runtime authorization."
        )
    expected_protocol = project / "configs/policy_improvement_v2/protocol.json"
    expected_registry = project / "configs/policy_improvement_v2/registry.json"
    if (
        inputs.protocol_path != expected_protocol
        or inputs.registry_path != expected_registry
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 theory requires the canonical protocol-v2 namespace."
        )
    try:
        protocol = validate_protocol(
            _load_stable_json(expected_protocol, label="protocol")
        )
        population_document = load_registered_populations(protocol, project)
        base_configs = load_registered_base_configs(protocol, project)
        registry_document = _load_stable_json(expected_registry, label="registry")
        registry = validate_registry_document(
            registry_document,
            protocol,
            [],
            base_configs=base_configs,
            populations_value=population_document,
        )
        regenerated = generate_registry(
            protocol,
            [],
            base_configs=base_configs,
            populations_value=population_document,
        )
    except PolicyImprovementSchemaError as exc:
        raise TheoryBridgeV2Error(
            "Stage 0 protocol or registry cannot be authenticated."
        ) from exc
    if canonical_json_bytes(registry) != canonical_json_bytes(regenerated):
        raise TheoryBridgeV2Error("Stage 0 registry regeneration differs.")
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    registry_sha256 = hashlib.sha256(canonical_json_bytes(registry)).hexdigest()
    if (
        protocol.get("schema_name") != "policy_improvement_protocol_v2"
        or authorization.get("protocol_sha256") != protocol_sha256
        or not isinstance(authorization.get("registry"), Mapping)
        or authorization["registry"].get("sha256") != registry_sha256
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 protocol or registry differs from runtime authorization."
        )
    expected_theory_amendment = (
        project / "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
    )
    if inputs.amendment_paths != (expected_theory_amendment,):
        raise TheoryBridgeV2Error(
            "Stage 0 bridge smoke accepts only the committed theory amendment."
        )
    theory_amendment = _load_stable_json(
        expected_theory_amendment,
        label="theory amendment",
    )
    theory_amendment_sha256 = hashlib.sha256(
        canonical_json_bytes(theory_amendment)
    ).hexdigest()
    request_identity = request.get("identity")
    authorization_amendments = authorization.get("amendments")
    if (
        not isinstance(request_identity, Mapping)
        or request_identity.get("theory_amendment_sha256") != theory_amendment_sha256
        or not isinstance(authorization_amendments, list)
        or len(authorization_amendments) != 1
        or not isinstance(authorization_amendments[0], Mapping)
        or authorization_amendments[0].get("schema_name")
        != "policy_improvement_theory_bridge_amendment_v2"
        or authorization_amendments[0].get("sha256") != theory_amendment_sha256
    ):
        raise TheoryBridgeV2Error(
            "Theory amendment differs from runtime authorization."
        )
    rows = [
        row
        for row in registry["rows"]
        if isinstance(row, Mapping) and row.get("run_id") == inputs.row_id
    ]
    if len(rows) != 1:
        raise TheoryBridgeV2Error("Stage 0 row does not resolve exactly.")
    row = rows[0]
    if (
        row.get("phase") != "stage0_smoke"
        or row.get("row_kind") != "concrete"
        or row.get("evaluation_split") != "train"
        or row.get("evaluation_population") != "stage0_smoke"
        or row.get("method_id")
        not in {"fixed_base_exact_persistent", "fixed_base_exact_episodic"}
        or request.get("run_id") != row.get("run_id")
        or request.get("method_id") != row.get("method_id")
        or request.get("n") != row.get("n")
        or request.get("bellman_horizon") != row.get("K")
        or request.get("alpha") != row.get("alpha")
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 theory request differs from its exact registered row."
        )
    population = population_document["populations"]["stage0_smoke"]
    if (
        population.get("split") != "train"
        or population.get("count") != 8
        or len(population.get("indices", [])) != 8
    ):
        raise TheoryBridgeV2Error("Stage 0 train-only population differs.")
    dataset_root = inputs.dataset_root
    expected_dataset = (project / str(protocol["dataset"]["root"])).resolve(strict=True)
    if dataset_root != expected_dataset:
        raise TheoryBridgeV2Error("Stage 0 dataset root differs from protocol.")
    owner = _private_owner_root(inputs.evidence_root, project_root=project)
    output = _canonical_relative(
        protocol["output_root"]["relative_path"],
        label="Stage 0 output root",
    )
    roles = {
        role["role"]: role
        for role in authorization["roles"]
        if isinstance(role, Mapping) and isinstance(role.get("role"), str)
    }
    training = roles.get("policy-improvement-training")
    evaluation = roles.get("policy-improvement-evaluation")
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        raise TheoryBridgeV2Error("Stage 0 authorization lacks training roles.")
    if any(
        training[field] != evaluation[field]
        for field in (
            "source_git_commit",
            "runtime_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
        )
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 training and evaluation runtime identities differ."
        )
    dataset_registration = protocol["dataset"]
    producer_registration = dataset_registration["producer_source"]
    from policy_improvement_smoke_runtime import SmokeContext

    context = SmokeContext(
        protocol=copy.deepcopy(dict(protocol)),
        protocol_sha256=protocol_sha256,
        registry=copy.deepcopy(dict(registry)),
        registry_sha256=registry_sha256,
        row=copy.deepcopy(dict(row)),
        registry_row_sha256=hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
        source_root=project,
        dataset_root=dataset_root,
        dataset_manifest_sha256=_available(
            dataset_registration["manifest_sha256"],
            label="dataset manifest",
        ),
        dataset_producer_source={
            field: _available(
                producer_registration[field],
                label=f"dataset producer {field}",
                length=40 if field == "git_commit" else 64,
            )
            for field in (
                "git_commit",
                "launcher_sha256",
                "runtime_sha256",
                "source_manifest_sha256",
            )
        },
        evidence_root=owner,
        run_root=owner / output / "runs" / str(row["run_id"]),
        segment_budget=32,
        segment_name="resume",
        runtime_sha256=str(training["runtime_sha256"]),
        runtime_authorization_sha256=runtime_authorization_sha256(authorization),
        runtime_profile_sha256=str(training["runtime_profile_sha256"]),
        selected_source_manifest_sha256=str(
            training["selected_source_manifest_sha256"]
        ),
        launcher_sha256=str(authorization["launcher_sha256"]),
        producer_commit=str(authorization["producer_git_commit"]),
        producer_manifest_sha256=str(authorization["producer_source_manifest_sha256"]),
    )
    return context, population, training


def _per_instance_documents(generation: Path) -> dict[str, object]:
    documents: dict[str, object] = {}
    for path in sorted(generation.rglob("*.per_instance.json")):
        value = _load_stable_json(path, label="per-instance document")
        digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        if digest in documents:
            raise TheoryBridgeV2Error("Per-instance document digest is duplicated.")
        documents[digest] = value
    return documents


def _independently_audit_stage0(
    *,
    request: Mapping[str, object],
    checkpoint_path: Path,
    inputs: TheoryBackendInputsV2,
    authorization: Mapping[str, object],
    context: SmokeContext,
) -> None:
    """Re-run the canonical four-row Stage 0 audit before theory restore."""

    from scripts.policy_improvement_audit import audit_result_set

    stage0_rows = [
        row for row in context.registry["rows"] if row.get("phase") == "stage0_smoke"
    ]
    expected_run_ids = {str(row["run_id"]) for row in stage0_rows}
    if len(stage0_rows) != 4 or len(expected_run_ids) != 4:
        raise TheoryBridgeV2Error(
            "Stage 0 theory requires the canonical four-row registry."
        )
    publication_root = context.run_root.parents[1]
    runs_root = publication_root / "runs"
    try:
        run_directories = list(runs_root.iterdir())
    except OSError as exc:
        raise TheoryBridgeV2Error("Stage 0 run inventory is unavailable.") from exc
    if (
        len(run_directories) != 4
        or {path.name for path in run_directories} != expected_run_ids
        or any(
            path.resolve(strict=True) != path
            or stat.S_ISLNK(path.lstat().st_mode)
            or not stat.S_ISDIR(path.lstat().st_mode)
            for path in run_directories
        )
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 evidence root differs from the canonical four-row inventory."
        )

    results: list[object] = []
    per_instance: dict[str, object] = {}
    for run_id in sorted(expected_run_ids):
        generation = runs_root / run_id / "segments/env_000000032"
        results.append(
            _load_stable_json(generation / "result.json", label="Stage 0 result")
        )
        for digest, document in _per_instance_documents(generation).items():
            if digest in per_instance:
                raise TheoryBridgeV2Error(
                    "Stage 0 per-instance document digest is duplicated."
                )
            per_instance[digest] = document

    theory_role = _runtime_role(authorization, "policy-improvement-theory-bridge")
    execution_identity = {
        "runtime_sha256": theory_role["runtime_sha256"],
        "runtime_profile_sha256": theory_role["runtime_profile_sha256"],
        "source_git_commit": theory_role["source_git_commit"],
        "launcher_sha256": authorization["launcher_sha256"],
        "producer_git_commit": authorization["producer_git_commit"],
        "producer_source_manifest_sha256": authorization[
            "producer_source_manifest_sha256"
        ],
    }
    try:
        report = audit_result_set(
            context.protocol,
            context.registry,
            results,
            per_instance,
            phases=("stage0_smoke",),
            amendment_history=(),
            base_configs=load_registered_base_configs(
                context.protocol, context.source_root
            ),
            project_root=context.source_root,
            dataset_root=context.dataset_root,
            evidence_root=publication_root,
            runtime_authorization=authorization,
            audit_execution_identity=execution_identity,
            checkpoint_validator=load_sealed_checkpoint_validator(),
            execution_role_name="policy-improvement-theory-bridge",
        )
    except (PolicyImprovementSchemaError, ValueError) as exc:
        raise TheoryBridgeV2Error(
            "Stage 0 canonical four-row re-audit failed."
        ) from exc
    if (
        report.get("expected_rows") != 4
        or report.get("complete_rows") != 4
        or report.get("failed_rows") != 0
        or report.get("historical_failed_attempt_count") != 0
        or report.get("per_instance_artifact_count") != 10
        or report.get("semantic_checkpoint_validation_count") != 8
        or report.get("compute_accounting_artifact_count") != 4
        or report.get("validation_data_opened") is not False
        or report.get("test_data_opened") is not False
        or report.get("test_open_verified") is not False
    ):
        raise TheoryBridgeV2Error("Stage 0 canonical four-row re-audit is incomplete.")
    requests = report.get("stage0_theory_requests")
    if not isinstance(requests, list) or len(requests) != 2:
        raise TheoryBridgeV2Error("Stage 0 canonical re-audit omitted theory requests.")
    matches = [
        item
        for item in requests
        if isinstance(item, Mapping) and item.get("run_id") == context.row["run_id"]
    ]
    if (
        len(matches) != 1
        or canonical_json_bytes(matches[0].get("request"))
        != canonical_json_bytes(request)
        or Path(str(matches[0].get("checkpoint_path"))).resolve(strict=True)
        != checkpoint_path
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 theory request differs from the independent four-row audit."
        )


def _resolve_stage0_checkpoint(
    *,
    request: Mapping[str, object],
    checkpoint_path: Path,
    inputs: TheoryBackendInputsV2,
    authorization: Mapping[str, object],
    context: SmokeContext,
) -> _AuthenticatedStage0Checkpoint:
    publication_root = context.run_root.parents[1]
    generation = context.run_root / "segments/env_000000032"
    result_path = generation / "result.json"
    try:
        result_document, result = validated_result_payload(
            _load_stable_json(result_path, label="Stage 0 result")
        )
        bind_v2_result_to_registration(
            result_document,
            context.row,
            context.protocol,
            context.registry,
            load_registered_populations(context.protocol, context.source_root),
        )
    except (PolicyImprovementSchemaError, PolicyImprovementV2SchemaError) as exc:
        raise TheoryBridgeV2Error("Stage 0 result schema is invalid.") from exc
    exact_bindings = {
        "run_id": context.row["run_id"],
        "phase": "stage0_smoke",
        "tier": "smoke",
        "evaluation_split": "train",
        "method_id": context.row["method_id"],
        "base_method_id": context.row["base_method_id"],
        "n": context.row["n"],
        "K": context.row["K"],
        "alpha": context.row["alpha"],
        "status": "complete",
    }
    if any(result.get(field) != expected for field, expected in exact_bindings.items()):
        raise TheoryBridgeV2Error("Stage 0 complete result differs from its row.")
    if result.get("amendment_history_sha256") != amendment_history_sha256([]):
        raise TheoryBridgeV2Error("Stage 0 result has nonempty training amendments.")
    request_identity = request.get("identity")
    if not isinstance(request_identity, Mapping):
        raise TheoryBridgeV2Error("Stage 0 theory request lacks identity.")
    checkpoint_identity = request_identity.get("checkpoint")
    if not isinstance(checkpoint_identity, Mapping):
        raise TheoryBridgeV2Error("Stage 0 theory request lacks checkpoint identity.")
    requested_checkpoint_sha256 = str(checkpoint_identity["sha256"])
    try:
        authenticated = authenticate_complete_generation(
            evidence_root=publication_root,
            result_path=result_path,
            result=result_document,
            protocol_sha256=context.protocol_sha256,
            registry_sha256=context.registry_sha256,
            registry_row_sha256=context.registry_row_sha256,
            expected_environment_interactions=32,
            per_instance_documents=_per_instance_documents(generation),
            checkpoint_validator=load_sealed_checkpoint_validator(),
            protocol=context.protocol,
            registry_row=context.row,
            project_root=context.source_root,
            dataset_root=context.dataset_root,
            runtime_authorization=authorization,
            amendment_history_sha256=amendment_history_sha256([]),
            authenticated_test_open_sha256=None,
            retain_checkpoint_sha256=requested_checkpoint_sha256,
        )
    except PolicyImprovementSchemaError as exc:
        raise TheoryBridgeV2Error(
            "Stage 0 complete generation cannot be authenticated."
        ) from exc
    sealed = authenticated.get("sealed_checkpoint")
    if not isinstance(sealed, SealedCheckpoint):
        raise TheoryBridgeV2Error(
            "Stage 0 complete generation did not retain its sealed checkpoint."
        )
    try:
        resolved_path = Path(str(authenticated["generation_path"])) / (
            sealed.generation_relative_path
        )
        if (
            checkpoint_path != resolved_path
            or checkpoint_identity.get("size_bytes") != sealed.size_bytes
            or checkpoint_identity.get("snapshot_kind") != "smoke_resume"
            or checkpoint_identity.get("environment_interactions") != 32
        ):
            raise TheoryBridgeV2Error(
                "Stage 0 checkpoint path or identity differs from complete evidence."
            )
        raw_validations = authenticated.get("semantic_validations")
        if not isinstance(raw_validations, list):
            raise TheoryBridgeV2Error("Stage 0 semantic validations are missing.")
        resume = [
            item
            for item in raw_validations
            if isinstance(item, Mapping)
            and item.get("checkpoint_sha256") == sealed.sha256
            and item.get("environment_interactions") == 32
        ]
        if len(resume) != 1:
            raise TheoryBridgeV2Error(
                "Stage 0 resume semantic validation does not resolve exactly."
            )
        parent = resume[0].get("parent_checkpoint_sha256")
        prepare = [
            item
            for item in raw_validations
            if isinstance(item, Mapping)
            and item.get("checkpoint_sha256") == parent
            and item.get("environment_interactions") == 16
            and item.get("parent_checkpoint_sha256") is None
        ]
        if (
            not isinstance(parent, str)
            or len(parent) != 64
            or len(prepare) != 1
            or authenticated.get("parent_generation_manifest_sha256") is None
        ):
            raise TheoryBridgeV2Error(
                "Stage 0 16 -> 32 checkpoint lineage is incomplete."
            )
        role_states = resume[0].get("role_state_sha256s")
        model_state = resume[0].get("model_state_sha256")
        theory_model_identity = resume[0].get("theory_model_identity")
        request_identity = request.get("identity")
        requested_model_identity = (
            request_identity.get("model")
            if isinstance(request_identity, Mapping)
            else None
        )
        if (
            not isinstance(role_states, Mapping)
            or not isinstance(model_state, str)
            or not isinstance(theory_model_identity, Mapping)
            or not isinstance(requested_model_identity, Mapping)
        ):
            raise TheoryBridgeV2Error(
                "Stage 0 resume model-state validation is incomplete."
            )
        if canonical_json_bytes(theory_model_identity) != canonical_json_bytes(
            requested_model_identity
        ):
            raise TheoryBridgeV2Error(
                "Stage 0 theory request model differs from sealed semantic validation."
            )
        retained = _AuthenticatedStage0Checkpoint(
            path=resolved_path,
            sealed=sealed,
            parent_checkpoint_sha256=parent,
            model_state_sha256=model_state,
            role_state_sha256s=copy.deepcopy(dict(role_states)),
        )
        sealed = None
        return retained
    finally:
        if sealed is not None:
            sealed.close()


def _expected_stage0_identity(
    *,
    request_identity: Mapping[str, object],
    context: SmokeContext,
    population: Mapping[str, object],
    training_role: Mapping[str, object],
    authorization: Mapping[str, object],
    inputs: TheoryBackendInputsV2,
) -> dict[str, object]:
    selected_indices = list(population["indices"])
    selected_records = [
        {"record_index": index, "dataset_record_sha256": record}
        for index, record in zip(selected_indices, population["record_sha256s"])
    ]
    expected = {
        "protocol_id": context.protocol["protocol_id"],
        "protocol_schema_name": context.protocol["schema_name"],
        "protocol_schema_version": context.protocol["schema_version"],
        "protocol_sha256": context.protocol_sha256,
        "population_registry_schema_name": "policy_improvement_populations_v2",
        "population_registry_schema_version": 1,
        "population_registry_sha256": context.protocol["population_registry"]["sha256"],
        "registry_schema_name": context.registry["schema_name"],
        "registry_schema_version": context.registry["registry_schema_version"],
        "registry_sha256": context.registry_sha256,
        "registry_row_schema_name": "policy_improvement_registry_row_v2",
        "registry_row_schema_version": 1,
        "theory_amendment_sha256": request_identity["theory_amendment_sha256"],
        "registry_row_sha256": context.registry_row_sha256,
        "checkpoint": copy.deepcopy(request_identity["checkpoint"]),
        "model": copy.deepcopy(request_identity["model"]),
        "config": {
            "file_sha256": next(
                method["config_sha256"]
                for method in context.protocol["methods"]
                if method["id"] == context.row["base_method_id"]
            ),
            "base_canonical_sha256": context.row["base_config_canonical_sha256"],
            "effective_config_sha256": context.row["expected_effective_config_sha256"],
        },
        "producer_source": {
            "git_commit": authorization["producer_git_commit"],
            "source_manifest_sha256": authorization["producer_source_manifest_sha256"],
        },
        "training_runtime": {
            "role": "policy-improvement-smoke",
            "source_git_commit": training_role["source_git_commit"],
            "source_manifest_sha256": training_role["selected_source_manifest_sha256"],
            "runtime_sha256": training_role["runtime_sha256"],
            "runtime_profile_sha256": training_role["runtime_profile_sha256"],
            "selected_source_manifest_sha256": training_role[
                "selected_source_manifest_sha256"
            ],
            "runtime_authorization_sha256": runtime_authorization_sha256(authorization),
            "launcher_sha256": authorization["launcher_sha256"],
        },
        "dataset_records": {
            "split": "train",
            "population_id": population["population_id"],
            "population_binding_sha256": population["binding_sha256"],
            "split_manifest_sha256": _available(
                context.protocol["dataset"]["splits"]["train"]["manifest_sha256"],
                label="train manifest",
            ),
            "ordered_record_sha256": population["ordered_record_sha256"],
            "ordered_input_sha256": population["ordered_input_sha256"],
            "selected_record_indices": selected_indices,
            "selected_record_indices_sha256": theory_document_sha256(selected_indices),
            "selected_records": selected_records,
            "selected_records_sha256": theory_document_sha256(selected_records),
            "selected_input_sha256s": list(population["input_sha256s"]),
            "selected_input_sha256s_sha256": theory_document_sha256(
                list(population["input_sha256s"])
            ),
            "record_count": population["count"],
        },
        "evaluator_source": copy.deepcopy(request_identity["evaluator_source"]),
        "evaluator_runtime": copy.deepcopy(request_identity["evaluator_runtime"]),
    }
    return expected


def _runtime_role(
    authorization: Mapping[str, object],
    role_name: str,
) -> Mapping[str, object]:
    roles = authorization.get("roles")
    if not isinstance(roles, list):
        raise TheoryBridgeV2Error("Runtime authorization has no role inventory.")
    matches = [
        role
        for role in roles
        if isinstance(role, Mapping) and role.get("role") == role_name
    ]
    if len(matches) != 1:
        raise TheoryBridgeV2Error(
            f"Runtime authorization does not bind exactly one {role_name} role."
        )
    return matches[0]


def _expected_validation_identity(
    *,
    request_identity: Mapping[str, object],
    registered_run: object,
    population: Mapping[str, object],
    checkpoint: object,
    authorization: Mapping[str, object],
) -> dict[str, object]:
    protocol = getattr(registered_run, "protocol", None)
    registry = getattr(registered_run, "registry", None)
    row = getattr(registered_run, "row", None)
    history = getattr(registered_run, "amendment_history", None)
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(registry, Mapping)
        or not isinstance(row, Mapping)
        or not isinstance(history, tuple)
    ):
        raise TheoryBridgeV2Error("Registered validation run is incomplete.")
    theory_amendments = [
        amendment
        for amendment in history
        if amendment.get("schema_name")
        == "policy_improvement_theory_bridge_amendment_v2"
    ]
    if len(theory_amendments) != 1:
        raise TheoryBridgeV2Error(
            "Registered validation run lacks one v2 theory amendment."
        )
    method = [
        item
        for item in protocol.get("methods", [])
        if isinstance(item, Mapping) and item.get("id") == row.get("base_method_id")
    ]
    if len(method) != 1:
        raise TheoryBridgeV2Error("Registered validation method does not resolve.")
    full_role = _runtime_role(authorization, "policy-improvement-full")
    theory_role = _runtime_role(authorization, "policy-improvement-theory-bridge")
    authorization_digest = runtime_authorization_sha256(authorization)
    evaluator_source = request_identity.get("evaluator_source")
    evaluator_runtime = request_identity.get("evaluator_runtime")
    expected_evaluator_source = {
        "git_commit": theory_role["source_git_commit"],
        "source_manifest_sha256": theory_role["selected_source_manifest_sha256"],
    }
    expected_evaluator_runtime = {
        "runtime_sha256": theory_role["runtime_sha256"],
        "runtime_profile_sha256": theory_role["runtime_profile_sha256"],
        "runtime_authorization_sha256": authorization_digest,
        "launcher_sha256": authorization["launcher_sha256"],
    }
    if (
        not isinstance(evaluator_source, Mapping)
        or dict(evaluator_source) != expected_evaluator_source
        or not isinstance(evaluator_runtime, Mapping)
        or dict(evaluator_runtime) != expected_evaluator_runtime
    ):
        raise TheoryBridgeV2Error(
            "Validation evaluator identity differs from runtime authorization."
        )
    indices = population.get("indices")
    records = population.get("record_sha256s")
    inputs = population.get("input_sha256s")
    if not all(isinstance(value, list) for value in (indices, records, inputs)):
        raise TheoryBridgeV2Error("Validation-bridge population is incomplete.")
    assert isinstance(indices, list)
    assert isinstance(records, list)
    assert isinstance(inputs, list)
    selected_records = [
        {"record_index": index, "dataset_record_sha256": digest}
        for index, digest in zip(indices, records)
    ]
    checkpoint_fields = {
        "sha256": getattr(checkpoint, "sha256", None),
        "size_bytes": getattr(checkpoint, "size_bytes", None),
        "snapshot_kind": getattr(checkpoint, "snapshot_kind", None),
        "environment_interactions": getattr(
            checkpoint, "environment_interactions", None
        ),
    }
    split_registration = protocol.get("dataset", {}).get("splits", {}).get("validation")
    population_registration = protocol.get("population_registry")
    if not isinstance(split_registration, Mapping) or not isinstance(
        population_registration, Mapping
    ):
        raise TheoryBridgeV2Error("Validation dataset registration is incomplete.")
    split_manifest = _available(
        split_registration.get("manifest_sha256"),
        label="validation manifest",
    )
    model = getattr(checkpoint, "theory_model_identity", None)
    if not isinstance(model, Mapping):
        raise TheoryBridgeV2Error(
            "Authenticated validation checkpoint lacks its theory model identity."
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_schema_name": protocol["schema_name"],
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": getattr(registered_run, "protocol_sha256"),
        "population_registry_schema_name": population_registration["schema_name"],
        "population_registry_schema_version": population_registration["schema_version"],
        "population_registry_sha256": population_registration["sha256"],
        "registry_schema_name": registry["schema_name"],
        "registry_schema_version": registry["registry_schema_version"],
        "registry_sha256": getattr(registered_run, "registry_sha256"),
        "registry_row_schema_name": row["schema_name"],
        "registry_row_schema_version": row["schema_version"],
        "theory_amendment_sha256": theory_document_sha256(theory_amendments[0]),
        "registry_row_sha256": getattr(registered_run, "registry_row_sha256"),
        "checkpoint": checkpoint_fields,
        "model": copy.deepcopy(dict(model)),
        "config": {
            "file_sha256": method[0]["config_sha256"],
            "base_canonical_sha256": row["base_config_canonical_sha256"],
            "effective_config_sha256": row["expected_effective_config_sha256"],
        },
        "producer_source": {
            "git_commit": authorization["producer_git_commit"],
            "source_manifest_sha256": authorization["producer_source_manifest_sha256"],
        },
        "training_runtime": {
            "role": "policy-improvement-full",
            "source_git_commit": full_role["source_git_commit"],
            "source_manifest_sha256": full_role["selected_source_manifest_sha256"],
            "runtime_sha256": full_role["runtime_sha256"],
            "runtime_profile_sha256": full_role["runtime_profile_sha256"],
            "selected_source_manifest_sha256": full_role[
                "selected_source_manifest_sha256"
            ],
            "runtime_authorization_sha256": authorization_digest,
            "launcher_sha256": authorization["launcher_sha256"],
        },
        "dataset_records": {
            "split": "validation",
            "population_id": population["population_id"],
            "population_binding_sha256": population["binding_sha256"],
            "split_manifest_sha256": split_manifest,
            "ordered_record_sha256": population["ordered_record_sha256"],
            "ordered_input_sha256": population["ordered_input_sha256"],
            "selected_record_indices": list(indices),
            "selected_record_indices_sha256": theory_document_sha256(indices),
            "selected_records": selected_records,
            "selected_records_sha256": theory_document_sha256(selected_records),
            "selected_input_sha256s": list(inputs),
            "selected_input_sha256s_sha256": theory_document_sha256(inputs),
            "record_count": population["count"],
        },
        "evaluator_source": expected_evaluator_source,
        "evaluator_runtime": expected_evaluator_runtime,
    }


def _create_validation_theory_backend_v2(
    *,
    request: Mapping[str, object],
    checkpoint_path: Path,
    inputs: TheoryBackendInputsV2,
    authorization: Mapping[str, object],
) -> SealedTheoryBackendV2:
    if (
        authorization.get("schema_name")
        != "policy_improvement_runtime_authorization_v3"
    ):
        raise TheoryBridgeV2Error(
            "Validation bridge requires runtime authorization v3."
        )
    if discover_clean_git_source(inputs.project_root) != {
        "git_commit": authorization["producer_git_commit"],
        "git_clean": True,
    }:
        raise TheoryBridgeV2Error(
            "Validation theory source differs from runtime authorization."
        )
    expected_protocol = (
        inputs.project_root / "configs/policy_improvement_v2/protocol.json"
    )
    expected_registry = (
        inputs.project_root / "configs/policy_improvement_v2/registry.json"
    )
    expected_theory = (
        inputs.project_root
        / "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
    )
    if (
        inputs.protocol_path != expected_protocol
        or inputs.registry_path != expected_registry
        or not inputs.amendment_paths
        or inputs.amendment_paths[0] != expected_theory
    ):
        raise TheoryBridgeV2Error(
            "Validation bridge requires the canonical protocol-v2 registration."
        )
    authorization_digest = runtime_authorization_sha256(authorization)
    try:
        registered_run = load_registered_full_run(
            project_root=inputs.project_root,
            protocol_path=inputs.protocol_path,
            registry_path=inputs.registry_path,
            amendment_paths=inputs.amendment_paths,
            evidence_root=inputs.evidence_root,
            dataset_root=inputs.dataset_root,
            row_id=inputs.row_id,
            runtime_authorization_sha256=authorization_digest,
            environment={FULL_EXECUTION_ENV: FULL_EXECUTION_VALUE},
            require_stage1_selection=True,
        )
    except FullRuntimeError as exc:
        raise TheoryBridgeV2Error(
            "Registered validation run cannot be authenticated."
        ) from exc
    authorization_amendments = authorization.get("amendments")
    theory_amendment_sha256 = theory_document_sha256(
        registered_run.amendment_history[0]
    )
    if (
        authorization.get("protocol_sha256") != registered_run.protocol_sha256
        or not isinstance(authorization.get("registry"), Mapping)
        or authorization["registry"].get("sha256") != registered_run.registry_sha256
        or not isinstance(authorization_amendments, list)
        or len(authorization_amendments) != 1
        or not isinstance(authorization_amendments[0], Mapping)
        or authorization_amendments[0].get("schema_name")
        != "policy_improvement_theory_bridge_amendment_v2"
        or authorization_amendments[0].get("sha256") != theory_amendment_sha256
    ):
        raise TheoryBridgeV2Error(
            "Validation registration differs from runtime authorization."
        )
    row = registered_run.row
    if (
        request.get("evaluation_population") != "validation_bridge"
        or row.get("row_kind") != "concrete"
        or row.get("evaluation_split") != "validation"
        or row.get("evaluation_population") != "validation_select"
        or request.get("run_id") != row.get("run_id")
        or request.get("method_id") != row.get("method_id")
        or request.get("n") != row.get("n")
        or request.get("bellman_horizon") != row.get("K")
        or request.get("alpha") != row.get("alpha")
    ):
        raise TheoryBridgeV2Error(
            "Validation theory request differs from its exact V_select row."
        )
    populations = (
        registered_run.population_document.get("populations")
        if isinstance(registered_run.population_document, Mapping)
        else None
    )
    population = (
        populations.get("validation_bridge")
        if isinstance(populations, Mapping)
        else None
    )
    if (
        not isinstance(population, Mapping)
        or population.get("split") != "validation"
        or population.get("count") != 128
    ):
        raise TheoryBridgeV2Error("Validation-bridge population does not resolve.")
    request_identity = request.get("identity")
    if not isinstance(request_identity, Mapping):
        raise TheoryBridgeV2Error("Validation theory request lacks identity.")
    checkpoint_identity = request_identity.get("checkpoint")
    if not isinstance(checkpoint_identity, Mapping):
        raise TheoryBridgeV2Error(
            "Validation theory request lacks checkpoint identity."
        )
    try:
        resolved = resolve_authenticated_full_checkpoint(
            registered_run,
            checkpoint_environment_interactions=int(
                checkpoint_identity["environment_interactions"]
            ),
            runtime_authorization=authorization,
            result_validator=validate_full_v2_result,
        )
    except (FullRuntimeError, TypeError, ValueError) as exc:
        raise TheoryBridgeV2Error(
            "Validation checkpoint is not bound to complete authenticated evidence."
        ) from exc
    try:
        if checkpoint_path != resolved.path:
            raise TheoryBridgeV2Error(
                "Validation checkpoint path differs from authenticated evidence."
            )
        expected_identity = _expected_validation_identity(
            request_identity=request_identity,
            registered_run=registered_run,
            population=population,
            checkpoint=resolved,
            authorization=authorization,
        )
        if canonical_json_bytes(request_identity) != canonical_json_bytes(
            expected_identity
        ):
            raise TheoryBridgeV2Error(
                "Validation theory request contains missing or mixed identities."
            )
        theory_amendment = registered_run.amendment_history[0]
        base_configs = load_registered_base_configs(
            registered_run.protocol,
            registered_run.project_root,
        )
        base_config = base_configs.get(str(row["base_method_id"]))
        if not isinstance(base_config, Mapping):
            raise TheoryBridgeV2Error(
                "Validation theory request lacks its base configuration."
            )
        canonical_request = build_validation_theory_request(
            amendment_value=theory_amendment,
            row=row,
            identity=expected_identity,
            effective_config={**base_config, **row["config_override"]},
            checkpoint_environment_interactions=int(
                checkpoint_identity["environment_interactions"]
            ),
            reference_depth_m=int(request["reference_depth_m"]),
        )
        if canonical_json_bytes(request) != canonical_json_bytes(canonical_request):
            raise TheoryBridgeV2Error(
                "Validation theory request differs from canonical registration."
            )
        full_backend = importlib.import_module("policy_improvement_full_backend")
        open_session = getattr(
            full_backend,
            "open_validation_theory_bridge_session_v2",
            None,
        )
        if not callable(open_session):
            raise TheoryBridgeV2Error(
                "Full backend does not expose the v2 validation theory API."
            )
        training_module = importlib.import_module("upi_trm_train")
        session = open_session(
            request_identity,
            resolved.path,
            registered_run=registered_run,
            evaluation_population=population,
            expected_checkpoint_sha256=resolved.sha256,
            sealed_checkpoint_descriptor=resolved.sealed_descriptor,
            authenticated_model_state_sha256=resolved.model_state_sha256,
            authenticated_role_state_sha256s=resolved.role_state_sha256s,
            authenticated_validation_sha256=resolved.validation_sha256,
            runtime_identity=expected_identity["training_runtime"],
            training_module=training_module,
        )
        return SealedTheoryBackendV2(session)
    finally:
        os.close(resolved.sealed_descriptor)


def create_theory_bridge_backend_v2(
    request_value: Mapping[str, object],
    checkpoint_path: Path,
    inputs: TheoryBackendInputsV2,
) -> SealedTheoryBackendV2:
    """Authenticate and restore one protocol-v2 theory checkpoint."""

    try:
        request = validate_theory_request(request_value)
        authorization = validate_runtime_authorization(inputs.runtime_authorization)
    except (PolicyImprovementSchemaError, ValueError) as exc:
        raise TheoryBridgeV2Error(
            "Theory request or runtime authorization is invalid."
        ) from exc
    if request["evaluation_population"] == "validation_bridge":
        raise TheoryBridgeV2Error(
            "Validation-bridge execution remains blocked until the canonical "
            "30-request schedule and its independently re-audited V_select "
            "selection provenance are implemented."
        )
    context, population, training_role = _stage0_context(
        request=request,
        inputs=inputs,
        authorization=authorization,
    )
    request_identity = request["identity"]
    assert isinstance(request_identity, Mapping)
    expected_identity = _expected_stage0_identity(
        request_identity=request_identity,
        context=context,
        population=population,
        training_role=training_role,
        authorization=authorization,
        inputs=inputs,
    )
    if canonical_json_bytes(request_identity) != canonical_json_bytes(
        expected_identity
    ):
        raise TheoryBridgeV2Error(
            "Stage 0 theory request contains missing or mixed identities."
        )
    theory_amendment = _load_stable_json(
        inputs.amendment_paths[0],
        label="theory amendment",
    )
    base_configs = load_registered_base_configs(context.protocol, context.source_root)
    base_config = base_configs.get(str(context.row["base_method_id"]))
    if not isinstance(base_config, Mapping):
        raise TheoryBridgeV2Error(
            "Stage 0 theory request lacks its registered base configuration."
        )
    try:
        canonical_request = build_stage0_theory_request(
            amendment_value=theory_amendment,
            row=context.row,
            identity=expected_identity,
            effective_config={**base_config, **context.row["config_override"]},
        )
    except (TypeError, PolicyImprovementSchemaError, ValueError) as exc:
        raise TheoryBridgeV2Error(
            "Stage 0 theory request cannot be reconstructed from registration."
        ) from exc
    if canonical_json_bytes(request) != canonical_json_bytes(canonical_request):
        raise TheoryBridgeV2Error(
            "Stage 0 theory request differs from the audited canonical request."
        )
    _independently_audit_stage0(
        request=request,
        checkpoint_path=checkpoint_path,
        inputs=inputs,
        authorization=authorization,
        context=context,
    )
    resolved = _resolve_stage0_checkpoint(
        request=request,
        checkpoint_path=checkpoint_path,
        inputs=inputs,
        authorization=authorization,
        context=context,
    )
    try:
        full_backend = importlib.import_module("policy_improvement_full_backend")
        open_session = getattr(
            full_backend,
            "open_stage0_theory_bridge_session_v2",
            None,
        )
        if not callable(open_session):
            raise TheoryBridgeV2Error(
                "Full backend does not expose the Stage 0 sealed theory API."
            )
        training_module = importlib.import_module("upi_trm_train")
        runtime_identity = dict(expected_identity["training_runtime"])
        runtime_identity["producer_source_manifest_sha256"] = authorization[
            "producer_source_manifest_sha256"
        ]
        session = open_session(
            request_identity,
            resolved.path,
            context=context,
            expected_checkpoint_sha256=resolved.sealed.sha256,
            sealed_checkpoint_descriptor=resolved.sealed.descriptor,
            parent_checkpoint_sha256=resolved.parent_checkpoint_sha256,
            authenticated_model_state_sha256=resolved.model_state_sha256,
            authenticated_role_state_sha256s=resolved.role_state_sha256s,
            runtime_identity=runtime_identity,
            training_module=training_module,
        )
        return SealedTheoryBackendV2(session)
    finally:
        resolved.sealed.close()
