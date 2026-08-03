"""Fail-closed inputs for persistent schema-v4 checkpoint diagnostics.

This module loads trusted local training checkpoints. Schema-v4 checkpoints
contain replay dataclasses and therefore require ``torch.load`` pickle support;
callers must not use it for downloaded or otherwise untrusted files.

The loader intentionally does not construct a trainer, restore RNG state from
the checkpoint, or synchronize any model states. Diagnostics must observe the
saved evaluator and policy pair exactly as written.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.replay import (
    ReplayIntegrityError,
    ReplayLatent,
    Transition,
    validate_transition,
    validate_transition_continuity,
)
from rl.task_config import get_task_config
from rl.training_setup import (
    OfflinePuzzleDataset,
    build_dataset_from_paths,
    offset_puzzle_identifiers,
    resolve_checker_from_dataset,
)
from utils.dataset_provenance import (
    DatasetProvenanceError,
    assert_matching_dataset_provenance,
    dataset_input_sha256s,
    dataset_puzzle_identifier_sha256s,
    dataset_sample_sha256s,
    dataset_source_build_metadata,
    ordered_record_sha256,
    validate_dataset_provenance,
)


CHECKPOINT_SCHEMA_VERSION = 4
DATASET_MANIFEST_SCHEMA_VERSION = 1
TRAINING_PROTOCOL = "fixed_base_exact"
UNVERIFIABLE_STATUS = "not verifiable from supplied evidence"


class PersistentDiagnosticInputError(RuntimeError):
    """Raised before diagnostics when a checkpoint or dataset is ambiguous."""


@dataclass(frozen=True)
class ModelStateIdentity:
    """Content hashes for one saved model snapshot."""

    full_state_sha256: str
    recurrent_map_sha256: str
    non_edit_state_sha256: str


@dataclass(frozen=True)
class ReplayValidation:
    """Validated persistent replay inventory retained by a checkpoint."""

    transition_count: int
    capacity: int
    episode_count: int
    terminal_transition_count: int
    first_episode_id: int
    last_episode_id: int


@dataclass(frozen=True)
class LoadedPersistentCheckpoint:
    """Read-only model bundle consumed by the diagnostic metric kernel."""

    evaluator: TinyRecursiveReasoningModel_ACTV1
    current_policy: TinyRecursiveReasoningModel_ACTV1
    candidate_policy: TinyRecursiveReasoningModel_ACTV1
    rl_config: RLConfig
    model_config: TinyRecursiveReasoningModel_ACTV1Config
    dataset_provenance: dict[str, Any]
    checkpoint_sha256: str
    dataset_provenance_sha256: str
    rl_config_sha256: str
    model_config_sha256: str
    evaluator_identity: ModelStateIdentity
    current_policy_identity: ModelStateIdentity
    candidate_policy_identity: ModelStateIdentity
    recurrent_map_shared: bool
    endpoint_policy_pair_invariants_verified: bool
    replay: ReplayValidation
    checkpoint_step: int
    environment_steps: int
    optimizer_updates: int
    source_execution_device: str
    checkpoint_schema_version: int
    training_protocol: str
    deployment_kind: str


@dataclass(frozen=True)
class VerifiedDiagnosticDataset:
    """Materialized datasets and environment matching checkpoint provenance."""

    train_dataset: OfflinePuzzleDataset
    eval_dataset: OfflinePuzzleDataset
    environment: PlanEditEnv
    checker: Callable[[Any, Any], float]
    checker_kind: str
    provenance: dict[str, Any]
    manifest_sha256: str
    train_ordered_sha256: str
    eval_ordered_sha256: str
    train_input_sha256s: tuple[str, ...]
    eval_input_sha256s: tuple[str, ...]
    train_identifier_count: int
    combined_identifier_count: int


@dataclass(frozen=True)
class PersistentDiagnosticContext:
    """Complete verified input context for persistent diagnostics."""

    checkpoint: LoadedPersistentCheckpoint
    dataset: VerifiedDiagnosticDataset


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        value = model.model_dump()
    elif hasattr(model, "dict"):
        value = model.dict()
    else:
        raise TypeError(f"Unsupported configuration type {type(model).__name__}.")
    if not isinstance(value, dict):
        raise TypeError("Configuration serialization must produce a dictionary.")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PersistentDiagnosticInputError(
            "A hash-linked input is not canonical JSON."
        ) from exc
    return encoded.encode("ascii")


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value using one stable, whitespace-free encoding."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_equal(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def file_sha256(path: str | Path) -> str:
    """Hash a regular file without retaining its path in the result."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PersistentDiagnosticInputError(f"Required file does not exist: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # Clone so a view into a larger storage cannot contribute bytes outside the
    # logical tensor to its content hash.
    value = tensor.detach().to(device="cpu").contiguous().clone()
    if value.numel() == 0:
        return b""
    return bytes(value.untyped_storage())


def _validate_state_dict(
    raw_state: object,
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise PersistentDiagnosticInputError(f"{label} must be a nonempty state dict.")
    if not all(isinstance(name, str) for name in raw_state):
        raise PersistentDiagnosticInputError(f"{label} contains a non-string key.")
    for name, value in raw_state.items():
        if not torch.is_tensor(value):
            raise PersistentDiagnosticInputError(
                f"{label}[{name!r}] is not a tensor."
            )
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise PersistentDiagnosticInputError(
                f"{label}[{name!r}] contains a non-finite value."
            )
    return raw_state


def state_dict_sha256(
    state: Mapping[str, torch.Tensor],
    *,
    include: Callable[[str], bool] | None = None,
) -> str:
    """Hash tensor names, types, shapes, and exact CPU bytes deterministically."""

    selected = sorted(name for name in state if include is None or include(name))
    if not selected:
        raise PersistentDiagnosticInputError("Cannot hash an empty state selection.")
    digest = hashlib.sha256()
    _update_length_prefixed(digest, b"upi-trm-state-dict-v1")
    for name in selected:
        value = state[name].detach().to(device="cpu").contiguous()
        _update_length_prefixed(digest, name.encode("utf-8"))
        _update_length_prefixed(digest, str(value.dtype).encode("ascii"))
        _update_length_prefixed(
            digest,
            _canonical_json_bytes(list(value.shape)),
        )
        _update_length_prefixed(digest, _tensor_bytes(value))
    return digest.hexdigest()


def _is_recurrent_map_state(name: str) -> bool:
    # The transition and policy conditioning path includes learned embeddings,
    # recurrent layers/buffers, and the optional state initializer. It excludes
    # the value and action heads, which have intentionally different roles.
    if name.startswith("z_init_encoder."):
        return True
    if not name.startswith("inner."):
        return False
    return not name.startswith(("inner.lm_head.", "inner.q_head."))


def _is_non_edit_state(name: str) -> bool:
    return not name.startswith("edit_policy.")


def _state_identity(state: Mapping[str, torch.Tensor]) -> ModelStateIdentity:
    return ModelStateIdentity(
        full_state_sha256=state_dict_sha256(state),
        recurrent_map_sha256=state_dict_sha256(
            state,
            include=_is_recurrent_map_state,
        ),
        non_edit_state_sha256=state_dict_sha256(
            state,
            include=_is_non_edit_state,
        ),
    )


def _state_selections_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    include: Callable[[str], bool],
) -> bool:
    left_names = {name for name in left if include(name)}
    right_names = {name for name in right if include(name)}
    if not left_names or left_names != right_names:
        return False
    for name in left_names:
        left_value = left[name]
        right_value = right[name]
        if (
            left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
            or not torch.equal(
                left_value.detach().to(device="cpu"),
                right_value.detach().to(device="cpu"),
            )
        ):
            return False
    return True


def _state_dict_structure_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        for name in left
    )


def _scalar_bool(value: torch.Tensor) -> bool:
    return bool(value.detach().to(device="cpu").reshape(()).item())


def _validate_persistent_replay(
    checkpoint: Mapping[str, object],
    *,
    action_count: int,
    max_edits: int,
    expected_latent_shape: tuple[int, ...],
) -> ReplayValidation:
    transitions = checkpoint.get("replay_transitions")
    if not isinstance(transitions, list) or not transitions:
        raise PersistentDiagnosticInputError(
            "Persistent diagnostics require a nonempty retained replay."
        )
    expected_size = checkpoint.get("replay_buffer_size")
    capacity = checkpoint.get("replay_capacity")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size != len(transitions)
    ):
        raise PersistentDiagnosticInputError(
            "Checkpoint replay size does not match its transition list."
        )
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < len(
        transitions
    ):
        raise PersistentDiagnosticInputError(
            "Checkpoint replay capacity is invalid."
        )

    terminal_count = 0
    episode_ids: set[int] = set()
    previous: Transition | None = None
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Transition):
            raise PersistentDiagnosticInputError(
                f"Replay record {index} is not a Transition."
            )
        if not isinstance(transition.latent, ReplayLatent) or not isinstance(
            transition.next_latent, ReplayLatent
        ):
            raise PersistentDiagnosticInputError(
                f"Replay record {index} is missing its persistent pre/post carry."
            )
        for label, latent in (
            ("latent", transition.latent),
            ("next_latent", transition.next_latent),
        ):
            if (
                tuple(latent.z_H.shape) != expected_latent_shape
                or tuple(latent.z_L.shape) != expected_latent_shape
            ):
                raise PersistentDiagnosticInputError(
                    f"Replay record {index} {label} has the wrong model shape."
                )
        try:
            validate_transition(transition, require_clock=True)
        except ReplayIntegrityError as exc:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} is not clock-complete: {exc}"
            ) from exc
        clock = int(
            transition.x["remaining_edits"].detach().cpu().reshape(()).item()
        )
        next_clock = int(
            transition.x_next["remaining_edits"]
            .detach()
            .cpu()
            .reshape(())
            .item()
        )
        if transition.timestep >= max_edits:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} starts beyond the configured edit horizon."
            )
        expected_clock = max_edits - transition.timestep
        if clock != expected_clock or next_clock != expected_clock - 1:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} clock disagrees with its timestep and max_edits."
            )
        if not _scalar_bool(transition.done) and next_clock == 0:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} is nonterminal at the zero-budget boundary."
            )
        if not torch.is_tensor(transition.action) or transition.action.numel() != 1:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has no scalar action."
            )
        if transition.action.dtype == torch.bool or torch.is_floating_point(
            transition.action
        ) or torch.is_complex(transition.action):
            raise PersistentDiagnosticInputError(
                f"Replay record {index} action does not have an integer dtype."
            )
        action = int(transition.action.detach().to(device="cpu").reshape(()).item())
        if action < 0 or action >= action_count:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has an out-of-range action."
            )
        if not torch.is_tensor(transition.reward) or transition.reward.numel() != 1:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has no scalar reward."
            )
        if not bool(
            torch.isfinite(transition.reward.detach().to(device="cpu")).all().item()
        ):
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has a non-finite reward."
            )
        if transition.behavior_log_prob is None or not torch.is_tensor(
            transition.behavior_log_prob
        ):
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has no collection-time behavior log probability."
            )
        behavior_log_prob = transition.behavior_log_prob.detach().to(device="cpu")
        if behavior_log_prob.numel() != 1 or not bool(
            torch.isfinite(behavior_log_prob).all().item()
        ):
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has an invalid behavior log probability."
            )
        if float(behavior_log_prob.reshape(()).item()) > 0.0:
            raise PersistentDiagnosticInputError(
                f"Replay record {index} has a positive behavior log probability."
            )

        if previous is not None:
            if transition.episode_id == previous.episode_id:
                try:
                    validate_transition_continuity(
                        previous,
                        transition,
                        require_clock=True,
                    )
                except ReplayIntegrityError as exc:
                    raise PersistentDiagnosticInputError(
                        f"Replay records {index - 1} and {index} are discontinuous: {exc}"
                    ) from exc
            else:
                if transition.episode_id <= previous.episode_id:
                    raise PersistentDiagnosticInputError(
                        "Replay episode IDs are not strictly increasing."
                    )
                if not _scalar_bool(previous.done):
                    raise PersistentDiagnosticInputError(
                        "Replay changes episode before the preceding record terminates."
                    )
                if transition.timestep != 0:
                    raise PersistentDiagnosticInputError(
                        "A replay episode boundary does not restart at timestep zero."
                    )
        terminal_count += int(_scalar_bool(transition.done))
        episode_ids.add(transition.episode_id)
        previous = transition

    return ReplayValidation(
        transition_count=len(transitions),
        capacity=capacity,
        episode_count=len(episode_ids),
        terminal_transition_count=terminal_count,
        first_episode_id=min(episode_ids),
        last_episode_id=max(episode_ids),
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise PersistentDiagnosticInputError(f"{label} must be a string-keyed mapping.")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PersistentDiagnosticInputError(f"{label} must be a nonnegative integer.")
    return value


def _strict_config(raw: object, config_type: Any, *, label: str) -> Any:
    mapping = dict(_require_mapping(raw, label=label))
    try:
        config = config_type(**mapping)
    except Exception as exc:
        raise PersistentDiagnosticInputError(f"Invalid {label}.") from exc
    if not _canonical_json_equal(_model_dump(config), mapping):
        raise PersistentDiagnosticInputError(
            f"{label} is incomplete or contains fields ignored by its schema."
        )
    return config


def _instantiate_model(
    model_config: TinyRecursiveReasoningModel_ACTV1Config,
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> TinyRecursiveReasoningModel_ACTV1:
    model = TinyRecursiveReasoningModel_ACTV1(_model_dump(model_config))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise PersistentDiagnosticInputError(
            f"{label} does not load strictly under the embedded model config."
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_persistent_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPersistentCheckpoint:
    """Load and validate one fixed-base persistent schema-v4 checkpoint.

    The pickle payload is loaded exactly once and always onto CPU. Models move
    to ``device`` only after every CPU-side validation succeeds.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_sha256 = file_sha256(path)
    try:
        raw_checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise PersistentDiagnosticInputError(
            "The trusted local checkpoint could not be loaded."
        ) from exc
    checkpoint = _require_mapping(raw_checkpoint, label="checkpoint")

    required_fields = {
        "checkpoint_schema_version",
        "training_protocol",
        "execution_device",
        "trainer_kind",
        "step",
        "progress",
        "model_state_dict",
        "policy_model_old_state_dict",
        "policy_model_candidate_state_dict",
        "target_model_state_dict",
        "value_optimizer_state_dict",
        "policy_optimizer_state_dict",
        "rng_state",
        "dataset_provenance",
        "model_config",
        "rl_config",
        "trainer_state",
        "replay_buffer_size",
        "replay_capacity",
        "replay_transitions",
    }
    missing = sorted(required_fields - set(checkpoint))
    if missing:
        raise PersistentDiagnosticInputError(
            f"Schema-v4 checkpoint is missing required fields: {missing}."
        )
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise PersistentDiagnosticInputError(
            f"Expected checkpoint schema {CHECKPOINT_SCHEMA_VERSION}."
        )
    if checkpoint.get("trainer_kind") != "UPITrmTrainer":
        raise PersistentDiagnosticInputError(
            "Persistent diagnostics require an UPITrmTrainer checkpoint."
        )

    rl_config = _strict_config(checkpoint["rl_config"], RLConfig, label="RL config")
    model_config = _strict_config(
        checkpoint["model_config"],
        TinyRecursiveReasoningModel_ACTV1Config,
        label="model config",
    )
    if checkpoint.get("training_protocol") != TRAINING_PROTOCOL or (
        rl_config.training_protocol != TRAINING_PROTOCOL
    ):
        raise PersistentDiagnosticInputError(
            "Diagnostics require matching fixed_base_exact protocol identities."
        )
    if bool(rl_config.episodic_latent):
        raise PersistentDiagnosticInputError(
            "Persistent diagnostics reject an episodic-latent checkpoint."
        )
    if not rl_config.is_fixed_base_proposal_exact():
        raise PersistentDiagnosticInputError(
            "The embedded RL config does not implement the fixed-base exact proposal."
        )
    if not rl_config.reward_shaping:
        raise PersistentDiagnosticInputError(
            "Theorem-facing persistent diagnostics require shaped rewards so the "
            "terminal transition matches the manuscript absorbing-state convention."
        )
    if not model_config.rl_enable_value_head or not model_config.rl_enable_policy_head:
        raise PersistentDiagnosticInputError(
            "Persistent diagnostics require both value and edit-policy heads."
        )
    model_rl_fields = {
        "batch_size": (model_config.batch_size, rl_config.batch_size),
        "rl_enable_contraction": (
            model_config.rl_enable_contraction,
            rl_config.enable_contraction,
        ),
        "rl_target_Lz": (model_config.rl_target_Lz, rl_config.target_Lz),
        "rl_target_Lv": (model_config.rl_target_Lv, rl_config.target_Lv),
        "rl_disable_value_head_norm": (
            model_config.rl_disable_value_head_norm,
            rl_config.disable_value_head_norm,
        ),
        "rl_latent_ball_radius": (
            model_config.rl_latent_ball_radius,
            rl_config.latent_ball_radius,
        ),
    }
    mismatched_model_fields = sorted(
        name for name, (model_value, rl_value) in model_rl_fields.items()
        if model_value != rl_value
    )
    if mismatched_model_fields:
        raise PersistentDiagnosticInputError(
            "Embedded model and RL configs disagree on fields: "
            f"{mismatched_model_fields}."
        )

    source_device = checkpoint.get("execution_device")
    valid_source_device = isinstance(source_device, str) and (
        source_device == "cpu"
        or (
            source_device.startswith("cuda:")
            and source_device.removeprefix("cuda:").isdigit()
        )
    )
    if not valid_source_device:
        raise PersistentDiagnosticInputError(
            "Schema-v4 checkpoint has an invalid execution-device identity."
        )
    assert isinstance(source_device, str)

    try:
        raw_provenance = _require_mapping(
            checkpoint["dataset_provenance"],
            label="dataset_provenance",
        )
        provenance = validate_dataset_provenance(raw_provenance)
    except (DatasetProvenanceError, TypeError) as exc:
        raise PersistentDiagnosticInputError(
            "Checkpoint dataset provenance is invalid."
        ) from exc
    _validate_provenance_record_sets(provenance)
    effective_puzzle_emb_len = (
        model_config.puzzle_emb_len
        if model_config.puzzle_emb_len != 0
        else -(
            model_config.puzzle_emb_ndim // -model_config.hidden_size
        )
    )
    replay_validation = _validate_persistent_replay(
        checkpoint,
        action_count=model_config.rl_num_actions,
        max_edits=rl_config.max_edits,
        expected_latent_shape=(
            1,
            model_config.seq_len + effective_puzzle_emb_len,
            model_config.hidden_size,
        ),
    )

    progress = _require_mapping(checkpoint["progress"], label="progress")
    trainer_state = _require_mapping(
        checkpoint["trainer_state"],
        label="trainer_state",
    )
    _require_mapping(checkpoint["rng_state"], label="rng_state")
    _require_mapping(
        checkpoint["value_optimizer_state_dict"],
        label="value_optimizer_state_dict",
    )
    _require_mapping(
        checkpoint["policy_optimizer_state_dict"],
        label="policy_optimizer_state_dict",
    )
    _require_mapping(
        trainer_state.get("environment_state"),
        label="trainer_state.environment_state",
    )
    collection_state = _require_mapping(
        trainer_state.get("collection_state"),
        label="trainer_state.collection_state",
    )
    if collection_state.get("schema_version") != 1:
        raise PersistentDiagnosticInputError(
            "Checkpoint collector state has an unsupported schema."
        )
    checkpoint_step = _require_nonnegative_int(checkpoint["step"], label="step")
    environment_steps = _require_nonnegative_int(
        progress.get("env_steps"),
        label="progress.env_steps",
    )
    optimizer_updates = _require_nonnegative_int(
        progress.get("optimizer_updates"),
        label="progress.optimizer_updates",
    )
    if replay_validation.capacity != rl_config.replay_capacity:
        raise PersistentDiagnosticInputError(
            "Checkpoint replay capacity disagrees with the embedded RL config."
        )
    if environment_steps < replay_validation.transition_count:
        raise PersistentDiagnosticInputError(
            "Checkpoint reports fewer interactions than retained replay transitions."
        )
    if trainer_state.get("env_step_count") != environment_steps or trainer_state.get(
        "train_step_count"
    ) != optimizer_updates:
        raise PersistentDiagnosticInputError(
            "Checkpoint progress counters disagree with trainer state."
        )

    evaluator_state = _validate_state_dict(
        checkpoint["model_state_dict"],
        label="model_state_dict",
    )
    current_state = _validate_state_dict(
        checkpoint["policy_model_old_state_dict"],
        label="policy_model_old_state_dict",
    )
    candidate_state = _validate_state_dict(
        checkpoint["policy_model_candidate_state_dict"],
        label="policy_model_candidate_state_dict",
    )
    target_state = _validate_state_dict(
        checkpoint["target_model_state_dict"],
        label="target_model_state_dict",
    )
    if not _state_dict_structure_equal(evaluator_state, target_state):
        raise PersistentDiagnosticInputError(
            "Target-model structure differs from the saved evaluator."
        )
    evaluator_identity = _state_identity(evaluator_state)
    current_identity = _state_identity(current_state)
    candidate_identity = _state_identity(candidate_state)

    recurrent_map_shared = _state_selections_equal(
        evaluator_state,
        current_state,
        include=_is_recurrent_map_state,
    ) and _state_selections_equal(
        current_state,
        candidate_state,
        include=_is_recurrent_map_state,
    )
    if not recurrent_map_shared:
        raise PersistentDiagnosticInputError(
            "Evaluator, current policy, and candidate do not share one recurrent map."
        )
    non_edit_state_shared = _state_selections_equal(
        evaluator_state,
        current_state,
        include=_is_non_edit_state,
    ) and _state_selections_equal(
        current_state,
        candidate_state,
        include=_is_non_edit_state,
    )
    if not non_edit_state_shared:
        raise PersistentDiagnosticInputError(
            "The fixed-base evaluator and policy pair disagree outside the edit head."
        )

    # Constructors use random initialization before strict state restoration.
    # Preserve the caller's CPU RNG stream so loading cannot affect diagnostic
    # trajectory selection or Monte Carlo draws.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        evaluator = _instantiate_model(
            model_config,
            evaluator_state,
            label="evaluator state",
        )
        current_policy = _instantiate_model(
            model_config,
            current_state,
            label="current-policy state",
        )
        candidate_policy = _instantiate_model(
            model_config,
            candidate_state,
            label="candidate-policy state",
        )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise PersistentDiagnosticInputError(
            "CUDA diagnostics were requested, but CUDA is unavailable."
        )
    for model in (evaluator, current_policy, candidate_policy):
        model.to(target_device)
        model.eval()

    return LoadedPersistentCheckpoint(
        evaluator=evaluator,
        current_policy=current_policy,
        candidate_policy=candidate_policy,
        rl_config=rl_config,
        model_config=model_config,
        dataset_provenance=provenance,
        checkpoint_sha256=checkpoint_sha256,
        dataset_provenance_sha256=canonical_json_sha256(provenance),
        rl_config_sha256=canonical_json_sha256(_model_dump(rl_config)),
        model_config_sha256=canonical_json_sha256(_model_dump(model_config)),
        evaluator_identity=evaluator_identity,
        current_policy_identity=current_identity,
        candidate_policy_identity=candidate_identity,
        recurrent_map_shared=True,
        endpoint_policy_pair_invariants_verified=True,
        replay=replay_validation,
        checkpoint_step=checkpoint_step,
        environment_steps=environment_steps,
        optimizer_updates=optimizer_updates,
        source_execution_device=source_device,
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        training_protocol=TRAINING_PROTOCOL,
        deployment_kind="exact_probability_mixture",
    )


def _load_provenance_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path).expanduser().resolve()
    manifest_sha256 = file_sha256(manifest_path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PersistentDiagnosticInputError(
                    f"Dataset manifest contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistentDiagnosticInputError(
            "Dataset provenance manifest is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise PersistentDiagnosticInputError(
            "Dataset provenance manifest must be a JSON object."
        )

    if parsed.get("manifest_schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
        raise PersistentDiagnosticInputError(
            "Unsupported or missing dataset-manifest schema version."
        )
    if set(parsed) != {"manifest_schema_version", "dataset_provenance"}:
        raise PersistentDiagnosticInputError(
            "Dataset manifest has missing or unknown top-level fields."
        )
    raw_provenance = parsed["dataset_provenance"]
    try:
        provenance = validate_dataset_provenance(raw_provenance)
    except (DatasetProvenanceError, TypeError) as exc:
        raise PersistentDiagnosticInputError(
            "External dataset provenance is invalid."
        ) from exc
    _validate_provenance_record_sets(provenance)
    return provenance, manifest_sha256


def _record_hashes(
    provenance: Mapping[str, Any],
    split: str,
) -> tuple[str, ...]:
    records = provenance["ordered_records"][split]["record_sha256s"]
    return tuple(str(value) for value in records)


def _validate_provenance_record_sets(provenance: Mapping[str, Any]) -> None:
    train_hashes = _record_hashes(provenance, "train")
    eval_hashes = _record_hashes(provenance, "eval")
    if len(set(train_hashes)) != len(train_hashes):
        raise PersistentDiagnosticInputError(
            "Dataset provenance contains duplicate training records."
        )
    if len(set(eval_hashes)) != len(eval_hashes):
        raise PersistentDiagnosticInputError(
            "Dataset provenance contains duplicate evaluation records."
        )
    overlap = set(train_hashes).intersection(eval_hashes)
    if overlap:
        raise PersistentDiagnosticInputError(
            "Dataset provenance train/eval record sets overlap."
        )


def _require_metadata(provenance: Mapping[str, Any]) -> Mapping[str, object]:
    metadata = provenance.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PersistentDiagnosticInputError(
            "Dataset provenance requires materialization metadata."
        )
    return metadata


def _validate_source_metadata(
    provenance: Mapping[str, Any],
    dataset_paths: Sequence[str],
) -> None:
    metadata = _require_metadata(provenance)
    saved_sources = metadata.get("source_build_metadata")
    if not isinstance(saved_sources, list) or not saved_sources:
        raise PersistentDiagnosticInputError(
            "Dataset provenance has no source build metadata."
        )
    try:
        current_sources = dataset_source_build_metadata(dataset_paths)
    except (ValueError, DatasetProvenanceError) as exc:
        raise PersistentDiagnosticInputError(
            "Current dataset source metadata is invalid."
        ) from exc
    for source in current_sources:
        generation_seed = source["generation_seed"]
        if (
            source["builder_name"] == "unrecorded"
            or source["builder_version"] == "unrecorded"
            or source["build_config_sha256"] is None
            or generation_seed is None
            or isinstance(generation_seed, bool)
        ):
            raise PersistentDiagnosticInputError(
                "Confirmatory dataset sources require a builder, version, seed, "
                "and build-config hash."
            )
    if not _canonical_json_equal(saved_sources, current_sources):
        raise PersistentDiagnosticInputError(
            "Dataset source builders, versions, seeds, or build-config hashes differ."
        )
    source_names = metadata.get("dataset_source_names")
    expected_source_names = [Path(path).name for path in dataset_paths]
    if not _canonical_json_equal(source_names, expected_source_names):
        raise PersistentDiagnosticInputError(
            "Dataset source-name order differs from provenance."
        )
    if metadata.get("materialization_seed") != 0 or isinstance(
        metadata.get("materialization_seed"),
        bool,
    ):
        raise PersistentDiagnosticInputError(
            "Dataset materialization seed is not the registered deterministic seed."
        )

    builder = provenance["dataset_builder"]
    if len(current_sources) == 1:
        source = current_sources[0]
        if (
            builder.get("name") != source["builder_name"]
            or builder.get("version") != source["builder_version"]
            or provenance.get("generation_seed") != source["generation_seed"]
        ):
            raise PersistentDiagnosticInputError(
                "Top-level dataset builder identity disagrees with its source metadata."
            )
    elif (
        builder.get("name") != "composite_materialized_dataset"
        or builder.get("version") != 1
        or provenance.get("generation_seed") is not None
    ):
        raise PersistentDiagnosticInputError(
            "Composite dataset builder identity is inconsistent."
        )


def _assert_exact_record_order(
    dataset: OfflinePuzzleDataset,
    expected: tuple[str, ...],
    *,
    split: str,
) -> None:
    actual = tuple(dataset_sample_sha256s(dataset))
    if len(actual) != len(expected):
        raise PersistentDiagnosticInputError(
            f"Materialized {split} count differs from its manifest."
        )
    if actual != expected:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(actual, expected))
            if left != right
        )
        raise PersistentDiagnosticInputError(
            f"Materialized {split} record order first differs at index {mismatch}."
        )


def _build_environment(
    eval_dataset: OfflinePuzzleDataset,
    checkpoint: LoadedPersistentCheckpoint,
    provenance: Mapping[str, Any],
) -> tuple[PlanEditEnv, Callable[[Any, Any], float], str]:
    raw_environment_config = dict(provenance["environment_config"])
    try:
        environment_config = PlanEditEnvConfig(**raw_environment_config)
    except TypeError as exc:
        raise PersistentDiagnosticInputError(
            "Dataset manifest environment config is invalid."
        ) from exc
    if not _canonical_json_equal(vars(environment_config), raw_environment_config):
        raise PersistentDiagnosticInputError(
            "Dataset manifest environment config is incomplete or has unknown fields."
        )
    if environment_config.enable_undo:
        raise PersistentDiagnosticInputError(
            "Persistent theorem-facing diagnostics reject UNDO because edit history "
            "is not part of the declared augmented state (x, y, z, h)."
        )
    checker_resolution = resolve_checker_from_dataset(
        rl_cfg=checkpoint.rl_config,
        dataset=eval_dataset,
        seq_len=eval_dataset.seq_len,
    )
    is_sudoku_checker = checker_resolution.checker_kind in {
        "solution",
        "constraint",
        "progress",
        "feasibility",
    }
    expected_environment_config = PlanEditEnvConfig(
        max_edits=checkpoint.rl_config.max_edits,
        gamma=checkpoint.rl_config.gamma,
        reward_shaping=checkpoint.rl_config.reward_shaping,
        vocab_size=checkpoint.model_config.vocab_size,
        solved_threshold=(
            checkpoint.rl_config.solved_threshold if is_sudoku_checker else None
        ),
        task_type=checkpoint.rl_config.task_name,
        stop_action_mode=checkpoint.rl_config.stop_action_mode,
        stop_action_penalty=checkpoint.rl_config.stop_action_penalty,
        fail_terminal_reward=checkpoint.rl_config.fail_terminal_reward,
        solve_terminal_reward=checkpoint.rl_config.solve_terminal_reward,
        disable_constraint_masking=checkpoint.rl_config.disable_constraint_masking,
    )
    if not _canonical_json_equal(
        vars(environment_config),
        vars(expected_environment_config),
    ):
        raise PersistentDiagnosticInputError(
            "Environment provenance disagrees with the embedded RL/model config."
        )
    task_config = None
    if is_sudoku_checker:
        task_config = get_task_config(
            "sudoku",
            disable_constraint_masking=environment_config.disable_constraint_masking,
        )
    elif checker_resolution.checker_kind == "dummy":
        task_config = get_task_config("dummy")

    environment = PlanEditEnv(
        dataset=eval_dataset,
        checker=checker_resolution.checker_fn,
        config=environment_config,
        task_config=task_config,
    )
    stop_action_id = checkpoint.model_config.rl_num_actions - 1
    environment.set_stop_action_id(stop_id=stop_action_id)
    expected_action_config = {
        "task_config_class": (
            type(task_config).__name__ if task_config is not None else None
        ),
        "task_config_name": (
            getattr(task_config, "name", None) if task_config is not None else None
        ),
        "disable_constraint_masking": environment_config.disable_constraint_masking,
        "stop_action_mode": environment._stop_mode,
        "stop_action_id": environment.stop_action_id,
        "enable_undo": environment._enable_undo,
        "undo_action_id": environment.undo_action_id,
        "vocab_size": environment.vocab_size,
        "num_actions": (
            environment.undo_action_id + 1
            if environment.undo_action_id is not None
            else checkpoint.model_config.rl_num_actions
        ),
        "masked_token_ids": [0, 1],
    }
    saved_action_config = dict(provenance["action_mask_config"])
    if not _canonical_json_equal(saved_action_config, expected_action_config):
        differing_fields = sorted(
            key
            for key in set(saved_action_config).union(expected_action_config)
            if saved_action_config.get(key) != expected_action_config.get(key)
        )
        raise PersistentDiagnosticInputError(
            "Action-mask provenance does not match the reconstructed environment; "
            f"differing fields: {differing_fields}."
        )
    return (
        environment,
        checker_resolution.checker_fn,
        checker_resolution.checker_kind,
    )


def materialize_verified_datasets(
    manifest_path: str | Path,
    dataset_paths: Sequence[str | Path],
    checkpoint: LoadedPersistentCheckpoint,
) -> VerifiedDiagnosticDataset:
    """Rebuild train/eval pools and match every provenance-bearing field."""

    if not dataset_paths:
        raise PersistentDiagnosticInputError(
            "At least one materialized dataset root is required."
        )
    # Preserve the final path component used when the source was registered;
    # resolving a symlink could silently change its anonymous source name.
    resolved_paths = [str(Path(path).expanduser().absolute()) for path in dataset_paths]
    if any(not Path(path).is_dir() for path in resolved_paths):
        raise PersistentDiagnosticInputError(
            "Every dataset path must be an existing directory."
        )

    provenance, manifest_sha256 = _load_provenance_manifest(manifest_path)
    try:
        assert_matching_dataset_provenance(
            checkpoint.dataset_provenance,
            provenance,
        )
    except DatasetProvenanceError as exc:
        raise PersistentDiagnosticInputError(
            "External manifest does not match checkpoint dataset provenance."
        ) from exc
    _validate_source_metadata(provenance, resolved_paths)

    train_hashes = _record_hashes(provenance, "train")
    eval_hashes = _record_hashes(provenance, "eval")
    train_split = str(provenance["splits"]["train"])
    eval_split = str(provenance["splits"]["eval"])
    if train_split == eval_split:
        raise PersistentDiagnosticInputError(
            "Training and evaluation manifests use the same split."
        )

    train_raw, train_seq_len, train_vocab_size, train_identifier_count = (
        build_dataset_from_paths(
            dataset_paths=resolved_paths,
            pool_size=len(train_hashes),
            split=train_split,
            allow_dummy_fallback=False,
        )
    )
    eval_raw, eval_seq_len, eval_vocab_size, eval_identifier_count = (
        build_dataset_from_paths(
            dataset_paths=resolved_paths,
            pool_size=len(eval_hashes),
            split=eval_split,
            allow_dummy_fallback=False,
        )
    )
    if not isinstance(train_raw, OfflinePuzzleDataset) or not isinstance(
        eval_raw, OfflinePuzzleDataset
    ):
        raise PersistentDiagnosticInputError(
            "Dataset materialization did not produce finite offline pools."
        )
    train_dataset = train_raw
    eval_dataset = eval_raw

    _assert_exact_record_order(train_dataset, train_hashes, split="train")
    _assert_exact_record_order(eval_dataset, eval_hashes, split="eval")
    train_inputs = tuple(dataset_input_sha256s(train_dataset))
    eval_inputs = tuple(dataset_input_sha256s(eval_dataset))
    if len(set(train_inputs)) != len(train_inputs):
        raise PersistentDiagnosticInputError(
            "Materialized training pool contains duplicate inputs."
        )
    if len(set(eval_inputs)) != len(eval_inputs):
        raise PersistentDiagnosticInputError(
            "Materialized evaluation pool contains duplicate inputs."
        )
    if set(train_inputs).intersection(eval_inputs):
        raise PersistentDiagnosticInputError(
            "Materialized training and evaluation inputs overlap."
        )

    if (train_seq_len, train_vocab_size) != (eval_seq_len, eval_vocab_size):
        raise PersistentDiagnosticInputError(
            "Materialized train/eval dimensions differ."
        )
    if (train_seq_len, train_vocab_size) != (
        checkpoint.model_config.seq_len,
        checkpoint.model_config.vocab_size,
    ):
        raise PersistentDiagnosticInputError(
            "Materialized dataset dimensions differ from the model config."
        )
    expected_action_count = train_seq_len * train_vocab_size + 1
    if checkpoint.model_config.rl_num_actions != expected_action_count:
        raise PersistentDiagnosticInputError(
            "Model action count is not the edit grid plus one STOP action."
        )

    metadata = _require_metadata(provenance)
    metadata_seq_len = _require_nonnegative_int(
        metadata.get("seq_len"),
        label="metadata.seq_len",
    )
    metadata_vocab_size = _require_nonnegative_int(
        metadata.get("vocab_size"),
        label="metadata.vocab_size",
    )
    if (
        metadata_seq_len < 1
        or metadata_vocab_size < 1
        or metadata_seq_len != train_seq_len
        or metadata_vocab_size != train_vocab_size
    ):
        raise PersistentDiagnosticInputError(
            "Dataset metadata dimensions disagree with materialized records."
        )
    saved_identifier_offset = _require_nonnegative_int(
        metadata.get("eval_puzzle_id_offset"),
        label="metadata.eval_puzzle_id_offset",
    )
    if saved_identifier_offset != train_identifier_count:
        raise PersistentDiagnosticInputError(
            "Evaluation puzzle-identifier offset cannot be reconstructed."
        )
    offset_puzzle_identifiers(eval_dataset, train_identifier_count)
    train_identifier_ordered_sha256 = ordered_record_sha256(
        dataset_puzzle_identifier_sha256s(train_dataset)
    )
    eval_identifier_ordered_sha256 = ordered_record_sha256(
        dataset_puzzle_identifier_sha256s(eval_dataset)
    )
    if metadata.get(
        "train_puzzle_identifier_ordered_sha256"
    ) != train_identifier_ordered_sha256 or metadata.get(
        "eval_puzzle_identifier_ordered_sha256"
    ) != eval_identifier_ordered_sha256:
        raise PersistentDiagnosticInputError(
            "Ordered puzzle identifiers disagree with checkpoint provenance."
        )
    combined_identifier_count = train_identifier_count + eval_identifier_count
    saved_identifier_count = _require_nonnegative_int(
        metadata.get("num_identifiers"),
        label="metadata.num_identifiers",
    )
    if saved_identifier_count != combined_identifier_count:
        raise PersistentDiagnosticInputError(
            "Combined puzzle-identifier count disagrees with provenance."
        )
    expected_model_identifier_count = max(
        combined_identifier_count,
        checkpoint.rl_config.batch_size,
    )
    if checkpoint.model_config.num_puzzle_identifiers != expected_model_identifier_count:
        raise PersistentDiagnosticInputError(
            "Model puzzle-identifier capacity disagrees with dataset provenance."
        )
    for sample in train_dataset.samples:
        identifier = int(torch.as_tensor(sample["puzzle_identifiers"]).max().item())
        if identifier < 0 or identifier >= train_identifier_count:
            raise PersistentDiagnosticInputError(
                "A training puzzle identifier is outside its reconstructed range."
            )
    for sample in eval_dataset.samples:
        identifier = int(torch.as_tensor(sample["puzzle_identifiers"]).max().item())
        if identifier < train_identifier_count or identifier >= expected_model_identifier_count:
            raise PersistentDiagnosticInputError(
                "An evaluation puzzle identifier is outside its reconstructed range."
            )

    train_ordered_sha256 = ordered_record_sha256(train_hashes)
    eval_ordered_sha256 = ordered_record_sha256(eval_hashes)
    if metadata.get("train_pool_sha256") != train_ordered_sha256 or metadata.get(
        "eval_pool_sha256"
    ) != eval_ordered_sha256:
        raise PersistentDiagnosticInputError(
            "Dataset pool hashes disagree with ordered record manifests."
        )

    environment, checker, checker_kind = _build_environment(
        eval_dataset,
        checkpoint,
        provenance,
    )
    return VerifiedDiagnosticDataset(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        environment=environment,
        checker=checker,
        checker_kind=checker_kind,
        provenance=provenance,
        manifest_sha256=manifest_sha256,
        train_ordered_sha256=train_ordered_sha256,
        eval_ordered_sha256=eval_ordered_sha256,
        train_input_sha256s=train_inputs,
        eval_input_sha256s=eval_inputs,
        train_identifier_count=train_identifier_count,
        combined_identifier_count=combined_identifier_count,
    )


def load_persistent_diagnostic_context(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    dataset_paths: Sequence[str | Path],
    *,
    device: str | torch.device = "cpu",
) -> PersistentDiagnosticContext:
    """Load a verified fixed-base checkpoint and its exact external datasets."""

    checkpoint = load_persistent_checkpoint(checkpoint_path, device=device)
    dataset = materialize_verified_datasets(
        manifest_path,
        dataset_paths,
        checkpoint,
    )
    return PersistentDiagnosticContext(checkpoint=checkpoint, dataset=dataset)
