#!/usr/bin/env python3
"""Fail-closed checkpoint loading and identity for Phase 4 publication."""

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.config import RLConfig, merge_rl_config_layer
from rl.replay import Transition
from scripts.phase4_result_schema import (
    PHASE4_CHECKPOINT_SCHEMA_VERSION,
    PHASE4_CHECKPOINT_STEP,
    PHASE4_CONDITION_SPECS,
    PHASE4_TRAINING_INVOCATION_SCHEMA_VERSION,
    phase4_checkpoint_relpath,
    phase4_run_id,
)
from utils.run_identity import canonical_json_sha256
from utils.dataset_provenance import (
    DatasetProvenanceError,
    validate_dataset_provenance,
)


class Phase4CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot support a publication record."""


@dataclass(frozen=True)
class Phase4CheckpointIdentity:
    """Content and configuration identity extracted from one full checkpoint."""

    checkpoint_sha256: str
    model_state_sha256: str
    checkpoint_step: int
    training_run_id: str
    config_sha256: str
    rl_config_sha256: str
    model_config_sha256: str
    dataset_provenance_sha256: str
    producer_git_commit: str
    producer_source_manifest_sha256: str
    training_runtime_artifact_sha256: str
    initialization_kind: str
    checkpoint_schema_version: int
    training_invocation_schema_version: int


@dataclass(frozen=True)
class LoadedPhase4Checkpoint:
    """Strictly loaded model and its verified identity."""

    model: TinyRecursiveReasoningModel_ACTV1
    rl_config: Dict[str, Any]
    model_config: Dict[str, Any]
    identity: Phase4CheckpointIdentity


def _model_dump(config: Any) -> Dict[str, Any]:
    value = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    if not isinstance(value, dict):
        raise Phase4CheckpointError("Validated configuration did not produce a mapping.")
    return value


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise Phase4CheckpointError(f"{label} must be a string-keyed mapping.")
    return dict(value)


def _validate_producer_source(value: Any, label: str) -> Dict[str, Any]:
    producer_source = _require_mapping(value, label)
    if set(producer_source) != {
        "git_commit",
        "git_clean",
        "source_manifest_sha256",
    }:
        raise Phase4CheckpointError(f"{label} has an invalid field inventory.")
    git_commit = producer_source["git_commit"]
    if not isinstance(git_commit, str) or len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise Phase4CheckpointError(f"{label} Git commit is invalid.")
    if producer_source["git_clean"] is not True:
        raise Phase4CheckpointError(f"{label} worktree was not clean.")
    _require_sha256(
        producer_source["source_manifest_sha256"],
        f"{label} manifest digest",
    )
    return producer_source


def _strict_config(raw: Any, config_type: Any, label: str) -> Any:
    mapping = _require_mapping(raw, label)
    try:
        config = config_type(**mapping)
    except Exception as exc:
        raise Phase4CheckpointError(f"Checkpoint has an invalid {label}.") from exc
    if _model_dump(config) != mapping:
        raise Phase4CheckpointError(
            f"Checkpoint {label} is incomplete or contains ignored fields."
        )
    return config


def _stable_read(path: Path, label: str) -> tuple[bytes, str]:
    path = path.expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Phase4CheckpointError(f"{label} is not a readable regular file.") from exc
    try:
        before = os.fstat(descriptor)
        if not os.path.isfile(f"/proc/self/fd/{descriptor}"):
            raise Phase4CheckpointError(f"{label} must be a regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise Phase4CheckpointError(f"{label} changed while it was read.")
    if len(payload) != before.st_size:
        raise Phase4CheckpointError(f"{label} size changed while it was read.")
    return payload, hashlib.sha256(payload).hexdigest()


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise Phase4CheckpointError(
                "Checkpoint model_state_dict must contain only named tensors."
            )
        cpu_tensor = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(cpu_tensor.dtype),
                "shape": list(cpu_tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _load_yaml_config(path: Path) -> tuple[Dict[str, Any], str]:
    payload, config_sha256 = _stable_read(path, "Phase 4 config")
    try:
        value = yaml.safe_load(payload) or {}
    except yaml.YAMLError as exc:
        raise Phase4CheckpointError("Phase 4 config is not valid YAML.") from exc
    return _require_mapping(value, "Phase 4 config"), config_sha256


def _registered_phase4_config_layer(condition: str) -> Dict[str, Any]:
    """Return the exact semantic YAML layer registered for one design cell."""

    if condition not in PHASE4_CONDITION_SPECS:
        raise Phase4CheckpointError(f"Unknown Phase 4 condition {condition!r}.")
    spec = PHASE4_CONDITION_SPECS[condition]
    return {
        "gamma": 0.99,
        "K": 1,
        "inner_unroll_n": 2,
        "max_edits": 16,
        "num_train_steps": 5000,
        "batch_size": 32,
        "rollout_episodes_per_step": 4,
        "value_lr": 0.0003,
        "policy_lr": 0.0001,
        "value_grad_clip": 1.0,
        "policy_grad_clip": 0.5,
        "mixture_alpha": 0.1,
        "policy_epsilon": 0.1,
        "target_ema_tau": 0.99,
        "entropy_coef": 0.05,
        "enable_contraction": spec["enable_contraction"],
        "target_Lz": 0.9,
        "disable_value_head_norm": spec["disable_value_head_norm"],
        "value_target_clip": 50.0,
        "advantage_clip": 10.0,
        "batch_centered_advantage": True,
        "log_interval": 50,
        "eval_interval": 100,
        "eval_num_episodes": 50,
        "episodic_latent": True,
        "latent_ball_radius": 10.0,
        "use_feasibility_checker": True,
        "feasibility_violation_weight": 2.0,
        "feasibility_zerocand_weight": 5.0,
        "reward_shaping": True,
        "solve_terminal_reward": 1.0,
        "fail_terminal_reward": -16.0,
        "C_max": 16.0,
        "solved_threshold": None,
        "stop_action_mode": "disabled",
        "track_theory_metrics": True,
    }


def _expected_rl_config(
    yaml_config: Mapping[str, Any],
    condition: str,
) -> Dict[str, Any]:
    """Resolve the exact CLI-default plus condition-file Phase 4 RL config."""

    registered_layer = _registered_phase4_config_layer(condition)
    if canonical_json_sha256(yaml_config) != canonical_json_sha256(
        registered_layer
    ):
        raise Phase4CheckpointError(
            "Phase 4 config does not match the registered condition layer."
        )

    base = RLConfig(
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
    merged = merge_rl_config_layer(_model_dump(base), yaml_config)
    try:
        return _model_dump(RLConfig(**merged))
    except Exception as exc:
        raise Phase4CheckpointError(
            "Phase 4 condition file does not resolve to a valid RL config."
        ) from exc


def _expected_model_config(rl_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Construct the registered dummy-data TRM architecture for Phase 4."""

    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=32,
        seq_len=16,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=32,
        vocab_size=32,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=bool(rl_config["enable_contraction"]),
        rl_target_Lz=float(rl_config["target_Lz"]),
        rl_target_Lv=float(rl_config["target_Lv"]),
        rl_disable_value_head_norm=bool(
            rl_config["disable_value_head_norm"]
        ),
        rl_enable_policy_head=True,
        rl_num_actions=513,
        rl_enable_z_init_encoder=False,
        rl_latent_projection_mode=rl_config["latent_projection_mode"],
        rl_latent_ball_radius=rl_config["latent_ball_radius"],
    )
    return _model_dump(config)


def _validate_phase4_dataset_provenance(
    raw_provenance: Any,
    *,
    seed: int,
    rl_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        provenance = validate_dataset_provenance(
            _require_mapping(raw_provenance, "dataset provenance")
        )
    except DatasetProvenanceError as exc:
        raise Phase4CheckpointError("Checkpoint dataset provenance is invalid.") from exc
    if provenance["dataset_builder"] != {
        "name": "rl.training_setup.DummyPuzzleDataset",
        "version": 1,
    }:
        raise Phase4CheckpointError("Phase 4 requires the registered dummy dataset.")
    if provenance["generation_seed"] != seed:
        raise Phase4CheckpointError("Dataset generation seed is not the training seed.")
    if provenance["splits"] != {"train": "dummy", "eval": "dummy"}:
        raise Phase4CheckpointError("Phase 4 requires the registered dummy splits.")
    for split in ("train", "eval"):
        if provenance["ordered_records"][split]["count"] != 32:
            raise Phase4CheckpointError("Phase 4 dummy splits must contain 32 records.")
    metadata = _require_mapping(provenance.get("metadata"), "dataset metadata")
    expected_metadata = {
        "seq_len": 16,
        "vocab_size": 32,
        "num_identifiers": 32,
        "eval_puzzle_id_offset": 0,
        "dataset_source_names": [],
    }
    mismatched_metadata = sorted(
        field
        for field, expected in expected_metadata.items()
        if metadata.get(field) != expected
    )
    if mismatched_metadata:
        raise Phase4CheckpointError(
            "Phase 4 dataset metadata differs on fields: "
            f"{mismatched_metadata}."
        )
    expected_metadata_fields = {
        *expected_metadata,
        "train_pool_sha256",
        "eval_pool_sha256",
        "train_puzzle_identifier_ordered_sha256",
        "eval_puzzle_identifier_ordered_sha256",
    }
    if set(metadata) != expected_metadata_fields:
        raise Phase4CheckpointError(
            "Phase 4 dataset metadata has an invalid field inventory."
        )
    for field in (
        "train_pool_sha256",
        "eval_pool_sha256",
        "train_puzzle_identifier_ordered_sha256",
        "eval_puzzle_identifier_ordered_sha256",
    ):
        _require_sha256(metadata[field], f"dataset metadata {field}")
    if provenance["ordered_records"]["train"] != provenance["ordered_records"]["eval"]:
        raise Phase4CheckpointError(
            "Phase 4 dummy train and evaluation records must be identical."
        )
    if metadata["train_pool_sha256"] != metadata["eval_pool_sha256"]:
        raise Phase4CheckpointError(
            "Phase 4 dummy train and evaluation pool digests must match."
        )
    if metadata["train_puzzle_identifier_ordered_sha256"] != metadata[
        "eval_puzzle_identifier_ordered_sha256"
    ]:
        raise Phase4CheckpointError(
            "Phase 4 dummy puzzle-identifier order must match across splits."
        )
    environment = _require_mapping(
        provenance["environment_config"],
        "dataset environment config",
    )
    environment_fields = {
        "max_edits": rl_config["max_edits"],
        "gamma": rl_config["gamma"],
        "reward_shaping": rl_config["reward_shaping"],
        "task_type": "sudoku",
        "vocab_size": model_config["vocab_size"],
        "solved_threshold": rl_config["solved_threshold"],
        "stop_action_mode": rl_config["stop_action_mode"],
        "stop_action_penalty": rl_config["stop_action_penalty"],
        "enable_undo": False,
        "fail_terminal_reward": rl_config["fail_terminal_reward"],
        "solve_terminal_reward": rl_config["solve_terminal_reward"],
        "C_max": rl_config["C_max"],
        "disable_constraint_masking": rl_config[
            "disable_constraint_masking"
        ],
    }
    if set(environment) != set(environment_fields):
        raise Phase4CheckpointError(
            "Phase 4 environment identity has an invalid field inventory."
        )
    mismatched_environment = sorted(
        field
        for field, expected in environment_fields.items()
        if environment.get(field) != expected
    )
    if mismatched_environment:
        raise Phase4CheckpointError(
            "Phase 4 environment identity differs on fields: "
            f"{mismatched_environment}."
        )
    action_mask = _require_mapping(
        provenance["action_mask_config"],
        "dataset action-mask config",
    )
    action_mask_fields = {
        "task_config_class": "SudokuTaskConfig",
        "task_config_name": "sudoku",
        "disable_constraint_masking": rl_config["disable_constraint_masking"],
        "stop_action_mode": rl_config["stop_action_mode"],
        "stop_action_id": model_config["rl_num_actions"] - 1,
        "enable_undo": False,
        "undo_action_id": None,
        "vocab_size": model_config["vocab_size"],
        "num_actions": model_config["rl_num_actions"],
        "masked_token_ids": [0, 1],
    }
    if set(action_mask) != set(action_mask_fields):
        raise Phase4CheckpointError(
            "Phase 4 action-mask identity has an invalid field inventory."
        )
    mismatched_action_mask = sorted(
        field
        for field, expected in action_mask_fields.items()
        if action_mask.get(field) != expected
    )
    if mismatched_action_mask:
        raise Phase4CheckpointError(
            "Phase 4 action-mask identity differs on fields: "
            f"{mismatched_action_mask}."
        )
    return provenance


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Phase4CheckpointError(f"{label} must be a lowercase SHA-256.")
    return value


def _require_exact_int(value: Any, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise Phase4CheckpointError(f"{label} must be the integer {expected}.")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase4CheckpointError(
            f"{label} must be a nonnegative integer."
        )
    return value


def _validate_optimizer_state(value: Any, label: str) -> None:
    state = _require_mapping(value, label)
    if set(state) != {"state", "param_groups"}:
        raise Phase4CheckpointError(f"{label} has an invalid field inventory.")
    optimizer_slots = state["state"]
    parameter_groups = state["param_groups"]
    if (
        not isinstance(optimizer_slots, Mapping)
        or not optimizer_slots
        or not isinstance(parameter_groups, list)
        or not parameter_groups
        or any(
            not isinstance(group, Mapping)
            or not isinstance(group.get("params"), list)
            or not group["params"]
            for group in parameter_groups
        )
    ):
        raise Phase4CheckpointError(f"{label} is not a complete optimizer state.")


def _validate_full_phase4_training_state(
    checkpoint: Mapping[str, Any],
    rl_config: Mapping[str, Any],
) -> None:
    """Require the resume-bearing inventory emitted by schema-4 UPI saves."""

    progress = _require_mapping(checkpoint["progress"], "checkpoint progress")
    expected_progress_fields = {
        "env_steps",
        "optimizer_updates",
        "optimizer_steps",
    }
    if set(progress) != expected_progress_fields:
        raise Phase4CheckpointError(
            "Checkpoint progress has an invalid field inventory."
        )
    _require_exact_int(
        progress["optimizer_updates"],
        PHASE4_CHECKPOINT_STEP,
        "Checkpoint optimizer-update count",
    )
    progress_env_steps = _require_nonnegative_int(
        progress["env_steps"],
        "Checkpoint environment-step count",
    )
    progress_optimizer_steps = _require_nonnegative_int(
        progress["optimizer_steps"],
        "Checkpoint optimizer-step count",
    )

    trainer_state = _require_mapping(
        checkpoint["trainer_state"],
        "checkpoint trainer state",
    )
    required_trainer_fields = {
        "train_step_count",
        "env_step_count",
        "optimizer_step_count",
        "collection_state",
        "environment_state",
        "terminal_reason_replay_version",
    }
    missing_trainer_fields = sorted(required_trainer_fields - set(trainer_state))
    if missing_trainer_fields:
        raise Phase4CheckpointError(
            "Checkpoint trainer state is missing fields: "
            f"{missing_trainer_fields}."
        )
    _require_exact_int(
        trainer_state["train_step_count"],
        PHASE4_CHECKPOINT_STEP,
        "Trainer train-step count",
    )
    trainer_env_steps = _require_nonnegative_int(
        trainer_state["env_step_count"],
        "Trainer environment-step count",
    )
    trainer_optimizer_steps = _require_nonnegative_int(
        trainer_state["optimizer_step_count"],
        "Trainer optimizer-step count",
    )
    if progress_env_steps != trainer_env_steps:
        raise Phase4CheckpointError(
            "Checkpoint environment-step counters disagree."
        )
    if progress_optimizer_steps != trainer_optimizer_steps:
        raise Phase4CheckpointError(
            "Checkpoint optimizer-step counters disagree."
        )
    if not isinstance(trainer_state["collection_state"], Mapping) or not isinstance(
        trainer_state["environment_state"], Mapping
    ):
        raise Phase4CheckpointError(
            "Checkpoint collector and environment state must be mappings."
        )
    _require_exact_int(
        trainer_state["terminal_reason_replay_version"],
        1,
        "Terminal-reason replay version",
    )

    rng_state = _require_mapping(checkpoint["rng_state"], "checkpoint RNG state")
    if set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise Phase4CheckpointError(
            "Checkpoint RNG state has an invalid field inventory."
        )
    if not torch.is_tensor(rng_state["torch_cpu"]):
        raise Phase4CheckpointError(
            "Checkpoint CPU RNG state must be a tensor."
        )
    if rng_state["torch_cuda"] is not None and not isinstance(
        rng_state["torch_cuda"], list
    ):
        raise Phase4CheckpointError(
            "Checkpoint CUDA RNG state must be null or a list."
        )

    replay_transitions = checkpoint["replay_transitions"]
    if not isinstance(replay_transitions, list):
        raise Phase4CheckpointError(
            "Checkpoint replay transitions must be a list."
        )
    replay_size = _require_nonnegative_int(
        checkpoint["replay_buffer_size"],
        "Checkpoint replay-buffer size",
    )
    replay_capacity = _require_nonnegative_int(
        checkpoint["replay_capacity"],
        "Checkpoint replay capacity",
    )
    expected_replay_capacity = rl_config.get("replay_capacity")
    if (
        replay_capacity <= 0
        or replay_capacity != expected_replay_capacity
        or replay_size <= 0
        or replay_size > replay_capacity
        or replay_size != len(replay_transitions)
        or any(
            not isinstance(transition, Transition)
            for transition in replay_transitions
        )
    ):
        raise Phase4CheckpointError(
            "Checkpoint replay size or capacity is inconsistent."
        )
    execution_device = checkpoint["execution_device"]
    if not isinstance(execution_device, str) or not execution_device:
        raise Phase4CheckpointError(
            "Checkpoint execution device must be a non-empty string."
        )
    _validate_optimizer_state(
        checkpoint["value_optimizer_state_dict"],
        "value optimizer state",
    )
    _validate_optimizer_state(
        checkpoint["policy_optimizer_state_dict"],
        "policy optimizer state",
    )


def _validate_auxiliary_model_states(
    checkpoint: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> None:
    """Require all saved policy/target models to match the strict model inventory."""

    expected_names = set(model_state)
    expected_shapes = {
        name: (tensor.dtype, tuple(tensor.shape))
        for name, tensor in model_state.items()
        if torch.is_tensor(tensor)
    }
    if set(expected_shapes) != expected_names:
        raise Phase4CheckpointError(
            "Checkpoint model state must contain only tensors."
        )
    for field in (
        "policy_model_old_state_dict",
        "policy_model_candidate_state_dict",
        "target_model_state_dict",
    ):
        state = _require_mapping(checkpoint[field], field)
        if set(state) != expected_names:
            raise Phase4CheckpointError(
                f"{field} does not match the strict model key inventory."
            )
        for name, expected in expected_shapes.items():
            tensor = state[name]
            if not torch.is_tensor(tensor) or (
                tensor.dtype,
                tuple(tensor.shape),
            ) != expected:
                raise Phase4CheckpointError(
                    f"{field} has an incompatible tensor for {name!r}."
                )
        if field == "policy_model_old_state_dict" and any(
            not torch.equal(state[name], model_state[name])
            for name in expected_names
        ):
            raise Phase4CheckpointError(
                "The deployed legacy policy state differs from model_state_dict."
            )


def load_phase4_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    condition: str,
    seed: int,
    expected_producer_source: Mapping[str, Any],
    device: str | torch.device = "cpu",
    expected_identity: Optional[Mapping[str, Any]] = None,
) -> LoadedPhase4Checkpoint:
    """Hash, validate, and strictly load one checkpoint-bound Phase 4 run."""

    if condition not in PHASE4_CONDITION_SPECS:
        raise Phase4CheckpointError(f"Unknown Phase 4 condition {condition!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Phase4CheckpointError("Phase 4 seed must be a nonnegative integer.")
    authorized_producer_source = _validate_producer_source(
        expected_producer_source,
        "Authorized training producer source",
    )

    checkpoint_payload, checkpoint_sha256 = _stable_read(
        Path(checkpoint_path),
        "Phase 4 checkpoint",
    )
    try:
        raw_checkpoint = torch.load(
            io.BytesIO(checkpoint_payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise Phase4CheckpointError("Phase 4 checkpoint cannot be loaded.") from exc
    checkpoint = _require_mapping(raw_checkpoint, "Phase 4 checkpoint")

    required_fields = {
        "checkpoint_schema_version",
        "training_protocol",
        "trainer_kind",
        "step",
        "model_state_dict",
        "model_config",
        "rl_config",
        "training_invocation",
        "dataset_provenance",
        "execution_device",
        "progress",
        "rng_state",
        "trainer_state",
        "policy_model_old_state_dict",
        "policy_model_candidate_state_dict",
        "target_model_state_dict",
        "value_optimizer_state_dict",
        "policy_optimizer_state_dict",
        "replay_transitions",
        "replay_buffer_size",
        "replay_capacity",
    }
    missing = sorted(required_fields - set(checkpoint))
    if missing:
        raise Phase4CheckpointError(
            f"Phase 4 checkpoint is missing required fields: {missing}."
        )
    _require_exact_int(
        checkpoint["checkpoint_schema_version"],
        PHASE4_CHECKPOINT_SCHEMA_VERSION,
        "Checkpoint schema version",
    )
    if checkpoint["training_protocol"] != "legacy":
        raise Phase4CheckpointError("Phase 4 requires its legacy ablation protocol.")
    if checkpoint["trainer_kind"] != "UPITrmTrainer":
        raise Phase4CheckpointError("Phase 4 requires an UPITrmTrainer checkpoint.")
    _require_exact_int(
        checkpoint["step"],
        PHASE4_CHECKPOINT_STEP,
        "Checkpoint step",
    )
    rl_config_object = _strict_config(checkpoint["rl_config"], RLConfig, "RL config")
    model_config_object = _strict_config(
        checkpoint["model_config"],
        TinyRecursiveReasoningModel_ACTV1Config,
        "model config",
    )
    rl_config = _model_dump(rl_config_object)
    model_config = _model_dump(model_config_object)
    _validate_full_phase4_training_state(checkpoint, rl_config)
    yaml_config, config_sha256 = _load_yaml_config(Path(config_path))
    expected_rl_config = _expected_rl_config(yaml_config, condition)
    if rl_config != expected_rl_config:
        raise Phase4CheckpointError(
            "Embedded RL config is not the registered CLI-default plus condition config."
        )
    expected_model_config = _expected_model_config(expected_rl_config)
    if model_config != expected_model_config:
        raise Phase4CheckpointError(
            "Embedded model config is not the registered Phase 4 architecture."
        )

    expected = PHASE4_CONDITION_SPECS[condition]
    for field in ("enable_contraction", "disable_value_head_norm"):
        if rl_config[field] != expected[field]:
            raise Phase4CheckpointError(
                f"Embedded RL config does not identify condition {condition}."
            )
    model_rl_fields = {
        "batch_size": (model_config["batch_size"], rl_config["batch_size"]),
        "rl_enable_contraction": (
            model_config["rl_enable_contraction"],
            rl_config["enable_contraction"],
        ),
        "rl_target_Lz": (model_config["rl_target_Lz"], rl_config["target_Lz"]),
        "rl_target_Lv": (model_config["rl_target_Lv"], rl_config["target_Lv"]),
        "rl_disable_value_head_norm": (
            model_config["rl_disable_value_head_norm"],
            rl_config["disable_value_head_norm"],
        ),
        "rl_latent_projection_mode": (
            model_config["rl_latent_projection_mode"],
            rl_config["latent_projection_mode"],
        ),
        "rl_latent_ball_radius": (
            model_config["rl_latent_ball_radius"],
            rl_config["latent_ball_radius"],
        ),
    }
    mismatched_model_fields = sorted(
        field for field, values in model_rl_fields.items() if values[0] != values[1]
    )
    if mismatched_model_fields:
        raise Phase4CheckpointError(
            "Embedded model and RL configs disagree on fields: "
            f"{mismatched_model_fields}."
        )
    if not model_config["rl_enable_value_head"] or not model_config["rl_enable_policy_head"]:
        raise Phase4CheckpointError("Phase 4 requires value and policy heads.")
    expected_actions = model_config["seq_len"] * model_config["vocab_size"] + 1
    if model_config["rl_num_actions"] != expected_actions:
        raise Phase4CheckpointError("Phase 4 model has an invalid action count.")
    dataset_provenance = _validate_phase4_dataset_provenance(
        checkpoint["dataset_provenance"],
        seed=seed,
        rl_config=rl_config,
        model_config=model_config,
    )

    invocation = _require_mapping(
        checkpoint["training_invocation"],
        "training invocation",
    )
    invocation_fields = {
        "schema_version",
        "training_seed",
        "run_id",
        "runtime_artifact_sha256",
        "config_sources",
        "rl_config_sha256",
        "model_config_sha256",
        "dataset_provenance_sha256",
        "initialization",
        "producer_source",
    }
    if set(invocation) != invocation_fields:
        raise Phase4CheckpointError("Training invocation has an invalid field inventory.")
    _require_exact_int(
        invocation["schema_version"],
        PHASE4_TRAINING_INVOCATION_SCHEMA_VERSION,
        "Training invocation schema version",
    )
    _require_exact_int(
        invocation["training_seed"],
        seed,
        "Checkpoint-bound training seed",
    )
    expected_run_id = phase4_run_id(condition, seed)
    if invocation["run_id"] != expected_run_id:
        raise Phase4CheckpointError("Checkpoint-bound Phase 4 run ID is incorrect.")
    training_runtime_artifact_sha256 = _require_sha256(
        invocation["runtime_artifact_sha256"],
        "Training runtime artifact digest",
    )
    expected_sources = [{"name": Path(config_path).name, "sha256": config_sha256}]
    if invocation["config_sources"] != expected_sources:
        raise Phase4CheckpointError("Checkpoint-bound config source is incorrect.")
    rl_config_sha256 = canonical_json_sha256(rl_config)
    model_config_sha256 = canonical_json_sha256(model_config)
    if invocation["rl_config_sha256"] != rl_config_sha256:
        raise Phase4CheckpointError("Checkpoint-bound RL config digest is incorrect.")
    if invocation["model_config_sha256"] != model_config_sha256:
        raise Phase4CheckpointError("Checkpoint-bound model config digest is incorrect.")
    dataset_provenance_sha256 = canonical_json_sha256(dataset_provenance)
    if invocation["dataset_provenance_sha256"] != dataset_provenance_sha256:
        raise Phase4CheckpointError(
            "Checkpoint-bound dataset provenance digest is incorrect."
        )
    if invocation["initialization"] != {
        "kind": "random",
        "artifact_sha256": None,
    }:
        raise Phase4CheckpointError(
            "Phase 4 publication requires random initialization with no warm start."
        )
    producer_source = _validate_producer_source(
        invocation["producer_source"],
        "Training producer source",
    )
    if producer_source != authorized_producer_source:
        raise Phase4CheckpointError(
            "Training producer source does not match the authorized checkout."
        )
    git_commit = producer_source["git_commit"]

    state_dict = _require_mapping(checkpoint["model_state_dict"], "model state")
    _validate_auxiliary_model_states(checkpoint, state_dict)
    model_state_sha256 = _state_dict_sha256(state_dict)
    model = TinyRecursiveReasoningModel_ACTV1(model_config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise Phase4CheckpointError(
            "Phase 4 model state does not load strictly under its embedded config."
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)

    identity = Phase4CheckpointIdentity(
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
        checkpoint_step=PHASE4_CHECKPOINT_STEP,
        training_run_id=expected_run_id,
        config_sha256=config_sha256,
        rl_config_sha256=rl_config_sha256,
        model_config_sha256=model_config_sha256,
        dataset_provenance_sha256=dataset_provenance_sha256,
        producer_git_commit=git_commit,
        producer_source_manifest_sha256=producer_source[
            "source_manifest_sha256"
        ],
        training_runtime_artifact_sha256=(
            training_runtime_artifact_sha256
        ),
        initialization_kind="random",
        checkpoint_schema_version=PHASE4_CHECKPOINT_SCHEMA_VERSION,
        training_invocation_schema_version=(
            PHASE4_TRAINING_INVOCATION_SCHEMA_VERSION
        ),
    )
    if expected_identity is not None:
        actual_identity = identity.__dict__
        unexpected = sorted(set(expected_identity) - set(actual_identity))
        mismatched = sorted(
            key
            for key, value in expected_identity.items()
            if key in actual_identity and actual_identity[key] != value
        )
        if unexpected or mismatched:
            raise Phase4CheckpointError(
                "Checkpoint identity does not match the publication record: "
                f"unexpected={unexpected}, mismatched={mismatched}."
            )
    return LoadedPhase4Checkpoint(
        model=model,
        rl_config=rl_config,
        model_config=model_config,
        identity=identity,
    )


def verify_phase4_summary_checkpoints(
    summary: Mapping[str, Any],
    checkpoint_root: str | Path,
    config_root: str | Path,
    *,
    expected_producer_source: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> int:
    """Revalidate every checkpoint named by an already validated summary."""

    authorized_producer_source = _validate_producer_source(
        expected_producer_source,
        "Authorized training producer source",
    )
    try:
        resolved_checkpoint_root = Path(checkpoint_root).expanduser().resolve(
            strict=True
        )
        resolved_config_root = Path(config_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Phase4CheckpointError(
            "Phase 4 checkpoint and config roots must exist."
        ) from exc
    if not resolved_checkpoint_root.is_dir() or not resolved_config_root.is_dir():
        raise Phase4CheckpointError(
            "Phase 4 checkpoint and config roots must be directories."
        )

    runs = summary.get("all_results")
    if not isinstance(runs, list):
        raise Phase4CheckpointError("Phase 4 summary has no run list.")
    identity_fields = {
        "checkpoint_sha256",
        "model_state_sha256",
        "checkpoint_step",
        "training_run_id",
        "config_sha256",
        "rl_config_sha256",
        "model_config_sha256",
        "dataset_provenance_sha256",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "training_runtime_artifact_sha256",
        "initialization_kind",
        "checkpoint_schema_version",
        "training_invocation_schema_version",
    }
    for index, run_value in enumerate(runs):
        run = _require_mapping(run_value, f"Phase 4 run {index}")
        condition = run.get("condition")
        seed = run.get("seed")
        checkpoint_relpath = run.get("checkpoint_path")
        if not isinstance(condition, str) or not isinstance(seed, int):
            raise Phase4CheckpointError(f"Phase 4 run {index} has invalid labels.")
        if checkpoint_relpath != phase4_checkpoint_relpath(condition, seed):
            raise Phase4CheckpointError(
                f"Phase 4 run {index} has an invalid checkpoint path."
            )
        checkpoint_path = resolved_checkpoint_root / checkpoint_relpath
        config_path = resolved_config_root / f"{condition}.yaml"
        try:
            resolved_checkpoint_path = checkpoint_path.resolve(strict=True)
            resolved_config_path = config_path.resolve(strict=True)
        except OSError as exc:
            raise Phase4CheckpointError(
                f"Phase 4 run {index} is missing its checkpoint or config."
            ) from exc
        if resolved_checkpoint_root not in resolved_checkpoint_path.parents:
            raise Phase4CheckpointError(
                f"Phase 4 run {index} checkpoint escapes its trusted root."
            )
        if resolved_config_root not in resolved_config_path.parents:
            raise Phase4CheckpointError(
                f"Phase 4 run {index} config escapes its trusted root."
            )
        expected_identity = {
            field: run[field]
            for field in identity_fields
            if field in run
        }
        if set(expected_identity) != identity_fields:
            raise Phase4CheckpointError(
                f"Phase 4 run {index} has incomplete checkpoint identity."
            )
        if (
            run["producer_git_commit"]
            != authorized_producer_source["git_commit"]
            or run["producer_source_manifest_sha256"]
            != authorized_producer_source["source_manifest_sha256"]
        ):
            raise Phase4CheckpointError(
                f"Phase 4 run {index} does not match the authorized producer source."
            )
        load_phase4_checkpoint(
            resolved_checkpoint_path,
            resolved_config_path,
            condition=condition,
            seed=seed,
            expected_producer_source=authorized_producer_source,
            device=device,
            expected_identity=expected_identity,
        )
    return len(runs)
