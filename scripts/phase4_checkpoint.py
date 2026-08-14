#!/usr/bin/env python3
"""Fail-closed checkpoint loading and identity for Phase 4 publication."""

import hashlib
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import yaml

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from rl.config import RLConfig, merge_rl_config_layer
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.replay import ReplayBuffer, ReplayIntegrityError, Transition
from rl.sudoku_checkers import make_sudoku_feasibility_checker
from rl.task_config import get_task_config
from rl.training_setup import DummyPuzzleDataset
from rl.upi_trm_trainer import EXACT_CENTERING_TOLERANCE, UPITrmTrainer
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
    dataset_pool_sha256,
    dataset_puzzle_identifier_sha256s,
    dataset_sample_sha256s,
    ordered_record_sha256,
    validate_dataset_provenance,
)


class Phase4CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot support a publication record."""


PHASE4_FINAL_ENV_STEPS: int = 320_000
PHASE4_FINAL_EPISODES: int = 20_000
PHASE4_FINAL_REPLAY_SIZE: int = 100_000
PHASE4_EPISODE_LENGTH: int = 16


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


def _registered_phase4_dataset(seed: int) -> DummyPuzzleDataset:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DummyPuzzleDataset(ensure_sudoku_action_support=True)


def _validate_phase4_dataset_provenance(
    raw_provenance: Any,
    *,
    seed: int,
    rl_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    expected_dataset: DummyPuzzleDataset,
) -> Dict[str, Any]:
    try:
        provenance = validate_dataset_provenance(
            _require_mapping(raw_provenance, "dataset provenance")
        )
    except DatasetProvenanceError as exc:
        raise Phase4CheckpointError("Checkpoint dataset provenance is invalid.") from exc
    if provenance["dataset_builder"] != {
        "name": "rl.training_setup.DummyPuzzleDataset",
        "version": 2,
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
    expected_records = dataset_sample_sha256s(expected_dataset)
    expected_record_digest = ordered_record_sha256(expected_records)
    expected_identifier_digest = ordered_record_sha256(
        dataset_puzzle_identifier_sha256s(expected_dataset)
    )
    for split in ("train", "eval"):
        record_manifest = provenance["ordered_records"][split]
        if (
            record_manifest["record_sha256s"] != expected_records
            or record_manifest["ordered_sha256"] != expected_record_digest
        ):
            raise Phase4CheckpointError(
                "Checkpoint dataset records do not match the registered seeded builder."
            )
    if (
        metadata["train_pool_sha256"]
        != dataset_pool_sha256(expected_dataset, len(expected_dataset))
        or metadata["train_puzzle_identifier_ordered_sha256"]
        != expected_identifier_digest
    ):
        raise Phase4CheckpointError(
            "Checkpoint dataset metadata does not match the registered seeded builder."
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


def _validate_optimizer_state(
    value: Any,
    label: str,
    *,
    require_slots: bool = True,
    require_complete_slots: bool = True,
) -> None:
    state = _require_mapping(value, label)
    if set(state) != {"state", "param_groups"}:
        raise Phase4CheckpointError(f"{label} has an invalid field inventory.")
    optimizer_slots = state["state"]
    parameter_groups = state["param_groups"]
    if (
        not isinstance(optimizer_slots, Mapping)
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
    if require_slots and not optimizer_slots:
        raise Phase4CheckpointError(f"{label} has no optimizer parameter state.")
    parameter_ids = [
        parameter_id
        for group in parameter_groups
        for parameter_id in group["params"]
    ]
    optimizer_slot_ids = set(optimizer_slots)
    if len(parameter_ids) != len(set(parameter_ids)) or not (
        optimizer_slot_ids == set(parameter_ids)
        if require_complete_slots
        else optimizer_slot_ids.issubset(parameter_ids)
    ):
        raise Phase4CheckpointError(
            f"{label} contains invalid optimizer parameter identifiers."
        )


def _validate_loaded_adam_state(
    optimizer: torch.optim.Optimizer,
    *,
    expected_step_count: int,
    label: str,
    allowed_missing_parameters: frozenset[torch.nn.Parameter] = frozenset(),
) -> None:
    if expected_step_count == 0:
        if optimizer.state:
            raise Phase4CheckpointError(
                f"{label} has parameter state despite a zero optimizer-step count."
            )
        return
    optimizer_parameters = {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_parameters = optimizer_parameters - allowed_missing_parameters
    if (
        not allowed_missing_parameters.issubset(optimizer_parameters)
        or set(optimizer.state) != expected_parameters
    ):
        raise Phase4CheckpointError(
            f"{label} does not have the exact registered Adam slot inventory."
        )
    for group in optimizer.param_groups:
        expected_fields = {"step", "exp_avg", "exp_avg_sq"}
        if bool(group.get("amsgrad", False)):
            expected_fields.add("max_exp_avg_sq")
        for parameter in group["params"]:
            if parameter in allowed_missing_parameters:
                continue
            slot = optimizer.state.get(parameter)
            if not isinstance(slot, Mapping) or set(slot) != expected_fields:
                raise Phase4CheckpointError(
                    f"{label} is missing complete Adam state for a parameter."
                )
            step = slot["step"]
            if torch.is_tensor(step):
                if step.numel() != 1 or not bool(torch.isfinite(step).item()):
                    raise Phase4CheckpointError(
                        f"{label} has an invalid Adam step tensor."
                    )
                step_value = float(step.item())
            else:
                step_value = _require_finite_number(step, f"{label} Adam step")
            if step_value != float(expected_step_count):
                raise Phase4CheckpointError(
                    f"{label} Adam step disagrees with the trainer counter."
                )
            for field in expected_fields - {"step"}:
                tensor = slot[field]
                if (
                    not torch.is_tensor(tensor)
                    or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype
                    or not bool(torch.isfinite(tensor).all().item())
                ):
                    raise Phase4CheckpointError(
                        f"{label} has incompatible Adam tensor state."
                    )


def _validate_optimizer_param_groups(
    raw_state: Any,
    optimizer: torch.optim.Optimizer,
    *,
    label: str,
) -> None:
    state = _require_mapping(raw_state, label)
    raw_groups = state.get("param_groups")
    if not isinstance(raw_groups, list) or len(raw_groups) != len(
        optimizer.param_groups
    ):
        raise Phase4CheckpointError(
            f"{label} has the wrong optimizer parameter-group inventory."
        )
    dynamic_fields = {"params", "lr", "initial_lr"}
    registered_state_groups = optimizer.state_dict()["param_groups"]
    for raw_group, registered_group, registered_state_group in zip(
        raw_groups,
        optimizer.param_groups,
        registered_state_groups,
    ):
        if not isinstance(raw_group, Mapping):
            raise Phase4CheckpointError(
                f"{label} has an invalid optimizer parameter group."
            )
        if raw_group.get("params") != registered_state_group.get("params"):
            raise Phase4CheckpointError(
                f"{label} parameter order differs from the registered optimizer."
            )
        raw_semantics = {
            key: value
            for key, value in raw_group.items()
            if key not in dynamic_fields
        }
        registered_semantics = {
            key: value
            for key, value in registered_group.items()
            if key not in dynamic_fields
        }
        if raw_semantics != registered_semantics:
            raise Phase4CheckpointError(
                f"{label} optimizer hyperparameters differ from the registered run."
            )


def _scheduled_lr(
    rl_config: RLConfig,
    *,
    base_lr: float,
    step_count: int,
) -> float:
    warmup = int(rl_config.lr_warmup_steps)
    total = int(rl_config.num_train_steps)
    minimum = float(rl_config.lr_min_factor)
    schedule = str(rl_config.lr_schedule)
    if step_count < warmup:
        factor = (step_count + 1) / max(warmup, 1)
    else:
        progress = min((step_count - warmup) / max(total - warmup, 1), 1.0)
        if schedule == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = minimum + (1.0 - minimum) * decay
        elif schedule == "linear":
            factor = 1.0 - progress * (1.0 - minimum)
        else:
            factor = 1.0
    return base_lr * factor


def _validate_loaded_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    raw_state: Any,
    *,
    rl_config: RLConfig,
    base_lr: float,
    expected_step_count: int,
    label: str,
) -> None:
    state = _require_mapping(raw_state, label)
    registered_state = scheduler.state_dict()
    if set(state) != set(registered_state):
        raise Phase4CheckpointError(f"{label} has an invalid field inventory.")
    dynamic_fields = {"last_epoch", "_step_count", "_last_lr"}
    if any(
        state[field] != registered_state[field]
        for field in state
        if field not in dynamic_fields
    ):
        raise Phase4CheckpointError(
            f"{label} semantics differ from the registered scheduler."
        )
    scheduler.load_state_dict(state)
    loaded = scheduler.state_dict()
    _require_exact_int(
        loaded.get("last_epoch"),
        expected_step_count,
        f"{label} last epoch",
    )
    _require_exact_int(
        loaded.get("_step_count"),
        expected_step_count + 1,
        f"{label} internal step count",
    )
    base_lrs = loaded.get("base_lrs")
    last_lrs = loaded.get("_last_lr")
    if (
        not isinstance(base_lrs, list)
        or not isinstance(last_lrs, list)
        or len(base_lrs) != len(optimizer.param_groups)
        or len(last_lrs) != len(optimizer.param_groups)
    ):
        raise Phase4CheckpointError(f"{label} has an invalid LR inventory.")
    expected_lr = _scheduled_lr(
        rl_config,
        base_lr=base_lr,
        step_count=expected_step_count,
    )
    for group, saved_base, saved_last in zip(
        optimizer.param_groups,
        base_lrs,
        last_lrs,
    ):
        values = (
            (saved_base, base_lr),
            (group.get("initial_lr"), base_lr),
            (saved_last, expected_lr),
            (group.get("lr"), expected_lr),
        )
        if any(
            not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            for actual, expected in values
        ):
            raise Phase4CheckpointError(
                f"{label} learning rates disagree with the registered schedule."
            )


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4CheckpointError(f"{label} must be finite numeric data.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise Phase4CheckpointError(f"{label} must be finite numeric data.")
    return parsed


def _validate_rng_state(value: Any, *, cuda_device_count: int) -> None:
    rng_state = _require_mapping(value, "checkpoint RNG state")
    if set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise Phase4CheckpointError(
            "Checkpoint RNG state has an invalid field inventory."
        )
    torch_cpu = rng_state["torch_cpu"]
    if (
        not torch.is_tensor(torch_cpu)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
    ):
        raise Phase4CheckpointError(
            "Checkpoint CPU RNG state must be a nonempty byte tensor."
        )
    torch_cuda = rng_state["torch_cuda"]
    if (
        not isinstance(torch_cuda, list)
        or len(torch_cuda) != cuda_device_count
        or any(
            not torch.is_tensor(item)
            or item.device.type != "cpu"
            or item.dtype != torch.uint8
            or item.ndim != 1
            or item.numel() != 16
            or not item.is_contiguous()
            for item in torch_cuda
        )
    ):
        raise Phase4CheckpointError(
            "Checkpoint CUDA RNG state does not match the registered device topology."
        )
    for item in torch_cuda:
        philox_offset = int.from_bytes(
            bytes(item[8:16].tolist()),
            byteorder="little",
            signed=True,
        )
        if philox_offset < 0 or philox_offset % 4 != 0:
            raise Phase4CheckpointError(
                "Checkpoint CUDA RNG state has an invalid Philox offset."
            )

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(torch_cpu)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Phase4CheckpointError(
            "Checkpoint RNG state cannot be restored."
        ) from exc
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _validate_phase4_replay_payload(
    transition: Transition,
    expected_dataset: DummyPuzzleDataset,
) -> None:
    """Require one replay record to match the registered episodic update ABI."""

    expected_state_fields = {
        "inputs",
        "puzzle_identifiers",
        "initial_plan",
        "solution",
        "remaining_edits",
    }
    for label, state in (("x", transition.x), ("x_next", transition.x_next)):
        if not isinstance(state, Mapping) or set(state) != expected_state_fields:
            raise Phase4CheckpointError(
                f"Phase 4 replay {label} has an invalid state field inventory."
            )
        tensor_specs = {
            "inputs": ((16,), torch.long),
            "puzzle_identifiers": ((1,), torch.long),
            "initial_plan": ((16,), torch.long),
            "solution": ((16,), torch.long),
            "remaining_edits": ((), torch.long),
        }
        for field, (shape, dtype) in tensor_specs.items():
            tensor = state[field]
            if (
                not torch.is_tensor(tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
            ):
                raise Phase4CheckpointError(
                    f"Phase 4 replay {label}.{field} has an invalid tensor ABI."
                )
        for field in ("inputs", "initial_plan", "solution"):
            tensor = state[field]
            if bool(((tensor < 0) | (tensor >= 32)).any().item()):
                raise Phase4CheckpointError(
                    f"Phase 4 replay {label}.{field} is outside the registered token domain."
                )
        identifier = int(state["puzzle_identifiers"].item())
        if identifier < 0 or identifier >= 32:
            raise Phase4CheckpointError(
                f"Phase 4 replay {label} has an out-of-range puzzle identifier."
            )
        expected_sample = expected_dataset[identifier]
        if any(
            not torch.equal(state[field], expected_sample[field])
            for field in ("inputs", "initial_plan", "solution")
        ):
            raise Phase4CheckpointError(
                f"Phase 4 replay {label} does not match its registered dataset record."
            )

    for label, plan in (("y", transition.y), ("y_next", transition.y_next)):
        if (
            not isinstance(plan, torch.Tensor)
            or plan.device.type != "cpu"
            or plan.dtype != torch.long
            or tuple(plan.shape) != (16,)
            or bool(((plan < 0) | (plan >= 32)).any().item())
        ):
            raise Phase4CheckpointError(
                f"Phase 4 replay {label} has an invalid plan tensor."
            )

    action = transition.action
    if (
        not torch.is_tensor(action)
        or action.device.type != "cpu"
        or action.dtype != torch.long
        or action.numel() != 1
        or int(action.item()) < 0
        or int(action.item()) >= 512
    ):
        raise Phase4CheckpointError(
            "Phase 4 replay action is not a registered scalar action."
        )
    reward = transition.reward
    if (
        not torch.is_tensor(reward)
        or reward.device.type != "cpu"
        or reward.dtype != torch.float32
        or tuple(reward.shape) != (1,)
        or not bool(torch.isfinite(reward).all().item())
    ):
        raise Phase4CheckpointError(
            "Phase 4 replay reward has an invalid tensor ABI."
        )
    done = transition.done
    if (
        not torch.is_tensor(done)
        or done.device.type != "cpu"
        or done.dtype != torch.bool
        or tuple(done.shape) != (1,)
    ):
        raise Phase4CheckpointError(
            "Phase 4 replay terminal flag has an invalid tensor ABI."
        )
    behavior_log_prob = transition.behavior_log_prob
    if (
        not isinstance(behavior_log_prob, torch.Tensor)
        or behavior_log_prob.device.type != "cpu"
        or behavior_log_prob.dtype != torch.float32
        or behavior_log_prob.numel() != 1
        or not bool(torch.isfinite(behavior_log_prob).all().item())
        or float(behavior_log_prob.item()) > 0.0
    ):
        raise Phase4CheckpointError(
            "Phase 4 replay behavior log-probability is unavailable or invalid."
        )
    if transition.latent is not None or transition.next_latent is not None:
        raise Phase4CheckpointError(
            "Registered episodic Phase 4 replay cannot contain recurrent carries."
        )


def _checkpoint_values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(
                _checkpoint_values_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _checkpoint_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return bool(left == right)


def _phase4_environment(
    rl_config: RLConfig,
    *,
    dataset: Any = (),
) -> PlanEditEnv:
    """Construct the registered environment shell used for resume preflight."""

    environment = PlanEditEnv(
        dataset=dataset,
        checker=make_sudoku_feasibility_checker(
            w_v=float(rl_config.feasibility_violation_weight),
            w_z=float(rl_config.feasibility_zerocand_weight),
        ),
        config=PlanEditEnvConfig(
            max_edits=int(rl_config.max_edits),
            gamma=float(rl_config.gamma),
            reward_shaping=bool(rl_config.reward_shaping),
            task_type=str(rl_config.task_name),
            vocab_size=32,
            solved_threshold=rl_config.solved_threshold,
            stop_action_mode=rl_config.stop_action_mode,
            stop_action_penalty=float(rl_config.stop_action_penalty),
            C_max=float(rl_config.C_max),
            enable_undo=False,
            fail_terminal_reward=float(rl_config.fail_terminal_reward),
            solve_terminal_reward=float(rl_config.solve_terminal_reward),
            disable_constraint_masking=bool(
                rl_config.disable_constraint_masking
            ),
        ),
        task_config=get_task_config(
            "sudoku",
            disable_constraint_masking=bool(
                rl_config.disable_constraint_masking
            ),
        ),
    )
    environment.set_stop_action_id(512)
    return environment


def _validate_phase4_replay_semantics(
    transitions: list[Transition],
    expected_dataset: DummyPuzzleDataset,
    rl_config: RLConfig,
) -> None:
    """Replay every retained episode through the exact registered environment."""

    environment = _phase4_environment(
        rl_config,
        dataset=expected_dataset,
    )
    active_episode_id: Optional[int] = None
    for transition in transitions:
        if transition.episode_id != active_episode_id:
            if active_episode_id is not None and not environment.done:
                raise Phase4CheckpointError(
                    "Phase 4 replay starts a new episode before the previous one terminates."
                )
            if (
                active_episode_id is not None
                and transition.episode_id != active_episode_id + 1
            ):
                raise Phase4CheckpointError(
                    "Phase 4 replay episode identifiers are not consecutive."
                )
            if transition.timestep != 0:
                raise Phase4CheckpointError(
                    "Phase 4 replay episode does not start at timestep zero."
                )
            identifier = int(
                transition.x["puzzle_identifiers"].reshape(()).item()
            )
            try:
                environment.reset(idx=identifier)
            except (AssertionError, IndexError, RuntimeError, TypeError, ValueError) as exc:
                raise Phase4CheckpointError(
                    "Phase 4 replay episode cannot be reset from the registered dataset."
                ) from exc
            active_episode_id = transition.episode_id

        if environment.done or transition.timestep != environment.step_count:
            raise Phase4CheckpointError(
                "Phase 4 replay timestep disagrees with the registered environment."
            )
        if not _checkpoint_values_equal(
            transition.x,
            environment.x,
        ) or not _checkpoint_values_equal(
            transition.y,
            environment.y,
        ):
            raise Phase4CheckpointError(
                "Phase 4 replay current state disagrees with the registered environment."
            )

        action = int(transition.action.item())
        action_mask = environment.get_action_mask()
        if (
            action_mask is None
            or action_mask.dtype != torch.bool
            or tuple(action_mask.shape) != (513,)
            or action < 0
            or action >= action_mask.numel()
            or not bool(action_mask.reshape(-1)[action].item())
        ):
            raise Phase4CheckpointError(
                "Phase 4 replay action is masked by the registered environment."
            )
        try:
            (expected_x_next, expected_y_next), reward, done, info = (
                environment.step(action)
            )
        except (AssertionError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise Phase4CheckpointError(
                "Phase 4 replay action cannot be executed by the registered environment."
            ) from exc

        if not _checkpoint_values_equal(
            transition.x_next,
            expected_x_next,
        ) or not _checkpoint_values_equal(
            transition.y_next,
            expected_y_next,
        ):
            raise Phase4CheckpointError(
                "Phase 4 replay successor disagrees with the registered environment."
            )
        expected_reward = torch.as_tensor(
            reward,
            dtype=torch.float32,
        ).view(1)
        if not _checkpoint_values_equal(transition.reward, expected_reward):
            raise Phase4CheckpointError(
                "Phase 4 replay reward disagrees with the registered environment."
            )
        expected_done = torch.tensor([done], dtype=torch.bool)
        if not _checkpoint_values_equal(transition.done, expected_done):
            raise Phase4CheckpointError(
                "Phase 4 replay terminal flag disagrees with the registered environment."
            )
        expected_reason = (
            str(info["done_reason"])
            if done and info.get("done_reason") is not None
            else None
        )
        if transition.terminal_reason != expected_reason:
            raise Phase4CheckpointError(
                "Phase 4 replay terminal reason disagrees with the registered environment."
            )

    if not environment.done:
        raise Phase4CheckpointError(
            "Phase 4 replay ends before its final registered episode terminates."
        )


def _preflight_phase4_resume_objects(
    checkpoint: Mapping[str, Any],
    *,
    model: TinyRecursiveReasoningModel_ACTV1,
    rl_config: RLConfig,
) -> None:
    """Load every mutable resume object into an isolated registered shell."""

    try:
        trainer = UPITrmTrainer(
            model=model,
            env=_phase4_environment(rl_config),
            rl_cfg=rl_config,
            device=torch.device("cpu"),
        )
        trainer.policy_model_old.load_state_dict(
            checkpoint["policy_model_old_state_dict"], strict=True
        )
        trainer.policy_model_candidate.load_state_dict(
            checkpoint["policy_model_candidate_state_dict"], strict=True
        )
        trainer.target_model.load_state_dict(
            checkpoint["target_model_state_dict"], strict=True
        )

        trainer_state = checkpoint["trainer_state"]
        value_unused_names = {
            "inner.lm_head.weight",
            "inner.q_head.weight",
            "inner.q_head.bias",
        }
        value_named_parameters = dict(trainer.model.named_parameters())
        if not value_unused_names.issubset(value_named_parameters):
            raise Phase4CheckpointError(
                "Registered Phase 4 value-optimizer ownership has changed."
            )
        value_unused_parameters = frozenset(
            value_named_parameters[name] for name in value_unused_names
        )
        optimizer_pairs = (
            (
                trainer.value_opt,
                "value_optimizer_state_dict",
                int(trainer_state["value_optimizer_step_count"]),
                value_unused_parameters,
            ),
            (
                trainer.policy_opt,
                "policy_optimizer_state_dict",
                int(trainer_state["policy_optimizer_step_count"]),
                frozenset(),
            ),
            (
                trainer.old_policy_distill_opt,
                "old_policy_distill_optimizer_state_dict",
                int(trainer_state["distill_optimizer_step_count"]),
                frozenset(),
            ),
        )
        for (
            optimizer,
            field,
            expected_step_count,
            allowed_missing_parameters,
        ) in optimizer_pairs:
            if optimizer is None or field not in checkpoint:
                raise Phase4CheckpointError(
                    f"Phase 4 checkpoint is missing required field {field!r}."
                )
            _validate_optimizer_param_groups(
                checkpoint[field],
                optimizer,
                label=field,
            )
            optimizer.load_state_dict(checkpoint[field])
            _validate_loaded_adam_state(
                optimizer,
                expected_step_count=expected_step_count,
                label=field,
                allowed_missing_parameters=allowed_missing_parameters,
            )

        scheduler_pairs = (
            (
                trainer.value_scheduler,
                trainer.value_opt,
                "value_scheduler_state_dict",
                float(rl_config.value_lr),
                int(trainer_state["value_optimizer_step_count"]),
            ),
            (
                trainer.policy_scheduler,
                trainer.policy_opt,
                "policy_scheduler_state_dict",
                float(rl_config.policy_lr),
                int(trainer_state["policy_optimizer_step_count"]),
            ),
        )
        for scheduler, optimizer, field, base_lr, expected_step_count in scheduler_pairs:
            if scheduler is None or field not in checkpoint:
                raise Phase4CheckpointError(
                    f"Phase 4 checkpoint is missing required field {field!r}."
                )
            _validate_loaded_scheduler(
                scheduler,
                optimizer,
                checkpoint[field],
                rl_config=rl_config,
                base_lr=base_lr,
                expected_step_count=expected_step_count,
                label=field,
            )

        replay = ReplayBuffer(
            capacity=int(checkpoint["replay_capacity"]),
            theorem_facing=True,
        )
        for transition in checkpoint["replay_transitions"]:
            replay.add(transition)
        trainer.replay.clear()
        for transition in replay.storage:
            trainer.replay.add(transition)

        trainer._next_episode_id = int(trainer_state["next_episode_id"])
        trainer.env.load_checkpoint_state(trainer_state["environment_state"])
        trainer.load_collection_checkpoint_state(
            trainer_state["collection_state"]
        )
        trainer.restore_exact_centering_checkpoint_state(
            trainer_state["exact_centering_state"],
            validate_only=True,
        )
        trainer.device = torch.device(checkpoint["execution_device"])
        trainer.restore_compute_accounting_checkpoint_state(
            trainer_state["compute_accounting_state"],
            validate_only=True,
        )
    except Phase4CheckpointError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError, ReplayIntegrityError) as exc:
        raise Phase4CheckpointError(
            "Phase 4 checkpoint resume state failed isolated preflight."
        ) from exc


def _validate_full_phase4_training_state(
    checkpoint: Mapping[str, Any],
    rl_config: Mapping[str, Any],
    expected_dataset: DummyPuzzleDataset,
    registered_rl_config: RLConfig,
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
    if progress_env_steps != PHASE4_FINAL_ENV_STEPS:
        raise Phase4CheckpointError(
            "Checkpoint environment-step count does not match the registered Phase 4 design."
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
        "next_episode_id",
        "train_step_count",
        "env_step_count",
        "optimizer_step_count",
        "value_optimizer_step_count",
        "policy_optimizer_step_count",
        "distill_optimizer_step_count",
        "puzzle_optimizer_step_count",
        "kl_coef",
        "term_stats",
        "debug_episode_lengths",
        "debug_episode_returns",
        "debug_stop_probs",
        "debug_score_changes",
        "drift_values",
        "plan_changes",
        "value_of_memory",
        "opnorm_clamp_warned",
        "collection_state",
        "environment_state",
        "exact_centering_state",
        "compute_accounting_state",
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
    next_episode_id = _require_nonnegative_int(
        trainer_state["next_episode_id"],
        "Trainer next episode ID",
    )
    if next_episode_id != PHASE4_FINAL_EPISODES:
        raise Phase4CheckpointError(
            "Trainer next episode ID does not match the registered Phase 4 design."
        )
    if progress_env_steps != trainer_env_steps:
        raise Phase4CheckpointError(
            "Checkpoint environment-step counters disagree."
        )
    if progress_optimizer_steps != trainer_optimizer_steps:
        raise Phase4CheckpointError(
            "Checkpoint optimizer-step counters disagree."
        )
    if progress_optimizer_steps != PHASE4_CHECKPOINT_STEP:
        raise Phase4CheckpointError(
            "Checkpoint optimizer-step count does not match the publication step."
        )
    for field in (
        "value_optimizer_step_count",
        "policy_optimizer_step_count",
        "distill_optimizer_step_count",
        "puzzle_optimizer_step_count",
    ):
        value = _require_nonnegative_int(
            trainer_state[field],
            f"Trainer {field.replace('_', ' ')}",
        )
        if value > PHASE4_CHECKPOINT_STEP:
            raise Phase4CheckpointError(
                f"Trainer {field.replace('_', ' ')} exceeds the publication step."
            )
    if (
        trainer_state["value_optimizer_step_count"] != PHASE4_CHECKPOINT_STEP
        or trainer_state["policy_optimizer_step_count"]
        != PHASE4_CHECKPOINT_STEP
    ):
        raise Phase4CheckpointError(
            "Phase 4 value and policy optimizer counts must match every registered outer update."
        )
    if (
        not bool(rl_config.get("distill_mixture_policy"))
        and trainer_state["distill_optimizer_step_count"] != 0
    ):
        raise Phase4CheckpointError(
            "Trainer records distillation steps for a registered run with distillation disabled."
        )
    if trainer_state["puzzle_optimizer_step_count"] != 0:
        raise Phase4CheckpointError(
            "Trainer records puzzle-embedding optimizer steps for an architecture without them."
        )
    if not math.isclose(
        _require_finite_number(
            trainer_state["kl_coef"],
            "Trainer KL coefficient",
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise Phase4CheckpointError(
            "Trainer KL coefficient changed even though the registered Phase 4 design disables KL adaptation."
        )
    term_stats = _require_mapping(
        trainer_state["term_stats"],
        "trainer termination statistics",
    )
    if set(term_stats) != {"stop", "solved", "budget"} or any(
        _require_finite_number(value, f"Trainer termination count {name}") < 0
        for name, value in term_stats.items()
    ):
        raise Phase4CheckpointError(
            "Trainer termination statistics are incomplete or negative."
        )
    if any(float(value) != 0.0 for value in term_stats.values()):
        raise Phase4CheckpointError(
            "Trainer termination statistics were not reset at the registered outer-step boundary."
        )
    for field in (
        "debug_episode_lengths",
        "debug_episode_returns",
        "debug_stop_probs",
        "debug_score_changes",
        "drift_values",
        "plan_changes",
        "value_of_memory",
    ):
        values = trainer_state[field]
        if not isinstance(values, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise Phase4CheckpointError(
                f"Trainer {field.replace('_', ' ')} must be finite numeric data."
            )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in trainer_state["debug_episode_lengths"]
    ):
        raise Phase4CheckpointError(
            "Trainer debug episode lengths must be nonnegative integers."
        )
    if not isinstance(trainer_state["opnorm_clamp_warned"], bool):
        raise Phase4CheckpointError(
            "Trainer operator-norm warning state must be boolean."
        )
    if not isinstance(trainer_state["collection_state"], Mapping) or not isinstance(
        trainer_state["environment_state"], Mapping
    ):
        raise Phase4CheckpointError(
            "Checkpoint collector and environment state must be mappings."
        )
    collection_state = trainer_state["collection_state"]
    if (
        set(collection_state)
        != {"schema_version", "completed_episodes_since_update", "active_episode"}
        or collection_state.get("schema_version") != 1
        or collection_state.get("completed_episodes_since_update") != 0
        or collection_state.get("active_episode") is not None
    ):
        raise Phase4CheckpointError(
            "Checkpoint collector is not at the registered outer-step boundary."
        )
    environment_state = trainer_state["environment_state"]
    if (
        environment_state.get("schema_version") != 1
        or environment_state.get("initialized") is not True
        or environment_state.get("done") is not True
        or environment_state.get("step_count") != PHASE4_EPISODE_LENGTH
    ):
        raise Phase4CheckpointError(
            "Checkpoint environment is not at the registered terminal edit boundary."
        )
    _require_exact_int(
        trainer_state["terminal_reason_replay_version"],
        1,
        "Terminal-reason replay version",
    )

    expected_exact_centering_state = {
        "schema_version": 1,
        "tolerance": EXACT_CENTERING_TOLERANCE,
        "batch_count": 0,
        "maximum_observed": None,
        "history_complete": True,
    }
    if trainer_state["exact_centering_state"] != expected_exact_centering_state:
        raise Phase4CheckpointError(
            "Exact-centering evidence changed even though the registered Phase 4 design disables exact baseline summation."
        )

    execution_device = checkpoint["execution_device"]
    if execution_device != "cuda:0":
        raise Phase4CheckpointError(
            "Phase 4 publication checkpoints require the registered cuda:0 training device."
        )
    _validate_rng_state(checkpoint["rng_state"], cuda_device_count=1)

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
        or replay_capacity != PHASE4_FINAL_REPLAY_SIZE
        or replay_size != PHASE4_FINAL_REPLAY_SIZE
        or replay_size != len(replay_transitions)
        or any(
            not isinstance(transition, Transition)
            for transition in replay_transitions
        )
    ):
        raise Phase4CheckpointError(
            "Checkpoint replay size or capacity is inconsistent."
        )
    for transition in replay_transitions:
        _validate_phase4_replay_payload(transition, expected_dataset)
    _validate_phase4_replay_semantics(
        replay_transitions,
        expected_dataset,
        registered_rl_config,
    )
    replay_tail = replay_transitions[-1]
    replay_head = replay_transitions[0]
    expected_first_episode = PHASE4_FINAL_EPISODES - (
        PHASE4_FINAL_REPLAY_SIZE // PHASE4_EPISODE_LENGTH
    )
    if (
        replay_head.episode_id != expected_first_episode
        or replay_head.timestep != 0
        or replay_tail.episode_id != PHASE4_FINAL_EPISODES - 1
        or replay_tail.timestep != PHASE4_EPISODE_LENGTH - 1
    ):
        raise Phase4CheckpointError(
            "Checkpoint replay endpoints do not match the registered final ring buffer."
        )
    tail_done = bool(replay_tail.done.reshape(-1)[0].item())
    expected_next_episode_id = (
        replay_tail.episode_id + 1 if tail_done else replay_tail.episode_id
    )
    if next_episode_id != expected_next_episode_id:
        raise Phase4CheckpointError(
            "Trainer next episode ID disagrees with the replay tail."
        )
    if not _checkpoint_values_equal(
        environment_state.get("x"),
        replay_tail.x_next,
    ) or not _checkpoint_values_equal(
        environment_state.get("y"),
        replay_tail.y_next,
    ):
        raise Phase4CheckpointError(
            "Checkpoint terminal environment state disagrees with the replay tail."
        )
    _validate_optimizer_state(
        checkpoint["value_optimizer_state_dict"],
        "value optimizer state",
        require_complete_slots=False,
    )
    _validate_optimizer_state(
        checkpoint["policy_optimizer_state_dict"],
        "policy optimizer state",
    )
    _validate_optimizer_state(
        checkpoint["old_policy_distill_optimizer_state_dict"],
        "old-policy distillation optimizer state",
        require_slots=trainer_state["distill_optimizer_step_count"] > 0,
        require_complete_slots=(
            trainer_state["distill_optimizer_step_count"] > 0
        ),
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
    expected_training_runtime_sha256: str,
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
    authorized_training_runtime_sha256 = _require_sha256(
        expected_training_runtime_sha256,
        "Authorized training runtime artifact digest",
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
        "old_policy_distill_optimizer_state_dict",
        "value_scheduler_state_dict",
        "policy_scheduler_state_dict",
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
    expected_dataset = _registered_phase4_dataset(seed)
    dataset_provenance = _validate_phase4_dataset_provenance(
        checkpoint["dataset_provenance"],
        seed=seed,
        rl_config=rl_config,
        model_config=model_config,
        expected_dataset=expected_dataset,
    )
    _validate_full_phase4_training_state(
        checkpoint,
        rl_config,
        expected_dataset,
        rl_config_object,
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
    if training_runtime_artifact_sha256 != authorized_training_runtime_sha256:
        raise Phase4CheckpointError(
            "Training runtime artifact does not match the externally authorized digest."
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
    _preflight_phase4_resume_objects(
        checkpoint,
        model=model,
        rl_config=rl_config_object,
    )
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
    expected_training_runtime_sha256: str,
    device: str | torch.device = "cpu",
) -> int:
    """Revalidate every checkpoint named by an already validated summary."""

    authorized_producer_source = _validate_producer_source(
        expected_producer_source,
        "Authorized training producer source",
    )
    authorized_training_runtime_sha256 = _require_sha256(
        expected_training_runtime_sha256,
        "Authorized training runtime artifact digest",
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
        if not isinstance(
            checkpoint_relpath, str
        ) or checkpoint_relpath != phase4_checkpoint_relpath(condition, seed):
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
            expected_training_runtime_sha256=(
                authorized_training_runtime_sha256
            ),
            device=device,
            expected_identity=expected_identity,
        )
    return len(runs)
