#!/usr/bin/env fbpython
"""Strict, isolated PPO continuation checkpoints for the Stage 0 smoke.

The normal training checkpoint deliberately treats baseline trainers as
weights-only.  Stage 0 needs a real two-process continuation check, so this
module owns a separate schema.  It is not accepted by the confirmatory or
Phase 4 checkpoint readers.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from policy_improvement_checkpoint_allowlist import (
    load_data_only_checkpoint,
    UnsafeCheckpointPayloadError,
)
from utils.compute_accounting import (
    add_model_counters,
    current_cuda_memory_peaks,
    execution_device_identity,
    process_peak_rss_bytes,
    restore_model_compute_state,
    validate_model_compute_state,
    validate_model_counters,
)
from utils.run_identity import canonical_json_sha256


PPO_SMOKE_CHECKPOINT_SCHEMA_VERSION = 1
_READ_SIZE = 1024 * 1024


class PolicyImprovementSmokeCheckpointError(RuntimeError):
    """Raised when a smoke checkpoint cannot establish exact continuation."""


def _clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _require_finite_tensor_tree(value: object, *, label: str) -> None:
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise PolicyImprovementSmokeCheckpointError(
                f"PPO smoke {label} contains a non-finite tensor."
            )
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_tensor_tree(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_tensor_tree(item, label=label)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke RNG state has an invalid inventory."
        )
    try:
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.random.set_rng_state(value["torch_cpu"])
        cuda_state = value["torch_cuda"]
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG state cannot be restored on this host.")
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke RNG state cannot be restored."
        ) from exc


def _capture_gradients(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _restore_gradients(
    module: torch.nn.Module, value: object, *, validate_only: bool
) -> None:
    parameters = dict(module.named_parameters())
    if not isinstance(value, Mapping) or set(value) != set(parameters):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke parameter-gradient inventory differs from the model."
        )
    for name, parameter in parameters.items():
        gradient = value[name]
        if gradient is not None and (
            not torch.is_tensor(gradient)
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise PolicyImprovementSmokeCheckpointError(
                f"PPO smoke gradient {name!r} is invalid."
            )
        if not validate_only:
            parameter.grad = (
                None
                if gradient is None
                else gradient.detach().to(parameter.device).clone()
            )


def _positive_or_zero_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyImprovementSmokeCheckpointError(
            f"PPO smoke field {name!r} must be a nonnegative integer."
        )
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyImprovementSmokeCheckpointError(
            f"PPO smoke field {name!r} must be numeric."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PolicyImprovementSmokeCheckpointError(
            f"PPO smoke field {name!r} must be finite and nonnegative."
        )
    return result


def _capture_rollout_buffer(trainer: Any) -> dict[str, Any]:
    buffer = trainer.rollout_buffer
    fields = (
        "x_list",
        "y_list",
        "actions",
        "log_probs",
        "rewards",
        "dones",
        "values",
        "action_masks",
    )
    return {field: _clone(getattr(buffer, field)) for field in fields}


def _validate_rollout_buffer(value: object) -> dict[str, Any]:
    expected = {
        "x_list",
        "y_list",
        "actions",
        "log_probs",
        "rewards",
        "dones",
        "values",
        "action_masks",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout buffer has an invalid inventory."
        )
    lengths = []
    for field in expected:
        items = value[field]
        if not isinstance(items, list):
            raise PolicyImprovementSmokeCheckpointError(
                f"PPO smoke rollout field {field!r} is not a list."
            )
        lengths.append(len(items))
    if len(set(lengths)) != 1:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout fields have different lengths."
        )
    if any(not isinstance(item, bool) for item in value["dones"]):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout terminal flags are invalid."
        )
    for field in ("actions", "log_probs", "values"):
        if any(not torch.is_tensor(item) for item in value[field]):
            raise PolicyImprovementSmokeCheckpointError(
                f"PPO smoke rollout {field!r} contains a non-tensor."
            )
    if any(not isinstance(item, Mapping) for item in value["x_list"]):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout states must be mappings."
        )
    if any(not torch.is_tensor(item) for item in value["y_list"]):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout plans must be tensors."
        )
    if any(
        item is not None and (not torch.is_tensor(item) or item.dtype != torch.bool)
        for item in value["action_masks"]
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout action masks are invalid."
        )
    if any(
        not isinstance(reward, (int, float)) or not math.isfinite(float(reward))
        for reward in value["rewards"]
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke rollout rewards are invalid."
        )
    return value


def _restore_rollout_buffer(trainer: Any, value: object) -> None:
    canonical = _validate_rollout_buffer(value)
    trainer.rollout_buffer.clear()
    for field, items in canonical.items():
        setattr(trainer.rollout_buffer, field, _clone(items))


def _tensor_tree_equal(left: object, right: object) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _tensor_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tensor_tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _validate_episode_state(value: object, environment_state: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "current_x",
        "current_y",
        "episode_rewards",
    }:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke live-episode state has an invalid inventory."
        )
    rewards = value["episode_rewards"]
    if not isinstance(rewards, list) or any(
        not isinstance(item, (int, float)) or not math.isfinite(float(item))
        for item in rewards
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke live-episode rewards are invalid."
        )
    if not isinstance(environment_state, dict):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke environment state is invalid."
        )
    initialized = environment_state.get("initialized") is True
    if initialized != (value["current_x"] is not None):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke environment and trainer episode initialization differ."
        )
    if initialized and (
        value["current_y"] is None
        or not _tensor_tree_equal(value["current_x"], environment_state.get("x"))
        or not _tensor_tree_equal(value["current_y"], environment_state.get("y"))
        or len(rewards) != int(environment_state.get("step_count", -1))
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke live episode differs from the environment checkpoint."
        )
    if not initialized and (value["current_y"] is not None or rewards):
        raise PolicyImprovementSmokeCheckpointError(
            "Uninitialized PPO smoke episode contains mutable state."
        )
    return value


def _capture_compute_state(trainer: Any) -> dict[str, Any]:
    return dict(trainer.compute_accounting_checkpoint_state())


def _restore_compute_state(trainer: Any, value: object, *, validate_only: bool) -> None:
    common_fields = {
        "schema_version",
        "model_compute_state",
        "training_model_work",
        "evaluation_model_work",
        "training_wall_time_seconds",
        "evaluation_wall_time_seconds",
        "peak_process_rss_bytes",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    }
    if not isinstance(value, dict):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke compute state has an invalid inventory."
        )
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in {1, 2}:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke compute state has an invalid inventory."
        )
    expected = common_fields | (
        {"execution_device_identity"} if schema_version == 2 else set()
    )
    if set(value) != expected:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke compute state has an invalid inventory."
        )
    if schema_version == 2 and value["execution_device_identity"] != (
        execution_device_identity(trainer.device)
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke compute state names another execution device."
        )
    training = validate_model_counters(
        value["training_model_work"], name="training_model_work"
    )
    evaluation = validate_model_counters(
        value["evaluation_model_work"], name="evaluation_model_work"
    )
    canonical_models = validate_model_compute_state(
        trainer._model_roles(), value["model_compute_state"]
    )
    raw_total = validate_model_counters(
        {field: 0 for field in training}, name="raw_total"
    )
    for item in canonical_models:
        raw_total = add_model_counters(raw_total, item["counters"])
    if raw_total != add_model_counters(training, evaluation):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke phase accounting differs from model counters."
        )
    training_wall = _finite_nonnegative(
        value["training_wall_time_seconds"], name="training_wall_time_seconds"
    )
    evaluation_wall = _finite_nonnegative(
        value["evaluation_wall_time_seconds"], name="evaluation_wall_time_seconds"
    )
    peak_rss = _positive_or_zero_int(
        value["peak_process_rss_bytes"], name="peak_process_rss_bytes"
    )
    cuda_values = []
    for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        item = value[field]
        if item is not None:
            item = _positive_or_zero_int(item, name=field)
        cuda_values.append(item)
    if trainer.device.type == "cpu" and cuda_values != [None, None]:
        raise PolicyImprovementSmokeCheckpointError(
            "CPU PPO smoke checkpoints cannot contain CUDA peaks."
        )
    if trainer.device.type == "cuda" and any(item is None for item in cuda_values):
        raise PolicyImprovementSmokeCheckpointError(
            "CUDA PPO smoke checkpoint is missing memory peaks."
        )
    if validate_only:
        return
    restore_model_compute_state(trainer._model_roles(), canonical_models)
    trainer._training_model_work = training
    trainer._evaluation_model_work = evaluation
    trainer._training_wall_time_seconds = training_wall
    trainer._evaluation_wall_time_seconds = evaluation_wall
    trainer._peak_process_rss_bytes = max(peak_rss, process_peak_rss_bytes())
    current_allocated, current_reserved = current_cuda_memory_peaks(trainer.device)
    if trainer.device.type == "cuda":
        assert cuda_values[0] is not None and cuda_values[1] is not None
        assert current_allocated is not None and current_reserved is not None
        trainer._peak_cuda_allocated_bytes = max(cuda_values[0], current_allocated)
        trainer._peak_cuda_reserved_bytes = max(cuda_values[1], current_reserved)
    else:
        trainer._peak_cuda_allocated_bytes = None
        trainer._peak_cuda_reserved_bytes = None


def _optimizer_group_signature(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke optimizer state must be a mapping."
        )
    groups = value.get("param_groups")
    if not isinstance(groups, list):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke optimizer parameter groups are invalid."
        )
    signature: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise PolicyImprovementSmokeCheckpointError(
                "PPO smoke optimizer parameter group is invalid."
            )
        signature.append(
            {key: _clone(item) for key, item in group.items() if key != "params"}
            | {"parameter_count": len(group["params"])}
        )
    return signature


def _validate_optimizer_state(trainer: Any) -> None:
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            state = trainer.optimizer.state.get(parameter, {})
            if not isinstance(state, dict):
                raise PolicyImprovementSmokeCheckpointError(
                    "PPO smoke optimizer parameter state is invalid."
                )
            for name, item in state.items():
                if torch.is_tensor(item):
                    if not bool(torch.isfinite(item).all().item()):
                        raise PolicyImprovementSmokeCheckpointError(
                            f"PPO smoke optimizer tensor {name!r} is non-finite."
                        )
                    if item.numel() != 1 and item.shape != parameter.shape:
                        raise PolicyImprovementSmokeCheckpointError(
                            f"PPO smoke optimizer tensor {name!r} has the wrong shape."
                        )


def build_ppo_smoke_checkpoint(
    trainer: Any,
    *,
    identity: Mapping[str, object],
    parent_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    """Capture every mutable PPO continuation input at an idle boundary."""

    if type(trainer).__name__ != "PPOTrainer":
        raise PolicyImprovementSmokeCheckpointError(
            "The PPO smoke schema accepts only PPOTrainer."
        )
    if bool(getattr(trainer, "_train_step_active", False)):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoints require an idle trainer boundary."
        )
    if len(trainer.rollout_buffer) != trainer.config.num_steps:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoints require a complete just-optimized rollout."
        )
    if parent_checkpoint_sha256 is not None and (
        not isinstance(parent_checkpoint_sha256, str)
        or len(parent_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in parent_checkpoint_sha256
        )
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke parent checkpoint digest is invalid."
        )
    canonical_identity = dict(identity)
    identity_sha256 = canonical_json_sha256(canonical_identity)
    environment_state = trainer.env.checkpoint_state()
    payload = {
        "checkpoint_schema_version": PPO_SMOKE_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "policy_improvement_v1_ppo_exact_continuation",
        "checkpoint_phase": "idle_after_complete_ppo_update",
        "identity": canonical_identity,
        "identity_sha256": identity_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "model_state_dict": _clone(trainer.model.state_dict()),
        "model_training": bool(trainer.model.training),
        "parameter_gradients": _capture_gradients(trainer.model),
        "optimizer_state_dict": _clone(trainer.optimizer.state_dict()),
        "trainer_config": dict(vars(trainer.config)),
        "progress": {
            "train_steps": int(trainer._train_step_count),
            "environment_interactions": int(trainer._env_step_count),
            "optimizer_steps": int(trainer._optimizer_step_count),
            "episodes": int(trainer._episode_count),
        },
        "termination_counts": dict(trainer.term_stats),
        "environment_state": environment_state,
        "episode_state": {
            "current_x": _clone(trainer._current_x),
            "current_y": _clone(trainer._current_y),
            "episode_rewards": list(trainer._episode_rewards),
        },
        "rollout_buffer": _capture_rollout_buffer(trainer),
        "compute_accounting_state": _capture_compute_state(trainer),
        "rng_state": _capture_rng_state(),
    }
    validate_ppo_smoke_checkpoint(
        payload,
        trainer,
        expected_identity=canonical_identity,
        validate_only=True,
    )
    return payload


def validate_ppo_smoke_checkpoint(
    value: object,
    trainer: Any,
    *,
    expected_identity: Mapping[str, object],
    validate_only: bool,
) -> None:
    """Validate on a fresh probe, then optionally restore that same fresh object."""

    expected_fields = {
        "checkpoint_schema_version",
        "checkpoint_kind",
        "checkpoint_phase",
        "identity",
        "identity_sha256",
        "parent_checkpoint_sha256",
        "model_state_dict",
        "model_training",
        "parameter_gradients",
        "optimizer_state_dict",
        "trainer_config",
        "progress",
        "termination_counts",
        "environment_state",
        "episode_state",
        "rollout_buffer",
        "compute_accounting_state",
        "rng_state",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoint field inventory differs."
        )
    if (
        value["checkpoint_schema_version"] != PPO_SMOKE_CHECKPOINT_SCHEMA_VERSION
        or value["checkpoint_kind"] != "policy_improvement_v1_ppo_exact_continuation"
        or value["checkpoint_phase"] != "idle_after_complete_ppo_update"
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoint header is unsupported."
        )
    if value["identity"] != dict(expected_identity) or value[
        "identity_sha256"
    ] != canonical_json_sha256(dict(expected_identity)):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoint identity differs from the active row."
        )
    parent_digest = value["parent_checkpoint_sha256"]
    if parent_digest is not None and (
        not isinstance(parent_digest, str)
        or len(parent_digest) != 64
        or any(character not in "0123456789abcdef" for character in parent_digest)
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke parent checkpoint digest is invalid."
        )
    if not isinstance(value["model_training"], bool):
        raise PolicyImprovementSmokeCheckpointError("PPO smoke model mode is invalid.")
    if value["trainer_config"] != dict(vars(trainer.config)):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke trainer configuration differs."
        )
    progress = value["progress"]
    if not isinstance(progress, dict) or set(progress) != {
        "train_steps",
        "environment_interactions",
        "optimizer_steps",
        "episodes",
    }:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke progress inventory differs."
        )
    canonical_progress = {
        name: _positive_or_zero_int(item, name=f"progress.{name}")
        for name, item in progress.items()
    }
    if canonical_progress["environment_interactions"] != (
        canonical_progress["train_steps"] * int(trainer.config.num_steps)
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke interaction and rollout counts disagree."
        )
    rollout_steps = int(trainer.config.num_steps)
    minibatch_size = max(rollout_steps // int(trainer.config.num_minibatches), 1)
    updates_per_rollout = int(trainer.config.num_epochs) * math.ceil(
        rollout_steps / minibatch_size
    )
    if canonical_progress["optimizer_steps"] != (
        canonical_progress["train_steps"] * updates_per_rollout
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke optimizer and rollout counts disagree."
        )
    counts = value["termination_counts"]
    if not isinstance(counts, dict) or set(counts) != {"stop", "solved", "budget"}:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke termination-count inventory differs."
        )
    canonical_counts = {
        reason: _positive_or_zero_int(counts[reason], name=f"term.{reason}")
        for reason in counts
    }
    if sum(canonical_counts.values()) != canonical_progress["episodes"]:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke termination counts differ from completed episodes."
        )

    _require_finite_tensor_tree(value["model_state_dict"], label="model state")
    model_probe = copy.deepcopy(trainer.model)
    try:
        model_probe.load_state_dict(value["model_state_dict"], strict=True)
        _restore_gradients(
            model_probe, value["parameter_gradients"], validate_only=False
        )
    except Exception as exc:
        if isinstance(exc, PolicyImprovementSmokeCheckpointError):
            raise
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke model state is incompatible."
        ) from exc
    optimizer_state = value["optimizer_state_dict"]
    if _optimizer_group_signature(optimizer_state) != _optimizer_group_signature(
        trainer.optimizer.state_dict()
    ):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke optimizer configuration differs."
        )
    optimizer_probe = copy.deepcopy(trainer.optimizer)
    try:
        optimizer_probe.load_state_dict(optimizer_state)
        original = trainer.optimizer
        trainer.optimizer = optimizer_probe
        try:
            _validate_optimizer_state(trainer)
        finally:
            trainer.optimizer = original
    except Exception as exc:
        if isinstance(exc, PolicyImprovementSmokeCheckpointError):
            raise
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke optimizer state is incompatible."
        ) from exc

    environment_probe = copy.copy(trainer.env)
    try:
        environment_probe.load_checkpoint_state(value["environment_state"])
    except Exception as exc:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke environment state is incompatible."
        ) from exc
    _validate_episode_state(value["episode_state"], value["environment_state"])
    rollout = _validate_rollout_buffer(value["rollout_buffer"])
    if len(rollout["rewards"]) != int(trainer.config.num_steps):
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoint does not contain one complete rollout."
        )
    _restore_compute_state(
        trainer, value["compute_accounting_state"], validate_only=True
    )
    current_rng = _capture_rng_state()
    try:
        _restore_rng_state(value["rng_state"])
    finally:
        _restore_rng_state(current_rng)
    if validate_only:
        return

    trainer.model.load_state_dict(value["model_state_dict"], strict=True)
    trainer.model.train(bool(value["model_training"]))
    _restore_gradients(trainer.model, value["parameter_gradients"], validate_only=False)
    trainer.optimizer.load_state_dict(optimizer_state)
    _validate_optimizer_state(trainer)
    trainer.env.load_checkpoint_state(value["environment_state"])
    episode = value["episode_state"]
    trainer._current_x = _clone(episode["current_x"])
    trainer._current_y = _clone(episode["current_y"])
    trainer._episode_rewards = list(episode["episode_rewards"])
    _restore_rollout_buffer(trainer, value["rollout_buffer"])
    trainer._train_step_count = canonical_progress["train_steps"]
    trainer._env_step_count = canonical_progress["environment_interactions"]
    trainer._optimizer_step_count = canonical_progress["optimizer_steps"]
    trainer._episode_count = canonical_progress["episodes"]
    trainer.term_stats = canonical_counts
    _restore_compute_state(
        trainer, value["compute_accounting_state"], validate_only=False
    )
    _restore_rng_state(value["rng_state"])


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(descriptor, _READ_SIZE), b""):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def load_stable_checkpoint(
    path: str | Path | int, *, expected_sha256: str
) -> tuple[object, str]:
    source = None if isinstance(path, int) else Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.dup(path) if isinstance(path, int) else os.open(source, flags)
    except OSError as exc:
        raise PolicyImprovementSmokeCheckpointError(
            "PPO smoke checkpoint cannot be opened."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            source is not None and before.st_nlink != 1
        ):
            raise PolicyImprovementSmokeCheckpointError(
                "PPO smoke checkpoint must be a singly linked regular file."
            )
        digest = _hash_descriptor(descriptor)
        if digest != expected_sha256:
            raise PolicyImprovementSmokeCheckpointError(
                "PPO smoke checkpoint digest differs from its parent manifest."
            )
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            # Data-only: the restricted unpickler refuses any global outside
            # the authenticated allowlist, so hostile bytes cannot execute.
            try:
                value = load_data_only_checkpoint(handle, map_location="cpu")
            except UnsafeCheckpointPayloadError as exc:
                raise PolicyImprovementSmokeCheckpointError(str(exc)) from exc
        after = os.fstat(descriptor)
        path_changed = False
        if source is not None:
            try:
                after_path = source.lstat()
            except OSError:
                path_changed = True
            else:
                path_changed = (after_path.st_dev, after_path.st_ino) != (
                    after.st_dev,
                    after.st_ino,
                )
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or path_changed
            or _hash_descriptor(descriptor) != digest
        ):
            raise PolicyImprovementSmokeCheckpointError(
                "PPO smoke checkpoint changed while it was loaded."
            )
        return value, digest
    finally:
        os.close(descriptor)


def publish_checkpoint(value: object, destination: str | Path) -> str:
    """Durably publish one new checkpoint without replacing any path."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    linked = False
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        expected = hashlib.sha256(temporary.read_bytes()).hexdigest()
        try:
            os.link(temporary, target)
            linked = True
        except FileExistsError as exc:
            raise PolicyImprovementSmokeCheckpointError(
                "PPO smoke checkpoint destination already exists."
            ) from exc
        temporary.unlink()
        directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _, digest = load_stable_checkpoint(target, expected_sha256=expected)
        return digest
    except BaseException:
        if linked:
            target.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
