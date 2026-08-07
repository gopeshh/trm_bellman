"""Checkpoint-independent finite-batch diagnostics for persistent UPI-TRM.

This module consumes already constructed models and a ``PlanEditEnv``.  It does
not load checkpoints, synchronize model state, or claim a supremum over an
augmented policy-pair closure.  Every aggregate produced here is a retained-
batch diagnostic.

The environment snapshot is the transition oracle.  Exact one-step backups and
Monte Carlo K-step backups restore ``PlanEditEnv.checkpoint_state()`` and call
``env.step`` so reward, masking, STOP/UNDO, terminal, and edit-clock semantics
cannot drift from evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry
from rl.replay import ReplayLatent
from rl.theory_diagnostics import (
    FINITE_BATCH_SCOPE,
    summarize_carry_continuity,
    summarize_centering_defect,
    summarize_depth_path,
    summarize_finite_batch_metric,
    summarize_policy_gap,
    summarize_residual_estimates,
)
from utils.lipschitz import compute_exact_baseline_summation


NOT_VERIFIABLE = "not verifiable from supplied evidence"
STATE_SELECTOR_VERSION = "clock_round_robin_occurrence_sha256_v1"
OUTPUT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PersistentDiagnosticConfig:
    """Fixed numerical protocol for one persistent diagnostic run."""

    inner_unroll_n: int
    reference_depth_m: int
    gamma: float
    k_horizon: int = 5
    mc_repeats: int = 32
    mc_seed: int = 1729
    collection_seed: int = 2718
    mixture_alpha: float = 0.1
    max_retained_states: int = 256
    chunk_size: int = 32
    advantage_clip: Optional[float] = None
    probability_tolerance: float = 1e-6
    ratio_denominator_tolerance: float = 1e-12
    initial_latent_perturbation_l2_norm: float = 0.01
    initial_latent_perturbation_seed: int = 1729

    def __post_init__(self) -> None:
        if self.inner_unroll_n < 1:
            raise ValueError("inner_unroll_n must be at least one.")
        if self.reference_depth_m <= self.inner_unroll_n:
            raise ValueError("reference_depth_m must exceed inner_unroll_n.")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one.")
        if self.k_horizon < 1:
            raise ValueError("k_horizon must be at least one.")
        if self.mc_repeats < 2:
            raise ValueError("mc_repeats must be at least two to estimate an SE.")
        if not 0.0 <= self.mixture_alpha <= 1.0:
            raise ValueError("mixture_alpha must lie in [0, 1].")
        if self.max_retained_states < 1:
            raise ValueError("max_retained_states must be at least one.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least one.")
        if self.advantage_clip is not None and self.advantage_clip <= 0.0:
            raise ValueError("advantage_clip must be positive when supplied.")
        if self.probability_tolerance < 0.0:
            raise ValueError("probability_tolerance must be nonnegative.")
        if self.ratio_denominator_tolerance < 0.0:
            raise ValueError("ratio_denominator_tolerance must be nonnegative.")
        if (
            not math.isfinite(self.initial_latent_perturbation_l2_norm)
            or self.initial_latent_perturbation_l2_norm <= 0.0
        ):
            raise ValueError(
                "initial_latent_perturbation_l2_norm must be finite and positive."
            )
        if self.initial_latent_perturbation_seed < 0:
            raise ValueError("initial_latent_perturbation_seed must be nonnegative.")


@dataclass
class AugmentedDiagnosticState:
    """One retained nonterminal occurrence of ``(x, y, z, h)``.

    ``environment_state`` is the snapshot immediately before the observed
    action.  ``successor_environment_state`` is the snapshot immediately after
    it.  The observed transition is used only for carry/clock integrity; exact
    and Monte Carlo backups branch again from ``environment_state``.
    """

    occurrence_id: str
    augmented_state_hash: str
    source_record_index: int
    episode_id: int
    timestep: int
    environment_state: Dict[str, Any]
    input_latent: ReplayLatent
    observed_post_unroll_latent: ReplayLatent
    observed_action: int
    observed_reward: float
    observed_done: bool
    successor_environment_state: Dict[str, Any]
    next_input_latent: Optional[ReplayLatent]
    next_input_remaining_edits: Optional[int]


@dataclass
class PersistentStateCollection:
    states: List[AugmentedDiagnosticState]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DeploymentSpec:
    """How the policy advertised as deployed should be reconstructed."""

    kind: str = "missing"
    model: Optional[Any] = None
    policy_dist_fn: Optional[Callable[..., Tuple[Any, Any]]] = None
    label: str = "deployed_policy"

    def __post_init__(self) -> None:
        allowed = {"policy_dist_callback", "concrete_model", "missing"}
        if self.kind not in allowed:
            raise ValueError(f"Unknown deployment kind {self.kind!r}.")
        if self.kind == "concrete_model" and self.model is None:
            raise ValueError("concrete_model deployment requires a model.")
        if self.kind != "concrete_model" and self.model is not None:
            raise ValueError(f"Deployment kind {self.kind!r} cannot carry a model.")
        if self.kind == "policy_dist_callback" and self.policy_dist_fn is None:
            raise ValueError("policy_dist_callback deployment requires a callback.")
        if self.kind != "policy_dist_callback" and self.policy_dist_fn is not None:
            raise ValueError(
                f"Deployment kind {self.kind!r} cannot carry a policy callback."
            )


@dataclass
class PersistentDiagnosticOutput:
    summary: Dict[str, Any]
    state_rows: List[Dict[str, Any]]
    action_rows: List[Dict[str, Any]]
    depth_rows: List[Dict[str, Any]]
    mc_rows: List[Dict[str, Any]]


@dataclass
class _PathEvaluation:
    latent_h: torch.Tensor
    latent_l: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor
    current_probs: torch.Tensor
    candidate_probs: Optional[torch.Tensor]
    pre_projection_norms: Optional[torch.Tensor]
    projection_active: Optional[torch.Tensor]


@dataclass
class _Particle:
    state_index: int
    repeat_index: int
    snapshot: Dict[str, Any]
    latent: ReplayLatent
    discounted_return: float
    discount: float
    actions: List[int]
    rewards: List[float]
    done: bool = False


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def clone_replay_latent(latent: ReplayLatent) -> ReplayLatent:
    """Detach and clone a replay carry onto CPU."""

    if not isinstance(latent, ReplayLatent):
        raise TypeError("Expected a ReplayLatent.")
    if latent.z_H.shape != latent.z_L.shape:
        raise ValueError("Replay latent H/L shapes must match.")
    if not bool(torch.isfinite(latent.z_H).all().item()) or not bool(
        torch.isfinite(latent.z_L).all().item()
    ):
        raise ValueError("Replay latent must be finite.")
    return ReplayLatent(
        z_H=latent.z_H.detach().cpu().clone(),
        z_L=latent.z_L.detach().cpu().clone(),
    )


def _carry_to_replay(latent: Any) -> ReplayLatent:
    if not hasattr(latent, "z_H") or not hasattr(latent, "z_L"):
        raise TypeError("Model did not return a recurrent H/L carry.")
    return clone_replay_latent(ReplayLatent(z_H=latent.z_H, z_L=latent.z_L))


def _hash_tensor(hasher: Any, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous().clone()
    hasher.update(b"tensor\0")
    hasher.update(str(cpu.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(b"\0")
    if cpu.numel() > 0:
        hasher.update(bytes(cpu.untyped_storage()))


def _hash_value(hasher: Any, value: Any) -> None:
    if isinstance(value, np.generic):
        _hash_value(hasher, value.item())
        return
    if isinstance(value, np.ndarray):
        _hash_tensor(hasher, torch.from_numpy(np.ascontiguousarray(value)))
        return
    if torch.is_tensor(value):
        _hash_tensor(hasher, value)
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(hasher, str(key))
            _hash_value(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"sequence[")
        for item in value:
            _hash_value(hasher, item)
        hasher.update(b"]")
        return
    if value is None:
        hasher.update(b"none")
        return
    if isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
        return
    if isinstance(value, int):
        hasher.update(f"int:{value}".encode("ascii"))
        return
    if isinstance(value, float):
        hasher.update(f"float:{value.hex()}".encode("ascii"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(f"str:{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
        return
    raise TypeError(f"Cannot hash value of type {type(value).__name__}.")


def hash_augmented_state(
    environment_state: Mapping[str, Any],
    latent: ReplayLatent,
) -> str:
    """Hash the content of one augmented state without identity-bearing paths."""

    if not bool(environment_state.get("initialized", False)):
        raise ValueError("Diagnostic environment state must be initialized.")
    payload = {
        "x": environment_state.get("x"),
        "y": environment_state.get("y"),
        "latent_h": latent.z_H,
        "latent_l": latent.z_L,
        "remaining_edits": _clock_from_snapshot(environment_state),
    }
    hasher = hashlib.sha256()
    hasher.update(b"upi-trm-augmented-state-v1\0")
    _hash_value(hasher, payload)
    return hasher.hexdigest()


_RECURRENT_CONFIG_FIELDS = (
    "seq_len",
    "puzzle_emb_ndim",
    "num_puzzle_identifiers",
    "vocab_size",
    "L_cycles",
    "L_layers",
    "hidden_size",
    "expansion",
    "num_heads",
    "pos_encodings",
    "rms_norm_eps",
    "rope_theta",
    "forward_dtype",
    "mlp_t",
    "puzzle_emb_len",
    "rl_enable_z_init_encoder",
    "rl_value_hidden_dim",
    "rl_latent_projection_mode",
    "rl_latent_ball_radius",
)


def _config_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "dict"):
        return dict(config.dict())
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("Model configuration is not serializable.")


def _is_recurrent_state_name(name: str) -> bool:
    if name.startswith("z_init_encoder."):
        return True
    if not name.startswith("inner."):
        return False
    excluded = ("inner.lm_head.", "inner.q_head.")
    return not name.startswith(excluded)


def hash_recurrent_map(model: Any) -> str:
    """Hash recurrent/initialization tensors and behavior-relevant config."""

    if not hasattr(model, "state_dict") or not hasattr(model, "config"):
        raise TypeError("Recurrent map hashing requires a model and configuration.")
    config = _config_dict(model.config)
    recurrent_config = {
        key: config.get(key) for key in _RECURRENT_CONFIG_FIELDS if key in config
    }
    recurrent_state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if _is_recurrent_state_name(name)
    }
    if not recurrent_state:
        raise ValueError("Model contains no recognized recurrent state.")
    hasher = hashlib.sha256()
    hasher.update(b"upi-trm-recurrent-map-v1\0")
    _hash_value(hasher, type(model).__module__ + "." + type(model).__qualname__)
    _hash_value(hasher, recurrent_config)
    _hash_value(hasher, recurrent_state)
    return hasher.hexdigest()


def recurrent_map_report(
    evaluator: Any,
    current_policy: Any,
    candidate_policy: Optional[Any],
    deployment: DeploymentSpec,
) -> Dict[str, Any]:
    hashes = {
        "evaluator": hash_recurrent_map(evaluator),
        "current_policy": hash_recurrent_map(current_policy),
    }
    if candidate_policy is not None:
        hashes["candidate_policy"] = hash_recurrent_map(candidate_policy)
    if deployment.kind == "concrete_model":
        hashes["deployed_policy"] = hash_recurrent_map(deployment.model)
    core_names = ["evaluator", "current_policy"]
    if "candidate_policy" in hashes:
        core_names.append("candidate_policy")
    core_equal = len({hashes[name] for name in core_names}) == 1
    return {
        "hash_schema": "upi-trm-recurrent-map-v1",
        "hashes": hashes,
        "evaluator_current_candidate_equal": core_equal,
        "condition": "exact tensor/config identity, excluding output heads",
    }


def _normalize_puzzle_id(value: torch.Tensor) -> torch.Tensor:
    if value.numel() != 1:
        raise ValueError("puzzle_identifiers must be scalar per state.")
    return value.reshape(())


def stack_full_tensor_state_fields(
    states: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Stack every tensor field while rejecting schema loss between records."""

    if not states:
        raise ValueError("Cannot stack an empty state sequence.")
    tensor_key_sets = [
        {key for key, value in state.items() if torch.is_tensor(value)}
        for state in states
    ]
    if any(keys != tensor_key_sets[0] for keys in tensor_key_sets[1:]):
        raise ValueError("State batch has inconsistent tensor fields.")
    required = {"inputs", "puzzle_identifiers", "remaining_edits"}
    missing = required - tensor_key_sets[0]
    if missing:
        raise KeyError(f"Augmented states are missing fields: {sorted(missing)}")

    batch: Dict[str, torch.Tensor] = {}
    for key in sorted(tensor_key_sets[0]):
        values = [state[key] for state in states]
        if key == "puzzle_identifiers":
            stacked = torch.stack([_normalize_puzzle_id(value) for value in values])
        elif key == "remaining_edits":
            if any(value.numel() != 1 for value in values):
                raise ValueError("remaining_edits must be scalar per state.")
            stacked = torch.stack([value.reshape(()) for value in values])
        else:
            stacked = torch.stack(values)
        batch[key] = stacked.to(device)
    return batch


def _plan_tensor(plan: Any) -> torch.Tensor:
    if isinstance(plan, Mapping):
        plan = plan.get("inputs")
    if plan is None:
        raise ValueError("Environment snapshot has no plan tensor.")
    if not torch.is_tensor(plan):
        plan = torch.as_tensor(plan)
    return plan


def _stack_plans(plans: Sequence[Any], device: torch.device) -> torch.Tensor:
    return torch.stack([_plan_tensor(plan) for plan in plans]).to(
        device=device,
        dtype=torch.long,
    )


def _stack_replay_latents(
    latents: Sequence[ReplayLatent],
    device: torch.device,
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    if not latents:
        raise ValueError("Cannot stack an empty latent sequence.")
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=torch.cat([latent.z_H for latent in latents], dim=0).to(device),
        z_L=torch.cat([latent.z_L for latent in latents], dim=0).to(device),
    )


def _slice_carry(carry: Any, index: int) -> ReplayLatent:
    return clone_replay_latent(
        ReplayLatent(
            z_H=carry.z_H[index : index + 1],
            z_L=carry.z_L[index : index + 1],
        )
    )


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _joint_norm(h: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, h.ndim))
    return torch.sqrt(h.pow(2).sum(dim=reduce_dims) + l.pow(2).sum(dim=reduce_dims))


def _joint_distance(
    left_h: torch.Tensor,
    left_l: torch.Tensor,
    right_h: torch.Tensor,
    right_l: torch.Tensor,
) -> torch.Tensor:
    return _joint_norm(left_h - right_h, left_l - right_l)


def _clock_from_snapshot(snapshot: Mapping[str, Any]) -> int:
    x = snapshot.get("x")
    if not isinstance(x, Mapping):
        raise ValueError("Environment snapshot x must be a mapping.")
    clock = x.get("remaining_edits")
    if not isinstance(clock, torch.Tensor) or clock.numel() != 1:
        raise ValueError("Environment snapshot lacks a scalar remaining_edits clock.")
    scalar = clock.detach().cpu().reshape(()).item()
    integer = int(scalar)
    if float(scalar) != float(integer) or integer < 0:
        raise ValueError("remaining_edits must be a nonnegative integer.")
    return integer


def _stable_seed(*parts: Any) -> int:
    hasher = hashlib.sha256()
    hasher.update(b"upi-trm-diagnostic-rng-v1\0")
    for part in parts:
        _hash_value(hasher, part)
    return int.from_bytes(hasher.digest()[:8], "little") % (2**63 - 1)


def _sample_probability_row(probabilities: torch.Tensor, seed: int) -> int:
    probs = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if probs.ndim != 1 or not bool(torch.isfinite(probs).all().item()):
        raise ValueError("Action probabilities must be one finite row.")
    if bool((probs < 0).any().item()) or not math.isclose(
        float(probs.sum().item()), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("Action probabilities must be normalized and nonnegative.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return int(torch.multinomial(probs, 1, generator=generator).item())


@contextmanager
def _preserve_runtime(models: Sequence[Any], env: Any) -> Iterator[None]:
    unique_models: List[Any] = []
    seen_ids = set()
    for model in models:
        if model is None or id(model) in seen_ids:
            continue
        seen_ids.add(id(model))
        unique_models.append(model)

    modes = [(model, bool(model.training)) for model in unique_models]
    env_state = _clone_value(env.checkpoint_state())
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for model in unique_models:
            model.eval()
        yield
    finally:
        env.load_checkpoint_state(env_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for model, was_training in modes:
            model.train(was_training)


def _snapshot_action_mask(snapshot: Mapping[str, Any]) -> torch.Tensor:
    mask = snapshot.get("action_mask")
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Exact diagnostics require a retained action mask.")
    mask = mask.detach().cpu().to(torch.bool).reshape(-1)
    if not bool(mask.any().item()):
        raise ValueError("Retained state has no valid action.")
    return mask


def _validate_collected_state(state: AugmentedDiagnosticState) -> None:
    if state.source_record_index < 0 or state.episode_id < 0 or state.timestep < 0:
        raise ValueError("State source, episode, and timestep must be nonnegative.")
    if bool(state.environment_state.get("done", False)):
        raise ValueError("A diagnostic start state must be nonterminal.")
    if _clock_from_snapshot(state.environment_state) < 1:
        raise ValueError("A diagnostic start state must have positive edit budget.")
    expected_hash = hash_augmented_state(state.environment_state, state.input_latent)
    if state.augmented_state_hash != expected_hash:
        raise ValueError("Augmented-state content hash does not match its payload.")
    _snapshot_action_mask(state.environment_state)
    clone_replay_latent(state.input_latent)
    clone_replay_latent(state.observed_post_unroll_latent)
    successor_clock = _clock_from_snapshot(state.successor_environment_state)
    if successor_clock != _clock_from_snapshot(state.environment_state) - 1:
        raise ValueError("Observed successor clock does not decrement exactly once.")
    if (
        bool(state.successor_environment_state.get("done", False))
        != state.observed_done
    ):
        raise ValueError("Observed done flag disagrees with successor snapshot.")
    if state.observed_done and state.next_input_latent is not None:
        raise ValueError("Terminal transition cannot have a next input carry.")
    if state.observed_done and state.next_input_remaining_edits is not None:
        raise ValueError("Terminal transition cannot have a next input clock.")
    if not state.observed_done and state.next_input_latent is None:
        raise ValueError("Nonterminal transition is missing its next input carry.")
    if not state.observed_done and state.next_input_remaining_edits is None:
        raise ValueError("Nonterminal transition is missing its next input clock.")
    if (
        state.next_input_remaining_edits is not None
        and state.next_input_remaining_edits != successor_clock
    ):
        raise ValueError("Observed successor clock differs from the next occurrence.")


def _select_clock_balanced(
    states: Sequence[AugmentedDiagnosticState],
    cap: int,
) -> List[AugmentedDiagnosticState]:
    buckets: Dict[int, List[AugmentedDiagnosticState]] = {}
    for state in states:
        buckets.setdefault(_clock_from_snapshot(state.environment_state), []).append(
            state
        )
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.occurrence_id)

    selected: List[AugmentedDiagnosticState] = []
    offsets = {clock: 0 for clock in buckets}
    while len(selected) < min(cap, len(states)):
        made_progress = False
        for clock in sorted(buckets):
            offset = offsets[clock]
            if offset >= len(buckets[clock]):
                continue
            selected.append(buckets[clock][offset])
            offsets[clock] += 1
            made_progress = True
            if len(selected) >= cap:
                break
        if not made_progress:
            break
    return selected


def collect_persistent_diagnostic_states(
    *,
    current_policy: Any,
    env: Any,
    record_indices: Sequence[int],
    config: PersistentDiagnosticConfig,
) -> PersistentStateCollection:
    """Collect reproducible persistent traces for explicit held-out indices."""

    indices = [int(index) for index in record_indices]
    if not indices:
        raise ValueError("record_indices must be nonempty.")
    if len(set(indices)) != len(indices):
        raise ValueError(
            "record_indices must be unique; implicit cycling is forbidden."
        )
    if not hasattr(env.dataset, "__len__"):
        raise TypeError("Diagnostic environment dataset must define __len__.")
    dataset_size = len(env.dataset)
    if any(index < 0 or index >= dataset_size for index in indices):
        raise IndexError("A requested diagnostic record is outside the dataset.")

    device = _model_device(current_policy)
    generated: List[AugmentedDiagnosticState] = []
    with _preserve_runtime([current_policy], env), torch.inference_mode():
        for episode_id, record_index in enumerate(indices):
            x, y = env.reset(idx=record_index)
            x_batch = stack_full_tensor_state_fields([x], device)
            y_batch = _stack_plans([y], device)
            carry = current_policy.init_latent(x_batch, y_batch)
            timestep = 0
            done = False
            while not done:
                if timestep >= int(env.config.max_edits):
                    raise RuntimeError(
                        "Diagnostic collector reached max_edits without a terminal."
                    )
                snapshot = _clone_value(env.checkpoint_state())
                input_latent = _carry_to_replay(carry)
                content_hash = hash_augmented_state(snapshot, input_latent)
                occurrence_hasher = hashlib.sha256()
                occurrence_hasher.update(b"upi-trm-state-occurrence-v1\0")
                _hash_value(
                    occurrence_hasher,
                    [record_index, episode_id, timestep, content_hash],
                )
                occurrence_id = occurrence_hasher.hexdigest()

                action_mask = _snapshot_action_mask(snapshot).to(device)
                x_batch = stack_full_tensor_state_fields([snapshot["x"]], device)
                y_batch = _stack_plans([snapshot["y"]], device)
                dist, post_carry = current_policy.policy_dist(
                    x_batch,
                    y_batch,
                    n=config.inner_unroll_n,
                    action_mask=action_mask,
                    z=carry,
                )
                action = _sample_probability_row(
                    dist.probs.reshape(1, -1)[0],
                    _stable_seed(
                        config.collection_seed,
                        occurrence_id,
                        "collection_action",
                    ),
                )
                (_, _), reward, done, _ = env.step(action)
                successor_snapshot = _clone_value(env.checkpoint_state())
                post_latent = _carry_to_replay(post_carry)
                generated.append(
                    AugmentedDiagnosticState(
                        occurrence_id=occurrence_id,
                        augmented_state_hash=content_hash,
                        source_record_index=record_index,
                        episode_id=episode_id,
                        timestep=timestep,
                        environment_state=snapshot,
                        input_latent=input_latent,
                        observed_post_unroll_latent=post_latent,
                        observed_action=action,
                        observed_reward=float(reward),
                        observed_done=bool(done),
                        successor_environment_state=successor_snapshot,
                        next_input_latent=None,
                        next_input_remaining_edits=None,
                    )
                )
                carry = post_carry
                timestep += 1

    for index, state in enumerate(generated):
        if state.observed_done:
            continue
        if index + 1 >= len(generated):
            raise ValueError("Collected nonterminal state has no observed successor.")
        successor = generated[index + 1]
        if (
            successor.episode_id != state.episode_id
            or successor.timestep != state.timestep + 1
        ):
            raise ValueError("Collected nonterminal state is not followed by its successor.")
        state.next_input_latent = clone_replay_latent(successor.input_latent)
        state.next_input_remaining_edits = _clock_from_snapshot(
            successor.environment_state
        )

    selected = _select_clock_balanced(generated, config.max_retained_states)
    generated_clock_counts: Dict[str, int] = {}
    retained_clock_counts: Dict[str, int] = {}
    for state in generated:
        key = str(_clock_from_snapshot(state.environment_state))
        generated_clock_counts[key] = generated_clock_counts.get(key, 0) + 1
    for state in selected:
        key = str(_clock_from_snapshot(state.environment_state))
        retained_clock_counts[key] = retained_clock_counts.get(key, 0) + 1
    return PersistentStateCollection(
        states=selected,
        metadata={
            "scope": FINITE_BATCH_SCOPE,
            "uniform_certificate": False,
            "source_episode_count": len(indices),
            "generated_state_count": len(generated),
            "retained_state_count": len(selected),
            "selector_version": STATE_SELECTOR_VERSION,
            "max_retained_states": config.max_retained_states,
            "generated_clock_counts": generated_clock_counts,
            "retained_clock_counts": retained_clock_counts,
            "record_indices_sha256": _hash_sequence(indices),
        },
    )


def _hash_sequence(values: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    _hash_value(hasher, list(values))
    return hasher.hexdigest()


def _state_components(
    states: Sequence[AugmentedDiagnosticState],
) -> Tuple[List[Mapping[str, Any]], List[Any], List[ReplayLatent]]:
    x_values = [state.environment_state["x"] for state in states]
    y_values = [state.environment_state["y"] for state in states]
    latents = [state.input_latent for state in states]
    return x_values, y_values, latents


def _evaluate_paths(
    evaluator: Any,
    current_policy: Any,
    candidate_policy: Optional[Any],
    states: Sequence[AugmentedDiagnosticState],
    config: PersistentDiagnosticConfig,
    *,
    endpoint_policy_pair_invariants_verified: bool,
) -> _PathEvaluation:
    device = _model_device(current_policy)
    all_h: List[torch.Tensor] = []
    all_l: List[torch.Tensor] = []
    all_values: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    all_current_probs: List[torch.Tensor] = []
    all_candidate_probs: List[torch.Tensor] = []
    all_pre_projection_norms: List[torch.Tensor] = []
    all_projection_active: List[torch.Tensor] = []
    projection_diagnostics_available = hasattr(
        current_policy,
        "update_latent_with_projection_info",
    )

    for start in range(0, len(states), config.chunk_size):
        chunk = states[start : start + config.chunk_size]
        x_values, y_values, latents = _state_components(chunk)
        x_batch = stack_full_tensor_state_fields(x_values, device)
        y_batch = _stack_plans(y_values, device)
        carry = _stack_replay_latents(latents, device)
        masks = torch.stack(
            [_snapshot_action_mask(state.environment_state) for state in chunk]
        ).to(device)

        h_by_depth: List[torch.Tensor] = []
        l_by_depth: List[torch.Tensor] = []
        values_by_depth: List[torch.Tensor] = []
        pre_projection_by_step: List[torch.Tensor] = []
        projection_active_by_step: List[torch.Tensor] = []
        z_n: Optional[Any] = None
        for depth in range(config.reference_depth_m + 1):
            h_by_depth.append(carry.z_H.detach().cpu())
            l_by_depth.append(carry.z_L.detach().cpu())
            values, _ = evaluator.used_value(x_batch, y_batch, n=0, z=carry)
            values_by_depth.append(values.detach().cpu().reshape(-1))
            if depth == config.inner_unroll_n:
                z_n = carry
            if depth < config.reference_depth_m:
                if projection_diagnostics_available:
                    carry, pre_norm, projection_active = (
                        current_policy.update_latent_with_projection_info(
                            carry,
                            y_batch,
                            x_batch,
                        )
                    )
                    pre_projection_by_step.append(pre_norm.detach().cpu())
                    projection_active_by_step.append(
                        projection_active.detach().cpu()
                    )
                else:
                    carry = current_policy.update_latent(carry, y_batch, x_batch)
        assert z_n is not None

        current_dist, _ = current_policy.policy_dist(
            x_batch,
            y_batch,
            n=0,
            action_mask=masks,
            z=z_n,
        )
        all_current_probs.append(current_dist.probs.detach().cpu())
        if endpoint_policy_pair_invariants_verified:
            if candidate_policy is None:
                raise ValueError(
                    "Verified endpoint policy-pair invariants require a candidate model."
                )
            candidate_dist, _ = candidate_policy.policy_dist(
                x_batch,
                y_batch,
                n=0,
                action_mask=masks,
                z=z_n,
            )
            all_candidate_probs.append(candidate_dist.probs.detach().cpu())

        all_h.append(torch.stack(h_by_depth, dim=1))
        all_l.append(torch.stack(l_by_depth, dim=1))
        all_values.append(torch.stack(values_by_depth, dim=1))
        all_masks.append(masks.detach().cpu())
        if projection_diagnostics_available:
            all_pre_projection_norms.append(
                torch.stack(pre_projection_by_step, dim=1)
            )
            all_projection_active.append(
                torch.stack(projection_active_by_step, dim=1)
            )

    return _PathEvaluation(
        latent_h=torch.cat(all_h, dim=0),
        latent_l=torch.cat(all_l, dim=0),
        values=torch.cat(all_values, dim=0),
        action_masks=torch.cat(all_masks, dim=0),
        current_probs=torch.cat(all_current_probs, dim=0),
        candidate_probs=(
            torch.cat(all_candidate_probs, dim=0) if all_candidate_probs else None
        ),
        pre_projection_norms=(
            torch.cat(all_pre_projection_norms, dim=0)
            if all_pre_projection_norms
            else None
        ),
        projection_active=(
            torch.cat(all_projection_active, dim=0)
            if all_projection_active
            else None
        ),
    )


def _evaluate_successor_values(
    evaluator: Any,
    current_policy: Any,
    records: Sequence[Tuple[Mapping[str, Any], Any, ReplayLatent]],
    config: PersistentDiagnosticConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not records:
        empty = torch.empty(0, dtype=torch.float64)
        return empty, empty
    device = _model_device(current_policy)
    values_n: List[torch.Tensor] = []
    values_m: List[torch.Tensor] = []
    for start in range(0, len(records), config.chunk_size):
        chunk = records[start : start + config.chunk_size]
        x_batch = stack_full_tensor_state_fields([item[0] for item in chunk], device)
        y_batch = _stack_plans([item[1] for item in chunk], device)
        carry = _stack_replay_latents([item[2] for item in chunk], device)
        value_n = None
        value_m = None
        for depth in range(config.reference_depth_m + 1):
            if depth in (config.inner_unroll_n, config.reference_depth_m):
                value, _ = evaluator.used_value(x_batch, y_batch, n=0, z=carry)
                if depth == config.inner_unroll_n:
                    value_n = value.detach().cpu().to(torch.float64).reshape(-1)
                else:
                    value_m = value.detach().cpu().to(torch.float64).reshape(-1)
            if depth < config.reference_depth_m:
                carry = current_policy.update_latent(carry, y_batch, x_batch)
        assert value_n is not None and value_m is not None
        values_n.append(value_n)
        values_m.append(value_m)
    return torch.cat(values_n), torch.cat(values_m)


def _clip_and_recenter(
    advantages: torch.Tensor,
    policy_probs: torch.Tensor,
    action_mask: torch.Tensor,
    clip_value: Optional[float],
) -> torch.Tensor:
    safe = torch.where(action_mask, advantages, torch.zeros_like(advantages))
    if clip_value is not None:
        safe = torch.where(
            action_mask,
            safe.clamp(-clip_value, clip_value),
            torch.zeros_like(safe),
        )
    center = (policy_probs * safe).sum(dim=-1, keepdim=True)
    return torch.where(action_mask, safe - center, torch.zeros_like(safe))


def _exact_one_step_backups(
    evaluator: Any,
    current_policy: Any,
    env: Any,
    states: Sequence[AugmentedDiagnosticState],
    path: _PathEvaluation,
    config: PersistentDiagnosticConfig,
) -> Dict[str, torch.Tensor]:
    state_count, action_count = path.current_probs.shape
    q_n = torch.full((state_count, action_count), -math.inf, dtype=torch.float64)
    q_m = torch.full_like(q_n, -math.inf)
    successor_records: List[Tuple[Mapping[str, Any], Any, ReplayLatent]] = []
    successor_locations: List[Tuple[int, int]] = []

    original_env_state = _clone_value(env.checkpoint_state())
    try:
        for state_index, state in enumerate(states):
            post_latent = ReplayLatent(
                z_H=path.latent_h[state_index : state_index + 1, config.inner_unroll_n],
                z_L=path.latent_l[state_index : state_index + 1, config.inner_unroll_n],
            )
            valid_actions = torch.nonzero(
                path.action_masks[state_index], as_tuple=True
            )[0].tolist()
            for action in valid_actions:
                env.load_checkpoint_state(_clone_value(state.environment_state))
                (x_next, y_next), reward, done, _ = env.step(int(action))
                q_n[state_index, action] = float(reward)
                q_m[state_index, action] = float(reward)
                if not done:
                    successor_locations.append((state_index, int(action)))
                    successor_records.append(
                        (
                            _clone_value(x_next),
                            _clone_value(y_next),
                            clone_replay_latent(post_latent),
                        )
                    )
    finally:
        env.load_checkpoint_state(original_env_state)

    successor_n, successor_m = _evaluate_successor_values(
        evaluator,
        current_policy,
        successor_records,
        config,
    )
    for offset, (state_index, action) in enumerate(successor_locations):
        q_n[state_index, action] += config.gamma * successor_n[offset]
        q_m[state_index, action] += config.gamma * successor_m[offset]

    masks = path.action_masks.to(torch.bool)
    probs = path.current_probs.to(torch.float64)
    safe_q_n = torch.where(masks, q_n, torch.zeros_like(q_n))
    safe_q_m = torch.where(masks, q_m, torch.zeros_like(q_m))
    backup_n = (probs * safe_q_n).sum(dim=-1)
    backup_m = (probs * safe_q_m).sum(dim=-1)
    raw_advantages_n = torch.where(
        masks,
        safe_q_n - backup_n.unsqueeze(-1),
        torch.zeros_like(safe_q_n),
    )
    raw_advantages_m = torch.where(
        masks,
        safe_q_m - backup_m.unsqueeze(-1),
        torch.zeros_like(safe_q_m),
    )
    advantages_n = _clip_and_recenter(
        raw_advantages_n,
        probs,
        masks,
        config.advantage_clip,
    )
    advantages_m = _clip_and_recenter(
        raw_advantages_m,
        probs,
        masks,
        config.advantage_clip,
    )
    value_n = path.values[:, config.inner_unroll_n].to(torch.float64)
    value_m = path.values[:, config.reference_depth_m].to(torch.float64)

    production_q_parts: List[torch.Tensor] = []
    production_baseline_parts: List[torch.Tensor] = []
    device = _model_device(evaluator)
    for start in range(0, len(states), config.chunk_size):
        chunk = states[start : start + config.chunk_size]
        x_batch = stack_full_tensor_state_fields(
            [state.environment_state["x"] for state in chunk],
            device,
        )
        x_batch["remaining_edits"] = torch.tensor(
            [
                _clock_from_snapshot(state.environment_state)
                for state in chunk
            ],
            dtype=torch.long,
            device=device,
        )
        y_batch = _stack_plans(
            [state.environment_state["y"] for state in chunk],
            device,
        )
        action_mask = masks[start : start + len(chunk)].to(device)
        probabilities = probs[start : start + len(chunk)].to(
            device=device,
            dtype=path.current_probs.dtype,
        )
        successor_latent = ReplayLatent(
            z_H=path.latent_h[
                start : start + len(chunk), config.inner_unroll_n
            ].to(device),
            z_L=path.latent_l[
                start : start + len(chunk), config.inner_unroll_n
            ].to(device),
        )
        production_baseline, production_q = compute_exact_baseline_summation(
            model=evaluator,
            x_batch=x_batch,
            y_batch=y_batch,
            env=env,
            n=config.inner_unroll_n,
            gamma=config.gamma,
            checker_fn=env.checker,
            action_mask=action_mask,
            policy_probs=probabilities,
            successor_latent=successor_latent,
        )
        production_q_parts.append(production_q.detach().cpu().to(torch.float64))
        production_baseline_parts.append(
            production_baseline.detach().cpu().to(torch.float64)
        )
    production_q = torch.cat(production_q_parts, dim=0)
    production_baseline = torch.cat(production_baseline_parts, dim=0)
    production_q_error = torch.where(
        masks,
        (production_q - q_n).abs(),
        torch.zeros_like(q_n),
    )
    return {
        "q_n": q_n,
        "q_m": q_m,
        "backup_n": backup_n,
        "backup_m": backup_m,
        "signed_residual_n": value_n - backup_n,
        "signed_residual_m": value_m - backup_m,
        "raw_advantages_n": raw_advantages_n,
        "raw_advantages_m": raw_advantages_m,
        "advantages_n": advantages_n,
        "advantages_m": advantages_m,
        "centering_defect": (probs * advantages_n).sum(dim=-1).abs(),
        "production_q_n": production_q,
        "production_baseline_n": production_baseline,
        "production_q_max_absolute_error": production_q_error.max(dim=-1).values,
        "production_baseline_absolute_error": (
            production_baseline - backup_n
        ).abs(),
    }


def _evaluate_deployed_probs(
    deployment: DeploymentSpec,
    states: Sequence[AugmentedDiagnosticState],
    masks: torch.Tensor,
    config: PersistentDiagnosticConfig,
    device: torch.device,
) -> torch.Tensor:
    concrete_model: Optional[Any] = None
    policy_dist_fn: Optional[Callable[..., Tuple[Any, Any]]] = None
    if deployment.kind == "concrete_model":
        assert deployment.model is not None
        concrete_model = deployment.model
        device = _model_device(concrete_model)
    elif deployment.kind == "policy_dist_callback":
        assert deployment.policy_dist_fn is not None
        policy_dist_fn = deployment.policy_dist_fn
    else:
        raise ValueError(f"Deployment kind {deployment.kind!r} is not executable.")
    outputs: List[torch.Tensor] = []
    for start in range(0, len(states), config.chunk_size):
        chunk = states[start : start + config.chunk_size]
        x_values, y_values, latents = _state_components(chunk)
        x_batch = stack_full_tensor_state_fields(x_values, device)
        y_batch = _stack_plans(y_values, device)
        z_batch = _stack_replay_latents(latents, device)
        action_mask = masks[start : start + len(chunk)].to(device)
        if concrete_model is not None:
            dist, _ = concrete_model.policy_dist(
                x_batch,
                y_batch,
                n=config.inner_unroll_n,
                action_mask=action_mask,
                z=z_batch,
            )
        else:
            assert policy_dist_fn is not None
            dist, _ = policy_dist_fn(
                x_batch,
                y_batch,
                n=config.inner_unroll_n,
                action_mask=action_mask,
                z=z_batch,
            )
        outputs.append(dist.probs.detach().cpu())
    return torch.cat(outputs, dim=0)


def _directional_kl_rows(
    reference: torch.Tensor,
    comparison: torch.Tensor,
) -> torch.Tensor:
    ref_support = reference > 0
    cmp_support = comparison > 0
    missing = ref_support & ~cmp_support
    common = ref_support & cmp_support
    safe_ref = torch.where(common, reference, torch.ones_like(reference))
    safe_cmp = torch.where(common, comparison, torch.ones_like(comparison))
    terms = torch.where(
        common,
        reference * (safe_ref.log() - safe_cmp.log()),
        torch.zeros_like(reference),
    )
    return terms.sum(dim=-1).clamp_min(0.0).masked_fill(missing.any(dim=-1), math.inf)


def _policy_gap_raw(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    ref = reference.to(torch.float64)
    cmp = comparison.to(torch.float64)
    valid = mask.to(torch.bool)
    masked_ref = torch.where(valid, ref, torch.zeros_like(ref))
    masked_cmp = torch.where(valid, cmp, torch.zeros_like(cmp))
    tv = 0.5 * (masked_ref - masked_cmp).abs().sum(-1)
    mismatch = ((masked_ref > 0) != (masked_cmp > 0)).any(dim=-1)
    return {
        "total_variation": tv,
        "kl_reference_to_comparison": _directional_kl_rows(
            masked_ref,
            masked_cmp,
        ),
        "kl_comparison_to_reference": _directional_kl_rows(
            masked_cmp,
            masked_ref,
        ),
        "support_mismatch": mismatch,
    }


def _monte_carlo_k_step(
    evaluator: Any,
    current_policy: Any,
    env: Any,
    states: Sequence[AugmentedDiagnosticState],
    start_values_n: torch.Tensor,
    start_values_m: torch.Tensor,
    config: PersistentDiagnosticConfig,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    particles = [
        _Particle(
            state_index=state_index,
            repeat_index=repeat_index,
            snapshot=_clone_value(state.environment_state),
            latent=clone_replay_latent(state.input_latent),
            discounted_return=0.0,
            discount=1.0,
            actions=[],
            rewards=[],
        )
        for state_index, state in enumerate(states)
        for repeat_index in range(config.mc_repeats)
    ]
    device = _model_device(current_policy)
    original_env_state = _clone_value(env.checkpoint_state())
    try:
        for step in range(config.k_horizon):
            active = [particle for particle in particles if not particle.done]
            if not active:
                break
            for start in range(0, len(active), config.chunk_size):
                chunk = active[start : start + config.chunk_size]
                x_batch = stack_full_tensor_state_fields(
                    [particle.snapshot["x"] for particle in chunk], device
                )
                y_batch = _stack_plans(
                    [particle.snapshot["y"] for particle in chunk], device
                )
                z_batch = _stack_replay_latents(
                    [particle.latent for particle in chunk], device
                )
                masks = torch.stack(
                    [_snapshot_action_mask(particle.snapshot) for particle in chunk]
                ).to(device)
                dist, post_carry = current_policy.policy_dist(
                    x_batch,
                    y_batch,
                    n=config.inner_unroll_n,
                    action_mask=masks,
                    z=z_batch,
                )
                for row, particle in enumerate(chunk):
                    state = states[particle.state_index]
                    action = _sample_probability_row(
                        dist.probs[row],
                        _stable_seed(
                            config.mc_seed,
                            state.occurrence_id,
                            particle.repeat_index,
                            step,
                        ),
                    )
                    env.load_checkpoint_state(_clone_value(particle.snapshot))
                    (_, _), reward, done, _ = env.step(action)
                    particle.discounted_return += particle.discount * float(reward)
                    particle.discount *= config.gamma
                    particle.actions.append(action)
                    particle.rewards.append(float(reward))
                    particle.done = bool(done)
                    particle.snapshot = _clone_value(env.checkpoint_state())
                    particle.latent = _slice_carry(post_carry, row)
    finally:
        env.load_checkpoint_state(original_env_state)

    bootstrap_particles = [particle for particle in particles if not particle.done]
    bootstrap_values_by_particle: Dict[Tuple[int, int], Tuple[float, float]] = {}
    if bootstrap_particles:
        records = [
            (particle.snapshot["x"], particle.snapshot["y"], particle.latent)
            for particle in bootstrap_particles
        ]
        bootstrap_values_n, bootstrap_values_m = _evaluate_successor_values(
            evaluator,
            current_policy,
            records,
            config,
        )
        for particle, value_n, value_m in zip(
            bootstrap_particles,
            bootstrap_values_n.tolist(),
            bootstrap_values_m.tolist(),
        ):
            bootstrap_values_by_particle[
                (particle.state_index, particle.repeat_index)
            ] = (float(value_n), float(value_m))

    returns_n = torch.empty((len(states), config.mc_repeats), dtype=torch.float64)
    returns_m = torch.empty_like(returns_n)
    raw_rows: List[Dict[str, Any]] = []
    for particle in particles:
        return_n = particle.discounted_return
        return_m = particle.discounted_return
        bootstrap = bootstrap_values_by_particle.get(
            (particle.state_index, particle.repeat_index)
        )
        if bootstrap is not None:
            return_n += particle.discount * bootstrap[0]
            return_m += particle.discount * bootstrap[1]
        returns_n[particle.state_index, particle.repeat_index] = return_n
        returns_m[particle.state_index, particle.repeat_index] = return_m
        raw_rows.append(
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "scope": FINITE_BATCH_SCOPE,
                "occurrence_id": states[particle.state_index].occurrence_id,
                "state_index": particle.state_index,
                "repeat_index": particle.repeat_index,
                "actions": list(particle.actions),
                "rewards": list(particle.rewards),
                "steps_taken": len(particle.actions),
                "terminated": particle.done,
                "bootstrapped": not particle.done,
                "k_step_return": return_n,
                "k_step_return_n": return_n,
                "k_step_return_m": return_m,
            }
        )
    operator_mean_n = returns_n.mean(dim=-1)
    operator_mean_m = returns_m.mean(dim=-1)
    operator_se_n = returns_n.std(dim=-1, unbiased=True) / math.sqrt(
        config.mc_repeats
    )
    operator_se_m = returns_m.std(dim=-1, unbiased=True) / math.sqrt(
        config.mc_repeats
    )
    signed_residual_n = start_values_n.to(torch.float64) - operator_mean_n
    signed_residual_m = start_values_m.to(torch.float64) - operator_mean_m
    return (
        {
            "returns_n": returns_n,
            "returns_m": returns_m,
            "operator_mean_n": operator_mean_n,
            "operator_mean_m": operator_mean_m,
            "operator_standard_error_n": operator_se_n,
            "operator_standard_error_m": operator_se_m,
            "signed_residual_n": signed_residual_n,
            "signed_residual_m": signed_residual_m,
        },
        raw_rows,
    )


def _depth_raw(
    path: _PathEvaluation,
    states: Sequence[AugmentedDiagnosticState],
    config: PersistentDiagnosticConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    h = path.latent_h.to(torch.float64)
    l = path.latent_l.to(torch.float64)
    values = path.values.to(torch.float64)
    latent_norms = torch.sqrt(
        h.pow(2).sum(dim=tuple(range(2, h.ndim)))
        + l.pow(2).sum(dim=tuple(range(2, l.ndim)))
    )
    increments = torch.sqrt(
        (h[:, 1:] - h[:, :-1]).pow(2).sum(dim=tuple(range(2, h.ndim)))
        + (l[:, 1:] - l[:, :-1]).pow(2).sum(dim=tuple(range(2, l.ndim)))
    )
    path_length = increments[:, config.inner_unroll_n : config.reference_depth_m].sum(
        dim=-1
    )
    discrepancy = (
        values[:, config.reference_depth_m] - values[:, config.inner_unroll_n]
    ).abs()
    rows: List[Dict[str, Any]] = []
    for state_index, state in enumerate(states):
        for depth in range(config.reference_depth_m + 1):
            row: Dict[str, Any] = {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "scope": FINITE_BATCH_SCOPE,
                "occurrence_id": state.occurrence_id,
                "state_index": state_index,
                "depth": depth,
                "latent_norm": float(latent_norms[state_index, depth].item()),
                "value": float(values[state_index, depth].item()),
            }
            if depth > 0:
                increment = increments[state_index, depth - 1]
                row["increment_from_previous"] = float(increment.item())
                if path.pre_projection_norms is not None:
                    row["pre_projection_joint_norm"] = float(
                        path.pre_projection_norms[state_index, depth - 1].item()
                    )
                if path.projection_active is not None:
                    row["projection_active"] = bool(
                        path.projection_active[state_index, depth - 1].item()
                    )
            if depth > 1:
                denominator = increments[state_index, depth - 2]
                if denominator > config.ratio_denominator_tolerance:
                    row["successive_increment_ratio"] = float(
                        (increments[state_index, depth - 1] / denominator).item()
                    )
                else:
                    row["successive_increment_ratio"] = None
            rows.append(row)
    return rows, {
        "latent_norms": latent_norms,
        "path_length": path_length,
        "depth_discrepancy": discrepancy,
        "increments": increments,
    }


def _carry_metrics(
    states: Sequence[AugmentedDiagnosticState],
    path: _PathEvaluation,
    config: PersistentDiagnosticConfig,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    input_h = torch.cat([state.input_latent.z_H for state in states], dim=0)
    input_l = torch.cat([state.input_latent.z_L for state in states], dim=0)
    observed_h = torch.cat(
        [state.observed_post_unroll_latent.z_H for state in states], dim=0
    )
    observed_l = torch.cat(
        [state.observed_post_unroll_latent.z_L for state in states], dim=0
    )
    recomputed_h = path.latent_h[:, config.inner_unroll_n]
    recomputed_l = path.latent_l[:, config.inner_unroll_n]
    update_distance = _joint_distance(observed_h, observed_l, input_h, input_l)
    recomputation_error = _joint_distance(
        observed_h,
        observed_l,
        recomputed_h,
        recomputed_l,
    )
    summary: Dict[str, Any] = {
        "metric": "persistent_carry_and_clock",
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "state_count": len(states),
        "recurrent_carry_update_distance": summarize_finite_batch_metric(
            update_distance,
            metric="recurrent_carry_update_distance",
        ),
        "observed_to_recomputed_post_unroll_error": summarize_finite_batch_metric(
            recomputation_error,
            metric="observed_to_recomputed_post_unroll_error",
        ),
    }

    linked = [state for state in states if not state.observed_done]
    if linked:
        current_h = torch.cat([state.input_latent.z_H for state in linked])
        current_l = torch.cat([state.input_latent.z_L for state in linked])
        successor_h = torch.cat(
            [state.observed_post_unroll_latent.z_H for state in linked]
        )
        successor_l = torch.cat(
            [state.observed_post_unroll_latent.z_L for state in linked]
        )
        next_latents = [state.next_input_latent for state in linked]
        if any(latent is None for latent in next_latents):
            raise ValueError("Nonterminal carry link is missing its next input latent.")
        next_h = torch.cat(
            [latent.z_H for latent in next_latents if latent is not None]
        )
        next_l = torch.cat(
            [latent.z_L for latent in next_latents if latent is not None]
        )
        next_input_clocks = [state.next_input_remaining_edits for state in linked]
        if any(clock is None for clock in next_input_clocks):
            raise ValueError("Nonterminal carry link is missing its next input clock.")
        current_clock = torch.tensor(
            [_clock_from_snapshot(state.environment_state) for state in linked]
        )
        successor_clock = torch.tensor(
            [
                _clock_from_snapshot(state.successor_environment_state)
                for state in linked
            ]
        )
        summary["adjacent_continuity"] = summarize_carry_continuity(
            current_h,
            current_l,
            successor_h,
            successor_l,
            next_h,
            next_l,
            current_clock,
            successor_clock,
            torch.tensor(
                [clock for clock in next_input_clocks if clock is not None]
            ),
        )
    else:
        summary["adjacent_continuity"] = {
            "metric": "persistent_carry_continuity",
            "scope": FINITE_BATCH_SCOPE,
            "uniform_certificate": False,
            "adjacent_pair_count": 0,
        }
    return summary, {
        "carry_update_distance": update_distance,
        "carry_recomputation_error": recomputation_error,
    }


def _projection_metrics(
    path: _PathEvaluation,
    current_policy: Any,
) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    if path.pre_projection_norms is None or path.projection_active is None:
        return (
            {
                "metric": "radial_projection_activation",
                "scope": FINITE_BATCH_SCOPE,
                "uniform_certificate": False,
                "status": NOT_VERIFIABLE,
                "reason_code": "model_projection_diagnostic_api_unavailable",
            },
            None,
        )
    pre_norms = path.pre_projection_norms.to(torch.float64)
    active = path.projection_active.to(torch.float64)
    per_state_rate = active.mean(dim=-1)
    per_state_max_pre_norm = pre_norms.max(dim=-1).values
    model_config = _config_dict(current_policy.config)
    projection_mode, radius = _projection_contract(model_config)
    summary = {
        "metric": "radial_projection_activation",
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "configured_mode": projection_mode,
        "configured_radius": radius,
        "recurrent_step_count": int(active.numel()),
        "active_step_count": int(active.sum().item()),
        "active_step_rate": float(active.mean().item()),
        "pre_projection_joint_norm": summarize_finite_batch_metric(
            pre_norms,
            metric="pre_projection_joint_norm",
        ),
        "per_state_active_step_rate": summarize_finite_batch_metric(
            per_state_rate,
            metric="per_state_active_step_rate",
        ),
        "per_state_maximum_pre_projection_norm": summarize_finite_batch_metric(
            per_state_max_pre_norm,
            metric="per_state_maximum_pre_projection_norm",
        ),
    }
    return summary, {
        "per_state_active_step_rate": per_state_rate,
        "per_state_maximum_pre_projection_norm": per_state_max_pre_norm,
    }


def _projection_contract(config: Mapping[str, Any]) -> Tuple[str, Optional[float]]:
    """Read the explicit projection contract, with legacy diagnostics fallback."""

    raw_radius = config.get("rl_latent_ball_radius")
    mode = config.get("rl_latent_projection_mode")
    if mode is None:
        # Non-production test doubles and historical diagnostic models predate
        # the explicit mode. Current serialized model configs always contain it.
        mode = (
            "enabled"
            if isinstance(raw_radius, (int, float)) and raw_radius > 0.0
            else "disabled"
        )
    if mode == "disabled":
        return mode, None
    if mode != "enabled" or isinstance(raw_radius, bool) or not isinstance(
        raw_radius, (int, float)
    ):
        raise ValueError("Invalid recurrent projection configuration.")
    radius = float(raw_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Enabled recurrent projection requires a finite R > 0.")
    return mode, radius


def _initial_latent_sensitivity(
    evaluator: Any,
    current_policy: Any,
    states: Sequence[AugmentedDiagnosticState],
    path: _PathEvaluation,
    config: PersistentDiagnosticConfig,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Compare actual carries with reset and registered perturbed carries."""

    device = _model_device(current_policy)
    projection_mode, radius = _projection_contract(
        _config_dict(current_policy.config)
    )
    reset_start_h: List[torch.Tensor] = []
    reset_start_l: List[torch.Tensor] = []
    reset_post_h: List[torch.Tensor] = []
    reset_post_l: List[torch.Tensor] = []
    reset_values: List[torch.Tensor] = []
    reset_probs: List[torch.Tensor] = []
    perturbed_start_h: List[torch.Tensor] = []
    perturbed_start_l: List[torch.Tensor] = []
    perturbed_post_h: List[torch.Tensor] = []
    perturbed_post_l: List[torch.Tensor] = []
    perturbed_values: List[torch.Tensor] = []
    perturbed_probs: List[torch.Tensor] = []

    def evaluate_variant(
        carry: Any,
        x_batch: Mapping[str, torch.Tensor],
        y_batch: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        for _ in range(config.inner_unroll_n):
            carry = current_policy.update_latent(carry, y_batch, x_batch)
        values, _ = evaluator.used_value(x_batch, y_batch, n=0, z=carry)
        distribution, _ = current_policy.policy_dist(
            x_batch,
            y_batch,
            n=0,
            action_mask=masks,
            z=carry,
        )
        return carry, values.reshape(-1), distribution.probs

    for start in range(0, len(states), config.chunk_size):
        chunk = states[start : start + config.chunk_size]
        x_batch = stack_full_tensor_state_fields(
            [state.environment_state["x"] for state in chunk],
            device,
        )
        y_batch = _stack_plans(
            [state.environment_state["y"] for state in chunk],
            device,
        )
        masks = torch.stack(
            [_snapshot_action_mask(state.environment_state) for state in chunk]
        ).to(device)

        reset_start = current_policy.init_latent(x_batch, y_batch)
        reset_post, reset_value, reset_probability = evaluate_variant(
            reset_start,
            x_batch,
            y_batch,
            masks,
        )
        reset_start_h.append(reset_start.z_H.detach().cpu())
        reset_start_l.append(reset_start.z_L.detach().cpu())
        reset_post_h.append(reset_post.z_H.detach().cpu())
        reset_post_l.append(reset_post.z_L.detach().cpu())
        reset_values.append(reset_value.detach().cpu())
        reset_probs.append(reset_probability.detach().cpu())

        perturbed_latents: List[ReplayLatent] = []
        for state in chunk:
            actual_h = state.input_latent.z_H.detach().cpu()
            actual_l = state.input_latent.z_L.detach().cpu()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(
                    config.initial_latent_perturbation_seed,
                    state.occurrence_id,
                    "initial_latent_perturbation",
                )
            )
            noise_h = torch.randn(
                actual_h.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(actual_h.dtype)
            noise_l = torch.randn(
                actual_l.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(actual_l.dtype)
            noise_norm = _joint_norm(noise_h, noise_l).reshape(-1, 1, 1)
            scale = config.initial_latent_perturbation_l2_norm / noise_norm.clamp_min(
                torch.finfo(noise_norm.dtype).tiny
            )
            perturbed_latents.append(
                ReplayLatent(
                    z_H=actual_h + noise_h * scale,
                    z_L=actual_l + noise_l * scale,
                )
            )
        perturbed_start = _stack_replay_latents(perturbed_latents, device)
        if projection_mode == "enabled":
            assert radius is not None
            joint_norm = _joint_norm(
                perturbed_start.z_H,
                perturbed_start.z_L,
            ).reshape(-1, 1, 1)
            safe_norm = torch.where(
                joint_norm > 0.0,
                joint_norm,
                torch.ones_like(joint_norm),
            )
            projection_scale = torch.where(
                joint_norm > radius,
                radius / safe_norm,
                torch.ones_like(joint_norm),
            )
            perturbed_start = ReplayLatent(
                z_H=perturbed_start.z_H * projection_scale,
                z_L=perturbed_start.z_L * projection_scale,
            )
        perturbed_post, perturbed_value, perturbed_probability = evaluate_variant(
            perturbed_start,
            x_batch,
            y_batch,
            masks,
        )
        perturbed_start_h.append(perturbed_start.z_H.detach().cpu())
        perturbed_start_l.append(perturbed_start.z_L.detach().cpu())
        perturbed_post_h.append(perturbed_post.z_H.detach().cpu())
        perturbed_post_l.append(perturbed_post.z_L.detach().cpu())
        perturbed_values.append(perturbed_value.detach().cpu())
        perturbed_probs.append(perturbed_probability.detach().cpu())

    actual_start_h = torch.cat([state.input_latent.z_H for state in states])
    actual_start_l = torch.cat([state.input_latent.z_L for state in states])
    actual_post_h = path.latent_h[:, config.inner_unroll_n]
    actual_post_l = path.latent_l[:, config.inner_unroll_n]
    actual_values = path.values[:, config.inner_unroll_n]
    actual_probs = path.current_probs
    reset_start_h_tensor = torch.cat(reset_start_h)
    reset_start_l_tensor = torch.cat(reset_start_l)
    reset_post_h_tensor = torch.cat(reset_post_h)
    reset_post_l_tensor = torch.cat(reset_post_l)
    reset_values_tensor = torch.cat(reset_values)
    reset_probs_tensor = torch.cat(reset_probs)
    perturbed_start_h_tensor = torch.cat(perturbed_start_h)
    perturbed_start_l_tensor = torch.cat(perturbed_start_l)
    perturbed_post_h_tensor = torch.cat(perturbed_post_h)
    perturbed_post_l_tensor = torch.cat(perturbed_post_l)
    perturbed_values_tensor = torch.cat(perturbed_values)
    perturbed_probs_tensor = torch.cat(perturbed_probs)

    raw = {
        "actual_to_reset_input_distance": _joint_distance(
            actual_start_h,
            actual_start_l,
            reset_start_h_tensor,
            reset_start_l_tensor,
        ),
        "actual_to_perturbed_input_distance": _joint_distance(
            actual_start_h,
            actual_start_l,
            perturbed_start_h_tensor,
            perturbed_start_l_tensor,
        ),
        "actual_to_reset_post_unroll_distance": _joint_distance(
            actual_post_h,
            actual_post_l,
            reset_post_h_tensor,
            reset_post_l_tensor,
        ),
        "actual_to_perturbed_post_unroll_distance": _joint_distance(
            actual_post_h,
            actual_post_l,
            perturbed_post_h_tensor,
            perturbed_post_l_tensor,
        ),
        "actual_to_reset_absolute_value_change": (
            actual_values - reset_values_tensor
        ).abs(),
        "actual_to_perturbed_absolute_value_change": (
            actual_values - perturbed_values_tensor
        ).abs(),
        "actual_to_reset_policy_total_variation": 0.5
        * (actual_probs - reset_probs_tensor).abs().sum(dim=-1),
        "actual_to_perturbed_policy_total_variation": 0.5
        * (actual_probs - perturbed_probs_tensor).abs().sum(dim=-1),
        "actual_to_reset_argmax_mismatch": (
            actual_probs.argmax(dim=-1) != reset_probs_tensor.argmax(dim=-1)
        ).to(torch.float64),
        "actual_to_perturbed_argmax_mismatch": (
            actual_probs.argmax(dim=-1) != perturbed_probs_tensor.argmax(dim=-1)
        ).to(torch.float64),
    }
    summary: Dict[str, Any] = {
        "metric": "initial_latent_sensitivity",
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "perturbation_protocol": {
            "kind": "occurrence_seeded_joint_gaussian_direction",
            "requested_l2_norm": config.initial_latent_perturbation_l2_norm,
            "seed": config.initial_latent_perturbation_seed,
            "projection_mode": projection_mode,
            "projected_to_configured_ball": projection_mode == "enabled",
        },
    }
    for name, values in raw.items():
        summary[name] = summarize_finite_batch_metric(values, metric=name)
    return summary, raw


def _clock_strata(
    clocks: torch.Tensor,
    metrics: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    if clocks.ndim != 1 or clocks.numel() == 0:
        raise ValueError("Clock strata require one nonempty clock vector.")
    if bool((clocks < 0).any().item()) or bool((clocks != clocks.round()).any().item()):
        raise ValueError("Clock strata require nonnegative integer clocks.")
    strata: Dict[str, Any] = {}
    for clock in sorted(int(value) for value in clocks.unique().tolist()):
        select = clocks == clock
        entry: Dict[str, Any] = {"state_count": int(select.sum().item())}
        for name, values in metrics.items():
            flat = values.detach().cpu().reshape(-1)
            if flat.shape != clocks.shape:
                raise ValueError(f"Clock metric {name!r} has the wrong shape.")
            entry[name] = summarize_finite_batch_metric(
                flat[select].abs(),
                metric=name,
            )
        strata[str(clock)] = entry
    return {
        "metric": "remaining_budget_strata",
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "model_clock_input": False,
        "interpretation": (
            "The current TRM does not consume remaining_edits directly; the clock "
            "changes environment rewards, terminal masks, and backups."
        ),
        "strata": strata,
    }


def run_persistent_diagnostics(
    *,
    evaluator: Any,
    current_policy: Any,
    candidate_policy: Optional[Any],
    deployment: DeploymentSpec,
    env: Any,
    states: Sequence[AugmentedDiagnosticState],
    config: PersistentDiagnosticConfig,
    endpoint_policy_pair_invariants_verified: bool,
    provenance: Optional[Mapping[str, Any]] = None,
) -> PersistentDiagnosticOutput:
    """Run bounded theorem-facing diagnostics on retained persistent states."""

    retained = list(states)
    if not retained:
        raise ValueError("Persistent diagnostics require at least one retained state.")
    occurrence_ids = [state.occurrence_id for state in retained]
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError("Retained state occurrence IDs must be unique.")
    for state in retained:
        _validate_collected_state(state)
    if not math.isclose(
        float(env.config.gamma), config.gamma, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("Diagnostic gamma does not match the environment.")
    if endpoint_policy_pair_invariants_verified and candidate_policy is None:
        raise ValueError(
            "Verified endpoint policy-pair invariants require a candidate policy."
        )

    map_report = recurrent_map_report(
        evaluator,
        current_policy,
        candidate_policy,
        deployment,
    )
    if endpoint_policy_pair_invariants_verified and not map_report[
        "evaluator_current_candidate_equal"
    ]:
        raise ValueError(
            "Verified endpoint policy-pair invariants require one shared recurrent map."
        )
    initial_map_hashes = dict(map_report["hashes"])
    runtime_models = [evaluator, current_policy, candidate_policy]
    if deployment.kind == "concrete_model":
        runtime_models.append(deployment.model)
    concrete_models = [model for model in runtime_models if model is not None]
    devices = {str(_model_device(model)) for model in concrete_models}
    if len(devices) != 1:
        raise ValueError(
            "Evaluator, policy pair, and concrete deployment must share one device."
        )

    with _preserve_runtime(runtime_models, env), torch.inference_mode():
        path = _evaluate_paths(
            evaluator,
            current_policy,
            candidate_policy,
            retained,
            config,
            endpoint_policy_pair_invariants_verified=(
                endpoint_policy_pair_invariants_verified
            ),
        )
        exact = _exact_one_step_backups(
            evaluator,
            current_policy,
            env,
            retained,
            path,
            config,
        )
        mc, mc_rows = _monte_carlo_k_step(
            evaluator,
            current_policy,
            env,
            retained,
            path.values[:, config.inner_unroll_n],
            path.values[:, config.reference_depth_m],
            config,
        )

        mixture_gap_summary: Dict[str, Any]
        mixture_gap_raw: Optional[Dict[str, torch.Tensor]] = None
        candidate_depth_discrepancy: Optional[torch.Tensor] = None
        candidate_estimator_depth_discrepancy: Optional[torch.Tensor] = None
        candidate_expected_advantage: Optional[torch.Tensor] = None
        exact_mixture_probs: Optional[torch.Tensor] = None
        deployed_probs: Optional[torch.Tensor] = None
        if endpoint_policy_pair_invariants_verified:
            assert path.candidate_probs is not None
            candidate_probs = path.candidate_probs.to(torch.float64)
            current_probs = path.current_probs.to(torch.float64)
            candidate_depth_discrepancy = (
                (
                    candidate_probs
                    * (
                        exact["raw_advantages_n"]
                        - exact["raw_advantages_m"]
                    )
                )
                .sum(dim=-1)
                .abs()
            )
            candidate_estimator_depth_discrepancy = (
                (candidate_probs * (exact["advantages_n"] - exact["advantages_m"]))
                .sum(dim=-1)
                .abs()
            )
            candidate_expected_advantage = (
                candidate_probs * exact["advantages_n"]
            ).sum(dim=-1)
            exact_mixture_probs = (
                1.0 - config.mixture_alpha
            ) * current_probs + config.mixture_alpha * candidate_probs
            if deployment.kind in {"policy_dist_callback", "concrete_model"}:
                deployed_probs = _evaluate_deployed_probs(
                    deployment,
                    retained,
                    path.action_masks,
                    config,
                    _model_device(current_policy),
                ).to(torch.float64)
            else:
                deployed_probs = None
            if deployed_probs is None:
                mixture_gap_summary = {
                    "metric": "deployment_policy_gap",
                    "scope": FINITE_BATCH_SCOPE,
                    "uniform_certificate": False,
                    "status": NOT_VERIFIABLE,
                    "reason_code": "deployed_policy_missing",
                }
            else:
                mixture_gap_summary = summarize_policy_gap(
                    exact_mixture_probs,
                    deployed_probs,
                    action_mask=path.action_masks,
                    reference_label="exact_probability_mixture",
                    comparison_label=deployment.label,
                    probability_tolerance=config.probability_tolerance,
                )
                if deployment.kind == "policy_dist_callback":
                    mixture_gap_summary.update(
                        {
                            "measurement_kind": (
                                "finite_batch_production_distribution_callback"
                            ),
                            "deployment_distribution_callback_tested": True,
                            "full_episode_evaluation_loop_tested": False,
                        }
                    )
                else:
                    mixture_gap_summary["measurement_kind"] = (
                        "finite_batch_concrete_policy_comparison"
                    )
                mixture_gap_raw = _policy_gap_raw(
                    exact_mixture_probs,
                    deployed_probs,
                    path.action_masks,
                )
        else:
            mixture_gap_summary = {
                "metric": "deployment_policy_gap",
                "scope": FINITE_BATCH_SCOPE,
                "uniform_certificate": False,
                "status": NOT_VERIFIABLE,
                "reason_code": "endpoint_policy_pair_invariants_unverified",
            }

        depth_summary = summarize_depth_path(
            path.latent_h,
            path.latent_l,
            path.values,
            n_depth=config.inner_unroll_n,
            m_depth=config.reference_depth_m,
            ratio_denominator_tolerance=config.ratio_denominator_tolerance,
        )
        depth_rows, depth_raw = _depth_raw(path, retained, config)
        carry_summary, carry_raw = _carry_metrics(retained, path, config)
        projection_summary, projection_raw = _projection_metrics(
            path,
            current_policy,
        )
        initial_sensitivity_summary, initial_sensitivity_raw = (
            _initial_latent_sensitivity(
                evaluator,
                current_policy,
                retained,
                path,
                config,
            )
        )

    final_map_report = recurrent_map_report(
        evaluator,
        current_policy,
        candidate_policy,
        deployment,
    )
    if final_map_report["hashes"] != initial_map_hashes:
        raise RuntimeError("Persistent diagnostics mutated a recurrent map.")
    map_report["unchanged_after_diagnostics"] = True

    one_step_abs = exact["signed_residual_n"].abs()
    one_step_reference_abs = exact["signed_residual_m"].abs()
    mc_abs = mc["signed_residual_n"].abs()
    mc_reference_abs = mc["signed_residual_m"].abs()
    one_step_summary = summarize_residual_estimates(
        one_step_abs,
        estimator="exact_action_sum_one_step",
        horizon=1,
    )
    one_step_summary["transition_oracle"] = "PlanEditEnv.step from full snapshot"
    one_step_reference_summary = summarize_residual_estimates(
        one_step_reference_abs,
        estimator="exact_action_sum_one_step_reference_depth",
        horizon=1,
    )
    one_step_reference_summary.update(
        {
            "reference_depth_m": config.reference_depth_m,
            "transition_oracle": "PlanEditEnv.step from full snapshot",
        }
    )
    mc_summary = summarize_residual_estimates(
        mc_abs,
        estimator="monte_carlo_k_step",
        horizon=config.k_horizon,
        monte_carlo_standard_errors=mc["operator_standard_error_n"],
    )
    mc_reference_summary = summarize_residual_estimates(
        mc_reference_abs,
        estimator="monte_carlo_k_step_reference_depth",
        horizon=config.k_horizon,
        monte_carlo_standard_errors=mc["operator_standard_error_m"],
    )
    mc_reference_summary.update(
        {
            "reference_depth_m": config.reference_depth_m,
            "repeat_count_per_state": config.mc_repeats,
            "seed": config.mc_seed,
            "standard_error_target": "monte_carlo_operator_mean",
            "absolute_residual_is_nonlinear_plugin_estimate": True,
            "transition_oracle": "PlanEditEnv.step from full snapshot",
        }
    )
    mc_summary.update(
        {
            "repeat_count_per_state": config.mc_repeats,
            "seed": config.mc_seed,
            "standard_error_target": "monte_carlo_operator_mean",
            "absolute_residual_is_nonlinear_plugin_estimate": True,
            "transition_oracle": "PlanEditEnv.step from full snapshot",
        }
    )
    centering_summary = summarize_centering_defect(
        path.current_probs,
        exact["advantages_n"],
        action_mask=path.action_masks,
        probability_tolerance=config.probability_tolerance,
    )
    production_baseline_parity = {
        "metric": "production_exact_baseline_oracle_parity",
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "oracle": "PlanEditEnv.step from full snapshot",
        "production_path": "utils.lipschitz.compute_exact_baseline_summation",
        "maximum_action_q_absolute_error_per_state": (
            summarize_finite_batch_metric(
                exact["production_q_max_absolute_error"],
                metric="maximum_action_q_absolute_error_per_state",
            )
        ),
        "baseline_absolute_error": summarize_finite_batch_metric(
            exact["production_baseline_absolute_error"],
            metric="baseline_absolute_error",
        ),
    }
    if candidate_depth_discrepancy is None:
        candidate_summary: Dict[str, Any] = {
            "metric": "finite_depth_candidate_advantage_discrepancy",
            "scope": FINITE_BATCH_SCOPE,
            "uniform_certificate": False,
            "status": NOT_VERIFIABLE,
            "reason_code": "endpoint_policy_pair_invariants_unverified",
        }
    else:
        assert candidate_estimator_depth_discrepancy is not None
        candidate_summary = {
            "metric": "finite_depth_candidate_advantage_discrepancy",
            "scope": FINITE_BATCH_SCOPE,
            "uniform_certificate": False,
            "definition": (
                "abs(E_candidate[A_n - A_m]) on each retained state; this is "
                "not epsilon_A,candidate against the unknown true advantage"
            ),
            "theorem_candidate_bias_to_true_advantage": {
                "status": NOT_VERIFIABLE,
            },
            "raw_exact_action_advantage_discrepancy": summarize_finite_batch_metric(
                candidate_depth_discrepancy,
                metric="raw_exact_action_advantage_discrepancy",
            ),
            "configured_estimator_advantage_discrepancy": (
                summarize_finite_batch_metric(
                    candidate_estimator_depth_discrepancy,
                    metric="configured_estimator_advantage_discrepancy",
                )
            ),
        }

    clocks = torch.tensor(
        [_clock_from_snapshot(state.environment_state) for state in retained],
        dtype=torch.float64,
    )
    clock_metrics: Dict[str, torch.Tensor] = {
        "one_step_absolute_residual": one_step_abs,
        "one_step_reference_depth_absolute_residual": one_step_reference_abs,
        "mc_k_step_absolute_residual": mc_abs,
        "mc_k_step_reference_depth_absolute_residual": mc_reference_abs,
        "centering_defect": exact["centering_defect"],
        "path_length_n_to_m": depth_raw["path_length"],
        "depth_value_discrepancy": depth_raw["depth_discrepancy"],
        "input_latent_norm": depth_raw["latent_norms"][:, 0],
        "post_unroll_latent_norm": depth_raw["latent_norms"][:, config.inner_unroll_n],
        "carry_update_distance": carry_raw["carry_update_distance"],
    }
    if candidate_depth_discrepancy is not None:
        clock_metrics["finite_depth_candidate_advantage_discrepancy"] = (
            candidate_depth_discrepancy
        )
    if projection_raw is not None:
        clock_metrics["projection_active_step_rate"] = projection_raw[
            "per_state_active_step_rate"
        ]
        clock_metrics["maximum_pre_projection_norm"] = projection_raw[
            "per_state_maximum_pre_projection_norm"
        ]
    for name in (
        "actual_to_reset_post_unroll_distance",
        "actual_to_perturbed_post_unroll_distance",
        "actual_to_reset_absolute_value_change",
        "actual_to_perturbed_absolute_value_change",
    ):
        clock_metrics[name] = initial_sensitivity_raw[name]
    clock_summary = _clock_strata(clocks, clock_metrics)

    state_rows: List[Dict[str, Any]] = []
    for index, state in enumerate(retained):
        row: Dict[str, Any] = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "scope": FINITE_BATCH_SCOPE,
            "state_index": index,
            "occurrence_id": state.occurrence_id,
            "augmented_state_hash": state.augmented_state_hash,
            "source_record_index": state.source_record_index,
            "episode_id": state.episode_id,
            "timestep": state.timestep,
            "remaining_edits": int(clocks[index].item()),
            "value_n": float(path.values[index, config.inner_unroll_n].item()),
            "value_m": float(path.values[index, config.reference_depth_m].item()),
            "exact_one_step_backup": float(exact["backup_n"][index].item()),
            "exact_one_step_signed_residual": float(
                exact["signed_residual_n"][index].item()
            ),
            "exact_one_step_absolute_residual": float(one_step_abs[index].item()),
            "exact_one_step_reference_backup": float(
                exact["backup_m"][index].item()
            ),
            "exact_one_step_reference_signed_residual": float(
                exact["signed_residual_m"][index].item()
            ),
            "exact_one_step_reference_absolute_residual": float(
                one_step_reference_abs[index].item()
            ),
            "centering_defect": float(exact["centering_defect"][index].item()),
            "production_q_max_absolute_error": float(
                exact["production_q_max_absolute_error"][index].item()
            ),
            "production_baseline_absolute_error": float(
                exact["production_baseline_absolute_error"][index].item()
            ),
            "mc_k_step_operator_mean": float(
                mc["operator_mean_n"][index].item()
            ),
            "mc_k_step_operator_standard_error": float(
                mc["operator_standard_error_n"][index].item()
            ),
            "mc_k_step_signed_residual": float(
                mc["signed_residual_n"][index].item()
            ),
            "mc_k_step_absolute_residual": float(mc_abs[index].item()),
            "mc_k_step_reference_operator_mean": float(
                mc["operator_mean_m"][index].item()
            ),
            "mc_k_step_reference_operator_standard_error": float(
                mc["operator_standard_error_m"][index].item()
            ),
            "mc_k_step_reference_signed_residual": float(
                mc["signed_residual_m"][index].item()
            ),
            "mc_k_step_reference_absolute_residual": float(
                mc_reference_abs[index].item()
            ),
            "path_length_n_to_m": float(depth_raw["path_length"][index].item()),
            "depth_value_discrepancy": float(
                depth_raw["depth_discrepancy"][index].item()
            ),
            "input_latent_norm": float(depth_raw["latent_norms"][index, 0].item()),
            "post_unroll_latent_norm": float(
                depth_raw["latent_norms"][index, config.inner_unroll_n].item()
            ),
            "carry_update_distance": float(
                carry_raw["carry_update_distance"][index].item()
            ),
            "carry_recomputation_error": float(
                carry_raw["carry_recomputation_error"][index].item()
            ),
        }
        if (
            candidate_depth_discrepancy is not None
            and candidate_estimator_depth_discrepancy is not None
            and candidate_expected_advantage is not None
        ):
            row["finite_depth_candidate_advantage_discrepancy"] = float(
                candidate_depth_discrepancy[index].item()
            )
            row["configured_estimator_candidate_advantage_discrepancy"] = float(
                candidate_estimator_depth_discrepancy[index].item()
            )
            row["candidate_expected_estimated_advantage"] = float(
                candidate_expected_advantage[index].item()
            )
        if mixture_gap_raw is not None:
            row["exact_mixture_total_variation_to_deployed"] = float(
                mixture_gap_raw["total_variation"][index].item()
            )
            row["exact_mixture_kl_to_deployed"] = float(
                mixture_gap_raw["kl_reference_to_comparison"][index].item()
            )
            row["deployed_kl_to_exact_mixture"] = float(
                mixture_gap_raw["kl_comparison_to_reference"][index].item()
            )
            row["exact_mixture_support_mismatch"] = bool(
                mixture_gap_raw["support_mismatch"][index].item()
            )
        if projection_raw is not None:
            row["projection_active_step_rate"] = float(
                projection_raw["per_state_active_step_rate"][index].item()
            )
            row["maximum_pre_projection_norm"] = float(
                projection_raw["per_state_maximum_pre_projection_norm"]
                [index]
                .item()
            )
        row["initial_latent_sensitivity"] = {
            name: float(values[index].item())
            for name, values in initial_sensitivity_raw.items()
        }
        state_rows.append(row)

    action_rows: List[Dict[str, Any]] = []
    candidate_probs_for_rows = (
        path.candidate_probs.to(torch.float64)
        if path.candidate_probs is not None
        else None
    )
    for state_index, state in enumerate(retained):
        for action_index in range(path.action_masks.shape[1]):
            valid = bool(path.action_masks[state_index, action_index].item())
            action_rows.append(
                {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "scope": FINITE_BATCH_SCOPE,
                    "state_index": state_index,
                    "occurrence_id": state.occurrence_id,
                    "augmented_state_hash": state.augmented_state_hash,
                    "action_index": action_index,
                    "valid_action": valid,
                    "current_policy_probability": float(
                        path.current_probs[state_index, action_index].item()
                    ),
                    "candidate_policy_probability": (
                        float(
                            candidate_probs_for_rows[
                                state_index, action_index
                            ].item()
                        )
                        if candidate_probs_for_rows is not None
                        else None
                    ),
                    "exact_mixture_probability": (
                        float(
                            exact_mixture_probs[state_index, action_index].item()
                        )
                        if exact_mixture_probs is not None
                        else None
                    ),
                    "deployed_policy_probability": (
                        float(deployed_probs[state_index, action_index].item())
                        if deployed_probs is not None
                        else None
                    ),
                    "q_n": (
                        float(exact["q_n"][state_index, action_index].item())
                        if valid
                        else None
                    ),
                    "production_q_n": (
                        float(
                            exact["production_q_n"]
                            [state_index, action_index]
                            .item()
                        )
                        if valid
                        else None
                    ),
                    "q_m": (
                        float(exact["q_m"][state_index, action_index].item())
                        if valid
                        else None
                    ),
                    "raw_advantage_n": (
                        float(
                            exact["raw_advantages_n"]
                            [state_index, action_index]
                            .item()
                        )
                        if valid
                        else None
                    ),
                    "raw_advantage_m": (
                        float(
                            exact["raw_advantages_m"]
                            [state_index, action_index]
                            .item()
                        )
                        if valid
                        else None
                    ),
                    "configured_estimator_advantage_n": (
                        float(
                            exact["advantages_n"][state_index, action_index].item()
                        )
                        if valid
                        else None
                    ),
                    "configured_estimator_advantage_m": (
                        float(
                            exact["advantages_m"][state_index, action_index].item()
                        )
                        if valid
                        else None
                    ),
                }
            )

    summary = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
        "model_clock_input": False,
        "absorbing_boundary": {
            "terminal_bootstrap": 0.0,
            "reward_shaping_enabled": bool(env.config.reward_shaping),
            "manuscript_negative_cmax_tail_folded_into_terminal_reward": bool(
                env.config.reward_shaping
            ),
        },
        "state_count": len(retained),
        "state_occurrence_ids_sha256": _hash_sequence(occurrence_ids),
        "config": asdict(config),
        "provenance": dict(provenance or {}),
        "endpoint_policy_pair_invariants_verified": (
            endpoint_policy_pair_invariants_verified
        ),
        "checkpoint_training_history_verified": False,
        "recurrent_map_identity": map_report,
        "exact_one_step_augmented_residual": one_step_summary,
        "exact_one_step_reference_depth_residual": one_step_reference_summary,
        "monte_carlo_k_step_augmented_residual": mc_summary,
        "monte_carlo_k_step_reference_depth_residual": mc_reference_summary,
        "statewise_centering": {
            **centering_summary,
            "construction": (
                "recomputed from exact action summation, configured clipping, "
                "and statewise recentering"
            ),
            "saved_training_time_estimator": {"status": NOT_VERIFIABLE},
        },
        "production_exact_baseline_oracle_parity": production_baseline_parity,
        "finite_depth_candidate_advantage_discrepancy": candidate_summary,
        "exact_mixture_deployment_gap": mixture_gap_summary,
        "finite_reference_depth_path": depth_summary,
        "persistent_carry_and_clock": carry_summary,
        "radial_projection_activation": projection_summary,
        "initial_latent_sensitivity": initial_sensitivity_summary,
        "remaining_budget_strata": clock_summary,
        "omitted_metrics": (
            [
                "local or global Lipschitz estimates",
                "uniform policy-pair closure enumeration",
            ]
            + (
                ["projection-active rate and pre-projection norms"]
                if projection_raw is None
                else []
            )
        ),
    }
    return PersistentDiagnosticOutput(
        summary=summary,
        state_rows=state_rows,
        action_rows=action_rows,
        depth_rows=depth_rows,
        mc_rows=mc_rows,
    )
