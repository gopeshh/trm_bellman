#!/usr/bin/env fbpython
"""Sealed learned backend for registered policy-improvement runs.

The backend is intentionally not a command-line program.  The authenticated
pre-import entrypoint constructs it from launcher attestation and injects it
into ``scripts.policy_improvement_full_runtime``.  A source-tree invocation of
the runtime therefore still has no backend.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import (
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace as dataclass_replace,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import torch
import yaml
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from policy_improvement_non_smoke_checkpoint import (
    evaluate_without_mutation,
    FullCheckpointArtifact,
    publish_and_validate_full_checkpoint,
    session_model_state_identity,
    validate_full_checkpoint_identity,
)
from policy_improvement_sealed_evidence import (
    authenticate_sealed_descriptor,
    SealedCheckpointError,
)
from policy_improvement_smoke_runtime import (
    _evaluation_payload as evaluate_registered_policy,
    _variant_specs as registered_variant_specs,
    build_stage0_validation_session,
    SmokeContext,
    SmokeSession,
    stage0_model_state_identity,
    validate_policy_improvement_smoke_identity,
)
from rl.batch_utils import prepare_batch_x, prepare_plan
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from rl.upi_trm_trainer import _clip_and_recenter_advantages, UPITrmTrainer
from scripts.policy_improvement_full_runtime import (
    BackendPackage,
    BackendRequest,
    FULL_EXECUTION_ENV,
    FULL_EXECUTION_VALUE,
    FullRunBackend,
    FullRunFailure,
    FullRuntimeError,
    load_registered_full_run,
    RegisteredFullRun,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    policy_variants_for_method,
    primary_policy_variant_for_method,
    RESULT_SCHEMA_VERSION,
    SCHEMA_NAME,
)
from utils.compute_accounting import (
    add_model_counters,
    build_training_compute_accounting,
    capture_model_compute_state,
    restore_model_compute_state,
    validate_compute_snapshot,
    zero_model_counters,
)
from utils.dataset_provenance import (
    build_dataset_provenance,
    dataset_input_sha256s,
    dataset_pool_sha256,
    dataset_puzzle_identifier_sha256s,
    dataset_sample_sha256s,
    dataset_source_build_metadata,
    ordered_record_sha256,
)
from utils.lipschitz import compute_exact_baseline_summation
from utils.run_identity import (
    build_run_identity,
    canonical_json_sha256,
    file_sha256,
)


_THEORY_TRAINING_RUNTIME_ROLES: frozenset[str] = frozenset(
    {"policy-improvement-full", "policy-improvement-smoke"}
)
_LOWER_HEX: frozenset[str] = frozenset("0123456789abcdef")


class FullBackendError(RuntimeError):
    """Raised when the sealed backend cannot execute the registered row."""


@dataclass(frozen=True)
class SealedRuntimeIdentity:
    """Launcher-authenticated identity captured before behavior imports."""

    role: str
    runtime_sha256: str
    source_git_commit: str
    source_manifest_sha256: str
    producer_source_manifest_sha256: str
    runtime_profile_sha256: str
    selected_source_manifest_sha256: str
    runtime_authorization_sha256: str
    launcher_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SealedRuntimeIdentity:
        fields = {
            "role",
            "runtime_sha256",
            "source_git_commit",
            "source_manifest_sha256",
            "producer_source_manifest_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
            "runtime_authorization_sha256",
            "launcher_sha256",
        }
        if set(value) != fields:
            raise FullBackendError("Sealed runtime identity field inventory differs.")
        role = value["role"]
        if role not in _THEORY_TRAINING_RUNTIME_ROLES:
            raise FullBackendError("Sealed runtime has the wrong launcher role.")
        for name in (
            "runtime_sha256",
            "source_manifest_sha256",
            "producer_source_manifest_sha256",
            "runtime_profile_sha256",
            "selected_source_manifest_sha256",
            "runtime_authorization_sha256",
            "launcher_sha256",
        ):
            _require_digest(value[name], name=name)
        source_commit = value["source_git_commit"]
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in _LOWER_HEX for character in source_commit)
        ):
            raise FullBackendError("Sealed runtime source commit is invalid.")
        if value["runtime_profile_sha256"] != value["source_manifest_sha256"]:
            raise FullBackendError("Runtime profile differs from its source manifest.")
        if value["selected_source_manifest_sha256"] != value["source_manifest_sha256"]:
            raise FullBackendError(
                "Selected training source differs from its authenticated profile."
            )
        return cls(**{name: str(value[name]) for name in fields})


class LearnedRunEngine(Protocol):
    """Internal engine used by the sealed backend and synthetic unit tests."""

    def execute(
        self,
        request: BackendRequest,
        runtime: SealedRuntimeIdentity,
    ) -> BackendPackage: ...


@dataclass
class LearnedSession:
    """One live training session or one disposable restored evaluator."""

    run: RegisteredFullRun
    model: Any
    trainer: Any
    rl_config: Any
    env_config: PlanEditEnvConfig
    train_dataset: Any
    evaluation_dataset: Any
    checker: Any
    task_config: Any
    dataset_provenance: dict[str, Any]
    effective_config: dict[str, Any]
    effective_config_sha256: str
    initialization_sha256: str
    device: torch.device
    config_path: Path
    method_config_sha256: str
    run_identity: dict[str, Any] | None
    evidence_identity: dict[str, Any]
    dataset_manifest_sha256: str
    train_ordered_records_sha256: str
    evaluation_ordered_records_sha256: str
    evaluation_pool_sha256: str
    evaluation_state_dicts: dict[str, dict[str, object]] = field(default_factory=dict)
    parent_environment_interactions: int | None = None
    evaluation_started: bool = False
    checkpoint_wall_time_seconds: float = 0.0
    external_evaluation_wall_time_seconds: float = 0.0
    external_evaluation_model_work: dict[str, int] = field(
        default_factory=zero_model_counters
    )


@dataclass(frozen=True)
class _TrainingUtilization:
    device_type: str
    sampling_interval_seconds: float | None
    samples: tuple[float, ...]


@dataclass
class _Snapshot:
    kind: str
    checkpoint: FullCheckpointArtifact
    environment_interactions: int
    recurrent_map_applications: int
    accelerator_seconds: float
    policy_evaluations: list[dict[str, object]]
    lineage_sha256: str


@dataclass(frozen=True)
class _EvaluationOutcome:
    policy_evaluations: list[dict[str, object]]
    model_work: dict[str, int]
    wall_time_seconds: float


def _require_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise FullBackendError(f"{name} must be 64 lowercase hexadecimal characters.")
    return value


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise FullBackendError("JSON artifact write made no progress.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def required_content_splits(run: RegisteredFullRun) -> tuple[str, str]:
    """Return the only dataset splits this row may deserialize."""

    split = str(run.row["evaluation_split"])
    if split == "validation":
        if run.test_open_sha256 is not None:
            raise FullBackendError("Validation execution cannot bind TEST_OPEN.json.")
        return ("train", "validation")
    if split == "test":
        if run.test_open_sha256 is None:
            raise FullBackendError(
                "Test content cannot be opened without authenticated TEST_OPEN.json."
            )
        return ("train", "test")
    raise FullBackendError("Registered evaluation split is unsupported.")


def load_authorized_dataset_splits(
    run: RegisteredFullRun,
    loader: Callable[[str], tuple[Any, int, int, int]],
) -> tuple[tuple[Any, int, int, int], tuple[Any, int, int, int]]:
    """Load exactly train plus the row-authorized evaluation split."""

    train_split, evaluation_split = required_content_splits(run)
    return loader(train_split), loader(evaluation_split)


def calibrate_training_split_throughput(
    session: LearnedSession,
    *,
    environment_interactions: int,
) -> dict[str, object]:
    """Run one mechanics-only synthetic/training calibration step."""

    if (
        isinstance(environment_interactions, bool)
        or not isinstance(environment_interactions, int)
        or environment_interactions <= 0
    ):
        raise FullBackendError("Calibration interaction count must be positive.")
    if session.run.row.get("evaluation_split") == "test" or session.evaluation_started:
        raise FullBackendError("Throughput calibration cannot open evaluation results.")
    before_interactions = int(session.trainer.get_env_step_count())
    if before_interactions != 0:
        raise FullBackendError("Throughput calibration requires a fresh session.")
    before_recurrent = TorchLearnedRunEngine._recurrent_work(session)
    sampler = _GpuTrainingSampler(session.device)
    sampler.start()
    started = time.perf_counter()
    try:
        TorchLearnedRunEngine._train_step(session, environment_interactions)
    finally:
        elapsed = time.perf_counter() - started
        utilization = sampler.stop()
    after_interactions = int(session.trainer.get_env_step_count())
    after_recurrent = TorchLearnedRunEngine._recurrent_work(session)
    interaction_delta = after_interactions - before_interactions
    recurrent_delta = after_recurrent - before_recurrent
    if interaction_delta != environment_interactions or recurrent_delta < 0:
        raise FullBackendError("Calibration trainer violated its mechanics-only cap.")
    compute_snapshot = TorchLearnedRunEngine._training_compute(session)
    compute_accounting = build_training_compute_accounting(
        compute_snapshot,
        device=session.device,
        checkpoint_seconds=0.0,
        cuda_utilization={
            "device_type": utilization.device_type,
            "sampling_interval_seconds": utilization.sampling_interval_seconds,
            "samples": list(utilization.samples),
        },
    )
    return {
        "schema_name": "policy_improvement_training_throughput_smoke_v2",
        "schema_version": 2,
        "source": "synthetic_or_training_split_only",
        "environment_interactions": interaction_delta,
        "recurrent_map_applications": recurrent_delta,
        "elapsed_seconds": elapsed,
        "compute_snapshot": compute_snapshot,
        "compute_accounting": compute_accounting,
        "scientific_selection": False,
        "test_data_opened": False,
    }


class _GpuTrainingSampler:
    def __init__(self, device: torch.device, interval_seconds: float = 0.1) -> None:
        self._device = device
        self._interval_seconds = interval_seconds
        self._samples: list[float] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._active = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        utilization = getattr(torch.cuda, "utilization", None)
        if not callable(utilization):
            raise FullBackendError("CUDA utilization sampling is unavailable.")
        sample = float(utilization(self._device)) / 100.0
        if not math.isfinite(sample) or not 0.0 <= sample <= 1.0:
            raise FullBackendError("CUDA utilization sample is outside [0, 1].")
        self._samples.append(sample)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            if not self._active.is_set():
                continue
            try:
                self._sample()
            except BaseException as exc:
                self._error = exc
                self._stop.set()

    def start(self) -> None:
        if self._device.type != "cuda":
            return
        self._sample()
        self._active.set()
        self._thread = threading.Thread(
            target=self._run,
            name="policy-full-gpu-utilization",
            daemon=True,
        )
        self._thread.start()

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        if self._device.type == "cuda":
            self._active.set()

    def stop(self) -> _TrainingUtilization:
        if self._device.type != "cuda":
            return _TrainingUtilization(self._device.type, None, ())
        self._stop.set()
        self._active.clear()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise FullBackendError("CUDA utilization sampling failed.") from self._error
        if len(self._samples) < 2:
            raise FullBackendError("CUDA utilization requires two interval samples.")
        return _TrainingUtilization(
            "cuda", self._interval_seconds, tuple(self._samples)
        )


class TorchLearnedRunEngine:
    """Production engine built from the authenticated training module."""

    def __init__(self, training_module: Any) -> None:
        self._module = training_module

    def _task_config(
        self, rl_config: Any, dataset: Any, seq_len: int
    ) -> tuple[Any, Any, str]:
        resolution = self._module.resolve_checker_from_dataset(
            rl_cfg=rl_config,
            dataset=dataset,
            seq_len=seq_len,
        )
        task = None
        try:
            from rl.task_config import get_task_config

            if resolution.checker_kind in {
                "solution",
                "constraint",
                "progress",
                "feasibility",
            }:
                task = get_task_config(
                    "sudoku",
                    disable_constraint_masking=(rl_config.disable_constraint_masking),
                )
        except ImportError:
            task = None
        return resolution.checker_fn, task, resolution.checker_kind

    def _build_session(
        self,
        run: RegisteredFullRun,
        runtime: SealedRuntimeIdentity,
    ) -> LearnedSession:
        if run.dataset_root is None:
            raise FullBackendError("The authenticated launcher omitted dataset root.")
        self._module.set_global_seed(int(run.row["seed"]))
        base_method = str(run.row["base_method_id"])
        methods = [
            method for method in run.protocol["methods"] if method["id"] == base_method
        ]
        if len(methods) != 1:
            raise FullBackendError("Registered base method does not resolve uniquely.")
        method = methods[0]
        config_path = (run.project_root / str(method["config_path"])).resolve(
            strict=True
        )
        method_config_sha256 = _require_digest(
            method["config_sha256"], name="method config SHA-256"
        )
        if file_sha256(config_path) != method_config_sha256:
            raise FullBackendError("Method config differs from the protocol digest.")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, Mapping):
            raise FullBackendError("Method config must contain one mapping.")
        config_layer = dict(raw_config)
        applied_layer = {**config_layer, **dict(run.row["config_override"])}
        if (
            canonical_json_sha256(applied_layer)
            != run.row["expected_effective_config_sha256"]
        ):
            raise FullBackendError("Applied config differs from the registered row.")
        if applied_layer.get("track_theory_metrics") is not False:
            raise FullBackendError("Full runs must not enable training theory metrics.")
        base_config = self._module.RLConfig(
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
        merged = self._module._config_dict(base_config)
        merged = self._module.merge_rl_config_layer(merged, config_layer)
        merged.update(dict(run.row["config_override"]))
        merged.update(
            {
                "eval_num_episodes": run.evaluation_records,
                "use_tqdm": False,
                "debug_checks": False,
                "track_theory_metrics": False,
            }
        )
        rl_config = self._module.RLConfig(**merged)

        def load_split(split: str) -> tuple[Any, int, int, int]:
            split_config = run.protocol["dataset"]["splits"][split]
            return self._module.build_dataset_from_paths(
                dataset_paths=[str(run.dataset_root)],
                pool_size=int(split_config["count"]),
                split=split,
            )

        train_values, evaluation_values = load_authorized_dataset_splits(
            run, load_split
        )
        train_dataset, seq_len, vocab_size, train_identifiers = train_values
        (
            evaluation_dataset,
            evaluation_seq_len,
            evaluation_vocab_size,
            evaluation_identifiers,
        ) = evaluation_values
        evaluation_split = str(run.row["evaluation_split"])
        if (evaluation_seq_len, evaluation_vocab_size) != (seq_len, vocab_size):
            raise FullBackendError("Train and evaluation split shapes differ.")
        if set(dataset_input_sha256s(train_dataset)).intersection(
            dataset_input_sha256s(evaluation_dataset)
        ):
            raise FullBackendError("Train and evaluation inputs overlap.")
        self._module.offset_puzzle_identifiers(evaluation_dataset, train_identifiers)
        num_identifiers = train_identifiers + evaluation_identifiers
        for split, dataset in (
            ("train", train_dataset),
            (evaluation_split, evaluation_dataset),
        ):
            expected_manifest = run.protocol["dataset"]["splits"][split][
                "manifest_sha256"
            ]["value"]
            observed_manifest = file_sha256(
                run.dataset_root / "manifests" / f"{split}.json"
            )
            if observed_manifest != expected_manifest:
                raise FullBackendError(f"{split} manifest digest differs.")
            self._module._validate_materialized_split_manifest(
                dataset_root=run.dataset_root,
                split=split,
                registered_sha256=expected_manifest,
                dataset=dataset,
            )
        dataset_manifest_sha256 = file_sha256(run.dataset_root / "MANIFEST.json")
        if (
            dataset_manifest_sha256
            != run.protocol["dataset"]["manifest_sha256"]["value"]
        ):
            raise FullBackendError("Dataset top-level manifest digest differs.")
        checker, task, checker_kind = self._task_config(
            rl_config, train_dataset, seq_len
        )
        env_config = PlanEditEnvConfig(
            max_edits=rl_config.max_edits,
            gamma=rl_config.gamma,
            reward_shaping=rl_config.reward_shaping,
            vocab_size=vocab_size,
            solved_threshold=(
                rl_config.solved_threshold
                if checker_kind in {"solution", "constraint", "progress", "feasibility"}
                else None
            ),
            task_type=rl_config.task_name,
            stop_action_mode=rl_config.stop_action_mode,
            stop_action_penalty=rl_config.stop_action_penalty,
            fail_terminal_reward=rl_config.fail_terminal_reward,
            solve_terminal_reward=rl_config.solve_terminal_reward,
            C_max=rl_config.C_max,
            disable_constraint_masking=rl_config.disable_constraint_masking,
        )
        env = PlanEditEnv(
            dataset=train_dataset,
            checker=checker,
            config=env_config,
            task_config=task,
        )
        action_count = seq_len * vocab_size + 1
        env.set_stop_action_id(action_count - 1)
        sources = dataset_source_build_metadata([str(run.dataset_root)])
        if len(sources) != 1:
            raise FullBackendError("Registered dataset must have one source record.")
        source = sources[0]
        train_records = dataset_sample_sha256s(train_dataset)
        evaluation_population = dataset_sample_sha256s(evaluation_dataset)
        evaluation_records = evaluation_population[: run.evaluation_records]
        train_order = ordered_record_sha256(train_records)
        evaluation_population_order = ordered_record_sha256(evaluation_population)
        if (
            train_order
            != run.protocol["dataset"]["splits"]["train"]["ordered_record_sha256"][
                "value"
            ]
            or evaluation_population_order
            != run.protocol["dataset"]["splits"][evaluation_split][
                "ordered_record_sha256"
            ]["value"]
        ):
            raise FullBackendError("Loaded dataset record order differs from protocol.")
        dataset_provenance = build_dataset_provenance(
            builder_name=str(source["builder_name"]),
            builder_version=source["builder_version"],
            generation_seed=source["generation_seed"],
            train_record_sha256s=train_records,
            eval_record_sha256s=evaluation_records,
            train_split="train",
            eval_split=evaluation_split,
            environment_config=dict(vars(env_config)),
            action_mask_config={
                "task_config_class": type(task).__name__ if task is not None else None,
                "task_config_name": getattr(task, "name", None),
                "disable_constraint_masking": env_config.disable_constraint_masking,
                "stop_action_mode": env._stop_mode,
                "stop_action_id": env.stop_action_id,
                "enable_undo": env._enable_undo,
                "undo_action_id": env.undo_action_id,
                "vocab_size": env.vocab_size,
                "num_actions": action_count,
                "masked_token_ids": [0, 1],
            },
            metadata={
                "source_build_metadata": sources,
                "train_pool_sha256": dataset_pool_sha256(
                    train_dataset, len(train_dataset)
                ),
                "eval_pool_sha256": dataset_pool_sha256(
                    evaluation_dataset, run.evaluation_records
                ),
                "train_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                    dataset_puzzle_identifier_sha256s(train_dataset)
                ),
                "eval_puzzle_identifier_ordered_sha256": ordered_record_sha256(
                    dataset_puzzle_identifier_sha256s(evaluation_dataset)[
                        : run.evaluation_records
                    ]
                ),
                "dataset_source_names": [run.dataset_root.name],
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "num_identifiers": num_identifiers,
                "eval_puzzle_id_offset": train_identifiers,
                "materialization_seed": 0,
            },
        )
        architecture = run.protocol["architecture"]
        model_config = {
            "batch_size": rl_config.batch_size,
            "seq_len": seq_len,
            "puzzle_emb_ndim": 0,
            "puzzle_emb_len": 0,
            "num_puzzle_identifiers": max(num_identifiers, rl_config.batch_size),
            "vocab_size": vocab_size,
            "H_cycles": int(architecture["h_cycles"]),
            "L_cycles": int(architecture["l_cycles"]),
            "H_layers": 0,
            "L_layers": int(architecture["l_layers"]),
            "hidden_size": int(architecture["hidden_size"]),
            "expansion": 2.0,
            "num_heads": max(4, int(architecture["hidden_size"]) // 16),
            "pos_encodings": "rope",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.0,
            "forward_dtype": "float32",
            "mlp_t": False,
            "no_ACT_continue": True,
            "rl_enable_value_head": True,
            "rl_enable_contraction": rl_config.enable_contraction,
            "rl_target_Lz": rl_config.target_Lz,
            "rl_target_Lv": rl_config.target_Lv,
            "rl_disable_value_head_norm": rl_config.disable_value_head_norm,
            "rl_enable_policy_head": True,
            "rl_num_actions": action_count,
            "rl_latent_projection_mode": rl_config.latent_projection_mode,
            "rl_latent_ball_radius": rl_config.latent_ball_radius,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyRecursiveReasoningModel_ACTV1(model_config)
        initialization_sha256 = state_dict_sha256(model.state_dict())
        baseline = self._module.select_baseline_from_configs(None, [str(config_path)])
        trainer = self._module.build_trainer(
            model=model,
            env=env,
            rl_cfg=rl_config,
            device=device,
            baseline_selection=baseline,
            cli_baseline=None,
            verbose=False,
        )
        if base_method == "matched_ppo":
            if type(trainer).__name__ != "PPOTrainer":
                raise FullBackendError("Registered PPO trainer differs.")
        elif not isinstance(trainer, UPITrmTrainer):
            raise FullBackendError("Registered UPI trainer differs.")
        if isinstance(trainer, UPITrmTrainer):
            trainer.set_checker_fn(checker)
        effective_config = {
            "schema_name": "policy_improvement_full_effective_config_v1",
            "protocol_sha256": run.protocol_sha256,
            "registry_row_sha256": run.registry_row_sha256,
            "amendment_history_sha256": run.amendment_history_sha256,
            "run_id": run.row["run_id"],
            "method_id": run.row["method_id"],
            "training_seed": run.row["seed"],
            "runtime_artifact_sha256": runtime.runtime_sha256,
            "source_manifest_sha256": runtime.source_manifest_sha256,
            "rl_config": self._module._config_dict(rl_config),
            "trainer_config": (
                self._module._config_dict(trainer.config)
                if getattr(trainer, "config", None) is not None
                else None
            ),
            "model_config": self._module._config_dict(model.config),
            "dataset_provenance_sha256": canonical_json_sha256(dataset_provenance),
            "config_source_sha256": method_config_sha256,
            "execution_device": self._module._canonical_device(device),
            "interaction_checkpoints": list(run.interaction_checkpoints),
            "compute_target_recurrent_map_applications": (
                run.compute_target_recurrent_map_applications
            ),
            "evaluation_records": run.evaluation_records,
        }
        effective_config_sha256 = canonical_json_sha256(effective_config)
        evidence_identity = {
            "schema_version": 2,
            "run_id": run.row["run_id"],
            "algorithm": ("trm_ppo" if base_method == "matched_ppo" else "upi_trm"),
            "training_seed": run.row["seed"],
            "producer_git_commit": runtime.source_git_commit,
            "effective_config_sha256": effective_config_sha256,
            "effective_config": effective_config,
            "dataset_provenance_sha256": canonical_json_sha256(dataset_provenance),
            "runtime_artifact_sha256": runtime.runtime_sha256,
        }
        run_identity = None
        if str(rl_config.training_protocol) == "fixed_base_exact":
            fixed_args = SimpleNamespace(
                config=[str(config_path)],
                confirmatory_cell=str(run.row["method_id"]),
                confirmatory_tier=str(run.row["tier"]),
                run_id=str(run.row["run_id"]),
                seed=int(run.row["seed"]),
                backbone="trm",
                train_split="train",
                eval_split=evaluation_split,
                env_step_budget=run.final_environment_interactions,
                save_interval=0,
                log_env_interval=run.interaction_checkpoints[0],
                eval_env_interval=run.final_environment_interactions,
                save_env_interval=run.interaction_checkpoints[0],
                puzzle_emb_lr=0.0,
                puzzle_emb_weight_decay=0.0,
                imitation_pretrain=False,
                imitation_epochs=0,
                debug_checks=False,
            )
            exact_effective = self._module._fixed_base_effective_config(
                args=fixed_args,
                rl_config=self._module._config_dict(rl_config),
                model_config=self._module._config_dict(model.config),
                execution_device=self._module._canonical_device(device),
                train_record_count=len(train_dataset),
                eval_record_count=run.evaluation_records,
                dataset_provenance=dataset_provenance,
                initialization_kind="random",
                initialization_artifact_sha256=None,
                registered_assignment={
                    "attempt_index": 0,
                    "registry_sha256": run.registry_sha256,
                },
                runtime_artifact_sha256=runtime.runtime_sha256,
            )
            run_identity = build_run_identity(
                run_id=str(run.row["run_id"]),
                training_seed=int(run.row["seed"]),
                git_lookup_root=run.project_root,
                effective_config=exact_effective,
                dataset_provenance=dataset_provenance,
                initialization_kind="random",
                initialization_artifact_sha256=None,
            )
        session = LearnedSession(
            run=run,
            model=model,
            trainer=trainer,
            rl_config=rl_config,
            env_config=env_config,
            train_dataset=train_dataset,
            evaluation_dataset=evaluation_dataset,
            checker=checker,
            task_config=task,
            dataset_provenance=dataset_provenance,
            effective_config=effective_config,
            effective_config_sha256=effective_config_sha256,
            initialization_sha256=initialization_sha256,
            device=device,
            config_path=config_path,
            method_config_sha256=method_config_sha256,
            run_identity=run_identity,
            evidence_identity=evidence_identity,
            dataset_manifest_sha256=dataset_manifest_sha256,
            train_ordered_records_sha256=train_order,
            evaluation_ordered_records_sha256=ordered_record_sha256(evaluation_records),
            evaluation_pool_sha256=canonical_json_sha256(evaluation_records),
        )
        if run.row["method_id"] == "fixed_base_distilled_realization":

            def capture_distillation_pair() -> None:
                session.evaluation_state_dicts["base"] = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in trainer.policy_model_old.state_dict().items()
                }
                session.evaluation_state_dicts["candidate"] = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in trainer.policy_model_candidate.state_dict().items()
                }

            capture_distillation_pair()
            distill_optimizer = trainer.old_policy_distill_opt
            if distill_optimizer is None:
                raise FullBackendError(
                    "Distilled realization has no deployment optimizer."
                )
            original_distill_step = distill_optimizer.step

            def capture_pair_then_distill(*args: Any, **kwargs: Any) -> Any:
                capture_distillation_pair()
                return original_distill_step(*args, **kwargs)

            distill_optimizer.step = capture_pair_then_distill
        return session

    @staticmethod
    def _training_compute(session: LearnedSession) -> dict[str, Any]:
        raw = session.trainer.compute_accounting_snapshot()
        snapshot = dict(raw)
        model_work = dict(snapshot["model_work"])
        training_work = dict(model_work["training"])
        live_evaluation_work = dict(model_work["evaluation"])
        evaluation_work = add_model_counters(
            live_evaluation_work,
            session.external_evaluation_model_work,
        )
        model_work["training"] = training_work
        model_work["evaluation"] = evaluation_work
        model_work["total"] = add_model_counters(training_work, evaluation_work)
        snapshot["model_work"] = model_work
        wall = dict(snapshot["wall_time_seconds"])
        wall["evaluation"] = float(wall["evaluation"]) + float(
            session.external_evaluation_wall_time_seconds
        )
        snapshot["wall_time_seconds"] = wall
        return validate_compute_snapshot(snapshot)

    @staticmethod
    def _recurrent_work(session: LearnedSession) -> int:
        snapshot = session.trainer.compute_accounting_snapshot()
        return int(snapshot["model_work"]["training"]["recurrent_latent_state_updates"])

    def _checkpoint_identity(
        self,
        session: LearnedSession,
        runtime: SealedRuntimeIdentity,
        *,
        kind: str,
        parent_checkpoint_sha256: str | None,
    ) -> dict[str, object]:
        return {
            "schema_name": "policy_improvement_full_checkpoint_identity_v1",
            "schema_version": 1,
            "run_id": session.run.row["run_id"],
            "method_id": session.run.row["method_id"],
            "protocol_sha256": session.run.protocol_sha256,
            "registry_row_sha256": session.run.registry_row_sha256,
            "amendment_history_sha256": session.run.amendment_history_sha256,
            "runtime_authorization_sha256": (session.run.runtime_authorization_sha256),
            "training_runtime_sha256": runtime.runtime_sha256,
            "training_source_git_commit": runtime.source_git_commit,
            "training_source_manifest_sha256": runtime.source_manifest_sha256,
            "launcher_sha256": runtime.launcher_sha256,
            "dataset_manifest_sha256": session.dataset_manifest_sha256,
            "dataset_provenance_sha256": canonical_json_sha256(
                session.dataset_provenance
            ),
            "effective_config_sha256": session.effective_config_sha256,
            "snapshot_kind": kind,
            "environment_interactions": session.trainer.get_env_step_count(),
            "recurrent_map_applications": self._recurrent_work(session),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "test_open_sha256": session.run.test_open_sha256,
        }

    def _capture_checkpoint(
        self,
        request: BackendRequest,
        runtime: SealedRuntimeIdentity,
        session: LearnedSession,
        *,
        kind: str,
        directory: Path,
        parent_checkpoint_sha256: str | None,
        parent_environment_interactions: int | None,
    ) -> FullCheckpointArtifact:
        session.parent_environment_interactions = parent_environment_interactions
        identity = self._checkpoint_identity(
            session,
            runtime,
            kind=kind,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
        )
        started = time.perf_counter()
        try:
            return publish_and_validate_full_checkpoint(
                training_module=self._module,
                session=session,
                fresh_session_factory=lambda: self._build_session(request.run, runtime),
                checkpoint_directory=directory,
                identity=identity,
                validator_execution_identity={
                    "role": "policy-improvement-training",
                    "source_git_commit": runtime.source_git_commit,
                    "runtime_sha256": runtime.runtime_sha256,
                    "runtime_profile_sha256": runtime.runtime_profile_sha256,
                    "selected_source_manifest_sha256": (
                        runtime.selected_source_manifest_sha256
                    ),
                    "runtime_authorization_sha256": (
                        runtime.runtime_authorization_sha256
                    ),
                    "launcher_sha256": runtime.launcher_sha256,
                },
                evaluation_state_dicts=session.evaluation_state_dicts,
            )
        finally:
            session.checkpoint_wall_time_seconds += time.perf_counter() - started

    @staticmethod
    def _live_fingerprint(session: LearnedSession) -> str:
        model_state, roles = session_model_state_identity(
            session,
            evaluation_state_dicts=session.evaluation_state_dicts,
        )
        return canonical_json_sha256(
            {
                "model_state_sha256": model_state,
                "role_state_sha256s": roles,
                "environment_interactions": session.trainer.get_env_step_count(),
            }
        )

    def _variant_specs(
        self,
        session: LearnedSession,
        restored_states: Mapping[str, Mapping[str, object]],
    ) -> list[tuple[str, Any, Callable[..., Any] | None, bool, tuple[Any, ...]]]:
        method_id = str(session.run.row["method_id"])
        if method_id != "fixed_base_distilled_realization":
            context = SimpleNamespace(row=session.run.row)
            return registered_variant_specs(context, session)
        if set(restored_states) != {"base", "candidate"}:
            raise FullBackendError(
                "Distilled realization checkpoint lacks base/candidate states."
            )
        base = copy.deepcopy(session.trainer.policy_model_old)
        candidate = copy.deepcopy(session.trainer.policy_model_candidate)
        base.load_state_dict(restored_states["base"], strict=True)
        candidate.load_state_dict(restored_states["candidate"], strict=True)
        return [
            ("base", base, None, False, ()),
            ("candidate", candidate, None, False, ()),
            ("realized_policy", session.trainer.policy_model_old, None, False, ()),
        ]

    def _evaluate_checkpoint(
        self,
        request: BackendRequest,
        runtime: SealedRuntimeIdentity,
        live_session: LearnedSession,
        checkpoint: FullCheckpointArtifact,
        *,
        kind: str,
    ) -> _EvaluationOutcome:
        def evaluate() -> _EvaluationOutcome:
            restored = self._build_session(request.run, runtime)
            loaded, observed = self._module._load_checkpoint_payload(
                str(checkpoint.path), expected_sha256=checkpoint.sha256
            )
            if type(restored.trainer).__name__ == "PPOTrainer":
                from policy_improvement_smoke_checkpoint import (
                    validate_ppo_smoke_checkpoint,
                )

                validate_ppo_smoke_checkpoint(
                    loaded,
                    restored.trainer,
                    expected_identity=loaded["identity"],
                    validate_only=False,
                )
                restored_states: Mapping[str, Mapping[str, object]] = {}
            else:
                self._module.resume_from_checkpoint(
                    str(checkpoint.path),
                    restored.model,
                    restored.trainer,
                    str(restored.device),
                    expected_dataset_provenance=restored.dataset_provenance,
                    expected_run_identity=restored.run_identity,
                    expected_checkpoint_sha256=checkpoint.sha256,
                )
                raw_states = loaded.get(
                    "policy_improvement_full_evaluation_state_dicts", {}
                )
                if not isinstance(raw_states, Mapping):
                    raise FullBackendError(
                        "Checkpoint evaluation-state inventory is invalid."
                    )
                restored_states = raw_states
            if observed != checkpoint.sha256:
                raise FullBackendError("Evaluation checkpoint digest changed.")
            context = SimpleNamespace(
                protocol={
                    "protocol_id": request.run.protocol["protocol_id"],
                    "budgets": {
                        "smoke": {"evaluation_records": request.run.evaluation_records}
                    },
                },
                row=request.run.row,
            )
            adapter = SmokeSession(
                model=restored.model,
                trainer=restored.trainer,
                rl_config=restored.rl_config,
                env_config=restored.env_config,
                train_dataset=restored.train_dataset,
                evaluation_dataset=restored.evaluation_dataset,
                checker=restored.checker,
                task_config=restored.task_config,
                dataset_provenance=restored.dataset_provenance,
                effective_config=restored.effective_config,
                effective_config_sha256=restored.effective_config_sha256,
                initialization_sha256=restored.initialization_sha256,
                device=restored.device,
                config_path=restored.config_path,
                method_config_sha256=restored.method_config_sha256,
                run_identity=restored.run_identity,
                evidence_identity=restored.evidence_identity,
            )
            evaluations: list[dict[str, object]] = []
            evaluation_dir = request.staging_generation / "evaluations" / kind
            for variant, model, callback, greedy, additional in self._variant_specs(
                restored, restored_states
            ):
                aggregate, instances, _, _ = evaluate_registered_policy(
                    context,
                    adapter,
                    variant,
                    model,
                    callback,
                    greedy,
                    additional,
                )
                instances["snapshot_kind"] = kind
                instances["evaluation_id"] = (
                    f"{request.run.row['run_id']}.{kind}.{variant}"
                )
                records = instances.get("records")
                if not isinstance(records, list):
                    raise FullBackendError(
                        "Per-instance evaluation record inventory is invalid."
                    )
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        raise FullBackendError(
                            "Per-instance evaluation record is invalid."
                        )
                    record["puzzle_id"] = (
                        f"{request.run.row['evaluation_split']}-{index:06d}"
                    )
                primary = {
                    "solve_rate": _available(aggregate["solve_rate"]),
                    "solved_count": _available(aggregate["solved_count"]),
                    "denominator": _available(aggregate["denominator"]),
                }
                edits = aggregate["edits_to_solve_mean"]
                secondary = {
                    "discounted_return_mean": _available(
                        aggregate["discounted_return_mean"]
                    ),
                    "edits_to_solve_mean": (
                        _available(edits)
                        if edits is not None
                        else _unavailable("not_applicable")
                    ),
                    "terminal_reason_counts": _available(
                        aggregate["terminal_reason_counts"]
                    ),
                    "value_calibration": _available(aggregate["value_calibration"]),
                }
                aggregate_evidence = {"primary": primary, "secondary": secondary}
                _write_json(
                    evaluation_dir / f"{variant}.json",
                    {**aggregate_evidence, "diagnostic_details": aggregate},
                )
                _write_json(evaluation_dir / f"{variant}.per_instance.json", instances)
                evaluations.append(
                    {
                        "evaluation_id": (
                            f"{request.run.row['run_id']}.{kind}.{variant}"
                        ),
                        "policy_variant": variant,
                        "evaluation_pool_sha256": _available(
                            restored.evaluation_pool_sha256
                        ),
                        "evaluation_artifact_sha256": _available(
                            canonical_json_sha256(aggregate_evidence)
                        ),
                        "per_instance_artifact_sha256": _available(
                            canonical_json_sha256(instances)
                        ),
                        "primary": primary,
                        "secondary": secondary,
                    }
                )
            expected = policy_variants_for_method(str(request.run.row["method_id"]))
            if tuple(item["policy_variant"] for item in evaluations) != expected:
                raise FullBackendError("Evaluation policy inventory differs.")
            compute = restored.trainer.compute_accounting_snapshot()
            evaluation_work = compute["model_work"]["evaluation"]
            evaluation_wall_time = compute["wall_time_seconds"]["evaluation"]
            if not isinstance(evaluation_work, dict) or isinstance(
                evaluation_wall_time, bool
            ):
                raise FullBackendError("Evaluation compute accounting is invalid.")
            return _EvaluationOutcome(
                policy_evaluations=evaluations,
                model_work={
                    name: int(value) for name, value in evaluation_work.items()
                },
                wall_time_seconds=float(evaluation_wall_time),
            )

        rng_state = self._module._capture_rng_state()
        try:
            outcome = evaluate_without_mutation(
                checkpoint_path=checkpoint.path,
                live_state_fingerprint=lambda: self._live_fingerprint(live_session),
                evaluator=evaluate,
            )
            if not isinstance(outcome, _EvaluationOutcome):
                raise FullBackendError(
                    "Checkpoint evaluator returned invalid accounting."
                )
            return outcome
        finally:
            self._module._restore_rng_state(rng_state)

    def _capture_snapshot(
        self,
        request: BackendRequest,
        runtime: SealedRuntimeIdentity,
        session: LearnedSession,
        *,
        kind: str,
        parent_checkpoint_sha256: str | None,
        parent_environment_interactions: int | None,
    ) -> _Snapshot:
        checkpoint = self._capture_checkpoint(
            request,
            runtime,
            session,
            kind=kind,
            directory=request.staging_generation / "checkpoints" / kind,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            parent_environment_interactions=parent_environment_interactions,
        )
        validation_path = (
            request.staging_generation
            / "validations"
            / kind
            / "checkpoint_validation.json"
        )
        _write_json(validation_path, checkpoint.validation)
        evaluation = self._evaluate_checkpoint(
            request,
            runtime,
            session,
            checkpoint,
            kind=kind,
        )
        session.external_evaluation_model_work = add_model_counters(
            session.external_evaluation_model_work,
            evaluation.model_work,
        )
        session.external_evaluation_wall_time_seconds += evaluation.wall_time_seconds
        compute = self._training_compute(session)
        interactions = session.trainer.get_env_step_count()
        recurrent = int(
            compute["model_work"]["training"]["recurrent_latent_state_updates"]
        )
        lineage = canonical_json_sha256(
            {
                "schema_name": "policy_improvement_full_checkpoint_lineage_v1",
                "run_id": request.run.row["run_id"],
                "snapshot_kind": kind,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "checkpoint_sha256": checkpoint.sha256,
                "environment_interactions": interactions,
                "recurrent_map_applications": recurrent,
            }
        )
        return _Snapshot(
            kind=kind,
            checkpoint=checkpoint,
            environment_interactions=interactions,
            recurrent_map_applications=recurrent,
            accelerator_seconds=float(compute["wall_time_seconds"]["training"]),
            policy_evaluations=evaluation.policy_evaluations,
            lineage_sha256=lineage,
        )

    @staticmethod
    def _train_step(session: LearnedSession, cap: int) -> dict[str, float]:
        if type(session.trainer).__name__ == "PPOTrainer" and cap < int(
            session.trainer.config.num_steps
        ):
            raise FullBackendError("PPO checkpoint boundary splits a rollout.")
        raw = session.trainer.train_step(max_env_steps_to_collect=cap)
        result: dict[str, float] = {}
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            measured = float(value)
            if not math.isfinite(measured):
                raise FullBackendError(f"Trainer emitted nonfinite metric {name!r}.")
            result[name] = measured
        return result

    def _complete_result(
        self,
        request: BackendRequest,
        runtime: SealedRuntimeIdentity,
        session: LearnedSession,
        snapshots: Mapping[str, _Snapshot],
        latest_metrics: Mapping[str, float],
        utilization: _TrainingUtilization,
        schedule: list[dict[str, object]],
    ) -> BackendPackage:
        interaction = snapshots["interaction_matched"]
        compute = snapshots["compute_matched"]
        validation_sha256s = {
            snapshot.kind: file_sha256(
                request.staging_generation
                / "validations"
                / snapshot.kind
                / "checkpoint_validation.json"
            )
            for snapshot in (interaction, compute)
        }
        state_inventory = {
            "schema_name": "policy_improvement_model_state_inventory_v1",
            "run_id": request.run.row["run_id"],
            "method_id": request.run.row["method_id"],
            "model_state_sha256": interaction.checkpoint.model_state_sha256,
            "role_state_sha256s": interaction.checkpoint.role_state_sha256s,
            "snapshot_state_bindings": [
                {
                    "snapshot_kind": snapshot.kind,
                    "checkpoint_sha256": snapshot.checkpoint.sha256,
                    "model_state_sha256": snapshot.checkpoint.model_state_sha256,
                    "checkpoint_validation_sha256": validation_sha256s[snapshot.kind],
                }
                for snapshot in (interaction, compute)
            ],
        }
        model_inventory_sha256 = _write_json(
            request.staging_generation / "model_state_inventory.json",
            state_inventory,
        )
        training_compute = self._training_compute(session)
        compute_snapshot_sha256: str | None = None
        compute_accounting_sha256: str | None = None
        if request.run.protocol.get("schema_name") == "policy_improvement_protocol_v2":
            compute_snapshot_sha256 = _write_json(
                request.staging_generation / "compute_snapshot.json",
                training_compute,
            )
            compute_accounting = build_training_compute_accounting(
                training_compute,
                device=session.device,
                checkpoint_seconds=session.checkpoint_wall_time_seconds,
                cuda_utilization={
                    "device_type": utilization.device_type,
                    "sampling_interval_seconds": utilization.sampling_interval_seconds,
                    "samples": list(utilization.samples),
                },
            )
            compute_accounting_sha256 = _write_json(
                request.staging_generation / "compute_accounting.json",
                compute_accounting,
            )
        run_manifest = {
            "schema_name": "policy_improvement_full_run_manifest_v1",
            "schema_version": 1,
            "run_id": request.run.row["run_id"],
            "method_id": request.run.row["method_id"],
            "environment_interactions": request.run.final_environment_interactions,
            "protocol_sha256": request.run.protocol_sha256,
            "registry_row_sha256": request.run.registry_row_sha256,
            "amendment_history_sha256": request.run.amendment_history_sha256,
            "runtime_authorization_sha256": (request.run.runtime_authorization_sha256),
            "runtime_sha256": runtime.runtime_sha256,
            "source_git_commit": runtime.source_git_commit,
            "source_manifest_sha256": runtime.source_manifest_sha256,
            "dataset_manifest_sha256": session.dataset_manifest_sha256,
            "checkpoint_sha256": interaction.checkpoint.sha256,
            "model_state_sha256": interaction.checkpoint.model_state_sha256,
            "model_state_sha256s": interaction.checkpoint.role_state_sha256s,
            "model_state_inventory_sha256": model_inventory_sha256,
            "checkpoint_schedule": schedule,
            "snapshots": [
                {
                    "snapshot_kind": snapshot.kind,
                    "checkpoint_path": snapshot.checkpoint.path.relative_to(
                        request.staging_generation
                    ).as_posix(),
                    "checkpoint_sha256": snapshot.checkpoint.sha256,
                    "environment_interactions": snapshot.environment_interactions,
                    "recurrent_map_applications": (snapshot.recurrent_map_applications),
                }
                for snapshot in (interaction, compute)
            ],
        }
        if (
            compute_snapshot_sha256 is not None
            and compute_accounting_sha256 is not None
        ):
            run_manifest["compute_snapshot_sha256"] = compute_snapshot_sha256
            run_manifest["compute_accounting_sha256"] = compute_accounting_sha256
        run_manifest_sha256 = _write_json(
            request.staging_generation / "RUN_MANIFEST.json", run_manifest
        )
        training_work = training_compute["model_work"]["training"]
        memory = training_compute["peak_memory_bytes"]
        device = self._module._canonical_device(session.device)
        if session.device.type == "cuda":
            utilization_fraction = _available(
                sum(utilization.samples) / len(utilization.samples)
            )
            utilization_count = _available(len(utilization.samples))
            utilization_interval = _available(utilization.sampling_interval_seconds)
        else:
            utilization_fraction = _unavailable("not_applicable")
            utilization_count = _unavailable("not_applicable")
            utilization_interval = _unavailable("not_applicable")
        diagnostics = {
            name: _unavailable("not_collected_by_registered_protocol")
            for name in (
                "L_preproj",
                "projection_active_rate",
                "absolute_depth_policy_agreement",
                "finite_depth_discrepancy",
                "fixed_target_residual",
                "propagated_target_lag",
                "exact_centering_defect",
                "candidate_current_tv",
                "mixture_realized_tv",
                "mixture_realized_kl",
                "fresh_initialization_discrepancy",
                "value_of_memory",
            )
        }

        def snapshot_document(
            snapshot: _Snapshot, target: int, unit: str
        ) -> dict[str, object]:
            return {
                "snapshot_kind": snapshot.kind,
                "status": "available",
                "unavailable_reason": None,
                "target": {
                    "unit": unit,
                    "registered_quantity": _available(target),
                },
                "observed_environment_interactions": _available(
                    snapshot.environment_interactions
                ),
                "observed_recurrent_map_applications": _available(
                    snapshot.recurrent_map_applications
                ),
                "accelerator_seconds_observed": _available(
                    snapshot.accelerator_seconds
                ),
                "checkpoint_sha256": _available(snapshot.checkpoint.sha256),
                "model_state_sha256": _available(
                    snapshot.checkpoint.model_state_sha256
                ),
                "checkpoint_lineage_sha256": _available(snapshot.lineage_sha256),
                "policy_evaluations": snapshot.policy_evaluations,
            }

        result = {
            "schema_name": SCHEMA_NAME,
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol_id": request.run.protocol["protocol_id"],
            "protocol_sha256": request.run.protocol_sha256,
            "amendment_history_sha256": request.run.amendment_history_sha256,
            **{
                field: request.run.row[field]
                for field in (
                    "run_id",
                    "phase",
                    "tier",
                    "seed",
                    "evaluation_split",
                    "method_id",
                    "base_method_id",
                    "n",
                    "K",
                    "alpha",
                    "ablation_variant",
                )
            },
            "registry_row_sha256": request.run.registry_row_sha256,
            "applied_config_override": request.run.row["config_override"],
            "primary_policy_variant": primary_policy_variant_for_method(
                str(request.run.row["method_id"])
            ),
            "evaluation_snapshots": [
                snapshot_document(
                    interaction,
                    request.run.final_environment_interactions,
                    "environment_interactions",
                ),
                snapshot_document(
                    compute,
                    request.compute_target_recurrent_map_applications,
                    "recurrent_map_applications",
                ),
            ],
            "status": "complete",
            "failure": None,
            "identities": {
                "producer_git_commit": runtime.source_git_commit,
                "git_clean": True,
                "runtime_authorization_sha256": (
                    request.run.runtime_authorization_sha256
                ),
                "training_source_git_commit": runtime.source_git_commit,
                "training_runtime_sha256": runtime.runtime_sha256,
                "training_runtime_profile_sha256": (runtime.runtime_profile_sha256),
                "training_selected_source_manifest_sha256": (
                    runtime.selected_source_manifest_sha256
                ),
                "launcher_sha256": runtime.launcher_sha256,
                "producer_manifest_sha256": (runtime.producer_source_manifest_sha256),
                "method_config_sha256": session.method_config_sha256,
                "effective_config_sha256": request.run.row[
                    "expected_effective_config_sha256"
                ],
                "dataset_manifest_sha256": session.dataset_manifest_sha256,
                "train_ordered_records_sha256": (session.train_ordered_records_sha256),
                "evaluation_ordered_records_sha256": (
                    session.evaluation_ordered_records_sha256
                ),
                "initialization_sha256": session.initialization_sha256,
                "checkpoint_sha256": _available(interaction.checkpoint.sha256),
                "model_state_sha256": _available(
                    interaction.checkpoint.model_state_sha256
                ),
                "evaluation_runtime_sha256": _available(runtime.runtime_sha256),
                "evaluation_source_git_commit": _available(runtime.source_git_commit),
                "evaluation_runtime_profile_sha256": _available(
                    runtime.runtime_profile_sha256
                ),
                "evaluation_selected_source_manifest_sha256": _available(
                    runtime.selected_source_manifest_sha256
                ),
                "evaluation_pool_sha256": _available(session.evaluation_pool_sha256),
                "test_open_sha256": (
                    _available(request.run.test_open_sha256)
                    if request.run.test_open_sha256 is not None
                    else _unavailable("test_data_not_opened")
                ),
                "device": device,
            },
            "metrics": {
                "training": {
                    "interactions_to_first_solve": _unavailable(
                        "not_collected_by_registered_protocol"
                    ),
                    "value_loss": (
                        _available(latest_metrics["loss_value"])
                        if "loss_value" in latest_metrics
                        else _unavailable("not_collected_by_registered_protocol")
                    ),
                    "policy_loss": (
                        _available(latest_metrics["loss_policy"])
                        if "loss_policy" in latest_metrics
                        else _unavailable("not_collected_by_registered_protocol")
                    ),
                    "wall_time_seconds": _available(
                        training_compute["wall_time_seconds"]["training"]
                    ),
                    "gpu_hours": _available(
                        float(training_compute["wall_time_seconds"]["training"])
                        / 3600.0
                        if session.device.type == "cuda"
                        else 0.0
                    ),
                    "gpu_utilization_fraction": utilization_fraction,
                    "gpu_utilization_sample_count": utilization_count,
                    "gpu_utilization_sampling_interval_seconds": (utilization_interval),
                    "peak_allocated_memory_bytes": _available(
                        int(memory["cuda_allocated"] or 0)
                    ),
                    "peak_reserved_memory_bytes": _available(
                        int(memory["cuda_reserved"] or 0)
                    ),
                    "optimizer_steps": _available(
                        training_compute["progress"]["optimizer_steps_total"]
                    ),
                    "cells_processed": _available(
                        int(training_work["recurrent_latent_state_updates"])
                        * int(session.model.config.seq_len)
                    ),
                    "actions_processed": _available(
                        int(training_work["action_logits_evaluated"])
                        + int(training_work["action_values_evaluated"])
                    ),
                    "tokens_processed": _available(
                        int(training_work["recurrent_latent_state_updates"])
                        * int(session.model.config.seq_len)
                    ),
                    "policy_head_calls": _available(
                        training_work["action_logits_evaluated"]
                    ),
                    "recurrent_map_applications": _available(
                        training_work["recurrent_latent_state_updates"]
                    ),
                    "value_head_calls": _available(training_work["value_api_calls"]),
                },
                "diagnostics": diagnostics,
            },
            "artifacts": {
                "checkpoint": _available(interaction.checkpoint.sha256),
                "checkpoint_validation": _available(
                    validation_sha256s["interaction_matched"]
                ),
                "model_state_inventory": _available(model_inventory_sha256),
                "run_manifest": _available(run_manifest_sha256),
            },
        }
        _write_json(request.staging_generation / "result.json", result)
        return BackendPackage(
            result=result,
            primary_checkpoint_relative_path=interaction.checkpoint.path.relative_to(
                request.staging_generation
            ).as_posix(),
        )

    def execute(
        self,
        request: BackendRequest,
        runtime: SealedRuntimeIdentity,
    ) -> BackendPackage:
        phase = "training"
        session: LearnedSession | None = None
        try:
            session = self._build_session(request.run, runtime)
            sampler = _GpuTrainingSampler(session.device)
            sampler.start()
            snapshots: dict[str, _Snapshot] = {}
            schedule: list[dict[str, object]] = []
            latest_metrics: dict[str, float] = {}
            parent_sha256 = None
            parent_interactions = None
            schedule_targets = iter(request.interaction_checkpoints)
            next_schedule = next(schedule_targets, None)
            compute_target = request.compute_target_recurrent_map_applications
            try:
                while session.trainer.get_env_step_count() < (
                    request.run.final_environment_interactions
                ):
                    current = session.trainer.get_env_step_count()
                    boundary = min(
                        item
                        for item in (
                            next_schedule,
                            request.run.final_environment_interactions,
                        )
                        if item is not None and item > current
                    )
                    latest_metrics = self._train_step(session, boundary - current)
                    after = session.trainer.get_env_step_count()
                    if after <= current or after > boundary:
                        raise FullBackendError(
                            "Trainer violated an exact interaction boundary."
                        )
                    recurrent = self._recurrent_work(session)
                    if "compute_matched" not in snapshots:
                        relative = abs(recurrent - compute_target) / compute_target
                        if relative <= request.compute_maximum_relative_mismatch:
                            phase = "checkpoint"
                            sampler.pause()
                            try:
                                snapshot = self._capture_snapshot(
                                    request,
                                    runtime,
                                    session,
                                    kind="compute_matched",
                                    parent_checkpoint_sha256=parent_sha256,
                                    parent_environment_interactions=parent_interactions,
                                )
                            finally:
                                sampler.resume()
                            snapshots[snapshot.kind] = snapshot
                            parent_sha256 = snapshot.checkpoint.sha256
                            parent_interactions = snapshot.environment_interactions
                            phase = "training"
                        elif recurrent > compute_target:
                            raise FullBackendError(
                                "Compute target was crossed outside its five-percent tolerance."
                            )
                    if after == next_schedule:
                        if after == request.run.final_environment_interactions:
                            phase = "checkpoint"
                            sampler.pause()
                            try:
                                snapshot = self._capture_snapshot(
                                    request,
                                    runtime,
                                    session,
                                    kind="interaction_matched",
                                    parent_checkpoint_sha256=parent_sha256,
                                    parent_environment_interactions=parent_interactions,
                                )
                            finally:
                                sampler.resume()
                            snapshots[snapshot.kind] = snapshot
                            parent_sha256 = snapshot.checkpoint.sha256
                            parent_interactions = snapshot.environment_interactions
                            phase = "training"
                        else:
                            phase = "checkpoint"
                            sampler.pause()
                            try:
                                artifact = self._capture_checkpoint(
                                    request,
                                    runtime,
                                    session,
                                    kind="scheduled",
                                    directory=(
                                        request.staging_generation
                                        / "checkpoints"
                                        / "scheduled"
                                        / f"env_{after:09d}"
                                    ),
                                    parent_checkpoint_sha256=parent_sha256,
                                    parent_environment_interactions=parent_interactions,
                                )
                            finally:
                                sampler.resume()
                            validation_path = (
                                request.staging_generation
                                / "validations"
                                / "scheduled"
                                / f"env_{after:09d}"
                                / "resume_validation.json"
                            )
                            validation_sha256 = _write_json(
                                validation_path, artifact.validation
                            )
                            schedule.append(
                                {
                                    "environment_interactions": after,
                                    "checkpoint_sha256": artifact.sha256,
                                    "resume_validation_sha256": validation_sha256,
                                }
                            )
                            parent_sha256 = artifact.sha256
                            parent_interactions = after
                            phase = "training"
                        next_schedule = next(schedule_targets, None)
                if "interaction_matched" not in snapshots:
                    raise FullBackendError("Interaction snapshot was not captured.")
                if "compute_matched" not in snapshots:
                    recurrent = self._recurrent_work(session)
                    relative = abs(recurrent - compute_target) / compute_target
                    if relative > request.compute_maximum_relative_mismatch:
                        raise FullBackendError(
                            "Compute target was not reached before the interaction budget."
                        )
                    phase = "checkpoint"
                    sampler.pause()
                    try:
                        snapshots["compute_matched"] = self._capture_snapshot(
                            request,
                            runtime,
                            session,
                            kind="compute_matched",
                            parent_checkpoint_sha256=parent_sha256,
                            parent_environment_interactions=parent_interactions,
                        )
                    finally:
                        sampler.resume()
                utilization = sampler.stop()
            except BaseException:
                try:
                    sampler.stop()
                except BaseException:
                    pass
                raise
            phase = "publication"
            return self._complete_result(
                request,
                runtime,
                session,
                snapshots,
                latest_metrics,
                utilization,
                schedule,
            )
        except FullRunFailure:
            raise
        except BaseException as exc:
            raise FullRunFailure(
                phase,
                build_failed_result(
                    request.run,
                    runtime,
                    phase=phase,
                    error=exc,
                    session=session,
                ),
            ) from exc


@dataclass(frozen=True)
class ReadOnlySnapshot:
    checkpoint_sha256: str
    snapshot_kind: str
    environment_interactions: int
    model_state_sha256: str
    training_state_sha256: str
    recurrent_transition_sha256: str


@dataclass(frozen=True)
class TheoryBridgeState:
    state_id: str
    record_index: int
    dataset_record_sha256: str
    registered_state_sha256: str
    action_mask: tuple[bool, ...]
    current_probabilities: tuple[float, ...]
    candidate_probabilities: tuple[float, ...]
    deployed_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TheoryBridgeOutcome:
    probability: float
    reward: float
    terminal: bool
    next_state_id: str | None


@dataclass(frozen=True)
class TheoryBridgeRollout:
    rewards: tuple[float, ...]
    terminal: bool
    bootstrap_state_id: str | None
    trajectory_sha256: str


@dataclass
class _BridgeStateRecord:
    state_id: str
    record_index: int
    dataset_record_sha256: str
    environment_state: dict[str, Any]
    x: Mapping[str, object]
    plan: torch.Tensor
    latent: object | None
    terminal: bool
    action_mask: tuple[bool, ...] | None = None
    current_probabilities: tuple[float, ...] | None = None
    candidate_probabilities: tuple[float, ...] | None = None
    deployed_probabilities: tuple[float, ...] | None = None
    deployed_next_latent: object | None = None


def _tensor_identity(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def _tree_identity(value: object) -> object:
    if torch.is_tensor(value):
        return _tensor_identity(value)
    if (
        type(value).__module__.startswith("numpy")
        and hasattr(value, "dtype")
        and hasattr(value, "shape")
        and callable(getattr(value, "tobytes", None))
    ):
        payload = value.tobytes(order="C")
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(name): _tree_identity(item)
            for name, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_tree_identity(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _tree_identity(getattr(value, item.name))
                for item in dataclass_fields(value)
            },
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _clone_tree(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {name: _clone_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_replace(
            value,
            **{
                item.name: _clone_tree(getattr(value, item.name))
                for item in dataclass_fields(value)
            },
        )
    return copy.deepcopy(value)


class ReadOnlyTheoryBridgeSession:
    """Disposable checkpoint adapter for the separately authenticated evaluator."""

    def __init__(
        self,
        *,
        request_identity: Mapping[str, object],
        checkpoint_descriptor: int,
        checkpoint_sha256: str,
        snapshot_kind: str,
        environment_interactions: int,
        session: LearnedSession,
        training_module: Any,
        evaluation_state_dicts: Mapping[str, Mapping[str, object]],
        registered_state_indices: tuple[int, ...],
        reported_record_indices: tuple[int, ...] | None = None,
    ) -> None:
        self._identity = copy.deepcopy(dict(request_identity))
        self._checkpoint_descriptor = checkpoint_descriptor
        self._checkpoint_sha256 = checkpoint_sha256
        self._snapshot_kind = snapshot_kind
        self._environment_interactions = environment_interactions
        self._session = session
        self._training_module = training_module
        self._evaluation_state_dicts = evaluation_state_dicts
        self._registered_state_indices = registered_state_indices
        self._reported_record_indices = (
            registered_state_indices
            if reported_record_indices is None
            else reported_record_indices
        )
        if len(self._reported_record_indices) != len(registered_state_indices):
            raise FullBackendError(
                "Theory dataset and reported record-index inventories differ."
            )
        self._states: dict[str, _BridgeStateRecord] = {}
        self._registered: tuple[TheoryBridgeState, ...] | None = None
        self._auxiliary_models: dict[str, Any] = {}
        self._read_only_transaction: tuple[str, str] | None = None
        if evaluation_state_dicts:
            for role, state in evaluation_state_dicts.items():
                model = copy.deepcopy(session.trainer.policy_model_old)
                model.load_state_dict(state, strict=True)
                model.eval()
                self._auxiliary_models[role] = model
        model_sha256, _ = session_model_state_identity(
            session,
            evaluation_state_dicts=evaluation_state_dicts,
        )
        self._model_state_sha256 = model_sha256
        recurrent_state = {
            name: value
            for name, value in self._deployed_model().state_dict().items()
            if not name.startswith("edit_policy.")
            and not name.startswith("value_head.")
        }
        self._recurrent_transition_sha256 = state_dict_sha256(recurrent_state)
        self._training_state_sha256 = self._mutable_training_identity()

    def close(self) -> None:
        descriptor = getattr(self, "_checkpoint_descriptor", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._checkpoint_descriptor = -1

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    def _deployed_model(self) -> Any:
        method = str(self._session.run.row["method_id"])
        if method == "matched_ppo":
            return self._session.model
        return self._session.trainer.policy_model_old

    def _fingerprint(self) -> str:
        return self._mutable_training_identity()

    def _mutable_training_identity(self) -> str:
        model_sha256, roles = session_model_state_identity(
            self._session,
            evaluation_state_dicts=self._evaluation_state_dicts,
        )
        trainer = self._session.trainer
        role_modules = {
            **trainer._model_roles(),
            **{
                f"theory_{name}": model
                for name, model in self._auxiliary_models.items()
            },
        }
        optimizer_states: dict[str, object] = {}
        for name, value in sorted(vars(trainer).items()):
            if not (
                name.endswith("_opt")
                or any(token in name for token in ("optimizer", "scheduler"))
            ):
                continue
            state_dict = getattr(value, "state_dict", None)
            if callable(state_dict):
                optimizer_states[name] = _tree_identity(state_dict())
        trainer_scalars = {
            name: value
            for name, value in sorted(vars(trainer).items())
            if value is None or isinstance(value, (bool, int, float, str))
        }
        compute_accounting_state = {
            name: _tree_identity(getattr(trainer, name))
            for name in (
                "_training_model_work",
                "_evaluation_model_work",
                "_training_wall_time_seconds",
                "_evaluation_wall_time_seconds",
                "_peak_process_rss_bytes",
                "_peak_cuda_allocated_bytes",
                "_peak_cuda_reserved_bytes",
            )
            if hasattr(trainer, name)
        }
        parameter_gradients = {
            role: {
                name: _tree_identity(parameter.grad)
                for name, parameter in module.named_parameters()
            }
            for role, module in sorted(role_modules.items())
        }
        environment = getattr(trainer, "env", None)
        environment_state = (
            environment.checkpoint_state()
            if environment is not None
            and callable(getattr(environment, "checkpoint_state", None))
            else None
        )
        replay = getattr(trainer, "replay", None)
        replay_state = (
            _tree_identity(list(replay.storage))
            if replay is not None and hasattr(replay, "storage")
            else None
        )
        collection_state = (
            trainer.collection_checkpoint_state()
            if callable(getattr(trainer, "collection_checkpoint_state", None))
            else None
        )
        centering_state = (
            trainer.exact_centering_checkpoint_state()
            if callable(getattr(trainer, "exact_centering_checkpoint_state", None))
            else None
        )
        return canonical_json_sha256(
            _tree_identity(
                {
                    "checkpoint_sha256": self._checkpoint_sha256,
                    "model_state_sha256": model_sha256,
                    "roles": roles,
                    "model_modes": {
                        name: bool(module.training)
                        for name, module in sorted(role_modules.items())
                    },
                    "trainer_scalars": trainer_scalars,
                    "optimizer_states": optimizer_states,
                    "compute_accounting_state": compute_accounting_state,
                    "parameter_gradients": parameter_gradients,
                    "environment_state": environment_state,
                    "collection_state": collection_state,
                    "exact_centering_state": centering_state,
                    "replay_state": replay_state,
                    "train_records": dataset_sample_sha256s(
                        self._session.train_dataset
                    ),
                    "evaluation_records": dataset_sample_sha256s(
                        self._session.evaluation_dataset
                    ),
                    "dataset_provenance": self._session.dataset_provenance,
                    "effective_config": self._session.effective_config,
                    "evidence_identity": self._session.evidence_identity,
                }
            )
        )

    def _guard(self, callback: Callable[[], Any]) -> Any:
        if self._read_only_transaction is not None:
            role_models = {
                **self._session.trainer._model_roles(),
                **{
                    f"theory_{name}": model
                    for name, model in self._auxiliary_models.items()
                },
            }
            compute_state = capture_model_compute_state(role_models)
            rng_state = self._training_module._capture_rng_state()
            try:
                return callback()
            finally:
                restore_model_compute_state(role_models, compute_state)
                self._training_module._restore_rng_state(rng_state)
        checkpoint_before, _ = _sealed_checkpoint_identity(self._checkpoint_descriptor)
        session_before = self._fingerprint()
        role_models = {
            **self._session.trainer._model_roles(),
            **{
                f"theory_{name}": model
                for name, model in self._auxiliary_models.items()
            },
        }
        compute_state = capture_model_compute_state(role_models)
        rng_state = self._training_module._capture_rng_state()
        result: Any = None
        callback_error: BaseException | None = None
        try:
            result = callback()
        except BaseException as exc:
            callback_error = exc
        finally:
            restore_model_compute_state(role_models, compute_state)
            self._training_module._restore_rng_state(rng_state)
        mutation_error: BaseException | None = None
        try:
            checkpoint_after, _ = _sealed_checkpoint_identity(
                self._checkpoint_descriptor
            )
            if checkpoint_after != checkpoint_before:
                raise FullBackendError("Theory bridge mutated its checkpoint.")
            if self._fingerprint() != session_before:
                raise FullBackendError("Theory bridge mutated restored training state.")
        except BaseException as exc:
            mutation_error = exc
        if mutation_error is not None:
            if callback_error is not None:
                raise mutation_error from callback_error
            raise mutation_error
        if callback_error is not None:
            raise callback_error
        return result

    def begin_read_only_evaluation(self) -> None:
        """Amortize mutation checks across one evaluator transaction."""

        if self._read_only_transaction is not None:
            raise FullBackendError("Theory read-only transaction is already active.")
        checkpoint_sha256, _ = _sealed_checkpoint_identity(self._checkpoint_descriptor)
        self._read_only_transaction = (checkpoint_sha256, self._fingerprint())

    def end_read_only_evaluation(self) -> None:
        """Close one transaction and reject any persistent state mutation."""

        transaction = self._read_only_transaction
        if transaction is None:
            raise FullBackendError("Theory read-only transaction is not active.")
        self._read_only_transaction = None
        checkpoint_before, session_before = transaction
        checkpoint_after, _ = _sealed_checkpoint_identity(self._checkpoint_descriptor)
        if checkpoint_after != checkpoint_before:
            raise FullBackendError("Theory bridge mutated its checkpoint.")
        if self._fingerprint() != session_before:
            raise FullBackendError("Theory bridge mutated restored training state.")

    def identity_bundle(self) -> dict[str, object]:
        return self._guard(lambda: copy.deepcopy(self._identity))

    def read_only_snapshot(self) -> ReadOnlySnapshot:
        return self._guard(
            lambda: ReadOnlySnapshot(
                checkpoint_sha256=self._checkpoint_sha256,
                snapshot_kind=self._snapshot_kind,
                environment_interactions=self._environment_interactions,
                model_state_sha256=self._model_state_sha256,
                training_state_sha256=self._training_state_sha256,
                recurrent_transition_sha256=self._recurrent_transition_sha256,
            )
        )

    def _mutable_state_sha256s(self) -> dict[str, str]:
        """Hash every mutable trainer component required by the v2 bridge."""

        trainer = self._session.trainer
        optimizer_states: dict[str, object] = {}
        scheduler_states: dict[str, object] = {}
        for name, value in sorted(vars(trainer).items()):
            state_dict = getattr(value, "state_dict", None)
            if not callable(state_dict):
                continue
            if "scheduler" in name:
                scheduler_states[name] = state_dict()
            elif name.endswith("_opt") or "optimizer" in name:
                optimizer_states[name] = state_dict()
        replay = getattr(trainer, "replay", None)
        replay_state = (
            list(replay.storage)
            if replay is not None and hasattr(replay, "storage")
            else None
        )
        collector_state = (
            trainer.collection_checkpoint_state()
            if callable(getattr(trainer, "collection_checkpoint_state", None))
            else None
        )
        environment = getattr(trainer, "env", None)
        environment_state = (
            environment.checkpoint_state()
            if environment is not None
            and callable(getattr(environment, "checkpoint_state", None))
            else None
        )
        exact_centering_state = (
            trainer.exact_centering_checkpoint_state()
            if callable(getattr(trainer, "exact_centering_checkpoint_state", None))
            else None
        )
        persistent_latent_state = None
        if isinstance(collector_state, Mapping):
            active_episode = collector_state.get("active_episode")
            if isinstance(active_episode, Mapping):
                persistent_latent_state = active_episode.get("latent")
        values = {
            "optimizer_states": optimizer_states,
            "scheduler_states": scheduler_states,
            "rng_states": self._training_module._capture_rng_state(),
            "replay_state": replay_state,
            "collector_state": collector_state,
            "environment_state": environment_state,
            "persistent_latent_state": persistent_latent_state,
            "exact_centering_state": exact_centering_state,
        }
        return {
            name: canonical_json_sha256(_tree_identity(value))
            for name, value in values.items()
        }

    def read_only_snapshot_v2(self) -> SimpleNamespace:
        """Return the expanded v2 mutation inventory without exposing objects."""

        return self._guard(
            lambda: SimpleNamespace(
                checkpoint_sha256=self._checkpoint_sha256,
                model_state_sha256=self._model_state_sha256,
                mutable_state_sha256s=self._mutable_state_sha256s(),
                recurrent_transition_sha256=self._recurrent_transition_sha256,
                snapshot_kind=self._snapshot_kind,
                environment_interactions=self._environment_interactions,
            )
        )

    def observed_model_identity(self, training_module: Any) -> dict[str, str]:
        """Reconstruct every training-side model identity in a theory request."""

        current, candidate, _, _ = self._policy_models()
        current_sha256 = state_dict_sha256(current.state_dict())
        candidate_sha256 = state_dict_sha256(candidate.state_dict())
        method = str(self._session.run.row["method_id"])
        if method in {
            "fixed_base_exact_persistent",
            "fixed_base_exact_episodic",
        }:
            deployed_sha256 = canonical_json_sha256(
                {
                    "kind": "exact_probability_mixture",
                    "current_policy_sha256": current_sha256,
                    "candidate_policy_sha256": candidate_sha256,
                    "alpha": float(self._session.run.row["alpha"]),
                    "recurrent_transition_sha256": (self._recurrent_transition_sha256),
                }
            )
        else:
            deployed_sha256 = state_dict_sha256(self._deployed_model().state_dict())
        return {
            "model_sha256": self._model_state_sha256,
            "model_config_sha256": canonical_json_sha256(
                training_module._config_dict(self._session.model.config)
            ),
            "current_policy_sha256": current_sha256,
            "candidate_policy_sha256": candidate_sha256,
            "deployed_policy_sha256": deployed_sha256,
            "recurrent_transition_sha256": self._recurrent_transition_sha256,
        }

    def _new_environment(self) -> PlanEditEnv:
        environment = PlanEditEnv(
            dataset=self._session.evaluation_dataset,
            checker=self._session.checker,
            config=self._session.env_config,
            task_config=self._session.task_config,
        )
        environment.set_stop_action_id(
            int(self._session.model.config.rl_num_actions) - 1
        )
        return environment

    def _state_id(
        self,
        *,
        record_index: int,
        x: Mapping[str, object],
        plan: torch.Tensor,
        latent: torch.Tensor | None,
        environment_state: Mapping[str, object],
    ) -> str:
        return canonical_json_sha256(
            {
                "record_index": record_index,
                "x": _tree_identity(x),
                "plan": _tree_identity(plan),
                "latent": _tree_identity(latent),
                "environment": _tree_identity(environment_state),
            }
        )

    def _initial_record(
        self,
        index: int,
        *,
        reported_record_index: int | None = None,
    ) -> _BridgeStateRecord:
        environment = self._new_environment()
        x, plan = environment.reset(idx=index)
        if not isinstance(x, Mapping) or not torch.is_tensor(plan):
            raise FullBackendError("Theory bridge state has unsupported tensors.")
        latent = None
        if not bool(self._session.rl_config.episodic_latent):
            batched_x = prepare_batch_x(x, device=self._session.device, batched=False)
            batched_plan = prepare_plan(
                plan, device=self._session.device, batched=False
            )
            latent = self._deployed_model().init_latent(batched_x, batched_plan)
        environment_state = environment.checkpoint_state()
        record_index = index if reported_record_index is None else reported_record_index
        state_id = self._state_id(
            record_index=record_index,
            x=x,
            plan=plan,
            latent=latent,
            environment_state=environment_state,
        )
        record = _BridgeStateRecord(
            state_id=state_id,
            record_index=record_index,
            dataset_record_sha256=dataset_sample_sha256s(
                self._session.evaluation_dataset
            )[index],
            environment_state=environment_state,
            x=_clone_tree(x),
            plan=plan.detach().clone(),
            latent=_clone_tree(latent) if latent is not None else None,
            terminal=False,
        )
        self._states[state_id] = record
        return record

    def _policy_models(self) -> tuple[Any, Any, Any, Callable[..., Any] | None]:
        method = str(self._session.run.row["method_id"])
        trainer = self._session.trainer
        if "exact_mixture" in policy_variants_for_method(method):
            return (
                trainer.policy_model_old,
                trainer.policy_model_candidate,
                trainer.policy_model_old,
                trainer._mixed_policy_dist,
            )
        if method == "legacy_parameter_interpolation":
            if (
                trainer.preinterpolation_policy_base is None
                or trainer.preinterpolation_policy_candidate is None
            ):
                raise FullBackendError("Legacy checkpoint lacks policy snapshots.")
            return (
                trainer.preinterpolation_policy_base,
                trainer.preinterpolation_policy_candidate,
                trainer.policy_model_old,
                None,
            )
        if method == "fixed_base_distilled_realization":
            if set(self._auxiliary_models) != {"base", "candidate"}:
                raise FullBackendError("Distilled checkpoint lacks policy snapshots.")
            return (
                self._auxiliary_models["base"],
                self._auxiliary_models["candidate"],
                trainer.policy_model_old,
                None,
            )
        return self._session.model, self._session.model, self._session.model, None

    def _populate_probabilities(self, record: _BridgeStateRecord) -> None:
        if record.deployed_probabilities is not None or record.terminal:
            return
        current, candidate, deployed, deployed_callback = self._policy_models()
        modules = {id(module): module for module in (current, candidate, deployed)}
        modes = {
            identifier: bool(module.training) for identifier, module in modules.items()
        }
        for module in modules.values():
            module.eval()
        try:
            with torch.no_grad():
                batched_x = prepare_batch_x(
                    record.x, device=self._session.device, batched=False
                )
                plan = prepare_plan(
                    record.plan, device=self._session.device, batched=False
                )
                mask_tensor = self._new_environment()
                mask_tensor.load_checkpoint_state(record.environment_state)
                raw_mask = mask_tensor.get_action_mask()
                action_mask = (
                    raw_mask.to(self._session.device) if raw_mask is not None else None
                )
                n = int(self._session.rl_config.inner_unroll_n)
                z = record.latent
                current_dist, current_latent = current.policy_dist(
                    batched_x,
                    plan,
                    n=n,
                    action_mask=action_mask,
                    z=z,
                )
                if z is not None and "exact_mixture" in policy_variants_for_method(
                    str(self._session.run.row["method_id"])
                ):
                    candidate_dist, _ = candidate.policy_dist(
                        batched_x,
                        plan,
                        n=0,
                        action_mask=action_mask,
                        z=current_latent,
                    )
                else:
                    candidate_dist, _ = candidate.policy_dist(
                        batched_x,
                        plan,
                        n=n,
                        action_mask=action_mask,
                        z=z,
                    )
                if deployed_callback is not None:
                    deployed_dist, next_latent = deployed_callback(
                        batched_x,
                        plan,
                        n=n,
                        action_mask=action_mask,
                        z=z,
                    )
                else:
                    deployed_dist, next_latent = deployed.policy_dist(
                        batched_x,
                        plan,
                        n=n,
                        action_mask=action_mask,
                        z=z,
                    )
                record.action_mask = tuple(
                    bool(value)
                    for value in (
                        raw_mask.tolist()
                        if raw_mask is not None
                        else [True] * int(deployed_dist.probs.shape[-1])
                    )
                )
                record.current_probabilities = tuple(
                    float(value) for value in current_dist.probs.reshape(-1).tolist()
                )
                record.candidate_probabilities = tuple(
                    float(value) for value in candidate_dist.probs.reshape(-1).tolist()
                )
                record.deployed_probabilities = tuple(
                    float(value) for value in deployed_dist.probs.reshape(-1).tolist()
                )
                record.deployed_next_latent = (
                    _clone_tree(next_latent)
                    if next_latent is not None
                    and not bool(self._session.rl_config.episodic_latent)
                    else None
                )
        finally:
            for identifier, module in modules.items():
                module.train(modes[identifier])

    def _theory_state(self, record: _BridgeStateRecord) -> TheoryBridgeState:
        self._populate_probabilities(record)
        if (
            record.action_mask is None
            or record.current_probabilities is None
            or record.candidate_probabilities is None
            or record.deployed_probabilities is None
        ):
            raise FullBackendError("Theory bridge state has no policy distribution.")
        registered_state_sha256 = canonical_json_sha256(
            {
                "state_id": record.state_id,
                "record_index": record.record_index,
                "dataset_record_sha256": record.dataset_record_sha256,
                "action_mask": list(record.action_mask),
                "current_probabilities": list(record.current_probabilities),
                "candidate_probabilities": list(record.candidate_probabilities),
                "deployed_probabilities": list(record.deployed_probabilities),
            }
        )
        return TheoryBridgeState(
            state_id=record.state_id,
            record_index=record.record_index,
            dataset_record_sha256=record.dataset_record_sha256,
            registered_state_sha256=registered_state_sha256,
            action_mask=record.action_mask,
            current_probabilities=record.current_probabilities,
            candidate_probabilities=record.candidate_probabilities,
            deployed_probabilities=record.deployed_probabilities,
        )

    def registered_states(self) -> tuple[TheoryBridgeState, ...]:
        def build() -> tuple[TheoryBridgeState, ...]:
            if self._registered is None:
                self._registered = tuple(
                    self._theory_state(
                        self._initial_record(
                            dataset_index,
                            reported_record_index=record_index,
                        )
                    )
                    for dataset_index, record_index in zip(
                        self._registered_state_indices,
                        self._reported_record_indices,
                    )
                )
            return self._registered

        return self._guard(build)

    def training_advantage_estimator(self, state_id: str) -> SimpleNamespace:
        """Reconstruct the exact frozen-checkpoint tensor used by training."""

        def reconstruct() -> SimpleNamespace:
            record = self._record(state_id)
            self._populate_probabilities(record)
            if (
                record.action_mask is None
                or record.current_probabilities is None
                or record.terminal
            ):
                raise FullBackendError(
                    "Theory trainer-estimator state is incomplete or terminal."
                )
            trainer = self._session.trainer
            checker_fn = getattr(trainer, "_checker_fn", None)
            if checker_fn is None:
                raise FullBackendError(
                    "Theory trainer-estimator reconstruction lacks its checker."
                )
            x_batch = prepare_batch_x(
                record.x,
                device=self._session.device,
                batched=False,
            )
            plan = prepare_plan(
                record.plan,
                device=self._session.device,
                batched=False,
            )
            mask = torch.tensor(
                record.action_mask,
                dtype=torch.bool,
                device=self._session.device,
            ).unsqueeze(0)
            probabilities = torch.tensor(
                record.current_probabilities,
                dtype=torch.float32,
                device=self._session.device,
            ).unsqueeze(0)
            environment = self._new_environment()
            environment.load_checkpoint_state(record.environment_state)
            baseline, q_values = compute_exact_baseline_summation(
                model=trainer.model,
                x_batch=x_batch,
                y_batch=plan,
                env=environment,
                n=int(self._session.rl_config.inner_unroll_n),
                gamma=float(self._session.rl_config.gamma),
                checker_fn=checker_fn,
                action_mask=mask,
                policy_probs=probabilities,
                successor_latent=(
                    record.deployed_next_latent
                    if not bool(self._session.rl_config.episodic_latent)
                    else None
                ),
            )
            advantages = torch.where(
                mask,
                q_values - baseline.unsqueeze(-1),
                torch.zeros_like(q_values),
            )
            configured_clip = getattr(self._session.rl_config, "advantage_clip", None)
            if configured_clip is not None and float(configured_clip) > 0.0:
                clip_value: float | None = float(configured_clip)
                clipping_kind = "clip_then_exact_recenter"
                advantages = _clip_and_recenter_advantages(
                    advantages,
                    probabilities,
                    mask,
                    clip_value,
                )
            else:
                clip_value = None
                clipping_kind = "none"
            return SimpleNamespace(
                action_mask=record.action_mask,
                current_probabilities=record.current_probabilities,
                advantages=tuple(
                    float(value) for value in advantages.reshape(-1).tolist()
                ),
                clipping_kind=clipping_kind,
                clip_value=clip_value,
            )

        return self._guard(reconstruct)

    def persistent_endpoint_witness(
        self,
        state_id: str,
        endpoint_depth: int,
    ) -> SimpleNamespace:
        """Prove that endpoint depth cannot replace the deployed F_n carry."""

        def witness() -> SimpleNamespace:
            if (
                isinstance(endpoint_depth, bool)
                or not isinstance(endpoint_depth, int)
                or endpoint_depth <= 0
            ):
                raise FullBackendError("Persistent endpoint depth must be positive.")
            if bool(self._session.rl_config.episodic_latent):
                raise FullBackendError(
                    "Persistent endpoint witness is unavailable in episodic mode."
                )
            record = self._record(state_id)
            self._populate_probabilities(record)
            if (
                record.action_mask is None
                or record.current_probabilities is None
                or record.candidate_probabilities is None
                or record.deployed_probabilities is None
                or record.deployed_next_latent is None
            ):
                raise FullBackendError("Persistent endpoint witness is incomplete.")
            deployed_depth = int(self._session.rl_config.inner_unroll_n)
            return SimpleNamespace(
                requested_endpoint_depth=endpoint_depth,
                deployed_transition_depth=deployed_depth,
                carried_successor_latent_sha256=canonical_json_sha256(
                    _tree_identity(record.deployed_next_latent)
                ),
                trajectory_sha256=canonical_json_sha256(
                    {
                        "state_id": record.state_id,
                        "environment_state": _tree_identity(record.environment_state),
                        "deployed_transition_depth": deployed_depth,
                    }
                ),
                action_probabilities_sha256=canonical_json_sha256(
                    {
                        "action_mask": list(record.action_mask),
                        "current": list(record.current_probabilities),
                        "candidate": list(record.candidate_probabilities),
                        "deployed": list(record.deployed_probabilities),
                    }
                ),
                recurrent_transition_sha256=self._recurrent_transition_sha256,
            )

        return self._guard(witness)

    def _record(self, state_id: str) -> _BridgeStateRecord:
        if state_id not in self._states:
            self.registered_states()
        try:
            return self._states[state_id]
        except KeyError as exc:
            raise FullBackendError("Unknown theory-bridge state ID.") from exc

    def endpoint_value(self, state_id: str, depth: int) -> float:
        def evaluate() -> float:
            if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
                raise FullBackendError("Endpoint reference depth must be positive.")
            record = self._record(state_id)
            if record.terminal:
                return 0.0
            model = self._session.model
            mode = bool(model.training)
            model.eval()
            try:
                with torch.no_grad():
                    value, _ = model.used_value(
                        prepare_batch_x(
                            record.x, device=self._session.device, batched=False
                        ),
                        prepare_plan(
                            record.plan,
                            device=self._session.device,
                            batched=False,
                        ),
                        n=depth,
                        z=record.latent,
                    )
                    scalar = float(value.reshape(-1)[0].item())
            finally:
                model.train(mode)
            if not math.isfinite(scalar):
                raise FullBackendError("Endpoint evaluator returned a nonfinite value.")
            return scalar

        return self._guard(evaluate)

    def _outcome(
        self, state_id: str, action_index: int
    ) -> tuple[TheoryBridgeOutcome, dict[str, object]]:
        record = self._record(state_id)
        self._populate_probabilities(record)
        probabilities = record.deployed_probabilities
        action_mask = record.action_mask
        if probabilities is None or action_mask is None:
            raise FullBackendError("Theory state has no deployed policy.")
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index < 0
            or action_index >= len(probabilities)
            or not action_mask[action_index]
        ):
            raise FullBackendError("Theory outcome action is masked or invalid.")
        environment = self._new_environment()
        environment.load_checkpoint_state(record.environment_state)
        (x_next, plan_next), reward, terminal, _ = environment.step(action_index)
        if not isinstance(x_next, Mapping) or not torch.is_tensor(plan_next):
            raise FullBackendError("Theory outcome produced unsupported state tensors.")
        next_latent = (
            record.deployed_next_latent
            if not bool(self._session.rl_config.episodic_latent)
            else None
        )
        environment_state = environment.checkpoint_state()
        next_state_id = None
        if not bool(terminal):
            next_state_id = self._state_id(
                record_index=record.record_index,
                x=x_next,
                plan=plan_next,
                latent=next_latent,
                environment_state=environment_state,
            )
        if next_state_id is not None and next_state_id not in self._states:
            self._states[next_state_id] = _BridgeStateRecord(
                state_id=next_state_id,
                record_index=record.record_index,
                dataset_record_sha256=record.dataset_record_sha256,
                environment_state=environment_state,
                x=_clone_tree(x_next),
                plan=plan_next.detach().clone(),
                latent=(_clone_tree(next_latent) if next_latent is not None else None),
                terminal=bool(terminal),
            )
        outcome = TheoryBridgeOutcome(
            probability=1.0,
            reward=float(reward),
            terminal=bool(terminal),
            next_state_id=next_state_id,
        )
        trace = {
            "state_id": state_id,
            "action_index": action_index,
            "probability": outcome.probability,
            "reward": outcome.reward,
            "terminal": outcome.terminal,
            "next_state_id": next_state_id,
        }
        return outcome, trace

    def exact_action_outcomes(
        self, state_id: str, action_index: int
    ) -> tuple[TheoryBridgeOutcome, ...]:
        return self._guard(lambda: (self._outcome(state_id, action_index)[0],))

    def sample_rollout(
        self, state_id: str, horizon: int, seed: int
    ) -> TheoryBridgeRollout:
        def sample() -> TheoryBridgeRollout:
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon <= 0
            ):
                raise FullBackendError("Theory rollout horizon must be positive.")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise FullBackendError("Theory rollout seed must be nonnegative.")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            current: str | None = state_id
            rewards: list[float] = []
            trace: list[dict[str, object]] = []
            terminal = False
            for _ in range(horizon):
                if current is None:
                    raise FullBackendError(
                        "Nonterminal theory rollout lost its successor state."
                    )
                record = self._record(current)
                if record.terminal:
                    terminal = True
                    break
                self._populate_probabilities(record)
                assert record.current_probabilities is not None
                probabilities = torch.tensor(
                    record.current_probabilities, dtype=torch.float64
                )
                action = int(
                    torch.multinomial(
                        probabilities, 1, replacement=True, generator=generator
                    ).item()
                )
                outcome, step_trace = self._outcome(current, action)
                rewards.append(outcome.reward)
                trace.append(step_trace)
                terminal = outcome.terminal
                if terminal:
                    current = None
                    break
                current = outcome.next_state_id
            return TheoryBridgeRollout(
                rewards=tuple(rewards),
                terminal=terminal,
                bootstrap_state_id=current,
                trajectory_sha256=canonical_json_sha256(
                    {
                        "initial_state_id": state_id,
                        "horizon": horizon,
                        "seed": seed,
                        "steps": trace,
                        "terminal": terminal,
                        "bootstrap_state_id": current,
                    }
                ),
            )

        return self._guard(sample)

    def sample_current_policy_return(
        self,
        state_id: str,
        seed: int,
        maximum_environment_steps: int,
        gamma: float,
    ) -> SimpleNamespace:
        """Sample one independent finite return under the frozen current policy."""

        def sample() -> SimpleNamespace:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise FullBackendError("Theory return seed must be nonnegative.")
            if (
                isinstance(maximum_environment_steps, bool)
                or not isinstance(maximum_environment_steps, int)
                or maximum_environment_steps <= 0
            ):
                raise FullBackendError(
                    "Theory return step limit must be a positive integer."
                )
            if (
                isinstance(gamma, bool)
                or not isinstance(gamma, (int, float))
                or not 0.0 <= float(gamma) < 1.0
            ):
                raise FullBackendError("Theory return gamma is invalid.")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            current: str | None = state_id
            discounted_return = 0.0
            discount = 1.0
            trace: list[dict[str, object]] = []
            terminal = False
            for _ in range(maximum_environment_steps):
                if current is None:
                    terminal = True
                    break
                record = self._record(current)
                if record.terminal:
                    terminal = True
                    break
                self._populate_probabilities(record)
                if record.current_probabilities is None:
                    raise FullBackendError(
                        "Theory current-policy return lacks action probabilities."
                    )
                probabilities = torch.tensor(
                    record.current_probabilities,
                    dtype=torch.float64,
                )
                action = int(
                    torch.multinomial(
                        probabilities,
                        1,
                        replacement=True,
                        generator=generator,
                    ).item()
                )
                outcome, step_trace = self._outcome(current, action)
                discounted_return += discount * outcome.reward
                discount *= float(gamma)
                trace.append(step_trace)
                terminal = outcome.terminal
                current = None if terminal else outcome.next_state_id
                if terminal:
                    break
            return SimpleNamespace(
                discounted_return=discounted_return,
                environment_steps=len(trace),
                terminal=terminal,
                trajectory_sha256=canonical_json_sha256(
                    {
                        "initial_state_id": state_id,
                        "seed": seed,
                        "maximum_environment_steps": maximum_environment_steps,
                        "gamma": float(gamma),
                        "steps": trace,
                        "terminal": terminal,
                    }
                ),
            )

        return self._guard(sample)


def _theory_mapping(
    value: object,
    fields: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FullBackendError(f"Theory {name} identity field inventory differs.")
    return value


def _validate_theory_evaluator_identity(identity: Mapping[str, object]) -> None:
    source = _theory_mapping(
        identity.get("evaluator_source"),
        {"git_commit", "source_manifest_sha256"},
        name="evaluator source",
    )
    commit = source["git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in _LOWER_HEX for character in commit)
    ):
        raise FullBackendError("Theory evaluator source commit is invalid.")
    _require_digest(
        source["source_manifest_sha256"],
        name="theory evaluator source manifest",
    )
    evaluator_runtime = _theory_mapping(
        identity.get("evaluator_runtime"),
        {
            "runtime_sha256",
            "runtime_profile_sha256",
            "runtime_authorization_sha256",
            "launcher_sha256",
        },
        name="evaluator runtime",
    )
    for field, value in evaluator_runtime.items():
        _require_digest(value, name=f"theory evaluator {field}")


def _validate_theory_identity_bundle(
    *,
    supplied: Mapping[str, object],
    run: RegisteredFullRun,
    runtime: SealedRuntimeIdentity,
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
    embedded_checkpoint_identity: Mapping[str, object],
    session: LearnedSession,
    adapter: ReadOnlyTheoryBridgeSession,
    training_module: Any,
    registered_state_indices: tuple[int, ...],
) -> None:
    expected_fields = {
        "protocol_sha256",
        "theory_amendment_sha256",
        "registry_row_sha256",
        "checkpoint",
        "model",
        "config",
        "producer_source",
        "training_runtime",
        "dataset_records",
        "evaluator_source",
        "evaluator_runtime",
    }
    if set(supplied) != expected_fields:
        raise FullBackendError("Theory identity bundle field inventory differs.")
    _validate_theory_evaluator_identity(supplied)
    evaluator_runtime = supplied["evaluator_runtime"]
    assert isinstance(evaluator_runtime, Mapping)
    if (
        evaluator_runtime["runtime_authorization_sha256"]
        != run.runtime_authorization_sha256
    ):
        raise FullBackendError("Theory evaluator mixes runtime authorizations.")

    theory_amendments = [
        amendment
        for amendment in run.amendment_history
        if amendment.get("schema_name")
        == "policy_improvement_theory_bridge_amendment_v1"
    ]
    if len(theory_amendments) != 1:
        raise FullBackendError(
            "Registered run must contain exactly one theory-bridge amendment."
        )
    methods = [
        method
        for method in run.protocol.get("methods", [])
        if isinstance(method, Mapping)
        and method.get("id") == run.row.get("base_method_id")
    ]
    if len(methods) != 1:
        raise FullBackendError("Theory method config does not resolve exactly.")
    split = str(run.row["evaluation_split"])
    if split not in {"train", "validation"} or run.test_open_sha256 is not None:
        raise FullBackendError("Theory bridge cannot open test data.")
    split_registration = run.protocol.get("dataset", {}).get("splits", {}).get(split)
    if not isinstance(split_registration, Mapping):
        raise FullBackendError("Theory dataset split registration is unavailable.")
    samples = dataset_sample_sha256s(session.evaluation_dataset)
    if not samples or max(registered_state_indices) >= len(samples):
        raise FullBackendError("Theory registered-state prefix exceeds the dataset.")
    actual_order = ordered_record_sha256(samples)
    registered_order = split_registration.get("ordered_record_sha256")
    registered_manifest = split_registration.get("manifest_sha256")
    if not isinstance(registered_order, Mapping) or not isinstance(
        registered_manifest, Mapping
    ):
        raise FullBackendError("Theory dataset registration is incomplete.")
    if actual_order != registered_order.get("value"):
        raise FullBackendError("Theory dataset record order differs from protocol.")
    record_pairs = [
        {
            "record_index": index,
            "dataset_record_sha256": samples[index],
        }
        for index in registered_state_indices
    ]
    expected_runtime = {
        "role": runtime.role,
        "source_git_commit": runtime.source_git_commit,
        "source_manifest_sha256": runtime.source_manifest_sha256,
        "runtime_sha256": runtime.runtime_sha256,
        "runtime_profile_sha256": runtime.runtime_profile_sha256,
        "selected_source_manifest_sha256": (runtime.selected_source_manifest_sha256),
        "runtime_authorization_sha256": runtime.runtime_authorization_sha256,
        "launcher_sha256": runtime.launcher_sha256,
    }
    expected = {
        "protocol_sha256": run.protocol_sha256,
        "theory_amendment_sha256": canonical_json_sha256(theory_amendments[0]),
        "registry_row_sha256": run.registry_row_sha256,
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size_bytes,
            "snapshot_kind": embedded_checkpoint_identity["snapshot_kind"],
            "environment_interactions": embedded_checkpoint_identity[
                "environment_interactions"
            ],
        },
        "model": adapter.observed_model_identity(training_module),
        "config": {
            "file_sha256": methods[0].get("config_sha256"),
            "base_canonical_sha256": run.row.get("base_config_canonical_sha256"),
            "effective_config_sha256": run.row.get("expected_effective_config_sha256"),
        },
        "producer_source": {
            "git_commit": runtime.source_git_commit,
            "source_manifest_sha256": runtime.producer_source_manifest_sha256,
        },
        "training_runtime": expected_runtime,
        "dataset_records": {
            "split": split,
            "split_manifest_sha256": registered_manifest.get("value"),
            "ordered_record_sha256": registered_order.get("value"),
            "selected_record_indices_sha256": canonical_json_sha256(
                list(registered_state_indices)
            ),
            "selected_records_sha256": canonical_json_sha256(record_pairs),
            "record_count": len(registered_state_indices),
        },
        "evaluator_source": copy.deepcopy(supplied["evaluator_source"]),
        "evaluator_runtime": copy.deepcopy(supplied["evaluator_runtime"]),
    }
    if dict(supplied) != expected:
        raise FullBackendError(
            "Theory request contains missing or mixed training identities."
        )

    embedded_expected = {
        "run_id": run.row["run_id"],
        "method_id": run.row["method_id"],
        "protocol_sha256": run.protocol_sha256,
        "registry_row_sha256": run.registry_row_sha256,
        "amendment_history_sha256": run.amendment_history_sha256,
        "runtime_authorization_sha256": run.runtime_authorization_sha256,
        "training_runtime_sha256": runtime.runtime_sha256,
        "training_source_git_commit": runtime.source_git_commit,
        "training_source_manifest_sha256": runtime.source_manifest_sha256,
        "launcher_sha256": runtime.launcher_sha256,
        "dataset_manifest_sha256": session.dataset_manifest_sha256,
        "dataset_provenance_sha256": canonical_json_sha256(session.dataset_provenance),
        "effective_config_sha256": session.effective_config_sha256,
        "test_open_sha256": None,
    }
    if any(
        embedded_checkpoint_identity.get(field) != value
        for field, value in embedded_expected.items()
    ):
        raise FullBackendError("Theory checkpoint mixes registered identities.")


def _registered_theory_checkpoint_snapshot_kind(
    run: RegisteredFullRun,
    environment_interactions: object,
) -> str:
    checkpoints = run.interaction_checkpoints
    final_interactions = run.final_environment_interactions
    if (
        not checkpoints
        or checkpoints[-1] != final_interactions
        or isinstance(environment_interactions, bool)
        or not isinstance(environment_interactions, int)
        or environment_interactions not in checkpoints
    ):
        raise FullBackendError(
            "Theory checkpoint progress differs from the registered run schedule."
        )
    return (
        "interaction_matched"
        if environment_interactions == final_interactions
        else "scheduled"
    )


def _sealed_checkpoint_identity(descriptor: int) -> tuple[str, int]:
    """Hash one stable write-sealed theory checkpoint descriptor.

    The seal-verification primitive is shared with the evidence-audit boundary
    so both trust boundaries cannot drift apart.
    """

    try:
        return authenticate_sealed_descriptor(descriptor)
    except SealedCheckpointError as exc:
        raise FullBackendError(str(exc)) from exc


def open_theory_bridge_session(
    request: Mapping[str, object],
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    sealed_checkpoint_descriptor: int,
    authenticated_model_state_sha256: str,
    authenticated_role_state_sha256s: Mapping[str, object],
    authenticated_validation_sha256: str,
    runtime_identity: Mapping[str, object],
    training_module: Any,
) -> ReadOnlyTheoryBridgeSession:
    """Open a disposable, mutation-guarded checkpoint for theory evaluation."""

    if set(request) != {
        "identity",
        "registered_run",
        "registered_state_indices",
    }:
        raise FullBackendError("Theory bridge request field inventory differs.")
    identity = request["identity"]
    run = request["registered_run"]
    indices = request["registered_state_indices"]
    if not isinstance(identity, Mapping):
        raise FullBackendError("Theory bridge request identity must be a mapping.")
    if not isinstance(run, RegisteredFullRun):
        raise FullBackendError("Theory bridge request lacks a registered full run.")
    if (
        not isinstance(indices, (list, tuple))
        or not indices
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= run.evaluation_records
            for index in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise FullBackendError("Theory bridge registered-state indices are invalid.")
    expected = _require_digest(
        expected_checkpoint_sha256, name="theory checkpoint SHA-256"
    )
    supplied_checkpoint = Path(checkpoint_path)
    if not supplied_checkpoint.is_absolute() or ".." in supplied_checkpoint.parts:
        raise FullBackendError("Theory checkpoint path must be absolute and canonical.")
    sealed_sha256, sealed_size_bytes = _sealed_checkpoint_identity(
        sealed_checkpoint_descriptor
    )
    if sealed_sha256 != expected:
        raise FullBackendError(
            "Sealed theory checkpoint digest differs before restore."
        )
    authenticated_model = _require_digest(
        authenticated_model_state_sha256,
        name="authenticated theory checkpoint model-state SHA-256",
    )
    authenticated_validation = _require_digest(
        authenticated_validation_sha256,
        name="authenticated theory checkpoint validation SHA-256",
    )
    if (
        not isinstance(authenticated_role_state_sha256s, Mapping)
        or not authenticated_role_state_sha256s
        or any(
            not isinstance(role, str) or not role
            for role in authenticated_role_state_sha256s
        )
    ):
        raise FullBackendError(
            "Authenticated theory checkpoint role-state inventory is invalid."
        )
    authenticated_roles = {
        role: _require_digest(
            digest,
            name=f"authenticated theory checkpoint role {role!r}",
        )
        for role, digest in authenticated_role_state_sha256s.items()
    }
    if canonical_json_sha256(authenticated_roles) != authenticated_model:
        raise FullBackendError(
            "Authenticated theory checkpoint model and role identities differ."
        )
    checkpoint_identity = _theory_mapping(
        identity.get("checkpoint"),
        {"sha256", "size_bytes", "snapshot_kind", "environment_interactions"},
        name="checkpoint",
    )
    if (
        checkpoint_identity.get("sha256") != expected
        or checkpoint_identity.get("size_bytes") != sealed_size_bytes
    ):
        raise FullBackendError("Theory request checkpoint identity differs.")
    runtime_mapping = dict(runtime_identity)
    if "producer_source_manifest_sha256" not in runtime_mapping:
        producer_identity = identity.get("producer_source")
        if not isinstance(producer_identity, Mapping):
            raise FullBackendError("Theory request lacks producer source identity.")
        runtime_mapping["producer_source_manifest_sha256"] = producer_identity.get(
            "source_manifest_sha256"
        )
    runtime = SealedRuntimeIdentity.from_mapping(runtime_mapping)
    if run.runtime_authorization_sha256 != runtime.runtime_authorization_sha256:
        raise FullBackendError("Theory request mixes runtime authorizations.")
    existing_runtime = getattr(training_module, "_PREVERIFIED_RUNTIME_SHA256", None)
    if existing_runtime not in {None, runtime.runtime_sha256}:
        raise FullBackendError("Theory training module names another runtime.")
    training_module._PREVERIFIED_RUNTIME_SHA256 = runtime.runtime_sha256
    engine = TorchLearnedRunEngine(training_module)
    rng_state = training_module._capture_rng_state()
    try:
        session = engine._build_session(run, runtime)
        if type(session.trainer).__name__ == "PPOTrainer":
            from policy_improvement_smoke_checkpoint import (
                load_stable_checkpoint,
                validate_ppo_smoke_checkpoint,
            )

            loaded, observed = load_stable_checkpoint(
                sealed_checkpoint_descriptor,
                expected_sha256=expected,
            )
            if not isinstance(loaded, Mapping):
                raise FullBackendError("Theory PPO checkpoint payload is invalid.")
            validate_ppo_smoke_checkpoint(
                loaded,
                session.trainer,
                expected_identity=loaded["identity"],
                validate_only=False,
            )
            embedded_identity = validate_full_checkpoint_identity(
                loaded.get("identity")
            )
            evaluation_states: Mapping[str, Mapping[str, object]] = {}
        else:
            loaded, observed = training_module._load_checkpoint_payload(
                sealed_checkpoint_descriptor,
                expected_sha256=expected,
            )
            if not isinstance(loaded, Mapping):
                raise FullBackendError("Theory UPI checkpoint payload is invalid.")
            training_module.resume_from_checkpoint_for_theory_evaluation(
                sealed_checkpoint_descriptor,
                session.model,
                session.trainer,
                str(session.device),
                expected_dataset_provenance=session.dataset_provenance,
                expected_run_identity=session.run_identity,
                expected_checkpoint_sha256=expected,
            )
            raw_states = loaded.get(
                "policy_improvement_full_evaluation_state_dicts", {}
            )
            if not isinstance(raw_states, Mapping):
                raise FullBackendError("Theory evaluation-state inventory is invalid.")
            evaluation_states = raw_states
            embedded_identity = validate_full_checkpoint_identity(
                loaded.get("policy_improvement_full_identity")
            )
        restored_sha256, restored_size_bytes = _sealed_checkpoint_identity(
            sealed_checkpoint_descriptor
        )
        if (
            observed != expected
            or restored_sha256 != expected
            or restored_size_bytes != sealed_size_bytes
        ):
            raise FullBackendError("Theory checkpoint changed during restore.")
    finally:
        training_module._restore_rng_state(rng_state)
    embedded_environment_interactions = embedded_identity["environment_interactions"]
    expected_snapshot_kind = _registered_theory_checkpoint_snapshot_kind(
        run,
        embedded_environment_interactions,
    )
    if (
        embedded_identity["snapshot_kind"] != expected_snapshot_kind
        or checkpoint_identity["snapshot_kind"] != expected_snapshot_kind
        or checkpoint_identity["environment_interactions"]
        != embedded_environment_interactions
    ):
        raise FullBackendError(
            "Theory bridge requires the exact registered checkpoint snapshot."
        )
    restored_model_state_sha256, restored_role_state_sha256s = (
        session_model_state_identity(
            session,
            evaluation_state_dicts=evaluation_states,
        )
    )
    if (
        restored_model_state_sha256 != authenticated_model
        or restored_role_state_sha256s != authenticated_roles
    ):
        raise FullBackendError(
            "Restored theory model differs from its authenticated checkpoint "
            f"validation {authenticated_validation}."
        )
    try:
        adapter_descriptor = os.dup(sealed_checkpoint_descriptor)
    except OSError as exc:
        raise FullBackendError("Sealed theory checkpoint cannot be retained.") from exc
    try:
        adapter = ReadOnlyTheoryBridgeSession(
            request_identity=identity,
            checkpoint_descriptor=adapter_descriptor,
            checkpoint_sha256=expected,
            snapshot_kind=str(embedded_identity["snapshot_kind"]),
            environment_interactions=int(embedded_identity["environment_interactions"]),
            session=session,
            training_module=training_module,
            evaluation_state_dicts=evaluation_states,
            registered_state_indices=tuple(indices),
        )
    except BaseException:
        os.close(adapter_descriptor)
        raise
    try:
        _validate_theory_identity_bundle(
            supplied=identity,
            run=run,
            runtime=runtime,
            checkpoint_sha256=expected,
            checkpoint_size_bytes=sealed_size_bytes,
            embedded_checkpoint_identity=embedded_identity,
            session=session,
            adapter=adapter,
            training_module=training_module,
            registered_state_indices=tuple(indices),
        )
    except BaseException:
        adapter.close()
        raise
    return adapter


def _stage0_learned_session(
    context: SmokeContext,
    session: SmokeSession,
) -> LearnedSession:
    population = session.evaluation_population
    if not isinstance(population, Mapping):
        raise FullBackendError(
            "Stage 0 theory evaluation lacks its registered train population."
        )
    indices = population.get("indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        )
    ):
        raise FullBackendError("Stage 0 theory population indices are invalid.")
    run = RegisteredFullRun(
        project_root=context.source_root,
        protocol_path=(
            context.source_root / "configs/policy_improvement_v2/protocol.json"
        ),
        registry_path=(
            context.source_root / "configs/policy_improvement_v2/registry.json"
        ),
        evidence_root=context.evidence_root,
        protocol=copy.deepcopy(context.protocol),
        registry=copy.deepcopy(context.registry),
        amendment_history=(),
        row=copy.deepcopy(context.row),
        protocol_sha256=context.protocol_sha256,
        registry_sha256=context.registry_sha256,
        amendment_history_sha256=hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        registry_row_sha256=context.registry_row_sha256,
        runtime_authorization_sha256=context.runtime_authorization_sha256,
        interaction_checkpoints=(16, 32),
        final_environment_interactions=32,
        compute_target_recurrent_map_applications=0,
        evaluation_records=len(indices),
        test_open_sha256=None,
        dataset_root=context.dataset_root,
    )
    train_order = context.protocol["dataset"]["splits"]["train"][
        "ordered_record_sha256"
    ]
    if not isinstance(train_order, Mapping) or train_order.get("status") != "available":
        raise FullBackendError("Stage 0 train-order identity is unavailable.")
    population_order = population.get("ordered_record_sha256")
    population_binding = population.get("binding_sha256")
    for value, label in (
        (population_order, "Stage 0 population record order"),
        (population_binding, "Stage 0 population binding"),
    ):
        _require_digest(value, name=label)
    return LearnedSession(
        run=run,
        model=session.model,
        trainer=session.trainer,
        rl_config=session.rl_config,
        env_config=session.env_config,
        train_dataset=session.train_dataset,
        evaluation_dataset=session.evaluation_dataset,
        checker=session.checker,
        task_config=session.task_config,
        dataset_provenance=copy.deepcopy(session.dataset_provenance),
        effective_config=copy.deepcopy(session.effective_config),
        effective_config_sha256=session.effective_config_sha256,
        initialization_sha256=session.initialization_sha256,
        device=session.device,
        config_path=session.config_path,
        method_config_sha256=session.method_config_sha256,
        run_identity=copy.deepcopy(session.run_identity),
        evidence_identity=copy.deepcopy(session.evidence_identity),
        dataset_manifest_sha256=context.dataset_manifest_sha256,
        train_ordered_records_sha256=str(train_order["value"]),
        evaluation_ordered_records_sha256=str(population_order),
        evaluation_pool_sha256=str(population_binding),
        parent_environment_interactions=16,
    )


def open_stage0_theory_bridge_session_v2(
    request: Mapping[str, object],
    checkpoint_path: str | Path,
    *,
    context: SmokeContext,
    expected_checkpoint_sha256: str,
    sealed_checkpoint_descriptor: int,
    parent_checkpoint_sha256: str,
    authenticated_model_state_sha256: str,
    authenticated_role_state_sha256s: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    training_module: Any,
) -> ReadOnlyTheoryBridgeSession:
    """Restore one authenticated 16 -> 32 Stage 0 checkpoint read-only."""

    if (
        context.segment_name != "resume"
        or context.segment_budget != 32
        or context.row.get("phase") != "stage0_smoke"
        or context.row.get("evaluation_split") != "train"
        or context.row.get("evaluation_population") != "stage0_smoke"
        or context.row.get("method_id")
        not in {"fixed_base_exact_persistent", "fixed_base_exact_episodic"}
    ):
        raise FullBackendError(
            "Stage 0 theory bridge requires one exact train-only resume row."
        )
    expected = _require_digest(
        expected_checkpoint_sha256,
        name="Stage 0 theory checkpoint SHA-256",
    )
    parent = _require_digest(
        parent_checkpoint_sha256,
        name="Stage 0 theory parent checkpoint SHA-256",
    )
    supplied_checkpoint = Path(checkpoint_path)
    if not supplied_checkpoint.is_absolute() or ".." in supplied_checkpoint.parts:
        raise FullBackendError("Stage 0 theory checkpoint path is not canonical.")
    sealed_sha256, sealed_size_bytes = _sealed_checkpoint_identity(
        sealed_checkpoint_descriptor
    )
    if sealed_sha256 != expected:
        raise FullBackendError("Stage 0 sealed checkpoint digest differs.")
    authenticated_model = _require_digest(
        authenticated_model_state_sha256,
        name="Stage 0 authenticated model-state SHA-256",
    )
    if not isinstance(authenticated_role_state_sha256s, Mapping):
        raise FullBackendError("Stage 0 authenticated role inventory is invalid.")
    authenticated_roles = {
        str(role): _require_digest(
            digest,
            name=f"Stage 0 authenticated role {role!r}",
        )
        for role, digest in authenticated_role_state_sha256s.items()
    }
    if (
        not authenticated_roles
        or canonical_json_sha256(authenticated_roles) != authenticated_model
    ):
        raise FullBackendError("Stage 0 model and role identities differ.")
    checkpoint_identity = _theory_mapping(
        request.get("checkpoint"),
        {"sha256", "size_bytes", "snapshot_kind", "environment_interactions"},
        name="Stage 0 checkpoint",
    )
    if dict(checkpoint_identity) != {
        "sha256": expected,
        "size_bytes": sealed_size_bytes,
        "snapshot_kind": "smoke_resume",
        "environment_interactions": 32,
    }:
        raise FullBackendError("Stage 0 request checkpoint identity differs.")
    runtime_mapping = dict(runtime_identity)
    runtime_mapping.setdefault(
        "producer_source_manifest_sha256",
        context.producer_manifest_sha256,
    )
    runtime = SealedRuntimeIdentity.from_mapping(runtime_mapping)
    if (
        runtime.role != "policy-improvement-smoke"
        or runtime.runtime_authorization_sha256 != context.runtime_authorization_sha256
        or runtime.runtime_sha256 != context.runtime_sha256
    ):
        raise FullBackendError("Stage 0 theory runtime identity differs.")
    existing_runtime = getattr(training_module, "_PREVERIFIED_RUNTIME_SHA256", None)
    if existing_runtime not in {None, runtime.runtime_sha256}:
        raise FullBackendError("Stage 0 training module names another runtime.")
    training_module._PREVERIFIED_RUNTIME_SHA256 = runtime.runtime_sha256
    smoke_session = build_stage0_validation_session(context, training_module)
    loaded, observed = training_module._load_checkpoint_payload(
        sealed_checkpoint_descriptor,
        expected_sha256=expected,
    )
    if not isinstance(loaded, Mapping):
        raise FullBackendError("Stage 0 theory checkpoint payload is invalid.")
    validate_policy_improvement_smoke_identity(
        loaded.get("policy_improvement_smoke_identity"),
        context,
        environment_interactions=32,
        parent_checkpoint_sha256=parent,
    )
    if (
        loaded.get("evidence_identity") != smoke_session.evidence_identity
        or loaded.get("run_identity") != smoke_session.run_identity
    ):
        raise FullBackendError("Stage 0 checkpoint mixes registered identities.")
    training_module.resume_from_checkpoint_for_theory_evaluation(
        sealed_checkpoint_descriptor,
        smoke_session.model,
        smoke_session.trainer,
        str(smoke_session.device),
        expected_dataset_provenance=smoke_session.dataset_provenance,
        expected_run_identity=smoke_session.run_identity,
        expected_checkpoint_sha256=expected,
    )
    restored_sha256, restored_size_bytes = _sealed_checkpoint_identity(
        sealed_checkpoint_descriptor
    )
    if (
        observed != expected
        or restored_sha256 != expected
        or restored_size_bytes != sealed_size_bytes
        or smoke_session.trainer.get_env_step_count() != 32
    ):
        raise FullBackendError("Stage 0 checkpoint restore is incomplete.")
    restored_model, restored_roles = stage0_model_state_identity(smoke_session)
    if restored_model != authenticated_model or restored_roles != authenticated_roles:
        raise FullBackendError(
            "Restored Stage 0 model differs from semantic checkpoint validation."
        )
    learned_session = _stage0_learned_session(context, smoke_session)
    population = smoke_session.evaluation_population
    assert isinstance(population, Mapping)
    reported_indices = population["indices"]
    assert isinstance(reported_indices, list)
    try:
        adapter_descriptor = os.dup(sealed_checkpoint_descriptor)
    except OSError as exc:
        raise FullBackendError("Stage 0 sealed checkpoint cannot be retained.") from exc
    try:
        adapter = ReadOnlyTheoryBridgeSession(
            request_identity=request,
            checkpoint_descriptor=adapter_descriptor,
            checkpoint_sha256=expected,
            snapshot_kind="smoke_resume",
            environment_interactions=32,
            session=learned_session,
            training_module=training_module,
            evaluation_state_dicts={},
            registered_state_indices=tuple(range(len(reported_indices))),
            reported_record_indices=tuple(int(index) for index in reported_indices),
        )
    except BaseException:
        os.close(adapter_descriptor)
        raise
    try:
        observed_model = adapter.observed_model_identity(training_module)
        supplied_model = request.get("model")
        if (
            not isinstance(supplied_model, Mapping)
            or dict(supplied_model) != observed_model
        ):
            raise FullBackendError(
                "Stage 0 theory request model identity differs after restore."
            )
    except BaseException:
        adapter.close()
        raise
    return adapter


def load_theory_bridge_registered_run(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    registry_path: str | Path,
    amendment_paths: tuple[str | Path, ...] | list[str | Path],
    evidence_root: str | Path,
    dataset_root: str | Path,
    row_id: str,
    runtime_authorization_sha256: str,
) -> RegisteredFullRun:
    """Authenticate one registered row for read-only checkpoint evaluation."""

    return load_registered_full_run(
        project_root=project_root,
        protocol_path=protocol_path,
        registry_path=registry_path,
        amendment_paths=amendment_paths,
        evidence_root=evidence_root,
        dataset_root=dataset_root,
        row_id=row_id,
        runtime_authorization_sha256=runtime_authorization_sha256,
        environment={FULL_EXECUTION_ENV: FULL_EXECUTION_VALUE},
    )


class SealedFullRunBackend(FullRunBackend):
    """FullRunBackend bound to one authenticated PAR and authorization."""

    def __init__(
        self,
        *,
        runtime_identity: Mapping[str, object],
        training_module: Any,
        engine: LearnedRunEngine | None = None,
    ) -> None:
        self._runtime = SealedRuntimeIdentity.from_mapping(runtime_identity)
        self._training_module = training_module
        existing_runtime = getattr(training_module, "_PREVERIFIED_RUNTIME_SHA256", None)
        if existing_runtime not in {None, self._runtime.runtime_sha256}:
            raise FullBackendError(
                "Training module was preflighted for another runtime artifact."
            )
        training_module._PREVERIFIED_RUNTIME_SHA256 = self._runtime.runtime_sha256
        self._engine = engine or TorchLearnedRunEngine(training_module)

    def execute(self, request: BackendRequest) -> BackendPackage:
        run = request.run
        if run.runtime_authorization_sha256 != (
            self._runtime.runtime_authorization_sha256
        ):
            raise FullRuntimeError(
                "Registered run names another runtime authorization."
            )
        required_content_splits(run)
        if run.dataset_root is None:
            raise FullRuntimeError(
                "Authenticated full execution requires dataset root."
            )
        if not request.require_strict_checkpoint_resume:
            raise FullRuntimeError("Full execution cannot disable strict resume.")
        if not request.require_interaction_and_compute_snapshots:
            raise FullRuntimeError("Full execution cannot omit a registered snapshot.")
        return self._engine.execute(request, self._runtime)


def _failed_snapshot(reason: str, kind: str) -> dict[str, object]:
    unavailable = _unavailable(reason)
    return {
        "snapshot_kind": kind,
        "status": "unavailable",
        "unavailable_reason": reason,
        "target": {
            "unit": (
                "environment_interactions"
                if kind == "interaction_matched"
                else "recurrent_map_applications"
            ),
            "registered_quantity": dict(unavailable),
        },
        "observed_environment_interactions": dict(unavailable),
        "observed_recurrent_map_applications": dict(unavailable),
        "accelerator_seconds_observed": dict(unavailable),
        "checkpoint_sha256": dict(unavailable),
        "model_state_sha256": dict(unavailable),
        "checkpoint_lineage_sha256": dict(unavailable),
        "policy_evaluations": [],
    }


def build_failed_result(
    run: RegisteredFullRun,
    runtime: SealedRuntimeIdentity,
    *,
    phase: str,
    error: BaseException,
    session: LearnedSession | None = None,
) -> dict[str, object]:
    """Build the schema-valid failure record required by immutable attempts."""

    reason = (
        "run_failed_before_checkpoint"
        if phase in {"training", "checkpoint"}
        else "run_failed_before_evaluation"
    )
    unavailable = _unavailable(reason)
    diagnostics = {
        name: dict(unavailable)
        for name in (
            "L_preproj",
            "projection_active_rate",
            "absolute_depth_policy_agreement",
            "finite_depth_discrepancy",
            "fixed_target_residual",
            "propagated_target_lag",
            "exact_centering_defect",
            "candidate_current_tv",
            "mixture_realized_tv",
            "mixture_realized_kl",
            "fresh_initialization_discrepancy",
            "value_of_memory",
        )
    }
    error_class = (
        "".join(
            character.lower() if character.isalnum() else "_"
            for character in type(error).__name__
        ).strip("_")
        or "runtime_error"
    )
    training_fields = {
        name: dict(unavailable)
        for name in (
            "interactions_to_first_solve",
            "value_loss",
            "policy_loss",
            "wall_time_seconds",
            "gpu_hours",
            "gpu_utilization_fraction",
            "gpu_utilization_sample_count",
            "gpu_utilization_sampling_interval_seconds",
            "peak_allocated_memory_bytes",
            "peak_reserved_memory_bytes",
            "optimizer_steps",
            "cells_processed",
            "actions_processed",
            "tokens_processed",
            "policy_head_calls",
            "recurrent_map_applications",
            "value_head_calls",
        )
    }
    methods = [
        method
        for method in run.protocol.get("methods", [])
        if isinstance(method, Mapping)
        and method.get("id") == run.row.get("base_method_id")
    ]
    if len(methods) != 1:
        raise FullBackendError(
            "Failed attempt cannot resolve its registered method identity."
        )
    split = str(run.row["evaluation_split"])
    dataset = run.protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise FullBackendError(
            "Failed attempt cannot resolve its registered dataset identity."
        )
    splits = dataset.get("splits")
    if not isinstance(splits, Mapping):
        raise FullBackendError(
            "Failed attempt cannot resolve its registered dataset splits."
        )
    train_registration = splits.get("train")
    evaluation_registration = splits.get(split)
    if not isinstance(train_registration, Mapping) or not isinstance(
        evaluation_registration, Mapping
    ):
        raise FullBackendError(
            "Failed attempt cannot resolve its registered dataset split identities."
        )

    def registered_digest(value: object, *, name: str) -> str:
        if not isinstance(value, Mapping) or value.get("status") != "available":
            raise FullBackendError(f"Failed attempt lacks registered {name}.")
        return _require_digest(value.get("value"), name=name)

    method_config_sha256 = _require_digest(
        methods[0].get("config_sha256"), name="failed method config SHA-256"
    )
    effective_config_sha256 = _require_digest(
        run.row.get("expected_effective_config_sha256"),
        name="failed effective config SHA-256",
    )
    dataset_manifest_sha256 = registered_digest(
        dataset.get("manifest_sha256"), name="failed dataset manifest SHA-256"
    )
    train_order_sha256 = registered_digest(
        train_registration.get("ordered_record_sha256"),
        name="failed train order SHA-256",
    )
    population_tier = (
        "confirmatory"
        if run.row["tier"] in {"confirmatory", "ablation"}
        else run.row["tier"]
    )
    populations = run.protocol.get("evaluation_populations")
    population = (
        populations.get(population_tier) if isinstance(populations, Mapping) else None
    )
    if not isinstance(population, Mapping):
        raise FullBackendError(
            "Failed attempt cannot resolve its registered evaluation population."
        )
    evaluation_order_sha256 = registered_digest(
        population.get("ordered_record_sha256"),
        name="failed evaluation order SHA-256",
    )
    initialization_sha256 = (
        session.initialization_sha256
        if session is not None
        else _unavailable("initialization_not_materialized")
    )
    if session is None:
        device_identity: object = _unavailable("execution_device_not_materialized")
    elif session.device.type == "cuda":
        device_index = session.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_identity = f"cuda:{device_index}"
    else:
        device_identity = str(session.device)
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": run.protocol["protocol_id"],
        "protocol_sha256": run.protocol_sha256,
        "amendment_history_sha256": run.amendment_history_sha256,
        **{
            field: run.row[field]
            for field in (
                "run_id",
                "phase",
                "tier",
                "seed",
                "evaluation_split",
                "method_id",
                "base_method_id",
                "n",
                "K",
                "alpha",
                "ablation_variant",
            )
        },
        "registry_row_sha256": run.registry_row_sha256,
        "applied_config_override": run.row["config_override"],
        "primary_policy_variant": primary_policy_variant_for_method(
            str(run.row["method_id"])
        ),
        "evaluation_snapshots": [
            _failed_snapshot(reason, "interaction_matched"),
            _failed_snapshot(reason, "compute_matched"),
        ],
        "status": "failed",
        "failure": {
            "phase": phase,
            "error_class": error_class,
            "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        },
        "identities": {
            "producer_git_commit": runtime.source_git_commit,
            "git_clean": True,
            "runtime_authorization_sha256": run.runtime_authorization_sha256,
            "training_source_git_commit": runtime.source_git_commit,
            "training_runtime_sha256": runtime.runtime_sha256,
            "training_runtime_profile_sha256": runtime.runtime_profile_sha256,
            "training_selected_source_manifest_sha256": (
                runtime.selected_source_manifest_sha256
            ),
            "launcher_sha256": runtime.launcher_sha256,
            "producer_manifest_sha256": runtime.producer_source_manifest_sha256,
            "method_config_sha256": (
                session.method_config_sha256
                if session is not None
                else method_config_sha256
            ),
            "effective_config_sha256": (effective_config_sha256),
            "dataset_manifest_sha256": (
                session.dataset_manifest_sha256
                if session is not None
                else dataset_manifest_sha256
            ),
            "train_ordered_records_sha256": (
                session.train_ordered_records_sha256
                if session is not None
                else train_order_sha256
            ),
            "evaluation_ordered_records_sha256": (
                session.evaluation_ordered_records_sha256
                if session is not None
                else evaluation_order_sha256
            ),
            "initialization_sha256": initialization_sha256,
            "checkpoint_sha256": dict(unavailable),
            "model_state_sha256": dict(unavailable),
            "evaluation_runtime_sha256": dict(unavailable),
            "evaluation_source_git_commit": dict(unavailable),
            "evaluation_runtime_profile_sha256": dict(unavailable),
            "evaluation_selected_source_manifest_sha256": dict(unavailable),
            "evaluation_pool_sha256": dict(unavailable),
            "test_open_sha256": (
                _available(run.test_open_sha256)
                if run.test_open_sha256 is not None
                else _unavailable("test_data_not_opened")
            ),
            "device": device_identity,
        },
        "metrics": {"training": training_fields, "diagnostics": diagnostics},
        "artifacts": {
            "checkpoint": dict(unavailable),
            "checkpoint_validation": dict(unavailable),
            "model_state_inventory": dict(unavailable),
            "run_manifest": dict(unavailable),
        },
    }
    return result
