#!/usr/bin/env python3

import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import torch
import numpy as np
from torch.distributions import Categorical

from scripts import audit_phase4_paper_ready
from scripts import eval_phase4_2x2_norm_ablation
from scripts import make_paper_figures_phase4
from scripts import phase4_checkpoint as phase4_checkpoint_module
from scripts.phase4_checkpoint import (
    PHASE4_EPISODE_LENGTH,
    PHASE4_FINAL_ENV_STEPS,
    PHASE4_FINAL_EPISODES,
    PHASE4_FINAL_REPLAY_SIZE,
    Phase4CheckpointError,
    load_phase4_checkpoint,
    phase4_checkpoint_relpath,
    phase4_run_id,
    verify_phase4_summary_checkpoints,
    _expected_model_config,
    _expected_rl_config,
    _phase4_environment,
    _registered_phase4_config_layer,
)
from scripts.phase4_diagnostic_inputs import (
    Phase4DiagnosticInputError,
    PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT,
    PHASE4_LIPSCHITZ_PERTURBATION_SCHEME,
    PHASE4_LIPSCHITZ_PERTURBATION_SEED,
    load_phase4_diagnostic_states,
    verify_phase4_diagnostic_inputs,
)
from scripts.phase4_result_schema import (
    PHASE4_METRIC_AVAILABILITY,
    PHASE4_SCHEMA_VERSION,
    Phase4SummaryValidationError,
    phase4_metric_availability,
    validate_phase4_summary,
    write_phase4_summary,
)
from scripts.phase4_source import (
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    PHASE4_FIGURE_SOURCE_PROFILE,
    Phase4SourceError,
    resolve_phase4_path,
    phase4_evaluator_source_manifest_sha256,
    resolve_phase4_source_roots,
    verify_phase4_producer_source,
    verify_phase4_runtime_sources,
)
from scripts.run_phase4_training import run_job
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from rl.replay import Transition
from rl.config import RLConfig
from rl.training_setup import DummyPuzzleDataset
from rl.upi_trm_trainer import UPITrmTrainer
from utils.run_identity import RunIdentityError, canonical_json_sha256
from utils.dataset_provenance import (
    build_dataset_provenance,
    dataset_pool_sha256,
    dataset_puzzle_identifier_sha256s,
    dataset_sample_sha256s,
    ordered_record_sha256,
)
from utils.source_identity import (
    SOURCE_MANIFEST_RELATIVE_PATH,
    build_producer_source_manifest,
)


RETIRED_RUN_FIELDS = {
    "success_trivial",
    "success_hard",
    "final_loss",
    "has_nan",
}
RETIRED_AGGREGATE_FIELDS = {
    "success_trivial_mean",
    "success_trivial_std",
    "success_hard_mean",
    "success_hard_std",
    "nan_count",
}


CONDITION_TOGGLES = {
    "nc_nv": (False, True),
    "nc_yv": (False, False),
    "yc_nv": (True, True),
    "yc_yv": (True, False),
}


def _fake_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_FAKE_TRAINING_RUNTIME_SHA256 = _fake_sha256("training runtime artifact")


def _final_phase4_replay(
    *,
    seed: int,
    rl_config: RLConfig,
) -> list[Transition]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dataset = DummyPuzzleDataset(
            ensure_sudoku_action_support=True,
        )
    environment = _phase4_environment(rl_config, dataset=dataset)
    environment.reset(idx=0)
    template = []
    for timestep in range(PHASE4_EPISODE_LENGTH):
        action_mask = environment.get_action_mask()
        if action_mask is None:
            raise AssertionError("Registered Phase 4 fixture has no action mask.")
        action = torch.tensor(
            2 + timestep % 4,
            dtype=torch.long,
        )
        if not bool(action_mask[int(action.item())].item()):
            raise AssertionError(
                "Registered Phase 4 fixture has no valid edit action."
            )
        x = copy.deepcopy(environment.x)
        y = copy.deepcopy(environment.y)
        (x_next, y_next), reward, done, info = environment.step(
            int(action.item())
        )
        terminal = timestep == PHASE4_EPISODE_LENGTH - 1
        if bool(done) != terminal or info.get("done_reason") != (
            "budget" if terminal else None
        ):
            raise AssertionError(
                "Registered Phase 4 fixture did not reach the exact edit budget."
            )
        template.append(
            Transition(
                x=x,
                y=copy.deepcopy(y),
                action=action.clone(),
                reward=torch.as_tensor(reward, dtype=torch.float32).view(1),
                x_next=copy.deepcopy(x_next),
                y_next=copy.deepcopy(y_next),
                done=torch.tensor([done], dtype=torch.bool),
                episode_id=0,
                timestep=timestep,
                behavior_log_prob=torch.tensor(0.0, dtype=torch.float32),
                terminal_reason="budget" if terminal else None,
            )
        )
    first_episode = PHASE4_FINAL_EPISODES - (
        PHASE4_FINAL_REPLAY_SIZE // PHASE4_EPISODE_LENGTH
    )
    transitions = []
    for episode_id in range(first_episode, PHASE4_FINAL_EPISODES):
        for template_transition in template:
            transition = copy.deepcopy(template_transition)
            transition.episode_id = episode_id
            transitions.append(transition)
    assert len(transitions) == PHASE4_FINAL_REPLAY_SIZE
    return transitions


def _fake_producer_source() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "git_clean": True,
        "source_manifest_sha256": _fake_sha256("producer manifest"),
    }


def _condition_result(
    condition: str = "nc_nv", seed: int = 41
) -> eval_phase4_2x2_norm_ablation.ConditionResult:
    condition_index = eval_phase4_2x2_norm_ablation.CONDITIONS.index(condition)
    offset = condition_index * 0.05 + (seed - 41) * 0.01
    enable_contraction, disable_value_head_norm = CONDITION_TOGGLES[condition]
    return eval_phase4_2x2_norm_ablation.ConditionResult(
        condition=condition,
        seed=seed,
        checkpoint_path=phase4_checkpoint_relpath(condition, seed),
        checkpoint_sha256=_fake_sha256(f"checkpoint:{condition}:{seed}"),
        model_state_sha256=_fake_sha256(f"state:{condition}:{seed}"),
        checkpoint_step=5000,
        training_run_id=phase4_run_id(condition, seed),
        config_sha256=_fake_sha256(f"config:{condition}"),
        rl_config_sha256=_fake_sha256(f"rl:{condition}"),
        model_config_sha256=_fake_sha256(f"model:{condition}"),
        dataset_provenance_sha256=_fake_sha256(f"dataset:{seed}"),
        producer_git_commit=str(_fake_producer_source()["git_commit"]),
        producer_source_manifest_sha256=str(
            _fake_producer_source()["source_manifest_sha256"]
        ),
        training_runtime_artifact_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
        initialization_kind="random",
        checkpoint_schema_version=4,
        training_invocation_schema_version=3,
        enable_contraction=enable_contraction,
        disable_value_head_norm=disable_value_head_norm,
        latent_projection_mode="enabled",
        latent_ball_radius=10.0,
        L_preproj=0.4 + offset,
        L_preproj_std=0.02,
        var_V=0.3 + offset,
        projection_active_rate=0.1 + offset,
        argmax_agreement_n4=0.9 - offset,
        argmax_agreement_n8=0.8 - offset,
        delta_V_n4=0.1 + offset,
        delta_V_n8=0.2 + offset,
        lipschitz_sample_count=150,
        stability_sample_count=100,
        projection_sample_count=50,
    )


def _valid_summary():
    results = [
        _condition_result(condition, seed)
        for condition in eval_phase4_2x2_norm_ablation.CONDITIONS
        for seed in eval_phase4_2x2_norm_ablation.SEEDS
    ]
    aggregates = eval_phase4_2x2_norm_ablation.aggregate_results(results)
    diagnostic_dataset = _fake_diagnostic_identity()
    return {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "metric_availability": phase4_metric_availability(),
        "experiment": "Phase4_2x2_norm_ablation",
        "description": "test summary",
        "generated_at": "2026-08-12T00:00:00",
        "evaluator_git_commit": "b" * 40,
        "evaluator_source_manifest_sha256": "c" * 64,
        "evaluator_runtime_artifact_sha256": "d" * 64,
        "diagnostic_dataset": diagnostic_dataset,
        "diagnostic_dataset_sha256": canonical_json_sha256(
            diagnostic_dataset
        ),
        "lipschitz_perturbation_seed": PHASE4_LIPSCHITZ_PERTURBATION_SEED,
        "lipschitz_perturbation_scheme": (
            PHASE4_LIPSCHITZ_PERTURBATION_SCHEME
        ),
        "conditions": list(eval_phase4_2x2_norm_ablation.CONDITIONS),
        "seeds": list(eval_phase4_2x2_norm_ablation.SEEDS),
        "all_results": [asdict(result) for result in results],
        "aggregates": [asdict(aggregate) for aggregate in aggregates],
    }


def _fake_diagnostic_identity():
    splits = []
    for split, available_records in (("train", 450), ("test", 50)):
        splits.append(
            {
                "name": split,
                "available_records": available_records,
                "selected_records": PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT,
                "inputs": {
                    "relative_path": (
                        f"sudoku-4x4-trivial/{split}/all__inputs.npy"
                    ),
                    "bytes": 100,
                    "sha256": _fake_sha256(f"{split}:inputs"),
                    "dtype": "int32",
                    "shape": [available_records, 16],
                },
                "puzzle_identifiers": {
                    "relative_path": (
                        "sudoku-4x4-trivial/"
                        f"{split}/all__puzzle_identifiers.npy"
                    ),
                    "bytes": 100,
                    "sha256": _fake_sha256(f"{split}:identifiers"),
                    "dtype": "int32",
                    "shape": [available_records],
                },
                "selected_records_sha256": _fake_sha256(
                    f"{split}:selected"
                ),
            }
        )
    return {
        "schema_version": 1,
        "dataset_name": "sudoku-4x4-trivial",
        "selection": "first_n_in_file_order",
        "records_per_split": PHASE4_DIAGNOSTIC_RECORDS_PER_SPLIT,
        "total_selected_records": 100,
        "splits": splits,
        "ordered_states_sha256": _fake_sha256("ordered states"),
    }


def _write_diagnostic_data(root: Path) -> Path:
    data_root = root / "data"
    for split, count in (("train", 55), ("test", 50)):
        split_root = data_root / "sudoku-4x4-trivial" / split
        split_root.mkdir(parents=True, exist_ok=True)
        values = np.arange(count * 16, dtype=np.int32).reshape(count, 16) % 32
        identifiers = np.arange(count, dtype=np.int32) % 32
        np.save(split_root / "all__inputs.npy", values, allow_pickle=False)
        np.save(
            split_root / "all__puzzle_identifiers.npy",
            identifiers,
            allow_pickle=False,
        )
    return data_root


class _IdentityInner:
    def __init__(self) -> None:
        self.contexts = []

    @staticmethod
    def _joint_carry_geometry(z_h, z_l):
        norm = torch.sqrt(
            z_h.pow(2).sum(dim=(1, 2), keepdim=True)
            + z_l.pow(2).sum(dim=(1, 2), keepdim=True)
        )
        safe = torch.where(norm > 0.0, norm, torch.ones_like(norm))
        return norm, z_h / safe, z_l / safe

    def latent_step_pre_projection(self, carry, input_embeddings, seq_info):
        self.contexts.append(input_embeddings.detach().clone())
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=carry.z_H,
            z_L=carry.z_L,
        )

    def latent_step(self, *args, **kwargs):
        raise AssertionError("L_preproj must not call the projected latent step")


class _IdentityLipschitzModel:
    def __init__(self) -> None:
        self.inner = _IdentityInner()
        self._carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.tensor([[[0.25, -0.5], [0.1, 0.2]]]),
            z_L=torch.tensor([[[-0.3, 0.4], [0.2, -0.1]]]),
        )

    def eval(self):
        return self

    def used_value(self, x, y, n):
        return torch.zeros(1), self._carry

    def _standardize_latent_batch(self, x, y):
        return {**x, "plan": y}

    def _resolve_latent_context(self, batch):
        inputs = batch["inputs"].float().unsqueeze(-1).expand(-1, -1, 2)
        plan = batch["plan"].float().unsqueeze(-1).expand(-1, -1, 2)
        return {
            "input_embeddings_with_plan": inputs + plan,
            "seq_info": {},
        }


class _MaskedPolicySpy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(rl_num_actions=513, vocab_size=32)
        self.calls = []

    def eval(self):
        return self

    def used_value(self, x, y, n):
        return torch.tensor([float(n)]), None

    def policy_dist(self, x, y, n, z=None, action_mask=None):
        assert action_mask is not None
        self.calls.append((n, z, action_mask.detach().clone()))
        logits = torch.full((1, 513), -100.0)
        logits[0, 512] = 100.0
        logits[0, 35 if n == 4 else 36] = 10.0
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits), None


def _write_synthetic_phase4_checkpoint(
    root: Path,
    *,
    condition: str = "nc_nv",
    seed: int = 41,
    mutate=None,
):
    config_dir = root / "configs"
    checkpoint_root = root / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / phase4_checkpoint_relpath(condition, seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"{condition}.yaml"
    import yaml

    yaml_config = _registered_phase4_config_layer(condition)
    config_path.write_text(yaml.safe_dump(yaml_config, sort_keys=True))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rl_config = _expected_rl_config(yaml_config, condition)
    model_config = _expected_model_config(rl_config)
    model = TinyRecursiveReasoningModel_ACTV1(model_config)
    rl_config_object = RLConfig(**rl_config)
    trainer = UPITrmTrainer(
        model=model,
        env=_phase4_environment(rl_config_object),
        rl_cfg=rl_config_object,
        device=torch.device("cpu"),
    )
    model_config = model.config.model_dump()
    replay_transitions = _final_phase4_replay(
        seed=seed,
        rl_config=rl_config_object,
    )
    replay_tail = replay_transitions[-1]
    trainer.replay.add(replay_tail)
    trainer._next_episode_id = PHASE4_FINAL_EPISODES
    trainer._train_step_count = 5000
    trainer._env_step_count = PHASE4_FINAL_ENV_STEPS
    trainer._value_optimizer_step_count = 5000
    trainer._policy_optimizer_step_count = 5000
    trainer._distill_optimizer_step_count = 0
    trainer.env.x = copy.deepcopy(replay_tail.x_next)
    trainer.env.y = cast(torch.Tensor, replay_tail.y_next).clone()
    trainer.env.step_count = PHASE4_EPISODE_LENGTH
    trainer.env.done = True
    trainer.env._cached_phi = 0.0
    trainer.env._action_mask = None
    trainer.env._original_inputs = replay_tail.x_next["inputs"].clone()
    trainer.env._stop_penalty = 0.0
    trainer.env._edit_history = []
    value_unused_parameters = {
        parameter
        for name, parameter in trainer.model.named_parameters()
        if name
        in {
            "inner.lm_head.weight",
            "inner.q_head.weight",
            "inner.q_head.bias",
        }
    }
    for optimizer in (
        trainer.value_opt,
        trainer.policy_opt,
    ):
        assert optimizer is not None
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = (
                    None
                    if optimizer is trainer.value_opt
                    and parameter in value_unused_parameters
                    else torch.zeros_like(parameter)
                )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for state in optimizer.state.values():
            state["step"] = torch.tensor(5000.0)
    for scheduler in (trainer.value_scheduler, trainer.policy_scheduler):
        assert scheduler is not None
        for _ in range(5000):
            scheduler.step()
    model_state = model.state_dict()
    compute_accounting_state = trainer.compute_accounting_checkpoint_state()
    compute_accounting_state["peak_cuda_allocated_bytes"] = 0
    compute_accounting_state["peak_cuda_reserved_bytes"] = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        registered_dataset = DummyPuzzleDataset(
            ensure_sudoku_action_support=True,
        )
    records = dataset_sample_sha256s(registered_dataset)
    pool_sha256 = dataset_pool_sha256(
        registered_dataset,
        len(registered_dataset),
    )
    identifier_sha256 = ordered_record_sha256(
        dataset_puzzle_identifier_sha256s(registered_dataset)
    )
    dataset_provenance = build_dataset_provenance(
        builder_name="rl.training_setup.DummyPuzzleDataset",
        builder_version=2,
        generation_seed=seed,
        train_record_sha256s=records,
        eval_record_sha256s=records,
        train_split="dummy",
        eval_split="dummy",
        environment_config={
            "max_edits": rl_config["max_edits"],
            "gamma": rl_config["gamma"],
            "reward_shaping": rl_config["reward_shaping"],
            "task_type": "sudoku",
            "vocab_size": 32,
            "solved_threshold": rl_config["solved_threshold"],
            "stop_action_mode": rl_config["stop_action_mode"],
            "stop_action_penalty": rl_config["stop_action_penalty"],
            "C_max": rl_config["C_max"],
            "enable_undo": False,
            "fail_terminal_reward": rl_config["fail_terminal_reward"],
            "solve_terminal_reward": rl_config["solve_terminal_reward"],
            "disable_constraint_masking": rl_config[
                "disable_constraint_masking"
            ],
        },
        action_mask_config={
            "task_config_class": "SudokuTaskConfig",
            "task_config_name": "sudoku",
            "disable_constraint_masking": rl_config[
                "disable_constraint_masking"
            ],
            "stop_action_mode": rl_config["stop_action_mode"],
            "stop_action_id": 512,
            "enable_undo": False,
            "undo_action_id": None,
            "vocab_size": 32,
            "num_actions": 513,
            "masked_token_ids": [0, 1],
        },
        metadata={
            "train_pool_sha256": pool_sha256,
            "eval_pool_sha256": pool_sha256,
            "seq_len": 16,
            "vocab_size": 32,
            "num_identifiers": 32,
            "eval_puzzle_id_offset": 0,
            "dataset_source_names": [],
            "train_puzzle_identifier_ordered_sha256": identifier_sha256,
            "eval_puzzle_identifier_ordered_sha256": identifier_sha256,
        },
    )
    checkpoint = {
        "checkpoint_schema_version": 4,
        "training_protocol": "legacy",
        "trainer_kind": "UPITrmTrainer",
        "step": 5000,
        "execution_device": "cuda:0",
        "progress": {
            "env_steps": PHASE4_FINAL_ENV_STEPS,
            "optimizer_updates": 5000,
            "optimizer_steps": 5000,
        },
        "model_state_dict": dict(model_state),
        "policy_model_old_state_dict": dict(
            trainer.policy_model_old.state_dict()
        ),
        "policy_model_candidate_state_dict": dict(
            trainer.policy_model_candidate.state_dict()
        ),
        "target_model_state_dict": dict(trainer.target_model.state_dict()),
        "value_optimizer_state_dict": trainer.value_opt.state_dict(),
        "policy_optimizer_state_dict": trainer.policy_opt.state_dict(),
        "old_policy_distill_optimizer_state_dict": (
            cast(
                torch.optim.Optimizer,
                trainer.old_policy_distill_opt,
            ).state_dict()
        ),
        "value_scheduler_state_dict": cast(
            torch.optim.lr_scheduler.LambdaLR,
            trainer.value_scheduler,
        ).state_dict(),
        "policy_scheduler_state_dict": cast(
            torch.optim.lr_scheduler.LambdaLR,
            trainer.policy_scheduler,
        ).state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": [torch.zeros(16, dtype=torch.uint8)],
        },
        "trainer_state": {
            "next_episode_id": PHASE4_FINAL_EPISODES,
            "train_step_count": 5000,
            "env_step_count": PHASE4_FINAL_ENV_STEPS,
            "optimizer_step_count": 5000,
            "value_optimizer_step_count": 5000,
            "policy_optimizer_step_count": 5000,
            "distill_optimizer_step_count": 0,
            "puzzle_optimizer_step_count": 0,
            "kl_coef": trainer._kl_coef,
            "term_stats": dict(trainer.term_stats),
            "debug_episode_lengths": [],
            "debug_episode_returns": [],
            "debug_stop_probs": [],
            "debug_score_changes": [],
            "drift_values": [],
            "plan_changes": [],
            "value_of_memory": [],
            "opnorm_clamp_warned": False,
            "collection_state": trainer.collection_checkpoint_state(),
            "environment_state": trainer.env.checkpoint_state(),
            "exact_centering_state": trainer.exact_centering_checkpoint_state(),
            "compute_accounting_state": compute_accounting_state,
            "terminal_reason_replay_version": 1,
        },
        "replay_transitions": replay_transitions,
        "replay_buffer_size": PHASE4_FINAL_REPLAY_SIZE,
        "replay_capacity": PHASE4_FINAL_REPLAY_SIZE,
        "model_config": model_config,
        "rl_config": rl_config,
        "dataset_provenance": dataset_provenance,
        "training_invocation": {
            "schema_version": 3,
            "training_seed": seed,
            "run_id": phase4_run_id(condition, seed),
            "runtime_artifact_sha256": _FAKE_TRAINING_RUNTIME_SHA256,
            "config_sources": [
                {"name": config_path.name, "sha256": config_sha256}
            ],
            "rl_config_sha256": canonical_json_sha256(rl_config),
            "model_config_sha256": canonical_json_sha256(model_config),
            "dataset_provenance_sha256": canonical_json_sha256(
                dataset_provenance
            ),
            "initialization": {
                "kind": "random",
                "artifact_sha256": None,
            },
            "producer_source": _fake_producer_source(),
        },
    }
    if mutate is not None:
        mutate(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_root, config_dir, checkpoint_path, config_path


class Phase4ReportingTest(unittest.TestCase):
    def test_registered_final_training_design_is_frozen(self) -> None:
        self.assertEqual(PHASE4_FINAL_ENV_STEPS, 320_000)
        self.assertEqual(PHASE4_FINAL_EPISODES, 20_000)
        self.assertEqual(PHASE4_FINAL_REPLAY_SIZE, 100_000)
        self.assertEqual(PHASE4_EPISODE_LENGTH, 16)

    def test_schema_v4_has_exact_availability_and_no_retired_fields(self) -> None:
        summary = _valid_summary()

        validate_phase4_summary(summary)

        self.assertEqual(summary["metric_availability"], PHASE4_METRIC_AVAILABILITY)
        self.assertFalse(RETIRED_RUN_FIELDS.intersection(summary["all_results"][0]))
        self.assertFalse(
            RETIRED_AGGREGATE_FIELDS.intersection(summary["aggregates"][0])
        )
        self.assertTrue(
            audit_phase4_paper_ready.check_publication_schema(summary).passed
        )

    def test_legacy_placeholder_and_null_fields_are_rejected(self) -> None:
        for field, value in (
            ("success_trivial", 0.0),
            ("success_hard", None),
            ("final_loss", 0.0),
            ("has_nan", False),
        ):
            with self.subTest(field=field, value=value):
                summary = _valid_summary()
                summary["all_results"][0][field] = value
                with self.assertRaises(Phase4SummaryValidationError):
                    validate_phase4_summary(summary)

        summary = _valid_summary()
        summary["aggregates"][0]["success_trivial_mean"] = None
        with self.assertRaises(Phase4SummaryValidationError):
            validate_phase4_summary(summary)

    def test_changed_availability_metadata_is_rejected(self) -> None:
        summary = _valid_summary()
        summary["metric_availability"]["final_loss"]["reason"] = "unknown"

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "metric_availability"
        ):
            validate_phase4_summary(summary)

    def test_aggregate_seed_count_must_match_measured_runs(self) -> None:
        summary = _valid_summary()
        summary["aggregates"][0]["n_seeds"] = 2

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "n_seeds must be 3"
        ):
            validate_phase4_summary(summary)

    def test_partial_design_is_not_publishable(self) -> None:
        summary = _valid_summary()
        summary["all_results"].pop()

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "complete four-condition"
        ):
            validate_phase4_summary(summary)

    def test_mixed_training_runtime_artifacts_are_rejected(self) -> None:
        summary = _valid_summary()
        summary["all_results"][-1]["training_runtime_artifact_sha256"] = (
            _fake_sha256("different training runtime artifact")
        )

        with self.assertRaisesRegex(
            Phase4SummaryValidationError,
            "share one training runtime artifact SHA-256",
        ):
            validate_phase4_summary(summary)

    def test_aggregate_values_are_recomputed_from_runs(self) -> None:
        summary = _valid_summary()
        summary["aggregates"][0]["var_V_mean"] += 0.01

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "recomputed from all_results"
        ):
            validate_phase4_summary(summary)

    def test_condition_toggles_are_bound_by_schema(self) -> None:
        summary = _valid_summary()
        summary["all_results"][0]["enable_contraction"] = True

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "does not match condition"
        ):
            validate_phase4_summary(summary)

    def test_aggregation_exposes_only_measured_metrics(self) -> None:
        aggregate = eval_phase4_2x2_norm_ablation.aggregate_results(
            [_condition_result()]
        )[0]

        self.assertFalse(RETIRED_RUN_FIELDS.intersection(asdict(_condition_result())))
        self.assertFalse(RETIRED_AGGREGATE_FIELDS.intersection(asdict(aggregate)))

    def test_claims_and_latex_do_not_publish_unmeasured_values(self) -> None:
        summary = _valid_summary()
        aggregates = eval_phase4_2x2_norm_ablation.aggregate_results(
            [
                _condition_result(condition, seed)
                for condition in eval_phase4_2x2_norm_ablation.CONDITIONS
                for seed in eval_phase4_2x2_norm_ablation.SEEDS
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            eval_phase4_2x2_norm_ablation.generate_claims_md(
                aggregates, output_dir
            )
            make_paper_figures_phase4.generate_latex_table(summary, output_dir)

            claims = (output_dir / "CLAIMS.md").read_text()
            latex = (
                output_dir / "table_phase4_2x2_norm_ablation.tex"
            ).read_text()

        self.assertNotIn("| Success |", claims)
        self.assertNotIn("NaN", claims)
        self.assertNotIn("expected to show", claims)
        for metric, metadata in PHASE4_METRIC_AVAILABILITY.items():
            self.assertIn(f"`{metric}`", claims)
            self.assertIn(f"`{metadata['reason']}`", claims)
        self.assertNotIn("Success", latex)
        self.assertNotIn("success_trivial", latex)

    def test_invalid_summary_fails_before_figure_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            summary_path = temp_path / "legacy_summary.json"
            output_path = temp_path / "figures"
            legacy_summary = _valid_summary()
            del legacy_summary["schema_version"]
            summary_path.write_text(json.dumps(legacy_summary))

            with mock.patch.object(
                sys,
                "argv",
                [
                    "make_paper_figures_phase4",
                    "--project_root",
                    str(temp_path),
                    "--fbcode_root",
                    str(temp_path),
                    "--producer_project_root",
                    str(temp_path),
                    "--expected_producer_git_commit",
                    "a" * 40,
                    "--expected_evaluator_runtime_sha256",
                    "d" * 64,
                    "--expected_training_runtime_sha256",
                    _FAKE_TRAINING_RUNTIME_SHA256,
                    "--evaluator_runtime_archive",
                    str(temp_path / "evaluator.par"),
                    "--training_runtime_archive",
                    str(temp_path / "training.par"),
                    "--summary_json",
                    str(summary_path),
                    "--checkpoint_dir",
                    str(temp_path / "checkpoints"),
                    "--data_dir",
                    str(temp_path / "data"),
                    "--out_dir",
                    str(output_path),
                ],
            ), mock.patch.object(
                make_paper_figures_phase4,
                "resolve_phase4_source_roots",
                return_value=(temp_path, temp_path),
            ), mock.patch.object(
                make_paper_figures_phase4,
                "verify_phase4_runtime_sources",
                return_value="d" * 64,
            ), mock.patch.object(
                make_paper_figures_phase4,
                "verify_runtime_archive_sha256",
                side_effect=lambda _path, digest, _label: digest,
            ), mock.patch.object(
                make_paper_figures_phase4,
                "verify_phase4_producer_source",
                return_value=_fake_producer_source(),
            ), mock.patch.object(
                make_paper_figures_phase4,
                "discover_clean_git_source",
                return_value={"git_commit": "a" * 40, "git_clean": True},
            ):
                self.assertEqual(
                    make_paper_figures_phase4.main(
                        runtime_attestation={
                            "runtime_sha256": "f" * 64,
                            "role": PHASE4_FIGURE_SOURCE_PROFILE,
                            "source_git_commit": "a" * 40,
                            "source_manifest_sha256": "d" * 64,
                        }
                    ),
                    1,
                )

            self.assertFalse(output_path.exists())

    def test_source_change_discards_staged_figure_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            summary_path = temp_path / "summary.json"
            summary_path.write_text(json.dumps(_valid_summary()))
            output_path = temp_path / "figures"

            def write_plot_outputs(_summary, staging_path: Path) -> None:
                for name in (
                    "fig_phase4_2x2_norm_ablation.pdf",
                    "fig_phase4_2x2_norm_ablation.png",
                ):
                    (staging_path / name).write_bytes(b"plot")

            def write_bar_output(_summary, staging_path: Path) -> None:
                (staging_path / "fig_phase4_bar_comparison.pdf").write_bytes(
                    b"bar"
                )

            def write_table_output(_summary, staging_path: Path) -> None:
                (
                    staging_path / "table_phase4_2x2_norm_ablation.tex"
                ).write_text("table")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "make_paper_figures_phase4",
                    "--project_root",
                    str(temp_path),
                    "--fbcode_root",
                    str(temp_path),
                    "--producer_project_root",
                    str(temp_path),
                    "--expected_producer_git_commit",
                    "a" * 40,
                    "--expected_evaluator_runtime_sha256",
                    "d" * 64,
                    "--expected_training_runtime_sha256",
                    _FAKE_TRAINING_RUNTIME_SHA256,
                    "--evaluator_runtime_archive",
                    str(temp_path / "evaluator.par"),
                    "--training_runtime_archive",
                    str(temp_path / "training.par"),
                    "--summary_json",
                    str(summary_path),
                    "--checkpoint_dir",
                    str(temp_path / "checkpoints"),
                    "--data_dir",
                    str(temp_path / "data"),
                    "--out_dir",
                    str(output_path),
                ],
            ), mock.patch.object(
                make_paper_figures_phase4,
                "resolve_phase4_source_roots",
                return_value=(temp_path, temp_path),
            ), mock.patch.object(
                make_paper_figures_phase4,
                "verify_phase4_runtime_sources",
                side_effect=["d" * 64, "e" * 64],
            ), mock.patch.object(
                make_paper_figures_phase4,
                "verify_runtime_archive_sha256",
                side_effect=lambda _path, digest, _label: digest,
            ), mock.patch.object(
                make_paper_figures_phase4,
                "verify_phase4_producer_source",
                return_value=_fake_producer_source(),
            ), mock.patch.object(
                make_paper_figures_phase4,
                "discover_clean_git_source",
                return_value={"git_commit": "b" * 40, "git_clean": True},
            ), mock.patch.object(
                make_paper_figures_phase4,
                "load_summary",
                return_value=(
                    _valid_summary(),
                    _fake_sha256("summary bytes"),
                ),
            ), mock.patch.object(
                make_paper_figures_phase4,
                "generate_2x2_plot",
                side_effect=write_plot_outputs,
            ), mock.patch.object(
                make_paper_figures_phase4,
                "generate_bar_comparison",
                side_effect=write_bar_output,
            ), mock.patch.object(
                make_paper_figures_phase4,
                "generate_latex_table",
                side_effect=write_table_output,
            ):
                self.assertEqual(
                    make_paper_figures_phase4.main(
                        runtime_attestation={
                            "runtime_sha256": "f" * 64,
                            "role": PHASE4_FIGURE_SOURCE_PROFILE,
                            "source_git_commit": "b" * 40,
                            "source_manifest_sha256": "d" * 64,
                        }
                    ),
                    1,
                )

            self.assertTrue(output_path.is_dir())
            self.assertFalse((output_path / "CURRENT.json").exists())
            self.assertEqual(
                list((output_path / "generations").iterdir()),
                [],
            )

    def test_nonfinite_measurement_is_rejected_before_serialization(self) -> None:
        summary = _valid_summary()
        summary["all_results"][0]["var_V"] = float("nan")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"
            with self.assertRaisesRegex(
                Phase4SummaryValidationError, "must be finite"
            ):
                write_phase4_summary(summary, output_path)
            self.assertFalse(output_path.exists())

    def test_schema_less_summary_is_historical_not_publishable(self) -> None:
        summary = copy.deepcopy(_valid_summary())
        del summary["schema_version"]

        result = audit_phase4_paper_ready.check_publication_schema(summary)

        self.assertFalse(result.passed)
        self.assertIn("historical", result.message.lower())
        self.assertIn("non-publishable", result.message.lower())

    def test_historical_audit_does_not_overwrite_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "AUDIT.md"
            audit_path.write_text("historical artifact\n")
            results = [
                audit_phase4_paper_ready.AuditResult(
                    "Publication Schema",
                    False,
                    "legacy summary",
                )
            ]

            audit_phase4_paper_ready.write_audit_md(
                Path(temp_dir), results, all_passed=False
            )

            self.assertEqual(audit_path.read_text(), "historical artifact\n")


class Phase4MetricKernelTest(unittest.TestCase):
    def test_lipschitz_uses_preprojection_map_and_joint_norm(self) -> None:
        model = _IdentityLipschitzModel()
        states = [
            {
                "inputs": [1, 2],
                "plan": [3, 4],
                "puzzle_identifier": 7,
            }
        ]

        mean, std, count = eval_phase4_2x2_norm_ablation.compute_lipschitz(
            model,
            states,
            n_steps=2,
            device="cpu",
        )

        self.assertAlmostEqual(mean, 1.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)
        self.assertEqual(count, 3)
        expected_context = torch.tensor(
            [[[4.0, 4.0], [6.0, 6.0]]]
        )
        self.assertTrue(
            all(torch.equal(context, expected_context) for context in model.inner.contexts)
        )

    def test_lipschitz_empty_sample_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            eval_phase4_2x2_norm_ablation.compute_lipschitz(
                _IdentityLipschitzModel(),
                [],
                device="cpu",
            )

    def test_lipschitz_uses_private_repeatable_rng(self) -> None:
        state = {
            "inputs": [1, 2],
            "plan": [3, 4],
            "puzzle_identifier": 7,
        }
        torch.manual_seed(12345)
        global_state = torch.random.get_rng_state().clone()

        first = eval_phase4_2x2_norm_ablation.compute_lipschitz(
            _IdentityLipschitzModel(),
            [state],
            device="cpu",
        )
        after_first = torch.random.get_rng_state().clone()
        second = eval_phase4_2x2_norm_ablation.compute_lipschitz(
            _IdentityLipschitzModel(),
            [state],
            device="cpu",
        )

        self.assertEqual(first, second)
        self.assertTrue(torch.equal(global_state, after_first))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_policy_stability_uses_masked_production_api(self) -> None:
        model = _MaskedPolicySpy()
        state = {
            "inputs": [2] + [1] * 15,
            "plan": [2, 2] + [1] * 14,
            "puzzle_identifier": 3,
        }
        result = eval_phase4_2x2_norm_ablation.compute_stability_metrics(
            model,
            [state],
            {
                "task_name": "sudoku",
                "disable_constraint_masking": False,
                "stop_action_mode": "disabled",
            },
            n_train=2,
            device="cpu",
        )

        self.assertEqual([call[0] for call in model.calls], [2, 4, 8])
        self.assertTrue(all(call[1] is None for call in model.calls))
        masks = [call[2] for call in model.calls]
        self.assertTrue(all(tuple(mask.shape) == (1, 513) for mask in masks))
        self.assertTrue(all(not bool(mask[0, 512]) for mask in masks))
        self.assertTrue(all(torch.equal(mask, masks[0]) for mask in masks[1:]))
        self.assertFalse(bool(masks[0][0, 3]))
        self.assertFalse(bool(masks[0][0, 32]))
        self.assertFalse(bool(masks[0][0, 33]))
        self.assertFalse(bool(masks[0][0, 34]))
        self.assertTrue(bool(masks[0][0, 35]))
        self.assertEqual(result["argmax_n4"], 0.0)
        self.assertEqual(result["argmax_n8"], 1.0)
        self.assertEqual(result["sample_count"], 1)

    def test_policy_stability_empty_sample_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            eval_phase4_2x2_norm_ablation.compute_stability_metrics(
                _MaskedPolicySpy(),
                [],
                {
                    "task_name": "sudoku",
                    "stop_action_mode": "disabled",
                },
                device="cpu",
            )


class Phase4DiagnosticInputTest(unittest.TestCase):
    def test_exact_input_bytes_and_order_are_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = _write_diagnostic_data(Path(temp_dir))
            states, identity = load_phase4_diagnostic_states(data_root)
            summary = _valid_summary()
            summary["diagnostic_dataset"] = identity
            summary["diagnostic_dataset_sha256"] = canonical_json_sha256(identity)

            self.assertEqual(len(states), 100)
            validate_phase4_summary(summary)
            self.assertEqual(
                verify_phase4_diagnostic_inputs(summary, data_root),
                100,
            )

            inputs_path = (
                data_root
                / "sudoku-4x4-trivial"
                / "test"
                / "all__inputs.npy"
            )
            inputs = np.load(inputs_path, allow_pickle=False)
            inputs[0, 0] = (inputs[0, 0] + 1) % 32
            np.save(inputs_path, inputs, allow_pickle=False)
            with self.assertRaisesRegex(
                Phase4DiagnosticInputError,
                "does not match",
            ):
                verify_phase4_diagnostic_inputs(summary, data_root)

    def test_short_diagnostic_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = _write_diagnostic_data(Path(temp_dir))
            test_root = data_root / "sudoku-4x4-trivial" / "test"
            np.save(
                test_root / "all__inputs.npy",
                np.ones((49, 16), dtype=np.int32),
                allow_pickle=False,
            )
            np.save(
                test_root / "all__puzzle_identifiers.npy",
                np.arange(49, dtype=np.int32) % 32,
                allow_pickle=False,
            )
            with self.assertRaisesRegex(
                Phase4DiagnosticInputError,
                "at least 50",
            ):
                load_phase4_diagnostic_states(data_root)


class Phase4CheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        # The public loader enforces the 20k-episode/100k-replay constants
        # asserted above. Scale only loop bounds in per-mutation unit fixtures.
        global PHASE4_FINAL_ENV_STEPS
        global PHASE4_FINAL_EPISODES
        global PHASE4_FINAL_REPLAY_SIZE
        self._production_design = (
            PHASE4_FINAL_ENV_STEPS,
            PHASE4_FINAL_EPISODES,
            PHASE4_FINAL_REPLAY_SIZE,
        )
        self._production_expected_rl_config = _expected_rl_config

        def compact_expected_rl_config(yaml_config, condition):
            resolved = self._production_expected_rl_config(
                yaml_config,
                condition,
            )
            return {**resolved, "replay_capacity": PHASE4_EPISODE_LENGTH}

        globals()["_expected_rl_config"] = compact_expected_rl_config
        phase4_checkpoint_module._expected_rl_config = (
            compact_expected_rl_config
        )
        PHASE4_FINAL_ENV_STEPS = PHASE4_EPISODE_LENGTH
        PHASE4_FINAL_EPISODES = 1
        PHASE4_FINAL_REPLAY_SIZE = PHASE4_EPISODE_LENGTH
        phase4_checkpoint_module.PHASE4_FINAL_ENV_STEPS = PHASE4_EPISODE_LENGTH
        phase4_checkpoint_module.PHASE4_FINAL_EPISODES = 1
        phase4_checkpoint_module.PHASE4_FINAL_REPLAY_SIZE = PHASE4_EPISODE_LENGTH
        self.addCleanup(self._restore_production_design)

    def _restore_production_design(self) -> None:
        global PHASE4_FINAL_ENV_STEPS
        global PHASE4_FINAL_EPISODES
        global PHASE4_FINAL_REPLAY_SIZE
        (
            PHASE4_FINAL_ENV_STEPS,
            PHASE4_FINAL_EPISODES,
            PHASE4_FINAL_REPLAY_SIZE,
        ) = self._production_design
        phase4_checkpoint_module.PHASE4_FINAL_ENV_STEPS = (
            PHASE4_FINAL_ENV_STEPS
        )
        phase4_checkpoint_module.PHASE4_FINAL_EPISODES = PHASE4_FINAL_EPISODES
        phase4_checkpoint_module.PHASE4_FINAL_REPLAY_SIZE = (
            PHASE4_FINAL_REPLAY_SIZE
        )
        globals()["_expected_rl_config"] = self._production_expected_rl_config
        phase4_checkpoint_module._expected_rl_config = (
            self._production_expected_rl_config
        )

    def test_registered_dummy_data_always_has_policy_support(self) -> None:
        for seed in (41, 42, 43):
            with self.subTest(seed=seed), torch.random.fork_rng():
                torch.manual_seed(seed)
                dataset = DummyPuzzleDataset(
                    ensure_sudoku_action_support=True,
                )
                rl_config = RLConfig(
                    **_expected_rl_config(
                        _registered_phase4_config_layer("nc_nv"),
                        "nc_nv",
                    )
                )
                environment = _phase4_environment(rl_config)
                environment.dataset = dataset
                for index in range(len(dataset)):
                    environment.reset(index)
                    action_mask = environment.get_action_mask()
                    self.assertIsNotNone(action_mask)
                    assert action_mask is not None
                    self.assertTrue(bool(action_mask.any().item()))

    def test_full_checkpoint_is_strictly_bound_and_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root, config_dir, checkpoint_path, config_path = (
                _write_synthetic_phase4_checkpoint(Path(temp_dir))
            )
            loaded = load_phase4_checkpoint(
                checkpoint_path,
                config_path,
                condition="nc_nv",
                seed=41,
                expected_producer_source=_fake_producer_source(),
                expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                device="cpu",
            )
            run = {
                "condition": "nc_nv",
                "seed": 41,
                "checkpoint_path": phase4_checkpoint_relpath("nc_nv", 41),
                **loaded.identity.__dict__,
            }

            self.assertEqual(
                verify_phase4_summary_checkpoints(
                    {"all_results": [run]},
                    checkpoint_root,
                    config_dir,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                ),
                1,
            )

    def test_fabricated_producer_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = (
                _write_synthetic_phase4_checkpoint(Path(temp_dir))
            )
            for field, fabricated_value in (
                ("git_commit", "b" * 40),
                (
                    "source_manifest_sha256",
                    _fake_sha256("fabricated producer manifest"),
                ),
            ):
                with self.subTest(field=field):
                    fabricated_source = _fake_producer_source()
                    fabricated_source[field] = fabricated_value
                    with self.assertRaisesRegex(
                        Phase4CheckpointError,
                        "authorized checkout",
                    ):
                        load_phase4_checkpoint(
                            checkpoint_path,
                            config_path,
                            condition="nc_nv",
                            seed=41,
                            expected_producer_source=fabricated_source,
                            expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                            device="cpu",
                        )

    def test_self_asserted_training_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = (
                _write_synthetic_phase4_checkpoint(Path(temp_dir))
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "externally authorized digest",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_fake_sha256(
                        "unauthorized training runtime"
                    ),
                    device="cpu",
                )

    def test_model_only_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root, _, checkpoint_path, config_path = (
                _write_synthetic_phase4_checkpoint(Path(temp_dir))
            )
            payload = torch.load(checkpoint_path, weights_only=False)
            torch.save(payload["model_state_dict"], checkpoint_path)

            with self.assertRaisesRegex(Phase4CheckpointError, "missing required"):
                load_phase4_checkpoint(
                    checkpoint_root / phase4_checkpoint_relpath("nc_nv", 41),
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_incomplete_training_checkpoint_is_rejected(self) -> None:
        def remove_target_state(checkpoint):
            del checkpoint["target_model_state_dict"]

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=remove_target_state,
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "missing required"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_truncated_resume_state_is_rejected(self) -> None:
        mutations = {
            "empty collector": lambda checkpoint: checkpoint["trainer_state"].__setitem__(
                "collection_state", {}
            ),
            "empty environment": lambda checkpoint: checkpoint["trainer_state"].__setitem__(
                "environment_state", {}
            ),
            "missing next episode": lambda checkpoint: checkpoint["trainer_state"].pop(
                "next_episode_id"
            ),
            "missing value scheduler": lambda checkpoint: checkpoint.pop(
                "value_scheduler_state_dict"
            ),
            "missing policy scheduler": lambda checkpoint: checkpoint.pop(
                "policy_scheduler_state_dict"
            ),
            "empty compute accounting": lambda checkpoint: checkpoint[
                "trainer_state"
            ].__setitem__("compute_accounting_state", {}),
            "malformed CUDA RNG": lambda checkpoint: checkpoint[
                "rng_state"
            ].__setitem__(
                "torch_cuda", [torch.tensor([1], dtype=torch.uint8)]
            ),
            "misaligned CUDA RNG offset": lambda checkpoint: checkpoint[
                "rng_state"
            ].__setitem__(
                "torch_cuda",
                [
                    torch.tensor(
                        [0] * 8 + [1] + [0] * 7,
                        dtype=torch.uint8,
                    )
                ],
            ),
            "noncanonical execution device": lambda checkpoint: checkpoint.__setitem__(
                "execution_device", "definitely-not-a-device"
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                _, _, checkpoint_path, config_path = (
                    _write_synthetic_phase4_checkpoint(
                        Path(temp_dir),
                        mutate=mutation,
                    )
                )
                with self.assertRaises(Phase4CheckpointError):
                    load_phase4_checkpoint(
                        checkpoint_path,
                        config_path,
                        condition="nc_nv",
                        seed=41,
                        expected_producer_source=_fake_producer_source(),
                        expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                        device="cpu",
                    )

    def test_optimizer_must_cover_the_registered_parameter_groups(self) -> None:
        def truncate_optimizer(checkpoint):
            checkpoint["value_optimizer_state_dict"] = {
                "state": {0: {"step": torch.tensor(1.0)}},
                "param_groups": [{"params": [0]}],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=truncate_optimizer,
            )
            with self.assertRaises(Phase4CheckpointError):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_optimizer_parameter_order_is_bound(self) -> None:
        def swap_optimizer_parameters(checkpoint):
            parameters = checkpoint["policy_optimizer_state_dict"][
                "param_groups"
            ][0]["params"]
            parameters[0], parameters[1] = parameters[1], parameters[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=swap_optimizer_parameters,
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "parameter order",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_replay_payload_supports_registered_policy_update(self) -> None:
        def use_masked_clue_action(checkpoint):
            transition = checkpoint["replay_transitions"][0]
            clue_positions = torch.nonzero(
                transition.x["inputs"] > 1,
                as_tuple=False,
            ).reshape(-1)
            transition.action = torch.tensor(
                int(clue_positions[0].item()) * 32 + 2,
                dtype=torch.long,
            )

        def change_successor(checkpoint):
            transitions = checkpoint["replay_transitions"]
            transition = transitions[0]
            successor = cast(torch.Tensor, transition.y_next).clone()
            successor[0] = (int(successor[0].item()) + 1) % 32
            transition.y_next = successor
            transitions[1].y = successor.clone()

        def change_successor_clock(checkpoint):
            transitions = checkpoint["replay_transitions"]
            successor = copy.deepcopy(transitions[0].x_next)
            successor["remaining_edits"] = (
                successor["remaining_edits"] + 1
            )
            transitions[0].x_next = successor
            transitions[1].x = copy.deepcopy(successor)

        def change_reward(checkpoint):
            transition = checkpoint["replay_transitions"][0]
            transition.reward = transition.reward + 1.0

        def change_terminal_flag(checkpoint):
            transition = checkpoint["replay_transitions"][-1]
            transition.done = torch.tensor([False], dtype=torch.bool)
            transition.terminal_reason = None

        def change_terminal_reason(checkpoint):
            checkpoint["replay_transitions"][-1].terminal_reason = "stop"

        mutations = {
            "missing behavior log-probability": (
                lambda checkpoint: setattr(
                    checkpoint["replay_transitions"][0],
                    "behavior_log_prob",
                    None,
                ),
                "behavior log-probability",
            ),
            "disabled STOP action": (
                lambda checkpoint: setattr(
                    checkpoint["replay_transitions"][0],
                    "action",
                    torch.tensor(512, dtype=torch.long),
                ),
                "registered scalar action",
            ),
            "masked clue action": (
                use_masked_clue_action,
                "action is masked",
            ),
            "mismatched successor": (
                change_successor,
                "successor disagrees",
            ),
            "mismatched successor clock": (
                change_successor_clock,
                "successor disagrees",
            ),
            "mismatched reward": (
                change_reward,
                "reward disagrees",
            ),
            "mismatched terminal flag": (
                change_terminal_flag,
                "terminal flag disagrees",
            ),
            "mismatched terminal reason": (
                change_terminal_reason,
                "terminal reason disagrees",
            ),
        }
        for label, (mutation, expected_error) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                _, _, checkpoint_path, config_path = (
                    _write_synthetic_phase4_checkpoint(
                        Path(temp_dir),
                        mutate=mutation,
                    )
                )
                with self.assertRaisesRegex(
                    Phase4CheckpointError, expected_error
                ):
                    load_phase4_checkpoint(
                        checkpoint_path,
                        config_path,
                        condition="nc_nv",
                        seed=41,
                        expected_producer_source=_fake_producer_source(),
                        expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                        device="cpu",
                    )

    def test_optimizer_slot_inventory_cannot_be_truncated(self) -> None:
        def remove_optimizer_slot(checkpoint):
            state = checkpoint["value_optimizer_state_dict"]["state"]
            del state[next(iter(state))]

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=remove_optimizer_slot,
            )
            with self.assertRaises(Phase4CheckpointError):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_optimizer_hyperparameters_must_match_registered_adam(self) -> None:
        for field, value in (
            ("betas", (0.0, 0.0)),
            ("eps", 1.0),
        ):
            def change_hyperparameter(checkpoint, field=field, value=value):
                checkpoint["value_optimizer_state_dict"]["param_groups"][0][
                    field
                ] = value

            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                _, _, checkpoint_path, config_path = (
                    _write_synthetic_phase4_checkpoint(
                        Path(temp_dir),
                        mutate=change_hyperparameter,
                    )
                )
                with self.assertRaises(Phase4CheckpointError):
                    load_phase4_checkpoint(
                        checkpoint_path,
                        config_path,
                        condition="nc_nv",
                        seed=41,
                        expected_producer_source=_fake_producer_source(),
                        expected_training_runtime_sha256=(
                            _FAKE_TRAINING_RUNTIME_SHA256
                        ),
                        device="cpu",
                    )

    def test_scheduler_state_must_match_optimizer_progress(self) -> None:
        def rewind_scheduler(checkpoint):
            scheduler = checkpoint["value_scheduler_state_dict"]
            scheduler["last_epoch"] = 0
            scheduler["_step_count"] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=rewind_scheduler,
            )
            with self.assertRaises(Phase4CheckpointError):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_relabelled_checkpoint_step_is_rejected(self) -> None:
        def relabel_progress(checkpoint):
            checkpoint["progress"]["optimizer_updates"] = 1000
            checkpoint["trainer_state"]["train_step_count"] = 1000

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=relabel_progress,
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "optimizer-update count",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_replay_capacity_must_match_embedded_config(self) -> None:
        def change_replay_capacity(checkpoint):
            checkpoint["replay_capacity"] = 50000

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=change_replay_capacity,
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "replay size"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_deployed_legacy_policy_must_match_model_state(self) -> None:
        def change_old_policy(checkpoint):
            state = checkpoint["policy_model_old_state_dict"]
            key = next(
                name
                for name, value in state.items()
                if torch.is_floating_point(value)
            )
            state[key] = state[key].clone() + 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=change_old_policy,
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "deployed legacy policy",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_missing_behavior_key_is_rejected(self) -> None:
        def remove_value_key(checkpoint):
            key = next(
                name
                for name in checkpoint["model_state_dict"]
                if "value_head" in name
            )
            del checkpoint["model_state_dict"][key]
            del checkpoint["policy_model_old_state_dict"][key]

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=remove_value_key,
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "strict model key inventory",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_unexpected_behavior_key_is_rejected(self) -> None:
        def add_unexpected_key(checkpoint):
            checkpoint["model_state_dict"]["unexpected.payload"] = torch.zeros(1)
            checkpoint["policy_model_old_state_dict"][
                "unexpected.payload"
            ] = torch.zeros(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=add_unexpected_key,
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "strict model key inventory",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_registered_architecture_is_required(self) -> None:
        def change_architecture(checkpoint):
            checkpoint["model_config"]["hidden_size"] = 128
            checkpoint["training_invocation"]["model_config_sha256"] = (
                canonical_json_sha256(checkpoint["model_config"])
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=change_architecture,
            )
            with self.assertRaisesRegex(
                Phase4CheckpointError,
                "registered Phase 4 architecture",
            ):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_registered_dataset_and_random_initialization_are_required(self) -> None:
        def change_dataset_seed(checkpoint):
            checkpoint["dataset_provenance"]["generation_seed"] = 42
            checkpoint["training_invocation"]["dataset_provenance_sha256"] = (
                canonical_json_sha256(checkpoint["dataset_provenance"])
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=change_dataset_seed,
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "generation seed"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

        def change_initialization(checkpoint):
            checkpoint["training_invocation"]["initialization"] = {
                "kind": "weights_checkpoint",
                "artifact_sha256": _fake_sha256("warm start"),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=change_initialization,
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "random initialization"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_checkpoint_integer_identity_is_strict(self) -> None:
        def change_schema_type(checkpoint):
            checkpoint["checkpoint_schema_version"] = 4.0

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=change_schema_type,
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "integer 4"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_checkpoint_bound_seed_and_condition_are_required(self) -> None:
        def swap_seed(checkpoint):
            checkpoint["training_invocation"]["training_seed"] = 42

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir),
                mutate=swap_seed,
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "training seed"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="nc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, checkpoint_path, config_path = _write_synthetic_phase4_checkpoint(
                Path(temp_dir)
            )
            with self.assertRaisesRegex(Phase4CheckpointError, "condition"):
                load_phase4_checkpoint(
                    checkpoint_path,
                    config_path,
                    condition="yc_nv",
                    seed=41,
                    expected_producer_source=_fake_producer_source(),
                    expected_training_runtime_sha256=_FAKE_TRAINING_RUNTIME_SHA256,
                    device="cpu",
                )

    def test_schema_rejects_duplicate_checkpoint_and_state_digests(self) -> None:
        for field in ("checkpoint_sha256", "model_state_sha256"):
            with self.subTest(field=field):
                summary = _valid_summary()
                summary["all_results"][1][field] = summary["all_results"][0][field]
                with self.assertRaisesRegex(
                    Phase4SummaryValidationError,
                    field,
                ):
                    validate_phase4_summary(summary)

    def test_schema_requires_exact_metric_sample_counts(self) -> None:
        for field, invalid_count, expected_count in (
            ("lipschitz_sample_count", 149, 150),
            ("stability_sample_count", 99, 100),
            ("projection_sample_count", 49, 50),
        ):
            with self.subTest(field=field):
                summary = _valid_summary()
                summary["all_results"][0][field] = invalid_count
                with self.assertRaisesRegex(
                    Phase4SummaryValidationError,
                    f"must be {expected_count}",
                ):
                    validate_phase4_summary(summary)

    def test_schema_requires_one_config_identity_per_condition(self) -> None:
        summary = _valid_summary()
        summary["all_results"][0]["model_config_sha256"] = _fake_sha256(
            "different model config"
        )
        with self.assertRaisesRegex(
            Phase4SummaryValidationError,
            "share one model_config_sha256",
        ):
            validate_phase4_summary(summary)


class Phase4TrainingOrchestratorTest(unittest.TestCase):
    def test_relative_paths_resolve_against_their_ownership_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            external = root / "external.json"
            self.assertEqual(
                resolve_phase4_path("results/checkpoints", root),
                root / "results" / "checkpoints",
            )
            self.assertEqual(
                resolve_phase4_path(external, root / "other"),
                external,
            )

    def test_roots_require_matching_fbcode_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            fbcode = root / "fbcode"
            (project / ".git").mkdir(parents=True)
            (project / "configs" / "phase4_2x2_norm_ablation").mkdir(
                parents=True
            )
            (project / "upi_trm_train.py").write_text("# producer\n")
            fbcode.mkdir()
            (fbcode / "buiksat_trm").symlink_to(
                project,
                target_is_directory=True,
            )

            self.assertEqual(
                resolve_phase4_source_roots(str(project), str(fbcode)),
                (project.resolve(), fbcode.resolve()),
            )
            wrong_project = root / "wrong"
            wrong_project.mkdir()
            with self.assertRaisesRegex(Phase4SourceError, "project root"):
                resolve_phase4_source_roots(str(wrong_project), str(fbcode))

    def test_job_uses_explicit_runtime_and_keeps_log_under_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            fbcode = Path(temp_dir) / "fbcode"
            phase4_launcher = Path(temp_dir) / "phase4_runtime_launcher"
            training_runtime = Path(temp_dir) / "upi_trm_train.par"
            project.mkdir()
            fbcode.mkdir()
            phase4_launcher.write_text("#!/bin/sh\n")
            training_runtime.write_bytes(b"PAR")
            expected_runtime_sha256 = "b" * 64
            expected_producer_git_commit = "c" * 40
            with mock.patch(
                "scripts.run_phase4_training.subprocess.Popen"
            ) as popen:
                process = run_job(
                    "nc_nv",
                    41,
                    0,
                    project_root=project,
                    fbcode_root=fbcode,
                    phase4_launcher=phase4_launcher,
                    training_runtime=training_runtime,
                    expected_runtime_sha256=expected_runtime_sha256,
                    expected_producer_git_commit=(
                        expected_producer_git_commit
                    ),
                )

            self.assertIs(process, popen.return_value)
            checkpoint_dir = (
                project
                / "results"
                / "phase4_2x2_norm_ablation"
                / "nc_nv_s41"
            )
            self.assertTrue((checkpoint_dir / "training.log").is_file())
            self.assertFalse(
                (project / "training_phase4_nc_nv_s41.log").exists()
            )
            self.assertEqual(popen.call_args.kwargs["cwd"], str(fbcode))
            command = popen.call_args.args[0]
            self.assertEqual(command[0], str(phase4_launcher))
            self.assertEqual(
                command[1:12],
                [
                    "--purpose",
                    "phase4-training",
                    "--runtime-archive",
                    str(training_runtime),
                    "--expected-runtime-sha256",
                    expected_runtime_sha256,
                    "--source-project-root",
                    str(project),
                    "--expected-source-git-commit",
                    expected_producer_git_commit,
                    "--",
                ],
            )
            child_arguments = command[12:]
            self.assertIn(phase4_run_id("nc_nv", 41), child_arguments)
            self.assertNotIn("--producer-repo-root", child_arguments)
            self.assertNotIn("buck2", command)


class Phase4RuntimeSourceIdentityTest(unittest.TestCase):
    @staticmethod
    def _write_source_tree(root: Path) -> None:
        relative_paths = (
            "phase4_runtime_entrypoint.py",
            "phase4_runtime_profile.py",
            "puzzle_dataset.py",
            "runtime_archive_preflight.py",
            "dataset/__init__.py",
            "dataset/common.py",
            "models/model.py",
            "rl/trainer.py",
            "utils/identity.py",
            "scripts/phase4_checkpoint.py",
            "scripts/phase4_diagnostic_inputs.py",
            "scripts/phase4_result_schema.py",
            "scripts/phase4_source.py",
            "scripts/eval_phase4_2x2_norm_ablation.py",
        )
        for index, relative_path in enumerate(relative_paths):
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# source {index}: {relative_path}\n")

    @staticmethod
    def _source_paths(root: Path) -> list[str]:
        return sorted(
            str(path.relative_to(root))
            for path in root.rglob("*.py")
        )

    @staticmethod
    def _write_producer_source_tree(root: Path) -> None:
        for relative_path in (
            "confirmatory_runtime_launcher.py",
            "phase4_runtime_profile.py",
            "policy_improvement_checkpoint_allowlist.py",
            "policy_improvement_smoke_checkpoint.py",
            "policy_improvement_smoke_runtime.py",
            "puzzle_dataset.py",
            "runtime_archive_preflight.py",
            "scripts/policy_improvement_populations.py",
            "scripts/policy_improvement_registry.py",
            "scripts/policy_improvement_schema.py",
            "scripts/policy_improvement_v2_registry.py",
            "scripts/policy_improvement_v2_schema.py",
            "upi_trm_train.py",
        ):
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"# {relative_path}\n")
        for directory in ("dataset", "evaluators", "models", "rl", "utils"):
            destination = root / directory / "module.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"# {directory}\n")
        confirmatory_config = root / "configs" / "iclr_confirmatory" / "cell.yaml"
        confirmatory_config.parent.mkdir(parents=True, exist_ok=True)
        confirmatory_config.write_text("gamma: 0.9\n")
        policy_config = (
            root / "configs" / "policy_improvement_v1" / "protocol.json"
        )
        policy_config.parent.mkdir(parents=True, exist_ok=True)
        policy_config.write_text("{}\n")
        phase4_config = root / "configs" / "phase4_2x2_norm_ablation"
        phase4_config.mkdir(parents=True, exist_ok=True)
        for condition in ("nc_nv", "nc_yv", "yc_nv", "yc_yv"):
            (phase4_config / f"{condition}.yaml").write_text(
                f"condition: {condition}\n"
            )
        manifest_path = root / SOURCE_MANIFEST_RELATIVE_PATH
        manifest_path.write_text(
            json.dumps(build_producer_source_manifest(root), sort_keys=True)
            + "\n"
        )

    def test_producer_source_binds_manifest_to_authorized_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            producer_root = Path(temp_dir) / "producer"
            self._write_producer_source_tree(producer_root)
            git_identity = {"git_commit": "a" * 40, "git_clean": True}

            with mock.patch(
                "scripts.phase4_source.discover_clean_git_source",
                return_value=git_identity,
            ), mock.patch(
                "scripts.phase4_source.assert_git_files_match_head"
            ):
                producer_source = verify_phase4_producer_source(
                    producer_root,
                    "a" * 40,
                )

            self.assertEqual(producer_source["git_commit"], "a" * 40)
            self.assertTrue(producer_source["git_clean"])
            self.assertEqual(
                producer_source["source_manifest_sha256"],
                hashlib.sha256(
                    (producer_root / SOURCE_MANIFEST_RELATIVE_PATH).read_bytes()
                ).hexdigest(),
            )

            with mock.patch(
                "scripts.phase4_source.discover_clean_git_source",
                return_value=git_identity,
            ), self.assertRaisesRegex(
                Phase4SourceError,
                "authorized commit",
            ):
                verify_phase4_producer_source(producer_root, "b" * 40)

    def test_runtime_source_bytes_bind_to_project_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            runtime = root / "runtime"
            self._write_source_tree(project)
            self._write_source_tree(runtime)

            with mock.patch(
                "scripts.phase4_source.git_files_at_head",
                return_value=self._source_paths(project),
            ), mock.patch(
                "scripts.phase4_source.assert_git_files_match_head"
            ):
                digest = verify_phase4_runtime_sources(
                    project,
                    PHASE4_EVALUATOR_SOURCE_PROFILE,
                    runtime_location=runtime,
                )

                self.assertEqual(
                    digest,
                    phase4_evaluator_source_manifest_sha256(project),
                )
                self.assertEqual(len(digest), 64)

                (runtime / "models" / "model.py").write_text(
                    "# stale runtime model\n"
                )
                with self.assertRaisesRegex(
                    Phase4SourceError,
                    "runtime source bytes differ",
                ):
                    verify_phase4_runtime_sources(
                        project,
                        PHASE4_EVALUATOR_SOURCE_PROFILE,
                        runtime_location=runtime,
                    )

    def test_runtime_source_inventory_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            runtime = root / "runtime"
            self._write_source_tree(project)
            self._write_source_tree(runtime)
            extra = runtime / "rl" / "stale.py"
            extra.write_text("# stale source\n")

            with mock.patch(
                "scripts.phase4_source.git_files_at_head",
                return_value=self._source_paths(project),
            ), mock.patch(
                "scripts.phase4_source.assert_git_files_match_head"
            ):
                with self.assertRaisesRegex(
                    Phase4SourceError,
                    "inventory differs",
                ):
                    verify_phase4_runtime_sources(
                        project,
                        PHASE4_EVALUATOR_SOURCE_PROFILE,
                        runtime_location=runtime,
                    )

    def test_project_inventory_rejects_ignored_python_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            runtime = root / "runtime"
            self._write_source_tree(project)
            self._write_source_tree(runtime)
            tracked_paths = self._source_paths(project)
            (project / "models" / "ignored.py").write_text(
                "# ignored project source\n"
            )
            (runtime / "models" / "ignored.py").write_text(
                "# ignored project source\n"
            )

            with mock.patch(
                "scripts.phase4_source.git_files_at_head",
                return_value=tracked_paths,
            ), mock.patch(
                "scripts.phase4_source.assert_git_files_match_head"
            ):
                with self.assertRaisesRegex(
                    Phase4SourceError,
                    "inventory differs from Git HEAD",
                ):
                    verify_phase4_runtime_sources(
                        project,
                        PHASE4_EVALUATOR_SOURCE_PROFILE,
                        runtime_location=runtime,
                    )

    def test_project_sources_reject_unsafe_index_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            runtime = root / "runtime"
            self._write_source_tree(project)
            self._write_source_tree(runtime)

            with mock.patch(
                "scripts.phase4_source.git_files_at_head",
                return_value=self._source_paths(project),
            ), mock.patch(
                "scripts.phase4_source.assert_git_files_match_head",
                side_effect=RunIdentityError("unsafe index flags"),
            ):
                with self.assertRaisesRegex(
                    Phase4SourceError,
                    "do not match Git HEAD",
                ):
                    verify_phase4_runtime_sources(
                        project,
                        PHASE4_EVALUATOR_SOURCE_PROFILE,
                        runtime_location=runtime,
                    )


if __name__ == "__main__":
    unittest.main()
