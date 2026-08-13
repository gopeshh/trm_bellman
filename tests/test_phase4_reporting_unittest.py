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
from unittest import mock

import torch
import numpy as np
from torch.distributions import Categorical

from scripts import audit_phase4_paper_ready
from scripts import eval_phase4_2x2_norm_ablation
from scripts import make_paper_figures_phase4
from scripts.phase4_checkpoint import (
    Phase4CheckpointError,
    load_phase4_checkpoint,
    phase4_checkpoint_relpath,
    phase4_run_id,
    verify_phase4_summary_checkpoints,
    _expected_model_config,
    _expected_rl_config,
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
    Phase4SourceError,
    phase4_evaluator_source_manifest_sha256,
    resolve_phase4_source_roots,
    verify_phase4_runtime_sources,
)
from scripts.run_phase4_training import run_job
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from rl.replay import Transition
from utils.run_identity import RunIdentityError, canonical_json_sha256
from utils.dataset_provenance import build_dataset_provenance


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
        producer_git_commit="a" * 40,
        producer_source_manifest_sha256=_fake_sha256("producer manifest"),
        initialization_kind="random",
        checkpoint_schema_version=4,
        training_invocation_schema_version=2,
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
    model_config = model.config.model_dump()
    model_state = model.state_dict()
    replay_transition = Transition(
        x={
            "inputs": torch.zeros(16, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0, dtype=torch.long),
            "remaining_edits": torch.tensor(1, dtype=torch.long),
        },
        y=torch.zeros(16, dtype=torch.long),
        action=torch.tensor(0, dtype=torch.long),
        reward=torch.tensor(0.0),
        x_next={
            "inputs": torch.zeros(16, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0, dtype=torch.long),
            "remaining_edits": torch.tensor(0, dtype=torch.long),
        },
        y_next=torch.zeros(16, dtype=torch.long),
        done=torch.tensor(True),
        episode_id=0,
        timestep=0,
        behavior_log_prob=torch.tensor(0.0),
        terminal_reason="budget",
    )
    records = [_fake_sha256(f"record:{seed}:{index}") for index in range(32)]
    pool_sha256 = _fake_sha256(f"pool:{seed}")
    identifier_sha256 = _fake_sha256("identifiers:0-31")
    dataset_provenance = build_dataset_provenance(
        builder_name="rl.training_setup.DummyPuzzleDataset",
        builder_version=1,
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
        "execution_device": "cpu",
        "progress": {
            "env_steps": 12345,
            "optimizer_updates": 5000,
            "optimizer_steps": 4999,
        },
        "model_state_dict": dict(model_state),
        "policy_model_old_state_dict": dict(model_state),
        "policy_model_candidate_state_dict": dict(model_state),
        "target_model_state_dict": dict(model_state),
        "value_optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(1.0)}},
            "param_groups": [{"params": [0]}],
        },
        "policy_optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(1.0)}},
            "param_groups": [{"params": [0]}],
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": None,
        },
        "trainer_state": {
            "train_step_count": 5000,
            "env_step_count": 12345,
            "optimizer_step_count": 4999,
            "collection_state": {},
            "environment_state": {},
            "terminal_reason_replay_version": 1,
        },
        "replay_transitions": [replay_transition],
        "replay_buffer_size": 1,
        "replay_capacity": 100000,
        "model_config": model_config,
        "rl_config": rl_config,
        "dataset_provenance": dataset_provenance,
        "training_invocation": {
            "schema_version": 2,
            "training_seed": seed,
            "run_id": phase4_run_id(condition, seed),
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
            "producer_source": {
                "git_commit": "a" * 40,
                "git_clean": True,
                "source_manifest_sha256": _fake_sha256("producer manifest"),
            },
        },
    }
    if mutate is not None:
        mutate(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_root, config_dir, checkpoint_path, config_path


class Phase4ReportingTest(unittest.TestCase):
    def test_schema_v3_has_exact_availability_and_no_retired_fields(self) -> None:
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
                    "--summary_json",
                    str(summary_path),
                    "--checkpoint_dir",
                    str(temp_path / "checkpoints"),
                    "--config_dir",
                    str(temp_path / "configs"),
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
            ):
                self.assertEqual(make_paper_figures_phase4.main(), 1)

            self.assertFalse(output_path.exists())

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
                    device="cpu",
                ),
                1,
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

    def test_job_log_stays_under_ignored_checkpoint_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            fbcode = Path(temp_dir) / "fbcode"
            project.mkdir()
            fbcode.mkdir()
            with mock.patch(
                "scripts.run_phase4_training.subprocess.Popen"
            ) as popen:
                process = run_job(
                    "nc_nv",
                    41,
                    0,
                    project_root=project,
                    fbcode_root=fbcode,
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
            self.assertIn(str(project), command)
            self.assertIn(phase4_run_id("nc_nv", 41), command)


class Phase4RuntimeSourceIdentityTest(unittest.TestCase):
    @staticmethod
    def _write_source_tree(root: Path) -> None:
        relative_paths = (
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
