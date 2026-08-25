#!/usr/bin/env fbpython
"""Focused tests for sealed full-backend identity and isolation guards."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from policy_improvement_checkpoint_validator import _validate_full_checkpoint
from policy_improvement_full_backend import (
    _canonical_masked_probabilities,
    _FLOAT32_MASS_ENVELOPE,
    _registered_full_result_document,
    _registered_theory_checkpoint_snapshot_kind,
    _select_registered_population,
    _validation_bridge_session_v2,
    build_failed_result,
    calibrate_training_split_throughput,
    FullBackendError,
    LearnedSession,
    load_authorized_dataset_splits,
    open_theory_bridge_session,
    ReadOnlyTheoryBridgeSession,
    required_content_splits,
    SealedFullRunBackend,
    SealedRuntimeIdentity,
    TorchLearnedRunEngine,
)
from policy_improvement_non_smoke_checkpoint import (
    evaluate_without_mutation,
    FullCheckpointError,
    session_model_state_identity,
    validate_full_checkpoint_identity,
)
from policy_improvement_sealed_evidence import seal_generation_checkpoint
from policy_improvement_smoke_checkpoint import (
    build_ppo_smoke_checkpoint,
    publish_checkpoint,
)
from rl.algos.ppo import PPOConfig, PPOTrainer
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from rl.sudoku_checkers import dummy_checker
from rl.upi_trm_trainer import UPITrmTrainer
from scripts.policy_improvement_full_runtime import (
    _validated_result_for_run,
    BackendPackage,
    BackendRequest,
    FullRuntimeError,
    RegisteredFullRun,
)
from scripts.policy_improvement_schema import validate_result, validated_result_payload
from utils.dataset_provenance import (
    dataset_input_sha256s,
    dataset_sample_sha256s,
    ordered_record_sha256,
)
from utils.run_identity import canonical_json_sha256


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


@contextmanager
def _sealed_checkpoint(path: Path):
    descriptor = os.memfd_create(
        "theory-checkpoint-test",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        payload = path.read_bytes()
        os.write(descriptor, payload)
        seals = (
            fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        yield descriptor
    finally:
        os.close(descriptor)


def _runtime_identity() -> dict[str, object]:
    source = _digest("source manifest")
    return {
        "role": "policy-improvement-full",
        "runtime_sha256": _digest("runtime"),
        "source_git_commit": "1" * 40,
        "source_manifest_sha256": source,
        "producer_source_manifest_sha256": _digest("producer manifest"),
        "runtime_profile_sha256": source,
        "selected_source_manifest_sha256": source,
        "runtime_authorization_sha256": _digest("authorization"),
        "launcher_sha256": _digest("launcher"),
    }


def _row() -> dict[str, object]:
    return {
        "run_id": "s1-fixed_base_exact_persistent-n2-k1-s7",
        "phase": "stage1_screen",
        "tier": "pilot",
        "seed": 7,
        "evaluation_split": "validation",
        "method_id": "fixed_base_exact_persistent",
        "base_method_id": "fixed_base_exact_persistent",
        "n": 2,
        "K": 1,
        "alpha": 0.1,
        "ablation_variant": None,
        "config_override": {
            "inner_unroll_n": 2,
            "K": 1,
            "mixture_alpha": 0.1,
        },
        "base_config_canonical_sha256": _digest("base config"),
        "expected_effective_config_sha256": _digest("effective config"),
    }


def _run(root: Path) -> RegisteredFullRun:
    row = _row()
    dataset = root / "dataset"
    dataset.mkdir()
    evidence = root / "evidence"
    evidence.mkdir(mode=0o700)
    project = root / "project"
    project.mkdir()
    return RegisteredFullRun(
        project_root=project,
        protocol_path=project / "protocol.json",
        registry_path=project / "registry.json",
        evidence_root=evidence,
        protocol={
            "protocol_id": "policy-improvement-v1-20260814",
            "output_root": {"relative_path": "policy_improvement_v1"},
            "methods": [
                {
                    "id": "fixed_base_exact_persistent",
                    "config_sha256": _digest("method config"),
                }
            ],
            "dataset": {
                "manifest_sha256": {
                    "status": "available",
                    "value": _digest("dataset manifest"),
                },
                "splits": {
                    "train": {
                        "ordered_record_sha256": {
                            "status": "available",
                            "value": _digest("train order"),
                        }
                    },
                    "validation": {
                        "ordered_record_sha256": {
                            "status": "available",
                            "value": _digest("evaluation order"),
                        }
                    },
                },
            },
            "evaluation_populations": {
                "pilot": {
                    "ordered_record_sha256": {
                        "status": "available",
                        "value": _digest("evaluation order"),
                    }
                }
            },
        },
        registry={"rows": [row]},
        amendment_history=({},),
        row=row,
        protocol_sha256=_digest("protocol"),
        registry_sha256=_digest("registry"),
        amendment_history_sha256=_digest("history"),
        registry_row_sha256=_digest("row"),
        runtime_authorization_sha256=_digest("authorization"),
        interaction_checkpoints=(10, 20),
        final_environment_interactions=20,
        compute_target_recurrent_map_applications=100,
        evaluation_records=8,
        test_open_sha256=None,
        dataset_root=dataset,
    )


def _v2_run(root: Path) -> RegisteredFullRun:
    run = _run(root)
    registered_dataset = run.project_root / "data/registered-v2"
    registered_dataset.mkdir(parents=True)
    population = {
        "population_id": "validation_select",
        "split": "validation",
        "count": 8,
        "ordered_record_sha256": _digest("evaluation order"),
        "ordered_input_sha256": _digest("evaluation inputs"),
        "binding_sha256": _digest("validation population"),
    }
    population_document = {
        "schema_name": "policy_improvement_populations_v2",
        "schema_version": 1,
        "populations": {"validation_select": population},
    }
    protocol = {
        **run.protocol,
        "schema_name": "policy_improvement_protocol_v2",
        "schema_version": 2,
        "protocol_id": "policy-improvement-v2-20260818",
        "dataset": {
            **run.protocol["dataset"],
            "root": "data/registered-v2",
        },
        "population_registry": {
            "path": "configs/policy_improvement_v2/populations.json",
            "schema_name": "policy_improvement_populations_v2",
            "schema_version": 1,
            "sha256": canonical_json_sha256(population_document),
        },
    }
    row = {
        **run.row,
        "schema_name": "policy_improvement_registry_row_v2",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "evaluation_population": "validation_select",
        "scientific_selection": True,
        "paper_evidence_eligible": False,
    }
    registry = {
        "schema_name": "policy_improvement_registry_v2",
        "registry_schema_version": 1,
        "rows": [row],
    }
    return replace(
        run,
        protocol=protocol,
        registry=registry,
        row=row,
        protocol_sha256=canonical_json_sha256(protocol),
        registry_sha256=canonical_json_sha256(registry),
        registry_row_sha256=canonical_json_sha256(row),
        population_document=population_document,
        dataset_root=registered_dataset,
    )


class _TrainingModule:
    _PREVERIFIED_RUNTIME_SHA256: str | None = None


class _Engine:
    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self, request: BackendRequest, runtime: SealedRuntimeIdentity
    ) -> BackendPackage:
        self.calls += 1
        return BackendPackage(result={}, primary_checkpoint_relative_path="unused.pt")


class _TinyDataset:
    def __init__(self) -> None:
        self.seq_len = 4
        self.vocab_size = 5
        self.num_identifiers = 4
        self._samples = []
        for index in range(4):
            inputs = torch.tensor([1, 1, 1, 1], dtype=torch.long)
            solution = torch.tensor([2, 3, 4, 2], dtype=torch.long)
            self._samples.append(
                {
                    "inputs": inputs,
                    "puzzle_identifiers": torch.tensor(index, dtype=torch.long),
                    "initial_plan": inputs.clone(),
                    "solution": solution,
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._samples[index]


def _tiny_ppo_session(
    run: RegisteredFullRun, *, train: bool = True
) -> LearnedSession:
    torch.manual_seed(17)
    dataset = _TinyDataset()
    env_config = PlanEditEnvConfig(
        max_edits=2,
        gamma=0.99,
        reward_shaping=True,
        vocab_size=dataset.vocab_size,
        task_type="dummy",
        stop_action_mode="terminal",
    )
    environment = PlanEditEnv(
        dataset=dataset,
        checker=dummy_checker,
        config=env_config,
    )
    environment.set_stop_action_id(dataset.seq_len * dataset.vocab_size)
    model = TinyRecursiveReasoningModel_ACTV1(
        {
            "batch_size": 4,
            "seq_len": dataset.seq_len,
            "puzzle_emb_ndim": 0,
            "puzzle_emb_len": 0,
            "num_puzzle_identifiers": dataset.num_identifiers,
            "vocab_size": dataset.vocab_size,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 0,
            "L_layers": 1,
            "hidden_size": 16,
            "expansion": 2.0,
            "num_heads": 4,
            "pos_encodings": "rope",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.0,
            "forward_dtype": "float32",
            "mlp_t": False,
            "no_ACT_continue": True,
            "rl_enable_value_head": True,
            "rl_enable_contraction": False,
            "rl_enable_policy_head": True,
            "rl_num_actions": dataset.seq_len * dataset.vocab_size + 1,
            "rl_latent_projection_mode": "enabled",
            "rl_latent_ball_radius": 10.0,
        }
    )
    trainer = PPOTrainer(
        model,
        environment,
        PPOConfig(
            num_steps=16,
            num_epochs=1,
            num_minibatches=1,
            num_train_steps=2,
            batch_size=16,
            inner_unroll_n=1,
        ),
        device=torch.device("cpu"),
    )
    if train:
        trainer.train_step(max_env_steps_to_collect=16)
    return LearnedSession(
        run=run,
        model=model,
        trainer=trainer,
        rl_config=SimpleNamespace(episodic_latent=True, inner_unroll_n=1),
        env_config=env_config,
        train_dataset=dataset,
        evaluation_dataset=dataset,
        checker=dummy_checker,
        task_config=None,
        dataset_provenance={"fixture": "tiny"},
        effective_config={"fixture": "tiny"},
        effective_config_sha256=_digest("effective"),
        initialization_sha256=_digest("initialization"),
        device=torch.device("cpu"),
        config_path=run.project_root / "tiny.yaml",
        method_config_sha256=_digest("method config"),
        run_identity=None,
        evidence_identity={"fixture": "tiny"},
        dataset_manifest_sha256=_digest("dataset manifest"),
        train_ordered_records_sha256=_digest("train order"),
        evaluation_ordered_records_sha256=_digest("evaluation order"),
        evaluation_pool_sha256=_digest("evaluation pool"),
    )


class FullBackendIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = _run(self.root)

    def _request(self, run: RegisteredFullRun | None = None) -> BackendRequest:
        return BackendRequest(
            run=self.run if run is None else run,
            staging_generation=self.root / "staging",
            interaction_checkpoints=(10, 20),
            compute_target_recurrent_map_applications=100,
            compute_maximum_relative_mismatch=0.05,
            require_strict_checkpoint_resume=True,
            require_interaction_and_compute_snapshots=True,
        )

    def test_backend_rejects_mixed_authorization_before_engine(self) -> None:
        engine = _Engine()
        backend = SealedFullRunBackend(
            runtime_identity=_runtime_identity(),
            training_module=_TrainingModule,
            engine=engine,
        )
        mixed = replace(
            self.run,
            runtime_authorization_sha256=_digest("another authorization"),
        )
        with self.assertRaisesRegex(FullRuntimeError, "another runtime"):
            backend.execute(self._request(mixed))
        self.assertEqual(engine.calls, 0)

    def test_backend_requires_both_snapshot_contracts(self) -> None:
        engine = _Engine()
        backend = SealedFullRunBackend(
            runtime_identity=_runtime_identity(),
            training_module=_TrainingModule,
            engine=engine,
        )
        request = replace(
            self._request(), require_interaction_and_compute_snapshots=False
        )
        with self.assertRaisesRegex(FullRuntimeError, "omit"):
            backend.execute(request)
        self.assertEqual(engine.calls, 0)

    def test_validation_loader_never_opens_test(self) -> None:
        opened: list[str] = []

        def loader(split: str) -> tuple[Any, int, int, int]:
            opened.append(split)
            return (split, 16, 5, 1)

        train, evaluation = load_authorized_dataset_splits(self.run, loader)
        self.assertEqual((train[0], evaluation[0]), ("train", "validation"))
        self.assertEqual(opened, ["train", "validation"])
        self.assertNotIn("test", opened)

    def test_validation_bridge_uses_separate_noncontiguous_population(self) -> None:
        import upi_trm_train

        def dataset(size: int) -> Any:
            samples = []
            for index in range(size):
                inputs = torch.tensor([index + 1, 1, 1, 1], dtype=torch.long)
                samples.append(
                    {
                        "inputs": inputs,
                        "puzzle_identifiers": torch.tensor(index, dtype=torch.long),
                        "initial_plan": inputs.clone(),
                        "solution": torch.tensor([2, 3, 4, 2], dtype=torch.long),
                    }
                )
            return upi_trm_train.OfflinePuzzleDataset(
                samples=samples,
                seq_len=4,
                vocab_size=5,
                num_identifiers=size,
            )

        full_validation = dataset(6)
        full_records = dataset_sample_sha256s(full_validation)
        full_inputs = dataset_input_sha256s(full_validation)
        selection_indices = [5, 1]
        bridge_indices = [4, 0]
        selection_dataset = upi_trm_train.select_materialized_dataset_records(
            full_validation,
            selection_indices,
            expected_record_sha256s=[
                full_records[index] for index in selection_indices
            ],
            expected_input_sha256s=[full_inputs[index] for index in selection_indices],
        )
        selection = {
            "population_id": "validation_select",
            "split": "validation",
            "count": 2,
            "indices": selection_indices,
            "record_sha256s": [full_records[index] for index in selection_indices],
            "input_sha256s": [full_inputs[index] for index in selection_indices],
            "ordered_record_sha256": ordered_record_sha256(
                [full_records[index] for index in selection_indices]
            ),
            "ordered_input_sha256": ordered_record_sha256(
                [full_inputs[index] for index in selection_indices]
            ),
            "binding_sha256": _digest("selection binding"),
        }
        bridge = {
            "population_id": "validation_bridge",
            "split": "validation",
            "count": 2,
            "indices": bridge_indices,
            "record_sha256s": [full_records[index] for index in bridge_indices],
            "input_sha256s": [full_inputs[index] for index in bridge_indices],
            "ordered_record_sha256": ordered_record_sha256(
                [full_records[index] for index in bridge_indices]
            ),
            "ordered_input_sha256": ordered_record_sha256(
                [full_inputs[index] for index in bridge_indices]
            ),
            "binding_sha256": _digest("bridge binding"),
        }
        selected_view, selected_records, selected_inputs = (
            _select_registered_population(
                training_module=upi_trm_train,
                materialized_dataset=full_validation,
                population=selection,
            )
        )
        self.assertEqual(selected_records, selection["record_sha256s"])
        self.assertEqual(selected_inputs, selection["input_sha256s"])
        self.assertEqual(
            [sample["original_dataset_index"] for sample in selected_view.samples],
            selection_indices,
        )
        v2_root = self.root / "v2-run"
        v2_root.mkdir()
        run = _v2_run(v2_root)
        manifest = run.dataset_root / "manifests/validation.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(b"validation manifest\n")
        validation_content = run.dataset_root / "validation"
        validation_content.mkdir()
        protocol = copy.deepcopy(run.protocol)
        protocol["dataset"]["splits"]["validation"].update(
            {
                "count": 6,
                "manifest_sha256": {
                    "status": "available",
                    "value": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "ordered_record_sha256": {
                    "status": "available",
                    "value": ordered_record_sha256(full_records),
                },
            }
        )
        run = replace(
            run,
            protocol=protocol,
            population_document={
                "schema_name": "policy_improvement_populations_v2",
                "schema_version": 1,
                "populations": {
                    "validation_select": selection,
                    "validation_bridge": bridge,
                },
            },
            evaluation_records=2,
        )
        checkpoint_session = replace(
            _tiny_ppo_session(run),
            evaluation_dataset=selection_dataset,
            evaluation_ordered_records_sha256=selection["ordered_record_sha256"],
            evaluation_pool_sha256=selection["binding_sha256"],
            dataset_provenance={
                "metadata": {"eval_puzzle_id_offset": 10},
            },
        )
        original_records = dataset_sample_sha256s(checkpoint_session.evaluation_dataset)
        opened_splits: list[str] = []

        def load_split(*, dataset_paths: list[str], pool_size: int, split: str) -> Any:
            del dataset_paths, pool_size
            opened_splits.append(split)
            return full_validation, 4, 5, 6

        wrong_root = self.root / "wrong-dataset-root"
        wrong_root.mkdir()
        with self.assertRaisesRegex(
            FullBackendError,
            "differs from protocol registration",
        ):
            _validation_bridge_session_v2(
                checkpoint_session=checkpoint_session,
                run=replace(run, dataset_root=wrong_root),
                population=bridge,
                training_module=SimpleNamespace(
                    build_dataset_from_paths=load_split,
                ),
            )
        self.assertEqual(opened_splits, [])

        with (
            mock.patch.object(
                upi_trm_train,
                "build_dataset_from_paths",
                side_effect=load_split,
            ),
            mock.patch.object(
                upi_trm_train,
                "_validate_materialized_split_manifest",
            ),
        ):
            bridge_session = _validation_bridge_session_v2(
                checkpoint_session=checkpoint_session,
                run=run,
                population=bridge,
                training_module=upi_trm_train,
            )
        self.assertIsNot(bridge_session, checkpoint_session)
        self.assertIsNot(
            bridge_session.evaluation_dataset,
            checkpoint_session.evaluation_dataset,
        )
        self.assertEqual(
            dataset_sample_sha256s(bridge_session.evaluation_dataset),
            bridge["record_sha256s"],
        )
        self.assertEqual(
            dataset_sample_sha256s(checkpoint_session.evaluation_dataset),
            original_records,
        )
        self.assertEqual(
            [
                sample["original_dataset_index"]
                for sample in bridge_session.evaluation_dataset.samples
            ],
            bridge_indices,
        )
        self.assertEqual(opened_splits, ["validation"])

        external_manifest = self.root / "external-validation.json"
        external_manifest.write_bytes(manifest.read_bytes())
        manifest.unlink()
        manifest.symlink_to(external_manifest)
        opened_splits.clear()
        with self.assertRaisesRegex(FullBackendError, "canonical regular file"):
            _validation_bridge_session_v2(
                checkpoint_session=checkpoint_session,
                run=run,
                population=bridge,
                training_module=upi_trm_train,
            )
        self.assertEqual(opened_splits, [])

        manifest.unlink()
        manifest.write_bytes(external_manifest.read_bytes())
        external_content = self.root / "external-validation"
        external_content.mkdir()
        validation_content.rmdir()
        validation_content.symlink_to(external_content, target_is_directory=True)
        with self.assertRaisesRegex(FullBackendError, "canonical directory"):
            _validation_bridge_session_v2(
                checkpoint_session=checkpoint_session,
                run=run,
                population=bridge,
                training_module=upi_trm_train,
            )
        self.assertEqual(opened_splits, [])

    def test_test_loader_fails_before_open_without_test_open(self) -> None:
        row = dict(self.run.row)
        row["evaluation_split"] = "test"
        test_run = replace(self.run, row=row, test_open_sha256=None)
        opened: list[str] = []
        with self.assertRaisesRegex(Exception, "TEST_OPEN"):
            load_authorized_dataset_splits(
                test_run,
                lambda split: opened.append(split) or (split, 16, 5, 1),
            )
        self.assertEqual(opened, [])

    def test_authenticated_test_row_loads_only_train_and_test(self) -> None:
        row = dict(self.run.row)
        row["evaluation_split"] = "test"
        test_run = replace(
            self.run,
            row=row,
            test_open_sha256=_digest("test open"),
        )
        opened: list[str] = []
        self.assertEqual(required_content_splits(test_run), ("train", "test"))
        load_authorized_dataset_splits(
            test_run,
            lambda split: opened.append(split) or (split, 16, 5, 1),
        )
        self.assertEqual(opened, ["train", "test"])

    def test_synthetic_throughput_calibration_is_mechanics_only(self) -> None:
        session = _tiny_ppo_session(self.run, train=False)
        result = calibrate_training_split_throughput(
            session,
            environment_interactions=16,
        )
        self.assertEqual(
            set(result),
            {
                "schema_name",
                "schema_version",
                "source",
                "environment_interactions",
                "recurrent_map_applications",
                "elapsed_seconds",
                "compute_snapshot",
                "compute_accounting",
                "scientific_selection",
                "test_data_opened",
            },
        )
        self.assertEqual(
            result["schema_name"],
            "policy_improvement_training_throughput_smoke_v2",
        )
        self.assertEqual(result["environment_interactions"], 16)
        self.assertEqual(
            result["compute_accounting"]["environment_interactions"]["value"],
            16,
        )
        self.assertEqual(
            result["compute_accounting"]["checkpoint_seconds"]["value"],
            0.0,
        )
        self.assertEqual(
            result["compute_accounting"]["audit_seconds"]["status"],
            "unavailable",
        )
        self.assertFalse(result["scientific_selection"])
        self.assertFalse(result["test_data_opened"])

    def test_inverse_cdf_normalizes_float32_probability_roundoff(self) -> None:
        rounded_probability = float(
            torch.tensor(1.0 / 21.0, dtype=torch.float32).item()
        )
        probabilities = (rounded_probability,) * 21
        self.assertGreater(abs(math.fsum(probabilities) - 1.0), 1e-8)
        self.assertEqual(
            ReadOnlyTheoryBridgeSession._inverse_cdf_action(
                probabilities,
                0.999999999,
            ),
            20,
        )
        self.assertEqual(
            ReadOnlyTheoryBridgeSession._inverse_cdf_action(
                (0.0, 0.50000006, 0.50000006),
                0.0,
            ),
            1,
        )
        for invalid in ((0.4, 0.4), (-0.1, 1.1), (float("nan"), 1.0)):
            with (
                self.subTest(probabilities=invalid),
                self.assertRaisesRegex(FullBackendError, "does not sum to one"),
            ):
                ReadOnlyTheoryBridgeSession._inverse_cdf_action(invalid, 0.5)


class FullCheckpointGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"immutable checkpoint")

    def test_evaluation_cannot_mutate_checkpoint_or_live_state(self) -> None:
        state = {"value": "unchanged"}
        self.assertEqual(
            evaluate_without_mutation(
                checkpoint_path=self.checkpoint,
                live_state_fingerprint=lambda: state["value"],
                evaluator=lambda: 17,
            ),
            17,
        )
        with self.assertRaisesRegex(FullCheckpointError, "checkpoint"):
            evaluate_without_mutation(
                checkpoint_path=self.checkpoint,
                live_state_fingerprint=lambda: state["value"],
                evaluator=lambda: self.checkpoint.write_bytes(b"changed"),
            )
        self.checkpoint.write_bytes(b"immutable checkpoint")

        def mutate_state() -> None:
            state["value"] = "changed"

        with self.assertRaisesRegex(FullCheckpointError, "training state"):
            evaluate_without_mutation(
                checkpoint_path=self.checkpoint,
                live_state_fingerprint=lambda: state["value"],
                evaluator=mutate_state,
            )

        state["value"] = "unchanged"

        def mutate_then_raise() -> None:
            state["value"] = "changed"
            raise RuntimeError("synthetic evaluator failure")

        with self.assertRaisesRegex(FullCheckpointError, "training state"):
            evaluate_without_mutation(
                checkpoint_path=self.checkpoint,
                live_state_fingerprint=lambda: state["value"],
                evaluator=mutate_then_raise,
            )

    def test_checkpoint_identity_rejects_missing_field(self) -> None:
        identity = {
            "schema_name": "policy_improvement_full_checkpoint_identity_v1",
            "schema_version": 1,
            "run_id": "run",
            "method_id": "method",
            "protocol_sha256": _digest("protocol"),
            "registry_row_sha256": _digest("row"),
            "amendment_history_sha256": _digest("history"),
            "runtime_authorization_sha256": _digest("authorization"),
            "training_runtime_sha256": _digest("runtime"),
            "training_source_git_commit": "1" * 40,
            "training_source_manifest_sha256": _digest("source"),
            "launcher_sha256": _digest("launcher"),
            "dataset_manifest_sha256": _digest("dataset"),
            "dataset_provenance_sha256": _digest("provenance"),
            "effective_config_sha256": _digest("config"),
            "snapshot_kind": "interaction_matched",
            "environment_interactions": 20,
            "recurrent_map_applications": 100,
            "parent_checkpoint_sha256": None,
            "test_open_sha256": None,
        }
        validate_full_checkpoint_identity(identity)
        del identity["launcher_sha256"]
        with self.assertRaisesRegex(FullCheckpointError, "inventory"):
            validate_full_checkpoint_identity(identity)

    def test_theory_checkpoint_kind_tracks_registered_schedule(self) -> None:
        run = _run(self.root)
        self.assertEqual(
            _registered_theory_checkpoint_snapshot_kind(run, 10),
            "scheduled",
        )
        self.assertEqual(
            _registered_theory_checkpoint_snapshot_kind(run, 20),
            "interaction_matched",
        )
        for interactions in (15, True, 30):
            with (
                self.subTest(interactions=interactions),
                self.assertRaisesRegex(FullBackendError, "registered run schedule"),
            ):
                _registered_theory_checkpoint_snapshot_kind(run, interactions)

    def test_model_identity_rejects_nonfinite_live_state(self) -> None:
        run = _run(self.root)
        session = _tiny_ppo_session(run)
        parameter = next(session.model.parameters())
        with torch.no_grad():
            parameter.reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(FullCheckpointError, "nonfinite"):
            session_model_state_identity(session)

    def test_failure_result_preserves_validation_test_isolation(self) -> None:
        run = _run(self.root)
        runtime = SealedRuntimeIdentity.from_mapping(_runtime_identity())
        result = build_failed_result(
            run,
            runtime,
            phase="training",
            error=RuntimeError("synthetic failure"),
        )
        checked = validate_result(result)
        self.assertEqual(checked["status"], "failed")
        self.assertEqual(
            checked["identities"]["test_open_sha256"],
            {"status": "unavailable", "reason": "test_data_not_opened"},
        )

    def test_v2_full_results_require_the_strict_registered_envelope(self) -> None:
        run = _v2_run(self.root)
        runtime = SealedRuntimeIdentity.from_mapping(_runtime_identity())
        document = build_failed_result(
            run,
            runtime,
            phase="training",
            error=RuntimeError("synthetic v2 failure"),
        )
        self.assertEqual(document["schema_name"], "policy_improvement_result_v2")
        self.assertTrue(document["validation_data_opened"])
        self.assertFalse(document["test_data_opened"])
        self.assertTrue(document["scientific_selection"])
        checked_document, payload = validated_result_payload(document)
        self.assertEqual(checked_document, document)
        self.assertEqual(
            payload["schema_name"], "policy_improvement_full_result_payload_v2"
        )
        self.assertEqual(payload["evaluation_population_id"], "validation_select")
        self.assertEqual(
            payload["identities"]["evaluation_ordered_records_sha256"],
            _digest("evaluation order"),
        )

        legacy_payload = dict(payload)
        del legacy_payload["evaluation_population_id"]
        del legacy_payload["evaluation_population_binding_sha256"]
        legacy_payload["schema_name"] = "policy_improvement_v1"
        legacy_payload["schema_version"] = 3
        validate_result(legacy_payload)
        with self.assertRaisesRegex(FullRuntimeError, "protocol namespace"):
            _validated_result_for_run(
                run,
                legacy_payload,
                result_validator=lambda value: value,
            )

        v1_run = replace(run, protocol={"schema_name": "policy_improvement_v1"})
        with self.assertRaisesRegex(FullRuntimeError, "protocol namespace"):
            _validated_result_for_run(
                v1_run,
                document,
                result_validator=lambda value: value,
            )

        hostile = dict(document)
        hostile["registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(FullRuntimeError, "authenticated registration"):
            _validated_result_for_run(
                run,
                hostile,
                result_validator=validate_result,
            )

        self.assertEqual(
            _registered_full_result_document(run, legacy_payload)["schema_name"],
            "policy_improvement_result_v2",
        )

    def test_theory_upi_restore_uses_only_quiet_resume_entrypoint(self) -> None:
        run = _run(self.root)
        runtime = _runtime_identity()
        checkpoint_identity = {
            "schema_name": "policy_improvement_full_checkpoint_identity_v1",
            "schema_version": 1,
            "run_id": run.row["run_id"],
            "method_id": run.row["method_id"],
            "protocol_sha256": run.protocol_sha256,
            "registry_row_sha256": run.registry_row_sha256,
            "amendment_history_sha256": run.amendment_history_sha256,
            "runtime_authorization_sha256": run.runtime_authorization_sha256,
            "training_runtime_sha256": runtime["runtime_sha256"],
            "training_source_git_commit": runtime["source_git_commit"],
            "training_source_manifest_sha256": runtime["source_manifest_sha256"],
            "launcher_sha256": runtime["launcher_sha256"],
            "dataset_manifest_sha256": _digest("dataset manifest"),
            "dataset_provenance_sha256": _digest("dataset provenance"),
            "effective_config_sha256": _digest("effective config"),
            "snapshot_kind": "interaction_matched",
            "environment_interactions": run.final_environment_interactions,
            "recurrent_map_applications": 10,
            "parent_checkpoint_sha256": None,
            "test_open_sha256": None,
        }
        self.checkpoint.write_bytes(b"sealed UPI checkpoint")
        checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        authenticated_roles = {"model": _digest("model role")}
        authenticated_model = canonical_json_sha256(authenticated_roles)
        quiet_resume = mock.Mock(return_value=0)
        normal_resume = mock.Mock(side_effect=AssertionError("normal resume called"))
        training_module = SimpleNamespace(
            _PREVERIFIED_RUNTIME_SHA256=None,
            _capture_rng_state=mock.Mock(return_value={}),
            _restore_rng_state=mock.Mock(),
            _load_checkpoint_payload=mock.Mock(
                return_value=(
                    {
                        "policy_improvement_full_identity": checkpoint_identity,
                        "policy_improvement_full_evaluation_state_dicts": {},
                    },
                    checkpoint_sha256,
                )
            ),
            resume_from_checkpoint=normal_resume,
            resume_from_checkpoint_for_theory_evaluation=quiet_resume,
        )

        class UPITrainerStub:
            pass

        session = SimpleNamespace(
            model=object(),
            trainer=UPITrainerStub(),
            device="cpu",
            dataset_provenance={},
            run_identity=None,
        )
        request = {
            "identity": {
                "checkpoint": {
                    "sha256": checkpoint_sha256,
                    "size_bytes": self.checkpoint.stat().st_size,
                    "snapshot_kind": "interaction_matched",
                    "environment_interactions": run.final_environment_interactions,
                }
            },
            "registered_run": run,
            "registered_state_indices": [0],
        }
        adapter = object()
        with (
            mock.patch.object(
                TorchLearnedRunEngine,
                "_build_session",
                return_value=session,
            ),
            mock.patch(
                "policy_improvement_full_backend.session_model_state_identity",
                return_value=(authenticated_model, authenticated_roles),
            ),
            mock.patch(
                "policy_improvement_full_backend.ReadOnlyTheoryBridgeSession",
                return_value=adapter,
            ),
            mock.patch(
                "policy_improvement_full_backend._validate_theory_identity_bundle"
            ),
            _sealed_checkpoint(self.checkpoint) as descriptor,
        ):
            observed = open_theory_bridge_session(
                request,
                self.checkpoint,
                expected_checkpoint_sha256=checkpoint_sha256,
                sealed_checkpoint_descriptor=descriptor,
                authenticated_model_state_sha256=authenticated_model,
                authenticated_role_state_sha256s=authenticated_roles,
                authenticated_validation_sha256=_digest("checkpoint validation"),
                runtime_identity=runtime,
                training_module=training_module,
            )

        self.assertIs(observed, adapter)
        quiet_resume.assert_called_once()
        normal_resume.assert_not_called()

    def test_real_tiny_ppo_checkpoint_is_not_theory_eligible(self) -> None:
        import upi_trm_train

        run = _run(self.root)
        theory_amendment = {
            "schema_name": "policy_improvement_theory_bridge_amendment_v1",
            "fixture": "tiny",
        }
        row = dict(run.row)
        row.update(
            {
                "run_id": "s1-matched_ppo-n2-k1-s7",
                "method_id": "matched_ppo",
                "base_method_id": "matched_ppo",
                "config_override": {"inner_unroll_n": 2},
                "base_config_canonical_sha256": _digest("base config"),
                "expected_effective_config_sha256": _digest("effective config"),
            }
        )
        split_manifest_sha256 = _digest("validation manifest")
        method_config_sha256 = _digest("method config")
        protocol = dict(run.protocol)
        protocol["methods"] = [
            {
                "id": "matched_ppo",
                "config_sha256": method_config_sha256,
            }
        ]
        protocol["budgets"] = {
            "pilot": {
                "checkpoint_environment_interactions": [16],
                "environment_interactions": 16,
                "evaluation_records": 1,
            }
        }
        runtime = _runtime_identity()
        authorization = {
            "producer_source_manifest_sha256": runtime[
                "producer_source_manifest_sha256"
            ],
            "launcher_sha256": runtime["launcher_sha256"],
            "roles": [
                {
                    "role": "policy-improvement-full",
                    "source_git_commit": runtime["source_git_commit"],
                    "runtime_sha256": runtime["runtime_sha256"],
                    "runtime_profile_sha256": runtime["runtime_profile_sha256"],
                    "selected_source_manifest_sha256": runtime[
                        "selected_source_manifest_sha256"
                    ],
                }
            ],
        }
        authorization_sha256 = canonical_json_sha256(authorization)
        runtime["runtime_authorization_sha256"] = authorization_sha256
        run = replace(
            run,
            protocol=protocol,
            amendment_history=(theory_amendment,),
            row=row,
            evaluation_records=1,
            final_environment_interactions=16,
            interaction_checkpoints=(16,),
            runtime_authorization_sha256=authorization_sha256,
        )
        session = _tiny_ppo_session(run)
        samples = dataset_sample_sha256s(session.evaluation_dataset)
        protocol["dataset"] = {
            "splits": {
                "validation": {
                    "manifest_sha256": {
                        "status": "available",
                        "value": split_manifest_sha256,
                    },
                    "ordered_record_sha256": {
                        "status": "available",
                        "value": ordered_record_sha256(samples),
                    },
                }
            }
        }
        checkpoint_identity = {
            "schema_name": "policy_improvement_full_checkpoint_identity_v1",
            "schema_version": 1,
            "run_id": row["run_id"],
            "method_id": row["method_id"],
            "protocol_sha256": run.protocol_sha256,
            "registry_row_sha256": run.registry_row_sha256,
            "amendment_history_sha256": run.amendment_history_sha256,
            "runtime_authorization_sha256": run.runtime_authorization_sha256,
            "training_runtime_sha256": runtime["runtime_sha256"],
            "training_source_git_commit": runtime["source_git_commit"],
            "training_source_manifest_sha256": runtime["source_manifest_sha256"],
            "launcher_sha256": runtime["launcher_sha256"],
            "dataset_manifest_sha256": session.dataset_manifest_sha256,
            "dataset_provenance_sha256": canonical_json_sha256(
                session.dataset_provenance
            ),
            "effective_config_sha256": session.effective_config_sha256,
            "snapshot_kind": "interaction_matched",
            "environment_interactions": 16,
            "recurrent_map_applications": 1,
            "parent_checkpoint_sha256": None,
            "test_open_sha256": None,
        }
        checkpoint = self.root / "tiny-ppo.pt"
        checkpoint_sha256 = publish_checkpoint(
            build_ppo_smoke_checkpoint(
                session.trainer,
                identity=checkpoint_identity,
                parent_checkpoint_sha256=None,
            ),
            checkpoint,
        )
        full_validation_request = {
            "schema_name": "policy_improvement_checkpoint_validation_request_v2",
            "schema_version": 2,
            "checkpoint_sha256": checkpoint_sha256,
            "protocol": protocol,
            "protocol_sha256": run.protocol_sha256,
            "registry_row": row,
            "registry_row_sha256": run.registry_row_sha256,
            "project_root": str(run.project_root),
            "dataset_root": str(run.dataset_root),
            "evidence_root": str(run.evidence_root),
            "runtime_authorization": authorization,
            "run_id": row["run_id"],
            "method_id": row["method_id"],
            "seed": row["seed"],
            "snapshot_kind": "interaction_matched",
            "environment_interactions": 16,
            "parent_checkpoint_sha256": None,
            "initialization_sha256": session.initialization_sha256,
            "registry_sha256": run.registry_sha256,
            "amendment_history_sha256": run.amendment_history_sha256,
            "runtime_authorization_sha256": authorization_sha256,
            "recurrent_map_applications": 1,
            "compute_target_recurrent_map_applications": 1,
            "test_open_sha256": None,
        }
        with (
            mock.patch(
                "policy_improvement_checkpoint_validator.discover_clean_git_source",
                return_value={
                    "git_commit": runtime["source_git_commit"],
                    "git_clean": True,
                },
            ),
            mock.patch.object(
                TorchLearnedRunEngine,
                "_build_session",
                return_value=session,
            ),
        ):
            sealed_full_checkpoint = seal_generation_checkpoint(
                checkpoint.parent,
                checkpoint.name,
                expected_sha256=checkpoint_sha256,
                expected_size_bytes=checkpoint.stat().st_size,
            )
            try:
                semantic = _validate_full_checkpoint(
                    full_validation_request,
                    protocol,
                    row,
                    authorization,
                    run.project_root,
                    run.dataset_root,
                    run.evidence_root,
                    sealed_full_checkpoint,
                )
            finally:
                sealed_full_checkpoint.close()
        self.assertEqual(semantic["checkpoint_sha256"], checkpoint_sha256)
        self.assertEqual(semantic["training_call_delta"], 0)
        self.assertEqual(semantic["optimizer_step_delta"], 0)
        self.assertIsNone(semantic["theory_model_identity"])
        model_sha256, role_state_sha256s = session_model_state_identity(session)
        checkpoint_validation_sha256 = _digest("checkpoint validation")
        model_state_sha256 = state_dict_sha256(session.model.state_dict())
        recurrent_transition_sha256 = state_dict_sha256(
            {
                name: value
                for name, value in session.model.state_dict().items()
                if not name.startswith("edit_policy.")
                and not name.startswith("value_head.")
            }
        )
        selected_records = [
            {
                "record_index": 0,
                "dataset_record_sha256": samples[0],
            }
        ]
        theory_identity = {
            "protocol_sha256": run.protocol_sha256,
            "theory_amendment_sha256": canonical_json_sha256(theory_amendment),
            "registry_row_sha256": run.registry_row_sha256,
            "checkpoint": {
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint.stat().st_size,
                "snapshot_kind": "interaction_matched",
                "environment_interactions": 16,
            },
            "model": {
                "model_sha256": model_sha256,
                "model_config_sha256": canonical_json_sha256(
                    upi_trm_train._config_dict(session.model.config)
                ),
                "current_policy_sha256": model_state_sha256,
                "candidate_policy_sha256": model_state_sha256,
                "deployed_policy_sha256": model_state_sha256,
                "recurrent_transition_sha256": recurrent_transition_sha256,
            },
            "config": {
                "file_sha256": method_config_sha256,
                "base_canonical_sha256": row["base_config_canonical_sha256"],
                "effective_config_sha256": row["expected_effective_config_sha256"],
            },
            "producer_source": {
                "git_commit": runtime["source_git_commit"],
                "source_manifest_sha256": runtime["producer_source_manifest_sha256"],
            },
            "training_runtime": {
                key: value
                for key, value in runtime.items()
                if key != "producer_source_manifest_sha256"
            },
            "dataset_records": {
                "split": "validation",
                "split_manifest_sha256": split_manifest_sha256,
                "ordered_record_sha256": ordered_record_sha256(samples),
                "selected_record_indices_sha256": canonical_json_sha256([0]),
                "selected_records_sha256": canonical_json_sha256(selected_records),
                "record_count": 1,
            },
            "evaluator_source": {
                "git_commit": "2" * 40,
                "source_manifest_sha256": _digest("evaluator source"),
            },
            "evaluator_runtime": {
                "runtime_sha256": _digest("evaluator runtime"),
                "runtime_profile_sha256": _digest("evaluator profile"),
                "runtime_authorization_sha256": run.runtime_authorization_sha256,
                "launcher_sha256": _digest("evaluator launcher"),
            },
        }
        request = {
            "identity": theory_identity,
            "registered_run": run,
            "registered_state_indices": [0],
        }
        with (
            mock.patch.object(
                TorchLearnedRunEngine,
                "_build_session",
                return_value=session,
            ),
            _sealed_checkpoint(checkpoint) as descriptor,
            self.assertRaisesRegex(
                FullBackendError,
                "no registered theory model identity",
            ),
        ):
            open_theory_bridge_session(
                request,
                checkpoint,
                expected_checkpoint_sha256=checkpoint_sha256,
                sealed_checkpoint_descriptor=descriptor,
                authenticated_model_state_sha256=model_sha256,
                authenticated_role_state_sha256s=role_state_sha256s,
                authenticated_validation_sha256=checkpoint_validation_sha256,
                runtime_identity=runtime,
                training_module=upi_trm_train,
            )
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_sha256
        )

    def test_real_upi_theory_capabilities_match_frozen_trainer_semantics(self) -> None:
        import upi_trm_train

        run = _run(self.root)
        base = _tiny_ppo_session(run)
        config = RLConfig(
            gamma=0.99,
            K=1,
            inner_unroll_n=2,
            max_edits=2,
            task_name="dummy",
            episodic_latent=False,
            stop_action_mode="terminal",
            reward_shaping=True,
            exact_k_step_targets=True,
            exact_baseline_summation=True,
            theory_exact_mixture=True,
            training_protocol="fixed_base_exact",
            value_target_clip=None,
            advantage_clip=0.05,
            enable_contraction=False,
            batch_centered_advantage=False,
            replay_capacity=32,
            batch_size=1,
            rollout_episodes_per_step=1,
            num_train_steps=1,
            use_tqdm=False,
        )
        environment = PlanEditEnv(
            dataset=base.evaluation_dataset,
            checker=dummy_checker,
            config=base.env_config,
        )
        environment.set_stop_action_id(int(base.model.config.rl_num_actions) - 1)
        trainer = UPITrmTrainer(
            base.model,
            environment,
            config,
            device=torch.device("cpu"),
        )
        trainer.set_checker_fn(dummy_checker)
        session = replace(
            base,
            trainer=trainer,
            rl_config=config,
            env_config=base.env_config,
        )
        checkpoint = self.root / "real-upi-theory.pt"
        checkpoint.write_bytes(b"sealed real UPI theory fixture")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with _sealed_checkpoint(checkpoint) as descriptor:
            adapter = ReadOnlyTheoryBridgeSession(
                request_identity={"fixture": "real-upi-v2"},
                checkpoint_descriptor=os.dup(descriptor),
                checkpoint_sha256=checkpoint_sha256,
                snapshot_kind="smoke_resume",
                environment_interactions=32,
                session=session,
                training_module=upi_trm_train,
                evaluation_state_dicts={},
                registered_state_indices=(0,),
                reported_record_indices=(749,),
            )
        self.addCleanup(adapter.close)
        adapter.begin_read_only_evaluation()

        def end_transaction_if_active() -> None:
            if adapter._read_only_transaction is not None:
                adapter.end_read_only_evaluation()

        self.addCleanup(end_transaction_if_active)

        before = adapter.read_only_snapshot_v2()
        (state,) = adapter.registered_states()
        self.assertEqual(state.record_index, 749)
        estimator = adapter.training_advantage_estimator(state.state_id)
        q_values = [0.0] * len(state.action_mask)
        for action_index, allowed in enumerate(state.action_mask):
            if not allowed:
                continue
            for outcome in adapter.exact_action_outcomes(
                state.state_id,
                action_index,
            ):
                next_value = 0.0
                if not outcome.terminal:
                    assert outcome.next_state_id is not None
                    next_value = adapter.endpoint_value(
                        outcome.next_state_id,
                        config.inner_unroll_n,
                    )
                q_values[action_index] += outcome.probability * (
                    outcome.reward + config.gamma * next_value
                )
        baseline = sum(
            probability * q_value
            for probability, q_value, allowed in zip(
                state.current_probabilities,
                q_values,
                state.action_mask,
            )
            if allowed
        )
        clipped = [
            max(-0.05, min(0.05, q_value - baseline)) if allowed else 0.0
            for q_value, allowed in zip(q_values, state.action_mask)
        ]
        clipped_mean = sum(
            probability * advantage
            for probability, advantage, allowed in zip(
                state.current_probabilities,
                clipped,
                state.action_mask,
            )
            if allowed
        )
        expected = [
            advantage - clipped_mean if allowed else 0.0
            for advantage, allowed in zip(clipped, state.action_mask)
        ]
        self.assertEqual(estimator.action_mask, state.action_mask)
        self.assertEqual(estimator.clipping_kind, "clip_then_exact_recenter")
        self.assertEqual(estimator.clip_value, 0.05)
        for observed, wanted in zip(estimator.advantages, expected):
            self.assertAlmostEqual(observed, wanted, places=5)

        deployed_witness = adapter.persistent_endpoint_witness(
            state.state_id,
            config.inner_unroll_n,
        )
        reference_witness = adapter.persistent_endpoint_witness(state.state_id, 8)
        self.assertEqual(deployed_witness.deployed_transition_depth, 2)
        self.assertEqual(reference_witness.deployed_transition_depth, 2)
        for field in (
            "carried_successor_latent_sha256",
            "trajectory_sha256",
            "action_probabilities_sha256",
            "recurrent_transition_sha256",
        ):
            self.assertEqual(
                getattr(deployed_witness, field),
                getattr(reference_witness, field),
            )

        first_return = adapter.sample_current_policy_return(
            state.state_id,
            90210,
            16,
            config.gamma,
        )
        second_return = adapter.sample_current_policy_return(
            state.state_id,
            90210,
            16,
            config.gamma,
        )
        self.assertEqual(first_return, second_return)
        self.assertTrue(first_return.terminal)
        self.assertGreater(first_return.environment_steps, 0)
        self.assertLessEqual(first_return.environment_steps, 16)
        self.assertTrue(math.isfinite(first_return.discounted_return))
        first_pair = adapter.sample_paired_policy_returns(
            state.state_id,
            90210,
            16,
            config.gamma,
            0.1,
        )
        second_pair = adapter.sample_paired_policy_returns(
            state.state_id,
            90210,
            16,
            config.gamma,
            0.2,
        )
        repeated_pair = adapter.sample_paired_policy_returns(
            state.state_id,
            90210,
            16,
            config.gamma,
            0.1,
        )
        self.assertEqual(first_pair, repeated_pair)
        self.assertEqual(first_pair.current_policy, second_pair.current_policy)
        self.assertEqual(
            first_pair.common_random_numbers_sha256,
            second_pair.common_random_numbers_sha256,
        )
        self.assertEqual(adapter.read_only_snapshot_v2(), before)
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint_sha256,
        )
        adapter.end_read_only_evaluation()


if __name__ == "__main__":
    unittest.main()


class CanonicalMaskedProbabilityTest(unittest.TestCase):
    """The float32 to binary64 renormalization must preserve the law exactly."""

    def _float32_softmax(self, logits: list[float], mask: list[bool]) -> list[float]:
        tensor = torch.tensor(logits, dtype=torch.float32)
        tensor = tensor.masked_fill(
            ~torch.tensor(mask, dtype=torch.bool), float("-inf")
        )
        return torch.softmax(tensor, dim=-1).tolist()

    def test_float32_rounding_is_normalized_without_changing_the_law(self) -> None:
        torch.manual_seed(1257297357)
        mask = [True] * 3000 + [False] * 1097
        logits = torch.randn(len(mask)).tolist()
        raw = self._float32_softmax(logits, mask)
        raw_mass = math.fsum(value for value, valid in zip(raw, mask) if valid)
        # A real float32 masked softmax misses one by far more than 1e-10.
        self.assertGreater(abs(raw_mass - 1.0), 1e-10)
        self.assertLess(abs(raw_mass - 1.0), _FLOAT32_MASS_ENVELOPE)

        canonical, mass_error, correction = _canonical_masked_probabilities(
            raw, mask, label="current policy"
        )
        canonical_mass = math.fsum(
            value for value, valid in zip(canonical, mask) if valid
        )
        # This is the bridge's precondition, unchanged at 1e-10.
        self.assertLessEqual(abs(canonical_mass - 1.0), 1e-10)
        self.assertAlmostEqual(mass_error, abs(raw_mass - 1.0), places=15)
        self.assertGreaterEqual(correction, 0.0)
        # The categorical law is preserved: every pairwise ratio is unchanged.
        first = next(index for index, valid in enumerate(mask) if valid and raw[index])
        for index, valid in enumerate(mask):
            if not valid or not raw[index]:
                continue
            self.assertAlmostEqual(
                canonical[index] / canonical[first],
                raw[index] / raw[first],
                places=12,
            )

    def test_masked_leakage_is_scaled_not_zeroed(self) -> None:
        mask = [True, True, False]
        raw = [0.5, 0.5, 1e-8]
        canonical, _, _ = _canonical_masked_probabilities(
            raw, mask, label="current policy"
        )
        # The leak must survive so the bridge's masked-mass check can reject it.
        self.assertGreater(canonical[2], 0.0)
        self.assertAlmostEqual(canonical[2], 1e-8, places=15)

    def test_invalid_probabilities_fail_closed(self) -> None:
        mask = [True, True]
        for values, mask_values in (
            ([0.5, -0.5], mask),
            ([0.5, float("nan")], mask),
            ([0.5, float("inf")], mask),
            ([0.0, 0.0], mask),
            ([0.5, 0.5, 0.5], mask),
            ([], []),
            ([0.5, 0.5], [False, False]),
        ):
            with self.subTest(values=values, mask=mask_values):
                with self.assertRaises(FullBackendError):
                    _canonical_masked_probabilities(
                        values, mask_values, label="current policy"
                    )

    def test_mass_outside_the_float32_envelope_fails_closed(self) -> None:
        mask = [True, True]
        # An unnormalized weight vector is a structural defect, not rounding.
        with self.assertRaisesRegex(FullBackendError, "float32 rounding envelope"):
            _canonical_masked_probabilities([3.0, 4.0], mask, label="current policy")
        with self.assertRaisesRegex(FullBackendError, "float32 rounding envelope"):
            _canonical_masked_probabilities(
                [0.4, 0.4], mask, label="current policy"
            )

    def test_normalization_does_not_force_mixture_equality(self) -> None:
        """Deployed stays its own law, so a non-mixture deployment is visible."""

        mask = [True, True, True]
        alpha = 0.1
        current = [0.70, 0.20, 0.10]
        candidate = [0.10, 0.30, 0.60]
        # A deployed law that is not the alpha mixture, off by well over 1e-6.
        deployed = [0.50, 0.25, 0.25]
        normalized = {
            name: _canonical_masked_probabilities(values, mask, label=name)[0]
            for name, values in (
                ("current", current),
                ("candidate", candidate),
                ("deployed", deployed),
            )
        }
        mixture = [
            (1.0 - alpha) * c + alpha * k
            for c, k in zip(normalized["current"], normalized["candidate"])
        ]
        deployment_tv = 0.5 * math.fsum(
            abs(left - right) for left, right in zip(mixture, normalized["deployed"])
        )
        self.assertGreater(deployment_tv, 1e-6)

        # A genuine exact mixture still lands inside the 1e-6 tolerance after
        # each law is normalized independently.
        exact = [(1.0 - alpha) * c + alpha * k for c, k in zip(current, candidate)]
        normalized_exact, _, _ = _canonical_masked_probabilities(
            exact, mask, label="deployed"
        )
        exact_tv = 0.5 * math.fsum(
            abs(left - right) for left, right in zip(mixture, normalized_exact)
        )
        self.assertLessEqual(exact_tv, 1e-6)
