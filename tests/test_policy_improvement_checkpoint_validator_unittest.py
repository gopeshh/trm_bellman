#!/usr/bin/env fbpython
"""Focused boundary tests for semantic Stage-0 checkpoint validation."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import policy_improvement_checkpoint_validator as validator
import policy_improvement_smoke_runtime as smoke
from scripts.policy_improvement_registry import generate_registry
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json,
    validate_protocol,
    validate_runtime_authorization,
)
from utils.run_identity import canonical_json_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs/policy_improvement_v1/protocol.json"


def _context(
    root: Path,
    *,
    method_id: str = "fixed_base_exact_persistent",
) -> smoke.SmokeContext:
    protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
    registry = generate_registry(protocol)
    row = next(
        item
        for item in registry["rows"]
        if item["phase"] == "stage0_smoke" and item["method_id"] == method_id
    )
    return smoke.SmokeContext(
        protocol=protocol,
        protocol_sha256=hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        registry=registry,
        registry_sha256=hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
        row=row,
        registry_row_sha256=hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
        source_root=root / "project",
        dataset_root=root / "dataset",
        dataset_manifest_sha256="1" * 64,
        dataset_producer_source={
            "git_commit": "a" * 40,
            "launcher_sha256": "2" * 64,
            "runtime_sha256": "3" * 64,
            "source_manifest_sha256": "4" * 64,
        },
        evidence_root=root / "evidence",
        run_root=root / "evidence" / "runs" / str(row["run_id"]),
        segment_budget=32,
        segment_name="resume",
        runtime_sha256="5" * 64,
        runtime_authorization_sha256="6" * 64,
        runtime_profile_sha256="7" * 64,
        selected_source_manifest_sha256="7" * 64,
        launcher_sha256="8" * 64,
        producer_commit="a" * 40,
        producer_manifest_sha256="7" * 64,
    )


def _authorization(protocol_sha256: str) -> dict[str, object]:
    return validate_runtime_authorization(
        {
            "schema_name": "policy_improvement_runtime_authorization_v1",
            "schema_version": 1,
            "authorization_id": "semantic-validator-test-v1",
            "created_at_utc": "2026-08-14T12:00:00Z",
            "protocol_sha256": protocol_sha256,
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": "1" * 64,
            "launcher_sha256": "2" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "a" * 40,
                    "runtime_sha256": "3" * 64,
                    "runtime_profile_sha256": "1" * 64,
                    "selected_source_manifest_sha256": "1" * 64,
                }
                for role in (
                    "policy-improvement-training",
                    "policy-improvement-evaluation",
                    "policy-improvement-audit",
                    "policy-improvement-analysis",
                )
            ],
        }
    )


def _request(
    root: Path,
    checkpoint: Path,
    *,
    environment_interactions: int = 16,
) -> dict[str, object]:
    protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
    registry = generate_registry(protocol)
    row = next(
        item
        for item in registry["rows"]
        if item["phase"] == "stage0_smoke" and item["method_id"] == "matched_ppo"
    )
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "schema_name": "policy_improvement_checkpoint_validation_request_v1",
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "registry_row": row,
        "registry_row_sha256": hashlib.sha256(canonical_json_bytes(row)).hexdigest(),
        "project_root": str(root / "project"),
        "dataset_root": str(root / "project" / str(protocol["dataset"]["root"])),
        "evidence_root": str(root / "evidence"),
        "runtime_authorization": _authorization(protocol_sha256),
        "run_id": row["run_id"],
        "method_id": row["method_id"],
        "seed": row["seed"],
        "snapshot_kind": "interaction_matched",
        "environment_interactions": environment_interactions,
        "parent_checkpoint_sha256": None,
        "initialization_sha256": "4" * 64,
    }


class _UPITrainer:
    def __init__(self) -> None:
        self.policy_model_old = torch.nn.Linear(2, 2)
        self.policy_model_candidate = torch.nn.Linear(2, 2)
        self.target_model = torch.nn.Linear(2, 2)
        self.preinterpolation_policy_base = None
        self.preinterpolation_policy_candidate = None

    def train_step(self) -> None:
        raise AssertionError("training must not run")


class PolicyImprovementCheckpointValidatorTest(unittest.TestCase):
    def test_request_rejects_non_smoke_interaction_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = validate_protocol(load_strict_json(PROTOCOL_PATH))
            (root / "project" / str(protocol["dataset"]["root"])).mkdir(parents=True)
            generation = (
                root
                / "evidence"
                / "runs"
                / "placeholder"
                / "segments"
                / "env_000000016"
                / "checkpoints"
            )
            generation.mkdir(parents=True)
            checkpoint = generation / "checkpoint.pt"
            checkpoint.write_bytes(b"not used")
            request = _request(root, checkpoint)
            run_id = str(request["run_id"])
            correct_parent = (
                root
                / "evidence"
                / "runs"
                / run_id
                / "segments"
                / "env_000000016"
                / "checkpoints"
            )
            correct_parent.mkdir(parents=True)
            correct_checkpoint = correct_parent / "checkpoint.pt"
            checkpoint.replace(correct_checkpoint)
            request["checkpoint_path"] = str(correct_checkpoint)
            request["checkpoint_sha256"] = hashlib.sha256(
                correct_checkpoint.read_bytes()
            ).hexdigest()

            validator._validate_request(request)
            request["environment_interactions"] = 17
            with self.assertRaisesRegex(
                validator.PolicyImprovementCheckpointValidationError,
                "Stage-0 snapshot",
            ):
                validator._validate_request(request)

    def test_ppo_validation_reopens_bytes_and_derives_native_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = torch.nn.Linear(2, 2)
            payload = {
                "progress": {"environment_interactions": 32},
                "parent_checkpoint_sha256": "9" * 64,
                "model_state_dict": model.state_dict(),
            }
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            request = {
                "checkpoint_sha256": digest,
            }
            session = SimpleNamespace(
                trainer=object(),
                effective_config={"registered": True},
            )
            with mock.patch.object(
                validator,
                "validate_ppo_smoke_checkpoint",
            ) as strict_validate:
                result = validator._validate_ppo(request, path, session)

            self.assertEqual(result[0], digest)
            self.assertEqual(result[1], "9" * 64)
            self.assertEqual(result[2], 32)
            self.assertEqual(result[3], canonical_json_sha256(result[4]))
            strict_validate.assert_called_once_with(
                mock.ANY,
                session.trainer,
                expected_identity=session.effective_config,
                validate_only=True,
            )

    def test_upi_validation_strictly_restores_every_model_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root)
            parent = "a" * 64
            model = torch.nn.Linear(2, 2)
            trainer = _UPITrainer()
            session = SimpleNamespace(
                model=model,
                trainer=trainer,
                device=torch.device("cpu"),
                dataset_provenance={"dataset": "registered"},
                run_identity={"run": "registered"},
                evidence_identity={"evidence": "registered"},
            )
            raw = {
                "policy_improvement_smoke_identity": {
                    "schema_version": 1,
                    "run_id": context.row["run_id"],
                    "method_id": context.row["method_id"],
                    "seed": context.row["seed"],
                    "environment_interactions": 32,
                    "parent_checkpoint_sha256": parent,
                },
                "evidence_identity": session.evidence_identity,
                "training_invocation": {
                    "training_seed": context.row["seed"],
                    "run_id": context.row["run_id"],
                },
                "progress": {"env_steps": 32},
                "checkpoint_lineage": {"parent_checkpoint_sha256": parent},
                "model_state_dict": model.state_dict(),
                "policy_model_old_state_dict": trainer.policy_model_old.state_dict(),
                "policy_model_candidate_state_dict": (
                    trainer.policy_model_candidate.state_dict()
                ),
                "target_model_state_dict": trainer.target_model.state_dict(),
            }
            checkpoint = root / "checkpoint.pt"
            torch.save(raw, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            class TrainingModule:
                audit_calls = 0

                @staticmethod
                def _load_checkpoint_payload(
                    checkpoint_path: str, *, expected_sha256: str
                ) -> tuple[object, str]:
                    observed = hashlib.sha256(
                        Path(checkpoint_path).read_bytes()
                    ).hexdigest()
                    if observed != expected_sha256:
                        raise RuntimeError("digest mismatch")
                    return (
                        torch.load(
                            checkpoint_path,
                            map_location="cpu",
                            weights_only=False,
                        ),
                        observed,
                    )

                @classmethod
                def validate_checkpoint_state_for_audit(
                    cls,
                    checkpoint_path: str,
                    active_model: torch.nn.Module,
                    active_trainer: _UPITrainer,
                    device: str,
                    **kwargs: object,
                ) -> int:
                    del checkpoint_path, device
                    cls.audit_calls += 1
                    self.assertEqual(
                        kwargs["authorized_originating_runtime_sha256"],
                        context.runtime_sha256,
                    )
                    active_model.load_state_dict(raw["model_state_dict"], strict=True)
                    active_trainer.policy_model_old.load_state_dict(
                        raw["policy_model_old_state_dict"], strict=True
                    )
                    active_trainer.policy_model_candidate.load_state_dict(
                        raw["policy_model_candidate_state_dict"], strict=True
                    )
                    active_trainer.target_model.load_state_dict(
                        raw["target_model_state_dict"], strict=True
                    )
                    return 0

            request = {
                "checkpoint_sha256": digest,
                "environment_interactions": 32,
                "parent_checkpoint_sha256": parent,
                "seed": context.row["seed"],
                "run_id": context.row["run_id"],
            }
            result = validator._validate_upi(
                request,
                checkpoint,
                context,
                session,
                TrainingModule,
            )
            self.assertEqual(TrainingModule.audit_calls, 1)
            self.assertEqual(result[0], digest)
            self.assertEqual(result[1], parent)
            self.assertEqual(result[2], 32)
            self.assertEqual(result[3], canonical_json_sha256(result[4]))

            request["parent_checkpoint_sha256"] = "b" * 64
            with self.assertRaises(smoke.PolicyImprovementSmokeError):
                validator._validate_upi(
                    request,
                    checkpoint,
                    context,
                    session,
                    TrainingModule,
                )

    def test_top_level_returns_zero_execution_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"authenticated by patched semantic loader")
            context = _context(root, method_id="matched_ppo")
            row = context.row
            request = {field: None for field in validator._REQUEST_FIELDS}
            request.update(
                {
                    "run_id": row["run_id"],
                    "method_id": row["method_id"],
                    "seed": row["seed"],
                    "snapshot_kind": "interaction_matched",
                    "environment_interactions": 16,
                    "parent_checkpoint_sha256": None,
                    "checkpoint_sha256": "1" * 64,
                    "initialization_sha256": "2" * 64,
                }
            )
            trainer = SimpleNamespace(train_step=lambda: None)
            method_registration = next(
                method
                for method in context.protocol["methods"]
                if method["id"] == row["method_id"]
            )
            session = SimpleNamespace(
                trainer=trainer,
                initialization_sha256="2" * 64,
                method_config_sha256=method_registration["config_sha256"],
                effective_config_sha256="3" * 64,
                dataset_provenance={"registered": True},
                run_identity=None,
                evaluation_started=False,
            )
            semantic = (
                "1" * 64,
                None,
                16,
                canonical_json_sha256({"model": "4" * 64}),
                {"model": "4" * 64},
            )
            with (
                mock.patch.object(
                    validator,
                    "_validate_request",
                    return_value=(
                        context.protocol,
                        row,
                        {},
                        root,
                        root,
                        root,
                        checkpoint,
                    ),
                ),
                mock.patch.object(validator, "_build_context", return_value=context),
                mock.patch.object(
                    validator,
                    "build_stage0_validation_session",
                    return_value=session,
                ),
                mock.patch.object(validator, "_validate_ppo", return_value=semantic),
                mock.patch.object(
                    validator.importlib,
                    "import_module",
                    return_value=object(),
                ),
            ):
                result = validator.validate_checkpoint(request)

            self.assertEqual(result["training_call_delta"], 0)
            self.assertEqual(result["evaluation_call_delta"], 0)
            self.assertEqual(result["optimizer_step_delta"], 0)
            self.assertEqual(result["initialization_sha256"], "2" * 64)

    def test_private_checkpoint_rewrite_binds_legacy_parent_in_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save({"checkpoint_schema_version": 4}, checkpoint)
            context = _context(root, method_id="legacy_parameter_interpolation")
            identity = smoke._policy_improvement_smoke_identity(
                context,
                environment_interactions=32,
                parent_checkpoint_sha256="f" * 64,
            )
            smoke._rewrite_checkpoint_with_smoke_identity(
                checkpoint,
                {"checkpoint_schema_version": 4},
                identity,
            )
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["policy_improvement_smoke_identity"], identity)
            smoke.validate_policy_improvement_smoke_identity(
                loaded["policy_improvement_smoke_identity"],
                context,
                environment_interactions=32,
                parent_checkpoint_sha256="f" * 64,
            )


if __name__ == "__main__":
    unittest.main()
