#!/usr/bin/env fbpython
"""Filesystem-bound tests for policy-improvement evidence publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from scripts.policy_improvement_evidence import authenticate_complete_generation
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    canonical_json_bytes,
    runtime_authorization_sha256,
    validate_result,
)
from scripts.policy_improvement_test_open import (
    _authenticate_record,
    _publish_record,
)


HEX64 = "a" * 64


def _available(value: object) -> dict[str, object]:
    return {"status": "available", "value": value}


def _unavailable(reason: str) -> dict[str, object]:
    return {"status": "unavailable", "reason": reason}


def _write_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class CompleteGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "evidence"
        self.root.mkdir(mode=0o700)
        self.project_root = Path(self.temporary.name) / "project"
        self.project_root.mkdir()
        self.dataset_root = Path(self.temporary.name) / "dataset"
        self.dataset_root.mkdir()
        self.run_id = "s0-fixed-base-s1"
        self.protocol_sha = "1" * 64
        self.registry_sha = "2" * 64
        self.row_sha = "3" * 64
        self.protocol = {
            "methods": [
                {
                    "id": "matched_ppo",
                    "config_sha256": "4" * 64,
                }
            ],
            "dataset": {"manifest_sha256": {"status": "available", "value": "8" * 64}},
        }
        self.row = {
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "base_method_id": "matched_ppo",
            "seed": 1,
            "expected_effective_config_sha256": "5" * 64,
        }
        self.runtime_authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v1",
            "schema_version": 1,
            "authorization_id": "complete-generation-fixture-v1",
            "created_at_utc": "2026-08-14T12:00:00Z",
            "protocol_sha256": self.protocol_sha,
            "producer_git_commit": "b" * 40,
            "producer_source_manifest_sha256": "e" * 64,
            "launcher_sha256": "0" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "b" * 40,
                    "runtime_sha256": "c" * 64,
                    "runtime_profile_sha256": "d" * 64,
                    "selected_source_manifest_sha256": "d" * 64,
                }
                for role in (
                    "policy-improvement-training",
                    "policy-improvement-evaluation",
                    "policy-improvement-audit",
                    "policy-improvement-analysis",
                )
            ],
        }
        self.runtime_authorization_digest = runtime_authorization_sha256(
            self.runtime_authorization
        )
        self.training_execution_identity = {
            "role": "policy-improvement-training",
            "source_git_commit": "b" * 40,
            "runtime_sha256": "c" * 64,
            "runtime_profile_sha256": "d" * 64,
            "selected_source_manifest_sha256": "d" * 64,
            "runtime_authorization_sha256": self.runtime_authorization_digest,
            "launcher_sha256": "0" * 64,
        }
        self.generation = (
            self.root / "runs" / self.run_id / "segments" / "env_000000032"
        )
        (self.generation / "checkpoints").mkdir(parents=True)
        (self.generation / "evaluations").mkdir()
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint bytes")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        parent_generation = (
            self.root / "runs" / self.run_id / "segments" / "env_000000016"
        )
        (parent_generation / "checkpoints").mkdir(parents=True)
        parent_checkpoint = parent_generation / "checkpoints" / "checkpoint.pt"
        parent_checkpoint.write_bytes(b"parent checkpoint bytes")
        parent_checkpoint_sha = hashlib.sha256(
            parent_checkpoint.read_bytes()
        ).hexdigest()
        _write_json(
            parent_generation / "checkpoint_validation.json",
            {
                "schema_name": "policy_improvement_checkpoint_validation_v1",
                "schema_version": 1,
                "validator": "policy_improvement_smoke_runtime",
                "validator_execution_identity": dict(self.training_execution_identity),
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "snapshot_kind": "interaction_matched",
                "environment_interactions": 16,
                "checkpoint_sha256": parent_checkpoint_sha,
                "parent_checkpoint_sha256": None,
                "model_state_sha256": hashlib.sha256(
                    canonical_json_bytes({"model": "9" * 64})
                ).hexdigest(),
                "role_state_sha256s": {"model": "9" * 64},
                "strict_resume_validated": True,
            },
        )
        parent_files = {
            "checkpoints/checkpoint.pt": _identity(parent_checkpoint),
            "checkpoint_validation.json": _identity(
                parent_generation / "checkpoint_validation.json"
            ),
        }
        _write_json(
            parent_generation / "MANIFEST.json",
            {
                "schema_name": "policy_improvement_smoke_segment_v1",
                "schema_version": 1,
                "protocol_sha256": self.protocol_sha,
                "registry_sha256": self.registry_sha,
                "registry_row_sha256": self.row_sha,
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "segment": "prepare",
                "environment_interactions": 16,
                "parent_checkpoint_sha256": None,
                "result_status": None,
                "outputs": {
                    "checkpoint": {
                        "path": "checkpoints/checkpoint.pt",
                        **parent_files["checkpoints/checkpoint.pt"],
                    },
                    "files": parent_files,
                },
                "storage_bytes": sum(
                    int(item["bytes"]) for item in parent_files.values()
                ),
            },
        )
        model_roles = {"model": "7" * 64}
        model_sha = hashlib.sha256(canonical_json_bytes(model_roles)).hexdigest()
        initialization_sha = "6" * 64
        checkpoint_lineage_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_name": "policy_improvement_smoke_checkpoint_lineage_v1",
                    "run_id": self.run_id,
                    "initialization_sha256": initialization_sha,
                    "parent_checkpoint_sha256": parent_checkpoint_sha,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
        ).hexdigest()
        checkpoint_validation_digest = _write_json(
            self.generation / "checkpoint_validation.json",
            {
                "schema_name": "policy_improvement_checkpoint_validation_v1",
                "schema_version": 1,
                "validator": "policy_improvement_smoke_runtime",
                "validator_execution_identity": dict(self.training_execution_identity),
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "snapshot_kind": "interaction_matched",
                "environment_interactions": 32,
                "checkpoint_sha256": checkpoint_sha,
                "parent_checkpoint_sha256": parent_checkpoint_sha,
                "model_state_sha256": model_sha,
                "role_state_sha256s": model_roles,
                "strict_resume_validated": True,
            },
        )
        model_inventory_digest = _write_json(
            self.generation / "model_state_inventory.json",
            {
                "schema_name": "policy_improvement_model_state_inventory_v1",
                "run_id": self.run_id,
                "method_id": "matched_ppo",
                "model_state_sha256": model_sha,
                "role_state_sha256s": model_roles,
                "snapshot_state_bindings": [
                    {
                        "snapshot_kind": "interaction_matched",
                        "checkpoint_sha256": checkpoint_sha,
                        "model_state_sha256": model_sha,
                        "checkpoint_validation_sha256": (checkpoint_validation_digest),
                    }
                ],
            },
        )
        self.per_instance = {
            "schema_name": "fixture_per_instance",
            "records": [{"registered_index": 0, "solved": True}],
        }
        per_digest = hashlib.sha256(canonical_json_bytes(self.per_instance)).hexdigest()
        primary = {
            "solve_rate": _available(1.0),
            "solved_count": _available(1),
            "denominator": _available(1),
        }
        secondary = {
            "discounted_return_mean": _available(1.0),
            "edits_to_solve_mean": _available(1.0),
            "terminal_reason_counts": _available({"solved": 1}),
            "value_calibration": _available(0.0),
        }
        aggregate_semantic = {"primary": primary, "secondary": secondary}
        aggregate_digest = hashlib.sha256(
            canonical_json_bytes(aggregate_semantic)
        ).hexdigest()
        _write_json(
            self.generation / "evaluations" / "realized_policy.json",
            {**aggregate_semantic, "diagnostic_details": {}},
        )
        _write_json(
            self.generation / "evaluations" / "realized_policy.per_instance.json",
            self.per_instance,
        )
        run_manifest = {
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "environment_interactions": 32,
            "protocol_sha256": self.protocol_sha,
            "registry_row_sha256": self.row_sha,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": model_sha,
            "model_state_sha256s": model_roles,
            "model_state_inventory_sha256": model_inventory_digest,
        }
        run_manifest_digest = _write_json(
            self.generation / "RUN_MANIFEST.json", run_manifest
        )
        self.result = {
            "protocol_id": "policy-improvement-v1-20260814",
            "protocol_sha256": self.protocol_sha,
            "run_id": self.run_id,
            "registry_row_sha256": self.row_sha,
            "phase": "stage0_smoke",
            "method_id": "matched_ppo",
            "base_method_id": "matched_ppo",
            "tier": "smoke",
            "seed": 1,
            "evaluation_split": "validation",
            "n": 2,
            "K": 1,
            "alpha": 0.1,
            "ablation_variant": None,
            "primary_policy_variant": "realized_policy",
            "applied_config_override": {"method": "matched_ppo"},
            "amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
            "artifacts": {
                "checkpoint": _available(checkpoint_sha),
                "checkpoint_validation": _available(checkpoint_validation_digest),
                "model_state_inventory": _available(model_inventory_digest),
                "run_manifest": _available(run_manifest_digest),
            },
            "identities": {
                "producer_git_commit": self.runtime_authorization[
                    "producer_git_commit"
                ],
                "producer_manifest_sha256": self.runtime_authorization[
                    "producer_source_manifest_sha256"
                ],
                "checkpoint_sha256": _available(checkpoint_sha),
                "model_state_sha256": _available(model_sha),
                "initialization_sha256": initialization_sha,
                "training_source_git_commit": self.training_execution_identity[
                    "source_git_commit"
                ],
                "training_runtime_sha256": self.training_execution_identity[
                    "runtime_sha256"
                ],
                "training_runtime_profile_sha256": self.training_execution_identity[
                    "runtime_profile_sha256"
                ],
                "training_selected_source_manifest_sha256": (
                    self.training_execution_identity["selected_source_manifest_sha256"]
                ),
                "runtime_authorization_sha256": self.training_execution_identity[
                    "runtime_authorization_sha256"
                ],
                "launcher_sha256": self.training_execution_identity["launcher_sha256"],
                "git_clean": True,
                "method_config_sha256": "4" * 64,
                "effective_config_sha256": "5" * 64,
                "dataset_manifest_sha256": "8" * 64,
                "train_ordered_records_sha256": "1" * 64,
                "evaluation_ordered_records_sha256": "2" * 64,
                "test_open_sha256": _unavailable("test_data_not_opened"),
                "device": "cpu",
            },
            "evaluation_snapshots": [
                {
                    "snapshot_kind": "interaction_matched",
                    "status": "available",
                    "observed_environment_interactions": _available(32),
                    "checkpoint_sha256": _available(checkpoint_sha),
                    "model_state_sha256": _available(model_sha),
                    "checkpoint_lineage_sha256": _available(checkpoint_lineage_sha),
                    "policy_evaluations": [
                        {
                            "policy_variant": "realized_policy",
                            "evaluation_artifact_sha256": _available(aggregate_digest),
                            "per_instance_artifact_sha256": _available(per_digest),
                            "primary": primary,
                            "secondary": secondary,
                        }
                    ],
                }
            ],
        }
        _write_json(self.generation / "result.json", self.result)
        files: dict[str, dict[str, object]] = {}
        for path in sorted(self.generation.rglob("*")):
            if path.is_file():
                files[path.relative_to(self.generation).as_posix()] = _identity(path)
        manifest = {
            "schema_name": "policy_improvement_smoke_segment_v1",
            "schema_version": 1,
            "protocol_sha256": self.protocol_sha,
            "registry_sha256": self.registry_sha,
            "registry_row_sha256": self.row_sha,
            "run_id": self.run_id,
            "method_id": "matched_ppo",
            "segment": "resume",
            "environment_interactions": 32,
            "parent_checkpoint_sha256": parent_checkpoint_sha,
            "result_status": "complete",
            "outputs": {
                "checkpoint": {
                    "path": "checkpoints/checkpoint.pt",
                    **files["checkpoints/checkpoint.pt"],
                },
                "files": files,
            },
            "storage_bytes": sum(int(item["bytes"]) for item in files.values()),
        }
        _write_json(self.generation / "MANIFEST.json", manifest)
        self.documents = {per_digest: self.per_instance}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _authenticate(
        self,
        *,
        historical_runtime_authorizations: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return authenticate_complete_generation(
            evidence_root=self.root,
            result_path=self.generation / "result.json",
            result=self.result,
            protocol_sha256=self.protocol_sha,
            registry_sha256=self.registry_sha,
            registry_row_sha256=self.row_sha,
            expected_environment_interactions=32,
            per_instance_documents=self.documents,
            checkpoint_validator=self._checkpoint_validator,
            protocol=self.protocol,
            registry_row=self.row,
            project_root=self.project_root,
            dataset_root=self.dataset_root,
            runtime_authorization=self.runtime_authorization,
            historical_runtime_authorizations=historical_runtime_authorizations,
        )

    def _historical_runtime_authorization(self) -> tuple[dict[str, object], str]:
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v1",
            "schema_version": 1,
            "authorization_id": "historical-failed-attempt-v1",
            "created_at_utc": "2026-08-14T11:00:00Z",
            "protocol_sha256": self.protocol_sha,
            "producer_git_commit": "c" * 40,
            "producer_source_manifest_sha256": "7" * 64,
            "launcher_sha256": "8" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "c" * 40,
                    "runtime_sha256": "9" * 64,
                    "runtime_profile_sha256": "a" * 64,
                    "selected_source_manifest_sha256": "a" * 64,
                }
                for role in (
                    "policy-improvement-training",
                    "policy-improvement-evaluation",
                    "policy-improvement-audit",
                    "policy-improvement-analysis",
                )
            ],
        }
        return authorization, runtime_authorization_sha256(authorization)

    def _materialize_historical_failed_attempt(
        self,
        *,
        bind_to_complete_generation: bool = True,
    ) -> tuple[Path, dict[str, object], str]:
        authorization, authorization_digest = self._historical_runtime_authorization()
        reason = "run_failed_before_checkpoint"
        measurement_reason = "run_failed_before_measurement"
        snapshots = []
        for kind, unit in (
            ("interaction_matched", "environment_interactions"),
            ("compute_matched", "recurrent_map_applications"),
        ):
            snapshots.append(
                {
                    "snapshot_kind": kind,
                    "status": "unavailable",
                    "unavailable_reason": reason,
                    "target": {
                        "unit": unit,
                        "registered_quantity": _unavailable(reason),
                    },
                    "observed_environment_interactions": _unavailable(reason),
                    "observed_recurrent_map_applications": _unavailable(reason),
                    "accelerator_seconds_observed": _unavailable(reason),
                    "checkpoint_sha256": _unavailable(reason),
                    "model_state_sha256": _unavailable(reason),
                    "checkpoint_lineage_sha256": _unavailable(reason),
                    "policy_evaluations": [],
                }
            )
        training_fields = (
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
        diagnostic_fields = (
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
        failed_result = validate_result(
            {
                "schema_name": "policy_improvement_v1",
                "schema_version": 3,
                "protocol_id": self.result["protocol_id"],
                "protocol_sha256": self.protocol_sha,
                "amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
                "run_id": self.run_id,
                "registry_row_sha256": self.row_sha,
                "phase": "stage0_smoke",
                "tier": "smoke",
                "seed": 1,
                "evaluation_split": "validation",
                "method_id": "matched_ppo",
                "base_method_id": "matched_ppo",
                "n": 2,
                "K": 1,
                "alpha": 0.1,
                "ablation_variant": None,
                "applied_config_override": {"method": "matched_ppo"},
                "primary_policy_variant": "realized_policy",
                "evaluation_snapshots": snapshots,
                "status": "failed",
                "failure": {
                    "phase": "training",
                    "error_class": "runtime_error",
                    "message_sha256": "6" * 64,
                },
                "identities": {
                    "producer_git_commit": authorization["producer_git_commit"],
                    "git_clean": True,
                    "runtime_authorization_sha256": authorization_digest,
                    "training_source_git_commit": authorization["roles"][0][
                        "source_git_commit"
                    ],
                    "training_runtime_sha256": authorization["roles"][0][
                        "runtime_sha256"
                    ],
                    "training_runtime_profile_sha256": authorization["roles"][0][
                        "runtime_profile_sha256"
                    ],
                    "training_selected_source_manifest_sha256": authorization["roles"][
                        0
                    ]["selected_source_manifest_sha256"],
                    "launcher_sha256": authorization["launcher_sha256"],
                    "producer_manifest_sha256": authorization[
                        "producer_source_manifest_sha256"
                    ],
                    "method_config_sha256": "4" * 64,
                    "effective_config_sha256": "5" * 64,
                    "dataset_manifest_sha256": "8" * 64,
                    "train_ordered_records_sha256": "1" * 64,
                    "evaluation_ordered_records_sha256": "2" * 64,
                    "initialization_sha256": self.result["identities"][
                        "initialization_sha256"
                    ],
                    "checkpoint_sha256": _unavailable(reason),
                    "model_state_sha256": _unavailable(reason),
                    "evaluation_runtime_sha256": _unavailable(
                        "run_failed_before_evaluation"
                    ),
                    "evaluation_source_git_commit": _unavailable(
                        "run_failed_before_evaluation"
                    ),
                    "evaluation_runtime_profile_sha256": _unavailable(
                        "run_failed_before_evaluation"
                    ),
                    "evaluation_selected_source_manifest_sha256": _unavailable(
                        "run_failed_before_evaluation"
                    ),
                    "evaluation_pool_sha256": _unavailable(
                        "run_failed_before_evaluation"
                    ),
                    "test_open_sha256": _unavailable("test_data_not_opened"),
                    "device": "cpu",
                },
                "metrics": {
                    "training": {
                        name: _unavailable(measurement_reason)
                        for name in training_fields
                    },
                    "diagnostics": {
                        name: _unavailable(measurement_reason)
                        for name in diagnostic_fields
                    },
                },
                "artifacts": {
                    "checkpoint": _unavailable(reason),
                    "checkpoint_validation": _unavailable(reason),
                    "model_state_inventory": _unavailable(reason),
                    "run_manifest": _unavailable(reason),
                },
            }
        )
        attempt_id = "0" * 32
        attempt = self.root / "runs" / self.run_id / "attempts" / "prepare" / attempt_id
        result_digest = _write_json(attempt / "result.json", failed_result)
        result_path = attempt / "result.json"
        _write_json(
            attempt / "MANIFEST.json",
            {
                "schema_name": "policy_improvement_smoke_failed_attempt_v1",
                "schema_version": 1,
                "attempt_id": attempt_id,
                "protocol_sha256": self.protocol_sha,
                "registry_row_sha256": self.row_sha,
                "runtime_authorization_sha256": authorization_digest,
                "run_id": self.run_id,
                "segment": "prepare",
                "failure_phase": "training",
                "result": {
                    "path": "result.json",
                    "bytes": result_path.stat().st_size,
                    "sha256": result_digest,
                },
            },
        )
        if bind_to_complete_generation:
            complete_manifest_path = self.generation / "MANIFEST.json"
            complete_manifest = json.loads(
                complete_manifest_path.read_text(encoding="ascii")
            )
            complete_manifest["schema_version"] = 2
            complete_manifest["prior_failed_attempts"] = [
                {
                    "segment": "prepare",
                    "attempt_id": attempt_id,
                    "generation_manifest_sha256": _identity(attempt / "MANIFEST.json")[
                        "sha256"
                    ],
                    "result_sha256": result_digest,
                }
            ]
            _write_json(complete_manifest_path, complete_manifest)
        return attempt, authorization, authorization_digest

    def _checkpoint_validator(self, request: dict[str, object]) -> dict[str, object]:
        checkpoint = Path(str(request["checkpoint_path"]))
        receipt = next(
            value
            for value in (
                json.loads(path.read_text(encoding="ascii"))
                for path in checkpoint.parents[1].rglob("checkpoint_validation.json")
            )
            if value["checkpoint_sha256"] == request["checkpoint_sha256"]
        )
        return {
            "schema_name": "policy_improvement_checkpoint_semantic_validation_v1",
            "schema_version": 1,
            "run_id": request["run_id"],
            "method_id": request["method_id"],
            "seed": request["seed"],
            "snapshot_kind": request["snapshot_kind"],
            "environment_interactions": request["environment_interactions"],
            "parent_checkpoint_sha256": request["parent_checkpoint_sha256"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "initialization_sha256": request["initialization_sha256"],
            "model_state_sha256": receipt["model_state_sha256"],
            "role_state_sha256s": receipt["role_state_sha256s"],
            "method_config_sha256": "4" * 64,
            "registered_effective_config_sha256": "5" * 64,
            "effective_config_sha256": "a" * 64,
            "dataset_manifest_sha256": "8" * 64,
            "dataset_provenance_sha256": "9" * 64,
            "run_identity_sha256": None,
            "training_call_delta": 0,
            "evaluation_call_delta": 0,
            "optimizer_step_delta": 0,
        }

    def test_complete_generation_reopens_every_artifact(self) -> None:
        observed = self._authenticate()
        self.assertEqual(
            observed["checkpoint_sha256"],
            self.result["artifacts"]["checkpoint"]["value"],
        )
        self.assertEqual(observed["historical_failed_attempts"], [])

    def test_complete_generation_authenticates_retained_failed_attempt(self) -> None:
        attempt, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt()
        )
        observed = self._authenticate(
            historical_runtime_authorizations={
                authorization_digest: authorization,
            }
        )
        self.assertEqual(
            observed["historical_failed_attempts"],
            [
                {
                    "attempt_id": "0" * 32,
                    "segment": "prepare",
                    "failure_phase": "training",
                    "runtime_authorization_sha256": authorization_digest,
                    "generation_manifest_sha256": _identity(attempt / "MANIFEST.json")[
                        "sha256"
                    ],
                    "result_sha256": _identity(attempt / "result.json")["sha256"],
                }
            ],
        )

    def test_retained_failed_attempt_requires_matching_authorization(self) -> None:
        _, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt()
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()
        wrong_authorization = json.loads(json.dumps(authorization))
        wrong_authorization["authorization_id"] = "wrong-historical-authorization-v1"
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                historical_runtime_authorizations={
                    authorization_digest: wrong_authorization,
                }
            )

    def test_posthoc_uncommitted_attempt_is_rejected(self) -> None:
        _, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt(
                bind_to_complete_generation=False
            )
        )
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "does not bind its exact prior failed-attempt set",
        ):
            self._authenticate(
                historical_runtime_authorizations={
                    authorization_digest: authorization,
                }
            )

    def test_tampered_historical_result_is_rejected(self) -> None:
        attempt, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt()
        )
        result = json.loads((attempt / "result.json").read_text(encoding="ascii"))
        result["failure"]["message_sha256"] = "7" * 64
        _write_json(attempt / "result.json", result)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                historical_runtime_authorizations={
                    authorization_digest: authorization,
                }
            )

    def test_historical_registered_identity_drift_is_rejected(self) -> None:
        attempt, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt()
        )
        result = json.loads((attempt / "result.json").read_text(encoding="ascii"))
        result["identities"]["dataset_manifest_sha256"] = "f" * 64
        result_digest = _write_json(attempt / "result.json", result)
        manifest_path = attempt / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest["result"]["sha256"] = result_digest
        manifest["result"]["bytes"] = (attempt / "result.json").stat().st_size
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "dataset_manifest_sha256 differs",
        ):
            self._authenticate(
                historical_runtime_authorizations={
                    authorization_digest: authorization,
                }
            )

    def test_tampered_historical_manifest_is_rejected(self) -> None:
        attempt, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt()
        )
        manifest = json.loads((attempt / "MANIFEST.json").read_text(encoding="ascii"))
        manifest["failure_phase"] = "publication"
        _write_json(attempt / "MANIFEST.json", manifest)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                historical_runtime_authorizations={
                    authorization_digest: authorization,
                }
            )

    def test_extra_historical_attempt_file_is_rejected(self) -> None:
        attempt, authorization, authorization_digest = (
            self._materialize_historical_failed_attempt()
        )
        (attempt / "unexpected.txt").write_bytes(b"unexpected")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                historical_runtime_authorizations={
                    authorization_digest: authorization,
                }
            )

    def test_detached_result_is_rejected(self) -> None:
        detached = self.root / "detached.json"
        _write_json(detached, self.result)
        with self.assertRaises(PolicyImprovementSchemaError):
            authenticate_complete_generation(
                evidence_root=self.root,
                result_path=detached,
                result=self.result,
                protocol_sha256=self.protocol_sha,
                registry_sha256=self.registry_sha,
                registry_row_sha256=self.row_sha,
                expected_environment_interactions=32,
                per_instance_documents=self.documents,
                checkpoint_validator=self._checkpoint_validator,
                protocol=self.protocol,
                registry_row=self.row,
                project_root=self.project_root,
                dataset_root=self.dataset_root,
                runtime_authorization=self.runtime_authorization,
            )

    def test_tampered_checkpoint_is_rejected(self) -> None:
        (self.generation / "checkpoints" / "checkpoint.pt").write_bytes(b"tampered")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()

    def test_tampered_parent_checkpoint_is_rejected(self) -> None:
        parent = (
            self.root
            / "runs"
            / self.run_id
            / "segments"
            / "env_000000016"
            / "checkpoints"
            / "checkpoint.pt"
        )
        parent.write_bytes(b"tampered parent")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()

    def test_extra_file_is_rejected(self) -> None:
        (self.generation / "extra").write_bytes(b"extra")
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()

    def test_symlink_and_hardlink_are_rejected(self) -> None:
        target = self.generation / "RUN_MANIFEST.json"
        alias = self.generation / "alias"
        os.link(target, alias)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()
        alias.unlink()
        alias.symlink_to(target)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate()


class TestOpenPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "test-open"
        self.root.mkdir(mode=0o700)
        self.record = {
            "schema_name": "policy_improvement_test_open_v1",
            "schema_version": 3,
            "record_id": "confirmatory-test-open-v1",
            "state": "opened_immutable",
            "protocol_sha256": "1" * 64,
            "registry_sha256": "2" * 64,
            "amendment_history_sha256": "3" * 64,
            "runtime_authorization_sha256": "4" * 64,
            "test_manifest_sha256": "5" * 64,
            "prior_open_record_sha256": "6" * 64,
            "open_ordinal": 1,
            "authorized_phases": ["stage2_confirmatory", "stage3_ablation"],
            "final_selection_created_at_utc": "2026-08-14T12:00:00Z",
            "opened_at_utc": "2026-08-14T12:01:00Z",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_publish_once_and_authenticate(self) -> None:
        identity = _publish_record(owner_root=self.root, record=self.record)
        self.assertEqual(
            _authenticate_record(
                owner_root=self.root,
                expected_record=self.record,
                expected_sha256=identity["sha256"],
            ),
            identity,
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            _publish_record(owner_root=self.root, record=self.record)

    def test_concurrent_publish_has_one_winner(self) -> None:
        outcomes: list[str] = []

        def publish() -> None:
            try:
                _publish_record(owner_root=self.root, record=self.record)
                outcomes.append("published")
            except PolicyImprovementSchemaError:
                outcomes.append("rejected")

        threads = [threading.Thread(target=publish) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(outcomes), ["published", "rejected"])
        self.assertEqual(
            [path.name for path in self.root.iterdir()], ["TEST_OPEN.json"]
        )

    def test_tamper_and_symlink_are_rejected(self) -> None:
        identity = _publish_record(owner_root=self.root, record=self.record)
        path = self.root / "TEST_OPEN.json"
        path.write_bytes(b"tampered")
        with self.assertRaises(PolicyImprovementSchemaError):
            _authenticate_record(
                owner_root=self.root,
                expected_record=self.record,
                expected_sha256=identity["sha256"],
            )
        path.unlink()
        outside = Path(self.temporary.name) / "outside"
        outside.write_bytes(canonical_json_bytes(self.record) + b"\n")
        path.symlink_to(outside)
        with self.assertRaises(PolicyImprovementSchemaError):
            _authenticate_record(
                owner_root=self.root,
                expected_record=self.record,
                expected_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
