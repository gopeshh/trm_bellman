#!/usr/bin/env fbpython
"""Filesystem-bound tests for policy-improvement evidence publication."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
import unittest
from pathlib import Path

import torch
from policy_improvement_checkpoint_allowlist import (
    load_data_only_checkpoint,
    UnsafeCheckpointPayloadError,
)
from policy_improvement_sealed_evidence import (
    authenticate_sealed_checkpoint_field,
    SEALED_CHECKPOINT_SCHEMA_NAME,
)
from scripts.policy_improvement_evidence import (
    _producer_runtime_role_name,
    authenticate_complete_generation,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    validate_result,
)
from scripts.policy_improvement_test_open import _authenticate_record, _publish_record


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
        self.materialize_generation()

    def test_v3_role_selection_separates_stage0_from_full_runs(self) -> None:
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v3"
        }
        self.assertEqual(
            _producer_runtime_role_name(authorization, {"tier": "smoke"}),
            "policy-improvement-training",
        )
        self.assertEqual(
            _producer_runtime_role_name(authorization, {"tier": "pilot"}),
            "policy-improvement-full",
        )
        self.assertEqual(
            _producer_runtime_role_name(
                {"schema_name": "policy_improvement_runtime_authorization_v2"},
                {"tier": "pilot"},
            ),
            "policy-improvement-training",
        )

    def materialize_generation(
        self,
        checkpoint_payload: bytes = b"checkpoint bytes",
        *,
        mutate_result: object = None,
    ) -> None:
        """Write one internally self-consistent immutable smoke generation.

        ``mutate_result`` runs after the result is built and before it is
        written, so a test can forge a package whose bytes and manifest agree
        with each other while one identity binding is wrong.
        """

        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        checkpoint.write_bytes(checkpoint_payload)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        parent_generation = (
            self.root / "runs" / self.run_id / "segments" / "env_000000016"
        )
        (parent_generation / "checkpoints").mkdir(parents=True, exist_ok=True)
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
        if mutate_result is not None:
            mutate_result(self.result)
        _write_json(self.generation / "result.json", self.result)
        (self.generation / "MANIFEST.json").unlink(missing_ok=True)
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
        self.amendment_history_sha256 = str(self.result["amendment_history_sha256"])
        self.checkpoint_sha = checkpoint_sha
        self.parent_checkpoint_sha = parent_checkpoint_sha
        self.parent_generation = parent_generation
        self.sealed_requests: list[dict[str, object]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _authenticate(
        self,
        *,
        historical_runtime_authorizations: dict[str, dict[str, object]] | None = None,
        checkpoint_validator: object | None = None,
        authenticated_test_open_sha256: str | None = None,
        amendment_history_sha256: str | None = None,
        retain_checkpoint_sha256: str | None = None,
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
            checkpoint_validator=(
                self._checkpoint_validator
                if checkpoint_validator is None
                else checkpoint_validator
            ),
            protocol=self.protocol,
            registry_row=self.row,
            project_root=self.project_root,
            dataset_root=self.dataset_root,
            runtime_authorization=self.runtime_authorization,
            amendment_history_sha256=(
                self.amendment_history_sha256
                if amendment_history_sha256 is None
                else amendment_history_sha256
            ),
            authenticated_test_open_sha256=authenticated_test_open_sha256,
            historical_runtime_authorizations=historical_runtime_authorizations,
            retain_checkpoint_sha256=retain_checkpoint_sha256,
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
        # The request names checkpoint bytes only by a write-sealed descriptor.
        # Authenticate it the way the real validator does, then read the
        # matching receipt out of the evidence tree by digest.
        sealed = authenticate_sealed_checkpoint_field(
            request["checkpoint"],
            expected_sha256=str(request["checkpoint_sha256"]),
        )
        self.sealed_requests.append(dict(sealed.as_request_field()))
        receipt = next(
            value
            for value in (
                json.loads(path.read_text(encoding="ascii"))
                for path in self.root.rglob("checkpoint_validation.json")
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
            "theory_model_identity": None,
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

    def test_complete_generation_can_transfer_only_the_authenticated_resume(
        self,
    ) -> None:
        before = len(os.listdir("/proc/self/fd"))
        observed = self._authenticate(
            retain_checkpoint_sha256=self.checkpoint_sha,
        )
        sealed = observed["sealed_checkpoint"]
        self.assertEqual(sealed.sha256, self.checkpoint_sha)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before + 1)
        sealed.close()
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "does not resolve exactly",
        ):
            self._authenticate(retain_checkpoint_sha256="f" * 64)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

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
                amendment_history_sha256=self.amendment_history_sha256,
                authenticated_test_open_sha256=None,
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


class _MaliciousReducer:
    """A pickle payload whose reducer writes a marker file when unpickled."""

    def __init__(self, marker: Path) -> None:
        self._marker = str(marker)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (_write_marker, (self._marker,))


def _write_marker(marker: str) -> str:
    Path(marker).write_bytes(b"the reducer executed")
    return marker


class SealedCheckpointEvidenceTest(CompleteGenerationTest):
    """Adversarial coverage for the pre-deserialization sealing boundary."""

    def setUp(self) -> None:
        super().setUp()
        self.marker = Path(self.temporary.name) / "reducer-marker"
        self.validator_calls: list[dict[str, object]] = []

    def _recording_validator(self, request: dict[str, object]) -> dict[str, object]:
        self.validator_calls.append(dict(request))
        return self._checkpoint_validator(request)

    def _unpickling_validator(self, request: dict[str, object]) -> dict[str, object]:
        """Stand in for a loader that deserializes without restriction."""

        self.validator_calls.append(dict(request))
        sealed = authenticate_sealed_checkpoint_field(
            request["checkpoint"],
            expected_sha256=str(request["checkpoint_sha256"]),
        )
        # Only the final generation carries a pickle in these fixtures; the
        # smoke parent is opaque bytes.
        if sealed.sha256 == self.checkpoint_sha:
            with os.fdopen(os.dup(sealed.descriptor), "rb") as handle:
                handle.seek(0)
                pickle.load(handle)
        return self._checkpoint_validator(request)

    def _data_only_validator(self, request: dict[str, object]) -> dict[str, object]:
        """Drive the real production loader over the sealed descriptor."""

        self.validator_calls.append(dict(request))
        sealed = authenticate_sealed_checkpoint_field(
            request["checkpoint"],
            expected_sha256=str(request["checkpoint_sha256"]),
        )
        if sealed.sha256 == self.checkpoint_sha:
            with os.fdopen(os.dup(sealed.descriptor), "rb") as handle:
                handle.seek(0)
                load_data_only_checkpoint(handle)
        return self._checkpoint_validator(request)

    def _unrestricted_validator(self, request: dict[str, object]) -> dict[str, object]:
        """The old policy, kept only as a control that the payload is live."""

        self.validator_calls.append(dict(request))
        sealed = authenticate_sealed_checkpoint_field(
            request["checkpoint"],
            expected_sha256=str(request["checkpoint_sha256"]),
        )
        if sealed.sha256 == self.checkpoint_sha:
            with os.fdopen(os.dup(sealed.descriptor), "rb") as handle:
                handle.seek(0)
                torch.load(handle, map_location="cpu", weights_only=False)
        return self._checkpoint_validator(request)

    def _torch_archive(self, value: object) -> bytes:
        path = Path(self.temporary.name) / "payload.pt"
        torch.save(value, path)
        payload = path.read_bytes()
        path.unlink()
        return payload

    # ---- 7. loaders receive only the sealed descriptor -------------------

    def test_validator_receives_a_sealed_descriptor_and_no_pathname(self) -> None:
        self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(len(self.validator_calls), 2)
        checkpoint_paths = {
            str(self.generation / "checkpoints" / "checkpoint.pt"),
            str(self.parent_generation / "checkpoints" / "checkpoint.pt"),
        }
        for request in self.validator_calls:
            self.assertNotIn("checkpoint_path", request)
            sealed = request["checkpoint"]
            self.assertIsInstance(sealed, dict)
            self.assertEqual(
                sealed["generation_relative_path"], "checkpoints/checkpoint.pt"
            )
            self.assertIsInstance(sealed["descriptor"], int)
            self.assertNotIsInstance(sealed["descriptor"], bool)
            for value in request.values():
                self.assertNotIn(str(value), checkpoint_paths)
        for sealed in self.sealed_requests:
            self.assertNotIn("path", sealed)

    def test_sealed_descriptors_are_closed_on_success_and_failure(self) -> None:
        before = len(os.listdir("/proc/self/fd"))
        self._authenticate()
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

        def exploding_validator(request: dict[str, object]) -> dict[str, object]:
            raise PolicyImprovementSchemaError("semantic rejection")

        for _ in range(8):
            with self.assertRaises(PolicyImprovementSchemaError):
                self._authenticate(checkpoint_validator=exploding_validator)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    # ---- 5. mutation between hashing and loading -------------------------

    def test_mutation_during_validation_cannot_change_deserialized_bytes(self) -> None:
        authentic = (self.generation / "checkpoints" / "checkpoint.pt").read_bytes()
        observed: list[bytes] = []

        def mutating_validator(request: dict[str, object]) -> dict[str, object]:
            checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
            if checkpoint.read_bytes() == authentic:
                # Swap the named file for a same-length hostile payload, both in
                # place and by relinking a fresh inode over the name.
                hostile = b"H" * len(authentic)
                checkpoint.write_bytes(hostile)
                replacement = checkpoint.with_name("swap.pt")
                replacement.write_bytes(hostile)
                replacement.replace(checkpoint)
            sealed = authenticate_sealed_checkpoint_field(
                request["checkpoint"],
                expected_sha256=str(request["checkpoint_sha256"]),
            )
            observed.append(os.pread(sealed.descriptor, 1 << 20, 0))
            return self._checkpoint_validator(request)

        # The post-validation re-inventory still rejects the tampered package,
        # but the bytes the loader saw were the authenticated ones regardless.
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=mutating_validator)
        self.assertTrue(observed)
        for payload in observed:
            self.assertNotEqual(payload, b"H" * len(authentic))
        self.assertIn(authentic, observed)

    # ---- 6. malicious pickle never runs when authentication fails --------

    def test_fully_authentic_malicious_package_executes_no_code(self) -> None:
        """The attack byte authentication alone cannot stop.

        Everything here is genuine: every digest in the package is recomputed
        from the hostile bytes, the generation is re-inventoried, and the
        result carries the runtime identities of the supplied authorization.
        Authentication has nothing to reject, so the sealed descriptor reaches
        the real loader.  Zero code executes anyway, because the loader has no
        capability to execute it.
        """

        hostile = self._torch_archive({"payload": _MaliciousReducer(self.marker)})
        self.materialize_generation(hostile)

        # The package really is authentic on every axis authentication checks.
        identities = self.result["identities"]
        self.assertEqual(
            identities["producer_manifest_sha256"],
            self.runtime_authorization["producer_source_manifest_sha256"],
        )
        self.assertEqual(
            identities["runtime_authorization_sha256"],
            self.runtime_authorization_digest,
        )
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            self.checkpoint_sha,
        )
        manifest = json.loads(
            (self.generation / "MANIFEST.json").read_text(encoding="ascii")
        )
        self.assertEqual(
            manifest["outputs"]["checkpoint"]["sha256"], self.checkpoint_sha
        )

        with self.assertRaises(UnsafeCheckpointPayloadError):
            self._authenticate(checkpoint_validator=self._data_only_validator)
        # The loader WAS reached: authentication found nothing wrong.
        self.assertTrue(self.validator_calls)
        # And still nothing ran.
        self.assertFalse(self.marker.exists())

        # Control 1: the identical authenticated bytes execute under the old
        # unrestricted policy, so the assertion above is not vacuous.
        self.validator_calls.clear()
        self._authenticate(checkpoint_validator=self._unrestricted_validator)
        self.assertTrue(self.marker.exists())

        # Control 2: an equally authentic package with a benign data-only
        # checkpoint authenticates and loads, so the rejection above is
        # attributable to the payload rather than to the package.
        self.marker.unlink()
        self.validator_calls.clear()
        benign = self._torch_archive({"payload": [1, 2, 3], "flag": True})
        self.materialize_generation(benign)
        self._authenticate(checkpoint_validator=self._data_only_validator)
        self.assertEqual(len(self.validator_calls), 2)
        self.assertFalse(self.marker.exists())

    def test_forged_identity_package_is_rejected_before_any_loader(self) -> None:
        hostile = self._torch_archive({"payload": _MaliciousReducer(self.marker)})
        self.materialize_generation(
            hostile,
            mutate_result=lambda result: result["identities"].update(
                {"producer_manifest_sha256": "f" * 64}
            ),
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._data_only_validator)
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.validator_calls, [])

    def test_tampered_checkpoint_never_reaches_a_loader(self) -> None:
        hostile = pickle.dumps(_MaliciousReducer(self.marker))
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        # Replace the bytes without re-deriving any manifest digest: the
        # classic swap of a published checkpoint for an attacker payload.
        checkpoint.write_bytes(hostile)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._unpickling_validator)
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.validator_calls, [])

    # ---- 1. external byte-identical copy ---------------------------------

    def test_byte_identical_copy_outside_the_generation_is_not_loadable(self) -> None:
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        authentic = checkpoint.read_bytes()
        outside = Path(self.temporary.name) / "outside-copy.pt"
        outside.write_bytes(authentic)
        self.assertEqual(
            hashlib.sha256(outside.read_bytes()).hexdigest(), self.checkpoint_sha
        )

        # 1. Remove the in-generation bytes. A byte-identical copy elsewhere on
        #    the filesystem must not become a fallback.
        checkpoint.unlink()
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

        # 2. Reach the copy through a symlink at the registered location.
        checkpoint.symlink_to(outside)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])
        checkpoint.unlink()

        # 3. Point the generation manifest at the copy by path traversal.
        checkpoint.write_bytes(authentic)
        self.materialize_generation()
        manifest_path = self.generation / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        identity = manifest["outputs"]["files"].pop("checkpoints/checkpoint.pt")
        manifest["outputs"]["files"]["../outside-copy.pt"] = identity
        manifest["outputs"]["checkpoint"]["path"] = "../outside-copy.pt"
        _write_json(manifest_path, manifest)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

        # Control: with the copy still present but the generation intact, the
        # sealed bytes come from inside the generation and validation succeeds.
        self.materialize_generation()
        self.assertTrue(outside.exists())
        self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(len(self.validator_calls), 2)

    def test_unregistered_sibling_copy_inside_the_generation_is_rejected(self) -> None:
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        copy = checkpoint.with_name("checkpoint.copy.pt")
        copy.write_bytes(checkpoint.read_bytes())
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

    # ---- 2/3. substitution between runs and snapshots --------------------

    def test_checkpoint_from_another_completed_run_is_rejected(self) -> None:
        other = self.root / "runs" / "s0-other-run" / "segments" / "env_000000032"
        (other / "checkpoints").mkdir(parents=True)
        other_checkpoint = other / "checkpoints" / "checkpoint.pt"
        other_checkpoint.write_bytes(b"another completed run checkpoint")
        other_sha = hashlib.sha256(other_checkpoint.read_bytes()).hexdigest()
        # Claim the other run's checkpoint without moving its bytes.
        self.result["artifacts"]["checkpoint"] = _available(other_sha)
        self.result["identities"]["checkpoint_sha256"] = _available(other_sha)
        self.result["evaluation_snapshots"][0]["checkpoint_sha256"] = _available(
            other_sha
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

        # Now also copy the bytes in, so only provenance distinguishes them.
        (self.generation / "checkpoints" / "other.pt").write_bytes(
            other_checkpoint.read_bytes()
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

    def test_parent_and_final_checkpoints_cannot_be_substituted(self) -> None:
        checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        parent_checkpoint = self.parent_generation / "checkpoints" / "checkpoint.pt"
        final_bytes = checkpoint.read_bytes()
        parent_bytes = parent_checkpoint.read_bytes()
        checkpoint.write_bytes(parent_bytes)
        parent_checkpoint.write_bytes(final_bytes)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

    # ---- 4. identity mixing fails closed ---------------------------------

    def test_identity_mixing_fails_closed_before_any_loader(self) -> None:
        mixers = {
            "progress": lambda result: result["evaluation_snapshots"][0].update(
                {"observed_environment_interactions": _available(31)}
            ),
            "model_state": lambda result: result["evaluation_snapshots"][0].update(
                {"model_state_sha256": _available("c" * 64)}
            ),
            "validation": lambda result: result["artifacts"].update(
                {"checkpoint_validation": _available("d" * 64)}
            ),
            "runtime": lambda result: result["identities"].update(
                {"training_runtime_sha256": "e" * 64}
            ),
            "lineage": lambda result: result["evaluation_snapshots"][0].update(
                {"checkpoint_lineage_sha256": _available("f" * 64)}
            ),
            "model_state_inventory": lambda result: result["artifacts"].update(
                {"model_state_inventory": _available("1" * 64)}
            ),
            "run_manifest": lambda result: result["artifacts"].update(
                {"run_manifest": _available("2" * 64)}
            ),
            "checkpoint_identity": lambda result: result["identities"].update(
                {"checkpoint_sha256": _available("3" * 64)}
            ),
            "initialization": lambda result: result["identities"].update(
                {"initialization_sha256": "4" * 64}
            ),
        }
        for name, mix in mixers.items():
            with self.subTest(identity=name):
                self.tearDown()
                self.setUp()
                self.materialize_generation(mutate_result=mix)
                with self.assertRaises(PolicyImprovementSchemaError):
                    self._authenticate(
                        checkpoint_validator=self._recording_validator,
                    )
                self.assertEqual(self.validator_calls, [], name)

    def test_amendment_history_mixing_fails_closed(self) -> None:
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                checkpoint_validator=self._recording_validator,
                amendment_history_sha256="9" * 64,
            )
        self.assertEqual(self.validator_calls, [])

    # ---- 8/9. test-split isolation ---------------------------------------

    def test_test_split_evidence_requires_an_authenticated_test_open(self) -> None:
        def open_test_split(result: dict[str, object]) -> None:
            result["evaluation_split"] = "test"
            result["identities"]["test_open_sha256"] = _available("b" * 64)

        self.materialize_generation(mutate_result=open_test_split)
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                checkpoint_validator=self._recording_validator,
                authenticated_test_open_sha256="c" * 64,
            )
        self.assertEqual(self.validator_calls, [])

        self._authenticate(
            checkpoint_validator=self._recording_validator,
            authenticated_test_open_sha256="b" * 64,
        )
        self.assertEqual(len(self.validator_calls), 2)

    def test_validation_split_evidence_cannot_carry_a_test_open(self) -> None:
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(
                checkpoint_validator=self._recording_validator,
                authenticated_test_open_sha256="b" * 64,
            )
        self.assertEqual(self.validator_calls, [])

        self.materialize_generation(
            mutate_result=lambda result: result["identities"].update(
                {"test_open_sha256": _available("b" * 64)}
            )
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

    def test_no_loader_runs_until_the_whole_generation_authenticates(self) -> None:
        """Sealing is a gate, not a step in the middle of authentication.

        The evaluation artifacts are authenticated after every checkpoint
        validation record.  Corrupting one of them must still keep every loader
        unreached, which is only true if sealing happens after the last
        authentication step rather than inside the snapshot loop.
        """

        aggregate = self.generation / "evaluations" / "realized_policy.json"
        stored = json.loads(aggregate.read_text(encoding="ascii"))
        stored["primary"]["solved_count"] = _available(0)
        self.materialize_generation()
        _write_json(aggregate, stored)
        # Keep the outer manifest consistent with the corrupted artifact so the
        # package is self-consistent about bytes and fails only on semantics.
        manifest_path = self.generation / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest_path.unlink()
        files = {
            path.relative_to(self.generation).as_posix(): _identity(path)
            for path in sorted(self.generation.rglob("*"))
            if path.is_file()
        }
        manifest["outputs"]["files"] = files
        manifest["storage_bytes"] = sum(int(item["bytes"]) for item in files.values())
        _write_json(manifest_path, manifest)

        with self.assertRaises(PolicyImprovementSchemaError):
            self._authenticate(checkpoint_validator=self._recording_validator)
        self.assertEqual(self.validator_calls, [])

    # ---- 10. valid historical evidence still validates --------------------

    def test_valid_historical_evidence_still_reaches_semantic_validation(self) -> None:
        observed = self._authenticate(checkpoint_validator=self._recording_validator)
        semantic = observed["semantic_validations"]
        self.assertIsInstance(semantic, list)
        self.assertEqual(len(semantic), 2)
        self.assertEqual(observed["checkpoint_sha256"], self.checkpoint_sha)
        self.assertEqual(
            {str(item["checkpoint_sha256"]) for item in semantic},
            {self.checkpoint_sha, self.parent_checkpoint_sha},
        )
        for sealed in self.sealed_requests:
            self.assertEqual(sealed["schema_name"], SEALED_CHECKPOINT_SCHEMA_NAME)


if __name__ == "__main__":
    unittest.main()
